// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(clippy::unusual_byte_groupings)]

//! Compute work queue.
//!
//! A compute queue consists of one underlying WorkQueue.
//! This module is in charge of creating all of the firmware structures required to submit compute
//! work to the GPU, based on the userspace command buffer.

use super::common;
use crate::alloc::Allocator;
use crate::debug::*;
use crate::fw::types::*;
use crate::gpu::GpuManager;
use crate::{
    file,
    fw,
    gpu,
    microseq, //
};
use crate::{
    inner_ptr,
    inner_weak_ptr, //
};
use core::sync::atomic::Ordering;

/// Compute UMA pool id 4 is shared by every compute queue and re-installed per
/// submission: keep the previous list alive for the firmware's write-back (667zzj).
static CPOOL_GRAVE: crate::buffer::UmaGrave<fw::compute::ComputePool> = crate::buffer::UmaGrave::new();
/// .667zzz: 0 = no device compute pool, 1 = being installed, 2 = installed.
static CPOOL_STATE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
use kernel::dma_fence::RawDmaFence;
use kernel::drm::sched::Job;
use kernel::prelude::*;
use kernel::sync::Arc;
use kernel::uapi;
use kernel::xarray;

const DEBUG_CLASS: DebugFlags = DebugFlags::Compute;

#[versions(AGX)]
impl super::QueueInner::ver {
    /// .667zzz: the device's compute UMA pool 4 (backing, free-page ring,
    /// extents, record), all in klow so every VM and context 63 see it.
    fn build_cpool(
        kalloc: &mut crate::gpu::KernelAllocators,
        mib: usize,
    ) -> Result<Arc<fw::compute::ComputePool>> {
        let backing = kalloc.klow.array_empty_tagged::<u8>(mib << 20, b"CPLB")?;
        let pages = backing.as_slice().len() >> 12;
        if pages >= fw::compute::CPOOL_RING_CAP {
            return Err(EINVAL);
        }
        let mut list = kalloc.klow.array_empty_tagged::<u8>(fw::compute::CPOOL_RING_CAP * 8, b"CPLL")?;
        // .667zzz42: the firmware adds every re-push's pages to the walk count (pool[0x44]); after a
        // shared render pool grow the compute pool's walk ran 0x20000 bytes past a UMA_EXTENTS_SIZE
        // table (gUPM read fault, r10). 8 MiB = room for ~127 re-pushes of 8192 pages.
        let mut exts = kalloc.klow.array_empty_tagged::<u8>(core::cmp::max(crate::buffer::UMA_EXTENTS_SIZE, 8 << 20), b"CPLE")?;
        let mut rec = kalloc.gpu.array_empty_tagged::<u8>(0x1000, b"CPLR")?;
        let bva = u64::from(backing.gpu_pointer());
        let lva = u64::from(list.gpu_pointer());
        let eva = u64::from(exts.gpu_pointer());
        let rva = u64::from(rec.gpu_pointer());
        {
            let l = list.as_mut_slice();
            for i in 0..pages {
                l[i * 8..i * 8 + 8].copy_from_slice(&(bva + ((i as u64) << 12)).to_le_bytes());
            }
        }
        const PPB: usize = 1024;
        let nblk = pages.div_ceil(PPB);
        {
            let e = exts.as_mut_slice();
            let mut idx = 0usize;
            for b in 0..nblk {
                idx = (idx + 7) & !7;
                let va = bva + ((b * PPB) as u64) * 0x1000;
                let n = core::cmp::min(PPB, pages - b * PPB) as u64;
                let v = ((va >> 12) << 12) | ((n & 0xfff) << 52);
                e[idx * 8..idx * 8 + 8].copy_from_slice(&v.to_le_bytes());
                idx += 1;
            }
        }
        {
            let p = rec.as_mut_slice();
            crate::uma_pool_init(&mut p[0..0x90], lva, (pages as u64) << 12, rva + 0x200);
            p[0x00..0x04].copy_from_slice(&0xbu32.to_le_bytes());
            p[0x08..0x0c].copy_from_slice(&4u32.to_le_bytes());
            p[0x0c..0x10].copy_from_slice(&1u32.to_le_bytes());
            p[0x10..0x14].copy_from_slice(&1u32.to_le_bytes());
            p[0x14..0x1c].copy_from_slice(&lva.to_le_bytes());
            p[0x1c..0x20].copy_from_slice(&(fw::compute::CPOOL_RING_CAP as u32).to_le_bytes());
            p[0x24..0x28].copy_from_slice(&(pages as u32).to_le_bytes());
            p[0x28..0x2c].copy_from_slice(&0u32.to_le_bytes());
            p[0x2c..0x30].copy_from_slice(&(pages as u32).to_le_bytes());
            p[0x34..0x3c].copy_from_slice(&eva.to_le_bytes());
            p[0x44..0x48].copy_from_slice(&((nblk * 8) as u32).to_le_bytes());
        }
        pr_info!(
            "asahi/cpool: device pool 4 built: record {:#x} list {:#x} backing {:#x} pages {} extents {:#x}\n",
            rva, lva, bva, pages, eva
        );
        Ok(Arc::new(
            fw::compute::ComputePool { backing, list, exts, rec, pages, nblk },
            GFP_KERNEL,
        )?)
    }

    /// Submit work to a compute queue.
    pub(super) fn submit_compute(
        &self,
        job: &mut Job<super::QueueJob::ver>,
        cmdbuf: &uapi::drm_asahi_cmd_compute,
        attachments: &microseq::Attachments,
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
        id: u64,
        flush_stamps: bool,
    ) -> Result {
        let gpu = match (*self.dev)
            .gpu
            .as_any()
            .downcast_ref::<gpu::GpuManager::ver>()
        {
            Some(gpu) => gpu,
            None => {
                dev_crit!(self.dev.as_ref(), "GpuManager mismatched with Queue!\n");
                return Err(EIO);
            }
        };

        let mut alloc = gpu.alloc();
        let kalloc = &mut *alloc;

        mod_dev_dbg!(self.dev, "[Submission {}] Compute!\n", id);

        if cmdbuf.flags & !super::COMPUTE_LARGE_WG != 0 {
            return Err(EINVAL);
        }

        let mut user_timestamps: fw::job::UserTimestamps = Default::default();
        user_timestamps.start = common::get_timestamp_object(objects, cmdbuf.ts.start)?;
        user_timestamps.end = common::get_timestamp_object(objects, cmdbuf.ts.end)?;

        // This sequence number increases per new client/VM? assigned to some slot,
        // but it's unclear *which* slot...
        let slot_client_seq: u8 = (self.id & 0xff) as u8;

        let vm_bind = job.vm_bind.clone();

        // 32.41.566: dump the CDM control stream and probe what its words point at
        // (the CDM faults reading 0x7b80 and ignores USC_EXEC_BASE_CP).
        if crate::debug::verbose() {
            let slot = vm_bind.slot();
            let cs = cmdbuf.cdm_ctrl_stream_base;
            let mut w = [0u32; 16];
            let n = gpu.read_vm(slot, cs, &mut w);
            pr_info!(
                "asahi/cdm cs {:#x} ({} words): {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
                cs, n, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11]
            );
            for &x in w.iter().take(12) {
                if x >= 0x1000 && (x & 0xf0000000) != 0x40000000 {
                    gpu.probe_vm(slot, u64::from(x & !0x3f), "cdm op bare");
                    gpu.probe_vm(slot, self.usc_exec_base + u64::from(x & !0x3f), "cdm op +usc");
                }
            }
        }

        // 32.41.572: comp_apple=2 maps Apple's own compute job at Apple's VAs
        // (once; leaked -- one-shot diagnostic, the machine is reset after each
        // run) and keeps a CPU view of its output buffer for the readback.
        let comp_apple = *crate::module_parameters::comp_apple.value();
        if comp_apple == 2 {
            static MAPPED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
            if !MAPPED.swap(true, core::sync::atomic::Ordering::SeqCst) {
                let mut nok = 0u32;
                for res in crate::replay::CRES.iter() {
                    // 32.41.574: Apple's VA-0 buffer would land on 1<<36, which is
                    // already mapped (map_pages: "Page at IOVA 0x1000000000 is
                    // mapped") -- mapping over it clobbers a live kernel object.
                    if res.va < crate::mmu::IOVA_USER_BASE && *crate::module_parameters::creplay_skip0.value() != 0 {
                        pr_info!("asahi/creplay: {:#x} NOT mapped (below user base)\n", res.va);
                        continue;
                    }
                    let va = if res.va < crate::mmu::IOVA_USER_BASE { res.va + (1u64 << 36) } else { res.va };
                    let r: Result = (|| {
                        let mut obj = crate::gem::new_kernel_object(&self.dev, res.size)?;
                        let osz = obj.size();
                        let map = obj.vmap()?;
                        let p = map.as_mut_ptr() as *mut u8;
                        // SAFETY: `p` is valid for `osz` bytes, `res.data.len() <= res.size <= osz`.
                        unsafe {
                            core::ptr::write_bytes(p, 0, osz);
                            core::ptr::copy_nonoverlapping(res.data.as_ptr(), p, res.data.len());
                            if res.va == 0x100_000b_0000 && *crate::module_parameters::creplay_patch.value() == 1 {
                                core::ptr::write_unaligned(p.add(0x28) as *mut u32, 0x4000_0000);
                                pr_info!("asahi/creplay: stream +0x28 barrier -> terminator\n");
                            }
                            if res.va == 0x100_000b_0000 && *crate::module_parameters::creplay_patch.value() == 2 {
                                core::ptr::write_unaligned(p as *mut u32, 0x000f_0000);
                                pr_info!("asahi/creplay: stream word0 0x00080000 -> 0x000f0000 (unmapped poison)\n");
                            }
                            if res.va == 0x100_000b_0000 && *crate::module_parameters::creplay_patch.value() == 3 {
                                core::ptr::write_unaligned(p.add(4) as *mut u32, 0x0200_0000);
                                pr_info!("asahi/creplay: stream word1 0x01000000 -> 0x02000000 (poison)\n");
                            }
                            if res.va == 0x100_000b_0000 && *crate::module_parameters::creplay_patch.value() == 12 {
                                // Hypothesis: stream word 2 = ESL address bits 6..31 (0x2400 -> 0x90000).
                                // Word 2 -> 0x2401 (0x90040); the ESL buffer gets the program moved to +0x40.
                                // LOG.md 32.41.651.
                                core::ptr::write_unaligned(p.add(8) as *mut u32, 0x2401);
                                pr_info!("asahi/creplay: stream word2 0x2400 -> 0x2401 (ESL at +0x40)\n");
                            }
                            if res.va == 0x100_0009_0000 && *crate::module_parameters::creplay_patch.value() == 12 {
                                let mut prog = [0u8; 0x9c];
                                core::ptr::copy_nonoverlapping(p, prog.as_mut_ptr(), 0x9c);
                                core::ptr::write_bytes(p, 0, 0x100);
                                core::ptr::write_unaligned(p as *mut u32, 0x0000_000e); // +0: bare `end`
                                core::ptr::copy_nonoverlapping(prog.as_ptr(), p.add(0x40), 0x9c);
                                pr_info!("asahi/creplay: ESL moved 0x10000090000 -> +0x40, `end` at +0\n");
                            }
                            if res.va == 0x100_0000_0000 && *crate::module_parameters::creplay_patch.value() == 5 {
                                // Empty shader: the G15 terminator 0e 00 00 00 at the LoadShader target.
                                core::ptr::write_unaligned(p.add(0x6c0) as *mut u32, 0x0000_000e);
                                pr_info!("asahi/creplay: shader 0x100000006c0 -> terminator only (empty program)\n");
                            }
                            if res.va == 0x100_0000_0000 && *crate::module_parameters::creplay_patch.value() == 8 {
                                // G13 sentinel kernel (Mesa AGX2 encodings): every thread stores
                                // 0xdeadbeef to out[0] (0x10000030000). LOG.md 32.41.644.
                                const G13: [u8; 30] = [
                                    0x62, 0x05, 0xef, 0xbe, 0xad, 0xde, // ldimm r1, 0xdeadbeef
                                    0x62, 0x11, 0x00, 0x00, 0x03, 0x00, // ldimm r4, 0x00030000
                                    0x62, 0x15, 0x00, 0x01, 0x00, 0x00, // ldimm r5, 0x00000100
                                    0x45, 0x09, 0x08, 0x05, 0x00, 0xc0, 0x12, 0x00, // store r1, dr4, 0, i32, x, a
                                    0x38, 0x00, // wait a
                                    0x88, 0x00, // stop
                                ];
                                core::ptr::write_bytes(p.add(0x6c0), 0, 0x40);
                                core::ptr::copy_nonoverlapping(G13.as_ptr(), p.add(0x6c0), G13.len());
                                pr_info!("asahi/creplay: shader 0x100000006c0 <- G13 sentinel kernel (patch 8)\n");
                            }
                            if res.va == 0x100_0000_0000 && *crate::module_parameters::creplay_patch.value() == 9 {
                                // First self-authored G15 kernel (encoded by Apple's AGX3 MCCodeEmitter
                                // under emulation): out[tid] = 2*tid + 5. LOG.md 32.41.647.
                                const G15: [u8; 42] = [
                                    0x1c, 0xa0, 0x10, 0x06, // op6721 r1 = tid
                                    0x9f, 0x01, 0x56, 0x00, 0x02, 0x04, 0x08, 0xa8, 0x11, 0x05, // op4408 r0 = r1 + r1 (no last-use: r1 is the store index)
                                    0x9f, 0x11, 0x54, 0x00, 0x02, 0x0a, 0x00, 0x88, 0x11, 0x04, // op4405 r0 = r0 + 5
                                    0xe7, 0x00, 0x54, 0x00, 0x00, 0x01, 0x21, 0x00, 0x11, 0x00, 0x00, 0x90, 0x11, 0x00, // op9055 store r0 -> u8[r1] (Apple's)
                                    0x0e, 0x00, 0x00, 0x00, // op623 end
                                ];
                                core::ptr::write_bytes(p.add(0x6c0), 0, 0x40);
                                core::ptr::copy_nonoverlapping(G15.as_ptr(), p.add(0x6c0), G15.len());
                                pr_info!("asahi/creplay: shader 0x100000006c0 <- own G15 kernel 2*tid+5 (patch 9)\n");
                            }
                            if res.va == 0x100_0009_0000 && *crate::module_parameters::creplay_patch.value() == 10 {
                                // Own ESL launch program (G15 ISA): uniforms from immediates instead of
                                // loads: u0:u2 = 0x10000030000 (out), u4 = 0x1000. Apple's LoadShader
                                // group + setup ops verbatim. Apple's kernel then writes tid - 0x1000.
                                // LOG.md 32.41.649.
                                const ESL: [u8; 116] = [
                                    0x0c, 0x80, 0x42, 0x00, 0x00, 0x00, 0x18, 0x00, // mov r16, 0x30000
                                    0x1c, 0x80, 0x42, 0x00, 0x04, 0x00, 0x00, 0x00, // mov r17, 0x100
                                    0x2c, 0x80, 0x42, 0x00, 0x00, 0x08, 0x00, 0x00, // mov r18, 0x1000
                                    0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00, // op8695 (LoadShader group)
                                    0x77, 0x01, 0xaa, 0x0d, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00, // op5724 kernel 0x100000006c0
                                    0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // op8696
                                    0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, // mov r1, 0
                                    0x14, 0x81, 0x11, 0x06, 0x00, 0x00, 0x00, 0x00, // op6722 get_sr
                                    0x0c, 0x80, 0x02, 0x00, 0x08, 0x00, 0x00, 0x00, // mov r0, 512
                                    0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x08, 0xa8, 0x10, 0x05, // iadd r0, r0, r1
                                    0x1c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00, // mov r1, 256
                                    0x0f, 0x12, 0x54, 0x00, 0x4c, 0x00, // op538 r0:r1
                                    0x0b, 0x20, 0x09, 0x44, // uniform_store u0 <- r16
                                    0x1b, 0x22, 0x09, 0x04, // uniform_store u2 <- r17
                                    0x2b, 0x24, 0x09, 0x04, // uniform_store u4 <- r18
                                    0x0e, 0x00, 0x00, 0x00, // end
                                ];
                                core::ptr::write_bytes(p, 0, 0x100);
                                core::ptr::copy_nonoverlapping(ESL.as_ptr(), p, ESL.len());
                                pr_info!("asahi/creplay: ESL 0x10000090000 <- own G15 launch program (patch 10)\n");
                            }
                            if res.va == 0x100_0009_0000 && *crate::module_parameters::creplay_patch.value() == 11 {
                                // patch 10 minus Apple's setup ops (0x58..0x88: r1=0, get_sr, r0=512,
                                // iadd, r1=256, op538). Are they needed? LOG.md 32.41.650.
                                const ESL: [u8; 68] = [
                                    0x0c, 0x80, 0x42, 0x00, 0x00, 0x00, 0x18, 0x00, // mov r16, 0x30000
                                    0x1c, 0x80, 0x42, 0x00, 0x04, 0x00, 0x00, 0x00, // mov r17, 0x100
                                    0x2c, 0x80, 0x42, 0x00, 0x00, 0x08, 0x00, 0x00, // mov r18, 0x1000
                                    0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00, // op8695 (LoadShader group)
                                    0x77, 0x01, 0xaa, 0x0d, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00, // op5724 kernel 0x100000006c0
                                    0xf7, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // op8696
                                    0x0b, 0x20, 0x09, 0x44, // uniform_store u0 <- r16
                                    0x1b, 0x22, 0x09, 0x04, // uniform_store u2 <- r17
                                    0x2b, 0x24, 0x09, 0x04, // uniform_store u4 <- r18
                                    0x0e, 0x00, 0x00, 0x00, // end
                                ];
                                core::ptr::write_bytes(p, 0, 0x100);
                                core::ptr::copy_nonoverlapping(ESL.as_ptr(), p, ESL.len());
                                pr_info!("asahi/creplay: ESL 0x10000090000 <- own minimal launch program (patch 11)\n");
                            }
                            if res.va == 0x100_0009_0000 && *crate::module_parameters::creplay_patch.value() == 7 {
                                // LoadShader core @+0x3c, word 1 h2 (@+0x48) = A bits 30..15:
                                // 0x100000006c0 -> 0x100070006c0 (unmapped). A fault proves the
                                // ESL executes; none means the USC never runs it.
                                core::ptr::write_unaligned(p.add(0x48) as *mut u16, 0x0e00);
                                pr_info!("asahi/creplay: ESL LoadShader -> 0x100070006c0 (unmapped)\n");
                            }
                            if res.va == 0x100_0009_0000 && *crate::module_parameters::creplay_patch.value() == 6 {
                                // Empty ESL program: terminator as its first instruction.
                                core::ptr::write_unaligned(p as *mut u32, 0x0000_000e);
                                pr_info!("asahi/creplay: ESL 0x10000090000 -> terminator only (empty program)\n");
                            }
                            if res.va == 0x100_0008_0000 && *crate::module_parameters::creplay_patch.value() == 4 {
                                // Apple's post-run words: u64 0x6f00000000 at +0x20, 0x58000 at +0x28/+0x2c/+0x38.
                                for (o, v) in [(0x24usize, 0x6fu32), (0x28, 0x58000), (0x2c, 0x58000), (0x38, 0x58000)] {
                                    core::ptr::write_unaligned(p.add(o) as *mut u32, v);
                                }
                                pr_info!("asahi/creplay: 0x10000080000 +0x20..0x3c <- Apple post-run words\n");
                            }
                        }
                        crate::replay::comp_replay_note(res, p);
                        if res.va == 0x100_0003_0000 {
                            crate::replay::COMP_REPLAY_OUT.store(p as usize, core::sync::atomic::Ordering::SeqCst);
                        }
                        core::mem::forget(map);
                        let m = obj.map_at(vm_bind.vm(), va, crate::mmu::uprot(crate::mmu::PROT_GPU_SHARED_RW), false)?;
                        core::mem::forget(m);
                        core::mem::forget(obj);
                        Ok(())
                    })();
                    match r {
                        Ok(()) => nok += 1,
                        Err(e) => pr_info!("asahi/creplay: *** MAP FAILED *** {:#x} -> {:#x}: {:?}\n", res.va, va, e),
                    }
                }
                pr_info!("asahi/creplay: mapped {} of {}\n", nok, crate::replay::CRES.len());
            }
            crate::replay::comp_replay_dump("pre");
        }

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] VM slot = {}\n",
            id,
            vm_bind.slot()
        );

        let notifier = self.notifier.clone();

        let fence = job.fence.clone();
        let comp_job = job.get_comp()?;
        let ev_comp = comp_job.event_info();

        let preempt2_off = gpu.get_cfg().compute_preempt1_size;
        let preempt3_off = preempt2_off + 8;
        let preempt4_off = preempt3_off + 8;
        let preempt5_off = preempt4_off + 8;
        let preempt_size = preempt5_off + 8;

        let preempt_buf = self
            .ualloc
            .lock()
            .array_empty_tagged(preempt_size, b"CPMT")?;

        // 32.41.572: Apple's op 0xb carries four spill VAs (+0x188, spaced
        // 0x1e400, 0x1e400, 0x1000 in the capture) and a UMA scratch pointer
        // (+0x178); handler 0xb turns them into 0xa5c1/0xa5c9/0x10091/0x10099/
        // 0x10229/0x140a8 and 0x14070.
        // .667zzy: regions 0/1 are bounded ([VA|5, next VA - 0x400] in
        // 0xa5c1/0xa5c9 and 0x10099/0x10091) and hold the USC state the CDM
        // stores on preemption. Apple's 0x1e400 was sized (allocateSpillBufferData,
        // per register class x core count) for its 64-thread capture kernel; a
        // full-GPU dispatch preempted at the PM tick resumed threads with lost
        // registers (faults at 0x0 / 0xf00). Scale them.
        let spill_scale = (*crate::module_parameters::spill_scale.value()).max(1) as u64;
        let spill_r = 0x1e400u64 * spill_scale;
        let spill_offs = [0u64, spill_r, 2 * spill_r, 2 * spill_r + 0x1000];
        let spill_size = (2 * spill_r + 0x1000 + 0x2800 * spill_scale) as usize;
        let spill_buf: Option<GpuArray<u8>> = if comp_apple != 0 {
            Some(self.ualloc.lock().array_empty_tagged(spill_size, b"SPIL")?)
        } else {
            None
        };
        let mut uma_cbuf: Option<GpuArray<u8>> = if comp_apple != 0 {
            Some(kalloc.gpu.array_empty_tagged(0x100000, b"UMAC")?)
        } else {
            None
        };
        let spill_va: u64 = if comp_apple == 2 {
            // 32.41.574: Apple's own spill VAs (mapped by the replay).
            0x100_0096_8000
        } else {
            spill_buf.as_ref().map(|b| u64::from(b.gpu_pointer())).unwrap_or(0)
        };
        let uma_cva: u64 = uma_cbuf.as_ref().map(|b| u64::from(b.gpu_pointer())).unwrap_or(0);
        // 32.41.586: give compute pool 4 real pages (record layout: LOG.md
        // 32.41.540/548): page list [0x14], per-block [0x1c], pages [0x24]/[0x2c],
        // extents [0x34]/[0x44], sync word [0x48]; then DeviceControl 0x1e.
        // .667zzz: ONE pool for the device, built and installed once, as macOS
        // does (its compute ktrace has no page-list fill at all). Re-installing
        // id 4 for every queue made the page manager write the previous owner's
        // cached pages back and refill from a new list while the job ran: the
        // job's first preemption store got page 0 (UL1C write 0x0/0x100/0x300,
        // restore read 0xf00), deterministically for one of the three rotating
        // klow list slots. Everything lives in klow, shared into every user VM.
        let cpool_mib = *crate::module_parameters::cpool_real.value() as usize;
        let cpool: Option<Arc<fw::compute::ComputePool>> = if comp_apple != 0 && cpool_mib != 0 {
            let mut tries = 0u32;
            loop {
                match CPOOL_STATE.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => match Self::build_cpool(kalloc, cpool_mib) {
                        Ok(cp) => {
                            let rva = u64::from(cp.rec.gpu_pointer());
                            let lva = u64::from(cp.list.gpu_pointer());
                            CPOOL_GRAVE.bury(cp.clone());
                            // DC+4 = the installing job's VM slot (gUPM reads the klow
                            // list through it; context 63 missed at level 3), DC+8 = 63:
                            // Apple sends its process context 6 / 63 the same way.
                            gpu.uma_add_pages(vm_bind.slot(), crate::buffer::UMA_KCTX, rva, lva, cp.pages as u32, 4, 1);
                            let wait_ms = *crate::module_parameters::cpool_wait_ms.value();
                            if wait_ms != 0 {
                                match crate::gpu::uma_wait_filled(4, wait_ms) {
                                    Some(us) => pr_info!("asahi/cpool: pool 4 filled after {} us\n", us),
                                    None => pr_warn!("asahi/cpool: pool 4 fill not retired after {} ms\n", wait_ms),
                                }
                            }
                            CPOOL_STATE.store(2, Ordering::Release);
                            break Some(cp);
                        }
                        Err(e) => {
                            CPOOL_STATE.store(0, Ordering::Release);
                            return Err(e);
                        }
                    },
                    Err(2) => break CPOOL_GRAVE.current(),
                    Err(_) => {
                        // another queue is installing it
                        tries += 1;
                        if tries > 1000 {
                            return Err(EBUSY);
                        }
                        kernel::time::delay::fsleep(kernel::time::Delta::from_micros(100));
                    }
                }
            }
        } else {
            None
        };
        // The pool record and the word holding its address live OUTSIDE the
        // command, as Apple's do (32.41.572b: in-command, the firmware's own
        // timestamp store overwrote it). Record: device-global; pool_ptr word
        // @+0x100 and UMA scratch @+0x1000 of the per-job page.
        let cpool_va = cpool.as_ref().map(|c| u64::from(c.rec.gpu_pointer())).unwrap_or(uma_cva);
        let (cpool_ptr_va, uma_scratch_va) = (uma_cva + 0x100, uma_cva + 0x1000);
        if let Some(b) = uma_cbuf.as_mut() {
            let sl = b.as_mut_slice();
            sl[8] = 4; // pool id -> 0x14028 = 4, Apple's value
            sl[0x100..0x108].copy_from_slice(&cpool_va.to_le_bytes());
        }
        if comp_apple != 0 && crate::debug::verbose() {
            pr_info!("asahi/comp_apple: mode {} spill {:#x} uma scratch {:#x} pool {:#x}\n", comp_apple, spill_va, uma_cva, cpool_va);
        }

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Event #{} {:#x?} -> {:#x?}\n",
            id,
            ev_comp.slot,
            ev_comp.value,
            ev_comp.value.next(),
        );

        let timestamps = Arc::new(
            kalloc.shared.new_default::<fw::job::JobTimestamps>()?,
            GFP_KERNEL,
        )?;
        let prof_ts = timestamps.clone();

        // .667zzz9 timestamp instrument (`ts_dbg`, LOG.md 32.41.667zzz9):
        // 1 = positive control (user start stamp -> a kernel-shared word, with
        // HwDataB+0x28 rebased onto the shared heap at init), 2 = print only.
        // At completion it prints the kernel stamps, the CPU view of the user
        // word and the command's user pointers.
        let ts_dbg_mode = *crate::module_parameters::ts_dbg.value();
        let ts_dbg_obj = if ts_dbg_mode != 0 {
            Some(Arc::new(
                kalloc.shared.new_default::<fw::job::JobTimestamps>()?,
                GFP_KERNEL,
            )?)
        } else {
            None
        };
        let user_ts_iova = user_timestamps
            .end
            .as_ref()
            .or(user_timestamps.start.as_ref())
            .map(|u| u.mapping.iova() + u.offset as u64)
            .unwrap_or(0);
        let ts_dbg = if ts_dbg_mode != 0 && user_ts_iova != 0 {
            // CPU view of the user timestamp word (vmap now, in process context).
            let vmap = user_timestamps
                .end
                .as_ref()
                .or(user_timestamps.start.as_ref())
                .and_then(|u| u.mapping.debug_gem().map(|(g, o)| (g, o + u.offset)))
                .and_then(|(g, o)| g.owned_vmap::<u8>().ok().map(|v| (v, o)));
            Some((timestamps.clone(), ts_dbg_obj.clone(), vmap))
        } else {
            None
        };

        let uuid = 0;
        mod_dev_dbg!(self.dev, "[Submission {}] UUID = {:#x?}\n", id, uuid);

        // TODO: check
        #[ver(V >= V13_0B4)]
        let count = self.counter.fetch_add(1, Ordering::Relaxed);

        let comp = GpuObject::new_init_prealloc(
            kalloc.gpu_ro.alloc_object()?,
            |ptr: GpuWeakPointer<fw::compute::RunCompute::ver>| {
                let notifier = notifier.clone();
                let vm_bind = vm_bind.clone();
                try_init!(fw::compute::RunCompute::ver {
                    preempt_buf: preempt_buf,
                    spill_buf,
                    uma_cbuf,
                    cpool,
                    micro_seq: {
                        let mut builder = microseq::Builder::new();

                        let stats = gpu.initdata.runtime_pointers.stats.comp.weak_pointer();

                        'ms: {
                        // 32.41.572: macOS's exact compute microsequence.
                        #[ver(G >= G14X && V >= V15_5)]
                        if comp_apple != 0 {
                            let uuid_a: u32 = 0x0e0313b9;
                            let start = builder.add(microseq::StartComputeApple {
                                header: 0xb,
                                unk_4: U64(0),
                                unk_c: U64(0),
                                registers: U64(inner_weak_ptr!(ptr, registers).into()),
                                stats: U64(stats.into()),
                                work_queue: U64(ev_comp.info_ptr.into()),
                                vm_slot: vm_bind.slot(),
                                unk_30: 1,
                                event_generation: self.id as u32,
                                event_seq: U64(ev_comp.event_seq),
                                unk_40: 0,
                                job_params2: U64(inner_weak_ptr!(ptr, job_params2).into()),
                                unk_4c: 0,
                                uuid: uuid_a,
                                unk_54: 0,
                                attachments: {
                                    // 32.41.588: Apple's op 0xb carries one attachment
                                    // {0x10005c08000, 0x8400, 0x18, 1}; handler 0xb turns each
                                    // into a range record for SGX 0xc238 (fw:0x14e88..0x14f44).
                                    let mut a = *attachments;
                                    if *crate::module_parameters::cattach.value() != 0 {
                                        a.list[0] = microseq::Attachment {
                                            address: U64(0x100_05c0_8000),
                                            size: 0x8400,
                                            unk_c: 0x18,
                                            unk_e: 1,
                                        };
                                        a.count = 1;
                                    }
                                    a
                                },
                                padding: 0,
                                pool_ptr: U64(cpool_va),
                                unk_168: U64(0),
                                unk_170: U64(0),
                                uma_scratch: U64(uma_scratch_va),
                                unk_180: U64(0),
                                spill: if comp_apple == 2 {
                                    [
                                        U64(spill_va),
                                        U64(spill_va + 0x1e400),
                                        U64(spill_va + 0x3c800),
                                        U64(spill_va + 0x3d800),
                                    ]
                                } else {
                                    spill_offs.map(|o| U64(spill_va + o))
                                },
                                unk_flag: U64(inner_weak_ptr!(ptr, unk_flag).into()),
                                counter: U64(count),
                                notifier_buf: U64(inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf).into()),
                                unk_1c0: 0,
                            })?;
                            builder.add(microseq::Timestamp::ver {
                                header: microseq::op::Timestamp::new_v15_5(true),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.start_addr),
                                work_queue: ev_comp.info_ptr,
                                // .667zzz7: hk timestamp queries (vkCmdWriteTimestamp) need the
                                // user pointers here too (the generic path's form)
                                user_ts_pointers: if user_timestamps.any() {
                                    Some(inner_weak_ptr!(ptr, user_timestamp_pointers))
                                } else {
                                    None
                                },
                                unk_ts: inner_weak_ptr!(ptr, context_store_req),
                                uuid: uuid_a,
                                unk_30_padding: 0,
                            })?;
                            builder.add(microseq::WaitForIdle {
                                header: microseq::op::WaitForIdle::HEADER_V15_5,
                            })?;
                            builder.add(microseq::Timestamp::ver {
                                header: microseq::op::Timestamp::new_v15_5(false),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.end_addr),
                                work_queue: ev_comp.info_ptr,
                                // .667zzz7: hk timestamp queries (vkCmdWriteTimestamp) need the
                                // user pointers here too (the generic path's form)
                                user_ts_pointers: if user_timestamps.any() {
                                    Some(inner_weak_ptr!(ptr, user_timestamp_pointers))
                                } else {
                                    None
                                },
                                unk_ts: inner_weak_ptr!(ptr, context_store_req),
                                uuid: uuid_a,
                                unk_30_padding: 0,
                            })?;
                            let off = builder.offset_to(start);
                            builder.add(microseq::FinalizeComputeApple {
                                header: 0xc,
                                stats: U64(stats.into()),
                                work_queue: U64(ev_comp.info_ptr.into()),
                                vm_slot: vm_bind.slot(),
                                job_params2: U64(inner_weak_ptr!(ptr, job_params2).into()),
                                unk_20: 0,
                                uuid: uuid_a,
                                unk_28: 0,
                                fw_stamp: U64(ev_comp.fw_stamp_pointer.into()),
                                stamp_value: ev_comp.value.next(),
                                unk_38: Default::default(),
                                pool_ptr_ptr: U64(cpool_ptr_va),
                                restart_branch_offset: off,
                                has_attachments: (attachments.count > 0) as u8,
                                unk_flag: U64(inner_weak_ptr!(ptr, unk_flag).into()),
                                unk_71: U64(inner_weak_ptr!(ptr, context_store_req).into()),
                                unk_79: 0, // Apple: 0 (the 0xff at +0x78 is the top byte of the +0x71 pointer). LOG.md 32.41.588
                                unk_7a: [0; 2],
                            })?;
                            builder.add(microseq::WaitForIdle2 {
                                header: microseq::op::WaitForIdle2::HEADER_END_V15_5,
                            })?;
                            break 'ms builder.build(&mut kalloc.private)?;
                        }

                        let start_comp = builder.add(microseq::StartCompute::ver {
                            #[ver(V >= V15_5)]
                            header: microseq::op::StartCompute::HEADER_V15_5,
                            #[ver(V < V15_5)]
                            header: microseq::op::StartCompute::HEADER,
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            #[ver(G < G14X)]
                            job_params1: Some(inner_weak_ptr!(ptr, job_params1)),
                            #[ver(G >= G14X)]
                            job_params1: None,
                            #[ver(G >= G14X)]
                            registers: inner_weak_ptr!(ptr, registers),
                            #[ver(V >= V15_5)]
                            unk_1c: 0,
                            stats,
                            work_queue: ev_comp.info_ptr,
                            vm_slot: vm_bind.slot(),
                            unk_28: 0x1,
                            event_generation: self.id as u32,
                            event_seq: U64(ev_comp.event_seq),
                            unk_38: 0x0,
                            job_params2: inner_weak_ptr!(ptr, job_params2),
                            unk_44: 0x0,
                            uuid,
                            #[ver(V >= V15_5)]
                            unk_58: U64(0),
                            attachments: *attachments,
                            padding: Default::default(),
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            unk_flag: inner_weak_ptr!(ptr, unk_flag),
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            counter: U64(count),
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            notifier_buf: inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf),
                            #[ver(V >= V15_5)]
                            pool_ptr: U64(inner_weak_ptr!(ptr, pool).into()),
                            #[ver(V >= V15_5)]
                            unk_170: U64(0),
                            #[ver(V >= V15_5)]
                            unk_178: 0,
                            #[ver(V >= V15_5)]
                            unk_17c: 0,
                            #[ver(V >= V15_5)]
                            unk_180: U64(0),
                            #[ver(V >= V15_5)]
                            unk_188: Default::default(),
                            #[ver(V >= V15_5)]
                            unk_flag: inner_weak_ptr!(ptr, unk_flag),
                            #[ver(V >= V15_5)]
                            counter: U64(count),
                            #[ver(V >= V15_5)]
                            notifier_buf: inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf),
                            #[ver(V >= V15_5)]
                            unk_1a8: 0,
                        })?;

                        if user_timestamps.any() {
                            builder.add(microseq::Timestamp::ver {
                                #[ver(V >= V15_5)]
                                header: microseq::op::Timestamp::new_v15_5(true),
                                #[ver(V < V15_5)]
                                header: microseq::op::Timestamp::new(true),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.start_addr),
                                work_queue: ev_comp.info_ptr,
                                user_ts_pointers: Some(inner_weak_ptr!(ptr, user_timestamp_pointers)),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, context_store_req),
                                uuid,
                                unk_30_padding: 0,
                            })?;
                        }

                        #[ver(G < G14X)]
                        builder.add(microseq::WaitForIdle {
                            header: microseq::op::WaitForIdle::new(microseq::Pipe::Compute),
                        })?;
                        #[ver(G >= G14X && V < V15_5)]
                        builder.add(microseq::WaitForIdle2 {
                            header: microseq::op::WaitForIdle2::HEADER,
                        })?;

                        if user_timestamps.any() {
                            builder.add(microseq::Timestamp::ver {
                                #[ver(V >= V15_5)]
                                header: microseq::op::Timestamp::new_v15_5(false),
                                #[ver(V < V15_5)]
                                header: microseq::op::Timestamp::new(false),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.end_addr),
                                work_queue: ev_comp.info_ptr,
                                user_ts_pointers: Some(inner_weak_ptr!(ptr, user_timestamp_pointers)),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, context_store_req),
                                uuid,
                                unk_30_padding: 0,
                            })?;
                        }

                        let off = builder.offset_to(start_comp);
                        builder.add(microseq::FinalizeCompute::ver {
                            #[ver(V >= V15_5)]
                            header: microseq::op::FinalizeCompute::HEADER_V15_5,
                            #[ver(V < V15_5)]
                            header: microseq::op::FinalizeCompute::HEADER,
                            stats,
                            work_queue: ev_comp.info_ptr,
                            vm_slot: vm_bind.slot(),
                            #[ver(V < V13_0B4)]
                            unk_18: 0,
                            #[ver(V >= V15_5)]
                            unk_18_15_5: 0,
                            job_params2: inner_weak_ptr!(ptr, job_params2),
                            #[ver(V < V15_5)]
                            unk_24: 0,
                            #[ver(V < V15_5)]
                            uuid,
                            #[ver(V >= V15_5)]
                            unk_24_ptr: inner_weak_ptr!(ptr, context_store_req),
                            fw_stamp: ev_comp.fw_stamp_pointer,
                            stamp_value: ev_comp.value.next(),
                            #[ver(V < V15_5)]
                            unk_38: 0,
                            #[ver(V < V15_5)]
                            unk_3c: 0,
                            #[ver(V < V15_5)]
                            unk_40: 0,
                            #[ver(V < V15_5)]
                            unk_44: 0,
                            #[ver(V < V15_5)]
                            unk_48: 0,
                            #[ver(V < V15_5)]
                            unk_4c: 0,
                            #[ver(V < V15_5)]
                            unk_50: 0,
                            #[ver(V < V15_5)]
                            unk_54: 0,
                            #[ver(V < V15_5)]
                            unk_58: 0,
                            #[ver(G == G14 && V < V13_0B4)]
                            unk_5c_g14: U64(0),
                            #[ver(V < V15_5)]
                            restart_branch_offset: off,
                            #[ver(V < V15_5)]
                            has_attachments: (attachments.count > 0) as u32,
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            unk_64: Default::default(),
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            unk_flag: inner_weak_ptr!(ptr, unk_flag),
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            unk_79: Default::default(),
                            #[ver(V >= V15_5)]
                            unk_38_blob: Default::default(),
                            #[ver(V >= V15_5)]
                            unk_58_15_5: 0,
                            #[ver(V >= V15_5)]
                            pool_ptr_ptr: inner_weak_ptr!(ptr, pool_self),
                            #[ver(V >= V15_5)]
                            restart_branch_offset: off,
                            #[ver(V >= V15_5)]
                            has_attachments: (attachments.count > 0) as u8,
                            #[ver(V >= V15_5)]
                            unk_69_ptr: inner_weak_ptr!(ptr, context_store_req),
                            #[ver(V >= V15_5)]
                            unk_71: U64(0),
                            #[ver(V >= V15_5)]
                            unk_79_15_5: U64(0),
                            #[ver(V >= V15_5)]
                            unk_81_ptr: inner_weak_ptr!(ptr, context_store_req),
                            #[ver(V >= V15_5)]
                            unk_89: Default::default(),
                        })?;

                        #[ver(V >= V15_5)]
                        builder.add(microseq::WaitForIdle2 {
                            header: microseq::op::WaitForIdle2::HEADER_END_V15_5,
                        })?;

                        #[ver(V < V15_5)]
                        builder.add(microseq::RetireStamp::ver {
                            #[ver(V >= V15_5)]
                            header: microseq::op::RetireStamp::HEADER_V15_5,
                            #[ver(V < V15_5)]
                            header: microseq::op::RetireStamp::HEADER,
                            #[ver(V >= V15_5)]
                            unk_4: U64(0),
                            #[ver(V >= V15_5)]
                            ts_c: inner_weak_ptr!(ptr, command_time),
                            #[ver(V >= V15_5)]
                            unk_14: Default::default(),
                            #[ver(V >= V15_5)]
                            ts_24: inner_weak_ptr!(ptr, command_time),
                            #[ver(V >= V15_5)]
                            ts_2c: inner_weak_ptr!(ptr, command_time),
                            #[ver(V >= V15_5)]
                            ts_34: inner_weak_ptr!(ptr, command_time),
                        })?;
                        builder.build(&mut kalloc.private)?
                        }
                    },
                    notifier,
                    vm_bind,
                    timestamps,
                    user_timestamps,
                    ts_dbg_obj,
                })
            },
            |inner, _ptr| {
                let vm_slot = vm_bind.slot();
                try_init!(fw::compute::raw::RunCompute::ver {
                    tag: fw::workqueue::CommandType::RunCompute,
                    #[ver(V >= V13_0B4)]
                    counter: U64(count),
                    unk_4: 0,
                    vm_slot,
                    notifier: inner.notifier.gpu_pointer(),
                    unk_pointee: Default::default(),
                    #[ver(G < G14X)]
                    __pad0: Default::default(),
                    #[ver(G < G14X)]
                    job_params1 <- try_init!(fw::compute::raw::JobParameters1 {
                        preempt_buf1: inner.preempt_buf.gpu_pointer(),
                        cdm_ctrl_stream_base: U64(cmdbuf.cdm_ctrl_stream_base),
                        // buf2-5 Only if internal program is used
                        preempt_buf2: inner.preempt_buf.gpu_offset_pointer(preempt2_off),
                        preempt_buf3: inner.preempt_buf.gpu_offset_pointer(preempt3_off),
                        preempt_buf4: inner.preempt_buf.gpu_offset_pointer(preempt4_off),
                        preempt_buf5: inner.preempt_buf.gpu_offset_pointer(preempt5_off),
                        usc_exec_base_cp: U64(self.usc_exec_base),
                        unk_38: U64(0x8c60),
                        helper_program: cmdbuf.helper.binary, // Internal program addr | 1
                        unk_44: 0,
                        helper_arg: U64(cmdbuf.helper.data), // Only if internal program used
                        helper_cfg: cmdbuf.helper.cfg, // 0x40 if internal program used
                        unk_54: 0,
                        unk_58: 1,
                        unk_5c: 0,
                        iogpu_unk_40: 0, // 0x1c if internal program used
                        __pad: Default::default(),
                    }),
                    #[ver(V >= V15_5)]
                    pad_regs_15_5: Default::default(),
                    #[ver(G >= G14X)]
                    registers: fw::job::raw::RegisterArray::new(
                        inner_weak_ptr!(_ptr, registers.registers),
                        |r| {
                            // 32.41.572: macOS's kext-side compute register list
                            // (dt-big `big.02.bin`, entries before the firmware's
                            // own 0x10201 append), verbatim except for buffer VAs.
                            if comp_apple != 0 {
                                let replay = comp_apple == 2;
                                let (pb, pstep) = if replay {
                                    (0x100_000e_0000u64, 0x1b80u64)
                                } else {
                                    (u64::from(inner.preempt_buf.gpu_pointer()), preempt2_off as u64)
                                };
                                let cs = if replay { 0x100_000b_0000 } else { cmdbuf.cdm_ctrl_stream_base };
                                if crate::debug::verbose() {
                                    pr_info!("asahi/comp_apple: regs preempt {:#x}+{:#x} cdm stream {:#x}\n", pb, pstep, cs);
                                }
                                r.add(0x1a510, pb);
                                r.add(0x1a420, cs);
                                r.add(0x1a4d0, pb + pstep);
                                r.add(0x1a4d8, pb + pstep + 8);
                                r.add(0x1a4e0, pb + pstep + 16);
                                r.add(0x1a4e8, pb + pstep + 24);
                                r.add(0x1a440, 0x154024201);
                                r.add(0x1a458, 0x10c08860);
                                r.add(0x12091, 0x1c);
                                r.add(0x101d9, 0x1c);
                                for reg in [0x1a089u32, 0x1a091, 0x1a059, 0x1a061, 0x1a0b9, 0x1a0c1, 0x101d1, 0xd479] {
                                    r.add(reg, 0);
                                }
                                r.add(0x1a0e9, 0x8);
                                r.add(0x107a1, match *crate::module_parameters::cdm_107a1.value() {
                                    0 => 0xff0000,
                                    v => v,
                                });
                                r.add(0xa599, 0x6000400020);
                                r.add(0xd411, 0x36000000120);
                                for reg in [0x1a540u32, 0x14a9, 0xa351] {
                                    r.add(reg, 0x0d031326_0e0313b9);
                                }
                                if !replay {
                                    // Mesa's pipeline words are USC-relative.
                                    r.add(0x10071, self.usc_exec_base);
                                }
                                return;
                            }
                            r.add(0x1a510, inner.preempt_buf.gpu_pointer().into());
                            // 32.41.565: log what the CDM is pointed at.
                            pr_info!(
                                "asahi/cdm: stream base {:#x} end {:#x} helper bin {:#x} data {:#x} usc_exec_base {:#x}\n",
                                cmdbuf.cdm_ctrl_stream_base,
                                cmdbuf.cdm_ctrl_stream_end,
                                u64::from(cmdbuf.helper.binary),
                                cmdbuf.helper.data,
                                self.usc_exec_base
                            );
                            r.add(0x1a420, cmdbuf.cdm_ctrl_stream_base);
                            // buf2-5 Only if internal program is used
                            r.add(0x1a4d0, inner.preempt_buf.gpu_offset_pointer(preempt2_off).into());
                            r.add(0x1a4d8, inner.preempt_buf.gpu_offset_pointer(preempt3_off).into());
                            r.add(0x1a4e0, inner.preempt_buf.gpu_offset_pointer(preempt4_off).into());
                            r.add(0x1a4e8, inner.preempt_buf.gpu_offset_pointer(preempt5_off).into());
                            // The render path submits a compute job too, and on
                            // G14X this is the only USC base register never
                            // tested. LOG.md 32.41.195
                            let usc_base_cp = match *crate::module_parameters::usc_base_cp.value() {
                                u64::MAX => self.usc_exec_base,
                                v => v,
                            };
                            pr_info!(
                                "asahi/usc cp: exec_base {:#x} -> reg 0x10071 {:#x}\n",
                                self.usc_exec_base,
                                usc_base_cp
                            );
                            r.add(0x10071, usc_base_cp); // USC_EXEC_BASE_CP
                            r.add(0x11841, cmdbuf.helper.binary.into());
                            r.add(0x11849, cmdbuf.helper.data);
                            r.add(0x11f81, cmdbuf.helper.cfg.into());
                            r.add(0x1a440, 0x24201);
                            r.add(0x12091, 0 /* iogpu_unk_40 */);
                            /*
                            r.add(0x10201, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x10428, 0x100); // Some kind of counter?? Does this matter?
                            */
                        }
                    ),
                    __pad1: Default::default(),
                    microsequence: inner.micro_seq.gpu_pointer(),
                    microsequence_size: inner.micro_seq.len() as u32,
                    job_params2 <- try_init!(fw::compute::raw::JobParameters2::ver {
                        #[ver(V >= V13_0B4)]
                        unk_0_0: 0,
                        unk_0: Default::default(),
                        preempt_buf1: inner.preempt_buf.gpu_pointer(),
                        cdm_ctrl_stream_end: U64(cmdbuf.cdm_ctrl_stream_end),
                        unk_34: Default::default(),
                        #[ver(G < G14X)]
                        unk_g14x: 0,
                        // = the job's 0x1a440 (CDM control) word: the 15.5
                        // resume reprograms 0x1a440 from {unk_g14x, unk_58} | 8
                        // (fw:0x17998). LOG.md 32.41.667zzy.
                        #[ver(G >= G14X && V >= V15_5)]
                        unk_g14x: if comp_apple != 0 { 0x5402_4201 } else { 0x24201 },
                        #[ver(G >= G14X && V < V15_5)]
                        unk_g14x: 0x24201,
                        #[ver(V >= V15_5)]
                        unk_58: if comp_apple != 0 { 1 } else { 0 },
                        #[ver(V < V15_5)]
                        unk_58: 0,
                        #[ver(V < V13_0B4)]
                        unk_5c: 0,
                    }),
                    encoder_params <- try_init!(fw::compute::raw::EncoderParams::ver {
                        unk_8: 0x0,     // fixed
                        sync_grow: 0x0, // check!
                        unk_10: 0x0,    // fixed
                        encoder_id: 0,
                        unk_18: 0x0, // fixed
                        unk_mask: 0xffffffff,
                        #[ver(V < V15_5)]
                        sampler_array: U64(cmdbuf.sampler_heap),
                        #[ver(V < V15_5)]
                        sampler_count: cmdbuf.sampler_count as u32,
                        #[ver(V < V15_5)]
                        sampler_max: (cmdbuf.sampler_count as u32) + 1,
                    }),
                    meta <- try_init!(fw::compute::raw::JobMeta::ver {
                        unk_0: 0,
                        unk_2: 0,
                        // 32.41.575: comp_nopreempt -- let a slow job run past the timeslice.
                        no_preemption: *crate::module_parameters::comp_nopreempt.value() as u8,
                        stamp: ev_comp.stamp_pointer,
                        fw_stamp: ev_comp.fw_stamp_pointer,
                        stamp_value: ev_comp.value.next(),
                        stamp_slot: ev_comp.slot,
                        #[ver(V < V15_5)]
                        evctl_index: 0, // fixed
                        #[ver(V < V15_5)]
                        flush_stamps: flush_stamps as u32,
                        #[ver(V < V15_5)]
                        uuid,
                        // 15.5: the firmware uses this word as the workqueue's slot in
                        // the notifier's 4-entry waiter table (pstamp_*[idx], bit idx of
                        // has_*; fw:0x1f45c). Apple = 1 << AGXCLWorkQueue+0xe0, a small
                        // per-queue index; we have one workqueue per pipe per notifier, so
                        // 0. The running event_seq overflowed slot 3 on the 5th job and
                        // wrote into in_list/list_head (fw crash 0x1f900). LOG.md 32.41.664
                        #[ver(V >= V15_5)]
                        event_seq: 0,
                        #[ver(V < V15_5)]
                        event_seq: ev_comp.event_seq as u32,
                    }),
                    command_time: U64(0),
                    timestamp_pointers <- try_init!(fw::job::raw::TimestampPointers {
                        start_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), start)),
                        end_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), end)),
                    }),
                    user_timestamp_pointers: {
                        let mut p = inner.user_timestamps.pointers()?;
                        if let Some(o) = inner.ts_dbg_obj.as_ref() {
                            if *crate::module_parameters::ts_dbg.value() == 1 {
                                p.start_addr = Some(inner_ptr!(o.gpu_pointer(), start));
                            }
                        }
                        p
                    },
                    client_sequence: slot_client_seq,
                    pad_2d1: Default::default(),
                    unk_2d4: 0,
                    unk_2d8: 0,
                    #[ver(V >= V13_0B4)]
                    context_store_req: U64(0),
                    #[ver(V >= V13_0B4 && V < V15_5)]
                    context_store_compl: U64(0),
                    #[ver(V >= V13_0B4 && V < V15_5)]
                    unk_2e9: Default::default(),
                    // The command-side UMA pool the 15.5 compute start routine
                    // dereferences. Points at the same object StartCompute's
                    // `pool_ptr` registers, the way the vertex path points both
                    // at `scene.uma_pool`. LOG.md 32.41.31.
                    #[ver(V >= V15_5)]
                    unk_83d: Default::default(),
                    #[ver(V >= V15_5)]
                    // 32.41.597: with comp_apple the finalize path (fw:0x1f420 ->
                    // 0x252a0) dereferences this record's +0x48 once pool 4 is
                    // installed (DC 0x1e): it must be the real record, not the
                    // inline copy whose +0x48 is null.
                    uma_pool: if comp_apple != 0 {
                        U64(cpool_va)
                    } else {
                        U64(inner_weak_ptr!(_ptr, pool).into())
                    },
                    #[ver(V >= V15_5)]
                    uma_pool_flush: *crate::module_parameters::uma_pool_flush.value() as u8,
                    #[ver(V >= V15_5)]
                    unk_84b: Default::default(),
                    #[ver(V >= V13_0B4)]
                    unk_flag: U32(0),
                    #[ver(V >= V13_0B4)]
                    unk_pad: Default::default(),
                    // See the fragment path: pool + self-pointer for the 15.5
                    // microsequence.  LOG.md 32.41.26.
                    #[ver(V >= V15_5)]
                    pool_self: U64(inner_weak_ptr!(_ptr, pool).into()),
                    #[ver(V >= V15_5)]
                    pool: {
                        // 32.41.572: Apple's appended 0x14028 = 4 -- the compute
                        // pool id, as 0x14308 = 5 is the vertex one.
                        let mut a: Array<0x90, u8> = Default::default();
                        if comp_apple != 0 {
                            a[8] = 4;
                        }
                        a
                    },
                })
            },
        )?;

        core::mem::drop(alloc);

        fence.add_command();
        // Keep the PM-tick quiesce away from the CDM while this job is in
        // flight (gpu::comp_busy_update). LOG.md 32.41.667zzy.
        crate::gpu::comp_busy_update(1);
        // ts_dbg: CPU addresses of the command's user pointers and of the word
        // the firmware's timestamp routine (fw:0x26c10) stores the end stamp into.
        #[ver(V >= V13_0B4)]
        let ts_cmd_dbg: (usize, usize) = comp.with(|raw, _| {
            (
                &raw.context_store_req as *const _ as usize,
                &raw.user_timestamp_pointers as *const _ as usize,
            )
        });
        #[ver(V < V13_0B4)]
        let ts_cmd_dbg: (usize, usize) =
            comp.with(|raw, _| (0, &raw.user_timestamp_pointers as *const _ as usize));
        let r = comp_job.add_cb(comp, vm_bind.slot(), move |error| {
            crate::gpu::comp_busy_update(-1);
            crate::stats::pass_prof_comp(prof_ts.with(|raw, _| {
                (raw.start.load(Ordering::Relaxed), raw.end.load(Ordering::Relaxed))
            }));
            if let Some((t, o, vmap)) = ts_dbg.as_ref() {
                let (s0, e0) = t.with(|raw, _| (raw.start.load(Ordering::Relaxed), raw.end.load(Ordering::Relaxed)));
                let uval = vmap.as_ref().map(|(v, o)| {
                    let m = v.get();
                    // SAFETY: o + 8 <= mapping size was checked by get_timestamp_object.
                    unsafe { core::ptr::read_volatile(m.as_ptr().add(*o) as *const u64) }
                });
                // SAFETY: debug only; the command object outlives its completion callback.
                let (csr, us, ue) = unsafe {
                    (
                        if ts_cmd_dbg.0 != 0 { core::ptr::read_volatile(ts_cmd_dbg.0 as *const u64) } else { 0 },
                        core::ptr::read_unaligned(ts_cmd_dbg.1 as *const u64),
                        core::ptr::read_unaligned((ts_cmd_dbg.1 + 8) as *const u64),
                    )
                };
                let ctl = o.as_ref().map(|o| o.with(|raw, _| raw.start.load(Ordering::Relaxed)));
                pr_info!(
                    "asahi/ts: kernel {:#x}..{:#x} user iova {:#x} cpu {:x?} csr {:#x} uptrs {:#x} {:#x} ctl {:x?}\n",
                    s0, e0, user_ts_iova, uval, csr, us, ue, ctl
                );
            }
            if comp_apple == 2 {
                crate::replay::comp_replay_dump("complete");
            }
            if comp_apple != 0 && (crate::debug::verbose() || error.is_some()) {
                pr_info!("asahi/comp_apple: job complete, error {:?}\n", error);
                if *crate::module_parameters::ktrace_hist.value() >= 3 {
                    crate::channel::ktrace_hist_tail(200);
                }
            }
            if let Some(err) = error {
                fence.set_error(err.into())
            }

            fence.command_complete();
        });
        if r.is_err() {
            crate::gpu::comp_busy_update(-1);
        }
        r?;

        comp_job.next_seq();

        Ok(())
    }
}

