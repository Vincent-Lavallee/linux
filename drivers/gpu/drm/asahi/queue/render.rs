// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(clippy::unusual_byte_groupings)]

//! Render work queue.
//!
//! A render queue consists of two underlying WorkQueues, one for vertex and one for fragment work.
//! This module is in charge of creating all of the firmware structures required to submit 3D
//! rendering work to the GPU, based on the userspace command buffer.

use super::common;
use crate::alloc::Allocator;
use crate::debug::*;
use crate::fw::types::*;
use crate::gpu::GpuManager;
use crate::util::*;
use crate::{
    buffer,
    file,
    fw,
    gpu,
    microseq, mmu, //
};
use crate::{
    inner_ptr,
    inner_weak_ptr, //
};
use core::sync::atomic::Ordering;
use kernel::dma_fence::RawDmaFence;
use kernel::drm::sched::Job;
use kernel::prelude::*;
use kernel::sync::Arc;
use kernel::uapi;
use kernel::xarray;

const DEBUG_CLASS: DebugFlags = DebugFlags::Render;

/// Tiling/Vertex control bit to disable using more than one GPU cluster. This results in decreased
/// throughput but also less latency, which is probably desirable for light vertex loads where the
/// overhead of clustering/merging would exceed the time it takes to just run the job on one
/// cluster.
const TILECTL_DISABLE_CLUSTERING: u32 = 1u32 << 0;

#[versions(AGX)]
impl super::QueueInner::ver {
    /// Get the appropriate tiling parameters for a given userspace command buffer.
    fn get_tiling_params(
        cmdbuf: &uapi::drm_asahi_cmd_render,
        num_clusters: u32,
    ) -> Result<buffer::TileInfo> {
        let width: u32 = cmdbuf.width_px as u32;
        let height: u32 = cmdbuf.height_px as u32;
        let layers: u32 = cmdbuf.layers as u32;

        if layers == 0 || layers > 2048 {
            cls_pr_debug!(Errors, "Layer count invalid ({})\n", layers);
            return Err(EINVAL);
        }

        // This is overflow safe: all these calculations are done in u32.
        // At 64Kx64K max dimensions above, this is 2**32 pixels max.
        // In terms of tiles that are always larger than one pixel,
        // this can never overflow. Note that real actual dimensions
        // are limited to 16K * 16K below anyway.
        //
        // Once we multiply by the layer count, then we need to check
        // for overflow or use u64.

        let tile_width = 32u32;
        let tile_height = 32u32;

        let utile_width = cmdbuf.utile_width_px as u32;
        let utile_height = cmdbuf.utile_height_px as u32;

        match (utile_width, utile_height) {
            (32, 32) | (32, 16) | (16, 16) => (),
            _ => {
                cls_pr_debug!(
                    Errors,
                    "uTile size invalid ({} x {})\n",
                    utile_width,
                    utile_height
                );
                return Err(EINVAL);
            }
        };

        let utiles_per_tile_x = tile_width / utile_width;
        let utiles_per_tile_y = tile_height / utile_height;

        let utiles_per_tile = utiles_per_tile_x * utiles_per_tile_y;

        let tiles_x = width.div_ceil(tile_width);
        let tiles_y = height.div_ceil(tile_height);
        let tiles = tiles_x * tiles_y;

        let mtiles_x = 4u32;
        let mtiles_y = 4u32;
        let mtiles = mtiles_x * mtiles_y;

        let tiles_per_mtile_x = align(tiles_x.div_ceil(mtiles_x), 4);
        let tiles_per_mtile_y = align(tiles_y.div_ceil(mtiles_y), 4);
        let tiles_per_mtile = tiles_per_mtile_x * tiles_per_mtile_y;

        let mtile_x1 = tiles_per_mtile_x;
        let mtile_x2 = 2 * tiles_per_mtile_x;
        let mtile_x3 = 3 * tiles_per_mtile_x;

        let mtile_y1 = tiles_per_mtile_y;
        let mtile_y2 = 2 * tiles_per_mtile_y;
        let mtile_y3 = 3 * tiles_per_mtile_y;

        let rgn_entry_size = 5;
        // Macrotile stride in 32-bit words
        let rgn_size = align(rgn_entry_size * tiles_per_mtile * utiles_per_tile, 4) / 4;
        let tilemap_size = (4 * rgn_size * mtiles) as usize * layers as usize;

        let tpc_entry_size = 8;
        // TPC stride in 32-bit words
        let tpc_mtile_stride = tpc_entry_size * utiles_per_tile * tiles_per_mtile / 4;
        let tpc_size =
            (4 * tpc_mtile_stride * mtiles) as usize * layers as usize * num_clusters as usize;

        // No idea where this comes from, but it fits what macOS does...
        // GUESS: Number of 32K heap blocks to fit a 5-byte region header/pointer per tile?
        // That would make a ton of sense...
        let meta1_layer_stride = if num_clusters > 1 {
            (align(tiles_x, 2) * align(tiles_y, 4) * utiles_per_tile).div_ceil(0x1980)
        } else {
            0
        };

        let mut min_tvb_blocks = align((tiles_x * tiles_y).div_ceil(128), 8);

        if num_clusters > 1 {
            min_tvb_blocks = min_tvb_blocks.max(7 + 2 * layers);
        }

        Ok(buffer::TileInfo {
            tiles_x,
            tiles_y,
            tiles,
            utile_width,
            utile_height,
            //mtiles_x,
            //mtiles_y,
            tiles_per_mtile_x,
            tiles_per_mtile_y,
            //tiles_per_mtile,
            utiles_per_mtile_x: tiles_per_mtile_x * utiles_per_tile_x,
            utiles_per_mtile_y: tiles_per_mtile_y * utiles_per_tile_y,
            //utiles_per_mtile: tiles_per_mtile * utiles_per_tile,
            tilemap_size,
            tpc_size,
            meta1_layer_stride,
            #[ver(G < G14X)]
            meta1_blocks: meta1_layer_stride * (cmdbuf.layers as u32),
            #[ver(G >= G14X)]
            meta1_blocks: meta1_layer_stride,
            layermeta_size: if layers > 1 { 0x100 } else { 0 },
            min_tvb_blocks: min_tvb_blocks as usize,
            params: fw::vertex::raw::TilingParameters {
                rgn_size,
                unk_4: 0x88,
                ppp_ctrl: cmdbuf.ppp_ctrl,
                x_max: (width - 1) as u16,
                y_max: (height - 1) as u16,
                te_screen: ((tiles_y - 1) << 12) | (tiles_x - 1),
                te_mtile1: mtile_x3 | (mtile_x2 << 9) | (mtile_x1 << 18),
                te_mtile2: mtile_y3 | (mtile_y2 << 9) | (mtile_y1 << 18),
                tiles_per_mtile,
                tpc_stride: tpc_mtile_stride,
                unk_24: 0x100,
                unk_28: if layers > 1 {
                    0xe000 | (layers - 1)
                } else {
                    0x8000
                },
                helper_cfg: cmdbuf.vertex_helper.cfg,
                __pad: Default::default(),
            },
        })
    }

    /// Submit work to a render queue.
    pub(super) fn submit_render(
        &self,
        job: &mut Job<super::QueueJob::ver>,
        cmdbuf: &uapi::drm_asahi_cmd_render,
        vertex_attachments: &microseq::Attachments,
        fragment_attachments: &microseq::Attachments,
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
        id: u64,
        flush_stamps: bool,
    ) -> Result {
        mod_dev_dbg!(self.dev, "[Submission {}] Render!\n", id);

        // .667zzz42 `render_sig`: print each new kind of render command once (UMA leak hunt).
        if *crate::module_parameters::render_sig.value() != 0 {
            use core::sync::atomic::{AtomicU64, Ordering};
            static SEEN: [AtomicU64; 128] = [const { AtomicU64::new(0) }; 128];
            let c = cmdbuf;
            let sig: u64 = (u64::from(c.flags) & 0xffff)
                ^ (u64::from(c.samples) << 16)
                ^ (u64::from(c.layers) << 20)
                ^ (u64::from(c.width_px) << 28)
                ^ (u64::from(c.height_px) << 42)
                ^ ((c.zls_ctrl & 0xffff_ffff).rotate_left(7))
                ^ (u64::from(c.sample_size_B) << 56)
                ^ (u64::from(c.depth.base != 0) << 60)
                ^ (u64::from(c.stencil.base != 0) << 61)
                ^ (u64::from(c.vertex_helper.binary != 0) << 62)
                ^ (u64::from(c.fragment_helper.binary != 0) << 63)
                | 1;
            let mut new = false;
            for s in SEEN.iter() {
                match s.compare_exchange(0, sig, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => { new = true; break; }
                    Err(v) if v == sig => break,
                    Err(_) => continue,
                }
            }
            if new {
                pr_info!(
                    "asahi/rsig: vm {} flags {:#x} {}x{} layers {} samples {} ssz {} utile {}x{} zls {:#x} depth {:#x}/{:#x} stencil {:#x}/{:#x} vhelper {} fhelper {} oclq {:#x} merge {:#x}/{:#x}\n",
                    job.vm_bind.slot(), c.flags, c.width_px, c.height_px, c.layers, c.samples, c.sample_size_B,
                    c.utile_width_px, c.utile_height_px, c.zls_ctrl, c.depth.base, c.depth.comp_base,
                    c.stencil.base, c.stencil.comp_base, c.vertex_helper.binary, c.fragment_helper.binary,
                    c.isp_oclqry_base, c.isp_merge_upper_x, c.isp_merge_upper_y
                );
            }
        }

        if cmdbuf.flags
            & !(uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_VERTEX_SCRATCH
                | uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES
                | uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING
                | uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_DBIAS_IS_INT) as u32
            != 0
        {
            cls_pr_debug!(Errors, "Invalid flags ({:#x})\n", cmdbuf.flags);
            return Err(EINVAL);
        }

        if cmdbuf.width_px == 0
            || cmdbuf.height_px == 0
            || cmdbuf.width_px > 16384
            || cmdbuf.height_px > 16384
        {
            cls_pr_debug!(
                Errors,
                "Invalid dimensions ({}x{})\n",
                cmdbuf.width_px,
                cmdbuf.height_px
            );
            return Err(EINVAL);
        }

        let mut vtx_user_timestamps: fw::job::UserTimestamps = Default::default();
        let mut frg_user_timestamps: fw::job::UserTimestamps = Default::default();

        vtx_user_timestamps.start = common::get_timestamp_object(objects, cmdbuf.ts_vtx.start)?;
        vtx_user_timestamps.end = common::get_timestamp_object(objects, cmdbuf.ts_vtx.end)?;
        frg_user_timestamps.start = common::get_timestamp_object(objects, cmdbuf.ts_frag.start)?;
        frg_user_timestamps.end = common::get_timestamp_object(objects, cmdbuf.ts_frag.end)?;

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

        let nclusters = gpu.get_dyncfg().id.num_clusters;

        // Can be set to false to disable clustering (for simpler jobs), but then the
        // core masks below should be adjusted to cover a single rolling cluster.
        let mut clustering = nclusters > 1;

        if debug_enabled(debug::DebugFlags::DisableClustering)
            || cmdbuf.flags
                & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING as u32
                != 0
        {
            clustering = false;
        }

        #[ver(G != G14)]
        let tiling_control = {
            let render_cfg = gpu.get_cfg().render;
            let mut tiling_control = render_cfg.tiling_control;

            if !clustering {
                tiling_control |= TILECTL_DISABLE_CLUSTERING;
            }
            tiling_control
        };

        // .667zzz43: shared UMA pool ring reset gate (before the allocator lock: draining waits
        // for other jobs' Scenes to drop). DC+4/+8 as the install below computes them.
        let _uma_gate = {
            let slot = job.vm_bind.slot();
            let hwid = if *crate::module_parameters::uma_hwid_slot.value() != 0 { slot } else { 1 };
            let ctx = if *crate::module_parameters::uma_klow.value() != 0 {
                crate::buffer::UMA_KCTX
            } else {
                match *crate::module_parameters::uma_dc.value() {
                    1 => slot,
                    v => v.wrapping_sub(2),
                }
            };
            crate::buffer::uma_shared_gate(gpu, hwid, ctx)
        };

        let mut alloc = gpu.alloc();
        let kalloc = &mut *alloc;

        // This sequence number increases per new client/VM? assigned to some slot,
        // but it's unclear *which* slot...
        let slot_client_seq: u8 = (self.id & 0xff) as u8;

        let tile_info = Self::get_tiling_params(&cmdbuf, if clustering { nclusters } else { 1 })?;

        let buffer = &self.buffer;
        let notifier = self.notifier.clone();

        let tvb_autogrown = buffer.auto_grow()?;
        if tvb_autogrown {
            let new_size = buffer.block_count() as usize;
            // .667zzz41: rare, and the only record of how a game's heap use evolves.
            dev_info!(
                self.dev.as_ref(),
                "[Submission {}] TVB grew to {} bytes ({} blocks) due to overflows\n",
                id,
                new_size * buffer::BLOCK_SIZE,
                new_size,
            );
        }

        // .667zzz33: a scene bigger than the heap the firmware was last given
        // needs the PB descriptor rebuilt, which is only safe once this buffer's
        // earlier scenes are done (see Buffer::publish_info).
        // .667zzz41: the same for an auto-grow -- a pipelined game always has a scene in flight,
        // so without waiting the grown heap was never published and the next big frame hit the
        // old size (TA OOM, House of Ashes). Only happens when the heap actually grew.
        let want_published = tile_info.min_tvb_blocks.max(buffer.block_count() as usize);
        if want_published > buffer.published_blocks()
            && *crate::module_parameters::tvb_publish_wait_ms.value() != 0
            && !buffer.wait_idle(*crate::module_parameters::tvb_publish_wait_ms.value())
        {
            dev_warn!(
                self.dev.as_ref(),
                "[Submission {}] TVB grow to {} blocks deferred: earlier scenes still in flight\n",
                id,
                want_published
            );
        }
        let tvb_grown = buffer.ensure_blocks(tile_info.min_tvb_blocks)?;
        if tvb_grown {
            cls_dev_dbg!(
                TVBStats,
                &self.dev,
                "[Submission {}] TVB grew to {} bytes ({} blocks) due to dimensions ({}x{})\n",
                id,
                tile_info.min_tvb_blocks * buffer::BLOCK_SIZE,
                tile_info.min_tvb_blocks,
                cmdbuf.width_px,
                cmdbuf.height_px
            );
        }

        let scene = Arc::new(buffer.new_scene(kalloc, &tile_info)?, GFP_KERNEL)?;
        // A forced base makes the readout unambiguous: point the aperture at a
        // deliberately unmapped VA and the next fault either reports it or does
        // not. LOG.md 32.41.53.
        let uma_aperture_base = match *crate::module_parameters::uma_aperture_probe.value() {
            0 => u64::from(scene.uma_aperture_pointer()),
            v => v,
        };
        let uma_scratch_base = u64::from(scene.uma_scratch_pointer());
        let uma_sync_base = u64::from(scene.uma_sync_pointer());
        mod_dev_dbg!(
            self.dev,
            "asahi/uma: aperture base {:#x} scratch base {:#x} sync base {:#x}\n",
            uma_aperture_base,
            uma_scratch_base,
            uma_sync_base
        );


        // macOS never leaves the sampler array null.  m1n1's capture has
        // `sampler_array = unk_buf._addr` -- a real allocation -- alongside
        // `sampler_count = 0` and `sampler_max = 0`: a valid pointer to an empty
        // heap, not a null one.  Mesa passes 0 whenever a draw uses no samplers,
        // and on 15.5/G15 `gUPM` faults reading VA 0 (LOG.md 32.41.43/44).
        // Fall back to a mapped, driver-owned buffer in the same VM.
        let sampler_heap = if cmdbuf.sampler_heap != 0 {
            cmdbuf.sampler_heap
        } else {
            u64::from(scene.tvb_heapmeta_pointer())
        };

        let vm_bind = job.vm_bind.clone();

        // 32.41.531: map Apple's own captured user buffers at Apple's VAs, once.
        // Mappings and objects are deliberately leaked: this is a one-shot
        // diagnostic and the machine is reset after every run.
        if *crate::module_parameters::apple_replay.value() != 0 {
            static MAPPED: core::sync::atomic::AtomicBool =
                core::sync::atomic::AtomicBool::new(false);
            if !MAPPED.swap(true, core::sync::atomic::Ordering::SeqCst) {
                let mut nok = 0u32;
                for res in crate::replay::RES.iter() {
                    // apple_replay=3: leave Apple's pipeline object unmapped, so the
                    // first fetch of it names its own address. LOG.md 32.41.534.
                    if *crate::module_parameters::apple_replay.value() == 3
                        && res.va == 0x100_0012_8000
                    {
                        pr_info!("asahi/replay: pipeline object 0x10000128000 left UNMAPPED (arm 3)\n");
                        continue;
                    }
                    let va = if res.va < mmu::IOVA_USER_BASE {
                        res.va + (1u64 << 36)
                    } else {
                        res.va
                    };
                    let r: Result = (|| {
                        let mut obj = crate::gem::new_kernel_object(&self.dev, res.size)?;
                        let osz = obj.size();
                        {
                            let map = obj.vmap()?;
                            let p = map.as_mut_ptr() as *mut u8;
                            // SAFETY: `p` is valid for `osz` bytes, `res.data.len() <= res.size <= osz`.
                            unsafe {
                                core::ptr::write_bytes(p, 0, osz);
                                // replay_post=1: Apple's post-draw descriptors (32.41.611).
                                let data = if res.va == 0x100_0011_8000
                                    && *crate::module_parameters::replay_post.value() != 0
                                {
                                    pr_info!("asahi/replay: 0x10000118000 <- post-draw blob ({:#x} bytes)\n", crate::replay::R118000_POST.len());
                                    crate::replay::R118000_POST
                                } else {
                                    res.data
                                };
                                core::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len().min(osz));
                                // apple_replay=2: retarget Apple's LoadShader (word 1 of the
                                // ESL core at +0) to unmapped 0xbeef000000 -- bytes from
                                // isa-evidence/ldshdr.py, the encoding .518/.520 verified.
                                if *crate::module_parameters::apple_replay.value() == 2
                                    && res.va == 0x100_0012_8000
                                {
                                    for (o, v) in [(11usize, 0u8), (13, 0xde), (14, 0x7d), (15, 0x01)] {
                                        *p.add(o) = v;
                                    }
                                    pr_info!("asahi/replay: LoadShader -> 0xbeef000000 (arm 2)\n");
                                }
                                // poison_fs=1: same retarget on the TRIANGLE's fragment ESL
                                // program (0x10000138640): a fault at 0xbeef000000 proves the
                                // ISP launches it. LOG.md 32.41.614.
                                // poison_fs=2: the BG program's core (0x10000138300) instead --
                                // the positive control (BG demonstrably runs).
                                let pfs = *crate::module_parameters::poison_fs.value();
                                // poison_fs=3: the PPP fragment-program pointer itself (block 2,
                                // 0x58000+0x1cd4 = 0x4e19) -> 0x0beef000 (<<6 = 0x2fbbc0000 + base).
                                // LOG.md 32.41.618.
                                // replay_vp=1: quarter viewport (tx,sx,ty,sy = 16,16,16,-16).
                                // LOG.md 32.41.619.
                                if *crate::module_parameters::replay_vp.value() == 1 && res.va == 0x5_8000 {
                                    for (i, f) in [0x4180_0000u32, 0x4180_0000, 0x4180_0000, 0xc180_0000].iter().enumerate() {
                                        core::ptr::write_unaligned(p.add(0x1d30 + 4 * i) as *mut u32, *f);
                                    }
                                    pr_info!("asahi/replay: viewport -> 16,16,16,-16 (replay_vp 1)\n");
                                }
                                if pfs == 3 && res.va == 0x5_8000 {
                                    core::ptr::write_unaligned(p.add(0x1cd4) as *mut u32, 0x0beef000);
                                    pr_info!("asahi/replay: PPP FS pointer -> 0x0beef000 (poison_fs 3)\n");
                                }
                                // fs_g13=1: G13 sentinel program over Apple's triangle FS. LOG.md 32.41.643.
                                if *crate::module_parameters::fs_g13.value() == 1 && res.va == 0x100_0000_0000 {
                                    const G13: [u8; 30] = [
                                        0x62, 0x05, 0xef, 0xbe, 0xad, 0xde, // ldimm r1, 0xdeadbeef
                                        0x62, 0x11, 0x00, 0x48, 0x04, 0x00, // ldimm r4, 0x00044800
                                        0x62, 0x15, 0x00, 0x01, 0x00, 0x00, // ldimm r5, 0x00000100
                                        0x45, 0x09, 0x08, 0x05, 0x00, 0xc0, 0x12, 0x00, // store r1, dr4, 0, i32, x, a
                                        0x38, 0x00, // wait a
                                        0x88, 0x00, // stop
                                    ];
                                    core::ptr::write_bytes(p.add(0x6c0), 0, 0x40);
                                    core::ptr::copy_nonoverlapping(G13.as_ptr(), p.add(0x6c0), G13.len());
                                    pr_info!("asahi/replay: triangle FS 0x100000006c0 <- G13 sentinel program (fs_g13 1)\n");
                                }
                                if (pfs == 1 || pfs == 2) && res.va == 0x100_0013_8000 {
                                    let core = if pfs == 2 { 0x300usize } else { 0x640 };
                                    for (o, v) in [(core + 11, 0u8), (core + 13, 0xde), (core + 14, 0x7d), (core + 15, 0x01)] {
                                        *p.add(o) = v;
                                    }
                                    pr_info!("asahi/replay: FS LoadShader @+{:#x} -> 0xbeef0000xx (poison_fs {})\n", core, pfs);
                                }
                            }
                        }
                        if res.va == 0 {
                            // Keep a CPU view of the VA-0 buffer for the readback (32.41.611).
                            let map = obj.vmap()?;
                            crate::replay::DRAW_VA0.store(map.as_mut_ptr() as usize, core::sync::atomic::Ordering::SeqCst);
                            core::mem::forget(map);
                        }
                        if res.va == 0x100_0004_0000 {
                            let map = obj.vmap()?;
                            crate::replay::DRAW_RT.store(map.as_mut_ptr() as usize, core::sync::atomic::Ordering::SeqCst);
                            core::mem::forget(map);
                        }
                        let m = obj.map_at(vm_bind.vm(), va, mmu::uprot(mmu::PROT_GPU_SHARED_RW), false)?;
                        core::mem::forget(m);
                        core::mem::forget(obj);
                        Ok(())
                    })();
                    match r {
                        Ok(()) => {
                            nok += 1;
                            // No GPU-side readback: `Uat::read_slot`'s temporary
                            // page-table view oopsed on drop here (.531 run 1).
                            let w = |o: usize| -> u32 {
                                if o + 4 <= res.data.len() {
                                    u32::from_le_bytes([res.data[o], res.data[o + 1], res.data[o + 2], res.data[o + 3]])
                                } else {
                                    0
                                }
                            };
                            pr_info!(
                                "asahi/replay: {:#x} -> va {:#x} size {:#x} data {:08x} {:08x} {:08x} {:08x}\n",
                                res.va, va, res.size, w(0), w(4), w(8), w(12)
                            );
                        }
                        Err(e) => pr_info!(
                            "asahi/replay: *** MAP FAILED *** {:#x} -> va {:#x}: {:?}\n",
                            res.va, va, e
                        ),
                    }
                }
                pr_info!("asahi/replay: mapped {} of {}\n", nok, crate::replay::RES.len());
            }
        }

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] VM slot = {}\n",
            id,
            vm_bind.slot()
        );

        // 667zzw: the per-submission pool use count (see `uma_submit_tick`).
        if *crate::module_parameters::uma_real.value() != 0 {
            scene.uma_submit_tick(*crate::module_parameters::uma_tick.value() != 0);
        }

        // LOG.md 32.41.545: hand the pool its pages the way macOS does
        // (AGXUMAFList::updateSubmitInfo -> DeviceControl 0x1e).
        if *crate::module_parameters::uma_dc.value() != 0 {
            if let Some((pool, list, pages, pool_id)) = scene.uma_add_pages_args(vm_bind.slot()) {
                let ctx = match *crate::module_parameters::uma_dc.value() {
                    1 => vm_bind.slot(),
                    v => v - 2,
                };
                // 32.41.552: instrument check -- an unmapped list must fault if the
                // page manager ever reads it.
                let list = if *crate::module_parameters::uma_list_poison.value() != 0 {
                    0x5f_0000_0000
                } else {
                    list
                };
                // 667l: DC+4 lands in SGX 0x14020 bits 56.. = the UAT context the
                // page manager reads the list through (Apple: 6 = its process
                // context; ours was a constant 1 = only right while the job VM sat
                // in slot 1 -> a second process faulted in the dead vm 1).
                // 667zzj: klow -> Apple's pair, DC+4 = job context, DC+8 = 63.
                let klow = *crate::module_parameters::uma_klow.value() != 0;
                let ctx = if klow { crate::buffer::UMA_KCTX } else { ctx };
                let hwid = if *crate::module_parameters::uma_hwid_slot.value() != 0 {
                    vm_bind.slot()
                } else {
                    1
                };
                scene.uma_note_install(hwid, ctx);
                gpu.uma_add_pages(hwid, ctx, pool, list, pages, pool_id, 1);
            }
            crate::buffer::uma_shared_grow_check(gpu, kalloc, vm_bind.slot());
        }

        let ev_vtx = job.get_vtx()?.event_info();
        let ev_frag = job.get_frag()?.event_info();

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Vert event #{} -> {:#x?}\n",
            id,
            ev_vtx.slot,
            ev_vtx.value.next(),
        );
        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Frag event #{} -> {:#x?}\n",
            id,
            ev_frag.slot,
            ev_frag.value.next(),
        );

        let uuid_3d = 0;
        let uuid_ta = 0;

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Vert UUID = {:#x?}\n",
            id,
            uuid_ta
        );
        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Frag UUID = {:#x?}\n",
            id,
            uuid_3d
        );

        let fence = job.fence.clone();
        let frag_job = job.get_frag()?;

        mod_dev_dbg!(self.dev, "[Submission {}] Create Barrier\n", id);
        let barrier = kalloc.private.new_init(
            pin_init::zeroed::<fw::workqueue::Barrier::ver>(),
            |_inner, _p| {
                // barrier_s2=3: FinalizeVertex's stamp lands in this barrier's own
                // padding (+0x38) and both stamp pointers watch it, so only the
                // vertex FINALIZATION releases the fragment. LOG.md 32.41.628.
                // SAFETY: +0x38 is inside the barrier object (padding, 8 bytes).
                let fvst: GpuWeakPointer<crate::fw::types::FwStamp> =
                    unsafe { _p.offset(0x38, core::ptr::null()) };
                let s3 = *crate::module_parameters::barrier_s2.value() == 3;
                // barrier_s2=4: both pointers on the CPU-visible vertex stamp. LOG.md 32.41.629.
                let s4 = *crate::module_parameters::barrier_s2.value() == 4;
                try_init!(fw::workqueue::raw::Barrier::ver {
                    tag: fw::workqueue::CommandType::Barrier,
                    wait_stamp: if s3 {
                        fvst
                    } else if s4 {
                        // SAFETY: same event slot, CPU-visible stamp word.
                        unsafe { ev_vtx.stamp_pointer.offset(0, core::ptr::null()) }
                    } else {
                        ev_vtx.fw_stamp_pointer
                    },
                    #[ver(V < V15_5)]
                    wait_value: ev_vtx.value.next(),
                    #[ver(V < V15_5)]
                    wait_slot: ev_vtx.slot,
                    #[ver(V < V15_5)]
                    stamp_self: ev_frag.value.next(),
                    #[ver(V < V15_5)]
                    uuid: uuid_3d,
                    #[ver(V < V15_5)]
                    external_barrier: 0,
                    #[ver(V < V15_5)]
                    internal_barrier_type: 0,
                    #[ver(V < V15_5)]
                    padding: Default::default(),
                    // 🔑 NOT optional, and not a "skip if null". The
                    // dispatcher's run predicate (`fw:0x1a688`) has exactly one
                    // path to "cannot run" -- `mov w0, #0` at `fw:0x1a844` --
                    // and both branches into it come from here:
                    //
                    //     1a790: ldur x9, [x23, #12]   ; wait_stamp2
                    //     1a794: cbz  x9, 0x1a844      ; NULL -> cannot run
                    //     1a798: ldr  w10, [x23, #20]  ; wait_value
                    //     1a79c: ldr  w9, [x9]
                    //     1a7a8: b.gt 0x1a844          ; not reached -> cannot run
                    //
                    // Left null, an armed barrier can never be satisfied no
                    // matter what the awaited stamp does, and the fragment
                    // queue sits at `Pending 1 Busy 0` forever. The awaited
                    // value is the same `wait_value` the first pointer is
                    // checked against, so this is the second of the pair the
                    // event slot carries: `fw_stamp` at +0x04 and the
                    // CPU-visible `stamp` here. LOG.md 32.41.107.
                    #[ver(V >= V15_5)]
                    // 32.41.626: fw:0x1a778 passes the barrier if EITHER stamp has
                    // reached wait_value. The CPU-visible stamp reaches 0x100 before
                    // FinalizeVertex (ktrace: StartFragment before 10a/00a), so the
                    // fragment ran on an unfinished TA. barrier_s2=1 points both at
                    // the fw stamp.
                    wait_stamp2: U64(if s3 {
                        u64::from(fvst)
                    } else if s4 {
                        u64::from(ev_vtx.stamp_pointer)
                    } else if *crate::module_parameters::barrier_s2.value() != 0 {
                        u64::from(ev_vtx.fw_stamp_pointer)
                    } else {
                        u64::from(ev_vtx.stamp_pointer)
                    }),
                    // barrier_s2=2: an unreachable value (+0x7e00) -- does the barrier
                    // gate the fragment at all? LOG.md 32.41.627.
                    #[ver(V >= V15_5)]
                    wait_value: if *crate::module_parameters::barrier_s2.value() == 2 {
                        {
                            let mut v = ev_vtx.value.next();
                            for _ in 0..0x7e {
                                v.increment();
                            }
                            v
                        }
                    } else {
                        ev_vtx.value.next()
                    },
                    #[ver(V >= V15_5)]
                    wait_value64: U64(0),
                    // 32.41.664: the awaited stamp slot (fw:0x193b8); 0 only
                    // matched while the vertex slot happened to be 0.
                    #[ver(V >= V15_5)]
                    barrier_type: if *crate::module_parameters::barrier_dag.value() != 0 {
                        ev_vtx.slot
                    } else {
                        0
                    },
                    // 32.41.610: +0x24 is the DAG word the barrier ASSIGNS to
                    // this queue's stamp-slot counter (fw:0x19308 -> 0x19bb0, bits
                    // 8..14), not the awaited slot. The next RunFragment must
                    // carry the same sequence, so it takes the fragment's value.
                    #[ver(V >= V15_5)]
                    flags_slot: if *crate::module_parameters::barrier_dag.value() != 0 {
                        ev_frag.value.next().counter()
                    } else {
                        ev_vtx.slot << 8
                    },
                    #[ver(V >= V15_5)]
                    stamp_self: ev_frag.value.next(),
                    #[ver(V >= V15_5)]
                    uuid: uuid_3d,
                    #[ver(V >= V15_5)]
                    armed: *crate::module_parameters::barrier_armed.value(),
                    #[ver(V >= V15_5)]
                    unk_34: 0,
                    #[ver(V >= V15_5)]
                    padding: Default::default(),
                })
            },
        )?;

        // DIAGNOSTIC: the firmware's run predicate is `*wait_stamp2 >= wait_value`
        // (fw:0x1a790..0x1a7a8). The fragment never starts even though the vertex
        // stamp reaches 0x100, so log what the barrier actually asks for.
        // LOG.md 32.41.205
        crate::sub_pr_info!(
            "asahi/barrier: wait_value {:#x} vtx_slot {} frag_stamp_self {:#x} armed {}\n",
            ev_vtx.value.next().counter(),
            ev_vtx.slot,
            ev_frag.value.next().counter(),
            *crate::module_parameters::barrier_armed.value()
        );
        // SAFETY: +0x38 is the barrier's padding; see barrier_s2=3 above.
        let barrier_fvst: GpuWeakPointer<crate::fw::types::FwStamp> =
            unsafe { barrier.weak_pointer().offset(0x38, core::ptr::null()) };
        mod_dev_dbg!(self.dev, "[Submission {}] Add Barrier\n", id);
        frag_job.add(barrier, vm_bind.slot())?;

        let timestamps = Arc::new(
            kalloc.shared.new_default::<fw::job::RenderTimestamps>()?,
            GFP_KERNEL,
        )?;

        let unk1 = false;

        let mut tile_config: u64 = 0;
        if !unk1 {
            tile_config |= 0x280;
        }
        if cmdbuf.layers > 1 {
            tile_config |= 1;
        }
        if cmdbuf.flags & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES as u32
            != 0
        {
            tile_config |= 0x10000;
        }

        let samples_log2 = match cmdbuf.samples {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => {
                cls_pr_debug!(Errors, "Invalid sample count {}\n", cmdbuf.samples);
                return Err(EINVAL);
            }
        };

        let utile_config = ((tile_info.utile_width / 16) << 12)
            | ((tile_info.utile_height / 16) << 14)
            | samples_log2;

        // Calculate the number of 2KiB blocks to allocate per utile. This is
        // just a bit of dimensional analysis.
        let pixels_per_utile: u32 =
            (cmdbuf.utile_width_px as u32) * (cmdbuf.utile_height_px as u32);
        let samples_per_utile: u32 = pixels_per_utile << samples_log2;
        let utile_size_bytes: u32 = (cmdbuf.sample_size_B as u32) * samples_per_utile;
        let block_size_bytes: u32 = 2048;
        let blocks_per_utile: u32 = utile_size_bytes.div_ceil(block_size_bytes);

        #[ver(G >= G14X)]
        let frg_tilecfg = 0x0000000_00036011
            | (((tile_info.tiles_x - 1) as u64) << 44)
            | (((tile_info.tiles_y - 1) as u64) << 53)
            | (if unk1 { 0 } else { 0x20_00000000 })
            | (if cmdbuf.layers > 1 { 0x1_00000000 } else { 0 })
            | ((utile_config as u64 & 0xf000) << 28);

        // TODO: check
        #[ver(V >= V13_0B4)]
        let count_frag = self.counter.fetch_add(2, Ordering::Relaxed);
        #[ver(V >= V13_0B4)]
        let count_vtx = count_frag + 1;

        // Unknowns handling

        #[ver(G >= G14)]
        let g14_unk = 0x4040404;
        #[ver(G < G14)]
        let g14_unk = 0;
        #[ver(G < G14X)]
        let frg_unk_140 = 0x8c60;
        let frg_unk_158 = 0x1c;
        #[ver(G >= G14)]
        let load_bgobjvals = cmdbuf.isp_bgobjvals as u64;
        #[ver(G < G14)]
        let load_bgobjvals = cmdbuf.isp_bgobjvals as u64 | 0x400;
        let reload_zlsctrl = cmdbuf.zls_ctrl;
        // Companion of VDM_CTRL_STREAM_BASE (0x1c880).  The 15.5 G15S firmware
        // emits the two as a pair -- (0x1c880, addr & ~3) then (0x1c898, 1) --
        // in the helper at fw:0x18a4c that StartVertex reaches from fw:0x17584.
        // Pre-15.5 traces always had it at 0, and m1n1 notes that setting the
        // low bit there faults in UL1C0, so keep the old value on old firmware.
        // LOG.md 32.41.30.
        // 🔑 macOS writes 0 here, unconditionally, on this exact silicon
        // (`AGXTAChannelG15::generateRegisterList`, `mov x4, #0` at
        // fffffe000911b084).  The `1` this driver carried came from reading the
        // firmware helper at fw:0x18a4c, but `regemit.py` says StartVertex
        // appends 0x1c880 and *not* 0x1c898, so the register list's value is
        // final and that reading was wrong.  With 1 the command-stream decoder
        // runs off the end of the 512 KiB stream BO on every submission; with 0
        // it parses the stream and stops at Mesa's terminator, 3/3.
        // That also retires the cs+0x0c byte poke from 32.41.185, which was
        // compensating for the mis-parse.  LOG.md 32.41.219.
        #[ver(G >= G14X && V >= V15_5)]
        let vdm_ctrl_stream_unk = 0;
        #[ver(G >= G14X && V < V15_5)]
        let vdm_ctrl_stream_unk = 0;
        // G15S encodes the control-stream address with a 2-bit, 1-based segment
        // in bits [37:36]: whatever is written, the hardware fetches from
        // `((segment + 1) << 36) | (addr & 0xfffffffff)`.  Measured on three
        // runs -- 0x11ffff04000 -> 0x2ffff04000 (seg 1 -> 2) and 0x02ffff04000
        // -> 0x3ffff04000 (seg 2 -> 3) -- and the one run that pre-decremented
        // fetched Mesa's base exactly.  Pre-decrement to compensate; the user VA
        // range is clamped to 1 << 38 in `file.rs` so the segment always fits.
        // LOG.md 32.41.37.
        #[ver(G >= G14X && V >= V15_5)]
        let vdm_ctrl_stream_base = (cmdbuf.vdm_ctrl_stream_base & !3).saturating_sub(
            if *crate::module_parameters::user_va_shift.value() != 0 {
                0
            } else {
                1 << 36
            },
        );
        #[ver(G >= G14X && V < V15_5)]
        let vdm_ctrl_stream_base = cmdbuf.vdm_ctrl_stream_base;
        // The same segment encoding applies to every address the register
        // array carries, not just the control stream: with the control stream
        // fixed the VDM ran on and faulted writing `tilemap + 2^36 + progress`
        // (0x3800150000 -> 0x4800178080), which is the identical +2^36.  So
        // pre-decrement every GPU VA programmed through a register.  The window
        // test is what keeps the sentinels (0, 0x8000_0000_0000_0000,
        // 0x4_0000_0000_0000) and the small USC-relative offsets out of it: only
        // addresses in the clamped user range [2^36, 2^38) are touched.
        // LOG.md 32.41.39.
        //
        // Superseded by `user_va_shift`: the same +2^36 turned up on an address
        // the VDM read out of Mesa's control stream (0x2ffff88000, mapped, asked
        // for as 0x3ffff88000), which no amount of register rewriting can reach.
        // So when the page tables carry the bias instead, the registers must go
        // back to holding the plain VA -- one knob picks the model.
        // 🔑 The two models were tied to one knob, and the macOS capture says
        // they are not exclusive. Live TA capture, 2026-09-11: the SAME buffer
        // appears in both register lists with different encodings --
        //
        //   fragment  0x16429 (tvb_tilemap)  = 0x1000308000   base + offset
        //   TA        0x1c039 (tvb_tilemap)  = 0x0000308000   offset only
        //   fragment  0x16098 (tvb_heapmeta) = 0x1000309000
        //   TA        0x1c079 (tvb_heapmeta) = 0x0000309000
        //
        // -- i.e. **TA address registers carry `VA - 0x1000000000` (2^36)**,
        // fragment ones carry the unfolded address. Every macOS TA address is
        // small (0x308000, 0x330000, 0x18000, 0x2c0060); ours are full VAs.
        //
        // `user_va_shift != 0` puts the bias in the page tables and hands the
        // registers the plain VA, which the VDM control-stream reads need
        // (32.41.105). But with those aliases mapped, a *folded* register value
        // resolves too -- so both can hold at once. `reg_va_fold` forces the
        // register-side fold independently of the page-table model.
        // LOG.md 32.41.372.
        #[ver(V >= V15_5)]
        let va_bias: u64 = if *crate::module_parameters::reg_va_fold.value() != 0 {
            1 << 36
        } else if *crate::module_parameters::user_va_shift.value() != 0 {
            0
        } else {
            1 << 36
        };
        #[ver(V < V15_5)]
        let va_bias: u64 = 0;
        // 32.41.433: the TA address field is a 32-bit offset into a 2^36 window,
        // not a VA. macOS's three tiler buffers sit at 0x1000308000 / 0x1000309000
        // / 0x1000330000 and its TA registers read 0x308000 / 0x309000 / 0x330000
        // -- a difference of exactly 2^36 in all three, so the register carries
        // `VA & 0xffffffff`. Every macOS TA address is <= 22 bits; every one of
        // ours is 38, because our allocator puts these buffers at 0x38_00000000.
        //
        // This is the half `reg_va_fold` could not express. That folds the raw VA
        // (0x3800148000 -> 0x2800148000, still 38 bits, pointing at nothing), which
        // is why it faulted 3/3 in 32.41.373. `user_va_win36` supplies the other
        // half -- it maps the buffer at 2^36 | (VA & 0xffffffff) -- but nothing made
        // the register carry the folded alias. Both are needed together.
        //
        // The tag is preserved: 0x1c031 carries `1 << 63` and 0x1c950 `4 << 48`,
        // both inside the top 16 bits, and `va()` is applied AFTER the tag is OR'd
        // in (see the tvb_heapmeta_ptr site), so fold only the low 48.
        #[ver(V >= V15_5)]
        let reg_win36 = *crate::module_parameters::reg_va_win36.value() != 0;
        #[ver(V < V15_5)]
        let reg_win36 = false;
        let va = |a: u64| -> u64 {
            if reg_win36 {
                let tag = a & 0xffff_0000_0000_0000;
                let addr = a & 0x0000_ffff_ffff_ffff;
                // Fold ONLY inside the configured win36 windows, so the register
                // value and the page-table alias can never disagree. Folding an
                // address the mapping side did not alias points the TA at nothing
                // and faults -- which is exactly how `reg_va_fold` failed in
                // 32.41.373. Our TA address registers span three ranges
                // (0x38xxxxxxxx tiler, 0x3c00040000 tpc, 0x2ffff04000), so the
                // window has to cover every one of them or the test is invalid.
                let w1min = *crate::module_parameters::win36_min_va.value();
                let w1max = *crate::module_parameters::win36_max_va.value();
                let w2min = *crate::module_parameters::win36_min_va2.value();
                let w2max = *crate::module_parameters::win36_max_va2.value();
                let in1 = (w1min != 0 || w1max != 0)
                    && addr >= w1min
                    && (w1max == 0 || addr < w1max);
                let in2 = (w2min != 0 || w2max != 0)
                    && addr >= w2min
                    && (w2max == 0 || addr < w2max);
                if (in1 || in2) && addr >= (1 << 32) {
                    return tag | (addr & 0xffff_ffff);
                }
                return a;
            }
            if va_bias != 0 && a >= (1 << 36) && a < (1 << 38) {
                a - va_bias
            } else {
                a
            }
        };

        // The ISP takes the BG/EOT program address from the register verbatim:
        // with `eot.usc = 0xfff68204` it faulted on exactly `0xfff68200`, and
        // neither reg 0x10069 nor `Start3D.usc_exec_base_isp` moved that
        // address when overridden. Meanwhile `usc_exec_base + usc` is mapped
        // and holds real shader words. So on G15S the driver has to add the
        // base itself rather than leave the address USC-relative. The low 3
        // bits are flags (`| 4`) and survive the addition, which is
        // page-aligned. LOG.md 32.41.105.
        #[ver(V >= V15_5)]
        let usc_bias: u64 = if *crate::module_parameters::usc_window.value() != 0 {
            *crate::module_parameters::usc_window.value()
        } else if *crate::module_parameters::usc_absolute.value() != 0 {
            self.usc_exec_base
        } else {
            0
        };
        #[ver(V < V15_5)]
        let usc_bias: u64 = 0;
        let usc = |a: u64| -> u64 { a + usc_bias };
        let iogpu_unk54 = 0x3a0012006b0003;
        let iogpu_unk56 = 1;
        #[ver(G < G14)]
        let tiling_control_2 = 0;
        #[ver(G >= G14X)]
        let tiling_control_2 = 4;
        #[ver(G >= G14X)]
        let vtx_unk_f0 = 0x1c;
        #[ver(G < G14)]
        let vtx_unk_f0 = 0x1c + (align(tile_info.meta1_blocks, 4) as u64);
        let vtx_unk_118 = 0x1c;

        // DRM_ASAHI_RENDER_DBIAS_IS_INT chosen to match hardware bit.
        let isp_ctl = 0xc000u32
            | (cmdbuf.flags & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_DBIAS_IS_INT as u32);

        // Always allow preemption at the UAPI level
        let no_preemption = false;

        mod_dev_dbg!(self.dev, "[Submission {}] Create Frag\n", id);
        let frag = GpuObject::new_init_prealloc(
            kalloc.gpu_ro.alloc_object()?,
            |ptr: GpuWeakPointer<fw::fragment::RunFragment::ver>| {
                let scene = scene.clone();
                let notifier = notifier.clone();
                let vm_bind = vm_bind.clone();
                let timestamps = timestamps.clone();
                let private = &mut kalloc.private;
                try_init!(fw::fragment::RunFragment::ver {
                    micro_seq: {
                        let mut builder = microseq::Builder::new();

                        let stats = inner_weak_ptr!(
                            gpu.initdata.runtime_pointers.stats.frag.weak_pointer(),
                            stats
                        );

                        let start_frag = builder.add(microseq::StartFragment::ver {
                            #[ver(V >= V15_5)]
                            header: microseq::op::StartFragment::HEADER_V15_5,
                            #[ver(V < V15_5)]
                            header: microseq::op::StartFragment::HEADER,
                            #[ver(G < G14X)]
                            job_params2: Some(inner_weak_ptr!(ptr, job_params2)),
                            #[ver(G < G14X)]
                            job_params1: Some(inner_weak_ptr!(ptr, job_params1)),
                            #[ver(G >= G14X)]
                            job_params1: None,
                            #[ver(G >= G14X)]
                            job_params2: None,
                            #[ver(G >= G14X)]
                            registers: inner_weak_ptr!(ptr, registers),
                            scene: scene.gpu_pointer(),
                            stats,
                            busy_flag: {
                                // DIAGNOSTIC: StartFragment reads *busy_flag (fw 0x15488
                                // reads [x19,#0x2c]). Set = the ISP started; clear = the
                                // handler bailed before starting. LOG.md 32.41.208
                                let bf = inner_weak_ptr!(ptr, busy_flag);
                                crate::sub_pr_info!(
                                    "asahi/frag: work_item {:#x} busy_flag {:#x}\n",
                                    u64::from(ptr),
                                    u64::from(bf)
                                );
                                bf
                            },
                            // 667zzx: Apple points this at scene+0x4c (LOG.md 32.41.608).
                            #[ver(G >= G14X)]
                            tvb_overflow_count: if *crate::module_parameters::tvb_ovf_scene.value() != 0 {
                                inner_weak_ptr!(scene.weak_pointer(), tvb_overflow)
                            } else {
                                inner_weak_ptr!(ptr, tvb_overflow_count)
                            },
                            #[ver(G < G14X)]
                            tvb_overflow_count: inner_weak_ptr!(ptr, tvb_overflow_count),
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            work_queue: ev_frag.info_ptr,
                            work_item: ptr,
                            vm_slot: vm_bind.slot(),
                            unk_50: 0x1, // fixed
                            // 32.41.634: low 6 bits of the fw-appended context register
                            // (fw:0x15b80: `vm_slot << 8 | this & 0x3f`). The vertex side is
                            // already 0 (vtx_qid_zero); leaving the queue id here gave the 3D
                            // 0x103 against the TA's 0x100. Apple: 0xb00 for both.
                            event_generation: if *crate::module_parameters::frag_qid_zero.value() != 0 {
                                0
                            } else {
                                self.id as u32
                            },
                            buffer_slot: scene.slot(),
                            sync_grow: 0,
                            event_seq: U64(ev_frag.event_seq),
                            unk_68: 0,
                            unk_758_flag: inner_weak_ptr!(ptr, unk_758_flag),
                            unk_job_buf: inner_weak_ptr!(ptr, unk_buf_0),
                            #[ver(V >= V13_3 && V < V15_5)]
                            unk_7c_0: U64(0),
                            #[ver(V >= V15_5)]
                            unk_84_0: 0,
                            unk_7c: 0,
                            unk_80: 0,
                            unk_84: unk1.into(),
                            uuid: uuid_3d,
                            #[ver(V >= V15_5)]
                            unk_98: U64(0),
                            // 32.41.635: Apple's captured StartFragment has count 0 (op+0x1a0);
                            // each entry makes fw:0x15c88 emit a cache-range command at ISP start.
                            attachments: if *crate::module_parameters::frag_no_att.value() != 0 {
                                let mut a = *fragment_attachments;
                                a.count = 0;
                                a
                            } else {
                                *fragment_attachments
                            },
                            padding: 0,
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            counter: U64(count_frag),
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            notifier_buf: inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf),
                            // 32.41.598: Apple's StartFragment+0x1a8 is the SAME pool record
                            // as StartVertex+0x190 (both 0xfffffc20c0908000), base (+0x1b0)
                            // 0, and its own scratch at +0x1c0 (0x...4c8000 vs vertex 0x...4a0000).
                            #[ver(V >= V15_5)]
                            pool_ptr: if *crate::module_parameters::uma_apple.value() != 0 {
                                U64(scene.uma_pool_pointer().into())
                            } else {
                                U64(inner_weak_ptr!(ptr, pool).into())
                            },
                            // The fragment half of the same pair: pool pointer
                            // then pool base.  A bare clear runs both pipes, so
                            // leaving this one null while the vertex side got a
                            // base tests nothing.  LOG.md 32.41.60/66.
                            #[ver(V >= V15_5)]
                            unk_1b0: U64(match *crate::module_parameters::uma_base_probe.value() {
                                _ if *crate::module_parameters::uma_apple.value() != 0 => 0,
                                0 if *crate::module_parameters::uma_pool_id.value() >= 0 => {
                                    uma_aperture_base
                                }
                                0 => 0,
                                v => v | 0x100000,
                            }),
                            #[ver(V >= V15_5)]
                            unk_1b8: 0,
                            #[ver(V >= V15_5)]
                            unk_1bc: 0,
                            // The fragment half of the same second register;
                            // its record starts at 0x1a8, so record+0x18 is
                            // 0x1c0. LOG.md 32.41.87.
                            #[ver(V >= V15_5)]
                            unk_1c0: U64(match *crate::module_parameters::uma_scratch_probe.value() {
                                _ if *crate::module_parameters::uma_apple.value() != 0 => {
                                    u64::from(scene.uma_fwbuf_pointer()) + 0x80000
                                }
                                0 => uma_scratch_base,
                                v => v | 0x100000,
                            }),
                            #[ver(V >= V15_5)]
                            unk_1c8: Default::default(),
                            #[ver(V >= V15_5)]
                            counter: U64(count_frag),
                            #[ver(V >= V15_5)]
                            notifier_buf: inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf),
                            #[ver(V >= V15_5)]
                            unk_1e0: 0,
                        })?;

                        // 32.41.598: Apple's 3D microsequence (capture4 raw-ta-microseq-draw
                        // finishEncode #1, scratchpad apple_3d_useq.bin) is StartFragment,
                        // Timestamp(1), WaitForIdle (bare 1), Timestamp(0), FinalizeFragment --
                        // the same shape as the TA's. Without the wait, FinalizeFragment runs
                        // ~3 us after the start.
                        let frg_apple_useq = *crate::module_parameters::vtx_apple_useq.value() != 0;
                        if frg_user_timestamps.any() || frg_apple_useq {
                            builder.add(microseq::Timestamp::ver {
                                #[ver(V >= V15_5)]
                                header: microseq::op::Timestamp::new_v15_5(true),
                                #[ver(V < V15_5)]
                                header: microseq::op::Timestamp::new(true),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.start_addr),
                                work_queue: ev_frag.info_ptr,
                                user_ts_pointers: Some(inner_weak_ptr!(ptr, user_timestamp_pointers)),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, unk_ts),
                                uuid: uuid_3d,
                                unk_30_padding: 0,
                            })?;
                        }

                        #[ver(G < G14X)]
                        builder.add(microseq::WaitForIdle {
                            header: microseq::op::WaitForIdle::new(microseq::Pipe::Fragment),
                        })?;
                        #[ver(G >= G14X && V < V15_5)]
                        builder.add(microseq::WaitForIdle2 {
                            header: microseq::op::WaitForIdle2::HEADER,
                        })?;
                        #[ver(V >= V15_5)]
                        if frg_apple_useq {
                            builder.add(microseq::WaitForIdle {
                                header: microseq::op::WaitForIdle::HEADER_V15_5,
                            })?;
                        }

                        if frg_user_timestamps.any() || frg_apple_useq {
                            builder.add(microseq::Timestamp::ver {
                                #[ver(V >= V15_5)]
                                header: microseq::op::Timestamp::new_v15_5(false),
                                #[ver(V < V15_5)]
                                header: microseq::op::Timestamp::new(false),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.end_addr),
                                work_queue: ev_frag.info_ptr,
                                user_ts_pointers: Some(inner_weak_ptr!(ptr, user_timestamp_pointers)),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, unk_ts),
                                uuid: uuid_3d,
                                unk_30_padding: 0,
                            })?;
                        }

                        let off = builder.offset_to(start_frag);
                        builder.add(microseq::FinalizeFragment::ver {
                            #[ver(V >= V15_5)]
                            header: microseq::op::FinalizeFragment::HEADER_V15_5,
                            #[ver(V < V15_5)]
                            header: microseq::op::FinalizeFragment::HEADER,
                            uuid: uuid_3d,
                            unk_8: 0,
                            fw_stamp: ev_frag.fw_stamp_pointer,
                            stamp_value: ev_frag.value.next(),
                            unk_18: 0,
                            scene: scene.weak_pointer(),
                            buffer: scene.weak_buffer_pointer(),
                            #[ver(V < V15_5)]
                            unk_2c: U64(1),
                            #[ver(V >= V15_5)]
                            unk_2c: 1,
                            stats,
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            busy_flag: inner_weak_ptr!(ptr, busy_flag),
                            work_queue: ev_frag.info_ptr,
                            work_item: ptr,
                            vm_slot: vm_bind.slot(),
                            #[ver(V < V15_5)]
                            unk_60: 0,
                            unk_758_flag: inner_weak_ptr!(ptr, unk_758_flag),
                            #[ver(V >= V13_3 && V < V15_5)]
                            unk_6c_0: U64(0),
                            #[ver(V < V15_5)]
                            unk_6c: U64(0),
                            #[ver(V < V15_5)]
                            unk_74: U64(0),
                            #[ver(V < V15_5)]
                            unk_7c: U64(0),
                            #[ver(V < V15_5)]
                            unk_84: U64(0),
                            #[ver(V < V15_5)]
                            unk_8c: U64(0),
                            #[ver(G == G14 && V < V13_0B4)]
                            unk_8c_g14: U64(0),
                            #[ver(V < V15_5)]
                            restart_branch_offset: off,
                            #[ver(V < V15_5)]
                            has_attachments: (fragment_attachments.count > 0) as u32,
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            unk_9c: Default::default(),
                            #[ver(V >= V15_5)]
                            unk_64: 0,
                            #[ver(V >= V15_5)]
                            unk_68: U64(0),
                            #[ver(V >= V15_5)]
                            unk_70: Default::default(),
                            #[ver(V >= V15_5)]
                            pool_ptr_ptr: if *crate::module_parameters::uma_apple.value() != 0 {
                                // SAFETY: uma_pool is 0x100 bytes; +0xa0 holds its own address.
                                unsafe { scene.uma_pool_pointer().offset::<U64>(0xa0, core::ptr::null()) }
                            } else {
                                inner_weak_ptr!(ptr, pool_self)
                            },
                            #[ver(V >= V15_5)]
                            restart_branch_offset: off,
                            #[ver(V >= V15_5)]
                            has_attachments: (fragment_attachments.count > 0) as u8,
                            #[ver(V >= V15_5)]
                            unk_9d_ptr: inner_weak_ptr!(ptr, unk_buf_0),
                            #[ver(V >= V15_5)]
                            unk_a5: Default::default(),
                        })?;

                        // DIAGNOSTIC: the fragment job is read by the firmware
                        // (`fw rptr 1`) and never acked (`fw ack 0`) while the
                        // firmware stays alive (`halted 0`). If the ISP never
                        // reports idle, this terminal wait is where it sits.
                        // frag_end_mode=1 omits it. LOG.md 32.41.202
                        #[ver(V >= V15_5)]
                        if *crate::module_parameters::frag_end_mode.value() == 0 {
                            builder.add(microseq::WaitForIdle2 {
                                header: microseq::op::WaitForIdle2::HEADER_END_V15_5,
                            })?;
                        }

                        #[ver(V < V15_5)]
                        builder.add(microseq::RetireStamp::ver {
                            #[ver(V >= V15_5)]
                            header: microseq::op::RetireStamp::HEADER_V15_5,
                            #[ver(V < V15_5)]
                            header: microseq::op::RetireStamp::HEADER,
                            #[ver(V >= V15_5)]
                            unk_4: U64(0),
                            #[ver(V >= V15_5)]
                            ts_c: inner_weak_ptr!(ptr, unk_buf_0),
                            #[ver(V >= V15_5)]
                            unk_14: Default::default(),
                            #[ver(V >= V15_5)]
                            ts_24: inner_weak_ptr!(ptr, unk_buf_8),
                            #[ver(V >= V15_5)]
                            ts_2c: inner_weak_ptr!(ptr, unk_buf_10),
                            #[ver(V >= V15_5)]
                            ts_34: inner_weak_ptr!(ptr, command_time),
                        })?;

                        builder.build(private)?
                    },
                    notifier,
                    scene,
                    vm_bind,
                    aux_fb: self.ualloc.lock().array_empty_tagged(0x8000, b"AXFB")?,
                    timestamps,
                    user_timestamps: frg_user_timestamps,
                })
            },
            |inner, _ptr| {
                let vm_slot = vm_bind.slot();
                let aux_fb_info = fw::fragment::raw::AuxFBInfo::ver {
                    isp_ctl: isp_ctl,
                    unk2: 0,
                    width: cmdbuf.width_px as u32,
                    height: cmdbuf.height_px as u32,
                    #[ver(V >= V13_0B4)]
                    unk3: U64(0x100000),
                };

                try_init!(fw::fragment::raw::RunFragment::ver {
                    tag: fw::workqueue::CommandType::RunFragment,
                    #[ver(V >= V13_0B4)]
                    counter: U64(count_frag),
                    vm_slot,
                    unk_8: 0,
                    microsequence: inner.micro_seq.gpu_pointer(),
                    microsequence_size: inner.micro_seq.len() as u32,
                    notifier: inner.notifier.gpu_pointer(),
                    buffer: inner.scene.buffer_pointer(),
                    scene: inner.scene.gpu_pointer(),
                    unk_buffer_buf: inner.scene.kernel_buffer_pointer(),
                    tvb_tilemap: inner.scene.tvb_tilemap_pointer(),
                    ppp_multisamplectl: U64(cmdbuf.ppp_multisamplectl),
                    samples: cmdbuf.samples as u32,
                    tiles_per_mtile_y: tile_info.tiles_per_mtile_y as u16,
                    tiles_per_mtile_x: tile_info.tiles_per_mtile_x as u16,
                    unk_50: U64(0),
                    unk_58: U64(0),
                    isp_merge_upper_x: F32::from_bits(cmdbuf.isp_merge_upper_x),
                    isp_merge_upper_y: F32::from_bits(cmdbuf.isp_merge_upper_y),
                    unk_68: U64(0),
                    tile_count: U64(tile_info.tiles as u64),
                    #[ver(G < G14X)]
                    job_params1 <- try_init!(fw::fragment::raw::JobParameters1::ver {
                        utile_config,
                        unk_4: 0,
                        bg: fw::fragment::raw::BackgroundProgram {
                            rsrc_spec: U64(cmdbuf.bg.rsrc_spec as u64),
                            address: U64(usc(cmdbuf.bg.usc as u64)),
                        },
                        ppp_multisamplectl: U64(cmdbuf.ppp_multisamplectl),
                        isp_scissor_base: U64(cmdbuf.isp_scissor_base),
                        isp_dbias_base: U64(cmdbuf.isp_dbias_base),
                        isp_oclqry_base: U64(cmdbuf.isp_oclqry_base),
                        aux_fb_info,
                        isp_zls_pixels: U64(cmdbuf.isp_zls_pixels as u64),
                        zls_ctrl: U64(cmdbuf.zls_ctrl),
                        #[ver(G >= G14)]
                        unk_58_g14_0: U64(g14_unk),
                        #[ver(G >= G14)]
                        unk_58_g14_8: U64(0),
                        z_load: U64(cmdbuf.depth.base),
                        z_store: U64(cmdbuf.depth.base),
                        s_load: U64(cmdbuf.stencil.base),
                        s_store: U64(cmdbuf.stencil.base),
                        #[ver(G >= G14)]
                        unk_68_g14_0: Default::default(),
                        z_load_stride: U64(cmdbuf.depth.stride as u64),
                        z_store_stride: U64(cmdbuf.depth.stride as u64),
                        s_load_stride: U64(cmdbuf.stencil.stride as u64),
                        s_store_stride: U64(cmdbuf.stencil.stride as u64),
                        z_load_comp: U64(cmdbuf.depth.comp_base),
                        z_load_comp_stride: U64(cmdbuf.depth.comp_stride as u64),
                        z_store_comp: U64(cmdbuf.depth.comp_base),
                        z_store_comp_stride: U64(cmdbuf.depth.comp_stride as u64),
                        s_load_comp: U64(cmdbuf.stencil.comp_base),
                        s_load_comp_stride: U64(cmdbuf.stencil.comp_stride as u64),
                        s_store_comp: U64(cmdbuf.stencil.comp_base),
                        s_store_comp_stride: U64(cmdbuf.stencil.comp_stride as u64),
                        tvb_tilemap: inner.scene.tvb_tilemap_pointer(),
                        tvb_layermeta: inner.scene.tvb_layermeta_pointer(),
                        mtile_stride_dwords: U64((4 * tile_info.params.rgn_size as u64) << 24),
                        tvb_heapmeta: inner.scene.tvb_heapmeta_pointer(),
                        tile_config: U64(tile_config),
                        aux_fb: inner.aux_fb.gpu_pointer(),
                        unk_108: Default::default(),
                        // Reg 0x10069 is inert on G15S: overriding it with
                        // 0x40000000 left the ISP still fetching BG/EOT at the
                        // bare USC-relative address. This struct field is the
                        // other candidate source for the base the ISP adds.
                        // LOG.md 32.41.105.
                        usc_exec_base_isp: U64(
                            match *crate::module_parameters::usc_base_isp_struct.value() {
                                0 => self.usc_exec_base,
                                v => v,
                            },
                        ),
                        unk_140: U64(frg_unk_140),
                        helper_program: cmdbuf.fragment_helper.binary,
                        unk_14c: 0,
                        helper_arg: U64(cmdbuf.fragment_helper.data),
                        unk_158: U64(frg_unk_158),
                        unk_160: U64(0),
                        __pad: Default::default(),
                        #[ver(V < V13_0B4)]
                        __pad1: Default::default(),
                    }),
                    #[ver(G < G14X)]
                    job_params2 <- try_init!(fw::fragment::raw::JobParameters2 {
                        eot_rsrc_spec: cmdbuf.eot.rsrc_spec,
                        eot_usc: cmdbuf.eot.usc,
                        unk_8: 0x0,
                        unk_c: 0x0,
                        isp_merge_upper_x: F32::from_bits(cmdbuf.isp_merge_upper_x),
                        isp_merge_upper_y: F32::from_bits(cmdbuf.isp_merge_upper_y),
                        unk_18: U64(0x0),
                        utiles_per_mtile_y: tile_info.utiles_per_mtile_y as u16,
                        utiles_per_mtile_x: tile_info.utiles_per_mtile_x as u16,
                        unk_24: 0x0,
                        tile_counts: ((tile_info.tiles_y - 1) << 12) | (tile_info.tiles_x - 1),
                        tib_blocks: blocks_per_utile,
                        isp_bgobjdepth: cmdbuf.isp_bgobjdepth,
                        // TODO: does this flag need to be exposed to userspace?
                        isp_bgobjvals: load_bgobjvals as u32,
                        unk_38: 0x0,
                        unk_3c: 0x1,
                        helper_cfg: cmdbuf.fragment_helper.cfg,
                        __pad: Default::default(),
                    }),
                    #[ver(V >= V15_5)]
                    pad_regs_15_5: Default::default(),
                    #[ver(G >= G14X)]
                    registers: fw::job::raw::RegisterArray::new(
                        inner_weak_ptr!(_ptr, registers.registers),
                        |r| {
                            let g15_regs =
                                *crate::module_parameters::g15_frag_regs.value() != 0;
                            r.add(0x1739, 1);
                            r.add(0x10009, utile_config.into());
                            // 🔑 Apple's CLEAR-only pass binds no background
                            // program: capture2/priority0 has 0x15369 = 0x15371
                            // = 0x15379 = 0 at all six render-target sizes,
                            // with only 0x15381 (EOT) set, and RESULTS.md
                            // FINDING 5 says in as many words that "zero for a
                            // clear is correct, not missing".  Mesa instead
                            // encodes a clear the way Apple encodes a DRAW --
                            // our values match `draw-triangle-64x64` exactly on
                            // every register that separates the two.  This
                            // reproduces Apple's configuration so the ISP can
                            // be asked for the simplest pass it supports: a
                            // store with nothing to fetch.  LOG.md 32.41.410.
                            let store_only =
                                *crate::module_parameters::frag_store_only.value() != 0;
                            r.add(0x15379, if store_only { 0u64 } else { cmdbuf.eot.rsrc_spec.into() });
                            r.add(0x15381, usc(cmdbuf.eot.usc.into()));
                            r.add(0x15369, if store_only { 0u64 } else { cmdbuf.bg.rsrc_spec.into() });
                            r.add(0x15371, if store_only { 0u64 } else { usc(cmdbuf.bg.usc.into()) });
                            r.add(0x15131, cmdbuf.isp_merge_upper_x.into());
                            r.add(0x15139, cmdbuf.isp_merge_upper_y.into());
                            r.add(0x100a1, 0);
                            r.add(0x15069, 0);
                            r.add(0x15071, 0); // pointer
                            r.add(0x16058, 0);
                            r.add(0x10019, cmdbuf.ppp_multisamplectl);
                            let isp_mtile_size = (tile_info.utiles_per_mtile_y
                                | (tile_info.utiles_per_mtile_x << 16))
                                .into();
                            r.add(0x100b1, isp_mtile_size); // ISP_MTILE_SIZE
                            r.add(0x16030, isp_mtile_size); // ISP_MTILE_SIZE
                            r.add(
                                0x100d9,
                                (((tile_info.tiles_y - 1) << 12) | (tile_info.tiles_x - 1)).into(),
                            ); // TE_SCREEN
                            // 🔑 macOS position 16.  `generateRegisterListFor3D`
                            // fffffe0009111280 builds this as
                            //     x8 = desc[0x741] ? 0x00ff0004 : 0x00ff0200
                            //     x4 = x8 | ((desc[0x9e8] != 0) << 24)
                            //     if desc[0x9e8] { x4 = (x4 & 0x00100204)
                            //                          | (desc[0xa08] << 16) }
                            // Both descriptor fields are zero for a plain
                            // single-attachment pass, and the live DTrace
                            // capture reads back 0x00ff0200 at every render
                            // target size from 64x64 to 1920x1080.
                            // capture2/priority0, LOG.md 32.41.215.
                            if g15_regs {
                                r.add(0x10791, 0x00ff_0200);
                            }
                            r.add(0x16098, va(inner.scene.tvb_heapmeta_pointer().into()) + *crate::module_parameters::frag_scene_bias.value());
                            r.add(0x15109, va(cmdbuf.isp_scissor_base)); // ISP_SCISSOR_BASE
                            r.add(0x15101, va(cmdbuf.isp_dbias_base)); // ISP_DBIAS_BASE
                            r.add(0x15021, isp_ctl.into()); // aux_fb_info.unk_1
                            r.add(
                                0x15211,
                                ((cmdbuf.height_px as u64) << 32) | cmdbuf.width_px as u64,
                            ); // aux_fb_info.{width, heigh
                            r.add(0x15049, if store_only { 0u64 } else { 0x100000 }); // s2.aux_fb_info.unk3
                            r.add(0x10051, if store_only { 0u64 } else { blocks_per_utile.into() }); // s1.unk_2c
                            r.add(0x15321, cmdbuf.isp_zls_pixels.into()); // ISP_ZLS_PIXELS
                            r.add(0x15301, cmdbuf.isp_bgobjdepth.into()); // ISP_BGOBJDEPTH
                            r.add(0x15309, if store_only { 0u64 } else { load_bgobjvals }); // ISP_BGOBJVALS
                            r.add(0x15311, va(cmdbuf.isp_oclqry_base)); // ISP_OCLQRY_BASE
                            r.add(0x15319, cmdbuf.zls_ctrl); // ISP_ZLSCTL
                            r.add(0x15349, if store_only { 0u64 } else { g14_unk }); // s2.unk_58_g14_0
                            r.add(0x15351, 0); // s2.unk_58_g14_8
                            r.add(0x15329, va(cmdbuf.depth.base)); // ISP_ZLOAD_BASE
                            r.add(0x15331, va(cmdbuf.depth.base)); // ISP_ZSTORE_BASE
                            r.add(0x15339, va(cmdbuf.stencil.base)); // ISP_STENCIL_LOAD_BASE
                            r.add(0x15341, va(cmdbuf.stencil.base)); // ISP_STENCIL_STORE_BASE
                            r.add(0x15231, 0);
                            r.add(0x15221, 0);
                            r.add(0x15239, 0);
                            r.add(0x15229, 0);
                            r.add(0x15401, cmdbuf.depth.stride as u64); // load
                            r.add(0x15421, cmdbuf.depth.stride as u64); // store
                            r.add(0x15409, cmdbuf.stencil.stride as u64); // load
                            r.add(0x15429, cmdbuf.stencil.stride as u64);
                            r.add(0x153c1, va(cmdbuf.depth.comp_base)); // load
                            r.add(0x15411, cmdbuf.depth.comp_stride as u64); // load
                            r.add(0x153c9, va(cmdbuf.depth.comp_base)); // store
                            r.add(0x15431, cmdbuf.depth.comp_stride as u64); // store
                            r.add(0x153d1, va(cmdbuf.stencil.comp_base)); // load
                            r.add(0x15419, cmdbuf.stencil.comp_stride as u64); // load
                            r.add(0x153d9, va(cmdbuf.stencil.comp_base)); // store
                            r.add(0x15439, cmdbuf.stencil.comp_stride as u64); // store
                            r.add(0x16429, match *crate::module_parameters::frag_tmap_poison.value() {
                                0 => va(inner.scene.tvb_tilemap_pointer().into()) + *crate::module_parameters::frag_scene_bias.value(),
                                v => v,
                            });
                            r.add(0x16060, va(inner.scene.tvb_layermeta_pointer().into()) + *crate::module_parameters::frag_scene_bias.value());
                            r.add(0x16431, (4 * tile_info.params.rgn_size as u64) << 24); // ISP_RGN?
                            // 32.41.415: PROCESS_EMPTY_TILES is a SEPARATE flag from
                            // the background-program removal. A clear has no geometry,
                            // so clearing it leaves the ISP with zero tiles to process
                            // and the unit idles having done nothing -- which is an
                            // alternative explanation for 32.41.412 and must be
                            // separable from it.
                            let keep_empty_tiles =
                                *crate::module_parameters::frag_keep_empty_tiles.value() != 0;
                            r.add(
                                0x10039,
                                if store_only && !keep_empty_tiles {
                                    tile_config & !0x10000
                                } else {
                                    tile_config
                                },
                            ); // tile_config ISP_CTL?
                            // macOS emits 0x16020 between 0x10039 and 0x16451;
                            // this driver had it eleven entries later.
                            if g15_regs {
                                r.add(0x16020, 0);
                            }
                            r.add(0x16451, 0x0); // ISP_RENDER_ORIGIN
                            // The fragment helper program is not in macOS's
                            // G15S list at all -- neither for a clear nor for a
                            // draw, at any render target size.  Keep the
                            // G13/G14X behaviour available for ablation.
                            if !g15_regs {
                                r.add(0x11821, va(cmdbuf.fragment_helper.binary.into()));
                                r.add(0x11829, va(cmdbuf.fragment_helper.data));
                                r.add(0x11f79, cmdbuf.fragment_helper.cfg.into());
                            }
                            r.add(0x15359, 0);
                            // 🔑 macOS position 58, from fffffe00091123c4:
                            //     x4 = desc[0x86a] ? 0x8ae0 : 0x8860
                            //     if cfg[0x4d4] > 4 || cfg[0x691] & 1 { x4 |= 0x1c }
                            // Measured 0x8860 on this exact part.
                            if g15_regs {
                                r.add(0x100b8, 0x8860);
                            }
                            // The ISP fetches BG/EOT programs at the *bare*
                            // USC-relative address -- `0xfff68200` for an
                            // `eot.usc` of `0xfff68204` -- while the program
                            // itself is demonstrably at
                            // `usc_exec_base + usc` (probed: mapped, with real
                            // shader words in it). So this base is contributing
                            // nothing. Overriding it with a small non-zero
                            // value says which: if the fault address moves by
                            // that amount the register works and 0x1100000000
                            // is being truncated to 32 bits; if it does not
                            // move, this register number is inert on G15S.
                            // LOG.md 32.41.105.
                            let usc_base_isp = match *crate::module_parameters::usc_base_isp.value()
                            {
                                0 => va(self.usc_exec_base),
                                v => v,
                            };
                            // 0x10069 does not appear anywhere in macOS's G15S
                            // fragment list -- not zero, absent -- which agrees
                            // with the hardware result that it is inert here.
                            if !g15_regs {
                                r.add(0x10069, usc_base_isp); // frag; USC_EXEC_BASE_ISP
                                r.add(0x16020, 0);
                            }
                            r.add(0x16461, va(inner.aux_fb.gpu_pointer().into()));
                            r.add(0x16090, va(inner.aux_fb.gpu_pointer().into()));
                            r.add(0x120a1, frg_unk_158);
                            // 0x101e9 carries the same value as 0x120a1; macOS
                            // computes both as `x24 & 0x1f` from the same
                            // register (fffffe0009112560 / fffffe000911261c).
                            if g15_regs {
                                r.add(0x101e9, frg_unk_158);
                            }
                            r.add(0x160a8, 0);
                            r.add(0x16068, frg_tilecfg);
                            r.add(0x160b8, 0x0);
                            // 🔑 The sixteen-register tail macOS programs after
                            // 0x160b8 and this driver never has.  Every value
                            // below is what the live DTrace capture read back
                            // on this exact part, cross-checked against the
                            // emitting code in `generateRegisterListFor3D`:
                            //
                            //   0x1a0a9..0x1a0e1  desc[0x9b8..0x9e0], all zero
                            //   0x101c1  (desc[0x9e8] & ~0x1f0000)
                            //              | (desc[0x9f0] << 16)        -> 0
                            //   0x0d469  desc[0x9f8]                    -> 0
                            //   0x1a0f9  desc[0xa0c] & ~7               -> 8
                            //   0x1c838  desc[0x808]                    -> 9
                            //   0x1ca28  *(desc[0x820] + 0x28) & ~0xf   -> 0x2c0060
                            //   0x0a5a9  constant when desc[0x738] == 0
                            //   0x0d429  a pair of hardware-config getters
                            //              (vtable +0x1018, +0x1020)
                            //
                            // The last three (0x160e0/0x1499/0xa341) all take
                            // `cmd[0xbcc] | (desc[0x774] << 32)`.  Both halves
                            // are telemetry ids copied straight out of the
                            // userspace IOGPUCommandDescriptor by
                            // AGX3DChannelSKU::submitBuffer, they are the only
                            // three values that differ between otherwise
                            // identical runs, and the tiler stage carries the
                            // same construction (`cmdTA[0x888] | ...`), so they
                            // are annotations rather than state: zero here.
                            // capture2/priority0/RESULTS.md, LOG.md 32.41.215.
                            if g15_regs {
                                r.add(0x1a0a9, 0);
                                r.add(0x1a0b1, 0);
                                r.add(0x1a079, 0);
                                r.add(0x1a081, 0);
                                r.add(0x1a0d9, 0);
                                r.add(0x1a0e1, 0);
                                r.add(0x101c1, 0);
                                r.add(0x0d469, 0);
                                r.add(0x1a0f9, 8);
                                // 32.41.666: 0x1c838 = desc[0x808] = the hardware
                                // buffer id (PB slot) the 3D side returns TVB pages
                                // to. Apple's capture had slot 9; a constant 9 made
                                // every fragment pass free into an unused slot, so
                                // our slot drained frame by frame -> TA OOM.
                                r.add(
                                    0x1c838,
                                    if *crate::module_parameters::pb_slot_regs.value() != 0 {
                                        inner.scene.slot() as u64
                                    } else {
                                        9
                                    },
                                );
                                // ⛔ 0x1ca28 is NOT copyable.  Programming
                                // macOS's literal 0x2c0060 made the Pixel
                                // Master read `0x10002c0040` = the code-GART
                                // base 0x1000000000 (AGXGart::getCodeGartRangeInfo,
                                // gart_ranges[1]) plus 0x2c0060 - 0x20, and
                                // fault there.  So the register is an address
                                // *relative to a hardware base of
                                // 0x1000000000*, and Apple's value names a
                                // buffer in Apple's own heap.  This driver's
                                // buffers are nowhere near that range, and the
                                // firmware appends its own 0x1ca28 from
                                // StartFragment anyway (LOG.md 32.41.48), so
                                // leave it to the firmware.  LOG.md 32.41.215.
                                //
                                // 32.41.602 correction: the firmware appends it
                                // only if StartFragment op+0x8c != 0, and Apple's
                                // is 0 too -- Apple's KEXT lists it. The +2^36 in
                                // the old fault is the usual hardware VA bias, so
                                // the value is OUR user_buffer in the TA's 0x1ca30
                                // form, not Apple's literal.
                                if *crate::module_parameters::frag_1ca28.value() != 0 {
                                    r.add(0x1ca28, va(u64::from(inner.scene.user_buffer_pointer())));
                                }
                                r.add(0x0a5a9, 0x0000_0060_0040_0020);
                                r.add(0x0d429, 0x0000_0360_0000_0120);
                                r.add(0x160e0, 0);
                                r.add(0x01499, 0);
                                r.add(0x0a341, 0);
                            }
                            /*
                            r.add(0x10201, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x10428, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x1c838, 1);  // ?
                            r.add(0x1ca28, 0x1502960f00); // ??
                            r.add(0x1731, 0x1); // ??
                            */
                            // The fragment half of the same list, from
                            // AGX3DChannelG15::generateRegisterListFor3D.  The
                            // two sets are parallel -- each data master gets its
                            // own register out of the same families -- which is
                            // why they read as real 15.5/G15 additions rather
                            // than noise.  LOG.md 32.41.56.
                            let probe = *crate::module_parameters::missing_regs_probe.value();
                            if probe != 0 {
                                for (i, reg) in [
                                    0x1498, 0xa340, 0xa5a8, 0xd428, 0xd468, 0x100b8,
                                    0x101c0, 0x160e0, 0x1a078, 0x1a080, 0x1a0a8, 0x1a0b0,
                                    0x1a0d8, 0x1a0e0, 0x1c838, 0x1ca28,
                                    0x14308, 0x14048, 0x14028, 0x14318, 0x14080, 0x14070,
                                ]
                                .iter()
                                .enumerate()
                                {
                                    r.add(*reg | (probe as u32 & 1), (probe & !0xfff) | (((i + 0x20) as u64) << 12));
                                }
                            }
                            // 32.41.602: under apple_replay the fragment pipeline
                            // pointers must be Apple's (they point into Apple's
                            // replayed buffers); Mesa's (0x11fff68104...) made the
                            // USC fetch through a null pointer (UL1C fault at 0).
                            // Values from capture2 registers/draw-triangle-64x64.
                            if *crate::module_parameters::apple_replay.value() != 0
                                && *crate::module_parameters::frag_apple_regs.value() != 0
                            {
                                for (reg, val) in [
                                    (0x15369u32, 0xffff800000000040u64),
                                    (0x15371, 0x10000138240),
                                    (0x15379, 0),
                                    (0x15381, 0x10000138480),
                                    (0x15101, 0x100002a0000),
                                    (0x15109, 0x10000148000),
                                    (0x16090, 0x10000250000),
                                    (0x16461, 0x10000250000),
                                    (0x15021, 0xc001),
                                    (0x15301, 0x3f800000),
                                ] {
                                    if !r.set(reg, val) {
                                        pr_info!("asahi/fragapple: {:#x} absent\n", reg);
                                    }
                                }
                                // =2: BG program unbound, as in Apple's clear-only
                                // passes, to split BG from EOT. LOG.md 32.41.603.
                                if *crate::module_parameters::frag_apple_regs.value() == 2 {
                                    r.set(0x15369, 0);
                                    r.set(0x15371, 0);
                                }
                                // =3: 0x15369's low 42 bits are VA 0x40 -- Apple's VA-0
                                // buffer, which the replay maps at +2^36 (VA 0 is below
                                // the allocator). Rebase it there. LOG.md 32.41.604.
                                if *crate::module_parameters::frag_apple_regs.value() == 3 {
                                    r.set(0x15369, 0xffff800000000040u64 + (1u64 << 36));
                                }
                                pr_info!("asahi/fragapple: applied (mode {})\n",
                                    *crate::module_parameters::frag_apple_regs.value());
                            }
                            if debug_enabled(DebugFlags::Render) {
                                r.dump("frg");
                            }
                        }
                    ),
                    job_params3 <- try_init!(fw::fragment::raw::JobParameters3::ver {
                        isp_dbias_base: fw::fragment::raw::ArrayAddr {
                            ptr: U64(cmdbuf.isp_dbias_base),
                            unk_padding: U64(0),
                        },
                        isp_scissor_base: fw::fragment::raw::ArrayAddr {
                            ptr: U64(cmdbuf.isp_scissor_base),
                            unk_padding: U64(0),
                        },
                        isp_oclqry_base: U64(cmdbuf.isp_oclqry_base),
                        unk_118: U64(0x0),
                        unk_120: Default::default(),
                        unk_partial_bg: fw::fragment::raw::BackgroundProgram {
                            rsrc_spec: U64(cmdbuf.partial_bg.rsrc_spec as u64),
                            address: U64(usc(cmdbuf.partial_bg.usc as u64)),
                        },
                        unk_258: U64(0),
                        unk_260: U64(0),
                        unk_268: U64(0),
                        unk_270: U64(0),
                        partial_bg: fw::fragment::raw::BackgroundProgram {
                            rsrc_spec: U64(cmdbuf.partial_bg.rsrc_spec as u64),
                            address: U64(usc(cmdbuf.partial_bg.usc as u64)),
                        },
                        zls_ctrl: U64(reload_zlsctrl),
                        unk_290: U64(g14_unk),
                        z_load: U64(cmdbuf.depth.base),
                        z_partial_stride: U64(cmdbuf.depth.stride as u64),
                        z_partial_comp_stride: U64(cmdbuf.depth.comp_stride as u64),
                        z_store: U64(cmdbuf.depth.base),
                        z_partial: U64(cmdbuf.depth.base),
                        z_partial_comp: U64(cmdbuf.depth.comp_base),
                        s_load: U64(cmdbuf.stencil.base),
                        s_partial_stride: U64(cmdbuf.stencil.stride as u64),
                        s_partial_comp_stride: U64(cmdbuf.stencil.comp_stride as u64),
                        s_store: U64(cmdbuf.stencil.base),
                        s_partial: U64(cmdbuf.stencil.base),
                        s_partial_comp: U64(cmdbuf.stencil.comp_base),
                        unk_2f8: Default::default(),
                        tib_blocks: blocks_per_utile,
                        unk_30c: 0x0,
                        aux_fb_info,
                        tile_config: U64(tile_config),
                        unk_328_padding: Default::default(),
                        unk_partial_eot: fw::fragment::raw::EotProgram::new(
                            cmdbuf.partial_eot.rsrc_spec,
                            cmdbuf.partial_eot.usc
                        ),
                        partial_eot: fw::fragment::raw::EotProgram::new(
                            cmdbuf.partial_eot.rsrc_spec,
                            cmdbuf.partial_eot.usc
                        ),
                        isp_bgobjdepth: cmdbuf.isp_bgobjdepth,
                        isp_bgobjvals: cmdbuf.isp_bgobjvals,
                        sample_size: cmdbuf.sample_size_B as u32,
                        unk_37c: 0x0,
                        unk_380: U64(0x0),
                        unk_388: U64(0x0),
                        #[ver(V >= V13_0B4)]
                        unk_390_0: U64(0x0),
                        isp_zls_pixels: U64(cmdbuf.isp_zls_pixels as u64),
                    }),
                    unk_758_flag: 0,
                    unk_75c_flag: 0,
                    unk_buf: Default::default(),
                    busy_flag: 0,
                    tvb_overflow_count: 0,
                    unk_878: 0,
                    encoder_params <- try_init!(fw::job::raw::EncoderParams {
                        // Maybe set when reloading z/s?
                        unk_8: 0,
                        sync_grow: 0,
                        unk_10: 0x0, // fixed
                        encoder_id: 0,
                        unk_18: 0x0, // fixed
                        unk_mask: 0xffffffffu32,
                        sampler_array: U64(sampler_heap),
                        sampler_count: cmdbuf.sampler_count as u32,
                        sampler_max: (cmdbuf.sampler_count as u32) + 1,
                    }),
                    process_empty_tiles: (cmdbuf.flags
                        & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES as u32
                        != 0) as u32,
                    // TODO: needs to be investigated
                    no_clear_pipeline_textures: 1,
                    // TODO: needs to be investigated
                    msaa_zs: 0,
                    unk_pointee: 0,
                    #[ver(V >= V13_3)]
                    unk_v13_3: 0,
                    meta <- try_init!(fw::job::raw::JobMeta::ver {
                        unk_0: 0,
                        unk_2: 0,
                        no_preemption: no_preemption as u8,
                        #[ver(V >= V15_5)]
                        unk_4: Default::default(),
                        stamp: ev_frag.stamp_pointer,
                        fw_stamp: ev_frag.fw_stamp_pointer,
                        stamp_value: ev_frag.value.next(),
                        stamp_slot: ev_frag.slot,
                        #[ver(V < V15_5)]
                        evctl_index: 0, // fixed
                        #[ver(V < V15_5)]
                        flush_stamps: flush_stamps as u32,
                        #[ver(V < V15_5)]
                        uuid: uuid_3d,
                        // 15.5: the firmware uses this word as the workqueue's slot in
                        // the notifier's 4-entry waiter table (pstamp_*[idx], bit idx of
                        // has_*; fw:0x1f45c). Apple = 1 << AGXCLWorkQueue+0xe0, a small
                        // per-queue index; we have one workqueue per pipe per notifier, so
                        // 0. The running event_seq overflowed slot 3 on the 5th job and
                        // wrote into in_list/list_head (fw crash 0x1f900). LOG.md 32.41.664
                        #[ver(V >= V15_5)]
                        event_seq: 0,
                        #[ver(V < V15_5)]
                        event_seq: ev_frag.event_seq as u32,
                    }),
                    unk_after_meta: unk1.into(),
                    unk_buf_0: U64(0),
                    unk_buf_8: U64(0),
                    #[ver(G < G14X)]
                    unk_buf_10: U64(1),
                    #[ver(G >= G14X)]
                    unk_buf_10: U64(0),
                    command_time: U64(0),
                    timestamp_pointers <- try_init!(fw::job::raw::TimestampPointers {
                        start_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), frag.start)),
                        end_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), frag.end)),
                    }),
                    user_timestamp_pointers: inner.user_timestamps.pointers()?,
                    client_sequence: slot_client_seq,
                    pad_925: Default::default(),
                    unk_928: 0,
                    unk_92c: 0,
                    #[ver(V >= V13_0B4)]
                    unk_ts: U64(0),
                    #[ver(V >= V13_0B4)]
                    #[ver(V < V15_5)]
                    unk_92d_8: Default::default(),
                    // The command-side UMA pool the 15.5 fragment start routine
                    // dereferences, pointed at the same object StartFragment's
                    // `pool_ptr` registers.  LOG.md 32.41.33.
                    #[ver(V >= V15_5)]
                    unk_92d_8: Default::default(),
                    // 32.41.598: with uma_apple both halves share the scene's pool record, as
                    // Apple's do; the inline copy's +0x48 is null (cf. compute, .597).
                    #[ver(V >= V15_5)]
                    uma_pool: if *crate::module_parameters::uma_apple.value() != 0 {
                        U64(inner.scene.uma_pool_pointer().into())
                    } else {
                        U64(inner_weak_ptr!(_ptr, pool).into())
                    },
                    #[ver(V >= V15_5)]
                    uma_pool_flush: *crate::module_parameters::uma_pool_flush.value() as u8,
                    #[ver(V >= V15_5)]
                    unk_c2b: Default::default(),
                    // The pool the 15.5 fragment microsequence registers, plus
                    // the word holding its address that FinalizeFragment's
                    // 0x90 field points at.  LOG.md 32.41.26.
                    #[ver(V >= V15_5)]
                    pool_self: if *crate::module_parameters::uma_apple.value() != 0 {
                        U64(inner.scene.uma_pool_pointer().into())
                    } else {
                        U64(inner_weak_ptr!(_ptr, pool).into())
                    },
                    #[ver(V >= V15_5)]
                    // RunFragment carries its own inline pool -- the vertex
                    // side uses the scene's -- and it was zeroed too, so its
                    // aperture was also based at VA 0. Same descriptor layout:
                    // the base lives at pool+0x14, stored shifted right by 7.
                    // LOG.md 32.41.52.
                    pool: {
                        let mut p = [0u8; 0x90];
                        if *crate::module_parameters::uma_aperture_size.value() != 0 {
                            crate::uma_pool_init(
                                &mut p,
                                uma_aperture_base,
                                *crate::module_parameters::uma_aperture_size.value() as u64,
                                uma_sync_base,
                            );
                        }
                        Array::new(p)
                    },
                })
            },
        )?;

        // 32.41.602: dump the fragment register list for a diff against Apple's
        // (capture2 priority0/fragment-registers-clear.txt, 82 entries).
        #[ver(G >= G14X)]
        if *crate::module_parameters::dump_vtx_cmd.value() != 0 {
            frag.with(|raw, _| {
                crate::gpu::DBG_FRAGCMD.store(
                    raw as *const _ as u64,
                    core::sync::atomic::Ordering::Relaxed,
                );
                crate::gpu::DBG_FRAGCMD_SIZE.store(
                    core::mem::size_of_val(raw) as u64,
                    core::sync::atomic::Ordering::Relaxed,
                );
                crate::gpu::dump_fragcmd("PRE");
                crate::gpu::DBG_FRAGREGS.store(
                    &raw.registers as *const _ as u64,
                    core::sync::atomic::Ordering::Relaxed,
                );
                let r = &raw.registers;
                let n = r.count as usize;
                pr_info!("asahi/fragregs count {}\n", n);
                for i in 0..n.min(128) {
                    let e = &r.registers[i];
                    let (num, val) = ({ e.number }, { e.value });
                    let v = val.0;
                    pr_info!("asahi/fragregs [{:3}] {:#07x} = {:#018x}\n", i, num, v);
                }
            });
        }

        // Map the firmware's notifier `pstamp_*` entries -- which are pointers
        // into these command objects -- back onto a struct offset. LOG.md
        // 32.41.91.
        crate::sub_dev_err!(
            self.dev.as_ref(),
            "asahi/cmd: frag {:#x} meta+{:#x} fw_stamp {:#x} stamp_value {:#x}\n",
            frag.gpu_va().get(),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, meta),
            u64::from(ev_frag.fw_stamp_pointer),
            ev_frag.value.next().counter(),
        );

        mod_dev_dbg!(self.dev, "[Submission {}] Add Frag\n", id);
        fence.add_command();

        // .667zzz44 `gpu_pass_prof`: per-pass GPU time
        let prof = {
            let c = cmdbuf;
            let i0 = u64::from(c.width_px & 0xffff)
                | (u64::from(c.height_px & 0xffff) << 16)
                | (u64::from(c.samples & 0xff) << 32)
                | (u64::from(c.layers & 0xff) << 40)
                | ((u64::from(c.flags) & 0xffff) << 48);
            let i1 = (c.zls_ctrl & 0xffff_ffff)
                | (u64::from(c.depth.base != 0) << 32)
                | (u64::from(c.stencil.base != 0) << 33)
                | (u64::from(vm_bind.slot()) << 40);
            let key = (i0.wrapping_mul(0x100_0000_01b3) ^ i1.rotate_left(17)) | 1;
            (key, i0, i1, crate::stats::now_ns())
        };
        let prof_ts = timestamps.clone();
        let prof_vts = timestamps.clone();
        frag_job.add_cb(frag, vm_bind.slot(), move |error| {
            if let Some(err) = error {
                fence.set_error(err.into());
            }
            {
                let ts = prof_ts.with(|raw, _| {
                    (raw.frag.start.load(Ordering::Relaxed), raw.frag.end.load(Ordering::Relaxed))
                });
                crate::stats::pass_prof_frag(prof.0, prof.1, prof.2, prof.3, ts);
            }
            // 32.41.613: dump while the scene is still alive (the HUNG report
            // runs after it is freed).
            if *crate::module_parameters::dump_vtx_cmd.value() != 0 {
                crate::replay::draw_rt_dump("DONE");
                crate::gpu::dump_scene_bufs("DONE");
                crate::gpu::dump_tvb("DONE");
            }

            fence.command_complete();
        })?;

        let fence = job.fence.clone();
        let vtx_job = job.get_vtx()?;

        if scene.rebind() || tvb_grown || tvb_autogrown {
            mod_dev_dbg!(self.dev, "[Submission {}] Create Bind Buffer\n", id);
            let bind_buffer = kalloc.private.new_init(
                {
                    let scene = scene.clone();
                    try_init!(fw::buffer::InitBuffer::ver { scene })
                },
                |inner, _ptr| {
                    let vm_slot = vm_bind.slot();
                    try_init!(fw::buffer::raw::InitBuffer::ver {
                        tag: fw::workqueue::CommandType::InitBuffer,
                        vm_slot,
                        buffer_slot: inner.scene.slot(),
                        unk_c: 0,
                        block_count: buffer.block_count(),
                        buffer: inner.scene.buffer_pointer(),
                        stamp_value: ev_vtx.value.next(),
                    })
                },
            )?;

            mod_dev_dbg!(self.dev, "[Submission {}] Add Bind Buffer\n", id);
            vtx_job.add(bind_buffer, vm_bind.slot())?;
        }

        mod_dev_dbg!(self.dev, "[Submission {}] Create Vertex\n", id);
        let vtx = GpuObject::new_init_prealloc(
            kalloc.gpu_ro.alloc_object()?,
            |ptr: GpuWeakPointer<fw::vertex::RunVertex::ver>| {
                let scene = scene.clone();
                let vm_bind = vm_bind.clone();
                let timestamps = timestamps.clone();
                let private = &mut kalloc.private;
                try_init!(fw::vertex::RunVertex::ver {
                    micro_seq: {
                        let mut builder = microseq::Builder::new();

                        let stats = inner_weak_ptr!(
                            gpu.initdata.runtime_pointers.stats.vtx.weak_pointer(),
                            stats
                        );

                        crate::gpu::DBG_STATS_VTX.store(
                            u64::from(gpu.initdata.runtime_pointers.stats.vtx.gpu_pointer()),
                            core::sync::atomic::Ordering::Relaxed,
                        );
                        crate::gpu::DBG_STATS_FRAG.store(
                            u64::from(gpu.initdata.runtime_pointers.stats.frag.gpu_pointer()),
                            core::sync::atomic::Ordering::Relaxed,
                        );
                        crate::gpu::DBG_STATS_COMP.store(
                            u64::from(gpu.initdata.runtime_pointers.stats.comp.gpu_pointer()),
                            core::sync::atomic::Ordering::Relaxed,
                        );

                        // Stash for the after-the-fact read in
                        // `handle_timeout` -- the probes below run before the
                        // kick and cannot see what the job did. LOG.md 32.41.137.
                        crate::gpu::DBG_SCENE_TILEMAP.store(
                            u64::from(scene.tvb_tilemap_pointer()),
                            core::sync::atomic::Ordering::Relaxed,
                        );
                        crate::gpu::DBG_SCENE_HEAPMETA.store(
                            u64::from(scene.tvb_heapmeta_pointer()),
                            core::sync::atomic::Ordering::Relaxed,
                        );
                        {
                            let (pass, total, blocks, pages) = scene.counter_ptrs();
                            crate::gpu::DBG_CNT_PASS
                                .store(pass, core::sync::atomic::Ordering::Relaxed);
                            crate::gpu::DBG_CNT_TOTAL
                                .store(total, core::sync::atomic::Ordering::Relaxed);
                            crate::gpu::DBG_CNT_BLOCKS
                                .store(blocks, core::sync::atomic::Ordering::Relaxed);
                            crate::gpu::DBG_CNT_PAGES
                                .store(pages, core::sync::atomic::Ordering::Relaxed);
                            crate::gpu::DBG_CNT_PMCT.store(
                                scene.pm_counter_ptr(),
                                core::sync::atomic::Ordering::Relaxed,
                            );
                        }
                        crate::gpu::DBG_SCENE_OBJ.store(
                            u64::from(scene.gpu_pointer()),
                            core::sync::atomic::Ordering::Relaxed,
                        );
                        crate::gpu::DBG_SCENE_INFO.store(
                            u64::from(scene.buffer_pointer()),
                            core::sync::atomic::Ordering::Relaxed,
                        );
                        crate::gpu::DBG_SCENE_SLOT.store(
                            vm_bind.slot() as u64,
                            core::sync::atomic::Ordering::Relaxed,
                        );
                        mod_dev_dbg!(
                            self.dev,
                            "asahi/addr vtx: vdm_ctrl {:#x} tilemap {:#x} tpc {:#x} heapmeta {:#x} usc_exec {:#x} layermeta {:#x} preempt1 {:#x} preempt2 {:#x}\n",
                            cmdbuf.vdm_ctrl_stream_base,
                            u64::from(scene.tvb_tilemap_pointer()),
                            u64::from(scene.tpc_pointer()),
                            u64::from(scene.tvb_heapmeta_pointer()),
                            self.usc_exec_base,
                            u64::from(scene.tvb_layermeta_pointer()),
                            u64::from(scene.preempt_buf_1_pointer()),
                            u64::from(scene.preempt_buf_2_pointer()),
                        );
                        // gUPM (the shader program-memory unit) faults reading
                        // VA 0 once the VDM is clear, so print the three
                        // shader-side addresses the driver passes through from
                        // Mesa untouched.  LOG.md 32.41.40.
                        mod_dev_dbg!(
                            self.dev,
                            "asahi/usc vtx: exec_base {:#x} helper_binary {:#x} helper_data {:#x} helper_cfg {:#x}\n",
                            self.usc_exec_base,
                            u64::from(cmdbuf.vertex_helper.binary),
                            cmdbuf.vertex_helper.data,
                            u64::from(cmdbuf.vertex_helper.cfg),
                        );
                        // Is Mesa's EOT/BG program where the driver's model of
                        // the USC address says it is? The UL1C fault names the
                        // *bare* USC-relative address (`0xfff68200` =
                        // `eot.usc & !7`), which says the ISP exec base was not
                        // applied -- but only if `exec_base + usc` is where the
                        // program actually lives. Probe both and let the page
                        // table answer. LOG.md 32.41.105.
                        // Standalone: the draw-CTRL correction is a fix, not a
                        // diagnostic, so it must not sit behind `probe_usc` --
                        // which is exactly why 32.41.252's two runs tested
                        // nothing.  LOG.md 32.41.253.
                        {
                            let slot = vm_bind.slot();
                            let cs = cmdbuf.vdm_ctrl_stream_base & !3;
                        // 🔑 Mesa emits a draw CTRL word that does not
                        // occur *anywhere* in Apple's G15 Metal driver:
                        // scanning `AGXMetalG15X_M1` for `0x61c00600` finds
                        // zero hits, for `0x61c40600` exactly one -- the
                        // entry in
                        // `VDMEncoderGen5<G15::ESLEncoder,...>::PrimitiveTypeToVDMCTRLTypeNonIndexedDraw`.
                        // Bit 18 is set in *every* entry of that table and
                        // never by Mesa, and the indexed table is
                        // `0x61e2xxxx` where Mesa emits `0x61e0xxxx`
                        // (bit 17).  Systematic, not per-primitive -- the
                        // shape of a record-format field.  LOG.md 32.41.200.
                        //
                        // That matters now for a reason it did not in
                        // 32.41.200: the CSD **walks** the stream
                        // (32.41.250), so a record whose length the decoder
                        // reads out of a format field it does not
                        // understand desyncs the walk, and a desynced walk
                        // is exactly "runs past the terminator to
                        // cs+0x80000".  Fix the words in place rather than
                        // with a fixed-offset poke, so it holds for any
                        // stream Mesa emits.  LOG.md 32.41.252.
                        // 🔬 Dump the control stream as the GPU sees it,
                        // after the draw-CTRL fix below has been applied.
                        //
                        // ⚠️ Placed HERE, at the same nesting as `ctrl_fix`, and
                        // deliberately NOT inside `if probe_usc != 0` -- that is
                        // where `cs_draw_ctrl_fix` sat as dead code for two runs
                        // (32.41.253) and where a first attempt at this dump sat
                        // for one more. The CSD walks the entire 512 KiB BO and
                        // faults 0x80 past its end (32.41.297), so it parses
                        // zeros as records; reading the stream back is the only
                        // way to check its records against Mesa's intent.
                        // LOG.md 32.41.298.
                        let cs_dump = *crate::module_parameters::cs_dump_count.value();
                        let ctrl_fix = *crate::module_parameters::cs_draw_ctrl_fix.value();
                        if ctrl_fix != 0 {
                            gpu.patch_vm(
                                slot,
                                cs,
                                ctrl_fix as usize,
                                0xffff_0000,
                                0x61c0_0000,
                                0x0004_0000,
                                0xffff_ffff,
                            );
                            gpu.patch_vm(
                                slot,
                                cs,
                                ctrl_fix as usize,
                                0xffff_0000,
                                0x61e0_0000,
                                0x0002_0000,
                                0xffff_ffff,
                            );
                            // 🔑 The stream-advance record.  Apple writes a
                            // *bare* `0x40000000` and advances the cursor by 4
                            // (`AGXMetalG15X_M1` at 0x268d30: `add x11, x9, #4;
                            // mov w8, #0x40000000; str w8, [x9]`), while Mesa
                            // emits `0x4000002e` -- the same record with a low
                            // half Apple never writes.  That is exactly what
                            // the hand-placed `cs_poke_off=0xc` constant has
                            // been correcting since 32.41.185; clearing the low
                            // 16 bits of every `0x4000xxxx` record makes it a
                            // rule instead of an offset.  LOG.md 32.41.262.
                            // ⛔ 32.41.262's premise is WRONG and this patch
                            // destroys the vertex shader state. Apple writes
                            // `0x40000000` as a *reserved* header and stashes its
                            // address (`AGXMetalG15X_M1` 0x268d4c: `add x11,x9,#4;
                            // mov w8,#0x40000000; str w8,[x9]; str x10,[x20,#96]`)
                            // precisely so it can patch the present bits in once
                            // the state words have been appended. Seeing the bare
                            // constant at the emitter proves the header starts
                            // empty, not that it stays empty.
                            //
                            // Mesa's `0x4000002e` declares four present words --
                            // VS Word 0, VS Word 1, Vertex Outputs, Vertex Unknown
                            // -- and Mesa does write all four (`0x00001212`,
                            // `0xfff68000` the pipeline, `0x00000505`,
                            // `0x00000002`). Clearing the mask leaves them in the
                            // stream unconsumed, so the vertex shader is never
                            // bound: nothing bins, `pass_pages` stays 0 and the
                            // job never completes. LOG.md 32.41.382.
                            if *crate::module_parameters::cs_keep_vdmstate_low.value() == 0 {
                            gpu.patch_vm(
                                slot,
                                cs,
                                ctrl_fix as usize,
                                0xffff_0000,
                                0x4000_0000,
                                0x0000_0000,
                                0xffff_0000,
                            );
                            }
                        }
                        // 🔑 The record-length fix.  Apple's G15 draw record is
                        // eight dwords, Mesa's is four, so the CSD swallows
                        // Mesa's terminator as a record field and runs to the
                        // end of the 512 KiB BO -- the fault at +0x80 past it
                        // that has been the top-level symptom since 32.41.185.
                        // The trailing values are transcribed from the built-in
                        // blit stream in AGXMetalG15X_M1 (`__const` 0x6de600),
                        // whose draw is `count 3, instance 1, start 0` -- the
                        // same draw this test issues.  LOG.md 32.41.299.
                        //
                        // Deliberately at the same nesting as `ctrl_fix`, NOT
                        // inside `if probe_usc != 0` (32.41.253, 32.41.298).
                        // Runs after `ctrl_fix`, so it matches the already
                        // corrected `0x61c4` CTRL word.
                        // 🔑 Apple's draw CTRL word is *exactly* 0x61c40000.
                        // Its encoder builds the record in a vector whose lane 0
                        // is set once (`mov w8, #0x61c40000; fmov s0, w8`) and
                        // never touched again before `str q0, [x9], #16`
                        // (AGXMetalG15X_M1 0x698b40..0x698b8c), so the low half
                        // is zero.  Mesa emits 0x61c00600; `cs_draw_ctrl_fix`
                        // only adds bit 18 and leaves 0x0600 behind.  If those
                        // low bits declare optional words, the CSD expects words
                        // Mesa never wrote and desyncs -- which is the runaway of
                        // 32.41.297.  Separate from `cs_draw_len_fix` (whose
                        // 8-dword premise is retracted, 32.41.300) so the two
                        // cannot be confounded.  LOG.md 32.41.302.
                        // 🔑 Relocate the draw to the next real record
                        // boundary.  The measured map (32.41.143-145, and the
                        // cs+0x48 boundary reproduced in 32.41.304) is:
                        //
                        //   0x00  8   0x08  8   0x10  24   0x28  32   0x48 ...
                        //
                        // Mesa lays the draw at 0x34 and terminates at 0x40,
                        // both of which fall *inside* record @0x28's 32 bytes,
                        // so the hardware consumes them as its payload and the
                        // draw never executes.  0x48 is the next boundary the
                        // hardware agrees with, so put the draw there and
                        // terminate after it.  The original bytes at 0x34/0x40
                        // are left alone: they are payload either way.
                        //
                        // Offsets are relative to the matched CTRL dword at
                        // 0x34 (dword 13): +5 -> 0x48, +9 -> 0x58.
                        // LOG.md 32.41.305.
                        // 🔑 The count fix as a RULE, not a fixed-offset poke.
                        // Runs before the dump so the read-back shows the
                        // patched stream. LOG.md 32.41.330.
                        // 🔬 Probe what the stream's operands point at. The
                        // stream itself is byte-identical between a run that
                        // faults and one that does not (32.41.330), so the
                        // difference has to be in the referenced buffers.
                        // Operands are 32-bit and resolve against a window base
                        // (32.41.313); probe them at the control stream's own
                        // region, which is where these values' full addresses
                        // live. LOG.md 32.41.333.
                        let pops = *crate::module_parameters::cs_probe_ops.value();
                        if pops != 0 {
                            let n = core::cmp::min(pops as usize, 64);
                            let mut buf = [0u32; 64];
                            gpu.read_vm(slot, cs, &mut buf[..n]);
                            let region = cs & !0xffff_ffffu64;
                            for i in 0..n {
                                let w = buf[i];
                                if (w & 0xffff_0000) == 0xfff8_0000
                                    || (w & 0xffff_0000) == 0xfff6_0000
                                {
                                    gpu.probe_vm(slot, region | u64::from(w), "csop");
                                }
                            }
                        }
                        let cfix = *crate::module_parameters::cs_count_fix.value();
                        if cfix != 0 {
                            gpu.fix_count_vm(slot, cs, cfix as usize);
                        }
                        let reloc = *crate::module_parameters::cs_draw_relocate.value();
                        if reloc != 0 {
                            let mut hdr = [0u32; 4];
                            gpu.read_vm(slot, cs + 0x34, &mut hdr);
                            let w: KVec<(usize, u32)> = kernel::kvec![
                                (5usize, hdr[0]),
                                (6usize, hdr[1]),
                                (7usize, hdr[2]),
                                (8usize, hdr[3]),
                                (9usize, 0xc000_0000u32)
                            ]?;
                            gpu.splice_vm(slot, cs, reloc as usize, 0xffff_0000, 0x61c4_0000, &w);
                        }
                        let ctrl_exact = *crate::module_parameters::cs_draw_ctrl_exact.value();
                        if ctrl_exact != 0 {
                            // 32.41.469: the search pattern was 0x61c4_0000 -- a draw
                            // CTRL that ALREADY has bit 18 set. Mesa emits 0x61c0_0600,
                            // which does not match, so with cs_draw_ctrl_fix off (it is
                            // the .447 regression and stays off) this parameter matched
                            // nothing and silently did nothing. It only ever worked
                            // because ctrl_fix had set bit 18 first.
                            //
                            // Match Mesa's form by default instead. cs_draw_ctrl_match
                            // makes the pattern explicit so a future change to Mesa's
                            // CTRL word cannot silently void the splice again.
                            let pat = *crate::module_parameters::cs_draw_ctrl_match.value();
                            let w: KVec<(usize, u32)> =
                                kernel::kvec![(0usize, ctrl_exact)]?;
                            gpu.splice_vm(slot, cs, 64, 0xffff_0000, pat, &w);
                            // Gate: read the stream back and report the draw CTRL
                            // actually present. A splice that matches nothing is
                            // indistinguishable from one that had no effect --
                            // that voided the first bit-18 run.
                            let mut v = [0u32; 64];
                            let got = gpu.read_vm(slot, cs, &mut v);
                            let mut found = 0u32;
                            let mut i = 0;
                            while i < got {
                                if (v[i] & 0xff00_0000) == 0x6100_0000 {
                                    found = v[i];
                                    break;
                                }
                                i += 1;
                            }
                            pr_info!(
                                "asahi/ctrl_exact: pattern {:#010x} -> {:#010x}; draw CTRL now {:#010x}\n",
                                pat, ctrl_exact, found
                            );
                        }
                        let len_fix = *crate::module_parameters::cs_draw_len_fix.value();
                        if len_fix != 0 {
                            let low = *crate::module_parameters::cs_draw_ctrl_low.value();
                            let mut writes: KVec<(usize, u32)> = KVec::new();
                            if low != 0 {
                                writes.push((0usize, 0x61c4_0000 | (low & 0xffff)), GFP_KERNEL)?;
                            }
                            // +0x10 = A, +0x18 = B, terminator to +0x20.
                            writes.push((4usize, 0x0000_0030), GFP_KERNEL)?;
                            writes.push((5usize, 0x0000_0000), GFP_KERNEL)?;
                            writes.push((6usize, 0x0000_0020), GFP_KERNEL)?;
                            writes.push((7usize, 0x0000_0000), GFP_KERNEL)?;
                            writes.push((8usize, 0xc000_0000), GFP_KERNEL)?;
                            gpu.splice_vm(
                                slot,
                                cs,
                                len_fix as usize,
                                0xffff_0000,
                                0x61c4_0000,
                                &writes,
                            );
                        }
                        if cs_dump != 0 {
                            gpu.dump_vm(slot, cs, cs_dump as usize, "cs");
                        }
                        }
                        if *crate::module_parameters::probe_usc.value() != 0 {
                            let slot = vm_bind.slot();
                            let bg = u64::from(cmdbuf.bg.usc) & !7;
                            let eot = u64::from(cmdbuf.eot.usc) & !7;
                            gpu.probe_vm(slot, eot, "usc eot bare");
                            gpu.probe_vm(slot, self.usc_exec_base + eot, "usc eot +base");
                            gpu.probe_vm(slot, bg, "usc bg bare");
                            gpu.probe_vm(slot, self.usc_exec_base + bg, "usc bg +base");
                            gpu.probe_vm(slot, self.usc_exec_base, "usc exec_base");
                            // The VDM runs 0x80080 bytes linearly from the
                            // control-stream base and falls off the end of the
                            // mapping without ever seeing a terminator, which
                            // is what an *empty* stream looks like. Dump the
                            // head of it and the page it dies on.
                            let cs = cmdbuf.vdm_ctrl_stream_base & !3;
                            // 🔬 Optional: overwrite one quadword of the
                            // stream before the kick. `cs_poke_off=8`
                            // `cs_poke_val=0xc0000000fff88010` rewrites the draw
                            // stream's second quadword to the clear stream's,
                            // which is the one construct the two differ by --
                            // and the clear stream does *not* make the VDM run
                            // off the end (32.41.139). If the fault goes away,
                            // the derailment is at or after cs+0x0c and the
                            // 0xc0000000 terminator is understood.
                            // 🔑 The record-walk pad fix -- a RULE, not an offset.
                            //
                            // G15 mis-parses Mesa's stream at one specific kind of
                            // word: a bare `0x00000000`. Block Type is bits [31:29],
                            // so zero reads as type 0 = PPP State Update, which is
                            // **8 bytes**; Mesa emits it as 4 bytes of padding. The
                            // hardware therefore eats the following record's header
                            // and every record after it is 4 bytes out of phase,
                            // which is how `cs+0x24`'s Viewport header got swallowed
                            // and `cs+0x28`/`cs+0x30` -- 32-bit addresses -- ended up
                            // in a count field, producing the ~64K-unit runaway that
                            // has been the top-level symptom since 32.41.185.
                            //
                            // A zero dword **cannot** be a real PPP State: `hi = 0`,
                            // `size = 0` reads zero words from address 0, a no-op. So
                            // it is unambiguously padding, and replacing it with a
                            // Barrier (type 1, 4 bytes -- the size is measured, since
                            // `cs+0x00 = 0x20000008` is a Barrier and the next record
                            // starts at `cs+0x04`) keeps the walk in phase wherever
                            // the pad appears. Apple emits a bare Barrier as filler
                            // itself (`AGXMetalG15X_M1` 0x2962b8).
                            //
                            // Only *header* positions are examined. Payload dwords are
                            // legitimately zero (the Index List's `start` at `cs+0x3c`,
                            // for one), so a blanket scan-and-replace would corrupt the
                            // stream -- hence the walk.
                            //
                            // ⚠️ The walk assumes **one dword per present bit** for the
                            // VDM State Update. That is verified for the Index List
                            // (bits 22/23/24 -> 3 words -> a 16-byte record, matching
                            // Apple's emitter and Mesa's terminator placement) but NOT
                            // for the state update, where G15 appears to consume a
                            // different count than Mesa writes -- the open question in
                            // 32.41.382 that needs Apple's reference stream. With the
                            // present mask at 0 the state update is 4 bytes and the
                            // assumption is irrelevant; with a non-zero mask the walk
                            // may desync, so the log line below prints every boundary
                            // it finds. LOG.md 32.41.389.
                            let walk = *crate::module_parameters::cs_walk_fix.value();
                            if walk != 0 {
                                let mut buf = [0u32; 96];
                                let lim = core::cmp::min(walk as usize, buf.len());
                                let n = gpu.read_vm(slot, cs, &mut buf[..lim]);
                                let mut i = 0usize;
                                let mut pads = 0u32;
                                while i < n {
                                    let w = buf[i];
                                    if w == 0 {
                                        // Padding. Rewrite as a Barrier, preserving the
                                        // other half of the containing quadword.
                                        let qva = cs + ((i as u64) & !1) * 4;
                                        let (lo, hi) = if i & 1 == 0 {
                                            (0x2000_0008u32, *buf.get(i + 1).unwrap_or(&0))
                                        } else {
                                            (buf[i - 1], 0x2000_0008u32)
                                        };
                                        gpu.poke_vm(
                                            slot,
                                            qva,
                                            (lo as u64) | ((hi as u64) << 32),
                                        );
                                        pr_info!(
                                            "asahi/cswalk pad at cs+{:#x} -> Barrier\n",
                                            i * 4
                                        );
                                        pads += 1;
                                        i += 1;
                                        continue;
                                    }
                                    let ty = w >> 29;
                                    let words = match ty {
                                        0 => 2,                     // PPP State, 8B
                                        1 => 1,                     // Barrier, 4B
                                        2 => 1 + (w & 0xff).count_ones() as usize,
                                        3 => {
                                            // present bits 20..27; bit 25 is 2 words
                                            let p = (w >> 20) & 0xff;
                                            let extra = ((p >> 5) & 1) as usize;
                                            1 + (p & 0xdf).count_ones() as usize + extra
                                        }
                                        4 => 2,                     // Stream Link, 8B
                                        6 => {
                                            pr_info!(
                                                "asahi/cswalk terminate at cs+{:#x} ({} pads fixed)\n",
                                                i * 4,
                                                pads
                                            );
                                            break;
                                        }
                                        _ => {
                                            pr_info!(
                                                "asahi/cswalk STOP: unknown type {} at cs+{:#x} word {:#010x}\n",
                                                ty,
                                                i * 4,
                                                w
                                            );
                                            break;
                                        }
                                    };
                                    pr_info!(
                                        "asahi/cswalk cs+{:#x} type {} words {} ({:#010x})\n",
                                        i * 4,
                                        ty,
                                        words,
                                        w
                                    );
                                    i += words;
                                }
                            }
                            // 🔬 Arbitrary-VA poke, in the job's own VM slot. The
                            // cs_poke family is all cs-relative, which cannot reach the
                            // USC word list (it lives at `usc_exec_base + VS Word 1's
                            // Pipeline`, a different object entirely).
                            //
                            // Purpose: decide whether the hardware actually consumes the
                            // USC list at the address VS Word 1 names. 32.41.394 found a
                            // contradiction that Mesa's own assert
                            // (`has_preamble == (nr_preamble_gprs > 0)`) says is
                            // impossible -- VS Word 0 declares preshader registers while
                            // that list ends in `No preshader` -- which would mean the
                            // list the shader runs against is not the one Mesa built for
                            // it. Poisoning a field of the list answers it directly: if
                            // the fault follows the poison, the hardware is using this
                            // list and the mismatch reading is dead; if the fault does not
                            // move, the hardware is reading a different list.
                            // LOG.md 32.41.395.
                            let gpva = *crate::module_parameters::gpu_poke_va.value();
                            if gpva != 0 {
                                let gpval = *crate::module_parameters::gpu_poke_val.value();
                                pr_info!(
                                    "asahi/gpoke {:#x} <- {:#018x}\n",
                                    gpva,
                                    gpval
                                );
                                gpu.poke_vm(slot, gpva, gpval);
                            }
                            // 32.41.452: reshape the VDM State Update into Apple's
                            // 28-byte form. Revised from 32.41.451 after deriving the
                            // slot semantics from AGXMetalG15X_M1:
                            //
                            //   wA = bfi  w10,w11,#8,#12   12-bit field at [19:8]
                            //   wB = ubfiz w8,w8,#3,#18    an 8-byte-aligned ADDRESS
                            //   wC = bfi  w12,w11,#8,#8    a byte at [15:8]
                            //   wD = (x & 0x40) | [4] | 0x30   -- ALWAYS has 0x30
                            //
                            // Mesa emits 0x1212 (fits wA), 0xfff68000 (fits wB) and
                            // 0x505 (fits wC) -- but nothing with 0x30, so Mesa writes
                            // only THREE payload words and its record ends at cs+0x1c.
                            // The 0x00000002 there is a SEPARATE record. So wD has to
                            // be synthesised, and the tail shifts from cs+0x1c by 3
                            // dwords (0x1c -> 0x28), not from cs+0x24 by 1.
                            //
                            // 32.41.451 also capped the shift at 16 dwords, which left
                            // a 4-byte discontinuity mid-stream -- a corruption, not a
                            // reshape. Shift a generous window instead.
                            // 32.41.464: Apple's record, CAPTURED (capture4,
                            // 2026-09-17), not derived. Three runs and six
                            // workload variants, byte-identical:
                            //
                            //   4000002e 00000000 01000000 00004a00
                            //            00000404 00000000 00000000
                            //
                            //   w1  0            always
                            //   w2  0x01000000   always
                            //   w3  PIPELINE     tex 0x5400, depth 0x4e00 --
                            //                    the pipeline VA >> 6, low 32 bits
                            //   w4  VERTEX OUTS  0x404, and 0x808 with one extra
                            //                    varying -- the only word that moved
                            //   w5  0            (this is the "wD" slot)
                            //   w6  0
                            //
                            // 🔑 Mesa writes THREE payload words -- 0x1212,
                            // 0xfff68000, 0x505 -- and its middle one is the
                            // pipeline, correctly encoded (0x2fffda0000 >> 6).
                            // It is simply in the WRONG SLOT: Mesa puts it at
                            // payload 2, the hardware reads it at payload 3.
                            //
                            // ⛔ The previous reshape (.452) shifted Mesa's tail
                            // correctly but left payload 3 = 0 and kept the
                            // pipeline at payload 2. So the USC fetched its
                            // pipeline from ZERO -- which is exactly the fault
                            // this phase has been chasing: a small index against
                            // a base that was never supplied. .458 guessed "a
                            // base register reading zero" from a broken decode;
                            // the base is real and it is this word.
                            //
                            // Mesa's 0x1212 is DROPPED: Apple writes 0 at w1 even
                            // for draws that bind a uniform, a texture+sampler or
                            // a vertex buffer. Bindings are not in the stream at
                            // all on G15 -- they live in the per-pipeline blob w3
                            // points at. cs_w1 puts it back for a bisect.
                            {
                                let n = *crate::module_parameters::cs_reshape.value() as usize;
                                if n != 0 {
                                    let n = core::cmp::min(n, 512);
                                    let w1 = *crate::module_parameters::cs_w1.value();
                                    let w2 = *crate::module_parameters::cs_w2.value();
                                    let w5 = *crate::module_parameters::cs_wd.value();
                                    // Mesa's three payload words, before anything
                                    // moves: [0] 0x1212, [1] pipeline, [2] outputs.
                                    let mut m = [0u32; 3];
                                    if gpu.read_vm(slot, cs + 0x10, &mut m) != 3 {
                                        pr_info!("asahi/reshape: SKIPPED -- cannot read cs+0x10\n");
                                    } else {
                                    // 32.41.464: Mesa's word is a RAW USC-relative
                                    // byte address -- 0xfff68000 is exactly
                                    // 0x11fff68000 - usc_exec_base. Apple's word 3
                                    // is the GPU VA >> 6, truncated to 32 bits,
                                    // which the capture confirms exactly for three
                                    // pipelines (0x100000128000>>6 = 0x4a00,
                                    // ...138000 = 0x4e00, ...150000 = 0x5400).
                                    //
                                    // ⚠️ capture4's RESULTS.md claims Mesa's word
                                    // already obeys the >>6 rule ("0x2fffda0000 >> 6").
                                    // That does not compute: 0x2fffda0000 >> 6 is
                                    // 0xbfff6800, not 0xfff68000. The subtraction
                                    // above does compute, exactly, so Mesa is writing
                                    // the G13 form and it needs re-encoding.
                                    //
                                    // cs_w3_raw=1 sends Mesa's word through unchanged,
                                    // so one run separates "wrong slot" from
                                    // "wrong slot AND wrong encoding".
                                    // .478: Apple emits 0x404 here for this exact draw
                                    // (0x808 with a varying bound); we pass Mesa's 0x505.
                                    let outputs = match *crate::module_parameters::cs_w4.value() {
                                        0 => m[2],
                                        v => v,
                                    };
                                    // 32.41.467: mode 2 is the derived one.
                                    //
                                    // seg-1 measured the hardware's rule exactly:
                                    //   addr = CODE_GART_BASE + ((w3 & 0xffffffff) << 6)
                                    // with CODE_GART_BASE = 0x10000000000 (capture2
                                    // FINDING 3 / gart_ranges[5]), and it is NOT any of
                                    // the five USC segment-base registers -- five
                                    // distinct probe values moved the offset within the
                                    // blob (0x80 -> 0x180) and left the base untouched.
                                    //
                                    // Compensating inside w3 is impossible: the value
                                    // needed is 0xc47ffda00, which does not fit the
                                    // 32-bit field. Addition cannot reach below its own
                                    // base. So the shader has to be mapped INSIDE the
                                    // window, which is what macOS does.
                                    //
                                    // With the USC heap aliased at win36_base =
                                    // 0x10000000000, our BO at 0x11fff68000 gains a
                                    // second mapping at 0x10000000000 | 0xfff68000, and
                                    // w3 is simply Mesa's USC-relative word >> 6:
                                    //   0x10000000000 + (0x3ffda00 << 6) = 0x100fff68000
                                    let pipeline = match *crate::module_parameters::cs_w3_raw.value() {
                                        1 => m[1],
                                        2 => m[1] >> 6,
                                        _ => (((self.usc_exec_base + m[1] as u64) >> 6) & 0xffff_ffff) as u32,
                                    };
                                    // Shift [cs+0x1c, +n dwords) up by 3 dwords, top
                                    // down so the overlap is safe. One dword at a time
                                    // via a 64-bit read-modify-write of the aligned
                                    // pair, so no alignment assumptions are needed.
                                    // 32.41.466: the shift origin was wrong, and it
                                    // manufactured the fault .465 chased.
                                    //
                                    // Mesa's mask 0x2e has FOUR bits set (1/2/3/5), so
                                    // Mesa writes FOUR payload words, then a 4-byte
                                    // zero pad:
                                    //
                                    //   0x0c mask | 0x10 w1 0x14 w2 0x18 w3 0x1c w4
                                    //            | 0x20 pad | 0x24 PPP...
                                    //
                                    // so Mesa's record + pad ends at cs+0x24, and
                                    // Apple's 28-byte record ends at cs+0x28. The tail
                                    // therefore moves by exactly ONE dword from
                                    // cs+0x24 -- not three dwords from cs+0x1c.
                                    //
                                    // ⛔ .452 said Mesa writes only THREE payload words.
                                    // That was wrong: it counted the slots a derived wD
                                    // could occupy rather than the bits in the mask.
                                    // Shifting from 0x1c by 3 pushed Mesa's 4th payload
                                    // word (0x00000002) and its pad out to cs+0x28,
                                    // where the decoder read 0x00000002 as a PPP State
                                    // header -- size 0, addr (0x2<<32)|0 = 0x200000000,
                                    // which the win36 alias maps to 0x1200000000. The
                                    // fault at 0x1200000080 was that record, which is
                                    // why it did not move when w3 changed (.465).
                                    // .479: push the pipeline pointer into unmapped space
                                    // to test whether the USC READS it at all here.
                                    let pipeline = pipeline
                                        .wrapping_add(*crate::module_parameters::cs_w3_add.value());
                                    let from = *crate::module_parameters::cs_shift_from.value() as u64;
                                    let by = *crate::module_parameters::cs_shift_by.value() as u64;
                                    let mut i = n;
                                    while i > 0 {
                                        i -= 1;
                                        let src = cs + from + (i as u64) * 4;
                                        let dst = src + by * 4;
                                        let mut v = [0u32; 1];
                                        if gpu.read_vm(slot, src, &mut v) != 1 {
                                            continue;
                                        }
                                        // read the aligned 8-byte container of dst,
                                        // replace the right half, write it back
                                        let base = dst & !7u64;
                                        let mut pair = [0u32; 2];
                                        if gpu.read_vm(slot, base, &mut pair) != 2 {
                                            continue;
                                        }
                                        if (dst & 4) == 0 {
                                            pair[0] = v[0];
                                        } else {
                                            pair[1] = v[0];
                                        }
                                        gpu.poke_vm(
                                            slot,
                                            base,
                                            (pair[0] as u64) | ((pair[1] as u64) << 32),
                                        );
                                    }
                                    // Now lay the six payload words out Apple's way.
                                    // cs is 8-aligned, so cs+0x10/+0x18/+0x20 each
                                    // cover one aligned pair.
                                    gpu.poke_vm(slot, cs + 0x10, (w1 as u64) | ((w2 as u64) << 32));
                                    gpu.poke_vm(
                                        slot,
                                        cs + 0x18,
                                        (pipeline as u64) | ((outputs as u64) << 32),
                                    );
                                    gpu.poke_vm(slot, cs + 0x20, w5 as u64);
                                    pr_info!(
                                        "asahi/reshape: w1 {:#x} w2 {:#x} w3(pipeline) {:#x} (mesa {:#x}, usc_base {:#x}) w4(outputs) {:#x} w5 {:#x} w6 0x0, shifted {} dwords from +{:#x} by {}\n",
                                        w1, w2, pipeline, m[1], self.usc_exec_base, outputs, w5, n, from, by
                                    );
                                    }
                                }
                            }
                            // 32.41.450: dump the head of Mesa's control stream.
                            // This whole sub-problem has been reasoned about from
                            // notes describing Mesa's record layout; the bytes have
                            // never been looked at. 32.41.449 derived Apple's
                            // mask-0x2e record as 28 bytes (header + 6 words) from
                            // AGXMetalG15X_M1, against the 20 bytes the notes
                            // ascribe to Mesa -- this checks that directly, and
                            // gives cs+0x10's real value so a poke at cs+0xc can
                            // preserve it instead of guessing.
                            {
                                let n = *crate::module_parameters::dump_cs.value() as usize;
                                if n != 0 {
                                    let mut buf = [0u32; 256];
                                    let n = core::cmp::min(n, buf.len());
                                    let got = gpu.read_vm(slot, cs, &mut buf[..n]);
                                    pr_info!("asahi/cs base {:#x} read {} dwords\n", cs, got);
                                    let mut i = 0usize;
                                    while i < got {
                                        pr_info!(
                                            "asahi/cs +{:#04x}: {:#010x} {:#010x} {:#010x} {:#010x}\n",
                                            i * 4,
                                            buf[i],
                                            if i + 1 < got { buf[i + 1] } else { 0 },
                                            if i + 2 < got { buf[i + 2] } else { 0 },
                                            if i + 3 < got { buf[i + 3] } else { 0 },
                                        );
                                        i += 4;
                                    }
                                }
                            }
                            // .521 -- DOES THE VDM REACH THE INDEX LIST?
                            //
                            // .520 showed the USC never executes the word-3 object, so the
                            // blocker is upstream of the USC.  .494 proved the VDM fetches
                            // the PPP block whose header sits at cs+0x2c/+0x30 -- the record
                            // immediately BEFORE the draw.  Whether it then executes the
                            // Index List at cs+0x38 has never been measured: our Index List
                            // record is byte-identical to Apple's (61c40600 00000003
                            // 00000001 00000000, 3 vertices 1 instance), so "the VDM reached
                            // it and issued a draw" and "the VDM stopped before it" are
                            // indistinguishable from any observable we have.
                            //
                            // Make them distinguishable: overwrite the Index List with a VDM
                            // Stream Link (block type 4) pointing at unmapped memory.  If the
                            // VDM executes that word it jumps there and the fault decoder
                            // names the address; if it does not, the VDM never got this far.
                            //
                            // Instrument validity: .377 planted a Stream Link and the VDM
                            // followed it, and .493/.494 show VDM fetch faults are reported.
                            //
                            // The offset is SCANNED, not hardcoded: find the first record
                            // whose block type (bits 31:29) is 3 (Index List).  cs_reshape
                            // moves everything after +0x24, so a literal +0x38 would silently
                            // rot the moment the reshape changes.
                            {
                                let link = *crate::module_parameters::cs_link_probe.value();
                                if link != 0 {
                                    let mut w = [0u32; 32];
                                    let got = gpu.read_vm(slot, cs, &mut w);
                                    let mut found: Option<usize> = None;
                                    for i in 0..got {
                                        if (w[i] >> 29) == 3 {
                                            found = Some(i);
                                            break;
                                        }
                                    }
                                    match found {
                                        Some(i) => {
                                            let hi = ((link >> 32) & 0xff) as u32;
                                            let lo = (link & 0xffff_ffff) as u32;
                                            let w0 = 0x8000_0000u32 | hi;
                                            gpu.poke_vm(
                                                slot,
                                                cs + (i as u64) * 4,
                                                ((lo as u64) << 32) | (w0 as u64),
                                            );
                                            let mut rb = [0u32; 4];
                                            let n = gpu.read_vm(slot, cs + (i as u64) * 4, &mut rb);
                                            pr_info!(
                                                "asahi/linkprobe: Index List found at cs+{:#x} (was {:#010x}), planted Stream Link -> {:#x}; readback[{}] {:08x} {:08x} {:08x} {:08x}\n",
                                                i * 4, w[i], link, n, rb[0], rb[1], rb[2], rb[3]
                                            );
                                        }
                                        None => {
                                            pr_info!(
                                                "asahi/linkprobe: *** VOID *** no Index List record (block type 3) in {} dwords at cs\n",
                                                got
                                            );
                                        }
                                    }
                                }
                            }
                            // Reorder the stream head into Apple's order (.498).
                            //
                            // Apple's own captured stream opens at dword 0 with the VDM
                            // State Update and emits ZERO PPP blocks before it. Mesa opens
                            // with a Barrier and one PPP block. .477/.478 compared PPP
                            // coverage and the state-update payload -- never record ORDER.
                            //
                            //   before: [Barrier 4][PPP 8][State Update 28][rest...]
                            //   after:  [State Update 28][Barrier 4][PPP 8][rest...]
                            //
                            // Size-preserving, so nothing downstream moves.
                            if *crate::module_parameters::cs_head_apple.value() != 0 {
                                let mut w = [0u32; 10];
                                if gpu.read_vm(slot, cs, &mut w) == 10 {
                                    let mut nw = [0u32; 10];
                                    for i in 0..7 {
                                        nw[i] = w[3 + i];
                                    }
                                    nw[7] = w[0];
                                    nw[8] = w[1];
                                    nw[9] = w[2];
                                    for i in 0..5 {
                                        gpu.poke_vm(
                                            slot,
                                            cs + (i as u64) * 8,
                                            (nw[i * 2] as u64) | ((nw[i * 2 + 1] as u64) << 32),
                                        );
                                    }
                                    // Read back: the head must now start 0x4000002e and the
                                    // Barrier must have moved to +0x1c. An unverified
                                    // permutation would be indistinguishable from a null.
                                    let mut rb = [0u32; 10];
                                    let nr = gpu.read_vm(slot, cs, &mut rb);
                                    pr_info!(
                                        "asahi/cs head[{}]: {:08x} {:08x} {:08x} {:08x} | {:08x} {:08x} {:08x} | {:08x} {:08x} {:08x}\n",
                                        nr, rb[0], rb[1], rb[2], rb[3], rb[4], rb[5], rb[6], rb[7], rb[8], rb[9]
                                    );
                                }
                            }
                            let poke_off = *crate::module_parameters::cs_poke_off.value();
                            if poke_off != 0xffffffff {
                                gpu.poke_vm(
                                    slot,
                                    cs + poke_off as u64,
                                    *crate::module_parameters::cs_poke_val.value(),
                                );
                                // Read it back. `dump_cs` prints BEFORE this poke,
                                // so without this a poke that never landed is
                                // indistinguishable from one that landed and did
                                // nothing -- which is precisely how
                                // `cs_draw_ctrl_exact` produced a false clean
                                // negative (.208). Never read a null from a poke
                                // that was not verified.
                                let mut pk = [0u32; 4];
                                let npk = gpu.read_vm(slot, cs + poke_off as u64, &mut pk);
                                pr_info!(
                                    "asahi/cs poke +{:#x}[{}]: {:08x} {:08x} {:08x} {:08x}\n",
                                    poke_off, npk, pk[0], pk[1], pk[2], pk[3]
                                );
                            }
                            // 🔬 Second poke. The pipeline fix needs cs+0x0c
                            // (avoid the CSD runaway) *and* cs+0x10 (the
                            // addr_hi byte) rewritten in the same submission.
                            // LOG.md 32.41.187
                            let poke2_off = *crate::module_parameters::cs_poke2_off.value();
                            if poke2_off != 0xffffffff {
                                gpu.poke_vm(
                                    slot,
                                    cs + poke2_off as u64,
                                    *crate::module_parameters::cs_poke2_val.value(),
                                );
                            }
                            // 🔬 Sweep: plant the terminator at *every* dword
                            // across a range in one submission. Each boot
                            // submits exactly one job, so bisecting a record
                            // boundary one candidate per boot costs ~5 minutes
                            // per bit; this answers "does the record end
                            // anywhere in [off, off+4*n)" in a single run.
                            // LOG.md 32.41.163.
                            let fill_off = *crate::module_parameters::cs_fill_off.value();
                            let fill_n = *crate::module_parameters::cs_fill_count.value();
                            if fill_off != 0xffffffff && fill_n != 0 {
                                gpu.poke_fill_vm(
                                    slot,
                                    cs + fill_off as u64,
                                    *crate::module_parameters::cs_fill_val.value(),
                                    fill_n as usize,
                                );
                            }
                            // 🔬 Probe a buffer the control stream points at.
                            // The stream stores only the low 32 bits of these
                            // addresses (e.g. 0xfff88b44 for 0x2ffff88b44), so
                            // rebuild the full VA from the control stream's own
                            // high half. Used to check whether the block the
                            // record at cs+0x2c points to really is PPP state.
                            // LOG.md 32.41.180.
                            // 32.41.471: dump EVERY PPP State block the stream
                            // points at, in one run.
                            //
                            // capture4 has Apple's nine blocks with contents; we
                            // have only Mesa's three record headers and their
                            // pointers, never the blocks themselves. That is the
                            // largest unexamined structural difference left:
                            // Apple sends 9 PPP State records totalling 46 words,
                            // Mesa sends 3 totalling 37. `probe_ptr` can reach one
                            // address per run, and at ~6 minutes a run with a 50%
                            // chance of losing the tail that is the wrong tool.
                            //
                            // PPP State is block type 0 (bits [31:29]) and the
                            // header is `size_words << 8 | addr[39:32]`, with the
                            // next dword the low 32 bits of the address --
                            // confirmed against Mesa's own 0x0000062f/0xfff88010
                            // and Apple's 0x00000500/0x00000040.
                            let dppp = *crate::module_parameters::dump_ppp.value() as usize;
                            if dppp != 0 {
                                let n = core::cmp::min(dppp, 64);
                                let mut w = [0u32; 64];
                                let got = gpu.read_vm(slot, cs, &mut w);
                                // ⛔ First attempt walked record-by-record and broke
                                // on dword 0, because Mesa's stream opens with a
                                // Barrier (type 1) and the walk only understood
                                // type 0. Scan instead, and key on the address high
                                // byte matching the control stream's own -- Mesa's
                                // records are 0x0000062f / 0x00000b2f / 0x0000142f
                                // against a stream at 0x2fxxxxxxxx, so that single
                                // check is highly selective and needs no record
                                // sizes at all.
                                let hi = ((cs >> 32) & 0xff) as u32;
                                let mut i = 0;
                                while i + 1 < got && i < n {
                                    let hdr = w[i];
                                    let sz = (hdr >> 8) & 0xff;
                                    if (hdr >> 29) == 0 && sz != 0 && (hdr & 0xff) == hi {
                                        let va = (((hdr & 0xff) as u64) << 32) | (w[i + 1] as u64);
                                        pr_info!(
                                            "asahi/ppp[{}] hdr {:#010x} size {} words va {:#x}\n",
                                            i, hdr, sz, va
                                        );
                                        // ⚠️ .471 used probe_vm here, which dumps a
                                        // FIXED 8 words -- so every block longer than
                                        // that was silently truncated, and the "the
                                        // distinctive words are all present" reading in
                                        // .471 was made against elided content. Dump the
                                        // block's ACTUAL size instead, so Mesa's state
                                        // can be compared with Apple's nine word for
                                        // word. LOG.md 32.41.477
                                        gpu.dump_vm(slot, va, sz as usize, "ppp");
                                        i += 2;
                                    } else {
                                        i += 1;
                                    }
                                }
                            }
                            // 32.41.474: dump the USC pipeline blob itself.
                            //
                            // capture4 captured Apple's blob for the same draw --
                            // 0x50 bytes at 0x1_0000_0128000, plus variants for
                            // uniform (0xa0), vbuf (0xb0) and tex (0xc0) -- and
                            // identified it as a prologue/preamble shader carrying
                            // the bindings. This is the object the USC actually
                            // consumes, and we have never looked at ours.
                            //
                            // The address is the same one w3 encodes, so it is
                            // derived, not guessed: pipeline = (w3 << 6), i.e.
                            // usc_exec_base + Mesa's word.
                            let dpipe = *crate::module_parameters::dump_pipeline.value() as usize;
                            if dpipe != 0 {
                                let mut mw = [0u32; 1];
                                if gpu.read_vm(slot, cs + 0x18, &mut mw) == 1 {
                                    // cs+0x18 is payload word 3 after the reshape.
                                    // ⚠️ w3's meaning depends on cs_w3_raw. In mode 2
                                    // it is the USC-RELATIVE offset >> 6, so w3<<6 is
                                    // an offset, not an address -- the first attempt
                                    // dumped 0xfff68000 and got "unreadable". Dump
                                    // both candidates and let the page tables say
                                    // which is real.
                                    let off = (mw[0] as u64) << 6;
                                    let n = core::cmp::min(dpipe, 64);
                                    pr_info!(
                                        "asahi/pipeline: w3 {:#x} -> off {:#x}; abs {:#x}; usc+off {:#x}\n",
                                        mw[0], off, off, self.usc_exec_base + off
                                    );
                                    gpu.dump_vm(slot, off, n, "pipe-abs");
                                    gpu.dump_vm(slot, self.usc_exec_base + off, n, "pipe-usc");
                                    // 32.41.516: the USC reads at +0x200, not at
                                    // the object's base -- that is where .492's
                                    // redirect faulted (UL1C) and where .479
                                    // measured Apple's own 0x50-byte preamble.
                                    // The dump above is capped at 64 dwords so it
                                    // never reached it, which is why the ESL
                                    // program this driver actually emits has never
                                    // been looked at. Dump that window too, so it
                                    // can be tiled offline with isa-evidence/esl.py.
                                    gpu.dump_vm(slot, off + 0x200, n, "pipe-abs200");
                                    gpu.dump_vm(
                                        slot,
                                        self.usc_exec_base + off + 0x200,
                                        n,
                                        "pipe-usc200",
                                    );
                                }
                            }
                            // 32.41.475/.476: does G15 still accept G13 instruction
                            // encoding? .475 proved Apple's M3 code is NOT G13 ISA and
                            // that Mesa emits only G13 -- but not whether the G15
                            // decoder is backward compatible, which the captures cannot
                            // answer. Overwrite the USC pipeline blob (the object w3
                            // points at, which .475 confirmed is a preamble SHADER:
                            // every captured variant ends with exactly one 0e 00 00 00,
                            // the terminator that ends every shader in Apple's heap)
                            // with a hand-assembled G13 shader that stores a sentinel
                            // to a deliberately UNMAPPED address:
                            //
                            //   62 01 <lo32>   mov_imm r0, addr_lo, 0
                            //   62 05 <hi32>   mov_imm r1, addr_hi, 0
                            //   62 09 efbedec0 mov_imm r2, 0xc0debeef, 0
                            //   4511000500c01200  device_store 0,i32,x,r2,r0_r1,0,signed,lsl 0,0
                            //   88 00          stop
                            //
                            // assembled and round-tripped with dougallj/applegpu; bytes
                            // and repro in m3-gpu-display/isa-evidence/. A fault at
                            // EXACTLY g13_compat_addr proves the USC executed and
                            // decoded G13 code. The fault decoder is a validated
                            // instrument -- it has produced non-null results repeatedly
                            // (.466, .467) -- unlike the nulls that voided earlier runs.
                            //
                            // pipe_va is derived, not guessed: dump_pipeline (.474)
                            // proved `usc_exec_base + (w3 << 6)` is the readable
                            // candidate and read Mesa's record chain out of it.
                            let g13t = *crate::module_parameters::g13_compat_test.value();
                            if g13t != 0 {
                                let mut mw = [0u32; 1];
                                if gpu.read_vm(slot, cs + 0x18, &mut mw) == 1 {
                                    let pipe_va = self.usc_exec_base + ((mw[0] as u64) << 6);
                                    let target = *crate::module_parameters::g13_compat_addr.value();
                                    // ⚠️ The USC does NOT start reading at +0x00.
                                    // .467 measured the dereference at
                                    // `0x10000000000 + (w3 << 6) + 0x80` -- offset
                                    // +0x80 into the blob. The first version of this
                                    // test wrote the shader at +0x00 only and came back
                                    // null; that null was a design error, not evidence.
                                    // So tile the 28-byte shader across [0, 0x100) at
                                    // 0x20 steps (8 copies), and make copy k store to
                                    // `target + k*0x20` -- then the FAULT ADDRESS names
                                    // the entry offset instead of leaving it unknown.
                                    // .479: the USC reads (w3<<6) + 0x200, NOT +0x80.
                                    // .476 tiled only [0,0x100) -- covering .467's +0x80 --
                                    // so it tested content at offsets the USC never reads,
                                    // and its "contents are irrelevant" does NOT hold.
                                    // Widen to 0x400 so every offset the hardware has been
                                    // observed to touch carries our shader.
                                    const SPAN: u64 = 0x400;
                                    const STEP: u64 = 0x20;
                                    // arm 3 must NOT clobber Mesa's chain -- it works on
                                    // the original bytes.
                                    for k in 0..(if g13t >= 3 { 0 } else { SPAN / STEP }) {
                                        let off = k * STEP;
                                        let mut sh: [u8; 32] = [
                                            0x62, 0x01, 0x0d, 0xf0, 0xfe, 0xca,
                                            0x62, 0x05, 0xef, 0xbe, 0xad, 0xde,
                                            0x62, 0x09, 0xef, 0xbe, 0xde, 0xc0,
                                            0x45, 0x11, 0x00, 0x05, 0x00, 0xc0, 0x12, 0x00,
                                            0x88, 0x00,
                                            0x00, 0x00, 0x00, 0x00,
                                        ];
                                        if g13t == 2 {
                                            // control arm: invalid opcodes across the
                                            // whole span. If THIS changes nothing
                                            // either, the blob is not executed at all
                                            // and arm 1's null is uninterpretable.
                                            sh = [0xffu8; 32];
                                        } else {
                                            let t = target + off;
                                            sh[2..6].copy_from_slice(
                                                &((t & 0xffff_ffff) as u32).to_le_bytes(),
                                            );
                                            sh[8..12].copy_from_slice(
                                                &(((t >> 32) & 0xffff_ffff) as u32).to_le_bytes(),
                                            );
                                        }
                                        for i in 0..4 {
                                            let mut b = [0u8; 8];
                                            b.copy_from_slice(&sh[i * 8..i * 8 + 8]);
                                            gpu.poke_vm(
                                                slot,
                                                pipe_va + off + (i as u64) * 8,
                                                u64::from_le_bytes(b),
                                            );
                                        }
                                    }
                                    // arm 3: leave Mesa's record chain intact but
                                    // redirect every address-shaped word in it. .474
                                    // read the blob as {descriptor, ADDRESS} pairs with
                                    // every second word 0x2ffff88x -- Mesa's own
                                    // buffers. If the USC consumes the blob as a G13
                                    // RECORD CHAIN it must dereference those, so
                                    // rewriting 0x2ffff88x -> 0x2beef88x should fault at
                                    // an address containing "beef88". A null in arms 1
                                    // and 2 is ambiguous between "not read" and "read as
                                    // data whose contents happen to be benign"; this
                                    // separates them, and it tests .474's central claim
                                    // (that G15 wants CODE here) rather than assuming it.
                                    if g13t == 3 {
                                        let mut w = [0u32; 64];
                                        let got = gpu.read_vm(slot, pipe_va, &mut w);
                                        let mut hits = 0;
                                        for i in 0..got {
                                            if (w[i] & 0xffff_fff0) == 0x2fff_f880 {
                                                let nw = (w[i] & 0xf) | 0x2bee_f880;
                                                // poke_vm writes 8 bytes; preserve the
                                                // neighbour word in the same u64
                                                let pair = i & !1usize;
                                                if pair + 1 >= got { continue; }
                                                let mut lo = w[pair];
                                                let mut hi = w[pair + 1];
                                                if i == pair { lo = nw } else { hi = nw }
                                                gpu.poke_vm(
                                                    slot,
                                                    pipe_va + (pair as u64) * 4,
                                                    (lo as u64) | ((hi as u64) << 32),
                                                );
                                                w[i] = nw;
                                                hits += 1;
                                            }
                                        }
                                        pr_info!(
                                            "asahi/g13compat: arm 3 redirected {} address-shaped words (0x2ffff88x -> 0x2beef88x) in {} dwords at {:#x}\n",
                                            hits, got, pipe_va
                                        );
                                    }
                                    // arms 4/5: plant APPLE'S OWN captured G15 preamble.
                                    //
                                    // This is a substitution, not a sweep. .475 captured
                                    // the 0x50-byte object macOS puts at VDM State Update
                                    // word 3 for a plain draw; .481/.482 tiled it
                                    // completely as Apple "Ioto" instructions, and .482
                                    // established from `LoadShader::emit` that the address
                                    // fields it CAN carry are ZERO in every capture. That
                                    // is what makes this viable: the program is
                                    // self-contained, so it can be relocated into our VM
                                    // without any fixup. Nothing else we could put here
                                    // has that property.
                                    //
                                    // Offset: .479 measured the USC reading (w3<<6)+0x200,
                                    // so arm 4 writes there. Arm 5 also writes +0x000, to
                                    // separate "wrong offset" from "not executed" -- the
                                    // ambiguity that made .476's null uninterpretable.
                                    // arm 6: make the object byte-identical to Apple's.
                                    //
                                    // Offline check (.495): in ALL SIX captured word-3
                                    // objects, every byte past the ESL program is ZERO --
                                    // the whole 32 KiB dump. Apple's object is "the program,
                                    // then nothing".
                                    //
                                    // .486 wrote Apple's 0x50 bytes at +0x00 but left Mesa's
                                    // record chain in place at +0x80 and +0x100, so what the
                                    // hardware saw was Apple's program PLUS foreign records --
                                    // not Apple's object. Zero the region first, then plant.
                                    // arm 7 (.518): arm 6, but with the LoadShader core
                                    // retargeted to OUR vertex shader.
                                    //
                                    // Arm 6 plants Apple's program verbatim, so its
                                    // LoadShader word 1 still carries Apple's own shader
                                    // address (0x10000000780) -- an address that means
                                    // nothing in our VM. .518 read the field out of
                                    // `LoadShader::emit` @0x68e020: the 28-byte core is
                                    // written by three stores, word 0 (+0x00) from
                                    // [this+76], word 1 (+0x08) from [this+8], word 2
                                    // (+0x12) from [this+40]. .481 read [this+40] and
                                    // reported "the address fields are zero"; that is
                                    // word 2. The shader address is word 1:
                                    //
                                    //   h1 bits 15..7 = A bits 14..6   h2 = A bits 30..15
                                    //   h3 = A bits 46..31             h4 bit 0 = A bit 47
                                    //
                                    // A bits 5..0 are never emitted -- 64-byte aligned.
                                    // Validated on all 11 captured cores; the 138000+0x640
                                    // core decodes to 0x100000006c0, which .516 found by an
                                    // unrelated method.
                                    //
                                    // The address is NOT guessed here. Mesa's own object,
                                    // which we are about to overwrite, is a well-formed G13
                                    // USC word list and it tiles exactly (56 bytes:
                                    // Sampler/Uniform x4/Shared/SHADER/Registers/No
                                    // preshader). The SHADER word (tag 0x0d, 6 bytes,
                                    // genxml `USC Shader`) carries `Code` as a u32 at byte
                                    // offset 2 -- the USC-relative byte offset of the vertex
                                    // shader, the same space as Mesa's 0xfff68000 pipeline
                                    // word. So: read it out, then form the address the
                                    // HARDWARE uses, against the same base it forms pipe_va
                                    // against (CODE_GART_BASE + (w3 << 6), .464/.467), not
                                    // against our own usc_exec_base view.
                                    // arm 10 (.523): poison the USC Shader `Code` in BOTH
                                    // pipeline objects, with DISTINCT values, in one boot.
                                    //
                                    // .522 poisoned only +0x000 and saw no fault -- but the
                                    // only offset the hardware has ever been PROVEN to read is
                                    // +0x200 (.479, by bumping w3 and taking a fault there,
                                    // during the VERTEX job).  .522 therefore poisoned an
                                    // object the USC was never shown to touch.  .479's own text
                                    // warns "an offset measured once, in one configuration, is
                                    // not THE offset"; this is that mistake mirrored.
                                    //
                                    // .519 tiled both objects exactly as USC word lists:
                                    //   +0x000  Sampler/Uniform x4/Shared/SHADER(0xfff60000)/
                                    //           Registers/No preshader          56 bytes
                                    //   +0x200  Texture/Shared/SHADER(0xfffbc100)/Registers/
                                    //           No preshader                    24 bytes
                                    // Distinct poisons mean a fault names WHICH object is live.
                                    if g13t == 10 {
                                        // ALIAS GATE (.523/.419).
                                        //
                                        // The hardware does not read the object through our
                                        // pipe_va; it reads through the win36 alias
                                        // win36_base | (pipe_va & 0xffffffff).  That alias can
                                        // be LOST to another BO sharing the low 32 bits (.419),
                                        // and in .523 it was -- the USC faulted at
                                        // CODE_GART_BASE + (w3<<6) and never saw the object, so
                                        // the whole arm measured nothing.  Any poison arm whose
                                        // conclusion depends on the hardware READING the object
                                        // is void unless the alias is verified live.  Gate on it
                                        // first, and say so loudly.
                                        let hw_alias = if *crate::module_parameters::user_va_win36
                                            .value()
                                            != 0
                                        {
                                            *crate::module_parameters::win36_base.value()
                                                | (pipe_va & 0xffff_ffff)
                                        } else {
                                            pipe_va
                                        };
                                        let mut ab = [0u32; 4];
                                        let an = gpu.read_vm(slot, hw_alias, &mut ab);
                                        if an == 0 {
                                            pr_info!(
                                                "asahi/aliasgate: *** VOID *** hw alias {:#x} UNREADABLE -- the USC cannot see the object; poison skipped\n",
                                                hw_alias
                                            );
                                        } else {
                                            pr_info!(
                                                "asahi/aliasgate: hw alias {:#x} OK [{}] {:08x} {:08x} {:08x} {:08x}\n",
                                                hw_alias, an, ab[0], ab[1], ab[2], ab[3]
                                            );
                                        }
                                        for (base, poison) in [
                                            (0u64, 0xdead_0000u32),
                                            (0x200u64, 0xfeed_0000u32),
                                        ] {
                                            let mut ub = [0u32; 32];
                                            let un = gpu.read_vm(slot, pipe_va + base, &mut ub);
                                            let mut bb = [0u8; 128];
                                            for i in 0..un {
                                                bb[i * 4..i * 4 + 4]
                                                    .copy_from_slice(&ub[i].to_le_bytes());
                                            }
                                            let nb = un * 4;
                                            let mut q = 0usize;
                                            let mut rec: Option<usize> = None;
                                            let mut old = 0u32;
                                            while q + 2 <= nb {
                                                let (len, is_shader) = match bb[q] {
                                                    0x0d => (6usize, true),
                                                    0x1d | 0x3d | 0x9d | 0xdd | 0x38 | 0x58 => (8, false),
                                                    0x4d | 0x8d => (4, false),
                                                    0x88 => (2, false),
                                                    _ => (0, false),
                                                };
                                                if len == 0 || q + len > nb {
                                                    break;
                                                }
                                                if is_shader {
                                                    old = u32::from_le_bytes([
                                                        bb[q + 2], bb[q + 3], bb[q + 4], bb[q + 5],
                                                    ]);
                                                    rec = Some(q);
                                                    break;
                                                }
                                                q += len;
                                            }
                                            match rec {
                                                Some(q0) => {
                                                    let pb = poison.to_le_bytes();
                                                    for k in 0..4 {
                                                        bb[q0 + 2 + k] = pb[k];
                                                    }
                                                    for g in ((q0 + 2) / 8)..=((q0 + 5) / 8) {
                                                        let mut w8 = [0u8; 8];
                                                        w8.copy_from_slice(&bb[g * 8..g * 8 + 8]);
                                                        gpu.poke_vm(
                                                            slot,
                                                            pipe_va + base + (g as u64) * 8,
                                                            u64::from_le_bytes(w8),
                                                        );
                                                    }
                                                    let mut rb = [0u32; 4];
                                                    let n = gpu.read_vm(
                                                        slot,
                                                        pipe_va + base + ((q0 / 4) as u64) * 4,
                                                        &mut rb,
                                                    );
                                                    let mut arb = [0u32; 4];
                                                    let arn = gpu.read_vm(
                                                        slot,
                                                        hw_alias + base + ((q0 / 4) as u64) * 4,
                                                        &mut arb,
                                                    );
                                                    pr_info!(
                                                        "asahi/uscpoison2: +{:#05x} SHADER at +{:#x} Code {:#x} -> {:#x}; readback[{}] {:08x} {:08x} {:08x} {:08x}; ALIAS[{}] {:08x} {:08x} {:08x} {:08x}\n",
                                                        base, q0, old, poison, n,
                                                        rb[0], rb[1], rb[2], rb[3],
                                                        arn, arb[0], arb[1], arb[2], arb[3]
                                                    );
                                                }
                                                None => {
                                                    pr_info!(
                                                        "asahi/uscpoison2: +{:#05x} *** VOID *** no SHADER record in {} bytes\n",
                                                        base, nb
                                                    );
                                                }
                                            }
                                        }
                                    }
                                    let mut esl_shader_va = 0u64;
                                    if g13t == 7 || g13t == 8 || g13t == 9 {
                                        let mut ub = [0u32; 32];
                                        let un = gpu.read_vm(slot, pipe_va, &mut ub);
                                        let mut ubytes = [0u8; 128];
                                        for i in 0.. un {
                                            ubytes[i * 4..i * 4 + 4]
                                                .copy_from_slice(&ub[i].to_le_bytes());
                                        }
                                        let nb = un * 4;
                                        // Walk the USC word list by tag. Lengths are
                                        // genxml's, not inferred.
                                        let mut q = 0usize;
                                        let mut code: Option<u32> = None;
                                        let mut shader_rec: Option<usize> = None;
                                        while q + 2 <= nb {
                                            let (len, is_shader) = match ubytes[q] {
                                                0x0d => (6usize, true),          // Shader
                                                0x1d | 0x3d | 0x9d | 0xdd => (8, false),
                                                0x4d | 0x8d => (4, false),
                                                0x38 => (8, false),              // Preshader
                                                0x58 => (8, false),              // Frag props
                                                0x88 => (2, false),              // No preshader
                                                _ => (0, false),
                                            };
                                            if len == 0 || q + len > nb {
                                                break;
                                            }
                                            if is_shader {
                                                shader_rec = Some(q);
                                                code = Some(u32::from_le_bytes([
                                                    ubytes[q + 2],
                                                    ubytes[q + 3],
                                                    ubytes[q + 4],
                                                    ubytes[q + 5],
                                                ]));
                                                break;
                                            }
                                            q += len;
                                        }
                                        // arm 9 (.522): POISON the USC Shader record's Code.
                                        //
                                        // .521 proved the VDM executes the draw record, and
                                        // .479/.492 proved the USC fetches the pipeline object
                                        // -- yet .520 showed an Ioto program planted there is
                                        // never executed.  The reading that fits all three is
                                        // that the G15 USC consumes the G13-style USC WORD LIST
                                        // Mesa already writes, and the ESL program is something
                                        // Apple's stack uses that the hardware does not require.
                                        //
                                        // Test it the same way everything else in this sequence
                                        // was tested: make the shader address unreachable.  If
                                        // the USC loads the shader through the word list it
                                        // faults and the address names it; if it does not fault,
                                        // the word list is not what feeds the shader either and
                                        // BOTH candidate paths are eliminated.
                                        //
                                        // Nothing is zeroed and nothing is planted -- only the
                                        // four Code bytes change, so this is the plain baseline
                                        // plus one field.
                                        if g13t == 9 {
                                            let poison = *crate::module_parameters::usc_code_poison
                                                .value();
                                            match shader_rec {
                                                Some(q0) if poison != 0 => {
                                                    let pb = poison.to_le_bytes();
                                                    for k in 0..4 {
                                                        ubytes[q0 + 2 + k] = pb[k];
                                                    }
                                                    // write back the 8-byte groups the Code
                                                    // field straddles
                                                    let g0 = (q0 + 2) / 8;
                                                    let g1 = (q0 + 5) / 8;
                                                    for g in g0..=g1 {
                                                        let mut b = [0u8; 8];
                                                        b.copy_from_slice(&ubytes[g * 8..g * 8 + 8]);
                                                        gpu.poke_vm(
                                                            slot,
                                                            pipe_va + (g as u64) * 8,
                                                            u64::from_le_bytes(b),
                                                        );
                                                    }
                                                    let mut rb = [0u32; 4];
                                                    let n = gpu.read_vm(
                                                        slot,
                                                        pipe_va + ((q0 / 4) as u64) * 4,
                                                        &mut rb,
                                                    );
                                                    pr_info!(
                                                        "asahi/uscpoison: USC Shader record at +{:#x}, Code {:#x} -> {:#x}; readback[{}] {:08x} {:08x} {:08x} {:08x}\n",
                                                        q0, code.unwrap_or(0), poison, n,
                                                        rb[0], rb[1], rb[2], rb[3]
                                                    );
                                                }
                                                _ => {
                                                    pr_info!(
                                                        "asahi/uscpoison: *** VOID *** shader_rec {} poison {:#x}\n",
                                                        shader_rec.unwrap_or(usize::MAX), poison
                                                    );
                                                }
                                            }
                                        }
                                        match code {
                                            Some(c) => {
                                                // The base the hardware forms pipe_va
                                                // against. With user_va_win36 on (the
                                                // baseline) that is win36_base; otherwise
                                                // our own usc_exec_base.
                                                let hw_base = if *crate::module_parameters::user_va_win36
                                                    .value()
                                                    != 0
                                                {
                                                    *crate::module_parameters::win36_base.value()
                                                } else {
                                                    self.usc_exec_base
                                                };
                                                esl_shader_va = hw_base + (c as u64);
                                                // arm 8 (.520): the DISCRIMINATOR arm 7 lacked.
                                                // Arm 7 planted a byte-verified, correctly
                                                // retargeted program and got a clean null --
                                                // but a null there cannot separate "the USC
                                                // executed this and it did nothing useful"
                                                // from "the USC never executed it at all".
                                                // That is exactly the ambiguity that voided
                                                // .476.  So point LoadShader at a deliberately
                                                // UNMAPPED, distinctive VA: if the USC runs
                                                // Ioto programs and follows LoadShader, the
                                                // existing fault decoder prints that address
                                                // (positive control: .479 and .492 both
                                                // produced pipeline-object faults).
                                                if g13t == 8 {
                                                    esl_shader_va = *crate::module_parameters::g13_compat_addr
                                                        .value()
                                                        & !0x3fu64;
                                                }
                                                let aligned: u32 =
                                                    if esl_shader_va & 0x3f == 0 { 1 } else { 0 };
                                                pr_info!(
                                                    "asahi/esl7: USC Shader Code {:#x} hw_base {:#x} -> shader VA {:#x} (aligned {})\n",
                                                    c,
                                                    hw_base,
                                                    esl_shader_va,
                                                    aligned
                                                );
                                            }
                                            None => {
                                                pr_info!(
                                                    "asahi/esl7: *** VOID *** no USC Shader (tag 0x0d) record in {} bytes at pipe_va\n",
                                                    nb
                                                );
                                            }
                                        }
                                    }
                                    if g13t == 6
                                        || ((g13t == 7 || g13t == 8) && esl_shader_va != 0)
                                    {
                                        for q in 0..(0x400u64 / 8) {
                                            gpu.poke_vm(slot, pipe_va + q * 8, 0);
                                        }
                                        pr_info!(
                                            "asahi/g13compat: arm {} zeroed pipe_va[0,0x400)\n",
                                            g13t
                                        );
                                    }
                                    if g13t == 4
                                        || g13t == 5
                                        || g13t == 6
                                        || ((g13t == 7 || g13t == 8) && esl_shader_va != 0)
                                    {
                                        // Apple's captured draw ESL state load program,
                                        // res-draw-pre/gpu-010000128000.bin, 0x50 bytes.
                                        const ESL_DRAW: [u8; 80] = [
                                            0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00,
                                            0x77, 0x01, 0x2a, 0x0f, 0x00, 0x00, 0x00, 0x02,
                                            0x00, 0x00, 0xf7, 0x00, 0x2a, 0x00, 0x01, 0x00,
                                            0x00, 0x00, 0x00, 0x00, 0x1c, 0x80, 0x02, 0x00,
                                            0x00, 0x00, 0x00, 0x00, 0x14, 0x81, 0x11, 0x06,
                                            0x00, 0x00, 0x00, 0x00, 0x0c, 0x80, 0x02, 0x00,
                                            0x10, 0x00, 0x00, 0x00, 0x9f, 0x11, 0x54, 0x00,
                                            0x02, 0x00, 0x08, 0xa8, 0x10, 0x05, 0x1c, 0x80,
                                            0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0f, 0x12,
                                            0x54, 0x00, 0x4c, 0x00, 0x0e, 0x00, 0x00, 0x00,
                                        ];
                                        // arm 7: retarget LoadShader word 1 (+0x08) to our
                                        // own vertex shader.  Only the address bits move;
                                        // h1 bits 6..1 are three 2-bit mode fields OR'd in
                                        // after emit's 0xffffff81 mask, not address, and h0
                                        // is not address either (.518).
                                        let mut esl = ESL_DRAW;
                                        if g13t == 7 || g13t == 8 {
                                            let a = esl_shader_va;
                                            let hw = |o: usize| -> u16 {
                                                u16::from_le_bytes([esl[8 + o * 2], esl[8 + o * 2 + 1]])
                                            };
                                            let h1 = (hw(1) & 0x007f)
                                                | ((((a >> 6) & 0x1ff) as u16) << 7);
                                            let h2 = ((a >> 15) & 0xffff) as u16;
                                            let h3 = ((a >> 31) & 0xffff) as u16;
                                            let h4 = (hw(4) & 0xfffe) | ((a >> 47) & 1) as u16;
                                            for (o, v) in [(1usize, h1), (2, h2), (3, h3), (4, h4)] {
                                                let b = v.to_le_bytes();
                                                esl[8 + o * 2] = b[0];
                                                esl[8 + o * 2 + 1] = b[1];
                                            }
                                            pr_info!(
                                                "asahi/esl7: LoadShader word 1 retargeted to {:#x}: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}\n",
                                                a,
                                                esl[8], esl[9], esl[10], esl[11], esl[12],
                                                esl[13], esl[14], esl[15], esl[16], esl[17]
                                            );
                                        }
                                        let bases = if g13t == 6 || g13t == 7 || g13t == 8 {
                                            [0u64, 0u64]
                                        } else {
                                            [0x200u64, 0u64]
                                        };
                                        let nplant = if g13t == 5 { 2 } else { 1 };
                                        for bi in 0..nplant {
                                            let base = bases[bi];
                                            for i in 0..(esl.len() / 8) {
                                                let mut b = [0u8; 8];
                                                b.copy_from_slice(&esl[i * 8..i * 8 + 8]);
                                                gpu.poke_vm(
                                                    slot,
                                                    pipe_va + base + (i as u64) * 8,
                                                    u64::from_le_bytes(b),
                                                );
                                            }
                                        }
                                        let where_ = match g13t {
                                            5 => " +0x000 and +0x200",
                                            6 => " +0x000, on a zeroed object (Apple's shape)",
                                            7 => " +0x000, zeroed, LoadShader retargeted to our VS",
                                            8 => " +0x000, zeroed, LoadShader -> UNMAPPED (execution probe)",
                                            _ => " +0x200",
                                        };
                                        pr_info!(
                                            "asahi/g13compat: arm {} planted Apple's 0x50-byte draw ESL program at pipe_va{}\n",
                                            g13t, where_
                                        );
                                        // read it back at +0x200 through the alias the
                                        // hardware actually forms, not just through our
                                        // own mapping (.419: the win36 alias can be lost
                                        // to another BO sharing the low 32 bits).
                                        let rb_off: u64 =
                                            if g13t == 6 || g13t == 7 || g13t == 8 { 0 } else { 0x200 };
                                        let mut eb = [0u32; 8];
                                        let ne = gpu.read_vm(slot, pipe_va + rb_off, &mut eb);
                                        pr_info!(
                                            "asahi/g13compat: ESL@+{:#05x}[{}] {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
                                            rb_off, ne, eb[0], eb[1], eb[2], eb[3], eb[4], eb[5], eb[6], eb[7]
                                        );
                                    }
                                    // read back, so a failed poke cannot masquerade as a
                                    // clean negative the way cs_draw_ctrl_exact did
                                    let mut rb = [0u32; 8];
                                    let n = gpu.read_vm(slot, pipe_va, &mut rb);
                                    let mut rb80 = [0u32; 8];
                                    let n80 = gpu.read_vm(slot, pipe_va + 0x80, &mut rb80);
                                    pr_info!(
                                        "asahi/g13compat: arm {} w3 {:#x} pipe_va {:#x} target {:#x} (readback follows)\n",
                                        g13t, mw[0], pipe_va, target
                                    );
                                    pr_info!(
                                        "asahi/g13compat: +0x00[{}] {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
                                        n, rb[0], rb[1], rb[2], rb[3], rb[4], rb[5], rb[6], rb[7]
                                    );
                                    pr_info!(
                                        "asahi/g13compat: +0x80[{}] {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
                                        n80, rb80[0], rb80[1], rb80[2], rb80[3], rb80[4], rb80[5], rb80[6], rb80[7]
                                    );
                                    // ⚠ .476 verified the write at pipe_va and ARGUED the
                                    // code-GART alias is the same page. It never READ the alias.
                                    // .419 records exactly this failure mode: the win36 alias is
                                    // `win36_base | (VA & 0xffffffff)` and two BOs sharing low
                                    // bits fight over it -- the USC code BO at 0x11fffbc000 once
                                    // LOST, leaving a zero page where the shader should have been.
                                    // The address the HARDWARE forms is CODE_GART_BASE + (w3<<6).
                                    // Read THAT. If it does not match, .476's "the blob is inert"
                                    // actually meant "we were writing to the wrong page".
                                    let alias = *crate::module_parameters::win36_base.value()
                                        | (pipe_va & 0xffff_ffff);
                                    let mut al = [0u32; 8];
                                    let na = gpu.read_vm(slot, alias, &mut al);
                                    pr_info!(
                                        "asahi/g13compat: ALIAS {:#x}[{}] {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
                                        alias, na, al[0], al[1], al[2], al[3], al[4], al[5], al[6], al[7]
                                    );
                                    // .479: the USC was measured reading (w3<<6)+0x200. Read back
                                    // every 0x100 across the tiled span so a null can be told from
                                    // a write that never landed at the offset that matters.
                                    for q in 1..4u64 {
                                        let mut qq = [0u32; 4];
                                        let nq = gpu.read_vm(slot, pipe_va + q * 0x100, &mut qq);
                                        pr_info!(
                                            "asahi/g13compat: +{:#05x}[{}] {:08x} {:08x} {:08x} {:08x}\n",
                                            q * 0x100, nq, qq[0], qq[1], qq[2], qq[3]
                                        );
                                    }
                                    gpu.probe_vm(slot, alias, "pipe-alias");
                                }
                            }
                            let pp = *crate::module_parameters::probe_ptr.value();
                            if pp != 0 {
                                let va = (cs & !0xffff_ffffu64) | (pp as u64);
                                gpu.probe_vm(slot, va, "ptr");
                                gpu.probe_vm(slot, va + (1u64 << 36), "ptr ALIAS");
                            }
                            // Absolute form. `probe_ptr` rebuilds the VA from the
                            // control stream's own high half, so it can only reach
                            // addresses sharing it -- it cannot express e.g.
                            // 0x12fff68000, the +2^36-biased pipeline address the
                            // record at cs+0x10 carries. LOG.md 32.41.183.
                            let pv = *crate::module_parameters::probe_va.value();
                            if pv != 0 {
                                gpu.probe_vm(slot, pv, "va");
                                gpu.probe_vm(slot, pv.wrapping_sub(1u64 << 36), "va -2^36");
                                gpu.probe_vm(slot, pv.wrapping_add(1u64 << 36), "va +2^36");
                            }
                            gpu.probe_vm(slot, cs, "vdm cs+0");
                            // 🔑 The GPU does not read the primary VA. It
                            // faults in *alias* space (0x3ffff84080 = the
                            // faulting address, = cs + 0x80080 + 2^36), so the
                            // stream it actually fetches is the `+2^36` alias
                            // `step_map` creates, not the address probed
                            // above. Those are supposed to be the same
                            // physical pages; the scene coming back all-zero
                            // while the primary carries real opcodes
                            // (32.41.135) says that assumption needs checking
                            // rather than trusting.
                            let bias = 1u64 << *crate::module_parameters::user_va_shift.value();
                            gpu.probe_vm(slot, cs + bias, "vdm cs+0 ALIAS");
                            gpu.probe_vm(slot, cs + 0x40 + bias, "vdm cs+0x40 ALIAS");
                            gpu.probe_vm(slot, cs + 0x40, "vdm cs+0x40");
                            gpu.probe_vm(slot, cs + 0x80, "vdm cs+0x80");
                            gpu.probe_vm(slot, cs + 0x7ffc0, "vdm cs+0x7ffc0");
                            gpu.probe_vm(slot, cs + 0x80000, "vdm cs+0x80000");
                            // Did the VDM do any *work*, or only walk? The two
                            // readings of the end-of-stream fault (benign
                            // read-ahead vs. decoding zeroes, 32.41.131) differ
                            // exactly here: a vertex that executed writes its
                            // tile lists and scene metadata, one that decoded
                            // 512 KiB of zeroes leaves them at their initial
                            // contents. These are the addresses `asahi/addr`
                            // already prints, so they cost nothing to reach.
                            gpu.probe_vm(slot, u64::from(scene.tvb_tilemap_pointer()), "scene tilemap");
                            gpu.probe_vm(slot, u64::from(scene.tvb_tilemap_pointer()) + bias, "scene tilemap ALIAS");
                            gpu.probe_vm(slot, u64::from(scene.tvb_heapmeta_pointer()), "scene heapmeta");
                            gpu.probe_vm(slot, u64::from(scene.tvb_layermeta_pointer()), "scene layermeta");
                        }
                        mod_dev_dbg!(
                            self.dev,
                            "asahi/usc frg: bg_usc {:#x} bg_rsrc {:#x} eot_usc {:#x} eot_rsrc {:#x} pbg_usc {:#x} peot_usc {:#x} sampler {:#x}/{} vtx_scratch {} flags {:#x}\n",
                            u64::from(cmdbuf.bg.usc),
                            u64::from(cmdbuf.bg.rsrc_spec),
                            u64::from(cmdbuf.eot.usc),
                            u64::from(cmdbuf.eot.rsrc_spec),
                            u64::from(cmdbuf.partial_bg.usc),
                            u64::from(cmdbuf.partial_eot.usc),
                            cmdbuf.sampler_heap,
                            cmdbuf.sampler_count,
                            cmdbuf.flags
                                & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_VERTEX_SCRATCH as u32,
                            cmdbuf.flags,
                        );
                        // The hardware walked TTBAT[3] for a job bound to
                        // TTBAT[1] (LOG.md 32.41.38).  Nothing the driver puts
                        // in StartVertex names a context on 15.5, but the
                        // firmware appends 0x10209/0x1c9f0/0x14320 with
                        // `(unk_44 & 0x3f) | (event_generation << 8)`, and
                        // `event_generation` is the queue id -- so print the ids
                        // next to the slot to see whether the 3 tracks one of
                        // them.  LOG.md 32.41.39.
                        mod_dev_dbg!(
                            self.dev,
                            "asahi/ids vtx: queue_id {} vm_slot {} buffer_slot {} event_seq {:#x}\n",
                            self.id,
                            vm_bind.slot(),
                            scene.slot(),
                            ev_vtx.event_seq,
                        );
                        // 32.41.510 GATE: print the (index, enable) pair that
                        // actually reaches MMIO 0xd8f0, so a run can prove the
                        // change landed rather than assuming it.
                        #[ver(V >= V15_5)]
                        {
                            let m = *crate::module_parameters::vtx_d8f0.value();
                            let idx = if m == 2 { 0xb } else { vm_bind.slot() };
                            let en = if m != 0 { 1 } else { scene.slot() };
                            let tag = if en != 0 { "SET" } else { "CLEAR" };
                            // pr_info!, not mod_dev_dbg!: the neighbouring
                            // `asahi/ids vtx` line is mod_dev_dbg and emitted
                            // ZERO times in the .510 run, which voided the
                            // gate.  A gate that can be silently suppressed is
                            // not a gate.
                            // 32.41.511 GATE: prove the corrected RunVertex
                            // layout is the one in this module.
                            crate::sub_pr_info!(
                                "asahi/rvlayout: size {:#x} command_time {:#x} tsp {:#x} unk_ts {:#x}\n",
                                core::mem::size_of::<crate::fw::vertex::raw::RunVertex::ver<'static>>(),
                                core::mem::offset_of!(crate::fw::vertex::raw::RunVertex::ver<'static>, command_time),
                                core::mem::offset_of!(crate::fw::vertex::raw::RunVertex::ver<'static>, timestamp_pointers),
                                core::mem::offset_of!(crate::fw::vertex::raw::RunVertex::ver<'static>, unk_ts),
                            );
                            crate::sub_pr_info!(
                                "asahi/d8f0: mode {} index {} enable {} -> {} bit {}\n",
                                m,
                                idx,
                                en,
                                tag,
                                idx,
                            );
                        }
                        let start_vtx = builder.add(microseq::StartVertex::ver {
                            #[ver(V >= V15_5)]
                            header: microseq::op::StartVertex::HEADER_V15_5,
                            #[ver(V < V15_5)]
                            header: microseq::op::StartVertex::HEADER,
                            #[ver(G < G14X)]
                            tiling_params: Some(inner_weak_ptr!(ptr, tiling_params)),
                            #[ver(G < G14X)]
                            job_params1: Some(inner_weak_ptr!(ptr, job_params1)),
                            #[ver(G >= G14X)]
                            tiling_params: None,
                            #[ver(G >= G14X)]
                            job_params1: None,
                            #[ver(G >= G14X)]
                            registers: inner_weak_ptr!(ptr, registers),
                            buffer: scene.weak_buffer_pointer(),
                            scene: scene.weak_pointer(),
                            stats,
                            work_queue: ev_vtx.info_ptr,
                            #[ver(V < V15_5)]
                            vm_slot: vm_bind.slot(),
                            #[ver(V < V15_5)]
                            unk_38: 1, // fixed
                            #[ver(V >= V15_5)]
                            unk_3c_ptr: inner_weak_ptr!(ptr, sync_grow),
                            // 🔑 On 15.5 the two u32s the firmware reads out of
                            // StartVertex for the address space are swapped
                            // relative to the pre-15.5 names this struct still
                            // carries.  The firmware appends registers 0x10209,
                            // 0x1c9f0 and 0x14320 with
                            //     (opcode[0x4c] & 0x3f) | (opcode[0x44] << 8)
                            // (fw:0x15000).  StartFragment, whose layout is
                            // unchanged, gets the identical treatment at
                            // fw:0x15b80 -- masked field at 0x5c, shifted at
                            // 0x54 -- and 0x54 is `vm_slot`, 0x5c is
                            // `event_generation`.  So the shifted field is the
                            // address space and the masked one is the queue id,
                            // which m1n1's macOS trace confirms: it captured
                            // 0x133 = (51 & 0x3f) | (1 << 8), i.e. queue 51 in
                            // VM slot 1.
                            //
                            // For vertex the driver was writing the *queue id*
                            // into 0x44, so the GPU walked TTBAT[queue_id]: the
                            // run in LOG.md 32.41.38 bound TTBAT[1] and faulted
                            // with vm_slot 3 because that job's queue id was 3.
                            // It was never a "+2".
                            #[ver(V < V15_5)]
                            event_generation: self.id as u32,
                            // 🔑🔑 32.41.510: on 15.5 these two are the
                            // (bit index, ENABLE) pair for MMIO 0xd8f0.
                            // fw:0x15088 `ldp w8, w9, [x28,#64]` (x28 =
                            // opcode+4) then `lsl x8, x27, x8` / `orr` /
                            // `bic` / `cmp w9,#0; csel ..., eq` -- w9 == 0
                            // CLEARS the bit.  `buffer_slot` is NOT a slot
                            // here; the firmware only ever tests it against
                            // zero.  scene.slot() is 0 for us, so every
                            // vertex job so far has cleared bit vm_slot
                            // rather than enabling anything.  Apple's own
                            // captured TA microsequence (both draw and
                            // clear) has 0x44 = 0xb, 0x48 = 1.
                            #[ver(V >= V15_5)]
                            event_generation: if *crate::module_parameters::vtx_d8f0.value() == 2 {
                                0xb
                            } else {
                                vm_bind.slot()
                            },
                            #[ver(V < V15_5)]
                            buffer_slot: scene.slot(),
                            #[ver(V >= V15_5)]
                            buffer_slot: if *crate::module_parameters::vtx_d8f0.value() != 0 {
                                1
                            } else {
                                scene.slot()
                            },
                            #[ver(V < V15_5)]
                            unk_44: 0,
                            // 32.41.600: low 6 bits of the fw-appended 0x10209/
                            // 0x1c9f0/0x14320 (`vm_slot << 8 | this & 0x3f`).
                            // Apple's captured StartVertex+0x4c is 0 (value 0xb00).
                            #[ver(V >= V15_5)]
                            unk_44: if *crate::module_parameters::vtx_qid_zero.value() != 0 {
                                0
                            } else {
                                self.id as u32
                            },
                            // 32.41.641: on 15.5 the firmware reads op+0x50 as the BUFFER
                            // SLOT: fw:0x150c0 stores it in the per-hw-slot map
                            // 0x1a1e20+0x3f8, and the slot state machine (fw:0x3874)
                            // resolves it through the table fw:0x1ce04 fills from
                            // RunVertex.buffer_slot. Apple: 0xd here = its buffer slot.
                            // Ours was event_seq (0), matching only while slot == 0 --
                            // why buf_slot_min=13 crashed in .630.
                            // 32.41.664: always the slot -- with buf_slot_min=0 this was
                            // event_seq, and glmark2's 2nd frame (event_seq 1, slot 0)
                            // crashed at fw:0x389c on the NULL table[1].
                            #[ver(V >= V15_5)]
                            event_seq: U64(scene.slot() as u64),
                            #[ver(V < V15_5)]
                            event_seq: U64(ev_vtx.event_seq),
                            unk_50: 0,
                            #[ver(V >= V15_5)]
                            unk_5c: 0,
                            #[ver(V >= V15_5)]
                            unk_60: 0,
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            // Apple's +0x6c = registers + 0x858 = cmd 0x898 (LOG 32.41.598).
                            #[ver(V >= V15_5)]
                            unk_job_buf: inner_weak_ptr!(ptr, unk_buf_10),
                            #[ver(V < V15_5)]
                            unk_job_buf: inner_weak_ptr!(ptr, unk_buf_0),
                            unk_64: 0x0, // fixed
                            unk_68: unk1.into(),
                            uuid: uuid_ta,
                            #[ver(V >= V15_5)]
                            unk_80: 0,
                            #[ver(V >= V15_5)]
                            unk_84: 0,
                            attachments: {
                                // 🔬 The firmware builds a *second*, 16-byte-entry
                                // list in the `RegisterArray`'s pad region
                                // (`fw:0x15110`, cursor `g[440]`, count `g[424]`)
                                // and hands it to the hardware with its own MMIO
                                // write at `hw_base + 0xc238` -- but only when
                                // that count is non-zero (`fw:0x200f4`).  The
                                // list is built from the attachments, so with
                                // none the hardware gets no descriptors at all.
                                // LOG.md 32.41.267.
                                crate::sub_pr_info!(
                                    "asahi/attachments: vertex count {}\n",
                                    vertex_attachments.count
                                );
                                // 32.41.636: an attachment is a hardware write-back range for the
                                // job (.635: without the fragment's RT entry the RT stayed
                                // invisible). The TA had none, so nothing forced its TVB/TMAP/HMTA
                                // writes out before the ISP read them. vtx_att_tvb=1: add the 2 MiB
                                // window holding the TVB blocks and scene buffers.
                                let mut a = *vertex_attachments;
                                if *crate::module_parameters::vtx_att_tvb.value() != 0
                                    && (a.count as usize) < microseq::MAX_ATTACHMENTS
                                {
                                    let base = u64::from(scene.tvb_tilemap_pointer()) & !0x1f_ffffu64;
                                    let size: u64 = 0x20_0000;
                                    a.list[a.count as usize] = microseq::Attachment {
                                        address: U64(base),
                                        size: (size >> 7) as u32,
                                        unk_c: 0x17,
                                        unk_e: 1,
                                    };
                                    a.count += 1;
                                    crate::sub_pr_info!("asahi/attachments: vtx_att_tvb {:#x} +{:#x} (count {})\n", base, size, a.count);
                                }
                                a
                            },
                            padding: 0,
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            counter: U64(count_vtx),
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            notifier_buf: inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf),
                            #[ver(V < V13_0B4)]
                            unk_178: 0x0, // padding?
                            #[ver(V >= V13_0B4 && V < V15_5)]
                            unk_178: (!clustering) as u32,
                            // The firmware dereferences this one: 0x24e34 does
                            // `ldr x26, [x0]; ldr w8, [x26, #8]; cmp w8, #0xff`
                            // on `opcode + 0x190`.  A zeroed pool reads back 0
                            // and takes the early path.
                            #[ver(V >= V15_5)]
                            pool_ptr: U64(scene.uma_pool_pointer().into()),
                            // 🔑 `fw:0x169cc` reads this (`add x8, x28, #0x194`,
                            // x28 = opcode + 4) as the **UMA pool base**, masks
                            // it with 0x7ffffffffc0 and appends it as register
                            // 0xa5c9.  `fw:0x24ed8` branches on it being zero,
                            // and the null path at `fw:0x24f78` still emits the
                            // pool base register -- with no address in it.  A
                            // pool based at VA 0 is exactly what `gUPM` faults
                            // on.  LOG.md 32.41.59/60.
                            // The UMA pool base. `fw:0x24edc` folds it into the
                            // appended pool register as `(base << 4) &
                            // 0x7ffffffffc00`, and `fw:0x24f78` -- the null path
                            // -- ORs in the *sign-extended* pool id instead, so
                            // a null base with id -1 emits all-ones. Base and id
                            // have to be consistent: either a real pool (base +
                            // id 0..0xfe) or none at all. LOG.md 32.41.65.
                            #[ver(V >= V15_5)]
                            unk_198: U64(match *crate::module_parameters::uma_base_probe.value() {
                                // LOG.md 32.41.536: Apple's StartVertex+0x198 is 0.
                                _ if *crate::module_parameters::uma_apple.value() != 0 => 0,
                                0 if *crate::module_parameters::uma_pool_id.value() >= 0 => {
                                    u64::from(scene.uma_aperture_pointer())
                                }
                                v => v,
                            }),
                            #[ver(V >= V15_5)]
                            unk_1a0: 0,
                            #[ver(V >= V15_5)]
                            unk_1a4: 0,
                            // 🔑 record+0x18. `fw:0x24fc0` *always* emits a
                            // second register after the pool base, from a
                            // second table `[0x14318, 0x14080, 0x14070]`:
                            //
                            //     ldr x11, [x20, #24]          ; record + 0x18
                            //     and x11, x11, #0x7ffffffffc0
                            //     orr x11, x11, #0x1
                            //
                            // Left null, register 0x14318 reads exactly `1` --
                            // which is what the read-back showed -- so the unit
                            // gets a base of zero. LOG.md 32.41.87.
                            #[ver(V >= V15_5)]
                            unk_1a8: U64(match *crate::module_parameters::uma_scratch_probe.value() {
                                // LOG.md 32.41.536: Apple's is a firmware-private VA
                                // (0xfffffc20004a0000); use a kernel/fw-visible buffer.
                                _ if *crate::module_parameters::uma_apple.value() != 0 => {
                                    u64::from(scene.uma_fwbuf_pointer())
                                }
                                0 => uma_scratch_base,
                                v => v,
                            }),
                            #[ver(V >= V15_5)]
                            unk_1b0: Default::default(),
                            #[ver(V >= V15_5)]
                            counter: U64(count_vtx),
                            #[ver(V >= V15_5)]
                            notifier_buf: inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf),
                            // `fw:0x161ec` reads this (op+0x1c8): 0 -> 0x1c8e0 =
                            // 0x1c8e8 = all-ones; non-zero -> round-robin cluster,
                            // 0x1c8e0 = 0xff << (cl*8), 0x1c8e8 = 0. Apple's is 1
                            // with clustering live. LOG.md 32.41.536.
                            #[ver(V >= V15_5)]
                            unk_178: match *crate::module_parameters::vtx_cluster_sel.value() {
                                0 => (!clustering) as u32,
                                v => v,
                            },
                        })?;

                        // 32.41.500: Apple emits BOTH Timestamp ops and the
                        // WaitForIdle between them unconditionally -- measured
                        // from macOS's own TA microsequence, not inferred. See
                        // `vtx_apple_useq` and the assertions in fw/microseq.rs.
                        let au = *crate::module_parameters::vtx_apple_useq.value();
                        // 1 = Apple's full shape. 2 = the two Timestamps but NO
                        // WaitForIdle -- the remaining cell of the 2x2 with .423
                        // (WFI alone, hard hang) and .502 (all three, soft stall).
                        // 3 = full shape with Apple's NULL user_ts_pointers.
                        let apple_useq = au != 0;
                        let apple_wfi = au != 0 && au != 2;
                        let apple_null_uts = au == 3;

                        if vtx_user_timestamps.any() || apple_useq {
                            builder.add(microseq::Timestamp::ver {
                                #[ver(V >= V15_5)]
                                header: microseq::op::Timestamp::new_v15_5(true),
                                #[ver(V < V15_5)]
                                header: microseq::op::Timestamp::new(true),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.start_addr),
                                work_queue: ev_vtx.info_ptr,
                                user_ts_pointers: if apple_null_uts {
                                    None
                                } else {
                                    Some(inner_weak_ptr!(ptr, user_timestamp_pointers))
                                },
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, unk_ts),
                                uuid: uuid_ta,
                                unk_30_padding: 0,
                            })?;
                        }

                        #[ver(G < G14X)]
                        builder.add(microseq::WaitForIdle {
                            header: microseq::op::WaitForIdle::new(microseq::Pipe::Vertex),
                        })?;
                        #[ver(G >= G14X && V < V15_5)]
                        builder.add(microseq::WaitForIdle2 {
                            header: microseq::op::WaitForIdle2::HEADER,
                        })?;

                        // 🔑 G15S/15.5 falls through BOTH gates above -- it is
                        // `G >= G14X` and `V >= V15_5` -- so this driver emits no
                        // TA wait at all before `FinalizeVertex`. Those two gates
                        // are ours, added while porting 15.5, and nothing ever
                        // validated the resulting gap.
                        //
                        // The kext has a dedicated method for exactly this op:
                        // `AGXTAChannelG15::encodeTAFinishWFI` emits a bare
                        // `0x00000001` (see `op::WaitForIdle::HEADER_V15_5`). The
                        // TA is the stage that never completes -- it parks at
                        // `0xc020 = 0x1b` even with an EMPTY control stream
                        // (32.41.422), which exonerated the stream and pointed at
                        // setup. A missing wait before the finalize is setup.
                        //
                        // Apple's call site is a virtual override from the shared
                        // AGFI layer and is not in this kext, so the *position* is
                        // inferred from our own `G < G14X` path, which puts the
                        // wait here. LOG.md 32.41.423.
                        #[ver(V >= V15_5)]
                        if *crate::module_parameters::vtx_pre_wfi.value() == 1 || apple_wfi {
                            builder.add(microseq::WaitForIdle {
                                header: microseq::op::WaitForIdle::HEADER_V15_5,
                            })?;
                        }

                        if vtx_user_timestamps.any() || apple_useq {
                            builder.add(microseq::Timestamp::ver {
                                #[ver(V >= V15_5)]
                                header: microseq::op::Timestamp::new_v15_5(false),
                                #[ver(V < V15_5)]
                                header: microseq::op::Timestamp::new(false),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.end_addr),
                                work_queue: ev_vtx.info_ptr,
                                user_ts_pointers: if apple_null_uts {
                                    None
                                } else {
                                    Some(inner_weak_ptr!(ptr, user_timestamp_pointers))
                                },
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, unk_ts),
                                uuid: uuid_ta,
                                unk_30_padding: 0,
                            })?;
                        }

                        let off = builder.offset_to(start_vtx);
                        builder.add(microseq::FinalizeVertex::ver {
                            #[ver(V >= V15_5)]
                            header: microseq::op::FinalizeVertex::HEADER_V15_5,
                            #[ver(V < V15_5)]
                            header: microseq::op::FinalizeVertex::HEADER,
                            scene: scene.weak_pointer(),
                            buffer: scene.weak_buffer_pointer(),
                            stats,
                            work_queue: ev_vtx.info_ptr,
                            #[ver(V < V15_5)]
                            vm_slot: vm_bind.slot(),
                            #[ver(V < V15_5)]
                            unk_28: 0x0, // fixed
                            #[ver(V >= V15_5)]
                            unk_24_ptr: inner_weak_ptr!(ptr, sync_grow),
                            #[ver(V < V15_5)]
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            #[ver(V < V15_5)]
                            unk_34: 0x0, // fixed
                            #[ver(V < V15_5)]
                            uuid: uuid_ta,
                            #[ver(V < V15_5)]
                            fw_stamp: ev_vtx.fw_stamp_pointer,
                            #[ver(V < V15_5)]
                            stamp_value: ev_vtx.value.next(),
                            #[ver(V < V15_5)]
                            unk_48: U64(0x0), // fixed
                            #[ver(V < V15_5)]
                            unk_50: 0x0, // fixed
                            #[ver(V < V15_5)]
                            unk_54: 0x0, // fixed
                            #[ver(V < V15_5)]
                            unk_58: U64(0x0), // fixed
                            #[ver(V < V15_5)]
                            unk_60: 0x0, // fixed
                            #[ver(V < V15_5)]
                            unk_64: 0x0, // fixed
                            #[ver(V < V15_5)]
                            unk_68: 0x0, // fixed
                            #[ver(G >= G14 && V < V13_0B4)]
                            unk_68_g14: U64(0),
                            #[ver(V < V15_5)]
                            restart_branch_offset: off,
                            #[ver(V < V15_5)]
                            has_attachments: (vertex_attachments.count > 0) as u32,
                            #[ver(V >= V13_0B4)]
                            #[ver(V < V15_5)]
                            unk_74: Default::default(),
                            #[ver(V >= V15_5)]
                            unk_2c: 0,
                            #[ver(V >= V15_5)]
                            unk_30: 0,
                            // Same pointer StartVertex carries in unk_job_buf.
                            #[ver(V >= V15_5)]
                            unk_34_ptr: inner_weak_ptr!(ptr, unk_pointee),
                            #[ver(V >= V15_5)]
                            unk_3c: 0,
                            #[ver(V >= V15_5)]
                            unk_40: U64(0),
                            #[ver(V >= V15_5)]
                            fw_stamp: if *crate::module_parameters::barrier_s2.value() == 3 {
                                barrier_fvst
                            } else {
                                ev_vtx.fw_stamp_pointer
                            },
                            #[ver(V >= V15_5)]
                            stamp_value: ev_vtx.value.next(),
                            #[ver(V >= V15_5)]
                            unk_54: Default::default(),
                            #[ver(V >= V15_5)]
                            unk_74: 0,
                            #[ver(V >= V15_5)]
                            unk_78_ptr: inner_weak_ptr!(ptr, unk_buf_8),
                            #[ver(V >= V15_5)]
                            restart_branch_offset: off,
                            #[ver(V >= V15_5)]
                            has_attachments: (vertex_attachments.count > 0) as u8,
                            #[ver(V >= V15_5)]
                            unk_85_ptr: inner_weak_ptr!(ptr, unk_buf_0),
                            #[ver(V >= V15_5)]
                            unk_8d: Default::default(),
                        })?;

                        // 15.5's sequence terminator: a bare 4-byte opcode 2
                        // with bit 30 set.  Every kext microsequence is sealed
                        // this way by 0x914525c, which is called immediately
                        // after the FinalizeVertex emitter returns.  LOG.md
                        // 32.41.25.
                        // 🔑 The terminal wait is where the VERTEX microsequence sits
                        // forever. 32.41.402/.403: the firmware's idle test is
                        // `(0xc020 & 0x1e) == 0`, our unit parks at `0x1b`
                        // (`& 0x1e == 0x1a`), and `0xc020` is read-only to the
                        // firmware (27 reads, 0 writes) so only the hardware can
                        // clear it. The END op therefore never returns, the stamp
                        // is never retired, `doneptr` stays 0 and userspace blocks
                        // in `glReadPixels` -- for a clear-only workload too, which
                        // is what proved this is not the draw path (32.41.400).
                        //
                        // `vtx_end_mode=1` omits it, mirroring `frag_end_mode` on
                        // the fragment side (32.41.202). Sealing the microsequence
                        // this way matches the kext (32.41.25), so omitting it is a
                        // DIAGNOSTIC to find out whether the retire happens without
                        // the wait -- not a fix. LOG.md 32.41.404.
                        // 🔑 `vtx_pre_wfi=2`: the same bare `0x00000001` but AFTER
                        // `FinalizeVertex`, immediately before the terminal seal.
                        //
                        // This is the better-motivated of the two placements and
                        // the one to prefer if only one is tried. The kext names
                        // the op `encodeTAFinishWFI` -- the WFI that *finishes*
                        // the TA -- and `AGXSKUEncoder::finishEncode` appends the
                        // `0x40000002` seal after the channel has finished
                        // encoding. So Apple's tail reads
                        //     ... FinalizeVertex, 0x00000001, 0x40000002
                        // `vtx_pre_wfi=1` put it before the finalize, inferred
                        // from our own `G < G14X` path, and hangs 5/5 with no
                        // diagnostics. LOG.md 32.41.423/.425.
                        #[ver(V >= V15_5)]
                        if *crate::module_parameters::vtx_pre_wfi.value() == 2 {
                            builder.add(microseq::WaitForIdle {
                                header: microseq::op::WaitForIdle::HEADER_V15_5,
                            })?;
                        }

                        #[ver(V >= V15_5)]
                        if *crate::module_parameters::vtx_end_mode.value() == 0 {
                            builder.add(microseq::WaitForIdle2 {
                                header: microseq::op::WaitForIdle2::HEADER_END_V15_5,
                            })?;
                        }

                        #[ver(V < V15_5)]
                        builder.add(microseq::RetireStamp::ver {
                            #[ver(V >= V15_5)]
                            header: microseq::op::RetireStamp::HEADER_V15_5,
                            #[ver(V < V15_5)]
                            header: microseq::op::RetireStamp::HEADER,
                            #[ver(V >= V15_5)]
                            unk_4: U64(0),
                            #[ver(V >= V15_5)]
                            ts_c: inner_weak_ptr!(ptr, unk_buf_0),
                            #[ver(V >= V15_5)]
                            unk_14: Default::default(),
                            #[ver(V >= V15_5)]
                            ts_24: inner_weak_ptr!(ptr, unk_buf_8),
                            #[ver(V >= V15_5)]
                            ts_2c: inner_weak_ptr!(ptr, unk_buf_10),
                            #[ver(V >= V15_5)]
                            ts_34: inner_weak_ptr!(ptr, command_time),
                        })?;
                        builder.build(private)?
                    },
                    notifier,
                    scene,
                    vm_bind,
                    timestamps,
                    user_timestamps: vtx_user_timestamps,
                })
            },
            |inner, _ptr| {
                let vm_slot = vm_bind.slot();
                #[ver(G < G14)]
                let core_masks = gpu.core_masks_packed();

                try_init!(fw::vertex::raw::RunVertex::ver {
                    tag: fw::workqueue::CommandType::RunVertex,
                    #[ver(V >= V13_0B4)]
                    counter: U64(count_vtx),
                    vm_slot,
                    unk_8: 0,
                    notifier: inner.notifier.gpu_pointer(),
                    buffer_slot: inner.scene.slot(),
                    unk_1c: 0,
                    buffer: inner.scene.buffer_pointer(),
                    scene: inner.scene.gpu_pointer(),
                    unk_buffer_buf: inner.scene.kernel_buffer_pointer(),
                    unk_34: 0,
                    #[ver(G < G14X)]
                    job_params1 <- try_init!(fw::vertex::raw::JobParameters1::ver {
                        unk_0: U64(if unk1 { 0 } else { 0x200 }), // sometimes 0
                        unk_8: f32!(1e-20),                       // fixed
                        unk_c: f32!(1e-20),                       // fixed
                        tvb_tilemap: inner.scene.tvb_tilemap_pointer(),
                        #[ver(G < G14)]
                        tvb_cluster_tilemaps: inner.scene.cluster_tilemaps_pointer(),
                        tpc: inner.scene.tpc_pointer(),
                        tvb_heapmeta: inner.scene.tvb_heapmeta_pointer().or(0x8000_0000_0000_0000),
                        iogpu_unk_54: U64(iogpu_unk54), // fixed
                        iogpu_unk_56: U64(iogpu_unk56), // fixed
                        #[ver(G < G14)]
                        tvb_cluster_meta1: inner
                            .scene
                            .meta_1_pointer()
                            .map(|x| x.or((tile_info.meta1_layer_stride as u64) << 50)),
                        utile_config,
                        unk_4c: 0,
                        ppp_multisamplectl: U64(cmdbuf.ppp_multisamplectl), // fixed
                        tvb_layermeta: inner.scene.tvb_layermeta_pointer(),
                        #[ver(G < G14)]
                        tvb_cluster_layermeta: inner.scene.tvb_cluster_layermeta_pointer(),
                        #[ver(G < G14)]
                        core_mask: Array::new([
                            *core_masks.first().unwrap_or(&0),
                            *core_masks.get(1).unwrap_or(&0),
                        ]),
                        preempt_buf1: inner.scene.preempt_buf_1_pointer(),
                        preempt_buf2: inner.scene.preempt_buf_2_pointer(),
                        unk_80: U64(0x1), // fixed
                        preempt_buf3: inner.scene.preempt_buf_3_pointer().or(0x4_0000_0000_0000), // check
                        vdm_ctrl_stream_base: U64(cmdbuf.vdm_ctrl_stream_base),
                        #[ver(G < G14)]
                        tvb_cluster_meta2: inner.scene.meta_2_pointer(),
                        #[ver(G < G14)]
                        tvb_cluster_meta3: inner.scene.meta_3_pointer(),
                        #[ver(G < G14)]
                        tiling_control,
                        #[ver(G < G14)]
                        unk_ac: tiling_control_2 as u32, // fixed
                        unk_b0: Default::default(), // fixed
                        // Reg 0x10061 is inert on G15S: programming it with 0
                        // left the fault at 0x22fff68000 unchanged, so the base
                        // the VDM SSD adds comes from this struct field or
                        // elsewhere. LOG.md 32.41.186
                        usc_exec_base_ta: U64({
                            // Log the value actually written. Without this an
                            // unchanged fault cannot be told apart from a param
                            // that never reached the struct. LOG.md 32.41.190
                            let v = match *crate::module_parameters::usc_base_ta_struct.value() {
                                u64::MAX => self.usc_exec_base,
                                v => v,
                            };
                            crate::sub_pr_info!("asahi/usc ta_struct: StartTA.usc_exec_base_ta {:#x}\n", v);
                            v
                        }),
                        #[ver(G < G14)]
                        tvb_cluster_meta4: inner
                            .scene
                            .meta_4_pointer()
                            .map(|x| x.or(0x3000_0000_0000_0000)),
                        #[ver(G < G14)]
                        unk_f0: U64(vtx_unk_f0),
                        unk_f8: U64(0x8c60),     // fixed
                        helper_program: cmdbuf.vertex_helper.binary,
                        unk_104: 0,
                        helper_arg: U64(cmdbuf.vertex_helper.data),
                        unk_110: Default::default(),      // fixed
                        unk_118: vtx_unk_118 as u32, // fixed
                        __pad: Default::default(),
                    }),
                    #[ver(G < G14X)]
                    tiling_params: tile_info.params,
                    #[ver(V >= V15_5)]
                    pad_regs_15_5: Default::default(),
                    #[ver(G >= G14X)]
                    registers: fw::job::raw::RegisterArray::new(
                        inner_weak_ptr!(_ptr, registers.registers),
                        |r| {
                            let g15_regs =
                                *crate::module_parameters::g15_vtx_regs.value() != 0;
                            r.add(0x10141, if unk1 { 0 } else { 0x200 }); // s2.unk_0
                            r.add(0x1c039, va(inner.scene.tvb_tilemap_pointer().into()));
                            r.add(0x1c9c8, va(inner.scene.tvb_tilemap_pointer().into()));

                            let cl_tilemaps_ptr = inner
                                .scene
                                .cluster_tilemaps_pointer()
                                .map_or(0, |a| va(a.into()));
                            r.add(0x1c041, cl_tilemaps_ptr);
                            r.add(0x1c9d0, cl_tilemaps_ptr);
                            r.add(0x1c0a1, va(inner.scene.tpc_pointer().into())); // TE_TPC_ADDR

                            let tvb_heapmeta_ptr = inner
                                .scene
                                .tvb_heapmeta_pointer()
                                .or(0x8000_0000_0000_0000)
                                .into();
                            let tvb_heapmeta_ptr = va(tvb_heapmeta_ptr);
                            r.add(0x1c031, tvb_heapmeta_ptr);
                            r.add(0x1c9c0, tvb_heapmeta_ptr);
                            r.add(0x1c051, iogpu_unk54); // iogpu_unk_54/55
                            r.add(0x1c061, iogpu_unk56); // iogpu_unk_56
                            r.add(0x10149, utile_config.into()); // s2.unk_48 utile_config
                            r.add(0x10139, cmdbuf.ppp_multisamplectl); // PPP_MULTISAMPLECTL
                            r.add(0x10111, va(inner.scene.preempt_buf_1_pointer().into()));
                            r.add(0x1c9b0, va(inner.scene.preempt_buf_1_pointer().into()));
                            r.add(0x10119, va(inner.scene.preempt_buf_2_pointer().into()));
                            r.add(0x1c9b8, va(inner.scene.preempt_buf_2_pointer().into()));
                            r.add(0x1c958, 1); // s2.unk_80
                            r.add(
                                0x1c950,
                                va(inner
                                    .scene
                                    .preempt_buf_3_pointer()
                                    .or(0x4_0000_0000_0000)
                                    .into()),
                            );
                            r.add(0x1c930, 0); // VCE related addr, lsb to enable
                            // The firmware masks the address with ~3 before
                            // handing it to the hardware; match it.
                            // 🔬 The CSD demonstrably reads from `cs + 0x80000`
                            // and walks forward (32.41.250).  If the start is
                            // simply biased, pre-subtracting the bias puts it
                            // on Mesa's first record.  LOG.md 32.41.251.
                            let vdm_ctrl_stream_base = vdm_ctrl_stream_base
                                .wrapping_sub(*crate::module_parameters::vdm_base_sub.value());
                            crate::sub_pr_info!(
                                "asahi/vdm cs base: {:#x}\n",
                                vdm_ctrl_stream_base
                            );
                            // 32.41.531: apple_replay points the VDM at Apple's own stream.
                            let vdm_ctrl_stream_base =
                                if *crate::module_parameters::apple_replay.value() != 0 {
                                    pr_info!(
                                        "asahi/replay: 0x1c880 {:#x} -> 0x18000 (Apple's stream)\n",
                                        vdm_ctrl_stream_base
                                    );
                                    0x18000
                                } else {
                                    vdm_ctrl_stream_base
                                };
                            r.add(0x1c880, vdm_ctrl_stream_base); // VDM_CTRL_STREAM_BASE
                            // 🔬 Overridable. This register is the companion of
                            // VDM_CTRL_STREAM_BASE and its value was *derived
                            // from firmware disassembly*, not measured: the
                            // helper at fw:0x18a4c emits (0x1c880, addr & ~3)
                            // then (0x1c898, 1). If it selects the control
                            // stream's format rather than merely enabling it,
                            // a wrong value makes the CSD parse records at the
                            // wrong lengths -- which is exactly the Phase 3
                            // symptom. Sweepable in one run. LOG.md 32.41.168.
                            let cs_unk = *crate::module_parameters::vdm_cs_unk.value();
                            r.add(
                                0x1c898,
                                if cs_unk == 0xffffffff {
                                    vdm_ctrl_stream_unk
                                } else {
                                    cs_unk.into()
                                },
                            );
                            r.add(
                                0x1c948,
                                inner.scene.meta_2_pointer().map_or(0, |a| va(a.into())),
                            ); // tvb_cluster_meta2
                            r.add(
                                0x1c888,
                                inner.scene.meta_3_pointer().map_or(0, |a| va(a.into())),
                            ); // tvb_cluster_meta3
                            // 🔬 `tiling_control` for t6030 is t602x's
                            // `0x180340` copied over as a guess (hw/t6030.rs).
                            // With 1024 TVB pages available and gPM still
                            // taking none (32.41.264), the question is whether
                            // the tiler is *enabled* at all, and this is the
                            // register that would do it.  LOG.md 32.41.265.
                            let tc = match *crate::module_parameters::tiling_control_ovr.value() {
                                u64::MAX => tiling_control.into(),
                                v => v,
                            };
                            let tc2 = match *crate::module_parameters::tiling_control2_ovr.value() {
                                u64::MAX => tiling_control_2,
                                v => v,
                            };
                            crate::sub_pr_info!("asahi/tilectl: 0x1c890 {:#x} 0x1c918 {:#x}\n", tc, tc2);
                            r.add(0x1c890, tc); // tvb_tiling_control
                            r.add(0x1c918, tc2);
                            r.add(0x1c079, va(inner.scene.tvb_layermeta_pointer().into()));
                            r.add(0x1c9d8, va(inner.scene.tvb_layermeta_pointer().into()));
                            let cl_layermeta_pointer =
                                inner.scene.tvb_cluster_layermeta_pointer().map_or(0, |a| va(a.into()));
                            r.add(0x1c089, cl_layermeta_pointer);
                            r.add(0x1c9e0, cl_layermeta_pointer);
                            let cl_meta_4_pointer =
                                inner.scene.meta_4_pointer().map_or(0, |a| va(a.into()));
                            r.add(0x16c41, cl_meta_4_pointer); // tvb_cluster_meta4
                            r.add(0x1ca40, cl_meta_4_pointer); // tvb_cluster_meta4
                            r.add(0x1c9a8, vtx_unk_f0); // + meta1_blocks? min_free_tvb_pages?
                            r.add(
                                0x1c920,
                                inner.scene.meta_1_pointer().map_or(0, |a| va(a.into())),
                            ); // ??? | meta1_blocks?
                            // 🔬 macOS fills all five of these from consecutive
                            // descriptor fields `desc[0xe50..0xe70]`, each
                            // masked to `& ~0xff` -- five page-aligned
                            // addresses in a row, which is the shape of a base
                            // table.  This driver writes them as zero.
                            //
                            // That matters because the VDM's USC base is the
                            // one number nothing on G14X+ can program (32.41.191):
                            // reg 0x10061 is inert, and both `usc_exec_base_*`
                            // struct fields are `#[ver(G < G14X)]`.  The base
                            // the hardware actually uses is `usc_exec_base +
                            // 2^32` and the next blocker after it resolves is
                            // `usc_exec_base + 4*2^32` -- both of the form
                            // `usc_exec_base + k*2^32`, i.e. an *index* into
                            // something.  agxdecode calls Mesa's pipeline
                            // pointer "handle 11", so the stream carries a
                            // segment index and the hardware is meant to look
                            // its base up.  These five registers are the only
                            // table-shaped thing in macOS's tiler list.
                            //
                            // Poison them with distinct addresses: whichever
                            // one the VDM reads, the fault names it.
                            // LOG.md 32.41.249.
                            let seg = *crate::module_parameters::usc_seg_probe.value();
                            r.add(0x10151, *crate::module_parameters::reg_10151.value());
                            for (i, reg) in [0x1c199u32, 0x1c1a1, 0x1c1a9, 0x1c1b1, 0x1c1b9]
                                .iter()
                                .enumerate()
                            {
                                let v = if seg != 0 {
                                    seg + (i as u64) * 0x100_0000
                                } else {
                                    match *crate::module_parameters::usc_seg_base.value() {
                                        0 => 0,
                                        u64::MAX => va(self.usc_exec_base),
                                        v => v,
                                    }
                                };
                                r.add(*reg, v);
                            }
                            if seg != 0 {
                                pr_info!(
                                    "asahi/usc seg probe: 0x1c199..0x1c1b9 <- {:#x}+k*0x1000000, 0x10151 {:#x}\n",
                                    seg,
                                    *crate::module_parameters::reg_10151.value()
                                );
                            }
                            // 🔑 macOS position 39.  fffffe000911b7f0 builds it
                            // exactly like the fragment's 0x100b8:
                            //     x4 = desc[0xa8] ? 0x8ae0 : 0x8860
                            //     if cfg[0x4d4] > 4 || cfg[0x691] & 1 { x4 |= 0x1c }
                            // The fragment half of that identical code measured
                            // 0x8860 live on this part, so both branches are
                            // pinned. capture2, LOG.md 32.41.216.
                            if g15_regs {
                                r.add(0x1c8f8, 0x8860);
                            }
                            // The VDM SSD forms the pipeline address as
                            // (this register) + the stream's USC-relative
                            // pointer, so the fault address is a direct
                            // readout of what the hardware used. LOG.md 32.41.185
                            let usc_base_ta = match *crate::module_parameters::usc_base_ta.value() {
                                u64::MAX => va(self.usc_exec_base),
                                v => v,
                            };
                            crate::sub_pr_info!(
                                "asahi/usc ta: exec_base {:#x} -> reg 0x10061 {:#x}\n",
                                self.usc_exec_base,
                                usc_base_ta
                            );
                            // Neither 0x10061 nor the vertex helper program
                            // appears anywhere in macOS's G15S tiler list, at
                            // any point in the sweep -- the same absence the
                            // fragment side shows for 0x10069 and 0x1182x.
                            if !g15_regs {
                                r.add(0x10061, usc_base_ta); // USC_EXEC_BASE_TA
                                r.add(0x11801, va(cmdbuf.vertex_helper.binary.into()));
                                r.add(0x11809, va(cmdbuf.vertex_helper.data));
                                r.add(0x11f71, cmdbuf.vertex_helper.cfg.into());
                            }
                            r.add(0x1c0b1, tile_info.params.rgn_size.into()); // TE_PSG
                            r.add(0x1c850, tile_info.params.rgn_size.into());
                            r.add(0x10131, tile_info.params.unk_4.into());
                            r.add(0x10121, tile_info.params.ppp_ctrl.into()); // PPP_CTRL
                            r.add(
                                0x10129,
                                tile_info.params.x_max as u64
                                    | ((tile_info.params.y_max as u64) << 16),
                            ); // PPP_SCREEN
                            r.add(0x101b9, tile_info.params.te_screen.into()); // TE_SCREEN
                            r.add(0x1c069, tile_info.params.te_mtile1.into()); // TE_MTILE1
                            r.add(0x1c071, tile_info.params.te_mtile2.into()); // TE_MTILE2
                            r.add(0x1c081, tile_info.params.tiles_per_mtile.into()); // TE_MTILE
                            r.add(0x1c0a9, tile_info.params.tpc_stride.into()); // TE_TPC
                            r.add(0x10171, tile_info.params.unk_24.into());
                            r.add(0x10169, tile_info.params.unk_28.into()); // TA_RENDER_TARGET_MAX
                            r.add(0x12099, vtx_unk_118);
                            // 0x101e1 is `getAgxCrAnisoConfigVdmOffset()` --
                            // the tiler twin of the fragment's 0x101e9, which
                            // the live capture shows carrying the same value as
                            // its 0x120a1.  Both offsets come from virtual
                            // getters on AGXAcceleratorG15X (vtable +0x1138 /
                            // +0x1140, resolved through __ZTV18AGXAcceleratorG15X
                            // and cross-checked against the call discriminators).
                            if g15_regs {
                                // ⚠️ Not in macOS's tiler list: the positional
                                // diff (32.41.266) shows this driver emitting
                                // `0x101e1` at [53] where macOS repeats
                                // `0x12099`.  A register macOS never writes in
                                // this path is a candidate for gating the TA
                                // stage off.  LOG.md 32.41.269.
                                if *crate::module_parameters::drop_extra_vtx_regs.value() == 0 {
                                    r.add(0x101e1, vtx_unk_118);
                                }
                            }
                            r.add(0x1c9e8, (tile_info.params.unk_28 & 0x4fff).into());
                            // 🔑 The tiler tail macOS programs after 0x1c9e8.
                            // Values read out of `generateRegisterList`; the
                            // 0x1a0xx block and 0x101c9/0xd471 are descriptor
                            // fields that are zero for a plain draw, and
                            // 0xa5a1/0xd419 take the same two code paths whose
                            // fragment twins (0xa5a9/0xd429) were measured live
                            // on this part.
                            //
                            // 0x10799 is `getAgxCrUscTpGtpResourceCfg0Offset()`,
                            // the twin of the fragment's 0x10791: with
                            // desc[0x12f8] zero it is the bare 0xff0000, the
                            // fragment's 0x00ff0200 minus a low field the tiler
                            // path does not have.
                            //
                            // Two entries are deliberately left out because
                            // their values are not derivable: 0x1ca48 (guarded
                            // by cfg[0x338] == 1, built from two descriptor
                            // fields) and 0x1c910 (an address run through the
                            // bit-42/0x7000000000 segment fold at
                            // fffffe000911c8fc).  LOG.md 32.41.216.
                            if g15_regs {
                                r.add(0x1a099, 0);
                                r.add(0x1a0a1, 0);
                                r.add(0x1a069, 0);
                                r.add(0x1a071, 0);
                                r.add(0x1a0c9, 0);
                                r.add(0x1a0d1, 0);
                                r.add(0x101c9, 0);
                                r.add(0x0d471, 0);
                                r.add(0x1a0f1, 8);
                                r.add(0x10799, 0x00ff_0000);
                                r.add(0x0a5a1, 0x0000_0060_0040_0020);
                                r.add(0x0d419, 0x0000_0360_0000_0120);
                                r.add(0x1ca10, 0);
                                r.add(0x014a1, 0);
                                r.add(0x0a349, 0);
                            }
                            // 🔑 The five TA registers macOS programs and this
                            // driver never has. Found by diffing the live macOS
                            // TA list captured 2026-09-11 (74 registers, via
                            // DTrace on `AGXTAChannelG15::generateRegisterList`)
                            // against this driver's list: exactly five ids are
                            // present there and absent here, a contiguous
                            // stride-8 block, all zero, classified CONSTANT
                            // across all 26 captured configurations.
                            //
                            // Writing zero is NOT equivalent to not writing:
                            // these are hardware registers that retain whatever
                            // a previous job left in them. macOS emits them on
                            // every TA submission, `clear` and `draw` alike.
                            // LOG.md 32.41.370 (entry LOST -- see 32.41.431).
                            // 🔑🔑 The GEOMETRY block: nine TA registers macOS
                            // leaves at zero (one at 1) for jobs **below 63
                            // primitives**, and only switches on above it.
                            // Measured live on macOS 2026-09-11 across 26
                            // configurations: the transition is exactly 62->63
                            // primitives, reproducible 3/3 each side, and it
                            // gates on primitive count -- not vertex count, not
                            // tile coverage. The values are address-shaped
                            // (tiler parameter-buffer / partial-render
                            // pointers) and macOS programs them only when a job
                            // is large enough to need them.
                            //
                            // This driver programs all nine unconditionally. On
                            // a one-triangle job that hands the TA pointers into
                            // partial-render machinery the job never set up,
                            // which matches the symptom exactly: the unit starts,
                            // goes busy, parks in state 3 and never reaches the
                            // state 5 FinalizeVertex polls for.
                            //
                            // `set()` rather than `add()`: the list length is
                            // load-bearing (it becomes the doorbell's count).
                            // LOG.md 32.41.371 (entry LOST -- see 32.41.431).
                            if *crate::module_parameters::vtx_small_job.value() != 0 {
                                for (reg, val) in [
                                    (0x1c041u32, 0u64),
                                    (0x1c9d0, 0),
                                    (0x1c948, 0),
                                    (0x1c888, 0),
                                    (0x1c920, 0),
                                    (0x16c41, 0),
                                    (0x1ca40, 0),
                                    (0x1c918, 0),
                                    (0x1c890, 1),
                                ] {
                                    if !r.set(reg, val) {
                                        pr_info!("asahi/smalljob: {:#x} absent\n", reg);
                                    }
                                }
                            }
                            // 🔑 The complete set of registers macOS holds at
                            // zero for a sub-63-primitive job, from the
                            // 2026-09-11 G15S capture (40 configurations).
                            // `vtx_small_job` covered nine of eleven: it took
                            // the nine CLASSIFICATION.md marks GEOMETRY and
                            // missed `0x1c089`/`0x1c9e0`, which are marked
                            // CONSTANT because macOS leaves them at zero in
                            // *every* configuration -- 1 to 20000 primitives,
                            // 64x64 to 1024x1024, depth, 4x MSAA. Those two are
                            // `tvb_cluster_layermeta`, and this driver programs
                            // them with a real pointer whenever clustering is
                            // on (t6030 reports 2 clusters, so it always is).
                            // So the cluster block has never actually been
                            // zeroed as a unit. LOG.md 32.41.375.
                            if *crate::module_parameters::vtx_macos_zeros.value() != 0 {
                                let mut nz = 0u32;
                                for (reg, val) in [
                                    (0x1c041u32, 0u64),
                                    (0x1c9d0, 0),
                                    (0x1c948, 0),
                                    (0x1c888, 0),
                                    (0x1c920, 0),
                                    (0x16c41, 0),
                                    (0x1ca40, 0),
                                    (0x1c918, 0),
                                    (0x1c890, 1),
                                    // the two vtx_small_job missed
                                    (0x1c089, 0),
                                    (0x1c9e0, 0),
                                ] {
                                    if !r.set(reg, val) {
                                        crate::sub_pr_info!("asahi/mzeros: {:#x} absent\n", reg);
                                    } else {
                                        nz += 1;
                                    }
                                }
                                // Positive gate (.528): "no 'absent' line" is also what a
                                // block that never ran looks like.  A gate that a disabled
                                // feature can pass is not a gate.
                                crate::sub_pr_info!("asahi/mzeros: applied {} of 11\n", nz);
                            }
                            if *crate::module_parameters::vtx_zero_regs.value() != 0 {
                                r.add(0x1c199, 0);
                                r.add(0x1c1a1, 0);
                                r.add(0x1c1a9, 0);
                                r.add(0x1c1b1, 0);
                                r.add(0x1c1b9, 0);
                            }
                            // 🔑 m1n1's macOS capture programs `buf_desc.unkptr_18`
                            // -- a 0x80-byte buffer that is exactly this driver's
                            // `Scene.user_buffer` (same size, same "list with 4x2
                            // bytes of header and 8 bytes per entry" shape) -- into
                            // 0x1ca30 and 0x16c39 for the vertex pipe and 0x1ca28
                            // for the fragment one. Upstream commented all three
                            // out under "these are actually appended by the
                            // firmware?", with the captured macOS addresses as
                            // their values. `regemit.py` says that guess is only
                            // half right: StartFragment does append 0x1ca28, but
                            // StartVertex appends neither 0x1ca30 nor 0x16c39, so
                            // on the vertex side they were never programmed at all
                            // and the hardware read whatever they reset to.
                            // LOG.md 32.41.48.
                            let user_buffer = u64::from(inner.scene.user_buffer_pointer());
                            // 32.41.666: 0x1c830 = desc[0x1040] = the PB slot (TA twin of 0x1c838).
                            r.add(
                                0x1c830,
                                if *crate::module_parameters::pb_slot_regs.value() != 0 {
                                    inner.scene.slot() as u64
                                } else {
                                    1
                                },
                            );
                            r.add(0x1ca30, va(user_buffer));
                            r.add(0x16c39, va(user_buffer));
                            // 🔬 The last register in macOS's tiler list, and
                            // the only one whose *value* could not be derived.
                            // `generateRegisterList` @ fffffe000911c8fc reads a
                            // GPU address out of the parameter-buffer block
                            // (`desc[0x1058]`, filled by
                            // AGXTACommandDescriptor::loadParameterManagementData
                            // from AGXParameterManagement) and folds it:
                            //
                            //   x11 = (a & 1<<42) ? 0 : 0x7000000000
                            //   reg = (((a >> 3) & 0x8000000000)
                            //          | ((x11 + a) & 0x7ffffffffe)) | 1
                            //
                            // which for any address at or above the code GART
                            // base is exactly `a - 0x1000000000`, with bit 0
                            // set.  Since `user_va_shift` maps every user page
                            // at both `VA` and `VA + 2^36`, the plain VA works
                            // as well as the folded one; the open question is
                            // only *which* buffer, so this selects among the
                            // scene pointers rather than guessing one.  The
                            // sibling field of the same block (+0x28) is what
                            // 0x1ca30/0x16c39 carry, i.e. `user_buffer`.
                            // LOG.md 32.41.221.
                            let src_1c910 = *crate::module_parameters::vtx_1c910_src.value();
                            if src_1c910 != 0 {
                                let a = match src_1c910 {
                                    1 => u64::from(inner.scene.tvb_tilemap_pointer()),
                                    2 => u64::from(inner.scene.tvb_heapmeta_pointer()),
                                    3 => u64::from(inner.scene.tpc_pointer()),
                                    4 => user_buffer,
                                    5 => inner.scene.meta_1_pointer().map_or(0, u64::from),
                                    6 => u64::from(inner.scene.buffer_pointer()),
                                    7 => u64::from(inner.scene.preempt_buf_1_pointer()),
                                    9 => inner.scene.tvb_page_list_pointer(),
                                    10 => inner.scene.tvb_block_list_pointer(),
                                    11 => inner.scene.tvb_block_ctl_pointer(),
                                    12 => inner.scene.control_word_pointer(),
                                    13 => inner.scene.tvb_block0_pointer(),
                                    // Poison: a never-mapped address, so one
                                    // run says whether the register is
                                    // dereferenced at all before any effort
                                    // goes into finding the right buffer.
                                    _ => 0x3bad_00_0000,
                                };
                                // ⚠️ The value is **folded**, not the plain VA. Apple
                                // does this in two independent places -- the
                                // register site (fffffe000911c8fc) and
                                // `AGXAcceleratorG15::halLoadPBDesc`
                                // (fffffe00091139cc), which reads a parameter
                                // buffer address out of
                                // `_AGFIParameterBufferState + 0x24` and folds it
                                // identically. That second site also identifies
                                // what the value *is*: a parameter-buffer GPU
                                // address.
                                //
                                //   bias = (a & 1<<42) ? 0 : 0x7000000000
                                //   seg  = (a >> 3) & 0x8000000000
                                //   reg  = (((bias + a) & 0x7ffffffffe) | seg) | 1
                                //
                                // For an address at or above the code GART base
                                // this is `a - 0x1000000000`, and it lifts bit 42
                                // of the address into bit 39 of the register.
                                // My first pass wrote `va(a) | 1` -- the right
                                // candidate in the wrong encoding, which is why
                                // 32.41.239 failed. LOG.md 32.41.241.
                                let fold = |x: u64| -> u64 {
                                    let bias = if x & (1u64 << 42) == 0 {
                                        0x70_0000_0000u64
                                    } else {
                                        0
                                    };
                                    let seg = (x >> 3) & 0x80_0000_0000u64;
                                    ((bias.wrapping_add(x)) & 0x7f_ffff_fffeu64) | seg
                                };
                                let folded = fold(va(a)) | 1;
                                crate::sub_pr_info!(
                                    "asahi/1c910: src {} addr {:#x} folded {:#x}\n",
                                    src_1c910,
                                    a,
                                    folded
                                );
                                if a != 0 {
                                    r.add(0x1c910, folded);
                                }
                            }
                            // The register-id diff against the 2026-09-11 G15S
                            // DTrace capture shows our list is macOS's list
                            // exactly, ids 0..72 in the same order, and stops
                            // one short: macOS appends `0x1c910` at index 73
                            // and we emit nothing.  The capture also settles
                            // what the value is -- `0xa000160b45` in all 40
                            // sweep configurations, from 1x1 to 1024x1024 and
                            // 1 to 20000 primitives, across separate
                            // processes.  A folded scene pointer would have
                            // moved at 1024x1024 like every other ALLOC-class
                            // register did, so `vtx_1c910_src`'s derivation
                            // (32.41.221) does not describe G15S.  Transcribe
                            // the constant.  LOG.md 32.41.373.
                            let raw_1c910 = *crate::module_parameters::vtx_1c910_raw.value();
                            if raw_1c910 != 0 && src_1c910 == 0 {
                                crate::sub_pr_info!("asahi/1c910: raw {:#x}\n", raw_1c910);
                                r.add(0x1c910, raw_1c910);
                            }
                            // The two non-address CONSTANT mismatches the G15S
                            // capture exposes: macOS uses 0x3c where we compute
                            // 0x1c (we source `0x1c9a8` from whatever feeds
                            // 0x12099/0x101e1, which really are 0x1c -- the
                            // right value in the wrong register), and 9 where we
                            // hardcode 1. Both CONSTANT across all 40 configs.
                            //
                            // ⚠️ Must sit HERE, after `r.add(0x1c830, 1)`:
                            // `r.set` only overwrites an entry that is already
                            // in the list, and an earlier placement logged
                            // `asahi/mconst: 0x1c830 absent` and silently left
                            // it at 1. LOG.md 32.41.376.
                            //
                            // 🔑 32.41.513 CORRECTION: `0x1c830` is NOT a
                            // constant.  Decoding Apple's RegisterArray with
                            // the firmware's OWN layout (12-byte entries, u16
                            // count at array+0x708 -- fw:0x14f8c) gives entry
                            // [70] `0x1c830 = 0xd`, and that job's
                            // `RunVertex.buffer_slot` is also `0xd`.  It is the
                            // BUFFER SLOT.  The "9" below generalised one
                            // capture's slot number into a constant; neither 9
                            // nor the hardcoded 1 is right for us.  Mode 2
                            // sources it from `scene.slot()`, which is what the
                            // field actually is.
                            let mconst = *crate::module_parameters::vtx_macos_consts.value();
                            if mconst != 0 {
                                let slot_1c830: u64 =
                                    if mconst >= 2 { inner.scene.slot() as u64 } else { 9 };
                                crate::sub_pr_info!("asahi/mconst: mode {} 0x1c830 {:#x}\n", mconst, slot_1c830);
                                for (reg, val) in [(0x1c9a8u32, 0x3cu64), (0x1c830, slot_1c830)] {
                                    if !r.set(reg, val) {
                                        crate::sub_pr_info!("asahi/mconst: {:#x} absent\n", reg);
                                    }
                                }
                            }
                            // 🔑 The sixteen registers macOS's own
                            // AGXTAChannelG15::generateRegisterList programs and
                            // this driver never has.  Extracted from the 15.5
                            // kernelcache, which is the right reference for
                            // G15 -- the m1n1 capture everything was diffed
                            // against until now is a 13.x trace on G13/G14.
                            // `missing_regs_probe` writes each one a distinct
                            // never-mapped address so a fault names the register
                            // that wanted it.  LOG.md 32.41.56.
                            let probe = *crate::module_parameters::missing_regs_probe.value();
                            if probe != 0 {
                                for (i, reg) in [
                                    0x14a0, 0xa348, 0xa5a0, 0xd418, 0xd470, 0x101c8,
                                    0x1a068, 0x1a070, 0x1a098, 0x1a0a0, 0x1a0c8, 0x1a0d0,
                                    0x1a0f0, 0x1c8f8, 0x1ca10, 0x1ca48,
                                    // The six UMA pool base registers the
                                    // firmware picks from its table at
                                    // fw:0x50bdc.  Never probed before -- they
                                    // are not in macOS's per-job lists because
                                    // the firmware appends them itself, but it
                                    // only appends *one*, so poisoning all six
                                    // tests the other five. LOG.md 32.41.67.
                                    0x14308, 0x14048, 0x14028, 0x14318, 0x14080, 0x14070,
                                ]
                                .iter()
                                .enumerate()
                                {
                                    // Low bit of the probe value doubles as
                                    // "also set number bit 0", the flag the
                                    // driver carries on many of its own entries
                                    // and macOS's bare numbers do not.
                                    r.add(*reg | (probe as u32 & 1), (probe & !0xfff) | ((i as u64) << 12));
                                }
                            }
                            if debug_enabled(DebugFlags::Render) {
                                r.dump("vtx");
                            }
                            /*
                            r.add(0x10209, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x1c9f0, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x1c830, 1); // ?
                            r.add(0x1ca30, 0x1502960e60); // ?
                            r.add(0x16c39, 0x1502960e60); // ?
                            r.add(0x1c910, 0xa0000b011d); // ?
                            r.add(0x1c8e0, 0xff); // cluster mask
                            r.add(0x1c8e8, 0); // ?
                            */
                        }
                    ),
                    tpc: inner.scene.tpc_pointer(),
                    tpc_size: U64(tile_info.tpc_size as u64),
                    microsequence: inner.micro_seq.gpu_pointer(),
                    microsequence_size: inner.micro_seq.len() as u32,
                    fragment_stamp_slot: ev_frag.slot,
                    fragment_stamp_value: ev_frag.value.next(),
                    unk_pointee: 0,
                    unk_pad: 0,
                    job_params2 <- try_init!(fw::vertex::raw::JobParameters2::ver {
                        unk_480: Default::default(), // fixed
                        unk_498: U64(0x0),           // fixed
                        unk_4a0: 0x0,                // fixed
                        preempt_buf1: inner.scene.preempt_buf_1_pointer(),
                        unk_4ac: 0x0,      // fixed
                        unk_4b0: U64(0x0), // fixed
                        unk_4b8: 0x0,      // fixed
                        unk_4bc: U64(0x0), // fixed
                        unk_4c4_padding: Default::default(),
                        unk_50c: 0x0,      // fixed
                        unk_510: U64(0x0), // fixed
                        unk_518: U64(0x0), // fixed
                        unk_520: U64(0x0), // fixed
                    }),
                    encoder_params <- try_init!(fw::job::raw::EncoderParams {
                        unk_8: 0x0,     // fixed
                        sync_grow: 0x0, // fixed
                        unk_10: 0x0,    // fixed
                        encoder_id: 0,
                        unk_18: 0x0, // fixed
                        unk_mask: 0xffffffffu32,
                        sampler_array: U64(sampler_heap),
                        sampler_count: cmdbuf.sampler_count as u32,
                        sampler_max: (cmdbuf.sampler_count as u32) + 1,
                    }),
                    unk_55c: 0,
                    unk_560: 0,
                    sync_grow: 0,
                    unk_568: 0,
                    uses_scratch: (cmdbuf.flags
                        & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_VERTEX_SCRATCH as u32
                        != 0) as u32,
                    meta <- try_init!(fw::job::raw::JobMeta::ver {
                        unk_0: 0,
                        unk_2: 0,
                        no_preemption: no_preemption as u8,
                        #[ver(V >= V15_5)]
                        unk_4: Default::default(),
                        stamp: ev_vtx.stamp_pointer,
                        fw_stamp: ev_vtx.fw_stamp_pointer,
                        stamp_value: ev_vtx.value.next(),
                        stamp_slot: ev_vtx.slot,
                        #[ver(V < V15_5)]
                        evctl_index: 0, // fixed
                        #[ver(V < V15_5)]
                        flush_stamps: flush_stamps as u32,
                        #[ver(V < V15_5)]
                        uuid: uuid_ta,
                        // 15.5: the firmware uses this word as the workqueue's slot in
                        // the notifier's 4-entry waiter table (pstamp_*[idx], bit idx of
                        // has_*; fw:0x1f45c). Apple = 1 << AGXCLWorkQueue+0xe0, a small
                        // per-queue index; we have one workqueue per pipe per notifier, so
                        // 0. The running event_seq overflowed slot 3 on the 5th job and
                        // wrote into in_list/list_head (fw crash 0x1f900). LOG.md 32.41.664
                        #[ver(V >= V15_5)]
                        event_seq: 0,
                        #[ver(V < V15_5)]
                        event_seq: ev_vtx.event_seq as u32,
                    }),
                    unk_after_meta: unk1.into(),
                    unk_buf_0: U64(0),
                    // FinalizeVertex's 0x78 is a pointer *to a pointer*: the
                    // handler does `ldur x20, [op+0x78]` and then 0x2518c does
                    // `ldr x21, [x20]; ldr w8, [x21, #12]`, so what it points
                    // at has to hold the pool address, not zero.  macOS aims
                    // it at StartVertex+0x190, whose first word is the same
                    // pool pointer; the builder cannot form an address inside
                    // the sequence it is still assembling, so carry the same
                    // value here instead.  LOG.md 32.41.25.
                    #[ver(V < V15_5)]
                    unk_buf_8: U64(0),
                    #[ver(V >= V15_5)]
                    unk_buf_8: U64(inner.scene.uma_pool_pointer().into()),
                    unk_buf_10: U64(0),
                    // 32.41.511: 0x10 the firmware never reads; command_time is
                    // at 0x8b0, not 0x8a0.
                    #[ver(V >= V15_5)]
                    unk_8a0: Default::default(),
                    command_time: U64(0),
                    timestamp_pointers <- try_init!(fw::job::raw::TimestampPointers {
                        start_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), vtx.start)),
                        end_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), vtx.end)),
                    }),
                    user_timestamp_pointers: inner.user_timestamps.pointers()?,
                    client_sequence: slot_client_seq,
                    pad_5d5: Default::default(),
                    unk_5d8: 0,
                    unk_5dc: 0,
                    #[ver(V >= V13_0B4)]
                    unk_ts: U64(0),
                    #[ver(V >= V13_0B4)]
                    unk_5dd_8: Default::default(),
                    // Its own zeroed object: aliasing this onto the buffer info
                    // fed the aperture descriptor at 0x24cf0 a garbage base and
                    // hung the machine. LOG.md 32.41.16.
                    #[ver(V >= V15_5)]
                    uma_pool: U64(inner.scene.uma_pool_pointer().into()),
                    #[ver(V >= V15_5)]
                    uma_pool_flush: *crate::module_parameters::uma_pool_flush.value() as u8,
                    #[ver(V >= V15_5)]
                    unk_5e7: Default::default(),
                    #[ver(V >= V15_5)]
                    unk_900: Default::default(),
                })
            },
        )?;

        core::mem::drop(alloc);

        if *crate::module_parameters::dump_vtx_cmd.value() != 0 {
            crate::gpu::dump_tvb("PRE");
        }
        {
            let m = *crate::module_parameters::prefill_scene.value();
            if m != 0 {
                crate::gpu::prefill_scene(m);
            }
        }
        crate::sub_dev_err!(
            self.dev.as_ref(),
            "asahi/cmd: vtx {:#x} meta+{:#x} fw_stamp {:#x} stamp_value {:#x} scene {:#x}\n",
            vtx.gpu_va().get(),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, meta),
            u64::from(ev_vtx.fw_stamp_pointer),
            ev_vtx.value.next().counter(),
            u64::from(scene.weak_pointer()),
        );

        // 🔑 Dump the TA command bytes so they can be diffed against Apple's.
        // `2026-09-11-ta-capture/out/raw-draw-triangle-64x64-full.txt` holds
        // 4096 bytes of Apple's `AGFIChannelCommandTA` -- the same struct --
        // captured live but never compared. Decoding Apple's bytes with THESE
        // field offsets lands every field on a sane value (32.41.428), so the
        // layout is confirmed; what is not yet checked is our VALUES, field by
        // field, the way the register lists were. LOG.md 32.41.429.
        // 32.41.444: publish the command's CPU address so the post-kick probe can
        // dump it AGAIN and diff. Fields the firmware writes back -- `unk_buf_0`
        // (ts_c), `unk_buf_8` (ts_24), `command_time`, `meta.stamp`/`fw_stamp` --
        // record execution progress, and need no optional microsequence ops, unlike
        // the Timestamp ops that made 32.41.443 void.
        if *crate::module_parameters::dump_vtx_cmd.value() != 0 {
            vtx.with(|raw, _| {
                crate::gpu::DBG_VTXCMD.store(
                    raw as *const _ as u64,
                    core::sync::atomic::Ordering::Relaxed,
                );
            });
        }
        if *crate::module_parameters::dump_vtx_cmd.value() != 0 {
            vtx.with(|raw, _| {
                let n = core::mem::size_of_val(raw) / 8;
                // SAFETY: `raw` is a valid reference to a live, 8-aligned
                // GpuStruct; we read `n` u64s entirely inside it.
                let q = unsafe {
                    core::slice::from_raw_parts(raw as *const _ as *const u64, n)
                };
                pr_info!("asahi/vtxcmd size {:#x} bytes\n", n * 8);
                // 32.41.616: TA register list for a diff against Apple's
                // (2026-09-11-ta-capture/out/vtx-draw-triangle-64x64.txt).
                #[ver(G >= G14X)]
                {
                    let r = &raw.registers;
                    let rn = r.count as usize;
                    pr_info!("asahi/vtxregs count {}\n", rn);
                    for i in 0..rn.min(150) {
                        let e = &r.registers[i];
                        let (num, val) = ({ e.number }, { e.value });
                        let v = val.0;
                        pr_info!("asahi/vtxregs [{:3}] {:#07x} = {:#018x}\n", i, num, v);
                    }
                }
                let mut i = 0usize;
                while i < n {
                    let a = q[i];
                    let b = if i + 1 < n { q[i + 1] } else { 0 };
                    let c = if i + 2 < n { q[i + 2] } else { 0 };
                    let d = if i + 3 < n { q[i + 3] } else { 0 };
                    pr_info!(
                        "asahi/vtxcmd {:#06x}: {:#018x} {:#018x} {:#018x} {:#018x}\n",
                        i * 8, a, b, c, d
                    );
                    i += 4;
                }
            });
        }

        mod_dev_dbg!(self.dev, "[Submission {}] Add Vertex\n", id);
        fence.add_command();
        vtx_job.add_cb(vtx, vm_bind.slot(), move |error| {
            {
                let ts = prof_vts.with(|raw, _| {
                    (raw.vtx.start.load(Ordering::Relaxed), raw.vtx.end.load(Ordering::Relaxed))
                });
                crate::stats::pass_prof_vtx(prof.0, prof.1, prof.2, prof.3, ts);
            }
            if let Some(err) = error {
                fence.set_error(err.into())
            }

            fence.command_complete();
        })?;

        mod_dev_dbg!(self.dev, "[Submission {}] Increment counters\n", id);

        // TODO: handle rollbacks, move to job submit?
        buffer.increment();

        job.get_vtx()?.next_seq();
        job.get_frag()?.next_seq();

        Ok(())
    }
}
