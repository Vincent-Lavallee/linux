// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU compute job firmware structures

use super::types::*;
use super::{
    event,
    job,
    workqueue, //
};
use crate::{
    microseq,
    mmu, //
};
use kernel::sync::Arc;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters1<'a> {
        pub(crate) preempt_buf1: GpuPointer<'a, &'a [u8]>,
        pub(crate) cdm_ctrl_stream_base: U64,
        pub(crate) preempt_buf2: GpuPointer<'a, &'a [u8]>,
        pub(crate) preempt_buf3: GpuPointer<'a, &'a [u8]>,
        pub(crate) preempt_buf4: GpuPointer<'a, &'a [u8]>,
        pub(crate) preempt_buf5: GpuPointer<'a, &'a [u8]>,
        pub(crate) usc_exec_base_cp: U64,
        pub(crate) unk_38: U64,
        pub(crate) helper_program: u32,
        pub(crate) unk_44: u32,
        pub(crate) helper_arg: U64,
        pub(crate) helper_cfg: u32,
        pub(crate) unk_54: u32,
        pub(crate) unk_58: u32,
        pub(crate) unk_5c: u32,
        pub(crate) iogpu_unk_40: u32,
        pub(crate) __pad: Pad<0xfc>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters2<'a> {
        #[ver(V >= V13_0B4)]
        pub(crate) unk_0_0: u32,
        pub(crate) unk_0: Array<0x24, u8>,
        pub(crate) preempt_buf1: GpuPointer<'a, &'a [u8]>,
        pub(crate) cdm_ctrl_stream_end: U64,
        // 0x20 on 15.5 too (Apple's command, LOG.md 32.41.592): the compute
        // resume path (fw:0x17998, x19 = op 0xb's job_params2) programs CDM
        // control 0x1a440 from the u64 at +0x58 = {unk_g14x, unk_58} | 8.
        // Apple: 0x54024201 / 1 = its 0x1a440 0x154024201. With a 4-byte
        // unk_34 the resume read EncoderParams instead and the resumed job
        // never answered the next store. LOG.md 32.41.667zzy.
        pub(crate) unk_34: Array<0x20, u8>,
        pub(crate) unk_g14x: u32,
        pub(crate) unk_58: u32,
        #[ver(V < V13_0B4)]
        pub(crate) unk_5c: u32,
    }

    /// Compute's EncoderParams: 15.5 compute commands have no sampler fields
    /// (Apple's command, LOG.md 32.41.592); the 0x10 bytes went to
    /// JobParameters2.unk_34.
    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct EncoderParams {
        pub(crate) unk_8: u32,
        pub(crate) sync_grow: u32,
        pub(crate) unk_10: u32,
        pub(crate) encoder_id: u32,
        pub(crate) unk_18: u32,
        pub(crate) unk_mask: u32,
        #[ver(V < V15_5)]
        pub(crate) sampler_array: U64,
        #[ver(V < V15_5)]
        pub(crate) sampler_count: u32,
        #[ver(V < V15_5)]
        pub(crate) sampler_max: u32,
    }

    /// Compute's JobMeta: 15.5 compute commands have no 0xc-byte `unk_4`
    /// (Apple's command, LOG.md 32.41.592), so `unk_2`/`no_preemption` sit
    /// at cmd+0x7e6/0x7e7 -- the byte the compute start routine reads
    /// (fw:0x1ea48) -- and the stamp stays at cmd+0x7e8.
    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobMeta {
        pub(crate) unk_0: u16,
        pub(crate) unk_2: u8,
        pub(crate) no_preemption: u8,
        pub(crate) stamp: GpuWeakPointer<Stamp>,
        pub(crate) fw_stamp: GpuWeakPointer<FwStamp>,
        pub(crate) stamp_value: EventValue,
        pub(crate) stamp_slot: u32,
        #[ver(V < V15_5)]
        pub(crate) evctl_index: u32,
        #[ver(V < V15_5)]
        pub(crate) flush_stamps: u32,
        #[ver(V < V15_5)]
        pub(crate) uuid: u32,
        pub(crate) event_seq: u32,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RunCompute<'a> {
        pub(crate) tag: workqueue::CommandType,

        #[ver(V >= V13_0B4)]
        pub(crate) counter: U64,

        pub(crate) unk_4: u32,
        pub(crate) vm_slot: u32,
        pub(crate) notifier: GpuPointer<'a, event::Notifier::ver>,
        pub(crate) unk_pointee: u32,
        #[ver(G < G14X)]
        pub(crate) __pad0: Array<0x50, u8>,
        #[ver(G < G14X)]
        pub(crate) job_params1: JobParameters1<'a>,
        #[ver(G >= G14X)]
        pub(crate) registers: job::raw::RegisterArray,

        // The register block is 0x720 bytes on G15X, not the 0x710 that
        // `RegisterArray` describes. Three readings agree: the vertex start
        // routine takes the microsequence from `[cmd+0x770]` (0x1ccd8), the
        // selector at 0x1cf88 takes it from `[cmd+0x760]` for compute, and the
        // timestamp pointer pair that routine walks sits 0x10 later than the
        // driver had it. The extra 0x10 is carried here rather than inside
        // `RegisterArray` because that type is not versioned and is reached
        // through `GpuWeakPointer` from the microsequence. LOG.md 32.41.6.
        #[ver(V >= V15_5)]
        pub(crate) pad_regs_15_5: Array<0x10, u8>,
        pub(crate) __pad1: Array<0x20, u8>,
        pub(crate) microsequence: GpuPointer<'a, &'a [u8]>,
        pub(crate) microsequence_size: u32,
        pub(crate) job_params2: JobParameters2::ver<'a>,
        pub(crate) encoder_params: EncoderParams::ver,
        pub(crate) meta: JobMeta::ver,
        pub(crate) command_time: U64,
        pub(crate) timestamp_pointers: job::raw::TimestampPointers<'a>,
        pub(crate) user_timestamp_pointers: job::raw::TimestampPointers<'a>,
        pub(crate) client_sequence: u8,
        pub(crate) pad_2d1: Array<3, u8>,
        pub(crate) unk_2d4: u32,
        pub(crate) unk_2d8: u8,
        #[ver(V >= V13_0B4)]
        pub(crate) context_store_req: U64,
        #[ver(V >= V13_0B4 && V < V15_5)]
        pub(crate) context_store_compl: U64,
        #[ver(V >= V13_0B4 && V < V15_5)]
        pub(crate) unk_2e9: Array<0x14, u8>,

        // 15.5 uses this run of bytes. The compute command start routine reads
        // an unaligned pool pointer at cmd+0x842 and its flush flag at
        // cmd+0x84a (fw:0x1eaf0, with x24 = cmd+0x81c set at fw:0x1ea64) and
        // hands both to the UMA pool updater at 0x24cf0, which panics on null.
        // Exactly the shape of RunVertex's cmd+0x8de/0x8e6. The 0x1c bytes
        // here are the same 0x1c that context_store_compl + unk_2e9 covered,
        // so nothing after them moves. LOG.md 32.41.31.
        #[ver(V >= V15_5)]
        pub(crate) unk_83d: Array<0x5, u8>,
        #[ver(V >= V15_5)]
        pub(crate) uma_pool: U64,
        #[ver(V >= V15_5)]
        pub(crate) uma_pool_flush: u8,
        #[ver(V >= V15_5)]
        pub(crate) unk_84b: Array<0xe, u8>,
        #[ver(V >= V13_0B4)]
        pub(crate) unk_flag: U32,
        #[ver(V >= V13_0B4)]
        pub(crate) unk_pad: Array<0x10, u8>,

        // See RunFragment: the 15.5 microsequence needs a pool and a word
        // holding its address.  LOG.md 32.41.26.
        #[ver(V >= V15_5)]
        pub(crate) pool_self: U64,
        #[ver(V >= V15_5)]
        pub(crate) pool: Array<0x90, u8>,
    }
}

// Pinned against the firmware: the compute command start routine reads the UMA
// pool pointer at cmd+0x842 (fw:0x1eaf0) and its flush flag at cmd+0x84a
// (fw:0x1eaf4).  LOG.md 32.41.31.
const _: () = {
    assert!(core::mem::offset_of!(raw::RunComputeG15XV15_5<'static>, uma_pool) == 0x842);
    assert!(core::mem::offset_of!(raw::RunComputeG15XV15_5<'static>, uma_pool_flush) == 0x84a);
};

/// comp_apple's compute UMA pool pages: backing, free-page ring and extents,
/// one set per queue (the per-job record page points at them). The hardware
/// frees pages to `list[tail % capacity]` with `tail` starting at `pages` and
/// capacity = record+0x1c = 0x100000, so the list holds the full capacity --
/// sized to `pages` it spilled freed-page pointers past its end (the 5th job
/// faulted at list+0x111c0). LOG.md 32.41.664.
#[derive(Debug)]
pub(crate) struct ComputePool {
    pub(crate) backing: GpuArray<u8>,
    pub(crate) list: GpuArray<u8>,
    pub(crate) exts: GpuArray<u8>,
    /// .667zzz: the pool record (DC 0x1e +0xc), device-global.
    pub(crate) rec: GpuArray<u8>,
    pub(crate) pages: usize,
    pub(crate) nblk: usize,
}

/// Free-page ring capacity (record+0x1c), Apple's value. LOG.md 32.41.599.
pub(crate) const CPOOL_RING_CAP: usize = 0x100000;

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct RunCompute {
    pub(crate) notifier: Arc<GpuObject<event::Notifier::ver>>,
    pub(crate) preempt_buf: GpuArray<u8>,
    // 32.41.572: comp_apple's spill buffer and UMA scratch, kept alive with the job.
    pub(crate) spill_buf: Option<GpuArray<u8>>,
    pub(crate) uma_cbuf: Option<GpuArray<u8>>,
    pub(crate) cpool: Option<Arc<ComputePool>>,
    pub(crate) micro_seq: microseq::MicroSequence,
    pub(crate) vm_bind: mmu::VmBind,
    pub(crate) timestamps: Arc<GpuObject<job::JobTimestamps>>,
    pub(crate) user_timestamps: job::UserTimestamps,
    /// .667zzz9 `ts_dbg` positive-control target (kernel-shared, CPU-readable).
    pub(crate) ts_dbg_obj: Option<Arc<GpuObject<job::JobTimestamps>>>,
}

#[versions(AGX)]
impl GpuStruct for RunCompute::ver {
    type Raw<'a> = raw::RunCompute::ver<'a>;
}

#[versions(AGX)]
impl workqueue::Command for RunCompute::ver {}

