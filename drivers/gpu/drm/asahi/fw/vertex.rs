// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU vertex job firmware structures

use super::types::*;
use super::{
    event,
    job,
    workqueue, //
};
use crate::{
    buffer,
    fw,
    microseq,
    mmu, //
};
use kernel::sync::Arc;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug, Default, Copy, Clone)]
    #[repr(C)]
    pub(crate) struct TilingParameters {
        pub(crate) rgn_size: u32,
        pub(crate) unk_4: u32,
        pub(crate) ppp_ctrl: u32,
        pub(crate) x_max: u16,
        pub(crate) y_max: u16,
        pub(crate) te_screen: u32,
        pub(crate) te_mtile1: u32,
        pub(crate) te_mtile2: u32,
        pub(crate) tiles_per_mtile: u32,
        pub(crate) tpc_stride: u32,
        pub(crate) unk_24: u32,
        pub(crate) unk_28: u32,
        pub(crate) helper_cfg: u32,
        pub(crate) __pad: Pad<0x70>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters1<'a> {
        pub(crate) unk_0: U64,
        pub(crate) unk_8: F32,
        pub(crate) unk_c: F32,
        pub(crate) tvb_tilemap: GpuPointer<'a, &'a [u8]>,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_tilemaps: Option<GpuPointer<'a, &'a [u8]>>,
        pub(crate) tpc: GpuPointer<'a, &'a [u8]>,
        pub(crate) tvb_heapmeta: GpuPointer<'a, &'a [u8]>,
        pub(crate) iogpu_unk_54: U64,
        pub(crate) iogpu_unk_56: U64,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_meta1: Option<GpuPointer<'a, &'a [u8]>>,
        pub(crate) utile_config: u32,
        pub(crate) unk_4c: u32,
        pub(crate) ppp_multisamplectl: U64,
        pub(crate) tvb_layermeta: GpuPointer<'a, &'a [u8]>,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_layermeta: Option<GpuPointer<'a, &'a [u8]>>,
        #[ver(G < G14)]
        pub(crate) core_mask: Array<2, u32>,
        pub(crate) preempt_buf1: GpuPointer<'a, &'a [u8]>,
        pub(crate) preempt_buf2: GpuPointer<'a, &'a [u8]>,
        pub(crate) unk_80: U64,
        pub(crate) preempt_buf3: GpuPointer<'a, &'a [u8]>,
        pub(crate) vdm_ctrl_stream_base: U64,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_meta2: Option<GpuPointer<'a, &'a [u8]>>,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_meta3: Option<GpuPointer<'a, &'a [u8]>>,
        #[ver(G < G14)]
        pub(crate) tiling_control: u32,
        #[ver(G < G14)]
        pub(crate) unk_ac: u32,
        pub(crate) unk_b0: Array<6, U64>,
        pub(crate) usc_exec_base_ta: U64,
        #[ver(G < G14)]
        pub(crate) tvb_cluster_meta4: Option<GpuPointer<'a, &'a [u8]>>,
        #[ver(G < G14)]
        pub(crate) unk_f0: U64,
        pub(crate) unk_f8: U64,
        pub(crate) helper_program: u32,
        pub(crate) unk_104: u32,
        pub(crate) helper_arg: U64,
        pub(crate) unk_110: U64,
        pub(crate) unk_118: u32,
        #[ver(G >= G14)]
        pub(crate) __pad: Pad<{ 8 * 9 + 0x268 }>,
        #[ver(G < G14)]
        pub(crate) __pad: Pad<0x268>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters2<'a> {
        pub(crate) unk_480: Array<4, u32>,
        pub(crate) unk_498: U64,
        pub(crate) unk_4a0: u32,
        pub(crate) preempt_buf1: GpuPointer<'a, &'a [u8]>,
        pub(crate) unk_4ac: u32,
        pub(crate) unk_4b0: U64,
        pub(crate) unk_4b8: u32,
        pub(crate) unk_4bc: U64,
        // 0x10 shorter on 15.5. The byte count is forced -- `meta` has to keep
        // the offset the firmware reads its DAG sequence word at, and the
        // register block ahead of it grew by exactly that much -- but which
        // field 15.5 actually took it out of is not known, so it comes off a
        // run of padding, leaving every named field on both sides in place.
        #[ver(V >= V15_5)]
        pub(crate) unk_4c4_padding: Array<0x38, u8>,
        #[ver(V < V15_5)]
        pub(crate) unk_4c4_padding: Array<0x48, u8>,
        pub(crate) unk_50c: u32,
        pub(crate) unk_510: U64,
        pub(crate) unk_518: U64,
        pub(crate) unk_520: U64,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RunVertex<'a> {
        pub(crate) tag: workqueue::CommandType,

        #[ver(V >= V13_0B4)]
        pub(crate) counter: U64,

        pub(crate) vm_slot: u32,
        pub(crate) unk_8: u32,
        pub(crate) notifier: GpuPointer<'a, event::Notifier::ver>,
        pub(crate) buffer_slot: u32,
        pub(crate) unk_1c: u32,
        pub(crate) buffer: GpuPointer<'a, fw::buffer::Info::ver>,
        pub(crate) scene: GpuPointer<'a, fw::buffer::Scene::ver>,
        pub(crate) unk_buffer_buf: GpuWeakPointer<[u8]>,
        pub(crate) unk_34: u32,

        #[ver(G < G14X)]
        pub(crate) job_params1: JobParameters1::ver<'a>,
        #[ver(G < G14X)]
        pub(crate) tiling_params: TilingParameters,
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

        pub(crate) tpc: GpuPointer<'a, &'a [u8]>,
        pub(crate) tpc_size: U64,
        pub(crate) microsequence: GpuPointer<'a, &'a [u8]>,
        pub(crate) microsequence_size: u32,
        pub(crate) fragment_stamp_slot: u32,
        pub(crate) fragment_stamp_value: EventValue,
        pub(crate) unk_pointee: u32,
        pub(crate) unk_pad: u32,
        pub(crate) job_params2: JobParameters2::ver<'a>,
        pub(crate) encoder_params: job::raw::EncoderParams,
        pub(crate) unk_55c: u32,
        pub(crate) unk_560: u32,
        pub(crate) sync_grow: u32,
        pub(crate) unk_568: u32,
        pub(crate) uses_scratch: u32,
        pub(crate) meta: job::raw::JobMeta::ver,
        pub(crate) unk_after_meta: u32,
        pub(crate) unk_buf_0: U64,
        pub(crate) unk_buf_8: U64,
        pub(crate) unk_buf_10: U64,
        // 32.41.511: 0x10 the firmware's vertex command start routine never
        // touches.  That routine (x21 = command, pinned by `[x21, #0x8e6]` =
        // uma_pool_flush at fw:0x1ce10) prefetches cmd+0x898, then works on
        // cmd+0x8b0 (pstl1keep -- it WRITES there), cmd+0x8b8 and cmd+0x8c0
        // (`ldr x9` + `prfm pstl1keep, [x9]` -- two POINTERS it writes
        // through), cmd+0x8c8 and cmd+0x8d0.  It never reads 0x8a0 or 0x8a8,
        // where this driver had `command_time` and `timestamp_pointers`.
        #[ver(V >= V15_5)]
        pub(crate) unk_8a0: Array<0x10, u8>,
        pub(crate) command_time: U64,
        pub(crate) timestamp_pointers: job::raw::TimestampPointers<'a>,
        // On 15.5 there is only ONE TimestampPointers here: two will not fit
        // between `command_time` and the firmware-anchored `uma_pool` at
        // 0x8de.  Apple's Timestamp record passes NULL for `user_ts_pointers`,
        // so the field's contents are inert; it is kept, in the tail filler,
        // purely so the user-timestamp plumbing still compiles.
        #[ver(V < V15_5)]
        pub(crate) user_timestamp_pointers: job::raw::TimestampPointers<'a>,
        pub(crate) client_sequence: u8,
        pub(crate) pad_5d5: Array<3, u8>,
        pub(crate) unk_5d8: u32,
        pub(crate) unk_5dc: u8,

        #[ver(V >= V13_0B4 && V < V15_5)]
        pub(crate) unk_ts: U64,

        #[ver(V >= V13_0B4 && V < V15_5)]
        pub(crate) unk_5dd_8: Array<0x1b, u8>,

        // 15.5 uses this run of bytes. The vertex start routine reads an
        // unaligned pool pointer at cmd+0x8de and its flush flag at cmd+0x8e6,
        // and hands both to the UMA pool updater, which panics on null.
        // LOG.md 32.41.12.
        // 32.41.506: on 15.5 `unk_ts` is NOT here. Apple's Timestamp record puts
        // it at cmd+0x900 (see below), so these 8 bytes are part of the filler
        // run before the pool pointer. 0x5 + 8 = 0xd.
        #[ver(V >= V15_5)]
        pub(crate) unk_5dd_8: Array<0xd, u8>,
        #[ver(V >= V15_5)]
        pub(crate) uma_pool: U64,
        #[ver(V >= V15_5)]
        pub(crate) uma_pool_flush: u8,
        // 0x19, not 0xd: the vertex command start routine's last touch of the
        // command is `ldrb w9, [x21, #0x8ff]` at fw:0x1cf4c, eleven bytes past
        // where the driver's RunVertex used to end (0x8f4).  Every command has
        // the same sub-block here -- pool pointer, its flush flag at +8, and a
        // byte at +0x21 -- and only the vertex one was too short to hold it:
        // compute reads 0x842/0x84a/0x863 and fragment 0x936/0x93e/0x957, both
        // inside their existing tails.  LOG.md 32.41.32.
        #[ver(V >= V15_5)]
        pub(crate) unk_5e7: Array<0x19, u8>,
        #[ver(V >= V15_5)]
        pub(crate) user_timestamp_pointers: job::raw::TimestampPointers<'a>,

        // The command dispatcher flushes a fixed length per tag, read straight
        // off `fw:0x19290`: RunVertex **0x920**, RunFragment 0xc80, RunCompute
        // 0x8a0, Barrier 0x38, InitBuffer 0x20. Only the vertex struct was
        // shorter than its flush (0x900), so the firmware was cleaning 0x20
        // bytes past the end of the object. LOG.md 32.41.96.
        // 32.41.506: `unk_ts` lives HERE on 15.5, at cmd+0x900 -- derived from
        // Apple's own Timestamp microsequence record, which points it at
        // `command_time + 0x60`. With `command_time` at 0x8a0 that is 0x900,
        // the first 8 bytes of what 32.41.96 could only call `unk_900`.
        #[ver(V >= V15_5)]
        pub(crate) unk_ts: U64,
        #[ver(V >= V15_5)]
        pub(crate) unk_900: Array<0x18, u8>,
    }
}

// Pinned against the firmware: the vertex command start routine reads the UMA
// pool pointer at cmd+0x8de (fw:0x1ce08), its flush flag at cmd+0x8e6
// (fw:0x1ce10) and a byte at cmd+0x8ff (fw:0x1cf4c), so the command must be at
// least 0x900 long.  LOG.md 32.41.31/32.
const _: () = {
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, uma_pool) == 0x8de);
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, uma_pool_flush) == 0x8e6);
    assert!(core::mem::size_of::<raw::RunVertexG15XV15_5<'static>>() == 0x930);

    // 32.41.511: the command's firmware VA base is 0xfffffc20c00a8700, NOT the
    // 0xfffffc20c00a8710 that 32.41.506 derived.  .506 solved for the base using
    // only the Timestamp record and this driver's then-current offsets, which is
    // circular -- it recovered the offsets it assumed.  Solving instead against
    // every pointer in the captured microsequence, and against offsets the
    // FIRMWARE fixes absolutely:
    //
    //   base 0xfffffc20c00a8700 -> 12 of 12 land on a field boundary
    //   base 0xfffffc20c00a8710 ->  1 of 12
    //
    //   StartVertex    op+0x14 -> cmd+0x040 registers   (RegisterArray header
    //                             0x20000010141 is at cmd+0x40 in Apple's own
    //                             command dump; the handler then reads
    //                             ptr+0x700/+0x708, inside it)
    //                  op+0x3c -> cmd+0x84c sync_grow   (what the driver itself
    //                             puts there)
    //                  op+0x64 -> cmd+0x784 unk_pointee
    //                  op+0x6c -> cmd+0x898 unk_buf_10
    //   FinalizeVertex +0x24   -> cmd+0x84c sync_grow
    //                  +0x34   -> cmd+0x784 unk_pointee
    //   Timestamp      +0x04   -> cmd+0x8b0 command_time
    //                  +0x0c   -> cmd+0x8b8 timestamp_pointers
    //        #1/#2     +0x14   -> cmd+0x8b8 / cmd+0x8c0  (.start / .end, the
    //                             documented 8-byte delta between the records)
    //                  +0x2c   -> cmd+0x910 unk_ts
    //
    // Independently, the firmware's vertex command start routine ignores
    // cmd+0x8a0 and cmd+0x8a8 entirely and operates on 0x8b0/0x8b8/0x8c0 -- so
    // the 0x10 at 0x8a0 is filler and the single TimestampPointers is at 0x8b8.
    // The whole tail then closes exactly onto the firmware-anchored
    // `uma_pool` at 0x8de.
    //
    // .506's `unk_ts` move (0x8d1 -> 0x900) was therefore right in direction and
    // 0x10 short.
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, unk_8a0) == 0x8a0);
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, command_time) == 0x8b0);
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, timestamp_pointers) == 0x8b8);
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, unk_buf_10) == 0x898);
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, unk_pointee) == 0x784);
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, sync_grow) == 0x84c);
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, unk_buf_0) == 0x888);
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, user_timestamp_pointers) == 0x900);
    assert!(core::mem::offset_of!(raw::RunVertexG15XV15_5<'static>, unk_ts) == 0x910);
};

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct RunVertex {
    pub(crate) notifier: Arc<GpuObject<event::Notifier::ver>>,
    pub(crate) scene: Arc<buffer::Scene::ver>,
    pub(crate) micro_seq: microseq::MicroSequence,
    pub(crate) vm_bind: mmu::VmBind,
    pub(crate) timestamps: Arc<GpuObject<job::RenderTimestamps>>,
    pub(crate) user_timestamps: job::UserTimestamps,
}

#[versions(AGX)]
impl GpuStruct for RunVertex::ver {
    type Raw<'a> = raw::RunVertex::ver<'a>;
}

#[versions(AGX)]
impl workqueue::Command for RunVertex::ver {}

#[versions(AGX)]
impl Drop for RunVertex::ver {
    /// The firmware stamps `vtx.start` when it enters the vertex pass and
    /// `vtx.end` when it leaves. A start with no end is the difference between
    /// "the pass never began" and "the pass began and hung" -- and the render
    /// job objects are only freed on teardown, which is exactly when we want to
    /// read them. LOG.md 32.41.89.
    fn drop(&mut self) {
        self.timestamps.with(|raw, _inner| {
            crate::sub_pr_info!(
                "asahi/hung: RenderTimestamps vtx {:#x}..{:#x} frag {:#x}..{:#x} scene {:#x}\n",
                raw.vtx.start.load(core::sync::atomic::Ordering::Relaxed),
                raw.vtx.end.load(core::sync::atomic::Ordering::Relaxed),
                raw.frag.start.load(core::sync::atomic::Ordering::Relaxed),
                raw.frag.end.load(core::sync::atomic::Ordering::Relaxed),
                u64::from(self.scene.weak_pointer()),
            );
        });
    }
}


