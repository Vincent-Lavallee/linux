// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU fragment job firmware structures

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

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct BackgroundProgram {
        pub(crate) rsrc_spec: U64,
        pub(crate) address: U64,
    }

    #[derive(Debug, Clone, Copy, Default)]
    #[repr(C)]
    pub(crate) struct EotProgram {
        pub(crate) unk_0: U64,
        pub(crate) unk_8: u32,
        pub(crate) rsrc_spec: u32,
        pub(crate) unk_10: u32,
        pub(crate) address: u32,
        pub(crate) unk_18: u32,
        pub(crate) unk_1c_padding: u32,
    }

    impl EotProgram {
        pub(crate) fn new(rsrc_spec: u32, address: u32) -> EotProgram {
            EotProgram {
                rsrc_spec,
                address,
                ..Default::default()
            }
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct ArrayAddr {
        pub(crate) ptr: U64,
        pub(crate) unk_padding: U64,
    }

    #[versions(AGX)]
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct AuxFBInfo {
        pub(crate) isp_ctl: u32,
        pub(crate) unk2: u32,
        pub(crate) width: u32,
        pub(crate) height: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk3: U64,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters1<'a> {
        pub(crate) utile_config: u32,
        pub(crate) unk_4: u32,
        pub(crate) bg: BackgroundProgram,
        pub(crate) ppp_multisamplectl: U64,
        pub(crate) isp_scissor_base: U64,
        pub(crate) isp_dbias_base: U64,
        pub(crate) aux_fb_info: AuxFBInfo::ver,
        pub(crate) isp_zls_pixels: U64,
        pub(crate) isp_oclqry_base: U64,
        pub(crate) zls_ctrl: U64,

        #[ver(G >= G14)]
        pub(crate) unk_58_g14_0: U64,
        #[ver(G >= G14)]
        pub(crate) unk_58_g14_8: U64,

        pub(crate) z_load: U64,
        pub(crate) z_store: U64,
        pub(crate) s_load: U64,
        pub(crate) s_store: U64,

        #[ver(G >= G14)]
        pub(crate) unk_68_g14_0: Array<0x20, u8>,

        pub(crate) z_load_stride: U64,
        pub(crate) z_store_stride: U64,
        pub(crate) s_load_stride: U64,
        pub(crate) s_store_stride: U64,
        pub(crate) z_load_comp: U64,
        pub(crate) z_load_comp_stride: U64,
        pub(crate) z_store_comp: U64,
        pub(crate) z_store_comp_stride: U64,
        pub(crate) s_load_comp: U64,
        pub(crate) s_load_comp_stride: U64,
        pub(crate) s_store_comp: U64,
        pub(crate) s_store_comp_stride: U64,
        pub(crate) tvb_tilemap: GpuPointer<'a, &'a [u8]>,
        pub(crate) tvb_layermeta: GpuPointer<'a, &'a [u8]>,
        pub(crate) mtile_stride_dwords: U64,
        pub(crate) tvb_heapmeta: GpuPointer<'a, &'a [u8]>,
        pub(crate) tile_config: U64,
        pub(crate) aux_fb: GpuPointer<'a, &'a [u8]>,
        pub(crate) unk_108: Array<0x6, U64>,
        pub(crate) usc_exec_base_isp: U64,
        pub(crate) unk_140: U64,
        pub(crate) helper_program: u32,
        pub(crate) unk_14c: u32,
        pub(crate) helper_arg: U64,
        pub(crate) unk_158: U64,
        pub(crate) unk_160: U64,

        #[ver(G < G14)]
        pub(crate) __pad: Pad<0x1d8>,
        #[ver(G >= G14)]
        pub(crate) __pad: Pad<0x1a8>,
        #[ver(V < V13_0B4)]
        pub(crate) __pad1: Pad<0x8>,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters2 {
        pub(crate) eot_rsrc_spec: u32,
        pub(crate) eot_usc: u32,
        pub(crate) unk_8: u32,
        pub(crate) unk_c: u32,
        pub(crate) isp_merge_upper_x: F32,
        pub(crate) isp_merge_upper_y: F32,
        pub(crate) unk_18: U64,
        pub(crate) utiles_per_mtile_y: u16,
        pub(crate) utiles_per_mtile_x: u16,
        pub(crate) unk_24: u32,
        pub(crate) tile_counts: u32,
        pub(crate) tib_blocks: u32,
        pub(crate) isp_bgobjdepth: u32,
        pub(crate) isp_bgobjvals: u32,
        pub(crate) unk_38: u32,
        pub(crate) unk_3c: u32,
        pub(crate) helper_cfg: u32,
        pub(crate) __pad: Pad<0xac>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters3 {
        pub(crate) isp_dbias_base: ArrayAddr,
        pub(crate) isp_scissor_base: ArrayAddr,
        pub(crate) isp_oclqry_base: U64,
        pub(crate) unk_118: U64,
        pub(crate) unk_120: Array<0x25, U64>,
        pub(crate) unk_partial_bg: BackgroundProgram,
        pub(crate) unk_258: U64,
        pub(crate) unk_260: U64,
        pub(crate) unk_268: U64,
        pub(crate) unk_270: U64,
        pub(crate) partial_bg: BackgroundProgram,
        pub(crate) zls_ctrl: U64,
        pub(crate) unk_290: U64,
        pub(crate) z_load: U64,
        pub(crate) z_partial_stride: U64,
        pub(crate) z_partial_comp_stride: U64,
        pub(crate) z_store: U64,
        pub(crate) z_partial: U64,
        pub(crate) z_partial_comp: U64,
        pub(crate) s_load: U64,
        pub(crate) s_partial_stride: U64,
        pub(crate) s_partial_comp_stride: U64,
        pub(crate) s_store: U64,
        pub(crate) s_partial: U64,
        pub(crate) s_partial_comp: U64,
        pub(crate) unk_2f8: Array<2, U64>,
        pub(crate) tib_blocks: u32,
        pub(crate) unk_30c: u32,
        pub(crate) aux_fb_info: AuxFBInfo::ver,
        pub(crate) tile_config: U64,
        pub(crate) unk_328_padding: Array<0x8, u8>,
        pub(crate) unk_partial_eot: EotProgram,
        pub(crate) partial_eot: EotProgram,
        pub(crate) isp_bgobjdepth: u32,
        pub(crate) isp_bgobjvals: u32,
        pub(crate) sample_size: u32,
        pub(crate) unk_37c: u32,
        pub(crate) unk_380: U64,
        pub(crate) unk_388: U64,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_390_0: U64,

        pub(crate) isp_zls_pixels: U64,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RunFragment<'a> {
        pub(crate) tag: workqueue::CommandType,

        #[ver(V >= V13_0B4)]
        pub(crate) counter: U64,

        pub(crate) vm_slot: u32,
        pub(crate) unk_8: u32,
        pub(crate) microsequence: GpuPointer<'a, &'a [u8]>,
        pub(crate) microsequence_size: u32,
        pub(crate) notifier: GpuPointer<'a, event::Notifier::ver>,
        pub(crate) buffer: GpuPointer<'a, fw::buffer::Info::ver>,
        pub(crate) scene: GpuPointer<'a, fw::buffer::Scene::ver>,
        pub(crate) unk_buffer_buf: GpuWeakPointer<[u8]>,
        pub(crate) tvb_tilemap: GpuPointer<'a, &'a [u8]>,
        pub(crate) ppp_multisamplectl: U64,
        pub(crate) samples: u32,
        pub(crate) tiles_per_mtile_y: u16,
        pub(crate) tiles_per_mtile_x: u16,
        pub(crate) unk_50: U64,
        pub(crate) unk_58: U64,
        pub(crate) isp_merge_upper_x: F32,
        pub(crate) isp_merge_upper_y: F32,
        pub(crate) unk_68: U64,
        pub(crate) tile_count: U64,

        #[ver(G < G14X)]
        pub(crate) job_params1: JobParameters1::ver<'a>,
        #[ver(G < G14X)]
        pub(crate) job_params2: JobParameters2,
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

        pub(crate) job_params3: JobParameters3::ver,
        pub(crate) unk_758_flag: u32,
        pub(crate) unk_75c_flag: u32,
        // 0x1c shorter on 15.5; same reasoning as RunVertex's padding.
        #[ver(V >= V15_5)]
        pub(crate) unk_buf: Array<0xf4, u8>,
        #[ver(V < V15_5)]
        pub(crate) unk_buf: Array<0x110, u8>,
        pub(crate) busy_flag: u32,
        pub(crate) tvb_overflow_count: u32,
        pub(crate) unk_878: u32,
        pub(crate) encoder_params: job::raw::EncoderParams,
        pub(crate) process_empty_tiles: u32,
        pub(crate) no_clear_pipeline_textures: u32,
        pub(crate) msaa_zs: u32,
        pub(crate) unk_pointee: u32,
        #[ver(V >= V13_3)]
        pub(crate) unk_v13_3: u32,
        pub(crate) meta: job::raw::JobMeta::ver,
        pub(crate) unk_after_meta: u32,
        pub(crate) unk_buf_0: U64,
        pub(crate) unk_buf_8: U64,
        pub(crate) unk_buf_10: U64,
        pub(crate) command_time: U64,
        pub(crate) timestamp_pointers: job::raw::TimestampPointers<'a>,
        pub(crate) user_timestamp_pointers: job::raw::TimestampPointers<'a>,
        pub(crate) client_sequence: u8,
        pub(crate) pad_925: Array<3, u8>,
        pub(crate) unk_928: u32,
        pub(crate) unk_92c: u8,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_ts: U64,

        #[ver(V >= V13_0B4 && V < V15_5)]
        pub(crate) unk_92d_8: Array<0x1b, u8>,

        // 15.5 uses this run of bytes for the same tail block RunVertex and
        // RunCompute carry: an unaligned UMA pool pointer, its flush flag 8
        // bytes later, and a byte 0x19 after that.  The shared command start
        // routine at fw:0x1d960 dispatches on the command tag at cmd+0 and, for
        // tag 1 (RunFragment), reads them at cmd+0xc22 (fw:0x1dc18, x24 =
        // cmd+0x8cc), cmd+0xc2a and cmd+0xc43.  LOG.md 32.41.33.
        #[ver(V >= V15_5)]
        pub(crate) unk_92d_8: Array<0x5, u8>,
        #[ver(V >= V15_5)]
        pub(crate) uma_pool: U64,
        #[ver(V >= V15_5)]
        pub(crate) uma_pool_flush: u8,
        #[ver(V >= V15_5)]
        pub(crate) unk_c2b: Array<0x19, u8>,

        // 15.5's microsequence needs two things the pre-15.5 command has
        // nowhere to put: a pool for `StartFragment + 0x1a8` to point at, and
        // a *word holding that pool's address* for `FinalizeFragment + 0x90`,
        // which the firmware dereferences twice (0x2518c does `ldr x21, [x20];
        // ldr w8, [x21, #12]`).  Both live here, at the end of the command, so
        // nothing before them moves.  LOG.md 32.41.26.
        #[ver(V >= V15_5)]
        pub(crate) pool_self: U64,
        #[ver(V >= V15_5)]
        pub(crate) pool: Array<0x90, u8>,
    }
}

// Pinned against the firmware: for command tag 1 the shared start routine reads
// the UMA pool pointer at cmd+0xc22 (fw:0x1dc18), its flush flag at cmd+0xc2a
// (fw:0x1dc20) and a byte at cmd+0xc43 (fw:0x1ddb0).  LOG.md 32.41.33.
const _: () = {
    assert!(core::mem::offset_of!(raw::RunFragmentG15XV15_5<'static>, uma_pool) == 0xc22);
    assert!(core::mem::offset_of!(raw::RunFragmentG15XV15_5<'static>, uma_pool_flush) == 0xc2a);
    assert!(core::mem::offset_of!(raw::RunFragmentG15XV15_5<'static>, pool_self) == 0xc44);
};

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct RunFragment {
    pub(crate) notifier: Arc<GpuObject<event::Notifier::ver>>,
    pub(crate) scene: Arc<buffer::Scene::ver>,
    pub(crate) micro_seq: microseq::MicroSequence,
    pub(crate) vm_bind: mmu::VmBind,
    pub(crate) aux_fb: GpuArray<u8>,
    pub(crate) timestamps: Arc<GpuObject<job::RenderTimestamps>>,
    pub(crate) user_timestamps: job::UserTimestamps,
}

#[versions(AGX)]
impl GpuStruct for RunFragment::ver {
    type Raw<'a> = raw::RunFragment::ver<'a>;
}

#[versions(AGX)]
impl workqueue::Command for RunFragment::ver {}
