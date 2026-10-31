// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU tiled vertex buffer control firmware structures

use super::types::*;
use super::workqueue;
use crate::{
    default_zeroed,
    no_debug,
    trivial_gpustruct, //
};
use kernel::sync::Arc;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct BlockControl {
        pub(crate) total: AtomicU32,
        pub(crate) wptr: AtomicU32,
        pub(crate) unk: AtomicU32,
        pub(crate) pad: Pad<0x34>,
    }
    default_zeroed!(BlockControl);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Counter {
        pub(crate) count: AtomicU32,
        __pad: Pad<0x3c>,
    }
    default_zeroed!(Counter);

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct Stats {
        pub(crate) max_pages: AtomicU32,
        pub(crate) max_b: AtomicU32,
        pub(crate) overflow_count: AtomicU32,
        pub(crate) gpu_c: AtomicU32,
        pub(crate) __pad0: Pad<0x10>,
        pub(crate) reset: AtomicU32,
        pub(crate) __pad1: Pad<0x1c>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Info<'a> {
        pub(crate) gpu_counter: u32,
        pub(crate) unk_4: u32,
        pub(crate) last_id: i32,
        pub(crate) cur_id: i32,
        pub(crate) unk_10: u32,
        pub(crate) gpu_counter2: u32,
        pub(crate) unk_18: u32,

        #[ver((V < V13_0B4 || G >= G14X) && V < V15_5)]
        pub(crate) unk_1c: u32,

        pub(crate) page_list: GpuPointer<'a, &'a [u32]>,

        // 15.5 drops `unk_1c` and puts 8 bytes between the page list and its
        // size. Fitted to every offset 0x29068 touches: a pointer at 0x1c
        // flushed with the length at 0x2c, then 0x30, 0x34 (the block list's
        // entry count), 0x38 (compared against block_ctl[4]), the block list
        // pointer at 0x40, block_ctl at 0x48 and 0x54. Only this arrangement
        // puts all ten where the firmware reads them. LOG.md 32.41.15.
        #[ver(V >= V15_5)]
        pub(crate) unk_24: Array<8, u8>,

        pub(crate) page_list_size: u32,
        pub(crate) page_count: AtomicU32,
        pub(crate) max_blocks: u32,
        pub(crate) block_count: AtomicU32,
        pub(crate) unk_38: u32,
        pub(crate) block_list: GpuPointer<'a, &'a [u32]>,
        pub(crate) block_ctl: GpuPointer<'a, super::BlockControl>,
        pub(crate) last_page: AtomicU32,
        pub(crate) gpu_page_ptr1: u32,
        pub(crate) gpu_page_ptr2: u32,
        pub(crate) unk_58: u32,
        pub(crate) block_size: u32,
        pub(crate) unk_60: U64,
        pub(crate) counter: GpuPointer<'a, super::Counter>,
        pub(crate) unk_70: u32,
        pub(crate) unk_74: u32,
        pub(crate) unk_78: u32,
        pub(crate) unk_7c: u32,
        pub(crate) unk_80: u32,
        pub(crate) max_pages: u32,
        pub(crate) max_pages_nomemless: u32,
        pub(crate) unk_8c: u32,
        pub(crate) unk_90: Array<0x30, u8>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Scene<'a> {
        #[ver(G >= G14X)]
        pub(crate) control_word: GpuPointer<'a, &'a [u32]>,
        #[ver(G >= G14X)]
        pub(crate) control_word2: GpuPointer<'a, &'a [u32]>,
        pub(crate) pass_page_count: AtomicU32,
        pub(crate) unk_4: u32,
        pub(crate) unk_8: U64,
        pub(crate) unk_10: U64,
        pub(crate) user_buffer: GpuPointer<'a, &'a [u8]>,
        pub(crate) unk_20: u32,
        /// 🔑 Four bytes the G15S firmware's own accesses require and this
        /// layout did not have.  `StartVertex` reads a u32 at `scene+0x38`
        /// and branches on it (`fw:0x14fe4`), writes a u32 at `+0x3c`
        /// (`fw:0x17558`) and zeroes one at `+0x48` (`fw:0x158c4`).  Without
        /// the pad, `unk_28` sits unaligned at 0x34, `stats` at 0x3c and
        /// `total_page_count` at 0x44, so the firmware's `+0x3c` write lands
        /// in the low half of the `stats` pointer and its `+0x48` write lands
        /// past the end of the struct.  With it, `unk_28` is 8-aligned at
        /// 0x38, `stats` is at 0x40 -- the address `fw:0x14f68` prefetches --
        /// and `total_page_count` is at 0x48, which is exactly the word
        /// `StartVertex` zeroes when it starts a fresh pass.  LOG.md 32.41.248.
        #[ver(G >= G14X)]
        pub(crate) __pad_34: u32,
        #[ver(V >= V13_3)]
        pub(crate) unk_28: U64,
        pub(crate) stats: GpuWeakPointer<super::Stats>,
        pub(crate) total_page_count: AtomicU32,
        /// 667zzx: Apple's StartFragment.tvb_overflow_count points HERE (capture4,
        /// LOG.md 32.41.608): fw:0x175dc stores 1 through it when a pass overflowed,
        /// and FinalizeFragment (fw:0x167e8) reads scene+0x4c, accumulates it into
        /// work_item+0xb74 and clears it.
        #[ver(G >= G14X)]
        pub(crate) tvb_overflow: u32,
        #[ver(G < G14X)]
        pub(crate) unk_30: U64, // pad
        #[ver(G < G14X)]
        pub(crate) unk_38: U64, // pad
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct InitBuffer<'a> {
        pub(crate) tag: workqueue::CommandType,
        pub(crate) vm_slot: u32,
        pub(crate) buffer_slot: u32,
        pub(crate) unk_c: u32,
        pub(crate) block_count: u32,
        pub(crate) buffer: GpuPointer<'a, super::Info::ver>,
        pub(crate) stamp_value: EventValue,
    }
}

trivial_gpustruct!(BlockControl);
trivial_gpustruct!(Counter);
trivial_gpustruct!(Stats);

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Info {
    pub(crate) block_ctl: GpuObject<BlockControl>,
    pub(crate) counter: GpuObject<Counter>,
    pub(crate) page_list: GpuArray<u32>,
    pub(crate) block_list: GpuArray<u32>,
}

#[versions(AGX)]
impl GpuStruct for Info::ver {
    type Raw<'a> = raw::Info::ver<'a>;
}

pub(crate) struct ClusterBuffers {
    pub(crate) tilemaps: GpuArray<u8>,
    pub(crate) meta: GpuArray<u8>,
}

/// Async-grow chunk slots per UMA pool (doubling from 32 MiB reaches the cap in a few).
pub(crate) const UMA_GROW_SLOTS: usize = 16;

/// The UMA pool objects (record, free-page list, backing, extents, ...),
/// one per Buffer. Field docs: see the old per-scene fields, LOG.md 32.41.51-.664.
pub(crate) struct UmaPool {
    pub(crate) pool: GpuArray<u8>,
    pub(crate) aperture: GpuArray<u8>,
    pub(crate) scratch: GpuArray<u8>,
    pub(crate) sync: GpuArray<u8>,
    pub(crate) fwbuf: GpuArray<u8>,
    pub(crate) backing: GpuArray<u8>,
    pub(crate) extents: GpuArray<u8>,
    /// DeviceControl 0x1e already sent for this pool.
    pub(crate) announced: core::sync::atomic::AtomicBool,
    /// Owner token of the pool's hardware id (crate::buffer::uma_alloc_id).
    pub(crate) token: u64,
    /// VM slot the pool was last installed for (DC 0x1e +4).
    pub(crate) slot: core::sync::atomic::AtomicU32,
    /// 667zzw: backing chunks added by async growth (KBox::into_raw), see
    /// `crate::buffer::uma_grow_on_retire`.
    pub(crate) grow: [core::sync::atomic::AtomicPtr<GpuArray<u8>>; UMA_GROW_SLOTS],
    /// Pages pushed to the firmware so far (initial backing + grow chunks); the
    /// page list holds them contiguously from `aperture`.
    pub(crate) total_pages: core::sync::atomic::AtomicU32,
    /// A grow is in progress (serialises growth).
    pub(crate) growing: core::sync::atomic::AtomicBool,
    /// Last record ring words printed by `uma_rec_trace`.
    pub(crate) last_rec: core::sync::atomic::AtomicU64,
    /// DC 0x1e +4 / +8 of the last install, reused for grow requests.
    pub(crate) hwid: core::sync::atomic::AtomicU32,
    pub(crate) ctx: core::sync::atomic::AtomicU32,
    /// The VM's allocator, for grow chunks.
    pub(crate) ualloc: Option<Arc<kernel::sync::Mutex<crate::alloc::DefaultAllocator>>>,
    /// .667zzz43: the shared pool's second record and page list, and whether jobs use them
    /// (`crate::buffer::uma_shared_gate` resets the ring by switching records: the firmware
    /// installs a job's record whenever it is not the one in its slot table, fw:0x24cf0).
    pub(crate) pool_b: Option<GpuArray<u8>>,
    pub(crate) aperture_b: Option<GpuArray<u8>>,
    pub(crate) rec_b: core::sync::atomic::AtomicBool,
}

impl core::fmt::Debug for UmaPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UmaPool")
            .field("id", &self.id())
            .field("token", &self.token)
            .field("total_pages", &self.total_pages)
            .finish()
    }
}

impl UmaPool {
    /// The record jobs reference now (A, or B after an odd number of ring resets).
    pub(crate) fn rec(&self) -> &GpuArray<u8> {
        match &self.pool_b {
            Some(b) if self.rec_b.load(core::sync::atomic::Ordering::Acquire) => b,
            _ => &self.pool,
        }
    }

    /// Hardware pool id (record +8).
    pub(crate) fn id(&self) -> u32 {
        let p = self.pool.as_slice();
        u32::from_le_bytes([p[8], p[9], p[10], p[11]])
    }
}

impl Drop for UmaPool {
    fn drop(&mut self) {
        crate::buffer::uma_release_id(self.id(), self.token);
        for s in self.grow.iter() {
            let p = s.swap(core::ptr::null_mut(), core::sync::atomic::Ordering::AcqRel);
            if !p.is_null() {
                // SAFETY: stored by `uma_grow_on_retire` from KBox::into_raw, once.
                drop(unsafe { kernel::alloc::KBox::from_raw(p) });
            }
        }
    }
}

#[versions(AGX)]
pub(crate) struct Scene {
    pub(crate) user_buffer: GpuArray<u8>,
    pub(crate) buffer: crate::buffer::Buffer::ver,
    pub(crate) tvb_heapmeta: GpuArray<u8>,
    pub(crate) tvb_tilemap: GpuArray<u8>,
    pub(crate) tpc: Arc<GpuArray<u8>>,
    pub(crate) clustering: Option<ClusterBuffers>,
    pub(crate) preempt_buf: GpuArray<u8>,
    #[ver(G >= G14X)]
    pub(crate) control_word: GpuArray<u32>,
    /// The gPM-facing half of the pass page counter.  `raw::Scene.control_word`
    /// (+0x00) and register `0x1c910` both carry this address, which has to sit
    /// in the *file* VM for gPM to reach it; `control_word` above stays in the
    /// kernel VM because the firmware CPU dereferences `+0x08` directly.
    /// LOG.md 32.41.245.
    #[ver(G >= G14X)]
    pub(crate) pm_counter: GpuArray<u32>,
    /// 32.41.664: the Buffer's UMA pool, shared by all its scenes.
    pub(crate) uma: Arc<UmaPool>,
}

#[versions(AGX)]
no_debug!(Scene::ver);

#[versions(AGX)]
impl GpuStruct for Scene::ver {
    type Raw<'a> = raw::Scene::ver<'a>;
}

#[versions(AGX)]
pub(crate) struct InitBuffer {
    pub(crate) scene: Arc<crate::buffer::Scene::ver>,
}

#[versions(AGX)]
no_debug!(InitBuffer::ver);

#[versions(AGX)]
impl workqueue::Command for InitBuffer::ver {}

#[versions(AGX)]
impl GpuStruct for InitBuffer::ver {
    type Raw<'a> = raw::InitBuffer::ver<'a>;
}
