// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU work queue firmware structes

use super::event;
use super::types::*;
use crate::event::EventValue;
use crate::{
    default_zeroed,
    trivial_gpustruct, //
};
use kernel::sync::Arc;

#[derive(Debug)]
#[repr(u32)]
pub(crate) enum CommandType {
    RunVertex = 0,
    RunFragment = 1,
    #[allow(dead_code)]
    RunBlitter = 2,
    RunCompute = 3,
    Barrier = 4,
    InitBuffer = 6,
}

pub(crate) trait Command: GpuStruct + Send + Sync {}

pub(crate) mod raw {
    use super::*;

    // 🔑 15.5 moves the awaited *value* from +0x0c to +0x14 and puts a second,
    // nullable stamp pointer where it used to be. Read off the barrier
    // evaluation at `fw:0x1a764`, which runs only for `tag == 4`:
    //
    //     ldr  w8, [x23, #48]        ; +0x30, gate: zero -> not evaluated
    //     ldr  w8, [x23, #32]        ; +0x20, == 0x80 selects a 64-bit compare
    //     ldr  w9, [x23, #20]        ; +0x14  <- the awaited value
    //     ldur x10, [x23, #4]        ; +0x04  <- the awaited stamp (unaligned)
    //     ldr  w10, [x10]
    //     sub  w9, w9, w10 ; cmp #1 ; b.lt   ; satisfied when current >= value
    //     ldur x9, [x23, #12]        ; +0x0c  <- a second pointer, null-checked
    //     ldr  x10, [x23, #24]       ; +0x18  <- 64-bit value for the 0x80 path
    //
    // and the dispatcher at `fw:0x19308`, which flushes **0x38** bytes and reads
    // `[x26, #36]` (+0x24) taking bits 14:8 as a slot.
    //
    // With `wait_value` at +0x0c the firmware read `wait_value | wait_slot << 32`
    // as its second pointer and, worse, took `stamp_self` (0x1000100) as the
    // value to wait for -- against a vertex stamp that stops at 0x100. The
    // fragment barrier could never be satisfied. LOG.md 32.41.96.
    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Barrier {
        pub(crate) tag: CommandType,
        pub(crate) wait_stamp: GpuWeakPointer<FwStamp>,

        #[ver(V < V15_5)]
        pub(crate) wait_value: EventValue,
        #[ver(V < V15_5)]
        pub(crate) wait_slot: u32,
        #[ver(V < V15_5)]
        pub(crate) stamp_self: EventValue,
        #[ver(V < V15_5)]
        pub(crate) uuid: u32,
        #[ver(V < V15_5)]
        pub(crate) external_barrier: u32,
        // G14X addition
        #[ver(V < V15_5)]
        pub(crate) internal_barrier_type: u32,
        #[ver(V < V15_5)]
        pub(crate) padding: Pad<0x1c>,

        // A second stamp pointer, explicitly null-checked at `fw:0x1a794`.
        // Left null until something says what belongs here.
        #[ver(V >= V15_5)]
        pub(crate) wait_stamp2: U64,
        #[ver(V >= V15_5)]
        pub(crate) wait_value: EventValue,
        // The 64-bit awaited value, only read when `barrier_type == 0x80`.
        #[ver(V >= V15_5)]
        pub(crate) wait_value64: U64,
        // `0x80` switches the compare to the 64-bit pair above.
        #[ver(V >= V15_5)]
        pub(crate) barrier_type: u32,
        // Bits 14:8 are the slot; the rest is unidentified.
        #[ver(V >= V15_5)]
        pub(crate) flags_slot: u32,
        #[ver(V >= V15_5)]
        pub(crate) stamp_self: EventValue,
        #[ver(V >= V15_5)]
        pub(crate) uuid: u32,
        // Gate at +0x30: `fw:0x1a768` skips the whole wait when this is zero.
        // Left zero -- that is what the driver has always emitted, so it is
        // the one field here that cannot be a regression.
        #[ver(V >= V15_5)]
        pub(crate) armed: u32,
        #[ver(V >= V15_5)]
        pub(crate) unk_34: u32,
        #[ver(V >= V15_5)]
        pub(crate) padding: Pad<0x8>,
    }

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct GpuContextData {
        pub(crate) unk_0: u8,
        pub(crate) unk_1: u8,
        unk_2: Array<0x2, u8>,
        pub(crate) unk_4: u8,
        pub(crate) unk_5: u8,
        unk_6: Array<0x18, u8>,
        pub(crate) unk_1e: u8,
        pub(crate) unk_1f: u8,
        unk_20: Array<0x3, u8>,
        pub(crate) unk_23: u8,
        unk_24: Array<0xf, u8>,
        /// .667zzz41: the context's hardware compute id. 0xff = unassigned: the firmware's CDM
        /// dispatch (fw:0x147e4) then takes a free bit of its id bitmap (fw 0x121910) and stores
        /// it here. Left 0, every context claimed id 0 and two concurrent compute contexts
        /// deadlocked (VK tess CTS -j8, two vkbig processes).
        pub(crate) unk_33: u8,
        unk_34: Array<0xc, u8>,
    }

    impl Default for GpuContextData {
        fn default() -> Self {
            Self {
                unk_0: 0xff,
                unk_1: 0xff,
                unk_2: Default::default(),
                // +3 (group) and +4 (partition weight) are overwritten by the firmware from
                // QueueInfo+0x30 / weight_table[QueueInfo+0x48] (fw:0x115e8). LOG.md 32.41.667zzz41
                unk_4: 0,
                unk_5: 1,
                unk_6: Default::default(),
                unk_1e: 0xff,
                unk_1f: 0,
                unk_20: Default::default(),
                unk_23: 2,
                unk_24: Default::default(),
                unk_33: if *crate::module_parameters::ctx_hwid_ff.value() != 0 { 0xff } else { 0 },
                unk_34: Default::default(),
            }
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RingState {
        pub(crate) gpu_doneptr: AtomicU32,
        __pad0: Pad<0xc>,
        pub(crate) unk_10: AtomicU32,
        __pad1: Pad<0xc>,
        pub(crate) unk_20: AtomicU32,
        __pad2: Pad<0xc>,
        pub(crate) gpu_rptr: AtomicU32,
        __pad3: Pad<0xc>,
        pub(crate) cpu_wptr: AtomicU32,
        __pad4: Pad<0xc>,
        pub(crate) rb_size: u32,
        __pad5: Pad<0xc>,
        // This isn't part of the structure, but it's here as a
        // debugging hack so we can inspect what ring position
        // the driver considered complete and freeable.
        pub(crate) cpu_freeptr: AtomicU32,
        __pad6: Pad<0xc>,
    }
    default_zeroed!(RingState);

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct Priority(
        pub(crate) u32,
        pub(crate) u32,
        pub(crate) U64,
        pub(crate) u32,
        pub(crate) u32,
        pub(crate) u32,
    );

    pub(crate) const PRIORITY: [Priority; 4] = [
        Priority(0, 0, U64(0xffff_ffff_ffff_0000), 1, 0, 1),
        Priority(1, 1, U64(0xffff_ffff_0000_0000), 0, 0, 0),
        Priority(2, 2, U64(0xffff_0000_0000_0000), 0, 0, 2),
        Priority(3, 3, U64(0x0000_0000_0000_0000), 0, 0, 3),
    ];

    impl Default for Priority {
        fn default() -> Priority {
            PRIORITY[2]
        }
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct QueueInfo<'a> {
        pub(crate) state: GpuPointer<'a, super::RingState>,
        pub(crate) ring: GpuPointer<'a, &'a [u64]>,
        pub(crate) notifier_list: GpuPointer<'a, event::NotifierList>,
        pub(crate) gpu_buf: GpuPointer<'a, &'a [u8]>,
        pub(crate) gpu_rptr1: AtomicU32,
        pub(crate) gpu_rptr2: AtomicU32,
        pub(crate) gpu_rptr3: AtomicU32,
        pub(crate) event_id: AtomicI32,
        pub(crate) priority: Priority,
        pub(crate) unk_4c: i32,
        pub(crate) uuid: u32,
        pub(crate) unk_54: i32,
        pub(crate) unk_58: U64,
        pub(crate) busy: AtomicU32,
        pub(crate) __pad: Pad<0x20>,
        #[ver(V >= V13_2 && G < G14X)]
        pub(crate) unk_84_0: u32,
        pub(crate) unk_84_state: AtomicU32,
        pub(crate) error_count: AtomicU32,
        pub(crate) unk_8c: u32,
        pub(crate) unk_90: u32,
        pub(crate) unk_94: u32,
        pub(crate) pending: AtomicU32,
        pub(crate) unk_9c: u32,

        #[ver(V < V15_5)]
        pub(crate) gpu_context: GpuPointer<'a, super::GpuContextData>,
        #[ver(V < V15_5)]
        pub(crate) unk_a8: U64,

        // 15.5 moved the context pointer four bytes up, into an unaligned slot
        // -- the channel processor reads it with `ldur x0, [x27, #164]` at
        // 0x3255c and again at 0x32524, where it flushes 0x38 bytes through it.
        // Left at 0xa0 the firmware picks up the pointer's high half and takes
        // a data abort on 0xfffffe08 the first time a pipe doorbell is
        // dispatched. The struct is still 0xb0 bytes: 0x32514 flushes exactly
        // that much. LOG.md 32.41.
        #[ver(V >= V15_5)]
        pub(crate) unk_a0: u32,
        #[ver(V >= V15_5)]
        pub(crate) gpu_context_15_5: Array<2, u32>,
        #[ver(V >= V15_5)]
        pub(crate) unk_ac: u32,

        #[ver(V >= V13_2 && G < G14X)]
        pub(crate) unk_b0: u32,
    }
}

// `trivial_gpustruct!` cannot be versioned (it would define `Barrier` once per
// variant), so spell out the versioned equivalent.
#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Barrier {}

#[versions(AGX)]
impl GpuStruct for Barrier::ver {
    type Raw<'a> = raw::Barrier::ver;
}

#[versions(AGX)]
default_zeroed!(Barrier::ver);
trivial_gpustruct!(RingState);

#[versions(AGX)]
impl Command for Barrier::ver {}

pub(crate) struct GpuContextData {
    pub(crate) _buffer: Arc<dyn core::any::Any + Send + Sync>,
}
impl GpuStruct for GpuContextData {
    type Raw<'a> = raw::GpuContextData;
}

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct QueueInfo {
    pub(crate) state: GpuObject<RingState>,
    pub(crate) ring: GpuArray<u64>,
    pub(crate) gpu_buf: GpuArray<u8>,
    pub(crate) notifier_list: Arc<GpuObject<event::NotifierList>>,
    pub(crate) gpu_context: Arc<crate::workqueue::GpuContext>,
}

#[versions(AGX)]
impl GpuStruct for QueueInfo::ver {
    type Raw<'a> = raw::QueueInfo::ver<'a>;
}

