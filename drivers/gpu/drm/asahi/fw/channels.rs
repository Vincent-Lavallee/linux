// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU communication channel firmware structures (ring buffers)

use super::types::*;
use crate::default_zeroed;
use core::sync::atomic::Ordering;
use kernel::static_assert;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct ChannelState<'a> {
        pub(crate) read_ptr: AtomicU32,
        /// 15.5 only, and only for TX channels: the slot's `+0x08` pointer aims
        /// here, and the firmware stores the write pointer it is about to drain
        /// up to into it (0x32d1c) before walking the ring. Nothing in the
        /// driver needs the value -- but it is the one place the firmware
        /// writes when it *looks* at a channel, so it distinguishes "never
        /// scanned this ring" from "scanned it and saw no work".
        pub(crate) read_ack: AtomicU32,
        __pad0: Pad<0x18>,
        pub(crate) write_ptr: AtomicU32,
        __pad1: Pad<0xc>,
        _p: PhantomData<&'a ()>,
    }
    default_zeroed!(<'a>, ChannelState<'a>);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct FwCtlChannelState<'a> {
        pub(crate) read_ptr: AtomicU32,
        __pad0: Pad<0xc>,
        pub(crate) write_ptr: AtomicU32,
        __pad1: Pad<0xc>,
        _p: PhantomData<&'a ()>,
    }
    default_zeroed!(<'a>, FwCtlChannelState<'a>);
}

pub(crate) trait RxChannelState: GpuStruct + Debug + Default
where
    for<'a> <Self as GpuStruct>::Raw<'a>: Default + Zeroable,
{
    const SUB_CHANNELS: usize;

    fn wptr(raw: &Self::Raw<'_>, index: usize) -> u32;
    fn set_rptr(raw: &Self::Raw<'_>, index: usize, rptr: u32);
}

#[derive(Debug, Default)]
pub(crate) struct ChannelState {}

impl GpuStruct for ChannelState {
    type Raw<'a> = raw::ChannelState<'a>;
}

impl RxChannelState for ChannelState {
    const SUB_CHANNELS: usize = 1;

    fn wptr(raw: &Self::Raw<'_>, _index: usize) -> u32 {
        raw.write_ptr.load(Ordering::Acquire)
    }

    fn set_rptr(raw: &Self::Raw<'_>, _index: usize, rptr: u32) {
        raw.read_ptr.store(rptr, Ordering::Release);
    }
}

#[derive(Debug, Default)]
pub(crate) struct FwLogChannelState {}

impl GpuStruct for FwLogChannelState {
    type Raw<'a> = Array<8, raw::ChannelState<'a>>;
}

impl RxChannelState for FwLogChannelState {
    // 15.5 has eight fwlog sub-channels, not six: the producer at 0x30a4c
    // clamps the sub-channel index with `cmp w0, #8`, indexes the state array
    // by `sub * 0x30` (0x30a5c) and the ring by `wptr + (sub << 8)` (0x30a90).
    //
    // Not `#[ver]`-gated because the versions macro cannot gate an associated
    // const, and eight is safe on older firmware anyway: the extra two state
    // slots stay zeroed, so `poll()` sees wptr == rptr and skips them, and the
    // ring and payload buffer are simply allocated two sub-channels larger.
    const SUB_CHANNELS: usize = 8;

    fn wptr(raw: &Self::Raw<'_>, index: usize) -> u32 {
        raw[index].write_ptr.load(Ordering::Acquire)
    }

    fn set_rptr(raw: &Self::Raw<'_>, index: usize, rptr: u32) {
        raw[index].read_ptr.store(rptr, Ordering::Release);
    }
}

/// The event channel's state.
///
/// 15.5's producer indexes `*(slot+0x10)` by `sub * 0x30` (0x30a5c) and the ring
/// by `wptr + (sub << 8)` (0x30a90), i.e. eight `ChannelState`s and eight
/// 256-entry sub-rings. Before 15.5 there is only one.
/// The event channel's state.
///
/// 15.5's producer indexes `*(slot+0x10)` by `sub * 0x30` (0x30a5c) and the ring
/// by `wptr + (sub << 8)` (0x30a90) -- eight `ChannelState`s and eight 256-entry
/// sub-rings. Earlier firmware only ever uses sub-channel 0, which sits at the
/// same address, so this is applied unconditionally rather than versioned: the
/// `#[ver]` attribute is not supported on associated types, and this tree only
/// targets 15.5. It would need gating before it could go anywhere else, exactly
/// like `FwCtlMsg::__pad_15_5`.
#[derive(Debug, Default)]
pub(crate) struct EventChannelState {}

impl GpuStruct for EventChannelState {
    type Raw<'a> = Array<8, raw::ChannelState<'a>>;
}

impl RxChannelState for EventChannelState {
    const SUB_CHANNELS: usize = 8;

    fn wptr(raw: &Self::Raw<'_>, index: usize) -> u32 {
        raw[index].write_ptr.load(Ordering::Acquire)
    }

    fn set_rptr(raw: &Self::Raw<'_>, index: usize, rptr: u32) {
        raw[index].read_ptr.store(rptr, Ordering::Release);
    }
}

#[derive(Debug, Default)]
pub(crate) struct FwCtlChannelState {}

impl GpuStruct for FwCtlChannelState {
    type Raw<'a> = raw::FwCtlChannelState<'a>;
}

pub(crate) trait TxChannelState: GpuStruct + Debug + Default {
    fn rptr(raw: &Self::Raw<'_>) -> u32;
    fn set_wptr(raw: &Self::Raw<'_>, wptr: u32);
    /// The counter 15.5 writes when it *looks* at this channel. Only the
    /// `ChannelState` layout has one; FwCtl keeps its write pointer there
    /// instead, so it reports 0.
    fn read_ack(raw: &Self::Raw<'_>) -> u32;
}

impl TxChannelState for ChannelState {
    fn rptr(raw: &Self::Raw<'_>) -> u32 {
        raw.read_ptr.load(Ordering::Acquire)
    }

    fn set_wptr(raw: &Self::Raw<'_>, wptr: u32) {
        raw.write_ptr.store(wptr, Ordering::Release);
    }

    fn read_ack(raw: &Self::Raw<'_>) -> u32 {
        raw.read_ack.load(Ordering::Relaxed)
    }
}

impl TxChannelState for FwCtlChannelState {
    fn rptr(raw: &Self::Raw<'_>) -> u32 {
        raw.read_ptr.load(Ordering::Acquire)
    }

    fn set_wptr(raw: &Self::Raw<'_>, wptr: u32) {
        raw.write_ptr.store(wptr, Ordering::Release);
    }

    fn read_ack(_raw: &Self::Raw<'_>) -> u32 {
        0
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum PipeType {
    #[default]
    Vertex = 0,
    Fragment = 1,
    Compute = 2,
}

#[versions(AGX)]
#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub(crate) struct RunWorkQueueMsg {
    pub(crate) pipe_type: PipeType,

    // 15.5 shrank this message from 48 bytes to 24 and re-aligned it. The pipe
    // consumer indexes the ring with `umaddl x19, w11, #0x18, x10` (0x324fc)
    // and then touches exactly six fields:
    //
    //   +0x08  ldr  x0, [x19, #8]   the QueueInfo pointer, now *8-aligned*,
    //                               invalidated 0xb0 bytes long (0x32514)
    //   +0x10  ldr  w10, [x19, #16] the pipe type, and it must match the ring
    //                               the message was posted on or the message is
    //                               skipped without being run at all
    //   +0x14  ldrh w28, [x19, #20] wptr, a halfword now: 0x32590 stores it
    //                               into RingState.gpu_rptr and
    //                               QueueInfo.gpu_rptr3 (0x3254c)
    //   +0x16  ldrb w1,  [x19, #22] event_slot, now a byte -- 0x18d94 takes it
    //                               as the index into the firmware's
    //                               128-entry slot table at 0xe4980, bounded
    //                               by `cmp w23, #0x80` at 0x192dc, and
    //                               NUM_EVENTS is 128 (0x325a8)
    //   +0x17  ldrb w8,  [x19, #23] is_new (0x3250c)
    //
    // The two were the other way round at first, which put 0 in the slot index
    // and left table[0].24 null: the first pipe doorbell then took a null
    // dereference at 0x1a3e8. WQ_SIZE is 0x500, so wptr cannot be the byte and
    // the assignment is forced. LOG.md 32.41.
    //
    // `GpuWeakPointer` is packed(1), so pre-15.5 the queue pointer sits at +4
    // right behind pipe_type; putting `wptr` in front of it is what moves it to
    // the +8 that 15.5 reads. Nothing in the consumer reads +0x00 or +0x04, so
    // pipe_type and wptr are kept for continuity rather than because the
    // firmware is known to want them there. LOG.md 32.
    #[ver(V >= V15_5)]
    pub(crate) unk_4: u32,

    pub(crate) work_queue: Option<GpuWeakPointer<super::workqueue::QueueInfo::ver>>,

    #[ver(V < V15_5)]
    pub(crate) wptr: u32,

    // 🔑 There are three pipe consumers, one per ring, and each one accepts
    // only its own pipe type here:
    //
    //   fw:0x32500  ldr w10,[x19,#16] ; cbnz w10, <loop tail>   vertex   -> 0
    //   fw:0x326e0  ldr w10,[x25,#16] ; cmp w10,#1 ; b.ne ...   fragment -> 1
    //   fw:0x328f8  ldr w10,[x26,#16] ; cmp w10,#2 ; b.ne ...   compute  -> 2
    //
    // and the reject path is the loop tail, which advances the channel read
    // pointer and moves on -- the kick is discarded in silence. Reading only
    // the vertex consumer made this look like a "must be zero" field, so the
    // driver hardcoded 0 and every fragment and compute kick was thrown away:
    // the vertex queue ran, the fragment queue sat at `rptr 0 wptr 2` forever,
    // and the render fence never signalled. This is `PipeType`, which pre-15.5
    // sits at +0x00 (where nothing on 15.5 reads it). LOG.md 32.41.103.
    #[ver(V >= V15_5)]
    pub(crate) pipe_kind: u32,
    #[ver(V >= V15_5)]
    pub(crate) wptr: u16,
    #[ver(V >= V15_5)]
    pub(crate) event_slot: u8,
    #[ver(V >= V15_5)]
    pub(crate) is_new: bool,

    #[ver(V < V15_5)]
    pub(crate) event_slot: u32,
    #[ver(V < V15_5)]
    pub(crate) is_new: bool,
    #[ver(V < V15_5 && (V >= V13_2 && G == G14))]
    pub(crate) __pad: Pad<0x2b>,
    #[ver(V < V15_5 && (V < V13_2 || G != G14))]
    pub(crate) __pad: Pad<0x1b>,
}

// The offsets the 15.5 pipe consumer reads, asserted rather than assumed --
// getting these wrong makes the firmware run garbage work instead of failing
// visibly.
#[versions(AGX)]
#[ver(V >= V15_5)]
const _: () = {
    assert!(core::mem::offset_of!(RunWorkQueueMsg::ver, work_queue) == 0x8);
    assert!(core::mem::offset_of!(RunWorkQueueMsg::ver, pipe_kind) == 0x10);
    assert!(core::mem::offset_of!(RunWorkQueueMsg::ver, wptr) == 0x14);
    assert!(core::mem::offset_of!(RunWorkQueueMsg::ver, event_slot) == 0x16);
    assert!(core::mem::offset_of!(RunWorkQueueMsg::ver, is_new) == 0x17);
    assert!(core::mem::size_of::<RunWorkQueueMsg::ver>() == 0x18);
};

#[versions(AGX)]
pub(crate) type PipeMsg = RunWorkQueueMsg::ver;

#[versions(AGX)]
pub(crate) const DEVICECONTROL_SZ: usize = {
    #[ver(V < V15_5 && (V < V13_2 || G != G14))]
    {
        0x2c
    }
    // 15.5 indexes the device control ring with `add x23, x8, x9, lsl #6` at
    // 0x32d6c, i.e. 64-byte entries -- the same size 13.2 already used on G14.
    #[ver(V >= V15_5 || (V >= V13_2 && G == G14))]
    {
        0x3c
    }
};

// TODO: clean up when arbitrary_enum_discriminant is stable
// https://github.com/rust-lang/rust/issues/60553

#[versions(AGX)]
#[derive(Debug, Copy, Clone)]
#[repr(C, u32)]
#[allow(dead_code)]
pub(crate) enum DeviceControlMsg {
    Unk00(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk01(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk02(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk03(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk04(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk05(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk06(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk07(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk08(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk09(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk0a(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk0b(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk0c(Array<DEVICECONTROL_SZ::ver, u8>),
    #[ver(V >= V13_3)]
    Unk0d(Array<DEVICECONTROL_SZ::ver, u8>),
    GrowTVBAck {
        unk_4: u32,
        buffer_slot: u32,
        vm_slot: u32,
        counter: u32,
        subpipe: u32,
        halt_count: U64,
        __pad: Pad<{ DEVICECONTROL_SZ::ver - 0x1c }>,
    },
    RecoverChannel {
        pipe_type: u32,
        work_queue: GpuWeakPointer<super::workqueue::QueueInfo::ver>,
        event_value: u32,
        __pad: Pad<{ DEVICECONTROL_SZ::ver - 0x10 }>,
    },
    IdlePowerOff {
        val: u32,
        __pad: Pad<{ DEVICECONTROL_SZ::ver - 0x4 }>,
    },
    Unk10(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk11(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk12(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk13(Array<DEVICECONTROL_SZ::ver, u8>),
    Unk14(Array<DEVICECONTROL_SZ::ver, u8>), // Init?
    Unk15(Array<DEVICECONTROL_SZ::ver, u8>), // Enable something
    Unk16(Array<DEVICECONTROL_SZ::ver, u8>), // Disable something
    DestroyContext {
        unk_4: u32,
        ctx_23: u8,
        #[ver(V < V13_3)]
        __pad0: Pad<3>,
        unk_c: U32,
        unk_10: U32,
        ctx_0: u8,
        ctx_1: u8,
        ctx_4: u8,
        #[ver(V < V13_3)]
        __pad1: Pad<1>,
        #[ver(V < V13_3)]
        unk_18: u32,
        gpu_context: Option<GpuWeakPointer<super::workqueue::GpuContextData>>,
        #[ver(V < V13_3)]
        __pad2: Pad<{ DEVICECONTROL_SZ::ver - 0x20 }>,
        #[ver(V >= V13_3)]
        __pad2: Pad<{ DEVICECONTROL_SZ::ver - 0x18 }>,
    },
    Unk18(Array<DEVICECONTROL_SZ::ver, u8>),
    Initialize(Pad<DEVICECONTROL_SZ::ver>), // Update RegionC
    // 15.5 UMA "add pages": how the host hands a pool its pages.
    // Kext `AGXUMAFList::updateSubmitInfo` builds it; firmware handler
    // fw:0x32ec0 -> 0x246b8 queues <=0xff8-page chunks, and 0x2495c drains
    // them into SGX 0x14020 = (page_list & 0x7ffffffffc0) | (ctx&0x3f)<<50 |
    // hw_buffer_id<<56 and 0x14018 = pool_id | pages<<52. With `install`,
    // it first installs the pool descriptor (0x24cf0). LOG.md 32.41.545.
    #[ver(V >= V15_5)]
    Unk1b(Array<DEVICECONTROL_SZ::ver, u8>),
    #[ver(V >= V15_5)]
    Unk1c(Array<DEVICECONTROL_SZ::ver, u8>),
    #[ver(V >= V15_5)]
    Unk1d(Array<DEVICECONTROL_SZ::ver, u8>),
    #[ver(V >= V15_5)]
    UmaAddPages {
        hw_buffer_id: u32,
        ctx: u32,
        unk_c: U64,
        pool_record: U64,
        unk_1c: U64,
        page_list: U64,
        pages: u32,
        pool_id: u32,
        install: u32,
        __pad: Pad<{ DEVICECONTROL_SZ::ver - 0x34 }>,
    },
}

#[versions(AGX)]
static_assert!(core::mem::size_of::<DeviceControlMsg::ver>() == 4 + DEVICECONTROL_SZ::ver);

// 0x32d6c indexes this ring with `lsl #6`.
#[versions(AGX)]
#[ver(V >= V15_5)]
static_assert!(core::mem::size_of::<DeviceControlMsg::ver>() == 0x40);

#[versions(AGX)]
default_zeroed!(DeviceControlMsg::ver);

#[derive(Copy, Clone, Default, Debug)]
#[repr(C)]
#[allow(dead_code)]
pub(crate) struct FwCtlMsg {
    pub(crate) addr: U64,
    pub(crate) unk_8: u32,
    pub(crate) slot: u32,
    pub(crate) page_count: u16,
    pub(crate) unk_12: u16,
    // 15.5 strides this ring by 24 bytes, not 20: the consumer at 0x2bb84 does
    // `umaddl x28, w22, w24, x21` with w24 = 0x18. The ring is 0x100 entries
    // either way (the firmware masks the index with 0xff), so the size has to
    // match or the driver and firmware disagree about where entry N starts.
    // LOG.md 30.9.
    //
    // Applied unconditionally rather than versioned: making FwCtlMsg versioned
    // cascades through TxChannel's generics, and this tree only targets 15.5.
    // It would need gating before it could go anywhere else.
    pub(crate) __pad_15_5: u32,
}

// The stride the firmware indexes this ring by.
const _: () = assert!(core::mem::size_of::<FwCtlMsg>() == 0x18);

// 15.5 indexes the event ring with `lsl #6` at 0x30a98, i.e. 64-byte entries,
// where 13.5 used 56. Unconditional for the same reason as EventChannelState.
pub(crate) const EVENT_SZ: usize = 0x3c;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C, u32)]
#[allow(dead_code)]
pub(crate) enum ChannelErrorType {
    MemoryError,
    DMKill,
    Aborted,
    Unk3,
    Unknown(u32),
}

#[derive(Debug, Copy, Clone)]
#[repr(C, u32)]
#[allow(dead_code)]
pub(crate) enum EventMsg {
    Fault,
    Flag {
        firing: [u32; 4],
        unk_14: u16,
    },
    Unk2(Array<EVENT_SZ, u8>),
    Unk3(Array<EVENT_SZ, u8>),
    // ⚠️ 15.5 does not lay this out as three u32s. The firmware builds the
    // message at `fw:0x5ba8` as `{u32 tag = 4, u64 block, i32 event_slot}` and
    // leaves the rest of the 0x40 bytes alone, confirmed by the raw dump:
    //
    //     Event raw: tag 4 | 0x00014000 0xfffffe0a 0xffffffff 0 0 0 0 0
    //
    // so `counter` and `unk_8` are the low and high halves of one pointer
    // (0xfffffe0a00014000 here -- the base of the register/state block that
    // `fw:0x4e44` memsets), and there is no counter at all. `event_slot` is
    // right. Left as-is because nothing reads these but the log line, and
    // splitting the variant by version would drag `RawEventMsg` and the whole
    // `EventChannel` into `#[versions]` for a debug print. LOG.md 32.41.114.
    Timeout {
        counter: u32,
        unk_8: u32,
        event_slot: i32,
    },
    Unk5(Array<EVENT_SZ, u8>),
    Unk6(Array<EVENT_SZ, u8>),
    GrowTVB {
        vm_slot: u32,
        buffer_slot: u32,
        counter: u32,
    },
    ChannelError {
        error_type: u32,
        pipe_type: u32,
        event_slot: u32,
        event_value: u32,
    },
    // Max discriminant: 0x8
}

static_assert!(core::mem::size_of::<EventMsg>() == 4 + EVENT_SZ);

pub(crate) const EVENT_MAX: u32 = 0x8;

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) union RawEventMsg {
    pub(crate) raw: (u32, Array<EVENT_SZ, u8>),
    pub(crate) msg: EventMsg,
}

default_zeroed!(RawEventMsg);

#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub(crate) struct RawFwLogMsg {
    pub(crate) msg_type: u32,
    __pad0: u32,
    pub(crate) msg_index: U64,
    // 15.5 indexes the fwlog ring with `add x8, x8, x24, lsl #6` at 0x10580,
    // i.e. 64-byte entries where 13.5 used 56. With the old stride every entry
    // after the first is read misaligned, which is why the only two messages
    // this driver has ever decoded came out as `msg_type: 26` and a nonsense
    // msg_index. Same fix as EVENT_SZ. LOG.md 32.24.
    __pad1: Pad<0x30>,
}

const _: () = assert!(core::mem::size_of::<RawFwLogMsg>() == 0x40);

#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub(crate) struct RawFwLogPayloadMsg {
    pub(crate) msg_type: u32,
    pub(crate) seq_no: u32,
    pub(crate) timestamp: U64,
    pub(crate) msg: Array<0xc8, u8>,
}

/// A firmware trace record.
///
/// 🔑 15.5 changed both the stride and the field order.  The emitter at
/// fw:0x201f8 indexes the ring with
///
///     fw:0x203a0  add x19, x8, x22, lsl #6
///
/// so a record is **0x40 bytes**, not the 0x38 the pre-15.5 layout implies, and
/// the stores that follow put
///
///     +0x00 w=5 (constant)      +0x2c  id | (cpu << 24)
///     +0x04 timestamp           +0x30  the w1 argument
///     +0x0c x2   +0x14 x3   +0x1c x4   +0x24 x5
///
/// with every 64-bit field unaligned.  `id` is the w0 literal at the call site,
/// which maps straight back to an address -- see the generated table in
/// `m3-gpu-display/ktrace-ids.txt`.  Its high byte is the category bit in the
/// mask at fw:0x1a383c that `Globals::ktrace_enable` feeds.
///
/// With the old 0x38 stride the driver read each record from a base 8 bytes
/// further off than the last (so everything but the first was garbage) *and*
/// the firmware wrote 0x1000 bytes past the end of the ring allocation.  That
/// never mattered while the mask was zero and nothing was ever emitted.
/// LOG.md 32.41.112.
#[versions(AGX)]
#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub(crate) struct RawKTraceMsg {
    #[ver(V < V15_5)]
    pub(crate) msg_type: u32,
    #[ver(V < V15_5)]
    pub(crate) timestamp: U64,
    #[ver(V < V15_5)]
    pub(crate) args: Array<4, U64>,
    #[ver(V < V15_5)]
    pub(crate) code: u8,
    #[ver(V < V15_5)]
    pub(crate) channel: u8,
    #[ver(V < V15_5)]
    __pad: Pad<1>,
    #[ver(V < V15_5)]
    pub(crate) thread: u8,
    #[ver(V < V15_5)]
    pub(crate) unk_flag: U64,

    #[ver(V >= V15_5)]
    pub(crate) kind: u32,
    #[ver(V >= V15_5)]
    pub(crate) timestamp: U64,
    #[ver(V >= V15_5)]
    pub(crate) args: Array<4, U64>,
    #[ver(V >= V15_5)]
    pub(crate) id_cpu: U32,
    #[ver(V >= V15_5)]
    pub(crate) flags: U32,
    #[ver(V >= V15_5)]
    __pad2: Pad<0xc>,
}

#[versions(AGX)]
#[ver(V >= V15_5)]
const _KTRACE_LAYOUT: () = {
    assert!(core::mem::size_of::<RawKTraceMsg::ver>() == 0x40);
    assert!(core::mem::offset_of!(RawKTraceMsg::ver, timestamp) == 0x04);
    assert!(core::mem::offset_of!(RawKTraceMsg::ver, args) == 0x0c);
    assert!(core::mem::offset_of!(RawKTraceMsg::ver, id_cpu) == 0x2c);
    assert!(core::mem::offset_of!(RawKTraceMsg::ver, flags) == 0x30);
};

#[versions(AGX)]
pub(crate) const STATS_SZ: usize = {
    #[ver(V < V13_0B4)]
    {
        0x2c
    }
    #[ver(V >= V13_0B4)]
    {
        0x3c
    }
};

#[versions(AGX)]
#[derive(Debug, Copy, Clone)]
#[repr(C, u32)]
#[allow(dead_code)]
pub(crate) enum StatsMsg {
    Power {
        // 0x00
        __pad: Pad<0x18>,
        power: U64,
    },
    Unk1(Array<{ STATS_SZ::ver }, u8>),
    PowerOn {
        // 0x02
        off_time: U64,
    },
    PowerOff {
        // 0x03
        on_time: U64,
    },
    Utilization {
        // 0x04
        timestamp: U64,
        util1: u32,
        util2: u32,
        util3: u32,
        util4: u32,
    },
    Unk5(Array<{ STATS_SZ::ver }, u8>),
    Unk6(Array<{ STATS_SZ::ver }, u8>),
    Unk7(Array<{ STATS_SZ::ver }, u8>),
    Unk8(Array<{ STATS_SZ::ver }, u8>),
    AvgPower {
        // 0x09
        active_cs: U64,
        unk2: u32,
        unk3: u32,
        unk4: u32,
        avg_power: u32,
    },
    Temperature {
        // 0x0a
        __pad: Pad<0x8>,
        raw_value: u32,
        scale: u32,
        tmin: u32,
        tmax: u32,
    },
    PowerState {
        // 0x0b
        timestamp: U64,
        last_busy_ts: U64,
        active: u32,
        poweroff: u32,
        unk1: u32,
        pstate: u32,
        unk2: u32,
        unk3: u32,
    },
    FwBusy {
        // 0x0c
        timestamp: U64,
        busy: u32,
    },
    PState {
        // 0x0d
        __pad: Pad<0x8>,
        ps_min: u32,
        unk1: u32,
        ps_max: u32,
        unk2: u32,
    },
    TempSensor {
        // 0x0e
        __pad: Pad<0x4>,
        sensor_id: u32,
        raw_value: u32,
        scale: u32,
        tmin: u32,
        tmax: u32,
    }, // Max discriminant: 0xe
}

#[versions(AGX)]
static_assert!(core::mem::size_of::<StatsMsg::ver>() == 4 + STATS_SZ::ver);

#[versions(AGX)]
pub(crate) const STATS_MAX: u32 = 0xe;

#[versions(AGX)]
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) union RawStatsMsg {
    pub(crate) raw: (u32, Array<{ STATS_SZ::ver }, u8>),
    pub(crate) msg: StatsMsg::ver,
}

#[versions(AGX)]
default_zeroed!(RawStatsMsg::ver);
