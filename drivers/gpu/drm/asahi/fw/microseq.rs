// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU firmware microsequence operations

use super::types::*;
use super::{
    buffer,
    compute,
    fragment,
    initdata,
    job,
    vertex,
    workqueue, //
};
use crate::default_zeroed;

pub(crate) trait Operation {}

#[derive(Debug, Copy, Clone)]
#[repr(u32)]
enum OpCode {
    WaitForIdle = 0x01,
    WaitForIdle2 = 0x02,
    RetireStamp = 0x18,
    #[allow(dead_code)]
    Timestamp = 0x19,
    StartVertex = 0x22,
    FinalizeVertex = 0x23,
    StartFragment = 0x24,
    FinalizeFragment = 0x25,
    StartCompute = 0x29,
    FinalizeCompute = 0x2a,
}

/// 15.5 renumbered the microsequence opcodes into a dense 1..0xe.
///
/// The firmware's interpreter dispatches on `tag & 0x3f` through a 14-entry
/// jump table and panics "Unknown SKU command type %u" on anything else, which
/// is exactly what the driver's 0x22 produced. `StartVertex = 5` is confirmed:
/// that handler's reads land on `buffer` 0x1c, `stats` 0x2c, `work_queue` 0x34,
/// `vm_slot` 0x3c and `event_generation` 0x44 -- five exact hits against the
/// real offsets, with no shift.
///
/// The whole map is now read directly off the interpreter's own jump table.
/// `0x14bf8` does `ldr w23, [x19]; and w8, w23, #0x3f; sub w16, w8, #1; cmp
/// w16, #0xd; b.hi <invalid>` and indexes a table of signed word offsets at
/// `0x187e8` relative to `0x14c20`, giving opcodes 1..0xe:
///
/// ```text
///  1 -> 0x14c2c    2 -> 0x1815c    3 -> 0x151b4    4 -> 0x1568c
///  5 -> 0x14f48    6 -> 0x15840    7 -> 0x15488    8 -> 0x15664
///  9 -> 0x15298    a -> 0x157a8    b -> 0x14d68    c -> 0x14c58
///  d -> 0x1831c    e -> 0x18188
/// ```
///
/// So it is a plain ascending compaction after all, and every Start/Finalize
/// pair is confirmed: vertex (5, 6), fragment (7, 8), compute (9, a).
/// LOG.md 32.41.19, 32.41.26.
#[derive(Debug, Copy, Clone)]
#[repr(u32)]
#[allow(dead_code)]
enum OpCode15_5 {
    WaitForIdle = 0x1,
    WaitForIdle2 = 0x2,
    // Handler 3's reads -- pointers at 0xc, 0x24, 0x2c, 0x34 -- match the
    // `Timestamp` struct (`ts_pointers`, `user_ts_pointers`, `unk_ts`, `uuid`),
    // not a bare `RetireStamp`, so these two are the other way round from the
    // plain ascending compaction. LOG.md 32.41.21.
    Timestamp = 0x3,
    RetireStamp = 0xd,
    StartVertex = 0x5,
    FinalizeVertex = 0x6,
    StartFragment = 0x7,
    FinalizeFragment = 0x8,
    StartCompute = 0x9,
    FinalizeCompute = 0xa,
}

#[derive(Debug, Copy, Clone)]
#[repr(u32)]
pub(crate) enum Pipe {
    Vertex = 1 << 0,
    Fragment = 1 << 8,
    Compute = 1 << 15,
}

pub(crate) const MAX_ATTACHMENTS: usize = 16;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct Attachment {
    pub(crate) address: U64,
    pub(crate) size: u32,
    pub(crate) unk_c: u16,
    pub(crate) unk_e: u16,
}
default_zeroed!(Attachment);

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct Attachments {
    pub(crate) list: Array<MAX_ATTACHMENTS, Attachment>,
    pub(crate) count: u32,
}

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub(crate) struct OpHeader(u32);

impl OpHeader {
    const fn new(opcode: OpCode) -> OpHeader {
        OpHeader(opcode as u32)
    }
    const fn with_args(opcode: OpCode, args: u32) -> OpHeader {
        OpHeader(opcode as u32 | args)
    }
}

macro_rules! simple_op {
    ($name:ident) => {
        #[allow(dead_code)]
        #[derive(Debug, Copy, Clone)]
        pub(crate) struct $name(OpHeader);

        impl $name {
            pub(crate) const HEADER: $name = $name(OpHeader::new(OpCode::$name));
            #[allow(dead_code)]
            pub(crate) const HEADER_V15_5: $name =
                $name(OpHeader(OpCode15_5::$name as u32));
        }
    };
}

pub(crate) mod op {
    use super::*;

    simple_op!(StartVertex);
    simple_op!(FinalizeVertex);
    simple_op!(StartFragment);
    simple_op!(FinalizeFragment);
    simple_op!(StartCompute);
    simple_op!(FinalizeCompute);
    simple_op!(WaitForIdle2);

    impl WaitForIdle2 {
        // The sequence terminator on 15.5.  The kext seals every microsequence
        // in `0x914525c`, which advances the write cursor by 4 and stores a
        // bare `0x40000002` -- opcode 2 with bit 30 set.  Bit 30 is the same
        // "this is the last opcode" flag `RetireStamp` carries on earlier
        // firmware; without it handler 2 runs, falls through the interpreter's
        // exit block and keeps walking past the end of the buffer until it
        // panics with "Unknown SKU command type 0".  LOG.md 32.41.25.
        pub(crate) const HEADER_END_V15_5: WaitForIdle2 =
            WaitForIdle2(OpHeader(OpCode15_5::WaitForIdle2 as u32 | 0x40000000));
    }

    #[allow(dead_code)]
    #[derive(Debug, Copy, Clone)]
    pub(crate) struct StartVertexPre(OpHeader);
    impl StartVertexPre {
        pub(crate) const HEADER: StartVertexPre = StartVertexPre(OpHeader(0x4));
    }

    #[allow(dead_code)]
    #[derive(Debug, Copy, Clone)]
    pub(crate) struct RetireStamp(OpHeader);
    impl RetireStamp {
        pub(crate) const HEADER: RetireStamp =
            RetireStamp(OpHeader::with_args(OpCode::RetireStamp, 0x40000000));
        #[allow(dead_code)]
        pub(crate) const HEADER_V15_5: RetireStamp = RetireStamp(OpHeader(
            OpCode15_5::RetireStamp as u32 | 0x40000000,
        ));
    }

    #[allow(dead_code)]
    #[derive(Debug, Copy, Clone)]
    pub(crate) struct WaitForIdle(OpHeader);
    impl WaitForIdle {
        pub(crate) const fn new(pipe: Pipe) -> WaitForIdle {
            WaitForIdle(OpHeader::with_args(OpCode::WaitForIdle, (pipe as u32) << 8))
        }

        /// 15.5's TA-finish wait: a bare opcode 1 with **no pipe field**.
        ///
        /// Taken from the kext, not guessed. `AGXTAChannelG15::encodeTAFinishWFI`
        /// (`fffffe000911a040`, 8 instructions) advances the encoder's write
        /// pointer by 4 and stores the literal `1`:
        ///
        /// ```text
        ///   ldr x8, [x0, #184]   ; encoder
        ///   ldr x9, [x8, #104]   ; write pointer
        ///   add x10, x9, #0x4
        ///   str x10, [x8, #104]
        ///   mov w8, #0x1
        ///   str w8, [x9]         ; the whole op: 0x00000001
        /// ```
        ///
        /// Note it is **not** `new(Pipe::Vertex)`, which would be `0x00000101`
        /// -- 15.5 carries no pipe bits here. LOG.md 32.41.423.
        pub(crate) const HEADER_V15_5: WaitForIdle =
            WaitForIdle(OpHeader(OpCode15_5::WaitForIdle as u32));
    }

    #[allow(dead_code)]
    #[derive(Debug, Copy, Clone)]
    pub(crate) struct Timestamp(OpHeader);
    impl Timestamp {
        #[allow(dead_code)]
        pub(crate) const fn new(flag: bool) -> Timestamp {
            Timestamp(OpHeader::with_args(OpCode::Timestamp, (flag as u32) << 31))
        }

        /// 15.5's timestamp opcode is **3**, not `OpCode::Timestamp` (0x19).
        ///
        /// `Timestamp` was the only op left without a `_V15_5` form -- every
        /// other one (`StartVertex`, `FinalizeVertex`, `StartFragment`,
        /// `FinalizeFragment`, `RetireStamp`, `WaitForIdle`, `StartCompute`,
        /// `FinalizeCompute`) has one and uses it. It was missed in the 15.5
        /// port because the only call sites are gated on userspace asking for
        /// timestamps, which Mesa never does, so nothing ever exercised it.
        ///
        /// Caught on hardware by 32.41.501: the op list came out at exactly
        /// Apple's offsets and sizes but with `hdr 0x80000019` where Apple's
        /// captured bytes read `0x80000003`.
        #[allow(dead_code)]
        pub(crate) const fn new_v15_5(flag: bool) -> Timestamp {
            Timestamp(OpHeader(
                OpCode15_5::Timestamp as u32 | ((flag as u32) << 31),
            ))
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub(crate) struct WaitForIdle {
    pub(crate) header: op::WaitForIdle,
}

impl Operation for WaitForIdle {}

#[derive(Debug)]
#[repr(C)]
pub(crate) struct WaitForIdle2 {
    pub(crate) header: op::WaitForIdle2,
}

impl Operation for WaitForIdle2 {}

/// Opcode 4 on 15.5, emitted immediately before `StartVertex`.
///
/// macOS builds it at `0x911a58c`: advance 0x14, header `4` (optionally with
/// bit 31 set), then two u32s and a pointer. The firmware's handler 4 matches
/// exactly -- `tbnz w23, #31`, then `ldp w20, w21, [op+4]` and
/// `ldr x22, [op+0xc]`. LOG.md 32.41.23.
#[versions(AGX)]
#[derive(Debug)]
#[repr(C)]
pub(crate) struct StartVertexPre {
    pub(crate) header: op::StartVertexPre,
    pub(crate) unk_4: u32,
    pub(crate) unk_8: u32,
    pub(crate) unk_c: GpuWeakPointer<U64>,
}

#[versions(AGX)]
impl Operation for StartVertexPre::ver {}

#[versions(AGX)]
#[derive(Debug)]
#[repr(C)]
pub(crate) struct RetireStamp {
    pub(crate) header: op::RetireStamp,

    // 15.5 turned this from a bare header into a timestamp op. Its handler
    // reads pointers at 0xc, 0x24, 0x2c and 0x34 and hands them to 0x26c10,
    // which stores 64-bit GPU timestamps through them (null-checked there, but
    // the handler then does `ldr x10, [0x2c]` with no check and accumulates
    // elapsed time into it). LOG.md 32.41.21.
    #[ver(V >= V15_5)]
    pub(crate) unk_4: U64,
    #[ver(V >= V15_5)]
    pub(crate) ts_c: GpuWeakPointer<U64>,
    #[ver(V >= V15_5)]
    pub(crate) unk_14: Array<0x10, u8>,
    #[ver(V >= V15_5)]
    pub(crate) ts_24: GpuWeakPointer<U64>,
    #[ver(V >= V15_5)]
    pub(crate) ts_2c: GpuWeakPointer<U64>,
    #[ver(V >= V15_5)]
    pub(crate) ts_34: GpuWeakPointer<U64>,
}

#[versions(AGX)]
impl Operation for RetireStamp::ver {}

#[versions(AGX)]
#[derive(Debug)]
#[repr(C)]
pub(crate) struct Timestamp<'a> {
    pub(crate) header: op::Timestamp,
    pub(crate) command_time: GpuWeakPointer<U64>,
    pub(crate) ts_pointers: GpuWeakPointer<job::raw::TimestampPointers<'a>>,
    // Unused?
    pub(crate) update_ts: GpuWeakPointer<Option<GpuPointer<'a, AtomicU64>>>,
    pub(crate) work_queue: GpuWeakPointer<workqueue::QueueInfo::ver>,
    /// Apple emits **NULL** here (capture4 `raw-ta-microseq-draw.txt`, +0x24 = 0).
    /// `Option<GpuWeakPointer<_>>` uses the `NonZeroU64` null niche, so this is
    /// layout-identical to the bare pointer -- the `size_of`/`offset_of`
    /// assertions below are what prove that, and they are checked at build
    /// time. LOG.md 32.41.502.
    pub(crate) user_ts_pointers: Option<GpuWeakPointer<job::raw::TimestampPointers<'a>>>,

    #[ver(V >= V13_0B4)]
    pub(crate) unk_ts: GpuWeakPointer<U64>,

    pub(crate) uuid: u32,
    pub(crate) unk_30_padding: u32,
}

#[versions(AGX)]
impl<'a> Operation for Timestamp::ver<'a> {}

#[versions(AGX)]
#[derive(Debug)]
#[repr(C)]
pub(crate) struct StartVertex<'a> {
    pub(crate) header: op::StartVertex,
    pub(crate) tiling_params: Option<GpuWeakPointer<vertex::raw::TilingParameters>>,
    pub(crate) job_params1: Option<GpuWeakPointer<vertex::raw::JobParameters1::ver<'a>>>,
    #[ver(G >= G14X)]
    pub(crate) registers: GpuWeakPointer<job::raw::RegisterArray>,
    pub(crate) buffer: GpuWeakPointer<buffer::Info::ver>,
    pub(crate) scene: GpuWeakPointer<buffer::Scene::ver>,
    pub(crate) stats: GpuWeakPointer<initdata::raw::GpuStatsVtx>,
    pub(crate) work_queue: GpuWeakPointer<workqueue::QueueInfo::ver>,

    // 15.5 replaces `vm_slot` + `unk_38` with a pointer: the handler does
    // `ldr x9, [opcode+0x3c]; str w27, [x9]` and later `ldr w9, [x9]; cbnz`.
    // Caught as a fault at 0x14fdc with FAR = 0x100000001, which is exactly
    // (unk_38 << 32) | vm_slot with the driver's 1 and 1. The two u32s and the
    // pointer are both 8 bytes, so nothing after this moves -- and nothing
    // does: `event_generation` 0x44, `unk_44` 0x4c and `event_seq` 0x50 all
    // still land where the firmware reads them. LOG.md 32.41.19.
    #[ver(V < V15_5)]
    pub(crate) vm_slot: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_38: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_3c_ptr: GpuWeakPointer<u32>,

    pub(crate) event_generation: u32,
    pub(crate) buffer_slot: u32,
    pub(crate) unk_44: u32,
    pub(crate) event_seq: U64,
    pub(crate) unk_50: u32,
    // 32.41.598: Apple's captured StartVertex (capture4 raw-ta-microseq-draw,
    // finishEncode #2) has two zero u32s here; its pointers sit at 0x64/0x6c
    // and the uuid at 0x7c (the TA-kick trace fw:0x162c0 logs op+0x7c). We had
    // everything 8 bytes early: the firmware read our pointer's low bits at
    // 0x5c as a count (<<12) and u32 garbage at 0x6c as a pointer.
    #[ver(V >= V15_5)]
    pub(crate) unk_5c: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_60: u32,
    pub(crate) unk_pointer: GpuWeakPointer<u32>,
    pub(crate) unk_job_buf: GpuWeakPointer<U64>,
    pub(crate) unk_64: u32,
    pub(crate) unk_68: u32,
    pub(crate) uuid: u32,

    // 15.5 inserts exactly 0x10 bytes before the attachment array -- a u32 at
    // 0x78, an 8-byte field at 0x7c and a u32 at 0x84 -- and appends 0x38 at
    // the tail.  Read off the kext's own stores into the memcpy source at
    // 0x911a064, and confirmed against the firmware's attachment walker at
    // 0x150f4, which does `add x26, x28, #0x92` (x28 = opcode + 4, so the
    // array is at opcode+0x88) and strides 0x10 with the count at
    // opcode+0x188 (`ldr w8, [x28, #388]`).  LOG.md 32.41.25.
    #[ver(V >= V15_5)]
    pub(crate) unk_80: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_84: u32,

    pub(crate) attachments: Attachments,
    pub(crate) padding: u32,

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) counter: U64,

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) notifier_buf: GpuWeakPointer<Array<0x8, u8>>,

    #[ver(V < V15_5)]
    pub(crate) unk_178: u32,

    // 15.5 tail, 0x190..0x1cc.  The handler passes `opcode + 0x190` itself
    // (`add x0, x28, #0x18c`) to the pool registration at 0x24e34, which
    // dereferences it as a struct:
    //
    //   [0x190]  pointer, dereferenced (`ldr x26, [x0]; ldr w8, [x26, #8]`)
    //   [0x198]  pointer, may be null (`ldr x8, [x20, #8]; cbz x8, ...`)
    //   [0x1a0]  u32, becomes bits 52+ of a hardware descriptor -- keep 0
    //   [0x1b0]  byte, becomes bit 9 of the same descriptor
    //
    // `notifier_buf` moved from 0x188 to 0x1c0: the kext stores `x22 + 0xa8`
    // there, and 0xa8 is exactly `NotifierState::unk_buf`.  LOG.md 32.41.25.
    #[ver(V >= V15_5)]
    pub(crate) pool_ptr: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_198: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_1a0: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_1a4: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_1a8: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_1b0: Array<0x8, u8>,
    #[ver(V >= V15_5)]
    pub(crate) counter: U64,
    #[ver(V >= V15_5)]
    pub(crate) notifier_buf: GpuWeakPointer<Array<0x8, u8>>,
    #[ver(V >= V15_5)]
    pub(crate) unk_178: u32,
}

#[versions(AGX)]
impl<'a> Operation for StartVertex::ver<'a> {}

// Pinned against the firmware.  `x28` in the microsequence interpreter is
// `opcode + 4` (`add x28, x19, #0x4` at fw:0x14f48), so the handler's
// `add x0, x28, #0x18c` (fw:0x150dc) is `opcode + 0x190` -- the pool pointer it
// dereferences -- and `add x8, x28, #0x194` (fw:0x169cc, fw:0x17210) is
// `opcode + 0x198`, the UMA pool base.  Without these, a probe on `unk_198`
// would be testing whatever happened to land at that offset.  LOG.md 32.41.60.
const _: () = {
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, pool_ptr) == 0x190);
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, unk_198) == 0x198);
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, unk_1a0) == 0x1a0);
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, unk_1b0) == 0x1b0);

    // 32.41.510: the `0xd8f0` bit-enable pair.  StartVertex's tail does
    //     ldp  w8, w9, [x28, #64]      ; x28 = opcode + 4
    //     str  #0x100, [MMIO + 0xd800]
    //     lsl  x8, x27, x8             ; x27 == 1  -> mask = 1 << w8
    //     ldr  x11, [MMIO + 0xd8f0]
    //     orr  x12, x11, x8            ; set
    //     bic  x8,  x11, x8            ; clear
    //     cmp  w9, #0
    //     csel x8, x8, x12, eq         ; w9 == 0  ->  CLEAR, else SET
    //     str  x8, [MMIO + 0xd8f0]
    // (fw:0x15088..0x150bc).  So `w8` = opcode+0x44 is a BIT INDEX and
    // `w9` = opcode+0x48 is an ENABLE FLAG whose zero value *disables* the
    // bit.  The identical idiom appears in the op-0xb handler (fw:0x14e50,
    // pair at opcode+0x2c/+0x30) and StartCompute (fw:0x153a8), so the
    // (index, enable) reading is general rather than a local coincidence.
    // These two offsets are what make that reading load-bearing, so pin them.
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, event_generation) == 0x44);
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, buffer_slot) == 0x48);
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, unk_44) == 0x4c);
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, unk_pointer) == 0x64);
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, unk_job_buf) == 0x6c);
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, uuid) == 0x7c);
    assert!(core::mem::offset_of!(StartVertexG15XV15_5<'static>, attachments) == 0x88);

    // 32.41.500: the Timestamp record, pinned against Apple's OWN captured TA
    // microsequence (capture4 `out/raw-ta-microseq-draw.txt`, finishEncode #2).
    // Apple emits TWO of these between `StartVertex` and `FinalizeVertex`, each
    // occupying exactly 0x3c bytes:
    //
    //   +0x000  op 0x05  StartVertex     0x1cc   (matches our size_of exactly)
    //   +0x1cc  op 0x03  Timestamp       0x03c   hdr 0x80000003  flag = 1
    //   +0x208  op 0x01  WaitForIdle     0x004   hdr 0x00000001
    //   +0x20c  op 0x03  Timestamp       0x03c   hdr 0x00000003  flag = 0
    //   +0x248  op 0x06  FinalizeVertex  0x094   (matches our size_of exactly)
    //   +0x2dc  op 0x02  WaitForIdle2    0x004   hdr 0x40000002, at the write ptr
    //
    // The two records differ in exactly one field -- `update_ts`, by 8 bytes
    // (`...8fb8` vs `...8fc0`) -- which is `timestamp_pointers.start_addr` vs
    // `.end_addr`, adjacent u64s. That delta is what confirms the field map
    // below rather than merely the total size.
    assert!(core::mem::size_of::<TimestampG15XV15_5<'static>>() == 0x3c);
    assert!(core::mem::offset_of!(TimestampG15XV15_5<'static>, command_time) == 0x04);
    assert!(core::mem::offset_of!(TimestampG15XV15_5<'static>, ts_pointers) == 0x0c);
    assert!(core::mem::offset_of!(TimestampG15XV15_5<'static>, update_ts) == 0x14);
    assert!(core::mem::offset_of!(TimestampG15XV15_5<'static>, work_queue) == 0x1c);
    assert!(core::mem::offset_of!(TimestampG15XV15_5<'static>, user_ts_pointers) == 0x24);
    assert!(core::mem::offset_of!(TimestampG15XV15_5<'static>, unk_ts) == 0x2c);
    assert!(core::mem::offset_of!(TimestampG15XV15_5<'static>, uuid) == 0x34);

    // The fragment op's record, never pinned before. `fw:0x15c84` passes
    // `add x0, x28, #0x1a4`, and `x28 = opcode + 4`, so the record starts at
    // `opcode + 0x1a8` and its base is at `opcode + 0x1b0`. The emitter at
    // `fw:0x24e34` reads `[x0]` as the pool pointer, `[x0+8]` as the base,
    // `[x0+0x10]` as a u32 and `byte [x0+0x20]`. LOG.md 32.41.86.
    let _: [(); core::mem::offset_of!(StartFragmentG15XV15_5<'static>, pool_ptr)] = [(); 0x1a8];
    let _: [(); core::mem::offset_of!(StartFragmentG15XV15_5<'static>, unk_1b0)] = [(); 0x1b0];
};

#[versions(AGX)]
#[derive(Debug)]
#[repr(C)]
pub(crate) struct FinalizeVertex {
    pub(crate) header: op::FinalizeVertex,
    pub(crate) scene: GpuWeakPointer<buffer::Scene::ver>,
    pub(crate) buffer: GpuWeakPointer<buffer::Info::ver>,
    pub(crate) stats: GpuWeakPointer<initdata::raw::GpuStatsVtx>,
    pub(crate) work_queue: GpuWeakPointer<workqueue::QueueInfo::ver>,

    // Same substitution as StartVertex: 15.5 replaces `vm_slot` + `unk_28`
    // with a pointer, dereferenced at 0x15ea4 (`ldr x8, [x22, #32]` with
    // `x22 = opcode + 4`). Caught as a fault with FAR = 1, the driver's
    // vm_slot. LOG.md 32.41.19.
    #[ver(V < V15_5)]
    pub(crate) vm_slot: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_28: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_24_ptr: GpuWeakPointer<u32>,

    #[ver(V < V15_5)]
    pub(crate) unk_pointer: GpuWeakPointer<u32>,
    #[ver(V < V15_5)]
    pub(crate) unk_34: u32,
    #[ver(V < V15_5)]
    pub(crate) uuid: u32,
    #[ver(V < V15_5)]
    pub(crate) fw_stamp: GpuWeakPointer<FwStamp>,
    #[ver(V < V15_5)]
    pub(crate) stamp_value: EventValue,
    #[ver(V < V15_5)]
    pub(crate) unk_48: U64,
    #[ver(V < V15_5)]
    pub(crate) unk_50: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_54: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_58: U64,
    #[ver(V < V15_5)]
    pub(crate) unk_60: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_64: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_68: u32,

    #[ver(G >= G14 && V < V13_0B4)]
    pub(crate) unk_68_g14: U64,

    #[ver(V < V15_5)]
    pub(crate) restart_branch_offset: i32,
    #[ver(V < V15_5)]
    pub(crate) has_attachments: u32, // Check DCMP errors bits 2,3 1=ktrace 2=log 3=panic

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) unk_74: Array<0x10, u8>,

    // 15.5 rebuilt everything from 0x2c up.  Read off the kext's op-6 emitter
    // at 0x911a60c, which writes the opcode in place rather than through a
    // memcpy, and cross-checked against every field the firmware's handler at
    // 0x15840 touches (it runs with `x22 = opcode + 4`, so its displacements
    // are 4 low):
    //
    //   0x34  pointer   `ldr x12, [x22, #48]`   -- same value the kext puts in
    //                                              StartVertex's unk_job_buf
    //   0x40  u64       `ldur x2, [x22, #60]`   -- trace path only
    //   0x48  pointer   `ldur x10, [x22, #68]` + `prfm pstl1strm, [x10]`
    //   0x50  u32       `ldr w8, [x22, #76]`
    //   0x58  struct    `add x0, x22, #0x54`, passed to 0x2b074
    //   0x78  pointer   `ldur x20, [x22, #116]` + `prfm pstl1strm, [x20]`
    //   0x84  u8        `ldrb w1, [x22, #128]`
    //   0x85  pointer   `ldur x25, [x22, #129]` (unaligned)
    //
    // The stamp pair at 0x48/0x50 is what the whole vertex path was missing:
    // the driver left 0x48 zero, so the firmware cached a null stamp pointer
    // in its per-pipe block (`0xe4098 + pipe * 0x168 + 0x268`, field +0x40)
    // and took a data abort at 0x1b21c on `str w20, [x21]` when the job
    // completed.  LOG.md 32.41.25.
    #[ver(V >= V15_5)]
    pub(crate) unk_2c: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_30: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_34_ptr: GpuWeakPointer<u32>,
    #[ver(V >= V15_5)]
    pub(crate) unk_3c: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_40: U64,
    #[ver(V >= V15_5)]
    pub(crate) fw_stamp: GpuWeakPointer<FwStamp>,
    #[ver(V >= V15_5)]
    pub(crate) stamp_value: EventValue,
    #[ver(V >= V15_5)]
    pub(crate) unk_54: Array<0x20, u8>,
    #[ver(V >= V15_5)]
    pub(crate) unk_74: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_78_ptr: GpuWeakPointer<U64>,
    // The interpreter takes the restart branch from opcode+0x80 on 15.5
    // (`ldrsw x8, [x19, #128]; add x19, x19, x8`), not 0x6c.
    #[ver(V >= V15_5)]
    pub(crate) restart_branch_offset: i32,
    #[ver(V >= V15_5)]
    pub(crate) has_attachments: u8,
    #[ver(V >= V15_5)]
    pub(crate) unk_85_ptr: GpuWeakPointer<U64>,
    // 0x94 on 15.5, against the driver's 0x84: the handler's normal exit sets
    // the interpreter's advance to 0x94 (`mov w21, #0x94` at 0x16368; the
    // 0x1c8 at 0x1635c is the other branch). LOG.md 32.41.19.
    #[ver(V >= V15_5)]
    pub(crate) unk_8d: Array<0x7, u8>,
}

#[versions(AGX)]
impl Operation for FinalizeVertex::ver {}

#[versions(AGX)]
#[derive(Debug)]
#[repr(C)]
pub(crate) struct StartFragment<'a> {
    pub(crate) header: op::StartFragment,
    pub(crate) job_params2: Option<GpuWeakPointer<fragment::raw::JobParameters2>>,
    pub(crate) job_params1: Option<GpuWeakPointer<fragment::raw::JobParameters1::ver<'a>>>,
    #[ver(G >= G14X)]
    pub(crate) registers: GpuWeakPointer<job::raw::RegisterArray>,
    pub(crate) scene: GpuPointer<'a, buffer::Scene::ver>,
    pub(crate) stats: GpuWeakPointer<initdata::raw::GpuStatsFrag::ver>,
    pub(crate) busy_flag: GpuWeakPointer<u32>,
    pub(crate) tvb_overflow_count: GpuWeakPointer<u32>,
    pub(crate) unk_pointer: GpuWeakPointer<u32>,
    pub(crate) work_queue: GpuWeakPointer<workqueue::QueueInfo::ver>,
    pub(crate) work_item: GpuWeakPointer<fragment::RunFragment::ver>,
    pub(crate) vm_slot: u32,
    pub(crate) unk_50: u32,
    pub(crate) event_generation: u32,
    pub(crate) buffer_slot: u32,
    pub(crate) sync_grow: u32,
    pub(crate) event_seq: U64,
    pub(crate) unk_68: u32,
    pub(crate) unk_758_flag: GpuWeakPointer<u32>,
    pub(crate) unk_job_buf: GpuWeakPointer<U64>,
    #[ver(V >= V13_3 && V < V15_5)]
    pub(crate) unk_7c_0: U64,

    // 15.5 keeps everything up to `unk_job_buf` where it was, replaces the
    // V13_3 `unk_7c_0` with a plain u32, and inserts an 8-byte pointer at 0x98
    // before the attachment array -- which lands at 0xa0, count at 0x1a0.
    // Confirmed by the firmware's own walker (handler 7 at 0x15488 reads the
    // count with `ldr w8, [x28, #412]`, x28 = opcode + 4) and by the kext's
    // builder at 0x910e238, whose payload is memcpy'd from `sp+0x80` with
    // length 0x1e0, so a field's offset is its sp offset minus 124.
    // LOG.md 32.41.26.
    #[ver(V >= V15_5)]
    pub(crate) unk_84_0: u32,

    pub(crate) unk_7c: u32,
    pub(crate) unk_80: u32,
    pub(crate) unk_84: u32,
    pub(crate) uuid: u32,

    #[ver(V >= V15_5)]
    pub(crate) unk_98: U64,

    pub(crate) attachments: Attachments,
    pub(crate) padding: u32,

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) counter: U64,

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) notifier_buf: GpuWeakPointer<Array<0x8, u8>>,

    // 15.5 tail, 0x1a8..0x1e4.  Same shape as StartVertex's: the handler hands
    // `opcode + 0x1a8` to the pool registration at 0x24e34 (`add x0, x28,
    // #0x1a4`), which dereferences the first word.
    #[ver(V >= V15_5)]
    pub(crate) pool_ptr: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_1b0: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_1b8: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_1bc: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_1c0: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_1c8: Array<0x8, u8>,
    #[ver(V >= V15_5)]
    pub(crate) counter: U64,
    #[ver(V >= V15_5)]
    pub(crate) notifier_buf: GpuWeakPointer<Array<0x8, u8>>,
    #[ver(V >= V15_5)]
    pub(crate) unk_1e0: u32,
}

#[versions(AGX)]
impl<'a> Operation for StartFragment::ver<'a> {}

#[versions(AGX)]
#[derive(Debug)]
#[repr(C)]
pub(crate) struct FinalizeFragment {
    pub(crate) header: op::FinalizeFragment,
    pub(crate) uuid: u32,
    pub(crate) unk_8: u32,
    pub(crate) fw_stamp: GpuWeakPointer<FwStamp>,
    pub(crate) stamp_value: EventValue,
    pub(crate) unk_18: u32,
    pub(crate) scene: GpuWeakPointer<buffer::Scene::ver>,
    pub(crate) buffer: GpuWeakPointer<buffer::Info::ver>,
    #[ver(V < V15_5)]
    pub(crate) unk_2c: U64,
    // 15.5 narrows `unk_2c` to a u32 and drops `unk_60`, so everything from
    // `stats` to `unk_758_flag` sits 4 and then 8 bytes lower than it used to.
    // The stamp pair does *not* move here -- unlike FinalizeVertex, the
    // fragment finalizer keeps `fw_stamp` at 0x0c and `stamp_value` at 0x14
    // (0x16494 and 0x1682c cache exactly `[x23, #8]` and `[x23, #16]` into the
    // per-pipe block, with x23 = opcode + 4).  LOG.md 32.41.26.
    #[ver(V >= V15_5)]
    pub(crate) unk_2c: u32,

    pub(crate) stats: GpuWeakPointer<initdata::raw::GpuStatsFrag::ver>,
    pub(crate) unk_pointer: GpuWeakPointer<u32>,
    pub(crate) busy_flag: GpuWeakPointer<u32>,
    pub(crate) work_queue: GpuWeakPointer<workqueue::QueueInfo::ver>,
    pub(crate) work_item: GpuWeakPointer<fragment::RunFragment::ver>,
    pub(crate) vm_slot: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_60: u32,
    pub(crate) unk_758_flag: GpuWeakPointer<u32>,
    #[ver(V >= V13_3 && V < V15_5)]
    pub(crate) unk_6c_0: U64,

    #[ver(V < V15_5)]
    pub(crate) unk_6c: U64,
    #[ver(V < V15_5)]
    pub(crate) unk_74: U64,
    #[ver(V < V15_5)]
    pub(crate) unk_7c: U64,
    #[ver(V < V15_5)]
    pub(crate) unk_84: U64,
    #[ver(V < V15_5)]
    pub(crate) unk_8c: U64,

    #[ver(G == G14 && V < V13_0B4)]
    pub(crate) unk_8c_g14: U64,

    #[ver(V < V15_5)]
    pub(crate) restart_branch_offset: i32,
    #[ver(V < V15_5)]
    pub(crate) has_attachments: u32, // Check DCMP errors bits 2,3 1=ktrace 2=log 3=panic

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) unk_9c: Array<0x10, u8>,

    // 15.5 tail from 0x64.  Handler 8's prologue at 0x15d68 names the three
    // pointers it will write through: `[x23, #8]` (fw_stamp), `[x23, #88]`
    // (unk_758_flag, now 0x5c) and `[x23, #140]` (0x90, the pool word) all get
    // a `prfm pstl1strm`.  The 32-byte run at 0x70 is the same zeroed blob
    // FinalizeVertex carries at 0x54.
    #[ver(V >= V15_5)]
    pub(crate) unk_64: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_68: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_70: Array<0x20, u8>,
    #[ver(V >= V15_5)]
    pub(crate) pool_ptr_ptr: GpuWeakPointer<U64>,
    #[ver(V >= V15_5)]
    pub(crate) restart_branch_offset: i32,
    #[ver(V >= V15_5)]
    pub(crate) has_attachments: u8,
    #[ver(V >= V15_5)]
    pub(crate) unk_9d_ptr: GpuWeakPointer<U64>,
    #[ver(V >= V15_5)]
    pub(crate) unk_a5: Array<0x7, u8>,
}

#[versions(AGX)]
impl Operation for FinalizeFragment::ver {}

#[versions(AGX)]
#[derive(Debug)]
#[repr(C)]
pub(crate) struct StartCompute<'a> {
    pub(crate) header: op::StartCompute,
    pub(crate) unk_pointer: GpuWeakPointer<u32>,
    pub(crate) job_params1: Option<GpuWeakPointer<compute::raw::JobParameters1<'a>>>,
    #[ver(G >= G14X)]
    pub(crate) registers: GpuWeakPointer<job::raw::RegisterArray>,

    // 15.5 inserts a u32 here, pushing everything from `stats` down by 4, and
    // an 8-byte field at 0x58, putting the attachment array at 0x60 and its
    // count at 0x160 -- which is what handler 9 reads (`ldr w8, [x28, #348]`,
    // x28 = opcode + 4).  The kext's builder at 0x910dbac memcpy's the payload
    // from `sp+0x40` with length 0x1a8, so a field's offset is its sp offset
    // minus 60.  LOG.md 32.41.26.
    #[ver(V >= V15_5)]
    pub(crate) unk_1c: u32,

    pub(crate) stats: GpuWeakPointer<initdata::GpuStatsComp>,
    pub(crate) work_queue: GpuWeakPointer<workqueue::QueueInfo::ver>,
    pub(crate) vm_slot: u32,
    pub(crate) unk_28: u32,
    pub(crate) event_generation: u32,
    pub(crate) event_seq: U64,
    pub(crate) unk_38: u32,
    pub(crate) job_params2: GpuWeakPointer<compute::raw::JobParameters2::ver<'a>>,
    pub(crate) unk_44: u32,
    pub(crate) uuid: u32,

    #[ver(V >= V15_5)]
    pub(crate) unk_58: U64,

    pub(crate) attachments: Attachments,
    pub(crate) padding: u32,

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) unk_flag: GpuWeakPointer<U32>,

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) counter: U64,

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) notifier_buf: GpuWeakPointer<Array<0x8, u8>>,

    // 15.5 tail, 0x168..0x1ac.  The pool registration takes `opcode + 0x168`
    // (`add x0, x28, #0x164` at 0x15a00).  Same shape as the vertex and
    // fragment tails, plus the extra 8-byte slot compute has always had.
    #[ver(V >= V15_5)]
    pub(crate) pool_ptr: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_170: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_178: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_17c: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_180: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_188: Array<0x8, u8>,
    #[ver(V >= V15_5)]
    pub(crate) unk_flag: GpuWeakPointer<U32>,
    #[ver(V >= V15_5)]
    pub(crate) counter: U64,
    #[ver(V >= V15_5)]
    pub(crate) notifier_buf: GpuWeakPointer<Array<0x8, u8>>,
    #[ver(V >= V15_5)]
    pub(crate) unk_1a8: u32,
}

#[versions(AGX)]
impl<'a> Operation for StartCompute::ver<'a> {}

#[versions(AGX)]
#[derive(Debug)]
#[repr(C)]
pub(crate) struct FinalizeCompute<'a> {
    pub(crate) header: op::FinalizeCompute,
    pub(crate) stats: GpuWeakPointer<initdata::GpuStatsComp>,
    pub(crate) work_queue: GpuWeakPointer<workqueue::QueueInfo::ver>,
    pub(crate) vm_slot: u32,
    #[ver(V < V13_0B4)]
    pub(crate) unk_18: u32,

    // 15.5 widens the slot at 0x18 to a u32 pair and turns `uuid` into an
    // 8-byte field, so the stamp pair lands 4 bytes lower than before: 0x2c
    // and 0x34, which is what 0x173d0 caches into the per-pipe block
    // (`ldr x8, [x23, #40]`, `ldr w8, [x23, #48]`, x23 = opcode + 4).  Read
    // off the kext's op-a emitter at 0x910e14c, which writes the opcode in
    // place.  LOG.md 32.41.26.
    #[ver(V >= V15_5)]
    pub(crate) unk_18_15_5: u32,

    pub(crate) job_params2: GpuWeakPointer<compute::raw::JobParameters2::ver<'a>>,

    #[ver(V < V15_5)]
    pub(crate) unk_24: u32,
    #[ver(V < V15_5)]
    pub(crate) uuid: u32,
    #[ver(V >= V15_5)]
    pub(crate) unk_24_ptr: GpuWeakPointer<U64>,

    pub(crate) fw_stamp: GpuWeakPointer<FwStamp>,
    pub(crate) stamp_value: EventValue,

    #[ver(V < V15_5)]
    pub(crate) unk_38: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_3c: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_40: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_44: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_48: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_4c: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_50: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_54: u32,
    #[ver(V < V15_5)]
    pub(crate) unk_58: u32,

    #[ver(G == G14 && V < V13_0B4)]
    pub(crate) unk_5c_g14: U64,

    #[ver(V < V15_5)]
    pub(crate) restart_branch_offset: i32,
    #[ver(V < V15_5)]
    pub(crate) has_attachments: u32, // Check DCMP errors bits 2,3 1=ktrace 2=log 3=panic

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) unk_64: Array<0xd, u8>,

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) unk_flag: GpuWeakPointer<U32>,

    #[ver(V >= V13_0B4 && V < V15_5)]
    pub(crate) unk_79: Array<0x7, u8>,

    // 15.5 tail from 0x38: the zeroed 32-byte blob, then the pool word, the
    // restart offset, the attachment flag byte, and three unaligned pointers.
    #[ver(V >= V15_5)]
    pub(crate) unk_38_blob: Array<0x20, u8>,
    #[ver(V >= V15_5)]
    pub(crate) unk_58_15_5: u32,
    #[ver(V >= V15_5)]
    pub(crate) pool_ptr_ptr: GpuWeakPointer<U64>,
    #[ver(V >= V15_5)]
    pub(crate) restart_branch_offset: i32,
    #[ver(V >= V15_5)]
    pub(crate) has_attachments: u8,
    #[ver(V >= V15_5)]
    pub(crate) unk_69_ptr: GpuWeakPointer<U64>,
    #[ver(V >= V15_5)]
    pub(crate) unk_71: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_79_15_5: U64,
    #[ver(V >= V15_5)]
    pub(crate) unk_81_ptr: GpuWeakPointer<U64>,
    #[ver(V >= V15_5)]
    pub(crate) unk_89: Array<0x3, u8>,
}

#[versions(AGX)]
impl<'a> Operation for FinalizeCompute::ver<'a> {}

/// Apple's 15.5 compute pair: opcodes **0xb / 0xc**, not 9 / 0xa.
///
/// Read off macOS's own compute microsequence (capture 2026-09-27, dt-big,
/// `apple-compute-useq.bin`): StartCompute 0xb (0x1c4 bytes) -> Timestamp(1)
/// -> WaitForIdle (bare 1) -> Timestamp(0) -> FinalizeCompute 0xc (0x7c) ->
/// 0x40000002. The firmware agrees: handler 0xb (fw:0x14d68) steps a 0x1c0
/// payload (fw:0x1599c) and reads the attachment count at op+0x158
/// (`ldr w8, [x28, #340]`), entries from op+0x58; handler 0xc steps 0x7c
/// (fw:0x16634). Handler 0xb appends 0x10201/0x10428 (vm_slot<<8|evgen), then
/// 0x14028/0x14070 from the pool record at op+0x160 (op+0x178 = UMA scratch),
/// then the six spill registers from the four VAs at op+0x188. All pointers
/// are raw `U64` so the layout is exactly the captured one. LOG.md 32.41.572.
#[derive(Debug)]
#[repr(C)]
pub(crate) struct StartComputeApple {
    pub(crate) header: u32,
    pub(crate) unk_4: U64,
    pub(crate) unk_c: U64,
    pub(crate) registers: U64,
    pub(crate) stats: U64,
    pub(crate) work_queue: U64,
    pub(crate) vm_slot: u32,
    pub(crate) unk_30: u32,
    pub(crate) event_generation: u32,
    pub(crate) event_seq: U64,
    pub(crate) unk_40: u32,
    pub(crate) job_params2: U64,
    pub(crate) unk_4c: u32,
    pub(crate) uuid: u32,
    pub(crate) unk_54: u32,
    pub(crate) attachments: Attachments,
    pub(crate) padding: u32,
    pub(crate) pool_ptr: U64,
    pub(crate) unk_168: U64,
    pub(crate) unk_170: U64,
    pub(crate) uma_scratch: U64,
    pub(crate) unk_180: U64,
    pub(crate) spill: [U64; 4],
    pub(crate) unk_flag: U64,
    pub(crate) counter: U64,
    pub(crate) notifier_buf: U64,
    pub(crate) unk_1c0: u32,
}
impl Operation for StartComputeApple {}

/// See `StartComputeApple`. Apple's 0xc: `job_params2` right after `vm_slot`,
/// `uuid` at 0x24, the stamp pair at 0x2c/0x34, then the same tail shape as
/// our 0xa up to 0x79, which holds 0xff. LOG.md 32.41.572.
#[derive(Debug)]
#[repr(C)]
pub(crate) struct FinalizeComputeApple {
    pub(crate) header: u32,
    pub(crate) stats: U64,
    pub(crate) work_queue: U64,
    pub(crate) vm_slot: u32,
    pub(crate) job_params2: U64,
    pub(crate) unk_20: u32,
    pub(crate) uuid: u32,
    pub(crate) unk_28: u32,
    pub(crate) fw_stamp: U64,
    pub(crate) stamp_value: EventValue,
    pub(crate) unk_38: Array<0x24, u8>,
    pub(crate) pool_ptr_ptr: U64,
    pub(crate) restart_branch_offset: i32,
    pub(crate) has_attachments: u8,
    pub(crate) unk_flag: U64,
    pub(crate) unk_71: U64,
    pub(crate) unk_79: u8,
    pub(crate) unk_7a: [u8; 2],
}
impl Operation for FinalizeComputeApple {}

const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<StartComputeApple>() == 0x1c4);
    assert!(offset_of!(StartComputeApple, registers) == 0x14);
    assert!(offset_of!(StartComputeApple, vm_slot) == 0x2c);
    assert!(offset_of!(StartComputeApple, job_params2) == 0x44);
    assert!(offset_of!(StartComputeApple, uuid) == 0x50);
    assert!(offset_of!(StartComputeApple, attachments) == 0x58);
    assert!(offset_of!(StartComputeApple, pool_ptr) == 0x160);
    assert!(offset_of!(StartComputeApple, uma_scratch) == 0x178);
    assert!(offset_of!(StartComputeApple, spill) == 0x188);
    assert!(offset_of!(StartComputeApple, unk_flag) == 0x1a8);
    assert!(offset_of!(StartComputeApple, notifier_buf) == 0x1b8);
    assert!(size_of::<FinalizeComputeApple>() == 0x7c);
    assert!(offset_of!(FinalizeComputeApple, job_params2) == 0x18);
    assert!(offset_of!(FinalizeComputeApple, uuid) == 0x24);
    assert!(offset_of!(FinalizeComputeApple, fw_stamp) == 0x2c);
    assert!(offset_of!(FinalizeComputeApple, stamp_value) == 0x34);
    assert!(offset_of!(FinalizeComputeApple, pool_ptr_ptr) == 0x5c);
    assert!(offset_of!(FinalizeComputeApple, restart_branch_offset) == 0x64);
    assert!(offset_of!(FinalizeComputeApple, unk_flag) == 0x69);
    assert!(offset_of!(FinalizeComputeApple, unk_79) == 0x79);
};
