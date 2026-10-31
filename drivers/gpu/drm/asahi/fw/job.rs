// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Common GPU job firmware structures

use super::types::*;
use crate::{
    default_zeroed,
    mmu,
    trivial_gpustruct, //
};
use kernel::prelude::Result;
use kernel::sync::Arc;

pub(crate) mod raw {
    use super::*;

    // 🔑 15.5 8-aligns the stamp quad and moves it from +0x04 to +0x10. The
    // firmware's completion walk takes the notifier's `pstamp_*` entries as
    // pointers to a 0x20-byte record and reads
    //
    //     ldp x26, x23, [x0]        ; record + 0x00, + 0x08   (two pointers)
    //     ldp w24, w25, [x0, #16]   ; record + 0x10, + 0x14   (value, slot)
    //     str w24, [x26]            ; write the stamp
    //
    // (`fw:0x1fb70`). The record it was handed for the vertex job was
    // `command + 0x868`, and `RunVertex::meta` sits at 0x858 -- so the quad
    // starts at `meta + 0x10`, not `meta + 0x04`. With the pre-15.5 layout the
    // firmware read `meta + 0x10` as the stamp pointer and got
    // `fw_stamp_hi | stamp_value << 32` = 0x100fffffe00, which is exactly the
    // FAR of the data abort it died on. The struct is still 0x2c bytes
    // (`unk_after_meta` is at 0x884 and the firmware reads `unk_buf_0` at
    // 0x888, both confirmed), so the four u32s move into the hole the quad
    // left. LOG.md 32.41.91.
    #[versions(AGX)]
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct JobMeta {
        pub(crate) unk_0: u16,
        pub(crate) unk_2: u8,
        pub(crate) no_preemption: u8,

        // ⚠️ 0x04..0x10 on 15.5 is *unidentified*. It is where `evctl_index`,
        // `flush_stamps` and `uuid` used to sit, but nothing says which of them
        // goes where now, and one of them is an **index**. Guessing the order
        // put a userspace-supplied UUID in that slot and wedged the machine
        // past ssh and past sysrq -- a power cycle, not a crash log. Zeroed
        // until something measures it. LOG.md 32.41.92.
        #[ver(V >= V15_5)]
        pub(crate) unk_4: Array<0xc, u8>,

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

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct EncoderParams {
        pub(crate) unk_8: u32,
        pub(crate) sync_grow: u32,
        pub(crate) unk_10: u32,
        pub(crate) encoder_id: u32,
        pub(crate) unk_18: u32,
        pub(crate) unk_mask: u32,
        pub(crate) sampler_array: U64,
        pub(crate) sampler_count: u32,
        pub(crate) sampler_max: u32,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobTimestamps {
        pub(crate) start: AtomicU64,
        pub(crate) end: AtomicU64,
    }
    default_zeroed!(JobTimestamps);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RenderTimestamps {
        pub(crate) vtx: JobTimestamps,
        pub(crate) frag: JobTimestamps,
    }
    default_zeroed!(RenderTimestamps);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Register {
        pub(crate) number: u32,
        pub(crate) value: U64,
    }
    default_zeroed!(Register);

    impl Register {
        fn new(number: u32, value: u64) -> Register {
            Register {
                number,
                value: U64(value),
            }
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RegisterArray {
        pub(crate) registers: Array<128, Register>,
        pub(crate) pad: Array<0x100, u8>,

        pub(crate) addr: GpuWeakPointer<Array<128, Register>>,
        pub(crate) count: u16,
        pub(crate) length: u16,
        pub(crate) unk_pad: u32,
    }

    impl RegisterArray {
        pub(crate) fn new(
            self_ptr: GpuWeakPointer<Array<128, Register>>,
            cb: impl FnOnce(&mut RegisterArray),
        ) -> RegisterArray {
            let mut array = RegisterArray {
                registers: Default::default(),
                pad: Default::default(),
                addr: self_ptr,
                count: 0,
                length: 0,
                unk_pad: 0,
            };

            cb(&mut array);

            array
        }

        /// Overwrite an already-added register's value in place, leaving the
        /// list length and ordering untouched. Returns false if absent.
        ///
        /// Used to apply macOS's *small-job* values over registers this driver
        /// programs unconditionally -- appending a second record for the same
        /// id would change the list length, which the firmware's count and the
        /// hardware's walk both depend on. LOG.md 32.41.371 (entry LOST -- see 32.41.431).
        pub(crate) fn set(&mut self, number: u32, value: u64) -> bool {
            for i in 0..(self.count as usize) {
                if self.registers[i].number == number {
                    self.registers[i] = Register::new(number, value);
                    return true;
                }
            }
            false
        }

        pub(crate) fn add(&mut self, number: u32, value: u64) {
            self.registers[self.count as usize] = Register::new(number, value);
            self.count += 1;
            self.length += core::mem::size_of::<Register>() as u16;
        }

        /// Print the array as built, so it can be diffed against a macOS
        /// capture (m1n1's `agx/render.py` carries one).  LOG.md 32.41.41.
        pub(crate) fn dump(&self, tag: &str) {
            for i in 0..(self.count as usize) {
                let number = self.registers[i].number;
                let value = self.registers[i].value.0;
                kernel::pr_info!("asahi/reg {} [{}] {:#x} = {:#x}\n", tag, i, number, value);
            }
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct TimestampPointers<'a> {
        pub(crate) start_addr: Option<GpuPointer<'a, AtomicU64>>,
        pub(crate) end_addr: Option<GpuPointer<'a, AtomicU64>>,
    }
}

trivial_gpustruct!(JobTimestamps);
trivial_gpustruct!(RenderTimestamps);

#[derive(Debug)]
pub(crate) struct UserTimestamp {
    pub(crate) mapping: Arc<mmu::KernelMapping>,
    pub(crate) offset: usize,
}

#[derive(Debug, Default)]
pub(crate) struct UserTimestamps {
    pub(crate) start: Option<UserTimestamp>,
    pub(crate) end: Option<UserTimestamp>,
}

impl UserTimestamps {
    pub(crate) fn any(&self) -> bool {
        self.start.is_some() || self.end.is_some()
    }

    pub(crate) fn pointers(&self) -> Result<raw::TimestampPointers<'_>> {
        Ok(raw::TimestampPointers {
            start_addr: self
                .start
                .as_ref()
                .map(|a| GpuPointer::from_mapping(&a.mapping, a.offset))
                .transpose()?,
            end_addr: self
                .end
                .as_ref()
                .map(|a| GpuPointer::from_mapping(&a.mapping, a.offset))
                .transpose()?,
        })
    }
}
