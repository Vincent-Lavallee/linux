// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU UAT (MMU) management
//!
//! AGX GPUs use an MMU called the UAT, which is largely compatible with the ARM64 page table
//! format. This module manages the global MMU structures, including a shared handoff structure
//! that is used to coordinate VM management operations with the firmware, the TTBAT which points
//! to currently active GPU VM contexts, as well as the individual `Vm` operations to map and
//! unmap buffer objects into a single user or kernel address space.
//!
//! The actual page table management is in the `pt` module.

use core::fmt::Debug;
use core::mem::size_of;
use core::num::NonZeroUsize;
use core::ops::Range;
use core::sync::atomic::{
    fence,
    AtomicU32,
    AtomicU64,
    AtomicU8,
    Ordering, //
};

use core::cmp;

use kernel::{
    addr::PhysicalAddr,
    bindings::drm_gpuvm_flags_DRM_GPUVM_IMMEDIATE_MODE,
    c_str,
    device,
    drm::{
        gem::shmem,
        gpuvm,
        mm, //
    },
    error::Result,
    io,
    new_mutex,
    prelude::*,
    static_lock_class,
    sync::{
        aref::ARef,
        lock::{
            mutex::MutexBackend,
            Guard, //
        },
        Arc,
        Mutex, //
    },
    time::{
        delay::fsleep,
        Delta,
        Instant,
        Monotonic, //
    }, //
};

use crate::debug::*;
use crate::module_parameters;
use crate::no_debug;
use crate::{
    driver,
    fw,
    gem,
    hw,
    mem,
    pgtable,
    slotalloc,
    util::RangeExt, //
};

// KernelMapping protection types
pub(crate) use crate::pgtable::Prot;

/// .593: `user_prot=1` maps user-visible GPU buffers with Apple's G15 user-RW
/// encoding (AP=0 + b53/b54, uncached) instead of AP=2.
/// .667zk: `user_memattr` overrides the memory attribute index of user mappings
/// (Apple's encodePTEFlags maps user memory cached = 0 unless flagged uncached).
pub(crate) fn uprot(p: Prot) -> Prot {
    let p = if *crate::module_parameters::user_prot.value() != 0 {
        pgtable::prot::PROT_GPU_FW_SHARED_RW
    } else {
        p
    };
    match *crate::module_parameters::user_memattr.value() {
        m @ 0..=2 => p.with_memattr(m as u8),
        _ => p,
    }
}
pub(crate) use pgtable::prot::*;
pub(crate) use pgtable::{
    UatPageTable,
    UAT_PGBIT,
    UAT_PGMSK,
    UAT_PGSZ, //
};

use pgtable::UAT_IAS;

use pin_init;

const DEBUG_CLASS: DebugFlags = DebugFlags::Mmu;

/// PPL magic number for the handoff region
const PPL_MAGIC: u64 = 0x4b1d000000000002;

/// Shift for the user address-space alias, if any.  See `user_va_shift`.
///
/// The G15 VDM asks the MMU for `VA + 2^36` for every address it forms --
/// registers and pointers it reads out of the control stream alike -- but
/// `gUPM` and `gPM` do **not**: once the mappings were moved up by 2^36 the VDM
/// ran clean and those two started faulting on plain, mapped IOVAs. So the two
/// halves of the GPU disagree about where user memory is, and every user page
/// is mapped at both addresses until the mechanism behind the VDM's offset is
/// found. LOG.md 32.41.39/42.
fn user_iova_bias() -> u64 {
    let shift = *module_parameters::user_va_shift.value();
    if shift == 0 {
        0
    } else {
        1u64 << shift
    }
}

/// Second, arbitrary user alias.
///
/// On G14X+ the driver has no working channel to program USC_EXEC_BASE: both
/// `JobParameters1::usc_exec_base_{ta,isp}` are inside `#[ver(G < G14X)]` and
/// reg 0x10061 is inert (proved with a distinctive value, LOG.md 32.41.191).
/// The VDM ends up forming USC addresses against `usc_exec_base + 2^32`, so the
/// pipeline is fetched at `VA + usc_exec_base` instead of `VA + 2^36`. Mapping a
/// third copy there lets the fetch resolve so the *next* blocker becomes
/// visible. Diagnostic, not a fix -- the real answer is whatever Apple programs.
fn user_iova_bias2() -> u64 {
    *module_parameters::user_va_bias2.value()
}

/// Third, arbitrary user alias.
///
/// `user_va_bias2` covers the USC fetches the VDM forms as
/// `Mesa's VA + one 4 GiB code-GART window` (LOG.md 32.41.258).  The blocker
/// after that is a *base-only* read -- offset 0 within a window -- at
/// `0x1500000000`, which wants the **start of Mesa's USC heap**
/// (`usc_exec_base`, `0x1100000000`), not a zeroed scratch page.  A third
/// alias at `0x400000000` puts it there.  Diagnostic, like the other two.
/// LOG.md 32.41.263.
fn user_iova_bias3() -> u64 {
    *module_parameters::user_va_bias3.value()
}

/// Alias at `2^36 | (VA & 0xffffffff)`, the address the VDM actually forms.
///
/// Measured, not guessed: poking a 32-bit value into the control stream's
/// operand slot at `cs+0x2c` makes the GPU fault at exactly
/// `2^36 + value + small`, tracked across three distinct values over seven runs
/// (LOG.md 32.41.312/.313). So stream operands are 32-bit offsets resolved
/// against a 2^36 window base, and Mesa's buffers -- which live at
/// `0x2fxxxxxxxx` and `0x38xxxxxxxx` -- are not reachable that way by any of the
/// constant `VA + k` aliases the driver already makes.
///
/// This one is *not* a constant bias: it depends on the address, because the
/// buffers do not share high bits. Returns a bias, so the existing loops can
/// carry it unchanged.
/// 32.41.436: the TA needs BOTH alias forms backed at once. win0-1 showed the
/// heapmeta read goes to the bare folded VA (`0x148180`) while the cluster
/// layermeta read goes to the 2^36 form (`0x1000178080`, with `0x178080` mapped).
/// Different TA accesses apply different bases, so a single target cannot satisfy
/// both; map the alias twice and let each access find its own.
fn win36_bias2(base: u64) -> u64 {
    if *module_parameters::win36_dual.value() == 0 {
        return 0;
    }
    win36_bias_to(base, *module_parameters::win36_base.value() ^ (1u64 << 36))
}

fn win36_bias(base: u64) -> u64 {
    win36_bias_to(base, *module_parameters::win36_base.value())
}

fn win36_bias_to(base: u64, target_base: u64) -> u64 {
    if *module_parameters::user_va_win36.value() == 0 {
        return 0;
    }
    // 32.41.419: a blanket win36 alias collides. Every BO is mapped at
    // `2^36 | (VA & 0xffffffff)`, so two BOs whose VAs differ only above bit 32
    // fight over one alias -- and the USC code BO at 0x11fffbc000 LOST to some
    // other buffer sharing low bits 0xfffbc000, leaving a zero page exactly
    // where the test needed the shader. Restricting the alias to the USC heap
    // window [win36_min_va, win36_max_va) lets the code win uncontested and
    // excludes the other heaps, which do not need it.
    //
    // It must be a WINDOW, not a floor. Mesa's heaps are USC at 0x11xxxxxxxx,
    // control stream at 0x2fxxxxxxxx, tiler at 0x38xxxxxxxx -- the other two sit
    // ABOVE the USC heap, so `VA >= usc_exec_base` excludes nothing at all. That
    // was 32.41.420's void run.
    // 32.41.433: a SECOND window is needed, not a wider one. The first window
    // is the USC heap (0x11xxxxxxxx); the tiler buffers the TA address registers
    // point at live at 0x38xxxxxxxx, and widening window 1 to reach them would
    // swallow Mesa's control stream (0x2fxxxxxxxx) in between -- exactly the
    // collision .419 traced to a zero page where the shader should have been.
    // Each window is checked independently.
    let min = *module_parameters::win36_min_va.value();
    let max = *module_parameters::win36_max_va.value();
    let min2 = *module_parameters::win36_min_va2.value();
    let max2 = *module_parameters::win36_max_va2.value();
    let in1 = base >= min && (max == 0 || base < max) && (min != 0 || max != 0);
    let in2 = base >= min2 && (max2 == 0 || base < max2) && (min2 != 0 || max2 != 0);
    if !in1 && !in2 {
        return 0;
    }
    // 32.41.435: the alias TARGET is a parameter, not a constant. With the
    // register folded, the GPU faulted on a read at the bare folded VA
    // (0x148180 = heapmeta 0x148000 + 0x180), *not* at 0x1000148180 -- so the TA
    // consumes the register value as a plain GPU VA and adds no 2^36 base. The
    // buffer therefore has to be mapped at `base & 0xffffffff`, which is
    // win36_base = 0. The 2^36 target remains the default for the USC-heap use.
    let target = target_base | (base & 0xffff_ffff);
    if target == base {
        return 0;
    }
    target.wrapping_sub(base)
}

/// Number of supported context entries in the TTBAT
const UAT_NUM_CTX: usize = 64;
/// First context available for users
const UAT_USER_CTX_START: usize = 1;
/// Number of available user contexts. Slot 63 is kept out of the pool: macOS
/// reserves it for hardware-read kernel tables (`kctx63`, LOG.md 32.41.550).
const UAT_USER_CTX: usize = UAT_NUM_CTX - UAT_USER_CTX_START - 1;

/// Top-level slot of the klow range (`gpu::IOVA_KERN_GPU_KLOW_RANGE`), shared into
/// every user VM. LOG.md 32.41.667zzj.
const KLOW_TOP_IDX: usize = (crate::gpu::IOVA_KERN_GPU_KLOW_RANGE.start >> 36) as usize;
const _: () = assert!(
    (crate::gpu::IOVA_KERN_GPU_KLOW_RANGE.end - 1) >> 36 == crate::gpu::IOVA_KERN_GPU_KLOW_RANGE.start >> 36
);

/// Top-level slot of the pass-page counter's LOW page (`gpu::IOVA_KERN_GPU_PMCT_LOW`),
/// shared into every user VM under `pm_dualmap`. LOG.md 32.41.667zzx.
const PMCT_TOP_IDX: usize = (crate::gpu::IOVA_KERN_GPU_PMCT_LOW >> 36) as usize;
const _: () = assert!(crate::gpu::IOVA_KERN_GPU_PMCT_LOW < (1 << 39) && PMCT_TOP_IDX >= 5);

/// Lower/user base VA
///
/// 2^36, not one page. macOS will not hand a GPU VA below 2^36 to any client:
/// `AGXGart::getCodeGartRangeInfo` returns base `0x1000000000` size
/// `0x5c00000000`, and `AGXMemoryMap::reserveGPUVirtualAddress` rejects both
/// `(va >> 36) == 0` and `(va >> 34) >= 0x1b`. That is the same fact as the
/// 1-based 2-bit segment the VDM carries in control-stream-base bits [37:36] --
/// there is no segment 0, so region 0 is unaddressable. Starting the user
/// allocator at one page hands out addresses the hardware cannot reach.
/// LOG.md 32.41.128.
///
/// ⚠️ Tested 2026-09-02 as `1 << 36` and reverted: Mesa's *only* sub-2^36
/// allocation is its zero page, at exactly 0x100000000, and refusing it kills
/// screen creation ("Failed to bind zero page", no EGL, no render at all).
/// Everything else Mesa binds is already >= 2^36. The interesting part is
/// where that page's `+user_va_shift` alias lands: 0x100000000 + 2^36 =
/// 0x1100000000, which is byte-for-byte Mesa's `usc_exec_base`.
pub(crate) const IOVA_USER_BASE: u64 = UAT_PGSZ as u64;
/// Lower/user top VA
pub(crate) const IOVA_USER_TOP: u64 = 1 << (UAT_IAS as u64);
/// Lower/user VA range
pub(crate) const IOVA_USER_RANGE: Range<u64> = IOVA_USER_BASE..IOVA_USER_TOP;

/// Upper/kernel base VA
#[cfg(CONFIG_DEV_COREDUMP)]
const IOVA_TTBR1_BASE: u64 = 0xfffffc0000000000;
// The driver-managed kernel VA range starts immediately after the firmware's
// own private VM region, which the ADT declares per SoC:
//
//     kern_va_base = rtkit-private-vm-region-base + rtkit-private-vm-region-size
//
// which is exactly how m1n1's AGX code derives it rather than hardcoding it.
// On t6030 the ADT says base 0xfffffc0000000000, size 0x2000000000, giving
// 0xfffffc2000000000 -- *not* the 0xffffffa000000000 these constants carried,
// which is the M1/M2 value.
//
// 🔑 LOG.md 32.41.601: this used to say 0xfffffe0000000000, an addition slip
// (0xfffffc00_0000_0000 + 0x20_0000_0000 = 0xfffffc20_0000_0000). Apple's own
// kernel GPU objects all sit at 0xfffffc20... . It mattered: the hardware's
// folded address forms (PB descriptor, 0x1c910, ...) keep only VA bits 0..38
// plus the TTB1 select, so with UAT_IAS = 42 they can reach 0xfffffc00_... ..
// 0xfffffc7f_... but never 0xfffffe0... (bit 41 set) -- gPM faulted at
// 0x40c00000080 for a list at 0xfffffe0c00000000.
//
// That mismatch is why the firmware faulted reading InitData (LOG.md 27.2):
// ESR 0x96000005 is a **level 1** translation fault, i.e. the very first table
// lookup missed, because the coprocessor's own page tables have no entry
// covering a VA that far up. Everything else about the mapping was correct --
// it was simply at an address the firmware does not map.
//
// TODO: read these from the DT instead of hardcoding the t6030 values.
/// Driver-managed kernel base VA
const IOVA_KERN_BASE: u64 = 0xfffffc2000000000;
/// Driver-managed kernel top VA
const IOVA_KERN_TOP: u64 = 0xfffffc3000000000;
/// Driver-managed kernel VA range
const IOVA_KERN_RANGE: Range<u64> = IOVA_KERN_BASE..IOVA_KERN_TOP;
/// Full kernel VA range
#[cfg(CONFIG_DEV_COREDUMP)]
const IOVA_KERN_FULL_RANGE: Range<u64> = IOVA_TTBR1_BASE..(!UAT_PGMSK as u64);

const TTBR_VALID: u64 = 0x1; // BIT(0)
const TTBR_ASID_SHIFT: usize = 48;

/// Address of a special dummy page?
//const IOVA_UNK_PAGE: u64 = 0x6f_ffff8000;
pub(crate) const IOVA_UNK_PAGE: u64 = IOVA_USER_TOP - 2 * UAT_PGSZ as u64;
/// User VA range excluding the unk page
pub(crate) const IOVA_USER_USABLE_RANGE: Range<u64> = IOVA_USER_BASE..IOVA_UNK_PAGE;

/// Print where a faulting GPU VA sits relative to the driver's address ranges.
///
/// The BIF fault report gives a 48-bit address but the UAT's input address size
/// is `UAT_IAS` bits, so anything above that is out of range for the walker and
/// comes back as `NoAccess` at level 0 -- indistinguishable at a glance from a
/// protection failure on a mapped page. LOG.md 32.41.62.
pub(crate) fn describe_iova(dev: &driver::AsahiDevice, addr: u64) {
    let ias = 1u64 << UAT_IAS;

    // Bit `UAT_IAS` of the reported address is the TTBR select, not part of the
    // VA: with it set the walker used ttb1 and the VA is the low `UAT_IAS` bits
    // sign-extended into the kernel half. Classifying the raw value instead is
    // what made 0x60000009680 look like out-of-range garbage for four runs; it
    // is 0xfffffe0000009680. LOG.md 32.41.70.
    let (va, ttbr1) = if addr & ias != 0 {
        // Sign-extend the UAT_IAS-bit VA into 64 bits, exactly as the walker
        // does for the upper half.
        let low = addr & (ias - 1);
        let sh = 64 - UAT_IAS;
        ((((low << sh) as i64) >> sh) as u64, true)
    } else {
        (addr, false)
    };

    let where_ = if addr >= (ias << 1) {
        "ABOVE the fault register's address field -- not a mapping"
    } else if va == 0 {
        "null"
    } else if IOVA_USER_RANGE.contains(&va) {
        "user / kernel-lower range"
    } else if IOVA_KERN_RANGE.contains(&va) {
        "driver-managed kernel range"
    } else {
        "in IAS but outside every range the driver manages"
    };
    dev_err!(
        dev.as_ref(),
        "  IOVA {:#x}: ttbr{} VA {:#x}: {} (IAS {:#x}, user {:#x?}, kern {:#x?})\n",
        addr,
        if ttbr1 { 1 } else { 0 },
        va,
        where_,
        ias,
        IOVA_USER_RANGE,
        IOVA_KERN_RANGE,
    );
}

/// A pre-allocated memory region for UAT management
struct UatRegion {
    base: PhysicalAddr,
    map: io::mem::Mem,
}

/// SAFETY: It's safe to share UAT region records across threads.
unsafe impl Send for UatRegion {}
/// SAFETY: It's safe to share UAT region records across threads.
unsafe impl Sync for UatRegion {}

/// Handoff region flush info structure
#[repr(C)]
struct FlushInfo {
    state: AtomicU64,
    addr: AtomicU64,
    size: AtomicU64,
}

/// UAT Handoff region layout
#[repr(C)]
struct Handoff {
    magic_ap: AtomicU64,
    magic_fw: AtomicU64,

    lock_ap: AtomicU8,
    lock_fw: AtomicU8,
    // Implicit padding: 2 bytes
    turn: AtomicU32,
    cur_slot: AtomicU32,
    // Implicit padding: 4 bytes
    flush: [FlushInfo; UAT_NUM_CTX + 1],

    unk2: AtomicU8,
    // Implicit padding: 7 bytes
    unk3: AtomicU64,
}

const HANDOFF_SIZE: usize = size_of::<Handoff>();

/// One VM slot in the TTBAT
#[repr(C)]
struct SlotTTBS {
    ttb0: AtomicU64,
    ttb1: AtomicU64,
}

const SLOTS_SIZE: usize = UAT_NUM_CTX * size_of::<SlotTTBS>();

// We need at least page 0 (ttb0)
const PAGETABLES_SIZE: usize = UAT_PGSZ;

/// Inner data for a Vm instance. This is reference-counted by the outer Vm object.
struct VmInner {
    dev: driver::AsahiDevRef,
    is_kernel: bool,
    /// Added to every IOVA before it reaches the page table; 0 for the kernel VM.
    iova_bias: u64,
    iova_bias2: u64,
    iova_bias3: u64,
    va_range: Range<u64>,
    page_table: UatPageTable,
    mm: mm::Allocator<(), KernelMappingInner>,
    uat_inner: Arc<UatInner>,
    binding: Arc<Mutex<VmBinding>>,
    id: u64,
}

/// Slot binding-related inner data for a Vm instance.
struct VmBinding {
    active_users: usize,
    /// `wait_idle` timed out once: do not wait again (667zzu).
    idle_wait_expired: bool,
    /// Work of this VM was abandoned by host recovery while the firmware may
    /// still run it: never unmap or free it. LOG.md 32.41.667zzz29.
    poisoned: bool,
    binding: Option<slotalloc::Guard<SlotInner>>,
    bind_token: Option<slotalloc::SlotToken>,
    ttb: u64,
}

struct VmBoInner {
    sgt: Option<shmem::SGTable<gem::AsahiObject>>,
    sg_vec: Option<KVVec<(usize, Range<usize>)>>,
}

/// Data associated with a VM <=> BO pairing
#[pin_data]
struct VmBo {
    #[pin]
    inner: Mutex<VmBoInner>,
}

impl gpuvm::DriverGpuVmBo for VmBo {
    fn new() -> impl PinInit<Self> {
        pin_init!(VmBo {
            inner <- new_mutex!(VmBoInner {
                sgt: None,
                sg_vec: None,
            }, "VmBinding"),
        })
    }
}

#[derive(Default)]
struct StepContext {
    new_va: Option<Pin<KBox<gpuvm::GpuVa<VmInner>>>>,
    prev_va: Option<Pin<KBox<gpuvm::GpuVa<VmInner>>>>,
    next_va: Option<Pin<KBox<gpuvm::GpuVa<VmInner>>>>,
    vm_bo: Option<ARef<gpuvm::GpuVmBo<VmInner>>>,
    prot: Prot,
}

impl gpuvm::DriverGpuVm for VmInner {
    type Driver = driver::AsahiDriver;
    type GpuVmBo = VmBo;
    type StepContext = StepContext;

    fn step_map(
        self: &mut gpuvm::UpdatingGpuVm<'_, Self>,
        op: &mut gpuvm::OpMap<Self>,
        ctx: &mut Self::StepContext,
    ) -> Result {
        let mut iova = op.addr();
        let mut left = op.range() as usize;
        let mut offset = op.offset() as usize;

        let bo = ctx.vm_bo.as_ref().expect("step_map with no BO");

        let one_page = op.flags().contains(gpuvm::GpuVaFlags::REPEAT);

        let mut do_map = |mut addr: usize, mut len: usize, offset: &mut usize| -> Result<bool> {
            if left == 0 {
                return Ok(false);
            }

            if *offset > 0 {
                let skip = len.min(*offset);
                addr += skip;
                len -= skip;
                *offset -= skip;
            }
            if len == 0 {
                return Ok(true);
            }
            assert!(*offset == 0);

            if one_page {
                len = left;
            } else {
                len = len.min(left);
            }

            mod_dev_dbg!(
                self.dev,
                "MMU: map: {:#x}:{:#x} -> {:#x} [OP={}]\n",
                addr,
                len,
                iova,
                one_page
            );

            self.page_table.map_pages(
                iova..(iova + len as u64),
                addr as PhysicalAddr,
                ctx.prot,
                one_page,
            )?;
            for bias in [self.iova_bias, self.iova_bias2, self.iova_bias3, win36_bias(iova), win36_bias2(iova)] {
                if bias != 0 {
                    self.page_table.map_pages(
                        (iova.wrapping_add(bias))..(iova.wrapping_add(bias) + len as u64),
                        addr as PhysicalAddr,
                        ctx.prot,
                        one_page,
                    )?;
                }
            }

            left -= len;
            iova += len as u64;
            Ok(true)
        };

        let guard = bo.inner().inner.lock();
        if let Some(sg_vec) = guard.sg_vec.as_ref() {
            let start_idx = sg_vec
                .binary_search_by(|range| {
                    if range.0 > offset {
                        cmp::Ordering::Greater
                    } else if (range.0 + range.1.len()) <= offset {
                        cmp::Ordering::Less
                    } else {
                        cmp::Ordering::Equal
                    }
                })
                .expect("sg_vec does not contain offset???");

            offset -= sg_vec[start_idx].0 as usize;

            for cur in start_idx..sg_vec.len() {
                let addr = sg_vec[cur].1.start as usize;
                let len: usize = sg_vec[cur].1.len() as usize;
                if do_map(addr, len, &mut offset)? == false {
                    break;
                }
            }
        } else {
            for range in guard.sgt.as_ref().expect("step_map with no SGT").iter() {
                // TODO: proper DMA address/length handling
                let addr = range.dma_address() as usize;
                let len: usize = range.dma_len() as usize;
                if do_map(addr, len, &mut offset)? == false {
                    break;
                }
            }
        }

        let gpuva = ctx.new_va.take().expect("Multiple step_map calls");

        if op
            .map_and_link_va(
                self,
                gpuva,
                ctx.vm_bo.as_ref().expect("step_map with no BO"),
            )
            .is_err()
        {
            dev_err!(
                self.dev.as_ref(),
                "map_and_link_va failed: {:#x} [{:#x}] -> {:#x}\n",
                op.offset(),
                op.range(),
                op.addr()
            );
            return Err(EINVAL);
        }
        Ok(())
    }
    fn step_unmap(
        self: &mut gpuvm::UpdatingGpuVm<'_, Self>,
        op: &mut gpuvm::OpUnMap<Self>,
        _ctx: &mut Self::StepContext,
    ) -> Result {
        let va = op.va().expect("step_unmap: missing VA");

        mod_dev_dbg!(self.dev, "MMU: unmap: {:#x}:{:#x}\n", va.addr(), va.range());

        self.page_table
            .unmap_pages(va.addr()..(va.addr() + va.range()))?;
        // The alias has to come out too, not just be invalidated: leaving it
        // mapped leaks a page table per BO, which shows up as a wall of
        // `UATPageTable::drop: Leaked page at IOVA ...` at module unload, every
        // one of them exactly `iova + bias`. LOG.md 32.41.64.
        for bias in [self.iova_bias, self.iova_bias2, self.iova_bias3, win36_bias(va.addr()), win36_bias2(va.addr())] {
            if bias != 0 {
                self.page_table.unmap_pages(
                    (va.addr().wrapping_add(bias))..(va.addr().wrapping_add(bias) + va.range()),
                )?;
            }
        }

        if let Some(asid) = self.slot() {
            fence(Ordering::SeqCst);
            mem::tlbi_range(asid as u8, va.addr() as usize, va.range() as usize);
            for bias in [self.iova_bias, self.iova_bias2, self.iova_bias3, win36_bias(va.addr()), win36_bias2(va.addr())] {
                if bias != 0 {
                    mem::tlbi_range(
                        asid as u8,
                        va.addr().wrapping_add(bias) as usize,
                        va.range() as usize,
                    );
                }
            }
            mod_dev_dbg!(
                self.dev,
                "MMU: flush range: asid={:#x} start={:#x} len={:#x}\n",
                asid,
                va.addr(),
                va.range(),
            );
            mem::sync();
        }

        if op.unmap_and_unlink_va_defer().is_none() {
            dev_err!(self.dev.as_ref(), "step_unmap: could not unlink gpuva");
        }
        Ok(())
    }
    fn step_remap(
        self: &mut gpuvm::UpdatingGpuVm<'_, Self>,
        op: &mut gpuvm::OpReMap<Self>,
        vm_bo: &gpuvm::GpuVmBo<Self>,
        ctx: &mut Self::StepContext,
    ) -> Result {
        let va = op.unmap().va().expect("No previous VA");
        let orig_addr = va.addr();
        let orig_range = va.range();

        // Only unmap the hole between prev/next, if they exist
        let unmap_start = if let Some(op) = op.prev_map() {
            op.addr() + op.range()
        } else {
            orig_addr
        };

        let unmap_end = if let Some(op) = op.next_map() {
            op.addr()
        } else {
            orig_addr + orig_range
        };

        mod_dev_dbg!(
            self.dev,
            "MMU: unmap for remap: {:#x}..{:#x} (from {:#x}:{:#x})\n",
            unmap_start,
            unmap_end,
            orig_addr,
            orig_range
        );

        let unmap_range = unmap_end - unmap_start;

        self.page_table.unmap_pages(unmap_start..unmap_end)?;
        for bias in [
            self.iova_bias,
            self.iova_bias2,
            self.iova_bias3,
            win36_bias(unmap_start),
            win36_bias2(unmap_start),
        ] {
            if bias != 0 {
                self.page_table.unmap_pages(
                    (unmap_start.wrapping_add(bias))..(unmap_end.wrapping_add(bias)),
                )?;
            }
        }

        if let Some(asid) = self.slot() {
            fence(Ordering::SeqCst);
            mem::tlbi_range(asid as u8, unmap_start as usize, unmap_range as usize);
            for bias in [
                self.iova_bias,
                self.iova_bias2,
                self.iova_bias3,
                win36_bias(unmap_start),
                win36_bias2(unmap_start),
            win36_bias2(unmap_start),
            ] {
                if bias != 0 {
                    mem::tlbi_range(
                        asid as u8,
                        unmap_start.wrapping_add(bias) as usize,
                        unmap_range as usize,
                    );
                }
            }
            mod_dev_dbg!(
                self.dev,
                "MMU: flush range: asid={:#x} start={:#x} len={:#x}\n",
                asid,
                unmap_start,
                unmap_range,
            );
            mem::sync();
        }

        if op.unmap().unmap_and_unlink_va_defer().is_none() {
            dev_err!(self.dev.as_ref(), "step_unmap: could not unlink gpuva");
        }

        if let Some(prev_op) = op.prev_map() {
            let prev_gpuva = ctx
                .prev_va
                .take()
                .expect("Multiple step_remap calls with prev_op");
            if prev_op.map_and_link_va(self, prev_gpuva, vm_bo).is_err() {
                dev_err!(self.dev.as_ref(), "step_remap: could not relink prev gpuva");
                return Err(EINVAL);
            }
        }

        if let Some(next_op) = op.next_map() {
            let next_gpuva = ctx
                .next_va
                .take()
                .expect("Multiple step_remap calls with next_op");
            if next_op.map_and_link_va(self, next_gpuva, vm_bo).is_err() {
                dev_err!(self.dev.as_ref(), "step_remap: could not relink next gpuva");
                return Err(EINVAL);
            }
        }

        Ok(())
    }
}

impl VmInner {
    /// Returns the slot index, if this VM is bound.
    fn slot(&self) -> Option<u32> {
        if self.is_kernel {
            // The GFX ASC does not care about the ASID. Pick an arbitrary one.
            // TODO: This needs to be a persistently reserved ASID once we integrate
            // with the ARM64 kernel ASID machinery to avoid overlap.
            Some(0)
        } else {
            // We don't check whether we lost the slot, which could cause unnecessary
            // invalidations against another Vm. However, this situation should be very
            // rare (e.g. a Vm lost its slot, which means 63 other Vms bound in the
            // interim, and then it gets killed / drops its mappings without doing any
            // final rendering). Anything doing active maps/unmaps is probably also
            // rendering and therefore likely bound.
            self.binding
                .lock()
                .bind_token
                .as_ref()
                .map(|token| token.last_slot() + UAT_USER_CTX_START as u32)
        }
    }

    /// Returns the translation table base for this Vm
    fn ttb(&self) -> u64 {
        self.page_table.ttb()
    }

    /// Map an `mm::Node` representing an mapping in VA space.
    fn map_node(&mut self, node: &mm::Node<(), KernelMappingInner>, prot: Prot) -> Result {
        let mut iova = node.start();
        let guard = node.bo.as_ref().ok_or(EINVAL)?.inner().inner.lock();
        let sgt = guard.sgt.as_ref().ok_or(EINVAL)?;
        let mut offset = node.offset;
        let mut left = node.mapped_size;

        for range in sgt.iter() {
            if left == 0 {
                break;
            }

            // TODO: proper DMA address/length handling
            let mut addr = range.dma_address() as usize;
            let mut len: usize = range.dma_len() as usize;

            if (offset | addr | len | iova as usize) & UAT_PGMSK != 0 {
                dev_err!(
                    self.dev.as_ref(),
                    "MMU: KernelMapping {:#x}:{:#x} -> {:#x} is not page-aligned\n",
                    addr,
                    len,
                    iova
                );
                return Err(EINVAL);
            }

            if offset > 0 {
                let skip = len.min(offset);
                addr += skip;
                len -= skip;
                offset -= skip;
            }

            len = len.min(left);

            if len == 0 {
                continue;
            }

            mod_dev_dbg!(
                self.dev,
                "MMU: map: {:#x}:{:#x} -> {:#x}\n",
                addr,
                len,
                iova
            );

            self.page_table.map_pages(
                iova..(iova + len as u64),
                addr as PhysicalAddr,
                prot,
                false,
            )?;
            for bias in [self.iova_bias, self.iova_bias2, self.iova_bias3, win36_bias(iova), win36_bias2(iova)] {
                if bias != 0 {
                    self.page_table.map_pages(
                        (iova.wrapping_add(bias))..(iova.wrapping_add(bias) + len as u64),
                        addr as PhysicalAddr,
                        prot,
                        false,
                    )?;
                }
            }

            // replay_va0: Apple's draw capture has a buffer at GPU VA 0 that the
            // replay places at 2^36 (page 0 is outside the allocator). The fragment
            // SCS loads read it unbiased (UL1C fault at 0x40), so alias it at page 0
            // too. Diagnostic. LOG.md 32.41.605.
            if *module_parameters::replay_va0.value() != 0
                && iova >= (1u64 << 36)
                && iova < (1u64 << 36) + 0x10000
            {
                let lo = iova - (1u64 << 36);
                self.page_table.map_pages(lo..(lo + len as u64), addr as PhysicalAddr, prot, false)?;
                pr_info!("asahi/replay_va0: aliased {:#x} -> {:#x} ({:#x} bytes)\n", iova, lo, len);
            }

            iova += len as u64;
            left -= len;
        }
        Ok(())
    }
}

/// Shared reference to a virtual memory address space ([`Vm`]).
#[derive(Clone)]
pub(crate) struct Vm {
    id: u64,
    inner: ARef<gpuvm::GpuVm<VmInner>>,
    dummy_obj: ARef<gem::Object>,
    binding: Arc<Mutex<VmBinding>>,
}
no_debug!(Vm);

/// Slot data for a [`Vm`] slot (nothing, we only care about the indices).
pub(crate) struct SlotInner();

impl slotalloc::SlotItem for SlotInner {
    type Data = ();
}

/// Represents a single user of a binding of a [`Vm`] to a slot.
///
/// The number of users is counted, and the slot will be freed when it drops to 0.
#[derive(Debug)]
pub(crate) struct VmBind(Vm, u32);

impl VmBind {
    /// Returns the slot that this `Vm` is bound to.
    pub(crate) fn slot(&self) -> u32 {
        self.1
    }

    /// The `Vm` this binding is for. LOG.md 32.41.531 (apple_replay).
    pub(crate) fn vm(&self) -> &Vm {
        &self.0
    }
}

impl Drop for VmBind {
    fn drop(&mut self) {
        let mut binding = self.0.binding.lock();

        assert_ne!(binding.active_users, 0);
        binding.active_users -= 1;
        mod_pr_debug!(
            "MMU: slot {} active users {}\n",
            self.1,
            binding.active_users
        );
        if binding.active_users == 0 {
            binding.binding = None;
        }
    }
}

impl Clone for VmBind {
    fn clone(&self) -> VmBind {
        let mut binding = self.0.binding.lock();

        binding.active_users += 1;
        mod_pr_debug!(
            "MMU: slot {} active users {}\n",
            self.1,
            binding.active_users
        );
        VmBind(self.0.clone(), self.1)
    }
}

/// Inner data required for an object mapping into a [`Vm`].
pub(crate) struct KernelMappingInner {
    // Drop order matters:
    // - Drop the GpuVmBo first, which resv locks its BO and drops a GpuVm reference
    // - Drop the GEM BO next, since BO free can take the resv lock itself
    // - Drop the owner GpuVm last, since that again can take resv locks when the refcount drops to 0
    bo: Option<ARef<gpuvm::GpuVmBo<VmInner>>>,
    _gem: Option<ARef<gem::Object>>,
    owner: ARef<gpuvm::GpuVm<VmInner>>,
    uat_inner: Arc<UatInner>,
    prot: Prot,
    offset: usize,
    mapped_size: usize,
}

/// An object mapping into a [`Vm`], which reserves the address range from use by other mappings.
pub(crate) struct KernelMapping(mm::Node<(), KernelMappingInner>);

impl KernelMapping {
    /// Debug: the backing GEM object and the BO offset this mapping starts at.
    pub(crate) fn debug_gem(&self) -> Option<(ARef<gem::Object>, usize)> {
        self.0._gem.as_ref().map(|g| (g.clone(), self.0.offset))
    }

    /// Returns the IOVA base of this mapping
    pub(crate) fn iova(&self) -> u64 {
        self.0.start()
    }

    /// Returns the size of this mapping in bytes
    pub(crate) fn size(&self) -> usize {
        self.0.mapped_size
    }

    /// Returns the IOVA base of this mapping
    pub(crate) fn iova_range(&self) -> Range<u64> {
        self.0.start()..(self.0.start() + self.0.mapped_size as u64)
    }

    /// Remap a cached mapping as uncached, then synchronously flush that range of VAs from the
    /// coprocessor cache. This is required to safely unmap cached/private mappings.
    fn remap_uncached_and_flush(&mut self) {
        let mut owner = self
            .0
            .owner
            .exec_lock(None, false)
            .expect("Failed to exec_lock in remap_uncached_and_flush");

        mod_dev_dbg!(
            owner.dev,
            "MMU: remap as uncached {:#x}:{:#x}\n",
            self.iova(),
            self.size()
        );

        // Remap in-place as uncached.
        // Do not try to unmap the guard page (-1)
        let prot = self.0.prot.as_uncached();
        if owner
            .page_table
            .reprot_pages(self.iova_range(), prot)
            .is_err()
        {
            dev_err!(
                owner.dev.as_ref(),
                "MMU: remap {:#x}:{:#x} failed\n",
                self.iova(),
                self.size()
            );
        }
        fence(Ordering::SeqCst);

        // If we don't have (and have never had) a VM slot, just return
        let slot = match owner.slot() {
            None => return,
            Some(slot) => slot,
        };

        let flush_slot = if owner.is_kernel {
            // If this is a kernel mapping, always flush on index 64
            UAT_NUM_CTX as u32
        } else {
            // Otherwise, check if this slot is the active one, otherwise return
            // Also check that we actually own this slot
            let ttb = owner.ttb() | TTBR_VALID | (slot as u64) << TTBR_ASID_SHIFT;

            let uat_inner = self.0.uat_inner.lock();
            uat_inner.handoff().lock();
            let cur_slot = uat_inner.handoff().current_slot();
            let ttb_cur = uat_inner.ttbs()[slot as usize].ttb0.load(Ordering::Relaxed);
            uat_inner.handoff().unlock();
            if cur_slot == Some(slot) && ttb_cur == ttb {
                slot
            } else {
                return;
            }
        };

        // FIXME: There is a race here, though it'll probably never happen in practice.
        // In theory, it's possible for the ASC to finish using our slot, whatever command
        // it was processing to complete, the slot to be lost to another context, and the ASC
        // to begin using it again with a different page table, thus faulting when it gets a
        // flush request here. In practice, the chance of this happening is probably vanishingly
        // small, as all 62 other slots would have to be recycled or in use before that slot can
        // be reused, and the ASC using user contexts at all is very rare.

        // Still, the locking around UAT/Handoff/TTBs should probably be redesigned to better
        // model the interactions with the firmware and avoid these races.
        // Possibly TTB changes should be tied to slot locks:

        // Flush:
        //  - Can early check handoff here (no need to lock).
        //      If user slot and it doesn't match the active ASC slot,
        //      we can elide the flush as the ASC guarantees it flushes
        //      TLBs/caches when it switches context. We just need a
        //      barrier to ensure ordering.
        //  - Lock TTB slot
        //      - If user ctx:
        //          - Lock handoff AP-side
        //              - Lock handoff dekker
        //                  - Check TTB & handoff cur ctx
        //      - Perform flush if necessary
        //          - This implies taking the fwring lock
        //
        // TTB change:
        //  - lock TTB slot
        //      - lock handoff AP-side
        //          - lock handoff dekker
        //              change TTB

        // Lock this flush slot, and write the range to it
        let flush = self.0.uat_inner.lock_flush(flush_slot);
        let pages = self.size() >> UAT_PGBIT;
        flush.begin_flush(self.iova(), self.size() as u64);
        if pages >= 0x10000 {
            dev_err!(
                owner.dev.as_ref(),
                "MMU: Flush too big ({:#x} pages))\n",
                pages
            );
        }

        let cmd = fw::channels::FwCtlMsg {
            addr: fw::types::U64(self.iova()),
            unk_8: 0,
            slot: flush_slot,
            page_count: pages as u16,
            unk_12: 2, // ?
            __pad_15_5: 0,
        };

        // Tell the firmware to do a cache flush
        if let Err(e) = (*owner.dev).gpu.fwctl(cmd) {
            dev_err!(
                owner.dev.as_ref(),
                "MMU: ASC cache flush {:#x}:{:#x} failed (err: {:?})\n",
                self.iova(),
                self.size(),
                e
            );
        }

        // Finish the flush
        flush.end_flush();

        // Slot is unlocked here
    }
}
no_debug!(KernelMapping);

impl Drop for KernelMapping {
    fn drop(&mut self) {
        // This is the main unmap function for UAT mappings.
        // The sequence of operations here is finicky, due to the interaction
        // between cached GFX ASC mappings and the page tables. These mappings
        // always have to be flushed from the cache before being unmapped.

        // For uncached mappings, just unmapping and flushing the TLB is sufficient.

        // For cached mappings, this is the required sequence:
        // 1. Remap it as uncached
        // 2. Flush the TLB range
        // 3. If kernel VA mapping OR user VA mapping and handoff.current_slot() == slot:
        //    a. Take a lock for this slot
        //    b. Write the flush range to the right context slot in handoff area
        //    c. Issue a cache invalidation request via FwCtl queue
        //    d. Poll for completion via queue
        //    e. Check for completion flag in the handoff area
        //    f. Drop the lock
        // 4. Unmap
        // 5. Flush the TLB range again

        if self.0.prot.is_cached_noncoherent() {
            mod_pr_debug!(
                "MMU: remap as uncached {:#x}:{:#x}\n",
                self.iova(),
                self.size()
            );
            self.remap_uncached_and_flush();
        }

        let mut owner = self
            .0
            .owner
            .exec_lock(None, false)
            .expect("exec_lock failed in KernelMapping::drop");
        mod_dev_dbg!(
            owner.dev,
            "MMU: unmap {:#x}:{:#x}\n",
            self.iova(),
            self.size()
        );

        let bias = owner.iova_bias;
        if bias != 0 {
            let r = self.iova_range();
            let _ = owner
                .page_table
                .unmap_pages((r.start + bias)..(r.end + bias));
        }
        if owner.page_table.unmap_pages(self.iova_range()).is_err() {
            dev_err!(
                owner.dev.as_ref(),
                "MMU: unmap {:#x}:{:#x} failed\n",
                self.iova(),
                self.size()
            );
        }

        if let Some(asid) = owner.slot() {
            fence(Ordering::SeqCst);
            mem::tlbi_range(asid as u8, self.iova() as usize, self.size());
            if owner.iova_bias != 0 {
                mem::tlbi_range(
                    asid as u8,
                    (self.iova() + owner.iova_bias) as usize,
                    self.size(),
                );
            }
            mod_dev_dbg!(
                owner.dev,
                "MMU: flush range: asid={:#x} start={:#x} len={:#x}\n",
                asid,
                self.iova(),
                self.size()
            );
            mem::sync();
        }
    }
}

/// Shared UAT global data structures
struct UatShared {
    kernel_ttb1: u64,
    map_kernel_to_user: bool,
    handoff_rgn: UatRegion,
    ttbs_rgn: UatRegion,
}

impl UatShared {
    /// Returns the handoff region area
    fn handoff(&self) -> &Handoff {
        // SAFETY: pointer is non-null per the type invariant
        unsafe { (self.handoff_rgn.map.ptr() as *mut Handoff).as_ref() }.unwrap()
    }

    /// Returns the TTBAT area
    fn ttbs(&self) -> &[SlotTTBS; UAT_NUM_CTX] {
        // SAFETY: pointer is non-null per the type invariant
        unsafe { (self.ttbs_rgn.map.ptr() as *mut [SlotTTBS; UAT_NUM_CTX]).as_ref() }.unwrap()
    }
}

// SAFETY: Nothing here is unsafe to send across threads.
unsafe impl Send for UatShared {}

/// Inner data for the top-level UAT instance.
#[pin_data]
struct UatInner {
    #[pin]
    shared: Mutex<UatShared>,
    #[pin]
    handoff_flush: [Mutex<HandoffFlush>; UAT_NUM_CTX + 1],
}

impl UatInner {
    /// Take the lock on the shared data and return the guard.
    fn lock(&self) -> Guard<'_, UatShared, MutexBackend> {
        self.shared.lock()
    }

    /// Take a lock on a handoff flush slot and return the guard.
    fn lock_flush(&self, slot: u32) -> Guard<'_, HandoffFlush, MutexBackend> {
        self.handoff_flush[slot as usize].lock()
    }
}

/// Top-level UAT manager object
pub(crate) struct Uat {
    dev: driver::AsahiDevRef,
    cfg: &'static hw::HwConfig,

    inner: Arc<UatInner>,
    slots: slotalloc::SlotAllocator<SlotInner>,

    kernel_vm: Vm,
    kernel_lower_vm: Vm,
}

impl Handoff {
    /// Lock the handoff region from firmware access
    fn lock(&self) {
        self.lock_ap.store(1, Ordering::Relaxed);
        fence(Ordering::SeqCst);

        while self.lock_fw.load(Ordering::Relaxed) != 0 {
            if self.turn.load(Ordering::Relaxed) != 0 {
                self.lock_ap.store(0, Ordering::Relaxed);
                while self.turn.load(Ordering::Relaxed) != 0 {}
                self.lock_ap.store(1, Ordering::Relaxed);
                fence(Ordering::SeqCst);
            }
        }
        fence(Ordering::Acquire);
    }

    /// Unlock the handoff region, allowing firmware access
    fn unlock(&self) {
        self.turn.store(1, Ordering::Relaxed);
        self.lock_ap.store(0, Ordering::Release);
    }

    /// Returns the current Vm slot mapped by the firmware for lower/unprivileged access, if any.
    fn current_slot(&self) -> Option<u32> {
        let slot = self.cur_slot.load(Ordering::Relaxed);
        if slot == 0 || slot == u32::MAX {
            None
        } else {
            Some(slot)
        }
    }

    /// Initialize the handoff region
    fn init(&self) -> Result {
        self.magic_ap.store(PPL_MAGIC, Ordering::Relaxed);
        self.cur_slot.store(0, Ordering::Relaxed);
        self.unk3.store(0, Ordering::Relaxed);
        fence(Ordering::SeqCst);

        let start = Instant::<Monotonic>::now();
        const TIMEOUT: Delta = Delta::from_millis(1000);

        self.lock();
        while start.elapsed() < TIMEOUT {
            if self.magic_fw.load(Ordering::Relaxed) == PPL_MAGIC {
                break;
            } else {
                self.unlock();
                fsleep(Delta::from_millis(10));
                self.lock();
            }
        }

        if self.magic_fw.load(Ordering::Relaxed) != PPL_MAGIC {
            self.unlock();
            pr_err!("Handoff: Failed to initialize (firmware not running?)\n");
            return Err(EIO);
        }

        self.unlock();

        for i in 0..=UAT_NUM_CTX {
            self.flush[i].state.store(0, Ordering::Relaxed);
            self.flush[i].addr.store(0, Ordering::Relaxed);
            self.flush[i].size.store(0, Ordering::Relaxed);
        }
        fence(Ordering::SeqCst);
        Ok(())
    }
}

/// Represents a single flush info slot in the handoff region.
///
/// # Invariants
/// The pointer is valid and there is no aliasing HandoffFlush instance.
struct HandoffFlush(*const FlushInfo);

// SAFETY: These pointers are safe to send across threads.
unsafe impl Send for HandoffFlush {}

impl HandoffFlush {
    /// Set up a flush operation for the coprocessor
    fn begin_flush(&self, start: u64, size: u64) {
        // SAFETY: Per the type invariant, this is safe
        let flush = unsafe { self.0.as_ref().unwrap() };

        let state = flush.state.load(Ordering::Relaxed);
        if state != 0 {
            pr_err!("Handoff: expected flush state 0, got {}\n", state);
        }
        flush.addr.store(start, Ordering::Relaxed);
        flush.size.store(size, Ordering::Relaxed);
        flush.state.store(1, Ordering::Relaxed);
    }

    /// Complete a flush operation for the coprocessor
    fn end_flush(&self) {
        // SAFETY: Per the type invariant, this is safe
        let flush = unsafe { self.0.as_ref().unwrap() };
        let state = flush.state.load(Ordering::Relaxed);
        if state != 2 {
            pr_err!("Handoff: expected flush state 2, got {}\n", state);
        }
        flush.state.store(0, Ordering::Relaxed);
    }
}

impl Vm {
    /// Create a new virtual memory address space
    fn new(
        dev: &driver::AsahiDevice,
        uat_inner: Arc<UatInner>,
        kernel_range: Range<u64>,
        cfg: &'static hw::HwConfig,
        ttb: Option<PhysicalAddr>,
        id: u64,
    ) -> Result<Vm> {
        let dummy_obj = gem::new_kernel_object(dev, UAT_PGSZ)?;
        let is_kernel = ttb.is_some();

        let mut page_table = if let Some(ttb) = ttb {
            UatPageTable::new_with_ttb(ttb, IOVA_KERN_RANGE, cfg.uat_oas)?
        } else {
            UatPageTable::new(cfg.uat_oas)?
        };

        // DIAGNOSTIC (LOG.md 32.41.45): once the VDM was clear, `gUPM` -- a USC
        // unit, per the kernelcache's `sAGXPerfCtr_G15_USC_GUPM` -- faulted
        // *reading* a flat VA 0, which no translation could have produced and
        // which pointing the sampler array at a real buffer did not move. Back
        // a zeroed scratch page onto IOVA 0 so the read succeeds and the next
        // blocker becomes visible. The page is deliberately leaked; this is not
        // a shipping behaviour, it hides exactly the class of bug it is here to
        // find.
        if !is_kernel && *module_parameters::map_null_page.value() != 0 {
            let page = kernel::page::Page::alloc_page(GFP_KERNEL | __GFP_ZERO)?;
            // map_null_page=2 fills the page with self-describing words instead
            // of zeros: word at byte offset `o` reads 0x3bad000000 | (o << 12),
            // an address that is never mapped.  If the faulting unit consumes
            // one of them as a pointer, the *next* fault address names the
            // offset it read.  LOG.md 32.41.54.
            if *module_parameters::map_null_page.value() >= 2 {
                page.with_page_mapped(|p| {
                    for o in (0..kernel::page::PAGE_SIZE).step_by(8) {
                        let v: u64 = 0x3bad000000u64 | ((o as u64) << 12);
                        // SAFETY: `p` is valid for PAGE_SIZE bytes and `o+8` is
                        // in bounds; the page is exclusively owned here.
                        unsafe { core::ptr::write_unaligned(p.add(o) as *mut u64, v) };
                    }
                });
            }
            let phys = kernel::page::Page::into_phys(page);
            // Every distinct alias, so the VDM's biased view of "null" is
            // backed too. A disabled bias is 0, which would otherwise re-map
            // base 0 a second time.
            let mut done = [u64::MAX; 4];
            for (i, base) in [0u64, user_iova_bias(), user_iova_bias2(), user_iova_bias3()]
                .into_iter()
                .enumerate()
            {
                if done.contains(&base) {
                    continue;
                }
                done[i] = base;
                let r = base..(base + UAT_PGSZ as u64);
                page_table.alloc_pages(r.clone())?;
                page_table.map_pages(r, phys, PROT_GPU_SHARED_RW, false)?;
            }
        }

        // DIAGNOSTIC: the same trick as `map_null_page`, at an arbitrary VA.
        // The VDM reads `0x1500000000` at offset 0 -- the start of a 4 GiB USC
        // segment that nothing maps and that no shader change moves. Back it
        // with a zeroed scratch range so the read succeeds and the next blocker
        // becomes visible. Deliberately leaked, never a shipping behaviour.
        // LOG.md 32.41.196
        for scratch_va in [
            *module_parameters::scratch_page_va.value(),
            *module_parameters::scratch_page_va2.value(),
            // Third slot. Added to satisfy the `gVDM_SSD_SP` read at
            // `0x2800001580` *narrowly* -- the blanket `VA - 2^36` alias that
            // was tried instead is a regression: it stops the firmware reaching
            // job execution 2 runs in 3 (32.41.341).
            *module_parameters::scratch_page_va3.value(),
        ]
        .into_iter()
        .enumerate()
        {
        let is_slot3 = scratch_va.0 == 2;
        let scratch_va = scratch_va.1;
        if !is_kernel && scratch_va != 0 {
            let page = kernel::page::Page::alloc_page(GFP_KERNEL | __GFP_ZERO)?;
            // 🔑 `scratch_page_word`: fill slot 3 only, with one repeating
            // 64-bit word. Slots 1 and 2 back reads that want zeros, so the
            // fill has to be scoped or it breaks them.
            //
            // Built for one experiment: the surviving fault is `dVDM_CSD`
            // reading `cs_alias + 0x80080` (unit 0xa7, level 0 -- a *different*
            // VDM instance from the `gVDM_CSD_SP` runaway 32.41.321 retired).
            // Mapping that page full of a VDM Stream Link back to Mesa's Index
            // List at `cs+0x34` means whatever 8-byte-aligned offset the unit
            // lands on, it jumps to the draw. Repeating the link every 8 bytes
            // is what makes it robust to not knowing the landing offset.
            // LOG.md 32.41.377.
            let word = *module_parameters::scratch_page_word.value();
            if is_slot3 && word != 0 {
                page.with_page_mapped(|p| {
                    for o in (0..kernel::page::PAGE_SIZE).step_by(8) {
                        // SAFETY: `p` is valid for PAGE_SIZE bytes, `o+8` is in
                        // bounds, and the page is exclusively owned here.
                        unsafe { core::ptr::write_unaligned(p.add(o) as *mut u64, word) };
                    }
                });
            }
            // scratch_page_fill=1: self-describing words, same trick as
            // map_null_page=2. Word at byte offset `o` reads
            // 0x3bad000000 | (o << 12), an address that is never mapped, so if
            // the unit consumes one as a pointer the *next* fault names the
            // offset it read. Zeros tell us nothing; these tell us the layout.
            // LOG.md 32.41.198
            if *module_parameters::scratch_page_fill.value() != 0 {
                page.with_page_mapped(|p| {
                    for o in (0..kernel::page::PAGE_SIZE).step_by(8) {
                        let v: u64 = 0x3bad000000u64 | ((o as u64) << 12);
                        // SAFETY: `p` is valid for PAGE_SIZE bytes and `o+8` is
                        // in bounds; the page is exclusively owned here.
                        unsafe { core::ptr::write_unaligned(p.add(o) as *mut u64, v) };
                    }
                });
            }
            let phys = kernel::page::Page::into_phys(page);
            let len = (*module_parameters::scratch_page_pages.value() as u64)
                * (UAT_PGSZ as u64);
            let r = scratch_va..(scratch_va + len);
            page_table.alloc_pages(r.clone())?;
            // one_page: the whole range aliases the single zeroed page.
            page_table.map_pages(r, phys, PROT_GPU_SHARED_RW, true)?;
        }
        }

        let (va_range, gpuvm_range) = if is_kernel {
            (IOVA_KERN_RANGE, kernel_range.clone())
        } else {
            (IOVA_USER_RANGE, IOVA_USER_USABLE_RANGE)
        };

        let mm = mm::Allocator::new(va_range.start, va_range.range(), ())?;

        let binding = Arc::pin_init(
            new_mutex!(
                VmBinding {
                    binding: None,
                    bind_token: None,
                    active_users: 0,
                    idle_wait_expired: false,
                    poisoned: false,
                    ttb: page_table.ttb(),
                },
                "VmBinding",
            ),
            GFP_KERNEL,
        )?;

        let binding_clone = binding.clone();
        Ok(Vm {
            id,
            dummy_obj: dummy_obj.gem.clone(),
            inner: gpuvm::GpuVm::new(
                c_str!("Asahi::GpuVm"),
                // TODO: should we using DRM_GPUVM_RESV_PROTECTED as well?
                drm_gpuvm_flags_DRM_GPUVM_IMMEDIATE_MODE,
                dev,
                dummy_obj.gem.clone(),
                gpuvm_range,
                kernel_range,
                init!(VmInner {
                    dev: dev.into(),
                    va_range,
                    is_kernel,
                    iova_bias: if *is_kernel { 0 } else { user_iova_bias() },
                    iova_bias2: if *is_kernel { 0 } else { user_iova_bias2() },
                    iova_bias3: if *is_kernel { 0 } else { user_iova_bias3() },
                    page_table,
                    mm,
                    uat_inner,
                    binding: binding_clone,
                    id,
                }),
            )?,
            binding,
        })
    }

    /// Get the translation table base for this Vm
    fn ttb(&self) -> u64 {
        self.binding.lock().ttb
    }

    /// Map a GEM object (using its `SGTable`) into this Vm at a free address in a given range.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn map_in_range(
        &self,
        gem: &gem::Object,
        object_range: Range<usize>,
        alignment: u64,
        range: Range<u64>,
        prot: Prot,
        guard: bool,
    ) -> Result<KernelMapping> {
        let size = object_range.range();
        let sgt = gem.owned_sg_table()?;
        let mut inner = self.inner.exec_lock(Some(gem), false)?;
        let vm_bo = self.inner.obtain_bo(gem)?;

        let mut vm_bo_guard = vm_bo.inner().inner.lock();
        if vm_bo_guard.sgt.is_none() {
            vm_bo_guard.sgt.replace(sgt);
        }
        core::mem::drop(vm_bo_guard);

        let uat_inner = inner.uat_inner.clone();
        let node = inner.mm.insert_node_in_range(
            KernelMappingInner {
                owner: self.inner.clone(),
                uat_inner,
                prot,
                bo: Some(vm_bo),
                _gem: Some(gem.into()),
                offset: object_range.start,
                mapped_size: size,
            },
            (size + if guard { UAT_PGSZ } else { 0 }) as u64, // Add guard page
            alignment,
            0,
            range.start,
            range.end,
            mm::InsertMode::Best,
        )?;

        let ret = inner.map_node(&node, prot);
        // Drop the exec_lock first, so that if map_node failed the
        // KernelMappingInner destructur does not deadlock.
        core::mem::drop(inner);
        ret?;
        Ok(KernelMapping(node))
    }

    /// Map a GEM object into this Vm at a specific address.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn map_at(
        &self,
        addr: u64,
        size: usize,
        gem: ARef<gem::Object>,
        prot: Prot,
        guard: bool,
    ) -> Result<KernelMapping> {
        let sgt = gem.owned_sg_table()?;
        let mut inner = self.inner.exec_lock(Some(&gem), false)?;

        let vm_bo = self.inner.obtain_bo(&gem)?;

        let mut vm_bo_guard = vm_bo.inner().inner.lock();
        if vm_bo_guard.sgt.is_none() {
            vm_bo_guard.sgt.replace(sgt);
        }
        core::mem::drop(vm_bo_guard);

        let uat_inner = inner.uat_inner.clone();
        let node = inner.mm.reserve_node(
            KernelMappingInner {
                owner: self.inner.clone(),
                uat_inner,
                prot,
                bo: Some(vm_bo),
                _gem: Some(gem.clone()),
                offset: 0,
                mapped_size: size,
            },
            addr,
            (size + if guard { UAT_PGSZ } else { 0 }) as u64, // Add guard page
            0,
        )?;

        let ret = inner.map_node(&node, prot);
        // Drop the exec_lock first, so that if map_node failed the
        // KernelMappingInner destructur does not deadlock.
        core::mem::drop(inner);
        ret?;
        Ok(KernelMapping(node))
    }

    /// Map a range of a GEM object into this Vm using GPUVM.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_object(
        &self,
        gem: &gem::Object,
        addr: u64,
        size: u64,
        offset: u64,
        prot: Prot,
        single_page: bool,
    ) -> Result {
        // Mapping needs a complete context
        let mut ctx = StepContext {
            new_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            prev_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            next_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            prot,
            ..Default::default()
        };

        let vm_bo = self.inner.obtain_bo(gem)?;
        {
            let mut vm_bo_guard = vm_bo.inner().inner.lock();
            if vm_bo_guard.sgt.is_none() {
                let sgt = gem.owned_sg_table()?;

                if vm_bo_guard.sg_vec.is_none() {
                    let mut sg_vec = KVVec::new();
                    let mut offset = 0;
                    for range in sgt.iter() {
                        let addr = range.dma_address() as usize;
                        let len = range.dma_len() as usize;
                        sg_vec.push((offset, addr..(addr + len)), GFP_KERNEL)?;
                        offset += len;
                    }
                    vm_bo_guard.sg_vec.replace(sg_vec);
                }
                vm_bo_guard.sgt.replace(sgt);
            }
            core::mem::drop(vm_bo_guard);
        }

        let mut inner = self.inner.exec_lock(Some(gem), true)?;

        // Preallocate the page tables, to fail early if we ENOMEM
        let bias = inner.iova_bias;
        inner.page_table.alloc_pages(addr..(addr + size))?;
        if bias != 0 {
            inner
                .page_table
                .alloc_pages((addr + bias)..(addr + bias + size))?;
        }

        ctx.vm_bo = Some(vm_bo);

        if (addr | size | offset) & (UAT_PGMSK as u64) != 0 {
            dev_err!(
                inner.dev.as_ref(),
                "MMU: Map step {:#x} [{:#x}] -> {:#x} is not page-aligned\n",
                offset,
                size,
                addr
            );
            return Err(EINVAL);
        }

        let (flags, gem_range) = if single_page {
            (gpuvm::GpuVaFlags::REPEAT, UAT_PGSZ as u32)
        } else {
            (gpuvm::GpuVaFlags::NONE, 0u32)
        };

        mod_dev_dbg!(
            inner.dev,
            "MMU: sm_map: {:#x} [{:#x}] -> {:#x}\n",
            offset,
            size,
            addr
        );
        inner.sm_map(&mut ctx, addr, size, offset, gem_range, flags)
    }

    /// Add a direct MMIO mapping to this Vm at a free address.
    pub(crate) fn map_io(
        &self,
        iova: u64,
        phys: usize,
        size: usize,
        prot: Prot,
    ) -> Result<KernelMapping> {
        let mut inner = self.inner.exec_lock(None, false)?;

        if (iova as usize | phys | size) & UAT_PGMSK != 0 {
            dev_err!(
                inner.dev.as_ref(),
                "MMU: KernelMapping {:#x}:{:#x} -> {:#x} is not page-aligned\n",
                phys,
                size,
                iova
            );
            return Err(EINVAL);
        }

        dev_info!(
            inner.dev.as_ref(),
            "MMU: IO map: {:#x}:{:#x} -> {:#x}\n",
            phys,
            size,
            iova
        );

        let uat_inner = inner.uat_inner.clone();
        let node = inner.mm.reserve_node(
            KernelMappingInner {
                owner: self.inner.clone(),
                uat_inner,
                prot,
                bo: None,
                _gem: None,
                offset: 0,
                mapped_size: size,
            },
            iova,
            size as u64,
            0,
        )?;

        let bias = inner.iova_bias;
        if bias != 0 {
            let _ = inner.page_table.map_pages(
                (iova + bias)..(iova + bias + size as u64),
                phys as PhysicalAddr,
                prot,
                false,
            );
        }
        let ret = inner.page_table.map_pages(
            iova..(iova + size as u64),
            phys as PhysicalAddr,
            prot,
            false,
        );
        // Drop the exec_lock first, so that if map_node failed the
        // KernelMappingInner destructur does not deadlock.
        core::mem::drop(inner);
        ret?;
        Ok(KernelMapping(node))
    }

    /// Unmap everything in an address range.
    pub(crate) fn unmap_range(&self, iova: u64, size: u64) -> Result {
        // Unmapping a range can only do a single split, so just preallocate
        // the prev and next GpuVas
        let mut ctx = StepContext {
            prev_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            next_va: Some(gpuvm::GpuVa::<VmInner>::new(pin_init::default())?),
            ..Default::default()
        };

        let mut inner = self.inner.exec_lock(None, false)?;

        mod_dev_dbg!(inner.dev, "MMU: sm_unmap: {:#x}:{:#x}\n", iova, size);
        inner.sm_unmap(&mut ctx, iova, size)
    }

    /// Drop mappings for a given bo.
    pub(crate) fn drop_mappings(&self, gem: &gem::Object) -> Result {
        // Removing whole mappings only does unmaps, so no preallocated VAs
        let mut ctx = Default::default();

        let inner = self.inner.exec_lock(Some(gem), false)?;

        if let Some(bo) = self.inner.find_bo(gem) {
            mod_dev_dbg!(inner.dev, "MMU: bo_unmap\n");
            self.inner.bo_unmap(&mut ctx, &bo)?;
            mod_dev_dbg!(inner.dev, "MMU: bo_unmap done\n");
            // We need to drop the exec_lock first, then the GpuVmBo since that will take the lock itself.
            core::mem::drop(inner);
            core::mem::drop(bo);
        }

        Ok(())
    }

    /// Returns the dummy GEM object used to hold the shared DMA reservation locks
    pub(crate) fn get_resv_obj(&self) -> ARef<gem::Object> {
        self.dummy_obj.clone()
    }

    /// Debug: walk this VM's page table for `iova` (leaf PTE + 8 data words).
    pub(crate) fn debug_probe(&self, iova: u64) -> Option<(u64, [u64; 8])> {
        let mut inner = self.inner.exec_lock(None, false).ok()?;
        inner.page_table.probe(iova)
    }

    /// Check whether an object is external to this GpuVm
    pub(crate) fn is_extobj(&self, gem: &gem::Object) -> bool {
        self.inner.is_extobj(gem)
    }

    /// Wait up to `timeout_ms` until no job holds a slot binding of this VM
    /// (every `QueueJob` keeps a `VmBind` until it is freed). Returns whether
    /// the VM went idle. LOG.md 32.41.667zzu.
    pub(crate) fn wait_idle(&self, timeout_ms: u32) -> bool {
        if self.binding.lock().poisoned {
            return false;
        }
        for _ in 0..timeout_ms {
            let b = self.binding.lock();
            if b.active_users == 0 {
                return true;
            }
            if b.idle_wait_expired || b.poisoned {
                return false;
            }
            core::mem::drop(b);
            kernel::time::delay::fsleep(kernel::time::Delta::from_millis(1));
        }
        let mut b = self.binding.lock();
        b.idle_wait_expired = b.active_users != 0;
        if b.idle_wait_expired {
            pr_warn!(
                "asahi: exiting process: VM still has {} jobs after {} ms, keeping its memory\n",
                b.active_users,
                timeout_ms
            );
        }
        !b.idle_wait_expired
    }

    /// Host recovery abandoned work of this VM that the firmware still holds:
    /// keep the VM, its page tables and every mapping alive for good (a
    /// deliberate leak, only after a GPU hang), so the stuck command never
    /// faults or reads freed memory. LOG.md 32.41.667zzz29.
    pub(crate) fn poison(&self) {
        let mut b = self.binding.lock();
        if b.poisoned {
            return;
        }
        b.poisoned = true;
        core::mem::drop(b);
        core::mem::forget(self.clone());
        pr_warn!("asahi: VM {} poisoned by a GPU hang: its memory is kept until reboot\n", self.id);
    }

    /// See `poison`.
    pub(crate) fn is_poisoned(&self) -> bool {
        self.binding.lock().poisoned
    }

    /// Check whether an object is external to this GpuVm
    pub(crate) fn bo_deferred_cleanup(&self) {
        self.inner.bo_deferred_cleanup()
    }
}

impl Drop for VmInner {
    fn drop(&mut self) {
        let mut binding = self.binding.lock();
        assert_eq!(binding.active_users, 0);

        mod_pr_debug!(
            "VmInner::Drop [{}]: bind_token={:?}\n",
            self.id,
            binding.bind_token
        );

        // Make sure this VM is not mapped to a TTB if it was
        if let Some(token) = binding.bind_token.take() {
            let idx = (token.last_slot() as usize) + UAT_USER_CTX_START;
            let ttb = self.ttb() | TTBR_VALID | (idx as u64) << TTBR_ASID_SHIFT;

            let uat_inner = self.uat_inner.lock();
            uat_inner.handoff().lock();
            let handoff_cur = uat_inner.handoff().current_slot();
            let ttb_cur = uat_inner.ttbs()[idx].ttb0.load(Ordering::SeqCst);
            let inval = ttb_cur == ttb;
            if inval {
                if handoff_cur == Some(idx as u32) {
                    pr_err!(
                        "VmInner::drop owning slot {}, but it is currently in use by the ASC?\n",
                        idx
                    );
                }
                uat_inner.ttbs()[idx].ttb0.store(0, Ordering::SeqCst);
                uat_inner.ttbs()[idx].ttb1.store(0, Ordering::SeqCst);
            }
            // Tear down the diagnostic mirrors too, so no TTBAT slot is left
            // pointing at page tables that are about to be freed.
            let mirror = *module_parameters::mirror_vm_slots.value() as usize;
            if mirror != 0 {
                for m in UAT_USER_CTX_START..=mirror.min(UAT_NUM_CTX - 1) {
                    uat_inner.ttbs()[m].ttb0.store(0, Ordering::SeqCst);
                    uat_inner.ttbs()[m].ttb1.store(0, Ordering::SeqCst);
                }
            }
            uat_inner.handoff().unlock();
            core::mem::drop(uat_inner);

            // In principle we dropped all the KernelMappings already, but we might as
            // well play it safe and invalidate the whole ASID.
            if inval {
                mod_pr_debug!(
                    "VmInner::Drop [{}]: need inval for ASID {:#x}\n",
                    self.id,
                    idx
                );
                mem::tlbi_asid(idx as u8);
                mem::sync();
            }
        }
    }
}

impl Uat {
    /// Map a bootloader-preallocated memory region
    fn map_region(
        dev: &device::Device,
        name: &CStr,
        size: usize,
        cached: bool,
    ) -> Result<UatRegion> {
        let of_node = dev.of_node().ok_or(EINVAL)?;
        let res = of_node.reserved_mem_region_to_resource_byname(name)?;
        let base = res.start();
        let res_size = res.size().try_into()?;

        if size > res_size {
            dev_err!(
                dev,
                "Region {} is too small (expected {}, got {})\n",
                name,
                size,
                res_size
            );
            return Err(ENOMEM);
        }

        let flags = if cached {
            io::mem::MemFlag::WB
        } else {
            io::mem::MemFlag::WC
        };

        // SAFETY: The safety of this operation hinges on the correctness of
        // much of this file and also the `pgtable` module, so it is difficult
        // to prove in a single safety comment. Such is life with raw GPU
        // page table management...
        let map = unsafe { io::mem::Mem::try_new(res, flags.into()) }.inspect_err(|_| {
            dev_err!(dev, "Failed to remap {} mem resource\n", name);
        })?;

        Ok(UatRegion { base, map })
    }

    /// Returns a reference to the global kernel (upper half) `Vm`
    pub(crate) fn kernel_vm(&self) -> &Vm {
        &self.kernel_vm
    }

    /// Returns a reference to the local kernel (lower half) `Vm`
    pub(crate) fn kernel_lower_vm(&self) -> &Vm {
        &self.kernel_lower_vm
    }

    /// Dump `count` dwords of a **kernel/firmware** address.
    ///
    /// `dump_slot`/`probe_slot` build their view with `IOVA_USER_RANGE`, so a
    /// firmware address (`0xfffffe...`) is outside the range they can walk and
    /// always reports "unreadable" -- which is why the `scene obj` probes have
    /// never resolved either. This goes through the kernel VM's own page table.
    /// LOG.md 32.41.337.
    pub(crate) fn dump_kern(&self, dev: &driver::AsahiDevice, va: u64, count: usize, tag: &str) {
        let mut inner = match self.kernel_vm.inner.exec_lock(None, false) {
            Ok(i) => i,
            Err(e) => {
                dev_err!(dev.as_ref(), "  kdump {} {:#x}: exec_lock failed: {:?}\n", tag, va, e);
                return;
            }
        };
        let mut buf = [0u32; 64];
        let n = core::cmp::min(count, 64);
        let got = inner.page_table.read_dwords(va, &mut buf[..n]);
        if got == 0 {
            dev_err!(dev.as_ref(), "  kdump {} {:#x}: unreadable\n", tag, va);
            return;
        }
        for row in 0..got.div_ceil(8) {
            let lo = row * 8;
            let hi = core::cmp::min(lo + 8, got);
            let mut line: [u32; 8] = [0; 8];
            line[..(hi - lo)].copy_from_slice(&buf[lo..hi]);
            dev_err!(dev.as_ref(),
                "  kdump {} +{:#05x}: {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
                tag, lo * 4,
                line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7]);
        }
    }

    #[cfg(CONFIG_DEV_COREDUMP)]
    /// Like `dump_kern`, for a larger range: walks `bytes` from `va` and prints
    /// only the 32-byte rows that are not all zero. LOG.md 32.41.667zzw.
    pub(crate) fn dump_kern_nz(&self, dev: &driver::AsahiDevice, va: u64, bytes: usize, tag: &str) {
        let mut inner = match self.kernel_vm.inner.exec_lock(None, false) {
            Ok(i) => i,
            Err(e) => {
                dev_err!(dev.as_ref(), "  kdump {} {:#x}: exec_lock failed: {:?}\n", tag, va, e);
                return;
            }
        };
        let mut buf = [0u32; 64];
        let mut off = 0usize;
        while off < bytes {
            let n = core::cmp::min(64, (bytes - off).div_ceil(4));
            let got = inner.page_table.read_dwords(va + off as u64, &mut buf[..n]);
            if got == 0 {
                dev_err!(dev.as_ref(), "  kdump {} {:#x}+{:#x}: unreadable\n", tag, va, off);
                return;
            }
            for row in 0..got.div_ceil(8) {
                let lo = row * 8;
                let hi = core::cmp::min(lo + 8, got);
                let mut line: [u32; 8] = [0; 8];
                line[..(hi - lo)].copy_from_slice(&buf[lo..hi]);
                if line.iter().all(|&w| w == 0) {
                    continue;
                }
                dev_err!(dev.as_ref(),
                    "  kdump {} +{:#05x}: {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
                    tag, off + lo * 4,
                    line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7]);
            }
            off += got * 4;
        }
    }

    pub(crate) fn dump_kernel_pages(&self) -> Result<KVVec<pgtable::DumpedPage>> {
        let mut inner = self.kernel_vm.inner.exec_lock(None, false)?;
        inner.page_table.dump_pages(IOVA_KERN_FULL_RANGE)
    }

    /// Dump the low TTBAT slots. Used from the fault path to tell "the driver
    /// bound the wrong slot" apart from "the entry the hardware walked is
    /// empty". LOG.md 32.41.39.
    pub(crate) fn dump_ttbs(&self, n: usize) {
        let inner = self.inner.lock();
        let ttbs = inner.ttbs();
        for i in 0..n.min(UAT_NUM_CTX) {
            let t0 = ttbs[i].ttb0.load(Ordering::Relaxed);
            let t1 = ttbs[i].ttb1.load(Ordering::Relaxed);
            if t0 != 0 || t1 != 0 || i < 4 {
                pr_err!("  TTBAT[{}] ttb0={:#x} ttb1={:#x}\n", i, t0, t1);
            }
        }
        pr_err!(
            "  TTBAT handoff current_slot={:?}\n",
            inner.handoff().current_slot()
        );
    }

    /// Probe a user IOVA in a given TTBAT slot and report what backs it.
    ///
    /// A fault report says "Unmapped" and stops there, which cannot tell a
    /// short alias mapping apart from an address userspace never allocated.
    /// This walks the slot's own ttb0 and prints the PTE plus the first words
    /// of the page, so the two cases are distinguishable in one capture.
    /// LOG.md 32.41.102.
    /// Write one 64-bit word at a user IOVA in a VM slot. `cs_poke_*` only.
    pub(crate) fn poke_slot(&self, dev: &driver::AsahiDevice, slot: usize, va: u64, val: u64) {
        let ttb0 = {
            let inner = self.inner.lock();
            if slot >= UAT_NUM_CTX {
                return;
            }
            inner.ttbs()[slot].ttb0.load(Ordering::Relaxed)
        };
        let ttb0 = ttb0 & ((1u64 << TTBR_ASID_SHIFT) - 1) & !(UAT_PGMSK as u64);
        if ttb0 == 0 {
            return;
        }
        let mut pt = match UatPageTable::new_view(ttb0, IOVA_USER_RANGE, self.cfg.uat_oas) {
            Ok(pt) => pt,
            Err(_) => return,
        };
        let ok = pt.poke(va, val);
        dev_err!(dev.as_ref(), "  poke {:#x} <- {:#018x}: {}\n", va, val,
                 if ok { "written" } else { "UNMAPPED" });
    }

    /// Fill a range of 32-bit words at a user IOVA in a VM slot. `cs_poke_*`.
    pub(crate) fn poke_fill_slot(
        &self,
        dev: &driver::AsahiDevice,
        slot: usize,
        va: u64,
        word: u32,
        count: usize,
    ) {
        let ttb0 = {
            let inner = self.inner.lock();
            if slot >= UAT_NUM_CTX {
                return;
            }
            inner.ttbs()[slot].ttb0.load(Ordering::Relaxed)
        };
        let ttb0 = ttb0 & ((1u64 << TTBR_ASID_SHIFT) - 1) & !(UAT_PGMSK as u64);
        if ttb0 == 0 {
            return;
        }
        let mut pt = match UatPageTable::new_view(ttb0, IOVA_USER_RANGE, self.cfg.uat_oas) {
            Ok(pt) => pt,
            Err(_) => return,
        };
        let ok = pt.poke_fill(va, word, count);
        dev_err!(dev.as_ref(), "  poke_fill {:#x}..{:#x} <- {:#010x} x{}: {}\n",
                 va, va + (count as u64) * 4, word, count,
                 if ok { "written" } else { "UNMAPPED" });
    }

    pub(crate) fn patch_slot(
        &self,
        dev: &driver::AsahiDevice,
        slot: usize,
        va: u64,
        count: usize,
        mask: u32,
        val: u32,
        or_val: u32,
        and_val: u32,
    ) {
        let ttb0 = {
            let inner = self.inner.lock();
            if slot >= UAT_NUM_CTX {
                return;
            }
            inner.ttbs()[slot].ttb0.load(Ordering::Relaxed)
        };
        let ttb0 = ttb0 & ((1u64 << TTBR_ASID_SHIFT) - 1) & !(UAT_PGMSK as u64);
        // ⚠️ Log the early returns.  A helper that is silent when it does
        // nothing is how 32.41.253's two runs were scored as if the fix had
        // applied when it had not.
        if ttb0 == 0 {
            dev_err!(dev.as_ref(), "  patch {:#x}: slot {} has no ttb0\n", va, slot);
            return;
        }
        let mut pt = match UatPageTable::new_view(ttb0, IOVA_USER_RANGE, self.cfg.uat_oas) {
            Ok(pt) => pt,
            Err(e) => {
                dev_err!(dev.as_ref(), "  patch {:#x}: new_with_ttb failed: {:?}\n", va, e);
                return;
            }
        };
        let n = pt.patch_dwords(va, count, mask, val, or_val, and_val);
        crate::sub_dev_err!(dev.as_ref(),
                 "  patch {:#x} x{}: (w & {:#x}) == {:#x} -> (w & {:#x}) | {:#x} : {} hit(s)\n",
                 va, count, mask, val, and_val, or_val, n);
    }

    /// Read `out.len()` dwords of a VM mapping into `out`.
    ///
    /// Non-printing twin of `dump_slot`, for when the driver needs the bytes
    /// rather than the operator. Used to copy the draw record before relocating
    /// it. LOG.md 32.41.305.
    pub(crate) fn read_slot(
        &self,
        dev: &driver::AsahiDevice,
        slot: usize,
        va: u64,
        out: &mut [u32],
    ) -> usize {
        let ttb0 = {
            let inner = self.inner.lock();
            if slot >= UAT_NUM_CTX {
                return 0;
            }
            inner.ttbs()[slot].ttb0.load(Ordering::Relaxed)
        };
        let ttb0 = ttb0 & ((1u64 << TTBR_ASID_SHIFT) - 1) & !(UAT_PGMSK as u64);
        if ttb0 == 0 {
            dev_err!(dev.as_ref(), "  read {:#x}: slot {} has no ttb0\n", va, slot);
            return 0;
        }
        let mut pt = match UatPageTable::new_view(ttb0, IOVA_USER_RANGE, self.cfg.uat_oas) {
            Ok(pt) => pt,
            Err(e) => {
                dev_err!(dev.as_ref(), "  read {:#x}: new_with_ttb failed: {:?}\n", va, e);
                return 0;
            }
        };
        pt.read_dwords(va, out)
    }

    /// Wrapper for `UatPageTable::fix_count_dwords`. LOG.md 32.41.330.
    pub(crate) fn fix_count_slot(
        &self,
        dev: &driver::AsahiDevice,
        slot: usize,
        va: u64,
        count: usize,
    ) {
        let ttb0 = {
            let inner = self.inner.lock();
            if slot >= UAT_NUM_CTX {
                return;
            }
            inner.ttbs()[slot].ttb0.load(Ordering::Relaxed)
        };
        let ttb0 = ttb0 & ((1u64 << TTBR_ASID_SHIFT) - 1) & !(UAT_PGMSK as u64);
        if ttb0 == 0 {
            dev_err!(dev.as_ref(), "  countfix {:#x}: slot {} has no ttb0\n", va, slot);
            return;
        }
        let mut pt = match UatPageTable::new_view(ttb0, IOVA_USER_RANGE, self.cfg.uat_oas) {
            Ok(pt) => pt,
            Err(e) => {
                dev_err!(dev.as_ref(), "  countfix {:#x}: new_with_ttb failed: {:?}\n", va, e);
                return;
            }
        };
        let n = pt.fix_count_dwords(va, count);
        dev_err!(dev.as_ref(), "  countfix {:#x} x{}: {} dword(s) patched\n", va, count, n);
    }

    /// Splice `writes` in relative to a dword matching `(w & mask) == val`.
    ///
    /// Wrapper for `UatPageTable::splice_dwords`, used to transcribe Apple's
    /// eight-dword draw record onto Mesa's four-dword one. Logs its early
    /// returns and its hit count for the same reason `patch_slot` does.
    /// LOG.md 32.41.299.
    pub(crate) fn splice_slot(
        &self,
        dev: &driver::AsahiDevice,
        slot: usize,
        va: u64,
        count: usize,
        mask: u32,
        val: u32,
        writes: &[(usize, u32)],
    ) {
        let ttb0 = {
            let inner = self.inner.lock();
            if slot >= UAT_NUM_CTX {
                return;
            }
            inner.ttbs()[slot].ttb0.load(Ordering::Relaxed)
        };
        let ttb0 = ttb0 & ((1u64 << TTBR_ASID_SHIFT) - 1) & !(UAT_PGMSK as u64);
        if ttb0 == 0 {
            dev_err!(dev.as_ref(), "  splice {:#x}: slot {} has no ttb0\n", va, slot);
            return;
        }
        let mut pt = match UatPageTable::new_view(ttb0, IOVA_USER_RANGE, self.cfg.uat_oas) {
            Ok(pt) => pt,
            Err(e) => {
                dev_err!(dev.as_ref(), "  splice {:#x}: new_with_ttb failed: {:?}\n", va, e);
                return;
            }
        };
        let n = pt.splice_dwords(va, count, mask, val, writes);
        dev_err!(dev.as_ref(),
                 "  splice {:#x} x{}: (w & {:#x}) == {:#x} -> {} write(s) : {} hit(s)\n",
                 va, count, mask, val, writes.len(), n);
    }

    /// Dump `count` dwords of a VM mapping, 8 per line, with byte offsets.
    ///
    /// The VDM control stream as the GPU actually sees it. `poke`/`fill` can
    /// write it but nothing could read it back, so every record-boundary
    /// question so far was answered by planting terminators and watching the
    /// fault move. LOG.md 32.41.298.
    pub(crate) fn dump_slot(
        &self,
        dev: &driver::AsahiDevice,
        slot: usize,
        va: u64,
        count: usize,
        tag: &str,
    ) {
        let ttb0 = {
            let inner = self.inner.lock();
            if slot >= UAT_NUM_CTX {
                return;
            }
            inner.ttbs()[slot].ttb0.load(Ordering::Relaxed)
        };
        let ttb0 = ttb0 & ((1u64 << TTBR_ASID_SHIFT) - 1) & !(UAT_PGMSK as u64);
        if ttb0 == 0 {
            dev_err!(dev.as_ref(), "  dump {} {:#x}: slot {} has no ttb0\n", tag, va, slot);
            return;
        }
        let mut pt = match UatPageTable::new_view(ttb0, IOVA_USER_RANGE, self.cfg.uat_oas) {
            Ok(pt) => pt,
            Err(e) => {
                dev_err!(dev.as_ref(), "  dump {} {:#x}: new_with_ttb failed: {:?}\n", tag, va, e);
                return;
            }
        };
        let mut buf = [0u32; 8];
        let mut done = 0usize;
        while done < count {
            let n = core::cmp::min(8, count - done);
            let got = pt.read_dwords(va + (done * 4) as u64, &mut buf[..n]);
            if got == 0 {
                dev_err!(dev.as_ref(), "  dump {} +{:#05x}: unreadable\n", tag, done * 4);
                return;
            }
            dev_err!(
                dev.as_ref(),
                "  dump {} +{:#05x}: {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
                tag, done * 4,
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            );
            done += n;
        }
    }

    pub(crate) fn probe_slot(&self, dev: &driver::AsahiDevice, slot: usize, va: u64, tag: &str) {
        let ttb0 = {
            let inner = self.inner.lock();
            if slot >= UAT_NUM_CTX {
                return;
            }
            inner.ttbs()[slot].ttb0.load(Ordering::Relaxed)
        };
        // A TTBAT entry is `phys | TTBR_VALID | asid << TTBR_ASID_SHIFT`, so
        // the ASID in bits [63:48] has to come off as well as the valid bit --
        // masking only the page offset leaves it in and `borrow_phys` then
        // rejects the address (EIO). LOG.md 32.41.102.
        let ttb0 = ttb0 & ((1u64 << TTBR_ASID_SHIFT) - 1) & !(UAT_PGMSK as u64);
        if ttb0 == 0 {
            dev_err!(dev.as_ref(), "  probe {} {:#x}: slot {} has no ttb0\n", tag, va, slot);
            return;
        }
        let mut pt = match UatPageTable::new_view(ttb0, IOVA_USER_RANGE, self.cfg.uat_oas) {
            Ok(pt) => pt,
            Err(e) => {
                dev_err!(dev.as_ref(), "  probe {} {:#x}: no page table ({:?})\n", tag, va, e);
                return;
            }
        };
        match pt.probe(va) {
            Some((pte, w)) => dev_err!(
                dev.as_ref(),
                "  probe {} {:#x}: MAPPED pte {:#x} data {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}\n",
                tag, va, pte, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
            ),
            None => {
                // Where does the mapping actually stop? Scanning back names the
                // last mapped page, which is the difference between "the alias
                // is short" and "userspace never allocated this far".
                let mut back = None;
                let page = UAT_PGSZ as u64;
                for i in 1..=64u64 {
                    let probe_va = va.saturating_sub(i * page) & !(page - 1);
                    if probe_va == 0 {
                        break;
                    }
                    if pt.probe(probe_va).is_some() {
                        back = Some(probe_va);
                        break;
                    }
                }
                match back {
                    Some(p) => dev_err!(
                        dev.as_ref(),
                        "  probe {} {:#x}: unmapped; last mapped page below is {:#x} ({:#x} lower)\n",
                        tag, va, p, va - p,
                    ),
                    None => dev_err!(
                        dev.as_ref(),
                        "  probe {} {:#x}: unmapped, and nothing mapped in the 1MiB below it\n",
                        tag, va,
                    ),
                }
            }
        }
        // ⚠️ `UatPageTable::drop` walks its whole `va_range` with `free = true`.
        // This table is borrowed from a live TTBAT slot, so letting it drop
        // would tear down userspace's page tables under it. Never drop it.
        core::mem::forget(pt);
    }

    /// Returns the base physical address of the TTBAT region.
    pub(crate) fn ttb_base(&self) -> u64 {
        let inner = self.inner.lock();

        inner.ttbs_rgn.base
    }

    /// Binds a `Vm` to a slot, preferring the last used one.
    pub(crate) fn bind(&self, vm: &Vm) -> Result<VmBind> {
        let mut binding = vm.binding.lock();

        if binding.binding.is_none() {
            assert_eq!(binding.active_users, 0);

            let isolation = *module_parameters::robust_isolation.value() != 0;

            self.slots.set_limit(if isolation {
                NonZeroUsize::new(1)
            } else {
                None
            });

            let slot = self.slots.get(binding.bind_token)?;
            if slot.changed() {
                mod_pr_debug!("Vm Bind [{}]: bind_token={:?}\n", vm.id, slot.token(),);
                let idx = (slot.slot() as usize) + UAT_USER_CTX_START;
                let ttb = binding.ttb | TTBR_VALID | (idx as u64) << TTBR_ASID_SHIFT;

                let uat_inner = self.inner.lock();

                let ttb1 = if uat_inner.map_kernel_to_user {
                    uat_inner.kernel_ttb1 | TTBR_VALID | (idx as u64) << TTBR_ASID_SHIFT
                } else {
                    0
                };

                let ttbs = uat_inner.ttbs();
                uat_inner.handoff().lock();
                if uat_inner.handoff().current_slot() == Some(idx as u32) {
                    pr_err!(
                        "Vm::bind to slot {}, but it is currently in use by the ASC?\n",
                        idx
                    );
                }
                ttbs[idx].ttb0.store(ttb, Ordering::Release);
                ttbs[idx].ttb1.store(ttb1, Ordering::Release);

                // DIAGNOSTIC (LOG.md 32.41.599): Apple's TA pool pages are
                // pushed with context 63. kctx63=2 puts the bound user tables
                // in TTBAT[63] (kernel TTB1 kept) so `uma_dc=65` can push our
                // user-VM pool pages under context 63 too.
                let k63 = *module_parameters::kctx63.value() == 2;
                if k63 {
                    let n = UAT_NUM_CTX - 1;
                    ttbs[n]
                        .ttb0
                        .store(binding.ttb | TTBR_VALID | (n as u64) << TTBR_ASID_SHIFT, Ordering::Release);
                    ttbs[n].ttb1.store(
                        uat_inner.kernel_ttb1 | TTBR_VALID | (n as u64) << TTBR_ASID_SHIFT,
                        Ordering::Release,
                    );
                }

                // DIAGNOSTIC (LOG.md 32.41.38/39): on 15.5 the hardware walked
                // context 3 for a job the driver had bound to TTBAT[1]. Nothing
                // the driver writes carries the index -- the microsequence's
                // `vm_slot` is gone on 15.5 -- so mirroring the same tables into
                // the low slots tells us whether `vm_slot` in the fault report
                // really is the TTBAT index, without changing anything we tell
                // the firmware. Off by default.
                let mirror = *module_parameters::mirror_vm_slots.value() as usize;
                if mirror != 0 {
                    for m in UAT_USER_CTX_START..=mirror.min(UAT_NUM_CTX - 1) {
                        if m == idx {
                            continue;
                        }
                        let mttb = binding.ttb | TTBR_VALID | (m as u64) << TTBR_ASID_SHIFT;
                        let mttb1 = if uat_inner.map_kernel_to_user {
                            uat_inner.kernel_ttb1 | TTBR_VALID | (m as u64) << TTBR_ASID_SHIFT
                        } else {
                            0
                        };
                        ttbs[m].ttb0.store(mttb, Ordering::Release);
                        ttbs[m].ttb1.store(mttb1, Ordering::Release);
                    }
                }

                uat_inner.handoff().unlock();
                core::mem::drop(uat_inner);

                // Make sure all TLB entries from the previous owner of this ASID are gone
                mem::tlbi_asid(idx as u8);
                if k63 {
                    mem::tlbi_asid((UAT_NUM_CTX - 1) as u8);
                }
                if mirror != 0 {
                    for m in UAT_USER_CTX_START..=mirror.min(UAT_NUM_CTX - 1) {
                        mem::tlbi_asid(m as u8);
                    }
                }
                mem::sync();
            }

            binding.bind_token = Some(slot.token());
            binding.binding = Some(slot);
        }

        binding.active_users += 1;

        let slot = binding.binding.as_ref().unwrap().slot() + UAT_USER_CTX_START as u32;
        mod_pr_debug!("MMU: slot {} active users {}\n", slot, binding.active_users);
        Ok(VmBind(vm.clone(), slot))
    }

    /// Creates a new `Vm` linked to this UAT.
    pub(crate) fn new_vm(&self, id: u64, kernel_range: Range<u64>) -> Result<Vm> {
        let vm = Vm::new(
            &self.dev,
            self.inner.clone(),
            kernel_range,
            self.cfg,
            None,
            id,
        )?;
        // 667zzj: every user VM sees the kernel-lower UMA tables (klow) at the same
        // VA, as macOS's process contexts see its 2^40 tables: gUPM walks the page
        // list and extents through the job's context.
        if *module_parameters::uma_klow.value() != 0 {
            let pte = self.kernel_lower_vm.inner.exec_lock(None, false)?.page_table.top_pte(KLOW_TOP_IDX)?;
            vm.inner.exec_lock(None, false)?.page_table.share_top(KLOW_TOP_IDX, pte)?;
        }
        if *module_parameters::pm_dualmap.value() != 0 {
            let pte = self.kernel_lower_vm.inner.exec_lock(None, false)?.page_table.top_pte(PMCT_TOP_IDX)?;
            vm.inner.exec_lock(None, false)?.page_table.share_top(PMCT_TOP_IDX, pte)?;
        }
        Ok(vm)
    }

    /// Creates the reference-counted inner data for a new `Uat` instance.
    #[inline(never)]
    fn make_inner(dev: &driver::AsahiDevice) -> Result<Arc<UatInner>> {
        let handoff_rgn = Self::map_region(dev.as_ref(), c_str!("handoff"), HANDOFF_SIZE, true)?;
        let ttbs_rgn = Self::map_region(dev.as_ref(), c_str!("ttbs"), SLOTS_SIZE, true)?;

        // SAFETY: The Handoff struct layout matches the firmware's view of memory at this address,
        // and the region is at least large enough per the size specified above.
        let handoff = unsafe { &(handoff_rgn.map.ptr() as *mut Handoff).as_ref().unwrap() };

        dev_info!(dev.as_ref(), "MMU: Initializing kernel page table\n");

        Arc::pin_init(
            try_pin_init!(UatInner {
                handoff_flush <- pin_init::pin_init_array_from_fn(|i| {
                    new_mutex!(HandoffFlush(&handoff.flush[i]), "handoff_flush")
                }),
                shared <- new_mutex!(
                    UatShared {
                        kernel_ttb1: 0,
                        map_kernel_to_user: false,
                        handoff_rgn,
                        ttbs_rgn,
                    },
                    "uat_shared"
                ),
            }),
            GFP_KERNEL,
        )
    }

    /// Creates a new `Uat` instance given the relevant hardware config.
    #[inline(never)]
    pub(crate) fn new(
        dev: &driver::AsahiDevice,
        cfg: &'static hw::HwConfig,
        map_kernel_to_user: bool,
    ) -> Result<Self> {
        dev_info!(dev.as_ref(), "MMU: Initializing...\n");

        let inner = Self::make_inner(dev)?;

        let of_node = dev.as_ref().of_node().ok_or(EINVAL)?;
        let res = of_node.reserved_mem_region_to_resource_byname(c_str!("pagetables"))?;
        let ttb1 = res.start();
        let ttb1size: usize = res.size().try_into()?;

        if ttb1size < PAGETABLES_SIZE {
            dev_err!(dev.as_ref(), "MMU: Pagetables region is too small\n");
            return Err(ENOMEM);
        }

        // Clear stale kernel page tables, mirroring m1n1's UAT init:
        //
        //     # Clear out any stale kernel page tables
        //     self.p.memset64(self.ttbr1_base + 0x10, 0, 0x3ff0)
        //     self.u.inst("tlbi vmalle1os")
        //
        // Entries 0 and 1 are left alone; entry 2 onward covers the VA range
        // this driver manages.
        //
        // Note this page is *not* the bootloader's: read from the m1n1 proxy it
        // is all zeroes, and it only becomes populated once start_cpu() runs
        // (LOG.md 28.6). The firmware builds it, sub-allocating entries 0 and 1
        // from gfx-shared-region for its own text and data, and pointing entry
        // 2 at gfx-shared-l2-region. It also re-creates entry 2 after this
        // clear, which is why the clear changes nothing on its own -- kept
        // because it matches m1n1 and removes a dependency on inherited state.
        {
            const SKIP: usize = 2 * size_of::<u64>();
            // SAFETY: ttb1 comes from the bootloader-owned pagetables region,
            // which Vm::new() below validates is mapped, and nothing else
            // references it until the kernel VM is created.
            let ttb1_page = unsafe { kernel::page::Page::borrow_phys_unchecked(&ttb1) };
            ttb1_page.with_pointer_into_page(SKIP, UAT_PGSZ - SKIP, |p| {
                // SAFETY: with_pointer_into_page() bounds-checked this range.
                unsafe { core::ptr::write_bytes(p, 0, UAT_PGSZ - SKIP) };
                Ok(())
            })?;
            mem::sync();
            mem::tlbi_all();
            mem::sync();
            dev_info!(dev.as_ref(), "MMU: Cleared inherited kernel page tables\n");
        }

        dev_info!(dev.as_ref(), "MMU: Creating kernel page tables\n");
        let kernel_lower_vm = Vm::new(dev, inner.clone(), IOVA_USER_RANGE, cfg, None, 1)?;
        let kernel_vm = Vm::new(dev, inner.clone(), IOVA_KERN_RANGE, cfg, Some(ttb1), 0)?;

        dev_info!(dev.as_ref(), "MMU: Kernel page tables created\n");

        let ttb0 = kernel_lower_vm.ttb();

        // 667zzj: create klow's top-level table now, so user VMs can share it.
        if *module_parameters::uma_klow.value() != 0 {
            let k = crate::gpu::IOVA_KERN_GPU_KLOW_RANGE.start;
            kernel_lower_vm
                .inner
                .exec_lock(None, false)?
                .page_table
                .alloc_pages(k..k + UAT_PGSZ as u64)?;
        }

        let uat = Self {
            dev: dev.into(),
            cfg,
            kernel_vm,
            kernel_lower_vm,
            inner,
            slots: slotalloc::SlotAllocator::new(
                UAT_USER_CTX as u32,
                (),
                |_inner, _slot| Some(SlotInner()),
                c_str!("Uat::SlotAllocator"),
                static_lock_class!(),
                static_lock_class!(),
            )?,
        };

        let mut inner = uat.inner.lock();

        inner.map_kernel_to_user = map_kernel_to_user;
        inner.kernel_ttb1 = ttb1;

        inner.handoff().init()?;

        dev_info!(dev.as_ref(), "MMU: Initializing TTBs\n");

        inner.handoff().lock();

        let ttbs = inner.ttbs();

        ttbs[0].ttb0.store(ttb0 | TTBR_VALID, Ordering::SeqCst);
        ttbs[0].ttb1.store(ttb1 | TTBR_VALID, Ordering::SeqCst);
        if *crate::module_parameters::kctx63.value() != 0 {
            ttbs[UAT_NUM_CTX - 1].ttb0.store(ttb0 | TTBR_VALID, Ordering::SeqCst);
            ttbs[UAT_NUM_CTX - 1].ttb1.store(ttb1 | TTBR_VALID, Ordering::SeqCst);
            dev_info!(dev.as_ref(), "MMU: kctx63: TTBAT[63] mirrors the kernel context\n");
        }

        // The firmware finds its page tables by reading this table out of the
        // gpu-region carveout, so if these values or their placement are wrong
        // every UAT address it touches faults -- which is what LOG.md 27.2
        // looks like from the firmware side.
        dev_info!(
            dev.as_ref(),
            "MMU: TTBAT[0] ttb0={:#x} ttb1={:#x} (region {:#x}, slot size {})\n",
            ttb0 | TTBR_VALID,
            ttb1 | TTBR_VALID,
            inner.ttbs_rgn.base,
            size_of::<SlotTTBS>(),
        );

        // Dump the top of the handed-over TTB. The firmware populates this page
        // itself when start_cpu() runs -- it reads as all zeroes from the m1n1
        // proxy before that -- so entries 0 and 1 are the firmware's own
        // mappings (its text lives at 0xfffffc0000000000) and entry 2 is the
        // kernel VA range this driver manages. Seeing all three is what
        // distinguishes "the firmware never set its tables up" from "it did and
        // still cannot read ours" (LOG.md 27.2).
        {
            // SAFETY: ttb1 is the handed-over page table, already validated as
            // mapped by Vm::new() above.
            let p = unsafe { kernel::page::Page::borrow_phys_unchecked(&ttb1) };
            let _ = p.with_pointer_into_page(0, 4 * size_of::<u64>(), |ptr| {
                for i in 0..4 {
                    // SAFETY: bounds-checked by with_pointer_into_page().
                    let v = unsafe { (ptr as *const u64).add(i).read_volatile() };
                    dev_info!(dev.as_ref(), "MMU: TTB1[{}] = {:#018x}\n", i, v);
                }
                Ok(())
            });
        }

        for ctx in &ttbs[1..] {
            ctx.ttb0.store(0, Ordering::Relaxed);
            ctx.ttb1.store(0, Ordering::Relaxed);
        }

        inner.handoff().unlock();

        core::mem::drop(inner);

        dev_info!(dev.as_ref(), "MMU: initialized\n");

        Ok(uat)
    }
}

impl Drop for Uat {
    fn drop(&mut self) {
        // Make sure we flush the TLBs
        fence(Ordering::SeqCst);
        mem::tlbi_all();
        mem::sync();
    }
}
