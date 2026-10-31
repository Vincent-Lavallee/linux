// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! UAT Page Table management
//!
//! AGX GPUs use an MMU called the UAT, which is largely compatible with the ARM64 page table
//! format. This module manages the actual page tables by allocating raw memory pages from
//! the kernel page allocator.

use core::fmt::Debug;
use core::mem::size_of;
use core::ops::Range;
use core::sync::atomic::{
    AtomicU64,
    Ordering, //
};

use kernel::{
    addr::PhysicalAddr,
    error::Result,
    page::Page,
    prelude::*, //
};
#[cfg(CONFIG_DEV_COREDUMP)]
use kernel::{
    types::Owned,
    uapi::{
        PF_R,
        PF_W,
        PF_X, //
    },
};

use crate::debug::*;
use crate::util::align;

const DEBUG_CLASS: DebugFlags = DebugFlags::PgTable;

/// Number of bits in a page offset.
pub(crate) const UAT_PGBIT: usize = 14;
/// UAT page size.
pub(crate) const UAT_PGSZ: usize = 1 << UAT_PGBIT;
/// UAT page offset mask.
pub(crate) const UAT_PGMSK: usize = UAT_PGSZ - 1;

type Pte = AtomicU64;

const PTE_BIT: usize = 3; // log2(sizeof(Pte))
const PTE_SIZE: usize = 1 << PTE_BIT;

/// Number of PTEs per page.
const UAT_NPTE: usize = UAT_PGSZ / size_of::<Pte>();

/// Number of address bits to address a level
const UAT_LVBIT: usize = UAT_PGBIT - PTE_BIT;
/// Number of entries per level
const UAT_LVSZ: usize = UAT_NPTE;
/// Mask of level bits
const UAT_LVMSK: u64 = (UAT_LVSZ - 1) as u64;

const UAT_LEVELS: usize = 3;

/// UAT input address space
///
/// **G15 widened this from 39 to 42 bits.** m1n1 carries the same conditional:
///
///     if Ver.check("G >= G15"):
///         UAT_L0_OFF = 42
///     else:
///         UAT_L0_OFF = 39
///
/// It decides two things at once: how much of a VA survives the mask, and how
/// many entries the top level has (`1 << (IAS - 36)`, so 8 at 39 bits and 64 at
/// 42). Getting it wrong does not fail loudly -- it silently walks a *different
/// top-level slot* than the hardware does. On t6030 the driver installed its L1
/// pointer at index 2 while the GPU walked index 58, and the firmware took a
/// level 1 translation fault reading InitData (LOG.md Part 30).
///
/// The proof that 42 is right: at 42 bits, top-level index 0 covers
/// `0xfffffc0000000000`, which is exactly where the firmware's own text is
/// mapped according to its crash log's address map. At 39 bits nothing lines up.
// TODO: make this per-generation rather than assuming G15.
pub(crate) const UAT_IAS: usize = 42;
const UAT_IASMSK: u64 = (1u64 << UAT_IAS) - 1;

const PTE_TYPE_BITS: u64 = 3;
const PTE_TYPE_LEAF_TABLE: u64 = 3;

const UAT_NON_GLOBAL: u64 = 1 << 11;
const UAT_AP_SHIFT: u32 = 6;
const UAT_AP_BITS: u64 = 3 << UAT_AP_SHIFT;
const UAT_HIGH_BITS_SHIFT: u32 = 52;
const UAT_HIGH_BITS: u64 = 0xfff << UAT_HIGH_BITS_SHIFT;
const UAT_MEMATTR_SHIFT: u32 = 2;
const UAT_MEMATTR_BITS: u64 = 7 << UAT_MEMATTR_SHIFT;

const UAT_PROT_BITS: u64 = UAT_AP_BITS | UAT_MEMATTR_BITS | UAT_HIGH_BITS;

const UAT_AF: u64 = 1 << 10;

const MEMATTR_CACHED: u8 = 0;
const MEMATTR_DEV: u8 = 1;
const MEMATTR_UNCACHED: u8 = 2;

const AP_FW_GPU: u8 = 0;
const AP_FW: u8 = 1;
const AP_GPU: u8 = 2;

const HIGH_BITS_PXN: u16 = 1 << 1;
const HIGH_BITS_UXN: u16 = 1 << 2;
const HIGH_BITS_GPU_ACCESS: u16 = 1 << 3;

#[cfg(CONFIG_DEV_COREDUMP)]
pub(crate) const PTE_ADDR_BITS: u64 = (!UAT_PGMSK as u64) & (!UAT_HIGH_BITS);

#[derive(Debug, Copy, Clone)]
pub(crate) struct Prot {
    memattr: u8,
    ap: u8,
    high_bits: u16,
}

// Firmware + GPU access
const PROT_FW_GPU_NA: Prot = Prot::from_bits(AP_FW_GPU, 0, 0);
const _PROT_FW_GPU_RO: Prot = Prot::from_bits(AP_FW_GPU, 0, 1);
const _PROT_FW_GPU_WO: Prot = Prot::from_bits(AP_FW_GPU, 1, 0);
const PROT_FW_GPU_RW: Prot = Prot::from_bits(AP_FW_GPU, 1, 1);

// Firmware only access
const PROT_FW_RO: Prot = Prot::from_bits(AP_FW, 0, 0);
const _PROT_FW_NA: Prot = Prot::from_bits(AP_FW, 0, 1);
const PROT_FW_RW: Prot = Prot::from_bits(AP_FW, 1, 0);
const PROT_FW_RW_GPU_RO: Prot = Prot::from_bits(AP_FW, 1, 1);

// GPU only access
const PROT_GPU_RO: Prot = Prot::from_bits(AP_GPU, 0, 0);
const PROT_GPU_WO: Prot = Prot::from_bits(AP_GPU, 0, 1);
const PROT_GPU_RW: Prot = Prot::from_bits(AP_GPU, 1, 0);
const _PROT_GPU_NA: Prot = Prot::from_bits(AP_GPU, 1, 1);

#[cfg(CONFIG_DEV_COREDUMP)]
const PF_RW: u32 = PF_R | PF_W;
#[cfg(CONFIG_DEV_COREDUMP)]
const PF_RX: u32 = PF_R | PF_X;

// For crash dumps
#[cfg(CONFIG_DEV_COREDUMP)]
const PROT_TO_PERMS_FW: [[u32; 4]; 4] = [
    [0, 0, 0, PF_RW],
    [0, PF_RW, 0, PF_RW],
    [PF_RX, PF_RX, 0, PF_R],
    [PF_RX, PF_RW, 0, PF_R],
];
#[cfg(CONFIG_DEV_COREDUMP)]
const PROT_TO_PERMS_OS: [[u32; 4]; 4] = [
    [0, PF_R, PF_W, PF_RW],
    [PF_R, 0, PF_RW, PF_RW],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
];

pub(crate) mod prot {
    pub(crate) use super::Prot;
    use super::*;

    /// Firmware MMIO R/W
    pub(crate) const PROT_FW_MMIO_RW: Prot = PROT_FW_RW.memattr(MEMATTR_DEV);
    /// Firmware MMIO R/O
    pub(crate) const PROT_FW_MMIO_RO: Prot = PROT_FW_RO.memattr(MEMATTR_DEV);
    /// Firmware shared (uncached) RW
    pub(crate) const PROT_FW_SHARED_RW: Prot = PROT_FW_RW.memattr(MEMATTR_UNCACHED);
    /// Firmware shared (uncached) RO
    pub(crate) const PROT_FW_SHARED_RO: Prot = PROT_FW_RO.memattr(MEMATTR_UNCACHED);
    /// Firmware private (cached) RW
    pub(crate) const PROT_FW_PRIV_RW: Prot = PROT_FW_RW.memattr(MEMATTR_CACHED);
    /// Firmware/GPU shared (uncached) RW
    pub(crate) const PROT_GPU_FW_SHARED_RW: Prot = PROT_FW_GPU_RW.memattr(MEMATTR_UNCACHED);
    /// Firmware/GPU shared (private) RW
    pub(crate) const PROT_GPU_FW_PRIV_RW: Prot = PROT_FW_GPU_RW.memattr(MEMATTR_CACHED);
    /// Firmware-RW/GPU-RO shared (private) RW
    pub(crate) const PROT_GPU_RO_FW_PRIV_RW: Prot = PROT_FW_RW_GPU_RO.memattr(MEMATTR_CACHED);
    /// GPU shared/coherent RW
    pub(crate) const PROT_GPU_SHARED_RW: Prot = PROT_GPU_RW.memattr(MEMATTR_UNCACHED);
    /// GPU shared/coherent RO
    pub(crate) const PROT_GPU_SHARED_RO: Prot = PROT_GPU_RO.memattr(MEMATTR_UNCACHED);
    /// GPU shared/coherent WO
    pub(crate) const PROT_GPU_SHARED_WO: Prot = PROT_GPU_WO.memattr(MEMATTR_UNCACHED);
}

impl Prot {
    const fn from_bits(ap: u8, uxn: u16, pxn: u16) -> Self {
        assert!(uxn <= 1);
        assert!(pxn <= 1);
        assert!(ap <= 3);

        Prot {
            high_bits: HIGH_BITS_GPU_ACCESS | (pxn * HIGH_BITS_PXN) | (uxn * HIGH_BITS_UXN),
            memattr: 0,
            ap,
        }
    }

    #[cfg(CONFIG_DEV_COREDUMP)]
    pub(crate) const fn from_pte(pte: u64) -> Self {
        Prot {
            high_bits: (pte >> UAT_HIGH_BITS_SHIFT) as u16,
            ap: ((pte & UAT_AP_BITS) >> UAT_AP_SHIFT) as u8,
            memattr: ((pte & UAT_MEMATTR_BITS) >> UAT_MEMATTR_SHIFT) as u8,
        }
    }

    #[cfg(CONFIG_DEV_COREDUMP)]
    pub(crate) const fn elf_flags(&self) -> u32 {
        let ap = (self.ap & 3) as usize;
        let uxn = if self.high_bits & HIGH_BITS_UXN != 0 {
            1
        } else {
            0
        };
        let pxn = if self.high_bits & HIGH_BITS_PXN != 0 {
            1
        } else {
            0
        };
        let gpu = self.high_bits & HIGH_BITS_GPU_ACCESS != 0;

        // Format:
        // [12 top bits of PTE] [12 bottom bits of PTE] [5 bits pad] [ELF RWX]
        let mut perms = if gpu {
            PROT_TO_PERMS_OS[ap][(uxn << 1) | pxn]
        } else {
            PROT_TO_PERMS_FW[ap][(uxn << 1) | pxn]
        };

        perms |= ((self.as_pte() >> 52) << 20) as u32;
        perms |= ((self.as_pte() & 0xfff) << 8) as u32;

        perms
    }

    const fn memattr(&self, memattr: u8) -> Self {
        Self { memattr, ..*self }
    }

    const fn as_pte(&self) -> u64 {
        (self.ap as u64) << UAT_AP_SHIFT
            | (self.high_bits as u64) << UAT_HIGH_BITS_SHIFT
            | (self.memattr as u64) << UAT_MEMATTR_SHIFT
            | UAT_AF
    }

    pub(crate) const fn is_cached_noncoherent(&self) -> bool {
        self.ap != AP_GPU && self.memattr == MEMATTR_CACHED
    }

    /// Same permissions with another memory attribute index (0 cached, 1 device, 2 uncached)
    pub(crate) const fn with_memattr(&self, memattr: u8) -> Self {
        self.memattr(memattr)
    }

    pub(crate) const fn as_uncached(&self) -> Self {
        self.memattr(MEMATTR_UNCACHED)
    }
}

impl Default for Prot {
    fn default() -> Self {
        PROT_FW_GPU_NA
    }
}

#[cfg(CONFIG_DEV_COREDUMP)]
pub(crate) struct DumpedPage {
    pub(crate) iova: u64,
    pub(crate) pte: u64,
    pub(crate) data: Option<Owned<Page>>,
}

pub(crate) struct UatPageTable {
    ttb: PhysicalAddr,
    ttb_owned: bool,
    /// A temporary view onto a live VM's tables (diagnostic helpers): the
    /// tables belong to that VM, so dropping the view must not free them.
    view: bool,
    va_range: Range<u64>,
    oas_mask: u64,
    /// Bitmask of top-level slots pointing at another VM's table (`share_top`):
    /// never walked, changed or freed through this table.
    shared_top: u64,
}

impl UatPageTable {
    pub(crate) fn new(oas: u32) -> Result<Self> {
        mod_pr_debug!("UATPageTable::new: oas={}\n", oas);
        let ttb_page = Page::alloc_page(GFP_KERNEL | __GFP_ZERO)?;
        let ttb = Page::into_phys(ttb_page);
        Ok(UatPageTable {
            ttb,
            ttb_owned: true,
            view: false,
            va_range: 0..(1u64 << UAT_IAS),
            oas_mask: (1u64 << oas) - 1,
            shared_top: 0,
        })
    }

    pub(crate) fn new_with_ttb(ttb: PhysicalAddr, va_range: Range<u64>, oas: u32) -> Result<Self> {
        mod_pr_debug!(
            "UATPageTable::new_with_ttb: ttb={:#x} range={:#x?} oas={}\n",
            ttb,
            va_range,
            oas
        );
        if ttb & (UAT_PGMSK as PhysicalAddr) != 0 {
            return Err(EINVAL);
        }
        if (va_range.start | va_range.end) & (UAT_PGMSK as u64) != 0 {
            return Err(EINVAL);
        }
        // SAFETY: The TTB is should remain valid (if properly mapped), as it is bootloader-managed.
        if unsafe { Page::borrow_phys(&ttb) }.is_none() {
            pr_err!(
                "UATPageTable::new_with_ttb: ttb at {:#x} is not mapped (DT using no-map?)\n",
                ttb
            );
            return Err(EIO);
        }

        Ok(UatPageTable {
            ttb,
            ttb_owned: false,
            view: false,
            va_range,
            oas_mask: (1u64 << oas) - 1,
            shared_top: 0,
        })
    }

    /// Borrow a live VM's page tables for a diagnostic read/poke.
    ///
    /// ⚠️ `new_with_ttb` is for the kernel VM, which owns the tables it
    /// allocates: its `Drop` walks with `free=true`. Used as a throwaway view
    /// onto a *user* VM (all the `*_slot` helpers in mmu.rs did), that walk
    /// freed every live L1/L0 table of the VM; the pages were then handed out
    /// again (e.g. as a GEM object) while the TTB still pointed at them, so
    /// both the GPU and later maps walked garbage. LOG.md 32.41.578.
    pub(crate) fn new_view(ttb: PhysicalAddr, va_range: Range<u64>, oas: u32) -> Result<Self> {
        let mut pt = Self::new_with_ttb(ttb, va_range, oas)?;
        pt.view = true;
        Ok(pt)
    }

    pub(crate) fn ttb(&self) -> PhysicalAddr {
        self.ttb
    }

    /// Top-level PTE `idx` (0 = none).
    pub(crate) fn top_pte(&self, idx: usize) -> Result<u64> {
        // SAFETY: the TTB page is ours (or firmware-managed) and stays allocated.
        let pt = unsafe { Page::borrow_phys_unchecked(&self.ttb) };
        pt.with_pointer_into_page(idx * PTE_SIZE, PTE_SIZE, |p| {
            // SAFETY: valid, aligned PTE pointer (with_pointer_into_page).
            Ok(unsafe { &*(p as *const _ as *const Pte) }.load(Ordering::Relaxed))
        })
    }

    /// Point top-level slot `idx` at another VM's table (`pte` from its `top_pte`).
    /// That subtree stays owned by the other VM: this table skips the slot in
    /// every walk (map/unmap/drop). LOG.md 32.41.667zzj.
    pub(crate) fn share_top(&mut self, idx: usize, pte: u64) -> Result {
        if pte & PTE_TYPE_BITS != PTE_TYPE_LEAF_TABLE || idx >= 64 || self.shared_top & (1 << idx) != 0 {
            return Err(EINVAL);
        }
        // SAFETY: as in top_pte.
        let pt = unsafe { Page::borrow_phys_unchecked(&self.ttb) };
        pt.with_pointer_into_page(idx * PTE_SIZE, PTE_SIZE, |p| {
            // SAFETY: valid, aligned PTE pointer (with_pointer_into_page).
            let ptep = unsafe { &*(p as *const _ as *const Pte) };
            if ptep.load(Ordering::Relaxed) != 0 {
                return Err(EBUSY);
            }
            ptep.store(pte, Ordering::Relaxed);
            Ok(())
        })?;
        self.shared_top |= 1 << idx;
        Ok(())
    }

    fn with_pages<F>(
        &mut self,
        iova_range: Range<u64>,
        alloc: bool,
        free: bool,
        mut cb: F,
    ) -> Result
    where
        F: FnMut(u64, &[Pte]) -> Result,
    {
        mod_pr_debug!(
            "UATPageTable::with_pages: {:#x?} alloc={} free={}\n",
            iova_range,
            alloc,
            free
        );
        if (iova_range.start | iova_range.end) & (UAT_PGMSK as u64) != 0 {
            pr_err!(
                "UATPageTable::with_pages: iova range not aligned: {:#x?}\n",
                iova_range
            );
            return Err(EINVAL);
        }

        if iova_range.is_empty() {
            return Ok(());
        }

        let mut iova = iova_range.start & UAT_IASMSK;
        let mut last_iova = iova;
        // Handle the case where iova_range.end is just at the top boundary of the IAS
        let end = ((iova_range.end - 1) & UAT_IASMSK) + 1;

        let mut pt_addr: [Option<PhysicalAddr>; UAT_LEVELS] = Default::default();
        pt_addr[UAT_LEVELS - 1] = Some(self.ttb);
        let shared_top = self.shared_top;

        'outer: while iova < end {
            mod_pr_debug!("UATPageTable::with_pages: iova={:#x}\n", iova);
            let addr_diff = last_iova ^ iova;
            for level in (0..UAT_LEVELS - 1).rev() {
                // If the iova has changed at this level or above, invalidate the physaddr
                if addr_diff & !((1 << (UAT_PGBIT + (level + 1) * UAT_LVBIT)) - 1) != 0 {
                    if let Some(phys) = pt_addr[level].take() {
                        if free {
                            mod_pr_debug!(
                                "UATPageTable::with_pages: free level {} {:#x?}\n",
                                level,
                                phys
                            );
                            // SAFETY: Page tables for our VA ranges always come from Page::into_phys().
                            unsafe { Page::from_phys(phys) };
                        }
                        mod_pr_debug!("UATPageTable::with_pages: invalidate level {}\n", level);
                    }
                }
            }
            last_iova = iova;
            for level in (0..UAT_LEVELS - 1).rev() {
                // Fetch the page table base address for this level
                if pt_addr[level].is_none() {
                    let phys = pt_addr[level + 1].unwrap();
                    mod_pr_debug!(
                        "UATPageTable::with_pages: need level {}, parent phys {:#x}\n",
                        level,
                        phys
                    );
                    let upidx = ((iova >> (UAT_PGBIT + (level + 1) * UAT_LVBIT) as u64) & UAT_LVMSK)
                        as usize;
                    if level == UAT_LEVELS - 2 && upidx < 64 && shared_top & (1 << upidx) != 0 {
                        // Another VM's subtree (share_top): skip it.
                        let block = 1 << (UAT_PGBIT + UAT_LVBIT * (level + 1));
                        iova = align(iova + 1, block);
                        continue 'outer;
                    }
                    // SAFETY: Page table addresses are either allocated by us, or
                    // firmware-managed and safe to borrow a struct page from.
                    let upt = unsafe { Page::borrow_phys_unchecked(&phys) };
                    mod_pr_debug!("UATPageTable::with_pages: borrowed phys {:#x}\n", phys);
                    pt_addr[level] =
                        upt.with_pointer_into_page(upidx * PTE_SIZE, PTE_SIZE, |p| {
                            let uptep = p as *const _ as *const Pte;
                            // SAFETY: with_pointer_into_page() ensures the pointer is valid,
                            // and our index is aligned so it is safe to deref as an AtomicU64.
                            let upte = unsafe { &*uptep };
                            let mut upte_val = upte.load(Ordering::Relaxed);
                            // Single-line: "{:#x?}" on the Option below prints
                            // one field per line and only the first survives.
                            mod_pr_debug!(
                                "UATPageTable::with_pages: level {} upidx {:#x} upte {:#x}\n",
                                level,
                                upidx,
                                upte_val
                            );
                            // Allocate if requested
                            if upte_val == 0 && alloc {
                                let pt_page = Page::alloc_page(GFP_KERNEL | __GFP_ZERO)?;
                                mod_pr_debug!("UATPageTable::with_pages: alloc PT at {:#x}\n", pt_page.phys());
                                let pt_paddr = Page::into_phys(pt_page);
                                // LOG.md 32.41.577: a table PTE shows up in a replay data
                                // page at page offset 0x2000 (index 0x400).  Name the walk.
                                if upidx == 0x400 && crate::debug::verbose() {
                                    pr_info!(
                                        "asahi/pt400: iova {:#x} level {} parent {:#x} -> new PT {:#x}\n",
                                        iova, level, phys, pt_paddr
                                    );
                                }
                                upte_val = pt_paddr | PTE_TYPE_LEAF_TABLE;
                                upte.store(upte_val, Ordering::Relaxed);
                            }
                            if upte_val & PTE_TYPE_BITS == PTE_TYPE_LEAF_TABLE {
                                Ok(Some(upte_val & self.oas_mask & (!UAT_PGMSK as u64)))
                            } else if upte_val == 0 || (!alloc && !free) {
                                mod_pr_debug!("UATPageTable::with_pages: no level {}\n", level);
                                Ok(None)
                            } else {
                                pr_err!("UATPageTable::with_pages: Unexpected Table PTE value {:#x} at iova {:#x} index {} phys {:#x}\n", upte_val,
                                        iova, level + 1, phys + ((upidx * PTE_SIZE) as PhysicalAddr));
                                Ok(None)
                            }
                        })?;
                    mod_pr_debug!(
                        "UATPageTable::with_pages: level {} PT {:#x?}\n",
                        level,
                        pt_addr[level]
                    );
                }
                // If we don't have a page table, skip this entire level
                if pt_addr[level].is_none() {
                    let block = 1 << (UAT_PGBIT + UAT_LVBIT * (level + 1));
                    let old = iova;
                    iova = align(iova + 1, block);
                    mod_pr_debug!(
                        "UATPageTable::with_pages: skip {:#x} {:#x} -> {:#x}\n",
                        block,
                        old,
                        iova
                    );
                    continue 'outer;
                }
            }

            let idx = ((iova >> UAT_PGBIT as u64) & UAT_LVMSK) as usize;
            let max_count = UAT_NPTE - idx;
            let count = (((end - iova) >> UAT_PGBIT) as usize).min(max_count);
            let phys = pt_addr[0].unwrap();
            mod_pr_debug!(
                "UATPageTable::with_pages: leaf PT at {:#x} idx {:#x} count {:#x} iova {:#x}\n",
                phys,
                idx,
                count,
                iova
            );
            // SAFETY: Page table addresses are either allocated by us, or
            // firmware-managed and safe to borrow a struct page from.
            let pt = unsafe { Page::borrow_phys_unchecked(&phys) };
            pt.with_pointer_into_page(idx * PTE_SIZE, count * PTE_SIZE, |p| {
                let ptep = p as *const _ as *const Pte;
                // SAFETY: We know this is a valid pointer to PTEs and the range is valid and
                // checked by with_pointer_into_page().
                let ptes = unsafe { core::slice::from_raw_parts(ptep, count) };
                cb(iova, ptes)?;
                Ok(())
            })?;

            let block = 1 << (UAT_PGBIT + UAT_LVBIT);
            iova = align(iova + 1, block);
        }

        if free {
            for level in (0..UAT_LEVELS - 1).rev() {
                if let Some(phys) = pt_addr[level] {
                    mod_pr_debug!(
                        "UATPageTable::with_pages: free level {} {:#x?}\n",
                        level,
                        phys
                    );
                    // SAFETY: Page tables for our VA ranges always come from Page::into_phys().
                    unsafe { Page::from_phys(phys) };
                }
            }
        }

        Ok(())
    }

    pub(crate) fn alloc_pages(&mut self, iova_range: Range<u64>) -> Result {
        mod_pr_debug!("UATPageTable::alloc_pages: {:#x?}\n", iova_range);
        self.with_pages(iova_range, true, false, |_, _| Ok(()))
    }

    fn pte_bits(&self) -> u64 {
        if self.ttb_owned {
            // Owned page tables are userspace, so non-global
            PTE_TYPE_LEAF_TABLE | UAT_NON_GLOBAL
        } else {
            // The sole non-owned page table is kernelspace, so global
            PTE_TYPE_LEAF_TABLE
        }
    }

    pub(crate) fn map_pages(
        &mut self,
        iova_range: Range<u64>,
        mut phys: PhysicalAddr,
        prot: Prot,
        one_page: bool,
    ) -> Result {
        mod_pr_debug!(
            "UATPageTable::map_pages: {:#x?} {:#x?} {:?}\n",
            iova_range,
            phys,
            prot
        );
        if phys & (UAT_PGMSK as PhysicalAddr) != 0 {
            pr_err!("UATPageTable::map_pages: phys not aligned: {:#x?}\n", phys);
            return Err(EINVAL);
        }

        let pte_bits = self.pte_bits();

        self.with_pages(iova_range, true, false, |iova, ptes| {
            for (idx, pte) in ptes.iter().enumerate() {
                let ptev = pte.load(Ordering::Relaxed);
                if ptev != 0 {
                    pr_err!(
                        "UATPageTable::map_pages: Page at IOVA {:#x} is mapped (PTE: {:#x})\n",
                        iova + (idx * UAT_PGSZ) as u64,
                        ptev
                    );
                }
                pte.store(phys | prot.as_pte() | pte_bits, Ordering::Relaxed);
                if !one_page {
                    phys += UAT_PGSZ as PhysicalAddr;
                }
            }
            Ok(())
        })
    }

    pub(crate) fn reprot_pages(&mut self, iova_range: Range<u64>, prot: Prot) -> Result {
        mod_pr_debug!(
            "UATPageTable::reprot_pages: {:#x?} {:?}\n",
            iova_range,
            prot
        );
        self.with_pages(iova_range, true, false, |iova, ptes| {
            for (idx, pte) in ptes.iter().enumerate() {
                let ptev = pte.load(Ordering::Relaxed);
                if ptev & PTE_TYPE_BITS != PTE_TYPE_LEAF_TABLE {
                    pr_err!(
                        "UATPageTable::reprot_pages: Page at IOVA {:#x} is unmapped (PTE: {:#x})\n",
                        iova + (idx * UAT_PGSZ) as u64,
                        ptev
                    );
                    continue;
                }
                pte.store((ptev & !UAT_PROT_BITS) | prot.as_pte(), Ordering::Relaxed);
            }
            Ok(())
        })
    }

    pub(crate) fn unmap_pages(&mut self, iova_range: Range<u64>) -> Result {
        mod_pr_debug!("UATPageTable::unmap_pages: {:#x?}\n", iova_range);
        self.with_pages(iova_range, false, false, |iova, ptes| {
            for (idx, pte) in ptes.iter().enumerate() {
                if pte.load(Ordering::Relaxed) & PTE_TYPE_LEAF_TABLE == 0 {
                    pr_err!(
                        "UATPageTable::unmap_pages: Page at IOVA {:#x} already unmapped\n",
                        iova + (idx * UAT_PGSZ) as u64
                    );
                }
                pte.store(0, Ordering::Relaxed);
            }
            Ok(())
        })
    }

    #[cfg(CONFIG_DEV_COREDUMP)]
    /// Look up a single IOVA: report its PTE and the first words of the page.
    ///
    /// `dump_pages` copies whole pages and allocates; this is the cheap version,
    /// for answering "is this address mapped, and what is in it" from a fault
    /// handler. Returns `None` when the leaf PTE is not present.
    /// Write one 64-bit word at a user IOVA, for the `cs_poke_*` diagnostic.
    ///
    /// Same page-table walk and `borrow_phys` argument as `probe`, plus a cache
    /// clean so the GPU (which reads this memory uncached) sees the store.
    /// Returns true if the page was mapped and the write happened.
    pub(crate) fn poke(&mut self, iova: u64, val: u64) -> bool {
        let base = iova & !(UAT_PGMSK as u64);
        let off = (iova & (UAT_PGMSK as u64)) as usize;
        let oas_mask = self.oas_mask;
        let mut done = false;

        let _ = self.with_pages(base..(base + UAT_PGSZ as u64), false, false, |_, ptes| {
            let pte = match ptes.first() {
                Some(p) => p.load(Ordering::Relaxed),
                None => return Ok(()),
            };
            if (pte & PTE_TYPE_LEAF_TABLE) != PTE_TYPE_LEAF_TABLE {
                return Ok(());
            }
            if off + 8 > UAT_PGSZ {
                return Ok(());
            }
            let phys = pte & oas_mask & (!UAT_PGMSK as u64);
            // SAFETY: same argument as `probe`.
            if let Some(page) = unsafe { Page::borrow_phys(&phys) } {
                page.with_page_mapped(|p| {
                    // SAFETY: `off + 8 <= UAT_PGSZ` keeps the write inside the
                    // mapped page.
                    unsafe {
                        core::ptr::write_unaligned(p.add(off) as *mut u64, val);
                    }
                    // ⚠️ The DSB between the store and the cache maintenance is
                    // load-bearing, not decoration. `DC CIVAC` is not ordered
                    // with respect to preceding stores without one, so without
                    // it the clean can run *before* the store lands and the GPU
                    // reads the stale line -- intermittently. That produced two
                    // byte-identical runs with opposite outcomes and cost the
                    // retraction in LOG.md 32.41.174.
                    crate::mem::sync();
                    crate::mem::dc_civac(unsafe { p.add(off) } as *const u8);
                });
                crate::mem::sync();
                done = true;
            }
            Ok(())
        });
        done
    }

    /// Fill `count` 32-bit words at `iova` with `word`, all within one page.
    ///
    /// The terminator sweep needs to plant `0xc0000000` at every dword across a
    /// range in a single run: each boot submits exactly one job, so probing one
    /// candidate boundary per boot costs ~5 minutes per bit. Filling a range
    /// answers "does the record end anywhere in here" in one run instead.
    /// LOG.md 32.41.163.
    pub(crate) fn poke_fill(&mut self, iova: u64, word: u32, count: usize) -> bool {
        let base = iova & !(UAT_PGMSK as u64);
        let off = (iova & (UAT_PGMSK as u64)) as usize;
        let oas_mask = self.oas_mask;
        let mut done = false;

        let _ = self.with_pages(base..(base + UAT_PGSZ as u64), false, false, |_, ptes| {
            let pte = match ptes.first() {
                Some(p) => p.load(Ordering::Relaxed),
                None => return Ok(()),
            };
            if (pte & PTE_TYPE_LEAF_TABLE) != PTE_TYPE_LEAF_TABLE {
                return Ok(());
            }
            // One page only: the control stream starts page-aligned, so any
            // sweep of interest lives in the first page.
            if off + count * 4 > UAT_PGSZ {
                return Ok(());
            }
            let phys = pte & oas_mask & (!UAT_PGMSK as u64);
            // SAFETY: same argument as `probe`.
            if let Some(page) = unsafe { Page::borrow_phys(&phys) } {
                page.with_page_mapped(|p| {
                    for i in 0..count {
                        // SAFETY: bounds checked above.
                        unsafe {
                            core::ptr::write_unaligned(
                                p.add(off + i * 4) as *mut u32,
                                word,
                            );
                        }
                    }
                    // All stores first, then one barrier, then the cleans --
                    // `DC CIVAC` is unordered with respect to prior stores
                    // without an intervening DSB. See `poke` above and
                    // LOG.md 32.41.175.
                    crate::mem::sync();
                    for i in 0..count {
                        // SAFETY: bounds checked above.
                        unsafe {
                            crate::mem::dc_civac(p.add(off + i * 4) as *const u8);
                        }
                    }
                });
                crate::mem::sync();
                done = true;
            }
            Ok(())
        });
        done
    }

    /// Read-modify-write every dword in `[iova, iova + count*4)` that matches
    /// `(w & mask) == val`, ORing `or_val` into it.  Returns how many were
    /// changed.
    ///
    /// Same single-page restriction and same store/DSB/DC-CIVAC ordering as
    /// `poke_fill`; see LOG.md 32.41.175 for why the barrier has to sit between
    /// the stores and the cleans.
    pub(crate) fn patch_dwords(
        &mut self,
        iova: u64,
        count: usize,
        mask: u32,
        val: u32,
        or_val: u32,
        and_val: u32,
    ) -> usize {
        let base = iova & !(UAT_PGMSK as u64);
        let off = (iova & (UAT_PGMSK as u64)) as usize;
        let oas_mask = self.oas_mask;
        let mut hits = 0usize;

        let _ = self.with_pages(base..(base + UAT_PGSZ as u64), false, false, |_, ptes| {
            let pte = match ptes.first() {
                Some(p) => p.load(Ordering::Relaxed),
                None => return Ok(()),
            };
            if (pte & PTE_TYPE_LEAF_TABLE) != PTE_TYPE_LEAF_TABLE {
                return Ok(());
            }
            if off + count * 4 > UAT_PGSZ {
                return Ok(());
            }
            let phys = pte & oas_mask & (!UAT_PGMSK as u64);
            // SAFETY: same argument as `probe`.
            if let Some(page) = unsafe { Page::borrow_phys(&phys) } {
                page.with_page_mapped(|p| {
                    for i in 0..count {
                        // SAFETY: bounds checked above.
                        let q = unsafe { p.add(off + i * 4) as *mut u32 };
                        // SAFETY: as above.
                        let w = unsafe { core::ptr::read_unaligned(q) };
                        if (w & mask) == val {
                            let n = (w & and_val) | or_val;
                            if n != w {
                                // SAFETY: as above.
                                unsafe { core::ptr::write_unaligned(q, n) };
                                hits += 1;
                            }
                        }
                    }
                    crate::mem::sync();
                    for i in 0..count {
                        // SAFETY: as above.
                        unsafe {
                            crate::mem::dc_civac(p.add(off + i * 4) as *const u8);
                        }
                    }
                });
                crate::mem::sync();
            }
            Ok(())
        });
        hits
    }

    /// Clear bits [31:16] of every stream dword the G15S VDM reads as a count.
    ///
    /// The defect (LOG.md 32.41.320/.321): the VDM takes bits [31:16] of a
    /// record's low dword as a count, and Mesa writes a 32-bit address there.
    /// `0xfff88b18` means "consume 65528 units", so the decoder walks the zero
    /// fill past the end of the stream and faults off the end of the BO.
    ///
    /// Identifying the record by *content* rather than by a fixed offset,
    /// because the offset moves between frames (32.41.329). A bare
    /// "any `0xfff8xxxx`" rule is wrong -- `cs+0x08` holds `0xfff88010` and is a
    /// perfectly well-behaved record (32.41.317, boundary 3/3). The two are
    /// distinguished by the dword that follows: `0x0000142f` for the pathological
    /// one, `0x40000000` for the good one. So require the successor to be small,
    /// which is the only discriminator the measurements support.
    ///
    /// Returns the number of dwords patched.
    pub(crate) fn fix_count_dwords(&mut self, iova: u64, count: usize) -> usize {
        let base = iova & !(UAT_PGMSK as u64);
        let off = (iova & (UAT_PGMSK as u64)) as usize;
        let oas_mask = self.oas_mask;
        let mut hits = 0usize;

        let _ = self.with_pages(base..(base + UAT_PGSZ as u64), false, false, |_, ptes| {
            let pte = match ptes.first() {
                Some(p) => p.load(Ordering::Relaxed),
                None => return Ok(()),
            };
            if (pte & PTE_TYPE_LEAF_TABLE) != PTE_TYPE_LEAF_TABLE {
                return Ok(());
            }
            if off + (count + 1) * 4 > UAT_PGSZ {
                return Ok(());
            }
            let phys = pte & oas_mask & (!UAT_PGMSK as u64);
            // SAFETY: same argument as `probe`.
            if let Some(page) = unsafe { Page::borrow_phys(&phys) } {
                page.with_page_mapped(|p| {
                    for i in 0..count {
                        // SAFETY: bounds checked above.
                        let q = unsafe { p.add(off + i * 4) as *mut u32 };
                        // SAFETY: as above.
                        let w = unsafe { core::ptr::read_unaligned(q) };
                        if (w & 0xffff_0000) != 0xfff8_0000 {
                            continue;
                        }
                        // SAFETY: `count + 1` bounds-checked above.
                        let nx = unsafe {
                            core::ptr::read_unaligned(p.add(off + (i + 1) * 4) as *const u32)
                        };
                        if nx >= 0x1_0000 {
                            continue;
                        }
                        // SAFETY: as above.
                        unsafe { core::ptr::write_unaligned(q, w & 0xffff) };
                        hits += 1;
                    }
                    if hits != 0 {
                        crate::mem::sync();
                        for i in 0..count {
                            // SAFETY: as above.
                            unsafe {
                                crate::mem::dc_civac(p.add(off + i * 4) as *const u8);
                            }
                        }
                    }
                });
                crate::mem::sync();
            }
            Ok(())
        });
        hits
    }

    /// Find a dword matching `(w & mask) == val` and write `writes` relative to it.
    ///
    /// `patch_dwords` can only rewrite the matched word itself, but the G15 draw
    /// record is eight dwords and the fields that need filling sit *after* its
    /// CTRL word (LOG.md 32.41.299). Each `(idx, value)` in `writes` is a dword
    /// index relative to the match, so Apple's record can be transcribed onto
    /// Mesa's in place. Only the first match is spliced -- the stream holds one
    /// draw. Same single-page limit and same store/DSB/DC-CIVAC ordering as
    /// `patch_dwords`.
    pub(crate) fn splice_dwords(
        &mut self,
        iova: u64,
        count: usize,
        mask: u32,
        val: u32,
        writes: &[(usize, u32)],
    ) -> usize {
        let base = iova & !(UAT_PGMSK as u64);
        let off = (iova & (UAT_PGMSK as u64)) as usize;
        let oas_mask = self.oas_mask;
        let mut hits = 0usize;

        // The furthest dword any write touches, so the whole record can be
        // bounds-checked against the page before anything is stored.
        let reach = writes.iter().map(|(i, _)| *i).max().unwrap_or(0);

        let _ = self.with_pages(base..(base + UAT_PGSZ as u64), false, false, |_, ptes| {
            let pte = match ptes.first() {
                Some(p) => p.load(Ordering::Relaxed),
                None => return Ok(()),
            };
            if (pte & PTE_TYPE_LEAF_TABLE) != PTE_TYPE_LEAF_TABLE {
                return Ok(());
            }
            if off + (count + reach) * 4 > UAT_PGSZ {
                return Ok(());
            }
            let phys = pte & oas_mask & (!UAT_PGMSK as u64);
            // SAFETY: same argument as `probe`.
            if let Some(page) = unsafe { Page::borrow_phys(&phys) } {
                page.with_page_mapped(|p| {
                    let mut lo = 0usize;
                    let mut hi = 0usize;
                    for i in 0..count {
                        // SAFETY: bounds checked above.
                        let q = unsafe { p.add(off + i * 4) as *mut u32 };
                        // SAFETY: as above.
                        let w = unsafe { core::ptr::read_unaligned(q) };
                        if (w & mask) != val {
                            continue;
                        }
                        for (idx, value) in writes.iter() {
                            // SAFETY: bounds checked above via `reach`.
                            let t = unsafe { p.add(off + (i + idx) * 4) as *mut u32 };
                            // SAFETY: as above.
                            unsafe { core::ptr::write_unaligned(t, *value) };
                        }
                        lo = i;
                        hi = i + reach;
                        hits += 1;
                        break;
                    }
                    if hits != 0 {
                        crate::mem::sync();
                        for i in lo..=hi {
                            // SAFETY: as above.
                            unsafe {
                                crate::mem::dc_civac(p.add(off + i * 4) as *const u8);
                            }
                        }
                    }
                });
                crate::mem::sync();
            }
            Ok(())
        });
        hits
    }

    /// Read `count` dwords from `iova` into `out`, returning how many were read.
    ///
    /// Read-only twin of `patch_dwords`: same single-page limit, same borrow of
    /// the leaf PTE's physical page. Used to dump the VDM control stream exactly
    /// as the GPU sees it, which is the only way to check the CSD's cursor
    /// advance against the records Mesa actually wrote. LOG.md 32.41.298.
    pub(crate) fn read_dwords(&mut self, iova: u64, out: &mut [u32]) -> usize {
        let base = iova & !(UAT_PGMSK as u64);
        let off = (iova & (UAT_PGMSK as u64)) as usize;
        let oas_mask = self.oas_mask;
        let count = out.len();
        let mut got = 0usize;

        let _ = self.with_pages(base..(base + UAT_PGSZ as u64), false, false, |_, ptes| {
            let pte = match ptes.first() {
                Some(p) => p.load(Ordering::Relaxed),
                None => return Ok(()),
            };
            if (pte & PTE_TYPE_LEAF_TABLE) != PTE_TYPE_LEAF_TABLE {
                return Ok(());
            }
            if off + count * 4 > UAT_PGSZ {
                return Ok(());
            }
            let phys = pte & oas_mask & (!UAT_PGMSK as u64);
            // SAFETY: same argument as `probe`.
            if let Some(page) = unsafe { Page::borrow_phys(&phys) } {
                page.with_page_mapped(|p| {
                    for i in 0..count {
                        // SAFETY: bounds checked above.
                        unsafe {
                            crate::mem::dc_civac(p.add(off + i * 4) as *const u8);
                        }
                    }
                    crate::mem::sync();
                    for i in 0..count {
                        // SAFETY: bounds checked above.
                        let q = unsafe { p.add(off + i * 4) as *const u32 };
                        // SAFETY: as above.
                        out[i] = unsafe { core::ptr::read_unaligned(q) };
                    }
                    got = count;
                });
            }
            Ok(())
        });
        got
    }

    pub(crate) fn probe(&mut self, iova: u64) -> Option<(u64, [u64; 8])> {
        let base = iova & !(UAT_PGMSK as u64);
        let off = (iova & (UAT_PGMSK as u64)) as usize;
        let oas_mask = self.oas_mask;
        let mut out: Option<(u64, [u64; 8])> = None;

        let _ = self.with_pages(base..(base + UAT_PGSZ as u64), false, false, |_, ptes| {
            let pte = match ptes.first() {
                Some(p) => p.load(Ordering::Relaxed),
                None => return Ok(()),
            };
            if (pte & PTE_TYPE_LEAF_TABLE) != PTE_TYPE_LEAF_TABLE {
                return Ok(());
            }
            let mut words = [0u64; 8];
            let phys = pte & oas_mask & (!UAT_PGMSK as u64);
            // SAFETY: same argument as `dump_pages` -- a present leaf PTE in a
            // live page table points at a page that is either firmware-owned,
            // a page table page, or a locked GEM page, all of which are safe to
            // borrow for a read.
            if let Some(page) = unsafe { Page::borrow_phys(&phys) } {
                page.with_page_mapped(|p| {
                    for (i, w) in words.iter_mut().enumerate() {
                        let o = off + i * 8;
                        if o + 8 <= UAT_PGSZ {
                            // SAFETY: `o + 8 <= UAT_PGSZ` keeps the read inside
                            // the mapped page.
                            *w = unsafe { core::ptr::read_unaligned(p.add(o) as *const u64) };
                        }
                    }
                });
            }
            out = Some((pte, words));
            Ok(())
        });
        out
    }

    pub(crate) fn dump_pages(&mut self, iova_range: Range<u64>) -> Result<KVVec<DumpedPage>> {
        let mut pages = KVVec::new();
        let oas_mask = self.oas_mask;
        let iova_base = self.va_range.start & !UAT_IASMSK;
        self.with_pages(iova_range, false, false, |iova, ptes| {
            let iova = iova | iova_base;
            for (idx, ppte) in ptes.iter().enumerate() {
                let pte = ppte.load(Ordering::Relaxed);
                if (pte & PTE_TYPE_LEAF_TABLE) != PTE_TYPE_LEAF_TABLE {
                    continue;
                }
                let memattr = ((pte & UAT_MEMATTR_BITS) >> UAT_MEMATTR_SHIFT) as u8;

                if !(memattr == MEMATTR_CACHED || memattr == MEMATTR_UNCACHED) {
                    pages.push(
                        DumpedPage {
                            iova: iova + (idx * UAT_PGSZ) as u64,
                            pte,
                            data: None,
                        },
                        GFP_KERNEL,
                    )?;
                    continue;
                }
                let phys = pte & oas_mask & (!UAT_PGMSK as u64);
                // SAFETY: GPU pages are either firmware/preallocated pages
                // (which the kernel isn't concerned with and are either in
                // the page map or not, and if they aren't, borrow_phys()
                // will fail), or GPU page table pages (which we own),
                // or GEM buffer pages (which are locked while they are
                // mapped in the page table), so they should be safe to
                // borrow.
                //
                // This does trust the firmware not to have any weird
                // mappings in its own internal page tables, but since
                // those are managed by the uPPL which is privileged anyway,
                // this trust does not actually extend any trust boundary.
                let src_page = match unsafe { Page::borrow_phys(&phys) } {
                    Some(page) => page,
                    None => {
                        pages.push(
                            DumpedPage {
                                iova: iova + (idx * UAT_PGSZ) as u64,
                                pte,
                                data: None,
                            },
                            GFP_KERNEL,
                        )?;
                        continue;
                    }
                };
                let dst_page = Page::alloc_page(GFP_KERNEL)?;
                src_page.with_page_mapped(|psrc| -> Result {
                    // SAFETY: This could technically still have a data race with the firmware
                    // or other driver code (or even userspace with timestamp buffers), but while
                    // the Rust language technically says this is UB, in the real world, using
                    // atomic reads for this is guaranteed to never cause any harmful effects
                    // other than possibly reading torn/unreliable data. At least on ARM64 anyway.
                    //
                    // (Yes, I checked with Rust people about this. ~~ Lina)
                    //
                    let src_items = unsafe {
                        core::slice::from_raw_parts(
                            psrc as *const AtomicU64,
                            UAT_PGSZ / core::mem::size_of::<AtomicU64>(),
                        )
                    };
                    dst_page.with_page_mapped(|pdst| -> Result {
                        // SAFETY: We own the destination page, so it is safe to view its contents
                        // as a u64 slice.
                        let dst_items = unsafe {
                            core::slice::from_raw_parts_mut(
                                pdst as *mut u64,
                                UAT_PGSZ / core::mem::size_of::<u64>(),
                            )
                        };
                        for (si, di) in src_items.iter().zip(dst_items.iter_mut()) {
                            *di = si.load(Ordering::Relaxed);
                        }
                        Ok(())
                    })?;
                    Ok(())
                })?;
                pages.push(
                    DumpedPage {
                        iova: iova + (idx * UAT_PGSZ) as u64,
                        pte,
                        data: Some(dst_page),
                    },
                    GFP_KERNEL,
                )?;
            }
            Ok(())
        })?;
        Ok(pages)
    }
}

impl Drop for UatPageTable {
    fn drop(&mut self) {
        mod_pr_debug!("UATPageTable::drop range: {:#x?}\n", &self.va_range);
        if self.view {
            return;
        }
        if self
            .with_pages(self.va_range.clone(), false, true, |iova, ptes| {
                // ⚠️ `user_va_bias2` leaks a page per user mapping *by design*
                // (mmu.rs `user_iova_bias2`), and one line each overruns the
                // kmsg ring: 32.41.253's `render-full2`/`full3` logs both carry
                // "/dev/kmsg buffer overrun, some messages lost", which is how
                // a `patch` line that may well have been printed went missing
                // and nearly got the fix scored as not applied.  Count them and
                // print one line.  LOG.md 32.41.254.
                let mut leaked = 0usize;
                let mut first = 0u64;
                for (idx, pte) in ptes.iter().enumerate() {
                    if pte.load(Ordering::Relaxed) != 0 {
                        if leaked == 0 {
                            first = iova + (idx * UAT_PGSZ) as u64;
                        }
                        leaked += 1;
                    }
                }
                if leaked != 0 {
                    pr_err!(
                        "UATPageTable::drop: {} leaked page(s), first at IOVA {:#x}\n",
                        leaked,
                        first
                    );
                }
                Ok(())
            })
            .is_err()
        {
            pr_err!("UATPageTable::drop failed to free page tables\n",);
        }
        if self.ttb_owned {
            mod_pr_debug!("UATPageTable::drop: Free TTB {:#x}\n", self.ttb);
            // SAFETY: If we own the ttb, it was allocated with Page::into_phys().
            unsafe {
                Page::from_phys(self.ttb);
            }
        }
    }
}
