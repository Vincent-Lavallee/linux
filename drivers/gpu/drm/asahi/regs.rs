// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU MMIO register abstraction
//!
//! Since the vast majority of the interactions with the GPU are brokered through the firmware,
//! there is very little need to interact directly with GPU MMIO register. This module abstracts
//! the few operations that require that, mainly reading the MMU fault status, reading GPU ID
//! information, and starting the GPU firmware coprocessor.

use crate::hw;
use kernel::{
    c_str,
    device::Core,
    devres::Devres,
    io::{
        mem::IoMem, //
        Io,
    },
    platform,
    prelude::*,
    sync::aref::ARef,
    time::{
        delay::fsleep,
        Delta, //
    },
};

/// Size of the ASC control MMIO region.
pub(crate) const ASC_CTL_SIZE: usize = 0x4000;

/// Size of the SGX MMIO region.
pub(crate) const SGX_SIZE: usize = 0x1000000;

const CPU_CONTROL: usize = 0x44;
const CPU_RUN: u32 = 0x1 << 4; // BIT(4)

const FAULT_INFO: usize = 0x17030;

const ID_VERSION: usize = 0xd04000;
const ID_UNK08: usize = 0xd04008;
const ID_COUNTS_1: usize = 0xd04010;
const ID_COUNTS_2: usize = 0xd04014;
const ID_UNK18: usize = 0xd04018;
const ID_CLUSTERS: usize = 0xd0401c;

const CORE_MASK_0: usize = 0xd01500;
const CORE_MASK_1: usize = 0xd01514;

const CORE_MASKS_G14X: usize = 0xe01500;
const FAULT_INFO_G14X: usize = 0xd8c0;
const FAULT_ADDR_G14X: usize = 0xd8c8;

/// Enum representing the unit that caused an MMU fault.
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum FaultUnit {
    /// Decompress / pixel fetch
    DCMP(u8),
    /// USC L1 Cache (device loads/stores)
    UL1C(u8),
    /// Compress / pixel store
    CMP(u8),
    GSL1(u8),
    IAP(u8),
    VCE(u8),
    /// Tiling Engine
    TE(u8),
    RAS(u8),
    /// Vertex Data Master
    VDM(u8),
    PPP(u8),
    /// ISP Parameter Fetch
    IPF(u8),
    IPF_CPF(u8),
    VF(u8),
    VF_CPF(u8),
    /// Depth/Stencil load/store
    ZLS(u8),

    /// Parameter Management
    dPM,
    /// Compute Data Master
    dCDM_KS(u8),
    dIPP,
    dIPP_CS,
    // Vertex Data Master
    dVDM_CSD,
    dVDM_SSD,
    dVDM_ILF,
    dVDM_ILD,
    dRDE(u8),
    FC,
    GSL2,

    /// Graphics L2 Cache Control?
    GL2CC_META(u8),
    GL2CC_MB,

    /// Parameter Management
    gPM_SP(u8),
    /// Vertex Data Master - CSD
    gVDM_CSD_SP(u8),
    gVDM_SSD_SP(u8),
    gVDM_ILF_SP(u8),
    gVDM_TFP_SP(u8),
    gVDM_MMB_SP(u8),
    /// Compute Data Master
    gCDM_CS_KS0_SP(u8),
    gCDM_CS_KS1_SP(u8),
    gCDM_CS_KS2_SP(u8),
    gCDM_KS0_SP(u8),
    gCDM_KS1_SP(u8),
    gCDM_KS2_SP(u8),
    gIPP_SP(u8),
    gIPP_CS_SP(u8),
    gRDE0_SP(u8),
    gRDE1_SP(u8),

    gCDM_CS,
    gCDM_ID,
    gCDM_CSR,
    gCDM_CSW,
    gCDM_CTXR,
    gCDM_CTXW,
    gIPP,
    gIPP_CS,
    gKSM_RCE,
    /// UMA page manager
    gUPM,

    Unknown(u8),
}

/// Reason for an MMU fault.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum FaultReason {
    Unmapped,
    AfFault,
    WriteOnly,
    ReadOnly,
    NoAccess,
    Unknown(u8),
}

/// Collection of information about an MMU fault.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct FaultInfo {
    pub(crate) address: u64,
    pub(crate) sideband: u8,
    pub(crate) vm_slot: u32,
    pub(crate) unit_code: u8,
    pub(crate) unit: FaultUnit,
    pub(crate) level: u8,
    pub(crate) unk_5: u8,
    pub(crate) read: bool,
    pub(crate) reason: FaultReason,
}

/// Device resources for this GPU instance.
pub(crate) struct Resources {
    dev: ARef<platform::Device>,
    sgx: Pin<KBox<Devres<IoMem<SGX_SIZE>>>>,
}

impl Resources {
    /// Map the required resources given our platform device.
    pub(crate) fn new(pdev: &platform::Device<Core>) -> Result<Resources> {
        let sgx_req = pdev.io_request_by_name(c_str!("sgx")).ok_or(EINVAL)?;
        let sgx_iomem = KBox::pin_init(sgx_req.iomap_sized::<SGX_SIZE>(), GFP_KERNEL)?;

        Ok(Resources {
            // SAFETY: This device does DMA via the UAT IOMMU.
            dev: pdev.into(),
            sgx: sgx_iomem,
        })
    }

    fn sgx_read32<const OFF: usize>(&self) -> u32 {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.relaxed().read32(OFF)
        } else {
            0
        }
    }

    /* Not yet used
    fn sgx_write32<OFF: usize>(&self, val: u32) {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.write32_relaxed(val, OFF)
        }
    }
    */

    fn sgx_read64<const OFF: usize>(&self) -> u64 {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.relaxed().read64(OFF)
        } else {
            0
        }
    }

    /* Not yet used
    fn sgx_write64<OFF: usize>(&self, val: u64) {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.write64_relaxed(val, OFF)
        }
    }
    */

    /// Read one 64-bit SGX register by runtime offset.
    fn sgx_read64_at(&self, off: usize) -> u64 {
        // `read64` bounds-checks at compile time and emits `rust_build_error`
        // for a runtime offset; `try_read64` does it at run time instead.
        if let Some(sgx) = self.sgx.try_access() {
            sgx.relaxed().try_read64(off).unwrap_or(0)
        } else {
            0
        }
    }

    /// Read the unit selector `0xc020` is windowed by (`fw:0x16888` masks it
    /// with 0x3f). Read-only here: the firmware owns this register.
    pub(crate) fn sgx_sel(&self) -> u32 {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.relaxed().try_read32(0x10398).unwrap_or(!0)
        } else {
            !0
        }
    }

    /// Read the GPU state register the firmware gates fragment work on.
    ///
    /// ⚠️ Only safe while the GPU is known to be powered -- see `dump_uma`.
    pub(crate) fn sgx_state(&self) -> u64 {
        self.sgx_read64_at(0xc020)
    }

    /// Sample the two populated units' `0xc020` *before* any job is submitted.
    ///
    /// `0xc020` is windowed by `0x10398`, and a bare read (what `sgx_state`
    /// does) returns whichever unit the firmware last selected -- which is why
    /// the post-kick diagnostic in `gpu.rs` reports 0. This selects units 0 and
    /// 1 explicitly, then puts the selector back.
    ///
    /// ⚠️ The firmware drives this selector too, so a sweep taken while the
    /// firmware is running can make *its* next read land on the wrong unit.
    /// Only two indices are touched and the original value is restored, to keep
    /// that window as short as possible. Behind `dump_uma_regs == 2`.
    ///
    /// The question this answers: `fw:0x675c` counts a unit quiescent only when
    /// `(v & 0x1d) == 0xd` *and* bits [31:24] are zero. We measure
    /// `0x0100000d` -- payload 1 -- after a hung job. If the payload is already
    /// 1 here, before any work is submitted, then the outstanding request is a
    /// boot-time condition and not something the job creates. LOG.md 32.41.287.
    pub(crate) fn dump_state_pre(&self, tag: &str) {
        let saved = self.sgx_sel();
        // 🔑 The latched selector, printed because the TA kick depends on it.
        // `fw:0x1ffe0` writes the kick to `0xc048` -- which sits in the same
        // 13-instance indirect block as `0xc020` -- and **never writes
        // `0x10398`** anywhere in that function. So the kick lands on whichever
        // unit was last selected by something else. If this reads anything but
        // a populated unit (0 or 1 on t6030), the kick went nowhere, which is
        // exactly what "hardware idle, no error" looks like. LOG.md 32.41.328.
        pr_info!("asahi/pre {} latched selector 0x10398 = {:#x}\n", tag, saved);
        for idx in 0..2u32 {
            self.sgx_write32_at(0x10398, idx);
            let v = self.sgx_read64_at(0xc020);
            pr_info!(
                "asahi/pre {} idx {} 0xc020 = {:#018x} (&0x1d = {:#04x} payload {:#04x}) 0xc050 = {:#018x}\n",
                tag,
                idx,
                v,
                v & 0x1d,
                (v >> 24) & 0xff,
                self.sgx_read64_at(0xc050),
            );
        }
        if saved != !0 {
            self.sgx_write32_at(0x10398, saved);
        }
    }

    /// Sample `0xc020` **without touching the `0x10398` selector**.
    ///
    /// This is the difference that matters. `dump_state_pre` writes the
    /// selector to visit units 0 and 1 and restores it, and even that narrow
    /// window measurably perturbs a run in flight (32.41.336). A *bare* read
    /// writes nothing at all, so it cannot. The cost is that it reports
    /// whichever unit the firmware last selected -- which is fine here,
    /// because the latched value is observed to be 0 or 1 across every run,
    /// and both are populated on t6030 (32.41.328).
    ///
    /// Answers what 32.41.327 could not: that entry read `0xc020 = 0` roughly
    /// 40 s after the kick and concluded the TA stage never starts, but an
    /// idle reading that late is equally consistent with "ran, did nothing,
    /// went idle". Only a sample taken *during* the job separates them, and
    /// only a non-perturbing one is trustworthy. LOG.md 32.41.361.
    pub(crate) fn poll_state_bare(&self, tag: &str, n: u32, ms: u32) {
        let mut last: Option<(u32, u64, u64, u64)> = None;
        for i in 0..n {
            let sel = self.sgx_sel();
            let st = self.sgx_state();
            let (ev, busy) = self.sgx_events();
            let cur = (sel, st, ev, busy);
            if last != Some(cur) {
                pr_info!(
                    "asahi/poll {} i={} t={}ms sel={:#x} 0xc020={:#018x} (&0x1d={:#04x} payload={:#04x}) 0xc128={:#018x} 0xc120={:#018x}\n",
                    tag, i, i * ms, sel, st, st & 0x1d, (st >> 24) & 0xff, ev, busy,
                );
                last = Some(cur);
            }
            if ms != 0 {
                fsleep(Delta::from_millis(ms as i64));
            }
        }
        pr_info!("asahi/poll {} done ({} samples @ {}ms)\n", tag, n, ms);
        self.dump_stall(tag);
    }

    /// Dump the registers the *firmware itself* reads, once the pipeline has
    /// stalled, for both populated units.
    ///
    /// The set is not guessed: it is every offset in `0xc000..0xd400` that the
    /// firmware image uses as the index of a load off `hw_base` (32.41.357).
    /// If the TA is parked waiting on something, the thing it is waiting on is
    /// almost certainly in here.
    ///
    /// Called only *after* the poll loop has finished, so the selector writes
    /// it needs cannot disturb the measurement that already happened. The
    /// selector is restored. LOG.md 32.41.363.
    pub(crate) fn dump_stall(&self, tag: &str) {
        const REGS: &[usize] = &[
            0xc020, 0xc028, 0xc038, 0xc048, 0xc050, 0xc058, 0xc060, 0xc068, 0xc070,
            0xc080, 0xc098, 0xc0b0, 0xc0b8, 0xc0c0, 0xc0c8, 0xc0d0, 0xc0d8, 0xc100,
            0xc120, 0xc128, 0xc138, 0xc140, 0xc148, 0xc238, 0xc800, 0xc868, 0xc870,
            0xc878, 0xc9f0, 0xcd08, 0xd020, 0xd028, 0xd060, 0xd088, 0xd090, 0xd098,
        ];
        let saved = self.sgx_sel();
        for idx in 0..2u32 {
            self.sgx_write32_at(0x10398, idx);
            for r in REGS {
                let v = self.sgx_read64_at(*r);
                if v != 0 {
                    pr_info!("asahi/stall {} unit{} {:#06x} = {:#018x}\n", tag, idx, *r, v);
                }
            }
        }
        if saved != !0 {
            self.sgx_write32_at(0x10398, saved);
        }
        pr_info!("asahi/stall {} done (selector restored to {:#x})\n", tag, saved);
    }

    /// The firmware's own USC diagnostic, replayed from the CPU. `fw:0x35a24`
    /// ("USC Pipe %u State Overview", reached only from its crash-dump path,
    /// which never fires on our runs) iterates a pipe mask and, per pipe:
    /// writes the index to `0xa010`, reads `AGX_CR_USC_{VDM,PDM,CDM}_SERV`
    /// (`0xa088/0xa068/0xa0a8`); writes it to `0xa000`, reads
    /// `AGX_CR_USC_{VDM,PDM,CDM}_DEBUG_STATUS` (`0xa090/0xa070/0xa0b0`).
    /// Both selectors are restored. LOG.md 32.41.533.
    pub(crate) fn dump_usc(&self, tag: &str, n: u32) {
        let s10 = self.sgx_read32_at(0xa010);
        let s00 = self.sgx_read32_at(0xa000);
        for p in 0..n {
            self.sgx_write32_at(0xa010, p);
            let serv = [self.sgx_read64_at(0xa088), self.sgx_read64_at(0xa068), self.sgx_read64_at(0xa0a8)];
            self.sgx_write32_at(0xa000, p);
            let dbg = [self.sgx_read64_at(0xa090), self.sgx_read64_at(0xa070), self.sgx_read64_at(0xa0b0)];
            pr_info!(
                "asahi/usc {} pipe{} SERV vdm {:#x} pdm {:#x} cdm {:#x} | DEBUG vdm {:#x} pdm {:#x} cdm {:#x}\n",
                tag, p, serv[0], serv[1], serv[2], dbg[0], dbg[1], dbg[2]
            );
        }
        self.sgx_write32_at(0xa010, s10);
        self.sgx_write32_at(0xa000, s00);
        pr_info!("asahi/usc {} done (sel restored {:#x} {:#x})\n", tag, s10, s00);
    }

    /// LOG.md 32.41.558: read unit-internal registers through the firmware's
    /// indirect debug window. The firmware writes `unit << 48 | 0x400 << 32 | reg`
    /// to `SGX + 0x4000 + T[agent]` and reads the value at `+8`, with
    /// T = {0xd0,0xe0,0xc0,0xa0,0xb0,0xf0,0x80} (fw:0x50bf4). Slot 0x90 is used by
    /// no firmware agent, so the host takes it. Known-answer first: the
    /// firmware's own UMA check (fw:0xb788) expects unit 0x20 reg 0xa5a8 == 0x20.
    /// 32.41.560: tight poll of the per-pipe USC state across the vertex job.
    /// Waits (1us steps, <=4ms) for the block to power up (0xc020 != 0, safe
    /// to read gated), then spins `iters` times rotating the pipe select
    /// (0xa010/0xa000) over 0..32 (the firmware walks a pipe *bitmask*, [[0x1a4370]+0x1500], so an 18-core part has pipes >= 16) and logs every value that differs from the
    /// last one seen for that pipe (VDM_SERV 0xa088, VDM_DEBUG_STATUS 0xa090).
    /// 32.41.598: read-only sampler for the TA event block after a vertex kick.
    /// TA events 4/3 (fw handlers 0x46ec / 0x49a8) are acked within microseconds,
    /// so sample the status words continuously and log every change with the
    /// 24 MHz counter (same clock as the firmware ktrace).
    pub(crate) fn ta_poll(&self, iters: u32) {
        // ⛔ no 0xc128 (a CPU read wedges the AP) and nothing untested beyond
        // the 0x10a event block + 0xc020.
        // ta_poll=N: 7 regs; ta_poll_wide=1: the whole 0x10a00..0x10af8 block + UMA 0x140xx/0x143xx.
        const NARROW: [usize; 7] = [0x10a10, 0x10a30, 0x10a40, 0x10a50, 0x10a90, 0x10a00, 0xc020];
        let mut regs_buf = [0usize; 64];
        let mut nr = 0usize;
        if *crate::module_parameters::ta_poll_wide.value() != 0 {
            let mut o = 0x10a00;
            while o < 0x10b00 { regs_buf[nr] = o; nr += 1; o += 8; }
            for &o in [0x14000usize, 0x14008, 0x14010, 0x14018, 0x14020, 0x14028, 0x14030, 0x14038,
                       0x14300, 0x14308, 0x14310, 0x14318, 0xc020, 0xa728].iter() {
                regs_buf[nr] = o; nr += 1;
            }
        } else {
            for &o in NARROW.iter() { regs_buf[nr] = o; nr += 1; }
        }
        let REGS = &regs_buf[..nr];
        let cnt = || -> u64 {
            let v: u64;
            // SAFETY: reading the virtual counter has no side effects.
            unsafe { core::arch::asm!("isb", "mrs {}, cntvct_el0", out(reg) v) };
            v
        };
        let t_door = cnt();
        let mut wait = 0u32;
        while self.sgx_state() == 0 && wait < 4000 {
            kernel::time::delay::fsleep(kernel::time::Delta::from_micros(1));
            wait += 1;
        }
        let mut last = [u64::MAX; 64];
        let mut rec = [(0u64, 0u8, 0u64); 160];
        let mut n = 0usize;
        for _ in 0..iters {
            for (k, &r) in REGS.iter().enumerate() {
                let v = self.sgx_read64_at(r);
                if v != last[k] {
                    if n < rec.len() {
                        rec[n] = (cnt(), k as u8, v);
                        n += 1;
                    }
                    last[k] = v;
                }
            }
        }
        dev_info!(
            self.dev.as_ref(),
            "asahi/tapoll: powered after {}us, {} iters, {} changes, door {:#x}\n",
            wait, iters, n, t_door
        );
        for r in rec.iter().take(n) {
            dev_info!(
                self.dev.as_ref(),
                "asahi/tapoll: t {:#x} reg {:#07x} = {:#018x}\n",
                r.0, REGS[r.1 as usize], r.2
            );
        }
    }

    pub(crate) fn usc_poll(&self, iters: u32) {
        // cntvct_el0 = the 24 MHz system counter the firmware ktrace "t" uses.
        let cnt = || -> u64 {
            let v: u64;
            // SAFETY: reading the virtual counter has no side effects.
            unsafe { core::arch::asm!("isb", "mrs {}, cntvct_el0", out(reg) v) };
            v
        };
        let t_door = cnt();
        let mut wait = 0u32;
        while self.sgx_state() == 0 && wait < 4000 {
            kernel::time::delay::fsleep(kernel::time::Delta::from_micros(1));
            wait += 1;
        }
        let mut last = [(0u64, 0u64); 32];
        let mut rec = [(0u32, 0u32, 0u64, 0u64); 48];
        let mut n = 0usize;
        let mut nz = 0u32;
        let t0 = cnt();
        for i in 0..iters {
            let p = i & 31;
            self.sgx_write32_at(0xa010, p);
            self.sgx_write32_at(0xa000, p);
            let v = (self.sgx_read64_at(0xa088), self.sgx_read64_at(0xa090));
            if v.0 | v.1 != 0 {
                nz += 1;
            }
            if v != last[p as usize] {
                if n < rec.len() {
                    rec[n] = (i, p, v.0, v.1);
                    n += 1;
                }
                last[p as usize] = v;
            }
        }
        let t1 = cnt();
        dev_info!(
            self.dev.as_ref(),
            "asahi/uscpoll: powered after {}us, {} iters, {} non-zero samples, {} changes; cnt door {:#x} start {:#x} end {:#x}\n",
            wait, iters, nz, n, t_door, t0, t1
        );
        for r in rec.iter().take(n) {
            dev_info!(
                self.dev.as_ref(),
                "asahi/uscpoll: iter {} pipe {} VDM_SERV {:#x} VDM_DBG {:#x}\n",
                r.0, r.1, r.2, r.3
            );
        }
    }

    /// 32.41.572d: the compute twin of `usc_poll`. Spins `iters` times after
    /// the compute doorbell, rotating the pipe select over 0..16 and logging
    /// every change of CDM_SERV (0xa0a8) / CDM_DEBUG_STATUS (0xa0b0) per pipe
    /// and of the bare 0xc020, each with its cntvct time.
    pub(crate) fn cdm_poll(&self, iters: u32) {
        let cnt = || -> u64 {
            let v: u64;
            // SAFETY: reading the virtual counter has no side effects.
            unsafe { core::arch::asm!("isb", "mrs {}, cntvct_el0", out(reg) v) };
            v
        };
        let t_door = cnt();
        let mut wait = 0u32;
        while self.sgx_state() == 0 && wait < 4000 {
            kernel::time::delay::fsleep(kernel::time::Delta::from_micros(1));
            wait += 1;
        }
        let mut last = [(0u64, 0u64); 32];
        let mut last_st = !0u64;
        let mut last_a040 = !0u64;
        let mut last_prc = (!0u64, !0u64);
        let mut rec = [(0u64, 0u32, 0u64, 0u64); 64];
        let mut n = 0usize;
        let mut nz = 0u32;
        for i in 0..iters {
            let p = i & 31;
            self.sgx_write32_at(0xa010, p);
            self.sgx_write32_at(0xa000, p);
            let v = (self.sgx_read64_at(0xa0a8), self.sgx_read64_at(0xa0b0));
            if v.0 | v.1 != 0 {
                nz += 1;
            }
            if v != last[p as usize] {
                if n < rec.len() {
                    rec[n] = (cnt(), p, v.0, v.1);
                    n += 1;
                }
                last[p as usize] = v;
            }
            if p == 0 {
                // 32.41.584: pipe 0's 0xa040 (USC pipe info) drops to 0 part-way
                // through a sweep -- log when, against the 0xc020 transitions.
                let a = self.sgx_read64_at(0xa040);
                if a != last_a040 {
                    if n < rec.len() {
                        rec[n] = (cnt(), 98, a, 0);
                        n += 1;
                    }
                    last_a040 = a;
                }
                // .595: FenderRegsPRC -- the hardware's current GPU perf state
                // (AGFHALWaitForPerfState polls 0xd4c004 for `idx | 0x10`).
                let prc = (
                    ((self.sgx_read32_at(0xd4c004) as u64) << 32) | self.sgx_read32_at(0xd4c000) as u64,
                    self.sgx_read32_at(0xd4c008) as u64,
                );
                if prc != last_prc {
                    if n < rec.len() {
                        rec[n] = (cnt(), 97, prc.0, prc.1);
                        n += 1;
                    }
                    last_prc = prc;
                }
                let st = self.sgx_state();
                if st != last_st {
                    if n < rec.len() {
                        rec[n] = (cnt(), 99, st, 0);
                        n += 1;
                    }
                    last_st = st;
                }
            }
        }
        let t1 = cnt();
        dev_info!(
            self.dev.as_ref(),
            "asahi/cdmpoll: powered after {}us, {} iters, {} non-zero, {} changes; cnt door {:#x} end {:#x}\n",
            wait, iters, nz, n, t_door, t1
        );
        for r in rec.iter().take(n) {
            if r.1 == 97 {
                dev_info!(self.dev.as_ref(), "asahi/cdmpoll: t {:#x} PRC 0xd4c004:0xd4c000 = {:#x} 0xd4c008 = {:#x}\n", r.0, r.2, r.3);
            } else if r.1 == 98 {
                dev_info!(self.dev.as_ref(), "asahi/cdmpoll: t {:#x} pipe0 0xa040 = {:#x}\n", r.0, r.2);
            } else if r.1 == 99 {
                dev_info!(self.dev.as_ref(), "asahi/cdmpoll: t {:#x} 0xc020 = {:#x}\n", r.0, r.2);
            } else {
                dev_info!(
                    self.dev.as_ref(),
                    "asahi/cdmpoll: t {:#x} pipe {} CDM_SERV {:#x} CDM_DBG {:#x}\n",
                    r.0, r.1, r.2, r.3
                );
            }
        }
    }

    /// 32.41.589: xnu's hard-coded AGX tunables. m1n1 applies them for
    /// t8103/t8112/t600x at SGX + 0x1000000 (tunables_static.c: "required to make
    /// coprocessor TLB coherency work properly") and does nothing on t6030. Only
    /// called while a job runs (GPU powered): a read of a gated SGX range SErrors.
    pub(crate) fn agx_tunables(&self, apply: bool) {
        const BASE: u64 = 0x2_9100_0000;
        // m1n1 t600x_agx_entry: (offset, mask, value)
        const T: [(usize, u32, u32); 12] = [
            (0x0, 0x1, 0x1), (0x10, 0xfff0000, 0xd0000), (0x14, 0x3, 0x1), (0x18, 0x3, 0x1),
            (0x1c, 0x3, 0x3), (0x20, 0x3, 0x3), (0x24, 0x3, 0x3), (0x28, 0x3, 0x3),
            (0x2c, 0x3, 0x3), (0x400, 0x400103ff, 0x40010001), (0x600, 0x1ffffff, 0x1ffffff),
            (0x800, 0x100, 0x100),
        ];
        // SAFETY: Apple MMIO, posted-write mapping as the platform requires.
        let p = unsafe { kernel::bindings::ioremap_np(BASE, 0x4000) } as *mut u8;
        if p.is_null() {
            dev_err!(self.dev.as_ref(), "asahi/agxtun: ioremap failed\n");
            return;
        }
        for (off, mask, val) in T {
            // SAFETY: `off + 4 <= 0x4000`, mapped above.
            let a = unsafe { p.add(off) } as *mut kernel::ffi::c_void;
            let before = unsafe { kernel::bindings::readl(a) };
            let mut after = before;
            if apply {
                unsafe { kernel::bindings::writel((before & !mask) | val, a) };
                after = unsafe { kernel::bindings::readl(a) };
            }
            dev_info!(
                self.dev.as_ref(),
                "asahi/agxtun: +{:#05x} {:#010x} -> {:#010x} (want mask {:#x} val {:#x})\n",
                off, before, after, mask, val
            );
        }
        // agx_tun=3 (apply == false path is also taken): full read-only dump.
        if *crate::module_parameters::agx_tun.value() == 3 {
            for off in (0..0x1000usize).step_by(4) {
                // SAFETY: in the 0x4000 mapping above.
                let v = unsafe { kernel::bindings::readl(p.add(off) as *mut kernel::ffi::c_void) };
                if v != 0 {
                    dev_info!(self.dev.as_ref(), "asahi/agxtun dump +{:#05x} = {:#010x}\n", off, v);
                }
            }
        }
        // SAFETY: mapped above.
        unsafe { kernel::bindings::iounmap(p as *mut kernel::ffi::c_void) };
    }

    /// 32.41.584: the kext's `AGXArmFenderChinook::{en,dis}ableDynamicFenderGating`:
    /// GFX 0xd06030 &= ~6 enables dynamic gating, |= 4 disables it. Log the
    /// value and optionally disable.
    pub(crate) fn fender_gating(&self, disable: bool) {
        let before = self.sgx_read32_at(0xd06030);
        if disable {
            self.sgx_write32_at(0xd06030, before | 4);
        }
        let after = self.sgx_read32_at(0xd06030);
        dev_info!(
            self.dev.as_ref(),
            "asahi/fender: 0xd06030 {:#x} -> {:#x} (disable={})\n",
            before, after, disable
        );
    }

    /// 32.41.573c: the kext's `AGXAcceleratorG15::printUSCSlotUsage` protocol.
    /// Per pipe: pipe -> 0xa000, 0xa040 bits 16..17; then per slot: slot ->
    /// 0x100e8 (VDM) / 0x100e0 (PDM) / 0x100f0 (CDM), read 0xa090 / 0xa070 /
    /// 0xa0b0. Logs only non-zero slots, plus a per-pipe summary.
    pub(crate) fn usc_slot_sweep(&self, tag: &str) {
        let mut total = 0u32;
        for pipe in 0..32u32 {
            self.sgx_write32_at(0xa000, pipe);
            let a040 = self.sgx_read64_at(0xa040);
            let mut nz = 0u32;
            for (sel, rd, name) in [(0x100e8usize, 0xa090usize, "vdm"), (0x100e0, 0xa070, "pdm"), (0x100f0, 0xa0b0, "cdm")] {
                for slot in 0..64u32 {
                    self.sgx_write32_at(sel, slot);
                    let v = self.sgx_read64_at(rd);
                    if v != 0 {
                        nz += 1;
                        if total < 64 {
                            dev_info!(
                                self.dev.as_ref(),
                                "asahi/uscslot {} pipe {} {} slot {} = {:#x}\n",
                                tag, pipe, name, slot, v
                            );
                        }
                        total += 1;
                    }
                }
            }
            if a040 != 0 || nz != 0 {
                dev_info!(
                    self.dev.as_ref(),
                    "asahi/uscslot {} pipe {} a040 {:#x} nonzero slots {}\n",
                    tag, pipe, a040, nz
                );
            }
        }
        // 32.41.587: 0x100e8/0x100e0/0x100f0 are in the firmware's CRE list with
        // value 0x100 -- restore them; the sweep used them as slot selectors.
        for sel in [0x100e8usize, 0x100e0, 0x100f0] {
            self.sgx_write32_at(sel, 0x100);
        }
        dev_info!(self.dev.as_ref(), "asahi/uscslot {} total nonzero {}\n", tag, total);
    }

    pub(crate) fn dump_dbgwin(&self, tag: &str) {
        // .559: the firmware's "USC Pipe %u State Overview" (fw:0x35a64) is
        // direct MMIO, not the window: pipe -> 0xa010, read 0xa088/0xa068/0xa0a8
        // (VDM/PDM/CDM_SERV); pipe -> 0xa000, read 0xa090/0xa070/0xa0b0
        // (VDM/PDM/CDM_DEBUG_STATUS).
        // Positive controls: selector readback and two registers the firmware
        // prints (GDIR_DIRECTOR_CTRL 0xd028, DIRECTOR_CONFIG 0x10190).
        self.sgx_write32_at(0xa010, 3);
        self.sgx_write32_at(0xa000, 5);
        dev_info!(
            self.dev.as_ref(),
            "asahi/uscpipe {} ctl a010={:#x} a000={:#x} d028={:#x} 10190={:#x}\n",
            tag,
            self.sgx_read32_at(0xa010),
            self.sgx_read32_at(0xa000),
            self.sgx_read64_at(0xd028),
            self.sgx_read64_at(0x10190)
        );
        // .561: USC config the firmware writes at boot (CRE-init region
        // fw:0x7ac8..0xbeb0). Intended: 0x1a040/0x1a048/0x1a050 = 4, 0xa498 = 1.
        // Zero here = lost across a power cycle and not restored.
        for r in [
            0xa020usize, 0xa498, 0xa728, 0x1a040, 0x1a048, 0x1a050, 0x1a100, 0x1a108, 0x1a110,
            0x1a118, 0x1a468, 0x10790, 0x10798, 0x107a0,
        ] {
            dev_info!(
                self.dev.as_ref(),
                "asahi/uscinit {} {:#x} = {:#x}\n",
                tag,
                r,
                self.sgx_read64_at(r)
            );
        }
        // .563: CRE register region (io_mapping CRERegisters = FENDER+0x50000
        // = SGX+0xd50000, 0x8000). The firmware logs "saved 548 registers";
        // its list records are {val, 0, 0x8001_0000|reg} (fw:0x8f3c).
        {
            let mut nz = 0u32;
            let mut off = 0usize;
            while off < 0x8000 {
                let w = [
                    self.sgx_read32_at(0xd50000 + off),
                    self.sgx_read32_at(0xd50000 + off + 4),
                    self.sgx_read32_at(0xd50000 + off + 8),
                    self.sgx_read32_at(0xd50000 + off + 12),
                ];
                if w != [0; 4] {
                    nz += 1;
                    dev_info!(
                        self.dev.as_ref(),
                        "asahi/cre {:#06x}: {:#010x} {:#010x} {:#010x} {:#010x}\n",
                        off, w[0], w[1], w[2], w[3]
                    );
                }
                off += 16;
            }
            dev_info!(self.dev.as_ref(), "asahi/cre {} non-zero records\n", nz);
        }
        // .562: are they R/W? Write the firmware's intended value, read back.
        for (r, v) in [(0x1a040usize, 4u64), (0x1a048, 4), (0x1a050, 4), (0xa498, 1)] {
            self.sgx_write64_at(r, v);
            dev_info!(
                self.dev.as_ref(),
                "asahi/uscinit {} wrote {:#x} <- {:#x}, reads {:#x}\n",
                tag,
                r,
                v,
                self.sgx_read64_at(r)
            );
        }
        for pipe in 0..32u32 {
            self.sgx_write32_at(0xa010, pipe);
            let serv = [
                self.sgx_read64_at(0xa088),
                self.sgx_read64_at(0xa068),
                self.sgx_read64_at(0xa0a8),
            ];
            self.sgx_write32_at(0xa000, pipe);
            let dbg = [
                self.sgx_read64_at(0xa090),
                self.sgx_read64_at(0xa070),
                self.sgx_read64_at(0xa0b0),
            ];
            dev_info!(
                self.dev.as_ref(),
                "asahi/uscpipe {} pipe {} SERV vdm {:#x} pdm {:#x} cdm {:#x} DBG vdm {:#x} pdm {:#x} cdm {:#x}\n",
                tag, pipe, serv[0], serv[1], serv[2], dbg[0], dbg[1], dbg[2]
            );
        }
        // .558: slot 0x90 read 0 for the KAT. Try every slot (the firmware is
        // idle post-job), and use the first that passes.
        // .558d: data reads 0x100 when idle and 0 right after a selector write;
        // poll until it changes (bounded), logging the transitions once.
        let rd_dbg = |sel: usize, unit: u64, reg: u64, log: bool| -> u64 {
            let want = (unit << 48) | (0x400u64 << 32) | reg;
            self.sgx_write64_at(sel, want);
            let latched = self.sgx_read64_at(sel);
            let mut last = self.sgx_read64_at(sel + 8);
            let mut i = 0u32;
            while i < 200 && last == 0 {
                let v = self.sgx_read64_at(sel + 8);
                if v != last {
                    if log {
                        pr_info!("asahi/dbgwin   sel {:#x} poll {}: {:#x} -> {:#x}\n", sel, i, last, v);
                    }
                    last = v;
                }
                i += 1;
            }
            if log {
                pr_info!("asahi/dbgwin   sel {:#x} wrote {:#x} latched {:#x} final {:#x}\n", sel, want, latched, last);
            }
            last
        };
        let rd = |sel: usize, unit: u64, reg: u64| -> u64 { rd_dbg(sel, unit, reg, false) };
        // .558g: the unit only answers when awake (.558f: 0 = no response;
        // a valid answer carries flags in bits 32+, e.g. 0x60_0040_0020).
        // Bracketed read: KAT ok, target, KAT ok -- retried up to N times.
        let release = |sel: usize| { self.sgx_write64_at(sel, 0); let _ = self.sgx_read64_at(sel + 8); };
        let one = |unit: u64, reg: u64| -> u64 { release(0x4080); rd(0x4080, unit, reg) };
        let kat_ok = |v: u64| v & 0xffff == 0x20 && v >> 32 != 0;
        let br = |unit: u64, reg: u64| -> Option<(u64, u32)> {
            let mut n = 0u32;
            while n < 300 {
                n += 1;
                if !kat_ok(one(0x20, 0xa5a8)) { continue; }
                let v = one(unit, reg);
                if kat_ok(one(0x20, 0xa5a8)) { return Some((v, n)); }
            }
            None
        };
        let mut hits = 0;
        for unit in [0x2u64, 0x202, 0x0, 0x102, 0x302] {
            for reg in [0xa040u64, 0xa088, 0xa090, 0xa068, 0xa070, 0xa0a8, 0xa0b0] {
                match br(unit, reg) {
                    Some((v, n)) => { hits += 1; pr_info!("asahi/dbgwin {} unit {:#x} reg {:#x} = {:#x} (try {})\n", tag, unit, reg, v, n); }
                    None => pr_info!("asahi/dbgwin {} unit {:#x} reg {:#x}: no bracketed answer\n", tag, unit, reg),
                }
                if hits == 0 && reg == 0xa040 && unit == 0x2 { /* keep going; each reg is independent */ }
            }
        }
        release(0x4080);
        pr_info!("asahi/dbgwin {} done, {} bracketed values\n", tag, hits);
    }

    /// Read one 32-bit SGX register by runtime offset.
    fn sgx_read32_at(&self, off: usize) -> u32 {
        if let Some(sgx) = self.sgx.try_access() {
            sgx.relaxed().try_read32(off).unwrap_or(!0)
        } else {
            !0
        }
    }

    /// Read the scheduler's pending-event bitmap and the per-pipe busy mask.
    ///
    /// `0xc128` is the wake signal the firmware's scheduler loop turns on:
    /// `fw:0x4270`/`fw:0x4328` re-run all four pipe passes while it is
    /// non-zero, and the idle path (`fw:0x7798`, `fw:0x53a0`) only sleeps when
    /// it is zero. It is read-only to the firmware -- 3 loads, no stores. If a
    /// command completing never sets a bit here, the scheduler is never
    /// re-entered and a deferred fragment is never re-offered. `0xc120` is the
    /// matching per-pipe busy bitmap (`fw:0x186bc` tests it against the mask it
    /// just wrote to the `0xc140` run-enable). LOG.md 32.41.129.
    pub(crate) fn sgx_events(&self) -> (u64, u64) {
        (self.sgx_read64_at(0xc128), self.sgx_read64_at(0xc120))
    }

    /// Write one 32-bit SGX register by runtime offset.
    fn sgx_write32_at(&self, off: usize, val: u32) {
        if let Some(sgx) = self.sgx.try_access() {
            let _ = sgx.relaxed().try_write32(val, off);
        }
    }

    /// Write one 64-bit SGX register by runtime offset.
    fn sgx_write64_at(&self, off: usize, val: u64) {
        if let Some(sgx) = self.sgx.try_access() {
            let _ = sgx.relaxed().try_write64(val, off);
        }
    }

    /// Emit the firmware's own "service this unit request" command, from the
    /// driver, and report whether the unit leaves state 3.
    ///
    /// `0xc020[31:24]` is a bitfield of 2-bit request codes (32.41.288: our job
    /// puts a single code 1 there on both units and nothing ever clears it).
    /// `fw:0x18924` and `fw:0x255f4` service them by writing, per non-zero code,
    ///
    /// ```text
    ///   0x4000 = slot>>2 << 59 | slot&3 << 57 | (code-1) << 43 | 1 << 42 | reg
    ///   0x4008 = 0   (fw:0x1899c)   or 1   (fw:0x2564c)
    /// ```
    ///
    /// where `reg` comes from the table at `fw:0x50450` -- `0x10710` for the
    /// pipe the sampler runs on. LOG.md 32.41.291.
    ///
    /// The firmware never reaches that code here, because the handler that
    /// samples units into its request table is never dispatched (32.41.290).
    /// This does by hand what that path would have done, to find out whether
    /// servicing the request is what moves the unit out of state 3 and into the
    /// state `FinalizeVertex` is waiting for.
    ///
    /// ⚠️ Post-mortem only: this writes a live command port, so it must not run
    /// while a job is in flight. Behind `unit_cmd_probe`.
    pub(crate) fn unit_cmd_probe(&self, second: u64) {
        let saved = self.sgx_sel();
        for idx in 0..2u32 {
            self.sgx_write32_at(0x10398, idx);
            let before = self.sgx_read64_at(0xc020);
            let mut payload = (before >> 24) & 0xff;
            if payload == 0 {
                pr_info!("asahi/unitcmd idx {} 0xc020 {:#018x} no payload\n", idx, before);
                continue;
            }
            let mut slot: u64 = 0;
            while payload != 0 {
                let code = payload & 3;
                if code != 0 {
                    let word = ((slot >> 2) << 59)
                        | ((slot & 3) << 57)
                        | ((code - 1) << 43)
                        | (1u64 << 42)
                        | 0x10710;
                    self.sgx_write64_at(0x4000, word);
                    self.sgx_write64_at(0x4008, second);
                    pr_info!("asahi/unitcmd idx {} slot {} code {} -> 0x4000 = {:#018x} 0x4008 = {:#x}\n",
                             idx, slot, code, word, second);
                }
                payload >>= 2;
                slot += 1;
            }
            let after = self.sgx_read64_at(0xc020);
            pr_info!(
                "asahi/unitcmd idx {} 0xc020 {:#018x} -> {:#018x} (state {} payload {:#04x})\n",
                idx, before, after, (after >> 2) & 7, (after >> 24) & 0xff,
            );
        }
        if saved != !0 {
            self.sgx_write32_at(0x10398, saved);
        }
    }

    /// Read the PRC (GPU perf-state controller) status pair.
    ///
    /// `AGFHALWaitForPerfState` (`fw:0x283c0`) writes `state | 0x10` to
    /// SGX + 0xd4c004 and then polls until `(reg & 0x1f) == state`, i.e. until
    /// the hardware clears bit 4. It never does, and the firmware spins there
    /// forever. SGX + 0xd4c000 is the current state (`fw:0x26590` reads it
    /// masked with 0x1f). This is io_mapping 38, `FenderRegsPRC` =
    /// `FENDER + 0x4c000` = `SGX + 0xd4c000`, inside the 16 MB window this
    /// driver already maps. LOG.md 32.41.123.
    pub(crate) fn prc(&self) -> (u32, u32) {
        if let Some(sgx) = self.sgx.try_access() {
            let r = sgx.relaxed();
            (
                r.try_read32(0xd4c000).unwrap_or(!0),
                r.try_read32(0xd4c004).unwrap_or(!0),
            )
        } else {
            (!0, !0)
        }
    }

    /// Read the GPU's free-running 54-bit timestamp counter.
    ///
    /// `fw:0x26c44` reads 0xc028 (and 0xc038), masks both with
    /// `0x3fffffffffffff` and adds a bias to convert them into the system
    /// timebase -- so this is a clock, not a work counter. If it stops
    /// advancing the GPU's clock domain has stopped. LOG.md 32.41.121.
    ///
    /// ⚠️ Only safe while the GPU is known to be powered -- see `dump_uma`.
    pub(crate) fn sgx_timestamp(&self) -> u64 {
        self.sgx_read64_at(0xc028)
    }

    /// Dump the UMA page-manager registers.
    ///
    /// The fault registers this driver already reads live at 0xd8c0/0xd8c8 in
    /// this same window, and the numbers the firmware and the job register
    /// lists use (0xd418, 0xa728, 0x14308, ...) are offsets into it -- so the
    /// base the page manager was actually handed can just be read back instead
    /// of inferred. The six pool-base numbers are the firmware's own table at
    /// `fw:0x50bdc`; 0xa728 is `USC_TP_GLOBAL_RESOURCE_CFG0`, whose bits 13:8
    /// are MAX_TGMEM. Called only from the fault path, where the GPU is known
    /// to be powered. LOG.md 32.41.84.
    pub(crate) fn dump_uma(&self) {
        const POOL_BASE_REGS: [usize; 6] = [0x14308, 0x14048, 0x14028, 0x14318, 0x14080, 0x14070];
        for off in POOL_BASE_REGS {
            pr_info!(
                "asahi/uma reg {:#07x} = {:#018x}\n",
                off,
                self.sgx_read64_at(off)
            );
        }
        // The GPU state register the fragment is gated on. `fw:0x166fc` and
        // `fw:0x15824` read this and require `(v & 0x1e) == 0x1a` before they
        // will start fragment work; other sites test `& 0x1c == 0x14`,
        // `& 0x1e == 0x18` and `& 0x1c == 0x0c`, so bits [4:2] are a 3-bit
        // state and bit 1 a flag. 0xc0b0/0xc0c8/0xc0d8 are the neighbours
        // `fw:0x5dd8` reads in the same breath. LOG.md 32.41.114.
        // Every offset in this block the firmware is known to touch, so none of
        // them is a blind poke at an unmapped register. `0xc020` is the one the
        // fragment is gated on; `0xc050` bit 10 gates the handshake that
        // `fw:0x16fc0` and `fw:0x132f0` drive by writing a mask to `0xc060`.
        // LOG.md 32.41.118.
        for off in [
            0xc000usize, 0xc008, 0xc020, 0xc028, 0xc030, 0xc038, 0xc048, 0xc050, 0xc058, 0xc060,
            0xc068, 0xc070, 0xc080, 0xc098, 0xc0b0, 0xc0b8, 0xc0c0, 0xc0c8, 0xc0d0, 0xc0d8, 0xc120,
            0xc128, 0xc138, 0xc140, 0xc148,
        ] {
            let v = self.sgx_read64_at(off);
            pr_info!("asahi/gpu reg {:#07x} = {:#018x}\n", off, v);
        }
        // 0xc020 is a *windowed* register: `fw:0x10398` holds a 6-bit index
        // (`fw:0x16888` masks the read with 0x3f) and every firmware site that
        // reads 0xc020 writes that index first -- `fw:0x6754`, `fw:0x34dc`,
        // `fw:0x255b8`. A bare read of 0xc020 therefore reports whichever unit
        // the firmware happened to select last, which is why the value looked
        // both constant and uninterpretable. Sweep the index instead.
        //
        // Only done from the post-mortem paths (fault, timeout, teardown): the
        // firmware also drives this index, so writing it while the firmware is
        // running would make *its* next read return the wrong unit's state.
        // LOG.md 32.41.121.
        for idx in 0..16u32 {
            self.sgx_write32_at(0x10398, idx);
            let v = self.sgx_read64_at(0xc020);
            let c050 = self.sgx_read64_at(0xc050);
            pr_info!(
                "asahi/gpu idx {:2} 0xc020 = {:#018x} (&0x1d = {:#04x}) 0xc050 = {:#018x}\n",
                idx,
                v,
                v & 0x1d,
                c050
            );
        }
        let (p0, p4) = self.prc();
        pr_info!("asahi/prc 0xd4c000 = {:#010x}  0xd4c004 = {:#010x}\n", p0, p4);
        for off in [0xa728usize, 0x14010, 0x14000, 0x14018, 0x14020] {
            pr_info!(
                "asahi/uma reg {:#07x} = {:#018x}\n",
                off,
                self.sgx_read64_at(off)
            );
        }
    }

    /// Initialize the MMIO registers for the GPU.
    pub(crate) fn init_mmio(&self) -> Result {
        // Nothing to do for now...

        Ok(())
    }

    /// Start the ASC coprocessor CPU.
    pub(crate) fn start_cpu(pdev: &platform::Device<Core>) -> Result {
        let asc_req = pdev.io_request_by_name(c_str!("asc")).ok_or(EINVAL)?;
        let asc_iomem = KBox::pin_init(asc_req.iomap_sized::<ASC_CTL_SIZE>(), GFP_KERNEL)?;
        let res = asc_iomem.access(pdev.as_ref())?.relaxed();

        let val = res.read32(CPU_CONTROL);
        res.write32(val | CPU_RUN, CPU_CONTROL);
        Ok(())
    }

    /// Get the GPU identification info from registers.
    ///
    /// See [`hw::GpuIdConfig`] for the result.
    pub(crate) fn get_gpu_id(&self) -> Result<hw::GpuIdConfig> {
        let id_version = self.sgx_read32::<ID_VERSION>();
        let id_unk08 = self.sgx_read32::<ID_UNK08>();
        let id_counts_1 = self.sgx_read32::<ID_COUNTS_1>();
        let id_counts_2 = self.sgx_read32::<ID_COUNTS_2>();
        let id_unk18 = self.sgx_read32::<ID_UNK18>();
        let id_clusters = self.sgx_read32::<ID_CLUSTERS>();

        dev_info!(
            self.dev.as_ref(),
            "GPU ID registers: {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}\n",
            id_version,
            id_unk08,
            id_counts_1,
            id_counts_2,
            id_unk18,
            id_clusters
        );

        let gpu_gen = (id_version >> 24) & 0xff;

        let mut core_mask_regs = KVec::new();

        let num_clusters = match gpu_gen {
            4 | 5 => {
                // G13 | G14G
                core_mask_regs.push(self.sgx_read32::<CORE_MASK_0>(), GFP_KERNEL)?;
                core_mask_regs.push(self.sgx_read32::<CORE_MASK_1>(), GFP_KERNEL)?;
                (id_clusters >> 12) & 0xff
            }
            // 6 = G14X, 7 = G15X. Read off t6030 (M3 Pro): ID_VERSION at
            // 0xd04000 = 0x07031100 -- generation byte 7, variant 3 (S), rev
            // 0x11 (B1) -- and the core mask lives at the G14X location,
            // CORE_MASKS_G14X = 0x0007fdff, popcount 18, exactly the 18-core
            // M3 Pro. m3-gpu-display/LOG.md 21.1.
            6 | 7 => {
                // G14X / G15X
                core_mask_regs.push(self.sgx_read32::<CORE_MASKS_G14X>(), GFP_KERNEL)?;
                core_mask_regs.push(self.sgx_read32::<{ CORE_MASKS_G14X + 4 }>(), GFP_KERNEL)?;
                core_mask_regs.push(self.sgx_read32::<{ CORE_MASKS_G14X + 8 }>(), GFP_KERNEL)?;
                // Clusters per die * num dies
                ((id_counts_1 >> 8) & 0xff) * ((id_counts_1 >> 16) & 0xf)
            }
            a => {
                dev_err!(self.dev.as_ref(), "Unknown GPU generation {}\n", a);
                return Err(ENODEV);
            }
        };

        let mut core_masks_packed = KVec::new();
        core_masks_packed.extend_from_slice(&core_mask_regs, GFP_KERNEL)?;

        // {:#x?} on a KVec formats one element per line and dev_info only emits
        // the first, so this used to print a bare "Core masks: [".
        for (i, m) in core_masks_packed.iter().enumerate() {
            dev_info!(self.dev.as_ref(), "Core mask reg {}: {:#010x}\n", i, m);
        }

        let num_cores = id_counts_1 & 0xff;

        // The mask packs one *slot* per bit and a cluster's slots need not all
        // be populated, so the distance between clusters in the mask is the
        // slot count, which is not always the enabled-core count. G13/G14 get
        // away with using num_cores because there the two are equal.
        //
        // On this t6030 (18-core M3 Pro) they are not: id_counts_1 = 0x110209
        // reports 9 cores per cluster, but the mask is 0x0007fdff -- bits 0-8,
        // a gap at bit 9, then bits 10-18. Stride 9 leaves a leftover bit and
        // yields 9+8=17 cores; stride 10 gives a clean 9+9=18. 10 is also what
        // G14X uses (hw::t602x max_num_cores), so treat it as the G15 slot
        // count rather than trying to read it out of a register we have not
        // identified.
        let core_stride = match gpu_gen {
            7 => 10,
            _ => num_cores,
        };

        if core_stride > 32 {
            dev_err!(
                self.dev.as_ref(),
                "Too many cores per cluster ({} > 32)\n",
                core_stride
            );
            return Err(ENODEV);
        }

        if core_stride * num_clusters > (core_mask_regs.len() * 32) as u32 {
            dev_err!(
                self.dev.as_ref(),
                "Too many total cores ({} x {} > {})\n",
                num_clusters,
                core_stride,
                core_mask_regs.len() * 32
            );
            return Err(ENODEV);
        }

        let mut core_masks = KVec::new();
        let mut total_active_cores: u32 = 0;

        let max_core_mask = ((1u64 << core_stride) - 1) as u32;
        for _ in 0..num_clusters {
            let mask = core_mask_regs[0] & max_core_mask;
            core_masks.push(mask, GFP_KERNEL)?;
            for i in 0..core_mask_regs.len() {
                core_mask_regs[i] >>= core_stride;
                if i < (core_mask_regs.len() - 1) {
                    core_mask_regs[i] |= core_mask_regs[i + 1] << (32 - core_stride);
                }
            }
            total_active_cores += mask.count_ones();
        }

        if core_mask_regs.iter().any(|a| *a != 0) {
            for (i, m) in core_mask_regs.iter().enumerate() {
                dev_err!(
                    self.dev.as_ref(),
                    "Leftover core mask {}: {:#010x} (stride {})\n",
                    i,
                    m,
                    core_stride
                );
            }
            return Err(EIO);
        }

        let (gpu_rev, gpu_rev_id) = match (id_version >> 8) & 0xff {
            0x00 => (hw::GpuRevision::A0, hw::GpuRevisionID::A0),
            0x01 => (hw::GpuRevision::A1, hw::GpuRevisionID::A1),
            0x10 => (hw::GpuRevision::B0, hw::GpuRevisionID::B0),
            0x11 => (hw::GpuRevision::B1, hw::GpuRevisionID::B1),
            0x20 => (hw::GpuRevision::C0, hw::GpuRevisionID::C0),
            0x21 => (hw::GpuRevision::C1, hw::GpuRevisionID::C1),
            a => {
                dev_err!(self.dev.as_ref(), "Unknown GPU revision {}\n", a);
                return Err(ENODEV);
            }
        };

        Ok(hw::GpuIdConfig {
            gpu_gen: match (id_version >> 24) & 0xff {
                4 => hw::GpuGen::G13,
                5 => hw::GpuGen::G14,
                6 => hw::GpuGen::G14, // G14X has a separate ID
                7 => hw::GpuGen::G15, // G15X likewise; t6030 reports 7
                a => {
                    dev_err!(self.dev.as_ref(), "Unknown GPU generation {}\n", a);
                    return Err(ENODEV);
                }
            },
            gpu_variant: match (id_version >> 16) & 0xff {
                1 => hw::GpuVariant::P, // Guess
                2 => hw::GpuVariant::G,
                3 => hw::GpuVariant::S,
                4 => {
                    if num_clusters > 4 {
                        hw::GpuVariant::D
                    } else {
                        hw::GpuVariant::C
                    }
                }
                a => {
                    dev_err!(self.dev.as_ref(), "Unknown GPU variant {}\n", a);
                    return Err(ENODEV);
                }
            },
            gpu_rev,
            gpu_rev_id,
            num_clusters,
            // Everywhere else this is the per-cluster *slot* count, not the
            // enabled-core count -- on an M2 Max it reads 10 while only 38 of
            // the 40 slots are live -- and the firmware sizes its per-core
            // arrays by it. So report the stride, not the register's 9.
            num_cores: core_stride,
            num_frags: core_stride, // Used to be id_counts_1[15:8] but does not work for G14X
            num_gps: (id_counts_2 >> 16) & 0xff,
            total_active_cores,
            core_masks,
            core_masks_packed,
        })
    }

    /// Get the fault information from the MMU status register, if one occurred.
    pub(crate) fn get_fault_info(&self, cfg: &'static hw::HwConfig) -> Option<FaultInfo> {
        let g14x = cfg.gpu_core as u32 >= hw::GpuCore::G14S as u32;
        let g15 = cfg.gpu_core as u32 >= hw::GpuCore::G15S as u32;

        let fault_info = if g14x {
            self.sgx_read64::<FAULT_INFO_G14X>()
        } else {
            self.sgx_read64::<FAULT_INFO>()
        };

        if fault_info & 1 == 0 {
            return None;
        }

        // 32.41.460: this is Apple's decode, read out of
        // `AGXAcceleratorG15::getMMUFaultInfo` (AGXG15S @ fffffe00091155a8),
        // not inferred. The one instruction that matters is
        //
        //     ldr   x8, [sp, #8]          // the raw FAULT_ADDR register
        //     ubfiz x8, x8, #6, #42       // addr = (raw & (2^42-1)) << 6
        //
        // so the field is 42 bits wide and the shift is 6 -- which is exactly
        // what this driver already did, because our raw values are only 37
        // bits and the mask is a no-op on them.
        //
        // ⛔ That REFUTES 32.41.458, which claimed the decode was wrong and the
        // faults were near-null. They are not. `0x1f0000050e` really does mean
        // `0x7c000014380`, and 32.41.457's reading -- a constant high part with
        // a varying low index -- stands. The "outside the IAS" argument was the
        // error: `ubfiz #6, #42` yields up to 48 bits by construction, so Apple
        // deliberately allows a fault address above the 42-bit input range.
        //
        // `fault_addr_bits` stays as an override, but its default is now 42,
        // Apple's width, rather than the 32 that .458 guessed.
        let fault_addr = if g14x {
            let raw = self.sgx_read64::<FAULT_ADDR_G14X>();
            let bits = *crate::module_parameters::fault_addr_bits.value();
            let bits = if bits == 0 || bits > 64 { 42 } else { bits };
            if bits >= 64 {
                raw
            } else {
                raw & ((1u64 << bits) - 1)
            }
        } else {
            fault_info >> 30
        };

        pr_info!(
            "asahi/fault raw: info {:#x} addr {:#x} (addr<<6 = {:#x})\n",
            fault_info,
            fault_addr,
            fault_addr << 6
        );

        // 32.41.460/.461: the rest of `fault_info`, straight out of Apple's
        // decoder. ⚠️ Corrected from .460, which claimed we decode none of
        // these -- in fact `reason`, `vm_slot`, `sideband` and `read` were
        // already right, name for name. Apple's layout is:
        //
        //   bit  0      fault valid                  (tbz w8, #0 -> bail)
        //   [3:1]       REASON, clamped to < 5       (ubfx x8, x8, #1, #3)
        //               Apple's names, resolved out of the kext's string
        //               table: 0 "Page Invalid", 1 "AF fault", 2 "Write Only",
        //               3 "Read Only", 4 "No Access", 5 "<invalid>" -- which is
        //               exactly this driver's `FaultReason`, so that mapping is
        //               now confirmed rather than inherited.
        //   bit  4      read/write -> MMUFaultInfo +37 (ubfx w10, w8, #4, #1)
        //   [8:7]       -> MMUFaultInfo +36          (ubfx w10, w8, #7, #2)
        //               ⚠️ This driver calls this field `level`. Apple gives it
        //               no name and no table; the thing that behaves like a
        //               level is [3:1]. The value is not wrong, the label is.
        //   [16:9]      unit code                    (ushl/and lane 0: >>9 & 0xff)
        //   [22:17]     context index                (ubfx w9, w8, #17, #6)
        //   [29:23]     -> MMUFaultInfo +24          (ushl/and lane 1: >>23 & 0x7f)
        //
        // 🔑 [22:17] is the interesting one. Apple uses it as an index (stride
        // 16) into a table fetched by a virtual call, masks the result with
        // 0x3ffffffc000 -- bits [41:14], a 16K-page-aligned 42-bit address --
        // and stores it as part of the fault report. That is a page-table base
        // per context, so the fault tells you WHICH ADDRESS SPACE the access
        // was made in. If this does not name our VM, the access never had a
        // chance of translating and no register value would have fixed it.
        if g15 {
            pr_info!(
                "asahi/fault fields: reason {} ({}) unk87 {} read {} vm {} sideband {:#x}\n",
                (fault_info >> 1) & 0x7,
                match (fault_info >> 1) & 0x7 {
                    0 => "Page Invalid",
                    1 => "AF fault",
                    2 => "Write Only",
                    3 => "Read Only",
                    4 => "No Access",
                    _ => "<invalid>",
                },
                (fault_info >> 7) & 0x3,
                (fault_info >> 4) & 0x1,
                (fault_info >> 17) & 0x3f,
                (fault_info >> 23) & 0x7f
            );
            // 32.41.460: `getFirstMMUFaultStatus` writes a selector to 0xd800
            // before reading 0xd8c0/0xd8c8, so those two are a WINDOW, exactly
            // like 0xc020 is windowed by 0x10398. This driver has never written
            // 0xd800 -- it reads whichever slot was last selected. Report what
            // is latched before drawing conclusions from the window's contents.
            pr_info!(
                "asahi/fault window: latched 0xd800 = {:#x}\n",
                self.sgx_read64_at(0xd800)
            );

            // 32.41.460: sweep the window.
            //
            // Apple's selector is built in `getFirstMMUFaultStatus` from a
            // single 64-bit field, `accel->0x17bd8`:
            //
            //     ubfx x9, x8, #2, #4      // sel[3:0]   = v[5:2]
            //     bfi  x9, x8, #18, #2     // sel[19:18] = v[1:0]
            //     str  x9, [mmio + 0xd800]
            //
            // and that field is not a constant -- `fffffe00090d6e7c` copies it
            // out of a firmware-published structure:
            //
            //     ldr x12, [x8, #20160]            // fwstruct + 0x4ec0
            //     add x13, x13, #0x17, lsl #12
            //     str x12, [x13, #3032]            // accel->0x17bd8
            //
            // So on G15S the FIRMWARE names which unit's fault registers to
            // read, and the kext selects that window before reading. This
            // driver has never written 0xd800 at all, so every fault address
            // this phase has recorded came from whichever slot was latched --
            // which is a complete explanation for values that match no
            // structure, no register and no VA range we own.
            //
            // Sweeping is the cheap way to find out whether a coherent fault
            // record exists in some other slot. Save/restore around it, the
            // same discipline `sgx_state_sweep` uses for 0x10398, because the
            // firmware drives this selector too.
            let sweep = *crate::module_parameters::fault_win_sweep.value();
            if sweep != 0 {
                let saved = self.sgx_read64_at(0xd800);
                for hi in 0..4u64 {
                    for lo in 0..16u64 {
                        let sel = lo | (hi << 18);
                        self.sgx_write32_at(0xd800, sel as u32);
                        let info = self.sgx_read64_at(0xd8c0);
                        let addr = self.sgx_read64_at(0xd8c8);
                        // Only slots reporting a valid fault are interesting,
                        // but always show slot 0 so the sweep is anchored
                        // against the value the driver has been reading.
                        if (info & 1) != 0 || sel == 0 {
                            pr_info!(
                                "asahi/fault win sel {:#07x} (cluster {} lo2 {}): info {:#018x} addr {:#018x} -> {:#x}\n",
                                sel,
                                lo,
                                hi,
                                info,
                                addr,
                                (addr & ((1u64 << 42) - 1)) << 6
                            );
                        }
                    }
                }
                self.sgx_write32_at(0xd800, saved as u32);
            }
        }

        let unit_code = ((fault_info >> 9) & 0xff) as u8;
        let unit = match unit_code {
            0x00..=0x9f => match unit_code & 0xf {
                0x0 => FaultUnit::DCMP(unit_code >> 4),
                0x1 => FaultUnit::UL1C(unit_code >> 4),
                0x2 => FaultUnit::CMP(unit_code >> 4),
                0x3 => FaultUnit::GSL1(unit_code >> 4),
                0x4 => FaultUnit::IAP(unit_code >> 4),
                0x5 => FaultUnit::VCE(unit_code >> 4),
                0x6 => FaultUnit::TE(unit_code >> 4),
                0x7 => FaultUnit::RAS(unit_code >> 4),
                0x8 => FaultUnit::VDM(unit_code >> 4),
                0x9 => FaultUnit::PPP(unit_code >> 4),
                0xa => FaultUnit::IPF(unit_code >> 4),
                0xb => FaultUnit::IPF_CPF(unit_code >> 4),
                0xc => FaultUnit::VF(unit_code >> 4),
                0xd => FaultUnit::VF_CPF(unit_code >> 4),
                0xe => FaultUnit::ZLS(unit_code >> 4),
                _ => FaultUnit::Unknown(unit_code),
            },
            0xa1 => FaultUnit::dPM,
            0xa2 => FaultUnit::dCDM_KS(0),
            0xa3 => FaultUnit::dCDM_KS(1),
            0xa4 => FaultUnit::dCDM_KS(2),
            0xa5 => FaultUnit::dIPP,
            0xa6 => FaultUnit::dIPP_CS,
            0xa7 => FaultUnit::dVDM_CSD,
            0xa8 => FaultUnit::dVDM_SSD,
            0xa9 => FaultUnit::dVDM_ILF,
            0xaa => FaultUnit::dVDM_ILD,
            0xab => FaultUnit::dRDE(0),
            0xac => FaultUnit::dRDE(1),
            0xad => FaultUnit::FC,
            0xae => FaultUnit::GSL2,
            0xb0..=0xb7 => FaultUnit::GL2CC_META(unit_code & 0xf),
            0xb8 => FaultUnit::GL2CC_MB,
            0xd0..=0xdf if g14x => match unit_code & 0xf {
                0x0 => FaultUnit::gCDM_CS,
                0x1 => FaultUnit::gCDM_ID,
                0x2 => FaultUnit::gCDM_CSR,
                0x3 => FaultUnit::gCDM_CSW,
                0x4 => FaultUnit::gCDM_CTXR,
                0x5 => FaultUnit::gCDM_CTXW,
                0x6 => FaultUnit::gIPP,
                0x7 => FaultUnit::gIPP_CS,
                0x8 => FaultUnit::gKSM_RCE,
                // The UMA page manager: the firmware names a register
                // AGX_CR_UMA_DUPM_SELF_RESERVE_COUNT and the kernelcache's
                // perf-counter groups list USC_GUPM next to USC_UMA_CORE.
                // Present in AGXG15X's name table at 0xd9 and missing here, so
                // it used to print as Unknown(0xd9).  LOG.md 32.41.40, .53.
                0x9 => FaultUnit::gUPM,
                _ => FaultUnit::Unknown(unit_code),
            },
            // G15 has four shader-pipe instances, not two, and the index is
            // split: bit 4 is its low bit and bit 3 its high bit, so 0xe0/0xf0/
            // 0xe8/0xf8 are instances 0/1/2/3 of the same unit.  Read out of
            // AGXAcceleratorG15X's own 256-entry name table in the macOS
            // kernelcache; the `& 0xf` reading below decodes 0xe8..0xef and
            // 0xf8..0xff as Unknown on this part.  LOG.md 32.41.30.
            0xe0..=0xff if g15 => {
                let sp = (((unit_code >> 3) & 1) << 1) | ((unit_code >> 4) & 1);
                match unit_code & 0x7 {
                    0x0 => FaultUnit::gPM_SP(sp),
                    0x1 => FaultUnit::gVDM_CSD_SP(sp),
                    0x2 => FaultUnit::gVDM_SSD_SP(sp),
                    0x3 => FaultUnit::gVDM_ILF_SP(sp),
                    0x4 => FaultUnit::gVDM_TFP_SP(sp),
                    0x5 => FaultUnit::gVDM_MMB_SP(sp),
                    0x6 => FaultUnit::gRDE0_SP(sp),
                    _ => FaultUnit::Unknown(unit_code),
                }
            }
            0xe0..=0xff if g14x => match unit_code & 0xf {
                0x0 => FaultUnit::gPM_SP((unit_code >> 4) & 1),
                0x1 => FaultUnit::gVDM_CSD_SP((unit_code >> 4) & 1),
                0x2 => FaultUnit::gVDM_SSD_SP((unit_code >> 4) & 1),
                0x3 => FaultUnit::gVDM_ILF_SP((unit_code >> 4) & 1),
                0x4 => FaultUnit::gVDM_TFP_SP((unit_code >> 4) & 1),
                0x5 => FaultUnit::gVDM_MMB_SP((unit_code >> 4) & 1),
                0x6 => FaultUnit::gRDE0_SP((unit_code >> 4) & 1),
                _ => FaultUnit::Unknown(unit_code),
            },
            0xe0..=0xff if !g14x => match unit_code & 0xf {
                0x0 => FaultUnit::gPM_SP((unit_code >> 4) & 1),
                0x1 => FaultUnit::gVDM_CSD_SP((unit_code >> 4) & 1),
                0x2 => FaultUnit::gVDM_SSD_SP((unit_code >> 4) & 1),
                0x3 => FaultUnit::gVDM_ILF_SP((unit_code >> 4) & 1),
                0x4 => FaultUnit::gVDM_TFP_SP((unit_code >> 4) & 1),
                0x5 => FaultUnit::gVDM_MMB_SP((unit_code >> 4) & 1),
                0x6 => FaultUnit::gCDM_CS_KS0_SP((unit_code >> 4) & 1),
                0x7 => FaultUnit::gCDM_CS_KS1_SP((unit_code >> 4) & 1),
                0x8 => FaultUnit::gCDM_CS_KS2_SP((unit_code >> 4) & 1),
                0x9 => FaultUnit::gCDM_KS0_SP((unit_code >> 4) & 1),
                0xa => FaultUnit::gCDM_KS1_SP((unit_code >> 4) & 1),
                0xb => FaultUnit::gCDM_KS2_SP((unit_code >> 4) & 1),
                0xc => FaultUnit::gIPP_SP((unit_code >> 4) & 1),
                0xd => FaultUnit::gIPP_CS_SP((unit_code >> 4) & 1),
                0xe => FaultUnit::gRDE0_SP((unit_code >> 4) & 1),
                0xf => FaultUnit::gRDE1_SP((unit_code >> 4) & 1),
                _ => FaultUnit::Unknown(unit_code),
            },
            _ => FaultUnit::Unknown(unit_code),
        };

        let reason = match (fault_info >> 1) & 0x7 {
            0 => FaultReason::Unmapped,
            1 => FaultReason::AfFault,
            2 => FaultReason::WriteOnly,
            3 => FaultReason::ReadOnly,
            4 => FaultReason::NoAccess,
            a => FaultReason::Unknown(a as u8),
        };

        Some(FaultInfo {
            address: fault_addr << 6,
            sideband: ((fault_info >> 23) & 0x7f) as u8,
            vm_slot: ((fault_info >> 17) & 0x3f) as u32,
            unit_code,
            unit,
            level: ((fault_info >> 7) & 3) as u8,
            unk_5: ((fault_info >> 5) & 3) as u8,
            read: (fault_info & (1 << 4)) != 0,
            reason,
        })
    }
}
