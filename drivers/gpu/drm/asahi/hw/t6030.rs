// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Hardware configuration for t6030 (M3 Pro) platforms.
//!
//! ⚠️ Status: first cut. This profile is enough to describe the SoC, but the
//! driver cannot bind on t6030 until the macOS 15.5 firmware ABI is ported into
//! `fw/*.rs` under `#[ver(V >= V15_5)]`. See `PORTING-15.5.md` in
//! `asahi/m3-gpu-display/` for the field-by-field map; everything there was
//! reverse engineered against this machine and validated on hardware (the
//! firmware now accepts InitData and completes DC_Init).
//!
//! Provenance of the numbers below:
//!   * io_mappings  -- read out of the running firmware's own index table at
//!     0xfffffc000005cea0, which maps io_mapping index -> register-block name
//!     for all 48 blocks it uses. 15.5 renumbered this array completely and
//!     grew it to 55 entries of 40 bytes. Not guesswork.
//!   * the non-Fender bases (1, 5, 7, 8, 13, 14, 15, 16, 18, 23, 43) are read
//!     out of `AGXAcceleratorG15X::configureDevice` in the macOS kernelcache
//!     (0xfffffe000911ff48). That function fills a 55-entry array of 1080-byte
//!     records -- `{flags, ?, phys, ..., size, ..., name}`, entry n at
//!     `x19 + 2896 + (n-1)*1080` -- and every record it writes is anchored by a
//!     name string, so the index is read rather than inferred. It branches on
//!     the G15X core ID; this profile follows the 0x17 (M3 Pro) path, which is
//!     also why 17, 19, 21 and 25 stay empty: that path never writes them.
//!     LOG.md 32.38.
//!   * the Fender bases (20, 31, 33-39, 41, 44, 47) come from the firmware's
//!     own address-to-region table at 0x50d80, twelve 32-byte records of
//!     `{io_mapping index, SGX-relative base, size}`. LOG.md 32.38.
//!   * clocks/power -- carried over from m1n1's t6030 work, where they were
//!     accepted by the firmware. Several are still t602x values; each is marked.
//!   * gpu_core     -- unverified, see hw::GpuCore::G15S.

use crate::f32;

use super::*;

/// 15.5 io_mappings for t6030.
///
/// The index order is NOT the 13.5 order -- 15.5 renumbered it (AICSWInt moved
/// 2 -> 1, PMPDoorbell 8 -> 6, MCache 11 -> 7, AICBanked 12 -> 8, PMGRScratch
/// 13 -> 9, NIA die0/1 14/15 -> 10/11). Names are the firmware's own.
///
/// The firmware skips any entry whose phys_addr is 0 (`cbz`), so a `None` is
/// safe until something needs that block -- at which point the cached pointer
/// is NULL and it faults. Decode such a fault with
/// `idx = (global - 0x1a3d50) / 40`.
///
/// The Fender family (28-47) are sub-blocks inside the 0x16c000 Fender region
/// at sgx reg[1], and their offsets are not guesswork: the firmware carries its
/// own address-to-region table for register tracing at 0x50d80, twelve 32-byte
/// records of `{io_mapping index, SGX-relative base, size}` terminated by
/// `0xdadadada`, and the classifier at 0x7908 walks the same boundaries. It
/// pins 20, 31, 33, 34, 35, 36, 37, 38, 39, 41, 44 and 47 exactly. LOG.md 32.38.
const fn iomaps() -> [Option<IOMapping>; 55] {
    const SGX: usize = 0x290000000; // /arm-io/sgx reg[0], "UTRegisters" (RGX)
    const FENDER: usize = SGX + 0xd00000; // /arm-io/sgx reg[1]
    const AIC: usize = 0x351000000; // AIC v3

    [
        None,                                                         // 0
        Some(IOMapping::new(AIC + 0x14000, false, 1, 0x4000, 0, true)), // 1 GPCAICSWInt
        None,                                                         // 2
        Some(IOMapping::new(SGX, false, 1, 0x20000, 0, true)),        // 3 UTRegisters
        None,                                                         // 4 DisplayURMagic
        Some(IOMapping::new(0x3502bc000, false, 1, 0x4000, 0, true)), // 5 TempSensor
        None,                                                         // 6 PMPDoorbell
        Some(IOMapping::new(0x2201c4000, false, 2, 0x18000, 0x2000000, true)), // 7 MCache
        Some(IOMapping::new(AIC + 0x4c000, false, 1, 0x4000, 0, false)), // 8 AICBankedRegisters
        None,                                                         // 9 PMGRScratch
        None,                                                         // 10 SpecialAgentIdleDie0
        None,                                                         // 11 SpecialAgentIdleDie1
        None,                                                         // 12
        Some(IOMapping::new(0x3503d0000, false, 1, 0x4000, 0, true)), // 13 PushTelemetryDashboard
        Some(IOMapping::new(0x3503c0000, false, 1, 0x4000, 0, true)), // 14 PushTelemetryDashboardReadSpace
        Some(IOMapping::new(0x3503d8000, false, 1, 0x4000, 0, true)), // 15 PushTelemetryDashboardConfigSpace
        Some(IOMapping::new(0x30945c000, false, 1, 0x4000, 0, true)), // 16 ANE0Doorbell
        None, // 17 ANE1Registers
        Some(IOMapping::new(0x350280000, false, 1, 0x8000, 0, true)), // 18 MTRPMSRegisters
        None, // 19 AONPTDSpace
        // Unnamed in the firmware's index table, so the io_mappings walker
        // skips it and the firmware instead copies all five fields -- including
        // the *virtual* address -- straight out of AGXHWDataB+0x960 at 0x2f928.
        // Left NULL it takes a data abort with FAR = the register offset alone;
        // the one that found this was `ldr x8, [x8, #22616]` at 0x22748 with
        // FAR = 0x5858, on the perf-state path. Decode such a fault with
        // `idx = (global - 0x1a3d50) / 40`.
        //
        // It is the UTRegisters/RGX space, not Fender: its offsets run
        // 0..0x1ec08, and the PIO-poll errors that name this block print
        // *absolute* SGX-relative addresses -- 0x14008, 0x1c800, 0x1c860,
        // 0x1c870, 0x1c878 -- which are exactly those offsets, so its base
        // offset is zero. LOG.md 32.36.
        Some(IOMapping::new(SGX, false, 1, 0x20000, 0, true)), // 20 (unnamed)
        None, // 21 AFRClkGenRegisters
        None, // 22 AFRClkGenRegisters1
        // AFRRegisters, the largest block the firmware touches. A dataflow
        // pass over the live firmware's whole __text (every load of this
        // entry's cached VA at 0x1a40f0, then every offset used through it)
        // gives the complete access set:
        //
        //   0x20190/94  0x30018/1c/20  0x30060  0x300a8  0x300f0
        //   0x60190/94  0xa0190/94
        //   and the same block again at +0x200000 (0x220190 .. 0x2a0194)
        //
        // so the block is two halves 0x200000 apart -- one per cluster -- each
        // holding three AFR units 0x40000 apart plus a control page at
        // +0x30000. The top register is 0x2a0194, so the block is 0x2a0198
        // bytes at minimum.
        //
        // That is what pins the base. `/arm-io/sgx` reg[0] is 0x32a4000 long,
        // and the m1n1 register scan found exactly one run in it big enough to
        // hold 0x2a0198 bytes: SGX + 0x3000000 .. SGX + 0x32c0000. Rounding the
        // access set up to the end of reg[0] gives 0x2a4000, i.e. the block
        // ends precisely where the sgx region does -- which is why reg[0] has
        // that odd size in the first place. LOG.md 32.38.
        //
        // The earlier guess of SGX + 0xde0000 (LOG 32.37) came from reading
        // only the first three writes and is wrong: 0xe0000 cannot hold a
        // 0x2a0198-byte block, so the second half ran off the end of the
        // mapping and took an SError.
        Some(IOMapping::new(SGX + 0x3000000, false, 1, 0x400000, 0, true)), // 23 AFRRegisters
        None, // 24
        None, // 25 GFXCAXI2AFRegisters
        None, // 26 DVFMAFRRegisters
        None, // 27 DVFMMgpuRegisters
        None, // 28 FenderRegsLegacyFenderCfg (not in the firmware's region table)
        None, // 29 FenderRegsLegacyGFXCfg (not in the firmware's region table)
        None, // 30 FenderRegsLegacyDPECfg (not in the firmware's region table)
        // GFenderCfg holds the GPU ID registers: the firmware reads +0x10 and
        // splits it as clusters = (v >> 8) & 0xff, dies = (v >> 16) & 0xf, which
        // is exactly ID_COUNTS_1 -- and regs.rs already knows that register is at
        // SGX + 0xd04010. So the block is FENDER + 0x4000.
        //
        // This one mattered: with the block at FENDER + 0 the firmware read zero
        // there, so its cluster-mask global came out 0 and
        // AGFHALPreInactivePowerStateChange waited forever for Bumper +0x70 to
        // read 0 when the hardware says 3 (two clusters). LOG.md 32.38.
        Some(IOMapping::new(FENDER + 0x4000, false, 1, 0x8000, 0, true)), // 31 FenderRegsGFenderCfg
        None, // 32 GMGIFAFRegisters
        // CRE polls +0x7c40 and the poll site's literal is 0xd57c40 (0x7e8c),
        // so the block is FENDER + 0x50000. LOG.md 32.38.
        Some(IOMapping::new(FENDER + 0x50000, false, 1, 0x8000, 0, true)), // 33 CRERegisters
        // GCPMS writes +0x80 at 0xd094 and the trace call two instructions
        // later names the same register as 0xd10010 + 0x70 = 0xd10080, so the
        // block is FENDER + 0x10000. Its offsets reach 0x305c. LOG.md 32.38.
        Some(IOMapping::new(FENDER + 0x10000, false, 1, 0x4000, 0, true)), // 34 FenderRegsGCPMS
        // MCW writes +0x800 at 0x2e574 and the very next instruction pair
        // builds the literal 0xd40800 for the trace call, so the block is
        // FENDER + 0x40000. LOG.md 32.38.
        Some(IOMapping::new(FENDER + 0x40000, false, 1, 0x4000, 0, true)), // 35 FenderRegsMCW
        Some(IOMapping::new(FENDER + 0x60000, false, 1, 0x20000, 0, true)), // 36 FenderRegsScratchRam
        // Bumper's registers are 0x70/0x88/0xa0/0xa4/0xa8 (dataflow over the
        // live firmware) and the poll error prints 0xd44088, so the block is at
        // FENDER + 0x44000 and 0x88 is its own offset. Mapped at FENDER + 0 it
        // read the wrong word and passed by accident. LOG.md 32.38.
        Some(IOMapping::new(FENDER + 0x44000, false, 1, 0x4000, 0, true)), // 37 FenderRegsBumper
        // The PRC block is NOT at the start of the Fender region. The firmware
        // polls its status register at offset 0 of this mapping and prints the
        // address it *meant* to reach when the poll times out:
        //
        //   FWLog: ERROR: PIO poll from AGFHALWaitForPerfStateCS timeout after
        //   250us [type:38 reg:0xd4c000 expected:0x1 got:0x0 max:250us]
        //
        // `type:38` is this io_mapping index and `0xd4c000` is SGX-relative, so
        // the block starts at FENDER + 0x4c000. With the whole region mapped
        // instead, every perf-state change waited forever on the wrong register
        // and no job ever ran. Mapped to the end of the Fender region because
        // the block's true length is unknown and over-mapping is harmless.
        // LOG.md 32.35.
        Some(IOMapping::new(FENDER + 0x4c000, false, 1, 0x200, 0, true)), // 38 FenderRegsPRC
        Some(IOMapping::new(FENDER + 0x100000, false, 1, 0x4000, 0, true)), // 39 FenderRegsDFenderMGPUCfg
        None, // 40 FenderRegsDFenderPowerTransG16G (G16G only)
        Some(IOMapping::new(FENDER + 0x140000, false, 1, 0x4000, 0, true)), // 41 FenderRegsDFenderCSCfg
        None, // 42 FenderRegsIVDM (not in the firmware's region table)
        Some(IOMapping::new(SGX + 0xe08000, false, 1, 0x8000, 0, true)), // 43 MTRRegisters
        Some(IOMapping::new(FENDER + 0x15c000, false, 1, 0x4000, 0, true)), // 44 GFXCLKGENMGPU
        None, // 45 ThrtlStatCntrsRegisters
        None, // 46 AVMRControllerFastRegisters
        Some(IOMapping::new(FENDER + 0x160000, false, 1, 0x50, 0, true)), // 47 FenderRegsAFRNIA
        None, // 48 MCacheG11Plane0Registers
        None, // 49 MCacheG11Plane1Registers
        None, // 50 MCacheG11Plane2Registers
        None, // 51 MCacheG11Plane3Registers
        None, // 52
        None, // 53
        None, // 54 DpwrMXURegisters
    ]
}

pub(crate) const HWCONFIG_T6030: super::HwConfig = HwConfig {
    chip_id: 0x6030,
    gpu_gen: GpuGen::G15,
    gpu_variant: GpuVariant::S,
    gpu_core: GpuCore::G15S,

    base_clock_hz: 24_000_000,
    uat_oas: 42,
    num_dies: 1,
    // Read out of the GPU's own ID registers, not guessed:
    //   id_counts_1 = 0x110209
    //     clusters = ((>>8) & 0xff) * ((>>16) & 0xf) = 2 * 1 = 2
    //     cores    = (& 0xff)                        = 9   -> 2 * 9 = 18 ✓
    //   id_counts_2 = 0x40404 -> num_gps = (>>16) & 0xff = 4
    // The core mask 0x0007fdff packs at a stride of 10, not 9: bits 0-8 are
    // cluster 0, bit 9 is the gap, bits 10-18 are cluster 1. So max_num_cores
    // is 10 with 9 populated per cluster -- same shape as t602x. A 3x6 split
    // left a bit over and the driver rejected it with "Leftover core mask".
    max_num_clusters: 2,
    max_num_cores: 10,
    max_num_frags: 10,
    max_num_gps: 4,

    preempt1_size: 0x540,
    preempt2_size: 0x280,
    preempt3_size: 0x40,
    compute_preempt1_size: 0x25980, // t602x value
    clustering: Some(HwClusteringConfig {
        meta1_blocksize: 0x44,
        meta2_size: 0xc0 * 16,
        meta3_size: 0x280 * 16,
        meta4_size: 0x10 * 128,
        max_splits: 64,
    }),

    render: HwRenderConfig {
        tiling_control: 0x180340, // t602x value
    },

    da: HwConfigA {
        unk_87c: 500,
        unk_8cc: 11000,
        unk_e24: 125,
    },
    db: HwConfigB {
        unk_454: 0,
        unk_4e0: 4,
        unk_534: 0,
        unk_ab8: 0, // Unused
        unk_abc: 0, // Unused
        unk_b30: 0,
    },
    shared1_tab: &[
        0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
        0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    ],
    shared1_a4: 0,
    shared2_tab: &[0x800, 0x1555, -1, -1, -1, -1, -1, -1, 0xaaaaa, 0],
    shared2_unk_508: 0xc00007,
    shared2_curves: Some(HwConfigShared2Curves {
        t1_coef: 11000,
        t2: &[
            0xf07, 0x4c0, 0x680, 0x8c0, 0xa80, 0xc40, 0xd80, 0xec0, 0xf40,
        ],
        t3_coefs: &[0, 20, 27, 36, 43, 50, 55, 60, 62],
        t3_scales: &[9, 3209, 10400],
    }),
    shared3_unk: 8,
    shared3_tab: &[
        125, 125, 125, 125, 125, 125, 125, 125, 7500, 125, 125, 125, 125, 125, 125, 125,
    ],
    idle_off_standby_timer_default: 700,
    unk_hws2_4: Some(f32!([1.0, 0.8, 0.2, 0.9, 0.1, 0.25, 0.7, 0.9])),
    unk_hws2_24: 6,
    global_unk_54: 4000,
    sram_k: f32!(1.02),
    unk_coef_a: &[
        &f32!([0.0, 8.2, 0.0, 6.9, 6.9]),
        &f32!([0.0, 0.0, 0.0, 6.9, 6.9]),
        &f32!([0.0, 8.2, 0.0, 6.9, 0.0]),
        &f32!([0.0, 0.0, 0.0, 6.9, 0.0]),
    ],
    unk_coef_b: &[
        &f32!([0.0, 9.0, 0.0, 8.0, 8.0]),
        &f32!([0.0, 0.0, 0.0, 8.0, 8.0]),
        &f32!([0.0, 9.0, 0.0, 8.0, 0.0]),
        &f32!([0.0, 0.0, 0.0, 8.0, 0.0]),
    ],
    global_tab: Some(&[
        0, 2, 2, 1, 1, 90, 75, 1, 1, 1, 2, 90, 75, 1, 1, 1, 2, 90, 75, 1, 1, 1, 1, 90, 75, 1, 1,
    ]),
    // false: this machine's ADT has no cs-perf-states / afr-perf-states at all
    // (checked directly -- both properties are absent), unlike t602x.
    has_csafr: false,
    // GUESS: t6020 values. The M3's sgx ADT node no longer carries the power
    // model, so these could not be read from the machine.
    fast_sensor_mask: [0xc000d00, 0],
    fast_sensor_mask_alt: [0x1d001d00, 0],
    fast_die0_sensor_present: 0, // Unused
    io_mappings: &iomaps(),
    // sgx reg[1] + 0x60000, by analogy with t602x (0x404d60000 = Fender +
    // 0x60000). Unverified on t6030.
    sram_base: Some(0x290d60000),
    sram_size: Some(0x20000),
};
