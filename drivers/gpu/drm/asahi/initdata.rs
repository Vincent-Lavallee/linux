// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(clippy::unusual_byte_groupings)]

//! GPU initialization data builder.
//!
//! The root of all interaction between the GPU firmware and the host driver is a complex set of
//! nested structures that we call InitData. This includes both GPU hardware/firmware configuration
//! and the pointers to the ring buffers and global data fields that are used for communication at
//! runtime.
//!
//! Many of these structures are poorly understood, so there are lots of hardcoded unknown values
//! derived from observing the InitData structures that macOS generates.

use crate::f32;
use crate::fw::initdata::*;
use crate::fw::types::*;
use crate::module_parameters;
use crate::{
    driver::AsahiDevice,
    gem,
    gpu,
    hw,
    mmu, //
};
use kernel::error::{
    Error,
    Result, //
};
use kernel::macros::versions;
use kernel::prelude::*;
use kernel::try_init;

use ::pin_init;

/// LOG.md 32.41.539: print the non-zero 16-byte rows of a HwData struct, for a
/// field-by-field diff against the stores macOS's kext makes in
/// `AGX{,Arm}Firmware::initFirmwareData`.
fn dump_nonzero(tag: &str, b: &[u8], limit: usize) {
    let n = core::cmp::min(b.len(), limit);
    let mut o = 0;
    while o < n {
        let e = core::cmp::min(o + 16, n);
        let row = &b[o..e];
        if row.iter().any(|&x| x != 0) {
            let w = |i: usize| -> u32 {
                let mut v = [0u8; 4];
                for k in 0..4 {
                    if i + k < row.len() {
                        v[k] = row[i + k];
                    }
                }
                u32::from_le_bytes(v)
            };
            pr_info!(
                "asahi/hwdata {} {:#06x}: {:08x} {:08x} {:08x} {:08x}\n",
                tag, o, w(0), w(4), w(8), w(12)
            );
        }
        o += 16;
    }
}
use ::pin_init::Init;

/// Builder helper for the global GPU InitData.
#[versions(AGX)]
pub(crate) struct InitDataBuilder<'a> {
    dev: &'a AsahiDevice,
    alloc: &'a mut gpu::KernelAllocators,
    cfg: &'static hw::HwConfig,
    dyncfg: &'a hw::DynConfig,
}

#[versions(AGX)]
impl<'a> InitDataBuilder::ver<'a> {
    /// Create a new InitData builder
    pub(crate) fn new(
        dev: &'a AsahiDevice,
        alloc: &'a mut gpu::KernelAllocators,
        cfg: &'static hw::HwConfig,
        dyncfg: &'a hw::DynConfig,
    ) -> InitDataBuilder::ver<'a> {
        InitDataBuilder::ver {
            dev,
            alloc,
            cfg,
            dyncfg,
        }
    }

    /// Create the HwDataShared1 structure, which is used in two places in InitData.
    fn hw_shared1(cfg: &'static hw::HwConfig) -> impl Init<raw::HwDataShared1> {
        init!(raw::HwDataShared1 {
            unk_a4: cfg.shared1_a4,
            ..Zeroable::init_zeroed()
        })
        .chain(|ret| {
            for (i, val) in cfg.shared1_tab.iter().enumerate() {
                ret.table[i] = *val;
            }
            Ok(())
        })
    }

    fn init_curve(
        curve: &mut raw::HwDataShared2Curve,
        unk_0: u32,
        unk_4: u32,
        t1: &[u16],
        t2: &[i16],
        t3: &[KVec<i32>],
    ) {
        curve.unk_0 = unk_0;
        curve.unk_4 = unk_4;
        (*curve.t1)[..t1.len()].copy_from_slice(t1);
        (*curve.t1)[t1.len()..].fill(t1[0]);
        (*curve.t2)[..t2.len()].copy_from_slice(t2);
        (*curve.t2)[t2.len()..].fill(t2[0]);
        for (i, a) in curve.t3.iter_mut().enumerate() {
            a.fill(0x3ffffff);
            if i < t3.len() {
                let b = &t3[i];
                (**a)[..b.len()].copy_from_slice(b);
            }
        }
    }

    /// Create the HwDataShared2 structure, which is used in two places in InitData.
    fn hw_shared2(
        cfg: &'static hw::HwConfig,
        dyncfg: &'a hw::DynConfig,
    ) -> impl Init<raw::HwDataShared2, Error> + 'a {
        try_init!(raw::HwDataShared2 {
            unk_28: Array::new([0xff; 16]),
            g14: Default::default(),
            unk_508: cfg.shared2_unk_508,
            ..Zeroable::init_zeroed()
        })
        .chain(|ret| {
            for (i, val) in cfg.shared2_tab.iter().enumerate() {
                ret.table[i] = *val;
            }

            let curve_cfg = match cfg.shared2_curves.as_ref() {
                None => return Ok(()),
                Some(a) => a,
            };

            let mut t1 = KVec::new();
            let mut t3 = KVec::new();

            for _ in 0..curve_cfg.t3_scales.len() {
                t3.push(KVec::new(), GFP_KERNEL)?;
            }

            // The shared2 curves are per-perf-state tables sized for the SoC
            // they were captured on: t602x has 9 entries. t6030 reports 14 perf
            // states, so indexing t3_coefs by the perf-state number panics with
            // "index out of bounds: the len is 9 but the index is 9".
            //
            // Resample rather than clamp, so the curve keeps its shape across
            // the longer pstate list. Same fix as the m1n1 side (see its
            // gpu/initdata.rs); like that one it is plausible rather than
            // measured, and is a suspect for any later DVFS misbehaviour.
            let n_ps = dyncfg.pwr.perf_states.len();
            let n_coef = curve_cfg.t3_coefs.len();
            for (i, ps) in dyncfg.pwr.perf_states.iter().enumerate() {
                let ci = if n_ps > 1 && n_coef > 1 {
                    (i * (n_coef - 1) + (n_ps - 1) / 2) / (n_ps - 1)
                } else {
                    0
                };
                let t3_coef = curve_cfg.t3_coefs[ci.min(n_coef - 1)];
                if t3_coef == 0 {
                    t1.push(0xffff, GFP_KERNEL)?;
                    for j in t3.iter_mut() {
                        j.push(0x3ffffff, GFP_KERNEL)?;
                    }
                    continue;
                }

                let f_khz = (ps.freq_hz / 1000) as u64;
                let v_max = ps.max_volt_mv() as u64;

                t1.push(
                    (1000000000 * (curve_cfg.t1_coef as u64) / (f_khz * v_max))
                        .try_into()
                        .unwrap(),
                    GFP_KERNEL,
                )?;

                for (j, scale) in curve_cfg.t3_scales.iter().enumerate() {
                    t3[j].push(
                        (t3_coef as u64 * 1000000100 * *scale as u64 / (f_khz * v_max * 6))
                            .try_into()
                            .unwrap(),
                        GFP_KERNEL,
                    )?;
                }
            }

            ret.g14.unk_14 = 0x6000000;
            Self::init_curve(
                &mut ret.g14.curve1,
                0,
                0x20000000,
                &[0xffff],
                &[0x0f07],
                &[],
            );
            Self::init_curve(&mut ret.g14.curve2, 7, 0x80000000, &t1, curve_cfg.t2, &t3);

            Ok(())
        })
    }

    /// Create the HwDataShared3 structure, which is used in two places in InitData.
    fn hw_shared3(cfg: &'static hw::HwConfig) -> impl Init<raw::HwDataShared3> {
        pin_init::init_zeroed::<raw::HwDataShared3>().chain(|ret| {
            if !cfg.shared3_tab.is_empty() {
                ret.unk_0 = 1;
                ret.unk_4 = 500;
                ret.unk_8 = cfg.shared3_unk;
                ret.table.copy_from_slice(cfg.shared3_tab);
                ret.unk_4c = 1;
            }
            Ok(())
        })
    }

    /// Create an unknown T81xx-specific data structure.
    fn t81xx_data(
        cfg: &'static hw::HwConfig,
        dyncfg: &'a hw::DynConfig,
    ) -> impl Init<raw::T81xxData> {
        let _perf_max_pstate = dyncfg.pwr.perf_max_pstate;

        pin_init::init_zeroed::<raw::T81xxData>().chain(move |_ret| {
            match cfg.chip_id {
                0x8103 | 0x8112 => {
                    #[ver(V < V13_3)]
                    {
                        _ret.unk_d8c = 0x80000000;
                        _ret.unk_d90 = 4;
                        _ret.unk_d9c = f32!(0.6);
                        _ret.unk_da4 = f32!(0.4);
                        _ret.unk_dac = f32!(0.38552);
                        _ret.unk_db8 = f32!(65536.0);
                        _ret.unk_dbc = f32!(13.56);
                        _ret.max_pstate_scaled = 100 * _perf_max_pstate;
                    }
                }
                _ => (),
            }
            Ok(())
        })
    }

    /// Create the HwDataA structure. This mostly contains power-related configuration.
    fn hwdata_a(&mut self) -> Result<GpuObject<HwDataA::ver>> {
        let pwr = &self.dyncfg.pwr;
        let period_ms = pwr.power_sample_period;
        let period_s = F32::from(period_ms) / f32!(1000.0);
        let ppm_filter_tc_periods = pwr.ppm_filter_time_constant_ms / period_ms;
        #[ver(V >= V13_0B4)]
        let ppm_filter_tc_ms_rounded = ppm_filter_tc_periods * period_ms;
        let ppm_filter_a = f32!(1.0) / ppm_filter_tc_periods.into();
        let perf_filter_a = f32!(1.0) / pwr.perf_filter_time_constant.into();
        let perf_filter_a2 = f32!(1.0) / pwr.perf_filter_time_constant2.into();
        let avg_power_target_filter_a = f32!(1.0) / pwr.avg_power_target_filter_tc.into();
        let avg_power_filter_tc_periods = pwr.avg_power_filter_tc_ms / period_ms;
        #[ver(V >= V13_0B4)]
        let avg_power_filter_tc_ms_rounded = avg_power_filter_tc_periods * period_ms;
        let avg_power_filter_a = f32!(1.0) / avg_power_filter_tc_periods.into();
        let pwr_filter_a = f32!(1.0) / pwr.pwr_filter_time_constant.into();

        let base_ps = pwr.perf_base_pstate;
        let base_ps_scaled = 100 * base_ps;
        let max_ps = pwr.perf_max_pstate;
        let max_ps_scaled = 100 * max_ps;
        let boost_ps_count = max_ps - base_ps;

        #[allow(unused_variables)]
        let base_clock_khz = self.cfg.base_clock_hz / 1000;
        let v_clocks_per_period = pwr.pwr_sample_period_aic_clks;

        #[allow(unused_variables)]
        let clocks_per_period_coarse = self.cfg.base_clock_hz / 1000 * pwr.power_sample_period;

        self.alloc
            .private
            .new_init(pin_init::init_zeroed(), |_inner, _ptr| {
                let cfg = &self.cfg;
                let dyncfg = &self.dyncfg;
                try_init!(raw::HwDataA::ver {
                    clocks_per_period: v_clocks_per_period,
                    #[ver(V >= V13_0B4)]
                    clocks_per_period_2: v_clocks_per_period,
                    pwr_status: AtomicU32::new(4),
                    unk_10: f32!(1.0),
                    actual_pstate: 1,
                    tgt_pstate: 1,
                    base_pstate_scaled: base_ps_scaled,
                    unk_40: 1,
                    max_pstate_scaled: max_ps_scaled,
                    min_pstate_scaled: 100,
                    unk_64c: 625,
                    pwr_filter_a_neg: f32!(1.0) - pwr_filter_a,
                    pwr_filter_a: pwr_filter_a,
                    pwr_integral_gain: pwr.pwr_integral_gain,
                    pwr_integral_min_clamp: pwr.pwr_integral_min_clamp.into(),
                    max_power_1: pwr.max_power_mw.into(),
                    pwr_proportional_gain: pwr.pwr_proportional_gain,
                    pwr_pstate_related_k: -F32::from(max_ps_scaled) / pwr.max_power_mw.into(),
                    pwr_pstate_max_dc_offset: pwr.pwr_min_duty_cycle as i32 - max_ps_scaled as i32,
                    max_pstate_scaled_2: max_ps_scaled,
                    max_power_2: pwr.max_power_mw,
                    max_pstate_scaled_3: max_ps_scaled,
                    ppm_filter_tc_periods_x4: ppm_filter_tc_periods * 4,
                    ppm_filter_a_neg: f32!(1.0) - ppm_filter_a,
                    ppm_filter_a: ppm_filter_a,
                    ppm_ki_dt: pwr.ppm_ki * period_s,
                    unk_6fc: f32!(65536.0),
                    ppm_kp: pwr.ppm_kp,
                    pwr_min_duty_cycle: pwr.pwr_min_duty_cycle,
                    max_pstate_scaled_4: max_ps_scaled,
                    unk_71c: f32!(0.0),
                    max_power_3: pwr.max_power_mw,
                    cur_power_mw_2: 0x0,
                    ppm_filter_tc_ms: pwr.ppm_filter_time_constant_ms,
                    #[ver(V >= V13_0B4)]
                    ppm_filter_tc_clks: ppm_filter_tc_ms_rounded * base_clock_khz,
                    perf_tgt_utilization: pwr.perf_tgt_utilization,
                    perf_boost_min_util: pwr.perf_boost_min_util,
                    perf_boost_ce_step: pwr.perf_boost_ce_step,
                    perf_reset_iters: pwr.perf_reset_iters,
                    unk_774: 6,
                    unk_778: 1,
                    perf_filter_drop_threshold: pwr.perf_filter_drop_threshold,
                    perf_filter_a_neg: f32!(1.0) - perf_filter_a,
                    perf_filter_a2_neg: f32!(1.0) - perf_filter_a2,
                    perf_filter_a: perf_filter_a,
                    perf_filter_a2: perf_filter_a2,
                    perf_ki: pwr.perf_integral_gain,
                    perf_ki2: pwr.perf_integral_gain2,
                    perf_integral_min_clamp: pwr.perf_integral_min_clamp.into(),
                    unk_79c: f32!(95.0),
                    perf_kp: pwr.perf_proportional_gain,
                    perf_kp2: pwr.perf_proportional_gain2,
                    boost_state_unk_k: F32::from(boost_ps_count) / f32!(0.95),
                    base_pstate_scaled_2: base_ps_scaled,
                    max_pstate_scaled_5: max_ps_scaled,
                    base_pstate_scaled_3: base_ps_scaled,
                    perf_tgt_utilization_2: pwr.perf_tgt_utilization,
                    base_pstate_scaled_4: base_ps_scaled,
                    unk_7fc: f32!(65536.0),
                    pwr_min_duty_cycle_2: pwr.pwr_min_duty_cycle.into(),
                    max_pstate_scaled_6: max_ps_scaled.into(),
                    max_freq_mhz: pwr.max_freq_mhz,
                    pwr_min_duty_cycle_3: pwr.pwr_min_duty_cycle,
                    min_pstate_scaled_4: f32!(100.0),
                    max_pstate_scaled_7: max_ps_scaled,
                    unk_alpha_neg: f32!(0.8),
                    unk_alpha: f32!(0.2),
                    fast_die0_sensor_mask: U64(cfg.fast_sensor_mask[0]),
                    #[ver(G >= G14X)]
                    fast_die1_sensor_mask: U64(cfg.fast_sensor_mask[1]),
                    fast_die0_release_temp_cc: 100 * pwr.fast_die0_release_temp,
                    unk_87c: cfg.da.unk_87c,
                    unk_880: 0x4,
                    unk_894: f32!(1.0),

                    fast_die0_ki_dt: pwr.fast_die0_integral_gain * period_s,
                    unk_8a8: f32!(65536.0),
                    fast_die0_kp: pwr.fast_die0_proportional_gain,
                    pwr_min_duty_cycle_4: pwr.pwr_min_duty_cycle,
                    max_pstate_scaled_8: max_ps_scaled,
                    max_pstate_scaled_9: max_ps_scaled,
                    fast_die0_prop_tgt_delta: 100 * pwr.fast_die0_prop_tgt_delta,
                    unk_8cc: cfg.da.unk_8cc,
                    max_pstate_scaled_10: max_ps_scaled,
                    max_pstate_scaled_11: max_ps_scaled,
                    unk_c2c: 1,
                    power_zone_count: pwr.power_zones.len() as u32,
                    max_power_4: pwr.max_power_mw,
                    max_power_5: pwr.max_power_mw,
                    max_power_6: pwr.max_power_mw,
                    avg_power_target_filter_a_neg: f32!(1.0) - avg_power_target_filter_a,
                    avg_power_target_filter_a: avg_power_target_filter_a,
                    avg_power_target_filter_tc_x4: 4 * pwr.avg_power_target_filter_tc,
                    avg_power_target_filter_tc_xperiod: period_ms * pwr.avg_power_target_filter_tc,
                    #[ver(V >= V13_0B4)]
                    avg_power_target_filter_tc_clks: period_ms
                        * pwr.avg_power_target_filter_tc
                        * base_clock_khz,
                    avg_power_filter_tc_periods_x4: 4 * avg_power_filter_tc_periods,
                    avg_power_filter_a_neg: f32!(1.0) - avg_power_filter_a,
                    avg_power_filter_a: avg_power_filter_a,
                    avg_power_ki_dt: pwr.avg_power_ki_only * period_s,
                    unk_d20: f32!(65536.0),
                    avg_power_kp: pwr.avg_power_kp,
                    avg_power_min_duty_cycle: pwr.avg_power_min_duty_cycle,
                    max_pstate_scaled_12: max_ps_scaled,
                    max_pstate_scaled_13: max_ps_scaled,
                    max_power_7: pwr.max_power_mw.into(),
                    max_power_8: pwr.max_power_mw,
                    avg_power_filter_tc_ms: pwr.avg_power_filter_tc_ms,
                    #[ver(V >= V13_0B4)]
                    avg_power_filter_tc_clks: avg_power_filter_tc_ms_rounded * base_clock_khz,
                    max_pstate_scaled_14: max_ps_scaled,
                    t81xx_data <- Self::t81xx_data(cfg, dyncfg),
                    #[ver(V >= V13_0B4)]
                    unk_e10_0 <- {
                        let filter_a = f32!(1.0) / pwr.se_filter_time_constant.into();
                        let filter_1_a = f32!(1.0) / pwr.se_filter_time_constant_1.into();
                        try_init!(raw::HwDataA130Extra {
                            unk_38: 4,
                            unk_3c: 8000,
                            gpu_se_inactive_threshold: pwr.se_inactive_threshold,
                            gpu_se_engagement_criteria: pwr.se_engagement_criteria,
                            gpu_se_reset_criteria: pwr.se_reset_criteria,
                            unk_54: 50,
                            unk_58: 0x1,
                            gpu_se_filter_a_neg: f32!(1.0) - filter_a,
                            gpu_se_filter_1_a_neg: f32!(1.0) - filter_1_a,
                            gpu_se_filter_a: filter_a,
                            gpu_se_filter_1_a: filter_1_a,
                            gpu_se_ki_dt: pwr.se_ki * period_s,
                            gpu_se_ki_1_dt: pwr.se_ki_1 * period_s,
                            unk_7c: f32!(65536.0),
                            gpu_se_kp: pwr.se_kp,
                            gpu_se_kp_1: pwr.se_kp_1,

                            #[ver(V >= V13_3)]
                            unk_8c: 100,
                            #[ver(V < V13_3)]
                            unk_8c: 40,

                            max_pstate_scaled_1: max_ps_scaled,
                            unk_9c: f32!(8000.0),
                            unk_a0: 1400,
                            gpu_se_filter_time_constant_ms: pwr.se_filter_time_constant * period_ms,
                            gpu_se_filter_time_constant_1_ms: pwr.se_filter_time_constant_1
                                * period_ms,
                            gpu_se_filter_time_constant_clks: U64((pwr.se_filter_time_constant
                                * clocks_per_period_coarse)
                                .into()),
                            gpu_se_filter_time_constant_1_clks: U64((pwr
                                .se_filter_time_constant_1
                                * clocks_per_period_coarse)
                                .into()),
                            unk_c4: f32!(65536.0),
                            unk_114: f32!(65536.0),
                            unk_124: 40,
                            max_pstate_scaled_2: max_ps_scaled,
                            ..Zeroable::init_zeroed()
                        })
                    },
                    fast_die0_sensor_mask_2: U64(cfg.fast_sensor_mask[0]),
                    #[ver(G >= G14X)]
                    fast_die1_sensor_mask_2: U64(cfg.fast_sensor_mask[1]),
                    unk_e24: cfg.da.unk_e24,
                    unk_e28: 1,
                    fast_die0_sensor_mask_alt: U64(cfg.fast_sensor_mask_alt[0]),
                    #[ver(G >= G14X)]
                    fast_die1_sensor_mask_alt: U64(cfg.fast_sensor_mask_alt[1]),
                    #[ver(V < V13_0B4)]
                    fast_die0_sensor_present: U64(cfg.fast_die0_sensor_present as u64),
                    unk_163c: 1,
                    unk_3644: 0,
                    hws1 <- Self::hw_shared1(cfg),
                    hws2 <- Self::hw_shared2(cfg, dyncfg),
                    hws3 <- Self::hw_shared3(cfg),
                    unk_3ce8: 1,
                    ..Zeroable::init_zeroed()
                })
                .chain(|raw| {
                    for i in 0..self.dyncfg.pwr.perf_states.len() {
                        raw.sram_k[i] = self.cfg.sram_k;
                    }
                    if *crate::module_parameters::dump_hwdata.value() == 2 {
                        // .667zzz24: field offsets, to align with Apple's 15.5 layout.
                        pr_info!("hwa-off {} {:#x}\n", "unk_0", core::mem::offset_of!(raw::HwDataA::ver, unk_0));
                        pr_info!("hwa-off {} {:#x}\n", "clocks_per_period", core::mem::offset_of!(raw::HwDataA::ver, clocks_per_period));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "clocks_per_period_2", core::mem::offset_of!(raw::HwDataA::ver, clocks_per_period_2));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8", core::mem::offset_of!(raw::HwDataA::ver, unk_8));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_status", core::mem::offset_of!(raw::HwDataA::ver, pwr_status));
                        pr_info!("hwa-off {} {:#x}\n", "unk_10", core::mem::offset_of!(raw::HwDataA::ver, unk_10));
                        pr_info!("hwa-off {} {:#x}\n", "unk_14", core::mem::offset_of!(raw::HwDataA::ver, unk_14));
                        pr_info!("hwa-off {} {:#x}\n", "unk_18", core::mem::offset_of!(raw::HwDataA::ver, unk_18));
                        pr_info!("hwa-off {} {:#x}\n", "unk_1c", core::mem::offset_of!(raw::HwDataA::ver, unk_1c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_20", core::mem::offset_of!(raw::HwDataA::ver, unk_20));
                        pr_info!("hwa-off {} {:#x}\n", "unk_24", core::mem::offset_of!(raw::HwDataA::ver, unk_24));
                        pr_info!("hwa-off {} {:#x}\n", "actual_pstate", core::mem::offset_of!(raw::HwDataA::ver, actual_pstate));
                        pr_info!("hwa-off {} {:#x}\n", "tgt_pstate", core::mem::offset_of!(raw::HwDataA::ver, tgt_pstate));
                        pr_info!("hwa-off {} {:#x}\n", "unk_30", core::mem::offset_of!(raw::HwDataA::ver, unk_30));
                        pr_info!("hwa-off {} {:#x}\n", "cur_pstate", core::mem::offset_of!(raw::HwDataA::ver, cur_pstate));
                        pr_info!("hwa-off {} {:#x}\n", "unk_38", core::mem::offset_of!(raw::HwDataA::ver, unk_38));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_3c_0", core::mem::offset_of!(raw::HwDataA::ver, unk_3c_0));
                        pr_info!("hwa-off {} {:#x}\n", "base_pstate_scaled", core::mem::offset_of!(raw::HwDataA::ver, base_pstate_scaled));
                        pr_info!("hwa-off {} {:#x}\n", "unk_40", core::mem::offset_of!(raw::HwDataA::ver, unk_40));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled));
                        pr_info!("hwa-off {} {:#x}\n", "unk_48", core::mem::offset_of!(raw::HwDataA::ver, unk_48));
                        pr_info!("hwa-off {} {:#x}\n", "min_pstate_scaled", core::mem::offset_of!(raw::HwDataA::ver, min_pstate_scaled));
                        pr_info!("hwa-off {} {:#x}\n", "freq_mhz", core::mem::offset_of!(raw::HwDataA::ver, freq_mhz));
                        pr_info!("hwa-off {} {:#x}\n", "unk_54", core::mem::offset_of!(raw::HwDataA::ver, unk_54));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_74_0", core::mem::offset_of!(raw::HwDataA::ver, unk_74_0));
                        pr_info!("hwa-off {} {:#x}\n", "sram_k", core::mem::offset_of!(raw::HwDataA::ver, sram_k));
                        pr_info!("hwa-off {} {:#x}\n", "unk_b4", core::mem::offset_of!(raw::HwDataA::ver, unk_b4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_1b4", core::mem::offset_of!(raw::HwDataA::ver, unk_1b4));
                        pr_info!("hwa-off {} {:#x}\n", "temp_c", core::mem::offset_of!(raw::HwDataA::ver, temp_c));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_mw", core::mem::offset_of!(raw::HwDataA::ver, avg_power_mw));
                        pr_info!("hwa-off {} {:#x}\n", "update_ts", core::mem::offset_of!(raw::HwDataA::ver, update_ts));
                        pr_info!("hwa-off {} {:#x}\n", "unk_1c8", core::mem::offset_of!(raw::HwDataA::ver, unk_1c8));
                        pr_info!("hwa-off {} {:#x}\n", "unk_1cc", core::mem::offset_of!(raw::HwDataA::ver, unk_1cc));
                        pr_info!("hwa-off {} {:#x}\n", "pad_644", core::mem::offset_of!(raw::HwDataA::ver, pad_644));
                        pr_info!("hwa-off {} {:#x}\n", "unk_64c", core::mem::offset_of!(raw::HwDataA::ver, unk_64c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_650", core::mem::offset_of!(raw::HwDataA::ver, unk_650));
                        pr_info!("hwa-off {} {:#x}\n", "pad_654", core::mem::offset_of!(raw::HwDataA::ver, pad_654));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_filter_a_neg", core::mem::offset_of!(raw::HwDataA::ver, pwr_filter_a_neg));
                        pr_info!("hwa-off {} {:#x}\n", "pad_65c", core::mem::offset_of!(raw::HwDataA::ver, pad_65c));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_filter_a", core::mem::offset_of!(raw::HwDataA::ver, pwr_filter_a));
                        pr_info!("hwa-off {} {:#x}\n", "pad_664", core::mem::offset_of!(raw::HwDataA::ver, pad_664));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_integral_gain", core::mem::offset_of!(raw::HwDataA::ver, pwr_integral_gain));
                        pr_info!("hwa-off {} {:#x}\n", "pad_66c", core::mem::offset_of!(raw::HwDataA::ver, pad_66c));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_integral_min_clamp", core::mem::offset_of!(raw::HwDataA::ver, pwr_integral_min_clamp));
                        pr_info!("hwa-off {} {:#x}\n", "max_power_1", core::mem::offset_of!(raw::HwDataA::ver, max_power_1));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_proportional_gain", core::mem::offset_of!(raw::HwDataA::ver, pwr_proportional_gain));
                        pr_info!("hwa-off {} {:#x}\n", "pad_67c", core::mem::offset_of!(raw::HwDataA::ver, pad_67c));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_pstate_related_k", core::mem::offset_of!(raw::HwDataA::ver, pwr_pstate_related_k));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_pstate_max_dc_offset", core::mem::offset_of!(raw::HwDataA::ver, pwr_pstate_max_dc_offset));
                        pr_info!("hwa-off {} {:#x}\n", "unk_688", core::mem::offset_of!(raw::HwDataA::ver, unk_688));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_2", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_2));
                        pr_info!("hwa-off {} {:#x}\n", "pad_690", core::mem::offset_of!(raw::HwDataA::ver, pad_690));
                        pr_info!("hwa-off {} {:#x}\n", "unk_694", core::mem::offset_of!(raw::HwDataA::ver, unk_694));
                        pr_info!("hwa-off {} {:#x}\n", "max_power_2", core::mem::offset_of!(raw::HwDataA::ver, max_power_2));
                        pr_info!("hwa-off {} {:#x}\n", "pad_69c", core::mem::offset_of!(raw::HwDataA::ver, pad_69c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_6b4", core::mem::offset_of!(raw::HwDataA::ver, unk_6b4));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_6b8_0", core::mem::offset_of!(raw::HwDataA::ver, unk_6b8_0));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_3", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_3));
                        pr_info!("hwa-off {} {:#x}\n", "unk_6bc", core::mem::offset_of!(raw::HwDataA::ver, unk_6bc));
                        pr_info!("hwa-off {} {:#x}\n", "pad_6c0", core::mem::offset_of!(raw::HwDataA::ver, pad_6c0));
                        pr_info!("hwa-off {} {:#x}\n", "ppm_filter_tc_periods_x4", core::mem::offset_of!(raw::HwDataA::ver, ppm_filter_tc_periods_x4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_6d8", core::mem::offset_of!(raw::HwDataA::ver, unk_6d8));
                        pr_info!("hwa-off {} {:#x}\n", "pad_6dc", core::mem::offset_of!(raw::HwDataA::ver, pad_6dc));
                        pr_info!("hwa-off {} {:#x}\n", "ppm_filter_a_neg", core::mem::offset_of!(raw::HwDataA::ver, ppm_filter_a_neg));
                        pr_info!("hwa-off {} {:#x}\n", "pad_6e4", core::mem::offset_of!(raw::HwDataA::ver, pad_6e4));
                        pr_info!("hwa-off {} {:#x}\n", "ppm_filter_a", core::mem::offset_of!(raw::HwDataA::ver, ppm_filter_a));
                        pr_info!("hwa-off {} {:#x}\n", "pad_6ec", core::mem::offset_of!(raw::HwDataA::ver, pad_6ec));
                        pr_info!("hwa-off {} {:#x}\n", "ppm_ki_dt", core::mem::offset_of!(raw::HwDataA::ver, ppm_ki_dt));
                        pr_info!("hwa-off {} {:#x}\n", "pad_6f4", core::mem::offset_of!(raw::HwDataA::ver, pad_6f4));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_integral_min_clamp_2", core::mem::offset_of!(raw::HwDataA::ver, pwr_integral_min_clamp_2));
                        pr_info!("hwa-off {} {:#x}\n", "unk_6fc", core::mem::offset_of!(raw::HwDataA::ver, unk_6fc));
                        pr_info!("hwa-off {} {:#x}\n", "ppm_kp", core::mem::offset_of!(raw::HwDataA::ver, ppm_kp));
                        pr_info!("hwa-off {} {:#x}\n", "pad_704", core::mem::offset_of!(raw::HwDataA::ver, pad_704));
                        pr_info!("hwa-off {} {:#x}\n", "unk_708", core::mem::offset_of!(raw::HwDataA::ver, unk_708));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_min_duty_cycle", core::mem::offset_of!(raw::HwDataA::ver, pwr_min_duty_cycle));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_4", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_714", core::mem::offset_of!(raw::HwDataA::ver, unk_714));
                        pr_info!("hwa-off {} {:#x}\n", "pad_718", core::mem::offset_of!(raw::HwDataA::ver, pad_718));
                        pr_info!("hwa-off {} {:#x}\n", "unk_71c", core::mem::offset_of!(raw::HwDataA::ver, unk_71c));
                        pr_info!("hwa-off {} {:#x}\n", "max_power_3", core::mem::offset_of!(raw::HwDataA::ver, max_power_3));
                        pr_info!("hwa-off {} {:#x}\n", "cur_power_mw_2", core::mem::offset_of!(raw::HwDataA::ver, cur_power_mw_2));
                        pr_info!("hwa-off {} {:#x}\n", "ppm_filter_tc_ms", core::mem::offset_of!(raw::HwDataA::ver, ppm_filter_tc_ms));
                        pr_info!("hwa-off {} {:#x}\n", "unk_72c", core::mem::offset_of!(raw::HwDataA::ver, unk_72c));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "ppm_filter_tc_clks", core::mem::offset_of!(raw::HwDataA::ver, ppm_filter_tc_clks));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_730_4", core::mem::offset_of!(raw::HwDataA::ver, unk_730_4));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_730_8", core::mem::offset_of!(raw::HwDataA::ver, unk_730_8));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_730_c", core::mem::offset_of!(raw::HwDataA::ver, unk_730_c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_730", core::mem::offset_of!(raw::HwDataA::ver, unk_730));
                        pr_info!("hwa-off {} {:#x}\n", "unk_734", core::mem::offset_of!(raw::HwDataA::ver, unk_734));
                        pr_info!("hwa-off {} {:#x}\n", "unk_738", core::mem::offset_of!(raw::HwDataA::ver, unk_738));
                        pr_info!("hwa-off {} {:#x}\n", "unk_73c", core::mem::offset_of!(raw::HwDataA::ver, unk_73c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_740", core::mem::offset_of!(raw::HwDataA::ver, unk_740));
                        pr_info!("hwa-off {} {:#x}\n", "unk_744", core::mem::offset_of!(raw::HwDataA::ver, unk_744));
                        pr_info!("hwa-off {} {:#x}\n", "unk_748", core::mem::offset_of!(raw::HwDataA::ver, unk_748));
                        pr_info!("hwa-off {} {:#x}\n", "unk_758", core::mem::offset_of!(raw::HwDataA::ver, unk_758));
                        pr_info!("hwa-off {} {:#x}\n", "perf_tgt_utilization", core::mem::offset_of!(raw::HwDataA::ver, perf_tgt_utilization));
                        pr_info!("hwa-off {} {:#x}\n", "pad_760", core::mem::offset_of!(raw::HwDataA::ver, pad_760));
                        pr_info!("hwa-off {} {:#x}\n", "perf_boost_min_util", core::mem::offset_of!(raw::HwDataA::ver, perf_boost_min_util));
                        pr_info!("hwa-off {} {:#x}\n", "perf_boost_ce_step", core::mem::offset_of!(raw::HwDataA::ver, perf_boost_ce_step));
                        pr_info!("hwa-off {} {:#x}\n", "perf_reset_iters", core::mem::offset_of!(raw::HwDataA::ver, perf_reset_iters));
                        pr_info!("hwa-off {} {:#x}\n", "pad_770", core::mem::offset_of!(raw::HwDataA::ver, pad_770));
                        pr_info!("hwa-off {} {:#x}\n", "unk_774", core::mem::offset_of!(raw::HwDataA::ver, unk_774));
                        pr_info!("hwa-off {} {:#x}\n", "unk_778", core::mem::offset_of!(raw::HwDataA::ver, unk_778));
                        pr_info!("hwa-off {} {:#x}\n", "perf_filter_drop_threshold", core::mem::offset_of!(raw::HwDataA::ver, perf_filter_drop_threshold));
                        pr_info!("hwa-off {} {:#x}\n", "perf_filter_a_neg", core::mem::offset_of!(raw::HwDataA::ver, perf_filter_a_neg));
                        pr_info!("hwa-off {} {:#x}\n", "perf_filter_a2_neg", core::mem::offset_of!(raw::HwDataA::ver, perf_filter_a2_neg));
                        pr_info!("hwa-off {} {:#x}\n", "perf_filter_a", core::mem::offset_of!(raw::HwDataA::ver, perf_filter_a));
                        pr_info!("hwa-off {} {:#x}\n", "perf_filter_a2", core::mem::offset_of!(raw::HwDataA::ver, perf_filter_a2));
                        pr_info!("hwa-off {} {:#x}\n", "perf_ki", core::mem::offset_of!(raw::HwDataA::ver, perf_ki));
                        pr_info!("hwa-off {} {:#x}\n", "perf_ki2", core::mem::offset_of!(raw::HwDataA::ver, perf_ki2));
                        pr_info!("hwa-off {} {:#x}\n", "perf_integral_min_clamp", core::mem::offset_of!(raw::HwDataA::ver, perf_integral_min_clamp));
                        pr_info!("hwa-off {} {:#x}\n", "unk_79c", core::mem::offset_of!(raw::HwDataA::ver, unk_79c));
                        pr_info!("hwa-off {} {:#x}\n", "perf_kp", core::mem::offset_of!(raw::HwDataA::ver, perf_kp));
                        pr_info!("hwa-off {} {:#x}\n", "perf_kp2", core::mem::offset_of!(raw::HwDataA::ver, perf_kp2));
                        pr_info!("hwa-off {} {:#x}\n", "boost_state_unk_k", core::mem::offset_of!(raw::HwDataA::ver, boost_state_unk_k));
                        pr_info!("hwa-off {} {:#x}\n", "base_pstate_scaled_2", core::mem::offset_of!(raw::HwDataA::ver, base_pstate_scaled_2));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_5", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_5));
                        pr_info!("hwa-off {} {:#x}\n", "base_pstate_scaled_3", core::mem::offset_of!(raw::HwDataA::ver, base_pstate_scaled_3));
                        pr_info!("hwa-off {} {:#x}\n", "pad_7b8", core::mem::offset_of!(raw::HwDataA::ver, pad_7b8));
                        pr_info!("hwa-off {} {:#x}\n", "perf_cur_utilization", core::mem::offset_of!(raw::HwDataA::ver, perf_cur_utilization));
                        pr_info!("hwa-off {} {:#x}\n", "perf_tgt_utilization_2", core::mem::offset_of!(raw::HwDataA::ver, perf_tgt_utilization_2));
                        pr_info!("hwa-off {} {:#x}\n", "pad_7c4", core::mem::offset_of!(raw::HwDataA::ver, pad_7c4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_7dc", core::mem::offset_of!(raw::HwDataA::ver, unk_7dc));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_7e0_0", core::mem::offset_of!(raw::HwDataA::ver, unk_7e0_0));
                        pr_info!("hwa-off {} {:#x}\n", "base_pstate_scaled_4", core::mem::offset_of!(raw::HwDataA::ver, base_pstate_scaled_4));
                        pr_info!("hwa-off {} {:#x}\n", "pad_7e4", core::mem::offset_of!(raw::HwDataA::ver, pad_7e4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_7e8", core::mem::offset_of!(raw::HwDataA::ver, unk_7e8));
                        pr_info!("hwa-off {} {:#x}\n", "unk_7fc", core::mem::offset_of!(raw::HwDataA::ver, unk_7fc));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_min_duty_cycle_2", core::mem::offset_of!(raw::HwDataA::ver, pwr_min_duty_cycle_2));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_6", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_6));
                        pr_info!("hwa-off {} {:#x}\n", "max_freq_mhz", core::mem::offset_of!(raw::HwDataA::ver, max_freq_mhz));
                        pr_info!("hwa-off {} {:#x}\n", "pad_80c", core::mem::offset_of!(raw::HwDataA::ver, pad_80c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_810", core::mem::offset_of!(raw::HwDataA::ver, unk_810));
                        pr_info!("hwa-off {} {:#x}\n", "pad_814", core::mem::offset_of!(raw::HwDataA::ver, pad_814));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_min_duty_cycle_3", core::mem::offset_of!(raw::HwDataA::ver, pwr_min_duty_cycle_3));
                        pr_info!("hwa-off {} {:#x}\n", "unk_81c", core::mem::offset_of!(raw::HwDataA::ver, unk_81c));
                        pr_info!("hwa-off {} {:#x}\n", "pad_820", core::mem::offset_of!(raw::HwDataA::ver, pad_820));
                        pr_info!("hwa-off {} {:#x}\n", "min_pstate_scaled_4", core::mem::offset_of!(raw::HwDataA::ver, min_pstate_scaled_4));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_7", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_7));
                        pr_info!("hwa-off {} {:#x}\n", "unk_82c", core::mem::offset_of!(raw::HwDataA::ver, unk_82c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_alpha_neg", core::mem::offset_of!(raw::HwDataA::ver, unk_alpha_neg));
                        pr_info!("hwa-off {} {:#x}\n", "unk_alpha", core::mem::offset_of!(raw::HwDataA::ver, unk_alpha));
                        pr_info!("hwa-off {} {:#x}\n", "unk_838", core::mem::offset_of!(raw::HwDataA::ver, unk_838));
                        pr_info!("hwa-off {} {:#x}\n", "unk_83c", core::mem::offset_of!(raw::HwDataA::ver, unk_83c));
                        pr_info!("hwa-off {} {:#x}\n", "pad_840", core::mem::offset_of!(raw::HwDataA::ver, pad_840));
                        pr_info!("hwa-off {} {:#x}\n", "unk_86c", core::mem::offset_of!(raw::HwDataA::ver, unk_86c));
                        pr_info!("hwa-off {} {:#x}\n", "fast_die0_sensor_mask", core::mem::offset_of!(raw::HwDataA::ver, fast_die0_sensor_mask));
                        #[ver(G >= G14X)]
                        pr_info!("hwa-off {} {:#x}\n", "fast_die1_sensor_mask", core::mem::offset_of!(raw::HwDataA::ver, fast_die1_sensor_mask));
                        pr_info!("hwa-off {} {:#x}\n", "fast_die0_release_temp_cc", core::mem::offset_of!(raw::HwDataA::ver, fast_die0_release_temp_cc));
                        pr_info!("hwa-off {} {:#x}\n", "unk_87c", core::mem::offset_of!(raw::HwDataA::ver, unk_87c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_880", core::mem::offset_of!(raw::HwDataA::ver, unk_880));
                        pr_info!("hwa-off {} {:#x}\n", "unk_884", core::mem::offset_of!(raw::HwDataA::ver, unk_884));
                        pr_info!("hwa-off {} {:#x}\n", "pad_888", core::mem::offset_of!(raw::HwDataA::ver, pad_888));
                        pr_info!("hwa-off {} {:#x}\n", "unk_88c", core::mem::offset_of!(raw::HwDataA::ver, unk_88c));
                        pr_info!("hwa-off {} {:#x}\n", "pad_890", core::mem::offset_of!(raw::HwDataA::ver, pad_890));
                        pr_info!("hwa-off {} {:#x}\n", "unk_894", core::mem::offset_of!(raw::HwDataA::ver, unk_894));
                        pr_info!("hwa-off {} {:#x}\n", "pad_898", core::mem::offset_of!(raw::HwDataA::ver, pad_898));
                        pr_info!("hwa-off {} {:#x}\n", "fast_die0_ki_dt", core::mem::offset_of!(raw::HwDataA::ver, fast_die0_ki_dt));
                        pr_info!("hwa-off {} {:#x}\n", "pad_8a0", core::mem::offset_of!(raw::HwDataA::ver, pad_8a0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8a4", core::mem::offset_of!(raw::HwDataA::ver, unk_8a4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8a8", core::mem::offset_of!(raw::HwDataA::ver, unk_8a8));
                        pr_info!("hwa-off {} {:#x}\n", "fast_die0_kp", core::mem::offset_of!(raw::HwDataA::ver, fast_die0_kp));
                        pr_info!("hwa-off {} {:#x}\n", "pad_8b0", core::mem::offset_of!(raw::HwDataA::ver, pad_8b0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8b4", core::mem::offset_of!(raw::HwDataA::ver, unk_8b4));
                        pr_info!("hwa-off {} {:#x}\n", "pwr_min_duty_cycle_4", core::mem::offset_of!(raw::HwDataA::ver, pwr_min_duty_cycle_4));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_8", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_8));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_9", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_9));
                        pr_info!("hwa-off {} {:#x}\n", "fast_die0_prop_tgt_delta", core::mem::offset_of!(raw::HwDataA::ver, fast_die0_prop_tgt_delta));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8c8", core::mem::offset_of!(raw::HwDataA::ver, unk_8c8));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8cc", core::mem::offset_of!(raw::HwDataA::ver, unk_8cc));
                        pr_info!("hwa-off {} {:#x}\n", "pad_8d0", core::mem::offset_of!(raw::HwDataA::ver, pad_8d0));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_8e4_0", core::mem::offset_of!(raw::HwDataA::ver, unk_8e4_0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8e4", core::mem::offset_of!(raw::HwDataA::ver, unk_8e4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8e8", core::mem::offset_of!(raw::HwDataA::ver, unk_8e8));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_10", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_10));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8f0", core::mem::offset_of!(raw::HwDataA::ver, unk_8f0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_8f4", core::mem::offset_of!(raw::HwDataA::ver, unk_8f4));
                        pr_info!("hwa-off {} {:#x}\n", "pad_8f8", core::mem::offset_of!(raw::HwDataA::ver, pad_8f8));
                        pr_info!("hwa-off {} {:#x}\n", "pad_8fc", core::mem::offset_of!(raw::HwDataA::ver, pad_8fc));
                        pr_info!("hwa-off {} {:#x}\n", "unk_900", core::mem::offset_of!(raw::HwDataA::ver, unk_900));
                        pr_info!("hwa-off {} {:#x}\n", "unk_coef_a1", core::mem::offset_of!(raw::HwDataA::ver, unk_coef_a1));
                        pr_info!("hwa-off {} {:#x}\n", "unk_coef_a2", core::mem::offset_of!(raw::HwDataA::ver, unk_coef_a2));
                        pr_info!("hwa-off {} {:#x}\n", "pad_b24", core::mem::offset_of!(raw::HwDataA::ver, pad_b24));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_11", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_11));
                        pr_info!("hwa-off {} {:#x}\n", "freq_with_off", core::mem::offset_of!(raw::HwDataA::ver, freq_with_off));
                        pr_info!("hwa-off {} {:#x}\n", "unk_b9c", core::mem::offset_of!(raw::HwDataA::ver, unk_b9c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_ba0", core::mem::offset_of!(raw::HwDataA::ver, unk_ba0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_ba8", core::mem::offset_of!(raw::HwDataA::ver, unk_ba8));
                        pr_info!("hwa-off {} {:#x}\n", "unk_bb0", core::mem::offset_of!(raw::HwDataA::ver, unk_bb0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_bb4", core::mem::offset_of!(raw::HwDataA::ver, unk_bb4));
                        #[ver(V >= V13_3)]
                        pr_info!("hwa-off {} {:#x}\n", "pad_bb8_0", core::mem::offset_of!(raw::HwDataA::ver, pad_bb8_0));
                        #[ver(V >= V13_5)]
                        pr_info!("hwa-off {} {:#x}\n", "pad_bb8_200", core::mem::offset_of!(raw::HwDataA::ver, pad_bb8_200));
                        pr_info!("hwa-off {} {:#x}\n", "pad_bb8", core::mem::offset_of!(raw::HwDataA::ver, pad_bb8));
                        pr_info!("hwa-off {} {:#x}\n", "unk_c2c", core::mem::offset_of!(raw::HwDataA::ver, unk_c2c));
                        pr_info!("hwa-off {} {:#x}\n", "power_zone_count", core::mem::offset_of!(raw::HwDataA::ver, power_zone_count));
                        pr_info!("hwa-off {} {:#x}\n", "max_power_4", core::mem::offset_of!(raw::HwDataA::ver, max_power_4));
                        pr_info!("hwa-off {} {:#x}\n", "max_power_5", core::mem::offset_of!(raw::HwDataA::ver, max_power_5));
                        pr_info!("hwa-off {} {:#x}\n", "max_power_6", core::mem::offset_of!(raw::HwDataA::ver, max_power_6));
                        pr_info!("hwa-off {} {:#x}\n", "unk_c40", core::mem::offset_of!(raw::HwDataA::ver, unk_c40));
                        pr_info!("hwa-off {} {:#x}\n", "unk_c44", core::mem::offset_of!(raw::HwDataA::ver, unk_c44));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_target_filter_a_neg", core::mem::offset_of!(raw::HwDataA::ver, avg_power_target_filter_a_neg));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_target_filter_a", core::mem::offset_of!(raw::HwDataA::ver, avg_power_target_filter_a));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_target_filter_tc_x4", core::mem::offset_of!(raw::HwDataA::ver, avg_power_target_filter_tc_x4));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_target_filter_tc_xperiod", core::mem::offset_of!(raw::HwDataA::ver, avg_power_target_filter_tc_xperiod));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_target_filter_tc_clks", core::mem::offset_of!(raw::HwDataA::ver, avg_power_target_filter_tc_clks));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_c58_4", core::mem::offset_of!(raw::HwDataA::ver, unk_c58_4));
                        pr_info!("hwa-off {} {:#x}\n", "power_zones", core::mem::offset_of!(raw::HwDataA::ver, power_zones));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_filter_tc_periods_x4", core::mem::offset_of!(raw::HwDataA::ver, avg_power_filter_tc_periods_x4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_cfc", core::mem::offset_of!(raw::HwDataA::ver, unk_cfc));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d00", core::mem::offset_of!(raw::HwDataA::ver, unk_d00));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_filter_a_neg", core::mem::offset_of!(raw::HwDataA::ver, avg_power_filter_a_neg));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d08", core::mem::offset_of!(raw::HwDataA::ver, unk_d08));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_filter_a", core::mem::offset_of!(raw::HwDataA::ver, avg_power_filter_a));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d10", core::mem::offset_of!(raw::HwDataA::ver, unk_d10));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_ki_dt", core::mem::offset_of!(raw::HwDataA::ver, avg_power_ki_dt));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d18", core::mem::offset_of!(raw::HwDataA::ver, unk_d18));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d1c", core::mem::offset_of!(raw::HwDataA::ver, unk_d1c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d20", core::mem::offset_of!(raw::HwDataA::ver, unk_d20));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_kp", core::mem::offset_of!(raw::HwDataA::ver, avg_power_kp));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d28", core::mem::offset_of!(raw::HwDataA::ver, unk_d28));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d2c", core::mem::offset_of!(raw::HwDataA::ver, unk_d2c));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_min_duty_cycle", core::mem::offset_of!(raw::HwDataA::ver, avg_power_min_duty_cycle));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_12", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_12));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_13", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_13));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d3c", core::mem::offset_of!(raw::HwDataA::ver, unk_d3c));
                        pr_info!("hwa-off {} {:#x}\n", "max_power_7", core::mem::offset_of!(raw::HwDataA::ver, max_power_7));
                        pr_info!("hwa-off {} {:#x}\n", "max_power_8", core::mem::offset_of!(raw::HwDataA::ver, max_power_8));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d48", core::mem::offset_of!(raw::HwDataA::ver, unk_d48));
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_filter_tc_ms", core::mem::offset_of!(raw::HwDataA::ver, avg_power_filter_tc_ms));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d50", core::mem::offset_of!(raw::HwDataA::ver, unk_d50));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "avg_power_filter_tc_clks", core::mem::offset_of!(raw::HwDataA::ver, avg_power_filter_tc_clks));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_d54_4", core::mem::offset_of!(raw::HwDataA::ver, unk_d54_4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d54", core::mem::offset_of!(raw::HwDataA::ver, unk_d54));
                        pr_info!("hwa-off {} {:#x}\n", "max_pstate_scaled_14", core::mem::offset_of!(raw::HwDataA::ver, max_pstate_scaled_14));
                        pr_info!("hwa-off {} {:#x}\n", "unk_d68", core::mem::offset_of!(raw::HwDataA::ver, unk_d68));
                        pr_info!("hwa-off {} {:#x}\n", "t81xx_data", core::mem::offset_of!(raw::HwDataA::ver, t81xx_data));
                        pr_info!("hwa-off {} {:#x}\n", "unk_dd0", core::mem::offset_of!(raw::HwDataA::ver, unk_dd0));
                        #[ver(V >= V13_2)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_e10_pad", core::mem::offset_of!(raw::HwDataA::ver, unk_e10_pad));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_e10_0", core::mem::offset_of!(raw::HwDataA::ver, unk_e10_0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_e10", core::mem::offset_of!(raw::HwDataA::ver, unk_e10));
                        pr_info!("hwa-off {} {:#x}\n", "fast_die0_sensor_mask_2", core::mem::offset_of!(raw::HwDataA::ver, fast_die0_sensor_mask_2));
                        #[ver(G >= G14X)]
                        pr_info!("hwa-off {} {:#x}\n", "fast_die1_sensor_mask_2", core::mem::offset_of!(raw::HwDataA::ver, fast_die1_sensor_mask_2));
                        pr_info!("hwa-off {} {:#x}\n", "unk_e24", core::mem::offset_of!(raw::HwDataA::ver, unk_e24));
                        pr_info!("hwa-off {} {:#x}\n", "unk_e28", core::mem::offset_of!(raw::HwDataA::ver, unk_e28));
                        pr_info!("hwa-off {} {:#x}\n", "unk_e2c", core::mem::offset_of!(raw::HwDataA::ver, unk_e2c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_coef_b1", core::mem::offset_of!(raw::HwDataA::ver, unk_coef_b1));
                        pr_info!("hwa-off {} {:#x}\n", "unk_coef_b2", core::mem::offset_of!(raw::HwDataA::ver, unk_coef_b2));
                        #[ver(G >= G14X)]
                        pr_info!("hwa-off {} {:#x}\n", "pad_1048_0", core::mem::offset_of!(raw::HwDataA::ver, pad_1048_0));
                        pr_info!("hwa-off {} {:#x}\n", "pad_1048", core::mem::offset_of!(raw::HwDataA::ver, pad_1048));
                        pr_info!("hwa-off {} {:#x}\n", "fast_die0_sensor_mask_alt", core::mem::offset_of!(raw::HwDataA::ver, fast_die0_sensor_mask_alt));
                        #[ver(G >= G14X)]
                        pr_info!("hwa-off {} {:#x}\n", "fast_die1_sensor_mask_alt", core::mem::offset_of!(raw::HwDataA::ver, fast_die1_sensor_mask_alt));
                        #[ver(V < V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "fast_die0_sensor_present", core::mem::offset_of!(raw::HwDataA::ver, fast_die0_sensor_present));
                        pr_info!("hwa-off {} {:#x}\n", "unk_163c", core::mem::offset_of!(raw::HwDataA::ver, unk_163c));
                        pr_info!("hwa-off {} {:#x}\n", "unk_1640", core::mem::offset_of!(raw::HwDataA::ver, unk_1640));
                        #[ver(G >= G14X)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_3640_0", core::mem::offset_of!(raw::HwDataA::ver, unk_3640_0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3640", core::mem::offset_of!(raw::HwDataA::ver, unk_3640));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3644", core::mem::offset_of!(raw::HwDataA::ver, unk_3644));
                        pr_info!("hwa-off {} {:#x}\n", "hws1", core::mem::offset_of!(raw::HwDataA::ver, hws1));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_hws2", core::mem::offset_of!(raw::HwDataA::ver, unk_hws2));
                        pr_info!("hwa-off {} {:#x}\n", "hws2", core::mem::offset_of!(raw::HwDataA::ver, hws2));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3c00", core::mem::offset_of!(raw::HwDataA::ver, unk_3c00));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3c04", core::mem::offset_of!(raw::HwDataA::ver, unk_3c04));
                        pr_info!("hwa-off {} {:#x}\n", "hws3", core::mem::offset_of!(raw::HwDataA::ver, hws3));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3c58", core::mem::offset_of!(raw::HwDataA::ver, unk_3c58));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3c94", core::mem::offset_of!(raw::HwDataA::ver, unk_3c94));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3c98", core::mem::offset_of!(raw::HwDataA::ver, unk_3c98));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3ca0", core::mem::offset_of!(raw::HwDataA::ver, unk_3ca0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3ca8", core::mem::offset_of!(raw::HwDataA::ver, unk_3ca8));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3cb0", core::mem::offset_of!(raw::HwDataA::ver, unk_3cb0));
                        pr_info!("hwa-off {} {:#x}\n", "ts_last_idle", core::mem::offset_of!(raw::HwDataA::ver, ts_last_idle));
                        pr_info!("hwa-off {} {:#x}\n", "ts_last_poweron", core::mem::offset_of!(raw::HwDataA::ver, ts_last_poweron));
                        pr_info!("hwa-off {} {:#x}\n", "ts_last_poweroff", core::mem::offset_of!(raw::HwDataA::ver, ts_last_poweroff));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3cd0", core::mem::offset_of!(raw::HwDataA::ver, unk_3cd0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3cd8", core::mem::offset_of!(raw::HwDataA::ver, unk_3cd8));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_3ce0_0", core::mem::offset_of!(raw::HwDataA::ver, unk_3ce0_0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3ce0", core::mem::offset_of!(raw::HwDataA::ver, unk_3ce0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3ce4", core::mem::offset_of!(raw::HwDataA::ver, unk_3ce4));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3ce8", core::mem::offset_of!(raw::HwDataA::ver, unk_3ce8));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3cec", core::mem::offset_of!(raw::HwDataA::ver, unk_3cec));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3cf0", core::mem::offset_of!(raw::HwDataA::ver, unk_3cf0));
                        pr_info!("hwa-off {} {:#x}\n", "core_leak_coef", core::mem::offset_of!(raw::HwDataA::ver, core_leak_coef));
                        pr_info!("hwa-off {} {:#x}\n", "sram_leak_coef", core::mem::offset_of!(raw::HwDataA::ver, sram_leak_coef));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "aux_leak_coef", core::mem::offset_of!(raw::HwDataA::ver, aux_leak_coef));
                        #[ver(V >= V13_0B4)]
                        pr_info!("hwa-off {} {:#x}\n", "unk_3d34_0", core::mem::offset_of!(raw::HwDataA::ver, unk_3d34_0));
                        pr_info!("hwa-off {} {:#x}\n", "unk_3d34", core::mem::offset_of!(raw::HwDataA::ver, unk_3d34));
                        #[ver(V >= V15_5)]
                        pr_info!("hwa-off {} {:#x}\n", "pad_15_5", core::mem::offset_of!(raw::HwDataA::ver, pad_15_5));
                    }
                    if *crate::module_parameters::dump_hwdata.value() != 0 {
                        // SAFETY: plain-old-data struct, read as bytes.
                        let b = unsafe {
                            core::slice::from_raw_parts(
                                raw as *const raw::HwDataA::ver as *const u8,
                                core::mem::size_of::<raw::HwDataA::ver>(),
                            )
                        };
                        dump_nonzero("A", b, 0xe00);
                    }

                    for (i, coef) in pwr.core_leak_coef.iter().enumerate() {
                        raw.core_leak_coef[i] = *coef;
                    }

                    for (i, coef) in pwr.sram_leak_coef.iter().enumerate() {
                        raw.sram_leak_coef[i] = *coef;
                    }

                    #[ver(V >= V13_0B4)]
                    if let Some(csafr) = pwr.csafr.as_ref() {
                        for (i, coef) in csafr.leak_coef_afr.iter().enumerate() {
                            raw.aux_leak_coef.cs_1[i] = *coef;
                            raw.aux_leak_coef.cs_2[i] = *coef;
                        }

                        for (i, coef) in csafr.leak_coef_cs.iter().enumerate() {
                            raw.aux_leak_coef.afr_1[i] = *coef;
                            raw.aux_leak_coef.afr_2[i] = *coef;
                        }
                    }

                    for i in 0..self.dyncfg.id.num_clusters as usize {
                        if let Some(coef_a) = self.cfg.unk_coef_a.get(i) {
                            (*raw.unk_coef_a1[i])[..coef_a.len()].copy_from_slice(coef_a);
                            (*raw.unk_coef_a2[i])[..coef_a.len()].copy_from_slice(coef_a);
                        }
                        if let Some(coef_b) = self.cfg.unk_coef_b.get(i) {
                            (*raw.unk_coef_b1[i])[..coef_b.len()].copy_from_slice(coef_b);
                            (*raw.unk_coef_b2[i])[..coef_b.len()].copy_from_slice(coef_b);
                        }
                    }

                    for (i, pz) in pwr.power_zones.iter().enumerate() {
                        raw.power_zones[i].target = pz.target;
                        raw.power_zones[i].target_off = pz.target - pz.target_offset;
                        raw.power_zones[i].filter_tc_x4 = 4 * pz.filter_tc;
                        raw.power_zones[i].filter_tc_xperiod = period_ms * pz.filter_tc;
                        let filter_a = f32!(1.0) / pz.filter_tc.into();
                        raw.power_zones[i].filter_a = filter_a;
                        raw.power_zones[i].filter_a_neg = f32!(1.0) - filter_a;
                        #[ver(V >= V13_0B4)]
                        raw.power_zones[i].unk_10 = 1320000000;
                    }

                    #[ver(V >= V13_0B4 && G >= G14X)]
                    for (i, j) in raw.hws2.g14.curve2.t1.iter().enumerate() {
                        raw.unk_hws2[i] = if *j == 0xffff { 0 } else { j / 2 };
                    }

                    if !dyncfg.hw_data_b.is_empty() {
                        unsafe {
                            let mut matches: bool = true;
                            let sla = core::slice::from_raw_parts(
                                raw as *const raw::HwDataA::ver as *const u8,
                                core::mem::size_of::<raw::HwDataA::ver>(),
                            );
                            if sla.len() != dyncfg.hw_data_a.len() {
                                matches = false;
                                dev_err!(
                                    self.dev.as_ref(),
                                    "!!! Hwdata A size mismatch: {} {}",
                                    sla.len(),
                                    dyncfg.hw_data_a.len(),
                                );
                            }
                            // Report every differing run, not just the first
                            // offset. m1n1 built the reference copy in the
                            // hw-cal-a carveout and the firmware accepted that
                            // one in the m1n1-side experiments, so each run
                            // here names a field this driver computes
                            // differently -- which is the shortlist for why
                            // the firmware's DC_Init never completes.
                            let n = core::cmp::min(sla.len(), dyncfg.hw_data_a.len());
                            let mut i = 0;
                            let mut runs = 0;
                            while i < n {
                                if sla[i] != dyncfg.hw_data_a[i] {
                                    matches = false;
                                    let start = i;
                                    while i < n && sla[i] != dyncfg.hw_data_a[i] {
                                        i += 1;
                                    }
                                    runs += 1;
                                    if runs <= 12 {
                                        dev_err!(
                                            self.dev.as_ref(),
                                            "!!! Hwdata A differs at {:#x}..{:#x} ({} bytes): ours {:02x?} m1n1 {:02x?}\n",
                                            start,
                                            i,
                                            i - start,
                                            &sla[start..core::cmp::min(start + 8, i)],
                                            &dyncfg.hw_data_a[start..core::cmp::min(start + 8, i)],
                                        );
                                    }
                                } else {
                                    i += 1;
                                }
                            }
                            if runs > 12 {
                                dev_err!(self.dev.as_ref(), "!!! Hwdata A: {} differing runs total\n", runs);
                            }
                            if matches {
                                dev_info!(self.dev.as_ref(), "!!! Hwdata A match");
                            }
                        }
                    }

                    Ok(())
                })
            })
    }

    /// Create the HwDataB structure. This mostly contains GPU-related configuration.
    ///
    /// ⚠️ `gpu_ro`, not `private`. A GPU *hardware* unit reads this structure:
    /// with the VA-0 read papered over, `gUPM` faults `NoAccess` at
    /// `0xfffffe0000009680`, which is HwDataB's own base address, and
    /// `NoAccess` rather than `Unmapped` because `PROT_FW_PRIV_RW` grants the
    /// firmware read/write and the GPU nothing. `PROT_GPU_RO_FW_PRIV_RW` keeps
    /// the same cached memattr and the same firmware access and adds the GPU
    /// read the hardware needs. LOG.md 32.41.71.
    fn hwdata_b(&mut self) -> Result<GpuObject<HwDataB::ver>> {
        self.alloc
            .gpu_ro
            .new_init(pin_init::init_zeroed(), |_inner, _ptr| {
                let cfg = &self.cfg;
                let dyncfg = &self.dyncfg;
                try_init!(raw::HwDataB::ver {
                    // Userspace VA map related
                    #[ver(V < V13_0B4)]
                    unk_0: U64(0x13_00000000),
                    unk_8: U64(0x14_00000000),
                    #[ver(V < V13_0B4)]
                    unk_10: U64(0x1_00000000),
                    unk_18: U64(0xffc00000),
                    // USC start
                    unk_20: U64(0), // U64(0x11_00000000),
                    unk_28: U64(0), // U64(0x11_00000000),
                    // Unknown page
                    //unk_30: U64(0x6f_ffff8000),
                    unk_30: U64(mmu::IOVA_UNK_PAGE),
                    timestamp_area_base: U64(if *crate::module_parameters::ts_dbg.value() == 1 {
                        gpu::IOVA_KERN_SHARED_RANGE.start
                    } else {
                        gpu::IOVA_KERN_TIMESTAMP_RANGE.start
                    }),
                    // TODO: yuv matrices
                    chip_id: cfg.chip_id,
                    unk_454: cfg.db.unk_454,
                    unk_458: 0x1,
                    unk_460: 0x1,
                    unk_464: 0x1,
                    unk_468: 0x1,
                    unk_47c: 0x1,
                    unk_484: 0x1,
                    unk_48c: 0x1,
                    base_clock_khz: cfg.base_clock_hz / 1000,
                    power_sample_period: dyncfg.pwr.power_sample_period,
                    unk_49c: 0x1,
                    unk_4a0: 0x1,
                    unk_4a4: 0x1,
                    unk_4c0: 0x1f,
                    unk_4e0: U64(cfg.db.unk_4e0),
                    unk_4f0: 0x1,
                    unk_4f4: 0x1,
                    unk_504: 0x31,
                    unk_524: 0x1, // use_secure_cache_flush
                    unk_534: cfg.db.unk_534,
                    num_frags: dyncfg.id.num_frags * dyncfg.id.num_clusters,
                    // 🔬 `num_gps` -- 4 per cluster on t6030, 8 total -- is
                    // read out of the ID registers, checked against
                    // `max_num_gps`, printed, and then **never sent to the
                    // firmware**.  The tiler runs on the GPs.  `unk_540` is the
                    // word directly after `num_frags`, zero here and in m1n1
                    // (which only ever ran G13).  LOG.md 32.41.270.
                    unk_540: *crate::module_parameters::initdata_unk_540.value(),
                    unk_554: 0x1,
                    uat_ttb_base: U64(dyncfg.uat_ttb_base),
                    gpu_core_id: cfg.gpu_core as u32,
                    gpu_rev_id: dyncfg.id.gpu_rev_id as u32,
                    num_cores: dyncfg.id.num_cores * dyncfg.id.num_clusters,
                    max_pstate: dyncfg.pwr.perf_states.len() as u32 - 1,
                    #[ver(V < V13_0B4)]
                    num_pstates: dyncfg.pwr.perf_states.len() as u32,
                    #[ver(V < V13_0B4)]
                    min_sram_volt: dyncfg.pwr.min_sram_microvolt / 1000,
                    #[ver(V < V13_0B4)]
                    unk_ab8: cfg.db.unk_ab8,
                    #[ver(V < V13_0B4)]
                    unk_abc: cfg.db.unk_abc,
                    #[ver(V < V13_0B4)]
                    unk_ac0: 0x1020,

                    #[ver(V >= V13_0B4)]
                    unk_ae4: Array::new([0x0, 0x3, 0x7, 0x7]),
                    #[ver(V < V13_0B4)]
                    unk_ae4: Array::new([0x0, 0xf, 0x3f, 0x3f]),
                    unk_b10: 0x1,
                    timer_offset: U64(0),
                    unk_b24: 0x1,
                    unk_b28: 0x1,
                    unk_b2c: 0x1,
                    unk_b30: cfg.db.unk_b30,
                    #[ver(V >= V13_0B4)]
                    unk_b38_0: 1,
                    #[ver(V >= V13_0B4)]
                    unk_b38_4: 1,
                    unk_b38: Array::new([0xffffffff; 12]),
                    #[ver(V >= V13_0B4 && V < V13_3)]
                    unk_c3c: 0x19,
                    #[ver(V >= V13_3)]
                    unk_c3c: 0x1a,
                    ..Zeroable::init_zeroed()
                })
                .chain(|raw| {
                    #[ver(V >= V13_3)]
                    for i in 0..16 {
                        raw.unk_arr_0[i] = i as u32;
                    }

                    // LOG.md 32.41.538: the firmware restores the core-enable
                    // mask from here on every power-up.
                    // LOG.md 32.41.539: macOS 15.5's constants in the words the
                    // firmware copies out at fw:0x2b638 (kext initFirmwareData).
                    #[ver(V >= V15_5)]
                    {
                        raw.unk_464 = 0; // 0xef4
                        raw.unk_478 = 1; // 0xf08
                        raw.unk_47c = 0; // 0xf0c
                        raw.unk_488 = 1; // 0xf18
                        raw.unk_48c = 0; // 0xf1c
                    }

                    #[ver(V >= V15_5)]
                    if *crate::module_parameters::hwb_core_mask.value() != 0 {
                        for (i, m) in self.dyncfg.id.core_masks_packed.iter().take(3).enumerate() {
                            raw.core_masks_15_5[i] = *m;
                        }
                        pr_info!(
                            "asahi: HwDataB core_masks (0x1c90) = {:#x} {:#x} {:#x}\n",
                            raw.core_masks_15_5[0], raw.core_masks_15_5[1], raw.core_masks_15_5[2]
                        );
                    }

                    let base_ps = self.dyncfg.pwr.perf_base_pstate as usize;
                    let max_ps = self.dyncfg.pwr.perf_max_pstate as usize;
                    let base_freq = self.dyncfg.pwr.perf_states[base_ps].freq_hz;
                    let max_freq = self.dyncfg.pwr.perf_states[max_ps].freq_hz;

                    for (i, ps) in self.dyncfg.pwr.perf_states.iter().enumerate() {
                        raw.frequencies[i] = ps.freq_hz / 1000000;
                        for (j, mv) in ps.volt_mv.iter().enumerate() {
                            let sram_mv = (*mv).max(self.dyncfg.pwr.min_sram_microvolt / 1000);
                            raw.voltages[i][j] = *mv;
                            raw.voltages_sram[i][j] = sram_mv;
                        }
                        for j in ps.volt_mv.len()..raw.voltages[i].len() {
                            raw.voltages[i][j] = raw.voltages[i][0];
                            raw.voltages_sram[i][j] = raw.voltages_sram[i][0];
                        }
                        raw.sram_k[i] = self.cfg.sram_k;
                        raw.rel_max_powers[i] = ps.pwr_mw * 100 / self.dyncfg.pwr.max_power_mw;
                        // .631: max_pstate == base_pstate makes the divisor 0
                        raw.rel_boost_freqs[i] = if i > base_ps && max_freq > base_freq {
                            (ps.freq_hz - base_freq) / ((max_freq - base_freq) / 100)
                        } else {
                            0
                        };
                    }

                    #[ver(V >= V13_0B4)]
                    if let Some(csafr) = self.dyncfg.pwr.csafr.as_ref() {
                        let aux = &mut raw.aux_ps;
                        aux.cs_max_pstate = (csafr.perf_states_cs.len() - 1).try_into()?;
                        aux.afr_max_pstate = (csafr.perf_states_afr.len() - 1).try_into()?;

                        for (i, ps) in csafr.perf_states_cs.iter().enumerate() {
                            aux.cs_frequencies[i] = ps.freq_hz / 1000000;
                            for (j, mv) in ps.volt_mv.iter().enumerate() {
                                let sram_mv = (*mv).max(csafr.min_sram_microvolt / 1000);
                                aux.cs_voltages[i][j] = *mv;
                                aux.cs_voltages_sram[i][j] = sram_mv;
                            }
                        }

                        for (i, ps) in csafr.perf_states_afr.iter().enumerate() {
                            aux.afr_frequencies[i] = ps.freq_hz / 1000000;
                            for (j, mv) in ps.volt_mv.iter().enumerate() {
                                let sram_mv = (*mv).max(csafr.min_sram_microvolt / 1000);
                                aux.afr_voltages[i][j] = *mv;
                                aux.afr_voltages_sram[i][j] = sram_mv;
                            }
                        }
                    }

                    // Special case override for T602x
                    #[ver(G == G14X)]
                    if dyncfg.id.gpu_rev_id == hw::GpuRevisionID::B1 {
                        raw.gpu_rev_id = hw::GpuRevisionID::B0 as u32;
                    }

                    if *crate::module_parameters::dump_hwdata.value() != 0 {
                        // SAFETY: plain-old-data struct, read as bytes.
                        let b = unsafe {
                            core::slice::from_raw_parts(
                                raw as *const raw::HwDataB::ver as *const u8,
                                core::mem::size_of::<raw::HwDataB::ver>(),
                            )
                        };
                        dump_nonzero("B", b, usize::MAX);
                    }
                    if !dyncfg.hw_data_b.is_empty() {
                        unsafe {
                            let mut matches: bool = true;
                            let sla = core::slice::from_raw_parts(
                                raw as *const raw::HwDataB::ver as *const u8,
                                core::mem::size_of::<raw::HwDataB::ver>(),
                            );
                            if sla.len() != dyncfg.hw_data_b.len() {
                                matches = false;
                                dev_err!(
                                    self.dev.as_ref(),
                                    "!!! Hwdata B size mismatch: {} {}",
                                    sla.len(),
                                    dyncfg.hw_data_b.len(),
                                );
                            }
                            // See the equivalent loop in hwdata_a().
                            let n = core::cmp::min(sla.len(), dyncfg.hw_data_b.len());
                            let mut i = 0;
                            let mut runs = 0;
                            while i < n {
                                if sla[i] != dyncfg.hw_data_b[i] {
                                    matches = false;
                                    let start = i;
                                    while i < n && sla[i] != dyncfg.hw_data_b[i] {
                                        i += 1;
                                    }
                                    runs += 1;
                                    if runs <= 12 {
                                        dev_err!(
                                            self.dev.as_ref(),
                                            "!!! Hwdata B differs at {:#x}..{:#x} ({} bytes): ours {:02x?} m1n1 {:02x?}\n",
                                            start,
                                            i,
                                            i - start,
                                            &sla[start..core::cmp::min(start + 8, i)],
                                            &dyncfg.hw_data_b[start..core::cmp::min(start + 8, i)],
                                        );
                                    }
                                } else {
                                    i += 1;
                                }
                            }
                            if runs > 12 {
                                dev_err!(self.dev.as_ref(), "!!! Hwdata B: {} differing runs total\n", runs);
                            }
                            if matches {
                                dev_info!(self.dev.as_ref(), "!!! Hwdata B match");
                            }
                        }
                    }

                    Ok(())
                })
            })
    }

    /// Create the Globals structure, which contains global firmware config including more power
    /// configuration data and globals used to exchange state between the firmware and driver.
    fn globals(&mut self) -> Result<GpuObject<Globals::ver>> {
        self.alloc
            .private
            .new_init(pin_init::init_zeroed(), |_inner, _ptr| {
                let cfg = &self.cfg;
                let dyncfg = &self.dyncfg;
                let pwr = &dyncfg.pwr;
                let period_ms = pwr.power_sample_period;
                let period_s = F32::from(period_ms) / f32!(1000.0);
                let avg_power_filter_tc_periods = pwr.avg_power_filter_tc_ms / period_ms;

                let max_ps = pwr.perf_max_pstate;
                let max_ps_scaled = 100 * max_ps;

                try_init!(raw::Globals::ver {
                    // 🔑 This is the firmware's *trace category mask*, not a
                    // boolean.  It lands in the byte at fw:0x1a383c, which 262
                    // logging sites across the firmware test with
                    //
                    //     adrp x8, 0x1a3000 ; ldrb w8,[x8,#0x83c] ; tbz w8,#N,skip
                    //
                    // before calling the trace emitter at fw:0x201f8.  Bits 0,1,
                    // 2, 4, 5 and 6 are used; every one of the 147 gated blocks
                    // contains nothing but that call, so turning them on adds
                    // trace records and changes no behaviour.  The records come
                    // back on the ktrace ring, which `KTraceChannel::poll`
                    // already drains -- print them with debug_flags bit 17.
                    // LOG.md 32.41.112.
                    ktrace_enable: *crate::module_parameters::ktrace_mask.value(),
                    #[ver(V >= V13_2)]
                    unk_24_0: 3000,
                    unk_24: 0,
                    #[ver(V >= V13_0B4)]
                    debug: 0,
                    unk_28: 1,
                    #[ver(G >= G14X)]
                    unk_2c_0: 1,
                    #[ver(V >= V13_0B4 && G < G14X)]
                    unk_2c_0: 0,
                    unk_2c: 1,
                    unk_30: 0,
                    unk_34: 120,
                    // sub <- try_init!(raw::GlobalsSub::ver {
                        unk_54: cfg.global_unk_54,
                        unk_56: 40,
                        unk_58: 0xffff,
                        unk_5e: U32(1),
                        unk_66: U32(1),
                    //     ..Zeroable::init_zeroed()
                    // }),
                    unk_8900: 1,
                    pending_submissions: AtomicU32::new(0),
                    max_power: pwr.max_power_mw,
                    max_pstate_scaled: max_ps_scaled,
                    max_pstate_scaled_2: max_ps_scaled,
                    max_pstate_scaled_3: max_ps_scaled,
                    power_zone_count: pwr.power_zones.len() as u32,
                    avg_power_filter_tc_periods: avg_power_filter_tc_periods,
                    avg_power_ki_dt: pwr.avg_power_ki_only * period_s,
                    avg_power_kp: pwr.avg_power_kp,
                    avg_power_min_duty_cycle: pwr.avg_power_min_duty_cycle,
                    avg_power_target_filter_tc: pwr.avg_power_target_filter_tc,
                    unk_89bc: cfg.da.unk_8cc,
                    fast_die0_release_temp: 100 * pwr.fast_die0_release_temp,
                    unk_89c4: cfg.da.unk_87c,
                    fast_die0_prop_tgt_delta: 100 * pwr.fast_die0_prop_tgt_delta,
                    fast_die0_kp: pwr.fast_die0_proportional_gain,
                    fast_die0_ki_dt: pwr.fast_die0_integral_gain * period_s,
                    unk_89e0: 1,
                    max_power_2: pwr.max_power_mw,
                    ppm_kp: pwr.ppm_kp,
                    ppm_ki_dt: pwr.ppm_ki * period_s,
                    #[ver(V >= V13_0B4)]
                    unk_89f4_8: 1,
                    unk_89f4: 0,
                    hws1 <- Self::hw_shared1(cfg),
                    hws2 <- Self::hw_shared2(cfg, dyncfg),
                    hws3 <- Self::hw_shared3(cfg),
                    #[ver(V >= V13_0B4)]
                    idle_off_standby_timer: pwr.idle_off_standby_timer,
                    #[ver(V >= V13_0B4)]
                    unk_hws2_4: cfg.unk_hws2_4.map(Array::new).unwrap_or_default(),
                    #[ver(V >= V13_0B4)]
                    unk_hws2_24: cfg.unk_hws2_24,
                    unk_900c: 1,
                    #[ver(V >= V13_0B4)]
                    unk_9010_0: 1,
                    #[ver(V >= V13_0B4)]
                    unk_903c: 1,
                    #[ver(V < V13_0B4)]
                    unk_903c: 0,
                    fault_control: *module_parameters::fault_control.value(),
                    do_init: 1,
                    progress_check_interval_3d: 40,
                    progress_check_interval_ta: 10,
                    progress_check_interval_cl: 250,
                    #[ver(V >= V13_0B4)]
                    unk_1102c_0: 1,
                    #[ver(V >= V13_0B4)]
                    unk_1102c_4: 1,
                    #[ver(V >= V13_0B4)]
                    unk_1102c_8: 100,
                    #[ver(V >= V13_0B4)]
                    unk_1102c_c: 1,
                    // 32.41.559: idle_off_ms keeps the GPU powered after a job
                    // so post-job MMIO dumps read live state.
                    idle_off_delay_ms: AtomicU32::new(
                        match *crate::module_parameters::idle_off_ms.value() {
                            0 => pwr.idle_off_delay_ms,
                            v => v,
                        },
                    ),
                    fender_idle_off_delay_ms: match *crate::module_parameters::idle_off_ms.value() {
                        0 => pwr.fender_idle_off_delay_ms,
                        v => v,
                    },
                    fw_early_wake_timeout_ms: pwr.fw_early_wake_timeout_ms,
                    cl_context_switch_timeout_ms: 40,
                    #[ver(V >= V13_0B4)]
                    cl_kill_timeout_ms: 50,
                    #[ver(V >= V13_0B4)]
                    unk_11edc: 0,
                    #[ver(V >= V13_0B4)]
                    unk_11efc: 0,
                    ..Zeroable::init_zeroed()
                })
                .chain(|raw| {
                    for (i, pz) in self.dyncfg.pwr.power_zones.iter().enumerate() {
                        raw.power_zones[i].target = pz.target;
                        raw.power_zones[i].target_off = pz.target - pz.target_offset;
                        raw.power_zones[i].filter_tc = pz.filter_tc;
                    }

                    if let Some(tab) = self.cfg.global_tab.as_ref() {
                        for (i, x) in tab.iter().enumerate() {
                            raw.unk_118ec[i] = *x;
                        }
                        raw.unk_118e8 = 1;
                    }

                    if !dyncfg.hw_globals.is_empty() {
                        unsafe {
                            let mut matches: bool = true;
                            let sla = core::slice::from_raw_parts(
                                raw as *const raw::Globals::ver as *const u8,
                                core::mem::size_of::<raw::Globals::ver>(),
                            );
                            if sla.len() != dyncfg.hw_globals.len() {
                                matches = false;
                                dev_err!(
                                    self.dev.as_ref(),
                                    "!!! Globals size mismatch: {} {}",
                                    sla.len(),
                                    dyncfg.hw_globals.len(),
                                );
                            }
                            for i in 0..core::cmp::min(sla.len(), dyncfg.hw_globals.len()) {
                                if sla[i] != dyncfg.hw_globals[i] {
                                    matches = false;
                                    dev_err!(self.dev.as_ref(), "!!! Globals first mismatch: {i}");
                                    break;
                                }
                            }
                            if matches {
                                dev_info!(self.dev.as_ref(), "!!! Globals match");
                            }
                        }
                    }

                    Ok(())
                })
            })
    }

    /// Create the RuntimePointers structure, which contains pointers to most of the other
    /// structures including the ring buffer channels, statistics structures, and HwDataA/HwDataB.
    fn runtime_pointers(&mut self) -> Result<GpuObject<RuntimePointers::ver>> {
        let mut hwa = self.hwdata_a()?;
        let hwb = self.hwdata_b()?;

        // The RegionB+0x451 buffer is not opaque scratch on 15.5. The firmware
        // keeps its push-telemetry accounting there -- `0x2ff8..0x5a50` -- and
        // the very first thing that code does, before it checks whether there
        // is any telemetry to push, is dereference a *host-supplied* pointer at
        // `+0x3a30` (`0x2be7c`, then `0x2c280`):
        //
        //     ldr x8, [x11, #2616]      ; unkptr_451 + 0x3a30
        //     ldr w8, [x8]              ; <- data abort when it is null
        //     ldr w9, [x11, #2608]      ; cached copy at +0x3a28
        //     cmp w8, w9
        //
        // i.e. a generation counter the host owns and the firmware watches. The
        // macOS kext fills the same slot from a virtual call at
        // `0xfffffe00091008a4`, and zeroes `+0x2ff8` for `0x2a58` bytes just
        // before. A valid, zero page is all that is needed: the counter never
        // changes, so the compare always takes the fast path, and with the slot
        // mask at `+0x3a38` left zero nothing is pushed at all. LOG.md 32.39.
        //
        // ...and the object it is a pointer *to* is HwDataA. 15.5 moved the
        // HwDataA pointer out of the aligned slot at RegionB+0x2b0 -- which the
        // firmware never reads, on any path -- into this unaligned one, which
        // it reads from 43 different sites. Two independent checks say so: the
        // periodic power tick at 0x37d20 loads its sample period from
        // `[ptr + 4]` (HwDataA.clocks_per_period) and its state from
        // `[ptr + 0x10]` (HwDataA.pwr_status). Pointed at a zeroed scratch
        // buffer the period was 0, so the tick's deadline was always in the
        // past and the power manager ran flat out -- ~27,000 iterations a
        // second, which starved RTKit's dispatch and is why no doorbell was
        // ever delivered after init. LOG.md 32.41.
        #[ver(V >= V15_5)]
        let ptd_watch = self.alloc.shared.array_empty_tagged::<u8>(0x80, b"IPTW")?;
        #[ver(V >= V15_5)]
        {
            // The watch pointer sits at +0x3a30 inside HwDataA, in the middle
            // of `unk_3640_0`, so there is no named field to assign; patch the
            // bytes. HwDataA carries `pad_15_5` precisely so the firmware's
            // accounting window (up to +0x62a4) stays inside the object.
            let va: u64 = ptd_watch.gpu_pointer().into();
            hwa.with_mut(|raw, _inner| {
                let size = core::mem::size_of_val(raw);
                pr_info!("asahi: HwDataA is {:#x} bytes\n", size);
                if size >= 0x3a38 {
                    let p = raw as *mut _ as *mut u8;
                    // SAFETY: bounds checked against the object's own size just
                    // above, and the object is uniquely borrowed here.
                    unsafe { core::ptr::write_unaligned(p.add(0x3a30) as *mut u64, va) };
                }
            });
        }

        let mut buffer_mgr_ctl = gem::new_kernel_object(self.dev, 0x4000)?;
        buffer_mgr_ctl.vmap()?.memset(0);
        // 15.5: this is the PB descriptor table (16 B per buffer slot); CPU
        // address kept for the hang dump. LOG.md 32.41.600.
        let buffer_mgr_ctl_cpu = buffer_mgr_ctl.vmap()?.as_ptr() as usize;

        // 32.41.508: the pass-page counter page, dual-mapped like the two above.
        // Zeroed here so the firmware's read-and-clear starts from a known state.
        let mut pm_counter_obj = gem::new_kernel_object(self.dev, 0x4000)?;
        pm_counter_obj.vmap()?.memset(0);

        // The UMA page pool descriptor table. macOS sizes it 0x2000 to the PB
        // table's 0x1000 (`createHardwareMappings`, kc:0x9113d3c); one 16K page
        // covers it. LOG.md 32.41.72.
        let mut uma_pool_ctl = gem::new_kernel_object(self.dev, 0x4000)?;
        uma_pool_ctl.vmap()?.memset(0);
        // Keep the CPU address so the fault path can show whether the firmware
        // ever writes a pool descriptor here -- the only direct evidence that
        // RegionB+0x2d0/0x2d8 really is the table `fw:0x24d94` writes through.
        // The `VMap` lives in the `ObjectRef`, so the pointer stays valid for
        // as long as the object does. LOG.md 32.41.77.
        let uma_pool_ctl_cpu = uma_pool_ctl.vmap()?.as_ptr() as usize;

        GpuObject::new_init_prealloc(
            self.alloc.private.alloc_object()?,
            |_ptr| {
                let alloc = &mut *self.alloc;
                try_init!(RuntimePointers::ver {
                    stats <- {
                        let alloc = &mut *alloc;
                        try_init!(Stats::ver {
                            vtx: alloc.private.new_default::<GpuGlobalStatsVtx>()?,
                            frag: alloc.private.new_init(
                                pin_init::init_zeroed::<GpuGlobalStatsFrag::ver>(),
                                |_inner, _ptr| {
                                    try_init!(raw::GpuGlobalStatsFrag::ver {
                                        total_cmds: 0,
                                        unk_4: 0,
                                        stats: Default::default(),
                                    })
                                }
                            )?,
                            comp: alloc.private.new_default::<GpuStatsComp>()?,
                        })
                    },

                    hwdata_a: hwa,
                    unkptr_190: alloc.private.array_empty_tagged(0x80, b"I190")?,
                    unkptr_198: alloc.private.array_empty_tagged(0xc0, b"I198")?,
                    hwdata_b: hwb,

                    unkptr_1b8: alloc.private.array_empty_tagged(0x1000, b"I1B8")?,
                    unkptr_1c0: alloc.private.array_empty_tagged(0x300, b"I1C0")?,
                    unkptr_1c8: alloc.private.array_empty_tagged(0x1000, b"I1C8")?,

                    #[ver(V >= V15_5)]
                    ptd_watch,

                    // The two pointer slots in 15.5's RegionB header, 0x4000
                    // each as m1n1 allocates them.
                    #[ver(V >= V15_5)]
                    unk_8_15_5: alloc.shared.array_empty_tagged(0x4000, b"IB08")?,
                    #[ver(V >= V15_5)]
                    unk_10_15_5: alloc.shared.array_empty_tagged(0x4000, b"IB10")?,

                    // One per pipe, for RegionB+0x244..0x25c. The firmware
                    // scans 64 entries of 24 bytes through each (0x321f8) and
                    // stores through the last one unconditionally at 0x36718,
                    // so they need to be real memory it may write.
                    // Four *different* firmware structures, not a uniform
                    // per-pipe array: slot 0 carries the 64 x 24-byte scan array
                    // at +0xc (0x321f8), slot 1 is read at +0xc40 (0x38f5c) and
                    // slot 3 at +0/+4 (0x38f44). 0x4000 each covers the deepest
                    // offset seen with room to spare. LOG.md 32.16.
                    #[ver(V >= V15_5)]
                    pipe_tables: alloc.shared.array_empty_tagged(4 * 0x4000, b"IPTB")?,

                    buffer_mgr_ctl,
                    buffer_mgr_ctl_cpu,
            pm_counter_obj,
            pm_counter_low_mapping: None,
            pm_counter_high_mapping: None,
                    buffer_mgr_ctl_low_mapping: None,
                    buffer_mgr_ctl_high_mapping: None,
                    uma_pool_ctl,
                    uma_pool_ctl_cpu,
                    uma_pool_ctl_low_mapping: None,
                    uma_pool_ctl_high_mapping: None,
                })
            },
            |inner, _ptr| {
                try_init!(raw::RuntimePointers::ver {
                    // 15.5's 0x18-byte RegionB header. Slot 0 is a second copy
                    // of the hwdata_b pointer -- the firmware dereferences
                    // RegionB+0 and reads the pstate tables through it, so it
                    // must match hwdata_b below. LOG.md 8.2, 8.3.
                    #[ver(V >= V15_5)]
                    hwdata_b_0: inner.hwdata_b.gpu_pointer(),
                    // Both are pointers the firmware dereferences; zero
                    // NULL-faults it (LOG.md 30.7).
                    #[ver(V >= V15_5)]
                    unk_8_15_5: U64(inner.unk_8_15_5.gpu_pointer().into()),
                    #[ver(V >= V15_5)]
                    unk_10_15_5: U64(inner.unk_10_15_5.gpu_pointer().into()),

                    // Non-zero or the firmware never sends a single
                    // MSG_RX_DOORBELL -- see the field's comment.
                    #[ver(V >= V15_5)]
                    events_enabled: 1,
                    #[ver(V >= V15_5)]
                    // 🔑 Per pipe, the *same* object the jobs' `stats` field
                    // points into: encodeTACommandSKUStream stores
                    // fw_addr(this+600)+4 at StartVertex+0x2c, encode3D...
                    // fw_addr(this+608)+8, encodeCL... fw_addr(this+616), and
                    // initFirmwareSharedData stores those three at
                    // RegionB+0x244/0x24c/0x254 (sizes 0xc18/0x1248/0xe10).
                    // LOG.md 32.41.543.
                    pipe_tables: if *crate::module_parameters::pipe_stats.value() != 0 {
                        Array::new([
                            U64(inner.stats.vtx.gpu_pointer().into()),
                            U64(inner.stats.frag.gpu_pointer().into()),
                            U64(inner.stats.comp.gpu_pointer().into()),
                            U64(u64::from(inner.pipe_tables.gpu_pointer()) + 3 * 0x4000),
                        ])
                    } else {
                        Array::new([
                            U64(inner.pipe_tables.gpu_pointer().into()),
                            U64(u64::from(inner.pipe_tables.gpu_pointer()) + 0x4000),
                            U64(u64::from(inner.pipe_tables.gpu_pointer()) + 2 * 0x4000),
                            U64(u64::from(inner.pipe_tables.gpu_pointer()) + 3 * 0x4000),
                        ])
                    },
                    #[ver(V >= V15_5)]
                    __pad0b: Default::default(),

                    // 15.5: filled in by GpuManager::init once the channels
                    // exist. There is no event slot on this firmware, and the
                    // ktrace and stats slots are bare 16-byte rings.
                    #[ver(V >= V15_5)]
                    fwlog_payload: None,
                    #[ver(V >= V15_5)]
                    fwlog_index: Default::default(),
                    #[ver(V >= V15_5)]
                    fwlog_overflow: Default::default(),

                    pipes: Default::default(),
                    device_control: Default::default(),
                    #[ver(V < V15_5)]
                    event: Default::default(),
                    #[ver(V < V15_5)]
                    fw_log: Default::default(),
                    #[ver(V < V15_5)]
                    ktrace: Default::default(),
                    #[ver(V >= V15_5)]
                    event_15_5: Default::default(),
                    #[ver(V >= V15_5)]
                    fw_log_15_5: Default::default(),
                    #[ver(V >= V15_5)]
                    ktrace_15_5: Default::default(),
                    #[ver(V >= V15_5)]
                    stats_15_5: Default::default(),
                    #[ver(V < V15_5)]
                    stats: Default::default(),

                    stats_vtx: inner.stats.vtx.gpu_pointer(),
                    stats_frag: inner.stats.frag.gpu_pointer(),
                    stats_comp: inner.stats.comp.gpu_pointer(),

                    hwdata_a: inner.hwdata_a.gpu_pointer(),
                    unkptr_190: inner.unkptr_190.gpu_pointer(),
                    #[ver(V < V15_5)]
                    unkptr_198: inner.unkptr_198.gpu_pointer(),
                    #[ver(V < V15_5)]
                    hwdata_b: inner.hwdata_b.gpu_pointer(),
                    #[ver(V < V15_5)]
                    hwdata_b_2: inner.hwdata_b.gpu_pointer(),

                    #[ver(V < V15_5)]
                    fwlog_buf: None,

                    #[ver(V < V15_5)]
                    unkptr_1b8: inner.unkptr_1b8.gpu_pointer(),

                    // 15.5: this is the descriptor-table + config block, not
                    // five pointers. LOG.md 32.41.73.
                    #[ver(V >= V15_5)]
                    buffer_mgr_ctl_gpu_addr_15_5: U64(gpu::IOVA_KERN_GPU_BUFMGR_LOW),
                    #[ver(V >= V15_5)]
                    buffer_mgr_ctl_fw_addr_15_5: U64(gpu::IOVA_KERN_GPU_BUFMGR_HIGH),
                    #[ver(V >= V15_5)]
                    uma_pool_ctl_gpu_addr: U64(
                        match *crate::module_parameters::uma_pool_ctl.value() {
                            0 => 0,
                            _ => gpu::IOVA_KERN_GPU_UMAPOOL_LOW,
                        },
                    ),
                    #[ver(V >= V15_5)]
                    uma_pool_ctl_fw_addr: U64(
                        match *crate::module_parameters::uma_pool_ctl.value() {
                            0 => 0,
                            _ => gpu::IOVA_KERN_GPU_UMAPOOL_HIGH,
                        },
                    ),
                    // 4 = `halGetDefaultUscMaxTgmem()`. Anything else -- zero
                    // included -- is an override the firmware writes straight
                    // into the config RAM. LOG.md 32.41.73.
                    #[ver(V >= V15_5)]
                    usc_max_tgmem: *crate::module_parameters::usc_max_tgmem.value(),
                    #[ver(V >= V15_5)]
                    unk_2e4: 0,

                    #[ver(G < G14X)]
                    unkptr_1c0: inner.unkptr_1c0.gpu_pointer(),
                    #[ver(G < G14X)]
                    unkptr_1c8: inner.unkptr_1c8.gpu_pointer(),

                    #[ver(V < V15_5)]
                    __pad0: Default::default(),

                    unk_160: U64(0),
                    unk_168: U64(0),

                    unk_1d0: 0,
                    unk_1d4: 0,
                    unk_1d8: Default::default(),

                    #[ver(V < V15_5)]
                    buffer_mgr_ctl_gpu_addr: U64(gpu::IOVA_KERN_GPU_BUFMGR_LOW),
                    #[ver(V < V15_5)]
                    buffer_mgr_ctl_fw_addr: U64(gpu::IOVA_KERN_GPU_BUFMGR_HIGH),
                    #[ver(V >= V15_5)]
                    unk_32c: Default::default(),

                    __pad1: Default::default(),
                    gpu_scratch: raw::RuntimeScratch::ver {
                        unk_6b38: 0xff,
                        #[ver(V >= V15_5)]
                        unkptr_451: U64(inner.hwdata_a.gpu_pointer().into()),
                        ..Default::default()
                    },
                })
            },
        )
    }

    /// Create the FwStatus structure, which is used to coordinate the firmware halt state between
    /// the firmware and the driver.
    fn fw_status(&mut self) -> Result<GpuObject<FwStatus>> {
        self.alloc
            .shared
            .new_object(Default::default(), |_inner| Default::default())
    }

    /// Create one UatLevelInfo structure, which describes one level of translation for the UAT MMU.
    fn uat_level_info(
        cfg: &'static hw::HwConfig,
        index_shift: usize,
        num_entries: usize,
    ) -> raw::UatLevelInfo {
        raw::UatLevelInfo {
            index_shift: index_shift as _,
            unk_1: 14,
            unk_2: 14,
            unk_3: 8,
            unk_4: 0x4000,
            num_entries: num_entries as _,
            unk_8: U64(1),
            unk_10: U64(((1u64 << cfg.uat_oas) - 1) & !(mmu::UAT_PGMSK as u64)),
            index_mask: U64(((num_entries - 1) << index_shift) as u64),
        }
    }

    /// Build the top-level InitData object.
    #[inline(never)]
    pub(crate) fn build(&mut self) -> Result<KBox<GpuObject<InitData::ver>>> {
        let runtime_pointers = self.runtime_pointers()?;
        let globals = self.globals()?;
        let fw_status = self.fw_status()?;
        let shared_ro = &mut self.alloc.shared_ro;

        let obj = self.alloc.private.new_init(
            try_init!(InitData::ver {
                unk_buf: shared_ro.array_empty_tagged(0x4000, b"IDTA")?,
                runtime_pointers,
                globals,
                fw_status,

                // 15.5's four InitData tail pointers. +0xb0 is indexed out to at
                // least 0xcdcc by the DC_Init path, so it gets 0x20000; the
                // other three take 0x4000, which is what m1n1 gave them.
                //
                // All four matter: LOG.md 9.3 measured the firmware reading
                // +0xb8 out to +0x330, and m1n1 -- which got the firmware
                // *further* into its own init than the driver ever has -- backed
                // every one of them. Leaving b8/c0 at zero means any path that
                // dereferences them reads from a null base. LOG.md 32.14.
                #[ver(V >= V15_5)]
                // Word 0 is the 15.5 ktrace category mask: fw:0x2d00 does
                // `ldr x8, [0x1a2f08]` (= InitData+0xa8, copied at 0x2b5e0),
                // `ldr w9, [x8]`, `str w9, [0x1a383c]` -- the mask every
                // `0x201f8` emitter tests. Globals.ktrace_enable no longer
                // reaches it, which is why 32.41.339 saw zero records.
                // LOG.md 32.41.543.
                unkbuf_a8: {
                    let mut buf = shared_ro.array_empty_tagged::<u8>(0x4000, b"IDA8")?;
                    buf.as_mut_slice()[0..4].copy_from_slice(
                        &(*crate::module_parameters::ktrace_mask.value()).to_le_bytes(),
                    );
                    buf
                },
                // b0[0x4fc4] is a request flag, not scratch: `0x30634` and
                // `0x5f40` both test it, and each one initialises the DAG
                // subsystem -- memsetting the 128-entry slot table at 0xe4980
                // and installing the per-command-kind handler pointers from
                // 0x5ce00 into 0x120878 -- and then zeroes the flag again. Left
                // clear, `0x120878` stays null and the first command dispatched
                // takes a `blraaz` through it. LOG.md 32.41.9.
                #[ver(V >= V15_5)]
                unkbuf_b0: {
                    let mut buf = shared_ro.array_empty_tagged::<u8>(0x20000, b"IDB0")?;
                    buf.as_mut_slice()[0x4fc4] = 1;
                    buf
                },
                #[ver(V >= V15_5)]
                unkbuf_b8: shared_ro.array_empty_tagged(0x4000, b"IDB8")?,
                #[ver(V >= V15_5)]
                unkbuf_c0: shared_ro.array_empty_tagged(0x4000, b"IDC0")?,
            }),
            |inner, _ptr| {
                let cfg = &self.cfg;
                try_init!(raw::InitData::ver {
                    #[ver(V == V13_5 && G != G14X)]
                    ver_info: Array::new([0x6ba0, 0x1f28, 0x601, 0xb0]),
                    #[ver(V == V13_5 && G == G14X)]
                    ver_info: Array::new([0xb390, 0x70f8, 0x601, 0xb0]),
                    // 15.5 / G15S. Read out of the firmware's own comparison
                    // instruction -- it is an exact-match magic, not a version
                    // triple, and a mismatch panics with "Host and FW Interface
                    // is mismatched". Confirmed on hardware: with this value the
                    // check passes and the firmware consumes InitData.
                    #[ver(V == V15_5 && G == G15X)]
                    ver_info: Array::new([0x04a0, 0x8386, 0xe2f5, 0x0c8c]),
                    unk_buf: inner.unk_buf.gpu_pointer(),
                    unk_8: 0,
                    unk_c: 0,
                    runtime_pointers: inner.runtime_pointers.gpu_pointer(),
                    globals: inner.globals.gpu_pointer(),
                    #[ver(V < V15_5)]
                    fw_status: inner.fw_status.gpu_pointer(),

                    // 15.5 reads host_mapped_fw_allocations here, not at +0xa8.
                    // Apple's builder stores the constant {0, 1}. LOG.md 10.2.
                    #[ver(V >= V15_5)]
                    unk_28: 0,
                    #[ver(V >= V15_5)]
                    host_mapped_fw_allocations_15_5: 1,

                    uat_page_size: 0x4000,
                    uat_page_bits: 14,
                    uat_num_levels: 3,
                    uat_level_info: Array::new([
                        Self::uat_level_info(cfg, 36, 8),
                        Self::uat_level_info(cfg, 25, 2048),
                        Self::uat_level_info(cfg, 14, 2048),
                    ]),
                    __pad0: Default::default(),

                    #[ver(V < V15_5)]
                    host_mapped_fw_allocations: 1,
                    #[ver(V < V15_5)]
                    unk_ac: 0,
                    #[ver(V < V15_5)]
                    unk_b0: 0,
                    #[ver(V < V15_5)]
                    unk_b4: 0,
                    #[ver(V < V15_5)]
                    unk_b8: 0,

                    #[ver(V >= V15_5)]
                    unkptr_a8: inner.unkbuf_a8.gpu_pointer(),
                    #[ver(V >= V15_5)]
                    unkptr_b0: inner.unkbuf_b0.gpu_pointer(),
                    #[ver(V >= V15_5)]
                    unkptr_b8: inner.unkbuf_b8.gpu_pointer(),
                    #[ver(V >= V15_5)]
                    unkptr_c0: inner.unkbuf_c0.gpu_pointer(),
                })
            },
        )?;
        Ok(KBox::new(obj, GFP_KERNEL)?)
    }
}
