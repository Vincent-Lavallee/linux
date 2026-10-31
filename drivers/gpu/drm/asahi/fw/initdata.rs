// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU initialization / global structures

use super::channels;
use super::types::*;
use crate::{
    default_zeroed,
    gem,
    mmu,
    no_debug,
    trivial_gpustruct, //
};

pub(crate) mod raw {
    use super::*;

    /// 15.5 doubled this from 16 to 32 bytes: it carries **two** state/ring
    /// pairs, not one. Read off the FWLog entry (RegionB+0x1d8) in the running
    /// firmware -- slots 0/2 are dereferenced as channel state (READ_PTR at +0,
    /// WRITE_PTR at +0x20) and slots 1/3 as rings indexed `<<6`. The firmware
    /// produces into **pair 2**, which is why m1n1 saw nothing until it looked
    /// there. See m3-gpu-display/LOG.md 9.6 and 17.1.
    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct ChannelRing<T: GpuStruct + Debug + Default, U: Copy> {
        pub(crate) state: Option<GpuWeakPointer<T>>,
        pub(crate) ring: Option<GpuWeakPointer<[U]>>,
    }

    /// The firmware-facing channel slot.
    ///
    /// 13.5 and earlier: exactly a `ChannelRing`, 16 bytes.
    /// 15.5: 32 bytes, four pointer-sized fields. Their meaning depends on
    /// which way the channel runs, so build one with [`ChannelSlot::new_tx`] or
    /// [`ChannelSlot::new_rx`] rather than filling it in by hand.
    ///
    /// Kept separate from `ChannelRing` so the generic channel plumbing in
    /// channel.rs stays version-agnostic.
    #[versions(AGX)]
    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct ChannelSlot<T: GpuStruct + Debug + Default, U: Copy> {
        pub(crate) pair1: ChannelRing<T, U>,

        #[ver(V >= V15_5)]
        pub(crate) pair2: ChannelRing<T, U>,
    }

    #[versions(AGX)]
    impl<T: GpuStruct + Debug + Default, U: Copy> ChannelSlot::ver<T, U> {
        /// Build a slot for a channel the **firmware** produces into: event,
        /// fwlog, ktrace, stats.
        ///
        /// The firmware reads the second pair's state as a whole `ChannelState`
        /// (read_ptr +0, write_ptr +0x20) and the second ring as the message
        /// buffer -- see the event producer at 0x30a54, which indexes
        /// `*(slot+0x10)` by `sub * 0x30` and then reads +0x20. Pointing both
        /// pairs at the ring the driver owns is therefore already right.
        /// LOG.md 9.6, 17.1.
        pub(crate) fn new_rx(r: ChannelRing<T, U>) -> Self {
            Self {
                #[ver(V >= V15_5)]
                pair2: ChannelRing {
                    state: r.state,
                    ring: r.ring,
                },
                pair1: r,
            }
        }

        /// Build a slot for a channel the **driver** produces into: the twelve
        /// pipe slots and device control.
        ///
        /// 15.5 does not read these as two `ChannelRing`s at all. It takes three
        /// pointers to bare `u32` counters plus the ring base:
        ///
        /// ```text
        ///   +0x00 -> read_ptr    firmware bumps it as it consumes   (0x32d48)
        ///   +0x08 -> read_ack    firmware stores the drained wptr   (0x32d1c)
        ///   +0x10 -> write_ptr   compared against +0x00 for work    (0x61f0)
        ///   +0x18 -> ring base   entry = ring + idx * sizeof(msg)   (0x32d6c)
        /// ```
        ///
        /// Aliasing the two pairs -- which is what reading the slot as two
        /// `ChannelRing`s produces -- puts the *read* pointer in the write
        /// pointer's slot. `*(+0x00)` and `*(+0x10)` then dereference to the
        /// same `u32`, the comparison at 0x61f0 never differs, and the firmware
        /// concludes every ring is permanently empty. That is exactly the
        /// observed "fw rptr stays 0 while our wptr climbs", and it is why *no*
        /// channel worked -- device control timed out the same way. LOG.md 32.
        pub(crate) fn new_tx(r: ChannelRing<T, U>) -> Self {
            Self {
                #[ver(V >= V15_5)]
                pair2: ChannelRing {
                    // SAFETY: T is a ChannelState, whose write_ptr sits at
                    // +0x20 (see raw::ChannelState), so this stays inside the
                    // object `r.state` points at.
                    state: r
                        .state
                        .map(|s| unsafe { s.offset(0x20, core::ptr::null::<T>()) }),
                    ring: r.ring,
                },
                pair1: ChannelRing {
                    state: r.state,
                    // The read-ack counter. +0x04 is inside the padding between
                    // read_ptr and write_ptr, which nothing else uses, so the
                    // firmware's store lands somewhere harmless and stays in the
                    // object the other two pointers already refer to.
                    // SAFETY: as above.
                    #[ver(V >= V15_5)]
                    ring: r.state.map(|s| unsafe { s.offset_slice(0x4) }),
                    #[ver(V < V15_5)]
                    ring: r.ring,
                },
            }
        }
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct PipeChannels {
        pub(crate) vtx: ChannelSlot::ver<channels::ChannelState, channels::PipeMsg::ver>,
        pub(crate) frag: ChannelSlot::ver<channels::ChannelState, channels::PipeMsg::ver>,
        pub(crate) comp: ChannelSlot::ver<channels::ChannelState, channels::PipeMsg::ver>,
    }
    #[versions(AGX)]
    default_zeroed!(PipeChannels::ver);

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct FwStatusFlags {
        pub(crate) halt_count: AtomicU64,
        __pad0: Pad<0x8>,
        pub(crate) halted: AtomicU32,
        __pad1: Pad<0xc>,
        pub(crate) resume: AtomicU32,
        __pad2: Pad<0xc>,
        pub(crate) unk_40: u32,
        __pad3: Pad<0xc>,
        pub(crate) unk_ctr: u32,
        __pad4: Pad<0xc>,
        pub(crate) unk_60: u32,
        __pad5: Pad<0xc>,
        pub(crate) unk_70: u32,
        __pad6: Pad<0xc>,
    }

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct FwStatus {
        pub(crate) fwctl_channel: ChannelRing<channels::FwCtlChannelState, channels::FwCtlMsg>,
        pub(crate) flags: FwStatusFlags,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataShared1 {
        pub(crate) table: Array<16, i32>,
        pub(crate) unk_44: Array<0x60, u8>,
        pub(crate) unk_a4: u32,
        pub(crate) unk_a8: u32,
    }
    default_zeroed!(HwDataShared1);

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct HwDataShared2Curve {
        pub(crate) unk_0: u32,
        pub(crate) unk_4: u32,
        pub(crate) t1: Array<16, u16>,
        pub(crate) t2: Array<16, i16>,
        pub(crate) t3: Array<8, Array<16, i32>>,
    }

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct HwDataShared2G14 {
        pub(crate) unk_0: Array<5, u32>,
        pub(crate) unk_14: u32,
        pub(crate) unk_18: Array<8, u32>,
        pub(crate) curve1: HwDataShared2Curve,
        pub(crate) curve2: HwDataShared2Curve,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataShared2 {
        pub(crate) table: Array<10, i32>,
        pub(crate) unk_28: Array<0x10, u8>,
        pub(crate) g14: HwDataShared2G14,
        pub(crate) unk_500: u32,
        pub(crate) unk_504: u32,
        pub(crate) unk_508: u32,
        pub(crate) unk_50c: u32,
    }
    default_zeroed!(HwDataShared2);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataShared3 {
        pub(crate) unk_0: u32,
        pub(crate) unk_4: u32,
        pub(crate) unk_8: u32,
        pub(crate) table: Array<16, u32>,
        pub(crate) unk_4c: u32,
    }
    default_zeroed!(HwDataShared3);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataA130Extra {
        pub(crate) unk_0: Array<0x38, u8>,
        pub(crate) unk_38: u32,
        pub(crate) unk_3c: u32,
        pub(crate) gpu_se_inactive_threshold: u32,
        pub(crate) unk_44: u32,
        pub(crate) gpu_se_engagement_criteria: i32,
        pub(crate) gpu_se_reset_criteria: u32,
        pub(crate) unk_50: u32,
        pub(crate) unk_54: u32,
        pub(crate) unk_58: u32,
        pub(crate) unk_5c: u32,
        pub(crate) gpu_se_filter_a_neg: F32,
        pub(crate) gpu_se_filter_1_a_neg: F32,
        pub(crate) gpu_se_filter_a: F32,
        pub(crate) gpu_se_filter_1_a: F32,
        pub(crate) gpu_se_ki_dt: F32,
        pub(crate) gpu_se_ki_1_dt: F32,
        pub(crate) unk_78: F32,
        pub(crate) unk_7c: F32,
        pub(crate) gpu_se_kp: F32,
        pub(crate) gpu_se_kp_1: F32,
        pub(crate) unk_88: u32,
        pub(crate) unk_8c: u32,
        pub(crate) max_pstate_scaled_1: u32,
        pub(crate) unk_94: u32,
        pub(crate) unk_98: u32,
        pub(crate) unk_9c: F32,
        pub(crate) unk_a0: u32,
        pub(crate) unk_a4: u32,
        pub(crate) gpu_se_filter_time_constant_ms: u32,
        pub(crate) gpu_se_filter_time_constant_1_ms: u32,
        pub(crate) gpu_se_filter_time_constant_clks: U64,
        pub(crate) gpu_se_filter_time_constant_1_clks: U64,
        pub(crate) unk_c0: u32,
        pub(crate) unk_c4: F32,
        pub(crate) unk_c8: Array<0x4c, u8>,
        pub(crate) unk_114: F32,
        pub(crate) unk_118: u32,
        pub(crate) unk_11c: u32,
        pub(crate) unk_120: u32,
        pub(crate) unk_124: u32,
        pub(crate) max_pstate_scaled_2: u32,
        pub(crate) unk_12c: Array<0x8c, u8>,
    }
    default_zeroed!(HwDataA130Extra);

    #[repr(C)]
    pub(crate) struct T81xxData {
        pub(crate) unk_d8c: u32,
        pub(crate) unk_d90: u32,
        pub(crate) unk_d94: u32,
        pub(crate) unk_d98: u32,
        pub(crate) unk_d9c: F32,
        pub(crate) unk_da0: u32,
        pub(crate) unk_da4: F32,
        pub(crate) unk_da8: u32,
        pub(crate) unk_dac: F32,
        pub(crate) unk_db0: u32,
        pub(crate) unk_db4: u32,
        pub(crate) unk_db8: F32,
        pub(crate) unk_dbc: F32,
        pub(crate) unk_dc0: u32,
        pub(crate) unk_dc4: u32,
        pub(crate) unk_dc8: u32,
        pub(crate) max_pstate_scaled: u32,
    }
    default_zeroed!(T81xxData);

    #[versions(AGX)]
    #[derive(Default, Copy, Clone)]
    #[repr(C)]
    pub(crate) struct PowerZone {
        pub(crate) val: F32,
        pub(crate) target: u32,
        pub(crate) target_off: u32,
        pub(crate) filter_tc_x4: u32,
        pub(crate) filter_tc_xperiod: u32,
        #[ver(V >= V13_0B4)]
        pub(crate) unk_10: u32,
        #[ver(V >= V13_0B4)]
        pub(crate) unk_14: u32,
        pub(crate) filter_a_neg: F32,
        pub(crate) filter_a: F32,
        pub(crate) pad: u32,
    }

    #[versions(AGX)]
    const MAX_CORES_PER_CLUSTER: usize = {
        #[ver(G >= G14X)]
        {
            16
        }
        #[ver(G < G14X)]
        {
            8
        }
    };

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct AuxLeakCoef {
        pub(crate) afr_1: Array<2, F32>,
        pub(crate) cs_1: Array<2, F32>,
        pub(crate) afr_2: Array<2, F32>,
        pub(crate) cs_2: Array<2, F32>,
    }

    #[versions(AGX)]
    #[repr(C)]
    pub(crate) struct HwDataA {
        pub(crate) unk_0: u32,
        pub(crate) clocks_per_period: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) clocks_per_period_2: u32,

        pub(crate) unk_8: u32,
        pub(crate) pwr_status: AtomicU32,
        pub(crate) unk_10: F32,
        pub(crate) unk_14: u32,
        pub(crate) unk_18: u32,
        pub(crate) unk_1c: u32,
        pub(crate) unk_20: u32,
        pub(crate) unk_24: u32,
        pub(crate) actual_pstate: u32,
        pub(crate) tgt_pstate: u32,
        pub(crate) unk_30: u32,
        pub(crate) cur_pstate: u32,
        pub(crate) unk_38: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_3c_0: u32,

        pub(crate) base_pstate_scaled: u32,
        pub(crate) unk_40: u32,
        pub(crate) max_pstate_scaled: u32,
        pub(crate) unk_48: u32,
        pub(crate) min_pstate_scaled: u32,
        pub(crate) freq_mhz: F32,
        pub(crate) unk_54: Array<0x20, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_74_0: u32,

        pub(crate) sram_k: Array<0x10, F32>,
        pub(crate) unk_b4: Array<0x100, u8>,
        pub(crate) unk_1b4: u32,
        pub(crate) temp_c: u32,
        pub(crate) avg_power_mw: u32,
        pub(crate) update_ts: U64,
        pub(crate) unk_1c8: u32,
        pub(crate) unk_1cc: Array<0x478, u8>,
        pub(crate) pad_644: Pad<0x8>,
        pub(crate) unk_64c: u32,
        pub(crate) unk_650: u32,
        pub(crate) pad_654: u32,
        pub(crate) pwr_filter_a_neg: F32,
        pub(crate) pad_65c: u32,
        pub(crate) pwr_filter_a: F32,
        pub(crate) pad_664: u32,
        pub(crate) pwr_integral_gain: F32,
        pub(crate) pad_66c: u32,
        pub(crate) pwr_integral_min_clamp: F32,
        pub(crate) max_power_1: F32,
        pub(crate) pwr_proportional_gain: F32,
        pub(crate) pad_67c: u32,
        pub(crate) pwr_pstate_related_k: F32,
        pub(crate) pwr_pstate_max_dc_offset: i32,
        pub(crate) unk_688: u32,
        pub(crate) max_pstate_scaled_2: u32,
        pub(crate) pad_690: u32,
        pub(crate) unk_694: u32,
        pub(crate) max_power_2: u32,
        pub(crate) pad_69c: Pad<0x18>,
        pub(crate) unk_6b4: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_6b8_0: Array<0x10, u8>,

        pub(crate) max_pstate_scaled_3: u32,
        pub(crate) unk_6bc: u32,
        pub(crate) pad_6c0: Pad<0x14>,
        pub(crate) ppm_filter_tc_periods_x4: u32,
        pub(crate) unk_6d8: u32,
        pub(crate) pad_6dc: u32,
        pub(crate) ppm_filter_a_neg: F32,
        pub(crate) pad_6e4: u32,
        pub(crate) ppm_filter_a: F32,
        pub(crate) pad_6ec: u32,
        pub(crate) ppm_ki_dt: F32,
        pub(crate) pad_6f4: u32,
        pub(crate) pwr_integral_min_clamp_2: u32,
        pub(crate) unk_6fc: F32,
        pub(crate) ppm_kp: F32,
        pub(crate) pad_704: u32,
        pub(crate) unk_708: u32,
        pub(crate) pwr_min_duty_cycle: u32,
        pub(crate) max_pstate_scaled_4: u32,
        pub(crate) unk_714: u32,
        pub(crate) pad_718: u32,
        pub(crate) unk_71c: F32,
        pub(crate) max_power_3: u32,
        pub(crate) cur_power_mw_2: u32,
        pub(crate) ppm_filter_tc_ms: u32,
        pub(crate) unk_72c: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) ppm_filter_tc_clks: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_730_4: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_730_8: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_730_c: u32,

        pub(crate) unk_730: F32,
        pub(crate) unk_734: u32,
        pub(crate) unk_738: u32,
        pub(crate) unk_73c: u32,
        pub(crate) unk_740: u32,
        pub(crate) unk_744: u32,
        pub(crate) unk_748: Array<0x4, F32>,
        pub(crate) unk_758: u32,
        pub(crate) perf_tgt_utilization: u32,
        pub(crate) pad_760: u32,
        pub(crate) perf_boost_min_util: u32,
        pub(crate) perf_boost_ce_step: u32,
        pub(crate) perf_reset_iters: u32,
        pub(crate) pad_770: u32,
        pub(crate) unk_774: u32,
        pub(crate) unk_778: u32,
        pub(crate) perf_filter_drop_threshold: u32,
        pub(crate) perf_filter_a_neg: F32,
        pub(crate) perf_filter_a2_neg: F32,
        pub(crate) perf_filter_a: F32,
        pub(crate) perf_filter_a2: F32,
        pub(crate) perf_ki: F32,
        pub(crate) perf_ki2: F32,
        pub(crate) perf_integral_min_clamp: F32,
        pub(crate) unk_79c: F32,
        pub(crate) perf_kp: F32,
        pub(crate) perf_kp2: F32,
        pub(crate) boost_state_unk_k: F32,
        pub(crate) base_pstate_scaled_2: u32,
        pub(crate) max_pstate_scaled_5: u32,
        pub(crate) base_pstate_scaled_3: u32,
        pub(crate) pad_7b8: u32,
        pub(crate) perf_cur_utilization: F32,
        pub(crate) perf_tgt_utilization_2: u32,
        pub(crate) pad_7c4: Pad<0x18>,
        pub(crate) unk_7dc: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_7e0_0: Array<0x10, u8>,

        pub(crate) base_pstate_scaled_4: u32,
        pub(crate) pad_7e4: u32,
        pub(crate) unk_7e8: Array<0x14, u8>,
        pub(crate) unk_7fc: F32,
        pub(crate) pwr_min_duty_cycle_2: F32,
        pub(crate) max_pstate_scaled_6: F32,
        pub(crate) max_freq_mhz: u32,
        pub(crate) pad_80c: u32,
        pub(crate) unk_810: u32,
        pub(crate) pad_814: u32,
        pub(crate) pwr_min_duty_cycle_3: u32,
        pub(crate) unk_81c: u32,
        pub(crate) pad_820: u32,
        pub(crate) min_pstate_scaled_4: F32,
        pub(crate) max_pstate_scaled_7: u32,
        pub(crate) unk_82c: u32,
        pub(crate) unk_alpha_neg: F32,
        pub(crate) unk_alpha: F32,
        pub(crate) unk_838: u32,
        pub(crate) unk_83c: u32,
        pub(crate) pad_840: Pad<0x2c>,
        pub(crate) unk_86c: u32,
        pub(crate) fast_die0_sensor_mask: U64,
        #[ver(G >= G14X)]
        pub(crate) fast_die1_sensor_mask: U64,
        pub(crate) fast_die0_release_temp_cc: u32,
        pub(crate) unk_87c: i32,
        pub(crate) unk_880: u32,
        pub(crate) unk_884: u32,
        pub(crate) pad_888: u32,
        pub(crate) unk_88c: u32,
        pub(crate) pad_890: u32,
        pub(crate) unk_894: F32,
        pub(crate) pad_898: u32,
        pub(crate) fast_die0_ki_dt: F32,
        pub(crate) pad_8a0: u32,
        pub(crate) unk_8a4: u32,
        pub(crate) unk_8a8: F32,
        pub(crate) fast_die0_kp: F32,
        pub(crate) pad_8b0: u32,
        pub(crate) unk_8b4: u32,
        pub(crate) pwr_min_duty_cycle_4: u32,
        pub(crate) max_pstate_scaled_8: u32,
        pub(crate) max_pstate_scaled_9: u32,
        pub(crate) fast_die0_prop_tgt_delta: u32,
        pub(crate) unk_8c8: u32,
        pub(crate) unk_8cc: u32,
        pub(crate) pad_8d0: Pad<0x14>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_8e4_0: Array<0x10, u8>,

        pub(crate) unk_8e4: u32,
        pub(crate) unk_8e8: u32,
        pub(crate) max_pstate_scaled_10: u32,
        pub(crate) unk_8f0: u32,
        pub(crate) unk_8f4: u32,
        pub(crate) pad_8f8: u32,
        pub(crate) pad_8fc: u32,
        pub(crate) unk_900: Array<0x24, u8>,

        pub(crate) unk_coef_a1: Array<8, Array<MAX_CORES_PER_CLUSTER::ver, F32>>,
        pub(crate) unk_coef_a2: Array<8, Array<MAX_CORES_PER_CLUSTER::ver, F32>>,

        pub(crate) pad_b24: Pad<0x70>,
        pub(crate) max_pstate_scaled_11: u32,
        pub(crate) freq_with_off: u32,
        pub(crate) unk_b9c: u32,
        pub(crate) unk_ba0: U64,
        pub(crate) unk_ba8: U64,
        pub(crate) unk_bb0: u32,
        pub(crate) unk_bb4: u32,

        #[ver(V >= V13_3)]
        pub(crate) pad_bb8_0: Pad<0x200>,
        #[ver(V >= V13_5)]
        pub(crate) pad_bb8_200: Pad<0x8>,

        pub(crate) pad_bb8: Pad<0x74>,
        pub(crate) unk_c2c: u32,
        pub(crate) power_zone_count: u32,
        pub(crate) max_power_4: u32,
        pub(crate) max_power_5: u32,
        pub(crate) max_power_6: u32,
        pub(crate) unk_c40: u32,
        pub(crate) unk_c44: F32,
        pub(crate) avg_power_target_filter_a_neg: F32,
        pub(crate) avg_power_target_filter_a: F32,
        pub(crate) avg_power_target_filter_tc_x4: u32,
        pub(crate) avg_power_target_filter_tc_xperiod: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) avg_power_target_filter_tc_clks: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_c58_4: u32,

        pub(crate) power_zones: Array<5, PowerZone::ver>,
        pub(crate) avg_power_filter_tc_periods_x4: u32,
        pub(crate) unk_cfc: u32,
        pub(crate) unk_d00: u32,
        pub(crate) avg_power_filter_a_neg: F32,
        pub(crate) unk_d08: u32,
        pub(crate) avg_power_filter_a: F32,
        pub(crate) unk_d10: u32,
        pub(crate) avg_power_ki_dt: F32,
        pub(crate) unk_d18: u32,
        pub(crate) unk_d1c: u32,
        pub(crate) unk_d20: F32,
        pub(crate) avg_power_kp: F32,
        pub(crate) unk_d28: u32,
        pub(crate) unk_d2c: u32,
        pub(crate) avg_power_min_duty_cycle: u32,
        pub(crate) max_pstate_scaled_12: u32,
        pub(crate) max_pstate_scaled_13: u32,
        pub(crate) unk_d3c: u32,
        pub(crate) max_power_7: F32,
        pub(crate) max_power_8: u32,
        pub(crate) unk_d48: u32,
        pub(crate) avg_power_filter_tc_ms: u32,
        pub(crate) unk_d50: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) avg_power_filter_tc_clks: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_d54_4: Array<0xc, u8>,

        pub(crate) unk_d54: Array<0x10, u8>,
        pub(crate) max_pstate_scaled_14: u32,
        pub(crate) unk_d68: Array<0x24, u8>,

        pub(crate) t81xx_data: T81xxData,

        pub(crate) unk_dd0: Array<0x40, u8>,

        #[ver(V >= V13_2)]
        pub(crate) unk_e10_pad: Array<0x10, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_e10_0: HwDataA130Extra,

        pub(crate) unk_e10: Array<0xc, u8>,

        pub(crate) fast_die0_sensor_mask_2: U64,
        #[ver(G >= G14X)]
        pub(crate) fast_die1_sensor_mask_2: U64,

        pub(crate) unk_e24: u32,
        pub(crate) unk_e28: u32,
        pub(crate) unk_e2c: Pad<0x1c>,
        pub(crate) unk_coef_b1: Array<8, Array<MAX_CORES_PER_CLUSTER::ver, F32>>,
        pub(crate) unk_coef_b2: Array<8, Array<MAX_CORES_PER_CLUSTER::ver, F32>>,

        #[ver(G >= G14X)]
        pub(crate) pad_1048_0: Pad<0x600>,

        pub(crate) pad_1048: Pad<0x5e4>,

        pub(crate) fast_die0_sensor_mask_alt: U64,
        #[ver(G >= G14X)]
        pub(crate) fast_die1_sensor_mask_alt: U64,
        #[ver(V < V13_0B4)]
        pub(crate) fast_die0_sensor_present: U64,

        pub(crate) unk_163c: u32,

        pub(crate) unk_1640: Array<0x2000, u8>,

        #[ver(G >= G14X)]
        pub(crate) unk_3640_0: Array<0x2000, u8>,

        pub(crate) unk_3640: u32,
        pub(crate) unk_3644: u32,
        pub(crate) hws1: HwDataShared1,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_hws2: Array<16, u16>,

        pub(crate) hws2: HwDataShared2,
        pub(crate) unk_3c00: u32,
        pub(crate) unk_3c04: u32,
        pub(crate) hws3: HwDataShared3,
        pub(crate) unk_3c58: Array<0x3c, u8>,
        pub(crate) unk_3c94: u32,
        pub(crate) unk_3c98: U64,
        pub(crate) unk_3ca0: U64,
        pub(crate) unk_3ca8: U64,
        pub(crate) unk_3cb0: U64,
        pub(crate) ts_last_idle: U64,
        pub(crate) ts_last_poweron: U64,
        pub(crate) ts_last_poweroff: U64,
        pub(crate) unk_3cd0: U64,
        pub(crate) unk_3cd8: U64,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_3ce0_0: u32,

        pub(crate) unk_3ce0: u32,
        pub(crate) unk_3ce4: u32,
        pub(crate) unk_3ce8: u32,
        pub(crate) unk_3cec: u32,
        pub(crate) unk_3cf0: u32,
        pub(crate) core_leak_coef: Array<8, F32>,
        pub(crate) sram_leak_coef: Array<8, F32>,

        #[ver(V >= V13_0B4)]
        pub(crate) aux_leak_coef: AuxLeakCoef,
        #[ver(V >= V13_0B4)]
        pub(crate) unk_3d34_0: Array<0x18, u8>,

        pub(crate) unk_3d34: Array<0x38, u8>,

        // 15.5 reaches HwDataA through the unaligned pointer at RegionB+0x451
        // and keeps its push-telemetry accounting inside it: 0x2ff8..0x5a50 is
        // zeroed wholesale, the periodic tick's own state sits at 0x5a50 and
        // its next deadline at 0x62a4. Whatever the exact tail of the 15.5
        // struct is, the object has to be at least that big or the firmware
        // writes past it. LOG.md 32.41.
        #[ver(V >= V15_5)]
        pub(crate) pad_15_5: Array<0x2800, u8>,
    }
    #[versions(AGX)]
    default_zeroed!(HwDataA::ver);
    #[versions(AGX)]
    no_debug!(HwDataA::ver);

    /// 15.5 widened this from 32 to 40 bytes. From the running firmware's
    /// io_mappings walker: `mov w12, #0x28` then
    /// `umull x15, w14, w12` / `ldr x16, [HWDataB + idx*40 + 0x640]`, and the
    /// fields it touches are entry+0/+8/+16/+24 (64-bit) and +32 (32-bit) --
    /// i.e. total_size and element_size widened to u64 and readwrite shrank to
    /// u32. See m3-gpu-display/LOG.md 12.2.
    #[versions(AGX)]
    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct IOMapping {
        pub(crate) phys_addr: U64,
        pub(crate) virt_addr: U64,

        #[ver(V < V15_5)]
        pub(crate) total_size: u32,
        #[ver(V < V15_5)]
        pub(crate) element_size: u32,
        #[ver(V < V15_5)]
        pub(crate) readwrite: U64,

        #[ver(V >= V15_5)]
        pub(crate) total_size: U64,
        #[ver(V >= V15_5)]
        pub(crate) element_size: U64,
        #[ver(V >= V15_5)]
        pub(crate) readwrite: u32,
        #[ver(V >= V15_5)]
        pub(crate) pad_24: Pad<0x4>,
    }

    #[versions(AGX)]
    const IO_MAPPING_COUNT: usize = {
        #[ver(V < V13_0B4)]
        {
            0x14
        }
        #[ver(V >= V13_0B4 && V < V13_3)]
        {
            0x17
        }
        #[ver(V >= V13_3 && V < V13_5)]
        {
            0x18
        }
        #[ver(V >= V13_5 && V < V15_5)]
        {
            0x19
        }
        // 15.5: the firmware's own index table at 0xfffffc000005cea0 names 48
        // register blocks and its highest io_mapping index is 54, so the array
        // is 55 entries. The arithmetic closes exactly against the measured
        // AGXHWDataB growth: 55*40 = 0x898, 0x640 + 0x898 = 0xed8, and
        // sgx_sram_ptr sits at 0xee8, leaving 0x10 -- total 0x588. LOG.md 14.1.
        #[ver(V >= V15_5)]
        {
            55
        }
    };

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataBAuxPStates {
        pub(crate) cs_max_pstate: u32,
        pub(crate) cs_frequencies: Array<0x10, u32>,
        pub(crate) cs_voltages: Array<0x10, Array<0x2, u32>>,
        pub(crate) cs_voltages_sram: Array<0x10, Array<0x2, u32>>,
        pub(crate) cs_unkpad: u32,
        pub(crate) afr_max_pstate: u32,
        pub(crate) afr_frequencies: Array<0x8, u32>,
        pub(crate) afr_voltages: Array<0x8, Array<0x2, u32>>,
        pub(crate) afr_voltages_sram: Array<0x8, Array<0x2, u32>>,
        pub(crate) afr_unkpad: u32,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct HwDataB {
        #[ver(V < V13_0B4)]
        pub(crate) unk_0: U64,

        pub(crate) unk_8: U64,

        #[ver(V < V13_0B4)]
        pub(crate) unk_10: U64,

        pub(crate) unk_18: U64,
        pub(crate) unk_20: U64,
        pub(crate) unk_28: U64,
        pub(crate) unk_30: U64,
        pub(crate) timestamp_area_base: U64,
        pub(crate) pad_40: Pad<0x20>,

        #[ver(V < V13_0B4)]
        pub(crate) yuv_matrices: Array<0xf, Array<3, Array<4, i16>>>,

        #[ver(V >= V13_0B4)]
        pub(crate) yuv_matrices: Array<0x3f, Array<3, Array<4, i16>>>,

        pub(crate) pad_1c8: Pad<0x8>,
        pub(crate) io_mappings: Array<IO_MAPPING_COUNT::ver, IOMapping::ver>,

        // 15.5 layout of 0xed8..0x1000, from macOS's own stores
        // (AGX{,Arm}Firmware::initFirmwareData, kext 0xfffffe00090dc598 /
        // 0xfffffe00091010c8) and the firmware's consumers (direct read of
        // +0xed8 at fw:0x2f9cc, bulk copy of 0xef0..0xfd8 at fw:0x2b638..):
        // there is NO 0x10 pad after io_mappings -- sgx_sram_ptr is at 0xed8,
        // base_clock_khz at 0xf20 -- but one extra u32 after
        // power_sample_period (0xf28) and 0xc bytes before uat_ttb_base, so
        // 0x1000 onwards is unchanged. With the old pad every field the
        // firmware copied from here was 0x10/0xc bytes off (it read
        // base_clock_khz = 0 and unk_4f4 = 0). LOG.md 32.41.539.

        #[ver(V >= V13_0B4)]
        pub(crate) sgx_sram_ptr: U64,

        pub(crate) chip_id: u32,
        pub(crate) unk_454: u32,
        pub(crate) unk_458: u32,
        pub(crate) unk_45c: u32,
        pub(crate) unk_460: u32,
        pub(crate) unk_464: u32,
        pub(crate) unk_468: u32,
        pub(crate) unk_46c: u32,
        pub(crate) unk_470: u32,
        pub(crate) unk_474: u32,
        pub(crate) unk_478: u32,
        pub(crate) unk_47c: u32,
        pub(crate) unk_480: u32,
        pub(crate) unk_484: u32,
        pub(crate) unk_488: u32,
        pub(crate) unk_48c: u32,
        pub(crate) base_clock_khz: u32,
        pub(crate) power_sample_period: u32,
        #[ver(V >= V15_5)]
        pub(crate) unk_498_15_5: u32,
        pub(crate) pad_498: Pad<0x4>,
        pub(crate) unk_49c: u32,
        pub(crate) unk_4a0: u32,
        pub(crate) unk_4a4: u32,
        pub(crate) pad_4a8: Pad<0x4>,
        pub(crate) unk_4ac: u32,
        pub(crate) pad_4b0: Pad<0x8>,
        pub(crate) unk_4b8: u32,
        pub(crate) unk_4bc: Array<0x4, u8>,
        pub(crate) unk_4c0: u32,
        pub(crate) unk_4c4: u32,
        pub(crate) unk_4c8: u32,
        pub(crate) unk_4cc: u32,
        pub(crate) unk_4d0: u32,
        pub(crate) unk_4d4: u32,
        pub(crate) unk_4d8: Array<0x4, u8>,
        pub(crate) unk_4dc: u32,
        pub(crate) unk_4e0: U64,
        pub(crate) unk_4e8: u32,
        pub(crate) unk_4ec: u32,
        pub(crate) unk_4f0: u32,
        pub(crate) unk_4f4: u32,
        pub(crate) unk_4f8: u32,
        pub(crate) unk_4fc: u32,
        pub(crate) unk_500: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_504_0: u32,

        pub(crate) unk_504: u32,
        pub(crate) unk_508: u32,
        pub(crate) unk_50c: u32,
        pub(crate) unk_510: u32,
        pub(crate) unk_514: u32,
        pub(crate) unk_518: u32,
        pub(crate) unk_51c: u32,
        pub(crate) unk_520: u32,
        pub(crate) unk_524: u32,
        pub(crate) unk_528: u32,
        pub(crate) unk_52c: u32,
        pub(crate) unk_530: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_534_0: u32,

        pub(crate) unk_534: u32,
        pub(crate) unk_538: u32,

        pub(crate) num_frags: u32,
        pub(crate) unk_540: u32,
        pub(crate) unk_544: u32,
        pub(crate) unk_548: u32,
        pub(crate) unk_54c: u32,
        pub(crate) unk_550: u32,
        pub(crate) unk_554: u32,
        #[ver(V >= V15_5)]
        pub(crate) unk_ff4_15_5: Array<3, u32>,
        pub(crate) uat_ttb_base: U64,
        pub(crate) gpu_core_id: u32,
        pub(crate) gpu_rev_id: u32,
        pub(crate) num_cores: u32,
        pub(crate) max_pstate: u32,

        #[ver(V < V13_0B4)]
        pub(crate) num_pstates: u32,

        pub(crate) frequencies: Array<0x10, u32>,
        pub(crate) voltages: Array<0x10, [u32; 0x8]>,
        pub(crate) voltages_sram: Array<0x10, [u32; 0x8]>,

        #[ver(V >= V13_3)]
        pub(crate) unk_9f4_0: Pad<64>,

        pub(crate) sram_k: Array<0x10, F32>,
        pub(crate) unk_9f4: Array<0x10, u32>,
        pub(crate) rel_max_powers: Array<0x10, u32>,
        pub(crate) rel_boost_freqs: Array<0x10, u32>,

        #[ver(V >= V13_3)]
        pub(crate) unk_arr_0: Array<32, u32>,

        #[ver(V < V13_0B4)]
        pub(crate) min_sram_volt: u32,

        #[ver(V < V13_0B4)]
        pub(crate) unk_ab8: u32,

        #[ver(V < V13_0B4)]
        pub(crate) unk_abc: u32,

        #[ver(V < V13_0B4)]
        pub(crate) unk_ac0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) aux_ps: HwDataBAuxPStates,

        #[ver(V >= V13_3)]
        pub(crate) pad_ac4_0: Array<0x44c, u8>,

        pub(crate) pad_ac4: Pad<0x8>,
        pub(crate) unk_acc: u32,
        pub(crate) unk_ad0: u32,
        pub(crate) pad_ad4: Pad<0x10>,
        pub(crate) unk_ae4: Array<0x4, u32>,
        pub(crate) pad_af4: Pad<0x4>,
        pub(crate) unk_af8: u32,
        #[ver(V < V15_5)]
        pub(crate) pad_afc: Pad<0x8>,
        #[ver(V < V15_5)]
        pub(crate) unk_b04: u32,
        #[ver(V < V15_5)]
        pub(crate) unk_b08: u32,
        // 15.5: the shader-core enable mask, 3 x u32 at HwDataB+0x1c90. Both
        // power-up paths (fw:0x31244, fw:0x37f44) read it as
        //     ldr x8, [RegionB] ; ldr x9, [x8, #0x1c90] ; ldr x8, [x8, #0x1c98]
        // and, only if non-zero, write it to Bumper+0xa0/a4/a8 and to
        // DFenderMGPUCfg+0x1500/04/08 (= CORE_MASKS_G14X, SGX+0xe01500), which
        // the USC pipe mask and the firmware cluster table are read from.
        // Zero = the mask is never restored after a power gate. LOG.md 32.41.538.
        #[ver(V >= V15_5)]
        pub(crate) pad_afc: Pad<0x4>,
        #[ver(V >= V15_5)]
        pub(crate) core_masks_15_5: Array<3, u32>,
        pub(crate) unk_b0c: u32,

        #[ver(G >= G14X)]
        pub(crate) pad_b10_0: Array<0x8, u8>,

        pub(crate) unk_b10: u32,
        pub(crate) timer_offset: U64,
        pub(crate) unk_b1c: u32,
        pub(crate) unk_b20: u32,
        pub(crate) unk_b24: u32,
        pub(crate) unk_b28: u32,
        pub(crate) unk_b2c: u32,
        pub(crate) unk_b30: u32,
        pub(crate) unk_b34: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_b38_0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_b38_4: u32,

        #[ver(V >= V13_3)]
        pub(crate) unk_b38_8: u32,

        pub(crate) unk_b38: Array<0xc, u32>,
        pub(crate) unk_b68: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_b6c: Array<0xd0, u8>,

        #[ver(G >= G14X)]
        pub(crate) unk_c3c_0: Array<0x8, u8>,

        #[ver(G < G14X && V >= V13_5)]
        pub(crate) unk_c3c_8: Array<0x10, u8>,

        #[ver(V >= V13_5)]
        pub(crate) unk_c3c_18: Array<0x20, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_c3c: u32,
    }
    #[versions(AGX)]
    default_zeroed!(HwDataB::ver);

    #[derive(Debug)]
    #[repr(C, packed)]
    pub(crate) struct GpuStatsVtx {
        // This changes all the time and we don't use it, let's just make it a big buffer
        pub(crate) opaque: Array<0x3000, u8>,
    }
    default_zeroed!(GpuStatsVtx);

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct GpuStatsFrag {
        // This changes all the time and we don't use it, let's just make it a big buffer
        // except for these two fields which may need init.
        #[ver(G >= G14X)]
        pub(crate) unk1_0: Array<0x910, u8>,
        pub(crate) unk1: Array<0x100, u8>,
        pub(crate) cur_stamp_id: i32,
        pub(crate) unk2: Array<0x14, u8>,
        pub(crate) unk_id: i32,
        pub(crate) unk3: Array<0x1000, u8>,
    }

    #[versions(AGX)]
    impl Default for GpuStatsFrag::ver {
        fn default() -> Self {
            Self {
                #[ver(G >= G14X)]
                unk1_0: Default::default(),
                unk1: Default::default(),
                cur_stamp_id: -1,
                unk2: Default::default(),
                unk_id: -1,
                unk3: Default::default(),
            }
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct GpuGlobalStatsVtx {
        pub(crate) total_cmds: u32,
        pub(crate) stats: GpuStatsVtx,
    }
    default_zeroed!(GpuGlobalStatsVtx);

    #[versions(AGX)]
    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct GpuGlobalStatsFrag {
        pub(crate) total_cmds: u32,
        pub(crate) unk_4: u32,
        pub(crate) stats: GpuStatsFrag::ver,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct GpuStatsComp {
        // This changes all the time and we don't use it, let's just make it a big buffer
        pub(crate) opaque: Array<0x3000, u8>,
    }
    default_zeroed!(GpuStatsComp);

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RuntimeScratch {
        #[ver(V < V15_5)]
        pub(crate) unk_280: Array<0x6800, u8>,

        // 15.5 keeps an unaligned pointer at RegionB+0x451, inside what 13.5
        // treats as one opaque blob. Three separate sites do
        //     ldr x8, [RegionB_global] ; add x8, x8, #0x451 ; ldr x8, [x8]
        // and then use the result as a struct base. Left at zero the firmware
        // NULL-derefs with FAR = 0x8e8, which is the `str w9, [x8, #2280]` at
        // 0x31d4 (LOG.md 14.2 and 30.5).
        //
        // Carved out of the blob so the surrounding size is unchanged.
        // gpu_scratch sits at RegionB+0x398, so 0x451 is +0xb9 into it; U64 is
        // packed(1) so the unaligned position needs no packing of this struct.
        #[ver(V >= V15_5)]
        pub(crate) unk_280a: Array<0xb9, u8>,
        #[ver(V >= V15_5)]
        pub(crate) unkptr_451: U64,
        #[ver(V >= V15_5)]
        pub(crate) unk_280b: Array<{ 0x6800 - 0xb9 - 8 }, u8>,
        pub(crate) unk_6a80: u32,
        pub(crate) gpu_idle: u32,
        pub(crate) unkpad_6a88: Pad<0x14>,
        pub(crate) unk_6a9c: u32,
        pub(crate) unk_ctr0: u32,
        pub(crate) unk_ctr1: u32,
        pub(crate) unk_6aa8: u32,
        pub(crate) unk_6aac: u32,
        pub(crate) unk_ctr2: u32,
        pub(crate) unk_6ab4: u32,
        pub(crate) unk_6ab8: u32,
        pub(crate) unk_6abc: u32,
        pub(crate) unk_6ac0: u32,
        pub(crate) unk_6ac4: u32,
        pub(crate) unk_ctr3: u32,
        pub(crate) unk_6acc: u32,
        pub(crate) unk_6ad0: u32,
        pub(crate) unk_6ad4: u32,
        pub(crate) unk_6ad8: u32,
        pub(crate) unk_6adc: u32,
        pub(crate) unk_6ae0: u32,
        pub(crate) unk_6ae4: u32,
        pub(crate) unk_6ae8: u32,
        pub(crate) unk_6aec: u32,
        pub(crate) unk_6af0: u32,
        pub(crate) unk_ctr4: u32,
        pub(crate) unk_ctr5: u32,
        pub(crate) unk_6afc: u32,
        pub(crate) pad_6b00: Pad<0x38>,

        #[ver(G >= G14X)]
        pub(crate) pad_6b00_extra: Array<0x4800, u8>,

        pub(crate) unk_6b38: u32,
        pub(crate) pad_6b3c: Pad<0x84>,
    }
    #[versions(AGX)]
    default_zeroed!(RuntimeScratch::ver);

    #[versions(AGX)]
    #[ver(V >= V15_5)]
    const _LAYOUT_HWDATAB_CORE_MASKS: () = {
        assert!(core::mem::offset_of!(HwDataB::ver, core_masks_15_5) == 0x1c90);
        assert!(core::mem::offset_of!(HwDataB::ver, io_mappings) == 0x640);
        assert!(core::mem::offset_of!(HwDataB::ver, sgx_sram_ptr) == 0xed8);
        assert!(core::mem::offset_of!(HwDataB::ver, chip_id) == 0xee0);
        assert!(core::mem::offset_of!(HwDataB::ver, base_clock_khz) == 0xf20);
        assert!(core::mem::offset_of!(HwDataB::ver, unk_49c) == 0xf30);
        assert!(core::mem::offset_of!(HwDataB::ver, unk_4c0) == 0xf54);
        assert!(core::mem::offset_of!(HwDataB::ver, unk_4e0) == 0xf74);
        assert!(core::mem::offset_of!(HwDataB::ver, unk_4f4) == 0xf88);
        assert!(core::mem::offset_of!(HwDataB::ver, uat_ttb_base) == 0x1000);
        assert!(core::mem::size_of::<HwDataB::ver>() == 0x1e0c);
    };

    /// Layout check for the 15.5 RegionB port.
    ///
    /// The firmware reads `fault_info` at 0x2b8 (LOG.md 8.3). If the 0x18
    /// header or the 32-byte ChannelSlot is wrong, everything below moves and
    /// the firmware faults at run time instead of here.
    ///
    /// ⚠️ The two companion claims that used to be pinned here -- "hwdata_b at
    /// 0x2c8, fwlog_ring2 at 0x2d8" -- were inferred from 13.5 spacing, never
    /// measured. 0x2c8 and 0x2d8 are the *FW addresses of the two descriptor
    /// tables* on 15.5; see the block comment on `buffer_mgr_ctl_gpu_addr_15_5`
    /// and `_LAYOUT_REGIONB_TAIL`. LOG.md 32.41.73.
    #[versions(AGX)]
    #[ver(V >= V15_5)]
    const _LAYOUT_RUNTIME_POINTERS: () = {
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, event_15_5) == 0x1b8);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, fw_log_15_5) == 0x1c8);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, ktrace_15_5) == 0x1d8);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, stats_15_5) == 0x1e8);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, fwlog_payload) == 0x1f8);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, fwlog_index) == 0x200);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, fwlog_overflow) == 0x220);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, events_enabled) == 0x240);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, pipe_tables) == 0x244);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, unkptr_190) == 0x2b8);
        // The unaligned pointer the firmware reads as RegionB+0x451: 0x398 for
        // gpu_scratch plus 0xb9 into the scratch blob. Getting this wrong is a
        // run-time NULL deref in the firmware, so pin it here instead.
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, gpu_scratch) == 0x398);
        assert!(core::mem::offset_of!(RuntimeScratch::ver, unkptr_451) == 0xb9);
    };

    #[versions(AGX)]
    #[repr(C)]
    pub(crate) struct RuntimePointers<'a> {
        // 15.5 prepended a 0x18-byte header. Slot 0 is a second copy of the
        // hwdata_b pointer -- proven, because the firmware dereferences
        // RegionB+0 and reads the pstate tables through it.
        //
        // Slots 1 and 2 are pointers too, and both need real buffers. The
        // earlier note here ("unidentified; no consumer touches them") was
        // wrong -- it was written before the driver could get this far. Slot 2
        // is dereferenced at 0x31790:
        //     ldr x8, [RegionB_global] ; ldr x8, [x8, #16] ; ldr w9, [x8]
        // which NULL-faults with FAR = 0 when left zero (LOG.md 30.7). m1n1
        // gives each a 0x4000 buffer.
        // Together with the 32-byte ChannelRing this puts the channels at +0x18
        // and shifts everything below by +0x128, which is confirmed by three
        // pointers the firmware reads at their shifted positions (fault_info
        // 0x2b8, hwdata_b 0x2c8, fwlog_ring2 0x2d8). LOG.md 8.3.
        #[ver(V >= V15_5)]
        pub(crate) hwdata_b_0: GpuPointer<'a, super::HwDataB::ver>,
        #[ver(V >= V15_5)]
        pub(crate) unk_8_15_5: U64,
        #[ver(V >= V15_5)]
        pub(crate) unk_10_15_5: U64,

        pub(crate) pipes: Array<4, PipeChannels::ver>,

        pub(crate) device_control:
            ChannelSlot::ver<channels::ChannelState, channels::DeviceControlMsg::ver>,

        // ⚠️ 15.5 has **no event and no stats slot here**. Only three channels
        // live in RegionB -- device control, fwlog and ktrace -- and the 0x50
        // bytes that 13.5 spends on `__pad0` hold the fwlog payload pointer and
        // its two per-sub-channel counter arrays. Read straight off the
        // producer at 0x30a40:
        //
        //     ldr x9,  [RegionB, #456]   ; 0x1c8 = fwlog slot (0x1b8) pair2 state
        //     ldr x10, [RegionB, #464]   ; 0x1d0 = fwlog slot (0x1b8) pair2 ring
        //     ldr x8,  [RegionB, #504]   ; 0x1f8 = payload buffer
        //     ldr w12, [RegionB + sub*4, #512]   ; 0x200 = payload index
        //     ldr w9,  [RegionB + sub*4, #544]   ; 0x220 = overflow counter
        //     ldr w9,  [RegionB, #576]   ; 0x240 = events_enabled
        //
        // and confirmed by what the driver received while it still believed the
        // 13.5 order: the slot at 0x1b8 delivered `{msg_type: 2, msg_index}`
        // (fwlog), and the one at 0x1d8 delivered `{msg_type: 26, timestamp}`
        // (ktrace). The old layout also had the firmware writing 216-byte log
        // records through what the driver had put at 0x1f8 -- the *ktrace
        // channel state pointer* -- and incrementing the ktrace ring pointer at
        // 0x200 once per log line. LOG.md 32.34.
        #[ver(V < V15_5)]
        pub(crate) event: ChannelSlot::ver<channels::EventChannelState, channels::RawEventMsg>,
        #[ver(V < V15_5)]
        pub(crate) fw_log: ChannelSlot::ver<channels::FwLogChannelState, channels::RawFwLogMsg>,
        #[ver(V < V15_5)]
        pub(crate) ktrace: ChannelSlot::ver<channels::ChannelState, channels::RawKTraceMsg::ver>,

        // 15.5's receive slots are **sixteen** bytes -- a bare
        // `{state, ring}` -- so all four of 13.5's rx channels still fit
        // between device control (0x198) and the fwlog payload pointer
        // (0x1f8), four slots instead of two:
        //
        //     0x1b8 event   0x1c8 fwlog   0x1d8 ktrace   0x1e8 stats
        //
        // The fwlog producer at 0x30a40 reads its state from 0x1c8 and its
        // ring from 0x1d0, which is what pins fwlog to the *second* slot and
        // therefore leaves the first one for events. With 32-byte
        // `ChannelSlot`s here, `new_rx` aliased each channel into the next
        // slot: fwlog's duplicate landed on 0x1c8 (which is why the firmware
        // log worked and looked like proof that fwlog lived at 0x1b8), ktrace's
        // landed on the stats slot -- and the event channel did not exist at
        // all, so no job completion was ever signalled and every submission
        // hung. LOG.md 32.39.
        #[ver(V >= V15_5)]
        pub(crate) event_15_5: ChannelRing<channels::EventChannelState, channels::RawEventMsg>,
        #[ver(V >= V15_5)]
        pub(crate) fw_log_15_5: ChannelRing<channels::FwLogChannelState, channels::RawFwLogMsg>,

        // ⚠️ The correction to the correction: 15.5 *does* have a statistics
        // channel here, and it is what the 32-byte ktrace slot was covering up.
        // Below device control and fwlog the slots are **sixteen** bytes, not
        // thirty-two -- ktrace at 0x1d8 and stats at 0x1e8, each just
        // `{state, ring}`:
        //
        //     0x10484  ldr x20, [RegionB, #488]   ; 0x1e8, the stats state
        //     0x1049c  ldr w24, [x20, #32]        ; write_ptr
        //     0x104a8  ldr w8,  [x20]             ; read_ptr, ring of 256
        //     0x1057c  ldr x8,  [RegionB, #496]   ; 0x1f0, the stats ring
        //
        // and when it fills, the firmware rings the host (`ep 0x20`, message
        // `0x42 << 48`), prints "Waiting for space to add statistics!" and
        // *spins*. With ktrace built as a `ChannelSlot`, `new_rx` duplicated
        // its state and ring into 0x1e8/0x1f0, so every statistic went into the
        // ktrace ring, nothing drained it, and the firmware sat in that spin
        // instead of servicing device control -- which is exactly the
        // "GpuContext: Failed to invalidate GPU context!" the driver reported.
        // LOG.md 32.39.
        #[ver(V >= V15_5)]
        pub(crate) ktrace_15_5: ChannelRing<channels::ChannelState, channels::RawKTraceMsg::ver>,
        #[ver(V >= V15_5)]
        pub(crate) stats_15_5: ChannelRing<channels::ChannelState, channels::RawStatsMsg::ver>,

        #[ver(V < V15_5)]
        pub(crate) stats: ChannelSlot::ver<channels::ChannelState, channels::RawStatsMsg::ver>,

        #[ver(V < V15_5)]
        pub(crate) __pad0: Pad<0x50>,

        #[ver(V >= V15_5)]
        pub(crate) fwlog_payload: Option<GpuWeakPointer<[channels::RawFwLogPayloadMsg]>>,
        #[ver(V >= V15_5)]
        pub(crate) fwlog_index: Array<8, u32>,
        #[ver(V >= V15_5)]
        pub(crate) fwlog_overflow: Array<8, u32>,

        // 15.5 put real fields in what 13.5 treats as padding.
        //
        // ⚠️ `events_enabled` is why the firmware has never once signalled this
        // driver. The event producer opens with
        //     0x30a44  ldr w9, [x8, #576]      ; RegionB + 0x240
        //     0x30a48  cbz w9, <return>
        // and the *only* MSG_RX_DOORBELL the firmware ever sends is at the far
        // end of that function (0x30b8c, `ep 0x20, msg 0x42 << 48`). Left zero,
        // the coprocessor is mute by construction. LOG.md 32.15.
        //
        // `pipe_tables` are four **unaligned** per-pipe pointers; 0x36718 does
        // `str wzr, [*(RegionB+0x25c)]`, an unconditional store through the
        // last of them.
        #[ver(V >= V15_5)]
        pub(crate) events_enabled: u32,
        #[ver(V >= V15_5)]
        pub(crate) pipe_tables: Array<4, U64>,
        #[ver(V >= V15_5)]
        pub(crate) __pad0b: Pad<0x24>,

        pub(crate) unk_160: U64,
        pub(crate) unk_168: U64,
        pub(crate) stats_vtx: GpuPointer<'a, super::GpuGlobalStatsVtx>,
        pub(crate) stats_frag: GpuPointer<'a, super::GpuGlobalStatsFrag::ver>,
        pub(crate) stats_comp: GpuPointer<'a, super::GpuStatsComp>,
        pub(crate) hwdata_a: GpuPointer<'a, super::HwDataA::ver>,
        pub(crate) unkptr_190: GpuPointer<'a, &'a [u8]>,

        // 🔑 15.5 replaces 13.5's five pointers here (`unkptr_198`, two more
        // copies of `hwdata_b`, a second `fwlog_buf` and `unkptr_1b8` -- forty
        // bytes, none of which any consumer of the RegionB global reads) with
        // the two descriptor-table address pairs and two config words.
        //
        // `AGXArmFirmware::initFirmwareData` (kc:0x9101264..0x9101358) fills
        // exactly this range, through four accelerator accessors resolved by
        // their PAC discriminators in the `AGXAcceleratorG15` vtable:
        //
        //     [0x2c0] = getPBDescriptorTableGPUAddress()
        //     [0x2c8] = getPBDescriptorTableFWAddress()
        //     [0x2d0] = getUMAPagePoolDescriptorTableGPUAddress()
        //     [0x2d8] = getUMAPagePoolDescriptorTableFWAddress()
        //     [0x2e0] = u32 from accel+0xf528
        //     [0x2e4] = u32 from accel+0xf59c
        //
        // and the firmware reads the last two straight off its RegionB global:
        // `adrp x21, 0x1a3000 ; ldr x8, [x21, #64] ; ldr w0, [x8, #740]`
        // (fw:0x8004) and the same for +736 at fw:0x8e44 and fw:0xb628. That is
        // what anchors this block to RegionB rather than to any assumed
        // spacing. LOG.md 32.41.73.
        #[ver(V >= V15_5)]
        pub(crate) buffer_mgr_ctl_gpu_addr_15_5: U64,
        #[ver(V >= V15_5)]
        pub(crate) buffer_mgr_ctl_fw_addr_15_5: U64,
        #[ver(V >= V15_5)]
        pub(crate) uma_pool_ctl_gpu_addr: U64,
        #[ver(V >= V15_5)]
        pub(crate) uma_pool_ctl_fw_addr: U64,

        // 🔑 `MAX_TGMEM`, and the firmware reads a 0 here as "override it to 0".
        // fw:0xb628 skips the whole config-RAM update only when this field
        // already reads 4:
        //
        //     ldr  x9, [x21, #64]        ; the RegionB global
        //     ldr  w9, [x9, #736]        ; this field
        //     cmp  w9, #0x4
        //     b.eq skip
        //     mov  w10, #0xa728          ; config-RAM selector
        //     ldr  x8, [x8, x10]
        //     and  x8, x8, #~0x3f00      ; clear bits 13:8
        //     orr  x19, x8, x9, lsl #8   ; insert this field
        //     str  x19, [x8, x9]         ; write it back
        //
        // So leaving it zero does not leave the hardware default alone -- the
        // firmware actively programs
        // `AGX_CR_USC_TP_GLOBAL_RESOURCE_CFG0_MAX_TGMEM` to **zero**, and its
        // own verification pass then warns that the register is 0 where it
        // expected 4. That warning has been in every capture since bring-up and
        // was filed as unrelated noise. `halGetDefaultUscMaxTgmem()` returns 4
        // and `halGetMaxUscMaxTgmem()` returns 0x3f -- exactly the six-bit
        // field at bits 13:8. A USC with no threadgroup memory has to source
        // all of it from UMA, which is what `gUPM` is. LOG.md 32.41.73.
        #[ver(V >= V15_5)]
        pub(crate) usc_max_tgmem: u32,
        #[ver(V >= V15_5)]
        pub(crate) unk_2e4: u32,

        #[ver(V < V15_5)]
        pub(crate) unkptr_198: GpuPointer<'a, &'a [u8]>,
        #[ver(V < V15_5)]
        pub(crate) hwdata_b: GpuPointer<'a, super::HwDataB::ver>,
        #[ver(V < V15_5)]
        pub(crate) hwdata_b_2: GpuPointer<'a, super::HwDataB::ver>,
        #[ver(V < V15_5)]
        pub(crate) fwlog_buf: Option<GpuWeakPointer<[channels::RawFwLogPayloadMsg]>>,
        #[ver(V < V15_5)]
        pub(crate) unkptr_1b8: GpuPointer<'a, &'a [u8]>,

        #[ver(G < G14X)]
        pub(crate) unkptr_1c0: GpuPointer<'a, &'a [u8]>,
        #[ver(G < G14X)]
        pub(crate) unkptr_1c8: GpuPointer<'a, &'a [u8]>,

        pub(crate) unk_1d0: u32,
        pub(crate) unk_1d4: u32,
        pub(crate) unk_1d8: Array<0x3c, u8>,
        // ⚠️ 13.5 only. On 15.5 the pair moved to 0x2c0/0x2c8 (see the block
        // above `unk_1d0`), and RegionB+0x328/0x32c/0x330 are three u32s the
        // firmware reads through its own global -- so writing an IOVA across
        // them here is actively wrong. LOG.md 32.41.73.
        #[ver(V < V15_5)]
        pub(crate) buffer_mgr_ctl_gpu_addr: U64,
        #[ver(V < V15_5)]
        pub(crate) buffer_mgr_ctl_fw_addr: U64,
        #[ver(V >= V15_5)]
        pub(crate) unk_32c: Array<0x10, u8>,

        pub(crate) __pad1: Pad<0x5c>,
        pub(crate) gpu_scratch: RuntimeScratch::ver,
    }
    #[versions(AGX)]
    no_debug!(RuntimePointers::ver<'_>);

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct PendingStamp {
        pub(crate) info: AtomicU32,
        pub(crate) wait_value: AtomicU32,
    }
    default_zeroed!(PendingStamp);

    #[derive(Debug, Clone, Copy)]
    #[repr(C, packed)]
    pub(crate) struct FaultInfo {
        pub(crate) unk_0: u32,
        pub(crate) unk_4: u32,
        pub(crate) queue_uuid: u32,
        pub(crate) unk_c: u32,
        pub(crate) unk_10: u32,
        pub(crate) unk_14: u32,
    }
    default_zeroed!(FaultInfo);

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct PowerZoneGlobal {
        pub(crate) target: u32,
        pub(crate) target_off: u32,
        pub(crate) filter_tc: u32,
    }
    default_zeroed!(PowerZoneGlobal);

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Globals {
        pub(crate) ktrace_enable: u32,
        pub(crate) unk_4: Array<0x20, u8>,

        #[ver(V >= V13_2)]
        pub(crate) unk_24_0: u32,

        pub(crate) unk_24: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) debug: u32,

        #[ver(V >= V13_3)]
        pub(crate) unk_28_4: u32,

        pub(crate) unk_28: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_2c_0: u32,

        pub(crate) unk_2c: u32,
        pub(crate) unk_30: u32,
        pub(crate) unk_34: u32,
        pub(crate) unk_38: Array<0x1c, u8>,

        // pub(crate) sub: GlobalsSub::ver,
        pub(crate) unk_54: u16,
        pub(crate) unk_56: u16,
        pub(crate) unk_58: u16,
        pub(crate) unk_5a: U32,
        pub(crate) unk_5e: U32,
        pub(crate) unk_62: U32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_66_0: Array<0xc, u8>,

        pub(crate) unk_66: U32,
        pub(crate) unk_6a: Array<0x16, u8>,
        // end GlobalsSub::ver

        pub(crate) unk_80: Array<0xf80, u8>,
        pub(crate) unk_1000: Array<0x7000, u8>,
        pub(crate) unk_8000: Array<0x900, u8>,

        #[ver(G >= G14X)]
        pub(crate) unk_8900_pad: Array<0x484c, u8>,

        #[ver(V >= V13_3)]
        pub(crate) unk_8900_pad2: Array<0x54, u8>,

        pub(crate) unk_8900: u32,
        pub(crate) pending_submissions: AtomicU32,
        pub(crate) max_power: u32,
        pub(crate) max_pstate_scaled: u32,
        pub(crate) max_pstate_scaled_2: u32,
        pub(crate) unk_8914: u32,
        pub(crate) unk_8918: u32,
        pub(crate) max_pstate_scaled_3: u32,
        pub(crate) unk_8920: u32,
        pub(crate) power_zone_count: u32,
        pub(crate) avg_power_filter_tc_periods: u32,
        pub(crate) avg_power_ki_dt: F32,
        pub(crate) avg_power_kp: F32,
        pub(crate) avg_power_min_duty_cycle: u32,
        pub(crate) avg_power_target_filter_tc: u32,
        pub(crate) power_zones: Array<5, PowerZoneGlobal>,
        pub(crate) unk_8978: Array<0x44, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_89bc_0: Array<0x3c, u8>,

        pub(crate) unk_89bc: u32,
        pub(crate) fast_die0_release_temp: u32,
        pub(crate) unk_89c4: i32,
        pub(crate) fast_die0_prop_tgt_delta: u32,
        pub(crate) fast_die0_kp: F32,
        pub(crate) fast_die0_ki_dt: F32,
        pub(crate) unk_89d4: Array<0xc, u8>,
        pub(crate) unk_89e0: u32,
        pub(crate) max_power_2: u32,
        pub(crate) ppm_kp: F32,
        pub(crate) ppm_ki_dt: F32,
        pub(crate) unk_89f0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_89f4_0: Array<0x8, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_89f4_8: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_89f4_c: Array<0x50, u8>,

        #[ver(V >= V13_3)]
        pub(crate) unk_89f4_5c: Array<0xc, u8>,

        pub(crate) unk_89f4: u32,
        pub(crate) hws1: HwDataShared1,
        pub(crate) hws2: HwDataShared2,

        #[ver(V >= V13_0B4)]
        pub(crate) idle_off_standby_timer: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_hws2_4: Array<0x8, F32>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_hws2_24: u32,

        pub(crate) unk_hws2_28: u32,

        pub(crate) hws3: HwDataShared3,
        pub(crate) unk_9004: Array<8, u8>,
        pub(crate) unk_900c: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_9010_0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_9010_4: Array<0x14, u8>,

        pub(crate) unk_9010: Array<0x2c, u8>,
        pub(crate) unk_903c: u32,
        pub(crate) unk_9040: Array<0xc0, u8>,
        pub(crate) unk_9100: Array<0x6f00, u8>,
        pub(crate) unk_10000: Array<0xe50, u8>,
        pub(crate) unk_10e50: u32,
        pub(crate) unk_10e54: Array<0x2c, u8>,

        #[ver((G >= G14X && V < V13_3) || (G <= G14 && V >= V13_3))]
        pub(crate) unk_x_pad: Array<0x4, u8>,

        // bit 0: sets sgx_reg 0x17620
        // bit 1: sets sgx_reg 0x17630
        pub(crate) fault_control: u32,
        pub(crate) do_init: u32,
        pub(crate) unk_10e88: Array<0x188, u8>,
        pub(crate) idle_ts: U64,
        pub(crate) idle_unk: U64,
        pub(crate) progress_check_interval_3d: u32,
        pub(crate) progress_check_interval_ta: u32,
        pub(crate) progress_check_interval_cl: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_4: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_8: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_c: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_1102c_10: u32,

        pub(crate) unk_1102c: u32,
        pub(crate) idle_off_delay_ms: AtomicU32,
        pub(crate) fender_idle_off_delay_ms: u32,
        pub(crate) fw_early_wake_timeout_ms: u32,
        #[ver(V == V13_3)]
        pub(crate) ps_pad_0: Pad<0x8>,
        pub(crate) pending_stamps: Array<0x100, PendingStamp>,
        #[ver(V != V13_3)]
        pub(crate) ps_pad_0: Pad<0x8>,
        pub(crate) unkpad_ps: Pad<0x78>,
        pub(crate) unk_117bc: u32,
        pub(crate) fault_info: FaultInfo,
        pub(crate) counter: u32,
        pub(crate) unk_118dc: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_118e0_0: Array<0x9c, u8>,

        #[ver(G >= G14X)]
        pub(crate) unk_118e0_9c: Array<0x580, u8>,

        #[ver(V >= V13_3)]
        pub(crate) unk_118e0_9c_x: Array<0x8, u8>,

        pub(crate) cl_context_switch_timeout_ms: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) cl_kill_timeout_ms: u32,

        pub(crate) cdm_context_store_latency_threshold: u32,
        pub(crate) unk_118e8: u32,
        pub(crate) unk_118ec: Array<0x400, u8>,
        pub(crate) unk_11cec: Array<0x54, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_11d40: Array<0x19c, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_11edc: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_11ee0: Array<0x1c, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_11efc: u32,

        #[ver(V >= V13_3)]
        pub(crate) unk_11f00: Array<0x280, u8>,
    }
    #[versions(AGX)]
    default_zeroed!(Globals::ver);

    #[derive(Debug, Default, Clone, Copy)]
    #[repr(C, packed)]
    pub(crate) struct UatLevelInfo {
        pub(crate) unk_3: u8,
        pub(crate) unk_1: u8,
        pub(crate) unk_2: u8,
        pub(crate) index_shift: u8,
        pub(crate) num_entries: u16,
        pub(crate) unk_4: u16,
        pub(crate) unk_8: U64,
        pub(crate) unk_10: U64,
        pub(crate) index_mask: U64,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct InitData<'a> {
        #[ver(V >= V13_0B4)]
        pub(crate) ver_info: Array<0x4, u16>,

        pub(crate) unk_buf: GpuPointer<'a, &'a [u8]>,
        pub(crate) unk_8: u32,
        pub(crate) unk_c: u32,
        pub(crate) runtime_pointers: GpuPointer<'a, super::RuntimePointers::ver>,
        pub(crate) globals: GpuPointer<'a, super::Globals::ver>,

        #[ver(V < V15_5)]
        pub(crate) fw_status: GpuPointer<'a, super::FwStatus>,

        // 15.5 dropped fw_status from InitData -- of the 0xc8 bytes it copies,
        // the only field anything reads back is +0x2c. Apple's own builder
        // stores a __DATA_CONST constant here: +0x28 = 0, +0x2c = 1, and +0x2c
        // is the field the "Host-mapped FW allocations are disabled" panic
        // checks. LOG.md 10.2.
        #[ver(V >= V15_5)]
        pub(crate) unk_28: u32,
        #[ver(V >= V15_5)]
        pub(crate) host_mapped_fw_allocations_15_5: u32,

        pub(crate) uat_page_size: u16,
        pub(crate) uat_page_bits: u8,
        pub(crate) uat_num_levels: u8,
        pub(crate) uat_level_info: Array<0x3, UatLevelInfo>,
        pub(crate) __pad0: Pad<0x14>,

        #[ver(V < V15_5)]
        pub(crate) host_mapped_fw_allocations: u32,
        #[ver(V < V15_5)]
        pub(crate) unk_ac: u32,
        #[ver(V < V15_5)]
        pub(crate) unk_b0: u32,
        #[ver(V < V15_5)]
        pub(crate) unk_b4: u32,
        #[ver(V < V15_5)]
        pub(crate) unk_b8: u32,

        // 15.5 replaced those five u32s with four 64-bit pointers, each stashed
        // in its own firmware global by the InitData consumer. Confirmed on
        // hardware: passing the literal 1 here (13.5's
        // host_mapped_fw_allocations) faults with FAR = 0x1. +0xa8 points at a
        // log/trace enable mask; +0xb0 at a >=0xcdcc structure that carries the
        // FWCtl channel at +0x4f78/+0x4f80. LOG.md 9.2, 9.3, 15.1.
        #[ver(V >= V15_5)]
        pub(crate) unkptr_a8: GpuPointer<'a, &'a [u8]>,
        #[ver(V >= V15_5)]
        pub(crate) unkptr_b0: GpuPointer<'a, &'a [u8]>,
        #[ver(V >= V15_5)]
        pub(crate) unkptr_b8: GpuPointer<'a, &'a [u8]>,
        #[ver(V >= V15_5)]
        pub(crate) unkptr_c0: GpuPointer<'a, &'a [u8]>,
    }

    /// Layout check for the 15.5 InitData port (LOG.md 10.2, 9.2).
    #[versions(AGX)]
    #[ver(V >= V15_5)]
    const _LAYOUT_INITDATA: () = {
        assert!(core::mem::offset_of!(InitData::ver<'static>, runtime_pointers) == 0x18);
        assert!(core::mem::offset_of!(InitData::ver<'static>, globals) == 0x20);
        assert!(core::mem::offset_of!(InitData::ver<'static>, host_mapped_fw_allocations_15_5) == 0x2c);
        assert!(core::mem::offset_of!(InitData::ver<'static>, unkptr_a8) == 0xa8);
        assert!(core::mem::size_of::<InitData::ver<'static>>() == 0xc8);
    };

    /// The 15.5 RegionB tail, anchored to the two offsets the firmware reads
    /// straight off its own global. LOG.md 32.41.73.
    #[versions(AGX)]
    #[ver(V >= V15_5)]
    const _LAYOUT_REGIONB_TAIL: () = {
        assert!(
            core::mem::offset_of!(RuntimePointers::ver<'static>, buffer_mgr_ctl_gpu_addr_15_5)
                == 0x2c0
        );
        assert!(
            core::mem::offset_of!(RuntimePointers::ver<'static>, uma_pool_ctl_gpu_addr) == 0x2d0
        );
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, usc_max_tgmem) == 0x2e0);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, unk_2e4) == 0x2e4);
        // Nothing below may move: the firmware reads `gpu_scratch` fields at
        // fixed RegionB offsets (RegionB+0x451, +0x6b38).
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, unk_1d0) == 0x2e8);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, unk_32c) == 0x32c);
        assert!(core::mem::offset_of!(RuntimePointers::ver<'static>, __pad1) == 0x33c);
        assert!(core::mem::size_of::<RuntimePointers::ver<'static>>() == 46296);
    };
}

#[derive(Debug)]
pub(crate) struct ChannelRing<T: GpuStruct + Debug + Default, U: Copy>
where
    for<'a> <T as GpuStruct>::Raw<'a>: Debug,
{
    pub(crate) state: GpuObject<T>,
    pub(crate) ring: GpuArray<U>,
}

impl<T: GpuStruct + Debug + Default, U: Copy> ChannelRing<T, U>
where
    for<'a> <T as GpuStruct>::Raw<'a>: Debug,
{
    pub(crate) fn to_raw(&self) -> raw::ChannelRing<T, U> {
        raw::ChannelRing {
            state: Some(self.state.weak_pointer()),
            ring: Some(self.ring.weak_pointer()),
        }
    }
}

trivial_gpustruct!(FwStatus);
trivial_gpustruct!(GpuGlobalStatsVtx);
#[versions(AGX)]
trivial_gpustruct!(GpuGlobalStatsFrag::ver);
trivial_gpustruct!(GpuStatsComp);

#[versions(AGX)]
trivial_gpustruct!(HwDataA::ver);

#[versions(AGX)]
trivial_gpustruct!(HwDataB::ver);

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Stats {
    pub(crate) vtx: GpuObject<GpuGlobalStatsVtx>,
    pub(crate) frag: GpuObject<GpuGlobalStatsFrag::ver>,
    pub(crate) comp: GpuObject<GpuStatsComp>,
}

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct RuntimePointers {
    pub(crate) stats: Stats::ver,

    pub(crate) hwdata_a: GpuObject<HwDataA::ver>,
    pub(crate) unkptr_190: GpuArray<u8>,
    pub(crate) unkptr_198: GpuArray<u8>,
    pub(crate) hwdata_b: GpuObject<HwDataB::ver>,

    pub(crate) unkptr_1b8: GpuArray<u8>,
    pub(crate) unkptr_1c0: GpuArray<u8>,
    pub(crate) unkptr_1c8: GpuArray<u8>,

    /// The u32 the firmware's push-telemetry accounting watches for change.
    /// Its GPU address lives at `HwDataA + 0x3a30`; the firmware dereferences
    /// that slot unconditionally, so it may not be null.
    #[ver(V >= V15_5)]
    pub(crate) ptd_watch: GpuArray<u8>,

    /// Backing for the two pointer slots in 15.5's RegionB header.
    #[ver(V >= V15_5)]
    pub(crate) unk_8_15_5: GpuArray<u8>,
    #[ver(V >= V15_5)]
    pub(crate) unk_10_15_5: GpuArray<u8>,

    /// Backing for RegionB+0x244..0x25c, four per-pipe tables (15.5 only).
    #[ver(V >= V15_5)]
    pub(crate) pipe_tables: GpuArray<u8>,

    pub(crate) buffer_mgr_ctl: gem::ObjectRef,
    /// CPU address of the PB descriptor table (15.5), for the hang dump.
    pub(crate) buffer_mgr_ctl_cpu: usize,
    pub(crate) buffer_mgr_ctl_low_mapping: Option<mmu::KernelMapping>,
    pub(crate) buffer_mgr_ctl_high_mapping: Option<mmu::KernelMapping>,

    /// The pass-page counter page, dual-mapped like `buffer_mgr_ctl`.
    /// `Scene.control_word` + register `0x1c910` carry the LOW address (gPM
    /// increments the u32 there); `Scene.control_word2` carries the HIGH one
    /// (the firmware CPU reads and clears it into `pass_page_count`).
    /// LOG.md 32.41.507/.508.
    pub(crate) pm_counter_obj: gem::ObjectRef,
    pub(crate) pm_counter_low_mapping: Option<mmu::KernelMapping>,
    pub(crate) pm_counter_high_mapping: Option<mmu::KernelMapping>,

    /// The UMA page pool descriptor table -- the second of the two hardware
    /// mappings `createHardwareMappings` builds, dual-mapped exactly like
    /// `buffer_mgr_ctl`. Allocated on every version; only 15.5 publishes its
    /// addresses. LOG.md 32.41.72.
    pub(crate) uma_pool_ctl: gem::ObjectRef,
    /// CPU address of `uma_pool_ctl`'s first page, for the fault-path dump.
    pub(crate) uma_pool_ctl_cpu: usize,
    pub(crate) uma_pool_ctl_low_mapping: Option<mmu::KernelMapping>,
    pub(crate) uma_pool_ctl_high_mapping: Option<mmu::KernelMapping>,
}

#[versions(AGX)]
impl GpuStruct for RuntimePointers::ver {
    type Raw<'a> = raw::RuntimePointers::ver<'a>;
}

#[versions(AGX)]
trivial_gpustruct!(Globals::ver);

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct InitData {
    pub(crate) unk_buf: GpuArray<u8>,
    pub(crate) runtime_pointers: GpuObject<RuntimePointers::ver>,
    pub(crate) globals: GpuObject<Globals::ver>,
    pub(crate) fw_status: GpuObject<FwStatus>,

    // 15.5 only: the two InitData tail pointers the firmware dereferences.
    // Contents unknown; zeroed is enough to get past bring-up, and +0xa8's
    // first word is a log/trace mask where zero means "quiet". LOG.md 10.4.
    #[ver(V >= V15_5)]
    pub(crate) unkbuf_a8: GpuArray<u8>,
    #[ver(V >= V15_5)]
    pub(crate) unkbuf_b0: GpuArray<u8>,
    #[ver(V >= V15_5)]
    pub(crate) unkbuf_b8: GpuArray<u8>,
    #[ver(V >= V15_5)]
    pub(crate) unkbuf_c0: GpuArray<u8>,
}

#[versions(AGX)]
impl GpuStruct for InitData::ver {
    type Raw<'a> = raw::InitData::ver<'a>;
}
