// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![recursion_limit = "2048"]

//! Driver for the Apple AGX GPUs found in Apple Silicon SoCs.

mod alloc;
mod buffer;
mod channel;
#[cfg(CONFIG_DEV_COREDUMP)]
mod crashdump;
mod debug;
mod driver;
mod event;
mod file;
mod float;
mod fw;
mod gem;
mod gpu;
mod hw;
mod initdata;
mod mem;
mod microseq;
mod mmu;
mod object;
mod pgtable;
mod queue;
mod regs;
mod replay;
mod slotalloc;
mod stats;
mod util;
#[cfg(CONFIG_DRM_ASAHI_MAPLE_TREE)]
mod vm;
mod workqueue;

kernel::module_platform_driver! {
    type: driver::AsahiDriver,
    name: "asahi",
    description: "AGX GPU driver for Apple silicon SoCs",
    license: "Dual MIT/GPL",
    params: {
        debug_flags: u64 {
            default: 0,
            // permissions: 0o644,
            description: "Debug flags",
        },
        fault_control: u32 {
            default: 0xb,
            // permissions: 0,
            description: "Fault control (0x0: hard faults, 0xb: macOS default)",
        },
        initial_tvb_size: usize {
            default: 512,
            // permissions: 0o644,
            description: "Initial TVB size in blocks",
        },
        robust_isolation: u32 {
            default: 0,
            // permissions: 0o644,
            description: "Fully isolate GPU contexts (limits performance)",
        },
        usc_base_isp: u64 {
            default: 0,
            description: "DIAGNOSTIC: override USC_EXEC_BASE_ISP (reg 0x10069); 0 = use the queue's usc_exec_base",
        },
        usc_base_isp_struct: u64 {
            default: 0,
            description: "DIAGNOSTIC: override Start3D.usc_exec_base_isp; 0 = use the queue's usc_exec_base",
        },
        dump_stats_vtx: u32 {
            default: 0,
            description: "G15S: dump this many dwords of the firmware's GpuStatsVtx block at the hang. Driver-allocated memory the firmware writes into, so reading it is a page-table walk and cannot perturb the run -- unlike every 0xc020 probe, which does (LOG.md 32.41.336). 0 = off. LOG.md 32.41.337",
        },
        reg_va_fold: u32 {
            default: 0,
            description: "G15S: force the register-side VA fold (a - 2^36) even when user_va_shift keeps the bias in the page tables. The live macOS TA capture (2026-09-11) shows TA address registers carry VA - 0x1000000000 while the fragment list carries the unfolded address -- same buffer, two encodings. The two models were tied to one knob; with the aliases mapped both can hold. 0 = off. LOG.md 32.41.372",
        },
        vtx_small_job: u32 {
            default: 0,
            description: "G15S: apply macOS's SMALL-JOB values to the nine TA geometry registers (0x1c041/0x1c9d0/0x1c948/0x1c888/0x1c920/0x16c41/0x1ca40/0x1c918 = 0, 0x1c890 = 1). macOS programs these only at >=63 primitives; this driver programs all nine unconditionally, handing a 1-triangle job pointers into partial-render machinery it never set up. 0 = off. LOG.md 32.41.371 (entry LOST -- see 32.41.431)",
        },
        vtx_zero_regs: u32 {
            default: 0,
            description: "G15S: also program the five TA registers macOS emits and this driver never has -- 0x1c199/0x1c1a1/0x1c1a9/0x1c1b1/0x1c1b9, all zero, a contiguous stride-8 block. Found by diffing the live macOS TA capture (2026-09-11) against this driver's list; classified CONSTANT across 26 configs. Writing 0 is not the same as not writing: the register keeps whatever the last job left. UNVERIFIED: .370 is lost and its premise conflicts with 32.41.373, which finds our id list is macOS's minus exactly one register. 0 = off. LOG.md 32.41.370 (entry LOST -- see 32.41.431)",
        },
        state_poll_n: u32 {
            default: 0,
            description: "G15S: after the kick, sample 0xc020/0xc128/0xc120 this many times with a BARE read that never writes the 0x10398 selector, logging only on change. Unlike state_in_flight (which uses dump_state_pre and perturbs, 32.41.336) this writes nothing, so it is safe in flight. Separates 'the TA never starts' from 'ran and went idle' -- 32.41.327's null was sampled ~40s late. 0 = off. LOG.md 32.41.361",
        },
        state_poll_ms: u32 {
            default: 1,
            description: "G15S: milliseconds between state_poll_n samples. LOG.md 32.41.361",
        },
        state_in_flight: u32 {
            default: 0,
            description: "G15S: sample the windowed state register 0xc020 while the job is in flight (at probe_delay_ms, just after the kick), selector saved and restored. 32.41.327's 'the TA never starts' rests on a sample taken ~40s later, which cannot distinguish 'never started' from 'ran and went idle'. 0 = off. LOG.md 32.41.335",
        },
        cs_probe_ops: u32 {
            default: 0,
            description: "G15S: read the first this-many dwords of the VDM control stream and probe every address-shaped operand (0xfff8xxxx) at the stream's own region base. The submitted stream is byte-identical between runs that fault and runs that do not (LOG.md 32.41.330), so whatever differs is in the buffers the operands point at, not the stream. 0 = off. LOG.md 32.41.333",
        },
        cs_count_fix: u32 {
            default: 0,
            description: "G15S: scan this many dwords of the VDM control stream and clear bits [31:16] of any dword of the form 0xfff8xxxx whose successor is < 0x10000. The VDM reads those bits as a COUNT and Mesa writes an address there, so it consumes 65528 units and runs off the end of the BO (LOG.md 32.41.320/.321). Content-based, unlike the fixed-offset cs_poke_off, because the record moves between frames (32.41.329). 0 = off",
        },
        probe_raw: u32 {
            default: 0,
            description: "Debug: let the hang/post-job scene probe dereference the CPU views published by the last submitted scene (tiler counters, buffer::Info, pool record, timestamps, command dumps). They can belong to an exited process -> kernel oops in the timeout handler. 0 = off. LOG.md 32.41.667zzz33",
        },
        dump_state_hung: u32 {
            default: 0,
            description: "G15S: sample the windowed GPU state register 0xc020 (units 0/1, selector saved and restored) from the hang path only. Separate from dump_uma_regs because that one also enables a post-kick sweep that writes the 0x10398 unit selector WHILE the firmware is running, which measurably reintroduces the CSD runaway -- 3/3 with it against 2/3 clean without, LOG.md 32.41.326. 0 = off",
        },
        user_va_win36: u32 {
            default: 1,
            description: "G15S: map every user BO a second time at 2^36 | (VA & 0xffffffff), the address the VDM actually forms for control-stream operands. Measured: poking a value into the operand slot at cs+0x2c faults at exactly 2^36 + value, tracked across three values over seven runs. Unlike the other aliases this is not a constant bias -- Mesa's buffers do not share high bits. 0 = off. LOG.md 32.41.313",
        },
        win36_min_va: u64 {
            default: 0x1100000000,
            description: "Only apply the user_va_win36 alias to BOs at VA >= this. A blanket win36 is unusable: every BO maps to 2^36 | (VA & 0xffffffff), so BOs differing only above bit 32 collide, and the USC code BO lost its alias to another buffer, leaving a zero page where the shader had to be (LOG.md 32.41.419). 0x1100000000 restricts it to Mesa's USC heap, which is the only consumer that needs it",
        },
        win36_base: u64 {
            default: 0x10000000000,
            description: "G15S: base the user_va_win36 alias is formed against -- alias = win36_base | (VA & 0xffffffff). Default 2^36. Set to 0 to alias at the bare low 32 bits, which is what the TA needs: with reg_va_win36=1 the GPU faulted on a read at 0x148180 (folded heapmeta 0x148000 + 0x180) and NOT at 0x1000148180, so it treats the register value as a plain GPU VA and adds no base. LOG.md 32.41.435",
        },
        win36_dual: u32 {
            default: 0,
            description: "G15S: map the user_va_win36 alias at BOTH win36_base and its 2^36 complement, so a folded TA address resolves whether or not the hardware adds a 2^36 base. win0-1 showed it adds none for the heapmeta read (faulted at 0x148180) but does for the cluster layermeta read (faulted at 0x1000178080 while 0x178080 was mapped). 0 = single alias. LOG.md 32.41.436",
        },
        win36_min_va2: u64 {
            default: 0,
            description: "G15S: start of a SECOND user_va_win36 window (inclusive). The TA address registers point at tiler buffers at 0x38xxxxxxxx, while window 1 covers the USC heap at 0x11xxxxxxxx; one widened window would swallow Mesa's control stream at 0x2fxxxxxxxx in between. 0 = unused. LOG.md 32.41.433",
        },
        win36_max_va2: u64 {
            default: 0,
            description: "G15S: end (exclusive) of the second user_va_win36 window; 0 = no upper bound. LOG.md 32.41.433",
        },
        reg_va_win36: u32 {
            default: 0,
            description: "G15S: TA address registers carry VA & 0xffffffff (the win36-folded offset) instead of the full VA, preserving any tag in the top 16 bits. Derived from the live macOS capture: its three tiler buffers sit at 0x1000308000/0x1000309000/0x1000330000 and its TA registers read 0x308000/0x309000/0x330000 -- a difference of exactly 2^36 in all three, i.e. the field is a 32-bit offset into a 2^36 window, not a VA. Every macOS TA address is <=22 bits; every one of ours is 38. MUST be paired with user_va_win36=1 and a window covering the tiler heap, or the folded value points at nothing mapped -- which is why reg_va_fold alone faulted 3/3 in 32.41.373. 0 = off. LOG.md 32.41.433",
        },
        win36_max_va: u64 {
            default: 0x1200000000,
            description: "Upper bound (exclusive) of the user_va_win36 window; 0 = no upper bound. MUST be set alongside win36_min_va: Mesa's control-stream (0x2fxxxxxxxx) and tiler (0x38xxxxxxxx) heaps sit ABOVE the USC heap (0x11xxxxxxxx), so a floor alone excludes nothing. 0x1200000000 confines the alias to the 4 GiB USC window (LOG.md 32.41.420)",
        },
        user_va_bias2: u64 {
            default: 0,
            description: "DIAGNOSTIC: map every user page at a second alias VA+this, on top of user_va_shift. LOG.md 32.41.191",
        },
        cs_poke2_off: u32 {
            default: 0xffffffff,
            description: "DIAGNOSTIC: second VDM control-stream quadword poke, offset. LOG.md 32.41.187",
        },
        cs_poke2_val: u64 {
            default: 0,
            description: "DIAGNOSTIC: second VDM control-stream quadword poke, value. LOG.md 32.41.187",
        },
        scratch_page_va: u64 {
            default: 0,
            description: "DIAGNOSTIC: back this user VA with a zeroed scratch range so an unmapped read succeeds; 0 = off. LOG.md 32.41.196",
        },
        vtx_end_mode: u32 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: 0 = terminal WaitForIdle2(END) on the vertex microsequence (current, matches the kext), 1 = omit it. The firmware spins in that wait because (0xc020 & 0x1e) never reaches 0, so the stamp is never retired (LOG.md 32.41.404)",
        },
        frag_end_mode: u32 {
            default: 0,
            description: "DIAGNOSTIC: 0 = terminal WaitForIdle2(END) on the fragment microsequence (current), 1 = omit it. LOG.md 32.41.202",
        },
        scratch_page_va2: u64 {
            default: 0,
            description: "DIAGNOSTIC: second scratch-backed VA (see scratch_page_va). LOG.md 32.41.200",
        },
        scratch_page_va3: u64 {
            default: 0,
            description: "DIAGNOSTIC: third scratch-backed VA (see scratch_page_va). Intended for the gVDM_SSD_SP read at 0x2800001580 -- satisfying it narrowly, because the blanket VA-2^36 alias stops the firmware reaching job execution 2/3 of the time. LOG.md 32.41.341",
        },
        scratch_page_word: u64 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: fill ONLY scratch_page_va3's page with this repeating 64-bit word (slots 1 and 2 back reads that want zeros). Used to plant a VDM Stream Link at the surviving dVDM_CSD fault page so the unit jumps to Mesa's Index List wherever it lands (LOG.md 32.41.377)",
        },
        scratch_page_fill: u32 {
            default: 0,
            description: "DIAGNOSTIC: fill scratch_page_va with self-describing words (0x3bad000000|off<<12) instead of zeros. LOG.md 32.41.198",
        },
        scratch_page_pages: u32 {
            default: 64,
            description: "DIAGNOSTIC: how many 16K pages scratch_page_va covers (all aliasing one zeroed page)",
        },
        cs_draw_relocate: u32 {
            default: 0,
            description: "G15S: copy the 4-dword draw record from cs+0x34 to cs+0x48 and terminate at cs+0x58. The measured record map (LOG.md 32.41.143-145, reproduced .304) has record @0x28 consuming 0x28..0x48, so the draw Mesa placed at 0x34 is swallowed as its payload and never executes; 0x48 is the next real boundary. Scans this many dwords for the CTRL word. 0 = off. LOG.md 32.41.305",
        },
        cs_draw_ctrl_exact: u32 {
            default: 0,
            description: "G15S: replace the VDM draw CTRL word outright with this value (0 = off). Apple's encoder writes exactly 0x61c40000 -- lane 0 of the vector is `mov w8, #0x61c40000` and is never touched again before `str q0, [x9], #16` (AGXMetalG15X_M1 0x698b40..0x698b8c) -- while Mesa emits 0x61c00600 and cs_draw_ctrl_fix only sets bit 18, leaving 0x0600 in the low half. Independent of cs_draw_len_fix. LOG.md 32.41.302",
        },
        cs_draw_ctrl_match: u32 {
            default: 0x61c0_0000,
            description: "G15S: the pattern (under mask 0xffff0000) cs_draw_ctrl_exact searches the control stream for. \u{26a0} Was hardcoded 0x61c40000 -- a CTRL word that ALREADY has bit 18 set -- so with cs_draw_ctrl_fix off (it is the 32.41.447 regression) the splice matched nothing and cs_draw_ctrl_exact silently did nothing; a run set up to test Apple's bit 18 was VOID and looked identical to the baseline. Mesa emits 0x61c00600, so 0x61c00000 is the pattern that matches. LOG.md 32.41.469",
        },
        cs_draw_len_fix: u32 {
            default: 0,
            description: "G15S: scan this many dwords of the VDM control stream and extend Mesa's 4-dword draw record to Apple's 8-dword one -- +0x10 = 0x30, +0x18 = 0x20, terminator moved to +0x20. Transcribed from AGXMetalG15X_M1's built-in blit stream at file 0x6de600. 0 = off. LOG.md 32.41.299",
        },
        cs_draw_ctrl_low: u32 {
            default: 0,
            description: "G15S: with cs_draw_len_fix, also replace the draw CTRL word's low half (Mesa writes 0x0600, Apple's blit has 0x0900). Value is the new low 16 bits; 0 = leave Mesa's. Separate knob so length and CTRL are testable apart. LOG.md 32.41.299",
        },
        cs_keep_vdmstate_low: u32 {
            default: 1,
            // permissions: 0o644,
            description: "G15S: with cs_draw_ctrl_fix, do NOT clear the low 16 bits of the 0x4000xxxx VDM State Update record. Those bits are Mesa's present mask (0x2e = VS word 0, VS word 1, Vertex Outputs, Vertex Unknown); clearing them leaves the four state words in the stream unconsumed and the vertex shader unbound. 1 = keep Mesa's mask (LOG.md 32.41.382)",
        },
        cs_draw_ctrl_fix: u32 {
            default: 64,
            description: "G15S: scan this many dwords of the VDM control stream and correct Mesa's draw CTRL words to Apple's G15 encoding -- 0x61c0xxxx |= bit 18, 0x61e0xxxx |= bit 17. 0 = off. LOG.md 32.41.252",
        },
        cs_dump_count: u32 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: dump this many dwords of the VDM control stream as the GPU sees it, at submit time, after all fixes are applied. The CSD consumes the whole 512 KiB BO, so the defect is a record-length advance in the first records; this is the only way to read them back. 0 = off. LOG.md 32.41.298",
        },
        initdata_unk_540: u32 {
            default: 0,
            description: "DIAGNOSTIC: InitData unk_540, the word immediately after num_frags. num_gps never reaches the firmware at all; the tiler runs on the GPs, and this driver reports 4 per cluster (8 total) to nobody. LOG.md 32.41.270",
        },
        drop_extra_vtx_regs: u32 {
            default: 0,
            description: "DIAGNOSTIC: omit the tiler registers this driver emits that macOS's own list does not (0x101e1). A register macOS never writes is a candidate for gating the TA stage off. LOG.md 32.41.269",
        },
        tiling_control_ovr: u64 {
            default: u64::MAX,
            description: "DIAGNOSTIC: override tvb_tiling_control (reg 0x1c890); u64::MAX = the SoC config value. t6030 currently carries t602x's 0x180340, which is a guess. LOG.md 32.41.265",
        },
        tiling_control2_ovr: u64 {
            default: u64::MAX,
            description: "DIAGNOSTIC: override tiling_control_2 (reg 0x1c918); u64::MAX = the built-in 4. LOG.md 32.41.265",
        },
        user_va_bias3: u64 {
            default: 0,
            description: "DIAGNOSTIC: map every user page at a third alias VA+this. 0x400000000 puts Mesa's USC heap start at 0x1500000000, the base-only USC read. LOG.md 32.41.263",
        },
        vdm_base_sub: u64 {
            default: 0,
            description: "DIAGNOSTIC: subtract this from the value programmed into VDM_CTRL_STREAM_BASE (0x1c880). The CSD reads from cs+0x80000 onwards, so 0x80000 tests whether the start is simply biased. LOG.md 32.41.251",
        },
        usc_seg_base: u64 {
            default: 0,
            description: "DIAGNOSTIC: write all five of 0x1c199..0x1c1b9 with this value; !0 (u64::MAX) means va(usc_exec_base). 0 = leave them zero. LOG.md 32.41.249",
        },
        usc_seg_probe: u64 {
            default: 0,
            description: "DIAGNOSTIC: write 0x1c199/0x1c1a1/0x1c1a9/0x1c1b1/0x1c1b9 with this base plus k*0x1000000 instead of 0, so a fault names which of macOS's five base registers the VDM's USC segment comes from. 0 = off. LOG.md 32.41.249",
        },
        reg_10151: u64 {
            default: 0,
            description: "DIAGNOSTIC: value for reg 0x10151, whose bit 1 is documented to enable 0x1c1a9. LOG.md 32.41.249",
        },
        usc_base_cp: u64 {
            default: 0xffff_ffff_ffff_ffff,
            description: "DIAGNOSTIC: override USC_EXEC_BASE_CP (reg 0x10071); !0 = use the queue's usc_exec_base. 0 is legal. LOG.md 32.41.195",
        },
        usc_base_ta_struct: u64 {
            default: 0xffff_ffff_ffff_ffff,
            description: "DIAGNOSTIC: override StartTA.usc_exec_base_ta; !0 = use the queue's usc_exec_base. 0 is legal. LOG.md 32.41.186",
        },
        usc_base_ta: u64 {
            default: 0xffff_ffff_ffff_ffff,
            description: "DIAGNOSTIC: override USC_EXEC_BASE_TA (reg 0x10061); !0 = use va(usc_exec_base). 0 is a legal value. LOG.md 32.41.185",
        },
        usc_absolute: u32 {
            default: 1,
            description: "G15S: program BG/EOT pipeline addresses absolute (usc_exec_base + usc) instead of USC-relative",
        },
        usc_window: u64 {
            default: 0x10000000000,
            description: "G15S (Mesa G15 path): BG/EOT launch-program registers = usc_window + usc, where Mesa passes the code-window offset (Apple: 0x15371 = 0x10000138240). Use 0x10000000000 with user_va_win36=1 win36_base=0x10000000000. Overrides usc_absolute. 0 = off. LOG.md 32.41.655",
        },
        ignore_fault: u32 {
            default: 0,
            description: "DIAGNOSTIC: report GPU faults but do not fail pending jobs (LOG.md 32.41.108)",
        },
        max_pstate: u32 {
            default: 0,
            description: "Clamp the maximum GPU performance state (0 = use the whole table; LOG.md 32.41.111)",
        },
        vtx_kick_first: u32 {
            default: 0,
            description: "Kick the vertex pipe before the fragment pipe, so the fragment's Barrier is evaluated after the vertex has run (LOG.md 32.41.113)",
        },
        kick_delay_us: u32 {
            default: 0,
            description: "With vtx_kick_first, sleep this long before the fragment kick (LOG.md 32.41.113)",
        },
        refrag_us: u32 {
            default: 0,
            description: "DIAGNOSTIC: this many microseconds after the vertex doorbell, ring the fragment doorbell again, so the firmware re-scans the pipe once the vertex has retired (LOG.md 32.41.124)",
        },
        refrag_n: u32 {
            default: 1,
            description: "With refrag_us, how many times to re-ring the fragment doorbell (LOG.md 32.41.124)",
        },
        prc_watch: u32 {
            default: 0,
            description: "DIAGNOSTIC: after the vertex kick, sample the PRC perf-state registers (SGX 0xd4c000/4) this many times at 20us and print every change (LOG.md 32.41.123)",
        },
        job_timeout_ms: u32 {
            default: 10000,
            description: "DRM scheduler job timeout in ms; on expiry the host stops the firmware for recovery (MSG_HALT, 15.5) and fails the hung queue's jobs (LOG.md 32.41.667zzz29)",
        },
        b0_selfptr: u32 {
            default: 0,
            description: "Initialise b0[0x4f90] (the firmware's recovery count) with b0's own address, as before .667zzz29 (blocks host recovery)",
        },
        host_recovery: u32 {
            default: 1,
            description: "On a DRM scheduler job timeout, run Apple's host-initiated firmware recovery (AGXArmFirmware::stopFirmwareForRecoveryInspection: doorbell-EP msg 0x85 | b0[0x4f90]) instead of leaving the GPU wedged (LOG.md 32.41.667zzz29)",
        },
        ignore_timeout: u32 {
            default: 0,
            description: "DIAGNOSTIC: report the firmware Timeout event but do not fail pending jobs or recover (LOG.md 32.41.114)",
        },
        kick_wake: u32 {
            default: 0,
            description: "Send doorbell type 0x86 (kickFirmware wake=true -> PM event 0x24) before every 0x83 kick. LOG.md 32.41.667zc",
        },
        // 0xff by default: with 0 (or bit 0 alone) the firmware only advances one frame per 8.33 ms DVFS tick
        // (fragment starts ~0x30400 ticks after the vertex ends); with all categories on it pairs TA->3D within
        // 4 ticks -- build 121 -> 231 FPS (LOG.md 32.41.667v-x). The gated code is trace-only, so the speedup is a
        // side effect of emitting records (the emitter's host doorbell / timing); root cause still open.
        ktrace_mask: u32 {
            default: 0xff,
            description: "Firmware trace category mask (Globals.ktrace_enable -> fw:0x1a383c); 0xff enables all six categories, print with debug_flags bit 17 (LOG.md 32.41.112). Default 0xff: see 667v-x",
        },
        probe_usc: u32 {
            default: 0,
            description: "Probe the VM for Mesa's BG/EOT USC programs at submit time",
        },
        user_va_shift: u32 {
            default: 36,
            // permissions: 0o644,
            description: "DIAGNOSTIC: bias user page-table mappings by 1 << N (36 on G15/15.5; LOG.md 32.41.39)",
        },
        uma_aperture_size: usize {
            default: 0x40000,
            // permissions: 0o644,
            description: "Backing size for the 15.5 UMA pool aperture; 0 leaves the descriptor zeroed (LOG.md 32.41.51)",
        },
        uma_aperture_probe: u64 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: force this base into both UMA pool descriptors (LOG.md 32.41.53)",
        },
        // Left at 0 (the struct's natural zero). -1 is what macOS's
        // populateInitialFirmwareState uses, but 32.41.65 showed it only
        // replaces a base of 0 with all-ones in the appended pool register,
        // because fw:0x24f78 ORs in the *sign-extended* id. It is not an
        // improvement on its own. LOG.md 32.41.65.
        initdata_ptr_probe: u64 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: poison RuntimePointers.unk_160/unk_168, candidate UMA descriptor-table GPU/FW addresses (LOG.md 32.41.68)",
        },
        uma_pool_id: i64 {
            // A real pool works now that `record+0x18` carries an address
            // (LOG.md 32.41.87). -1 was the workaround for the null, not the
            // answer: it made the firmware install no descriptor at all.
            default: 0,
            // permissions: 0o644,
            description: "UMA pool descriptor slot id; macOS's initial firmware state uses -1 = no slot (LOG.md 32.41.61)",
        },
        uma_base_probe: u64 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: UMA pool base in StartVertex+0x198 / StartFragment+0x1b0, which the driver leaves null (LOG.md 32.41.60)",
        },
        info_poison: u64 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: fill the buffer manager Info's unknown fields with distinct unmapped addresses (LOG.md 32.41.58)",
        },
        g15_frag_regs: u32 {
            default: 1,
            // permissions: 0o644,
            description: "Program the fragment register list macOS actually uses on G15S (82 entries, from the 15.5 DTrace capture) instead of the 67-entry G13-derived set; 0 restores the old set (LOG.md 32.41.215)",
        },
        g15_vtx_regs: u32 {
            default: 1,
            // permissions: 0o644,
            description: "Program the vertex/TA register list macOS uses on G15S (74 entries, derived from AGXTAChannelG15::generateRegisterList in the 15.5 kernelcache) instead of the 60-entry G13-derived set; 0 restores the old set (LOG.md 32.41.216)",
        },
        vtx_1c910_src: u32 {
            default: 12,
            // permissions: 0o644,
            description: "Program the tiler's 0x1c910 (last register in macOS's G15S list, an address folded to GART-relative with bit 0 set) from: 0 off, 1 tvb_tilemap, 2 tvb_heapmeta, 3 tpc, 4 user_buffer, 5 meta_1, 6 buffer Info, 7 preempt_buf_1 (LOG.md 32.41.221)",
        },
        vtx_macos_zeros: u32 {
            default: 1,
            // permissions: 0o644,
            description: "Zero the complete set of 11 vertex registers macOS holds at zero for a sub-63-primitive job (the 9 vtx_small_job covers, plus 0x1c089/0x1c9e0 tvb_cluster_layermeta, which macOS leaves zero in ALL 40 captured configurations) (LOG.md 32.41.375)",
        },
        vtx_macos_consts: u32 {
            default: 2,
            // permissions: 0o644,
            description: "Fix the two non-address scalar mismatches against Apple's captured G15S register list: 0x1c9a8 = 0x3c (we compute 0x1c -- the right value sourced into the wrong register) and 0x1c830. 1 = the old behaviour, 0x1c830 = 9. 2 = 0x1c830 = scene.slot(). WARNING 32.41.513: 0x1c830 is NOT a constant -- decoding Apple's RegisterArray with the firmware's own layout (12-byte entries, u16 count at array+0x708, fw:0x14f8c) gives entry [70] 0x1c830 = 0xd, and that job's RunVertex.buffer_slot is also 0xd. It is the buffer slot; the '9' generalised one capture's slot number (LOG.md 32.41.375, corrected 32.41.513)",
        },
        tvb_ovf_scene: u32 {
            default: 0,
            description: "G15/15.5: point StartFragment.tvb_overflow_count at Scene+0x4c as Apple does (capture4, LOG.md 32.41.608) instead of a private work-item word. fw:0x175dc writes 1 there on an overflowed pass; FinalizeFragment (fw:0x167e8) reads and clears scene+0x4c. 0 = off (LOG.md 32.41.667zzx)",
        },
        pm_dualmap: u32 {
            default: 1,
            description: "G14X+/15.5: point Scene.control_word (+0x00) and control_word2 (+0x08) at ONE dual-mapped page instead of two separate allocations. pass_page_count is a COPY of the u32 behind control_word2 (fw:0x15f20 reads and clears it); gPM increments that word and reaches it through register 0x1c910, which can only express a LOW address. So the two fields must name the same page in two address spaces -- which this driver never did: control_word pointed at pm_counter (ualloc_priv) and control_word2 at a different allocation, so gPM incremented a word nobody read and pass_pages could never be non-zero in ANY configuration (LOG.md 32.41.507). The page is now mapped exactly like buffer_mgr_ctl: kernel_lower_vm at IOVA_KERN_GPU_PMCT_LOW for gPM, kernel_vm at IOVA_KERN_GPU_PMCT_HIGH for the firmware CPU. Pair with vtx_1c910_src=12, which folds control_word_pointer() into 0x1c910. 667zzx: the LOW page now sits in top-level slot 7, shared into every user VM (gPM walks it through the job context -- the .508 fault), with one 64-byte cell per buffer slot; pass_pages is real and auto_grow grows the TVB from it. Default on. 0 = off (LOG.md 32.41.508, 667zzx)",
        },
        fs_g13: u32 {
            default: 0,
            description: "DIAGNOSTIC (apple_replay): replace Apple's G15 triangle fragment shader (0x100000006c0) with a hand-assembled G13 program (ldimm r1,0xdeadbeef; ldimm r4/r5 = 0x10000044800; store r1,dr4; wait; stop -- encodings from Mesa's AGX2 test vectors). 0xdeadbeef at RT+0x4800 => G15 executes G13 ISA. LOG.md 32.41.643",
        },
        frag_scene_bias: u64 {
            default: 0x1000000000,
            description: "Add this to the fragment's TVB scene registers 0x16429 (tilemap), 0x16098 (heapmeta), 0x16060 (layermeta). With user_va_shift=36 every user page is also mapped at VA+2^36; if the tiler writes the TMAP through that alias while the ISP reads the plain VA, a VA-tagged GPU cache hands the ISP stale data (the .637 oracle). Apple's ISP reads 0x1000308000 = the +2^36 form. 0x1000000000 = read through the alias. LOG.md 32.41.642",
        },
        tmap_poll: u32 {
            default: 0,
            description: "After the vertex kick, poll TMAP[0]/HMTA[0]/TVB page-1 word 0 from the CPU for up to N us and print the cntvct of each first non-zero value (compare with ktrace t). LOG.md 32.41.639",
        },
        prefill_scene: u32 {
            default: 0,
            description: "Oracle: bit0 = write the known-final TMAP (5 words) and HMTA (1 word) from the CPU before the kick, to test whether the ISP reads a stale tile map in the fast (no-delay) render (LOG.md 32.41.637)",
        },
        vtx_att_tvb: u32 {
            default: 0,
            description: "15.5: add a StartVertex attachment (hardware write-back range, fw:0x150f4 -> hw_base+0xc238) covering the 2 MiB window of the TVB blocks and scene buffers, so the TA's output is written back before the ISP reads it (LOG.md 32.41.636)",
        },
        frag_no_att: u32 {
            default: 0,
            description: "15.5: StartFragment attachment count 0, as in Apple's captured 3D microsequence (op+0x1a0 = 0). Each entry makes fw:0x15c88 append a 128-byte-granular cache-range command (trace 0x116) at ISP start (LOG.md 32.41.635)",
        },
        frag_qid_zero: u32 {
            default: 0,
            description: "15.5: StartFragment event_generation (opcode+0x5c) = 0 instead of the queue id. The firmware appends a context register valued `vm_slot << 8 | (opcode+0x5c & 0x3f)` for the 3D (fw:0x15b80), the same idiom as the TA's opcode+0x4c (fw:0x15000, already 0 via vtx_qid_zero). With the queue id the 3D ran with 0x103 while its TA ran with 0x100; Apple uses 0xb00 for both (LOG.md 32.41.634)",
        },
        vtx_d8f0: u32 {
            default: 1,
            description: "15.5: fix StartVertex's MMIO 0xd8f0 bit-enable. The handler's tail (fw:0x15088..0x150bc) reads a PAIR out of the opcode -- `ldp w8, w9, [x28,#64]` with x28 = opcode+4, so opcode+0x44 and opcode+0x48 -- then does `lsl x8, x27, x8` (x27 == 1), reads 0xd8f0, computes both `orr` (set) and `bic` (clear), and selects with `cmp w9, #0; csel x8, x8, x12, eq`: w9 == 0 CLEARS the bit, non-zero SETS it. The same (index, enable) idiom appears in the op-0xb handler (fw:0x14e50) and StartCompute (fw:0x153a8), so the reading is general. opcode+0x48 is the pre-15.5 `buffer_slot`, and the firmware reads it ONLY as a boolean here -- it is an ENABLE FLAG on 15.5, not a slot. This driver passes scene.slot(), which is 0 (see `asahi/ids vtx: ... buffer_slot 0`, LOG.md 32.41.39), so every vertex job of this phase has been CLEARING bit vm_slot instead of setting anything. Apple's own captured TA microsequence has opcode+0x44 = 0xb and opcode+0x48 = 1 in BOTH the draw and the clear (capture4, useq-evidence/), i.e. it SETS bit 11; its FinalizeVertex pair is (0,0), identical to ours, so only the Start side is wrong. 0 = off (scene.slot(), the old behaviour), 1 = enable 1 with the index left at vm_slot, 2 = enable 1 AND index forced to Apple's 0xb (LOG.md 32.41.510)",
        },
        vtx_1c910_raw: u64 {
            default: 0,
            // permissions: 0o644,
            description: "Program the tiler's 0x1c910 with this literal value instead of deriving it. REFUTED 32.41.513: the 'device constant' claim is REFUTED -- capture4 raw-ta-complete-draw-1 has 0xa000161045 where the 40-config sweep saw 0xa000160b45 (xor 0x1b00). It VARIES, so it is a folded address after all, consistent with .507's derivation. Prefer vtx_1c910_src; 0 = off (LOG.md 32.41.373, corrected 32.41.513)",
        },
        missing_regs_probe: u64 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: append the 16 registers macOS programs and this driver does not, each with a distinct unmapped address (LOG.md 32.41.56)",
        },
        uma_scratch_probe: u64 {
            default: 0,
            // permissions: 0o644,
            description: "Override the second pool register's address (microsequence record+0x18); 0 uses the scene's UMA aperture (LOG.md 32.41.87)",
        },
        dump_uma_regs: u32 {
            default: 0,
            // permissions: 0o644,
            description: "On a fault, read back the GPU's own UMA pool-base registers from the SGX window (LOG.md 32.41.84)",
        },
        unit_cmd_probe: u32 {
            default: 0,
            // permissions: 0o644,
            description: "On a fault, emit the firmware's own service command for each outstanding 0xc020 unit request and report whether the unit leaves state 3; 1 = write 0 to 0x4008 (fw:0x1899c), 2 = write 1 (fw:0x2564c) (LOG.md 32.41.291)",
        },
        uma_pool_probe: u64 {
            default: 0,
            // permissions: 0o644,
            description: "Poison the three UMA pool GPU-address fields the driver leaves zero (0x34, 0x48, 0x50) with probe|0x10000, |0x20000, |0x30000 (LOG.md 32.41.82)",
        },
        uma_list_probe: u64 {
            default: 0,
            // permissions: 0o644,
            description: "Fill the UMA aperture with self-describing 8-byte words so a fault names the free-list entry format (LOG.md 32.41.80)",
        },
        uma_pool_flush: u32 {
            // 32.41.556: Apple's command carries 1 here (cmd+0x8e6).
            default: 1,
            // permissions: 0o644,
            description: "Set the UMA pool flush byte so the firmware invalidates the pool before reading it (LOG.md 32.41.55)",
        },
        usc_max_tgmem: u32 {
            // Bits 5:0 land in MAX_TGMEM; 4 is halGetDefaultUscMaxTgmem(). The
            // bit at 24 is there only to make the value differ from 4, because
            // fw:0xb630 skips the write when the field already reads exactly 4
            // -- it shifts up to bit 32, clear of the six-bit field.
            // LOG.md 32.41.74.
            // 32.41.551: Apple passes exactly 4. 0x1000004 also leaked bits 40/46 into
            // SGX 0x14898 via fw:0x8e48; with 4 there is no MAX_TGMEM warning here either.
            default: 4,
            // permissions: 0o644,
            description: "USC MAX_TGMEM at RuntimePointers+0x2e0; bits 5:0 are the value, and the firmware ignores the field entirely if it reads exactly 4 (LOG.md 32.41.73/.74)",
        },
        uma_pool_ctl: u32 {
            default: 1,
            // permissions: 0o644,
            description: "Publish the UMA page pool descriptor table addresses in RuntimePointers+0x2d0/0x2d8; 0 leaves the pair null as before (LOG.md 32.41.72)",
        },
        cs_walk_fix: u32 {
            default: 0,
            // permissions: 0o644,
            description: "G15S: walk this many dwords of the VDM control stream by record size and replace every bare 0x00000000 found at a HEADER position with a Barrier. A zero dword reads as an 8-byte PPP State (hi=0,size=0 -- a no-op, so Mesa means it as 4 bytes of padding), which puts every later record 4 bytes out of phase and causes the CSD runaway. A rule, replacing the fixed-offset cs_poke2 workaround. 0 = off (LOG.md 32.41.389)",
        },
        gpu_poke_va: u64 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: absolute GPU VA to overwrite (quadword) in the job's VM slot before the kick, for objects the cs-relative pokes cannot reach such as the USC word list; 0 disables (LOG.md 32.41.395)",
        },
        gpu_poke_val: u64 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: value written by gpu_poke_va (LOG.md 32.41.395)",
        },
        cs_poke_off: u32 {
            default: 0xffffffff,
            description: "DIAGNOSTIC: byte offset into the VDM control stream to overwrite before submitting, or 0xffffffff for off. With cs_poke_val=0xc0000000fff88010 and cs_poke_off=8 the draw stream is truncated to exactly the clear stream, which does not make the VDM run away (LOG.md 32.41.139/140)",
        },
        cs_poke_val: u64 {
            default: 0xc0000000fff88010,
            description: "DIAGNOSTIC: the 64-bit word cs_poke_off writes (LOG.md 32.41.140)",
        },
        vdm_cs_unk: u32 {
            default: 0xffffffff,
            description: "DIAGNOSTIC: value for SGX register 0x1c898, the companion of VDM_CTRL_STREAM_BASE, or 0xffffffff to keep the firmware-derived default (1 on 15.5). Derived from disassembly rather than measured; if it selects the control-stream format, a wrong value mis-parses every record (LOG.md 32.41.168)",
        },
        probe_delay_ms: u32 {
            default: 0,
            description: "DIAGNOSTIC: milliseconds to wait after the kick before dumping the scene's TVB tilemap from inside the submit path; 0 disables. A non-zero tilemap is direct evidence the vertex stage executed (LOG.md 32.41.176)",
        },
        probe_ptr: u32 {
            default: 0,
            description: "DIAGNOSTIC: low 32 bits of an address the VDM control stream points at (e.g. 0xfff88b44); the full VA is rebuilt from the control stream's high half and dumped. 0 disables (LOG.md 32.41.180)",
        },
        probe_va: u64 {
            default: 0,
            description: "DIAGNOSTIC: absolute GPU VA to dump after building the job, with its -2^36 and +2^36 neighbours; 0 disables (LOG.md 32.41.183)",
        },
        cs_fill_off: u32 {
            default: 0xffffffff,
            description: "DIAGNOSTIC: byte offset into the VDM control stream at which to write cs_fill_count copies of cs_fill_val (32-bit), or 0xffffffff for off. Sweeps a whole range of candidate record boundaries in one submission instead of one per boot (LOG.md 32.41.163)",
        },
        cs_fill_val: u32 {
            default: 0xc0000000,
            description: "DIAGNOSTIC: the 32-bit word cs_fill_off repeats; 0xc0000000 is the VDM stream terminator",
        },
        cs_fill_count: u32 {
            default: 0,
            description: "DIAGNOSTIC: how many 32-bit words cs_fill_off writes",
        },
        frag_store_only: u32 {
            default: 0,
            // permissions: 0o644,
            description: "G15S: program the fragment pass the way macOS programs a CLEAR-ONLY pass instead of the way Mesa does -- no background program (0x15369/0x15371/0x15379 = 0), no PROCESS_EMPTY_TILES (0x10039 &= ~0x10000), and 0x15049/0x10051/0x15309/0x15349 = 0, leaving only the EOT program. Every value is macOS-captured at matched 64x64 geometry (capture2/priority0/registers/clear-64x64.txt). Mesa encodes a clear the way Apple encodes a DRAW, so this is the only genuinely reduced fragment job available: the ISP has nothing to fetch. It will NOT produce the right clear colour -- the signal is whether the fragment stage COMPLETES (stamp moves, doneptr advances). 0 = off (LOG.md 32.41.410)",
        },
        frag_keep_empty_tiles: u32 {
            default: 0,
            // permissions: 0o644,
            description: "With frag_store_only=1, KEEP PROCESS_EMPTY_TILES set (0x10039 bit 16) instead of clearing it to match macOS. Splits 32.41.412's result: frag_store_only removed both the background program AND empty-tile processing, and a clear has no geometry, so clearing the latter alone leaves the ISP with zero tiles and it idles having done nothing. With this set, only the background program is gone and the ISP still has all tiles to process -- so an idle unit means the background program really was the wedge (LOG.md 32.41.415)",
        },
        vtx_pre_wfi: u32 {
            default: 0,
            // permissions: 0o644,
            description: "G15S/15.5: emit a bare WaitForIdle (0x00000001) before FinalizeVertex in the vertex microsequence. This driver emits NO TA wait there on 15.5 -- both existing gates (G < G14X, and G >= G14X && V < V15_5) exclude G15S, and both gates are ours. The kext has AGXTAChannelG15::encodeTAFinishWFI which writes exactly this word. The TA parks at 0xc020 = 0x1b even with an empty control stream, so the cause is in setup, and a missing finish-wait is setup. 1 = before FinalizeVertex (hangs 5/5, 32.41.423); 2 = AFTER it, immediately before the 0x40000002 terminal seal, which matches the kext ordering (encodeTAFinishWFI finishes the TA, then AGXSKUEncoder::finishEncode appends the seal) and is the better-motivated placement; 0 = off (LOG.md 32.41.425)",
        },
        vtx_apple_useq: u32 {
            default: 1,
            description: "G15S/15.5: emit the TA microsequence in the shape Apple ACTUALLY emits, measured from macOS's own encoder (capture4 out/raw-ta-microseq-draw.txt, AGXSKUEncoder::finishEncode #2, inside encodeTACommandSKUStream). Apple's sequence is StartVertex(0x1cc) + Timestamp(0x3c, flag=1) + WaitForIdle(0x4) + Timestamp(0x3c, flag=0) + FinalizeVertex(0x94) + WaitForIdle2(0x4) = 0x2e0. This driver emits StartVertex + FinalizeVertex + WaitForIdle2 = 0x264 (32.41.445) because BOTH Timestamp ops are gated on vtx_user_timestamps.any(), which Mesa never sets, and the WaitForIdle is gated off entirely on 15.5. So we are missing 0x7c bytes = three ops. The record decode is pinned by static assertions in fw/microseq.rs against Apple's bytes: size 0x3c and all seven field offsets, and the two records differ in exactly one field (update_ts, by 8 = start_addr vs end_addr). 1 = emit all three unconditionally. 2 = the two Timestamps but NO WaitForIdle -- the remaining cell of the 2x2 with 32.41.423 (WFI alone, hard-hung the machine 5/5) and 32.41.502 (all three, soft stall: stamp[0] 0x100 -> 0x0, no fault). 3 = the full shape with Apple's NULL user_ts_pointers (Option<GpuWeakPointer> uses the NonZeroU64 null niche, so the record layout is unchanged and the static assertions prove it). 0 = off. This supersedes vtx_pre_wfi, whose arm 1 put the WaitForIdle before FinalizeVertex with NO surrounding Timestamps and hung 5/5 (32.41.423) -- the placement was right, the two 0x3c records around it were missing. LOG.md 32.41.500",
        },
        cs_reshape: u32 {
            default: 0,
            description: "G15S: rewrite Mesa's VDM State Update into the 28-byte shape Apple emits (header + six payload words, structural zeros at rel +0x0c/+0x18) and shift this many dwords from cs+0x1c up by 3 so the following records land on cs+0x28. Mesa writes 20 bytes where G15 consumes 28, so it resumes on a raw USC address and runs away (32.41.450). Derived from AGXMetalG15X_M1, six agreeing sites. \u{26a0} REQUIRES probe_usc=1: the block is lexically inside `if probe_usc != 0`, so with probe_usc at its default this parameter is accepted and does NOTHING -- the same dead-code trap cs_draw_ctrl_fix sat in for two runs (32.41.252/.253). It is left there because the stream dump it is verified against sits in the same block and must run after it. Also requires the standing baseline user_va_shift=36 usc_absolute=1 barrier_armed=1 (32.41.224/.246). 0 = off. LOG.md 32.41.451, .452, .461",
        },
        cs_wd: u32 {
            default: 0,
            description: "G15S: payload word 5 (\"wD\") of the VDM State Update, at cs+0x20. ⚠️ Default changed 0x30 -> 0 on 32.41.464: capture4 shows Apple writes 0x00000000 here for a plain user draw, in three runs and six workload variants. 0x30 was DERIVED from `orr ..., #0x30` being unconditional at two sites in AGXMetalG15X_M1, which narrowed it to 0x30/0x34/0x70/0x74 and observed none of them. 0x30 is a real value -- it appears in the prebuilt 28-byte templates in AGXMetal's heap (alongside 0x00/0x31/0x01) and Apple's own internal clear-pass draws carry 0x34 -- but it is not what a user draw gets. LOG.md 32.41.452, .464",
        },
        cs_shift_from: u32 {
            default: 0x24,
            description: "G15S: byte offset in the VDM control stream where cs_reshape starts moving the tail. Default 0x24: Mesa's mask 0x2e has FOUR bits set so Mesa writes four payload words (cs+0x10..0x1f) plus a 4-byte zero pad at cs+0x20, ending at cs+0x24, while Apple's 28-byte record ends at cs+0x28. ⛔ Was 0x1c/3 dwords, which pushed Mesa's 4th payload word (0x00000002) out to cs+0x28 where the decoder read it as a PPP State header -- size 0, addr 0x200000000, aliased to 0x1200000000 -- and faulted at 0x1200000080. LOG.md 32.41.466",
        },
        cs_shift_by: u32 {
            default: 1,
            description: "G15S: how many dwords cs_reshape moves the tail up by, from cs_shift_from. Default 1 (4 bytes): 0x24 + 4 = 0x28, where Apple's PPP chain begins. See cs_shift_from. LOG.md 32.41.466",
        },
        cs_w1: u32 {
            default: 0,
            description: "G15S: payload word 1 of the VDM State Update, at cs+0x10. Apple writes 0 (capture4, every variant) where Mesa writes 0x00001212. Mesa's value is a G13-ism describing uniforms/texture-state/samplers/preshader; on G15 those bindings are NOT in the control stream at all -- they live in the per-pipeline blob that payload word 3 points at, which is why 0x1212 has no slot here. Set to 0x1212 to put Mesa's word back for a bisect. LOG.md 32.41.464",
        },
        cs_w3_raw: u32 {
            default: 0,
            description: "G15S: how payload word 3 (the pipeline pointer) is formed. 0 = (usc_exec_base + Mesa's word) >> 6, an absolute VA scaled. 1 = Mesa's word VERBATIM (refuted as the base by 32.41.465: changing it did not move the fault). 2 = Mesa's word >> 6, the USC-RELATIVE offset scaled -- the derived answer. seg-1 measured the hardware rule as addr = CODE_GART_BASE + ((w3 & 0xffffffff) << 6) with CODE_GART_BASE = 0x10000000000 (capture2 FINDING 3 / gart_ranges[5]), and it is not any of the five USC segment bases. Compensating inside w3 is impossible -- the needed value 0xc47ffda00 does not fit 32 bits -- so the shader must be mapped inside the window. Mode 2 MUST be paired with user_va_win36=1 win36_base=0x10000000000 and a window over the USC heap, or w3 points at nothing mapped. LOG.md 32.41.464, .465, .467",
        },
        cs_w2: u32 {
            default: 0x01000000,
            description: "G15S: payload word 2 of the VDM State Update, at cs+0x14. Apple writes 0x01000000 in every captured variant (draw, uniform, varying, vbuf, tex, depth, tri2, inst2); its own internal clear-pass draws write 0x01000040. Mesa puts the PIPELINE POINTER in this slot, which is the defect 32.41.464 fixes -- the pipeline belongs at word 3. LOG.md 32.41.464",
        },
        fault_addr_bits: u32 {
            default: 42,
            description: "G15S: width of the address field in FAULT_ADDR_G14X before the <<6 scale. 42 is Apple's, read out of AGXAcceleratorG15::getMMUFaultInfo as a single instruction -- `ubfiz x8, x8, #6, #42`. Our raw values are 37 bits, so the mask is a no-op and the decode equals what the driver always did. This REFUTES 32.41.458, which guessed 32 from an 'outside the IAS' argument: ubfiz #6,#42 produces up to 48 bits by construction, so Apple allows a fault address above the 42-bit input range on purpose. Left as an override only. 0 or >64 = 42. LOG.md 32.41.460",
        },
        fault_win_sweep: u32 {
            default: 0,
            description: "G15S: in the fault path, sweep the 0xd800 selector over all 64 windows and dump every 0xd8c0/0xd8c8 pair that reports a valid fault. Derived from AGXAcceleratorG15::getFirstMMUFaultStatus, which writes a selector to 0xd800 BEFORE reading 0xd8c0/0xd8c8 -- so those two registers are a window, like 0xc020 is windowed by 0x10398. Apple builds the selector from accel->0x17bd8, which fffffe00090d6e7c copies out of a firmware-published structure (fwstruct+0x4ec0), i.e. the firmware names which unit's fault registers to read. This driver has never written 0xd800, so every fault address recorded in this phase came from whichever slot was latched. Saves and restores the selector. 0 = off. LOG.md 32.41.460",
        },
        pm_sentinel: u32 {
            default: 0,
            description: "G15S: preload the control_word (Scene+0x08 target) with this value before the job. fw:0x15f20 in FinalizeVertex copies *(scene+8) into pass_page_count and then clears it, so pass_page_count is a COPY of that word. If pass_pages comes back == this sentinel, the firmware read OUR word and pass_pages 0 genuinely meant nothing was binned; if it comes back 0, the firmware is reading a word we never wrote and every pass_pages reading in this phase is void. 0 = off. LOG.md 32.41.472",
        },
        dump_pipeline: u32 {
            default: 0,
            description: "G15S: dump this many dwords of the USC pipeline blob that VDM State Update payload word 3 points at (address = w3 << 6). capture4 captured Apple's equivalent for the same 64x64 one-triangle draw -- 0x50 bytes, growing to 0xa0/0xb0/0xc0 with a uniform / vertex buffer / texture bound -- and identified it as a prologue shader carrying the bindings. It is what the USC actually consumes and ours has never been looked at. 0 = off. LOG.md 32.41.474",
        },
        cs_w3_add: u32 {
            default: 0,
            description: "G15S: add this to the computed VDM State Update payload word 3 (the USC pipeline pointer) so the address the hardware forms, CODE_GART_BASE + (w3 << 6), lands in UNMAPPED space. Decides the question 32.41.476 left open: that entry proved the blob's CONTENTS are irrelevant (a G13 shader, 0xff filler and redirected record addresses all changed nothing), but not whether the USC reads the object at all in the win36 config -- 32.41.467's dereference was measured with win36 OFF. If a bump produces a fault at CODE_GART_BASE + ((w3+N) << 6) the USC is reading and simply ignoring what it finds; if it produces no fault, the USC never reads the object in this configuration and the whole preamble/pipeline line is the wrong target. 0 = off. LOG.md 32.41.479",
        },
        cs_w4: u32 {
            default: 0,
            description: "G15S: override VDM State Update payload word 4, the vertex-outputs word, instead of passing Mesa's through. capture4 shows Apple emits 0x404 for the same 64x64 one-triangle draw (and 0x808 once a varying is added, so the field is (X<<8)|X and scales with the outputs), where our reshape passes Mesa's 0x505. The difference has never been tested, and it is the cheapest untested delta left after 32.41.477 closed the PPP-state question. 0 = use Mesa's value. LOG.md 32.41.478",
        },
        usc_code_poison: u32 {
            default: 0,
            description: "G15S: with g13_compat_test=9, overwrite the `Code` field of the USC Shader record (tag 0x0d, u32 at byte 2, genxml `USC Shader`) inside the word-3 pipeline object with this USC-RELATIVE offset, chosen to be unmapped. .521 proved the VDM executes the draw record and .479/.492 proved the USC fetches the pipeline object, yet .520 showed an Ioto ESL program planted there is never executed -- so the reading that fits is that the G15 USC consumes the G13-style USC word list Mesa already writes, and the ESL program is something Apple's stack uses that the hardware does not require. If the USC loads the shader through the word list, poisoning Code produces a fault naming the address (hw VA = win36_base + Code); if it does not fault, the word list does not feed the shader either and both candidate paths are eliminated. Nothing is zeroed and nothing is planted -- only four bytes change, so this is the plain baseline plus one field. Suggested 0xdead0000. 0 = off. LOG.md 32.41.522",
        },
        comp_nopreempt: u32 {
            default: 0,
            description: "Set JobMeta.no_preemption on compute jobs (tests slow-vs-deadlocked USC). LOG.md 32.41.575",
        },
        creplay_skip0: u32 {
            default: 1,
            description: "comp_apple=2: do not map Apple's buffers below the user VA base (VA 0 -> 1<<36 collides with a live mapping). LOG.md 32.41.574",
        },
        creplay_patch: u32 {
            default: 0,
            description: "comp_apple=2: patch Apple's CDM stream before mapping. 1 = replace the 0x60000160 record at +0x28 with the terminator 0x40000000 (drop the barrier). 2 = poison launch word 0 (descriptor 0x10000080000 -> unmapped 0x100000f0000). 4 = pre-seed Apple post-run words at 0x10000080000+0x20. 5 = empty shader (terminator at 0x100000006c0). 6 = empty ESL program (terminator at 0x10000090000). LOG.md 32.41.574",
        },
        cdm_107a1: u64 {
            default: 0,
            description: "G15S comp_apple: override register 0x107a1 (Apple 0xff0000; if it is a USC pipe mask that is pipes 16..23 = cluster 1 only). 0 = Apple's value. LOG.md 32.41.573",
        },
        fw_usc_dump: u32 {
            default: 0,
            description: "G15S: at the compute doorbell set IDB0 word 0 = 1. The firmware's periodic path (after ktrace 0x106, fw:0x35a00) then prints its own 'USC Pipe %u State Overview' (VDM/PDM/CDM SERV + DEBUG_STATUS for every pipe in its mask) to FWLog and clears the word. LOG.md 32.41.573",
        },
        comp_apple: u32 {
            default: 1,
            description: "G15S compute in Apple's shape (macOS capture 2026-09-27): 1 = opcodes 0xb/0xc with Apple's op layouts, Apple's Timestamp/WaitForIdle/Timestamp sequence, Apple's kext register list, a spill buffer and UMA scratch, Mesa's CDM stream. 2 = as 1 but replay Apple's own compute job (its buffers mapped at Apple's VAs, CDM at Apple's stream) and read back its output. LOG.md 32.41.572",
        },
        usc_poll: u32 {
            default: 0,
            description: "G15S: after the vertex doorbell, wait for power-up then spin N iterations reading USC VDM_SERV/DEBUG_STATUS per pipe. LOG.md 32.41.560. 667zzy: default 0 -- it ran on EVERY compute kick (~40 ms spin + fender gating + slot sweeps) and stalled the GPU under Vulkan compute",
        },
        idle_off_ms: u32 {
            default: 600000,
            description: "Override idle_off_delay_ms and fender_idle_off_delay_ms (0 = ADT/default). LOG.md 32.41.559",
        },
        dump_dbgwin: u32 {
            default: 0,
            description: "G15S: on the Nth hung dump (2 = after the job), read unit registers through the firmware's indirect debug window at SGX 0x4090 (unused by the firmware): known-answer UMA 0xa5a8 == 0x20 first, then USC 0xa040/SERV/DEBUG_STATUS for units 0x2/0x202/0x102/0x302. LOG.md 32.41.558",
        },
        dump_usc: u32 {
            default: 0,
            description: "G15S: at the post-kick hung dump, read the firmware's own USC diagnostic registers per pipe (AGX_CR_USC_{VDM,PDM,CDM}_SERV 0xa088/0xa068/0xa0a8 via selector 0xa010, *_DEBUG_STATUS 0xa090/0xa070/0xa0b0 via selector 0xa000), exactly as fw:0x35a24 does; selectors restored. Pipes 0..total_active_cores. LOG.md 32.41.533",
        },
        uma_apple: u32 {
            default: 1,
            description: "G15S: give StartVertex Apple's UMA pool shape. capture4's reused TA buffer holds what Apple's FIRMWARE appended after an identical job: 0x14308 = 0x5 (fw:0x24f78 no-base path: pool id 5, StartVertex+0x198 = 0) and 0x14318 = 0x420004a0001 (StartVertex+0x1a8 = fw-private VA 0xfffffc20004a0000, confirmed in Apple's captured StartVertex). Ours: 0x38001980100 / 0x38001d8001 (user-VM aperture + scratch). 1 = pool id 5, +0x198 = 0, +0x1a8 = a zeroed 1 MiB kernel/firmware-visible buffer. LOG.md 32.41.536",
        },
        uma_real: u32 {
            default: 32,
            description: "G15S: back the UMA pool with real memory, as AGXUMAFList::prepareBufferResources does: a 4 MiB block in the job VM, pool[0x14] = page list (u64 VA per 4 KiB page), pool[0x34] = extent list ((va>>12)<<12 | pages<<52), pool[0x44] = 1, pool[0x1c]/[0x2c] = 1024 pages. Until now the pool only ever carried a base and a zeroed list. Use with uma_apple=1 (id 5, no-base register form). LOG.md 32.41.540",
        },
        frag_1ca28: u32 {
            default: 1,
            description: "G15: add 0x1ca28 = user_buffer (TA 0x1ca30 form) to the fragment list, as Apple's kext does; the firmware only appends it when StartFragment op+0x8c != 0, which is 0 for Apple too. LOG.md 32.41.602",
        },
        frag_tmap_poison: u64 {
            default: 0,
            description: "DIAGNOSTIC: fragment tilemap pointer (kbuf[0], appended by the firmware as 0x16429) = this VA. LOG.md 32.41.619",
        },
        buf_slot_min: u32 {
            default: 0,
            description: "Lowest TVB buffer-manager slot handed out (Apple's jobs use 0xd; slot 0 may be a firmware sentinel). LOG.md 32.41.630",
        },
        scene_cached: u32 {
            default: 0,
            description: "DIAGNOSTIC: allocate the scene tilemap + heap meta from the GPU/FW-private (cached memattr) allocator. LOG.md 32.41.629",
        },
        barrier_s2: u32 {
            default: 0,
            description: "15.5 Barrier wait_stamp2 = the vertex fw_stamp (not the CPU-visible stamp, which reaches the value before FinalizeVertex and releases the fragment early). 2 = unreachable value (gating test). 3 = FinalizeVertex stamps barrier+0x38 and the barrier waits on it. LOG.md 32.41.626-628",
        },
        replay_vp: u32 {
            default: 0,
            description: "DIAGNOSTIC (apple_replay): 1 = patch Apple's PPP viewport to 16,16,16,-16 (triangle in the top-left quarter). LOG.md 32.41.619",
        },
        layermeta_at0: u32 {
            default: 0,
            description: "G15: TVB layer metadata pointer = heap metadata base (Apple's 0x16060 == 0x16098), not +0x200. LOG.md 32.41.615",
        },
        tvb_cpu: u32 {
            default: 1,
            description: "DIAGNOSTIC: allocate TVB blocks CPU-mapped and dump the tile-list pages at render completion. 2 = same, from the GPU/FW-private (cached memattr) allocator. LOG.md 32.41.614/623",
        },
        poison_fs: u32 {
            default: 0,
            description: "DIAGNOSTIC (apple_replay): retarget the triangle fragment ESL program's LoadShader (0x10000138640) to unmapped 0xbeef000000. LOG.md 32.41.614",
        },
        replay_post: u32 {
            default: 0,
            description: "With apple_replay: upload Apple's POST-draw 0x10000118000 (fragment/EOT image descriptors at +0x600..+0x8e8, absent from the pre blob). LOG.md 32.41.611",
        },
        barrier_dag: u32 {
            default: 1,
            description: "15.5 Barrier +0x24 = the fragment's stamp value (the DAG sequence the barrier assigns, fw:0x19308) instead of vtx_slot<<8. Pair with dag_from_stamp=2. LOG.md 32.41.610",
        },
        replay_va0: u32 {
            default: 0,
            description: "DIAGNOSTIC: also map anything placed in [2^36, 2^36+64KiB) at VA - 2^36, so Apple's replayed VA-0 buffer exists at page 0 as well. LOG.md 32.41.605",
        },
        frag_apple_regs: u32 {
            default: 0,
            description: "With apple_replay: set the fragment pipeline/buffer registers (0x15369/71/79/81, 0x15101/09, 0x16090/0x16461, 0x15021, 0x15301) to Apple's draw-triangle values, which point into the replayed Apple buffers. Mesa's pipeline pointers made the USC fetch through VA 0. LOG.md 32.41.602. 2 = BG unbound. 3 = 0x15369 rebased by 2^36 onto the replayed VA-0 buffer (32.41.604)",
        },
        info_155: u32 {
            default: 15,
            description: "15.5 buffer::Info fixes from the PB-descriptor code (fw:0x28d40/0x29068): +0x24 = page-list GPU address (PB descriptor word 0), +0x54 = block size in bytes (0 made the descriptor's page count 0 -> TA out of memory, event bit 4). Bitmask: 1 = those two fields, 2 = Info+0x18 = 1 (fw:0x1d100 builds the PB descriptor only if it is non-zero). 4 = page/block lists in kernel GPU+FW shared memory (the firmware CPU faults on the job-VM copies). 8 = Counter pointer at Info+0x60 (fragment completion fw:0x1e6fc derefs it; LOG.md 32.41.606). LOG.md 32.41.600",
        },
        vtx_qid_zero: u32 {
            default: 1,
            description: "15.5: StartVertex+0x4c = 0 like Apple (was the queue id). The firmware appends 0x10209/0x1c9f0/0x14320 = vm_slot << 8 | (op+0x4c & 0x3f); Apple's draw has 0xb00, ours 0x103. LOG.md 32.41.600",
        },
        pool_apple: u32 {
            default: 2,
            description: "G15S: with uma_real, make the UMA pool record match Apple's live one: [0]=0xb [0xc]=1 [0x10]=1 (firmware queue select), [0x1c]=0x100000 pages/block, [0x24]=total pages, extent count 8 per block. LOG.md 32.41.548. 2 = only the free-page ring: [0x1c] capacity 0x100000, [0x24] tail = pages (with capacity 1024 the tail wrapped to 0 == head, an empty ring), 8 extents per block; no [0xc]. LOG.md 32.41.599",
        },
        glob_pcheck: u32 {
            default: 2,
            description: "15.5: the progress-check intervals (timeslice quanta) at Globals+0x98c/0x990/0x994, where the 15.5 firmware reads them (fw:0x2e08). 1 = Apple's 3D 40 / TA 10 / CL 250 ms; 2 = never reached (no timeslice GTP pauses: after one trips GIN-1072 every later pause ends in a GPU timeout, LOG.md .667zzz36). With 0 there every pipe is preempted every 1 ms tick and long compute jobs hang. 0 = leave Globals as is (LOG.md 32.41.667zzy)",
        },
        glob_apple: u32 {
            default: 1,
            description: "G15S: overwrite Globals (InitData+0x20) bytes 0..0x8000 with Apple's live copy from this Mac (the 0x9800 object the macOS capture labelled HwDataA; fw:0x2d48 reads it as globals+0x56). 15.5 packs it from 0x50; ours is 13.5's layout. LOG.md 32.41.547",
        },
        agx_tun: u32 {
            default: 0,
            description: "Compute usc_poll path, while the job runs: 1 = log xnu's AGX tunable registers at SGX+0x1000000 (m1n1 tunables_static.c offsets), 2 = also apply m1n1's t600x values. LOG.md 32.41.589",
        },
        cattach: u32 {
            default: 1,
            description: "comp_apple: give op 0xb Apple's single attachment {0x10005c08000, 0x8400, 0x18, 1} (handler 0xb -> SGX 0xc238 range records). LOG.md 32.41.588",
        },
        ta_poll_wide: u32 {
            default: 0,
            description: "ta_poll: sample the whole 0x10a00 block + UMA regs (32.41.599)",
        },
        ta_poll: u32 {
            default: 0,
            description: "After a vertex kick, sample the TA event block (0x10a10..) N times, log changes (32.41.598)",
        },
        dag_from_stamp: u32 {
            default: 2,
            description: "Keep stamp_value as the DAG word instead of overwriting it with the queue counter. 1 = RunCompute (32.41.597); 2 = also RunVertex/RunFragment/RunBlitter (fragment stamp was overwritten with 0, 32.41.609)",
        },
        buf_addr_shift: u32 {
            default: 0,
            description: "DIAGNOSTIC: leak one small allocation from the private, shared and gpu kernel heaps before building each TVB Buffer, so a new Buffer never reuses the freed one's firmware-visible addresses (Info, block_ctl, counter, page/block lists). Tests whether the firmware caches per-buffer state by address (2nd glmark2 scene: TA out-of-memory on its first kick). LOG.md 32.41.664",
        },
        cpool_real: u32 {
            default: 32,
            description: "comp_apple: back compute UMA pool 4 with N MiB of real pages (list/extents/record in Apple's layout) and push them with DeviceControl 0x1e before the job. LOG.md 32.41.586",
        },
        min_sram_uv: u32 {
            default: 0,
            description: "Override apple,min-sram-microvolt (dtsi guess 790000). The live macOS ADT sgx perf-states-sram floors the SRAM rail at 810 mV for perf states 0-5. LOG.md 32.41.594",
        },
        user_prot: u32 {
            default: 0,
            description: "G15: map user/replay buffers with Apple's G15 user-RW PTE encoding (AP=0 + bits 53/54, uncached = PROT_GPU_FW_SHARED_RW) instead of the G13 AP=2 GPU-only encoding. AGXUnifiedAddressTranslator::encodePTEFlags' permission table (kext 0xfffffe00071166f4) never produces AP=2. LOG.md 32.41.593",
        },
        glob_poke_off: u32 {
            default: 0,
            description: "Byte offset into Globals (InitData+0x20) to overwrite with glob_poke_val at init (0 = off). Globals+0x9b0 = firmware early-wake timeout (kext AGXArmFirmware::setFWEarlyWakeTimeout writes it; fw:0x2df4 -> 0x1a38e0; Apple 5). LOG.md 32.41.667zu",
        },
        glob_poke_val: u32 {
            default: 0,
            description: "Value for glob_poke_off. LOG.md 32.41.667zu",
        },
        pm_period_ms: u32 {
            default: 1,
            description: "Override the ADT power sample period (ms; 0 = ADT, 8 on t6030; default 1). The PM tick is what wakes an idle-parked GPU, so it bounds per-submission latency. LOG.md 32.41.667zz",
        },
        pm_period_clks: u32 {
            default: 2500,
            description: "Override the PM tick in AIC clocks (HwDataA clocks_per_period; ADT 200000 = 8 ms on t6030; 0 = derived from pm_period_ms/ADT; default 2500 = 0.1 ms: 800x600 frame 8.4 -> 0.9 ms, glmark2 suite 1879). LOG.md 32.41.667zz",
        },
        dc_0a: u32 {
            default: 0,
            description: "If non-zero, send DeviceControl 0x0a {val} after Initialize. fw:0x33670 stores the value in [0x1a3864], one of the gates of PM case 0x24 (kick-with-wake doorbell 0x86, fw:0x38048); 0 calls the power-off path fw:0x30634. LOG.md 32.41.667zw",
        },
        ktrace_hist: u32 {
            default: 0,
            description: "Keep the last 512 firmware trace records in memory (no printing) and print them once on the first job timeout or teardown cancel; 2 also prints the last 80 on every GPU fault; 3 keeps only classes 0x0xx/0x6xx and prints each compute job's records at completion; 4 = the filter of 3, printed only on the first timeout. LOG.md 32.41.667zs/667zzz/667zzz41",
        },
        cpool_wait_ms: u32 {
            default: 20,
            description: "Wait up to N ms after a compute pool DC 0x1e until the firmware retires every page-list chunk (event 13) before queueing the job; 0 = no wait. A preemption store before the fill completes stores USC context into page 0. LOG.md 32.41.667zzz",
        },
        spill_scale: u32 {
            default: 1,
            description: "Scale the compute spill regions 0/1 (Apple's capture: 0x1e400 each, sized by the kext for a 64-thread kernel). The CDM stores preempted USC state there. LOG.md 32.41.667zzy",
        },
        comp_busy: u32 {
            default: 0,
            description: "Write this non-idle state into pipe0 slot 63 while compute jobs are in flight (0 = off). Diagnostic: the per-PM-tick quiesce context-stores a running CDM job; its resume hung until JobParameters2 got Apple's 15.5 layout (resume CDM control = JobParameters2+0x58). The power-up store still happens with the slot busy. LOG.md 32.41.667zzy",
        },
        busy_slot63: u32 {
            default: 0,
            description: "Write this non-idle state into pipe0 slot 63 (stats.vtx + 0xc + 63*24) so the firmware's power-up slot scan (fw:0x35e50) never finds all slots idle and stops the pipes under a launching StartFragment. 0 = off. LOG.md 32.41.667zm",
        },
        user_memattr: u32 {
            default: 0xff,
            description: "Memory attribute index for user GPU mappings (0 cached, 1 device, 2 uncached; other = keep the G13 uncached default). Fragment fill measured ~500 MB/s with uncached user memory. LOG.md 32.41.667zk",
        },
        fender_nogate: u32 {
            default: 0,
            description: "Compute usc_poll path: set GFX 0xd06030 bit 2 right after the kick (= the kext's AGXArmFenderChinook::disableDynamicFenderGating). The value is logged either way. LOG.md 32.41.584",
        },
        verbose: u32 {
            default: 0,
            description: "Bring-up diagnostics in the kernel log (per-job CDM stream dump + probes, comp_apple lines, TVB/UMA pool records, firmware stats samples). 0 = quiet (.667zzz9)",
        },
        ts_dbg: u32 {
            default: 0,
            description: "Timestamp instrument (LOG.md 32.41.667zzz9): 1 = positive control (HwDataB+0x28 rebased onto the kernel shared heap, user start stamps go to a kernel-shared word), 2 = print only (per-job kernel/user stamps, bind-time PTE vs BO page)",
        },
        hwa_apple: u32 {
            default: 1,
            description: "G15S: overwrite HwDataA with Apple's 15.5-layout HwDataA (hw/t6030-hwdataa-apple.bin, Apple's initPowerAndPerformanceData emulated on this Mac's captured objects). Ours is 13.x's layout: the firmware's DVFS clamps read 0 and pin the GPU at 338 MHz. 1 = keep our PM tick (+0x4/+0x8), 2 = Apple's tick too. LOG.md 32.41.667zzz25",
        },
        hwb_apple: u32 {
            default: 1,
            description: "G15S: overwrite HwDataB with Apple's live HwDataB from this Mac (macOS 15.5, 2026-09-26 capture, hw/t6030-hwdatab-apple.bin), keeping our timestamp VA (+0x28), io_mapping firmware VAs, sgx_sram_ptr and uat_ttb_base. 515 static words differed, incl. the io_mapping entry layout (+0x10 u32 total, +0x14 u32 elem, +0x18 SGX offset, +0x20 flag 2/0), 0xef4=1, the 0x1c74.. tail. LOG.md 32.41.546",
        },
        uma_dc: u32 {
            default: 1,
            description: "G15S: with uma_real, send DeviceControl 0x1e (UMA add pages, fw:0x32ec0 -> 0x246b8 -> SGX 0x14020/0x14018) at submit, as AGXUMAFList::updateSubmitInfo does. Without it the hardware page manager is never given the pool's pages. 1 = page-list context = the job's VM slot; N>=2 = context N-2 (Apple uses 63). LOG.md 32.41.545",
        },
        uma_list_poison: u32 {
            default: 0,
            description: "Instrument check: with uma_dc, send DC 0x1e with page list 0x5f00000000 (unmapped). A fault proves the page manager reads the list. LOG.md 32.41.552",
        },
        kctx63: u32 {
            default: 1,
            description: "G15S: mirror the kernel context (TTBAT slot 0) into slot 63, the context macOS tags its hardware-read kernel tables with ([accel+0x6c0] = 63, DC 0x1e ctx 0x3f, page list/bufmgr at 2^40). Slot 63 is never handed to users. LOG.md 32.41.550",
        },
        pipe_stats: u32 {
            default: 1,
            description: "G15S: RegionB+0x244/0x24c/0x254 (pipe_tables[0..2]) = the global stats objects the jobs already point at (stats.vtx / stats.frag / stats.comp). The kext's encode{TA,3D,CL}CommandSKUStream put fw_addr(AGXArmFirmware+600/608/616) +4/+8/+0 in the op's stats field, and initFirmwareSharedData puts the same three objects at RegionB+0x244.. -- one object per pipe, not two. The bit-4 event handler (fw:0x4800) writes RegionB+0x244 + slot*24 + 12. 0 = separate zeroed IPTB. LOG.md 32.41.543",
        },
        dump_hwdata: u32 {
            default: 0,
            description: "Print the non-zero 16-byte rows of HwDataB (all) and HwDataA (first 0xe00) at init, for a field-by-field diff against the stores macOS's kext makes in AGX{,Arm}Firmware::initFirmwareData (hwstores.txt). LOG.md 32.41.539",
        },
        hwb_core_mask: u32 {
            default: 1,
            description: "G15S: fill HwDataB+0x1c90 (3 x u32) with the packed core masks (CORE_MASKS_G14X). The firmware's power-up paths (fw:0x31244/0x37f44) write it to Bumper+0xa0 and DFenderMGPUCfg+0x1500 only if non-zero; left zero, the core-enable mask is never restored after a power gate. 0 = legacy zero. LOG.md 32.41.538",
        },
        vtx_cluster_sel: u32 {
            default: 1,
            description: "G15S: StartVertex op+0x1c8 (unk_178). fw:0x161ec emits cluster registers from it: 0 -> 0x1c8e0 = 0x1c8e8 = all-ones (ours); non-zero -> 0x1c8e0 = 0xff << (cluster*8), 0x1c8e8 = 0 (Apple, whose captured StartVertex has 1). 0 = legacy (!clustering). LOG.md 32.41.536",
        },
        apple_replay: u32 {
            default: 0,
            description: "G15S: REPLAY Apple's own user-side buffers for one 64x64 clear + triangle draw (macOS 15.5 capture4, res-draw-pre, 14 buffers, embedded in the module). At the first render submission each buffer is mapped at Apple's exact GPU VA in the job's VM (VAs below the user base go to VA + 2^36, the address the VDM forms), and the TA register 0x1c880 (VDM control stream base) is set to Apple's 0x18000. Everything kernel-side -- the TVB, tilemaps, scene, microsequence, the rest of the register list -- stays ours. Bisects the whole phase in one run: if the tiler bins (pm_counter > 0) the defect is in what Mesa generates; if not, it is in the kernel/firmware path. Must run WITHOUT user_va_win36 (its alias lands Mesa's USC heap on 0x100_0000_0000, exactly Apple's shader window). 0 = off. LOG.md 32.41.531",
        },
        cs_link_probe: u64 {
            default: 0,
            description: "G15S: overwrite the Index List record in the VDM control stream with a VDM Stream Link (block type 4) pointing at this GPU VA, to measure whether the VDM ever REACHES the draw. .520 showed the USC never executes the word-3 object -- a byte-verified Ioto program with LoadShader pointing at unmapped memory produced zero faults -- so the blocker is upstream of the USC, and the .517 chain's proximate claim is refuted. .494 proved the VDM fetches the PPP block at cs+0x2c/+0x30, the record immediately before the draw; whether it goes on to execute the Index List at cs+0x38 has never been measured, because our Index List is byte-identical to Apple's (61c40600 00000003 00000001 00000000) and so 'reached it and issued a draw' looks exactly like 'stopped before it'. A Stream Link to unmapped memory separates them: if the VDM executes the record it jumps there and the fault decoder names the address; no fault means the VDM never got that far. The offset is scanned (first record with block type 3), not hardcoded, because cs_reshape moves everything after +0x24. Instrument validity: .377 planted a Stream Link and the VDM followed it; .493/.494 show VDM fetch faults are reported. Suggested value 0xbeef000000 (unmapped, distinctive, inside the 42-bit IAS). 0 = off. LOG.md 32.41.521",
        },
        cs_head_apple: u32 {
            default: 0,
            description: "G15S: reorder the head of the VDM control stream into Apple's order -- VDM State Update FIRST, then the Barrier and the leading PPP state block. Derived from Apple's OWN captured control stream (res-draw-pre/gpu-000000018000.bin, LOG.md 32.41.498): Apple opens at dword 0 with 0x4000002e and emits ZERO PPP blocks before the state update, then nine after it, then the Index List, then the terminator. Mesa opens with a Barrier 0x20000008 and one PPP block, and only then the state update. 32.41.477/.478 compared PPP group COVERAGE and the state-update PAYLOAD and found them equivalent -- neither compared record ORDER, so this difference has never been tested. The permutation is size-preserving (4+8+28 bytes rearranged in place), so nothing downstream shifts and no length has to be recomputed. Requires probe_usc=1 and cs_reshape (the 28-byte state update must already be in place). 0 = off. LOG.md 32.41.498",
        },
        g13_compat_test: u32 {
            default: 0,
            description: "G15S: overwrite the USC pipeline blob that VDM State Update word 3 points at with a hand-assembled G13 shader, to settle whether G15 hardware still accepts G13 instruction encoding. 32.41.475 established that Apple's M3 shader code is NOT G13 ISA (G13 `stop` = 88 00; Apple's shaders end 0e 00 00 00, 31 per 64KiB heap tracking the shader count, while 88 00 appears once as noise; no clean G13 run of >=4 instructions ending in a real stop exists anywhere in 47KiB of Apple code, at any alignment) and that Mesa emits ONLY G13 (AGX_CHIP_ stops at G14X, agx_device.c maps generation>=14 to G14X so the M3 is treated as an M2, libagx builds only g13g/g13x). What that does NOT settle is whether the G15 decoder is backward compatible -- Apple's compiler would emit the new encoding either way. 1 = write the 28-byte G13 sentinel shader (mov_imm r0/r1 = g13_compat_addr, mov_imm r2 = 0xc0debeef, device_store r2 -> [r0_r1], stop) assembled with dougallj/applegpu; the target address is deliberately UNMAPPED, so if the USC executes and decodes G13 code the existing fault decoder prints that exact address. 2 = write 0xff filler instead (invalid opcodes), the control arm that says whether the blob is executed AT ALL -- needed because a null in arm 1 is otherwise ambiguous between 'not executed' and 'not decoded'. 3 = leave Mesa's record chain intact but redirect its address-shaped words 0x2ffff88x -> 0x2beef88x, separating 'not read' from 'read as benign data'. 4 = plant APPLE'S OWN captured 0x50-byte draw ESL state load program at pipe_va+0x200, the offset .479 measured the USC reading. This is a substitution, not a sweep: .481/.482 tiled that object completely as Apple 'Ioto' instructions and established from LoadShader::emit that the address fields it can carry are ZERO in every capture, so the program is self-contained and relocates into our VM with no fixup -- nothing else we could put here has that property. 5 = the same, written at BOTH +0x000 and +0x200, to separate 'wrong offset' from 'not executed' (the ambiguity that made .476's null uninterpretable). 6 = zero pipe_va[0,0x400) first, then plant at +0x000 -- Apple's actual object shape (.495: every byte past the program is zero in all six captures), and +0x000 is the placement (.495 refutes +0x200: Apple has zeros there and Apple renders). 7 = arm 6 with the LoadShader core RETARGETED to our own vertex shader (.518). Arm 6 plants Apple's program verbatim, so its word 1 still carries Apple's shader address 0x10000000780, which means nothing in our VM. .518 read the field out of LoadShader::emit @0x68e020: the 28-byte core is written by three stores -- word 0 (+0x00) from [this+76], word 1 (+0x08) from [this+8], word 2 (+0x12) from [this+40]. .481 read [this+40] and reported the address fields zero; that is word 2. Word 1 carries the shader address as h1 bits 15..7 = A bits 14..6, h2 = A bits 30..15, h3 = A bits 46..31, h4 bit 0 = A bit 47, with A bits 5..0 never emitted (64-byte aligned). Validated on all 11 captured cores. The address is NOT guessed: arm 7 parses Mesa's own object before overwriting it -- it tiles exactly as USC words (Sampler/Uniform x4/Shared/SHADER/Registers/No preshader, 56 bytes) -- takes Code out of the USC Shader record (tag 0x0d, u32 at byte 2, genxml `USC Shader`), and forms the address against the same base the hardware forms pipe_va against (win36_base when user_va_win36 is on, else usc_exec_base). If no Shader record is found it prints *** VOID *** and plants nothing. 0 = off. LOG.md 32.41.475, 32.41.482, 32.41.518",
        },
        g13_compat_addr: u64 {
            default: 0xbeef000000,
            description: "G15S: the unmapped GPU VA that g13_compat_test=1 makes the G13 sentinel shader store to. Distinctive and inside the 42-bit IAS. A fault reported at EXACTLY this address proves the USC executed G13-encoded instructions and decoded both the mov_imm pair and the device_store. LOG.md 32.41.475",
        },
        dump_ppp: u32 {
            default: 0,
            description: "G15S: walk this many dwords of the VDM control stream and dump the contents of every PPP State block it points at (block type 0; header = size_words<<8 | addr[39:32], next dword = addr low 32). capture4 captured Apple's NINE PPP blocks with contents for the same 64x64 one-triangle draw; Mesa sends THREE, and we have never looked inside them. 0 = off. LOG.md 32.41.471",
        },
        dump_cs: u32 {
            default: 0,
            description: "G15S: dump this many dwords (max 32) from the head of Mesa's VDM control stream before the kick. The record layout has been argued from notes, never observed. 32.41.449 derived Apple's mask-0x2e VDM State Update as 28 bytes (header + 6 payload words) from AGXMetalG15X_M1 against the 20 the notes ascribe to Mesa; this checks it directly and yields cs+0x10's real value. 0 = off. LOG.md 32.41.450",
        },
        dump_microseq: u32 {
            default: 0,
            description: "G15S: log every microsequence op the driver emits -- opcode, offset, struct size and raw header. The #[ver] gating makes the actual G15S op list hard to read off the source, and a struct size that disagrees with the firmware handler's stride (the w21 it sets) would desync the interpreter. 0 = off. LOG.md 32.41.445",
        },
        dump_vtx_cmd: u32 {
            default: 0,
            description: "Dump the vertex (TA) command struct as u64s, for a field-by-field diff against Apple's AGFIChannelCommandTA captured live on 2026-09-11 (out/raw-draw-triangle-64x64-full.txt, 4096 bytes). The layout is already confirmed -- Apple's bytes decode sanely at our offsets -- but our VALUES have never been compared the way the register lists were (LOG.md 32.41.428/.429)",
        },
        vtx_usc_poll: u32 {
            default: 0,
            description: "also run the usc_poll instrumentation after every VERTEX kick (it spins in the doorbell path; 0 = compute only). LOG.md 32.41.666",
        },
        pb_slot_regs: u32 {
            default: 1,
            description: "G15S: program TA 0x1c830 and 3D 0x1c838 (kext desc[0x1040]/desc[0x808], the hardware buffer id) with the scene's PB slot instead of Apple's captured constants 1/9. With 9 the fragment pass returned TVB pages to an unused PB slot and glmark2 bump drained the heap -> TA OOM (LOG.md 32.41.666). 0 = old constants",
        },
        uma_hwid_slot: u32 {
            default: 1,
            // permissions: 0o644,
            description: "G15S: DC 0x1e +4 (\"hw_buffer_id\") = the job's VM slot. The firmware puts it in SGX 0x14020 bits 56.. and the page manager reads the page list through that UAT context (Apple sends its process context 6 there, 63 at +8). The old constant 1 only worked while the client VM was slot 1: a second process faulted gUPM in the dead vm 1 (LOG.md 32.41.667l). 0 = constant 1",
        },
        uma_shared: u32 {
            default: 960,
            // permissions: 0o644,
            description: "G15S: MiB of ONE device-wide render UMA pool (hw id 5, list/backing/extents all in klow, installed once), shared by every process, as Apple shares UMAFLists per priority type and as compute pool 4 already is. Per-process pools have only 6 hw ids: a 7th rendering process took an id from a live pool, whose in-flight jobs then faulted in the UMAB region (House of Ashes + Steam + Plasma, .667zzz42). 0 = per-process pools",
        },
        render_sig: u32 {
            default: 0,
            // permissions: 0o644,
            description: "debug (.667zzz42): print each new kind of render command once (flags, size, layers, samples, ZLS, helpers) as asahi/rsig lines",
        },
        uma_shared_max: u32 {
            default: 0,
            // permissions: 0o644,
            description: "G15S: cap (MiB) for on-demand growth of the shared render UMA pool (.667zzz42). At each render submission, when the pool's free-page ring (firmware descriptor, sysfs gpu_uma_ring) is below half of its pages, the driver adds up to 256 MiB of klow pages with DeviceControl 0x1e install=0 (a separate page list; the firmware feeds it to gUPM in <= 0xff8-page chunks), as macOS grows AGXUMAFList up to 4-16 GiB. The tiler runs ahead of the 3D by hundreds of scenes in game loading (House of Ashes drained 256 MiB in 30 s) and the firmware's TA-OOM partial render never resumes. 0 = no growth",
        },
        uma_reset_pct: u32 {
            default: 25,
            // permissions: 0o644,
            description: "G15S (.667zzz43): reset the shared render UMA pool's free-page ring when it falls below this percentage of its pages. Spilling fragment shaders (and House of Ashes, ~2400 pages/s) lose pages the page manager never returns; at 0 free the tiler hangs and only a GPU recovery (which re-latches the ring) brings them back. The reset gates new render submissions, waits for every in-flight scene on the pool to retire, rewrites the other of the pool's two page lists with all its pages and re-installs it (DC 0x1e install = 1): with nothing in flight no page is in use, and the page manager writes its cached pages back to the previous list. 0 = never",
        },
        uma_klow: u32 {
            default: 1,
            // permissions: 0o644,
            description: "G15S: allocate the UMA page list in the kernel-lower VM (klow, 0x21_0000_0000..) instead of the client VM, as macOS does (list at 2^40, DC 0x1e ctx 63). Use with uma_dc=65 kctx63=1. The page manager keeps the context of hw buffer 1's first push, so a user-VM list faults (vm of a dead process) once a second process runs (LOG.md 32.41.667k). 667zzj: default 1: list, extents, scratch in klow at 0x200_0000_0000 (own top-level slot, shared into every user VM because gUPM walks them through the job's context), DC+4 = job slot, DC+8 = 63 (kctx63) as Apple. The page manager writes a re-installed id's cached pages back to the previous owner's list through the NEW context, so a user-VM list of a dead process faulted (Plasma); a klow list has the same address in every install",
        },
        dc_destroy_155: u32 {
            default: 1,
            // permissions: 0o644,
            description: "15.5: send DestroyContext / firmware cache flush as DC 0x14 with the compact layout (fw:0x33030). 0 = the enum-position tag 0x18, which on 15.5 is a timeout setter (fw:0x339ec): no context is destroyed and no cache flushed before garbage reuse (LOG.md 32.41.667zzg)",
        },
        q_quarantine: u32 {
            default: 0,
            // permissions: 0o644,
            description: "Queue-drop diagnostics: bit0 leak notifier + notifier_list, bit1 leak the work queues (and through them the GPU context), bit2 shared notifier + print cur_count/threshold/in_list, bit3 print the queue's firmware object addresses. Written to chase the context-churn firmware crashes (fw:0x1ff6c/0x15f28/0x4b4a8), whose cause was the misnumbered 15.5 DestroyContext (dc_destroy_155, LOG.md 32.41.667zzg)",
        },
        allow_foreign_clients: u32 {
            default: 0,
            // permissions: 0o644,
            description: "Accept queues without DRM_ASAHI_QUEUE_G15 on G15 GPUs. Off by default: stock Mesa (G13/G14 code, e.g. inside Steam's muvm guest) faults every job and crashed the firmware (LOG.md 32.41.667zzu)",
        },
        drop_dump: u32 {
            default: 0,
            // permissions: 0o644,
            description: "Queue drop: dump work queues, notifier and the full hung state (LOG.md 32.41.81/.89). Off by default: it runs on every context teardown and dump_scene_bufs can read freed scene buffers (Oops, 32.41.667i)",
        },
        xbarrier_armed: u32 {
            default: 1,
            description: "Barrier+0x30 for EXPLICIT userspace barriers (cdm_barrier/vdm_barrier: render waits for compute in the same submission, GS/tess/XFB, compute after render). Must be nonzero for correctness -- fw:0x1a768 skips the wait when 0 and the render then reads the compute output before it exists (G15 geometry shaders drew nothing). barrier_armed below keeps controlling the internal vertex->fragment barrier. 0 = follow barrier_armed. LOG.md 32.41.667zzz14",
        },
        barrier_armed: u32 {
            default: 0,
            // permissions: 0o644,
            description: "value for Barrier+0x30; fw:0x1a768 skips the wait entirely when zero (LOG.md 32.41.96)",
        },
        uma_rec_trace: u32 {
            default: 0,
            description: "Print the UMA pool record's ring words (+0x20 head, +0x24 tail, +0x2c pages) at each render submission when they changed. LOG.md 32.41.667zzw",
        },
        uma_grow_max: u32 {
            default: 0,
            description: "G15S: cap (MiB) for asynchronous UMA pool growth: when the firmware retires the last chunk of a pool's pages (event 13) the driver doubles the pool with another DeviceControl 0x1e, as macOS's grow requests do (max 4-16 GiB there). 0 = fixed pool. LOG.md 32.41.667zzw",
        },
        uma_tick: u32 {
            default: 0,
            description: "G15S: increment the UMA pool's sync word (pool[0x48]) at every render submission, as AGXUMAFList::updateSubmitInfo does. Crashed the firmware at boot here (fw:0x138dc) -- the firmware appears to own the word; kept off, uma_sync_expect=1 instead. LOG.md 32.41.667zzw",
        },
        uma_sync_expect: u64 {
            // 667zzw: 1, not 0. fw:0x252a0 frees the firmware pool record when
            // *pool[0x48] == pool[0x50]; with both 0 a job that drained the whole
            // pool had it freed under the pending fragment -> PM walk fw:0x13790
            // hit the NULL slot (FAR 0x58). Tested: 1M..8M-triangle frames pass.
            default: 1,
            // permissions: 0o644,
            description: "value stored at UMA pool+0x50, compared against *(pool+0x48) (LOG.md 32.41.90)",
        },
        map_null_page: u32 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: back IOVA 0 with a zeroed page; 2 fills it with self-describing words (LOG.md 32.41.45, .54)",
        },
        mirror_vm_slots: u32 {
            default: 0,
            // permissions: 0o644,
            description: "DIAGNOSTIC: also install each VM's page tables in TTBAT slots 1..N (LOG.md 32.41.39)",
        },
        tvb_publish_wait_ms: u32 {
            default: 2000,
            description: "Before a render submit, wait up to this long for the buffer's scenes in flight so a grown TVB heap gets published (0 = never wait). LOG.md 32.41.667zzz41",
        },
        ctx_leak_held: u32 {
            default: 1,
            description: "Never destroy a GPU context whose hardware compute id is still held (a compute job abandoned by host recovery): the firmware's per-id stats walk would dereference the cleared slot (fw:0x34b24 crash). LOG.md 32.41.667zzz41",
        },
        large_wg_serial: u32 {
            default: 1,
            description: "Run compute submissions flagged as having workgroups > 512 threads one at a time across all processes (two such jobs from different contexts deadlock the CDM). LOG.md 32.41.667zzz41",
        },
        ctx_hwid_ff: u32 {
            default: 1,
            description: "Initialise GpuContextData+0x33 (hardware compute id) to 0xff = unassigned, so the firmware gives each context its own id (0 = old behaviour: all contexts share id 0 and concurrent compute deadlocks). LOG.md 32.41.667zzz41",
        },
        ualloc_jitter: u32 {
            default: 0,
            description: "DIAGNOSTIC: offset each file's shared kernel-VA heap by (file id % 31) * 256 MiB. LOG.md 32.41.667zzz41",
        },
        comp_serial: u32 {
            default: 0,
            description: "At most N submissions with compute commands in flight across all processes (each depends on the one N earlier; max 16, 0 = unlimited). More concurrent compute contexts hang the GPU (VK tess CTS -j8). LOG.md 32.41.667zzz41",
        },
    },
}

/// Fill a UMA pool descriptor the way macOS's
/// `AGXUMAFList::populateInitialFirmwareState` does (kext 0xfffffe0009150754).
///
/// The firmware packs this into the 32-byte hardware descriptor at `fw:0x24d94`:
///
/// | word | contents                                                     |
/// |------|--------------------------------------------------------------|
/// | 0    | `(pages_per_block << 41) \| (base >> 7)`                      |
/// | 1    | `((pages == 0) << 61) \| ((pages % pages_per_block) << 33)`   |
/// | 2    | total pages                                                  |
///
/// Units are **4 KiB pages**, not the UAT's 16 KiB.
///
/// `pool[8]` is the descriptor *slot*, and the installer bails above 0xff
/// (`fw:0x24d20`), so macOS's initial -1 means "owns no slot". Leaving it at the
/// struct's zero claims slot 0 and has the firmware install an all-zero
/// descriptor there -- which is what `gUPM` was faulting on. LOG.md 32.41.61.
/// Fill the 0x90-byte UMA pool struct the way `AGXUMAFList::populateInitialFirmwareState`
/// (kext 0xfffffe0009150754) does.
///
/// Read off the stores, not off a summary of them -- the earlier reading in
/// LOG.md 32.41.61 mis-attributed several because of register reuse:
///
/// ```text
///   pool[0x08] = -1                    ; slot id; the installer bails above 0xff
///   pool[0x0c] = byte  this[24]+170
///   pool[0x10] = word  this[24]+172
///   pool[0x14] = GPU address A         ; unaligned u64, this[80]  -> vt+0x158
///   pool[0x1c] = pages per block       ; this[96]
///   pool[0x20] = 0
///   pool[0x24] = pages % per_block
///   pool[0x28] = (pages == 0)
///   pool[0x2c] = total pages           ; this[112] >> 12, in 4 KiB units
///   pool[0x30] = 0
///   pool[0x34] = GPU address C         ; unaligned u64, this[192] -> vt+0x158
///   pool[0x44] = word  this[208]
///   pool[0x48] = GPU address B         ; this[448] -> vt+0x158
///   pool[0x50] = *(this[456])
///   pool[0x58] = byte this[24]+169, if byte this[24]+168 == 1
/// ```
///
/// **Three** GPU addresses, not one. This driver has only ever filled `0x14`,
/// so `0x34`, `0x48` and `0x50` are all zero -- and something the page manager
/// reaches through one of them would land on VA 0, which is the fault that has
/// outlived every other explanation. `uma_pool_probe` poisons the three with
/// distinguishable never-mapped values. LOG.md 32.41.82.
pub(crate) fn uma_pool_init(pool: &mut [u8], base: u64, size: u64, sync_addr: u64) {
    const UMA_PAGE_SHIFT: u32 = 12;
    // G15S: halGetUMABlockSize() = 0x400000, i.e. 4 MiB blocks.
    const UMA_BLOCK_SIZE: u64 = 0x400000;

    let pages = (size >> UMA_PAGE_SHIFT) as u32;
    let per_block = (UMA_BLOCK_SIZE >> UMA_PAGE_SHIFT) as u32;
    let id = *module_parameters::uma_pool_id.value() as u32;

    pool[0x08..0x0c].copy_from_slice(&id.to_le_bytes());
    pool[0x14..0x1c].copy_from_slice(&base.to_le_bytes());
    pool[0x1c..0x20].copy_from_slice(&per_block.to_le_bytes());
    pool[0x20..0x24].copy_from_slice(&0u32.to_le_bytes());
    pool[0x24..0x28].copy_from_slice(&(pages % per_block).to_le_bytes());
    pool[0x28..0x2c].copy_from_slice(&u32::from(pages == 0).to_le_bytes());
    pool[0x2c..0x30].copy_from_slice(&pages.to_le_bytes());

    // 🔑 `pool[0x48]` is dereferenced, not just stored. `fw:0x252a0` does
    //
    //     ldr  w8, [x0, #8]            ; pool id, bail above 0xff
    //     ldr  x8, [x20, x8, lsl #3]   ; pool table at 0x1a02d0, bail if null
    //     ldp  x8, x9, [x0, #72]       ; pool + 0x48, pool + 0x50
    //     ldr  x8, [x8]                ; <-- data abort, FAR = 0
    //     cmp  x8, x9
    //
    // so once the firmware has actually installed a descriptor for this pool
    // (which only happens on some runs, which is why this showed up as a hang
    // before it showed up as a crash), a null at 0x48 is a guaranteed null
    // dereference inside the firmware. macOS fills the pair from a GPU buffer:
    // `pool[0x48]` is the buffer's address and `pool[0x50]` a snapshot of its
    // first u64 -- so the firmware is asking "has this word changed since
    // setup?". Mirror that: a live zeroed word, and the value it currently
    // holds. LOG.md 32.41.90.
    pool[0x48..0x50].copy_from_slice(&sync_addr.to_le_bytes());
    pool[0x50..0x58].copy_from_slice(&module_parameters::uma_sync_expect.value().to_le_bytes());

    let probe = *module_parameters::uma_pool_probe.value();
    if probe != 0 {
        pool[0x34..0x3c].copy_from_slice(&(probe | 0x1_0000).to_le_bytes());
        pool[0x48..0x50].copy_from_slice(&(probe | 0x2_0000).to_le_bytes());
        pool[0x50..0x58].copy_from_slice(&(probe | 0x3_0000).to_le_bytes());
    }
}
