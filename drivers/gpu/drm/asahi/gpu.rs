// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Top-level GPU manager
//!
//! This module is the root of all GPU firmware management for a given driver instance. It is
//! responsible for initialization, owning the top-level managers (events, UAT, etc.), and
//! communicating with the raw RtKit endpoints to send and receive messages to/from the GPU
//! firmware.
//!
//! It is also the point where diverging driver firmware/GPU variants (using the versions macro)
//! are unified, so that the top level of the driver itself (in `driver`) does not have to concern
//! itself with version dependence.

use core::any::Any;
use core::ops::Range;
use core::slice;
use core::sync::atomic::{
    AtomicBool,
    AtomicU64,
    Ordering, //
};

use kernel::{
    c_str,
    drm::gem::shmem,
    error::code::*,
    io::mem::{
        Mem,
        MemFlag, //
    },
    iosys_map::IoSysMapRef,
    macros::versions,
    new_mutex,
    prelude::*,
    soc::apple::rtkit,
    sync::{
        lock::{
            mutex::MutexBackend,
            Guard, //
        },
        Arc,
        Mutex,
        UniqueArc, //
    },
    time::{
        Delta,
        Instant,
        Monotonic, //
    },
    types::ForeignOwnable, //
};
#[cfg(CONFIG_DEV_COREDUMP)]
use kernel::{
    devcoredump,
    time::msecs_to_jiffies, //
};

use crate::alloc::Allocator;
use crate::debug::*;
use crate::driver::{
    AsahiDevRef,
    AsahiDevice, //
};
use crate::fw::channels::{
    ChannelErrorType,
    PipeType, //
};
use crate::fw::types::{
    U32,
    U64, //
};
use crate::{
    alloc,
    buffer,
    channel,
    event,
    fw,
    gem,
    hw,
    initdata,
    mem,
    mmu,
    queue,
    regs,
    workqueue, //
};

const DEBUG_CLASS: DebugFlags = DebugFlags::Gpu;

/// Firmware endpoint for init & incoming notifications.
const EP_FIRMWARE: u8 = 0x20;

/// Doorbell endpoint for work/message submissions.
const EP_DOORBELL: u8 = 0x21;

/// Initialize the GPU firmware.
const MSG_INIT: u64 = 0x81 << 48;
const INIT_DATA_MASK: u64 = (1 << 44) - 1;

/// TX channel doorbell.
const MSG_TX_DOORBELL: u64 = 0x83 << 48;
/// Firmware control channel doorbell.
const MSG_FWCTL: u64 = 0x84 << 48;
/// TX doorbell with the wake flag (`kickFirmware(idx, _, true)` in the kext): PM event bit 36.
const MSG_TX_WAKE: u64 = 0x86 << 48;
/// RTKit management endpoint.
const EP_MGMT: u8 = 0x0;
/// Management type 0xa with a zero payload: take an RTKit power reference.
/// RTKit management type 0xa: the coprocessor's own power refcount adjuster.
///
/// `0x4461c` does `cmp w1, #0; cset w0, eq` and calls `0x47100`, which is a
/// saturating decrement followed by
/// `csinc w10, w10, w9, eq` -- and CSINC is `if cond then Wn else Wm + 1`, so
/// the *eq* case (w0 == 1, i.e. payload zero) keeps the decrement and the *ne*
/// case increments. In other words a payload of **0** takes a reference and a
/// payload of 1 drops one. Reading that backwards is why every earlier attempt
/// to hold the link open drove the count the wrong way; measured on hardware,
/// fifty payload-1 messages spread across the quiesce window all arrived (the
/// RTKit trace ring counted them) and left the count at zero. LOG.md 32.40.
const MSG_MGMT_PWR_REF: u64 = 0xa << 52;
/// RTKit management SET_IOP_PWR_STATE with `APPLE_RTKIT_PWR_STATE_ON`.
///
/// Processing InitData drives the coprocessor's own power refcount at 0x5c348
/// to zero, its `power` thread runs, and the link goes to `QUIESCED` (0x10) --
/// "running but no communication". Measured directly: `0x5c360` reads 0x10
/// after init, and from that moment the RTKit trace ring at 0x19b540 stops
/// growing, no EP 0x21 doorbell is counted at 0x1208fc, and the pipe rings are
/// never drained. Device control still works, but only because the firmware
/// polls it itself at 0x31dbc. LOG.md 32.40.
const MSG_MGMT_SET_IOP_PWR_ON: u64 = (0x6 << 52) | 0x20;
/// Host-initiated recovery request, sent on EP_DOORBELL with the low 44 bits =
/// the firmware's recovery count b0[0x4f90] (fw:0x319d8; kext
/// AGXArmFirmware::stopFirmwareForRecoveryInspection). A stale count is ignored.
const MSG_HALT: u64 = 0x85 << 48;

/// Receive channel doorbell notification.
const MSG_RX_DOORBELL: u64 = 0x42 << 48;

/// Doorbell number for firmware kicks/wakeups.
const DOORBELL_KICKFW: u64 = 0x10;
/// Doorbell number for device control channel kicks.
const DOORBELL_DEVCTRL: u64 = 0x11;

// Upper kernel half VA address ranges.
/// Private (cached) firmware structure VA range base.
const IOVA_KERN_PRIV_RANGE: Range<u64> = 0xfffffc2000000000..0xfffffc2600000000;
/// Private (cached) GPU-RO firmware structure VA range base.
const IOVA_KERN_GPU_RO_RANGE: Range<u64> = 0xfffffc2600000000..0xfffffc2800000000;
/// Shared (uncached) firmware structure VA range base.
pub(crate) const IOVA_KERN_SHARED_RANGE: Range<u64> = 0xfffffc2800000000..0xfffffc2a00000000;
/// Shared (uncached) read-only firmware structure VA range base.
const IOVA_KERN_SHARED_RO_RANGE: Range<u64> = 0xfffffc2a00000000..0xfffffc2c00000000;
/// GPU/FW shared structure VA range base.
const IOVA_KERN_GPU_RANGE: Range<u64> = 0xfffffc2c00000000..0xfffffc2e00000000;
/// GPU/FW shared structure VA range base.
// Must sit inside mmu::IOVA_KERN_RANGE, which moved to the ADT-derived base
// (see the comment on IOVA_KERN_BASE). Kept at the same offset within the range
// as the old 0xfffffe0e00000000 was within the old one.
const IOVA_KERN_RTKIT_RANGE: Range<u64> = 0xfffffc2e00000000..0xfffffc2e10000000;
/// Shared (uncached) timestamp region.
pub(crate) const IOVA_KERN_TIMESTAMP_RANGE: Range<u64> = 0xfffffc2e10000000..0xfffffc2e14000000;
/// FW MMIO VA range base.
const IOVA_KERN_MMIO_RANGE: Range<u64> = 0xfffffc2f00000000..0xfffffc3000000000;

/// GPU/FW buffer manager control address (context 0 low)
pub(crate) const IOVA_KERN_GPU_BUFMGR_LOW: u64 = 0x20_0000_0000;
/// GPU/FW buffer manager control address (context 0 high)
pub(crate) const IOVA_KERN_GPU_BUFMGR_HIGH: u64 = 0xfffffc2effff0000;

/// Fixed IOVAs for the UMA page pool descriptor table, the second of the two
/// mappings `AGXAcceleratorG15::createHardwareMappings` builds. Same shape as
/// the buffer-manager pair above: one allocation, GPU-shared in the
/// kernel-lower VM and FW-shared in the kernel VM. LOG.md 32.41.72.
/// The pass-page counter, dual-mapped exactly like `buffer_mgr_ctl`: gPM reads
/// it through register `0x1c910` (which can only express a LOW address -- bits
/// 1..38 plus bit 42), and the firmware CPU dereferences it through
/// `Scene.control_word2` (which needs a kernel-VM address). One page, both.
/// LOG.md 32.41.507/.508.
/// 667zzx: gPM walks 0x1c910 through the JOB's context (the .508 fault), so the
/// LOW page sits alone in top-level slot 7 (0x70_0000_0000, above the 2^38 user
/// range and its 2^36 alias, below the fold's 2^39 limit), which every user VM
/// shares (mmu.rs `PMCT_TOP_IDX`), like klow.
pub(crate) const IOVA_KERN_GPU_PMCT_LOW: u64 = 0x7f_fff8_0000;
pub(crate) const IOVA_KERN_GPU_PMCT_HIGH: u64 = 0xfffffc2efffd0000;
pub(crate) const IOVA_KERN_GPU_UMAPOOL_LOW: u64 = 0x20_0004_0000;
/// `klow` allocator range (kernel-lower VM, below 2^43 so it fits the UMA page-list
/// field SGX 0x14020 bits 6..42). 16 GiB = ~2000 8-MiB UMA page lists.
/// 667zzj: its own top-level slot (32 = 0x200_0000_0000, 64 GiB), which every user VM
/// shares (mmu.rs `KLOW_TOP_IDX`): gUPM reads/writes the list and extents through the
/// job's context, like macOS's 2^40 tables through process context 6.
pub(crate) const IOVA_KERN_GPU_KLOW_RANGE: core::ops::Range<u64> = 0x200_0000_0000..0x204_0000_0000;
pub(crate) const IOVA_KERN_GPU_UMAPOOL_HIGH: u64 = 0xfffffc2efffe0000;

/// Timeout for entering the halt state after a fault or request.
const HALT_ENTER_TIMEOUT: Delta = Delta::from_millis(100);

/// Maximum amount of firmware-private memory garbage allowed before collection.
/// Collection flushes the FW cache and is expensive, so this needs to be
/// reasonably high.
const MAX_FW_ALLOC_GARBAGE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum count of firmware-private memory garbage objects allowed before collection.
/// This works out to 16K of memory in the garbage list (8 bytes each), which keeps us
/// within the safe range for kmalloc (on 16K page systems).
const MAX_FW_ALLOC_GARBAGE_OBJECTS: usize = 2048;

/// Global allocators used for kernel-half structures.
pub(crate) struct KernelAllocators {
    pub(crate) private: alloc::DefaultAllocator,
    pub(crate) shared: alloc::DefaultAllocator,
    pub(crate) shared_ro: alloc::DefaultAllocator,
    #[allow(dead_code)]
    pub(crate) gpu: alloc::DefaultAllocator,
    pub(crate) gpu_ro: alloc::DefaultAllocator,
    /// Kernel-lower VM (context 0 TTB0, mirrored to 63 by `kctx63`): hardware-read
    /// tables that must outlive any user VM, like Apple's 2^40 objects. LOG.md 32.41.667k.
    pub(crate) klow: alloc::DefaultAllocator,
}

/// Receive (GPU->driver) ring buffer channels.
#[versions(AGX)]
#[pin_data]
struct RxChannels {
    event: channel::EventChannel::ver,
    fw_log: channel::FwLogChannel,
    ktrace: channel::KTraceChannel::ver,
    stats: channel::StatsChannel::ver,
}

/// GPU work submission pipe channels (driver->GPU).
#[versions(AGX)]
struct PipeChannels {
    pub(crate) vtx: KVec<Pin<KBox<Mutex<channel::PipeChannel::ver>>>>,
    pub(crate) frag: KVec<Pin<KBox<Mutex<channel::PipeChannel::ver>>>>,
    pub(crate) comp: KVec<Pin<KBox<Mutex<channel::PipeChannel::ver>>>>,
}

/// Misc command transmit (driver->GPU) channels.
#[versions(AGX)]
#[pin_data]
struct TxChannels {
    pub(crate) device_control: channel::DeviceControlChannel::ver,
}

/// Number of work submission pipes per type, one for each priority level.
const NUM_PIPES: usize = 4;

/// A generic monotonically incrementing ID used to uniquely identify object instances within the
/// driver.
pub(crate) struct ID(AtomicU64);

impl ID {
    /// Create a new ID counter with a given value.
    fn new(val: u64) -> ID {
        ID(AtomicU64::new(val))
    }

    /// Fetch the next unique ID.
    pub(crate) fn next(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for ID {
    /// IDs default to starting at 2, as 0/1 are considered reserved for the system.
    fn default() -> Self {
        Self::new(2)
    }
}

/// A guard representing one active submission on the GPU. When dropped, decrements the active
/// submission count.
pub(crate) struct OpGuard(Arc<dyn GpuManagerPriv>);

impl Drop for OpGuard {
    fn drop(&mut self) {
        self.0.end_op();
    }
}

/// CPU address of pipe0 slot 63's state word (stats.vtx + 0xc + 63 * 24), 0 if
/// `comp_busy` is off. LOG.md 32.41.667zzy.
static COMP_BUSY_PTR: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
/// CPU pointers of stats.vtx / stats.frag / stats.comp (= RegionB+0x244/0x24c/0x254
/// with pipe_stats), for host recovery. LOG.md 32.41.667zzz29.
/// Set while `host_recover` runs.
static RECOVERING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static STATS_PTRS: [core::sync::atomic::AtomicUsize; 3] = [
    core::sync::atomic::AtomicUsize::new(0),
    core::sync::atomic::AtomicUsize::new(0),
    core::sync::atomic::AtomicUsize::new(0),
];
/// Compute jobs submitted and not yet retired.
static COMP_INFLIGHT: core::sync::atomic::AtomicI32 = core::sync::atomic::AtomicI32::new(0);

/// Account a compute job entering (+1) or leaving (-1) flight and mark slot 63
/// busy while any is in flight. Every PM tick runs the quiesce path
/// (fw:0x32160 -> fw:0x35e9c); with all 64 slots idle it context-stores the
/// running CDM job, and on 15.5 the resumed job never answers the next store
/// (LOG.md 32.41.667zzy). A busy slot makes the scan skip the quiesce. The
/// count is re-read after each write so a racing submit/retire cannot leave a
/// stale state behind: the last writer always re-validates.
/// .667zzz: DC 0x1e page-list chunks (<= 0xff8 pages each, fw:0x24b44 -> SGX
/// 0x14018/0x14020 DMA) the firmware has not retired yet, per pool id. The
/// hardware pool fills asynchronously; a compute job whose first preemption
/// store lands before the fill completes stores USC context into page 0
/// (UL1C write 0x0/0x100/0x300, restore read 0xf00).
pub(crate) static UMA_CHUNKS: [core::sync::atomic::AtomicI32; 8] =
    [const { core::sync::atomic::AtomicI32::new(0) }; 8];

/// One chunk of pool `id` retired (event 13).
pub(crate) fn uma_chunk_retired(id: u32) {
    if let Some(c) = UMA_CHUNKS.get(id as usize) {
        let _ = c.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some((v - 1).max(0)));
    }
}

/// Wait (bounded by `ms`) until pool `id` has no chunks in flight. Returns the
/// time waited in microseconds, or None on timeout.
pub(crate) fn uma_wait_filled(id: u32, ms: u32) -> Option<u32> {
    let c = UMA_CHUNKS.get(id as usize)?;
    let mut us = 0u32;
    while c.load(Ordering::Acquire) > 0 {
        if us >= ms * 1000 {
            c.store(0, Ordering::Release);
            return None;
        }
        kernel::time::delay::fsleep(kernel::time::Delta::from_micros(50));
        us += 50;
    }
    Some(us)
}

pub(crate) fn comp_busy_update(delta: i32) {
    let p = COMP_BUSY_PTR.load(Ordering::Acquire);
    if p == 0 {
        return;
    }
    COMP_INFLIGHT.fetch_add(delta, Ordering::AcqRel);
    let busy = *crate::module_parameters::comp_busy.value();
    loop {
        let n = COMP_INFLIGHT.load(Ordering::Acquire);
        // SAFETY: points into stats.vtx, alive as long as the GPU manager.
        unsafe { core::ptr::write_volatile(p as *mut u32, if n > 0 { busy } else { 0 }) };
        core::sync::atomic::fence(Ordering::SeqCst);
        if COMP_INFLIGHT.load(Ordering::Acquire) == n {
            break;
        }
    }
}

/// Set of global sequence IDs used in the driver.
#[derive(Default)]
pub(crate) struct SequenceIDs {
    /// `File` instance ID.
    pub(crate) file: ID,
    /// `Vm` instance ID.
    pub(crate) vm: ID,
    /// Submission instance ID.
    pub(crate) submission: ID,
    /// `Queue` instance ID.
    pub(crate) queue: ID,
}
/// Receive-poll accounting; see `poll_rx`.
/// Scene addresses stashed at submission so the timeout path can read them back.
///
/// The `probe_usc` dumps in `queue/render.rs` run inside the *submission* path,
/// before the kick, so they can only ever show a scene buffer's initial
/// contents -- which is how 32.41.135 came to claim "the vertex executes
/// nothing" from what was really just an unused buffer (retracted in
/// 32.41.137). Reading the same addresses in `handle_timeout`, after the
/// firmware has given up, is an actual after-the-fact measurement: a tiler that
/// ran leaves tile lists behind, one that never decoded the control stream does
/// not.
/// GPU VA of the firmware `Scene` object and the `buffer::Info` for the last
/// submission, so `probe_scene` can read the tiler's own page accounting after
/// the kick.  LOG.md 32.41.234.
/// CPU addresses of the tiler page counters for the in-flight scene.
/// See `Scene::counter_ptrs`.  LOG.md 32.41.236.
/// CPU address of the in-flight job's `RenderTimestamps` object.
///
/// 32.41.443: an EXECUTION probe, not a state probe. `RenderTimestamps` is
/// `{vtx:{start,end}, frag:{start,end}}`, four u64s, and the *firmware writes
/// them from microsequence ops* -- so a non-zero field is direct evidence that
/// execution reached the op that writes it. Every other instrument in this phase
/// samples end state; this one samples progress. Needed because 32.41.442 showed
/// the `0xc020` waits are never entered, which leaves the place the sequence
/// stops completely unlocated.
pub(crate) static DBG_TS_OBJ: AtomicU64 = AtomicU64::new(0);
/// CPU address of the in-flight `RunVertex` command, for the post-kick re-dump.
/// LOG.md 32.41.444.
pub(crate) static DBG_VTXCMD: AtomicU64 = AtomicU64::new(0);
/// CPU address of the in-flight fragment command's register list (post-kick
/// dump of what the firmware appended). LOG.md 32.41.602.
pub(crate) static DBG_FRAGREGS: AtomicU64 = AtomicU64::new(0);
/// CPU address of the in-flight RunFragment, to dump its JobMeta stamp record
/// (+0xb80..+0xbe0) after the job. LOG.md 32.41.607.
pub(crate) static DBG_FRAGCMD: AtomicU64 = AtomicU64::new(0);

/// Scene buffers (TMAP/HMTA/CTMP/CMTA) CPU views, for the post-job dump. LOG.md 32.41.613.
static DBG_BUF_PTR: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static DBG_BUF_LEN: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static DBG_BUF_NAME: [&str; 5] = ["TMAP", "HMTA", "CTMP", "CMTA", "UMAB"];

pub(crate) fn dbg_note_buf(i: usize, _name: &str, p: usize, len: usize) {
    DBG_BUF_PTR[i].store(p as u64, Ordering::Relaxed);
    DBG_BUF_LEN[i].store(len as u64, Ordering::Relaxed);
}

/// Count non-zero words in each noted scene buffer and print the first few.
/// `tvb_cpu=1`: CPU views of the first TVB blocks (tile-list pages). LOG.md 32.41.614.
static DBG_TVB_PTR: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static DBG_TVB_VA: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

pub(crate) fn dbg_note_tvb(i: usize, p: usize, va: u64) {
    if i < 8 {
        DBG_TVB_PTR[i].store(p as u64, Ordering::Relaxed);
        DBG_TVB_VA[i].store(va, Ordering::Relaxed);
    }
}

/// 32.41.637 oracle: write the known-final TMAP/HMTA words (render-replayd1, identical in every
/// run) from the CPU before the kick. If the fast (no-delay) render then works, the ISP was
/// reading a stale tile map. mode bit0 = TMAP+HMTA.
/// 32.41.639: when do the TA's writes reach memory? Poll TMAP[0], HMTA[0] and TVB page 1 word 0
/// (first tile list) from the CPU after the vertex kick, `dc civac` before every read, and print the
/// cntvct (same 24 MHz timebase as ktrace `t`) of each first non-zero value.
pub(crate) fn tmap_poll(budget_us: u32, res: &crate::regs::Resources) {
    let cnt = || -> u64 {
        let v: u64;
        // SAFETY: reading the virtual counter has no side effects.
        unsafe { core::arch::asm!("isb", "mrs {}, cntvct_el0", out(reg) v) };
        v
    };
    let rd = |a: u64| -> u32 {
        // SAFETY: a is a live, mapped scene/TVB word; civac only affects the CPU cache line.
        unsafe {
            core::arch::asm!("dc civac, {}", "dsb sy", in(reg) a);
            core::ptr::read_volatile(a as *const u32)
        }
    };
    let tmap = DBG_BUF_PTR[0].load(Ordering::Relaxed);
    let hmta = DBG_BUF_PTR[1].load(Ordering::Relaxed);
    let tvb1 = {
        let p = DBG_TVB_PTR[0].load(Ordering::Relaxed);
        if p != 0 { p + 0x8000 } else { 0 }
    };
    if tmap == 0 || hmta == 0 || tvb1 == 0 {
        pr_info!("asahi/tmappoll: buffers not noted ({:#x} {:#x} {:#x})\n", tmap, hmta, tvb1);
        return;
    }
    let t0 = cnt();
    let end = t0 + 24 * budget_us as u64;
    let (mut tt, mut th, mut tv) = (0u64, 0u64, 0u64);
    let (mut vt, mut vh, mut vv) = (0u32, 0u32, 0u32);
    let mut n = 0u64;
    // 0xc020 transitions (TA state) on the same CPU timeline.
    let mut st = [(0u64, 0u64); 12];
    let mut ns = 0usize;
    let mut last = u64::MAX;
    let mut now = t0;
    while now <= end {
        now = cnt();
        let s = res.sgx_state();
        if s != last && ns < st.len() { st[ns] = (now - t0, s); ns += 1; last = s; }
        if tt == 0 { let v = rd(tmap); if v != 0 { tt = now - t0; vt = v; } }
        if th == 0 { let v = rd(hmta); if v != 0 { th = now - t0; vh = v; } }
        if tv == 0 { let v = rd(tvb1); if v != 0 { tv = now - t0; vv = v; } }
        n += 1;
        // Yield: a busy loop here blocks the doorbell (.640 -- the TA never ran during the poll).
        kernel::time::delay::fsleep(kernel::time::Delta::from_micros(2));
    }
    pr_info!("asahi/tmappoll: span {} us, polls {} | TVBp1 @{} us ({:#x}) | HMTA @{} us ({:#x}) | TMAP @{} us ({:#x})\n",
             (now - t0) / 24, n, tv / 24, vv, th / 24, vh, tt / 24, vt);
    for (t, s) in st.iter().take(ns) {
        pr_info!("asahi/tmappoll:   c020 @{} us = {:#x}\n", t / 24, s);
    }
}

static PREFILL_DRAW: AtomicU64 = AtomicU64::new(0);

pub(crate) fn prefill_scene(mode: u32) {
    // mode 2: split per draw -- draw 1 TMAP only, draw 2 HMTA only (LOG.md 32.41.638).
    let draw = PREFILL_DRAW.fetch_add(1, Ordering::Relaxed);
    if mode == 2 {
        let tmap = DBG_BUF_PTR[0].load(Ordering::Relaxed);
        let hmta = DBG_BUF_PTR[1].load(Ordering::Relaxed);
        if draw == 0 && tmap != 0 {
            for (o, v) in [(0x00u64, 0x00018002u32), (0x04, 0x02000238), (0x08, 0x80023800), (0x0c, 0x02380000), (0x10, 0x38000100)] {
                // SAFETY: TMAP is at least 320 words and owned by the live scene.
                unsafe { core::ptr::write_volatile((tmap + o) as *mut u32, v) };
            }
            pr_info!("asahi/prefill: draw {} TMAP only\n", draw);
        } else if draw == 1 && hmta != 0 {
            // SAFETY: HMTA is at least 128 words and owned by the live scene.
            unsafe { core::ptr::write_volatile(hmta as *mut u32, 0xc) };
            pr_info!("asahi/prefill: draw {} HMTA only\n", draw);
        }
        return;
    }
    if mode & 1 != 0 {
        let tmap = DBG_BUF_PTR[0].load(Ordering::Relaxed);
        let hmta = DBG_BUF_PTR[1].load(Ordering::Relaxed);
        if tmap != 0 && hmta != 0 {
            let words: [(usize, u32); 5] = [
                (0x00, 0x00018002),
                (0x04, 0x02000238),
                (0x08, 0x80023800),
                (0x0c, 0x02380000),
                (0x10, 0x38000100),
            ];
            for (o, v) in words {
                // SAFETY: TMAP is at least 320 words and owned by the live scene.
                unsafe { core::ptr::write_volatile((tmap + o as u64) as *mut u32, v) };
            }
            // SAFETY: HMTA is at least 128 words and owned by the live scene.
            unsafe { core::ptr::write_volatile(hmta as *mut u32, 0xc) };
            pr_info!("asahi/prefill: TMAP {:#x} (5 words) + HMTA {:#x} written\n", tmap, hmta);
        } else {
            pr_info!("asahi/prefill: TMAP/HMTA not noted, skipped\n");
        }
    }
}

/// Per 32 KiB TVB page: non-zero word count and the first 24 non-zero words.
pub(crate) fn dump_tvb(tag: &str) {
    for b in 0..8 {
        let p = DBG_TVB_PTR[b].load(Ordering::Relaxed);
        let va = DBG_TVB_VA[b].load(Ordering::Relaxed);
        if p == 0 {
            continue;
        }
        for pg in 0..crate::buffer::PAGES_PER_BLOCK {
            let base = p + (pg * crate::buffer::PAGE_SIZE) as u64;
            let n = crate::buffer::PAGE_SIZE / 4;
            let mut nz = 0usize;
            let mut shown = 0;
            for w in 0..n {
                // SAFETY: the block lives as long as the Buffer, which outlives the probe.
                let v = unsafe { core::ptr::read_volatile((base + 4 * w as u64) as *const u32) };
                if v != 0 {
                    nz += 1;
                    if shown < 24 {
                        pr_info!("asahi/tvb {} {:#x} +{:#06x} = {:#010x}\n", tag,
                                 va + (pg * crate::buffer::PAGE_SIZE) as u64, w * 4, v);
                        shown += 1;
                    }
                }
            }
            if nz != 0 {
                pr_info!("asahi/tvb {} page {:#x}: {} non-zero words\n", tag,
                         va + (pg * crate::buffer::PAGE_SIZE) as u64, nz);
            }
        }
    }
}

pub(crate) fn dump_scene_bufs(tag: &str) {
    for i in 0..5 {
        let p = DBG_BUF_PTR[i].load(Ordering::Relaxed);
        let n = DBG_BUF_LEN[i].load(Ordering::Relaxed) as usize / 4;
        if p == 0 || n == 0 {
            continue;
        }
        let mut nz = 0usize;
        let mut shown = 0;
        for w in 0..n {
            // SAFETY: the buffer is owned by the live scene for the duration of this probe.
            let v = unsafe { core::ptr::read_volatile((p + 4 * w as u64) as *const u32) };
            if v != 0 {
                nz += 1;
                // UMAB (tile-list pages, LOG.md 32.41.614) gets a longer listing.
                if shown < if i == 4 { 160 } else { 12 } {
                    pr_info!("asahi/scenebuf {} {} +{:#07x} = {:#010x}\n", tag, DBG_BUF_NAME[i], w * 4, v);
                    shown += 1;
                }
            }
        }
        pr_info!("asahi/scenebuf {} {}: {} of {} words non-zero\n", tag, DBG_BUF_NAME[i], nz, n);
    }
}

/// Size of the in-flight RunFragment, and its words at submit time, so the
/// post-job report can print exactly what the firmware/GPU wrote. LOG.md 32.41.608.
pub(crate) static DBG_FRAGCMD_SIZE: AtomicU64 = AtomicU64::new(0);
static FRAGCMD_SNAP: [core::sync::atomic::AtomicU32; 0x400] =
    [const { core::sync::atomic::AtomicU32::new(0) }; 0x400];

/// Snapshot (tag "PRE") or diff against the snapshot (any other tag).
pub(crate) fn diff_fragcmd(tag: &str) {
    let c = DBG_FRAGCMD.load(Ordering::Relaxed);
    let n = (DBG_FRAGCMD_SIZE.load(Ordering::Relaxed) as usize / 4).min(FRAGCMD_SNAP.len());
    if c == 0 || n == 0 {
        return;
    }
    let mut changed = 0u32;
    for i in 0..n {
        // SAFETY: published by the in-flight job, which owns the command for
        // the duration of this probe; `n` words lie inside the object.
        let v = unsafe { core::ptr::read_volatile((c + 4 * i as u64) as *const u32) };
        if tag == "PRE" {
            FRAGCMD_SNAP[i].store(v, Ordering::Relaxed);
        } else {
            let o = FRAGCMD_SNAP[i].load(Ordering::Relaxed);
            if o != v {
                changed += 1;
                pr_info!("asahi/fragdiff {} +{:#05x}: {:#010x} -> {:#010x}\n", tag, 4 * i, o, v);
            }
        }
    }
    pr_info!("asahi/fragdiff {}: {} words of {:#x} changed\n", tag, changed, 4 * n);
}

/// Print RunFragment bytes 0xb80..0xbe0 as u32s.
pub(crate) fn dump_fragcmd(tag: &str) {
    diff_fragcmd(tag);
    let c = DBG_FRAGCMD.load(Ordering::Relaxed);
    if c == 0 {
        return;
    }
    let mut w = [0u32; 24];
    for (i, x) in w.iter_mut().enumerate() {
        // SAFETY: published by the in-flight job, which owns the command for
        // the duration of this probe; RunFragment is larger than 0xbe0 bytes.
        *x = unsafe { core::ptr::read_volatile((c + 0xb80 + 4 * i as u64) as *const u32) };
    }
    for r in 0..3 {
        pr_info!(
            "asahi/fragcmd {} +{:#05x}: {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
            tag, 0xb80 + r * 32, w[r * 8], w[r * 8 + 1], w[r * 8 + 2], w[r * 8 + 3],
            w[r * 8 + 4], w[r * 8 + 5], w[r * 8 + 6], w[r * 8 + 7]
        );
    }
}
pub(crate) static DBG_CNT_PASS: AtomicU64 = AtomicU64::new(0);
/// CPU address of `pm_counter` -- the word gPM increments, as opposed to
/// `pass_page_count`, which is the firmware's copy of it. LOG.md 32.41.508.
pub(crate) static DBG_CNT_PMCT: AtomicU64 = AtomicU64::new(0);
pub(crate) static DBG_CNT_TOTAL: AtomicU64 = AtomicU64::new(0);
/// CPU address of the vertex UMA pool record (0x50 bytes dumped at the
/// hang: [0x20]/[0x24] ring head/tail). LOG.md 32.41.599.
pub(crate) static DBG_POOLREC: AtomicU64 = AtomicU64::new(0);
pub(crate) static DBG_CNT_BLOCKS: AtomicU64 = AtomicU64::new(0);
pub(crate) static DBG_CNT_PAGES: AtomicU64 = AtomicU64::new(0);
/// GPU address of the firmware's vertex stats block, stashed at submit time.
///
/// `GpuStatsVtx` is 0x3000 bytes of driver-allocated memory the *firmware*
/// writes into. Reading it is a page-table walk, not MMIO, so unlike every
/// `0xc020` probe it cannot perturb the run (32.41.336). If the firmware records
/// anything for the vertex stage, it lands here. LOG.md 32.41.337.
pub(crate) static DBG_STATS_VTX: AtomicU64 = AtomicU64::new(0);
/// Companion stats blocks, dumped as a **positive control** for the vertex one.
///
/// An empty `GpuStatsVtx` (32.41.337) only means something if the firmware
/// writes these blocks at all. If `frag` or `comp` carry non-zero content while
/// `vtx` does not, the instrument works and the vertex silence is real; if all
/// three are zero, the probe proves nothing. LOG.md 32.41.338.
pub(crate) static DBG_STATS_FRAG: AtomicU64 = AtomicU64::new(0);
pub(crate) static DBG_STATS_COMP: AtomicU64 = AtomicU64::new(0);
pub(crate) static DBG_SCENE_OBJ: AtomicU64 = AtomicU64::new(0);
pub(crate) static DBG_SCENE_INFO: AtomicU64 = AtomicU64::new(0);
pub(crate) static DBG_SCENE_TILEMAP: AtomicU64 = AtomicU64::new(0);
pub(crate) static DBG_SCENE_HEAPMETA: AtomicU64 = AtomicU64::new(0);
pub(crate) static DBG_SCENE_SLOT: AtomicU64 = AtomicU64::new(!0);

static RX_STATS_TOTAL: AtomicU64 = AtomicU64::new(0);
static RX_POLLS: AtomicU64 = AtomicU64::new(0);
static RX_POLLS_NOCB: AtomicU64 = AtomicU64::new(0);


/// Top-level GPU manager that owns all the global state relevant to the driver instance.
#[versions(AGX)]
#[pin_data]
pub(crate) struct GpuManager {
    dev: AsahiDevRef,
    cfg: &'static hw::HwConfig,
    dyncfg: hw::DynConfig,
    pub(crate) initdata: fw::types::GpuObject<fw::initdata::InitData::ver>,
    uat: mmu::Uat,
    crashed: AtomicBool,
    #[pin]
    alloc: Mutex<KernelAllocators>,
    io_mappings: KVec<mmu::KernelMapping>,
    next_mmio_iova: u64,
    #[pin]
    rtkit: Mutex<Option<rtkit::RtKit<GpuManager::ver>>>,
    #[pin]
    rx_channels: Mutex<RxChannels::ver>,
    #[pin]
    tx_channels: Mutex<TxChannels::ver>,
    #[pin]
    fwctl_channel: Mutex<channel::FwCtlChannel>,
    pipes: PipeChannels::ver,
    event_manager: Arc<event::EventManager>,
    buffer_mgr: buffer::BufferManager::ver,
    ids: SequenceIDs,
    #[allow(clippy::vec_box)]
    #[pin]
    garbage_contexts: Mutex<KVec<KBox<fw::types::GpuObject<fw::workqueue::GpuContextData>>>>,
}

/// Trait used to abstract the firmware/GPU-dependent variants of the GpuManager.
pub(crate) trait GpuManager: Send + Sync {
    /// Cast as an Any type.
    fn as_any(&self) -> &dyn Any;
    /// Cast Arc<Self> as an Any type.
    fn arc_as_any(self: Arc<Self>) -> Arc<dyn Any + Sync + Send>;
    /// Initialize the GPU.
    fn init(&self) -> Result;
    /// Update the GPU globals from global info
    ///
    /// TODO: Unclear what can and cannot be updated like this.
    fn update_globals(&self);
    /// Get a reference to the KernelAllocators.
    fn alloc(&self) -> Guard<'_, KernelAllocators, MutexBackend>;
    /// Create a new `Vm` given a unique `File` ID.
    fn new_vm(&self, kernel_range: Range<u64>) -> Result<mmu::Vm>;
    /// Bind a `Vm` to an available slot and return the `VmBind`.
    fn bind_vm(&self, vm: &mmu::Vm) -> Result<mmu::VmBind>;
    /// Create a new user command queue.
    fn new_queue(
        &self,
        vm: mmu::Vm,
        ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
        ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
        priority: u32,
        usc_exec_base: u64,
        uma: crate::buffer::UmaSlot,
    ) -> Result<KBox<dyn queue::Queue>>;
    /// Return a reference to the global `SequenceIDs` instance.
    fn ids(&self) -> &SequenceIDs;
    /// Kick the firmware (wake it up if asleep).
    ///
    /// This should be useful to reduce latency on work submission, so we can ask the firmware to
    /// wake up while we do some preparatory work for the work submission.
    fn kick_firmware(&self) -> Result;
    /// Flush the entire firmware cache.
    ///
    /// TODO: Does this actually work?
    fn flush_fw_cache(&self) -> Result;
    /// Handle a GPU work timeout event.
    fn handle_timeout(&self, counter: u32, event_slot: i32, unk: u32);
    /// Handle a GPU fault event.
    fn handle_fault(&self);
    /// Stop the firmware from the host and run the recovery handshake (for a
    /// job the firmware itself never reports as hung). Returns true when the
    /// firmware came back. LOG.md 32.41.667zzz29.
    fn host_recover(&self) -> bool;
    /// Handle a channel error event.
    fn handle_channel_error(
        &self,
        error_type: ChannelErrorType,
        pipe_type: u32,
        event_slot: u32,
        event_value: u32,
    );
    /// Acknowledge a Buffer grow op.
    fn ack_grow(&self, buffer_slot: u32, vm_slot: u32, counter: u32);
    /// Send DeviceControl 0x1e (UMA add pages). LOG.md 32.41.545.
    #[allow(clippy::too_many_arguments)]
    fn uma_add_pages(&self, hw_buffer_id: u32, ctx: u32, pool_record: u64, page_list: u64,
                     pages: u32, pool_id: u32, install: u32);
    /// Send a firmware control command (secure cache flush).
    fn fwctl(&self, msg: fw::channels::FwCtlMsg) -> Result;
    /// Dump the state that tells a wedged firmware apart from a stuck job.
    ///
    /// A job that hangs without faulting never reaches `handle_fault()`, so the
    /// register dump there is unreachable. Queue teardown is the only moment we
    /// still get. LOG.md 32.41.89.
    fn dump_hung_state(&self);
    /// Get the static GPU configuration for this SoC.
    fn get_cfg(&self) -> &'static hw::HwConfig;
    /// Get the dynamic GPU configuration for this SoC.
    fn get_dyncfg(&self) -> &hw::DynConfig;
    /// Register an unused context as garbage
    fn free_context(&self, data: KBox<fw::types::GpuObject<fw::workqueue::GpuContextData>>);
    /// Check whether the GPU is crashed
    fn is_crashed(&self) -> bool;
    /// Map a BO as a timestamp buffer
    fn map_timestamp_buffer(
        &self,
        bo: gem::ObjectRef,
        range: Range<usize>,
    ) -> Result<mmu::KernelMapping>;
}

/// Private generic trait for functions that don't need to escape this module.
trait GpuManagerPriv {
    /// Decrement the pending submission counter.
    fn end_op(&self);
}

pub(crate) struct RtkitObject {
    vmap: shmem::VMap<gem::AsahiObject, u8>,
    mapping: mmu::KernelMapping,
}

impl rtkit::Buffer for RtkitObject {
    fn iova(&self) -> Result<usize> {
        Ok(self.mapping.iova() as usize)
    }
    fn buf(&mut self) -> Result<IoSysMapRef<'_, u8>> {
        Ok(self.vmap.get())
    }
}

#[versions(AGX)]
#[vtable]
impl rtkit::Operations for GpuManager::ver {
    type Data = Arc<GpuManager::ver>;
    type Buffer = RtkitObject;

    fn recv_message(data: <Self::Data as ForeignOwnable>::Borrowed<'_>, ep: u8, msg: u64) {
        let dev = &data.dev;
        //dev_info!(dev.as_ref(), "RtKit message: {:#x}:{:#x}\n", ep, msg);

        if ep != EP_FIRMWARE || msg != MSG_RX_DOORBELL {
            dev_err!(dev.as_ref(), "Unknown message: {:#x}:{:#x}\n", ep, msg);
            return;
        }

        data.poll_rx();
    }

    fn crashed(data: <Self::Data as ForeignOwnable>::Borrowed<'_>, crashlog: Option<&[u8]>) {
        let dev = &data.dev;

        data.crashed.store(true, Ordering::Relaxed);
        // A firmware crash is as much a dead end as a timeout: show the trace
        // records that led to it (ktrace_hist=1). LOG.md 32.41.667zzw.
        crate::channel::ktrace_hist_dump();

        #[cfg(CONFIG_DEV_COREDUMP)]
        if let Err(e) = data.generate_crashdump(crashlog) {
            dev_err!(dev.as_ref(), "Could not generate crashdump: {:?}\n", e);
        }
        #[cfg(not(CONFIG_DEV_COREDUMP))]
        let _ = crashlog;

        if debug_enabled(DebugFlags::OopsOnGpuCrash) {
            panic!("GPU firmware crashed");
        } else {
            dev_err!(dev.as_ref(), "GPU firmware crashed, failing all jobs\n");
            data.event_manager.fail_all(workqueue::WorkError::NoDevice);
        }
    }

    fn shmem_alloc(
        data: <Self::Data as ForeignOwnable>::Borrowed<'_>,
        size: usize,
    ) -> Result<Self::Buffer> {
        let dev = &data.dev;
        mod_dev_dbg!(dev, "shmem_alloc() {:#x} bytes\n", size);

        let mut obj = gem::new_kernel_object(dev, size)?;
        let vmap = obj.gem.owned_vmap()?;
        let mapping = obj.map_into_range(
            data.uat.kernel_vm(),
            IOVA_KERN_RTKIT_RANGE,
            mmu::UAT_PGSZ as u64,
            mmu::PROT_FW_SHARED_RW,
            true,
        )?;
        mod_dev_dbg!(dev, "shmem_alloc() -> VA {:#x}\n", mapping.iova());
        Ok(RtkitObject { vmap, mapping })
    }

    // The 15.5 firmware asks for two RTKit buffers at addresses it has already
    // chosen (0x4000 at 0x1000176c000 and 0x1000 just after), rather than
    // asking us to allocate them -- these are its log buffers, carved out below
    // the start of Linux's RAM and still owned by the coprocessor. Earlier
    // firmwares only ever request iova = 0, so upstream implements only
    // shmem_alloc() and the default shmem_map() declines with EINVAL.
    //
    // Declining is deliberate, not a stub. The requested address is a physical
    // one, so honouring it means the DCP's approach -- memremap() the range and
    // hand RTKit a CPU pointer (see dcp_rtk_shmem_setup) -- not a GEM object
    // mapped into the kernel VM. Mapping it into the kernel VM instead was
    // tried and is worse than refusing: it wedges the firmware boot into a
    // timeout (-ETIME) where declining lets probe continue. RTKit clears the
    // buffer on error and carries on; the cost is firmware log output.
    fn shmem_map(
        data: <Self::Data as ForeignOwnable>::Borrowed<'_>,
        iova: usize,
        size: usize,
    ) -> Result<Self::Buffer> {
        dev_info!(
            data.dev.as_ref(),
            "RTKit asked to map {:#x} bytes at a fixed address {:#x}; declining (needs memremap, not a kernel VM mapping)\n",
            size,
            iova
        );
        Err(EINVAL)
    }
}

#[versions(AGX)]
impl GpuManager::ver {
    /// Create a new GpuManager of this version/GPU combination.
    #[inline(never)]
    pub(crate) fn new(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
    ) -> Result<Arc<GpuManager::ver>> {
        let uat = Self::make_uat(dev, cfg)?;
        let dyncfg = Self::make_dyncfg(dev, res, cfg, &uat)?;

        let mut alloc = KernelAllocators {
            private: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_PRIV_RANGE,
                0x80,
                mmu::PROT_FW_PRIV_RW,
                1024 * 1024,
                true,
                fmt!("Kernel Private"),
                true,
            )?,
            shared: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_SHARED_RANGE,
                0x80,
                mmu::PROT_FW_SHARED_RW,
                1024 * 1024,
                true,
                fmt!("Kernel Shared"),
                // 667zzf: keep freed shared objects out of reuse until the next
                // context invalidation / FW cache flush, like the private heap.
                // The firmware still writes some dead objects after the kernel
                // drops them: a reused slot became a new Scene and its
                // control_word2 lost bits 32..47 (0xfffffc2c006c0000 ->
                // 0xffff0000006c0000) -> FinalizeVertex data abort at fw:0x15f28.
                true,
            )?,
            shared_ro: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_SHARED_RO_RANGE,
                0x80,
                // 15.5 writes to these buffers. With the UAT_IAS fix in place
                // the firmware gets far enough to touch them and takes
                // ESR 0x9600004f -- a *permission* fault at level 3 -- on
                // 0xfffffe0a00004014, i.e. inside this range. Read-only is a
                // 13.5-ism; the InitData scratch buffers allocated from here
                // (IDTA/IDA8/IDB0) are read/write on this firmware.
                // TODO: narrow to the buffer that actually needs it, and gate
                // on version rather than applying to the whole allocator.
                mmu::PROT_FW_SHARED_RW,
                64 * 1024,
                true,
                fmt!("Kernel RO Shared"),
                false,
            )?,
            gpu: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_GPU_RANGE,
                0x80,
                mmu::PROT_GPU_FW_SHARED_RW,
                64 * 1024,
                true,
                fmt!("Kernel GPU Shared"),
                false,
            )?,
            gpu_ro: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_vm(),
                IOVA_KERN_GPU_RO_RANGE,
                0x80,
                mmu::PROT_GPU_RO_FW_PRIV_RW,
                1024 * 1024,
                true,
                fmt!("Kernel GPU RO Shared"),
                true,
            )?,
            klow: alloc::DefaultAllocator::new(
                dev,
                uat.kernel_lower_vm(),
                IOVA_KERN_GPU_KLOW_RANGE,
                0x80,
                mmu::PROT_GPU_SHARED_RW,
                1024 * 1024,
                true,
                fmt!("Kernel GPU Lower"),
                false,
            )?,
        };

        let event_manager = Self::make_event_manager(&mut alloc)?;
        let mut initdata = Self::make_initdata(dev, cfg, &dyncfg, &mut alloc)?;

        initdata.runtime_pointers.buffer_mgr_ctl_low_mapping =
            Some(initdata.runtime_pointers.buffer_mgr_ctl.map_at(
                uat.kernel_lower_vm(),
                IOVA_KERN_GPU_BUFMGR_LOW,
                mmu::PROT_GPU_SHARED_RW,
                false,
            )?);
        initdata.runtime_pointers.buffer_mgr_ctl_high_mapping =
            Some(initdata.runtime_pointers.buffer_mgr_ctl.map_at(
                uat.kernel_vm(),
                IOVA_KERN_GPU_BUFMGR_HIGH,
                mmu::PROT_FW_SHARED_RW,
                false,
            )?);

        // 32.41.508: one page, both address spaces -- the same trick as
        // `buffer_mgr_ctl` above. gPM reaches the counter through `0x1c910`,
        // which can only carry a LOW address; the firmware CPU reaches it
        // through `Scene.control_word2`, which needs the kernel VM.
        initdata.runtime_pointers.pm_counter_low_mapping =
            Some(initdata.runtime_pointers.pm_counter_obj.map_at(
                uat.kernel_lower_vm(),
                IOVA_KERN_GPU_PMCT_LOW,
                mmu::PROT_GPU_SHARED_RW,
                false,
            )?);
        initdata.runtime_pointers.pm_counter_high_mapping =
            Some(initdata.runtime_pointers.pm_counter_obj.map_at(
                uat.kernel_vm(),
                IOVA_KERN_GPU_PMCT_HIGH,
                mmu::PROT_FW_SHARED_RW,
                false,
            )?);

        initdata.runtime_pointers.uma_pool_ctl_low_mapping =
            Some(initdata.runtime_pointers.uma_pool_ctl.map_at(
                uat.kernel_lower_vm(),
                IOVA_KERN_GPU_UMAPOOL_LOW,
                mmu::PROT_GPU_SHARED_RW,
                false,
            )?);
        initdata.runtime_pointers.uma_pool_ctl_high_mapping =
            Some(initdata.runtime_pointers.uma_pool_ctl.map_at(
                uat.kernel_vm(),
                IOVA_KERN_GPU_UMAPOOL_HIGH,
                mmu::PROT_FW_SHARED_RW,
                false,
            )?);

        let mut mgr = Self::make_mgr(dev, cfg, dyncfg, uat, alloc, event_manager, initdata)?;

        {
            let fwctl = mgr.fwctl_channel.lock();
            let p_fwctl = fwctl.to_raw();
            core::mem::drop(fwctl);

            #[ver(V < V15_5)]
            mgr.as_mut()
                .initdata_mut()
                .fw_status
                .with_mut(|raw, _inner| {
                    raw.fwctl_channel = p_fwctl;
                });

            // 15.5 dropped the fw_status pointer from InitData and moved this
            // channel into the buffer at InitData+0xb0. The consumer at 0x2bb3c
            // reads it as a plain ChannelRing -- state at +0x4f78, ring at
            // +0x4f80 -- then treats state[0] as the read pointer and state[16]
            // as the write pointer, exactly matching FwCtlChannelState.
            // Leaving it zero NULL-faults at 0x2bb4c on the first submission.
            // LOG.md 30.9.
            #[ver(V >= V15_5)]
            {
                const FWCTL_OFF: usize = 0x4f78;
                let ring_size = core::mem::size_of_val(&p_fwctl);
                // The ChannelRing is two GPU pointers; copy it in as bytes
                // rather than reproducing the layout by hand.
                let src = &p_fwctl as *const _ as *const u8;
                let buf = mgr.as_mut().initdata_mut().unkbuf_b0.as_mut_slice();
                // SAFETY: p_fwctl is a live repr(C) value of ring_size bytes,
                // and the destination range is checked to be inside the buffer.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        src,
                        buf[FWCTL_OFF..FWCTL_OFF + ring_size].as_mut_ptr(),
                        ring_size,
                    )
                };
            }

            // b0[0x4f90] is the firmware's u64 RECOVERY COUNT, not a pointer
            // (.667zzz29): fw:0x5770 loads it into x26, 0x5bb0 copies it into
            // the recovery event packet, and 0x5bb4 reloads x26 = b0 before
            // the stores below; 0x5ec4 increments it per recovery, and the
            // doorbell msg 0x85 (host recovery) must carry its low 44 bits
            // (fw:0x319d8), so a self-pointer made host recovery impossible.
            // `b0_selfptr=1` restores the old value. Old (wrong) reading:
            // b0[0x4f90]: a self-pointer.
            //
            // The DC_Init path opens with `ldr x26, [b0 + 0x4f90]` (0x5770) and
            // then writes through x26 at +0x4ec0, +0x4ed4, +0x4ee4 ... +0x4f6c
            // -- the run of offsets immediately *below* the FWCtl slots this
            // driver already fills at +0x4f78/+0x4f80. A separate object would
            // not interleave with b0's own layout like that, so x26 is b0.
            //
            // Left NULL, the very first of those stores is a write through
            // zero. It never faults because the function never runs -- and that
            // function is what registers the EP 0x21 doorbell handler at
            // 0x60dc, which is why no doorbell has ever been dispatched and no
            // channel has ever been drained. LOG.md 32.16.
            // The gate on DC_Init. fn 0x38f3c loads *(RegionB+0x24c) and reads
            // a u32 at +0xc40; `cbz w9, 0x39024` skips the call to DC_Init
            // (0x38fec) when it is zero. DC_Init is where the EP 0x21 doorbell
            // handler gets registered (0x60dc), so while this word is zero no
            // doorbell is ever dispatched -- which matches the firmware's own
            // per-index doorbell counters at 0x1208fc reading zero. Meaning
            // unknown; 1 is the smallest thing that opens the gate.
            #[ver(V >= V15_5)]
            mgr.as_mut()
                .initdata_mut()
                .runtime_pointers
                .with_mut(|_raw, inner| {
                    let t = inner.pipe_tables.as_mut_slice();
                    t[0x4000 + 0xc40..0x4000 + 0xc44].copy_from_slice(&1u32.to_le_bytes());
                    // With pipe_stats, RegionB+0x24c is stats.frag: the gate
                    // word lives there instead. LOG.md 32.41.543.
                    if *crate::module_parameters::pipe_stats.value() != 0 {
                        inner.stats.vtx.with_mut(|raw, _| {
                            const _: () = assert!(
                                core::mem::size_of::<crate::fw::initdata::raw::GpuGlobalStatsVtx>() >= 0xc14
                            );
                            STATS_PTRS[0].store(raw as *mut _ as usize, Ordering::Release);
                        });
                        inner.stats.comp.with_mut(|raw, _| {
                            const _: () = assert!(
                                core::mem::size_of::<crate::fw::initdata::raw::GpuStatsComp>() >= 0xe08
                            );
                            STATS_PTRS[2].store(raw as *mut _ as usize, Ordering::Release);
                        });
                        inner.stats.frag.with_mut(|raw, _| {
                            const _: () = assert!(
                                core::mem::size_of::<crate::fw::initdata::raw::GpuGlobalStatsFrag::ver>()
                                    >= 0x1240
                            );
                            STATS_PTRS[1].store(raw as *mut _ as usize, Ordering::Release);
                            // SAFETY: in bounds (asserted above); plain u32 store.
                            unsafe {
                                let p = (raw as *mut _ as *mut u8).add(0xc40) as *mut u32;
                                p.write_unaligned(1);
                            }
                        });
                        // .667zm: keep pipe0 slot 63 non-idle. The firmware's
                        // power-up handler (fw:0x35e50) stops the pipes when all
                        // 64 slot states ([RegionB+0x244] + 0xc + slot*24) are
                        // 0/5; only the slot-event handler (fw:0x4804) marks
                        // one busy, and it never runs here, so every power-up
                        // raced StartFragment into a defer (fw:0x17b44) until the
                        // next 8.3 ms PM tick. LOG.md 32.41.667zm.
                        let busy = *crate::module_parameters::busy_slot63.value();
                        if busy == 0 && *crate::module_parameters::comp_busy.value() != 0 {
                            inner.stats.vtx.with_mut(|raw, _| {
                                // SAFETY: in bounds (asserted below); the object
                                // lives as long as the GPU manager.
                                let p = unsafe { (raw as *mut _ as *mut u8).add(0xc + 63 * 24) };
                                COMP_BUSY_PTR.store(p as usize, Ordering::Release);
                            });
                        }
                        if busy != 0 {
                            inner.stats.vtx.with_mut(|raw, _| {
                                const _: () = assert!(
                                    core::mem::size_of::<crate::fw::initdata::raw::GpuGlobalStatsVtx>()
                                        >= 0xc + 63 * 24 + 4
                                );
                                // SAFETY: in bounds (asserted above); plain u32 store.
                                unsafe {
                                    let p = (raw as *mut _ as *mut u8).add(0xc + 63 * 24) as *mut u32;
                                    p.write_unaligned(busy);
                                }
                            });
                        }
                    }
                });

            #[ver(V >= V15_5)]
            {
                const SELF_OFF: usize = 0x4f90;
                let b0_va: u64 = if *crate::module_parameters::b0_selfptr.value() != 0 {
                    mgr.initdata.unkbuf_b0.gpu_pointer().into()
                } else {
                    0
                };
                let buf = mgr.as_mut().initdata_mut().unkbuf_b0.as_mut_slice();
                buf[SELF_OFF..SELF_OFF + 8].copy_from_slice(&b0_va.to_le_bytes());
            }
        }

        {
            let txc = mgr.tx_channels.lock();
            let p_device_control = txc.device_control.to_raw();
            core::mem::drop(txc);

            let rxc = mgr.rx_channels.lock();
            let p_event = rxc.event.to_raw();
            let p_fw_log = rxc.fw_log.to_raw();
            let p_ktrace = rxc.ktrace.to_raw();
            let p_stats = rxc.stats.to_raw();
            let p_fwlog_buf = rxc.fw_log.get_buf();
            core::mem::drop(rxc);

            mgr.as_mut()
                .initdata_mut()
                .runtime_pointers
                .with_mut(|raw, _inner| {
                    // ChannelSlot is a plain ChannelRing before 15.5 and a
                    // four-pointer slot from 15.5 on, whose layout depends on
                    // which side produces; new_tx/new_rx handle both versions.
                    raw.device_control =
                        fw::initdata::raw::ChannelSlot::ver::new_tx(p_device_control);
                    #[ver(V < V15_5)]
                    {
                        raw.event = fw::initdata::raw::ChannelSlot::ver::new_rx(p_event);
                        raw.stats = fw::initdata::raw::ChannelSlot::ver::new_rx(p_stats);
                    }
                    #[ver(V < V15_5)]
                    {
                        raw.fw_log = fw::initdata::raw::ChannelSlot::ver::new_rx(p_fw_log);
                        raw.ktrace = fw::initdata::raw::ChannelSlot::ver::new_rx(p_ktrace);
                    }
                    // 15.5's four receive slots are bare 16-byte {state, ring}
                    // pairs at 0x1b8/0x1c8/0x1d8/0x1e8. See the field comments
                    // in fw/initdata.rs -- getting this wrong cost both the
                    // event channel and the stats channel.
                    #[ver(V >= V15_5)]
                    {
                        raw.event_15_5 = p_event;
                        raw.fw_log_15_5 = p_fw_log;
                        raw.ktrace_15_5 = p_ktrace;
                        raw.stats_15_5 = p_stats;
                    }
                    // 13.5 only: 15.5 has no second fwlog pointer here at all
                    // -- 0x2d8 is the UMA descriptor table's FW address.
                    // LOG.md 32.41.73.
                    #[ver(V < V15_5)]
                    {
                        raw.fwlog_buf = Some(p_fwlog_buf);
                    }
                    // 15.5's fwlog producer takes the payload ring from
                    // RegionB+0x1f8. Left wrong, it wrote 216-byte log records
                    // through whatever pointer happened to sit there.
                    #[ver(V >= V15_5)]
                    {
                        raw.fwlog_payload = Some(p_fwlog_buf);
                    }
                });
        }

        let mut p_pipes: KVec<fw::initdata::raw::PipeChannels::ver> = KVec::new();

        for ((v, f), c) in mgr
            .pipes
            .vtx
            .iter()
            .zip(&mgr.pipes.frag)
            .zip(&mgr.pipes.comp)
        {
            p_pipes.push(
                fw::initdata::raw::PipeChannels::ver {
                    vtx: fw::initdata::raw::ChannelSlot::ver::new_tx(v.lock().to_raw()),
                    frag: fw::initdata::raw::ChannelSlot::ver::new_tx(f.lock().to_raw()),
                    comp: fw::initdata::raw::ChannelSlot::ver::new_tx(c.lock().to_raw()),
                },
                GFP_KERNEL,
            )?;
        }

        mgr.as_mut()
            .initdata_mut()
            .runtime_pointers
            .with_mut(|raw, _inner| {
                for (i, p) in p_pipes.into_iter().enumerate() {
                    raw.pipes[i].vtx = p.vtx;
                    raw.pipes[i].frag = p.frag;
                    raw.pipes[i].comp = p.comp;
                }
            });

        // .667zzz25: the 15.5 firmware reads HwDataA in the 15.5 layout (0x6390
        // bytes); ours is 13.x's, so its DVFS clamps (fw:0x2154c +0xa28 in
        // [+0x9e0,+0x9e4], fw:0x2b430 +0x2bc0 in [+0x2b78,+0x2b7c], fw:0x2298c
        // +0x2148 in [+0x212c,+0x2130]) all read 0 and the arbitration at
        // fw:0x11cf0 pins the GPU to pstate 1 (338 MHz). hw/t6030-hwdataa-apple.bin
        // = Apple's AGXArmFirmware::initPowerAndPerformanceData run under Unicorn
        // (m3-gpu-display/kextemu.py) on this Mac's captured accelerator/firmware
        // objects; its first 0x600 bytes match the live capture except runtime
        // words. Kept ours: the PM tick (+0x4/+0x8, unless hwa_apple=2) and the
        // watch-block VA (+0x3a30).
        #[ver(V >= V15_5)]
        if *crate::module_parameters::hwa_apple.value() != 0 {
            static APPLE_HWA: &[u8] = include_bytes!("hw/t6030-hwdataa-apple.bin");
            let keep_tick = *crate::module_parameters::hwa_apple.value() != 2;
            mgr.as_mut()
                .initdata_mut()
                .runtime_pointers
                .hwdata_a
                .with_mut(|raw, _| {
                    let n = core::mem::size_of_val(raw).min(APPLE_HWA.len());
                    // SAFETY: plain-old-data firmware struct, rewritten as bytes.
                    let b = unsafe {
                        core::slice::from_raw_parts_mut(raw as *mut _ as *mut u8, n)
                    };
                    let mut tick = [0u8; 8];
                    tick.copy_from_slice(&b[4..12]);
                    let mut watch = [0u8; 8];
                    watch.copy_from_slice(&b[0x3a30..0x3a38]);
                    let mut diffs = 0;
                    for (d, a) in b.iter_mut().zip(APPLE_HWA.iter()) {
                        if *d != *a {
                            diffs += 1;
                        }
                        *d = *a;
                    }
                    if keep_tick {
                        b[4..12].copy_from_slice(&tick);
                    }
                    b[0x3a30..0x3a38].copy_from_slice(&watch);
                    pr_info!(
                        "asahi/hwa_apple: HwDataA <- Apple 15.5 ({} bytes, {} differed, tick {})\n",
                        n,
                        diffs,
                        if keep_tick { "ours" } else { "Apple" }
                    );
                });
        }

        // .667zzz24: live HwDataA fields for the stats sysfs files (pstate,
        // frequency, power) and the writable minimum pstate.
        mgr.initdata.runtime_pointers.hwdata_a.with(|raw, _| {
            let base = raw as *const _ as usize;
            crate::stats::set_uma_ctl(mgr.initdata.runtime_pointers.uma_pool_ctl_cpu);
            crate::stats::set_hwdata_a(crate::stats::HwaPtrs {
                base,
                size: core::mem::size_of_val(raw),
                actual_pstate: &raw.actual_pstate as *const _ as usize,
                tgt_pstate: &raw.tgt_pstate as *const _ as usize,
                cur_pstate: &raw.cur_pstate as *const _ as usize,
                min_pstate_scaled: &raw.min_pstate_scaled as *const _ as usize,
                max_pstate_scaled: &raw.max_pstate_scaled as *const _ as usize,
                // .667zzz26: 15.5 (both layouts): the firmware's live
                // frequency is an f32 in MHz at +0x64 and its filtered GPU
                // power (mW) is at +0x1d0 (869 idle, ~20300 under glheavy).
                freq_mhz: base + 0x64,
                cur_power_mw_2: base + 0x1d0,
                max_freq_mhz: mgr.dyncfg.pwr.max_freq_mhz as usize,
            });
        });

        // Dump the TX slots exactly as the firmware will read them. 15.5 wants
        // {&read_ptr, &read_ack, &write_ptr, ring}; if +0x00 and +0x10 ever
        // print the same value again, the aliasing bug is back and no channel
        // will ever report work. LOG.md 32.5.
        #[ver(V >= V15_5)]
        mgr.initdata.runtime_pointers.with(|raw, _inner| {
            // A closure cannot be generic over the message type, and every slot
            // carries a different one, so this has to be a macro.
            macro_rules! show {
                ($name:expr, $s:expr) => {
                    dev_info!(
                        dev.as_ref(),
                        "slot {}: rptr={:#x} rack={:#x} wptr={:#x} ring={:#x}\n",
                        $name,
                        $s.pair1.state.map(u64::from).unwrap_or(0),
                        $s.pair1.ring.map(u64::from).unwrap_or(0),
                        $s.pair2.state.map(u64::from).unwrap_or(0),
                        $s.pair2.ring.map(u64::from).unwrap_or(0),
                    )
                };
            }
            show!("devctl", raw.device_control);
            for i in 0..raw.pipes.len() {
                show!("vtx", raw.pipes[i].vtx);
                show!("frag", raw.pipes[i].frag);
                show!("comp", raw.pipes[i].comp);
            }
        });

        for (i, map) in cfg.io_mappings.iter().enumerate() {
            if let Some(map) = map.as_ref() {
                Self::iomap(&mut mgr, cfg, i, map)?;
            }
        }

        #[ver(V >= V13_0B4)]
        if let Some(base) = cfg.sram_base {
            let size = cfg.sram_size.unwrap();
            let iova = mgr.as_mut().alloc_mmio_iova(size);

            let mapping = mgr
                .uat
                .kernel_vm()
                .map_io(iova, base, size, mmu::PROT_FW_SHARED_RW)?;

            mgr.as_mut()
                .initdata_mut()
                .runtime_pointers
                .hwdata_b
                .with_mut(|raw, _| {
                    raw.sgx_sram_ptr = U64(mapping.iova());
                });

            mgr.as_mut().io_mappings_mut().push(mapping, GFP_KERNEL)?;
        }

        // LOG.md 32.41.546: Apple's live HwDataB (macOS 15.5, this Mac,
        // 2026-09-26 capture), wholesale, keeping only what must be ours: the
        // timestamp-area VA (+0x28), each io_mapping's firmware VA (+8),
        // sgx_sram_ptr and uat_ttb_base.
        #[ver(V >= V15_5)]
        if *crate::module_parameters::hwb_apple.value() != 0 {
            static APPLE_HWB: &[u8] = include_bytes!("hw/t6030-hwdatab-apple.bin");
            mgr.as_mut()
                .initdata_mut()
                .runtime_pointers
                .hwdata_b
                .with_mut(|raw, _| {
                    let n = core::mem::size_of_val(raw).min(APPLE_HWB.len());
                    // SAFETY: plain-old-data struct, rewritten as bytes.
                    let b = unsafe {
                        core::slice::from_raw_parts_mut(raw as *mut _ as *mut u8, n)
                    };
                    let mut keep: KVec<(usize, [u8; 8])> = KVec::new();
                    let mut save = |o: usize, b: &[u8]| {
                        let mut v = [0u8; 8];
                        v.copy_from_slice(&b[o..o + 8]);
                        let _ = keep.push((o, v), GFP_KERNEL);
                    };
                    save(0x28, b);
                    save(0xed8, b);
                    save(0x1000, b);
                    for i in 0..55usize {
                        save(0x640 + i * 40 + 8, b);
                    }
                    let mut diffs = 0;
                    for (i, (d, a)) in b.iter_mut().zip(APPLE_HWB.iter()).enumerate() {
                        if *d != *a {
                            diffs += 1;
                            let _ = i;
                        }
                        *d = *a;
                    }
                    for (o, v) in keep.iter() {
                        b[*o..*o + 8].copy_from_slice(v);
                    }
                    pr_info!("asahi/hwb_apple: HwDataB <- Apple ({} bytes, {} differed)\n", n, diffs);
                });
        }

        // LOG.md 32.41.547: the 0x9800 object macOS's capture called
        // "HwDataA" (AGXArmFirmware+0x2b0) is Globals (InitData+0x20):
        // fw:0x2d48 reads `ldur w12, [globals, #0x56]`. 15.5 packs it from
        // 0x50; ours is the 13.5 layout. Apple's live 0..0x8000, wholesale
        // (0x8000.. is a firmware-written VA log).
        #[ver(V >= V15_5)]
        if *crate::module_parameters::glob_apple.value() != 0 {
            static APPLE_GLOB: &[u8] = include_bytes!("hw/t6030-globals-apple.bin");
            mgr.as_mut()
                .initdata_mut()
                .globals
                .with_mut(|raw, _| {
                    let n = core::mem::size_of_val(raw).min(0x8000);
                    // SAFETY: plain-old-data firmware struct, rewritten as bytes.
                    let b = unsafe {
                        core::slice::from_raw_parts_mut(raw as *mut _ as *mut u8, n)
                    };
                    let mut diffs = 0;
                    for (d, a) in b.iter_mut().zip(APPLE_GLOB[..n].iter()) {
                        if *d != *a {
                            diffs += 1;
                        }
                        *d = *a;
                    }
                    pr_info!("asahi/glob_apple: Globals <- Apple ({} bytes, {} differed)\n", n, diffs);
                });
        }

        // .667zu: fw:0x2df4 copies Globals+0x9a8/0x9ac/0x9b0 to 0x1a38d8..; the
        // PM power state machine (fw:0xe8a8) waits [0x1a38e0] * 1000 us before
        // powering off. Apple's live Globals has 2/40/5 there. Print ours, and
        // optionally poke one word (glob_poke_off/glob_poke_val).
        #[ver(V >= V15_5)]
        mgr.as_mut().initdata_mut().globals.with_mut(|raw, _| {
            let n = core::mem::size_of_val(raw);
            // SAFETY: plain-old-data firmware struct, accessed as bytes.
            let b = unsafe { core::slice::from_raw_parts_mut(raw as *mut _ as *mut u8, n) };
            let w = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            // 667zzy: 15.5 reads the per-pipe progress-check intervals (the
            // scheduler's timeslice quanta) from Globals+0x98c (3D), +0x990 (TA)
            // and +0x994 (CL), copied at fw:0x2e08 to 0x1a38c4..cc and compared
            // per 1 ms tick at fw:0x6d4c. Ours sat at the 13.x offsets and these
            // were 0, so every pipe hit its quantum on every tick: a compute job
            // longer than ~1 ms was preempted and never resumed (Vulkan/GL compute
            // > ~512K invocations hung). Apple's live Globals: 40 / 10 / 250.
            //
            // 667zzz36: with Apple's quanta, a pipe whose context outlives its
            // quantum while other work waits gets a GTP pause (fw:0x6c00, the
            // 1 ms tick). Once one pause trips the GIN-1072 erratum check
            // (fw:0x7518: pause status low 3 bits all set) the firmware sets
            // "gtp pausing disabled" (0x1a2c6c) for good, and every later pause
            // fails 5 times and ends in a GPU timeout (fw:0x74c4 -> 0x78d4,
            // "gin_1072_gtp_pausing_disabled 1"): Minecraft hit one every ~90 s
            // after 17 min, then host recovery failed. 2 (default) = quanta the
            // tick never reaches: contexts switch at job boundaries only; hung
            // jobs are still caught by the host job timeout.
            let pcheck = *crate::module_parameters::glob_pcheck.value();
            if pcheck != 0 && n >= 0x998 {
                let q: [u32; 3] = if pcheck == 2 { [0x3fff_ffff; 3] } else { [40, 10, 250] };
                for (o, v) in [(0x98c, q[0]), (0x990, q[1]), (0x994, q[2])] {
                    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            let off = *crate::module_parameters::glob_poke_off.value() as usize;
            if off != 0 && off + 4 <= n {
                let v = *crate::module_parameters::glob_poke_val.value();
                pr_info!("asahi/glob_poke: Globals+{:#x} {:#x} -> {:#x}\n", off, w(b, off), v);
                b[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
            pr_info!(
                "asahi/glob: +0x9a8 {:#x} +0x9ac {:#x} +0x9b0 {:#x} (PM power-off delay ms) +0x9b4 {:#x} +0x9b8 {:#x} +0x9bc {:#x}\n",
                w(b, 0x9a8), w(b, 0x9ac), w(b, 0x9b0), w(b, 0x9b4), w(b, 0x9b8), w(b, 0x9bc)
            );
        });

        let mgr = Arc::from(mgr);

        let rtkit = rtkit::RtKit::<GpuManager::ver>::new(dev.as_ref(), None, 0, mgr.clone())?;

        *mgr.rtkit.lock() = Some(rtkit);

        {
            let mut rxc = mgr.rx_channels.lock();
            rxc.event.set_manager(mgr.clone());
        }

        Ok(mgr)
    }

    /// Return a mutable reference to the initdata member
    fn initdata_mut(
        self: Pin<&mut Self>,
    ) -> &mut fw::types::GpuObject<fw::initdata::InitData::ver> {
        // SAFETY: initdata does not require structural pinning.
        unsafe { &mut self.get_unchecked_mut().initdata }
    }

    /// Return a mutable reference to the io_mappings member
    fn io_mappings_mut(self: Pin<&mut Self>) -> &mut KVec<mmu::KernelMapping> {
        // SAFETY: io_mappings does not require structural pinning.
        unsafe { &mut self.get_unchecked_mut().io_mappings }
    }

    /// Allocate an MMIO iova range
    fn alloc_mmio_iova(self: Pin<&mut Self>, size: usize) -> u64 {
        // SAFETY: next_mmio_iova does not require structural pinning.
        let next_ref = unsafe { &mut self.get_unchecked_mut().next_mmio_iova };

        let addr = *next_ref;
        let next = addr + (size + mmu::UAT_PGSZ) as u64;

        assert!(next <= IOVA_KERN_MMIO_RANGE.end);

        *next_ref = next;

        addr
    }

    /// Build the entire GPU InitData structure tree and return it as a boxed GpuObject.
    fn make_initdata(
        dev: &AsahiDevice,
        cfg: &'static hw::HwConfig,
        dyncfg: &hw::DynConfig,
        alloc: &mut KernelAllocators,
    ) -> Result<KBox<fw::types::GpuObject<fw::initdata::InitData::ver>>> {
        let mut builder = initdata::InitDataBuilder::ver::new(dev, alloc, cfg, dyncfg);
        builder.build()
    }

    /// Create a fresh boxed Uat instance.
    ///
    /// Force disable inlining to avoid blowing up the stack.
    #[inline(never)]
    fn make_uat(dev: &AsahiDevice, cfg: &'static hw::HwConfig) -> Result<KBox<mmu::Uat>> {
        // G14X has a new thing in the Scene structure that unfortunately requires
        // write access from user contexts. Hopefully it's not security-sensitive.
        #[ver(G >= G14X)]
        let map_kernel_to_user = true;
        #[ver(G < G14X)]
        let map_kernel_to_user = false;

        Ok(KBox::new(
            mmu::Uat::new(dev, cfg, map_kernel_to_user)?,
            GFP_KERNEL,
        )?)
    }

    /// Actually create the final GpuManager instance, as a UniqueArc.
    ///
    /// Force disable inlining to avoid blowing up the stack.
    #[inline(never)]
    fn make_mgr(
        dev: &AsahiDevice,
        cfg: &'static hw::HwConfig,
        dyncfg: KBox<hw::DynConfig>,
        uat: KBox<mmu::Uat>,
        mut alloc: KernelAllocators,
        event_manager: Arc<event::EventManager>,
        initdata: KBox<fw::types::GpuObject<fw::initdata::InitData::ver>>,
    ) -> Result<Pin<UniqueArc<GpuManager::ver>>> {
        let mut pipes = PipeChannels::ver {
            vtx: KVec::new(),
            frag: KVec::new(),
            comp: KVec::new(),
        };

        for _i in 0..=NUM_PIPES - 1 {
            pipes.vtx.push(
                KBox::pin_init(
                    new_mutex!(channel::PipeChannel::ver::new(dev, &mut alloc)?, "pipe_vtx",),
                    GFP_KERNEL,
                )?,
                GFP_KERNEL,
            )?;
            pipes.frag.push(
                KBox::pin_init(
                    new_mutex!(
                        channel::PipeChannel::ver::new(dev, &mut alloc)?,
                        "pipe_frag",
                    ),
                    GFP_KERNEL,
                )?,
                GFP_KERNEL,
            )?;
            pipes.comp.push(
                KBox::pin_init(
                    new_mutex!(
                        channel::PipeChannel::ver::new(dev, &mut alloc)?,
                        "pipe_comp",
                    ),
                    GFP_KERNEL,
                )?,
                GFP_KERNEL,
            )?;
        }

        let fwctl_channel = channel::FwCtlChannel::new(dev, &mut alloc)?;

        let buffer_mgr = buffer::BufferManager::ver::new()?;
        let event_manager_clone = event_manager.clone();
        let buffer_mgr_clone = buffer_mgr.clone();
        let alloc_ref = &mut alloc;
        let rx_channels = KBox::init(
            try_init!(RxChannels::ver {
                event: channel::EventChannel::ver::new(
                    dev,
                    alloc_ref,
                    event_manager_clone,
                    buffer_mgr_clone,
                )?,
                fw_log: channel::FwLogChannel::new(dev, alloc_ref)?,
                ktrace: channel::KTraceChannel::ver::new(dev, alloc_ref)?,
                stats: channel::StatsChannel::ver::new(dev, alloc_ref)?,
            }),
            GFP_KERNEL,
        )?;

        let alloc_ref = &mut alloc;
        let tx_channels = KBox::init(
            try_init!(TxChannels::ver {
                device_control: channel::DeviceControlChannel::ver::new(dev, alloc_ref)?,
            }),
            GFP_KERNEL,
        )?;

        let x = UniqueArc::pin_init(
            try_pin_init!(GpuManager::ver {
                dev: dev.into(),
                cfg,
                dyncfg: KBox::<hw::DynConfig>::into_inner(dyncfg),
                initdata: KBox::<fw::types::GpuObject<fw::initdata::InitData::ver>>::into_inner(initdata),
                uat: KBox::<mmu::Uat>::into_inner(uat),
                io_mappings: KVec::new(),
                next_mmio_iova: IOVA_KERN_MMIO_RANGE.start,
                rtkit <- new_mutex!(None, "rtkit"),
                crashed: AtomicBool::new(false),
                event_manager,
                alloc <- new_mutex!(alloc, "alloc"),
                fwctl_channel <- new_mutex!(fwctl_channel, "fwctl_channel"),
                rx_channels <- new_mutex!(KBox::<RxChannels::ver>::into_inner(rx_channels), "rx_channels"),
                tx_channels <- new_mutex!(KBox::<TxChannels::ver>::into_inner(tx_channels), "tx_channels"),
                pipes,
                buffer_mgr,
                ids: Default::default(),
                garbage_contexts <- new_mutex!(KVec::new(), "garbage_contexts"),
            }),
            GFP_KERNEL,
        )?;

        Ok(x)
    }

    fn load_hwdata_blob(dev: &AsahiDevice, name: &CStr, size_name: &CStr) -> Result<KVVec<u8>> {
        let of_node = dev.as_ref().of_node().ok_or(EINVAL)?;
        let size: usize = dev
            .as_ref()
            .fwnode()
            .ok_or(ENOENT)?
            .property_read::<u32>(size_name)
            .or(0)
            .try_into()?;
        let res = of_node.reserved_mem_region_to_resource_byname(name)?;
        // SAFETY: No dma here, just loading init data.
        let mem = unsafe { Mem::try_new(res, (MemFlag::WB).into())? };
        if size > mem.size() {
            return Err(ENOENT);
        }
        // SAFETY: trusting the bootloader to fill it out correctly
        let blob_sl = unsafe { slice::from_raw_parts(mem.ptr(), size) };
        let mut blob = KVVec::new();
        blob.extend_from_slice(blob_sl, GFP_KERNEL)?;
        Ok(blob)
    }

    /// Fetch and validate the GPU dynamic configuration from the device tree and hardware.
    ///
    /// Force disable inlining to avoid blowing up the stack.
    #[inline(never)]
    fn make_dyncfg(
        dev: &AsahiDevice,
        res: &regs::Resources,
        cfg: &'static hw::HwConfig,
        uat: &mmu::Uat,
    ) -> Result<KBox<hw::DynConfig>> {
        let gpu_id = res.get_gpu_id()?;

        dev_info!(dev.as_ref(), "GPU Information:\n");
        dev_info!(
            dev.as_ref(),
            "  Type: {:?}{:?}\n",
            gpu_id.gpu_gen,
            gpu_id.gpu_variant
        );
        dev_info!(dev.as_ref(), "  Clusters: {}\n", gpu_id.num_clusters);
        dev_info!(
            dev.as_ref(),
            "  Cores: {} ({})\n",
            gpu_id.num_cores,
            gpu_id.num_cores * gpu_id.num_clusters
        );
        dev_info!(
            dev.as_ref(),
            "  Frags: {} ({})\n",
            gpu_id.num_frags,
            gpu_id.num_frags * gpu_id.num_clusters
        );
        dev_info!(
            dev.as_ref(),
            "  GPs: {} ({})\n",
            gpu_id.num_gps,
            gpu_id.num_gps * gpu_id.num_clusters
        );
        dev_info!(dev.as_ref(), "  Core masks: {:#x?}\n", gpu_id.core_masks);
        dev_info!(
            dev.as_ref(),
            "  Active cores: {}\n",
            gpu_id.total_active_cores
        );

        dev_info!(dev.as_ref(), "Getting configuration from device tree...\n");
        let pwr_cfg = hw::PwrConfig::load(dev, cfg)?;
        dev_info!(dev.as_ref(), "Dynamic configuration fetched\n");

        if gpu_id.gpu_gen != cfg.gpu_gen || gpu_id.gpu_variant != cfg.gpu_variant {
            dev_err!(
                dev.as_ref(),
                "GPU type mismatch (expected {:?}{:?}, found {:?}{:?})\n",
                cfg.gpu_gen,
                cfg.gpu_variant,
                gpu_id.gpu_gen,
                gpu_id.gpu_variant
            );
            return Err(EIO);
        }
        if gpu_id.num_clusters > cfg.max_num_clusters {
            dev_err!(
                dev.as_ref(),
                "Too many clusters ({} > {})\n",
                gpu_id.num_clusters,
                cfg.max_num_clusters
            );
            return Err(EIO);
        }
        if gpu_id.num_cores > cfg.max_num_cores {
            dev_err!(
                dev.as_ref(),
                "Too many cores ({} > {})\n",
                gpu_id.num_cores,
                cfg.max_num_cores
            );
            return Err(EIO);
        }
        if gpu_id.num_frags > cfg.max_num_frags {
            dev_err!(
                dev.as_ref(),
                "Too many frags ({} > {})\n",
                gpu_id.num_frags,
                cfg.max_num_frags
            );
            return Err(EIO);
        }
        if gpu_id.num_gps > cfg.max_num_gps {
            dev_err!(
                dev.as_ref(),
                "Too many GPs ({} > {})\n",
                gpu_id.num_gps,
                cfg.max_num_gps
            );
            return Err(EIO);
        }

        let fwnode = dev.as_ref().fwnode().ok_or(ENOENT)?;

        Ok(KBox::new(
            hw::DynConfig {
                pwr: pwr_cfg,
                uat_ttb_base: uat.ttb_base(),
                id: gpu_id,
                firmware_version: fwnode
                    .property_read_array_vec(c_str!("apple,firmware-version"), 3)?
                    .or(kernel::kvec![0; 3]?),

                hw_data_a: Self::load_hwdata_blob(
                    dev,
                    c_str!("hw-cal-a"),
                    c_str!("debug,hw-cal-a-size"),
                )
                .unwrap_or(KVVec::new()),
                hw_data_b: Self::load_hwdata_blob(
                    dev,
                    c_str!("hw-cal-b"),
                    c_str!("debug,hw-cal-b-size"),
                )
                .unwrap_or(KVVec::new()),
                hw_globals: Self::load_hwdata_blob(
                    dev,
                    c_str!("globals"),
                    c_str!("debug,globals-size"),
                )
                .unwrap_or(KVVec::new()),
            },
            GFP_KERNEL,
        )?)
    }

    /// Create the global GPU event manager, and return an `Arc<>` to it.
    fn make_event_manager(alloc: &mut KernelAllocators) -> Result<Arc<event::EventManager>> {
        Ok(Arc::new(event::EventManager::new(alloc)?, GFP_KERNEL)?)
    }

    /// Create a new MMIO mapping and add it to the mappings list in initdata at the specified
    /// index.
    fn iomap(
        this: &mut Pin<UniqueArc<GpuManager::ver>>,
        cfg: &'static hw::HwConfig,
        index: usize,
        map: &hw::IOMapping,
    ) -> Result {
        let dies = if map.per_die {
            cfg.num_dies as usize
        } else {
            1
        };

        let off = map.base & mmu::UAT_PGMSK;
        let base = map.base - off;
        let end = (map.base + map.size + mmu::UAT_PGMSK) & !mmu::UAT_PGMSK;
        let map_size = end - base;

        // Array mappings must be aligned
        assert!((off == 0 && map_size == map.size) || (map.count == 1 && !map.per_die));
        assert!(map.count > 0);

        let iova = this.as_mut().alloc_mmio_iova(map_size * map.count * dies);
        let mut cur_iova = iova;

        for die in 0..dies {
            for i in 0..map.count {
                let phys_off = die * 0x20_0000_0000 + i * map.stride;

                let mapping = this.uat.kernel_vm().map_io(
                    cur_iova,
                    base + phys_off,
                    map_size,
                    if map.writable {
                        mmu::PROT_FW_MMIO_RW
                    } else {
                        mmu::PROT_FW_MMIO_RO
                    },
                )?;

                this.as_mut().io_mappings_mut().push(mapping, GFP_KERNEL)?;
                cur_iova += map_size as u64;
            }
        }

        this.as_mut()
            .initdata_mut()
            .runtime_pointers
            .hwdata_b
            .with_mut(|raw, _| {
                raw.io_mappings[index] = fw::initdata::raw::IOMapping::ver {
                    phys_addr: U64(map.base as u64),
                    virt_addr: U64(iova + off as u64),

                    #[ver(V < V15_5)]
                    total_size: (map.size * map.count * dies) as u32,
                    #[ver(V < V15_5)]
                    element_size: map.size as u32,
                    #[ver(V < V15_5)]
                    readwrite: U64(map.writable as u64),

                    // 15.5 widened the size fields and shrank readwrite.
                    #[ver(V >= V15_5)]
                    total_size: U64((map.size * map.count * dies) as u64),
                    #[ver(V >= V15_5)]
                    element_size: U64(map.size as u64),
                    #[ver(V >= V15_5)]
                    readwrite: map.writable as u32,
                    #[ver(V >= V15_5)]
                    pad_24: Default::default(),
                };
            });

        Ok(())
    }

    /// Mark work associated with currently in-progress event slots as failed, after a fault or
    /// timeout.
    fn mark_pending_events(&self, culprit_slot: Option<u32>, error: workqueue::WorkError) {
        dev_err!(self.dev.as_ref(), "  Pending events:\n");

        self.initdata.globals.with(|raw, _inner| {
            for (index, i) in raw.pending_stamps.iter().enumerate() {
                let info = i.info.load(Ordering::Relaxed);
                let wait_value = i.wait_value.load(Ordering::Relaxed);

                if info & 1 != 0 {
                    #[ver(V >= V13_5)]
                    let slot = (info >> 4) & 0x7f;
                    #[ver(V < V13_5)]
                    let slot = (info >> 3) & 0x7f;
                    #[ver(V >= V13_5)]
                    let flags = info & 0xf;
                    #[ver(V < V13_5)]
                    let flags = info & 0x7;
                    dev_err!(
                        self.dev.as_ref(),
                        "    [{}:{}] flags={} value={:#x}\n",
                        index,
                        slot,
                        flags,
                        wait_value
                    );
                    let error = if culprit_slot.is_some() && culprit_slot != Some(slot) {
                        workqueue::WorkError::Killed
                    } else {
                        error
                    };
                    self.event_manager.mark_error(slot, wait_value, error);
                    i.info.store(0, Ordering::Relaxed);
                    i.wait_value.store(0, Ordering::Relaxed);
                }
            }
        });
    }

    /// Probe a user VA in a bound VM. See `Uat::probe_slot`.
    ///
    /// Exposed so the submit path can ask "is Mesa's program actually where the
    /// address we are about to program says it is", without waiting for a fault
    /// that may name a derived address rather than the one at fault.
    /// Dump the firmware's vertex stats block. A kernel page-table read, so
    /// safe to call while the job is in flight -- unlike every MMIO probe.
    /// LOG.md 32.41.337.
    pub(crate) fn dump_stats_now(&self, tag: &str) {
        let n = *crate::module_parameters::dump_stats_vtx.value() as usize;
        if n == 0 {
            return;
        }
        for (which, addr) in [
            ("vtx", DBG_STATS_VTX.load(Ordering::Relaxed)),
            ("frag", DBG_STATS_FRAG.load(Ordering::Relaxed)),
            ("comp", DBG_STATS_COMP.load(Ordering::Relaxed)),
        ] {
            if addr != 0 {
                self.uat.dump_kern(&self.dev, addr, n, which);
            }
        }
        let _ = tag;
    }

    /// Sample the windowed GPU state register, selector saved and restored.
    pub(crate) fn dump_state_now(&self, tag: &str) {
        (*self.dev).resources.dump_state_pre(tag);
    }

    /// Non-perturbing in-flight state poll -- see `Resources::poll_state_bare`.
    pub(crate) fn poll_state_now(&self, tag: &str, n: u32, ms: u32) {
        (*self.dev).resources.poll_state_bare(tag, n, ms);
    }

    pub(crate) fn probe_vm(&self, slot: u32, va: u64, tag: &str) {
        self.uat.probe_slot(&self.dev, slot as usize, va, tag);
    }

    /// Write one 64-bit word into a user buffer, for the `cs_poke_*` diagnostic.
    pub(crate) fn poke_vm(&self, slot: u32, va: u64, val: u64) {
        self.uat.poke_slot(&self.dev, slot as usize, va, val);
    }

    pub(crate) fn patch_vm(
        &self,
        slot: u32,
        va: u64,
        count: usize,
        mask: u32,
        val: u32,
        or_val: u32,
        and_val: u32,
    ) {
        self.uat
            .patch_slot(&self.dev, slot as usize, va, count, mask, val, or_val, and_val);
    }

    pub(crate) fn dump_vm(&self, slot: u32, va: u64, count: usize, tag: &str) {
        self.uat.dump_slot(&self.dev, slot as usize, va, count, tag);
    }

    pub(crate) fn fix_count_vm(&self, slot: u32, va: u64, count: usize) {
        self.uat.fix_count_slot(&self.dev, slot as usize, va, count);
    }

    pub(crate) fn read_vm(&self, slot: u32, va: u64, out: &mut [u32]) -> usize {
        self.uat.read_slot(&self.dev, slot as usize, va, out)
    }

    pub(crate) fn splice_vm(
        &self,
        slot: u32,
        va: u64,
        count: usize,
        mask: u32,
        val: u32,
        writes: &[(usize, u32)],
    ) {
        self.uat
            .splice_slot(&self.dev, slot as usize, va, count, mask, val, writes);
    }

    pub(crate) fn poke_fill_vm(&self, slot: u32, va: u64, word: u32, count: usize) {
        self.uat.poke_fill_slot(&self.dev, slot as usize, va, word, count);
    }

    /// Dump the scene's TVB tilemap and heap metadata for the last submission.
    ///
    /// Called from `handle_timeout` and again when the job is dropped. Once the
    /// control-stream runaway is fixed there is no timeout, so the timeout-only
    /// probe stopped firing exactly when the answer became interesting: a
    /// non-zero tilemap is direct evidence the vertex stage executed, and
    /// "no fault" on its own cannot distinguish that from a skipped draw.
    /// LOG.md 32.41.174.
    pub(crate) fn probe_scene(&self, tag: &str) {
        let slot = DBG_SCENE_SLOT.load(Ordering::Relaxed);
        if slot == !0 {
            return;
        }
        let tm = DBG_SCENE_TILEMAP.load(Ordering::Relaxed);
        let hm = DBG_SCENE_HEAPMETA.load(Ordering::Relaxed);
        let bias = 1u64 << *crate::module_parameters::user_va_shift.value();
        if tm != 0 {
            self.probe_vm(slot as u32, tm, tag);
            self.probe_vm(slot as u32, tm + bias, tag);
        }
        if hm != 0 {
            self.probe_vm(slot as u32, hm, tag);
        }
        // 🔬 The tiler's own bookkeeping, which says whether it ran at all.
        // `Scene` carries `pass_page_count` at +0x10 and `total_page_count` at
        // +0x44 (measured, 32.41.231); `buffer::Info` carries `block_count` at
        // +0x38 and `block_ctl` at +0x48. A zero tilemap cannot distinguish "the
        // tiler ran and emitted nothing" from "the tiler never ran"; a non-zero
        // page or block count can.  LOG.md 32.41.234.
        let so = DBG_SCENE_OBJ.load(Ordering::Relaxed);
        if so != 0 {
            self.probe_vm(slot as u32, so, "scene obj");
            self.probe_vm(slot as u32, so + 0x40, "scene obj+0x40");
        }
        let si = DBG_SCENE_INFO.load(Ordering::Relaxed);
        if si != 0 {
            self.probe_vm(slot as u32, si + 0x20, "info+0x20");
            self.probe_vm(slot as u32, si + 0x40, "info+0x40");
        }
        // .667zzz33: everything below dereferences CPU pointers published by the
        // LAST submitted scene, which may belong to a process that has exited
        // since: a hang dump oopsed on a freed vmap (GLES31 run), and the dead
        // timeout worker left drm_sched_fini waiting forever. Debug only.
        if *crate::module_parameters::probe_raw.value() == 0 {
            return;
        }
        // The counters themselves, read from the CPU side. Non-zero movement
        // here is direct evidence the tiler allocated TVB pages, which is
        // upstream of anything being binned. LOG.md 32.41.236.
        let rd = |p: u64| -> i64 {
            if p == 0 {
                -1
            } else {
                // SAFETY: the pointer was published by the in-flight job, which
                // owns the scene and its buffer for the duration of this probe.
                (unsafe { &*(p as *const core::sync::atomic::AtomicU32) })
                    .load(Ordering::Relaxed) as i64
            }
        };
        {
            let c = DBG_FRAGREGS.load(Ordering::Relaxed);
            if c != 0 {
                // SAFETY: published by the in-flight job, which owns the command
                // for the duration of this probe. 150 records of 12 bytes.
                let b = unsafe { core::slice::from_raw_parts(c as *const u8, 150 * 12) };
                for i in 0..150 {
                    let o = i * 12;
                    let reg = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                    let mut v = [0u8; 8];
                    v.copy_from_slice(&b[o + 4..o + 12]);
                    let val = u64::from_le_bytes(v);
                    if reg != 0 || val != 0 {
                        pr_info!("asahi/fragpost {} [{:3}] {:#07x} = {:#018x}\n", tag, i, reg, val);
                    }
                }
            }
        }
        dump_fragcmd(tag);
        crate::replay::draw_va0_dump(tag);
        crate::replay::draw_rt_dump(tag);
        // 667i: not here -- the noted scene buffers may belong to a scene that is
        // already freed (Oops); the DONE path in render.rs dumps them while alive.
        {
            let c = DBG_VTXCMD.load(Ordering::Relaxed);
            if c != 0 {
                // SAFETY: published by the in-flight job, which owns the command
                // for the duration of this probe.
                let q = unsafe { core::slice::from_raw_parts(c as *const u64, 0x920 / 8) };
                pr_info!("asahi/vtxpost {} size 0x920\n", tag);
                let mut i = 0usize;
                while i < q.len() {
                    pr_info!(
                        "asahi/vtxpost {} {:#06x}: {:#018x} {:#018x} {:#018x} {:#018x}\n",
                        tag, i * 8, q[i], q[i + 1], q[i + 2], q[i + 3],
                    );
                    i += 4;
                }
            }
        }
        {
            let t = DBG_TS_OBJ.load(Ordering::Relaxed);
            if t != 0 {
                // SAFETY: published by the in-flight job, which owns the
                // timestamps object for the duration of this probe.
                let q = |o: usize| -> u64 {
                    (unsafe { &*((t + o as u64) as *const core::sync::atomic::AtomicU64) })
                        .load(Ordering::Relaxed)
                };
                pr_info!(
                    "asahi/ts {}: vtx.start {:#x} vtx.end {:#x} frag.start {:#x} frag.end {:#x}\n",
                    tag, q(0), q(8), q(16), q(24),
                );
            } else {
                pr_info!("asahi/ts {}: not published\n", tag);
            }
        }
        pr_info!(
            "asahi/tiler {}: pm_counter {} pass_pages {} total_pages {} blocks {} pages {}\n",
            tag,
            rd(DBG_CNT_PMCT.load(Ordering::Relaxed)),
            rd(DBG_CNT_PASS.load(Ordering::Relaxed)),
            rd(DBG_CNT_TOTAL.load(Ordering::Relaxed)),
            rd(DBG_CNT_BLOCKS.load(Ordering::Relaxed)),
            rd(DBG_CNT_PAGES.load(Ordering::Relaxed)),
        );
        {
            // buffer::Info (15.5): page_count is at +0x30. LOG.md 32.41.600.
            let pc = DBG_CNT_PAGES.load(Ordering::Relaxed);
            if pc != 0 {
                // SAFETY: the scene/buffer is held by the in-flight job.
                let w = unsafe { core::slice::from_raw_parts((pc - 0x30) as *const u32, 0x80 / 4) };
                for r in 0..8 {
                    pr_info!(
                        "asahi/info {} +{:#04x}: {:08x} {:08x} {:08x} {:08x}\n",
                        tag, r * 16, w[r * 4], w[r * 4 + 1], w[r * 4 + 2], w[r * 4 + 3],
                    );
                }
            }
        }
        {
            let pr = DBG_POOLREC.load(Ordering::Relaxed);
            if pr != 0 {
                // SAFETY: the scene owning the record is held by the in-flight job.
                let w = unsafe { core::slice::from_raw_parts(pr as *const u32, 0x50 / 4) };
                for r in 0..5 {
                    pr_info!(
                        "asahi/poolrec {} +{:#04x}: {:08x} {:08x} {:08x} {:08x}\n",
                        tag, r * 16, w[r * 4], w[r * 4 + 1], w[r * 4 + 2], w[r * 4 + 3],
                    );
                }
            }
        }
    }

    /// Fetch the GPU MMU fault information from the hardware registers.
    fn get_fault_info(&self) -> Option<regs::FaultInfo> {
        let res = &(*self.dev).resources;

        let info = res.get_fault_info(self.cfg);
        if info.is_some() {
            dev_err!(
                self.dev.as_ref(),
                "  Fault info: {:#x?}\n",
                info.as_ref().unwrap()
            );
            // The reported vm_slot is a TTBAT index; print the table so a
            // mismatch against the slot the driver bound is visible in the same
            // capture. LOG.md 32.41.38.
            self.uat.dump_ttbs(8);
            crate::channel::ktrace_hist_tail(80);
            // Classify the address so a fault above the 42-bit input address
            // size is not mistaken for a mapping problem: the walker rejects it
            // at level 0 and reports NoAccess rather than Unmapped, which reads
            // like a permission bug and is not one. LOG.md 32.41.62.
            if let Some(i) = info.as_ref() {
                mmu::describe_iova(&self.dev, i.address);
                // The faulting address and its un-aliased twin. With
                // `user_va_shift` on, every user page is mapped at both `va`
                // and `va + 2^36`; if the low address is unmapped too then the
                // pointer the GPU formed was never a valid allocation, and the
                // alias machinery is not at fault. LOG.md 32.41.102.
                let slot = i.vm_slot as usize;
                mmu::Uat::probe_slot(&self.uat, &self.dev, slot, i.address, "fault");
                let bias = 1u64 << 36;
                if i.address >= bias {
                    mmu::Uat::probe_slot(
                        &self.uat,
                        &self.dev,
                        slot,
                        i.address - bias,
                        "fault-2^36",
                    );
                }
            }
            if *crate::module_parameters::dump_uma_regs.value() != 0 {
                res.dump_uma();
            }
            // Service the outstanding 0xc020 unit requests by hand, exactly as
            // the firmware's own path would, and see whether that moves the
            // unit out of state 3. The firmware never gets here itself: the
            // handler that samples units into its request table is not
            // dispatched on this machine. LOG.md 32.41.290, 32.41.291.
            let ucp = *crate::module_parameters::unit_cmd_probe.value();
            if ucp != 0 {
                res.unit_cmd_probe(u64::from(ucp - 1));
            }
            // Has the firmware written a UMA page pool descriptor into the
            // table published at RegionB+0x2d0/0x2d8? All-zero means it is
            // using some other table and ours is decoration. LOG.md 32.41.77.
            let p = self.initdata.runtime_pointers.uma_pool_ctl_cpu;
            if p != 0 {
                for row in 0..4 {
                    // SAFETY: `uma_pool_ctl` is a live 0x4000-byte kernel object
                    // whose VMap is owned by its ObjectRef; 0x80 bytes is well
                    // inside it.
                    let w = |k: usize| unsafe {
                        core::ptr::read_volatile((p + row * 32 + k * 8) as *const u64)
                    };
                    dev_err!(
                        self.dev.as_ref(),
                        "  uma_pool_ctl[{:#04x}]: {:#018x} {:#018x} {:#018x} {:#018x}\n",
                        row * 32,
                        w(0),
                        w(1),
                        w(2),
                        w(3),
                    );
                }
            }
        }
        info
    }

    /// Resume the GPU firmware after it halts (due to a timeout, fault, or request).
    fn recover(&self) -> bool {
        // 15.5: firmware-initiated recovery is the b0[0x4fa0] handshake (fw:0x5ed8,
        // kext AGXArmFirmware::isFirmwareStoppedFor{Pre,Post}Recovery /
        // resumeFirmware{Pre,Post}Recovery): fw posts 1 and spins; host writes 2;
        // fw flushes, resets its pipe state, posts 3 and spins; host writes 0.
        // The 13.x halted/resume flags are never set. LOG.md 32.41.667zzh.
        #[ver(V >= V15_5)]
        {
            let b0 = self.initdata.unkbuf_b0.as_slice();
            // SAFETY: b0 is a live firmware-shared buffer of at least 0x5000 bytes;
            // 0x4fa0 is the u32 request word both sides poll.
            let req = unsafe { b0.as_ptr().add(0x4fa0) as *mut u32 };
            let rd = || unsafe { core::ptr::read_volatile(req) };
            let wait_for = |want: u32, ms: i64| {
                let start = Instant::<Monotonic>::now();
                while rd() != want {
                    if start.elapsed() > Delta::from_millis(ms) {
                        return false;
                    }
                    mem::sync();
                }
                true
            };
            if !wait_for(1, 200) {
                dev_err!(self.dev.as_ref(), "  recovery: no b0 request (b0[0x4fa0] = {:#x}); cannot recover\n", rd());
                return false;
            }
            dev_err!(self.dev.as_ref(), "  recovery: firmware stopped (pre), resuming\n");
            // SAFETY: see above.
            unsafe { core::ptr::write_volatile(req, 2) };
            mem::sync();
            if !wait_for(3, 2000) {
                dev_err!(self.dev.as_ref(), "  recovery: firmware never reached post-recovery (b0[0x4fa0] = {:#x})\n", rd());
                return false;
            }
            // SAFETY: see above.
            unsafe { core::ptr::write_volatile(req, 0) };
            mem::sync();
            dev_err!(self.dev.as_ref(), "  recovery: firmware resumed\n");
            return true;
        }
        #[allow(unreachable_code)]
        let mut ok = false;
        self.initdata.fw_status.with(|raw, _inner| {
            let halt_count = raw.flags.halt_count.load(Ordering::Relaxed);
            let mut halted = raw.flags.halted.load(Ordering::Relaxed);
            dev_err!(self.dev.as_ref(), "  Halt count: {}\n", halt_count);
            dev_err!(self.dev.as_ref(), "  Halted: {}\n", halted);

            if halted == 0 {
                let start = Instant::<Monotonic>::now();
                while start.elapsed() < HALT_ENTER_TIMEOUT {
                    halted = raw.flags.halted.load(Ordering::Relaxed);
                    if halted != 0 {
                        break;
                    }
                    mem::sync();
                }
                halted = raw.flags.halted.load(Ordering::Relaxed);
            }

            if debug_enabled(DebugFlags::NoGpuRecovery) {
                dev_crit!(
                    self.dev.as_ref(),
                    "  GPU recovery is disabled, wedging forever!\n"
                );
            } else if halted != 0 {
                dev_err!(self.dev.as_ref(), "  Attempting recovery...\n");
                raw.flags.halted.store(0, Ordering::SeqCst);
                raw.flags.resume.store(1, Ordering::SeqCst);
                ok = true;
            } else {
                dev_err!(self.dev.as_ref(), "  Cannot recover.\n");
            }
        });
        ok
    }

    /// Return the packed GPU enabled core masks.
    // Only used for some versions
    #[allow(dead_code)]
    pub(crate) fn core_masks_packed(&self) -> &[u32] {
        self.dyncfg.id.core_masks_packed.as_slice()
    }

    /// Drain every receive ring.
    ///
    /// Upstream only ever calls this from `recv_message`, i.e. when the
    /// firmware rings MSG_RX_DOORBELL. On 15.5 that is not enough to keep the
    /// link alive. The firmware sends exactly four doorbells, all during init;
    /// after that its statistics producer fills the 256-entry ring at
    /// RegionB+0x1e8, prints "Waiting for space to add statistics!" and blocks
    /// -- and a blocked producer cannot ring the doorbell that would make the
    /// AP drain it. Everything else the coprocessor does stops with it, which
    /// is why no message sent after init was ever dispatched. Polling from the
    /// submission path breaks the deadlock from our side. LOG.md 32.41.
    fn poll_rx(&self) {
        let mut ch = self.rx_channels.lock();

        ch.fw_log.poll();
        ch.ktrace.poll();
        let stats = ch.stats.poll();
        let events = ch.event.poll();
        core::mem::drop(ch);

        // Rate-limited: the statistics ring refills fast enough on 15.5 that a
        // line per poll wraps the kernel log in seconds and takes everything
        // else with it.
        let total = RX_STATS_TOTAL.fetch_add(stats as u64, Ordering::Relaxed) + stats as u64;
        let polls = RX_POLLS.fetch_add(1, Ordering::Relaxed) + 1;
        if events != 0 || polls % 512 == 0 {
            cls_pr_debug!(
                Debug0,
                "asahi: poll_rx: {} polls, {} stats total, {} events this poll\n",
                polls,
                total,
                events
            );
        }
    }

    /// Drain the receive rings that cannot call back into this driver.
    ///
    /// The event ring can: a Fault or Timeout message runs `handle_fault` /
    /// `handle_timeout`, which take the tx_channels lock and walk the work
    /// queues. So it must not be drained from the submission path or from
    /// under tx_channels. Statistics is the ring that actually blocks the
    /// firmware, and its handler only logs, so draining these three is both
    /// sufficient and safe anywhere. See poll_rx.
    fn poll_rx_nocb(&self) {
        // try_lock: run_job's caller holds the WorkQueue lock, and poll_rx holds
        // rx_channels while it signals WorkQueues -> blocking here is an ABBA
        // deadlock (666j). A busy rx_channels means someone is draining already.
        let mut ch = match self.rx_channels.try_lock() {
            Some(ch) => ch,
            None => return,
        };

        ch.fw_log.poll();
        ch.ktrace.poll();
        let stats = ch.stats.poll();
        core::mem::drop(ch);

        RX_STATS_TOTAL.fetch_add(stats as u64, Ordering::Relaxed);
        let polls = RX_POLLS_NOCB.fetch_add(1, Ordering::Relaxed) + 1;
        if polls % 64 == 0 {
            cls_pr_debug!(
                Debug0,
                "asahi: poll_rx_nocb: {} polls, {} stats this poll\n",
                polls,
                stats
            );
        }
    }

    /// Kick a submission pipe for a submitted job to tell the firmware to start processing it.
    pub(crate) fn run_job(&self, job: workqueue::JobSubmission::ver<'_>) -> Result {
        mod_dev_dbg!(self.dev, "GPU: run_job\n");

        // Before anything else: give the firmware room on the receive rings.
        // See poll_rx -- if the statistics ring is full the coprocessor is
        // blocked and will not look at the pipe we are about to kick.
        self.poll_rx_nocb();

        let pipe_type = job.pipe_type();
        mod_dev_dbg!(self.dev, "GPU: run_job: pipe_type={:?}\n", pipe_type);

        let pipes = match pipe_type {
            PipeType::Vertex => &self.pipes.vtx,
            PipeType::Fragment => &self.pipes.frag,
            PipeType::Compute => &self.pipes.comp,
        };

        let index: usize = job.priority() as usize;
        let mut pipe = pipes.get(index).ok_or(EIO)?.lock();

        // The b0 handshake probe (LOG.md 32.28): DC_Init posts a request code
        // in b0[0x4fa0] and spins until the AP zeroes it. A non-zero value here
        // means the coprocessor is parked waiting on us, which would also
        // explain a transport that stops delivering after InitData.
        #[ver(V >= V15_5)]
        {
            let b0 = self.initdata.unkbuf_b0.as_slice();
            let req = u32::from_le_bytes(b0[0x4fa0..0x4fa4].try_into().unwrap());
            let param = u64::from_le_bytes(b0[0x4fb0..0x4fb8].try_into().unwrap());
            if req != 0 {
                dev_err!(
                    self.dev.as_ref(),
                    "GPU: firmware is parked on the b0 handshake: req {:#x} param {:#x}\n",
                    req,
                    param
                );
            }
        }

        mod_dev_dbg!(self.dev, "GPU: run_job: run()\n");
        job.run(&mut pipe);
        mod_dev_dbg!(self.dev, "GPU: run_job: ring doorbell\n");

        let mut guard = self.rtkit.lock();
        let mut rtk = guard.as_mut().as_pin_mut().unwrap();
        // Re-arm the link before every kick. See MSG_MGMT_SET_IOP_PWR_ON: the
        // quiesce happens while InitData is being processed, so sending this
        // once from init() is too early to matter.
        #[ver(V >= V15_5)]
        {
            rtk.as_mut().send_message(EP_MGMT, MSG_MGMT_PWR_REF)?;
            rtk.as_mut().send_message(EP_MGMT, MSG_MGMT_SET_IOP_PWR_ON)?;
        }
        // 667zc: doorbell type 0x86 = `AGXArmFirmware::kickFirmware(idx, _, wake=true)`: the firmware's decoder
        // (fw:0x31910, type 6) posts PM-thread event bit 36 -> PM case 0x24 (fw:0x38048). Plain 0x83 kicks only
        // post a queue scan, which fw:0x32c58 drops while the GPU sits in its idle-stopped state, so work waited
        // for the next 8.33 ms DVFS tick. Apple's own trace shows PM event 0x24.
        if *crate::module_parameters::kick_wake.value() != 0 {
            rtk.as_mut().send_message(EP_DOORBELL, MSG_TX_WAKE | ((index as u64) << 2))?;
        }
        rtk.as_mut().send_message(
            EP_DOORBELL,
            MSG_TX_DOORBELL | pipe_type as u64 | ((index as u64) << 2),
        )?;
        core::mem::drop(guard);

        // Ring the fragment doorbell a second time, after the vertex has had
        // time to retire.
        //
        // The firmware offers the fragment 43 us after the vertex *starts*,
        // defers it correctly (a TBDR cannot begin 3D before the TA phase
        // ends), and then never re-scans the pipe: once the vertex retires it
        // goes straight to the idle/register-block reset at `fw:0x4e44` and
        // reports a Timeout with `event_slot: -1`. Nothing in the trace after
        // the vertex stamp is written looks like a scheduler pass. So give it
        // an explicit reason to make one. LOG.md 32.41.124.
        if pipe_type == PipeType::Vertex && *crate::module_parameters::refrag_us.value() != 0 {
            // The window is narrow: the vertex retires ~120us after its
            // doorbell and the firmware resets the register block ~100us after
            // that, so a single re-ring at +1000us lands after the reset and
            // does nothing. Ring repeatedly across the whole window instead.
            let us = *crate::module_parameters::refrag_us.value();
            let n = *crate::module_parameters::refrag_n.value();
            for _ in 0..n {
                kernel::time::delay::fsleep(kernel::time::Delta::from_micros(us as i64));
                let mut guard = self.rtkit.lock();
                let mut rtk = guard.as_mut().as_pin_mut().unwrap();
                rtk.as_mut().send_message(
                    EP_DOORBELL,
                    MSG_TX_DOORBELL | PipeType::Fragment as u64 | ((index as u64) << 2),
                )?;
                core::mem::drop(guard);
            }
            dev_err!(
                self.dev.as_ref(),
                "asahi/refrag: rang the fragment doorbell {} more times, every {}us\n",
                n,
                us
            );
        }

        // The GPU state register the fragment is gated on: `fw:0x166fc` reads
        // SGX 0xc020 and refuses to start fragment work unless
        // `(v & 0x1e) == 0x1a`, deferring the command instead (LOG.md
        // 32.41.114). Read it here, right after the doorbell, because that is
        // the one moment the GPU is certainly powered -- reading this window
        // when it is not wedges the CPU on the fabric with no fault and no
        // response to an NMI (32.41.97), which is why this sits behind
        // `dump_uma_regs` like every other SGX read.
        // Watch the PRC perf-state handshake across the whole job. The firmware
        // writes `state | 0x10` to SGX+0xd4c004 and spins until bit 4 clears;
        // it never does, so the scheduler never runs again and the deferred
        // fragment is never re-offered. Record every distinct value so the
        // exact moment the request is raised is visible. LOG.md 32.41.123.
        if *crate::module_parameters::prc_watch.value() != 0 && pipe_type == PipeType::Vertex {
            let n = *crate::module_parameters::prc_watch.value() as usize;
            let mut seen: [(u32, u32, u32); 16] = [(0, 0, 0); 16];
            let mut k = 0usize;
            let mut last = (!0u32, !0u32);
            for i in 0..n {
                let v = (*self.dev).resources.prc();
                if v != last {
                    if k < seen.len() {
                        seen[k] = (i as u32, v.0, v.1);
                        k += 1;
                    }
                    last = v;
                }
                kernel::time::delay::fsleep(kernel::time::Delta::from_micros(20));
            }
            for i in 0..k {
                dev_err!(
                    self.dev.as_ref(),
                    "asahi/prc +{}us: 0xd4c000 = {:#010x} 0xd4c004 = {:#010x}\n",
                    seen[i].0 * 20,
                    seen[i].1,
                    seen[i].2
                );
            }
        }
        // 32.41.573: ask the firmware for its own USC pipe state dump.
        #[ver(V >= V15_5)]
        if pipe_type == PipeType::Compute && *crate::module_parameters::fw_usc_dump.value() != 0 {
            let p = self.initdata.unkbuf_b0.as_slice().as_ptr() as *mut u32;
            // SAFETY: IDB0 is a live 0x20000-byte shared buffer; word 0 is the
            // firmware's dump-request flag (fw:0x35a00 reads, 0x35b5c clears).
            unsafe { core::ptr::write_volatile(p, 1) };
            dev_info!(self.dev.as_ref(), "asahi/fwuscdump: IDB0[0] = 1\n");
        }
        // 32.41.572d: compute twin -- CDM per-pipe state and the replay output.
        if pipe_type == PipeType::Compute && *crate::module_parameters::usc_poll.value() != 0 {
            // 32.41.573c: sweep slots once the CDM is running (poll a bit first).
            (*self.dev)
                .resources
                .fender_gating(*crate::module_parameters::fender_nogate.value() != 0);
            let tun = *crate::module_parameters::agx_tun.value();
            if tun != 0 {
                (*self.dev).resources.agx_tunables(tun == 2);
            }
            (*self.dev).resources.cdm_poll(2000);
            (*self.dev).resources.usc_slot_sweep("cdm-running");
            for round in 0..8u32 {
                (*self.dev).resources.cdm_poll(*crate::module_parameters::usc_poll.value() / 8);
                crate::replay::comp_spill_snap(if round == 0 { "r0" } else if round < 4 { "r1-3" } else { "r4-7" });
            }
            crate::replay::comp_replay_dump("after poll");
            crate::replay::comp_replay_diff("after poll");
        }
        if pipe_type == PipeType::Vertex && *crate::module_parameters::tmap_poll.value() != 0 {
            crate::gpu::tmap_poll(*crate::module_parameters::tmap_poll.value(), &(*self.dev).resources);
        }
        if pipe_type == PipeType::Vertex && *crate::module_parameters::ta_poll.value() != 0 {
            (*self.dev)
                .resources
                .ta_poll(*crate::module_parameters::ta_poll.value());
        }
        // 32.41.666: usc_poll stays on for compute (needed there); on the vertex path it was
        // pure instrumentation spinning in the doorbell path on every kick (flat 62.7 ms/frame).
        if pipe_type == PipeType::Vertex
            && *crate::module_parameters::usc_poll.value() != 0
            && *crate::module_parameters::vtx_usc_poll.value() != 0
        {
            (*self.dev)
                .resources
                .usc_poll(*crate::module_parameters::usc_poll.value());
        }
        if *crate::module_parameters::dump_uma_regs.value() != 0 {
            let v = (*self.dev).resources.sgx_state();
            let ready = if (v & 0x1e) == 0x1a { 1 } else { 0 };
            dev_err!(
                self.dev.as_ref(),
                "asahi/state after {:?} kick: 0xc020 = {:#018x} state {} flag {} fragready {}\n",
                pipe_type,
                v,
                (v >> 2) & 7,
                (v >> 1) & 1,
                ready
            );
            // A single sample right after the doorbell is too early: the GPU
            // block is still gated (the whole window reads 0 until the firmware
            // powers it up, which is why the pre-render dump reads 0 for every
            // register and the fault-time dump reads real values). The
            // interesting moment is ~100us later, when the scheduler evaluates
            // the fragment and `fw:0x16708` compares this against 0x1a. Sample
            // across that window and print only the transitions.
            //
            // Done after the *vertex* kick, which is the last one the default
            // order issues, so nothing is delayed that the firmware is waiting
            // on. LOG.md 32.41.117.
            if pipe_type == PipeType::Vertex && *crate::module_parameters::dump_uma_regs.value() >= 2
            {
                // Record (selector, state) pairs, not the state alone: 0xc020
                // is windowed by 0x10398, so the same value means different
                // things for different units. The question this answers is
                // whether *any* unit is ever in the state 5 or state 6 the
                // fragment gate demands. LOG.md 32.41.124.
                // ⚠️ 8 ms, not more. The GPU powers itself down a few hundred
                // microseconds after the vertex retires, and a read of this
                // window with the block gated hangs the CPU on the fabric with
                // no fault and no NMI response (32.41.97). A 400 ms sweep
                // killed the machine mid-loop and produced nothing.
                let mut seen: [u64; 24] = [0; 24];
                let mut ts: [u64; 24] = [0; 24];
                let mut at: [u32; 24] = [0; 24];
                let mut n = 0usize;
                let mut last = (!0u32, !0u64);
                // Sampled in the same loop: 0xc128, the scheduler's
                // pending-event bitmap. 32.41.129 established that the
                // firmware only re-enters the scheduler while it is non-zero,
                // so a fragment that is deferred and never re-offered is
                // exactly what a 0xc128 that never gets a bit set would
                // produce. Tracked separately from 0xc020 because it is *not*
                // windowed by 0x10398 and so does not need the selector.
                let mut ev: [u64; 24] = [0; 24];
                let mut evat: [u32; 24] = [0; 24];
                let mut evbusy: [u64; 24] = [0; 24];
                let mut en = 0usize;
                let mut lastev = (!0u64, !0u64);
                for i in 0..400u32 {
                    let sel = (*self.dev).resources.sgx_sel();
                    let s = (*self.dev).resources.sgx_state();
                    if (sel, s) != last {
                        if n < seen.len() {
                            seen[n] = s;
                            ts[n] = (sel as u64) << 32 | ((s >> 1) & 0xf);
                            at[n] = i * 20;
                            n += 1;
                        }
                        last = (sel, s);
                    }
                    // ⚠️ Only when the block is powered. `s == 0` is the
                    // whole SGX window reading zero, which is what a
                    // power-gated block looks like (32.41.121). Reading
                    // `0xc020` in that state is survivable -- every previous
                    // run has done it -- but reading `0xc128`/`0xc120` there
                    // hangs the CPU on the fabric with no fault and no NMI
                    // response, and cost a wedged machine on 2026-09-02.
                    // Skipping the sample loses nothing: a gated GPU has no
                    // events to report.
                    let e = if s != 0 {
                        (*self.dev).resources.sgx_events()
                    } else {
                        lastev
                    };
                    if e != lastev {
                        if en < ev.len() {
                            ev[en] = e.0;
                            evbusy[en] = e.1;
                            evat[en] = i * 20;
                            en += 1;
                        }
                        lastev = e;
                    }
                    kernel::time::delay::fsleep(kernel::time::Delta::from_micros(20));
                }
                for i in 0..en {
                    dev_err!(
                        self.dev.as_ref(),
                        "asahi/events +{}us: 0xc128 = {:#018x}  0xc120 = {:#018x}\n",
                        evat[i],
                        ev[i],
                        evbusy[i]
                    );
                }
                // Second, coarser pass: the GPU timestamp counter (0xc028) is a
                // clock, so if the block is alive at all it must advance
                // between these samples. A frozen counter next to a "busy"
                // 0xc020 says the clock domain stopped, not that work is
                // queued. LOG.md 32.41.121.
                for i in 0..n {
                    dev_err!(
                        self.dev.as_ref(),
                        "asahi/state +{}us: sel {} 0xc020 = {:#018x}  (&0x1e = {:#04x}, want 0x1a or &0x1c 0x14) state {}\n",
                        at[i],
                        ts[i] >> 32,
                        seen[i],
                        seen[i] & 0x1e,
                        ts[i] & 0xf
                    );
                }
            }
        }

        mod_dev_dbg!(self.dev, "GPU: run_job: done\n");

        Ok(())
    }

    pub(crate) fn start_op(self: &Arc<GpuManager::ver>) -> Result<OpGuard> {
        if self.is_crashed() {
            return Err(ENODEV);
        }

        let val = self
            .initdata
            .globals
            .with(|raw, _inner| raw.pending_submissions.fetch_add(1, Ordering::Acquire));

        mod_dev_dbg!(self.dev, "OP start (pending: {})\n", val + 1);
        self.kick_firmware()?;

        Ok(OpGuard(self.clone()))
    }

    fn invalidate_context(
        &self,
        context: &fw::types::GpuObject<fw::workqueue::GpuContextData>,
    ) -> Result {
        mod_dev_dbg!(
            self.dev,
            "Invalidating GPU context @ {:?}\n",
            context.weak_pointer()
        );

        if self.is_crashed() {
            return Err(ENODEV);
        }

        let mut guard = self.alloc.lock();
        let (garbage_count, _) = guard.private.garbage();
        let (garbage_count_gpuro, _) = guard.gpu_ro.garbage();
        let (garbage_count_shared, _) = guard.shared.garbage();

        // 15.5 renumbered the device-control ids: tag 0x18 (what the enum
        // position gives DestroyContext) is now a timeout setter (fw:0x339ec),
        // so contexts were never destroyed and nothing was flushed before the
        // garbage was reused. The real handler is DC 0x14 (fw:0x33030) with a
        // compact layout. LOG.md 32.41.667zzg.
        #[ver(V >= V15_5)]
        if *crate::module_parameters::dc_destroy_155.value() != 0 {
            let dc = context.with(|raw, _inner| {
                Self::destroy_context_dc_155(
                    raw.unk_23,
                    raw.unk_0,
                    raw.unk_1,
                    raw.unk_4,
                    u64::from(context.weak_pointer()),
                )
            });
            return self.send_destroy_context(dc, &mut guard, garbage_count, garbage_count_gpuro, garbage_count_shared, context);
        }

        let dc = context.with(
            |raw, _inner| fw::channels::DeviceControlMsg::ver::DestroyContext {
                unk_4: 0,
                ctx_23: raw.unk_23,
                #[ver(V < V13_3)]
                __pad0: Default::default(),
                unk_c: U32(0),
                unk_10: U32(0),
                ctx_0: raw.unk_0,
                ctx_1: raw.unk_1,
                ctx_4: raw.unk_4,
                #[ver(V < V13_3)]
                __pad1: Default::default(),
                #[ver(V < V13_3)]
                unk_18: 0,
                gpu_context: Some(context.weak_pointer()),
                __pad2: Default::default(),
            },
        );

        self.send_destroy_context(dc, &mut guard, garbage_count, garbage_count_gpuro, garbage_count_shared, context)
    }

    fn destroy_context_dc_old() -> fw::channels::DeviceControlMsg::ver {
        fw::channels::DeviceControlMsg::ver::DestroyContext {
            unk_4: 0,

            ctx_23: 0,
            #[ver(V < V13_3)]
            __pad0: Default::default(),
            unk_c: U32(0),
            unk_10: U32(0),
            ctx_0: 0xff,
            ctx_1: 0xff,
            ctx_4: 0,
            #[ver(V < V13_3)]
            __pad1: Default::default(),
            #[ver(V < V13_3)]
            unk_18: 0,
            gpu_context: None,
            __pad2: Default::default(),
        }
    }

    /// 15.5 DestroyContext: DC 0x14, {unk_4 u32, ctx_23, ctx_0, ctx_1, ctx_4, gpu_context u64}
    /// at +4 (fw:0x33030). ctx_0/ctx_1 == 0xff: cache flush only (fw:0x33108).
    #[ver(V >= V15_5)]
    fn destroy_context_dc_155(
        ctx_23: u8,
        ctx_0: u8,
        ctx_1: u8,
        ctx_4: u8,
        ptr: u64,
    ) -> fw::channels::DeviceControlMsg::ver {
        let mut a: crate::fw::types::Array<{ fw::channels::DEVICECONTROL_SZ::ver }, u8> =
            Default::default();
        a[4] = ctx_23;
        a[5] = ctx_0;
        a[6] = ctx_1;
        a[7] = ctx_4;
        for (i, b) in ptr.to_le_bytes().iter().enumerate() {
            a[8 + i] = *b;
        }
        fw::channels::DeviceControlMsg::ver::Unk13(a)
    }

    fn send_destroy_context(
        &self,
        dc: fw::channels::DeviceControlMsg::ver,
        guard: &mut Guard<'_, KernelAllocators, MutexBackend>,
        garbage_count: usize,
        garbage_count_gpuro: usize,
        garbage_count_shared: usize,
        context: &fw::types::GpuObject<fw::workqueue::GpuContextData>,
    ) -> Result {
        mod_dev_dbg!(self.dev, "Context invalidation command: {:?}\n", &dc);

        let mut txch = self.tx_channels.lock();

        let token = txch.device_control.send(&dc);

        {
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
        }

        // A full statistics ring blocks the firmware before it ever looks at
        // device control, so drain first and then again on each retry. This is
        // the wait that used to time out at 1 s on the first context
        // invalidation after driver load. See poll_rx.
        #[ver(V >= V15_5)]
        {
            self.poll_rx_nocb();
        }
        let mut res = txch.device_control.wait_for(token);
        #[ver(V >= V15_5)]
        for _ in 0..4 {
            if res.is_ok() {
                break;
            }
            self.poll_rx_nocb();
            res = txch.device_control.wait_for(token);
        }
        res?;

        mod_dev_dbg!(
            self.dev,
            "GPU context invalidated: {:?}\n",
            context.weak_pointer()
        );

        // The invalidation does a cache flush, so it is okay to collect garbage
        guard.private.collect_garbage(garbage_count);
        guard.gpu_ro.collect_garbage(garbage_count_gpuro);
        guard.shared.collect_garbage(garbage_count_shared);

        Ok(())
    }

    #[cfg(CONFIG_DEV_COREDUMP)]
    fn generate_crashdump(&self, crashlog: Option<&[u8]>) -> Result {
        // Lock the allocators, to block kernel/FW memory mutations (mostly)
        let kalloc = self.alloc();
        let pages = self.uat.dump_kernel_pages()?;
        core::mem::drop(kalloc);

        let mut crashdump = crate::crashdump::CrashDumpBuilder::new(pages)?;
        let initdata_addr = self.initdata.gpu_va().get();
        crashdump.add_agx_info(self.cfg, &self.dyncfg, initdata_addr)?;
        if let Some(crashlog) = crashlog {
            crashdump.add_crashlog(crashlog)?;
        }
        let crashdump = KBox::new(crashdump.finalize()?, GFP_KERNEL)?;

        devcoredump::dev_coredump(
            self.dev.as_ref(),
            &crate::THIS_MODULE,
            crashdump,
            GFP_KERNEL,
            msecs_to_jiffies(60 * 60 * 1000),
        );

        Ok(())
    }
}

#[versions(AGX)]
impl GpuManager for GpuManager::ver {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn arc_as_any(self: Arc<Self>) -> Arc<dyn Any + Sync + Send> {
        self as Arc<dyn Any + Sync + Send>
    }

    fn init(&self) -> Result {
        // The firmware tells us how big it thinks a command is: the DAG walker
        // invalidates 0x980 bytes from the ring entry at 0x19178, and reads the
        // sequence word it checks at +0x8bc with the byte at +0x8c0. Print what
        // the driver actually built so the two can be compared. LOG.md 32.41.
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi: command sizes: vertex {:#x} fragment {:#x} compute {:#x}\n",
            core::mem::size_of::<fw::vertex::raw::RunVertex::ver<'_>>(),
            core::mem::size_of::<fw::fragment::raw::RunFragment::ver<'_>>(),
            core::mem::size_of::<fw::compute::raw::RunCompute::ver<'_>>(),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunVertex: tag {:#x} counter {:#x} vm_slot {:#x} unk_8 {:#x} notifier {:#x} buffer_slot {:#x}\\n",
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, tag),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, counter),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, vm_slot),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_8),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, notifier),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, buffer_slot),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off StartVertex: registers {:#x} work_queue {:#x} unk_3c_ptr {:#x} event_generation {:#x} unk_job_buf {:#x} attachments {:#x} pool_ptr {:#x} counter {:#x} notifier_buf {:#x} size {:#x}\n",
            core::mem::offset_of!(fw::microseq::StartVertex::ver<'_>, registers),
            core::mem::offset_of!(fw::microseq::StartVertex::ver<'_>, work_queue),
            core::mem::offset_of!(fw::microseq::StartVertex::ver<'_>, unk_3c_ptr),
            core::mem::offset_of!(fw::microseq::StartVertex::ver<'_>, event_generation),
            core::mem::offset_of!(fw::microseq::StartVertex::ver<'_>, unk_job_buf),
            core::mem::offset_of!(fw::microseq::StartVertex::ver<'_>, attachments),
            core::mem::offset_of!(fw::microseq::StartVertex::ver<'_>, pool_ptr),
            core::mem::offset_of!(fw::microseq::StartVertex::ver<'_>, counter),
            core::mem::offset_of!(fw::microseq::StartVertex::ver<'_>, notifier_buf),
            core::mem::size_of::<fw::microseq::StartVertex::ver<'_>>(),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off FinalizeVertex: scene {:#x} buffer {:#x} stats {:#x} work_queue {:#x} unk_34_ptr {:#x} unk_40 {:#x} fw_stamp {:#x} stamp_value {:#x} unk_78_ptr {:#x} restart {:#x} has_attachments {:#x} unk_85_ptr {:#x} size {:#x}\n",
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, scene),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, buffer),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, stats),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, work_queue),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, unk_34_ptr),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, unk_40),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, fw_stamp),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, stamp_value),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, unk_78_ptr),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, restart_branch_offset),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, has_attachments),
            core::mem::offset_of!(fw::microseq::FinalizeVertex::ver, unk_85_ptr),
            core::mem::size_of::<fw::microseq::FinalizeVertex::ver>(),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off StartFragment: registers {:#x} scene {:#x} work_queue {:#x} work_item {:#x} vm_slot {:#x} unk_98 {:#x} attachments {:#x} pool_ptr {:#x} notifier_buf {:#x} size {:#x}\n",
            core::mem::offset_of!(fw::microseq::StartFragment::ver<'_>, registers),
            core::mem::offset_of!(fw::microseq::StartFragment::ver<'_>, scene),
            core::mem::offset_of!(fw::microseq::StartFragment::ver<'_>, work_queue),
            core::mem::offset_of!(fw::microseq::StartFragment::ver<'_>, work_item),
            core::mem::offset_of!(fw::microseq::StartFragment::ver<'_>, vm_slot),
            core::mem::offset_of!(fw::microseq::StartFragment::ver<'_>, unk_98),
            core::mem::offset_of!(fw::microseq::StartFragment::ver<'_>, attachments),
            core::mem::offset_of!(fw::microseq::StartFragment::ver<'_>, pool_ptr),
            core::mem::offset_of!(fw::microseq::StartFragment::ver<'_>, notifier_buf),
            core::mem::size_of::<fw::microseq::StartFragment::ver<'_>>(),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off FinalizeFragment: fw_stamp {:#x} stamp_value {:#x} stats {:#x} work_item {:#x} unk_758_flag {:#x} pool_ptr_ptr {:#x} restart {:#x} has_attachments {:#x} size {:#x}\n",
            core::mem::offset_of!(fw::microseq::FinalizeFragment::ver, fw_stamp),
            core::mem::offset_of!(fw::microseq::FinalizeFragment::ver, stamp_value),
            core::mem::offset_of!(fw::microseq::FinalizeFragment::ver, stats),
            core::mem::offset_of!(fw::microseq::FinalizeFragment::ver, work_item),
            core::mem::offset_of!(fw::microseq::FinalizeFragment::ver, unk_758_flag),
            core::mem::offset_of!(fw::microseq::FinalizeFragment::ver, pool_ptr_ptr),
            core::mem::offset_of!(fw::microseq::FinalizeFragment::ver, restart_branch_offset),
            core::mem::offset_of!(fw::microseq::FinalizeFragment::ver, has_attachments),
            core::mem::size_of::<fw::microseq::FinalizeFragment::ver>(),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off StartCompute: registers {:#x} stats {:#x} work_queue {:#x} event_seq {:#x} job_params2 {:#x} attachments {:#x} pool_ptr {:#x} notifier_buf {:#x} size {:#x}\n",
            core::mem::offset_of!(fw::microseq::StartCompute::ver<'_>, registers),
            core::mem::offset_of!(fw::microseq::StartCompute::ver<'_>, stats),
            core::mem::offset_of!(fw::microseq::StartCompute::ver<'_>, work_queue),
            core::mem::offset_of!(fw::microseq::StartCompute::ver<'_>, event_seq),
            core::mem::offset_of!(fw::microseq::StartCompute::ver<'_>, job_params2),
            core::mem::offset_of!(fw::microseq::StartCompute::ver<'_>, attachments),
            core::mem::offset_of!(fw::microseq::StartCompute::ver<'_>, pool_ptr),
            core::mem::offset_of!(fw::microseq::StartCompute::ver<'_>, notifier_buf),
            core::mem::size_of::<fw::microseq::StartCompute::ver<'_>>(),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off FinalizeCompute: stats {:#x} work_queue {:#x} job_params2 {:#x} fw_stamp {:#x} stamp_value {:#x} pool_ptr_ptr {:#x} restart {:#x} has_attachments {:#x} size {:#x}\n",
            core::mem::offset_of!(fw::microseq::FinalizeCompute::ver<'_>, stats),
            core::mem::offset_of!(fw::microseq::FinalizeCompute::ver<'_>, work_queue),
            core::mem::offset_of!(fw::microseq::FinalizeCompute::ver<'_>, job_params2),
            core::mem::offset_of!(fw::microseq::FinalizeCompute::ver<'_>, fw_stamp),
            core::mem::offset_of!(fw::microseq::FinalizeCompute::ver<'_>, stamp_value),
            core::mem::offset_of!(fw::microseq::FinalizeCompute::ver<'_>, pool_ptr_ptr),
            core::mem::offset_of!(fw::microseq::FinalizeCompute::ver<'_>, restart_branch_offset),
            core::mem::offset_of!(fw::microseq::FinalizeCompute::ver<'_>, has_attachments),
            core::mem::size_of::<fw::microseq::FinalizeCompute::ver<'_>>(),
        );
        // 🔬 The firmware tests `Scene + 0x38` with `cbz` in both
        // `StartFragment` (fw:0x15524) and `FinalizeVertex` (fw:0x158ac), and
        // something writes `Scene + 0x48`.  Under this driver's packing the
        // struct is only 0x48 bytes, so that write would land past the end.
        // The field names (`unk_20` u32 then `unk_28` U64) imply a 4-byte hole
        // that `#[repr(C)]` with `packed(1)` members does not create.  Print the
        // real offsets rather than reasoning about them.  LOG.md 32.41.231.
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off Scene: user_buffer {:#x} unk_20 {:#x} unk_28 {:#x} stats {:#x} total_page_count {:#x} size {:#x}\n",
            core::mem::offset_of!(fw::buffer::raw::Scene::ver<'_>, user_buffer),
            core::mem::offset_of!(fw::buffer::raw::Scene::ver<'_>, unk_20),
            core::mem::offset_of!(fw::buffer::raw::Scene::ver<'_>, unk_28),
            core::mem::offset_of!(fw::buffer::raw::Scene::ver<'_>, stats),
            core::mem::offset_of!(fw::buffer::raw::Scene::ver<'_>, total_page_count),
            core::mem::size_of::<fw::buffer::raw::Scene::ver<'_>>(),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off Info: last_id {:#x} page_list {:#x} page_list_size {:#x} max_blocks {:#x} block_count {:#x} block_list {:#x} block_ctl {:#x} gpu_page_ptr1 {:#x} size {:#x}\n",
            core::mem::offset_of!(fw::buffer::raw::Info::ver<'_>, last_id),
            core::mem::offset_of!(fw::buffer::raw::Info::ver<'_>, page_list),
            core::mem::offset_of!(fw::buffer::raw::Info::ver<'_>, page_list_size),
            core::mem::offset_of!(fw::buffer::raw::Info::ver<'_>, max_blocks),
            core::mem::offset_of!(fw::buffer::raw::Info::ver<'_>, block_count),
            core::mem::offset_of!(fw::buffer::raw::Info::ver<'_>, block_list),
            core::mem::offset_of!(fw::buffer::raw::Info::ver<'_>, block_ctl),
            core::mem::offset_of!(fw::buffer::raw::Info::ver<'_>, gpu_page_ptr1),
            core::mem::size_of::<fw::buffer::raw::Info::ver<'_>>(),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunVertex: unk_1c {:#x} buffer {:#x} scene {:#x} unk_buffer_buf {:#x} unk_34 {:#x} registers {:#x}\\n",
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_1c),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, buffer),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, scene),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_buffer_buf),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_34),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, registers),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunVertex: tpc {:#x} tpc_size {:#x} microsequence {:#x} microsequence_size {:#x} fragment_stamp_slot {:#x} fragment_stamp_value {:#x}\\n",
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, tpc),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, tpc_size),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, microsequence),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, microsequence_size),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, fragment_stamp_slot),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, fragment_stamp_value),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunVertex: unk_pointee {:#x} unk_pad {:#x} job_params2 {:#x} encoder_params {:#x} unk_55c {:#x} unk_560 {:#x}\\n",
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_pointee),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_pad),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, job_params2),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, encoder_params),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_55c),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_560),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunVertex: sync_grow {:#x} unk_568 {:#x} uses_scratch {:#x} meta {:#x} unk_after_meta {:#x} unk_buf_0 {:#x}\\n",
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, sync_grow),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_568),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, uses_scratch),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, meta),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_after_meta),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_buf_0),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunVertex: unk_buf_8 {:#x} unk_buf_10 {:#x} command_time {:#x} timestamp_pointers {:#x} user_timestamp_pointers {:#x} client_sequence {:#x}\\n",
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_buf_8),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_buf_10),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, command_time),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, timestamp_pointers),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, user_timestamp_pointers),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, client_sequence),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunVertex: pad_5d5 {:#x} unk_5d8 {:#x} unk_5dc {:#x} unk_ts {:#x} unk_5dd_8 {:#x}\\n",
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, pad_5d5),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_5d8),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_5dc),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_ts),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_5dd_8),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunFragment: tag {:#x} counter {:#x} vm_slot {:#x} unk_8 {:#x} microsequence {:#x} microsequence_size {:#x}\\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, tag),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, counter),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, vm_slot),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_8),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, microsequence),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, microsequence_size),
        );
        // fw:0x17830..0x178f4 appends the bg/eot USC registers itself, reading
        // them from cmd+0x928/0x930 (background rsrc_spec/usc) and
        // cmd+0xa14/0xa1c (end-of-tile). The driver only ever puts those in the
        // register array, so print where its own fields actually land to see
        // what the firmware is picking up instead. LOG.md 32.41.50.
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off JP3: base {:#x} unk_partial_bg {:#x} partial_bg {:#x} unk_partial_eot {:#x} partial_eot {:#x} zls_ctrl {:#x} size {:#x}\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, job_params3),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, job_params3)
                + core::mem::offset_of!(fw::fragment::raw::JobParameters3::ver, unk_partial_bg),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, job_params3)
                + core::mem::offset_of!(fw::fragment::raw::JobParameters3::ver, partial_bg),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, job_params3)
                + core::mem::offset_of!(fw::fragment::raw::JobParameters3::ver, unk_partial_eot),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, job_params3)
                + core::mem::offset_of!(fw::fragment::raw::JobParameters3::ver, partial_eot),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, job_params3)
                + core::mem::offset_of!(fw::fragment::raw::JobParameters3::ver, zls_ctrl),
            core::mem::size_of::<fw::fragment::raw::JobParameters3::ver>(),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunFragment: notifier {:#x} buffer {:#x} scene {:#x} unk_buffer_buf {:#x} tvb_tilemap {:#x} ppp_multisamplectl {:#x}\\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, notifier),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, buffer),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, scene),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_buffer_buf),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, tvb_tilemap),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, ppp_multisamplectl),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunFragment: samples {:#x} tiles_per_mtile_y {:#x} tiles_per_mtile_x {:#x} unk_50 {:#x} unk_58 {:#x} isp_merge_upper_x {:#x}\\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, samples),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, tiles_per_mtile_y),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, tiles_per_mtile_x),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_50),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_58),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, isp_merge_upper_x),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunFragment: isp_merge_upper_y {:#x} unk_68 {:#x} tile_count {:#x} registers {:#x} job_params3 {:#x} unk_758_flag {:#x}\\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, isp_merge_upper_y),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_68),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, tile_count),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, registers),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, job_params3),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_758_flag),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunFragment: unk_75c_flag {:#x} unk_buf {:#x} busy_flag {:#x} tvb_overflow_count {:#x} unk_878 {:#x} encoder_params {:#x}\\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_75c_flag),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_buf),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, busy_flag),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, tvb_overflow_count),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_878),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, encoder_params),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunFragment: process_empty_tiles {:#x} no_clear_pipeline_textures {:#x} msaa_zs {:#x} unk_pointee {:#x} unk_v13_3 {:#x} meta {:#x}\\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, process_empty_tiles),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, no_clear_pipeline_textures),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, msaa_zs),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_pointee),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_v13_3),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, meta),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunFragment: unk_after_meta {:#x} unk_buf_0 {:#x} unk_buf_8 {:#x} unk_buf_10 {:#x} command_time {:#x} timestamp_pointers {:#x}\\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_after_meta),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_buf_0),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_buf_8),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_buf_10),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, command_time),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, timestamp_pointers),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunFragment: user_timestamp_pointers {:#x} client_sequence {:#x} pad_925 {:#x} unk_928 {:#x} unk_92c {:#x} unk_ts {:#x}\\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, user_timestamp_pointers),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, client_sequence),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, pad_925),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_928),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_92c),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_ts),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunFragment: unk_92d_8 {:#x}\\n",
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_92d_8),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunCompute: tag {:#x} counter {:#x} unk_4 {:#x} vm_slot {:#x} notifier {:#x} unk_pointee {:#x}\\n",
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, tag),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, counter),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, unk_4),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, vm_slot),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, notifier),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, unk_pointee),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunCompute: registers {:#x} __pad1 {:#x} microsequence {:#x} microsequence_size {:#x} job_params2 {:#x} encoder_params {:#x}\\n",
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, registers),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, __pad1),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, microsequence),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, microsequence_size),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, job_params2),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, encoder_params),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunCompute: meta {:#x} command_time {:#x} timestamp_pointers {:#x} user_timestamp_pointers {:#x} client_sequence {:#x} pad_2d1 {:#x}\\n",
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, meta),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, command_time),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, timestamp_pointers),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, user_timestamp_pointers),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, client_sequence),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, pad_2d1),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunCompute: unk_2d4 {:#x} unk_2d8 {:#x} context_store_req {:#x} uma_pool {:#x} uma_pool_flush {:#x} unk_flag {:#x}\\n",
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, unk_2d4),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, unk_2d8),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, context_store_req),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, uma_pool),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, uma_pool_flush),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, unk_flag),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi/off RunCompute: unk_pad {:#x}\\n",
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, unk_pad),
        );
        #[ver(V >= V15_5)]
        pr_info!(
            "asahi: seq word/flag: vertex {:#x}/{:#x} fragment {:#x}/{:#x} compute {:#x}/{:#x}\n",
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_5d8),
            core::mem::offset_of!(fw::vertex::raw::RunVertex::ver<'_>, unk_5dc),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_928),
            core::mem::offset_of!(fw::fragment::raw::RunFragment::ver<'_>, unk_92c),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, unk_2d4),
            core::mem::offset_of!(fw::compute::raw::RunCompute::ver<'_>, unk_2d8),
        );

        self.tx_channels.lock().device_control.send(
            &fw::channels::DeviceControlMsg::ver::Initialize(Default::default()),
        );

        let dc0a = *crate::module_parameters::dc_0a.value();
        if dc0a != 0 {
            let mut a: crate::fw::types::Array<{ fw::channels::DEVICECONTROL_SZ::ver }, u8> = Default::default();
            for (i, b) in dc0a.to_le_bytes().iter().enumerate() {
                a[i] = *b;
            }
            self.tx_channels
                .lock()
                .device_control
                .send(&fw::channels::DeviceControlMsg::ver::Unk0a(a));
            pr_info!("asahi/dc_0a: sent DeviceControl 0x0a {{{:#x}}}\n", dc0a);
        }

        let initdata = self.initdata.gpu_va().get();
        // The firmware rebuilds this pointer as `msg | 0xfffff00000000000` and
        // dereferences it immediately (LOG.md 27.2), so both the VA we allocated
        // and the truncated form we send are worth having in the log.
        dev_info!(
            self.dev.as_ref(),
            "GPU: InitData at VA {:#x}, sending MSG_INIT {:#x}\n",
            initdata,
            MSG_INIT | (initdata & INIT_DATA_MASK),
        );
        let mut guard = self.rtkit.lock();
        let mut rtk = guard.as_mut().as_pin_mut().unwrap();

        rtk.as_mut().boot()?;
        rtk.as_mut().start_endpoint(EP_FIRMWARE)?;
        rtk.as_mut().start_endpoint(EP_DOORBELL)?;
        // Hold an RTKit power reference across InitData processing.
        //
        // Processing InitData drives the firmware's RTKit power refcount at
        // 0x5c348 from 1 to 0, which wakes the `power` thread (entry 0x46a08)
        // and takes the link ON (0x21) -> QUIESCED (0x10). A quiesced RTKit
        // stops receiving, which is exactly what the counters show: ep 0x21
        // takes in 1-3 messages with InitData and 21+ without it. LOG.md 32.32.
        //
        // Management type 0xa is the refcount adjuster -- undocumented upstream,
        // which only defines up to SET_AP_PWR_STATE (0xb). A payload of **1**
        // takes the reference; measured, not read off the disassembly, because
        // the obvious reading of `cmp w1,#0; cset w0,eq` at 0x4461c plus 0x47100
        // says the opposite and a payload of 0 demonstrably does nothing.
        //
        // ⚠️ This works -- refcount stays 1, the link stays ON, and the `power`
        // thread stays parked instead of being dequeued -- but it does **not**
        // restore receive on its own. ep 0x21 still takes in only the two
        // doorbells sent before InitData. The quiesce is a symptom that travels
        // with the real fault, not the fault itself. Kept because a coprocessor
        // we intend to use should not be quiescing underneath us.
        // Management type 0xa is a refcount adjuster: `cmp w1,#0; cset w0,eq`
        // then 0x47100, which *increments* when its argument is zero -- i.e. a
        // non-zero payload takes a reference. When the count reaches zero the
        // firmware notifies the `power` thread, which quiesces the link, and a
        // quiesced coprocessor ignores everything the AP sends: the RTKit trace
        // ring stops growing, doorbells are never counted, and the pipe rings
        // are never drained.
        //
        // One reference is not enough -- processing InitData drops more than it
        // takes, and the count still lands on zero. Take several, so the link
        // is still ON when the driver starts submitting. LOG.md 32.40.
        #[ver(V >= V15_5)]
        for _ in 0..8 {
            rtk.as_mut().send_message(EP_MGMT, MSG_MGMT_PWR_REF)?;
        }

        rtk.as_mut()
            .send_message(EP_FIRMWARE, MSG_INIT | (initdata & INIT_DATA_MASK))?;
        core::mem::drop(guard);

        {
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
        }

        self.kick_firmware()?;

        // Sample `0xc020` on both units before any job exists. The firmware is
        // booted and the GPU is powered here, and nothing has been submitted,
        // so a payload of 1 at this point means the outstanding request the
        // hung-job dumps report is a boot-time condition rather than something
        // the job creates. LOG.md 32.41.287.
        if *crate::module_parameters::dump_uma_regs.value() == 2 {
            (*self.dev).resources.dump_state_pre("post-init");
        }

        // Hold the link open across the window in which the coprocessor
        // decides to quiesce.
        //
        // The RTKit trace ring shows the quiesce landing a little after init
        // returns -- roughly 0x290 timer ticks past the last message the driver
        // sends -- and a burst of references sent back to back does not survive
        // it (only eight of sixty-four are even traced; whatever queues them
        // coalesces the rest). Spreading them over the window is the thing left
        // to try. LOG.md 32.40.
        #[ver(V >= V15_5)]
        for _ in 0..50 {
            {
                let mut guard = self.rtkit.lock();
                let mut rtk = guard.as_mut().as_pin_mut().unwrap();
                rtk.as_mut().send_message(EP_MGMT, MSG_MGMT_PWR_REF)?;
            }
            kernel::time::delay::fsleep(kernel::time::Delta::from_millis(2));
        }

        Ok(())
    }

    fn update_globals(&self) {
        let mut timeout: u32 = 2;
        if debug_enabled(DebugFlags::WaitForPowerOff) {
            timeout = 0;
        } else if debug_enabled(DebugFlags::KeepGpuPowered) {
            timeout = 5000;
        }

        self.initdata.globals.with(|raw, _inner| {
            raw.idle_off_delay_ms.store(timeout, Ordering::Relaxed);
        });
    }

    fn alloc(&self) -> Guard<'_, KernelAllocators, MutexBackend> {
        /* Clean up idle contexts */
        let mut garbage_ctx = KVec::new();
        core::mem::swap(&mut *self.garbage_contexts.lock(), &mut garbage_ctx);

        for ctx in garbage_ctx {
            // .667zzz41: a context whose hardware compute id is still assigned (+0x33 != 0xff)
            // has a compute job the firmware never retired (abandoned by host recovery). The
            // firmware's per-id statistics walk (fw:0x34b10) keeps reading the context through
            // its slot-table entry, which DestroyContext clears -> NULL deref at fw:0x34b24 some
            // time later. Keep such contexts alive (one leaked id + object per recovery).
            let hwid = ctx.with(|raw, _inner| raw.unk_33);
            if hwid != 0xff && *crate::module_parameters::ctx_leak_held.value() != 0 {
                dev_warn!(
                    self.dev.as_ref(),
                    "GpuContext: hw compute id {} still held at free; keeping the context\n",
                    hwid
                );
                core::mem::forget(ctx);
                continue;
            }
            if self.invalidate_context(&ctx).is_err() {
                dev_err!(
                    self.dev.as_ref(),
                    "GpuContext: Failed to invalidate GPU context!\n"
                );
                if debug_enabled(DebugFlags::OopsOnGpuCrash) {
                    panic!("GPU firmware timed out");
                }
            }
        }

        let mut guard = self.alloc.lock();
        let (garbage_count, garbage_bytes) = guard.private.garbage();
        let (ro_garbage_count, ro_garbage_bytes) = guard.gpu_ro.garbage();
        let (sh_garbage_count, sh_garbage_bytes) = guard.shared.garbage();

        if garbage_bytes > MAX_FW_ALLOC_GARBAGE_BYTES
            || ro_garbage_bytes > MAX_FW_ALLOC_GARBAGE_BYTES
            || sh_garbage_bytes > MAX_FW_ALLOC_GARBAGE_BYTES
            || garbage_count > MAX_FW_ALLOC_GARBAGE_OBJECTS
            || ro_garbage_count > MAX_FW_ALLOC_GARBAGE_OBJECTS
            || sh_garbage_count > MAX_FW_ALLOC_GARBAGE_OBJECTS
        {
            mod_dev_dbg!(
                self.dev,
                "Collecting kalloc garbage (private: {} objects, {} bytes, gpuro: {} objects, {} bytes)\n",
                garbage_count,
                garbage_bytes,
                ro_garbage_count,
                ro_garbage_bytes
            );
            if self.flush_fw_cache().is_err() {
                dev_err!(self.dev.as_ref(), "Failed to flush FW cache\n");
            } else {
                guard.private.collect_garbage(garbage_count);
                guard.gpu_ro.collect_garbage(ro_garbage_count);
                guard.shared.collect_garbage(sh_garbage_count);
            }
        }

        guard
    }

    fn new_vm(&self, kernel_range: Range<u64>) -> Result<mmu::Vm> {
        self.uat.new_vm(self.ids.vm.next(), kernel_range)
    }

    fn bind_vm(&self, vm: &mmu::Vm) -> Result<mmu::VmBind> {
        self.uat.bind(vm)
    }

    fn new_queue(
        &self,
        vm: mmu::Vm,
        ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
        ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
        priority: u32,
        usc_exec_base: u64,
        uma: crate::buffer::UmaSlot,
    ) -> Result<KBox<dyn queue::Queue>> {
        let mut kalloc = self.alloc();
        let id = self.ids.queue.next();
        Ok(KBox::new(
            queue::Queue::ver::new(
                &self.dev,
                vm,
                &mut kalloc,
                ualloc,
                ualloc_priv,
                self.event_manager.clone(),
                &self.buffer_mgr,
                id,
                priority,
                usc_exec_base,
                uma,
            )?,
            GFP_KERNEL,
        )?)
    }

    fn kick_firmware(&self) -> Result {
        if self.is_crashed() {
            return Err(ENODEV);
        }

        let mut guard = self.rtkit.lock();
        let rtk = guard.as_mut().as_pin_mut().unwrap();
        rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_KICKFW)?;

        Ok(())
    }

    fn flush_fw_cache(&self) -> Result {
        mod_dev_dbg!(self.dev, "Flushing coprocessor data cache\n");

        if self.is_crashed() {
            return Err(ENODEV);
        }

        // ctx_0 == 0xff or ctx_1 == 0xff cause no effect on context,
        // but this command does a full cache flush too, so abuse it
        // for that.

        #[ver(V >= V15_5)]
        let dc = if *crate::module_parameters::dc_destroy_155.value() != 0 {
            Self::destroy_context_dc_155(0, 0xff, 0xff, 0, 0)
        } else {
            Self::destroy_context_dc_old()
        };
        #[ver(V < V15_5)]
        let dc = Self::destroy_context_dc_old();

        let mut txch = self.tx_channels.lock();

        let token = txch.device_control.send(&dc);
        {
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
        }

        #[ver(V >= V15_5)]
        {
            self.poll_rx_nocb();
        }
        let mut res = txch.device_control.wait_for(token);
        #[ver(V >= V15_5)]
        for _ in 0..4 {
            if res.is_ok() {
                break;
            }
            self.poll_rx_nocb();
            res = txch.device_control.wait_for(token);
        }
        res?;
        Ok(())
    }

    fn ids(&self) -> &SequenceIDs {
        &self.ids
    }

    fn handle_timeout(&self, counter: u32, event_slot: i32, unk: u32) {
        // Read the scene *after* the job died. See DBG_SCENE_TILEMAP.
        self.probe_scene("POST");
        dev_err!(self.dev.as_ref(), " (\\________/) \n");
        dev_err!(self.dev.as_ref(), "  |        |  \n");
        dev_err!(self.dev.as_ref(), "'.| \\  , / |.'\n");
        dev_err!(self.dev.as_ref(), "--| / (( \\ |--\n");
        dev_err!(self.dev.as_ref(), ".'|  _-_-  |'.\n");
        dev_err!(self.dev.as_ref(), "  |________|  \n");
        dev_err!(self.dev.as_ref(), "** GPU timeout nya~!!!!! **\n");
        dev_err!(self.dev.as_ref(), "  Event slot: {}\n", event_slot);
        dev_err!(self.dev.as_ref(), "  Timeout count: {}\n", counter);
        dev_err!(self.dev.as_ref(), "  Unk: {}\n", unk);
        if event_slot >= 0 {
            match self.event_manager.get_owner(event_slot as u32) {
                Some(owner) => owner.describe(),
                None => dev_err!(self.dev.as_ref(), "  slot owner: none\n"),
            }
        }

        // The 15.5 firmware sends this Timeout *every* run, roughly 200us after
        // the vertex retires and long before the fragment has been kicked --
        // even when nothing is outstanding (`event_slot: -1`). Taking it at face
        // value fails the whole submission and parks the firmware, so the
        // fragment kick that follows lands on a dead GPU. `ignore_timeout=1`
        // keeps the jobs pending and the link alive so we can see whether the
        // fragment would have run. Same shape as `ignore_fault`. LOG.md 32.41.114.
        if *crate::module_parameters::ignore_timeout.value() != 0 {
            dev_err!(
                self.dev.as_ref(),
                "GPU timeout IGNORED (ignore_timeout=1): jobs left pending, no recover\n"
            );
            return;
        }

        // If we have fault info, consider it a fault.
        let error = match self.get_fault_info() {
            Some(info) => workqueue::WorkError::Fault(info),
            None => workqueue::WorkError::Timeout,
        };
        self.mark_pending_events(event_slot.try_into().ok(), error);
        self.recover();
    }

    fn handle_fault(&self) {
        dev_err!(self.dev.as_ref(), " (\\________/) \n");
        dev_err!(self.dev.as_ref(), "  |        |  \n");
        dev_err!(self.dev.as_ref(), "'.| \\  , / |.'\n");
        dev_err!(self.dev.as_ref(), "--| / (( \\ |--\n");
        dev_err!(self.dev.as_ref(), ".'|  _-_-  |'.\n");
        dev_err!(self.dev.as_ref(), "  |________|  \n");
        dev_err!(self.dev.as_ref(), "GPU fault nya~!!!!!\n");
        let error = match self.get_fault_info() {
            Some(info) => workqueue::WorkError::Fault(info),
            None => workqueue::WorkError::Unknown,
        };
        // The VDM's read past the end of Mesa's control-stream BO looks benign:
        // the buffer is exactly 512 KiB, the fault lands 0x80..0x280 past its
        // end (the distance varies run to run, which is prefetch, not a
        // deterministic parse failure), the firmware does **not** set `halted`
        // -- `recover()` says "Cannot recover" precisely because it was never
        // halted -- and the vertex job retires normally in the same run. What
        // actually stops the render is this `mark_pending_events`, which fails
        // the fragment before it ever gets a turn. LOG.md 32.41.108.
        if *crate::module_parameters::ignore_fault.value() != 0 {
            dev_err!(
                self.dev.as_ref(),
                "GPU fault IGNORED (ignore_fault=1): jobs left pending\n"
            );
        } else {
            self.mark_pending_events(None, error);
        }
        self.recover();
    }

    fn host_recover(&self) -> bool {
        // A job the firmware does not consider hung (e.g. a TA context stored
        // on TVB OOM and never resumed: no slot is busy, so its own watchdog
        // at fw:0x35784 never fires) wedges the GPU for every client. macOS
        // stops the firmware from the host instead
        // (AGXArmFirmware::stopFirmwareForRecoveryInspection): EP_DOORBELL msg
        // 0x85 | recovery count (b0[0x4f90], u64). fw:0x319d8 compares the
        // count, marks the request host-initiated ([regs+0x25c]+4 = 1) and
        // wakes the halt path fw:0x4e44, which increments b0[0x4f90], sets
        // b0[0x4fb0] (isFirmwareStoppedForHostRecovery) and b0[0x4fa0] = 1 --
        // from there it is the same 1 -> 2 -> 3 -> 0 handshake as `recover()`.
        #[ver(V >= V15_5)]
        {
            if *crate::module_parameters::host_recovery.value() == 0 {
                return false;
            }
            // One recovery at a time: the vertex and fragment halves of a hung
            // render time out together.
            if RECOVERING
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                let start = Instant::<Monotonic>::now();
                while RECOVERING.load(Ordering::Acquire) && start.elapsed() < Delta::from_millis(5000) {
                    kernel::time::delay::fsleep(Delta::from_millis(1));
                }
                return true;
            }
            let b0 = self.initdata.unkbuf_b0.as_slice();
            // SAFETY: b0 is a live firmware-shared buffer of at least 0x5000
            // bytes; 0x4f90 is the firmware's u64 recovery count.
            let cnt = unsafe { core::ptr::read_volatile(b0.as_ptr().add(0x4f90) as *const u64) };
            dev_err!(
                self.dev.as_ref(),
                "host recovery: MSG_HALT (recovery count {}, b0[0x4fa0] = {:#x})\n",
                cnt,
                // SAFETY: as above.
                unsafe { core::ptr::read_volatile(b0.as_ptr().add(0x4fa0) as *const u32) }
            );
            {
                let mut guard = self.rtkit.lock();
                let rtk = guard.as_mut().as_pin_mut().unwrap();
                if rtk
                    .send_message(EP_DOORBELL, MSG_HALT | (cnt & 0xfff_ffff_ffff))
                    .is_err()
                {
                    dev_err!(self.dev.as_ref(), "host recovery: send failed\n");
                    RECOVERING.store(false, Ordering::Release);
                    return false;
                }
            }
            // The halt path flushes caches and drains the doorbell EP first;
            // give it longer than recover()'s 200 ms.
            let start = Instant::<Monotonic>::now();
            loop {
                // SAFETY: as above.
                let req = unsafe { core::ptr::read_volatile(b0.as_ptr().add(0x4fa0) as *const u32) };
                if req == 1 {
                    break;
                }
                if start.elapsed() > Delta::from_millis(2000) {
                    dev_err!(self.dev.as_ref(), "host recovery: firmware did not stop (b0[0x4fa0] = {:#x})\n", req);
                    RECOVERING.store(false, Ordering::Release);
                    return false;
                }
                mem::sync();
            }
            // SAFETY: as above.
            let (n, why) = unsafe {
                (
                    core::ptr::read_volatile(b0.as_ptr().add(0x4f90) as *const u64),
                    core::ptr::read_volatile(b0.as_ptr().add(0x4fb0) as *const u32),
                )
            };
            dev_err!(self.dev.as_ref(), "host recovery: firmware stopped (count {}, host flag {})\n", n, why);
            // Resume. The host-flagged halt path keeps the channels' state and
            // the scheduler re-dispatches every in-flight command (fw:0x32160
            // -> 0x18d94): nothing may be completed or freed host-side here.
            // Failing them (fail_all) freed the command objects the firmware
            // then replayed from reused memory -> "DAG: Non Sequential Stamp
            // Updates" assert 2-10 ms later (.667zzz29 tests 1-2).
            let ok = self.recover();
            // macOS re-submits every work queue after a restart
            // (restartWorkQueue). Give the scheduler a reason to rescan
            // every pipe/priority so replayed channels are re-dispatched.
            if ok {
                let mut guard = self.rtkit.lock();
                let mut rtk = guard.as_mut().as_pin_mut().unwrap();
                let _ = rtk.as_mut().send_message(EP_MGMT, MSG_MGMT_PWR_REF);
                let _ = rtk.as_mut().send_message(EP_MGMT, MSG_MGMT_SET_IOP_PWR_ON);
                for index in 0..4u64 {
                    let _ = rtk.as_mut().send_message(EP_DOORBELL, MSG_TX_WAKE | (index << 2));
                    for pt in 0..3u64 {
                        let _ = rtk.as_mut().send_message(EP_DOORBELL, MSG_TX_DOORBELL | pt | (index << 2));
                    }
                }
            }
            RECOVERING.store(false, Ordering::Release);
            return ok;
        }
        #[allow(unreachable_code)]
        false
    }

    fn handle_channel_error(
        &self,
        error_type: ChannelErrorType,
        pipe_type: u32,
        event_slot: u32,
        event_value: u32,
    ) {
        dev_err!(self.dev.as_ref(), " (\\________/) \n");
        dev_err!(self.dev.as_ref(), "  |        |  \n");
        dev_err!(self.dev.as_ref(), "'.| \\  , / |.'\n");
        dev_err!(self.dev.as_ref(), "--| / (( \\ |--\n");
        dev_err!(self.dev.as_ref(), ".'|  _-_-  |'.\n");
        dev_err!(self.dev.as_ref(), "  |________|  \n");
        dev_err!(self.dev.as_ref(), "GPU channel error nya~!!!!!\n");
        dev_err!(self.dev.as_ref(), "  Error type: {:?}\n", error_type);
        dev_err!(self.dev.as_ref(), "  Pipe type: {}\n", pipe_type);
        dev_err!(self.dev.as_ref(), "  Event slot: {}\n", event_slot);
        dev_err!(self.dev.as_ref(), "  Event value: {:#x?}\n", event_value);

        self.event_manager.mark_error(
            event_slot,
            event_value,
            workqueue::WorkError::ChannelError(error_type),
        );

        let wq = match self.event_manager.get_owner(event_slot) {
            Some(wq) => wq,
            None => {
                dev_err!(
                    self.dev.as_ref(),
                    "Workqueue not found for this event slot!\n"
                );
                return;
            }
        };

        let wq = match wq.as_any().downcast_ref::<workqueue::WorkQueue::ver>() {
            Some(wq) => wq,
            None => {
                dev_crit!(self.dev.as_ref(), "GpuManager mismatched with WorkQueue!\n");
                return;
            }
        };

        if debug_enabled(DebugFlags::VerboseFaults) {
            wq.dump_info();
        }

        let dc = fw::channels::DeviceControlMsg::ver::RecoverChannel {
            pipe_type,
            work_queue: wq.info_pointer(),
            event_value,
            __pad: Default::default(),
        };

        mod_dev_dbg!(self.dev, "Recover Channel command: {:?}\n", &dc);
        let mut txch = self.tx_channels.lock();

        let token = txch.device_control.send(&dc);
        {
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            if rtk
                .send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)
                .is_err()
            {
                dev_err!(
                    self.dev.as_ref(),
                    "Failed to send Recover Channel command\n"
                );
            }
        }

        if txch.device_control.wait_for(token).is_err() {
            dev_err!(
                self.dev.as_ref(),
                "Timed out waiting for Recover Channel command\n"
            );
        }

        if debug_enabled(DebugFlags::VerboseFaults) {
            wq.dump_info();
        }
    }

    fn ack_grow(&self, buffer_slot: u32, vm_slot: u32, counter: u32) {
        let halt_count = self
            .initdata
            .fw_status
            .with(|raw, _inner| raw.flags.halt_count.load(Ordering::Relaxed));

        let dc = fw::channels::DeviceControlMsg::ver::GrowTVBAck {
            unk_4: 1,
            buffer_slot,
            vm_slot,
            counter,
            subpipe: 0, // TODO
            halt_count: U64(halt_count),
            __pad: Default::default(),
        };

        mod_dev_dbg!(self.dev, "TVB Grow Ack command: {:?}\n", &dc);

        let mut txch = self.tx_channels.lock();

        txch.device_control.send(&dc);
        {
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            if rtk
                .send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)
                .is_err()
            {
                dev_err!(self.dev.as_ref(), "Failed to send TVB Grow Ack command\n");
            }
        }
    }

    fn uma_add_pages(&self, hw_buffer_id: u32, ctx: u32, pool_record: u64, page_list: u64,
                     pages: u32, pool_id: u32, install: u32) {
        #[ver(V >= V15_5)]
        {
            let dc = fw::channels::DeviceControlMsg::ver::UmaAddPages {
                hw_buffer_id,
                ctx,
                unk_c: U64(0),
                pool_record: U64(pool_record),
                unk_1c: U64(0),
                page_list: U64(page_list),
                pages,
                pool_id,
                install,
                __pad: Default::default(),
            };
            // SAFETY: repr(C, u32) enum; the first word is the discriminant.
            let tag = unsafe { *(&dc as *const _ as *const u32) };
            if tag != 0x1e {
                pr_err!("asahi/uma: UmaAddPages discriminant is {:#x}, not 0x1e; not sent\n", tag);
                return;
            }
            if crate::debug::verbose() {
                pr_info!(
                    "asahi/uma: DC 0x1e hwid {} ctx {} pool {:#x} list {:#x} pages {} id {} install {}\n",
                    hw_buffer_id, ctx, pool_record, page_list, pages, pool_id, install
                );
            }
            if let Some(c) = UMA_CHUNKS.get(pool_id as usize) {
                c.fetch_add(pages.div_ceil(0xff8) as i32, Ordering::AcqRel);
            }
            let mut txch = self.tx_channels.lock();
            txch.device_control.send(&dc);
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            if rtk
                .send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)
                .is_err()
            {
                dev_err!(self.dev.as_ref(), "Failed to send UMA add-pages command\n");
            }
        }
    }

    fn dump_hung_state(&self) {
        let dev = &self.dev;

        self.initdata.fw_status.with(|raw, _inner| {
            dev_err!(
                dev.as_ref(),
                "asahi/hung: fw halt_count {} halted {} resume {}\n",
                raw.flags.halt_count.load(Ordering::Relaxed),
                raw.flags.halted.load(Ordering::Relaxed),
                raw.flags.resume.load(Ordering::Relaxed),
            );
        });

        // Which pipe kicks the firmware actually consumed. `submit_seq` says
        // the driver sent one; only the pipe ring's `read_ptr` says the
        // firmware took it. LOG.md 32.41.101.
        for (name, pipes) in [
            ("vtx", &self.pipes.vtx),
            ("frag", &self.pipes.frag),
            ("comp", &self.pipes.comp),
        ] {
            for (i, pipe) in pipes.iter().enumerate() {
                pipe.lock().dump_state(name, i);
            }
        }

        // What the barrier is actually waiting on. LOG.md 32.41.110.
        self.event_manager.dump_stamps(&self.dev, 8);

        // 32.41.468: the tiler counters on the DRIVER-side hang path too.
        //
        // `probe_scene("POST")` is only reached from `handle_timeout`, i.e. the
        // FIRMWARE's timeout callback. Since .467 removed the last fault the
        // firmware no longer times out at all, so `pass_pages` -- the one number
        // that says whether anything was binned -- stopped being printed exactly
        // when the run started getting far enough for it to matter. code-1/2/3
        // all retired the vertex stage with no fault and no tiler line.
        self.probe_scene("HUNG");

        // Force a channel drain. The stats ring goes quiet the instant the
        // firmware picks the job up; if that is because interrupts stopped
        // rather than because the firmware stopped talking, everything it has
        // said since pickup is still sitting in the rings and comes out here.
        dev_err!(dev.as_ref(), "asahi/hung: forcing RX poll\n");
        self.poll_rx();
        dev_err!(dev.as_ref(), "asahi/hung: RX poll done\n");

        // Sample 0xc020 on the hang path. Since 32.41.293 the good
        // configuration produces *no fault*, so the post-fault sweep never
        // runs and the unit state on a clean hang has never been read. This
        // is the same two-index, selector-restoring sample as `init()` uses,
        // so it is safe here. Behind `dump_uma_regs == 2`, like the pre-job
        // one, so the two are always taken together and are comparable.
        // ⚠️ Gated on its OWN parameter, not on `dump_uma_regs`. That one also
        // turns on the post-kick sweep, which writes the `0x10398` unit selector
        // while the firmware is running and measurably reintroduces the CSD
        // runaway: 3/3 faults with it against 2/3 clean without, same config
        // otherwise (LOG.md 32.41.326). Sampling here is safe -- the job is
        // already over and the selector is saved and restored.
        if *crate::module_parameters::dump_state_hung.value() != 0
            || *crate::module_parameters::dump_uma_regs.value() == 2
        {
            (*self.dev).resources.dump_state_pre("hung");
        }
        // LOG.md 32.41.558: indirect debug window, on the Nth hung dump only
        // (the first one fires before the job; .533's pre-job access hung the kick).
        {
            static DBGWIN_CALLS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
            let want = *crate::module_parameters::dump_dbgwin.value();
            let n = DBGWIN_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed) + 1;
            if want != 0 && n == want {
                (*self.dev).resources.dump_dbgwin("hung");
                // 32.41.564: the firmware state block [0x1a37e8] is IDB0.
                // +0x4048/+0x4050 feed AGFHALGetVDSIDQuotaLinesUsed (fw:0x2d960):
                // slot = [+0x4050] + idx*0x930; quota = slot[+0xf2 + 8k] | slot[+0xf4 + 8k] << 9.
                #[ver(V >= V15_5)]
                {
                    let b0 = self.initdata.unkbuf_b0.as_slice();
                    let b0_va: u64 = self.initdata.unkbuf_b0.gpu_pointer().into();
                    let q = |o: usize| u64::from_le_bytes(b0[o..o + 8].try_into().unwrap());
                    let mut o = 0x4000;
                    while o < 0x4100 {
                        if q(o) != 0 {
                            dev_info!(self.dev.as_ref(), "asahi/idb0 +{:#x} = {:#x}\n", o, q(o));
                        }
                        o += 8;
                    }
                    let slots = q(0x4050);
                    dev_info!(
                        self.dev.as_ref(),
                        "asahi/idb0 base {:#x} slots {:#x} (+0x4048 {:#x})\n",
                        b0_va, slots, q(0x4048)
                    );
                    if slots >= b0_va && slots < b0_va + 0x20000 {
                        let so = (slots - b0_va) as usize;
                        for i in 0..4usize {
                            let base = so + i * 0x930;
                            if base + 0x130 > b0.len() {
                                break;
                            }
                            let mut o = 0;
                            while o < 0x130 {
                                if q(base + o) != 0 {
                                    dev_info!(
                                        self.dev.as_ref(),
                                        "asahi/idb0 slot {} +{:#x} = {:#x}\n",
                                        i, o, q(base + o)
                                    );
                                }
                                o += 8;
                            }
                        }
                    }
                }
            }
        }
        // LOG.md 32.41.533: the firmware's USC service/debug status, never read before.
        if *crate::module_parameters::dump_usc.value() != 0 {
            let n = self.dyncfg.id.total_active_cores.min(32);
            (*self.dev).resources.dump_usc("hung", n);
        }

        // The firmware's own vertex stats. A memory read, so it is safe where
        // every MMIO probe is not (32.41.336). Non-zero content is evidence the
        // firmware processed vertex work at all -- which `0xc020` cannot answer
        // because sampling it mid-job changes the outcome. LOG.md 32.41.337.
        //
        // ⚠️ Slot 0, not the scene's user slot. The stats block is a *firmware*
        // address (`0xfffffe...`), so it lives in the kernel context's page
        // tables; reading it out of the user slot returns "unreadable", which is
        // the same reason the `scene obj` probes have always come back unmapped.
        // Partial render (.667zzw): FinalizeFragment decides restart vs final
        // from the frag stats record (op.stats + ctx*48: +4, +0x14, +0x1c,
        // +0x20, fw:0x15664..0x1672c) and the TA OOM handler marks
        // [RegionB+0x244] + slot*24 + 0xc. Show both as the firmware left them.
        let sf = DBG_STATS_FRAG.load(Ordering::Relaxed);
        if sf != 0 {
            self.uat.dump_kern_nz(&self.dev, sf, 0xd00, "statsfrag");
        }
        let svx = DBG_STATS_VTX.load(Ordering::Relaxed);
        if svx != 0 {
            self.uat.dump_kern_nz(&self.dev, svx, 0x700, "statsvtx-nz");
        }
        let sv = DBG_STATS_VTX.load(Ordering::Relaxed);
        if sv != 0 && *crate::module_parameters::dump_stats_vtx.value() != 0 {
            let n = *crate::module_parameters::dump_stats_vtx.value() as usize;
            self.uat.dump_kern(&self.dev, sv, n, "statsvtx");
        }

        // ⚠️ Reading the SGX window here is NOT free. `Queue::drop` runs at
        // process teardown, by which point the GPU may be powered down, and a
        // read of a register that is not answering hangs the CPU on the fabric
        // -- no fault, no oops, and no response to the RCU stall detector's
        // NMI. The three machine-killing wedges of 2026-08-30 all began within
        // 0.3 s of the test process being killed, not during the render.
        // Behind `dump_uma_regs` from here on. LOG.md 32.41.97.
        // ...and behind >= 3, not merely != 0: a *clean* failure reaches here
        // with the GPU powered down, and the read wedges every CPU. That is
        // what turned the ktrace run of 32.41.121 into a power cycle.
        if *crate::module_parameters::dump_uma_regs.value() >= 3 {
            let res = &(*self.dev).resources;
            res.dump_uma();
        }

        let p = self.initdata.runtime_pointers.buffer_mgr_ctl_cpu;
        if p != 0 {
            // PB descriptor table, 16 B per buffer slot. LOG.md 32.41.600.
            for row in 0..2 {
                // SAFETY: live 0x4000-byte kernel object, VMap owned by its ObjectRef.
                let w =
                    |k: usize| unsafe { core::ptr::read_volatile((p + row * 32 + k * 8) as *const u64) };
                dev_err!(
                    dev.as_ref(),
                    "asahi/hung: pb_desc[{:#04x}]: {:#018x} {:#018x} {:#018x} {:#018x}\n",
                    row * 32,
                    w(0),
                    w(1),
                    w(2),
                    w(3),
                );
            }
        }
        let p = self.initdata.runtime_pointers.uma_pool_ctl_cpu;
        if p != 0 {
            // LOG.md 32.41.541: 8 slots -- uma_apple puts the vertex pool in slot 5.
            for row in 0..8 {
                // SAFETY: `uma_pool_ctl` is a live 0x4000-byte kernel object
                // whose VMap is owned by its ObjectRef; 0x100 bytes is well
                // inside it.
                let w =
                    |k: usize| unsafe { core::ptr::read_volatile((p + row * 32 + k * 8) as *const u64) };
                dev_err!(
                    dev.as_ref(),
                    "asahi/hung: uma_pool_ctl[{:#04x}]: {:#018x} {:#018x} {:#018x} {:#018x}\n",
                    row * 32,
                    w(0),
                    w(1),
                    w(2),
                    w(3),
                );
            }
        }
    }

    fn fwctl(&self, msg: fw::channels::FwCtlMsg) -> Result {
        if self.is_crashed() {
            return Err(ENODEV);
        }

        let mut fwctl = self.fwctl_channel.lock();
        let token = fwctl.send(&msg);
        {
            let mut guard = self.rtkit.lock();
            let rtk = guard.as_mut().as_pin_mut().unwrap();
            rtk.send_message(EP_DOORBELL, MSG_FWCTL)?;
        }
        fwctl.wait_for(token)?;
        Ok(())
    }

    fn get_cfg(&self) -> &'static hw::HwConfig {
        self.cfg
    }

    fn get_dyncfg(&self) -> &hw::DynConfig {
        &self.dyncfg
    }

    fn free_context(&self, ctx: KBox<fw::types::GpuObject<fw::workqueue::GpuContextData>>) {
        let mut garbage = self.garbage_contexts.lock();

        if garbage.push(ctx, GFP_KERNEL).is_err() {
            dev_err!(
                self.dev.as_ref(),
                "Failed to reserve space for freed context, deadlock possible.\n"
            );
        }
    }

    fn is_crashed(&self) -> bool {
        self.crashed.load(Ordering::Relaxed)
    }

    fn map_timestamp_buffer(
        &self,
        mut bo: gem::ObjectRef,
        range: Range<usize>,
    ) -> Result<mmu::KernelMapping> {
        let off = range.start;
        let sgt0 = if *crate::module_parameters::ts_dbg.value() != 0 {
            bo.gem
                .owned_sg_table()
                .ok()
                .and_then(|t| t.iter().next().map(|r| r.dma_address() as u64))
        } else {
            None
        };
        let m = bo.map_range_into_range(
            self.uat.kernel_vm(),
            range,
            IOVA_KERN_TIMESTAMP_RANGE,
            mmu::UAT_PGSZ as u64,
            mmu::PROT_FW_SHARED_RW,
            false,
        )?;
        // ts_dbg (.667zzz9): does the firmware-visible PTE point at the BO's page?
        if *crate::module_parameters::ts_dbg.value() == 0 {
            return Ok(m);
        }
        let p = self.uat.kernel_vm().debug_probe(m.iova());
        pr_info!(
            "asahi/ts: bind iova {:#x} bo off {:#x} sgt0 {:#x?} pte {:#x?}\n",
            m.iova(),
            off,
            sgt0,
            p.map(|(pte, w)| (pte, w[0]))
        );
        Ok(m)
    }
}

#[versions(AGX)]
impl GpuManagerPriv for GpuManager::ver {
    fn end_op(&self) {
        let val = self
            .initdata
            .globals
            .with(|raw, _inner| raw.pending_submissions.fetch_sub(1, Ordering::Release));

        mod_dev_dbg!(self.dev, "OP end (pending: {})\n", val - 1);
    }
}
