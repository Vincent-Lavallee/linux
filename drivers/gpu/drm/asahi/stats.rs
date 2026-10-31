// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU usage statistics for system monitors (.667zzz22).
//!
//! Exported as read-only sysfs files on the GPU platform device
//! (`/sys/class/drm/card*/device/`):
//!
//! - `gpu_busy_ns`: total time any firmware work queue had work pending (the
//!   union over all queues, so at most wall time), monotonic. A monitor
//!   samples it twice: utilisation = delta(busy) / delta(wall).
//! - `gpu_mem_used_bytes`: bytes of live GEM objects (user + kernel).
//! - `gpu_mem_objects`: number of live GEM objects.
//!
//! One GPU per system, so the counters are module globals. The busy union is
//! kept with atomics (no lock: the transitions run from both the submission
//! path and the event interrupt path); a racing 1 -> 0 / 0 -> 1 pair can at
//! worst misattribute a few nanoseconds.

use core::ffi::c_char;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use kernel::{bindings, platform, prelude::*, sync::aref::ARef};

static ACTIVE: AtomicU32 = AtomicU32::new(0);
static BUSY_SINCE: AtomicU64 = AtomicU64::new(0);
static BUSY_NS: AtomicU64 = AtomicU64::new(0);
static MEM_BYTES: AtomicU64 = AtomicU64::new(0);
static MEM_OBJECTS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn now_ns() -> u64 {
    // SAFETY: ktime_get() has no preconditions.
    unsafe { bindings::ktime_get() as u64 }
}

/// A work queue went from idle to having work pending.
pub(crate) fn busy_begin() {
    if ACTIVE.fetch_add(1, Ordering::AcqRel) == 0 {
        BUSY_SINCE.store(now_ns(), Ordering::Release);
    }
}

/// A work queue drained (or was torn down with work pending).
pub(crate) fn busy_end() {
    let since = BUSY_SINCE.load(Ordering::Acquire);
    match ACTIVE.fetch_update(Ordering::AcqRel, Ordering::Acquire, |a| a.checked_sub(1)) {
        Ok(1) => {
            BUSY_NS.fetch_add(now_ns().saturating_sub(since), Ordering::AcqRel);
        }
        Ok(_) => {}
        Err(_) => pr_warn!("asahi stats: busy_end without busy_begin\n"),
    }
}

fn busy_total_ns() -> u64 {
    let total = BUSY_NS.load(Ordering::Acquire);
    if ACTIVE.load(Ordering::Acquire) > 0 {
        total + now_ns().saturating_sub(BUSY_SINCE.load(Ordering::Acquire))
    } else {
        total
    }
}

/// Accounts one GEM object's size for its lifetime (a field of `AsahiObject`).
pub(crate) struct MemAcct(u64);

impl MemAcct {
    pub(crate) fn new(size: usize) -> MemAcct {
        MEM_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        MEM_OBJECTS.fetch_add(1, Ordering::Relaxed);
        MemAcct(size as u64)
    }
}

impl Drop for MemAcct {
    fn drop(&mut self) {
        MEM_BYTES.fetch_sub(self.0, Ordering::Relaxed);
        MEM_OBJECTS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Firmware statistics ring (RegionB+0x1e8 on 15.5), raw view (.667zzz24).
/// Apple's decoder is AGXStatistics::recordStatisticsEntryFromFW (kext
/// fffffe000914ab40): tags 16..28, payload offsets below are from the tag.
const FW_TAGS: usize = 32;
const FW_WORDS: usize = 15;
static FW_COUNT: [AtomicU64; FW_TAGS] = [const { AtomicU64::new(0) }; FW_TAGS];
static FW_LAST: [[AtomicU32; FW_WORDS]; FW_TAGS] =
    [const { [const { AtomicU32::new(0) }; FW_WORDS] }; FW_TAGS];

/// Records one firmware statistics message (tag + payload bytes).
pub(crate) fn fw_stat(tag: u32, payload: &[u8]) {
    let t = (tag as usize).min(FW_TAGS - 1);
    FW_COUNT[t].fetch_add(1, Ordering::Relaxed);
    for (i, w) in payload.chunks_exact(4).take(FW_WORDS).enumerate() {
        let v = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        FW_LAST[t][i].store(v, Ordering::Relaxed);
    }
}

/// CPU addresses of live HwDataA fields (set once at init, .667zzz24).
#[derive(Copy, Clone)]
pub(crate) struct HwaPtrs {
    pub(crate) base: usize,
    pub(crate) size: usize,
    pub(crate) actual_pstate: usize,
    pub(crate) tgt_pstate: usize,
    pub(crate) cur_pstate: usize,
    pub(crate) min_pstate_scaled: usize,
    pub(crate) max_pstate_scaled: usize,
    pub(crate) freq_mhz: usize,
    pub(crate) cur_power_mw_2: usize,
    /// A value, not an address: the top pstate's frequency.
    pub(crate) max_freq_mhz: usize,
}

static HWA: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

pub(crate) fn set_hwdata_a(p: HwaPtrs) {
    let v = [
        p.base, p.size, p.actual_pstate, p.tgt_pstate, p.cur_pstate, p.min_pstate_scaled,
        p.max_pstate_scaled, p.freq_mhz, p.cur_power_mw_2, p.max_freq_mhz,
    ];
    for (i, x) in v.iter().enumerate() {
        HWA[i].store(*x as u64, Ordering::Release);
    }
}

fn hwa(i: usize) -> usize {
    HWA[i].load(Ordering::Acquire) as usize
}

fn hwa_read(i: usize) -> u32 {
    let p = hwa(i);
    if p == 0 {
        return 0;
    }
    // SAFETY: points into HwDataA, alive as long as the GPU manager (the
    // driver never unbinds the GPU while files are readable: SysfsFiles drops
    // first).
    unsafe { core::ptr::read_volatile(p as *const u32) }
}

/// Live GPU clock: 0 while the firmware has the GPU powered off (HwDataA+0x10
/// pwr_status == 4: 40/40 samples at an idle desktop, 1/40 under glheavy),
/// else its DVFS frequency (f32 MHz at +0x64, which keeps the last value while off).
fn cur_freq_mhz() -> u64 {
    let base = hwa(0);
    if base == 0 {
        return 0;
    }
    // SAFETY: HwDataA is at least 0x14 bytes and alive (see hwa_read).
    let status = unsafe { core::ptr::read_volatile((base + 0x10) as *const u32) };
    if status == 4 {
        return 0;
    }
    f32_bits_to_u32(hwa_read(7)) as u64
}

/// GPU die temperature in milli-°C: the 15.5 fast-die thermal controller keeps
/// the live reading in centi-°C at HwDataA+0x2200 (copy at +0x219c; +0x21a8 =
/// 8000 release threshold, +0x2230 its pstate limit). 48.75 °C idle, 75.81 °C
/// after 60 s of glheavy, 55.18 °C 60 s later (.667zzz26).
fn gpu_temp_mc() -> u64 {
    let base = hwa(0);
    if base == 0 || hwa(1) < 0x2204 {
        return 0;
    }
    // SAFETY: bounds checked against HwDataA's size above.
    let cc = unsafe { core::ptr::read_volatile((base + 0x2200) as *const u32) };
    cc as u64 * 10
}

/// Bounded writer into a sysfs show buffer.
struct SysfsBuf {
    buf: *mut c_char,
    len: usize,
}

const SYSFS_CAP: usize = 4096;

impl core::fmt::Write for SysfsBuf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            if self.len >= SYSFS_CAP - 1 {
                return Err(core::fmt::Error);
            }
            // SAFETY: sysfs show buffers are at least PAGE_SIZE (>= 4096) bytes.
            unsafe { *self.buf.add(self.len) = b as c_char };
            self.len += 1;
        }
        Ok(())
    }
}

/// `gpu_fw_stats`: one line per tag seen, `tag count w0 .. w14` (hex words).
fn fw_stats_show(buf: *mut c_char) -> isize {
    use core::fmt::Write;
    let mut w = SysfsBuf { buf, len: 0 };
    for t in 0..FW_TAGS {
        let n = FW_COUNT[t].load(Ordering::Relaxed);
        if n == 0 {
            continue;
        }
        let _ = write!(w, "{} {}", t, n);
        for i in 0..FW_WORDS {
            let _ = write!(w, " {:x}", FW_LAST[t][i].load(Ordering::Relaxed));
        }
        let _ = writeln!(w);
    }
    w.len as isize
}

/// Integer part of a non-negative IEEE-754 single (no FPU in the kernel).
fn f32_bits_to_u32(b: u32) -> u32 {
    let exp = ((b >> 23) & 0xff) as i32 - 127;
    if b >> 31 != 0 || exp < 0 {
        return 0;
    }
    if exp > 31 {
        return u32::MAX;
    }
    let m = (b & 0x7f_ffff) | 0x80_0000;
    if exp >= 23 {
        m << (exp - 23)
    } else {
        m >> (23 - exp)
    }
}

/// `gpu_pstate`: firmware view, `actual tgt cur limit_scaled max_scaled freq_mhz power_mw`
/// (+0x54 is the binding limit min(base, max) the firmware recomputes every tick, not a floor).
fn pstate_show(buf: *mut c_char) -> isize {
    use core::fmt::Write;
    let mut w = SysfsBuf { buf, len: 0 };
    let f = f32_bits_to_u32(hwa_read(7));
    let _ = writeln!(
        w,
        "{} {} {} {} {} {} {}",
        hwa_read(2),
        hwa_read(3),
        hwa_read(4),
        hwa_read(5),
        hwa_read(6),
        f,
        hwa_read(8)
    );
    w.len as isize
}

/// .667zzz42: CPU address of the firmware's UMA pool descriptor table (initdata
/// `uma_pool_ctl`, 32 bytes per hw pool id), for `gpu_uma_ring`.
static UMA_CTL: AtomicU64 = AtomicU64::new(0);

pub(crate) fn set_uma_ctl(cpu: usize) {
    UMA_CTL.store(cpu as u64, Ordering::Release);
}

/// .667zzz43: hw pool `id`'s raw ring (head, tail) indices, None if the id has no descriptor.
pub(crate) fn uma_ring_raw(id: u32) -> Option<(u64, u64)> {
    let base = UMA_CTL.load(Ordering::Acquire) as usize;
    if base == 0 || id >= 8 {
        return None;
    }
    let id = id as usize;
    // SAFETY: the 0x4000-byte table lives as long as the GPU manager.
    let (w0, w1) = unsafe {
        (
            core::ptr::read_volatile((base + id * 32) as *const u64),
            core::ptr::read_volatile((base + id * 32 + 8) as *const u64),
        )
    };
    if w0 == 0 {
        return None;
    }
    Some(((w1 >> 5) & 0xfff_ffff, (w1 >> 33) & 0xfff_ffff))
}

/// Free pages in hw pool `id`'s ring (head/tail are ring indices mod the capacity), None if
/// the id has no descriptor.
pub(crate) fn uma_ring_free(id: u32) -> Option<u64> {
    let base = UMA_CTL.load(Ordering::Acquire) as usize;
    if base == 0 || id >= 8 {
        return None;
    }
    let id = id as usize;
    // SAFETY: the 0x4000-byte table lives as long as the GPU manager.
    let (w0, w1) = unsafe {
        (
            core::ptr::read_volatile((base + id * 32) as *const u64),
            core::ptr::read_volatile((base + id * 32 + 8) as *const u64),
        )
    };
    if w0 == 0 {
        return None;
    }
    let head = (w1 >> 5) & 0xfff_ffff;
    let tail = (w1 >> 33) & 0xfff_ffff;
    let cap = (w0 >> 41).max(1);
    Some((tail + cap - head % cap) % cap)
}

/// .667zzz44 `gpu_pass_prof`: GPU time per kind of render pass (game performance). Keyed by the
/// pass signature (size, samples, layers, flags, ZLS, attachments, VM slot); fragment time =
/// completion minus max(previous fragment completion, submission) -- the GPU runs fragment jobs
/// one after another, so while it is busy that is the job's own time -- plus the firmware's
/// start/end stamps (24 MHz ticks) when it wrote them. Reading prints the table since the previous
/// read and resets it.
const PROF_N: usize = 256;
struct PassProf {
    key: AtomicU64,
    info0: AtomicU64,
    info1: AtomicU64,
    count: AtomicU64,
    frag_ns: AtomicU64,
    frag_max_ns: AtomicU64,
    fw_ticks: AtomicU64,
    vtx_ns: AtomicU64,
    vtx_ticks: AtomicU64,
}
#[allow(clippy::declare_interior_mutable_const)]
const PROF_Z: PassProf = PassProf {
    key: AtomicU64::new(0),
    info0: AtomicU64::new(0),
    info1: AtomicU64::new(0),
    count: AtomicU64::new(0),
    frag_ns: AtomicU64::new(0),
    frag_max_ns: AtomicU64::new(0),
    fw_ticks: AtomicU64::new(0),
    vtx_ns: AtomicU64::new(0),
    vtx_ticks: AtomicU64::new(0),
};
static PROF_COMP_N: AtomicU64 = AtomicU64::new(0);
static PROF_COMP_TICKS: AtomicU64 = AtomicU64::new(0);

/// .667zzz44 `gpu_timeline`: the last TL_N jobs' firmware start/end stamps (kind 0 vertex,
/// 1 fragment, 2 compute), to find the GPU's idle gaps between jobs.
const TL_N: usize = 4096;
static TL: [AtomicU64; TL_N * 2] = [const { AtomicU64::new(0) }; TL_N * 2];
static TL_W: AtomicU64 = AtomicU64::new(0);

fn tl_push(kind: u64, ts: (u64, u64)) {
    if ts.0 == 0 || ts.1 <= ts.0 || ts.1 - ts.0 > (1 << 32) {
        return;
    }
    let i = (TL_W.fetch_add(1, Ordering::Relaxed) as usize) % TL_N;
    TL[i * 2].store(ts.0, Ordering::Relaxed);
    TL[i * 2 + 1].store(((ts.1 - ts.0) << 2) | kind, Ordering::Relaxed);
}

fn timeline_show(buf: *mut c_char) -> isize {
    use core::fmt::Write;
    let mut w = SysfsBuf { buf, len: 0 };
    let n = core::cmp::min(TL_W.load(Ordering::Relaxed) as usize, TL_N);
    let mut v: KVec<(u64, u64, u64)> = KVec::new();
    for i in 0..n {
        let s = TL[i * 2].load(Ordering::Relaxed);
        let d = TL[i * 2 + 1].load(Ordering::Relaxed);
        if s != 0 {
            let _ = v.push((s, s + (d >> 2), d & 3), GFP_KERNEL);
        }
    }
    if v.len() < 2 {
        return 0;
    }
    v.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let span = v[v.len() - 1].1.max(v[0].1) - v[0].0;
    // merge into busy intervals; a gap is attributed to (kind that ended last -> kind that starts)
    let (mut busy, mut cur_end, mut end_kind) = (0u64, v[0].1, v[0].2);
    let mut cur_start = v[0].0;
    let mut gaps = [[(0u64, 0u64); 3]; 3];
    let mut hist = [0u64; 6]; // <10us <50 <100 <500 <2000 >=2000
    let mut sum_kind = [0u64; 3];
    for &(s, e, k) in v.iter() {
        sum_kind[k as usize] += e - s;
        if s > cur_end {
            busy += cur_end - cur_start;
            let g = s - cur_end;
            let gr = &mut gaps[end_kind as usize][k as usize];
            gr.0 += 1;
            gr.1 += g;
            let us = g / 24;
            let b = if us < 10 { 0 } else if us < 50 { 1 } else if us < 100 { 2 } else if us < 500 { 3 } else if us < 2000 { 4 } else { 5 };
            hist[b] += g;
            cur_start = s;
            cur_end = e;
            end_kind = k;
        } else if e > cur_end {
            cur_end = e;
            end_kind = k;
        }
    }
    busy += cur_end - cur_start;
    let ms = |t: u64| t / 24_000;
    let _ = writeln!(
        w,
        "jobs {}  span {} ms  busy (union) {} ms = {}%  sum vtx {} frag {} comp {} ms (overlap {} ms)",
        v.len(), ms(span), ms(busy), busy * 100 / span.max(1), ms(sum_kind[0]), ms(sum_kind[1]),
        ms(sum_kind[2]), ms((sum_kind[0] + sum_kind[1] + sum_kind[2]).saturating_sub(busy))
    );
    let names = ["vtx", "frag", "comp"];
    for a in 0..3 {
        for b in 0..3 {
            let (c, t) = gaps[a][b];
            if c != 0 {
                let _ = writeln!(w, "idle {:>4} -> {:<4}: {:>6} gaps {:>6} ms  avg {} us", names[a], names[b], c, ms(t), t / c / 24);
            }
        }
    }
    let _ = writeln!(
        w,
        "idle by gap length (ms): <10us {}  <50us {}  <100us {}  <500us {}  <2ms {}  >=2ms {}",
        ms(hist[0]), ms(hist[1]), ms(hist[2]), ms(hist[3]), ms(hist[4]), ms(hist[5])
    );
    w.len as isize
}

unsafe extern "C" fn show_timeline(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *mut c_char,
) -> isize {
    timeline_show(buf)
}

/// A compute job completed (firmware start/end stamps).
pub(crate) fn pass_prof_comp(ts: (u64, u64)) {
    tl_push(2, ts);
    PROF_COMP_N.fetch_add(1, Ordering::Relaxed);
    if ts.1 > ts.0 && ts.0 != 0 {
        PROF_COMP_TICKS.fetch_add(ts.1 - ts.0, Ordering::Relaxed);
    }
}
static PROF: [PassProf; PROF_N] = [PROF_Z; PROF_N];
static PROF_LAST_FRAG: AtomicU64 = AtomicU64::new(0);
static PROF_LAST_VTX: AtomicU64 = AtomicU64::new(0);
static PROF_SINCE: AtomicU64 = AtomicU64::new(0);
static PROF_DROPPED: AtomicU64 = AtomicU64::new(0);

fn prof_slot(key: u64, info0: u64, info1: u64) -> Option<&'static PassProf> {
    let h = (key ^ (key >> 29)).wrapping_mul(0x9e37_79b9_7f4a_7c15) as usize;
    for i in 0..PROF_N {
        let p = &PROF[(h + i) % PROF_N];
        let k = p.key.load(Ordering::Acquire);
        if k == key {
            return Some(p);
        }
        if k == 0 && p.key.compare_exchange(0, key, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            p.info0.store(info0, Ordering::Release);
            p.info1.store(info1, Ordering::Release);
            return Some(p);
        }
    }
    PROF_DROPPED.fetch_add(1, Ordering::Relaxed);
    None
}

/// A render pass's fragment job completed (`submit_ns` = its submission, `ts` = firmware stamps).
pub(crate) fn pass_prof_frag(key: u64, info0: u64, info1: u64, submit_ns: u64, ts: (u64, u64)) {
    let now = now_ns();
    let prev = PROF_LAST_FRAG.swap(now, Ordering::AcqRel);
    let dur = now.saturating_sub(core::cmp::max(prev, submit_ns));
    tl_push(1, ts);
    if let Some(p) = prof_slot(key, info0, info1) {
        p.count.fetch_add(1, Ordering::Relaxed);
        p.frag_ns.fetch_add(dur, Ordering::Relaxed);
        p.frag_max_ns.fetch_max(dur, Ordering::Relaxed);
        if ts.1 > ts.0 && ts.0 != 0 {
            p.fw_ticks.fetch_add(ts.1 - ts.0, Ordering::Relaxed);
        }
    }
}

/// Its vertex (tiling) job completed.
pub(crate) fn pass_prof_vtx(key: u64, info0: u64, info1: u64, submit_ns: u64, ts: (u64, u64)) {
    let now = now_ns();
    let prev = PROF_LAST_VTX.swap(now, Ordering::AcqRel);
    let dur = now.saturating_sub(core::cmp::max(prev, submit_ns));
    tl_push(0, ts);
    if let Some(p) = prof_slot(key, info0, info1) {
        p.vtx_ns.fetch_add(dur, Ordering::Relaxed);
        if ts.1 > ts.0 && ts.0 != 0 {
            p.vtx_ticks.fetch_add(ts.1 - ts.0, Ordering::Relaxed);
        }
    }
}

fn pass_prof_show(buf: *mut c_char) -> isize {
    use core::fmt::Write;
    let mut w = SysfsBuf { buf, len: 0 };
    let now = now_ns();
    let span = now.saturating_sub(PROF_SINCE.swap(now, Ordering::AcqRel)).max(1);
    // snapshot + reset
    let mut rows: KVec<(u64, u64, u64, u64, u64, u64, u64)> = KVec::new();
    let mut total = 0u64;
    let mut vtotal = 0u64;
    let (mut fwf, mut fwv) = (0u64, 0u64);
    for p in PROF.iter() {
        if p.key.load(Ordering::Acquire) == 0 {
            continue;
        }
        let c = p.count.swap(0, Ordering::AcqRel);
        let f = p.frag_ns.swap(0, Ordering::AcqRel);
        let m = p.frag_max_ns.swap(0, Ordering::AcqRel);
        let t = p.fw_ticks.swap(0, Ordering::AcqRel);
        let v = p.vtx_ns.swap(0, Ordering::AcqRel);
        let vt = p.vtx_ticks.swap(0, Ordering::AcqRel);
        fwf += t;
        fwv += vt;
        let v = vt / 24;
        total += f;
        vtotal += v;
        // forget the kind (a new client re-keys; the open-addressing table otherwise fills with
        // stale kinds of earlier processes)
        p.key.store(0, Ordering::Release);
        if c != 0 {
            let _ = rows.push(
                (f, c, m, t, v, p.info0.load(Ordering::Relaxed), p.info1.load(Ordering::Relaxed)),
                GFP_KERNEL,
            );
        }
    }
    rows.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let _ = writeln!(
        w,
        "span {} ms  frag {} ms ({}%)  vtx {} ms  kinds {}  dropped {}",
        span / 1_000_000, total / 1_000_000, total * 100 / span, vtotal / 1_000_000, rows.len(),
        PROF_DROPPED.swap(0, Ordering::Relaxed)
    );
    let cn = PROF_COMP_N.swap(0, Ordering::AcqRel);
    let ct = PROF_COMP_TICKS.swap(0, Ordering::AcqRel);
    let _ = writeln!(
        w,
        "fw stamps (24 MHz): frag {} ms  vtx {} ms  compute {} ms in {} jobs",
        fwf / 24_000, fwv / 24_000, ct / 24_000, cn
    );
    let _ = writeln!(w, "frag_ms  %  count  avg_us  max_us  fw_us  fwvtx_us_total  WxH xS L flags zls d s vm");
    for r in rows.iter() {
        if w.len > 3900 {
            break;
        }
        let (f, c, m, t, v, i0, i1) = *r;
        let _ = writeln!(
            w,
            "{} {} {} {} {} {} {} {}x{} x{} L{} {:#x} {:#x} {} {} {}",
            f / 1_000_000, f * 100 / total.max(1), c, f / c / 1000, m / 1000, t / c / 24, v / 1_000_000,
            i0 & 0xffff, (i0 >> 16) & 0xffff, (i0 >> 32) & 0xff, (i0 >> 40) & 0xff, i0 >> 48,
            i1 & 0xffff_ffff, (i1 >> 32) & 1, (i1 >> 33) & 1, i1 >> 40
        );
    }
    w.len as isize
}

unsafe extern "C" fn show_pass_prof(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *mut c_char,
) -> isize {
    pass_prof_show(buf)
}

/// `gpu_uma_ring`: per installed hw pool id, the free-page ring the firmware built
/// (fw:0x24d94: w0 = cap<<41 | list>>7, w1 = tail<<33 | head<<5): "id head tail free cap".
fn uma_ring_show(buf: *mut c_char) -> isize {
    use core::fmt::Write;
    let mut w = SysfsBuf { buf, len: 0 };
    let base = UMA_CTL.load(Ordering::Acquire) as usize;
    if base == 0 {
        return 0;
    }
    for id in 0..8usize {
        // SAFETY: the 0x4000-byte table lives as long as the GPU manager.
        let (w0, w1) = unsafe {
            (
                core::ptr::read_volatile((base + id * 32) as *const u64),
                core::ptr::read_volatile((base + id * 32 + 8) as *const u64),
            )
        };
        if w0 == 0 {
            continue;
        }
        let head = (w1 >> 5) & 0xfff_ffff;
        let tail = (w1 >> 33) & 0xfff_ffff;
        let cap = (w0 >> 41).max(1);
        let _ = writeln!(w, "{} {} {} {} {}", id, head, tail, (tail + cap - head % cap) % cap, cap);
    }
    w.len as isize
}

unsafe extern "C" fn show_uma_ring(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *mut c_char,
) -> isize {
    uma_ring_show(buf)
}

/// `gpu_hwdata_a`: raw dump of HwDataA's first 0x800 bytes (hex words, 8 per line).
/// Window start for `gpu_hwdata_a` (root writes a decimal byte offset, .667zzz26).
static HWA_WIN: AtomicU64 = AtomicU64::new(0);

fn hwa_dump(buf: *mut c_char) -> isize {
    use core::fmt::Write;
    let mut w = SysfsBuf { buf, len: 0 };
    let base = hwa(0);
    if base == 0 {
        return 0;
    }
    let off = (HWA_WIN.load(Ordering::Relaxed) as usize).min(hwa(1)) & !3;
    let n = (hwa(1) - off).min(0x400) / 4;
    for i in 0..n {
        // SAFETY: within HwDataA (off + 4 * i < size).
        let v = unsafe { core::ptr::read_volatile((base + off + 4 * i) as *const u32) };
        let _ = write!(w, "{:x}{}", v, if i % 8 == 7 { "\n" } else { " " });
    }
    w.len as isize
}

/// Parses a decimal u32 from a sysfs store buffer.
fn parse_u32(buf: *const c_char, count: usize) -> Option<u32> {
    let mut v: u32 = 0;
    let mut any = false;
    for i in 0..count {
        // SAFETY: sysfs passes `count` valid bytes.
        let c = unsafe { *buf.add(i) } as u8;
        match c {
            b'0'..=b'9' => {
                v = v.checked_mul(10)?.checked_add((c - b'0') as u32)?;
                any = true;
            }
            b'\n' | b' ' => break,
            _ => return None,
        }
    }
    any.then_some(v)
}

/// Writes `v` and a newline into a sysfs buffer (PAGE_SIZE, always enough).
fn emit(buf: *mut c_char, v: u64) -> isize {
    let mut digits = [0u8; 24];
    let mut n = 0;
    let mut x = v;
    loop {
        digits[n] = b'0' + (x % 10) as u8;
        n += 1;
        x /= 10;
        if x == 0 {
            break;
        }
    }
    for i in 0..n {
        // SAFETY: sysfs show buffers are PAGE_SIZE bytes.
        unsafe { *buf.add(i) = digits[n - 1 - i] as c_char };
    }
    // SAFETY: as above.
    unsafe { *buf.add(n) = b'\n' as c_char };
    (n + 1) as isize
}

macro_rules! show_fn {
    ($name:ident, $val:expr) => {
        unsafe extern "C" fn $name(
            _dev: *mut bindings::device,
            _attr: *mut bindings::device_attribute,
            buf: *mut c_char,
        ) -> isize {
            emit(buf, $val)
        }
    };
}

show_fn!(show_busy, busy_total_ns());
show_fn!(show_freq, cur_freq_mhz());
show_fn!(show_max_freq, hwa(9) as u64);
show_fn!(show_power, hwa_read(8) as u64);
show_fn!(show_temp, gpu_temp_mc());
show_fn!(show_mem, MEM_BYTES.load(Ordering::Relaxed));
show_fn!(show_objs, MEM_OBJECTS.load(Ordering::Relaxed));

unsafe extern "C" fn show_fw_stats(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *mut c_char,
) -> isize {
    fw_stats_show(buf)
}

#[repr(transparent)]
struct Attr(bindings::device_attribute);

// SAFETY: the attributes are immutable after construction; sysfs only reads them.
unsafe impl Sync for Attr {}

macro_rules! attr {
    ($name:literal, $show:ident) => {
        Attr(bindings::device_attribute {
            attr: bindings::attribute {
                name: $name.as_ptr(),
                mode: 0o444,
                // SAFETY: the remaining (lockdep) fields are valid zeroed.
                ..unsafe { core::mem::zeroed() }
            },
            show: Some($show),
            store: None,
        })
    };
}

unsafe extern "C" fn show_pstate(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *mut c_char,
) -> isize {
    pstate_show(buf)
}

unsafe extern "C" fn show_hwa(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *mut c_char,
) -> isize {
    hwa_dump(buf)
}

unsafe extern "C" fn show_min_pstate(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *mut c_char,
) -> isize {
    emit(buf, (hwa_read(5) / 100) as u64)
}

/// `gpu_min_pstate` (root): floor for the firmware's DVFS, as a pstate index.
/// Writes HwDataA.min_pstate_scaled (100 * index), clamped to the max.
unsafe extern "C" fn store_min_pstate(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *const c_char,
    count: usize,
) -> isize {
    let Some(v) = parse_u32(buf, count) else {
        return -(bindings::EINVAL as isize);
    };
    let p = hwa(5);
    if p == 0 {
        return -(bindings::ENODEV as isize);
    }
    let max = hwa_read(6).max(100);
    let scaled = (v.max(1) * 100).min(max);
    // SAFETY: points into HwDataA, see hwa_read().
    unsafe { core::ptr::write_volatile(p as *mut u32, scaled) };
    pr_info!("asahi stats: min_pstate_scaled = {}\n", scaled);
    count as isize
}

/// Parses "OFF VAL" (hex or decimal, `0x` prefix optional for hex).
fn parse_two(buf: *const c_char, count: usize) -> Option<(u32, u32)> {
    let mut vals = [0u32; 2];
    let mut n = 0;
    let mut i = 0;
    while i < count && n < 2 {
        // SAFETY: sysfs passes `count` valid bytes.
        let at = |k: usize| unsafe { *buf.add(k) } as u8;
        while i < count && (at(i) == b' ' || at(i) == b'\n') {
            i += 1;
        }
        if i >= count {
            break;
        }
        let mut radix = 10;
        if i + 1 < count && at(i) == b'0' && (at(i + 1) == b'x' || at(i + 1) == b'X') {
            radix = 16;
            i += 2;
        }
        let mut v: u32 = 0;
        let mut any = false;
        while i < count {
            let d = match at(i) {
                c @ b'0'..=b'9' => (c - b'0') as u32,
                c @ b'a'..=b'f' if radix == 16 => (c - b'a' + 10) as u32,
                c @ b'A'..=b'F' if radix == 16 => (c - b'A' + 10) as u32,
                b' ' | b'\n' => break,
                _ => return None,
            };
            v = v.checked_mul(radix)?.checked_add(d)?;
            any = true;
            i += 1;
        }
        if !any {
            return None;
        }
        vals[n] = v;
        n += 1;
    }
    (n == 2).then_some((vals[0], vals[1]))
}

/// `gpu_hwdata_a` store: select the 0x400-byte window (decimal byte offset).
unsafe extern "C" fn store_hwa_win(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *const c_char,
    count: usize,
) -> isize {
    let Some(off) = parse_u32(buf, count) else {
        return -(bindings::EINVAL as isize);
    };
    HWA_WIN.store(off as u64, Ordering::Relaxed);
    count as isize
}

/// `gpu_hwa_poke` (root, diagnostic): "OFF VAL" writes one u32 into HwDataA.
unsafe extern "C" fn store_hwa_poke(
    _dev: *mut bindings::device,
    _attr: *mut bindings::device_attribute,
    buf: *const c_char,
    count: usize,
) -> isize {
    let Some((off, val)) = parse_two(buf, count) else {
        return -(bindings::EINVAL as isize);
    };
    let base = hwa(0);
    if base == 0 || off & 3 != 0 || off as usize + 4 > hwa(1) {
        return -(bindings::EINVAL as isize);
    }
    // SAFETY: bounds and alignment checked against HwDataA's size above.
    unsafe { core::ptr::write_volatile((base + off as usize) as *mut u32, val) };
    pr_info!("asahi stats: HwDataA[{:#x}] = {:#x}\n", off, val);
    count as isize
}

static ATTRS: [Attr; 15] = [
    attr!(c"gpu_pass_prof", show_pass_prof),
    attr!(c"gpu_timeline", show_timeline),
    attr!(c"gpu_busy_ns", show_busy),
    attr!(c"gpu_freq_mhz", show_freq),
    attr!(c"gpu_max_freq_mhz", show_max_freq),
    attr!(c"gpu_power_mw", show_power),
    attr!(c"gpu_temp_mc", show_temp),
    attr!(c"gpu_mem_used_bytes", show_mem),
    attr!(c"gpu_mem_objects", show_objs),
    attr!(c"gpu_fw_stats", show_fw_stats),
    attr!(c"gpu_pstate", show_pstate),
    attr!(c"gpu_uma_ring", show_uma_ring),
    Attr(bindings::device_attribute {
        attr: bindings::attribute {
            name: c"gpu_hwdata_a".as_ptr(),
            mode: 0o644,
            // SAFETY: the remaining (lockdep) fields are valid zeroed.
            ..unsafe { core::mem::zeroed() }
        },
        show: Some(show_hwa),
        store: Some(store_hwa_win),
    }),
    Attr(bindings::device_attribute {
        attr: bindings::attribute {
            name: c"gpu_min_pstate".as_ptr(),
            mode: 0o644,
            // SAFETY: the remaining (lockdep) fields are valid zeroed.
            ..unsafe { core::mem::zeroed() }
        },
        show: Some(show_min_pstate),
        store: Some(store_min_pstate),
    }),
    Attr(bindings::device_attribute {
        attr: bindings::attribute {
            name: c"gpu_hwa_poke".as_ptr(),
            mode: 0o200,
            // SAFETY: the remaining (lockdep) fields are valid zeroed.
            ..unsafe { core::mem::zeroed() }
        },
        show: None,
        store: Some(store_hwa_poke),
    }),
];

/// The sysfs files, removed again when the driver is unbound.
pub(crate) struct SysfsFiles {
    pdev: ARef<platform::Device>,
}

impl SysfsFiles {
    pub(crate) fn new(pdev: ARef<platform::Device>) -> SysfsFiles {
        for a in ATTRS.iter() {
            // SAFETY: the device is bound and `a` is 'static.
            let ret = unsafe { bindings::device_create_file(pdev.as_ref().as_raw(), &a.0) };
            if ret != 0 {
                dev_warn!(pdev.as_ref(), "stats: sysfs file failed ({})\n", ret);
            }
        }
        SysfsFiles { pdev }
    }
}

impl Drop for SysfsFiles {
    fn drop(&mut self) {
        for a in ATTRS.iter() {
            // SAFETY: removing a file that was never created is a no-op.
            unsafe { bindings::device_remove_file(self.pdev.as_ref().as_raw(), &a.0) };
        }
    }
}
