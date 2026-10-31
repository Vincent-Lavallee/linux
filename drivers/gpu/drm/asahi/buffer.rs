// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Tiled Vertex Buffer management
//!
//! This module manages the Tiled Vertex Buffer, also known as the Parameter Buffer (in imgtec
//! parlance) or the tiler heap (on other architectures). This buffer holds transformed primitive
//! data between the vertex/tiling stage and the fragment stage.
//!
//! On AGX, the buffer is a heap of 128K blocks split into 32K pages (which must be aligned to a
//! multiple of 32K in VA space). The buffer can be shared between multiple render jobs, and each
//! will allocate pages from it during vertex processing and return them during fragment processing.
//!
//! If the buffer runs out of free pages, the vertex pass stops and a partial fragment pass occurs,
//! spilling the intermediate render target state to RAM (a partial render). This is all managed
//! transparently by the firmware. Since partial renders are less efficient, the kernel must grow
//! the heap in response to feedback from the firmware to avoid partial renders in the future.
//! Currently, we only ever grow the heap, and never shrink it.
//!
//! AGX also supports memoryless render targets, which can be used for intermediate results within
//! a render pass. To support partial renders, it seems the GPU/firmware has the ability to borrow
//! pages from the TVB buffer as a temporary render target buffer. Since this happens during a
//! partial render itself, if the buffer runs out of space, it requires synchronous growth in
//! response to a firmware interrupt. This is not currently supported, but may be in the future,
//! though it is unclear whether it is worth the effort.
//!
//! This module is also in charge of managing the temporary objects associated with a single render
//! pass, which includes the top-level tile array, the tail pointer cache, preemption buffers, and
//! other miscellaneous structures collectively managed as a "scene".
//!
//! To avoid runaway memory usage, there is a maximum size for buffers (at that point it's unlikely
//! that partial renders will incur much overhead over the buffer data access itself). This is
//! different depending on whether memoryless render targets are in use, and is currently hardcoded.
//! to the most common value used by macOS.

use crate::debug::*;
use crate::module_parameters;
use crate::fw::buffer;
use crate::fw::types::*;
use crate::util::*;
use crate::{
    alloc,
    fw,
    gpu,
    hw,
    mmu,
    slotalloc, //
};
use core::mem::ManuallyDrop;
use core::sync::atomic::Ordering;
use kernel::new_mutex;
use kernel::prelude::*;
use kernel::sync::{
    Arc,
    Mutex, //
};
use kernel::{
    c_str,
    static_lock_class, //
};

const DEBUG_CLASS: DebugFlags = DebugFlags::Buffer;

/// There are 127 GPU/firmware-side buffer manager slots (yes, 127, not 128).
const NUM_BUFFERS: u32 = 127;
/// Stride of the per-slot pass-page counter cells in the PMCT page (`pm_dualmap`).
const PMCT_CELL: u64 = 64;
const _: () = assert!(NUM_BUFFERS as u64 * PMCT_CELL <= 0x4000);

/// Page size bits for buffer pages (32K). VAs must be aligned to this size.
pub(crate) const PAGE_SHIFT: usize = 15;
/// Page size for buffer pages.
pub(crate) const PAGE_SIZE: usize = 1 << PAGE_SHIFT;
/// Number of pages in a buffer block, which should be contiguous in VA space.
pub(crate) const PAGES_PER_BLOCK: usize = 4;
/// Size of a buffer block.
pub(crate) const BLOCK_SIZE: usize = PAGE_SIZE * PAGES_PER_BLOCK;

/// First word of every fresh TVB page (667zzw, `Buffer::tvb_high_water`).
const TVB_SENTINEL: u32 = 0x7b5e_a1ed;

/// Metadata about the tiling configuration for a scene. This is computed in the `render` module.
/// based on dimensions, tile size, and other info.
pub(crate) struct TileInfo {
    /// Tile count in the X dimension. Tiles are always 32x32.
    pub(crate) tiles_x: u32,
    /// Tile count in the Y dimension. Tiles are always 32x32.
    pub(crate) tiles_y: u32,
    /// Total tile count.
    pub(crate) tiles: u32,
    /// Micro-tile width (16 or 32).
    pub(crate) utile_width: u32,
    /// Micro-tile height (16 or 32).
    pub(crate) utile_height: u32,
    // Macro-tiles in the X dimension. Always 4.
    //pub(crate) mtiles_x: u32,
    // Macro-tiles in the Y dimension. Always 4.
    //pub(crate) mtiles_y: u32,
    /// Tiles per macro-tile in the X dimension.
    pub(crate) tiles_per_mtile_x: u32,
    /// Tiles per macro-tile in the Y dimension.
    pub(crate) tiles_per_mtile_y: u32,
    // Total tiles per macro-tile.
    //pub(crate) tiles_per_mtile: u32,
    /// Micro-tiles per macro-tile in the X dimension.
    pub(crate) utiles_per_mtile_x: u32,
    /// Micro-tiles per macro-tile in the Y dimension.
    pub(crate) utiles_per_mtile_y: u32,
    // Total micro-tiles per macro-tile.
    //pub(crate) utiles_per_mtile: u32,
    /// Size of the top-level tilemap, in bytes (for all layers, one cluster).
    pub(crate) tilemap_size: usize,
    /// Size of the Tail Pointer Cache, in bytes (for all layers * clusters).
    pub(crate) tpc_size: usize,
    /// Number of blocks in the clustering meta buffer (for clustering) per layer.
    pub(crate) meta1_layer_stride: u32,
    /// Number of blocks in the clustering meta buffer (for clustering).
    pub(crate) meta1_blocks: u32,
    /// Layering metadata size.
    pub(crate) layermeta_size: usize,
    /// Minimum number of TVB blocks for this render.
    pub(crate) min_tvb_blocks: usize,
    /// Tiling parameter structure passed to firmware.
    pub(crate) params: fw::vertex::raw::TilingParameters,
}

/// A VM's render UMA pool, shared by every Buffer (queue) in it -- Apple keeps one
/// AGXUMAFList per process and never re-sends DeviceControl 0x1e for it. LOG.md 32.41.664.
pub(crate) type UmaSlot = Arc<Mutex<Option<Arc<buffer::UmaPool>>>>;

/// Render UMA pool ids (uma_pool_ctl has 8 hardware slots, 4 is compute's).
const UMA_IDS: [u32; 6] = [5, 6, 7, 1, 2, 3];

/// Owner token per hardware UMA pool id (0 = free) and its last submission stamp.
/// The ids used to rotate blindly: a 7th live pool took a live process's id, and
/// once a pool's process died its id still named the dead list -> gUPM write
/// faults in a dead VM (Plasma session, LOG.md 32.41.667zzh).
static UMA_ID_OWNER: [core::sync::atomic::AtomicU64; 8] = [const { core::sync::atomic::AtomicU64::new(0) }; 8];
static UMA_ID_LAST: [core::sync::atomic::AtomicU64; 8] = [const { core::sync::atomic::AtomicU64::new(0) }; 8];
static UMA_TOKEN_SEQ: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

/// Render submissions in flight per hardware pool id (counted at submit, released when the job's
/// Scene drops). Apple never moves a pool to another FList while it has work outstanding
/// (AGXUMAPool::canSubmitLocked / lockUMAPoolAndValidateFListSeed): an id taken over under a
/// running tiler hung the GPU with 8 concurrent deqp-vk (.667zzz40, CTS tessellation -j8).
static UMA_ID_BUSY: [core::sync::atomic::AtomicU32; 8] = [const { core::sync::atomic::AtomicU32::new(0) }; 8];
/// Token of the pool whose list was last installed (DC 0x1e) on each hardware id.
static UMA_ID_INSTALLED: [core::sync::atomic::AtomicU64; 8] = [const { core::sync::atomic::AtomicU64::new(0) }; 8];
/// Per id: serialises "drain the id, take it, count the submission" across processes.
static UMA_INSTALL_LOCK: [core::sync::atomic::AtomicBool; 8] = [const { core::sync::atomic::AtomicBool::new(false) }; 8];
/// Upper bound on waiting for another pool's work on an id (a job blocked on a fence that only a
/// later submission signals must not deadlock us; the old take-over then happens anyway).
const UMA_DRAIN_MS: u32 = 2000;
/// An id changes hands only after its pool has been idle this long: back-to-back re-installs
/// between two busy pools (every few ms, each drained) still hung the GPU at -j8 -- the page
/// manager's write-back of the previous list is asynchronous (FINDINGS 667zzj).
const UMA_SETTLE_NS: u64 = 20_000_000;
/// ktime when each id's in-flight count last dropped to 0.
static UMA_ID_IDLE_NS: [core::sync::atomic::AtomicU64; 8] = [const { core::sync::atomic::AtomicU64::new(0) }; 8];

/// Host recovery leaked the abandoned jobs' Scenes (queue/mod.rs): forget every in-flight
/// count so take-overs do not wait on them. Live Scenes still release (saturating at 0).
pub(crate) fn uma_busy_reset() {
    let now = crate::stats::now_ns();
    for (b, t) in UMA_ID_BUSY.iter().zip(UMA_ID_IDLE_NS.iter()) {
        b.store(0, core::sync::atomic::Ordering::Release);
        t.store(now, core::sync::atomic::Ordering::Release);
    }
}

fn uma_install_lock(id: u32) {
    use core::sync::atomic::Ordering;
    while UMA_INSTALL_LOCK[id as usize]
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        kernel::time::delay::fsleep(kernel::time::Delta::from_micros(100));
    }
}

fn uma_install_unlock(id: u32) {
    UMA_INSTALL_LOCK[id as usize].store(false, core::sync::atomic::Ordering::Release);
}

/// Wait (bounded) until no submission is in flight on hardware pool `id`.
fn uma_drain_id(id: u32, tok: u64) {
    use core::sync::atomic::Ordering;
    let mut ms = 0;
    while UMA_ID_BUSY[id as usize].load(Ordering::Acquire) != 0
        || crate::stats::now_ns().saturating_sub(UMA_ID_IDLE_NS[id as usize].load(Ordering::Acquire))
            < UMA_SETTLE_NS
    {
        if ms >= UMA_DRAIN_MS {
            pr_warn!(
                "asahi/uma: pool {} takes id {} with {} job(s) still in flight\n",
                tok, id, UMA_ID_BUSY[id as usize].load(Ordering::Acquire)
            );
            return;
        }
        kernel::time::delay::fsleep(kernel::time::Delta::from_millis(1));
        ms += 1;
    }
}
static UMA_USE_SEQ: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

/// Pools of a hardware UMA id, kept alive past their owner. On (re-)install of an id
/// the page manager writes the cached pages back to the PREVIOUS list of that id
/// (fw:0x246b8 drain, LOG.md 32.41.667zzj), so a dead pool's list must outlive the
/// next owner's install: `cur` = last installed pool, `prev` = the one before, freed
/// when a third pool installs the id.
pub(crate) struct UmaGrave<T> {
    cur: core::sync::atomic::AtomicPtr<T>,
    prev: core::sync::atomic::AtomicPtr<T>,
}

impl<T> UmaGrave<T> {
    pub(crate) const fn new() -> Self {
        Self {
            cur: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
            prev: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    /// `pool` is being installed for this id.
    pub(crate) fn bury(&self, pool: Arc<T>) {
        use core::sync::atomic::Ordering;
        let new = Arc::into_raw(pool) as *mut T;
        let old = self.cur.swap(new, Ordering::AcqRel);
        if old == new {
            // SAFETY: `old` came from Arc::into_raw; drop the duplicate reference.
            drop(unsafe { Arc::from_raw(old) });
            return;
        }
        let gone = self.prev.swap(old, Ordering::AcqRel);
        if !gone.is_null() {
            // SAFETY: `gone` came from Arc::into_raw and left both slots.
            drop(unsafe { Arc::from_raw(gone) });
        }
    }
}

impl<T> UmaGrave<T> {
    /// A new reference to the pool last installed for this id, if any.
    pub(crate) fn current(&self) -> Option<Arc<T>> {
        use core::sync::atomic::Ordering;
        let p = self.cur.load(Ordering::Acquire);
        if p.is_null() {
            return None;
        }
        // SAFETY: `p` came from Arc::into_raw and the grave holds that reference
        // until two more installs of this id; borrow it to take our own.
        let a = unsafe { Arc::from_raw(p) };
        let c = a.clone();
        core::mem::forget(a);
        Some(c)
    }
}

// SAFETY: only holds Arc<T> references.
unsafe impl<T: Send + Sync> Sync for UmaGrave<T> {}

/// 667zzw: asynchronous UMA pool growth, the AGXUMAFList grow-request scheme.
///
/// The pool started as a fixed 32 MiB, and a job whose tiler needed more drained
/// it: the firmware hands the pages to the hardware page manager in <= 0xff8-page
/// chunks and retires each chunk with event 13. macOS's limit is 4/8/16 GiB
/// (AGXAcceleratorG15X::getUMAPoolMaxSize) and the host adds pages with further
/// DeviceControl 0x1e "grow requests" (AGXUMAFList::updateSubmitInfo), each
/// retired by event 13 (AGXUMAFList::retireAsyncGrowRequest).
///
/// When the retired chunk is the tail of everything this pool has pushed, the
/// page manager is down to its last chunk: allocate more backing, append its
/// pages to the list (the list has room for 0x100000 entries) and push them.
/// `uma_grow_max` MiB caps a pool. Called from the event channel (rtkit rx work,
/// may sleep).
pub(crate) fn uma_grow_on_retire(gpu: &dyn crate::gpu::GpuManager, id: u32, list: u64, pages: u32) {
    if id as usize >= UMA_GRAVE.len() {
        return;
    }
    let Some(pool) = UMA_GRAVE[id as usize].current() else {
        return;
    };
    // .667zzz42: the shared pool grows by usage (uma_shared_grow_check), not on retire.
    if pool.ualloc.is_none() {
        return;
    }
    let base = u64::from(pool.aperture.gpu_pointer());
    let total = pool.total_pages.load(Ordering::Acquire) as u64;
    if list + u64::from(pages) * 8 != base + total * 8 {
        return;
    }
    let max_pages = (*module_parameters::uma_grow_max.value() as u64) << 8;
    if total >= max_pages {
        pr_warn!("asahi/uma: pool id {} at its {} MiB cap; the tiler may run out of memory\n",
                 id, total >> 8);
        return;
    }
    if pool.growing.swap(true, Ordering::AcqRel) {
        return;
    }
    let r = uma_grow_once(gpu, &pool, base, total, max_pages);
    pool.growing.store(false, Ordering::Release);
    if let Err(e) = r {
        pr_err!("asahi/uma: pool id {} grow from {} pages failed: {:?}\n", id, total, e);
    }
}

fn uma_grow_once(
    gpu: &dyn crate::gpu::GpuManager,
    pool: &buffer::UmaPool,
    base: u64,
    total: u64,
    max_pages: u64,
) -> Result {
    let list_cap = (pool.aperture.as_slice().len() / 8) as u64;
    let limit = core::cmp::min(max_pages, list_cap);
    if total >= limit {
        return Ok(());
    }
    // Double, by 4 MiB blocks (halGetUMABlockSize), at least 32 MiB.
    let add = core::cmp::min(core::cmp::max(total, 8192), limit - total) & !1023;
    if add == 0 {
        return Ok(());
    }
    let slot = pool
        .grow
        .iter()
        .find(|s| s.load(Ordering::Acquire).is_null())
        .ok_or(ENOSPC)?;
    // .667zzz42: the shared pool (no VM) grows in klow.
    let chunk = match pool.ualloc.as_ref() {
        Some(ualloc) => ualloc.lock().array_empty_tagged((add as usize) << 12, b"UMAG")?,
        None => gpu.alloc().klow.array_empty_tagged((add as usize) << 12, b"UMAG")?,
    };
    let cva = u64::from(chunk.gpu_pointer());
    let lp = pool.aperture.as_slice().as_ptr() as *mut u64;
    for i in 0..add {
        // SAFETY: total + add <= list_cap entries; the firmware only reads
        // entries it was given, and these are handed over by the DC below.
        unsafe { lp.add((total + i) as usize).write_volatile(cva + (i << 12)) };
    }
    let b = KBox::new(chunk, GFP_KERNEL)?;
    slot.store(KBox::into_raw(b), Ordering::Release);
    pool.total_pages.store((total + add) as u32, Ordering::Release);
    core::sync::atomic::fence(Ordering::SeqCst);
    pr_info!(
        "asahi/uma: pool id {} grows {} -> {} MiB (+{:#x} pages at {:#x})\n",
        pool.id(), total >> 8, (total + add) >> 8, add, cva
    );
    gpu.uma_add_pages(
        pool.hwid.load(Ordering::Acquire),
        pool.ctx.load(Ordering::Acquire),
        u64::from(pool.pool.gpu_pointer()),
        base + total * 8,
        add as u32,
        pool.id(),
        1,
    );
    Ok(())
}

/// Bytes of a UMA extent table: the firmware's re-push walk can cover
/// extents + pages entries (<= 64 + 8192 u64s), see the `UMAE` allocation.
pub(crate) const UMA_EXTENTS_SIZE: usize = 0x20000;

/// DC 0x1e +8 for kernel-lower UMA tables: TTBAT[63] mirrors the kernel context
/// (`kctx63`), as macOS's ctx 63 (DC+4 stays the job's context). LOG.md 32.41.667zzj.
pub(crate) const UMA_KCTX: u32 = 63;

static UMA_GRAVE: [UmaGrave<buffer::UmaPool>; 8] = [const { UmaGrave::new() }; 8];

/// A free render pool id for `token`, else the least recently submitted one
/// (its owner re-installs on its next submission, see `uma_add_pages_args`).
fn uma_alloc_id(token: u64) -> u32 {
    use core::sync::atomic::Ordering;
    for &id in UMA_IDS.iter() {
        if UMA_ID_OWNER[id as usize]
            .compare_exchange(0, token, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return id;
        }
    }
    // Prefer an id with nothing in flight (its owner re-installs later without waiting on us).
    let id = *UMA_IDS
        .iter()
        .min_by_key(|&&id| {
            (
                UMA_ID_BUSY[id as usize].load(Ordering::Relaxed) != 0,
                UMA_ID_LAST[id as usize].load(Ordering::Relaxed),
            )
        })
        .unwrap();
    let old = UMA_ID_OWNER[id as usize].swap(token, Ordering::AcqRel);
    pr_info!("asahi/uma: all pool ids busy, pool {} takes id {} from {}\n", token, id, old);
    id
}

/// Pool dropped: free its id if it still owns it.
pub(crate) fn uma_release_id(id: u32, token: u64) {
    use core::sync::atomic::Ordering;
    if (id as usize) < UMA_ID_OWNER.len() {
        let _ = UMA_ID_OWNER[id as usize].compare_exchange(token, 0, Ordering::AcqRel, Ordering::Acquire);
    }
}

/// .667zzz42: the device-wide render pool (`uma_shared`).
static UMA_SHARED: UmaGrave<buffer::UmaPool> = UmaGrave::new();

/// The shared render UMA pool, built on first use (`alloc` is the device allocator lock,
/// so builds are serialised). Everything the page manager touches is in klow, which
/// every user VM maps at the same address, so any process's job can use it; hw id 5
/// is installed once and never changes hands. None when `uma_shared` = 0.
fn uma_shared_pool(alloc: &mut gpu::KernelAllocators) -> Result<Option<Arc<buffer::UmaPool>>> {
    let mib = *module_parameters::uma_shared.value() as usize;
    if mib == 0 || *module_parameters::uma_real.value() == 0 {
        return Ok(None);
    }
    if let Some(u) = UMA_SHARED.current() {
        return Ok(Some(u));
    }
    const PPB: usize = 1024;
    const CAP: usize = 0x100000;
    // The install pushes a page list through the firmware's DC 0x1e queue: 64 chunks of 0xff8
    // pages, and the enqueue spins when the queue is full (fw:0x247e0) -> a 1 GiB (65-chunk) list
    // deadlocked the firmware. Cap the pool at 60 chunks.
    let mib = core::cmp::min(mib, (60 * 0xff8) >> 8);
    let backing = alloc.klow.array_empty_tagged::<u8>(mib << 20, b"UMAB")?;
    let pages = backing.as_slice().len() >> 12;
    if pages >= CAP {
        return Err(EINVAL);
    }
    let nblk = pages.div_ceil(PPB);
    let mut aperture = alloc.klow.array_empty_tagged::<u8>(
        core::cmp::max(*module_parameters::uma_aperture_size.value(), CAP * 8), b"UMAA")?;
    let scratch = alloc.klow.array_empty_tagged::<u8>(*module_parameters::uma_aperture_size.value(), b"UMAS")?;
    let aperture_b = alloc.klow.array_empty_tagged::<u8>(aperture.as_slice().len(), b"UMAA")?;
    // The firmware's re-push walk (fw:0x25118) covers extents + pages entries, and every DC 0x1e
    // grow adds its pages to that count (r9: gUPM read fault 0x40600 bytes past an extents table
    // sized for the initial pages, right after a 64 -> 128 MiB grow): size it for the growth cap.
    let max_pages = core::cmp::max(mib, *module_parameters::uma_shared_max.value() as usize) << 8;
    let mut extents = alloc.klow.array_empty_tagged::<u8>(
        core::cmp::max(UMA_EXTENTS_SIZE, (max_pages / 1024 * 8 + 2 * max_pages + 4096) * 8), b"UMAE")?;
    let sync = alloc.gpu.array_empty_tagged::<u8>(0x40, b"UMAQ")?;
    let mut pool = alloc.gpu.array_empty_tagged::<u8>(0x100, b"UMAP")?;
    let fwbuf = alloc.gpu.array_empty_tagged::<u8>(0x100000, b"UMAF")?;
    let token = UMA_TOKEN_SEQ.fetch_add(1, Ordering::Relaxed);
    let bva = u64::from(backing.gpu_pointer());
    let lva = u64::from(aperture.gpu_pointer());
    let eva = u64::from(extents.gpu_pointer());
    {
        let l = aperture.as_mut_slice();
        for i in 0..pages {
            l[i * 8..i * 8 + 8].copy_from_slice(&(bva + ((i as u64) << 12)).to_le_bytes());
        }
    }
    {
        let e = extents.as_mut_slice();
        let mut idx = 0usize;
        for b in 0..nblk {
            idx = (idx + 7) & !7;
            let va = bva + ((b * PPB) as u64) * 0x1000;
            let n = core::cmp::min(PPB, pages - b * PPB) as u64;
            e[idx * 8..idx * 8 + 8].copy_from_slice(&(((va >> 12) << 12) | ((n & 0xfff) << 52)).to_le_bytes());
            idx += 1;
        }
    }
    {
        let self_va = u64::from(pool.gpu_pointer());
        let p = pool.as_mut_slice();
        crate::uma_pool_init(p, lva, *module_parameters::uma_aperture_size.value() as u64,
                             u64::from(sync.gpu_pointer()));
        p[0xa0..0xa8].copy_from_slice(&self_va.to_le_bytes());
        p[0x08..0x0c].copy_from_slice(&UMA_IDS[0].to_le_bytes());
        p[0x14..0x1c].copy_from_slice(&lva.to_le_bytes());
        p[0x1c..0x20].copy_from_slice(&(CAP as u32).to_le_bytes());
        p[0x20..0x24].copy_from_slice(&0u32.to_le_bytes());
        // tail = pages: a recovery re-latches head/tail from this record (a 0 here left an empty
        // ring after the first recovery -> every later job hung). The install pushes the page
        // list itself (as per-process pools do); pushing the extents on top of tail = pages
        // counted every page twice.
        p[0x24..0x28].copy_from_slice(&(pages as u32).to_le_bytes());
        p[0x28..0x2c].copy_from_slice(&0u32.to_le_bytes());
        p[0x2c..0x30].copy_from_slice(&(pages as u32).to_le_bytes());
        p[0x34..0x3c].copy_from_slice(&eva.to_le_bytes());
        p[0x44..0x48].copy_from_slice(&((nblk * 8) as u32).to_le_bytes());
    }
    // .667zzz43: record B = record A with its own address and list (ring resets switch records)
    let mut pool_b = alloc.gpu.array_empty_tagged::<u8>(0x100, b"UMAP")?;
    {
        let self_va = u64::from(pool_b.gpu_pointer());
        let lva_b = u64::from(aperture_b.gpu_pointer());
        let b = pool_b.as_mut_slice();
        b.copy_from_slice(pool.as_slice());
        b[0xa0..0xa8].copy_from_slice(&self_va.to_le_bytes());
        b[0x14..0x1c].copy_from_slice(&lva_b.to_le_bytes());
    }
    pr_info!(
        "asahi/uma: shared render pool id {} record {:#x} list {:#x} backing {:#x} ({} MiB) extents {:#x}\n",
        UMA_IDS[0], u64::from(pool.gpu_pointer()), lva, bva, mib, eva
    );
    let u = Arc::new(
        buffer::UmaPool {
            pool,
            aperture,
            scratch,
            sync,
            fwbuf,
            backing,
            extents,
            announced: core::sync::atomic::AtomicBool::new(false),
            token,
            slot: core::sync::atomic::AtomicU32::new(u32::MAX),
            grow: [const { core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()) }; buffer::UMA_GROW_SLOTS],
            total_pages: core::sync::atomic::AtomicU32::new(pages as u32),
            growing: core::sync::atomic::AtomicBool::new(false),
            last_rec: core::sync::atomic::AtomicU64::new(u64::MAX),
            hwid: core::sync::atomic::AtomicU32::new(1),
            ctx: core::sync::atomic::AtomicU32::new(UMA_KCTX),
            ualloc: None,
            pool_b: Some(pool_b),
            aperture_b: Some(aperture_b),
            rec_b: core::sync::atomic::AtomicBool::new(false),
        },
        GFP_KERNEL,
    )?;
    UMA_SHARED.bury(u.clone());
    Ok(Some(u))
}

/// .667zzz43 `uma_reset_pct`: submissions between the gate and their Scene's UMA_ID_BUSY count.
static UMA_RESETTING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static UMA_SUBMITTING: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
static UMA_RESET_LAST_NS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Held by a render submission from `uma_shared_gate` to the end of `submit_render`.
pub(crate) struct UmaSubmitGuard;

impl Drop for UmaSubmitGuard {
    fn drop(&mut self) {
        UMA_SUBMITTING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// .667zzz43: entry of every render submission (before the device allocator is taken). Waits
/// out a reset of the shared pool's ring, and starts one when the ring is below `uma_reset_pct`.
///
/// The page manager loses pages (House of Ashes ~2400/s, a spilling 1080p FS ~13 per pass:
/// taken from the ring, never written back); at 0 free the tiler hangs. A GPU recovery re-latches
/// the ring from the record and everything renders on, so the pages are not in use -- only
/// lost. The reset does the same without a hang: no submission may be between this gate and its
/// UMA_ID_BUSY count, and every in-flight scene on the id must have retired (no page is in use).
/// Then jobs switch to the pool's other record, whose list is rewritten with every page once
/// (head 0, tail = pages): the firmware installs a job's pool record whenever its slot table holds
/// another one for the id (fw:0x24cf0 at job start: optional hardware pool reset 0x14008, then the
/// uma_pool_ctl descriptor built from the record's list/cap/head/tail/pages). What did not work:
/// DC 0x1e install = 1 on the live id (ring unchanged, GPU hung); install = 0 (re-push from the
/// extents, but at a later job start, under other jobs -> faults at 0); a host recovery while idle
/// (left the ring as it was; only fault recoveries re-latch it). `hwid`/`ctx` are unused.
pub(crate) fn uma_shared_gate(gpu: &dyn crate::gpu::GpuManager, _hwid: u32, _ctx: u32) -> UmaSubmitGuard {
    loop {
        while UMA_RESETTING.load(Ordering::SeqCst) {
            kernel::time::delay::fsleep(kernel::time::Delta::from_micros(200));
        }
        UMA_SUBMITTING.fetch_add(1, Ordering::SeqCst);
        if !UMA_RESETTING.load(Ordering::SeqCst) {
            break;
        }
        UMA_SUBMITTING.fetch_sub(1, Ordering::SeqCst);
    }
    let guard = UmaSubmitGuard;
    let pct = *module_parameters::uma_reset_pct.value() as u64;
    if pct == 0 {
        return guard;
    }
    let Some(pool) = UMA_SHARED.current() else {
        return guard;
    };
    if pool.pool_b.is_none() || pool.aperture_b.is_none() || !pool.announced.load(Ordering::Acquire) {
        return guard;
    }
    let id = pool.id();
    let Some(free) = crate::stats::uma_ring_free(id) else {
        return guard;
    };
    let total = pool.total_pages.load(Ordering::Acquire) as u64;
    let now = crate::stats::now_ns();
    if free * 100 >= total * pct || now.saturating_sub(UMA_RESET_LAST_NS.load(Ordering::Acquire)) < 500_000_000 {
        return guard;
    }
    if UMA_RESETTING.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        // another submission resets; it waits for us
        return guard;
    }
    UMA_SUBMITTING.fetch_sub(1, Ordering::SeqCst);
    let t0 = now;
    let mut ms = 0u32;
    let ok = loop {
        let idle = UMA_SUBMITTING.load(Ordering::SeqCst) == 0
            && UMA_ID_BUSY[id as usize].load(Ordering::SeqCst) == 0
            && crate::gpu::UMA_CHUNKS[id as usize].load(Ordering::Acquire) <= 0;
        if idle {
            break true;
        }
        if ms >= UMA_DRAIN_MS {
            break false;
        }
        kernel::time::delay::fsleep(kernel::time::Delta::from_millis(1));
        ms += 1;
    };
    if ok {
        let pages = pool.total_pages.load(Ordering::Acquire) as u64;
        let to_b = !pool.rec_b.load(Ordering::Acquire);
        let (rec, list) = if to_b {
            (pool.pool_b.as_ref().unwrap(), pool.aperture_b.as_ref().unwrap())
        } else {
            (&pool.pool, &pool.aperture)
        };
        let lva = u64::from(list.gpu_pointer());
        let bva = u64::from(pool.backing.gpu_pointer());
        let lp = list.as_slice().as_ptr() as *mut u64;
        let rp = rec.as_slice().as_ptr() as *mut u8;
        // SAFETY: the list holds CAP >= pages entries and is not live (the other record's list is,
        // and the page manager writes cached pages back to that one when this record installs);
        // the record words are the ones the shared pool's setup wrote: +0x14 list, +0x20 head,
        // +0x24 tail, +0x2c pages -- fw:0x24cf0 builds the descriptor from them.
        unsafe {
            for i in 0..pages {
                lp.add(i as usize).write_volatile(bva + (i << 12));
            }
            (rp.add(0x14) as *mut u64).write_unaligned(lva);
            (rp.add(0x20) as *mut u32).write_volatile(0);
            (rp.add(0x24) as *mut u32).write_volatile(pages as u32);
            (rp.add(0x28) as *mut u32).write_volatile(0);
            (rp.add(0x2c) as *mut u32).write_volatile(pages as u32);
        }
        core::sync::atomic::fence(Ordering::SeqCst);
        pool.rec_b.store(to_b, Ordering::Release);
        pr_info!(
            "asahi/uma: shared pool id {} ring reset: {} of {} pages were free; jobs now use record {} (drained in {} ms)\n",
            id, free, pages, if to_b { 'B' } else { 'A' }, (crate::stats::now_ns() - t0) / 1_000_000
        );
    } else {
        pr_warn!(
            "asahi/uma: shared pool id {} ring reset skipped: {} submission(s), {} scene(s) in flight after {} ms ({} of {} pages free)\n",
            id, UMA_SUBMITTING.load(Ordering::SeqCst), UMA_ID_BUSY[id as usize].load(Ordering::SeqCst), ms, free, total
        );
    }
    UMA_RESET_LAST_NS.store(crate::stats::now_ns(), Ordering::Release);
    UMA_SUBMITTING.fetch_add(1, Ordering::SeqCst);
    UMA_RESETTING.store(false, Ordering::SeqCst);
    guard
}

/// .667zzz42: grow the shared render pool before the tiler drains it (`uma_shared_max`).
/// Reads the free-page ring the firmware keeps for the pool (stats::uma_ring_free); below
/// half of the pool's pages, adds min(pool, 256 MiB) of klow pages through a separate page
/// list (DC 0x1e install = 0: the firmware queues the list to gUPM in <= 0xff8-page chunks,
/// fw:0x246b8; install = 1 would re-latch the descriptor and reset the ring). Called with the
/// device allocator held (render submit). `hwid` = the submitting job's VM slot: gUPM reads the
/// page list through that UAT context (DC+4 -> SGX 0x14020 bits 56..); the slot of the pool's
/// first install may be unbound by now (r9-r11: greeter KWin's slot 1, TTBAT[1] = 0 -> gUPM read
/// fault at the new list's first chunk, then every 0xff8 entries).
pub(crate) fn uma_shared_grow_check(gpu: &dyn crate::gpu::GpuManager, kalloc: &mut gpu::KernelAllocators, hwid: u32) {
    let max_mib = *module_parameters::uma_shared_max.value() as u64;
    if max_mib == 0 {
        return;
    }
    let Some(pool) = UMA_SHARED.current() else {
        return;
    };
    let id = pool.id();
    let Some(free) = crate::stats::uma_ring_free(id) else {
        return;
    };
    let total = pool.total_pages.load(Ordering::Acquire) as u64;
    if free * 2 >= total {
        return;
    }
    let max_pages = max_mib << 8;
    if total >= max_pages {
        static WARNED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            pr_warn!("asahi/uma: shared pool at its {} MiB cap ({} pages free)\n", total >> 8, free);
        }
        return;
    }
    if pool.growing.swap(true, Ordering::AcqRel) {
        return;
    }
    let add = core::cmp::min(core::cmp::min(total, max_pages - total), 65536) & !1023;
    let r = (|| -> Result<u64> {
        if add == 0 {
            return Err(EINVAL);
        }
        let mut slots = pool.grow.iter().filter(|s| s.load(Ordering::Acquire).is_null());
        let (s_chunk, s_list) = match (slots.next(), slots.next()) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(ENOSPC),
        };
        let chunk = kalloc.klow.array_empty_tagged::<u8>((add as usize) << 12, b"UMAG")?;
        let mut list = kalloc.klow.array_empty_tagged::<u8>((add as usize) * 8, b"UMGL")?;
        let cva = u64::from(chunk.gpu_pointer());
        {
            let l = list.as_mut_slice();
            for i in 0..add as usize {
                l[i * 8..i * 8 + 8].copy_from_slice(&(cva + ((i as u64) << 12)).to_le_bytes());
            }
        }
        let lva = u64::from(list.gpu_pointer());
        s_chunk.store(KBox::into_raw(KBox::new(chunk, GFP_KERNEL)?), Ordering::Release);
        s_list.store(KBox::into_raw(KBox::new(list, GFP_KERNEL)?), Ordering::Release);
        pool.total_pages.store((total + add) as u32, Ordering::Release);
        core::sync::atomic::fence(Ordering::SeqCst);
        Ok(lva)
    })();
    match r {
        Ok(lva) => {
            pr_info!(
                "asahi/uma: shared pool id {} grows {} -> {} MiB ({} pages were free)\n",
                id, total >> 8, (total + add) >> 8, free
            );
            gpu.uma_add_pages(hwid, UMA_KCTX, u64::from(pool.pool.gpu_pointer()), lva, add as u32, id, 0);
        }
        Err(e) => pr_err!("asahi/uma: shared pool id {} grow from {} pages failed: {:?}\n", id, total, e),
    }
    pool.growing.store(false, Ordering::Release);
}

/// A single scene, representing a render pass and its required buffers.
#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Scene {
    object: GpuObject<buffer::Scene::ver>,
    slot: u32,
    rebind: bool,
    preempt2_off: usize,
    preempt3_off: usize,
    // Note: these are dead code only on some version variants.
    // It's easier to do this than to propagate the version conditionals everywhere.
    #[allow(dead_code)]
    meta1_off: usize,
    #[allow(dead_code)]
    meta2_off: usize,
    #[allow(dead_code)]
    meta3_off: usize,
    #[allow(dead_code)]
    meta4_off: usize,
    /// GPU VA of the G14X+ `control_word`, or 0 elsewhere.  See
    /// `control_word_pointer`.
    control_word_va: u64,
    /// Hardware UMA pool id this submission is counted on (`UMA_ID_BUSY`), u32::MAX = none.
    uma_busy_id: core::sync::atomic::AtomicU32,
}

#[versions(AGX)]
impl Scene::ver {
    /// Returns true if the buffer was bound to a fresh manager slot, and therefore needs an init
    /// command before a render.
    pub(crate) fn rebind(&self) -> bool {
        self.rebind
    }

    /// Returns the buffer manager slot this scene's buffer was bound to.
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }

    /// Returns the GPU pointer to the [`buffer::Scene::ver`].
    pub(crate) fn gpu_pointer(&self) -> GpuPointer<'_, buffer::Scene::ver> {
        self.object.gpu_pointer()
    }

    /// Returns the GPU weak pointer to the [`buffer::Scene::ver`].
    pub(crate) fn weak_pointer(&self) -> GpuWeakPointer<buffer::Scene::ver> {
        self.object.weak_pointer()
    }

    /// Returns the GPU weak pointer to the kernel-side temp buffer.
    /// (purpose unknown...)
    pub(crate) fn kernel_buffer_pointer(&self) -> GpuWeakPointer<[u8]> {
        self.object.buffer.inner.lock().kernel_buffer.weak_pointer()
    }

    /// Returns the GPU pointer to the zeroed UMA pool object.
    ///
    /// Allocated on every version rather than gated: the `#[versions]` macro
    /// does not take `#[ver]` on methods, and this is a Rust-side allocation
    /// that never appears in a firmware struct, so carrying 0x90 unused bytes
    /// on the versions that do not read it costs nothing.
    pub(crate) fn uma_pool_pointer(&self) -> GpuWeakPointer<[u8]> {
        self.object.uma.rec().weak_pointer()
    }

    /// Returns the GPU pointer to the `buffer::Info::ver` object associated with this Scene.
    pub(crate) fn buffer_pointer(&self) -> GpuPointer<'_, buffer::Info::ver> {
        // SAFETY: We can't return the strong pointer directly since its lifetime crosses a lock,
        // but we know its lifetime will be valid as long as &self since we hold a reference to the
        // buffer, so just construct the strong pointer with the right lifetime here.
        unsafe { self.weak_buffer_pointer().upgrade() }
    }

    /// Returns the GPU weak pointer to the `buffer::Info::ver` object associated with this Scene.
    pub(crate) fn weak_buffer_pointer(&self) -> GpuWeakPointer<buffer::Info::ver> {
        self.object.buffer.inner.lock().info.weak_pointer()
    }

    /// Arguments for DeviceControl 0x1e when the pool is backed (`uma_real`):
    /// (pool record, page list, pages, pool id). LOG.md 32.41.545.
    pub(crate) fn uma_add_pages_args(&self, vm_slot: u32) -> Option<(u64, u64, u32, u32)> {
        if *crate::module_parameters::uma_real.value() == 0 {
            return None;
        }
        use core::sync::atomic::Ordering;
        let pool = self.object.uma.pool.as_slice();
        let id = u32::from_le_bytes([pool[8], pool[9], pool[10], pool[11]]);
        let tok = self.object.uma.token;
        let owned = (id as usize) < UMA_ID_OWNER.len();
        if !owned {
            return self.uma_add_pages_args_locked(vm_slot, id, tok, false);
        }
        uma_install_lock(id);
        let r = self.uma_add_pages_args_locked(vm_slot, id, tok, true);
        // count this submission on the id (released in Drop) -- inside the lock, so a
        // concurrent take-over drains it
        UMA_ID_BUSY[id as usize].fetch_add(1, Ordering::AcqRel);
        self.uma_busy_id.store(id, Ordering::Release);
        uma_install_unlock(id);
        r
    }

    fn uma_add_pages_args_locked(
        &self,
        vm_slot: u32,
        id: u32,
        tok: u64,
        owned: bool,
    ) -> Option<(u64, u64, u32, u32)> {
        use core::sync::atomic::Ordering;
        let pool = self.object.uma.pool.as_slice();
        if owned {
            UMA_ID_LAST[id as usize].store(UMA_USE_SEQ.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
        }
        // 32.41.664: announce the pool once, as Apple does for a new pool;
        // established pools never send 0x1e again -- unless another pool took
        // the id meanwhile (667zzh): take it back and re-install.
        let first = !self.object.uma.announced.swap(true, Ordering::AcqRel);
        let stolen = owned && UMA_ID_OWNER[id as usize].load(Ordering::Acquire) != tok;
        // .667zzz40: re-install only when another pool's list is live on the id, after that
        // pool's in-flight work drained. A take-over that never installed (uma_alloc_id swaps the
        // owner at pool creation) is reclaimed silently: re-sending DC 0x1e for our own live list
        // under our own running jobs hung the GPU (CTS tessellation -j8).
        let foreign = owned && UMA_ID_INSTALLED[id as usize].load(Ordering::Acquire) != tok;
        if stolen && !foreign {
            UMA_ID_OWNER[id as usize].store(tok, Ordering::Release);
        }
        // The page manager reads/writes the list through the UAT context given at
        // install (DC+4 = the VM slot then); a VM rebound to another slot must
        // re-install or the UPM keeps using the old slot's tables.
        // 667zzj: a klow list is read through context 0 whatever the job's slot.
        let moved = self.object.uma.slot.swap(vm_slot, Ordering::AcqRel) != vm_slot
            && *crate::module_parameters::uma_klow.value() == 0;
        if !first && !foreign && !moved {
            return None;
        }
        if moved && !first {
            pr_info!("asahi/uma: pool {} id {} re-installs for VM slot {}\n", tok, id, vm_slot);
        }
        if foreign {
            uma_drain_id(id, tok);
            let old = UMA_ID_OWNER[id as usize].swap(tok, Ordering::AcqRel);
            if !first {
                pr_info!("asahi/uma: pool {} re-installs id {} (held by {})\n", tok, id, old);
            }
        }
        if owned {
            UMA_ID_OWNER[id as usize].store(tok, Ordering::Release);
            UMA_ID_INSTALLED[id as usize].store(tok, Ordering::Release);
            UMA_GRAVE[id as usize].bury(self.object.uma.clone());
        }
        // 667zzw: a re-install covers the grow chunks too (contiguous in the list).
        let pages = core::cmp::max(
            u32::from_le_bytes([pool[0x2c], pool[0x2d], pool[0x2e], pool[0x2f]]),
            self.object.uma.total_pages.load(Ordering::Acquire),
        );
        Some((
            u64::from(self.object.uma.rec().gpu_pointer()),
            u64::from(self.object.uma.aperture.gpu_pointer()),
            pages,
            id,
        ))
    }

    /// Returns the GPU pointer to the memory the UMA pool's hardware aperture
    /// descriptor is based at. LOG.md 32.41.51.
    pub(crate) fn uma_aperture_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.uma.aperture.gpu_pointer()
    }

    /// 667zzw: AGXUMAFList::updateSubmitInfo does `[[this+456]] += 1` on every
    /// submission -- the word `pool[0x48]` points at, while `pool[0x50]` keeps
    /// its value from setup. `fw:0x252a0` frees the firmware's pool record when
    /// the two are equal, i.e. only for a list nobody has submitted against.
    /// Ours never moved, so a job that drained the whole pool got its pool freed
    /// under the pending fragment and the PM walk (fw:0x13790) dereferenced the
    /// NULL slot: firmware crash, FAR 0x58.
    pub(crate) fn uma_submit_tick(&self, tick: bool) {
        if *module_parameters::uma_rec_trace.value() != 0 {
            let r = self.object.uma.rec().as_slice();
            let w = |i: usize| u32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]);
            let sw = self.object.uma.sync.as_slice();
            let sync = u64::from_le_bytes([sw[0], sw[1], sw[2], sw[3], sw[4], sw[5], sw[6], sw[7]]);
            let key = (u64::from(w(0x20)) << 40) ^ (u64::from(w(0x24)) << 16) ^ sync;
            if self.object.uma.last_rec.swap(key, Ordering::AcqRel) != key {
                pr_info!(
                    "asahi/umarec: id {} head {:#x} tail {:#x} [0x28] {:#x} pages {:#x} [0x30] {:#x} sync {:#x} total {:#x}\n",
                    self.object.uma.id(), w(0x20), w(0x24), w(0x28), w(0x2c), w(0x30), sync,
                    self.object.uma.total_pages.load(Ordering::Relaxed)
                );
            }
        }
        if !tick {
            return;
        }
        let p = self.object.uma.sync.as_slice().as_ptr() as *const core::sync::atomic::AtomicU64;
        // SAFETY: `sync` is a live, 64-byte-aligned 0x40-byte GPU object.
        unsafe { (*p).fetch_add(1, Ordering::Release) };
    }

    /// Remember DC 0x1e's context words for grow requests (667zzw).
    pub(crate) fn uma_note_install(&self, hwid: u32, ctx: u32) {
        self.object.uma.hwid.store(hwid, Ordering::Release);
        self.object.uma.ctx.store(ctx, Ordering::Release);
    }

    /// Returns the GPU pointer to the word `pool + 0x48` points at.
    /// LOG.md 32.41.90.
    pub(crate) fn uma_sync_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.uma.sync.gpu_pointer()
    }

    pub(crate) fn uma_fwbuf_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.uma.fwbuf.gpu_pointer()
    }

    /// Returns the GPU pointer for `record + 0x18`, the second pool register.
    /// LOG.md 32.41.88.
    pub(crate) fn uma_scratch_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.uma.scratch.gpu_pointer()
    }

    /// GPU address of the first TVB block -- the parameter buffer's actual
    /// backing memory, as opposed to the page/block *metadata* arrays.
    ///
    /// Apple's `AGXParameterBufferBlock` holds an `IOMemoryDescriptor` at +0x18
    /// that `prepareBacking` maps through
    /// `AGXUnifiedAddressTranslator::createMappingInAperture`; the block's +0x00
    /// -- the word register `0x1c910` is built from -- is therefore most likely
    /// the GPU address of that backing, not of any bookkeeping structure. This
    /// driver's equivalent is `blocks[0]`, which is a different kind of object
    /// from everything `vtx_1c910_src` swept before (page_list, block_list,
    /// block_ctl are all metadata).  LOG.md 32.41.239.
    pub(crate) fn tvb_block0_pointer(&self) -> u64 {
        let inner = self.object.buffer.inner.lock();
        match inner.blocks.first() {
            Some(b) => b.gpu_va().get(),
            None => inner.cpu_blocks.first().map_or(0, |b| u64::from(b.gpu_pointer())),
        }
    }

    /// CPU addresses of the tiler's page counters, for the mid-flight probe.
    ///
    /// The zero tilemap cannot distinguish "the tiler ran and emitted nothing"
    /// from "the tiler never ran"; these counters can, because the firmware and
    /// the hardware parameter manager advance them as TVB pages are handed out.
    /// They have to be read *during* the job -- `Scene::drop` never executes
    /// when the render wedges (32.41.116) -- and the one hook that always runs
    /// is `probe_scene`, which has no way to reach this object.  So hand it raw
    /// addresses: the scene and its buffer are both owned by the in-flight job,
    /// so they outlive the probe.  Diagnostic only.  LOG.md 32.41.236.
    /// CPU address of the word gPM actually increments.
    ///
    /// 32.41.508: `pass_page_count` is only a COPY the firmware makes from
    /// behind `control_word2`, so it cannot see gPM's writes while the two
    /// halves are separate allocations. `pm_counter` IS the word register
    /// `0x1c910` points at, and `ualloc_priv` is created with `cpu_maps`, so
    /// reading it directly is a measurement of gPM rather than of the copy.
    pub(crate) fn pm_counter_ptr(&self) -> u64 {
        #[ver(G >= G14X)]
        let p = self
            .object
            .with(|_raw, inner| &inner.pm_counter[0] as *const _ as u64);
        #[ver(G < G14X)]
        let p = 0u64;
        p
    }

    pub(crate) fn counter_ptrs(&self) -> (u64, u64, u64, u64) {
        let (pass, total) = self.object.with(|raw, _| {
            (
                &raw.pass_page_count as *const _ as u64,
                &raw.total_page_count as *const _ as u64,
            )
        });
        let inner = self.object.buffer.inner.lock();
        let (blocks, pages) = inner.info.with(|raw, _| {
            (
                &raw.block_count as *const _ as u64,
                &raw.page_count as *const _ as u64,
            )
        });
        (pass, total, blocks, pages)
    }

    /// Returns the GPU pointer to the G14X+ "control word" -- the first field
    /// of the firmware `Scene` struct.
    ///
    /// macOS's tiler register `0x1c910` takes `*(pbb + 0x00)` and `0x1ca28` /
    /// `0x1ca30` / `0x16c39` take `*(pbb + 0x28)` from the same
    /// parameter-buffer block.  `0x1ca30`/`0x16c39` are `user_buffer`
    /// (32.41.48), and in this driver's `Scene` layout on G14X `user_buffer`
    /// sits at exactly **+0x28** -- behind `control_word`, `control_word2`,
    /// `pass_page_count`+`unk_4`, `unk_8` and `unk_10`.  So Apple's block is
    /// this struct, and `pbb + 0x00` is `control_word`, which upstream
    /// allocates zeroed and does not explain.  LOG.md 32.41.229.
    ///
    /// Captured at scene creation rather than read from the object here: the
    /// field only exists on G14X+ and `#[versions]` does not accept `#[ver]`
    /// on a method.
    pub(crate) fn control_word_pointer(&self) -> u64 {
        self.control_word_va
    }

    /// Returns the GPU pointers to the TVB page pool structures the hardware
    /// parameter manager walks: the page-number array, the block-number array,
    /// and the block control word.  Needed to point register 0x1c910 at one of
    /// them; macOS takes its value out of the parameter-buffer block that
    /// `AGXTACommandDescriptor::loadParameterManagementData` installs.
    /// LOG.md 32.41.222.
    pub(crate) fn tvb_page_list_pointer(&self) -> u64 {
        u64::from(
            self.object
                .buffer
                .inner
                .lock()
                .info
                .page_list
                .gpu_pointer(),
        )
    }

    pub(crate) fn tvb_block_list_pointer(&self) -> u64 {
        u64::from(
            self.object
                .buffer
                .inner
                .lock()
                .info
                .block_list
                .gpu_pointer(),
        )
    }

    pub(crate) fn tvb_block_ctl_pointer(&self) -> u64 {
        u64::from(self.object.buffer.inner.lock().info.block_ctl.gpu_pointer())
    }

    /// Returns the GPU pointer to the "user buffer" -- m1n1's
    /// `BufferThing.unkptr_18`, the 0x80-byte list macOS also programs into
    /// registers 0x1ca30/0x16c39 (vertex) and 0x1ca28 (fragment).
    /// LOG.md 32.41.48.
    pub(crate) fn user_buffer_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.user_buffer.gpu_pointer()
    }

    /// Returns the GPU pointer to the TVB heap metadata buffer.
    pub(crate) fn tvb_heapmeta_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.tvb_heapmeta.gpu_pointer()
    }

    /// Returns the GPU pointer to the layer metadata buffer.
    pub(crate) fn tvb_layermeta_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        // layermeta_at0=1: Apple's G15 single-layer draw programs 0x16060 (layer
        // meta) == 0x16098 (heap meta), not heapmeta+0x200. LOG.md 32.41.615.
        if *module_parameters::layermeta_at0.value() != 0 {
            self.object.tvb_heapmeta.gpu_pointer()
        } else {
            self.object.tvb_heapmeta.gpu_offset_pointer(0x200)
        }
    }

    /// Returns the GPU pointer to the top-level TVB tilemap buffer.
    pub(crate) fn tvb_tilemap_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.tvb_tilemap.gpu_pointer()
    }

    /// Returns the GPU pointer to the Tail Pointer Cache buffer.
    pub(crate) fn tpc_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.tpc.gpu_pointer()
    }

    /// Returns the GPU pointer to the first preemption scratch buffer.
    pub(crate) fn preempt_buf_1_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object.preempt_buf.gpu_pointer()
    }

    /// Returns the GPU pointer to the second preemption scratch buffer.
    pub(crate) fn preempt_buf_2_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object
            .preempt_buf
            .gpu_offset_pointer(self.preempt2_off)
    }

    /// Returns the GPU pointer to the third preemption scratch buffer.
    pub(crate) fn preempt_buf_3_pointer(&self) -> GpuPointer<'_, &'_ [u8]> {
        self.object
            .preempt_buf
            .gpu_offset_pointer(self.preempt3_off)
    }

    /// Returns the GPU pointer to the per-cluster tilemap buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn cluster_tilemaps_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.tilemaps.gpu_pointer())
    }

    /// Returns the GPU pointer to the clustering layer metadata buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn tvb_cluster_layermeta_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_pointer())
    }

    /// Returns the GPU pointer to the clustering metadata 1 buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn meta_1_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_offset_pointer(self.meta1_off))
    }

    /// Returns the GPU pointer to the clustering metadata 2 buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn meta_2_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_offset_pointer(self.meta2_off))
    }

    /// Returns the GPU pointer to the clustering metadata 3 buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn meta_3_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_offset_pointer(self.meta3_off))
    }

    /// Returns the GPU pointer to the clustering metadata 4 buffer, if clustering is enabled.
    #[allow(dead_code)]
    pub(crate) fn meta_4_pointer(&self) -> Option<GpuPointer<'_, &'_ [u8]>> {
        self.object
            .clustering
            .as_ref()
            .map(|c| c.meta.gpu_offset_pointer(self.meta4_off))
    }
}

#[versions(AGX)]
impl Drop for Scene::ver {
    fn drop(&mut self) {
        // .667zzz40: this submission no longer holds its UMA pool id
        let busy = self.uma_busy_id.load(core::sync::atomic::Ordering::Acquire);
        if (busy as usize) < UMA_ID_BUSY.len() {
            if UMA_ID_BUSY[busy as usize].fetch_update(
                core::sync::atomic::Ordering::AcqRel,
                core::sync::atomic::Ordering::Acquire,
                |v| v.checked_sub(1),
            ) == Ok(1)
            {
                UMA_ID_IDLE_NS[busy as usize]
                    .store(crate::stats::now_ns(), core::sync::atomic::Ordering::Release);
            }
        }
        // 🔬 The tiler's own bookkeeping, read after the job is over.
        //
        // A zero tilemap cannot tell "the tiler ran and emitted nothing" from
        // "the tiler never ran": both leave it untouched. The page counters
        // can. `pass_page_count` and `total_page_count` live in the firmware
        // `Scene`, and `block_count` / `page_count` / `block_ctl.{total,wptr}`
        // in `buffer::Info`; the firmware and the hardware parameter manager
        // advance them as TVB pages are handed out. Non-zero movement here is
        // direct evidence the tiler allocated, which is upstream of anything
        // being binned.
        //
        // Drop is the right place: it runs after the job completes or is
        // killed, and it is the one point that still holds both objects.
        // LOG.md 32.41.234.
        //
        // 667zzf: the firmware faulted in FinalizeVertex reading a garbage
        // `control_word2` (+0x08, 0x6c0000) of a freshly built Scene. Check
        // whether the kernel copy was overwritten (a stale writer into reused
        // shared-heap memory) and dump the head of the object if so. Cause:
        // DestroyContext/cache flush sent with the 13.x DC id, so the firmware
        // never released contexts before reuse (LOG.md 32.41.667zzg).
        self.object.with(|raw, _| {
            let p = raw as *const _ as *const u64;
            // SAFETY: the raw Scene is at least 0x40 bytes of plain data.
            let w: [u64; 8] = core::array::from_fn(|i| unsafe { p.add(i).read_volatile() });
            if (w[1] >> 40) != 0xfffffc {
                kernel::pr_err!(
                    "asahi/scenecheck: Scene {:#x} control_word2 corrupt: {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}\n",
                    u64::from(self.object.weak_pointer()),
                    w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]
                );
            }
        });
        {
            let (pass, total) = self.object.with(|raw, _| {
                (
                    raw.pass_page_count.load(Ordering::Relaxed),
                    raw.total_page_count.load(Ordering::Relaxed),
                )
            });
            let inner = self.object.buffer.inner.lock();
            let (blocks, pages, last) = inner.info.with(|raw, _| {
                (
                    raw.block_count.load(Ordering::Relaxed),
                    raw.page_count.load(Ordering::Relaxed),
                    raw.last_page.load(Ordering::Relaxed),
                )
            });
            let (ctl_total, ctl_wptr) = inner.info.block_ctl.with(|raw, _| {
                (
                    raw.total.load(Ordering::Relaxed),
                    raw.wptr.load(Ordering::Relaxed),
                )
            });
            core::mem::drop(inner);
            crate::sub_pr_info!(
                "asahi/tiler: pass_pages {} total_pages {} blocks {} pages {} last_page {} ctl_total {} ctl_wptr {}\n",
                pass,
                total,
                blocks,
                pages,
                last,
                ctl_total,
                ctl_wptr,
            );
        }
        let mut inner = self.object.buffer.inner.lock();
        assert_ne!(inner.active_scenes, 0);
        inner.active_scenes -= 1;

        if inner.active_scenes == 0 {
            mod_pr_debug!(
                "Buffer: no scenes left, dropping slot {}",
                inner.active_slot.take().unwrap().slot()
            );
            inner.active_slot = None;
        }
    }
}

/// Inner data for a single TVB buffer object.
#[versions(AGX)]
struct BufferInner {
    /// Handed to the manager's `info_pool` on drop (if `info_pooled`), never freed.
    info: ManuallyDrop<GpuObject<buffer::Info::ver>>,
    info_pooled: bool,
    ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
    ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
    blocks: KVec<GpuOnlyArray<u8>>,
    /// `tvb_cpu=1`: TVB blocks allocated CPU-mapped instead, so the tile
    /// lists can be read back. LOG.md 32.41.614.
    cpu_blocks: KVec<GpuArray<u8>>,
    max_blocks: usize,
    max_blocks_nomemless: usize,
    mgr: BufferManager::ver,
    active_scenes: usize,
    active_slot: Option<slotalloc::Guard<BufferSlotInner::ver>>,
    last_token: Option<slotalloc::SlotToken>,
    tpc: Option<Arc<GpuArray<u8>>>,
    kernel_buffer: GpuArray<u8>,
    stats: GpuObject<buffer::Stats>,
    cfg: &'static hw::HwConfig,
    preempt1_size: usize,
    preempt2_size: usize,
    preempt3_size: usize,
    num_clusters: usize,
    /// The VM's UMA pool slot, filled by the first scene of any queue. LOG.md 32.41.664.
    uma: UmaSlot,
    /// .667zzz33: blocks added while a scene was in flight, not yet handed to the firmware.
    info_pending: bool,
    /// Block count the firmware's PB descriptor was last (re)built from.
    published_blocks: usize,
}

#[versions(AGX)]
impl Drop for BufferInner::ver {
    fn drop(&mut self) {
        // SAFETY: `info` is never touched again after this.
        let info = unsafe { ManuallyDrop::take(&mut self.info) };
        if !self.info_pooled {
            drop(info);
            return;
        }
        // No scene holds this Buffer, so the slot lock is not held here
        // (`owners` keeps the Buffer alive while it has a slot).
        if let Err(info) = self.mgr.0.with_inner(|m| {
            if m.info_pool.reserve(1, GFP_KERNEL).is_err() {
                return Err(info);
            }
            m.info_pool.push(info, GFP_KERNEL).ok();
            Ok(())
        }) {
            pr_warn!("Buffer: leaking a dead Info (no memory for its pool slot)\n");
            core::mem::forget(info);
        }
    }
}

/// Locked and reference counted TVB buffer.
#[versions(AGX)]
pub(crate) struct Buffer {
    inner: Arc<Mutex<BufferInner::ver>>,
}

#[versions(AGX)]
impl Buffer::ver {
    /// Create a new Buffer for a given VM, given the per-VM allocators.
    pub(crate) fn new(
        gpu: &dyn gpu::GpuManager,
        alloc: &mut gpu::KernelAllocators,
        ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
        ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
        mgr: &BufferManager::ver,
        uma: UmaSlot,
    ) -> Result<Buffer::ver> {
        // These are the typical max numbers on macOS.
        // 8GB machines have this halved.
        let max_size: usize = 862_322_688; // bytes
        let max_size_nomemless = max_size / 3;

        let max_blocks = max_size / BLOCK_SIZE;
        let max_blocks_nomemless = max_size_nomemless / BLOCK_SIZE;
        let max_pages = max_blocks * PAGES_PER_BLOCK;
        let max_pages_nomemless = max_blocks_nomemless * PAGES_PER_BLOCK;

        let num_clusters = gpu.get_dyncfg().id.num_clusters as usize;
        let num_clusters_adj = if num_clusters > 1 {
            num_clusters + 1
        } else {
            1
        };

        let preempt1_size = num_clusters_adj * gpu.get_cfg().preempt1_size;
        let preempt2_size = num_clusters_adj * gpu.get_cfg().preempt2_size;
        let preempt3_size = num_clusters_adj * gpu.get_cfg().preempt3_size;

        if *module_parameters::buf_addr_shift.value() != 0 {
            let a = alloc.private.array_empty_tagged::<u8>(0x400, b"SHFT")?;
            let b = alloc.shared.array_empty_tagged::<u8>(0x400, b"SHFT")?;
            let c = alloc.gpu.array_empty_tagged::<u8>(0x4000, b"SHFT")?;
            pr_info!(
                "asahi/bufshift: leaked {:#x} {:#x} {:#x}\n",
                u64::from(a.gpu_pointer()),
                u64::from(b.gpu_pointer()),
                u64::from(c.gpu_pointer())
            );
            core::mem::forget(a);
            core::mem::forget(b);
            core::mem::forget(c);
        }
        let shared = &mut alloc.shared;
        // 32.41.600: on 15.5 the firmware CPU flushes/reads the page and block
        // lists itself (fw:0x28d40, crash FAR = the job-VM page list), and the
        // PB descriptor accepts a high (TTB1) address, so info_155 & 4 puts
        // both lists in kernel GPU+FW shared memory.
        let kgpu = &mut alloc.gpu;
        let lists_kernel = *module_parameters::info_155.value() & 4 != 0;
        // .667zzz39: the firmware never forgets an Info (see BufferManagerInner::
        // info_pool), so a dead Buffer's Info is reused rather than freed.
        let pooled = if lists_kernel {
            mgr.0.with_inner(|m| m.info_pool.pop())
        } else {
            None
        };
        let info = if let Some(mut info) = pooled {
            Self::reset_info(&mut info);
            info
        } else { alloc.private.new_init(
            {
                let ualloc_priv = &ualloc_priv;
                try_init!(buffer::Info::ver {
                    block_ctl: shared.new_default::<buffer::BlockControl>()?,
                    counter: shared.new_default::<buffer::Counter>()?,
                    page_list: if lists_kernel {
                        kgpu.array_empty_tagged(max_pages, b"PLST")?
                    } else {
                        ualloc_priv.lock().array_empty_tagged(max_pages, b"PLST")?
                    },
                    block_list: if lists_kernel {
                        kgpu.array_empty_tagged(max_blocks * 2, b"BLST")?
                    } else {
                        ualloc_priv.lock().array_empty_tagged(max_blocks * 2, b"BLST")?
                    },
                })
            },
            |inner, _p| {
                let poison = |off: u64| -> u64 {
                    match *module_parameters::info_poison.value() {
                        0 => 0,
                        base => base | (off << 12),
                    }
                };
                try_init!(buffer::raw::Info::ver {
                    gpu_counter: 0x0,
                    unk_4: 0,
                    last_id: 0x0,
                    cur_id: -1,
                    unk_10: 0x0,
                    gpu_counter2: 0x0,
                    // 32.41.600: fw:0x1d100 (vertex start) builds the PB
                    // descriptor (fw:0x28d40) only if Info+0x18 != 0; else the
                    // PB table stays empty and the TA runs out of memory.
                    unk_18: if *module_parameters::info_155.value() & 2 != 0 { 1 } else { 0x0 },
                    #[ver((V < V13_0B4 || G >= G14X) && V < V15_5)]
                    unk_1c: 0x0,
                    page_list: inner.page_list.gpu_pointer(),
                    #[ver(V >= V15_5)]
                    // 32.41.600: 15.5's PB descriptor (fw:0x28f20 / 0x292c0)
                    // takes the hardware's page-list address from Info+0x24.
                    unk_24: {
                        let mut a = [0u8; 8];
                        if *module_parameters::info_155.value() & 1 != 0 {
                            a.copy_from_slice(&u64::from(inner.page_list.gpu_pointer()).to_le_bytes());
                        } else if *module_parameters::info_poison.value() != 0 {
                            a.copy_from_slice(&poison(0x24).to_le_bytes());
                        }
                        Array::new(a)
                    },
                    page_list_size: (4 * max_pages).try_into()?,
                    page_count: AtomicU32::new(0),
                    max_blocks: max_blocks.try_into()?,
                    block_count: AtomicU32::new(0),
                    unk_38: 0x0,
                    block_list: inner.block_list.gpu_pointer(),
                    block_ctl: inner.block_ctl.gpu_pointer(),
                    last_page: AtomicU32::new(0),
                    // 32.41.600: on 15.5 the firmware reads Info+0x54 as the
                    // block size in bytes: pages/block = (0x20000/bs)*(bs>>15),
                    // and bs < 0x8000 makes the PB descriptor's page count 0
                    // (fw:0x28e94, 0x291f4) -> TA out of memory at once.
                    gpu_page_ptr1: if *module_parameters::info_155.value() & 1 != 0 {
                        BLOCK_SIZE as u32
                    } else {
                        0x0
                    },
                    gpu_page_ptr2: 0x0,
                    unk_58: 0x0,
                    // 32.41.606: on 15.5 the firmware's fragment-completion path
                    // (fw:0x1e6fc) reads Info+0x60 as the Counter pointer and
                    // compares *counter with Info+0 (gpu_counter); with BLOCK_SIZE
                    // here it dereferenced 0x20000 and crashed. info_155 & 8 puts
                    // the Counter address at +0x60 (this word + low half of unk_60).
                    block_size: if *module_parameters::info_155.value() & 8 != 0 {
                        u64::from(inner.counter.gpu_pointer()) as u32
                    } else {
                        BLOCK_SIZE as u32
                    },
                    // `info_poison` fills every unknown slot of the buffer
                    // manager's Info with a distinct never-mapped address:
                    // 0x3bad0NN000, where NN is the field's byte offset. The
                    // *hardware* page manager walks this struct, and the 15.5
                    // layout was fitted to the firmware's accesses rather than
                    // known (32.41.15), so a pointer the hardware expects at an
                    // offset the driver leaves zero would read as VA 0 -- which
                    // is exactly `gUPM`'s fault. LOG.md 32.41.58.
                    unk_60: if *module_parameters::info_155.value() & 8 != 0 {
                        U64(u64::from(inner.counter.gpu_pointer()) >> 32)
                    } else {
                        U64(poison(0x60))
                    },
                    counter: inner.counter.gpu_pointer(),
                    unk_70: poison(0x70) as u32,
                    unk_74: (poison(0x70) >> 32) as u32,
                    unk_78: poison(0x78) as u32,
                    unk_7c: (poison(0x78) >> 32) as u32,
                    unk_80: if *module_parameters::info_poison.value() != 0 { poison(0x80) as u32 } else { 0x1 },
                    max_pages: max_pages.try_into()?,
                    max_pages_nomemless: max_pages_nomemless.try_into()?,
                    unk_8c: 0x0,
                    unk_90: {
                        let mut a = [0u8; 0x30];
                        if *module_parameters::info_poison.value() != 0 {
                            for k in 0..6usize {
                                let v = poison(0x90 + (k as u64) * 8);
                                a[k * 8..k * 8 + 8].copy_from_slice(&v.to_le_bytes());
                            }
                        }
                        Array::new(a)
                    },
                })
            },
        )? };

        // Technically similar to Scene below, let's play it safe.
        let kernel_buffer = alloc.shared.array_empty_tagged(0x40, b"KBUF")?;
        let stats = alloc
            .shared
            .new_object(Default::default(), |_inner| buffer::raw::Stats {
                reset: AtomicU32::from(1),
                ..Default::default()
            })?;

        Ok(Buffer::ver {
            inner: Arc::pin_init(
                new_mutex!(BufferInner::ver {
                    info: ManuallyDrop::new(info),
                    info_pooled: lists_kernel,
                    ualloc,
                    ualloc_priv,
                    blocks: KVec::new(),
                    cpu_blocks: KVec::new(),
                    max_blocks,
                    max_blocks_nomemless,
                    mgr: mgr.clone(),
                    active_scenes: 0,
                    active_slot: None,
                    last_token: None,
                    tpc: None,
                    kernel_buffer,
                    stats,
                    cfg: gpu.get_cfg(),
                    preempt1_size,
                    preempt2_size,
                    preempt3_size,
                    num_clusters,
                    uma,
                    info_pending: false,
                    published_blocks: 0,
                }),
                GFP_KERNEL,
            )?,
        })
    }

    /// Returns a pooled Info to the state a fresh one is built in. Its lists are
    /// rewritten from block 0 as the new Buffer grows.
    fn reset_info(info: &mut GpuObject<buffer::Info::ver>) {
        info.with_mut(|raw, inner| {
            raw.gpu_counter = 0;
            raw.unk_4 = 0;
            // Not a slot id: the first slot assignment then rebuilds the PB descriptor.
            raw.last_id = -1;
            raw.cur_id = -1;
            raw.unk_10 = 0;
            raw.gpu_counter2 = 0;
            raw.unk_18 = if *module_parameters::info_155.value() & 2 != 0 { 1 } else { 0 };
            raw.page_count.store(0, Ordering::Relaxed);
            raw.block_count.store(0, Ordering::Relaxed);
            raw.unk_38 = 0;
            raw.last_page.store(0, Ordering::Relaxed);
            raw.gpu_page_ptr2 = 0;
            raw.unk_58 = 0;
            inner.block_ctl.with(|c, _| {
                c.total.store(0, Ordering::Relaxed);
                c.wptr.store(0, Ordering::Relaxed);
                c.unk.store(0, Ordering::Relaxed);
            });
            inner.counter.with(|c, _| c.count.store(0, Ordering::Relaxed));
        });
    }

    /// Returns the total block count allocated to this Buffer.
    pub(crate) fn block_count(&self) -> u32 {
        { let i = self.inner.lock(); (i.blocks.len() + i.cpu_blocks.len()) as u32 }
    }

    /// 667zzw: TVB high-water mark in pages, from the sentinels written into the
    /// first word of every page before the block was handed to the GPU. The 15.5
    /// firmware leaves `Stats.max_pages` (and the scene page counters) at zero, so
    /// this is the only usage signal: a page whose sentinel is gone has been used
    /// by some job. Read-only, so safe while jobs are in flight.
    fn tvb_high_water(inner: &BufferInner::ver) -> usize {
        let mut hw = 0;
        for (bi, b) in inner.cpu_blocks.iter().enumerate().rev() {
            let s = b.as_slice();
            for pg in (0..PAGES_PER_BLOCK).rev() {
                let o = pg * PAGE_SIZE;
                // SAFETY: in bounds; the GPU may be writing, hence volatile.
                let v = unsafe { core::ptr::read_volatile(s[o..].as_ptr() as *const u32) };
                if v != TVB_SENTINEL {
                    hw = bi * PAGES_PER_BLOCK + pg + 1;
                    break;
                }
            }
            if hw != 0 {
                break;
            }
        }
        hw
    }

    /// Automatically grow the Buffer based on feedback from the statistics.
    pub(crate) fn auto_grow(&self) -> Result<bool> {
        let inner = self.inner.lock();
        if debug_enabled(DebugFlags::TVBStats) {
            pr_info!(
                "asahi/tvbhw: high water {} of {} pages\n",
                Self::tvb_high_water(&inner),
                (inner.blocks.len() + inner.cpu_blocks.len()) * PAGES_PER_BLOCK
            );
        }

        let used_pages = inner.stats.with(|raw, _inner| {
            let used = raw.max_pages.load(Ordering::Relaxed);
            // 667zzw: is the 15.5 firmware filling these at all?
            if debug_enabled(DebugFlags::TVBStats) {
                pr_info!(
                    "asahi/tvbstats: max_pages {:#x} max_b {:#x} overflows {:#x} gpu_c {:#x} reset {} blocks {}\n",
                    used,
                    raw.max_b.load(Ordering::Relaxed),
                    raw.overflow_count.load(Ordering::Relaxed),
                    raw.gpu_c.load(Ordering::Relaxed),
                    raw.reset.load(Ordering::Relaxed),
                    inner.blocks.len() + inner.cpu_blocks.len()
                );
            }
            raw.reset.store(1, Ordering::Release);
            used as usize
        });

        let need_blocks = (used_pages * 2)
            .div_ceil(PAGES_PER_BLOCK)
            .min(inner.max_blocks_nomemless);
        let want_blocks = (used_pages * 3)
            .div_ceil(PAGES_PER_BLOCK)
            .min(inner.max_blocks_nomemless);

        let cur_count = inner.blocks.len() + inner.cpu_blocks.len();

        if need_blocks <= cur_count {
            Ok(false)
        } else {
            // Grow to 3x requested size (same logic as macOS)
            core::mem::drop(inner);
            self.ensure_blocks(want_blocks)?;
            Ok(true)
        }
    }

    /// Synchronously grow the Buffer.
    pub(crate) fn sync_grow(&self) {
        let inner = self.inner.lock();

        let cur_count = inner.blocks.len() + inner.cpu_blocks.len();
        core::mem::drop(inner);
        if self.ensure_blocks(cur_count + 10).is_err() {
            pr_err!("BufferManager: Failed to grow buffer synchronously\n");
        }
    }

    /// Ensure that the buffer has at least a certain minimum size in blocks.
    pub(crate) fn ensure_blocks(&self, min_blocks: usize) -> Result<bool> {
        let mut inner = self.inner.lock();

        let cur_count = inner.blocks.len() + inner.cpu_blocks.len();
        if cur_count >= min_blocks {
            return Ok(false);
        }
        if min_blocks > inner.max_blocks {
            return Err(ENOMEM);
        }

        let add_blocks = min_blocks - cur_count;
        let new_count = min_blocks;

        let mut new_blocks: KVec<GpuOnlyArray<u8>> = KVec::new();

        // Allocate the new blocks first, so if it fails they will be dropped
        let mut new_cpu: KVec<GpuArray<u8>> = KVec::new();
        let tvb_cpu = *module_parameters::tvb_cpu.value() != 0;
        // tvb_cpu=2: from the GPU/FW-private allocator = MEMATTR_CACHED (Apple maps user
        // memory uncached unless flagged cached). LOG.md 32.41.623.
        let tvb_cached = *module_parameters::tvb_cpu.value() == 2;
        let ualloc_arc = if tvb_cached { inner.ualloc_priv.clone() } else { inner.ualloc.clone() };
        let mut ualloc = ualloc_arc.lock();
        for _i in 0..add_blocks {
            if tvb_cpu {
                let mut b = ualloc.array_empty_tagged(BLOCK_SIZE, b"TVBC")?;
                // 667zzw: usage sentinels, see `tvb_high_water`.
                for pg in 0..PAGES_PER_BLOCK {
                    let o = pg * PAGE_SIZE;
                    b.as_mut_slice()[o..o + 4].copy_from_slice(&TVB_SENTINEL.to_le_bytes());
                }
                new_cpu.push(b, GFP_KERNEL)?;
            } else {
                new_blocks.push(ualloc.array_gpuonly(BLOCK_SIZE)?, GFP_KERNEL)?;
            }
        }
        core::mem::drop(ualloc);

        // Then actually commit them
        inner.blocks.reserve(new_blocks.len(), GFP_KERNEL)?;
        inner.cpu_blocks.reserve(new_cpu.len(), GFP_KERNEL)?;

        let mut vas: KVec<u64> = KVec::new();
        for block in new_blocks.into_iter() {
            vas.push(block.gpu_va().get(), GFP_KERNEL)?;
            inner
                .blocks
                .push(block, GFP_KERNEL)
                .expect("push() failed after reserve()");
        }
        for block in new_cpu.into_iter() {
            let va = u64::from(block.gpu_pointer());
            crate::gpu::dbg_note_tvb(cur_count + vas.len(), block.as_slice().as_ptr() as usize, va);
            vas.push(va, GFP_KERNEL)?;
            inner
                .cpu_blocks
                .push(block, GFP_KERNEL)
                .expect("push() failed after reserve()");
        }

        for (i, va) in vas.into_iter().enumerate() {
            let page_num = (va >> PAGE_SHIFT) as u32;

            inner.info.block_list[2 * (cur_count + i)] = page_num;
            for j in 0..PAGES_PER_BLOCK {
                inner.info.page_list[(cur_count + i) * PAGES_PER_BLOCK + j] = page_num + j as u32;
            }
        }

        inner.info.block_ctl.with(|raw, _inner| {
            raw.total.store(new_count as u32, Ordering::SeqCst);
            raw.wptr.store(new_count as u32, Ordering::SeqCst);
        });

        /* Only do this update if the buffer manager is idle (which means we own it) */
        if inner.active_scenes == 0 {
            Self::publish_info(&mut inner);
        } else {
            inner.info_pending = true;
        }

        // The TVB page pool is the one user-VM structure the *hardware* page
        // manager walks, and `gUPM` -- which is that unit (LOG.md 32.41.53) --
        // faults reading VA 0. An empty or short pool would have it read a zero
        // page number and form exactly that address, so print what the pool
        // actually holds.  LOG.md 32.41.57.
        if crate::debug::verbose() {
        pr_info!(
            "asahi/tvb: blocks {} -> {} (max {}), pages {}, page_list {:#x}[{}], block_list {:#x}, block_ctl {:#x}, first_page {:#x}\n",
            cur_count,
            new_count,
            inner.max_blocks,
            new_count * PAGES_PER_BLOCK,
            u64::from(inner.info.page_list.gpu_pointer()),
            inner.info.page_list.len(),
            u64::from(inner.info.block_list.gpu_pointer()),
            u64::from(inner.info.block_ctl.gpu_pointer()),
            inner.info.page_list[0],
        );
        }

        Ok(true)
    }

    /// Hand the current block count to the firmware. On 15.5 the PB descriptor
    /// (page count, ring indexes) is only rebuilt by fw:0x28d40 while Info+0x18
    /// != 0, so a grown heap stays invisible to the TA without it: a 16384x16384
    /// scene after a small one ran out at the old 2048 pages and hung
    /// (dEQP-GLES31 fbo.no_attachments.maximums.all). The rebuild resets the
    /// ring, so only call this with no scene in flight. LOG.md 32.41.667zzz33.
    fn publish_info(inner: &mut BufferInner::ver) {
        let count = inner.blocks.len() + inner.cpu_blocks.len();
        let page_count = (count * PAGES_PER_BLOCK) as u32;
        inner.info.with(|raw, _inner| {
            raw.page_count.store(page_count, Ordering::Relaxed);
            raw.block_count.store(count as u32, Ordering::Relaxed);
            raw.last_page.store(page_count.saturating_sub(1), Ordering::Relaxed);
        });
        #[ver(V >= V15_5)]
        if *module_parameters::info_155.value() & 2 != 0 && count != inner.published_blocks {
            inner.info.with_mut(|raw, _inner| raw.unk_18 = 1);
        }
        inner.published_blocks = count;
        inner.info_pending = false;
    }

    /// Hand the grown heap to the firmware although a scene is in flight. Only
    /// for hang recovery: the hung scene's TA is replayed from its start, and
    /// without the rebuild the replay runs out at the old size again (the
    /// "replay never finishes even with a 4x heap" of .667zzz29). .667zzz33
    pub(crate) fn force_publish(&self) {
        let mut inner = self.inner.lock();
        Self::publish_info(&mut inner);
    }

    /// Block count the firmware currently sees.
    pub(crate) fn published_blocks(&self) -> usize {
        self.inner.lock().published_blocks
    }

    /// Wait (up to `ms`) until no scene of this buffer is in flight.
    pub(crate) fn wait_idle(&self, ms: u32) -> bool {
        for _ in 0..ms {
            if self.inner.lock().active_scenes == 0 {
                return true;
            }
            kernel::time::delay::fsleep(kernel::time::Delta::from_millis(1));
        }
        self.inner.lock().active_scenes == 0
    }

    /// Create a new [`Scene::ver`] (render pass) using this buffer.
    pub(crate) fn new_scene(
        &self,
        alloc: &mut gpu::KernelAllocators,
        tile_info: &TileInfo,
    ) -> Result<Scene::ver> {
        let mut inner = self.inner.lock();

        let tilemap_size = tile_info.tilemap_size;
        let tpc_size = tile_info.tpc_size;

        // TODO: what is this exactly?
        mod_pr_debug!("Buffer: Allocating TVB buffers\n");

        // This seems to be a list, with 4x2 bytes of headers and 8 bytes per entry.
        // On single-cluster devices, the used length always seems to be 1.
        // On M1 Ultra, it can grow and usually doesn't exceed 64 entries.
        // macOS allocates a whole 64K * 0x80 for this, so let's go with
        // that to be safe...
        let user_buffer = inner.ualloc.lock().array_empty_tagged(
            if inner.num_clusters > 1 {
                0x10080
            } else {
                0x80
            },
            b"UBUF",
        )?;

        // scene_cached=1: tilemap + heap meta from the GPU/FW-private (cached
        // memattr) allocator, like the TVB with tvb_cpu=2. LOG.md 32.41.629.
        let scene_alloc = if *module_parameters::scene_cached.value() != 0 {
            inner.ualloc_priv.clone()
        } else {
            inner.ualloc.clone()
        };
        let tvb_heapmeta = scene_alloc
            .lock()
            .array_empty_tagged(0x200 + tile_info.layermeta_size, b"HMTA")?;
        let tvb_tilemap = scene_alloc
            .lock()
            .array_empty_tagged(tilemap_size, b"TMAP")?;

        // 32.41.613: remember the tilemap/heapmeta CPU views for the post-job dump.
        crate::sub_pr_info!("asahi/scenebuf: TMAP gpu {:#x} HMTA gpu {:#x}\n", tvb_tilemap.gpu_va().get(), tvb_heapmeta.gpu_va().get());
        crate::gpu::dbg_note_buf(0, "TMAP", tvb_tilemap.as_slice().as_ptr() as usize, tilemap_size);
        crate::gpu::dbg_note_buf(1, "HMTA", tvb_heapmeta.as_slice().as_ptr() as usize, 0x200 + tile_info.layermeta_size);

        mod_pr_debug!("Buffer: Allocating misc buffers\n");
        let preempt_buf = inner.ualloc.lock().array_empty_tagged(
            inner.preempt1_size + inner.preempt2_size + inner.preempt3_size,
            b"PRMT",
        )?;

        let tpc = match inner.tpc.as_ref() {
            Some(buf) if buf.len() >= tpc_size => buf.clone(),
            _ => {
                // MacOS allocates this as shared GPU+FW, but
                // priv seems to work and might be faster?
                // Needs to be FW-writable anyway, so ualloc
                // won't work.
                let buf = Arc::new(
                    inner.ualloc_priv.lock().array_empty_tagged(
                        (tpc_size + mmu::UAT_PGMSK) & !mmu::UAT_PGMSK,
                        b"TPC ",
                    )?,
                    GFP_KERNEL,
                )?;
                inner.tpc = Some(buf.clone());
                buf
            }
        };

        let mut clmeta_size = 0;
        let mut meta1_size = 0;
        let mut meta2_size = 0;
        let mut meta3_size = 0;

        let clustering = if inner.num_clusters > 1 {
            let cfg = inner.cfg.clustering.as_ref().unwrap();

            clmeta_size = tile_info.layermeta_size * cfg.max_splits;
            // Maybe: (4x4 macro tiles + 1 global page)*n, 32bit each (17*4*n)
            // Unused on t602x?
            meta1_size = align(tile_info.meta1_blocks as usize * cfg.meta1_blocksize, 0x80);
            meta2_size = align(cfg.meta2_size, 0x80);
            meta3_size = align(cfg.meta3_size, 0x80);
            let meta4_size = cfg.meta4_size;

            let meta_size = clmeta_size + meta1_size + meta2_size + meta3_size + meta4_size;

            mod_pr_debug!("Buffer: Allocating clustering buffers\n");
            let tilemaps = inner
                .ualloc
                .lock()
                .array_empty_tagged(cfg.max_splits * tilemap_size, b"CTMP")?;
            let meta = inner.ualloc.lock().array_empty_tagged(meta_size, b"CMTA")?;
            crate::sub_pr_info!("asahi/scenebuf: CTMP gpu {:#x} CMTA gpu {:#x}\n", tilemaps.gpu_va().get(), meta.gpu_va().get());
            crate::gpu::dbg_note_buf(2, "CTMP", tilemaps.as_slice().as_ptr() as usize, cfg.max_splits * tilemap_size);
            crate::gpu::dbg_note_buf(3, "CMTA", meta.as_slice().as_ptr() as usize, meta_size);
            Some(buffer::ClusterBuffers { tilemaps, meta })
        } else {
            None
        };

        // Could be made strong, but we wind up with a deadlock if we try to grab the
        // pointer through the inner.buffer path inside the closure.
        let stats_pointer = inner.stats.weak_pointer();

        // 🔑 The 15.5 UMA pool is a hardware aperture descriptor, and the driver
        // has been handing the firmware a zeroed one. `fw:0x24d94..0x24dd8`
        // builds the descriptor out of it:
        //
        //     word0 = (pool[0x1c] << 41) | (pool[0x14] >> 7)
        //     word1 = (pool[0x24] << 33) | (pool[0x20] << 5) | (pool[0x28] << 61)
        //     word2 =  pool[0x2c]
        //
        // and writes it to a 32-byte slot indexed by the pool id at `pool[8]`.
        // `pool[0x14] >> 7` is a 128-byte-aligned **base address** -- so with the
        // pool zeroed the aperture is based at VA 0, which is what `gUPM` has
        // been reading since 32.41.40. Back it with a real allocation in the
        // client's VM. LOG.md 32.41.51.
        // 32.41.664: with Apple's ring capacity (record+0x1c = 0x100000) the
        // hardware frees pages to list[tail % 0x100000], tail starting at the
        // page count, so the list must span the full capacity (8 MiB), not
        // `uma_aperture_size` -- the compute pool overran a `pages`-sized list.
        // 32.41.664: ONE UMA pool per Buffer (render queue), built on the first
        // scene and kept, like Apple's per-process AGXUMAFList (its live record's
        // head/tail reached 0x400a0/0x468a0 on one record). Rebuilding it per
        // scene re-registered a new 8192-page list under the same id every frame
        // (duplicate pages in the hardware ring) and a second context's pool was
        // never pushed -> TA out-of-memory on its first kick.
        let uma_existing = inner.uma.lock().clone();
        let uma: Arc<buffer::UmaPool> = if let Some(u) = uma_existing {
            u
        } else if let Some(u) = uma_shared_pool(alloc)? {
            u
        } else {
            let uma_list_size = if *module_parameters::uma_real.value() != 0
                && *module_parameters::pool_apple.value() != 0
            {
                core::cmp::max(*module_parameters::uma_aperture_size.value(), 0x100000 * 8)
            } else {
                *module_parameters::uma_aperture_size.value()
            };
            // 667k: `uma_klow` = Apple's arrangement (kernel-lower list, ctx 63).
            let mut uma_aperture = if *module_parameters::uma_klow.value() != 0 {
                alloc.klow.array_empty_tagged(uma_list_size, b"UMAA")?
            } else {
                inner
                    .ualloc
                    .lock()
                    .array_empty_tagged(uma_list_size, b"UMAA")?
            };

            // `AGXUMAFList` is a free *list*, and `pool[0x14]` is what it is based
            // at -- so this allocation is very likely the list itself, not the
            // pages. Zeroed, its first entry reads 0, and a page manager that pops
            // entry 0 and dereferences it lands on exactly the VA 0 this has been
            // faulting at since 32.41.40, without ever faulting on the list read
            // itself (that address is mapped). `uma_list_probe` fills the buffer
            // with self-describing words so the next fault address names both the
            // entry width and how the entry is turned into an address:
            //
            //   8-byte address entries -> fault at `probe`
            //   4-byte page numbers    -> fault at `(probe & 0xffffffff) << shift`
            //
            // LOG.md 32.41.80.
            if *module_parameters::uma_list_probe.value() != 0 {
                let probe = *module_parameters::uma_list_probe.value();
                let list = uma_aperture.as_mut_slice();
                let n = list.len() / 8;
                for i in 0..n {
                    let v = probe | ((i as u64) << 12);
                    list[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
                }
            }

            // A separate buffer for the second pool register. Pointing both
            // registers at the same memory is not obviously wrong but is not
            // obviously right either, and macOS's pool struct carries three
            // distinct GPU addresses. LOG.md 32.41.88.
            // 667zzj: with `uma_klow` every table the page manager touches (list,
            // scratch, extents) is kernel-lower; gUPM walks them through the JOB's
            // context (fault vm = job slot with DC+4 = DC+8 = 63), so klow's
            // top-level slot is shared into every user VM (mmu.rs KLOW_TOP_IDX).
            let uma_klow = *module_parameters::uma_klow.value() != 0;
            let uma_scratch = if uma_klow {
                alloc.klow.array_empty_tagged(*module_parameters::uma_aperture_size.value(), b"UMAS")?
            } else {
                inner
                    .ualloc
                    .lock()
                    .array_empty_tagged(*module_parameters::uma_aperture_size.value(), b"UMAS")?
            };

            // LOG.md 32.41.540: AGXUMAFList::prepareBufferResources backs the pool
            // with real pages. `uma_real` gives it one 4 MiB block.
            let uma_real = *module_parameters::uma_real.value() != 0;
            let uma_backing = inner
                .ualloc
                .lock()
                .array_empty_tagged(
                    if uma_real {
                        // `uma_real` = MiB; 1 keeps the original 4 MiB. LOG.md 32.41.541.
                        let mib = *module_parameters::uma_real.value() as usize;
                        if mib <= 1 { 0x400000 } else { mib << 20 }
                    } else {
                        0x4000
                    },
                    b"UMAB",
                )?;
            // 667zzo: when a pool's firmware slot is empty at job start
            // (fw:0x25034, e.g. after a recovery) the firmware re-pushes the pool
            // by walking pool[0x34] for pool[0x44] entries (fw:0x25118 ->
            // 0x246b8), and on the GPU pool[0x44] becomes extents + pages
            // (0x40 + 0x2000; Firefox: gUPM read fault at extents + 0x4000,
            // recovery empties the slot again -> a fault/timeout loop). Size the
            // table for the worst case; entries past the real ones stay 0 pages.
            let mut uma_extents = if uma_klow {
                alloc.klow.array_empty_tagged(UMA_EXTENTS_SIZE, b"UMAE")?
            } else {
                inner.ualloc.lock().array_empty_tagged(UMA_EXTENTS_SIZE, b"UMAE")?
            };

            let _gpu = &mut alloc.gpu;
            // The word `pool[0x48]` points at. `fw:0x252a0` loads *through* it, so
            // it cannot be null once the firmware has installed a descriptor for
            // this pool -- that is a data abort inside the firmware with FAR = 0.
            // LOG.md 32.41.90.
            let uma_sync = _gpu.array_empty_tagged(0x40, b"UMAQ")?;
            let uma_sync_addr = u64::from(uma_sync.gpu_pointer());
            // 0x90-byte record + (32.41.598) a self-pointer word at +0xa0: FinalizeFragment's
            // pool_ptr_ptr must point OUTSIDE RunFragment (Apple's does), because the
            // firmware stores a timestamp over the in-command `pool_self` (cmd+0xc44).
            let mut uma_pool = _gpu.array_empty_tagged(0x100, b"UMAP")?;
            let uma_token = UMA_TOKEN_SEQ.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            {
                let self_va = u64::from(uma_pool.gpu_pointer());
                uma_pool.as_mut_slice()[0xa0..0xa8].copy_from_slice(&self_va.to_le_bytes());
            }
            if *module_parameters::uma_aperture_size.value() != 0 {
                let base = match *module_parameters::uma_aperture_probe.value() {
                    0 => u64::from(uma_aperture.gpu_pointer()),
                    v => v,
                };
                crate::uma_pool_init(
                    uma_pool.as_mut_slice(),
                    base,
                    *module_parameters::uma_aperture_size.value() as u64,
                    uma_sync_addr,
                );
            }
            // LOG.md 32.41.536: Apple's firmware appended 0x14308 = 0x5 for the same job
            // (capture4, stale-but-real entries past `count`): pool id 5, no base.
            if *module_parameters::uma_apple.value() != 0 {
                // 32.41.664: a fresh id per pool (rotating), as Apple gives each
                // process's UMAFList its own; re-registering id 5 for a new context's
                // pool was never pushed by the firmware.
                // 667zzh: ids are owned (free list, LRU steal), not rotated.
                let id = uma_alloc_id(uma_token);
                uma_pool.as_mut_slice()[0x08..0x0c].copy_from_slice(&id.to_le_bytes());
            }
            let uma_fwbuf = _gpu.array_empty_tagged(0x100000, b"UMAF")?;

            // LOG.md 32.41.540, from AGXUMAFList::prepareBufferResources
            // (kext 0xfffffe0009150254): per backing block of `size` bytes at `blk`,
            //   page list  (pool[0x14]): u64 entry[i] = blk + i*0x1000
            //   extent list(pool[0x34]): u64 = (blk>>12)<<12 | min(pages,0xfff)<<52,
            //                            8-aligned per block; count at pool[0x44]
            // and populateInitialFirmwareState: [0x1c] pages/block, [0x24] pages %
            // per_block, [0x28] empty, [0x2c] total pages.
            if uma_real {
                let bva = u64::from(uma_backing.gpu_pointer());
                let pages = uma_backing.as_slice().len() >> 12;
                pr_info!("asahi/scenebuf: UMAB gpu {:#x} ({:#x} bytes)\n", bva, uma_backing.as_slice().len());
                crate::gpu::dbg_note_buf(4, "UMAB", uma_backing.as_slice().as_ptr() as usize, uma_backing.as_slice().len());
                let list = uma_aperture.as_mut_slice();
                let n = core::cmp::min(pages, list.len() / 8);
                for i in 0..n {
                    let v = bva + ((i as u64) << 12);
                    list[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
                }
                // One extent per 4 MiB block (1024 pages <= 0xfff), at an 8-aligned
                // index per block, exactly as prepareBufferResources lays them out.
                const PPB: usize = 1024;
                let nblk = (pages + PPB - 1) / PPB;
                let mut idx = 0usize;
                let mut ext = 0u64;
                {
                    let e = uma_extents.as_mut_slice();
                    for b in 0..nblk {
                        idx = (idx + 7) & !7;
                        let va = bva + ((b * PPB) as u64) * 0x1000;
                        let n = core::cmp::min(PPB, pages - b * PPB) as u64;
                        let v = ((va >> 12) << 12) | ((n & 0xfff) << 52);
                        if b == 0 {
                            ext = v;
                        }
                        if idx * 8 + 8 <= e.len() {
                            e[idx * 8..idx * 8 + 8].copy_from_slice(&v.to_le_bytes());
                        }
                        idx += 1;
                    }
                }
                let ext_count = idx as u32;
                let lva = u64::from(uma_aperture.gpu_pointer());
                let eva = u64::from(uma_extents.gpu_pointer());
                let p = uma_pool.as_mut_slice();
                p[0x14..0x1c].copy_from_slice(&lva.to_le_bytes());
                p[0x1c..0x20].copy_from_slice(&(PPB as u32).to_le_bytes());
                p[0x20..0x24].copy_from_slice(&0u32.to_le_bytes());
                p[0x24..0x28].copy_from_slice(&((pages % PPB) as u32).to_le_bytes());
                p[0x28..0x2c].copy_from_slice(&0u32.to_le_bytes());
                p[0x2c..0x30].copy_from_slice(&(pages as u32).to_le_bytes());
                p[0x34..0x3c].copy_from_slice(&eva.to_le_bytes());
                p[0x44..0x48].copy_from_slice(&ext_count.to_le_bytes());
                // LOG.md 32.41.548: Apple's live pool record (2026-09-26 capture,
                // poolrec.pre.bin): [0]=0xb [0xc]=1 [0x10]=1, pages/block
                // [0x1c]=0x100000 with [0x24]=total (no blocking), extent count =
                // 8 per 4 MiB block (0xb8 for 0x5c00 pages).
                // LOG.md 32.41.599: [0x20]/[0x24] are the free-page ring's head
                // and tail, [0x1c] its capacity (descriptor fw:0x24da4: w0 |=
                // [0x1c]<<41, w1 = [0x20]<<5 | [0x24]<<33). Apple's in-use record
                // has [0x20]=0x400a0 [0x24]=0x468a0 = head + 0x6800 pages. With
                // capacity 1024, pages % 1024 == 0 for every pool we built, so
                // tail == head: an EMPTY ring -> TA out-of-memory (event bit 4).
                // Mode 2 fixes only the ring, without Apple's [0xc]=1 (0x138dc).
                if *module_parameters::pool_apple.value() == 2 {
                    p[0x1c..0x20].copy_from_slice(&0x100000u32.to_le_bytes());
                    p[0x24..0x28].copy_from_slice(&(pages as u32).to_le_bytes());
                    p[0x44..0x48].copy_from_slice(&((nblk * 8) as u32).to_le_bytes());
                } else if *module_parameters::pool_apple.value() != 0 {
                    p[0x00..0x04].copy_from_slice(&0xbu32.to_le_bytes());
                    p[0x0c..0x10].copy_from_slice(&1u32.to_le_bytes());
                    p[0x10..0x14].copy_from_slice(&1u32.to_le_bytes());
                    p[0x1c..0x20].copy_from_slice(&0x100000u32.to_le_bytes());
                    p[0x24..0x28].copy_from_slice(&(pages as u32).to_le_bytes());
                    p[0x44..0x48].copy_from_slice(&((nblk * 8) as u32).to_le_bytes());
                }
                if crate::debug::verbose() {
                pr_info!(
                    "asahi/uma: real pool id {} list {:#x} backing {:#x} pages {} extents {:#x} n {} [{:#x}]\n",
                    u32::from_le_bytes([p[8], p[9], p[10], p[11]]), lva, bva, pages, eva, ext_count, ext
                );
                for r in 0..5 {
                    let w = |i: usize| u32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]);
                    pr_info!(
                        "asahi/poolrec built +{:#04x}: {:08x} {:08x} {:08x} {:08x}\n",
                        r * 16, w(r * 16), w(r * 16 + 4), w(r * 16 + 8), w(r * 16 + 12),
                    );
                }
                }
                crate::gpu::DBG_POOLREC
                    .store(p.as_ptr() as u64, core::sync::atomic::Ordering::Relaxed);
            }
            let uma_total_pages = if uma_real { (uma_backing.as_slice().len() >> 12) as u32 } else { 0 };
            let u = Arc::new(
                buffer::UmaPool {
                    pool: uma_pool,
                    aperture: uma_aperture,
                    scratch: uma_scratch,
                    sync: uma_sync,
                    fwbuf: uma_fwbuf,
                    backing: uma_backing,
                    extents: uma_extents,
                    announced: core::sync::atomic::AtomicBool::new(false),
                    token: uma_token,
                    slot: core::sync::atomic::AtomicU32::new(u32::MAX),
                    grow: [const { core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()) };
                        buffer::UMA_GROW_SLOTS],
                    total_pages: core::sync::atomic::AtomicU32::new(uma_total_pages),
                    growing: core::sync::atomic::AtomicBool::new(false),
                    last_rec: core::sync::atomic::AtomicU64::new(u64::MAX),
                    hwid: core::sync::atomic::AtomicU32::new(1),
                    ctx: core::sync::atomic::AtomicU32::new(UMA_KCTX),
                    ualloc: Some(inner.ualloc.clone()),
                    pool_b: None,
                    aperture_b: None,
                    rec_b: core::sync::atomic::AtomicBool::new(false),
                },
                GFP_KERNEL,
            )?;
            *inner.uma.lock() = Some(u.clone());
            u
        };

        // 🔑 `raw::Scene.control_word`/`control_word2` are Apple's
        // `BufferThing.unk0_addr`/`unk0_addr2`, and `pbb + 0x00` -- the value
        // register `0x1c910` carries -- is the first of the two.  m1n1's
        // `BufferThing` settles the identification: its G14X layout is
        // `unk0_addr` @0x00, `unk0_addr2` @0x08, `unk_0`/`unk_8`/`unk_10`, then
        // `unkptr_18` @0x28, which is exactly this struct's `control_word` /
        // `control_word2` / `user_buffer`.  The kernel's tiler list reads
        // `*(pbb + 0x00)` for `0x1c910` and `*(pbb + 0x28)` for
        // `0x1ca30`/`0x16c39`, and `+0x28` is already proven to be
        // `user_buffer`.  LOG.md 32.41.244.
        //
        // The two halves cannot be the same allocation here, because their
        // consumers live in different address spaces:
        //
        //   +0x08  the *firmware CPU* dereferences, at fw:0x15f20
        //            ldr x8,[x22]; ldr x9,[x8,#8]; ldr w10,[x9]
        //            str x10,[x8,#16]; str wzr,[x9]
        //          -- a read-and-clear of a u32 page counter into
        //          `pass_page_count` (+0x10).  Pointing it at a file-VM address
        //          is an instant firmware data abort, ESR 0x96000004 with FAR
        //          equal to the pointer.  It has to stay in the kernel VM.
        //
        //   +0x00  gPM reads, via `0x1c910`.  A kernel VA cannot even be
        //          expressed there: the register is 40 bits and the fold keeps
        //          only bits 1..38 plus bit 42, so `0xfffffe0c…` truncates and
        //          faults (`FAULT 0x1c00000180 unit gPM_SP`, 32.41.229).  It
        //          has to be a file-VM address, like `page_list`/`block_list`
        //          -- the other two buffers gPM walks, both from `ualloc_priv`.
        //
        // macOS presumably maps one page at both, which this driver has no
        // plumbing for yet; two words is enough to find out whether gPM will
        // take the register at all.  LOG.md 32.41.245.
        // 32.41.472: the sentinel test.
        //
        // fw:0x15f20 (inside FinalizeVertex) does
        //     ldr x9,[scene+8]; ldr w10,[x9]; str x10,[scene+0x10]; str wzr,[x9]
        // -- `pass_page_count` is a COPY of the word behind `control_word2`, not a
        // counter the firmware maintains. So `pass_pages 0` has two readings and we
        // cannot tell them apart: nothing incremented that word, or the firmware is
        // reading a word we never wrote because the pointer is not reachable the way
        // we set it up.
        //
        // Writing a sentinel here separates them in ONE run, with no plumbing:
        //   pass_page_count == sentinel  -> the firmware read OUR word. The pointer
        //                                   is fine and 0 previously meant nothing
        //                                   incremented it, i.e. nothing was binned.
        //   pass_page_count == 0         -> the firmware is reading a different word.
        //                                   `pass_pages` is a broken instrument and
        //                                   every reading of it in this phase is void.
        #[ver(G >= G14X)]
        let mut control_word = alloc.gpu.array_empty_tagged(1, b"CWRD")?;
        #[ver(G >= G14X)]
        {
            let sentinel = *module_parameters::pm_sentinel.value();
            if sentinel != 0 {
                control_word[0] = sentinel;
                pr_info!("asahi/pm_sentinel: control_word[0] <- {:#x}\n", sentinel);
            }
        }
        #[ver(G >= G14X)]
        let pm_counter = inner.ualloc_priv.lock().array_empty_tagged(1, b"PMCT")?;
        // 32.41.508: with the counter dual-mapped, `0x1c910` must carry the LOW
        // address of the shared page, not the address of a private allocation.
        #[ver(G >= G14X)]
        let control_word_va = if *module_parameters::pm_dualmap.value() != 0 {
            crate::gpu::IOVA_KERN_GPU_PMCT_LOW
        } else {
            u64::from(pm_counter.gpu_pointer())
        };
        #[ver(G < G14X)]
        let control_word_va: u64 = 0;

        // Captured before the buffers move into the Scene; see the
        // `unk_buffer_buf` note below for what these are for.
        let kbuf_vals = [
            // frag_tmap_poison: the fragment-side tilemap (kbuf[0] -> 0x16429) at an
            // unmapped VA, to prove the ISP reads it. LOG.md 32.41.619.
            match *module_parameters::frag_tmap_poison.value() {
                0 => u64::from(tvb_tilemap.gpu_pointer()),
                v => v,
            },
            // layermeta_at0: see Scene::tvb_layermeta_pointer. LOG.md 32.41.615.
            if *module_parameters::layermeta_at0.value() != 0 {
                u64::from(tvb_heapmeta.gpu_pointer())
            } else {
                u64::from(tvb_heapmeta.gpu_offset_pointer(0x200))
            },
            u64::from(user_buffer.gpu_pointer()),
        ];

        // macOS allocates this as private. However, the firmware does not
        // DC CIVAC this before reading it (like it does most other things),
        // which causes odd cache incoherency bugs when combined with
        // speculation on the firmware side (maybe). This doesn't happen
        // on macOS because these structs are a circular pool that is mapped
        // already initialized. Just mark this shared for now.
        let mut scene = alloc.shared.new_init(
            try_init!(buffer::Scene::ver {
                user_buffer: user_buffer,
                buffer: self.clone(),
                tvb_heapmeta: tvb_heapmeta,
                tvb_tilemap: tvb_tilemap,
                tpc: tpc,
                clustering: clustering,
                preempt_buf: preempt_buf,
                #[ver(G >= G14X)]
                control_word: control_word,
                #[ver(G >= G14X)]
                pm_counter: pm_counter,
                // 15.5's command start routines hand a "UMA pool" pointer to
                // 0x24cf0, which panics on null and builds a hardware aperture
                // descriptor from it. Aliasing this onto `Info` (32.41.14) fed
                // the GPU a garbage base and hung the machine, so give it its
                // own zeroed object until we know what it should describe.
                // LOG.md 32.41.16.
                uma: uma.clone(),
            }),
            |inner, _p| {
                try_init!(buffer::raw::Scene::ver {
                    // Opt-in: only point `+0x00` at the file-VM word when
                    // `0x1c910` is actually being programmed from it, so the
                    // default build stays byte-identical to the pre-32.41.245
                    // driver and the baseline stays comparable.
                    // 32.41.508: `pm_dualmap` makes these two name the SAME page
                    // in the two address spaces their consumers live in -- gPM
                    // through the low mapping, the firmware CPU through the high
                    // one. Without it they are two different allocations and the
                    // counter can never be read back.
                    #[ver(G >= G14X)]
                    control_word: if *module_parameters::pm_dualmap.value() != 0 {
                        GpuPointer::from_raw(crate::gpu::IOVA_KERN_GPU_PMCT_LOW)
                            .ok_or(EINVAL)?
                    } else if *module_parameters::vtx_1c910_src.value() != 0 {
                        inner.pm_counter.gpu_pointer()
                    } else {
                        inner.control_word.gpu_pointer()
                    },
                    #[ver(G >= G14X)]
                    control_word2: if *module_parameters::pm_dualmap.value() != 0 {
                        GpuPointer::from_raw(crate::gpu::IOVA_KERN_GPU_PMCT_HIGH)
                            .ok_or(EINVAL)?
                    } else {
                        inner.control_word.gpu_pointer()
                    },
                    pass_page_count: AtomicU32::new(0),
                    unk_4: 0,
                    unk_8: U64(0),
                    unk_10: U64(0),
                    user_buffer: inner.user_buffer.gpu_pointer(),
                    unk_20: 0,
                    #[ver(G >= G14X)]
                    __pad_34: 0,
                    #[ver(V >= V13_3)]
                    unk_28: U64(0),
                    stats: stats_pointer,
                    total_page_count: AtomicU32::new(0),
                    #[ver(G >= G14X)]
                    tvb_overflow: 0,
                    #[ver(G < G14X)]
                    unk_30: U64(0),
                    #[ver(G < G14X)]
                    unk_38: U64(0),
                })
            },
        )?;

        // 🔑 `unk_buffer_buf` is not opaque and it is not optional.  The
        // firmware's StartFragment handler dereferences it and appends three
        // registers from its first three quadwords -- fw:0x17de8..0x17e4c,
        // records at +0x90/+0x9c/+0xa8 with the 12-byte stride of
        // `RegisterArray`:
        //
        //     kbuf[0x00] -> 0x16429   (TVB tilemap)
        //     kbuf[0x08] -> 0x16060   (TVB layer metadata)
        //     kbuf[0x10] -> 0x1ca28   (the gPM parameter-manager pointer)
        //
        // `x8 + 0x38` there is exactly this driver's
        // `RunFragment::unk_buffer_buf` (`asahi/off RunFragment` prints
        // `scene 0x30 unk_buffer_buf 0x38`, and 0x30 is used two instructions
        // later as the scene pointer, which pins the base).
        //
        // Upstream allocates this zeroed and calls it "purpose unknown", so the
        // firmware has been appending **three zeros** to every fragment register
        // list -- and because it appends, those zeros land *after* the driver's
        // own correct 0x16429 and 0x16060 entries and win.  The fragment stage
        // has therefore been running with a null tilemap, null layer metadata
        // and a null parameter-manager pointer.
        //
        // The third value is `user_buffer`: macOS takes 0x1ca28 and the tiler's
        // 0x1ca30/0x16c39 from the same field of its parameter-buffer block
        // (+0x28), and 0x1ca30/0x16c39 are already known to be `user_buffer`
        // here (32.41.48).  LOG.md 32.41.225.
        {
            for (i, v) in kbuf_vals.iter().enumerate() {
                for (j, b) in v.to_le_bytes().iter().enumerate() {
                    inner.kernel_buffer[i * 8 + j] = *b;
                }
            }
            crate::sub_pr_info!(
                "asahi/kbuf: {:#x} tilemap {:#x} layermeta {:#x} user_buffer {:#x}\n",
                u64::from(inner.kernel_buffer.gpu_pointer()),
                kbuf_vals[0],
                kbuf_vals[1],
                kbuf_vals[2],
            );
        }

        let mut rebind = false;

        if inner.active_scenes == 0 && inner.info_pending {
            Self::publish_info(&mut inner);
        }

        if inner.active_slot.is_none() {
            assert_eq!(inner.active_scenes, 0);

            let slot = inner.mgr.0.get_inner(inner.last_token, |inner, mgr| {
                inner.owners[mgr.slot() as usize] = Some(self.clone());
                Ok(())
            })?;
            rebind = slot.changed();

            mod_pr_debug!("Buffer: assigning slot {} (rebind={})", slot.slot(), rebind);

            // 15.5 takes the PB descriptor slot from Info+0x08 (fw:0x28d6c), the TA from
            // RunVertex.buffer_slot: they must agree, or the TA reads another slot's (empty)
            // descriptor and reports out-of-memory on its first allocation. A new slot also
            // needs the descriptor rebuilt, which fw:0x1d124 does only while Info+0x18 != 0.
            // No scene is active here, so the firmware is not using this Info. LOG.md 32.41.665.
            #[ver(V >= V15_5)]
            {
                let s = slot.slot() as i32;
                inner.info.with_mut(|raw, _inner| {
                    if raw.last_id != s {
                        raw.last_id = s;
                        if *module_parameters::info_155.value() & 2 != 0 {
                            raw.unk_18 = 1;
                        }
                    }
                });
            }

            inner.last_token = Some(slot.token());
            inner.active_slot = Some(slot);
        }

        inner.active_scenes += 1;

        // 667zzx: one counter cell per buffer slot, as macOS points 0x1c910 at a
        // per-PB word. A single shared word hands one context's page count to
        // whichever job reads (and clears) it next.
        let slot = inner.active_slot.as_ref().unwrap().slot();
        #[ver(G >= G14X)]
        let control_word_va = if *module_parameters::pm_dualmap.value() != 0 {
            let off = u64::from(slot) * PMCT_CELL;
            scene.with_mut(|raw, _inner| -> Result {
                raw.control_word =
                    GpuPointer::from_raw(crate::gpu::IOVA_KERN_GPU_PMCT_LOW + off).ok_or(EINVAL)?;
                raw.control_word2 =
                    GpuPointer::from_raw(crate::gpu::IOVA_KERN_GPU_PMCT_HIGH + off).ok_or(EINVAL)?;
                Ok(())
            })?;
            control_word_va + off
        } else {
            control_word_va
        };

        Ok(Scene::ver {
            object: scene,
            slot,
            rebind,
            preempt2_off: inner.preempt1_size,
            preempt3_off: inner.preempt1_size + inner.preempt2_size,
            meta1_off: clmeta_size,
            meta2_off: clmeta_size + meta1_size,
            meta3_off: clmeta_size + meta1_size + meta2_size,
            meta4_off: clmeta_size + meta1_size + meta2_size + meta3_size,
            control_word_va,
            uma_busy_id: core::sync::atomic::AtomicU32::new(u32::MAX),
        })
    }

    /// Increment the buffer manager usage count. Should we done once we know the Scene is ready
    /// to be committed and used in commands submitted to the GPU.
    pub(crate) fn increment(&self) {
        let inner = self.inner.lock();
        inner.info.counter.with(|raw, _inner| {
            // We could use fetch_add, but the non-LSE atomic
            // sequence Rust produces confuses the hypervisor.
            // We have inner locked anyway, so this is not racy.
            let v = raw.count.load(Ordering::Relaxed);
            raw.count.store(v + 1, Ordering::Relaxed);
        });
    }

    pub(crate) fn any_ref(&self) -> Arc<dyn core::any::Any + Send + Sync> {
        self.inner.clone()
    }
}

#[versions(AGX)]
impl Clone for Buffer::ver {
    fn clone(&self) -> Self {
        Buffer::ver {
            inner: self.inner.clone(),
        }
    }
}

#[versions(AGX)]
struct BufferSlotInner();

#[versions(AGX)]
impl slotalloc::SlotItem for BufferSlotInner::ver {
    type Data = BufferManagerInner::ver;

    fn release(&mut self, data: &mut Self::Data, slot: u32) {
        mod_pr_debug!("BufferManager: Released slot {}\n", slot);
        data.owners[slot as usize] = None;
    }
}

/// Inner data for the buffer manager, to be protected by the SlotAllocator lock.
#[versions(AGX)]
pub(crate) struct BufferManagerInner {
    owners: KVec<Option<Buffer::ver>>,
    /// .667zzz39: Infos of dead Buffers, for reuse. The firmware's slot -> Info
    /// table (fw 0x1a1e20) is written at every vertex start (fw:0x1ce04) and never
    /// cleared, and its recovery routine (fw:0x373cc) walks all 127 entries
    /// comparing Info+0 with *(Info+0x60). A freed Info there crashed the
    /// firmware at fw:0x37424 in the first recovery after a killed client.
    /// Bounded by the peak number of live Buffers.
    info_pool: KVec<GpuObject<buffer::Info::ver>>,
}

/// The GPU-global buffer manager, used to allocate and release buffer slots from the pool.
#[versions(AGX)]
pub(crate) struct BufferManager(slotalloc::SlotAllocator<BufferSlotInner::ver>);

#[versions(AGX)]
impl BufferManager::ver {
    pub(crate) fn new() -> Result<BufferManager::ver> {
        let mut owners = KVec::new();
        for _i in 0..(NUM_BUFFERS as usize) {
            owners.push(None, GFP_KERNEL)?;
        }
        let alloc = slotalloc::SlotAllocator::new(
            NUM_BUFFERS,
            BufferManagerInner::ver { owners, info_pool: KVec::new() },
            |_inner, _slot| Some(BufferSlotInner::ver()),
            c_str!("BufferManager::SlotAllocator"),
            static_lock_class!(),
            static_lock_class!(),
        )?;
        // buf_slot_min: Apple's jobs use buffer slot 0xd; slot 0 may be a
        // firmware sentinel. LOG.md 32.41.630.
        alloc.set_min(*module_parameters::buf_slot_min.value() as usize);
        Ok(BufferManager::ver(alloc))
    }

    /// Signals a Buffer to synchronously grow.
    pub(crate) fn grow(&self, slot: u32) {
        match self
            .0
            .with_inner(|inner| inner.owners[slot as usize].as_ref().cloned())
        {
            Some(owner) => {
                pr_err!(
                    "BufferManager: Unexpected grow request for slot {}. This might deadlock. Please report this bug.\n",
                    slot
                );
                owner.sync_grow();
            }
            None => {
                pr_err!(
                    "BufferManager: Received grow request for empty slot {}\n",
                    slot
                );
            }
        }
    }
}

#[versions(AGX)]
impl Clone for BufferManager::ver {
    fn clone(&self) -> Self {
        BufferManager::ver(self.0.clone())
    }
}
