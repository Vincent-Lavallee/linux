// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Submission queue management
//!
//! This module implements the userspace view of submission queues and the logic to map userspace
//! submissions to firmware queues.

use kernel::dma_fence::*;
use kernel::prelude::*;
use kernel::{
    c_str,
    dma_fence,
    drm::sched,
    macros::versions,
    sync::{
        Arc,
        LockClassKey,
        Mutex, //
    },
    time::{
        delay::fsleep,
        Delta, //
    },
    uapi,
    xarray, //
};

use crate::alloc::Allocator;
use crate::debug::*;
use crate::driver::{AsahiDevRef, AsahiDevice};
use crate::file::MAX_COMMANDS_PER_SUBMISSION;
use crate::fw::types::*;
use crate::gpu::GpuManager;
use crate::inner_weak_ptr;
use crate::microseq;
use crate::module_parameters;
use crate::util::{
    AnyBitPattern,
    Reader, //
};
use crate::{
    alloc,
    buffer,
    channel,
    event,
    file,
    fw,
    gpu,
    mmu,
    workqueue, //
};

use core::sync::atomic::{
    AtomicU64,
    Ordering, //
};

const DEBUG_CLASS: DebugFlags = DebugFlags::Queue;

const WQ_SIZE: u32 = 0x500;

mod common;
mod compute;
mod render;

/// Trait implemented by all versioned queues.
pub(crate) trait Queue: Send + Sync {
    fn submit(
        &mut self,
        id: u64,
        syncs: KVec<file::SyncItem>,
        in_sync_count: usize,
        cmdbuf_raw: &[u8],
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
    ) -> Result;
}


/// drm_asahi_cmd_compute.flags bit 0 (this fork): the command stream has a dispatch with more than
/// 512 threads per workgroup. LOG.md 32.41.667zzz41.
pub(crate) const COMPUTE_LARGE_WG: u32 = 1;

/// .667zzz41 (`comp_serial` = K): the finished fences of the last K compute submissions (ring),
/// so submission N depends on submission N-K: at most K compute jobs from all processes in flight.
const COMP_SERIAL_MAX: usize = 16;
/// .667zzz44: each entry carries the submitting queue's id -- a queue's own submissions are already
/// ordered on the CDM, and waiting for them on the host put a ~2 ms bubble behind every large-WG
/// submission of House of Ashes (GPU executing 60% of the time).
struct CompSerialLast(
    core::cell::UnsafeCell<([Option<(kernel::dma_fence::Fence, u64)>; COMP_SERIAL_MAX], usize)>,
);
// SAFETY: only accessed while COMP_SERIAL_BUSY is held.
unsafe impl Sync for CompSerialLast {}
static COMP_SERIAL_LAST: CompSerialLast =
    CompSerialLast(core::cell::UnsafeCell::new(([const { None }; COMP_SERIAL_MAX], 0)));
static COMP_SERIAL_BUSY: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

struct CompSerialGuard;
impl CompSerialGuard {
    fn new() -> Self {
        use core::sync::atomic::Ordering;
        while COMP_SERIAL_BUSY
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            kernel::time::delay::fsleep(kernel::time::Delta::from_micros(20));
        }
        CompSerialGuard
    }
}
impl Drop for CompSerialGuard {
    fn drop(&mut self) {
        COMP_SERIAL_BUSY.store(false, core::sync::atomic::Ordering::Release);
    }
}

#[versions(AGX)]
struct SubQueue {
    wq: Arc<workqueue::WorkQueue::ver>,
}

#[versions(AGX)]
impl SubQueue::ver {
    fn new_job(&mut self, fence: dma_fence::Fence) -> SubQueueJob::ver {
        SubQueueJob::ver {
            wq: self.wq.clone(),
            fence: Some(fence),
            job: None,
        }
    }
}

#[versions(AGX)]
struct SubQueueJob {
    wq: Arc<workqueue::WorkQueue::ver>,
    job: Option<workqueue::Job::ver>,
    fence: Option<dma_fence::Fence>,
}

#[versions(AGX)]
impl SubQueueJob::ver {
    fn get(&mut self) -> Result<&mut workqueue::Job::ver> {
        if self.job.is_none() {
            mod_pr_debug!("SubQueueJob: Creating {:?} job\n", self.wq.pipe_type());
            self.job
                .replace(self.wq.new_job(self.fence.take().unwrap())?);
        }
        Ok(self.job.as_mut().expect("expected a Job"))
    }

    fn commit(&mut self) -> Result {
        match self.job.as_mut() {
            Some(job) => job.commit(),
            None => Ok(()),
        }
    }

    fn can_submit(&self) -> Option<Fence> {
        self.job.as_ref().and_then(|job| job.can_submit())
    }
}

#[versions(AGX)]
pub(crate) struct Queue {
    dev: AsahiDevRef,
    _sched: sched::Scheduler<QueueJob::ver>,
    entity: sched::Entity<QueueJob::ver>,
    vm: mmu::Vm,
    q_vtx: Option<SubQueue::ver>,
    q_frag: Option<SubQueue::ver>,
    q_comp: Option<SubQueue::ver>,
    fence_ctx: FenceContexts,
    inner: QueueInner::ver,
}

#[versions(AGX)]
pub(crate) struct QueueInner {
    dev: AsahiDevRef,
    ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
    buffer: buffer::Buffer::ver,
    gpu_context: Arc<workqueue::GpuContext>,
    notifier_list: Arc<GpuObject<fw::event::NotifierList>>,
    notifier: Arc<GpuObject<fw::event::Notifier::ver>>,
    usc_exec_base: u64,
    id: u64,
    #[ver(V >= V13_0B4)]
    counter: AtomicU64,
    /// comp_apple's compute UMA pool, built on the first compute job and kept for
    /// the queue's lifetime. LOG.md 32.41.664.
    cpool: Arc<Mutex<Option<Arc<fw::compute::ComputePool>>>>,
}

#[versions(AGX)]
#[derive(Default)]
pub(crate) struct JobFence {
    id: u64,
    pending: AtomicU64,
}

#[versions(AGX)]
impl JobFence::ver {
    fn add_command(self: &FenceObject<Self>) {
        self.pending.fetch_add(1, Ordering::Relaxed);
    }

    fn command_complete(self: &FenceObject<Self>) {
        let remain = self.pending.fetch_sub(1, Ordering::Relaxed) - 1;
        mod_pr_debug!(
            "JobFence[{}]: Command complete (remain: {})\n",
            self.id,
            remain
        );
        if remain == 0 {
            mod_pr_debug!("JobFence[{}]: Signaling\n", self.id);
            self.signal();
        }
    }
}

#[versions(AGX)]
#[vtable]
impl dma_fence::FenceOps for JobFence::ver {
    fn get_driver_name<'a>(self: &'a FenceObject<Self>) -> &'a CStr {
        c_str!("asahi")
    }
    fn get_timeline_name<'a>(self: &'a FenceObject<Self>) -> &'a CStr {
        c_str!("queue")
    }
}

/// Monotonic ns of the last host-initiated GPU restart (0 = none).
static LAST_RESTART: AtomicU64 = AtomicU64::new(0);
/// Lowest job id (= earliest submission) that timed out since that restart:
/// the job everything else was waiting behind, i.e. the guilty one.
static EPOCH_MIN_ID: AtomicU64 = AtomicU64::new(u64::MAX);

#[versions(AGX)]
pub(crate) struct QueueJob {
    dev: AsahiDevRef,
    vm_bind: mmu::VmBind,
    op_guard: Option<gpu::OpGuard>,
    sj_vtx: Option<SubQueueJob::ver>,
    sj_frag: Option<SubQueueJob::ver>,
    sj_comp: Option<SubQueueJob::ver>,
    fence: UserFence<JobFence::ver>,
    notifier: Arc<GpuObject<fw::event::Notifier::ver>>,
    notification_count: u32,
    did_run: bool,
    id: u64,
    /// The queue's TVB heap, grown when a render times out (host recovery).
    buffer: Option<buffer::Buffer::ver>,
    /// Host recoveries attempted for this job.
    recoveries: u32,
}

#[versions(AGX)]
impl QueueJob::ver {
    fn get_vtx(&mut self) -> Result<&mut workqueue::Job::ver> {
        self.sj_vtx
            .as_mut()
            .ok_or_else(|| {
                cls_pr_debug!(Errors, "No vertex queue\n");
                EINVAL
            })?
            .get()
    }
    fn get_frag(&mut self) -> Result<&mut workqueue::Job::ver> {
        self.sj_frag
            .as_mut()
            .ok_or_else(|| {
                cls_pr_debug!(Errors, "No fragment queue\n");
                EINVAL
            })?
            .get()
    }
    fn get_comp(&mut self) -> Result<&mut workqueue::Job::ver> {
        self.sj_comp
            .as_mut()
            .ok_or_else(|| {
                cls_pr_debug!(Errors, "No compute queue\n");
                EINVAL
            })?
            .get()
    }

    fn commit(&mut self) -> Result {
        mod_dev_dbg!(self.dev, "QueueJob {}: Committing\n", self.id);

        self.sj_vtx.as_mut().map(|a| a.commit()).unwrap_or(Ok(()))?;
        self.sj_frag
            .as_mut()
            .map(|a| a.commit())
            .unwrap_or(Ok(()))?;
        self.sj_comp.as_mut().map(|a| a.commit()).unwrap_or(Ok(()))
    }
}

#[versions(AGX)]
impl sched::JobImpl for QueueJob::ver {
    fn prepare(job: &mut sched::Job<Self>) -> Option<Fence> {
        mod_dev_dbg!(job.dev, "QueueJob {}: Checking runnability\n", job.id);

        if let Some(sj) = job.sj_vtx.as_ref() {
            if let Some(fence) = sj.can_submit() {
                mod_dev_dbg!(
                    job.dev,
                    "QueueJob {}: Blocking due to vertex queue full\n",
                    job.id
                );
                return Some(fence);
            }
        }
        if let Some(sj) = job.sj_frag.as_ref() {
            if let Some(fence) = sj.can_submit() {
                mod_dev_dbg!(
                    job.dev,
                    "QueueJob {}: Blocking due to fragment queue full\n",
                    job.id
                );
                return Some(fence);
            }
        }
        if let Some(sj) = job.sj_comp.as_ref() {
            if let Some(fence) = sj.can_submit() {
                mod_dev_dbg!(
                    job.dev,
                    "QueueJob {}: Blocking due to compute queue full\n",
                    job.id
                );
                return Some(fence);
            }
        }
        None
    }

    #[allow(unused_assignments)]
    fn run(job: &mut sched::Job<Self>) -> Result<Option<dma_fence::Fence>> {
        mod_dev_dbg!(job.dev, "QueueJob {}: Running Job\n", job.id);

        // We can only increase the notifier threshold here, now that we are
        // actually running the job. We cannot increase it while queueing the
        // job without introducing subtle race conditions. Suppose we did, as
        // early versions of drm/asahi did:
        //
        // 1. When processing the ioctl submit, a job is queued to drm_sched.
        //    Incorrectly, the notifier threshold is increased, gating firmware
        //    events.
        // 2. When DRM schedules an event, the hardware is kicked.
        // 3. When the number of processed jobs equals the threshold, the
        //    firmware signals the complete event to the kernel
        // 4. When the kernel gets a complete event, we signal the out-syncs.
        //
        // Does that work? There are a few scenarios.
        //
        // 1. There is nothing else ioctl submitted before the job completes.
        //    The job is scheduled, completes, and signals immediately.
        //    Everything works.
        // 2. There is nontrivial sync across different queues. Since each queue
        //    has a separate own notifier threshold, submitting one does not
        //    block scheduling of the other. Everything works the way you'd
        //    expect. drm/sched handles the wait/signal ordering.
        // 3. Two ioctls are submitted back-to-back. The first signals a fence
        //    that the second waits on. Due to the notifier threshold increment,
        //    the first job's completion event is deferred. But in good
        //    conditions, drm/sched will schedule the second submit anyway
        //    because it kills the pointless intra-queue sync. Then both
        //    commands execute and are signalled together.
        // 4. Two ioctls are submitted back-to-back as above, but conditions are
        //    bad. Reporting completion of the first job is still masked by the
        //    notifier threshold, but the intra-queue fences are not optimized
        //    out in drm/sched... drm/sched doesn't schedule the second job
        //    until the first is signalled, but the first isn't signalled until
        //    the second is completed, but the second can't complete until it's
        //    scheduled. We hang!
        //
        // In good conditions, everything works properly and/or we win the race
        // to mask the issue. So the issue here is challenging to hit.
        // Nevertheless, we do need to get it right.
        //
        // The intention with drm/sched is that jobs that are not yet scheduled
        // are "invisible" to the firmware. Incrementing the notifier threshold
        // earlier than this violates that which leads to circles like the
        // above. Deferring the increment to submit solves the race.
        job.notifier.threshold.with(|raw, _inner| {
            raw.increase(job.notification_count);
        });

        let gpu = match (*job.dev)
            .gpu
            .clone()
            .arc_as_any()
            .downcast::<gpu::GpuManager::ver>()
        {
            Ok(gpu) => gpu,
            Err(_) => {
                dev_crit!(job.dev.as_ref(), "GpuManager mismatched with QueueJob!\n");
                return Err(EIO);
            }
        };

        if job.op_guard.is_none() {
            job.op_guard = Some(gpu.start_op()?);
        }

        // First submit all the commands for each queue. This can fail.

        let mut frag_job = None;
        let mut frag_sub = None;
        if let Some(sj) = job.sj_frag.as_mut() {
            frag_job = sj.job.take();
            if let Some(wqjob) = frag_job.as_mut() {
                mod_dev_dbg!(job.dev, "QueueJob {}: Submit fragment\n", job.id);
                frag_sub = Some(wqjob.submit()?);
            }
        }

        let mut vtx_job = None;
        let mut vtx_sub = None;
        if let Some(sj) = job.sj_vtx.as_mut() {
            vtx_job = sj.job.take();
            if let Some(wqjob) = vtx_job.as_mut() {
                mod_dev_dbg!(job.dev, "QueueJob {}: Submit vertex\n", job.id);
                vtx_sub = Some(wqjob.submit()?);
            }
        }

        let mut comp_job = None;
        let mut comp_sub = None;
        if let Some(sj) = job.sj_comp.as_mut() {
            comp_job = sj.job.take();
            if let Some(wqjob) = comp_job.as_mut() {
                mod_dev_dbg!(job.dev, "QueueJob {}: Submit compute\n", job.id);
                comp_sub = Some(wqjob.submit()?);
            }
        }

        // Now we fully commit to running the job.
        //
        // 🔑 The kick order matters on 15.5.  The firmware evaluates the
        // fragment queue's `Barrier` the moment the fragment pipe is kicked;
        // with the fragment kicked first the vertex has not run, the wait is
        // unsatisfied, and the barrier parks *holding the event slot's
        // in-flight refcount* (`fw:0xe4980 + slot*0x38 + 0x30`).  Every retry
        // then hits `fw:0x13a1c`, which refuses any command on a slot whose
        // refcount is non-zero, and logs trace id 0x54 with code 0x8b3 -- the
        // barrier deadlocking against itself.  Observed directly:
        //
        //   0x0054 | 0 0     <frag wq> 0   OK    index 0 (Barrier)
        //   0x0054 | 1 0x8b3 <frag wq> 0   FAIL  index 0, retried
        //   0x0054 | 1 0x8b3 <frag wq> 0   FAIL  index 0, again
        //   0x000a | 0x100 ...                   vertex completes -- too late
        //
        // `vtx_kick_first` reverses only the kicks, not the ring commits, so
        // the barrier still precedes RunFragment in the queue.  `kick_delay_us`
        // additionally gives the vertex time to retire first, which separates
        // "the order is wrong" from "the barrier can never be re-evaluated".
        // LOG.md 32.41.113.
        if *crate::module_parameters::vtx_kick_first.value() != 0 {
            mod_dev_dbg!(job.dev, "QueueJob {}: Run vertex (first)\n", job.id);
            vtx_sub.map(|a| gpu.run_job(a)).transpose()?;

            mod_dev_dbg!(job.dev, "QueueJob {}: Run compute\n", job.id);
            comp_sub.map(|a| gpu.run_job(a)).transpose()?;

            let delay = *crate::module_parameters::kick_delay_us.value();
            if delay != 0 {
                fsleep(Delta::from_micros(delay as i64));
            }

            mod_dev_dbg!(job.dev, "QueueJob {}: Run fragment (last)\n", job.id);
            frag_sub.map(|a| gpu.run_job(a)).transpose()?;
        } else {
            mod_dev_dbg!(job.dev, "QueueJob {}: Run fragment\n", job.id);
            frag_sub.map(|a| gpu.run_job(a)).transpose()?;

            mod_dev_dbg!(job.dev, "QueueJob {}: Run vertex\n", job.id);
            vtx_sub.map(|a| gpu.run_job(a)).transpose()?;

            mod_dev_dbg!(job.dev, "QueueJob {}: Run compute\n", job.id);
            comp_sub.map(|a| gpu.run_job(a)).transpose()?;
        }

        // 🔬 Probe the scene a fixed delay after the kick, still inside
        // `run()`. The timeout probe stopped firing once the control-stream
        // runaway was fixed (no fault -> no `handle_timeout`), and the drop
        // probe never runs either: on a clean non-completion the CPU wedges in
        // `Queue::drop` *before* `QueueJob::drop` (32.41.116). This is the one
        // place guaranteed to execute while the job is in flight. A non-zero
        // TVB tilemap here is direct evidence the vertex stage ran, which is
        // what separates "the stream parses" from "the draw executed".
        // LOG.md 32.41.176.
        // 🔑 Non-perturbing in-flight state poll, started as early as this path
        // allows -- before the `probe_delay_ms` sleep, so it covers the window
        // right after the kick rather than one fixed instant well after it.
        //
        // This uses a BARE read of `0xc020`: it never writes the `0x10398`
        // selector. That is the whole point. `state_in_flight` above goes
        // through `dump_state_pre`, which writes the selector even though it
        // restores it, and that measurably perturbs a run (32.41.336) -- so
        // every reading it produced is suspect. Writing nothing cannot.
        //
        // What it is for: 32.41.327 concluded "the TA stage never starts" from
        // `0xc020 = 0` sampled ~40 s after the kick, which is equally
        // consistent with "started, did nothing, went idle" (32.41.346). A
        // dense sample starting at the kick separates them. LOG.md 32.41.361.
        let poll_n = *crate::module_parameters::state_poll_n.value();
        if poll_n != 0 {
            gpu.poll_state_now("inflight", poll_n, *crate::module_parameters::state_poll_ms.value());
        }

        let probe_ms = *crate::module_parameters::probe_delay_ms.value();
        if probe_ms != 0 {
            fsleep(Delta::from_millis(probe_ms as i64));
            gpu.probe_scene("POSTKICK");
            // 🔑 Sample the GPU state register *while the job is in flight*.
            //
            // 32.41.327 read `0xc020 = 0` at the hang and concluded the TA stage
            // never starts -- but that sample is ~40 s after the kick, which is
            // equally consistent with "ran, finished, went idle". Only a sample
            // taken during the job separates the two.
            //
            // Uses `dump_state_pre`, which saves and restores the `0x10398`
            // selector and touches two indices. That is NOT `dump_uma`'s 0..16
            // sweep, which leaves the selector at 15 and measurably
            // reintroduces the runaway (32.41.326). LOG.md 32.41.335.
            if *crate::module_parameters::state_in_flight.value() != 0 {
                gpu.dump_state_now("inflight");
            }
            // The firmware's vertex stats, in flight. This one is a kernel
            // page-table read rather than MMIO, so it does not perturb the run
            // the way the state sample above does (32.41.336) -- and it is the
            // only way to see the stats on a CLEAN run, where no fault and no
            // timeout means `handle_timeout` never fires. LOG.md 32.41.337.
            gpu.dump_stats_now("inflight");
        }

        mod_dev_dbg!(job.dev, "QueueJob {}: Drop compute job\n", job.id);
        core::mem::drop(comp_job);
        mod_dev_dbg!(job.dev, "QueueJob {}: Drop vertex job\n", job.id);
        core::mem::drop(vtx_job);
        mod_dev_dbg!(job.dev, "QueueJob {}: Drop fragment job\n", job.id);
        core::mem::drop(frag_job);

        job.did_run = true;

        Ok(Some(Fence::from_fence(&job.fence)))
    }

    fn timed_out(job: &mut sched::Job<Self>) -> sched::Status {
        // FIXME: Handle timeouts properly
        dev_err!(
            job.dev.as_ref(),
            "QueueJob {}: Job timed out on the DRM scheduler, things will probably break (ran: {})\n",
            job.id, job.did_run
        );

        // This is the only place a stuck job is reachable. `Queue::drop` never
        // runs while the process is still blocked on the fence, so the state
        // that says *which* pipe is stuck and how far the firmware got has been
        // going unread. LOG.md 32.41.98.
        for sj in [job.sj_vtx.as_ref(), job.sj_frag.as_ref(), job.sj_comp.as_ref()]
            .into_iter()
            .flatten()
        {
            sj.wq.dump_state(&job.dev);
            sj.wq.dump_info();
            workqueue::WorkQueue::describe(&*sj.wq);
        }
        job.notifier.with(|raw, _inner| {
            let vm_slot = raw.state.vm_slot;
            let in_list = raw.state.in_list;
            let has_vtx = raw.state.has_vtx;
            let has_frag = raw.state.has_frag;
            let pstamp_vtx = raw.state.pstamp_vtx[0].0;
            let pstamp_frag = raw.state.pstamp_frag[0].0;
            dev_err!(
                job.dev.as_ref(),
                "  Notifier: generation {:#x} cur_count {:#x} unk_10 {:#x} vm_slot {} in_list {} has_vtx {} pstamp_vtx {:#x} has_frag {} pstamp_frag {:#x}\n",
                raw.generation.load(Ordering::Relaxed),
                raw.cur_count.load(Ordering::Relaxed),
                raw.unk_10.load(Ordering::Relaxed),
                vm_slot,
                in_list,
                has_vtx,
                pstamp_vtx,
                has_frag,
                pstamp_frag,
            );
        });
        (*job.dev).gpu.dump_hung_state();
        crate::channel::ktrace_hist_dump();

        // Host recovery = Apple's channel restart: stop the firmware, resume
        // it, and it re-dispatches every in-flight command. The known hang
        // is a first frame bigger than the TVB heap (TA OOM, partial render
        // never resumes the TA), so grow the heap first and the replay
        // fits. Nothing is failed host-side (see host_recover). After three
        // attempts on one job, give up. LOG.md 32.41.667zzz29.
        if job.recoveries < 3 {
            job.recoveries += 1;
            if job.sj_vtx.is_some() {
                if let Some(buf) = job.buffer.as_ref() {
                    let cur = buf.block_count() as usize;
                    match buf.ensure_blocks(cur * 4) {
                        Ok(_) => {
                            buf.force_publish();
                            dev_err!(job.dev.as_ref(), "QueueJob {}: TVB heap {} -> {} blocks (published)\n",
                                     job.id, cur, buf.block_count())
                        }
                        Err(e) => dev_err!(job.dev.as_ref(), "QueueJob {}: TVB grow failed: {:?}\n", job.id, e),
                    }
                }
            }
            // A job that was only queued behind the hung one times out right
            // after it: the restart that just ran covers it too.
            let now = crate::stats::now_ns();
            let last = LAST_RESTART.load(Ordering::Acquire);
            let recent = last != 0 && now.saturating_sub(last) < 15_000_000_000;
            if recent {
                EPOCH_MIN_ID.fetch_min(job.id, Ordering::AcqRel);
            }
            if recent || (*job.dev).gpu.host_recover() {
                if !recent {
                    LAST_RESTART.store(now, Ordering::Release);
                    EPOCH_MIN_ID.store(job.id, Ordering::Release);
                }
                dev_err!(job.dev.as_ref(), "QueueJob {}: GPU restarted{} (attempt {}), job replayed\n",
                         job.id, if recent { " (shared)" } else { "" }, job.recoveries);
                // drm_sched does not re-arm a timeout for this job. The replay
                // puts a still-hung command in a busy slot, where the
                // firmware's own watchdog (Globals+0x9bc = 50 ticks, ~6.6 s)
                // raises its Timeout event and the firmware-initiated
                // recovery drops it cleanly (handle_timeout). Only if that
                // never happens, abandon the work (fences complete with
                // -ETIMEDOUT, command objects retained, queue banned).
                let sjs = || {
                    [job.sj_vtx.as_ref(), job.sj_frag.as_ref(), job.sj_comp.as_ref()]
                        .into_iter()
                        .flatten()
                };
                for _ in 0..15000 {
                    if !sjs().any(|sj| sj.wq.has_pending()) {
                        dev_err!(job.dev.as_ref(), "QueueJob {}: replay completed\n", job.id);
                        return sched::Status::Nominal;
                    }
                    kernel::time::delay::fsleep(kernel::time::Delta::from_millis(1));
                }
                // The earliest-submitted job of this restart is the guilty one.
                // A job that only waited behind it loses its channel too: its
                // in-flight command never completes after the restart and
                // every later one would time out again (.667zzz29 test 15),
                // so ban both; robust clients recreate their context.
                let guilty = job.id <= EPOCH_MIN_ID.load(Ordering::Acquire);
                let n: usize = sjs()
                    .map(|sj| sj.wq.abandon(workqueue::WorkError::Timeout, true))
                    .sum();
                // The firmware still holds the abandoned commands in its slots
                // (its PM code dereferences them, fw:0x373a0): leak the work
                // queues (which own them) and the TVB heap so process exit
                // cannot free anything it may touch. Only after a hang.
                for sj in sjs() {
                    core::mem::forget(sj.wq.clone());
                }
                if let Some(b) = job.buffer.as_ref() {
                    core::mem::forget(b.clone());
                }
                // ... and the VM: its slot stays bound and nothing is unmapped.
                job.vm_bind.vm().poison();
                // .667zzz40: leaked Scenes never release their UMA pool ids
                crate::buffer::uma_busy_reset();
                core::mem::forget(job.vm_bind.clone());
                dev_err!(job.dev.as_ref(), "QueueJob {}: replay did not complete; abandoned {} commands{}\n",
                         job.id, n, if guilty { ", queue lost (guilty)" } else { ", queue lost (innocent)" });
                return sched::Status::Nominal;
            }
        }

        sched::Status::NoDevice
    }

    fn cancel(job: &mut sched::Job<Self>) {
        dev_info!(
            job.dev.as_ref(),
            "QueueJob {}: Job canceled on DRM scheduler teardown\n",
            job.id
        );
        crate::channel::ktrace_hist_dump();
    }
}

#[versions(AGX)]
impl Drop for QueueJob::ver {
    fn drop(&mut self) {
        mod_dev_dbg!(self.dev, "QueueJob {}: Dropping\n", self.id);
        // A job that completes cleanly, or one whose control stream now parses
        // without running away, never reaches `handle_timeout` -- so the scene
        // probe there stopped firing precisely when it became the thing worth
        // reading. Probe again on the way out. LOG.md 32.41.174.
        if *crate::module_parameters::probe_usc.value() != 0 {
            if let Ok(gpu) = (*self.dev)
                .gpu
                .clone()
                .arc_as_any()
                .downcast::<gpu::GpuManager::ver>()
            {
                gpu.probe_scene("DROP");
            }
        }
    }
}

static QUEUE_NAME: &CStr = c_str!("asahi_fence");
static QUEUE_CLASS_KEY: Pin<&LockClassKey> = kernel::static_lock_class!();

#[versions(AGX)]
impl Queue::ver {
    /// Create a new user queue.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dev: &AsahiDevice,
        vm: mmu::Vm,
        alloc: &mut gpu::KernelAllocators,
        ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
        ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
        event_manager: Arc<event::EventManager>,
        mgr: &buffer::BufferManager::ver,
        id: u64,
        priority: u32,
        usc_exec_base: u64,
        uma: buffer::UmaSlot,
    ) -> Result<Queue::ver> {
        mod_dev_dbg!(dev, "[Queue {}] Creating queue\n", id);

        // Must be shared, no cache management on this one!
        let mut notifier_list = alloc.shared.new_default::<fw::event::NotifierList>()?;

        let self_ptr = notifier_list.weak_pointer();
        notifier_list.with_mut(|raw, _inner| {
            raw.list_head.next = Some(inner_weak_ptr!(self_ptr, list_head));
        });

        let threshold = alloc.shared.new_default::<fw::event::Threshold>()?;

        // The notifier's `has_vtx`/`pstamp_*`/`in_list` bookkeeping is the only
        // window on why a queue parked on a barrier is not being re-examined,
        // and out of `alloc.private` (cached) every read of it from the CPU is
        // stale -- all-zero for a job the firmware demonstrably has. Follow
        // `WorkQueue`'s lead and move it to shared memory under
        // `VerboseFaults` so the dump means something. LOG.md 32.41.107.
        let notifier: Arc<GpuObject<fw::event::Notifier::ver>> = Arc::new(
            if debug_enabled(DebugFlags::VerboseFaults)
                || *crate::module_parameters::q_quarantine.value() & 4 != 0
            {
                &mut alloc.shared
            } else {
                &mut alloc.private
            }
            .new_init(
                /*try_*/ init!(fw::event::Notifier::ver { threshold }),
                |inner, _p| {
                    try_init!(fw::event::raw::Notifier::ver {
                        threshold: inner.threshold.gpu_pointer(),
                        generation: AtomicU32::new(id as u32),
                        cur_count: AtomicU32::new(0),
                        unk_10: AtomicU32::new(0x50),
                        state: Default::default()
                    })
                },
            )?,
            GFP_KERNEL,
        )?;

        // Priorities are handled by the AGX scheduler, there is no meaning within a
        // per-queue scheduler. Use a single run queue wth Kernel priority.
        let sched =
            sched::Scheduler::new(
                dev.as_ref(),
                1,
                WQ_SIZE,
                0,
                (*module_parameters::job_timeout_ms.value()).max(1000),
                c_str!("asahi_sched"),
            )?;
        let entity = sched::Entity::new(&sched, sched::Priority::Kernel)?;

        let buffer =
            buffer::Buffer::ver::new(&*(*dev).gpu, alloc, ualloc.clone(), ualloc_priv, mgr, uma)?;

        let mut ret = Queue::ver {
            dev: dev.into(),
            _sched: sched,
            entity,
            vm,
            q_vtx: None,
            q_frag: None,
            q_comp: None,
            fence_ctx: FenceContexts::new(1, QUEUE_NAME, QUEUE_CLASS_KEY)?,
            inner: QueueInner::ver {
                dev: dev.into(),
                ualloc,
                gpu_context: Arc::new(
                    workqueue::GpuContext::new(dev, alloc, buffer.any_ref())?,
                    GFP_KERNEL,
                )?,

                buffer,
                notifier_list: Arc::new(notifier_list, GFP_KERNEL)?,
                notifier,
                usc_exec_base,
                id,
                #[ver(V >= V13_0B4)]
                counter: AtomicU64::new(0),
                cpool: Arc::pin_init(kernel::new_mutex!(None), GFP_KERNEL)?,
            },
        };

        // Rendering structures
        let tvb_blocks = *module_parameters::initial_tvb_size.value();

        ret.inner.buffer.ensure_blocks(tvb_blocks)?;

        ret.q_vtx = Some(SubQueue::ver {
            wq: workqueue::WorkQueue::ver::new(
                dev,
                alloc,
                event_manager.clone(),
                ret.inner.gpu_context.clone(),
                ret.inner.notifier_list.clone(),
                channel::PipeType::Vertex,
                id,
                priority,
                WQ_SIZE,
            )?,
        });

        ret.q_frag = Some(SubQueue::ver {
            wq: workqueue::WorkQueue::ver::new(
                dev,
                alloc,
                event_manager.clone(),
                ret.inner.gpu_context.clone(),
                ret.inner.notifier_list.clone(),
                channel::PipeType::Fragment,
                id,
                priority,
                WQ_SIZE,
            )?,
        });

        // Compute structures
        ret.q_comp = Some(SubQueue::ver {
            wq: workqueue::WorkQueue::ver::new(
                dev,
                alloc,
                event_manager,
                ret.inner.gpu_context.clone(),
                ret.inner.notifier_list.clone(),
                channel::PipeType::Compute,
                id,
                priority,
                WQ_SIZE,
            )?,
        });

        mod_dev_dbg!(dev, "[Queue {}] Queue created\n", id);
        Ok(ret)
    }
}

const SQ_RENDER: usize = 0;
const SQ_COMPUTE: usize = 1;
const SQ_COUNT: usize = 2;

// SAFETY: All bit patterns are valid by construction.
unsafe impl AnyBitPattern for uapi::drm_asahi_cmd_header {}
unsafe impl AnyBitPattern for uapi::drm_asahi_cmd_render {}
unsafe impl AnyBitPattern for uapi::drm_asahi_cmd_compute {}
unsafe impl AnyBitPattern for uapi::drm_asahi_attachment {}

fn build_attachments(reader: &mut Reader<'_>, size: usize) -> Result<microseq::Attachments> {
    const STRIDE: usize = core::mem::size_of::<uapi::drm_asahi_attachment>();
    let count = size / STRIDE;

    if count > microseq::MAX_ATTACHMENTS {
        return Err(EINVAL);
    }

    let mut attachments: microseq::Attachments = Default::default();
    attachments.count = count as u32;

    for i in 0..count {
        let att: uapi::drm_asahi_attachment = reader.read()?;

        if att.flags != 0 || att.pad != 0 {
            return Err(EINVAL);
        }

        // Some kind of power-of-2 exponent related to attachment size, in
        // bounds [1, 6]? We don't know what this is exactly yet.
        let unk_e = 1;

        let cache_lines = (att.size + 127) >> 7;
        attachments.list[i as usize] = microseq::Attachment {
            address: U64(att.pointer),
            size: cache_lines.try_into()?,
            unk_c: 0x17,
            unk_e: unk_e as u16,
        };
    }

    Ok(attachments)
}

#[versions(AGX)]
impl Queue for Queue::ver {
    fn submit(
        &mut self,
        id: u64,
        mut syncs: KVec<file::SyncItem>,
        in_sync_count: usize,
        cmdbuf_raw: &[u8],
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
    ) -> Result {
        let gpu = match (*self.dev)
            .gpu
            .clone()
            .arc_as_any()
            .downcast::<gpu::GpuManager::ver>()
        {
            Ok(gpu) => gpu,
            Err(_) => {
                dev_crit!(self.dev.as_ref(), "GpuManager mismatched with JobImpl!\n");
                return Err(EIO);
            }
        };

        mod_dev_dbg!(self.dev, "[Submission {}] Submit job\n", id);

        if gpu.is_crashed() {
            dev_err!(
                self.dev.as_ref(),
                "[Submission {}] GPU is crashed, cannot submit\n",
                id
            );
            return Err(ENODEV);
        }

        // A queue whose work was abandoned by host recovery is lost (as a
        // guilty context is banned elsewhere): Mesa reports a lost context.
        if [self.q_vtx.as_ref(), self.q_frag.as_ref(), self.q_comp.as_ref()]
            .into_iter()
            .flatten()
            .any(|q| q.wq.is_dead())
        {
            dev_err!(self.dev.as_ref(), "[Submission {}] queue lost to a GPU hang, refusing\n", id);
            return Err(ECANCELED);
        }

        let op_guard = if in_sync_count > 0 {
            Some(gpu.start_op()?)
        } else {
            None
        };

        let mut events: [KVec<Option<workqueue::QueueEventInfo::ver>>; SQ_COUNT] =
            Default::default();

        events[SQ_RENDER].push(
            self.q_frag.as_ref().and_then(|a| a.wq.event_info()),
            GFP_KERNEL,
        )?;
        events[SQ_COMPUTE].push(
            self.q_comp.as_ref().and_then(|a| a.wq.event_info()),
            GFP_KERNEL,
        )?;

        let vm_bind = gpu.bind_vm(&self.vm)?;
        let vm_slot = vm_bind.slot();

        mod_dev_dbg!(self.dev, "[Submission {}] Creating job\n", id);

        // FIXME: I think this can violate the fence seqno ordering contract.
        // If we have e.g. a render submission with no barriers and then a compute submission
        // with no barriers, it's possible for the compute submission to complete first, and
        // therefore its fence. Maybe we should have separate fence contexts for render
        // and compute, and then do a ? (Vert+frag should be fine since there is no vert
        // without frag, and frag always serializes.)
        let fence: UserFence<JobFence::ver> = self
            .fence_ctx
            .new_fence::<JobFence::ver>(
                0,
                JobFence::ver {
                    id,
                    pending: Default::default(),
                },
            )?
            .into();

        let mut cmdbuf = Reader::new(cmdbuf_raw);

        // First, parse the headers to determine the number of compute/render
        // commands. This will be used to determine when to flush stamps.
        //
        // We also use it to determine how many notifications the job will
        // generate. We could calculate that in the second pass since we don't
        // need until much later, but it's convenient to gather everything at
        // the same time.
        let mut nr_commands = 0;
        let mut last_compute = 0;
        let mut last_render = 0;
        let mut nr_render = 0;
        let mut nr_compute = 0;

        let mut large_wg = false;
        while !cmdbuf.is_empty() {
            let header: uapi::drm_asahi_cmd_header = cmdbuf.read()?;
            if header.cmd_type as u32 == uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_COMPUTE {
                let c: uapi::drm_asahi_cmd_compute = cmdbuf.peek_up_to(header.size as usize)?;
                large_wg |= c.flags & COMPUTE_LARGE_WG != 0;
            }
            cmdbuf.skip(header.size as usize);
            nr_commands += 1;

            match header.cmd_type as u32 {
                uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_RENDER => {
                    last_compute = nr_commands;
                    nr_render += 1;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_COMPUTE => {
                    last_render = nr_commands;
                    nr_compute += 1;
                }
                _ => {}
            }
        }

        let mut job = self.entity.new_job(
            1,
            QueueJob::ver {
                dev: self.dev.clone(),
                vm_bind,
                op_guard,
                sj_vtx: self
                    .q_vtx
                    .as_mut()
                    .map(|a| a.new_job(Fence::from_fence(&fence))),
                sj_frag: self
                    .q_frag
                    .as_mut()
                    .map(|a| a.new_job(Fence::from_fence(&fence))),
                sj_comp: self
                    .q_comp
                    .as_mut()
                    .map(|a| a.new_job(Fence::from_fence(&fence))),
                fence,
                notifier: self.inner.notifier.clone(),
                buffer: Some(self.inner.buffer.clone()),
                recoveries: 0,

                // Each render command generates 2 notifications: 1 for the
                // vertex part, 1 for the fragment part. Each compute command
                // generates 1 notification. Sum up to calculate the total
                // notification count for the job.
                notification_count: (2 * nr_render) + nr_compute,

                did_run: false,
                id,
            },
        )?;

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Adding {} in_syncs\n",
            id,
            in_sync_count
        );
        for sync in syncs.drain(0..in_sync_count) {
            if let Some(fence) = sync.fence {
                job.add_dependency(fence)?;
            }
        }
        // .667zzz41: chain compute submissions across processes -- all of them (comp_serial = K,
        // diagnostic) or only those with workgroups > 512 threads (large_wg_serial, default):
        // two contexts each running workgroups larger than half a core deadlock the CDM.
        let comp_serial = nr_compute > 0
            && (*crate::module_parameters::comp_serial.value() != 0
                || (large_wg && *crate::module_parameters::large_wg_serial.value() != 0));
        let _comp_serial_guard = if comp_serial {
            let g = CompSerialGuard::new();
            // SAFETY: COMP_SERIAL_LAST is only touched while a CompSerialGuard is held.
            let (ring, n) = unsafe { &*COMP_SERIAL_LAST.0.get() };
            let k = (*crate::module_parameters::comp_serial.value() as usize).clamp(1, COMP_SERIAL_MAX);
            if let Some((prev, owner)) = ring[*n % k].clone() {
                if owner != self.inner.id || *crate::module_parameters::comp_serial.value() != 0 {
                    job.add_dependency(prev)?;
                }
            }
            Some(g)
        } else {
            None
        };

        // Validate the number of hardware commands, ignoring software commands
        let nr_hw_commands = nr_render + nr_compute;
        if nr_hw_commands == 0 || nr_hw_commands > MAX_COMMANDS_PER_SUBMISSION {
            cls_pr_debug!(
                Errors,
                "submit: Command count {} out of valid range [1, {}]\n",
                nr_hw_commands,
                MAX_COMMANDS_PER_SUBMISSION - 1
            );
            return Err(EINVAL);
        }

        cmdbuf.rewind();

        let mut command_index = 0;
        let mut vertex_attachments: microseq::Attachments = Default::default();
        let mut fragment_attachments: microseq::Attachments = Default::default();
        let mut compute_attachments: microseq::Attachments = Default::default();

        // Parse the full command buffer submitting as we go
        while !cmdbuf.is_empty() {
            let header: uapi::drm_asahi_cmd_header = cmdbuf.read()?;
            let header_size = header.size as usize;

            // Pre-increment command index to match last_compute/last_render
            command_index += 1;

            for (queue_idx, index) in [header.vdm_barrier, header.cdm_barrier].iter().enumerate() {
                if *index == uapi::DRM_ASAHI_BARRIER_NONE as u16 {
                    continue;
                }
                if let Some(event) = events[queue_idx].get(*index as usize).ok_or_else(|| {
                    cls_pr_debug!(Errors, "Invalid barrier #{}: {}\n", queue_idx, index);
                    EINVAL
                })? {
                    let mut alloc = gpu.alloc();
                    let queue_job = match header.cmd_type as u32 {
                        uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_RENDER => job.get_vtx()?,
                        uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_COMPUTE => job.get_comp()?,
                        _ => return Err(EINVAL),
                    };
                    mod_dev_dbg!(self.dev, "[Submission {}] Create Explicit Barrier\n", id);
                    let barrier = alloc.private.new_init(
                        pin_init::zeroed::<fw::workqueue::Barrier::ver>(),
                        |_inner, _p| {
                            let queue_job = &queue_job;
                            try_init!(fw::workqueue::raw::Barrier::ver {
                                tag: fw::workqueue::CommandType::Barrier,
                                wait_stamp: event.fw_stamp_pointer,
                                #[ver(V < V15_5)]
                                wait_value: event.value,
                                #[ver(V < V15_5)]
                                wait_slot: event.slot,
                                #[ver(V < V15_5)]
                                stamp_self: queue_job.event_info().value.next(),
                                #[ver(V < V15_5)]
                                uuid: 0xffffbbbb,
                                #[ver(V < V15_5)]
                                external_barrier: 0,
                                #[ver(V < V15_5)]
                                internal_barrier_type: 1,
                                #[ver(V < V15_5)]
                                padding: Default::default(),
                                #[ver(V >= V15_5)]
                                wait_stamp2: U64(0),
                                #[ver(V >= V15_5)]
                                wait_value: event.value,
                                #[ver(V >= V15_5)]
                                wait_value64: U64(0),
                                // 32.41.664: +0x20 low byte = the AWAITED stamp slot
                                // (fw:0x193b8: 0x80 = none, else root 0xe4980 + slot *
                                // 0x38; the dependency is linked to that slot's DAG node
                                // for wait_value's entry). 0 linked glmark2's wait on the
                                // fragment stamp (slot 1) to the vertex slot's finished
                                // node -> frame 3 never started.
                                #[ver(V >= V15_5)]
                                barrier_type: if *crate::module_parameters::barrier_dag.value() != 0 {
                                    event.slot
                                } else {
                                    0
                                },
                                // 32.41.664: +0x24 is the DAG word the barrier ASSIGNS
                                // to this queue's stamp-slot counter (fw:0x19308 ->
                                // 0x19bb0), so it must be the next command's stamp, as
                                // the render path's internal barrier already does. The
                                // awaited slot (<< 8) reset the vertex counter to 1 and
                                // glmark2's 2nd RunVertex (entry 2) hit "DAG: Non
                                // Sequential Stamp Updates".
                                #[ver(V >= V15_5)]
                                flags_slot: if *crate::module_parameters::barrier_dag.value() != 0 {
                                    queue_job.event_info().value.next().counter()
                                } else {
                                    event.slot << 8
                                },
                                #[ver(V >= V15_5)]
                                stamp_self: queue_job.event_info().value.next(),
                                #[ver(V >= V15_5)]
                                uuid: 0xffffbbbb,
                                #[ver(V >= V15_5)]
                                armed: if *crate::module_parameters::xbarrier_armed.value() != 0 {
                                    1
                                } else {
                                    *crate::module_parameters::barrier_armed.value()
                                },
                                #[ver(V >= V15_5)]
                                unk_34: 0,
                                #[ver(V >= V15_5)]
                                padding: Default::default(),
                            })
                        },
                    )?;
                    crate::sub_pr_info!(
                        "asahi/xbarrier: cmd {} type {} q{} index {} awaits slot {} value {:#x} -> own next {:#x}\n",
                        command_index,
                        header.cmd_type,
                        queue_idx,
                        index,
                        event.slot,
                        event.value.counter(),
                        queue_job.event_info().value.next().counter()
                    );
                    mod_dev_dbg!(self.dev, "[Submission {}] Add Explicit Barrier\n", id);
                    queue_job.add(barrier, vm_slot)?;
                } else {
                    assert!(*index == 0);
                }
            }

            match header.cmd_type as u32 {
                uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_RENDER => {
                    let render: uapi::drm_asahi_cmd_render = cmdbuf.read_up_to(header_size)?;

                    self.inner.submit_render(
                        &mut job,
                        &render,
                        &vertex_attachments,
                        &fragment_attachments,
                        objects,
                        id,
                        command_index == last_render,
                    )?;
                    events[SQ_RENDER].push(
                        Some(
                            job.sj_frag
                                .as_ref()
                                .expect("No frag queue?")
                                .job
                                .as_ref()
                                .expect("No frag job?")
                                .event_info(),
                        ),
                        GFP_KERNEL,
                    )?;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_CMD_COMPUTE => {
                    let compute: uapi::drm_asahi_cmd_compute = cmdbuf.read_up_to(header_size)?;

                    self.inner.submit_compute(
                        &mut job,
                        &compute,
                        &compute_attachments,
                        objects,
                        id,
                        command_index == last_compute,
                    )?;
                    events[SQ_COMPUTE].push(
                        Some(
                            job.sj_comp
                                .as_ref()
                                .expect("No comp queue?")
                                .job
                                .as_ref()
                                .expect("No comp job?")
                                .event_info(),
                        ),
                        GFP_KERNEL,
                    )?;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_SET_VERTEX_ATTACHMENTS => {
                    vertex_attachments = build_attachments(&mut cmdbuf, header_size)?;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS => {
                    fragment_attachments = build_attachments(&mut cmdbuf, header_size)?;
                }
                uapi::drm_asahi_cmd_type_DRM_ASAHI_SET_COMPUTE_ATTACHMENTS => {
                    compute_attachments = build_attachments(&mut cmdbuf, header_size)?;
                }
                _ => {
                    cls_pr_debug!(Errors, "Unknown command type {}\n", header.cmd_type);
                    return Err(EINVAL);
                }
            }
        }

        mod_dev_dbg!(
            self.dev,
            "Queue {}: Committing job {}\n",
            self.inner.id,
            job.id
        );
        job.commit()?;

        mod_dev_dbg!(self.dev, "Queue {}: Arming job {}\n", self.inner.id, job.id);
        let mut job = job.arm();
        let out_fence = job.fences().finished();
        if comp_serial {
            // SAFETY: the CompSerialGuard is held until the end of submit().
            let (ring, n) = unsafe { &mut *COMP_SERIAL_LAST.0.get() };
            let k = (*crate::module_parameters::comp_serial.value() as usize).clamp(1, COMP_SERIAL_MAX);
            ring[*n % k] = Some((out_fence.clone(), self.inner.id));
            *n = n.wrapping_add(1);
        }
        mod_dev_dbg!(
            self.dev,
            "Queue {}: Pushing job {}\n",
            self.inner.id,
            job.id
        );
        job.push();

        mod_dev_dbg!(
            self.dev,
            "Queue {}: Adding {} out_syncs\n",
            self.inner.id,
            syncs.len()
        );
        for mut sync in syncs {
            if let Some(chain) = sync.chain_fence.take() {
                sync.syncobj
                    .add_point(chain, &out_fence, sync.timeline_value);
            } else {
                sync.syncobj.replace_fence(Some(&out_fence));
            }
        }

        Ok(())
    }
}

#[versions(AGX)]
impl Drop for Queue::ver {
    fn drop(&mut self) {
        mod_dev_dbg!(self.dev, "[Queue {}] Dropping queue\n", self.inner.id);
        let qq = *crate::module_parameters::q_quarantine.value();
        // 667zzf: firmware list nodes of this queue, to match against a later
        // firmware fault (q_quarantine bit 3).
        if qq & 8 != 0 {
            pr_info!(
                "asahi/qlife: drop queue {} notifier {:#x} notifier_list {:#x}\n",
                self.inner.id,
                u64::from(self.inner.notifier.weak_pointer()),
                u64::from(self.inner.notifier_list.weak_pointer())
            );
            for (n, q) in [("vtx", self.q_vtx.as_ref()), ("frag", self.q_frag.as_ref()), ("comp", self.q_comp.as_ref())] {
                if let Some(q) = q {
                    pr_info!(
                        "asahi/qlife: drop queue {} {} info {:#x} ctx {:#x}\n",
                        self.inner.id,
                        n,
                        u64::from(q.wq.info_pointer()),
                        u64::from(self.inner.gpu_context.gpu_pointer())
                    );
                }
            }
        }
        // 667zzf diagnostics (q_quarantine): bit 2 reads the notifier the way
        // the firmware sees it (shared). Measured: cur_count == threshold and
        // in_list 0 on every drop -- notifiers are unlinked in time; the reuse
        // corruption was the misnumbered 15.5 DestroyContext (32.41.667zzg).
        // Bits 0/1 leak the queue's firmware objects instead of freeing them.
        if qq & 4 != 0 {
            // shared (uncached) notifier: these reads are the firmware's view
            let thr = self.inner.notifier.threshold.with(|raw, _inner| raw.0.load(Ordering::Relaxed));
            self.inner.notifier.with(|raw, _inner| {
                let in_list = raw.state.in_list;
                pr_info!(
                    "asahi/qlife: drop queue {} cur_count {:#x} threshold {:#x} in_list {} gen {:#x}\n",
                    self.inner.id,
                    raw.cur_count.load(Ordering::Relaxed),
                    thr,
                    in_list,
                    raw.generation.load(Ordering::Relaxed)
                );
            });
        }
        if qq & 1 != 0 {
            core::mem::forget(self.inner.notifier.clone());
            core::mem::forget(self.inner.notifier_list.clone());
        }
        if qq & 2 != 0 {
            for q in [self.q_vtx.as_ref(), self.q_frag.as_ref(), self.q_comp.as_ref()]
                .into_iter()
                .flatten()
            {
                core::mem::forget(q.wq.clone());
            }
        }
        // 667i: this dump runs on every GL context teardown and dump_hung_state
        // reads scene buffers through raw pointers a previous scene may have
        // freed (Oops in dump_scene_bufs) -> diagnostic only, off by default.
        if *crate::module_parameters::drop_dump.value() == 0 {
            return;
        }
        // Read the firmware's completion stamps on the way out. If they have
        // advanced to the submitted value the work finished and only the
        // notification is missing; if they have not, the job never ran.
        // LOG.md 32.41.81.
        for q in [self.q_vtx.as_ref(), self.q_frag.as_ref(), self.q_comp.as_ref()]
            .into_iter()
            .flatten()
        {
            q.wq.dump_state(&self.dev);
            q.wq.dump_info();
        }

        // The notifier is what the firmware bumps as it makes progress through
        // a job; `cur_count` still at its submitted value with the stamps
        // unmoved means the pass was entered and never left. LOG.md 32.41.89.
        self.inner.notifier.with(|raw, _inner| {
            let vm_slot = raw.state.vm_slot;
            let in_list = raw.state.in_list;
            let has_vtx = raw.state.has_vtx;
            let has_frag = raw.state.has_frag;
            let pstamp_vtx = raw.state.pstamp_vtx[0].0;
            let pstamp_frag = raw.state.pstamp_frag[0].0;
            dev_err!(
                self.dev.as_ref(),
                "  Notifier: generation {:#x} cur_count {:#x} unk_10 {:#x} vm_slot {} in_list {} has_vtx {} pstamp_vtx {:#x} has_frag {} pstamp_frag {:#x}\n",
                raw.generation.load(Ordering::Relaxed),
                raw.cur_count.load(Ordering::Relaxed),
                raw.unk_10.load(Ordering::Relaxed),
                vm_slot,
                in_list,
                has_vtx,
                pstamp_vtx,
                has_frag,
                pstamp_frag,
            );
        });

        (*self.dev).gpu.dump_hung_state();
    }
}
