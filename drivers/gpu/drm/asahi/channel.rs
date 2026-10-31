// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU ring buffer channels
//!
//! The GPU firmware use a set of ring buffer channels to receive commands from the driver and send
//! it notifications and status messages.
//!
//! These ring buffers mostly follow uniform conventions, so they share the same base
//! implementation.

use crate::debug::*;
use crate::driver::{
    AsahiDevRef,
    AsahiDevice, //
};
use crate::fw::channels::*;
use crate::fw::initdata::{
    raw,
    ChannelRing, //
};
use crate::fw::types::*;
use crate::{
    buffer,
    event,
    gpu,
    mem, //
};
use core::sync::atomic::{
    AtomicU32,
    Ordering, //
};
use kernel::{
    c_str,
    prelude::*,
    sync::Arc,
    time::{
        delay::fsleep,
        Delta,
        Instant,
        Monotonic, //
    },
};

pub(crate) use crate::fw::channels::PipeType;

/// A receive (FW->driver) channel.
pub(crate) struct RxChannel<T: RxChannelState, U: Copy + Default>
where
    for<'a> <T as GpuStruct>::Raw<'a>: Debug + Default + Zeroable,
{
    ring: ChannelRing<T, U>,
    // FIXME: needs feature(generic_const_exprs)
    //rptr: [u32; T::SUB_CHANNELS],
    rptr: [u32; 8],
    count: u32,
}

impl<T: RxChannelState, U: Copy + Default> RxChannel<T, U>
where
    for<'a> <T as GpuStruct>::Raw<'a>: Debug + Default + Zeroable,
{
    /// Allocates a new receive channel with a given message count.
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators, count: usize) -> Result<RxChannel<T, U>> {
        Ok(RxChannel {
            ring: ChannelRing {
                state: alloc.shared.new_default()?,
                ring: alloc.shared.array_empty(T::SUB_CHANNELS * count)?,
            },
            rptr: Default::default(),
            count: count as u32,
        })
    }

    /// Receives a message on the specified sub-channel index, optionally leaving in the ring
    /// buffer.
    ///
    /// Returns None if the channel is empty.
    fn get_or_peek(&mut self, index: usize, peek: bool) -> Option<U> {
        self.ring.state.with(|raw, _inner| {
            let wptr = T::wptr(raw, index);
            let rptr = &mut self.rptr[index];
            if wptr == *rptr {
                None
            } else {
                let off = self.count as usize * index;
                let msg = self.ring.ring[off + *rptr as usize];
                if !peek {
                    *rptr = (*rptr + 1) % self.count;
                    T::set_rptr(raw, index, *rptr);
                }
                Some(msg)
            }
        })
    }

    /// Receives a message on the specified sub-channel index, and dequeues it from the ring buffer.
    ///
    /// Returns None if the channel is empty.
    pub(crate) fn get(&mut self, index: usize) -> Option<U> {
        self.get_or_peek(index, false)
    }

    /// Peeks a message on the specified sub-channel index, leaving it in the ring buffer.
    ///
    /// Returns None if the channel is empty.
    pub(crate) fn peek(&mut self, index: usize) -> Option<U> {
        self.get_or_peek(index, true)
    }
}

/// A transmit (driver->FW) channel.
pub(crate) struct TxChannel<T: TxChannelState, U: Copy + Default>
where
    for<'a> <T as GpuStruct>::Raw<'a>: Debug + Default + Zeroable,
{
    ring: ChannelRing<T, U>,
    wptr: u32,
    count: u32,
}

impl<T: TxChannelState, U: Copy + Default> TxChannel<T, U>
where
    for<'a> <T as GpuStruct>::Raw<'a>: Debug + Default + Zeroable,
{
    /// Allocates a new cached transmit channel with a given message count.
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators, count: usize) -> Result<TxChannel<T, U>> {
        Ok(TxChannel {
            ring: ChannelRing {
                state: alloc.shared.new_default()?,
                ring: alloc.private.array_empty(count)?,
            },
            wptr: 0,
            count: count as u32,
        })
    }

    /// Allocates a new uncached transmit channel with a given message count.
    pub(crate) fn new_uncached(
        alloc: &mut gpu::KernelAllocators,
        count: usize,
    ) -> Result<TxChannel<T, U>> {
        Ok(TxChannel {
            ring: ChannelRing {
                state: alloc.shared.new_default()?,
                ring: alloc.shared.array_empty(count)?,
            },
            wptr: 0,
            count: count as u32,
        })
    }

    /// Send a message to the ring, returning a cookie with the ring buffer position.
    ///
    /// This will poll/block if the ring is full, which we don't really expect to happen.
    pub(crate) fn put(&mut self, msg: &U) -> u32 {
        self.ring.state.with(|raw, _inner| {
            let next_wptr = (self.wptr + 1) % self.count;
            let mut rptr = T::rptr(raw);
            if next_wptr == rptr {
                pr_err!(
                    "TX ring buffer is full! Waiting... ({}, {})\n",
                    next_wptr,
                    rptr
                );
                // TODO: block properly on incoming messages?
                while next_wptr == rptr {
                    fsleep(Delta::from_millis(8));
                    rptr = T::rptr(raw);
                }
            }
            self.ring.ring[self.wptr as usize] = *msg;
            mem::sync();
            T::set_wptr(raw, next_wptr);
            self.wptr = next_wptr;

            // Whether the firmware is draining this ring at all. rptr is
            // written by the firmware, so if it stays 0 while wptr climbs, our
            // messages are never being consumed -- which is the difference
            // between "the firmware ignores the doorbell" and "it runs the work
            // but never reports completion". LOG.md 30.11.
            // The state VA identifies *which* channel this is; without it every
            // TX channel prints the same line and the pipes cannot be told
            // apart from device control. Cross-reference it against the slot
            // dump gpu.rs prints at init.
            cls_pr_debug!(
                PipeCh,
                "TX put: state {:#x} wptr {} fw rptr {} fw ack {}\n",
                u64::from(self.ring.state.weak_pointer()),
                next_wptr,
                rptr,
                T::read_ack(raw)
            );
        });
        self.wptr
    }

    /// Report the firmware's read pointer against ours.
    ///
    /// The pipe consumer sets `RingState.gpu_rptr` from the message's `wptr`
    /// *as it consumes the message* (`fw:0x32590`), so a work queue stuck at
    /// `rptr 0` cannot distinguish "the firmware read the kick and refused the
    /// work" from "the firmware never read the kick". This does: the channel's
    /// own `read_ptr` is written by the firmware when it pops a message off
    /// the pipe ring, and it is plain shared memory -- no MMIO, so it is safe
    /// to call at teardown. LOG.md 32.41.101.
    pub(crate) fn dump(&self, dev: &AsahiDevice, name: &str, index: usize) {
        self.ring.state.with(|raw, _inner| {
            dev_err!(
                dev,
                "asahi/hung: pipe {}[{}] state {:#x} cpu wptr {} fw rptr {} fw ack {}\n",
                name,
                index,
                u64::from(self.ring.state.weak_pointer()),
                self.wptr,
                T::rptr(raw),
                T::read_ack(raw),
            );
        });
    }

    /// Wait for a previously submitted message to be popped off of the ring by the GPU firmware.
    ///
    /// This busy-loops, and is intended to be used for rare cases when we need to block for
    /// completion of a cache management or invalidation operation synchronously (which
    /// the firmware normally completes fast enough not to be worth sleeping for).
    /// If the poll takes longer than 10ms, this switches to sleeping between polls.
    pub(crate) fn wait_for(&mut self, wptr: u32, timeout_ms: i64) -> Result {
        const MAX_FAST_POLL: i64 = 10;
        let start = Instant::<Monotonic>::now();
        let timeout_ms = timeout_ms.max(1);
        let timeout_fast = Delta::from_millis(timeout_ms.min(MAX_FAST_POLL));
        let timeout_slow = Delta::from_millis(timeout_ms);
        self.ring.state.with(|raw, _inner| {
            while start.elapsed() < timeout_fast {
                if T::rptr(raw) == wptr {
                    return Ok(());
                }
                mem::sync();
            }
            while start.elapsed() < timeout_slow {
                if T::rptr(raw) == wptr {
                    return Ok(());
                }
                fsleep(Delta::from_millis(5));
                mem::sync();
            }
            pr_err!(
                "asahi: channel wait timed out: wptr {}, firmware rptr {}, read_ack {}\n",
                wptr,
                T::rptr(raw),
                T::read_ack(raw),
            );
            Err(ETIMEDOUT)
        })
    }
}

/// Device Control channel for global device management commands.
#[versions(AGX)]
pub(crate) struct DeviceControlChannel {
    dev: AsahiDevRef,
    ch: TxChannel<ChannelState, DeviceControlMsg::ver>,
}

#[versions(AGX)]
impl DeviceControlChannel::ver {
    const COMMAND_TIMEOUT_MS: i64 = 1000;

    /// Allocate a new Device Control channel.
    pub(crate) fn new(
        dev: &AsahiDevice,
        alloc: &mut gpu::KernelAllocators,
    ) -> Result<DeviceControlChannel::ver> {
        Ok(DeviceControlChannel::ver {
            dev: dev.into(),
            ch: TxChannel::<ChannelState, DeviceControlMsg::ver>::new(alloc, 0x100)?,
        })
    }

    /// Returns the raw `ChannelRing` structure to pass to firmware.
    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, DeviceControlMsg::ver> {
        self.ch.ring.to_raw()
    }

    /// Submits a Device Control command.
    pub(crate) fn send(&mut self, msg: &DeviceControlMsg::ver) -> u32 {
        cls_dev_dbg!(DeviceControlCh, self.dev, "DeviceControl: {:?}\n", msg);
        self.ch.put(msg)
    }

    /// Waits for a previously submitted Device Control command to complete.
    pub(crate) fn wait_for(&mut self, wptr: u32) -> Result {
        self.ch.wait_for(wptr, Self::COMMAND_TIMEOUT_MS)
    }
}

/// Pipe channel to submit WorkQueue execution requests.
#[versions(AGX)]
pub(crate) struct PipeChannel {
    dev: AsahiDevRef,
    ch: TxChannel<ChannelState, PipeMsg::ver>,
}

#[versions(AGX)]
impl PipeChannel::ver {
    /// Allocate a new Pipe submission channel.
    pub(crate) fn new(
        dev: &AsahiDevice,
        alloc: &mut gpu::KernelAllocators,
    ) -> Result<PipeChannel::ver> {
        Ok(PipeChannel::ver {
            dev: dev.into(),
            // Uncached: the 15.5 pipe consumer reads the 24-byte records with
            // no cache maintenance (fw:0x324f0), so a firmware-cached ring hands
            // it a line fetched while the next slot was still empty -> a zero
            // record, QueueInfo NULL, data abort at fw:0x3255c (666n, record 35
            // of 36 at 0xfffffc200003f0c8, a private-range VA).
            ch: TxChannel::<ChannelState, PipeMsg::ver>::new_uncached(alloc, 0x100)?,
        })
    }

    /// Returns the raw `ChannelRing` structure to pass to firmware.
    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, PipeMsg::ver> {
        self.ch.ring.to_raw()
    }

    /// Dump this pipe's ring pointers. See `TxChannel::dump`.
    pub(crate) fn dump_state(&self, name: &str, index: usize) {
        self.ch.dump(&self.dev, name, index);
    }

    /// Submits a Pipe kick command to the firmware.
    pub(crate) fn send(&mut self, msg: &PipeMsg::ver) {
        cls_dev_dbg!(PipeCh, self.dev, "Pipe: {:?}\n", msg);
        self.ch.put(msg);
    }
}

/// Firmware Control channel, used for secure cache flush requests.
pub(crate) struct FwCtlChannel {
    dev: AsahiDevRef,
    ch: TxChannel<FwCtlChannelState, FwCtlMsg>,
}

impl FwCtlChannel {
    const COMMAND_TIMEOUT_MS: i64 = 1000;

    /// Allocate a new Firmware Control channel.
    pub(crate) fn new(
        dev: &AsahiDevice,
        alloc: &mut gpu::KernelAllocators,
    ) -> Result<FwCtlChannel> {
        Ok(FwCtlChannel {
            dev: dev.into(),
            ch: TxChannel::<FwCtlChannelState, FwCtlMsg>::new_uncached(alloc, 0x100)?,
        })
    }

    /// Returns the raw `ChannelRing` structure to pass to firmware.
    pub(crate) fn to_raw(&self) -> raw::ChannelRing<FwCtlChannelState, FwCtlMsg> {
        self.ch.ring.to_raw()
    }

    /// Submits a Firmware Control command to the firmware.
    pub(crate) fn send(&mut self, msg: &FwCtlMsg) -> u32 {
        cls_dev_dbg!(FwCtlCh, self.dev, "FwCtl: {:?}\n", msg);
        self.ch.put(msg)
    }

    /// Waits for a previously submitted Firmware Control command to complete.
    pub(crate) fn wait_for(&mut self, wptr: u32) -> Result {
        self.ch.wait_for(wptr, Self::COMMAND_TIMEOUT_MS)
    }
}

/// Event channel, used to notify the driver of command completions, GPU faults and errors, and
/// other events.
#[versions(AGX)]
pub(crate) struct EventChannel {
    dev: AsahiDevRef,
    ch: RxChannel<EventChannelState, RawEventMsg>,
    ev_mgr: Arc<event::EventManager>,
    buf_mgr: buffer::BufferManager::ver,
    gpu: Option<Arc<dyn gpu::GpuManager>>,
}

#[versions(AGX)]
impl EventChannel::ver {
    /// Allocate a new Event channel.
    pub(crate) fn new(
        dev: &AsahiDevice,
        alloc: &mut gpu::KernelAllocators,
        ev_mgr: Arc<event::EventManager>,
        buf_mgr: buffer::BufferManager::ver,
    ) -> Result<EventChannel::ver> {
        Ok(EventChannel::ver {
            dev: dev.into(),
            ch: RxChannel::<EventChannelState, RawEventMsg>::new(alloc, 0x100)?,
            ev_mgr,
            buf_mgr,
            gpu: None,
        })
    }

    /// Registers the managing `Gpu` instance that will handle events on this channel.
    pub(crate) fn set_manager(&mut self, gpu: Arc<dyn gpu::GpuManager>) {
        self.gpu = Some(gpu);
    }

    /// Returns the raw `ChannelRing` structure to pass to firmware.
    pub(crate) fn to_raw(&self) -> raw::ChannelRing<EventChannelState, RawEventMsg> {
        self.ch.ring.to_raw()
    }

    /// Polls for new Event messages on this ring.
    pub(crate) fn poll(&mut self) -> usize {
        let mut n = 0;
        for i in 0..EventChannelState::SUB_CHANNELS {
        while let Some(msg) = self.ch.get(i) {
            n += 1;
            // SAFETY: The raw view is always valid for all bit patterns.
            let tag = unsafe { msg.raw.0 };
            // SAFETY: The raw view is always valid for all bit patterns.
            let rawv = unsafe { msg.raw };
            match tag {
                0..=EVENT_MAX => {
                    // SAFETY: Since we have checked the tag to be in range,
                    // accessing the enum view is valid.
                    let msg = unsafe { msg.msg };

                    cls_dev_dbg!(EventCh, self.dev, "Event: {:?}\n", msg);
                    // The 13.5 field offsets inside each variant are *not*
                    // verified for 15.5 -- the ring stride already had to grow
                    // from 0x38 to 0x40, and RawKTraceMsg's whole layout moved.
                    // Dump the bytes for anything that is not the Flag event we
                    // already trust, so the variant can be re-derived offline
                    // instead of guessed. LOG.md 32.41.114.
                    if tag != 1 {
                        // SAFETY: The raw view is valid for all bit patterns.
                        let raw = rawv;
                        let b: &[u8; EVENT_SZ] = &raw.1;
                        let w = |i: usize| -> u32 {
                            u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
                        };
                        cls_dev_dbg!(
                            EventCh,
                            self.dev,
                            "Event raw: tag {} | {:#010x} {:#010x} {:#010x} {:#010x} {:#010x} {:#010x} {:#010x} {:#010x}\n",
                            raw.0,
                            w(0), w(4), w(8), w(0xc), w(0x10), w(0x14), w(0x18), w(0x1c)
                        );
                    }
                    match msg {
                        EventMsg::Fault => match self.gpu.as_ref() {
                            Some(gpu) => gpu.handle_fault(),
                            None => {
                                dev_crit!(
                                    self.dev.as_ref(),
                                    "EventChannel: No GPU manager available!\n"
                                )
                            }
                        },
                        EventMsg::Timeout {
                            counter,
                            unk_8,
                            event_slot,
                        } => match self.gpu.as_ref() {
                            Some(gpu) => gpu.handle_timeout(counter, event_slot, unk_8),
                            None => {
                                dev_crit!(
                                    self.dev.as_ref(),
                                    "EventChannel: No GPU manager available!\n"
                                )
                            }
                        },
                        EventMsg::Flag { firing, .. } => {
                            for (i, flags) in firing.iter().enumerate() {
                                for j in 0..32 {
                                    if flags & (1u32 << j) != 0 {
                                        self.ev_mgr.signal((i * 32 + j) as u32);
                                    }
                                }
                            }
                        }
                        EventMsg::GrowTVB {
                            vm_slot,
                            buffer_slot,
                            counter,
                        } => match self.gpu.as_ref() {
                            Some(gpu) => {
                                self.buf_mgr.grow(buffer_slot);
                                gpu.ack_grow(buffer_slot, vm_slot, counter);
                            }
                            None => {
                                dev_crit!(
                                    self.dev.as_ref(),
                                    "EventChannel: No GPU manager available!\n"
                                )
                            }
                        },
                        EventMsg::ChannelError {
                            error_type,
                            pipe_type,
                            event_slot,
                            event_value,
                        } => match self.gpu.as_ref() {
                            Some(gpu) => {
                                let error_type = match error_type {
                                    0 => ChannelErrorType::MemoryError,
                                    1 => ChannelErrorType::DMKill,
                                    2 => ChannelErrorType::Aborted,
                                    3 => ChannelErrorType::Unk3,
                                    a => ChannelErrorType::Unknown(a),
                                };
                                gpu.handle_channel_error(
                                    error_type,
                                    pipe_type,
                                    event_slot,
                                    event_value,
                                );
                            }
                            None => {
                                dev_crit!(
                                    self.dev.as_ref(),
                                    "EventChannel: No GPU manager available!\n"
                                )
                            }
                        },
                        msg => {
                            dev_crit!(self.dev.as_ref(), "Unknown event message: {:?}\n", msg);
                        }
                    }
                }
                13 => {
                    // 15.5 UMA grow retired: the firmware has drained one
                    // <=0xff8-page chunk of a DC 0x1e page list (fw:0x246b8).
                    // The kext frees its staging list here
                    // (AGXUMAFList::retireAsyncGrowRequest); our page list is
                    // the pool's persistent aperture, so nothing to free.
                    // {u32 hwid, u32 pool id, u64 pool record, u64 cookie
                    // (DC unk_1c), u64 list address, u32 pages}. LOG.md 32.41.667zzu.
                    let b: &[u8; EVENT_SZ] = &rawv.1;
                    let w = |i: usize| -> u32 {
                        u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
                    };
                    let q = |i: usize| -> u64 { u64::from(w(i)) | (u64::from(w(i + 4)) << 32) };
                    cls_dev_dbg!(
                        EventCh,
                        self.dev,
                        "UMA grow retired: hwid {} pool {} record {:#x} list {:#x} pages {:#x}\n",
                        w(0), w(4), q(8), q(0x18), w(0x20)
                    );
                    crate::gpu::uma_chunk_retired(w(4));
                    // 667zzw: the pool's last chunk is out -> push more pages.
                    if let Some(gpu) = self.gpu.as_ref() {
                        crate::buffer::uma_grow_on_retire(&**gpu, w(4), q(0x18), w(0x20));
                    }
                }
                _ => {
                    // SAFETY: The raw view is always valid for all bit patterns.
                    dev_warn!(self.dev.as_ref(), "Unknown event message: {:?}\n", unsafe {
                        msg.raw
                    });
                }
            }
        }
        }
        n
    }
}

/// Firmware Log channel. This one is pretty special, since it has 6 sub-channels (for different log
/// levels), and it also uses a side buffer to actually hold the log messages, only passing around
/// pointers in the main buffer.
pub(crate) struct FwLogChannel {
    dev: AsahiDevRef,
    ch: RxChannel<FwLogChannelState, RawFwLogMsg>,
    payload_buf: GpuArray<RawFwLogPayloadMsg>,
}

impl FwLogChannel {
    const RING_SIZE: usize = 0x100;
    const BUF_SIZE: usize = 0x100;

    /// Allocate a new Firmware Log channel.
    pub(crate) fn new(
        dev: &AsahiDevice,
        alloc: &mut gpu::KernelAllocators,
    ) -> Result<FwLogChannel> {
        Ok(FwLogChannel {
            dev: dev.into(),
            ch: RxChannel::<FwLogChannelState, RawFwLogMsg>::new(alloc, Self::RING_SIZE)?,
            payload_buf: alloc
                .shared
                .array_empty(Self::BUF_SIZE * FwLogChannelState::SUB_CHANNELS)?,
        })
    }

    /// Returns the raw `ChannelRing` structure to pass to firmware.
    pub(crate) fn to_raw(&self) -> raw::ChannelRing<FwLogChannelState, RawFwLogMsg> {
        self.ch.ring.to_raw()
    }

    /// Returns the GPU pointers to the firmware log payload buffer.
    pub(crate) fn get_buf(&self) -> GpuWeakPointer<[RawFwLogPayloadMsg]> {
        self.payload_buf.weak_pointer()
    }

    /// Polls for new log messages on all sub-rings.
    pub(crate) fn poll(&mut self) {
        for i in 0..=FwLogChannelState::SUB_CHANNELS - 1 {
            while let Some(msg) = self.ch.peek(i) {
                cls_dev_dbg!(FwLogCh, self.dev, "FwLog{}: {:?}\n", i, msg);
                if msg.msg_type != 2 {
                    dev_warn!(self.dev.as_ref(), "Unknown FWLog{} message: {:?}\n", i, msg);
                    self.ch.get(i);
                    continue;
                }
                if msg.msg_index.0 as usize >= Self::BUF_SIZE {
                    dev_warn!(
                        self.dev.as_ref(),
                        "FWLog{} message index out of bounds: {:?}\n",
                        i,
                        msg
                    );
                    self.ch.get(i);
                    continue;
                }
                let index = Self::BUF_SIZE * i + msg.msg_index.0 as usize;
                let payload = &self.payload_buf.as_slice()[index];
                if payload.msg_type != 3 {
                    dev_warn!(
                        self.dev.as_ref(),
                        "Unknown FWLog{} payload: {:?}\n",
                        i,
                        payload
                    );
                    self.ch.get(i);
                    continue;
                }
                let msg = if let Some(end) = payload.msg.iter().position(|&r| r == 0) {
                    CStr::from_bytes_with_nul(&(*payload.msg)[..end + 1])
                        .unwrap_or(c_str!("cstr_err"))
                } else {
                    dev_warn!(
                        self.dev.as_ref(),
                        "FWLog{} payload not NUL-terminated: {:?}\n",
                        i,
                        payload
                    );
                    self.ch.get(i);
                    continue;
                };
                match i {
                    0 => dev_dbg!(self.dev.as_ref(), "FWLog: {}\n", msg),
                    1 => dev_info!(self.dev.as_ref(), "FWLog: {}\n", msg),
                    2 => dev_notice!(self.dev.as_ref(), "FWLog: {}\n", msg),
                    3 => dev_warn!(self.dev.as_ref(), "FWLog: {}\n", msg),
                    4 => dev_err!(self.dev.as_ref(), "FWLog: {}\n", msg),
                    5 => dev_crit!(self.dev.as_ref(), "FWLog: {}\n", msg),
                    _ => (),
                };
                self.ch.get(i);
            }
        }
    }
}

#[versions(AGX)]
pub(crate) struct KTraceChannel {
    dev: AsahiDevRef,
    ch: RxChannel<ChannelState, RawKTraceMsg::ver>,
}

/// KTrace channel, used to receive detailed execution trace markers from the firmware.
///
/// Silent unless `Globals::ktrace_enable` (the `ktrace_mask` module parameter)
/// turns on at least one of the six firmware trace categories.
#[versions(AGX)]
impl KTraceChannel::ver {
    /// Allocate a new KTrace channel.
    pub(crate) fn new(
        dev: &AsahiDevice,
        alloc: &mut gpu::KernelAllocators,
    ) -> Result<KTraceChannel::ver> {
        Ok(KTraceChannel::ver {
            dev: dev.into(),
            ch: RxChannel::<ChannelState, RawKTraceMsg::ver>::new(alloc, 0x200)?,
        })
    }

    /// Returns the raw `ChannelRing` structure to pass to firmware.
    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, RawKTraceMsg::ver> {
        self.ch.ring.to_raw()
    }

    /// Polls for new KTrace messages on this ring.
    ///
    /// Printed compactly rather than with `{:?}` because all six categories
    /// together produce thousands of records.  LOG.md 32.41.112.
    pub(crate) fn poll(&mut self) {
        while let Some(msg) = self.ch.get(0) {
            #[ver(V < V15_5)]
            cls_dev_dbg!(KTraceCh, self.dev, "KTrace: {:?}\n", msg);
            #[ver(V >= V15_5)]
            {
                let (a0, a1, a2, a3) =
                    (msg.args[0].0, msg.args[1].0, msg.args[2].0, msg.args[3].0);
                let idcpu = msg.id_cpu.0;
                let flags = msg.flags.0;
                let kind = msg.kind;
                // The firmware's own timestamp matters more than the dmesg one:
                // records are only drained when the driver polls, so a burst
                // that prints in one millisecond can span a minute of firmware
                // time.  LOG.md 32.41.112.
                let ts = msg.timestamp.0;
                // ⚠️ `cls_dev_dbg!` lowers to `dev_dbg`, which this kernel
                // config compiles out -- 32.41.276 read "no trace records" off
                // a print that could never have appeared.  The flag check is
                // kept; the emitter is a `pr_info!` so the records actually
                // reach dmesg.  LOG.md 32.41.277.
                // skip the periodic 1 ms timer (0x105/0x106) and DVFS (0x2xx) records
                let hid = idcpu & 0xffffff;
                if *crate::module_parameters::ktrace_hist.value() != 0
                    && hid != 0x105
                    && hid != 0x106
                    && (*crate::module_parameters::ktrace_hist.value() >= 2
                        || !(0x200..0x300).contains(&hid))
                    // .667zzz: 3 keeps only the job/context classes (0x0xx, 0x6xx) so a
                    // tail dumped at each compute completion covers the whole job.
                    && (*crate::module_parameters::ktrace_hist.value() < 3
                        || !matches!(hid >> 8, 1 | 2 | 4))
                {
                    ktrace_hist_push([ts, u64::from(idcpu), (u64::from(kind) << 32) | u64::from(flags), a0, a1, a2, a3]);
                }
                if crate::debug::debug_enabled(crate::debug::DebugFlags::KTraceCh) {
                    kernel::pr_info!(
                    "KTrace: t {:#x} id {:#06x} cpu {} kind {} flags {:#x} | {:#x} {:#x} {:#x} {:#x}\n",
                    ts,
                    idcpu & 0xffffff,
                    idcpu >> 24,
                    kind,
                    flags,
                    a0,
                    a1,
                    a2,
                    a3
                );
                }
            }
        }
    }
}
static KT_HIST_LAST: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
/// .667zs: the last KT_HIST_N firmware trace records, kept without printing
/// (printing them perturbs the timing enough to hide a pipelining hang) and
/// dumped once on the first job timeout/cancel. `ktrace_hist=1`.
const KT_HIST_N: usize = 512;
const KT_Z: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static KT_HIST: [core::sync::atomic::AtomicU64; KT_HIST_N * 7] = [KT_Z; KT_HIST_N * 7];
static KT_HIST_W: AtomicU32 = AtomicU32::new(0);
static KT_TAIL_LAST: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
static KT_HIST_DUMPED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

fn ktrace_hist_push(r: [u64; 7]) {
    // 667zzy: a looping record (e.g. 0x110 every 1 ms while a compute job hangs)
    // flushed the whole history; keep one copy of each run of identical records.
    let key = r[1] ^ r[3].rotate_left(8) ^ r[4].rotate_left(16) ^ r[5].rotate_left(24) ^ r[6].rotate_left(32);
    if KT_HIST_LAST.swap(key, Ordering::Relaxed) == key {
        return;
    }
    let i = (KT_HIST_W.fetch_add(1, Ordering::Relaxed) as usize) % KT_HIST_N;
    for (k, v) in r.iter().enumerate() {
        KT_HIST[i * 7 + k].store(*v, Ordering::Relaxed);
    }
}

/// .667zzz: `ktrace_hist=2` also prints the last `n` records on every GPU fault
/// (repeatable), so an intermittent fault carries the firmware events before it.
pub(crate) fn ktrace_hist_tail(n: usize) {
    // .667zzz41: 4 = the class filter of 3, dumped only on the first timeout.
    if !matches!(*crate::module_parameters::ktrace_hist.value(), 2 | 3) {
        return;
    }
    let w = KT_HIST_W.load(Ordering::Relaxed) as usize;
    // 3: only what arrived since the previous tail (one job's records).
    let n = if *crate::module_parameters::ktrace_hist.value() >= 3 {
        n.min(w - KT_TAIL_LAST.swap(w, Ordering::Relaxed).min(w))
    } else {
        n
    };
    let n = w.min(n).min(KT_HIST_N);
    kernel::pr_info!("asahi/kttail: {} records (of {} seen)\n", n, w);
    for j in 0..n {
        let i = (w - n + j) % KT_HIST_N;
        let g = |k: usize| KT_HIST[i * 7 + k].load(Ordering::Relaxed);
        let (idcpu, kf) = (g(1), g(2));
        kernel::pr_info!(
            "KTrace: t {:#x} id {:#06x} cpu {} kind {} flags {:#x} | {:#x} {:#x} {:#x} {:#x}\n",
            g(0),
            idcpu & 0xffffff,
            idcpu >> 24,
            kf >> 32,
            kf & 0xffff_ffff,
            g(3),
            g(4),
            g(5),
            g(6)
        );
    }
}

/// Print the kept trace records, oldest first, in the `KTrace:` format
/// decode-ktrace.py reads. Only the first call prints.
pub(crate) fn ktrace_hist_dump() {
    if *crate::module_parameters::ktrace_hist.value() == 0
        || KT_HIST_DUMPED.swap(true, Ordering::AcqRel)
    {
        return;
    }
    let w = KT_HIST_W.load(Ordering::Relaxed) as usize;
    let n = w.min(KT_HIST_N);
    kernel::pr_info!("asahi/kthist: {} records (of {} seen)\n", n, w);
    for j in 0..n {
        let i = (w - n + j) % KT_HIST_N;
        let g = |k: usize| KT_HIST[i * 7 + k].load(Ordering::Relaxed);
        let (idcpu, kf) = (g(1), g(2));
        kernel::pr_info!(
            "KTrace: t {:#x} id {:#06x} cpu {} kind {} flags {:#x} | {:#x} {:#x} {:#x} {:#x}\n",
            g(0),
            idcpu & 0xffffff,
            idcpu >> 24,
            kf >> 32,
            kf & 0xffff_ffff,
            g(3),
            g(4),
            g(5),
            g(6)
        );
    }
}

/// How many statistics messages have ever been logged as samples.
static STATS_SAMPLES: AtomicU32 = AtomicU32::new(0);


/// Statistics channel, reporting power-related statistics to the driver.
/// Not really implemented other than debug logs yet...
#[versions(AGX)]
pub(crate) struct StatsChannel {
    dev: AsahiDevRef,
    ch: RxChannel<ChannelState, RawStatsMsg::ver>,
}

#[versions(AGX)]
impl StatsChannel::ver {
    /// Allocate a new Statistics channel.
    pub(crate) fn new(
        dev: &AsahiDevice,
        alloc: &mut gpu::KernelAllocators,
    ) -> Result<StatsChannel::ver> {
        Ok(StatsChannel::ver {
            dev: dev.into(),
            ch: RxChannel::<ChannelState, RawStatsMsg::ver>::new(alloc, 0x100)?,
        })
    }

    /// Returns the raw `ChannelRing` structure to pass to firmware.
    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, RawStatsMsg::ver> {
        self.ch.ring.to_raw()
    }

    /// Polls for new statistics messages on this ring.
    ///
    /// Returns how many were drained. The count matters on 15.5: the firmware
    /// stops sending MSG_RX_DOORBELL once this ring fills, so the only evidence
    /// that the AP-side poll is keeping up is how much it takes off the ring.
    pub(crate) fn poll(&mut self) -> usize {
        let mut n = 0;
        while let Some(msg) = self.ch.get(0) {
            n += 1;
            // Sample the first few messages ever seen, unconditionally. The
            // ring refills at ~80k messages/s on 15.5, which is either a real
            // statistics storm or a misread ring; the raw bytes tell which,
            // and a bounded sample does not wrap the kernel log. LOG.md 32.41.
            if crate::debug::verbose() && STATS_SAMPLES.fetch_add(1, Ordering::Relaxed) < 24 {
                // SAFETY: The raw view is always valid for all bit patterns.
                pr_info!("asahi: stats sample: {:?}\n", unsafe { msg.raw });
            }
            // SAFETY: The raw view is always valid for all bit patterns.
            let tag = unsafe { msg.raw.0 };
            // SAFETY: as above.
            crate::stats::fw_stat(tag, unsafe { &*msg.raw.1 });
            match tag {
                0..=STATS_MAX::ver => {
                    // SAFETY: Since we have checked the tag to be in range,
                    // accessing the enum view is valid.
                    let msg = unsafe { msg.msg };
                    cls_dev_dbg!(StatsCh, self.dev, "Stats: {:?}\n", msg);
                }
                _ => {
                    // 15.5 renumbered this enum and added to it -- tags 25, 27
                    // and 28 arrive continuously once the stats ring at
                    // RegionB+0x1e8 is published. They are informational, and
                    // at warn level they flood the kernel log hard enough to
                    // wrap out everything else, so keep them behind StatsCh.
                    // SAFETY: The raw view is always valid for all bit patterns.
                    cls_dev_dbg!(StatsCh, self.dev, "Unknown stats message: {:?}\n", unsafe {
                        msg.raw
                    });
                }
            }
        }
        n
    }
}
