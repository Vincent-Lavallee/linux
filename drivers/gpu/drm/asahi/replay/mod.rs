// SPDX-License-Identifier: GPL-2.0-only OR MIT
//! Apple's own user-side GPU buffers for one 64x64 clear + triangle draw,
//! captured on macOS 15.5 (capture4, res-draw-pre, stream terminator from
//! res-draw-post). Trailing zeros trimmed. LOG.md 32.41.531.

pub(crate) struct Res { pub(crate) va: u64, pub(crate) size: usize, pub(crate) data: &'static [u8] }

pub(crate) static RES: &[Res] = &[
    Res { va: 0x0, size: 0x10000, data: include_bytes!("r000000000000.bin") }, // 772 bytes
    Res { va: 0x18000, size: 0x8000, data: include_bytes!("r000000018000.bin") }, // 120 bytes
    Res { va: 0x48000, size: 0x8000, data: include_bytes!("r000000048000.bin") }, // 4 bytes
    Res { va: 0x58000, size: 0x8000, data: include_bytes!("r000000058000.bin") }, // 7516 bytes
    Res { va: 0x10000000000, size: 0x10000, data: include_bytes!("r010000000000.bin") }, // 47044 bytes
    Res { va: 0x10000018000, size: 0x20000, data: include_bytes!("r010000018000.bin") }, // 11520 bytes
    Res { va: 0x10000050000, size: 0x10000, data: include_bytes!("r010000050000.bin") }, // 1284 bytes
    Res { va: 0x10000090000, size: 0x74000, data: include_bytes!("r010000090000.bin") }, // 469316 bytes
    Res { va: 0x10000108000, size: 0x8000, data: include_bytes!("r010000108000.bin") }, // 24 bytes
    Res { va: 0x10000118000, size: 0x8000, data: include_bytes!("r010000118000.bin") }, // 1524 bytes
    Res { va: 0x10000128000, size: 0x8000, data: include_bytes!("r010000128000.bin") }, // 80 bytes
    Res { va: 0x10000138000, size: 0x8000, data: include_bytes!("r010000138000.bin") }, // 1684 bytes
    Res { va: 0x10000148000, size: 0x100000, data: include_bytes!("r010000148000.bin") }, // 16 bytes
    Res { va: 0x10000250000, size: 0x8000, data: include_bytes!("r010000250000.bin") }, // 2952 bytes
    // The 9 resources Apple's draw also had mapped but that were all-zero at
    // capture time, so capture4 did not save them (res-draw-pre/MAP.txt).
    // LOG.md 32.41.555.
    Res { va: 0x28000, size: 0x8000, data: &[] },
    Res { va: 0x38000, size: 0x8000, data: &[] },
    Res { va: 0x6f00000000, size: 0x4000, data: &[] },
    Res { va: 0x10000068000, size: 0x20000, data: &[] },
    Res { va: 0x10000260000, size: 0x8000, data: &[] },
    Res { va: 0x10000270000, size: 0x8000, data: &[] },
    Res { va: 0x10000280000, size: 0x8000, data: &[] },
    Res { va: 0x10000290000, size: 0x8000, data: &[] },
    Res { va: 0x100002a0000, size: 0xc0000, data: &[] },
    // The render target Apple's EOT descriptor (0x10000118620, field << 4) names; an IOSurface the
    // capture never saved. 64x64 RGBA8 = 0x4000; mapped 0x8000. LOG.md 32.41.612.
    Res { va: 0x10000040000, size: 0x8000, data: &[] },
];

/// Apple's own compute job (GLES-equivalent 64-invocation SSBO write, 0x11111111 ->
/// 0xC0FFEE00+i at 0x10000030000), captured on macOS 15.5 before dispatch
/// (2026-09-27-compute, dt-big, res-compute-pre). CDM stream at 0x100000b0000.
/// LOG.md 32.41.572.
pub(crate) static CRES: &[Res] = &[
    Res { va: 0x0, size: 0x10000, data: include_bytes!("c000000000000.bin") }, // 388 bytes
    Res { va: 0x6f00000000, size: 0x4000, data: &[] },
    Res { va: 0x10000000000, size: 0x10000, data: include_bytes!("c010000000000.bin") }, // 22020 bytes
    Res { va: 0x10000018000, size: 0x10000, data: include_bytes!("c010000018000.bin") }, // 516 bytes
    Res { va: 0x10000030000, size: 0x20000, data: include_bytes!("c010000030000.bin") }, // 11776 bytes
    Res { va: 0x10000058000, size: 0x20000, data: &[] },
    Res { va: 0x10000080000, size: 0x8000, data: include_bytes!("c010000080000.bin") }, // 192 bytes
    Res { va: 0x10000090000, size: 0x8000, data: include_bytes!("c010000090000.bin") }, // 156 bytes
    Res { va: 0x100000a0000, size: 0x8000, data: &[] },
    Res { va: 0x100000b0000, size: 0x8000, data: include_bytes!("c0100000b0000.bin") }, // 48 bytes; +0x2c terminator 0x40000000 from res-compute-post (written at commit)
    Res { va: 0x100000c0000, size: 0x8000, data: &[] },
    Res { va: 0x100000d0000, size: 0x8000, data: &[] },
    Res { va: 0x100000e0000, size: 0xc000, data: include_bytes!("c0100000e0000.bin") }, // 7080 bytes
    // 32.41.574: Apple's kext spill buffer for this job (dt-big op 0xb +0x188 / the
    // firmware-appended 0xa5c1.. registers): 0x10000968000, +0x1e400, +0x3c800, +0x3d800.
    Res { va: 0x10000968000, size: 0x40000, data: &[] },
];

/// Apple's fragment/EOT descriptors as they are AFTER the capture's draw (0x10000118600..0x8e8);
/// the pre-draw blob ends at 0x5f4, so the EOT had no image descriptors. LOG.md 32.41.611.
pub(crate) static R118000_POST: &[u8] = include_bytes!("r010000118000-post.bin");

/// CPU view of the replayed draw's VA-0 buffer (0x10000 bytes); the render target
/// descriptor at 0x10000118620 names 0x1000004000 (= VA 0x4000 in it). LOG.md 32.41.611.
pub(crate) static DRAW_VA0: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
/// CPU view of the render target mapped at 0x10000040000.
pub(crate) static DRAW_RT: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Summarise the 64x64 RGBA8 render target: distinct colours and a coarse 8x8 map.
pub(crate) fn draw_rt_dump(when: &str) {
    let p = DRAW_RT.load(core::sync::atomic::Ordering::SeqCst) as *const u32;
    if p.is_null() {
        return;
    }
    // SAFETY: the mapping is leaked and 0x8000 bytes long; we read the first 0x4000.
    let px = |x: usize, y: usize| unsafe { core::ptr::read_volatile(p.add(y * 64 + x)) };
    let mut cols = [(0u32, 0u32); 8];
    let mut ncols = 0usize;
    let mut nz = 0u32;
    for y in 0..64 {
        for x in 0..64 {
            let v = px(x, y);
            if v != 0 {
                nz += 1;
            }
            let cap = cols.len();
            match cols[..ncols].iter_mut().find(|c| c.0 == v) {
                Some(c) => c.1 += 1,
                None if ncols < cap => {
                    cols[ncols] = (v, 1);
                    ncols += 1;
                }
                None => {}
            }
        }
    }
    // SAFETY: +0x4800 is inside the 0x8000-byte leaked mapping. LOG.md 32.41.643.
    kernel::pr_info!("asahi/rt {}: sentinel +0x4800 = {:#010x}\n", when, unsafe { core::ptr::read_volatile(p.add(0x4800 / 4)) });
    kernel::pr_info!("asahi/rt {}: {} of 4096 pixels non-zero; px(0,0) {:#010x} px(63,63) {:#010x} px(32,32) {:#010x}\n",
        when, nz, px(0, 0), px(63, 63), px(32, 32));
    for c in &cols[..ncols] {
        kernel::pr_info!("asahi/rt {}: colour {:#010x} x {}\n", when, c.0, c.1);
    }
    // Runs of non-zero words over the whole 0x8000 mapping (first 24), to see
    // the surface's real layout (tiled/twiddled vs linear). LOG.md 32.41.613.
    let mut runs = 0u32;
    let mut w = 0usize;
    while w < 0x2000 {
        // SAFETY: inside the 0x8000-byte leaked mapping.
        if unsafe { core::ptr::read_volatile(p.add(w)) } != 0 {
            let start = w;
            let first = unsafe { core::ptr::read_volatile(p.add(w)) };
            while w < 0x2000 && unsafe { core::ptr::read_volatile(p.add(w)) } != 0 {
                w += 1;
            }
            if runs < 24 {
                kernel::pr_info!("asahi/rt {}: run +{:#06x}..+{:#06x} ({} px) first {:#010x}\n",
                    when, start * 4, w * 4, w - start, first);
            }
            runs += 1;
        } else {
            w += 1;
        }
    }
    kernel::pr_info!("asahi/rt {}: {} runs\n", when, runs);
    // Raw words: first 16 of each 0x400-byte tile, then the 8-byte-per-tile
    // metadata plane at +0x4000. LOG.md 32.41.619.
    // SAFETY: inside the 0x8000-byte leaked mapping.
    let rd = |i: usize| unsafe { core::ptr::read_volatile(p.add(i)) };
    for t in 0..16usize {
        let b = t * 0x100;
        if (0..16).all(|i| rd(b + i) == 0) {
            continue;
        }
        kernel::pr_info!("asahi/rt {}: tile{:2} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
            when, t, rd(b), rd(b + 1), rd(b + 2), rd(b + 3), rd(b + 4), rd(b + 5), rd(b + 6), rd(b + 7),
            rd(b + 8), rd(b + 9), rd(b + 10), rd(b + 11), rd(b + 12), rd(b + 13), rd(b + 14), rd(b + 15));
    }
    for r in 0..4usize {
        let b = 0x1000 + r * 8;
        kernel::pr_info!("asahi/rt {}: meta+{:#04x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}\n",
            when, r * 32, rd(b), rd(b + 1), rd(b + 2), rd(b + 3), rd(b + 4), rd(b + 5), rd(b + 6), rd(b + 7));
    }
    for gy in 0..8 {
        let mut row = [b'.'; 8];
        for gx in 0..8 {
            let v = px(gx * 8 + 4, gy * 8 + 4);
            row[gx] = if v == 0 { b'.' } else if v == cols[0].0 { b'a' } else { b'#' };
        }
        kernel::pr_info!("asahi/rt {}: |{}|\n", when, core::str::from_utf8(&row).unwrap_or("?"));
    }
}

/// Print what changed in the VA-0 buffer versus the uploaded blob.
pub(crate) fn draw_va0_dump(when: &str) {
    let p = DRAW_VA0.load(core::sync::atomic::Ordering::SeqCst) as *const u32;
    if p.is_null() {
        return;
    }
    let orig = RES[0].data;
    let mut n = 0u32;
    let mut first = [0usize; 4];
    for w in 0..(0x10000 / 4) {
        // SAFETY: the mapping is leaked and 0x10000 bytes long.
        let v = unsafe { core::ptr::read_volatile(p.add(w)) };
        let o = w * 4;
        let was = if o + 4 <= orig.len() {
            u32::from_le_bytes([orig[o], orig[o + 1], orig[o + 2], orig[o + 3]])
        } else {
            0
        };
        if v != was {
            if (n as usize) < first.len() {
                first[n as usize] = o;
            }
            if n < 16 {
                kernel::pr_info!("asahi/va0 {}: +{:#06x} = {:#010x} (was {:#010x})\n", when, o, v, was);
            }
            n += 1;
        }
    }
    // SAFETY: as above; pixel (0,0) and (63,63) of a 64x64 RGBA8 surface at +0x4000.
    let (px0, px1) = unsafe {
        (core::ptr::read_volatile(p.add(0x4000 / 4)), core::ptr::read_volatile(p.add((0x4000 + 63 * 256 + 63 * 4) / 4)))
    };
    kernel::pr_info!(
        "asahi/va0 {}: {} words changed; +0x4000 px(0,0) {:#010x} px(63,63) {:#010x}\n",
        when, n, px0, px1
    );
}

/// CPU view of Apple's replayed output buffer (0x10000030000), set by comp_apple=2.
pub(crate) static COMP_REPLAY_OUT: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Print the replayed output buffer: Apple's shader writes 0xC0FFEE00+i to the
/// first 64 words (0x11111111 before). LOG.md 32.41.572.
pub(crate) fn comp_replay_dump(when: &str) {
    let p = COMP_REPLAY_OUT.load(core::sync::atomic::Ordering::SeqCst) as *const u32;
    if p.is_null() {
        return;
    }
    let mut w = [0u32; 64];
    for (i, x) in w.iter_mut().enumerate() {
        // SAFETY: the mapping is leaked and at least 0x20000 bytes long.
        *x = unsafe { core::ptr::read_volatile(p.add(i)) };
    }
    let good = w.iter().enumerate().filter(|(i, &v)| v == 0xc0ffee00 + *i as u32).count();
    let untouched = w.iter().filter(|&&v| v == 0x11111111).count();
    let own = w.iter().enumerate().filter(|(i, &v)| v == 2 * *i as u32 + 5).count();
    let esl = w.iter().enumerate().filter(|(i, &v)| v == 0xfffff000u32.wrapping_add(*i as u32)).count();
    kernel::pr_info!(
        "asahi/creplay: output {}: {:08x} {:08x} {:08x} {:08x} ... {:08x} -- {}/64 correct, {} untouched, {}/64 == 2i+5, {}/64 == i-0x1000\n",
        when, w[0], w[1], w[2], w[3], w[63], good, untouched, own, esl
    );
}

/// CPU views of every replayed compute buffer, to diff after the run.
static COMP_REPLAY_PTRS: [core::sync::atomic::AtomicUsize; 16] =
    [const { core::sync::atomic::AtomicUsize::new(0) }; 16];

/// Remember where replay buffer `res` lives in CPU space. LOG.md 32.41.573.
pub(crate) fn comp_replay_note(res: &Res, p: *mut u8) {
    for (i, r) in CRES.iter().enumerate() {
        if core::ptr::eq(r, res) && i < COMP_REPLAY_PTRS.len() {
            COMP_REPLAY_PTRS[i].store(p as usize, core::sync::atomic::Ordering::SeqCst);
        }
    }
}

/// Log every word of every replayed buffer that differs from what was
/// uploaded (first 24 per buffer): what the GPU wrote during the run.
pub(crate) fn comp_replay_diff(when: &str) {
    for (i, r) in CRES.iter().enumerate() {
        let p = COMP_REPLAY_PTRS[i].load(core::sync::atomic::Ordering::SeqCst) as *const u32;
        if p.is_null() {
            continue;
        }
        let mut n = 0u32;
        for w in 0..(r.size / 4) {
            // SAFETY: the mapping is leaked and `r.size` bytes long.
            let v = unsafe { core::ptr::read_volatile(p.add(w)) };
            let o = w * 4;
            let orig = if o + 4 <= r.data.len() {
                u32::from_le_bytes([r.data[o], r.data[o + 1], r.data[o + 2], r.data[o + 3]])
            } else {
                0
            };
            if v != orig {
                if n < 24 {
                    kernel::pr_info!(
                        "asahi/cdiff {}: {:#x}+{:#x} = {:#010x} (was {:#010x})\n",
                        when, r.va, o, v, orig
                    );
                }
                n += 1;
            }
        }
        if n != 0 {
            kernel::pr_info!("asahi/cdiff {}: {:#x}: {} words changed\n", when, r.va, n);
        }
    }
}

/// Snapshot the head of Apple's spill buffer (+0..+0x60) and a whole-buffer
/// checksum, to compare USC context across successive preemptions. LOG.md 32.41.576.
pub(crate) fn comp_spill_snap(when: &str) {
    for (i, r) in CRES.iter().enumerate() {
        if r.va != 0x100_0096_8000 {
            continue;
        }
        let p = COMP_REPLAY_PTRS[i].load(core::sync::atomic::Ordering::SeqCst) as *const u32;
        if p.is_null() {
            return;
        }
        // SAFETY: the mapping is leaked and `r.size` bytes long.
        let rd = |w: usize| unsafe { core::ptr::read_volatile(p.add(w)) };
        let mut sum = 0u64;
        let mut nz = 0u32;
        for w in 0..(r.size / 4) {
            let v = rd(w);
            sum = sum.wrapping_mul(31).wrapping_add(v as u64);
            if v != 0 {
                nz += 1;
            }
        }
        let mut h = [0u32; 24];
        for (k, x) in h.iter_mut().enumerate() {
            *x = rd(k);
        }
        kernel::pr_info!(
            "asahi/spill {}: nz {} sum {:#x} head {:08x?}\n",
            when, nz, sum, h
        );
    }
}
