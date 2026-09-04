//! PatchMatch-based text inpainting — single source of truth shared by both the
//! browser (WebAssembly) and the Python backend (wasmtime).
//!
//! Why this exists: the previous JS (patchmatch.js) and Python (patch_fill.py)
//! ports used *different* PRNGs (mulberry32 vs PCG64) and *different* fill-loop
//! structures (per-pixel vs batched priority), so their outputs could never match.
//! This module reimplements the algorithm ONCE in Rust; both ends call the exact
//! same wasm bytes, so their output is byte-identical by construction.
//!
//! Algorithm (Criminisi priority fill + PatchMatch), faithfully mirroring
//! `text_eraser/patch_fill.py::inpaint`'s numpy loop:
//!   1. candidate source centres = erode(known, P×P), minus half-px border
//!   2. per boundary pixel (highest Criminisi priority first): search a pool of
//!      K random candidates + 4 neighbourhood-coherence sources via SSD over the
//!      *known* pixels of the target patch (mean-compat penalty added)
//!   3. copy the best source patch into the target hole pixels with local colour
//!      self-adaptation (anchor = original-known snapshot, expanding window 7→11→17)
//!   4. direction mode (direction_deg != None) restricts candidates to the line
//!      through each target pixel at that angle
//!   5. any residual holes are smoothed by OpenCV TELEA (telea.rs), exactly like
//!      the Python cv2.inpaint(..., INPAINT_TELEA) fallback
//!
//! The smooth-gradient *background pre-check* that the Python reference
//! `patch_fill.inpaint` performs BEFORE the PatchMatch loop (lines 116-153) is
//! ALSO ported here (see `pm_smooth_telea`), so the shared core reproduces the
//! FULL Python algorithm — not just the loop. On smooth / low-texture
//! backgrounds PatchMatch produces flat / colour-difference blocks, so the
//! Python path falls back to a whole-image TELEA inpaint; we do the same with
//! `telea.rs`, keeping the browser / backend / original-cv2 pipeline 1:1.

const PM_PADM: i32 = 4;

#[inline]
fn pm_idx(i: i32, len: i32) -> i32 {
    if i < 0 {
        0
    } else if i >= len {
        len - 1
    } else {
        i
    }
}

#[inline]
fn pm_clamp(v: f32) -> f32 {
    if v < 0.0 {
        0.0
    } else if v > 255.0 {
        255.0
    } else {
        v
    }
}

/// mulberry32 PRNG —— 与浏览器 linalg.js 的 mulberry32、Python patch_fill._Mulberry32
/// 逐位一致(f64 输出语义)。注意: 旧实现其实不是 mulberry32 而是 xorshift32,
/// 且共享核所有调用点 seed=0 —— xorshift32 零状态恒输出 0, 随机候选池退化成
/// 固定取 cand[0], 这是填充纹理与 Python 核心(真随机 PCG64)不一致的根源。
fn pm_mulberry32(s: &mut u32) -> f64 {
    *s = s.wrapping_add(0x6D2B79F5);
    let mut t = *s;
    t = (t ^ (t >> 15)).wrapping_mul(t | 1);
    t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t));
    let o = t ^ (t >> 14);
    (o as f64) / 4294967296.0f64
}

fn pm_sobel(gray: &[f32], h: i32, w: i32, gx: &mut [f32], gy: &mut [f32]) {
    for y in 0..h {
        for x in 0..w {
            let xa = [pm_idx(x - 1, w), pm_idx(x, w), pm_idx(x + 1, w)];
            let ya = [pm_idx(y - 1, h), pm_idx(y, h), pm_idx(y + 1, h)];
            let tl = gray[(ya[0] * w + xa[0]) as usize];
            let tc = gray[(ya[0] * w + xa[1]) as usize];
            let tr = gray[(ya[0] * w + xa[2]) as usize];
            let ml = gray[(ya[1] * w + xa[0]) as usize];
            let mr = gray[(ya[1] * w + xa[2]) as usize];
            let bl = gray[(ya[2] * w + xa[0]) as usize];
            let bc = gray[(ya[2] * w + xa[1]) as usize];
            let br = gray[(ya[2] * w + xa[2]) as usize];
            let o = (y * w + x) as usize;
            gx[o] = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
            gy[o] = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
        }
    }
}

/// dst[i] = mean of src over the P×P window centred at i (clamped at borders).
fn pm_box_mean(src: &[f32], dst: &mut [f32], h: i32, w: i32, p: i32) {
    let half = p / 2;
    let mut s = vec![0f64; ((h + 1) * (w + 1)) as usize];
    for y in 0..h {
        let mut row = 0f64;
        for x in 0..w {
            let v = src[(y * w + x) as usize] as f64;
            row += v;
            let idx = ((y + 1) * (w + 1) + (x + 1)) as usize;
            s[idx] = s[((y) * (w + 1) + (x + 1)) as usize] + row;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let y0 = (y - half).max(0);
            let y1 = (y + half).min(h - 1);
            let x0 = (x - half).max(0);
            let x1 = (x + half).min(w - 1);
            let sum = s[((y1 + 1) * (w + 1) + (x1 + 1)) as usize]
                - s[((y1 + 1) * (w + 1) + x0) as usize]
                - s[(y0 * (w + 1) + (x1 + 1)) as usize]
                + s[(y0 * (w + 1) + x0) as usize];
            let cnt = ((y1 - y0 + 1) * (x1 - x0 + 1)) as f32;
            dst[(y * w + x) as usize] = (sum as f32) / cnt;
        }
    }
}

fn pm_erode3x3(m: &[u8], dst: &mut [u8], h: i32, w: i32) {
    for y in 0..h {
        for x in 0..w {
            let mut v = 255u8;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let p = m[(pm_idx(y + dy, h) * w + pm_idx(x + dx, w)) as usize];
                    if p < v {
                        v = p;
                    }
                }
            }
            dst[(y * w + x) as usize] = v;
        }
    }
}

fn pm_dilate_max3x3(m: &[f32], dst: &mut [f32], h: i32, w: i32) {
    for y in 0..h {
        for x in 0..w {
            let mut v = 0f32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let p = m[(pm_idx(y + dy, h) * w + pm_idx(x + dx, w)) as usize];
                    if p > v {
                        v = p;
                    }
                }
            }
            dst[(y * w + x) as usize] = v;
        }
    }
}

/// dst[i] = 255 iff all P×P neighbours (in-bounds and known) of i are known.
fn pm_erode_rect(known: &[u8], dst: &mut [u8], h: i32, w: i32, p: i32) {
    let half = p / 2;
    for y in 0..h {
        for x in 0..w {
            let mut v = 255u8;
            'outer: for dy in -half..=half {
                for dx in -half..=half {
                    let ny = y + dy;
                    let nx = x + dx;
                    if ny < 0 || ny >= h || nx < 0 || nx >= w {
                        v = 0;
                        break 'outer;
                    }
                    if known[(ny * w + nx) as usize] == 0 {
                        v = 0;
                        break 'outer;
                    }
                }
            }
            dst[(y * w + x) as usize] = v;
        }
    }
}

/// Gather a P×P×3 patch centred at (cy,cx) from `work` (clamped/replicated at edges).
fn pm_gather(work: &[f32], cy: i32, cx: i32, pw: i32, ph: i32, half: i32, pp: usize, pp3: usize) -> Vec<f32> {
    let mut out = vec![0f32; pp3];
    let mut o = 0;
    for j in 0..pp {
        let dy = (j as i32 / (half * 2 + 1)) - half;
        let dx = (j as i32 % (half * 2 + 1)) - half;
        let yy = pm_idx(cy + dy, ph);
        let xx = pm_idx(cx + dx, pw);
        let s = ((yy * pw + xx) * 3) as usize;
        out[o] = work[s];
        out[o + 1] = work[s + 1];
        out[o + 2] = work[s + 2];
        o += 3;
    }
    out
}

fn pm_patch_mean(p: &[f32], pp: usize) -> [f32; 3] {
    let mut m = [0f32; 3];
    for j in 0..pp {
        let o = j * 3;
        m[0] += p[o];
        m[1] += p[o + 1];
        m[2] += p[o + 2];
    }
    let n = pp as f32;
    [m[0] / n, m[1] / n, m[2] / n]
}

fn pm_write_out(work: &[f32], h: usize, w: usize, pw: i32, out_ptr: *mut f32) {
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, h * w * 3) };
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let si = (((y + PM_PADM) * pw + (x + PM_PADM)) * 3) as usize;
            let di = ((y * w as i32 + x) * 3) as usize;
            out[di] = pm_clamp(work[si]);
            out[di + 1] = pm_clamp(work[si + 1]);
            out[di + 2] = pm_clamp(work[si + 2]);
        }
    }
}

/// Find the best source patch centre for a single target pixel, in the spirit of
/// `patch_fill.inpaint`'s `_best_source` / fast-path pool. Returns (sy, sx).
fn pm_best_source(
    work: &[f32], known: &[u8], spad: Option<&[u8]>, cand: &[u8],
    ph: i32, pw: i32, HALF: i32, PP: usize, PP3: usize,
    ty: i32, tx: i32,
    cand_y: &[i32], cand_x: &[i32], nc: usize, k: usize,
    rng: &mut u32, use_dir: bool, dir_vec: Option<(f32, f32, i32, i32)>,
    nnf_y: &[i32], nnf_x: &[i32], nnf_set: &[u8],
) -> (i32, i32) {
    const DY4: [i32; 4] = [-1, 1, 0, 0];
    const DX4: [i32; 4] = [0, 0, -1, 1];

    let tpatch = pm_gather(work, ty, tx, pw, ph, HALF, PP, PP3);
    // known positions in the target window (patch-flat indices)
    // 注意: 必须 - HALF 使窗口以目标为中心(ty-3..ty+3), 与 pm_gather 一致;
    // 此前漏减导致 known 蒙版采样窗口整体右下偏移, SSD 在错误的已知位置上
    // 计算, 与 numpy 参照结构性不一致。
    let mut tkidx: Vec<usize> = Vec::new();
    for j in 0..PP {
        let wy = ty + (j as i32 / (HALF * 2 + 1)) - HALF;
        let wx = tx + (j as i32 % (HALF * 2 + 1)) - HALF;
        if wy >= 0 && wy < ph && wx >= 0 && wx < pw {
            let i = (wy * pw + wx) as usize;
            if known[i] != 0 {
                tkidx.push(j);
            }
        }
    }
    let tkn_sum = (tkidx.len() as f32).max(1.0);
    let mut tm = [0f32; 3];
    for &j in &tkidx {
        let o = j * 3;
        tm[0] += tpatch[o];
        tm[1] += tpatch[o + 1];
        tm[2] += tpatch[o + 2];
    }
    tm[0] /= tkn_sum;
    tm[1] /= tkn_sum;
    tm[2] /= tkn_sum;

    // ---- build candidate pool ----
    let mut pool_y: Vec<i32> = Vec::new();
    let mut pool_x: Vec<i32> = Vec::new();
    if use_dir {
        let (ux, uy, maxd, step) = dir_vec.unwrap();
        let mut any_line = false;
        for sign in [1i32, -1i32] {
            let mut d = step;
            while d <= maxd {
                let cy = (ty as f32 + sign as f32 * d as f32 * uy).round() as i32;
                let cx = (tx as f32 + sign as f32 * d as f32 * ux).round() as i32;
                if cy < HALF || cy >= ph - HALF || cx < HALF || cx >= pw - HALF {
                    d += step;
                    continue;
                }
                // Only accept line points whose *full* P×P patch is known (cand =
                // erode(known, P)). A bare `known[cy,cx]` centre can sit <P px from
                // the hole, so its gather window would include white hole pixels and
                // get copied verbatim when the target patch is fully inside the hole
                // (tkidx empty => SSD always 0). This matches the non-dir pool, which
                // already draws only from `cand`.
                let ii = (cy * pw + cx) as usize;
                let ok = cand[ii] != 0 && (if let Some(s) = spad { s[ii] != 0 } else { true });
                if ok {
                    pool_y.push(cy);
                    pool_x.push(cx);
                    any_line = true;
                }
                d += step;
            }
        }
        if !any_line {
            for _ in 0..k {
                let ci = ((pm_mulberry32(rng) * nc as f64) as usize).min(nc - 1);
                pool_y.push(cand_y[ci]);
                pool_x.push(cand_x[ci]);
            }
        }
    } else {
        for _ in 0..k {
            let ci = ((pm_mulberry32(rng) * nc as f64) as usize).min(nc - 1);
            pool_y.push(cand_y[ci]);
            pool_x.push(cand_x[ci]);
        }
    }
    // neighbourhood coherence: reuse already-filled neighbours' source centre
    for kk in 0..4 {
        let ny = ty + DY4[kk];
        let nx = tx + DX4[kk];
        if ny >= 0 && ny < ph && nx >= 0 && nx < pw {
            let i = (ny * pw + nx) as usize;
            if nnf_set[i] != 0 {
                pool_y.push(nnf_y[i]);
                pool_x.push(nnf_x[i]);
            }
        }
    }

    // ---- evaluate pool: SSD over known target positions + mean-compat penalty ----
    let mut best_s = f32::INFINITY;
    let mut best_i = 0usize;
    for (pi, (&cy, &cx)) in pool_y.iter().zip(pool_x.iter()).enumerate() {
        let sp = pm_gather(work, cy, cx, pw, ph, HALF, PP, PP3);
        let sm = pm_patch_mean(&sp, PP);
        let mut ssd = 0f32;
        for &j in &tkidx {
            let o = j * 3;
            let d0 = sp[o] - tpatch[o];
            let d1 = sp[o + 1] - tpatch[o + 1];
            let d2 = sp[o + 2] - tpatch[o + 2];
            ssd += d0 * d0 + d1 * d1 + d2 * d2;
        }
        if !use_dir {
            let dm0 = sm[0] - tm[0];
            let dm1 = sm[1] - tm[1];
            let dm2 = sm[2] - tm[2];
            ssd += 4.0 * tkn_sum * (dm0 * dm0 + dm1 * dm1 + dm2 * dm2);
        }
        if ssd < best_s {
            best_s = ssd;
            best_i = pi;
        }
    }
    (pool_y[best_i], pool_x[best_i])
}

/// Copy the source patch (via local colour self-adaptation) into the target hole
/// pixels, and update known/hole/NNF. Mirrors `patch_fill.inpaint`'s `_copy_patch`.
fn pm_copy_patch(
    work: &mut [f32], known: &mut [u8], hole: &mut [u8],
    nnf_y: &mut [i32], nnf_x: &mut [i32], nnf_set: &mut [u8],
    orig_known: &[u8],
    ph: i32, pw: i32, HALF: i32, PP: usize, PP3: usize,
    ty: i32, tx: i32, sy: i32, sx: i32,
) {
    let wy0 = ty - HALF;
    let wx0 = tx - HALF;
    let mut src = pm_gather(work, sy, sx, pw, ph, HALF, PP, PP3);

    // local colour self-adaptation: anchor = orig_known values in the target window.
    // Python (`patch_fill._copy_patch`) semantics, mirrored EXACTLY:
    //   ta = orig_known inside the 7x7 target window; tv = values there.
    //   If tv has < 8 pixels, tv is *REPLACED* by the values of the first
    //   expanding window (r=5 -> 11x11, r=8 -> 17x17) holding >= 8 orig_known
    //   pixels. If neither window reaches 8, tv stays the 7x7 values and the
    //   adaptation is skipped (len < 8).
    // (The previous code ACCUMULATED 7x7 + 11x11 + 17x17 values with duplicates,
    //  which silently shifted tmean/tstd away from the Python reference.)
    let mut ta: Vec<[f32; 3]> = Vec::new();
    for j in 0..PP {
        let wy = wy0 + (j as i32 / (HALF * 2 + 1));
        let wx = wx0 + (j as i32 % (HALF * 2 + 1));
        if wy >= 0 && wy < ph && wx >= 0 && wx < pw {
            let i = (wy * pw + wx) as usize;
            if orig_known[i] != 0 {
                ta.push([work[i * 3], work[i * 3 + 1], work[i * 3 + 2]]);
            }
        }
    }
    if ta.len() < 8 {
        for r in [5i32, 8i32] {
            let by0 = (ty - r).max(0);
            let by1 = (ty + r + 1).min(ph);
            let bx0 = (tx - r).max(0);
            let bx1 = (tx + r + 1).min(pw);
            let mut win: Vec<[f32; 3]> = Vec::new();
            for yy in by0..by1 {
                for xx in bx0..bx1 {
                    let i = (yy * pw + xx) as usize;
                    if orig_known[i] != 0 {
                        win.push([work[i * 3], work[i * 3 + 1], work[i * 3 + 2]]);
                    }
                }
            }
            if win.len() >= 8 {
                ta = win;
                break;
            }
        }
    }
    if ta.len() >= 8 {
        let tl = ta.len() as f32;
        let mut tmean = [0f32; 3];
        for v in &ta {
            tmean[0] += v[0];
            tmean[1] += v[1];
            tmean[2] += v[2];
        }
        tmean[0] /= tl;
        tmean[1] /= tl;
        tmean[2] /= tl;
        let mut tvar = [0f32; 3];
        for v in &ta {
            let a = v[0] - tmean[0];
            let b = v[1] - tmean[1];
            let c = v[2] - tmean[2];
            tvar[0] += a * a;
            tvar[1] += b * b;
            tvar[2] += c * c;
        }
        let tstd0 = (tvar[0] / tl).sqrt() + 1e-3;
        let tstd1 = (tvar[1] / tl).sqrt() + 1e-3;
        let tstd2 = (tvar[2] / tl).sqrt() + 1e-3;
        let mut sm = [0f32; 3];
        for i in 0..PP {
            let o = i * 3;
            sm[0] += src[o];
            sm[1] += src[o + 1];
            sm[2] += src[o + 2];
        }
        sm[0] /= PP as f32;
        sm[1] /= PP as f32;
        sm[2] /= PP as f32;
        let mut sv = [0f32; 3];
        for i in 0..PP {
            let o = i * 3;
            let a = src[o] - sm[0];
            let b = src[o + 1] - sm[1];
            let c = src[o + 2] - sm[2];
            sv[0] += a * a;
            sv[1] += b * b;
            sv[2] += c * c;
        }
        let sstd0 = (sv[0] / PP as f32).sqrt() + 1e-3;
        let sstd1 = (sv[1] / PP as f32).sqrt() + 1e-3;
        let sstd2 = (sv[2] / PP as f32).sqrt() + 1e-3;
        for i in 0..PP3 {
            let ch = i % 3;
            let srcv = src[i];
            let smv = sm[ch];
            let tmv = tmean[ch];
            let sstd = [sstd0, sstd1, sstd2][ch];
            let tstd = [tstd0, tstd1, tstd2][ch];
            src[i] = (srcv - smv) * (tstd / sstd) + tmv;
        }
    }

    // write the (colour-adapted) source into the hole pixels of the target patch
    for j in 0..PP {
        let wy = wy0 + (j as i32 / (HALF * 2 + 1));
        let wx = wx0 + (j as i32 % (HALF * 2 + 1));
        if wy < 0 || wy >= ph || wx < 0 || wx >= pw {
            continue;
        }
        let i = (wy * pw + wx) as usize;
        if hole[i] == 0 {
            continue;
        }
        let o = j * 3;
        let d = i * 3;
        work[d] = src[o];
        work[d + 1] = src[o + 1];
        work[d + 2] = src[o + 2];
        known[i] = 255;
        hole[i] = 0;
    }
    let bi = (ty * pw + tx) as usize;
    nnf_y[bi] = sy;
    nnf_x[bi] = sx;
    nnf_set[bi] = 1;
}

// ===========================================================================
// Smooth-gradient background pre-check (port of `patch_fill.inpaint` 116-153).
//
// When the hole sits on a smooth / low-texture background (e.g. the foggy golden
// glow of 换装.png, or any near-uniform region), PatchMatch's 7x7 block copy
// cannot preserve the gradient continuity and emits flat / colour-difference
// blocks (filling chroma_std ≈ 2.4 vs background 0.76). The Python reference
// detects this and falls back to a whole-image `cv2.inpaint(..., INPAINT_TELEA)`,
// which smoothly diffuses the local gradient into the hole instead. We reproduce
// the exact test and use `telea.rs` so the shared core matches the original
// (non-Rust) pipeline byte-for-byte.
// ===========================================================================

/// Median of a float slice (in-place sort). Matches numpy `np.median`.
fn pm_median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) * 0.5
    }
}

/// Median gray of the NON-mask pixels inside the [y0,y1) x [x0,x1) band. Returns
/// None if every pixel in the band is a hole (so it contributes no edge band).
fn pm_band_median(
    gray: &[f32], mask: &[u8], h: usize, w: usize,
    y0: usize, y1: usize, x0: usize, x1: usize,
) -> Option<f32> {
    let mut v: Vec<f32> = Vec::new();
    for y in y0..y1 {
        let base = y * w;
        for x in x0..x1 {
            let i = base + x;
            if mask[i] == 0 {
                v.push(gray[i]);
            }
        }
    }
    if v.is_empty() {
        None
    } else {
        Some(pm_median(&mut v))
    }
}

/// Separable (2-pass) box max-filter dilation of a 0/255 mask with square radius
/// `r` — equivalent to `cv2.dilate(mask, ones(2r+1, 2r+1))`. O(n·(2r+1)).
fn pm_box_dilate(m: &[u8], h: i32, w: i32, r: i32) -> Vec<u8> {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    let mut tmp = vec![0u8; n];
    for y in 0..h {
        let base = y * w;
        for x in 0..w {
            let x0 = ((x as i32 - r).max(0)) as usize;
            let x1 = ((x as i32 + r).min(w as i32 - 1)) as usize;
            let mut vv = 0u8;
            for xx in x0..=x1 {
                let val = m[base + xx];
                if val > vv {
                    vv = val;
                }
            }
            tmp[base + x] = vv;
        }
    }
    let mut out = vec![0u8; n];
    for x in 0..w {
        let mut col = x;
        for y in 0..h {
            let y0 = ((y as i32 - r).max(0)) as usize;
            let y1 = ((y as i32 + r).min(h as i32 - 1)) as usize;
            let mut vv = 0u8;
            for yy in y0..=y1 {
                let val = tmp[yy * w + x];
                if val > vv {
                    vv = val;
                }
            }
            out[col] = vv;
            col += w;
        }
    }
    out
}

/// Replicate `patch_fill.inpaint`'s smooth-gradient pre-check with an
/// adjustable `flat_tex` threshold. Returns `Some(telea_filled)` if the test
/// fires (caller should write it out and return early), or `None` to proceed
/// with the normal PatchMatch loop.
///
/// Mirrors exactly:
///   - 4 edge bands (12px) outside the mask bbox → median gray of the non-hole
///     pixels in each; need >= 2 bands with data (skips tiny / border masks).
///   - Sobel gradient magnitude `grad0` on the clamped grayscale.
///   - `ring0 = dilate(mask, 41x41) & ~mask` (20px ring around the hole).
///   - `tex = median(grad0[ring0])`; the flat `span` test was removed upstream,
///     only `tex < flat_tex` remains.
///   - Only runs when there is NO direction mode (`direction is None` on the
///     Python side → `use_dir == false` here).
pub(crate) fn pm_smooth_telea_with_flat_tex(rgb: &[f32], mask: &[u8], h: i32, w: i32, flat_tex: f32) -> Option<Vec<f32>> {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    if n == 0 {
        return None;
    }
    // grayscale (cv2 RGB2GRAY fixed-point; identical to rgb_to_gray)
    let mut gray = vec![0f32; n];
    for i in 0..n {
        let r = pm_clamp(rgb[i * 3]) as i32;
        let g = pm_clamp(rgb[i * 3 + 1]) as i32;
        let b = pm_clamp(rgb[i * 3 + 2]) as i32;
        let v = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
        gray[i] = if v < 0 { 0 } else if v > 255 { 255 } else { v } as f32;
    }
    // mask bbox
    let mut ymin = h;
    let mut ymax = 0usize;
    let mut xmin = w;
    let mut xmax = 0usize;
    let mut has = false;
    for y in 0..h {
        let base = y * w;
        for x in 0..w {
            if mask[base + x] != 0 {
                has = true;
                if y < ymin {
                    ymin = y;
                }
                if y > ymax {
                    ymax = y;
                }
                if x < xmin {
                    xmin = x;
                }
                if x > xmax {
                    xmax = x;
                }
            }
        }
    }
    if !has {
        return None;
    }
    let band = 12usize;
    let mut edges_med: Vec<f32> = Vec::new();
    if ymin >= band {
        if let Some(md) = pm_band_median(&gray, mask, h, w, ymin - band, ymin, xmin, xmax + 1) {
            edges_med.push(md);
        }
    }
    if ymax + band < h {
        if let Some(md) =
            pm_band_median(&gray, mask, h, w, ymax + 1, (ymax + band + 1).min(h), xmin, xmax + 1)
        {
            edges_med.push(md);
        }
    }
    if xmin >= band {
        if let Some(md) = pm_band_median(&gray, mask, h, w, ymin, ymax + 1, xmin - band, xmin) {
            edges_med.push(md);
        }
    }
    if xmax + band < w {
        if let Some(md) =
            pm_band_median(&gray, mask, h, w, ymin, ymax + 1, xmax + 1, (xmax + band + 1).min(w))
        {
            edges_med.push(md);
        }
    }
    if edges_med.len() < 2 {
        return None;
    }
    // Sobel gradient magnitude on the grayscale (ksize=3, matches cv2.Sobel)
    let mut gx = vec![0f32; n];
    let mut gy = vec![0f32; n];
    pm_sobel(&gray, h as i32, w as i32, &mut gx, &mut gy);
    let mut grad = vec![0f32; n];
    for i in 0..n {
        grad[i] = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
    }
    // ring0 = dilate(mask, 41x41) & ~mask  (Chebyshev radius 20)
    let dil = pm_box_dilate(mask, h as i32, w as i32, 20);
    let mut ring: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        if dil[i] != 0 && mask[i] == 0 {
            ring.push(grad[i]);
        }
    }
    let tex = if ring.is_empty() { 0.0 } else { pm_median(&mut ring) };
    if tex < flat_tex {
        return Some(crate::telea::telea_inpaint(rgb, mask, h, w, 3));
    }
    None
}

/// Default-threshold wrapper matching `patch_fill.inpaint`'s `flat_tex=20.0`.
fn pm_smooth_telea(rgb: &[f32], mask: &[u8], h: i32, w: i32) -> Option<Vec<f32>> {
    pm_smooth_telea_with_flat_tex(rgb, mask, h, w, 20.0)
}

#[no_mangle]
pub extern "C" fn patchmatch_inpaint(
    rgb_ptr: *const f32,
    h: i32,
    w: i32,
    mask_ptr: *const u8,
    sample_ptr: *const u8,
    has_sample: i32,
    p: i32,
    direction_deg: f32,
    seed: u32,
    out_ptr: *mut f32,
) {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    let rgb = unsafe { std::slice::from_raw_parts(rgb_ptr, n * 3) };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, n) };
    let sample = if has_sample != 0 {
        Some(unsafe { std::slice::from_raw_parts(sample_ptr, n) })
    } else {
        None
    };

    // ---- smooth-gradient background pre-check (port of patch_fill.inpaint 116-153) ----
    // If the hole sits on a low-texture background, TELEA diffuses the gradient
    // better than PatchMatch. `use_dir` mirrors Python's `direction is None`.
    let use_dir = direction_deg > -1.0;
    if !use_dir {
        if let Some(filled) = pm_smooth_telea(rgb, mask, h as i32, w as i32) {
            let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * 3) };
            for i in 0..n * 3 {
                out[i] = pm_clamp(filled[i]);
            }
            return;
        }
    }

    let ph = (h as i32) + 2 * PM_PADM;
    let pw = (w as i32) + 2 * PM_PADM;
    let pn = (ph * pw) as usize;
    let mut work = vec![0f32; pn * 3];
    let mut mpad = vec![0u8; pn];
    let mut spad = if sample.is_some() {
        vec![0u8; pn]
    } else {
        Vec::new()
    };
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let sy = pm_idx(y, h as i32);
            let sx = pm_idx(x, w as i32);
            let dy = y + PM_PADM;
            let dx = x + PM_PADM;
            let si = (sy * w as i32 + sx) as usize;
            let di = ((dy * pw + dx) * 3) as usize;
            work[di] = rgb[si * 3];
            work[di + 1] = rgb[si * 3 + 1];
            work[di + 2] = rgb[si * 3 + 2];
            mpad[(dy * pw + dx) as usize] = if mask[si] != 0 { 255 } else { 0 };
            if let Some(sm) = sample {
                spad[(dy * pw + dx) as usize] = if sm[si] != 0 { 255 } else { 0 };
            }
        }
    }

    // Edge-replicate the padding (cv2 BORDER_REPLICATE) so source patches gathered
    // near the image border read real pixel values instead of the uninitialised 0
    // fill. Without this, candidate centres a few px inside the border would copy
    // black padding pixels into the hole.
    for dy in 0..ph {
        for dx in 0..pw {
            if dy < PM_PADM || dy >= ph - PM_PADM || dx < PM_PADM || dx >= pw - PM_PADM {
                let sy = (pm_idx(dy - PM_PADM, h as i32) + PM_PADM) as usize;
                let sx = (pm_idx(dx - PM_PADM, w as i32) + PM_PADM) as usize;
                let si = (sy * pw as usize + sx) * 3;
                let di = (dy as usize * pw as usize + dx as usize) * 3;
                work[di] = work[si];
                work[di + 1] = work[si + 1];
                work[di + 2] = work[si + 2];
            }
        }
    }

    let P = p as usize;
    let HALF = (P / 2) as i32;
    let PP = P * P;
    let PP3 = PP * 3;

    let mut known = vec![0u8; pn];
    for i in 0..pn {
        known[i] = if mpad[i] != 0 { 0 } else { 255 };
    }
    let orig_known = known.clone();
    let mut hole = mpad.clone();

    let mut erode_buf = vec![0u8; pn];
    pm_erode_rect(&known, &mut erode_buf, ph, pw, p);
    let mut cand = erode_buf.clone();
    // Python zeroes the candidate ring on the *unpadded* ROI (`sub`): rows/cols
    // [0, half) and [sh-half, sh). The wasm input (h,w) IS that unpadded ROI and
    // this function pads it by PM_PADM, so the ring maps to padded indices
    // < HALF+PADM or >= h+PADM-HALF (not [0,half) / [ph-half,ph) as before).
    let sh_i = h as i32;
    let sw_i = w as i32;
    for y in 0..ph as i32 {
        for x in 0..pw as i32 {
            let i = (y * pw + x) as usize;
            if y < HALF + PM_PADM || y >= sh_i + PM_PADM - HALF
                || x < HALF + PM_PADM || x >= sw_i + PM_PADM - HALF {
                cand[i] = 0;
            }
            if !spad.is_empty() && spad[i] == 0 {
                cand[i] = 0;
            }
        }
    }
    let mut cand_y: Vec<i32> = Vec::new();
    let mut cand_x: Vec<i32> = Vec::new();
    for y in 0..ph as i32 {
        for x in 0..pw as i32 {
            if cand[(y * pw + x) as usize] != 0 {
                cand_y.push(y);
                cand_x.push(x);
            }
        }
    }
    let nc = cand_y.len();
    if nc == 0 {
        // nothing to sample from: leave holes for the residual TELEA pass below.
        let mut res = vec![0f32; n * 3];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let si = (((y + PM_PADM) * pw + (x + PM_PADM)) * 3) as usize;
                let di = ((y * w as i32 + x) * 3) as usize;
                res[di] = work[si];
                res[di + 1] = work[si + 1];
                res[di + 2] = work[si + 2];
            }
        }
        let mut hm = vec![0u8; n];
        let mut any = false;
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let i = ((y + PM_PADM) * pw + (x + PM_PADM)) as usize;
                if hole[i] != 0 {
                    hm[(y * w as i32 + x) as usize] = 255;
                    any = true;
                }
            }
        }
        if any {
            let filled = crate::telea::telea_inpaint(&res, &hm, h, w, 3);
            let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * 3) };
            for i in 0..n * 3 {
                out[i] = pm_clamp(filled[i]);
            }
        } else {
            pm_write_out(&work, h, w, pw, out_ptr);
        }
        return;
    }

    // data term: Sobel on grayscale, Dmap = dilate_max3x3(grad*known)
    let mut gray = vec![0f32; pn];
    for i in 0..pn {
        let r = work[i * 3] as i32;
        let g = work[i * 3 + 1] as i32;
        let b = work[i * 3 + 2] as i32;
        let v = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
        gray[i] = if v < 0 { 0 } else if v > 255 { 255 } else { v } as f32;
    }
    let mut gx = vec![0f32; pn];
    let mut gy = vec![0f32; pn];
    pm_sobel(&gray, ph, pw, &mut gx, &mut gy);
    let mut grad = vec![0f32; pn];
    for i in 0..pn {
        grad[i] = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
    }
    // numpy 参照: Dmap = cv2.dilate(grad*known) 经 _shared_core.dilate shim,
    // 输入先被 astype(np.uint8) 截断回绕(C 风格: 截断到整数再 mod 256), 再做
    // 3x3 max。Rust 侧必须复刻同一量化, 否则优先级与排序顺序分歧。
    let mut gk = vec![0f32; pn];
    for i in 0..pn {
        let v = grad[i] * (known[i] as f32) / 255.0;
        gk[i] = ((v as i32) & 0xFF) as f32;
    }
    let mut dmap = vec![0f32; pn];
    pm_dilate_max3x3(&gk, &mut dmap, ph, pw);

    let mut rng = seed;
    let k = (256.min(32.max(nc / 4))) as usize;
    // `direction_deg == -1.0` is the "no direction" sentinel (Python passes
    // `direction=None` => binding sends -1.0). Only a real (>=0) angle enables
    // directional sampling. `use_dir` is already computed above (for the pre-check).
    let dir_vec = if use_dir {
        let rad = direction_deg as f64 * std::f64::consts::PI / 180.0;
        let maxd = ((ph * ph + pw * pw) as f64).sqrt() as i32 + 1;
        Some((rad.cos() as f32, rad.sin() as f32, maxd, 2))
    } else {
        None
    };

    // NNF
    let mut nnf_y = vec![0i32; pn];
    let mut nnf_x = vec![0i32; pn];
    let mut nnf_set = vec![0u8; pn];

    // sample region as an optional slice (None when no sample mask was supplied)
    let spad_ref: Option<&[u8]> = if spad.is_empty() { None } else { Some(&spad) };

    let chunk = 512usize;
    loop {
        // boundary = hole & ~erode3x3(hole) — recomputed once per PASS, exactly
        // like the Python `while True:` iteration (not per 512-pixel chunk).
        let mut eroded = vec![0u8; pn];
        pm_erode3x3(&hole, &mut eroded, ph, pw);
        let mut by: Vec<i32> = Vec::new();
        let mut bx: Vec<i32> = Vec::new();
        for i in 0..pn as i32 {
            if hole[i as usize] != 0 && eroded[i as usize] == 0 {
                by.push(i / pw);
                bx.push(i % pw);
            }
        }
        if by.is_empty() {
            break;
        }

        // confidence map (box mean over known) — Python computes
        // `cv2.boxFilter(known,(P,P))` ONCE per pass; all boundary pixels of the
        // pass share it (later chunks see updated known via _copy_patch only).
        let known_f: Vec<f32> = known.iter().map(|&v| v as f32 / 255.0).collect();
        let mut cmap = vec![0f32; pn];
        pm_box_mean(&known_f, &mut cmap, ph, pw, p);

        if use_dir {
            // direction mode: one highest-priority pixel per pass (Python dir loop)
            let mut best = -1e30f32;
            let mut bi = -1i32;
            for kk in 0..by.len() as i32 {
                let i = (by[kk as usize] * pw + bx[kk as usize]) as usize;
                let pr = cmap[i] * dmap[i];
                if pr > best {
                    best = pr;
                    bi = kk;
                }
            }
            if bi < 0 {
                break;
            }
            let ty = by[bi as usize];
            let tx = bx[bi as usize];
            let (sy, sx) = pm_best_source(
                &work, &known, spad_ref, &cand, ph, pw, HALF, PP, PP3, ty, tx,
                &cand_y, &cand_x, nc, k, &mut rng, true, dir_vec,
                &nnf_y, &nnf_x, &nnf_set,
            );
            pm_copy_patch(
                &mut work, &mut known, &mut hole, &mut nnf_y, &mut nnf_x, &mut nnf_set,
                &orig_known, ph, pw, HALF, PP, PP3, ty, tx, sy, sx,
            );
        } else {
            // Python fast path: priority sort covers the WHOLE boundary of this
            // pass, then ALL of it is processed in 512-pixel chunks. Within a
            // chunk every best_source reads the state as of the chunk start
            // (Python gathers `filled`/nnf for the whole batch before any
            // _copy_patch), then the copies are applied sequentially — later
            // chunks see the earlier chunks' writes, but the priority/boundary
            // lists stay stale until the next pass, exactly like Python.
            let mut prio: Vec<f32> = Vec::with_capacity(by.len());
            for kk in 0..by.len() {
                let i = (by[kk] * pw + bx[kk]) as usize;
                prio.push(cmap[i] * dmap[i]);
            }
            let mut order: Vec<usize> = (0..by.len()).collect();
            order.sort_by(|&a, &b| prio[b].partial_cmp(&prio[a]).unwrap_or(std::cmp::Ordering::Equal));
            let mut c0 = 0usize;
            while c0 < order.len() {
                let c1 = (c0 + chunk).min(order.len());
                let mut srcs: Vec<(i32, i32)> = Vec::with_capacity(c1 - c0);
                for &kk in &order[c0..c1] {
                    let ty = by[kk];
                    let tx = bx[kk];
                    srcs.push(pm_best_source(
                        &work, &known, spad_ref, &cand, ph, pw, HALF, PP, PP3, ty, tx,
                        &cand_y, &cand_x, nc, k, &mut rng, false, dir_vec,
                        &nnf_y, &nnf_x, &nnf_set,
                    ));
                }
                for (off, &kk) in order[c0..c1].iter().enumerate() {
                    let ty = by[kk];
                    let tx = bx[kk];
                    let (sy, sx) = srcs[off];
                    pm_copy_patch(
                        &mut work, &mut known, &mut hole, &mut nnf_y, &mut nnf_x, &mut nnf_set,
                        &orig_known, ph, pw, HALF, PP, PP3, ty, tx, sy, sx,
                    );
                }
                c0 = c1;
            }
        }
    }

    // residual holes: OpenCV TELEA (matches the Python cv2.inpaint TELEA fallback)
    let mut res = vec![0f32; n * 3];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let si = (((y + PM_PADM) * pw + (x + PM_PADM)) * 3) as usize;
            let di = ((y * w as i32 + x) * 3) as usize;
            res[di] = work[si];
            res[di + 1] = work[si + 1];
            res[di + 2] = work[si + 2];
        }
    }
    let mut hm = vec![0u8; n];
    let mut any = false;
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let i = ((y + PM_PADM) * pw + (x + PM_PADM)) as usize;
            if hole[i] != 0 {
                hm[(y * w as i32 + x) as usize] = 255;
                any = true;
            }
        }
    }
    if any {
        let filled = crate::telea::telea_inpaint(&res, &hm, h, w, 3);
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * 3) };
        for i in 0..n * 3 {
            out[i] = pm_clamp(filled[i]);
        }
    } else {
        pm_write_out(&work, h, w, pw, out_ptr);
    }
}

#[cfg(test)]
mod rng_tests {
    #[test]
    fn mulberry32_stream() {
        let mut s: u32 = 0;
        let mut out = Vec::new();
        for _ in 0..4 {
            s = s.wrapping_add(0x6D2B79F5);
            let mut t = s;
            t = (t ^ (t >> 15)).wrapping_mul(t | 1);
            t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t));
            out.push(t ^ (t >> 14));
        }
        println!("RUST stream: {:?}", out);
    }
}
