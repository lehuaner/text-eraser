//! textcore — shared algorithm core compiled to WebAssembly.
//!
//! Single source of truth consumed by BOTH the browser (via WebAssembly)
//! and the Python backend (via wasmtime). C-ABI exports only; no wasm-bindgen
//! glue, so any WASM host can call them uniformly.
//!
//! Memory contract: callers allocate flat byte buffers through `alloc` and
//! free them with `dealloc(ptr, size)`. Pointers are linear-memory offsets.

const INF: f64 = 1e20;

/// PatchMatch inpainting — single source of truth shared by browser & backend.
mod patchmatch;

/// Text-mask synthesis (textMask + edge -> fill/sample masks) — single source of
/// truth shared by browser & backend.
mod masksynth;

/// De-glow (`_deglow_full_green_v2`) — single source of truth shared by browser
/// & backend. Pure-Rust port, no cv2.
mod deglow;

/// OpenCV `INPAINT_TELEA` faithful port (smooth-gradient fallback fill).
mod telea;

// Re-export the C-API entry points at the crate root so integration tests
// (cargo test --target <native>) can call them directly with raw pointers.
// Does NOT affect the wasm symbol names (`#[no_mangle]` controls those).
pub use deglow::deglow_full_green_v2;
pub use deglow::erase_text_glyphs;
pub use patchmatch::patchmatch_inpaint;

/// Allocate `size` bytes in the module's linear memory. Returns a pointer
/// (linear-memory offset). Free with `dealloc(ptr, size)`.
#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let layout = match std::alloc::Layout::from_size_align(size, 4) {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };
    unsafe { std::alloc::alloc(layout) }
}

/// Free a buffer previously returned by `alloc`. `size` must match the value
/// passed to `alloc`.
#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    if let Ok(layout) = std::alloc::Layout::from_size_align(size, 4) {
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

/// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher 1D/2D EDT).
///
/// `mask` is an H*W u8 array. Non-zero entries are SOURCE pixels (distance 0);
/// zero entries are seeds at distance INF. This mirrors
/// `cv2.distanceTransform((cur == 0), DIST_L2)` where `cur` is the text mask,
/// i.e. distance to the nearest TEXT pixel.
///
/// Writes H*W f32 (sqrt of squared distance) into `out`.
#[no_mangle]
pub extern "C" fn distance_transform_edt(
    mask_ptr: *const u8,
    h: i32,
    w: i32,
    out_ptr: *mut f32,
) {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n) };
    let mut f = vec![INF; n];
    for i in 0..n {
        if mask[i] != 0 {
            f[i] = 0.0;
        }
    }
    edt2d(&mut f, h, w);
    for i in 0..n {
        out[i] = f[i].sqrt() as f32;
    }
}

fn edt1d(f: &[f64], d: &mut [f64], z: &mut [f64], v: &mut [usize], n: usize) {
    let mut k: usize = 0;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = ((f[q] + (q as f64) * (q as f64))
            - (f[v[k]] + (v[k] as f64) * (v[k] as f64)))
            / (2.0 * (q as f64 - v[k] as f64));
        while s <= z[k] {
            k -= 1;
            s = ((f[q] + (q as f64) * (q as f64))
                - (f[v[k]] + (v[k] as f64) * (v[k] as f64)))
                / (2.0 * (q as f64 - v[k] as f64));
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < (q as f64) {
            k += 1;
        }
        let dq = (q as f64) - (v[k] as f64);
        d[q] = dq * dq + f[v[k]];
    }
}

fn edt2d(f: &mut [f64], h: usize, w: usize) {
    let m = h.max(w);
    let mut d = vec![0f64; m];
    let mut z = vec![0f64; m + 1];
    let mut v = vec![0usize; m + 1];
    // Horizontal pass: each row independently.
    for y in 0..h {
        let base = y * w;
        edt1d(
            &f[base..base + w],
            &mut d[..w],
            &mut z[..w + 1],
            &mut v[..w + 1],
            w,
        );
        for x in 0..w {
            f[base + x] = d[x];
        }
    }
    // Vertical pass: each column independently.
    let mut col = vec![0f64; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = f[y * w + x];
        }
        edt1d(&col[..h], &mut d[..h], &mut z[..h + 1], &mut v[..h + 1], h);
        for y in 0..h {
            f[y * w + x] = d[y];
        }
    }
}

// ===========================================================================
// RGB -> GRAY (cv2.cvtColor RGB2GRAY fixed-point).
// rgb: H*W*3 f32 (values 0..255, truncated toward zero like JS `|0`). out: H*W u8.
#[no_mangle]
pub extern "C" fn rgb_to_gray(rgb_ptr: *const f32, h: i32, w: i32, out_ptr: *mut u8) {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    let rgb = unsafe { std::slice::from_raw_parts(rgb_ptr, n * 3) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n) };
    for i in 0..n {
        let r = rgb[i * 3] as i32;
        let g = rgb[i * 3 + 1] as i32;
        let b = rgb[i * 3 + 2] as i32;
        let v = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
        out[i] = if v < 0 { 0 } else if v > 255 { 255 } else { v as u8 };
    }
}

// ===========================================================================
// Otsu threshold (cv2.threshold THRESH_OTSU). src: H*W u8, out: H*W u8 (0/255).
// Returns the chosen threshold (0..255). First maximal between-class variance wins
// ties (strict `>`), mirroring the pure-JS implementation.
#[no_mangle]
pub extern "C" fn threshold_otsu(src_ptr: *const u8, out_ptr: *mut u8, n: i32) -> f32 {
    let n = n as usize;
    let src = unsafe { std::slice::from_raw_parts(src_ptr, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n) };
    let mut hist = [0i64; 256];
    let mut sum = 0i64;
    for &v in src.iter() {
        hist[v as usize] += 1;
        sum += v as i64;
    }
    let mut sum_b = 0i64;
    let mut w_b = 0i64;
    let mut max_between = -1.0f64;
    let mut thr = 0i32;
    for t in 0..256 {
        w_b += hist[t];
        if w_b == 0 {
            continue;
        }
        let w_f = n as i64 - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += (t as i64) * hist[t];
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum - sum_b) as f64 / w_f as f64;
        let between = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if between > max_between {
            max_between = between;
            thr = t as i32;
        }
    }
    for i in 0..n {
        out[i] = if src[i] as i32 > thr { 255 } else { 0 };
    }
    thr as f32
}

// ===========================================================================
// Binary morphology on a 0/1 (or 0/255) u8 mask using an EXPLICIT kernel bitmap.
// `kern_ptr` is a kh*kw u8 array (1 = include this cell, 0 = skip). The caller
// supplies the exact cv2/opencv.js structuring element, so the precise (rasterized)
// ELLIPSE shape is honored on BOTH ends — no disk approximation.
// op: 0 = ERODE, 1 = DILATE. Anchor = (kh/2, kw/2). Out-of-bounds neighbors are
// treated as 0 (BORDER_CONSTANT), matching cv2.dilate/erode defaults.
#[no_mangle]
pub extern "C" fn morphology(
    mask_ptr: *const u8,
    out_ptr: *mut u8,
    h: i32,
    w: i32,
    kern_ptr: *const u8,
    kh: i32,
    kw: i32,
    op: i32,
) {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n) };
    let kh = kh as usize;
    let kw = kw as usize;
    let kern = unsafe { std::slice::from_raw_parts(kern_ptr, kh * kw) };
    let ax = (kw / 2) as i32; // anchor x = floor(kw/2)
    let ay = (kh / 2) as i32; // anchor y = floor(kh/2)
    let is_dilate = op != 0;
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut res = if is_dilate { false } else { true };
            'kern: for ky in 0..kh as i32 {
                for kx in 0..kw as i32 {
                    if kern[(ky as usize) * kw + (kx as usize)] == 0 {
                        continue;
                    }
                    let nx = x + (kx - ax);
                    let ny = y + (ky - ay);
                    // Out-of-bounds kernel cells are CLIPPED (not applied), matching
                    // cv2.erode/dilate — cv2 does not treat them as 0 (BORDER_CONSTANT);
                    // it simply ignores kernel elements that fall outside the image.
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let bit = mask[(ny as usize) * w + (nx as usize)] != 0;
                    if is_dilate {
                        if bit {
                            res = true;
                            break 'kern;
                        }
                    } else if !bit {
                        res = false;
                        break 'kern;
                    }
                }
            }
            out[(y as usize) * w + (x as usize)] = if res { 1 } else { 0 };
        }
    }
}

// ===========================================================================
// Connected components (8-connectivity) on a 0/255 u8 mask.
// Labels written to labels_ptr (H*W i32); label 0 = background placeholder.
// Returns the total label count INCLUDING the background placeholder (cv convention).
#[no_mangle]
pub extern "C" fn connected_components(
    mask_ptr: *const u8,
    labels_ptr: *mut i32,
    h: i32,
    w: i32,
) -> i32 {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, n) };
    let labels = unsafe { std::slice::from_raw_parts_mut(labels_ptr, n) };
    for i in 0..n {
        labels[i] = 0;
    }
    let dx = [-1i32, -1, -1, 0, 0, 1, 1, 1];
    let dy = [-1i32, 0, 1, -1, 1, -1, 0, 1];
    let mut queue = vec![0i32; n];
    let mut ncomp = 1i32; // background placeholder is label 0
    for s in 0..n as i32 {
        if mask[s as usize] == 0 || labels[s as usize] != 0 {
            continue;
        }
        let comp = ncomp;
        ncomp += 1;
        labels[s as usize] = comp;
        let mut head = 0i32;
        let mut tail = 0i32;
        queue[tail as usize] = s;
        tail += 1;
        while head < tail {
            let p = queue[head as usize];
            head += 1;
            let py = p / w as i32;
            let px = p - py * w as i32;
            for k in 0..8 {
                let nx = px + dx[k];
                let ny = py + dy[k];
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let np = ny * w as i32 + nx;
                if mask[np as usize] != 0 && labels[np as usize] == 0 {
                    labels[np as usize] = comp;
                    queue[tail as usize] = np;
                    tail += 1;
                }
            }
        }
    }
    ncomp
}

// ===========================================================================
// Per-component stats (left, top, width, height, area) as 5*i32 blocks.
// labels_ptr: H*W i32 (from connected_components). stats_ptr: n_labels*5 i32.
// stats[0] (background) is all zeros.
#[no_mangle]
pub extern "C" fn connected_components_stats(
    labels_ptr: *const i32,
    stats_ptr: *mut i32,
    h: i32,
    w: i32,
    n_labels: i32,
) {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    let labels = unsafe { std::slice::from_raw_parts(labels_ptr, n) };
    let stats = unsafe { std::slice::from_raw_parts_mut(stats_ptr, (n_labels as usize) * 5) };
    for i in 0..(n_labels as usize) * 5 {
        stats[i] = 0;
    }
    let mut minx = vec![w as i32; n_labels as usize];
    let mut miny = vec![h as i32; n_labels as usize];
    let mut maxx = vec![-1i32; n_labels as usize];
    let mut maxy = vec![-1i32; n_labels as usize];
    let mut area = vec![0i32; n_labels as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let lab = labels[(y as usize) * w + (x as usize)] as usize;
            if lab == 0 {
                continue;
            }
            if x < minx[lab] {
                minx[lab] = x;
            }
            if x > maxx[lab] {
                maxx[lab] = x;
            }
            if y < miny[lab] {
                miny[lab] = y;
            }
            if y > maxy[lab] {
                maxy[lab] = y;
            }
            area[lab] += 1;
        }
    }
    for lab in 0..(n_labels as usize) {
        let base = lab * 5;
        if lab == 0 || maxx[lab] < 0 {
            continue; // already zeroed
        }
        stats[base] = minx[lab];
        stats[base + 1] = miny[lab];
        stats[base + 2] = maxx[lab] - minx[lab] + 1;
        stats[base + 3] = maxy[lab] - miny[lab] + 1;
        stats[base + 4] = area[lab];
    }
}

// ===========================================================================
// INTER_CUBIC (a=-0.75) separable upscale of a U8 single-channel array.
// Mirrors cv-bridge resizeGrayU8 exactly (rounded, clamped to 0..255).
#[no_mangle]
pub extern "C" fn resize_gray_cubic(
    src_ptr: *const u8,
    out_ptr: *mut u8,
    h: i32,
    w: i32,
    h2: i32,
    w2: i32,
) {
    let h = h as usize;
    let w = w as usize;
    let h2 = h2 as usize;
    let w2 = w2 as usize;
    let src = unsafe { std::slice::from_raw_parts(src_ptr, h * w) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, h2 * w2) };
    let a = -0.75f64;
    let cubic = |x: f64| -> f64 {
        let x = x.abs();
        if x <= 1.0 {
            (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
        } else if x < 2.0 {
            a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
        } else {
            0.0
        }
    };
    let mut tmp = vec![0f64; h * w2];
    let sx = w as f64 / w2 as f64;
    for y in 0..h {
        for x2 in 0..w2 {
            let center = (x2 as f64 + 0.5) * sx - 0.5;
            let ix = center.floor() as i32;
            let fx = center - ix as f64;
            let w0 = cubic(1.0 + fx);
            let w1 = cubic(fx);
            let w2v = cubic(1.0 - fx);
            let w3 = cubic(2.0 - fx);
            let i0 = clamp_idx(ix - 1, w as i32);
            let i1 = clamp_idx(ix, w as i32);
            let i2 = clamp_idx(ix + 1, w as i32);
            let i3 = clamp_idx(ix + 2, w as i32);
            tmp[y * w2 + x2] = src[y * w + i0] as f64 * w0
                + src[y * w + i1] as f64 * w1
                + src[y * w + i2] as f64 * w2v
                + src[y * w + i3] as f64 * w3;
        }
    }
    let sy = h as f64 / h2 as f64;
    for y2 in 0..h2 {
        let center = (y2 as f64 + 0.5) * sy - 0.5;
        let iy = center.floor() as i32;
        let fy = center - iy as f64;
        let w0 = cubic(1.0 + fy);
        let w1 = cubic(fy);
        let w2v = cubic(1.0 - fy);
        let w3 = cubic(2.0 - fy);
        let j0 = clamp_idx(iy - 1, h as i32);
        let j1 = clamp_idx(iy, h as i32);
        let j2 = clamp_idx(iy + 1, h as i32);
        let j3 = clamp_idx(iy + 2, h as i32);
        for x2 in 0..w2 {
            let s = tmp[j0 * w2 + x2] * w0
                + tmp[j1 * w2 + x2] * w1
                + tmp[j2 * w2 + x2] * w2v
                + tmp[j3 * w2 + x2] * w3;
            let v = if s < 0.0 {
                0i32
            } else if s > 255.0 {
                255i32
            } else {
                (s + 0.5).floor() as i32
            };
            out[y2 * w2 + x2] = v as u8;
        }
    }
}

// ===========================================================================
// INTER_LINEAR (bilinear) resize of a Float32Array. Mirrors cv-bridge resizeFloat.
#[no_mangle]
pub extern "C" fn resize_float_linear(
    src_ptr: *const f32,
    out_ptr: *mut f32,
    h: i32,
    w: i32,
    h2: i32,
    w2: i32,
) {
    let h = h as usize;
    let w = w as usize;
    let h2 = h2 as usize;
    let w2 = w2 as usize;
    let src = unsafe { std::slice::from_raw_parts(src_ptr, h * w) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, h2 * w2) };
    let sx = w as f64 / w2 as f64;
    let sy = h as f64 / h2 as f64;
    for y2 in 0..h2 {
        let cy = (y2 as f64 + 0.5) * sy - 0.5;
        let iy = cy.floor() as i32;
        let fy = cy - iy as f64;
        let y0 = clamp_idx(iy, h as i32);
        let y1 = clamp_idx(iy + 1, h as i32);
        for x2 in 0..w2 {
            let cx = (x2 as f64 + 0.5) * sx - 0.5;
            let ix = cx.floor() as i32;
            let fx = cx - ix as f64;
            let x0 = clamp_idx(ix, w as i32);
            let x1 = clamp_idx(ix + 1, w as i32);
            let v00 = src[y0 * w + x0] as f64;
            let v01 = src[y0 * w + x1] as f64;
            let v10 = src[y1 * w + x0] as f64;
            let v11 = src[y1 * w + x1] as f64;
            let top = v00 * (1.0 - fx) + v01 * fx;
            let bot = v10 * (1.0 - fx) + v11 * fx;
            out[y2 * w2 + x2] = (top * (1.0 - fy) + bot * fy) as f32;
        }
    }
}

#[inline]
fn clamp_idx(i: i32, len: i32) -> usize {
    if i < 0 {
        0
    } else if i >= len {
        (len - 1) as usize
    } else {
        i as usize
    }
}
