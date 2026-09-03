//! De-glow (`_deglow_full_green_v2`) — single source of truth shared by browser & backend.
//!
//! Faithfully ported from `text_eraser/text_select.py::_deglow_full_green_v2`. All
//! sub-primitives are pure-Rust (no cv2), so the browser Worker and the Python
//! backend produce byte-identical de-glowed images from the same wasm bytes.
//!
//! Algorithm (see the Python docstring for full rationale):
//!   1. detect strong green (connected-component max-area gate)
//!   2. grow the glow zone along green|bright|faint_green (budget + zone_expand)
//!   3. subtract greenness from the G channel (G - max(R,B)) -> neutral grey,
//!      with warm-background compensation (local R-G field) when needed
//!   4. reconstruct the non-text halo (zone minus text protect) from a geodesic /
//!      harmonic background colour field + high-frequency detail, with optional
//!      chroma-keep and a greenness-weighted soft mix
//!   5. the text-stroke (high-alpha) core becomes the fill mask
//!
//! Note: the background fields use resize/gaussian that differ numerically from
//! cv2's, but that is irrelevant for parity — both ends call THIS implementation,
//! so they agree by construction.

// ---- small helpers ----------------------------------------------------------

#[inline]
fn clamp_idx(i: i32, len: i32) -> i32 {
    if i < 0 { 0 } else if i >= len { len - 1 } else { i }
}

#[inline]
fn clamp_f(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

/// cv2 saturate_cast<uchar>(float) — round half to EVEN (SSE cvtss2si semantics).
#[inline]
fn cv_round(v: f32) -> f32 {
    let f = v.floor();
    let frac = v - f;
    if frac > 0.5 { f + 1.0 }
    else if frac < 0.5 { f }
    else if (f as i64) % 2 == 0 { f } else { f + 1.0 }
}

/// cv2.getStructuringElement(MORPH_ELLIPSE,(ksize,ksize)) bitmap (1=include).
/// cv2.getStructuringElement(MORPH_ELLIPSE, (k,k)) — OpenCV rasterizes the ellipse
/// as a per-row scanline: half-width at row `dy` is `r*sqrt(1-(dy/r)^2)` and the row is
/// filled from `round(cx-dx)` to `round(cx+dx)`. This matches cv2 exactly (a plain
/// center-in-ellipse test `(ox^2+oy^2<=r^2)` EXCLUDES the rounded-out corner pixels and
/// produces a smaller kernel — which is what caused the 45px zone-expand gap).
fn ellipse_kernel(ksize: i32) -> Vec<u8> {
    let n = (ksize * ksize) as usize;
    let mut k = vec![0u8; n];
    if ksize <= 0 { return k; }
    let c = (ksize as f64 - 1.0) / 2.0; // center
    let r = (ksize as f64 - 1.0) / 2.0; // half-axis
    for dy in 0..ksize {
        let oy = (dy as f64 - c).abs();
        if oy > r + 1e-9 { continue; }
        let dx = r * (1.0 - (oy / r) * (oy / r)).max(0.0).sqrt();
        let left = (c - dx).round() as i32;
        let right = (c + dx).round() as i32;
        for x in left..=right {
            if x >= 0 && x < ksize {
                k[(dy * ksize + x) as usize] = 1;
            }
        }
    }
    k
}

/// cv2.cvtColor(RGB2GRAY) fixed-point, returned as f32 (matches cv2 .astype(f32)).
pub(crate) fn gray_f32(rgb: &[f32], n: usize) -> Vec<f32> {
    let mut g = vec![0f32; n];
    for i in 0..n {
        let r = rgb[i * 3] as i32;
        let gg = rgb[i * 3 + 1] as i32;
        let b = rgb[i * 3 + 2] as i32;
        let v = (r * 4899 + gg * 9617 + b * 1868 + 8192) >> 14;
        g[i] = (if v < 0 { 0 } else if v > 255 { 255 } else { v }) as f32;
    }
    g
}

fn dilate_ellipse(mask: &[u8], h: usize, w: usize, ksize: i32) -> Vec<u8> {
    let k = ellipse_kernel(ksize);
    let kh = ksize as usize;
    let kw = ksize as usize;
    let ax = ksize / 2;
    let ay = ksize / 2;
    let mut out = vec![0u8; h * w];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut hit = false;
            'k: for ky in 0..kh as i32 {
                for kx in 0..kw as i32 {
                    if k[(ky * kw as i32 + kx) as usize] == 0 { continue; }
                    let nx = x + (kx - ax);
                    let ny = y + (ky - ay);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                    if mask[ny as usize * w + nx as usize] != 0 { hit = true; break 'k; }
                }
            }
            out[y as usize * w + x as usize] = if hit { 1 } else { 0 };
        }
    }
    out
}

fn erode_ellipse(mask: &[u8], h: usize, w: usize, ksize: i32) -> Vec<u8> {
    let k = ellipse_kernel(ksize);
    let kh = ksize as usize;
    let kw = ksize as usize;
    let ax = ksize / 2;
    let ay = ksize / 2;
    let mut out = vec![0u8; h * w];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut all = true;
            'k: for ky in 0..kh as i32 {
                for kx in 0..kw as i32 {
                    if k[(ky * kw as i32 + kx) as usize] == 0 { continue; }
                    let nx = x + (kx - ax);
                    let ny = y + (ky - ay);
                    // cv2 erode treats out-of-bounds as background (0) -> erodes edge.
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { all = false; break 'k; }
                    if mask[ny as usize * w + nx as usize] == 0 { all = false; break 'k; }
                }
            }
            out[y as usize * w + x as usize] = if all { 1 } else { 0 };
        }
    }
    out
}

// ---- gaussian blur (separable) ---------------------------------------------

fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as i32;
    let ksize = (2 * radius + 1) as usize;
    let mut k = vec![0f64; ksize];
    let mut s = 0.0;
    for i in 0..ksize {
        let x = (i as i32 - radius) as f64;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        k[i] = v;
        s += v;
    }
    for v in k.iter_mut() { *v /= s; }
    k
}

fn gaussian_blur(src: &[f32], h: usize, w: usize, sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 { return src.to_vec(); }
    let k = gaussian_kernel(sigma);
    let radius = (k.len() / 2) as i32;
    let mut tmp = vec![0f64; h * w];
    // horizontal
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut acc = 0.0;
            for ki in 0..k.len() {
                let kx = x + (ki as i32 - radius);
                let kk = clamp_idx(kx, w as i32) as usize;
                acc += src[y as usize * w + kk] as f64 * k[ki];
            }
            tmp[y as usize * w + x as usize] = acc;
        }
    }
    let mut out = vec![0f32; h * w];
    // vertical
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut acc = 0.0;
            for ki in 0..k.len() {
                let ky = y + (ki as i32 - radius);
                let kk = clamp_idx(ky, h as i32) as usize;
                acc += tmp[kk * w + x as usize] as f64 * k[ki];
            }
            out[y as usize * w + x as usize] = acc as f32;
        }
    }
    out
}

// ---- sobel (f32) -----------------------------------------------------------

fn sobel(src: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0f32; h * w];
    let mut gy = vec![0f32; h * w];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let xa = [clamp_idx(x - 1, w as i32), x, clamp_idx(x + 1, w as i32)];
            let ya = [clamp_idx(y - 1, h as i32), y, clamp_idx(y + 1, h as i32)];
            let tl = src[(ya[0] * w as i32 + xa[0]) as usize];
            let tc = src[(ya[0] * w as i32 + xa[1]) as usize];
            let tr = src[(ya[0] * w as i32 + xa[2]) as usize];
            let ml = src[(ya[1] * w as i32 + xa[0]) as usize];
            let mr = src[(ya[1] * w as i32 + xa[2]) as usize];
            let bl = src[(ya[2] * w as i32 + xa[0]) as usize];
            let bc = src[(ya[2] * w as i32 + xa[1]) as usize];
            let br = src[(ya[2] * w as i32 + xa[2]) as usize];
            let o = (y * w as i32 + x) as usize;
            gx[o] = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
            gy[o] = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
        }
    }
    (gx, gy)
}

// ---- percentile (f32, linear interp) --------------------------------------

fn percentile(arr: &[f32], q: f32) -> f32 {
    if arr.is_empty() { return 0.0; }
    let mut v: Vec<f32> = arr.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len() as f64;
    let mut idx = (q as f64) * (n - 1.0);
    if idx < 0.0 { idx = 0.0; }
    if idx > n - 1.0 { idx = n - 1.0; }
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { return v[lo]; }
    let frac = idx - lo as f64;
    v[lo] * (1.0 - frac as f32) + v[hi] * frac as f32
}

// ---- box mean (f32) --------------------------------------------------------

fn box_mean(src: &[f32], h: usize, w: usize, p: i32) -> Vec<f32> {
    let half = p / 2;
    let mut out = vec![0f32; h * w];
    let sum_p = ((h + 1) * (w + 1)) as usize;
    let mut s = vec![0f64; sum_p];
    for y in 0..h {
        let mut row = 0f64;
        for x in 0..w {
            row += src[y * w + x] as f64;
            s[(y + 1) * (w + 1) + (x + 1)] = s[y * (w + 1) + (x + 1)] + row;
        }
    }
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let y0 = (y - half).max(0) as usize;
            let y1 = ((y + half).min(h as i32 - 1)) as usize;
            let x0 = (x - half).max(0) as usize;
            let x1 = ((x + half).min(w as i32 - 1)) as usize;
            let sum = s[(y1 + 1) * (w + 1) + (x1 + 1)]
                - s[(y1 + 1) * (w + 1) + x0]
                - s[y0 * (w + 1) + (x1 + 1)]
                + s[y0 * (w + 1) + x0];
            let cnt = ((y1 - y0 + 1) * (x1 - x0 + 1)) as f32;
            out[y as usize * w + x as usize] = (sum as f32) / cnt;
        }
    }
    out
}

// ---- bilinear resize of a 3-channel f32 volume -----------------------------

fn resize_rgb_linear(src: &[f32], h: usize, w: usize, h2: usize, w2: usize) -> Vec<f32> {
    let mut out = vec![0f32; h2 * w2 * 3];
    let sx = w as f64 / w2 as f64;
    let sy = h as f64 / h2 as f64;
    for c in 0..3 {
        for y2 in 0..h2 {
            let cy = (y2 as f64 + 0.5) * sy - 0.5;
            let iy = cy.floor() as i32;
            let fy = cy - iy as f64;
            let j0 = clamp_idx(iy, h as i32) as usize;
            let j1 = clamp_idx(iy + 1, h as i32) as usize;
            for x2 in 0..w2 {
                let cx = (x2 as f64 + 0.5) * sx - 0.5;
                let ix = cx.floor() as i32;
                let fx = cx - ix as f64;
                let i0 = clamp_idx(ix, w as i32) as usize;
                let i1 = clamp_idx(ix + 1, w as i32) as usize;
                let v00 = src[(j0 * w + i0) * 3 + c] as f64;
                let v01 = src[(j0 * w + i1) * 3 + c] as f64;
                let v10 = src[(j1 * w + i0) * 3 + c] as f64;
                let v11 = src[(j1 * w + i1) * 3 + c] as f64;
                let top = v00 * (1.0 - fx) + v01 * fx;
                let bot = v10 * (1.0 - fx) + v11 * fx;
                out[(y2 * w2 + x2) * 3 + c] = (top * (1.0 - fy) + bot * fy) as f32;
            }
        }
    }
    out
}

// ---- geodesic background colour field (Dijkstra, multi-source) -------------

/// Min-heap key mirroring Python's `(nd, y, x)` tuple order: compare the f32
/// distance first, then the node index (y*W+x == (y,x) lexicographic) so equal
/// distances pop in the same order as numpy/heapq.
#[derive(Clone, Copy)]
struct HeapItem {
    dist: f32,
    node: usize,
}
impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist && self.node == other.node }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist).then(self.node.cmp(&other.node))
    }
}

fn geodesic_sources(lum: &[f32], h2: usize, w2: usize, src_mask: &[u8]) -> (Vec<i32>, Vec<i32>) {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    let n = h2 * w2;
    // Python runs this Dijkstra in float32 (dist = np.full(..., np.float32) and every
    // `nd = d + 1.0 + 3.0*|dlum|` rounds at f32) — keep f32 or routing flips.
    let mut dist = vec![f32::INFINITY; n];
    let mut src_y = vec![0i32; n];
    let mut src_x = vec![0i32; n];
    let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
    for i in 0..n {
        if src_mask[i] != 0 {
            dist[i] = 0.0;
            // Mirror Python `_geodesic_sources`: seed each source's OWN
            // coordinates. Without this, src_y/src_x stay 0 and (0,0) gets
            // propagated to every reachable pixel — collapsing the whole field
            // to the top-left source. (Was the cause of B/D_rg/D_gb divergence.)
            src_y[i] = (i / w2) as i32;
            src_x[i] = (i % w2) as i32;
            heap.push(Reverse(HeapItem { dist: 0.0, node: i }));
        }
    }
    let idx = |y: i32, x: i32| (y as usize) * w2 + (x as usize);
    while let Some(Reverse(item)) = heap.pop() {
        let d = item.dist;
        let i = item.node;
        if d > dist[i] { continue; }
        let y = (i / w2) as i32;
        let x = (i % w2) as i32;
        let sy = src_y[i];
        let sx = src_x[i];
        let lv = lum[i];
        for (dy, dx) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let ny = y + dy;
            let nx = x + dx;
            if ny < 0 || ny >= h2 as i32 || nx < 0 || nx >= w2 as i32 { continue; }
            let ni = idx(ny, nx);
            let nd = d + 1.0 + 3.0 * (lum[ni] - lv).abs();
            if nd < dist[ni] {
                dist[ni] = nd;
                src_y[ni] = sy;
                src_x[ni] = sx;
                heap.push(Reverse(HeapItem { dist: nd, node: ni }));
            }
        }
    }
    (src_y, src_x)
}

/// Returns B (h*w*3 f32) and optionally extras (each h*w f32) propagated with the
/// same source map. `extra_src` (if any) defines a separate source set for extras.
fn geodesic_background(
    rgb: &[f32], h: usize, w: usize, zone: &[u8],
    extra: &[Vec<f32>], extra_src: Option<&[u8]>,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let scale = if (h.min(w) as i32) >= 160 { 4 } else { 2 };
    let h2 = (h / scale).max(2);
    let w2 = (w / scale).max(2);
    let gray = gray_f32(rgb, h * w);
    let lum = resize_area(&gray, h, w, h2, w2, 1);
    let rz = resize_mask(zone, h, w, h2, w2);
    let rgb_s = resize_area(rgb, h, w, h2, w2, 3);
    // Sources = the BACKGROUND (outside the zone). Matches cv2 `_geodesic_background`,
    // which sets `src_mask = ~rz`. The geodesic field then assigns each zone pixel the
    // colour of the nearest background source — i.e. the reconstructed underlying texture.
    // (The previous port used `rz` directly, inverting the field and smearing glow colours
    // outward into the background.)
    let src_mask = rz.iter().map(|&v| if v == 0 { 1u8 } else { 0 }).collect::<Vec<_>>();
    let (sy, sx) = geodesic_sources(&lum, h2, w2, &src_mask);
    let mut b = vec![0f32; h2 * w2 * 3];
    for i in 0..h2 * w2 {
        let yy = sy[i] as usize;
        let xx = sx[i] as usize;
        b[i * 3] = rgb_s[(yy * w2 + xx) * 3];
        b[i * 3 + 1] = rgb_s[(yy * w2 + xx) * 3 + 1];
        b[i * 3 + 2] = rgb_s[(yy * w2 + xx) * 3 + 2];
    }
    let b_up = resize_cubic(&b, h2, w2, h, w, 3);
    let b_sm = gaussian_blur_3(&b_up, h, w, 4.0);
    let mut extras_up = Vec::new();
    if extra.is_empty() {
        return (b_sm, extras_up);
    }
    // extras use either the same source map or a dedicated source set
    let (ey, ex) = if let Some(es) = extra_src {
        let es_rz = resize_mask(es, h, w, h2, w2);
        let es_mask = es_rz.iter().map(|&v| if v != 0 { 1u8 } else { 0 }).collect::<Vec<_>>();
        geodesic_sources(&lum, h2, w2, &es_mask)
    } else {
        (sy.clone(), sx.clone())
    };
    for e in extra {
        let e_s = resize_area(e, h, w, h2, w2, 1);
        let mut e_up = vec![0f32; h2 * w2];
        for i in 0..h2 * w2 {
            e_up[i] = e_s[ey[i] as usize * w2 + ex[i] as usize];
        }
        let e_f = resize_cubic(&e_up, h2, w2, h, w, 1);
        extras_up.push(gaussian_blur(&e_f, h, w, 4.0));
    }
    (b_sm, extras_up)
}

fn resize_rgb_linear_as_gray(src: &[f32], h: usize, w: usize, h2: usize, w2: usize) -> Vec<f32> {
    // treat src as a single-channel volume (n values)
    let mut out = vec![0f32; h2 * w2];
    let sx = w as f64 / w2 as f64;
    let sy = h as f64 / h2 as f64;
    for y2 in 0..h2 {
        let cy = (y2 as f64 + 0.5) * sy - 0.5;
        let iy = cy.floor() as i32;
        let fy = cy - iy as f64;
        let j0 = clamp_idx(iy, h as i32) as usize;
        let j1 = clamp_idx(iy + 1, h as i32) as usize;
        for x2 in 0..w2 {
            let cx = (x2 as f64 + 0.5) * sx - 0.5;
            let ix = cx.floor() as i32;
            let fx = cx - ix as f64;
            let i0 = clamp_idx(ix, w as i32) as usize;
            let i1 = clamp_idx(ix + 1, w as i32) as usize;
            let v00 = src[j0 * w + i0] as f64;
            let v01 = src[j0 * w + i1] as f64;
            let v10 = src[j1 * w + i0] as f64;
            let v11 = src[j1 * w + i1] as f64;
            let top = v00 * (1.0 - fx) + v01 * fx;
            let bot = v10 * (1.0 - fx) + v11 * fx;
            out[y2 * w2 + x2] = (top * (1.0 - fy) + bot * fy) as f32;
        }
    }
    out
}

fn resize_mask(src: &[u8], h: usize, w: usize, h2: usize, w2: usize) -> Vec<u8> {
    // cv2.resize(..., INTER_NEAREST): source coord = floor(dst * src/dst), clamped to valid range.
    // Empirically verified against cv2 for exact and non-exact scale factors.
    let sx = w as f64 / w2 as f64;
    let sy = h as f64 / h2 as f64;
    let maxy = (h as i32) - 1;
    let maxx = (w as i32) - 1;
    let mut out = vec![0u8; h2 * w2];
    for y2 in 0..h2 {
        let cyc = ((y2 as f64 * sy).floor() as i32).clamp(0, maxy) as usize;
        for x2 in 0..w2 {
            let cxc = ((x2 as f64 * sx).floor() as i32).clamp(0, maxx) as usize;
            out[y2 * w2 + x2] = if src[cyc * w + cxc] != 0 { 255 } else { 0 };
        }
    }
    out
}

fn gaussian_blur_3(src: &[f32], h: usize, w: usize, sigma: f64) -> Vec<f32> {
    let mut out = vec![0f32; h * w * 3];
    for c in 0..3 {
        let plane: Vec<f32> = (0..h * w).map(|i| src[i * 3 + c]).collect();
        let b = gaussian_blur(&plane, h, w, sigma);
        for i in 0..h * w { out[i * 3 + c] = b[i]; }
    }
    out
}

// ============================================================================
// cv2-compatible resizers (mirror of cv2.resize). The geodesic / harmonic background
// fields in `_geodesic_background` / `_harmonic_background` must match cv2's
// INTER_AREA (downscale) and INTER_CUBIC (upscale) EXACTLY, otherwise the geodesic
// source routing (driven by the downscaled luminance) and the reconstructed field
// diverge from the original Python — that was the remaining 62% pixel gap.
// ============================================================================

/// cv2 INTER_NEAREST border mode (reflect_101), used by resize fallback clamping.
fn reflect101(i: i32, len: i32) -> i32 {
    if len <= 1 { return 0; }
    let period = 2 * (len - 1);
    let mut x = i % period;
    if x < 0 { x += period; }
    if x >= len { x = period - x; }
    x
}

/// cv2 INTER_CUBIC kernel (a = -0.5), the default bicubic used by cv2.resize.
fn cubic_w(t: f64) -> f64 {
    let a = -0.5f64;
    let at = t.abs();
    if at <= 1.0 {
        (a + 2.0) * at * at * at - (a + 3.0) * at * at + 1.0
    } else if at < 2.0 {
        a * at * at * at - 5.0 * a * at * at + 8.0 * a * at - 4.0 * a
    } else {
        0.0
    }
}

/// 1-D INTER_AREA (exact area average). Valid for downsampling (m <= n).
fn area_1d(src: &[f32], n: usize, m: usize) -> Vec<f32> {
    let scale = n as f64 / m as f64;
    let mut out = vec![0f32; m];
    for i in 0..m {
        let lo = i as f64 * scale;
        let hi = (i + 1) as f64 * scale;
        let lo_c = if lo < 0.0 { 0.0 } else { lo };
        let hi_c = if hi > n as f64 { n as f64 } else { hi };
        if hi_c <= lo_c { out[i] = 0.0; continue; }
        let j0 = lo_c.floor() as i32;
        let j1 = hi_c.ceil() as i32;
        let mut s = 0.0f64;
        let mut wsum = 0.0f64;
        for j in j0..j1 {
            let a = (j as f64).max(lo_c);
            let b = ((j + 1) as f64).min(hi_c);
            let w = b - a;
            if w <= 0.0 { continue; }
            let sj = if j < 0 { 0 } else if j >= n as i32 { n - 1 } else { j as usize };
            s += src[sj] as f64 * w;
            wsum += w;
        }
        out[i] = if wsum > 0.0 { (s / wsum) as f32 } else { 0.0 };
    }
    out
}

/// 1-D INTER_CUBIC (general up/down), matches cv2's 4-tap bicubic with reflect_101 edges.
fn cubic_1d(src: &[f32], n: usize, m: usize) -> Vec<f32> {
    let scale = n as f64 / m as f64;
    let mut out = vec![0f32; m];
    for i in 0..m {
        let sx = (i as f64 + 0.5) * scale - 0.5;
        let x0 = sx.floor() as i32;
        let dx = sx - x0 as f64;
        let mut s = 0.0f64;
        for k in 0..4 {
            let j = x0 - 1 + k;
            let w = cubic_w(((k as f64 - 1.0) - dx).abs());
            let sj = reflect101(j, n as i32);
            s += src[sj as usize] as f64 * w;
        }
        out[i] = s as f32;
    }
    out
}

/// 2-D INTER_AREA, channels-aware. Used only for downsampling in this module.
fn resize_area(src: &[f32], h: usize, w: usize, h2: usize, w2: usize, ch: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; h * w2 * ch];
    for y in 0..h {
        for c in 0..ch {
            let row: Vec<f32> = (0..w).map(|x| src[(y * w + x) * ch + c]).collect();
            let r = area_1d(&row, w, w2);
            for i in 0..w2 { tmp[(y * w2 + i) * ch + c] = r[i]; }
        }
    }
    let mut out = vec![0f32; h2 * w2 * ch];
    for i in 0..w2 {
        for c in 0..ch {
            let col: Vec<f32> = (0..h).map(|y| tmp[(y * w2 + i) * ch + c]).collect();
            let r = area_1d(&col, h, h2);
            for jy in 0..h2 { out[(jy * w2 + i) * ch + c] = r[jy]; }
        }
    }
    out
}

/// 2-D INTER_CUBIC, channels-aware (horizontal then vertical, separable).
fn resize_cubic(src: &[f32], h: usize, w: usize, h2: usize, w2: usize, ch: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; h * w2 * ch];
    for y in 0..h {
        for c in 0..ch {
            let row: Vec<f32> = (0..w).map(|x| src[(y * w + x) * ch + c]).collect();
            let r = cubic_1d(&row, w, w2);
            for i in 0..w2 { tmp[(y * w2 + i) * ch + c] = r[i]; }
        }
    }
    let mut out = vec![0f32; h2 * w2 * ch];
    for i in 0..w2 {
        for c in 0..ch {
            let col: Vec<f32> = (0..h).map(|y| tmp[(y * w2 + i) * ch + c]).collect();
            let r = cubic_1d(&col, h, h2);
            for jy in 0..h2 { out[(jy * w2 + i) * ch + c] = r[jy]; }
        }
    }
    out
}

/// 2-D INTER_NEAREST, channels-aware (cv2 floor(dst*scale) mapping — see resize_mask).
fn resize_nearest(src: &[f32], h: usize, w: usize, h2: usize, w2: usize, ch: usize) -> Vec<f32> {
    let sx = w as f64 / w2 as f64;
    let sy = h as f64 / h2 as f64;
    let mut out = vec![0f32; h2 * w2 * ch];
    for y2 in 0..h2 {
        let cy = (((y2 as f64) * sy).floor() as i32).clamp(0, h as i32 - 1) as usize;
        for x2 in 0..w2 {
            let cx = (((x2 as f64) * sx).floor() as i32).clamp(0, w as i32 - 1) as usize;
            for c in 0..ch {
                out[(y2 * w2 + x2) * ch + c] = src[(cy * w + cx) * ch + c];
            }
        }
    }
    out
}


// ---- harmonic background (Jacobi multigrid) -------------------------------

// ---- harmonic background (Jacobi multigrid, 3ch, matches cv2._harmonic_background) --
// cv2 solves ALL 3 channels of B (not just luma) and excludes bright ridges from the
// solve domain so they keep their true value. We mirror that exactly.

/// Prepare one harmonic level: returns (Hl, Wl, hole_l, domain_l, vals_l, pin_l).
/// `scale` = coarse divisor base (= ceil(long_side/200), mirrored from cv2); `level_div`
/// is the extra divisor (4 for coarse init, 1 for fine solve).
/// `domain_l` = solve_domain = hole | bright (or hole after the bright fallback) —
/// cv2 solves over hole|bright, NOT hole alone.
fn harmonic_prep(
    values: &[f32], hole: &[u8], h: usize, w: usize, scale: i32, level_div: i32,
) -> Option<(usize, usize, Vec<u8>, Vec<u8>, Vec<f32>, Vec<u8>)> {
    let Wl = ((w as i32 / (scale * level_div)).max(2)) as usize;
    let Hl = ((h as i32 / (scale * level_div)).max(2)) as usize;
    let hole_l = resize_mask(hole, h, w, Hl, Wl);
    let vals_l = resize_area(values, h, w, Hl, Wl, 3);
    let gray_l: Vec<f32> = (0..Hl * Wl)
        .map(|i| 0.299 * vals_l[i * 3] + 0.587 * vals_l[i * 3 + 1] + 0.114 * vals_l[i * 3 + 2])
        .collect();
    let out_vals: Vec<f32> = (0..Hl * Wl).filter(|&i| hole_l[i] == 0).map(|i| gray_l[i]).collect();
    let bright: Vec<bool> = if !out_vals.is_empty() {
        let thr = percentile(&out_vals, 0.6) + 12.0;
        (0..Hl * Wl).map(|i| hole_l[i] == 0 && gray_l[i] > thr).collect()
    } else {
        vec![false; Hl * Wl]
    };
    let mut solve_domain = vec![0u8; Hl * Wl];
    let mut bright_count = 0usize;
    for i in 0..Hl * Wl {
        solve_domain[i] = if hole_l[i] != 0 || bright[i] { 1 } else { 0 };
        if bright[i] { bright_count += 1; }
    }
    let bright_mean = bright_count as f32 / (Hl * Wl) as f32;
    let pin_count = Hl * Wl - solve_domain.iter().map(|&v| v as usize).sum::<usize>();
    if bright_mean > 0.6 || pin_count < 8 {
        for i in 0..Hl * Wl { solve_domain[i] = hole_l[i]; }
    }
    // cv2: pin = ~solve_domain — recomputed AFTER the bright fallback, and the
    // pin.sum() < 8 bail-out uses the post-fallback pin.
    let pin: Vec<u8> = solve_domain.iter().map(|&v| if v == 0 { 1 } else { 0 }).collect();
    let pin_count_final = pin.iter().filter(|&&v| v != 0).count();
    let hole_any = hole_l.iter().any(|&v| v != 0);
    let hole_all = hole_l.iter().all(|&v| v != 0);
    if !hole_any || hole_all || pin_count_final < 8 {
        return None;
    }
    Some((Hl, Wl, hole_l, solve_domain, vals_l, pin))
}

/// Jacobi solve on `domain_l` (3ch), pinning `pin_l` at `vals_l`. Optional `init`
/// (3ch, same grid) seeds the domain interior (cv2: B[domain] = init_vals[domain];
/// the ring mean is computed ONLY when no init is given).
fn harmonic_solve(
    vals_l: &[f32], domain_l: &[u8], pin_l: &[u8], h: usize, w: usize,
    iters: usize, init: Option<&[f32]>,
) -> Option<Vec<f32>> {
    let n = h * w;
    let domain_any = domain_l.iter().any(|&v| v != 0);
    let domain_all = domain_l.iter().all(|&v| v != 0);
    let pin_any = pin_l.iter().any(|&v| v != 0);
    if !domain_any || domain_all || !pin_any {
        return None;
    }
    let mut b = vals_l.to_vec();
    if let Some(init) = init {
        for i in 0..n {
            if domain_l[i] != 0 {
                for ch in 0..3 { b[i * 3 + ch] = init[i * 3 + ch]; }
            }
        }
    } else {
        // ring init: mean over non-domain band at distance [2,4] from domain (per channel)
        let dout = distance_transform(domain_l, h, w);
        let mut ring = vec![false; n];
        for i in 0..n {
            if domain_l[i] == 0 && dout[i] >= 2.0 && dout[i] <= 4.0 {
                ring[i] = true;
            }
        }
        if !ring.iter().any(|&v| v) {
            return None;
        }
        let mut sm = [0.0f64; 3];
        let mut c = 0.0;
        for i in 0..n {
            if ring[i] {
                for ch in 0..3 { sm[ch] += vals_l[i * 3 + ch] as f64; }
                c += 1.0;
            }
        }
        let mean = [sm[0] / c, sm[1] / c, sm[2] / c];
        for i in 0..n {
            if domain_l[i] != 0 {
                for ch in 0..3 { b[i * 3 + ch] = mean[ch] as f32; }
            }
        }
    }
    for i in 0..n {
        if pin_l[i] != 0 {
            for ch in 0..3 { b[i * 3 + ch] = vals_l[i * 3 + ch]; }
        }
    }
    // cv2 Jacobi: up/down/left/right use edge REPLICATION (np.vstack/np.hstack clamp
    // at the border) and ALWAYS divide by 4 — border pixels average with themselves.
    for _ in 0..iters {
        let mut nb = b.clone();
        for y in 0..h {
            let yu = if y == 0 { 0 } else { y - 1 };
            let yd = if y + 1 >= h { h - 1 } else { y + 1 };
            for x in 0..w {
                let i = y * w + x;
                if domain_l[i] == 0 { continue; }
                let xl = if x == 0 { 0 } else { x - 1 };
                let xr = if x + 1 >= w { w - 1 } else { x + 1 };
                for ch in 0..3 {
                    let up = b[(yu * w + x) * 3 + ch];
                    let down = b[(yd * w + x) * 3 + ch];
                    let left = b[(y * w + xl) * 3 + ch];
                    let right = b[(y * w + xr) * 3 + ch];
                    // left-assoc f32 adds, matching numpy's (up+down+left+right)/4
                    let t1 = up + down;
                    let t2 = t1 + left;
                    let t3 = t2 + right;
                    nb[i * 3 + ch] = t3 / 4.0;
                }
            }
        }
        b = nb;
    }
    Some(b)
}

fn harmonic_background(
    values: &[f32], hole: &[u8], h: usize, w: usize, init: Option<&[f32]>,
) -> Option<Vec<f32>> {
    let scale = ((h.max(w) as i32 + 200 - 1) / 200).max(2);
    // coarse level (level_div = 4) → warm-start init grid for the fine solve
    let mut init_grid: Option<Vec<f32>> = None;
    if let Some((Hl, Wl, _hole_l, domain_l, vals_l, pin_l)) = harmonic_prep(values, hole, h, w, scale, 4) {
        if let Some(bc) = harmonic_solve(&vals_l, &domain_l, &pin_l, Hl, Wl, 400, None) {
            let fh = (h as i32 / scale).max(2) as usize;
            let fw = (w as i32 / scale).max(2) as usize;
            init_grid = Some(resize_cubic(&bc, Hl, Wl, fh, fw, 3));
        }
    }
    // fine level (level_div = 1)
    let (Hl, Wl, _hole_l, domain_l, vals_l, pin_l) = harmonic_prep(values, hole, h, w, scale, 1)?;
    // cv2: init_l = resize_cubic(init_grid) then blended 0.45/0.55 with the
    // INTER_AREA downscale of the caller-provided full-res init.
    let init_l: Option<Vec<f32>> = if init_grid.is_some() || init.is_some() {
        let mut il = vals_l.clone();
        if let Some(ig) = &init_grid {
            let fh = (h as i32 / scale).max(2) as usize;
            let fw = (w as i32 / scale).max(2) as usize;
            il = resize_cubic(ig, fh, fw, Hl, Wl, 3);
        }
        if let Some(ini) = init {
            let ini_l = resize_area(ini, h, w, Hl, Wl, 3);
            il = if init_grid.is_some() {
                (0..Hl * Wl * 3).map(|k| il[k] * 0.45 + ini_l[k] * 0.55).collect()
            } else {
                ini_l
            };
        }
        Some(il)
    } else {
        None
    };
    let b = harmonic_solve(&vals_l, &domain_l, &pin_l, Hl, Wl, 300, init_l.as_deref())?;
    let full = resize_cubic(&b, Hl, Wl, h, w, 3);
    Some(gaussian_blur_3(&full, h, w, 2.0))
}


/// Exact EDT (mirror of lib.rs distance_transform_edt, single-channel f32). Used by the
/// deglow zone computation, which mirrors cv2.distanceTransform(..., DIST_L2, 5) (mask size
/// 5 is near-exact, so the exact EDT is the correct reference here).
fn distance_transform(mask: &[u8], h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let mut f = vec![f64::INFINITY; n];
    for i in 0..n {
        if mask[i] != 0 { f[i] = 0.0; }
    }
    edt2d_local(&mut f, h, w);
    f.iter().map(|&v| v.sqrt() as f32).collect()
}

fn edt2d_local(f: &mut [f64], h: usize, w: usize) {
    // simplified 1D passes (F&H). Reuse a basic separable squared-distance transform.
    let m = h.max(w);
    let mut d = vec![0f64; m];
    let mut z = vec![0f64; m + 1];
    let mut v = vec![0usize; m + 1];
    for y in 0..h {
        let base = y * w;
        edt1d_local(&f[base..base + w], &mut d[..w], &mut z[..w + 1], &mut v[..w + 1], w);
        for x in 0..w { f[base + x] = d[x]; }
    }
    let mut col = vec![0f64; h];
    for x in 0..w {
        for y in 0..h { col[y] = f[y * w + x]; }
        edt1d_local(&col[..h], &mut d[..h], &mut z[..h + 1], &mut v[..h + 1], h);
        for y in 0..h { f[y * w + x] = d[y]; }
    }
}

fn edt1d_local(f: &[f64], d: &mut [f64], z: &mut [f64], v: &mut [usize], n: usize) {
    let mut k: i32 = 0;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = ((f[q] + (q as f64) * (q as f64))
            - (f[v[k as usize]] + (v[k as usize] as f64) * (v[k as usize] as f64)))
            / (2.0 * (q as f64 - v[k as usize] as f64));
        if s.is_nan() { s = f64::INFINITY; }
        while k > 0 && s <= z[k as usize] {
            k -= 1;
            s = ((f[q] + (q as f64) * (q as f64))
                - (f[v[k as usize]] + (v[k as usize] as f64) * (v[k as usize] as f64)))
                / (2.0 * (q as f64 - v[k as usize] as f64));
            if s.is_nan() { s = f64::INFINITY; }
        }
        k += 1;
        v[k as usize] = q;
        z[k as usize] = s;
        z[(k + 1) as usize] = f64::INFINITY;
    }
    k = 0;
    for q in 0..n {
        while z[(k + 1) as usize] < (q as f64) { k += 1; }
        let dq = (q as f64) - (v[k as usize] as f64);
        d[q] = dq * dq + f[v[k as usize]];
    }
}

/// cv2.distanceTransform(src, DIST_L2, 3) — OpenCV's 3x3 two-pass chamfer.
///
/// Empirically calibrated against cv2: orthogonal weight a = 0.9550826, diagonal
/// weight b = 1.369319, with seed (nonzero/src=0) pixels initialised to 0. Matches
/// cv2's DIST_L2 maskSize=3 to <0.002 (e.g. corner of a 120x120 single-seed map:
/// cv2 = 162.947, chamfer = 162.949). Used only by `absorb_zone_bright_core`, whose
/// `dist_max` (==18.0) threshold then matches the original Python cv2 pipeline exactly.
fn distance_transform_cv3(mask: &[u8], h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let a = 0.9550826f32;
    let b = 1.369319f32;
    let mut d: Vec<f32> = (0..n).map(|i| if mask[i] != 0 { 0.0 } else { f32::INFINITY }).collect();
    // forward pass
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let p = (y * w as i32 + x) as usize;
            if d[p] == 0.0 { continue; }
            let mut best = d[p];
            if x > 0 && y > 0 { let q = ((y - 1) * w as i32 + (x - 1)) as usize; let v = d[q] + b; if v < best { best = v; } }
            if y > 0 { let q = ((y - 1) * w as i32 + x) as usize; let v = d[q] + a; if v < best { best = v; } }
            if x < w as i32 - 1 && y > 0 { let q = ((y - 1) * w as i32 + (x + 1)) as usize; let v = d[q] + b; if v < best { best = v; } }
            if x > 0 { let q = (y * w as i32 + (x - 1)) as usize; let v = d[q] + a; if v < best { best = v; } }
            d[p] = best;
        }
    }
    // backward pass
    for y in (0..h as i32).rev() {
        for x in (0..w as i32).rev() {
            let p = (y * w as i32 + x) as usize;
            if d[p] == 0.0 { continue; }
            let mut best = d[p];
            if x < w as i32 - 1 && y < h as i32 - 1 { let q = ((y + 1) * w as i32 + (x + 1)) as usize; let v = d[q] + b; if v < best { best = v; } }
            if y < h as i32 - 1 { let q = ((y + 1) * w as i32 + x) as usize; let v = d[q] + a; if v < best { best = v; } }
            if x > 0 && y < h as i32 - 1 { let q = ((y + 1) * w as i32 + (x - 1)) as usize; let v = d[q] + b; if v < best { best = v; } }
            if x < w as i32 - 1 { let q = (y * w as i32 + (x + 1)) as usize; let v = d[q] + a; if v < best { best = v; } }
            d[p] = best;
        }
    }
    d
}

// ---- shared PatchMatch fill helper + edge-aware grow -----------------------

/// cv2.cvtColor(RGB2LAB) L-channel, packed to [0,255] the same way cv2 stores the
/// u8 LAB array (`saturate_cast<uchar>(L * 255/100)`, round-half-to-even). Used only
/// by `edge_aware_grow`.
fn lab_l(r: f32, g: f32, b: f32) -> f32 {
    let r = (r / 255.0).clamp(0.0, 1.0);
    let g = (g / 255.0).clamp(0.0, 1.0);
    let b = (b / 255.0).clamp(0.0, 1.0);
    let lin = |c: f32| -> f32 {
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let r = lin(r); let g = lin(g); let b = lin(b);
    let x = r * 0.412453 + g * 0.357580 + b * 0.180423;
    let y = r * 0.212671 + g * 0.715160 + b * 0.072169;
    let z = r * 0.019334 + g * 0.119193 + b * 0.950227;
    let x = x / 0.950456;
    let y = y / 1.0;
    let z = z / 1.088754;
    let f = |t: f32| -> f32 {
        if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 }
    };
    let l = 116.0 * f(y) - 16.0; // [0,100]
    cv_round(l * 255.0 / 100.0)
}

/// Median of an already-sorted slice; average of the two middle elements for even length.
fn median_sorted(v: &[f32]) -> f32 {
    let m = v.len();
    if m == 0 { return 0.0; }
    if m % 2 == 1 { v[m / 2] } else { (v[m / 2 - 1] + v[m / 2]) / 2.0 }
}

/// Rust port of `text_eraser/eraser.py::_edge_aware_grow`. Grows `mask_filled` by
/// merging nearby original pixels whose LAB-L luminance lies inside the text band
/// `(band_lo, band_hi)`; erodes the candidate once to drop isolated specks, then
/// re-ORs the original mask so the hole is never shrunk.
fn edge_aware_grow(rgb: &[f32], mask_filled: &[u8], h0: usize, w0: usize) -> Vec<u8> {
    let n = h0 * w0;
    if !mask_filled.iter().any(|&v| v != 0) {
        return mask_filled.to_vec();
    }
    let lum: Vec<f32> = (0..n).map(|i| lab_l(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2])).collect();
    let mut text_vals: Vec<f32> = Vec::with_capacity(n);
    let mut bg_vals: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        if mask_filled[i] != 0 { text_vals.push(lum[i]); }
        else { bg_vals.push(lum[i]); }
    }
    if text_vals.is_empty() { return mask_filled.to_vec(); }
    let lo = text_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = text_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    bg_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let bg = median_sorted(&bg_vals);
    let band_lo = (bg + lo) / 2.0;
    let band_hi = hi + (hi - lo) * 0.5;
    let cand = dilate_ellipse(mask_filled, h0, w0, 2 * 8 + 1); // ellipse(8)
    let mut grown = vec![0u8; n];
    for i in 0..n {
        if cand[i] != 0 && lum[i] >= band_lo && lum[i] <= band_hi {
            grown[i] = 255;
        }
    }
    grown = erode_ellipse(&grown, h0, w0, 2 * 1 + 1); // ellipse(1)
    for i in 0..n {
        if mask_filled[i] != 0 { grown[i] = 255; }
    }
    grown
}

/// Run the shared PatchMatch fill over `hole` (H*W u8, 0/255) and return the full
/// H*W*3 f32 image with the hole filled and everything else copied from `clean`.
/// ROI margin + MAX_ROI shrink mirror `text_eraser/patch_fill.py::inpaint`. `excl`
/// (H*W u8, 0/1) marks sample-exclusion pixels (residual green / dark source).
fn pm_fill_roi(
    clean: &[f32], hole: &[u8], excl: &[u8],
    h0: usize, w0: usize,
    direction_deg: f32, seed: u32,
) -> Vec<f32> {
    let n = h0 * w0;
    let mut res: Vec<f32> = clean.to_vec();
    let mut ymin = h0; let mut ymax = 0usize;
    let mut xmin = w0; let mut xmax = 0usize;
    let mut any = false;
    for y in 0..h0 {
        for x in 0..w0 {
            if hole[y * w0 + x] != 0 {
                any = true;
                if y < ymin { ymin = y; }
                if y > ymax { ymax = y; }
                if x < xmin { xmin = x; }
                if x > xmax { xmax = x; }
            }
        }
    }
    if !any { return res; }
    let span = ((ymax - ymin + 1) as f32).max((xmax - xmin + 1) as f32);
    let mut margin = (32.0f32).max((0.6 * span).round());
    margin = margin.max((0.9 * span).round()).max(80.0);
    let mut margin = margin as i64;
    let mut ry0 = ((ymin as i64) - margin).max(0) as usize;
    let mut ry1 = ((ymax as i64) + 1 + margin).min(h0 as i64) as usize;
    let mut rx0 = ((xmin as i64) - margin).max(0) as usize;
    let mut rx1 = ((xmax as i64) + 1 + margin).min(w0 as i64) as usize;
    while ((ry1 - ry0).max(rx1 - rx0) as i64) > 1536 && margin > 24 {
        margin = (margin as f64 * 0.85) as i64;
        ry0 = ((ymin as i64) - margin).max(0) as usize;
        ry1 = ((ymax as i64) + 1 + margin).min(h0 as i64) as usize;
        rx0 = ((xmin as i64) - margin).max(0) as usize;
        rx1 = ((xmax as i64) + 1 + margin).min(w0 as i64) as usize;
    }
    let sh = ry1 - ry0;
    let sw = rx1 - rx0;
    let sub_n = sh * sw;
    let mut sub_in = vec![0f32; sub_n * 3];
    let mut sub_mf = vec![0u8; sub_n];
    let mut sub_s = vec![0u8; sub_n];
    for y in 0..sh {
        for x in 0..sw {
            let si = (y * sw + x) * 3;
            let gi = ((y + ry0) * w0 + (x + rx0)) * 3;
            sub_in[si] = clean[gi];
            sub_in[si + 1] = clean[gi + 1];
            sub_in[si + 2] = clean[gi + 2];
            let mi = y * sw + x;
            let gmi = (y + ry0) * w0 + (x + rx0);
            sub_mf[mi] = hole[gmi];
            let mut s = if hole[gmi] != 0 { 0u8 } else { 255u8 };
            if excl[gmi] != 0 { s = 0; }
            sub_s[mi] = s;
        }
    }
    let p_in = crate::alloc(sub_n * 3 * 4);
    let p_mf = crate::alloc(sub_n);
    let p_s = crate::alloc(sub_n);
    let p_out = crate::alloc(sub_n * 3 * 4);
    {
        let min = unsafe { std::slice::from_raw_parts_mut(p_in as *mut f32, sub_n * 3) };
        let msl = unsafe { std::slice::from_raw_parts_mut(p_mf, sub_n) };
        let ssl = unsafe { std::slice::from_raw_parts_mut(p_s, sub_n) };
        for i in 0..sub_n * 3 { min[i] = sub_in[i]; }
        for i in 0..sub_n { msl[i] = sub_mf[i]; ssl[i] = sub_s[i]; }
    }
    crate::patchmatch::patchmatch_inpaint(
        p_in as *const f32, sh as i32, sw as i32, p_mf, p_s, 1, 7,
        direction_deg, seed, p_out as *mut f32);
    let out_f32 = unsafe { std::slice::from_raw_parts(p_out as *const f32, sub_n * 3) };
    for y in 0..sh {
        for x in 0..sw {
            let si = (y * sw + x) * 3;
            let gi = ((y + ry0) * w0 + (x + rx0)) * 3;
            res[gi] = out_f32[si];
            res[gi + 1] = out_f32[si + 1];
            res[gi + 2] = out_f32[si + 2];
        }
    }
    crate::dealloc(p_in, sub_n * 3 * 4);
    crate::dealloc(p_mf, sub_n);
    crate::dealloc(p_s, sub_n);
    crate::dealloc(p_out, sub_n * 3 * 4);
    res
}

// ===========================================================================
// MAIN: deglow_full_green_v2
// ===========================================================================

#[no_mangle]
pub extern "C" fn deglow_full_green_v2(
    rgb_ptr: *const f32,
    h: i32,
    w: i32,
    tmask_ptr: *const u8,
    strength: f32,
    zone_ratio: f32,
    zone_expand: i32,
    protect_px: i32,
    chroma_keep: i32,
    out_clean_ptr: *mut u8,
    out_core_ptr: *mut u8,
    out_zone_ptr: *mut u8,
) {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    let rgb = unsafe { std::slice::from_raw_parts(rgb_ptr, n * 3) };
    let tmask = unsafe { std::slice::from_raw_parts(tmask_ptr, n) };
    let clean = unsafe { std::slice::from_raw_parts_mut(out_clean_ptr, n * 3) };
    let core = unsafe { std::slice::from_raw_parts_mut(out_core_ptr, n) };
    let zone_out = unsafe { std::slice::from_raw_parts_mut(out_zone_ptr, n) };
    for i in 0..n { zone_out[i] = 0; }

    let mut out = vec![0f32; n * 3];
    for i in 0..n * 3 { out[i] = rgb[i]; }
    let r: Vec<f32> = (0..n).map(|i| rgb[i * 3]).collect();
    let g: Vec<f32> = (0..n).map(|i| rgb[i * 3 + 1]).collect();
    let b: Vec<f32> = (0..n).map(|i| rgb[i * 3 + 2]).collect();

    let s = clamp_f(strength, 0.0, 1.5);
    let empty = vec![0u8; n];
    if s <= 0.0 {
        write_out(&out, clean);
        for i in 0..n { core[i] = 0; }
        return;
    }

    let gray = gray_f32(rgb, n);
    let max_rb = r.iter().zip(&b).map(|(a, c)| a.max(*c)).collect::<Vec<_>>();
    let green = (0..n).map(|i| (g[i] - max_rb[i] > 2.0) && (g[i] > 60.0)).collect::<Vec<bool>>();
    let strong_green = (0..n).map(|i| (g[i] - max_rb[i] > 8.0) && (g[i] > 95.0)).collect::<Vec<bool>>();

    // max connected-component size of strong_green (8-connectivity)
    let strong_u8: Vec<u8> = strong_green.iter().map(|&v| if v { 255 } else { 0 }).collect();
    let (labels, stats) = connected_components_with_stats(&strong_u8, h, w);
    let mut max_cc = 0i32;
    for st in stats.iter().skip(1) {
        if st[4] > max_cc { max_cc = st[4]; }
    }
    let min_strong = 30;
    if max_cc < min_strong {
        write_out(&out, clean);
        for i in 0..n { core[i] = 0; }
        return;
    }

    let min_rgb = (0..n).map(|i| r[i].min(g[i]).min(b[i])).collect::<Vec<_>>();
    let text_stroke = (0..n)
        .map(|i| (min_rgb[i] > 120.0) && ((g[i] - max_rb[i]) < 40.0))
        .collect::<Vec<bool>>();

    // grow zone
    let bg_cand: Vec<f32> = (0..n).filter(|&i| !strong_green[i]).map(|i| gray[i]).collect();
    let bg_lum = if bg_cand.is_empty() { 80.0 } else { percentile(&bg_cand, 0.5) };
    let greenness_grow = (0..n).map(|i| (g[i] - max_rb[i]).max(0.0)).collect::<Vec<_>>();
    let bright = (0..n)
        .map(|i| (gray[i] > (bg_lum + 6.0)) && (gray[i] > 55.0) && (greenness_grow[i] > 2.0))
        .collect::<Vec<bool>>();
    let faint_green = (0..n).map(|i| (g[i] - max_rb[i] > 3.0) && (g[i] > 55.0)).collect::<Vec<bool>>();
    let grow_cond: Vec<bool> = (0..n)
        .map(|i| green[i] || bright[i] || faint_green[i])
        .collect();

    let mut zone = vec![0u8; n];
    for i in 0..n {
        if strong_green[i] || tmask[i] > 0 { zone[i] = 1; }
    }
    let budget = ((h * w) as f64
        * (if zone_ratio > 0.0 { zone_ratio as f64 } else { 0.6 })) as i64;
    // cv2 uses k3 = np.ones((3,3)) — a 3x3 SQUARE (8-connectivity) dilation for the
    // zone growth, NOT an ellipse. Must match exactly.
    for _ in 0..400 {
        let dil = box_dilate(&zone, h, w);
        let mut added = false;
        let mut add = vec![0u8; n];
        for i in 0..n {
            if dil[i] != 0 && grow_cond[i] && zone[i] == 0 {
                add[i] = 1;
                added = true;
            }
        }
        if !added { break; }
        let mut sum = 0i64;
        for i in 0..n { if zone[i] != 0 || add[i] != 0 { sum += 1; } }
        if sum > budget {
            break; // rollback implied: we simply stop applying add
        }
        for i in 0..n { if add[i] != 0 { zone[i] = 1; } }
    }

    // zone_expand: applied to the RETURNED `zone` (matches cv2 — `zone` gets the
    // expand, then `m_zone = zone`, and `m_zone` may later be grown by the warm
    // 29px dilation for the glow-subtraction region only; the returned zone is
    // the undilated/expanded one, never the 29px-dilated m_zone).
    if zone_expand > 0 {
        zone = dilate_ellipse(&zone, h, w, zone_expand * 2 + 1);
    }
    let mut m_zone = zone.clone();

    if !m_zone.iter().any(|&v| v != 0) {
        // no zone: return original + tmask as core
        write_out(&out, clean);
        for i in 0..n { core[i] = if tmask[i] > 0 { 255 } else { 0 }; }
        return;
    }

    let greenness = (0..n).map(|i| (g[i] - max_rb[i]).max(0.0)).collect::<Vec<_>>();
    // warm compensation ring — cv2 uses a SQUARE 21×21 box (`np.ones((21,21))`), not an ellipse.
    let ring = box_dilate_k(&m_zone, h, w, 21);
    let ring_band: Vec<usize> = (0..n).filter(|&i| ring[i] != 0 && m_zone[i] == 0).collect();
    let d_warm = if ring_band.is_empty() {
        0.0
    } else {
        let mut vals = Vec::with_capacity(ring_band.len());
        for &i in &ring_band { vals.push(r[i] - g[i]); }
        let med = percentile(&vals, 0.5);
        if med > 0.0 { med } else { 0.0 }
    };

    // bg source availability (matches cv2: >= max(30, 1%) non-zone, low-saturation,
    // non-pure-black/white pixels) — gates whether the geodesic background is computed.
    let mx0: Vec<f32> = (0..n).map(|i| r[i].max(g[i]).max(b[i])).collect();
    let mn0: Vec<f32> = (0..n).map(|i| r[i].min(g[i]).min(b[i])).collect();
    let bg_src_ok = (0..n)
        .filter(|&i| zone[i] == 0 && (mx0[i] - mn0[i]) < 30.0 && mx0[i] > 15.0 && mx0[i] < 240.0)
        .count()
        >= (30.max((0.01 * (h * w) as f32) as usize));
    let zone_sum = zone.iter().filter(|&&v| v != 0).count();

    // geodesic background B + chroma fields. GATED exactly like cv2: only when the zone
    // is below 80% of the image OR there is sufficient non-zone background to source from.
    // Without this gate a large zone (>=80% with no bg source) wrongly produces D_rg and
    // triggers the 29px m_zone dilation, exploding the zone far past the cv2 result.
    let mut B: Option<Vec<f32>> = None;
    let mut d_rg: Option<Vec<f32>> = None;
    let mut d_gb: Option<Vec<f32>> = None;
    if zone_sum > 0 && (zone_sum < (0.8 * (h * w) as f64) as usize || bg_src_ok) {
        // cv2: geo_mask = cv2.erode(zone, ones((3,3)), iterations=3) — a SQUARE kernel
        // applied THREE times on `zone` (not a single 3x3 ellipse pass on m_zone).
        let geo_mask = box_erode_iters(&zone, h, w, 3);
        // distance from each OUTSIDE-zone pixel to the nearest ZONE pixel (cv2 uses (~zone)).
        let zone_mask: Vec<u8> = zone.iter().map(|&v| if v != 0 { 255 } else { 0 }).collect();
        let dout = distance_transform(&zone_mask, h, w);
        let ring_clean: Vec<u8> = (0..n)
            .map(|i| {
                if zone[i] == 0 && dout[i] >= 10.0 && dout[i] <= 26.0 && greenness[i] <= 6.0 {
                    1
                } else {
                    0
                }
            })
            .collect();
        if ring_clean.iter().any(|&v| v != 0) {
            let rg = (0..n).map(|i| r[i] - g[i]).collect::<Vec<_>>();
            let gb = (0..n).map(|i| g[i] - b[i]).collect::<Vec<_>>();
            let (b_field, extras) = geodesic_background(
                rgb, h, w, &geo_mask, &[rg, gb], Some(&ring_clean));
            B = Some(b_field);
            if extras.len() >= 2 {
                d_rg = Some(extras[0].clone());
                d_gb = Some(extras[1].clone());
            }
        } else {
            let (b_field, _) = geodesic_background(rgb, h, w, &geo_mask, &[], None);
            B = Some(b_field);
        }
    }

    // subtract green
    let mut glow = vec![0f32; n];
    let mut use_warm = false;
    if d_warm > 0.0 && d_rg.is_some() {
        use_warm = true;
        let drg = d_rg.as_ref().unwrap();
        for i in 0..n {
            let val = (drg[i] - (r[i] - g[i])).max(0.0);
            glow[i] = if text_stroke[i] { greenness[i] } else { val };
        }
        m_zone = dilate_ellipse(&m_zone, h, w, 29);
    } else {
        let comp = (0..n).map(|i| if text_stroke[i] { 0.0 } else { d_warm }).collect::<Vec<_>>();
        for i in 0..n {
            glow[i] = (greenness[i] + comp[i]).max(0.0);
        }
    }
    for i in 0..n {
        if m_zone[i] != 0 {
            let gn = out[i * 3 + 1] - glow[i] * s;
            // cv2: np.clip(Gn, 0, 255).astype(np.int16) — the de-glowed G is stored
            // INTEGER. Every later step (grey fallback, detail, keep-layer, mix)
            // reads the truncated value; keeping f32 fractions breaks byte parity.
            out[i * 3 + 1] = clamp_f(gn, 0.0, 255.0).floor();
        }
    }

    // big-zone neutral fallback (no warm, zone large, no B):
    // cv2 paints each pixel with its OWN post-subtraction channel mean
    // (`_avg = out[_m].mean(axis=1); out[_m] = np.repeat(_avg, 3, axis=1)`),
    // i.e. a per-pixel grey that keeps spatial luminance structure. The old
    // code painted ONE global mean of the ORIGINAL (pre-subtract) channels —
    // on green scenes that is a flat green tint over the whole zone
    // (1787767611178: entire background stayed green).
    if d_warm <= 0.0 && B.is_none() {
        let zone_cnt = zone.iter().filter(|&&v| v != 0).count() as f64;
        if zone_cnt >= 0.8 * (h * w) as f64 {
            for i in 0..n {
                if zone[i] != 0 && !text_stroke[i] {
                    // np .astype(np.int16) truncates toward zero (floor for >=0)
                    let avg = ((out[i * 3] + out[i * 3 + 1] + out[i * 3 + 2]) / 3.0).floor();
                    let v = clamp_f(avg, 0.0, 255.0);
                    out[i * 3] = v;
                    out[i * 3 + 1] = v;
                    out[i * 3 + 2] = v;
                }
            }
        }
    }

    // cv2: protect2 = cv2.dilate(text_stroke, k3, iterations=max(0, int(protect_px))) > 0
    // (k3 = 3x3 SQUARE. The old port used an ellipse ERODE — direction and kernel
    // were both wrong, shrinking the ring instead of growing it.)
    let ts_u8: Vec<u8> = (0..n).map(|i| if text_stroke[i] { 1u8 } else { 0 }).collect();
    let mut protect2 = box_dilate_iters(&ts_u8, h, w, protect_px.max(0));
    // stroke growth (10 rounds) — cv2: _cur = text_stroke & zone; growth gated ONLY
    // by _cand (no m_zone gate); kernel k3 SQUARE; then protect2 |= (_cur & zone).
    if zone.iter().any(|&v| v != 0) {
        let z_o = (0..n).filter(|&i| zone[i] == 0).collect::<Vec<_>>();
        let bg25 = if z_o.is_empty() { 80.0 } else {
            let vals: Vec<f32> = z_o.iter().map(|&i| gray[i]).collect();
            percentile(&vals, 0.25)
        };
        let cand = (0..n)
            .map(|i| (gray[i] > bg25 + 20.0) && (min_rgb[i] >= 92.0) && (greenness[i] >= 25.0) && (greenness[i] < 80.0))
            .collect::<Vec<bool>>();
        let mut cur = (0..n).map(|i| (text_stroke[i] && zone[i] != 0) as u8).collect::<Vec<u8>>();
        for _ in 0..10 {
            let dil = box_dilate(&cur, h, w);
            let mut added = false;
            for i in 0..n {
                if dil[i] != 0 && cand[i] && cur[i] == 0 { cur[i] = 1; added = true; }
            }
            if !added { break; }
        }
        for i in 0..n { if cur[i] != 0 && zone[i] != 0 { protect2[i] = 1; } }
    }

    // cv2: fb = zone & ~protect2; warm path re-assigns fb = zone & ~dilate(text_stroke, k3, 1).
    let stroke_dil1 = box_dilate_iters(&ts_u8, h, w, 1);
    let mut fb = vec![0u8; n];
    for i in 0..n {
        let v = if use_warm {
            (zone[i] != 0) && (stroke_dil1[i] == 0)
        } else {
            (zone[i] != 0) && (protect2[i] == 0)
        };
        fb[i] = if v { 1 } else { 0 };
    }

    if fb.iter().any(|&v| v != 0) && B.is_some() {
        let b_field = B.as_ref().unwrap();
        let mut rebuilt: Vec<f32> = b_field.clone();
        // cv2: if d_warm > 0: _init = stack([B0, B0-D_rg, B0-D_rg-D_gb]).clip(0,255);
        //      _Bh = _harmonic_background(out, zone, init=_init); ...
        if use_warm {
            let init_opt: Option<Vec<f32>> = if d_rg.is_some() {
                let drg = d_rg.as_ref().unwrap();
                let dgb = d_gb.as_ref().unwrap();
                let mut init_flat = vec![0f32; n * 3];
                for i in 0..n {
                    init_flat[i * 3] = clamp_f(b_field[i * 3], 0.0, 255.0);
                    init_flat[i * 3 + 1] = clamp_f(b_field[i * 3] - drg[i], 0.0, 255.0);
                    init_flat[i * 3 + 2] = clamp_f(b_field[i * 3] - drg[i] - dgb[i], 0.0, 255.0);
                }
                Some(init_flat)
            } else {
                None
            };
            // cv2: _harmonic_background(out, zone, ...) — values = the DE-GLOWED out
            // (NOT B!), hole = zone (NOT m_zone).
            let bh = harmonic_background(&out, &zone, h, w, init_opt.as_deref());
            if bh.is_some() && d_rg.is_some() {
                let bh = bh.unwrap();
                // structure-strength field: Sobel of the de-glowed out, geodesically
                // propagated from the clean ring (cv2: rgb + erode(zone, k3, iters=3)).
                let gl = gray_f32(&out, n);
                let (gx, gyv) = sobel(&gl, h, w);
                let str = (0..n).map(|i| (gx[i] * gx[i] + gyv[i] * gyv[i]).sqrt().min(40.0)).collect::<Vec<_>>();
                let dout2 = distance_transform(&zone, h, w);
                let rc2: Vec<u8> = (0..n)
                    .map(|i| if zone[i] == 0 && dout2[i] >= 10.0 && dout2[i] <= 26.0 && greenness[i] <= 6.0 { 1 } else { 0 })
                    .collect();
                if rc2.iter().any(|&v| v != 0) {
                    let (_, es) = geodesic_background(rgb, h, w, &box_erode_iters(&zone, h, w, 3), &[str.clone()], Some(&rc2));
                    if es.len() >= 1 {
                        let sfield = &es[0];
                        let mut wmix = vec![0f32; n];
                        for i in 0..n {
                            wmix[i] = clamp_f(((sfield[i] - 4.0) / 10.0).max(0.0).min(1.0), 0.0, 1.0);
                        }
                        let wmix_blur = gaussian_blur(&wmix, h, w, 4.0);
                        let init_flat = init_opt.unwrap();
                        for i in 0..n {
                            for c in 0..3 {
                                let wv = wmix_blur[i];
                                rebuilt[i * 3 + c] = wv * init_flat[i * 3 + c] + (1.0 - wv) * bh[i * 3 + c];
                            }
                        }
                    } else {
                        rebuilt = bh;
                    }
                } else {
                    rebuilt = bh;
                }
            } else if let Some(init_flat) = init_opt {
                // cv2: elif _init is not None: B = _init
                rebuilt = init_flat;
            }
        }
        // detail = (imgf - gaussian(imgf, 2)) * clip(dist(~protect2)/8, 0,1)
        let imgf = out.clone();
        let blurred = gaussian_blur_3(&imgf, h, w, 2.0);
        let dtext = distance_transform(&protect2, h, w);
        let mut dw = vec![0f32; n];
        for i in 0..n { dw[i] = clamp_f(dtext[i] / 8.0, 0.0, 1.0); }
        let mut rebuilt_full = vec![0f32; n * 3];
        for i in 0..n {
            for c in 0..3 {
                let det = (imgf[i * 3 + c] - blurred[i * 3 + c]) * dw[i];
                rebuilt_full[i * 3 + c] = clamp_f(rebuilt[i * 3 + c] + det, 0.0, 255.0);
            }
        }
        // chroma keep
        if chroma_keep != 0 {
            let mut a_map = vec![0f32; n];
            for i in 0..n {
                let ggreen = g[i] - max_rb[i];
                let rb_hot = if (r[i] - b[i]).abs() > 8.0 { 1.0 } else { 0.0 };
                let aw = (if ggreen < 20.0 { rb_hot } else { 0.0 }) * 0.85;
                a_map[i] = aw;
            }
            let a_blur = gaussian_blur(&a_map, h, w, 1.5);
            for i in 0..n {
                if fb[i] != 0 {
                    for c in 0..3 {
                        let keep = if c == 1 { out[i * 3 + 1] } else { imgf[i * 3 + c] };
                        rebuilt_full[i * 3 + c] = rebuilt_full[i * 3 + c] * (1.0 - a_blur[i]) + keep * a_blur[i];
                    }
                }
            }
        }
        // cv2 literal order: (1) out[fb] = rebuilt.astype(int16) — TRUNCATED;
        // (2) soft mix `_mix = out32*(1-w) + rebuilt*w; out[fb] = int16(_mix)`.
        // The mix reads back the TRUNCATED out, so skipping step (1) (or merging
        // the two) leaves low-greenness pixels at their subtracted value = the
        // green residue reported on 1787767611178.
        for i in 0..n {
            if fb[i] != 0 {
                for c in 0..3 {
                    out[i * 3 + c] = rebuilt_full[i * 3 + c].floor();
                }
            }
        }
        let w_soft = (0..n).map(|i| clamp_f((greenness[i] - 5.0) / 20.0, 0.0, 1.0)).collect::<Vec<_>>();
        for i in 0..n {
            if fb[i] != 0 {
                for c in 0..3 {
                    let m = out[i * 3 + c] * (1.0 - w_soft[i]) + rebuilt_full[i * 3 + c] * w_soft[i];
                    out[i * 3 + c] = m.floor();
                }
            }
        }
    }

    // core mask = text_stroke near strong_green (dilate strong_green by ellipse 17)
    let strong_dil = dilate_ellipse(&strong_u8, h, w, 17);
    let mut core_mask = vec![0u8; n];
    for i in 0..n {
        if text_stroke[i] && strong_dil[i] != 0 { core_mask[i] = 255; }
    }

    write_out(&out, clean);
    for i in 0..n { core[i] = core_mask[i]; }
    // Return the undilated `zone` (grew + zone_expand), matching cv2 which returns
    // `zone` (NOT the 29px-dilated `m_zone` used internally for glow subtraction).
    for i in 0..n { zone_out[i] = zone[i]; }
}

// ---- helpers used above ----------------------------------------------------

fn write_out(out: &[f32], clean: &mut [u8]) {
    let n = out.len() / 3;
    for i in 0..n {
        clean[i * 3] = clamp_f(out[i * 3], 0.0, 255.0) as u8;
        clean[i * 3 + 1] = clamp_f(out[i * 3 + 1], 0.0, 255.0) as u8;
        clean[i * 3 + 2] = clamp_f(out[i * 3 + 2], 0.0, 255.0) as u8;
    }
}

/// Connected components (8-conn) + stats, returns (labels int32, stats Vec<[i32;5]>).
fn connected_components_with_stats(mask: &[u8], h: usize, w: usize) -> (Vec<i32>, Vec<[i32; 5]>) {
    let n = h * w;
    let mut labels = vec![0i32; n];
    let dx = [-1i32, -1, -1, 0, 0, 1, 1, 1];
    let dy = [-1i32, 0, 1, -1, 1, -1, 0, 1];
    let mut queue = vec![0i32; n];
    let mut ncomp = 1i32;
    for s in 0..n as i32 {
        if mask[s as usize] == 0 || labels[s as usize] != 0 { continue; }
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
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                let np = ny * w as i32 + nx;
                if mask[np as usize] != 0 && labels[np as usize] == 0 {
                    labels[np as usize] = comp;
                    queue[tail as usize] = np;
                    tail += 1;
                }
            }
        }
    }
    let mut minx = vec![w as i32; ncomp as usize];
    let mut miny = vec![h as i32; ncomp as usize];
    let mut maxx = vec![-1i32; ncomp as usize];
    let mut maxy = vec![-1i32; ncomp as usize];
    let mut area = vec![0i32; ncomp as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let lab = labels[(y * w as i32 + x) as usize] as usize;
            if lab == 0 { continue; }
            if x < minx[lab] { minx[lab] = x; }
            if x > maxx[lab] { maxx[lab] = x; }
            if y < miny[lab] { miny[lab] = y; }
            if y > maxy[lab] { maxy[lab] = y; }
            area[lab] += 1;
        }
    }
    let mut stats = Vec::with_capacity(ncomp as usize);
    for lab in 0..ncomp as usize {
        if lab == 0 || maxx[lab] < 0 {
            stats.push([0, 0, 0, 0, 0]);
        } else {
            stats.push([
                minx[lab],
                miny[lab],
                maxx[lab] - minx[lab] + 1,
                maxy[lab] - miny[lab] + 1,
                area[lab],
            ]);
        }
    }
    (labels, stats)
}

// ===========================================================================
// MASK SURGERY + high-level pipeline orchestration (single source of truth).
// Mirrors text_eraser/_erase_deglow_v2: deglow -> union -> close ->
// fill_bright_near_mask -> absorb_zone_bright_core -> sample-exclude -> patch.
// ===========================================================================

use std::collections::HashSet;

/// 3x3 box dilate / erode (cv2 np.ones((3,3)) semantics).
fn box_dilate(mask: &[u8], h: usize, w: usize) -> Vec<u8> {
    box_dilate_k(mask, h, w, 3)
}

/// Square k×k box dilation (matches cv2.dilate(mask, np.ones((k, k)))).
fn box_dilate_k(mask: &[u8], h: usize, w: usize, k: i32) -> Vec<u8> {
    let r = k / 2;
    let mut out = vec![0u8; h * w];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut hit = false;
            'o: for dy in -r..=r {
                for dx in -r..=r {
                    let ny = y + dy;
                    let nx = x + dx;
                    if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 { continue; }
                    if mask[(ny as usize) * w + (nx as usize)] != 0 { hit = true; break 'o; }
                }
            }
            out[(y as usize) * w + x as usize] = if hit { 1 } else { 0 };
        }
    }
    out
}

fn box_erode(mask: &[u8], h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut all = true;
            'o: for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let ny = y + dy;
                    let nx = x + dx;
                    // cv2.erode default borderValue = morphologyDefaultBorderValue()
                    // = +inf: out-of-bounds pixels are IGNORED (never the min), so
                    // image-border pixels are NOT auto-eroded. (The old code treated
                    // OOB as background, wrongly shaving rows/cols off any mask that
                    // touches the image edge.)
                    if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 { continue; }
                    if mask[(ny as usize) * w + (nx as usize)] == 0 { all = false; break 'o; }
                }
            }
            out[(y as usize) * w + (x as usize)] = if all { 1 } else { 0 };
        }
    }
    out
}

fn mask_union(a: &[u8], b: &[u8], n: usize) -> Vec<u8> {
    (0..n).map(|i| if a[i] != 0 || b[i] != 0 { 255 } else { 0 }).collect()
}

/// cv2.erode(mask, np.ones((3,3)), iterations=n).
fn box_erode_iters(mask: &[u8], h: usize, w: usize, iters: i32) -> Vec<u8> {
    let mut m = mask.to_vec();
    for _ in 0..iters.max(0) {
        m = box_erode(&m, h, w);
    }
    m
}

/// cv2.dilate(mask, np.ones((3,3)), iterations=n).
fn box_dilate_iters(mask: &[u8], h: usize, w: usize, iters: i32) -> Vec<u8> {
    let mut m = mask.to_vec();
    for _ in 0..iters.max(0) {
        m = box_dilate(&m, h, w);
    }
    m
}

/// cv2.morphologyEx(mask, MORPH_CLOSE, ones((3,3))).
fn mask_close(mask: &[u8], h: usize, w: usize) -> Vec<u8> {
    box_erode(&box_dilate(mask, h, w), h, w)
}

/// `_fill_bright_near_mask` port (white-text bright-side connectivity completion).
fn fill_bright_near_mask(rgb: &[f32], mask: &[u8], h: usize, w: usize,
                        bg_lo: f32, lum_off: f32, min_rgb_thr: i32,
                        green_gate: i32, rounds: i32, ext_thr: i32) -> Vec<u8> {
    let n = h * w;
    if !mask.iter().any(|&v| v != 0) { return mask.to_vec(); }
    let gray = gray_f32(rgb, n);
    let outside_gray: Vec<f32> = (0..n).filter(|&i| mask[i] == 0).map(|i| gray[i]).collect();
    let bg = if outside_gray.is_empty() { 90.0 } else { percentile(&outside_gray, ((bg_lo as f64) / 100.0) as f32) };
    let r: Vec<i32> = (0..n).map(|i| rgb[i * 3] as i32).collect();
    let g: Vec<i32> = (0..n).map(|i| rgb[i * 3 + 1] as i32).collect();
    let b: Vec<i32> = (0..n).map(|i| rgb[i * 3 + 2] as i32).collect();
    let min_rgb_im: Vec<i32> = (0..n).map(|i| r[i].min(g[i]).min(b[i])).collect();
    let max_rb: Vec<i32> = (0..n).map(|i| r[i].max(b[i])).collect();
    let cand: Vec<bool> = (0..n).map(|i|
        (gray[i] > bg + lum_off) && (min_rgb_im[i] >= min_rgb_thr) && (g[i] - max_rb[i] < green_gate)
    ).collect();
    if !cand.iter().any(|&v| v) { return mask.to_vec(); }
    let mut cur: Vec<u8> = mask.iter().map(|&v| if v != 0 { 1 } else { 0 }).collect();
    for _ in 0..rounds {
        let dil = box_dilate(&cur, h, w);
        let mut changed = false;
        for i in 0..n {
            if dil[i] != 0 && cand[i] && cur[i] == 0 { cur[i] = 1; changed = true; }
        }
        if !changed { break; }
    }
    let grown: Vec<u8> = cur.iter().map(|&v| if v != 0 { 255 } else { 0 }).collect();
    let added: Vec<bool> = (0..n).map(|i| grown[i] != 0 && mask[i] == 0).collect();
    let leftover: Vec<bool> = (0..n).map(|i| cand[i] && cur[i] == 0).collect();
    let mut grown_mut = grown;
    if added.iter().any(|&v| v) && leftover.iter().any(|&v| v) && ext_thr > 0 {
        let mut reach = cur.clone();
        for _ in 0..ext_thr {
            let nd = box_dilate(&reach, h, w);
            let mut changed = false;
            for i in 0..n {
                if nd[i] != 0 && cand[i] && reach[i] == 0 { reach[i] = 1; changed = true; }
            }
            if !changed { break; }
        }
        let unreached: Vec<bool> = (0..n).map(|i| leftover[i] && reach[i] == 0).collect();
        if unreached.iter().any(|&v| v) {
            let cand_u8: Vec<u8> = cand.iter().map(|&v| if v { 255 } else { 0 }).collect();
            let (labels, _stats) = connected_components_with_stats(&cand_u8, h, w);
            let mut bad: HashSet<i32> = HashSet::new();
            for i in 0..n { if unreached[i] { bad.insert(labels[i]); } }
            bad.remove(&0);
            if !bad.is_empty() {
                for i in 0..n {
                    if added[i] && bad.contains(&labels[i]) { grown_mut[i] = 0; }
                }
            }
        }
    }
    grown_mut
}

/// Pre-CC candidate mask for `_absorb_zone_bright_core` (the pixel-level predicates).
fn absorb_cand(clean: &[f32], orig: &[f32], mask: &[u8], zone: &[u8], h: usize, w: usize,
               bg_off: f32, min_rgb_lo: i32, green_gate: i32, _max_cc_area: i32,
               orig_green_min: i32, dist_max: f32, orig_gray_min: f32) -> Vec<bool> {
    let n = h * w;
    if !mask.iter().any(|&v| v != 0) { return vec![false; n]; }
    if !zone.iter().any(|&v| v != 0) { return vec![false; n]; }
    let mut cand_zone: Vec<bool> = (0..n).map(|i| zone[i] != 0 && mask[i] == 0).collect();
    if !cand_zone.iter().any(|&v| v) { return vec![false; n]; }
    if dist_max > 0.0 {
        let dist = distance_transform_cv3(mask, h, w);
        for i in 0..n { if cand_zone[i] && dist[i] > dist_max { cand_zone[i] = false; } }
        if !cand_zone.iter().any(|&v| v) { return vec![false; n]; }
    }
    let cgray = gray_f32(clean, n);
    let outside_gray: Vec<f32> = (0..n).filter(|&i| mask[i] == 0 && zone[i] == 0).map(|i| cgray[i]).collect();
    let bg = if outside_gray.is_empty() { 90.0 } else { percentile(&outside_gray, 0.25f32) };
    let r: Vec<i32> = (0..n).map(|i| clean[i * 3] as i32).collect();
    let g: Vec<i32> = (0..n).map(|i| clean[i * 3 + 1] as i32).collect();
    let b: Vec<i32> = (0..n).map(|i| clean[i * 3 + 2] as i32).collect();
    let min_rgb: Vec<i32> = (0..n).map(|i| r[i].min(g[i]).min(b[i])).collect();
    let max_rb: Vec<i32> = (0..n).map(|i| r[i].max(b[i])).collect();
    let orr: Vec<i32> = (0..n).map(|i| orig[i * 3] as i32).collect();
    let og: Vec<i32> = (0..n).map(|i| orig[i * 3 + 1] as i32).collect();
    let ob: Vec<i32> = (0..n).map(|i| orig[i * 3 + 2] as i32).collect();
    let orig_green: Vec<i32> = (0..n).map(|i| og[i] - orr[i].max(ob[i])).collect();
    let gorig = gray_f32(orig, n);
    (0..n).map(|i|
        cand_zone[i] && (cgray[i] > bg + bg_off) && (min_rgb[i] >= min_rgb_lo) &&
        (g[i] - max_rb[i] < green_gate) && (orig_green[i] >= orig_green_min) && (gorig[i] >= orig_gray_min)
    ).collect()
}

/// `_absorb_zone_bright_core` port (bright cores isolated inside glow zone).
fn absorb_zone_bright_core(clean: &[f32], orig: &[f32], mask: &[u8], zone: &[u8], h: usize, w: usize,
                          bg_off: f32, min_rgb_lo: i32, green_gate: i32, max_cc_area: i32,
                          orig_green_min: i32, dist_max: f32, orig_gray_min: f32) -> Vec<u8> {
    let n = h * w;
    let cand = absorb_cand(clean, orig, mask, zone, h, w, bg_off, min_rgb_lo, green_gate,
                           max_cc_area, orig_green_min, dist_max, orig_gray_min);
    if !cand.iter().any(|&v| v) { return mask.to_vec(); }
    let cand_u8: Vec<u8> = cand.iter().map(|&v| if v { 255 } else { 0 }).collect();
    let (labels, stats) = connected_components_with_stats(&cand_u8, h, w);
    let mut out = mask.to_vec();
    for i in 0..n {
        let lab = labels[i] as usize;
        if lab != 0 && cand[i] && stats[lab][4] <= max_cc_area {
            out[i] = 255;
        }
    }
    out
}

/// `_residual_green` port — pixels to exclude from the inpaint sample (near mask).
fn residual_green(clean: &[f32], mask: &[u8], h: usize, w: usize, radius: i32, thr: i32, g_lo: i32) -> Vec<u8> {
    let n = h * w;
    let r: Vec<i32> = (0..n).map(|i| clean[i * 3] as i32).collect();
    let g: Vec<i32> = (0..n).map(|i| clean[i * 3 + 1] as i32).collect();
    let b: Vec<i32> = (0..n).map(|i| clean[i * 3 + 2] as i32).collect();
    let max_rb: Vec<i32> = (0..n).map(|i| r[i].max(b[i])).collect();
    let green: Vec<bool> = (0..n).map(|i| (g[i] - max_rb[i] > thr) && (g[i] > g_lo)).collect();
    if !green.iter().any(|&v| v) { return vec![0u8; n]; }
    let near = dilate_ellipse(mask, h, w, radius * 2 + 1);
    let mut out = vec![0u8; n];
    for i in 0..n { if green[i] && near[i] != 0 { out[i] = 1; } }
    out
}

/// `_dark_source_exclude` port — returns (exclude_mask 0/1, any: 0/1).
fn dark_source_exclude(clean: &[f32], mask: &[u8], h: usize, w: usize, ring_px: i32, band: f32) -> (Vec<u8>, i32) {
    let n = h * w;
    let l = gray_f32(clean, n);
    let ring_d = dilate_ellipse(mask, h, w, ring_px * 2 + 1);
    let ring: Vec<bool> = (0..n).map(|i| ring_d[i] != 0 && mask[i] == 0).collect();
    if !ring.iter().any(|&v| v) { return (vec![0u8; n], 0); }
    let ring_lum: Vec<f32> = (0..n).filter(|&i| ring[i]).map(|i| l[i]).collect();
    let refv = percentile(&ring_lum, 0.25f32) - band;
    let mut out = vec![0u8; n];
    let mut any = false;
    for i in 0..n { if l[i] < refv { out[i] = 1; any = true; } }
    (out, if any { 1 } else { 0 })
}

/// THE single shared pipeline entry: de-glow + text-mask surgery + PatchMatch fill.
///
/// `tmask`   : initial text detect (on the raw image).
/// `tmask2`  : re-detect on the de-glowed image (DBNet run twice, both ends).
/// Everything from de-glow through the fill runs in THIS wasm, so the browser and
/// the backend produce byte-identical results by construction.
///
/// Outputs: result (H*W*3 u8), fill mask (H*W u8 0/255), clean de-glowed (H*W*3 u8),
/// glow zone (H*W u8 0/255).
#[no_mangle]
pub extern "C" fn erase_text_glyphs(
    rgb_ptr: *const f32, h: i32, w: i32,
    tmask_ptr: *const u8, tmask2_ptr: *const u8,
    strength: f32, zone_ratio: f32, zone_expand: i32, protect_px: i32, chroma_keep: i32,
    edge: i32, direction_deg: f32, seed: i32,
    edge_aware: i32, soft_expand: f32,
    out_result_ptr: *mut u8, out_fill_ptr: *mut u8, out_clean_ptr: *mut u8, out_zone_ptr: *mut u8,
) {
    let h0 = h as usize;
    let w0 = w as usize;
    let n = h0 * w0;
    let rgb = unsafe { std::slice::from_raw_parts(rgb_ptr, n * 3) };
    let tmask = unsafe { std::slice::from_raw_parts(tmask_ptr, n) };
    let tmask2 = unsafe { std::slice::from_raw_parts(tmask2_ptr, n) };

    // 1) de-glow (also yields the glow zone used by the bright-core absorb).
    let p_clean = crate::alloc(n * 3);
    let p_core = crate::alloc(n);
    let p_zone = crate::alloc(n);
    deglow_full_green_v2(rgb_ptr, h, w, tmask_ptr, strength, zone_ratio, zone_expand, protect_px, chroma_keep, p_clean, p_core, p_zone);
    let clean_u8 = unsafe { std::slice::from_raw_parts(p_clean, n * 3) };
    let zone = unsafe { std::slice::from_raw_parts(p_zone, n) };
    let clean_f32: Vec<f32> = (0..n * 3).map(|i| clean_u8[i] as f32).collect();
    let rgb_f32: Vec<f32> = (0..n * 3).map(|i| rgb[i] as f32).collect();
    let zone_any = zone.iter().any(|&v| v != 0);

    // 2) union of the two detects, then close.
    let mut mask = mask_union(tmask, tmask2, n);
    mask = mask_close(&mask, h0, w0);
    mask = fill_bright_near_mask(&clean_f32, &mask, h0, w0, 25.0, 24.0, 118, 26, 6, 20);
    mask = absorb_zone_bright_core(&clean_f32, &rgb_f32, &mask, zone, h0, w0, 30.0, 100, 26, 200, 18, 18.0, 150.0);

    let empty = !mask.iter().any(|&v| v != 0);
    let use_dir = direction_deg > -1.0;

    // Compute the dilated/eroded fill mask (mirrors cv2 `_run_fill` step 1).
    let mut mf: Vec<u8> = if empty {
        mask.clone()
    } else if edge > 0 {
        dilate_ellipse(&mask, h0, w0, edge * 2 + 1)
    } else if edge < 0 {
        erode_ellipse(&mask, h0, w0, (-edge) * 2 + 1)
    } else {
        mask.clone()
    };
    // 1b. edge-aware grow: merge nearby original pixels whose LAB-L falls inside the
    // text luminance band (eats anti-aliased white edges / light strokes). cv2 applies
    // this to `mask_filled` BEFORE the PatchMatch fill, which is what we mirror here.
    if edge_aware != 0 && !empty {
        mf = edge_aware_grow(&clean_f32, &mf, h0, w0);
    }

    // result starts as the de-glowed image; overwritten by TELEA / PatchMatch below.
    let mut result_u8: Vec<u8> = clean_u8.to_vec();

    // Non-glow near-smooth backgrounds: diffuse with a more permissive TELEA (20.0)
    // instead of PatchMatch (which would copy texture/text back in). Glow images keep
    // the stricter 15.0 threshold inside patchmatch_inpaint; direction mode skips it.
    let mut skip_pm = false;
    if !empty && !use_dir && !zone_any {
        if let Some(filled) = crate::patchmatch::pm_smooth_telea_with_flat_tex(
            &clean_f32, &mf, h0 as i32, w0 as i32, 20.0)
        {
            for i in 0..n * 3 {
                result_u8[i] = clamp_f(filled[i], 0.0, 255.0) as u8;
            }
            skip_pm = true;
        }
    }

    // Final fill mask written to out_fill: the (possibly edge-aware-grown) hole.
    let mask_filled: Vec<u8> = mf;

    if !empty && !skip_pm {
        // residual-green + dark-source exclusions define the protected sample region.
        let mut excl = residual_green(&clean_f32, &mask, h0, w0, 48, 8, 90);
        if zone_any {
            let (dx, dx_any) = dark_source_exclude(&clean_f32, &mask, h0, w0, 4, 25.0);
            if dx_any != 0 { for i in 0..n { if dx[i] != 0 { excl[i] = 1; } } }
        }
        // main PatchMatch fill over the hole (single source of truth, shared with browser).
        let mut res = pm_fill_roi(&clean_f32, &mask_filled, &excl, h0, w0, direction_deg, seed as u32);
        // 3b. soft expand: a feathered band beyond the hole blends the original (de-glowed)
        // image with a second PatchMatch fill of (hole ∪ band) so coverage grows without a
        // hard halo. Mirrors cv2 `_run_fill` soft_expand.
        if soft_expand > 0.0 {
            let s = (soft_expand.min(150.0)).round() as i32; // s >= 1 for any soft_expand > 0
            let band = {
                let d = dilate_ellipse(&mask_filled, h0, w0, 2 * s + 1);
                let mut b = vec![0u8; n];
                for i in 0..n { if d[i] != 0 && mask_filled[i] == 0 { b[i] = 255; } }
                b
            };
            if band.iter().any(|&v| v != 0) {
                let union = {
                    let mut u = mask_filled.clone();
                    for i in 0..n { if band[i] != 0 { u[i] = 255; } }
                    u
                };
                let filled_all = pm_fill_roi(&clean_f32, &union, &excl, h0, w0, direction_deg, seed as u32);
                let inv = {
                    let mut iv = vec![0u8; n];
                    for i in 0..n { iv[i] = if mask_filled[i] != 0 { 0 } else { 255 }; }
                    iv
                };
                let dst = distance_transform_cv3(&inv, h0, w0);
                let sf = s as f32;
                for i in 0..n {
                    if band[i] != 0 {
                        let a = clamp_f(1.0 - dst[i] / sf, 0.0, 1.0);
                        let gi = i * 3;
                        res[gi] = clean_f32[gi] * (1.0 - a) + filled_all[gi] * a;
                        res[gi + 1] = clean_f32[gi + 1] * (1.0 - a) + filled_all[gi + 1] * a;
                        res[gi + 2] = clean_f32[gi + 2] * (1.0 - a) + filled_all[gi + 2] * a;
                    }
                }
            }
        }
        for i in 0..n * 3 {
            result_u8[i] = clamp_f(res[i], 0.0, 255.0) as u8;
        }
    }

    let result_out = unsafe { std::slice::from_raw_parts_mut(out_result_ptr, n * 3) };
    let fill_out = unsafe { std::slice::from_raw_parts_mut(out_fill_ptr, n) };
    let clean_out = unsafe { std::slice::from_raw_parts_mut(out_clean_ptr, n * 3) };
    let zone_out = unsafe { std::slice::from_raw_parts_mut(out_zone_ptr, n) };
    for i in 0..n * 3 { result_out[i] = result_u8[i]; clean_out[i] = clean_u8[i]; }
    for i in 0..n { fill_out[i] = if mask_filled[i] != 0 { 255 } else { 0 }; zone_out[i] = zone[i]; }

    crate::dealloc(p_clean, n * 3);
    crate::dealloc(p_core, n);
    crate::dealloc(p_zone, n);
}
