//! Faithful Rust port of OpenCV `cv::inpaint(..., INPAINT_TELEA)`.
//!
//! The previous `telea.rs` used image-gradient normals and an 8-neighbour loop,
//! which silently left text-shaped holes unfilled (e.g. 1787766251689). This
//! implementation mirrors OpenCV 4.x `modules/photo/src/inpaint.cpp` exactly:
//!   - padded state/distance arrays
//!   - cross-dilated narrow band
//!   - Fast Marching distance propagation (`FastMarching_solve`)
//!   - distance-field gradient `gradT` as the normal
//!   - (2r+1)x(2r+1) neighbourhood with exact `dst*lev*dir` weights
//!   - Ia / Jx / Jy accumulation and final `sat = Ia/s + (Jx+Jy)/|J|`
//!
//! Only the 3-channel (RGB) path is implemented because that is all the shared
//! core needs.

const KNOWN: u8 = 0;
const BAND: u8 = 1;
const INSIDE: u8 = 2;
const CHANGE: u8 = 3;

#[derive(Clone, Copy)]
struct HeapElem {
    t: f32,
    i: i32,
    j: i32,
    order: u64,
}

impl PartialEq for HeapElem {
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t && self.order == other.order
    }
}
impl Eq for HeapElem {}

impl Ord for HeapElem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // min-heap by (t, order)
        other
            .t
            .partial_cmp(&self.t)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.order.cmp(&self.order))
    }
}
impl PartialOrd for HeapElem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct PriorityQueue {
    heap: std::collections::BinaryHeap<HeapElem>,
    next_order: u64,
}

impl PriorityQueue {
    fn new() -> Self {
        Self {
            heap: std::collections::BinaryHeap::new(),
            next_order: 0,
        }
    }
    fn push(&mut self, i: i32, j: i32, t: f32) {
        self.heap.push(HeapElem {
            t,
            i,
            j,
            order: self.next_order,
        });
        self.next_order += 1;
    }
    fn pop(&mut self) -> Option<(i32, i32, f32)> {
        self.heap.pop().map(|e| (e.i, e.j, e.t))
    }
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    if v < 0.0 {
        0
    } else if v > 255.0 {
        255
    } else {
        v.round() as u8
    }
}

#[inline]
fn min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a.min(b).min(c.min(d))
}

fn fast_marching_solve(i1: i32, j1: i32, i2: i32, j2: i32, f: &[u8], t: &[f32], cols: i32) -> f32 {
    let a11 = t[(i1 * cols + j1) as usize];
    let a22 = t[(i2 * cols + j2) as usize];
    let m12 = a11.min(a22);

    if f[(i1 * cols + j1) as usize] != INSIDE {
        if f[(i2 * cols + j2) as usize] != INSIDE {
            if (a11 - a22).abs() >= 1.0 {
                1.0 + m12
            } else {
                (a11 + a22 + (2.0 - (a11 - a22) * (a11 - a22)).sqrt()) * 0.5
            }
        } else {
            1.0 + a11
        }
    } else if f[(i2 * cols + j2) as usize] != INSIDE {
        1.0 + a22
    } else {
        1.0 + m12
    }
}

/// Propagate distances from the initial heap into the INSIDE region.
fn calc_fmm(f: &mut [u8], t: &mut [f32], heap: &mut PriorityQueue, rows: i32, cols: i32, negate: bool) {
    while let Some((ii, jj, _)) = heap.pop() {
        let known = if negate { CHANGE } else { KNOWN };
        f[(ii * cols + jj) as usize] = known;

        for q in 0..4 {
            let (i, j) = match q {
                0 => (ii - 1, jj),
                1 => (ii, jj - 1),
                2 => (ii + 1, jj),
                _ => (ii, jj + 1),
            };
            // Skip the 1-pixel padded border (border is KNOWN anyway).
            if i <= 0 || j <= 0 || i >= rows - 1 || j >= cols - 1 {
                continue;
            }
            if f[(i * cols + j) as usize] == INSIDE {
                let dist = min4(
                    fast_marching_solve(i - 1, j, i, j - 1, f, t, cols),
                    fast_marching_solve(i + 1, j, i, j - 1, f, t, cols),
                    fast_marching_solve(i - 1, j, i, j + 1, f, t, cols),
                    fast_marching_solve(i + 1, j, i, j + 1, f, t, cols),
                );
                t[(i * cols + j) as usize] = dist;
                f[(i * cols + j) as usize] = BAND;
                heap.push(i, j, dist);
            }
        }
    }

    if negate {
        for i in 0..(rows * cols) as usize {
            if f[i] == CHANGE {
                f[i] = KNOWN;
            }
        }
    }
}

/// Cross (3x3) dilation of a padded mask -> band mask (INSIDE = part of band).
fn cross_dilate(mask: &[u8], rows: i32, cols: i32) -> Vec<u8> {
    let n = (rows * cols) as usize;
    let mut out = vec![0u8; n];
    let dy = [-1i32, 0, 0, 0, 1];
    let dx = [0i32, -1, 0, 1, 0];
    for i in 1..rows - 1 {
        for j in 1..cols - 1 {
            let mut v = 0u8;
            for k in 0..5 {
                let y = i + dy[k];
                let x = j + dx[k];
                if mask[(y * cols + x) as usize] == INSIDE {
                    v = INSIDE;
                    break;
                }
            }
            out[(i * cols + j) as usize] = v;
        }
    }
    out
}

/// Rectangular dilation of a padded mask -> region mask.
fn rect_dilate(mask: &[u8], rows: i32, cols: i32, r: i32) -> Vec<u8> {
    let n = (rows * cols) as usize;
    let mut tmp = vec![0u8; n];
    // horizontal pass
    for i in 0..rows {
        for j in 0..cols {
            let j0 = (j - r).max(0);
            let j1 = (j + r).min(cols - 1);
            let mut v = 0u8;
            for x in j0..=j1 {
                if mask[(i * cols + x) as usize] == INSIDE {
                    v = INSIDE;
                    break;
                }
            }
            tmp[(i * cols + j) as usize] = v;
        }
    }
    // vertical pass
    let mut out = vec![0u8; n];
    for j in 0..cols {
        for i in 0..rows {
            let i0 = (i - r).max(0);
            let i1 = (i + r).min(rows - 1);
            let mut v = 0u8;
            for y in i0..=i1 {
                if tmp[(y * cols + j) as usize] == INSIDE {
                    v = INSIDE;
                    break;
                }
            }
            out[(i * cols + j) as usize] = v;
        }
    }
    out
}

/// `out` is HxW uint8; `f`/`t` are (H+2)x(W+2). (i,j) and (k,l) are padded coords.
fn telea_inpaint_fmm(
    f: &mut [u8],
    t: &[f32],
    out: &mut [u8],
    range: i32,
    heap: &mut PriorityQueue,
    rows: i32,
    cols: i32,
    w0: i32,
) {
    // out index helper: (y_out, x_out) are 0-based indices into the HxW output image.
    let oidx = |y: i32, x: i32| -> usize { ((y * w0 + x) * 3) as usize };

    while let Some((ii, jj, _)) = heap.pop() {
        f[(ii * cols + jj) as usize] = KNOWN;

        for q in 0..4 {
            let (i, j) = match q {
                0 => (ii - 1, jj),
                1 => (ii, jj - 1),
                2 => (ii + 1, jj),
                _ => (ii, jj + 1),
            };
            // Border pixels are KNOWN; skip them before any access.
            if i <= 0 || j <= 0 || i >= rows - 1 || j >= cols - 1 {
                continue;
            }
            if f[(i * cols + j) as usize] != INSIDE {
                continue;
            }

            let dist = min4(
                fast_marching_solve(i - 1, j, i, j - 1, f, t, cols),
                fast_marching_solve(i + 1, j, i, j - 1, f, t, cols),
                fast_marching_solve(i - 1, j, i, j + 1, f, t, cols),
                fast_marching_solve(i + 1, j, i, j + 1, f, t, cols),
            );

            // gradT: distance-field gradient at (i,j) for each colour
            let mut grad_t: [[f32; 2]; 3] = [[0.0; 2]; 3];
            for color in 0..3 {
                // x
                grad_t[color][0] = if f[(i * cols + (j + 1)) as usize] != INSIDE {
                    if f[(i * cols + (j - 1)) as usize] != INSIDE {
                        (t[(i * cols + (j + 1)) as usize] - t[(i * cols + (j - 1)) as usize]) * 0.5
                    } else {
                        t[(i * cols + (j + 1)) as usize] - t[(i * cols + j) as usize]
                    }
                } else if f[(i * cols + (j - 1)) as usize] != INSIDE {
                    t[(i * cols + j) as usize] - t[(i * cols + (j - 1)) as usize]
                } else {
                    0.0
                };
                // y
                grad_t[color][1] = if f[((i + 1) * cols + j) as usize] != INSIDE {
                    if f[((i - 1) * cols + j) as usize] != INSIDE {
                        (t[((i + 1) * cols + j) as usize] - t[((i - 1) * cols + j) as usize]) * 0.5
                    } else {
                        t[((i + 1) * cols + j) as usize] - t[(i * cols + j) as usize]
                    }
                } else if f[((i - 1) * cols + j) as usize] != INSIDE {
                    t[(i * cols + j) as usize] - t[((i - 1) * cols + j) as usize]
                } else {
                    0.0
                };
            }

            let mut ia = [0.0f32; 3];
            let mut jx = [0.0f32; 3];
            let mut jy = [0.0f32; 3];
            let mut s = [1.0e-20f32; 3];

            for k in (i - range)..=(i + range) {
                if k <= 0 || k >= rows - 1 {
                    continue;
                }
                // clamped output-row indices for safe gradient access
                let km = (k - 1 + if k == 1 { 1 } else { 0 }) as i32;
                let kp = (k - 1 - if k == rows - 2 { 1 } else { 0 }) as i32;
                for l in (j - range)..=(j + range) {
                    if l <= 0 || l >= cols - 1 {
                        continue;
                    }
                    if f[(k * cols + l) as usize] == INSIDE {
                        continue;
                    }
                    let dy = l - j;
                    let dx = k - i;
                    if dy * dy + dx * dx > range * range {
                        continue;
                    }

                    let ry = (i - k) as f32;
                    let rx = (j - l) as f32;
                    let r_len_sq = rx * rx + ry * ry;
                    let r_len = r_len_sq.sqrt();
                    let dst = 1.0 / (r_len_sq * r_len);
                    let lev = 1.0 / (1.0 + (t[(k * cols + l) as usize] - t[(i * cols + j) as usize]).abs());

                    // clamped output-col indices
                    let lm = (l - 1 + if l == 1 { 1 } else { 0 }) as i32;
                    let lp = (l - 1 - if l == cols - 2 { 1 } else { 0 }) as i32;

                    for color in 0..3 {
                        let dir = rx * grad_t[color][0] + ry * grad_t[color][1];
                        let dir = if dir.abs() <= 0.01 { 0.000001f32 } else { dir };
                        let wgt = (dst * lev * dir).abs();

                        // image gradient at neighbour (k,l), from live out buffer
                        let grad_i_x = if f[(k * cols + (l + 1)) as usize] != INSIDE {
                            if f[(k * cols + (l - 1)) as usize] != INSIDE {
                                (out[oidx(km, lp + 1) + color] as f32
                                    - out[oidx(km, lm - 1) + color] as f32)
                                    * 2.0
                            } else {
                                out[oidx(km, lp + 1) + color] as f32
                                    - out[oidx(km, lm) + color] as f32
                            }
                        } else if f[(k * cols + (l - 1)) as usize] != INSIDE {
                            out[oidx(km, lp) + color] as f32
                                - out[oidx(km, lm - 1) + color] as f32
                        } else {
                            0.0
                        };

                        let grad_i_y = if f[((k + 1) * cols + l) as usize] != INSIDE {
                            if f[((k - 1) * cols + l) as usize] != INSIDE {
                                (out[oidx(kp + 1, lm) + color] as f32
                                    - out[oidx(km - 1, lm) + color] as f32)
                                    * 2.0
                            } else {
                                out[oidx(kp + 1, lm) + color] as f32
                                    - out[oidx(km, lm) + color] as f32
                            }
                        } else if f[((k - 1) * cols + l) as usize] != INSIDE {
                            out[oidx(km, lm) + color] as f32
                                - out[oidx(km - 1, lm) + color] as f32
                        } else {
                            0.0
                        };

                        ia[color] += wgt * out[oidx(k - 1, l - 1) + color] as f32;
                        jx[color] -= wgt * grad_i_x * rx;
                        jy[color] -= wgt * grad_i_y * ry;
                        s[color] += wgt;
                    }
                }
            }

            let out_i = oidx(i - 1, j - 1);
            for color in 0..3 {
                let sat = ia[color] / s[color]
                    + (jx[color] + jy[color])
                        / ((jx[color] * jx[color] + jy[color] * jy[color]).sqrt() + 1.0e-20);
                out[out_i + color] = clamp_u8(sat);
            }

            f[(i * cols + j) as usize] = BAND;
            heap.push(i, j, dist);
        }
    }
}

/// OpenCV TELEA inpaint. `rgb_f32` H*W*3 float (0..255); `mask` H*W (nonzero = hole).
/// Returns inpainted H*W*3 float.
pub fn telea_inpaint(rgb_f32: &[f32], mask: &[u8], h: usize, w: usize, radius: i32) -> Vec<f32> {
    let range = radius.clamp(1, 100);
    let rows = (h + 2) as i32;
    let cols = (w + 2) as i32;
    let n = (rows * cols) as usize;

    // 8-bit working output image (matches OpenCV uchar processing)
    let mut out_u8 = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let o = (y * w + x) * 3;
            out_u8[o] = clamp_u8(rgb_f32[o]);
            out_u8[o + 1] = clamp_u8(rgb_f32[o + 1]);
            out_u8[o + 2] = clamp_u8(rgb_f32[o + 2]);
        }
    }

    // padded state/distance arrays
    let mut f = vec![KNOWN; n];
    let mut t = vec![1.0e6f32; n];
    let mut mask_pad = vec![KNOWN; n];

    // copy input mask into padded array with 1-pixel border
    let cols_u = cols as usize;
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] != 0 {
                mask_pad[(y + 1) * cols_u + (x + 1)] = INSIDE;
            }
        }
    }
    // border already KNOWN (0)

    f.copy_from_slice(&mask_pad);

    // cross-dilated narrow band
    let band = cross_dilate(&mask_pad, rows, cols);
    let mut heap = PriorityQueue::new();
    for i in 0..rows {
        for j in 0..cols {
            let idx = (i * cols + j) as usize;
            if band[idx] == INSIDE && mask_pad[idx] != INSIDE {
                f[idx] = BAND;
                t[idx] = 0.0;
                heap.push(i, j, 0.0);
            }
        }
    }

    // TELEA: precompute distances on the range-dilated region around the hole
    let out_region = rect_dilate(&mask_pad, rows, cols, range);
    let mut out_heap = PriorityQueue::new();
    for i in 0..rows {
        for j in 0..cols {
            let idx = (i * cols + j) as usize;
            if f[idx] == BAND {
                out_heap.push(i, j, 0.0);
            }
        }
    }
    let mut out_region_mask = vec![0u8; n];
    for i in 0..rows {
        for j in 0..cols {
            let idx = (i * cols + j) as usize;
            if out_region[idx] == INSIDE && f[idx] != BAND {
                out_region_mask[idx] = INSIDE;
            }
        }
    }
    // temporarily mark out_region INSIDE so calc_fmm propagates into it
    let f_backup = f.clone();
    for i in 0..n {
        if out_region_mask[i] == INSIDE {
            f[i] = INSIDE;
        }
    }
    calc_fmm(&mut f, &mut t, &mut out_heap, rows, cols, true);
    // restore states: band stays BAND, out_region becomes KNOWN, original hole stays INSIDE
    f.copy_from_slice(&f_backup);

    // Run the actual TELEA fill using the band heap
    telea_inpaint_fmm(
        &mut f, &t, &mut out_u8, range, &mut heap, rows, cols, w as i32,
    );

    // convert back to f32
    let mut out = vec![0.0f32; h * w * 3];
    for i in 0..h * w * 3 {
        out[i] = out_u8[i] as f32;
    }
    out
}
