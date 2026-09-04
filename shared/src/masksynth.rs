//! Text-mask synthesis — single source of truth for turning a raw text mask into the
//! fill + sample masks that feed PatchMatch.
//!
//! Mirrors the browser `eraseTextGlyphs` contract exactly:
//!   filled = ellipse-(dilate|erode)(textMask, edge)   (edge=0 => identity)
//!   sample = whole - filled
//!   (optional) filled &= limit
//! Both the browser Worker and the Python backend call this, so the fill/sample
//! regions are byte-identical by construction — the third divergence root cause
//! (different mask synthesis) is eliminated.

/// ELLIPSE structuring element of diameter `ksize`, matching
/// cv2.getStructuringElement(MORPH_ELLIPSE,(ksize,ksize)) and the browser's
/// `kernBitmap('ellipse', ksize)` / `morphOffsets('ellipse', ksize)`.
fn ellipse_kernel(ksize: i32) -> Vec<u8> {
    let ca = (ksize as f64 - 1.0) / 2.0;
    let mut k = vec![0u8; (ksize * ksize) as usize];
    for dy in 0..ksize {
        for dx in 0..ksize {
            let ox = (dx as f64 - ca).abs();
            let oy = (dy as f64 - ca).abs();
            if ox * ox + oy * oy <= ca * ca + 1e-6 {
                k[(dy * ksize + dx) as usize] = 1;
            }
        }
    }
    k
}

#[no_mangle]
pub extern "C" fn synthesize_masks(
    text_mask_ptr: *const u8,
    h: i32,
    w: i32,
    edge: i32,
    limit_ptr: *const u8,
    has_limit: i32,
    out_fill_ptr: *mut u8,
    out_sample_ptr: *mut u8,
) {
    let h = h as usize;
    let w = w as usize;
    let n = h * w;
    let tm = unsafe { std::slice::from_raw_parts(text_mask_ptr, n) };
    let fill = unsafe { std::slice::from_raw_parts_mut(out_fill_ptr, n) };
    let sample = unsafe { std::slice::from_raw_parts_mut(out_sample_ptr, n) };

    let mut filled = vec![0u8; n];
    for i in 0..n {
        filled[i] = if tm[i] > 0 { 1 } else { 0 };
    }

    if edge != 0 {
        let ksize = if edge > 0 { edge * 2 + 1 } else { -edge * 2 + 1 };
        let is_dilate = edge > 0;
        let kern = ellipse_kernel(ksize);
        let kh = ksize as usize;
        let kw = ksize as usize;
        let ax = ksize / 2;
        let ay = ksize / 2;
        let src = filled.clone();
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let mut res = !is_dilate; // dilate starts false, erode starts true
                'kern: for ky in 0..kh as i32 {
                    for kx in 0..kw as i32 {
                        if kern[(ky as usize) * kw + (kx as usize)] == 0 {
                            continue;
                        }
                        let nx = x + (kx - ax);
                        let ny = y + (ky - ay);
                        // cv2 ignores out-of-bounds kernel cells (no BORDER_CONSTANT fill).
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let bit = src[(ny as usize) * w + (nx as usize)] != 0;
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
                filled[(y as usize) * w + (x as usize)] = if res { 1 } else { 0 };
            }
        }
    }

    if has_limit != 0 {
        let lim = unsafe { std::slice::from_raw_parts(limit_ptr, n) };
        for i in 0..n {
            if lim[i] == 0 {
                filled[i] = 0;
            }
        }
    }

    for i in 0..n {
        fill[i] = if filled[i] != 0 { 255 } else { 0 };
        sample[i] = if filled[i] != 0 { 0 } else { 255 };
    }
}
