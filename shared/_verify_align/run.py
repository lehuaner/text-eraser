"""wasm vs original cv2 pipeline alignment harness.

Runs the ORIGINAL cv2 `_erase_deglow_v2` fallback (forced: _shared_core core disabled)
and the wasm `erase_text_glyphs` on IDENTICAL inputs, and reports per-channel max_diff
for clean / zone / fill(mask_filled) / result. This isolates where the Rust port diverges
from the original Python (stages vs fill algorithm).
"""
import os, sys, time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "text_eraser"))

from text_eraser.text_select import (
    _deglow_full_green_v2, _fill_bright_near_mask, _absorb_zone_bright_core,
)
from text_eraser.eraser import _residual_green, _dark_source_exclude, _run_fill
import text_eraser._shared_core as sc


def build_input(H=100, W=140):
    rng = np.random.default_rng(12345)
    bg = np.zeros((H, W, 3), np.float32)
    for c, (base, grad) in enumerate([(205, 18), (188, -10), (165, 6)]):
        g = base + grad * (np.arange(H)[:, None] / float(H))
        bg[:, :, c] = g
    bg += (rng.random((H, W, 3)).astype(np.float32) - 0.5) * 8.0
    rgb = bg.copy()
    # green glow blob
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H * 0.45, W * 0.5
    ry, rx = H * 0.28, W * 0.33
    glow = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    rgb[glow, 0] += 12.0
    rgb[glow, 1] += 72.0
    rgb[glow, 2] += 6.0
    # bright white core inside glow (outside text) -> exercises zone absorb
    rgb[40:46, 60:70, :] = 235.0
    # text masks (bars)
    tmask = np.zeros((H, W), np.uint8)
    for y0 in (18, 30, 42):
        tmask[y0:y0 + 4, 20:60] = 255
    tmask[25:35, 28:40] = 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # tm_clean: re-detect on deglowed image (here: same as tmask for a controlled test)
    tm_clean = tmask.copy()
    return rgb, tmask, tm_clean


def cv2_ref(rgb, tmask, tm_clean, strength, zone_ratio, zone_expand, protect_px,
            chroma_keep, edge, direction):
    clean0, _, zone0 = _deglow_full_green_v2(
        rgb, tmask, strength=strength, zone_ratio=zone_ratio,
        zone_expand=zone_expand, protect_px=protect_px,
        deglow_chroma_keep=bool(chroma_keep), return_zone=True)
    mask = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = _fill_bright_near_mask(clean0, mask)
    mask = _absorb_zone_bright_core(clean0, rgb, mask, zone0, min_rgb_lo=100)
    sample_exclude = _residual_green(clean0, mask)
    if zone0 is not None and bool((zone0 > 0).any()):
        dx = _dark_source_exclude(clean0, mask)
        if dx is not None:
            sample_exclude = (dx | sample_exclude) if sample_exclude is not None else dx
    result, mask_filled, _ = _run_fill(
        clean0, mask, [], edge=edge, direction=direction, edge_aware=False,
        return_mask=True, t0=time.time(), sample_exclude=sample_exclude, soft_expand=0.0)
    return dict(clean=clean0, zone=zone0, fill=mask_filled, result=result)


def wasm_run(rgb, tmask, tm_clean, strength, zone_ratio, zone_expand, protect_px,
             chroma_keep, edge, direction, seed=0):
    core = sc._get_core()
    if core is None:
        raise RuntimeError("wasm core not available")
    H, W = rgb.shape[:2]
    rgb_f32 = rgb.astype(np.float32)
    res = core.erase_text_glyphs(
        rgb_f32, H, W, tmask, tm_clean, strength, zone_ratio, zone_expand,
        protect_px, chroma_keep, edge, float(direction if direction is not None else -1.0), seed)
    if res is None:
        raise RuntimeError("wasm erase_text_glyphs returned None")
    return dict(clean=res[2], zone=res[3], fill=res[1], result=res[0])


def maxdiff(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return None, f"shape {a.shape} vs {b.shape}"
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return int(d.max()), int((d > 0).sum())


def main():
    # cv2 reference calls pure-cv2 functions directly (no wasm involved).
    rgb, tmask, tm_clean = build_input()
    P = dict(strength=1.15, zone_ratio=0.6, zone_expand=10, protect_px=1,
             chroma_keep=1, edge=1, direction=None)
    ref = cv2_ref(rgb, tmask, tm_clean, **P)

    # wasm run uses the shared core via _shared_core.erase_text_glyphs
    wasm = wasm_run(rgb, tmask, tm_clean, **P)

    print(f"{'channel':8s} {'maxdiff':>8s} {'#diff_px':>10s}  note")
    for ch in ("clean", "zone", "fill", "result"):
        md, nd = maxdiff(ref[ch], wasm[ch])
        print(f"{ch:8s} {str(md):>8s} {str(nd):>10s}")
    # result diff split: inside fill region vs outside
    fill = wasm["fill"] > 0
    rc = np.asarray(ref["result"]).astype(np.int32)
    wc = np.asarray(wasm["result"]).astype(np.int32)
    d = np.abs(rc - wc)
    if fill.any():
        print("result maxdiff INSIDE fill :", int(d[fill].max()), " outside fill:",
              int(d[~fill].max()) if (~fill).any() else 0)


if __name__ == "__main__":
    main()
