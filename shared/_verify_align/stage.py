"""Stage-by-stage compare of wasm mask surgery vs the original cv2 pipeline.

Each stage is compared in ISOLATION: the SAME cv2-produced input is fed to both
the wasm function and the cv2 function, so a diff means the function itself diverges.
"""
import os, sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "text_eraser"))

from text_eraser.text_select import (
    _deglow_full_green_v2, _fill_bright_near_mask, _absorb_zone_bright_core,
)
import text_eraser._shared_core as sc


def build_input(H=100, W=140):
    rng = np.random.default_rng(12345)
    bg = np.zeros((H, W, 3), np.float32)
    for c, (base, grad) in enumerate([(205, 18), (188, -10), (165, 6)]):
        g = base + grad * (np.arange(H)[:, None] / float(H))
        bg[:, :, c] = g
    bg += (rng.random((H, W, 3)).astype(np.float32) - 0.5) * 8.0
    rgb = bg.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H * 0.45, W * 0.5
    ry, rx = H * 0.28, W * 0.33
    glow = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    rgb[glow, 0] += 12.0
    rgb[glow, 1] += 72.0
    rgb[glow, 2] += 6.0
    rgb[40:46, 60:70, :] = 235.0
    tmask = np.zeros((H, W), np.uint8)
    for y0 in (18, 30, 42):
        tmask[y0:y0 + 4, 20:60] = 255
    tmask[25:35, 28:40] = 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    tm_clean = tmask.copy()
    return rgb, tmask, tm_clean


def md(a, b):
    # binary-aware: compare set/unset (thresh >0), since cv2 uses 0/255 and wasm 0/1.
    a = (np.asarray(a) > 0).astype(np.int32)
    b = (np.asarray(b) > 0).astype(np.int32)
    d = np.abs(a - b)
    return int(d.max()), int((d > 0).sum())


def main():
    rgb, tmask, tm_clean = build_input()
    H, W = rgb.shape[:2]
    clean0, _, zone0 = _deglow_full_green_v2(
        rgb, tmask, strength=1.15, zone_ratio=0.6, zone_expand=10, protect_px=1,
        deglow_chroma_keep=True, return_zone=True)
    core = sc._get_core()
    if core is None:
        raise RuntimeError("no wasm core")

    # Stage 0: union
    cv_mask = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
    w_mask = core.dbg_mask_union(tmask, tm_clean, H, W)
    print("union         :", md(w_mask, cv_mask))

    # Stage 0b: close  -- feed cv_mask to both
    cv_close = cv2.morphologyEx(cv_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    w_close = core.dbg_mask_close(cv_mask, H, W)
    print("mask_close    :", md(w_close, cv_close))

    # Stage 1: fill_bright -- feed cv_close to both
    cv_fb = _fill_bright_near_mask(clean0, cv_close)
    w_fb = core.dbg_fill_bright(clean0.astype(np.float32), cv_close, H, W)
    print("fill_bright   :", md(w_fb, cv_fb))

    # Stage 2: absorb -- feed cv_fb to both
    cv_abs = _absorb_zone_bright_core(clean0, rgb, cv_fb, zone0, min_rgb_lo=100)
    w_abs = core.dbg_absorb(clean0.astype(np.float32), rgb.astype(np.float32), cv_fb, zone0, H, W)
    print("absorb        :", md(w_abs, cv_abs))

    # Final mask_filled (ellipse dilate) -- feed cv_abs to both
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv_mf = cv2.dilate(cv_abs, ek)
    w_mf = core.morphology(w_abs, H, W, ek, 3, 3, "dilate")
    print("mask_filled   :", md(w_mf, cv_mf))

    # full erase_text_glyphs vs cv2_ref (reuse run.py style)
    res = core.erase_text_glyphs(rgb.astype(np.float32), H, W, tmask, tm_clean,
                                 1.15, 0.6, 10, 1, 1, 1, -1.0, 0)
    print("erase fill    :", md(res[1], cv_mf))
    print("erase result  : see run.py for full compare")


if __name__ == "__main__":
    main()
