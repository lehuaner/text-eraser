"""Verify the smooth-gradient TELEA pre-check is now ported into the wasm core.

Run twice:
  python verify_smooth_fill.py wasm   -> uses the wasm core (TEXTCORE_BACKEND=1)
  python verify_smooth_fill.py cv2    -> pure-cv2 Python reference (TEXTCORE_BACKEND=0)

Then `compare_smooth_fill.py` checks the two agree (both take the TELEA path on
smooth backgrounds) and that the wasm result equals direct cv2 INPAINT_TELEA.
"""
import os
import sys
import numpy as np

# make the repo root importable regardless of how the script is launched
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODE = sys.argv[1] if len(sys.argv) > 1 else "wasm"
if MODE == "cv2":
    os.environ["TEXTCORE_BACKEND"] = "0"

import cv2

OUT = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(OUT, exist_ok=True)


def smooth_img(h=220, w=220):
    """Gentle linear gradient (Sobel magnitude ~0.5 < flat_tex=15) -> pre-check fires."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = (xx / w) * 120.0 + 20.0          # 20..140 left->right
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = g + 6.0
    img[..., 1] = g
    img[..., 2] = g - 6.0
    return np.clip(img, 0, 255)


def textured_img(h=220, w=220):
    """High-frequency stripe texture (Sobel magnitude large) -> pre-check does NOT fire."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (xx / w) * 120.0 + 20.0
    stripe = (np.sin(xx / 2.0) * 60.0)          # strong horizontal stripes
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = base + stripe + 6.0
    img[..., 1] = base + stripe
    img[..., 2] = base + stripe - 6.0
    return np.clip(img, 0, 255)


def hole_mask(h=220, w=220):
    m = np.zeros((h, w), np.uint8)
    cv2.putText(m, "TEXT", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 6)
    return (m > 0).astype(np.uint8)


def crop_roi(img, m, margin=80):
    ys, xs = np.where(m)
    y0, y1 = ys.min() - margin, ys.max() + margin + 1
    x0, x1 = xs.min() - margin, xs.max() + margin + 1
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(img.shape[0], y1), min(img.shape[1], x1)
    return img[y0:y1, x0:x1].copy(), m[y0:y1, x0:x1].copy(), (y0, x0)


def main():
    for tag, gen in (("smooth", smooth_img), ("textured", textured_img)):
        img = gen()
        m = hole_mask()
        sub, subm, _ = crop_roi(img, m)
        if MODE == "wasm":
            from text_eraser._shared_core import patchmatch_inpaint_fill
            res = patchmatch_inpaint_fill(sub, subm, None, 7, -1.0, 0)
            res = np.ascontiguousarray(res, dtype=np.float32)
        else:
            from text_eraser import patch_fill
            res = patch_fill.inpaint(sub, subm, sample_mask=None, direction=None)
            res = res.astype(np.float32)
        np.save(os.path.join(OUT, f"{tag}_{MODE}.npy"), res)
        np.save(os.path.join(OUT, f"{tag}_sub.npy"), sub)
        np.save(os.path.join(OUT, f"{tag}_subm.npy"), subm)
        # also store the direct cv2 TELEA reference for the smooth case
        if MODE == "cv2" and tag == "smooth":
            telea = cv2.inpaint(np.clip(sub, 0, 255).astype(np.uint8),
                                subm, 3, cv2.INPAINT_TELEA).astype(np.float32)
            np.save(os.path.join(OUT, "smooth_cv2telea.npy"), telea)
        print(f"[{MODE}] {tag}: result shape={res.shape}, dtype={res.dtype}")


if __name__ == "__main__":
    main()
