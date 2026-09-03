#!/usr/bin/env python3
"""Generate a synthetic TEXTURED image + hole mask so the flat-gradient TELEA
branch does NOT fire (ring median gradient >= flatTex=15) and the PatchMatch
core runs on both the Python reference and the JS port.

Usage: python gen_synth.py <out_dir>
Writes input.png (RGB) and mask.png (255 = hole).
"""
from __future__ import annotations
import os
import sys
import numpy as np


def main() -> None:
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    H = W = 160
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # structured high-frequency texture (stripes) + gentle gradient
    base = (40 * np.sin(xx / 2.0) + 35 * np.sin(yy / 2.3)
            + 25 * np.sin((xx + yy) / 3.0) + (xx + yy) * 0.15 + 128)
    img = np.stack([base, base * 0.96 + 8, base * 0.9 + 4], axis=-1)
    rng = np.random.default_rng(7)
    img = img + rng.normal(0, 10, size=img.shape)   # noise keeps tex high
    img = np.clip(img, 0, 255).astype(np.uint8)

    mask = np.zeros((H, W), np.uint8)
    mask[62:98, 62:98] = 255   # 36x36 hole, centered

    import cv2
    cv2.imwrite(os.path.join(out, "input.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out, "mask.png"), mask)
    print("wrote synthetic", H, "x", W, "hole 36x36 ->", out)


if __name__ == "__main__":
    main()
