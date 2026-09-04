#!/usr/bin/env python3
"""Pixel-level comparator.

Compares a JS-produced raw RGB (float32 H*W*3) against the Python reference PNG.

Usage:
    python browser/smoke/compare.py <reference.png> <js_output.rgb> <work_dir>

Reports whole-image, per-channel, and within-mask (the original text region)
max_abs_diff. The outside-mask pixels must be ~0 (only the hole is filled).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import cv2


def load_rgb(p: str) -> np.ndarray:
    b = cv2.imread(p, cv2.IMREAD_COLOR)
    return cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32)


def main() -> None:
    ref_png = sys.argv[1]
    js_raw = sys.argv[2]
    wd = sys.argv[3]

    H, W = (int(x) for x in open(os.path.join(wd, "dims.txt")).read().split())
    mask = np.fromfile(os.path.join(wd, "input.mask"), dtype=np.uint8).reshape(H, W)

    ref = load_rgb(ref_png)
    js = np.fromfile(js_raw, dtype=np.float32).reshape(H, W, 3)
    if js.shape != ref.shape:
        print(f"SHAPE MISMATCH ref={ref.shape} js={js.shape}")
        return

    d = np.abs(js - ref)
    print(f"[whole image]       max={d.max():.2f}  mean={d.mean():.3f}  median={np.median(d):.3f}")
    for c, nm in enumerate("RGB"):
        print(f"   {nm}: max={d[..., c].max():.2f}  mean={d[..., c].mean():.3f}")

    md = d[mask > 0]
    if md.size:
        print(f"[within text mask]  count={md.size}  max={md.max():.2f}  mean={md.mean():.3f}")
    known = d[mask == 0]
    if known.size:
        print(f"[outside mask]      max={known.max():.4f}  mean={known.mean():.5f}  (must be ~0)")


if __name__ == "__main__":
    main()
