#!/usr/bin/env python3
"""Prep step for the pixel comparison.

Converts the Python-generated reference PNGs into raw arrays the Node runner can
read, and copies the reference PNGs so the comparator can read both sides.

Usage:
    python browser/smoke/_prep.py <ref_dir> <work_dir>

<ref_dir> must contain: input.png, mask.png, reference_inpaint.png, reference_erase.png
"""
from __future__ import annotations
import os
import sys
import numpy as np
import cv2


def main() -> None:
    ref = sys.argv[1]
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)

    inp = cv2.imread(os.path.join(ref, "input.png"), cv2.IMREAD_COLOR)
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)
    mask = cv2.imread(os.path.join(ref, "mask.png"), cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 127).astype(np.uint8)
    H, W = inp.shape[:2]

    inp.tofile(os.path.join(out, "input.rgb"))
    mask.tofile(os.path.join(out, "input.mask"))
    with open(os.path.join(out, "dims.txt"), "w", encoding="utf-8") as f:
        f.write(f"{H} {W}")

    for name in ("reference_inpaint.png", "reference_erase.png"):
        p = os.path.join(ref, name)
        if os.path.exists(p):
            cv2.imwrite(os.path.join(out, name), cv2.imread(p))

    print(f"prepped {H}x{W} -> {out}")


if __name__ == "__main__":
    main()
