#!/usr/bin/env python3
"""Generate a Python reference for the browser smoke harness.

Mirrors the exact pipeline the browser module runs so the smoke test can measure a
pixel-level maxDiff against `textpatch` (scripts/smoke_universal.py reports ≈3).

Pipeline (matches browser/src/index.js eraseTextGlyphs + patchmatch.js inpaint):
  * inpaint reference : text_eraser.patch_fill.inpaint(rgb, mask, flat_span=40, flat_tex=15)
  * erase  reference : _deglow_faint_green (thr=6, near_r=24, g_lo=85, protect=1, s=1)
                       -> mask_filled = ellipse_dilate(mask, edge)
                       -> sample = 255 - mask_filled
                       -> patch_fill.inpaint(rgb, mask_filled, sample_mask=sample)

Outputs into --out: input.png, mask.png (fill), textmask.png, reference_inpaint.png,
reference_erase.png, config.json.

Run:
    python browser/smoke/gen_reference.py --image in.png --mask mask.png \
        --out browser/smoke/reference
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import cv2

# make the repo's text_eraser importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from text_eraser import patch_fill  # noqa: E402
from text_eraser import text_select  # noqa: E402


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise SystemExit(f"cannot read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 127).astype(np.uint8) * 255


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: str, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True, help="fill/text mask PNG (255=target)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edge", type=int, default=1)
    ap.add_argument("--deglow", action="store_true", default=True)
    ap.add_argument("--flat-span", type=int, default=40)
    ap.add_argument("--flat-tex", type=float, default=15.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rgb = load_rgb(args.image)
    mask = load_mask(args.mask)

    # 1) inpaint reference
    ref_inpaint = patch_fill.inpaint(
        rgb, mask, sample_mask=None, direction=None,
        flat_span=args.flat_span, flat_tex=args.flat_tex,
    )

    # 2) erase reference (mirror browser eraseTextGlyphs)
    work = rgb.copy()
    if args.deglow:
        work, _ = text_select._deglow_faint_green(
            work, mask, thr=6, near_r=24, g_lo=85,
            text_protect=1.0, strength=1.0,
        )
    k = max(1, args.edge * 2 + 1)
    if args.edge > 0:
        mask_filled = cv2.dilate(mask, ellipse(k))
    elif args.edge < 0:
        mask_filled = cv2.erode(mask, ellipse(-args.edge * 2 + 1))
    else:
        mask_filled = mask.copy()
    sample = (255 - mask_filled).astype(np.uint8)
    ref_erase = patch_fill.inpaint(
        work, mask_filled, sample_mask=sample, direction=None,
        flat_span=args.flat_span, flat_tex=args.flat_tex,
    )

    save_rgb(os.path.join(args.out, "input.png"), rgb)
    cv2.imwrite(os.path.join(args.out, "mask.png"), mask)
    cv2.imwrite(os.path.join(args.out, "textmask.png"), mask)
    save_rgb(os.path.join(args.out, "reference_inpaint.png"), ref_inpaint)
    save_rgb(os.path.join(args.out, "reference_erase.png"), ref_erase)

    cfg = {
        "image": os.path.basename(args.image),
        "edge": args.edge,
        "deglow": args.deglow,
        "flatSpan": args.flat_span,
        "flatTex": args.flat_tex,
        "expectedMaxDiff": 3,
        "note": "browser maxDiff should be <= expectedMaxDiff vs these reference PNGs",
    }
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("wrote reference into", args.out)


if __name__ == "__main__":
    main()
