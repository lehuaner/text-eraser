"""Full-pipeline fuzz: cv2 reference (_erase_deglow_v2 style) vs wasm erase_text_glyphs.

Reuses run.py comparison logic across many random inputs to find any divergence
in the de-glow port (clean/zone/fill/result).
"""
import os, sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "text_eraser"))

import run as R  # the run.py harness (cv2_ref, wasm_run, maxdiff)


def gen_input(rng, mode):
    H, W = int(rng.integers(60, 130)), int(rng.integers(60, 160))
    bg = np.zeros((H, W, 3), np.float32)
    if mode == 0:  # neutral bg, vertical gradient
        for c, (base, grad) in enumerate([(200, 15), (200, -5), (200, 3)]):
            bg[:, :, c] = base + grad * (np.arange(H)[:, None] / float(H))
    elif mode == 1:  # warm bg (R>G>B) with gradient
        for c, (base, grad) in enumerate([(212, 12), (190, -8), (158, 5)]):
            bg[:, :, c] = base + grad * (np.arange(H)[:, None] / float(H))
    elif mode == 2:  # cool/flat
        bg[:, :, 0] = 150; bg[:, :, 1] = 160; bg[:, :, 2] = 178
    elif mode == 3:  # bright layer on top + dark below (556-style)
        bg[:, :W//2, :] = 200.0
        bg[:, W//2:, :] = 90.0
    bg += (rng.random((H, W, 3)).astype(np.float32) - 0.5) * 6.0
    rgb = bg.copy()
    # random green glow blob(s)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(int(rng.integers(1, 3))):
        cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        ry = int(rng.integers(8, H // 2)); rx = int(rng.integers(8, W // 2))
        glow = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= (rng.random() * 0.6 + 0.4)
        rgb[glow, 0] += rng.uniform(5, 22)
        rgb[glow, 1] += rng.uniform(40, 95)
        rgb[glow, 2] += rng.uniform(2, 14)
    if rng.random() < 0.6:  # bright white core inside glow
        y0, x0 = int(rng.integers(0, H - 8)), int(rng.integers(0, W - 10))
        rgb[y0:y0 + 6, x0:x0 + 8, :] = 235.0
    tmask = np.zeros((H, W), np.uint8)
    if rng.random() < 0.95:
        for _ in range(int(rng.integers(1, 5))):
            y0 = int(rng.integers(0, H - 4)); x0 = int(rng.integers(0, W - 40))
            tmask[y0:y0 + 3, x0:x0 + 30] = 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, tmask


def main():
    rng = np.random.default_rng(20260902)
    mism = 0
    tested = 0
    worst = {}
    for it in range(4):
        mode = it % 4
        rgb, tmask = gen_input(rng, mode)
        P = dict(
            strength=float(rng.choice([0.8, 1.0, 1.15, 1.5])),
            zone_ratio=float(rng.choice([0.3, 0.6, 0.8, 1.0])),
            zone_expand=int(rng.choice([0, 5, 10, 24])),
            protect_px=1, chroma_keep=int(rng.choice([0, 1])),
            edge=int(rng.choice([0, 1, 3])), direction=None,
        )
        try:
            ref = R.cv2_ref(rgb, tmask, tmask, **P)
            wasm = R.wasm_run(rgb, tmask, tmask, **P)
        except Exception as e:
            print("ERR it=%d" % it, repr(e)[:200]); continue
        tested += 1
        for ch in ("clean", "zone", "fill", "result"):
            md, nd = R.maxdiff(ref[ch], wasm[ch])
            if md is None:
                continue
            key = (ch,)
            if key not in worst or md > worst[key][0]:
                worst[key] = (md, nd, it, mode, dict(P))
            if md > 0:
                mism += 1
                if mism <= 12:
                    print(f"[diff] it={it} mode={mode} ch={ch} maxdiff={md} #px={nd} P={P}")
                break
    print(f"\ntested={tested}")
    print("worst per channel:")
    for ch in ("clean", "zone", "fill", "result"):
        if (ch,) in worst:
            md, nd, it, mode, P = worst[(ch,)]
            print(f"  {ch:7s} maxdiff={md} #px={nd} (it={it} mode={mode} P={P})")
    if mism == 0:
        print("NO DIVERGENCE across all fuzzed inputs.")


if __name__ == "__main__":
    main()
