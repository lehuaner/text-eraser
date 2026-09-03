#!/usr/bin/env python3
"""Diagnose the erase outside-mask diff: split it into (a) the ellipse-3 FILL ring
(the region the erase pipeline legitimately fills in BOTH JS and Python — expected
PatchMatch divergence) and (b) the truly-untouched region (must be ~0)."""
import os, sys
import numpy as np, cv2

wd = sys.argv[1]
H, W = (int(x) for x in open(os.path.join(wd, "dims.txt")).read().split())
mask = np.fromfile(os.path.join(wd, "input.mask"), dtype=np.uint8).reshape(H, W)

ref = cv2.cvtColor(cv2.imread(os.path.join(wd, "reference_erase.png"), cv2.IMREAD_COLOR),
                   cv2.COLOR_BGR2RGB).astype(np.float32)
js = np.fromfile(os.path.join(wd, "out_erase.rgb"), dtype=np.float32).reshape(H, W, 3)

# ellipse-3 dilation == both pipelines' fill region
dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
d = np.abs(js - ref)

print(f"[erase] outside FILL (dilated) mask: max={d[dil == 0].max():.4f}  mean={d[dil == 0].mean():.5f}")
print(f"[erase] within FILL (dilated) mask: max={d[dil > 0].max():.2f}  mean={d[dil > 0].mean():.3f}")
print(f"[erase]   of which ORIGINAL 36x36 hole: max={d[mask > 0].max():.2f}  mean={d[mask > 0].mean():.3f}")
ring = (dil > 0) & (mask == 0)
print(f"[erase]   RING (dilated minus original): max={d[ring].max():.2f}  mean={d[ring].mean():.3f}  count={int(ring.sum())}")

# same for inpaint (fill region == original mask, no dilation)
refi = cv2.cvtColor(cv2.imread(os.path.join(wd, "reference_inpaint.png"), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB).astype(np.float32)
jsi = np.fromfile(os.path.join(wd, "out_inpaint.rgb"), dtype=np.float32).reshape(H, W, 3)
di = np.abs(jsi - refi)
print(f"[inpaint] outside FILL (original) mask: max={di[mask == 0].max():.4f}  mean={di[mask == 0].mean():.5f}")
print(f"[inpaint] within FILL (original) mask: max={di[mask > 0].max():.2f}  mean={di[mask > 0].mean():.3f}")
