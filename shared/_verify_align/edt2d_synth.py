"""Localize the 2D exact-EDT bug with a small synthetic multi-source mask + brute-force gt."""
import sys, os
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import text_eraser._shared_core as sc
core = sc._get_core()

H, W = 25, 25
mask = np.zeros((H, W), np.uint8)
# two source blocks
mask[8:12, 8:12] = 1
mask[2:4, 18:21] = 1
# scattered sources
mask[20, 3] = 1; mask[22, 22] = 1; mask[5, 2] = 1
yy, xx = np.mgrid[0:H, 0:W]
gt = np.full((H, W), np.inf, np.float32)
sy, sx = np.where(mask > 0)
for (y, x) in zip(sy, sx):
    gt = np.minimum(gt, np.sqrt((yy - y) ** 2 + (xx - x) ** 2).astype(np.float32))

wv = core.dbg_dist_l2(mask, H, W).reshape(H, W)
print("wasm vs gt maxdiff =", np.abs(wv - gt).max())
print("wasm max =", wv.max(), " gt max =", gt.max())
# print where capped (wasm < gt significantly)
bad = np.abs(wv - gt) > 0.5
print("bad count:", int(bad.sum()))
if bad.sum():
    ys, xs = np.where(bad)
    for (y, x) in zip(ys[:15], xs[:15]):
        print(f"  ({y},{x}) wasm={wv[y,x]:.2f} gt={gt[y,x]:.2f}")
