"""Reproduce EDT cap with a large zero-blob (source sea + one big island)."""
import sys, os
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import text_eraser._shared_core as sc
core = sc._get_core()

H, W = 68, 152
mask = np.ones((H, W), np.uint8)        # source sea
mask[20:55, 40:110] = 0                  # big zero island (zone), 35x70
yy, xx = np.mgrid[0:H, 0:W]
gt = np.full((H, W), np.inf, np.float32)
sy, sx = np.where(mask > 0)
for (y, x) in zip(sy, sx):
    gt = np.minimum(gt, np.sqrt((yy - y) ** 2 + (xx - x) ** 2).astype(np.float32))
wv = core.dbg_dist_l2(mask, H, W).reshape(H, W)
print("big island: wasm max=%.2f gt max=%.2f  wasm vs gt maxdiff=%.4f" % (wv.max(), gt.max(), np.abs(wv-gt).max()))
print("  wasm min/median/max of gt where gt>0: ", "%.1f/%.1f/%.1f" % (gt[gt>0].min(), np.median(gt[gt>0]), gt.max()))
print("  wasm min/median/max of wv where gt>0: ", "%.1f/%.1f/%.1f" % (wv[gt>0].min(), np.median(wv[gt>0]), wv.max()))

# also test: many small islands (opposite structure)
mask2 = np.ones((H, W), np.uint8)
for cy in range(10, H, 15):
    for cx in range(10, W, 20):
        mask2[cy:cy+3, cx:cx+3] = 0
gt2 = np.full((H, W), np.inf, np.float32)
sy, sx = np.where(mask2 > 0)
for (y, x) in zip(sy, sx):
    gt2 = np.minimum(gt2, np.sqrt((yy - y) ** 2 + (xx - x) ** 2).astype(np.float32))
wv2 = core.dbg_dist_l2(mask2, H, W).reshape(H, W)
print("many small islands: wasm max=%.2f gt max=%.2f wasm vs gt maxdiff=%.4f" % (wv2.max(), gt2.max(), np.abs(wv2-gt2).max()))
