"""Proper EDT comparison: identical mask to cv2 and wasm. First confirm cv2 semantics
with a known case, then compare on the real divergent input's zone."""
import sys, os
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import text_eraser._shared_core as sc
core = sc._get_core()

# Known case: single seed at (0,0). distance to nearest seed should be euclidean.
H, W = 7, 9
seed = np.zeros((H, W), np.uint8); seed[0, 0] = 1
# build ground-truth euclidean distance to (0,0) manually
yy, xx = np.mgrid[0:H, 0:W]
gt = np.sqrt((yy-0)**2 + (xx-0)**2).astype(np.float32)
cv = cv2.distanceTransform(seed, cv2.DIST_L2, 5).astype(np.float32)
wv = core.dbg_dist_l2(seed, H, W).reshape(H, W)
print("single-seed@(0,0):")
print("  cv2 :", np.round(cv, 2).ravel()[:9])
print("  wasm:", np.round(wv, 2).ravel()[:9])
print("  gt  :", np.round(gt, 2).ravel()[:9])
print("  cv2 vs gt maxdiff:", np.abs(cv-gt).max(), " wasm vs gt maxdiff:", np.abs(wv-gt).max())

# Now the real divergent input: reproduce it=0 zone
def gen_it0():
    rng = np.random.default_rng(20260902)
    H, W = int(rng.integers(60, 130)), int(rng.integers(60, 160))
    bg = np.zeros((H, W, 3), np.float32)
    for c, (base, grad) in enumerate([(200, 15), (200, -5), (200, 3)]):
        bg[:, :, c] = base + grad * (np.arange(H)[:, None] / float(H))
    bg += (rng.random((H, W, 3)).astype(np.float32) - 0.5) * 6.0
    rgb = bg.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(int(rng.integers(1, 3))):
        cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        ry = int(rng.integers(8, H // 2)); rx = int(rng.integers(8, W // 2))
        glow = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= (rng.random() * 0.6 + 0.4)
        rgb[glow, 0] += rng.uniform(5, 22); rgb[glow, 1] += rng.uniform(40, 95); rgb[glow, 2] += rng.uniform(2, 14)
    if rng.random() < 0.6:
        y0, x0 = int(rng.integers(0, H - 8)), int(rng.integers(0, W - 10))
        rgb[y0:y0 + 6, x0:x0 + 8, :] = 235.0
    tmask = np.zeros((H, W), np.uint8)
    if rng.random() < 0.95:
        for _ in range(int(rng.integers(1, 5))):
            y0 = int(rng.integers(0, H - 4)); x0 = int(rng.integers(0, W - 40))
            tmask[y0:y0 + 3, x0:x0 + 30] = 255
    return np.clip(rgb, 0, 255).astype(np.uint8), tmask

from text_eraser.text_select import _deglow_full_green_v2
rgb, tmask = gen_it0()
H, W = rgb.shape[:2]
yy, xx = np.mgrid[0:H, 0:W]
clean0, core0, zone0 = _deglow_full_green_v2(rgb, tmask, strength=1.0, zone_ratio=0.6, zone_expand=0, protect_px=1, deglow_chroma_keep=False, return_zone=True)
# distance from each non-zone pixel to nearest zone pixel (matches Python's (~zone) call)
cv2d = cv2.distanceTransform((~zone0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
wv2d = core.dbg_dist_l2((~zone0).astype(np.uint8), H, W).reshape(H, W)
# ground truth via brute force
ys, xs = np.where(zone0>0)
gt2 = np.full((H,W), np.inf, np.float32)
for (zy,zx) in zip(ys, xs):
    gt2 = np.minimum(gt2, np.sqrt((yy-zy)**2 + (xx-zx)**2).astype(np.float32))
print("\nreal zone (distance to nearest zone pixel):")
print("  cv2 max=%.2f wasm max=%.2f gt max=%.2f" % (cv2d.max(), wv2d.max(), gt2.max()))
print("  wasm vs gt maxdiff=%.4f #px=%.0f" % (np.abs(wv2d-gt2).max(), (np.abs(wv2d-gt2)>1e-3).sum()))
print("  cv2 vs gt maxdiff=%.4f #px=%.0f" % (np.abs(cv2d-gt2).max(), (np.abs(cv2d-gt2)>1e-3).sum()))
