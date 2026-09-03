"""Compare Rust reconstruction primitives vs cv2 on a divergent input."""
import os, sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import text_eraser._shared_core as sc
core = sc._get_core()

def gen_input_it0():
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
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, tmask

rgb, tmask = gen_input_it0()
H, W = rgb.shape[:2]
print("rgb", rgb.shape)

# compute cv2 zone (to feed distance_transform)
from text_eraser.text_select import _deglow_full_green_v2
clean0, core0, zone0 = _deglow_full_green_v2(rgb, tmask, strength=1.0, zone_ratio=0.6, zone_expand=0, protect_px=1, deglow_chroma_keep=False, return_zone=True)
print("cv2 zone px", int(zone0.sum()), "zone.sum/HW=", float(zone0.sum())/(H*W))

# --- distance_transform comparison ---
cv_dt = cv2.distanceTransform((~zone0).astype(np.uint8), cv2.DIST_L2, 5)
w_dt = core.dbg_dist_l2((~zone0).astype(np.uint8), H, W)  # distance to nearest nonzero of (~zone0) = dist to zone
w_dt = np.asarray(w_dt, np.float32).reshape(H, W)
dd = np.abs(cv_dt.astype(np.float32) - w_dt)
print(f"[distance_transform] maxdiff={dd.max():.4f} #px={int((dd>1e-3).sum())}  cv_range=[{cv_dt.min():.3f},{cv_dt.max():.3f}] w_range=[{w_dt.min():.3f},{w_dt.max():.3f}]")

# --- gaussian_blur comparison ---
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
cv_g = cv2.GaussianBlur(gray, (0, 0), 2.0)
w_g = core.dbg_gauss(rgb.astype(np.float32), H, W, 2.0)
w_g = np.asarray(w_g, np.float32).reshape(H, W)
dg = np.abs(cv_g - w_g)
print(f"[gaussian_blur σ=2] maxdiff={dg.max():.4f} #px={int((dg>0.5).sum())}  cv_range=[{cv_g.min():.2f},{cv_g.max():.2f}] w_range=[{w_g.min():.2f},{w_g.max():.2f}]")
# also report what cv2 kernel size is
ks = cv2.getGaussianKernel(2.0, -1)
print("cv2 getGaussianKernel(2.0) len =", len(ks))
