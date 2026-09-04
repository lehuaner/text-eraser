"""On a REAL image: (1) verify EDT with CORRECT orientation (distance to zone),
(2) compare gaussian_blur sigma=2 (the detail term) cv2 vs wasm."""
import sys, os, json
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import text_eraser._shared_core as sc
from text_eraser.text_select import _deglow_full_green_v2
core = sc._get_core()

HID = "1788062081665"
raw = open(os.path.join(ROOT, "data", "history", HID, "orig.bin"), "rb").read()
bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
meta = json.load(open(os.path.join(ROOT, "data", "history", HID, "meta.json")))
m = np.zeros((H, W), np.uint8)
for b in meta["boxes"]:
    m[b["y0"]:b["y1"], b["x0"]:b["x1"]] = 255
tmask = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
clean0, core0, zone0 = _deglow_full_green_v2(rgb, tmask, strength=1.0, zone_ratio=0.6,
                                             zone_expand=10, protect_px=1,
                                             deglow_chroma_keep=True, return_zone=True)
print("zone0 px", int(zone0.sum()))

# (1) EDT correct orientation: distance to nearest ZONE pixel
yy, xx = np.mgrid[0:H, 0:W]
gt = np.full((H, W), np.inf, np.float32)
sy, sx = np.where(zone0 > 0)
for (y, x) in zip(sy, sx):
    gt = np.minimum(gt, np.sqrt((yy - y) ** 2 + (xx - x) ** 2).astype(np.float32))
# wasm: distance_transform((zone0>0)) = distance to nearest zone
wv = core.dbg_dist_l2((zone0 > 0).astype(np.uint8), H, W).reshape(H, W)
print(f"[EDT (dist-to-zone)] wasm vs gt maxdiff={np.abs(wv-gt).max():.4f} #px={int((np.abs(wv-gt)>1e-3).sum())}")
# also cv2 distanceTransform((~zone0)) gives distance-to-zone at non-zone pixels
cv = cv2.distanceTransform((~zone0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
cv_at_nonzone = cv[zone0 == 0]
gt_at_nonzone = gt[zone0 == 0]
print(f"[EDT] cv2(~zone0) vs gt (at non-zone) maxdiff={np.abs(cv_at_nonzone-gt_at_nonzone).max():.4f} #px={int((np.abs(cv_at_nonzone-gt_at_nonzone)>1e-3).sum())}")

# (2) gaussian_blur sigma=2
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
cv_g = cv2.GaussianBlur(gray, (0, 0), 2.0)
wv_g = core.dbg_gauss(rgb.astype(np.float32), H, W, 2.0).reshape(H, W)
dg = np.abs(cv_g - wv_g)
print(f"[gaussian_blur σ=2] maxdiff={dg.max():.4f} #px(diff>0.5)={int((dg>0.5).sum())}  cv_range=[{cv_g.min():.2f},{cv_g.max():.2f}] wv_range=[{wv_g.min():.2f},{wv_g.max():.2f}]")
ks_cv = cv2.getGaussianKernel(2.0, cv2.CV_64F)  # float kernel size
print("cv2 getGaussianKernel(2.0, CV_64F) len =", len(ks_cv))
