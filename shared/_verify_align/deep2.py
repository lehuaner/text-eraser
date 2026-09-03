import sys, time, numpy as np, cv2
ROOT = 'D:/Code/Project/Python/TextPatch'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/text_eraser')
from text_eraser.text_select import _deglow_full_green_v2, _fill_bright_near_mask, _absorb_zone_bright_core
from text_eraser.eraser import _residual_green, _dark_source_exclude, _run_fill, _edge_aware_grow
from text_eraser.patch_fill import _normalize_sample_mask
import text_eraser._shared_core as sc

rng = np.random.default_rng(12345); H, W = 100, 140
bg = np.zeros((H, W, 3), np.float32)
for c, (b, g) in enumerate([(205, 18), (188, -10), (165, 6)]):
    bg[:, :, c] = b + g * (np.arange(H)[:, None] / float(H))
bg += (rng.random((H, W, 3)).astype(np.float32) - 0.5) * 8.0
rgb = bg.copy()
yy, xx = np.mgrid[0:H, 0:W]; cy, cx = H * 0.45, W * 0.5; ry, rx = H * 0.28, W * 0.33
glow = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
rgb[glow, 0] += 12; rgb[glow, 1] += 72; rgb[glow, 2] += 6; rgb[40:46, 60:70, :] = 235
tmask = np.zeros((H, W), np.uint8)
for y0 in (18, 30, 42): tmask[y0:y0 + 4, 20:60] = 255
tmask[25:35, 28:40] = 255
rgb = np.clip(rgb, 0, 255).astype(np.uint8)
tm_clean = tmask.copy()

clean0, _, zone0 = _deglow_full_green_v2(rgb, tmask, strength=1.15, zone_ratio=0.6,
                                         zone_expand=10, protect_px=1, deglow_chroma_keep=True, return_zone=True)
mask = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
mask = _fill_bright_near_mask(clean0, mask)
mask = _absorb_zone_bright_core(clean0, rgb, mask, zone0, min_rgb_lo=100)
se = _residual_green(clean0, mask)
if (zone0 > 0).any():
    dx = _dark_source_exclude(clean0, mask)
    if dx is not None: se = (dx | se)
cv2_result, mask_filled_cv2, _ = _run_fill(clean0, mask, [], edge=1, direction=None,
                                           edge_aware=False, return_mask=True,
                                           t0=time.time(), sample_exclude=se, soft_expand=0.0)

# ---- EXACT replication of patch_fill.inpaint's crop+padm, calling wasm directly ----
img = clean0.astype(np.float32).copy()
m = (mask_filled_cv2 > 0).astype(np.uint8)
OH, OW = img.shape[:2]
padm = 4
imgp = cv2.copyMakeBorder(img, padm, padm, padm, padm, cv2.BORDER_REPLICATE)
mp = np.pad(m, padm, constant_values=False)
Hp, Wp = imgp.shape[:2]
sample_mask_cv2 = (255 - mask_filled_cv2).astype(np.uint8)
if se is not None: sample_mask_cv2[se] = 0
sm = _normalize_sample_mask(sample_mask_cv2, OH, OW)
if sm is not None: sm = np.pad(sm, padm, constant_values=False)
ys, xs = np.where(mp)
hy0, hy1 = int(ys.min()), int(ys.max()) + 1
hx0, hx1 = int(xs.min()), int(xs.max()) + 1
span = max(hy1 - hy0, hx1 - hx0)
margin = max(32, int(0.6 * span))
if sm is not None: margin = max(margin, int(0.9 * span), 80)
y0 = max(0, hy0 - margin); y1 = min(Hp, hy1 + margin)
x0 = max(0, hx0 - margin); x1 = min(Wp, hx1 + margin)
MAX_ROI = 1536
while max(y1 - y0, x1 - x0) > MAX_ROI and margin > 24:
    margin = int(margin * 0.85)
    y0 = max(0, hy0 - margin); y1 = min(Hp, hy1 + margin)
    x0 = max(0, hx0 - margin); x1 = min(Wp, hx1 + margin)
sub = imgp[y0:y1, x0:x1].copy()
subm = mp[y0:y1, x0:x1].copy()
sh, sw = sub.shape[:2]
subsm = sm[y0:y1, x0:x1] if sm is not None else None
print("crop y0,y1,x0,x1 =", y0, y1, x0, x1, " sh,sw=", sh, sw, " OH,OW=", OH, OW)
_deg = -1.0
filled = sc.patchmatch_inpaint_fill(sub, subm, subsm, 7, _deg, 0)
imgp[y0:y1, x0:x1] = np.clip(filled, 0, 255)
out = np.clip(imgp, 0, 255)[padm:padm + OH, padm:padm + OW].astype(np.uint8)
print("crop+padm wasm vs cv2_result:", (np.abs(out.astype(int)-cv2_result.astype(int)).max(),
                                         int((np.abs(out.astype(int)-cv2_result.astype(int))>0).sum())))

# A/B: call patch_fill.inpaint directly (the real cv2 path) with SAME inputs as replication
from text_eraser.patch_fill import inpaint as pf_inpaint
ab = pf_inpaint(clean0.astype(np.uint8), mask_filled_cv2, sample_mask_cv2, 7, None, 15.0)
print("patch_fill.inpaint(via wasm) vs cv2_result:", (np.abs(ab.astype(int)-cv2_result.astype(int)).max(),
                                                      int((np.abs(ab.astype(int)-cv2_result.astype(int))>0).sum())))
print("my crop replication vs patch_fill.inpaint:", (np.abs(out.astype(int)-ab.astype(int)).max(),
                                                     int((np.abs(out.astype(int)-ab.astype(int))>0).sum())))
