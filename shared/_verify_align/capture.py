import sys, time, numpy as np, cv2
ROOT = 'D:/Code/Project/Python/TextPatch'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/text_eraser')
from text_eraser.text_select import _deglow_full_green_v2, _fill_bright_near_mask, _absorb_zone_bright_core
from text_eraser.eraser import _residual_green, _dark_source_exclude, _run_fill
from text_eraser.patch_fill import inpaint as pf_inpaint, _normalize_sample_mask
import text_eraser._shared_core as sc
from text_eraser._shared_core import patchmatch_inpaint_fill

# capture what patch_fill.inpaint's wasm call actually receives
captured = {}
orig = patchmatch_inpaint_fill
def capture(*a, **k):
    # a = (sub, subm, subsm, patch, direction, seed)
    sub, subm, subsm, patch, direction, seed = a
    captured['sub_shape'] = sub.shape
    captured['subm_px'] = int((np.asarray(subm) > 0).sum()) if subm is not None else None
    captured['subsm_px'] = int((np.asarray(subsm) > 0).sum()) if subsm is not None else None
    captured['patch'] = patch; captured['direction'] = direction; captured['seed'] = seed
    # also save raw arrays for later comparison
    captured['sub'] = sub.copy()
    captured['subm'] = np.asarray(subm).copy()
    captured['subsm'] = np.asarray(subsm).copy() if subsm is not None else None
    return orig(*a, **k)
import text_eraser.patch_fill as pf
pf.patchmatch_inpaint_fill = capture
import text_eraser._shared_core as sc2
sc2.patchmatch_inpaint_fill = capture
print("using_shared_core:", sc.using_shared_core(), " core:", sc._get_core() is not None)

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

print("captured patch/dir/seed:", captured['patch'], captured['direction'], captured['seed'])
print("captured subm_px (hole):", captured['subm_px'], " subsm_px (sample):", captured['subsm_px'])
print("cv2 mask_filled px:", int((mask_filled_cv2 > 0).sum()))

# Now my Rust hole (mf) is the dilated mask. Compute what cv2's captured subm looks like vs dilated mask_filled
cv_hole = (captured['subm'] > 0)
print("captured subm after pad+crop shape:", cv_hole.shape, "px:", int(cv_hole.sum()))
# compare to raw mask_filled (before pad/crop)
print("raw mask_filled px:", int((mask_filled_cv2 > 0).sum()))
# the captured subm is padded+cropped; check center region matches mask_filled
cy0, cy1 = 4, 4+H  # padm=4
cx0, cx1 = 4, 4+W
sub_center = cv_hole[cy0:cy1, cx0:cx1]
print("center-region hole vs raw mask_filled xor:", int((sub_center ^ (mask_filled_cv2>0)).sum()))
# sample comparison: captured subsm (can-sample) center vs (255-mask_filled)&~se
sm_ref = ((255 - mask_filled_cv2) & (~se.astype(np.uint8)*255)).astype(np.uint8)
sm_ref_bin = (sm_ref > 0)
subsm_center = (captured['subsm'][cy0:cy1, cx0:cx1] > 0) if captured['subsm'] is not None else None
if subsm_center is not None:
    print("center-region sample vs ref xor:", int((subsm_center ^ sm_ref_bin).sum()))
else:
    print("subsm is None (full image sampling)")
