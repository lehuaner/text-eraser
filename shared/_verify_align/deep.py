import sys, time, numpy as np, cv2
ROOT = 'D:/Code/Project/Python/TextPatch'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/text_eraser')
from text_eraser.text_select import _deglow_full_green_v2, _fill_bright_near_mask, _absorb_zone_bright_core
from text_eraser.eraser import _residual_green, _dark_source_exclude, _run_fill, _edge_aware_grow
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

# cv2 reference exactly as run.py does
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

# wasm
res = sc._get_core().erase_text_glyphs(rgb.astype(np.float32), H, W, tmask, tm_clean,
                                       1.15, 0.6, 10, 1, 1, 1, -1.0, 0)
wasm_result = res[0].reshape(H, W, 3)
wasm_fill = res[1].reshape(H, W)

# replicate the hole cv2 uses -> call wasm patchmatch directly (no internal crop) for isolation
mask_filled_cv2_bin = (mask_filled_cv2 > 0)
sample_mask_cv2 = (255 - mask_filled_cv2).astype(np.uint8)
if se is not None: sample_mask_cv2[se] = 0
pm_full = sc.patchmatch_inpaint_fill(clean0.astype(np.float32), mask_filled_cv2, sample_mask_cv2, 7, -1.0, 0)

print("cv2 mask_filled px:", int(mask_filled_cv2_bin.sum()), "wasm fill px:", int((wasm_fill > 0).sum()))
print("xor hole:", int((mask_filled_cv2_bin ^ (wasm_fill > 0)).sum()))
print()
print("wasm_result vs cv2_result:", (np.abs(wasm_result.astype(int)-cv2_result.astype(int)).max(), int((np.abs(wasm_result.astype(int)-cv2_result.astype(int))>0).sum())))
print("pm_full      vs cv2_result:", (np.abs(pm_full.astype(int)-cv2_result.astype(int)).max(), int((np.abs(pm_full.astype(int)-cv2_result.astype(int))>0).sum())))
print("wasm_result  vs pm_full     :", (np.abs(wasm_result.astype(int)-pm_full.astype(int)).max(), int((np.abs(wasm_result.astype(int)-pm_full.astype(int))>0).sum())))
# is wasm_result outside hole == clean0?
dd = np.abs(wasm_result.astype(int) - clean0.astype(int))
print("wasm_result vs clean0 OUTSIDE hole xor px:", int((dd>0 & ~mask_filled_cv2_bin).sum()))
print("pm_full     vs clean0 OUTSIDE hole xor px:", int((np.abs(pm_full.astype(int)-clean0.astype(int))>0 & ~mask_filled_cv2_bin).sum()))
