import os, sys, time, numpy as np, cv2
ROOT = 'D:/Code/Project/Python/TextPatch'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/text_eraser')
from text_eraser.text_select import _deglow_full_green_v2, _fill_bright_near_mask, _absorb_zone_bright_core
from text_eraser.eraser import _residual_green, _dark_source_exclude, _run_fill
import text_eraser._shared_core as sc

rng = np.random.default_rng(12345)
H, W = 100, 140
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

# --- cv2 reference (the ORIGINAL python source) ---
clean0, _, zone0 = _deglow_full_green_v2(rgb, tmask, strength=1.15, zone_ratio=0.6,
                                         zone_expand=10, protect_px=1, deglow_chroma_keep=True,
                                         return_zone=True)
mask = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
mask = _fill_bright_near_mask(clean0, mask)
mask = _absorb_zone_bright_core(clean0, rgb, mask, zone0, min_rgb_lo=100)
se = _residual_green(clean0, mask)
if (zone0 > 0).any():
    dx = _dark_source_exclude(clean0, mask)
    if dx is not None: se = (dx | se)
cv2_result, mask_filled_cv2, _ = _run_fill(clean0, mask, None, edge=1, direction=None,
                                           edge_aware=True, return_mask=True,
                                           sample_exclude=se, t0=time.time())

# --- replicate cv2's EXACT mask_filled + sample for wasm-inpaint direct call ---
edge = 1
mask_filled_cv2 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge * 2 + 1, edge * 2 + 1)))
sample_mask_cv2 = (255 - mask_filled_cv2).astype(np.uint8)
if se is not None: sample_mask_cv2[se] = 0

core = sc._get_core()
# call patchmatch_inpaint_fill directly with cv2's exact inputs (NO crop -> tests crop effect)
pm_full = sc.patchmatch_inpaint_fill(clean0.astype(np.float32), mask_filled_cv2, sample_mask_cv2, 7, -1.0, 0)

# --- wasm erase_text_glyphs ---
res = core.erase_text_glyphs(rgb.astype(np.float32), H, W, tmask, tm_clean, 1.15, 0.6, 10, 1, 1, 1, -1.0, 0)
wasm_result = res[0].reshape(H, W, 3)


def md(a, b):
    a = np.asarray(a).astype(int); b = np.asarray(b).astype(int)
    d = np.abs(a - b)
    return int(d.max()), int((d > 0).sum())


print("== triple comparison (result) ==")
print("cv2 _run_fill vs wasm erase_text_glyphs :", md(cv2_result, wasm_result))
print("cv2 _run_fill vs patchmatch_inpaint_fill(full,no-crop) :", md(cv2_result, pm_full))
print("wasm erase_text_glyphs vs patchmatch_inpaint_fill(full) :", md(wasm_result, pm_full))

# where does wasm differ from cv2?
dd = np.abs(cv2_result.astype(int) - wasm_result.astype(int)).max(axis=2)
ys, xs = np.where(dd > 0)
print("\nwasm!=cv2 region: px=%d  y[%d..%d] x[%d..%d]" % (len(ys), int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())))
print("mask_filled_cv2 px:", int((mask_filled_cv2 > 0).sum()), " mask px:", int((mask > 0).sum()))
# Is the wasm diff inside mask_filled?
print("wasm diff inside mask_filled:", int(((dd > 0) & (mask_filled_cv2 > 0)).sum()))
print("wasm diff outside mask_filled:", int(((dd > 0) & (mask_filled_cv2 == 0)).sum()))
# compare the inpaint CONTENT (subtract clean0): is it the same patch shifted?
clean = clean0.astype(int)
print("\ncv2 inpaint delta max:", int(np.abs(cv2_result.astype(int) - clean).max()))
print("wasm inpaint delta max:", int(np.abs(wasm_result.astype(int) - clean).max()))
