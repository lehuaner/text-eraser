import sys, numpy as np, cv2
sys.path.insert(0, "D:/Code/Project/Python/TextPatch")
sys.path.insert(0, "D:/Code/Project/Python/TextPatch/text_eraser")
import text_eraser._shared_core as sc
from text_eraser.text_select import _deglow_full_green_v2

rng = np.random.default_rng(12345); H, W = 100, 140
bg = np.zeros((H, W, 3), np.float32)
for c, (b, g) in enumerate([(205, 18), (188, -10), (165, 6)]):
    bg[:, :, c] = b + g * (np.arange(H)[:, None] / float(H))
bg += (rng.random((H, W, 3)).astype(np.float32) - 0.5) * 8.0
rgb = bg.copy()
yy, xx = np.mgrid[0:H, 0:W]; cy, cx = H*0.45, W*0.5; ry, rx = H*0.28, W*0.33
glow = ((yy-cy)/ry)**2 + ((xx-cx)/rx)**2 <= 1.0
rgb[glow, 0] += 12; rgb[glow, 1] += 72; rgb[glow, 2] += 6; rgb[40:46, 60:70, :] = 235
tmask = np.zeros((H, W), np.uint8)
for y0 in (18, 30, 42): tmask[y0:y0+4, 20:60] = 255
tmask[25:35, 28:40] = 255
rgb = np.clip(rgb, 0, 255).astype(np.uint8)

core = sc._get_core()
rgb_f = rgb.astype(np.float32).reshape(-1); tm = tmask.reshape(-1).astype(np.uint8)
_, _, zone0 = core.deglow_full_green_v2(rgb_f, H, W, tm, 1.15, 0.6, 10, 1, 1)
zone0 = np.asarray(zone0).reshape(H, W) > 0
_, _, c_zone = _deglow_full_green_v2(rgb, tmask, strength=1.15, zone_ratio=0.6, zone_expand=10, protect_px=1, deglow_chroma_keep=True, return_zone=True)
c_zone = c_zone > 0

ys, xs = np.where(c_zone & ~zone0)  # cv2-only (wasm missing)
print("wasm zone px:", int(zone0.sum()), "cv2 zone px:", int(c_zone.sum()), "wasm-missing:", len(ys))
print("missing y range:", int(ys.min()), int(ys.max()), "x range:", int(xs.min()), int(xs.max()))
# characterize missing pixels
r = rgb[..., 0].astype(np.int16); g = rgb[..., 1].astype(np.int16); b = rgb[..., 2].astype(np.int16)
strong_green = ((g - np.maximum(r, b)) > 8) & (g > 95)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
bg_cand = gray[~strong_green]; bg_lum = float(np.median(bg_cand)) if bg_cand.size else 80.0
gn = np.maximum(g - np.maximum(r, b), 0)
bright = ((gray > (bg_lum+6)) & (gray > 55) & (gn > 2))
faint = (g - np.maximum(r, b) > 3) & (g > 55)
green = (g - np.maximum(r, b) > 2) & (g > 60)
grow_cond = green | bright | faint
for i in range(min(20, len(ys))):
    y, x = int(ys[i]), int(xs[i])
    print(f"  ({y},{x}) grow_cond={bool(grow_cond[y,x])} strong_green={bool(strong_green[y,x])} tmask={bool(tmask[y,x]>0)} gray={gray[y,x]:.0f} g-mr={int(g[y,x]-max(r[y,x],b[y,x]))} in_glow={bool(glow[y,x])}")
