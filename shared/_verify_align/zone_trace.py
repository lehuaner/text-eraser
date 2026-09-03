import sys, numpy as np, cv2
sys.path.insert(0, 'D:/Code/Project/Python/TextPatch'); sys.path.insert(0, 'D:/Code/Project/Python/TextPatch/text_eraser')
from text_eraser.text_select import _deglow_full_green_v2
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
for y0 in (18, 30, 42): tmask[y0:y0+4, 20:60] = 255
tmask[25:35, 28:40] = 255
rgb = np.clip(rgb, 0, 255).astype(np.uint8)

# Replicate cv2 grow exactly
r = rgb[..., 0].astype(np.int16); g = rgb[..., 1].astype(np.int16); b = rgb[..., 2].astype(np.int16)
strong_green = ((g - np.maximum(r, b)) > 8) & (g > 95)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
bg_cand = gray[~strong_green]; bg_lum = float(np.median(bg_cand)) if bg_cand.size else 80.0
_greenness = np.maximum(g - np.maximum(r, b), 0)
green = (g - np.maximum(r, b) > 2) & (g > 60)
bright = ((gray > (bg_lum + 6)) & (gray > 55) & (_greenness > 2))
faint_green = (g - np.maximum(r, b) > 3) & (g > 55)
grow_cond = green | bright | faint_green
zone = (strong_green | (tmask > 0)).copy(); cur = zone
budget = int(H * W * 0.6); k3 = np.ones((3, 3), np.uint8)
target = (13, 11)
for it in range(400):
    dil = cv2.dilate(cur.astype(np.uint8), k3) > 0
    add = dil & grow_cond & ~zone
    if add[target]:
        print(f"target (13,11) added at iteration {it}; cur-sum before={int(zone.sum())}")
        # show which neighbor triggered it
        break
    zone |= add
    if int(zone.sum()) > budget:
        zone &= ~add; break
    cur = zone
else:
    print("target (13,11) NEVER added by grow loop; final zone px:", int(zone.sum()))
# Now test zone_expand
ze = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*10+1, 2*10+1))
zone2 = cv2.dilate(zone.astype(np.uint8), ze) > 0
print("after zone_expand(10): target in zone?", bool(zone2[target]), "px:", int(zone2.sum()))
