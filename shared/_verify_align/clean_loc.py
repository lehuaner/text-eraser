"""Localize clean divergence for one divergent input. Compares cv2 vs wasm clean
spatially and against intermediate masks."""
import os, sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import run as R
from text_eraser.text_select import _deglow_full_green_v2, _fill_bright_near_mask, _absorb_zone_bright_core
import text_eraser._shared_core as sc

rng = np.random.default_rng(20260902)
# reproduce it=0 mode=0
def gen_input(rng, mode):
    H, W = int(rng.integers(60, 130)), int(rng.integers(60, 160))
    bg = np.zeros((H, W, 3), np.float32)
    if mode == 0:
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

# drive rng to it=0 (mode=0)
for _ in range(0):
    pass
# replicate exact sequence: main() loop it=0 mode=0%4=0, then gen with rng
rgb, tmask = gen_input(rng, 0)
print("rgb shape", rgb.shape)
P = dict(strength=1.0, zone_ratio=0.6, zone_expand=0, protect_px=1, chroma_keep=0, edge=3, direction=None)
ref = R.cv2_ref(rgb, tmask, tmask, **P)
wasm = R.wasm_run(rgb, tmask, tmask, **P)

# cv2 intermediates for localization
clean0, core0, zone0 = _deglow_full_green_v2(rgb, tmask, strength=1.0, zone_ratio=0.6, zone_expand=0, protect_px=1, deglow_chroma_keep=False, return_zone=True)
mask = ((tmask>0)|(tmask>0)).astype(np.uint8)*255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
mask = _fill_bright_near_mask(clean0, mask)
mask = _absorb_zone_bright_core(clean0, rgb, mask, zone0, min_rgb_lo=100)

cv_clean = np.asarray(ref["clean"], np.int32)
wv_clean = np.asarray(wasm["clean"], np.int32)
d = np.abs(cv_clean - wv_clean)
print("clean maxdiff", int(d.max()), "#px", int((d>0).sum()), "of", d.size)
ys, xs = np.where(d.max(axis=2) > 0)
if len(ys):
    print("diff bbox y", ys.min(), ys.max(), "x", xs.min(), xs.max())
    print("diff correlates with zone0:", float(np.mean(zone0[ys, xs])), "zone0 px:", int(zone0.sum()))
    print("diff correlates with mask(absorb/fill):", float(np.mean(mask[ys, xs])), "mask px:", int(mask.sum()))
# per-channel maxdiff location
for c, name in enumerate("RGB"):
    dc = d[:,:,c]
    print(f"  ch{name} maxdiff={int(dc.max())} #px={int((dc>0).sum())}")
# sample a few diff pixels: show cv vs wasm and zone/mask membership
print("sample diff pixels (y,x, cv, wasm, zone, mask):")
idx = np.argwhere(d.max(axis=2) > 0)[:8]
for (y,x) in idx:
    print(f"  ({y},{x}) cv=({cv_clean[y,x,0]},{cv_clean[y,x,1]},{cv_clean[y,x,2]}) wv=({wv_clean[y,x,0]},{wv_clean[y,x,1]},{wv_clean[y,x,2]}) zone={int(zone0[y,x])} mask={int(mask[y,x])}")
