import sys, numpy as np
sys.path.insert(0, 'D:/Code/Project/Python/TextPatch')
sys.path.insert(0, 'D:/Code/Project/Python/TextPatch/text_eraser')
import cv2
import text_eraser._shared_core as sc

core = sc._get_core()
H, W = 120, 160
g = np.linspace(20, 220, W, dtype=np.float32)
img = np.stack([g, g, g], axis=-1)[None].repeat(H, 0)
m = np.zeros((H, W), np.uint8)
m[40:80, 60:110] = 255

cv2_res = cv2.inpaint(img.astype(np.uint8), m, 3, cv2.INPAINT_TELEA)
wasm = core.dbg_telea(img.astype(np.float32), m, H, W, 3)

# Per-channel known-change for cv2 (is cv2 modifying a band?)
for c in range(3):
    known = m == 0
    diff_cv = np.abs(cv2_res[:, :, c].astype(float) - img[:, :, c].astype(float))[known]
    diff_wasm = np.abs(wasm[:, :, c].astype(float) - img[:, :, c].astype(float))[known]
    print(f'chan{c}: cv2 known changed(>0.5)={int((diff_cv>0.5).sum())}, wasm known changed(>0.5)={int((diff_wasm>0.5).sum())}')

# Width of cv2's modified band: distance from hole
import scipy.ndimage as ndi
edt = ndi.distance_transform_edt(m == 0)  # distance from hole edge to each outside pixel
known = m == 0
for thresh in [0, 1, 2, 3, 5, 10, 20]:
    band = known & (edt <= thresh)
    if band.sum() == 0:
        continue
    d = np.abs(cv2_res.astype(float) - img.astype(float))[band]
    print(f'cv2 band<= {thresh}px: px={int(band.sum())}, maxdiff={float(d.max()):.3f}, #(>0.5)={int((d>0.5).sum())}')

# HOLE region: per-channel max/mean diff between wasm and cv2
holes = m > 0
for c in range(3):
    d = np.abs(wasm[:, :, c].astype(float) - cv2_res[:, :, c].astype(float))[holes]
    print(f'HOLE chan{c}: maxdiff={float(d.max()):.3f}, meandiff={float(d.mean()):.3f}, #(>1)={int((d>1).sum())}, #(>3)={int((d>3).sum())}')
