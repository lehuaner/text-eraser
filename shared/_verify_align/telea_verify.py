"""Isolate the TELEA inpaint: compare wasm dbg_telea vs cv2.inpaint(..., INPAINT_TELEA).

The original `patch_fill.inpaint` falls back to OpenCV TELEA for smooth-gradient
backgrounds. To make the shared wasm 1:1 with that path, `telea_inpaint` must
reproduce cv2's output. This harness measures it directly (no deglow/mask-surgery
noise), over several shapes, so a residual shows as a pixel diff.
"""
import sys, numpy as np
sys.path.insert(0, "D:/Code/Project/Python/TextPatch")
sys.path.insert(0, "D:/Code/Project/Python/TextPatch/text_eraser")
from text_eraser import _cv as cv2
import text_eraser._shared_core as sc

core = sc._get_core()


def md(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    return int(d.max()), int((d > 0.5).sum())


def run(name, rgb, mask):
    H, W = rgb.shape[:2]
    cv2_res = cv2.inpaint(rgb.astype(np.uint8), mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    wasm_res = core.dbg_telea(rgb.astype(np.float32), mask.astype(np.uint8), H, W, 3)
    m, p = md(cv2_res, wasm_res)
    print(f"{name:24s} maxdiff={m:4d}  #diff={p:6d}  (mask px={int((mask>0).sum())})")
    return m, p


# A: smooth vertical gradient + rect hole
H, W = 120, 160
g = np.linspace(20, 220, W, dtype=np.float32)
imgA = np.stack([g, g, g], axis=-1)[None].repeat(H, 0).repeat(1, 1).reshape(H, W, 3)
mA = np.zeros((H, W), np.uint8); mA[40:80, 60:110] = 255
run("A grad+rect", imgA, mA)

# B: diagonal gradient + circular hole
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
gd = (yy / H * 120 + xx / W * 120)
imgB = np.stack([gd, gd + 15, gd + 30], axis=-1)
cy, cx = H // 2, W // 2
mB = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 30 ** 2).astype(np.uint8) * 255
run("B diag+circle", imgB, mB)

# C: warm synthetic image like the deglow test, with the text-mask region as hole
rng = np.random.default_rng(7)
bg = np.zeros((100, 140, 3), np.float32)
for c, (b, gg) in enumerate([(205, 18), (188, -10), (165, 6)]):
    bg[:, :, c] = b + gg * (np.arange(100)[:, None] / 100.0)
bg += (rng.random((100, 140, 3)).astype(np.float32) - 0.5) * 8
imgC = bg.copy()
yy, xx = np.mgrid[0:100, 0:140]
glow = (((yy - 45) / 28) ** 2 + ((xx - 70) / 33) ** 2) <= 1.0
imgC[glow, 0] += 12; imgC[glow, 1] += 72; imgC[glow, 2] += 6
imgC[40:46, 60:70, :] = 235
mC = np.zeros((100, 140), np.uint8)
for y0 in (18, 30, 42):
    mC[y0:y0 + 4, 20:60] = 255
mC[25:35, 28:40] = 255
run("C warm+textmask", imgC, mC)

# D: high-detail random (texture direction test)
imgD = (rng.random((100, 140, 3)).astype(np.float32) * 255).astype(np.float32)
mD = mC.copy()
run("D random+textmask", imgD, mD)
