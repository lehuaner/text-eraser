"""Isolate the EDT bug: 1D exact EDT (h=1) vs cv2."""
import sys, os
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import text_eraser._shared_core as sc
core = sc._get_core()

# 1D mask: source at index 0, rest far
N = 20
mask = np.zeros((1, N), np.uint8)
mask[0, 0] = 1
cv = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
wv = core.dbg_dist_l2(mask, 1, N).reshape(1, N)
print("1D source@0:")
print("  cv2 :", np.round(cv[0], 2))
print("  wasm:", np.round(wv[0], 2))

# 1D mask: two sources, gap in middle
mask2 = np.zeros((1, 21), np.uint8)
mask2[0, 0] = 1
mask2[0, 20] = 1
cv2b = cv2.distanceTransform((~mask2).astype(np.uint8), cv2.DIST_L2, 5)
wv2 = core.dbg_dist_l2(mask2, 1, 21).reshape(1, 21)
print("1D sources@0,20:")
print("  cv2 :", np.round(cv2b[0], 2))
print("  wasm:", np.round(wv2[0], 2))

# 2D: single source center, should be radially increasing
H = W = 31
mask3 = np.zeros((H, W), np.uint8)
mask3[H//2, W//2] = 1
cv3 = cv2.distanceTransform((~mask3).astype(np.uint8), cv2.DIST_L2, 5)
wv3 = core.dbg_dist_l2(mask3, H, W)
print("2D center source: cv2 max=%.2f wasm max=%.2f" % (cv3.max(), wv3.max()))
print("  cv3[0,0]=%.2f wasm[0,0]=%.2f" % (cv3[0,0], wv3[0,0]))
