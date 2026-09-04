import sys, numpy as np
sys.path.insert(0, "D:/Code/Project/Python/TextPatch")
sys.path.insert(0, "D:/Code/Project/Python/TextPatch/text_eraser")
from text_eraser import _cv as cv2
import text_eraser._shared_core as sc

core = sc._get_core()

# Tiny 8x8 image: smooth gradient column, 2x2 hole at center
H, W = 8, 8
g = np.linspace(30, 230, W, dtype=np.float32)
img = np.stack([g, g, g], axis=-1)[None].repeat(H, 0).copy()  # H,W,3
mask = np.zeros((H, W), np.uint8)
mask[3:5, 3:5] = 255

print("img[:,:,0] (gradient):")
print(img[0, :, 0].astype(int))
print("mask:")
print(mask)

cv2_res = cv2.inpaint(img.astype(np.uint8), mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
wasm_res = core.dbg_telea(img.astype(np.float32), mask.astype(np.uint8), H, W, 3)

print("cv2 result[:,:,0]:")
print(cv2_res[:, :, 0])
print("wasm result[:,:,0]:")
print(wasm_res[:, :, 0].astype(int))

# known pixel outside hole: should be IDENTICAL
print("known pixel (0,0) cv2/wasm:", int(cv2_res[0,0,0]), int(wasm_res[0,0,0]))
print("hole pixel (3,3) cv2/wasm:", int(cv2_res[3,3,0]), int(wasm_res[3,3,0]))
# count differing known pixels
knowndiff = int((np.abs(cv2_res.astype(int) - wasm_res.astype(int)) > 0).sum())
print("total differing elements:", knowndiff, "of", H*W*3)
