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

known3 = np.broadcast_to((m == 0)[:, :, None], (H, W, 3))
holes3 = ~known3
dk = np.abs(wasm.astype(float) - cv2_res.astype(float))[known3]
dh = np.abs(wasm.astype(float) - cv2_res.astype(float))[holes3]
print('KNOWN: maxdiff', float(dk.max()), ' #(>0.5)', int((dk > 0.5).sum()), '/ total', int(known3.sum()))
print('HOLE : maxdiff', float(dh.max()), ' #(>0.5)', int((dh > 0.5).sum()), '/ total', int(holes3.sum()))
print('hole(60,85): cv2', cv2_res[60, 85], 'wasm', np.round(wasm[60, 85], 2))
print('hole(50,85): cv2', cv2_res[50, 85], 'wasm', np.round(wasm[50, 85], 2))
