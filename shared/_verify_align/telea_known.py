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

# Does cv2 modify KNOWN pixels (outside mask) at all?
known = m == 0
cv2_vs_input = np.abs(cv2_res.astype(float) - img.astype(float))[known]
wasm_vs_input = np.abs(wasm.astype(float) - img.astype(float))[known]
print('cv2  known px changed vs input: #(>0.5)', int((cv2_vs_input > 0.5).sum()), '/', int(known.sum()))
print('wasm known px changed vs input: #(>0.5)', int((wasm_vs_input > 0.5).sum()), '/', int(known.sum()))
print('cv2  max known change:', float(cv2_vs_input.max()))
print('wasm max known change:', float(wasm_vs_input.max()))

# Was wasm output = input for the far corner (0,0)?
print('input(0,0)=', float(img[0,0,0]), 'cv2(0,0)=', float(cv2_res[0,0,0]), 'wasm(0,0)=', float(wasm[0,0,0]))

# Distance of changed known pixels from the hole
import scipy.ndimage as ndi
dilate10 = ndi.binary_dilation(m > 0, iterations=10)
band = dilate10 & known
print('band known px:', int(band.sum()))
print('  cv2 changed in band:', int((cv2_vs_input[band] > 0.5).sum()))
print('  wasm changed in band:', int((wasm_vs_input[band] > 0.5).sum()))
print('  wasm changed OUTSIDE band:', int((wasm_vs_input[~dilate10 & known] > 0.5).sum()))
