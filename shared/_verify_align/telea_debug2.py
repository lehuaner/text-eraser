import sys, numpy as np
sys.path.insert(0, 'D:/Code/Project/Python/TextPatch')
sys.path.insert(0, 'D:/Code/Project/Python/TextPatch/text_eraser')
import cv2
import text_eraser._shared_core as sc

core = sc._get_core()

def run(H, W, hole, label):
    # non-linear background: gentle 2D bump so gradient varies
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 120.0 + 40.0*np.sin(xx/18.0) + 30.0*np.cos(yy/22.0)
    img = np.stack([base, base, base], axis=-1).astype(np.float32)
    m = np.zeros((H, W), np.uint8)
    m[hole] = 255
    cv2_res = cv2.inpaint(img.astype(np.uint8), m, 3, cv2.INPAINT_TELEA)
    wasm = core.dbg_telea(img.astype(np.float32), m, H, W, 3)
    # known pixels preserved?
    known = m == 0
    dk = np.abs(wasm.astype(float) - cv2_res.astype(float))[known]
    dh = np.abs(wasm.astype(float) - cv2_res.astype(float))[~known]
    print(f"[{label}] known: maxdiff={dk.max():.3f} #(>0.5)={int((dk>0.5).sum())}/{int(known.sum())}")
    print(f"[{label}] HOLE : maxdiff={dh.max():.3f} meandiff={dh.mean():.3f} #(>1)={int((dh>1).sum())}/{int((~known).sum())}")
    # sample a hole pixel
    hys, hxs = np.where(~known)
    if len(hys):
        y, x = hys[len(hys)//2], hxs[len(hxs)//2]
        print(f"   sample hole ({y},{x}): cv2={cv2_res[y,x]}, wasm={np.round(wasm[y,x],1)}")

# 1-pixel hole
run(40, 40, (slice(20,21), slice(20,21)), "1px")
# 3x3 hole
run(40, 40, (slice(19,22), slice(19,22)), "3x3")
# 5x5 hole
run(40, 40, (slice(18,23), slice(18,23)), "5x5")
