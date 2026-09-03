"""Compare cv2 RGB2GRAY vs my JS fixed-point formula on rgb.raw for several images."""
import os, struct
import numpy as np
import cv2

WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work'

def load_rgb(name):
    d = os.path.join(WORK, name)
    H, W = map(int, open(os.path.join(d, 'dims.txt')).read().strip().split())
    buf = open(os.path.join(d, 'rgb.raw'), 'rb').read()
    rgb = np.frombuffer(buf, dtype=np.float32).reshape(H, W, 3)
    return rgb.astype(np.uint8), H, W

def js_gray(rgb_u8, H, W):
    out = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            r = int(rgb_u8[y, x, 0]); g = int(rgb_u8[y, x, 1]); b = int(rgb_u8[y, x, 2])
            out[y, x] = (r*4899 + g*9617 + b*1868 + 8192) >> 14
    return out.astype(np.float64)

for name in ['556_orig', '178_orig', '668_orig']:
    rgb, H, W = load_rgb(name)
    cv_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    my_gray = js_gray(rgb, H, W)
    diff = np.abs(cv_gray - my_gray)
    print(f"{name}: maxdiff={diff.max():.1f} meandiff={diff.mean():.4f} mismatches={int((diff>0).sum())}/{H*W}")
