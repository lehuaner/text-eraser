"""Compare JS exact EDT (js.raw) against the ground-truth exact EDT (scipy) and cv2.

scipy.ndimage.distance_transform_edt((mask==0)) is the true exact Euclidean EDT;
use it to judge whether the JS F&H implementation is correct. cv2 DIST_L2 mask3/5
are approximate in many builds, so they are shown only for context.
"""
import os
import numpy as np
from scipy import ndimage

WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work/_edt'

def load(name, fn, dt=np.float32):
    return np.fromfile(os.path.join(WORK, name, fn), dtype=dt)

for name in sorted(os.listdir(WORK)):
    d = os.path.join(WORK, name)
    H, W = map(int, open(os.path.join(d, 'dims.txt')).read().strip().split())
    mask = np.fromfile(os.path.join(d, 'mask.raw'), dtype=np.uint8).reshape(H, W)
    js = load(name, 'js.raw').reshape(H, W)
    gt = ndimage.distance_transform_edt(mask == 0).astype(np.float32)  # exact
    d_js = np.abs(js - gt)
    d_cv3 = np.abs(load(name, 'cv3.raw').reshape(H, W) - gt)
    d_cv5 = np.abs(load(name, 'cv5.raw').reshape(H, W) - gt)
    def stat(x):
        return f"max={x.max():.4f} mean={x.mean():.4f} >0.5={int((x>0.5).sum())} >1={int((x>1.0).sum())}"
    print(f"{name}: H={H} W={W}")
    print(f"  JS  vs exact : {stat(d_js)}")
    print(f"  cv2 m3 vs exact: {stat(d_cv3)}")
    print(f"  cv2 m5 vs exact: {stat(d_cv5)}")
