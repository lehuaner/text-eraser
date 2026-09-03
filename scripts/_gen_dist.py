"""Generate test masks + reference cv2 distance transforms for the EDT unit test.
Outputs per-case dirs under scripts/_cmp_work/_edt/<name>/:
  mask.raw  (uint8, 1 = foreground/text, 0 = background)
  cv3.raw   (float32, cv2.distanceTransform((mask==0), DIST_L2, 3))
  cv5.raw   (float32, cv2.distanceTransform((mask==0), DIST_L2, 5))
  dims.txt  (H W)
"""
import os, sys
import numpy as np
import cv2

WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work'
OUT = os.path.join(WORK, '_edt')
os.makedirs(OUT, exist_ok=True)

def emit(name, H, W, mask):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    mask = mask.astype(np.uint8)
    np.ndarray.tofile(np.ascontiguousarray(mask), os.path.join(d, 'mask.raw'))
    src = (mask == 0).astype(np.uint8)  # distanceTransform source: zeros = foreground(text)
    cv3 = cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32)
    cv5 = cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(np.float32)
    cv3.tofile(os.path.join(d, 'cv3.raw'))
    cv5.tofile(os.path.join(d, 'cv5.raw'))
    with open(os.path.join(d, 'dims.txt'), 'w') as f:
        f.write(f'{H} {W}')

# --- synthetic stress masks ----------------------------------------------
rng = np.random.default_rng(1234)

# 1) scattered rectangles (text-like)
H, W = 240, 360
m = np.zeros((H, W), np.uint8)
for _ in range(40):
    x, y = rng.integers(0, W-20), rng.integers(0, H-10)
    w, h = rng.integers(3, 18), rng.integers(3, 9)
    m[y:y+h, x:x+w] = 1
emit('syn_rects', H, W, m)

# 2) thin lines + isolated points (stress EDT envelope)
H, W = 200, 300
m = np.zeros((H, W), np.uint8)
for k in range(0, W, 7):
    m[100, k] = 1
m[50:150:2, 30] = 1
for _ in range(30):
    m[rng.integers(0, H), rng.integers(0, W)] = 1
emit('syn_lines', H, W, m)

# 3) annulus (distance to two opposite arcs)
H, W = 256, 256
m = np.zeros((H, W), np.uint8)
yy, xx = np.mgrid[0:H, 0:W]
cx, cy = W//2, H//2
r = np.sqrt((xx-cx)**2 + (yy-cy)**2)
m[(r > 60) & (r < 70)] = 1
emit('syn_annulus', H, W, m)

# --- real foreground masks (final text masks) ----------------------------
for name in ['_ab_synth', '_d5814_orig']:
    d = os.path.join(WORK, name)
    if not os.path.exists(os.path.join(d, 'refmask.raw')):
        continue
    dims = open(os.path.join(d, 'dims.txt')).read().strip().split()
    H, W = int(dims[0]), int(dims[1])
    ref = np.fromfile(os.path.join(d, 'refmask.raw'), dtype=np.uint8)[:H*W].reshape(H, W)
    emit(f'real_{name}', H, W, (ref > 0).astype(np.uint8))

print('generated:', os.listdir(OUT))
