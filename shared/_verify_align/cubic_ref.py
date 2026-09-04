"""Isolate cv2 INTER_CUBIC: implement the separable bicubic in pure Python with the
same kernel (a=-0.5) and reflect_101 edges as deglow.rs, and compare to cv2.resize.
Also try a=-0.75. This tells us which scheme cv2 actually uses."""
import numpy as np
import cv2


def reflect101(i, n):
    if n <= 1:
        return 0
    p = 2 * (n - 1)
    x = i % p
    if x < 0:
        x += p
    if x >= n:
        x = p - x
    return x


def cubic_w(t, a=-0.5):
    t = abs(t)
    if t <= 1.0:
        return (a + 2.0) * t**3 - (a + 3.0) * t**2 + 1.0
    elif t < 2.0:
        return a * t**3 - 5.0 * a * t**2 + 8.0 * a * t - 4.0 * a
    return 0.0


def cubic_1d_py(src, m, a=-0.5):
    n = len(src)
    scale = n / m
    out = np.zeros(m, np.float64)
    for i in range(m):
        sx = (i + 0.5) * scale - 0.5
        x0 = int(np.floor(sx))
        dx = sx - x0
        s = 0.0
        for k in range(4):
            j = x0 - 1 + k
            w = cubic_w((k - 1) - dx, a)
            s += src[reflect101(j, n)] * w
        out[i] = s
    return out


def cubic_2d_py(src, h2, w2, a=-0.5):
    h, w = src.shape[:2]
    ch = src.shape[2] if src.ndim == 3 else 1
    tmp = np.zeros((h, w2, ch), np.float64)
    for y in range(h):
        for c in range(ch):
            row = src[y, :, c].astype(np.float64) if ch > 1 else src[y, :].astype(np.float64)
            tmp[y, :, c] = cubic_1d_py(row, w2, a)
    out = np.zeros((h2, w2, ch), np.float64)
    for x in range(w2):
        for c in range(ch):
            col = tmp[:, x, c]
            out[:, x, c] = cubic_1d_py(col, h2, a)
    return out


def maxdiff(a, b):
    d = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
    return float(d.max()), int((d > 0.5).sum())


rng = np.random.default_rng(3)
base = rng.random((139, 109, 3)).astype(np.float32) * 255.0
yy, xx = np.mgrid[0:139, 0:109].astype(np.float32)
base = (base + (yy[:, :, None] * 1.3 + xx[:, :, None] * 0.7)) % 255.0

for (h2, w2) in [(278, 218), (69, 54)]:
    for a in (-0.5, -0.75):
        ref = cv2.resize(base, (w2, h2), interpolation=cv2.INTER_CUBIC)
        mine = cubic_2d_py(base, h2, w2, a)
        md, nd = maxdiff(ref, mine)
        print(f"cubic a={a}  {139}x109->{h2}x{w2}  md={md:.3f} #>0.5={nd}")
