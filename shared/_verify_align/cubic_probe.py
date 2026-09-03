"""Reverse-engineer cv2 INTER_CUBIC from a tiny 1D signal. If cubic interpolates a
linear ramp exactly, the kernel is standard; otherwise we see the deviation pattern."""
import numpy as np
import cv2

# 1x5 linear ramp, upscale x3 -> 1x15
sig = np.array([0.0, 64.0, 128.0, 192.0, 255.0], np.float32).reshape(1, 5, 1)
out = cv2.resize(sig, (15, 1), interpolation=cv2.INTER_CUBIC)
print("src :", sig.ravel())
print("cubic:", np.round(out.ravel(), 3))
# expected linear: positions 0,0.333,0.667,...  (dst i maps to src (i+0.5)*5/15-0.5)
exp = []
for i in range(15):
    sx = (i + 0.5) * 5.0 / 15.0 - 0.5
    # linear interp
    x0 = int(np.floor(sx)); dx = sx - x0
    a = sig.ravel()[min(max(x0,0),4)]; b = sig.ravel()[min(max(x0+1,0),4)]
    exp.append(a*(1-dx)+b*dx)
print("linear:", np.round(exp, 3))
print("maxdiff cubic vs linear:", np.abs(out.ravel()-np.array(exp)).max())

# Also test a 2x2 constant-ish small upscale to see center weight behavior
sig2 = np.array([[0.0,100.0],[0.0,100.0]], np.float32)
out2 = cv2.resize(sig2, (4,4), interpolation=cv2.INTER_CUBIC)
print("\n2x2->4x4 cubic:\n", np.round(out2, 3))
