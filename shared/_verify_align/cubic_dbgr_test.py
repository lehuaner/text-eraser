"""Compare my Rust dbg_rc cubic to cv2.resize on the linear ramp (1x5->1x15)."""
import os, sys
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from shared.bindings.textcore import get_core
core = get_core()

sig = np.array([0.0, 64.0, 128.0, 192.0, 255.0], np.float32).reshape(1, 5, 1)
ref = cv2.resize(sig, (15, 1), interpolation=cv2.INTER_CUBIC).reshape(15)
wasm = core.dbg_resize(sig, 1, 5, 1, 1, 15, "cubic").reshape(15)
print("cv2  :", np.round(ref, 3))
print("wasm :", np.round(wasm, 3))
print("diff :", np.round(ref - wasm, 3))
print("maxdiff:", np.abs(ref - wasm).max())
