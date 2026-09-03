"""Binding-level cross-host test (backend side).

Uses shared/bindings/textcore.py (wasmtime) — the real integration layer the
backend will use. Writes <f>.bind_py.bin for identity comparison.
"""
import sys
import os
import struct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bindings"))
from textcore import TextCore

f = sys.argv[1]
with open(f, "rb") as fh:
    h, w = struct.unpack("<ii", fh.read(8))
    mask = np.frombuffer(fh.read(), dtype=np.uint8).reshape(h, w)

core = TextCore()
out = core.distance_transform_edt(mask, h, w)
out.tofile(f + ".bind_py.bin")
print(f"py-binding: {os.path.basename(f)} {h}x{w} wrote .bind_py.bin")
