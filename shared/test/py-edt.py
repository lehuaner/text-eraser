"""Python-side consumer of the shared WASM core (backend stand-in).

Loads build/textcore.wasm via wasmtime, runs distance_transform_edt on a mask
file, and verifies the output against scipy's exact Euclidean EDT (ground truth).
Also writes <maskfile>.out_py.bin for the cross-consumer identity check.
"""
import sys
import os
import struct
import numpy as np
from scipy import ndimage
from wasmtime import Store, Module, Instance, Memory

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # TextPatch
wasm = os.path.join(repo, "shared", "build", "textcore.wasm")
mask_file = sys.argv[1]

with open(mask_file, "rb") as f:
    h, w = struct.unpack("<ii", f.read(8))
    mask = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)

store = Store()
module = Module.from_file(store.engine, wasm)
inst = Instance(store, module, [])
ex = inst.exports(store)
mem = ex["memory"]
n = h * w
p_mask = ex["alloc"](store, n)
p_out = ex["alloc"](store, n * 4)
mem.write(store, mask.astype(np.uint8).tobytes(), p_mask)
ex["distance_transform_edt"](store, p_mask, h, w, p_out)
out_bytes = mem.read(store, p_out, p_out + n * 4)
out = np.frombuffer(bytes(out_bytes), dtype=np.float32).reshape(h, w)
ex["dealloc"](store, p_mask, n)
ex["dealloc"](store, p_out, n * 4)

# scipy ground truth: distance to nearest TEXT pixel (mask nonzero).
sp_text = ndimage.distance_transform_edt((mask == 0).astype(np.uint8))
sp_bg = ndimage.distance_transform_edt((mask != 0).astype(np.uint8))
err_text = float(np.max(np.abs(out - sp_text)))
err_bg = float(np.max(np.abs(out - sp_bg)))
print(
    f"py: {os.path.basename(mask_file)} {h}x{w} "
    f"scipy(text-src) maxerr={err_text:.4f} scipy(bg-src) maxerr={err_bg:.4f}"
)
assert err_text < 1e-2, f"wasm mismatch vs scipy(text): {err_text}"

out.tofile(mask_file + ".out_py.bin")
print(f"  wrote {os.path.basename(mask_file)}.out_py.bin")
