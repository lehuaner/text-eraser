"""Cross-consumer identity check.

The browser (node-edt.cjs) and backend (py-edt.py) both run the SAME wasm.
For identical inputs they must produce bit-identical floats. This proves one
algorithm serves both ends.
"""
import sys
import numpy as np

f = sys.argv[1]
a = np.fromfile(f + ".out_node.bin", dtype=np.float32)
b = np.fromfile(f + ".out_py.bin", dtype=np.float32)
assert a.shape == b.shape, (a.shape, b.shape)
maxabs = float(np.max(np.abs(a - b)))
status = "IDENTICAL" if maxabs == 0.0 else ("OK" if maxabs < 1e-5 else "MISMATCH")
print(f"compare {f}: node vs py maxabs={maxabs:.6f} -> {status}")
assert maxabs == 0.0, f"two consumers diverged: {maxabs}"
