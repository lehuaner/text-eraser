"""Byte/md5 comparison across the three cross-end runners.

Run AFTER gen.py + node.js + browser_glue.js have dumped their *_*.bin files.
Prints an md5 matrix and per-channel byte equality. Parity is proven when
py == node == brw for all four channels with diff_bytes == 0.
"""
import sys, os, hashlib

def md5(p):
    return hashlib.md5(open(p, "rb").read()).digest().hex()

base = sys.argv[1] if len(sys.argv) > 1 else "."
sets = {
    "python":  ["py_result.bin", "py_fill.bin", "py_clean.bin", "py_zone.bin"],
    "node":    ["node_result.bin", "node_fill.bin", "node_clean.bin", "node_zone.bin"],
    "browser": ["brw_result.bin", "brw_fill.bin", "brw_clean.bin", "brw_zone.bin"],
}
data = {k: {f: md5(os.path.join(base, f)) for f in files} for k, files in sets.items()}

print("md5 matrix:")
print(f"{'channel':10s} {'python':34s} {'node':34s} {'browser':34s}")
for ch in ("result", "fill", "clean", "zone"):
    py = data["python"][f"py_{ch}.bin"]
    nd = data["node"][f"node_{ch}.bin"]
    br = data["browser"][f"brw_{ch}.bin"]
    print(f"{ch:10s} {py} {nd} {br}")

print("\nbyte equality:")
ok = True
for ch in ("result", "fill", "clean", "zone"):
    a = open(os.path.join(base, f"py_{ch}.bin"), "rb").read()
    b = open(os.path.join(base, f"node_{ch}.bin"), "rb").read()
    c = open(os.path.join(base, f"brw_{ch}.bin"), "rb").read()
    same_pn = a == b
    same_pb = a == c
    ok = ok and same_pn and same_pb
    print(f"  {ch:8s} len py/node/brw = {len(a)}/{len(b)}/{len(c)}  py==node:{same_pn}  py==brw:{same_pb}")
print("\nALL IDENTICAL" if ok else "\nDIVERGENCE DETECTED")
