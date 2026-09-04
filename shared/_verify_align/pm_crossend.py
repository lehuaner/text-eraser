"""Cross-end byte-identity check for patchmatch_inpaint (Python wasmtime vs Node).

Generates deterministic inputs (gradient bg + text bars + optional sample mask),
runs the SAME wasm through both runtimes, compares raw float32 outputs bytewise.
Exercises: default mode with a >512px boundary (multi-chunk pass), direction mode,
sample-mask filtering. Pure smoke: no cv2 needed.
"""
import hashlib
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bindings"))
from textcore import get_core  # noqa: E402

NODE = r"C:/Users/乐幻/.workbuddy/binaries/node/versions/22.22.2-2/node.exe"
CORE = get_core()


def make_case(H, W, bars, with_sample):
    rng = np.random.default_rng(7)
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    rgb = np.stack([
        190 + 0.25 * yy + 3 * np.sin(xx / 9.0),
        180 + 0.20 * yy + 0 * xx,
        160 + 0.15 * yy + 3 * np.cos(xx / 11.0),
    ], axis=-1)
    rgb += rng.normal(0, 1.2, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.float32)
    mask = np.zeros((H, W), np.uint8)
    for (y0, y1, x0, x1) in bars:
        mask[y0:y1, x0:x1] = 255
        rgb[y0:y1, x0:x1] = 255.0  # white text pixels inside the hole
    sample = None
    if with_sample:
        sample = np.where(mask > 0, 0, 255).astype(np.uint8)
        # carve two protected streaks out of the sample region (like excl masks)
        sample[:, W // 3:W // 3 + 3] = 0
        sample[H // 2:H // 2 + 3, :] = 0
    return rgb, mask, sample


def run_node(rgb, mask, sample, p, direction_deg, seed, tag):
    H, W = mask.shape
    fin = os.path.join(HERE, f"_pm_in_{tag}.bin")
    fout = os.path.join(HERE, f"_pm_out_{tag}.bin")
    with open(fin, "wb") as f:
        f.write(np.ascontiguousarray(rgb, np.float32).tobytes())
        f.write(mask.tobytes())
        f.write((sample if sample is not None else mask).tobytes())
        f.write(np.array([H, W, p, 1 if sample is not None else 0,
                          float(direction_deg), seed], dtype=np.float32).tobytes())
    node_js = os.path.join(HERE, "_pm_node.mjs")
    subprocess.run([NODE, node_js, fin, fout], check=True)
    with open(fout, "rb") as f:
        return f.read()


def run_py(rgb, mask, sample, p, direction_deg, seed):
    out = CORE.patchmatch_inpaint(rgb, mask.shape[0], mask.shape[1], mask,
                                  sample, p, direction_deg, seed)
    return np.ascontiguousarray(out, np.float32).tobytes()


cases = [
    ("big_bars", 200, 200, [(30, 36, 5, 195), (60, 66, 5, 195), (90, 150, 5, 195)], False, -1.0, None),
    ("big_bars_sample", 200, 200, [(30, 36, 5, 195), (60, 66, 5, 195), (90, 150, 5, 195)], True, -1.0, None),
    ("top_band", 200, 200, [(30, 36, 5, 195), (60, 66, 5, 195), (90, 150, 5, 195)], True, -1.0, "top"),
    ("bottom_band", 200, 200, [(30, 36, 5, 195), (60, 66, 5, 195), (90, 150, 5, 195)], True, -1.0, "bottom"),
    ("dir_mode", 200, 200, [(70, 130, 40, 160)], True, 45.0, None),
    ("small", 120, 120, [(50, 70, 40, 80)], False, -1.0, None),
    ("edge_touch", 150, 150, [(0, 10, 0, 150), (140, 150, 0, 150)], True, -1.0, None),
]

ok = True
outs = {}
for tag, H, W, bars, with_sample, direction, band in cases:
    rgb, mask, sample = make_case(H, W, bars, with_sample)
    if band == "top":
        sample = np.zeros((H, W), np.uint8); sample[:25, :] = 255
    elif band == "bottom":
        sample = np.zeros((H, W), np.uint8); sample[H - 25:, :] = 255
    py = run_py(rgb, mask, sample, 7, direction, 0)
    nd = run_node(rgb, mask, sample, 7, direction, 0, tag)
    outs[tag] = py
    h_py = hashlib.md5(py).hexdigest()
    h_nd = hashlib.md5(nd).hexdigest()
    status = "OK " if py == nd else "FAIL"
    if py != nd:
        ok = False
        a = np.frombuffer(py, np.float32)
        b = np.frombuffer(nd, np.float32)
        d = np.argmax(a != b) if (a != b).any() else -1
        print(f"[{status}] {tag}: py={h_py} node={h_nd} first_diff_idx={d}")
    else:
        print(f"[{status}] {tag}: byte-identical  md5={h_py}")

# sensitivity: the two band-constrained fills must differ from each other,
# otherwise the sample path is not actually influencing the result.
if outs.get("top_band") == outs.get("bottom_band"):
    ok = False
    print("[FAIL] sensitivity: top_band == bottom_band (sample path ignored!)")
else:
    print("[OK ] sensitivity: top_band != bottom_band (sample path active)")

for tag, *_ in cases:
    for ext in ("in", "out"):
        pth = os.path.join(HERE, f"_pm_{ext}_{tag}.bin")
        if os.path.exists(pth):
            os.remove(pth)

print("CROSS-END FILL:", "ALL OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
