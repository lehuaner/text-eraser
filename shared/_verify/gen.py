"""Cross-end byte-identical verification for erase_text_glyphs.

Generates a deterministic synthetic input (warm bg + green glow + text mask),
runs it through the shared WASM via the Python (wasmtime) binding, and dumps the
four outputs. Node (WebAssembly.instantiate) reads the SAME input binaries so the
only difference between the two runs is the host — the wasm bytes are identical.

Outputs (md5 printed):
  rgb.bin   H*W*3 float32 LE  (shared input)
  tm.bin    H*W   uint8        (shared input, text mask)
  py_result.bin / py_fill.bin / py_clean.bin / py_zone.bin
"""
import os
import sys
import hashlib
import numpy as np

H, W = 64, 80
N = H * W
OUT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(OUT, "..", "bindings"))
from textcore import get_core  # noqa: E402


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def build_input():
    rng = np.random.default_rng(12345)
    # warm background with a gentle vertical gradient
    bg = np.zeros((H, W, 3), dtype=np.float32)
    for c, (base, grad) in enumerate([(205, 18), (188, -10), (165, 6)]):
        g = base + grad * (np.arange(H)[:, None] / float(H)) * 255.0 / 255.0
        bg[:, :, c] = g
    # global subtle noise
    bg += (rng.random((H, W, 3)).astype(np.float32) - 0.5) * 8.0
    rgb = bg.copy()

    # green glow blob near center: lift G strongly, R/B a little -> green halo
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H * 0.45, W * 0.5
    ry, rx = H * 0.28, W * 0.33
    glow = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    rgb[glow, 0] += 12.0
    rgb[glow, 1] += 72.0
    rgb[glow, 2] += 6.0

    # text mask: a few glyph bars
    tm = np.zeros((H, W), dtype=np.uint8)
    for y0 in (18, 30, 42):
        tm[y0:y0 + 4, 20:60] = 255
    # a small closed shape to force non-empty mask + PatchMatch fill
    tm[25:35, 28:40] = 255
    return rgb.astype(np.float32), tm


def main():
    rgb, tm = build_input()
    rgb.tofile(os.path.join(OUT, "rgb.bin"))
    tm.tofile(os.path.join(OUT, "tm.bin"))

    core = get_core()
    # Use the backend's real _erase_deglow_v2 defaults so this cross-end check
    # proves true frontend/backend parity (same params -> same wasm -> same bytes).
    result, fill, clean, zone = core.erase_text_glyphs(rgb, H, W, tm, None,
                                                      1.15, 0.6, 10, 1, 1, 1, -1.0, 0)
    result.tofile(os.path.join(OUT, "py_result.bin"))
    fill.tofile(os.path.join(OUT, "py_fill.bin"))
    clean.tofile(os.path.join(OUT, "py_clean.bin"))
    zone.tofile(os.path.join(OUT, "py_zone.bin"))

    for name in ("rgb.bin", "tm.bin", "py_result.bin", "py_fill.bin",
                 "py_clean.bin", "py_zone.bin"):
        print(f"{name:16s} {md5(os.path.join(OUT, name))}")
    print("n_text_px =", int((fill > 0).sum()))


if __name__ == "__main__":
    main()
