"""Smoke test: backend shared-core wiring for erase_text_glyphs.

Builds a synthetic warm-bg + green-glow + white-text image, runs the shared
wasm pipeline through the backend binding (`text_eraser._shared_core`), and
asserts:
  - using_shared_core() is True (wasm loaded),
  - result differs from clean inside the text hole (fill happened),
  - deglow_full_green_v2 also runs through the core.
No ML detector required.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_eraser import _shared_core as sc

H, W = 80, 100
rng = np.random.default_rng(7)
rgb = np.zeros((H, W, 3), np.float32)
for c, b in enumerate((210, 190, 168)):
    rgb[..., c] = b + (rng.random((H, W)).astype(np.float32) - 0.5) * 6.0
# green glow blob
yy, xx = np.mgrid[0:H, 0:W]
gl = ((yy - H * 0.4) / (H * 0.3)) ** 2 + ((xx - W * 0.5) / (W * 0.35)) ** 2 <= 1.0
rgb[gl, 0] += 10
rgb[gl, 1] += 70
rgb[gl, 2] += 5
# white text bars -> the mask to fill
tmask = np.zeros((H, W), np.uint8)
for y0 in (25, 45):
    tmask[y0:y0 + 5, 25:75] = 255

print("using_shared_core =", sc.using_shared_core())
assert sc.using_shared_core(), "wasm core failed to load"

clean, _core, _zone = sc.deglow_full_green_v2(rgb, tmask, strength=1.0, zone_ratio=0.6,
                                       zone_expand=10, protect_px=1, chroma_keep=1)
print("deglow clean shape", clean.shape, "core_px", int((_core > 0).sum()))

res = sc.erase_text_glyphs(rgb, tmask, None, strength=1.0, zone_ratio=0.6,
                           zone_expand=10, protect_px=1, chroma_keep=1,
                           edge=1, direction_deg=-1.0, seed=0)
assert res is not None, "erase_text_glyphs returned None"
result, fill, clean2, zone = res
print("result", result.shape, "fill_px", int((fill > 0).sum()),
      "clean", clean2.shape, "zone_px", int((zone > 0).sum()))

# fill must have happened inside the text hole: result != clean2 at text pixels
th = (tmask > 0)
diff = np.abs(result.astype(int) - clean2.astype(int)).sum(axis=2)
assert diff[th].sum() > 0, "no fill applied where text mask is"
print("OK: fill applied at text pixels; backend shared-core wiring works")
