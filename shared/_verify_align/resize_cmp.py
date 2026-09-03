"""Validate cv2-compatible resizers (AREA / CUBIC / NEAREST) in the wasm core
against cv2.resize. Goal: maxdiff ~ float epsilon so the geodesic/harmonic
background fields match the original Python pipeline exactly.

Uses random float images + the two history images, at several scale factors.
"""
import os, sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from shared.bindings.textcore import get_core

core = get_core()


def maxdiff(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    return float(d.max()), int((d > 0.5).sum()), int((d > 0).sum())


def cv2_resize(img, h2, w2, kind):
    interp = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST}[kind]
    return cv2.resize(img, (w2, h2), interpolation=interp)


def test_one(name, img, h, w, ch, h2, w2):
    # img is HxWxch float32 already (values arbitrary, float)
    for kind in ("area", "cubic", "nearest"):
        wasm = core.dbg_resize(img, h, w, ch, h2, w2, kind)
        ref = cv2_resize(img, h2, w2, kind)
        md, nd05, nd = maxdiff(wasm, ref)
        flag = "OK " if md < 1.0 else "!!!"
        print(f"  [{flag}] {name:22s} {kind:7s} {h}x{w}->{h2}x{w2} ch{ch}  maxdiff={md:.4f} #>0.5={nd05} #>0={nd}")


print("=== random float images (smooth gradients + noise, float32) ===")
rng = np.random.default_rng(7)
for (h, w, ch), (h2, w2) in [
    ((139, 109, 3), (69, 54)), ((139, 109, 3), (278, 218)),
    ((81, 84, 3), (40, 42)), ((200, 150, 1), (100, 75)),
    ((100, 140, 3), (33, 46)), ((64, 64, 1), (32, 32)),
    ((139, 109, 3), (3, 3)), ((81, 84, 3), (160, 168)),
]:
    base = rng.random((h, w, ch)).astype(np.float32) * 255.0
    # add a smooth gradient so area/cubic have structure to interpolate
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (base + (yy[:, :, None] * 1.3 + xx[:, :, None] * 0.7)) % 255.0
    test_one(f"rand{h}x{w}", base, h, w, ch, h2, w2)

print("\n=== real history images (RGB uint8 -> float32) ===")
for hid in ("1788062081665", "1788077005814"):
    raw = open(os.path.join(ROOT, "data", "history", hid, "orig.bin"), "rb").read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = rgb.shape[:2]
    scale = 4 if min(h, w) >= 160 else 2
    h2, w2 = max(2, h // scale), max(2, w // scale)
    test_one(f"hist{hid}", rgb, h, w, 3, h2, w2)
    # upscale path (cubic/area/nearest used for B upsample)
    test_one(f"hist{hid}_up", rgb, h, w, 3, h * 2, w * 2)

print("\n=== real images as 1ch gray (geodesic lum / extras) ===")
for hid in ("1788062081665", "1788077005814"):
    raw = open(os.path.join(ROOT, "data", "history", hid, "orig.bin"), "rb").read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    scale = 4 if min(h, w) >= 160 else 2
    h2, w2 = max(2, h // scale), max(2, w // scale)
    test_one(f"hist{hid}_gray", g[:, :, None], h, w, 1, h2, w2)
