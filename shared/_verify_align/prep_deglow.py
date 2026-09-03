"""Prep raw inputs for the native Rust deglow comparison.

Decodes data/history/<id>/orig.bin (BGR bytes) -> RGB f32, builds a text mask
from meta.json boxes, and writes raw .bin + meta.json into _verify_align/deglass_io/.
The Rust native test (shared/tests/deglow_verify.rs) reads these, runs the exact
Rust `deglow_full_green_v2`, and writes back clean/core/zone raw files. Then
cmp_deglow.py diffs Rust vs the Python `_deglow_full_green_v2` original.
"""
import os, sys, json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deglow_io")
os.makedirs(IO, exist_ok=True)

IDS = ["1787767611178", "1787822778556"]

# canonical params matching Python `_deglow_full_green_v2` defaults
PARAMS = dict(strength=1.0, zone_ratio=0.6, zone_expand=24, protect_px=1, chroma_keep=0)


def decode_orig(hid):
    raw = open(os.path.join(ROOT, "data", "history", hid, "orig.bin"), "rb").read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_from_boxes(hid, H, W):
    meta = json.load(open(os.path.join(ROOT, "data", "history", hid, "meta.json")))
    m = np.zeros((H, W), np.uint8)
    for b in meta.get("boxes", []):
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        m[y0:y1, x0:x1] = 255
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)


for hid in IDS:
    orig = decode_orig(hid)
    H, W = orig.shape[:2]
    tmask = mask_from_boxes(hid, H, W)
    rgb_f32 = orig.astype(np.float32)
    rgb_f32.tofile(os.path.join(IO, f"{hid}_rgb.bin"))
    tmask.tofile(os.path.join(IO, f"{hid}_tmask.bin"))
    meta = dict(id=hid, H=H, W=W, **PARAMS)
    json.dump(meta, open(os.path.join(IO, f"{hid}_meta.json"), "w"))
    print(f"prepped {hid}: {W}x{H} tmask_px={int((tmask>0).sum())}")
print("done")
