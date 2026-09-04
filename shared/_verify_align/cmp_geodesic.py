"""Compare Rust geodesic fields (dbg_geodesic_fields -> *_geoB/geoDrg/geoDgb.bin)
against the Python `_geodesic_background` original on the warm test image 556."""
import os, sys, json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deglow_io")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
from text_eraser.text_select import _geodesic_background, _deglow_full_green_v2

IDS = ["1787822778556"]


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


def load_f32(name, shape):
    return np.fromfile(os.path.join(IO, name), dtype=np.float32).reshape(shape)


def load_u8(name, shape):
    return np.fromfile(os.path.join(IO, name), dtype=np.uint8).reshape(shape)


def stats(a, b, name):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print(f"  {name}: max={d.max():.3f} mean={d.mean():.3f} #>1={int((d>1).sum())} #>3={int((d>3).sum())}")


for hid in IDS:
    orig = decode_orig(hid)
    H, W = orig.shape[:2]
    tmask = mask_from_boxes(hid, H, W)
    _, _, zone = _deglow_full_green_v2(orig, tmask, strength=1.0, zone_ratio=0.6,
                                        zone_expand=24, protect_px=1,
                                        deglow_chroma_keep=False, return_zone=True)
    zone = zone.astype(np.uint8)
    zone_bool = zone.astype(bool)

    r = orig[..., 0].astype(np.int16)
    g = orig[..., 1].astype(np.int16)
    b = orig[..., 2].astype(np.int16)
    max_rb = np.maximum(r, b)
    greenness = np.maximum(g - max_rb, 0).astype(np.float32)
    k3 = np.ones((3, 3), np.uint8)
    geo_mask = cv2.erode(zone.astype(np.uint8), k3, iterations=3) > 0
    dout = cv2.distanceTransform((~zone_bool).astype(np.uint8), cv2.DIST_L2, 5)
    ring_clean = (~zone_bool) & (dout >= 10.0) & (dout <= 26.0) & (greenness <= 6)
    rg = (r - g).astype(np.float32)
    gb = (g - b).astype(np.float32)
    B, (D_rg, D_gb) = _geodesic_background(orig, geo_mask, extra=[rg, gb], extra_src=ring_clean)

    rsB = load_f32(f"{hid}_geoB.bin", (H, W, 3))
    rsDrg = load_f32(f"{hid}_geoDrg.bin", (H, W))
    rsDgb = load_f32(f"{hid}_geoDgb.bin", (H, W))

    print(f"\n===== {hid} geodesic fields =====")
    stats(B, rsB, "B(RGB)")
    stats(D_rg, rsDrg, "D_rg")
    stats(D_gb, rsDgb, "D_gb")

print("\ndone")
