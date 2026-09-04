"""Bisect the geodesic B divergence: compare Rust vs Python at each stage
(rgb_s low-res, source-routed low-res, upscaled pre-gaussian)."""
import os, sys, json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deglow_io")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
from text_eraser.text_select import _geodesic_sources

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


def stats(a, b, name):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print(f"  {name}: max={d.max():.3f} mean={d.mean():.3f} #>1={int((d>1).sum())} #>3={int((d>3).sum())}")


for hid in IDS:
    orig = decode_orig(hid)
    H, W = orig.shape[:2]
    tmask = mask_from_boxes(hid, H, W)
    rgbs = load_f32(f"{hid}_st_rgbs.bin", None)  # shape unknown yet
    # determine h2,w2 from file size
    nb = os.path.getsize(os.path.join(IO, f"{hid}_st_rgbs.bin"))
    h2w2 = nb // (3 * 4)
    # infer h2,w2: scale=4 if min>=160 else 2
    scale = 4 if min(H, W) >= 160 else 2
    h2 = max(2, H // scale); w2 = max(2, W // scale)
    assert h2 * w2 == h2w2, f"{h2}x{w2} != {h2w2}"
    rgbs = load_f32(f"{hid}_st_rgbs.bin", (h2, w2, 3))
    blow = load_f32(f"{hid}_st_blow.bin", (h2, w2, 3))
    bup = load_f32(f"{hid}_st_bup.bin", (H, W, 3))

    # Python reference stages
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    scale = 4 if min(H, W) >= 160 else 2
    H2, W2 = max(2, H // scale), max(2, W // scale)
    rgb_s_py = cv2.resize(orig, (W2, H2), interpolation=cv2.INTER_AREA)
    lum = cv2.resize(gray, (W2, H2), interpolation=cv2.INTER_AREA)
    # src_mask = background (we don't have zone here; rebuild from deglow)
    # Recompute zone via _deglow_full_green_v2 return_zone
    from text_eraser.text_select import _deglow_full_green_v2
    _, _, zone = _deglow_full_green_v2(orig, tmask, strength=1.0, zone_ratio=0.6,
                                        zone_expand=24, protect_px=1,
                                        deglow_chroma_keep=False, return_zone=True)
    rz = cv2.resize((zone > 0).astype(np.uint8) * 255, (W2, H2), interpolation=cv2.INTER_NEAREST) > 127
    src_mask = ~rz
    src_y, src_x = _geodesic_sources(lum, src_mask)
    B_s = rgb_s_py[src_y, src_x].astype(np.float32)
    B_up = cv2.resize(B_s, (W, H), interpolation=cv2.INTER_CUBIC)
    B = cv2.GaussianBlur(B_up, (0, 0), 4.0)

    print(f"\n===== {hid} geodesic B stages =====")
    print(f"  h2,w2 = {H2},{W2}")
    stats(rgb_s_py, rgbs, "rgb_s (lowres AREA)")
    stats(B_s, blow, "b_low (source-routed)")
    stats(B_up, bup, "b_up (CUBIC upscale)")
    # also full B after gaussian
    geoB = load_f32(f"{hid}_geoB.bin", (H, W, 3))
    stats(B, geoB, "B (final)")

print("\ndone")
