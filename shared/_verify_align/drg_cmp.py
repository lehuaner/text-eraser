"""Compare the geodesic chroma extra D_rg between cv2 and wasm.
cv2 side: call _geodesic_background directly to get D_rg.
wasm side: recover D_rg from clean via glow = (orig_g - clean_g)/s, glow = D_rg - (r-g) -> D_rg = glow + (r-g).
"""
import os, sys, json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import run as R
from text_eraser.text_select import _deglow_full_green_v2, _geodesic_background


def decode_orig(hid):
    raw = open(os.path.join(ROOT, "data", "history", hid, "orig.bin"), "rb").read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_from_boxes(hid, H, W):
    meta = json.load(open(os.path.join(ROOT, "data", "history", hid, "meta.json")))
    m = np.zeros((H, W), np.uint8)
    for b in meta.get("boxes", []):
        m[b["y0"]:b["y1"], b["x0"]:b["x1"]] = 255
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)


def main():
    for hid in ["1788062081665", "1788077005814"]:
        orig = decode_orig(hid); H, W = orig.shape[:2]
        tmask = mask_from_boxes(hid, H, W)
        rgb = orig.astype(np.float32)
        r = rgb[..., 0]; g = rgb[..., 1]; b = rgb[..., 2]
        P = dict(strength=1.0, zone_ratio=0.6, zone_expand=10, protect_px=1,
                 chroma_keep=1, edge=1, direction=None)
        wasm = R.wasm_run(orig, tmask, tmask, **P)

        # --- cv2 D_rg via _geodesic_background ---
        _gn, _, zone0 = _deglow_full_green_v2(rgb, tmask, strength=1.0, zone_ratio=0.6,
                                              zone_expand=10, protect_px=1,
                                              deglow_chroma_keep=True, return_zone=True)
        green = np.maximum(g.astype(np.int16) - np.maximum(r, b), 0).astype(np.float32)
        zone = zone0 > 0
        k3 = np.ones((3, 3), np.uint8)
        geo_mask = cv2.erode(zone.astype(np.uint8), k3, iterations=3) > 0
        _dout = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
        _ring_clean = ((~zone) & (_dout >= 10.0) & (_dout <= 26.0) & (green <= 6))
        if _ring_clean.any():
            _, (D_rg, D_gb) = _geodesic_background(rgb, geo_mask,
                extra=[(r - g).astype(np.float32), (g - b).astype(np.float32)],
                extra_src=_ring_clean)
        else:
            D_rg = None

        # --- wasm D_rg recovered from clean ---
        clean = wasm["clean"].astype(np.float32)
        glow_w = (g - clean[..., 1]) / 1.0
        drg_w = glow_w + (r - g)

        if D_rg is not None:
            d = np.abs(D_rg - drg_w)
            print(f"{hid}: cv2 D_rg vs wasm-recovered D_rg  maxdiff={d.max():.3f}  mean={d.mean():.3f}  #px={int((d>0.5).sum())}  (total {H*W})")
            # where do they differ? compare to zone
            print(f"   diff inside zone0: {int((d[zone0>0]>0.5).sum())}/{int((zone0>0).sum())}")
        else:
            print(f"{hid}: cv2 D_rg is None (no clean ring)")


if __name__ == "__main__":
    main()
