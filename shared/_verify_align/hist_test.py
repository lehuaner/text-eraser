"""End-to-end de-glow comparison on two historical images (1788062081665, 1788077005814).

Compares the ORIGINAL cv2 pipeline (run.cv2_ref) vs the CURRENT wasm shared core
(run.wasm_run) on the real inputs, using the historical text-detection boxes as the
mask (most faithful to how these were originally processed). Produces:
  - per-channel maxdiff / #diff px for clean, zone, fill, result
  - greenness residual stats in the original glow region (success = ~0 leftover green)
  - side-by-side compare PNGs
"""
import os, sys, json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import run as R

HIST = os.path.join(ROOT, "data", "history")
OUT = os.path.join(ROOT, "shared", "_verify_align", "hist_cmp")
os.makedirs(OUT, exist_ok=True)

IDS = ["1788062081665", "1788077005814"]


def decode_orig(hid):
    raw = open(os.path.join(HIST, hid, "orig.bin"), "rb").read()
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_from_boxes(hid, H, W):
    meta = json.load(open(os.path.join(HIST, hid, "meta.json")))
    m = np.zeros((H, W), np.uint8)
    for b in meta.get("boxes", []):
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        m[y0:y1, x0:x1] = 255
    # mimic edge dilation (edge=1)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    return m


def greenness_stats(rgb, mask):
    rgb = np.asarray(rgb, np.float32)
    g = rgb[..., 1]; r = rgb[..., 0]; b = rgb[..., 2]
    green = g - np.maximum(r, b)
    if mask.sum() == 0:
        return None
    return float(green[mask > 0].mean()), float(green[mask > 0].max())


def save_compare(hid, orig, cv_clean, wv_clean, diff):
    H, W = orig.shape[:2]
    # orig | cv2 clean | wasm clean | diff*6 (scaled for visibility, 3ch)
    diff3 = np.clip(np.stack([diff, diff, diff], axis=2).astype(np.float32) * 6, 0, 255).astype(np.uint8)
    row = np.concatenate([orig, cv_clean, wv_clean, diff3], axis=1)
    cv2.imwrite(os.path.join(OUT, f"{hid}_compare.png"),
                cv2.cvtColor(row, cv2.COLOR_RGB2BGR))


def main():
    for hid in IDS:
        orig = decode_orig(hid)
        H, W = orig.shape[:2]
        tmask = mask_from_boxes(hid, H, W)
        # historical params
        P = dict(strength=1.0, zone_ratio=0.6, zone_expand=10, protect_px=1,
                 chroma_keep=1, edge=1, direction=None)
        ref = R.cv2_ref(orig, tmask, tmask, **P)
        wasm = R.wasm_run(orig, tmask, tmask, **P)
        print(f"\n===== {hid}  ({W}x{H}) =====")
        print(f"{'channel':8s} {'maxdiff':>8s} {'#diff_px':>10s}")
        for ch in ("clean", "zone", "fill", "result"):
            md, nd = R.maxdiff(ref[ch], wasm[ch])
            print(f"{ch:8s} {str(md):>8s} {str(nd):>10s}")
        # greenness residual in original strong_green region
        from text_eraser.text_select import _deglow_full_green_v2
        _, _, zone0 = _deglow_full_green_v2(orig, tmask, strength=1.0, zone_ratio=0.6,
                                             zone_expand=10, protect_px=1,
                                             deglow_chroma_keep=True, return_zone=True)
        # original glow region = where cv2 subtracted (zone0)
        gg = greenness_stats(orig, zone0)
        gc = greenness_stats(ref["clean"], zone0)
        gw = greenness_stats(wasm["clean"], zone0)
        print(f"greenness in glow(zone0) region: orig mean/max={gg}")
        print(f"  cv2 clean  green mean/max = {gc}")
        print(f"  wasm clean green mean/max = {gw}")
        # diff image
        diff = np.abs(np.asarray(ref["clean"], np.int32) - np.asarray(wasm["clean"], np.int32)).sum(axis=2)
        save_compare(hid, orig, ref["clean"], wasm["clean"], diff)
        # also save the two cleans + orig separately
        cv2.imwrite(os.path.join(OUT, f"{hid}_orig.png"), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT, f"{hid}_cv2_clean.png"), cv2.cvtColor(ref["clean"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT, f"{hid}_wasm_clean.png"), cv2.cvtColor(wasm["clean"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT, f"{hid}_cv2_result.png"), cv2.cvtColor(ref["result"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT, f"{hid}_wasm_result.png"), cv2.cvtColor(wasm["result"], cv2.COLOR_RGB2BGR))
        print(f"  saved compare + cleans + results to {OUT}")


if __name__ == "__main__":
    main()
