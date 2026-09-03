"""Parametrized verify_align regression — direction / edge_aware / soft_expand / large-ROI.

Compares the wasm shared-core path (`erase_text_glyphs` / `patchmatch_inpaint`)
against the pure-cv2 reference path (TEXTCORE_BACKEND forced off) for the four
parameter regimes called out in the ②–⑥ audit.

For each scenario we report:
  RESULT gap   (mean|d|, max|d|, #px with |d|>3)   wasm vs cv2
  FILL   gap   (mask_filled mismatch, #px)           wasm vs cv2
plus, where the wasm core is known to *not* implement a param
(edge_aware / soft_expand), an extra "wasm ignores param" check:
  wasm(param=True) vs wasm(param=False)  -> should be ~0 if the core drops it.

Run:  python shared/_verify_align/verify_align_params.py
"""
import io, json, os, sys
import numpy as np
from PIL import Image

ROOT = r"D:\Code\Project\Python\TextPatch"
sys.path.insert(0, ROOT)
from text_eraser import eraser
import text_eraser._shared_core as sc

HID = "1787766251689"
BASE = rf"D:\Code\Project\Python\TextPatch\data\history\{HID}"
OUT = os.path.join(ROOT, "shared", "_verify_align", "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# path toggles
# ---------------------------------------------------------------------------
def force_wasm():
    sc._core_ok = None          # next _get_core() reloads the wasm
    ok = sc._get_core() is not None
    return ok

def force_cv2():
    sc._core_ok = False
    sc._core = None
    return sc._get_core() is None

# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------
def make_stripe(h, w, angle=60, base=130, amp=70):
    """Directional wood-grain-like stripe texture (gray)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle)
    proj = xx * np.cos(a) + yy * np.sin(a)
    stripe = (np.sin(proj / 7.0) * 0.5 + 0.5)
    g = (base + amp * stripe).clip(0, 255).astype(np.uint8)
    return np.stack([g, g, g], -1)

def add_text_block(rgb, boxes, val=235):
    out = rgb.copy()
    for (x0, y0, x1, y1) in boxes:
        out[y0:y1, x0:x1] = val
    return out

def make_large_text_bar(H, W):
    """Big image with a long central text bar -> ROI span triggers MAX_ROI=1536."""
    rgb = make_stripe(H, W, angle=90, base=140, amp=60)
    # central horizontal bar spanning most of the width
    bar_h = 70
    y0, y1 = H // 2 - bar_h // 2, H // 2 + bar_h // 2
    x0, x1 = 60, W - 60
    rgb[y0:y1, x0:x1] = 240
    return rgb, [(x0, y0, x1, y1)]

# ---------------------------------------------------------------------------
# scenario runner
# ---------------------------------------------------------------------------
def run_erase(rgb, params):
    res = eraser.erase_text(rgb, return_mask=True, auto_edge=False, **params)
    result, mask_filled, m = res
    return np.asarray(result, np.uint8), np.asarray(mask_filled, np.uint8), m

def stats(name, a, b):
    a = np.asarray(a); b = np.asarray(b)
    d = a.astype(int) - b.astype(int)
    ad = np.abs(d)
    if ad.ndim == 3:
        changed = int((ad.sum(2) > 3).sum())
    else:
        changed = int((ad > 3).sum())
    print(f"    [{name}] mean|d|={ad.mean():.3f}  max|d|={int(ad.max())}  "
          f">3px={changed:,}  tot={ad.size:,}")
    return ad.mean(), int(ad.max()), changed

# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
report = {}

def scenario_direction(angle=60):
    print(f"\n### S1 direction mode (deglow off, angle={angle}) ###")
    H, W = 900, 900
    rgb = make_stripe(H, W, angle=angle)
    boxes = [(300, 410, 600, 490), (330, 300, 560, 360)]
    rgb = add_text_block(rgb, boxes)
    params = dict(deglow_scheme="off", direction=float(angle), edge=1,
                  q_off=55.0, ml_max_side=960)
    force_wasm(); rw, fw, mw = run_erase(rgb, params)
    force_cv2(); rc, fc, mc = run_erase(rgb, params)
    stats("RESULT", rw, rc)
    stats("FILL mask", fw, fc)
    report[f"direction_{angle}"] = dict(wasm_method=mw.get("method"),
                                        cv2_method=mc.get("method"),
                                        result_max=stats_last_max(rw, rc),
                                        fill_mismatch=int((fw > 0).sum() ^ (fc > 0).sum()) or int(np.abs((fw > 0).sum() - (fc > 0).sum())))
    Image.fromarray(rw).save(f"{OUT}/dir{wm()}_wasm.png")
    Image.fromarray(rc).save(f"{OUT}/dir{wm()}_cv2.png")

def stats_last_max(a, b):
    return int(np.abs(a.astype(int) - b.astype(int)).max())

def wm():
    return ""

def scenario_edge_aware():
    print("\n### S3 edge_aware=True (座驾, v2 default path) ###")
    raw = open(f"{BASE}/orig.bin", "rb").read()
    rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), np.uint8)
    base_params = dict(deglow_scheme="v2", edge=1, q_off=55.0,
                       ml_max_side=960, deglow_strength=1.0)
    # wasm ignores edge_aware in the v2 core path -> True vs False should be ~equal
    force_wasm(); rw_t, _, _ = run_erase(rgb, {**base_params, "edge_aware": True})
    force_wasm(); rw_f, _, _ = run_erase(rgb, {**base_params, "edge_aware": False})
    m_wasm, mx_wasm, c_wasm = stats("wasm(edge_aware=T) vs wasm(edge_aware=F)", rw_t, rw_f)
    # cv2 reference APPLIES edge_aware
    force_wasm(); rwa, fwa, mwa = run_erase(rgb, {**base_params, "edge_aware": True})
    force_cv2(); rca, fca, mca = run_erase(rgb, {**base_params, "edge_aware": True})
    stats("RESULT (edge_aware=T)", rwa, rca)
    stats("FILL mask (edge_aware=T)", fwa, fca)
    report["edge_aware"] = dict(wasm_ignores_param=(c_wasm < 50),
                                result_max=int(np.abs(rwa.astype(int) - rca.astype(int)).max()),
                                fill_mismatch=int(np.abs((fwa > 0).sum() - (fca > 0).sum())))

def scenario_soft_expand():
    print("\n### S4 soft_expand>0 (座驾, v2 default path) ###")
    raw = open(f"{BASE}/orig.bin", "rb").read()
    rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), np.uint8)
    base_params = dict(deglow_scheme="v2", edge=1, q_off=55.0,
                       ml_max_side=960, deglow_strength=1.0, deglow_mask_soft=20.0)
    force_wasm(); rw_t, _, _ = run_erase(rgb, base_params)
    force_wasm(); rw_f, _, _ = run_erase(rgb, {**base_params, "deglow_mask_soft": 0.0})
    m_wasm, mx_wasm, c_wasm = stats("wasm(soft_expand=20) vs wasm(soft_expand=0)", rw_t, rw_f)
    force_wasm(); rwa, fwa, mwa = run_erase(rgb, base_params)
    force_cv2(); rca, fca, mca = run_erase(rgb, base_params)
    stats("RESULT (soft_expand=20)", rwa, rca)
    stats("FILL mask (soft_expand=20)", fwa, fca)
    report["soft_expand"] = dict(wasm_ignores_param=(c_wasm < 50),
                                 result_max=int(np.abs(rwa.astype(int) - rca.astype(int)).max()),
                                 fill_mismatch=int(np.abs((fwa > 0).sum() - (fca > 0).sum())))

def scenario_large_roi():
    print("\n### S5 large ROI / MAX_ROI=1536 trigger ###")
    H, W = 1700, 1500
    rgb, boxes = make_large_text_bar(H, W)
    params = dict(deglow_scheme="off", direction=None, edge=1, q_off=55.0,
                  ml_max_side=960)
    force_wasm(); rw, fw, mw = run_erase(rgb, params)
    force_cv2(); rc, fc, mc = run_erase(rgb, params)
    stats("RESULT", rw, rc)
    stats("FILL mask", fw, fc)
    report["large_roi"] = dict(wasm_method=mw.get("method"), cv2_method=mc.get("method"),
                               result_max=int(np.abs(rw.astype(int) - rc.astype(int)).max()),
                               fill_mismatch=int(np.abs((fw > 0).sum() - (fc > 0).sum())),
                               roi_span=max(boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]))

if __name__ == "__main__":
    print("wasm available:", force_wasm())
    scenario_direction(60)
    scenario_direction(90)
    scenario_edge_aware()
    scenario_soft_expand()
    scenario_large_roi()
    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n=== REPORT ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved ->", OUT)
