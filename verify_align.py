"""Measure the pixel gap between the current wasm erase path and the pure-cv2
reference path (TEXTCORE_BACKEND=0). Used to localize which stage of the wasm
port diverges from the Python original (text_eraser/patch_fill.py + eraser.py).
"""
import io, json, sys, os
import numpy as np
from PIL import Image

sys.path.insert(0, r"D:\Code\Project\Python\TextPatch")
HID = "1787766251689"
base = rf"D:\Code\Project\Python\TextPatch\data\history\{HID}"
meta = json.load(open(f"{base}/meta.json"))
raw = open(f"{base}/orig.bin", "rb").read()
img = Image.open(io.BytesIO(raw)).convert("RGB")
rgb = np.asarray(img, dtype=np.uint8)

from text_eraser import eraser

p = meta["params"]


def run(force_cv2: bool):
    if force_cv2:
        # disable wasm core entirely -> pure cv2 reference path
        import text_eraser._shared_core as sc
        sc._core_ok = False
        sc._core = None
    res = eraser.erase_text(
        rgb,
        deglow_scheme=p.get("deglow_scheme", "v2"),
        edge=p.get("edge", 1),
        q_off=p.get("q_off", 55.0),
        max_area_ratio=p.get("max_area_ratio", 0.4),
        max_box_ratio=p.get("max_box_ratio", 0.4),
        ml_max_side=p.get("ml_max_side", 960),
        direction=p.get("direction"),
        edge_aware=p.get("edge_aware", False),
        fill_white=p.get("fill_white", True),
        fill_max_dist=p.get("fill_max_dist", 12),
        deglow_strength=p.get("deglow_strength", 1.0),
        deglow_zone_ratio=p.get("deglow_zone_ratio", 0.6),
        deglow_zone_expand=p.get("deglow_zone_expand", 10),
        deglow_protect_px=p.get("deglow_protect_px", 1),
        deglow_chroma_keep=p.get("deglow_chroma_keep", True),
        return_mask=True,
    )
    result, mask_filled, m = res
    return (np.asarray(result, np.uint8), np.asarray(mask_filled, np.uint8),
            np.asarray(m.get("deglow_img"), np.uint8), m)


print("=== wasm path ===")
rw, fw, dw, mw = run(False)
print("  method:", mw.get("method"), "mask_filled_pix:", int((fw > 0).sum()))
print("=== cv2 reference path ===")
rc, fc, dc, mc = run(True)
print("  method:", mc.get("method"), "mask_filled_pix:", int((fc > 0).sum()))

np.save(f"{base}/_v_result_wasm.npy", rw)
np.save(f"{base}/_v_result_cv2.npy", rc)
np.save(f"{base}/_v_fill_wasm.npy", fw)
np.save(f"{base}/_v_fill_cv2.npy", fc)
np.save(f"{base}/_v_deglow_wasm.npy", dw)
np.save(f"{base}/_v_deglow_cv2.npy", dc)

# ---- compare ----
def stats(name, a, b):
    a = np.asarray(a); b = np.asarray(b)
    d = a.astype(int) - b.astype(int)
    ad = np.abs(d)
    if ad.ndim == 3:
        changed = int((ad.sum(2) > 3).sum())
    else:
        changed = int((ad > 3).sum())
    print(f"[{name}] mean|diff|: {ad.mean():.3f}  max|diff|: {int(ad.max())}  "
          f">3px: {changed:,}  total_px: {ad.size:,}")


def vs_orig(name, a):
    """How much this result changed vs the ORIGINAL image (text removed => small in bbox)."""
    a = np.asarray(a)
    d = a.astype(int) - rgb.astype(int)
    ad = np.abs(d)
    print(f"[{name} vs ORIG] mean|diff|: {ad.mean():.3f}  max|diff|: {int(ad.max())}  "
          f">3px: {int((ad.sum(2) > 3).sum()):,}")

print("\n=== stage gaps (wasm - cv2) ===")
stats("RESULT (whole)", rw, rc)
stats("DEGLOW img", dw, dc)
stats("FILL mask", fw, fc)
# region inside fill mask
fm = (fc > 0) | (fw > 0)
if fm.any():
    stats("RESULT inside fill-region", rw[fm], rc[fm])
# text bbox
boxes = meta["boxes"]
if boxes:
    xs = [b["x0"] for b in boxes] + [b["x1"] for b in boxes]
    ys = [b["y0"] for b in boxes] + [b["y1"] for b in boxes]
    x0, x1 = max(0, min(xs)), min(rgb.shape[1], max(xs))
    y0, y1 = max(0, min(ys)), min(rgb.shape[0], max(ys))
    stats(f"RESULT text-bbox[{y0}:{y1},{x0}:{x1}]", rw[y0:y1, x0:x1], rc[y0:y1, x0:x1])

print("\n=== each path vs ORIGINAL (text removed => low in bbox) ===")
vs_orig("RESULT wasm", rw)
vs_orig("RESULT cv2", rc)
stats("RESULT bg top-left(20x20)", rw[0:20, 0:20], rc[0:20, 0:20])

# ---- save images for visual inspection ----
Image.fromarray(rgb).save(f"{base}/_v_orig.png")
Image.fromarray(rw).save(f"{base}/_v_wasm.png")
Image.fromarray(rc).save(f"{base}/_v_cv2.png")
Image.fromarray(dw).save(f"{base}/_v_deglow.png")
# text-bbox crops
yb, xb = slice(y0, y1), slice(x0, x1)
Image.fromarray(rgb[yb, xb]).save(f"{base}/_v_crop_orig.png")
Image.fromarray(rw[yb, xb]).save(f"{base}/_v_crop_wasm.png")
Image.fromarray(rc[yb, xb]).save(f"{base}/_v_crop_cv2.png")
# diff visualization (wasm - cv2), scaled
dd = rw.astype(int) - rc.astype(int)
dd8 = np.clip(np.abs(dd) * 3, 0, 255).astype(np.uint8)
Image.fromarray(dd8).save(f"{base}/_v_diff_wasm_cv2.png")
print("\nsaved crops to", base)
