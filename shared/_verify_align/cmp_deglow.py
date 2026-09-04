"""Diff the Rust native deglow (deglow_verify.rs -> *_clean/core/zone.bin) against
the Python `_deglow_full_green_v2` original on the two test images. Prints stats and
writes comparison PNGs into the same deglow_io dir."""
import os, sys, json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deglow_io")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "text_eraser"))

from text_eraser.text_select import _deglow_full_green_v2

IDS = ["1787767611178", "1787822778556"]


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


def load_u8(name, n):
    return np.fromfile(os.path.join(IO, name), dtype=np.uint8).reshape(n)


def load_rgb(name, n):
    return np.fromfile(os.path.join(IO, name), dtype=np.uint8).reshape(n[0], n[1], 3)


def stats(a, b, name):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    maxd = int(d.max()) if d.size else 0
    meand = float(d.mean()) if d.size else 0.0
    ndiff = int((d > 0).sum())
    ndiff2 = int((d > 2).sum())
    print(f"  {name}: maxdiff={maxd}  meandiff={meand:.3f}  #>0={ndiff}  #>2={ndiff2}")
    return maxd, d


for hid in IDS:
    orig = decode_orig(hid)
    H, W = orig.shape[:2]
    tmask = mask_from_boxes(hid, H, W)

    py_clean, py_core, py_zone = _deglow_full_green_v2(
        orig, tmask, strength=1.0, zone_ratio=0.6, zone_expand=24,
        protect_px=1, deglow_chroma_keep=False, return_zone=True)

    rs_clean = load_rgb(f"{hid}_clean.bin", (H, W))
    rs_core = load_u8(f"{hid}_core.bin", (H * W,))
    rs_zone = load_u8(f"{hid}_zone.bin", (H * W,))
    rs_core = rs_core.reshape(H, W)
    rs_zone = rs_zone.reshape(H, W)

    print(f"\n===== {hid} ({W}x{H}) =====")
    print("--- CLEAN (RGB u8) ---")
    stats(py_clean[..., 0], rs_clean[..., 0], "R")
    stats(py_clean[..., 1], rs_clean[..., 1], "G")
    stats(py_clean[..., 2], rs_clean[..., 2], "B")
    print("--- CORE mask ---")
    stats(py_core, rs_core, "core")
    print("--- ZONE mask ---")
    stats(py_zone.astype(np.uint8), rs_zone, "zone")

    # zone agreement
    pa = (py_zone.astype(bool) == rs_zone.astype(bool))
    print(f"  zone pixel-agree: {pa.mean()*100:.2f}%  py_px={int(py_zone.sum())} rs_px={int(rs_zone.sum())}")
    ca = (py_core.astype(bool) == rs_core.astype(bool))
    print(f"  core pixel-agree: {ca.mean()*100:.2f}%  py_px={int(py_core.sum())} rs_px={int(rs_core.sum())}")

    # recovered glow field (orig_g - clean_g)/s on changed region
    def rec_glow(o, c):
        return (o[..., 1].astype(np.float32) - c[..., 1].astype(np.float32))
    pg = rec_glow(orig, py_clean); rg = rec_glow(orig, rs_clean)
    chg_py = np.abs(pg) > 0.5
    chg_rs = np.abs(rg) > 0.5
    print(f"  python subtracted px={int(chg_py.sum())}  rust subtracted px={int(chg_rs.sum())}")
    both = chg_py & chg_rs
    only_py = chg_py & ~chg_rs
    only_rs = ~chg_py & chg_rs
    print(f"  both={int(both.sum())} only_py={int(only_py.sum())} only_rs={int(only_rs.sum())}")
    ok = both & (orig[..., 1].astype(np.float32) > 1.5) & (orig[..., 1].astype(np.float32) < 253.5)
    gd = np.abs(pg[ok] - rg[ok])
    if ok.sum() > 0:
        print(f"  glow maxdiff(shared unclamped)={gd.max():.3f} mean={gd.mean():.3f} #>{0.5}={int((gd>0.5).sum())}")

    # save montage: orig | py_clean | rs_clean | |py-rs| (G channel scaled x4)
    diff = np.abs(py_clean.astype(np.int16) - rs_clean.astype(np.int16)).astype(np.uint8)
    diff_s = np.clip(diff.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
    row = np.hstack([orig, py_clean, rs_clean, diff_s])
    outp = os.path.join(IO, f"{hid}_cmp.png")
    cv2.imwrite(outp, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    print(f"  saved {outp}")

print("\ndone")
