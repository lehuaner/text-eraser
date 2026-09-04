"""Localize the clean divergence: recover each side's per-pixel glow field from
clean (= orig_g - glow*s, s=1.0) and compare. Also compare the subtraction mask
(m_zone) by checking which pixels were actually changed in clean vs orig."""
import os, sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "text_eraser"))
import run as R

IDS = ["1788062081665", "1788077005814"]


def decode_orig(hid):
    raw = open(os.path.join(ROOT, "data", "history", hid, "orig.bin"), "rb").read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_from_boxes(hid, H, W):
    import json
    meta = json.load(open(os.path.join(ROOT, "data", "history", hid, "meta.json")))
    m = np.zeros((H, W), np.uint8)
    for b in meta.get("boxes", []):
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        m[y0:y1, x0:x1] = 255
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)


def recovered_glow(orig, clean, s=1.0):
    # glow = (orig_g - clean_g)/s ; only meaningful where not clamped
    og = orig[..., 1].astype(np.float32)
    cg = clean[..., 1].astype(np.float32)
    glow = (og - cg) / s
    changed = np.abs(og - cg) > 0.5  # pixel where subtraction actually happened
    clamped = (cg <= 0.5) | (cg >= 254.5)
    return glow, changed, clamped


for hid in IDS:
    orig = decode_orig(hid)
    H, W = orig.shape[:2]
    tmask = mask_from_boxes(hid, H, W)
    P = dict(strength=1.0, zone_ratio=0.6, zone_expand=10, protect_px=1,
             chroma_keep=1, edge=1, direction=None)
    ref = R.cv2_ref(orig, tmask, tmask, **P)
    wasm = R.wasm_run(orig, tmask, tmask, **P)
    cg_ref, ch_ref, cl_ref = recovered_glow(orig, ref["clean"], 1.0)
    cg_w, ch_w, cl_w = recovered_glow(orig, wasm["clean"], 1.0)

    # where did cv2 actually subtract? where did wasm?
    print(f"\n===== {hid} ({W}x{H}) =====")
    print(f"cv2 subtracted px (clean changed): {int(ch_ref.sum())}")
    print(f"wasm subtracted px (clean changed): {int(ch_w.sum())}")
    both = ch_ref & ch_w
    only_cv2 = ch_ref & ~ch_w
    only_wasm = ~ch_ref & ch_w
    print(f"  both={int(both.sum())}  only_cv2={int(only_cv2.sum())}  only_wasm={int(only_wasm.sum())}")
    # glow diff on the 'both' region (neither clamped)
    ok = both & ~cl_ref & ~cl_w
    d = np.abs(cg_ref - cg_w)
    if ok.sum() > 0:
        print(f"glow maxdiff on shared-unclamped region: {d[ok].max():.3f}  mean={d[ok].mean():.3f}  #px={int((d[ok]>0.5).sum())}")
    else:
        print("glow: no shared unclamped region to compare")
    # overall clean per-channel diff
    for ch, name in enumerate("rgb"):
        dd = np.abs(ref["clean"][..., ch].astype(np.int16) - wasm["clean"][..., ch].astype(np.int16))
        print(f"clean {name}: maxdiff={dd.max()} #diff={int((dd>0).sum())}")
