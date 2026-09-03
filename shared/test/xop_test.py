"""Cross-end consistency + ground-truth test for the shared WASM core.

For every operator:
  * run it via the Python binding (wasmtime) and the Node binding (WebAssembly)
    on the SAME input, assert the raw outputs are BYTE-IDENTICAL
    -> proves "frontend and backend share one algorithm set".
  * (optional, if cv2/scipy present) compare against the reference implementation.

Run with the backend-capable Python (needs wasmtime + cv2 + scipy):
  python xop_test.py
"""
import os
import struct
import subprocess
import sys

import numpy as np

try:
    import cv2  # noqa: F401  (only needed for the optional cv2 ground-truth checks)
except Exception:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bindings"))
from textcore import get_core  # noqa: E402

NODE = r"C:/Users/乐幻/.workbuddy/binaries/node/versions/22.22.2-2/node.exe"
CORE = get_core()

rng = np.random.default_rng(20260902)


def write_in(path, arr, h, w):
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", h, w))
        f.write(np.ascontiguousarray(arr).tobytes())


def run_node(op, infile, outfile, *args):
    cmd = [NODE, os.path.join(HERE, "xop_node.cjs"), op, infile, outfile] + [str(a) for a in args]
    subprocess.run(cmd, check=True)


def cmp_bytes(py_bytes, node_bytes, name):
    if py_bytes == node_bytes:
        print(f"  [OK] {name}: Node == Python (wasmtime) byte-identical ({len(py_bytes)} bytes)")
        return True
    # find first diff
    a = py_bytes if isinstance(py_bytes, (bytes, bytearray)) else py_bytes.tobytes()
    b = node_bytes if isinstance(node_bytes, (bytes, bytearray)) else node_bytes.tobytes()
    n = min(len(a), len(b))
    i = next((i for i in range(n) if a[i] != b[i]), n)
    print(f"  [FAIL] {name}: differ at byte {i} (len {len(a)} vs {len(b)})")
    return False


def test_rgb_to_gray():
    print("rgb_to_gray")
    rgb = rng.integers(0, 256, (40, 55, 3), dtype=np.uint8)
    h, w = rgb.shape[:2]
    write_in("t_in.bin", rgb, h, w)
    py = CORE.rgb_to_gray(rgb, h, w)
    run_node("rgb_to_gray", "t_in.bin", "t_node.bin")
    node = np.frombuffer(open("t_node.bin", "rb").read(), dtype=np.uint8).reshape(h, w)
    ok = cmp_bytes(py.tobytes(), node.tobytes(), "rgb_to_gray")
    try:
        import cv2
        gt = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        print(f"    cv2 maxdiff={int(np.max(np.abs(py.astype(int)-gt.astype(int))))}")
    except Exception as e:
        print(f"    (cv2 skip: {e})")
    return ok


def test_distance():
    print("distance_transform_edt")
    mask = (rng.random((37, 41)) > 0.6).astype(np.uint8)
    h, w = mask.shape
    write_in("t_in.bin", mask, h, w)
    py = CORE.distance_transform_edt(mask, h, w)
    run_node("distance", "t_in.bin", "t_node.bin")
    node = np.frombuffer(open("t_node.bin", "rb").read(), dtype=np.float32).reshape(h, w)
    ok = cmp_bytes(py.tobytes(), node.tobytes(), "distance")
    try:
        from scipy import ndimage
        gt = ndimage.distance_transform_edt((mask == 0).astype(np.uint8))
        print(f"    scipy maxerr={float(np.max(np.abs(py-gt))):.6f}")
    except Exception as e:
        print(f"    (scipy skip: {e})")
    return ok


def test_otsu():
    print("threshold_otsu")
    g = rng.integers(0, 256, (38, 46), dtype=np.uint8)
    h, w = g.shape
    write_in("t_in.bin", g, h, w)
    thr, py = CORE.threshold_otsu(g, h, w)
    run_node("otsu", "t_in.bin", "t_node.bin")
    node = np.frombuffer(open("t_node.bin", "rb").read(), dtype=np.uint8).reshape(h, w)
    ok = cmp_bytes(py.tobytes(), node.tobytes(), "otsu")
    try:
        import cv2
        t2, b2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print(f"    cv2 thr={t2} ours={thr} bin maxdiff={int(np.max(np.abs(py.astype(int)-b2.astype(int))))}")
    except Exception as e:
        print(f"    (cv2 skip: {e})")
    return ok


def test_morph():
    print("morphology")
    ok_all = True
    # RECT only for the cross-end run: both ends build an identical all-ones kernel, so
    # node/python must be byte-identical. The exact cv2-ellipse fidelity is validated by
    # text_eraser/test_shared_core_smoke.py (which forwards cv2.getStructuringElement).
    for ksize, shape, op in [(3, "rect", "dilate"), (5, "rect", "dilate"),
                             (3, "rect", "erode"), (4, "rect", "erode"), (7, "rect", "dilate")]:
        mask = (rng.random((50, 50)) > 0.55).astype(np.uint8)
        # pad so border semantics (skip vs constant-0) don't confound the cv2 check
        pad = ksize + 1
        pmask = np.pad(mask, pad, constant_values=0)
        h, w = pmask.shape
        k = cv2.getStructuringElement(cv2.MORPH_RECT if shape == "rect" else cv2.MORPH_ELLIPSE, (ksize, ksize))
        kbits = (k != 0).astype(np.uint8)
        kh, kw = k.shape
        write_in("t_in.bin", pmask, h, w)
        py = CORE.morphology(pmask, h, w, kbits, kh, kw, op)
        run_node("morph", "t_in.bin", "t_node.bin", ksize, 1 if shape == "rect" else 0, 1 if op == "dilate" else 0)
        node = np.frombuffer(open("t_node.bin", "rb").read(), dtype=np.uint8).reshape(h, w)
        if not cmp_bytes(py.tobytes(), node.tobytes(), f"morph {ksize}{shape[0]}{op[0]}"):
            ok_all = False
        try:
            if cv2 is not None:
                fn = cv2.dilate if op == "dilate" else cv2.erode
                gt = fn(pmask, k)
                # compare interior (exclude padding border where semantics differ)
                d = np.max(np.abs(py[pad:-pad, pad:-pad].astype(int) - gt[pad:-pad, pad:-pad].astype(int)))
                print(f"    cv2({ksize}{shape[0]}{op[0]}) interior maxdiff={int(d)}")
            else:
                print("    (cv2 unavailable — skipped ground-truth check)")
        except Exception as e:
            print(f"    (cv2 skip: {e})")
    return ok_all


def test_cc():
    print("connected_components")
    # synthetic shapes so components are well-defined
    m = np.zeros((60, 80), dtype=np.uint8)
    m[10:20, 10:25] = 1
    m[30:45, 50:70] = 1
    m[5:8, 60:65] = 1
    h, w = m.shape
    write_in("t_in.bin", m, h, w)
    n, py_lab, py_stats = CORE.connected_components(m, h, w)
    run_node("cc", "t_in.bin", "t_node.bin")
    raw = open("t_node.bin", "rb").read()
    nn = struct.unpack("<i", raw[:4])[0]
    off = 4
    node_lab = np.frombuffer(raw, dtype=np.int32, count=h * w, offset=off).reshape(h, w)
    off += h * w * 4
    node_stats = np.frombuffer(raw, dtype=np.int32, count=nn * 5, offset=off).reshape(nn, 5)
    ok = cmp_bytes(py_lab.tobytes(), node_lab.tobytes(), "cc.labels")
    ok2 = cmp_bytes(np.array([(s["left"], s["top"], s["width"], s["height"], s["area"]) for s in py_stats], dtype=np.int32).tobytes(),
                   node_stats.tobytes(), "cc.stats")
    try:
        import cv2
        _, gt_lab, gt_stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        # compare component stats as sets (label order differs)
        gt_set = set(tuple(int(x) for x in gt_stats[i]) for i in range(1, gt_stats.shape[0]))
        our_set = set((s["left"], s["top"], s["width"], s["height"], s["area"]) for s in py_stats[1:])
        print(f"    cv2 components={len(gt_set)} ours={len(our_set)} stats_match={gt_set == our_set}")
    except Exception as e:
        print(f"    (cv2 skip: {e})")
    return ok and ok2


def test_resize():
    print("resize")
    ok_all = True
    for (h, w, h2, w2, interp) in [(30, 40, 60, 80, "cubic"), (33, 47, 17, 23, "linear")]:
        if interp == "cubic":
            src = rng.integers(0, 256, (h, w), dtype=np.uint8)
            write_in("t_in.bin", src, h, w)
            py = CORE.resize_gray_cubic(src, h2, w2)
            run_node("resize_cubic", "t_in.bin", "t_node.bin", h2, w2)
            node = np.frombuffer(open("t_node.bin", "rb").read(), dtype=np.uint8).reshape(h2, w2)
            if not cmp_bytes(py.tobytes(), node.tobytes(), "resize_cubic"):
                ok_all = False
        else:
            src = rng.random((h, w)).astype(np.float32)
            write_in("t_in.bin", src, h, w)
            py = CORE.resize_float_linear(src, h2, w2)
            run_node("resize_linear", "t_in.bin", "t_node.bin", h2, w2)
            node = np.frombuffer(open("t_node.bin", "rb").read(), dtype=np.float32).reshape(h2, w2)
            if not cmp_bytes(py.tobytes(), node.tobytes(), "resize_linear"):
                ok_all = False
        try:
            import cv2
            if interp == "cubic":
                gt = cv2.resize(src, (w2, h2), interpolation=cv2.INTER_CUBIC)
                d = int(np.max(np.abs(py.astype(int) - gt.astype(int))))
            else:
                gt = cv2.resize(src, (w2, h2), interpolation=cv2.INTER_LINEAR)
                d = float(np.max(np.abs(py - gt)))
            print(f"    cv2({interp}) maxdiff={d}")
        except Exception as e:
            print(f"    (cv2 skip: {e})")
    return ok_all


if __name__ == "__main__":
    os.chdir(HERE)
    results = [
        test_rgb_to_gray(),
        test_distance(),
        test_otsu(),
        test_morph(),
        test_cc(),
        test_resize(),
    ]
    print("\n=== SUMMARY ===")
    print("ALL CROSS-END IDENTICAL" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)
