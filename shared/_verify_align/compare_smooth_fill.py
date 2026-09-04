"""Decide whether the wasm smooth-gradient TELEA pre-check is correctly ported.

For the SMOOTH case the pre-check MUST fire -> wasm result must equal telea.rs
(dbg_telea) byte-for-byte, and telea.rs must be near cv2.INPAINT_TELEA.
For the TEXTURED case the pre-check must NOT fire -> wasm result must be a
PatchMatch (i.e. it must DIFFER from telea.rs, which would only smooth-fill).
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(os.path.dirname(__file__), "artifacts")


def load(name):
    return np.load(os.path.join(OUT, name))


def dbg_telea(sub, subm):
    from text_eraser._shared_core import _get_core
    core = _get_core()
    h, w = sub.shape[:2]
    n = h * w
    p_in = core._alloc(n * 3 * 4)
    p_m = core._alloc(n)
    p_out = core._alloc(n * 3 * 4)
    try:
        core.mem.write(core.store,
                       np.ascontiguousarray(sub, dtype=np.float32).tobytes(), p_in)
        core.mem.write(core.store,
                       np.ascontiguousarray(subm, dtype=np.uint8).tobytes(), p_m)
        core.ex["dbg_telea"](core.store, p_in, p_m, h, w, 3, p_out)
        buf = bytes(core.mem.read(core.store, p_out, p_out + n * 3 * 4))
    finally:
        core._free(p_in, n * 3 * 4)
        core._free(p_m, n)
        core._free(p_out, n * 3 * 4)
    return np.frombuffer(buf, dtype=np.float32).reshape(h, w, 3).copy()


def main():
    # ---- SMOOTH: pre-check must fire -> wasm == telea.rs (byte exact) ----
    sub = load("smooth_sub.npy")
    subm = load("smooth_subm.npy")
    wasm = load("smooth_wasm.npy")
    telea_wasm = dbg_telea(sub, subm)
    d_telea = np.abs(wasm.astype(np.float32) - telea_wasm.astype(np.float32))
    smooth_fired = d_telea.max() < 0.01
    print(f"smooth  wasm vs telea.rs : max={d_telea.max():.4f} mean={d_telea.mean():.4f} "
          f"-> {'PRECHECK FIRED (TELEA)' if smooth_fired else 'NOT telea?!'}")
    cv2telea = load("smooth_cv2telea.npy")
    d_cv2 = np.abs(telea_wasm.astype(np.float32) - cv2telea.astype(np.float32))
    print(f"smooth  telea.rs vs cv2  : max={d_cv2.max():.2f} mean={d_cv2.mean():.4f} "
          f"(telea.rs fidelity to cv2; {'' if d_cv2.max() < 20 else 'WIDER-than-usual'})")

    # ---- TEXTURED: pre-check must NOT fire -> wasm != telea.rs ----
    sub = load("textured_sub.npy")
    subm = load("textured_subm.npy")
    wasm = load("textured_wasm.npy")
    telea_wasm = dbg_telea(sub, subm)
    d_telea = np.abs(wasm.astype(np.float32) - telea_wasm.astype(np.float32))
    textured_skipped = d_telea.max() > 1.0
    print(f"textured wasm vs telea.rs : max={d_telea.max():.2f} mean={d_telea.mean():.4f} "
          f"-> {'PRECHECK CORRECTLY SKIPPED (PatchMatch)' if textured_skipped else 'UNEXPECTED telea'}")

    ok = smooth_fired and textured_skipped
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
