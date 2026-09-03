"""Python backend binding for the shared WASM core (textcore.wasm).

This is the SINGLE integration point the backend uses to call shared algorithms.
It loads the same .wasm the browser loads, so any operator added here is
automatically available — and identical — on both ends.

Requires: wasmtime  (pip install wasmtime)
"""
import os
import numpy as np
from wasmtime import Store, Module, Instance, Memory

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WASM = os.path.join(_REPO, "shared", "build", "textcore.wasm")


class TextCore:
    def __init__(self, wasm_path: str = _WASM):
        self.store = Store()
        self.module = Module.from_file(self.store.engine, wasm_path)
        self.inst = Instance(self.store, self.module, [])
        self.ex = self.inst.exports(self.store)
        self.mem = self.ex["memory"]

    def _alloc(self, nbytes: int) -> int:
        return int(self.ex["alloc"](self.store, nbytes))

    def _free(self, ptr: int, nbytes: int) -> None:
        self.ex["dealloc"](self.store, ptr, nbytes)

    def distance_transform_edt(self, mask_u8: np.ndarray, h: int, w: int) -> np.ndarray:
        """Distance to nearest TEXT (nonzero) pixel. Mirrors
        cv2.distanceTransform((cur == 0), DIST_L2, 3) semantics but exact."""
        n = h * w
        p_mask = self._alloc(n)
        p_out = self._alloc(n * 4)
        try:
            self.mem.write(self.store, np.ascontiguousarray(mask_u8, dtype=np.uint8).tobytes(), p_mask)
            self.ex["distance_transform_edt"](self.store, p_mask, h, w, p_out)
            out_bytes = bytes(self.mem.read(self.store, p_out, p_out + n * 4))
        finally:
            self._free(p_mask, n)
            self._free(p_out, n * 4)
        return np.frombuffer(out_bytes, dtype=np.float32).reshape(h, w).copy()

    def rgb_to_gray(self, rgb, h: int, w: int) -> np.ndarray:
        """RGB (H*W*3, any dtype 0..255) -> grayscale u8 (H*W)."""
        n = h * w
        arr = np.ascontiguousarray(rgb, dtype=np.float32)
        p_in = self._alloc(n * 3 * 4)
        p_out = self._alloc(n)
        try:
            self.mem.write(self.store, arr.tobytes(), p_in)
            self.ex["rgb_to_gray"](self.store, p_in, h, w, p_out)
            out = np.frombuffer(bytes(self.mem.read(self.store, p_out, p_out + n)), dtype=np.uint8).reshape(h, w).copy()
        finally:
            self._free(p_in, n * 3 * 4)
            self._free(p_out, n)
        return out

    def threshold_otsu(self, u8, h: int, w: int):
        """Otsu threshold. Returns (thr: float, bin: u8 H*W with 0/255)."""
        n = h * w
        p_in = self._alloc(n)
        p_out = self._alloc(n)
        try:
            self.mem.write(self.store, np.ascontiguousarray(u8, dtype=np.uint8).tobytes(), p_in)
            thr = float(self.ex["threshold_otsu"](self.store, p_in, p_out, n))
            bin = np.frombuffer(bytes(self.mem.read(self.store, p_out, p_out + n)), dtype=np.uint8).reshape(h, w).copy()
        finally:
            self._free(p_in, n)
            self._free(p_out, n)
        return thr, bin

    def morphology(self, mask_u8, h: int, w: int, kern_u8, kh: int, kw: int, op: str = "dilate") -> np.ndarray:
        """Binary morphology on a 0/1 (or 0/255) mask using an explicit kernel bitmap.

        `kern_u8` is a kh*kw u8 array (1 = include / 0 = skip) — e.g. the exact
        cv2.getStructuringElement output. Returns 0/1 u8 H*W. op: "dilate"/"erode".
        """
        n = h * w
        nk = kh * kw
        op_code = 0 if op == "erode" else 1
        p_in = self._alloc(n)
        p_out = self._alloc(n)
        p_k = self._alloc(nk)
        try:
            self.mem.write(self.store, np.ascontiguousarray(mask_u8, dtype=np.uint8).tobytes(), p_in)
            self.mem.write(self.store, np.ascontiguousarray(kern_u8, dtype=np.uint8).tobytes(), p_k)
            self.ex["morphology"](self.store, p_in, p_out, h, w, p_k, kh, kw, op_code)
            out = np.frombuffer(bytes(self.mem.read(self.store, p_out, p_out + n)), dtype=np.uint8).reshape(h, w).copy()
        finally:
            self._free(p_in, n)
            self._free(p_out, n)
            self._free(p_k, nk)
        return out

    def connected_components(self, mask_u8, h: int, w: int):
        """8-connected components. Returns (n, labels int32 H*W, stats list of dicts).
        stats[0] is the background placeholder."""
        n = h * w
        p_in = self._alloc(n)
        p_labels = self._alloc(n * 4)
        try:
            self.mem.write(self.store, np.ascontiguousarray(mask_u8, dtype=np.uint8).tobytes(), p_in)
            ncomp = int(self.ex["connected_components"](self.store, p_in, p_labels, h, w))
            labels = np.frombuffer(bytes(self.mem.read(self.store, p_labels, p_labels + n * 4)), dtype=np.int32).reshape(h, w).copy()
        finally:
            self._free(p_in, n)
            # keep p_labels for the stats call below
        p_stats = self._alloc(ncomp * 5 * 4)
        try:
            self.ex["connected_components_stats"](self.store, p_labels, p_stats, h, w, ncomp)
            raw = np.frombuffer(bytes(self.mem.read(self.store, p_stats, p_stats + ncomp * 5 * 4)), dtype=np.int32).reshape(ncomp, 5).copy()
        finally:
            self._free(p_stats, ncomp * 5 * 4)
            self._free(p_labels, n * 4)
        stats = [{"left": int(r[0]), "top": int(r[1]), "width": int(r[2]),
                  "height": int(r[3]), "area": int(r[4])} for r in raw]
        return ncomp, labels, stats

    def resize_gray_cubic(self, u8, h2: int, w2: int) -> np.ndarray:
        h, w = u8.shape[:2]
        n = h * w
        n2 = h2 * w2
        p_in = self._alloc(n)
        p_out = self._alloc(n2)
        try:
            self.mem.write(self.store, np.ascontiguousarray(u8, dtype=np.uint8).tobytes(), p_in)
            self.ex["resize_gray_cubic"](self.store, p_in, p_out, h, w, h2, w2)
            out = np.frombuffer(bytes(self.mem.read(self.store, p_out, p_out + n2)), dtype=np.uint8).reshape(h2, w2).copy()
        finally:
            self._free(p_in, n)
            self._free(p_out, n2)
        return out

    def resize_float_linear(self, f32, h2: int, w2: int) -> np.ndarray:
        h, w = f32.shape[:2]
        n = h * w
        n2 = h2 * w2
        p_in = self._alloc(n * 4)
        p_out = self._alloc(n2 * 4)
        try:
            self.mem.write(self.store, np.ascontiguousarray(f32, dtype=np.float32).tobytes(), p_in)
            self.ex["resize_float_linear"](self.store, p_in, p_out, h, w, h2, w2)
            out = np.frombuffer(bytes(self.mem.read(self.store, p_out, p_out + n2 * 4)), dtype=np.float32).reshape(h2, w2).copy()
        finally:
            self._free(p_in, n * 4)
            self._free(p_out, n2 * 4)
        return out

    def patchmatch_inpaint(self, rgb_f32, h: int, w: int, mask_u8,
                           sample_u8=None, p: int = 7, direction_deg: float = -1.0,
                           seed: int = 0) -> np.ndarray:
        """PatchMatch inpainting — THE shared fill routine (browser + backend call this).

        rgb_f32 : H*W*3 float32 (values 0..255). Mutated to the filled result.
        mask_u8 : H*W, >0 = hole to fill.
        sample_u8 : optional H*W, >0 = allowed source region.
        p : patch size (odd). direction_deg : < -1e30 disables direction mode.
        Returns H*W*3 float32 filled image.
        """
        n = h * w
        arr = np.ascontiguousarray(rgb_f32, dtype=np.float32)
        m = np.ascontiguousarray(mask_u8, dtype=np.uint8)
        has_sample = 0
        p_sample = 0
        p_s = 0
        if sample_u8 is not None:
            has_sample = 1
            s = np.ascontiguousarray(sample_u8, dtype=np.uint8)
            p_s = self._alloc(n)
            self.mem.write(self.store, s.tobytes(), p_s)
        p_in = self._alloc(n * 3 * 4)
        p_out = self._alloc(n * 3 * 4)
        p_mask = self._alloc(n)
        try:
            self.mem.write(self.store, arr.tobytes(), p_in)
            self.mem.write(self.store, m.tobytes(), p_mask)
            self.ex["patchmatch_inpaint"](
                self.store, p_in, h, w, p_mask, p_s, has_sample, p,
                float(direction_deg), int(seed) & 0xFFFFFFFF, p_out)
            out = np.frombuffer(bytes(self.mem.read(self.store, p_out, p_out + n * 3 * 4)),
                               dtype=np.float32).reshape(h, w, 3).copy()
        finally:
            self._free(p_in, n * 3 * 4)
            self._free(p_out, n * 3 * 4)
            self._free(p_mask, n)
            if p_s:
                self._free(p_s, n)
        return out

    def synthesize_masks(self, text_mask_u8, h: int, w: int, edge: int = 1,
                        limit_u8=None) -> tuple:
        """Text-mask synthesis — THE shared operator (browser + backend call this).

        text_mask_u8 : H*W, >0 = text. edge : >0 dilate / <0 erode / 0 identity
                      (diameter = abs(edge)*2+1 ellipse). limit_u8 : optional H*W,
                      >0 restricts the fill region (sample unaffected).
        Returns (fill_mask H*W u8 0/255, sample_mask H*W u8 0/255) where
        sample = whole - fill.
        """
        n = h * w
        tm = np.ascontiguousarray(text_mask_u8, dtype=np.uint8)
        has_limit = 0
        p_lim = 0
        if limit_u8 is not None:
            has_limit = 1
            p_lim = self._alloc(n)
            self.mem.write(self.store, np.ascontiguousarray(limit_u8, dtype=np.uint8).tobytes(), p_lim)
        p_in = self._alloc(n)
        p_fill = self._alloc(n)
        p_smpl = self._alloc(n)
        try:
            self.mem.write(self.store, tm.tobytes(), p_in)
            self.ex["synthesize_masks"](
                self.store, p_in, h, w, edge, p_lim, has_limit, p_fill, p_smpl)
            fill = np.frombuffer(bytes(self.mem.read(self.store, p_fill, p_fill + n)),
                                dtype=np.uint8).reshape(h, w).copy()
            sample = np.frombuffer(bytes(self.mem.read(self.store, p_smpl, p_smpl + n)),
                                  dtype=np.uint8).reshape(h, w).copy()
        finally:
            self._free(p_in, n)
            self._free(p_fill, n)
            self._free(p_smpl, n)
            if p_lim:
                self._free(p_lim, n)
        return fill, sample

    def deglow_full_green_v2(self, rgb_f32, h: int, w: int, tmask_u8,
                            strength: float = 1.0, zone_ratio: float = 0.6,
                            zone_expand: int = 0, protect_px: int = 0,
                            chroma_keep: int = 0) -> tuple:
        """De-glow (full green v2) — THE shared operator (browser + backend call this).

        rgb_f32  : H*W*3 float32 (0..255).
        tmask_u8 : H*W, >0 = known text.
        Returns (clean H*W*3 u8, core_mask H*W u8 0/255).
        """
        n = h * w
        arr = np.ascontiguousarray(rgb_f32, dtype=np.float32)
        tm = np.ascontiguousarray(tmask_u8, dtype=np.uint8)
        p_in = self._alloc(n * 3 * 4)
        p_tm = self._alloc(n)
        p_clean = self._alloc(n * 3)
        p_core = self._alloc(n)
        p_zone = self._alloc(n)
        try:
            self.mem.write(self.store, arr.tobytes(), p_in)
            self.mem.write(self.store, tm.tobytes(), p_tm)
            self.ex["deglow_full_green_v2"](
                self.store, p_in, h, w, p_tm,
                float(strength), float(zone_ratio), int(zone_expand),
                int(protect_px), int(chroma_keep), p_clean, p_core, p_zone)
            clean = np.frombuffer(bytes(self.mem.read(self.store, p_clean, p_clean + n * 3)),
                                 dtype=np.uint8).reshape(h, w, 3).copy()
            core = np.frombuffer(bytes(self.mem.read(self.store, p_core, p_core + n)),
                                 dtype=np.uint8).reshape(h, w).copy()
            zone = np.frombuffer(bytes(self.mem.read(self.store, p_zone, p_zone + n)),
                                 dtype=np.uint8).reshape(h, w).copy()
        finally:
            self._free(p_in, n * 3 * 4)
            self._free(p_tm, n)
            self._free(p_clean, n * 3)
            self._free(p_core, n)
            self._free(p_zone, n)
        return clean, core, zone

    def erase_text_glyphs(self, rgb_f32, h, w, tmask_u8, tmask2_u8=None,
                          strength: float = 1.0, zone_ratio: float = 0.6,
                          zone_expand: int = 0, protect_px: int = 0,
                          chroma_keep: int = 0, edge: int = 0,
                          direction_deg: float = -1.0, seed: int = 0,
                          edge_aware: int = 0, soft_expand: float = 0.0) -> tuple:
        """Single shared pipeline entry — run the FULL de-glow + mask-surgery +
        PatchMatch fill (browser + backend call this identically).

        rgb_f32   : H*W*3 float32 (0..255).
        tmask_u8  : H*W, >0 = known text (raw detect).
        tmask2_u8 : optional H*W second detect (on cleaned image) to union in.
        Returns (result H*W*3 u8, fill_mask H*W u8, clean H*W*3 u8, zone H*W u8).
        """
        n = h * w
        arr = np.ascontiguousarray(rgb_f32, dtype=np.float32)
        tm = np.ascontiguousarray(tmask_u8, dtype=np.uint8)
        if tmask2_u8 is None:
            tm2 = np.zeros((h, w), dtype=np.uint8)
        else:
            tm2 = np.ascontiguousarray(tmask2_u8, dtype=np.uint8)
        p_in = self._alloc(n * 3 * 4)
        p_tm = self._alloc(n)
        p_tm2 = self._alloc(n)
        p_result = self._alloc(n * 3)
        p_fill = self._alloc(n)
        p_clean = self._alloc(n * 3)
        p_zone = self._alloc(n)
        try:
            self.mem.write(self.store, arr.tobytes(), p_in)
            self.mem.write(self.store, tm.tobytes(), p_tm)
            self.mem.write(self.store, tm2.tobytes(), p_tm2)
            self.ex["erase_text_glyphs"](
                self.store, p_in, h, w, p_tm, p_tm2,
                float(strength), float(zone_ratio), int(zone_expand),
                int(protect_px), int(chroma_keep), int(edge),
                float(direction_deg), int(seed) & 0xFFFFFFFF,
                int(edge_aware), float(soft_expand),
                p_result, p_fill, p_clean, p_zone)
            result = np.frombuffer(bytes(self.mem.read(self.store, p_result, p_result + n * 3)),
                                   dtype=np.uint8).reshape(h, w, 3).copy()
            fill = np.frombuffer(bytes(self.mem.read(self.store, p_fill, p_fill + n)),
                                 dtype=np.uint8).reshape(h, w).copy()
            clean = np.frombuffer(bytes(self.mem.read(self.store, p_clean, p_clean + n * 3)),
                                  dtype=np.uint8).reshape(h, w, 3).copy()
            zone = np.frombuffer(bytes(self.mem.read(self.store, p_zone, p_zone + n)),
                                 dtype=np.uint8).reshape(h, w).copy()
        finally:
            self._free(p_in, n * 3 * 4)
            self._free(p_tm, n)
            self._free(p_tm2, n)
            self._free(p_result, n * 3)
            self._free(p_fill, n)
            self._free(p_clean, n * 3)
            self._free(p_zone, n)
        return result, fill, clean, zone


# Module-level singleton so the backend pays the load cost once.
_default = None


def get_core() -> TextCore:
    global _default
    if _default is None:
        _default = TextCore()
    return _default
