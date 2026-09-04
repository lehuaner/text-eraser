"""textcore.wasm binding — 共享算法核的唯一集成点（随包分发）。

加载与浏览器 Worker 完全相同的 `textcore.wasm`（打包在
``text_eraser/assets/textcore.wasm``），因此任何新增算子自动两端一致。

设计要点：
- **线程本地实例**: wasmtime 的 Store/Instance 不是线程安全的，且单实例上的
  alloc/dealloc 在并发请求下会互相踩内存。这里用 ``threading.local`` 让每个
  线程（FastAPI 线程池 worker / 批处理 worker）持有独立实例，从而：
  ① 并发请求天然隔离；② 多线程批量处理可获得真实并行度（受 GIL 限制见 README）。
- wasm 查找顺序: 包内 assets（pip 安装场景）→ 仓库 ``shared/build/``（开发场景）。
- 核加载失败直接抛 ``CoreLoadError``（不再静默回退 Python 实现 —— 0.3.0 起
  Python 核心已删除，wasm 是唯一算法核心）。

Requires: wasmtime  (pip install wasmtime)
"""
from __future__ import annotations

import os
import threading

import numpy as np
from wasmtime import Store, Module, Instance

__all__ = ["TextCore", "get_core", "CoreLoadError", "wasm_path"]


class CoreLoadError(RuntimeError):
    """textcore.wasm 加载失败（文件缺失 / wasmtime 未安装 / 实例化失败）。"""


def _package_wasm() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "textcore.wasm")


def wasm_path() -> str:
    """返回实际使用的 textcore.wasm 路径（包内资产优先，仓库 build 兜底）。"""
    candidates = [
        _package_wasm(),
        # 开发仓库场景: text_eraser/ 上一级是仓库根
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "shared", "build", "textcore.wasm"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise CoreLoadError(
        "textcore.wasm 未找到。已尝试:\n  " + "\n  ".join(candidates)
        + "\n若为源码运行, 请先构建: cd shared && cargo build --target wasm32-unknown-unknown --release"
    )


class TextCore:
    """单线程使用的 wasm 核实例。跨线程请各自调用 ``get_core()``。"""

    def __init__(self, wasm_path_: str | None = None):
        path = wasm_path_ or wasm_path()
        try:
            self.store = Store()
            self.module = Module.from_file(self.store.engine, path)
            self.inst = Instance(self.store, self.module, [])
            self.ex = self.inst.exports(self.store)
            self.mem = self.ex["memory"]
        except CoreLoadError:
            raise
        except Exception as e:  # wasmtime 缺失 / 实例化失败
            raise CoreLoadError(
                f"textcore.wasm 加载失败 ({path}): {e}。"
                f"请安装运行时依赖: pip install wasmtime"
            ) from e

    # ---- 低层内存工具 ----

    def _alloc(self, nbytes: int) -> int:
        return int(self.ex["alloc"](self.store, nbytes))

    def _free(self, ptr: int, nbytes: int) -> None:
        self.ex["dealloc"](self.store, ptr, nbytes)

    def _write(self, arr: np.ndarray, ptr: int) -> None:
        self.mem.write(self.store, np.ascontiguousarray(arr).tobytes(), ptr)

    def _read(self, ptr: int, nbytes: int, dtype, shape):
        raw = bytes(self.mem.read(self.store, ptr, ptr + nbytes))
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    # ---- 算子 ----

    def distance_transform_edt(self, mask_u8, h: int, w: int) -> np.ndarray:
        """到最近文字(非零)像素的精确欧氏距离。镜像
        cv2.distanceTransform((cur == 0), DIST_L2, 3) 语义但精确。"""
        n = h * w
        p_mask = self._alloc(n)
        p_out = self._alloc(n * 4)
        try:
            self._write(mask_u8, p_mask)
            self.ex["distance_transform_edt"](self.store, p_mask, h, w, p_out)
            return self._read(p_out, n * 4, np.float32, (h, w))
        finally:
            self._free(p_mask, n)
            self._free(p_out, n * 4)

    def rgb_to_gray(self, rgb, h: int, w: int) -> np.ndarray:
        """RGB (H*W*3, 0..255 任意 dtype) -> 灰度 u8 (H*W)。"""
        n = h * w
        arr = np.ascontiguousarray(rgb, dtype=np.float32)
        p_in = self._alloc(n * 3 * 4)
        p_out = self._alloc(n)
        try:
            self._write(arr, p_in)
            self.ex["rgb_to_gray"](self.store, p_in, h, w, p_out)
            return self._read(p_out, n, np.uint8, (h, w))
        finally:
            self._free(p_in, n * 3 * 4)
            self._free(p_out, n)

    def threshold_otsu(self, u8, h: int, w: int):
        """Otsu 阈值。返回 (thr: float, bin: u8 H*W, 0/255)。"""
        n = h * w
        p_in = self._alloc(n)
        p_out = self._alloc(n)
        try:
            self._write(u8, p_in)
            thr = float(self.ex["threshold_otsu"](self.store, p_in, p_out, n))
            return thr, self._read(p_out, n, np.uint8, (h, w))
        finally:
            self._free(p_in, n)
            self._free(p_out, n)

    def morphology(self, mask_u8, h: int, w: int, kern_u8, kh: int, kw: int,
                   op: str = "dilate") -> np.ndarray:
        """显式核位图的二值形态学。kern_u8: kh*kw u8 (1=参与, 如
        cv2.getStructuringElement 输出)。返回 0/1 u8。op: "dilate"/"erode"。"""
        n = h * w
        nk = kh * kw
        op_code = 0 if op == "erode" else 1
        p_in = self._alloc(n)
        p_out = self._alloc(n)
        p_k = self._alloc(nk)
        try:
            self._write(mask_u8, p_in)
            self._write(kern_u8, p_k)
            self.ex["morphology"](self.store, p_in, p_out, h, w, p_k, kh, kw, op_code)
            return self._read(p_out, n, np.uint8, (h, w))
        finally:
            self._free(p_in, n)
            self._free(p_out, n)
            self._free(p_k, nk)

    def connected_components(self, mask_u8, h: int, w: int):
        """8 连通域。返回 (n, labels int32 H*W, stats 字典列表)，
        stats[0] 为背景占位。"""
        n = h * w
        p_in = self._alloc(n)
        p_labels = self._alloc(n * 4)
        try:
            self._write(mask_u8, p_in)
            ncomp = int(self.ex["connected_components"](self.store, p_in, p_labels, h, w))
            labels = self._read(p_labels, n * 4, np.int32, (h, w))
        finally:
            self._free(p_in, n)
        p_stats = self._alloc(max(1, ncomp) * 5 * 4)
        try:
            self.ex["connected_components_stats"](self.store, p_labels, p_stats, h, w, ncomp)
            raw = self._read(p_stats, ncomp * 5 * 4, np.int32, (ncomp, 5))
        finally:
            self._free(p_stats, max(1, ncomp) * 5 * 4)
            self._free(p_labels, n * 4)
        stats = [{"left": int(r[0]), "top": int(r[1]), "width": int(r[2]),
                  "height": int(r[3]), "area": int(r[4])} for r in raw]
        return ncomp, labels, stats

    def resize_gray_cubic(self, u8, h2: int, w2: int) -> np.ndarray:
        h, w = u8.shape[:2]
        n, n2 = h * w, h2 * w2
        p_in = self._alloc(n)
        p_out = self._alloc(n2)
        try:
            self._write(u8, p_in)
            self.ex["resize_gray_cubic"](self.store, p_in, p_out, h, w, h2, w2)
            return self._read(p_out, n2, np.uint8, (h2, w2))
        finally:
            self._free(p_in, n)
            self._free(p_out, n2)

    def resize_float_linear(self, f32, h2: int, w2: int) -> np.ndarray:
        h, w = f32.shape[:2]
        n, n2 = h * w, h2 * w2
        p_in = self._alloc(n * 4)
        p_out = self._alloc(n2 * 4)
        try:
            self._write(f32, p_in)
            self.ex["resize_float_linear"](self.store, p_in, p_out, h, w, h2, w2)
            return self._read(p_out, n2 * 4, np.float32, (h2, w2))
        finally:
            self._free(p_in, n * 4)
            self._free(p_out, n2 * 4)

    def patchmatch_inpaint(self, rgb_f32, h: int, w: int, mask_u8,
                           sample_u8=None, p: int = 7,
                           direction_deg: float = -1.0, seed: int = 0) -> np.ndarray:
        """PatchMatch 修复 —— 共享填充例程（浏览器与后端调同一份）。

        rgb_f32 : H*W*3 float32 (0..255)。
        mask_u8 : H*W, >0 = 待填充洞。
        sample_u8 : 可选 H*W, >0 = 允许取样区域。
        direction_deg : < -1e30 关闭方向模式。
        返回 H*W*3 float32 填充结果。
        """
        n = h * w
        arr = np.ascontiguousarray(rgb_f32, dtype=np.float32)
        m = np.ascontiguousarray(mask_u8, dtype=np.uint8)
        has_sample, p_s = 0, 0
        if sample_u8 is not None:
            has_sample = 1
            p_s = self._alloc(n)
            self._write(sample_u8, p_s)
        p_in = self._alloc(n * 3 * 4)
        p_out = self._alloc(n * 3 * 4)
        p_mask = self._alloc(n)
        try:
            self._write(arr, p_in)
            self._write(m, p_mask)
            self.ex["patchmatch_inpaint"](
                self.store, p_in, h, w, p_mask, p_s, has_sample, p,
                float(direction_deg), int(seed) & 0xFFFFFFFF, p_out)
            return self._read(p_out, n * 3 * 4, np.float32, (h, w, 3))
        finally:
            self._free(p_in, n * 3 * 4)
            self._free(p_out, n * 3 * 4)
            self._free(p_mask, n)
            if p_s:
                self._free(p_s, n)

    def synthesize_masks(self, text_mask_u8, h: int, w: int, edge: int = 1,
                         limit_u8=None) -> tuple:
        """文字蒙版合成 —— 共享算子（浏览器与后端调同一份）。

        text_mask_u8 : H*W, >0 = 文字。edge: >0 膨胀 / <0 腐蚀 / 0 不变
        (直径 = abs(edge)*2+1 椭圆)。limit_u8: 可选, >0 限定填充区(不影响取样)。
        返回 (fill_mask 0/255, sample_mask 0/255)，sample = 全图 − fill。
        """
        n = h * w
        tm = np.ascontiguousarray(text_mask_u8, dtype=np.uint8)
        has_limit, p_lim = 0, 0
        if limit_u8 is not None:
            has_limit = 1
            p_lim = self._alloc(n)
            self._write(limit_u8, p_lim)
        p_in = self._alloc(n)
        p_fill = self._alloc(n)
        p_smpl = self._alloc(n)
        try:
            self._write(tm, p_in)
            self.ex["synthesize_masks"](
                self.store, p_in, h, w, edge, p_lim, has_limit, p_fill, p_smpl)
            fill = self._read(p_fill, n, np.uint8, (h, w))
            sample = self._read(p_smpl, n, np.uint8, (h, w))
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
        """去发光（整片减绿度 v2）—— 共享算子（浏览器与后端调同一份）。

        rgb_f32: H*W*3 float32 (0..255); tmask_u8: H*W, >0 = 已知文字。
        返回 (clean H*W*3 u8, core_mask 0/255, zone 0/255)。
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
            self._write(arr, p_in)
            self._write(tm, p_tm)
            self.ex["deglow_full_green_v2"](
                self.store, p_in, h, w, p_tm,
                float(strength), float(zone_ratio), int(zone_expand),
                int(protect_px), int(chroma_keep), p_clean, p_core, p_zone)
            clean = self._read(p_clean, n * 3, np.uint8, (h, w, 3))
            core = self._read(p_core, n, np.uint8, (h, w))
            zone = self._read(p_zone, n, np.uint8, (h, w))
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
        """单一共享管线入口 —— 去发光 + 蒙版修复 + PatchMatch 填充一次完成
        （浏览器与后端逐字节一致）。

        rgb_f32   : H*W*3 float32 (0..255)。
        tmask_u8  : H*W, >0 = 原图检测的文字蒙版。
        tmask2_u8 : 可选 H*W 第二次检测（去发光图上, tint 开）并入并集。
        返回 (result H*W*3 u8, fill_mask H*W u8, clean H*W*3 u8, zone H*W u8)。
        """
        n = h * w
        arr = np.ascontiguousarray(rgb_f32, dtype=np.float32)
        tm = np.ascontiguousarray(tmask_u8, dtype=np.uint8)
        tm2 = (np.zeros((h, w), dtype=np.uint8) if tmask2_u8 is None
               else np.ascontiguousarray(tmask2_u8, dtype=np.uint8))
        p_in = self._alloc(n * 3 * 4)
        p_tm = self._alloc(n)
        p_tm2 = self._alloc(n)
        p_result = self._alloc(n * 3)
        p_fill = self._alloc(n)
        p_clean = self._alloc(n * 3)
        p_zone = self._alloc(n)
        try:
            self._write(arr, p_in)
            self._write(tm, p_tm)
            self._write(tm2, p_tm2)
            self.ex["erase_text_glyphs"](
                self.store, p_in, h, w, p_tm, p_tm2,
                float(strength), float(zone_ratio), int(zone_expand),
                int(protect_px), int(chroma_keep), int(edge),
                float(direction_deg), int(seed) & 0xFFFFFFFF,
                int(edge_aware), float(soft_expand),
                p_result, p_fill, p_clean, p_zone)
            result = self._read(p_result, n * 3, np.uint8, (h, w, 3))
            fill = self._read(p_fill, n, np.uint8, (h, w))
            clean = self._read(p_clean, n * 3, np.uint8, (h, w, 3))
            zone = self._read(p_zone, n, np.uint8, (h, w))
        finally:
            self._free(p_in, n * 3 * 4)
            self._free(p_tm, n)
            self._free(p_tm2, n)
            self._free(p_result, n * 3)
            self._free(p_fill, n)
            self._free(p_clean, n * 3)
            self._free(p_zone, n)
        return result, fill, clean, zone


# ---------------------------------------------------------------------------
# 线程本地实例：FastAPI 线程池并发请求 / 批处理各自持有独立 Store，
# 互不踩内存。wasm 仅 245KB，每线程一份实例开销可忽略。
# ---------------------------------------------------------------------------
_local = threading.local()


def get_core() -> TextCore:
    """返回当前线程的 TextCore 实例（惰性创建）。失败抛 CoreLoadError。"""
    core = getattr(_local, "core", None)
    if core is None:
        core = TextCore()
        _local.core = core
    return core


def reset_core() -> None:
    """丢弃当前线程的实例（测试 / 热更新用）。"""
    _local.core = None
