"""
文字擦除: DBNet → mask 2px 椭圆膨胀 → patch_fill(sample_mask=整图-mask) → 出图

0.3.0 架构（wasm 单核）:
  - 去发光/填充算法只有一份实现: textcore.wasm（与浏览器 Worker 同一份，逐字节
    一致）。原 Python 核心通道法(auto/autov1.1/deglow_first)、v4 实验方案与 numpy
    PatchMatch 已随 0.3.0 删除 —— 见 README「迁移 0.2.x → 0.3.0」。
  - 后端保留的 Python 职责: DBNet 文字检测(onnxruntime)、蒙版修复编排、取样剔除、
    透明度扩展(soft_expand)、auto_edge 判定 —— 均为编排逻辑，非核心算法。

设计原则:
- 只做 patchmatch, 不做 TELE / 颜色匹配. 后两者会破坏纹理造成"糊一团".
  (平滑渐变背景例外: 由 wasm pm_smooth_telea_full 权威判定并填充.)
- mask 2px 膨胀是必须的: 吃掉字形抗锯齿边缘, 否则 SSD 比较时这些 AA
  像素会被当成"已知上下文"污染 patchmatch 结果 (Δmean 30→0.1).
- sample_mask = 整图 - 文字mask: 强制 patchmatch 不在文字区取样.
"""
from __future__ import annotations
import os
import time
import numpy as np
# Shared-algorithm-core cv2 shim: routes dilate/erode/morphologyEx/connectedComponents/
# cvtColor(RGB2GRAY) through textcore.wasm (same operators the browser runs) and falls
# through to the real cv2 for everything else (LAB/Sobel 等检测链算子).
from text_eraser import _cv as cv2
from text_eraser import _shared_core

from text_eraser.text_select import (detect_text_mask, _fill_bright_near_mask,
                              _absorb_zone_bright_core)
from text_eraser.patch_fill import inpaint as pm_inpaint


def _ellipse(p: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p * 2 + 1, p * 2 + 1))


def _edge_aware_grow(rgb: np.ndarray, mask_filled: np.ndarray) -> np.ndarray:
    """边缘感知扩张：把紧邻 mask、且亮度落在文字带内的原图像素并入填充区。

    只解决"抗锯齿白边/浅色描边漏填"——这些像素在原图是亮于背景的半透明文字边，
    但被二值 mask 切在门外。我们只在 mask 周围一圈(ellipse(4))内、且亮度介于
    (背景中值, 文字上限) 之间的像素才并入，因此不会像大膨胀那样把背景纹理也吃进来。
    """
    if not mask_filled.any():
        return mask_filled
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lum = lab[..., 0].astype(np.float32)
    text_lum = lum[mask_filled > 0]
    lo = float(text_lum.min())
    hi = float(text_lum.max())
    bg = float(np.median(lum[mask_filled == 0]))
    band_lo = (bg + lo) / 2.0          # 文字带下限：背景与文字最低值折中
    band_hi = hi + (hi - lo) * 0.5     # 文字带上限：略放宽，兜住亮边
    cand = cv2.dilate(mask_filled, _ellipse(8))           # 候选扩张圈(大一些，吃到远处 AA 白边)
    keep = (lum >= band_lo) & (lum <= band_hi)
    grown = ((cand > 0) & keep).astype(np.uint8) * 255
    grown = cv2.erode(grown, _ellipse(1))                # 收一下孤立噪点
    grown = cv2.bitwise_or(grown, mask_filled)           # 永远保留原 mask
    return grown


def _erase_once(
    rgb: np.ndarray,
    *,
    edge: int = 1,
    q_off: float = 55.0,
    max_area_ratio: float = 0.40,
    max_box_ratio: float = 0.40,
    ml_max_side: int = 960,
    direction: float | None = None,
    edge_aware: bool = False,
    return_mask: bool = False,
    tint_fill: bool = True,
    fill_white: bool = True,
    fill_max_dist: int = 12,
    deglow_strength: float = 1.0,
    deglow_mask_soft: float = 0.0,
    deglow_zone_ratio: float = 0.6,
    deglow_zone_expand: int = 10,
    deglow_protect_px: int = 1,
    deglow_chroma_keep: bool = True,
    deglow_scheme: str = "v2",
    tmask_hint: np.ndarray | None = None,
):
    """最小化文字擦除管线（wasm 单核版）.

    Args:
        rgb: HxWx3 uint8 RGB 图像
        edge: 「移动边缘」—— 蒙版(展示)与**填充区域**同步外扩/收缩
            (正=扩, 0=仅取 Otsu 字形不扩不缩, 负=收缩选区)。默认 1: 在 Otsu 字形
            基础上椭圆膨胀 1px, 刚好吸收字形 AA 抗锯齿边缘。
        q_off: 传给 detect_text_mask, [30,70], 越高 mask 越贴字形
        max_area_ratio: 传给 detect_text_mask, 给单字粘连大块放行
        ml_max_side: DBNet 推理尺度
        edge_aware: 默认 False. 历史 ellipse(8) 方案会把 mask 膨胀到占图 60%+;
            现默认关掉, edge=1 已足够。
        return_mask: True 时返回 (result, mask, meta), 否则 (result, meta)
        tint_fill: True 时启用色偏区域生长(_grow_color_tint, 红蒙版叠加/淡绿光晕
            自动并入蒙版)。False 则不做色偏生长。
        fill_white: True 时启用「临近纯白补全」，把紧邻蒙版的亮白/抗锯齿像素并入
            蒙版，修复描边字漏白。
        fill_max_dist: fill_nearby_white 「孤立纯白段」步骤的最大吞并距离(px)。0=关闭。
        deglow_scheme: 去发光方案
            "v2"   (默认) — 减绿度去发光(wasm) → 去发光图上再检测 → wasm 填充。
            "off"       — 关闭所有去发光。
        deglow_*: v2 减绿度参数（语义见 erase_text）。
        tmask_hint: 调用方已算好的原图文字蒙版（如 auto_edge 判定时的检测结果），
            传入可跳过重复 DBNet 推理（并发/性能优化）。
    Returns:
        result: HxWx3 uint8 RGB 已擦除文字的图片
        mask:   HxW uint8 (255=将被填充的区域[移动边缘 edge 后])  仅当 return_mask=True
        meta:   dict with keys {mask_pix, inpaint_seconds, method}
    """
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    t0 = time.time()

    if deglow_scheme == "v2":
        # v2 减绿度允许略过冲(上限1.5)以彻底去净淡绿残迹
        return _erase_deglow_v2(
            rgb, edge=edge, q_off=q_off,
            max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
            ml_max_side=ml_max_side, direction=direction,
            edge_aware=edge_aware,
            return_mask=return_mask, fill_white=fill_white,
            fill_max_dist=fill_max_dist,
            deglow_strength=max(float(deglow_strength), 1.15),
            deglow_zone_ratio=deglow_zone_ratio,
            deglow_zone_expand=deglow_zone_expand,
            deglow_protect_px=deglow_protect_px,
            deglow_chroma_keep=deglow_chroma_keep,
            soft_expand=float(max(0.0, min(deglow_mask_soft, 150.0))),
            tmask_hint=tmask_hint,
        )
    if deglow_scheme != "off":
        raise ValueError(
            f"deglow_scheme={deglow_scheme!r} 不支持；0.3.0 仅支持 'v2'(默认) 与 'off'")

    # ---- scheme="off": 不去发光，检测 → 填充 ----
    mask, boxes = detect_text_mask(
        rgb, method="ml", q_off=q_off,
        max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
        max_side=ml_max_side, tint_fill=False, fill_white=fill_white,
        fill_max_dist=fill_max_dist,
    ) if tmask_hint is None else (tmask_hint, _mask_to_boxes(tmask_hint))

    if not mask.any():
        out = (rgb, mask, {"mask_pix": 0, "inpaint_seconds": 0.0,
                           "method": "ml", "boxes": []}) if return_mask \
              else (rgb, {"mask_pix": 0, "inpaint_seconds": 0.0,
                          "method": "ml", "boxes": []})
        return out

    return _run_fill(rgb, mask, boxes, edge=edge, direction=direction,
                     edge_aware=edge_aware,
                     return_mask=return_mask, t0=t0, sample_exclude=None,
                     soft_expand=float(max(0.0, min(deglow_mask_soft, 150.0))))


def _mask_to_boxes(mask: np.ndarray) -> list:
    """从蒙版粗略提取外接框列表（tmask_hint 复用路径用；框仅作 meta 展示）。"""
    from text_eraser.text_select import _mask_to_boxes as _m2b
    return _m2b(mask)


def erase_text(
    rgb: np.ndarray,
    *,
    edge: int = 1,
    auto_edge: bool = True,
    auto_max_edge: int = 2,
    q_off: float = 55.0,
    max_area_ratio: float = 0.40,
    max_box_ratio: float = 0.40,
    ml_max_side: int = 960,
    direction: float | None = None,
    edge_aware: bool = False,
    return_mask: bool = False,
    tint_fill: bool = True,
    fill_white: bool = True,
    fill_max_dist: int = 12,
    deglow_strength: float = 1.0,
    deglow_green_thr: float = 6.0,
    deglow_range: int = 24,
    deglow_glo: float = 85.0,
    deglow_protect: float = 1.0,
    deglow_mask_soft: float = 0.0,
    deglow_zone_ratio: float = 0.6,
    deglow_zone_expand: int = 10,
    deglow_protect_px: int = 1,
    deglow_chroma_keep: bool = True,
    deglow_scheme: str = "v2",
):
    """文字擦除入口（后端引擎，wasm 单核）。

    0.3.0 变更: 移除 ``glow_mode``（通道法及全部 Python 去发光变体已删除）；
    ``deglow_scheme`` 仅支持 "v2"(默认) / "off"。算法执行位置：
      - 后端引擎 = 本函数（Python 进程内经 wasmtime 调 textcore.wasm）；
      - 浏览器引擎 = `text-eraser-browser` ESM 包（browser/src/index.js 的
        ``erase()`` / ``eraseTextGlyphs()``），两端共用同一份 wasm，逐字节一致；
        自定义管线可用 ``text_eraser.core``（后端）或
        ``import { ... } from 'text-eraser-browser'``（浏览器）自由编排。

    auto_edge=True 时，先按原图文字蒙版外围的「文字色残留」逐环判定所需的最小
    移动边缘 edge（默认从 ``edge`` 起，至多 ``auto_max_edge``），再走普通管线。
    判定依据：若蒙版外第 (e+1) 环仍含显著文字色/抗锯齿边（肉眼会看成鬼影），
    则 e+1；否则保持。这样「大多数图默认 1、少数硬图自动到 2」，且纹理损伤最小。
    返回的 meta 含 ``auto_edge`` / ``edge_used`` 便于前端标注实际使用的 edge。
    """
    if auto_edge:
        return _erase_auto(
            rgb, edge=edge, auto_max_edge=auto_max_edge,
            q_off=q_off, max_area_ratio=max_area_ratio,
            max_box_ratio=max_box_ratio, ml_max_side=ml_max_side,
            direction=direction, edge_aware=edge_aware,
            return_mask=return_mask, tint_fill=tint_fill,
            fill_white=fill_white, fill_max_dist=fill_max_dist,
            deglow_strength=deglow_strength,
            deglow_mask_soft=deglow_mask_soft,
            deglow_zone_ratio=deglow_zone_ratio,
            deglow_zone_expand=deglow_zone_expand,
            deglow_protect_px=deglow_protect_px,
            deglow_chroma_keep=deglow_chroma_keep,
            deglow_scheme=deglow_scheme)

    # 兼容参数: deglow_green_thr / deglow_range / deglow_glo / deglow_protect 是
    # 0.2.x 通道法旋钮, 0.3.0 起通道法已删除、v2 不使用 —— 仅为 API 兼容保留在
    # 签名中, 传入会被忽略(不报错, 便于旧调用方平滑升级)。
    return _erase_once(
        rgb, edge=edge, q_off=q_off, max_area_ratio=max_area_ratio,
        max_box_ratio=max_box_ratio, ml_max_side=ml_max_side,
        direction=direction, edge_aware=edge_aware,
        return_mask=return_mask, tint_fill=tint_fill,
        fill_white=fill_white, fill_max_dist=fill_max_dist,
        deglow_strength=deglow_strength,
        deglow_mask_soft=deglow_mask_soft,
        deglow_zone_ratio=deglow_zone_ratio,
        deglow_zone_expand=deglow_zone_expand,
        deglow_protect_px=deglow_protect_px,
        deglow_chroma_keep=deglow_chroma_keep,
        deglow_scheme=deglow_scheme)


def _erase_auto(rgb, *, edge, auto_max_edge, return_mask, **kw):
    """auto_edge 内部：先判定实际 edge，再走 _erase_once，并在 meta 标注。

    并发优化: 判定所需的原图检测蒙版与 v2 管线第一步完全同参，这里检测一次后
    经 ``tmask_hint`` 传下去，省掉一次 DBNet 推理（大图 detect≈1.8s/次）。
    """
    det_kw = dict(
        method="ml", q_off=kw["q_off"],
        max_area_ratio=kw["max_area_ratio"], max_box_ratio=kw["max_box_ratio"],
        max_side=kw["ml_max_side"], tint_fill=False,
        fill_white=kw["fill_white"], fill_max_dist=kw["fill_max_dist"])
    tmask, _ = detect_text_mask(rgb, **det_kw)
    chosen = edge
    if tmask.any():
        chosen = _decide_edge(rgb, tmask, preferred=edge, max_edge=auto_max_edge)
    if return_mask:
        result, mask_filled, meta = _erase_once(rgb, edge=chosen, return_mask=True,
                                                tmask_hint=tmask, **kw)
    else:
        result, meta = _erase_once(rgb, edge=chosen, return_mask=False,
                                   tmask_hint=tmask, **kw)
    meta["auto_edge"] = True
    meta["edge_used"] = chosen
    if return_mask:
        return result, mask_filled, meta
    return result, meta


def _decide_edge(rgb, mask, *, preferred: int = 1, max_edge: int = 2) -> int:
    """从 ``preferred`` 起，若蒙版外第 (e+1) 环仍含显著文字色/抗锯齿边则 e+1，
    至多 ``max_edge``。返回最终选用的最小 edge。
    """
    if not mask.any():
        return preferred
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    text_lab = lab[mask > 0].mean(0)
    far = cv2.dilate(mask, _ellipse(16)) == 0
    bg_lab = lab[far].mean(0) if far.any() else lab.reshape(-1, 3).mean(0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    e = int(preferred)
    while e < int(max_edge):
        if _ring_dirty(lab, text_lab, bg_lab, grad, mask, radius=e + 1):
            e += 1
        else:
            break
    return e


def _ring_dirty(lab, text_lab, bg_lab, grad, mask, radius: int) -> bool:
    """蒙版外第 ``radius`` 环是否仍含显著文字色/抗锯齿边（肉眼会看成鬼影）。

    文字色残留 = 该环像素比更接近文字色(而非背景色)且明显偏离背景；
    配合高梯度(真实笔画边)作为强佐证，二者任一显著即判为「脏」。
    阈值经 79188(需2, ring2 混色边366/梯度174) 与 6464/6251689(默认1好,
    ring2 混色边≤13/梯度≤42) 标定：混色边>50 或 梯度>140。
    """
    if radius < 1:
        return False
    cur = cv2.dilate(mask, _ellipse(radius)) > 0
    prev = cv2.dilate(mask, _ellipse(radius - 1)) > 0
    ring = cur & ~prev
    n = int(ring.sum())
    if n < 30:
        return False
    sub = lab[ring]
    d_text = np.sqrt(((sub - text_lab) ** 2).sum(1))
    d_bg = np.sqrt(((sub - bg_lab) ** 2).sum(1))
    blend = int(((d_text <= d_bg) & (d_bg > 18)).sum())
    gmean = float(grad[ring].mean())
    return blend > 50 or gmean > 140


def _residual_green(rgb: np.ndarray, mask: np.ndarray,
                    radius: int = 48, thr: int = 8,
                    g_lo: int = 90) -> np.ndarray:
    """找到紧邻蒙版的残余绿色像素(供填充取样排除)。

    用于发光处理：去发光后文字边缘可能仍残留未净的绿色像素，若被填充
    PatchMatch 当作取样源复制进文字区，就会造成"泛绿/尿渍"。这里把它们
    圈出来(蒙版膨胀 radius 圈内、绿通道主导)，让 _run_fill 从取样区剔除。
    无发光的普通图绿色像素≈0 → 返回全 False，不影响取样。
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    green = (g - np.maximum(r, b) > thr) & (g > g_lo)
    if not green.any():
        return np.zeros(rgb.shape[:2], bool)
    near = cv2.dilate(mask, _ellipse(radius)) > 0
    return green & near


def _dark_source_exclude(clean: np.ndarray, mask: np.ndarray,
                         ring_px: int = 4, band: int = 25):
    """发光图填充的「暗源剔除」：把比蒙版紧邻上下文暗带下限还暗 band 的像素
    从取样区剔除，防止 patchmatch 把远处深色斑纹拉进文字区填成黑块。

    背景(556 实测)：去发光后笔画上下文 lum≈74~85，但大块字洞内部锚定弱，
    patchmatch 从下半部枯枝暗纹(lum 35~55)拉黑斑填进笔画区 → 文字区黑块。
    参考值取蒙版外 ring_px 环带亮度的 **25 分位**(暗侧)再减 band —— 自校准：
    亮结构(米黄带)环绕时不会被误剔，深色背景图的合法暗源也不受影响。
    只在确实发生过去发光的 v2 路径调用；普通路径不受影响。

    Returns: bool mask 或 None(无需剔除)。
    """
    L = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ring = (cv2.dilate(mask, _ellipse(ring_px)) > 0) & (mask == 0)
    if not ring.any():
        return None
    ref = float(np.percentile(L[ring], 25)) - band
    out = L < ref
    return out if out.any() else None


def _run_fill(rgb, mask, boxes, *, edge, direction, edge_aware,
              return_mask, t0, sample_exclude=None,
              soft_expand: float = 0.0):
    """共用填充步骤：膨胀 mask → sample → patch_fill(wasm) → meta。

    sample_exclude: 可选 bool mask(HxW)，这些像素从取样区剔除(不参与复制)。
        用于发光处理时剔除残余绿色像素，避免填充把绿复制进文字区(泛绿/尿渍)。
    soft_expand: 「透明度扩展」半径(px, 0=关)。>0 时在真实填充区(mask_filled)
        外围扩展一圈软带：软带内用原始图(已去发光)与填充结果按距离衰减渐变
        混合 —— 内缘贴近核心(完全填充)、外缘回归原始 → 扩大覆盖范围的同时
        保留底层纹理不被整块覆盖。软带会被红蒙版以半透明显示(透明度扩展)。
    """
    # 1. 移动边缘(edge): 椭圆膨胀(>0)/腐蚀(<0)蒙版, 吸收字形 AA 抗锯齿边缘。
    #    edge=1(默认)=膨胀 1px 吃掉 AA; edge=0=仅取 Otsu 字形; edge<0=收缩选区。
    mask_pre_edge = mask.copy()          # 移动边缘前的文字蒙版(前端分步展示用)
    if edge > 0:
        mask_filled = cv2.dilate(mask, _ellipse(edge))
    elif edge < 0:
        mask_filled = cv2.erode(mask, _ellipse(-edge))
    else:
        mask_filled = mask.copy()

    # 1b. 边缘感知扩张: 把"紧邻 mask 且亮度落在文字带"的原图像素也并入填充区,
    #     专门吃掉抗锯齿白边(白字) / 浅色描边, 又不会像大膨胀那样产生 halo。
    if edge_aware:
        mask_filled = _edge_aware_grow(rgb, mask_filled)

    # 2. sample_mask: 只在文字外取样
    sample_mask = (255 - mask_filled).astype(np.uint8)
    if sample_exclude is not None:
        sample_mask[sample_exclude] = 0

    # 3. patch_fill (wasm 共享核; direction 非空时启用方向填充)
    result = pm_inpaint(rgb, mask_filled, sample_mask=sample_mask, direction=direction)

    # 3b. 透明度扩展: 在填充区外围做一层渐变混合带(扩覆盖、保留纹理)
    soft_alpha = None
    if soft_expand > 0:
        s = int(round(min(soft_expand, 150.0)))
        core = mask_filled > 0
        band = (cv2.dilate(mask_filled, _ellipse(s)) > 0) & ~core
        if band.any():
            union = core.astype(np.uint8) * 255 + 0  # 255=核心
            union[band] = 255
            union_u8 = union.astype(np.uint8)
            sample_u8 = (255 - union_u8).astype(np.uint8)
            if sample_exclude is not None:
                sample_u8[sample_exclude] = 0
            filled_all = pm_inpaint(rgb, union_u8, sample_mask=sample_u8,
                                    direction=direction)
            # 软带每个像素到核心的距离 → 透明度从内缘 1 衰减到外缘 0
            dst = cv2.distanceTransform((255 - mask_filled).astype(np.uint8),
                                        cv2.DIST_L2, 5)
            a = np.clip(1.0 - dst[band] / float(s), 0.0, 1.0)
            idx = np.flatnonzero(band)
            rr = result.astype(np.float32)
            aa = filled_all.astype(np.float32)
            cc = rgb.astype(np.float32)
            rr.reshape(-1, 3)[idx] = (cc.reshape(-1, 3)[idx] * (1 - a)[:, None]
                                      + aa.reshape(-1, 3)[idx] * a[:, None])
            result = rr.clip(0, 255).astype(np.uint8)
            soft_alpha = np.zeros(rgb.shape[:2], np.float32)
            soft_alpha[band] = a

    elapsed = time.time() - t0
    meta = {
        "mask_pix": int(mask.sum() // 255),
        "mask_filled_pix": int(mask_filled.sum() // 255),
        "mask_soft_pix": int(np.count_nonzero(soft_alpha)) if soft_alpha is not None else 0,
        "inpaint_seconds": round(elapsed, 3),
        "method": "ml",
        "boxes": boxes,
    }
    if soft_alpha is not None and (soft_alpha > 0).any():
        meta["soft_alpha"] = soft_alpha   # 供红蒙版半透明显示(不落历史)
    meta["mask_pre_edge"] = mask_pre_edge  # 移动边缘前的文字蒙版(前端分步展示)
    if return_mask:
        # 展示蒙版 = 真实填充区(移动边缘 edge 后) —— 所见即所得
        return result, mask_filled, meta
    return result, meta


def erase_batch(images, *, workers: int | None = None, return_mask: bool = False,
                **kw):
    """批量并发擦除（多图并行；线程间经线程本地 wasm 核实例隔离）。

    images    : 可迭代的 HxWx3 uint8 RGB 数组列表。
    workers   : 并发线程数，默认 min(图数, CPU 核数)。
    return_mask / **kw : 透传给 erase_text（每张图同一组参数）。

    返回与输入同序的结果列表（erase_text 的返回元组）。

    并发可行性依据：wasmtime 的 FFI 调用会释放 GIL（实测 2 线程双图填充
    1.7x 加速），每个线程持有独立 TextCore 实例（wasmtime Store 非线程安全，
    单例会在并发 alloc/dealloc 时踩内存）；DBNet 的 onnxruntime
    InferenceSession.run 官方保证线程安全。单张图内部不做并行 —— 逐框拆分
    会改变填充结果、破坏前后端逐字节一致，属有意为之的设计。
    """
    from concurrent.futures import ThreadPoolExecutor
    imgs = list(images)
    if not imgs:
        return []
    n = workers or min(len(imgs), max(1, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(erase_text, img, return_mask=return_mask, **kw)
                for img in imgs]
        return [f.result() for f in futs]


def _erase_deglow_v2(rgb, *, edge, q_off, max_area_ratio, max_box_ratio,
                     ml_max_side, direction, edge_aware,
                     return_mask, fill_white: bool = True,
                     fill_max_dist: int = 12,
                     deglow_strength: float = 1.0,
                     deglow_zone_ratio: float = 0.6,
                     deglow_zone_expand: int = 10,
                     deglow_protect_px: int = 1,
                     deglow_chroma_keep: bool = True,
                     soft_expand: float = 0.0,
                     tmask_hint: np.ndarray | None = None):
    """v2 入口（唯一去发光算法，wasm 单核）：先减绿度去发光 → 再对「去完发光的图」
    走普通去字算法。

    算法分解:
      - 去发光用「减绿度」(只动 G 通道、绿晕→中性灰、永不变黑、底层纹理保留)，
        由共享核 `deglow_full_green_v2` 完成（与浏览器同一份 wasm）；
      - 去字阶段完全复用「非高亮」算法：在去完发光的 clean 图上
        detect_text_mask(tint_fill=True, fill_white=...) → 并集 → 闭运算 →
        亮核吸收 → patch_fill，不用自定义 core_mask 走捷径。发光图与普图的
        去字质量因此保持一致。
    展示上「去发光」与「去文字」分两步：meta["deglow_img"] = clean(去发光中间图)，
    而 result = 在 clean 上去字后的最终结果；前端可分别呈现这两张图。
    """
    t0 = time.time()
    # 1) 定位文字(保护 + 生长种子；不做色偏生长, 避免把光晕并入蒙版)
    if tmask_hint is not None:
        tmask, _boxes = tmask_hint, []
    else:
        tmask, _boxes = detect_text_mask(
            rgb, method="ml", q_off=q_off,
            max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
            max_side=ml_max_side, tint_fill=False, fill_white=fill_white,
            fill_max_dist=fill_max_dist,
        )
    if not tmask.any():
        meta = {"mask_pix": 0, "mask_filled_pix": 0, "inpaint_seconds": 0.0,
                "method": "ml", "boxes": []}
        return (rgb, tmask, meta) if return_mask else (rgb, meta)

    # 2) 去发光: 取 clean(供步骤3「去发光图上再检测」); 同时取 zone(亮核吸收用)。
    #    只走 wasm(与浏览器同一份 Rust 实现, 两端逐字节一致); Python 实现已删除。
    clean0, _core_unused, zone0 = _shared_core.deglow_full_green_v2(
        rgb, tmask, strength=deglow_strength,
        zone_ratio=deglow_zone_ratio, zone_expand=deglow_zone_expand,
        protect_px=deglow_protect_px, chroma_keep=1 if deglow_chroma_keep else 0)
    zone0 = (zone0 > 0).astype(np.uint8) * 255

    # 3) 普通去文字蒙版 = 原图检测 ∪ 去发光图检测(tint=True) → 闭运算补断裂
    tm_clean, boxes = detect_text_mask(
        clean0, method="ml", q_off=q_off,
        max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
        max_side=ml_max_side, tint_fill=True, fill_white=fill_white,
        fill_max_dist=fill_max_dist,
    )
    mask = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if not mask.any():
        meta = {"mask_pix": 0, "mask_filled_pix": 0, "inpaint_seconds": 0.0,
                "method": "ml", "boxes": [], "deglow_img": clean0}
        return (clean0, np.zeros(rgb.shape[:2], np.uint8), meta) if return_mask \
            else (clean0, meta)

    # 4) 蒙版修复(zone 亮核吸收) + 取样剔除 + 填充(wasm patchmatch / wasm TELEA)
    mask = _fill_bright_near_mask(clean0, mask)
    mask = _absorb_zone_bright_core(clean0, rgb, mask, zone0, min_rgb_lo=100)
    if not mask.any():
        meta = {"mask_pix": 0, "mask_filled_pix": 0, "inpaint_seconds": 0.0,
                "method": "ml", "boxes": [], "deglow_img": clean0}
        return (clean0, np.zeros(rgb.shape[:2], np.uint8), meta) if return_mask \
            else (clean0, meta)
    sample_exclude = _residual_green(clean0, mask)
    if zone0 is not None and bool((zone0 > 0).any()):
        _dx = _dark_source_exclude(clean0, mask)
        if _dx is not None:
            sample_exclude = (_dx | sample_exclude) if sample_exclude is not None \
                else _dx
    res = _run_fill(clean0, mask, boxes, edge=edge, direction=direction,
                    edge_aware=edge_aware,
                    return_mask=return_mask, t0=t0,
                    sample_exclude=sample_exclude, soft_expand=soft_expand)
    if return_mask:
        result, mask_filled, meta = res
    else:
        result, meta = res
    meta["deglow_img"] = clean0
    meta["glow_zone"] = zone0
    return (result, mask_filled, meta) if return_mask else (result, meta)
