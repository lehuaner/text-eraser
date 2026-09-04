"""内容识别填充（Content-Aware Fill）—— PatchMatch 范例式修复。

0.3.0 起填充算法只有**一份实现**：`shared/src/patchmatch.rs` 编译的
textcore.wasm（与浏览器 Worker 调同一份，逐字节一致）。原 numpy 参考实现
(`_patch_fill_loop` / `_Mulberry32`) 已删除 —— 见
`shared/_verify_align/pm_crossend.py`（跨端逐字节校验 harness）保留算法基准。

本模块保留的职责（编排，非核心算法）:
  - sample_mask 规范化（HxW 数组 / 前端画笔笔画栅格化）；
  - 平滑渐变背景预检（cv2 Sobel/环带统计，判定与浏览器 JS 预检同构；
    最终判定仍由 wasm `pm_smooth_telea_full` 权威给出）;
  - ROI 裁剪 / 安全内边距 / 边距收缩保护。

签名保持 inpaint(rgb, mask) 与旧接口一致。
"""
from __future__ import annotations

import numpy as np
# Shared-algorithm-core cv2 shim: routes dilate/erode/morphologyEx/connectedComponents/
# cvtColor(RGB2GRAY) through textcore.wasm (same operators the browser runs) and falls
# through to the real cv2 for everything else (Sobel/line/circle 等检测链算子).
from text_eraser import _cv as cv2
from text_eraser._shared_core import patchmatch_inpaint_fill, smooth_telea_full


def _normalize_sample_mask(sample_mask, H, W):
    """把 sample_mask 统一成 HxW bool 数组；不支持的格式返回 None。

    支持两种传入：
      ① HxW 数组(>0 为取样区) —— 直接复用；
      ② 前端画笔笔画结构 {"brush":半径(px),"strokes":[[[x,y],...],...]}
         （图像坐标），自动栅格化（粗线 + 端点圆，保证稀疏点也覆盖）。
    """
    if sample_mask is None:
        return None
    # 情形①：已是 2D 数组且与图像同形
    try:
        arr = np.asarray(sample_mask)
    except Exception:
        arr = None
    if arr is not None and arr.ndim == 2 and arr.shape == (H, W):
        return (arr > 0)
    # 情形②：前端笔画结构 / 直接 strokes 列表
    strokes = None
    brush = 20
    if isinstance(sample_mask, dict):
        strokes = sample_mask.get("strokes")
        brush = int(sample_mask.get("brush") or 20)
    elif isinstance(sample_mask, (list, tuple)) and len(sample_mask) \
            and isinstance(sample_mask[0], (list, tuple)):
        strokes = sample_mask
    if not strokes:
        return None
    canvas = np.zeros((H, W), np.uint8)
    r = max(1, int(round(brush)))
    for st in strokes:
        pts = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in st if len(p) >= 2]
        # 限制在图像范围内
        pts = [[min(max(px, 0), W - 1), min(max(py, 0), H - 1)] for px, py in pts]
        if len(pts) >= 2:
            for k in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[k]), tuple(pts[k + 1]),
                         255, thickness=r * 2, lineType=cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, tuple(pts[0]), r, 255, -1)
    return (canvas > 0)


def inpaint(image_rgb, mask, sample_mask=None, should_cancel=None, direction=None,
            flat_span: int = 40, flat_tex: float = 20.0):
    """
    内容识别填充（PatchMatch，wasm 共享核实现 —— 与浏览器逐字节一致）。

    image_rgb    : HxWx3 uint8 / float 图像
    mask         : HxW uint8/bool，>0 为待填充(去文字)区域
    sample_mask  : 可选 HxW uint8/bool，>0 为「参考/取样区域」(PS 自定义取样区)。
                   提供时**只从该区域内**取样源块；为 None 时自动取洞周围局部纹理。
                   sample_mask **不会扩大** ROI；只用来在 ROI 内过滤候选源像素。
    should_cancel: 兼容参数（0.2.x numpy 实现支持中断）。wasm 填充不可中断，
                   0.3.0 起为 no-op，仅为 API 兼容保留。
    direction    : 可选 float(角度°, 图像坐标 0°=+x右 / 90°=+y下)。
                   提供时进入**方向填充模式**：源候选被限制在过目标点、沿 direction
                   的直线上双向采样 —— 适合有主导纹理方向的图像(木纹/岩石条带)。
    flat_span / flat_tex : 平滑渐变背景自适应门。洞外环带的梯度**中位数与 p75
                   同时** <flat_tex 时，判定背景为「无纹理可复制」→ 扩散插值
                   (TELEA)填充（wasm `pm_smooth_telea_full`，判定+填充与浏览器
                   逐字节一致）。p75 双门控修复细纹理背景被中位数误判的涂抹感。
                   纹理背景或均匀背景不受影响，仍走 PatchMatch。
    return       : HxWx3 uint8
    raises       : text_eraser._textcore.CoreLoadError（wasm 核不可用时快速失败）
    """
    img = np.ascontiguousarray(image_rgb[..., :3], dtype=np.float32)
    OH, OW = img.shape[:2]
    m = (np.asarray(mask) > 0)
    if not m.any():
        return img.astype(np.uint8).copy()

    # ---- 平滑渐变背景预检（编排逻辑；权威判定在 wasm pm_smooth_telea_full）----
    # 杂色全面修复(1788077005814): 环带纹理低(tex<flat_tex)即视为"可扩散"背景,
    # TELEA 把局部梯度平滑插值进洞, 保留渐变(纹理)同时消除杂色。纹理背景
    # (tex>=flat_tex)仍走 patchmatch。span 检查已移除——对真正光滑背景(span≈0)
    # 同样适用; 保留 n>=2 边带检查(避免极小 mask 边带不足时误触发)。
    gray0 = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                         cv2.COLOR_RGB2GRAY).astype(np.float32)
    ys0, xs0 = np.where(m)
    y0_, y1_, x0_, x1_ = ys0.min(), ys0.max(), xs0.min(), xs0.max()
    band = 12
    edges_med = []
    for sl in (np.s_[max(0, y0_-band):y0_+1, x0_:x1_+1],
               np.s_[y1_:min(OH, y1_+band+1), x0_:x1_+1],
               np.s_[y0_:y1_+1, max(0, x0_-band):x0_+1],
               np.s_[y0_:y1_+1, x1_:min(OW, x1_+band+1)]):
        vals = gray0[sl][~m[sl]]
        if vals.size:
            edges_med.append(float(np.median(vals)))
    if len(edges_med) >= 2 and direction is None:
        gx0 = cv2.Sobel(gray0, cv2.CV_32F, 1, 0, ksize=3)
        gy0 = cv2.Sobel(gray0, cv2.CV_32F, 0, 1, ksize=3)
        grad0 = np.sqrt(gx0 ** 2 + gy0 ** 2)
        ring0 = (cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~m
        # 2026-09-04 涂抹感修复: 中位数 + p75 双门控(与 wasm 权威判定一致)。
        # 只看中位数时, 纹理像素占比不足一半的细纹理背景(实测 p75=31~42)被误判
        # 平滑 → TELEA 抹平纹理; p75 同低于阈值才真平滑(实测平滑背景 p75<=16)。
        if ring0.any():
            rv = grad0[ring0]
            tex = float(np.median(rv))
            tex_p75 = float(np.percentile(rv, 75))
        else:
            tex = tex_p75 = 0.0
        if tex < flat_tex and tex_p75 < flat_tex:
            # wasm 权威判定: 触发→返回 TELEA 填充; 未触发(阈值 ULP 级跨线)→
            # 与浏览器同步继续走 patchmatch, 不允许任何 Python 侧降级。
            _t = smooth_telea_full(image_rgb, m, flat_tex)
            if _t is not None:
                return _t

    # 安全内边距：文字贴图像边缘时，ROI 内的 PxP 块切片会越界(历史崩溃点)。
    # 把整图四周复制扩展 padm 像素，ROI 计算用 padding 后尺寸(坐标相对不变)，
    # 最终输出时裁剪回原尺寸。handbrush sample_mask 仍按原始尺寸栅格化。
    padm = 4
    img = cv2.copyMakeBorder(img, padm, padm, padm, padm, cv2.BORDER_REPLICATE)
    m = np.pad(m, padm, constant_values=False)
    H, W = img.shape[:2]

    # 手工参考区域：限定从哪些已知像素取样(PS 自定义取样区)。
    sm = _normalize_sample_mask(sample_mask, OH, OW)
    if sm is not None:
        sm = np.pad(sm, padm, constant_values=False)

    # ---- 局部 ROI：只在「洞 + 边距」 范围内搜索源块 ----
    # 注意：sample_mask 不再扩大 ROI（修复前会扩展成整图→超 MAX_ROI→TELEA
    # "刷子感"）。sample_mask 只在 ROI 内过滤候选（自动排除其他文字等不要的区域）。
    ys, xs = np.where(m)
    hy0, hy1 = int(ys.min()), int(ys.max()) + 1
    hx0, hx1 = int(xs.min()), int(xs.max()) + 1
    margin = max(32, int(0.6 * max(hy1 - hy0, hx1 - hx0)))
    if sm is not None:
        # 精确文字模式(sample_mask 常为整图-字形)：希望参考圈多覆盖一圈周围布料，
        # 让 patchmatch 拿到更连续的纹理；仍受下面的 MAX_ROI 上限保护。
        margin = max(margin, int(0.9 * max(hy1 - hy0, hx1 - hx0)), 80)
    y0 = max(0, hy0 - margin); y1 = min(H, hy1 + margin)
    x0 = max(0, hx0 - margin); x1 = min(W, hx1 + margin)

    # ROI 上限保护：超限逐步缩小边距重算——大文字仍保留 PatchMatch 的纹理
    # 连续性，只是参考圈略小（不回退 TELEA，避免"刷子感"）。
    MAX_ROI = 1536
    while max(y1 - y0, x1 - x0) > MAX_ROI and margin > 24:
        margin = int(margin * 0.85)
        y0 = max(0, hy0 - margin); y1 = min(H, hy1 + margin)
        x0 = max(0, hx0 - margin); x1 = min(W, hx1 + margin)

    sub = img[y0:y1, x0:x1].copy()
    subm = m[y0:y1, x0:x1].copy()
    subsm = sm[y0:y1, x0:x1] if sm is not None else None

    # ---- shared-core fill (single source of truth, identical to the browser) ----
    _deg = direction if direction is not None else -1.0
    _filled = patchmatch_inpaint_fill(sub, subm, subsm, 7, _deg, 0)
    img[y0:y1, x0:x1] = np.clip(_filled, 0, 255)
    return np.clip(img, 0, 255)[padm:padm + OH, padm:padm + OW].astype(np.uint8)
