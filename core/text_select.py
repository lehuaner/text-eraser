"""
文字模式：自动检测图中文字区域，作为「需要填充(去文字)」的框。

设计要点（针对初版在大图(4096x2160 截图)上的"106 个长横条"事故的修复）：
  - 尺度归一：先下采样到工作最长边 <= 1280，让所有阈值与原图分辨率无关；
    再把框缩放回原图坐标。这样无论原图多大，同样的 magic number 都能稳定工作。
  - 强连通域过滤：UI 分隔线/按钮边框/扫描线伪影的典型特征是"又长又细"
    (aspect > 25)；自适应阈值把它们误当文字的前身就是它们。先在连通域层挡掉。
  - 行级二次过滤：合并后再用「行高/行宽比」「行高是否合理」「行内有效密度」
    过滤残余 UI 长条。
  - 笔画宽度一致性：距离变换 + 中位笔画宽度。文字的笔画宽度在数像素量级且
    较一致；UI 边框/纹理要么笔画宽度过大(>10px)、要么极不均匀。

零新依赖：仅 numpy + cv2(项目已有)。离线可用，无需模型下载。
如需更强的中文/艺术字识别，可在此函数内替换为 DBNet/EAST 等模型推理，接口保持不变。
"""
from __future__ import annotations

import numpy as np
import cv2

from PIL import Image


def to_rgb_uint8(raw):
    """把 PIL/numpy 输入统一成 HxWx3 uint8 RGB numpy（不依赖 extractor.py）。"""
    if isinstance(raw, Image.Image):
        return np.asarray(raw.convert("RGB"), dtype=np.uint8)
    return np.asarray(raw[..., :3], dtype=np.uint8)

# 各参数默认（用户可覆盖）；灵敏度 strength∈[0,1] 越高越灵敏（也更易误检）。
DEFAULTS = {
    "strength": 1.0,          # 检测灵敏度 [0,1]：越高→自适应阈值更低/边缘门限更低(更灵敏)
    "min_area": 30,           # 工作尺度下连通域最小面积(px)
    "max_area_ratio": 0.05,   # 工作尺度下单块最大占比（>此多半是面板/背景/大色块，丢弃）
    "max_box_ratio": 0.40,    # 最终框最大占比：超过视为误检(如整片衣物纹理)直接丢弃；0.40 兼容小图大字
    "vthr": 8,                # 同属一行的垂直容差(px, 工作尺度下)
    "pad": 3,                 # 框外扩(px, 工作尺度下)，原图尺度
    "work_max": 1280,         # 工作尺度最长边(像素)，保证尺度无关
}

# ---------------------------------------------------------------------------
# 文字检测核心 (经典 CV)
# ---------------------------------------------------------------------------
def detect_text(raw: np.ndarray | "Image.Image",
                strength: float = DEFAULTS["strength"],
                min_area: int = DEFAULTS["min_area"],
                max_area_ratio: float = DEFAULTS["max_area_ratio"],
                max_box_ratio: float = DEFAULTS["max_box_ratio"],
                vthr: int = DEFAULTS["vthr"],
                pad: int = DEFAULTS["pad"],
                work_max: int = DEFAULTS["work_max"],
                method: str = "classic",          # classic | ml
                box_threshold: float = 0.3,      # 仅 ml 生效
                max_side: int = 960,              # 仅 ml 生效
                ) -> list[dict]:
    """
    检测图中的文字区域，返回文字框列表（原图坐标）：
        [{"x0":int,"y0":int,"x1":int,"y1":int}, ...]
    框已按行合并、外扩 pad 并裁剪到图像边界。未检测到时返回空列表。

    method:
        "classic"  - 纯 CV（中值模糊+梯度），离线，零依赖（本函数实现）
        "ml"       - 轻量 DBNet (PP-OCRv4 det ONNX, ~5MB), 中文/小字召回更好
                     首次使用会自动从 HuggingFace 下载模型 (core/models/det/),
                     下载一次后离线可用, 之后 ~0.08s/张 (4096x2160, CPU).
    """
    if method == "ml":
        from core import ml_text_select as _ml
        return _ml.detect_text_ml(
            raw, strength=strength,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            max_box_ratio=max_box_ratio,
            box_threshold=box_threshold,
            max_side=max_side,
            pad=pad,
        )

    return _detect_text_classic(
        raw, strength=strength, min_area=min_area,
        max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
        vthr=vthr, pad=pad, work_max=work_max,
    )


def _classic_cand(rgb, strength, work_max=1280):
    """经典 CV 文字候选图（工作尺度二值图）与尺度信息。

    返回 (cand, scale, wH, wW, H, W)；rgb 为空时返回 None。
    cand 为工作尺度下的「文字候选」二值图(>0 即候选文字像素)，供
    _detect_text_classic(合并成框) 与 _detect_text_mask_classic(上采样成蒙版) 复用。
    """
    H, W = rgb.shape[:2]
    total = H * W
    if total == 0:
        return None
    s = float(np.clip(strength, 0, 1))

    # 1) 尺度归一：下采样到工作尺寸(最长边 <= work_max)。
    scale = 1.0
    if max(H, W) > work_max:
        scale = work_max / max(H, W)
        work = cv2.resize(rgb, (max(1, int(round(W * scale))), max(1, int(round(H * scale)))),
                          interpolation=cv2.INTER_AREA)
    else:
        work = rgb
    wH, wW = work.shape[:2]

    # 2) 块大小与图尺寸成比例
    block = max(15, ((min(wH, wW) // 18) | 1))
    c_val = max(3, int(round(8 - 5 * s)))

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)

    # 3) 局部对比：自适应阈值捕捉"相对邻域更暗"或"更亮"的清晰笔画
    adapt_dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block, c_val)   # 深字
    adapt_light = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, block, c_val)     # 浅字
    adapt = cv2.bitwise_or(adapt_dark, adapt_light)

    # 4) 真边缘门限：文字笔画是清晰边缘；平纹面料梯度弱，被挡掉
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    gthr = max(8.0, float(np.percentile(grad, 88 + 2 * (1 - s))),
               0.12 * float(grad.max()))
    edge = (grad > gthr).astype(np.uint8) * 255

    # 5) 候选 = 局部显著对比 且 是真边缘
    cand = cv2.bitwise_and(adapt, edge)

    # 6) 形态学：先闭(连笔画)后开(去孤立点)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return cand, scale, wH, wW, H, W


def _clean_text_mask(mask, H, W, min_area=30, max_area_ratio=0.05):
    """连通域清理：去掉极小噪点(<max(min_area,8))、过大块(>整图比例)、过细/过粗组件。
    返回清理后的 HxW uint8 蒙版(255=文字)。"""
    total = H * W
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < max(min_area, 8):
            continue
        if a > total * max_area_ratio:
            continue
        if w / h > 25 or h / w > 6:     # 又长又细 → UI 横线/分隔线
            continue
        keep[lbl == i] = 255
    return keep


def _mask_to_boxes(mask):
    """由蒙版连通域生成显示用框列表（原图坐标，未合并）。"""
    boxes = []
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append({"x0": x, "y0": y, "x1": x + w, "y1": y + h})
    return boxes


def _detect_text_classic(
    raw,
    strength: float = DEFAULTS["strength"],
    min_area: int = DEFAULTS["min_area"],
    max_area_ratio: float = DEFAULTS["max_area_ratio"],
    max_box_ratio: float = DEFAULTS["max_box_ratio"],
    vthr: int = DEFAULTS["vthr"],
    pad: int = DEFAULTS["pad"],
    work_max: int = DEFAULTS["work_max"],
):
    """经典 CV 文字检测实现: 自适应阈值 + 梯度 + 行合并."""
    rgb = to_rgb_uint8(raw)
    res = _classic_cand(rgb, strength, work_max)
    if res is None:
        return []
    cand, scale, wH, wW, H, W = res
    wTotal = wH * wW

    # 7) 强连通域过滤：尺度归一后用一套与图大小无关的几何门限
    min_h = max(5, int(round(0.012 * min(wH, wW))))   # 文字行最小高度(工作尺度)
    max_h = max(min_h + 4, int(round(0.35 * min(wH, wW))))  # 文字行最大高度
    solid_area_gate = wTotal * 0.004                   # 仅对"较大块"启用实心度判定

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    raw_comps = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0 or a < min_area:
            continue
        if a > wTotal * max_area_ratio:
            continue
        # 几何过滤 —— UI 分隔线/按钮边框/扫描线伪影的核心特征
        if h < min_h or h > max_h:
            continue
        if w / h > 25:                                 # 又长又细 → UI 横线/分隔线
            continue
        if h / w > 6:                                  # 又细又长（垂直分隔线）
            continue
        # 组件级密度：文字字符 bbox 内前景占 0.15~0.80（经 3x3 闭运算后多数达 0.65~0.80）；
        # 太稀疏(<0.08) 是 UI 薄线/孤立点；太密(>0.85) 是实心面板/大色块。
        density = a / float(w * h)
        if density < 0.08 or density > 0.85:
            continue
        raw_comps.append((i, x, y, w, h, a))

    # 8) 笔画宽度一致性过滤：用距离变换的骨架半径(90 分位)做判定。
    #    文字字符笔画骨架半径通常在 0.5~0.06×min_dim 之间；
    #    UI 薄线骨架半径 <= 0.5(1px 线)；实心面板 >= 0.08×min_dim。
    if raw_comps:
        dist = cv2.distanceTransform(cand, cv2.DIST_L2, 3)
        keep = []
        ref = min(wH, wW)
        sw_lo = 0.5           # 太细 → 1px 薄线/噪点
        sw_hi = ref * 0.08    # 太粗 → 实心大块
        for (i, x, y, w, h, a) in raw_comps:
            sub_dist = dist[y:y + h, x:x + w]
            sub_lbl = lbl[y:y + h, x:x + w]
            mask = (sub_lbl == i)
            if not mask.any():
                continue
            ds = sub_dist[mask]
            if ds.size == 0:
                continue
            skel = float(np.percentile(ds, 90))
            if skel < sw_lo or skel > sw_hi:
                continue
            keep.append([x, y, w, h])
        comps = keep
    else:
        comps = []

    # 9) 行合并 + 行级二次过滤 + 外扩 + 缩放回原图
    return _merge_and_filter(comps, vthr=vthr, pad=pad,
                             wH=wH, wW=wW, H=H, W=W, scale=scale,
                             max_box_ratio=max_box_ratio, total=wTotal,
                             cand=cand)


def _merge_and_filter(comps, vthr, pad, wH, wW, H, W, scale,
                      max_box_ratio, total, cand):
    """工作尺度下合并组件 → 行级二次过滤 → 缩放回原图坐标。"""
    if not comps:
        return []

    # 按 (y 中心, x) 排序，贪心合并到行
    comps = sorted(comps, key=lambda b: (b[1] + b[3] / 2.0, b[0]))
    groups = []
    comp_widths = []  # 累计每组中的组件宽度之和,用于后面算覆盖率
    for (x, y, w, h) in comps:
        cy = y + h / 2.0
        # 垂直容差只看组件自身高度(不随已合并行高膨胀)
        v_tol = max(8.0, 0.5 * h)
        # 横向合并容差：相邻字符间距 ≈ 字符高度，超此即"另起一行"
        max_gap = max(4.0, 1.5 * h)
        placed = False
        for gi, g in enumerate(groups):
            if abs(cy - g["cy"]) <= v_tol:
                # 计算与当前行最右端的横向 gap
                gap = x - g["x1"]
                if gap <= max_gap:
                    # 真合并：扩展 bbox
                    g["x0"] = min(g["x0"], x); g["y0"] = min(g["y0"], y)
                    g["x1"] = max(g["x1"], x + w); g["y1"] = max(g["y1"], y + h)
                    g["cy"] = (g["y0"] + g["y1"]) / 2.0
                    comp_widths[gi] += w
                    placed = True
                    break
                # 否则同 y 带但 gap 过大 → 不合入，留作"独立候选"
        if not placed:
            groups.append({"x0": x, "y0": y, "x1": x + w, "y1": y + h, "cy": cy})
            comp_widths.append(w)

    # 行级二次过滤：在工作尺度下判断"像不像一行文字"
    line_min_h = max(6, int(round(0.014 * min(wH, wW))))
    line_max_h = max(line_min_h + 4, int(round(0.30 * min(wH, wW))))
    line_max_w = int(round(0.45 * wW))  # 文字行最长 ≈ 半个图宽,再长就不是文字
    out = []
    for gi, g in enumerate(groups):
        gx0, gy0, gx1, gy1 = g["x0"], g["y0"], g["x1"], g["y1"]
        line_h = gy1 - gy0
        line_w = gx1 - gx0
        if line_h < line_min_h or line_h > line_max_h:
            continue
        if line_w > line_max_w:                            # 行宽过大 → 跨半图的杂物合流
            continue
        if line_w / line_h > 12:                           # 又长又扁 → UI 长条
            continue
        if line_w <= 0 or line_h <= 0:
            continue
        # 组件覆盖率：累计组件宽 / 行宽
        #   文字行 ≈ 0.4~0.8（字符彼此紧邻或间隔均匀）
        #   织物零散合流 ≈ 0.05~0.25
        coverage = comp_widths[gi] / float(line_w)
        if coverage < 0.30:
            continue
        # 工作尺度下外扩 pad（按工作尺度下像素计），裁剪到图界
        px0 = max(0, int(gx0 - pad)); py0 = max(0, int(gy0 - pad))
        px1 = min(wW - 1, int(gx1 + pad)); py1 = min(wH - 1, int(gy1 + pad))
        if px1 - px0 <= 1 or py1 - py0 <= 1:
            continue
        # 行内有效密度（候选前景占比）—— 文字通常 0.05~0.4；UI 长条 < 0.03；实心面板 > 0.6
        sub = cand[py0:py1 + 1, px0:px1 + 1]
        density = float((sub > 0).sum()) / max(1, sub.size)
        if density < 0.025 or density > 0.6:
            continue
        # 缩放回原图坐标
        inv = 1.0 / scale
        x0 = max(0, int(round(px0 * inv)))
        y0 = max(0, int(round(py0 * inv)))
        x1 = min(W - 1, int(round((px1 + 1) * inv)))
        y1 = min(H - 1, int(round((py1 + 1) * inv)))
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            continue
        # 原图尺度超大框防护
        if (x1 - x0) * (y1 - y0) > total * max_box_ratio:
            continue
        out.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return out


# ---------------------------------------------------------------------------
# 文字「边缘蒙版」检测（逐像素字形，非整框）
# ---------------------------------------------------------------------------
def _detect_text_mask_classic(raw, boxes=None, strength=DEFAULTS["strength"],
                              min_area=DEFAULTS["min_area"],
                              q_off: float = 50.0):
    """逐像素文字蒙版（Otsu 双峰分割，类 PS 魔棒精度）。

    对每个文字框：
      1) Otsu 找亮度直方图双峰间的谷底阈值(自动适配暗字/亮字)；
      2) 取像素较少的少数侧作为文字(UI 文字在框内通常<50%)，相等时取更极端侧；
      3) 低对比度防护(两均值差<20 视为无字形)→ 跳过(避免织物/均匀块误选)；
      4) 1px MORPH_CLOSE 桥接笔画内 1px 抗锯齿断口(保护小字细笔画)；
      5) 紧密度 q_off 控附加膨胀 0~2px(越高越贴字形)；
      6) 框内连通域清理(放宽下限保小字, 去掉整片背景)。

    放弃「亮度分位+强边相交+2px 膨胀」旧方案:
      旧方案对"明显颜色差"双峰会选错分位→把背景大块圈进来; 强边只留轮廓细线
      → 2px 膨胀把细线变成散布红点; 细笔画小字(<20px)被 CC 过滤删光。
    """
    rgb = to_rgb_uint8(raw)
    H, W = rgb.shape[:2]
    if not boxes:
        boxes = _detect_text_classic(rgb, strength=strength)
    if not boxes:
        return np.zeros((H, W), np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros((H, W), np.uint8)

    # q_off∈[30,70] -> 附加膨胀: 70→0(最紧), 50→1, 30→2(略胖)
    extra_dilate = int(round((60.0 - float(q_off)) / 10.0))
    extra_dilate = max(0, min(2, extra_dilate))

    pad = 8
    for b in boxes:
        x0 = max(0, int(b["x0"]) - pad); y0 = max(0, int(b["y0"]) - pad)
        x1 = min(W, int(b["x1"]) + pad); y1 = min(H, int(b["y1"]) + pad)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        sub = gray[y0:y1, x0:x1]
        # 1) Otsu 双峰谷底阈值
        thr, _ = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if thr <= 0:
            thr = 1
        if thr >= 255:
            thr = 254
        below = sub <= thr
        above = ~below
        cnt_b = int(below.sum()); cnt_a = int(above.sum())
        if cnt_b == 0 or cnt_a == 0:
            continue
        m_b = float(sub[below].mean()); m_a = float(sub[above].mean())
        # 3) 低对比度防护(织物/均匀块): 两均值差<20 视为无字形
        if abs(m_b - m_a) < 20:
            continue
        # 2) 少数侧 = 文字(相等时取更极端侧, 远离 127.5)
        if cnt_b < cnt_a:
            sel = below
        elif cnt_a < cnt_b:
            sel = above
        else:
            sel = below if abs(m_b - 127.5) > abs(m_a - 127.5) else above
        sel = sel.astype(np.uint8) * 255
        # 4) 1px 桥接抗锯齿断口(小字细笔画靠这个连起来)
        sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        # 5) 紧密度: 附加膨胀
        if extra_dilate > 0:
            sel = cv2.dilate(sel, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                             iterations=extra_dilate)
        # 6) 框内连通域清理(放宽下限, 保护小字细笔画)
        box_area = max(1, (x1 - x0) * (y1 - y0))
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(sel, connectivity=8)
        min_keep = max(4, int(box_area * 0.001))   # >= 4px, 保小字
        max_keep = int(box_area * 0.85)
        keep = np.zeros_like(sel)
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < min_keep or a > max_keep:
                continue
            keep[lbl == i] = 255
        # 紧致度防护: 最大连通域 bbox 填充率>0.70 且占框>12%
        # → 整片色块(织物/均匀块), 丢弃(文字笔画密度 0.3~0.5, 图标 0.4~0.6)
        if keep.any() and n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            idx = int(np.argmax(areas))
            a_max = int(stats[idx + 1, cv2.CC_STAT_AREA])
            w_max = int(stats[idx + 1, cv2.CC_STAT_WIDTH])
            h_max = int(stats[idx + 1, cv2.CC_STAT_HEIGHT])
            if a_max > int(box_area * 0.12) and w_max > 0 and h_max > 0:
                if (a_max / float(w_max * h_max)) > 0.70:
                    keep[:] = 0
        mask[y0:y1, x0:x1][keep > 0] = 255
    return mask


def _fill_nearby_white(rgb: np.ndarray, mask: np.ndarray,
                       pad: int = 6, min_lum: int = 200,
                       rounds: int = 5, max_dist: int = 12,
                       aa_lum: int = 185, aa_dist: int = 3,
                       aa_tail_lum: int = 145, aa_tail_rounds: int = 6) -> np.ndarray:
    """把**紧邻蒙版**的高亮像素并入蒙版。

    结构：① 连通扩散(pad×rounds)吃纯白外延线；② 距离场(<=max_dist)吃孤立
    纯白段；③ 近距低亮度档(距蒙版<=aa_dist 且 亮度>=aa_lum)吃 AA 渐隐的
    字尖/弯钩尾(185~199 灰白, 200 阈值吃不到, 距字形边缘 1~3px)；
    ④ 尾部 AA 连续外推(1px 步进, ^aa_tail_rounds 轮): 从蒙版出发, 只沿
    八连通的 >=aa_tail_lum 像素走, 兜住更深的渐隐尾(145~184); 连通性保证
    不会跨背景(<aa_tail_lum)跳到远处亮块, 也就不会把背景纹理圈进来。
    实测座驾2.png: 广字撇/马弯钩尾的最后一溜渐隐像素亮度 150~164、8连通
    紧贴蒙版(dist≈1), 165 档吃不到 → 降到 145(背景中值≈43, p95≈90,
    阈值显著高于背景, 不会误吞); DBNet 常把整行字合成单框, 广撇尾尖离框
    5~6px, 4 轮外推差一步 → 增至 6 轮(座驾2 仅 +2px 兜住尾尖)。
    ①②③④ 的绝对亮度下限随距离收紧, 中灰背景/描边不被误收。
    """
    if not mask.any():
        return mask
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cur = mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    for _ in range(rounds):
        dil = cv2.dilate(cur, kernel)
        add = (dil > 0) & (lum > min_lum)
        new = cv2.bitwise_or(cur, add.astype(np.uint8) * 255)
        if not bool((new != cur).any()):
            break
        cur = new
    add = np.zeros_like(cur, bool)
    if max_dist > 0 and max_dist < 1024:
        dist = cv2.distanceTransform((cur == 0).astype(np.uint8),
                                     cv2.DIST_L2, 3)
        add |= (dist <= max_dist) & (lum > min_lum)
        if aa_lum > 0 and aa_dist > 0:
            add |= (dist <= aa_dist) & (lum >= aa_lum)
    if add.any():
        cur = cv2.bitwise_or(cur, add.astype(np.uint8) * 255)
    # ④ 尾部 AA 连续外推: 1px 椭圆逐步外扩, 只走亮度>=aa_tail_lum 的像素
    if aa_tail_lum > 0 and aa_tail_rounds > 0:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(aa_tail_rounds):
            dil = cv2.dilate(cur, k1)
            add = (dil > 0) & (lum >= aa_tail_lum)
            new = cv2.bitwise_or(cur, add.astype(np.uint8) * 255)
            if not bool((new != cur).any()):
                break
            cur = new
    return cur


def _grow_color_tint(rgb: np.ndarray, mask: np.ndarray,
                     red_thr: int = 30, green_thr: int = 15,
                     green_g: int = 100, rounds_max: int = 120,
                     max_grow_ratio: float = 5.0) -> np.ndarray:
    """沿「色偏像素」从文字蒙版出发八连通生长，吞并整片色偏覆盖区。

    解决两类「亮度法(Otsu)吃不到」的半透明色偏文字：
      - 红蒙版叠加(如座驾2_蒙版.png)：红色覆盖区向外延展可超蒙版 30~50px，
        R - max(G,B) > 30 主导；
      - 淡绿光晕(白字发光)：G - max(R,B) > 15 且 G > 100 的绿色光晕区。
    只沿色偏像素生长、不跨背景，因此无需猜膨胀半径即可 100% 吞并与蒙版
    八连通的整片色偏区域(实测红蒙版覆盖 65.8%→100%)。

    防误伤(避免影响无发光的普通文字)：
      - 生长面积上限 = mask × max_grow_ratio。光晕/红蒙版叠加实测生长量仅
        1.5~2.5×；若文字紧贴大块红/绿背景，生长会超过该上限 → 回退到上一轮，
        防止把整块彩色背景吞进蒙版造成过度填充。普通文字(无红绿色偏邻域)
        生长量 = 0，蒙版完全不变。
    """
    if not mask.any():
        return mask
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    red_tint = (r - np.maximum(g, b) > red_thr)
    green_tint = (g - np.maximum(r, b) > green_thr) & (g > green_g)
    tint = red_tint | green_tint
    if not tint.any():
        return mask
    cur = (mask > 0).astype(np.uint8)
    cap = max(1, int((mask > 0).sum() * max_grow_ratio))
    k = np.ones((3, 3), np.uint8)
    prev = cur.copy()
    for _ in range(rounds_max):
        dil = cv2.dilate(cur, k) > 0
        add = dil & tint & (cur == 0)
        if not add.any():
            break
        cur[add] = 1
        if int(cur.sum()) > cap:      # 面积超限 → 回退, 视为吞到背景色块
            cur = prev
            break
        prev = cur.copy()
    return cur * 255


def _recover_bg(P: np.ndarray, a: np.ndarray, glow_c: np.ndarray,
                bg: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """把发光像素「拉到背景色」，使去发光区与周围背景亮度一致。

    P      : (n,3) 观察到的像素值
    a      : (n,)  按 G 通道估计的发光透明度(仅作参考, 不再参与最终颜色)
    glow_c : (3,)  光晕色(仅参考)
    bg     : (3,)  背景色
    strength: [0,1] 去除力度。1=完全拉到背景色, 0=不动。

    反解公式 rec=(P-a*glow)/(1-a) 对「文字边缘+光晕」的混合像素极不稳定,
    会爆出紫/黄绿等杂色(用户实测: 尿渍一样泛绿); 而按 alpha 比例拉动又会
    残留部分原始亮度, 使去发光区比周围背景亮(用户实测)。因此统一**按
    strength 比例全部拉到背景色**: out = lerp(P, bg, s)。s=1 时去发光区
    与背景完全一致, 不偏亮、不泛绿。
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0:
        return P
    return P + (bg[None] - P) * s


def _text_like_thresholds(protect: float) -> tuple[int, int]:
    """白字保护阈值随保护强度缩放。protect=1 → (150,170) 现有行为;
    protect=0 → (256,255) 关闭保护(所有弱绿都可拉平, 可能伤文字 AA)。
    """
    p = float(np.clip(protect, 0.0, 1.0))
    return int(round(150 + 106 * (1 - p))), int(round(170 + 85 * (1 - p)))


def _deglow_faint_green(rgb: np.ndarray, mask: np.ndarray,
                        near_r: int = 24, thr: int = 6,
                        g_lo: int = 85, thr_strong: int = 15,
                        g_strong: int = 100,
                        text_protect: float = 1.0,
                        min_strong: int = 50,
                        strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """对文字蒙版外围的「弱绿光晕边缘」做通道法去发光（仅弱区，强区留给填充）。

    思路(A 通道法)：半透明光晕可建模为 观察值 = 背景*(1-a) + 光晕色*a。
      对每个弱光晕像素用 G 通道估计 alpha，再反解出背景并回填 —— 把残余的
      淡绿渐隐边缘直接还原成背景色，而不是把整圈弱区都塞进填充蒙版。
    边界条件(防误伤普通文字)：
      - 无发光时绿色像素 = 0 → 直接返回原图，普通图片零改动；
      - 必须存在「强光晕」信号(强绿像素 >= min_strong) 才启用 —— 无发光图片
        即使有个别绿色边缘也不会被重着色；
      - 只处理「弱」(thr=6 档) 且 紧邻蒙版/强光晕(near_r 圈)内 的像素，
        远处的真绿色物体不会被误去色；
      - 强光晕(>=thr_strong 且 G>g_strong)由 _grow_color_tint 并入蒙版填充，
        这里显式排除，避免双重处理。
      - 白字保护：红/蓝通道高(>=150)或全通道偏亮 → 文字及抗锯齿边缘不反解，
        避免文字被"吹胖"。
    strength ∈ [0,1]：去发光力度。0=不处理，1=完全反解(默认)。

    返回 (去发光后的 rgb, 弱区 mask)。
    """
    out = rgb.astype(np.int16)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0:
        return rgb, np.zeros(rgb.shape[:2], bool)
    strong_green = (g - np.maximum(r, b) > thr_strong) & (g > g_strong)
    # 无强光晕信号 → 普通文字图, 不启用去发光, 零改动
    if int(strong_green.sum()) < min_strong:
        return rgb, strong_green & ~(mask > 0)
    green_weak = (g - np.maximum(r, b) > thr) & (g > g_lo)
    strong = (mask > 0) | strong_green
    if not green_weak.any():
        return rgb, green_weak & ~strong
    near = cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (near_r * 2 + 1, near_r * 2 + 1))) > 0
    # 白字/文字边缘保护(阈值随 text_protect 缩放)
    tl_rb, tl_min = _text_like_thresholds(text_protect)
    text_like = (r > tl_rb) | (b > tl_rb) | (np.minimum(np.minimum(r, g), b) > tl_min)
    weak = green_weak & ~strong & near & ~text_like
    if not weak.any():
        return rgb, weak
    # 背景色：非绿非亮区中值
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg_msk = ~green_weak & (gray < 160)
    bg = rgb[bg_msk].mean(0).astype(np.float32) if bg_msk.any() \
        else np.array([89, 81, 72], np.float32)
    # 光晕色：强区 G-B 最高 30% 像素均值
    gb = (g - b)[strong]
    if gb.size:
        sel = strong & ((g - b) >= np.percentile(gb, 70))
        glow_c = rgb[sel].mean(0).astype(np.float32)
    else:
        glow_c = np.array([160, 220, 140], np.float32)
    # 用 G 通道(信号最强)估计 alpha (0~1); 力度由 _recover_bg 的 strength 控制
    a = np.clip((g[weak].astype(np.float32) - bg[1]) / (glow_c[1] - bg[1] + 1e-6), 0.0, 1.0)
    P = out[weak].astype(np.float32)
    out[weak] = _recover_bg(P, a, glow_c, bg, strength=s)
    return out.clip(0, 255).astype(np.uint8), weak


def _deglow_full_green(rgb: np.ndarray, tmask: np.ndarray,
                       g_thr: int = 2, g_lo: int = 70,
                       min_strong: int = 30,
                       white_floor: int = 120,
                       rounds_max: int = 400,
                       strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """实验性「先去发光」：从检测出的文字蒙版出发，沿「绿色或比背景亮」的像素
    生长出整片发光区；**发光区内除真实笔画外全部拉到背景色**，文字笔画提亮回纯白。

    设计要点(针对用户反馈)：
      - 发光范围自适应且不写死：生长条件 = 绿主导 或 比背景亮。发光边缘即使
        绿色已淡到阈值以下，也因「比暗背景亮」被纳入发光区 → 完整覆盖真实
        光晕，范围不会偏小。
      - 文字笔画判定：min(R,G,B)>white_floor 且 g-max(R,B)<40 —— 白色文字
        (min 高、g 仅略高≈23) 与发光(g 明显主导≈47+) 区分开。保护圈=笔画+2px。
      - 去除：发光区内「保护圈外」的**全部像素**都拉到背景色(不只是偏绿)。
        光晕比背景整体偏亮, 只去绿会留下"亮而不绿"的残迹(用户实测偏亮),
        全部拉成背景后与周围一致。
      - 背景色用「暗背景分位数」估计(非全局均值, 全局均值被发光残迹拉偏、
        导致去发光区比真实背景亮 16 个灰度)。
      - 文字笔画里的绿色铸色**提亮回纯白**(三通道取 max), 不拉低(拉低会变暗、
        笔画被 Otsu 切掉→缺笔画)。缺笔画由该实际算法解决, 不扩大亮度筛选。

    返回 (去发光后的 rgb, 发光区 mask)。
    """
    out = rgb.astype(np.int16)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0 or not tmask.any():
        return rgb, np.zeros(rgb.shape[:2], bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    green = (g - np.maximum(r, b) > g_thr) & (g > g_lo)
    strong_green = (g - np.maximum(r, b) > 15) & (g > 100)
    # 无强发光信号 → 普通图, 零改动
    if int(strong_green.sum()) < min_strong:
        return rgb, green
    # 文字笔画: min 够亮 且 非强绿(白色文字 g 仅略高; 发光 g 明显主导)
    min_rgb = np.minimum(np.minimum(r, g), b)
    text_stroke = (min_rgb > white_floor) & ((g - np.maximum(r, b)) < 40)
    k3 = np.ones((3, 3), np.uint8)
    # 保护圈: 笔画 + 1px(只保笔画与最外层抗锯齿边; 再宽会把内圈发光也保护住→残留绿)
    protect = cv2.dilate(text_stroke.astype(np.uint8), k3, iterations=1) > 0
    # 生长条件: 绿色 或 比背景亮(光晕亮晕, 含微弱边缘)。背景亮度取「非强绿区」中值。
    bg_cand = gray[~strong_green]
    bg_lum = float(np.median(bg_cand)) if bg_cand.size else 80.0
    bright = (gray > (bg_lum + 10)) & (gray > 70)
    grow_cond = green | bright
    # 从文字蒙版沿 grow_cond 生长出「发光区」(自适应真实范围)
    zone = (tmask > 0).copy()
    cur = zone
    for _ in range(rounds_max):
        dil = cv2.dilate(cur.astype(np.uint8), k3) > 0
        add = dil & grow_cond & ~zone
        if not add.any():
            break
        zone |= add
        cur = zone
    # 去除: 发光区内「文字保护圈之外」的全部像素拉到背景色
    glow = zone & ~protect
    if not glow.any():
        return rgb, glow
    # 背景色: 暗背景分位数估计(排除发光残迹的亮像素, 避免把背景估亮)
    ng = ~green
    bg_thr = float(np.percentile(gray[ng], 30)) if ng.any() else 110.0
    bg_msk = ng & (gray < max(bg_thr, 90))
    bg = rgb[bg_msk].mean(0).astype(np.float32) if bg_msk.any() \
        else np.array([74, 66, 59], np.float32)
    # 光晕色：强绿像素 G-B 最高 30% 均值(仅用于 alpha 估计)
    gb = (g - b)[strong_green]
    if gb.size:
        sel = strong_green & ((g - b) >= np.percentile(gb, 70))
        glow_c = rgb[sel].mean(0).astype(np.float32)
    else:
        glow_c = np.array([160, 220, 140], np.float32)
    # 用 G 通道估计 alpha (0~1); 力度由 _recover_bg 的 strength 控制
    a = np.clip((g[glow].astype(np.float32) - bg[1]) / (glow_c[1] - bg[1] + 1e-6), 0.0, 1.0)
    P = out[glow].astype(np.float32)
    out[glow] = _recover_bg(P, a, glow_c, bg, strength=s)

    # 文字笔画及保护圈内的「绿色铸色」处理: 提亮回纯白(三通道取 max), 不拉低。
    # 保护圈里的内圈发光也会被提亮成中性白灰(不残留绿、不残留亮环)。
    zg = protect & (g - np.maximum(r, b) > 4)
    if zg.any():
        fullmax = np.maximum(np.maximum(r, g), b)
        out[zg, 0] = fullmax[zg]
        out[zg, 1] = fullmax[zg]
        out[zg, 2] = fullmax[zg]
    return out.clip(0, 255).astype(np.uint8), glow


def _deglow_full_green_v2(rgb: np.ndarray, tmask: np.ndarray,
                          g_thr: int = 2, g_lo: int = 70,
                          min_strong: int = 30,
                          white_floor: int = 120,
                          rounds_max: int = 400,
                          strength: float = 1.0,
                          alpha_core: float = 0.65,
                          zone_ratio: float = 0.6,
                          debug: bool = False) -> tuple:
    """原型 v2：发光区用「真·alpha 分解」恢复底层纹理，mask 只收紧到高α核心+文字。

    与 _deglow_full_green(整片发光区 lerp→背景色平涂) 的根本区别：
      - 外圈(α 小、半透明)用 B=(I − α·Glow)/(1−α) 反解恢复底层背景纹理，
        不再把整片拉成单一背景色 → 纹理不丢；
      - alpha 估计在 α 小处数值稳定(分母 1−α 接近 1)，正是光晕主体；
        仅在 α→1 的近文字高不透明区才发散，所以那部分(α>alpha_core)才
        并入填充 mask，交给 patchmatch 只填这一小块 → 不瞎猜大块纹理。
    返回 (去发光后的 rgb, 填充用 mask 0/255)。普通图(无强绿信号)零改动。
    strength∈[0,1]：恢复力度，1=完全按分解恢复。
    """
    out = rgb.astype(np.int16)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    s = float(np.clip(strength, 0.0, 1.0))
    H, W = rgb.shape[:2]
    empty = np.zeros((H, W), np.uint8)
    if s <= 0:
        return rgb, empty

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    green = (g - np.maximum(r, b) > g_thr) & (g > g_lo)
    strong_green = (g - np.maximum(r, b) > 15) & (g > 100)
    if int(strong_green.sum()) < min_strong:        # 无强发光 → 普通图, 零改动
        return rgb, empty

    # 文字笔画: min 够亮 且 非强绿(白色文字 g 仅略高; 发光 g 明显主导)
    min_rgb = np.minimum(np.minimum(r, g), b)
    text_stroke = (min_rgb > white_floor) & ((g - np.maximum(r, b)) < 40)

    # 从文字蒙版/强绿像素沿「绿 | 比背景亮」自适应生长出整片发光区(非写死范围)
    # 以强绿像素播种: 即使强发光图文字检测失败(tmask 空), 发光区也能被覆盖。
    bg_cand = gray[~strong_green]
    bg_lum = float(np.median(bg_cand)) if bg_cand.size else 80.0
    bright = (gray > (bg_lum + 10)) & (gray > 70)
    grow_cond = green | bright
    zone = (strong_green | (tmask > 0)).copy()
    cur = zone
    budget = int(H * W * zone_ratio)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(rounds_max):
        dil = cv2.dilate(cur.astype(np.uint8), k3) > 0
        add = dil & grow_cond & ~zone
        if not add.any():
            break
        zone |= add
        if int(zone.sum()) > budget:        # 超预算(可能吞大块亮背景) → 回退
            zone &= ~add
            break
        cur = zone

    # 按"绿度"(G−max(R,B))从 G 通道减去绿光, 直接去发光:
    #   发光物理 ≈ 在背景上叠加绿光 → I 的 G 通道高出 R/B 的部分就是叠加的绿光
    #   去发光 = G' = G − greenness → G' 接近 max(R,B) → 中性灰(原背景色)
    # R/B 通道保持 → 底层纹理完整保留；只减 G 通道 → 不可能减到黑, 数值天然稳定。
    # 强度 strength 控制去绿力度(1=完全去绿, <1=保留少量绿光晕)。
    m_zone = zone & ~text_stroke
    if not m_zone.any():
        return rgb, (tmask > 0).astype(np.uint8)
    greenness = np.maximum(g.astype(np.int16) - np.maximum(r, b), 0)  # 绿度, 非负
    if m_zone.any():
        Gn = out[m_zone, 1].astype(np.float32) - greenness[m_zone].astype(np.float32) * s
        out[m_zone, 1] = np.clip(Gn, 0, 255).astype(np.int16)

    # 文字笔画约束到「真正的强绿区」附近(dilate(strong_green)), 避免吞远处亮背景。
    _k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    text_stroke_z = text_stroke & (cv2.dilate(strong_green.astype(np.uint8), _k8) > 0)
    # 填充 mask 只含白色文字笔画。发光区已按"减绿度"去光变成中性灰(背景色),
    # 不再进 fill(进 fill 会让 patchmatch 在分解后的单色区上瞎填纹理)。
    core_mask = text_stroke_z.astype(np.uint8) * 255
    clean = out.clip(0, 255).astype(np.uint8)
    if debug:
        dbg = dict(strong_green=strong_green, zone=zone, text_stroke=text_stroke,
                   m_zone=m_zone, greenness=greenness)
        return clean, core_mask, dbg
    return clean, core_mask



def _deglow_faint_green_v11(rgb: np.ndarray, tmask: np.ndarray,
                            thr: int = 6, g_lo: int = 85,
                            thr_strong: int = 15, g_strong: int = 100,
                            min_strong: int = 50,
                            near_r: int = 24,
                            max_zone_ratio: float = 0.25,
                            white_floor: int = 120,
                            text_protect: float = 1.0,
                            tint_fill: bool = True,
                            strength: float = 1.0,
                            ) -> tuple[np.ndarray, np.ndarray, dict]:
    """auto v1.1(保守版)：与 auto 的 A+B 完全同语义，仅把 A 路径的光晕范围
    从「固定 near_r 圈」扩展为「固定 near_r 圈 ∪ 绿|比背景亮 连通生长」。

    为什么并集而不只连通生长：淡绿光晕可能被非绿间隙与强区隔开(如文字暗部)，
    纯连通生长过不去会漏(实测某历史图 weak=0 而 auto 能圈到)；保留 auto 的
    24px 圈保证覆盖不弱于 auto，再叠加连通生长把更远的渐隐光晕补齐。

    其余全部沿用 auto(_deglow_faint_green) 已验证逻辑：
      - weak 判定/文字保护/背景色估计/光晕色估计/G 通道 alpha/_recover_bg
        全局背景拉平 —— 不引入局部背景/减法等新语义，不产紫/杂色，绿残留
        不高于 auto；
      - B 路径(强光晕并入蒙版填充)不变。

    返回 (去发光后的 rgb, 并入填充蒙版的强区 mask 0/255, 统计 dict)。
    普通图(无强绿信号) → 零改动, 返回 (rgb, 全 0, stats)。
    """
    H, W = rgb.shape[:2]
    out = rgb.astype(np.int16)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    s = float(np.clip(strength, 0.0, 1.0))
    stats = {"zone_px": 0, "strong_px": 0, "weak_px": 0, "mode": "none"}
    if s <= 0 or not tmask.any():
        return rgb, np.zeros((H, W), bool), stats

    strong_green = (g - np.maximum(r, b) > thr_strong) & (g > g_strong)
    if int(strong_green.sum()) < min_strong:      # 无强发光 → 普通图, 零改动
        return rgb, np.zeros((H, W), bool), stats
    k3 = np.ones((3, 3), np.uint8)

    # B 路径：色偏生长(红蒙版叠加 + 强绿) → 并入填充蒙版(v1 同款)
    if tint_fill:
        add_mask = _grow_color_tint(rgb, tmask)
    else:
        add_mask = np.zeros_like(tmask)
    add = add_mask > 0
    stats["strong_px"] = int(add.sum())
    strong = add | strong_green

    # A 路径：自适应连通生长(绿|比背景亮) 代替固定 near_r 圈 —— 范围补齐
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg_cand = gray[~strong_green]
    bg_lum = float(np.median(bg_cand)) if bg_cand.size else 80.0
    green_weak = (g - np.maximum(r, b) > thr) & (g > g_lo)
    bright = (gray > (bg_lum + 10)) & (gray > 70)
    grow_cond = green_weak | bright
    zone = strong.copy()
    cur = zone
    budget = int(H * W * max_zone_ratio)
    for _ in range(300):
        dil = cv2.dilate(cur.astype(np.uint8), k3) > 0
        new = dil & grow_cond & ~zone
        if not new.any():
            break
        zone |= new
        if int(zone.sum()) > budget:      # 超预算(可能吞到大块亮背景) → 回退
            zone &= ~new
            break
        cur = zone
    stats["zone_px"] = int(zone.sum())

    # 弱区 = (auto 固定圈 ∪ 连通生长) 内的淡绿像素(排除强区/文字/亮白边缘)
    tl_rb, tl_min = _text_like_thresholds(text_protect)
    text_like = (r > tl_rb) | (b > tl_rb) | (np.minimum(np.minimum(r, g), b) > tl_min)
    if near_r > 0:
        near = cv2.dilate(add_mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (near_r * 2 + 1, near_r * 2 + 1))) > 0
        cover = zone | near
    else:
        cover = zone
    weak = green_weak & ~strong & cover & ~text_like
    stats["weak_px"] = int(weak.sum())
    stats["mode"] = "range_only" if weak.any() else "none"
    if not weak.any():
        return out.clip(0, 255).astype(np.uint8), add_mask, stats

    # 背景色：非绿非亮区均值(与 auto 一致)
    bg_msk = ~green_weak & (gray < 160)
    bg = rgb[bg_msk].mean(0).astype(np.float32) if bg_msk.any() \
        else np.array([89, 81, 72], np.float32)
    # 光晕色：强区 G-B 最高 30% 均值(与 auto 一致)
    gb = (g - b)[strong]
    if gb.size:
        sel = strong & ((g - b) >= np.percentile(gb, 70))
        glow_c = rgb[sel].mean(0).astype(np.float32)
    else:
        glow_c = np.array([160, 220, 140], np.float32)
    # G 通道估计 alpha, _recover_bg 按 strength 拉到全局背景(与 auto 一致)
    a = np.clip((g[weak].astype(np.float32) - bg[1]) / (glow_c[1] - bg[1] + 1e-6), 0.0, 1.0)
    P = out[weak].astype(np.float32)
    out[weak] = _recover_bg(P, a, glow_c, bg, strength=s)
    return out.clip(0, 255).astype(np.uint8), add_mask, stats


def detect_text_mask(raw, strength: float = 1.0, method: str = "ml",
                     min_area: int = 30, max_area_ratio: float = 0.05,
                     max_box_ratio: float = 0.40,
                     max_side: int = 960, work_max: int = 1280,
                     q_off: float = 50.0, tint_fill: bool = True,
                     fill_white: bool = True,
                     fill_max_dist: int = 12):
    """
    文字「边缘蒙版」检测：返回 (mask, boxes)。

        mask  : HxW uint8, 255=文字像素（逐像素字形，非整框）
        boxes : 显示用文字框列表（原图坐标）

    与 detect_text(只返回整框) 的区别：这里给出**逐像素**文字蒙版，
    使 patchmode 只填充字形本身、参考区自动取文字之外的全部纹理。

    method 影响**文字框定位**（detect_text）：
        "ml"      - 轻量 DBNet(PP-OCRv4 det)，中文/小字召回更好 → 框更准；
        "classic" - 经典 CV 候选图合并，离线零依赖。
    字形蒙版统一走 **Otsu 双峰分割 + 亮白补全 + 背景相对补全 + 色偏生长**：
        框内 Otsu 取文字侧(少数侧)+1px 桥接+低对比度防护(_detect_text_mask_classic)；
        再对蒙版邻域内的高亮纯白像素补全(_fill_nearby_white) —— 修复描边字
        (白字身+灰/红描边)上 Otsu 只取一侧导致的漏白，且不会把背景灰块圈进来；
        fill_white=False 时跳过该步，蒙版回到纯 Otsu(近似早期 eoff 行为)，
        小张图的非文字浅色区/衣物高光不再被误锁进填充蒙版；
        fill_max_dist 控制「孤立纯白段」步骤的最大距离(默认 12px)。
        字符白边/AA 通常 <8px(由 ④ AA 尾部外推兜住),12 足够;
        原默认 32 会在"暗背景+亮光斑/光效"图上把远处光斑误锁为填充蒙版
        (换装.png 实测顶部 31px 远的光斑被吞 497px),需要更收敛。
        再做背景相对亮度补全(_fill_bright_near_mask) —— 兜住发光文字较暗的笔画尾；
        最后沿红/绿色偏像素区域生长(_grow_color_tint, tint_fill=True 时)——
        吞并红蒙版叠加区/淡绿光晕区，修复亮度法吃不到的半透明色偏文字。

    蒙版紧密度：
        q_off : [30,70]，越高蒙版越贴字形(附加膨胀越少); 越低略胖。
    """
    rgb = to_rgb_uint8(raw)
    H, W = rgb.shape[:2]
    if H * W == 0:
        return np.zeros((H, W), np.uint8), []

    # 1) 文字框定位(用所选 method)
    boxes = detect_text(rgb, strength=strength, method=method,
                        max_area_ratio=max_area_ratio,
                        max_box_ratio=max_box_ratio, work_max=work_max,
                        max_side=max_side, min_area=min_area)
    if not boxes:
        return np.zeros((H, W), np.uint8), []

    # 2) 精细字形蒙版(亮度+强边)
    mask = _detect_text_mask_classic(rgb, boxes=boxes, strength=strength,
                                    min_area=min_area, q_off=q_off)
    if not mask.any():
        return np.zeros((H, W), np.uint8), []

    # 3) 临近纯白补全：只补紧邻蒙版的亮白像素, 不进背景
    #    fill_white=False 跳过此步 → 蒙版回到纯 Otsu(早期 eoff 行为),
    #    小张图非文字浅色区/衣物高光不再被误锁进填充蒙版。
    if fill_white:
        mask = _fill_nearby_white(rgb, mask, max_dist=fill_max_dist)
    # 4) 色偏区域生长：吞并紧邻蒙版的整片红/绿色偏覆盖区(红蒙版叠加/淡绿光晕)
    if tint_fill:
        mask = _grow_color_tint(rgb, mask)
    mask = _clean_text_mask(mask, H, W, min_area=min(
        min_area, 8), max_area_ratio=0.9)
    return mask, _mask_to_boxes(mask)