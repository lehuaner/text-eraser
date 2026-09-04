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
# Shared-algorithm-core cv2 shim: routes dilate/erode/morphologyEx/connectedComponents/
# cvtColor(RGB2GRAY) through textcore.wasm (same operators the browser runs) and falls
# through to the real cv2 for everything else. Keeps the backend + browser parity.
from text_eraser import _cv as cv2
from text_eraser._shared_core import grow_color_tint as _sc_grow_color_tint

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
        from text_eraser import ml_text_select as _ml
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
    # 先过面积门, 收集候选; 细长门带「远超同伴」约束: 字形的孤立竖/横笔画
    # (测1787981 贝部竖 5x40, 长宽比 8.0)与相邻字高相当(h40 vs 同掩码字高中位
    # 51), 纯长宽比门 h/w>6 会把它当「垂直分隔线」整条删掉 → 蒙版缺笔画;
    # UI 分隔线通常远超文字行高 → 只有高/宽同时超过长宽比门 **且** 超过其余
    # 组件中位尺寸 1.5 倍才删。无同伴时维持旧的纯长宽比判定(纯分隔线图)。
    cand = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < max(min_area, 8) or a > total * max_area_ratio:
            continue
        cand.append(i)
    hs = sorted(int(stats[i, cv2.CC_STAT_HEIGHT]) for i in cand)
    ws = sorted(int(stats[i, cv2.CC_STAT_WIDTH]) for i in cand)
    keep = np.zeros((H, W), np.uint8)
    for k, i in enumerate(cand):
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        # 同伴 = 除自己外的其余候选(否则单一分隔线的中位数就是它自己, 删不掉)
        if len(cand) > 1:
            oh = hs[:k] + hs[k + 1:]
            ow = ws[:k] + ws[k + 1:]
            med_h = float(oh[len(oh) // 2])
            med_w = float(ow[len(ow) // 2])
            tall_gate = h > 1.5 * med_h
            wide_gate = w > 1.5 * med_w
        else:
            tall_gate = wide_gate = True   # 无同伴 → 纯长宽比判定(纯分隔线图)
        if w / h > 25 and wide_gate:       # 又长又细且远超同伴宽 → UI 横线
            continue
        if h / w > 6 and tall_gate:        # 又细又长且远超同伴高 → 垂直分隔线
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
                              q_off: float = 50.0,
                              upscale: bool = True):
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

    def seg_box(gs):
        """单框字形分割(Otsu 少数侧 + 桥接/膨胀 + 连通域清理), 返回 gs 形状的
        0/255 蒙版。gs 可为原尺度或放大后的灰度子图, 门槛随 gs 面积等比。
        """
        thr, _ = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if thr <= 0:
            thr = 1
        if thr >= 255:
            thr = 254
        below = gs <= thr
        above = ~below
        cnt_b = int(below.sum()); cnt_a = int(above.sum())
        if cnt_b == 0 or cnt_a == 0:
            return np.zeros_like(gs)
        m_b = float(gs[below].mean()); m_a = float(gs[above].mean())
        # 3) 低对比度防护(织物/均匀块): 两均值差<20 视为无字形
        if abs(m_b - m_a) < 20:
            return np.zeros_like(gs)
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
        box_area = max(1, gs.shape[0] * gs.shape[1])
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
        return keep

    pad = 8
    for b in boxes:
        x0 = max(0, int(b["x0"]) - pad); y0 = max(0, int(b["y0"]) - pad)
        x1 = min(W, int(b["x1"]) + pad); y1 = min(H, int(b["y1"]) + pad)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        # 方案A(低分辨率蒙版覆盖): 小字框内先把灰度放大 2~3x 再分割一次, 与
        # 原尺度结果**取并集**(只增不减) —— 放大后细笔画(1~2px)越过 CC 面积
        # 下限/长宽比门, 修低清图上被整段误删的撇/钩尾; 原尺度结果保底,
        # 放大带来的 Otsu 阈值微移不会让蒙版变小。
        # 判定用「去 pad 后的字形高度」(框已外扩 2*pad), 字符 >56px 时分辨率
        # 已足够 → 不放大, 普通图/大图零变化。
        sub = gray[y0:y1, x0:x1]
        keep = seg_box(sub)
        h_char = (y1 - y0) - 2 * pad
        s_fac = 1
        if upscale and h_char < 56:
            s_fac = 2 if h_char >= 24 else 3
        if s_fac > 1:
            sub_up = cv2.resize(sub, ((x1 - x0) * s_fac, (y1 - y0) * s_fac),
                                interpolation=cv2.INTER_CUBIC)
            keep_up = seg_box(sub_up)
            # 缩回原尺度(细笔画 >127 保留), 并入原尺度结果
            keep_up = cv2.resize(keep_up, (x1 - x0, y1 - y0),
                                 interpolation=cv2.INTER_AREA) > 127
            keep = cv2.bitwise_or(keep, keep_up.astype(np.uint8) * 255)
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
    H, W = mask.shape[:2]

    def _bbox(cur):
        ys, xs = np.where(cur > 0)
        return ys, xs

    def _crop_round(cur, kernel, thr):
        """一轮「裁剪域膨胀 + 阈值吸收」：膨胀是局部算子, 裁剪边距 = 2×核半径+1
        (本轮可被新增的像素 ≤ 核半径, 其膨胀窗再外扩核半径) 时, 裁剪域内的结果
        与全图膨胀逐位一致 —— 把 O(全图) 轮次降到 O(局部)。"""
        ys, xs = np.where(cur > 0)
        r = kernel.shape[0] // 2 * 2 + 1
        y0 = max(0, int(ys.min()) - r); y1 = min(H, int(ys.max()) + 1 + r)
        x0 = max(0, int(xs.min()) - r); x1 = min(W, int(xs.max()) + 1 + r)
        sub = cur[y0:y1, x0:x1]
        dil = cv2.dilate(sub, kernel)
        add = (dil > 0) & (lum[y0:y1, x0:x1] > thr)
        new_sub = cv2.bitwise_or(sub, add.astype(np.uint8) * 255)
        changed = bool((new_sub != sub).any())
        if changed:
            cur[y0:y1, x0:x1] = new_sub
        return changed

    cur = mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    for _ in range(rounds):
        if not _crop_round(cur, kernel, min_lum):
            break
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
            if not _crop_round(cur, k1, aa_tail_lum - 1):
                break
    return cur


def _fill_bright_near_mask(rgb: np.ndarray, mask: np.ndarray,
                           bg_lo: int = 25, lum_off: int = 24,
                           min_rgb: int = 118, green_gate: int = 26,
                           rounds: int = 6, ext_thr: int = 20) -> np.ndarray:
    """白字亮侧连通补全（方案B）：吃掉文字边缘 1~3px 的浅色残留。

    实测低分辨率白字(如 180px 缩略图上的「新」)在 v2 结果里的「碎块」全部落在
    距填充蒙版 0~2px、亮度 130~137 的中性灰带 —— 是白字与背景之间的 AA 渐隐
    环带(1~2px), Otsu 核心蒙版止于 ~255 高亮处, 这条带被切在蒙版外; 低清图
    上它还连着更淡的光晕残迹, 视觉上像没擦干净/碎块。膨胀只能包围已有种子,
    对整条环带无效; 这里从蒙版出发, 沿「比背景亮 lum_off+ 且 近白」像素连通
    生长 ≤rounds 轮(默认 6px, 覆盖 1~3px 环带 + 少量余量, 不扫远处大块), 把
    环带并入填充蒙版, 由 patch_fill 一并抹平。

    近白门限 = min(R,G,B) ≥ min_rgb 且 绿度 G−max(R,B) < green_gate：
      - 环带本身是中性的(实测 (130,130,130)) → 天然满足；
      - 挡掉「去完发光的暗光晕」——v2 减绿度后光晕 G 被减到 max(R,B) 附近,
        但 R/B 本底暗 → min_rgb 不够(多为 60~120 的暗棕/暗灰), 亮度也不够；
      - 挡掉强绿光晕(原始图上绿度 ≥green_gate 不进)与灰色大背景块。
    局部背景 = 蒙版外像素亮度 bg_lo 分位——光晕等亮区集中在高分位，低分位≈
    真实背景，阈值随背景自适应(暗背景自动放宽、亮背景自动收紧)。
    仅在去完发光的干净图上启用(如 v2 路径在并集蒙版上调用)：原始图的亮绿光晕
    会被绿度门放行一部分, 因此不在原图上跑。

    背景亮纹理门(ext_thr): 候选门是「比背景亮 + 近白 + 非绿」的纯亮度条件,
    非均匀背景上挡不住「字旁边恰好有一块更亮的石头/墙面纹理」——武器1787
    「器」右侧的石纹亮带(灰 117~134, 全图背景 p25=93)全门通过, 6 轮连通
    生长吞下 621px 非字形蒙版(亮带 + 右下亮斑串)。判据: 真字 AA 环带是
    「有限结构」——从生长结果出发沿候选区测地行走必在十来步内耗尽(668 软边
    环带实测 14 步走完); 背景亮场(石纹亮带/光斑串)比生长预算厚得多, ext_thr
    步仍走不完(武器1787 实测 >64 步、残留 2827px) —— 含「走不完候选」的
    候选连通块不是环带, 回退其全部新增。实测七图: 武器 621→54, 668 白块
    覆盖保持 62/63, 其余五图蒙版逐像素不变。
    """
    if not mask.any():
        return mask
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    outside = (mask == 0)
    bg = float(np.percentile(gray[outside], bg_lo)) if outside.sum() > 0 else 90.0
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    min_rgb_im = np.minimum(np.minimum(r, g), b)
    cand = ((gray > (bg + lum_off)) &
            (min_rgb_im >= min_rgb) &
            ((g - np.maximum(r, b)) < green_gate))
    if cand.any():
        cur = (mask > 0).astype(np.uint8)
        k3 = np.ones((3, 3), np.uint8)
        for _ in range(rounds):
            dil = cv2.dilate(cur, k3) > 0
            add = dil & cand & (cur == 0)
            if not add.any():
                break
            cur[add] = 1
        grown = np.where(cur > 0, 255, 0).astype(np.uint8)
        # 背景亮纹理门: 从生长结果沿候选区测地走 ext_thr 步, 若候选连通块里
        # 仍剩走不到的候选 → 该结构比环带厚 → 回退其全部新增。
        added = (grown > 0) & (mask == 0)
        leftover = cand & (cur == 0)
        if added.any() and leftover.any() and ext_thr > 0:
            reach = cur.copy()
            for _ in range(ext_thr):
                nxt = (cv2.dilate(reach, k3) > 0) & cand & (reach == 0)
                if not nxt.any():
                    break
                reach[nxt] = 1
            unreached = leftover & (reach == 0)
            if unreached.any():
                n, lab = cv2.connectedComponents(cand.astype(np.uint8), 8)
                bad = np.unique(lab[unreached])
                bad = bad[bad != 0]
                if bad.size:
                    grown[np.isin(lab, bad) & added] = 0
        return grown
    return mask


def _absorb_zone_bright_core(clean_rgb: np.ndarray, orig_rgb: np.ndarray,
                             mask: np.ndarray, zone: np.ndarray,
                             bg_off: int = 30, min_rgb_lo: int = 118,
                             green_gate: int = 26, max_cc_area: int = 200,
                             orig_green_min: int = 18,
                             dist_max: int = 18,
                             orig_gray_min: int = 150) -> np.ndarray:
    """发光区内亮核吸收（668「新」字蒙版覆盖不全修复）。

    现象：绿晕把文字的孤立小部件（离主笔画 >方案B生长半径，668 实测 9.6~18.8px）
    与主体隔开 —— DBNet 不框（Otsu 只在框内分割）、方案B 从蒙版只长 6px 够不着、
    背景重建的 detail=(原图−σ2)×近字衰减 又把它当「背景细节」保留 → 去字后残留
    8×6 白块。668 实测：clean 上 (R,G,B)≈(125,121,123)、gray≈122、绿度≈−4。

    判定（亮度/近白/绿度在**去发光图**上）：落在发光区(zone)内、亮度显著高于
    背景、近白、非绿的小连通块 → 并入蒙版交给 patch_fill 抹平：
      - zone 约束：浅色块/远处光斑不在发光区内，天然隔离（防越界吞背景）；
      - 亮度门 bg+bg_off：背景取 zone 外灰度 25 分位（暗背景自动放宽）；
      - 近白门 min_rgb≥min_rgb_lo + 绿度门 <green_gate：与方案B 同语义，
        只收「去发光后已变中性」的亮结构，排除残余绿晕；
      - **原图绿度门 ≥orig_green_min**：真「被绿晕包裹/染色的文字部件」在原图上
        带明显绿度（668 实测 +42）；而本来就中性的亮背景纹理（635 右上布纹，
        原图绿度仅 +3~15）与发光无关，不得误吞 —— 该门是二者的分界
        （18 时 635 误吞 54→7px，668 修复量不变）。
      - 面积门 ≤max_cc_area：漏检的是「笔画部件」级小块；整片亮背景
        （即使被 zone 外扩啃进一小条，也是大连通块）不会误吞。
    以连通块为单位整体并入（而非逐像素），保证部件的暗 AA 边一并进填充区。
    """
    if not mask.any() or zone is None or not zone.any():
        return mask
    cand_zone = (zone > 0) & (mask == 0)
    if not cand_zone.any():
        return mask
    # 距离约束: 文字的孤立部件离主蒙版不会太远(668 白块 9.6~18.8px)。
    # 防的是「zone 边缘啃进亮背景(如浅色块)时, 边缘小条全部满足像素门」的
    # 误吞 —— 668 前端默认参数(zone_expand=10)下, zone 上缘在浅色块里切出
    # 9x4 小条(y≈89, 距主蒙版 22px)被整条吸收, 混进文字蒙版。dist_max=18
    # 把它挡掉(实测顶部小点 26→0px, 白块吸收 56/63 不受影响)。
    if dist_max and dist_max > 0:
        dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        cand_zone &= (dist <= float(dist_max))
        if not cand_zone.any():
            return mask
    gray = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    outside = (mask == 0) & (zone == 0)
    bg = float(np.percentile(gray[outside], 25)) if outside.sum() else 90.0
    r = clean_rgb[..., 0].astype(np.int16)
    g = clean_rgb[..., 1].astype(np.int16)
    b = clean_rgb[..., 2].astype(np.int16)
    min_rgb = np.minimum(np.minimum(r, g), b)
    # 原图绿度: 文字部件被绿晕染色 → 原图 G−max(R,B) 明显; 中性亮背景 ≈0
    orr = orig_rgb[..., 0].astype(np.int16)
    og = orig_rgb[..., 1].astype(np.int16)
    ob = orig_rgb[..., 2].astype(np.int16)
    orig_green = og - np.maximum(orr, ob)
    # 原图亮度门: 文字部件在原图上就是白字亮核(668 白块 orig≈164、两横亮部
    # 150+); 而「色块分界台阶」类背景结构亮度中等 —— 635(hist6635)灰带底边
    # 小条(orig≈136, 距主蒙版仅6px, 距离门挡不住)被整条误吸收。orig≥150
    # 实测: 635 上方小块 17→0px, 668 白块/两横覆盖几乎不变。
    gorig = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cand = (cand_zone &
            (gray > (bg + bg_off)) &
            (min_rgb >= min_rgb_lo) &
            ((g - np.maximum(r, b)) < green_gate) &
            (orig_green >= orig_green_min) &
            (gorig >= orig_gray_min))
    if not cand.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    absorb = np.zeros(cand.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= max_cc_area:
            absorb |= (labels == i)
    if not absorb.any():
        return mask
    mask = mask.copy()
    mask[absorb] = 255
    return mask


def _grow_color_tint(rgb: np.ndarray, mask: np.ndarray,
                     red_thr: int = 30, green_thr: int = 15,
                     green_g: int = 100, rounds_max: int = 120,
                     max_grow_ratio: float = 5.0) -> np.ndarray:
    """沿「色偏像素」从文字蒙版出发八连通生长，吞并整片色偏覆盖区（wasm 实现）。

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

    0.3.0: 逐轮闭合由共享核 `grow_color_tint` 一次 wasm 调用完成（与浏览器
    逐位一致）；Python 逐轮循环已随 Python 核心一并删除。
    """
    if not mask.any():
        return mask
    return _sc_grow_color_tint(rgb, mask, red_thr, green_thr,
                               green_g, rounds_max, max_grow_ratio)


def detect_text_mask(raw, strength: float = 1.0, method: str = "ml",
                     min_area: int = 30, max_area_ratio: float = 0.05,
                     max_box_ratio: float = 0.40,
                     max_side: int = 960, work_max: int = 1280,
                     q_off: float = 50.0, tint_fill: bool = True,
                     fill_white: bool = True,
                     fill_max_dist: int = 12,
                     upscale: bool = True,
                     bright_bridge: bool = False):
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
        upscale=True(默认): 小字框(高度<48px)内先放大 2~3x 再 Otsu/连通域,
        找回低分辨率下被面积下限/长宽比门误删的细笔画(方案A)；
        bright_bridge=True: 沿「比背景亮+近白」像素从蒙版连通生长, 兜住被
        Otsu 切掉的白字细笔画段(_fill_bright_near_mask, 方案B, 默认关)；
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

    # 2) 精细字形蒙版(亮度+强边)。upscale: 小字框自动放大分割, 补低分辨率
    #    细笔画被 CC 下限误删的缺口(方案A)
    mask = _detect_text_mask_classic(rgb, boxes=boxes, strength=strength,
                                    min_area=min_area, q_off=q_off,
                                    upscale=upscale)
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
    # 4b) 白字亮侧连通补全(方案B): 沿「比背景亮+近白」像素从蒙版连通生长,
    #     找回低分辨率下被 Otsu 切掉的白字细笔画段。默认关(不改变现有方案);
    #     v2 等「先去发光再去字」路径在干净图上启用(亮光晕已减绿变暗, 不会被吞)
    if bright_bridge:
        mask = _fill_bright_near_mask(rgb, mask)
    mask = _clean_text_mask(mask, H, W, min_area=min(
        min_area, 8), max_area_ratio=0.9)
    return mask, _mask_to_boxes(mask)