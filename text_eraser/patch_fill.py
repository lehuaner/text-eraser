"""
内容识别填充（Content-Aware Fill）—— PatchMatch 范例式修复。

与 PS「内容识别填充」同族（Criminisi 优先级 + Barnes PatchMatch 加速）：
  1. 填充前沿优先级 = 置信度 × 数据项(边缘梯度)。置信度用「块内已知像素占比」近似，
     因此越靠近已知边界、结构越强(高梯度)的像素越先填 —— 衣领、缝线、花纹等结构
     能连续延续，而不是像旧实现那样只从外往里机械拷贝单像素。
  2. 对当前最高优先级像素的目标块，在「已知区域」中用 PatchMatch 思路找最相似源块：
     随机候选 + 邻域相干(复用已填像素的源中心)的亚线性搜索。支持 sample_mask 限定
     取样区域(对应 PS 的 Custom Sampling Area / 手动绘制参考区)。
  3. 把整块源纹理拷贝进洞(而非单像素)，并做轻量「颜色自适应」(均值/方差对齐局部
     上下文)消除接缝 —— 这是旧弱版 patch_fill(只搬中心像素)没做到的，也是 PS 效果关键。
  4. 更新已知图并迭代，直到洞填满或被取消(should_cancel)。
  5. 残余 1~2px 边界用 TELEA 兜底柔化。

性能：源候选限制在「洞 bbox + 局部边距」(或 sample_mask bbox) 的 ROI 内，配合 PatchMatch
的亚线性搜索，普通衣物文字区域(数百 px)可在毫秒~秒级完成；全程可中断。

复用（不造轮子）：
  - 沿用旧 patch_fill 的「局部 ROI 裁剪 / distanceTransform 思路 / TELEA 兜底」结构；
  - 签名保持 inpaint(rgb, mask) 与旧接口一致，仅新增 sample_mask / should_cancel 可选参数，
    因此 extractor.py 调度处只需一行调用即可接入，无需改动其它算法。
"""
from __future__ import annotations

import numpy as np
# Shared-algorithm-core cv2 shim: routes dilate/erode/morphologyEx/connectedComponents/
# cvtColor(RGB2GRAY) through textcore.wasm (same operators the browser runs) and falls
# through to the real cv2 for everything else. Keeps the backend + browser parity.
from text_eraser import _cv as cv2
# Shared PatchMatch fill: when the wasm core is available, the fill loop runs the SAME
# Rust implementation the browser Worker runs → frontend/backend byte-identical fills.
from text_eraser._shared_core import patchmatch_inpaint_fill, using_shared_core


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
    内容识别填充（PatchMatch 范例式修复）。

    image_rgb    : HxWx3 uint8 / float 图像
    mask         : HxW uint8/bool，>0 为待填充(去文字)区域
    sample_mask  : 可选 HxW uint8/bool，>0 为「参考/取样区域」(PS 自定义取样区)。
                   提供时**只从该区域内**取样源块（常见：整图减字形 → 即"边缘之外
                   非文字部分"全部可用作参考）；为 None 时自动取洞周围局部纹理。
                   关键：sample_mask **不会扩大** ROI（避免传入"整图-字形"时 ROI
                   退化为整图→触发 TELEA）；ROI 仍按 mask bbox + 边距算，
                   sample_mask 只用来在 ROI 内**过滤候选源像素**。
    direction    : 可选 float(角度°, 图像坐标 0°=+x右 / 90°=+y下)。
                   提供时进入**方向填充模式**：每个目标块的源候选被限制在
                   "过该目标点、沿 direction 的一条直线"上双向采样；去掉随机远距
                   候选(只保留邻域相干)，于是填充像素只借"这条线"上的内容，
                   不会把别处的纹理搬进来 —— 适合有主导纹理方向的图像
                   (木纹/岩石条带/布料织纹)，让填充沿 60° 之类方向平滑延展。
                   不提供(默认 None)时维持原 PatchMatch 行为。
    should_cancel: 可选零参 callable，返回 True 时尽快中断(返回当前已填结果)。
    flat_span / flat_tex : 平滑渐变背景自适应门。洞外四边环带的中位亮度极差
                   ≥flat_span **且** 环带梯度中位 <flat_tex 时，判定背景为
                   「强亮度渐变 + 无纹理可复制」(如换装.png 的雾面金色光效)：
                   PatchMatch 的 7×7 块直拷无法维持渐变的亮度连续性 —— 相邻
                   块选中不同色调的源、同圈互相冲突、下一圈又以冲突像素为锚，
                   碎块伪影随填充内传放大(实测换装.png 文字区黑碎块)。这种
                   背景本身没有纹理可保，扩散插值(TELEA)的连续性更优 → 直接
                   TELEA。纹理背景(布纹/岩石)或均匀背景不受影响。
    return       : HxWx3 uint8
    """
    img = np.ascontiguousarray(image_rgb[..., :3], dtype=np.float32)
    OH, OW = img.shape[:2]
    m = (np.asarray(mask) > 0)
    if not m.any():
        return img.astype(np.uint8).copy()

    # ---- 平滑渐变背景检测(见 flat_span/flat_tex 说明) ----
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
        span = float(np.max(edges_med) - np.min(edges_med))
        gx0 = cv2.Sobel(gray0, cv2.CV_32F, 1, 0, ksize=3)
        gy0 = cv2.Sobel(gray0, cv2.CV_32F, 0, 1, ksize=3)
        grad0 = np.sqrt(gx0 ** 2 + gy0 ** 2)
        ring0 = (cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~m
        tex = float(np.median(grad0[ring0])) if ring0.any() else 0.0
        # 杂色全面修复(1788077005814): 此前要求 span>=40 + tex<15 才走 TELEA,
        # 漏掉了"局部均匀/平滑渐变"(文字完全嵌在光滑背景里, 环带四边中位
        # 几乎相等 span≈0, tex<15)—— 这类图正是杂色高发区: patchmatch 7x7
        # 直拷无法维持渐变连续, 产出 flat+色差块(填区 chroma_std≈2.4 vs
        # 背景 0.76), 形成肉眼暗斑。环带纹理低(tex<flat_tex)即视为"可扩散"
        # 背景, TELEA 把局部梯度平滑插值进洞, 保留渐变(纹理)同时消除杂色。
        # 既不是"假设纯色"(梯度仍被插值)也不是"模糊"(PDE 修复, 沿结构传播)。
        # 仍保留 n>=2 边带检查(避免极小 mask 边带不足时误触发); span 检查
        # 移除——对真正光滑背景(span≈0)同样适用。纹理背景(tex>=flat_tex)
        # 仍走 patchmatch, 不损失纹理。
        # 平滑背景(环带纹理低 tex<flat_tex)一律用 cv2 TELEA 扩散填充——这与浏览器
        # (patchmatch.js 走 opencv.js INPAINT_TELEA) 和 Python 核心(无 core 回退) 行为
        # 一致, 是「三端一致」的 host 侧判定。Rust 共享核只负责非平滑纹理区的 PatchMatch
        # 填充(下方 using_shared_core 分支 / 浏览器的 patchmatchInpaintShared)。此前此处
        # 对 wasm 模式误跳过 TELEA、改走 Rust PatchMatch, 导致后端 wasm 与 Python 核心/
        # 浏览器在平滑区填充内容不一致。
        if tex < flat_tex:
            out = cv2.inpaint(np.clip(img, 0, 255).astype(np.uint8),
                              m.astype(np.uint8), 3, cv2.INPAINT_TELEA)
            return out

    # 安全内边距：文字贴图像边缘时，ROI 内的 PxP 块切片会越界(历史崩溃点)。
    # 把整图四周复制扩展 padm 像素，ROI 计算用 padding 后尺寸(坐标相对不变)，
    # 最终输出时裁剪回原尺寸。handbrush sample_mask 仍按原始尺寸栅格化。
    padm = 4
    img = cv2.copyMakeBorder(img, padm, padm, padm, padm, cv2.BORDER_REPLICATE)
    m = np.pad(m, padm, constant_values=False)
    H, W = img.shape[:2]

    # 手工参考区域：限定从哪些已知像素取样(PS 自定义取样区)。
    # 支持两种传入：① HxW 数组(>0 为取样区)；② 前端画笔笔画结构
    #   {"brush":半径(px),"strokes":[[[x,y],...],...]}（图像坐标），自动栅格化。
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

    # ROI 上限保护：超限不回退 TELEA（避免"刷子感"），而是逐步缩小边距重算——
    # 这样大文字仍保留 PatchMatch 的纹理连续性，只是参考圈略小。
    MAX_ROI = 1536
    while max(y1 - y0, x1 - x0) > MAX_ROI and margin > 24:
        margin = int(margin * 0.85)
        y0 = max(0, hy0 - margin); y1 = min(H, hy1 + margin)
        x0 = max(0, hx0 - margin); x1 = min(W, hx1 + margin)

    sub = img[y0:y1, x0:x1].copy()
    subm = m[y0:y1, x0:x1].copy()
    sh, sw = sub.shape[:2]
    subsm = sm[y0:y1, x0:x1] if sm is not None else None

    # ---- shared-core fill (single source of truth, identical to the browser) ----
    # Replaces the numpy PatchMatch loop below with the SAME Rust implementation the
    # browser Worker calls. Smooth-gradient TELEA (above) and any residual cleanup
    # still use cv2, which is byte-identical across platforms.
    if using_shared_core():
        _deg = direction if direction is not None else -1.0
        _filled = patchmatch_inpaint_fill(
            sub, subm, subsm, 7, _deg, 0)
        if _filled is not None:
            img[y0:y1, x0:x1] = np.clip(_filled, 0, 255)
            return np.clip(img, 0, 255)[padm:padm + OH, padm:padm + OW].astype(np.uint8)

    P = 7                        # 块大小(奇数)
    half = P // 2
    known = ~subm               # 已知区域(随填充扩张)
    orig_known = known.copy()   # 真·已知(原图纹理)快照：颜色自适应只锚定它，不随填充扩张
    hole = subm.copy()

    # 候选源块中心：块完全落在已知区(腐蚀保证)，且整块在 ROI 内(显式剔除边距带，
    # 避免 erode 的边界处理把贴边行也判为候选——那种位置 P×P 块会越界)。
    kpad = np.ones((P, P), np.uint8)
    cand_mask = cv2.erode(known.astype(np.uint8), kpad, iterations=1).astype(bool)
    cand_mask[:half] = False
    cand_mask[sh - half:] = False
    cand_mask[:, :half] = False
    cand_mask[:, sw - half:] = False
    if subsm is not None:
        cand_mask = cand_mask & subsm
    cand_y, cand_x = np.where(cand_mask)
    if cand_y.size == 0:
        out = cv2.inpaint(img.astype(np.uint8), m.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        return out[padm:padm + OH, padm:padm + OW]

    # ---- 方向填充模式预计算 ----
    # direction 非 None 时，_best_source 只沿过目标点的该角度直线取源候选，
    # 因此填充像素只借"这条线"上的内容，不会把别处纹理搬进来。
    dir_vec = None
    if direction is not None:
        rad = float(direction) * np.pi / 180.0
        dir_ux = np.cos(rad)
        dir_uy = np.sin(rad)
        dir_maxd = int(np.hypot(sh, sw)) + 1
        dir_step = 2
        dir_vec = (dir_ux, dir_uy, dir_maxd, dir_step)

    # 数据项：已知区 Sobel 梯度幅值，洞边界取其已知邻域最大梯度(结构优先)
    gray = cv2.cvtColor(np.clip(sub, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    ones3 = np.ones((3, 3), np.uint8)
    Dmap = cv2.dilate((grad * known.astype(np.float32)), ones3)

    # NNF(每个像素记录最佳源中心) + 相干标记
    nnf_y = np.zeros((sh, sw), np.int32)
    nnf_x = np.zeros((sh, sw), np.int32)
    nnf_set = np.zeros((sh, sw), bool)

    filled = sub.copy()
    rng = np.random.default_rng(0)
    K = min(256, max(32, cand_y.size // 4))   # 每步随机候选数(随候选规模自适应)

    dy = np.arange(-half, half + 1)
    dx = np.arange(-half, half + 1)
    dy4 = np.array([-1, 1, 0, 0])
    dx4 = np.array([0, 0, -1, 1])

    def _best_source(ty, tx):
        """在当前目标块中心(ty,tx)用 PatchMatch 思路找最相似源块中心。"""
        wy0, wy1 = ty - half, ty + half + 1
        wx0, wx1 = tx - half, tx + half + 1
        tpatch = filled[wy0:wy1, wx0:wx1]
        tknown = known[wy0:wy1, wx0:wx1]

        if dir_vec is not None:
            # 方向模式：候选 = 过该点的 direction 直线上的已知像素(双向采样)
            ux, uy, maxd, step = dir_vec
            ds = np.arange(step, maxd + 1, step)
            line_y, line_x = [], []
            for sign in (1, -1):
                cy = np.round(ty + sign * ds * uy).astype(np.int64)
                cx = np.round(tx + sign * ds * ux).astype(np.int64)
                # 同非方向模式：候选必须距 ROI 边界 ≥ half，保证源块 P×P 切片不越界
                inside = (cy >= half) & (cy < sh - half) & (cx >= half) & (cx < sw - half)
                cy, cx = cy[inside], cx[inside]
                ok = subsm[cy, cx] if subsm is not None else known[cy, cx]
                sel = np.where(ok)[0]
                line_y.append(cy[sel]); line_x.append(cx[sel])
            pool_y = np.concatenate(line_y) if line_y else np.empty((0,), np.int64)
            pool_x = np.concatenate(line_x) if line_x else np.empty((0,), np.int64)
            # 整条线都还没已知点(极端情况)：退回随机候选
            if pool_y.size == 0:
                ridx = rng.integers(0, cand_y.size, size=K)
                pool_y = cand_y[ridx].astype(np.int64)
                pool_x = cand_x[ridx].astype(np.int64)
        else:
            # 原 PatchMatch：随机候选
            ridx = rng.integers(0, cand_y.size, size=K)
            pool_y = list(cand_y[ridx]); pool_x = list(cand_x[ridx])

        # 邻域相干：已填四邻域的源中心(结构延续)。方向模式下也保留，
        # 因为相邻像素的线彼此平行，其源也在平行线上 → 连续。
        # 注意：循环变量用 ndy/ndx，避免遮蔽全局 dy/dx(numpy 数组)——
        # 否则在结尾 slice 时会变成 int 触发崩溃。
        nb_y = []; nb_x = []
        for ndy, ndx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = ty + ndy, tx + ndx
            if 0 <= ny < sh and 0 <= nx < sw and nnf_set[ny, nx]:
                nb_y.append(nnf_y[ny, nx]); nb_x.append(nnf_x[ny, nx])
        if nb_y:
            pool_y = np.concatenate([pool_y, np.asarray(nb_y, np.int64)])
            pool_x = np.concatenate([pool_x, np.asarray(nb_x, np.int64)])

        pool_y = np.asarray(pool_y, np.int64)
        pool_x = np.asarray(pool_x, np.int64)
        # 取出候选源块 (K,P,P,C)：每个候选中心的完整 P×P 邻域
        yy = np.clip(pool_y[:, None, None] + dy[None, :, None], 0, sh - 1)
        xx = np.clip(pool_x[:, None, None] + dx[None, None, :], 0, sw - 1)
        src = filled[yy, xx]
        # 只在目标块「已知位置」上比较(洞位置不参与 SSD)
        diff = (src - tpatch[None]) * tknown[..., None]
        ssd = np.einsum('kpqc,kpqc->k', diff, diff)
        bi = int(np.argmin(ssd))
        return int(pool_y[bi]), int(pool_x[bi])

    def _copy_patch(ty, tx, sy, sx):
        """把源块纹理(经局部颜色自适应)直接拷贝进目标块内的洞像素，更新已知图。

        设计要点：
          - 局部颜色自适应：把源块均值/方差对齐到目标块内「原图已知像素」的
            均值/方差，消除块与块/块与背景之间的颜色接缝；锚点是 orig_known
            快照(不随填充扩张)，避免每层都向"已被填过的上下文"重新对齐、
            把纹理对比越压越平。
          - 不做重叠 0.5 平均：任何相邻块的混合都会把两块不同纹理按 0.5 叠加
            → 方差几何衰减 → 涂抹/糊成一团(这是历史版本"模糊"的根因)。
            块级直拷 + sample_mask(只在同图连续纹理取样) + 邻域相干 已足以
            给出无可见接缝的填充，颜色自适应进一步消色差。
        """
        wy0, wy1 = ty - half, ty + half + 1
        wx0, wx1 = tx - half, tx + half + 1
        tpatch = filled[wy0:wy1, wx0:wx1]
        src = filled[sy - half:sy + half + 1, sx - half:sx + half + 1].astype(np.float32)
        # 局部颜色自适应：锚定 orig_known 内 ≥8 个真纹理像素时启用。
        # 锚窗逐级扩大(7x7 → 11x11 → 17x17): 锯齿/凹角处的边界块在 7x7 内
        # 已知像素常 <8, 自适应被跳过 → 源块**原样直拷**。双色块背景图
        # (1787980309628: 字上方浅色块 126~155 全在取样池)里, 弱约束边界块
        # 选中亮灰源块, 直拷后在深色区呈现为「小白块」(实测全部贴蒙版边界)。
        # 扩锚窗后任何边界块都对齐本地真背景的均值/方差(方差对齐保留纹理
        # 对比度, 不是模糊), 亮灰源块被拉回本地色调。
        ta = orig_known[wy0:wy1, wx0:wx1]
        tv = tpatch[ta]
        if int(ta.sum()) < 8:
            for r in (5, 8):
                by0, by1 = max(0, ty - r), min(sh, ty + r + 1)
                bx0, bx1 = max(0, tx - r), min(sw, tx + r + 1)
                ta2 = orig_known[by0:by1, bx0:bx1]
                if int(ta2.sum()) >= 8:
                    tv = filled[by0:by1, bx0:bx1][ta2]
                    break
        if len(tv) >= 8:
            tmean = tv.mean(0)
            tstd = tv.std(0) + 1e-3
            smean = src.reshape(-1, 3).mean(0)
            sstd = src.reshape(-1, 3).std(0) + 1e-3
            src = (src - smean) * (tstd / sstd) + tmean
        win = hole[wy0:wy1, wx0:wx1]
        view = filled[wy0:wy1, wx0:wx1]
        view[win] = src[win]
        known[wy0:wy1, wx0:wx1][win] = True
        hole[wy0:wy1, wx0:wx1][win] = False

    if dir_vec is None:
        # ---- 快速路径：批量边界填充 ----
        # 一次处理一圈边界(按 Criminisi 优先级排序)的 CHUNK 个像素，向量化
        # best_source(随机候选+邻域相干一起 gather+SSD)，把每像素一次循环
        # 的开销摊到批量上，大蒙版(数万 px)整体约 3 倍提速。
        CHUNK = 512
        while True:
            if should_cancel is not None and should_cancel():
                break
            boundary = hole & ~cv2.erode(hole.astype(np.uint8), ones3).astype(bool)
            if not boundary.any():
                break
            Cmap = cv2.boxFilter(known.astype(np.uint8), -1, (P, P),
                                 normalize=False).astype(np.float32) / (P * P)
            priority = Cmap * Dmap
            priority[~boundary] = -1.0
            b_y, b_x = np.where(boundary)
            order = np.argsort(-priority[b_y, b_x])
            b_y = b_y[order]; b_x = b_x[order]
            for c0 in range(0, len(b_y), CHUNK):
                c1 = min(c0 + CHUNK, len(b_y))
                cy, cx = b_y[c0:c1], b_x[c0:c1]
                # 批量随机候选
                n = len(cy)
                ridx = rng.integers(0, cand_y.size, size=(n, K))
                pool_y = cand_y[ridx].astype(np.int64)   # (n,K)
                pool_x = cand_x[ridx].astype(np.int64)
                # 邻域相干：有效邻居的 nnf 源中心
                ny = cy[:, None] + dy4[None, :]
                nx = cx[:, None] + dx4[None, :]
                valid = (ny >= 0) & (ny < sh) & (nx >= 0) & (nx < sw)
                cyy = np.clip(ny, 0, sh - 1); cxx = np.clip(nx, 0, sw - 1)
                nset = valid & nnf_set[cyy, cxx]
                if nset.any():
                    neb = [[nnf_y[cyy[i][nset[i]], cxx[i][nset[i]]],
                            nnf_x[cyy[i][nset[i]], cxx[i][nset[i]]]] for i in range(n)]
                    maxn = max((len(x[0]) for x in neb), default=0)
                    if maxn > 0:
                        # 用合法候选 cand_y[0]/cand_x[0] 填充空缺位，避免 0,0 越界
                        py_e = np.full((n, maxn), cand_y[0], np.int64)
                        px_e = np.full((n, maxn), cand_x[0], np.int64)
                        for i in range(n):
                            k = len(neb[i][0])
                            if k:
                                py_e[i, :k] = neb[i][0]; px_e[i, :k] = neb[i][1]
                        pool_y = np.concatenate([pool_y, py_e], 1)
                        pool_x = np.concatenate([pool_x, px_e], 1)
                # gather 候选源块 (n,Kc,P,P,C)
                yy = np.clip(pool_y[:, :, None, None] + dy[None, None, :, None], 0, sh - 1)
                xx = np.clip(pool_x[:, :, None, None] + dx[None, None, :, None], 0, sw - 1)
                src = filled[yy, xx]
                tyy = np.clip(cy[:, None, None] + dy[None, :, None], 0, sh - 1)
                txx = np.clip(cx[:, None, None] + dx[None, :, None], 0, sw - 1)
                tpatch = filled[tyy, txx]                       # (n,P,P,C)
                tkn = known[tyy, txx]                           # (n,P,P)
                diff = (src - tpatch[:, None]) * tkn[:, None, ..., None]
                ssd = np.einsum('nkpqc,nkpqc->nk', diff, diff)
                # 均值兼容惩罚: 弱约束边界块(已知像素少)的 SSD 区分度低, 异色区
                # 亮源块能凭几个已知像素胜出 → 填充边界白块(1787980309628)。
                # 把「源块均值 vs 目标块已知均值」的差按已知像素数记入 SSD ——
                # 相当于假设全部块像素都该有此量级的差; 合法结构延续的均值
                # 本来就与局部相近, 罚分可忽略。
                tkn_sum = np.clip(tkn.sum((1, 2)), 1.0, None)
                tmean = (tpatch * tkn[..., None]).sum((1, 2)) / tkn_sum[..., None]
                smean = src.mean((2, 3))                       # (n,K,C)
                ssd = ssd + 4.0 * tkn_sum[:, None] * \
                    ((smean - tmean[:, None, :]) ** 2).sum(-1)
                bi = np.argmin(ssd, 1)
                sy = pool_y[np.arange(n), bi]
                sx = pool_x[np.arange(n), bi]
                for i in range(n):
                    _copy_patch(int(cy[i]), int(cx[i]), int(sy[i]), int(sx[i]))
                    nnf_y[cy[i], cx[i]], nnf_x[cy[i], cx[i]] = sy[i], sx[i]
                    nnf_set[cy[i], cx[i]] = True
    else:
        # ---- 方向填充模式：逐像素(每目标沿 direction 直线取样) ----
        while True:
            if should_cancel is not None and should_cancel():
                break
            boundary = hole & ~cv2.erode(hole.astype(np.uint8), ones3).astype(bool)
            if not boundary.any():
                break
            # 置信度 ≈ 块内已知像素占比(随填充扩张，自然内→外)
            Cmap = cv2.boxFilter(known.astype(np.uint8), -1, (P, P),
                                 normalize=False).astype(np.float32) / (P * P)
            priority = Cmap * Dmap
            priority[~boundary] = -1.0
            ty, tx = np.unravel_index(int(np.argmax(priority)), priority.shape)
            sy, sx = _best_source(ty, tx)
            _copy_patch(ty, tx, sy, sx)
            nnf_y[ty, tx], nnf_x[ty, tx] = sy, sx
            nnf_set[ty, tx] = True

    # 残余极小洞：TELEA 兜底柔化
    if hole.any():
        sub8 = np.clip(filled, 0, 255).astype(np.uint8)
        filled = cv2.inpaint(sub8, hole.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    img[y0:y1, x0:x1] = np.clip(filled, 0, 255)
    return np.clip(img, 0, 255)[padm:padm + OH, padm:padm + OW].astype(np.uint8)
