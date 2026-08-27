"""
文字擦除: DBNet → mask 2px 椭圆膨胀 → patch_fill(sample_mask=整图-mask) → 出图

设计原则:
- 只做 patchmatch, 不做 TELEA / 颜色匹配. 后两者会破坏纹理造成"糊一团".
- mask 2px 膨胀是必须的: 吃掉字形抗锯齿边缘, 否则 SSD 比较时这些 AA
  像素会被当成"已知上下文"污染 patchmatch 结果 (Δmean 30→0.1).
- sample_mask = 整图 - 文字mask: 强制 patchmatch 不在文字区取样.

参数选择证据 (在 needExtractAndPatch.png 305x150 上):
  dil=0 → Δ=30  ghost=572
  dil=1 → Δ=5.6 ghost=29
  dil=2 → Δ=0.1 ghost=0   ← 选这个
  dil=3 → Δ=0.7 ghost=0
  dil=4 → Δ=2.7 ghost=0
"""
from __future__ import annotations
import time
import numpy as np
import cv2

from core.text_select import (detect_text_mask, _deglow_faint_green,
                              _deglow_faint_green_v11, _deglow_full_green)
from core.patch_fill import inpaint as pm_inpaint


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


def erase_text(
    rgb: np.ndarray,
    *,
    mask_pad: int = 2,
    q_off: float = 55.0,
    max_area_ratio: float = 0.40,
    max_box_ratio: float = 0.40,
    ml_max_side: int = 960,
    direction: float | None = None,
    edge_aware: bool = False,
    edge_extend: int = 1,
    return_mask: bool = False,
    tint_fill: bool = True,
    glow_mode: str = "auto",
    deglow_strength: float = 1.0,
    deglow_green_thr: float = 6.0,
    deglow_range: int = 24,
    deglow_glo: float = 85.0,
    deglow_protect: float = 1.0,
    deglow_mask_soft: float = 0.0,
    deglow_scheme: str = "channel",
):
    """最小化文字擦除管线.

    Args:
        rgb: HxWx3 uint8 RGB 图像
        mask_pad: mask 预外扩像素 (默认 2, 刚好吃掉字形 AA 边缘)
        q_off: 传给 detect_text_mask, [30,70], 越高 mask 越贴字形
        max_area_ratio: 传给 detect_text_mask, 给单字粘连大块放行
        ml_max_side: DBNet 推理尺度
        edge_aware: 默认 False. 历史版本用 ellipse(8) 试图吞 AA 边缘, 但会把
            mask 膨胀到占图 60%+, 反而让 patch_fill sample 区不够、产生更多
            白/红残留; 现在默认关掉, 只用 mask_pad=2 就足够.
        edge_extend: PS「移动边缘」等价物 —— 蒙版(展示)与**填充区域**同步
            外扩/收缩 (>0 外扩, 0 不变, <0 收缩)。默认 +1: 把字形最外层 AA
            过渡边整体包进红蒙版并真正填掉, 消除"边缘露白/露红"。
        return_mask: True 时返回 (result, mask, meta), 否则 (result, meta)
        tint_fill: True 时启用色偏区域生长(_grow_color_tint, 红蒙版叠加/淡绿光晕
            自动并入蒙版)。False 则不做色偏生长(发光仍可由 glow_mode 处理)。
        glow_mode: 发光处理策略（在「去发光方案=通道法」时生效）
            "auto" (默认)   — A+B 混合：强光晕并入蒙版填充(B) + 弱光晕边缘
                              通道法去发光(A)。
            "autov1.1"      — A+B 混合 + 范围补齐：与 auto 完全同语义, 仅把
                              A 路径的固定 near_r 圈换成「绿|比背景亮」连通生长,
                              淡绿渐隐边缘(超出固定圈的光晕)不再漏掉。
            "deglow_first"  — 实验性：先用通道法把整片绿光晕从图中去除(先去除
                              发光)，再在干净图上检测文字蒙版并填充(再去除文字)。
            "off"           — 完全关闭发光处理，行为回到发光改造前。
        deglow_strength: [0,1] 去发光力度。0=不去除发光颜色，1=完全去除(默认)。
            通道法与 v4 通用方案共用此语义。
        deglow_green_thr: 绿检出阈值(通道法 auto/autov1.1 弱区判定用,
            g − max(R,B) 超过该值才视为淡绿光晕)。默认 6(只处理明显偏绿);
            调小(如 3)可把更淡的绿边缘也拉净 → 减少绿色溢出; 调大则更保守。
        deglow_range: 去发光作用半径(px, 通道法 auto 的 A 路径 near_r, v1.1 的
            固定圈)。默认 24; 调大(如 40~60)覆盖更远的光晕残迹, 调小更收敛。
            对 v4 方案不生效。
        deglow_glo: 弱绿像素的最低亮度(G 通道下限 g_lo, 默认 85)。只有当
            绿像素亮度超过该值才被当作"光晕"处理; 溢出时调低(如 60~70)可把
            更暗的残绿也拉净, 调高则更保守。
        deglow_protect: 白字/文字边缘保护强度 [0,1], 默认 1(完全保护, 现有
            行为)。保护会跳过亮白像素(文字及 AA 边)不去发光, 防文字被"吹胖";
            若文字周围仍有残绿, 可降到 0.5~0 减少保护、多去绿(可能轻度伤文字
            AA 边)。
        deglow_mask_soft: 蒙版「透明度扩展」半径(px, 默认 0=关)。>0 时在真实
            填充区外扩展一圈软带：软带内缘完全填充、外缘回归原始(按距离衰减
            渐变混合) → 扩大覆盖范围的同时保留底层纹理不整块覆盖。红蒙版会以
            半透明显示该软带。适合"光晕没被吞进蒙版"的截图类图(如 deglow_first
            优于 auto 的场景)。
        deglow_scheme: 去发光方案（前端「去发光方案」下拉）
            "channel" (默认) — 现有通道法（绿光晕专用），由 glow_mode 控制流程。
            "v4"              — v4.1 通用方案(deglow 包)：自动辨识
                                  blend/additive/screen 三模式 + 反演/重建/溯源。
            "off"             — 关闭所有去发光（等价于 glow_mode="off"）。
    Returns:
        result: HxWx3 uint8 RGB 已擦除文字的图片
        mask:   HxW uint8 (255=将被填充的区域[mask_pad+移动边缘后])  仅当 return_mask=True
        meta:   dict with keys {mask_pix, inpaint_seconds, method}
    """
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    t0 = time.time()

    # 方案=off 或 v4 → 覆盖面优先级高于 glow_mode 细节
    if deglow_scheme == "off":
        glow_mode_eff = "off"
    elif deglow_scheme == "v4":
        return _erase_v4_deglow(
            rgb, mask_pad=mask_pad, q_off=q_off,
            max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
            ml_max_side=ml_max_side, direction=direction,
            edge_aware=edge_aware, edge_extend=edge_extend,
            return_mask=return_mask, deglow_strength=deglow_strength,
            glow_mode=glow_mode,
        )
    else:
        glow_mode_eff = glow_mode

    # 通道法发光旋钮: 绿检出阈值 / 作用范围(自动钳制到安全区间)
    gthr = float(max(0.5, min(deglow_green_thr, 20.0)))
    grange = int(max(0, min(deglow_range, 200)))
    glo = float(max(40.0, min(deglow_glo, 160.0)))
    gprot = float(max(0.0, min(deglow_protect, 1.0)))
    msoft = float(max(0.0, min(deglow_mask_soft, 150.0)))

    # auto v1.1: A+B 混合 + 范围补齐 + 弱区纹理保留
    if glow_mode_eff == "autov1.1":
        return _erase_auto_v11(
            rgb, mask_pad=mask_pad, q_off=q_off,
            max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
            ml_max_side=ml_max_side, direction=direction,
            edge_aware=edge_aware, edge_extend=edge_extend,
            return_mask=return_mask, deglow_strength=deglow_strength,
            tint_fill=tint_fill, deglow_green_thr=gthr, deglow_range=grange,
            deglow_glo=glo, deglow_protect=gprot, soft_expand=msoft,
        )

    # 实验性: 先去发光再去字
    if glow_mode_eff == "deglow_first":
        return _erase_deglow_first(
            rgb, mask_pad=mask_pad, q_off=q_off,
            max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
            ml_max_side=ml_max_side, direction=direction,
            edge_aware=edge_aware, edge_extend=edge_extend,
            return_mask=return_mask, deglow_strength=deglow_strength,
            soft_expand=msoft,
        )

    use_tint = tint_fill and (glow_mode_eff != "off")
    mask, boxes = detect_text_mask(
        rgb, method="ml", q_off=q_off,
        max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
        max_side=ml_max_side, tint_fill=use_tint,
    )

    if not mask.any():
        out = (rgb, mask, {"mask_pix": 0, "inpaint_seconds": 0.0,
                           "method": "ml", "boxes": []}) if return_mask \
              else (rgb, {"mask_pix": 0, "inpaint_seconds": 0.0,
                          "method": "ml", "boxes": []})
        return out

    # 0b. 弱绿光晕边缘通道法去发光(A)：在蒙版外围把淡绿渐隐边缘还原成背景，
    #     使最终填充结果不残留光晕。强光晕已由 detect_text_mask 并入蒙版填充。
    #     仅 glow_mode_eff != "off" 时启用；力度受 deglow_strength 控制。
    sample_exclude = None
    if glow_mode_eff != "off":
        rgb, _ = _deglow_faint_green(rgb, mask, thr=gthr, near_r=grange,
                                     g_lo=glo, text_protect=gprot,
                                     strength=deglow_strength)
        # 填充取样时排除残余绿(未完全去净的发光边缘)，防止把绿复制进文字区
        sample_exclude = _residual_green(rgb, mask)

    # 1~3. 共用填充步骤
    return _run_fill(rgb, mask, boxes, mask_pad=mask_pad, direction=direction,
                     edge_aware=edge_aware, edge_extend=edge_extend,
                     return_mask=return_mask, t0=t0, sample_exclude=sample_exclude,
                     soft_expand=msoft)


def _erase_auto_v11(rgb, *, mask_pad, q_off, max_area_ratio, max_box_ratio,
                    ml_max_side, direction, edge_aware, edge_extend,
                    return_mask, deglow_strength=1.0, tint_fill=True,
                    deglow_green_thr=6.0, deglow_range=24,
                    deglow_glo=85.0, deglow_protect=1.0,
                    soft_expand: float = 0.0):
    """auto v1.1(保守版)：A+B 混合 + 仅补光晕范围。

    与 auto 的差异全部收敛在 _deglow_faint_green_v11：
      仅把 A 路径的固定 near_r 圈换成「绿|比背景亮」连通生长 —— 淡绿渐隐
      边缘(超出固定圈的光晕)被完整覆盖，范围不再偏小。
      weak 判定/文字保护/背景拉平/取样剔除等语义与 auto 完全一致，
      不引入紫/杂色、不改变绿残留水平。
    路径顺序与 auto 一致：检测基础文字蒙版(不含色偏生长) → v1.1 去发光
    (强区并入蒙版) → 填充。
    """
    t0 = time.time()

    # 1) 基础文字蒙版(色偏生长由 _deglow_faint_green_v11 按需并入)
    tmask, boxes = detect_text_mask(
        rgb, method="ml", q_off=q_off,
        max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
        max_side=ml_max_side, tint_fill=False,
    )
    if not tmask.any():
        meta = {"mask_pix": 0, "mask_filled_pix": 0, "inpaint_seconds": 0.0,
                "method": "ml", "boxes": []}
        return (rgb, tmask, meta) if return_mask else (rgb, meta)

    # 2) v1.1 去发光：返回(去发光图, 强区并入蒙版, 统计)
    clean, add_mask, dstats = _deglow_faint_green_v11(
        rgb, tmask, strength=deglow_strength, tint_fill=tint_fill,
        thr=deglow_green_thr, near_r=deglow_range,
        g_lo=deglow_glo, text_protect=deglow_protect)

    # 3) 填充蒙版 = 基础蒙版 ∪ 强光晕区；取样剔除残余绿
    mask = tmask
    if add_mask.any():
        mask = cv2.bitwise_or(mask, add_mask)
    sample_exclude = _residual_green(clean, mask)

    # 4~6. 共用填充步骤
    if return_mask:
        result, mask_filled, meta = _run_fill(
            clean, mask, boxes, mask_pad=mask_pad, direction=direction,
            edge_aware=edge_aware, edge_extend=edge_extend,
            return_mask=True, t0=t0, sample_exclude=sample_exclude,
            soft_expand=soft_expand)
    else:
        result, meta = _run_fill(
            clean, mask, boxes, mask_pad=mask_pad, direction=direction,
            edge_aware=edge_aware, edge_extend=edge_extend,
            return_mask=False, t0=t0, sample_exclude=sample_exclude,
            soft_expand=soft_expand)
    meta["deglow_v11"] = dstats
    return (result, mask_filled, meta) if return_mask else (result, meta)


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


def _run_fill(rgb, mask, boxes, *, mask_pad, direction, edge_aware,
              edge_extend, return_mask, t0, sample_exclude=None,
              soft_expand: float = 0.0):
    """共用填充步骤：膨胀 mask → sample → patch_fill → meta。

    sample_exclude: 可选 bool mask(HxW)，这些像素从取样区剔除(不参与复制)。
        用于发光处理时剔除残余绿色像素，避免填充把绿复制进文字区(泛绿/尿渍)。
    soft_expand: 「透明度扩展」半径(px, 0=关)。>0 时在真实填充区(mask_filled)
        外围扩展一圈软带：软带内用原始图(已去发光)与填充结果按距离衰减渐变
        混合 —— 内缘贴近核心(完全填充)、外缘回归原始 → 扩大覆盖范围的同时
        保留底层纹理不被整块覆盖。软带会被红蒙版以半透明显示(透明度扩展)。
    """
    # 1. mask 2px 椭圆膨胀: 吃掉 AA 边缘, 否则 patchmatch 会把 AA 像素当上下文
    if mask_pad > 0:
        mask_filled = cv2.dilate(mask, _ellipse(mask_pad))
    else:
        mask_filled = mask.copy()

    # 1b. 边缘感知扩张: 把"紧邻 mask 且亮度落在文字带"的原图像素也并入填充区,
    #     专门吃掉抗锯齿白边(白字) / 浅色描边, 又不会像大膨胀那样产生 halo。
    if edge_aware:
        mask_filled = _edge_aware_grow(rgb, mask_filled)

    # 1c. PS「移动边缘」(edge_extend): 让**实际填充区域**也随之外扩/收缩,
    #     与展示红蒙版保持一致 —— 用户外扩时, 露白的外围会被真正填掉。
    if edge_extend > 0:
        mask_filled = cv2.dilate(mask_filled, _ellipse(edge_extend))
    elif edge_extend < 0:
        mask_filled = cv2.erode(mask_filled, _ellipse(-edge_extend))

    # 2. sample_mask: 只在文字外取样
    sample_mask = (255 - mask_filled).astype(np.uint8)
    if sample_exclude is not None:
        sample_mask[sample_exclude] = 0

    # 3. patch_fill (单步); direction 非空时启用方向填充
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
    if return_mask:
        # 展示蒙版 = 真实填充区(mask_pad 膨胀 + 移动边缘后) —— 所见即所得
        return result, mask_filled, meta
    return result, meta


def _erase_deglow_first(rgb, *, mask_pad, q_off, max_area_ratio, max_box_ratio,
                        ml_max_side, direction, edge_aware, edge_extend,
                        return_mask, deglow_strength=1.0,
                        soft_expand: float = 0.0):
    """实验性 glow_mode="deglow_first"：先去发光，再走普通去文字路径。

    路径顺序与普通模式完全一致，只是在最前面插入一步「去发光」：
      1) 原图定位文字蒙版(仅用于保护文字不被去发光误伤，不做色偏生长)；
      2) 去发光：把文字外围的整片绿光晕拉向背景色，文字笔画原样保留；
      3) 在干净图上走**普通去文字路径**：detect_text_mask(tint_fill=True)
         → 膨胀 → 填充。残余绿由 tint_fill 并入蒙版 + 取样剔除兜底。
    """
    t0 = time.time()

    # 1) 定位文字(保护用)
    tmask, _ = detect_text_mask(
        rgb, method="ml", q_off=q_off,
        max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
        max_side=ml_max_side, tint_fill=False,
    )
    if not tmask.any():
        meta = {"mask_pix": 0, "mask_filled_pix": 0, "inpaint_seconds": 0.0,
                "method": "ml", "boxes": []}
        return (rgb, tmask, meta) if return_mask else (rgb, meta)

    # 2) 先去发光(文字保护, 去除外围绿光晕)
    clean, _ = _deglow_full_green(rgb, tmask, strength=deglow_strength)

    # 3) 普通去文字路径(干净图上): 检测蒙版(自动并入残余绿) → 填充
    mask, boxes = detect_text_mask(
        clean, method="ml", q_off=q_off,
        max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
        max_side=ml_max_side, tint_fill=True,
    )
    if not mask.any():
        meta = {"mask_pix": 0, "mask_filled_pix": 0, "inpaint_seconds": 0.0,
                "method": "ml", "boxes": [], "deglow_img": clean}
        return (clean, mask, meta) if return_mask else (clean, meta)

    # 填充取样剔除残余绿, 防复制进文字区
    sample_exclude = _residual_green(clean, mask)
    res = _run_fill(clean, mask, boxes, mask_pad=mask_pad, direction=direction,
                    edge_aware=edge_aware, edge_extend=edge_extend,
                    return_mask=return_mask, t0=t0, sample_exclude=sample_exclude,
                    soft_expand=soft_expand)
    if return_mask:
        result, mask_filled, meta = res
    else:
        result, meta = res
    # 去发光中间图供前端展示「去除发光后的全图」
    meta["deglow_img"] = clean
    return (result, mask_filled, meta) if return_mask else (result, meta)


def _erase_v4_deglow(rgb, *, mask_pad, q_off, max_area_ratio, max_box_ratio,
                     ml_max_side, direction, edge_aware, edge_extend,
                     return_mask, deglow_strength=1.0, glow_mode="auto"):
    """v4.1 通用去发光方案：先去发光（deglow 包，三模式辨识+反演+重建），
    再原样走普通去文字路径（detect_text_mask → 膨胀 → patch_fill）。

    严格遵循项目约束「发光去除路径：先去发光 → 原样走普通去文字路径」。
    """
    t0 = time.time()

    # 1) 定位文字（保护用，传给管线作为载体；tint 关闭避免把光晕并入蒙版）
    tmask, _ = detect_text_mask(
        rgb, method="ml", q_off=q_off,
        max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
        max_side=ml_max_side, tint_fill=False,
    )

    # 2) 通用去发光（v4 管线；strength=0 即纯保护路径）
    from deglow import pipeline as v4_pipeline
    res = v4_pipeline.run(rgb, carrier_mask=tmask,
                          deglow_strength=deglow_strength)
    clean = np.clip(res.image, 0, 255).astype(np.uint8)
    if not res.has_glow:
        clean = rgb
    # 3) 普通去文字路径（干净图上）
    mask, boxes = detect_text_mask(
        clean, method="ml", q_off=q_off,
        max_area_ratio=max_area_ratio, max_box_ratio=max_box_ratio,
        max_side=ml_max_side, tint_fill=True,
    )
    if not mask.any():
        meta = {"mask_pix": 0, "mask_filled_pix": 0, "inpaint_seconds": 0.0,
                "method": "ml", "boxes": [], "deglow_img": clean}
        if res.has_glow:
            meta["dglow_report"] = res.report
        return (clean, mask, meta) if return_mask else (clean, meta)

    # 填充取样剔除残余绿（未净发光边缘），防复制进文字区
    sample_exclude = _residual_green(clean, mask)
    fill = _run_fill(clean, mask, boxes, mask_pad=mask_pad,
                     direction=direction, edge_aware=edge_aware,
                     edge_extend=edge_extend, return_mask=return_mask,
                     t0=t0, sample_exclude=sample_exclude)
    if return_mask:
        result, mask_filled, meta = fill
    else:
        result, meta = fill
    meta["deglow_img"] = clean
    if res.has_glow:
        # 结构化报告 JSON 化（含每域 mode/α 分位数/三态计数/dye/σ̂_g）
        meta["dglow_report"] = jsonify_report(res.report)
    return (result, mask_filled, meta) if return_mask else (result, meta)


def jsonify_report(rep: dict) -> dict:
    """把 report 中的 numpy 标量转成原生 JSON 类型。"""
    import json

    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, (np.ndarray,)):
            return v.tolist()
        return v

    return json.loads(json.dumps(conv(rep), ensure_ascii=False))