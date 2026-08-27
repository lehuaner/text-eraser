"""v4.1 通用去发光 · 数据结构与枚举（对应规格 §2.2）。

内部约定：全部 float32、值域 [0,255]³，仅在最终输出处取整；预警记号：
  - P      观测图  HxWx3
  - B̂_global 全局背景场（仅用于方向/种子/初判，禁止进入幅值计算）
  - B̂_ring   域内环带背景场（用于幅值/校验/反演）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Prov:
    """溯源态标签常量（uint8 写入图像：四态 + 合成）。"""
    ORIGINAL = 0      # 原样保留（无发光 / 文字笔画 / 未处理背景）
    INVERTED = 1      # 档1/档2 反演通过（blend 精准反解）
    SUBTRACTED = 2   # 档2 反演通过（沿发光方向投影精确减除，含 unknown）
    FILLED = 3        # 档3：路由到重建填充（校验失败/饱和/不可辨识）
    SYNTH = 4         # 合成回退（源不足，B̂_ring 底 + 谱匹配纹理）


class Mode(str):
    """域内发光模式（模式拟合 M-E 输出）。"""
    BLEND = "blend"
    ADDITIVE = "additive"
    SCREEN = "screen"
    UNKNOWN = "unknown"


class Dye(str):
    """染色关系：发光罩在字上(through) / 字画在发光上(behind)。"""
    THROUGH = "through"
    BEHIND = "behind"
    UNKNOWN = "unknown"


@dataclass
class TexStats:
    """M-A 纹理实测：一切阈值（k·σ_tex、margin 等）的度量单位来源。"""
    sigma_tex: np.ndarray          # HxW float32 robust σ 图（clamp ≥1）
    l_tex: int                     # 纹理相关长度（clamp [2,12]）
    bar: float                     # 均值 σ̄_tex（平坦区下限 clamp 1）


@dataclass
class Domain:
    """一个发光域（M-C 种子 → M-D 生长 → M-E~M-G 处理）。"""
    id: int
    mask: np.ndarray               # HxW bool，域内发光像素
    carrier_mask: np.ndarray       # HxW bool，域内载体（笔画）掩码
    B_ring: np.ndarray | None = None        # HxWx3 float32，域内有效
    ring_fill: np.ndarray | None = None     # HxWx3 float32 常数环带背景色（FILLED 重建用）
    B_global: np.ndarray | None = None      # HxWx3 float32，全局背景场快照
    mode: str = Mode.UNKNOWN
    u_hat: np.ndarray | None = None         # (3,) 单位方向；screen 用方向场
    u_field: np.ndarray | None = None       # HxWx3 逐像素方向场（screen）
    G: np.ndarray | None = None             # blend 标定的光晕色场；additive/screen 置 None
    alpha_max: float | None = None          # blend 标定值
    calibrated: bool = False
    sigma_g: float = 3.0                    # 光晕 σ̂_g（α-距离衰减拟合，M-F）
    alpha: np.ndarray | None = None         # HxW float32，域内 α 场
    dye: str = Dye.UNKNOWN                  # 逐域染色判断（M-F）
    saturated: np.ndarray | None = None     # HxW bool，饱和像素（253+ 2LSB）
    tier: np.ndarray | None = None          # HxW uint8 记录逐像素路由档（0/1/2/3）
    kv_local: float = 1.0                   # 曲率信任后的校验系数
    report: dict = field(default_factory=dict)


@dataclass
class DeGlowResult:
    """管线输出：去发光图 + 溯源图 + 置信度 + 结构化报告（规格 §2.2）。"""
    image: np.ndarray            # HxWx3 uint8 去发光图
    prov: np.ndarray             # HxW uint8（Prov 枚举）
    conf: np.ndarray             # HxW float32 ∈ [0,1]
    report: dict                 # 每域统计 + ε 核算钩子
    domains: list = field(default_factory=list)   # 处理过的 Domain（供残差回灌）
    has_glow: bool = True        # False = 未检出发光域（no-op 由调用方处理）
    text_layer: np.ndarray | None = None   # 可选：去底文字层（T̂，MVP 未启用）