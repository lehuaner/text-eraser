"""deglow · 通用去发光包（v4.1 方案工程落地）。

入口：deglow.pipeline.run(rgb, carrier_mask=None, deglow_strength=1.0, params=None)
产出去发光图 + 溯源图(prov) + 置信度(conf) + 结构化报告(report)。
"""
from deglow.core.types import DeGlowResult, Domain, Mode, Prov
from deglow.pipeline import run

__all__ = ["run", "DeGlowResult", "Domain", "Mode", "Prov"]