"""M0 · 合成 GT 生成器与分层指标（规格 §5 / §6）。"""

from deglow.gt.generator import (GenCase, compose, hash_cases, iter_cases,
                                 self_check)
from deglow.gt.metrics import evaluate_case, layered_metrics, routing_epsilon

__all__ = [
    "GenCase", "compose", "hash_cases", "iter_cases", "self_check",
    "evaluate_case", "layered_metrics", "routing_epsilon",
]
