"""M8 · 夜间回归与检测器确定性契约（规格 §8 / M0）。

regression.run_regression：在 GT 用例集上跑 pipeline，聚合分层指标，
任一轴任一层失败即阻断；支持 frozen.json gt_hash 一致性断言。
det_contract.test：同输入 100 次逐位一致（检测器确定性契约，M0 验收门）。
"""
