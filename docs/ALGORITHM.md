# Text Eraser 文字擦除算法说明（函数级 / 文件级调用指南）

> 阅读对象：需要在**代码里直接调用**本项目的 AI / 开发者（不经过 FastAPI）。
> 全文以 **Python import + 函数调用** 为准；Web 端只是 `text_eraser/eraser.py` 的一个薄封装。
> 语言惯例：参数/函数名保持英文原文，说明用中文。

---

## 1. 文件清单与职责

| 文件 | 提供 | 依赖 |
|---|---|---|
| `text_eraser/eraser.py` | **管线级入口** `erase_text()`（测全流程用这一个就够） | text_select + patch_fill |
| `text_eraser/text_select.py` | 蒙版/检测：`detect_text_mask()`、`detect_text()`、`_detect_text_mask_classic()`、`_fill_nearby_white()`、`_clean_text_mask()`、`_mask_to_boxes()`、`to_rgb_uint8()` | classic 路径零依赖；ml 路径内部按需 import ml_text_select |
| `text_eraser/ml_text_select.py` | DBNet 推理：`detect_text_ml()`、`detect_text_mask_ml()`、`_dbnet_infer()`、`ensure_model()`/`is_model_available()` | onnxruntime（Lazy），模型自动下载 ~5MB |
| `text_eraser/patch_fill.py` | 内容识别填充：`inpaint()`（可独立用于去水印/杂物，不限于文字） | numpy + cv2 |

> 注意：`text_eraser/text_select.py` 里有 **DBNet 检测框默认回退 `max_area_ratio=0.05`**，见 §6 坑 1。

---

## 2. 快速上手（最小调用示例）

```python
import numpy as np
import cv2
from PIL import Image
from text_eraser.eraser import erase_text
from text_eraser.text_select import to_rgb_uint8, detect_text_mask
from text_eraser.patch_fill import inpaint

# ---- A. 全流程一步出结果（推荐） ----
rgb = to_rgb_uint8(Image.open("demo.png"))          # HxWx3 uint8 RGB
result, mask, meta = erase_text(rgb, return_mask=True)
# result: 已擦除文字的图;  mask: 逐字蒙版(255=文字);  meta: {mask_pix, mask_filled_pix, inpaint_seconds, method, boxes}
Image.fromarray(result).save("out.png")

# ---- B. 分步：只要蒙版(调试/可视化/自定义流程) ----
mask, boxes = detect_text_mask(
    rgb, method="ml", q_off=55.0,
    max_area_ratio=0.40, max_box_ratio=0.40,   # 必须显式传 0.40，勿用默认 0.05
)
# boxes = [{"x0","y0","x1","y1"}, ...] 显示用文字框

# ---- C. 分步：只用填充器（去水印/去任意内容） ----
hole = np.zeros(rgb.shape[:2], np.uint8)          # >0 = 要清除的区域
hole[10:40, 20:80] = 255
sample = (255 - hole)                              # 取样区 = 整图去掉区域
filled = inpaint(rgb, hole, sample_mask=sample)
Image.fromarray(filled).save("removed.png")
```

---

## 3. 函数 API 详解

### 3.1 `text_eraser/eraser.py::erase_text`（管线级，唯一业务入口）

```python
def erase_text(
    rgb: np.ndarray,                 # HxWx3 uint8 RGB
    *,
    edge: int = 1,                   # 「移动边缘」：蒙版(展示)与填充区同步外扩(>0)/收缩(<0)；1=膨胀1px吃掉AA边缘
    q_off: float = 55.0,             # 蒙版紧密度 [30,70]，越高越贴字形
    max_area_ratio: float = 0.40,    # 单块文字占图比例上限
    max_box_ratio: float = 0.40,     # 最终框占图比例上限
    ml_max_side: int = 960,          # DBNet 推理最长边
    direction: float | None = None,  # 纹理方向°(木纹/条带时用)
    edge_aware: bool = False,        # 历史实验项，保持 False
    return_mask: bool = False,
)
# return_mask=False -> (result, meta)；True -> (result, mask, meta)
```

管线内部 = `detect_text_mask(method="ml", …)` → `dilate/erode(mask, ellipse(edge))`（移动边缘，>0 膨胀/<0 腐蚀）→ `sample_mask=整图−膨胀蒙版` → `patch_fill.inpaint(...)`。展示蒙版即真实填充区（移动边缘后），所见即所得。

### 3.2 `text_eraser/text_select.py`（检测 + 蒙版）

```python
def detect_text_mask(raw, strength=1.0, method="ml",
                     min_area=30, max_area_ratio=0.05, max_box_ratio=0.40,
                     max_side=960, work_max=1280, q_off=50.0):
    """-> (mask, boxes)。method="ml": DBNet 框; "classic": 纯 CV 框。
    蒙版统一 = 框内 Otsu + 迭代纯白补全 (§4)。"""
    # ⚠️ 直接调用必须传 max_area_ratio=0.40，否则大字框会被默认 0.05 过滤

def detect_text(raw, strength=1.0, min_area=30, max_area_ratio=0.05,
                max_box_ratio=0.40, vthr=8, pad=3, work_max=1280,
                method="classic", box_threshold=0.3, max_side=960):
    """-> 文字框列表。method 默认 "classic"（零依赖）。"""
    # 注：detect_text 与 detect_text_mask 的默认 method 不一致（classic vs ml）

def _detect_text_mask_classic(raw, boxes=None, strength=1.0, min_area=30, q_off=50.0):
    """-> HxW uint8 蒙版。纯 Otsu 分割（无纯白补全），用于调试/底层。"""

def _fill_nearby_white(rgb, mask, pad=3, min_lum=200, rounds=2):
    """迭代把蒙版邻域内的绝对纯白(>200)并入 —— 字顶收尖/1px 细白线全靠它。"""

def _clean_text_mask(mask, H, W, min_area=30, max_area_ratio=0.05):
    """连通域去噪：极小点/超大块/过细长(UI 线) 剔除。"""

def _mask_to_boxes(mask): -> list[dict]
def to_rgb_uint8(raw): -> HxWx3 uint8
```

### 3.3 `text_eraser/ml_text_select.py`（DBNet）

```python
def is_model_available() -> bool                 # 模型文件是否就绪
def ensure_model() -> str                        # 缺则下载, 返回模型路径(线程安全)
def detect_text_ml(raw, strength=1.0, min_area=30, max_area_ratio=0.05,
                   max_box_ratio=0.20, box_threshold=0.3, max_side=960, pad=3):
    """-> 文字框列表（原图坐标）。阈值=box_threshold-0.15*strength。"""
def detect_text_mask_ml(raw, strength=1.0, min_area=30, max_area_ratio=0.05,
                        box_threshold=0.3, max_side=960,
                        mask_threshold=0.4, mask_max_side=1600):
    """-> (mask, boxes)。DBNet 概率图直接出的逐字蒙版 —— 只盖字形约 40%，
    对同色/描边字不完整，勿作为默认蒙版路径（见 §6 坑5）。"""
def _dbnet_infer(rgb, strength, box_threshold, max_side):
    """-> (prob, nw, nh, H, W, thr)。概率图 HxW float32，可自行取阈值。"""
```

### 3.4 `text_eraser/patch_fill.py::inpaint`（填充器，可独立复用）

```python
def inpaint(image_rgb, mask,
            sample_mask=None,          # >0 为唯一取样源; 支持 dict(画笔笔画)/HxW bool
            should_cancel=None,        # 返回 True 中断(返回当前结果)
            direction=None):           # 角度°，沿直线取源(主导纹理)
    """-> HxWx3 uint8。PatchMatch(优先级+邻域相干+颜色自适应) + TELEA 兜底。
    内部已对整图做 4px replicate 内边距，文字贴边不崩。"""
```

---

## 4. 算法机制（怎么工作的）

1. **框定位**：`method="ml"` 用 DBNet（`max_side=960`）出文字框；`"classic"` 用 自适应阈值+边缘门限 出候选再并框。
2. **Otsu 逐字蒙版**（`_detect_text_mask_classic`，每框）：Otsu 双峰 → 取**少数侧**为文字（相等取离 127.5 更远侧）→ 两侧均值差<20 判无字跳过 → 1px CLOSE 桥接 AA 断口 → `q_off` 控制 0~2px 附加膨胀 → 框内连通域清理。
   - 局限：描边/多色字只选一侧会漏白字身 → 下一步兜底。
3. **临近高亮补全**（`_fill_nearby_white`）：三档、亮度下限随距离收紧——
   ① 连通扩散(pad×rounds，默认 6×5)吃纯白外延线，随图像放大需加大距离；
   ② 距离场(≤max_dist=32)吃孤立纯白段(不要求连通路径)；
   ③ 近距 AA 档(距蒙版≤3px 且 ≥185)吃 AA 渐隐的字尖/弯钩尾(200 阈值吃不到)。
   中灰背景/描边不误收（座驾2 放大 2 倍 + 尾尖实测：漏白 795→0，误收 0）。
4. **填充**（`inpaint`）：Criminisi 优先级（置信度×梯度数据项）→ PatchMatch 找最相似源块（随机+邻域相干）→ 整块搬运 + 颜色自适应消缝；残余边界 TELEA。
   - **重叠软混合**：后续源块覆盖已填像素时 50% 融合，相邻块软过渡消除硬拼缝（实用）。
   - ⚠️ 教训：**不要**全局环色对齐，也**不要**填后 bilateral 平滑——两者都已实测毁掉纹理，
     产生"钉子砸在文字上"的平板感（2026-08 两次失败后回滚）。颜色自适应必须保持
     "目标块局部已知上下文"锚定（原版行为），纯色背景的色块拼接感属可接受水平。
5. **展示蒙版**：`edge`（移动边缘）对返回的 mask 做整体膨胀(>0)/腐蚀(<0)（=PS 移动边缘），展示蒙版即真实填充区，所见即所得。

---

## 5. 参数速查（一表到底）

| 参数 | 出现处 | 默认 | 推荐/说明 |
|---|---|---|---|
| `edge` | erase_text | 1 | 「移动边缘」：蒙版(展示)与填充区同步外扩(>0)/收缩(<0)；1=膨胀1px吃掉AA边缘，2≈旧 mask_pad=2+eoff，0=仅取Otsu字形，负=收缩选区 |
| `q_off` | erase_text/detect_text_mask | 55.0 | [30,70]，越高越贴字形 |
| `max_area_ratio` | 检测类 | 0.05(函数)/0.40(管线) | **直接调函数必须显式 0.40** |
| `max_box_ratio` | 检测类 | 0.40 | 小图大字调高到 0.6+ |
| `min_area` | 检测类 | 30 | 越小召回小字、越多噪点 |
| `max_side` | DBNet | 960 | 越大小字召回越好、越慢 |
| `mask_threshold`/`mask_max_side` | detect_text_mask_ml | 0.4/1600 | 仅备选 ML 蒙版路径用 |
| `direction` | inpaint/erase_text | None | 角度°（图像坐标） |

---

## 6. 坑与教训（接手必读）

1. **函数裸调默认 `max_area_ratio=0.05` → 大字框被过滤 → 返回空蒙版**。所有直接调用 `detect_text` / `detect_text_mask` 都必须显式传 0.40（erase_text 已带内部默认，无需管）。
2. **`detect_text_ml` 的 `max_box_ratio` 与 `pad` 互斥**：pad 调大 → 框变大超 ratio → 框被丢。别靠大 pad 捞小字。
3. **`edge_aware=True` 勿用**（ellipse(8) 大膨胀 → 蒙版占图 60%+，残留反增）。
4. **“相对亮度带生长”（曾在 ml_text_select，已删除）勿恢复**：以环内亮度中值作锚会把字旁中灰背景块误圈（`武器` 图 790px 灰块即此例）。需要补边缘 → 用 `_fill_nearby_white`（绝对纯白）或 `edge`（移动边缘，均匀外扩 1px）。
5. **`detect_text_mask_ml` 只盖字形约 40%**，是"备选独立 API"，不要作为默认蒙版路径。
6. **回归基线**（默认参数）：`武器` 盖白 2039/2039、残白 0；`座驾` 盖白 1719/1719、残白 0。指标看“白字身覆盖率 + 中灰误收数”，别只看残白。
7. 回归脚本：`scripts/final_v3.py`（统计 + 3 栏对比图 → `data/final/`）。

---

## 7. 自定义扩展点

- **换文字检测器**：实现一个 `fn(raw, **) -> list[{"x0","y0","x1","y1"}]`，replace 掉 `detect_text` 的 ml/classic 分支即可（蒙版生成只依赖框）。
- **只换蒙版策略**：在 `erase_text` 前自行调 `detect_text_mask(...)` 或写新蒙版函数，再喂给 `patch_fill.inpaint(rgb, dil_mask, sample_mask=255-dil_mask)`。
- **只换填充**：替换 `patch_fill.inpaint` 调用点，只要签名 `f(image_rgb, mask, sample_mask)->uint8`。
- **去水印/去任意内容**：无需文字检测，直接构造 hole + sample_mask 调 `inpaint`（见 §2 示例 C）。