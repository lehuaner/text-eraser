# Text Eraser 文字擦除算法说明（函数级 / 文件级调用指南）

> 阅读对象：需要在**代码里直接调用**本项目的 AI / 开发者（不经过 FastAPI）。
> 全文以 **Python import + 函数调用** 为准；Web 端只是 `text_eraser/eraser.py` 的一个薄封装。
> 语言惯例：参数/函数名保持英文原文，说明用中文。
>
> **0.3.0 架构**：算法核心只有一份实现 —— `textcore.wasm`（Rust→WebAssembly，随包分发）。
> 后端经 wasmtime 调用、浏览器经 WebAssembly 调用，**同一份字节码，结果逐字节一致**。
> Python/cv2 核心实现（numpy PatchMatch、通道法去发光等）已删除。

---

## 1. 文件清单与职责

| 文件 | 提供 | 依赖 |
|---|---|---|
| `text_eraser/eraser.py` | **管线级入口** `erase_text()` / 批量 `erase_batch()`（测全流程用这两个就够） | text_select + patch_fill + _shared_core |
| `text_eraser/core.py` | **共享核算子公开门面**（自定义管线入口，见 §7） | _shared_core → _textcore |
| `text_eraser/_shared_core.py` | 后端调用 wasm 算子的唯一集成点（deglow / patchmatch / 形态学 / 连通域 / 灰度…） | wasmtime |
| `text_eraser/_textcore.py` | textcore.wasm binding（**线程本地实例**；wasm 打包在 `assets/`） | wasmtime |
| `text_eraser/text_select.py` | 蒙版/检测：`detect_text_mask()`、`detect_text()`、`_fill_nearby_white()`、`_fill_bright_near_mask()`、`_absorb_zone_bright_core()`、`to_rgb_uint8()` | classic 路径零依赖；ml 路径内部按需 import ml_text_select |
| `text_eraser/ml_text_select.py` | DBNet 推理：`detect_text_ml()`、`detect_text_mask_ml()`、`_dbnet_infer()`、`ensure_model()`/`is_model_available()` | onnxruntime（Lazy），模型自动下载 ~5MB |
| `text_eraser/patch_fill.py` | 内容识别填充编排：`inpaint()`（wasm 填充；可独立用于去水印/杂物） | numpy + cv2(取样栅格化/预检) + wasm |
| `shared/src/*.rs` | 算法核心 Rust 源码（deglow / patchmatch / telea / masksynth / lib） | cargo → wasm32 |
| `shared/bindings/` | 三端 binding（py / node / browser）+ 跨端一致性 harness（`shared/_verify_align/`） | — |
| `browser/src/` | 浏览器引擎 ESM 包（`erase()` / `eraseTextGlyphs()` / `inpaint()`） | opencv.js + onnxruntime-web |

> 注意：`text_eraser/text_select.py` 里有 **DBNet 检测框默认回退 `max_area_ratio=0.05`**，见 §6 坑 1。
> cv2 (OpenCV) 仍为环境依赖，但**只用于 DBNet 检测链**（resize INTER_AREA / GaussianBlur 等
> float 算子位级复刻不可达）与画笔栅格化；核心算法不再经过 cv2。

---

## 2. 快速上手（最小调用示例）

```python
import numpy as np
from PIL import Image
from text_eraser.eraser import erase_text
from text_eraser.text_select import to_rgb_uint8, detect_text_mask
from text_eraser.patch_fill import inpaint

# ---- A. 全流程一步出结果（推荐） ----
rgb = to_rgb_uint8(Image.open("demo.png"))          # HxWx3 uint8 RGB
result, mask, meta = erase_text(rgb, return_mask=True)
# result: 已擦除文字的图;  mask: 填充蒙版(255=填充区);
# meta: {mask_pix, mask_filled_pix, inpaint_seconds, method, boxes, edge_used, deglow_img?, glow_zone?}
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

# ---- D. 批量并发（多图并行，线程本地 wasm 实例） ----
from text_eraser import erase_batch
results = erase_batch([rgb1, rgb2, rgb3], workers=4)   # 返回与输入同序
```

---

## 3. 函数 API 详解

### 3.1 `text_eraser/eraser.py::erase_text`（管线级，唯一业务入口）

```python
def erase_text(
    rgb: np.ndarray,                 # HxWx3 uint8 RGB
    *,
    edge: int = 1,                   # 「移动边缘」：蒙版(展示)与填充区同步外扩(>0)/收缩(<0)；1=膨胀1px吃掉AA边缘
    auto_edge: bool = True,          # 按文字色残留自动判定最小外扩(多数图1, 硬图自动到2)
    auto_max_edge: int = 2,
    q_off: float = 55.0,             # 蒙版紧密度 [30,70]，越高越贴字形
    max_area_ratio: float = 0.40,    # 单块文字占图比例上限
    max_box_ratio: float = 0.40,     # 最终框占图比例上限
    ml_max_side: int = 960,          # DBNet 推理最长边
    direction: float | None = None,  # 纹理方向°(木纹/条带时用)
    edge_aware: bool = False,        # 历史实验项，保持 False
    return_mask: bool = False,
    tint_fill: bool = True,          # 色偏区域生长(红蒙版叠加/淡绿光晕并入蒙版)
    fill_white: bool = True,         # 临近纯白补全(描边字漏白)
    fill_max_dist: int = 12,         # 孤立纯白段最大吞并距离(px)
    deglow_scheme: str = "v2",       # 去发光: "v2"(默认/唯一) / "off"
    deglow_strength: float = 1.0,    # [0,1] 去发光力度(v2 内部钳到≥1.15 允许过冲)
    deglow_zone_ratio: float = 0.6,  # 发光区判定: 强核占比门
    deglow_zone_expand: int = 10,    # 发光区扩边(px)
    deglow_protect_px: int = 1,      # 文字边缘保护(px)
    deglow_chroma_keep: bool = True, # 保留色度(去绿不伤暖色)
    deglow_mask_soft: float = 0.0,   # 「透明度扩展」软带半径(px, 0=关)
    # 兼容参数(deglow_green_thr/range/glo/protect): 0.2.x 通道法旋钮, 已不生效
)
# return_mask=False -> (result, meta)；True -> (result, mask, meta)
```

v2 管线（`_erase_deglow_v2`）：
`detect_text_mask(rgb, tint=False)` → **wasm** `deglow_full_green_v2`（减绿度去发光，
G 通道动刀、绿晕→中性灰、永不变黑）→ `detect_text_mask(clean, tint=True)` 再检测 →
并集 → 3×3 闭运算 → `_fill_bright_near_mask` 亮核吸收 → `_absorb_zone_bright_core` →
残余绿/暗源剔除取样 → 移动边缘 → **wasm** PatchMatch 填充。
meta 额外携带 `deglow_img`（去发光中间图）与 `glow_zone`（发光区），前端分步展示。

`erase_batch(images, *, workers=None, return_mask=False, **kw)`：
多图并行。每线程持有独立 wasm 核实例（wasmtime Store 非线程安全）；
wasmtime FFI 释放 GIL，实测接近线性加速。单图内部不做并行——逐框拆分会改变
填充结果、破坏前后端逐字节一致（有意取舍）。

### 3.2 `text_eraser/text_select.py`（检测 + 蒙版）

```python
def detect_text_mask(raw, strength=1.0, method="ml",
                     min_area=30, max_area_ratio=0.05, max_box_ratio=0.40,
                     max_side=960, work_max=1280, q_off=50.0,
                     tint_fill=True, fill_white=True, fill_max_dist=12):
    """-> (mask, boxes)。method="ml": DBNet 框; "classic": 纯 CV 框。
    蒙版统一 = 框内 Otsu + 迭代纯白补全 (§4)。"""
    # ⚠️ 直接调用必须传 max_area_ratio=0.40，否则大字框会被默认 0.05 过滤

def detect_text(raw, strength=1.0, min_area=30, max_area_ratio=0.05,
                max_box_ratio=0.40, vthr=8, pad=3, work_max=1280,
                method="classic", box_threshold=0.3, max_side=960):
    """-> 文字框列表。method 默认 "classic"（零依赖）。"""
    # 注：detect_text 与 detect_text_mask 的默认 method 不一致（classic vs ml）

def _fill_nearby_white(rgb, mask, pad=3, min_lum=200, rounds=2):
    """迭代把蒙版邻域内的绝对纯白(>200)并入 —— 字顶收尖/1px 细白线全靠它。"""

def _fill_bright_near_mask(rgb, mask):
    """v2 蒙版修复①: 并入去发光后仍偏亮的紧邻像素。"""

def _absorb_zone_bright_core(clean, orig, mask, zone, min_rgb_lo=100):
    """v2 蒙版修复②: 发光区(zone)内的亮核并入填充蒙版。"""

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
            should_cancel=None,        # ⚠️ 0.3.0 起为 no-op(wasm 填充不可中断), 仅兼容保留
            direction=None,            # 角度°，沿直线取源(主导纹理)
            flat_span=40, flat_tex=20.0):  # 平滑渐变背景门(环带梯度<flat_tex → TELEA 扩散)
    """-> HxWx3 uint8。填充算法 = wasm `patchmatch_inpaint`(与浏览器逐字节一致)。
    平滑渐变+无纹理背景由 wasm `pm_smooth_telea_full` 权威判定并做扩散插值
    (保留渐变、消杂色); 纹理/均匀背景走 PatchMatch。
    内部已对整图做 4px replicate 内边距，文字贴边不崩。
    核加载失败抛 CoreLoadError(无 Python 回退)。"""
```

### 3.5 `text_eraser/core.py`（共享核算子门面，自定义管线用）

```python
from text_eraser import core

# 高层算子（浏览器与后端调同一份 wasm）
core.deglow_full_green_v2(rgb, tmask, strength, zone_ratio, zone_expand,
                          protect_px, chroma_keep)   # -> (clean, core_mask, zone)
core.erase_text_glyphs(rgb, tmask, tmask2=None, strength, …, edge,
                       direction_deg, seed)          # -> (result, fill, clean, zone)
core.patchmatch_inpaint_fill(roi_f32, roi_mask, sample=None, p=7, direction_deg, seed)
core.smooth_telea_full(rgb, mask, flat_tex)          # 平滑分支未触发返回 None
core.grow_color_tint(rgb, mask, …)                   # 色偏生长闭合

# 底层原语（与浏览器 cv-bridge 一致）
core.rgb2gray / threshold_otsu / dilate / erode / morphology_ex
core.connected_components(_with_stats) / edt_to_nearest_zero
core.resize_gray_cubic / resize_float_linear

core.using_shared_core() -> bool      # wasm 就绪?
core.get_core() / reset_core()        # 线程本地实例管理
```

---

## 4. 算法机制（怎么工作的）

1. **框定位**：`method="ml"` 用 DBNet（`max_side=960`）出文字框；`"classic"` 用自适应阈值+边缘门限出候选再并框。
2. **Otsu 逐字蒙版**（每框）：Otsu 双峰 → 取**少数侧**为文字（相等取离 127.5 更远侧）→ 两侧均值差<20 判无字跳过 → 1px CLOSE 桥接 AA 断口 → `q_off` 控制 0~2px 附加膨胀 → 框内连通域清理。
   - 局限：描边/多色字只选一侧会漏白字身 → 下一步兜底。
3. **临近高亮补全**（`_fill_nearby_white`）：三档、亮度下限随距离收紧——
   ① 连通扩散(pad×rounds，默认 6×5)吃纯白外延线；
   ② 距离场(≤max_dist=12)吃孤立纯白段(不要求连通路径)；
   ③ 近距 AA 档(距蒙版≤3px 且 ≥185)吃 AA 渐隐的字尖/弯钩尾。
   中灰背景/描边不误收。
4. **去发光 v2**（wasm `deglow_full_green_v2`）：强绿信号门 → 减绿度（只动 G 通道，
   绿晕→中性灰、永不变黑、底层纹理保留）→ alpha 分解 + 调和场插值恢复背景 →
   zone（发光区）供亮核吸收。大发光区经 B+detail 重建恢复纹理（仓库内回归脚本验证）。
5. **填充**（wasm `patchmatch_inpaint`）：Criminisi 优先级（置信度×梯度数据项）→
   PatchMatch 找最相似源块（随机 mulberry32 + 邻域相干，f32 SSD 顺序累加）→
   整块搬运 + 颜色自适应消缝（锚定「目标块局部已知上下文」，方差对齐保留纹理对比度）。
   - ⚠️ 教训：**不要**全局环色对齐，也**不要**填后 bilateral 平滑或重叠块 50% 平均——
     三者都已实测毁掉纹理（涂抹/平板感）。平滑渐变背景例外：wasm 判定
     `tex<flat_tex` 时走 TELEA 扩散插值（保留渐变、消杂色）。
6. **展示蒙版**：`edge`（移动边缘）对返回的 mask 做整体膨胀(>0)/腐蚀(<0)（=PS 移动边缘），展示蒙版即真实填充区，所见即所得。
7. **一致性保证**：形态学/连通域/灰度/Otsu/EDT/resize/去发光/蒙版修复/PatchMatch
   全部在共享 wasm 核内，后端(Python/wasmtime)、Node、浏览器三端对同输入
   md5 逐字节一致（harness：`shared/_verify_align/`；验收门：`scripts/parity_check.py`）。

---

## 5. 参数速查（一表到底）

| 参数 | 出现处 | 默认 | 推荐/说明 |
|---|---|---|---|
| `edge` | erase_text | 1 | 「移动边缘」：1=膨胀1px吃掉AA边缘，0=仅取Otsu字形，负=收缩选区 |
| `auto_edge`/`auto_max_edge` | erase_text | True/2 | 自动判定最小外扩（多数图 1，硬图自动 2） |
| `q_off` | erase_text/detect_text_mask | 55.0 | [30,70]，越高越贴字形 |
| `max_area_ratio` | 检测类 | 0.05(函数)/0.40(管线) | **直接调函数必须显式 0.40** |
| `max_box_ratio` | 检测类 | 0.40 | 小图大字调高到 0.6+ |
| `min_area` | 检测类 | 30 | 越小召回小字、越多噪点 |
| `max_side` | DBNet | 960 | 越大小字召回越好、越慢 |
| `direction` | inpaint/erase_text | None | 角度°（图像坐标），主导纹理背景用 |
| `tint_fill`/`fill_white`/`fill_max_dist` | erase_text | True/True/12 | 色偏生长 / 纯白补全 / 孤立白段距离(0=关) |
| `deglow_scheme` | erase_text | "v2" | "v2" / "off"（0.3.0 起无其他值） |
| `deglow_strength` | erase_text | 1.0 | v2 内部钳到 ≥1.15 允许过冲去净残绿 |
| `deglow_zone_ratio/expand`、`deglow_protect_px`、`deglow_chroma_keep` | erase_text | 0.6/10/1/True | v2 发光区门/扩边/文字保护/保色度 |
| `deglow_mask_soft` | erase_text | 0.0 | 透明度扩展软带半径(px)，光晕未吞进蒙版的截图类用 |
| `flat_tex` | inpaint | 20.0 | 平滑背景 TELEA 门（环带梯度中位阈值） |
| `workers` | erase_batch | CPU 核数 | 批量并发线程数 |

---

## 6. 坑与教训（接手必读）

1. **函数裸调默认 `max_area_ratio=0.05` → 大字框被过滤 → 返回空蒙版**。所有直接调用 `detect_text` / `detect_text_mask` 都必须显式传 0.40（erase_text 已带内部默认，无需管）。
2. **`detect_text_ml` 的 `max_box_ratio` 与 `pad` 互斥**：pad 调大 → 框变大超 ratio → 框被丢。别靠大 pad 捞小字。
3. **`edge_aware=True` 勿用**（ellipse(8) 大膨胀 → 蒙版占图 60%+，残留反增）。
4. **“相对亮度带生长”（曾在 ml_text_select，已删除）勿恢复**：以环内亮度中值作锚会把字旁中灰背景块误圈。需要补边缘 → 用 `_fill_nearby_white`（绝对纯白）或 `edge`（移动边缘）。
5. **`detect_text_mask_ml` 只盖字形约 40%**，是"备选独立 API"，不要作为默认蒙版路径。
6. **wasm 是唯一判定者**：不要在 Python 侧复刻判定逻辑再决定走哪条分支（float 算子
   ULP 级跨线会造成与浏览器分歧）。需要分支信息就用 wasm 算子的返回值（如
   `smooth_telea_full` 返回 None 即未触发）。
7. **wasmtime Store 非线程安全**：永远经 `core.get_core()`（线程本地）使用，不要缓存
   跨线程共享的 TextCore 实例。
8. **验收/回归工具**（0.3.0 现存）：`scripts/parity_check.py`（后端 vs 浏览器 wasm
   逐字节 parity，4 基准图）、`scripts/render.py`（全管线对照图）、
   `shared/_verify_align/`（三端算子 md5 harness）。

---

## 7. 自定义扩展点

- **引擎选择**：同一份 wasm，跑在后端（`from text_eraser import erase_text`）或浏览器
  （`import { erase, eraseTextGlyphs, inpaint } from 'text-eraser-browser'`，见 `browser/`）。
  Web 界面的「后端计算 / 本地浏览器计算」切换只是 demo。
- **换文字检测器**：实现一个 `fn(raw, **) -> list[{"x0","y0","x1","y1"}]`，replace 掉 `detect_text` 的 ml/classic 分支即可（蒙版生成只依赖框）。
- **只换蒙版策略**：在 `erase_text` 前自行调 `detect_text_mask(...)` 或写新蒙版函数，再喂给 `patch_fill.inpaint(rgb, dil_mask, sample_mask=255-dil_mask)`。
- **只换/复用某一步**：`from text_eraser import core` 直接取共享核算子自由编排
  （去发光、纯填充、蒙版合成、形态学……），签名见 §3.5。
- **去水印/去任意内容**：无需文字检测，直接构造 hole + sample_mask 调 `inpaint`（见 §2 示例 C）。
- **批量吞吐**：`erase_batch()`（多图并行）；Web 服务并发已由线程本地核实例天然支持。
