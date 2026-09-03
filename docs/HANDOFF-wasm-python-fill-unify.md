# 交接文档：wasm 核心 vs Python 核心「文字填充内容」不一致

> 生成时间：2026-09-03
> 分支：`python-core-toggle`（HEAD = `78fb099`，**本地分支，远程没有**）
> 工作区状态：`text_eraser/eraser.py`、`text_eraser/patch_fill.py` 有**未提交改动，且当前是回退状态（比改之前更差）**
> 复现脚本：`data/_diag3_repro.py`　诊断产物：`data/_diag3/`

---

## 0. TL;DR（接手后先做这三件事）

1. **先回滚 `eraser.py`**（那处 `_run_fill` 改法是错的，见 §4）：
   ```bash
   git checkout -- text_eraser/eraser.py
   ```
   `patch_fill.py` 的改动**保留**（原理正确，见 §3.1）。
2. 按 §5 的方案重做 `eraser.py` 那一处：**不要走 `_run_fill`，直接 `pm_inpaint(clean0, fill, ...)`**。
3. 重跑 `python data/_diag3_repro.py`，验收标准：**MASK diff 保持 16~19px 不变，RESULT diff 从 36.8%/57.6% 大幅下降**。

---

## 1. 问题现象（用户报的）

Web 端 `python-core-toggle` 分支有个「Python 核心 / wasm 核心」互斥开关。对两张**发光小字覆盖完整图**：

| 图 ID | 尺寸 | box |
|---|---|---|
| `1787767556635` | 75×78 | x0=18, y0=23, x1=64, y1=67 |
| `1787767611178` | 65×59 | x0=11, y0=5, x1=56, y1=49 |

两种核心的擦除结果不一致，**区别在文字填充内容部分**。用户判断「应该是某一个点没对上」，要求**以 Python 为目标继续统一**。

---

## 2. 诊断结论（已定量确认，这部分是可信的）

用 `data/_diag3_repro.py` 分别以 `python_core(True)` 和 wasm 模式跑同一张图、同一套 web 参数
（`edge=1, auto_edge=True, q_off=55.0, max_area_ratio=0.4, max_box_ratio=0.4, direction=None,
edge_aware=False, tint_fill=True, fill_white=True, fill_max_dist=12, glow_mode='auto',
deglow_scheme='v2', deglow_strength=1.0, deglow_zone_ratio=0.6, deglow_zone_expand=10,
deglow_protect_px=1, deglow_chroma_keep=True`），逐阶段对比：

### 修复前的基线数据

| 图 | CLEAN(去发光) diff | MASK diff | RESULT diff | 差异位置 |
|---|---|---|---|---|
| `1787767611178` | **max=0（逐字节一致）** | 16 px | 36.8% | **100% 落在洞内** |
| `1787767556635` | max=18（轻微） | 19 px | 57.6% | 绝大多数落在洞内 |

**→ 分歧不在去发光、也不在蒙版，纯粹是「洞里填什么」不同。** 这一点很关键，说明整条上游链路（DBNet 检测、去发光 v2、蒙版合成）其实已经对齐得很好了，只剩最后一步填充。

### 根因：三个引擎的平滑背景填充策略不一致

`patch_fill.inpaint()` 里有个判据：环带纹理 `tex < flat_tex` 时视为**平滑背景**，用扩散填充而不是 PatchMatch。三端行为：

| 引擎 | 平滑区（`tex < flat_tex`） | 纹理区 |
|---|---|---|
| **Python 核心**（`python_core(True)` → `_get_core()` 返回 None） | cv2 `INPAINT_TELEA` | numpy PatchMatch（`patch_fill.inpaint` 主循环） |
| **浏览器**（`browser/src/patchmatch.js` L142–147） | opencv.js `INPAINT_TELEA` | Rust 共享核 `cvb.patchmatchInpaintShared`（L196） |
| **后端 wasm 模式** | ❌ **Rust PatchMatch**（TELEA 被跳过） | Rust PatchMatch |

后端 wasm 是**唯一的异类**。原因是 `patch_fill.py` 里那个 `not using_shared_core()` 门控：

```python
if tex < flat_tex and not using_shared_core():   # ← wasm 模式下这里为 False，TELEA 被跳过
```

这两张图是小字 + 平滑渐变背景，正好命中 `tex < flat_tex`，所以 Python/浏览器走 TELEA、后端 wasm 走 Rust PatchMatch → 洞里内容自然不同。

### 一个被推翻的错误假设（避免接手的人重犯）

`patch_fill.py` 原注释写着「装了 shared core 时平滑背景也走 wasm PatchMatch（与后端/前端同字节）」。
**这句注释是错的。** 读 `browser/src/patchmatch.js` L142–147 可确认浏览器平滑区走的是 opencv.js TELEA，
不是 Rust PatchMatch。所以原来的门控不是「为了对齐浏览器」，而是**制造了后端 wasm 的单点分歧**。
`browser/src/patchmatch.js` L96–101 的注释同样有误导性，建议一并修正。

---

## 3. 已做的两处改动

### 3.1 `text_eraser/patch_fill.py`（**保留，原理正确**）

去掉 `not using_shared_core()` 门控，平滑区一律 cv2 TELEA：

```python
# 平滑背景(环带纹理低 tex<flat_tex)一律用 cv2 TELEA 扩散填充——这与浏览器
# (patchmatch.js 走 opencv.js INPAINT_TELEA) 和 Python 核心(无 core 回退) 行为
# 一致, 是「三端一致」的 host 侧判定。Rust 共享核只负责非平滑纹理区的 PatchMatch
# 填充(下方 using_shared_core 分支 / 浏览器的 patchmatchInpaintShared)。
if tex < flat_tex:
    out = cv2.inpaint(np.clip(img, 0, 255).astype(np.uint8),
                      m.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return out
```

### 3.2 `text_eraser/eraser.py`（❌ **错的，请回滚**）

`_erase_deglow_v2` 第 4 步原本直接用 Rust `erase_text_glyphs` 返回的 `result` / `fill`。
我改成了「只取 Rust 的 `clean0`/`fill`，把填充交回 `_run_fill` 统一调度」：

```python
_result_c, fill, clean0, zone0 = res
sample_exclude = _residual_green(clean0, fill)
...
rf = _run_fill(clean0, fill, boxes, edge=edge, direction=direction, ...)
```

**动机是对的**（想让 wasm 模式也经过 host 侧的 TELEA 判据），**但实现踩了坑**。

---

## 4. 这处改动为什么把结果搞得更差（回退证据）

`_run_fill` 开头（`eraser.py` L517–530）会**先按 `edge` 再膨胀一次蒙版**：

```python
if edge > 0:
    mask_filled = cv2.dilate(mask, _ellipse(edge))   # ← edge=1，又膨胀 1px
```

而 Rust `erase_text_glyphs` 返回的 `fill` **已经是膨胀+修复完的最终蒙版**。
再过一遍 `_run_fill` = **二次膨胀 + 二次 soft_expand**，蒙版直接涨了约 200px：

| 图 | MASK diff（改前 → 改后） | 蒙版像素数（改后 wasm vs py） | RESULT diff（改前 → 改后） |
|---|---|---|---|
| `1787767611178` | 16 → **223** | 1695 vs 1472 | 36.8% → **42.3%** |
| `1787767556635` | 19 → **219** | 1726 vs 1507 | 57.6% → **59.3%** |

原本已经对齐到 16~19px 的蒙版被破坏，填充差异也没解决 → **净负收益，必须回滚这一处。**

---

## 5. 正确的修法（我的想法，未实施）

核心思路：**保住 Rust 那份 `fill` 蒙版原样不动**（它和 Python 蒙版只差 16~19px，已经很好），
只把「填充」这一步换成走 host 侧的 `patch_fill.inpaint`，让 TELEA 判据生效。

即：**不要调 `_run_fill`**（它会改蒙版），而是直接调 `pm_inpaint`：

```python
if res is not None:
    _result_c, fill, clean0, zone0 = res          # fill 原样使用，不再膨胀
    # 与 cv2 回退路径一致的取样剔除
    sample_exclude = _residual_green(clean0, fill)
    if zone0 is not None and bool((zone0 > 0).any()):
        _dx = _dark_source_exclude(clean0, fill)
        if _dx is not None:
            sample_exclude = (_dx | sample_exclude) if sample_exclude is not None else _dx
    sample_mask = (255 - fill).astype(np.uint8)
    if sample_exclude is not None:
        sample_mask[sample_exclude] = 0
    # 走 host 侧 patch_fill：平滑区 → cv2 TELEA(同 Python/浏览器)
    #                       纹理区 → Rust 共享 PatchMatch(同浏览器)
    result = pm_inpaint(clean0, fill, sample_mask=sample_mask, direction=direction)
    mask_filled = fill
    meta = {"mask_pix": int((mask > 0).sum()),
            "mask_filled_pix": int((mask_filled > 0).sum()),
            "inpaint_seconds": time.time() - t0,
            "method": "ml-shared-core", "boxes": boxes,
            "deglow_img": clean0, "glow_zone": zone0}
    return (result, mask_filled, meta) if return_mask else (result, meta)
```

**注意点：**
- `soft_expand` 在这条路径下需要单独处理（Rust 侧已接收 `soft_expand` 参数，别重复应用）。确认 Rust 的 `fill` 是否已含软带，避免和 host 侧再叠一层。
- 验收：MASK diff 必须**保持** 16~19px（不能涨），RESULT diff 应显著下降。
- 若 RESULT 仍有残差，看是否落在**纹理区**——那是 §6.2 的 RNG 问题，不是这次修的范围。

---

## 6. 仍然没解决的问题（后续议题）

### 6.1 wasm 蒙版手术 ≠ Python 蒙版手术（那 16~19px 的来源）

两条链路的蒙版合成逻辑本身就是两套：

- **Python/cv2 路径**：`DBNet(tint=F) ∪ DBNet(clean,tint=T)` → `morphologyEx CLOSE(3×3)` →
  `_fill_bright_near_mask` → `_absorb_zone_bright_core` → `_residual_green` → `_dark_source_exclude`
- **Rust `erase_text_glyphs`**：在 Rust 内部自己做「zone 亮核吸收 + 残余绿/暗源剔除」，产出 `fill`

目前差 16~19px（很小），用户这次也只点名了填充内容，所以**暂时可以不动**。
若要彻底逐字节一致，得让 Rust 的蒙版手术严格复刻 cv2 那一串，或反过来让 host 侧完全信任 Rust 蒙版。

### 6.2 Rust PatchMatch vs numpy PatchMatch 的随机流不同（纹理区无法逐字节一致）

已知问题，之前就记录过：Rust 用 **mulberry32**，numpy 用 **PCG64**。
纹理区（`tex >= flat_tex`）两端种子/迭代顺序不同 → 结果必然有差异（换 seed 的自然差异量级 mean|d|≈36/765，不是 bug）。
这两张图主要命中平滑区，所以 §5 的修法应该能解决绝大部分。但若用户后续要求**纹理区也逐字节一致**，
必须统一 RNG（让 numpy 侧也用 mulberry32，或让两端都调 Rust 那一份）。

### 6.3 `erase_text_glyphs` 是后端专用，浏览器没走同一入口

浏览器用的是 `patchmatchInpaintShared`（**只填充**）+ JS 侧 TELEA + JS 侧蒙版逻辑；
后端 wasm 用的是 `erase_text_glyphs`（**去发光+蒙版手术+填充 一体**）。
两者虽共享 wasm 二进制，但**入口不同 → 编排不同**。这是「三端一致」的结构性隐患，
`patch_fill.py` 那个错误注释就是这个混乱的产物。长期看应统一入口。

---

## 7. 关于「当前 Python 核心是否不完整」（用户问题 #2，已回答：**不必担心**）

用户怀疑当前 python-core 代码本身不完整，要求对比远程；「如果一致那就算了」。

核查结果：

- `python-core-toggle` 是**纯本地分支**，远程只有 `origin/main`(`84d9c97`) 和 `origin/feat/shared-wasm-core`(`00522c5`)，**没有** `origin/python-core-toggle`。
- `git diff origin/main` 显示：`eraser.py`(+111)、`patch_fill.py`(+27)、`text_select.py`(+5)，以及新增文件
  `_cv.py`、`_shared_core.py`、测试、static 资源。
- **关键**：这些改动全是**wasm 集成脚手架**，且都门控在 `not using_shared_core()` / `_get_core() is None` 之后。
  原始 cv2/numpy 的 Python 逻辑**被完整保留、没有被改写**。

**结论：Python 核心没有损坏或缺失，其行为与原始实现一致。** 与 `origin/main` 的差异就是这次刻意加的
「核心切换」功能（尚未 push）。所以按用户的「如果一致那就算了」——**这条不用再追。**

---

## 8. 复现与验收手册

```bash
# 复现对比（会输出各阶段 diff 数值 + 保存对比图）
python data/_diag3_repro.py

# 产物
data/_diag3/<id>_cmp.png        # orig | python核心 | wasm核心 三联对比
data/_diag3/<id>_cmp_big.png    # 放大版
data/_diag3/<id>_maskdiff.png   # 蒙版差异
data/_diag3/<id>_reshm.png      # 结果差异热力图
data/_diag3/<id>_{py,wasm}_{mask,res}.npy
```

**验收门槛（改完 §5 后必须同时满足）：**
1. `MASK diff` ≤ 19px（不得高于基线）
2. `RESULT diff` 相比基线 36.8% / 57.6% 显著下降
3. 残余差异（若有）应位于纹理区，可用 `tex` 值核对

**切核心的方式：**
- 进程级：环境变量 `TEXTCORE_BACKEND=0` 关掉 wasm
- 单请求级：`with python_core(True):`（`text_eraser/_shared_core.py` 的 ContextVar）

---

## 9. 相关文件索引

| 文件 | 作用 |
|---|---|
| `text_eraser/patch_fill.py` | `inpaint()` — `tex < flat_tex` 平滑判据 + TELEA/PatchMatch 分流（**已改，保留**） |
| `text_eraser/eraser.py` | `_erase_deglow_v2()` L708~ 第 4 步 wasm 分支（**已改，请回滚重做**）；`_run_fill()` L505~（会二次膨胀，别乱用） |
| `text_eraser/_shared_core.py` | `python_core()` ContextVar、`_get_core()`、`using_shared_core()`、`erase_text_glyphs()`、`patchmatch_inpaint_fill()` |
| `text_eraser/_cv.py` | cv2 shim，dilate/erode/morphologyEx/connectedComponents/rgb2gray 走 wasm |
| `browser/src/patchmatch.js` | L142–147 opencv.js TELEA（平滑）；L196 `patchmatchInpaintShared`（纹理）；L96–101 注释有误导，建议修正 |
| `shared/src/deglow.rs` | Rust `erase_text_glyphs` 一体化入口 |
| `shared/build/textcore.wasm` | 共享 wasm 二进制（216KB） |
| `data/_diag3_repro.py` | 复现/对比脚本 |
| `data/history/<id>/{orig.bin,meta.json}` | 两张问题图与其参数 |

---

## 10. 提交状态提醒

- 当前 `python-core-toggle` **未 push**，用户也没要求 push。
- 工作区有**未提交的回退改动**（§3.2 那处）。接手第一步就是 `git checkout -- text_eraser/eraser.py`。
- ⚠️ 本仓库已知 git 坑：带斜杠的分支名（如 `feat/shared-wasm-core`）在本机 git 2.55 偶发只写 reflog、
  不写 `refs/heads/<dir>/<name>` 文件，导致 `git status` 把所有文件误列为 `new file`。
  若遇到，手动 `mkdir -p .git/refs/heads/feat && printf '<sha>\n' > .git/refs/heads/feat/<name>` 恢复，
  **不要重建提交**。

---

## 11. 勘误与执行结果（2026-09-04，feat/shared-wasm-core）

### 11.1 §2 的关键前提被实测推翻

**这两张图并不命中 `tex < flat_tex` 平滑分支。** 复刻 `patch_fill.inpaint` 判据实测：
`1787767611178` 环带纹理中位 `tex=30.27 > flat_tex=20.0` → **两侧都走 PatchMatch**
（`1787767556635` 同理）。因此 §5 落地后 MASK diff 守住（16/19px 未回退），但 RESULT diff
基本不变（57.6%/37.2%）。同输入（同一 clean、同一蒙版、同一 sample_mask）下
numpy PM vs Rust 共享 PM 的差异（1413px, mean 33.4）≈ 真实两侧残差（1425px, mean 32.7）
→ **残差 100% 是 §6.2 的双引擎分歧（PCG64 vs mulberry32），不是编排问题**。

§3.1 的 patch_fill 门控去除仍然正确且必要：tex<20 的真平滑图，wasm 模式此前确实跳过
TELEA；且 Rust 内部的 `pm_smooth_telea_with_flat_tex`（deglow.rs L1830）被 `!zone_any`
门住，发光图永远走不到——这是 §2 表格没写透的另一半机制。

### 11.2 已落地（feat/shared-wasm-core，未 push）

1. `eraser.py`：wasm 分支只取 Rust 的 clean/fill/zone，填充交回 `_run_fill` 传
   `edge=0/edge_aware=False`（等价「蒙版已定型、不再二次膨胀」，比 §5 原方案直接调
   `pm_inpaint` 多覆盖了软带），soft_expand 软带混合在 host 侧重做（Rust `out_fill`
   不含软带，deglow.rs L1842/L1899 已核实），并补了空 fill 分支与 `sample_exclude`
   对齐 cv2 回退路径。
2. `patch_fill.py`：§3.1 原样保留。
3. `browser/src/`：`flatTex` 默认 15.0 → **20.0** 对齐 Python `flat_tex`（否则
   tex∈[15,20) 的图后端走 TELEA、浏览器走 PatchMatch），误导注释更新。
4. `data/_diag3_repro.py`：改为子进程 `TEXTCORE_BACKEND` 切核心（feat 分支无
   `python_core` ContextVar，进程级切换与其等价），两分支通用。

### 11.3 剩余工作（按优先级）

1. **§6.2 RNG 统一**：numpy PM 改用 mulberry32(0)（随机数消费顺序与 Rust 严格一致），
   纹理区三端才能逐字节一致——这是这两张图收敛的必要条件。
2. **§6.1 蒙版手术**：16~19px 蒙版差经填充传播会放大（同 clean 两蒙版直接填充对比
   即差 1308px, mean 24.7），逐字节一致还需统一两侧蒙版手术。
3. **浏览器编排对齐**：浏览器 SDK 发光路径仍直接取 Rust 一体结果（te-bundle.js
   `erase()` 的共享核分支）。要与后端逐字节一致，需把本次后端 Python 的改法在 JS
   复刻：取 clean/fill/zone 后由浏览器 host 侧填充。原语 cv-bridge 均已具备
   （opencv.js TELEA、patchmatchInpaintShared、dilateMask、distanceTransform），
   另需接上 soft_expand（当前 SDK 共享核路径传 0、无软带混合）。
