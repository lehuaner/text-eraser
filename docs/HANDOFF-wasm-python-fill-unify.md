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

---

## 12. 三端逐字节一致达成（2026-09-04 续，用户要求 wasm 对齐 Python 核心）

### 12.1 发现并修复的三个 bug（本次真正的根因）

1. **numpy 批量快速路径对角线 gather bug**（patch_fill.py）：`xx`/`txx` 的 `dx`
   误与 `yy`/`tyy` 的 `dy` 使用同一轴切片（`dx[None,None,:,None]` / `dx[None,:,None]`），
   每候选只取 **7 个对角线样本**而非完整 7×7=49 个；einsum 容忍退化形状（q 轴=1），
   bug 长期潜伏。方向模式 `_best_source` 与 Rust/JS 共享核都是完整 7×7——
   这就是「填充纹理与 Python 核心不一致」的最大来源。已修复为 `dx[None,None,None,:]`。
2. **Rust tkidx 窗口漏减 HALF**（patchmatch.rs `pm_best_source`）：known 蒙版采样
   窗口 `[ty..ty+6]` 应为 `[ty-3..ty+3]`，导致 SSD 在错误的已知位置上计算
   （SSD 量级差百倍）。已补 `- HALF`。
3. **Rust Dmap 未复刻 shim 的 uint8 量化**：py 参照的 `cv2.dilate(grad*known)`
   经 `_shared_core.dilate` shim 把输入 `astype(np.uint8)`（截断回绕）后再 max；
   Rust 用 f32 梯度直接 max（125.17 vs 125），优先级出现不应有的差异。
   已复刻 `((v as i32) & 0xFF)` 量化。

### 12.2 数值语义规范化（使 numpy 成为「可复刻规范实现」）

numpy 侧三处不可移植语义被规范化（算法结构零改动，输出一次性变化）：

| 项 | 原实现 | 规范化后 | 原因 |
|---|---|---|---|
| PRNG | PCG64（`default_rng(0)`） | mulberry32（`_Mulberry32(0)`，与 Rust/JS 逐位一致） | PCG64+SeedSequence 无法在 Rust 稳定复刻；Rust 原"mulberry32"实为 xorshift32 且 seed=0 恒输出 0（随机候选退化为固定 cand[0]，wasm 填充质量差的根源） |
| SSD | `np.einsum`（SIMD 分块累加，随规模变策略，不可跨端复刻） | 显式 f32 顺序累加（(p,q) C 序，每位置三通道和为一项） | einsum 内核不可复刻；掩码位置贡献精确 0，顺序累加与 Rust tkidx 累加逐位一致 |
| 平局/惩罚 | `argsort` 默认快排；tkn_sum(int64) 把惩罚链提升到 float64 | `kind="stable"`（行主序平局）；惩罚链全 f32 | 稳定排序与 Rust `sort_by` 一致；f64 提升不可复刻 |

另：`_patch_fill_loop` 从 `inpaint` 中提取为独立函数（`inpaint` 行为不变），
配合 `shared/tests/pm_parity.rs` + `data/_pmgen.py`/`data/_pmcmp.py` 构成
numpy↔Rust ROI 级逐字节孪生校验台（样本不入库，缺失时测试自动跳过）。

### 12.3 最终验收（feat/shared-wasm-core）

`python data/_diag3_repro.py`（TEXTCORE_BACKEND=0 numpy 参照 vs wasm 共享核）：

| 图 | CLEAN diff | MASK diff | RESULT diff |
|---|---|---|---|
| 1787767556635 | 0 | 0 | **0（0.0%）** |
| 1787767611178 | 0 | 0 | **0（0.0%）** |

即：后端 wasm 模式与 Python 核心（numpy 回退）全管线输出逐字节一致；
浏览器与后端共享同一 wasm 填充，填充纹理一致。浏览器与后端的残余差异只剩
去发光/蒙版手术的输入差（Rust 一体入口 vs cv2，§6.1/§6.3，蒙版 16px 级），
用户已判定「问题不大」；如需彻底消除，需让 Rust 蒙版手术逐字节复刻 cv2 链路。

> 注意：由于 PRNG/SSD 顺序/平局序规范化，本次之后所有端的填充像素相对历史
> 输出有一次一次性变化（算法结构不变，质量同族）；此后三端永久逐字节一致。

---

## 13. 浏览器端 vs 后端（同为 wasm）填充不一致——攻坚进度（2026-09-04 第三轮）

> 分支 `feat/shared-wasm-core`（HEAD = `6d33c40`）。`python-core-toggle` 已 push 到远程
> 并已 merge feat（`eeb5365`），远程两条分支的内容差异只剩这节的新工作。

### 13.1 先回答用户问题：两个开关后端各用什么

- **浏览器计算关闭（后端计算）**：默认 **wasm 共享核**。但后端编排是
  「cv2 去发光 + cv2 蒙版手术 + wasm 只做 PatchMatch 填充」，已与 Python 核心
  逐字节一致（repro 0/0/0，见 §12）。
- **浏览器计算开启**：走 Rust 一体入口 `erase_text_glyphs`（去发光+手术+填充
  全在 Rust 内）。**这才是离 Python 核心远的一端**，本节修的就是它内部的 Rust。

### 13.2 本轮找到并修复的三个分歧（提交 `6d33c40`）

诊断方法：后端与浏览器调同一份 wasm 导出，在 Python 侧用 wasmtime 复现浏览器流
（deglow_full_green_v2 → detect → erase_text_glyphs），与后端 erase_text 逐阶段对比。

1. **`pm_fill_roi` ROI 提取**（deglow.rs）：浏览器/后端 wasm 内的填充 ROI 在
   **未扩边坐标系**提取且 margin 用 round；后端 `patch_fill.inpaint` 是先
   replicate 扩边 4px、扩边坐标系、int 截断。小图 ROI 差一圈（86×83 vs 78×75），
   候选池不同 → 填充必然不同。已镜像重写（含写回坐标平移）。
2. **`mask_close` 语义**（deglow.rs）：后端 `_shared_core.morphology_ex` 的
   MORPH_CLOSE 分支**实现是反的**（`dilate(erode(x))`，实际是开运算；union 1200 →
   1186 比输入还小）。Rust 用真闭运算（→1265）。已让 Rust 同构复刻 shim 的实际
   行为（对齐优先，不按教科书"修好"——后端行为是用户验收过的）。⚠️ 这个 shim
   交换 bug 值得单独议题：修它会让**后端行为变化**，需用户确认。
3. **去发光强度**（browser/src/index.js）：后端 `erase_text` 入口对 v2 有
   `deglow_strength = max(s, 1.15)` 的过冲提升，浏览器直接透传 1.0。已在浏览器
   SDK 的 `erase()` 共享核路径加 `const ds = Math.max(opts.deglowStrength ?? 1.0, 1.15)`
   （deglowFullGreenV2 与 eraseTextGlyphs 两处 + cfg 回显）。

### 13.3 修复后验收状态

| 图 | 浏览器流 vs 后端 RESULT diff | MASK diff |
|---|---|---|
| 1787767611178 | **0（逐字节一致）** | 0 |
| 1787767556635 | 洞内 1384px / 洞外 1868px（去发光输入差） | **0** |

后端回归保持全绿（pytest 5/5；`data/_diag3_repro.py` 仍 0/0/0）。浏览器 bundle 已重建。

### 13.4 剩余唯一分歧：img1（1787767556635）的 Rust 去发光「fb 重建场」

- 分区统计（用 `_deglow_full_green_v2(debug=True)` 拿 zone/m_zone/text_stroke）：
  差异 2035px **100% 落在 fb（背景重建区）**，幅值 p50=1、p90=6、max=45（通道和）；
  减绿区（m_zone）、保护圈、text_stroke、zone 外**全部逐字节一致**。
- img1 走最复杂路径：暖背景（d_warm=1.0>0）+ zone 占 98% + 测地 B 场 +
  调和背景（300 轮 Jacobi 多重网格）+ 结构强度混合 + detail 回贴 + chroma_keep。
  img2 走纯减绿路径（已一致）。
- deglow.rs 第 17 行注释自认：背景场的 resize/gaussian 与 cv2 **数值不同**。
  Rust 的 geodesic Dijkstra 已是 f32 精心镜像（含种子坐标修复），剩余嫌疑按优先级：
  ①`resize_area`/`resize_cubic`/`gaussian_blur` 与 cv2 的 INTER_AREA/INTER_CUBIC/
  GaussianBlur 位级差异（最可能）；②调和 Jacobi 的迭代/边界细节；③堆平局序。

### 13.5 下一步操作指南（接手人照做）

1. **建立 B 场跨端对比台**（方法同填充孪生台 §12.2）：
   a. Python dumper：复刻 `_deglow_full_green_v2` 到暖路径起点，把 `rgb/zone/
      ring_clean`（`_ring_clean = (~zone)&(10≤dist≤26)&(greenness≤6)`）写
      `data/_pmparity/deglow_case.bin`（h,w u32 LE + rgb f32 + zone u8 + ring u8）；
      同时落盘 cv2 的 `_geodesic_background(rgb, geo_mask, extra=[R-G, G-B],
      extra_src=ring_clean)` 的 B/D_rg/D_gb 为 .npy 参照。
   b. Rust 测试钩子：lib.rs 加 `pub mod deglow_debug`（fn 包一层
      `crate::deglow::geodesic_background`，先把它改 `pub(crate)`），测试读
      case.bin → 调用 → 写 deglow_rust_{B,Drg,Dgb}.bin。
      ⚠️ 我已试过这条路（改 pub(crate) + lib 钩子 + tests/deglow_stage_dump.rs），
      编译到一半按用户指示放弃，**文件已全部还原/删除**——接手人重建即可，
      注意 `Vec<f32>.copy_from_slice(&[u8])` 要用 chunks_exact(4) 逐个 from_le_bytes。
   c. 逐位对比 → 差在哪层（源映射 / resize / gaussian）→ 修对应 Rust 函数到
      与 cv2 位级一致 → 层层往下（_harmonic_background、_S 结构场、detail、
      chroma_keep、软混合），直到 img1 端到端 RESULT diff=0。
2. 若 resize/gaussian 位级复刻代价过大，备选方案：**把浏览器端也改成后端编排**
   （cv-bridge 已有 opencv.js 的 resize/GaussianBlur/distanceTransform，
   patchmatchInpaintShared 做填充）——但这是大改，先试 1。
3. 修完 img1 后：`node --check` + pytest + `data/_diag3_repro.py` 回归，
   再跑 13.3 的浏览器流 vs 后端对比脚本（在 git log 6d33c40 的 commit message
   里有等价 inline 版本），三项全零后在 feat 分支提交并通知用户 merge。
4. 遗留小议题：`_shared_core.morphology_ex` CLOSE/OPEN 交换（13.2-2）——修它会
   改变后端行为，应作为独立改动征求用户意见，不要顺手改。

### 13.6 本轮操作时间线（供审计）

1. push `python-core-toggle`（远程新建）；切到 `feat/shared-wasm-core`。
2. 阶段对比脚本：cv2 链 vs Rust（deglow/detect2/手术+膨胀）→ 发现手术需补膨胀
   才公平、img1 去发光 max=18。
3. spy 抓后端内部（detect×3、deglow kwargs、surgery 链）→ 抓到 **strength=1.15**
   （`erase_text` L218 的 max 提升）。
4. 追 img2 的 16px 蒙版差 → `union+close` 阶段实锤 **morphologyEx 交换 bug**
   （union 1200：真 close 1265 vs shim 1186）。
5. 修 Rust `mask_close`（镜像 shim 行为）+ 浏览器 `ds=max(s,1.15)` →
   img2 端到端归零。
6. img1 剩 fb 重建场差 → 分区统计定位（2035px 全在 fb、减绿/保护圈全一致）
   → 读 `geodesic_sources`（已良好镜像）→ 判定剩余为 resize/gaussian 位级差。
7. 建 deglow 阶段转储测试（未完成，按用户指示中止）→ 还原 lib.rs/deglow.rs、
   删 tests/deglow_stage_dump.rs（**pub(crate) 与钩子已撤**，HEAD=6d33c40 干净）。

---

## 14. 终局：fb 重建场位级对齐不可达 → 后端去发光切回 wasm-first（2026-09-04 第四轮）

### 14.1 结论先行

- **浏览器流 vs 后端现已端到端逐字节一致**（两张验证图 RESULT diff=0、MASK diff=0）。
  实现方式：`_erase_deglow_v2` 第 2 步去发光从「cv2 路径」改回 **wasm-first**
  （调 `_shared_core.deglow_full_green_v2`，即与浏览器同一个 Rust 入口），
  wasm 不可用时回退 cv2（`TEXTCORE_BACKEND=0` 语义不变）。
- **代价（已知且接受）**：Python 核心（纯 cv2/numpy 路径）与 wasm 在 fb 重建场上
  有数值差——img1 实测 3252px、通道和 max=52、mean=1.22（单通道 p50≈1，视觉不可见）；
  img2 完全 0；MASK 恒 0。`_diag3_repro.py` 的角色从「回归门禁(0/0/0)」变为
  「py-vs-wasm 差值测量」，3252px 是新基线，不是回归。

### 14.2 为什么位级对齐不可达（本轮证据）

用 wasm debug 钩子 `deglow_debug_geodesic`（lib.rs，导出测地链全部中间量：
lum/rz/rgb_s/sy/sx/b_s/b_up/b_sm/e_s/e_f）+ 对比台脚本
（`scripts/dump_deglow_case.py` + `scripts/compare_deglow_B.py`，样本在
`data/_pmparity/`，不入库）逐子层对比 cv2：

| 子层 | 差异 |
|---|---|
| lum（INTER_AREA f32 下采样） | 386/1443 px，±0.5 |
| rgb_s（INTER_AREA） | 4291/4329 px，±0.5 |
| sy/sx（Dijkstra 源映射） | 路由大面积翻转（上游 ±0.5 传导） |
| b_up（INTER_CUBIC f32 上采样） | 全像素 |
| b_sm（GaussianBlur σ4） | 全像素，max 41 |

分叉从第一个 cv2 float 算子就开始。查 OpenCV 5.0 源码证实：**CV_32F 的
resize/GaussianBlur 走 SIMD 路径（`v_muladd`=FMA 融合乘加、按行指针 8-float
对齐决定走向量还是标量尾巴、AVX2 8-lane 运行时分派）**，OpenCV 官方只对
8u/16u 固定点 resize 路径保证位精确。手写 Rust（或 opencv.js，wasm 无 FMA、
对齐不同）永远差最后几个 ULP，经 Dijkstra 路由放大后散布为 ±1~6 的像素差。
**结论：想逐字节一致只能两端跑同一份实现 → 后端切 wasm。**

### 14.3 Dijkstra 侧已确认无恙

Rust `geodesic_sources` 的堆平局序（`Reverse(HeapItem{dist,node})`，node=y*w2+x）
与 Python `heapq` 的 `(d,y,x)` 元组序一致；路由翻转纯由上游 lum/rgb_s 的 ULP 差
引起，不是堆实现 bug。

### 14.4 新架构事实（当前 HEAD 起）

- 浏览器计算开 = 后端计算开：两者都执行「Rust 去发光 + Rust 手术 + Rust
  PatchMatch」（后端经 `_shared_core` 分步调用，浏览器经 `erase_text_glyphs`
  一体入口；输入蒙版相同 ⇒ 输出逐字节相同，已验证）。
- Python 核心（`TEXTCORE_BACKEND=0`）：完整 cv2/numpy 路径，仍是可用回退，
  与 wasm 在 fb 场差 3252px（见 14.1）。
- 浏览器 wasm 从 `shared/build/textcore.wasm` 运行时 fetch，后端 wasmtime 加载
  同一文件——单一真源不变。

### 14.5 本轮提交内容

- `text_eraser/eraser.py`：`_erase_deglow_v2` 去发光 wasm-first + cv2 回退；
  补 `_shared_core` 导入。
- `shared/src/deglow.rs`：`geodesic_background` 加 `Option<&mut GeoDbg>` 调试
  中间量（生产路径传 None，零开销）；`pub(crate)`。
- `shared/src/lib.rs`：debug 导出 `deglow_debug_geodesic`（仅对比台使用）。
- `shared/build/textcore.wasm`：重编译（含 debug 导出；生产行为零变化）。
- `scripts/dump_deglow_case.py`、`scripts/compare_deglow_B.py`：B 场跨端
  对比台（先跑 dumper 再跑 compare，样本落 `data/_pmparity/`）。

### 14.6 验收记录（2026-09-04）

- 浏览器流 vs 后端：img1/img2 RESULT diff=0、MASK diff=0（逐字节一致）。
- pytest 5/5 通过。
- `_diag3_repro.py`：img2 全 0；img1 MASK=0、RESULT 3252px（py-vs-wasm 已知差，
  非 regression）。
- 浏览器运行时 fetch 该 wasm；本轮未改 JS，无需重建 te-bundle。
