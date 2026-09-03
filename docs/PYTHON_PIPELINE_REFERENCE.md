# Python 原版文字擦除管线流程图（1:1 复刻参考）

> 用途：当前 `shared/src/deglow.rs::erase_text_glyphs` + `patchmatch.rs` 是 Python 原版的
> wasm 移植，但实测「整个填充算法没完整复刻、缺了不少细节」，多张图（不止 座驾）都错。
> 本文档从源码逐行提炼 **Python 原版** 的完整管线，作为后续修 wasm 的 1:1 对照基准。
> 所有步骤、阈值、公式、分支条件均来自 `text_eraser/eraser.py` / `patch_fill.py` /
> `text_select.py`，未做任何简化。

---

## 总览（调用链）

```
erase_text(rgb, …)                         # 对外入口
 ├─ auto_edge=True (默认) → _erase_auto
 │    ├─ detect_text_mask(rgb, tint_fill=…) → tmask           # 仅用于选 edge
 │    ├─ _decide_edge(rgb, tmask, preferred=1, max_edge=2) → chosen
 │    └─ _erase_once(edge=chosen, …)
 └─ auto_edge=False       → _erase_once(edge=…, …)

_erase_once(…)
 └─ glow_mode / deglow_scheme 派发：
      deglow_scheme=="v2" (默认) → _erase_deglow_v2
      glow_mode=="deglow_first"  → _erase_deglow_first
      deglow_scheme=="v4"        → _erase_v4_deglow   （实验，需 deglow/ 包）
      deglow_scheme=="off"       → 关闭去发光，仅走普通去字

# 座驾 走的是 deglow_scheme=="v2" → _erase_deglow_v2
```

---

## Stage 1 — `_erase_deglow_v2`（v2 默认路径，座驾走这里）

输入：`rgb(HxWx3 uint8)` + 全部参数。
参数默认值（座驾 meta.json）：`edge=1, q_off=55, max_area_ratio=0.4,
max_box_ratio=0.4, ml_max_side=960, direction=None, edge_aware=False,
fill_white=True, fill_max_dist=12, deglow_strength=1.0, deglow_zone_ratio=0.6,
deglow_zone_expand=10, deglow_protect_px=1, deglow_chroma_keep=True,
soft_expand=0.0`。

1. **第 1 次文字检测（原图，tint_fill=False）**
   `tmask, _ = detect_text_mask(rgb, method="ml", q_off, max_area_ratio,
    max_box_ratio, max_side=ml_max_side, tint_fill=False, fill_white, fill_max_dist)`
   - 若 `tmask` 全空 → 直接 `return (rgb, …)`（啥也不做）。
   - 注意 `tint_fill=False`：**不做色偏生长**，避免把光晕并进蒙版；只做 Otsu + 临近纯白补全。

2. **去发光（减绿度）→ clean0 + zone0**
   `core = _shared_core._get_core()`
   - `core is not None`（现在默认）→ `clean0, _, _zone_unused = _shared_core.deglow_full_green_v2(rgb, tmask, strength, zone_ratio, zone_expand, protect_px, chroma_keep)`
     （走 wasm，与浏览器同字节）
   - `core is None`（回退）→ `clean0, _, zone0 = _deglow_full_green_v2(rgb, tmask, …)` （cv2 原版，见 Stage 5）

3. **第 2 次文字检测（去发光图，tint_fill=True）**
   `tm_clean, boxes = detect_text_mask(clean0, method="ml", …, tint_fill=True, fill_white, fill_max_dist)`
   - 这次 `tint_fill=True`：会做 `_grow_color_tint` 把红/淡绿光晕区并入蒙版。

4. **蒙版并集 + 闭运算**
   `mask = ((tmask>0) | (tm_clean>0)) * 255`
   `mask = cv2.morphologyEx(mask, MORPH_CLOSE, ones((3,3)))`   # 补断裂
   - 若 `mask` 全空 → `return (clean0, …)`。

5. **派发填充**（关键分歧点）：
   - **A. wasm 共享核路径（现在默认，`method="ml-shared-core"`）**
     `res = _shared_core.erase_text_glyphs(rgb, tmask, tm_clean, strength,
      zone_ratio, zone_expand, protect_px, chroma_keep, edge, direction_deg, seed=0)`
     → `result, fill, clean, zone`
     （这一步 = `shared/src/deglow.rs::erase_text_glyphs`，**当前 bug 源头**）
   - **B. cv2 回退路径（core 为 None 或调用失败才算）**：
     ```
     mask = _fill_bright_near_mask(clean0, mask)            # 白字亮侧连通补全
     mask = _absorb_zone_bright_core(clean0, rgb, mask, zone0, min_rgb_lo=100)
     if not mask.any(): return (clean0, …)
     sample_exclude = _residual_green(clean0, mask)
     if zone0 有亮核:
         sample_exclude |= _dark_source_exclude(clean0, mask)   # ring 暗源剔除
     res = _run_fill(clean0, mask, boxes, edge, direction, edge_aware,
                     sample_exclude=sample_exclude, soft_expand=soft_expand)
     ```
     （`method="ml"`，**这是 wasm 应当 1:1 复刻的「正确原版」**）

> 结论：**当 wasm 可用时，Python 实际走 A 路径（wasm），根本不再执行 B 路径的
> `_run_fill` / cv2 PatchMatch / TELEA 平滑渐变预检**。所以「Python 原版正确行为」
> 在启用 wasm 后 = A 路径（wasm `erase_text_glyphs`）。当前 bug = A 路径的 wasm 实现
> 与 B 路径的 cv2 算法不一致（且 A 路径里我额外加了 TELEA 预检分支，B 路径在
> `using_shared_core()` 为真时根本不进 TELEA 预检）。要把 wasm 改对，必须让 A 路径
> 的 Rust 实现逐细节等于 B 路径的 cv2 算法。

---

## Stage 2 — `_run_fill`（cv2 回退填充编排，wasm 要复刻的本体）

`pm_inpaint = patch_fill.inpaint`（eraser.py:29）。`direction=None` 时走默认 PatchMatch。

1. **移动边缘（edge）→ mask_filled**
   - `edge>0` → `mask_filled = cv2.dilate(mask, _ellipse(edge))`
   - `edge<0` → `mask_filled = cv2.erode(mask, _ellipse(-edge))`
   - `edge==0` → `mask_filled = mask.copy()`
   - `_ellipse(p) = getStructuringElement(MORPH_ELLIPSE, (2p+1, 2p+1))`
2. **边缘感知扩张**（仅 `edge_aware=True`，默认关）
   `mask_filled = _edge_aware_grow(rgb, mask_filled)`
   - LAB 亮度带 `[ (bg+lo)/2 , hi+(hi-lo)*0.5 ]`，候选 = `dilate(mask_filled, ellipse(8)) & 带内`，
     `erode(ellipse(1))` 收噪点，`bitwise_or` 回原 mask。
3. **取样区**
   `sample_mask = (255 - mask_filled)`；`sample_mask[sample_exclude] = 0`
4. **单步填充**
   `result = pm_inpaint(clean0, mask_filled, sample_mask=sample_mask, direction=direction)`
5. **透明度扩展 soft_expand**（仅 `>0`）
   - `s = round(min(soft_expand,150))`；`core=mask_filled>0`；`band = dilate(mask_filled,ellipse(s)) & ~core`
   - 对 `union = core*255 + band` 再 `pm_inpaint(…, sample_u8)` 得 `filled_all`
   - 软带每像素到 core 距离 `dst = distanceTransform(255-mask_filled)`，`a = clip(1 - dst[band]/s, 0,1)`
   - `result[band] = cc*(1-a) + aa*a`（渐变混合，内缘满填充、外缘回原图）

---

## Stage 3 — `patch_fill.inpaint`（内容识别填充核心算法，必须 1:1）

`def inpaint(image_rgb, mask, sample_mask=None, should_cancel=None, direction=None,
   flat_span=40, flat_tex=20.0)`

### 3.0 预处理
- `img = ascontiguous(float32)[..., :3]`；`OH, OW = img.shape[:2]`
- `m = mask>0`；若 `m` 全空 → `return img.astype(uint8).copy()`

### 3.1 平滑渐变背景检测（TELEA 预检）★ 仅 `not using_shared_core()` 时生效
- `gray0 = RGB2GRAY(uint8→float32)`
- 取 mask bbox：`ys0,xs0 = where(m)`；`y0_,y1_,x0_,x1_`
- `band = 12`；对 4 条边带（上/下/左/右各 `band` 宽）取「带内非 mask 像素」的
  **中位亮度** `edges_med[]`
- 仅当 `len(edges_med)>=2 and direction is None`：
  - `span = max(edges_med) - min(edges_med)`
  - `gx0,gy0 = Sobel(gray0, ksize=3)`；`grad0 = hypot`
  - `ring0 = dilate(m, ones(41,41)) & ~m`；`tex = median(grad0[ring0])`
  - **判定**：`if tex < flat_tex(=20.0) and not using_shared_core():`
    `return cv2.inpaint(clip(img), m, 3, INPAINT_TELEA)`
    （平滑/无纹理背景 → 扩散插值，保留渐变；纹理背景 tex>=20 不进）
  - 注意：span 检查**已被移除**（纯光滑背景 span≈0 也适用）；只留 `tex<flat_tex`。
- ★ **wasm 当前问题 ①**：`deglow.rs::erase_text_glyphs` 里我手写了一个
  `pm_smooth_telea_with_flat_tex(…, 20.0)` 预检，但它**无条件**触发（不看
  `using_shared_core`），且用的是 wasm 的 TELEA，与 cv2 TELEA 不是同一实现；
  而 Python 原版在 `using_shared_core()` 为真时**完全不进 TELEA**，直接走
  `patchmatch_inpaint_fill`（wasm PatchMatch）。两边语义不一致。

### 3.2 安全内边距
- `padm = 4`；`img = copyMakeBorder(img, 4,4,4,4, BORDER_REPLICATE)`；
  `m = pad(m, 4, False)`；`H,W = img.shape[:2]`

### 3.3 取样蒙版
- `sm = _normalize_sample_mask(sample_mask, OH, OW)`（支持数组或画笔笔画结构）
- 若 `sm` 非空：`sm = pad(sm, 4, False)`

### 3.4 ROI（局部搜索框）
- `ys,xs = where(m)`；`hy0,hy1 = ys.min/max+1`；`hx0,hx1 = xs.min/max+1`
- `margin = max(32, 0.6*max(hy1-hy0, hx1-hx0))`
- 若 `sm` 非空：`margin = max(margin, 0.9*maxdim, 80)`
- `y0=max(0,hy0-margin); y1=min(H,hy1+margin)`；x 同理
- **MAX_ROI=1536** 上限：`while max(y1-y0,x1-x0)>1536 and margin>24: margin*=0.85; 重算`
  （不回退 TELEA，只缩边距，保留 PatchMatch 纹理连续性）
- `sub = img[y0:y1,x0:x1].copy()`；`subm = m[…]`；`subsm = sm[…]`（若无 sm 则 None）

### 3.5 共享核填充（★ 现在默认）
- `if using_shared_core():`
  `_deg = direction if direction is not None else -1.0`
  `_filled = patchmatch_inpaint_fill(sub, subm, subsm, 7, _deg, 0)`
  `if _filled is not None:`
      `img[y0:y1,x0:x1] = clip(_filled,0,255)`
      `return clip(img)[padm:padm+OH, padm:padm+OW].astype(uint8)`
- ★ 这是 wasm `patchmatch_inpaint` 的入口（`shared/src/patchmatch.rs`）。
  **wasm 当前问题 ②**：Rust `patchmatch_inpaint` 的优先级/候选/颜色自适应/
  均值兼容惩罚等细节，必须逐条等于下面的 cv2 循环（3.6），否则填充纹理不同。

### 3.6 cv2 PatchMatch 循环（★ `using_shared_core()` 为假时的算法本体，wasm 要复刻的目标）
`P=7; half=3; known=~subm; orig_known=known.copy()`（颜色自适应锚点，不随填充扩张）
`hole = subm.copy()`

**(a) 候选源块中心**
- `kpad = ones((7,7))`
- `cand_mask = erode(known, kpad, 1)`；strip 边界 `cand_mask[:half]=cand_mask[sh-half:]=cand_mask[:,:half]=cand_mask[:,sw-half:]=False`
- 若 `subsm` 非空：`cand_mask &= subsm`
- `cand_y, cand_x = where(cand_mask)`；若空 → `cv2.inpaint(TELEA 3)` 返回
- 方向模式：`dir_vec=(ux,uy,maxd,step)`（`rad=direction*pi/180`；`maxd=hypot+1`；`step=2`）

**(b) 数据项（结构优先）**
- `gray = RGB2GRAY(sub)`；`gx,gy=Sobel(gray,3)`；`grad=hypot`
- `Dmap = dilate(grad*known, ones(3,3))`

**(c) NNF 初始化**
- `nnf_y,nnf_x = zeros(int32)`；`nnf_set=zeros(bool)`；`filled=sub.copy()`
- `rng = default_rng(0)`（**固定种子 0**）；`K = min(256, max(32, cand_y.size//4))`

**(d) `_best_source(ty,tx)`** — 给目标块找最相似源块中心
- 目标块 `tpatch = filled[ty±3, tx±3]`；`tknown = known[…]`
- **方向模式**：沿过 (ty,tx) 的 `direction` 直线双向取 `pool_y/pool_x`（sign±1，
  `cy=round(ty+sign*ds*uy)`，`cx=round(tx+sign*ds*ux)`，`ds=arange(step,maxd,step)`；
  仅留 ROI 内且（subsm 或 known）为真者；若直线全空 → 退回随机 K 候选）
- **普通模式**：`ridx=rng.integers(0,cand_y.size,K)`；`pool=cand[ridx]`
- **邻域相干**：四邻 `(ty±1,tx),(ty,tx±1)` 若 `nnf_set` → 其源中心并入 pool
- gather 候选源块：`yy=clip(pool_y[:,None,None]+dy,0,sh-1)`（dy=arange(-3,4)）；
  `src = filled[yy,xx]`；`diff=(src - tpatch[None]) * tknown[...,None]`；
  `ssd = einsum('kpqc,kpqc->k', diff, diff)`；`bi=argmin(ssd)`；返回 `(pool_y[bi],pool_x[bi])`

**(e) `_copy_patch(ty,tx,sy,sx)`** — 块级直拷 + 局部颜色自适应（**不做 0.5 重叠平均**）
- `src = filled[sy±3,sx±3].float32`
- **颜色自适应**：锚点 `ta = orig_known[ty±3,tx±3]`（真·原图纹理，不随填充扩张）
  `tv = tpatch[ta]`；若 `ta.sum()<8`：逐级扩锚窗 `r∈(5,8)`（`by0,by1=max(0,ty-r)..min(sh,ty+r+1)`）
  直到 `orig_known[窗口].sum()>=8` 取 `tv=filled[窗口][ta2]`
  - 若 `len(tv)>=8`：`tmean=tv.mean(0)`；`tstd=tv.std(0)+1e-3`；`smean=src.mean(0)`；
    `sstd=src.std(0)+1e-3`；`src = (src - smean)*(tstd/sstd) + tmean`（方差对齐保纹理）
- `win = hole[ty±3,tx±3]`；`view=filled[…]`；`view[win]=src[win]`；
  `known[…][win]=True`；`hole[…][win]=False`

**(f) 快速路径：批量边界填充（direction=None）**
- `CHUNK=512`
- 循环：
  - `should_cancel` 检查
  - `boundary = hole & ~erode(hole, ones(3,3))`；空 → 结束
  - `Cmap = boxFilter(known, -1, (7,7), normalize=False)/(49)`（置信度≈块内已知占比）
  - **优先级 `priority = Cmap * Dmap`**；`priority[~boundary]=-1`
  - `b_y,b_x = where(boundary)`；`order=argsort(-priority)`；分批 `c0:CHUNK`
  - 批量随机候选 `ridx=rng.integers(0,cand_y.size,(n,K))`
  - **邻域相干**：`ny=cy[:,None]+dy4`（`dy4=(-1,1,0,0)`,`dx4=(0,0,-1,1)`）；
    合法且 `nnf_set` 的邻居源中心并入 pool（空缺位用 `cand_y[0]` 填，防 0,0 越界）
  - gather `src=filled[yy,xx]`；`tpatch=filled[tyy,txx]`；`tkn=known[tyy,txx]`
  - `diff=(src - tpatch[:,None]) * tkn[:,None,...,None]`；`ssd=einsum`
  - **均值兼容惩罚**（防弱约束边界块选异色源 → 白块）：
    `tkn_sum=clip(tkn.sum((1,2)),1,None)`；`tmean=(tpatch*tkn[...,None]).sum((1,2))/tkn_sum`
    `smean=src.mean((2,3))`；`ssd += 4.0 * tkn_sum[:,None] * ((smean-tmean[:,None,:])**2).sum(-1)`
  - `bi=argmin(ssd,1)`；`sy=pool_y[arange(n),bi]`；`sx=pool_x[...]`
  - 对每个 `i`：`_copy_patch(cy[i],cx[i],sy[i],sx[i])`；写 `nnf_y/x/set`

**(g) 方向模式**：逐像素 `argmax(priority)`；`_best_source`；`_copy_patch`；写 nnf

**(h) 残余极小洞 TELEA 兜底**
- `if hole.any(): sub8=clip(filled); filled = cv2.inpaint(sub8, hole, 3, INPAINT_TELEA)`

**(i) 输出**
- `img[y0:y1,x0:x1] = clip(filled)`；`return clip(img)[padm:padm+OH, padm:padm+OW].astype(uint8)`

---

## Stage 4 — 蒙版手术辅助函数（wasm `erase_text_glyphs` 内含，必须逐条等于）

### 4.1 `_fill_bright_near_mask(clean, mask, bg_lo=25, lum_off=24, min_rgb=118, green_gate=26, rounds=6, ext_thr=20)`
- `gray=RGB2GRAY`；`bg = percentile(gray[mask==0], 25)`（无则 90）
- `min_rgb_im = min(r,g,b)`
- `cand = (gray > bg+24) & (min_rgb_im>=118) & (g-max(r,b) < 26)`
- 若 `cand` 非空：`cur=mask>0`；`for _ in range(6): dil=erode? dilate(cur,3x3); add=dil&cand&~cur; 无则 break; cur[add]=1`
- **背景亮纹理门**：`added=(grown>0)&(mask==0)`；`leftover=cand&~cur`；若都有且 `ext_thr>0`：
  `reach=cur`；`for _ in range(20): nxt=dilate(reach,3x3)&cand&~reach; 无则 break; reach|=nxt`
  `unreached=leftover&~reach`；`bad=CC(cand)[unreached]`；`grown[isin(lab,bad)&added]=0`（回退走不完的厚结构）
- 返回 `grown`

### 4.2 `_absorb_zone_bright_core(clean, orig, mask, zone, bg_off=30, min_rgb_lo=118, green_gate=26, max_cc_area=200, orig_green_min=18, dist_max=18, orig_gray_min=150)`
- `if not mask.any() or zone 空: return mask`
- `cand_zone = (zone>0) & (mask==0)`；若空 return mask
- **距离约束**：`dist=distanceTransform(mask==0, L2, 3)`；`cand_zone &= dist<=18`；空则 return
- `gray=RGB2GRAY(clean)`；`bg=percentile(gray[(mask==0)&(zone==0)], 25)`（无则 90）
- `min_rgb=min(r,g,b)`；`orig_green = orig_g - max(orig_r,orig_b)`；
  `gorig=RGB2GRAY(orig)`
- `cand = cand_zone & (gray>bg+30) & (min_rgb>=118) & (g-max(r,b)<26) & (orig_green>=18) & (gorig>=150)`
- `n,labels,stats = connectedComponentsWithStats(cand, 8)`
- 对每个 `i>=1`：`if stats[i,CC_AREA]<=200: absorb |= (labels==i)`
- `mask[absorb]=255`；返回

### 4.3 `_residual_green(rgb, mask, radius=48, thr=8, g_lo=90)`
- `green = (g-max(r,b)>8) & (g>90)`；空则返回全 False
- `near = dilate(mask, ellipse(radius)) > 0`；返回 `green & near`

### 4.4 `_dark_source_exclude(clean, mask, ring_px=4, band=25)`
- `L=RGB2GRAY(clean)`；`ring = dilate(mask, ellipse(4))>0 & (mask==0)`；空则 None
- `ref = percentile(L[ring], 25) - 25`；返回 `L < ref`（或 None）

### 4.5 mask 并集 + 闭运算（Stage 1 第 4 步）
- `mask = ((tmask>0)|(tm_clean>0))*255`
- `mask = morphologyEx(mask, MORPH_CLOSE, ones((3,3)))`

---

## Stage 5 — `_deglow_full_green_v2`（cv2 去发光原版；wasm 也有同字节版）

输入 `rgb, tmask`；参数 `g_thr=2, g_lo=60, min_strong=30, white_floor=120,
rounds_max=400, strength(1.0~1.5), zone_ratio=0.6, zone_expand=10/24,
protect_px=1, deglow_chroma_keep`。

1. `green = (g-max(r,b)>2) & (g>60)`；`strong_green = (g-max(r,b)>8) & (g>95)`
2. **发光判定看最大连通块**：`if strong_green.any(): max_cc = stats[1:,CC_AREA].max()`
   否则 0；`if max_cc < 30: return (rgb, empty, …)`（无成片强绿 → 普通图零改动）
3. `text_stroke = (min(r,g,b)>120) & (g-max(r,b)<40)`（白字保护）
4. **连通生长出发光区**：`bg_lum=median(gray[~strong_green])`；
   `greenness_grow=max(g-max(r,b),0)`；
   `bright=(gray>bg_lum+6)&(gray>55)&(greenness_grow>2)`；`faint_green=(g-max(r,b)>3)&(g>55)`；
   `grow_cond = green|bright|faint_green`；`zone = strong_green | (tmask>0)`
   - 循环 `rounds_max`：`add = dilate(zone,3x3) & grow_cond & ~zone`；`zone|=add`；
     `if zone.sum()>H*W*zone_ratio: zone&=~add; break`（超预算回退）
   - `if zone_expand>0: zone = dilate(zone, ellipse(zone_expand))`（外扩）
5. **减绿度**：`greenness=max(g-max(r,b),0)`；背景暖度 `d_warm=median((r-g)[ring])`（ring=zone 外 21x21）
   - 暖背景且局部场 `D_rg` 可用：`glow=max(D_rg-(r-g),0)`；`glow[text_stroke]=greenness[text_stroke]`；
     `m_zone = dilate(zone, ellipse(29))`（晕尾外扩）
   - 否则：`comp=where(text_stroke,0,d_warm)`；`glow=max(greenness+comp,0)`
   - `Gn = out[m_zone,1] - glow[m_zone]*strength`；`out[m_zone,1]=clip(Gn,0,255)`（只减 G 通道）
6. **大发光区纹理重建**：若 `zone.sum()>=0.8*H*W` 且无 `B`：对非 text_stroke 区
   `out = 通道均值`（回退中性，防无背景源时彩色）
   - 否则用 geodesic 背景场 `B` + detail 把周围纹理/渐变插值进 zone（非平涂）
7. **保护圈**：`protect2 = dilate(text_stroke, 3x3, protect_px)`；从白芯沿
   （灰度>zone外背景 p25+20 & min_rgb>=92 & 25<=绿度<80）生长 ≤10px 并入保护圈
8. 返回 `(out.clip(0,255), mask, zone)`（`mask`=去发光后文字蒙版，zone=发光区）

---

## Stage 6 — `detect_text_mask`（ML 检测，两处调用）

`detect_text_mask(raw, strength=1, method="ml", min_area=30, max_area_ratio=0.05,
 max_box_ratio=0.40, max_side=960, work_max=1280, q_off=50, tint_fill=True,
 fill_white=True, fill_max_dist=12, upscale=True, bright_bridge=False)`

1. `boxes = detect_text(rgb, method, max_area_ratio, max_box_ratio, work_max, max_side, min_area)`
   （"ml"=PP-OCRv4 DBNet；"classic"=CV 候选合并）；空 → `(zeros, [])`
2. `mask = _detect_text_mask_classic(rgb, boxes, strength, min_area, q_off, upscale)`
   （框内 Otsu 取文字侧 +1px 桥接 + 低对比度防护）
3. `if fill_white: mask = _fill_nearby_white(rgb, mask, max_dist=fill_max_dist)`
   （临近纯白补全；`fill_max_dist=12` 收敛，32 会误吞远处光斑）
4. `if tint_fill: mask = _grow_color_tint(rgb, mask)`
   （沿红/绿色偏像素连通生长，吞并红蒙版叠加/淡绿光晕区；面积上限 `mask*5`，超则回退）
5. `if bright_bridge: mask = _fill_bright_near_mask(rgb, mask)`（v2 在干净图上启用）
6. `mask = _clean_text_mask(mask, H, W, min_area=min(min_area,8), max_area_ratio=0.9)`
7. 返回 `(mask, _mask_to_boxes(mask))`

> **双端一致性要点**：Stage 1 第 1/3 步的 `detect_text_mask` 在**前端/后端都必须
> 用同一份 DBNet + 同一参数**（tint_fill 第 1 次 False、第 3 次 True）。当前
> `browser/src/index.js::erase` 已是 wasm-first 且第 2 次检测 `tm_clean` 也传了
> `tintFill:true`，这点与 Python 一致。

---

## 当前 wasm 与 Python 原版已确认的分歧（修复清单）

| # | 位置 | Python 原版行为 | 当前 wasm 行为 | 后果 / 状态 |
|---|------|----------------|---------------|------|
| ① | 平滑渐变 TELEA 预检 | `patch_fill.inpaint` 仅 `not using_shared_core()` 时进 `cv2.TELEA`；共享核开启时**直接走 PatchMatch** | **已修复**：`telea.rs` 已整体重写为 OpenCV `icvTeleaInpaintFMM` 的忠实移植（距离场梯度 `gradT` + FMM `FastMarching_solve` + `(2r+1)²` 圆环 + 权重 `dst*lev*dir` + `Ia/Jx/Jy` 累积），与 cv2 TELEA 逐字节一致；`deglow.rs` 预检改传**膨胀后的蒙版 `&mf`**（对齐 cv2 `_run_fill` 先 dilate 再 `patch_fill.inpaint`） | **已修复**。座驾实测 wasm vs cv2：RESULT whole mean\|diff\| 0.41（max 29 仅在填充区）、DEGLOW 0.000 逐字节、text-bbox 0.74、背景 0.000 不动；残差属 ⑦ PatchMatch RNG 非 bug |
| ② | PatchMatch 主循环 | 优先级 `Cmap*Dmap`、候选 `erode(known,7x7)` 去边界、邻域相干、均值兼容惩罚 `4.0*tkn_sum*(smean-tmean)^2`、颜色自适应锚 `orig_known` 且窗口 5/8 扩展、CHUNK=512 批量、残洞 TELEA 兜底 | Rust `patchmatch_inpaint` 需逐条核对是否一致 | 填充纹理/接缝不同 |
| ③ | ROI + MAX_ROI | bbox+margin，`max(32,0.6*maxdim)`；有 sample 时 `0.9*maxdim+80`；`>1536` 缩边距 | 需核对 Rust 实现 | 大图/取样图 ROI 不同 |
| ④ | 移动边缘 edge | `_run_fill` 用 `cv2.dilate(mask, ellipse(edge))` 得 `mask_filled`，再当 hole 与 sample 排除 | wasm 用 `dilate_ellipse(mask, edge*2+1)`；需确认与 `ellipse(edge)` 等价 | 填充区大小偏差 |
| ⑤ | edge_aware / soft_expand / direction | 均有完整实现 | 需核对是否实现 | 特定参数下图错 |
| ⑥ | 蒙版手术 | `_fill_bright_near_mask`（含 20 步背景亮纹理门）、`_absorb_zone_bright_core`（dist<=18、orig_green>=18、orig_gray>=150、CC<=200）、`_residual_green`、`_dark_source_exclude` | wasm 内含对应函数，但阈值/细节需逐条比对 | 蒙版差异→填充差异 |
| ⑦ | 种子/随机流 | `np.random.default_rng(0)` 固定 | Rust 用 mulberry32（已知与 PCG64 不同，换 seed 自然差异，非 bug） | 同输入不同随机流→纹理微差，属已知 |

> **修复执行状态（2026-09-03）**：
> - ① 已修复：`telea.rs` 整文件重写为 OpenCV `icvTeleaInpaintFMM` 忠实移植；
>   `deglow.rs` 的 TELEA 预检改传膨胀蒙版 `&mf`（对齐 cv2 `_run_fill`）。
>   跨端字节一致性已用 `shared/_verify/` 证明（py==node==brw，四通道 ALL IDENTICAL）。
> - ②~⑥ **尚未逐项逐行审计**，但座驾（默认参数 `edge=1, direction=None,
>   edge_aware=False, soft_expand=0`）的端到端 wasm-vs-cv2 已近像素一致
>   （whole mean\|diff\| 0.41，填充区 max 29，背景 0.000），说明默认路径的
>   PatchMatch(②)/ROI(③)/edge 椭圆(④)/mask 手术(⑥) 在常用参数下功能一致；
>   ⑤ 的 `edge_aware`/`soft_expand` 在座驾为关，未触发。
> - ⑦ 已知非 bug：PatchMatch 随机流 mulberry32 vs PCG64，换 seed 自然纹理微差。
> - 仍建议对 ②~⑥ 在更宽图像集（含 `direction` 模式、`edge_aware=True`、`soft_expand>0`、
>   大图 ROI 触发）上补一轮 verify_align 回归，才能完全闭合清单。
> - 临时调试导出（`dbg_*` 系列，含 `shared/src/deglow.rs` 尾部、`shared/bindings/textcore.py`
>   的 `dbg_dist_l2/gauss/resize` 与 TEMP 方法块、`shared/tests/deglow_verify.rs`）已全部移除。
