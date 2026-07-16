---
title: "Muon 与 Newton–Schulz 的收敛"
description: "直接分析实际有限步 Newton–Schulz Muon，而不是用精确 SVD 替代实现。"
topic: "muon"
section: "papers"
slug: "muon-convergence-newton-schulz"
legacyPaths: ["/notes/muon-convergence-newton-schulz/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 74
source:
  repository: "J-shang/Muon"
  path: "论文精读/13-Convergence-with-Newton-Schulz.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/13-Convergence-with-Newton-Schulz.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:916d7ca12d706a69c790b8fe4b22e88e7c616a913dfd3013c0a9042e74343fe4"
  manifest: "muon"
  managed: true
---
> 原文：[arXiv:2601.19156](https://arxiv.org/abs/2601.19156)，核验版本 v1（2026-01-27）；arXiv 标注已被 ICLR 2026 接收
> 来源类型：同行评审会议论文，含理论与多工作负载实验
> 阅读提醒：收敛定理只在论文的多项式、谱和光滑性假设下成立；现实通用性仍需逐实现检查。

## 它回答什么问题

本文直接分析实际 Muon 的有限步 Newton–Schulz，而不是用 exact SVD polar 代替。作者证明：给定 NS steps 时，非凸 stationarity rate 与 SVD-polar idealization 相同到一个常数因子；该因子随 NS steps 双指数趋近 1，并证明在论文采用的范数与光滑性度量下，相对 vector SGD-momentum 消除典型的 square-root-of-rank loss。

## NS 的谱标量视角

若迭代形式为

$$
X_{k+1}=aX_k+bX_kX_k^\top X_k+c(X_kX_k^\top)^2X_k,
$$

且 $X_k=U\operatorname{diag}(\sigma_i^{(k)})V^\top$，则奇异向量保持不变，每个奇异值独立经过

$$
\sigma^{(k+1)}
=a\sigma^{(k)}+b(\sigma^{(k)})^3+c(\sigma^{(k)})^5.
$$

所以收敛问题可转成：初始缩放后各 $\sigma_i$ 是否落在 polynomial 的吸引域，以及有限 $q$ 步后离 1 多远。

## 可核查锚点：标准 cubic 的局部平方收敛

对经典 polar cubic

$$
x_{k+1}=\frac32x_k-\frac12x_k^3,
$$

令 $x_k=1-e_k$，展开得

$$
e_{k+1}=\frac32e_k^2-\frac12e_k^3.
$$

当 $e_k$ 小时，$e_{k+1}\approx\tfrac32e_k^2$：误差每步近似平方，因此随迭代次数呈双指数下降。这解释“少数步骤即可很准”的数学来源，也提醒：生产 quintic 系数、BF16 rounding 和极小奇异值可能不满足同一局部常数。

## rank 改善怎样理解

矩阵 norm 之间有

$$
\|M\|_*\le\sqrt{r}\|M\|_F,
\qquad r=\operatorname{rank}(M).
$$

不同 geometry 的 convergence bound 在转换 norm 时会出现 $\sqrt r$。论文的比较说明 matrix-aware polar direction 可避免某个 vectorized 分析中的 rank 损失。它是特定 smoothness/noise/stationarity 定义下的 bound 改善，不等于任何秩越高训练就越快 $\sqrt r$。

## 与 Beyond the Ideal 的互补

- 本文对具体 NS polynomial 建模，强调 approximation error 随 steps 的快速衰减和 convergence constant。
- *Beyond the Ideal* 使用一般 inexact LMO，强调 precision 与 LR/momentum co-tuning。
- 第一处分歧是 error parameterization，不必二选一。区分实验应同时计算 polynomial error、LMO gap 和训练最佳点。

## 论文报告与边界

作者在 MLP、CIFAR 模型、nanoGPT 和最高约 1.3B GPT-2 风格模型上做消融，报告少量 NS steps 接近 SVD-polar。正式采用结论前仍需检查：初始化缩放、polynomial degree、rank/smoothness 假设、有限精度与通信成本。

## 精读后的任务

画 production quintic 的 $x\mapsto p(x)$、误差 $|p^{\circ q}(x)-1|$ 和 BF16/FP32 差异，覆盖 $x\in[10^{-6},1]$。把理论吸引域与真实 momentum singular-value histogram 叠在同一图上。

## 自测

1. 为什么矩阵 polynomial 不改变奇异向量，只变换奇异值？
2. “双指数随 NS steps 收敛”需要哪些 basin/precision 条件？
3. bound 中改善 $\sqrt r$ 与 wall-clock speedup 有何逻辑距离？

**掌握标准**：能从标量 polynomial 推回矩阵行为，并主动检查 production 系数与定理假设是否一致。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **论文公开反对“Muon 是二阶法”的分类**：Related Work 明确说 Shampoo/SOAP 维护曲率统计，而 Muon 不估计/反演 curvature；这是与 *Practical Efficiency* 的术语冲突。
2. **分析同时包含 iteration 与 wall-clock complexity**：附录计算每步 orthogonalization FLOPs，理论主张不是“同 rate”就自动更快，而是把少步矩阵乘成本一起比较。
3. **polynomial degree 与 step 数是两个变量**：作者不只分析 $q$ steps，还分析近似方向 polynomial 的 degree $\kappa$，并做 degree ablation。
4. **rank 优势是 norm/smoothness measure 下的 bound**：论文自己把它作为相对 vectorized SGD-momentum 的理论比较；不能外推为所有数据的实际 $\sqrt r$ speedup。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| *Practical Efficiency* 称 Muon second-order | curvature-state definition | **分类口径直接冲突**；本笔记采用“是否估计曲率”时归为非二阶 | state/公式检查 |
| Moonlight：更精确 NS 未必训练更好 | 理论是 stationarity complexity constant；实证是固定 recipe 的 loss | **outcome/超参不同** | 同步报告 theorem proxy、polar error、loss/time |
| *Beyond the Ideal* 用 additive LMO error | 本文把 concrete polynomial 误差显式带入 rate | **互补 parameterization** | 计算同 snapshot 的两种 error bound tightness |
| 原始 tuned quintic 不是经典 polar polynomial | 定理对 polynomial degree、系数和 basin 有假设 | **实现适配问题** | 逐条检查 production $(a,b,c)$ 是否满足 theorem assumptions |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| finite NS 与 SVD-polar 有相同 stationarity rate，到常数因子 | Abstract、Theorem 1 | 论文定理，只在其假设下成立 |
| 常数随 steps 双指数趋近 1，并受 degree 改善 | Theorem 2、§4.2 | 论文定理，只在给定 polynomial 与 basin 假设下成立 |
| 相对 SGD-momentum 改善 $\sqrt r$ rank factor | Theorem 3、Table 1 | 论文理论比较，不是实测 wall-clock 倍数 |
| 多工作负载实验到约 1.3B | experimental appendices | 作者实验报告 |
| 本笔记经典 cubic 的 $e_{k+1}=\frac32e_k^2-\frac12e_k^3$ | 教学 special case | 本文精确展开，不等于 production quintic |
| “少数 steps 在现实 BF16 一定足够” | 理论/实验均有条件 | 原文不支持这个普适结论；仍需检查真实 spectrum 和 dtype |
