---
title: "Old Optimizer, New Norm"
description: "从范数约束重建 Adam、Shampoo 与 Muon 的最陡下降解释。"
topic: "muon"
section: "papers"
slug: "old-optimizer-new-norm"
legacyPaths: ["/notes/old-optimizer-new-norm/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 65
source:
  repository: "J-shang/Muon"
  path: "论文精读/04-Old-Optimizer-New-Norm.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/04-Old-Optimizer-New-Norm.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:5337cdbb428d6b0244bbeb4504c83cf4335e259abf42b8a6a684ce3597fdf5e7"
  manifest: "muon"
  managed: true
---
> 原文：[arXiv:2409.20325](https://arxiv.org/abs/2409.20325)，核验版本 v2（2024-12-06）
> 来源类型：理论预印本
> 阅读提醒：范数最陡下降推导可复核；它只解释部分更新方向，不等于覆盖完整训练配方。

## 它解决什么问题

很多 optimizer 看起来像互不相关的坐标公式。论文换一个问题：若把“允许走多大一步”所使用的范数改变，熟悉的 update 是否可统一理解为 steepest descent？

给定局部线性化 $\langle G,\Delta\rangle$，范数约束形式是

$$
\Delta^*\in\arg\min_{\|\Delta\|\le \eta}\langle G,\Delta\rangle.
$$

由对偶范数定义，最优值为

$$
\min_{\|\Delta\|\le \eta}\langle G,\Delta\rangle
=-\eta\|G\|_*.
$$

optimizer 的“归一化方向”可看成把 dual gradient 映回 primal update 的 dualization map。

## 可核查锚点：谱范数球给出 polar

令 $G=U\Sigma V^\top$。考虑

$$
\min_{\|\Delta\|_2\le1}\langle G,\Delta\rangle_F.
$$

谱范数的对偶是核范数，因此最优值为 $-\|G\|_*$。取

$$
\Delta=-UV^\top
$$

时，$\|\Delta\|_2=1$，且

$$
\langle G,-UV^\top\rangle_F
=-\operatorname{tr}(\Sigma)
=-\|G\|_*.
$$

所以 polar direction 是这个 matrix trust region 的最陡方向。秩亏时 null space 上的补全可能不唯一，但 $-UV^\top$ 是自然的薄 polar 解。

## 这对 Muon 解释了什么

- 解释更新为何压平非零奇异值：不是在估计 Hessian，而是在换 trust-region geometry。
- 解释 shape scale 为什么不能遗漏：unit spectral norm 不等于 unit element RMS。
- 给出 Adam/SignSGD 与 $\ell_\infty$-type geometry 的类比入口。

但这套静态推导本身不自动覆盖 momentum、有限步 NS、decoupled weight decay、参数路由和 stochastic dynamics。

## 容易误读的地方

1. **“一阶”不等于逐元素**：只使用 gradient 也可以做全矩阵 dualization。
2. **“像二阶”不等于 Hessian method**：非对角矩阵变换和二阶曲率估计是不同定义轴。
3. **方向正确不等于配方完整**：同一方向乘不同 shape scale 会产生不同函数扰动。
4. **单步最陡不保证全程最快**：训练是带噪、带 momentum、非凸的动态过程。

## 知识关系

- **直接解释**：exact-polar Muon 的方向。
- **后续延伸**：Modular Duality 与 spectral-constraint 分析建立在类似几何语言上。
- **不能推出**：实际有限步 NS 等于精确 trust-region oracle。

## 精读后的任务

分别求 $\ell_2$、$\ell_\infty$、matrix spectral norm 单位球上线性目标的最优方向，列出其 dual norm。再对同一 $2\times2$ 梯度画出三个 trust region 和接触点。

## 自测

1. 梯度为什么自然属于 dual space，而 update 属于 primal space？
2. 谱范数的对偶为何是核范数？上面的等号由哪个可行解达到？
3. 若使用 5 步 NS，哪一条等式从 exact 变成 approximation？

**掌握标准**：能独立重建约束问题、对偶范数和 polar 最优解，不依赖 optimizer 名称记忆。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **论文的单位不是单个矩阵，而是 layer list**：Story I 用 max-of-max norm 重解释 sign descent；Story II 的 Shampoo 结论是“跨层 max spectral norm”下的 steepest descent。逐层 polar 只是解这个组合问题的一部分。
2. **Adam 故事明确关掉 EMA**：论文将无 EMA Adam 化为 sign descent；完整 Adam 的一、二阶历史没有被该静态 norm 问题覆盖。
3. **sharp operator 与固定-radius LMO 要区分**：steepest step 的尺度通常含 dual norm；实践 Muon 常只取 scale-invariant polar/LMO，再用外部 LR/shape factor 定尺度。
4. **Appendix A 是数值方法入口**：论文列出 SVD、inverse roots 与 Newton–Schulz 等获得 semi-orthogonal direction 的策略，影响后来 Muon 的实现路线。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| Scion：实践 Muon 省略 sharp operator 中的 $\|g\|_*$，更像 spectral-ball LMO | update 是否乘 dual-norm magnitude | **重要公式差异**；方向相同，尺度不同 | 为 polar direction 加/不加 $\|g\|_*$，比较 scale invariance |
| 谱约束论文：momentum 与 decoupled decay 不受单步 steepest view 自然覆盖 | 静态一步问题 vs 完整动态递推 | **后续扩展，不是直接反驳** | 固定 pointwise direction，加入 momentum/decay 后比较 Lyapunov/constraint |
| Practical Efficiency 称 Muon second-order | 本文把无累积 Shampoo/Muon 方向写成 first-order norm geometry | **分类口径冲突** | 先规定“二阶”是否要求 curvature state |
| SOAP 保留 Shampoo history | 本文为得到 polar 明确关闭 accumulation | **special-case 边界** | accumulator on/off |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| norm-constrained steepest descent 与 dual norm 的统一语言 | Story I–III | 论文明确 |
| accumulation-free Shampoo 产生 semi-orthogonal update | Story II, Eqs. 15–16 | 论文明确 |
| spectral norm ball 的单矩阵最优方向为 $-UV^\top$ | Story II 的 SVD 推导 | 论文明确/可复核 |
| 本笔记写的单矩阵 trust region | 论文更强地使用跨层 max norm | 正确的特殊情形；已补充层级边界 |
| “解释 exact-polar Muon” | 论文版本围绕 Shampoo/duality，不负责完整 Muon recipe | 跨论文比较，只覆盖方向，不覆盖 momentum/NS/scale/decay |
| $\ell_2$、$\ell_\infty$、spectral 三个球的练习 | 教学扩展 | 本文练习，不是论文实验 |
