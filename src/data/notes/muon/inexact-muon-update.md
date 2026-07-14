---
title: "非精确 Muon Update"
description: "研究有限步 Newton–Schulz 误差与学习率、动量及训练目标之间的耦合。"
topic: "muon"
section: "papers"
slug: "inexact-muon-update"
legacyPaths: ["/notes/inexact-muon-update/"]
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 73
source:
  repository: "J-shang/Muon"
  path: "论文精读/12-Inexact-Muon-Update.md"
  url: "https://github.com/J-shang/Muon/blob/ae2b5f9e6ee06b411aef2220e361c75988a7d753/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/12-Inexact-Muon-Update.md"
  revision: "ae2b5f9e6ee06b411aef2220e361c75988a7d753"
  syncedAt: "2026-07-14"
  contentHash: "sha256:54bf8e2a32c7d393509af41481445f4fba23c94b9bebd126dd478fd9c00c4fa3"
  manifest: "muon"
  managed: true
---
> source: [arXiv:2510.19933](https://arxiv.org/abs/2510.19933)
> source class: 理论 + nanoGPT/CIFAR 实验预印本
> confidence: error-bound 结论限于模型假设；co-tuning 现象 `supported in-scope`

## 它修复了哪条理论—实践裂缝

早期理论常把 Muon 的核心替换成 exact SVD polar，但实际算法依赖少量 Newton–Schulz steps。本文在 LMO 框架中引入 additive oracle error，直接分析近似方向。

理想 oracle 为

$$
S^*\in\arg\min_{S\in\mathcal K}\langle G,S\rangle.
$$

一种 inexact 模型写成

$$
\langle G,\widetilde S\rangle
\le \langle G,S^*\rangle+\delta,
$$

其中 $\delta$ 衡量近似方向在线性子问题上的损失，而不是只看矩阵元素误差。

## 为什么这个 error 比 $\|\widetilde S-S^*\|_F$ 更贴近优化

优化一步的 first-order decrease 由 $\langle G,S\rangle$ 决定。两个方向可能 Frobenius 距离不小，却对当前 $G$ 有相近 inner product；反之，小的矩阵误差也可能集中在高权重 singular direction 上。论文选择 LMO error，是把数值误差翻译成优化相关量。

## 可核查锚点：同一 NS 精度不应孤立复用 LR

简化 smoothness 上界：

$$
f(W-\eta\widetilde S)
\le f(W)-\eta\langle G,\widetilde S\rangle
+\frac{L\eta^2}{2}\|\widetilde S\|^2.
$$

oracle error 会削弱线性下降项，而过大的 $\eta$ 放大二次余项。因此近似变差时，原先对 exact oracle 最优的 LR 可能不再最优。本文进一步报告：较低 oracle precision 倾向需要更小 step size、不同/更大的 momentum 参数，并在 nanoGPT 实验中观察到最优 LR 随 precision 移动。

这不是说“NS steps 少就把 LR 固定乘某常数”，而是说明 NS precision、LR、momentum 构成联合超参数面。

## 与 polar accuracy 的区别

至少记录三种指标：

1. 数值：$\|\widetilde S-S^*\|_F$、singular values、orthogonality error；
2. 优化：LMO gap/inner product、update cosine、predicted decrease；
3. 训练：loss-vs-token/FLOP/time 和稳定性。

三者排序不保证一致。固定 LR 比不同 NS steps，可能只是比较了不同程度的 hyperparameter mismatch。

## 论文报告与边界

- **理论贡献**：给出 inexact LMO 下确定性/随机收敛界，并显式连接 error、step size、momentum。
- **实验支持**：nanoGPT/FineWeb 和 CNN/CIFAR-10 观察到 precision 与最佳 LR 的耦合。
- **剩余缺口**：error model 对具体 quintic/bfloat16/kernel 的紧致性；大模型长 horizon；shape-specific precision。

## 精读后的任务

对固定 momentum snapshots 扫描 NS steps、dtype 和系数，计算 LMO gap；再对每个 precision 独立扫描 LR/momentum。比较“固定超参排序”和“各自调优后的 Pareto 排序”。

## 自测

1. 为什么 LMO gap 比纯 Frobenius error 更直接对应一阶下降？
2. 若减少 NS steps 却不重调 LR，实验混入了什么变量？
3. additive error 模型在哪些 rank-deficient 或低精度情况下可能过松？

**掌握标准**：不再把 `ns_steps` 当孤立 kernel knob，而是把它放进优化超参数联合面。

## 二次审计：补漏、分歧与原文核查

### A. 还值得学习的点

1. **层间 precision 不应默认相同**：Appendix C.2 的 block-wise analysis 为每层引入不同 norm、smoothness 和 inexactness，指出统一 NS steps 可能不是计算最优。
2. **实验主要用 PolarExpress 而不只 NS**：nanoGPT 主实验用 PolarExpress iterations 作为 $\delta$ proxy，CNN 也涉及不同 approximation；“少一步 NS”不是全部实证对象。
3. **precision 还改变 stable region**：Figure 1 不只看最佳 loss，也显示更高 precision 扩大 LR/momentum 的稳定超参区域。
4. **理论还有 generalized smoothness/time-varying extensions**：附录把主结果扩展到 $(L_0,L_1)$-smoothness、layer-wise 和 time-varying parameters；学习时应区分主定理与扩展。

### B. 与其他论文或学者观点的冲突检查

| 对照观点 | 第一处分歧 | 判断 | 判别检查 |
|---|---|---|---|
| Moonlight：更多 NS steps 更准确但未提升表现 | approximation algorithm、模型/训练规模、LR/momentum 是否为每个 precision重调 | **真实经验张力，尚不能裁决** | 同模型同 kernel，对每个 steps 独立调参并报 loss-vs-time |
| finite-NS convergence论文给具体 polynomial 的快速误差界 | 本文用一般 additive LMO error $\delta$ | **抽象层不同、互补** | 将 concrete NS error 映射为 empirical LMO gap |
| Muon² 通过 second moment 改善 NS 输入 condition | 本文把 approximation quality 当外生/可分层配置 | **Muon² 提供一种降低 $\delta$ 的机制，但也会改 exact direction** | exact SVD on preconditioned/unpreconditioned snapshots |
| 原始博客接受 singular values 在 $[0.7,1.3]$ | 本文固定设置发现更高 precision通常改善 | **训练 regime 与评价不同** | 对齐原 quintic和 PolarExpress、固定/调优超参两套比较 |

### C. 本笔记知识核查表

| 本笔记学习项 | 原文位置 | 核查结论 |
|---|---|---|
| additive inexact LMO 模型 | §2 Assumption 1, Eq. 3 | `论文明确` |
| error 与 step size、momentum coupling | Abstract、Corollaries、§4 experiments | `论文定理 + 作者实验支持` |
| 低精度需要更小 step size、不同/更大 momentum | Abstract/理论最优参数讨论 | `论文主张 under assumptions`，不是通用固定换算 |
| nanoGPT/CIFAR 实验 | §4 | `论文明确`；nanoGPT 主 proxy 是 PolarExpress iterations |
| 本笔记 smoothness descent inequality | 标准 smoothness 教学展开，不是论文原式逐字复现 | `仓库内解释`，与分析方向一致 |
| Frobenius error、LMO gap、training utility 三层指标 | 跨数值/优化/训练综合 | `项目审计框架`，不是作者三分法 |
