---
title: "Practical Efficiency of Muon"
description: "从多尺度、large batch、μP 与 compute-time Pareto 评估 Muon 的实践效率。"
topic: "muon"
section: "papers"
slug: "practical-efficiency-of-muon"
legacyPaths: ["/notes/practical-efficiency-of-muon/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 69
source:
  repository: "J-shang/Muon"
  path: "论文精读/08-Practical-Efficiency-of-Muon.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/08-Practical-Efficiency-of-Muon.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:cb9e9fe79a52931455682dc23ff25ebc548f36a75f864b73e6e4032045473cf1"
  manifest: "muon"
  managed: true
---
> 原文：[arXiv:2505.02222](https://arxiv.org/abs/2505.02222)，核验版本 v4（2025-05-20）
> 来源类型：多尺度实验预印本
> 阅读提醒：实验趋势只覆盖作者训练栈；跨数据、架构和系统的外推仍待验证。

## 它补了什么证据

Moonlight 给出一个大规模 scaling-law 故事；这篇工作更关注实践效率的多个横轴：数据/iteration efficiency、per-step overhead、wall-clock、batch size 和超参数跨规模迁移。实验覆盖到约 4B 参数，并用 muP/telescoping 思路调优。

## 核心阅读框架：不要只看 loss-vs-step

同一实验至少要同时记录

$$
\text{tokens}=B_{\text{tokens}}\times T,
$$

$$
\text{training FLOPs}\approx T\times \text{FLOPs/step},
$$

以及实测

$$
\text{wall-clock}=\sum_{t=1}^{T}
(t_{\mathrm{fwd/bwd}}+t_{\mathrm{comm}}+t_{\mathrm{optimizer}}).
$$

Muon 可能减少达到目标 loss 的 steps，却增加 optimizer-step cost；large batch 可能提升硬件利用率，却改变 noise scale。只有 Pareto frontier 能把这些效应合并。

## 可核查锚点：两个“更快”可以排序相反

假设 AdamW 需要 1000 steps、每步 1.00 s；Muon 需要 700 steps、每步 1.30 s：

$$
t_{\text{AdamW}}=1000\text{s},
\qquad
t_{\text{Muon}}=910\text{s}.
$$

Muon 的 iteration speedup 为 $1000/700\approx1.43\times$，wall-clock speedup 只有 $1000/910\approx1.10\times$。若 batch 不同，还必须再比较 tokens 和 FLOPs。这个手算例是阅读所有效率图的最低检查。

## muP 与 telescoping 怎样读

muP 旨在让部分超参数沿宽度迁移，telescoping 用较小模型筛选候选再逐级放大，降低调参成本。它们改善“如何公平调优”的方法学，但不保证：

- optimizer 改变后所有超参数仍按同一规则迁移；
- width 迁移等于 depth、batch、token horizon 迁移；
- 小模型排序在更大规模必然保持。

因此每个跨尺度结论都应记录实际重调了什么。

## 论文报告与边界

- **论文报告**：Muon 在受测多尺度预训练中形成有竞争力的 compute/time frontier，并在较大 batch 区域保持较好效率。
- **证据价值**：比单模型单预算更能支持趋势，但仍来自同一研究管线。
- **剩余缺口**：更长 horizon、数据重复、不同架构/并行栈、相同调参预算的独立复现。

## 与 critical batch 的关系

本文给出实证 operating point；critical-batch 理论试图解释 batch 增大到何处不再线性减少 steps。前者是实验测量，后者是理论模型，不能把理论下界直接当作本文最佳 batch 的预测值。

## 精读后的任务

从论文重画一张证据表，每个点记录 model size、batch、tokens、optimizer recipe、是否重调 LR/WD/momentum、per-step time、目标 loss。禁止只抄相对百分比。

## 自测

1. iteration efficiency、data efficiency、compute efficiency、wall-clock efficiency 各用什么横轴？
2. muP 支持了哪类迁移，又没有覆盖哪些变化？
3. 若 Muon 支持更大 batch，它可能同时改变哪三种效率指标？

**掌握标准**：看到“更快”会先追问目标值、横轴、batch、硬件和调参预算。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **论文真正的新评价对象是 iso-loss compute–time frontier**：§2.3 同时改变 device count 与 batch，把到达同一 loss 的资源—时间可行集画出来；这比单独 loss-vs-step 或 wall-clock 曲线信息更全。
2. **“beyond critical batch”有专门度量**：论文定义 AdamW/Muon 到同一 loss 的 token-consumption ratio $R_L(B)$，研究它随 batch 增长是否保持/上升，而不是只凭曲线 kink。
3. **telescoping 不是免费 transfer**：§3 承认有限宽和搜索网格误差，通过逐宽度收缩网格把 calibration 推向大模型，估计额外成本 $O(C\log N)$。
4. **实验边界很具体**：最大 4B、batch 至 16M tokens；$\mu$P transfer 实验只变 width，固定 depth、sequence length、batch 和 steps。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| 本文称 Muon 为 simplest second-order optimizer；NS 收敛论文明确称其非二阶 | 是否把 matrix-structured transform 算作二阶，还是要求 curvature estimate | **分类口径 真冲突**，学习时保留两种定义 | 检查 persistent curvature state，而不是按名称投票 |
| Moonlight 使用 AdamW-style decay；本文明确写 coupled WD | decay parameterization 与 transfer rule | **recipe 差异** | 用同一公式和 $\mu$P scale 复现，不能混用超参 |
| SOAP 的 critical batch 是线性 scaling break point；Sato 是 SFO minimizer | critical batch 的数学定义 | **定义冲突** | 同一 $T(b)$ 曲线同时标两种 $b^*$ |
| Moonlight 的 compute-optimal FLOPs 比较 | 本文比较 variable-resource iso-loss frontier | **互补指标**，不是重复验证 | 对同一 run 同时算 FLOPs 和 compute-time frontier |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| Muon 扩展 compute–time Pareto frontier | Abstract、Figures 1–2、§2.3 | 作者在该实验范围内报告 |
| large batch 下相对 AdamW 保持更好 data efficiency | Abstract、§2.4、Figure 3 | 作者报告，依赖数据/模型/目标 loss |
| 最大约 4B、多数据/架构 ablation | Abstract、§2.2、appendices | 论文明确 |
| $\mu$P + telescoping，成本 $O(C\log N)$ | §3.3–3.4 | 论文主张；依赖无新 peak 等假设/网格模型 |
| 本笔记 1000s vs 910s 算例 | 无原文例子 | 本文教学算例，算术可核查 |
| “至少同时报告 token/FLOP/time” | 由论文的 compute-time 动机与本项目公平实验规范综合 | 跨论文的方法建议，不是作者唯一规定 |
