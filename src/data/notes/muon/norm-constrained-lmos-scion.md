---
title: "Norm-Constrained LMOs 与 Scion"
description: "把 polar update 放进线性最小化 oracle，并区分优化器更新与参数约束。"
topic: "muon"
section: "papers"
slug: "norm-constrained-lmos-scion"
legacyPaths: ["/notes/norm-constrained-lmos-scion/"]
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 67
source:
  repository: "J-shang/Muon"
  path: "论文精读/06-Norm-Constrained-LMOs-Scion.md"
  url: "https://github.com/J-shang/Muon/blob/ae2b5f9e6ee06b411aef2220e361c75988a7d753/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/06-Norm-Constrained-LMOs-Scion.md"
  revision: "ae2b5f9e6ee06b411aef2220e361c75988a7d753"
  syncedAt: "2026-07-14"
  contentHash: "sha256:10118f394a5c13908363be4c77c6cb50df8f5be130fb0b6e2bc2d35436b0bf70"
  manifest: "muon"
  managed: true
---
> source: [arXiv:2502.07529](https://arxiv.org/abs/2502.07529)
> source class: 理论与实验预印本
> confidence: LMO 关系 `verified`；Scion 的训练优势 `supported in-scope`

## 它解决什么问题

如果 optimizer 的核心是“在某个 norm ball 上找最能降低线性化目标的方向”，那么这个子问题就是 Linear Minimization Oracle（LMO）：

$$
S_t\in\arg\min_{\|S\|\le1}\langle G_t,S\rangle.
$$

论文从 norm-constrained LMO 出发构造训练算法 Scion，并研究如何在不维护 Adam 式 weights+两份 moments 的情况下训练模型。

## 可核查锚点：matrix LMO 就是 polar direction

对 matrix spectral norm ball，

$$
S_t=-U_tV_t^\top,
\qquad G_t=U_t\Sigma_tV_t^\top.
$$

验证：$\|S_t\|_2=1$ 且 $\langle G_t,S_t\rangle=-\|G_t\|_*$，达到由 dual norm 给出的下界。这与 exact-polar Muon 共用一个 oracle。

但“共用 LMO”不代表完整算法相同。Scion 的参数递推、约束处理、momentum/state 与 Muon recipe 必须逐式比较。

## 重要区分：update constraint 与 parameter constraint

- LMO 子问题约束的是候选方向 $S_t$。
- 通过 convex combination、weight decay 或特定递推，参数 $W_t$ 可能被保持/吸引在某个 norm ball。
- 标准 Muon 的一步 polar direction 只直接给出 update spectral norm；要推出 parameter spectral constraint，还需分析完整 recurrence。

把这三层混在一起，会错误地从“update 半正交”推出“weight 正交”。

## 论文贡献和边界

- **论文报告**：提出基于 norm-constrained LMO 的 unconstrained stochastic algorithms，并以 Scion 在 nanoGPT 等设置验证；强调较低 optimizer-state memory。
- **稳固关系**：spectral LMO 与 polar 的代数等价。
- **未自动解决**：finite-step NS 的 oracle error、生产级 distributed matrix semantics、与 tuned AdamW 的全尺度 Pareto。

## 与其他来源的关系

- `formalizes` → *Old Optimizer, New Norm* 的 trust-region 子问题。
- `shares-core-oracle-with` → exact-polar Muon。
- `prerequisite-for` → *Beyond the Ideal* 的 inexact LMO error model。
- `not-equivalent-to` → 含 Nesterov、decay、shape scaling 的实际 Muon。

## 精读后的任务

写出三列递推：纯 LMO direction、Frank–Wolfe convex combination、Muon-style decoupled decay + polar update。用相同初值跑一个凸二次函数，画参数 norm、update norm 和 objective；观察“方向受约束”何时转化为“参数受约束”。

## 自测

1. LMO 的输入、可行集和输出分别是什么？
2. 为什么 exact polar 是 oracle 解，但有限步 NS 只能是 inexact oracle？
3. 一份 optimizer state 的内存结论还遗漏哪些临时 buffer 和 scalar 参数组？

**掌握标准**：能严格区分 oracle、optimizer recurrence、parameter feasible set 三层对象。

## 二次审计：补漏、分歧与原文核查

### A. 还值得学习的点

1. **sharp operator 与 LMO 的尺度差异是论文关键点**：Related Work 的 Eq. 7 显示 sharp step 含 $\|g\|_*$；作者指出实践 Muon 忽略该 scale，因此更像 spectral-norm ball 上的 LMO。
2. **Scion 不只是一个算法名**：论文区分 constrained/unconstrained variants，并用 norm 选择导出 sign、row-normalized、spectral 等不同 update；方法族的公共对象是 LMO。
3. **宽度不变分析有简化假设**：理论为得到 clean derivation 取特定 momentum 设置；不能把 width transfer 无条件外推到实用 momentum recipe。
4. **内存技巧有实现前提**：ScionLight 把 averaged gradient 存在 backprop gradient buffer 中，所以训练过程中不能按常规随意 zero gradient。

### B. 与其他论文或学者观点的冲突检查

| 对照观点 | 第一处分歧 | 判断 | 判别检查 |
|---|---|---|---|
| *Old Optimizer, New Norm* / Modular Duality 的 sharp map | sharp update 含 dual-norm magnitude；Scion LMO scale-invariant | **真实公式差异，方向可相同** | 增加 gradient global scale，检查 update 是否变化 |
| Moonlight 的 consistent/match RMS | Scion 强调 LMO 的 magnitude invariance；Moonlight再加 shape/RMS scale | **外部尺度约定不同** | 固定 direction，对齐 per-layer RMS 后比较 |
| “Muon 普遍优于其他方法” | Scion 在自己的 nanoGPT/large-batch 设置报告优于 tuned Muon | **经验结果有张力，非普适定律** | 同代码、调参预算、batch 和 stopping target 复现 |
| modular norm/$\mu$P 的宽度迁移 | Scion 选具体 operator norms并在简化 momentum 下分析 | **specialization，不是完整等价** | 同架构逐层对齐 norm 和 LR scaling |

### C. 本笔记知识核查表

| 本笔记学习项 | 原文位置 | 核查结论 |
|---|---|---|
| norm-constrained LMO 定义及 spectral-ball 解为 polar | §2–4、Muon related-work subsection | `论文明确` |
| Muon 可写成 momentum + spectral LMO recurrence | Related Work “Muon” | `论文明确` |
| update constraint、parameter constraint、optimizer recurrence 需区分 | 由论文 SCG/uSCG/Scion 算法对照得出 | `跨算法综合`，概念区分正确但非单句原文 |
| 只存一份 weights 和一份 half-precision gradients | Abstract、实验结论、Appendix E.2 | `作者实现报告`；依赖 gradient-buffer reuse |
| 本笔记三条凸二次轨迹任务 | 无原文实验 | `仓库练习` |
| “Scion 是 Muon” | 原文没有此等价；只共享 LMO geometry | `明确否定`，完整 recurrence/state 不同 |
