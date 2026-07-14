---
title: "Scalable Optimization in the Modular Norm"
description: "把层角色、operator norm 与可迁移 update scale 放进统一的架构语言。"
topic: "muon"
section: "papers"
slug: "scalable-optimization-modular-norm"
legacyPaths: ["/notes/scalable-optimization-modular-norm/"]
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 63
source:
  repository: "J-shang/Muon"
  path: "论文精读/02-Scalable-Optimization-in-the-Modular-Norm.md"
  url: "https://github.com/J-shang/Muon/blob/ae2b5f9e6ee06b411aef2220e361c75988a7d753/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/02-Scalable-Optimization-in-the-Modular-Norm.md"
  revision: "ae2b5f9e6ee06b411aef2220e361c75988a7d753"
  syncedAt: "2026-07-14"
  contentHash: "sha256:5b8fe7943751cc2125cbd07f0cda3440d44abe5e0f458ef7cbf22d6d6c2c0cb9"
  manifest: "muon"
  managed: true
---
> source: [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8629b0fff229b8a27efb1422e990605f-Abstract-Conference.html)
> source class: 同行评审会议论文
> confidence: 形式体系与论文内定理 `verified in-paper`；跨架构 recipe 收益 `supported in-scope`

## 它解决什么问题

网络变宽或变深时，同一个 raw parameter update 对函数的扰动并不保持同尺度。论文不为每种 optimizer 手工补一组 scale factor，而是让网络结构递归地定义完整参数空间的 **modular norm**，再把 base optimizer 的 update 归一到这个范数中。

核心观念是：一个参数矩阵不是孤立数组。它作为 linear、embedding、normalization、residual branch 等不同模块时，对输入—输出 operator 的影响不同；自然 update 尺度应由模块角色和组合规则共同决定。

## 最小形式化对象

设模块 $f_W:X\to Y$ 的输入输出空间各有范数。参数扰动 $\Delta W$ 的自然大小应控制函数扰动，例如

$$
\|f_{W+\Delta W}(x)-f_W(x)\|_Y
\lesssim \|\Delta W\|_{W}\,\|x\|_X.
$$

对 linear layer，$\|\Delta W\|_W$ 自然连接到 induced/operator norm；对串联、并联和 residual composition，模块范数按架构递归组合。论文进一步给出“well-behaved”原子模块下 gradient Lipschitz 常数的递归控制。

## 可核查锚点：宽度变化为何会破坏 raw RMS 直觉

取 $x\in\mathbb{R}^n$，每个分量方差为 1，$\Delta W\in\mathbb{R}^{m\times n}$ 的元素独立且 RMS 为 $r$。则单个输出分量的扰动方差约为

$$
\operatorname{Var}[(\Delta Wx)_i]\approx nr^2.
$$

若希望宽度 $n$ 变化时函数扰动保持 $O(1)$，需要 $r=O(n^{-1/2})$。这只是随机各向同性例子，不等于 modular norm 全理论，但它揭示了为什么“所有矩阵同一元素 RMS”不保证可迁移训练。

Muon 的满秩 polar update 自带

$$
\operatorname{RMS}(UV^\top)=\frac{1}{\sqrt{\max(m,n)}},
$$

因此它与 operator-norm/shape-aware scaling 有天然联系；具体 recipe 是否等于 modular normalization 仍需逐层核对。

## 与 Muon 的关系

- `prerequisite-for` → 理解 shape-aware update scale 不是“调参补丁”。
- `generalizes` → 从单个 linear layer 的自然 norm 扩展到完整架构组合。
- `analogy-to` → muP 的可迁移超参数目标；二者形式和适用假设不同。
- `not-equivalent-to` → Muon 是产生方向的 optimizer，modular normalization 可包裹不同 base optimizer。

## 论文报告与边界

- **论文贡献**：定义递归 modular norm；证明相关 Lipschitz 性质；报告宽度和深度上的 LR transfer；提供 Modula 包。
- **不能推出**：任意新模块都自动“well-behaved”，或任何 Muon 实现只要乘一个 shape factor 就获得跨规模迁移。
- **需要检查**：residual multiplier、normalization、attention softmax、embedding/readout 的模块角色。

## 精读后的任务

选一个两层 MLP 和一个 residual block，为每个参数写：输入/输出 shape、module role、候选 operator norm、raw update RMS、期望函数扰动。再把 Muon 的 shape scale 代入，找出仍依赖 depth 或 residual multiplier 的项。

## 自测

1. 为什么参数数组 shape 相同但 module role 不同，natural norm 仍可能不同？
2. modular norm 解决的是 update direction、update scale，还是两者都解决？
3. 若 LR 能跨 width 迁移，能否推出也能跨 depth、batch 和 optimizer 迁移？

**掌握标准**：面对新架构，能从模块组合推导需要检查的尺度，而不是只查 `fan_in`。

## 二次审计：补漏、分歧与原文核查

### A. 还值得学习的点

1. **mass 不只是 norm 的别名**：论文给模块分配 mass，并讨论 feature learning 如何随 mass 在复合模块间分配。它为 residual branches 的相对学习强度提供可操作旋钮。
2. **atomic module 与 bond module 分工**：附录把 Linear/Embedding/Conv2D 等带参数原子，与 Add 等无参数“胶水”分开；递归理论依赖这种 typed composition。
3. **Embedding 的输入 norm 不同**：附录明确因 one-hot 输入而用 $L_1$ 输入 norm、RMS 输出 norm。这解释“数组同为二维”仍不能按 Linear 路由。
4. **实验证据有固定轴**：GPT/TinyStories、ResMLP、ResNet 的 width transfer 在固定 depth 下做，depth transfer 在固定 width 下做；不能把它读成所有维度同时变化的无条件 transfer。

### B. 与其他论文或学者观点的冲突检查

| 对照观点 | 第一处分歧 | 判断 | 判别检查 |
|---|---|---|---|
| $\mu$P 从无限宽 parameterization 讨论 hyperparameter transfer | modular norm 从有限模块的 operator norm/组合规则出发 | **不同理论路径，目标部分重叠**，不是已证明等价 | 同一 width/depth grid 比较各自 scale prescription |
| Muon 原始说明：embedding 用 AdamW，output head 也经验上用 AdamW | modular theory能解释 embedding 角色；原作者称 output-head 差异未由理论直接给出 | **理论覆盖不完整，不是冲突** | 分别消融 embedding/head routing |
| Scion 选择具体 $1\to\infty$、RMS$\to\infty$ norm | 本文提供递归 modular framework而非唯一 norm 选择 | **specialization** | 对齐 module type 和 norm 后比较 duality map |
| “统一 shape scale 即可” | 本文依赖 module role、mass、composition | **与简化 recipe 真有张力** | 相同 shape 不同 role 的 function-space perturbation |

### C. 本笔记知识核查表

| 本笔记学习项 | 原文位置 | 核查结论 |
|---|---|---|
| modular norm 随架构递归定义，并用于 normalize base optimizer updates | Abstract、§3–4 | `论文明确` |
| 对 well-behaved atomic modules，gradient 在 modular norm 下 Lipschitz，常数递归 | Abstract、理论章节 | `论文明确且有条件` |
| width/depth LR transfer | 主文实验及附录 Figures 10–12 | `论文报告`，固定另一轴、任务和训练预算 |
| 本笔记随机 $\operatorname{Var}[(\Delta Wx)_i]\approx nr^2$ 例子 | 无对应原文推导 | `教学性仓库推导`，只说明 fan-in scaling，不等于 modular theorem |
| Muon polar RMS 与 modular scaling 的“天然联系” | 论文不讨论 Muon 名称或该 RMS 式 | `跨论文综合/analogy`，不得称为论文结论 |
| 新模块需检查 residual multiplier、attention、embedding/readout | 来自论文 type/composition 逻辑与本项目实现需求 | `综合后的操作清单`，不是逐字结论 |
