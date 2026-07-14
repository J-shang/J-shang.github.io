---
title: "Muon 实验路线与记录规范"
description: "把数学、数值、训练与分布式理解转化为可复现、可证伪的实验产物。"
topic: "muon"
section: "research-practice"
slug: "muon-experiment-roadmap"
legacyPaths: ["/notes/muon-experiment-roadmap/"]
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 83
source:
  repository: "J-shang/Muon"
  path: "experiments/README.md"
  url: "https://github.com/J-shang/Muon/blob/f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13/experiments/README.md"
  revision: "f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13"
  syncedAt: "2026-07-14"
  contentHash: "sha256:5059e121ae350733e9e2ce30163caa424d1a29d5ee2919dbeb28642b794b69e7"
  manifest: "muon"
  managed: true
---
> 信息截点：2026-07-14。
> 目标：把“我理解了 Muon”变成可复现、可证伪的数学、实现和训练产物。
> artifact 边界：默认只提交代码、配置、短日志摘要和小图；不提交数据、checkpoint、完整训练输出或缓存。

## 总原则

每个实验先写：问题、竞争解释、共同 operating point、变量、失败判据和能区分解释的观测。没有区分性预测的 A/B，只是在收集曲线。

训练效率至少拆成四个对象：

$$
\begin{aligned}
E_{\text{token}}(L^*)&=\text{达到目标 loss }L^*\text{ 的 token 数},\\
E_{\text{FLOP}}(L^*)&=\text{达到 }L^*\text{ 的训练 FLOPs},\\
E_{\text{time}}(L^*)&=\text{达到 }L^*\text{ 的墙钟时间},\\
M_{\text{peak}}&=\text{端到端峰值设备内存}.
\end{aligned}
$$

它们不是等价指标。所有“加速”必须说清分母。

## 实验 0：数学与数值合同

### 问题

有限步、不同 dtype 的 Newton–Schulz 在哪些矩阵谱上仍给出可用的 polar 近似？

### 设置

- shape：方阵、$4d\times d$、$d\times4d$、fused QKV-like shape。
- spectrum：良态、对数均匀、近秩亏、精确秩亏。
- reference：fp64 `torch.linalg.svd` 的薄 SVD polar。
- variants：fp32/bf16；1/3/5/7 steps；不同 coefficient type；标准与 Gram 路径（若可用）。

### 指标

对 $X\in\mathbb{R}^{m\times n}$，记录：

$$
e_{\mathrm{orth}}=
\begin{cases}
\|X^\top X-I_n\|_F/\sqrt n,&m\ge n,\\
\|XX^\top-I_m\|_F/\sqrt m,&m<n,
\end{cases}
$$

并记录与 reference 的 cosine、奇异值 min/max、update RMS、`isfinite`、kernel time 和峰值临时内存。

### 通过条件

- 测试能主动暴露错误的 transpose/shape 检查。
- 报告数值精度与训练可能有用性是不同指标。
- 每个结论带矩阵谱、dtype、系数和 steps 的适用范围。

## 实验 1：小模型公平 A/B

### 竞争解释

- H1：Muon 的几何在对齐 update scale 后仍提高 token efficiency。
- H2：观察到的收益主要来自不公平 LR/WD/batch/参数路由或 speedrun 的其他系统改动。

### 最小矩阵

| 组 | optimizer | routing | scale | WD | NS |
|---|---|---|---|---|---|
| A | tuned AdamW | 标准 decayed/non-decayed 分组 | n/a | tuned | n/a |
| B | Muon + AdamW | hidden matrix / scalar 分开 | original/spectral | tuned | 5 |
| C | Muon + AdamW | 同 B | unit/match RMS | tuned | 5 |
| D | Muon ablation | 同 B | 与胜出组相同 | 0 | 5 |
| E | Muon ablation | 同 B | 与胜出组相同 | tuned | 3 |
| F | Muon semantic ablation | Q/K/V split vs fused | 相同 | 相同 | 相同 |

两类公平性问题要分别回答：

1. **迁移公平**：固定 AdamW recipe，只换 optimizer，衡量 drop-in 能力。
2. **上限公平**：给每个方法相近调参预算，衡量各自 Pareto frontier。

不要把两种结果混成一个排名。

### 必报结果

- train/validation loss vs token、FLOP、wall-clock。
- tokens/s、整个 step 与 optimizer substep 时间。
- peak memory、persistent optimizer-state bytes、temporary bytes。
- grad norm、update RMS、weight RMS、update/weight ratio、attention max logit。
- 非有限值、loss spike、跳过 step、clip/guard 触发次数。
- seed、置信区间和失败 run；不能只保留最好的一次。

## 实验 2：scale 与 batch sweep

### 问题

Muon 的大 batch 优势来自更低梯度噪声、更合适的 update scale，还是硬件吞吐？

### 设计

- 至少 2–3 个模型宽度和 3 个 global batch。
- 每个 optimizer/batch 重新搜索 LR 与 warmup；记录共同调参预算。
- 固定 exact token budget 和数据顺序策略。
- 分别画 loss-vs-token、loss-vs-FLOP、loss-vs-time。
- 若使用 muP，记录 base/delta shapes、参数角色和 Muon scale mode；仍保留最大模型确认点。

### 判别

- token 曲线赢、time 曲线输：算法方向可能更省数据，但实现成本未摊平。
- step 曲线赢、token 曲线输：主要是 batch/横轴错觉。
- 只在某个 scale mode 赢：收益与参数化/缩放耦合，不能归因于 polar 本身。

## 实验 3：分布式语义与成本

### 问题

`duplicated`、`distributed`、`blockwise` 在当前 TP/DP/EP 配置下分别保持了什么数学合同，付出多少通信？

### 设置与检查

1. 构造同一 global matrix/gradient，按真实 layout 分片。
2. 每种模式执行一个 step，收集 global update。
3. 与单 rank reference 比较相对误差和 cosine。
4. 改变 TP size；若声明 exact implementation，global update 应在容差内保持。
5. profile NS GEMM、collective bytes、等待时间、parameter sync 和整个 step。
6. 覆盖 QKV split、MLP up/down 与 expert matrix；不能只测方阵。

`blockwise` 可以是有用近似，但实验标题和结论必须写 `approximation`，不能写成等价 distributed Muon。

## 实验卡模板

```text
question / competing explanations:
date / owner:
code commits / dependency versions:
hardware / topology / software:
model / parameterization / seed:
dataset / tokenizer / exact tokens:
optimizer routing assertion:
Muon: lr, momentum, nesterov, wd, scale mode, coefficients, steps, dtype, TP mode
scalar optimizer: type, lr, betas, eps, wd groups
schedule / warmup / batch / accumulation:
changed variable:
controlled variables:
failure / stop criteria:
metrics and profiler ranges:
result:
source-reported fact vs local observation:
interpretation and non-claims:
confidence: verified | supported | plausible | open
next discriminating check:
artifact links (no checkpoints/datasets by default):
```

## 完成定义

一轮 Muon 学习实验只有同时满足以下条件才算完成：

- 配置和 commit 足以重跑；参数路由有 exact-cover assertion。
- 数学/shape/state 合同有最小 correctness test。
- 结果至少同时在 token 与 wall-clock 横轴上报告；FLOP 与 memory 在适用时补齐。
- 失败 run、假设、近似、硬件边界和未证明的因果解释没有被隐藏。
- 下一步不是“多跑一点”，而是一个能区分当前竞争解释的检查。
