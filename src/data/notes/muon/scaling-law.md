---
title: "Scaling Law"
description: "读懂 compute-optimal 比较，不把特定拟合区间的优势误写成普遍规律。"
topic: "muon"
section: "experiments"
slug: "scaling-law"
legacyPaths: ["/notes/scaling-law/"]
date: 2026-07-01
updated: 2026-07-16
order: 40
readtime: 7
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/LLM 实验方法/scaling law.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/LLM%20%E5%AE%9E%E9%AA%8C%E6%96%B9%E6%B3%95/scaling%20law.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:55ca5894baa41db263df819ea5ce8f53c86fe9f8e1288da4633f8d3796754493"
  manifest: "muon"
  managed: true
---
## 先记住什么

scaling law 用经验规律描述模型规模、数据规模、计算量和 loss 之间的关系，是判断优化器是否真的提升训练效率的坐标系。

## 核心定义

LLM scaling law 通常研究 loss 随参数量 $N$、训练 token 数 $D$、计算量 $C$ 的变化。经典形式会拟合幂律项，例如 loss 随 $N$ 或 $D$ 增大而下降但边际收益递减。Chinchilla 风格结果强调在固定计算预算下，模型大小和 token 数需要配平；只增大模型或只增加数据都可能偏离 compute-optimal。

## 相关知识展开

### 1. scaling law 想回答什么问题？

训练大模型太贵，不可能把所有规模都完整试一遍。scaling law 的目标是用一系列小到中等规模实验，拟合 loss 与规模变量之间的关系，再预测更大规模下的趋势。

典型问题包括：参数量翻倍，loss 大概降多少；token 数翻倍，收益是否还明显；固定 FLOPs 时，应该训练更大模型还是喂更多 token；某个优化器的优势是否随规模保持。

### 2. 三个核心变量：N、D、C

常见记号是：

- $N$：模型参数量；
- $D$：训练 token 数；
- $C$：训练计算量，通常近似与 $ND$ 成正比。

非常粗略地，dense Transformer 训练 FLOPs 常被估成

$$
C\approx 6ND.
$$

这里的 6 是经验近似，真实数值会受架构、sequence length、attention、MoE active 参数等影响。读论文时要看作者如何计算 FLOPs。

### 3. 为什么会有 compute-optimal？

如果模型太小、token 太多，模型容量不足，后面 token 的收益变低；如果模型太大、token 太少，模型没训练够，也浪费参数。compute-optimal 说的是：在固定计算预算下，选择合适的 $N$ 和 $D$ 配比，让 loss 最低。

Chinchilla 之后，一个重要经验是许多早期大模型相对“参数太多、token 不够”。但这个结论不是永恒常数，数据质量、架构、目标任务和训练 recipe 都会影响最佳配比。

### 4. isoFLOP 曲线怎么读？

isoFLOP 是固定计算预算，扫描不同模型大小和 token 数。比如同样 $C$ 下训练 300M、600M、1B 模型，各自 token 数配平，然后看哪个 loss 最低。

如果 Muon 论文说在 compute-optimal scaling law 设置下更省 FLOPs，你要看它是否做了类似扫描，而不是只拿一个模型、一个 batch、一个 learning rate 做单点比较。

### 5. 拟合和外推的风险

scaling law 通常要外推。外推越远，越依赖假设。风险包括小模型瓶颈与大模型瓶颈不同、数据重复在长训练里改变收益、optimizer 在大 batch 下行为变化、工程实现让 wall-clock 与 FLOPs 脱钩、MoE 的 active 参数和 total 参数让 $N$ 定义变复杂。

### 6. Muon 结果应该放在哪张图里？

最理想的 Muon 评估不只画 loss-vs-step，而要画：

- loss-vs-token：token efficiency；
- loss-vs-FLOP：compute efficiency；
- loss-vs-wall-clock：真实训练时间；
- 不同规模下的趋势：是否随 $N,D,C$ 保持优势。

只有这样才能说清楚 Muon 是“优化更快”，还是“每步更贵但 token 更省”，或者“FLOP 省但墙钟没省”。

## 和 Muon 的关系

Muon 论文中“达到 AdamW 同等表现需要更少 FLOPs”这类说法，必须放在 scaling-law 实验设计里理解：它比较的是在一组模型、数据、batch 和训练预算下拟合出的效率曲线，不是任意单次训练都自动 2× 加速。Muon 的收益也可能随规模、batch、路由、NS 成本和实现质量改变。

## 需要掌握到什么程度

- 能区分参数规模、token 数、训练 FLOPs 和最终 loss。
- 能解释 compute-optimal 不是“最大模型”或“最多数据”，而是预算下的配比。
- 能读懂论文里的 isoFLOP、loss-vs-compute 曲线和 extrapolation。
- 能判断一个 Muon 结果是在小模型 proxy、正式 scaling sweep，还是单点大训练。

## 常见误区

- 把某个单点实验的 step 数减少当作 scaling law 结论。
- 忽略数据质量、tokenizer、架构改动和训练 recipe 对 scaling 曲线的影响。
- 认为 scaling law 外推天然可靠；外推范围越远，不确定性越大。

## 自测问题

1. 固定 FLOPs 时，为什么模型参数量和 token 数需要共同选择？
2. Muon 的 “FLOP efficiency” 结论和 “wall-clock speedup” 有什么差别？
3. 一个只在 125M 模型上做的 A/B，能否直接预测 10B 结果？

## 参考入口

- [Kaplan et al., *Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) —— 读取参数量、数据量与 loss 的早期幂律拟合；重点看拟合区间和外推假设。
- [Hoffmann et al., *Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) —— 理解固定 compute 下模型/数据配比怎样改变，避免沿用单一 scaling law。
- [Liu et al., *Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982) —— 查看 Muon/AdamW 的 compute-optimal 对照；只在其模型族、数据和拟合设置内解释“约 2×”。
