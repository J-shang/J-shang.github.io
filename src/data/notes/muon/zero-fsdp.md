---
title: "ZeRO 与 FSDP"
description: "参数、梯度与优化器状态如何分片，以及 Muon 的整矩阵语义为何棘手。"
topic: "muon"
section: "engineering"
slug: "zero-fsdp"
legacyPaths: ["/notes/zero-fsdp/"]
date: 2026-07-01
updated: 2026-07-16
order: 32
readtime: 7
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/深度学习工程/ZeRO-FSDP.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B/ZeRO-FSDP.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:4e7120ff2a1ca431db6a695841cbffd13976c57183134b6a474160e5882afc09"
  manifest: "muon"
  managed: true
---
## 先记住什么

ZeRO 和 FSDP 都是在数据并行中切分参数、梯度和优化器状态，让单卡显存不再完整承载整个模型训练状态。

## 核心定义

ZeRO（Zero Redundancy Optimizer）按阶段切分 optimizer state、gradient 和 parameter；FSDP（Fully Sharded Data Parallel）在 PyTorch 生态中提供类似全分片训练能力。它们的共同目标是减少数据并行副本之间的冗余：每张卡只保存一部分状态，需要计算时再 all-gather 或 reduce-scatter。

## 相关知识展开

### 1. 数据并行为什么浪费显存？

普通数据并行中，每张 GPU 都保存完整模型参数、完整 optimizer state，并处理不同 mini-batch shard。梯度通过 all-reduce 同步。这样实现简单，但当模型变大时，重复保存状态非常浪费。

如果有 8 张卡，每张都保存同一份 AdamW state，那么 optimizer state 被复制了 8 份。ZeRO/FSDP 的目标就是消除这种冗余。

### 2. ZeRO 三个阶段的大意

ZeRO 可以粗略理解为：

- **ZeRO-1**：切分 optimizer state；
- **ZeRO-2**：进一步切分 gradients；
- **ZeRO-3**：进一步切分 parameters。

阶段越高，单卡显存越省，但通信和实现复杂度越高。训练时需要在计算前收集参数，计算后释放或重新分片。

### 3. FSDP 的基本循环

FSDP 会把模块参数分片保存。进入某个模块 forward 前 all-gather 出完整参数，计算完成后可以释放；backward 时再按需 all-gather，并 reduce-scatter 梯度。这样用通信换显存。

FSDP 的包裹粒度很重要。包得太大，峰值显存高；包得太小，通信和调度开销大。

### 4. optimizer state 分片和参数语义

对 AdamW 来说，一个参数 shard 的一阶/二阶矩也可以对应切分保存，逐元素更新比较自然。Muon 不同：它需要知道某个二维矩阵参数的完整行列结构，并对更新矩阵做正交化。若只拿到一个 shard，就可能把“局部小矩阵”误当成完整线性算子。

### 5. 为什么 shard 上做 Muon 不一定等价？

设完整矩阵 $M$ 被按列切成

$$
M=[M_1\;M_2].
$$

一般来说

$$
\operatorname{polar}(M)\neq[\operatorname{polar}(M_1)\;\operatorname{polar}(M_2)].
$$

因为奇异向量是全矩阵的全局结构。局部 shard 的奇异方向不等于完整矩阵的奇异方向。

### 6. 分布式 Muon 的可能策略

工程上可能有几种路线：

- all-gather 出完整更新矩阵再正交化；
- 根据张量并行切分设计专门的局部等价或近似算法；
- 对 Q/K/V、MLP、MoE expert 使用不同路由；
- 在通信成本和数学等价之间做近似折中。

这些选择会显著影响 wall-clock，因此不能把 Muon 只看成单机公式。

### 7. 读代码时要盯哪些字段？

看 ZeRO/FSDP + Muon 实现时，重点找：参数原始 global shape、当前 shard shape、optimizer state 是否分片、NS 在 shard 上还是 full tensor 上做、QKV fused 权重是否拆开、通信发生在正交化前还是后、update scaling 使用 global shape 还是 local shape。

## 和 Muon 的关系

Muon 对“完整二维矩阵”的语义很敏感。如果一个线性层权重被分片，局部 shard 未必代表正确的矩阵更新对象。朴素地对 shard 分别做 Muon 可能改变算法含义；更严谨的实现需要在合适维度重组、通信或使用专门的 distributed Muon 方案。ZeRO/FSDP 因此不只是外层工程包装，而会影响 Muon 的正确性和成本。

## 需要掌握到什么程度

- 能区分 ZeRO/FSDP 切分的是 optimizer state、grad 还是 parameter。
- 能说明 all-gather、reduce-scatter 与显存/通信的基本取舍。
- 能理解为什么 Muon 需要知道参数原始矩阵形状和分片方式。
- 能在读 Megatron-Core 或分布式 Muon 代码时关注参数路由和矩阵重组。

## 常见误区

- 认为任何 optimizer 都能无脑套 ZeRO/FSDP。矩阵结构优化器对分片语义更敏感。
- 只看显存下降，不看每步通信和 optimizer step 延迟。
- 把局部 shard 的正交化结果当作全矩阵正交化的等价物；通常不等价。

## 自测问题

1. ZeRO-1、ZeRO-2、ZeRO-3 大致分别切分什么？
2. 为什么 Muon 对局部分片矩阵做 NS 可能改变更新方向？
3. 分布式 Muon 的性能瓶颈可能来自计算还是通信？

## 参考入口

- [Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054) —— ZeRO-1/2/3 的原始状态、梯度和参数分片模型。
- [PyTorch Fully Sharded Data Parallel](https://docs.pytorch.org/docs/stable/fsdp.html) —— 核对当前 all-gather/reduce-scatter、state dict 与包裹策略的实际 API。
- [本地 Megatron-Core `TensorParallelMuon`](https://github.com/NVIDIA/Megatron-LM/blob/0823c731ed7d793aef047b6a64f2dbbf32bf6e2c/megatron/core/optimizer/emerging_optimizers.py) —— 查看矩阵语义如何进入 TP mode 和 adaptive state；具体行为只对页内固定源码版本负责。
