---
title: "张量并行"
description: "Column/row parallel、fused QKV 和局部正交化中的高频正确性陷阱。"
topic: "muon"
section: "engineering"
slug: "tensor-parallelism"
legacyPaths: ["/notes/tensor-parallelism/"]
date: 2026-07-01
updated: 2026-07-16
order: 33
readtime: 8
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/深度学习工程/张量并行.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:45bbaae8111c8ec49ba029b8e8378c58b94f7147b0c1aac61fbb532d85cb53d0"
  manifest: "muon"
  managed: true
---
## 先记住什么

张量并行把单个大矩阵或算子切到多张卡上计算，是大模型训练中处理超大线性层的核心并行方式。

## 核心定义

与数据并行复制完整模型不同，张量并行会沿行、列或更复杂维度切分权重矩阵。例如 Transformer 的 MLP up/down projection、attention QKV projection 可以按输出通道或输入通道拆到不同 GPU。计算时需要配套的 all-reduce、all-gather 或 reduce-scatter，保证数学结果等价于未切分矩阵。

## 相关知识展开

### 1. 张量并行和数据并行的区别

数据并行是每张卡放一份完整模型，喂不同数据；张量并行是一个层本身太大，于是把矩阵乘拆到多张卡上共同完成。同一个训练任务里，两者常常同时存在：比如 TP×DP×PP 的三维并行。

### 2. column parallel 线性层

对线性层

$$
Y=XW,
$$

如果按输出列切分

$$
W=[W_1\;W_2\;\cdots\;W_k],
$$

每张卡计算一部分输出

$$
Y_i=XW_i.
$$

这适合把输出 hidden dimension 拆开。后续如果需要完整 $Y$，就要 all-gather；如果下一层也按匹配方式切分，可能延迟通信。

### 3. row parallel 线性层

如果按输入维度切分

$$
W=\begin{bmatrix}W_1\\W_2\\\vdots\\W_k\end{bmatrix},\qquad
X=[X_1\;X_2\;\cdots\;X_k],
$$

每张卡计算局部乘积 $X_iW_i$，最后求和：

$$
Y=\sum_i X_iW_i.
$$

这通常需要 all-reduce 或 reduce-scatter。

### 4. QKV fused 是 Muon 的高频坑

实现里 attention 的 Q、K、V projection 常被融合成一个大矩阵，提高 kernel 效率。但语义上 Q、K、V 是三个不同线性算子。Muon 如果直接对 fused QKV 整体正交化，就可能把三者的奇异方向混在一起。

更稳的做法通常是按 Q/K/V 语义拆分后分别路由，再结合张量并行布局决定局部或全局正交化策略。

### 5. MoE expert 的额外复杂度

MoE 模型里每个 expert 有自己的线性层，可能还叠加 expert parallel。Muon 要处理的不只是一个大矩阵，而是许多 expert 权重矩阵。不同 expert 的 token 负载不均，也可能让 momentum 统计差异变大。

### 6. update scaling 要用哪个 shape？

Muon 的 update RMS 与矩阵形状有关。张量并行下，每张卡看到的是 local shard shape，但算法语义可能对应 global weight shape。如果缩放用错 shape，不同 TP size 下训练行为会变，复现实验也会变得困难。

### 7. 张量并行下的性能账

Muon 增加的 NS 矩阵乘可能在局部 shard 上较小、kernel 效率不高；也可能因为避免完整 all-gather 而更快。最终要看 GEMM 形状是否适合硬件、是否需要额外通信、正交化和主训练计算能否 overlap、fused 权重拆分是否破坏原有高效 kernel。

## 和 Muon 的关系

Muon 要对二维线性权重的更新矩阵做正交化，而张量并行改变了每张卡看到的矩阵形状。对 row-parallel、column-parallel、QKV fused 权重，Muon 需要知道“语义矩阵”到底是哪一块：Q、K、V 最好按语义分开；MLP 分片要匹配原始线性算子；MoE expert 还要考虑 expert 并行。错误切分会让正交化对象错位。

## 需要掌握到什么程度

- 能说明 row parallel 和 column parallel 的大致通信模式。
- 能识别 fused QKV 权重为什么需要 split 后路由。
- 能理解张量并行下局部矩阵 shape 与全局语义矩阵 shape 的差别。
- 能读懂分布式 Muon 实现里关于 TP mode、split QKV、expert shard 的配置。

## 常见误区

- 把“二维 tensor”直接等同于“一个完整线性算子”。张量并行下它可能只是 shard。
- 忽略 fused 权重的语义边界；Q/K/V 混在一起正交化通常不是想要的更新。
- 只比较 optimizer 计算时间，忘记 TP 通信也会改变 wall-clock。

## 自测问题

1. column-parallel 线性层和 row-parallel 线性层分别常需要什么通信？
2. 为什么 fused QKV 对 Muon 参数路由是一个坑？
3. 如果 TP size 改变，Muon 的局部矩阵 shape 和更新缩放可能如何变化？

## 参考入口

- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053) 与本地 Megatron-Core 代码 —— 建立 column/row parallel 的 global/local shape 与 collective 时序。
- [本地 `TensorParallelMuon`](https://github.com/NVIDIA/Megatron-LM/blob/0823c731ed7d793aef047b6a64f2dbbf32bf6e2c/megatron/core/optimizer/emerging_optimizers.py) —— 逐分支核对 `duplicated`、`distributed`、`blockwise` 和 QKV split。
- [Liu et al., *Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982) —— 对照论文中的通信与状态目标和实际框架实现，不默认二者版本等价。
