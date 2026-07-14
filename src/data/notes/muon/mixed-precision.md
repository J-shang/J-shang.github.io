---
title: "Mixed Precision"
description: "理解 autocast、loss scaling、master weights，以及 BF16 Muon 的数值边界。"
topic: "muon"
section: "engineering"
slug: "mixed-precision"
legacyPaths: ["/notes/mixed-precision/"]
date: 2026-07-01
updated: 2026-07-01
order: 30
readtime: 6
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/深度学习工程/mixed precision.md"
  url: "https://github.com/J-shang/Muon/blob/7458bf6dbff95ca416a8ca9069308d5cc6907f96/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B/mixed%20precision.md"
  revision: "7458bf6dbff95ca416a8ca9069308d5cc6907f96"
  syncedAt: "2026-07-14"
  contentHash: "sha256:8ae77da5be5c283680a9498ba3e8c12c506a8bc42ec46912a2027b65f2dc6317"
  manifest: "muon"
  managed: true
---
> 层次：深度学习工程

## 一句话定位

mixed precision 是把训练中的不同张量和运算放在不同精度上，以换取吞吐、显存和稳定性的平衡。

## 核心定义

典型混合精度训练会让 forward/backward 的大部分矩阵乘使用 fp16、bf16、tf32 或 fp8，同时保留部分 fp32 计算或 fp32 master copy。fp16 因动态范围较小常配合 loss scaling；bf16 动态范围接近 fp32，通常更易稳定，但尾数位更少。混合精度不是“把所有东西都转半精度”，而是为不同环节选择合适 dtype。

## 相关知识展开

### 1. mixed precision 的目标不是“越低越好”

训练需要同时满足三件事：算得快、放得下、数值稳定。低精度能提升吞吐和降低显存，但也可能带来溢出、下溢和舍入误差。mixed precision 的核心是把不同环节放在不同精度上，而不是一刀切。

常见策略是：大矩阵乘使用 bf16/fp16/tf32；归一化、softmax、loss、优化器关键状态保留更高精度；必要时维护 fp32 master weights。

### 2. autocast 做了什么？

PyTorch AMP 的 autocast 会根据算子类型自动选择较低精度或保留 fp32。例如 GEMM/conv 往往可以低精度，某些 reduction 或数值敏感算子则保留 fp32。但 autocast 主要覆盖 forward；optimizer state、手写矩阵乘和自定义 kernel 不一定自动安全。

Muon 实现如果在 optimizer step 里手动写 NS，需要明确内部 dtype。不能假设开了 AMP，Muon 的正交化就自动使用最佳精度。

### 3. loss scaling 为什么存在？

fp16 梯度可能太小，下溢成 0。loss scaling 把 loss 乘以一个较大系数 $S$，反传得到放大的梯度 $Sg$，在 optimizer step 前再除以 $S$。如果发现 Inf/NaN，就降低 scale。

bf16 因为指数范围大，通常不需要动态 loss scaling，但尾数更粗；所以它解决的是动态范围问题，不是所有精度问题。

### 4. master weights 与 optimizer state

有些混合精度训练会用低精度参数参与 forward/backward，但保留 fp32 master weights 用于更新。AdamW 也常把一阶/二阶矩保存在 fp32。这样稳定但占显存。

Muon 的状态主要是 momentum。这个 momentum 用 bf16 还是 fp32，会影响内存和正交化质量。若用低精度存 momentum，再用更高精度做 NS，读写与 cast 成本也要计入。

### 5. mixed precision 对 wall-clock 的影响

低精度不保证总时间更短。瓶颈可能在数据加载、通信、optimizer step、kernel launch 或 checkpoint。Muon 尤其要看 optimizer step：如果 NS 矩阵乘没有高效 kernel，forward/backward 变快也可能被 optimizer 抵消。

### 6. 一个检查清单

调 Muon mixed precision 时建议逐项确认：

- forward/backward 是否启用 AMP；
- attention softmax 和 norm 是否稳定；
- gradient unscale 是否在 clipping 前完成；
- momentum state dtype 是什么；
- NS 内部是否使用足够稳定的 accumulate dtype；
- update cast 回参数 dtype 前是否做了缩放和检查。

## 和 Muon 的关系

Muon 除了模型 forward/backward，还在 optimizer step 内部做矩阵正交化。也就是说，mixed precision 策略要覆盖 momentum buffer、NS 迭代、权重衰减和最终参数更新。若只优化模型计算而忽略 Muon 的矩阵乘 dtype，可能出现 optimizer step 成为瓶颈，或在低精度下近似质量下降。

## 需要掌握到什么程度

- 能说明 fp16、bf16、tf32、fp32 的大致角色。
- 知道 autocast 主要影响算子选择，不等于所有状态都自动安全。
- 能解释 loss scaling 的目的，以及为什么 bf16 常可少依赖它。
- 能在 Muon 实验里记录 forward dtype、state dtype、NS dtype 和 accumulate dtype。

## 常见误区

- 把 mixed precision 当成单一开关。不同框架和硬件默认策略差异很大。
- 忽略归一化、softmax、optimizer update 这些对数值更敏感的环节。
- 认为 Muon 节省 AdamW 二阶状态后，dtype 就不重要了；正交化矩阵乘仍然可能很贵。

## 自测问题

1. fp16 和 bf16 的主要差异是什么？
2. 为什么 optimizer 内部计算也应纳入 mixed precision 设计？
3. 如果 wall-clock 没变快，你如何判断瓶颈是在 forward/backward 还是 optimizer step？

## 参考入口

- PyTorch Automatic Mixed Precision 文档。
- NVIDIA Mixed Precision Training 文档。
- Liu et al., *Muon is Scalable for LLM Training* 的工程实现部分。
