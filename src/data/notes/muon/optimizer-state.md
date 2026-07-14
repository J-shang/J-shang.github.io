---
title: "Optimizer State"
description: "逐项计算优化器状态内存，理解 Muon 相对 AdamW 的真实收益。"
topic: "muon"
section: "engineering"
slug: "optimizer-state"
legacyPaths: ["/notes/optimizer-state/"]
date: 2026-07-01
updated: 2026-07-14
order: 31
readtime: 7
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/深度学习工程/optimizer state.md"
  url: "https://github.com/J-shang/Muon/blob/f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B/optimizer%20state.md"
  revision: "f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13"
  syncedAt: "2026-07-14"
  contentHash: "sha256:33d7ed42f4384b33f12bcf7e2cedc35094425b913e606e468fa3bb63ef7874d4"
  manifest: "muon"
  managed: true
---
> 层次：深度学习工程

## 一句话定位

optimizer state 是参数之外为更新规则保存的历史信息，常常决定大模型训练的显存下限。

## 核心定义

训练时显存不只包括参数，还包括梯度、激活、optimizer state 和临时 buffer。以 AdamW 为例，每个参数通常需要一阶矩 $m$ 和二阶矩 $v$，再加上可能存在的 fp32 master weights。SGD momentum 通常只保存一个 momentum buffer。状态数量、dtype 和分片方式都会显著影响可训练模型规模。

## 相关知识展开

### 1. 训练显存由哪些部分组成？

一个大模型训练 step 里，显存通常包括：

- parameters：模型参数；
- gradients：反向传播得到的梯度；
- optimizer state：如 momentum、Adam 的一阶/二阶矩；
- activations：反向传播需要保存的中间激活；
- temporary buffers：通信、矩阵乘、正交化、checkpoint 等临时张量。

只看参数量会严重低估训练成本。一个 7B 参数模型，若参数本身用 bf16 约 14GB，但 AdamW fp32 状态就可能额外需要约 56GB。

### 2. AdamW 状态为什么贵？

AdamW 通常保存

$$
m_t \quad\text{和}\quad v_t.
$$

如果两者都是 fp32，每个参数额外 8 bytes。若还有 fp32 master weights，又多 4 bytes。再加上梯度和参数本体，总训练显存可以是参数文件大小的数倍。

### 3. Muon 状态省在哪里？

标准 Muon 主要保存 momentum buffer，不保存逐坐标二阶矩 $v_t$。对走 Muon 的大矩阵权重，这可以减少 optimizer state。尤其是隐藏层线性权重占参数大头时，节省会比较明显。

但 Muon 不是没有额外成本。NS 迭代需要临时矩阵，分布式情况下可能需要 all-gather、reduce-scatter 或特殊布局。节省的是持久状态，不一定节省所有峰值显存。

### 4. 混合优化器要分别统计

真实 Muon 配方常常是：hidden linear weights 走 Muon；embedding/output head 走 AdamW；bias/norm scale 走 AdamW 或其他简单规则。因此不能说“这个模型用了 Muon，所以状态就是 1 个 buffer”。要按参数组统计：多少参数走 Muon、多少参数走 AdamW、各自 state dtype 是什么。

### 5. state 与 checkpoint

训练 checkpoint 如果保存 optimizer state，AdamW checkpoint 会很大。Muon checkpoint 可能更小，但仍要保存 momentum 和 AdamW 组状态。若只保存模型权重，不保存 optimizer state，恢复训练时优化器历史丢失，loss 曲线可能出现跳变。

### 6. ZeRO/FSDP 怎样改变 state 成本？

ZeRO/FSDP 可以把 optimizer state 分片到多张卡上。比如数据并行度为 $D$，理想情况下每张卡只保存约 $1/D$ 的状态。但通信和重组成本会上升。Muon 的矩阵结构还会让分片语义更敏感：局部 shard 的 momentum 未必是应该正交化的完整矩阵。

### 7. 一个估算模板

估算某参数组状态时，可以写：

$$
\text{state memory}=\text{num params}\times\text{states per param}\times\text{bytes per state}.
$$

例如 10B 个参数走 AdamW，两个 fp32 state 约为 $10^{10}\times2\times4=80$ GB。若同样参数走 Muon 且只保存一个 bf16 momentum，则约 20 GB。实际还要加临时 buffer 和分片策略。

## 和 Muon 的关系

标准 Muon 的主要状态是 momentum，而不是 AdamW 的一阶矩加二阶矩，因此 optimizer-state footprint 有潜在优势。但 Muon 并非“零额外成本”：NS 迭代需要临时矩阵，分布式实现可能需要通信或重组矩阵，非 Muon 参数还常由 AdamW 管理。评估 Muon 内存收益时必须按参数组统计，而不是只看核心公式。

## 需要掌握到什么程度

- 能列出训练显存中的参数、梯度、激活、optimizer state、通信 buffer。
- 能粗略估算 AdamW 与 Muon momentum state 的字节数差异。
- 能理解 ZeRO/FSDP 为什么要切分 optimizer state。
- 能在实验报告中说明哪些参数走 Muon、哪些仍走 AdamW。

## 常见误区

- 只统计模型参数大小，忽略 AdamW 状态可能是参数量的数倍。
- 把“状态更少”直接等同于“训练更快”。计算和通信可能抵消部分收益。
- 忽略临时 buffer 峰值；optimizer step 内部的正交化也会占用显存。

## 自测问题

1. AdamW 对每个参数通常保存哪些状态？
2. Muon 的状态优势在哪些参数组上最明显？
3. 为什么混合优化器需要分别统计 Muon 组和 AdamW 组的状态？

## 参考入口

- DeepSpeed ZeRO paper 与文档 —— 学习 persistent state 按 DP rank 分片后的字节数与通信代价。
- PyTorch FSDP 文档 —— 核对 optimizer state dict 的 full/sharded 形式和恢复约束。
- KellerJordan/Muon 与 Liu et al., *Muon is Scalable for LLM Training* —— 分别读取最小 momentum state 和规模化混合 optimizer/分布式 state 设计。
