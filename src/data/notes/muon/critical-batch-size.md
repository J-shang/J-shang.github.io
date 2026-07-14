---
title: "Critical Batch Size"
description: "理解 batch 增大何时不再带来等比例优化收益，以及它对 Muon 实验的影响。"
topic: "muon"
section: "experiments"
slug: "critical-batch-size"
legacyPaths: ["/notes/critical-batch-size/"]
date: 2026-07-01
updated: 2026-07-01
order: 41
readtime: 7
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/LLM 实验方法/critical batch size.md"
  url: "https://github.com/J-shang/Muon/blob/7458bf6dbff95ca416a8ca9069308d5cc6907f96/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/LLM%20%E5%AE%9E%E9%AA%8C%E6%96%B9%E6%B3%95/critical%20batch%20size.md"
  revision: "7458bf6dbff95ca416a8ca9069308d5cc6907f96"
  syncedAt: "2026-07-14"
  contentHash: "sha256:2bf3a590ede68425d068e8a554cf3253de55dd1ca60dc9a12f0221ff1091689a"
  manifest: "muon"
  managed: true
---
> 层次：LLM 实验方法

## 一句话定位

critical batch size 描述 batch 增大到某个范围后，继续加 batch 的效率收益开始明显递减。

## 核心定义

小 batch 区间里，增大 batch 往往能近似线性提高硬件吞吐，并允许相应调大学习率或减少更新噪声。但超过某个临界范围后，每个 token 提供的边际优化信息下降，训练可能需要更多 token 或更复杂调参才能达到同样 loss。这个临界点与模型规模、数据、训练阶段、优化器和学习率 schedule 都相关。

## 相关知识展开

### 1. batch size 改变了什么？

global batch size 决定一次 optimizer update 使用多少 token 或样本。batch 增大时，梯度估计更稳定，硬件并行度可能更高；但每个 update 消耗的数据也更多。如果 loss 按 step 下降更快，却每步吃掉更多 token，token efficiency 未必更好。

### 2. 三个 batch 概念

大模型训练里要区分：

- **micro batch size**：单张卡一次 forward/backward 处理的样本数；
- **gradient accumulation steps**：累计多少个 micro batch 后再更新；
- **global batch size**：所有数据并行卡、所有 accumulation 合起来的一次更新总样本/token 数。

常见关系是

$$
B_\text{global}=B_\text{micro}\times \text{DP size}\times \text{accumulation steps}.
$$

序列长度固定时，也可换算成 tokens per update。

### 3. 为什么会有临界 batch？

当 batch 很小时，梯度噪声大，增大 batch 可以显著改善每步方向，并提升硬件利用率。当 batch 超过某个范围后，额外样本提供的信息开始重复，单次 update 的收益不再随 batch 线性增加。

这时继续加 batch 可能仍提高吞吐，但达到同样 loss 所需 token 也会增加。临界 batch size 就是这种收益开始明显递减的区域。

### 4. linear scaling rule 不是万能规则

经典经验是 batch 放大 $k$ 倍，学习率也放大 $k$ 倍，至少在某些范围内有效。但 LLM 预训练里，还要考虑 warmup、AdamW/Muon 动力学、梯度裁剪、序列长度和数据顺序。超过临界 batch 后，简单线性放大学习率往往不再可靠。

### 5. Muon 为什么可能影响 critical batch？

Muon 改变了矩阵更新方向，把 momentum 的奇异值谱压平。若这种方向更充分利用大 batch 提供的稳定梯度，它可能在更大 batch 下保持效率。但这只是需要实验验证的假设，不是从公式直接推出的结论。

评估时应比较不同 global batch 下的 loss-vs-token 和 loss-vs-wall-clock，而不是只看最大可跑 batch。

### 6. 怎么设计 batch 扫描？

一个最小但有用的扫描可以固定模型、数据、token budget，比较 AdamW 小/中/大 batch 与 Muon 小/中/大 batch，并让各自重新调学习率和 warmup，同时记录吞吐、NS/optimizer step 时间、最终 loss。

如果只把 AdamW 的最佳 batch 直接套给 Muon，或只把 Muon 的最佳 batch 直接套给 AdamW，都不公平。

## 和 Muon 的关系

Muon 的一个重要实验方向是大 batch 训练。若 Muon 在更大 batch 下仍能保持 token efficiency 或更好利用更新方向，它就可能提高硬件并行效率。但这不等于“batch 越大越好”：Muon 的 NS/通信成本、更新缩放、weight decay 和稳定性机制都会影响最终 wall-clock。

## 需要掌握到什么程度

- 能解释为什么 batch size 影响梯度噪声和硬件利用率。
- 能区分 global batch size、micro batch size、gradient accumulation steps。
- 能读懂 loss-vs-token 与 loss-vs-step 曲线在 batch 扫描中的差别。
- 能把 critical batch size 当作实验对象，而不是固定常数。

## 常见误区

- 只看每秒 token 数，不看达到目标 loss 需要多少 token。
- 把 gradient accumulation 当成完全等价的大 batch；数值细节、通信和 schedule 仍可能不同。
- 认为某个 optimizer 的最佳 batch size 可以不经调参直接迁移到另一个 optimizer。

## 自测问题

1. global batch size、micro batch size 和 accumulation step 的关系是什么？
2. 为什么大 batch 可能提高吞吐，却降低 token efficiency？
3. 评估 Muon 的大 batch 优势时，至少应画哪些曲线？

## 参考入口

- McCandlish et al., *An Empirical Model of Large-Batch Training*。
- Shallue et al., *Measuring the Effects of Data Parallelism on Neural Network Training*。
- *Practical Efficiency of Muon for Pretraining*。
