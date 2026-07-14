---
title: "Token、FLOP 与 Wall-clock"
description: "建立公平比较框架：更少 token、更少 FLOP 和更短墙钟时间不是同一件事。"
topic: "muon"
section: "experiments"
slug: "token-flop-wall-clock"
legacyPaths: ["/notes/token-flop-wall-clock/"]
date: 2026-07-01
updated: 2026-07-01
order: 43
readtime: 7
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/LLM 实验方法/token-FLOP-wall-clock 公平比较.md"
  url: "https://github.com/J-shang/Muon/blob/7458bf6dbff95ca416a8ca9069308d5cc6907f96/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/LLM%20%E5%AE%9E%E9%AA%8C%E6%96%B9%E6%B3%95/token-FLOP-wall-clock%20%E5%85%AC%E5%B9%B3%E6%AF%94%E8%BE%83.md"
  revision: "7458bf6dbff95ca416a8ca9069308d5cc6907f96"
  syncedAt: "2026-07-14"
  contentHash: "sha256:63e822886f2deb56d612d5ce402205d1d83add9b386169a0cbd502156a06cbba"
  manifest: "muon"
  managed: true
---
> 层次：LLM 实验方法

## 一句话定位

token、FLOP 和 wall-clock 是三种不同效率口径；Muon 讨论中最常见的误读，就是把其中一种胜利当成全部胜利。

## 核心定义

**token efficiency** 关心达到同等 loss 需要多少训练 token。**FLOP efficiency** 关心达到同等 loss 消耗多少计算量，通常要计入模型大小、序列长度、训练步数和主要算子。**wall-clock efficiency** 关心真实时间，受硬件利用率、通信、kernel、数据加载、checkpoint、optimizer step 等影响。三者相关但不等价。

## 相关知识展开

### 1. token efficiency：每个 token 学到多少

token efficiency 关心达到同样验证 loss 需要多少训练 token。若优化器 A 用 80B tokens 达到某个 loss，优化器 B 需要 100B tokens，那么 A 的 token efficiency 更好。

但 token efficiency 不关心每个 token 算得贵不贵。如果 A 每 token 的计算更重，FLOP 或 wall-clock 未必更省。

### 2. FLOP efficiency：算力预算是否更省

FLOP efficiency 关心达到同样 loss 消耗多少浮点运算。对 dense Transformer，训练 FLOPs 常粗略近似为参数量、token 数和常数因子的乘积。对 MoE，则要区分 active 参数和 total 参数。

Muon 的 NS 正交化也消耗 FLOPs。若论文只按 forward/backward 估算训练 FLOPs，却不计 optimizer step，可能高估 Muon 的 compute efficiency。

### 3. wall-clock efficiency：真实时间账

wall-clock 是用户最直观关心的：训练跑了多久。它受很多 FLOP 之外的因素影响，例如 GPU 利用率、kernel 是否 fused、通信开销、数据加载、checkpoint、optimizer step、pipeline bubble、straggler 和故障恢复。

Muon 如果 token/FLOP 更省，但 NS kernel 很慢或分布式通信重，wall-clock 可能不占优。

### 4. step 数最容易误导

step 数下降不代表训练更便宜。因为

$$
\text{tokens}=\text{steps}\times\text{global batch tokens}.
$$

如果 batch 翻倍、step 减半，token 数不变；如果 batch 翻倍、step 只减少 25%，token 数反而增加。任何“更少 step 达到同样 loss”都必须换算成 token 和 FLOP。

### 5. 公平 A/B 的最低要求

比较 Muon 和 AdamW 时，至少应固定或明确报告：模型架构和参数量、tokenizer 与数据顺序、global batch tokens、token budget、学习率和 schedule、weight decay 和参数路由、dtype 与并行策略、optimizer step 时间是否计入。

如果两边都充分调参，回答的是“各自上限”；如果只替换 optimizer，回答的是“配方迁移能力”。这两个问题都可以研究，但不能混在一起。

### 6. 一张实用记录表

每次实验可以记录：

| 指标 | 记录方式 |
|---|---|
| token efficiency | 达到固定 validation loss 的 tokens |
| FLOP efficiency | 达到固定 validation loss 的估算训练 FLOPs |
| wall-clock | 达到固定 validation loss 的小时数 |
| optimizer overhead | optimizer step 占总 step 时间比例 |
| throughput | tokens/s/GPU 或 tokens/s/cluster |

有了这张表，讨论 Muon “快不快”才不会变成各说各话。

## 和 Muon 的关系

Muon 可能让 loss 按 token 或 FLOP 更快下降，但每步 optimizer 需要 NS 矩阵乘和可能的分布式通信。如果实现未优化，wall-clock 可能没有同步改善。反过来，若 Muon 允许更大 batch 或减少 optimizer state，让系统吞吐提升，也可能带来实际时间收益。评估 Muon 必须同时报告这些口径。

## 需要掌握到什么程度

- 能为一次实验分别计算或近似 token 数、训练 FLOPs 和实际耗时。
- 能解释 step 数、token 数和 batch size 的关系。
- 能识别论文中的“speedup”分母是什么。
- 能在 A/B 设计里保证数据顺序、token budget、模型规模和调参预算尽量公平。

## 常见误区

- 把“达到同样 loss 的 step 更少”直接等同于“训练更便宜”。
- 只报告 GPU hours，不说明硬件、并行策略、实现成熟度和利用率。
- 忽略 optimizer step 成本；Muon 的 NS 计算和通信必须计入 wall-clock。

## 自测问题

1. batch size 翻倍而 step 数减半时，token 数是否一定变化？
2. 一个 optimizer token efficiency 更好，但每步慢 30%，最终 wall-clock 可能怎样？
3. 公平比较 Muon 和 AdamW 时，哪些配置必须记录在实验表里？

## 参考入口

- Hoffmann et al., *Training Compute-Optimal Large Language Models*。
- Liu et al., *Muon is Scalable for LLM Training*。
- Essential AI, *Practical Efficiency of Muon for Pretraining*。
