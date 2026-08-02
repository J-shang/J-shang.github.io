---
title: "Reasoning with Latent Thoughts：逐篇解析"
description: "对齐 iso-parameter、iso-FLOP 与 middle-looping 比较，判断 effective depth、reasoning 与 latent thoughts 的真实证据。"
topic: "looped-transformer"
section: "core"
slug: "reasoning-with-latent-thoughts"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
featured: true
order: 24
source:
  repository: "J-shang/looped-transformer"
  path: "papers/08-reasoning-with-latent-thoughts.md"
  url: "https://github.com/J-shang/looped-transformer/blob/9ab82eeb3178ddd627b592ac2cba22de91e7be66/papers/08-reasoning-with-latent-thoughts.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-08-02"
  contentHash: "sha256:c9c6667ba14e25276be2093d10b3a3380187a9eea53ebea8f33e291ffdf0f038"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份

- 论文：*Reasoning with Latent Thoughts: On the Power of Looped Transformers*
- 作者：Nikunj Saunshi、Nishanth Dikkala、Zhiyuan Li、Sanjiv Kumar、Sashank J. Reddi
- 版本：arXiv:2502.17416v1，2025-02-24
- 发表状态：ICLR 2025
- 主来源：[arXiv](https://arxiv.org/abs/2502.17416)、[HTML 全文](https://arxiv.org/html/2502.17416)
- 阅读范围：正文 §1–6、Table 1–3/8、Figure 1–4/7、Theorem 5.1–5.4 与 Appendix A/B
- 信息截止：2026-07-24

## 30 秒结论

**[论文报告]** 许多 reasoning 任务主要需要更大有效深度，不必同比增加独立参数。一个 $k$ 层模型循环 $L$ 次，在 addition、multi-hop 与 1B language modeling 的推理任务上可接近甚至超过同 FLOPs 的 $kL$ 层 untied 模型；但 perplexity 和 closed-book memorization 往往更差。

**[综合判断]** 论文支持“recurrent depth 是 reasoning 的一个可扩展轴”，但没有证明 hidden states 就是可读自然语言思维，也没有获得免费 FLOPs。

## 5 分钟论文地图

1. §2、Table 1–2：addition、$p$-hop、合成数学。
2. §3.1、Table 3：1B causal LM。
3. §3.2、Figure 2/7：同 perplexity 下的下游偏置。
4. §3.3：middle looping 与 gradual stacking。
5. §3.4、Figure 3–4：effective-depth scaling 与 latent thoughts。
6. §4：looping-based regularization。
7. §5、Theorem 5.1–5.4：group composition、模拟 untied 层、$p$-hop、CoT。
8. Appendix A/B：训练细节、全部表格和构造证明。

前置知识：causal language modeling、FLOPs/parameter accounting、multi-hop reasoning、构造性模拟。最小例子是 1 个 block 调 12 次：独立参数仍只有一层，但串行 block calls 与 12 层模型相当。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $k$ | 保存的物理 block 数 | 正整数 | 架构超参数 |
| $L$ | 每组 block 的循环次数 | 正整数 | 训练/推理计算超参数 |
| $kL$ | effective depth / block calls | 正整数 | 派生计算量 |
| $h^{(t)}$ | 第 $t$ 个 effective depth 的序列状态 | $n\times d$ | activation |
| $F_\theta$ | 被循环的 block 或 block group | sequence-to-sequence map | $\theta$ 可训练且跨 loop 共享 |
| $p$ | $p$-hop 任务的链长 | 正整数 | task difficulty |
| $T$ | 理论 CoT 模拟中的推理步数 | 正整数 | 构造预算；不同于实验 loop $L$ |

本文“iso-param”固定独立参数规模，“iso-FLOP”近似固定 block calls；两种对照回答不同问题。理论中的 $T$ 与实验的 $L$ 作用相似但语境不同。

## 贡献账本与论证链

```text
reasoning 可能更依赖有效深度而非独立参数
  → 用同一 block 重复提供串行计算
  → 做 iso-param 与 iso-FLOP 三角对照
  → 在合成任务和 1B LM 分离 reasoning/memorization 表现
  → 用构造定理说明 recurrent depth 的表达能力
```

| 可检查贡献 | 类型与最近基线增量 | 支持位置 | 没有建立 |
|---|---|---|---|
| reasoning 的 recurrent-depth scaling 证据 | 经验发现 | §2–3、Table 1–3/8、Figure 3–4 | 所有 reasoning 都只需共享深度 |
| 同 perplexity 下的任务偏置分析 | 经验分析 | §3.2、Figure 2/7 | looping 是差异的唯一因果机制 |
| middle looping/regularization 变体 | 机制与经验发现 | §3.3–4 | 已找到最优共享位置 |
| group、untied 模拟、$p$-hop、CoT 构造 | 理论 | §5、Theorem 5.1–5.4 | SGD 会学到可读 latent CoT |

## 公平比较框架

记 $k$-layer 模型 loop $L$ 次：

- 与 $k$-layer baseline 近似 iso-param；
- 与 $kL$-layer baseline 近似 iso-FLOP、iso-effective-depth。

这个三角比较防止把“少参数”误写成“少计算”：

| 模型 | 独立参数 | block 执行数 |
|---|---:|---:|
| shallow baseline | $k$ | $k$ |
| looped | $k$ | $kL$ |
| deep untied | $kL$ | $kL$ |

**最小账本。** 若每个 block 有 100M 参数、单次调用 10 GFLOPs，则 1 block × 12 loops 约有 100M block 参数和 120 GFLOPs；12 untied blocks 约有 1.2B block 参数和同量级 120 GFLOPs；1 shallow block 则约 100M 和 10 GFLOPs。真实 embedding/head/attention 长度项会使数值偏离，这只是核算原则。

![Iso-parameter, looped, iso-FLOP, and middle-looping designs](/assets/looped-transformer/08-reasoning-with-latent-thoughts/figure-1-looping-baselines.png)

*原图：Figure 1，PDF p. 2；来源：arXiv:2502.17416v1。看图重点：looped 模型与 shallow baseline 对齐独立参数量，与 deep baseline 对齐有效深度/FLOPs；middle looping 则保留首尾独立层。读后续结果时必须先确认在比较哪条轴，不能把 iso-param 和 iso-FLOP 结论混在一起。*

## 实验证据：问题—结果—边界

| 实验问题 | 设置与观察 | 支持结论 | 剩余不确定性 |
|---|---|---|---|
| addition 是否需要独立层参数？ | 12-layer 为 100%；1-layer 约 0%，1-layer ×12 为 99.6%–100%，2-layer ×6 也近 100%（Table 1） | 在该分布，串行有效深度比独立参数更关键 | 任务生成规则较合成 |
| $p$-hop 是否受益于重复计算？ | 1-layer 约 49%，1-layer ×6 为 99.5%–99.9%，接近 6-layer（Table 2） | 共享 block 可完成多跳传播 | 不能识别内部是否逐 hop |
| 推理 loops 是否形成 scaling curve？ | 合成数学：8-layer 73.2%；1-layer ×2/4/8 为 52.3/69.9/73.2% | 增加 recurrent compute 逐步提升 | 只覆盖有限 loop 范围 |
| 1B LM 的能力偏置是什么？ | Pile 250B tokens；looped perplexity/closed-book 较差，open-book/math/reasoning 更接近或超过深 baseline；reasoning primitive 平均 47.5（24L）、55.3（8L×3）、56.9（4L×6）（Table 3/8） | 共享深度改变能力组合 | benchmark 分组、污染、方差仍是混杂 |

![Scaling with effective depth across language-model task groups](/assets/looped-transformer/08-reasoning-with-latent-thoughts/figure-3-scaling-behavior.png)

*原图：Figure 3，PDF p. 9；来源：arXiv:2502.17416v1。看图重点：橙线靠新增独立层增加深度，蓝线靠增加 loops；两者在 perplexity、closed-book、open-book、math 与 reasoning primitives 上的斜率不同。尤其 reasoning primitives 中蓝线可与橙线相当或更陡，但 perplexity 更差，说明共享深度改变的是能力组合，而非在所有指标上统一占优。各点来自有限规模与任务分组，斜率差仍是经验相关证据。*

### Claim–evidence map

| Claim | 证据 | 强度 | Gap |
|---|---|---|---|
| reasoning 更依赖 effective depth | 合成三类任务 + 1B task groups | moderate–strong | “reasoning”由 benchmark operationalize |
| looping 带来不同于 perplexity 的偏置 | checkpoint isoplot、Figure 2/7 | moderate | 相关分析，不是随机干预机制证明 |
| latent states 可替代显式 CoT 的表达能力 | Theorem 5.4 构造 | strong（定理假设内） | 不说明训练得到的状态可解释 |

## “同 perplexity 更会推理”如何判断

作者在多个训练 checkpoint 上拟合 log perplexity 与下游分数关系。

- **[论文报告]** closed-book QA 的 looped 与 baseline 拟合线接近；
- **[论文报告]** open-book、math、reasoning primitives 中，looped 在同 perplexity 下更高；
- **[论文报告]** 原因仍是 open question。

**[综合判断]** 这是 inductive-bias 的相关证据，不是因果机制证明。拟合范围外外推、任务分组方式和 benchmark contamination 都需要复验。

## Middle looping

作者保留独立的前后层，只循环中间 block。**[论文报告]** 它通常比全模型循环有更好 perplexity、更均匀下游提升，说明首尾层可能承担特殊角色。

这与 ALBERT 的共享消融、Loopie 的相邻 layer-loop 动机形成一致线索：并非所有 depth role 都适合共享。

## 理论部分

- **[论文报告]** Theorem 5.1：1-layer block 重复对数级 loops 可做有限群 composition，并接近已知深度界。
- **[论文报告]** Theorem 5.2：含有限种 distinct layers 的 untied Transformer，可用一个更宽的 1-layer looped Transformer加 depth counter 模拟。
- **[论文报告]** Corollary 5.3：$p$-hop 可由常数物理层、随问题规模增长的 loops 求解。
- **[论文报告]** Theorem 5.4：在固定输入长度等条件下，looped 模型用 $T$ loops 可模拟另一个模型的 $T$ 步 CoT。

### Latent thoughts 不是隐藏文字

**[综合判断]** 理论构造通过 dummy tokens、masking、shift、counter 和编码/解码模拟 CoT。它说明内部并行状态具有至少同样的表达能力，不说明真实模型每轮状态能逐句翻译成自然语言 reasoning trace。

## Looped regularization

论文还训练完整深模型，同时鼓励对应层参数相似，试图兼得独立容量与 looping bias。Appendix 表格显示对 math word problems 和 reasoning primitives 有提升，但这属于额外训练方法，不应与严格权重共享结果混为一谈。

## 局限

- 1B/Pile 规模小于现代旗舰 LLM；
- 训练与推理 FLOPs 没有因参数共享而消失；
- loops 严格串行，延迟可能比同 FLOPs 宽并行计算更差；
- reasoning/memorization 二分依赖任务分组；
- 理论是构造性模拟，不是 SGD 学到 latent CoT 的证明；
- 论文只覆盖一部分 reasoning 类型，作者明确提出多模态、常识等仍未知。

## 超出论文：区分“更多深度”与“共享偏置”

**[扩展假设] Proposal：** 在固定 block calls 下连续插值共享程度：全 untied、相邻两次共享、分组共享、全共享。

- Reasoning chain：iso-FLOP 的两个端点同时改变参数量与共享距离；中间设计可定位是压缩、正则化还是迭代复用主导。
- Predicted observation：memorization 随独立参数增加单调改善；algorithmic reasoning 在中等或完全共享处出现峰值。
- Falsification condition：所有任务仅由总参数量单调解释，分享拓扑无额外作用。
- Minimum experiment：相同数据顺序、tokens、block calls 与 optimizer-step time；至少 3 seeds；报告 perplexity 和各任务组，不只平均分。
- Cost/risk：不同共享拓扑的最佳初始化/学习率不同，需要等额调参预算。

## 推荐复现

1. 精确复现 iso-param、iso-FLOP 两类 baseline。
2. 分开报告 perplexity、closed-book、open-book、math、algorithmic tasks。
3. 对每个 checkpoint 画 loss–downstream isoplot。
4. 对全模型 loop 与 middle loop 做对照。
5. 测 inference loops 超过训练范围后的收益、饱和与 overthinking。

## 一句话带走

**许多 reasoning 能力需要的是串行有效深度，而不一定需要每一步都有新参数；looping 能提供这种深度，但要付出真实计算和延迟。**
