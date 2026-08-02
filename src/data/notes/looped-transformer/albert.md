---
title: "ALBERT：逐篇解析"
description: "拆解 factorized embedding、跨层参数共享与训练配方，分清参数效率、执行深度和实际计算成本。"
topic: "looped-transformer"
section: "foundations"
slug: "albert"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
order: 12
source:
  repository: "J-shang/looped-transformer"
  path: "papers/03-albert.md"
  url: "https://github.com/J-shang/looped-transformer/blob/9ab82eeb3178ddd627b592ac2cba22de91e7be66/papers/03-albert.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-08-02"
  contentHash: "sha256:12c9837862ae506e57cfda1c52f9244befd9b6321a99a088d9764ebec16fa0b2"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份

- 论文：*ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*
- 作者：Zhenzhong Lan 等
- 版本：arXiv:1909.11942v6，2020-02-09
- 发表状态：ICLR 2020
- 主来源：[arXiv 摘要](https://arxiv.org/abs/1909.11942)、[PDF](https://arxiv.org/pdf/1909.11942)
- 官方实现：[google-research/albert](https://github.com/google-research/albert)
- 阅读范围：正文 §1–6、Figure 1–2、Table 1–7 与正文所述训练/ensemble 条件
- 信息截止：2026-07-24（官方仓库仅用于确认 artifact，不用后续结果改写原论文）

## 30 秒结论

**[论文报告]** ALBERT 用 factorized embeddings 和跨层参数共享显著减少 BERT 参数，并用 sentence-order prediction 替代 next-sentence prediction。共享参数让“物理参数量”与“执行深度”分离，但模型仍按固定层数执行。

**[综合判断]** ALBERT 是理解 looped 模型成本核算的最佳对照：少参数不代表少计算；weight tying 也不自动代表学会了迭代算法。

## 5 分钟论文地图

1. §2：BERT 扩展遇到的内存与退化问题。
2. §3.1：factorized embedding。
3. §3.2：cross-layer parameter sharing。
4. §3.3：sentence-order prediction（SOP）。
5. §4、Table 1–4：模型大小、参数效率与共享消融。
6. §5：GLUE、SQuAD、RACE 的最终结果与 ensemble 设置。

前置知识：BERT、masked language modeling（MLM）、embedding、参数共享。最小例子是：12 层 BERT 通常保存 12 组 block 参数；完整共享的 ALBERT 保存 1 组、调用 12 次。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $V$ | vocabulary size | 正整数，词项数 | 固定 |
| $E$ | embedding width | 正整数，且常有 $E\ll H$ | 超参数 |
| $H$ | hidden width | 正整数 | 超参数 |
| $L$ | 执行层数 | 12 或 24 等 | 超参数 |
| $X_{\text{emb}}$ | 词嵌入表 | $V\times E$ | 可训练参数 |
| $W_{\text{emb}}$ | embedding-to-hidden 投影 | $E\times H$ | 可训练参数 |
| $h^{(\ell)}$ | 第 $\ell$ 次 block 后的 token 表示 | $n\times H$，省略 batch | activation |
| $F_\theta$ | 共享 Transformer block | $\mathbb{R}^{n\times H}\to\mathbb{R}^{n\times H}$ | $\theta$ 可训练 |

“层数”在本笔记指执行深度 $L$；“物理参数层数”指保存了多少套不同 $\theta$。两者在 ALBERT 中不再相等。

## 贡献账本与论证链

```text
BERT 扩宽时 embedding 和 block 参数迅速膨胀
  → 分解 embedding + 跨层共享
  → 用节省的参数预算扩大 hidden size
  → 用 SOP 修正 NSP 的任务缺陷
  → 预训练与下游任务验证参数效率
```

| 可检查贡献 | 类型与相对 BERT 的增量 | 支持位置 | 不足以证明 |
|---|---|---|---|
| factorized embedding parameterization | 参数化机制：$V\times H$ 改成 $V\times E$ 与 $E\times H$ | §3.1、Table 2 | 总训练 FLOPs 同比例下降 |
| cross-layer parameter sharing | 参数共享机制 | §3.2、Table 3–4 | 学到可无限迭代算法 |
| SOP objective | 训练目标：同文档 segment 顺序判别 | §3.3、Table 5 | 所有句间任务都由 SOP 改善 |
| 大规模 ALBERT benchmark 结果 | 经验发现 | §4–5、Table 6–7 | 增益可全部归因于共享 |

## 三个方法组件

### 1. Factorized embedding

BERT 把 vocabulary embedding size 直接等于 hidden size $H$，embedding 参数为 $O(VH)$。ALBERT 引入较小的 embedding 维度 $E$：

$$
O(VH)
\quad\longrightarrow\quad
O(VE+EH),
\qquad E\ll H.
$$

这里 $V,E,H$ 都是维度；左侧是一张 $V\times H$ 可训练表，右侧由 $X_{\text{emb}}\in\mathbb{R}^{V\times E}$ 和 $W_{\text{emb}}\in\mathbb{R}^{E\times H}$ 两组参数构成。这让 hidden size 可以增大，而词表矩阵不同比例膨胀。

**最小算例。** 若 $V=30{,}000,H=768,E=128$，直接 embedding 有 23.04M 参数；分解后为 $30{,}000\times128+128\times768\approx3.94$M，约为前者的 17.1%。这是 embedding 子模块的压缩比，不是整模型速度比。

### 2. Cross-layer sharing

ALBERT 可以共享 attention 参数、FFN 参数，或整层参数。默认完整共享时：

$$
h^{(\ell+1)}=F_\theta(h^{(\ell)}),
$$

这里 $h^{(\ell)}\in\mathbb{R}^{n\times H}$ 是运行时状态，$\theta$ 是 optimizer 更新的唯一共享 block 参数，$\ell=0,\ldots,L-1$。反向传播会把同一 $\theta$ 在 $L$ 次调用产生的梯度相加。形式上与固定次数 loop 相同，但训练目标仍是固定 12 或 24 层的 MLM；论文不要求推理时多执行几层后继续收敛。

![Layerwise input-output representation change in BERT and ALBERT](/assets/looped-transformer/03-albert/figure-1-layer-transition.png)

*原图：Figure 1，PDF p. 4；来源：arXiv:1909.11942v6。看图重点：ALBERT 的跨层变化比 BERT 更平滑，但距离与角度并未趋近于 0；固定次数的共享层仍持续改变表示。因此这张图支持“ALBERT 不等于已收敛 DEQ”，却不能仅凭相似度曲线断言共享造成了性能变化。*

### 3. Sentence-order prediction

SOP 的负样本来自同一文档但交换两个 segment 顺序，避免 NSP 只学到 topic mismatch。它与 looped 架构关系较弱，但属于 ALBERT 完整贡献，不能把整篇论文缩减成“层共享”。

## 参数量、深度与速度

**[论文报告]** Table 1：

| 模型 | 层数 | 参数量 |
|---|---:|---:|
| BERT Base | 12 | 108M |
| BERT Large | 24 | 334M |
| ALBERT Base | 12 | 12M |
| ALBERT Large | 24 | 18M |
| ALBERT XLarge | 24 | 60M |
| ALBERT XXLarge | 12 | 235M |

**[论文报告]** Table 2 同时显示：ALBERT Base/Large 参数少且训练更快，但 XLarge/XXLarge 尽管参数仍较省，计算更重，速度相对 BERT Large 分别约 0.6× 和 0.3×。

**[综合判断]** 这直接说明模型文件大小、optimizer state、每 token FLOPs 和 wall-clock 是四个不同量。

![Effects of adding data and removing dropout during ALBERT training](/assets/looped-transformer/03-albert/figure-2-data-and-dropout.png)

*原图：Figure 2，PDF p. 9；来源：arXiv:1909.11942v6。看图重点：在训练中途增加数据或移除 dropout 后，dev MLM accuracy 出现明显跃升；这提醒我们 ALBERT 的最终 benchmark 来自共享、数据、正则化和训练时长的组合，不能把全部增益归给 cross-layer sharing。曲线是训练过程诊断，不是针对共享机制的因果消融。*

## 共享消融透露了什么

**[论文报告]** Table 4 中，完整共享会损失平均表现；共享 FFN 带来的下降最大，而只共享 attention 的损失较小，某一 E=128 设置甚至略有提升。

**[综合判断]** 不同深度并非完全冗余。独立 FFN 可能承担更明显的层级专门化；强行共享会产生“一个算子要兼顾多个深度角色”的冲突。这与 Loopie 后来选择 layer-loop、让共享调用发生在相邻 effective depth 的动机相呼应。

## 实验证据：问题—结果—混杂

| 实验问题 | 设置与观察 | 支持结论 | 剩余不确定性 |
|---|---|---|---|
| 压缩是否保留性能？ | Table 1–3 比较 BERT 与多档 ALBERT 的参数、训练速度和 dev 指标 | 分解与共享能显著降低参数 | 模型 width、训练时长等同时变化 |
| 应共享哪部分？ | Table 4：full sharing 有损；FFN-only sharing 的下降大于 attention-only sharing | FFN 的层级专门化可能更强 | 局部消融不是机制证明 |
| 状态会不会很快到固定点？ | Figure 1：跨层表示仍持续变化 | ALBERT 不等价于 DEQ | 表示相似度不直接刻画任务残差 |
| 最终 benchmark 是否更强？ | XXLarge 在 GLUE/SQuAD/RACE 强；部分最终结果用 checkpoint selection/ensemble（§5、Table 6–7） | 完整训练配方有效 | 不能把全部增益归因于共享 |

因此 ALBERT 不是 DEQ：重复状态可以持续变化，且没有 root solver 或 equilibrium loss。

### Claim–evidence map

| Claim | 证据 | 强度 | Gap |
|---|---|---|---|
| 参数量与执行深度可解耦 | 参数公式、Table 1 | strong | 不涉及无限深度 |
| full sharing 可实用 | 下游结果、共享消融 | moderate–strong | 不是无损且受规模/配方影响 |
| SOP 比 NSP 更合适 | Table 5 的任务消融 | moderate | objective 与数据细节仍耦合 |

## 与 Looped Transformer 的关系

相同点：

- 同一参数跨 effective depth 重用；
- 参数量随物理层数而非展开深度增长；
- 共享引入正则化和可复用变换偏置。

不同点：

- ALBERT 通常固定展开深度；
- 不强调每步输入注入、step encoding 或动态 halting；
- 不以训练外 depth extrapolation、算法收敛或 test-time compute 为主要目标。

## 局限与常见误读

- **[综合判断]** “参数少 10×”不能改写为“训练/推理便宜 10×”。
- **[论文报告]** 完整共享不是无损压缩，Table 4 明确显示性能代价。
- **[综合判断]** 下游 SOTA 同时来自更大 hidden size、更多训练和 ensemble，不能全部归因于共享。
- **[综合判断]** 固定深度共享模型即便形式上是 recurrence，也不等于学到了可无限调用的稳定算法。

## 超出论文：可证伪扩展

**[扩展假设] Proposal：** 训练时随机采样执行深度 $L$，并测试 $L$ 超出训练范围时的 MLM 与下游变化。

- Reasoning chain：固定 $L$ 允许共享 block 记住深度特定轨迹；随机 $L$ 迫使中间状态更可读。
- Predicted observation：随机深度模型在未见 $L$ 上更平滑，但固定深度峰值可能略低。
- Falsification condition：两者在未见深度上同样失稳，或随机化只带来全面退化。
- Minimum experiment：attention-only、FFN-only、full sharing 三种设置，各 5 seeds；报告 loss、hidden norm、cosine drift、参数/FLOPs/step time。
- Cost/risk：不同深度看到的总 block 调用量不同，必须按总调用数或 tokens×calls 归一。

## 推荐复现

1. 在小 BERT 上比较不共享、attention-only sharing、FFN-only sharing、full sharing。
2. 同时报告参数量、optimizer memory、FLOPs、step time。
3. 记录各层 hidden cosine distance，检查表示是否收敛。
4. 在训练层数外继续执行共享层，观察 MLM loss 和表示范数是否发散。

## 一句话带走

**ALBERT 证明跨层共享可以极大压缩存储参数，但也清楚展示：共享深度不等于免费计算，更不等于自动获得可外推的迭代算法。**
