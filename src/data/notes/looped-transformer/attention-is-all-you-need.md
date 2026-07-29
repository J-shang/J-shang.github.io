---
title: "Attention Is All You Need：逐篇解析"
description: "从 self-attention、multi-head、mask 与位置编码建立标准 Transformer 基线，明确它与 depth recurrence 的边界。"
topic: "looped-transformer"
section: "foundations"
slug: "attention-is-all-you-need"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
order: 10
source:
  repository: "local/looped-transformer"
  path: "papers/01-attention-is-all-you-need.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-07-29"
  contentHash: "sha256:ab87ab0dd6615a687ed0a1ede60f9dd1f2f74916e0f0eb6c3edd64ff6177356c"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份

- 论文：*Attention Is All You Need*
- 作者：Ashish Vaswani 等
- 版本：arXiv:1706.03762v7，2023-08-02；原工作发表于 NeurIPS 2017
- 主来源：[arXiv 摘要](https://arxiv.org/abs/1706.03762)、[HTML 全文](https://arxiv.org/html/1706.03762)
- 阅读范围：正文 §1–7、Figure 1–2、Table 1–4；本笔记不使用后续 Transformer 变体反推原论文结论
- 信息截止：2026-07-24（只用于本项目中的路线定位）
- 本笔记定位：标准 Transformer 基线；它本身不是 looped Transformer

## 30 秒结论

**[论文报告]** 论文用 self-attention 完全替代序列方向的 recurrence 和 convolution，构成可并行训练的 encoder–decoder。标准 block 由 attention、逐位置 FFN、residual、LayerNorm 组成；后续 Looped Transformer 重复调用的正是这类 block。

**[综合判断]** 对本项目最重要的不是背 BLEU，而是建立两个基准：普通 Transformer 的不同深度通常有不同参数；self-attention 消除了 token 方向的串行依赖，却没有消除 depth 方向逐层执行的串行依赖。Looped Transformer 改造的是第二点。

## 5 分钟论文地图

1. §1–2：RNN、CNN 与 attention 的动机。
2. §3、Figure 1–2：完整 Transformer 架构和 scaled dot-product attention。
3. §3.3–3.5：FFN、embedding、sinusoidal position encoding。
4. §4、Table 1：不同层类型的复杂度、串行操作数和最大路径长度。
5. §5–6、Table 2–4：机器翻译、模型消融和 constituency parsing。
6. §7：结论。

前置知识：矩阵乘法、softmax、residual connection、sequence-to-sequence。最小直觉例子是：一句话有 4 个 token 时，self-attention 在同一层直接构造一个 $4\times4$ 的“谁读取谁”矩阵；RNN 则必须沿 4 个位置顺序推进。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $n$ | 序列长度 | 正整数，token 数 | 输入决定 |
| $d_{\text{model}}$ | hidden width | 正整数，原文 Base 为 512 | 超参数 |
| $Q,K,V$ | query、key、value | 运行时矩阵；一般为 $n\times d_k$、$n\times d_k$、$n\times d_v$ | activation |
| $W_i^Q,W_i^K,W_i^V,W^O$ | 第 $i$ 个 head 的线性投影和输出投影 | 可训练矩阵，形状由 $d_{\text{model}},d_k,d_v$ 决定 | 参数 |
| $h$ | attention head 数 | 正整数，原文 Base 为 8 | 超参数 |
| $H^{(\ell)}$ | 第 $\ell$ 层的整段表示 | $n\times d_{\text{model}}$ | activation |
| $F_{\theta_\ell}$ | 第 $\ell$ 个 Transformer block | 矩阵到矩阵的参数化映射 | 参数化算子 |
| $\operatorname{softmax}$ | 对每个 query 行归一化 | 行和为 1 | 固定算子 |

本文按“token 为行、feature 为列”书写；省略 batch 轴。$\ell$ 是独立层索引，而后续 looped 文献常用 $t$ 表示同一 block 的重复调用次数。

## 问题、逻辑链与贡献账本

当时的 sequence-to-sequence 模型多用 RNN。其关键限制是：第 $i$ 个 token 的 hidden state 依赖前一个 token，因而同一序列内部难以完全并行。卷积可以并行，但远距离交互常需多层传播。论文的逻辑链是：

```text
token 方向的递归限制并行
  → 用全局 self-attention 一次连接所有位置
  → 用位置编码恢复顺序、用 FFN 提供逐位置非线性
  → 训练路径更并行、远距路径更短
  → 翻译与 parsing 实验验证架构可用
```

| 可检查贡献 | 类型与相对基线的增量 | 证据位置 | 不足以证明 |
|---|---|---|---|
| 只靠 attention 构造 encoder–decoder | 架构机制；相对 RNN/CNN 去掉 token recurrence/convolution | §3、Figure 1 | attention 对所有任务都优于 recurrence |
| multi-head 并行读取不同表示子空间 | 机制；单 attention 的多投影版本 | §3.2.2、Table 3(A) | 每个 head 必然对应可解释语言现象 |
| 全局 self-attention 缩短位置间路径 | 复杂度分析 | §4、Table 1 | 在任何硬件上 wall-clock 都更快 |
| 在翻译和 parsing 上达到强结果 | 经验发现 | §5–6、Table 2/4 | 后续 decoder-only LLM 的全部能力都来自同一因素 |

## 方法拆解

### 1. Scaled dot-product attention

论文 §3.2 定义：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

这里 $Q,K\in\mathbb{R}^{n\times d_k}$，$V\in\mathbb{R}^{n\times d_v}$ 都是输入依赖的 activation，不是 optimizer 直接更新的参数；$QK^\top$ 为 $n\times n$。$\sqrt{d_k}$ 是固定缩放。除以它是为了避免维度增大后点积幅值过大，使 softmax 落入梯度很小的饱和区域。

**最小算例。** 两个 token、一个标量 value 时，若某个 query 对两个 key 的缩放后分数为 $(0,\log 3)$，softmax 权重为 $(1/4,3/4)$，输出就是 $v_1/4+3v_2/4$。attention 的核心是“按输入内容生成加权读取”，不是保存一个固定邻接矩阵。

### 2. Multi-head attention

多个 head 使用不同投影：

$$
\operatorname{head}_i
=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V),
$$

这里 $W_i^Q,W_i^K,W_i^V$ 是第 $i$ 个 head 的可训练参数；投影后的三项才是运行时 activation。各 head 输出沿 feature 轴拼接后乘可训练的 $W^O$。它不是简单地“重复同一 attention”，因为每个 head 有独立参数并可形成不同的注意模式。

### 3. Encoder、decoder 与 masking

- Encoder 每层包含 self-attention 和 FFN。
- Decoder 还包含 encoder–decoder cross-attention。
- Decoder self-attention 使用 causal mask，禁止当前位置读取未来 token。
- 每个子层外有 residual，再做 LayerNorm；原论文 Figure 1 是 Post-LN 形式。

### 4. Position encoding

纯 attention 对输入排列本身不敏感，所以必须注入位置信息。原论文采用不同频率的正弦、余弦编码，并报告 learned 与 sinusoidal 版本表现接近（§3.5、Table 3）。

### 视觉证据：标准 block 与 attention 内部结构

![Transformer encoder–decoder architecture](/assets/looped-transformer/01-attention-is-all-you-need/figure-1-transformer-architecture.png)

*原图：Figure 1，PDF p. 3；来源：arXiv:1706.03762v7。看图重点：左侧 encoder 与右侧 decoder 都按 $N$ 层展开，每层有独立的串行 depth 位置；Looped Transformer 后来改变的是这些位置是否复用同一组参数，而不是 attention 的基本数据流。*

![Scaled dot-product attention and multi-head attention](/assets/looped-transformer/01-attention-is-all-you-need/figure-2-attention-mechanisms.png)

*原图：Figure 2，PDF p. 4；来源：arXiv:1706.03762v7。看图重点：左图给出 $QK^\top$、缩放、可选 mask、softmax、与 $V$ 相乘的顺序；右图显示 multi-head 是多组独立线性投影和 attention 并行执行后再拼接，不是把同一结果复制多份。*

两张图只定义了标准 Transformer 的一次 block 调用，不能单独说明跨 depth 共享、循环稳定性或推理时增加 loops 的效果；这些是后续论文新增的问题。

## 实验证据：问题—结果—边界

| 实验问题 | 设置与结果 | 支持的结论 | 剩余不确定性 |
|---|---|---|---|
| 无 recurrence/convolution 的架构能否做好翻译？ | WMT14 En–De：Transformer Big 28.4 BLEU；En–Fr：41.8 BLEU（摘要、Table 2） | 该架构在论文训练协议下有效 | 与今天的数据、tokenizer、解码协议不可直接横比 |
| multi-head 与位置编码怎样影响结果？ | Table 3(A/E)：head 太少或太多会变差；learned 与 sinusoidal position encoding 接近 | 具体配置需要调节；正弦位置编码不是唯一可行方案 | 单组模型消融不能推出普遍最优 head 数 |
| 能否迁移到翻译之外？ | constituency parsing 结果见 Table 4 | 架构不只适用于翻译 | 仍是有限任务集 |

这些结果证明了 2017 年设定下的架构有效性，却不能回答参数共享、循环深度或推理时增加计算是否有效。

### Claim–evidence map

| Claim | 论文证据 | 强度 | Gap |
|---|---|---|---|
| token 方向训练可并行 | 架构依赖图、§4/Table 1 | strong（结构性质） | 不等价于全系统无串行瓶颈 |
| 短路径有利于远距依赖 | Table 1 + 翻译结果 | moderate | 没有隔离“路径长度”做因果消融 |
| multi-head 有价值 | Table 3(A) | moderate | 搜索预算与方差未充分报告 |

## 与 Looped Transformer 的连接

普通深层模型可写为：

$$
h^{(\ell+1)}=F_{\theta_\ell}(h^{(\ell)}).
$$

这里 $h^{(\ell)}\in\mathbb{R}^{n\times d_{\text{model}}}$ 是第 $\ell$ 层运行时状态，$\theta_\ell$ 是该层独立参数；公式表示每加一层通常也增加一组参数。

Looped Transformer 则是：

$$
h^{(t+1)}=F_\theta(h^{(t)},x,e_t).
$$

这里 $t$ 是执行步，$x$ 是固定输入，$e_t$ 是可选 step encoding，唯一的 $\theta$ 在所有步共享。二者使用相似 block，但 looped 版本在不同 effective depth 重用 $\theta$。这会同时改变参数量、梯度聚合、表示分工和训练稳定性。

## 局限与常见误读

- **[论文报告]** 原论文主任务是 encoder–decoder 翻译，不是今天常见的 decoder-only LLM。
- **[综合判断]** Table 1 的 $O(n^2d)$ 只描述标准 attention 主项；真实系统速度还取决于 kernel、显存带宽和硬件利用率。
- **[综合判断]** token 方向可并行不等于整个网络无串行计算，几十层仍需按 depth 顺序执行。
- **[综合判断]** attention 可视化是解释线索，不足以单独证明模型学到了句法规则。

## 超出论文：一个可证伪扩展

**[扩展假设] Proposal：** 在相同参数量和 block 调用数下，把 6 个 untied encoder block 与 1 个 block 循环 6 次比较。

- Reasoning chain：如果有效深度本身足够，权重共享应保留大部分性能；若层级专门化关键，共享会明显掉点。
- Predicted observation：算法任务的差距小于 memorization-heavy 任务。
- Falsification condition：两类任务都表现出同方向、同量级差距。
- Minimum experiment：相同数据、初始化族、token 数、optimizer、5 seeds，同时报告参数、FLOPs、step time 与准确率。
- Cost/risk：共享模型的最优学习率可能不同；完全共用搜索预算会产生新的公平性争议。

## 复现与阅读任务

1. 手写单头 attention，核对 $QK^\top$、mask、softmax、$V$ 的 shape。
2. 实现原始 Post-LN block，再实现 Pre-LN，比较深层梯度。
3. 统计 6 个独立 block 与 1 个 block 循环 6 次的参数、FLOPs 和 activation。
4. 阅读顺序建议：Figure 1 → §3.2 → §3.1/3.3/3.5 → Table 1 → Table 2–3。

## 一句话带走

**Transformer 首先把 token 方向的序列计算变成可并行 attention；Looped Transformer 随后追问，depth 方向的计算能否由同一个可复用规则反复完成。**
