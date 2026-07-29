---
title: "Block-Recurrent Transformers：逐篇解析"
description: "复原 token-block recurrence、recurrent state 与 gating，厘清 sequence recurrence 和 depth recurrence 的不同循环轴。"
topic: "looped-transformer"
section: "adjacent"
slug: "block-recurrent-transformers"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
order: 31
source:
  repository: "local/looped-transformer"
  path: "papers/10-block-recurrent-transformers.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-07-29"
  contentHash: "sha256:16f829422cd85ef6fe8d65fa57f8a3261ee4e28b54f1d3c96a8e1f0d8480d341"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份与证据范围

- 论文：*Block-Recurrent Transformers*
- 作者：DeLesley Hutchins、Imanol Schlag、Yuhuai Wu、Ethan Dyer、Behnam Neyshabur
- 分析版本：arXiv:2203.07852v3，2022-11-02
- 发表状态：NeurIPS 2022 camera-ready
- 主来源：[arXiv](https://arxiv.org/abs/2203.07852)、[全文](https://ar5iv.labs.arxiv.org/html/2203.07852)
- 官方实现：arXiv 论文所链接的开源代码
- 阅读范围：正文 §1–5、Figure 1–3、Table 1–2、方法/消融/定性附录的正文引用
- 信息截止：2026-07-24

## 30 秒结论

**[论文报告]** Block-Recurrent Transformer 把长序列分成 token blocks，在 block 内用 attention 并行，在 block 之间传递一组 recurrent state vectors：

$$
s_{j+1}=G_\theta(s_j,x_j).
$$

它把普通 Transformer layer “旋转 90°”：vertical 路径更新 tokens，horizontal 路径更新 recurrent states，并用 self-/cross-attention 与 gates 交换信息。论文在 PG19、arXiv 和 GitHub 长文本 LM 上，以接近 13-layer sliding baseline 的成本显著改善 bits-per-token。

**[综合判断]** 这篇是本项目的边界对照：它沿 sequence block 循环，而核心 Looped Transformer 沿 model depth 对同一输入状态循环。两者共享参数和状态，但解决的是长时记忆与 latent computation 两个不同问题。

## 5 分钟论文地图

```text
全局 attention 对超长文档昂贵、局部窗口看不到远处
  → 在当前 token block 外维护固定数量的 state vectors
  → token↔state 双向 cross-attention + gated state update
  → block 内并行、block 间递归、跨 segment cache
  → 长文本 LM 对比 sliding/Transformer-XL/memory baseline
  → 证据支持 recurrent state 是有效长距记忆，不等于 depth reasoning
```

阅读顺序：

1. Figure 1、§3.1：vertical/horizontal cell。
2. §3.2–3.5：state IDs、gate、初始化与配置。
3. §3.6：成本对照。
4. Table 1、§4.1–4.3：主要结果与消融。
5. §4.6：状态主要用于何种长距信息。

前置知识：sliding-window attention、Transformer-XL cache、truncated BPTT、LSTM gate。最小例子是把 8,192 tokens 切成 16 个 512-token blocks：每个 block 内并行，只有 16 次 horizontal recurrence。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $j$ | 当前 token block 索引 | $0,\ldots,J-1$ | sequence-time index |
| $W$ | local attention window/block size | token 数，论文主设置 512 | 超参数 |
| $S$ | 每个训练 segment 长度 | token 数，论文主设置 4096 | 超参数 |
| $x_j$ | 第 $j$ 个 token block 表示 | $W\times d$ | runtime input/activation |
| $s_j$ | 进入 block $j$ 的 recurrent states | $M\times d$ | runtime recurrent state |
| $M$ | state-vector 数量 | 正整数；与 $W$ 独立，主实验常令 $M=W$ | 超参数 |
| $G_\theta$ | recurrent cell | $(M\times d,W\times d)\to M\times d$ | $\theta$ 可训练并跨 $j$ 共享 |
| $\odot,\sigma$ | 逐元素乘与 sigmoid | 固定算子 | 运算 |
| $g$ | fixed/LSTM gate 值 | $M\times d$ 或广播向量 | activation / learned constant |

token/state vector 为行，feature 为列；省略 batch。$j$ 沿文档推进，不是 depth step。缓存到下一 training segment 的 state/KV 被视为 non-differentiable，形成 truncated BPTT（§3、Figure 2）。

## 贡献账本

| 可检查贡献 | 类型与最小增量 | 支持位置 | 没有建立 |
|---|---|---|---|
| block-level recurrent Transformer cell | 架构机制 | Figure 1、§3.1 | 同一输入上的 latent-depth refinement |
| 大规模 state vectors + state IDs | memory mechanism | §3.2 | 每个 state 有稳定可解释语义 |
| gate/初始化稳定化 | 训练机制 | §3.3–3.5 | LSTM gate 必然优于 fixed gate |
| 与一层额外 Transformer 近似等成本 | 系统核算 | §3.6、Table 1 | 所有硬件/实现速度相同 |
| 长文本 LM 增益 | 经验发现 | §4、Table 1–2 | 长程 reasoning 的因果提升 |

## 方法复原

### 1. Baseline：sliding attention

文档分为长度 $S$ 的 training segments，再细分成大小 $W$ 的 blocks。每个 block 只对自身和前一个 block 做 causal attention，因此 local attention matrix 大小由 $W$ 控制；总成本对 segment length $S$ 线性、对 window $W$ 二次（§3、Figure 2）。

前一 segment 的最后 block K/V 被 non-differentiable cache 到下一 step，扩展可见上下文但截断跨 segment 梯度。

### 2. Cell 的两条路径

对 block $j$：

```text
vertical:
  token self-attention(xj)
  + token queries cross-attend(sj)
  → next token representations

horizontal:
  state self-attention(sj)
  + state queries cross-attend(xj)
  → gated next states s[j+1]
```

vertical 类似多一项 cross-attention 的 Transformer layer；horizontal 用 gate 替代 residual。K/V 在两方向共享，但 query 不共享（§3.1）。

整体 recurrence 为：

$$
s_{j+1}=G_\theta(s_j,x_j).
$$

这里 $s_j\in\mathbb{R}^{M\times d}$ 和 $x_j\in\mathbb{R}^{W\times d}$ 都是 runtime tensors，$\theta$ 是跨 block $j$ 重复使用的 attention/MLP/gate 参数。输出状态形状始终是 $M\times d$，因此记忆容量不随文档长度增长。

**三-block trace。** $s_0$ 是初始/缓存状态；处理 $x_0$ 得 $s_1$，处理 $x_1$ 得 $s_2$，处理 $x_2$ 得 $s_3$。预测 $x_2$ 中 token 时可通过 $s_2$ 间接读取 $x_0$ 信息，但不能直接访问已经离开 local window 的 token 表示。

![Vertical token path and horizontal recurrent-state path](/assets/looped-transformer/10-block-recurrent-transformers/figure-1-recurrent-cell.png)

*原图：Figure 1，PDF p. 2；来源：NeurIPS 2022 camera-ready。看图重点：左图是普通纵向 token-layer 视角，右图把同一 cell 旋转成沿 token blocks 推进的 horizontal recurrence；token 与 recurrent states 通过两组 cross-attention 交换信息，state 路径用 gate 替代 residual。循环索引是“新 block 到来”，不是“同一输入上增加 latent reasoning depth”。*

### 3. State IDs

所有 state vectors 共用同一投影参数。若初始化完全对称且没有身份信号，它们可能给出相同 query。论文为每个 state 加 learned state ID，本质类似 state 轴上的 learned position embedding（§3.2）。

### 4. Gate

fixed gate 可抽象为：

$$
s_{j+1}=g\odot s_j+(1-g)\odot \tilde s_j,
$$

其中 $\tilde s_j$ 是当前 attention/MLP candidate，$g=\sigma(b)$ 是训练后对所有 block 固定的可训练广播向量；$\odot$ 是逐元素乘。它近似 exponential moving average。

LSTM gate 则让 forget/input gate 依赖当前 $s_j$ 和输入，表达力更强（§3.3），但实验中 fixed gate 往往更好（§4.2）。

### 5. 成本

论文 12-layer 模型把第 10 层设为 recurrent；该 cell 的额外 attention 成本近似再加一个普通层，所以与 13-layer Transformer-XL/sliding baseline 比较。论文报告参数范围约 151M–164M，并按相同 tokens/optimizer step 调整 batch（§3.6、§4.1）。

这不是“无限上下文免费”：state 是有损压缩，且 blocks 仍按 $j$ 串行。

## 实验证据：问题—设置—结果—边界

Table 1 的 bits-per-token 越低越好：

| 实验问题 | 设置与准确结果 | 支持结论 | 剩余不确定性 |
|---|---|---|---|
| recurrent state 是否优于多一层？ | Slide:13L：arXiv 1.42、GitHub 1.17、relative step time 1.00；Rec:fixed:skip：1.24、0.976、0.99（Table 1） | 近似成本下 recurrence 明显改善长文本 LM | PG19 token 列的 HTML 排版混杂，数值应看原表 |
| 是否胜过更大 attention window？ | XL:2048：arXiv 1.31、GitHub 1.01、step time 2.11；Rec:fixed:skip 仍为 1.24/0.976、0.99 | state 压缩可比 4× window 更有效/快 | 依赖论文 TPU 实现 |
| 扩大模型是否保留收益？ | 40M–1.3B scale sweep，Figure 3；650M/1.3B PG19 word PPL 28.46/26.50（Table 2） | 现象跨论文测试的规模存在 | 与外部 published baseline 的 vocab/配方不完全相同 |
| 哪种 gate 更好？ | fixed 往往优于更强的 LSTM gate（Table 1、§4.2） | 表达力更强不等于更好优化 | 没有穷尽 gate 调参 |
| 更多 recurrent layers/states 是否总有益？ | 邻接两层 recurrence 无收益；states 从 128 增到 1024 小幅改善，2048 变差（§4.3） | 存在容量/优化边界 | 原因只有作者假设 |
| state 保存什么？ | 随机 5 本书的 top-4 改善 token 中 17/20 是 proper names，19/20 不在 window（§4.6） | state 至少用于长距 lookup | 只有 20 个选择性样本，不能概括全部机制 |

![Scaling comparison between Block-Recurrent Transformer and Transformer-XL](/assets/looped-transformer/10-block-recurrent-transformers/figure-3-scaling.png)

*原图：Figure 3，PDF p. 8；来源：NeurIPS 2022 camera-ready。看图重点：40M 到 1.3B 参数范围内，Block-Recurrent 曲线始终低于同预算 Transformer-XL，且差距在较大规模没有消失。横轴按参数量、比较在论文协议中对齐 FLOPs；它支持该长文本 LM 架构的 scaling 现象，不能证明 recurrent state 在做一般推理。*

### Claim–evidence map

| Claim | 证据 | 强度 | Gap |
|---|---|---|---|
| block recurrence 是有效长距 memory | Table 1、§4.6 | strong（论文协议内） | 不是无损 memory |
| 比 long-range TXL 快约 2× | relative step time 0.99 vs. 2.11 | moderate–strong | 实现/硬件依赖 |
| 成本近似普通一层 | §3.6、参数/FLOPs/step time | strong（该实现） | 不同 kernels 可能改变 |
| 适合 reasoning recurrence | 结构类比 | weak | 论文没有 depth-reasoning 干预 |

## 与 Looped Transformer 的精确边界

| 维度 | Block-Recurrent | Recurrent-depth / Looped |
|---|---|---|
| 迭代索引 | token block $j$ | effective depth $t$ |
| 每步输入 | 新 block $x_j$ | 通常同一 prompt/state |
| 目标 | 压缩过去长上下文 | 增加当前样本串行计算 |
| state | 固定 $M$ 个 memory vectors | 全序列 hidden state |
| 超范围测试 | 更长文档/更多 blocks | 更多 loops/更难推理 |
| 主要瓶颈 | memory compression、sequence recurrence | depth latency、稳定性、共享冲突 |

二者可以组合，但不能用 Block-Recurrent 的长文档 PPL 证明 depth looping 学会算法。

## 局限与开放问题

- recurrent state 跨 training segments 的梯度被截断；
- blocks 沿 sequence 串行，长 segment 内并行不能消除整个文档串行性；
- state size 固定，是有损瓶颈；
- gate 初始化敏感，模型可能陷入完全忽略 state 的 local optimum（§3.4）；
- 2048 states 反而更差，多 recurrence layers 收益有限（§4.3）；
- strongest 定性证据主要是 proper-name lookup，不是多步推理；
- 结果基于特定 TPU/sliding-window 实现与 2022 LM 配方。

## 超出论文：sequence recurrence × depth recurrence

**[扩展假设] Proposal：** 每个 token block 内先做 $R$ 次共享 depth refinement，再只把最终状态写入 sequence memory。

- Reasoning chain：sequence recurrence解决“记住什么”，depth recurrence解决“当前 block 计算多久”；二者应在长文档 multi-hop 任务上互补。
- Predicted observation：普通 LM PPL 的增益有限，但需要跨 blocks 组合两条证据的任务随 $R$ 改善。
- Falsification condition：相同 block calls 下，增加 ordinary untied depth 始终等效或更好。
- Minimum experiment：长距 name lookup、跨块 two-hop QA、PG19 LM；控制参数、block calls、memory $M$，报告 latency 与 state utilization。
- Cost/risk：双重串行使训练吞吐显著下降，且 BPTT 截断会阻断跨块 reasoning credit。

## 复现与阅读路径

1. Figure 1 → §3.1 → §3.6：先画清两条路径和成本。
2. Table 1 → §4.3 → §4.6：再看主要证据、负面消融和状态用途。
3. 最小复现：两个 32-token blocks、8 个 state vectors，检查 $s_{j+1}$ 只依赖 $s_j,x_j$。
4. 监控 gate saturation、state attention entropy、把 state 置零后的 PPL 差。
5. 必须对照：同参数/同 step time 的普通额外层，而不只是相同 12-layer baseline。

## 一句话带走

**Block-Recurrent Transformer 用 recurrent states 把长文档历史压缩到 block 之间传递；它证明 sequence-block recurrence 是有用记忆机制，但不是 depth-loop reasoning 的直接证据。**
