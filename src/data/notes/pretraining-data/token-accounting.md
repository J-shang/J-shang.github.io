---
title: "Token 核算与数据单位"
description: "统一 raw、unique、sampled、non-padding 与 loss token，建立可从配置推导和从运行计数器复核的分母。"
topic: "pretraining-data"
section: "foundations"
slug: "token-accounting"
date: 2026-07-14
updated: 2026-07-15
order: 10
readtime: 16
source:
  repository: "J-shang/pt-data-learning"
  path: "knowledge-map/00-foundations/token-accounting.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/knowledge-map/00-foundations/token-accounting.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:7d129c03b4d94222eb783c10e6e63367a7a20bac1bde95bae3d4f2b069885d0a"
  manifest: "pretraining-data"
  managed: true
---
> 层级：00 Foundations
> 状态：`core`
> 初始资料核查截止：2026-07-14
<!-- maintenance: reasoning-path=`constraint-driven derivation` -->
> 证据说明：计数恒等式在声明的 loader contract 下为 `verified`；跨 tokenizer/任务的效应判断通常为 `supported` 或 `plausible`

## 这篇笔记帮助你回答什么

Token accounting 是把原始数据规模、采样配方和训练计算统一到可审计分母上的方法，是解释所有数据指标和实验预算的前置条件。

## 为什么需要这个概念

两个团队都说“训练了 1T tokens”，仍可能使用了不同 tokenizer、重复曝光、packing、padding 和 loss mask。所需性质不是再造一个总量，而是建立一套**可以从配置推导、从运行计数器复核、并能定位差异来源**的共同计量语言。

这里采用教学重构而非历史叙事：先要求“名义预算必须等于可解释的计数分解”，再从 batch 与 mask 约束推出各类 token。

## 先看一个最小例子

两个 run 都读取一个长度为 2048 的 sequence。Run A 的 2048 个位置全部计入 loss；Run B 有 400 个 padding/prefix 位置被 mask。二者的 `sampled_tokens` 都是 2048，但 `loss_tokens` 分别是 2048 和 1648。仅比较“训练 tokens”会把输入吞吐与优化目标分母混为一谈。

## 核心定义

对一个冻结 tokenizer $\tau$ 和文档集合 $D=\{d_i\}_{i=1}^n$，unique token 数是：

$$
N_{\text{unique}}(D;\tau)=\sum_{i=1}^{n}|\tau(d_i)|
$$

它依赖 tokenizer、规范化、是否插入 BOS/EOS，以及文档在 tokenize 前还是后被去重。训练过程实际采样的 token 数 $N_{\text{sampled}}$ 可大于或小于它；真正进入 loss 分母的 $N_{\text{loss}}$ 还要扣除 padding、被 mask 的边界/前缀等位置。

## 这些结论依赖哪些前提

| 关系或结论 | 类型与条件 | 置信状态 |
|---|---|---|
| $N_{\text{sampled}}=U W b a S$ | `exact identity`，仅当 $W,b,a,S$ 固定、每步完整执行且没有动态长度/丢 batch | `verified` |
| $N_{\text{loss}}=\sum m_{u,j}$ | `exact identity`，前提是 $m$ 与训练框架实际 loss normalization 使用同一 mask | `verified` |
| $N_{\text{loss}}\approx N_{\text{sampled}}\bar\rho$ | `approximation`；用平均有效比例代替逐位置求和，误差来自 step/domain 的 mask 波动 | `verified within approximation` |
| $r_k=E_k/N_k$ 表示平均曝光倍数 | `approximation`；假设 domain 内近似均匀 token sampling，忽略长度、内部重复和 sampler correlation | `supported` |
| fertility 影响跨语言/代码的有效字符覆盖 | `empirical association`；依赖 normalization、tokenizer revision、统计单位和语料 | `supported` |

本文默认 causal LM、固定 tokenizer revision、明确 BOS/EOS 和 loss mask 语义。动态 seqlen、token-based batching、sequence parallel 对 $WbaS$ 的简单形式有影响时，应直接累计实际 sampled/loss counters，不套用固定长度公式。

## 机制与相关知识

### 1. 六个不可互换的规模单位

| 单位 | 适合回答 | 主要陷阱 |
|---|---|---|
| raw bytes | 存储、网络和解析成本 | 压缩格式/编码不同，不代表语言内容量 |
| Unicode chars | 文本长度粗估 | 不同文字系统和 normalization 不可直接公平比较 |
| documents | source/pipeline yield | 文档长度重尾，均值很容易误导 |
| unique tokens | 冻结语料的模型输入量 | 强依赖 tokenizer 和去重位置 |
| sampled tokens | 训练期间的总曝光 | 重采样会重复相同 unique token |
| loss tokens | 优化目标实际分母 | padding/mask/packing 可造成名义预算偏差 |

因此，数据 manifest 至少记录：`bytes_raw`、`documents_in/out`、`chars_out`、`tokenizer_id`、`unique_tokens`；训练日志至少记录：`sampled_tokens`、`loss_tokens`、`padding_tokens` 和 per-domain exposure。

### 2. 从 global batch 到训练 token

假设 data-parallel world size 为 $W$，每个 rank 的 micro-batch 为 $b$，梯度累积步数为 $a$，seqlen 为 $S$，optimizer steps 为 $U$。没有动态长度时，名义 sampled tokens 为：

$$
\begin{aligned}
B_{\text{seq}} &= Wba,\\
N_{\text{sampled}} &= U\,B_{\text{seq}}\,S
\end{aligned}
$$

这里 $B_{\text{seq}}$ 是每个 optimizer step 的全局 sequence 数。若每个 sequence 的有效 loss mask 比例为 $\rho_u$，则：

$$
N_{\text{loss}}=\sum_{u=1}^{U}\sum_{j=1}^{B_{\text{seq}}S}m_{u,j}
\approx N_{\text{sampled}}\bar{\rho}
$$

实现 invariant：数据 loader 的累计 `loss_mask.sum()` 应与训练框架用于 loss normalization 的分母一致；否则不同 packing 策略的 loss 不能公平比较。

### 3. Tokenizer fertility 与跨域比较

对文档 $d$ 的 fertility 可定义为 tokens/character 或 tokens/word：

$$
F_{\text{char}}(d;\tau)=\frac{|\tau(d)|}{|d|_{\text{char}}}
$$

它不是文本质量分数，而是 tokenizer 表示效率的一个代理。同一个 10 GB 语料用两个 tokenizer 可能对应不同 token 预算；同一 tokenizer 对中文、英文、代码的 fertility 也不同。因此：

- 按 byte/char 控制的数据对照，不等价于按 training tokens 控制。
- 按 token 计算 mixture 会隐式受到 tokenizer 对不同 domain 编码效率的影响。
- 比较跨模型 perplexity 时，如果 tokenizer/vocabulary 不同，token-level PPL 通常不能直接横比；可补充 bits-per-byte 等更接近共同单位的量，但仍需统一评估协议。

### 4. Packing、边界和“浪费”

若把多个文档 packing 到长度 $S$ 的 sequence，设第 $j$ 个 sequence 中 non-padding tokens 为 $v_j$，packing efficiency 为：

$$
\eta_{\text{pack}}=\frac{\sum_j v_j}{JS}
$$

但高 $\eta_{\text{pack}}$ 不自动等于高数据质量。还要检查：

- 是否为每个文档插入 EOS，避免模型把相邻文档当连续语义。
- attention 是否允许跨文档，loss 是否包含边界 token。
- 超长文档是截断、滑窗还是跨 sequence 延续；三者改变上下文分布。
- 短文档是否反复组合造成 source/domain 的 batch 相关性。

### 5. 最小数值例子

设 $W=8$、$b=2$、$a=4$、$S=2048$、$U=1000$：

$$
N_{\text{sampled}}=8\times2\times4\times2048\times1000
=131{,}072{,}000
$$

若 2% 为 padding、0.5% 为不计 loss 的特殊/边界位置，且两者不重叠，则：

$$
N_{\text{loss}}=131{,}072{,}000\times(1-0.02-0.005)
=127{,}795{,}200
$$

若某代码域含 10M unique tokens，但目标 mixture 给它 20%，则预期 sampled exposure 是 26.2144M tokens，近似重复曝光 $2.62$ 次。报告“代码占 20%”而不报告这个倍数，会隐藏小域 oversampling。

### 6. 可执行的 token accounting 流程

```text
for each frozen domain shard:
    verify source/config/version/hash
    count documents, bytes, normalized characters
    tokenize with exact tokenizer revision
    count unique tokens and length quantiles

for each training step:
    count sampled tokens by domain
    count loss-mask tokens, padding, boundary tokens
    compare cumulative counts to sampler expectation

at checkpoint:
    persist counters with model/data/config revisions
```

最小正确性测试：用 3 个长度已知的人工文档，手算 BOS/EOS、padding 和 loss mask，再逐项比对 loader 输出。先通过这个 fixture，才在大数据上相信计数器。

## 它怎样影响 pretraining data 工作

关系类型：`prerequisite-for` 可比的数据实验；`implemented-by` loader/sampler/mask counters。

这是所有数据实验的共同坐标系。过滤器改变 unique tokens；mixture 改变 sampled tokens；packing/masking 改变 loss tokens；validation 指标的加权分母也通常是 loss tokens。若不先统一口径，data recipe 的“等预算对照”很可能实际上没有等预算。

## 读完后应该掌握什么

- 能从训练配置手算名义 sampled tokens，并从 loss mask 得到实际 loss tokens。
- 能解释 tokenizer、去重、packing、重采样分别改变哪一种计数。
- 能为每个 domain 计算 unique tokens、target exposure 与重复曝光倍数。
- 能写出一个 3-document fixture 验证 loader 计数和边界语义。

## 常见误区

- 把 dataset card 的 token 数直接当作自己的训练 token 数，忽略 tokenizer/version。
- 把一次 epoch 当作跨数据集可比单位；streaming mixture 和 oversampling 下 epoch 常无统一语义。
- 只看 seqlen 与 step 数，不检查 padding、mask 和 dropped remainder。
- 用不同 tokenizer 的 token-level perplexity 直接排名模型或语料。

## 用这些问题检查自己

1. 两个实验都报告训练 100B tokens，但一个把 prompt prefix mask 掉 20%，另一个不 mask。它们在哪些意义上不等预算？你会补报什么？
2. tokenizer B 对代码的 fertility 比 tokenizer A 高 30%。在固定 10B sampled tokens 时，如何影响可见字符量和 mixture 解释？
3. 一个占 mixture 1% 的小域只有 50M unique tokens，总训练预算 1T tokens。估算曝光倍数，并设计两个过拟合/记忆风险诊断。

## 来源与建议阅读位置

- [Hoffmann et al., 2022, Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — 从 token 数与模型规模共同约束 compute 的实验视角阅读；重点看其 token/parameter 结论依赖的训练设置。
- [DataTrove official repository](https://github.com/huggingface/datatrove) — 阅读 token estimation、tokenization pipeline 和 summary statistics，观察工程计数如何进入 sampler 配置。
- [The Pile paper](https://arxiv.org/abs/2101.00027) — 对照其 component 规模与 sampling weight，练习区分文档量、数据量和训练 exposure。
