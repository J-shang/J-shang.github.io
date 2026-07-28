---
title: "MUDDFormer：为 Q、K、V、Residual 建立四路动态 dense connections"
description: "追踪 MUDDFormer 的四路 token-wise depth mixing，结合 2.8B、300B-token 实验检查质量、吞吐、显存与 artifact 完整度。"
topic: "residual"
section: "methods"
slug: "muddformer"
date: 2026-07-27
updated: 2026-07-27
cutoff: 2026-07-27
order: 16
readtime: 17
source:
  repository: "J-shang/residual"
  path: "papers/07-muddformer.md"
  url: "https://github.com/J-shang/residual/blob/c55583707ab645bef7408c588fc33cbf91b809a5/papers/07-muddformer.md"
  revision: "c55583707ab645bef7408c588fc33cbf91b809a5"
  syncedAt: "2026-07-28"
  contentHash: "sha256:c8b48119603c0583ea62c3b423ddbf4592d59ab0a012a4749ab75160edd57364"
  manifest: "residual"
  managed: true
---
<!-- paper-order: 07 -->

> **论文**：Da Xiao、Qingye Meng、Shengping Li、Xingyuan Yuan, *MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections*<br>
> **机构**：北京邮电大学、ColorfulClouds Technology（彩云科技 / Caiyun）<br>
> **版本**：ICML 2025 / PMLR 267；arXiv:2502.12170v2，2025-05-28<br>
> **状态**：ICML 2025 正式发表，不是 Independent Researcher 投稿<br>
> **主来源**：[PMLR 正式页面](https://proceedings.mlr.press/v267/xiao25d.html) · [arXiv](https://arxiv.org/abs/2502.12170) · [PDF](https://arxiv.org/pdf/2502.12170v2) · [OpenReview](https://openreview.net/forum?id=qkhgzNiEdj)<br>
> **官方 artifact**：[Caiyun-AI/MUDDFormer](https://github.com/Caiyun-AI/MUDDFormer)，公开 JAX 训练、PyTorch 推理与 pretrained models<br>
> **阅读范围**：完整 19 页，包括方法、伪代码、复杂度、300B-token 实验、效率、内存、ViT 与可视化附录<br>
> **信息核对日期**：2026-07-27<br>
> **前置阅读**：[DenseFormer](/topics/residual/denseformer/) · [DeepCrossAttention](/topics/residual/deepcrossattention/)

## 证据标签

- **[论文报告]**：论文正文、公式、图表或附录直接报告。
- **[复原推导]**：由论文定义可直接推出。
- **[综合判断]**：结合证据与工程约束形成的解释。
- **[扩展假设]**：需要额外实验。
- **[待验证]**：当前材料不足。

---

## 先给结论

### 30 秒版

MUDDFormer 把 DenseFormer 的一个静态 depth mixture 连续扩展两次：

1. **Dynamic**：不同 token 位置生成不同 depth weights；
2. **Multiway**：Q、K、V、Residual 四种角色各有自己的 depth weights。

给定所有 block outputs：

$$
\mathcal X_i=\{X_0,\ldots,X_i\},
$$

MUDD 为每个 token 生成：

$$
A_i\in\mathbb R^{B\times T\times4\times(i+1)},
$$

并得到：

$$
X_i^c
=
\sum_{j=0}^{i}
A_{i,j}^{c}\odot X_j,
\qquad
c\in\{Q,K,V,R\}.
$$

下一 block 不再把同一个 $X$ 同时当 Q/K/V/residual，而是：

$$
X_A'
=
\operatorname{MHA}
(\operatorname{LN}(X^Q),
 \operatorname{LN}(X^K),
 \operatorname{LN}(X^V))
+X^R.
$$

这不是四条持久 residual streams。持久保存的仍是每层单一 block output history；四路 state 是为下一 block 动态聚合出的四个 views。

### 最强实验结果

**[论文报告]**

- scaling experiments 从 405M/7B tokens 到 1.4B/26B tokens，MUDDFormer consistently 优于 Transformer++、DenseFormer、HC 和单路 DDFormer。
- 拟合估计：834M MUDDFormer 达到相同 validation loss 时，Transformer++ 需约 $1.89\times$ compute。
- 300B tokens：
  - MUDDPythia-2.8B 的 Pile PPL 6.29，与 Pythia-6.9B 相同；
  - 0-shot 平均 55.0 vs Pythia-6.9B 55.1；
  - 5-shot 平均 57.0，高于 Pythia-6.9B 的 56.4，接近 Pythia-12B 的 57.2。
- Transformer++ recipe 的 MUDDFormer-2.8B：Pile PPL 6.01、0-shot 56.9、5-shot 58.4，超过表中 Pythia-12B。
- 额外理论 FLOPs 仅约 0.4%，但实测训练吞吐为 baseline 的 84%–96%、推理为 88%–94%，训练峰值内存增加约 17%–29%。

最后一条尤其重要：

> 轻量参数 / FLOPs，不等于轻量 wall-clock / memory。

### 我的总判断

**[综合判断]** 四篇中，MUDDFormer 的大规模语言模型证据最强，也公开了最完整 artifact。它把“动态、分路 depth routing”从 50M–449M 的 DCA 实验推进到了 2.8B、300B tokens，并给出真实训练/推理效率与内存。

它最值得保留的疑问是：

- 大规模主结果看起来是 single run，没有 seeds / error bars；
- parameter re-allocation 与 MUDD 主机制共同出现，虽有小规模 ablation，但会增加归因复杂度；
- “2.46× compute match”比较的是不同模型 size 的 Pythia，不能简化成端到端成本一定降低 2.46×；
- history activation 带来的 20%–30% 内存增量不可忽略。

---

## 5 分钟方法地图

| 层面 | MUDDFormer |
|---|---|
| motivating pressure | Pre-Norm 表示趋同；单 residual stream 作为跨层通信通道可能过载 |
| stored state | $\{X_0,\ldots,X_i\}$，每层一个 block output |
| read rule | 四个 DA 聚合器从同一 history 读 Q/K/V/R |
| write rule | multi-input block 产生一个新 $X_i$，追加到 history |
| residual update | Attention output 加到 $X^R$；FFN 再做普通 residual |
| weight granularity | token × destination depth × source depth × way |
| dynamic generator | RMSNorm + 2-layer MLP + static prior |
| initialization | $W_2=0$，static prior one-hot-current，精确恢复 Transformer |
| deep composition | 无 stochastic manifold 约束；可选 Pre/Post-DA RMSNorm |
| optional re-allocation | FFN width 从 $0.5D_f$ 线性增至 $1.5D_f$ |
| sparse variants | dilation×period、sliding window |
| main system cost | history lifetime、四路 aggregation、memory traffic |

### 贡献账本

| 类型 | 贡献 | 新颖性与证据 |
|---|---|---|
| conceptual | 将跨层通信视为 residual-stream bandwidth 问题 | 与 circuit composition / representation collapse 联系 |
| methodological | token-wise dynamic dense connection | 超越 DenseFormer 静态权重 |
| architectural | Q/K/V/R 四路独立 depth mixtures | MUDD 核心贡献 |
| optimization | depth-wise FFN parameter re-allocation、可选 PrePostDANorm | 辅助设计，不应与核心机制混同 |
| empirical | 405M–2.8B、7B–300B tokens、dense/MoE/ViT | 四篇中规模证据最强 |
| systems | 理论 FLOPs、实测吞吐/decode/内存、稀疏 variants | 明确揭示 FLOPs 与 wall-clock 差距 |
| artifact | JAX training、PyTorch inference、pretrained models | 官方公开，便于独立核对 |

### 符号、shape 与身份

| 符号 | 含义 | shape / 类型 | 身份 |
|---|---|---:|---|
| $L$ | Transformer layers 数 | integer | 架构超参数 |
| $X_i$ | 第 $i$ 个 multi-input block 的单一输出 | `[B,T,D]` | 追加到 history 的状态 |
| $\mathcal X_i$ | $\{X_0,\ldots,X_i\}$ | list / `[i+1,B,T,D]` | depth history |
| $X_i^Q,X_i^K,X_i^V,X_i^R$ | 为下一 block 聚合出的四个 views | 各 `[B,T,D]` | 临时运行时张量 |
| $A_i$ | 四路 dynamic depth weights | `[4,B,T,i+1]` | 运行时张量 |
| $W_{1,i},W_{2,i},a_i$ | weight generator 参数与 static prior | 见正文 | 可训练参数 |
| $D_f'(i)$ | 第 $i$ 层重分配后的 FFN width | integer | 可选架构配置 |
| $k,p$ | dilation 与 DA period | integer | 稀疏化超参数 |

---

## 1. 从 static dense 到 dynamic dense

### 1.1 Static Dense

标准 Transformer：

$$
X_0=\operatorname{Embedding}(X),
\qquad
X_i=B_i(X_{i-1}).
$$

静态 dense connection：

$$
\bar X_i
=
\sum_{j=0}^{i}a_{i,j}X_j.
$$

$a_{i,j}$ 是全局 scalar，对所有 batch、token、feature 共用。这就是论文所说的 DenseFormer-style baseline。

### 1.2 Dynamic Dense

MUDD 把每个 scalar 扩成 token-wise vector：

$$
A_{i,j}\in\mathbb R^{B\times T}.
$$

堆叠后：

$$
A_i\in\mathbb R^{B\times T\times(i+1)}.
$$

聚合：

$$
\bar X_i
=
\sum_{j=0}^{i}
A_{i,j}[:,:,None]\odot X_j.
$$

若：

$$
X_j\in\mathbb R^{B\times T\times D},
$$

则广播后每个 token 的所有 $D$ 个 features 共用一个 source-depth 权重。

### 1.3 权重生成器

论文定义：

$$
A_i(X_i)
=
\operatorname{GELU}
\left(
\operatorname{RMSNorm}(X_i)W_{1,i}
\right)
W_{2,i}
+a_i,
$$

概念性的单路 shape：

$$
W_{1,i}\in\mathbb R^{D\times(i+1)},
\qquad
W_{2,i}\in\mathbb R^{(i+1)\times(i+1)},
\qquad
a_i\in\mathbb R^{i+1}.
$$

正式伪代码为提高效率把四路合并，隐藏维设为：

$$
K_i=4(i+1),
$$

并一次输出 $4(i+1)$ 个 weights，再 reshape 成 Q/K/V/R。

### 1.4 它不是真正的 query–key depth attention

论文把 dynamic dense 类比为 query-side depth attention：

- 当前 $X_i$ 作为 query；
- 历史 $\{X_j\}$ 作为 values；
- weights 只由当前 $X_i$ 的 MLP 生成。

没有：

$$
q_i^\top k_j
$$

这样的 source-specific key dot product，也没有 depth softmax。因此是 attention analogy，而非标准 attention identity。

---

## 2. Multiway：Q/K/V/R 四路分流

### 2.1 多输入 Transformer block

标准 Pre-LN block：

$$
\begin{aligned}
X_A&=\operatorname{MHA}(\operatorname{LN}(X),
\operatorname{LN}(X),\operatorname{LN}(X))+X,\\
B(X)&=\operatorname{FFN}(\operatorname{LN}(X_A))+X_A.
\end{aligned}
$$

MUDD 定义：

$$
\begin{aligned}
X_A'
&=
\operatorname{MHA}
\left(
\operatorname{LN}(X^Q),
\operatorname{LN}(X^K),
\operatorname{LN}(X^V)
\right)
+X^R,\\
B'(X^Q,X^K,X^V,X^R)
&=
\operatorname{FFN}(\operatorname{LN}(X_A'))+X_A'.
\end{aligned}
$$

四个 inputs 具有相同 shape `[B,T,D]`，但语义不同。

### 2.2 完整递归

初始化：

$$
X_0^Q=X_0^K=X_0^V=X_0^R=X_0.
$$

第 $i$ 层先算 block output：

$$
X_i
=
B_i'
(X_{i-1}^Q,X_{i-1}^K,X_{i-1}^V,X_{i-1}^R).
$$

追加到 history 后，为下一层生成：

$$
\begin{aligned}
X_i^Q&=\operatorname{DA}_i^Q(X_0,\ldots,X_i),\\
X_i^K&=\operatorname{DA}_i^K(X_0,\ldots,X_i),\\
X_i^V&=\operatorname{DA}_i^V(X_0,\ldots,X_i),\\
X_i^R&=\operatorname{DA}_i^R(X_0,\ldots,X_i).
\end{aligned}
$$

最终输出：

$$
\operatorname{MUDDFormer}(X)=X_L^R.
$$

### 2.3 存储身份的精确化

四路 $X_i^Q,X_i^K,X_i^V,X_i^R$ 是当前 depth 为下一 block 构造的 views。真正追加到长期 history 的是：

$$
X_i.
$$

所以不能把 MUDDFormer 画成 HC 那样的固定四条 persistent residual streams。

---

## 3. 初始化、可选归一化与参数重分配

### 3.1 Function-preserving initialization

论文初始化：

$$
W_{1,i}\sim\mathcal N(0,1/D),
\qquad
W_{2,i}=0,
$$

并令 static prior：

$$
a_{i,j}=
\begin{cases}
1,&j=i,\\
0,&j\ne i.
\end{cases}
$$

于是动态项初始为 0，四路都等于当前 $X_i$：

$$
X_i^Q=X_i^K=X_i^V=X_i^R=X_i.
$$

multi-input block 因而精确恢复标准 Transformer。论文称这一初始化对良好性能是 critical。

### 3.2 PrePostDANorm

对 depth/width ratio 较大的 DeepNarrow 与 ViT，作者使用：

$$
\widetilde{\mathcal X}_i
=
\{\operatorname{Norm}(X_0),\ldots,\operatorname{Norm}(X_i)\},
$$

$$
\bar X_i
=
\operatorname{Norm}(\operatorname{DA}_i(\widetilde{\mathcal X}_i))
+X_i.
$$

Pre-DA RMSNorm scale 初始化 1，Post-DA scale 初始化 $10^{-3}$，static $a_i$ 初始化 0。

这是另一种 special case：小尺度 DA branch 加在 identity $X_i$ 上，而不是直接 one-hot select 当前状态。

### 3.3 Parameter re-allocation

作者假设上层能访问更丰富 history，因此把 FFN width 从浅到深线性增加：

$$
D_f'(i)
=
\frac{0.5(L-i)+1.5(i-1)}{L-1}D_f,
$$

即约从 $0.5D_f$ 到 $1.5D_f$，总参数保持不变。

**[综合判断]** 这是独立架构改动，不是 MUDD 定义不可分割的一部分。Table 5 显示：

- MUDD + reallocation：10.83 → 10.77；
- Transformer++ 单独 reallocation：11.68 → 11.93。

因此 reallocation 似乎与 dense history 有交互，但也意味着主模型不是只改变 connections。

---

## 4. 复杂度

论文定义：

$$
\eta=\frac{L+3}{D},
\qquad
\rho=\frac{T}{D}.
$$

额外参数比例约：

$$
R_{\Delta\mathrm{params}}
=
\frac{\eta}{6},
$$

额外 FLOPs 比例约：

$$
R_{\Delta\mathrm{FLOPs}}
=
\frac{\eta}{3+\rho/4}.
$$

典型值：

| model | $L$ | $D$ | $T$ | extra params | extra FLOPs |
|---|---:|---:|---:|---:|---:|
| 1.4B | 24 | 2048 | 4096 | 0.22% | 0.38% |
| 1.34B DeepNarrow | 42 | 1536 | 4096 | 0.49% | 0.80% |
| 2.8B | 32 | 2560 | 4096 | 0.23% | 0.40% |
| 6.9B | 32 | 4096 | 4096 | 0.14% | 0.26% |

这些是 arithmetic estimates。真实系统还受到：

- history reads；
- small MLP / elementwise kernels；
- layout transforms；
- compiler fusion；
- activation lifetime。

因此论文的实测吞吐 overhead 远大于理论 FLOPs ratio。

---

## 5. 实验审计

### 5.1 scaling-law 小到中型实验

| params | layers | $D$ | tokens |
|---:|---:|---:|---:|
| 405M | 24 | 1024 | 7B |
| 834M | 24 | 1536 | 15B |
| 1.4B | 24 | 2048 | 26B |
| 797M DeepNarrow | 34 | 1280 | 15B |
| 1.34B DeepNarrow | 42 | 1536 | 26B |

共同设置包括 Pile、context 2048、global batch 0.5M tokens。

Figure 3 比较：

- original Transformer；
- Transformer++；
- DenseFormer；
- Dynamic Hyper-Connections；
- DDFormer（dynamic single-way）；
- MUDDFormer。

结果顺序支持：

$$
\text{static}
\rightarrow
\text{dynamic}
\rightarrow
\text{multiway dynamic}
$$

每一步都贡献质量。

作者拟合认为 834M MUDDFormer 对应 Transformer++ 约 $1.89\times$ compute。由于拟合点少、没有报告参数不确定性，宜称为 paper estimate。

### 5.2 depth scaling

Transformer++ 从 24 层继续加深收益趋平，而 MUDD DeepNarrow 到 42 层仍有改善。

这支持“MUDD 帮助利用深度”，但不能唯一证明 residual bottleneck 是因果机制；parameter reallocation 与 PrePostDANorm 也参与 DeepNarrow 设置。

### 5.3 300B-token Pythia 对照

| 模型 | Pile PPL ↓ | 0-shot avg ↑ | 5-shot avg ↑ |
|---|---:|---:|---:|
| Pythia-1.4B | 7.29 | 50.8 | 51.2 |
| MUDDPythia-1.4B | 6.92 | 51.7 | 52.5 |
| Pythia-2.8B | 6.63 | 53.1 | 54.1 |
| MUDDPythia-2.8B | 6.29 | 55.0 | 57.0 |
| Pythia-6.9B | 6.29 | 55.1 | 56.4 |
| Pythia-12B | 6.01 | 56.5 | 57.2 |
| MUDDFormer-2.8B, Transformer++ recipe | 6.01 | 56.9 | 58.4 |

MUDDPythia 尽量复用 Pythia architecture 与 training hyperparameters，因而 1.4B / 2.8B 同规模比较很有价值。

跨规模“2.8B matches 6.9B”则同时改变：

- 参数；
- 每 step FLOPs；
- 并行效率；
- memory；
- MUDD overhead。

所以 $2.46\times$ 是基于模型计算规模的比较，不是完整 TCO 或 wall-clock ratio。

### 5.4 component ablation

405M：

| 配置 | PPL ↓ |
|---|---:|
| Transformer++ | 11.68 |
| + Static Dense | 11.44 |
| + Dynamic Dense | 11.09 |
| + Multiway Static Dense | 11.27 |
| + Multiway Dynamic Dense | 10.83 |
| + Multiway Dynamic Dense + Re-allocation | 10.77 |
| Transformer++ + Re-allocation | 11.93 |

从 full MUDD 去掉某一路：

| ablation | PPL ↓ |
|---|---:|
| full MUDD | 10.77 |
| -Q dense | 10.89 |
| -K dense | 10.90 |
| -V dense | 11.05 |
| -R dense | 11.14 |

论文称 V benefits most，是根据“改回普通 residual 后恶化”分析；数值上去掉 R 的 PPL 更差。更精确的说法是：

- Q/K removal 伤害较小；
- V/R 都很重要；
- 作者结合 circuit / head activation 分析特别强调 V-stream。

### 5.5 实测效率

| size | training tokens/s，相对 baseline | decode tokens/s，相对 baseline |
|---:|---:|---:|
| 1.3B | 89.8% | 88.1% |
| 2.8B | 84.0% | 90.0% |
| 6.9B | 95.6% | 94.0% |

训练：TPU v5p-128，context 2048，batch 2M tokens。<br>
推理：A100 80GB，prompt 4096，batch 1，生成 128 tokens，3 次平均。

这说明实测 overhead 约 4%–16%（训练）和 6%–12%（推理），而不是理论 FLOPs 的 0.26%–0.8%。

### 5.6 内存

论文在 gradient checkpointing 假设下估计 MUDD 额外 activation ratio：

$$
\frac{L+3}{L+17+3NT/D}.
$$

实测总内存增量约：

- 405M：29%；
- 834M：21%；
- 1.4B：28%；
- 2.8B：22%–25%；
- 6.9B：17%。

这是对系统评审很关键的事实。MUDD 是 parameter-light、arithmetic-light，但不是 activation-light。

### 5.7 稀疏变体

- `k×p`：dilation $k$、period $p$；
- `SWn`：只看最近 $n$ 层和 embedding。

1.3B efficiency 测试中，从 full `1×1` 换 `2×2`：

- training speed：89.8% → 97.8%；
- inference speed：88.1% → 93.4%；
- 405M PPL 只差 0.18。

`SW8` 的 speed/PPL trade-off 更差，论文据此强调超过 8 层的长程连接。

### 5.8 ViT

| 模型 | params | epoch 300 top-1 |
|---|---:|---:|
| ViT-S/16 | 22M | 76.0 |
| MUDDViT-S/16 | 约 22.15M | 78.1 |
| ViT-M/16 | 39M | 77.9 |

这提供跨 domain 证据，但仍是单一视觉任务。

---

## 6. 机制证据怎样解读

### 6.1 相邻层 cosine similarity

MUDDPythia 的四路 inputs 与前层表示的 cosine similarity 低于 Pythia，说明它确实构造了不同的跨层 mixture。

这支持“representation 不再只是近邻传递”，但低 similarity 本身不等于更好或更有信息。

### 6.2 attention-head activation

作者把“最大 attention weight 不落在前两个位置或若干 sink tokens”定义为 active，观察到 MUDDPythia activation ratio 约高 $2.4\times$。

这是一个 operational metric，不是社区统一定义。它支持 sink 减少的关联，但不能单独证明性能因果。

### 6.3 dynamic weight patterns

Q/K/V/R 的 weight pattern 不同，V 常对第一层有显著、动态的长程连接。

这与 multiway 设计动机一致；要证明某条路径必要，仍需更细 causal patching / knockout。

---

## 7. Claim–evidence 对照

| 主张 | 证据 | 强度 | 缺口 |
|---|---|---|---|
| dynamic 优于 static | 405M ablation、scaling curves | 较强 | 未见多 seed |
| multiway 优于 single-way | DDFormer/MUDD 与 component ablation | 较强 | 与 generator capacity 可能耦合 |
| 更好利用 depth | DeepNarrow curves | 中等 | normalization/reallocation confound |
| 约 1.8×–2.4× compute efficiency | scaling fit、Pythia size match | 中等 | 不是统一 wall-clock/TCO |
| overhead negligible | 理论 params/FLOPs | 仅 arithmetic 上成立 | 实测速度和内存不 negligible |
| 改善 in-context learning | 5-shot、FLAN PPL、head analysis | 中等 | task avg 掩盖逐任务差异 |
| generalize 到 vision | ImageNet ViT | 中等 | 单数据集 |

---

## 8. 分布式与实现含义

### 8.1 Tensor ownership

history：

$$
H_i\in\mathbb R^{(i+1)\times B\times T\times D}.
$$

dynamic weights：

$$
A_i\in\mathbb R^{4\times B\times T\times(i+1)}.
$$

DA 可写为：

$$
X_i^{c}[b,t,d]
=
\sum_{j=0}^{i}
A_i[c,b,t,j]H_i[j,b,t,d].
$$

权重不依赖 $d$，所以 hidden-sharded TP rank 可以用相同 $A_i$ 本地聚合自己的 hidden shard。权重生成器的 $D\to4(i+1)$ projection 则需要设计为 column/row parallel，并处理 reduction。

### 8.2 Sequence / context parallel

每个 token 独立生成 depth weights；若 token ownership 保持一致，DA 可本地执行。Attention 本身的 CP collective 不因 DA 消失，Q/K/V 由不同 history mixtures 构造后仍需满足 attention backend layout。

### 8.3 Pipeline parallel

full history 跨 PP stage 是主要成本：

- 传递所有 $X_j$ 会扩大 payload；
- 复制 history 增加显存；
- stage-local history 会改变算法；
- backward 需决定历史保存还是重算；
- 四路 Q/K/V/R 可重算，论文内存分析就是采用不长期保存前三路的思路。

### 8.4 KV cache

MUDD 不直接扩张最终 K/V cache 的每层 tensor shape，但每层 K/V 是由跨层 history 动态构造的。decode 时仍需当前 token 的 depth history，且生成 K/V 前多出 DA。

论文说推理 activation memory 仍由 KV cache 主导，不应误读为 DA 无额外 latency；表 4 已显示 decode 慢 6%–12%。

---

## 9. CPU reference tests

最小伪流程：

```python
history = [embedding(tokens)]
q = k = v = r = history[0]

for block, da in zip(blocks, aggregators):
    attn_state = block.attn(norm(q), norm(k), norm(v)) + r
    x = block.ffn(norm(attn_state)) + attn_state
    history.append(x)
    q, k, v, r = da(history)

return r
```

测试清单：

1. history 与 weights 的 depth 顺序；
2. `[B,T,D]` 广播；
3. 一次 projection 后 reshape 为 `[4,B,T,L]`；
4. $W_2=0$ + one-hot prior 恢复 baseline；
5. Q/K/V/R 参数和输出可不同；
6. final output 使用 $R$；
7. parameter reallocation 总参数近似守恒；
8. sparse `k×p` 与 `SWn` index；
9. `gradcheck`；
10. reference loop 与 einsum 前后向对齐；
11. checkpoint 下只重算 Q/K/V/R，不破坏 history；
12. Gloo 两 rank 验证 hidden-sharded DA。

---

## 10. 主要局限

- 大规模主结果未报告 seeds / error bars；
- 300B-token 数据都是 Pile，现代数据 mixture 的外推有限；
- 机制解释多为相关性分析；
- parameter reallocation、PrePostDANorm 与核心 connection 在部分实验耦合；
- 实测内存增加 17%–29%，不能称系统上“negligible”；
- 缺少 custom kernel，论文承认吞吐可进一步优化；
- 未展示真实多轴 TP/PP/CP 布局；
- 2.8B 是主结果上限，未验证更大 frontier scale；
- 对 alignment、instruction tuning、long-context 没有主实验；
- 没有结构性约束保证深层 depth mixing 的 norm 或 conditioning。

---

## 11. 与 HC/mHC 的区别

MUDD：

$$
\text{growing depth history}
\rightarrow
\text{four temporary mixtures}.
$$

HC/mHC：

$$
\text{fixed }N\text{ persistent streams}
\rightarrow
\text{read/mix/write}.
$$

资源形态不同：

| 维度 | MUDD | HC/mHC |
|---|---|---|
| state count | 随 depth 增长 | 固定 $N$ |
| 主 block input | Q/K/V/R 四 views | 通常从 $N$ streams 读出一份 |
| residual update | history append + 下一层聚合 | $N\times N$ stream update |
| 稳定约束 | 可选 norm，无 stochastic constraint | mHC 用 doubly stochastic mapping |
| PP 难点 | 任意历史访问 | 扩大固定 payload |

它们都改善跨层通信，但不是同一状态机。

---

## 12. 最终评价

### 可信度

- **论文身份**：高；ICML 2025 正式发表。
- **作者/机构**：北邮与彩云科技，清楚可核。
- **artifact**：四篇中最完整之一；JAX、PyTorch、checkpoints 均公开。
- **规模证据**：较强；到 2.8B / 300B tokens。
- **统计证据**：中等；主结果缺多 seed。
- **系统证据**：较强；同时报告理论成本、吞吐、decode 和内存。
- **机制因果性**：中等偏弱。

### 一句话定位

> MUDDFormer 是 DenseFormer → dynamic dense → multiway dynamic dense 这条路线中证据最完整的版本；它显著提高了跨层表达力，但用真实的 history memory 与带宽换来了极小的参数/FLOPs 增量。
