---
title: "LAuReL：用 scalar gate、低秩映射和短历史增强 residual path"
description: "统一 LAuReL-RW、LR 与 PA 三种轻量残差增强，核查参数预算、1B/4B 实验、效率收益和可复现性缺口。"
topic: "residual"
section: "methods"
slug: "laurel"
date: 2026-07-27
updated: 2026-07-27
cutoff: 2026-07-27
order: 17
readtime: 15
source:
  repository: "J-shang/residual"
  path: "papers/08-laurel.md"
  url: "https://github.com/J-shang/residual/blob/c55583707ab645bef7408c588fc33cbf91b809a5/papers/08-laurel.md"
  revision: "c55583707ab645bef7408c588fc33cbf91b809a5"
  syncedAt: "2026-07-28"
  contentHash: "sha256:f9286ad6322acdafe02a4b411dca7f258cd299102885bccefd87b8749bd469e4"
  manifest: "residual"
  managed: true
---
<!-- paper-order: 08 -->

> **论文**：Gaurav Menghani、Ravi Kumar、Sanjiv Kumar, *LAuReL: Learned Augmented Residual Layer*<br>
> **机构**：Google Research<br>
> **版本**：ICML 2025 / PMLR 267；arXiv:2411.07501v4，2025-06-24<br>
> **状态**：ICML 2025 正式发表，不是 Independent Researcher 投稿<br>
> **主来源**：[PMLR 正式页面](https://proceedings.mlr.press/v267/menghani25a.html) · [arXiv](https://arxiv.org/abs/2411.07501) · [PDF](https://arxiv.org/pdf/2411.07501)<br>
> **官方 artifact**：论文未给出公开训练代码或 checkpoint；LLM 数据、完整模型配置与训练实现为内部设置<br>
> **阅读范围**：arXiv v4 全文 11 页，包括三种变体、1B/4B LLM、效率、ablation 与 naive-scaling 对照<br>
> **信息核对日期**：2026-07-27<br>
> **前置阅读**：[路线图中的 residual / identity mapping 阶段](/topics/residual/) · [DenseFormer](/topics/residual/denseformer/)

## 证据标签

- **[论文报告]**：论文直接报告。
- **[复原推导]**：由定义直接推出。
- **[综合判断]**：结合证据形成的解释。
- **[扩展假设]**：需要新实验。
- **[待验证]**：当前不能确认。

---

## 先给结论

### 30 秒版

LAuReL 不是单一算子，而是 residual augmentation 的一个小型设计族：

$$
x_{i+1}
=
\alpha_i f_i(x_i)
+
g_i(x_i,x_{i-1},\ldots,x_0).
$$

论文重点研究三种便宜的 $g_i$：

1. **RW（Residual Weights）**

   $$
   x_{i+1}
   =
   \alpha_i f_i(x_i)+\beta_i x_i.
   $$

2. **LR（Low Rank）**

   $$
   x_{i+1}
   =
   f_i(x_i)+x_i+B_iA_ix_i.
   $$

3. **PA（Previous Activations）**

   $$
   x_{i+1}
   =
   f_i(x_i)+x_i+
   \sum_{j=0}^{k-1}\gamma_{i,j}h_i(x_{i-j}).
   $$

RW 学 branch/residual 的相对尺度；LR 在 identity path 上加低秩线性修正；PA 再引入最近 $k$ 个历史状态。

与另外三篇相比，LAuReL 的核心不是全 depth routing，而是“尽量不改模型拓扑，用少量参数增强原 residual path”。

### 最强结果

**[论文报告]**

- ResNet-50 / ImageNet-1K，5 trials：
  - baseline 74.95；
  - 加一层 75.20，参数 +4.37%；
  - RW 75.10，参数 +0.003%；
  - RW+LR 75.20，参数 +1.68%；
  - RW+LR+PA 75.25，参数 +2.40%。
- 1B LLM、约 0.5T tokens、RW+LR rank 4：
  - 参数 +0.012%，几乎无 latency 增量；
  - 9 个任务中 8 个改善，MBPP 持平。
- 4B multilingual/multimodal LLM、约 0.5T tokens、rank 64：
  - 参数约 +0.1%，prefill/generation latency 约 +1%–2%；
  - 10 个任务中 8 个改善，WMT23 与 DocVQA 基本持平。
- 157M / C4 10B-token ablation 中，所有 LAuReL variants 的 test loss 都优于 24-layer baseline，也优于更大的 28-layer baseline。

### 我的总判断

**[综合判断]** LAuReL 的思路合理、正式发表、作者机构明确，但四篇中它的公开可复现性最弱：

- 1B/4B 模型的 architecture、数据 mixture、tokenization、训练超参数并不完整公开；
- 没有 public code 或 checkpoint；
- 大模型基本是 baseline 与 LAuReL 各一次昂贵训练；
- 任务分数的统计协议只部分公开；
- “nearly no latency”与“1%–2%”来自内部 TPU/手机测量，缺 profiler 和硬件细节。

因此它值得作为低风险 residual augmentation baseline 学习，但不宜仅凭内部 1B/4B 表格认定它优于公开、可复现的 alternatives。

---

## 5 分钟方法地图

| 层面 | LAuReL |
|---|---|
| motivating pressure | identity residual 太固定；增大模型未必是最佳额外预算用法 |
| stored state | RW/LR 只需当前 $x_i$；PA 需最近 $k$ 个 activations |
| read rule | $f_i$ 仍读当前 $x_i$ |
| write rule | branch output 与 learned linear residual component 相加 |
| state update | 单条 residual stream |
| 权重粒度 | per-layer scalar；或 low-rank feature map；PA 再加 per-lag scalar |
| input-dependent | 基本变体否；参数训练后固定 |
| 初始化 | LR 的 $B=0$ 可精确恢复 identity augmentation；RW 需明确 gate 初始化 |
| deep composition | 无 stochastic / norm-preserving 保证；RW 用 sigmoid/softmax 限制 gate |
| 参数成本 | RW 1–2；LR $2rD$；PA 依 $h$ 定义 |
| 主要系统成本 | low-rank matmul；PA activation window |

### 贡献账本

| 类型 | 贡献 | 新颖性与证据 |
|---|---|---|
| conceptual | 将 identity residual 推广为 learned linear residual component | 统一 RW/LR/PA |
| methodological | per-layer scalar gate（RW） | 极低参数 residual scaling |
| methodological | identity + low-rank residual map（LR） | function-preserving 初始化可行 |
| methodological | 最近 $k$ 个 activations 的加权线性增强（PA） | 有界 history 版本 |
| empirical | ResNet-50、1B/4B LLM、小型 C4 ablation | 覆盖广，但大模型不可公开复现 |
| systems | 参数、memory、step time 与内部 serving latency | 有成本意识，硬件细节有限 |
| artifact | 正式论文公开 | 未公开训练代码、checkpoint 与完整 LLM recipe |

### 符号、shape 与身份

| 符号 | 含义 | shape / 类型 | 身份 |
|---|---|---:|---|
| $x_i$ | 第 $i$ 个 residual connection 输入 | `[B,T,D]` 或视觉对应张量 | 运行时状态 |
| $f_i$ | nonlinear branch，含必要 normalization | 与 $x_i$ 同 shape | 普通模型函数 |
| $\alpha_i,\beta_i$ | branch / residual gates | scalar | 可训练参数 |
| $A_i,B_i$ | low-rank down/up projections | `[r,D]`、`[D,r]` | 可训练参数 |
| $r$ | low-rank dimension | integer，$r\ll D$ | 架构超参数 |
| $k$ | PA 访问的最近 activation 数 | integer | 架构超参数 |
| $\gamma_{i,j}$ | 第 $j$ 个 lag 的 mixing scalar | scalar | 可训练参数 |
| $h_i$ | PA 的 identity 或 low-rank linear map | `[B,T,D]\to[B,T,D]` | 设计选择 |

---

## 1. 统一公式

标准 residual：

$$
x_{i+1}=f_i(x_i)+x_i.
$$

LAuReL：

$$
x_{i+1}
=
\alpha_i f_i(x_i)
+
g_i(x_i,x_{i-1},\ldots,x_0),
$$

其中：

- $\alpha_i$ 是 learned scalar；
- $g_i$ 是 learned linear function；
- normalization 可以折进 $f_i$ 的定义。

论文的核心假设是：

> residual path 不必永远是固定 identity；可以把一部分线性表达能力放到轻量 residual augmentation 中，让主 branch 更专注非线性变换。

这是设计动机，不是理论证明。

---

## 2. 三个变体

### 2.1 LAuReL-RW

$$
x_{i+1}
=
\alpha_i f_i(x_i)+\beta_i x_i.
$$

每层只加 2 个 scalars；也可只学习一个 logit，通过 sigmoid 构造互补权重。

论文发现 $\alpha,\beta$ 不能无界增长，推荐用 softmax 或 sigmoid normalization。

若使用 two-logit softmax：

$$
[\alpha_i,\beta_i]
=
\operatorname{softmax}([a_i,b_i]),
$$

则：

$$
\alpha_i,\beta_i>0,\qquad
\alpha_i+\beta_i=1.
$$

**[综合判断]** 这会把标准 residual 的 $1+1$ 系数变为凸组合尺度，而不是自动恢复 $f(x)+x$。实现必须查明具体 gate parameterization 与初始化，不能只把 logits 设成相等就声称 function-preserving。

论文没有像 DenseFormer/MUDD 那样给出一个统一、明确的 RW baseline-equivalent 初始化公式。

### 2.2 LAuReL-LR

先设：

$$
g_i(x_i)=W_ix_i,
\qquad
W_i=I+B_iA_i.
$$

若：

$$
A_i\in\mathbb R^{r\times D},
\qquad
B_i\in\mathbb R^{D\times r},
$$

则：

$$
x_{i+1}
=
f_i(x_i)+x_i+B_iA_ix_i.
$$

论文记号中 $A,B$ 的书写方向偶有转置差异；实现 contract 应以：

$$
[B,T,D]\to[B,T,r]\to[B,T,D]
$$

为准。

参数增量：

$$
2rD
$$

per LAuReL layer。

论文令第二个 factor $B_i=0$，第一 factor 用 Xavier（ResNet）或 column-orthogonal（LLM）初始化。于是：

$$
B_iA_ix_i=0,
$$

LR 在初始化时精确恢复标准 residual。

### 2.3 LAuReL-PA

使用最近 $k$ 个 activations：

$$
x_{i+1}
=
f_i(x_i)+x_i
+
\sum_{j=0}^{k-1}
\gamma_{i,j}h_i(x_{i-j}).
$$

$\gamma_{i,j}$ 是 learned lag weights，$h_i$ 可为 identity 或 low-rank map。

如果同一 $h_i$ 共享给所有 lags，论文给出的每层参数近似：

$$
2rD+k.
$$

若每个 lag 使用独立 $A_{i,j},B_{i,j}$，则 combined 公式会达到：

$$
2rkD+k.
$$

论文的总框架允许这两种风格，但 Table 4 对 `LR+PA` 使用后者的成本。实现前必须固定是否跨 lag 共享 low-rank factors。

### 2.4 组合变体

RW+LR：

$$
x_{i+1}
=
\alpha_i f_i(x_i)
+
\beta_i(x_i+B_iA_ix_i).
$$

RW+LR+PA：

$$
x_{i+1}
=
\alpha_i f_i(x_i)
+
\beta_i
\left(
x_i+
\sum_{j=0}^{k-1}
\gamma_{i,j}B_{i,j}A_{i,j}x_{i-j}
\right).
$$

这些都是单 residual stream 的增强，不产生 HC 式多流状态。

---

## 3. 成本模型

论文 Table 4 按单次 LAuReL instantiation 给出：

| 变体 | 参数 | 额外 memory | 论文列出的 latency 上界 |
|---|---:|---:|---:|
| RW | 1 或 2 | $\Theta(1)$ | $O(1)$ |
| LR | $2rD$ | $\Theta(2rD)$ | $O(rD^2)$ |
| PA，$h=I$ | $k$ | $\Theta(kD)$ | $O(kD)$ |
| LR+PA，独立 factor | $2rkD+k$ | $\Theta(2rkD+k)$ | $O(krD^2+kD)$ |

### 对 LR latency 公式的精确化

**[复原推导]** 对单个 $D$ 维 vector，低秩路径：

$$
x\in\mathbb R^D
\xrightarrow{A}
\mathbb R^r
\xrightarrow{B}
\mathbb R^D
$$

朴素乘加量是：

$$
O(Dr)+O(rD)=O(Dr),
$$

而不是 $O(rD^2)$。对 `[B,T,D]` 是 $O(BTDr)$。

论文 Table 4/正文给出 $O(rD^2)$ upper bound，并注明具体 matmul algorithm 可更紧。工程预算更应使用 $O(BTDr)$ 的实际矩阵 shape 计算，再以 profiler 验证。

### PA 的 activation lifetime

PA 只保留最近 $k$ 个 states：

$$
M_{\text{window}}
\approx
kBTD\,s.
$$

与 full DenseFormer/MUDD 的 depth-growing history 相比更易控制，也更适合 pipeline-local 实现。

---

## 4. 实验审计

### 4.1 ResNet-50 / ImageNet-1K

**[论文报告]**

- 16 TPU v5e；
- 文中写“over one epoch”，同时采用 data augmentation；
- baseline learning-rate schedule 经 5 trials 调优；
- 报告 5 trials 的 average best accuracy@1。

| 模型 | top-1 ↑ | 参数增量 |
|---|---:|---:|
| baseline | $74.95\pm0.01$ | 0 |
| baseline + 1 layer | $75.20\pm0.12$ | 4.37% |
| RW | $75.10\pm0.10$ | 0.003% |
| RW+LR, $r=16$ | $75.20\pm0.07$ | 1.68% |
| RW+LR+PA | $75.25\pm0.09$ | 2.40% |

结论：

- RW 用极少参数获得 0.15 point；
- RW+LR 匹配加一层，并少用 $2.6\times$ 的额外参数；
- 加 PA 再提高 0.05 point。

保留意见：

- 差值很小，best-accuracy protocol 对训练噪声敏感；
- 论文给出 $\pm$ 但未在表旁清楚说明是 standard deviation、standard error 还是其他区间；
- “one epoch”对于 ImageNet ResNet-50 非常特殊，需要完整 recipe 才能复现和解释。

### 4.2 1B LLM

**[论文报告]**

- decoder-only Transformer；
- baseline 与 LAuReL 从头预训练；
- 约 0.5T text tokens，含 webpages、books、code、translations；
- 256 TPU v5e，约两周；
- RW+LR，$r=4$；
- 参数 +0.012%，额外 latency 在噪声内。

| task | baseline | LAuReL |
|---|---:|---:|
| MATH | 3.54 | 3.70 |
| GSM8K-CoT | 8.34 | 8.79 |
| MMLU | 25.72 | 25.89 |
| BoolQ | 58.07 | 65.66 |
| TyDi QA GoldP | 67.98 | 72.58 |
| HellaSwag | 64.84 | 65.06 |
| HumanEval | 18.29 | 18.90 |
| MBPP | 27.00 | 27.00 |
| GSM8K-PAL | 10.31 | 11.37 |

最大相对提升来自低 baseline 区域或 BoolQ；不应只看相对百分比，绝对 point 同样重要。

### 4.3 4B multilingual/multimodal LLM

**[论文报告]**

- 约 0.5T multimodal + multilingual tokens；
- 1024 TPU v4，稍多于两天；
- RW+LR，$r=64$；
- 参数约 +0.1%；
- server TPU 与“leading smartphone”上 prefill/generation latency 约 +1%–2%。

| task | baseline | LAuReL |
|---|---:|---:|
| MATH | 14.70 | 15.30 |
| MGSM | 20.00 | 23.09 |
| MMLU | 49.85 | 51.12 |
| Belebele | 58.40 | 63.23 |
| BookQA | 50.36 | 60.46 |
| WMT23 | 68.32 | 68.24 |
| MMMU | 32.22 | 36.33 |
| COCO Caption | 95.69 | 99.15 |
| DocVQA | 68.28 | 68.34 |
| TextVQA | 60.07 | 62.64 |

8 个改善，WMT23 微降，DocVQA 基本持平。

**[综合判断]** 结果覆盖面不错，但模型、数据、eval prompts/decoding、seeds 与 checkpoint 均未公开，证据更像工业内部 case study。

### 4.4 rank sweep

ResNet RW+LR：

- $r=4,8$ 相对 RW 改善不明显；
- $r=16,32$ 最好；
- 更高 rank 反而下降。

作者提出两个可能解释：

1. 更大容量需要重新调 learning rate / regularization；
2. low-rank factors 初始化敏感。

论文明确报告初始化对 LR 性能影响大，这是部署/复现时的高风险点。

### 4.5 小型 C4 ablation

约 10B C4 tokens，4×4 TPU v6e：

| 模型 | params | test loss ↓ | peak GB | step s |
|---|---:|---:|---:|---:|
| 24-layer baseline | 157.20M | 3.0159 | 11.65 | 0.095 |
| 28-layer baseline | 179.23M | 2.9963 | 13.23 | 0.105 |
| RW | 157.20M | 2.9557 | 11.93 | 0.095 |
| LR | 158.40M | 2.9624 | 12.29 | 0.098 |
| PA | 157.22M | 2.9512 | 12.55 | 0.100 |
| RW+LR | 158.40M | 2.9531 | 12.57 | 0.099 |
| RW+LR+PA | 160.83M | 2.9499 | 12.90 | 0.104 |

所有 variants 都优于更深 baseline，但 full combination 的 loss 优势很小、内存和 step time 已接近 28-layer baseline。

RW 反而拥有很强 Pareto 点，这也是作者建议先试 RW 的原因。

### 4.6 4.4B naive scaling 对照

| 模型 | params | avg step |
|---|---:|---:|
| baseline-40 | 4.400B | 1.65 s |
| baseline-41 | 4.560B (+3.63%) | 1.68 s (+1.81%) |
| LAuReL-40 | 4.404B (+0.1%) | 1.69 s (+2.42%) |

LAuReL 参数少得多，但 step latency 略高于加一层。原因是 low-rank path 分布在每层，可能更难高效执行。

因此“参数更高效”不自动等于“wall-clock 更高效”。

---

## 5. 初始化与深层组合

### 5.1 LR 的 Jacobian

单层：

$$
x_{i+1}
=
f_i(x_i)+(I+B_iA_i)x_i.
$$

Jacobian：

$$
J_i
=
J_{f_i}(x_i)+I+B_iA_i.
$$

标准 residual 的显式 identity 项仍存在，但多了 learned low-rank perturbation。

初始化 $B_i=0$ 时：

$$
J_i=J_{f_i}(x_i)+I.
$$

训练后，$\|B_iA_i\|$ 没有结构性上界。稳定性来自优化、gate normalization 和模型 recipe，而不是 LR 参数化本身。

### 5.2 RW 的尺度

$$
J_i=\alpha_iJ_{f_i}(x_i)+\beta_iI.
$$

如果 sigmoid/softmax 把 $\beta_i$ 压得很小，identity gradient path 也会被缩放。限制 gate 无界增长有助数值控制，却不等于保证跨深度 Jacobian 接近 1。

这也是 LAuReL 与 mHC 的根本区别：mHC 约束 residual mixing matrix 的 manifold；LAuReL 主要靠低维参数化和 gate normalization。

---

## 6. Claim–evidence 对照

| 主张 | 证据 | 强度 | 主要缺口 |
|---|---|---|---|
| 可作为 in-situ replacement | ResNet、1B、4B、157M | 中等 | residual placement 与初始化细节未完整开源 |
| 优于 naive scaling | ImageNet、157M、4.4B 对照 | 较强 | scaling baseline 调参公平性 |
| 额外参数很少 | 解析计数与实测 | 强 | 参数不是总成本 |
| latency 很小 | 内部 TPU/手机、ablation step time | 中等 | 硬件和 profiler 不公开 |
| vision/language 都有效 | ImageNet + 多种 LLM tasks | 中等偏强 | 每个 domain 的公开复现不足 |
| 无训练不稳定 | 作者经验陈述 | 弱 | 无稳定性指标/多 run |
| PA 提供历史访问收益 | ImageNet 与小 LM | 中等 | 大 LLM 未测试 PA |

---

## 7. 系统与分布式含义

### 7.1 RW

RW 最容易部署：

- scalar replicated；
- 可与 residual add 融合；
- TP/CP/PP layout 不变；
- 但 softmax/sigmoid 的具体位置必须统一 reference 与 fused path。

### 7.2 LR

低秩路径：

$$
[B,T,D]\to[B,T,r]\to[B,T,D].
$$

TP 选择：

- $A$ 做 column-parallel、$B$ 做 row-parallel；
- 中间 rank $r$ 较小，collective overhead 可能相对 matmul 偏大；
- 也可复制 low-rank factors，但增加参数/optimizer state duplication。

小矩阵不一定能达到主干 GEMM 的硬件效率，正好解释参数少但 step time 仍可能明显增加。

### 7.3 PA

只保留 $k$ 个历史 states，适合 ring buffer：

$$
\text{buffer shape}=[k,B,T,D].
$$

PP 下若 $k$ 跨 stage boundary，需要携带 recent history；如果每个 stage 重置 window，则是另一个算法。

activation checkpointing 要决定：

- ring buffer 保存；
- 从 stage checkpoint 重算；
- low-rank projected history 是否比原 history 更值得保存。

### 7.4 FSDP / optimizer

LR factors 参数虽少，但每层都有独立小参数。FSDP flattening、all-gather granularity 和 optimizer kernel launch 可能主导成本。应将它们合并打包，而不是逐层发起小 collective。

---

## 8. CPU reference tests

### RW

1. gate normalization；
2. baseline-equivalent 初始化是否真的成立；
3. scalar broadcast；
4. branch 和 residual 的系数没有对调。

### LR

1. shape `[B,T,D] -> [B,T,r] -> [B,T,D]`；
2. $B=0$ 时逐元素等于普通 residual；
3. $r=0$ / disabled path；
4. `float64 gradcheck`；
5. factor ordering与论文记号转置；
6. fused linear 与两步 oracle 对齐。

### PA

1. history 的 lag 0 是否包含当前 $x_i$；
2. 网络开头 history 不足 $k$ 时的规则；
3. factor 是否跨 lag 共享；
4. ring-buffer wraparound；
5. PP boundary contract；
6. backward 不被 in-place buffer overwrite 破坏。

---

## 9. 主要局限

- 没有 public code、checkpoint 或完整 LLM recipe；
- LLM 训练数据是非公开 mixture；
- baseline 模型细节不足，外部无法精确复现；
- 大模型实验看起来没有 independent seeds；
- downstream 评估 protocol 与显著性计算不完整；
- RW 的具体 normalization / initialization 描述不够精确；
- LR 初始化敏感，rank 最优值不单调；
- PA 没有在昂贵的大 LLM 实验中验证；
- latency 数据缺少具体 smartphone、serving batch、compiler 和 profiler；
- 理论 cost 表对 LR 给出过松上界，容易误导；
- 没有与同年 DCA/MUDD 的强同框架对照；
- 没有深层 Jacobian、activation norm 或 conditioning 分析。

---

## 10. 与其他三篇的关系

| 方法 | 全历史 | 动态 | 多路 | 低秩 transform | 最近窗口 |
|---|---:|---:|---:|---:|---:|
| DenseFormer | 是 | 否 | 否 | 否 | 可用其他 sparsity |
| DCA | 可选 | 是 | Q/K/V | 否 | first+last-$k$ |
| MUDDFormer | 是 | 是 | Q/K/V/R | 否 | SW / dilation-period |
| LAuReL | PA 仅最近 $k$ | 否 | 否 | 是 | 是 |

LAuReL 解决的是不同问题：

$$
\text{如何低成本增强单条 residual update}
$$

而 DCA/MUDD 更关注：

$$
\text{当前层应该从哪些历史深度读什么}.
$$

---

## 11. 推荐实践顺序

结合论文证据与工程风险：

1. **RW**：最小代码和状态变化，先验证是否有收益；
2. **LR**：确认 $B=0$ function-preserving 初始化与 rank sweep；
3. **RW+LR**：如果两者分别有效，再组合；
4. **PA**：只有在 activation / PP 预算允许时加入；
5. **RW+LR+PA**：最后比较真实 quality–memory–latency Pareto，而不是只看参数。

这个顺序与论文作者建议基本一致。

---

## 12. 最终评价

### 可信度

- **论文身份**：高；ICML 2025 正式发表。
- **作者/机构**：Google Research，清楚可核。
- **方法合理性**：高；RW/LR/PA 都容易形式化和做 CPU oracle。
- **公开复现性**：四篇中偏弱；缺代码、模型与数据。
- **实验规模**：较强；1B/4B、约 0.5T tokens。
- **统计与归因**：中等偏弱；昂贵主实验缺多 run 和完整 protocol。
- **系统结论**：有内部实测，但外部可验证性有限。

### 一句话定位

> LAuReL 是一组务实的 residual augmentation：RW 最轻，LR 增加低秩线性能力，PA 加短历史；正式发表和大规模内部实验让它值得认真对待，但公开证据还不足以把它视为已被独立验证的默认方案。
