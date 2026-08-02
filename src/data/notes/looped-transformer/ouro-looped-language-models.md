---
title: "Ouro：7.7T-token Looped LM 与 Adaptive Latent Reasoning"
description: "分析完整 model-loop、7.7T-token staged training 与 adaptive exit，并审视训练深度之外的性能退化。"
topic: "looped-transformer"
section: "llm-pretraining"
slug: "ouro-looped-language-models"
date: 2026-08-02
updated: 2026-08-02
cutoff: 2026-08-02
order: 51
source:
  repository: "J-shang/looped-transformer"
  path: "papers/15-ouro-looped-language-models.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-02"
  contentHash: "sha256:7962012b160f4ecda74ba66582b992404502f69e6bd6a9a15a1fb4f0c7ffbc55"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
> 论文：*Scaling Latent Reasoning via Looped Language Models*<br>
> 精确版本：[arXiv:2510.25741v5](https://arxiv.org/abs/2510.25741v5)，2026-07-01<br>
> 作者与机构：ByteDance Seed 联合 UC Santa Cruz、Princeton、Mila / Université de Montréal、Peking University、CMU、UPenn、University of Manchester、M-A-P 等；作者包括 Yoshua Bengio<br>
> 官方 artifacts：[project](https://ouro-llm.github.io/) · [Ouro-2.6B model](https://huggingface.co/ByteDance/Ouro-2.6B)<br>
> 证据标签：**B**。base/Thinking weights 已发布；核对时 project page 的 code 仍标注 “Coming Soon”，完整 data recipe 不能像 Huginn/OLMo 一样逐条审计。

## 1. 30 秒结论

Ouro 把 Looped LM 从“中等规模可行性”推向 multi-trillion-token 训练：1.4B 与 2.6B 模型执行完整 $N$-layer stack 多次，在每个 recurrent step 都能预测 token 与退出概率，并通过 7.7T-token staged training 获得较强的 base 与 math reasoning 能力。

它最重要的正面证据是：共享参数的较小模型在 MMLU、BBH、GSM8K、MATH 等任务上达到有竞争力的结果，并能学习平均约 2–3 次的 adaptive exit。最重要的负面证据是：训练最大 depth 为 4 时，继续推到 6–8 往往退化。Ouro 展示的是**训练范围内的 latent compute allocation**，不是无界 depth extrapolation。

## 2. 学习目标

1. 区分 model-loop 与 Huginn 的 recurrent-core loop；
2. 推导 exit probability、survival probability 与 expected loss；
3. 理解 7.7T tokens 被哪些 pretraining/mid-training stages 消耗；
4. 区分 base 模型、Thinking SFT 与 adaptive gate 的贡献；
5. 用完整 depth curve 审核“test-time scaling”说法。

## 3. 符号表

| 符号 | 含义 |
|---|---|
| $F_\theta$ | 共享的完整 $N$-layer Transformer stack |
| $h^{(t)}$ | 第 $t$ 次完整 model loop 后的 hidden state |
| $T_{\max}$ | 最大 recurrent steps，主要训练设为 4 |
| $\lambda_t$ | 在 step $t$ 退出的条件概率 |
| $S_t$ | 经过前 $t$ 步仍未退出的 survival probability |
| $p(t)$ | 最终在第 $t$ 步退出的概率 |
| $\mathcal L_t$ | 从第 $t$ 步 state 计算的 next-token loss |

循环为：

$$
h^{(t)}=F_\theta\left(h^{(t-1)},x\right),\qquad t=1,\ldots,T_{\max}.
$$

每一步都接 LM head 与 exit gate，而不是只在最后一步读出。

![Ouro 的训练与 adaptive inference](/assets/looped-transformer/15-ouro-looped-language-models/figure-3-architecture.png)

*原论文 Figure 3，PDF p.4，[arXiv v5 PDF](https://arxiv.org/pdf/2510.25741v5)。左图表示训练时每一步都有 LM/exit 信号；右图表示 inference 可提前退出。看图重点：loop 单位是完整 $N$-layer stack，因此一次额外 recurrence 的成本远高于单层重复。*

## 4. Adaptive exit 的概率模型

gate 从 state 产生退出概率：

$$
\lambda_t=\sigma\left(W h^{(t)}+b\right).
$$

前 $t$ 步都不退出的概率为：

$$
S_t=\prod_{j=1}^{t}(1-\lambda_j).
$$

因此在第 $t<T_{\max}$ 步退出的概率是：

$$
p(t)=\lambda_t S_{t-1},
$$

最后一步吸收全部剩余概率，保证 $\sum_t p(t)=1$。训练可最小化各 step loss 的期望，并用 entropy regularization 防止 gate 过早塌缩：

$$
\mathcal L_{\text{adaptive}}
=\sum_{t=1}^{T_{\max}}p(t)\mathcal L_t-\beta H(p).
$$

论文把它解释为 uniform prior 下的 ELBO 形式。这个解释说明目标的概率结构，但不自动保证更深一步的 expected loss 单调下降。

### 一个最小例子

设前三步 gate 输出 $\lambda=(0.2,0.5,1)$：

$$
p(1)=0.2,\quad p(2)=0.5(1-0.2)=0.4,\quad p(3)=0.4.
$$

平均执行步数为 $0.2+0.8+1.2=2.2$。若第二步质量提升不大，gate 应把更多质量移到 $p(2)$；若困难样本第三步仍大幅改善，理想 gate 应保留较高 survival probability。

## 5. 两阶段 gate training

论文区分联合训练 gate 与后续专门优化 gate。后者使用实际 performance improvement 构造 continuation/exit target，试图同时减少：

- **underthinking**：过早退出，错过有效 refinement；
- **overthinking**：继续执行，但输出已不再改善甚至退化。

专门 gate 在论文的 quality–average-depth Pareto curve 上优于仅靠标准 gate 的版本。注意这仍是模型内 proxy；要证明 production latency 收益，还要在 dynamic/continuous batching 中测真实调度。

## 6. 7.7T-token training flow

![Ouro 的端到端训练阶段](/assets/looped-transformer/15-ouro-looped-language-models/figure-4-training-pipeline.png)

*原论文 Figure 4，PDF p.8，[arXiv v5 PDF](https://arxiv.org/pdf/2510.25741v5)。看图重点：1.4B 与 upcycled 2.6B 路径共享早期训练逻辑，之后都经历 annealing、long-context、mid-training 与 reasoning SFT；最终能力不是 architecture 单变量的产物。*

| 阶段 | tokens | recurrent steps | 主要作用 |
|---|---:|---:|---|
| Pre-train I | 3T | 8 | broad web 为主，建立可循环的 base state dynamics |
| Pre-train II | 3T | 4 | stable training；2.6B 路径发生 upcycling |
| CT annealing | 1.4T | 4 | 提高 math/code 与高质量数据占比，cosine decay |
| LongCT | 20B | 4 | context 扩到 64K，调整 RoPE base |
| Mid-training | 300B | 4 | QA/CoT、math/code 与 replay mixture |
| Reasoning SFT | 单独阶段 | 4 | 得到 Ouro-1.4B/2.6B-Thinking |

合计约 $3+3+1.4+0.02+0.3=7.72$T tokens，论文概括为 7.7T。这个 accounting 很关键：与只训练几百 B tokens 的小模型横比时，不能把全部差异归因于 loop。

模型使用 MHA、SwiGLU、RoPE、sandwich RMSNorm。1.4B 为 24 层、hidden size 2048；2.6B 为 48 层、hidden size 2048。模型执行完整 stack 多次，因此 stored depth 与 effective depth 的差异比 Huginn 更直接。

## 7. 数据 recipe 的能力归因

早期数据包含 Nemotron-CC、M-A-P web corpus、code 与 math。CT annealing 提升高质量和 math/code 权重；LongCT 使用长文数据；mid-training 约包含 90B QA/CoT、30B 第一阶段 replay 与 180B 第二阶段 replay。

这说明 Ouro 的 reasoning gain 至少有四个可能来源：

1. shared full-stack recurrence；
2. multi-trillion-token broad pretraining；
3. high-quality/math/code annealing；
4. reasoning SFT。

只有 stage-by-stage checkpoint 和 matched vanilla control 才能把它们拆开。

## 8. 主要实验结果

### 8.1 Base capability

| 模型 / depth | MMLU | BBH | GSM8K | MATH-500 |
|---|---:|---:|---:|---:|
| Ouro-1.4B, $T=4$ | 67.35 | 71.02 | 78.92 | 82.40 |
| Ouro-2.6B, $T=4$ | 74.60 | 80.46 | — | 90.85 |

2.6B 的 MMLU-Pro 为 55.73。对 stored parameter count 而言结果很强，但比较表中的其他模型通常有不同 tokenizer、data、tokens 与 post-training，因此属于定位证据而非 architecture causal proof。

### 8.2 Thinking model

论文对 1.4B Thinking 模型报告 AIME 2024 pass@1/pass@k 约 65.0/83.3，AIME 2025 为 46.3/73.3；2.6B 为 64.7/90.0 与 50.3/76.7。它说明较小 recurrent model 在特定 reasoning recipe 下可以匹配更大模型的部分数学结果。

不能忽略 sampling budget：pass@k 与 pass@1 回答的是不同问题，且 explicit output tokens、latent loops 和 candidate selection 都是 inference compute。

### 8.3 深度外推的负面证据

训练的主要最大 depth 是 4。1.4B base 在 MMLU 上约从 $T=4$ 的 67.45 降到 $T=8$ 的 64.49；2.6B Thinking 在 AIME 2024 上约从 $T=3$ 的 70.33 降到 $T=8$ 的 39.00。

因此最稳妥的结论是：

> Ouro 在训练覆盖的有限 depth 区间内学到了可分配 latent compute，但没有学到“多做一步必然更好”的稳定迭代算子。

这条负面结果比最佳 benchmark 更重要，因为它决定 adaptive exit 是可选优化还是必要组件。

## 9. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| scale | Looped LM 扩展到 7.7T-token staged training | 强：paper + released weights |
| architecture | 完整 model stack recurrence，每步可读出 | 强 |
| adaptive compute | 概率 exit 与专门 gate training | 中强：有 Pareto 曲线，缺 production scheduler 证据 |
| capability | 1.4B/2.6B base 与 Thinking 结果有竞争力 | 中：外部比较 data/post-training 不完全 matched |
| interpretability | 中间 step 可反映逐步 refinement/faithfulness proxy | 弱到中：观察关联，不是 causal tracing |
| openness | weights 与 project 信息 | 中：code/data 尚不完整 |

## 10. Claim–evidence map

| 可说的 claim | 直接证据 | 不可外推 |
|---|---|---|
| Looped LM 能完成 multi-trillion-token training | 7.7T recipe、checkpoints | 训练成本低于相同质量 vanilla |
| adaptive exit 能改善 quality–average-depth tradeoff | gate comparison | 部署 wall-clock 必然同比下降 |
| 小 stored model 可达到强 base/reasoning 分数 | tables | recurrence 是唯一原因 |
| $T=3$–$5$ 常有有效 refinement | depth sweeps | $T>4$ 仍单调改善 |
| latent states 展现某些 reasoning 关联 | probe/trajectory observations | latent state 等价于可忠实解释的 CoT |

## 11. 关键边界

1. **code/data 开放度**：weights 可用，但训练 code 与完整 data mixture 还不足以逐项复现。
2. **计算匹配**：完整 stack 重复使单次 loop 成本高；parameter-matched 不是 FLOP-matched。
3. **depth 稳定性**：超过训练 depth 明显退化，不能宣传成 unbounded test-time scaling。
4. **stage 混杂**：7.7T data curriculum 与 reasoning SFT 都很强，architecture 归因需要 stage controls。
5. **通用 post-training**：主要展示 base 与 reasoning，multi-turn chat、tool use、safety、preference retention 仍不充分。
6. **“单调改进”**：任何 $\mathbb E[\mathcal L_{t+1}]\le\mathbb E[\mathcal L_t]$ 表述都应理解为训练目标或经验趋势，不是架构定理。

## 12. 与其他论文的连接

- 与 **Huginn**：Huginn 循环内部 core，Ouro 循环完整 stack；前者 artifacts 更开放，后者 data scale 更大。
- 与 **MoR**：Ouro 主要做 sequence-level exit，MoR 尝试 token-level routing。
- 与 **LoopRPT**：LoopRPT 直接从 Ouro checkpoint 出发，把 reward 放到 latent recurrence 与 exit 上。
- 与 **LOTUS**：Ouro 用每步 next-token/gate 目标；LOTUS 用 gold CoT 对 latent blocks 做直接 supervision。
- 与 **Loopie**：两者都接近现代通用 LLM，但 Loopie 更强调 MoE 和 measured wall-clock comparison。

## 13. 复现路线

### 只用公开 checkpoint

1. 固定 Ouro-1.4B/2.6B checkpoint；
2. 扫描 $T=1\ldots 8$，分别测 base、knowledge、math/code、perplexity；
3. 对 adaptive exit 报平均/分位数 depth，而不只报 mean；
4. 比较 static batching 与按 depth 分组的 continuous batching；
5. 记录超过 $T=4$ 后退化集中在哪类 token 与任务。

### 最小 matched control

在一个 fully-open 1B–3B base recipe 中训练：

- vanilla 固定深度；
- full-stack loop，固定 $T=4$；
- full-stack loop，训练时 depth sampling；
- 同一 loop model 加 gate-only stage。

分别固定 tokens、analytical FLOPs 与 wall-clock，避免只得到一种偏向性的结论。

## 14. 可证伪的新问题

- **过度计算是否来自 state drift？** 测 $\|h^{(t+1)}-h^{(t)}\|$、logit KL 与 accuracy；若 state 收敛但 accuracy 仍崩溃，则不能只怪 drift。
- **gate 是否只学到 task/difficulty prior？** 打乱 prompt difficulty 或做 adversarially easy-looking hard questions；若 depth allocation 不变，gate 可能未读取真实 state improvement。
- **普通 DPO/RLVR 会不会破坏 exit calibration？** 保存每个 post-training stage，比较 reliability curve 与 performance–depth curve。

## 15. 自测题

1. 写出 $p(t)$ 与 $S_t$ 的关系，并计算一个三步示例。
2. 为什么 Ouro 的 7.7T tokens 是理解 benchmark 的必要背景？
3. 完整 stack loop 与 recurrent-core loop 在 latency 和参数占比上有什么不同？
4. $T=4$ 最好、$T=8$ 变差说明了哪两种可能机制？
5. 如何验证 adaptive exit 的真实 wall-clock 收益，而不只验证平均 FLOPs？

## 16. 一句话定位

Ouro 是当前“把 Looped LM 训练成有竞争力通用 base/Thinking model”的关键规模证据；它也用清晰的 depth 退化结果提醒我们，adaptive compute 的核心是**知道何时停止**，不是无条件增加循环。
