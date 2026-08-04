---
title: "Ouro：7.7T-token Looped LM 与 Adaptive Latent Reasoning"
description: "分析完整 model-loop、7.7T-token staged training 与 adaptive exit，并审视训练深度之外的性能退化。"
topic: "looped-transformer"
section: "llm-pretraining"
slug: "ouro-looped-language-models"
date: 2026-08-02
updated: 2026-08-04
cutoff: 2026-08-02
order: 51
source:
  repository: "J-shang/looped-transformer"
  path: "papers/15-ouro-looped-language-models.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-04"
  contentHash: "sha256:06fe708150d7cc32dbe47a73b610354b7e9b1b0baa8e00d90a57c644e0e36624"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
> 论文：*Scaling Latent Reasoning via Looped Language Models*<br>
> 精确版本：[arXiv:2510.25741v5](https://arxiv.org/abs/2510.25741v5)，2026-07-01<br>
> 作者与机构：ByteDance Seed 联合 UC Santa Cruz、Princeton、Mila / Université de Montréal、Peking University、CMU、UPenn、University of Manchester、M-A-P 等；作者包括 Yoshua Bengio<br>
> 官方 artifacts：[project](https://ouro-llm.github.io/) · [Ouro-1.4B model](https://huggingface.co/ByteDance/Ouro-1.4B) · [Ouro-2.6B model](https://huggingface.co/ByteDance/Ouro-2.6B) · [1.4B inference implementation](https://huggingface.co/ByteDance/Ouro-1.4B/blob/main/modeling_ouro.py)<br>
> 证据标签：**B**。base/Thinking weights 与 Hugging Face inference implementation 已发布；完整 training code 和 data recipe 仍不能像 Huginn/OLMo 一样逐条审计。本文对公开推理代码的核对截止 **2026-08-03**。

## 1. 30 秒结论

Ouro 把 Looped LM 从“中等规模可行性”推向 multi-trillion-token 训练：1.4B 与 2.6B 模型执行完整 $N$-layer stack 多次，在每个 recurrent step 都能预测 token 与退出概率，并通过 7.7T-token staged training 获得较强的 base 与 math reasoning 能力。

它最重要的正面证据是：共享参数的较小模型在 MMLU、BBH、GSM8K、MATH 等任务上达到有竞争力的结果，并能学习平均约 2–3 次的 adaptive exit。最重要的负面证据是：最终 checkpoint 的后续训练主要使用 depth 4，推理继续加到 6–8 往往退化。Ouro 展示的是**训练稳定区间附近的 latent compute allocation**，不是无界 depth extrapolation。

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
| $h_i^{(t)}$ | 第 $t$ 次 loop 后、第 $i$ 个 token 位置的 hidden state，shape 为 $d=2048$ |
| $T_{\max}$ | 最大 recurrent steps，主要训练设为 4 |
| $\lambda_{i,t}$ | token 位置 $i$ 在 step $t$ 退出的条件概率；gate 的 runtime 输出 |
| $S_{i,t}$ | token 位置 $i$ 经过前 $t$ 步仍未退出的 survival probability |
| $p_i(t)$ | token 位置 $i$ 最终在第 $t$ 步退出的概率 |
| $\mathcal L_i^{(t)}$ | 从第 $t$ 步 state 计算的 token-level next-token loss |
| $q$ | Q-exit 的全局或任务校准阈值；越小越容易早退 |

循环为：

$$
h^{(t)}=F_\theta\left(h^{(t-1)},x\right),\qquad t=1,\ldots,T_{\max}.
$$

每一步都接 LM head 与 exit gate，而不是只在最后一步读出。

![Ouro 的训练与 adaptive inference](/assets/looped-transformer/15-ouro-looped-language-models/figure-3-architecture.png)

*原论文 Figure 3，PDF p.4，[arXiv v5 PDF](https://arxiv.org/pdf/2510.25741v5)。左图表示训练时每一步都有 LM/exit 信号；右图表示 inference 可提前退出。看图重点：loop 单位是完整 $N$-layer stack，因此一次额外 recurrence 的成本远高于单层重复。*

## 4. Adaptive exit 的概率模型

gate 是一个共享的线性层，对每个 token 位置的 state 分别产生退出概率：

$$
\lambda_{i,t}=\sigma\left(W h_i^{(t)}+b\right).
$$

该 token 前 $t$ 步都不退出的概率为：

$$
S_{i,t}=\prod_{j=1}^{t}(1-\lambda_{i,j}).
$$

因此在第 $t<T_{\max}$ 步退出的概率是：

$$
p_i(t)=\lambda_{i,t} S_{i,t-1},
$$

最后一步吸收全部剩余概率，保证 $\sum_t p_i(t)=1$。训练可最小化各 step loss 的期望，并用 entropy regularization 防止 gate 塌缩到总是使用最深一步：

$$
\mathcal L_{\text{adaptive}}
=\sum_i\sum_{t=1}^{T_{\max}}p_i(t)\mathcal L_i^{(t)}-\beta\sum_i H(p_i).
$$

这里把论文对 sequence 的期望/求和显式展开成 token 位置 $i$，是为了说明粒度；论文 Equation 2–4 在部分位置省略了 $i$，但 Section 3.4 的 gate target 明确是 per-token。论文把该目标解释为 uniform prior 下的 ELBO 形式。这个解释说明目标的概率结构，但不自动保证更深一步的 expected loss 单调下降。

### 一个最小例子

设前三步 gate 输出 $\lambda=(0.2,0.5,1)$：

$$
p(1)=0.2,\quad p(2)=0.5(1-0.2)=0.4,\quad p(3)=0.4.
$$

平均执行步数为 $0.2+0.8+1.2=2.2$。若第二步质量提升不大，gate 应把更多质量移到 $p_i(2)$；若困难 token 在第三步仍大幅改善，理想 gate 应保留较高 survival probability。

推理不直接采样 $p_i(t)$，而使用 Q-exit：

$$
t_{\mathrm{exit},i}=\min\left\{t:\sum_{j=1}^{t}p_i(j)\ge q\right\}.
$$

$q=1$ 必然走到最后一步；降低 $q$ 才会更积极地早退。它不是“取 $p_i(t)$ 最大的 step”，也不是让模型凭空生成一个整数轮数。

## 5. 两阶段 gate training

论文区分联合训练 gate 与后续专门优化 gate。后者使用实际 performance improvement 构造 continuation/exit target，试图同时减少：

- **underthinking**：过早退出，错过有效 refinement；
- **overthinking**：继续执行，但输出已不再改善甚至退化。

专门 gate 在论文的 quality–average-depth Pareto curve 上优于仅靠标准 gate 的版本。注意这仍是模型内 proxy；要证明 production latency 收益，还要在 dynamic/continuous batching 中测真实调度。

更具体地，Stage II 冻结语言模型，只训练 gate。它比较同一 token 在相邻两轮的 detached cross-entropy：

$$
I_i^{(t)}=\max\left(0,\mathcal L_{i,\mathrm{stop}}^{(t-1)}-\mathcal L_{i,\mathrm{stop}}^{(t)}\right),
\qquad
w_i^{(t)}=\sigma\left(50\left(I_i^{(t)}-0.005\right)\right).
$$

$w_i^{(t)}$ 接近 1 表示“多算一轮仍有收益，继续”，接近 0 表示“收益已停，退出”；gate 用 binary cross-entropy 拟合这个 continuation target。因此轮数是**学出来的 loss-improvement proxy + 部署时阈值 $q$**共同决定的。

## 6. Upcycling、1.4B/2.6B 差异与 7.7T-token training flow

![Ouro 的端到端训练阶段](/assets/looped-transformer/15-ouro-looped-language-models/figure-4-training-pipeline.png)

*原论文 Figure 4，PDF p.8，[arXiv v5 PDF](https://arxiv.org/pdf/2510.25741v5)。看图重点：1.4B 与 upcycled 2.6B 路径共享早期训练逻辑，之后都经历 annealing、long-context、mid-training 与 reasoning SFT；最终能力不是 architecture 单变量的产物。*

### 6.1 `upcycle` 在这里具体是什么意思

它不是量化、蒸馏或把 dense FFN 改成 MoE，而是**拿已经训练过的小模型参数初始化更大的模型，再继续训练**：

1. Stage 1a 先训练一个 24-layer、$T=8$ 的共同起点，共 3T tokens；
2. 为解决 8 次 recurrence 的 loss spike / gradient oscillation，Stage 1b 把 $T$ 降到 4；
3. 1.4B 分支保留原来的 24 个物理层；
4. 2.6B 分支通过 layer duplication 把 24 层扩成 48 层，然后继续训练。复制出的层是独立参数，否则参数量不会从 1.4B 增至 2.6B；continued training 允许两份副本随后分化。

[复原推导] 这一步还解释了为什么作者说 upcycling 很“顺”：Stage 1a 的执行深度是 $24\times 8=192$ 次 block calls；2.6B 分支变成 $48\times4=192$，把一部分原先由 recurrence 提供的深度“展开”为独立层，保持了相同的有效 block-call 深度。论文只明确说 “24→48 via layer duplication”；并未把每个 layer 的精确复制映射写成可审计算法，所以不应进一步断言是逐层相邻复制还是整段 stack 复制。

### 6.2 两种规模逐项结构对照

以下结合论文 Table 2 与发布 checkpoint 的 [1.4B `config.json`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/main/config.json)、[2.6B `config.json`](https://huggingface.co/ByteDance/Ouro-2.6B/blob/main/config.json)；2.6B 的论文与 config 都是 48 层。2.6B model card 正文中曾出现“24 layers”的陈旧表述，与论文及实际 config 不一致，不应据此实现。

| 配置 | Ouro-1.4B | Ouro-2.6B |
|---|---:|---:|
| trainable parameters | 约 1.4B | 约 2.6B |
| 物理 / stored layers $N$ | 24 | 48 |
| 默认 recurrent steps $T$ | 4 | 4 |
| 默认 effective block calls $N\times T$ | 96 | 192 |
| hidden size $d$ | 2048 | 2048 |
| SwiGLU intermediate size | 5632 | 5632 |
| query heads / KV heads | 16 / 16（MHA） | 16 / 16（MHA） |
| head dimension | 128 | 128 |
| vocabulary | 49,152 | 49,152 |
| RoPE $\theta$ / max positions | $10^6$ / 65,536 | $10^6$ / 65,536 |
| token embedding 与 LM head | 不共享 | 不共享 |

单个 Transformer block 的实际残差路径可写为：

$$
h' = h + \operatorname{RMSNorm}_2\!\left(\operatorname{MHA}\!\left(\operatorname{RMSNorm}_1(h)\right)\right),
$$

$$
h'' = h' + \operatorname{RMSNorm}_4\!\left(\operatorname{SwiGLU}\!\left(\operatorname{RMSNorm}_3(h')\right)\right).
$$

也就是 attention 与 FFN 各自都有 pre-norm 和 post-norm（sandwich RMSNorm）。完整 $N$ 层跑完后再做一次 stack-final RMSNorm，得到 $h^{(t)}$，接共享 LM head 和一个 $2048\to1$ exit gate；然后同一套 $N$ 层以 $h^{(t)}$ 为输入进入下一轮。token embedding 只在最开始做一次，没有每轮重新 embedding。

`Base` 与 `Thinking` 不是另一套 backbone：1.4B/2.6B 各自的 Thinking 版本是在相同结构上做 reasoning SFT（约 8.3M examples）得到的，默认仍是 $T=4$。

### 6.3 训练阶段

| 阶段 | tokens | recurrent steps | 主要作用 |
|---|---:|---:|---|
| Pre-train I | 3T | 8 | broad web 为主，建立可循环的 base state dynamics |
| Pre-train II | 3T | 4 | stable training；2.6B 路径发生 upcycling |
| CT annealing | 1.4T | 4 | 提高 math/code 与高质量数据占比，cosine decay |
| LongCT | 20B | 4 | context 扩到 64K，调整 RoPE base |
| Mid-training | 300B | 4 | QA/CoT、math/code 与 replay mixture |
| Reasoning SFT | 单独阶段 | 4 | 得到 Ouro-1.4B/2.6B-Thinking |

合计约 $3+3+1.4+0.02+0.3=7.72$T tokens，论文概括为 7.7T。这个 accounting 很关键：与只训练几百 B tokens 的小模型横比时，不能把全部差异归因于 loop。

模型执行完整 stack 多次，因此 stored depth、effective depth 与 parameter count 必须分开：2.6B 不只是“比 1.4B 多一点参数”，在默认 $T=4$ 下它还执行两倍 block calls，单 token 串行计算也大约翻倍。

## 7. 实际自回归推理：loop、KV cache 与 early exit

### 7.1 一次 decode step 到底怎样执行

设当前已有 $s$ 个 token，要预测第 $s+1$ 个 token。默认 fixed-$T=4$ 的路径是：

```text
第 s 个位置的 embedding / 当前输入 hidden
  → loop 1: layer 1 ... N → h_s^(1), gate_1, logits_1
  → loop 2: layer 1 ... N → h_s^(2), gate_2, logits_2
  → loop 3: layer 1 ... N → h_s^(3), gate_3, logits_3
  → loop 4: layer 1 ... N → h_s^(4), gate_4, logits_4
  → 选择某一步的 logits → sample/argmax 得到 token s+1
```

注意 loop 是生成一个新 token **内部**的 latent-depth 迭代；它不增加输出序列长度，也不是生成 4 个候选 token。

### 7.2 默认 KV cache：每个“loop × 物理层”各一份

公开 Hugging Face 实现把 cache 索引写成：

$$
\text{cache\_idx}=t\cdot N+\ell,
$$

其中 $t\in\{0,1,2,3\}$ 是 loop，$\ell\in\{0,\ldots,N-1\}$ 是物理层。因此：

- 1.4B 默认有 $4\times24=96$ 组 layer KV cache；
- 2.6B 默认有 $4\times48=192$ 组 layer KV cache；
- 在 decode 当前 token 时，loop $t$、layer $\ell$ 读取过去 token 在同一个 effective layer $(t,\ell)$ 的 K/V，并把当前 token 的 K/V 追加进去。

prefill 时整个 prompt 依次通过四个完整 stack，同时填满四套 cache。论文 Section 5.4.2 报告：prefill 强行共享不同 loop 的 cache 会让 GSM8K 下降超过 10 points，所以四轮都需要自己的表示。以 bf16、16 个 KV heads、head dim 128 粗算，不含 allocator/metadata：

$$
\text{KV bytes/token}=2(K,V)\times16\times128\times2\text{ bytes}\times N\times T.
$$

所以 full-cache 约为：1.4B 每个序列 token **768 KiB**，2.6B 每个序列 token **1.5 MiB**；4K context 分别约 **3 GiB** 与 **6 GiB/sequence**。这里使用的是 MHA 而非 GQA，因此 KV cache 尤其昂贵。

### 7.3 论文的 decode-only cache reuse

论文在 autoregressive decoding 阶段测试了只保留一个 loop cache 的近似：

| cache 策略 | GSM8K | MATH-500 | 相对 full-cache 内存 |
|---|---:|---:|---:|
| 四轮各自 cache | 78.92 | 82.40 | $1\times$ |
| 只复用第一轮 | 18.73 | 8.43 | $1/4$ |
| 只复用最后一轮 | 78.85 | 80.40 | $1/4$ |
| 四轮 K/V 平均后复用 | 78.73 | 78.52 | $1/4$ |

结论是：**prefill 仍需 full loop-specific cache；进入 decode 后，丢弃/复用到 last-step cache 可以近似保住质量并省约 4× cache memory**。但公开 Hugging Face `UniversalTransformerCache` 的默认路径仍分配 $N\times T$ 个 cache slots；论文的 last-step reuse 是额外 serving optimization，不应假定普通 `transformers.generate()` 已自动启用。

### 7.4 每个 token 独立决定轮数吗？数学上是；默认高吞吐实现里不完全是

- **训练目标：是。** gate 输出 shape 是 `[batch, seq, 1]`，$\lambda_{i,t}$、loss improvement $I_i^{(t)}$ 与 continuation target $w_i^{(t)}$ 都按 token 位置计算。
- **自回归语义：是。** 每次生成只关心当前最后一个位置；预测不同 next token 时，gate 可以给出不同 $t_{\mathrm{exit}}$。更精确地说，是“用于预测下一个 token 的当前位置”独立选择 latent depth。
- **官方 checkpoint 默认配置：不是动态的。** `early_exit_threshold=1.0`，所以每个 token 都执行/选择第 4 轮。改变 `total_ut_steps` 是对整个请求固定成 1、2、3、4…轮。
- **当前 Hugging Face reference forward：只做 token-wise 输出选择，未做真实 compute early-exit。** 代码先运行全部 `total_ut_steps`，收集所有 `hidden_states_list` 和 gate，再按每个 token 的 CDF 选择输出 hidden state。因此即便把 threshold 调低，reference path 也不会少算前面的完整 loops；它只会改变用哪一轮的 logits。
- **vLLM model-card 路径：固定 full steps。** 官方 model card 明确说明当前 vLLM 不支持 adaptive exit，会始终执行 `total_ut_steps`。

[综合判断] 真正的 token-wise 加速还要解决两件事：一是 batch 内不同样本在不同轮退出，需要 compaction/ragged scheduling；二是较浅 token 没产生更深 loop 的 K/V，而未来较深 token 可能需要这些 cache。可以继续全算 cache（没有算力收益）、使用论文的 last-step cache reuse（近似），或设计 selective/ragged KV cache。Ouro 证明了 gate 与 decode cache reuse 各自可用，但没有把“token-wise halting + continuous batching + exact KV semantics”完整解决成一个现成的通用 serving 系统。

### 7.5 实际该怎样选轮数

1. **复现实验先用 fixed $T=4$**：这是 checkpoint 默认值和主要训练深度。
2. **追求速度可先固定扫 $T=2,3,4$**：这会真的减少 block calls，部署最简单；2.6B Thinking 在若干任务甚至 $T=3$ 最好。
3. **再校准 Q-exit 阈值 $q$**：在 validation set 上画 quality–平均 depth–wall-clock 曲线，而不是只看 gate 的平均轮数。$q=1$ 固定最后一步，越小越早退。
4. **不要盲目设 $T>4$**：虽然 config 允许，论文中多数能力在 3–5 附近峰值，6–8 常明显退化。
5. **区分“选择早期 logits”和“真的少算”**：只有 runtime 在 gate 触发后停止后续 stack，并正确维护 batch/cache，才产生 latency/FLOP 收益。

## 8. 数据 recipe 的能力归因

早期数据包含 Nemotron-CC、M-A-P web corpus、code 与 math。CT annealing 提升高质量和 math/code 权重；LongCT 使用长文数据；mid-training 约包含 90B QA/CoT、30B 第一阶段 replay 与 180B 第二阶段 replay。

这说明 Ouro 的 reasoning gain 至少有四个可能来源：

1. shared full-stack recurrence；
2. multi-trillion-token broad pretraining；
3. high-quality/math/code annealing；
4. reasoning SFT。

只有 stage-by-stage checkpoint 和 matched vanilla control 才能把它们拆开。

## 9. 主要实验结果

### 9.1 Base capability

| 模型 / depth | MMLU | BBH | GSM8K | MATH-500 |
|---|---:|---:|---:|---:|
| Ouro-1.4B, $T=4$ | 67.35 | 71.02 | 78.92 | 82.40 |
| Ouro-2.6B, $T=4$ | 74.60 | 80.46 | — | 90.85 |

2.6B 的 MMLU-Pro 为 55.73。对 stored parameter count 而言结果很强，但比较表中的其他模型通常有不同 tokenizer、data、tokens 与 post-training，因此属于定位证据而非 architecture causal proof。

### 9.2 Thinking model

论文对 1.4B Thinking 模型报告 AIME 2024 pass@1/pass@k 约 65.0/83.3，AIME 2025 为 46.3/73.3；2.6B 为 64.7/90.0 与 50.3/76.7。它说明较小 recurrent model 在特定 reasoning recipe 下可以匹配更大模型的部分数学结果。

不能忽略 sampling budget：pass@k 与 pass@1 回答的是不同问题，且 explicit output tokens、latent loops 和 candidate selection 都是 inference compute。

### 9.3 深度外推的负面证据

训练的主要最大 depth 是 4。1.4B base 在 MMLU 上约从 $T=4$ 的 67.45 降到 $T=8$ 的 64.49；2.6B Thinking 在 AIME 2024 上约从 $T=3$ 的 70.33 降到 $T=8$ 的 39.00。

因此最稳妥的结论是：

> Ouro 在训练覆盖的有限 depth 区间内学到了可分配 latent compute，但没有学到“多做一步必然更好”的稳定迭代算子。

这条负面结果比最佳 benchmark 更重要，因为它决定 adaptive exit 是可选优化还是必要组件。

## 10. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| scale | Looped LM 扩展到 7.7T-token staged training | 强：paper + released weights |
| architecture | 完整 model stack recurrence，每步可读出 | 强 |
| adaptive compute | 概率 exit 与专门 gate training | 中强：有 Pareto 曲线，缺 production scheduler 证据 |
| capability | 1.4B/2.6B base 与 Thinking 结果有竞争力 | 中：外部比较 data/post-training 不完全 matched |
| interpretability | 中间 step 可反映逐步 refinement/faithfulness proxy | 弱到中：观察关联，不是 causal tracing |
| openness | weights 与 project 信息 | 中：code/data 尚不完整 |

## 11. Claim–evidence map

| 可说的 claim | 直接证据 | 不可外推 |
|---|---|---|
| Looped LM 能完成 multi-trillion-token training | 7.7T recipe、checkpoints | 训练成本低于相同质量 vanilla |
| adaptive exit 能改善 quality–average-depth tradeoff | gate comparison | 部署 wall-clock 必然同比下降 |
| 小 stored model 可达到强 base/reasoning 分数 | tables | recurrence 是唯一原因 |
| $T=3$–$5$ 常有有效 refinement | depth sweeps | $T>4$ 仍单调改善 |
| latent states 展现某些 reasoning 关联 | probe/trajectory observations | latent state 等价于可忠实解释的 CoT |

## 12. 关键边界

1. **code/data 开放度**：weights 可用，但训练 code 与完整 data mixture 还不足以逐项复现。
2. **计算匹配**：完整 stack 重复使单次 loop 成本高；parameter-matched 不是 FLOP-matched。
3. **depth 稳定性**：超过训练 depth 明显退化，不能宣传成 unbounded test-time scaling。
4. **stage 混杂**：7.7T data curriculum 与 reasoning SFT 都很强，architecture 归因需要 stage controls。
5. **通用 post-training**：主要展示 base 与 reasoning，multi-turn chat、tool use、safety、preference retention 仍不充分。
6. **“单调改进”**：任何 $\mathbb E[\mathcal L_{t+1}]\le\mathbb E[\mathcal L_t]$ 表述都应理解为训练目标或经验趋势，不是架构定理。

## 13. 与其他论文的连接

- 与 **Huginn**：Huginn 循环内部 core，Ouro 循环完整 stack；前者 artifacts 更开放，后者 data scale 更大。
- 与 **MoR**：Ouro 的 gate/输出选择可以是 token-wise，但完整 stack 仍以 dense 方式执行；MoR 进一步把 token-level routing 落到选择性执行与 selective KV cache。
- 与 **LoopRPT**：LoopRPT 直接从 Ouro checkpoint 出发，把 reward 放到 latent recurrence 与 exit 上。
- 与 **LOTUS**：Ouro 用每步 next-token/gate 目标；LOTUS 用 gold CoT 对 latent blocks 做直接 supervision。
- 与 **Loopie**：两者都接近现代通用 LLM，但 Loopie 更强调 MoE 和 measured wall-clock comparison。

## 14. 复现路线

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

## 15. 可证伪的新问题

- **过度计算是否来自 state drift？** 测 $\|h^{(t+1)}-h^{(t)}\|$、logit KL 与 accuracy；若 state 收敛但 accuracy 仍崩溃，则不能只怪 drift。
- **gate 是否只学到 task/difficulty prior？** 打乱 prompt difficulty 或做 adversarially easy-looking hard questions；若 depth allocation 不变，gate 可能未读取真实 state improvement。
- **普通 DPO/RLVR 会不会破坏 exit calibration？** 保存每个 post-training stage，比较 reliability curve 与 performance–depth curve。

## 16. 自测题

1. 写出 $p(t)$ 与 $S_t$ 的关系，并计算一个三步示例。
2. 为什么 Ouro 的 7.7T tokens 是理解 benchmark 的必要背景？
3. 完整 stack loop 与 recurrent-core loop 在 latency 和参数占比上有什么不同？
4. $T=4$ 最好、$T=8$ 变差说明了哪两种可能机制？
5. 如何验证 adaptive exit 的真实 wall-clock 收益，而不只验证平均 FLOPs？

## 17. 一句话定位

Ouro 是当前“把 Looped LM 训练成有竞争力通用 base/Thinking model”的关键规模证据；它也用清晰的 depth 退化结果提醒我们，adaptive compute 的核心是**知道何时停止**，不是无条件增加循环。
