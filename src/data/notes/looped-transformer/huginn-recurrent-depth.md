---
title: "Huginn：Recurrent Depth 与 Latent Test-Time Compute"
description: "复原 3.5B/800B-token recurrent pretraining 配方，核查随机深度、input injection、truncated BPTT 与测试时深度扩展。"
topic: "looped-transformer"
section: "llm-pretraining"
slug: "huginn-recurrent-depth"
date: 2026-08-02
updated: 2026-08-04
cutoff: 2026-08-02
featured: true
order: 50
source:
  repository: "J-shang/looped-transformer"
  path: "papers/14-huginn-recurrent-depth.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-04"
  contentHash: "sha256:2fbd57c18f19a04b4329176488fda59113955aab1a65888fccf99450bc7f280b"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
> 论文：*Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*<br>
> 精确版本：[arXiv:2502.05171v2](https://arxiv.org/abs/2502.05171v2)，2025-02-17<br>
> 作者与机构：Jonas Geiping 等；ELLIS Institute Tübingen、Max Planck Institute for Intelligent Systems、University of Maryland、LLNL 等<br>
> 官方 artifacts：[code](https://github.com/seal-rg/recurrent-pretraining) · [model](https://huggingface.co/tomg-group-umd/huginn-0125) · [training data](https://huggingface.co/datasets/tomg-group-umd/huginn-dataset)<br>
> 证据标签：**A**。paper、code、weights、data 都可追踪，但作者明确把 3.5B/800B-token 结果定位为 proof of concept。

## 1. 30 秒结论

Huginn 是研究通用 Looped LLM 最好的起点之一。它不是把普通模型在推理时随意重复，而是从 pretraining 开始就让同一个 recurrent block 经历随机次数的复用，并用 input injection、随机初始 state、normalization 和 truncated BPTT 维持稳定。训练后的 3.5B 模型可以在 inference 增加 recurrence，以不增加 stored parameters 的方式改善 GSM8K、code 等任务。

最重要的边界同样清楚：更多 loop 仍然消耗串行 FLOPs 和 latency；不同任务的收益会饱和；800B tokens 与 3.5B 规模不足以证明它已经优于现代 fully trained 7B/8B vanilla LLM。

## 2. 学习目标

读完后应能回答：

1. prelude–recurrent core–coda 与普通 depth-tied Transformer 有什么区别；
2. 为什么模型必须在 training 时见过 depth distribution，而不是只在 inference 重复 block；
3. random state、input injection、truncated BPTT 分别解决什么问题；
4. 哪些任务真正表现出 test-time depth scaling，哪些任务很快饱和；
5. 为什么“参数不增加”不等于“计算免费”。

## 3. 符号与结构

| 符号 | 含义 |
|---|---|
| $x$ | 输入 token sequence |
| $e=P(x)$ | prelude $P$ 产生、每轮可重新注入的输入表示 |
| $s_i$ | 第 $i$ 次 recurrence 后的 latent state |
| $R$ | 共享参数的 recurrent core |
| $C$ | coda 与 LM head |
| $r$ | 当前样本执行的 recurrence 数 |
| $k$ | 反向传播保留的最后 recurrence 数，主实验为 8 |

核心计算可写成：

$$
e=P(x),\qquad s_0\sim\mathcal N(0,\sigma^2 I),
$$

$$
s_i=R(e,s_{i-1}),\quad i=1,\ldots,r,\qquad
p(\text{next token})=C(s_r).
$$

这里有两个容易忽略的设计：$R$ 的参数在所有 $i$ 上共享；$e$ 每轮注入，因此 state 不必在第一次访问后永久保存全部输入信息。

![Huginn 的 prelude、recurrent block 与 coda](/assets/looped-transformer/14-huginn-recurrent-depth/figure-2-architecture.png)

*原论文 Figure 2，PDF p.3，[arXiv v2 PDF](https://arxiv.org/pdf/2502.05171v2)。看图重点：绿色 block 是同一组参数的重复调用；虚线是 input injection；blocked gradient 表示早期 recurrent states 不进入完整反向图。原图承担结构与梯度路径证据，不代表系统吞吐。*

## 4. 论文解决的真正问题

标准 decoder-only LLM 把能力主要存进更多不同层的参数。显式 Chain-of-Thought 则在输出空间增加 sequential tokens。Huginn 探索第三个轴：让内部 latent state 经过同一计算模块多次更新。

一个最小类比是草稿迭代：

```text
输入表示 e
  → state 草稿 s1
  → 用同一编辑器 R 修订为 s2
  → 再修订为 s3
  → coda 只读取最终版本
```

关键难点不是“能否写出循环”，而是同一 block 在不同深度收到分布不断变化的 state，容易出现 activation drift、梯度爆炸/消失或只适应固定 depth。论文的贡献主要针对这些训练问题。

## 5. 方法复原

### 5.1 物理深度与有效深度

主模型约 3.5B parameters，使用 8 个实际保存的 Transformer layers，布局为 $(2,4,2)$：2 层 prelude、4 层 recurrent core、2 层 coda。若 core 重复 32 次，effective execution depth 远大于 8，但 stored parameters 不变。

参数大致分布为：prelude/coda 约 1.5B、recurrent core 约 1.5B、embedding 约 0.5B。这个比例提醒我们：parameter sharing 只作用于 core，embedding、首尾层和 LM head 仍然占显著空间。

#### Huginn-0125 的逐层配置

下面是主 checkpoint 的实际结构，而不只是 `prelude → core → coda` 的概念图。数值来自论文 §3.2、§4.1 与[官方 checkpoint config](https://huggingface.co/tomg-group-umd/huginn-0125/blob/main/config.json)，算子顺序与 bias/dropout 由[官方最小推理实现](https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/raven_modeling_minimal.py)核对。

| 项目 | Huginn-0125 |
|---|---|
| 模型类型 | decoder-only autoregressive Transformer，dense model，不是 MoE |
| stored Transformer blocks | 8：2 prelude + 4 recurrent core + 2 coda |
| effective block depth | $2+4r+2$；默认 $r=32$ 时为 132 |
| hidden size | $h=5280$ |
| context / block size | 4096 tokens |
| attention | dense causal self-attention；无 sliding window、local/sparse attention |
| heads | 55 query heads、55 KV heads、head dim 96；因此是普通 MHA，不是 GQA/MQA |
| position encoding | RoPE，base 50,000，施加在 Q/K 上 |
| attention bias/dropout | 只有 Q、K 有 learned bias；V、attention output projection 无 bias；attention dropout 为 0 |
| FFN | dense gated-SiLU / SwiGLU-style MLP，$5280\to 2\times17920\to17920\to5280$，无 bias、无 experts/router |
| normalization | learned-scale RMSNorm，$\epsilon=10^{-6}$；每个 block 有 4 个独立 RMSNorm |
| embeddings/head | vocabulary 65,536；input embedding 与 LM head 权重绑定；embedding 输出乘 $\sqrt{5280}\approx72.66$ |
| input injection | 每轮把 $s_{i-1}$ 与固定的 prelude 输出 $e$ concat，经过无 bias 的 $10560\to5280$ learned linear adapter |
| recurrent state init | 与 token 序列同 shape 的随机 state；主模型用截断正态，最终每维方差为 $2/5$ |

这里的 **full attention** 需要消除一个术语歧义：如果它指“每个 token 可看见窗口内所有历史 token”，答案是 **是**；如果它指 BERT 式双向 all-to-all attention，答案是 **否**。Huginn 是自回归 LM，所以 attention mask 是下三角 causal mask。实现调用 scaled dot-product attention，prefill 路径设 `is_causal=True`、`dropout_p=0.0`；没有 attention sink、linear attention 或分层 local/global pattern。

单个 `SandwichBlock` 的精确计算为：

$$
u=\operatorname{RMSNorm}_2\!\left(x+
\operatorname{MHA}\!\left(\operatorname{RMSNorm}_1(x)\right)\right),
$$

$$
y=\operatorname{RMSNorm}_4\!\left(u+
\operatorname{MLP}\!\left(\operatorname{RMSNorm}_3(u)\right)\right).
$$

所以它既不是纯 pre-norm，也不是纯 post-norm，而是每个 attention/MLP 子层前后都 norm：

```text
x ─RMSNorm₁─MHA─Add(x)─RMSNorm₂─RMSNorm₃─SwiGLU-MLP─Add─RMSNorm₄ → y
```

`RMSNorm₃` 的输入已经经过 `RMSNorm₂`，论文明确说它在数学上可视为冗余，但最终 checkpoint 确实保留了它。RMSNorm 带一个可学习的逐通道 scale，均方根统计在官方实现中以 FP32 计算后再 cast 回 activation dtype；它不是 parameter-free RMSNorm，也不是 LayerNorm，因此没有减均值步骤和 beta bias。除此之外，官方实现还复用一个 `ln_f` RMSNorm：一次放在 recurrent core 结束、进入 coda 之前，一次放在两层 coda 之后、LM head 之前。

整个 forward 可以写得更具体：

```text
token ids
  → tied embedding × √5280
  → 2 × SandwichBlock                         = e
  → random s₀
  → repeat r times:
        Linear([sᵢ₋₁ ; e])                    # 10560 → 5280
        → 4 × the same shared SandwichBlocks  = sᵢ
  → shared final RMSNorm
  → 2 × SandwichBlock
  → the same final RMSNorm
  → tied vocabulary projection → logits
```

这里共享的是 **4 个 core blocks 的整组参数**：第 1 个 core block 在所有 recurrence 中复用自己的权重，第 2–4 个同理；并不是 4 层彼此也共享成同一层。prelude、core 内四层、coda 都各有独立参数。$r=32$ 时，执行的是 2 个首层、$4\times32$ 次 core block call 和 2 个尾层。

### 5.2 随机 recurrent depth

训练时不固定 $r=32$，而是从 lognormal–Poisson 组合分布抽样。直觉上：

- 浅样本提供便宜、稳定的梯度；
- 深样本让模型学会在多轮更新中保持可用状态；
- depth variability 防止模型把“第 32 轮”硬编码为唯一读出点。

这是一种训练分布设计，不构成任意 depth extrapolation 保证。若 inference depth 远超训练分布，仍可能饱和或退化。

### 5.3 随机初始 state 与 input injection

$s_0$ 不是简单复制 embedding，而是随机 state；原输入表示 $e$ 每轮注入。这样会迫使 recurrent core 学习“在有噪声的工作区中持续利用输入”，降低只依赖某个固定初始化捷径的风险。

但随机 state 不是 latent reasoning 的充分条件。它只是改变 optimization pressure；是否形成算法式迭代，需要用跨 loop 的 loss、表示和任务曲线验证。

### 5.4 truncated BPTT

若对 32 次甚至 64 次 recurrence 完整反向传播，activation memory 和训练 FLOPs 会很高。Huginn 只对最后 $k=8$ 个 recurrent steps 反传，之前的 state stop-gradient：

$$
\frac{\partial \mathcal L}{\partial s_i}=0,
\qquad i<r-k.
$$

优点是训练可扩展；代价是早期 steps 只通过其产生的 state 间接影响后续，缺少完整长程 credit assignment。它可能偏向“最后几轮修正”，而不是端到端优化整条轨迹。

### 5.5 normalization 与初始化

论文报告多次失败 run，并强调 activation/gradient norm、初始化和 normalization 对深 recurrence 极其敏感。这不是附属工程细节：共享 block 的小偏差会被反复放大，所以 recurrent pretraining 的稳定区间通常比普通固定深度模型窄。

## 6. 训练 recipe

| 项目 | 主模型设置 |
|---|---|
| parameters | 约 3.5B |
| stored layers | $(2,4,2)$，共 8 层 |
| 目标平均 recurrence | 32 |
| pretraining tokens | 800B |
| global tokens / optimizer step | 约 16M |
| optimizer steps | 约 47k |
| 硬件 | 4096 AMD MI250X GPUs |
| learning-rate schedule | 主阶段近似 constant LR，无常规 cooldown |
| backward horizon | 最后 8 个 recurrent steps |

公开 repo 中的 `train.py`、`recpre/model_dynamic.py`、`recpre/model_registry.py` 和 launch configurations 是 paper 之外最重要的证据：它们固定了 depth sampling、动态 forward、checkpoint 与训练配置。复现时应记录代码 commit，而不是只写“按论文实现”。

## 7. 实验问题与证据

### 7.1 增加 inference recurrence 是否有用

主模型在 $r=32$ 附近报告：ARC-Challenge 38.23、HellaSwag 65.21、MMLU 31.38、GSM8K CoT strict/flexible 34.80/42.08。绝对分数不是最值得记忆的部分；关键是同一 checkpoint 的任务曲线不同。

![不同任务随 test-time recurrence 的变化](/assets/looped-transformer/14-huginn-recurrent-depth/figure-7-test-time-scaling.png)

*原论文 Figure 7，PDF p.8，[arXiv v2 PDF](https://arxiv.org/pdf/2502.05171v2)。看图重点：HellaSwag 在较浅 recurrence 已接近平台，GSM8K 与 HumanEval 在更深区间仍受益；到 32–64 后也趋于饱和。图显示 task-dependent scaling，不证明每个 prompt 都应使用同一 depth。*

这说明 latent depth 更像可分配资源，而不是统一的“越多越好”。合理的下一步是 per-token 或 per-sequence early exit，而不是把所有请求固定到最大 $r$。

### 7.2 收益是否只是“执行了更多 FLOPs”

在一个 180B-token controlled comparison 中，固定深度模型与 recurrent 模型的部分结果为：

| 指标 | fixed-depth | recurrent | 差值 |
|---|---:|---:|---:|
| ARC-Challenge | 26.96 | 29.18 | +2.22 |
| HellaSwag | 37.34 | 48.80 | +11.46 |
| GSM8K CoT strict | 1.82 | 9.02 | +7.20 |
| GSM8K CoT flexible | 2.20 | 10.24 | +8.04 |

这个实验支持“训练 recurrence 改变了能力”，但并未消除所有 compute confound。公平比较至少要同时给出 stored parameters、executed block calls、training tokens、analytical FLOPs 和 measured wall-clock。

### 7.3 是否已经达到现代通用 LLM 水平

没有。论文也没有这样主张。3.5B/800B 是为了展示 recurrent pretraining 与 test-time scaling 可以在自然语言模型上成立；它仍落后于训练更充分、data recipe 更现代的 open-weight LLM。跨模型 leaderboard 只能给定位，不能归因 architecture。

## 8. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| architecture | prelude–recurrent core–coda 与 input injection | 强：paper + code |
| optimization | 随机 depth、random state、truncated BPTT 的大规模组合 | 中强：有消融与失败 run，但规模仍单一 |
| scaling | 同一 checkpoint 随 test-time recurrence 改变性能 | 中强：多任务曲线；存在饱和 |
| openness | release code、model、training dataset | 强 |
| generality | 通用语言建模上的可行性 | 中：不是现代全流程 instruct/chat model |

## 9. Claim–evidence map

| 可说的 claim | 直接证据 | 不能外推成什么 |
|---|---|---|
| 3.5B recurrent LM 可稳定训练到 800B tokens | 训练 run、checkpoint、data、code | trillion-token 训练已普遍解决 |
| 部分 reasoning/code 任务随 recurrence 改善 | 同 checkpoint depth sweep | 所有任务单调改善 |
| 参数量固定时可增加 effective depth | architecture | inference FLOPs/latency 不增加 |
| random depth 让多个读出深度可用 | 多 depth 评测 | 任意远超训练范围仍可靠 |
| Huginn 是开放度较高的研究平台 | 四类 artifact | 所有训练细节都已无歧义复现 |

## 10. 局限与风险

1. **规模边界**：3.5B/800B 低于当前主流 7B–70B、multi-trillion-token recipe。
2. **训练目标边界**：主要是 base LM pretraining，没有完整 SFT、DPO、RLVR、tool-use 与 safety flow。
3. **计算边界**：更深 recurrence 增加串行 block calls；parameter efficiency 不等于 throughput efficiency。
4. **评测边界**：academic tasks 多，multi-turn chat、knowledge retention、long context 和 production latency 不足。
5. **优化边界**：truncated BPTT 改变了真实 gradient；深层 credit assignment 仍是开放问题。

## 11. 与通用 LLM 生命周期的连接

Huginn 主要覆盖：

```text
broad data → recurrent pretraining → base checkpoint → variable-depth inference
```

它没有回答：普通 SFT/DPO/RLVR 能否保留 depth scaling；reasoning RL 应奖励 final answer 还是 latent steps；chat/safety alignment 是否会让 recurrence collapse。后续应与 Retrofitted Recurrence、LOTUS、LoopRPT 组合阅读。

## 12. 复现与扩展建议

### 最小复现

1. 固定 Huginn checkpoint、tokenizer、prompt 与 decoding；
2. 扫描 $r\in\{1,2,4,8,16,32,64\}$；
3. 同时报 perplexity、knowledge、math/code、latency 与 peak memory；
4. 不只记录最佳值，画出每个任务的完整 curve；
5. 按样本难度分桶，检查“困难样本需要更多 depth”是否成立。

### 可证伪扩展

- **假设 A**：普通 instruction SFT 会压缩可用 depth 区间。比较 SFT 前后完整 depth curve；若平台位置与斜率不变则否定。
- **假设 B**：per-example exit 能在几乎不降质量时减少平均 block calls。必须在 continuous batching 下测 wall-clock；只算 FLOPs 不足以支持。
- **假设 C**：full BPTT 比 truncated BPTT 更利于远距离 depth extrapolation。用小模型控制 memory，比较训练范围外 recurrence；若仅训练 loss 改善则不成立。

## 13. 自测题

1. 若模型只有 8 个 stored layers、recurrent core 重复 32 次，为什么不能把它简单称为“128 层模型”？
2. input injection 如何降低 state 遗忘输入的风险？它为什么不保证 convergence？
3. truncated BPTT 节省了什么，又切断了什么？
4. Figure 7 为什么支持 adaptive depth，却不支持“64 loops 最好”？
5. 设计一个 parameter-matched、FLOP-matched 和 wall-clock-matched 三联 baseline。

## 14. 一句话定位

Huginn 建立了“通用 base LM 可以从 pretraining 起学习 recurrent depth，并把 inference loops 变成 latent compute budget”的可信可复现起点；它还没有建立现代通用 LLM 全流程中的最终优势。
