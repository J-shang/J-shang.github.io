---
title: "Retrofitted Recurrence：把 Pretrained LLM 改造成 Recurrent-Depth Model"
description: "复原 pretrained-model surgery、recurrence curriculum 与两阶段 healing，评估约 50B continued-training tokens 的收益和边界。"
topic: "looped-transformer"
section: "llm-retrofit"
slug: "retrofitted-recurrence"
date: 2026-08-02
updated: 2026-08-02
cutoff: 2026-08-02
featured: true
order: 60
source:
  repository: "J-shang/looped-transformer"
  path: "papers/17-retrofitted-recurrence.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-02"
  contentHash: "sha256:930be3bde5ca042353e317452a22a2675e9c8538f3dfecbe9aefcc3fe40359dd"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
> 论文：*Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence*<br>
> 精确版本：[arXiv:2511.07384v1](https://arxiv.org/abs/2511.07384v1)，2025-11-10<br>
> 作者与机构：Sean McLeish、Ang Li、John Kirchenbauer 等；University of Maryland、NYU、LLNL、UNC、ELLIS / MPI、Columbia<br>
> 官方 artifacts：[code](https://github.com/mcleish7/retrofitting-recurrence) · [models](https://huggingface.co/collections/tomg-group-umd/retrofitting-recurrence)<br>
> 证据标签：**B**。code 与 models 公开；continued training 主要在约 1B 规模和 math-oriented data 上完成。

## 1. 30 秒结论

这篇论文回答一个非常现实的问题：如果已有一个 pretrained vanilla LLM，能否删掉中间层、把后部若干层变成 recurrent core，再用远少于完整 pretraining 的 continued training “愈合”模型？答案是可以恢复并在数学任务上获得 test-time depth scaling，但不是无成本转换。

最强证据包括：pretrained initialization 明显优于随机初始化，估计随机模型需约 950B tokens 才能追平；recurrence curriculum 与 Muon 提高稳定性；高质量 data 的两阶段 healing 对 general benchmark retention 很重要。边界是模型约 1B、额外训练约 50B tokens、能力目标偏数学。

## 2. 学习目标

1. 复原 model surgery 的层选择与 adapter；
2. 理解 pretrained initialization 为什么仍可利用被保留的深层参数；
3. 区分 recurrence curriculum、math continued training 与 data healing 的贡献；
4. 用 compute formula 比较 retrofit 和继续训练原 vanilla model；
5. 判断何时应 retrofit，何时应从头 recurrent pretraining。

## 3. 符号表

| 符号 | 含义 |
|---|---|
| $(p,r,c)$ | prelude、recurrent core、coda 各自的 stored layers 数 |
| $e$ | prelude 输出 |
| $s_i$ | recurrent core 第 $i$ 次访问后的 state |
| $A$ | 拼接 $e$ 与 state 后的 linear adapter |
| $N_1$ | 不参与 recurrent backward 的参数量 |
| $N_2$ | recurrent / activation-related 计算项对应参数量 |
| $D$ | continued-training tokens |

## 4. Model surgery

论文从 pretrained model 中保留：

- 最早若干层作为 prelude；
- 较后的若干层作为 recurrent block 与 coda；
- 删除中间层；
- 把 prelude output 与 recurrent state 拼接，经 adapter 后进入共享 core。

![从 pretrained layers 到 recurrent model 的 surgery](/assets/looped-transformer/17-retrofitted-recurrence/figure-1-model-surgery.png)

*原论文 Figure 1，PDF p.1，[arXiv v1 PDF](https://arxiv.org/pdf/2511.07384v1)。看图重点：层并非简单等距折叠；早层保留输入处理，晚层承担 recurrence/readout，中间层被移除。转换后必须继续训练以修复新的 state distribution。*

代表性布局：

| Base family | retrofit layout | body retained |
|---|---:|---:|
| TinyLlama | $(4,8,4)$ | 约 700M，72.7% |
| OLMo | $(4,6,4)$ | 约 900M，87.5% |
| Llama | $(4,6,4)$ | 约 1B 级 retained body |

这个转换同时改变 parameter count、depth、residual path 和中间表示分布。因此合理 baseline 不是未训练的 surgery model，而是：原始 vanilla、等 tokens 继续训练的 vanilla、pretrained-init retrofit、random-init recurrent。

## 5. 为什么 pretrained initialization 有效

尽管层的连接方式改变，保留层仍含语言、知识和局部计算模式。continued training 的任务更像 distribution repair，而不是从零学习全部 representation。论文的 scaling estimate 指出，随机初始化 recurrent model 可能需要约 950B tokens 才追上 pretrained-init retrofit 的水平。

这个数字是根据所测 scaling trend 的估计，不是实际完成的 950B-token run。可说“预训练参数提供显著 sample-efficiency”，不可说“精确节省了 950B tokens”。

## 6. Recurrence curriculum 与梯度

模型起初执行较少 loops，训练中逐渐提高 lognormal–Poisson depth distribution 的 mean，最终目标约为 32。curriculum 避免一开始就把尚未适配的 block 放入极深动态。

与 Huginn 类似，模型只对最后 8 次 recurrence 做 backpropagation。粗略训练 FLOPs 记为：

$$
\mathrm{FLOPs}\approx(6N_1+2N_2)D.
$$

该式把参数相关前向/反向项区分开，适合做 analytical accounting；它仍不能替代 kernel、communication、activation checkpointing 与 wall-clock 测量。

## 7. Optimizer 与稳定性

论文发现 Muon 相比 AdamW 更适合这组 recurrent continued-training runs。可能原因包括 matrix update 的尺度与方向更稳定，但实验只能支持“在作者配置中更稳/更好”，不能外推为 recurrent LM 的普遍最优 optimizer。

复现时必须同时固定：optimizer implementation、parameter groups、learning rate、warmup、norm clipping、precision 和 sequence length。只写“使用 Muon”不足以复现。

## 8. Math-oriented continued training

模型使用约 50B Nemotron-CC-Math-v1-4plus 等数据继续训练。在固定训练 FLOPs 下，recurrent retrofit 在 GSM8K/MATH 上可优于继续训练原 vanilla model。

![不同模型家族的数学任务与 recurrence](/assets/looped-transformer/17-retrofitted-recurrence/figure-7-math-scaling.png)

*原论文 Figure 7，PDF p.9，[arXiv v1 PDF](https://arxiv.org/pdf/2511.07384v1)。看图重点：TinyLlama、Llama、OLMo 三个家族的 recurrent checkpoints 在 GSM8K/MATH 上随 inference recurrence 改变；这提高了跨 backbone 可信度，但仍只覆盖约 1B 与 math-heavy continued training。*

结果支持：已有 general base model 的参数可以被重排后继续利用，且 inference recurrence 在数学任务上成为可调 budget。它不证明被删除的层完全冗余，也不证明 general capabilities 无损。

## 9. 两阶段 healing 与能力保持

只用混合数据一次性训练容易让 surgery 后的模型同时面对结构适配和 domain shift。论文比较：

1. **single phase**：26B mixed FineWeb-Edu / Nemotron General / Math；
2. **two phase**：先 26B FineWeb-Edu healing，再 26B mixed data。

![高质量数据与两阶段 healing](/assets/looped-transformer/17-retrofitted-recurrence/figure-8-healing-phases.png)

*原论文 Figure 8，PDF p.10，[arXiv v1 PDF](https://arxiv.org/pdf/2511.07384v1)。左图是训练过程，右图是 inference recurrence；两阶段方案在 non-reasoning benchmark 上恢复更好。看图重点：data curriculum 不只是 reasoning specialization，也负责修复 architecture surgery。*

两阶段 recurrent model 在 test recurrence 32 时报告 ARC-Challenge 37.7、MMLU 44.8、GSM8K 51.2；两阶段 static TinyLlama 为 36.5、44.4、45.2。差值支持 recurrence 的部分收益，但幅度和方向随任务变化，应保留完整表而不是只报 GSM8K。

## 10. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| model conversion | pretrained vanilla layers → prelude/core/coda | 强：多 family + code/models |
| sample efficiency | pretrained init 显著优于 random init | 中强：有 scaling extrapolation，950B 非实跑 |
| optimization | recurrence curriculum、truncated BPTT、Muon | 中：组合有效，独立因果仍有限 |
| data curriculum | general healing → mixed math training | 中强：有 controlled comparison |
| generality | 三个约 1B family | 中；规模与任务范围有限 |

## 11. Claim–evidence map

| Claim | Evidence | Boundary |
|---|---|---|
| pretrained LLM 可转换为 recurrent model | 三个 family 的 continued-training runs | 不是无训练、无损转换 |
| pretrained init 节省大量 tokens | random vs pretrained trend | 950B 是 extrapolation |
| math 性能可随 recurrence 改善 | GSM8K/MATH depth sweeps | 不代表 knowledge/chat/safety 同样改善 |
| 两阶段 healing 改善能力保持 | controlled data curriculum | data 来源与许可会影响复现 |
| retrofit 在给定 training FLOPs 下优于 vanilla post-training | 论文 compute comparison | 不等于所有 wall-clock/hardware 设置都更快 |

## 12. 局限与工程风险

1. 模型约 1B，尚无 7B/70B 或大 MoE retrofit 证据。
2. 额外训练约 50B，仍是显著成本；“无需重新预训练”不等于“轻量 LoRA”。
3. data recipe 偏数学，通用 chat/tool/safety retention 不充分。
4. repo 固定的 Transformers/KV-cache 行为较旧，升级依赖可能改变 generation correctness。
5. 超过训练 recurrence 的稳定性仍未解决。
6. 部分 Nemotron data 的访问与 license 会限制完整复现。

## 13. 在通用 LLM 流程中的位置

```text
fully-open base checkpoint
  → model surgery
  → general-data healing
  → reasoning/math continued pretraining
  → recurrent base checkpoint
  → 尚待统一 SFT / DPO / RLVR
```

这条路线最适合本项目，因为它能把 architecture experiment 与昂贵的从头 pretraining 分开。理想平台是 OLMo 这样的 fully-open base：可同时获得原 data recipe、中间 checkpoint 和 vanilla post-training control。

## 14. 推荐复现方案

### Phase A：surgery correctness

- 在 1B–3B fully-open base 上实现 $(p,r,c)$；
- 验证原模型与 retained layers 的数值；
- 测 conversion 后未训练 perplexity、activation norms、KV cache；
- 用 100M–1B tokens 做 small healing，比较 layer selection。

### Phase B：matched continued training

- vanilla base + $D$ tokens continued training；
- retrofit + $D$ tokens；
- random-init recurrent + $D$ tokens；
- 至少同时报告 token-matched、FLOP-matched、wall-clock-matched。

### Phase C：post-training retention

对 vanilla/retrofit 同时执行相同 SFT → DPO → RLVR。每一 stage 都评测 base knowledge、perplexity、instruction following、math/code、chat/tool、安全和 depth curve。

## 15. 可证伪扩展

- **层选择假设**：首尾层比均匀抽层更适合 recurrence。枚举相同 retained parameter count 的层组合；若差异消失则否定。
- **healing 假设**：general data phase 的收益来自结构 repair 而非单纯更多 tokens。用等 tokens 的 mixed-only 与 general-only controls 区分。
- **optimizer 假设**：Muon 的优势来自稳定 recurrent Jacobian。测谱范数/梯度方向和 AdamW 调参后结果；若 AdamW 可追平，则不是结构性结论。

## 16. 自测题

1. 为什么删除中间层后仍可能利用晚层参数？
2. 950B-token 数字为什么必须标成 extrapolation？
3. general healing 与 math specialization 分两阶段有什么因果意义？
4. retrofit 与 vanilla 的公平比较至少要固定哪三种预算？
5. 为什么旧 Transformers KV-cache 依赖值得单列为复现风险？

## 17. 一句话定位

Retrofitted Recurrence 把 looped LLM 从“必须重做 pretraining”的架构研究，变成可在公开 base checkpoint 上进行的 continued-training 实验；其下一道关卡是更大规模和统一 post-training 下的通用能力保持。
