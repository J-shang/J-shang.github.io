---
title: "Moonshot Kimi 与 Moonlight 的数据实践"
description: "正交核算长上下文处理、rephrasing、optimizer/data 协同、agent provenance 与多模态训练时序。"
topic: "pretraining-data"
section: "china-cases"
slug: "moonshot-kimi-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 104
readtime: 38
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/moonshot-kimi.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/china/moonshot-kimi.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:3611f187799774a68a68b692f6996570204e47f6ee8bea3e502104ffa150a399"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family/historical trace`
> 研究锚点：Kimi k1.5、Moonlight、Kimi K2、Kimi K2.5；Kimi-VL 只作多模态实现旁证。

## 定位与 motivating problem

Moonshot 的公开材料把三个通常分散的问题连在一起：数据配方、optimizer token efficiency 与 agentic data。Kimi k1.5 给出语言/多模态清洗、cooldown、长上下文激活和 RL prompt curation；Moonlight 用同架构、同数据预算的 Muon/AdamW 对照研究 optimizer；K2 则把 MuonClip、15.5T corpus、knowledge/math rephrasing 与 agent trajectory synthesis 合并到一个大模型；K2.5 又研究固定 vision-text token budget 下的 injection timing、ratio 和 multimodal RL。

这里最容易出现四种错误：

1. 把 optimizer 的 compute/token efficiency 写成“数据质量更高”；
2. 把 base、cooldown、long-context activation、SFT、RL rollouts 和 tool execution 都加成一个 token 总数；
3. 把同一 source 的重复、rephrasing 与新增 unique source 混为一谈；
4. 把 synthetic tool spec、task、trajectory、environment feedback 与 verifier label 都记成同一类 synthetic sample。

本笔记的核心问题是：**在 optimizer、data transformation、context curriculum 和交互式训练同时扩展时，怎样保留能够区分各机制的统计单位与实验边界？**

## 代际与阶段表

| 模型/阶段 | 阶段 | 数据与操作 | 公开规模/长度 | 关键边界 | 披露 |
|---|---|---|---|---|---|
| Kimi k1.5 base | `P0` | English、Chinese、code、math/reasoning、knowledge；caption/interleaving/OCR/knowledge/QA | token 总量 `unknown`；vision-text 最终提高到 30% | source snapshot、总 mixture、loss tokens 未公开 | `D2` |
| Kimi k1.5 cooldown | `P2` | 高保真语言/视觉子集；math/knowledge/code synthetic QA + rejection/validation | token 数 `unknown` | synthetic QA 是 cooldown augmentation，不是整个 base corpus | `D2` |
| Kimi k1.5 long activation | `P3` | 40% full-attention + 60% partial-attention；natural long + synthetic long QA/summary | 4K→32K→131,072；token 数 `unknown` | full/partial attention 是 attention/data treatment，不等于 40/60 unique source | `D2` |
| Kimi k1.5 SFT | `A1` | 约 1M text + 1M text-vision examples | 32K 1 epoch → 128K 1 epoch | examples、packed sequences、loss tokens 不可互换 | `D2` |
| Kimi k1.5 RL | `A2` | STEM、code、general reasoning；text/image QA；verifiable prompts | rollout/trajectory/loss tokens `unknown`；context 到 128K | pass@10 difficulty、guess-without-CoT hack check、curriculum/prioritized sampling | `D2` |
| Moonlight | `P0/P2` | source/mix 未在 optimizer report 内完整展开；末 500B 使用最高质量 math/code/reasoning | 0–33B warmup；33B–5.2T cosine；5.2–5.7T cooldown | 5.7T 是 sampled exposure；source lineage 不从 k1.5 默认继承 | `D2/D3` |
| Kimi K2 base | `P0/P2/P3` | web text、code、math、knowledge；knowledge/math rephrasing | 总 15.5T；4K；末段 400B@4K + 60B@32K，YaRN 到 128K | 400B+60B 看似包含在 15.5T 末段，报告未给逐段闭合表 | `D2/D3` |
| Kimi K2 SFT | `A1` | human annotation、prompt engineering、expert candidate、judge；agent synthesis | tens of thousands agent examples；3K+ real MCP tools、20K+ synthetic tools | tool、agent、task、trajectory、accepted example 分列 | `D2/D3` |
| Kimi K2 RL | `A2/A3` | verifiable gym、self-critique rubric、real/simulated environments、PTX replay | prompt/rollout/trajectory/loss tokens `unknown` | task-specific budget、partial rollout、temperature decay改变在线分布 | `D2/D3` |
| Kimi K2.5 ViT/joint | multimodal `P0/P1` | image/video alt text、synthetic captions、grounding、OCR；joint text/vision | 表列 ViT 1T、joint 15T@4K | 简介“约15T”与分阶段数不可直接相加；见冲突节 | `D2/D3` |
| Kimi K2.5 long/mid | `P2/P3` | high-quality text/multimodal、long text/video、reasoning、long-CoT | 表列 500B→200B；32K→262,144 | 箭头表示两个阶段/长度，不应解释为减少 300B | `D2/D3` |
| Kimi K2.5 SFT/RL/PARL | `A1/A2/A3` | zero-vision SFT、outcome visual RL、joint RL、Agent Swarm synthetic prompts | sample/trajectory/loss tokens `unknown` | orchestrator 与 frozen subagents 分离训练；并行 wall time 不等于总 token compute | `D2/D3` |

## Assumption ledger

| 项 | 本笔记的处理 |
|---|---|
| 研究对象 | k1.5 data pipeline、Moonlight optimizer experiment、K2 text/agent、K2.5 multimodal 分开 |
| token 单位 | 报告中的 training/processed tokens 记 sampled exposure；unique、non-padding、loss tokens 未披露时均为 `unknown` |
| image/video token | 不假设与 text token 具有相同信息密度；K2.5 的 mixed token 只在其 tokenizer/packing 内成立 |
| epoch | K2 rephrasing experiment 的 epoch 是对选定 knowledge subset 的 exposure，不代表全 corpus epoch |
| optimizer relation | Muon/MuonClip 与数据配方是相互作用关系；只有固定架构、数据与预算的对照支持 optimizer 局部结论 |
| long context | maximum sequence、full/partial attention、input context、output budget 分开 |
| agent unit | tool spec、agent prompt、task、rubric、trajectory、turn、tool call、environment state、accepted sample 分开 |
| cutoff/rights | 全家族统一 cutoff 与 source-level rights 基本 `unknown`；公开 source 不等于统一许可 |
| lineage | K2.5 明确建立在 K2 checkpoint；Moonlight、k1.5 与 K2 的具体 corpus/checkpoint 共用关系不自动补齐 |
| 省略效应 | optimizer、MoE sparsity、attention、data、sequence length、post-training 与 inference budget 会同时变化 |

## 统一数据字段表

| 字段 | k1.5 | Moonlight / K2 | K2.5 | 置信 |
|---|---|---|---|---|
| source | public web/PDF、code、academic/educational、open/in-house multimodal；完整清单未知 | Moonlight source 未展开；K2 四域 web/code/math/knowledge | K2 四域 + caption/interleaving/OCR/knowledge/perception/video/agent | 已列类别 `verified`；完整 manifest `open` |
| rights | source-level rights/consent `unknown` | `unknown` | `unknown` | `open` |
| cutoff | `unknown` | `unknown` | `unknown` | `open` |
| scale | base token `unknown`；SFT 约1M text+1M vision-text | Moonlight 5.7T；K2 15.5T | 简介约15T；阶段表 1T、15T、500B→200B | 原文数字 `verified`；闭合关系 `open` |
| parse/OCR | math-specific OCR/cleaner；academic layout/OCR quality label | K2 沿用 k1.5 方法并增加 rephrasing | academic layout parser；native-resolution image/video | 机制 `verified`；error rate `open` |
| dedup | rule、embedding near-dup、multimodal dedup | K2称 rigorous validation，阈值未给 | vision corpus称 filtering/dedup/QC，阈值未给 | 存在性 `verified`；协议 `open` |
| quality | rule + FastText + embedding + LLM score；quality-dependent sampling | K2 domain correctness/quality experiment；fidelity check | strict synthetic-caption limits；multi-stage verification；visual task rewards | 机制 `verified`；calibration `open` |
| mixture | language五域；多模态五类；vision-text末段30%；long 40/60 attention treatment | K2四域比例未知；末段高质量与长上下文 | early moderate vision ratio；code upweight；七类 vision | 局部 `verified`；最终比例多为 `open` |
| synthetic | cooldown QA、interleaving、vision RL、test cases | knowledge/math rephrasing；agent tools/tasks/trajectories | captions、structured problems、image-code、RFT trajectories、swarm prompts | roles `verified`；provenance/version `open` |
| validation | per-source isolated checks；small-model mixture ablation；RM spot check | Muon/AdamW iso-data；rephrasing/epoch；scaling/validation loss | fixed-budget injection/ratio ablation；vision-RL before/after；multi-run eval | 已报告实验 `verified`；外推 `open` |
| artifacts | k1.5 report；无 corpus/checkpoint | Moonlight optimizer code/checkpoints；K2 weights/report/code | weights/report/deployment code；无 corpus/order/logs | `D2/D3` |

## k1.5：先把数据管线与训练阶段拆开

### 多维质量不是一个通用分数

`[来源事实 | verified]` k1.5 对中英文文本组合 rule、FastText、embedding similarity 与 LLM quality assessment；最终按组合分数动态上采高质量、下采低质量。代码数据对 markup language 下采、对 32 种主要语言上采。数学数据因通用 parser/OCR 的 false negative 较高，使用专用 OCR，再用 FastText 与 fine-tuned LM 两阶段清洗。知识数据另标 OCR quality、educational value 与 document type。

`[推导结论 | supported]` 这些分数的研究对象不同：OCR artifact、语言流畅性、near-duplicate、教育价值和 programming-language balance 不能归约为同一“质量”。实施时至少要保存每个子分数、统计单位、threshold 和 sampling weight。

### 三阶段数据关系

```text
language foundation → gradual vision-language integration (vision-text up to 30%)
  --followed-by-->
high-quality cooldown + validated synthetic QA
  --followed-by-->
long-context activation: 4K → 32K → 128K
```

`[来源事实 | verified]` long-context 阶段使用 40% full-attention data 与 60% partial-attention data。前者来自高质量 natural long data 与 synthetic long QA/summary，后者来自 cooldown data 的 uniform sample。

`[综合判断 | supported]` 40/60 不是 natural/synthetic mixture，也不是 unique/sample token 比例；它首先是 attention treatment 与 source construction 的组合字段。复现实验需要同时记录 sample origin、sequence length、attention mask 和 loss mask。

## k1.5 RL：prompt set 是在线训练分布的入口

### coverage、difficulty 与 evaluability

`[来源事实 | verified]` prompt 涵盖 STEM、competition、code、general reasoning 和 text/image QA；tagging 用于 domain balance。每题由 SFT model 高温生成 10 次，以 pass rate 作为难度 proxy。multiple-choice、true/false、proof 等易产生 false-positive verification 的问题被排除；一般 QA 还做无 CoT 猜答案测试，8 次内命中则移除。

最小 operational anchor：

$$
\widehat d_i=1-\frac{1}{10}\sum_{j=1}^{10}\mathbf{1}[\operatorname{verify}(y_{ij})=1]
$$

其中 $i$ 是 prompt，$j$ 是高温 rollout，$\widehat d_i$ 是 model-relative difficulty，不是题目的固有难度。其实现后果是：SFT checkpoint、temperature、verifier 版本变化都会重排 curriculum。

### curriculum 与 prioritized sampling

`[来源事实 | verified]` 训练先从较易问题开始，再转向更难问题；同时跟踪每题在线 success rate $s_i$，按 $1-s_i$ 提高失败题的采样概率。

$$
p_i=\frac{1-s_i}{\sum_j(1-s_j)}
$$

这是 implementation relation，不是固定数据集 mixture。policy 改变会更新 $s_i$，进而改变下轮训练分布。

### verifier data 的单位

`[来源事实 | verified]` math classic RM 与 CoT RM 各使用约 800K 标注点；人工 spot check 报告约 84.4 与 98.5 accuracy。coding test generation 的 1,000 个 contest problem 样本中，614 个无需 special judge，463 个生成器产生至少 40 个有效测试，最终 323 题进入训练。

`[综合判断 | supported]` RM label、problem、generated test、ground-truth submission、rollout 与 accepted trajectory 是不同层；只记录“800K + 323”没有可解释性。

## Moonlight：optimizer/data 协同的可辨识边界

### 训练 schedule

$$
T_{\text{Moonlight}}=33\text{B}+(5.2\text{T}-33\text{B})+0.5\text{T}=5.7\text{T}
$$

`[来源事实 | verified]` 0–33B 为 warmup；33B–5.2T 为 cosine decay；5.2–5.7T cooldown 使用最高质量 math、code、reasoning。最大上下文 8K。

### 能说什么，不能说什么

`[来源事实 | verified]` scaling-law proxy 在 399M–1.5B dense models 上比较 Muon 与调参后的 AdamW；报告估计 Muon 达到相同 compute-optimal loss 约需 AdamW 52% FLOPs。Moonlight 与 Moonlight-A 在相同架构和约 1.2T exposure 下只替换 optimizer，另有 validation loss 与 benchmark 对照。

`[推导结论 | supported]` 这支持“在报告设置内 Muon 提升 token/compute efficiency”，不支持“Muon 等价于更高质量数据”。Moonlight final 5.7T 还加入 cooldown；final benchmark 与其他模型的差异不能识别 optimizer、数据或架构贡献。

关键 invariant：

```text
optimizer ablation valid only if
  model architecture + tokenizer + data order/mix + token budget
  + batch/LR search protocol + evaluation checkpoint
are controlled
```

## K2 pretraining：rephrasing 不是无条件的数据扩容

### 15.5T 与末段 schedule

`[来源事实 | verified]` K2 的 15.5T corpus 分为 Web Text、Code、Mathematics、Knowledge。训练先以 4K sequence：10T constant LR，随后 5.5T cosine decay。报告又称末期包含 400B@4K annealing 与 60B@32K long-context activation，并用 YaRN 扩展到 128K。

`[综合判断 | supported]` 最自然的阅读是 460B 属于 15.5T 的末段，而不是额外训练；但报告没有逐段闭合表，因此该关系标 `ambiguous`。不能写成 15.96T。

### knowledge rephrasing

`[来源事实 | verified]` 长文档分 chunk 顺序改写后拼回，并做 source/rewrite fidelity check。局部实验比较：raw data 重复 10 epochs、改写一次再重复 10 epochs、改写 10 次各训练 1 epoch；SimpleQA accuracy 依次为 23.76、27.39、28.94。大规模 corpus 每份最多改写两次。

该实验区分的是：

```text
same semantic source + repeated surface form
vs.
same semantic source + diverse rephrased surface forms
```

`[综合判断 | supported]` 改写增加 sampled tokens 和表面多样性，但不增加 source-level independent evidence。factual error 可能在同一 source lineage 内相关，故需要 source→rewrite provenance、fidelity false-negative audit 与 source-grouped validation。

### mathematics rephrasing

`[来源事实 | verified]` 高质量数学材料被改写为 learning-note style，并把其他语言的数学材料翻译为英文。报告明确把 hallucination、toxicity 和跨 domain scaling 作为未解决问题。

`[推导结论 | supported]` translation 与 stylistic rewrite 会同时改变语言比例、token fertility、pedagogical format 与重复结构；不能只把它当作“math tokens 增加”。

## K2 agentic data：provenance graph 而非 synthetic 布尔值

### 三阶段 factory

```text
real MCP specs (3K+) + synthetic specs (20K+)
  --sampled-into-->
tool sets → agent system prompts → tasks + explicit rubrics
  --executed-by-->
user simulator + tool/world simulator or real sandbox
  --filtered-by-->
rubric LLM judge / executable tests → accepted trajectories
```

`[来源事实 | verified]` pipeline 生成 thousands of agents 和 tens of thousands training examples；simulation 维护 state 并注入 controlled stochasticity，coding/SWE 用 real execution sandbox 和 test pass rate ground truth。

`[综合判断 | supported]` simulator 扩 coverage，sandbox 提升部分任务的 execution fidelity；二者不是同质 environment。每条 trajectory 至少要记录 tool origin、simulator/sandbox version、task/rubric generator、policy checkpoint、judge、accept/reject reason 和随机种子。

### RL 数据分布

`[来源事实 | verified]` verifiable gym 包含 math/STEM/logic、complex instruction、faithfulness、coding/SWE。difficulty 仍由 SFT pass@k 筛到中等区间；instruction 同时使用 deterministic checker、LLM judge 和 hack check。open-ended tasks 用 self-critique rubric；verifiable on-policy rollouts 又持续更新 critic。RL objective 还混入 hand-selected high-quality PTX samples 防遗忘。

`[推导结论 | supported]` policy、critic 与 sampler 构成闭环：critic 更新改变 reward，reward 改变 policy，policy 改变 rollout distribution。静态“RL dataset size”不足以描述训练数据。

### long-horizon rollout

`[来源事实 | verified]` 长环境等待通过大量 concurrent rollout 摊薄；long-tail unfinished trajectory 用 partial rollout 跨 iteration 暂停/恢复。每类任务另设 max output token budget，超限截断并惩罚。

`[综合判断 | supported]` partial rollout 会改变“一个 iteration 看到多少完整任务”的分布；budget 则改变答案长度与成功率。诊断应同时报 completed-task rate、carried-over trajectories、environment wait、tokens/turn 与 success by length。

## K2.5：固定 token budget 下研究 multimodal timing

### early/mid/late fusion ablation

`[来源事实 | verified]` 在固定总 vision-text token budget 下，报告比较：early 10:90、mid 20:80、late 50:50。为了严格满足比例，mid/late 先训练计算得到的 text-only tokens，再注入 vision。early/low-ratio 在 vision、text、code 多项 proxy 上更优，并避免首次注入 vision 时 text metric 的 dip-and-recover。

`[综合判断 | supported]` 该实验同时改变 injection timing 与 vision ratio，不能分别识别两者的独立效应；它支持的是组合策略排序。更强的 factorial design 应在每个 timing 下比较多个 ratio。

### 15T 口径冲突

`[来源事实 | verified]` 简介称从 K2-Base continued pretraining 约 15T mixed visual/text tokens。阶段表列：ViT training 1T@4K、joint pretraining 15T@4K、joint long-context/mid-training 500B→200B@32K→262K；正文又称 joint stage 从 near-end K2 checkpoint 训练 additional 15T。

`[未知 | open]` 公开文字不足以判断 1T ViT target tokens 和 500B/200B 是否包含在 rounded 15T、是否采用相同 tokenizer/计量、或是否与 LLM sampled tokens可加。故本笔记不计算 16.7T。

### text/vision data

`[来源事实 | verified]` text 延续 K2 四域并提高 repo、issue、review、commit history 与 PDF/web code document 权重。vision 七类为 caption、interleaving、OCR、knowledge、perception、video、agent；synthetic caption 比例受限；另含 academic problem reformulation、rendered image-code pairs、GUI screenshot/action trajectories、human demonstrations、grounding/segmentation。

`[来源事实 | verified]` data loader 支持 dynamic shuffle/blend/tokenization/loss masking/packing；保存 random seed 与 worker state，使中断恢复后的 data sequence 与不中断训练一致。

这提供一个难得的 implementation invariant：

```text
resume(checkpoint, loader_state, worker_state, seed)
  must reproduce
next_sample_ids + augmentation parameters + packing/loss masks
```

### zero-vision SFT 与 joint RL

`[来源事实 | verified]` zero-vision SFT 只用 text SFT，通过 IPython programmatic operations 激活 visual tool use；初步实验中 text-vision SFT 更差，报告推测是高质量 vision SFT 不足。随后用必须看图才能解的 grounding/counting、chart/document、vision-critical STEM 做 outcome RL，并把成功轨迹做 RFT。joint RL 按能力而非模态组织 expert。

`[综合判断 | plausible]` “text-only SFT 激活 vision”可能依赖此前 15T joint pretraining 的跨模态表示，因此不是 text-only data 对任意 VLM 的普遍效果。需要从相同 base 比较 zero-vision、matched high-quality vision SFT 与混合 SFT。

### Agent Swarm 的 compute accounting

`[来源事实 | verified]` PARL 训练 orchestrator 动态创建和调度 frozen subagents；synthetic prompts 不直接要求并行，而设计成并行分解有优势。Agent Swarm 可降低 wall-clock latency，但 subagent 数和总生成 tokens 可能增加。

`[推导结论 | verified]` 并行只改变 critical path 时：

$$
T_{\text{wall}}\approx T_{\text{orchestrator}}+\max_k T_{\text{subagent},k}
$$

而总生成 compute 更接近：

$$
C_{\text{generation}}\propto N_{\text{orchestrator tokens}}+\sum_k N_{\text{subagent},k}
$$

因此“4.5× faster”不能解释为 token efficiency 或 training-data efficiency 提高。

## Validation、ablation 与区分性检查

### 有辨识力的公开锚点

| 锚点 | 控制 | 变化 | 能支持 | 仍不能支持 |
|---|---|---|---|---|
| Moonlight Muon vs AdamW | model、约1.2T data budget | optimizer | 局部 optimizer efficiency | K2 final gain 全归 optimizer |
| K2 rephrase vs repeat | early checkpoint、source subset | surface diversification/exposure | rephrase 优于简单重复 | 全 corpus factual reliability |
| K2 sparsity scaling | activated experts/FLOPs | total experts | local validation-loss relation | data-independent universal optimum |
| K2.5 fusion experiment | total mixed-token budget | timing+ratio组合 | tested组合排序 | timing、ratio 独立因果 |
| K2.5 vision RL before/after | checkpoint链 | visual RL | within-run cross-modal association | 无 matched text-RL 时的唯一机制 |
| k1.5 curriculum | RL setup | sampling schedule | within-report training efficiency | 固定 dataset 的普遍最优策略 |

### 应补的区分性检查

1. rephrasing：按 source group 切 validation，比较重复、改写、检索新 source，并人工审 fidelity false negatives；
2. optimizer/data：固定 sample IDs/order，同时记录 loss、update RMS、max logits 与 per-domain held-out；
3. long context：固定 token budget，分别改变真实长文比例、synthetic QA、packing 和 attention mask；
4. agent synthesis：real sandbox 与 simulator 分层报告 acceptance、transfer、tool error 和 verifier disagreement；
5. multimodal fusion：做 timing × ratio factorial grid，而不是三个耦合点；
6. zero-vision SFT：与等质量、等 token 的 vision SFT 和 mixed SFT 对照；
7. swarm：同时报告 wall time、total generated tokens、tool calls、subagent count、success 和 cost。

### 缺失的 validation contract

所有家族仍缺少统一的 pretraining held-out manifest：source snapshot、time cutoff、dedup/decontam boundary、tokenizer、domain token 数、micro/macro/per-domain loss 与 checkpoint selection rule。benchmark 表不能代替该 contract。

## 表面冲突与边界

### K2 15.5T 与 400B+60B

报告把 10T+5.5T 写成总 15.5T，并在“towards the end”描述 400B+60B。首个差异是 nested schedule 与 additive schedule；在没有 step/token ledger 前保留 `ambiguous`。

### K2 high-quality token scarcity 与 rephrasing

rephrasing 增加表面形式，不增加 independent source facts。两者不矛盾；它优化的是每个 source 的 exposure utility，但可能复制 source error。

### k1.5 synthetic caption 限制与 cooldown synthetic 收益

前者针对 caption hallucination，后者针对经过 rejection/validation 的 math/knowledge/code QA。数据对象与 verifier 不同，不能归纳为“synthetic 好/坏”。

### K2.5 约15T 与阶段表

简介总数、ViT target token、LLM mixed token 和 long/mid token 可能不是同一分母。首个差异是 model update scope 与 tokenizer/统计单位；故不做加法。

### early fusion 与 k1.5 gradual fusion

k1.5 从语言 foundation 逐步提高 vision 到30%；K2.5 在已有 K2 text checkpoint 上比较 continued-training 内的 early/mid/late injection。二者的“early”起点不同，属于不同 initial condition，不能直接判为方法反转。

### zero-vision SFT 与 native multimodal

zero-vision 只描述 post-training input；其能力依赖此前 joint multimodal pretraining。不是“没有视觉训练数据”。

## 可迁移与不可外推

### 可迁移

- `[综合判断 | supported]` optimizer 数据效率实验必须固定 sample/order/budget，并记录 optimizer-specific stability metrics。
- `[综合判断 | supported]` rephrased data 要保留 source lineage；surface diversity 与 independent evidence 分列。
- `[综合判断 | supported]` RL prompt sampling 是随 policy 更新的数据分布，需保存每轮 difficulty/success/selection probability。
- `[综合判断 | supported]` agentic data 应用 provenance graph 表达 tool→agent→task→trajectory→verifier。
- `[综合判断 | supported]` multimodal ablation 要固定 mixed-token budget，并分离 timing、ratio 与 update scope。
- `[综合判断 | supported]` 长上下文实验同时记录 length、source、packing、attention mask、loss mask 和真实/合成比例。
- `[综合判断 | supported]` 并行 agent 的 wall time、generated tokens 与成功率是三个指标。

### 不可外推

- `[未知 | open]` 不能从 Moonlight/K2 benchmark 反推 source mix、cutoff、rights 或 contamination。
- `[综合判断 | open]` 不能把 Muon 的 52% proxy 直接外推为 K2 对任意 optimizer、模型尺度和数据分布的固定倍数。
- `[综合判断 | open]` K2 knowledge rephrasing 的局部结果不能证明任意 domain 或大规模多次改写都安全。
- `[综合判断 | open]` K2.5 early/low-ratio 组合优于两个对照，不等于单独证明“越早越好”或“vision ratio 越低越好”。
- `[综合判断 | open]` zero-vision SFT 的结果不能外推到没有大规模 joint pretraining 的模型。
- `[综合判断 | open]` synthetic simulator 上的 accept rate 不能替代 real-environment transfer。

## 明确未知项

- `[未知 | open]` k1.5、Moonlight、K2、K2.5 完整 source list、snapshot、rights、consent 与统一 cutoff。
- `[未知 | open]` unique documents/tokens、exact repeats、padding/packing waste、loss tokens。
- `[未知 | open]` Moonlight 与 k1.5/K2 的确切 corpus、tokenizer 和 checkpoint lineage。
- `[未知 | open]` K2 四域最终比例、rephrased token 占比、source-grouped fidelity error。
- `[未知 | open]` K2 的 460B 是否完全包含在 5.5T decay，以及 annealing/long data mixture。
- `[未知 | open]` K2 agent tasks、trajectories、acceptance、simulator/sandbox 比例与 verifier agreement。
- `[未知 | open]` K2.5 约15T与 1T/15T/500B/200B 的包含关系和各阶段 update scope 下的 loss-token 数。
- `[未知 | open]` K2.5 最终 vision:text ratio、各 vision category exposure、synthetic-caption上限。
- `[未知 | open]` benchmark contamination/decontamination 的 n-gram/semantic/temporal protocol。
- `[未知 | open]` 所有模型的 frozen per-domain pretraining validation contract。

## 掌握标准与推理型自测

掌握本案例后，应能：

1. 画出 k1.5 base/cooldown/long/SFT/RL 与 K2 base/agent SFT/RL 的阶段边界；
2. 解释为什么 Muon efficiency 不是 data-quality score；
3. 区分 repeat、rephrase、translation 与 independent new source；
4. 为 agent trajectory 写出包含 simulator、policy、rubric、judge 的 provenance record；
5. 解释 K2 15.5T 和 K2.5 约15T为何不能按可见数字直接相加；
6. 指出 K2.5 fusion ablation 中 timing 与 ratio 的耦合；
7. 同时报出 Agent Swarm 的 latency、total tokens 与 task success。

自测：

1. 某知识文档生成 10 个改写并各训练一次，unique source tokens 与 sampled tokens 如何记？
2. Muon run 在相同 tokens 下 loss 更低，为什么还不能说其“看到的数据更高质量”？
3. pass@10 从 policy A 改为 policy B 后，difficulty label 是否仍可直接复用？
4. simulator trajectory 通过 LLM rubric，但 real sandbox 失败，应修改哪个质量字段？
5. 15T mixed tokens 中 image token 和 text token 是否能按信息量等价解释？为什么？
6. swarm wall time 减半、generated tokens 翻倍时，能否称 token efficiency 提高？

## 来源与阅读顺序

1. [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)：先读§2.1 prompt curation、§2.3 sampling/verifier、§2.5 stages，再读Appendix B的language/multimodal pipeline、30% vision与40/60 long-context treatment。
2. [Muon is Scalable for LLM Training / Moonlight](https://arxiv.org/abs/2502.16982)：读§3.2 fixed-budget scaling law、§3.3的5.7T三阶段与1.2T Muon/AdamW对照；不要从optimizer report补写未披露source。
3. [Moonlight官方模型卡](https://huggingface.co/moonshotai/Moonlight-16B-A3B)：固定weights、config与checkpoint资产；不作为corpus manifest。
4. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)：读§2.2 rephrasing、§2.5的15.5T/460B schedule、§3.1 agent synthesis、§3.2 joint RL与PTX。
5. [Kimi K2官方仓库](https://github.com/MoonshotAI/Kimi-K2)：固定base/instruct weights、report、tool-call template与deployment code。
6. [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276)：读§2 fixed-budget fusion/zero-vision SFT、§4.3 stage table、Appendix B text/vision data与Appendix C deterministic loader。
7. [Kimi K2.5官方仓库](https://github.com/MoonshotAI/Kimi-K2.5)：固定报告版本、model card、weights与256K配置。
8. [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491)：只用于补充MoonViT、native resolution与128K多模态实现背景；不要把其训练规模回填到K2/K2.5。
