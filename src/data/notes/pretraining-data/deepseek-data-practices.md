---
title: "DeepSeek 家族的数据实践"
description: "追踪 DeepSeek LLM、V2/V3、R1 与后续代际中的去重、remix、continued training 和 reasoning data 边界。"
topic: "pretraining-data"
section: "china-cases"
slug: "deepseek-data-practices"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 100
readtime: 34
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/deepseek.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/china/deepseek.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:871768aeca6bb8d10b1a615008908d6d95e50090e41226009b1f537226fe5de5"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`method-family/historical trace` -->
> 主要模型/资料：DeepSeek LLM、DeepSeek-V2/V3、DeepSeek-R1、DeepSeek-V3.1/V3.2、DeepSeek-V4 Preview。

## 这篇案例要回答什么

DeepSeek 的公开材料适合研究两个常被混写的问题。第一，数据工程从早期的全局去重、过滤、remixing，怎样演化为 math/code/multilingual/long-document/agentic data 的分阶段配方；第二，同一个 base checkpoint 之后的 context extension、continued pretraining、reasoning RL、rejection sampling、distillation 分别使用什么数据单位。

本案例不能被概括成“2T → 8.1T → 14.8T → 33T”。这些数字属于不同模型、tokenizer 和阶段；R1 的 804,745 条监督样本、V3.2 的 85,000 个 agent prompts 和 V4 的 32T/33T pretraining tokens 更不是同一统计量。最有价值的公开信息是这些数字之间的边界，以及少数可检查的 data mechanism 与 ablation。

## 各代模型和 training stage

| 代际 | 阶段 | 已披露数据与规模 | 数据机制 / context | 披露 |
|---|---|---|---|---|
| DeepSeek LLM 7B/67B | `P0` | 2T 中英为主 tokens | dedup → filtering → remixing；91 个 Common Crawl dumps 跨 dump 去重 | `D2/D3` |
| DeepSeek LLM Chat | `A1/A2` | 约 1.5M 中英 instruction instances；其中 1.2M helpful、300K safety；随后 DPO | 两阶段 SFT、DPO；MC 数据另做 20M-instance ablation | `D2/D3` |
| DeepSeek-V2 | `P0` | 8.1T tokens；中文 token 约比英文多 12% | 延续 LLM pipeline，扩大中文、改进质量过滤并恢复误删数据；4K | `D2/D3` |
| DeepSeek-V2 | `P3` | 1,000 steps × 576 sequences × 32K，约 18.9B sampled sequence slots；论文未声明 non-padding/loss tokens | YaRN，4K → 128K；训练只用 32K sequence，NIAH 测到 128K | `D2/D3` |
| DeepSeek-V3 | `P0` | 14.8T tokens | 增加 math/code、扩展非中英语种、减少冗余；document packing；FIM rate 0.1；128K vocab | `D2/D3` |
| DeepSeek-V3 | `P3` | 2 × 1,000 steps；先 32K、后 128K | YaRN，4K → 32K → 128K | `D2/D3` |
| DeepSeek-V3 | `A1/A2` | 1.5M SFT instances，reasoning 数据由内部 R1 生成；随后 GRPO | rejection sampling、expert generation、human verification | `D2/D3` |
| DeepSeek-R1-Zero | `A2` | reasoning prompts 与 rollout；总 prompt/trajectory token 数 `unknown` | 从 V3-Base 直接做大规模 reasoning RL，无 cold-start SFT | `D2/D3` |
| DeepSeek-R1 | `A1/A2` | thousands cold-start → RL → 804,745 SFT samples → 第二轮 RL | 约 600K reasoning rejection samples + 约 200K non-reasoning；rule/generative rewards | `D2/D3` |
| R1-Distill | `A1` | 同一 800K teacher-generated samples | 对 Qwen/Llama students 只做 SFT，不做 RL | `D2/D3` |
| DeepSeek-V3.1 Base | `P1/P3` | 在 V3 上增加 840B continued-pretraining tokens | long-context extension；128K；tokenizer/chat template 发生变化 | `D2/D3` |
| DeepSeek-V3.2 | `P1/P3` | indexer warm-up 2.1B tokens；sparse stage 943.7B tokens | 从 V3.1-Terminus 继续训练；两阶段都沿用其 128K data distribution | `D2/D3` |
| DeepSeek-V3.2 | `A2/A3` | 1,827 environments、85K+ complex prompts | “hard to solve, easy to verify” agent synthesis，thinking-in-tool-use | `D2/D3` |
| DeepSeek-V4 Flash/Pro | `P0/P1` | Flash 32T、Pro 33T；math、code、web、long documents 等 | 过滤批量自动生成/模板内容；mid-training 加 agentic data；sample-level attention mask；1M context | `D2/D3` |
| DeepSeek-V4 | `A1/A2` | 各领域 specialist 的 SFT + GRPO，样本/token 数 `unknown` | math/code/agent/instruction specialists → multi-teacher on-policy distillation | `D2/D3` |

公开权重、报告和部分 inference/eval code 支持 `D3`；但 corpus、manifest、处理代码、训练顺序、完整日志和 source-level rights 未发布，因此数据管线本身不达到 `D4`。

## 阅读这些结论前先确认的前提

| 项 | 本笔记的处理 |
|---|---|
| 研究对象 | Base、Chat/Reasoner、Distill 分开；API 名称只在能映射到固定 release 时使用 |
| token 单位 | 报告中的 “trained on” 暂记 sampled exposure claim；`unique_tokens` 与 `loss_tokens` 均为 `unknown` |
| tokenizer | LLM/V2 为 100K BBPE；V3/V4 为 128K byte-level BPE；跨代 token 总量不做倍数换算 |
| 阶段 | V3 的 14.8T 是 `P0`；V3.1 的 840B 和 V3.2 的 945.8B 是后续 continued/context training；R1 数据属于 `A1/A2` |
| 分布 | “math/code 比例增加”等是方向性 remix；没有百分比时不补成权重 |
| cutoff | 公开报告未提供可覆盖各代的 source snapshot/data cutoff；全部保留 `unknown` |
| 近似 | V2 context-extension 的 18.9B 由 `steps × sequences × sequence_length` 计算，只是最大 sequence slots，不等于有效 loss tokens |
| 省略效应 | 代际提升同时受 MoE/attention、optimizer、precision、compute、tokenizer、data mix 与 post-training 影响 |

## 厂商公开了哪些 data fields

| 字段 | LLM / V2 | V3 / R1 / V3.1–3.2 | V4 | 置信 |
|---|---|---|---|---|
| source | 中英为主、多源 web 与高质量 source；完整列表 `unknown` | web/math/code/multilingual；R1 含公开教材与在线 repository、模型生成轨迹 | math、code、web、long documents，特别提 scientific papers/technical reports；agentic mid-training | 类别声明 `verified`；逐 source `open` |
| rights | source-level license/授权 `unknown`；模型权重按各 release license | 同左；R1 distilled weights 为 MIT | V4 weights MIT；训练 source rights `unknown` | 发布许可 `verified`；语料权利 `open` |
| token scale | LLM 2T；V2 8.1T | V3 14.8T；V3.1 +840B；V3.2 2.1B + 943.7B | Flash 32T、Pro 33T | 报告范围 `verified`；unique/loss tokens `open` |
| cutoff | `unknown` | `unknown` | `unknown` | `open` |
| mixture | LLM 中英为主；V2 中文 tokens 约比英文多 12% | V3 增加 math/code，扩展中英之外语言；具体百分比 `unknown` | multilingual、long-doc、math/code；比例 `unknown` | 方向/相对量 `verified`；完整 mix `open` |
| dedup | LLM 跨 91 个 CC dumps 全局去重；V2 声称延续处理阶段 | V3 只披露减少 redundancy、保留 diversity；精确算法/scope `unknown` | 大体沿用 V3 preprocessing；不能据此恢复 dedup 配置 | LLM `verified by report`；后代 `open` |
| filtering | linguistic + semantic document quality；V2 改进质量过滤、恢复误删、过滤 contentious content | V3 细节 `unknown`；R1 过滤语言混杂、长段落、code block，并保留正确轨迹 | 移除批量自动生成和模板化 web content | 存在性 `verified`；阈值/误杀率 `open` |
| remix/sampling | 提升 underrepresented domains；V2 增加中文 | V3 增加 math/code/multilingual；V3.1/3.2 long-context distribution 未公开 | 加强 multilingual/long documents；mid-training 加 agentic data | 方向性声明 `verified`；step-level exposure `open` |
| packing/mask | V2 `unknown` | V3 document packing，不做 cross-sample attention mask；FIM 0.1 | 跨 source packing，做 sample-level attention mask | 报告范围 `verified` |
| decontam | LLM 的 20M MC ablation 对 C-Eval validation 与 CMMLU test 去重；全面 protocol `unknown` | V3/R1 评测集污染协议不完整 | `unknown` | 局部事实 `verified`；全面无污染 `open` |
| synthetic/distill | alignment 中模型生成 preference candidates | V3 用 R1 生成 reasoning SFT；R1 rejection sampling；V3.2 agent synthesis | specialist + on-policy multi-teacher distillation | 阶段存在 `verified`；teacher/exposure 细节部分 `open` |
| validation | LLM 100M-token train-like validation、bits-per-byte scaling；新发布 held-out tasks；V2 bilingual benchmarks/NIAH | V3 short/long benchmark；R1 stage-wise table 与 reward ablation；V3.2 synthetic-agent ablation | unified internal base-model framework、LongBench-V2/1M tests；pretraining held-out contract 不完整 | 报告结果 `verified`；独立复现 `open` |
| artifacts | weights、papers、部分 code/config | weights、papers、R1/V3 inference code | weights、report、encoding/inference code | `D3` |

## 从全局去重到 remix：最清晰的 pipeline 锚点

### DeepSeek LLM 的三个阶段

早期报告把数据处理明确拆成：

```text
cross-dump deduplication
  -> linguistic + semantic filtering
  -> remix underrepresented domains
  -> BBPE tokenization
```

跨 Common Crawl dump 的 scope 是关键。报告给出的 removal rate 从单个 dump 的 22.2%，随 dump 数增加到 91 个时达到 89.8%。它支持的结论是：`dedup scope --empirically-associated-with--> measured duplicate removal`；不能据此推出 89.8% 被删内容全部低质量，也不能假定后代沿用相同阈值。

过滤被描述为同时看 linguistic 和 semantic signals；remixing 用来补 underrepresented domains。二者目标不同：过滤提高信息密度，remix 改变 exposure 分布。若把两者都汇总成“quality score”，会丢失可诊断性。

### V2：恢复误删也是质量工程

V2 在延续旧 pipeline 的同时明确说优化 cleaning 以恢复大量被误删的互联网数据，并改进 quality-based filter 以移除 non-beneficial data、保留 valuable data。这提示过滤审计至少要同时记录：

$$
\text{retention by source/domain},\quad
\text{rejection-reason distribution},\quad
\text{estimated false-positive rate}
$$

V2 还过滤与地区文化有关的 contentious content。论文随后报告它在 MMLU Humanity-Moral 上较弱，并由三名标注者检查 420 个场景，观察到标注者与 ground truth 一致率低。这个结果只能说明该 benchmark 与过滤目标之间存在 setting-specific association；不能证明某种“去偏过滤”一般会提高或降低模型价值观质量。

### V3/V4：remix 与 sequence construction 成为显式阶段

V3 相对 V2 增加 math/programming 比例、扩展中英以外语言，并优化冗余与 diversity。它把 FIM 以 0.1 概率在 document-level pre-packing 中应用。V3 的 document packing 保持文档完整性，但不做 cross-sample attention masking；V4 则明确加入 sample-level attention mask。这是一个 implementation change，不能从“都使用 packing”推断 attention leakage behavior 相同。

V4 还增加三类本轮以前没有被同样清楚写出的选择：过滤 batched auto-generated/templated web content、优先长学术文档、在 mid-training 加 agentic data。它们分别对应生成数据风险、长上下文有效长度和能力专项化，不能合并成一个静态 `P0 mixture`。

## Token accounting 与阶段边界

### V2 context extension 的可检查上界

V2 的长上下文训练为 1,000 steps、每 step 576 sequences、每 sequence 32K：

$$
N_{slots}=1000\times576\times32768
\approx 18.87\times10^9\text{ token slots}
$$

变量是 sampled sequence slots；padding、document boundary、mask 与 token dropping 是否进入 loss 未完全说明，因此 `loss_tokens` 仍为 `unknown`。而“训练 32K、评测到 128K”又表明 context capability 与训练 seqlen 不是同义字段。

### V3、V3.1、V3.2 不能简单相加成一个 corpus size

- V3：14.8T `P0` exposure claim。
- V3 long-context：两个 1,000-step `P3` phases，4K → 32K → 128K。
- V3.1：在 V3 上再训练 840B，官方把它称为 long-context continued pretraining。
- V3.2：从 V3.1-Terminus 出发，2.1B-token dense indexer warm-up 只更新 indexer；943.7B sparse stage 更新主模型和 indexer。

尤其 V3.2 的两个数字虽然同属 continued training，但参数更新集合不同。若研究“主模型新增 exposure”，不能把 2.1B warm-up 与 943.7B 当作同质 loss tokens。

### V4 的 32T 与 33T

V4 报告明确区分 Flash 32T 与 Pro 33T，因此模型卡摘要中的 “more than 32T for both models” 是 rounded family statement。这里记录每个固定模型的 sampled exposure claim，不将二者相加，也不把它们解释为 unique corpus size。

## Scaling：数据分布会改变拟合结果

DeepSeek LLM 用 non-embedding FLOPs/token $M$ 表示模型规模，并令数据规模 $D$ 为 tokens：

$$
C=MD,
\qquad
(M_{opt},D_{opt})=
\arg\min_{M,D:\,C=MD} L(M,D)
$$

在 8 个 compute budgets、每个约 10 种 model/data allocation 上，报告使用与训练分布相近、独立的 100M-token validation，以 bits-per-byte 作为 generalization metric，得到：

$$
M_{opt}=0.1715C^{0.5243},
\qquad
D_{opt}=5.8316C^{0.4757}
$$

更重要的是，在 early in-house、current in-house、OpenWebText2 三种数据上重新拟合时，$(a,b)$ 分别为 $(0.450,0.550)$、$(0.524,0.476)$、$(0.578,0.422)$。报告将更大的 $a$ 与内部认定的更高质量相联系。

这是一种 `empirical association`，不是从 scaling exponent 直接测量“质量”的 exact identity。数据规模、处理强度、distribution、tokenizer 和 validation match 都可能混杂。可区分实验应在相同 raw pool、token budget、tokenizer、optimizer 和 validation contract 下，只改变可追踪的 filtering/remix 版本，并检查多个 seeds 与 target-scale transfer。

## Reasoning 数据：R1 的多阶段 provenance

R1-Zero 与 R1 的差异首先是数据阶段，不是“都用了 RL”：

```text
V3-Base
  -> reasoning RL without SFT                         -> R1-Zero

V3-Base
  -> thousands of cold-start long-CoT SFT
  -> reasoning RL with rule rewards + language consistency
  -> rejection sampling / generative judging
  -> 804,745-example SFT
  -> mixed reasoning + preference RL                 -> R1
```

804,745 条 SFT 数据的报告分布为：

| domain | samples | avg rounds | avg tokens |
|---|---:|---:|---:|
| Math | 395,285 | 1.0 | 6,094.2 |
| Code | 211,129 | 1.1 | 7,435.7 |
| STEM | 10,124 | 1.0 | 4,928.8 |
| Logic | 10,395 | 1.0 | 2,739.0 |
| General | 177,812 | 1.1 | 1,419.8 |
| Total | 804,745 | 1.0 | 5,355.3 |

reasoning 部分从第一轮 RL checkpoint 采样多个回答，使用 rule-based 或 DeepSeek-V3 generative judge，过滤语言混杂、长段落与 code blocks，并只保留正确回答；约 600K。non-reasoning 约 200K。报告还明确指出大多数是 single-turn，这构成 multi-turn capability 的数据边界。

R1-Distill 使用同一 800K teacher-generated dataset 微调 Qwen/Llama students，并明确不加入 RL。因此：

```text
R1 teacher trajectories --used-by(A1 SFT)--> R1-Distill
```

而不是：

```text
R1-Distill --trained-by--> the same RL pipeline as R1
```

## V3.2 agent 数据：可验证合成的 operational anchor

V3.2 的 general-agent synthesis 生成 1,827 个 task-oriented environments，并形成 `<environment, tools, task, verifier>` tuples：

```text
task category + sandbox
  -> retrieve/generate database content
  -> synthesize task-specific tool functions
  -> propose task + solution + Python verifier
  -> verify solution
  -> iteratively increase difficulty / augment tools
  -> RL on retained solvable-but-nontrivial instances
```

这比“使用 synthetic agent data”更可操作：验证器与工具边界是数据 schema 的一部分。报告用 50 个随机合成任务检查 difficulty；V3.2-Exp pass@1 为 12%，被比较的 frontier models 最高 62%。另一个 ablation 从同一 SFT checkpoint 出发，只用 non-thinking synthetic agent tasks 做 RL，再与未做这一步的 checkpoint 比较，以测试跨任务 generalization。

局限是：任务数据库可从互联网检索内容，但 source list、rights、去重、时间 cutoff 与 verifier false-positive rate 未披露；“自动可验证”也不等于任务语义、工具实现和 reward 全部正确。

## V4：specialist 到统一模型的数据流

V4 的 post-training 把不同领域先分开，再汇合：

```text
base model
  -> domain-specific SFT
  -> domain-specific GRPO + reward signals
  -> math / code / agent / instruction specialists
  -> student on-policy trajectories
  -> multi-teacher reverse-KL distillation
  -> unified model
```

因此 V4 的 on-policy distillation 与 R1-Distill 的 offline teacher-output SFT 只是 method-family relation，不是同一实现。V4 学生从当前 policy 采样 trajectory，再根据任务选择相应 teacher 分布；公开报告没有给各 specialist 的数据量、teacher routing 频率、trajectory 数或 source rights。

## Validation 与 ablation 审计

### 有辨识力的证据

- `[来源事实 | verified]` DeepSeek LLM scaling 使用 100M-token、与训练分布相近的独立 validation，并同时变化 compute 与 model/data allocation。
- `[来源事实 | verified]` DeepSeek LLM 用近期 LeetCode、Hungarian Exam、IFEval 作 held-out-style 检查；这降低已知 benchmark overfitting 风险，但没有完整 time-based contamination audit。
- `[来源事实 | verified]` V2 在 data debiasing 后检查 Humanity-Moral，并用三名标注者复核 420 条样本，暴露 benchmark label disagreement。
- `[来源事实 | verified]` V3 context extension 同时报告 32K/128K phases 和 NIAH；但没有逐 domain 的短上下文回退表。
- `[来源事实 | verified]` R1 论文给出 stage-wise checkpoint comparison，并报告 language-consistency reward 会轻微损害 reasoning performance。
- `[来源事实 | verified]` V3.2 对 synthetic agent difficulty 与 RL transfer 分别做检查，而不是只报告最终 leaderboard。

### 仍然不能回答的问题

- `[未知 | open]` base pretraining validation 的 source、冻结时间、去重/decontam、每域样本数和 micro/macro 指标。
- `[未知 | open]` V2/V3/V4 data-change ablation 是否控制 tokenizer、架构、compute、random seed 与 checkpoint selection。
- `[未知 | open]` V4 的 “refined data quality” 与 architecture/Muon/scale 对提升的独立贡献。
- `[未知 | open]` R1/V3.2 的 rollout 总量、失败轨迹保留策略、reward false positive 与 benchmark leakage audit。

## 看似矛盾的说法怎样区分

### V3 14.8T、V3.1 +840B、V3.2 945.8B 是不是 16.586T？

- 三者按 lineage 可形成训练顺序，但 tokenizer/chat-template、参数更新范围、长上下文 distribution 与 checkpoint 身份发生变化。
- V3.2 的 2.1B warm-up 只训练 indexer；它与主模型 exposure 不是同一对象。
- 区分性检查：需要每阶段 sampler manifest、mask/loss contract、tokenizer revision 和参数-freeze config。
- 处理：矩阵分阶段记录，不给出简单累计 `loss_tokens`。`open`。

### V4 是 “more than 32T” 还是 32T/33T？

- model card 摘要用 family-level rounded claim；technical report 分别写 Flash 32T、Pro 33T。
- 两者兼容；第一个差异是摘要粒度，不是事实矛盾。
- 处理：具体模型使用 32T/33T，family summary 只作为索引。`verified`。

### R1 是“无 SFT 的纯 RL”还是“先 cold-start SFT”？

- 前者严格对应 R1-Zero；R1 本体明确使用 cold-start SFT、多轮 RL 与第二轮 SFT。
- 第一个差异是 checkpoint identity，而不是报告冲突。
- 处理：R1-Zero 与 R1 分行；任何“DeepSeek-R1 纯 RL”概括都必须注明是在说 R1-Zero。`verified`。

### DeepSeek LLM 的数据质量是否能由 scaling exponent 排名？

- 报告发现内部质量判断与 $a$ 增大一致，但 OpenWebText2 同时更小、处理更精细、分布不同。
- 第一个差异是 corpus distribution/scale/processing，不是只有抽象的 quality。
- 区分性检查：固定 raw pool 与 validation，仅替换可复现的 filter/remix，并多 seed 重拟合。
- 处理：保留为 `empirical association | supported`，不把 $a$ 当通用质量分数。

## 目前仍不知道什么

- `[未知 | open]` 各代完整 source list、source-level rights、raw/processed document 数、data cutoff 与 manifest。
- `[未知 | open]` 2T/8.1T/14.8T/32T/33T 分别对应 unique、sampled、non-padding/loss tokens 的精确 contract。
- `[未知 | open]` LLM 的 global dedup 算法、相似度阈值与 cross-source behavior；V3/V4 是否逐配置沿用。
- `[未知 | open]` V2/V3 quality filter 的模型、features、阈值、per-domain rejection 与 false-positive rate。
- `[未知 | open]` V3/V4 的完整静态与动态 mixture、重复 exposure 和训练顺序。
- `[未知 | open]` 全代际 benchmark decontamination 规则、版本、n-gram/semantic threshold 与 clean-subset 结果。
- `[未知 | open]` R1、V3.2、V4 的 prompt source、rollout tokens、rejected trajectories、reward error 与 verifier audit。
- `[未知 | open]` V4 1M-context pretraining 的长度直方图、真实/拼接比例与每长度 validation sample count。

## 哪些经验可以借鉴，哪些不能直接照搬

### 可迁移

- `[综合判断 | supported]` 去重必须同时记录粒度和 scope；跨 91 个 dumps 的 document dedup 与单 dump dedup 是不同干预。
- `[综合判断 | supported]` 过滤系统要记录恢复误删的机制与 false-positive proxy，而不只报告 rejection rate。
- `[综合判断 | supported]` scaling-law proxy 依赖数据分布；换 filter/remix 后应重新拟合，不能复用固定 exponent。
- `[综合判断 | supported]` reasoning 数据应保存 prompt、rollout、judge/verifier、rejection reason、SFT 与 RL 的 provenance DAG。
- `[综合判断 | supported]` agentic synthetic data 的 verifier 本身需要测试集和 false-positive/false-negative 审计。
- `[综合判断 | supported]` long-context accounting 要区分 context window、sequence slots、有效非 padding tokens 和进入 loss 的 tokens。

### 不可外推

- 91-dump 去重的 89.8% removal rate 不适用于其他 crawl、算法、阈值或保留策略。
- V2 的中文 token 相对量不能外推为 V3/V4 的中英比例。
- DeepSeek LLM 的 $(a,b)$ 不是通用 scaling 常数，也不是数据质量的单一测量仪。
- R1 的 800K examples 不是 pretraining tokens；R1-Distill 也没有复制 R1 的 RL 过程。
- V3.2 合成任务可自动验证，不证明它们无 source/right 问题、无 reward hacking 或代表真实用户分布。
- V4 的 1M context 与 32T/33T 不能证明 1M 长文档有高占比，也不能从 benchmark 反推 agentic mid-training mixture。

## 读完后应该能回答的问题

掌握本案例后应能：

1. 解释为什么 DeepSeek LLM 的 global dedup、V3 的 redundancy reduction 和 V4 的 templated-content filter 不是同一个字段。
2. 把 V3、V3.1、V3.2 的 token 数按 checkpoint、参数更新集合和训练阶段拆开。
3. 画出 R1 的 cold-start、RL、rejection sampling、804,745-example SFT 与第二轮 RL provenance。
4. 说明 scaling exponent 与 data quality 的关系为何是 empirical association，不是 exact identity。
5. 为 V3.2 agent verifier 设计最小的错误率与跨环境泛化审计。

自测：如果一个新 filter 使 91-dump removal rate 从 89.8% 升到 94%，同时 1B proxy 的 validation loss 下降，你还需要哪些 per-domain retention、contamination、memorization、target-scale 与多 seed 检查，才能决定是否用于 33T run？

## 来源与建议阅读位置

- [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism, arXiv:2401.02954v1](https://arxiv.org/abs/2401.02954)：为什么读：这是 global dedup、filter/remix、100M-token validation 和 data-dependent scaling 的主要来源；重点读 §2.1、§3.2–3.3、§4–5.3。
- [DeepSeek-V2 technical report, arXiv:2405.04434](https://arxiv.org/abs/2405.04434)：为什么读：固定 8.1T、中英 token 相对量、恢复误删、contentious-content filtering 与 4K→128K 阶段；重点读 §3.1 与 Appendix E。
- [DeepSeek-V3 technical report, arXiv:2412.19437](https://arxiv.org/abs/2412.19437)：为什么读：固定 14.8T、math/code/multilingual remix、packing/FIM/tokenizer、long-context phases 与 1.5M SFT；重点读 §4–5。
- [DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948)：为什么读：区分 R1-Zero 与 R1，并恢复 cold start、RL、rejection sampling、804,745-example SFT、reward ablation 和 distillation；重点读 §3、Appendix B.3、F。
- [DeepSeek-V3.1 official release, 2025-08-21](https://api-docs.deepseek.com/news/news250821/)：为什么读：固定 V3.1 Base 在 V3 上新增 840B continued-pretraining tokens、128K 和 tokenizer/chat-template 变化。
- [DeepSeek-V3.2 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)：为什么读：固定 2.1B/943.7B continued-training accounting、post-training compute、1,827 environments、85K+ prompts 与 synthetic-agent ablation；重点读 §2.1.1、§3.2、§4.3。
- [DeepSeek-V3.2 official release, 2025-12-01](https://api-docs.deepseek.com/news/news251201/)：为什么读：固定 release identity、thinking-in-tool-use 与官方对合成数据规模的摘要。
- [DeepSeek-V4 technical report, arXiv:2606.19348](https://arxiv.org/abs/2606.19348)：为什么读：固定 Flash 32T/Pro 33T、V4 data construction、sample-level mask、agentic mid-training 与 specialist→on-policy distillation；重点读 §4.1–4.2、§5.1。
- [DeepSeek-V4-Pro official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)：为什么读：固定开放权重、参数/context、license 和报告映射；不要只用模型卡摘要替代技术报告的分模型 token 数。
