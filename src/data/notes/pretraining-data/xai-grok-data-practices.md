---
title: "xAI Grok 的数据实践"
description: "分层处理训练快照、X 数据通道、实时检索、future-training controls 与 targeted mid-training。"
topic: "pretraining-data"
section: "international-cases"
slug: "xai-grok-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 125
readtime: 29
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/xai-grok.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/international/xai-grok.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:61295fd5a8684c4514ab706dd51ca8d5871672f27ad1f51d1aff4821150724aa"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family/historical trace`
> 辅助视角：`implementation-trace`，用于 Grok-1 开放权重与 X/Grok 数据控制
> 研究锚点：Grok-1、3、4/4.1/4.20/4.5；Grok Code Fast 1；xAI FAIF data disclosure

## 定位

xAI 的材料最容易把三种时间关系混在一起：

1. checkpoint 在 pretraining/mid-training/post-training 中已经看过的数据；
2. 产品在 inference 时通过 X Search/Web Search 取回的实时内容；
3. X 或 Grok 用户数据按政策、controls 与反馈路径进入未来模型训练的可能性。

这三者的技术关系是：

```text
training snapshot --updates--> model weights
X/Web search result --conditions-at-inference--> one response
eligible user/X data --may-feed--> future training or finetuning
```

实时检索不更新当前 checkpoint 权重；政策允许使用某类数据，也不证明某个已发布 checkpoint 实际包含该类数据。

## Motivating problem：Grok 的“实时 X 知识”不是训练集声明

Grok 从首次发布就强调通过 X 获得实时知识。若不拆层，容易出现：

- 把 X Search 的结果当成 pretraining mixture；
- 把 X 帮助页的“may share data for training”写成某一代 Grok 的已确认 corpus；
- 把家族级“trillions to tens of trillions of tokens”分配给某个模型；
- 把 Grok 4 的“pretraining-scale RL compute”误写成 pretraining tokens；
- 把 Grok 4.1/4.20 的 targeted mid-training 与 SFT/RL 合并。

本篇的核心问题是：**公开证据能否把 source class、训练阶段、时间边界、数据 controls 与实际 checkpoint lineage 连起来。**

## 代际与训练阶段表

| 代际 | 阶段 | 公开数据事实 | token/context/cutoff | 披露等级 | 结论状态 |
|---|---|---|---|---|---|
| Grok-1 | `P0/A1` | Internet + AI Tutors；pretraining next-token prediction，之后使用 humans 与早期 Grok-0 feedback finetune | token/mixture `unknown`；Internet cutoff Q3 2023；8,192 context；base pretraining 2023-10结束 | `D1/D3` | source/cutoff/artifact `verified`；recipe `open` |
| Grok-1 open release | raw `P0` checkpoint | 314B MoE、8 experts/2 active、JAX inference code、tokenizer、weights | 训练量/顺序/logs `unknown` | `D3` | architecture/weights `verified`；data reproducibility `open` |
| Grok 2 | `P0/A*` | family disclosure覆盖 public Internet、third-party、user/contractor、internal；专属代际细节很少 | 约2024-02开始训练；token/mix/cutoff `unknown` | `D1` | existence/start `verified`；recipe `open` |
| Grok 3 / 3 Reasoning | `P0/A3` | Colossus上大规模pretraining；reasoning用large-scale RL；tool/search是产品能力 | family-level tokens未分配；1M context；knowledge cutoff 2024-11 | `D1` | stage/context/cutoff `verified`；source/mix `open` |
| Grok 4 | `P0/A1/A3` | public Internet、third-party for xAI、users/contractors、internal；dedup/classification；SFT + human feedback/verifiable rewards/model grading | token/mix `unknown`；knowledge cutoff 2024-11；RL compute称pretraining-scale但不是token | `D1/D2` | categories/mechanisms `verified`；accounting `open` |
| Grok 4.1 | `P0/P2/A1/A3` | 同上 source classes；首次明确 targeted mid-training；post含SFT、human feedback、verifiable rewards、model graders | 各阶段 token/order/corpus `unknown` | `D1/D2` | stage sequence `verified`；mid data `open` |
| Grok 4.20 | `P0/P2/A1/A3` | public、third-party、internal；targeted mid-training；SFT + RL on human/synthetic reward signals | 1M产品context；各阶段token/cutoff `unknown` | `D1/D2` | stage sequence `verified`；user-data omission含义 `open` |
| Grok Code Fast 1 | coding `P0/A1` | coding-focused pretraining mixture；post-training含coding/tool-use/harness demonstrations；官方博客称真实PR/coding tasks | token/source share/cutoff `unknown` | `D1/D2` | focus/stages `verified`；provenance `open` |
| Grok 4.5 | `P*/A3` | coding/science/engineering/math；dedup、quality scoring、domain selection；数十万多步SWE/technical RL tasks，automated/model grading | raw/base token、mixture、cutoff、rollout/loss token `unknown`；500K产品context | `D1/D2` | current training description `verified`；lineage/accounting `open` |
| Grok 5 | training boundary | 2026-01官方仅称正在训练 | 所有data字段 `unknown` | `D0/D1` | status `verified`；recipe `open` |

## Assumption ledger

| 维度 | 本篇约束 |
|---|---|
| 研究对象 | Grok 模型家族、X/Grok 数据政策与产品检索，不把公司级披露直接分配到单模型 |
| 统计单位 | corpus token、sampled token、loss token、RL task、rollout、tool call、search result 分开 |
| 时间 | data collection、training start、knowledge cutoff、release date、search time 分开 |
| X 数据 | public X data、Grok-on-X interaction、grok.com interaction、feedback、private post、enterprise data 分开 |
| “may use” | 代表政策/处理通道存在，不代表某个 checkpoint 的实际 inclusion |
| RL | “pretraining-scale compute”是 compute comparison，不是 pretraining stage 或 token identity |
| 省略效应 | architecture、pretraining scale、mid-training、RL、tools 与 test-time compute 同时变化 |

## 统一数据字段

| 字段 | Grok-1 | Grok 3/4 | Grok 4.1/4.20/4.5 | 当前结论 |
|---|---|---|---|---|
| source | Internet、AI Tutors | public/third-party/user-contractor/internal | 4.1同类；4.20省略user/contractor；4.5只列domain | source classes部分可见，具体corpus `unknown` |
| rights | `unknown` | FAIF称public domain、licensed/purchased或有necessary rights | 同家族声明；逐source license未知 | 不能由公司声明重建rights manifest |
| cutoff | Q3 2023 Internet cutoff | Grok 3/4 knowledge cutoff 2024-11 | 4.1+ comparable cutoff `unknown` | cutoff披露不连续 |
| token accounting | `unknown` | family级trillions→tens of trillions | 全部`unknown` | 无法跨代计算data scaling |
| mixture | `unknown` | `unknown` | 4.5只有domain focus | 无source/domain/modality share |
| filtering | 未披露 | Grok4 dedup+classification | 4.5加quality scoring/domain selection | algorithm/threshold/retention未知 |
| mid-training | 未分列 | Grok4 card未分列 | 4.1/4.20明确有targeted mid-training | corpus/token/context目标未知 |
| synthetic | Grok-0 feedback不等同synthetic corpus | internal data + model grading；FAIF称用于RL/finetune/post | 4.20 synthetic reward；4.5 model grading | teacher/generator/acceptance/lineage未知 |
| validation | benchmark + 一次post-collection数学考试 | capability/safety/tool/agent eval | system-card safety与发布benchmark | pretrain held-out contract未知 |
| artifacts | raw base weights、tokenizer、inference code | cards/blog/API | cards/blog/system prompts | 除Grok-1外无训练资产 |

## 家族级 token 披露不能分配到具体 checkpoint

`[来源事实 | verified]` 2025-12-31 版 xAI Frontier Artificial Intelligence Framework（FAIF）的 AB-2013 data disclosure 写道：xAI 的模型使用包含“trillions to tens of trillions of tokens”的 datasets 和 dynamic datasets。

设公司披露的范围为：

$$
T_{\text{family}} \in [10^{12},\; O(10^{13})].
$$

它没有给出模型索引 $m$、阶段索引 $s$ 或 token 类型 $u$，因此不能推出：

$$
T_{m,s,u}=T_{\text{family}}.
$$

`[未知 | open]` 无法知道该范围描述 unique pool、dynamic snapshots、sampled exposures、loss tokens，还是多个模型/阶段的并集。也不能从训练开始日期反推出数据截止或训练时长。

FAIF 给出的训练启动时间只能作为 chronology anchor：Grok 1/1.5 约2023-08，Grok 2约2024-02，Grok 3/4/Code Fast/4 Fast约2024-09，Grok 4.1约2025-05。

## X ecosystem：四条不同的数据路径

### 1. Pretraining snapshot

`[来源事实 | verified]` Grok-1 model card 明确写 Internet up to Q3 2023 与 AI Tutor data。Grok-1 本身没有 user data；FAIF 对“data from users”也明确排除 Grok 1。

`[来源事实 | verified]` Grok 4/4.1 cards 将 users or contractors 列为 pretraining recipe 的 source class，但没有说明这些 users 来自 X、grok.com、alpha program或其他渠道，也没有比例和时间边界。

### 2. X Search / Web Search at inference

`[来源事实 | verified]` X 帮助页与 xAI 文档说明 Grok 可以在回答时决定搜索 public X posts 和实时 web。xAI API 文档同时说明：不启用 search tools 时，模型不知道训练数据之后的实时事件。

因此：

```text
public X post retrieved at time t
  --in-context evidence-for-->
response at time t

public X post retrieved at time t
  --does-not-prove-->
post was in pretraining snapshot
```

### 3. X data sharing for future training/finetuning

`[来源事实 | verified]` X 的帮助页说，X **may share** public X data 与 Grok-on-X 的 interactions/inputs/results 给 xAI，用于训练和 fine-tune Grok 或其他生成模型。public X data 被定义为 public posts、关联 metadata（engagement/reposts）、public Spaces 与 public profiles。

`[来源事实 | verified]` 用户可以关闭相应 data-sharing control；把账户设为 private 也会阻止 posts 用于训练或作为搜索结果出现。即使 opt out，主动提交 feedback 的对话仍可能被使用。

`[综合判断 | supported]` 这里的关键数据字段不是一个 `uses_X=true`，而是：

```text
X object type
  -> visibility state
  -> user control at collection time
  -> sharing eligibility
  -> training/finetuning selection
  -> model/version inclusion manifest
```

公开材料只覆盖前四步的一部分；后两步没有 checkpoint-level manifest。

### 4. grok.com consumer、Private Chat 与 enterprise

`[来源事实 | verified]` xAI Consumer FAQ称：普通consumer interactions可能用于训练；用户可关闭“Improve the model”；Private Chat不用于训练；主动feedback可能绕过一般opt-out路径进入改进用途。

`[来源事实 | verified]` xAI称 business/enterprise customer content 不用于改进模型；Collections data除非用户同意也不用于训练。当前政策不能倒推历史模型的训练事实。

## Pretraining → targeted mid-training → post-training

Grok 4.1 与4.20提供了明确但定量不足的阶段顺序：

```text
P0 broad pretraining
  -> P2 targeted mid-training for knowledge/capabilities
  -> A1 supervised finetuning
  -> A3 reinforcement learning
```

`[来源事实 | verified]` Grok 4.1 的 post-training reward sources 包括 human feedback、verifiable rewards 与 model-based graders；Grok 4.20 则概括为 human and synthetic reward signals。

`[未知 | open]` targeted mid-training 没有披露：

- source/domain 和是否复用 base corpus；
- unique/sample/loss tokens；
- sequence length、packing、replay 和 optimizer schedule；
- 是否包含 X snapshot、synthetic textbook、tool traces或long-context data；
- 与 SFT/RL 的去重和 contamination boundary。

`[综合判断 | supported]` 不能因 mid-training 的目标是“specific knowledge and capabilities”就把它等同于 domain SFT；它位于 post-training 之前，最安全的记录是独立 `P2` 阶段。

## Reasoning、verifiable data 与 agent trajectory

### Grok 3/4

`[来源事实 | verified]` Grok 3 reasoning 使用 large-scale RL；Grok 4 把 RL compute 扩到官方所称 pretraining scale，并将 verifiable training data 从主要 math/code 扩展到更多 domains。Grok 4 还用 RL 学会 code interpreter、web browsing 与 X search tools。

需要区分：

- verifiable **problem/task**；
- generated **rollout/trajectory**；
- verifier/model-grader **reward event**；
- tool call与environment transition；
- sampled output token与实际loss token；
- parallel test-time agents（Grok 4 Heavy）生成的 inference tokens。

`[未知 | open]` “over an order of magnitude more compute”没有给出 task、rollout、token、accepted trajectory 或 loss-token 分母。

### Grok 4.5

`[来源事实 | verified]` 2026-07-08发布页称 Grok 4.5 使用 coding/science/engineering/math data；pre/overall pipeline包含deduplication、quality scoring 与 domain-focused selection。RL覆盖数十万任务，重点是多步SWE与technical work，使用automated/model-based grading；异步rollout可持续数小时。

最小 provenance graph 应为：

```text
repository/task source
  -> environment snapshot
  -> prompt + tool schema
  -> multi-hour rollout
  -> automated/model grader
  -> acceptance/replay policy
  -> supervised/reward/loss tokens
```

`[未知 | open]` “hundreds of thousands of tasks”不是 trajectory 数、独立repository数、tool-call数或训练token数；同一task可产生多个rollout并重复采样。

## Grok Code Fast 1：specialist mixture 的边界

`[来源事实 | verified]` model card称其从 coding-focused mixture pretrain，再用不同agentic harness中的coding/tool demonstrations post-train；发布博客进一步称 pretraining corpus 富含 programming content，post-training包含real-world pull requests与coding tasks。

`[综合判断 | supported]` 这能支持“specialist data focus”，但不能支持“PR数据占比”或“全部来源为开源代码”。需要 repository license、commit/PR snapshot、file-level dedup、benchmark repository exclusion 与 build/test verifier manifest。

## Validation 与 contamination

### Grok-1 的一次有价值但有限的时间切分

`[来源事实 | verified]` 首发材料承认常用benchmark可能出现在web训练集中，无法排除污染；因此额外人工评分2023-05发布的匈牙利高中数学毕业考试，并称其发布时间晚于数据收集。

`[综合判断 | supported]` 这是比“不知道是否污染的web benchmark”更强的时间锚点，但仍缺：题目是否在AI Tutor/finetune中出现、translation/solution leakage、split unit、grader blind protocol与多次prompt选择。

### 后续 benchmark 的 decision exposure

`[来源事实 | verified]` Grok 3发布页在2025 AIME发布7天后报告结果，同时说明模型仍在训练；Grok 4/4.5也公开大量release-time benchmark。

`[综合判断 | supported]` “benchmark晚于knowledge cutoff”不自动证明clean：post-training、RL task curation、tool-access evaluation、prompt/harness选择和模型迭代都可能晚于 base cutoff。必须记录 decision exposure，而不只记录 corpus duplicate。

### 当前缺失的 contract

`[未知 | open]` 未见按模型固定的：

- pretraining validation source、time boundary、tokenizer与domain token counts；
- exact/near/semantic/code-repository decontamination范围；
- mid-training、RL与benchmark之间的交叉去污染；
- micro、macro与per-domain training-loss/validation-loss；
- data ablation、proxy-to-target transfer与seed variance。

## 表面冲突与区分性检查

| 表面冲突 | 首个不同点 | 区分性检查 |
|---|---|---|
| “Grok有实时X知识” vs cutoff 2024-11 | tool retrieval vs weights | 同一prompt关闭/开启X Search，记录citations与时间 |
| Grok4.1列users/contractors，4.20未列 | wording/lineage vs actual exclusion | 请求versioned source manifest；不能以省略推断删除 |
| family有trillions→tens of trillions tokens，但单模型未知 | company aggregate vs checkpoint accounting | 每模型/阶段unique-sampled-loss表 |
| Grok4 RL是pretraining-scale | compute magnitude vs training stage | 报告RL FLOPs、rollout/loss tokens与P0 token分列 |
| 4.5数十万tasks是否代表多样性 | task ID vs repo/environment/semantic group | 按source group去重并报告effective unique tasks |
| post-cutoff benchmark是否clean | base corpus cutoff vs later decision exposure | 冻结benchmark、prompt、grader和模型选择协议 |

## 可迁移经验与不可外推部分

### 可迁移

- `[综合判断 | supported]` 任何带实时检索的模型都要分开记录 checkpoint knowledge、retrieval corpus 与future-training feedback channel。
- `[综合判断 | supported]` mid-training应独立于SFT/RL记账，即使官方只给目标、不公开token。
- `[综合判断 | supported]` family-level token range不能分配给具体模型；至少需要模型、阶段、unique/sample/loss三个索引。
- `[综合判断 | supported]` 用户控制必须带采集时间和产品面：X、grok.com、Private Chat、feedback、enterprise不是一个consent布尔值。
- `[综合判断 | supported]` agent RL的task count必须展开为environment、rollout、tool call、reward和loss token。

### 不可外推

- `[未知 | open]` 任一 Grok 代际中 public X posts 的实际占比、snapshot date或selection policy；
- `[未知 | open]` Grok 2→4.5 的单模型 token、mixture、repeat与loss accounting；
- `[未知 | open]` Grok 4.1/4.20 targeted mid-training的corpus和规模；
- `[未知 | open]` Grok 4.5与Grok 4.20/4.3的base lineage；
- `[未知 | open]` Grok 5训练配方；正在训练不提供可审计数据事实。

## 掌握标准

读完后应能：

1. 解释为什么X实时搜索不等于X posts已进入pretraining；
2. 列出Grok-1与Grok 4 source disclosure的差别；
3. 正确解释family-level“trillions to tens of trillions”为什么不能填入单模型矩阵；
4. 把Grok 4.1的targeted mid-training与SFT/RL分开；
5. 为数十万agent tasks设计task/trajectory/tool/loss多分母审计。

## 推理型自测

1. Grok关闭搜索后答不出昨天事件、开启X Search后答出。这个实验支持哪种关系，不支持哪种训练结论？
2. 用户在2025-01 opt out。如何证明2024 snapshot与2026 model各自是否包含其public posts？还缺什么manifest？
3. Grok 4.5有300K tasks、每task平均生成8条rollout。为什么2.4M trajectories仍不等于unique training examples或loss tokens？
4. benchmark发布时间晚于2024-11 cutoff，但在2025 RL期间被用于checkpoint选择。它是否仍是clean held-out？

## 来源与建议阅读位置

1. [Grok-1 Model Card / Announcing Grok](https://x.ai/news/grok)
   - 为什么读：固定Internet Q3 2023、AI Tutors、8K与早期污染承认。
   - 建议位置：Grok-1 Model Card；Journey to Grok-1中的匈牙利考试。
2. [Grok-1 Open Release](https://x.ai/news/grok-os)
   - 为什么读：区分raw base checkpoint与对话finetune，确认权重/架构开放边界。
   - 建议位置：Model Details与GitHub链接。
3. [Grok-1 GitHub repository](https://github.com/xai-org/grok-1)
   - 为什么读：检查tokenizer、8-expert/2-active架构与inference artifact，而不是寻找不存在的数据manifest。
   - 建议位置：README、model.py、tokenizer.model。
4. [Grok 3 launch](https://x.ai/news/grok-3)
   - 为什么读：区分massive pretraining、reasoning RL、1M context与test-time compute。
   - 建议位置：Thinking Harder、Pretraining on a Massive Scale。
5. [Grok 4 launch](https://x.ai/news/grok-4)
   - 为什么读：固定pretraining-scale RL compute、verifiable data扩域与X/Web tool-use训练。
   - 建议位置：Scaling Up Reinforcement Learning、Native Tool Use。
6. [Grok 4 Model Card](https://data.x.ai/2025-08-20-grok-4-model-card.pdf)
   - 为什么读：固定public/third-party/user-contractor/internal source taxonomy与dedup/classification。
   - 建议位置：§3.1 Data and Training。
7. [Grok 4.1 Model Card](https://data.x.ai/2025-11-17-grok-4-1-model-card.pdf)
   - 为什么读：首次明确pretraining→targeted mid-training→SFT/RL阶段顺序。
   - 建议位置：§3.1 Data and Training。
8. [Grok 4.20 System Card](https://data.x.ai/2026-04-07-grok-4-20-model-card.pdf)
   - 为什么读：核对multi-agent代际的source wording、mid-training和human/synthetic reward signals。
   - 建议位置：§1.2 Model development、release process。
9. [Introducing Grok 4.5](https://x.ai/news/grok-4-5)
   - 为什么读：当前cutoff前最新data/filter/domain selection与asynchronous agent RL描述。
   - 建议位置：Training Grok 4.5。
10. [xAI FAIF data disclosure](https://data.x.ai/2025-12-31-xai-frontier-artificial-intelligence-framework.pdf)
    - 为什么读：固定family token范围、rights声明、training start chronology与Grok-1 user-data exception。
    - 建议位置：xAI Data Disclosure。
11. [About Grok on X](https://help.x.com/en/using-x/about-grok)
    - 为什么读：把X Search、public X data sharing、opt-out、private posts与feedback例外拆开。
    - 建议位置：How does Grok use data；training/fine-tuning controls。
12. [xAI Consumer FAQ](https://x.ai/legal/faq)
    - 为什么读：核对grok.com consumer、Private Chat、feedback与training controls。
    - 建议位置：Does xAI use my content for model training。
13. [xAI model docs](https://docs.x.ai/developers/models)
    - 为什么读：固定Grok 3/4的2024-11 knowledge cutoff，并确认search tools与weights知识的边界。
    - 建议位置：Additional Information Regarding Models。
14. [Grok Code Fast 1 Model Card](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf)
    - 为什么读：学习coding-focused P0与agentic harness demonstrations的specialist阶段边界。
    - 建议位置：Introduction与training description。

## 与主线的关系

```text
time-aware data manifest --prerequisite-for--> checkpoint cutoff claim
product data control --governs--> eligibility for future training
retrieval trace --explains--> realtime answer provenance
mid-training accounting --prerequisite-for--> domain attribution
task provenance graph --prerequisite-for--> agent RL data audit
open base weights --does-not-imply--> open training corpus
```
