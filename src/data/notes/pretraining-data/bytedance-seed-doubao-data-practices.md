---
title: "ByteDance Seed / 豆包的数据实践"
description: "分层审计 multimodal、code 与 agent 配方，以及消费产品数据的训练适格性和 checkpoint inclusion。"
topic: "pretraining-data"
section: "china-cases"
slug: "bytedance-seed-doubao-data-practices"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 107
readtime: 27
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/bytedance-seed-doubao.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/china/bytedance-seed-doubao.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:709374eccbf173efa4512c59964f161519b617fca0609f7213bcf538fabb0103"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`method-family/historical trace` -->
<!-- maintenance: secondary-view=`implementation-trace`，用于 Seed-Coder 的 model-centric pipeline -->
> 主要模型/资料：Seed1.5-VL、Seed1.5-Thinking、Seed-Coder、Seed-OSS-36B、Seed1.8/2.0 与豆包产品数据政策

## 这篇案例研究什么

ByteDance 的公开材料包含两类不能互相补齐的证据：

1. **Seed 研究模型与开放权重**：Seed1.5-VL 和 Seed-Coder 给出较细的数据构造、阶段 token、过滤与 ablation；Seed-OSS 提供带/不带 synthetic instruction data 的成对 base checkpoint；
2. **豆包产品与隐私政策**：说明哪些用户内容在何种控制下可能用于未来模型改进，却不说明它们是否进入某个 Seed checkpoint、占比多少或经历了哪些处理。

因此技术关系必须写成：

```text
Seed technical report --documents--> named research-model recipe
Seed model --may-power--> Doubao / Volcano Engine product endpoint
Doubao eligible user data --may-contribute-to--> future model improvement
policy eligibility --does-not-imply--> inclusion in a named checkpoint
```

## 核心问题：同一品牌下至少有四套统计单位

本案例最容易产生四种错误：

- 把 Seed1.5-VL 的 3T multimodal source tokens 与其语言骨干此前的 text-only 训练量相加，却不标 lineage 和重复暴露；
- 把 Seed-Coder 的 6T sampled training tokens、74M commits、3M SFT pairs 和20K DPO pairs记成同一规模；
- 把 Seed-OSS 的 synthetic instruction pretraining 对照解释为“synthetic 总是有效”；
- 把豆包当前允许用于模型改进的数据类型倒推为历史 checkpoint 的训练来源。

本篇用阶段、数据对象、统计单位和证据边界逐项拆开这些问题。

## 各代模型和 training stage

| 模型/产品 | 阶段 | 已披露数据与规模 | context / mixture | 披露 | 状态 |
|---|---|---|---|---|---|
| Seed1.5-VL | adapter warmup | 16B sampled source tokens | 32,768；只训练 MLP adapter | `D2` | budget `verified` |
| Seed1.5-VL | multimodal `P0/P1` | 3T；caption、interleaved image-text、grounding、OCR 为主，约5% text-only | 32,768；全部参数训练 | `D2` | stage/mix `verified`；unique/loss `open` |
| Seed1.5-VL | capability/context stage | 240B | 加入 video、coding、3D；32K→131,072 | `D2` | budget/context `verified` |
| Seed1.5-VL | `A1/A2` | 初始13K人工 + 从1.5M open pool蒸馏/拒绝采样约30K；最终约50K multimodal SFT；之后 preference/RL | SFT 131K；RL prompt/rollout/loss token未知 | `D2` | counts `verified`；完整 provenance `open` |
| Seed1.5-Thinking | cold start + reasoning RL | 数十万竞赛级 math/code/science/reasoning problems；另有 non-verifiable prompts | LongCoT 最长32K；动态采样、verifier、streaming rollout | `D2` | object types `verified`；token accounting `open` |
| Seed-Coder Base | regular `P0` | 1T code-related web/math web + 4T curated code | 8K；FIM ratio 0.5 | `D2/D3` | sampled budget `verified` |
| Seed-Coder Base | continued `P1/P3` | 400B + 600B = 1T high-quality/long-context code | 8K→32K；FIM ratio 0.1 | `D2/D3` | budget/order `verified` |
| Seed-Coder Instruct/Reasoning | `A1/A2` | 3M SFT pairs、20K DPO pairs；LongCoT warmup + code RL | problem/pair/trajectory/loss token分母不同 | `D2/D3` | counts `verified`；rollout tokens `open` |
| Seed-OSS-36B | `P0/P3` | 12T；Base含 synthetic instruction data，Base-woSyn排除 | native 512K；完整 source/mix未知 | `D1/D3` | total/checkpoint control `verified` |
| Seed1.8 / Seed2.0 | multimodal/agent `P*/A*` | search、code、GUI、real-world agent评测；定量训练配方未披露 | serving/context与training exposure不可互换 | `D1` | model scope `verified`；recipe `open` |
| 豆包产品 | policy/consent | 登录用户可按数据类型关闭未来模型改进使用；未登录、未成年人模式内容不用于该目的 | 不是训练阶段或checkpoint manifest | policy `D1` | current policy `verified`；checkpoint inclusion `open` |

## 阅读这些结论前先确认的前提

| 维度 | 本篇处理 |
|---|---|
| 研究对象 | Seed family model recipe、开放 checkpoint 与豆包产品 policy 分开 |
| token 单位 | technical report 的“tokens”按 sampled source exposure记录；unique、non-padding、loss tokens无明示即 `unknown` |
| multimodal accounting | image/video经encoder后的token-equivalence、被mask token和语言loss token未完全公开 |
| lineage | Seed1.5-VL 的20B-active语言骨干已在trillions text-only tokens上训练；该历史不并入VLM 3.256T阶段和 |
| synthetic | pretraining instruction、web rewrite、SFT response、RL rollout与verifier-generated feedback分开 |
| consent | 当前政策只说明未来可用性；不倒推历史数据、checkpoint、比例或合法依据 |
| cutoff | 各训练 corpus 的统一时间边界未完整披露；benchmark日期不能替代 source cutoff |
| 省略效应 | 模型、架构、tokenizer、context、mixture、learning rate、filter和post-training常同时变化 |

## 厂商公开了哪些 data fields

| 字段 | Seed1.5-VL | Seed-Coder | Seed-OSS / Seed1.8/2.0 | 豆包政策 |
|---|---|---|---|---|
| source | caption、interleaved、OCR、grounding、video、text/code等 | GitHub code/commit、code-related Common Crawl、math web、synthetic tasks | OSS称以publicly available sources为主；最新模型细目未知 | conversation/output、上传媒体/文件、语音、屏幕与视频帧等eligible类别 |
| rights | source-level license/consent mix `unknown` | repository license distribution与web rights未完整披露 | weights license不等于语料权利 | 用户控制与去标识政策，不是source manifest |
| cutoff | `unknown` | RL集合部分明确使用时间边界；完整P0 cutoff未知 | `unknown` | 当前政策生效边界，不能变成knowledge cutoff |
| token accounting | 16B+3T+240B阶段数；modal/loss细分未知 | 5T+1T sampled；source retained/unique 与loss未知 | OSS 12T；1.8/2.0未知 | 不适用 |
| quality | CLIP、尺寸/比例/长度、URL/domain；task-specific filters | LLM file scorer、FastText recall filter、LLM page score、sandbox | OSS recipe细节不足；latest cards偏evaluation | 去标识与安全控制，不是质量评分 |
| dedup | image exact/near dedup；benchmark/task去污染 | SHA-256 exact + MinHash near dedup；repo/file粒度 | 完整阈值未知 | 不说明训练corpus dedup |
| mixture/order | 三阶段按能力与context变化 | regular→continued；file→repo dependency-aware long context | OSS synthetic paired control；latest order未知 | 不适用 |
| validation | 60项public/internal VLM eval、data scaling proxy、去污染 | model/pipeline ablation、held-out classifier split、code benchmarks | checkpoint pair与agent eval；pretrain held-out contract不足 | 不适用 |
| artifacts | report；无corpus/order/code | weights、repo、report；无完整raw corpus/manifest | OSS weights；1.8/2.0 cards | FAQ/instructions；无checkpoint lineage |

## Seed1.5-VL：先把语言骨干历史与多模态阶段分开

### 3T corpus 与3.256T阶段和不是同一个定义

`[来源事实 | verified]` Seed1.5-VL 以约20B active MoE LLM为语言骨干，该骨干此前已在 trillions of text-only tokens 上训练。VLM 报告另列三段 source-token exposure：16B、3T、240B。

若只对表中阶段做算术：

$$
T_{\text{VLM,stage}}
=16\text{B}+3\text{T}+240\text{B}
=3.256\text{T}.
$$

这里 $T_{\text{VLM,stage}}$ 的单位是报告表中的 sampled source tokens；假设三段互不重叠地计入 training schedule。摘要所说的“3T high-quality source tokens”可能指主要 corpus、Stage 1预算或舍入后的配方规模，报告没有明确给出两种口径的 exact identity。

`[推导结论 | supported]` 因此应同时保留 headline 3T 与阶段和3.256T，并将关系标为 `open`；不能擅自把256B解释为unique增量，也不能再把语言骨干历史tokens无条件相加。

### 三阶段 curriculum

`[来源事实 | verified]` Stage 0只训练MLP adapter；Stage 1训练全模型，主要包含caption、interleaved image-text、grounding和OCR，并保留约5% text-only data以维持语言能力。Stage 2加入video、coding、3D等高层能力，同时把 seqlen 从32,768扩到131,072。

```text
16B adapter alignment
  -> 3T broad multimodal learning at 32K
  -> 240B capability remix + 131K context
```

这是 `implementation relation`，不构成“240B数据本身导致长上下文能力”的独立因果结论，因为mixture、可训练参数、context和训练阶段同时变化。

### 图文过滤与 rare/common knowledge proxy

`[来源事实 | verified]` 对web image-text，报告列出CLIP score、image size/aspect、text length、image exact/near dedup以及URL/domain filter。不同任务还有OCR、grounding、instruction等专用构造，因此不能把CLIP score当成全模态质量分。

`[来源事实 | verified]` rare/common knowledge实验固定为12B tokens，对比随机46M samples、每concept最多1K的46M samples、每concept最多100的15M samples。限制头部concept可改善rare knowledge，但过强限制也会损害common knowledge。

`[综合判断 | supported]` 该实验说明固定token budget下，独立concept coverage与sample repetition必须分开；“样本更少但更均衡”并不等于全域更优。应报告concept定义、每concept cap、重复暴露和rare/common分域指标。

报告还给出OCR/grounding任务上的局部data-scaling拟合；它们只在指定模型、数据构造与预算范围内成立，不能外推为通用多模态scaling exponent。

### post-training 数据不是 pretraining token

`[来源事实 | verified]` multimodal SFT 从约13K crowdsourced样本起步，再从约1.5M open pool中蒸馏和拒绝采样约30K，最终约50K multimodal SFT，并混入内部text与LongCoT数据。之后使用human/synthetic preference与RL，包含capability tagging、stratified sampling和带污染检查的可验证视觉谜题。

这些量应分别记录为seed examples、candidate pool、accepted pairs、prompts、rollouts和loss tokens；50K不能与3T相加。

## Seed-Coder：model-centric pipeline 的实现锚点

### 6T tokens 怎样分到不同阶段

设regular pretraining与continued pretraining的sampled exposure分别为：

$$
\begin{aligned}
T_{\text{regular}} &= 1\text{T}_{\text{web+math}}+4\text{T}_{\text{curated code}}=5\text{T},\\
T_{\text{continued}} &= 400\text{B}+600\text{B}=1\text{T},\\
T_{\text{total}} &= 6\text{T}.
\end{aligned}
$$

regular阶段使用8K context与0.5 FIM ratio；continued阶段提升高质量和长上下文数据，把context扩到32K并把FIM ratio降到0.1。unique corpus与actual loss token未公开，不能将6T解释为6T独立代码证据。

### code、commit 与web是三条管线

```text
GitHub files/repos
  -> SHA-256 exact dedup + MinHash near dedup
  -> LLM quality scorer
  -> file/repository sequence construction

commit history
  -> message/context/patch extraction
  -> README + tree + BM25 top files
  -> patch-prediction examples

Common Crawl pages
  -> 10M annotation seed, 70/30 train/validation
  -> high-recall FastText candidate filter
  -> LLM 0–10 category-specific scoring
  -> code-related web corpus
```

`[来源事实 | verified]` file scorer关注readability、modularity、clarity与reusability；bottom约10%会被过滤。commit管线从约140K high-quality repositories中处理74M commits，形成约100B tokens。web分类器以约10M页面训练，高召回阶段在报告设置中达到约99% recall/45% precision，仅把约3% Common Crawl送入更贵的LLM评分，最终code-related web约1.2T tokens。

`[综合判断 | supported]` 这是“便宜高召回→昂贵高精度”的cascade，而不是一个统一threshold。必须分别保存候选率、每层recall/precision、domain acceptance与最终token retention。

### long-context sequence 需要 repository provenance

`[来源事实 | verified]` continued pretraining包含约1T long-context training tokens，兼有file与repository粒度。Python/Java/C使用dependency-aware topological concatenation；HTML/SQL/Shell等采用随机顺序；超大repository则抽取dependency subgraph。

最小manifest应记录：repository snapshot、commit SHA、license、file hash、dependency graph、concat order、sequence boundary、FIM transform与loss mask。否则只能复现token量，不能复现依赖结构。

### SFT、DPO 与code RL

`[来源事实 | verified]` instruct模型最终使用约3M高质量SFT pairs与20K DPO pairs。任务经过Tree-sitter/model filters、difficulty筛选和sandbox执行/自修正。reasoning warmup从CodeContests、ICPC、DeepSeek-R1、open-r1等获取数千LongCoT，再经sandbox rejection；RL还使用有明确时间边界的LiveCodeBench问题，并排除不可验证任务。

`[综合判断 | supported]` code data不能只记“有执行验证”。至少应区分parse success、build success、unit-test pass、hidden-test pass、repair attempts和最终loss mask；同一个accepted pair可能消耗多条rollout。

## Seed-OSS：synthetic pretraining 的成对checkpoint控制

`[来源事实 | verified]` Seed-OSS-36B Base公开权重，训练12T tokens，native context 512K；`Base`在pretraining中包含synthetic instruction data，`Base-woSyn`排除该类数据。

这形成一个比跨厂商benchmark更接近因果检查的配对：

```text
Base-woSyn --add synthetic instruction data under family recipe--> Base
```

但仍需核对两者是否严格固定总token、order、learning rate、seed和除synthetic外的mixture。官方benchmark表中synthetic版本并非每项都更好，因此结果只支持“在该配方与指标集合中的任务依赖效应”，不能外推为synthetic instruction data普遍有益。

`[未知 | open]` synthetic teacher、prompt family、acceptance、source-to-generation lineage、占比和loss-token accounting未随权重完整发布。

## Seed1.5-Thinking、Seed1.8 与 Seed2.0：agent数据对象继续增加

`[来源事实 | verified]` Seed1.5-Thinking 的RL数据分为可验证、答案确定的问题和non-verifiable prompts。前者包含数十万competition-grade math/code/science/reasoning problems，结合公开竞赛与内部来源，并进行参考答案修订、规格检查、difficulty filtering和verifier评估；后者来自豆包1.5 Pro的RL数据池，通过动态采样排除score variance过低或过高的极端。

streaming rollout使用不同版本checkpoint产生可能off-policy的segments。因此需要保存：problem版本、verifier版本、policy checkpoint、rollout时间、reward与token-level PPO mask。

`[来源事实 | verified]` Seed1.8面向search、code execution和GUI的统一agent interaction；Seed2.0继续强调real-world multimodal与复杂workflow。当前model cards主要给能力、系统和evaluation信息，没有与Seed1.5-VL/Thinking同粒度的source、token、mixture、dedup与order。

`[综合判断 | supported]` 最新代际的评测范围更广，不代表训练配方披露等级更高。Seed1.8/2.0暂记`D1`，不能继承Seed1.5-VL的3T或Seed-Coder的6T。

## 豆包产品数据政策：eligibility 不是 checkpoint inclusion

`[来源事实 | verified]` 豆包“模型训练数据常见问题”说明：登录用户可按数据类型关闭内容用于模型改进；关闭后不再用于未来训练。类别包括text conversation及模型输出、上传image/video/file及输出、live call/voice、screen share/video call frames等。页面还说明去标识/随机ID处理，未登录内容与未成年人模式内容不用于模型训练改进。

应使用四层记录：

```text
content created under product terms
  -> eligible under current policy and user control
  -> selected/processed for a future training pool
  -> included in a named checkpoint with measurable exposure
```

只有第一到第二层由当前FAQ支持。第三、四层的snapshot、retention、sampling、去重、consent proof与checkpoint lineage均`unknown`。

`[综合判断 | supported]` current opt-out policy不能证明历史checkpoint未使用某类内容，也不能证明所有eligible内容会被训练。产品search/tool outputs还可能含第三方内容，其rights与训练适格性需要另行审计。

## 如何验证这些结论，并检查 contamination

| 问题 | 首个可能差异 | 区分性检查 |
|---|---|---|
| 3T与3.256T为何不同 | corpus headline vs stage exposure | 发布stage manifest、repeat ratio与每阶段loss tokens |
| rare knowledge提升是否来自覆盖 | concept cap vs quality/repetition | 固定12B、固定filter，正交改变cap与sampling replacement |
| long context收益来自 sequence 结构还是更多tokens | dependency order vs random concat | 同repo/snapshot/token budget比较topological、random、document-shuffle |
| synthetic instruction是否有效 | synthetic presence vs总budget/order | Base/woSyn固定seed、steps、LR、总token并做per-domain paired eval |
| code verifier是否泄漏benchmark | source cutoff/near-duplicate vs执行正确 | repository/statement/solution三级exact+semantic audit，报告clean subset |
| 产品数据是否进入checkpoint | policy eligibility vs data lineage | consent/version/snapshot/transform/checkpoint manifest |

公开报告提供了大量release benchmark与部分data ablation，但完整pretraining validation contract仍缺：冻结版本、时间边界、split unit、每域token、tokenizer、decision exposure和跨模态污染范围不能同时闭合。

## 哪些经验可以借鉴，哪些不能直接照搬

### 可迁移

- `[综合判断 | supported]` 多模态curriculum要同时记录可训练参数、modality mix、context和stage exposure；只报总token会隐藏主要处理。
- `[综合判断 | supported]` high-recall cheap filter与high-precision model scorer应分别校准，Seed-Coder给出了recall/precision/candidate-rate的可操作锚点。
- `[综合判断 | supported]` repository long context需要保存dependency/concat provenance；seqlen 本身不证明长依赖质量。
- `[综合判断 | supported]` 带/不带某类数据的开放checkpoint pair比跨模型benchmark更适合归因，但仍需固定总budget、order和seed。
- `[综合判断 | supported]` consumer consent、training eligibility、selected pool和checkpoint inclusion必须是四个字段。

### 不可外推

- `[未知 | open]` Seed1.5-VL、Seed-Coder、Seed-OSS、Seed1.8与Seed2.0是否共享哪些具体corpus、filter或tokenizer版本；
- `[未知 | open]` 各模型的完整source/rights/cutoff、unique/sample/loss token闭合与训练顺序；
- `[未知 | open]` 豆包eligible user data在任一named checkpoint中的占比或是否被采用；
- `[待验证假设 | plausible]` Seed-Coder的model-centric pipeline可无损迁移到低资源语言；scorer与sandbox覆盖可能造成系统盲区；
- `[待验证假设 | plausible]` Seed1.5-VL的rare-concept cap在更大预算和不同概念taxonomy下仍维持收益。

## 读完后应该掌握什么

读完后应能：

1. 解释3T source headline、3.256T阶段和与语言骨干历史训练为何不能合成一个精确token数；
2. 写出 Seed-Coder 5T regular + 1T continued 分别属于哪个阶段，并列出 code/commit/web 三条管线；
3. 说明Seed-OSS Base/woSyn能支持什么归因、还缺哪些控制；
4. 把数十万RL problems、rollouts、tool calls与loss tokens分开；
5. 解释豆包当前policy为何不能证明某个checkpoint的数据组成。

## 用这些问题检查自己

1. Seed1.5-VL Stage 2的240B同时增加video并扩context。若long-video指标提升，最小的正交ablation是什么？
2. Seed-Coder classifier达到99% recall/45% precision。为什么不能据此推断最终1.2T web corpus有45%“高质量页面”？
3. Base比Base-woSyn在多数benchmark更好、少数更差。怎样区分task transfer、budget replacement与seed noise？
4. 用户未关闭某类数据的模型改进开关。还缺哪几层证据才能声称内容进入Seed2.0？

## 来源与建议阅读位置

1. [Seed1.5-VL Technical Report](https://arxiv.org/abs/2505.07062)
   - 为什么读：固定3T headline、16B/3T/240B阶段、rare/common proxy、multimodal filtering和post-training counts。
   - 建议位置：§3 Data、§4 Pre-training、§5 Post-training、Appendix data ablations。
2. [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914)
   - 为什么读：区分verifiable/non-verifiable RL data、动态采样、LongCoT与streaming rollout。
   - 建议位置：§3 SFT、§4 RL data/reward、streaming rollout部分。
3. [Seed-Coder paper](https://arxiv.org/abs/2506.03524)
   - 为什么读：复核 6T tokens 的阶段划分、code/commit/web pipeline、long-context construction 和 post-training data。
   - 建议位置：§2 Pre-training Data、§3 Training、§4 Post-training、Appendix pipeline details。
4. [Seed-Coder repository](https://github.com/ByteDance-Seed/Seed-Coder)
   - 为什么读：核对开放权重、模型版本、license与可运行artifact边界。
   - 建议位置：README、model table、license与data statement。
5. [Seed-OSS repository](https://github.com/ByteDance-Seed/seed-oss)
   - 为什么读：固定12T、512K和Base/Base-woSyn配对checkpoint。
   - 建议位置：README model introduction、training data与evaluation table。
6. [Seed-OSS-36B-Base model card](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Base)
   - 为什么读：确认公开权重、source高层描述与synthetic instruction data控制。
   - 建议位置：Introduction、Pre-training Data、Model Variants。
7. [Seed1.8 Model Card](https://arxiv.org/abs/2603.20633)
   - 为什么读：固定generalized agency的search/code/GUI范围，并确认quantitative recipe披露缺口。
   - 建议位置：Introduction、Model Overview、Evaluation。
8. [Seed2.0 Model Card](https://arxiv.org/abs/2607.00248)
   - 为什么读：核对最新模型谱系、real-world multimodal/agent范围及当前披露边界。
   - 建议位置：Introduction、Model Family、Evaluation与Limitations。
9. [豆包模型训练数据常见问题](https://www.doubao.com/legal/model_training_faq)
   - 为什么读：固定当前用户控制、eligible data types、去标识和排除项；不是checkpoint manifest。
   - 建议位置：数据类型、如何使用、如何关闭、未登录/未成年人说明。
10. [豆包大模型算法原理](https://www.doubao.com/legal/instructions)
    - 为什么读：确认产品对pretraining、SFT、RL的高层阶段描述及其定量缺口。
    - 建议位置：训练流程与数据处理说明。
11. [ByteDance Seed Models](https://seed.bytedance.com/en/models)
    - 为什么读：核对研究模型、开放模型与产品endpoint的命名关系，避免把品牌名当成同一checkpoint。
    - 建议位置：Seed1.5、Seed1.8、Seed-OSS与Seed-Coder条目。

## 这篇案例与主线知识的关系

```text
token accounting --prerequisite-for--> 3T vs 3.256T interpretation
multimodal pair accounting --prerequisite-for--> Seed1.5-VL mixture audit
model-based cascade --implemented-by--> Seed-Coder web pipeline
repository provenance --prerequisite-for--> dependency-aware long context
paired checkpoint control --supports-local-attribution--> Seed-OSS synthetic study
consent record --does-not-imply--> named-checkpoint inclusion
```
