---
title: "Apple Foundation Models 的数据实践"
description: "按 core、continued、context 与 multimodal 阶段核算 target/teacher exposure、端云蒸馏和披露变化。"
topic: "pretraining-data"
section: "international-cases"
slug: "apple-foundation-models-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 129
readtime: 27
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/apple-foundation-models.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/international/apple-foundation-models.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:55a590229355ba747dee04de2df79b8158bac5627d9066f615cab084734daaf1"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family trace`
> 辅助视角：`implementation-trace`，用于2024三阶段配方、2025多模态数据流与OpenELM开放对照
> 研究锚点：OpenELM、AFM 2024、AFM 2025、AFM 3（2026）

## 定位

Apple 同时提供三种不同披露形态：

1. OpenELM发布公开数据组成、配置、日志和checkpoint；
2. 2024/2025 Apple Foundation Models（AFM）技术报告给生产模型较完整的阶段token与pipeline，但不发布corpus/权重；
3. 2026 AFM 3发布时只给source taxonomy、共同基础与specialization高层关系，技术报告尚待发布。

这使同一组织内部也不能使用永久“透明/不透明”标签。正确主键仍是：

```text
(family, generation, training stage, artifact version)
```

本篇最重要的可检查关系是：

```text
source/pipeline
  -> core pretraining
  -> continued high-quality/domain remix
  -> multimodal adaptation / context lengthening
  -> SFT
  -> RLHF / task adapters
```

每个箭头都有不同的数据分母，不能把所有数字相加后称作“语料规模”。

## Motivating problem：一个“训练 token 总量”会隐藏什么

以2024 AFM-server为例：core为6.3T@4K，continued为1T@8K，context lengthening为100B@32K。若只写“7.4T”，会丢失至少三件事：

- 1T阶段down-weight bulk crawl、up-weight math/code并加入licensed long data；
- 100B阶段混入synthetic long-context QA；
- context、学习率与mixture同时变化，不能把效果只归因于数据量。

在阶段互不重叠且“tokens”统计口径相同的假设下，才可计算target sampled exposure：

$$
T_{\text{AFM-2024,target}}
=6.3\text{T}+1.0\text{T}+0.1\text{T}
=7.4\text{T}.
$$

这个推导不包括on-device pruning-mask学习的188B、teacher预训练、SFT/RL或adapter数据；也不等于unique corpus。

## 代际与训练阶段表

| 家族/代际 | 阶段 | 数据与规模 | sequence / mixture | 披露 | 状态 |
|---|---|---|---|---|---|
| OpenELM | `P0` | RefinedWeb、dedup PILE、RedPajama subset、Dolma v1.6 subset；pool约1.8T，模型sample 1.5T | 2K、约4M tokens/batch、350K steps | `D3/D4` | sources/config/logs `verified`；full order/decontam需查release |
| OpenELM Instruct | `A1/A2` | cleaned UltraFeedback 60K prompts | statistical rejection sampling或DPO | `D3` | dataset/recipe `verified` |
| AFM 2024 server | `P0` | licensed、public/open、Applebot web、license-filtered code、math；6.3T | 4K | `D2` | total/source/pipeline `verified`；unique/mix `open` |
| AFM 2024 on-device | `P0` | 同core mixture；6.3T student + 188B mask-learning exposure；初始化自pruned 6.4B teacher | teacher top-1权重0.9的distillation | `D2` | student/mask recipe `verified`；teacher total `open` |
| AFM 2024 server/on-device | `P2` | 1T；down-weight bulk web、up-weight math/code、加入licensed data | 8K；on-device此阶段不使用distill loss | `D2` | stage total/direction `verified`；exact percentages `open` |
| AFM 2024 server/on-device | `P3` | 100B；continued mix + synthetic long-context QA | 32K；多数原始documents显著短于32K | `D2` | exposure/recipe `verified`；natural/synthetic ratio `open` |
| AFM 2024 | `A1/A2` | human demonstration/preference + math/tool/code synthetic；code SFT有12K verified triplets | SFT→RLHF/RLOO；committee distillation | `D2` | mechanisms/selected counts `verified`；overall tokens/mix `open` |
| AFM 2025 server | `P0` | text sources延续高层taxonomy；13.4T | 150K tokenizer；PT-MoE | `D2` | total `verified`；base mix/cutoff `open` |
| AFM 2025 on-device | `P0/P2` | dense约14T；64-expert MoE teacher用1T HQ；student最后约1.4T做teacher distillation | sparse-upcycle teacher→last-10% distill | `D2` | exposure/lineage `verified`；overlap/order `open` |
| AFM 2025 text continued | `P2` | synthetic + highest-quality organic +部分bulk；multilingual总权重8%→30% | code/math/knowledge/multilingual；language temperature sampling | `D2` | mix change `verified`；stage token总量 `open` |
| AFM 2025 multimodal | `P1/A3` | 60% HQ text、10% interleaved、28.5% captions、1.5% domain image-text | on-device 1.3T@16K；server 420B@8K；每图144 embeddings | `D2` | mix/exposure/shape `verified`；loss mask/unique images `open` |
| AFM 2025 context | `P3` | licensed books/code、synthetic ICL/retrieval long data、continued replay | sequences up to65K；token与各类比例未知 | `D1/D2` | source/length `verified`；accounting `open` |
| AFM 2025 | `A1/A2/A3` | human+synthetic multilingual/multimodal/tool data；SFT/RL均English:multilingual=80:20 | SFT→asynchronous RLOO；RM/rule/code/judge rewards | `D2` | stage/mix `verified`；counts/rollout tokens `open` |
| AFM 3（2026） | `P*/A*` | public、licensed/purchased、open-source、dedicated studies、synthetic；无用户私密数据/交互 | common initial foundation→audio/image/long/reasoning specialization→SFT+multi-stage RL | `D1` | family/stage taxonomy `verified`；token/mix/pipeline `open` |

## Assumption ledger

| 维度 | 本篇约束 |
|---|---|
| target vs teacher | student、pruning-mask、MoE teacher和distillation replay分别记账；不把teacher 1T直接加到student exposure |
| unique vs sampled | OpenELM约1.8T为汇总pool、模型表为1.5T sampled；AFM的“trained for”按sampled exposure处理，unique未知 |
| tokenizers | 2024 server/on-device分别100K/49K；2025扩至150K。跨代token总量不视为等量文本 |
| image accounting | pair/document/image数与joint-training tokens并列；144 image embeddings不自动等于144个text tokens的相同loss |
| mixture | 2025的60/10/28.5/1.5按training data mix记录；未说明精确sampling unit时不换算unique images |
| multilingual | 8%→30%属于text continued mixture；SFT/RL的80:20属于后训练，不能合并 |
| decontam | 2024明确811 benchmarks及4–13 gram规则；2025报告未重复完整规则，不默认无变化继承 |
| privacy | “不使用用户private personal data/interactions”是官方范围声明；dedicated studies等仍需各自consent/rights manifest |
| AFM 3 | 2026-06-08为预览性发布，报告承诺later summer；不从Google协作推测具体训练模型或数据 |
| 省略效应 | architecture、tokenizer、hardware、quantization、optimizer、context与data均跨代变化 |

## 统一数据字段

| 字段 | OpenELM | AFM 2024/2025 | AFM 3（2026） |
|---|---|---|---|
| source | 公开dataset与subset名称 | licensed publishers、public/open、Applebot、GitHub code、math；2025加入licensed/public images和synthetic | public、licensed/purchased、open-source、dedicated studies、synthetic |
| rights | dataset各自license | robots.txt opt-out；code列MIT/Apache/BSD/CC0/CC-BY/Unlicensed/ISC/Artistic；其余逐源未公开 | 高层分类；逐源rights未知 |
| cutoff | dataset version可追 | source snapshot/data cutoff未给 | 未给 |
| accounting | pool约1.8T、sampled 1.5T | 多阶段sampled tokens、image pairs/docs/images；unique/loss多未知 | tokens全未知 |
| parse/filter | on-the-fly tokenizer；<200 chars或<256 tokens跳过 | Safari Reader+Boilerpipe→safety/PII/quality；2025加入headless render、LLM extraction、language-tuned model filters | 未给 |
| dedup | PILE标dedup，其他随源版本 | 2024 global fuzzy LSH；code/math/image另有dedup | 未给 |
| decontam | release需逐项核对 | 2024对811 benchmark做4–13 gram document removal，common threshold=1000 | 未给 |
| sampling/order | config/logs/checkpoints可查 | core→continued→context；2025增加multimodal与distill lineage | common foundation→specialization→SFT/multi-stage RL |
| synthetic | post UltraFeedback | long QA、math/tool/code；2025 image captions/QA/charts/translation | 有，比例/teacher未知 |
| validation | open harness/logs | benchmark、阶段checkpoint、human/feature/safety/locale eval；2024 SFT对eval prompts decontam | model/feature human eval，beta仍滚动 |
| artifacts | weights、CoreNet、configs、logs、checkpoints | papers；生产weights/data/order不开放 | high-level launch，technical report待发布 |

## OpenELM：同组织的开放控制组

OpenELM的pretraining pool约1.8T：RefinedWeb 665B；RedPajama中的GitHub 59B、Books 26B、ArXiv 28B、Wikipedia 24B、StackExchange 20B、C4 175B；PILE 207B；Dolma中的The Stack 411B、Reddit 89B、PeS2o 70B、Project Gutenberg 6B、Wikipedia+Wikibooks 4.3B。

各模型训练350K steps、context 2048、batch约4M tokens，论文表报告1.5T pretraining tokens。粗算：

$$
350{,}000\times4\text{M}
\approx1.4\text{T},
$$

与1.5T headline接近，差异来自“约4M”及batch/packing细节。它是sanity check，不覆盖正式日志计数。

on-the-fly过滤要求sequence至少200 characters且至少256 tokens；使用Llama tokenizer。开放framework、配置、日志和multiple checkpoints使loss curve与data/config版本可关联。

`[重要边界 | verified]` pool 1.8T与sampled 1.5T不是同一量；“比OLMo少2× data”是特定模型/benchmark比较，不是架构普适数据效率定律。

## AFM 2024：可追踪的三阶段配方

### source 与处理漏斗

Applebot web pipeline为：

```text
HTML
  -> Safari Reader + Boilerpipe body extraction
  -> heuristic/model safety and profanity filters
  -> global fuzzy dedup by locality-sensitive n-gram hashing
  -> heuristic/model quality filters
  -> 811-benchmark decontamination
```

decontam规则是与benchmark发生4–13 gram collision即移除整个document，除非该n-gram在语料中的collision count达到1000的common-usage threshold。

`[来源事实 | verified]` code来自GitHub中license-filtered open-source repositories，主要覆盖14种语言；另有3B math Q&A和14B math pages，经tag/symbol/domain/LM quality filters后dedup、decontam和PII removal。

`[综合判断 | supported]` 这给出比“已去污染”更强的operational contract，但仍缺benchmark revision、tokenizer前后匹配、Unicode/代码标准化、collision统计分母与clean-subset结果。

### core、continued 与 context

AFM-server从头训练6.3T@4096。AFM-on-device初始化自pruned 6.4B model：先在core mixture上用188B学习mask，再用teacher top-1预测权重0.9与真实label混合，训练6.3T。

continued阶段两者各1T@8192，upweight math/code、downweight bulk web、加入licensed long data；on-device此阶段不继续使用distillation loss。

context阶段各100B@32768，复用continued mix并加入synthetic long-context QA。报告明确多数pretraining documents显著短于32K，因此32K sequence并不等价于100B natural long-document tokens。

### post-training data factory

AFM 2024用human demonstrations、pairwise/single-sided preference与synthetic data：

- math：seed rephrase/reversal、depth/breadth evolution、embedding dedup、solvability/difficulty filter，再以ground truth或LLM judge筛CoT；
- tool：synthetic单工具bootstrap→human multi-tool/multi-step→oracle distractor、parallel calls与intent detection；
- code：71 programming topics→questions/tests/solutions→compile/execute rejection，最终12K high-quality SFT triplets；
- SFT selection：human rating、model filter、embedding dedup、synthetic/rejection sampling；
- RLHF：online human preference、reward model、committee rejection/distillation，小模型扩大distillation prompts。

报告还称超过10%的training data为adversarial或safety-related，但上下文覆盖SFT/RLHF/adapter safety流程，不能当作P0比例。

## AFM 2025：text、vision 与 multilingual 三条账本

### web/data pipeline diff

相对2024，2025报告新增：

- Applebot覆盖hundreds of billions of pages；
- domain-level language ID、topic distribution和URL-path signals；
- headless rendering、JavaScript execution与交互模拟；
- 特定domain使用LLM提取main content；
- 减少过度aggressive heuristic rules，增加按支持语言调优的model filters。

`[综合判断 | supported]` “保留更多informative tokens”是precision/recall trade-off的方向性描述；没有retention table或downstream controlled ablation，不能量化改进。

### 13.4T server 与 on-device teacher lineage

server text pretraining为13.4T。on-device先dense训练约14T，再把checkpoint sparse-upcycle为64-expert隔层MoE teacher，用1T high-quality data训练teacher；随后dense student只重训最后10%/约1.4T，使用MoE teacher distillation loss。

正确lineage是：

```text
dense student checkpoint @ ~12.6T
  + final ~1.4T teacher-distilled replay
  -> final dense on-device

dense checkpoint --sparse-upcycle--> MoE teacher
  -> 1T high-quality teacher training
  -> provides logits to student
```

`[重要边界 | verified]` teacher 1T影响student supervision，但不是student直接sampled exposure。若核算总项目compute/data processing，可另列teacher branch；不能加入student token headline。

### text continued multilingual curriculum

continued mixture由synthetic、highest-quality organic和部分bulk组成，code/math/multilingual加权。multilingual总权重从8%逐步提升到30%，bucket内部用temperature sampling平衡low-resource under/overfitting。

`[来源事实 | verified]` 报告称curriculum达到固定30%近似收益，同时缓解English regression。由于阶段token数、temperature和per-language loss未给，这仍是厂商实验结论，不能跨模型外推。

### image data规模与provenance

2025图像数据至少包含：

- web alt-text经alignment/compliance/filter/dedup后>10B pairs；
- 175M interleaved documents、>550M images；
- filtered public interleaved data 294M images；
- in-house captioner生成>5B synthetic image-caption pairs；
- text-rich PDFs/docs/charts/tables与OCR/QA；
- chart/plot synthetic values→code render→teacher QA；
- domain image retrieval→CLIP filter→teacher QA。

vision encoder先以>6B image-text pairs做CLIP式pretraining，再和302M decoder joint align，并加入text/interleaved/domain data。

`[未知 | open]` 各集合可能重叠，不能把10B+5B+550M+294M直接当unique images；还缺image hash/near-dedup、source URL/license、caption teacher version、NSFW/face/PII规则和train/eval group split。

### multimodal adaptation 的精确 mixture

两种模型都使用：60% HQ text、10% interleaved、28.5% image-caption、1.5% domain image-text。on-device为1.3T@16K，server为420B@8K；图像经encoder后各表示为144 token embeddings。

若百分比按sampled positions/token成立，则on-device名义分配为：

$$
\begin{aligned}
T_{\text{text}}&=0.60\times1.3\text{T}=780\text{B},\\
T_{\text{interleaved}}&=130\text{B},\\
T_{\text{caption}}&=370.5\text{B},\\
T_{\text{domain}}&=19.5\text{B}.
\end{aligned}
$$

这是`[推导结论 | supported]`：报告称training data比例但未在邻近文本定义分母，故不能把结果当unique text/image量或loss tokens。

### context 与 post-training

context lengthening使用最长65K sequences，来源为licensed books/code、synthetic ICL/retrieval long data和continued data replay；stage token总量与length histogram未知。

后训练在SFT后用asynchronous RLOO。RL prompts含text/multilingual和image-text的reward-model rewards、math和image-STEM的rule verification；trajectory generator与policy updater通过replay buffer异步解耦。SFT和RLHF均以English:multilingual=80:20采样，human-written与synthetic混合；mixed-language prompt/response仅占multilingual SFT mixture的0.4%。

tool annotation保留树状provenance：main stem为修正后的valid multi-turn trajectory，abandoned attempts为branches。若发布，至少应保留environment/database version、tool schema、reset point、annotator edit和final reward。

## AFM 3（2026）：共同基础与五模型 specialization

2026-06-08发布五个家族成员：AFM 3 Core、Core Advanced、Cloud、ADM 3 Cloud Image和Cloud Pro。官方称与Google协作构建，Cloud Pro另与Google/NVIDIA扩展Private Cloud Compute。

`[来源事实 | verified]` 数据高层分类为publicly available、licensed/purchased third-party、open-source、dedicated studies和synthetic；声明不使用用户private personal data或user interactions，并尊重publisher opt-out。

`[来源事实 | verified]` 所有模型共享common initial foundation，再按audio、image understanding、long-context reasoning、visual generation等specialize，之后SFT与multi-stage RL。

`[未知 | open]` 没有token、mixture、tokenizer、cutoff、crawl/filter/dedup/decontam、teacher、multimodal数量或post-training accounting。报告承诺2026夏季稍后发布，因此2025配方不能自动继承到AFM 3。

## Validation、contamination 与归因

2024提供stage checkpoint benchmark：core→continued→context可观察能力变化；long-context使用NIAH、RULER与13个synthetic tasks×500 examples。但mixture、LR、context/RoPE同时变化，属于recipe ablation而非纯data ablation。

2024 SFT数据还会对evaluation prompts去污染。生产评测包括open/licensed/proprietary payload、human preference、feature-specific、safety/adversarial与locale评价；2025进一步用machine translation和targeted synthetic扩展locale eval并由native speakers refine。

最低可复现补充应是：

```text
checkpoint + exact data manifest
  -> stage input/output counts
  -> sample order / tokenizer / mask
  -> eval revision + time boundary
  -> overlap report + clean subset
  -> micro + locale/domain macro/per-domain
```

### 表面冲突 1：2024 7.4T vs 2025 13.4T/14T

- 第一个差异：generation与tokenizer先不同；2025还加入multimodal/CPT阶段。
- 区分检查：固定bytes corpus测fertility，分列P0/P2/P3/A3 exposure，不直接算增长倍数。
- 结论：都是各代sampled headline，不能视为同一文本量，`open`。

### 表面冲突 2：不使用用户数据 vs dedicated studies

- private Apple user data/interactions被明确排除；dedicated studies是另一个source category。
- 区分检查：需要study consent、collection protocol、retention、PII处理和checkpoint manifest。
- 结论：逻辑上不冲突，但studies rights需逐项审计。

### 表面冲突 3：10B pairs + 5B captions + 6B vision pairs

- 10B是filtered crawl pairs；5B是synthetic pairs；>6B是vision encoder sampled/pair dataset描述。
- 可能存在抽样、重叠与重复exposure。
- 区分检查：发布source-to-derived lineage和unique image hash counts。
- 结论：不得相加为21B unique pairs。

## 明确未知项

- AFM各代完整source manifest、snapshot/cutoff、逐源rights与删除传播；
- raw/processed/unique corpus、packing、padding/mask与loss-token；
- 2024/2025 base/continued的完整domain mixture和sample order；
- 2025 text continued及context lengthening总token/长度分布；
- image集合交集、unique hash、teacher版本、caption rejection与rights；
- SFT/preference/RL prompts、rollouts、loss tokens和人类/合成比例；
- 2025是否完全复用811-benchmark规则及clean-subset结果；
- AFM 3 token、pipeline、model-to-model lineage和Google协作的具体数据边界。

## 可迁移经验与不可外推

可迁移：

- 训练账本分target、mask-learning、teacher和distillation replay；
- core→continued→context/multimodal阶段分别报告token、sequence与mixture；
- decontam必须给n-gram范围、document policy和common-phrase exception；
- 图像同时报告unique image、pairs、interleaved docs、sampled embeddings与loss mask；
- 多语言分别报告P0/P2、SFT与RL的分母；
- tree-structured agent data保留failed branches和environment state。

不可外推：

- 不把7.4T、13.4T和14T直接按倍数比较文本量；
- 不把teacher 1T加入student sampled tokens；
- 不把>10B crawl pairs、>5B synthetic pairs和>6B vision pairs相加为unique；
- 不把2024的811-benchmark合同自动继承到2025/2026；
- 不把“quality more important than quantity”当跨规模定律；
- 不从2026合作伙伴关系推测AFM 3训练数据或基础checkpoint。

## 与主线关系

```text
token accounting --prerequisite-for--> target/teacher exposure separation
dedup + decontam --implemented-by--> 2024 Applebot n-gram pipeline
continued mixture --changes-distribution-of--> stage exposure
vision encoder lineage --prerequisite-for--> multimodal accounting
locale validation --diagnoses--> multilingual mixture regressions
OpenELM --open-control-for--> AFM artifact disclosure
```

## 掌握标准

读者应能：

1. 从2024 AFM三阶段推导7.4T，并列出不能加入的数字；
2. 解释188B mask-learning、teacher 1T与student 1.4T distillation的区别；
3. 复述811-benchmark去污染规则及其剩余未知；
4. 将2025的8%→30%与80:20放进不同训练阶段；
5. 不把image pair、image、document、embedding和loss token混为一项；
6. 用lineage解释10B/5B/6B为何不能直接相加；
7. 说明AFM 3哪些字段已知、哪些必须等技术报告。

## 推理型自测

1. 若2024 core/continued/context存在大量replay，7.4T能否称为unique tokens？为什么？
2. 2025 server的13.4T和on-device的14T使用相同tokenizer吗？缺什么证据才能比文本量？
3. 60/10/28.5/1.5若按sequence而非token采样，上述780B等推导会怎样失效？
4. 为什么4–13 gram去污染加common threshold仍可能漏掉semantic contamination？
5. tree-structured tool data应按trajectory、turn、tool call还是loss token统计？请为不同问题选单位。
6. AFM 3声明不使用user interactions后，为什么dedicated studies仍需要单独rights ledger？

## 来源

1. [OpenELM](https://machinelearning.apple.com/research/openelm)：读公开数据组成、on-the-fly过滤、训练配置、logs/checkpoints与Instruct数据；重点看Section 2.2、Table 2/9和release links。
2. [Apple Intelligence Foundation Language Models 2024](https://arxiv.org/abs/2407.21075)：核心来源；读Section 3的Applebot/811 benchmarks、6.3T→1T→100B、on-device pruning/distillation，Section 4 synthetic/SFT/RL与Appendix阶段eval。
3. [Introducing Apple’s On-Device and Server Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models)：读2024产品映射、隐私与评测版本；技术细节以论文为准。
4. [Apple Intelligence Foundation Language Models Tech Report 2025](https://arxiv.org/abs/2507.13575)：读web pipeline diff、10B/5B/6B图像账本、13.4T/14T、8%→30%、multimodal mixture、65K context和异步RL。
5. [Updates to Apple’s On-Device and Server Foundation Language Models](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)：读2025产品/报告映射与adapter recovery；用于确认版本，不替代technical report。
6. [Introducing the Third Generation of Apple’s Foundation Models](https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models)：2026-06-08 AFM 3五模型、source taxonomy、common foundation→specialization→multi-stage RL与待发布报告边界。
7. [DataComp-LM](https://machinelearning.apple.com/research/datacomp-lm-search)：开放数据选择控制；读240T candidate pool、标准训练/eval设计，不能视为AFM生产pipeline。
8. [Language Models Improve When Pretraining Data Matches Target Tasks](https://machinelearning.apple.com/research/pretraining-data-matches)：读BETR、500+ proxy models和target leakage风险；作为validation/decision-layer专题来源，不用于补AFM语料。

## 后续复核触发器

- Apple发布承诺的AFM 3 technical report；
- AFM 2025/2026 dataset、opt-out、license或privacy statement更新；
- OpenELM/CoreNet dataset revision、logs/checkpoints变化；
- Apple公开context-stage tokens、multimodal lineage或decontamination report；
- Foundation Models framework adapter训练数据/许可边界变化。
