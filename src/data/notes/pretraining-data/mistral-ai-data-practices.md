---
title: "Mistral AI 的数据实践"
description: "区分开放权重与数据可审计性，并约束 tokenizer、语言、context 线索对 corpus 账本的外推。"
topic: "pretraining-data"
section: "international-cases"
slug: "mistral-ai-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 128
readtime: 25
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/mistral-ai.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/international/mistral-ai.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:55071a178e7c923c36ca6371acf40f325b33c333cee0abddbb3093e80048c271"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family trace`
> 辅助视角：`implementation-trace`，用于 tokenizer、context 与后训练阶段
> 研究锚点：Mistral 7B、Mixtral 8x7B、Mistral NeMo、Pixtral 12B、Mistral Large 2/3、Mistral Small/Ministral、Forge

## 定位

Mistral 是研究“开放模型不等于开放训练数据”的清晰反例。组织发布了多代 base/instruct 权重、架构报告和推理代码；但截至 2026-07-14，帮助中心明确表示不披露训练数据集，也将训练逻辑和生产资源列为 proprietary assets。

因此本篇不尝试从 benchmark 或模型自述反推 corpus，而把可知信息拆成四层：

```text
released weights / architecture
  != model-training corpus manifest
tokenizer-training languages
  != model-training language mixture
context-window capability
  != long-document exposure
product data-use eligibility
  != a checkpoint's actual training membership
```

这条边界本身就是可复用结论：权重可下载提高了行为与实现可检查性，但无法补回 source、rights、cutoff、dedup、sample order 和 validation split。

## Motivating problem：能运行权重，能否审计训练数据

最小例子是 Mistral 7B 与 Mixtral 8x7B。两者都有 Apache 2.0 权重和技术论文；Mixtral 论文还说明用 multilingual data 预训练、32K context，并给出 SFT→DPO 的后训练顺序。但论文没有给 P0 source manifest、token 总量、语言比例、cutoff、过滤或去污染合同。

若把“open weights”直接记成“open data”，会把两个不同命题混在一起：

$$
\text{weight reproducibility}
\not\Rightarrow
\text{data provenance reproducibility}.
$$

前者允许固定 checkpoint 做 inference、finetuning 和部分 mechanistic checks；后者要求至少能追踪 source snapshot、处理版本、mixture、sampled exposure 和 benchmark overlap。

## 代际与训练阶段表

| 模型/代际 | 阶段 | 数据披露 | context / post-training | 权重/许可 | 状态 |
|---|---|---|---|---|---|
| Mistral 7B v0.1 | `P0` | source、token、cutoff、mixture均 `unknown` | 8K；架构报告以 sliding-window attention 为主 | base/instruct Apache 2.0 | weights/architecture `verified`；P0 data `open` |
| Mistral 7B Instruct | `A1` | 论文称使用 Hugging Face 上公开 instruction datasets，未用 proprietary data；未列完整 dataset/version/count | simple instruction fine-tuning | Apache 2.0 | 高层来源 `verified`；manifest/order `open` |
| Mixtral 8x7B | `P0` | 只披露 multilingual pretraining；source、token、比例、cutoff未知 | 32K；每token选2/8 experts | base/instruct Apache 2.0 | multilingual/context `verified`；data accounting `open` |
| Mixtral 8x7B Instruct | `A1/A2` | instruction dataset→paired feedback dataset；名称、规模、rights未知 | SFT→DPO | Apache 2.0 | stage order `verified`；rows/tokens `open` |
| Mistral NeMo 12B | `P0/A1` | 未披露P0 corpus/tokens；强调11种语言、function calling和advanced alignment | 128K；quantization-aware；Tekken tokenizer | base/instruct Apache 2.0 | capabilities/tokenizer `verified`；mixture/long exposure `open` |
| Pixtral 12B | `P0/A1/A3` | interleaved image-text；natural image/document；数量、pair source和比例未知 | 400M vision encoder from scratch + NeMo decoder；128K | Apache 2.0；后已deprecated | objective/architecture `verified`；multimodal provenance `open` |
| Mistral Large 2 | `P0/A1` | “very large proportion of code”“large proportion of multilingual data”，无分母/比例/token | 123B、128K；80+ code languages、dozens natural languages | instruct weights，Mistral Research License | qualitative mix `verified`；accounting `open` |
| Mistral Small 3.1 | `P0/A1/A3` | 无corpus/token/phase披露 | 24B级、128K、多语言/视觉/function calling | base/instruct Apache 2.0 | assets/capability `verified`；data `open` |
| Mistral Large 3 v25.12 | `P0/A1/A3` | 无公开source/token/mixture/cutoff | 675B total/41B active MoE、multimodal、256K | base/instruct open weights，Apache 2.0 | architecture/assets `verified`；data `open` |
| Ministral 3 v25.12 | `P0/A1/A2/A3` | 3B/8B/14B Base/Instruct/Reasoning；未披露语料账本 | 视觉、edge family；context依固定card | open weights | variants `verified`；stage data `open` |
| Forge | customer lifecycle | internal docs、code、structured data、operational records | pretraining/post-training/RL、synthetic generation、eval monitoring | 企业产品，非base corpus release | control-plane claims `verified`；每个客户实现 `open` |

## Assumption ledger

| 维度 | 本篇约束 |
|---|---|
| 研究对象 | 只把固定 model/version 与论文/card绑定；API别名、deprecated/replacement不自动继承数据字段 |
| 统计单位 | context length、tokenizer compression、parameter count和training tokens分开；Mistral多数代际training tokens未知 |
| “multilingual” | 表示官方声明或benchmark覆盖；除非给token分母，不转换成language mixture |
| “code proportion” | Large 2的“very large proportion”无数值分母，记作qualitative，不估算 |
| long context | 8K/32K/128K/256K是能力/配置，不证明多少长序列进入P0/P3 |
| multimodal | interleaved image-text说明objective/sequence形态，不说明图像独立数、pair数或loss positions |
| post-training | Mistral 7B、Mixtral有局部阶段说明；NeMo/Small/Large的“alignment”不自动映射为同一数据配方 |
| product policy | 2026-06-05用户数据政策是当日按产品/plan的eligibility；不能证明历史checkpoint实际包含哪些会话 |
| 省略效应 | architecture、tokenizer、context、quantization awareness、compute与data同时变化，跨代benchmark不能作data ablation |

## 统一数据字段

| 字段 | 已知 | 未知/边界 |
|---|---|---|
| source | Mixtral为multilingual；Pixtral为interleaved image-text并面向natural images/documents；Forge列举客户内部数据类型 | base corpus的具体web/code/book/licensed/user/synthetic source与snapshot |
| rights | model weights按Apache 2.0、MRL等逐代不同；Mistral 7B Instruct称不用proprietary data | P0 source-by-source license、opt-out/removal和derivative rights |
| cutoff | release/version日期可固定 | 不能把发布日期当knowledge/data cutoff |
| raw/unique/sample/loss | 均未给可比较的P0数值 | token总量、unique corpus、repetition、packing、mask |
| quality/filter | 无生产级公开pipeline | parser、语言阈值、quality classifier、rejection distribution |
| dedup/decontam | 无跨代统一合同 | exact/near/substr范围、benchmark版本/阈值/clean subset |
| sampling/order | SFT→DPO是Mixtral阶段顺序 | P0 mixture、curriculum、anneal、P3 exposure、seed/order |
| synthetic | Forge支持生成synthetic data以hill-climb evals | base/后训练各代synthetic比例、teacher、verifier与lineage |
| validation | 多数发布给benchmark表；Pixtral称同一harness/prompt且计划开放prompts | held-out source/time/dedup/token count，污染审计和per-domain loss |
| artifacts | 多代weights、配置、推理/finetune代码，部分base checkpoints | corpus/manifest、processing code、sample order、train logs与完整eval data |

## Mistral 7B：论文主要解释架构，而非数据

`[来源事实 | verified]` Mistral 7B技术报告重点是 grouped-query attention、sliding-window attention、rolling buffer和benchmark。论文没有给P0训练语料来源、token数、cutoff、mixture、dedup或decontamination。

`[来源事实 | verified]` Instruct部分只说明模型在Hugging Face repository公开的instruction datasets上fine-tune，未使用proprietary data或training tricks。它没有列出dataset revision、样本数、packing、顺序或拒绝规则。

因此可审计账本最多写成：

```text
Mistral-7B-v0.1 base
  P0 corpus/tokens/order: unknown
  architecture/weights: inspectable
  -> Mistral-7B-Instruct
     public HF instruction datasets, exact manifest unknown
```

`[综合判断 | supported]` “public datasets”比完全无来源更强，但不够复现。Hugging Face是托管位置，不是稳定dataset identity；需要repo id、revision、config、split与row hash。

## Mixtral：multilingual 与 SFT→DPO 是局部锚点

`[来源事实 | verified]` Mixtral 8x7B论文称以multilingual data预训练、context size为32K；每层router为每个token选择8个experts中的2个。它未给语言列表、token分母或P0 corpus。

`[来源事实 | verified]` Instruct训练为：

```text
base checkpoint
  -> SFT on an instruction dataset
  -> DPO on a paired feedback dataset
  -> instruct checkpoint
```

dataset名称、规模、annotator/teacher、pair构造、rights和loss-token均未披露。

`[重要边界 | verified]` 论文路由分析使用The Pile的不同domain作为validation/analysis input，不能据此推出The Pile是Mixtral的训练集。分析数据与训练数据是不同关系：

```text
The Pile subset --used-to-analyze-routing-of--> Mixtral
The Pile subset --trained-on--> Mixtral    # 未建立
```

这也是避免“论文出现dataset名称就记入训练源”的最小反例。

## NeMo 与 Tekken：tokenizer evidence 不等于 model mixture

`[来源事实 | verified]` Mistral NeMo是12B、128K模型，发布base/instruct Apache 2.0 checkpoints；官方称其对11种列举语言尤其强，并经过function calling和advanced fine-tuning/alignment。

`[来源事实 | verified]` Tekken基于Tiktoken，在超过100种语言上训练。官方报告相较旧Mistral SentencePiece tokenizer，对source code及中/意/法/德/西/俄文本约压缩30%，对韩语约2倍、阿拉伯语约3倍，并在约85%的语言上比Llama 3 tokenizer更高效。

若一段文本包含 $B$ bytes、被编码为 $N$ tokens，则可用：

$$
c_{\text{byte/token}}=\frac{B}{N}
$$

比较同一规范化文本的compression。更高 $c$ 表示每token承载更多bytes；但它不等于语言建模quality，也不能推出P0 language exposure。

`[综合判断 | supported]` tokenizer改变会改变token budget的语义。若模型A/B都声称“1T tokens”，但tokenizer不同，则应同时报告固定benchmark corpus的bytes/token或characters/token，不直接把1T视为等量文本。

## Pixtral：interleaved objective 是可检查 operational anchor

Pixtral 12B由从头训练的400M vision encoder和基于NeMo的12B decoder组成。图像按16×16 patches转为tokens，行间加入`[IMG BREAK]`、末尾加入`[IMG END]`；decoder在interleaved image-text序列上预测下一个text token。

对宽高为 $W,H$ 的图像，若不考虑缩放/边界padding，patch token的近似数量为：

$$
N_{\text{patch}}
\approx
\left\lceil\frac{W}{16}\right\rceil
\left\lceil\frac{H}{16}\right\rceil.
$$

这解释“native resolution会消耗不同数量tokens”，但不回答训练集中有多少图像、document screenshots、OCR pairs或纯文本replay。

`[未知 | open]` 需要图像独立数、pair/sequence数、模态sampling weight、caption/OCR provenance、图像dedup、text-image leakage和按模态loss mask，才能与其他多模态模型比较。

## Large/Small/Ministral：能力标签逐渐丰富，数据账本仍未闭合

### Large 2 的定性 mixture

Large 2官方给出123B、128K，并称训练使用“very large proportion of code”和“large proportion of multilingual data”；支持80+ programming languages与多种自然语言。

`[重要边界 | verified]` 这些是无分母的定性描述：80+ coding languages是覆盖列表，不是code mixture；支持语言也不是每语token比例。矩阵只能记`qualitative`。

### Small 3.1 与 Large 3

Small 3.1发布base/instruct、Apache 2.0，具备multilingual、multimodal和128K能力。Large 3 v25.12为675B total/41B active granular MoE、multimodal、256K，并发布base/instruct权重。

`[来源事实 | verified]` 这些资产使architecture、checkpoint behavior、tokenizer/config和downstream tuning更可检查。

`[未知 | open]` 官方资料没有给相应P0 token、source mix、cutoff、P3 long exposure、multimodal pair accounting或后训练数据量。不能把NeMo的Tekken语言结论、Large 2的code描述或Pixtral的interleaved recipe自动继承到Large 3。

### Ministral 与 Reasoning variants

2024 Ministral 3B/8B强调128K edge deployment；2025-12 Ministral 3扩为3B/8B/14B，并有Base、Instruct、Reasoning及视觉能力。

variant命名只建立产品/权重关系：

```text
Base --post-trained-into?--> Instruct / Reasoning
```

问号不能删除，直到官方给出checkpoint lineage、SFT/RL/DPO阶段、数据版本和是否共享base。

## 用户数据政策：eligibility 与 checkpoint membership 分离

`[来源事实 | verified]` 2026-06-05帮助中心按产品/plan说明：

- Vibe Free/Pro/Education的input/output默认用于训练，除非opt out；
- Vibe Team/Enterprise不用于训练，也不能手动opt in；
- 任一plan主动提交thumb feedback及comment，会授权使用rating、input与output改善模型行为；
- Mistral Studio Free可能使用input/output并允许opt out；Scale pay-per-use不用于训练。

正确的数据模型至少包含：

```text
(product, plan, policy_version, consent/opt-out state, feedback action)
  -> eligible_for_training
  -> selected_for_dataset?        unknown
  -> included_in_checkpoint?      unknown
```

`[综合判断 | supported]` 政策证明的是某类数据在某日的使用资格与控制，不证明所有eligible对话均被采样，也不能反向解释2023–2025 checkpoint。

## Forge：公开的是定制生命周期，不是基础语料

Forge列出企业可用internal documentation、codebase、structured data和operational records做pretraining，并继续post-training与RL；支持dense/MoE、多模态、synthetic generation、hyperparameter search和eval regression monitoring。

这给出一个有用control plane：

```text
versioned internal data
  -> data pipeline
  -> pretrain/post-train/RL job
  -> internal eval + policy constraints
  -> deployment/retraining loop
```

`[重要边界 | verified]` Forge的产品能力不能作为Mistral基础模型实际pipeline的证据。每个客户项目仍需补manifest、rights、retention、teacher、eval isolation和deployment feedback loop。

## Validation、contamination 与可区分检查

Mistral发布常给统一benchmark表；Pixtral明确说用同一evaluation harness和prompt重评模型，并计划开放prompts。它改善了横向prompt一致性，但没有闭合训练污染问题。

最低validation合同应补：

```text
eval dataset + exact revision
  -> time boundary
  -> train/eval exact + near + semantic overlap
  -> tokenizer and prompt template
  -> per-domain sample/token count
  -> micro + macro + per-domain metric
  -> clean subset result
```

### 表面冲突 1：开放模型 vs 不披露训练数据

- 声明A：多代模型是open/open-weight，并发布base权重。
- 声明B：Mistral明确不披露training datasets/logic/resources。
- 第一个差异：A描述artifact access；B描述data/process disclosure。
- 区分检查：逐项检查weights、config、code、data、manifest、order、logs，而不是给组织一个总“open/closed”标签。
- 结论：两者不冲突，属于不同字段，`verified`。

### 表面冲突 2：128K/256K context vs long-context training data

- capability claim：模型能接受长输入。
- data claim：训练时长序列的token比例、长度分布和P3 recipe。
- 区分检查：需要sequence-length histogram、long/source mixture、positions进入loss的比例及context-extension ablation。
- 结论：前者不能推出后者，后者保持`open`。

### 表面冲突 3：multilingual capability vs multilingual mixture

- capability evidence：语言benchmark、支持列表、tokenizer compression。
- exposure evidence：按固定tokenizer计的每语sampled/loss tokens。
- 区分检查：固定bytes corpus报告tokenization fertility，再给training language histogram和per-language held-out loss。
- 结论：现有证据支持capability，不支持定量mixture。

## 明确未知项

- 各代P0 raw/processed/unique/sample/loss token；
- source manifest、snapshot、licensed/partnership/user/synthetic比例与rights；
- knowledge/data cutoff及删除请求如何传播到后续dataset版本；
- parser、quality filter、language ID、dedup与benchmark decontamination；
- P0→P2/P3阶段切换、长序列长度分布、replay与sample order；
- 除局部论文外的SFT、preference、RL、reasoning和agent trajectory规模；
- multimodal image/audio/document来源、独立数、pair数、dedup和模态loss accounting；
- production train/validation split、per-domain loss、ablation和训练日志。

## 可迁移经验与不可外推

可迁移：

- 把artifact openness拆为weights/config/code/data/manifest/order/logs；
- tokenizer compression作为token accounting的必要伴随指标；
- 在模型卡中分开capability、exposure和validation evidence；
- 用户数据保留`policy eligibility -> selection -> checkpoint membership`三段lineage；
- multimodal模型至少记录patch/token shape与模态采样分母。

不可外推：

- 不因Apache权重推断训练语料可再分发或已合法审计；
- 不因语言/代码benchmark强而反推mixture比例；
- 不把context size当P3 token exposure；
- 不把The Pile路由分析误记成Mixtral训练source；
- 不把2026用户政策外推到历史checkpoint或所有产品；
- 不把Forge能力当Mistral base-model pipeline事实。

## 与主线关系

```text
artifact taxonomy --prerequisite-for--> disclosure scoring
tokenizer fertility --changes-unit-of--> token budget comparison
context capability --requires-separate-evidence-from--> long-context exposure
product policy --governs-eligibility-of--> user-derived data
manifest + overlap audit --prerequisite-for--> contamination-aware validation
Mistral case --counterexample-to--> open weights imply open data
```

## 掌握标准

读者应能：

1. 解释为什么Mistral权重开放与P0数据`open`可以同时成立；
2. 将Mixtral的multilingual、32K、SFT→DPO分别放进正确字段；
3. 不把Tekken的100+语言训练误写为NeMo模型的language mixture；
4. 从Pixtral图像尺寸估算patch token，同时指出缺失的dataset分母；
5. 将Large 2“very large proportion of code”保留为qualitative；
6. 用eligibility/selection/membership三层解释用户数据政策；
7. 设计能区分context capability与long exposure的检查。

## 推理型自测

1. 两个模型都训练“1T tokens”，其中一个tokenizer对中文压缩率高30%。需要哪些额外量才能比较中文文本exposure？
2. 论文用The Pile画expert routing图，为什么不能把The Pile填进训练source列？
3. Large 2支持80+代码语言，能否推出code占P0的80%以上？第一个缺失分母是什么？
4. Pixtral支持128K和多图输入，如何验证训练时真的见过多图长序列而非主要靠泛化？
5. Vibe Free会话默认eligible，为什么仍不能断言某条会话进入Large 3？
6. 若未来发布“15T training tokens”，至少要问哪四个accounting问题，才能与其他厂商比较？

## 来源

1. [Mistral 7B（arXiv:2310.06825）](https://arxiv.org/abs/2310.06825)：读架构与Instruct数据边界；重点看abstract、instruction fine-tuning和limitations，确认P0 corpus并未给出。
2. [Mixtral of Experts（arXiv:2401.04088）](https://arxiv.org/abs/2401.04088)：读multilingual/32K和SFT→DPO；重点看abstract、introduction、Section 4，以及The Pile只用于routing analysis的上下文。
3. [Mistral NeMo](https://mistral.ai/news/mistral-nemo/)：读12B/128K、base/instruct许可、Tekken训练语言与compression；不要把tokenizer资料当P0 mixture。
4. [Announcing Pixtral 12B](https://mistral.ai/news/pixtral-12b/)：读interleaved objective、16×16 patches、400M encoder/12B decoder和统一eval harness；该页现标deprecated，历史结论按2024-09-17版本使用。
5. [Large Enough / Mistral Large 2](https://mistral.ai/news/mistral-large-2407/)：读code/multilingual定性描述、128K、123B与MRL；重点看Code & Reasoning、Language diversity。
6. [Mistral Small 3.1](https://mistral.ai/news/mistral-small-3-1/)：读base/instruct、multimodal/128K和Apache 2.0；用于确认2025小模型资产，不用于补写corpus。
7. [Mistral Large 3 model card](https://docs.mistral.ai/models/model-cards/mistral-large-3-25-12)：读v25.12、675B/41B、256K与权重入口；训练数据字段仍为空。
8. [Mistral models catalog](https://docs.mistral.ai/models/)：读固定版本、replacement/deprecation及Ministral 3 variants；产品目录只用于identity mapping。
9. [Does Mistral disclose its training datasets?](https://help.mistral.ai/en/articles/347390-does-mistral-disclose-its-training-datasets)：2026-06-05官方明确不披露训练数据集/逻辑/资源，是本篇披露边界的一手来源。
10. [Do you use my user data to train your AI models?](https://help.mistral.ai/en/articles/347617-do-you-use-my-user-data-to-train-your-artificial-intelligence-models)：2026-06-05按Vibe/Studio plan及feedback拆分使用政策；用于eligibility，不用于checkpoint归因。
11. [Introducing Forge](https://mistral.ai/news/forge/)：读企业internal data、pretraining/post-training/RL、synthetic和eval loop；仅作为custom training control-plane案例。

## 后续复核触发器

- Mistral发布training data card、source/rights/cutoff或token accounting；
- Large 3/Ministral 3出现固定技术报告或base/post lineage；
- 用户数据、opt-out、feedback或ZDR政策更新；
- model card/license/replacement发生变化；
- 任一版本发布context-extension、multimodal data或contamination报告。
