---
title: "Meta Llama 家族的数据实践"
description: "沿 Llama 代际追踪 mixture、global dedup、长上下文、multimodal token 与 data ablation。"
topic: "pretraining-data"
section: "international-cases"
slug: "meta-llama-data-practices"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 121
readtime: 26
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/meta-llama.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/international/meta-llama.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:71960259f48c6363b81475304e350ee63b5429aac2da8dcf2ce8c605306f6b54"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`method-family/historical trace` -->
> 主要模型/资料：Llama 2、Llama 3/3.1、Llama 3.2、Llama 4 Scout/Maverick。

## 这篇案例要回答什么

Llama 位于 OLMo 与闭源 API 模型之间：Meta 发布权重、模型卡和较详细技术报告，但没有发布可逐文档追踪的 pretraining corpus、训练顺序、完整配置和训练日志。它适合检验：在 D2–D3 资料下，哪些 data mechanism 可以研究，哪些仍停留在发布方报告。

家族演化的核心不是 token 数从 2T 增到 15.6T/22T/40T，而是数据对象同时发生了变化：英语为主的公开网络文本，变为带明确 code/math/multilingual 配比的文本，再变为 text/image/video early-fusion mixture 和单独 mid-training。不同代际的 tokens 因 tokenizer、模态和阶段不同，不能只按总量排名。

## 各代模型和 training stage

| 代际 | 阶段 | 已披露数据与规模 | cutoff / context | 披露 |
|---|---|---|---|---|
| Llama 2 7B/13B/70B | `P0` | 公开在线来源的新 mixture；2.0T pretraining tokens；不含 Meta user data | 2022-09；4K | `D2/D3` |
| Llama 3.1 405B | `P0` | 15.6T multilingual text tokens；约 50% general、25% math/reasoning、17% code、8% multilingual | source knowledge through 2023-12；8K initial context | `D3` |
| Llama 3.1 405B | `P3` | 约 800B long-context training tokens，六阶段从 8K 扩到 128K | 128K | `D3` |
| Llama 3.1 405B | `P2` | 最后 40M tokens 线性降 LR 到 0，上采样 very-high-quality sources，并做 checkpoint averaging | 128K | `D3` |
| Llama 3.2 1B/3B | `P0/P1` | 最多 9T tokens；剪枝后用 Llama 3.1 8B/70B logits 做 token-level knowledge distillation | 2023-12；128K | `D3` |
| Llama 4 Scout | `P0` | 模型卡：约 40T multimodal tokens；public、licensed、Meta products/services，包括公开 IG/FB posts 与部分 Meta AI interactions | 2024-08；10M | `D3` |
| Llama 4 Maverick | `P0` | 同类 source；模型卡约 22T multimodal tokens | 2024-08；1M | `D3` |
| Llama 4 family | `P1/P3` | mid-training 改善核心能力，并用 specialized datasets 做 long-context extension | Scout 10M、Maverick 1M | `D2/D3` |
| Llama 4 Instruct | `A1/A2` | lightweight SFT → online RL → lightweight DPO；SFT 移除 50%+ easy data，RL 持续筛 medium-to-hard prompts | n/a | `D2/D3` |

`D3` 主要来自权重、模型卡、论文和部分 eval artifacts；没有训练 corpus/manifest/order/logs，因此达不到本项目的 `D4`。

## 阅读这些结论前先确认的前提

| 项 | 本笔记的处理 |
|---|---|
| 研究对象 | base/pretrained 与 Instruct 分开；Llama 4 Scout/Maverick 分行，不把 Behemoth teacher 当已发布模型 |
| token 统计 | Llama 2/3 为 text tokens；Llama 4 模型卡称 multimodal tokens，但各模态 tokenization/分母未完整披露 |
| 阶段 | Llama 3 initial pretraining、long-context continued pretraining、annealing 分开；Llama 4 mid-training 与 post-training 分开 |
| mixture 单位 | Llama 3 报告 token 百分比；Llama 4 只报告语言覆盖和总 token，没有 text/image/video 比例 |
| tokenizer | Llama 2 SentencePiece；Llama 3/4 TikToken-based。不同 tokenizer 下 token 总量不直接可比 |
| cutoff | 采用模型卡/source snapshot 的绝对日期，不采用模型自述 |
| 省略效应 | 长上下文、annealing 和 capability 变化同时受架构、LR、data mix、checkpoint averaging 与 post-training 影响 |

## 厂商公开了哪些 data fields

| 字段 | Llama 2 | Llama 3/3.1 | Llama 4 | 置信 |
|---|---|---|---|---|
| source | publicly available online data | variety of web/data sources；具体 corpus 名单未发布 | public、licensed、Meta products/services | 官方类别 `verified`；逐 source `open` |
| rights | source-level rights `unknown`；权重为 Llama custom license | 同左；Llama 3.1 Community License | 同左；Llama 4 License | 发布许可 `verified`；语料许可 `open` |
| token scale | 2.0T | 405B：15.6T text tokens；3.2 小模型最多 9T | Scout ~40T、Maverick ~22T multimodal tokens | 报告范围 `verified`；loss tokens `open` |
| cutoff | 2022-09 | 2023-12 | 2024-08 | model-card claim `verified` |
| mixture | `unknown` | 50% general、25% math/reasoning、17% code、8% multilingual | 200 languages；100+ languages 各超过 1B tokens；模态/领域比例 `unknown` | Llama 3/4 声明 `verified` |
| parse | `unknown` | custom HTML parser；保留 math/code 结构与 alt text；报告 plain text 优于保留 markdown | `unknown` | Llama 3 `verified by report` |
| dedup | 论文有 curation 描述但本轮不沿用到后代 | global URL、global MinHash document、30M-doc bucket 内 line dedup；多语种内 document/line dedup | 具体方法 `unknown`，不能默认沿用 Llama 3 | Llama 3 `verified by report` |
| filtering | 高层质量/安全说明 | heuristics、fastText/DistilRoBERTa quality、Llama 2 labels、domain-specific code/reasoning classifiers、PII/adult domain filters | source 与 post-training difficulty filtering 有高层说明；pretraining 阈值 `unknown` | Llama 3 `supported`；Llama 4 `open` |
| decontam | `unknown` | anneal 不含常用 benchmark train sets；post-training prompts 对 benchmark 做 exact match | `unknown` | 声明范围 `verified`；全面无污染 `open` |
| long context | 4K | 800B tokens，8K→128K 六阶段；用 short eval recovery + needle test 判定适配 | specialized datasets mid-training；精确规模/mix `unknown` | Llama 3 `verified`; Llama 4 `open` |
| synthetic/distill | post-training 人工数据为主 | post-training 大量 synthetic；3.2 用 larger-model logits distill | Behemoth teacher distillation；各阶段 token/占比 `unknown` | 存在性 `verified`; 定量 `open` |
| artifacts | weights、inference code、paper/card | weights、inference code、paper/card、部分 eval outputs | weights、inference code、card/blog | `verified` |

## Llama 3：可操作的数据机制锚点

### 三层去重

Llama 3 报告的 web pipeline 可写成：

```text
URL dedup（全局，保留较新版本）
  -> document MinHash dedup（全局 near-duplicate）
  -> line dedup（每 30M documents bucket，出现 >6 次的行）
  -> heuristic/model/domain filters
  -> knowledge-category mix
```

这里 `line dedup --implemented-by（30M-doc bucket）--> frequent-line removal`，不是全局 exact line dedup。论文明确承认它也会移除高质量常见文本，但报告 empirical evaluation 显示总体改善；这属于 setting-specific `empirical association`，不是“越 aggressive 越好”的普遍结论。

### 数据配比与动态调整

最终静态摘要为：

$$
q=(0.50_{general},,0.25_{math/reasoning},,0.17_{code},,0.08_{multilingual})
$$

但训练 recipe 还会随阶段增加非英语与数学、加入较新 web、下采样后来识别的低质量子集。因此这组 $q$ 是报告的 final mix summary，不足以恢复每个训练 step 的 exposure。

### Annealing 作为数据源筛选代理

论文用 50%-trained Llama 3 8B，在 40B tokens 上把 LR 线性降到 0；候选新数据权重 30%，默认 mix 70%，用来评估小型 domain dataset：

$$
q_{probe}=0.3q_{candidate}+0.7q_{default}
$$

这是 candidate-ranking probe，不是 full-run 的 exact substitute。8B 上加入 GSM8K/MATH train data 的 annealing 报告明显收益，而 405B 上收益接近零，说明 proxy value 随规模变化；同时正式 anneal 不包含常用 benchmark train sets。

## Long-context 数据与 validation

### Llama 3

`P3` 在约 800B tokens 内分六阶段从 8K 扩到 128K。每次扩展的成功标准同时要求：

1. short-context evaluations 完全恢复；
2. 在目标长度上通过 needle-in-a-haystack。

这比只报告“支持 128K”更可操作，但仍缺少 long-document source、长度直方图、packing 与每阶段 tokens。post-training 另用 synthetic QA、hierarchical summarization 和 repository reasoning；ablation 报告将 0.1% synthetic long-context data 混入 short-context SFT mix 能平衡短/长任务。这个 0.1% 属于 `A1`，不得回填为 `P3` mixture。

### Llama 4

Meta 只说明 mid-training 使用 specialized datasets 做 long-context extension，并达到 Scout 10M/Maverick 1M。没有公开逐长度 curriculum、tokens、真实/拼接文档比例或 short-context recovery protocol，因此“更长 context”不能证明模型在真实 10M 文档分布上接受过等比例训练。

## Multilingual 与 multimodal 边界

Llama 3 的 8% multilingual 是 text-token mix；语言识别覆盖 176 languages，但正式支持语言更少。Llama 4 宣称 pretraining 覆盖 200 languages，100+ languages 各超过 1B tokens，并有 Llama 3 的 10× multilingual tokens；这三个数字分别描述 pipeline coverage、最低 exposure 门槛和相对规模，不等价于均匀语言分布。

Llama 4 early fusion 联合训练 text/image/video。模型卡的 22T/40T 称为 multimodal tokens，但没有披露 image/video token 的定义、分母与 mixture；因此不能与 Llama 3 的 15.6T text tokens 做严格倍数比较。

## 看似矛盾的说法怎样区分

### Llama 2 是 2.0T 还是 1.8T？

- 模型卡给出 2.0T；Llama 3 paper 在代际摘要中写 1.8T。
- 第一个差异可能是 rounded model-card budget vs paper 使用的实际/特定模型 exposure，但来源没有在同一处解释。
- 区分性检查：需要 Llama 2 固定 run 的 sampler/step accounting 或作者勘误。
- 处理：保留模型卡 2.0T 作为该代主记录，同时注明跨论文摘要 1.8T；不擅自统一。`open`。

### Llama 4 overall >30T、Scout ~40T、Maverick ~22T

- 博客说 overall data mixture >30T；模型卡给出两个模型各自 pretraining token count。
- 第一个差异：overall mixture/corpus 或家族描述，与单模型 sampled exposure 可能不是同一对象。
- 区分性检查：需要 shared corpus size、各模型 sampler manifest、重复 exposure 和 modality token definitions。
- 处理：分别记录，不把 40T+22T 相加，也不把 >30T 当两模型共同 exposure。`open`。

### Llama 3.1 15T+ 与 paper 15.6T

- 模型卡 `15T+` 是 rounded family-level summary；paper 15.6T 明确对应 405B。
- 两者可兼容，但不能由 405B 数字推断所有 8B/70B checkpoint 的精确 exposure。
- 处理：405B 用 15.6T，家族其他模型保留 `15T+/unknown exact`。`supported`。

## 目前仍不知道什么

- `[未知 | open]` Llama 3/4 的完整 source list、逐 source rights、dataset manifest 和训练顺序。
- `[未知 | open]` 公开 token 数是否等于 non-padding loss tokens；packing/mask 只在 Llama 3 部分披露。
- `[未知 | open]` Llama 4 text/image/video tokenization 与模态 mixture。
- `[未知 | open]` Llama 4 pretraining dedup/filter/decontam 是否沿用 Llama 3 pipeline。
- `[未知 | open]` Llama 4 mid-training 的 long-context tokens、长度 curriculum 和真实/合成比例。
- `[未知 | open]` Behemoth teacher distillation 在 Scout/Maverick 的具体训练阶段、target 与 token budget。

## 哪些经验可以借鉴，哪些不能直接照搬

### 可迁移

- `[综合判断 | supported]` 去重必须记录 URL/document/line 粒度和 scope；“global dedup”不能概括 bucketed line dedup。
- `[综合判断 | supported]` long-context extension 要同时检查长任务达标和短任务恢复，避免只优化最大长度。
- `[综合判断 | supported]` 用 small-model scaling/annealing probe 决定 mixture 时，应专门检查 proxy-to-target 的规模反转。
- `[综合判断 | supported]` 模态 token、文本 token 与 post-training examples 必须分阶段、分 tokenizer 报告。

### 不可外推

- Llama 3 的 final mix 不是跨模型通用最优配方，也不是每个 step 的固定采样分布。
- 8B annealing 的数据收益不能直接外推到 405B；论文给出了反例。
- Llama 4 的 10M context 不等价于 10M 长训练样本广泛覆盖。
- 开放权重与社区许可不等价于公开训练数据或可复现训练顺序。
- 不能从 Llama 4 benchmark 提升反推 image/video 或 code/math mixture。

## 读完后应该能回答的问题

掌握本案例后应能：

1. 把 Llama 3 的 15.6T 拆成 initial、800B long-context 和 40M anneal 的嵌套/阶段关系，而不是相加。
2. 解释为什么 0.1% synthetic long-context data 属于 SFT mix，不能填进 pretraining matrix。
3. 设计区分 Llama 4 40T multimodal tokens 与 Llama 3 15.6T text tokens 的 accounting contract。
4. 说明 8B candidate annealing probe 为什么可能在 405B 失效。

自测：如果某候选 math corpus 在 8B/40B-token annealing probe 上显著改善 MATH，但 405B 无提升，你会检查 exposure、contamination、baseline competence、optimizer 和 eval 的哪些差异？

## 来源与建议阅读位置

- [Llama 2 official model card](https://github.com/meta-llama/llama-models/blob/main/models/llama2/MODEL_CARD.md)：为什么读：固定 2T、2022-09 cutoff、不含 Meta user data 与阶段边界；重点读 Training Data。
- [The Llama 3 Herd of Models, arXiv:2407.21783v3](https://arxiv.org/abs/2407.21783)：为什么读：本家族最完整的数据机制来源；重点读 §3.1、§3.2.1、§3.4、§4.3.4、§5.1。
- [Llama 3.1 official model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)：为什么读：固定 15T+、2023-12 cutoff、支持语言和 post-training synthetic 数量；不要用模型卡替代论文的 pipeline 细节。
- [Llama 3.2 official model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md)：为什么读：区分 pretraining data 与剪枝后 token-level distillation；重点读 Training Data。
- [Llama 4 official release, 2025-04-05](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)：为什么读：定位 early fusion、200 languages、overall >30T、mid-training 与 post-training difficulty curriculum。
- [Llama 4 Maverick official model card](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E)：为什么读：固定 Scout/Maverick 的 40T/22T、2024-08 cutoff、source 类别与 license；重点读 Model Information、Training Data。
- [Meta Llama official repository](https://github.com/meta-llama/llama-models)：为什么读：固定家族发布日期、context、tokenizer、model card 与 license 路径；它提供 inference utilities，不等于完整训练代码。
