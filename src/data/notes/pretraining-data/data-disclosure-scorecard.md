---
title: "数据披露证据卡：逐字段等级，而非透明度总分"
description: "按模型代际、训练阶段和字段记录 D0–D4 证据形状，避免用组织级单一分数掩盖未知项。"
topic: "pretraining-data"
section: "methodology"
slug: "data-disclosure-scorecard"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 71
readtime: 22
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/disclosure-scorecard.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/disclosure-scorecard.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:4abaaf75e6465dafd348f58185a888cba94945420076ae7f67fb7ea489c5160b"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`unified-framework` -->
> 统计主键：`model generation × training stage × artifact version`
> 开放控制：AI2 OLMo/Dolma、Apple OpenELM

## 这篇专题要回答什么

把厂商压成一个“透明度8/10”会隐藏证据形状：Mistral可能权重/配置开放但训练数据未知；Apple可能不给权重，却给阶段token和去污染规则；NVIDIA可能发布大量data/pipeline资产，但生产模型仍含private slices。

因此本卡不求和、不排名，只回答：哪个字段能在什么版本和范围内被检查。

```text
organization-level reputation
  != generation-stage evidence
  != field-level reproducibility
```

## 等级定义

| 等级 | 本卡中的操作性含义 |
|---|---|
| `D0` | 无可用一手字段，或只有营销能力描述 |
| `D1` | 官方高层taxonomy/方向，缺定义、分母或版本化细节 |
| `D2` | 技术报告给定量规模、阶段、pipeline或实验，但关键资产未开放 |
| `D3` | 固定权重/代码/config/部分data或logs可检查，仍无法闭合全链 |
| `D4` | data/manifest、处理代码、order、metrics与eval基本可追踪 |

等级只评估披露证据，不评估模型能力、数据质量、合法性或安全性。

## 字段轴

| 轴 | 核心问题 |
|---|---|
| identity/time | generation、stage、release/cutoff、artifact revision是否固定 |
| source/rights | source snapshot、许可、opt-out、删除传播是否可追 |
| accounting | raw/unique/sample/loss tokens及tokenizer/packing是否闭合 |
| pipeline | parse/filter/LID/dedup/decontam与阈值/计数是否可查 |
| mixture/order | sampling、replay、curriculum、seed与sample order是否可查 |
| synthetic/post | teacher、prompt、verifier、SFT/preference/RL lineage是否可查 |
| validation | held-out/time/split、micro/macro/per-domain、clean/ablation是否可查 |
| artifacts | weights、data、code、configs、logs、checkpoints、eval assets |

## 证据卡

以下是截至核查日的首轮索引；复杂caveat以厂商笔记为准。`D2→D3`表示同一行不同子字段/资产处于区间，不是小数分。

| 锚点（阶段） | identity/time | source/rights | accounting | pipeline | mix/order | synth/post | validation | artifacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [AI2 OLMo 1/2 + Dolma](/topics/pretraining-data/ai2-olmo-dolma-data-practices/) `P0/P2` | D4 | D4 | D3 | D4 | D4 | D2 | D4 | D4 |
| [Apple OpenELM](/topics/pretraining-data/apple-foundation-models-data-practices/) `P0/A1/A2` | D4 | D3 | D3 | D3 | D3 | D3 | D3 | D4 |
| [OpenAI GPT-3](/topics/pretraining-data/openai-data-practices/) `P0` | D2 | D1 | D2 | D1 | D2 | D0 | D2 | D0 |
| [OpenAI o1/GPT-5.6](/topics/pretraining-data/openai-data-practices/) `P*/A*` | D1 | D1 | D0 | D0-D1 | D0 | D1 | D1-D2 | D0 |
| [Meta Llama 3/3.1](/topics/pretraining-data/meta-llama-data-practices/) `P0/P3/A*` | D3 | D1 | D2 | D2 | D2 | D2 | D2 | D3 |
| [Meta Llama 4](/topics/pretraining-data/meta-llama-data-practices/) `P0/P1/P3/A*` | D3 | D1 | D1-D2 | D1 | D1 | D1-D2 | D1-D2 | D3 |
| [DeepSeek V3/R1](/topics/pretraining-data/deepseek-data-practices/) `P0/P2/A2` | D3 | D1 | D2 | D2 | D2 | D2-D3 | D2 | D3 |
| [Qwen2.5/Qwen3](/topics/pretraining-data/qwen-data-practices/) `P0/P2/A*` | D3 | D1 | D2 | D2 | D2 | D2-D3 | D2 | D3 |
| [Xiaomi MiMo](/topics/pretraining-data/mimo-data-practices/) `P0…A3` | D3 | D1 | D2 | D2 | D2 | D2-D3 | D2 | D3 |
| [MiniMax Text-01/M1/M2](/topics/pretraining-data/minimax-data-practices/) `P0…A3` | D2-D3 | D1 | D2 | D2 | D2 | D2 | D2-D3 | D2-D3 |
| [Moonshot Kimi K2/K2.5](/topics/pretraining-data/moonshot-kimi-data-practices/) `P0…A3` | D3 | D1 | D2 | D1-D2 | D2 | D2-D3 | D2 | D3 |
| [Z.ai GLM-4.5/5](/topics/pretraining-data/glm-data-practices/) `P0…A3` | D3 | D1 | D2 | D2 | D2 | D2-D3 | D2 | D3 |
| [StepFun](/topics/pretraining-data/stepfun-data-practices/) `P0…A*` | D2-D3 | D1 | D2 | D2 | D2 | D2 | D2 | D2-D3 |
| [Microsoft MAI/Phi](/topics/pretraining-data/microsoft-mai-phi-data-practices/) `P0/P1/P3/A*` | D2-D3 | D1 | D2 | D1-D2 | D2 | D2 | D2 | D2-D3 |
| [Anthropic Claude](/topics/pretraining-data/anthropic-claude-data-practices/) `P*/A*` | D1-D2 | D1 | D0-D1 | D0-D1 | D0 | D1-D2 | D2 | D0 |
| [Google Gemini](/topics/pretraining-data/google-gemini-gemma-data-practices/) `P*/A*` | D2 | D1 | D0-D1 | D1 | D0-D1 | D1 | D2 | D0-D1 |
| [Google Gemma/WebLI](/topics/pretraining-data/google-gemini-gemma-data-practices/) `P0/P3/A*` | D3 | D2 | D2 | D2 | D1-D2 | D2 | D2 | D3 |
| [xAI Grok](/topics/pretraining-data/xai-grok-data-practices/) `P*/A*` | D1-D3 | D1 | D0-D1 | D0-D1 | D0-D1 | D1-D2 | D1-D2 | D1-D3 |
| [ByteDance Seed](/topics/pretraining-data/bytedance-seed-doubao-data-practices/) `P0…A3` | D2-D3 | D1 | D2 | D2 | D2 | D2-D3 | D2 | D2-D3 |
| [Tencent Hunyuan](/topics/pretraining-data/tencent-hunyuan-hy-data-practices/) `P0…A3` | D2-D3 | D1 | D2 | D2 | D2 | D2 | D2 | D2-D3 |
| [Baidu ERNIE](/topics/pretraining-data/baidu-ernie-data-practices/) `P0…A*` | D2-D3 | D1 | D1-D2 | D2 | D1-D2 | D2 | D2 | D2-D3 |
| [Huawei PanGu/openPangu](/topics/pretraining-data/huawei-pangu-openpangu-data-practices/) `P0…A*` | D2-D3 | D1 | D2 | D2 | D2 | D1-D2 | D2 | D2-D3 |
| [Amazon Nova](/topics/pretraining-data/amazon-nova-forge-data-practices/) `P*/A*` | D2 | D1 | D0-D1 | D1 | D1-D2 | D1-D2 | D2 | D1 |
| [Amazon Nova Forge](/topics/pretraining-data/amazon-nova-forge-data-practices/) customer `P2…A2` | D3 | D0-D1 | D1-D2 | D1-D3 | D3 | D2-D3 | D2-D3 | D3 control / D0 data |
| [NVIDIA Nemotron-CC](/topics/pretraining-data/nvidia-nemotron-data-practices/) corpus | D4 | D2-D3 | D3 | D4 | D3 | D3-D4 | D3 | D4 |
| [NVIDIA Nemotron 3](/topics/pretraining-data/nvidia-nemotron-data-practices/) `P0…A3` | D3-D4 | D1-D2 | D3 | D3 | D3 | D3 | D3 | D3-D4 |
| [Mistral 7B→Large3](/topics/pretraining-data/mistral-ai-data-practices/) `P0…A3` | D3 | D0-D1 | D0 | D0 | D0-D1 | D1 | D1-D2 | D3 |
| [Apple AFM 2024](/topics/pretraining-data/apple-foundation-models-data-practices/) `P0…A2` | D2 | D1-D2 | D2 | D2 | D2 | D2 | D2 | D2 |
| [Apple AFM 2025](/topics/pretraining-data/apple-foundation-models-data-practices/) `P0…A3` | D2 | D1-D2 | D2 | D2 | D2 | D2 | D2 | D2 |
| [Apple AFM 3](/topics/pretraining-data/apple-foundation-models-data-practices/) `P*/A*` | D1-D2 | D1 | D0 | D0-D1 | D1 | D1 | D1-D2 | D1 |

## 如何读证据形状

### 开放控制不是全满分

OLMo/Dolma接近D4，但loss-token、某些下游post-training或rights解释仍可能不闭合。D4表示在声明范围内基本可追，不表示没有偏差或许可争议。

OpenELM的公开framework很强，但dataset由多个上游集合组成，cross-source order/decontam和每个上游rights仍需按revision核对。

### 开放权重与开放数据是不同轴

Mistral多代weights/config为D3，P0 source/accounting/pipeline却为D0/D1。把两者平均成“D2”会删除最重要的研究结论。

类似地，openPangu、Grok-1、Gemma等也需逐字段判断。

### 报告详细但资产不开放

Apple 2024/2025给D2级token、pipeline、mixture与实验，却不开放production corpus/order/logs。其科学问题可以更具体，但不能完整复现。

### 发布数据不等于生产闭合

Nemotron-CC的corpus/pipeline接近D4；Nemotron 3模型仍含private third-party/NVIDIA slices。open data可作为大部分recipe锚点，不等于full production closure。

## 不能求和的原因

设字段等级为 $d_j$。禁止计算：

$$
S=\sum_j w_jd_j.
$$

原因：

1. 等级是ordinal，不保证D4-D3与D2-D1距离相同；
2. 字段不可替代，开放weights不能补source rights；
3. 权重取决于研究问题，license审计与optimizer复现关注轴不同；
4. 一个总分会掩盖D0关键字段；
5. 版本更新会让总分产生虚假可比趋势。

需要决策时，使用门槛向量。例如可复现data ablation要求：

```text
identity>=D3
accounting>=D3
pipeline>=D3
mix/order>=D3
validation>=D3
```

而只运行权重的门槛可能只要求artifacts>=D3。

## 与具体研究问题映射

| 问题 | 最低字段 |
|---|---|
| token budget比较 | identity、accounting、tokenizer均≥D2 |
| source/rights审计 | source/rights与manifest≥D3 |
| filter因果复现 | pipeline、order、accounting、validation≥D3 |
| synthetic lineage | source、synthetic/post、accounting≥D3 |
| contamination结论 | pipeline与validation≥D2，最好有clean manifest D3 |
| production full closure | 所有关键轴D3/D4且private slices有边界 |

## 版本与更新规则

每个cell应伴随内部record：

```yaml
organization: ...
family_generation: ...
stage: ...
field: accounting
grade: D2
checked_on: 2026-07-14
artifact: report@date-or-commit
claim_scope: ...
evidence: ...
unknowns: ...
next_trigger: new report/card/data revision
```

新代际追加新行，不覆盖历史。若网页更新，保存绝对日期和变更内容；API rolling model无法映射checkpoint时，identity降级或标`rolling/unknown`。

## 看似矛盾的说法怎样区分

### D3 artifacts vs D0 data

- 不冲突：字段不同。
- 检查：列出weights/config/code/data/manifest/order/logs八项，而非“开源”。

### D4 data vs rights只有D2

- data可下载不等于逐页许可闭合。
- 检查：source-by-source license、opt-out、derivative/teacher obligations。

### 技术报告更长 vs等级更高

- 篇幅不是证据类型。
- 检查：是否给固定version、定义、分母、阈值、资产与可复现实验。

### 最新模型能力更高 vs披露等级更低

- 能力与披露无单调关系。
- 检查：保持两张独立表，禁止用benchmark补字段。

## 维护诊断

每次复核运行：

- 所有行是否有固定generation/stage；
- `D2+`定量字段是否回链一手来源；
- 是否从旧代继承未重述mixture/cutoff；
- artifact链接/许可/模型名是否变更；
- `unknown`是否被营销语言悄悄覆盖；
- scorecard、comparison matrix和厂商笔记是否一致。

## 不可外推

- 不将等级解释为模型/数据质量；
- 不将D4解释为合法、安全或无污染；
- 不给组织永久等级；
- 不在不同stage间取最高等级；
- 不用第三方推测把D0补成D1/D2；
- 不对ordinal cells做平均/排名。

## 读完后应该能回答的问题

读者应能：

1. 解释Mistral为什么可同时artifacts D3、accounting D0；
2. 解释Apple D2报告与OLMo D4资产的区别；
3. 解释Nemotron-CC D4不使Nemotron 3 production data自动D4；
4. 为“复现filter ablation”和“下载权重”写不同门槛向量；
5. 在新代发布时追加行而不覆盖历史；
6. 拒绝计算单一透明度总分。

推理题：模型A在artifacts/pipeline/accounting为D3/D0/D0，模型B为D0/D2/D2。哪个更适合运行本地权重？哪个更适合研究token/mixture？为什么不能说谁“总体更透明”？

## 来源与建议阅读位置

1. [统一方法](/topics/pretraining-data/industry-data-methodology/)：等级定义、字段contract和更新规则的规范来源。
2. [Comparison matrix](/topics/pretraining-data/industry-data-comparison-matrix/)：查看每一行的事实摘要；本卡只评证据形状。
3. [OLMo/Dolma](/topics/pretraining-data/ai2-olmo-dolma-data-practices/)与[OpenELM/Apple AFM](/topics/pretraining-data/apple-foundation-models-data-practices/)：开放控制、D2生产报告与代际收缩对比。
4. [Mistral](/topics/pretraining-data/mistral-ai-data-practices/)：open weights不等于open data的反例。
5. [NVIDIA Nemotron](/topics/pretraining-data/nvidia-nemotron-data-practices/)：open corpus/pipeline与private production slices并存。
6. 其余厂商笔记：所有cell必须回链对应代际/阶段的来源与unknowns，不能只读本卡。
