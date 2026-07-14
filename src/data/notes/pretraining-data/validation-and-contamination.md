---
title: "Validation 与污染：从 held-out 到决策层过拟合"
description: "同时审计内容重叠、祖先或衍生重叠，以及反复用 benchmark 做数据决策带来的 decision exposure。"
topic: "pretraining-data"
section: "cross-cutting"
slug: "validation-and-contamination"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 84
readtime: 15
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/validation-and-contamination.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/validation-and-contamination.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:b4a10c6e145635bced1d418aaa6b8fada37bd313ce58cd4ce90553e5345a87ed"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`unified-framework`
> 案例锚点：AI2 OLMo/Dolma、Apple AFM、Meta Llama、Anthropic Claude、Google Gemini/Gemma、NVIDIA Nemotron、Qwen、OpenAI

## Motivating problem

训练集没有benchmark题面，并不保证评测独立：题目可能有近似/翻译/衍生版本，或研究者反复用同一benchmark选择filter、mixture和checkpoint。后者没有内容重叠，却仍会使最终报告过拟合评测决策。

```text
content overlap
derivative/ancestor overlap
decision exposure
```

本专题把validation set视为有版本、时间、来源、split、tokenizer和决策访问历史的数据产品，而不是一个名称。

## validation contract

| 字段 | 必须说明 |
|---|---|
| identity | dataset/revision/config/split/hash |
| source/time | 来源、采集窗口、冻结日期、未来数据策略 |
| split unit | document/repo/problem family/image/video/user/session |
| isolation | 与train、synthetic ancestors、teacher/judge的关系 |
| tokenizer | 名称/version、sample/token count |
| metrics | 分母、micro/macro/per-domain、CI/seed |
| protocol | prompt/template、shots、decode、judge/version |
| decision log | 哪些实验/人看过结果、何时锁箱 |

## 三层污染

### content layer

exact、n-gram、substring、MinHash、embedding或code AST重叠。它回答“内容有多像”。

### derivative layer

train/eval共享更早祖先：同一book章节、repo fork、problem seed、image、翻译或teacher-generated variant。最终文本可能完全不同。

### decision layer

benchmark反复用于：训练quality classifier、选择data bucket、调mixture、选teacher/prompt、决定checkpoint或发表阈值。即使train rows不含benchmark，评测已进入研发反馈环。

## 匹配规则与分母

对eval item $e$，定义是否命中train/ancestor：

$$
I(e)=\mathbf{1}[\exists x\in D_{train}:\operatorname{match}(e,x)\ge\tau].
$$

污染率：

$$
r=\frac{\sum_e I(e)}{|E|}.
$$

必须报告match family、threshold、normalization和item分母。document removal率、eval item命中率和token重叠率不是同一指标。

clean score与full score并列：

$$
\Delta_{contam}=Q_{full}-Q_{clean}.
$$

clean subset更小，需给CI，不能只比较点估计。

## 案例恢复

| 案例 | validation/污染锚点 | 优点 | 边界 |
|---|---|---|---|
| [OLMo/Dolma](/topics/pretraining-data/ai2-olmo-dolma-data-practices/) | data版本、configs、logs/checkpoints、eval code可追 | 可关联训练状态与评测 | loss-token与全部决策访问仍需日志 |
| [Apple AFM 2024](/topics/pretraining-data/apple-foundation-models-data-practices/) | 811 benchmarks，4–13 gram碰撞删整doc，common threshold=1000 | operational rule明确 | revision/semantic overlap/2025继承未知 |
| [Meta Llama](/topics/pretraining-data/meta-llama-data-practices/) | dedup/decontam、long阶段短能力恢复、data ablation | stage checkpoint诊断 | Llama4详细合同收缩 |
| [Anthropic Claude](/topics/pretraining-data/anthropic-claude-data-practices/) | canary、transcript incident、targeted mitigation | machine-readable tracking与事件响应 | token/mix/pipeline多未知 |
| [Google WebLI/Gemma](/topics/pretraining-data/google-gemini-gemma-data-practices/) | image-text pair accounting与decontam；JEST reference selection | 多模态与decision exposure案例 | WebLI/JEST不等于Gemini production lineage |
| [Nemotron-CC](/topics/pretraining-data/nvidia-nemotron-data-practices/) | 50B bucket probes、1T/15T、synthetic ablation；v1明确未decontam | validation-calibrated quality buckets | 9-task feedback本身形成decision exposure |
| [Qwen](/topics/pretraining-data/qwen-data-practices/) | clean-subset contamination、threshold proxy | 显式分clean结果 | 完整生产overlap manifest不足 |
| [OpenAI](/topics/pretraining-data/openai-data-practices/) | GPT-3 contamination与后续system-card高层边界 | 历史定量锚点 | 最新训练数据/协议收缩 |

## 框架的信息损失

- Apple给pretraining removal规则，Anthropic canary监测事件；两者不是同一防线。
- Nemotron用benchmark校准quality bucket，既是validation优势也是decision exposure。
- WebLI多模态去污染的split unit不同于text benchmark。
- OLMo开放控制能重跑protocol，不保证研究者从未看过某结果。

因此不能用单一“去污染：是/否”列恢复所有案例。

## held-out构造

### 时间隔离

滚动web训练需冻结validation来源/日期。future set要防止后续训练回流；一旦用于选配方，它从final lockbox降级为development set。

### group split

按最早共享祖先分组：

```text
document family / domain
repository + forks + commits
problem seed + translations + variants
image/video source + captions/QA
user/session/trajectory environment
```

random row split通常会泄漏。

### validation规模与domain

每域报告samples和tokens。micro被大域支配；macro为：

$$
Q_{macro}=\frac{1}{K}\sum_{k=1}^K Q_k.
$$

若各域业务重要性不同，另报预注册weighted metric，不用事后改权重掩盖退化。

## 数据质量proxy与目标泄漏

用benchmark训练document selector或挑bucket，会显式优化目标。Nemotron 50B probes、Qwen threshold proxy、Apple/Google的targeted selection都应记录：

- selector target suite；
- evaluation suite是否重叠；
- proxy model/scale/seed；
- tested recipes总数；
- held-out disjoint suite。

最佳实践是三层：development targets、disjoint transfer suite、一次性lockbox。

## canary与事件响应

canary是machine-readable标记，用于检测数据意外进入训练/评测流。最小字段：

```text
canary_id, owner, scope, inserted_at
expected_absence/presence
scanner_version, scan_time, result
incident_ticket, mitigation, affected checkpoints
```

canary只检测标记/衍生匹配，不替代一般语义污染审计。Anthropic transcript事件说明发现后还需影响范围、targeted mitigation与checkpoint追踪。

## 多模态与agent污染

图像要做exact/perceptual hash并按source image group；caption不同不能视为独立。视频按source video/时间邻近clip分组。OCR按原document revision。

agent/code要固定repo commit、tests、container、tool schema和database state。若训练任务从benchmark repo生成，即使issue文本改写，也属于ancestor overlap。

## judge contamination

LLM-as-judge可能：

- 训练见过benchmark/reference；
- 偏好自身风格/长度；
- 随版本漂移；
- 与被评模型共享teacher lineage。

固定judge snapshot、blind order、长度控制、human calibration与disagreement subset。judge分数不是ground truth。

## 统计与实验选择

重复试验后只报告最佳结果会产生winner's curse。记录所有配方、seed、stopping和checkpoint selection。对差值给bootstrap/paired CI；多指标选择需预注册primary metric或做校正。

数据ablation必须固定model、tokenizer、token budget、optimizer、compute和eval。否则只能写recipe association。

## 表面冲突与区分性检查

### exact overlap低 vs semantic leakage高

- 首个差异：match relation。
- 检查：exact+n-gram+embedding+ancestor group，人工审计disagreement。

### decontam降低污染 vs删掉常用短语伤数据

- 首个差异：collision threshold/common phrase policy。
- 检查：precision/recall fixture、retained-domain分布、clean score与data loss trade-off。

### clean subset分数稳定 vs决策仍过拟合

- 首个差异：content isolation与decision isolation。
- 检查：未触碰lockbox、完整experiment registry与disjoint future suite。

### canary未命中 vs数据安全

- canary覆盖有限。
- 检查：一般PII/license/provenance扫描、语义/祖先审计与抽样人工检查。

## 最小报告模板

```yaml
eval_id: dataset@revision
frozen_on: YYYY-MM-DD
source_window: ...
split_group: ...
tokenizer: ...
per_domain_counts: ...
matchers:
  - {type: ngram, n: 4-13, threshold: ...}
  - {type: semantic, model: ..., threshold: ...}
ancestor_graph_version: ...
full_score: ...
clean_score: ...
micro_macro_per_domain: ...
decision_access_log: ...
judge_and_prompt: ...
```

## 不可外推

- 不把“已去污染”当无污染；
- 不把exact clean当semantic/ancestor clean；
- 不把development benchmark当final validation；
- 不只报global loss/score；
- 不把internal set名称当可审计合同；
- 不用最新judge覆盖历史结果而不记录版本。

## 掌握标准与自测

读者应能：

1. 区分content、derivative和decision污染；
2. 解释Apple 4–13 gram规则与common threshold；
3. 说明Nemotron bucket probe为何同时是优势和decision exposure；
4. 为repo/image/problem选择group split；
5. 设计development/transfer/lockbox三层；
6. 同时报micro、macro、per-domain和clean score。

推理题：1000条eval中120条n-gram命中、其中80条semantic也命中；另有60条仅semantic命中。若按任一matcher移除，clean subset多大？为什么仍不能说它在ancestor与decision层clean？

## 来源与阅读路径

1. [OLMo/Dolma](/topics/pretraining-data/ai2-olmo-dolma-data-practices/)：开放data/config/order/logs/eval控制。
2. [Apple AFM](/topics/pretraining-data/apple-foundation-models-data-practices/)：读811 benchmarks规则、stage eval与2025继承边界。
3. [Anthropic Claude](/topics/pretraining-data/anthropic-claude-data-practices/)：读canary、transcript incident和mitigation。
4. [NVIDIA Nemotron](/topics/pretraining-data/nvidia-nemotron-data-practices/)与[Qwen](/topics/pretraining-data/qwen-data-practices/)：比较proxy selection、明确未decontam和clean subset。
5. [Google Gemini/Gemma](/topics/pretraining-data/google-gemini-gemma-data-practices/)、[Meta Llama](/topics/pretraining-data/meta-llama-data-practices/)、[OpenAI](/topics/pretraining-data/openai-data-practices/)：补多模态、长阶段与历史披露差异。
6. [Validation design](/topics/pretraining-data/validation-design/)：主知识地图的held-out与诊断合同。
