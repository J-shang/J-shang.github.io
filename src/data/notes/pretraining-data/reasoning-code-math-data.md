---
title: "Reasoning、Code 与 Math 数据"
description: "区分预训练文档、SFT traces、RL rollouts、verifier 与 benchmark seeds，避免跨阶段重复核算。"
topic: "pretraining-data"
section: "cross-cutting"
slug: "reasoning-code-math-data"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 82
readtime: 17
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/reasoning-code-math-data.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/reasoning-code-math-data.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:4be63548054ca602343701ba455b7beead174f8684a84c5f8ebd56872c65e5bc"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`unified-framework` -->
> 主要参考案例：DeepSeek、Qwen、Apple AFM、ByteDance Seed、GLM、MiniMax、NVIDIA Nemotron；开放控制：AI2 OLMo/Dolma

## 这篇专题要回答什么

“reasoning data”可能是数学网页、代码仓库、教材式合成文本、SFT chain-of-thought、RL问题、每题多条rollout，或仅用于reward的tests。它们对模型施加的监督不同，统计单位也不同。

```text
P0/P2: documents / code / problems / explanations
A1: prompt-response or trace
A2: prompt -> K rollouts -> rewards -> selected/weighted tokens
A3: environment state -> actions/tool calls -> outcome
```

如果把这些相加，会同时重复计算prompt、candidate与accepted trajectory，并掩盖benchmark seed泄漏。

## 统一任务图

每个任务实例至少表示为：

$$
x=(s,p,e,v),
$$

其中 $s$ 是最早source/seed，$p$ 是problem/spec，$e$ 是environment与工具版本，$v$ 是verifier。训练记录再附：policy checkpoint、rollouts、reward和loss mask。

```text
source/seed
  -> problem/spec
  -> prompt representation
  -> policy@checkpoint generates trajectory
  -> verifier@version returns evidence
  -> selection/advantage
  -> loss-bearing tokens
```

## 分阶段数据 contract

| 阶段 | 主单位 | 关键分母 | 常见错误 |
|---|---|---|---|
| `P0/P2` | document/repo/problem tokens | unique vs sampled vs loss | 把QA格式全称为SFT |
| `A1` | examples/packed sequences | response loss tokens | 用example数比较不同长度trace |
| `A2` | prompts/rollouts/tokens | submitted、generated、verified、consumed | 把K rollouts都算独立任务 |
| `A3` | trajectories/turns/tool calls | completed/accepted、environment steps | 忽略失败分支与状态版本 |

RL每轮的nominal rollout tokens为：

$$
T_{\text{rollout}}=\sum_{i=1}^{N_{\text{prompt}}}\sum_{k=1}^{K_i}L_{ik},
$$

真正用于policy loss的tokens还取决于截断、mask、advantage和replay。

## verifier taxonomy

| verifier | 适用对象 | 建立的证据 | 盲点 |
|---|---|---|---|
| exact answer | arithmetic/closed QA | final answer相同 | 错误过程碰巧正确、格式 |
| symbolic/proof checker | math/formal | proof在固定系统通过 | formalization错误、library泄漏 |
| compiler + unit tests | code | tests覆盖行为通过 | hidden spec、弱tests、依赖漂移 |
| sandbox/environment | agent/code | 状态转移与outcome | flaky service、奖励捷径 |
| execution trace rules | tool use | schema/sequence 约束 | 语义正确性不完整 |
| LLM judge/RM | open reasoning | learned preference | self-bias、verbosity/style、drift |
| human review | ambiguous tasks | contextual judgment | 成本、分歧、版本化困难 |

“verifiable”必须写被验证property，不是全局正确标签。

## 厂商公开资料实际告诉了我们什么

| 案例 | P0/P2 | A1/A2/A3 | 关键边界 |
|---|---|---|---|
| [DeepSeek](/topics/pretraining-data/deepseek-data-practices/) | code/math/remix、continued/context | R1 reasoning RL、specialist distillation、agent provenance | base、CoT、RL rollout不能合并 |
| [Qwen](/topics/pretraining-data/qwen-data-practices/) | Qwen2.5-Coder/Math specialist CPT | off/on-policy distillation、execution/answer rewards | specialist tokens与post examples分列 |
| [Apple AFM](/topics/pretraining-data/apple-foundation-models-data-practices/) | 3B math QA+14B math pages；code/math upweight | math evolution、12K executed code triplets、rule/RM RL | tests只验证覆盖行为 |
| [Seed-Coder](/topics/pretraining-data/bytedance-seed-doubao-data-practices/) | 5T regular+1T continued；GitHub/commits/web/math | 3M SFT、20K DPO、LongCoT/RL | repo/group license与benchmark exclusion |
| [GLM](/topics/pretraining-data/glm-data-practices/) | 500B repo code+500B synthetic reasoning | reasoning/agent experts→self-distill；async environments | difficulty/length filter与environment版本 |
| [MiniMax](/topics/pretraining-data/minimax-data-practices/) | STEM/code/books/reasoning CPT | verifiable RL、long-output curriculum、GenRM monitoring | reward length bias与synthetic collapse |
| [Nemotron](/topics/pretraining-data/nvidia-nemotron-data-practices/) | math/code/SFT-style phase data | code/search/tool/GPU/RTL factories、MOPD | benchmark concept seeds与teacher lineage |
| [OLMo/Dolma控制](/topics/pretraining-data/ai2-olmo-dolma-data-practices/) | open source/version/order/logs | 不代表完整reasoning RL | 校准P0 artifact与eval可复现性 |

## 统一比较会丢掉哪些重要差异

- code与math的verifier可执行性不同，不能用一个“pass rate”横向排名；
- on-policy数据分布随checkpoint变，静态manifest需加policy/version与iteration；
- repo任务的group是repository/commit lineage，数学任务可能是problem family；
- OLMo开放控制没有生产级agent RL，只能校准source/order/logs，不校准rollout economics。

## difficulty 与curriculum

常见difficulty proxy：pass@$K$、response length、verifier margin、human label、model score。它们都可能与domain/style混杂。

若每题成功概率估计为 $\hat p_i$，保留中等难度窗口 $a<\hat p_i<b$ 会避免全易/全难，但需报告：

- $K$与sampling temperature；
- policy checkpoint；
- 各domain retention；
- 题族去重后的分布；
- 下一checkpoint上的迁移。

MiniMax的length-based curriculum与GLM的response-length filter只是特定recipe关联，不是difficulty的精确定义。

## code data 的四级去重

```text
file exact/hash
  -> file/function near dedup
  -> repository/fork lineage
  -> task/spec/benchmark semantic overlap
```

只做文件MinHash无法阻止同一issue、solution或fork跨split。agent/SWE任务还需固定base commit、tests、dependencies、tool image和网络资源。

## math data 的祖先图

同一题可被翻译、改数值、反转、扩展或生成CoT。最早seed应作为group：

```text
original theorem/problem
  -> translation
  -> parameterized variants
  -> solutions/CoT
  -> preference/rollout records
```

train/eval在派生后随机split会造成semantic leakage。需要problem-family clustering、symbolic canonicalization与人工抽查。

## benchmark contamination 的三层

1. **content layer**：题面/答案/代码直接或近似出现；
2. **derivative layer**：从benchmark train/test概念生成variants；
3. **decision layer**：反复看benchmark结果选择filter、mixture、teacher或checkpoint。

第三层即使没有文本重叠也会过拟合。应保存experiment registry：每次decision看过哪些eval、何时冻结clean set。

## RL data 需要分别记录哪些对象

推荐按domain $d$ 记录：

| 指标 | 含义 |
|---|---|
| submitted prompts | scheduler请求数 |
| completed prompts | 至少一rollout完成 |
| generated rollouts/tokens | policy产出 |
| verifier pass/fail/error | reward可用性 |
| accepted/consumed | 进入replay/gradient |
| success by length | 检查verbosity reward |
| unique problem families | 防多variant虚增 |

异步系统还需记录staleness：trajectory由哪个policy生成、被哪个policy更新。

## validation 与归因实验

最低四组：

1. no specialist data；
2. natural code/math documents；
3. synthetic problems/solutions，fixed tokens；
4. verified RL，在相同base上追加。

分开报告base loss、pass@k、sample efficiency、human preference和clean held-out。若SFT/RL改变输出长度，不能只看exact-match均值。

### verifier ablation

固定生成candidates，比较：exact、tests、judge、ensemble与human audit。报告false accept/negative taxonomy，而不只报告最终benchmark。

### seed provenance ablation

比较benchmark-derived concepts与domain-matched independent seeds，在clean out-of-family eval上测迁移，区分task-format specialization和广泛能力。

## 看似矛盾的说法怎样区分

### more CoT improves reasoning vs visible CoT causes imitation

- 首个差异：阶段、teacher质量、是否有outcome/process验证。
- 检查：answer-only、verified CoT、unverified CoT fixed-loss-token对照。

### execution-verified code is correct vs tests可被gaming

- 首个差异：test coverage/spec completeness。
- 检查：hidden/property/mutation tests、跨环境复现与人工bug taxonomy。

### hard prompts更有效 vs过难产生噪声

- 首个差异：policy能力、pass-rate窗口和verifier error。
- 检查：按difficulty bins画learning gain/accepted token，跟踪跨checkpoint迁移。

## 不可外推

- 不把P0 math/code tokens、SFT examples和RL rollouts相加；
- 不把unit-test pass称为完整正确；
- 不把response length当真实difficulty；
- 不把benchmark train concepts生成的数据称为clean independent source；
- 不把某家pass@K、domain ratio或prompt count当通用最优；
- 不根据最终reasoning benchmark反推 data recipe。

## 最少需要保存哪些 manifest 信息

```yaml
problem_family: ...
source_or_seed: ...
license: ...
stage: A2
policy_checkpoint: ...
environment_image: ...
prompt_id: ...
rollout_id: ...
verifier_version: ...
reward_components: ...
completion_status: ...
loss_token_count: ...
benchmark_overlap: exact|near|semantic|none|unknown
```

## 读完后应该能回答的问题

读者应能：

1. 把document、example、prompt、rollout、trajectory和loss token分开；
2. 解释verifiable只对特定property成立；
3. 为repo与math problem选择正确split group；
4. 识别content/derivative/decision三层污染；
5. 设计answer-only/verified/unverified CoT对照；
6. 为异步RL记录policy staleness。

推理题：10K prompts各采8条平均2K-token rollouts，75%完成，完成中40%通过，训练只消费通过rollouts的80% response tokens。分别计算nominal、completed、passed与consumed tokens；还缺哪些mask信息才能得到loss tokens？

## 来源与建议阅读位置

1. [DeepSeek](/topics/pretraining-data/deepseek-data-practices/)与[Qwen](/topics/pretraining-data/qwen-data-practices/)：比较specialist pretraining、reasoning RL与distillation阶段。
2. [Apple AFM](/topics/pretraining-data/apple-foundation-models-data-practices/)：读math evolution、code execution、tool data与RLOO rewards。
3. [ByteDance Seed](/topics/pretraining-data/bytedance-seed-doubao-data-practices/)：读GitHub/commit pipeline、SFT/DPO/RL和benchmark exclusion。
4. [GLM](/topics/pretraining-data/glm-data-practices/)、[MiniMax](/topics/pretraining-data/minimax-data-practices/)、[NVIDIA Nemotron](/topics/pretraining-data/nvidia-nemotron-data-practices/)：读difficulty、异步环境、reward bias和seed provenance。
5. [OLMo/Dolma](/topics/pretraining-data/ai2-olmo-dolma-data-practices/)：开放P0 source/config/order/logs控制。
6. [Filtering/dedup/decontamination](/topics/pretraining-data/filtering-dedup-decontamination/)：用于祖先group与污染审计。
