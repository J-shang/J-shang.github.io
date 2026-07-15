---
title: "Validation 设计与诊断"
description: "从独立性、可比性与决策目标推导 split、指标、checkpoint 选择和 contamination 合同。"
topic: "pretraining-data"
section: "validation"
slug: "validation-design"
date: 2026-07-14
updated: 2026-07-15
order: 60
readtime: 18
source:
  repository: "J-shang/pt-data-learning"
  path: "knowledge-map/05-validation/validation-design.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/knowledge-map/05-validation/validation-design.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:8d98db7f42ba4bd9f0f32c8e250c2ab872309491a44a27cc680c1c1311a2a0ce"
  manifest: "pretraining-data"
  managed: true
---
> 层级：05 Validation
> 状态：`core`
> 初始资料核查截止：2026-07-14
<!-- maintenance: reasoning-path=`constraint-driven derivation` -->
> 证据说明：聚合/配对统计的代数关系为 `verified`；独立性、污染影响和 checkpoint 决策的结论受 split/eval 假设约束

## 这篇笔记帮助你回答什么

Validation 是一份在看结果前冻结的测量合同：它既要对训练分布敏感，又要对目标能力和泄漏风险敏感，并允许把变化归因到具体 domain、数据版本和训练阶段。

## 为什么需要这个概念

如果 validation 与训练共享文档/duplicate cluster，或同一 benchmark 被反复用于挑 filter、mixture、checkpoint，那么分数虽然可计算，却不再支持“对未见数据泛化”的解释。一个可用方案必须同时满足：抽样单元尽可能独立、不同 run 的测量协议可比、指标与决策目标对齐、污染和选择次数可审计。

本笔记从这些约束推出 split unit、三层 suite、聚合、checkpoint 和 contamination contract；不是从现有 benchmark 名单倒推流程。

## 先看一个最小例子

一本书被切成 1000 个 chunks 后随机做 99/1 train/validation split。即使没有完全相同 chunk，章节、角色和长片段仍可能跨集合；validation loss 偏低不能区分真正泛化与近重复记忆。把 split unit 提升到作品/duplicate cluster 能减少该路径，但仍需 cross-split audit。

## 核心定义

本项目把 validation 分为三层：

1. **In-distribution held-out**：从目标训练分布的独立单元留出，监控优化和过拟合。
2. **Diagnostic/domain suites**：固定的分语言/source/体裁/难度集合，用于定位 trade-off。
3. **Downstream/benchmark eval**：测目标能力，但必须单独审计 contamination、评测噪声和决策层过拟合。

三层不能相互替代；一个 validation contract 必须冻结版本、抽样单元、时间边界、去污染、tokenizer、mask、指标、聚合、checkpoint 和决策规则。

## 这些结论依赖哪些前提

| 关系或结论 | 类型与条件 | 置信状态 |
|---|---|---|
| group sets 不相交 | 声明的 group 定义下的 `exact split invariant`；group metadata 错误或语义近重复仍可泄漏 | `verified but insufficient` |
| $L_{\text{train-mix}}=\sum q_kL_k$ | $q$ 与评测聚合权重一致、各 $L_k$ 分母兼容时的 `exact identity` | `verified` |
| 二项 accuracy 的标准误公式 | 样本近似独立同分布、大样本正态近似下的 `approximation` | `supported within regime` |
| paired mean difference $\Delta$ | 给定同一 items 的描述性 `exact statistic`；置信推断还依赖正确 resampling unit | `verified statistic` |
| decontamination 后 eval 独立 | 检测覆盖范围内的 `implementation claim`；未知上游、paraphrase、决策层过拟合仍可能破坏解释 | `plausible unless audited` |

本文默认比较的是冻结 tokenizer、mask、prompt/eval harness 和数据版本下的模型。若每个 mixture 使用不同 train-mix validation 权重，`L_{\text{train-mix}}` 的变化同时包含模型与测量分布变化，不能直接纵向归因。

## Alternative Explanations or Conflicts

污染后 clean subset 分数下降并不自动证明原分数被污染抬高：移除的样本可能系统性更容易、更短或来自特定 domain。两种解释首先在 subset selection 上分叉。区分检查包括按长度/source/difficulty 匹配的 clean/flagged 对照、污染强度梯度、时间切分和预先冻结的去污染规则；若无法匹配，应把影响保留为 `open`，只陈述检测到 overlap。

## 机制与相关知识

### 1. Split unit 决定独立性

随机按 sequence 切分常常太晚。相同网页的 chunk、同一 repo 文件、同一本书章节、同一 duplicate cluster 会跨 train/validation，造成近重复泄漏。

应按能近似独立的 group 切分，例如：

- web：canonical URL/host + duplicate cluster + crawl time；
- code：repository 或 project，而非随机函数；
- books/papers：作品/document ID，而非段落；
- dialogue：conversation/user boundary（并遵守隐私）；
- temporal eval：严格以时间 cutoff 切 source snapshot。

若 group $g(d)$ 表示独立单元，split invariant 是：

$$
\{g(d):d\in D_{\text{train}}\}\cap
\{g(d):d\in D_{\text{val}}\}=\varnothing
$$

并在 split 后做 cross-split exact/near/substr audit，不能假设 group metadata 完美。

### 2. Validation 应与 mixture 相关，但不必复制 mixture

训练分布的 micro held-out set 适合估计平均 NLL；diagnostic set 需要足够覆盖每个重要 domain；downstream set 对齐能力目标。因此至少保留两套聚合：

$$
L_{\text{train-mix}}=\sum_k q_kL_k,
\qquad
L_{\text{macro}}=\frac{1}{K}\sum_kL_k
$$

第一项回答“在目标 mixture 上平均如何”，第二项帮助发现小域退化。还应报告每个 $L_k$、样本/token 数和不确定性。

当比较不同 mixture 时，建议 diagnostic validation 保持固定；若 validation 也跟着新 mixture 变，指标变化混合了模型变化和测量分布变化。

### 3. Loss、PPL 与 downstream 各自扮演什么角色

- held-out NLL/PPL：便宜、稳定、可高频，贴近训练目标；不直接等于任务能力。
- domain NLL：对数据配比和训练异常敏感；依赖 tokenizer 和 domain 定义。
- downstream accuracy/F1/pass@k：更贴近能力，但任务有限、方差/提示敏感，且更易 contamination。
- human/model-graded generation：覆盖开放输出，但 judge bias/version 也要冻结。
- memorization/privacy eval：回答风险，不应混入“能力总分”。

对于 accuracy，若 $n$ 个近似独立二项样本、正确率 $\hat p$，粗略标准误为：

$$
\operatorname{SE}(\hat p)\approx
\sqrt{\frac{\hat p(1-\hat p)}{n}}
$$

但 benchmark items 可能相关，prompt/checkpoint selection 也引入额外不确定性；可以用按 task/group bootstrap、多个 seeds 和 paired comparison，而不是只报单次小数点变化。

### 4. Paired comparison 比两个孤立分数更有信息

对同一 eval items，模型 A/B 的 per-item metric 为 $z_i^A,z_i^B$，分析差值：

$$
\Delta=\frac{1}{n}\sum_{i=1}^n(z_i^B-z_i^A)
$$

对差值按 item/group bootstrap 能利用配对，通常比把两个总体置信区间独立比较更敏感。数据 ablation 还应尽量固定：模型架构、初始化/seed、tokenizer、总 sampled/loss tokens、优化器 schedule、compute、checkpoint 规则和 eval prompts。

若只能跑一个 seed，明确结论是“该 run 的观察”，并用中间 checkpoint 曲线、多个 domain 和小型复现实验降低偶然性。

### 5. Contamination audit 是 validation contract 的组成部分

在正式训练前：

- 冻结 eval 原始版本、上游来源和处理脚本；
- 对 train candidates 做 exact、n-gram containment、near/substr 检查；
- 对问题、答案、代码测试、解释分别检查；
- 记录 overlap threshold、被删除/标记的 train/eval items；
- 若无法可靠去除，预注册 contaminated/clean slice 分开报告。

正式训练后发现污染，不应悄悄改 benchmark 并覆盖旧分数；创建新 version，保存原始与 clean subset，并说明 selection bias（clean subset 可能更难或分布不同）。

### 6. Training-time validation cadence 与 checkpoint 选择

validation 太稀会错过 divergence/phase transition，太频繁会浪费 compute 并诱导反复调参。可按训练 tokens 而非 wall-clock/steps 设 cadence，以便 global batch 变化时仍可比较。

checkpoint selection 要预先定义，例如：

- 固定最终 token budget；
- 预定义 composite metric 最优；
- 特定 domain 不低于 guardrail 的 Pareto 规则；
- 只用 dev 选 checkpoint，test 仅最终一次。

如果用同一个 benchmark 同时调 filter、mixture、超参和 checkpoint，它已成为开发集，不再是独立 test。

### 7. Minimal validation contract 示例

```yaml
contract_version: 1
data_cutoff: 2026-06-30
tokenizer: name@immutable-revision
split_unit: source_group_plus_dedup_cluster
suites:
  train_mix_holdout:
    domains: [web, code, math]
    aggregation: [per_domain, train_mixture_micro]
  diagnostic:
    domains: [zh, en, code, math, long_document]
    aggregation: [per_domain, macro]
  downstream:
    protocol: frozen-prompts-and-harness-revision
decontamination:
  references: immutable-manifest-hash
  methods: [exact, ngram_containment, near_duplicate]
checkpoint_rule: fixed_loss_tokens
comparison: paired_items_and_three_seeds_when_affordable
```

这个 contract 还需要实际的 manifest hash、每域 token/items 数、mask、置信区间/bootstrapping 代码版本和 guardrail 阈值。

### 8. 诊断矩阵：曲线模式到数据假设

| 观察 | 数据侧假设 | 优先检查 |
|---|---|---|
| train loss 降、所有 val loss 不降 | 重复/污染 train、实现或优化问题 | loss mask、cross-split dup、loader |
| micro 改善、小域退化 | mixture/过滤偏向大域 | per-domain exposure、stage yield |
| 单一 source loss spike | source/schema/解析版本异常 | batch lineage、parser failure |
| train–val gap 随 exposure 扩大 | 小域过采样/记忆 | repeat histogram、unique tokens |
| downstream 升、held-out NLL 平 | 能力相关少数 slice 改善或污染 | task-aligned slice、decontam |
| 全部指标突然跳变 | validation/tokenizer/config 版本漂移 | contract hash、mask、checkpoint |

矩阵只生成待验证假设；最终归因需要样本审计或受控 ablation。

## 它怎样影响 pretraining data 工作

关系类型：冻结 validation contract `prerequisite-for` 可解释的数据 ablation；具体 suite 是该合同的 `implementation relation`，不是唯一实现。

Validation 是数据系统的反馈面。它把 curation/mixture 的变化映射到可复现结果，并阻止团队用一个受污染或被反复调优的 benchmark 自我确认。没有冻结 contract，数据迭代很难积累知识。

## 读完后应该掌握什么

- 能选择正确 split unit 并验证 cross-split independence。
- 能设计三层 validation suites，报告 per-domain/micro/macro 与不确定性。
- 能用 paired comparison 评估数据 ablation，列出必须控制的变量。
- 能写并冻结 validation contract，区分 dev、test 与决策层污染。

## 常见误区

- 随机按 chunk/sequence split，导致同文档或重复簇跨集合。
- 每次数据版本都重抽 validation，却把曲线直接纵向比较。
- 用 validation loss 选所有方案，又把它称为 unbiased test。
- benchmark 分数上升就归因于数据质量，忽略 contamination、prompt 或 checkpoint。
- 只报告相对提升，不报告绝对差、方差、样本量与 guardrail 退化。

## 用这些问题检查自己

1. code corpus 若按函数随机 split，会有哪些泄漏路径？请设计 repo-level split 与 cross-split audit。
2. 新 mixture 的 train-mix weighted loss 改善，但固定 diagnostic macro loss 退化。什么时候仍可能接受该方案？需要哪些 guardrail？
3. 你用 benchmark 选择了 30 个过滤阈值。即使 benchmark 文本未进入训练，为什么 test 解释已改变？如何重建可信评测？
4. clean subset 分数下降能否直接证明污染抬高原分数？还要考虑哪些 selection effects？

## 来源与建议阅读位置

- [Brown et al., 2020, GPT-3](https://arxiv.org/abs/2005.14165) — 阅读 contamination detection 与 clean benchmark 分析，练习区分检测上界和实际影响。
- [Lee et al., 2021, Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) — 关注 train–validation overlap 对评测准确性的影响。
- [Li et al., 2024, DataComp-LM](https://arxiv.org/abs/2406.11794) — 观察固定模型规模、训练预算与 evaluation suite 如何服务数据方案对照。
- [EleutherAI lm-evaluation-harness official repository](https://github.com/EleutherAI/lm-evaluation-harness) — 用于理解评测任务、prompt/config 和版本冻结的工程边界；使用时固定 commit 与任务定义。
