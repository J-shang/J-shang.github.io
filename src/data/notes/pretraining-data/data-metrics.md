---
title: "训练数据指标体系"
description: "把 corpus、pipeline、sampler 和模型行为连接成可下钻的指标系统，不把数据质量压成一个总分。"
topic: "pretraining-data"
section: "measurement"
slug: "data-metrics"
date: 2026-07-14
updated: 2026-07-14
order: 50
readtime: 17
source:
  repository: "J-shang/pt-data-learning"
  path: "knowledge-map/04-measurement/data-metrics.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/48b6c6907a65afc718659f922895f835335be1d3/knowledge-map/04-measurement/data-metrics.md"
  revision: "48b6c6907a65afc718659f922895f835335be1d3"
  syncedAt: "2026-07-14"
  contentHash: "sha256:09ecae2a9edbc7eb281e9fe2625d22e5d20c44db8f1011ced0e3a520652a8af8"
  manifest: "pretraining-data"
  managed: true
---
> 层级：04 Measurement
> 状态：`core`
> 初始资料核查截止：2026-07-14
> 主要 reasoning path：`phenomenon-to-mechanism`
> 证据姿态：指标定义/聚合恒等式为 `verified`；从曲线模式到数据原因的映射只生成 `plausible` 假设

## 一句话定位

数据指标体系把 corpus、pipeline、sampler 和模型行为连接成分层可观测系统；它的任务是定位和决策，而不是制造一个“数据质量总分”。

## Motivating Problem

一个全局 validation loss 可以改善，同时代码域、低资源语言或某个 source 严重退化；一个“quality score”也可能上升，却只是过滤器更偏好某种文体。指标系统需要的性质是：每个数都有明确对象和分母，能从异常下钻到 exposure、pipeline 与 source，并把因果故事保留为可检验假设。

本笔记从现象到机制：先把异常变成可测模式，再列竞争解释，最后用 slice、lineage、counter 和 ablation 区分。

## Minimal Motivating Example

validation 中 web 占 90%、code 占 10%。如果 web loss 降 0.05、code loss 升 0.30，micro average 仍可能改善。一个全局数字因此不能回答“所有关键能力是否改善”，但这也不等于 code 退化必然由数据 mixture 引起；tokenizer、loader、评测版本和训练配置都是竞争解释。

## 核心定义

一个完整指标必须是 tuple，而不只是数值：

$$
m=(\text{name},\text{object},\text{unit},\text{denominator},
\text{slice},\text{version},\text{estimator},\text{uncertainty})
$$

例如 `duplicate_rate=12%` 不完整；需要说明是 document 还是 token、exact 还是 near、比较范围、threshold、canonicalization、数据版本以及估计误差。

## Assumptions and Validity

| 关系或结论 | 类型与条件 | 置信状态 |
|---|---|---|
| 指标 tuple 完整描述测量语义 | 项目采用的 `design contract`，不是唯一标准；需要实现保存这些字段 | `supported` |
| PPL $=\exp(\mathrm{NLL})$ | 自然对数、同一 tokenizer/mask/normalization 下的 `exact identity` | `verified` |
| micro/macro/worst-group 公式 | 给定 domain partition 和权重的 `exact aggregation identities` | `verified` |
| Shannon entropy 衡量 domain 分布不均衡 | 对给定离散分布的 `exact statistic`；它与“数据价值”只有弱经验关系 | `verified statistic / open value claim` |
| loss pattern 指向某类数据问题 | `diagnostic association`，不是因果；同一模式可由优化、实现或评测变化产生 | `plausible until discriminated` |

默认 domain 标签互斥且覆盖被聚合对象；若是多标签 domain，micro/macro 分母需重新定义。跨 tokenizer 的 token-level NLL/PPL 不在同一测量空间，不能直接按本文公式作模型优劣比较。

## Alternative Explanations or Conflicts

“训练 loss 降得更快说明数据更好”至少与三种解释竞争：数据更有学习价值、数据更容易、数据更重复。它们在 unique exposure、held-out transfer、memorization 和 loss slope 上作出不同预测。区分检查是固定 sampled/loss tokens 与优化配置，比较 dedup 后的 unique coverage、独立 held-out loss、重复序列提取和下游 slice；未做这些检查前，结论保持 `plausible`。

## 相关知识展开

### 1. 四层指标：不要跨层偷换结论

| 层 | 典型对象 | 指标例子 | 能回答 |
|---|---|---|---|
| Corpus | 文档/token/domain | 规模、长度、语言、source、时间、重复、coverage | 数据“有什么” |
| Pipeline | stage/规则/job | yield、reject reason、吞吐、失败、成本 | 数据“怎么变” |
| Exposure | sampler/sequence/batch | target vs actual share、repeat、packing、mask | 模型“看到什么” |
| Model response | checkpoint/domain/eval | NLL/PPL、loss slope、memorization、downstream | 模型“学到/表现什么” |

Corpus 更大不推导出 model 更好；filter score 更高不推导出 downstream 提升。跨层因果需要受控实验。

### 2. Corpus 与 pipeline 健康指标

建议最小集合：

- `documents/bytes/chars/unique_tokens` 及长度分位数；
- source/language/domain/time/license coverage；
- parse success、stage doc/token yield、reject reason；
- exact/near cluster size、保留规则、重复 span 占比；
- quality feature 分布，而非仅 threshold 后的 keep rate；
- PII/policy action 计数与人工 audit 误差；
- 每 stage wall-clock、CPU/GPU hours、I/O 与峰值内存。

长度应看分布，例如 p50/p90/p99 和重尾图，而非只看均值。聚合时保留 denominator：一个规则删除 1% documents 可能删除 20% tokens。

### 3. Diversity 与 coverage：先定义“空间”

多样性没有唯一标量。可以在不同空间测量：

- lexical：unique n-gram、type-token ratio（强受长度影响）；
- semantic：embedding cluster coverage/density；
- metadata：source、domain、语言、时间、地理/作者群体；
- structural：文档类型、代码语言、公式/表格/对话形态；
- knowledge/task：概念、技能、难度或 benchmark-related coverage。

对离散 domain 分布 $p$，Shannon entropy 为：

$$
H(p)=-\sum_{k=1}^{K}p_k\log p_k
$$

但高 entropy 不等同高价值：随机噪声也可产生高 lexical diversity。指标必须与样本审计、质量/可学习性和目标能力联用。

### 4. 模型相关指标：NLL、PPL 与学习曲线

固定 tokenizer 和 loss mask 后，分域平均 NLL：

$$
L_k=-\frac{1}{M_k}\sum_{i\in V_k}\sum_t m_{i,t}
\log p_\theta(x_{i,t}\mid x_{i,<t})
$$

perplexity 为：

$$
\operatorname{PPL}_k=\exp(L_k)
$$

PPL 是 NLL 的单调变换，不提供额外排序信息；它仍依赖 tokenizer、文档边界和 mask。训练诊断更应保存曲线和斜率：

$$
s_k^{(u)}=\frac{L_k^{(u)}-L_k^{(u-\Delta)}}{\Delta}
$$

其中 $u$ 为 step/token position。某域 loss 高但下降快，与高且不降的含义不同；后者可能是容量不足、数据噪声、标签错误或该域几乎没被采样。

### 5. Micro、macro 与 worst-group

设每域 token 数 $M_k$、loss $L_k$：

$$
L_{\text{micro}}=\frac{\sum_kM_kL_k}{\sum_kM_k},
\quad
L_{\text{macro}}=\frac{1}{K}\sum_kL_k,
\quad
L_{\text{worst}}=\max_kL_k
$$

- micro 表示评测 token 分布下的平均体验；大域支配。
- macro 平等看待 domain，但小域高方差也获得同等权重。
- worst-group 关注最差域，但对噪声和样本量非常敏感。

正确报告方式不是三选一，而是 per-domain + 合适聚合 + 置信区间，并解释决策目标。

### 6. 数据利用率与 exposure 指标

至少跟踪：

$$
\eta_{\text{loss}}=\frac{N_{\text{loss}}}{N_{\text{sampled}}},
\qquad
r_k=\frac{E_k}{N_k},
\qquad
\epsilon_k=\hat q_k-q_k
$$

$\eta_{\text{loss}}$ 暴露 padding/mask 浪费；$r_k$ 暴露小域重复曝光；$\epsilon_k$ 暴露 sampler 偏差。可进一步记录 documents seen 的 coverage、repeat-count histogram、batch 内 source/domain concentration。

### 7. 一个诊断例子：全局 loss 正常但代码域退化

假设 validation token 中 web 占 90%、code 占 10%。旧/新模型 loss：

| Domain | 权重 | 旧 loss | 新 loss | 变化 |
|---|---:|---:|---:|---:|
| web | 0.9 | 2.00 | 1.95 | -0.05 |
| code | 0.1 | 1.50 | 1.80 | +0.30 |

$$
L_{\text{micro,old}}=1.95,\qquad
L_{\text{micro,new}}=1.935
$$

全局 micro loss 改善 0.015，却隐藏 code 大幅退化。诊断路径：

1. 核对 code target/actual exposure 与 tokenizer fertility。
2. 核对 filter/dedup 的 code yield、长度和 source 分布变化。
3. 检查 code validation 是否冻结且无 parser/schema 变化。
4. 看 loss 曲线从何时分叉，映射该时间点的 mixture/phase/config。
5. 在固定 token/compute 下做 data change paired run。

### 8. Dashboard 应支持 drill-down，而非堆 KPI

```text
run summary
  -> per-domain validation curve
     -> exposure target vs actual
        -> data build version and stage funnel
           -> source/language/length/quality slices
              -> sampled accepted/rejected examples + lineage
```

每个指标卡记录 owner、更新频率、expected range、alert threshold、runbook 和版本。alert 只指出异常，不直接给因果结论。

## 与 Pretraining Data 主线的关系

关系类型：指标 schema `implemented-by` counters/manifests/dashboards；异常模式与根因仅是待区分的 `empirically-associated-with`。

指标体系是闭环的观察层：token accounting 给分母，lifecycle 给 lineage，curation 给 stage 变化，mixture 给 exposure，validation 给模型响应。它使数据版本之间的差异可以定位，并为 ablation 提供假设。

## 目标掌握程度

- 能为任意指标补全 object/unit/denominator/slice/version/uncertainty。
- 能推导 micro/macro/worst-group 并解释各自掩盖什么。
- 能从分域 loss 异常沿 exposure → pipeline → source 反查。
- 能设计不超过 15 个核心视图、但支持 drill-down 的 observability 规范。

## 常见误区

- 把多个 feature 加权成不透明“quality score”，却不保留原始分布。
- 把 training loss 降得更快直接解释成数据更好；重复/更易数据也会如此。
- 只报均值，不报 quantile、slice、样本量和不确定性。
- 跨 tokenizer 比较 token-level PPL。
- 指标阈值变了却沿用同一名称和历史曲线。

## 自测问题

1. 过滤后 PPL 下降、downstream 也上升，仍不能证明过滤器普遍有效。列出需要控制或切片的变量。
2. 何时 macro average 会比 micro average 更误导？构造一个小域样本量极小的例子并提出修正。
3. 某 domain 的 target mixture 正常，但 loss tokens share 偏低。可能是哪几类 packing/mask/loader 问题？

## 参考入口

- [Li et al., 2024, DataComp-LM](https://arxiv.org/abs/2406.11794) — 阅读其固定训练与评测设计，观察数据策略如何用模型响应比较。
- [Penedo et al., 2024, FineWeb](https://arxiv.org/abs/2406.17557) — 关注 curation ablation、数据统计和小模型评估如何结合。
- [Soldaini et al., 2024, Dolma](https://arxiv.org/abs/2402.00159) — 观察开放 corpus 对组成和中间构建状态的文档化。
- [DataTrove official repository](https://github.com/huggingface/datatrove) — 阅读 summary statistics 与 pipeline logging，把本文指标 tuple 落到实际 stage outputs。
