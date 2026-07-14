---
title: "质量过滤、去重与去污染"
description: "用共同框架区分质量代理、训练集内部冗余和训练—评测泄漏，并检查各自的误删与验证边界。"
topic: "pretraining-data"
section: "curation"
slug: "filtering-dedup-decontamination"
date: 2026-07-14
updated: 2026-07-14
order: 30
readtime: 19
source:
  repository: "J-shang/pt-data-learning"
  path: "knowledge-map/02-curation/filtering-dedup-decontamination.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/48b6c6907a65afc718659f922895f835335be1d3/knowledge-map/02-curation/filtering-dedup-decontamination.md"
  revision: "48b6c6907a65afc718659f922895f835335be1d3"
  syncedAt: "2026-07-14"
  contentHash: "sha256:9c2dc77b209b5c7d1b2fe04d2c3e0f877416314616b3c09881ec1d63f2946e44"
  manifest: "pretraining-data"
  managed: true
---
> 层级：02 Curation
> 状态：`core`
> 初始资料核查截止：2026-07-14
> 主要 reasoning path：`method-family trace`
> 证据姿态：集合/概率关系在理想假设下为 `verified`；删除策略对能力、隐私和记忆的效应通常为 setting-specific `supported`

## 一句话定位

Filtering、deduplication 和 decontamination 都会删除或降权数据，但它们分别优化代理质量、训练集内部冗余、训练—评测独立性；混为一个“清洗率”会同时破坏能力、多样性和评测可信度。

## Motivating Problem

数据管线常把所有删除都汇总成“清洗率”，但删除乱码、合并训练集内部副本、移除 benchmark 泄漏解决的是不同问题，错误代价也不同。需要一个共同比较框架，能指出每类方法改变什么对象、依据什么关系、保存什么状态、如何验证以及在哪里误删。

本笔记采用 method-family trace：先固定比较轴，再看 exact、near、substring、semantic 和 decontamination 的操作差异，不用名称或删除率代替机制。

## Minimal Motivating Example

考虑三类记录：A/B 文本完全相同；C 与 A 共享模板但正文不同；长文档 D 完整包含一道短 benchmark 题。Exact dedup 应合并 A/B，near-dedup 需要判断 C 是否只是模板重复，decontamination 则应检查 D 对评测独立性的影响。一个统一阈值无法正确表达三种决策。

## 核心定义

给文档 $d$：质量过滤器根据 feature/score $s(d)$ 和 policy $\pi$ 决定保留、拒绝或降权；dedup 根据训练语料内部相似关系 $R(d_i,d_j)$ 形成重复簇；decontamination 根据训练对象 $d$ 与受保护评测对象 $e\in E$ 的重叠关系决定移除或标记。

$$
\begin{aligned}
\text{filter}(d)&=\mathbb{1}[s(d)\ge \tau]\quad\text{（简化形式）},\\
\text{dedup}(d_i,d_j)&=\mathbb{1}[R(d_i,d_j)\ge \gamma],\\
\text{decontam}(d,E)&=\mathbb{1}[\max_{e\in E}R'(d,e)\ge \delta]
\end{aligned}
$$

三个阈值的语义、相似函数、比较单位和错误代价不同，不能共享一个模糊的“重复/低质”标签。

## Assumptions and Validity

| 关系或结论 | 类型与条件 | 置信状态 |
|---|---|---|
| $p_{\text{keep}}(d)\propto a(d)p(d)$ | selection model 下的 `exact identity`；假设 $a(d)$ 是给定文档的保留概率且分母非零 | `verified` |
| MinHash collision probability 等于 Jaccard | 理想 min-wise independent permutation、集合 shingle 表示下的 `exact identity`；工程 hash 为近似 | `verified in ideal model` |
| $1-(1-s^r)^b$ | band/row 独立假设下的候选概率；真实 hash correlation、bucket 截断会引入误差 | `approximation in implementation` |
| 去重降低 memorization/privacy risk | 已有设置中的 `empirical association`，效应依赖重复分布、模型、训练 exposure 和攻击定义 | `supported, not universal` |
| 去污染使 benchmark “无污染” | 检测器覆盖范围内的 `implementation claim`；paraphrase、翻译和未知上游来源会造成漏检 | `plausible unless audited` |

公式中的文档被简化为集合或 feature score；真实数据有长度、模板、语言和 source 权重。任何阈值都必须与 normalization、比较单位、候选生成和 clustering semantics 一起报告。

## Alternative Explanations or Conflicts

“更激进过滤提高数据质量”与“更激进过滤损失多样性”并非自动兼容。二者首先在目标对象上分叉：前者通常观察固定短训练预算下的平均 benchmark，后者关注 unique coverage、长 token horizon 或少数域。区分性检查是同时固定 sampled/loss tokens，比较短/长训练 horizon、per-domain validation、unique exposure 与 rejected-slice 表现；若 operating point 不同，就不能把结果当直接矛盾或普遍结论。

## 相关知识展开

### 1. Quality filtering：代理目标与选择偏差

常见 feature 可分为：

- 结构/格式：长度、行长、符号率、重复行、HTML 残留、乱码。
- 语言与文本性：语言识别置信度、字符集比例、停用词/功能词模式。
- 内容代理：困惑度、与 reference corpus 的相似度、分类器/LLM 质量分。
- policy：PII、恶意内容、许可/source denylist；这类不等同统计质量。

过滤器对 observed distribution $p(d)$ 施加 selection probability $a(d)$，保留分布为：

$$
p_{\text{keep}}(d)=\frac{a(d)p(d)}{\mathbb{E}_{p}[a(d)]}
$$

因此过滤必然重塑分布。若 classifier 的“高质量”reference 偏向正式英语，它可能同时抬升英语 benchmark、降低口语/方言/少数语言 coverage。正确问题不是“过滤率多少”，而是 accepted/rejected 各 slice 的 precision、coverage 与 downstream trade-off。

建议每条规则拥有稳定 `reason_code`，并保留：阈值附近样本、各 slice 拒绝率、多人/盲审样本和 filter ablation。多个规则串联时记录首次拒绝原因与所有命中原因，否则无法分析交互。

### 2. Exact、near 与 substring dedup 的机制差异

| 家族 | 改变的对象 | 核心操作 | 保存状态 | 主要失败模式 |
|---|---|---|---|---|
| Exact | 完整规范化文档/段落 | 内容 hash 相等 | hash → canonical ID | normalization 太弱漏检，太强误合并 |
| MinHash/LSH | shingle 集近似相似 | 估计 Jaccard + bucket candidates | signatures、buckets、cluster edges | 短文档不稳；参数改变 precision/recall |
| Exact substring | 跨文档长公共串 | suffix array/相关索引寻找长重复片段 | substring spans / removal map | 内存/计算重；公共模板可能被过删 |
| Semantic | 语义表示 | embedding/模型相似 | vectors/index | 同主题不等同重复，误删多样表达 |

MinHash 针对 shingle 集 $A,B$ 估计 Jaccard：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|},
\qquad
\Pr[h_{\min}(A)=h_{\min}(B)]=J(A,B)
$$

使用 $m$ 个独立/近似独立 hash 的签名，一致比例估计 $J$。LSH 把签名切成 $b$ 个 band、每 band $r$ 行，候选概率近似：

$$
P_{\text{candidate}}(s)=1-(1-s^r)^b
$$

$s$ 是真实 Jaccard。调 $b,r$ 就是在候选召回、误报和计算之间做选择，不能只写“用了 MinHash”。实现报告需包含 normalization、shingle 单位/大小、signature 数、band 参数、相似阈值、cluster 保留规则和跨 shard/global 范围。

### 3. 重复簇不是一个对称的“都删掉”问题

检测到 cluster $C=\{d_1,\ldots,d_m\}$ 后，还需选择 canonical document。选择规则可能考虑：

- source 可信度/许可；
- timestamp（保留最早或最新）；
- 抽取质量、metadata 完整性；
- 长度和模板占比；
- 保留一个、按权重降采样，或仅删除重复 span。

cluster 图还会出现非传递问题：$J(A,B)$ 和 $J(B,C)$ 都超过阈值，但 $J(A,C)$ 不超过。connected components 会把三者合并，greedy representative 可能不会。实现与统计必须记录 clustering semantics。

### 4. Decontamination：保护评测边界

去污染的 reference 是冻结的 validation/test/benchmark $E$，目标是避免训练样本泄漏评测问题、答案或强近似变体。检测粒度包括：

- exact 文档/题目 hash；
- normalized n-gram overlap；
- question 与 answer 分别匹配；
- code problem/function signature/test case；
- paraphrase/translation/semantic 变体；
- benchmark 来源网页或上游数据集，而不只最终 JSON。

对 eval example $e$，可定义 n-gram containment：

$$
C_n(e,d)=\frac{|G_n(e)\cap G_n(d)|}{|G_n(e)|}
$$

containment 与 Jaccard 不同：当训练文档很长、完整包含短题目时，Jaccard 可能很小，而 containment 接近 1。短题、模板化指令和常见代码片段需要额外最小长度/稀有度规则，否则误报很高。

decontamination 不是一次性操作。benchmark 在开发中反复用于选择 filter/mixture/checkpoint 时，即使文本没有进入 gradient，也会发生“决策层污染”。所以需要区分：

- **train contamination**：内容进入训练数据；
- **validation overfitting**：反复按 validation 选择方案；
- **benchmark exposure**：团队/模型生成器可能接触 benchmark；
- **temporal leakage**：训练 cutoff 晚于待测事件/材料。

### 5. 一个小型 dedup worked example

文档经小写、空白归一化后得到 3-shingle 集：

```text
A: {the cat sat, cat sat on, sat on mat}
B: {the cat sat, cat sat on, sat on rug}
C: {quantum field theory, field theory notes}
```

$$
J(A,B)=\frac{2}{4}=0.5,\qquad J(A,C)=0
$$

若 near-dedup 阈值 $\gamma=0.8$，A/B 都保留；若阈值 0.5，它们成为候选重复。这个例子揭示三点：

1. shingle 大小决定局部改写是否仍相似；
2. 对只有几个 shingle 的短文档，一个 shingle 就会显著改变相似度；
3. “阈值 0.8”没有脱离 normalization、tokenization 和长度分布的独立意义。

最小测试集应人工植入：完全重复、只改标点、模板相同正文不同、长文包含短 benchmark、同主题但不重复、翻译/改写。对每类报告 precision/recall，不只报告删除率。

### 6. 删除率不是成功指标：建立反事实审计

对每个 curation stage，至少报告四类结果：

1. **操作结果**：doc/token removal、cluster size、candidate pairs、runtime/memory。
2. **检测质量**：人工或合成标注上的 precision/recall，按长度/语言/source 切片。
3. **分布影响**：before/after 的 domain、语言、长度、质量 feature、多样性变化。
4. **模型影响**：等 sampled token/compute 的 proxy training，比较分域 loss、下游能力、memorization 与污染敏感 eval。

如果去重后只剩较少 unique tokens，但仍固定训练更大 sampled token budget，survivors 会被重复采样。此时“dedup 降低重复”在语料集合层成立，在训练 exposure 层未必成立；必须结合 mixture/exposure 指标。

### 7. Pipeline 顺序与实现 invariant

一个常见但需实证的顺序：轻量 normalization/exact dedup → 解析/质量特征 → 规则/模型过滤 → global near/substr dedup → decontamination → split/freeze。需要检查：

- normalization 后能否追溯原文，hash 是否版本化；
- dedup 是 per snapshot、per source 还是跨全部 snapshots；
- split 前是否消除 train/validation 近重复；
- benchmark reference 是否含 dev/test、答案、变体和上游来源；
- removal map 是否能支持版权/隐私删除向下游 shard 传播；
- 被删除文档是否仍通过缓存、旧 manifest 或 sampler 泄漏。

## 与 Pretraining Data 主线的关系

关系类型：三者都是 lifecycle 中的 `implementation stages`；它们与能力、记忆和风险结果是有条件的 `empirically-associated-with`，不是指标等价关系。

这三类操作决定 unique data 的构成，也决定 validation 是否可信。它们必须同时连接到 token accounting（删了多少、剩余数据会暴露几次）、mixture（哪些群体被过度删除/重采样）和实验（在等 compute 下验证收益）。

## 目标掌握程度

- 能沿“对象—操作—状态—实现—失败模式”比较 exact/MinHash/substr/decontam。
- 能推导并解释 Jaccard、LSH 候选概率和 n-gram containment。
- 能设计人工 fixture 测 precision/recall，并进行 accepted/rejected slice audit。
- 能区分 train contamination、validation overfitting 与 temporal leakage。

## 常见误区

- 把语义相近当成重复，删除对同一概念的多样表达。
- 报告 MinHash 阈值，却不报告 normalization、shingle、signature 和 clustering。
- 只在每个 shard/source 内 dedup，却称作 global dedup。
- 只匹配 benchmark 最终文本，不追踪上游网页、答案和常见变体。
- 以删除率越高为越好，忽略 false positive 和长尾 coverage。

## 自测问题

1. 为什么长训练文档包含完整短题时，Jaccard 可能漏检而 containment 能检出？请构造集合例子。
2. global dedup 删除了 40% unique tokens，但总训练 token 不变。什么情况下 memorization 风险反而可能集中到剩余小域？
3. 过滤器使总体 proxy loss 下降，但低资源语言 loss 上升。你如何判断这是预期 trade-off、分类器偏差还是 tokenizer/采样计数错误？
4. 若在同一相似函数和阈值下 $R(A,B)\ge\gamma$、$R(B,C)\ge\gamma$，但 $R(A,C)<\gamma$，connected-component 与 greedy dedup 会分别做什么？哪种更适合保留多样性？

## 参考入口

- [Lee et al., 2021, Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) — 先读方法与 validation overlap 分析，理解 exact substring/near duplication 如何影响记忆与评测。
- [Kandpal et al., 2022, Deduplicating Training Data Mitigates Privacy Risks in Language Models](https://arxiv.org/abs/2202.06539) — 关注训练序列重复次数与可提取/记忆风险的经验关系。
- [Brown et al., 2020, Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — 读 contamination 方法与 clean subset 分析，注意其检测规则和结论边界。
- [Penedo et al., 2024, The FineWeb Datasets](https://arxiv.org/abs/2406.17557) — 对照实际 web curation ablation，观察过滤和去重策略如何通过小模型训练比较。
- [DataTrove official repository](https://github.com/huggingface/datatrove) — 阅读 exact、MinHash、sentence/substr dedup 示例；检查 pipeline state、summary statistics 与跨 task 执行。
