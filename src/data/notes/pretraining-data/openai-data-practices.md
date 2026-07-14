---
title: "OpenAI GPT 与 Reasoning 模型的数据实践"
description: "比较 GPT-3 到 reasoning 模型的 mixture、来源类别、训练阶段和披露边界如何变化。"
topic: "pretraining-data"
section: "international-cases"
slug: "openai-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 120
readtime: 25
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/openai.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/international/openai.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:07d92182a567c698f78978fcbbdff2cb5db0457fe981f45c1c4db8e944ccf182"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family/historical trace`
> 研究锚点：GPT-3、GPT-4、GPT-4.5、o1 与 GPT-5.6；不把 API 小版本自动视为新的训练代际。

## 定位与 motivating problem

OpenAI 是闭源压力测试：同一组织在 GPT-3 论文中披露了定量 mixture、exposure 和 contamination bug；GPT-4 以后更多只披露来源类别、过滤目标、训练阶段和安全/能力评测。研究重点不是填补商业机密，而是确定每一代公开证据允许的最强结论。

截至 2026-07-14，最新锚点为 2026-07-09 发布的 GPT-5.6。它的 system card 能确认高层 source 类别、过滤和 reasoning RL，但不能确认 token 规模、mixture、cutoff、dedup 或 pretraining validation contract。

## 代际与阶段表

| 代际 | 阶段 | 已披露数据/方法 | 关键未知 | 披露等级 |
|---|---|---|---|---|
| GPT-3 175B | `P0` | 300B training tokens；5 类 corpus mixture、组件规模与 exposure；filtered Common Crawl、document fuzzy dedup、contamination analysis | rights、训练代码、数据版本/顺序、loss tokens | `D2` |
| GPT-4 | `P0` | public internet + licensed third-party data；next-token pretraining；用小模型预测 loss/capability | dataset construction、token、mixture、cutoff、dedup | `D1` |
| GPT-4 | `A2` | RLHF | feedback 数据量、mixture 与版本 | `D1` |
| GPT-4.5 | `P0/P1`（官方未细分） | 扩大 pretraining；官方称使用可扩展方法，以较小模型派生的数据训练更大模型 | derived data 的生成者、占比、验证/rejection、进入哪个阶段 | `D1` |
| GPT-4.5 | `A1/A2` | new supervision + SFT + RLHF | 数据规模和构成 | `D1` |
| o1/o1-mini | `P0` | public/open data、reasoning data、scientific literature、partnership/paywalled/specialized archives、in-house custom data；PII/safety filtering | token、mixture、cutoff、pipeline 阈值 | `D1` |
| o1 | `A2` | large-scale RL 训练 chain-of-thought reasoning | rollout/verifier/teacher、token accounting | `D1` |
| GPT-5.6 family | `P0/P1`（官方未细分） | public internet、third-party partnerships、users/human trainers/researchers provide or generate；质量/PII/safety filtering | token、mixture、cutoff、dedup、synthetic 占比、pretraining validation | `D1` |
| GPT-5.6 reasoning models | `A2` | reasoning through reinforcement learning | 数据配方、verifier、rollout、训练 compute | `D1` |

## Assumption ledger

| 项 | 本笔记的处理 |
|---|---|
| 研究对象 | GPT-5.6 是 Sol/Terra/Luna family；只在 system card 明确共同描述时汇总，不把路由/effort 当数据阶段 |
| 统计单位 | GPT-3 的 300B 视为 sampled training tokens；OpenAI 后续“trillions of tokens”组织级说明不映射到某一模型 |
| 阶段 | base/continued pretraining 与 SFT/RL 分开；o1/GPT-5.6 reasoning RL 不计入 pretraining token 数 |
| 分布 | GPT-3 有 example mixture；GPT-4+ 只有 source 类别，不能假设自然采样或沿用 GPT-3 权重 |
| cutoff | 除明确来源时间外不采用模型自述 knowledge cutoff |
| synthetic | “researchers provide or generate”“data derived from smaller models”表明生成数据存在，但不自动等价于某代 pretraining 的定量 synthetic mixture |
| 省略效应 | benchmark 改善混合了模型、数据、compute、post-training、test-time reasoning 与 eval 版本，不能用于数据因果归因 |

## 数据字段表

| 字段 | GPT-3 | GPT-4 / GPT-4.5 | o1 / GPT-5.6 | 置信 |
|---|---|---|---|---|
| source | filtered Common Crawl、WebText2、Books1/2、Wikipedia | GPT-4：public + licensed；GPT-4.5 精确 source `unknown` | public internet/open、partnership、user/human/researcher provided/generated；o1 另提 reasoning/science/paywalled/specialized | 官方声明 `verified` |
| rights | `unknown` | GPT-4 有 licensed third-party 类别；逐 source 条款 `unknown` | partnership 表明部分授权访问；逐 source 权利 `unknown` | `open` |
| cutoff | Common Crawl 使用 2016–2019 的 41 个 monthly shards；其他不完整 | `unknown` | `unknown` | GPT-3 CC `verified`；其余 `open` |
| token scale | 300B sampled training tokens | unique/sample/loss 全 `unknown` | unique/sample/loss 全 `unknown` | GPT-3 `verified`；其余 `open` |
| mixture | 60/22/8/8/3%，按训练 examples 抽取，原表四舍五入合计 101% | `unknown` | `unknown` | GPT-3 `verified` |
| filter | CC 参考高质量语料过滤；后续只给高层 quality/PII/safety filtering | GPT-4 construction 未披露；GPT-4.5 `unknown` | o1 提及 Moderation API/safety classifiers；GPT-5.6 提及 PII reduction 与 safety classifiers | 声明范围 `verified`；实现 `open` |
| dedup | document-level fuzzy dedup within/across datasets | `unknown` | `unknown` | GPT-3 `verified` |
| decontam | 清理 benchmark overlap 时有 bug；用 13-gram clean subset 做事后分析 | GPT-4 对考试/benchmark 做 contamination checks，部分污染项移除或报告较低分 | pretraining contamination protocol `unknown` | 报告事实 `verified`；完整无污染 `open` |
| synthetic | 未作为独立 mixture 披露 | GPT-4.5 有 data derived from smaller models，但生成/筛选细节 `unknown` | generated information 与 RL reasoning 数据存在；占比/阶段 `unknown` | 存在性 `supported`；定量 `open` |
| artifacts | paper，无训练 data/code/order/checkpoints/logs | technical/system/product cards | system cards | `verified` |

## GPT-3：定量披露锚点

GPT-3 是本家族中最适合做 token accounting 的公开代际。论文表 2.2 给出：

| 数据集 | corpus quantity | training mix weight | 300B 训练下 exposure |
|---|---:|---:|---:|
| Common Crawl (filtered) | 410B tokens | 60% | 0.44 epochs |
| WebText2 | 19B | 22% | 2.9 epochs |
| Books1 | 12B | 8% | 1.9 epochs |
| Books2 | 55B | 8% | 0.43 epochs |
| Wikipedia | 3B | 3% | 3.4 epochs |

注意：权重因表中整数四舍五入合计为 101%，不能再把它们当精确概率回算到个位 token。论文说明权重是训练时 examples 的抽取比例，而不是 corpus 自然大小比例；因此高质量小域会被重复 exposure。

### 可检查锚点

对于第 $k$ 个 component，若把表中权重 $q_k$ 近似视为 token exposure 权重，则：

$$
E_k \approx T_{\text{train}}q_k,\qquad
r_k \approx \frac{E_k}{N_k}
$$

其中 $T_{\text{train}}=300\text{B}$，$N_k$ 是论文按其 tokenizer 统计的 component tokens。这里是 approximation：原表称 mixture 按 examples 抽取、百分比被取整，不能替代原报告的 epoch 数。

## 从 GPT-3 到 GPT-5.6 的披露变化

### GPT-3 -> GPT-4

- changed field：dataset construction/token/mixture。
- previous：GPT-3 给出 component quantity、weight、epoch 和污染 bug。
- new：GPT-4 只确认 public/third-party licensed 来源，并明确不提供 dataset construction 等细节。
- relation：historical disclosure change，不是数据实践本身退化的证据。
- confidence：`verified`。

### GPT-4 -> GPT-4.5

- changed field：公开提到 model-derived data。
- new：官方称发展了可扩展技术，以较小模型派生的数据训练更大模型，同时扩大 pretraining/post-training。
- first unknown：没有说明 teacher、生成阶段、占比、validation/rejection 或是否属于 `P0/P1/A1`。
- discriminating check：需要 model card 中的 stage-specific manifest/mixture 或可复现生成 pipeline。
- confidence：存在 model-derived data 的声明为 `verified`；“主要是合成预训练”之类外推为 `open`。

### o1 -> GPT-5.6 reasoning

- common method：官方均说明 reasoning models 通过 reinforcement learning 学会在回答前推理。
- unknown：是否共享 base、RL data、verifier 或 rollout recipe。
- relation：method-family similarity；不是 exact implementation identity。
- confidence：`supported`（高层方法）；配方同一性 `open`。

## Validation、contamination 与数据归因

### GPT-3 contamination case

`[来源事实]` GPT-3 团队尝试移除训练集与所有研究 benchmark dev/test 的 overlap，但过滤 bug 导致只部分移除。事后将与 pretraining 数据存在 13-gram overlap 的样本保守标记并构造 clean subset，对比原始/clean score。

这能支持“作者检查了检测到的潜在污染对报告分数的影响”，不能支持“训练集没有污染”。clean subset 还可能改变题目难度分布；检测规则也只覆盖可由该匹配方法找到的 overlap。

### GPT-4 scaling/validation anchor

`[来源事实]` GPT-4 报告用最多小 10,000× compute 的模型预测最终 loss；loss 数据来自不在训练集的内部 codebase。它还对 exam/benchmark 做 contamination checks，并对检测到的题目移除或报告两版较低分。

这是 validation/scaling 工程的证据，但没有公开 pretraining mixture 和数据 pipeline，因此不能用 loss scaling 拆解“数据质量提升”。

### GPT-5.6 evaluation 不等于 pretraining validation

GPT-5.6 system card 公开大量 safety、health、alignment、cyber/biology 和 deployment simulation 评测；这些能审计部署/安全主张的部分边界。它们不提供固定 tokenizer、source-domain held-out loss、每域 token 数或 pretraining decontamination contract，因此不能替代本项目定义的 pretraining validation。

## 表面冲突与区分性检查

### “OpenAI 使用 synthetic data”与“GPT-5.6 的 synthetic 占比未知”

- 第一个差异：组织级存在性 vs 模型代际/阶段的定量 composition。
- 官方资料确认 researchers may generate information，GPT-4.5 也确认 smaller-model-derived data；但没有 GPT-5.6 `P0/P1` 的 mixture。
- 区分性检查：需要 model-specific、stage-specific 数据卡或 manifest。
- 结论：两者不冲突。存在性 `supported/verified by statement`，占比 `open`。

### “GPT-3 contamination 影响很小”与“过滤有 bug”

- 第一个差异：pipeline correctness 与对已检测 overlap 的 score sensitivity 是不同对象。
- clean subset score 接近只能说明在该检测规则/子集上报告分数不敏感，不能证明 bug 无后果或未知污染不存在。
- 区分性检查：复现原始 manifest、修复 filter 后重训，或至少做多种 overlap detector 和难度匹配 clean subset。
- 结论：bug 为 `verified`；广义影响大小为 `open`。

## 明确未知项

- `[未知 | open]` GPT-4、GPT-4.5、o1、GPT-5.6 的 unique/sample/loss token 数与 tokenizer accounting。
- `[未知 | open]` GPT-5.6 的精确 data cutoff、domain/language/modality mixture、dedup 和 benchmark decontamination pipeline。
- `[未知 | open]` user opt-in、human/researcher-generated、partnership 与 public data 在各训练阶段的占比。
- `[未知 | open]` GPT-4.5 smaller-model-derived data 与 GPT-5.6 的谱系关系。
- `[未知 | open]` reasoning RL 的 rollout、verifier、rejection、token budget 和与 base pretraining 的边界。
- `[未知 | open]` 最新产品别名与固定训练 checkpoint 的一一映射；线上版本可能持续更新。

## 可迁移经验与不可外推

### 可迁移

- `[综合判断 | supported]` GPT-3 展示了 mixture weight 与 corpus size 必须同时报告；只报总 tokens 会隐藏小域重复 exposure。
- `[综合判断 | supported]` 污染过滤要保存 rejection manifest 并做 clean subset sensitivity analysis；检测 bug 和负结果也应保留。
- `[综合判断 | supported]` 组织级训练数据说明、model-specific card 和 stage-specific manifest 是三种不同证据层级，不能互相替代。

### 不可外推

- 不沿用 GPT-3 mixture 推测 GPT-4/4.5/5.6。
- 不从 GPT-5.6 的能力、tokenizer 或回答内容推断语言/code/math 比例。
- 不把 RL reasoning 的性能提升归因于某类 pretraining data。
- 不把公开 source 称为开放许可，也不把 partnership 自动视为所有用途授权。
- 不把 D1 披露等级解释为数据质量低；它只说明外部审计能力有限。

## 掌握标准与推理型自测

掌握本案例后应能：

1. 用 GPT-3 表 2.2 解释 corpus quantity、mix weight 和 epochs 的不同。
2. 说明为什么 GPT-4.5 的 model-derived data 不能直接填成 GPT-5.6 synthetic percentage。
3. 区分 GPT-5.6 safety evaluation 与 pretraining held-out validation contract。
4. 在没有 token/mixture 时写出有信息量的 `unknown`，而不是根据 benchmark 猜测。

自测：如果一份新 system card 只写“加入更多高质量 code data，代码能力提升”，需要哪些 stage、token、control 和 contamination 信息，才能把关系从 empirical association 提升为受控数据因果证据？

## 一手来源

- [Brown et al., 2020, GPT-3](https://arxiv.org/abs/2005.14165)：为什么读：OpenAI 家族最完整的公开 mixture/token/contamination 锚点；重点读 §2.2、§4.1、Appendix B/C。
- [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf)：为什么读：明确写出 public/licensed source 与“不披露 dataset construction”的边界，同时包含 scaling loss 与 contamination 分析；重点读 §2、§3、§4 和 Appendix C/D。
- [Introducing GPT-4.5](https://openai.com/index/introducing-gpt-4-5/) 与 [GPT-4.5 System Card](https://openai.com/index/gpt-4-5-system-card/)：为什么读：定位 scaled pretraining、smaller-model-derived data 和 SFT/RLHF 的阶段；重点读 Scaling unsupervised learning、Introduction。
- [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)：为什么读：给出 o1 source 分类、filtering 与 large-scale reasoning RL 的高层边界；重点读 Model data and training、Regurgitation evaluations。
- [GPT-5.6 System Card, 2026-07-09](https://deploymentsafety.openai.com/gpt-5-6)：为什么读：本轮最新模型锚点；重点读 §2 Model Data and Training，并把后续 safety eval 与 pretraining validation 分开。
- [OpenAI Training Data Summary](https://help.openai.com/en/articles/20001044-training-data-summary-pursuant-to-california-civil-code-section-3111)：为什么读：确认组织级 source 类别、synthetic data 与“trillions of tokens”只适用于系统开发总体说明；不能映射为单一模型 token budget。
