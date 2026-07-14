---
title: "Zhipu / Z.ai GLM 家族的数据实践"
description: "沿模型代际拆分 ARC、semantic dedup、thinking mode、长上下文与 async agent RL 的训练阶段。"
topic: "pretraining-data"
section: "china-cases"
slug: "glm-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 105
readtime: 31
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/glm.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/china/glm.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:09630a0b2924c6ab35953bddf5ceaa392409d343cf2cc6035c7407945792fb9f"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family/historical trace`
> 研究锚点：GLM-4.5、GLM-5；GLM-5.1/5.2 只记录官方模型卡可确认的增量。

## 定位与 motivating problem

GLM-4.5 与 GLM-5 是少数把 base pretraining、code/reasoning mid-training、long-context adaptation、SFT、reasoning RL 和 agent RL 串成完整链路的公开报告。它们适合回答：

1. “23T/28.5T”怎样拆回各阶段，而不是被当成一个 corpus size；
2. web quality、semantic dedup、code repository、issue/PR 与 synthetic reasoning 如何改变 sampled distribution；
3. thinking/non-thinking、tool trajectory 与 asynchronous agent RL 的训练数据分别从哪里来；
4. 长上下文、agent 能力和优化器/架构同时变化时，公开 benchmark 能识别到什么程度。

本笔记把 GLM-4.5 的 `15T + 7T + 500B + 500B + 100B` 与 GLM-5 的 `27T + 1T + 500B + 50B` 保留为报告中的阶段口径。前者算术和为 23.1T、后者为 28.55T，而报告标题分别使用 23T、28.5T；差额按 rounding discrepancy 记录，不擅自删改某一阶段。

## 代际与阶段表

| 模型 | 阶段 | 已披露 sampled exposure | 数据与 context | 披露 |
|---|---|---:|---|---|
| GLM-4.5 | `P0` general pretraining | 15T | web、social media、books、papers、code repositories；4K | `D2/D3` |
| GLM-4.5 | `P1` code/reasoning continual pretraining | 7T | 上采样 code、math、science；4K | `D2/D3` |
| GLM-4.5 | `P1/P3` repository code | 500B | repo-level code、issue/PR/commit；32K | `D2/D3` |
| GLM-4.5 | `P1/P3` synthetic reasoning | 500B | synthetic math/science/code reasoning；32K | `D2/D3` |
| GLM-4.5 | `P3` long context & agent | 100B | long documents、synthetic agent trajectories；128K | `D2/D3` |
| GLM-4.5 | `A1` SFT/self-distillation | millions of samples；精确值 `unknown` | reasoning、agent、general experts；thinking/non-thinking；128K | `D2/D3` |
| GLM-4.5 | `A2` RL | prompts/rollouts/tokens `unknown` | reasoning、coding、science、search、SWE、function calling、general | `D2/D3` |
| GLM-5 | `P0/P1` pretraining | 27T | general/code/math/science；pretraining context 阶段细分未完整披露 | `D2/D3` |
| GLM-5 | `P1/P3` 32K mid-training | 1T | long documents、repo/SWE、synthetic agent data | `D2/D3` |
| GLM-5 | `P3` 128K mid-training | 500B | 上采样 long documents 与 synthetic agents | `D2/D3` |
| GLM-5 | `P3` 200K mid-training | 50B | 极长依赖；少量 MRCR-like synthetic data | `D2/D3` |
| GLM-5 | architecture adaptation | 约 2.84B dense-adaptation sampled tokens；另有 20B sparse adaptation | 14 sequences × 202,752 tokens × 1,000 steps；是否包含于 28.5T `unknown` | `D2/D3` |
| GLM-5 | `A1` SFT | 数量 `unknown` | general、reasoning、coding/agent；202,752 context；per-turn thinking control | `D2/D3` |
| GLM-5 | `A2/A3` RL/distillation | >10K real-world agent environments；prompt/trajectory/token 数 `unknown` | SWE、terminal、search、slides；async rollout；on-policy cross-stage distillation | `D2/D3` |

两代 headline 的核算关系是：

$$
15+7+0.5+0.5+0.1=23.1\text{T}\approx23\text{T}
$$

$$
27+1+0.5+0.05=28.55\text{T}\approx28.5\text{T}
$$

变量单位是报告声明的 sampled tokens；不是 unique corpus tokens，也不是 attention/loss mask 后的 effective tokens。0.1T 和 0.05T 的表面差额与报告使用一位小数/整数 headline 相容，但没有 step log，故关系标记为 `[推导结论 | supported]`，不是 exact identity。

## Assumption ledger

| 项 | 本笔记的处理 |
|---|---|
| 研究对象 | base、mid-training、SFT、RL、agent environment 与 rollout 分开；不把整个 GLM 家族汇成单一配方 |
| token 单位 | 所有 T/B 数先记 sampled exposure；unique、non-padding、loss tokens 均 `unknown` |
| 数据版本 | 报告未提供可下载 manifest、snapshot hash 与完整 cutoff；source 类别不等于固定 corpus |
| mixture | “upsample/highest quality >3.2 epochs”等为相对调度；没有完整 token proportion 时不反推 |
| synthetic | reasoning response、agent task、environment、trajectory 与 teacher logits 是不同 artifact |
| architecture adaptation | DSA dense/sparse adaptation 是否计入 28.5T 未明确；单列，不强行相加 |
| lineage | GLM-5.1/5.2 的官方卡引用 GLM-5 报告，但未披露完整训练数据 diff；不默认完全同配方 |
| 省略效应 | 两代同时改变参数、MoE/attention、optimizer、context、数据、SFT/RL 与基础设施 |

## 统一数据字段表

| 字段 | GLM-4.5 | GLM-5 | 置信 |
|---|---|---|---|
| source | web、social、books、papers、code repos；FineWeb-2；repo issue/PR/commits；synthetic reasoning/agent | 更新 web/code/math/science；SWE issue–PR、terminal、search URLs、slides；long documents | 类别 `verified`；清单 `open` |
| rights | source-level licenses、consent、opt-out 与删除流程未系统披露 | public/real-world 不等于统一许可；完整 rights ledger `unknown` | `open` |
| cutoff | `unknown` | `unknown`；搜索环境 URL 构建日期未给固定 manifest | `open` |
| token scale | 23T headline；分项 23.1T | 28.5T headline；分项 28.55T；DSA adaptation 归属不明 | 官方数字 `verified`；闭合关系 `supported/open` |
| parse/extract | web、code、paper 处理有高层描述；parser metrics `unknown` | 改进 PDF/文档解析，长文按 chunk 聚合质量分数 | 机制 `verified`；误差 `open` |
| dedup | MinHash + embedding SemDedup；后者补 templated-page 漏检 | web/code/long doc 去重；search graph >2M deduplicated URLs | 高层机制 `verified`；参数/版本 `open` |
| quality | web quality buckets；教育价值/领域 classifier；code 三档；math/science LLM score | DCLM sentence-embedding classifier + world-knowledge classifier；code/math/science刷新 | `verified/supported` |
| mixture | high-quality web >3.2 epochs；最低档丢弃；第二阶段提升 code/math/science | 后期提升 long documents/synthetic agents；完整比例 `unknown` | 已述关系 `verified`；分布 `open` |
| synthetic | 500B reasoning + long-context agent trajectories；post-training prompt/trajectory synthesis | long-dependency、terminal/search/task/environment、SFT/RL trajectories | 机制 `verified`；provenance manifest `open` |
| decontam | 无覆盖训练语料与所有 benchmark 的完整 protocol | 无完整 protocol；source-specific verifier 不等于 decontamination | `open` |
| long context | 32K repo/reasoning → 128K long docs/agents；mid-training best-fit packing | 32K 1T → 128K 500B → 200K 50B；自然长文 + synthetic dependency | scheduler `verified`；data-only effect `open` |
| validation | base/reasoning/agent suites；若干 small-model prompt/RL ablation | base、long-context、agent/SWE/search suites；system/infra diagnostics | 评测存在 `verified`；归因 `open` |
| artifacts | weights、report、inference/training framework；无训练 corpus/order log | weights、report、inference；无完整 corpus/manifest/loader state | `D2/D3` |

## GLM-4.5：从 web bucket 到 ARC 数据链路

### quality 与 dedup 不是同一个操作

GLM-4.5 将 web 文档按质量分桶，最高质量部分重复采样超过 3.2 epochs，最低质量部分排除。对 multilingual web，又结合自采网页与 FineWeb-2，训练 educational-quality classifier 并上采样高分文档。

MinHash 对字面近重复有效，但对共享模板、正文改写或页面结构相似的集合可能漏检；报告因此增加 embedding-based SemDedup。其 typed relation 是：

```text
MinHash --detects lexical near-duplicates--> candidate clusters
SemDedup --detects embedding-near duplicates missed by templates/paraphrase--> extra clusters
quality bucket --controls retained/sampled weight--> training exposure
```

SemDedup 不是 MinHash 的 exact replacement，也不自动保证保留 cluster 中最优代表。要复现需记录 embedding checkpoint、distance、cluster rule、representative selection 以及 quality-before/after-dedup 的执行次序；报告未全部公开。

### code 与 math/science 的分域处理

代码先做 rule filtering，再按编程语言分别分成质量档；高档上采样，低档排除，并对 source code 使用 fill-in-the-middle。web 中的 code content 另用 FastText/tag retrieval、quality model 与 parser 处理。

math/science 来自 web、books、papers。LLM 评估教育内容占比并训练较小 classifier，再按阈值上采样。这里的“quality”统计单位可能是 document、chunk 或 page，报告没有统一 confusion matrix，因此不能把 code tier 与 math score 横向当成同一标尺。

### ARC：agent、reasoning、coding 的中期训练

GLM-4.5 的 ARC 可还原为：

```text
15T general @4K
 -> 7T code/math/science @4K
 -> 500B repo code + 500B synthetic reasoning @32K
 -> 100B long documents + synthetic agent trajectories @128K
```

repo 数据将文件与 issue、pull request、commit information 组织成完整上下文，也使用 diff 表示。32K synthetic reasoning 覆盖 math/science/code；128K 阶段加入长文与大型 synthetic agent trajectories。中期训练使用 best-fit packing 以降低长样本截断，而早期 pretraining 可随机截断。这意味着“长数据占比”至少受三件事影响：source sampling、length bucket 与 packer admission；只报告 raw document count 不足以复核 exposure。

## GLM-4.5：thinking/non-thinking 与 expert self-distillation

post-training 第一阶段分别训练 reasoning、agent 与 general expert；第二阶段把专家行为 self-distill 到统一模型。SFT 总量为 millions of samples，最大 128K，同时混合完整 reasoning traces 与不显式展示思维过程的回答，从而支持 thinking/non-thinking 两种模式。

这条链路应记录：

```text
prompt provenance
 -> expert assignment/checkpoint
 -> response or tool trajectory
 -> verifier/judge/rejection reason
 -> thinking-mode label
 -> unified SFT example
```

报告披露了机制但没有 example-level manifest、teacher revision、采样温度、去重和 prompt overlap，因此“专家生成数据改善统一模型”只在该联合 recipe 中是 `[来源事实 | supported]`，不能视作通用蒸馏定律。

function calling 使用 XML-like template 以减少代码 escaping。格式模板是 serialization intervention，不等于工具数据本身更正确；validation 应分开测 schema validity、argument correctness、terminal success 与 multi-turn recovery。

## GLM-4.5：RL prompt、environment 与 trajectory

### reasoning/general RL

rejection sampling 会过滤重复、过短、截断、格式错误的输出；客观题使用 correctness verifier，主观任务使用 reward model，工具任务检查 terminal state。报告还给出小规模局部结果：按 response length 删除最容易的底部 50% prompts，在 math/science 上提升约 2–4%；困难题为每题保留四个候选 responses 又增加约 1–2%。这些是特定小模型/配方下的 empirical association，不应外推为固定阈值。

science RL 报告倾向使用 expert-verified multiple-choice data；code/science 对 loss aggregation 的选择也不同。general RL 约 5K prompts，并用 7/33/139 的能力 taxonomy；instruction following 使用 7/151 taxonomy。这里的 5K 是 prompt 数，不是 rollout 或 token 数。

### agent data factory

```text
tool/framework/MCP inventory
 -> synthetic tools
 -> tasks
 -> user-simulator interaction
 -> multi-turn trajectories
 -> multiple judge agents
 -> successful retained trajectories
```

web-search tasks 从 multi-hop knowledge graph 构造并经过人工抽取/混淆；SWE tasks 从真实 issue/PR 形成并带 executable tests。训练 loss 只作用于 model tokens，environment feedback 不计入 loss，但它仍占 context/compute 并影响后续 action distribution。

因此至少区分：

$$
N_{tasks}\neq N_{environments}\neq N_{trajectories}\neq N_{toolcalls}\neq T_{loss}
$$

reasoning/general 可同步 rollout，agent 由于环境延迟使用异步路径。基础设施支持这些数据流，但公开 slime/data-buffer schema 不等于训练 snapshot 已复现。

## GLM-5：27T base 与三档 mid-training

### base corpus 的三个增量

GLM-5 沿用大类但补了更具体的处理：

- web 增加 DCLM sentence-embedding classifier；另用 Wikipedia 与 LLM labels 训练 world-knowledge classifier，尝试恢复普通质量模型低估的长尾知识；
- code 刷新代码托管站点与网页源，修复 Software Heritage metadata，改进语言分类与低资源编程语言模型；报告称 fuzzy-deduplicated unique tokens 增加 28%；
- math/science 改进 PDF/document parsing、LLM quality scoring 与长文 chunk aggregation；该基础组成明确避免 synthetic、AI-generated 与 template data。

“基础 math/science 不含 synthetic”是 component-specific claim，不代表 GLM-5 整个 27T 或 mid-training 不含合成数据。code 的 28% 也是相对于内部前代 pipeline 的 unique-token change；没有公开 manifest，不能换算成 total sampled exposure。

### 32K→128K→200K

GLM-5 mid-training 在后段逐步提高 long documents 与 synthetic agents。SWE 数据包含 repository files、diff、issues 与 PR；团队放宽 repository filter、强化 issue filter，最终 issue–PR 部分约 160B unique tokens。这个 160B 是 unique corpus slice，不应与 1T sampled exposure直接相加。

自然长文来自 books、papers 与 general documents，经过 perplexity、dedup、length 筛选并提升 knowledge-rich material。synthetic long dependency 通过把相关文本交错组织产生；200K 阶段还加入少量 MRCR-like synthetic data。其合理验证不是只测 needle retrieval，而应按自然/合成、长度和任务类型分别报告，并保留短上下文能力。

### DSA adaptation 的 22.84B 边界项

报告在 mid-training 末端适配 Dynamic Sparse Attention。dense adaptation 的可计算 exposure 是：

$$
1{,}000\times14\times202{,}752=2{,}838{,}528{,}000\approx2.84\text{B tokens}
$$

另有 20B sparse adaptation。报告没有清楚说明 2.84B/20B 是否已经包含在 28.5T headline，也未给 padding/loss mask 后 token 数，因此它们单列为 `ambiguous`。这是 architecture-data co-adaptation，不可将之后 long-context/agent 改善完全归因于 corpus。

## GLM-5：SFT、异步 agent RL 与可追溯 token IDs

SFT 统一 general、reasoning、coding/agent，最大 context 202,752。相对前代，数据支持 interleaved thinking、coding 多轮中保留 thinking state、以及 per-turn thinking control。执行环境里允许保留含错误的早期 trajectory segment，但在 loss 中 mask；这保护了恢复路径上下文，又避免直接模仿错误 action。

该做法要求同时报告：

$$
T_{context}=T_{supervised}+T_{masked}+T_{environment}+T_{padding}
$$

只报 sequence length 会隐藏真正产生梯度的比例。

GLM-5 的异步 agent RL 覆盖超过 10K real-world SWE、terminal、search environments。TITO gateway 保留 rollout 的 exact token IDs 与 metadata，避免把文本重新 tokenize 后产生 off-policy mismatch；再配合 model-version gap 过滤 stale samples、排除 environment collapse、处理 group padding/drop，并用 DP-aware routing。

主要数据源与构造：

- SWE：数千 repositories、9 种语言、>10K environments；issue/test/repo snapshot 的许可证与日期仍需 manifest；
- terminal：seed → task draft → environment construction → refinement，超过 90% Docker construction 成功率是环境构建指标，不是任务正确率；
- search：从 >2M deduplicated URLs 构造 world-knowledge graph，再按 tool-free/early-agent 成功率与 verifier 一致性分阶段筛难度；
- slides：生成任务先 RL，再做 rejection SFT/masking，并专门检查 reward hacking。

on-policy cross-stage distillation 继续把前阶段 policy 产生的 trajectories 交给更强阶段信号，而不是固定离线 teacher answer。provenance 至少需要 model version、rollout token IDs、prompt/environment snapshot、tool responses、reward/judge version、mask 与 acceptance reason。

## Validation、ablation 与区分性检查

| 公开结论 | 首个混淆 | 有辨识力的检查 |
|---|---|---|
| SemDedup 补 MinHash 漏检 | embedding 同时改变 topic/quality composition | 固定 raw URLs 与 retained tokens，比较 exact/minhash/semantic clusters、代表选择、per-domain loss 与 memorization |
| high-quality web 多 epoch 有益 | quality 与 repetition 同时变化 | quality-matched unique expansion vs repetition；固定 sampled tokens，报告 train/held-out gap |
| ARC 提升 code/reasoning/agent | data、context、packer、LR、Muon/MTP 同时变化 | stage-boundary checkpoints + matched-token replay；按 repo/synthetic/long-agent 分桶 |
| length-based hard prompt selection 有益 | response length 只是 difficulty proxy | 与 pass-rate、verifier margin、human difficulty 对照，报告保留域与错误类型 |
| GLM-5 长上下文更强 | corpus、200K schedule、DSA 同时改变 | natural/synthetic × context × DSA 的 factorial ablation；同时测短上下文回退 |
| async agent RL 可扩展 | environment reliability 与 policy learning 混合 | 分开报 build/reset/success、staleness、tool latency、reward、task success 与 loss tokens |

已公开 benchmark 能验证 checkpoint 行为，但不能反推出训练数据的独立因果。特别是 GLM-4.5/5 的参数规模、attention、MoE、optimizer 与 post-training 都变化，跨代总分只能标 `empirical association`。

## 表面冲突与处理

1. **23T vs 23.1T**：分项算术为 23.1T，headline 23T；按 rounding 保留，`supported`。
2. **28.5T vs 28.55T**：同理为一位小数舍入；但 DSA 22.84B 是否另计仍为 `open`。
3. **sampled vs unique**：GLM-5 的 160B issue–PR 与 code +28% 是 unique-corpus 指标，1T/27T 是 exposure；不能相加。
4. **natural-only reasoning data**：只限定 GLM-5 base math/science component；mid-training 和 post-training 明确包含 synthetic。
5. **“real-world environment” vs rights**：真实 repository/terminal/search 来源不自动建立许可、隐私、cutoff 或 benchmark isolation。
6. **GLM-5.1/5.2 lineage**：官方模型卡显示它们继续强化长程 agentic engineering，但没有可闭合的数据 recipe diff；所有新增 source/token 字段为 `unknown`。

## 可迁移结论与不可外推部分

### 可迁移

- token headline 必须拆到 stage、context 与 artifact type；小数舍入也应显式保留。
- lexical dedup、semantic dedup、quality scoring 与 representative selection 应分别记录。
- repo code、issue/PR、environment、task、trajectory、tool call 与 loss token 必须分单位统计。
- 长上下文训练要同时保存 length bucket、packer/truncation、自然/合成来源与短上下文回归。
- 异步 RL 的 exact token IDs、policy version、environment snapshot 与 reward version 是可审计 provenance 的核心。
- thinking/non-thinking 或 masked-error trajectory 需要 loss-mask accounting，不能只报 sequence count。

### 不可外推

- 不把 >3.2 epochs、50% length filter、四候选 response 或特定 RL aggregation 当成通用最优值。
- 不用 GLM benchmark 排名证明某个 web filter、Muon、DSA 或 agent corpus 的单独作用。
- 不把 source 类别列表理解为 rights、cutoff、language mix 或 contamination 已披露。
- 不假设 GLM-5.1/5.2 沿用 GLM-5 的全部数据，亦不根据模型能力反推新增语料。

## 明确未知项

- 完整 URL/repository/document manifest、snapshot date、cutoff、许可证、opt-out 与删除记录；
- tokenizer revision、unique/sample/loss token 对账、packing/padding/mask 统计；
- 各语言、domain、source、quality bucket 与自然/合成数据的 token mixture；
- dedup threshold、embedding checkpoint、cluster representative 与 benchmark decontamination protocol；
- synthetic prompt/response/trajectory 的 teacher、seed、生成参数和 prompt overlap；
- >10K environments 的固定 snapshot、成功构建分母、task count、rollout count 与 tool-call distribution；
- GLM-5 DSA adaptation 是否包含在 28.5T；GLM-5.1/5.2 的训练数据增量。

## 掌握标准与推理型自测

完成本笔记后，应能：

1. 重建两代 headline 的阶段算术，并解释为何它们不是 unique corpus size；
2. 画出 ARC 与 GLM-5 agent data provenance graph；
3. 区分 issue–PR unique tokens、mid-training sampled tokens、trajectory context tokens 与 supervised loss tokens；
4. 为 SemDedup、长上下文和异步 agent RL 各设计一个有区分力的对照实验；
5. 说明为什么 GLM-5.1/5.2 的能力变化不能补全未披露的数据字段。

自测：

1. 如果 160B issue–PR unique tokens 在 1T 阶段被平均采样 2.5 次，它最多解释多少 sampled tokens？剩余 exposure 还可能来自哪些 source？
2. 若 202,752-token trajectory 中 35% 为 environment feedback、10% 为 masked errors、5% 为 padding，最多多少 token 参与监督 loss？
3. 如何区分 SemDedup 的收益来自“去重复”还是来自它无意改变了 topic/quality mixture？
4. 为什么 `23.1T≈23T` 可以是合理近似，而把 DSA 22.84B 自动加到 28.55T 却不成立？

## 原始来源与阅读位置

- [GLM-4.5: Agentic, Reasoning, and Coding (arXiv)](https://arxiv.org/abs/2508.06471)：主锚点；读 pre-training data、mid-training、expert training、RL data 与 agent infrastructure。
- [GLM-4.5 official repository](https://github.com/zai-org/GLM-4.5)：核对权重、推理实现、许可证与报告版本；不等同于训练数据发布。
- [GLM-5: from Vibe Coding to Agentic Engineering (arXiv)](https://arxiv.org/abs/2602.15763)：读 27T corpus changes、1T/500B/50B mid-training、DSA adaptation、SFT mask、async agent RL 与环境构造。
- [GLM-5 official model card](https://huggingface.co/zai-org/GLM-5)：核对 checkpoint、context、参数与报告链接；参数总数可能受 embedding/output counting convention 影响。
- [GLM-5.1 official model card](https://huggingface.co/zai-org/GLM-5.1)：只用于确认该 checkpoint 的官方定位与 GLM-5 报告 lineage；未用它补写未披露数据。
- [GLM-5.2 official model card](https://huggingface.co/zai-org/GLM-5.2)：用于核对最新家族入口；训练数据增量仍记 `unknown`。

## 与主线的关系

```text
token-accounting --audits--> 23T/28.5T stage arithmetic
deduplication --implemented-by--> MinHash + SemDedup
data-mixture --scheduled-as--> general -> ARC -> long/agent
long-context-data --depends-on--> source length + packing + context adaptation
agent-data --provenance-through--> task -> environment -> trajectory -> judge -> loss mask
validation --cannot-identify-alone--> data-only causal effect
```
