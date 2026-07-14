---
title: "Alibaba Qwen 家族的数据实践"
description: "沿 Qwen 代际审计多语言、code/math、continued pretraining、质量代理和 distillation 的证据链。"
topic: "pretraining-data"
section: "china-cases"
slug: "qwen-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 101
readtime: 33
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/qwen.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/china/qwen.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:0e3318d24e0f24d8d26ff1f93e59e75aaa291d9470d29f8867f0c7d1263bbc15"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family/historical trace`
> 研究锚点：Qwen、Qwen2、Qwen2.5、Qwen2.5-Coder/Math、Qwen3、Qwen3-Coder。

## 定位与 motivating problem

Qwen 家族公开了权重、技术报告、部分训练与评测代码，并在若干代际中披露了可操作的数据实验。它尤其适合回答三个问题：质量阈值与 token 规模发生冲突时怎样决策；general model 与 code/math specialist 的 continued pretraining 如何拆分；用模型做过滤、PDF extraction、synthetic generation 和 strong-to-weak distillation 时，数据 provenance 如何表示。

本案例不把 3T、7T、18T、36T、5.5T 与 7.5T 排成单一增长曲线。前四个主要描述 general-base 代际，后两个分别属于 code specialist；Qwen3 的 36T 又包含 general、reasoning 和 long-context 三阶段。不同 tokenizer revision、checkpoint 与训练目标下，token 数只能在来源声明的范围内比较。

## 代际与阶段表

| 代际 | 阶段 | 已披露数据与规模 | 关键机制 / context | 披露 |
|---|---|---|---|---|
| Qwen 1.8B/7B/14B/72B | `P0` | 2.2T/2.4T/3.0T/3.0T；web、encyclopedia、books、code，中英占较大部分 | exact normalized + MinHash/LSH fuzzy dedup；rule/model filtering；source upsampling；2K | `D2/D3` |
| Qwen | `P3` | 精确 tokens `unknown` | longer-context continual pretraining；多数模型到 32K；L-Eval/NIAH | `D2/D3` |
| Qwen2 dense（除 0.5B） | `P0` | >7T，约 30 languages | Qwen-assisted filtering/synthesis；small-model mixture experiments | `D2/D3` |
| Qwen2-0.5B | `P0` | 12T relaxed-threshold dataset | 7T vs 12T 数据阈值反例 | `D2/D3` |
| Qwen2 MoE | `P0/P1` | 7T base lineage + additional 4.5T upcycling pretraining | checkpoint/exposure 关系有披露，unique/loss tokens `unknown` | `D2/D3` |
| Qwen2 | `P3` | 精确 tokens `unknown` | final phase 4K → 32K，增加 high-quality lengthy data；YARN/DCA extrapolate 128K | `D2/D3` |
| Qwen2.5 general | `P0/P3` | 18T；29 languages；最后从 4K → 32K | Qwen2-Instruct quality scoring/domain classification；math/code/synthetic；strategic remix | `D2/D3` |
| Qwen2.5-Turbo | `P3` | 各 stage token 数 `unknown` | 32K→64K→128K→256K；每阶段 40% max-length + 60% shorter；推理到 1M | `D2` |
| Qwen2.5 | `A1/A2` | >1M SFT examples；2 epochs、32K；offline DPO + online GRPO | critic + multi-agent filtering、execution feedback、cross-lingual translation | `D2/D3` |
| Qwen2.5-Coder | `P1/P3` | 5.2T file-level + 300B repo-level = 5.5T continued pretraining | 70% code、20% text、10% math；8K→32K；file/repo FIM | `D2/D3` |
| Qwen2.5-Math | `P1` | Qwen Math Corpus v2 >1T（v1 700B）；4K | web/books/code 多轮召回 + synthetic math；中英 | `D2/D3` |
| Qwen3 general | `P0` S1 | >30T，119 languages/dialects | 4K；language/general knowledge | `D2/D3` |
| Qwen3 reasoning | `P1/P2` S2 | about 5T higher-quality tokens | 提升 STEM/code/reasoning/synthetic；4K；加速 LR decay | `D2/D3` |
| Qwen3 long context | `P3` | hundreds of billions tokens | 32K；75% 为 16K–32K，25% 为 4K–16K；YARN/DCA 推理到 128K | `D2/D3` |
| Qwen3 flagship | `A1/A2` | long-CoT cold start → reasoning RL → thinking-mode fusion → general RL；数量 `unknown` | think/no-think hybrid；rule/model rewards | `D2/D3` |
| Qwen3 lightweight | `A1/A2` | teacher output/prompt/trajectory 数 `unknown` | off-policy response distillation → on-policy logit KL distillation | `D2/D3` |
| Qwen3-Coder | `P1/P3` | 7.5T，70% code | Qwen2.5-Coder clean/rewrite；native 256K、YaRN 1M；repo/PR dynamic data | `D1/D3` |
| Qwen3-Coder | `A2/A3` | execution-driven code RL + long-horizon agent RL；trajectory 数 `unknown` | 20,000 parallel environments 是系统并发，不是 dataset size | `D1/D3` |

Qwen3.5/3.6 等 hosted product aliases 在本轮没有可映射到固定 base-training report 的同等资料，因此不把 API 名称回填为 Qwen3 的数据配方。开放权重与 eval/finetune code 支持 `D3`，但 corpus、manifest、processing code、order 与 logs 未达到 `D4`。

## Assumption ledger

| 项 | 本笔记的处理 |
|---|---|
| 研究对象 | general base、instruct、Coder、Math 与 hosted API 分开；specialist 的 token 不并入 general model |
| token 单位 | “pre-trained on”记作 sampled exposure claim；`unique_tokens`、padding/mask 后 `loss_tokens` 通常 `unknown` |
| 阶段 | Qwen2 MoE additional 4.5T、Qwen2.5-Coder 5.5T、Qwen3 S2 均是 checkpoint-based continued/specialized exposure，不等于独立 raw corpus |
| tokenizer | Qwen 初代约 152K byte-level BPE；Qwen2/2.5/3 保持约 151K vocabulary 但 control/special tokens 改变；跨版本不做精确 token 换算 |
| mixture 单位 | Qwen2.5-Coder 70/20/10 是 token ratio；Qwen3 75/25 是 long-context length bucket ratio；二者不可互换 |
| cutoff | general model 的全面 data cutoff 多数 `unknown`；Qwen2.5-Coder public repositories 明确为 2024-02 前创建 |
| 省略效应 | 代际变化同时包含 architecture、optimizer/LR、data scale/mix、tokenizer、context 与 post-training |

## 统一数据字段表

| 字段 | Qwen / Qwen2 | Qwen2.5 / specialists | Qwen3 / Qwen3-Coder | 置信 |
|---|---|---|---|---|
| source | public web、encyclopedia、books、code；Qwen2 增加 multilingual/math/code | general source list不完整；Coder 含 GitHub、PR、commits、notebooks、Kaggle、code web；Math 含 web/books/code | web、PDF-like documents、books、multilingual、synthetic、code/STEM；Coder 含 repo/PR dynamic data | 类别 `verified`；逐 source/占比 `open` |
| rights | source-level rights `unknown`；weights 按模型 card license | GitHub public 不等于统一开放许可；weights 多为 Apache 2.0，部分规模例外 | weights Apache 2.0（按具体 card）；训练 source rights `unknown` | model license `verified`；语料 rights `open` |
| token scale | 2.2–3T；Qwen2 7T/12T，MoE +4.5T | general 18T；Coder 5.5T；Math >1T corpus | general 36T；Coder 7.5T | 报告范围 `verified`；unique/loss tokens `open` |
| language | 中英为主的 multilingual → Qwen2 约 30 | Qwen2.5 29；Math 中英；Coder 92 programming languages | 119 languages/dialects；不同语言 exposure `unknown` | coverage claim `verified`；分布 `open` |
| parse/extract | HTML extraction + language ID | 细节不完整；Coder 从 code/web 构造 file/repo samples | Qwen2.5-VL OCR/extract PDF-like docs，再由 Qwen2.5 refine | 机制存在 `verified`；error rates `open` |
| dedup | normalized exact + MinHash/LSH fuzzy dedup | general 细节 `unknown`；Coder 只高层说明 cleaning | `unknown`；不能默认沿用初代配置 | 初代 `verified by report`；后代 `open` |
| quality | rule + LM/text-quality/offensive models + manual sample audit；Qwen2 用 Qwen filter | Qwen2-Instruct multi-dimensional scoring；reward models 筛 synthetic；Coder hierarchical filters | >30T instance annotations：educational value/domain/safety；specialist models clean/generate | 存在性 `verified`；阈值/误杀 `open` |
| mixture | source upsampling；Qwen2 small proxy optimization | downsample commerce/social/entertainment，upsample tech/science/academic；Coder 70/20/10 | instance-level label mixture + proxy ablation；S2 增 STEM/code/reasoning/synthetic | 方向/已给比例 `verified`；full schedule `open` |
| decontam | Qwen 初代 13-gram；Qwen2 normalized LCS+13 并报告 strict clean subset | Coder 10-gram across pre/post；Math 13-gram + normalized LCS >0.6 | Qwen3 report未给完整 protocol | 已述范围 `verified`；全面 audit `open` |
| long context | continual/ending phase 4K→32K；token 数 `unknown` | general 4K→32K；Turbo 40/60 length curriculum；Coder repo 300B | general hundreds B 75/25 length mix；Coder native 256K | 阶段 `verified`；真实/拼接比例 `open` |
| synthetic/distill | Qwen2 用 Qwen synthesize pretraining data | Qwen2.5 teachers + general/math RMs；Coder executor retention；Math teacher synthesis | PDF refinement + trillions synthetic tokens；lightweight off/on-policy distillation | 存在/teacher `verified`；generation manifest `open` |
| artifacts | weights、reports、finetune/eval code | weights、reports、eval code | weights、reports、inference/finetune/eval code | `D3` |

## Qwen2：质量阈值与规模的反例

Qwen2 报告了一个难得的负结果：标准高质量 dataset 为 7T；放宽质量阈值得到 12T，但在试验模型上没有显著优于 7T。最终大多数 dense models 使用 7T，而 0.5B 使用 12T。

```text
same pipeline family
  -> stricter threshold -> 7T -> chosen for larger dense models
  -> relaxed threshold  -> 12T -> no significant proxy gain; used by 0.5B
```

这支持“更多 token 不自动更好”，但仍不是 data-only causal proof：报告没有给出所有 proxy 的模型规模、seeds、compute matching、重复 exposure 和 per-domain retention。也不能把 0.5B 的 12T 结果外推到 72B。

Qwen2 同时将 Qwen models 用作 low-quality filter 与 synthetic-data generator，并用 scaled-down models 优化 source/domain mixture。这形成 teacher/filter/proxy 三种角色；审计时应为同一模型记录不同 provenance edge，避免把“由 Qwen 处理”视为单一操作。

## Qwen2.5：模型过滤、domain remix 与长上下文

Qwen2.5 用 Qwen2-Instruct 做 multi-dimensional quality scoring，也用它分类 domain。报告明确指出 web-scale data 中 e-commerce、social media、entertainment 过量且常带重复、模板或机器生成内容；technology、science、academic research 相对不足，因此前者下采样、后者上采样。

```text
Qwen2-Instruct quality scores
  + domain labels
  + Qwen2/Qwen2-Math teacher generations
  + general/math reward-model filtering
  -> strategic mixture
  -> 18T general pretraining exposure
```

这是 `model-based annotation --implemented-by--> filtering/remix labels`，不说明分数已被人工校准，也没有披露每语言阈值、score distribution 或 false-positive rate。

长上下文阶段另行处理。普通 Qwen2.5 从 4K 到 32K；Turbo 用四阶段 32K→64K→128K→256K，每阶段保持：

$$
q_{length}=0.4q_{current\ max}+0.6q_{shorter}
$$

该比例按 sequence length bucket，不是 source/domain mixture，也不表示 40% loss tokens 必然来自真实长文档。推理时通过 YARN/DCA 扩到 1M，与训练最大 256K 需要分开记录。

## Qwen2.5-Coder：continued pretraining 的可操作范例

### 数据构成与 cutoff

Coder-Data 包含 source code、text-code grounding、synthetic、math、text 五类。public GitHub repositories 创建时间在 2024-02 之前，覆盖 92 programming languages；还包括 PR、commits、Jupyter notebooks 与 Kaggle data。这个日期只是 code-source 边界，不等于全部 5.5T 的统一 cutoff。

### 四级过滤与 proxy validation

Common Crawl text-code grounding 采用 coarse-to-fine hierarchical filters。四阶段后 token pool 从图中约 582B 逐步到 370B、147B、118B；使用相应数据训练 Qwen2.5-Coder-1.5B，HumanEval/MBPP 平均从 41.6 提升到 46.8。

这是一种 filter-depth proxy，但 token 数和 benchmark 同时变化。若要区分“质量提高”与“训练分布/规模变化”，需要固定 sampled budget，从各阶段 pool 重采样，保持 optimizer、seed、checkpoint 与 contamination contract。

### mixture ablation

报告比较 code:text:math 的 100:0:0、85:10:5、70:20:10，并选择：

$$
q_{coder}=0.70q_{code}+0.20q_{text}+0.10q_{math}
$$

70/20/10 在报告综合任务上优于 code 比例更高的配置。这是同一 7B setting 的 `empirical association`，不是所有 code model 的普遍最优 mix。

### file-level 到 repo-level

```text
Qwen2.5 base
  -> 5.2T file-level, 8K, NTP + FIM
  -> 300B repo-level, 32K, repo metadata/separators + repo FIM
  -> SFT + DPO
```

5.2T + 300B 与总计 5.5T 对齐，是 staged sampled exposure；不能解释为 5.5T unique code corpus。repo stage 的 128K 是 YARN inference capacity，训练 sequence 为 32K。

synthetic code 由 CodeQwen1.5 生成，并用 executor 只保留可执行代码；可执行是必要但非充分条件，不能验证语义正确、复杂度、安全或 license。post-training 的 code DPO 则使用 multilingual sandbox execution feedback 与 LLM judgment，属于 `A2`，不并入 5.5T。

## Qwen2.5-Math：污染规则必须按阶段展开

Qwen Math Corpus v2 从 700B 扩到 >1T，含 web/books/code 与 Qwen2-Math 生成的 synthetic data。它使用 13-gram matching，并加归一化后的 longest-common-subsequence ratio >0.6；pretraining 对 GSM8K/MATH 清理，SFT/RM/RL query 则对全部报告评测集做清理。

这说明同一 family 内 decontamination target 随阶段变化。`13-gram + LCS` 与 Coder 的 word-level 10-gram 也不是相同 matcher；矩阵不能只写“已去污染”而省略对象和阈值。

## Qwen3：instance-level mixture 与三阶段 pretraining

### 数据构造

Qwen3 披露 36T、119 languages/dialects，并明确两条 model-mediated data path：

```text
PDF-like documents
  -> Qwen2.5-VL text recognition
  -> Qwen2.5 refinement
  -> trillions of extracted/refined tokens

domain prompts/content
  -> Qwen2.5 / Math / Coder generation
  -> textbooks + QA + instructions + code snippets
  -> trillions of synthetic tokens
```

30T+ tokens 被标注 educational value、field、domain、safety 等维度，再通过 small-proxy ablations 做 instance-level mixture。这里的“instance-level”表示混合决策粒度比 source/domain 更细，不说明每个 instance 都有独立在线采样权重，也没有给出 label schema、calibration、proxy size 与 target-transfer error。

### 三阶段

```text
S1 General: >30T, 4K, 119 languages
  -> S2 Reasoning: ~5T, 4K, more STEM/code/reasoning/synthetic + faster LR decay
  -> S3 Long Context: hundreds B, 32K, explicit length-bucket mix
```

长阶段的 sequence mix 为：

$$
q_{long}=0.75q_{16K\text{–}32K}+0.25q_{4K\text{–}16K}
$$

36T 是家族总体 rounded exposure；由于 S1 是 “over 30T”、S2 about 5T、S3 hundreds B，不能用三个近似数反推精确总数或每阶段 `loss_tokens`。

## Qwen3：strong-to-weak distillation 不是一个步骤

旗舰模型经过 long-CoT cold start、reasoning RL、thinking-mode fusion、general RL。lightweight models 则用两阶段 distillation：

```text
teacher /think + /no_think outputs
  -> off-policy response distillation
  -> student samples /think or /no_think trajectories
  -> teacher logits on student trajectories
  -> on-policy KL distillation
```

teacher 为 Qwen3-32B 或 Qwen3-235B-A22B。报告从同一 off-policy distilled 8B checkpoint 比较 direct RL 与 on-policy distillation：RL 用 17,920 GPU hours，on-policy distillation 用 1,800 GPU hours，后者在报告 math/code/general benchmarks 上更好。这是在给定 teacher、queries、checkpoint 与实现下的 ablation；不能推广为“distillation 总优于 RL”，也不能将 GPU hours 直接换成 data quantity。

Qwen3 的三个 reward family 也应分开：rule-based、带 reference 的 model-based judge、不带 reference 的 preference RM。它们的可验证性、false positive 和 reward hacking 风险不同。

## Validation 与 contamination 审计

### Qwen2 clean-subset 分析

Qwen2 先对 pre/post-training data 做 normalized 13-token LCS 条件：

$$
|LCS(s_t,s_e)|\ge13
\quad\land\quad
|LCS(s_t,s_e)|\ge0.6\min(|s_t|,|s_e|)
$$

随后用更严格的 13-gram overlap（无 LCS ratio）从 test set 排除可疑样本，报告 original 与 non-contaminated scores。strict criterion 对 HumanEval 标出 75.0%、MATH 31.7%，但许多可能是常见代码/公式导致的 false positives；72B 的差异较小，7B 在 HumanEval/MATH clean subset 反而更高。这个结果说明 matcher 会同时产生 contamination signal 与 domain-common-pattern false positives，不能只报告“污染率”。

### 有辨识力的 data ablation

- `[来源事实 | verified]` Qwen2 的 7T vs relaxed 12T 说明 threshold relaxation 没有带来显著 proxy 改善。
- `[来源事实 | verified]` Qwen2.5-Coder 的四级 filter trace 同时记录 retained tokens 与 proxy benchmark。
- `[来源事实 | verified]` Qwen2.5-Coder 的 3-way mixture ablation给出 code/math/general 的 per-group metrics。
- `[来源事实 | verified]` Qwen3 对同一 8B starting checkpoint 比较 on-policy distillation 与 direct RL，并报告 GPU hours。
- `[来源事实 | verified]` Qwen3 long-context evaluation 给出 RULER 按 4K–128K 分长度结果，并观察 thinking mode 在 retrieval tasks 上轻微退化。

### 仍缺失的 validation contract

- `[未知 | open]` pretraining held-out data 的来源、冻结日期、每语言/domain 数量、与训练集的 dedup scope。
- `[未知 | open]` mixture/filter proxy 的模型规模、training tokens、seeds、checkpoint rule 与 target-scale rank correlation。
- `[未知 | open]` Qwen3 synthetic/PDF data 的 source leakage、OCR/refinement error 和 teacher memorization audit。
- `[未知 | open]` 36T/7.5T 代际的统一 benchmark decontamination protocol与 clean-subset 结果。

## 表面冲突与区分性检查

### Qwen2 是 7T 还是 12T？

- 7T 是高质量选择并用于大多数 dense models；12T 是放宽 threshold 的 dataset，用于 Qwen2-0.5B。
- 第一个差异是 dataset variant 与 model identity，不是同一 run 的冲突。
- 区分性检查：固定模型/compute/sampler，比较两个 pool 的 per-domain retention 与多 seed target metrics。
- 处理：逐模型分行，不能把 Qwen2 family 一律记为 12T。`verified`。

### Qwen2.5-Coder 是 5.2T、5.5T 还是 “over 5.5T”？

- technical report 给 file-level 5.2T + repo-level 300B = 5.5T；摘要有 “over 5.5T” 的 rounded wording。
- 第一个差异是 stage-specific accounting 与高层 rounding。
- 处理：阶段表写 5.2T/300B，family summary 写 5.5T，不用 “over” 推导额外未披露 tokens。`supported`。

### Qwen3 36T 与 30T+5T+hundreds B

- 36T 是 total rounded claim；三个 stage 各自也使用近似量词。
- 第一个差异是 summary 与 stage rounding，且 S1 的 “over 30T” 上界未知。
- 区分性检查：需要 fixed run 的 step/batch/sequence mask accounting。
- 处理：不将近似阶段数精确求和，不生成伪精确 35.xT。`open`。

### Qwen2 的 13/LCS 与 Coder 的 10-gram、Math 的 13/LCS

- matcher 的 token/word unit、normalization、target benchmarks 与训练阶段不同。
- 第一个差异是 matching contract，不是“严格程度”可以单轴排序的问题。
- 区分性检查：在同一 benchmark/source fixture 上测 precision/recall、common-pattern false positive 与 near-copy false negative。
- 处理：逐模型保留规则，不统一成“n-gram decontam”。`verified/open`。

## 明确未知项

- `[未知 | open]` 各 general 代完整 source list、source-level rights、raw/processed scale、dataset manifest 与统一 cutoff。
- `[未知 | open]` 3T/7T/18T/36T/5.5T/7.5T 的 unique、sampled、non-padding 与 loss-token contract。
- `[未知 | open]` Qwen2.5/3 的 exact/near/subdocument dedup 算法、scope、threshold 与 removal distribution。
- `[未知 | open]` model-based filter/annotation 的 rubric、score calibration、每语言阈值、human agreement 与 false-positive rate。
- `[未知 | open]` Qwen3 instance-level sampling weights、stage order/seed、重复 exposure 和 source transitions。
- `[未知 | open]` PDF-like source rights、OCR quality、layout/table/math preservation 与 Qwen2.5 refinement hallucination rate。
- `[未知 | open]` synthetic pretraining 的 prompts、teacher revisions、rejection rate、diversity/duplication 与 contamination audit。
- `[未知 | open]` hosted Qwen3.5/3.6 aliases 与公开 Qwen3/Qwen3-Coder checkpoints 的精确训练 lineage。

## 可迁移经验与不可外推

### 可迁移

- `[综合判断 | supported]` 放宽质量阈值前应以固定 sampled budget 做 proxy，并把 retained scale 与收益共同记录。
- `[综合判断 | supported]` code specialist 的 general/math data 不是“杂质”；mixture ablation 应同时检查 code、math 与 general retention。
- `[综合判断 | supported]` long-context 配方应报告 sequence-length histogram，而不只写最大 context。
- `[综合判断 | supported]` model-mediated curation 要区分 extractor、refiner、filter、generator、judge 与 teacher 六种角色。
- `[综合判断 | supported]` contamination audit 应同时报告 matcher contract、flagged rate、clean-subset delta 与人工 precision sample。
- `[综合判断 | supported]` strong-to-weak distillation 需要分别记录 off-policy teacher responses 与 on-policy student trajectories。

### 不可外推

- Qwen2 的 7T 优于 relaxed 12T 不代表所有更大数据集都更差，也不证明质量可由单一 threshold 表示。
- Qwen2.5-Coder 的 70/20/10 不是跨模型、跨语言、跨代码任务的普遍最优 mixture。
- 92 programming languages 或 119 human languages 不等于均匀 exposure、支持质量或相同 tokenizer fertility。
- 1M inference context 不等于 1M training sequences；Qwen2.5-Turbo 训练最大长度为 256K。
- Qwen3 的 teacher-generated trillions 不能仅凭 benchmark 提升证明无 diversity collapse、错误放大或版权风险。
- 8B 上 on-policy distillation 优于 direct RL 的结果不能直接外推到 flagship models 或其他 reward/teacher setting。

## 掌握标准与推理型自测

掌握本案例后应能：

1. 解释 Qwen2 7T/12T 与 Qwen2.5-Coder 5.2T/300B 的模型和阶段边界。
2. 区分 source/domain mixture、instance-level quality labels 与 sequence-length mixture。
3. 恢复 Qwen3 的 PDF extraction、synthetic generation、三阶段 pretraining 与两阶段 distillation provenance。
4. 说明为什么 clean-subset score delta 比单独的 “contamination percentage” 更有信息。
5. 为 model-based multilingual filtering 设计按语言的 calibration、retention 与 downstream validation。

自测：如果一个 Qwen3-style proxy 建议把 educational-value top bucket 上采样 4×，但低资源语言的 retention 降低 35%，你会怎样构造 micro/macro/per-language validation、固定 token budget ablation 和 target-scale transfer check？

## 一手来源

- [Qwen Technical Report, arXiv:2309.16609v1](https://arxiv.org/abs/2309.16609)：为什么读：固定初代 source、exact/fuzzy dedup、filter/upsampling、13-gram 与 tokenizer；重点读 §2.1–2.5。
- [Qwen2 Technical Report, arXiv:2407.10671](https://arxiv.org/abs/2407.10671)：为什么读：固定 7T vs 12T 负结果、MoE +4.5T、约 30 languages、long-context finishing 与 clean-subset contamination；重点读 §3、§5.2.6。
- [Qwen2.5 Technical Report, arXiv:2412.15115](https://arxiv.org/abs/2412.15115)：为什么读：固定 18T、model-based filtering/synthetic/remix、Turbo length curriculum 与 >1M SFT examples；重点读 §3–4。
- [Qwen2.5-Coder Technical Report, arXiv:2409.12186](https://arxiv.org/abs/2409.12186)：为什么读：固定五类数据、2024-02 code cutoff、四级过滤、70/20/10、5.2T→300B stages、executor 与 10-gram；重点读 §3–6。
- [Qwen2.5-Math official release](https://qwenlm.github.io/blog/qwen2.5-math/)：为什么读：固定 700B→>1T math corpus、synthetic data 与 pre/SFT/RM/RL 分阶段 decontamination。
- [Qwen3 Technical Report, arXiv:2505.09388](https://arxiv.org/abs/2505.09388)：为什么读：固定 36T/119 languages、PDF/synthetic trillions、30T+ instance labels、三阶段 pretraining、四阶段 post-training 与 two-phase distillation；重点读 §3–4、§6.4。
- [Qwen3 official release, 2025-04-29](https://qwenlm.github.io/blog/qwen3/)：为什么读：作为报告的官方摘要与 release identity 对照；定量字段以技术报告为主。
- [Qwen3-Coder official release, 2025-07-22](https://qwenlm.github.io/blog/qwen3-coder/)：为什么读：固定 7.5T/70% code、256K/1M、synthetic clean/rewrite 与 20,000 parallel RL environments；注意它的披露粒度低于 technical report。
