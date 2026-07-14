---
title: "Xiaomi MiMo 家族的数据实践"
description: "分层表示 reasoning-aware extraction、多阶段 mixture、多模态核算、verifier 与 agent 数据 provenance。"
topic: "pretraining-data"
section: "china-cases"
slug: "mimo-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 102
readtime: 34
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/mimo.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/china/mimo.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:8ee891feeeefc56fea2fb13bdb2e2b7bb7a2c8448b88ef5b6ba90d98deeefc2b"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family/historical trace`
> 研究锚点：MiMo-7B、MiMo-VL-7B、MiMo-V2-Flash、MiMo-V2.5、MiMo-V2.5-Pro。

## 定位与 motivating problem

MiMo 的公开材料把“提高 reasoning-pattern density”从口号展开成了可检查的数据操作：保留网页中的公式与代码、全局去重后再按质量重配、用小模型做多维标签、生成 synthetic reasoning responses，并在训练末段同时改变 mixture 与 context length。V2-Flash 又公开了 22T→26T→27T 的调度、长依赖代码源和大规模可验证 agent environments，因此适合研究三个问题：

1. extraction、filtering、mixture 与 synthetic generation 怎样共同改变“推理数据密度”；
2. token curriculum 与 context curriculum 同时变化时，怎样避免错误归因；
3. pretraining samples、SFT pairs、RL prompts、agent environments 与 on-policy trajectories 怎样分别计量。

本笔记不把 25T、2.4T、27T、48T 和 27T 排成一个单调 scaling curve。MiMo-VL 是从 MiMo-7B 语言骨干出发的多模态训练；MiMo-V2.5（310B/15B active）与 MiMo-V2.5-Pro（1.02T/42B active）是两个并列规模，不是同一模型的 48T/27T 冲突记录。

## 代际与阶段表

| 模型 | 阶段 | 已披露数据与规模 | mixture / context | 披露 |
|---|---|---|---|---|
| MiMo-7B-Base | `P0` Stage 1 | 约 18T：由学习率日程可还原为 84B warmup + 10.2T constant + 7.5T decay；不是独立 unique corpus 计数 | 排除 reasoning-query synthetic responses；压低 ads/news/jobs/低知识密度，提升专业高质量域；8K | `D2/D3` |
| MiMo-7B-Base | `P1` Stage 2 | 4T sampled exposure | math + code 提升到约 70%；8K | `D2/D3` |
| MiMo-7B-Base | `P1/P3` Stage 3 | 2T，其中末 500B 再降学习率；总计约 25T | 约 10% math/code/creative-writing synthetic responses；8K→32K | `D2/D3` |
| MiMo-7B-SFT / RL | `A1/A2` | 初版 SFT 约 500K、后续 checkpoint 扩至 6M；RL 为 100K math + 30K code problems | 规则 verifier；code test difficulty reward；10% easy-pool resampling；RL context 后续 32K→48K | `D2/D3` |
| MiMo-VL-7B-SFT | multimodal `P0/P1/A1` | 四阶段合计 2.4T tokens；各阶段 tokens `unknown` | projector warmup → vision-language alignment → general multimodal pretraining → long-context SFT | `D2/D3` |
| MiMo-VL-7B-RL | `A2` | prompt/rollout/loss tokens `unknown` | Mixed On-policy RL：perception、grounding、reasoning、human/AI preference；text/image/video | `D2/D3` |
| MiMo-V2-Flash-Base | `P0` Stage 1 | 0–22T | general high-quality corpus；native 32K | `D2/D3` |
| MiMo-V2-Flash-Base | `P1` Stage 2 | 22–26T | 上采样 code；约 5% synthetic reasoning；32K | `D2/D3` |
| MiMo-V2-Flash-Base | `P3` Stage 3 | 26–27T | 沿用 Stage 2 分布并上采样 long-range-dependency data；32K→256K | `D2/D3` |
| MiMo-V2-Flash | `A1` | millions of SFT samples；精确数 `unknown` | general conversation、reasoning、coding、agent；thinking/non-thinking | `D2/D3` |
| MiMo-V2-Flash | `A2/A3` | 90K real-prompt code-agent + 30K synthetic-prompt code-agent + 150K search-agent + 50K general-agent tasks | specialized RL teachers → multi-teacher on-policy distillation；约 120K environments 的 code-agent on-policy trace | `D2/D3` |
| MiMo-V2.5-Base | `P0/P1/P3` | 总计约 48T；阶段细分 `unknown` | text pretraining → projector warmup → multimodal pretraining；base context 256K | `D1/D3` |
| MiMo-V2.5 | `A1/A2/A3` | 数量 `unknown` | SFT + agentic RL；context 32K→256K→1M；MOPD | `D1/D3` |
| MiMo-V2.5-Pro-Base | `P0/P3` | 27T；native 32K，base context 256K | source/mix、阶段边界与 long-context token 数 `unknown` | `D1/D3` |
| MiMo-V2.5-Pro | `A1/A2/A3` | 数量 `unknown` | SFT → domain-specialized RL teachers → MOPD；推理 context 1M | `D1/D3` |

MiMo-7B Stage 1 的约 18T 来自官方优化器日程中的 exposure 边界；这是 `implementation relation`，不是论文直接给出的 corpus-size row。优化器段落可直接读出的三个阶段约为 17.784T + 4T + 2T = 23.784T，而摘要把总量写成 approximately 25T。两者相差约 1.216T，可能来自阶段边界简写或总量四舍五入；没有 step-level log 时保留为 `ambiguous`，不把差额擅自分配给 Stage 3。

## Assumption ledger

| 项 | 本笔记的处理 |
|---|---|
| 研究对象 | text base、multimodal adaptation、SFT、RL、agent RL 与 distillation 分开；MiMo-V2.5 和 Pro 分开 |
| token 单位 | 报告中的 “trained on” 记为 sampled exposure；unique、non-padding、attention-mask 后 loss tokens 均 `unknown` |
| 多模态 token | MiMo-VL 的 2.4T 没有公开 text/image/video token-equivalence 细则，不与 text-only 25T 直接比较 |
| stage 归属 | MiMo-VL 把 long-context SFT 列入四阶段“pre-training”；本仓库仍按训练目标把它标为 `A1` 边界项 |
| mixture 单位 | 70%、10%、5%按报告描述视为 token-mixture claim；没有 source manifest 验证 sampled 与 loss 占比 |
| lineage | V2-Flash 说 pipeline largely follows MiMo-7B；只继承已明确重述的关系，不默认继承阈值、语言比例或 cutoff |
| cutoff | 全家族训练数据时间边界均 `unknown`；GitHub issue/task 的 snapshot 日期也未完整披露 |
| 省略效应 | 代际同时改变参数规模、dense/MoE、attention、MTP、precision、数据、context、optimizer 与 post-training |

## 统一数据字段表

| 字段 | MiMo-7B | MiMo-VL | V2-Flash / V2.5 | 置信 |
|---|---|---|---|---|
| source | web、papers、books、code、synthetic | text、image、video、GUI/reasoning；逐数据集表需报告复核 | public web、books、papers、code、math/STEM；repo code、PR、issues、commit histories；V2.5 multimodal sources未列全 | 类别 `verified`；清单/比例 `open` |
| rights | source-level license/consent `unknown` | weights/repo license 与语料 rights 分离 | public GitHub 不等于统一可训练许可；source rights `unknown` | `open` |
| cutoff | `unknown` | `unknown` | `unknown` | `open` |
| token scale | 约 25T sampled exposure | 四阶段 2.4T，modality accounting `unknown` | V2-Flash 27T；V2.5 48T；V2.5-Pro 27T | 官方数字 `verified`；可比性 `open` |
| parse/extract | math/code/forum-aware HTML extractor；增强 STEM/code PDF parsing | native-resolution visual input；OCR/document pipeline 细节未完整展开 | V2-Flash 沿用 7B family 且偏 long documents；具体 parser error metrics `unknown` | 机制 `verified`；误差 `open` |
| dedup | URL + cross-dump MinHash global dedup | `unknown` | 报告只说 largely follows 7B；V2.5 `unknown` | 7B `verified`；后代 `open` |
| quality | small LLM domain classifier + multi-dimensional tagger；去重后按质量重配 | query selection、large-reasoning-model regeneration、rejection sampling | high-quality sources；SFT/agent tasks有 verifier；阈值、校准与误杀率 `unknown` | 高层机制 `verified`；实现 `open` |
| mixture | S1 general remix；S2 math/code ~70%；S3 synthetic ~10% | 四阶段分工；阶段比例 `unknown` | V2 S2 synthetic ~5%并提升 code；S3提升 long dependency；V2.5 `unknown` | 已披露比例 `verified`；完整 distribution `open` |
| synthetic | STEM source analysis、math/code solution、general/creative queries | long-CoT regeneration + rejection sampling | reasoning responses、synthetic prompts/environments、teacher logits；V2.5 multimodal/agent synthetic 数量 `unknown` | 关系 `verified`；provenance `open` |
| decontam | 无完整 protocol；使用 LiveCodeBench 不等于训练数据已全面去污染 | `unknown` | `unknown` | `open` |
| long context | 8K→32K；与加入 10% synthetic 同阶段 | final long-context SFT；length/source mix `unknown` | V2 32K→256K，长文与 repo history 上采样；V2.5 post-training到1M | 阶段 `verified`；data-only effect `open` |
| validation | broad base suite、NIAH；无完整 held-out corpus contract | 50+ task eval code；混合 RL 存在 domain interference | base/long-context suite、NIAH、GSM-Infinite；agent env/reward trace；V2.5主要是模型卡 benchmark | 评测存在 `verified`；数据归因 `open` |
| artifacts | weights、report、inference code；无 corpus/manifest/order/logs | weights、report、完整 eval prompts/framework | weights、reports、MTP weights、inference code；无 corpus/processing code/order/logs | `D2/D3`，V2.5 data fields `D1` |

## MiMo-7B：先保护 reasoning patterns，再决定混合权重

### extraction 不是中性的前置步骤

报告指出通用网页抽取器会丢失公式、代码块与论坛结构，因此专门优化 HTML extractor，并增强 STEM/code PDF parsing。这意味着：

```text
raw page
  -> generic extraction -> prose-heavy retained text
  -> reasoning-aware extraction -> equations + code + forum structure retained
```

`reasoning density` 在过滤之前已经被 parser 改变。只比较 filter retention 而不固定 extractor 版本，会把解析召回变化误归因为质量模型。

### global dedup 后的质量补偿

MiMo 使用 URL 去重和跨所有 web dumps 的 MinHash 去重。报告同时指出内容无知的 dedup 会平等处理高低质量近重复，因此随后依据多维质量分数调整最终分布。其关系是：

```text
global dedup --removes redundancy without value judgment--> retained pool
quality/domain tagging --reweights--> sampled distribution
```

这不是“质量分数修复了被删文档”的 exact identity：已删除的 instance 不能由后续重权重恢复。它只能补偿 retained clusters 的 sampling distribution；若高质量副本恰被删而低质量副本保留，仍需 representative selection 或 cluster-aware dedup 才能解决。

### heuristic filter 的误杀与 model tagger

团队观察到常用 heuristic filters 会错误删除含大量 math/code 的高质量页面，改用 fine-tuned small LLM 做 domain classification 和 multi-dimensional assessment。这个结果说明单一 web-quality score 与 reasoning value 不是同一个目标，但未公开：label rubric、annotator agreement、各语言 calibration、threshold、false-positive rate、tagger checkpoint 与 inference version。因此只标 `supported`，不能外推成“LLM filter 普遍优于规则”。

### 三阶段 mixture

```text
S1: natural/general sources; no reasoning-query synthetic responses; domain remix; 8K
 -> S2: math + code ~70%; 8K
 -> S3: synthetic reasoning/creative ~10%; context 32K
```

S3 同时改变 synthetic ratio、context length、batch size、RoPE 与学习率。它支持“联合配方有效”的经验关联，不能识别 synthetic 或 long context 的独立贡献。

## 25T 的 operational accounting

MiMo-7B 的训练日程给出一组可检查的 exposure 边界：

$$
T_{S1}=84\text{B}+10.2\text{T}+7.5\text{T}\approx17.784\text{T}
$$

$$
T_{S2}=4\text{T},\qquad
T_{S3}=1.5\text{T}+0.5\text{T}=2\text{T}
$$

报告把 Stage 3 描述为 1.5T fixed-LR 后 0.5T decay，所以阶段表使用约 18T/4T/2T。由这些段落相加得到约 23.784T，与 approximately 25T 的总数不完全闭合；实现审计应从 step × global batch × sequence length 重建：

$$
T_{sampled}=\sum_s N_{steps,s}\,B_{seq,s}\,L_s
$$

并另报：

$$
T_{loss}=\sum_s\sum_{i\in batch_s}\sum_t
\mathbf{1}[attention_{i,t}=1]\mathbf{1}[lossmask_{i,t}=1]
$$

官方报告没有提供 packing、padding、document boundary 与 loss-mask 统计，所以 25T 不能自动写成 25T effective loss tokens。

## Synthetic reasoning：生成路径与“多 epoch 不过拟合”

MiMo-7B 有三种生成路径：

- 从高 reasoning-depth STEM content 生成分析与深入思考；
- 对 math/code problems 生成解答；
- 对 general queries，尤其 creative writing，生成响应。

报告的 preliminary experiment 称 synthetic reasoning data 可训练很多 epochs 而未见普通 synthetic data 的过拟合风险。这个陈述的对象、epoch 定义、unique prompts、teacher diversity、validation split 和 memorization metric 未完整披露，故记为 `[来源事实 | plausible]`，不当作普遍规律。区分性检查应固定 unique prompts 与 sampled tokens，比较：重复同一响应、同题多样响应、不同 teacher、自然数据重复，并同时测 held-out loss、exact/fuzzy memorization 与 per-domain capability。

## RL data：130K 是 problems，不是 trajectories

数学数据先用解答验证、答案一致性与 16-rollout difficulty filtering，约删除 50% easy problems，得到 100K；代码数据要求 test cases，并检查 golden solution 或 16 rollouts，得到 30K。训练过程中每个 prompt 会产生多个 on-policy rollouts，因此：

$$
N_{problems}=130\text{K}\neq N_{trajectories}\neq T_{sampled}\neq T_{loss}
$$

动态采样会移除全对问题，但直接永久删除 easy problems 导致更新不稳定。MiMo 保留 easy pool，以 10% 概率重新采样。这是一个 training-state-dependent scheduler：样本权重取决于当前 policy pass rate，而不是静态 manifest。

代码 reward 按 test-case pass rate 估计难度并做 hard/soft group scoring，扩大 sparse reward 的分辨率。它验证 test execution，不验证题目授权、测试完备性、安全性或与 benchmark 的重合。

## MiMo-VL：2.4T 不是可与文本 token 直接相加的量

MiMo-VL 依次进行 projector warmup、vision-language alignment、general multimodal pretraining 与 long-context SFT，总计 2.4T tokens。报告强调从多样 queries 出发，让 large reasoning models 重写 long-CoT responses，再 rejection sample，并在后期 pretraining 中大量使用，而非只放进最后 SFT。

这里至少有三种统计单位待补：text tokenizer tokens、image/video visual tokens、GUI action/coordinate tokens。没有 modality-specific accounting 时，2.4T 只能作为该报告内部 exposure claim。MORL 又把 perception、grounding、reasoning 与 preference rewards 混合；报告承认 domain interference，因此总平均提升不能证明每个域都单调改善。公开的 50+ task eval framework 和 prompts 能复核推理协议，但不能复核 training mixture。

## V2-Flash：从 27T schedule 到 agent environment factory

### pretraining schedule

V2-Flash 的 source 类别包括 web、books、academic papers、code、math 和 broader STEM。相对 7B，它明确向 long-range dependencies 偏移：长网页、repository-level code、PR、issues 与 commit histories。

```text
0–22T general, 32K
 -> 22–26T upsample code + ~5% synthetic reasoning, 32K
 -> 26–27T same family mix + upsample long dependencies, 256K
```

Stage 3 同时改变数据、sequence length、RoPE、batch size、learning rate 和 expert-bias update。因此 NIAH/GSM-Infinite 的长上下文提升属于联合干预。

### SFT 与 MoE-specific monitor

SFT 包含 millions of general、reasoning、coding 和 agent samples，并混合 thinking/non-thinking responses。报告用每个 MoE layer 中未分到 token 的 experts 数 `num-zeros` 监控负载：上升被解释为 load imbalance，下降被解释为 overfitting。这个指标是 implementation diagnostic，不是数据质量分数；它依赖 batch、router、capacity 与并行布局。

### agent task accounting

报告的表格给出 90K real-environment/real-prompt code tasks、30K real-environment/synthetic-prompt code tasks、150K real-environment/synthetic-prompt search tasks、50K synthetic/synthetic general-agent tasks。另称 code-agent pipeline 覆盖 100K+ code tasks、8 languages，环境自动构建成功率 70%，集群可并发 10K+ pods。

这些数字必须分开：

```text
task catalog size
  != successfully built environments
  != active RL environments per update
  != rollouts
  != tool calls
```

约 120K environments 的 on-policy scaling trace 与 90K+30K code-task table 可能是同一 code-agent slice 的不同训练时刻/口径；没有固定 snapshot manifest，关系保留为 `ambiguous`。

### MOPD 的 provenance graph

```text
general SFT student
  -> domain-specific RL on math/code/search/tools/safety
  -> specialized teacher checkpoints
student on-policy trajectories
  -> teacher logits / token-level KL rewards
  -> unified student
```

MOPD 不是离线 teacher-response dataset。student 自己生成轨迹，teacher 对这些轨迹提供 dense token-level supervision；因此 provenance 至少要记录 prompt set、student checkpoint、sampling config、teacher/version、domain router、logits/reward transform 与 rollout seed。报告表中部分领域超过 best teacher，也有 GPQA、creative writing、SWE-Bench、BrowseComp 退化，正好说明 capability integration 不是无损 exact merge。

## V2.5 与 V2.5-Pro：同代并列，不把数字强行对齐

截至核查日，MiMo-V2.5 card 披露 310B/15B active、约 48T、text→projector warmup→multimodal pretraining，并在 post-training 将 context 从 32K 逐级扩到 1M。MiMo-V2.5-Pro card 则披露 1.02T/42B active、27T、native 32K 与 1M context。

`48T > 27T` 不说明小模型看了“更多 unique data”，也不说明 Pro 做了回退。两张 card 没给统一 token accounting、modality composition、repeat rate、stage schedule 或 shared-corpus overlap。可检查的最小表示是：

| checkpoint | total/active params | reported exposure | modality | native/base/max context |
|---|---:|---:|---|---|
| V2.5 | 310B/15B | ~48T | text+image+video+audio | 未完整披露 / 256K / 1M |
| V2.5-Pro | 1.02T/42B | 27T | card以语言模型描述 | 32K / 256K / 1M |

只有固定模型 card revision 和训练报告后，才可以继续问它们是否共享 V2-Flash 27T corpus、48T 是否包含 visual/audio tokenization、以及 Pro 是否从独立初始化训练。

## Validation、ablation 与区分性检查

### 已公开的锚点

- `[来源事实 | verified]` MiMo-7B 报告 broad base benchmarks、reasoning/code/math、Chinese 与 NIAH，并给出 pretraining stage、LR 与 context schedule。
- `[来源事实 | supported]` MiMo-7B 指出 heuristic filter 对 math/code 页面误杀，并以 reasoning-aware extraction + model tagger替换；缺 false-positive 对照表。
- `[来源事实 | verified]` RL 数据构造给出 100K/30K retention、16-rollout filter、10% easy resampling 与 code test-difficulty reward。
- `[来源事实 | verified]` V2-Flash 给出 22T/4T/1T scheduler、base/long-context per-length evaluation、agent-task composition 与 MOPD domain table。
- `[来源事实 | verified]` MiMo-VL 公开 checkpoints 和 50+ task evaluation framework/prompts；训练 corpus/manifest 未公开。

### 主要混淆与建议实验

| 公开结论 | 首个混淆 | 有辨识力的检查 |
|---|---|---|
| reasoning-aware extraction 提高能力 | extractor 同时改变保留长度/domain/token count | 固定 raw URLs 与 sampled token budget，仅切换 parser；按公式/code/forum/prose 报 recall、noise、held-out loss |
| model tagger 优于 heuristics | label rubric、threshold 与 domain mix 一起变化 | 同一候选池、相同 retention；blind human audit + per-language/domain false positive；固定模型训练 |
| S2/S3 mixture 提高 reasoning | LR、batch、context、RoPE、synthetic 同时变化 | checkpoint fork，分别改变 mix、context 与两者；至少 3 seeds，micro/macro/per-domain 报告 |
| synthetic 可多 epoch 不饱和 | prompt/teacher diversity 与重复曝光未知 | 固定 unique tokens，比较 exact repeat、multi-response、multi-teacher；测 memorization 与 held-out capability |
| long-dependency data 支持 256K | attention architecture/context extension 同时变化 | 2×2：short/long data × no-extension/extension；按真实长文、拼接、repo history 分层 |
| MOPD 保留 specialized skills | teacher 与 student queries/compute 不同 | same student checkpoint、same prompts/rollout budget，对照 sequential RL、offline distill、MOPD；报每域回退 |

### validation contract 的缺口

仍缺少 pretraining held-out corpus 的 source、时间边界、dedup unit、freeze date、tokenizer revision、每域 token 数、micro/macro loss、decision threshold 和 checkpoint-selection rule。benchmark suite 不能替代这一 contract；LiveCodeBench 的 contamination-aware 设计也不能证明其他 benchmark 或训练阶段已去污染。

## 表面冲突与边界

### 25T 与 Stage 1/2/3 数字

约 25T 是总 sampled exposure；约 17.784T/4T/2T 是由学习率段落直接读出的阶段 exposure，不是 unique source pool。两组数字不能精确闭合，差额约 1.216T；在缺少 step log 时保留为 `ambiguous`，不能反推额外训练阶段。

### 2.4T multimodal 与 25T text

二者 tokenizer 与 modality accounting 不同。MiMo-VL 又从 MiMo-7B backbone 出发，2.4T 是额外多模态四阶段 exposure，不是完整“从零训练总 token”或可直接加总后的统一单位。

### 27T V2-Flash 与 27T V2.5-Pro

数字相同不证明 corpus、顺序或重复率相同。Pro card 只给高层 total/native-context 描述；需训练报告或 manifest 才能建立 exact lineage。

### V2.5 48T 与 Pro 27T

它们是不同参数规模/模态 checkpoint。应做 model-specific accounting，不能用“大模型通常看更多/更少数据”的先验改写官方字段。

## 可迁移与不可外推

### 可迁移

- `[综合判断 | supported]` extraction recall 应按高价值结构单独评估；公式、code block、forum tree 与长文档结构不能只看纯文本字符保留率。
- `[综合判断 | supported]` content-blind dedup 与 quality-aware sampling 是两个不同算子；需要 cluster representative 与 post-dedup distribution 两套指标。
- `[综合判断 | supported]` reasoning pretraining 应同时记录 natural/synthetic provenance、teacher、verification、repeat exposure 与 context bucket。
- `[综合判断 | supported]` RL task catalog、environment build success、rollouts、trajectories、tool calls 和 loss tokens必须分开计量。
- `[综合判断 | supported]` on-policy distillation 需要版本化 student-generated trajectories 与 teacher logits，不能只保留原始 prompts。

### 不可外推

- `[未知 | open]` 70% math/code、10%或5% synthetic 不是跨模型最优配比。
- `[未知 | open]` 不能从 MiMo benchmark 表现反推出未披露 source、language ratio、rights、cutoff 或 contamination rate。
- `[未知 | open]` 不能由 70% environment-build success 推断 30% 失败任务的分布无偏；失败很可能与语言、依赖复杂度和年代相关。
- `[未知 | open]` 不能将 V2.5 的 48T、多模态 encoder 与 1M context 的收益分解为 data-only effect。

## 明确未知项

- 完整 source manifest、language/domain/modality token ratio、source rights/consent 与全局 cutoff；
- tokenizer revision、visual/audio token accounting、unique/sample/loss token 对照；
- parser benchmark、dedup thresholds/cluster representative policy、tagger labels/calibration/false positives；
- 各阶段 sample order、random seed、repeat exposure、packing、document boundary 与 loss masks；
- benchmark decontamination targets、matcher、threshold、clean-subset sizes 与 temporal validation；
- synthetic prompt/teacher/version/temperature/rejection manifest 和 memorization audit；
- SFT/RL/MOPD 的 prompt、trajectory、tool-call、accepted-token 与 loss-token 数；
- V2.5 与 Pro 对 V2-Flash corpus/checkpoint 的精确继承关系。

## 掌握标准与推理型自测

完成本案例后应能：

1. 解释为什么 reasoning-aware parser 是数据配方的一部分，而不是无影响的工程细节；
2. 从 25T/27T schedule 区分总 exposure、stage exposure、unique corpus 与 loss tokens；
3. 说明 MiMo-VL 2.4T 为什么不能直接与 text-only 25T 比质量或算力；
4. 把 130K RL problems、120K environments、10K pods 与 trajectories 正确放入不同统计层；
5. 为 synthetic mixture × context extension 设计能分离主效应的实验；
6. 说明 MOPD 与 offline response distillation 的 provenance 差异。

自测：

- 如果新 parser 使 math tokens 增加 30%，validation math loss 下降，如何排除只是更长文档或 contamination 增加？
- 若 Stage 3 只有 1T long-context tokens，如何报告 256K sequences 中真实长文、repo history 与拼接样本的比例？
- environment build 失败率从 30% 降到 10%，应检查 RL capability 变化还是 sampling-population 变化？
- V2.5 与 Pro 都支持 1M，怎样区分 native training、post-training curriculum 与 inference extrapolation？

## 来源与阅读顺序

1. [MiMo: Unlocking the Reasoning Potential of Language Model — From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608)：先读 §2.1 的 extraction/dedup/filter/synthetic/mixture，再读 §2.3 的 token/LR/context 边界，最后读 §3.2–3.3 的 130K 构造与 easy resampling。
2. [MiMo 官方仓库](https://github.com/XiaomiMiMo/MiMo)：用于固定发布 checkpoint、初版/0530 SFT 规模与 RL context 更新；不要把 README 更新反写成初版报告配方。
3. [MiMo-VL Technical Report](https://arxiv.org/abs/2506.03569)：读四阶段 2.4T、synthetic long-CoT/rejection sampling 与 MORL；特别区分 multimodal pretraining 和 long-context SFT。
4. [MiMo-VL 官方仓库与 evaluation framework](https://github.com/XiaomiMiMo/MiMo-VL)：用于固定 checkpoints、2508 diff 与公开 eval prompts；它不是训练数据发布。
5. [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)：重点读 §3 的 22T/4T/1T schedule、§4 的 MOPD、agent task table、environment pipeline 与 per-domain gains/regressions。
6. [MiMo-V2-Flash 官方仓库](https://github.com/XiaomiMiMo/MiMo-V2-Flash)：用于固定模型卡、MTP weights 和部署资产。
7. [MiMo-V2.5 model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.5)：读取 310B/15B、48T、四个训练阶段与 1M context；由于没有同等技术报告，数据字段只记 `D1`。
8. [MiMo-V2.5-Pro model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro)：读取 1.02T/42B、27T/native 32K、SFT→specialized RL→MOPD；与 V2.5 分行记录。
