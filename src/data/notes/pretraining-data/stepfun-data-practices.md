---
title: "StepFun Step 家族的数据实践"
description: "区分 annealing corpus、多模态 task/token、replay 和 scaling-law proxy 到目标模型的迁移边界。"
topic: "pretraining-data"
section: "china-cases"
slug: "stepfun-data-practices"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 106
readtime: 31
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/stepfun.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/china/stepfun.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:6c8079752cdc864e603f8b758c8f1ae9e6c4fc9a20c11b510582722bc17e19a5"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`method-family/historical trace` -->
> 主要模型/资料：Step 3.5 Flash、STEP3-VL-10B、Predictable Scale / Step Law；Step-1/2/3/3.7 只作披露边界。

## 这篇案例要回答什么

StepFun 的早期 Step-1/2 产品材料很难支持精细的数据审计；Step 3.5 Flash 报告则公开了从 14.6T open-domain 到 3T annealing、750B mid-training，再到 871K/7.23B SFT 和 agent RL 的完整链路。STEP3-VL-10B 又提供 1.2T multimodal tokens 的单阶段全参数训练和 900B→300B quality cooldown，对比价值很高。

这个案例用于回答：

1. “annealing corpus”具体改变了 source、质量层级、context、mask 与 learning rate 中的哪些量；
2. code、PR/issue/commit、tool use 与 long documents 怎样从 pretraining 延续到 mid-training/post-training；
3. 1.2T multimodal tokens 怎样拆成视觉任务家族、两段 quality schedule 与后续 SFT/RL；
4. 3,700 个 proxy models / 约 100T aggregate training exposure 的 Step Law 能预测什么，不能替代什么。

## 各代模型和 training stage

| 模型/研究 | 阶段 | 已披露数据规模 | mixture / context | 披露 |
|---|---|---:|---|---|
| Step-1/2/3 | `P0/P1/A*` | token、source mixture 与 stage schedule `unknown` | 官方产品/系统材料给模型与推理信息，不能还原训练数据 | `D0-D1` |
| Step 3.5 Flash | `P0` Stage 1 | 14.6T | broad open-domain；4K | `D2/D3` |
| Step 3.5 Flash | `P2/P3` Stage 2 | 3T = 2T@4K + 1T@32K | 高质量 knowledge、code、PR/issue/commit、reasoning-dense；secondary LR decay | `D2/D3` |
| Step 3.5 Flash | `P1/P3` mid stage 1 | 386B@32K，其中 81B replay（21%） | software-engineering、tool use；pretraining replay | `D2/D3` |
| Step 3.5 Flash | `P1/P3` mid stage 2 | 364B@128K，其中 10.5B replay | synthetic long-horizon reasoning、自然长文、code/search/tool agent | `D2/D3` |
| Step 3.5 Flash | `A1` SFT stage 1 | 870,687 samples、7.23B tokens | 9 domains；最大 128K；multi-domain | `D2/D3` |
| Step 3.5 Flash | `A1` expert self-distillation | 样本/token 数 `unknown` | math/code/STEM/general/agent experts → unified student | `D2/D3` |
| Step 3.5 Flash | `A2/A3` RL | prompt/rollout/loss tokens `unknown` | math/code/STEM/general、code/search/tool agent；offline actor 与异步 agent | `D2/D3` |
| STEP3-VL-10B | multimodal `P0/P2` | 1.2T = 900B broad + 300B higher-quality cooldown | fully unfrozen、single-stage architecture；4K | `D2/D3` |
| STEP3-VL-10B | `A1` SFT | 190B text-dominant（9:1）+36B balanced（1:1） | 128K；text:multimodal ratio | `D2/D3` |
| STEP3-VL-10B | `A2` RL | 600 + 300 + 500 = 1,400 iterations | RLVR→RLHF→parallel coordinated reasoning；24K→32K→64K | `D2/D3` |
| Step Law | proxy experiments | 3,700 models、约 100T aggregate training tokens | 400M@40B、1B@100B 等网格；多种 data distributions | `D3/D4` 部分资产 |
| Step 3.7 Flash | `P*/A*` | 完整 data recipe diff `unknown` | 官方开放权重/推理资产；不继承 3.5 的定量配方 | `D1/D3` |

Step 3.5 的阶段算术闭合为：

$$
T_{pre}=14.6\text{T}+2\text{T}+1\text{T}=17.6\text{T}
$$

$$
T_{mid}=386\text{B}+364\text{B}=750\text{B}
$$

其中 replay 已包含在各 mid-training stage 总数中，不能再加一次：

$$
T_{new,mid}=750-(81+10.5)=658.5\text{B}
$$

这是 sampled-exposure accounting；自然长文、90B PR-dialogue、12B rewrite 等 corpus slice 可能被重复采样，不能与阶段总数直接相加。

## 阅读这些结论前先确认的前提

| 项 | 本笔记的处理 |
|---|---|
| token 单位 | T/B training 数先记 sampled exposure；90B/12B 等若报告称 dataset tokens 则记 corpus tokens |
| multimodal token | 1.2T 使用报告内部 tokenizer/visual-token accounting；没有 text/image token-equivalence 表，不与 text-only 17.6T 比效率 |
| annealing | 同时改变 mixture、质量、context、LR、batch、MTP weight、router bias 与 RoPE；不是纯 data intervention |
| mid-training | replay 是 stage 内子集；natural long docs 按 >32K 从 pretraining pool 选出 |
| Step Law | 预测 optimal LR/batch 的局部经验规律，不是 corpus-quality scaling law |
| model lineage | 3.5、VL-10B 与 3.7 架构/模态不同；不默认共享 corpus 或 tokenizer |
| cutoff | corpus snapshot 与统一 cutoff 未公开；RL 明确排除 2024–2026 competitions 是局部去污染，不是全局 cutoff |
| rights | robots.txt、licensed/open-source 等 source 声明逐项记录，不据此推断所有下游使用权 |

## 厂商公开了哪些 data fields

| 字段 | Step 3.5 Flash | STEP3-VL-10B | 置信 |
|---|---|---|---|
| source | StepCrawl/Common Crawl、HTML/ePub/PDF、code、GitHub PR/issues/commits、math/STEM、synthetic/semi-synthetic reasoning/tool/agent | text；CC+StepCrawl interleaved、LAION/COYO 等 pairs、education/OCR/grounding/VQA/GUI、synthetic | 类别 `verified`；完整 manifest `open` |
| rights | StepCrawl 声称遵守 robots/site policy；GitHub/open/用户数据的逐样本 rights `unknown` | 部分 licensed exam materials + open datasets；commercial search retrieval 与全部 image rights `unknown` | `open` |
| cutoff | `unknown`；RL 排除 2024–2026 competition problems | `unknown` | `open` |
| token scale | 17.6T pre +750B mid；SFT 7.23B | pre 1.2T；SFT 190B+36B；RL rollout totals `unknown` | headline `verified`；loss tokens `open` |
| quality | six-scorer ensemble tiers；domain filters；knowledge-density；code hit0–6 | alignment/aesthetic、task-specific filters、higher-quality 300B mix；RL all-agree/visual relevance/difficulty | 机制 `verified`；校准 `open` |
| dedup | MinHash infrastructure；100K+ embedding clusters rebalanced；benchmark-specific code dedup | concept-balanced CLIP clusters；SFT exact+64-gram decontam；完整 pretrain dedup `unknown` | 部分 `verified` |
| mixture | Stage 1 broad→anneal code/high knowledge→mid agent/long；精确 source ratios未全给 | 900B broad→300B higher quality；SFT 9:1→1:1 | scheduler `verified`；完整比例 `open` |
| synthetic | knowledge rephrase/QA、90B dialog、12B rewrite、tool FSM、long tasks、agent env | OCR/infographic/geometry/caption/QA、teacher SFT、parallel-message cache | 类型 `verified`；source lineage `open` |
| decontam | PR data vs SWE benchmarks；SFT exact/digit-mask/N-gram；RL排除近年赛事 | SFT exact +64-gram；RL prompt filters | 局部 `verified`；全局 `open` |
| validation | 30B-A3B fixed-budget data proxy、held-out compression、full benchmark；stage base checkpoints | matched-token pretrain ablations、multimodal/reasoning eval、RL curves | protocol存在 `verified`；target transfer `open` |
| artifacts | base、midtrain、instruct weights；SteptronOss；报告；SFT data release承诺/版本需固定 | base/instruct weights、report、repo | `D2/D3`；Step Law部分 `D4` |

## Step 3.5：StepCrawl、质量分层与 cluster rebalancing

StepCrawl 不只是一个 URL 集合，而是带 LM-in-the-loop 的 crawl scheduler：WebOrganizer-style model 过滤 SEO/低效页面，并按 site category 分配 crawl budget，防止 tool/e-commerce 等类别压倒其他主题。报告称系统每天处理约 1B pages，并遵守 robots.txt 与 site-specific access policy。

其数据链可表示为：

```text
site/URL candidates
 -> category/utility model
 -> crawl-budget allocation
 -> HTML/ePub/PDF extraction
 -> six-scorer quality tiers
 -> dedup + sanitization
 -> embedding clusters (100K+)
 -> mass rebalancing / domain downsampling
 -> sampled mixture
```

web 数据保留 High、Medium-High、Medium，丢弃 Medium-Low/Low；books/papers 在 annealing 只保留 High/Medium-High。另用 high-value entity/concept/relation inventory 在 embedding space 挖 knowledge-dense passages，再对一部分 controlled rephrase 与 QA synthesis。

100K+ cluster rebalancing 对超大 cluster 降采样，支持“减少分布偏斜”的机制判断，但不是 dedup 的 exact synonym：cluster 可以含同主题不同事实，也可能把近重复拆散。需分别报告 cluster mass、duplicate rate、topic coverage、quality retention 与 held-out compression。

## Code 与软件工程数据：质量—多样性不是单调阈值

pure-code 采用修改后的 OpenCoder rules，以 heuristic violation 数分 hit0、hit1…；内部 30B-A3B proxy 显示 hit0-only 过度裁剪、无过滤噪声过多，hit0–6 整体最好。这是 local optimum，不表示“六次违规”具跨 parser/语言的稳定含义。

GitHub 选择 >10 stars repositories，形成约 5M sample foundation，并派生：

- base PR/issue/commit：GHArchive/API、commit history、20+ languages，部分以 `git diff` 验证；
- 90B PR-dialogue：file localization 与 SEARCH/REPLACE code repair；
- 12B Python reasoning rewrite：problem analysis/root cause/design/implementation 与 active-reading notes；
- hundreds of thousands environment seed samples：problem、change、tests，需可复现 environment。

90B dialog 在 pretraining annealing 只 mask template scaffolding；在 mid-training 转成 chat，mask user prompts。同一 corpus 因 serialization/loss mask 不同，产生的监督目标不同：

$$
T_{context}=T_{payload}+T_{scaffold}+T_{prompt},\qquad
T_{loss}=T_{context}-T_{masked}
$$

报告没有给两阶段的 masked-token totals，所以不能从 90B corpus tokens 推出 effective loss exposure。

## Annealing 与 mid-training：联合调度而非单一 cooldown corpus

```text
14.6T broad @4K
 -> 2T annealed @4K
 -> 1T same annealed family @32K
 -> 386B software/tool @32K (81B replay)
 -> 364B long/agent @128K (10.5B replay)
```

annealing 提升 high-quality knowledge、reasoning-dense、code 与 PR/issue/commit，同时 secondary cosine decay；32K 段又改变 batch、MTP loss、router-bias schedule 与 RoPE。mid-training 冻结 router、关闭 balance loss，并把自然 >32K 文档与 synthetic long-horizon tasks、code/search/tool-agent data 混合。

因此“annealing corpus 有效”最多支持 tested joint recipe。区分数据效应需要固定 optimizer/context，重放 broad 与 annealed mixture；区分 context 效应需在相同 document/packing 下比较 4K/32K；区分 replay 作用需固定 new-data exposure 比较 0%、21%、matched replay。

## Metadata token：context 与 loss accounting 的一个少见公开例子

每个样本前添加 human-readable metadata：content type、language、domain、source。前约 3.8T tokens 对 metadata 与 payload 都计算 loss，之后 metadata 保留在 context 但被 mask，只预测 payload：

$$
\mathcal{L}_{mask}=-\sum_{t=1}^{|x|}\log P_\theta(x_t\mid M,x_{<t})
$$

这里 $M$ 是 metadata，$x$ 是 payload。这是 `implementation relation`：metadata 仍改变条件分布和 attention compute，只是不再直接贡献 target tokens。报告将“此时模型已学会使用 metadata”称为 hypothesis；没有 metadata-remove/mask-timing factorial ablation 时，置信标 `plausible`。

## 30B-A3B proxy 与 Step Law：两种不同的“小模型实验”

### data proxy

data ablation 使用 30B total / 3B active MoE、Muon、固定 token budget，比较 quality tier、cluster rebalance、code hit threshold、knowledge augmentation 等，并结合 benchmark 与 held-out compression/perplexity。团队称更小模型会低估 long-tail memory、偏好 repetition，所以选 30B-A3B 以提高 full-scale trend fidelity。

这个判断本身仍需 target-scale rank correlation。更稳妥的 protocol 是：对每个 data candidate 在多个 proxy size/seed 上估计 effect 与 uncertainty，再在少量 target checkpoint 验证排序，而不是只选一个“足够大”的 proxy。

### Step Law

Predictable Scale Part I 训练 3,700 models，总 exposure 约 100T，部分网格如 400M@40B 与 1B@100B；每个位置可含 45–120 个 hyperparameter runs。给出的经验式是：

$$
\eta(N,D)=1.79N^{-0.713}D^{0.307},\qquad
B(D)=0.58D^{0.571}
$$

$N$ 为 non-vocabulary model parameter convention，$D$ 为 training tokens，$\eta$ 为 optimal LR，$B$ 为 token-wise batch size；单位与适用范围必须沿论文配置。报告在 English-dominant、Chinese-English、Code-English、Code-dominant distributions 测试稳定性。

Step Law 预测的是给定 training setup 的 optimizer hyperparameters，不是 data mixture 的最优配比，也不证明同一公式能跨 tokenizer、optimizer、extreme $D/N$ 或未来 architecture 无条件外推。约 100T 是 3,700 runs 的 aggregate exposure，不是某个生产模型的语料量。

## STEP3-VL-10B：1.2T multimodal tokens 的单阶段融合

### source/task family

模型把 language-aligned 1.8B perception encoder、Qwen3-8B decoder 与 projector 全部解冻，在单一 pretraining stage 中联合训练。数据包括：

- knowledge：CC/StepCrawl interleaved、LAION/COYO/BLIP-CCS/Zero pairs、keyword retrieval、mosaic；
- education：约 15M，K-12、university、adult exams，含 licensed/open/synthetic；
- OCR：10M real +30M synthetic image-to-text、10M+ open +15M synthetic image-to-code、80M pages、约 4M tables/100M formulas；
- grounding/counting：约 400M；VQA 约 10M，OCR-VQA 约 20M；
- GUI：约 23M，其中 700K captions、1M+ QA、2M+ trajectories、19M+ grounding；另 crawl 约 30M webpages 做 element coordinates。

这些是 sample/task counts，不与 1.2T tokens 相加。一个 page、image pair、trajectory 的 visual/text token 数不同，且 mosaic/multi-crop 会改变 exposure。

### 900B + 300B quality schedule

全程 4K、global batch 8,192、370K iterations：前 900B broad representation，LR 5e-5→1e-5；后 300B 转向 higher-quality mixture，LR 1e-5→6e-6，强调 OCR、grounding 与 reasoning。报告称 single-stage fully unfrozen，指组件不先后冻结；它仍有两段 mixture/LR schedule，不应误写成“数据分布恒定”。

### SFT 与 1,400 RL iterations

SFT 最大 128K，Stage 1 为 190B、text:multimodal=9:1；Stage 2 为 36B、1:1。精确 token share 与 sample ratio 是否完全相同需 loader stats 才能确认。SFT 用 internal frontier teacher responses，并以 exact +64-gram 做 benchmark decontamination。

RL 分三段：

1. RLVR：600 iterations，512 prompts ×16 rollouts，max 24K；all-agree verification、visual relevance、24-rollout some-accept difficulty；
2. RLHF：300 iterations，512×8，max 32K；不可验证 prompts + preference/constraint rewards；
3. parallel coordinated reasoning：500 iterations，64×16，max 64K；复用 difficulty 阶段 24 rollouts 作为 message cache，再做 synthesis filtration。

按固定每 iteration prompt/rollout 设置可算 rollout count：

$$
600\times512\times16+300\times512\times8+500\times64\times16=6{,}660{,}096
$$

这只是 nominal rollouts，不是 unique prompts 或 rollout tokens；early stop、过滤、失败和实际长度均未由该算式恢复。parallel stage 复用 message cache，还需避免把同一 rollout 同时计作 filter artifact 与训练 context 的“独立样本”。

## Step 3.5 post-training：SFT 表与 agent environment

第一阶段 SFT 的 871K/7.23B 分为 math 68,055/0.98B、code 86,421/1.23B、STEM 120,399/0.55B、logic 93,323/0.81B、general 314,495/0.80B、code agent 37,240/0.90B、tool-use 114,507/0.76B、search agent 20,256/0.50B、long context 15,565/0.70B。

样本贡献和 token contribution 不同。比如 long context 只占约 1.8% samples，却占约 9.7% tokens；报告表中的 `Corpus Contribution` 是采样/配方权重概念，不能与 raw token share 混用。

tool-use 采用 FSM 把 abstract intent 与 parameterized constraints 分离，经 sample→execute→verify，在真实环境 deterministic validation，构造 >100K trajectories、billions tokens。code environment pipeline 报告 40% build success，得到 50K verified environments、15K+ GitHub repos、20+ languages。

这些分母必须分开：

```text
candidate records -> attempted builds -> successful environments
 -> tasks -> trajectories -> tool calls -> supervised/rollout tokens
```

40% 不能与 50K 反推出唯一 candidate count，除非二者来自同一固定 snapshot；报告没有明确建立这个 exact relation。

## 如何验证这些结论，并检查 contamination

| 公开结论 | 首个混淆 | 有辨识力的检查 |
|---|---|---|
| six-scorer quality tiers 提高效率 | retention、repetition、domain mix同时变 | 固定 sampled tokens与domain，质量层 × repetition factorial；per-domain compression/benchmark |
| 100K+ cluster rebalance有效 | 去冗余与topic重配混合 | 保存 cluster/quality交叉表；random-downsample、near-dup-only、mass-cap 三组 |
| hit0–6 code最佳 | violation定义依赖语言/parser | 按语言/规则报告 precision、retention、compile/test率；跨proxy/target验证排序 |
| annealing/long-context提高agent能力 | mixture、LR、context、mask、RoPE同时变 | matched-data replay；4K/32K context-only；broad/annealed mixture-only |
| 30B-A3B proxy更可信 | 无完整target rank correlation | 多size/seed effect ranking + target checkpoint confidence interval |
| VL fully-unfrozen 1.2T有效 | encoder、corpus、tokens、schedule联合变化 | freeze/projector-only/unfreeze matched-token；自然/合成与task family分桶 |
| parallel reasoning RL有效 | 更多total context/rollout evidence | 固定total generated tokens、wall time与unique prompts，对比sequential/parallel |

## 看似矛盾的说法怎样区分

1. **17.6T 与 14.6T+3T**：exact closure；750B mid 独立列出，不混为 18.35T “pretraining corpus”。
2. **386B/364B 与 81B/10.5B replay**：replay 是子集 exposure，不能重复相加。
3. **single-stage VL 与 two-phase schedule**：single-stage 描述参数解冻/架构流程；mixture/LR 明确分 900B/300B，不冲突。
4. **1.2T tokens 与数亿 images/tasks**：统计单位不同；没有 modality accounting 不能互换。
5. **Step Law ~100T**：是 3,700 experiments 的总训练量，不是 Step 3.5 生产 corpus。
6. **Step-3.7 是后继版本**：最新不等于数据披露更完整；未公开字段保持 `unknown`。

## 哪些经验可以借鉴，哪些不能直接照搬

### 可迁移

- annealing audit 应同时保存 mixture、quality tier、context、LR、mask、batch、tokenizer 与 replay。
- corpus slice、sample、environment、trajectory、rollout、context token 和 loss token 必须分单位。
- data proxy 需要固定 token budget、held-out compression、多个 seed/size，并验证 target ranking。
- crawl-budget allocation、cluster mass rebalancing 与 source sampling 都属于 mixture pipeline，不只是“过滤”。
- multimodal recipe 要分别报告 task/sample counts、visual/text tokens、natural/synthetic 与 SFT ratio。
- 可执行 agent data 需保存 repo/environment snapshot、build attempt、test、tool trace 与 rejection reason。

### 不可外推

- 不把 hit0–6、quality-tier thresholds、21% replay 或 9:1→1:1 当成通用最优值。
- 不用 Step Law 代替 data scaling、mixture search 或 target-scale data ablation。
- 不把遵守 robots.txt 等同于版权许可或训练授权已解决。
- 不把 VL benchmark/RL 增长直接归因于 1.2T pretraining；teacher、SFT、RL 与 test-time compute均变化。
- 不从 Step-3.7 能力或架构反推其沿用 3.5/VL 数据。

## 目前仍不知道什么

- Step-1/2/3/3.7 的完整 source、token、mixture、cutoff、pipeline 与 validation contract；
- StepCrawl URL/document manifest、snapshot date、retention、language/domain proportions 与 rights ledger；
- 17.6T/750B 的 unique corpus、repeat distribution、packing/padding/mask 后 loss tokens；
- six scorers、knowledge-density model、embedding checkpoint、cluster threshold 与 calibration；
- 90B/12B/code/math slices 在各 stage 的实际 sampled exposure；
- VL 1.2T 的 text/visual token breakdown、image repeat、task share、modality tokenizer accounting；
- agent/RL 的 unique prompts、accepted trajectories、tool calls、actual rollout tokens 和 fixed environment snapshot。

## 读完后应该能回答的问题

应能：

1. 重建 17.6T+750B scheduler，解释 replay 为何不能重复相加；
2. 区分 annealing data、optimization 与 context intervention；
3. 比较 Step 3.5 与 STEP3-VL 的 data units；
4. 说明 Step Law 与 data proxy 的目标差异；
5. 设计可区分 quality、repeat、cluster rebalance、context 与 LR 的实验。

自测：

1. 81B replay 占 386B 的 21%，但若其中每个 unique token 平均出现 2 次，能否推出 unique replay corpus size？还缺什么？
2. 由 90B PR-dialogue 能否算出 annealing loss tokens？为什么 mask 策略使答案不确定？
3. STEP3-VL 的 900B/300B 数据结论怎样与“single-stage fully unfrozen”同时成立？
4. 6,660,096 nominal rollouts 为什么不是 RL sample efficiency 的充分分母？
5. 怎样验证 30B-A3B proxy 的 data ranking 能迁移到 196B-A11B，而不训练所有候选到终点？

## 来源与建议阅读位置

- [Step 3.5 Flash technical report](https://arxiv.org/abs/2602.10604)：主锚点；读 §4 training curriculum、§5 data synthesis、Appendix A.3/C/D/E。
- [Step 3.5 Flash official repository](https://github.com/stepfun-ai/Step-3.5-Flash)：核对权重、模型卡、训练代码入口和版本；数据发布承诺需按 revision 复核。
- [Step 3.5 Flash Base-Midtrain model card](https://huggingface.co/stepfun-ai/Step-3.5-Flash-Base-Midtrain)：核对 base/mid checkpoint 和可复现资产边界。
- [STEP3-VL-10B technical report](https://arxiv.org/abs/2601.09668)：读 §2 data construction/training recipe、§3 SFT/RL 与 matched-token ablation说明。
- [STEP3-VL-10B official repository](https://github.com/stepfun-ai/Step3-VL-10B)：核对 base/instruct weights 与推理实现。
- [Predictable Scale Part I / Step Law](https://step-law.github.io/)：读 3,700-model grid、公式、data-distribution tests 和 release roadmap；不要把 aggregate 100T 当生产语料。
- [Step-3 system report](https://arxiv.org/abs/2507.19427)：用于确认 Step-3 重点为 model-system co-design；不提供等价的训练数据 recipe。
- [Step-3.7 Flash official repository](https://github.com/stepfun-ai/Step-3.7-Flash)：用于固定最新开放 checkpoint 边界；本轮不从其能力反推数据。

## 这篇案例与主线知识的关系

```text
crawl-scheduling --changes--> source distribution
quality-tier + cluster-rebalance --implemented-by--> annealed mixture
annealed-mixture --co-varies-with--> LR + context + mask + RoPE
proxy-experiment --estimates--> local data effect / optimal hyperparameters
multimodal-pretraining --accounted-as--> tokens + task samples + modality
agent-environment --produces--> trajectories --masked/weighted-as--> loss tokens
```
