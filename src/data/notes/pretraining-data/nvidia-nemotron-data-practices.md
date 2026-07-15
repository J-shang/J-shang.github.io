---
title: "NVIDIA Nemotron / Nemotron-CC 的数据实践"
description: "连接 crawl 漏斗、quality buckets、synthetic rewrite lineage、采样暴露与开放训练资产。"
topic: "pretraining-data"
section: "international-cases"
slug: "nvidia-nemotron-data-practices"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 127
readtime: 29
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/nvidia-nemotron.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/international/nvidia-nemotron.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:a8f693c2bb852b55b0bb54e2f8a0ebd8758253fc9b85f7f6ad7ce28a52b7f677"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`implementation-trace` -->
<!-- maintenance: secondary-view=`method-family/historical trace`，用于 Nemotron-4 → Nemotron 3 Super/Ultra -->
> 主要模型/资料：Nemotron-CC、Nemotron Pre-Training Dataset v2/v2.1、Nemotron-4 340B、Nemotron 3 Super/Ultra

## 这篇案例研究什么

NVIDIA 的数据实践可沿一条较完整链路研究：

```text
Common Crawl WARC
  -> extraction / language / dedup
  -> model ensemble quality buckets
  -> low-quality repair + high-quality diversification
  -> released dataset + curation code
  -> model phase mixture
  -> base/SFT/RL checkpoints + evaluation recipes
```

它的重要性不只在 token 多：Nemotron-CC公开了从raw crawl到质量桶和synthetic derivative的provenance；Nemotron 3报告又给出phase mixture、long-context exposure与post-training factories。不过最新model cards同时列出private third-party/NVIDIA data，因此“open training data”仍不等于完整production corpus全量可下载。

## 核心问题：被 filter 丢弃的网页只能删除吗

传统pipeline通常在quality threshold处二选一：保留噪声或丢掉coverage。Nemotron-CC尝试第三条路径：

```text
high-quality original -> retain + distill/extract/QA/list transformations
medium-quality original -> retain by quality bucket, no synthesis in v1
low-quality original -> Wikipedia-style rewrite to reduce noise
```

这增加的是derived surface/token exposure，不增加独立source evidence。source document、prompt、teacher、output与filter必须形成lineage，否则重写后的2T看起来会像2T新知识。

## 各代模型和 training stage

| 模型/数据 | 阶段 | 已披露数据与规模 | context/recipe | 披露 | 状态 |
|---|---|---|---|---|---|
| Nemotron-CC v1 | corpus | 99个CC snapshots；4.4T globally fuzzy-dedup real + 1.9T synthetic = 6.3T English | 5 quality levels；real/synthetic slices | `D4` pipeline / `D3` data | totals/snapshots/pipeline `verified`；rights/decontam `open` |
| Nemotron-CC-HQ | proxy corpus | 0.6T real + 0.5T diverse QA = 1.1T | short-horizon high-quality subset | `D3` | composition `verified` |
| Nemotron Pre-Training Dataset v2 | broader bundle | 6.5858T：3.3598T English CC、1.2573T synthetic CC、QA/translated QA/math/code/SFT | Nano 2/Nemotron 3 ingredient | `D3` | category totals `verified`；exact model exposure `open` |
| Nemotron Pre-Training Dataset v2.1 | additive refresh | new CC 2.5448T + CC-Code 427.9B + GitHub metadata约340B + specialized synthetic | intended with v2, not replacement | `D3` | subsets/models/metadata `verified`；license flow需逐项 |
| Nemotron-4 340B | `P0/P2` | 9T：8T P0 +1T continued；70% English、15% 53-language、15% 43-code-language | 4K | `D2/D3` | totals/mix `verified`；source manifest/order `open` |
| Nemotron-4 340B | `A1/A2` | 约20K human；SFT/preference >98% synthetic | staged SFT + preference fine-tuning + reward model | `D2/D3` | pipeline/proportions `verified`；full rows/counts `open` |
| Nemotron 3 Super 120B-A12B | `P0/P2` | 10T unique curated、25T sampled；20T diversity +5T quality/anneal | 16 categories；NVFP4 from first update | `D3/D4` | total/mixture/assets `verified`；all-source closure `open` |
| Nemotron 3 Super | `A1/A2/A3` | open SFT/RL environments；tool/search/finance/code synthetic factories | two-stage SFT → multi-stage RL/PivotRL | `D3` | many datasets/recipes open；rollout accounting partial |
| Nemotron 3 Ultra 550B-A55B | `P0/P2` | 20T：15T diversity +5T quality；19 categories；cutoff 2025-09 | NVFP4、MTP；web real+synthetic约49%→38% | `D3/D4` | totals/mix/cutoff `verified`；private slices prevent full closure |
| Nemotron 3 Ultra | long `P3` | 33B；46% long-context/54% phase-2 replay；92% iterations@1M、8%@4K | 25,165,824 tokens/iteration | `D3/D4` | exposure/length schedule `verified` |
| Nemotron 3 Ultra | `A1` | Stage1 204,800 packed samples@294,912；Stage2 19,200@515K；多类agent/safety data | two-stage SFT、MTP auxiliary loss | `D3` | nominal positions/tasks `verified`；loss masks/overall mix partial |
| Nemotron 3 Ultra | `A2/A3` | multi-environment RLVR + >10 specialized teachers MOPD；1024 prompts/batch、1 rollout、max192K | token-level distillation/masking | `D3` | framework/selected data `verified`；total rollout/loss `open` |

## 阅读这些结论前先确认的前提

| 维度 | 本篇约束 |
|---|---|
| unique real | Nemotron-CC的4.4T是对real tokens做global fuzzy dedup后的估计；synthetic不计入此unique列 |
| total dataset vs exposure | 6.3T/6.5858T/2.5448T是dataset规模；9T/25T/20T/33B是模型sampled exposure |
| synthetic | rewrite/distill/QA/translation/code transform按source derivative记录，不当作independent source |
| “open” | data files、pipeline、weights、recipe、logs逐项评估；受协议gating或private slice不等于不可用，但降低完整复现性 |
| model family | Nemotron-4 340B从头训练；Llama-Nemotron family的continued training不用于解释Nemotron 3 lineage |
| context | Ultra 20T short/base与33B long-context分开；1M serving有明确33B exposure锚点 |
| SFT positions | packed sample count×max length是nominal positions；实际loss tokens受mask/MTP影响 |
| cutoff | Ultra pretraining 2025-09、post-training 2026-05；早期Nemotron-CC以snapshot id记录 |
| 省略效应 | phase mixture、LR anneal、precision、architecture、checkpoint merge与data同时变化 |

## 厂商公开了哪些 data fields

| 字段 | Nemotron-CC/data releases | Nemotron-4 / Nemotron 3 models |
|---|---|---|
| source | Common Crawl WARC、GitHub、Wikipedia、academic/PDF、math/code、public task seeds | 上述开放源 + crawled/synthetic + third-party/NVIDIA private data |
| rights | CC原页权利逐页异质；NVIDIA Data Access Agreement限制到model training；部分specialized为CC/GFDL | model license与data licenses分开；teacher license可能给下游分发带来义务 |
| cutoff | v1 CC-MAIN-2013-20→2024-30；v2.1至2025-26 | Ultra pretrain 2025-09、code refresh 2025-09-30、post 2026-05 |
| accounting | real unique、synthetic、category tokens与WARC metadata | unique/sample/long/SFT samples/RL prompts分列；loss tokens多为unknown |
| quality | 3 classifier ensemble→20 quantiles→5 downstream-calibrated levels | phase1 diversity→phase2 quality；domain-specific validators与teacher/judge |
| dedup | document fuzzy + exact substring；synthetic/post code另有title/schema/embedding dedup | code benchmarks exact+embedding decontam；agent data按repo/task需group split |
| decontam | v1明确未做dataset decontam | Super code synthetic对4 benchmarks；Ultra多处task-specific；全库manifest仍不完整 |
| order | dataset slices独立 | Nemotron-4 8T→1T；Super 20T→5T；Ultra 15T→5T→33B LC |
| validation | 50B bucket anneal、1T/15T proxy、filter/synthetic ablations | checkpoint branches/merge、100B domain injections、base/post/long eval |
| artifacts | dataset cards/files、WARC ids、teacher/prompt metadata、NeMo Curator | base/post/quantized weights、Megatron/NeMo RL/Evaluator、recipes；部分private data |

## Nemotron-CC：从99个snapshot构造6.3T

### extraction、language 与 dedup

`[来源事实 | verified]` v1覆盖CC-MAIN-2013-20至2024-30共99 snapshots。对13个snapshot的对比显示，dedup后Trafilatura-filtered、jusText-filtered、jusText分别产出994B、1.380T、1.804T tokens；按FineWeb-Edu 3–5分定义的HQ绝对量为80B、104B、127B。

pipeline选择jusText以提高absolute HQ yield，然后：

- pycld2与FastText lid176识别English，threshold 0.3；
- global fuzzy document dedup；
- 将snapshots按八分组做exact substring dedup；
- 对低质量split才使用heuristic + Wikipedia/books KenLM perplexity filters，避免误删model classifier判为HQ的token。

`[综合判断 | supported]` “HQ占比略降但HQ绝对量上升”是long-horizon设计目标，不代表多保留数据无成本；重复、rights、toxicity和low-value exposure仍需独立约束。

### 三个classifier、20桶与5个quality levels

两个quality regressors分别用Mistral-8x22B-Instruct和Nemotron-4-340B-Instruct给460K FineWeb documents打0–5 educational value标签，再在Snowflake embedding上训练；第三个为DCLM fastText classifier。

每个classifier把全库rank切成20个约5% quantile，document最终分数取三者integer score的最大值。随后用一个训练到70%进度的8B model做50B-token continued-pretraining probe：

$$
p_{\text{default}}=0.66,qquad
p_{\text{tested bucket}}=0.34,qquad
T_{\text{probe}}=50\text{B}.
$$

按9个task平均结果将20桶合成High、Medium-High、Medium、Medium-Low、Low五级，token分别为553B、504B、2.023T、894B、402B。

`[综合判断 | supported]` 这比直接把classifier score当quality更接近operational validation，但仍把9-task distribution、70%-checkpoint与50B horizon写进了quality label；换target scale/domain后可能rank reversal。

### 1.9T synthetic不是1.9T新source

低质量文档用Mistral-Nemo-12B-Instruct做Wikipedia-style rewrite，产出336.3B。高质量文档做Wikipedia rewrite、Diverse QA、Distill、Extract Knowledge、Knowledge List，合计约1.5T；medium quality因资源限制未合成。

最终：

$$
T_{\text{Nemotron-CC}}
=4.4\text{T real unique}+1.9\text{T synthetic}
=6.3\text{T}.
$$

报告表中具体synthetic子项约1.8739T，与1.9T headline差异属于舍入。metadata至少应保存：

```text
WARC/source document id
  -> quality classifiers/scores
  -> selected segment
  -> prompt id + teacher version + decoding config
  -> synthetic output + post-filter reason
```

`[未知 | open]` 报告明确未逐条验证rewrite factuality；低质量rephrase ablation平均提升1.5点但部分task下降，可能存在misinformation。

### validation与未去污染

1T对照固定73% tested crawl +27%同一non-crawl blend；Nemotron-CC-HQ比DCLM的MMLU高5.6点。15T horizon对照也报告结果。synthetic ablation用fixed 1T，把low raw换成rewrite，或将8次HQ repeat中的4次换成synthetic derivatives。

`[来源事实 | verified]` v1论文明确没有decontaminate dataset。作者认为比较双方都可能污染且先前DCLM分析未显示污染解释结果；这不能替代clean evaluation。开放对照仍应发布benchmark exact/near/semantic overlap与clean subset。

## v2/v2.1：从单一English crawl扩成pretraining bundle

### v2的6.5858T

托管在`Nemotron-CC-v2`下的pretraining bundle给出：English CC 3.3598T、English synthetic CC 1.2573T、diverse QA 692.9B、translated QA 558.2B、math 206.2B、math SFT 190.6B、synthetic code 174.9B、MMLU SFT 81.6B、code SFT 58.5B、general SFT 5.7B，总计6.5858T。

`[重要边界 | verified]` 这不是“Nemotron-CC v1从6.3T小改到6.6T”的简单版本升级，而是供模型训练的多category bundle；比较时必须用dataset card version与category表。

### v2.1是增量，不是replacement

v2.1 card明确要求与既有数据一起使用，包含：

- 2.5448T new English CC：organic、translated-to-English与synthetic rephrases；其中2.1228T为medium-high synthetic；
- 427.9B CC code，经Lynx rendering、Phi-4 cleaning、Qwen3 quality 1–3过滤，并保存WARC id/models used；
- GitHub code v2增加377M filtered/dedup files，约340B raw-source tokens，并衍生QA/review/student-teacher/rewrite/transpile；
- specialized synthetic：RQA 134.6B、InfiniByte 19.4B、Wiki rewrite 7.9B、scientific code 1.2B、math textbooks 25.1B、STEM SFT 82.5B。

Dataset Access Agreement允许model training但要求登录同意；specialized subsets另有CC BY/SA/GFDL。card还提示Qwen/DeepSeek/Phi teacher licenses可能对发布下游模型产生redistribution/use obligations。

`[综合判断 | supported]` synthetic provenance不仅是科学问题，也是license graph：每条derivative需记录source license、teacher model/license、prompt与output license interpretation，不能只给最终dataset一个总标签。

## Nemotron-4：9T base/continued 与98% synthetic alignment

### 8T + 1T

Nemotron-4 340B总训9T：70% English natural、15% multilingual natural（53 languages）、15% source code（43 languages）；前8T formal pretraining，后1T continued training。

continued阶段多数复用已见token但提高高质量source权重；另一小部分加入QA-style examples并upweight模型低准确率domain，同时做更陡LR decay。因而1T不是全新unique data：

$$
T_{\text{continued}}=T_{\text{replay-remix}}+T_{\text{QA/weak-domain}},
\qquad \text{breakdown}=\text{unknown}.
$$

### alignment synthetic flywheel

报告称alignment只使用约20K human annotations：10K SFT、10K HelpSteer2/reward/preference；SFT与preference training中>98%为synthetic。pipeline从约3K topics、12K Python keywords、17K math keywords生成prompts/dialogues/preferences，并由ground truth、program verifier、LLM judge或Nemotron reward model排序。

generator从Mixtral-8x7B-Instruct迭代到中间Nemotron instruct checkpoints；这是checkpoint-dependent data flywheel：

```text
teacher checkpoint -> generated dataset version
-> aligned student checkpoint -> next teacher
```

“学生超过初始teacher”支持tested iterative recipe，不证明synthetic supervision没有bias ceiling；base能力、data与judge一起变化。

## Nemotron 3 Super：10T unique、25T sampled与两阶段mixture

`[来源事实 | verified]` Super在10T unique curated corpus上sample 25T：前20T强调diversity/coverage，后5T强调quality并做LR annealing。16类包括quality-tier web及其synthetic、math、Wikipedia、code/CC-code、academic、Crawl++、multilingual、finepdfs和general/STEM/code SFT-style data。

phase1→phase2并非只“换高质量网页”：phase2把finepdfs从6.1%变为finepdfs-high 14.3%，synthetic-crawl-high仍22.4%、code仍14%、math仍6.4%，移除部分低级crawl并调整STEM/SFT。因此可解释为joint mixture curriculum。

Super的synthetic code示例提供可检查provenance：从HumanEval抽取91 concepts、生成约14M problems/23M pairs、清洗到15M；另一个0.2B code集对HumanEval/MBPP/CRUXEval/LiveCodeBench做solution exact和Qwen embedding >0.8过滤。100B redo ablation报告1–2点局部提升。

`[综合判断 | supported]` 从benchmark training concepts衍生pretraining data即使去除test近邻，也可能造成task-format specialization；应同时报告seed provenance、semantic threshold与非同构clean benchmark。

## Nemotron 3 Ultra：20T + 33B、公开资产与private slices

### 15T diversity → 5T quality

Ultra在NVFP4上训练20T，约15T/75%后切到phase2。19类中quality-filtered/synthetic web约占phase1 49%、phase2 38%；其余包括code 14%、math 6.4%、multilingual 5%、SFT-style、PDF、academic、legal等。model card给pretraining cutoff 2025-09；source-code refresh新增173B、cutoff 2025-09-30。

Ultra从public benchmark train splits生成多选/开放QA，held-out test不作seed，并做schema/dedup/task filter。100B phase-3 ablation使GPQA 30.8→41.9等；仍需注意task format与benchmark decision exposure。

### 33B long-context exposure

Ultra对选定checkpoint再做33B CPT：46% long-context data、54% phase2 replay；92% iterations使用1,048,576长度，8%使用4,096，并且每iteration 25,165,824 tokens。4K iterations只放math/code SFT-style以保短任务能力。

`[推导结论 | verified]` 若每iteration token数恒定，则nominal long-vs-short positions约为：

$$
T_{1M}\approx0.92\times33\text{B}=30.36\text{B},\qquad
T_{4K}\approx2.64\text{B}.
$$

这是按iteration share推导的sampled positions；46/54是data category blend，不能与92/8 length schedule逐项对应。

### SFT nominal positions与provenance

Stage1为204,800 packed samples@294,912，Stage2为19,200@515,000：

$$
T_{\text{SFT nominal}}
=204{,}800\times294{,}912
+19{,}200\times515{,}000
\approx70.29\text{B positions}.
$$

实际loss tokens因prompt/reasoning-budget mask、padding、MTP auxiliary objective而不同。公开细节包括：

- safety：45K English + 六语各约15K = 135K，back-translation similarity <0.8移除，每语约去10–15%；
- OpenResearcher：97K raw中筛commercial-OK约21.7K trajectories；
- terminal use：约370K multi-turn conversations；
- long context：512K synthetic document reasoning；
- software issue、tool use、search、GPU kernel、RTL、multilingual等各有teacher/harness/filter。

MOPD使用>10个specialized teachers，batch 1,024 prompts、每prompt 1 rollout、max generation 192K，并以token-level dense guidance合并。总iterations、generated/accepted/loss tokens仍需训练日志闭合。

### “open data”仍有缺口

Ultra发布base/post/quantized weights、many datasets、Megatron/NeMo recipes与Evaluator examples；但model card同时列出public/crawled、third-party private、NVIDIA private和synthetic datasets。故应写：

`substantial training assets are open --does-not-imply--> every production sample is public`。

开放复现还需将每个private slice的token share、data contract、validation影响和是否可替代写入manifest。

## 如何验证这些结论，并检查 contamination

| 问题 | 首个可能差异 | 区分性检查 |
|---|---|---|
| jusText更好是否只因token更多 | extraction quality vs absolute yield | fixed raw pages、fixed sampled tokens与fixed compute三组；per-domain retention |
| ensemble bucket是否可迁移 | 9-task proxy/8B stage vs target model/domain | 多scale checkpoint、domain macro、rank correlation与recalibration |
| rewrite是否修复而非幻觉 | denoise vs factual mutation/style collapse | source-conditioned factuality、entity/number consistency、source-group split |
| synthetic替换repeat是否有效 | fresh surface vs QA task format | same source groups，rewrite/QA/paraphrase factorial；非QA clean eval |
| phase2 quality收益来自data吗 | mixture、LR anneal、precision、merge共同变化 | fixed checkpoint/LR，仅swap mixture；另做LR-only与merge-only |
| 1M能力来自何种data | 46/54 category vs92/8 length | length×category factorial，per-length/per-domain macro |
| benchmark-seeded synthetic是否污染 | test exact match vs train-format transfer | held-out source family、semantic/procedural overlap、time-split benchmark |
| open recipe能否重建Ultra | private slice与released slice差异 | full manifest token closure、private replacement ablation、checkpoint hash/log |

## 哪些经验可以借鉴，哪些不能直接照搬

### 可迁移

- `[综合判断 | supported]` quality classifier应通过downstream proxy校准，但quality label必须携带proxy model、horizon、task mix与版本。
- `[综合判断 | supported]` synthetic rewrite的统计单位是source-derived tokens，应保存source→teacher→prompt→output lineage。
- `[综合判断 | supported]` dataset tokens、unique real tokens、sampled exposure、nominal positions 与 loss tokens 需要分成五列统计。
- `[综合判断 | supported]` long-context recipe必须同时报告data-category mix与 seqlen schedule；两者不是同一分布。
- `[综合判断 | supported]` dataset license需要provenance graph，尤其当多teacher license可能传递义务。
- `[综合判断 | supported]` 开放程度应逐artifact评估；大量open assets与少量private slices可以同时存在。

### 不可外推

- `[未知 | open]` Nemotron-CC v1逐页面rights、完整toxicity/PII策略及clean benchmark score；
- `[未知 | open]` Nemotron-4 1T continued中replay/QA/weak-domain比例与unique tokens；
- `[未知 | open]` Super 10T unique如何映射25T每个source的repeat与loss exposure；
- `[未知 | open]` Ultra private third-party/NVIDIA data的token share与可替代性；
- `[待验证假设 | plausible]` 50B/8B质量桶排序对550B-A55B、20T horizon仍保持单调。

## 读完后应该掌握什么

读完后应能：

1. 复原Nemotron-CC的4.4T real +1.9T synthetic，并解释synthetic为何不是新source；
2. 说明20 quality buckets怎样经50B anneal变成5 levels；
3. 区分Nemotron-CC v1、v2 broader bundle与v2.1 additive refresh；
4. 分别计算 Ultra 的 20T+33B 和约 70.29B SFT nominal positions，并说明它们为何不能直接相加；
5. 解释Nemotron 3为何接近D4但不能无条件称完整production data可复现。

## 用这些问题检查自己

1. 一个rewrite保持topic但修改关键数字，应在哪些自动/人工指标上被拒绝？
2. 质量桶在MMLU上排序稳定、在低资源语言上反转，如何重做label taxonomy？
3. 1T实验中73%crawl固定，但synthetic QA比例变化，能否把MMLU增益归为web quality？
4. Ultra 46% long data与92% 1M iterations为什么不能直接相乘成“长文档1M token量”？
5. teacher license变化时，如何从dataset row追踪到受影响的released checkpoint？

## 来源与建议阅读位置

1. [Nemotron-CC paper](https://arxiv.org/abs/2412.02595)
   - 为什么读：固定99 snapshots、extraction/dedup、3-classifier→20→5桶、1.9T synthetic、1T/15T ablation与未decontam限制。
   - 建议位置：§2、Tables 1–4、§3.3–3.5、Limitations。
2. [Nemotron-CC pipeline in NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)
   - 为什么读：检查可执行的extraction、dedup、quality labeling与SDG实现，不只依赖论文流程图。
   - 建议位置：Nemotron-CC example/pipeline、configs、dedup/classifier/SDG modules。
3. [Nemotron-CC v2 dataset card](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2)
   - 为什么读：核对6.5858T broader pretraining bundle的十类token分布与teacher totals。
   - 建议位置：Data Overview、Data distribution、Filtering、License。
4. [Nemotron Pre-Training Dataset v2.1 card](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1)
   - 为什么读：固定2.5448T CC refresh、427.9B CC code、340B GitHub metadata、specialized synthetic、WARC/teacher/prompt/license metadata。
   - 建议位置：Dataset Details、Quantification、License/Terms、Ethical Considerations。
5. [Nemotron-4 340B report](https://arxiv.org/abs/2406.11704)
   - 为什么读：核对9T 70/15/15、8T→1T continued与>98% synthetic alignment flywheel。
   - 建议位置：§2.1–2.3、§3.2–3.3。
6. [Nemotron 3 Super report](https://arxiv.org/abs/2604.12374)
   - 为什么读：固定25T两phase、16类mixture、specialized pretraining data、SFT/RL factories与open artifacts。
   - 建议位置：§2.3–2.4、Figure 10、§3.1–3.2。
7. [Nemotron 3 Ultra report](https://arxiv.org/abs/2606.15007)
   - 为什么读：固定20T+33B、19类phase mixture、cutoff/code refresh、SFT nominal exposure、multi-teacher MOPD与agent data。
   - 建议位置：§2.3–2.5、Figure 4、§3.1–3.3。
8. [Nemotron 3 Ultra model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
   - 为什么读：确认pre/post cutoff、public/private/synthetic source taxonomy、weights/license与evaluation assets。
   - 建议位置：Model Overview、Training and Evaluation Datasets、Training Methodology、License。
9. [NVIDIA Nemotron pretraining dataset collection](https://huggingface.co/collections/nvidia/nemotron-v3-pre-training)
   - 为什么读：定位每个release dataset及其版本，不把collection、dataset repository和model exposure混为一谈。
   - 建议位置：collection items、linked cards、release dates。
10. [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)
    - 为什么读：复核Ultra benchmark container/config与可重放evaluation边界。
    - 建议位置：Nemotron examples、task configs、container versions。

## 这篇案例与主线知识的关系

```text
WARC snapshot --processed-by--> extraction/filter/dedup version
quality classifier score --calibrated-by--> 50B downstream probe
source document --parent-of--> synthetic rewrite/QA/distill
dataset unique tokens --sampled-into--> model phase exposure
seqlen schedule --orthogonal-to--> data-category mixture
released assets --partially-reconstruct--> production training run
teacher/source licenses --propagate-through--> synthetic data provenance graph
```
