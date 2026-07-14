---
title: "Huawei PanGu / openPangu 的数据实践"
description: "沿 raw-to-retained 漏斗、domain routing、vertical replay 与阶段 mixture 审计开放资产边界。"
topic: "pretraining-data"
section: "china-cases"
slug: "huawei-pangu-openpangu-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 110
readtime: 32
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/huawei-pangu-openpangu.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/china/huawei-pangu-openpangu.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:25d1f8349466d9bb2cadaa00dba4dbee9c8e0bb195682b993a209216f2e2bbf7"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family/historical trace`
> 辅助视角：`implementation-trace`，用于 PanGu-α 数据管线、PanGu-Σ domain routing 与 openPangu artifact boundary
> 研究锚点：PanGu-α、PanGu-Σ、PanGu-π/YunShan、Pangu Ultra、openPangu Embedded/2.0

## 定位

PanGu 家族的数据史包含三条不能互相补空的证据链：

- PanGu-α 披露近 80TB raw Chinese data 如何经过规则、模型过滤和跨源模糊去重变成 1.1TB corpus；
- PanGu-Σ 与 PanGu-π 分别把 domain identity 用于专家路由和行业 continued pretraining，公开了 domain token 预算与格式；
- Pangu Ultra 给出 13.2T 的阶段 mixture、质量评分 proxy 与 post-training taxonomy，而 2025–2026 openPangu 模型卡发布权重和总 token，却通常不发布 source-level corpus、顺序或 validation manifest。

因此，“openPangu”中的 open 主要描述模型与推理资产的可获得性，不能自动解释为 training data 开放。

## Motivating problem：同一个“行业模型”标签可能对应三种数据关系

```text
general corpus + domain tag
  -> domain-conditioned expert routing             (PanGu-Σ)

general checkpoint + vertical corpus + replay
  -> continued pretraining + instruction tuning    (PanGu-π/YunShan)

general pretraining mixture already contains industries
  -> one shared dense model                         (Pangu Ultra)
```

三者分别改变 routing、training distribution 与 source mixture。行业 benchmark 提升不能区分这三种机制，也不能反推出 proprietary industry records 的存在。

## 代际与训练阶段表

| 模型 | 阶段 | 已披露数据与规模 | objective/context | 披露 | 状态 |
|---|---|---|---|---|---|
| PanGu-α 2.6B/13B/200B | `P0` | 近 80TB raw → 1.1TB 中文 corpus；200B 所用 1TB 数据含约 258.5B corpus tokens | autoregressive LM；2048 context | `D2/D3` | raw/retained/source/filter `verified`；sampled/loss exposure `open` |
| PanGu-Σ 1.085T | continued `P0/P2` | 40 domains、329B consumed tokens；四大域 304.12B，其他 36 域 >25B | domain ID → Random Routed Experts；1024 packing | `D2` | domain tokens/format/routing `verified`；unique/repeat `open` |
| PanGu-π 1B/7B | `P0` | 1.6T tokens，中英 1:1，约 1 epoch | BPE、约 100K vocab | `D2` | total/language ratio `verified`；source/cutoff/filter `open` |
| YunShan | domain `P2/A1` | finance 36.5B、legal 111.7B；旧高质量数据与 vertical corpus 1:1；995K domain instructions | continued pretraining + one-stage SFT | `D2` | corpus/task counts `verified`；sampled exposure `open` |
| Pangu Ultra 135B dense | `P0/P2/P3` | 12T general + 0.8T reasoning + 0.4T annealing = 13.2T | 4K→8K→32K→128K；domain-aware tokenizer | `D2` | phase totals/mixture `verified`；long-context sub-budget `open` |
| Pangu Ultra | `A1/A2` | open/industrial/synthetic instruction；约 6/7 为 math/code/logic reasoning | SFT → outcome-reward RL | `D2` | source/task proportions `verified`；samples/rollouts/loss `open` |
| openPangu Embedded 1B/7B V1.1 | `P0/A*` | 1B：10T；7B：25T；from scratch | native 32K；7B fast/slow fusion，1B on-policy distillation + multi-source reward RL | `D1/D3` | total/model card/artifacts `verified`；corpus recipe `open` |
| openPangu-R-7B-Diffusion | continued `P2/A1` | Embedded-7B 后 700B@8K + 100B@32K annealing + 10 epochs on 10B slow-thinking SFT | autoregressive base → diffusion LM | `D1/D3` | stage headline `verified`；10B unique/sample relation `open` |
| openPangu Ultra-MoE-718B V1.1 | `P*/A*` | card 未给 pretraining tokens/source；V1.1 强化 agent、降幻觉 | 718B total/39B active；fast/slow | `D1/D3` | model/artifact `verified`；data recipe `open` |
| openPangu 2.0 Flash | `P*/A1/A2/A3` | 约 34T total training tokens；source/mix 未给 | 约 92B/6B active、512K；SFT + multi-domain RL + online policy distillation | `D1/D3` | total/stage names `verified`；accounting/corpus `open` |

## Assumption ledger

| 维度 | 本篇约束 |
|---|---|
| 组织边界 | Peng Cheng Laboratory、Huawei/MindSpore 与 openPangu community 的共同作者/资产按具体来源记录，不把合作论文全部归成单一企业产品 |
| token 单位 | corpus tokens、consumed/sampled tokens、SFT examples、epochs 与 context positions 分开 |
| “general phase” | Pangu Ultra 的 12T general 包含 7.4T 与 4.6T 两个质量子阶段，不等于只含 general text |
| “industry” | source domain、data tag、expert route、continued-pretraining corpus 与 product use case 分开 |
| “open” | weights/code/license 与 training corpus/manifest/rights 分开评定 |
| lineage | dense Pangu Ultra 135B、openPangu Ultra-MoE 718B 与 openPangu 2.0 Flash 是不同模型；名称相似不建立 data inheritance |
| context | native/serving context 与已量化 training exposure 分开；未给 32K/128K token 数时不反推 |
| 省略效应 | architecture、hardware、tokenizer、data、context、routing 与 post-training 跨代同时改变 |

## 统一数据字段

| 字段 | PanGu-α/Σ | PanGu-π/Ultra | openPangu 当前模型 |
|---|---|---|---|
| source | public datasets、Common Crawl、百科、电子书、新闻；Σ 加 Pile/C4、代码和多语/行业域 | π 仅称 Internet 中英；YunShan 给 finance/legal sources；Ultra 为中英 web/books/encyclopedia、多语/industry、math/code/instruction | model cards 多只给 total tokens 与任务能力，source manifest 未给 |
| rights | source-level license/consent 未形成完整表；电子书与 web 尤需逐源核对 | public + crawled + industrial queries；逐样本 rights 未披露 | model license 不覆盖 corpus；training rights `unknown` |
| cutoff | α Common Crawl 为 2018-01 至 2020-12、news 为 1992–2011；其余 snapshot 不完整 | `unknown` | `unknown` |
| token accounting | α 1.1TB/约 258.5B corpus tokens；Σ 329B consumed | π 1.6T；Ultra 13.2T 分三 phase；post counts unknown | Embedded 10T/25T、2.0 Flash 34T；unique/sample/loss 与阶段闭合 unknown |
| quality | α rule + fastText spam + quality classifier + manual/PPL loop；Σ code compile/length filters | Ultra rule/model scores cleanliness、fluency、educational value、richness，并按分数优先采样 | 高层称数据/训练策略优化；阈值、retention、calibration 未给 |
| dedup | α 文内段落去重 + 全源 fuzzy document dedup；Σ 跨源去重细节不足 | Ultra 只称 minimal redundancy | `unknown` |
| order | α source sampling weights；Σ domain ID 随 sample 进入 routing | Ultra general→reasoning→annealing、4K→128K curriculum | 只披露部分 stage names；global order/seed/restart invariant 未开放 |
| synthetic | α 未建立现代 synthetic recipe | Ultra reasoning 与 annealing 广泛使用 LLM synthetic；post 也含 synthetic problems | fast/slow SFT、RL、distillation 存在；生成/验证 provenance 未量化 |
| validation | α 人审 + 350M proxy PPL；Ultra 2.6B quality proxy、benchmark/architecture ablation | π general/domain benchmarks；Ultra pre/post/long tests | release benchmarks 多，pretrain held-out/contamination contract 不完整 |
| artifacts | α MindSpore code、预处理入口、模型配置；Σ/π/Ultra reports | reports，无 corpus/order/logs | weights、custom model/inference code、license；无 corpus manifest |

## PanGu-α：80TB → 1.1TB 的可检查数据漏斗

### source 与 retained corpus

`[来源事实 | verified]` PanGu-α 从近 80TB raw data 建立 1.1TB 中文 corpus。最终 source 组成按 storage 为：public datasets 27.9GB、encyclopedia 22GB、e-books 299GB、Common Crawl 714.9GB、news 35.5GB；Common Crawl snapshot 为 2018-01 至 2020-12，news 为 1992–2011。

200B 模型使用的 1TB training dataset 在论文表中按 corpus token 统计为：

$$
T_{\text{corpus}}=
25.8+30.9+176.2+19.8+5.8
=258.5\text{B tokens}.
$$

这里的 $T_{\text{corpus}}$ 是各 source 可用 token 量，不是训练 loss token。source sampling weight、repeat/epoch 与 packing 会改变实际 exposure；论文表格不足以恢复逐 token global order。

### filtering 与 iterative evaluation

Common Crawl cleaning 先移除中文字符比例低于 60%、少于 150 字、只有标题、广告/导航、重复段落等文档，并转换繁体为简体。后续三类 filter 为：敏感词规则、fastText spam classifier、仿 GPT-3 的 quality classifier；最后做跨 source fuzzy document dedup。

`[来源事实 | verified]` 规则/阈值不是一次固定：团队对随机样本做人审，并在候选处理数据的 30GB sample 上训练 PanGu-α 350M，以高质量 development set PPL 比较处理版本，再修改规则和模型。

```text
raw snapshot
  -> deterministic/rule cleaning
  -> spam and quality classifiers
  -> cross-source fuzzy dedup
  -> human review + 350M proxy PPL
  -> revise rules/models
  -> source-weighted training selection
```

`[综合判断 | supported]` 这是一条早期 human-model data loop，但 PPL 只衡量 proxy 在特定 development distribution 上的拟合；它不是安全性、rights、事实性或低资源 domain coverage 的统一质量分数。

## PanGu-Σ：domain ID 是 training record 的一部分

`[来源事实 | verified]` PanGu-Σ 从 PanGu-α continual training，消费 329B tokens、覆盖 40 domains。四大域为 bilingual 77.51B、Chinese 75.47B、English 75.90B、Python/Java code 75.24B，共 304.12B；其余 36 域（26 自然语言、6 编程语言、finance/health/law/poetry）超过 25B。

每个 sample 除 token IDs 外还带 domain ID；Random Routed Experts 用 domain ID 与固定随机 routing table 决定 token 进入哪些专家。预训练数据插入 language/code control token 与 `<EOT>`，随后按 1024 长度连续 packing。

```text
source document
  -> domain assignment
  -> control token + EOT formatting
  -> concatenate and slice @1024
  -> (token_ids, domain_id)
  -> domain-conditioned expert route
```

`[综合判断 | supported]` 若 domain classifier、taxonomy 或 packing boundary 改变，模型计算路径也会改变；所以 domain label 应像 token sequence 一样版本化。该关系是 `data label --implemented-by--> expert routing`，不是普通 metadata。

`[未知 | open]` 329B 是否精确等于各 domain unique corpus token、各域 repeat 次数、cross-domain near-duplicate 与 validation split 均未完整披露。

## PanGu-π / YunShan：vertical data 与 replay 的边界

### general recipe

`[来源事实 | verified]` PanGu-π 的 general pretraining data 为 Internet 中英文 1:1，tokenize 后约 1.6T tokens，训练约 1 epoch；BPE/SentencePiece vocab 约 100,883。报告没有给 source-level mixture、snapshot、rights、dedup 或 benchmark decontamination manifest。

### finance/legal continued pretraining

YunShan 的 finance corpus 为 company announcements、financial news/articles/exams，清洗后 36.5B tokens；legal corpus 为 regulations、case text、papers/exams，清洗后 111.7B。模型把 previous high-quality data 与 vertical corpus 按 1:1 混合，以缓解 catastrophic forgetting；之后将 general 与 995K domain instructions 混合做 one-stage SFT。

设 vertical sampled tokens 为 $T_v$，若 1:1 指 token-level sampling，则 nominal replay 为：

$$
T_{\text{continued}}=T_v+T_{\text{replay}}=2T_v,
\qquad T_{\text{replay}}=T_v.
$$

`[推导结论 | open]` 这是条件推导，不是报告给出的总 exposure：1:1 也可能按 batch/example 实现，且未给 epochs、sequence length、packing 与 loss mask。复现必须核对 sampler config。

## Pangu Ultra：13.2T phase mixture 与质量分数实验

### 三阶段配方

Pangu Ultra 的 pretraining exposure 按报告闭合为：

$$
T_{\text{P0/P2/P3}}=12.0\text{T}+0.8\text{T}+0.4\text{T}=13.2\text{T}.
$$

12T general 又分 7.4T 与 4.6T，后半使用更高质量数据。各 phase 的 source mixture 为：

| 类别 | General 12T | Reasoning 0.8T | Annealing 0.4T |
|---|---:|---:|---:|
| General English | 54% | 14% | 21% |
| General Chinese | 13% | 6% | 20% |
| Multilingual | 8% | 4% | 3% |
| Instruction | 2% | 11% | 20% |
| Math | 6% | 28% | 18% |
| Code | 17% | 37% | 18% |

reasoning phase 的 math+code 为 65%，并广泛使用 LLM-generated synthetic data；annealing 将 instruction 提至 20%，包括内部题库与精炼的短/长 CoT。reasoning data 按质量和难度标注并做 simple→complex curriculum。

### domain-aware tokenizer

Pangu Ultra 不直接让高资源 general text 的全局频率决定词表，而是分别统计 general Chinese、general English、code 与 math，再 merge/deduplicate，形成 153,376 vocab。该设计把 data mixture 影响前移到 tokenizer construction；审计需要保留每个 domain 的 tokenizer-training sample 与 merge policy。

### quality evaluator 与 1.6× proxy

`[来源事实 | verified]` 团队用人工标注微调 PanGu evaluator，对超过 10T pretraining corpus 按 cleanliness、fluency、educational value、richness 打分，并提高高分 sample 的采样概率。2.6B proxy 实验中，低分数据需 1.6× tokens 才达到高分数据的 comparable performance。

`[综合判断 | supported]` 该 1.6× 是特定 evaluator、proxy、metric 与 data partition 下的 empirical association；它不能转换为跨语言、跨规模的“质量系数”，也不能证明 evaluator 未继承 Pangu family 的偏好。

### context 与 post-training 的未闭合分母

训练先 4K，12.0T–12.8T 提至 8K，之后两段扩至 32K、128K；报告没有给后两段各自 token 数，因此 13.2T 只能按 phase 总量记录，不能再额外相加 long-context exposure。

post-training source 包括 open instruction、real-world industrial queries 与从 pretraining corpus 合成的问题，按 domain×task hierarchy 平衡；约 6/7 为 math/code/logic reasoning。SFT 后使用 outcome reward RL，但 sample、prompt、rollout、accepted trajectory 与 loss token 均未知。

## openPangu：开放权重不等于开放数据

### Embedded 与 diffusion continuation

`[来源事实 | verified]` 截止 2026-07-14，Embedded-1B-V1.1 card 给 10T、Embedded-7B-V1.1 给 25T，均称从零在 Ascend NPU 训练、native 32K。7B 具备 fast/slow fusion；1B V1.1 另称 offline on-policy distillation 与 multi-source reward RL。

openPangu-R-7B-Diffusion 明确从 Embedded-7B 继续 700B@8K、100B@32K annealing，再对一个 10B slow-thinking SFT corpus 做 10 epochs。这里至少有三种分母：continued-pretraining sampled tokens、SFT unique tokens、SFT nominal epoch exposure；不能把“10 epochs × 10B”直接当 loss tokens。

### Ultra-MoE 与 2.0 Flash

openPangu Ultra-MoE-718B-V1.1 card 给出 718B total/39B active、fast/slow 与 V1.1 agent/post-training变化，但未给 pretraining token 或 source。它与 135B dense Pangu Ultra 的命名相似，公开资料不足以建立 checkpoint 或 corpus inheritance。

`[来源事实 | verified]` 2026-07 发布的 openPangu 2.0 Flash card 给约 34T total training tokens、512K context，以及 SFT、multi-domain RL、online policy distillation 的 stage names；没有说明 34T 是否仅为 pretraining sampled tokens、是否包含 continued/mid-training，也没有 source/mix/cutoff/filter/unique/loss accounting。

因此只可记录：

```text
model weights + architecture/inference code + release evaluation
  --supports--> artifact inspection and serving reproduction

missing corpus manifest + processing config + global order + logs
  --prevents--> data recipe reproduction
```

## Validation、污染与区分性检查

| 问题 | 首个可能差异 | 区分性检查 |
|---|---|---|
| α quality loop 是否有效 | filter 版本 vs source mix/retention | 固定 raw snapshot 与 retained tokens，仅替换 filter；per-source PPL + macro downstream + safety/rights audit |
| Σ domain experts 是否优于普通 MoE | domain labels vs routing capacity | fixed data/compute，random/domain/learned routing 三组；报告 per-domain 与 load balance |
| YunShan replay 是否防遗忘 | 1:1 unit、domain exposure、tokenizer 扩展 | token-level 与 batch-level 1:1 对照，general/domain validation 双轴 |
| Ultra 1.6× 是否迁移到 target scale | evaluator calibration、proxy size、metric | 多 seed/scale、不同 domain 与 clean held-out；检查 quality rank reversal |
| synthetic reasoning 是否增加独立证据 | source problem vs teacher surface diversity | 按 source problem group split，记录 generator/verifier/acceptance 与 solution diversity |
| 34T openPangu 是否可与 13.2T Ultra 比较 | model lineage、tokenizer、stage accounting | 官方 checkpoint lineage + per-stage unique/sample/loss token ledger；否则保持 `open` |

PanGu-α 的 350M PPL loop、Pangu Ultra 的 2.6B proxy 与 architecture ablations提供局部可检查锚点，但都缺完整 source-group contamination manifest。openPangu cards主要报告 release benchmarks；训练期 held-out source、time boundary、per-domain token count、decision-layer exposure 与 repeated benchmark selection仍不完整。

## 可迁移经验与不可外推部分

### 可迁移

- `[综合判断 | supported]` data filter 应作为迭代实验版本化，并同时保留 raw/retained counts、proxy distribution 与非 PPL 风险指标。
- `[综合判断 | supported]` domain label 若进入 expert routing，就属于训练输入与计算路径的一部分，必须带 taxonomy/version/hash。
- `[综合判断 | supported]` domain continued pretraining 的 replay ratio 必须注明 sampling unit；“1:1”本身不足以复现 exposure。
- `[综合判断 | supported]` tokenizer vocabulary 也有 domain mixture，需与模型训练 mixture 分开审计。
- `[综合判断 | supported]` weight/code openness与data/order openness应逐字段评分，不能合成一个 open/closed 标签。

### 不可外推

- `[未知 | open]` PanGu-α 1.1TB 的完整 source rights、精确 sampled/loss tokens 与 benchmark contamination；
- `[未知 | open]` PanGu-Σ 329B 的 unique/repeated breakdown 和 domain assignment error；
- `[未知 | open]` Pangu Ultra 32K/128K 各段 token 数、post-training总量及 6/7 reasoning 的统计单位；
- `[未知 | open]` Embedded/Ultra-MoE/2.0 的 source、rights、cutoff、mixture、dedup/decontam 与 total-token阶段边界；
- `[待验证假设 | plausible]` Pangu-family evaluator 的质量排序能跨语言、domain、architecture scale 保持校准。

## 掌握标准

读完后应能：

1. 解释 80TB raw、1.1TB retained、258.5B corpus tokens 与 sampled/loss tokens的区别；
2. 说明 PanGu-Σ 的 domain ID 为什么不只是日志字段；
3. 在未知 sampler unit 时，正确解释 YunShan 的 1:1 replay；
4. 复原 Pangu Ultra 的 13.2T 三阶段 mixture，并指出 long-context 子预算未知；
5. 解释为什么 openPangu 2.0 的 34T 不能直接与 dense Pangu Ultra 13.2T做数据效率比较。

## 推理型自测

1. 若 α 新 filter 使 PPL 降低但法律文本保留率减半，应如何设计多目标 acceptance gate？
2. Σ 某小域得分提升，怎样排除 expert capacity 而不是 domain data 的作用？
3. YunShan 的 1:1 replay 若按 example 而非 token 实现，长法律文档会造成什么 exposure 偏差？
4. Ultra 的 quality evaluator 与 target model 同 family，怎样检测 evaluator-induced style narrowing？
5. open weights、token total 与 benchmark 表都存在时，数据复现还缺哪些最小 artifact？

## 来源与建议阅读位置

1. [PanGu-α technical report](https://arxiv.org/abs/2104.12369)
   - 为什么读：固定 80TB→1.1TB 漏斗、source 表、过滤/去重、350M PPL loop 与 training selection。
   - 建议位置：§3.1–3.2、Tables 2–4。
2. [MindSpore PanGu-α implementation](https://gitee.com/mindspore/models/tree/master/official/nlp/Pangu_alpha)
   - 为什么读：检查 text→MindRecord preprocessing、tokenizer/config 与训练入口；同时确认代码并未附带完整 1.1TB corpus manifest。
   - 建议位置：README 的数据处理、pretrain/export/eval 命令与 config。
3. [PanGu-Σ technical report](https://arxiv.org/abs/2303.10845)
   - 为什么读：核对 329B/40 domains、四大域 token、数据格式、packing 与 domain-conditioned RRE。
   - 建议位置：§2 RRE、§3 Dataset、Table 1。
4. [PanGu-π paper](https://arxiv.org/abs/2312.17276)
   - 为什么读：固定 1.6T 中英 1:1 general recipe，以及 YunShan finance/legal corpus、1:1 replay 与 995K instructions。
   - 建议位置：§5 Training Data、§6.1–6.3。
5. [Pangu Ultra report](https://arxiv.org/abs/2504.07866)
   - 为什么读：核对 13.2T phase closure、mixture 表、domain-aware tokenizer、quality evaluator/1.6× proxy、context 与 post-training。
   - 建议位置：§2.3、§3.1–3.3、Tables 1–2。
6. [openPangu Embedded-7B V1.1 model card](https://huggingface.co/openpangu/openPangu-Embedded-7B-V1.1)
   - 为什么读：固定 25T、native context、fast/slow 与当前 weight/license boundary。
   - 建议位置：简介、模型架构、license。
7. [openPangu Embedded-1B V1.1 model card](https://huggingface.co/openpangu/openPangu-Embedded-1B-V1.1)
   - 为什么读：固定 10T 与 1B 后训练的 on-policy distillation/multi-source reward 描述。
   - 建议位置：简介、模型架构、技术报告链接。
8. [openPangu-R-7B-Diffusion model card](https://huggingface.co/openpangu/openPangu-R-7B-Diffusion)
   - 为什么读：取得 700B@8K、100B@32K 与 10B/10-epoch SFT 的明确 stage accounting。
   - 建议位置：简介与 training process。
9. [openPangu Ultra-MoE-718B V1.1 model card](https://huggingface.co/openpangu/openPangu-Ultra-MoE-718B-V1.1)
   - 为什么读：确认 718B/39B active、V1.1 post-training变化，以及 pretraining data未披露这一边界。
   - 建议位置：简介、架构、V1.0/V1.1评测说明。
10. [openPangu 2.0 Flash model card](https://huggingface.co/openpangu/openPangu-2.0-Flash)
    - 为什么读：固定截至核查日最新 34T、512K、SFT/RL/OPD 描述，并检查缺失的 source/accounting字段。
    - 建议位置：简介、架构、license。

## 与主线的关系

```text
raw snapshot --processed-by--> versioned filtering and dedup
proxy validation --used-for--> filter and sampling revision
domain ID --implemented-by--> expert routing
domain replay ratio --controls--> forgetting/adaptation trade-off
domain-aware tokenizer data --prerequisite-for--> vocabulary construction
open weights --does-not-imply--> open training corpus
```
