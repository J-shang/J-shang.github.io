---
title: "Amazon Nova / Nova Forge 的数据实践"
description: "按 catalog、stage checkpoint、两层 mixture、RAI 与 managed artifact 边界审计可编排训练控制面。"
topic: "pretraining-data"
section: "international-cases"
slug: "amazon-nova-forge-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 126
readtime: 28
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/amazon-nova-forge.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/international/amazon-nova-forge.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:bbdd04055065ba3061d4893e83c75c3d3c34c772e16a24a35d0a7926193c6801"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`implementation-trace`
> 辅助视角：`method-family/historical trace`，用于 Nova 1→2 checkpoint/stage 演化
> 研究锚点：Nova Micro/Lite/Pro、Nova 2 Lite、Nova Forge CPT/SFT/RFT/data mixing

## 定位

Amazon Nova 的公开材料形成一个少见的分层结构：

- base model 技术报告只披露 source classes、200+ languages、multimodal 与 post-training taxonomy，总 token、mixture、cutoff 仍模糊；
- Nova Forge 不开放 raw corpus 或可下载权重，却让订阅客户选择早期 checkpoint、混入 customer data、调整 Nova catalog category，并保存派生 checkpoint；
- AWS 文档把 data mix、checkpoint plasticity、safety data 与 evaluation 写进可执行 recipe，因此能研究“控制面怎样改变数据 exposure”，但不能复现 Amazon 原始 production run。

这里的 “open training” 是受控环境中的训练入口，不是 open weights、open data 或 D4 reproduction。

## Motivating problem：数据可用、数据可见与数据可下载不是一回事

```text
Amazon proprietary/curated catalog
  -> available to managed training sampler
  -> category proportions configurable
  -> raw examples not disclosed/downloaded

customer S3 data
  -> visible and governed by customer
  -> mixed with Nova catalog inside job
  -> updates proprietary checkpoint in escrow
```

这产生四种不同的审计能力：能否选择 category、能否知道 sample/token ratio、能否检查具体 example、能否导出 resulting weights。Nova Forge 对前两项较强，对后两项仍有限。

## 代际与训练阶段表

| 模型/服务 | 阶段 | 已披露数据与规模 | context/objective | 披露 | 状态 |
|---|---|---|---|---|---|
| Nova 1 Micro/Lite/Pro | `P0` | licensed、proprietary、open-source、publicly available；200+ languages；总量仅称 trillions | Micro text；Lite/Pro text+image/video；context 128K/300K 为产品/评测上限 | `D1/D2` | source/language/modality `verified`；tokens/mix/cutoff `open` |
| Nova 1 | staged checkpoints | text-partial after constant-LR → text-full after text pre/mid → mm-full after multimodal → prod | text/mm exposure 与 post alignment逐层增加 | `D1/D3` | stage semantics `verified`；每段token `open` |
| Nova 1 | `A1/A2` | instruction-demonstration pairs（含multimodal）+ human preference data | iterative SFT + RLHF | `D1/D2` | method/source class `verified`；samples/rollouts/loss `open` |
| Nova 2 Lite | `P0/P2/P3` | 同类 licensed/proprietary/open/public sources；text-RD为trillions；text-CE数量未知 | constant LR + ramp-down → context extension；prod再alignment/safety | `D1/D3` | checkpoints/stage order `verified`；recipe `open` |
| Nova 2 Lite | `A1/A2` | SFT + RLHF（service card）；reasoning/tool/multimodal定量未知 | extended thinking、multimodal；service最高1M，Forge model id 256K | `D1/D3` | behavior/stage `verified`；training context/exposure `open` |
| Nova Forge CPT/mid | customer customization `P2/P3` | customer raw text + proprietary Nova P0 catalog；15 text categories可调 | early/mid/prod checkpoint；recipe可设steps/batch/max length | `D3` control / `D0` examples | sampler/config可审；raw Nova rows/rights `open` |
| Nova Forge SFT | customer `A1` | customer demonstrations + Nova SFT catalog；text 23 categories，另有multimodal catalog | full-rank/LoRA；Nova 2 reasoning/plain、image/video分别成homogeneous dataset | `D3` control | category/sample ratio `verified`；Nova examples/provenance `open` |
| Nova Forge RFT | customer `A2` | messages + reference answer/evaluation criteria；可编程或model judge reward | text-only；adaptive curriculum | `D3` control | schema/config `verified`；rollout/loss accounting需job日志 |
| RAI Toolkit | `A1/A2/eval/runtime` | RAI training category + safety eval + runtime controls | customization后shared responsibility | `D2/D3` | components `verified`；training/eval overlap `open` |

## Assumption ledger

| 维度 | 本篇约束 |
|---|---|
| “trillions” | 范围词，不转成精确数字；text-only与multimodal checkpoint的 token 定义也未统一 |
| model context | Nova 1 128K/300K、Nova 2 service 1M、Forge model id 256K、recipe max_length分开 |
| Forge stage labels | PRE-TRAINED/MID-TRAINED/POST-TRAINED 是产品 checkpoint label，映射到本仓库 taxonomy 时保留原 stage description |
| mixing unit | CPT 文档以 tokens 算；SFT serverless明确以 training samples 算；不可互换 |
| Nova catalog | category name/ratio不等于 raw source、license、cutoff 或 production base-model mixture |
| customer data | Bedrock inference input不用于训练；Forge中显式提交的数据用于客户定制训练，二者不是同一 consent path |
| weights | base/derived weights proprietary；escrow URI 可供后续 AWS job引用，不等于客户拥有可检查的完整权重文件 |
| 省略效应 | checkpoint、learning rate、total exposure、customer ratio、Nova category 与 safety alignment会同时改变 |

## 统一数据字段

| 字段 | Nova production family | Nova Forge customization |
|---|---|---|
| source | licensed、Amazon proprietary、open-source datasets、publicly available；200+ languages；multimodal | customer S3 + proprietary/curated Nova catalogs；category可选，example不可见 |
| rights | source-level license/consent表未公开；Bedrock inference I/O不用于base training | customer负责其data rights；Nova catalog权利仍由AWS控制，逐source未知 |
| cutoff | `unknown` | customer snapshot自定；Nova catalog cutoff `unknown` |
| token accounting | Nova 1/2仅称trillions；unique/sample/loss、modality换算未知 | CPT可由max_length×batch×steps算nominal positions；SFT按sample mixing；padding/loss mask另计 |
| language/modality | 200+ languages，重点15种；Lite/Pro多模态，Micro text | CPT当前customer data仅text；SFT可text/image或text/video，但单dataset modality需同质 |
| quality | curated；生成模型另称EMR/Batch filtering/dedup/enrichment | customer自行prep；Nova category预curated但阈值/calibration不可见 |
| dedup/decontam | Canvas/Reel高层称dedup；Nova understanding model方法/benchmark manifest未知 | customer需自行做跨customer/Nova/benchmark dedup，但无法查看Nova rows造成盲区 |
| order | constant LR→ramp-down→context extension→alignment/safety | base checkpoint + mix config + step schedule；完整global sample order/seed文档不充分 |
| validation | public/internal capability与RAI eval、red teaming | built-in/BYOD/BYOM；SFT 2.0 user validation input有实现限制 |
| artifacts | report/service cards/API；无weights/data/config/logs | SDK、recipes、manifest/logging、checkpoint URI；Nova data/weights仍专有 |

## Nova production recipe：披露到 source class 为止

`[来源事实 | verified]` Nova technical report称 Micro/Lite/Pro先在大量 multilingual/multimodal data上pretrain；来源包括 licensed、proprietary、open-source datasets 和 publicly available data where appropriate。数据覆盖 200+ languages，重点包括 Arabic、Dutch、English、French、German、Hebrew、Hindi、Italian、Japanese、Korean、Portuguese、Russian、Simplified Chinese、Spanish、Turkish。

post-training 迭代使用 instruction-demonstration pairs（含multimodal）做SFT，并从human preference data做RLHF。RAI流程还使用多语言 single/multi-turn demonstrations，按helpfulness/harmfulness studies决定SFT mix。

`[未知 | open]` 报告未给 total token、各语言/模态比例、snapshot/cutoff、source-level rights、filter/dedup retention、train-validation split 或 benchmark decontamination。200+ languages 是 coverage statement，不是低资源语言 exposure 证明。

Nova Canvas/Reel 报告高层说明 pretraining+fine-tuning，并通过 EMR/Batch data filtering、deduplication、enrichment；这只能证明生成模型存在对应 pipeline，不能回填为 Nova Micro/Lite/Pro 的同一实现。

## Forge checkpoint ladder：plasticity 作为产品控制面

### Nova 1 与 Nova 2 的 stage mapping

Nova 1 暴露：

```text
pretraining-text-partial
  = after constant learning-rate text stage, trillions text tokens

pretraining-text-full
  = after all text pretraining/mid-training, trillions text tokens

pretraining-mm-full (Lite/Pro)
  = after multimodal pre/mid-training, trillions tokens

prod
  = after instruction alignment and safety training
```

Nova 2 Lite 暴露 text-RD（constant LR + ramp-down）、text-CE（context extension）与 prod。AWS建议大规模 domain data从较早 checkpoint开始，小规模 customization可从prod开始保留instruction behavior。

`[来源事实 | verified]` 文档一处用 >10B/<10B 作为 early-vs-prod guidance，另一处给1T+ CPT、100B+ mid-training、1B+ structured mid、1K+ SFT examples的选择表。这些是操作建议/量级门槛，不是 Nova 原始训练预算，也不是普遍收敛定律。

`[综合判断 | supported]` 选择 earlier checkpoint 同时改变已有data exposure、learning-rate phase和safety alignment；任何 adaptation 对照都要把起点 checkpoint 视为实验因素，不能只记录 customer corpus。

## CPT data mixing：可计算的 exposure 与一个实验混杂

### 两层 mixture

Forge CPT先决定 customer data 占比 $c$，再决定 Nova 部分内部 category distribution $q_k$：

$$
p_{\text{customer}}=c,\qquad
p_{\text{Nova},k}=(1-c)q_k,\qquad
\sum_k q_k=1.
$$

文档中的 Nova 1 text catalog示例含15类：English entertainment/factual/legal/long-form/mined/other/scientific/social/techQA/finance、code、high-util languages、low-util languages、math、tables，并给一组和为100%的 category recipe。

`[重要边界 | verified]` 这些比例是可用于客户 CPT 的 catalog recipe，不证明它等于 Nova 1 production pretraining mixture。类别也不是source manifest：`en-mined`甚至包含rewritten web content，仍缺 source→rewrite lineage。

### fixed-customer-token 设计改变总计算量

AWS建议固定5B customer tokens，再将customer占比从100%改为90%/70%/50%。若固定customer exposure为$D$，Nova exposure与总量为：

$$
T_{\text{Nova}}=D\frac{1-c}{c},
\qquad
T_{\text{total}}=\frac{D}{c}.
$$

因此 $D=5$B 时：

| customer share $c$ | customer | Nova | total |
|---:|---:|---:|---:|
| 100% | 5B | 0 | 5B |
| 90% | 5B | 0.56B | 5.56B |
| 70% | 5B | 2.14B | 7.14B |
| 50% | 5B | 5B | 10B |

`[推导结论 | verified]` 该设计回答“在不减少customer exposure时，多加多少Nova replay值得成本”；它不单独识别mix ratio效果，因为total tokens/compute同步增加。

若要估计纯mixture effect，还需一组fixed-total对照：

$$
T_{\text{customer}}=cT,\qquad
T_{\text{Nova}}=(1-c)T,
$$

并同时报告in-domain与general per-domain macro、loss trajectory、forgetting与token cost。两种实验回答不同问题，都应保留。

### nominal tokens不等于loss tokens

CPT文档给出：

$$
T_{\text{nominal positions}}
=L_{\max}\times B_{\text{global}}\times N_{\text{steps}}.
$$

$L_{\max}$是configured sequence length，$B$是每step samples。若dynamic padding、packing、masked positions或short documents存在，实际input/loss tokens会小于nominal positions；manifest应另存non-padding与supervised token counts。

## SFT/RFT：sample catalog、reasoning trace 与验证缺口

### SFT 的23类 Nova catalog

Nova 2 text-only SFT serverless 用 `customer_data_percent` 按training samples混合，Nova内部有23类，包括agents、baseline、chat、code、factuality、identity、long-context、math、planning、RAG、RAI、STEM、translation及对应的reasoning类别。默认示例把45% Nova portion给reasoning-instruction-following。

`[来源事实 | verified]` 自定义category时必须给齐23类且和为100%；这是Nova portion内的条件分布，不是overall mix。例如customer 50%、某Nova category 45%，其overall sample share为：

$$
p_{\text{overall}}=0.5\times0.45=22.5\%.
$$

multimodal SFT另有charts/docs/grounding/screenshot/video等catalog。Nova 2 customer dataset必须是text-only、text+image或text+video之一，不能在同一dataset混合image和video；reasoning SFT当前只支持text input。

### user validation data被忽略

`[来源事实 | verified]` Nova 2 SFT文档同时暴露 `validation_data_s3_path` 配置字段，但明确注明当前不支持SFT validation dataset，提供时会被忽略。因而不能把配置存在解释为validation实际进入checkpoint selection。

最小审计应检查：

```text
submitted validation URI
  -> job manifest says consumed or ignored
  -> evaluation run uses frozen external split
  -> checkpoint selection rule and decision count logged
```

否则 `save_top_k`、training metrics 与外部eval可能属于不同decision layer。

### RFT accounting

Nova 2 RFT example以JSONL记录messages与`reference_answer`/evaluation criteria，支持programmatic reward或LLM judge，当前text-only。一个input row可产生多个rollout，因此至少记录：

$$
N_{\text{problems}},
N_{\text{rollouts}},
N_{\text{accepted}},
T_{\text{generated}},
T_{\text{loss}}
$$

而不能把dataset rows当作RL exposure。

## RAI 与 privacy：base service 和 customer training 两条路径

Nova report称会去标识或移除某些personal data（when feasible），并使用public/proprietary动态RAI benchmarks、expert adversarial prompts与red teaming。Forge RAI Toolkit将training category、evaluation benchmark与runtime controls分开，并明确customization可改变safety/fairness，客户需在代表性数据上端到端测试。

`[来源事实 | verified]` Amazon Bedrock的prompts/completions不用于训练Amazon Nova，也不在客户间或与model provider共享。这个承诺适用于managed inference I/O。

Forge customer data是客户主动提交给CPT/SFT/RFT job的training input，显然用于其派生checkpoint；因此：

```text
Bedrock inference I/O --not-used-for--> base model training
Forge submitted dataset --used-for--> customer-specific checkpoint
```

当前政策不能回答Nova base corpus中每个licensed/proprietary/public source的权利细节。

## Artifact 与可复现边界

Forge公开SDK、recipe schema、category names、training/evaluation入口、manifest和部分logging；客户可串联迭代checkpoint。由于Nova weights proprietary，派生checkpoint存于AWS-managed escrow S3，客户通过IAM和URI在后续job中引用，而不是把完整权重复制到自有bucket。

因此披露等级应拆开：

| 层 | 可检查性 |
|---|---|
| customer dataset/version | customer可做到 `D4` |
| mix config/steps/checkpoint lineage | 约 `D3` |
| Nova category labels/ratios | `D2/D3` |
| Nova raw examples/source rights/cutoff | `D0/D1` |
| base/derived full weights | managed access，非open weights |
| original Nova global order/logs | `D0` |

## Validation、污染与区分性检查

| 问题 | 首个可能差异 | 区分性检查 |
|---|---|---|
| Nova replay是否防遗忘 | replay ratio vs total extra tokens | 同时做fixed-customer与fixed-total factorial；general/domain双轴 |
| earlier checkpoint是否更plastic | stage exposure vs learning rate/safety alignment | 同customer data/compute，partial/text-full/prod起点；记录初始与最终loss |
| 某Nova category是否有效 | category内容 vs相邻category重叠 | fixed total，one-category swap；per-source acceptance与domain eval |
| long-context能力来自training吗 | service limit vs context-extension exposure | 固定length buckets记录sample/loss tokens，按length/domain验证 |
| SFT validation是否参与选择 | config field vs backend consumption | 查manifest/logs，外部frozen eval复验所有候选checkpoint |
| customer data是否污染benchmark | customer corpus与Nova catalog都不可完全互查 | customer侧exact/near/semantic scan + time split；Nova侧要求canary/attestation |
| RAI category是否只训练拒答 | data mix、runtime filter、base alignment | raw model/runtime-filter on-off、helpfulness/safety双指标 |

Nova production report有大量release benchmark和RAI evaluation，却没有完整pretraining validation contract。Forge使客户实验更可操作，但customer无法查看Nova raw rows，因而cross-corpus dedup/decontamination存在结构性盲区。

## 可迁移经验与不可外推部分

### 可迁移

- `[综合判断 | supported]` checkpoint stage是domain adaptation的数据变量；必须与customer corpus和learning rate一起记录。
- `[综合判断 | supported]` 两层mix应保存overall customer share与Nova内部conditional distribution，避免把45%误读成overall 45%。
- `[综合判断 | supported]` fixed-customer-token与fixed-total实验回答不同问题；前者评估追加replay的成本收益，后者更接近pure mix ablation。
- `[综合判断 | supported]` managed proprietary catalog也可以提供category-level控制，但不能替代source/provenance/rights审计。
- `[综合判断 | supported]` training-time RAI data、evaluation与runtime filter必须分层，防止把filter效果归因给model weights。

### 不可外推

- `[未知 | open]` Nova 1/2精确pretraining tokens、mixture、cutoff、source-level rights、dedup/decontam与loss accounting；
- `[未知 | open]` Forge catalog recipe与production Nova原始mixture的关系；
- `[未知 | open]` Nova 2 context-extension阶段token/length distribution，以及1M service context的训练证据；
- `[未知 | open]` proprietary Nova catalog与customer validation/benchmark的跨库污染；
- `[待验证假设 | plausible]` 文档中的>10B、1T+/100B+/1B+ checkpoint guidance可跨domain和model size稳定迁移。

## 掌握标准

读完后应能：

1. 区分Nova source disclosure、Forge category control、raw data visibility与weight export；
2. 复原Nova 1/2 checkpoint ladder，并说明early checkpoint为何不是单纯“少训一点”；
3. 写出两层mixture公式，正确计算overall category share；
4. 解释AWS 5B示例为何同时改变mix和total compute；
5. 说明Bedrock inference privacy承诺为何不能替代Forge training-data lineage。

## 推理型自测

1. 90% customer mix优于100%，怎样判断收益来自0.56B replay还是总token增加？
2. 客户看不到Nova examples时，怎样最低限度审计benchmark contamination？
3. 用pre-RAI checkpoint做CPT后，为什么runtime filter通过不能证明weights仍安全？
4. SFT job接收validation URI但backend忽略，应该怎样重建checkpoint selection protocol？
5. customer sample平均长8K、Nova sample平均长1K时，50/50 sample mix对应多少token mix？还缺什么统计？

## 来源与建议阅读位置

1. [Amazon Nova family technical report](https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card)
   - 为什么读：固定source classes、200+ languages、multimodal/SFT/RLHF、RAI data与系统artifact边界。
   - 建议位置：§1 family overview、§2.3 long context、§5 responsible development、§6 system。
2. [Nova Forge CPT and mid-training guide](https://docs.aws.amazon.com/nova/latest/userguide/nova-forge-cpt.html)
   - 为什么读：最关键的stage checkpoint、15类catalog、mix公式、5B实验设计与nominal token accounting。
   - 建议位置：intermediate checkpoints、finding optimal ratio、data mixing categories、parameter guide。
3. [Nova Forge SFT guide](https://docs.aws.amazon.com/nova/latest/userguide/nova-forge-sft.html)
   - 为什么读：核对SFT/mid/CPT数据量级建议、text/mm catalogs、RAI Toolkit与shared responsibility。
   - 建议位置：training stages/approaches、data mixing、RAI Toolkit、evaluation methods。
4. [Nova 2 SFT guide](https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-sft-2-fine-tune.html)
   - 为什么读：检查reasoning/multimodal schema、max_length与validation-data ignored限制。
   - 建议位置：reasoning mode、data upload、training configuration、hyperparameter guidance。
5. [SageMaker SFT data mixing](https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-forge-sft-datamix-smtj.html)
   - 为什么读：固定`customer_data_percent`按sample解释、23类条件分布及默认recipe。
   - 建议位置：data mixing parameter、default category distribution、limitations。
6. [Nova 2 Lite AI Service Card](https://docs.aws.amazon.com/ai/responsible-ai/nova-2-lite/overview.html)
   - 为什么读：确认Nova 2 source classes、SFT/RLHF、managed service boundary与current release date。
   - 建议位置：Design、Controllability、Performance Expectations。
7. [Nova Forge Responsible AI Toolkit](https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-responsible-ai-toolkit.html)
   - 为什么读：分开training data、evaluation与runtime control，并明确customization后的shared responsibility。
   - 建议位置：Responsible AI toolkit全文。
8. [Nova responsible-use/privacy guide](https://docs.aws.amazon.com/nova/latest/nova2-userguide/responsible-use.html)
   - 为什么读：固定Bedrock inference I/O不用于训练，避免与Forge显式training data混淆。
   - 建议位置：Privacy。
9. [Nova iterative training](https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-iterative-training.html)
   - 为什么读：核对派生checkpoint manifest、escrow S3、SFT→RFT lineage与rollback。
   - 建议位置：checkpoint location/access、iterative workflow、limitations。
10. [Nova Forge SDK](https://github.com/aws/nova-forge-sdk)
    - 为什么读：检查可执行的dataset transform、recipe、training/eval orchestration和公开代码边界。
    - 建议位置：dataset loader、trainer/config enums、examples与license。

## 与主线的关系

```text
checkpoint stage --controls--> adaptation plasticity and inherited exposure
customer share + Nova conditional mix --implements--> two-level sampler
nominal positions --upper-bounds--> non-padding and loss tokens
managed catalog access --does-not-imply--> source-level data transparency
RAI training data --distinct-from--> RAI evaluation --distinct-from--> runtime filter
Bedrock inference I/O --not-used-for--> Nova base training
```
