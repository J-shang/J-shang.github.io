---
title: "Tencent Hunyuan / Hy 的数据实践"
description: "核算 synthetic pretraining、阶段 token、长上下文、多模态 pairs 与开放权重资产的证据边界。"
topic: "pretraining-data"
section: "china-cases"
slug: "tencent-hunyuan-hy-data-practices"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 108
readtime: 26
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/tencent-hunyuan-hy.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/china/tencent-hunyuan-hy.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:814211df4dd08693f6c1bfbd2b30347cd3e4e1a2968003f2e86d3cfbdc3d6f23"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`method-family/historical trace` -->
<!-- maintenance: secondary-view=`implementation-trace`，用于 TurboS/A13B 数据管线与 HunyuanOCR 多模态阶段 -->
> 主要模型/资料：初代混元、Hunyuan-Large、Hunyuan-TurboS、Hunyuan-A13B、HunyuanOCR、Hy3-preview

## 这篇案例研究什么

腾讯混元的公开谱系提供了三种不同强度的证据：

1. **Hunyuan-Large** 首次把大规模 synthetic instruction data 放进 pretraining 并公开7T/1.5T、四步生成和长上下文阶段；
2. **TurboS 与 A13B** 公开更完整的 preprocessing→model extraction→semantic dedup、annealing、long-context和RL数据对象；
3. **HunyuanOCR 与 Hy3** 分别代表多模态专用模型的高粒度recipe和最新通用模型的低粒度disclosure boundary。

开放的是权重、推理/微调代码和报告；没有开放原始corpus、source manifest、训练顺序、完整token mask或数据权利清单。因此多数代际是`D2/D3`，不是`D4`。

## 核心问题：同一个“trained tokens”可能漏掉后续阶段

如果只抄模型摘要，会得到7T、16T、20T、454B四个数，但至少有三种统计结构：

- Hunyuan-Large：7T headline，另列32K与256K各约10B，二者是否已含在7T内未完全明确；
- TurboS：16T base，之后300B annealing与30B+20B context extension，阶段和为16.35T；
- A13B：20T foundation，之后300B annealing，32K/256K长上下文token未披露；
- HunyuanOCR：约200M image-text pairs，经四阶段产生50B+300B+80B+24B=454B sampled tokens。

因此 headline corpus、sampled stage exposure、pair count 与 loss tokens必须分列。

## 各代模型和 training stage

| 模型 | 阶段 | 已披露数据与规模 | mixture / context | 披露 | 状态 |
|---|---|---|---|---|---|
| 初代腾讯混元（2023） | `P0` | 官方发布稿称超过2T pretraining tokens | source/mix/unique/loss未知 | `D1` | scale claim `verified`；recipe `open` |
| Hunyuan-Large | `P0` | 7T，其中近1.5T synthetic；中英为主 | 质量/教育价值/毒性过滤；128K vocab | `D2/D3` | total/synthetic `verified` |
| Hunyuan-Large | anneal + `P3` | 最后5%高质量anneal；32K和256K各约10B long-context corpus | long:normal≈25:75；long只用natural books/code | `D2/D3` | stage facts `verified`；7T inclusion `open` |
| Hunyuan-Large | `A1/A2` | >1M SFT，来自>10M instruction pool；之后offline+online DPO | math/code/agent/long等；3 epochs | `D2/D3` | counts/pipeline `verified` |
| Hunyuan-TurboS | foundation `P0` | 16T | 4K；topic/quality/diversity mixture models | `D2` | budget/pipeline `verified` |
| Hunyuan-TurboS | annealing `P2` | 300B | 4K；pretrain/code/math/STEM/instruction/LongCoT/synthetic/web | `D2` | budget/mix classes `verified` |
| Hunyuan-TurboS | context `P3` | 30B@32K + 20B@256K | short:long=3:1；curated natural long documents | `D2` | budget/context `verified` |
| Hunyuan-TurboS | `A1/A2/A3` | 3M SFT；reasoning GRPO 300K；general GRPO 160K并回放10% reasoning | adaptive long/short CoT、deliberation、two-stage GRPO | `D2` | sample counts `verified`；rollout/loss `open` |
| Hunyuan-A13B | foundation `P0` | 20T；含250B high-quality STEM corpus | 4K；13.5T cosine decay后保持最低LR | `D2/D3` | budget/STEM `verified` |
| Hunyuan-A13B | annealing/context | 300B@8K；之后32K→256K，token未知 | refined knowledge labels + multidimensional difficulty | `D2/D3` | anneal `verified`；context exposure `open` |
| Hunyuan-A13B | `A1/A2/A3` | reasoning RL 150K，math:code:logic:science=2:2:1:1；10%与SFT重叠 | 24K→32K RL；16 subtopics、30+ scoring services | `D2/D3` | prompt counts/mix `verified`；rollout tokens `open` |
| HunyuanOCR 1B | multimodal `P0/P1/P3/A1` | >200M image-text pairs；四阶段50B+300B+80B+24B=454B | 8K→32K；text≤10%于前两段 | `D2/D3` | pair/stage counts `verified` |
| HunyuanOCR | `A2` | open/synthetic OCR tasks；prompt/rollout数未知 | task-specific rewards、pass-rate与reward-variance filtering | `D2/D3` | mechanism `verified`；accounting `open` |
| Hy3-preview | `P*/A*` | base/instruct weights；训练数据token/source/mix未披露 | 295B/21B active；256K；reasoning/code/agent | `D1/D3` | model/artifacts `verified`；recipe `open` |

## 阅读这些结论前先确认的前提

| 维度 | 本篇约束 |
|---|---|
| 研究对象 | text base、annealing、context extension、SFT/RL和OCR多模态分开；不假设各代共享同一corpus |
| token单位 | 报告中的trained/processed tokens按sampled exposure；unique、non-padding、loss tokens无明示即`unknown` |
| 7T口径 | Hunyuan-Large的2×10B长上下文阶段与7T是否包含关系记`open` |
| 16T口径 | TurboS明确写foundation完成后再anneal/context，因此阶段和16.35T是`implementation relation` |
| 20T口径 | A13B 20T foundation后有300B anneal；两段long-context exposure未知，不能写精确总量 |
| multimodal | HunyuanOCR的pair、token、image resolution、language与loss mask是不同字段 |
| rights | public/open-source、procurement/vendor-sourced、web crawl、copyrighted与licensed逐词保留；模型license不外推给语料 |
| cutoff | 通用pretraining source cutoff未公开；benchmark和发布日不替代cutoff |
| 省略效应 | 架构、active参数、tokenizer、mixture、context、LR和post-training跨代同时变化 |

## 厂商公开了哪些 data fields

| 字段 | Large | TurboS / A13B | HunyuanOCR | Hy3 |
|---|---|---|---|---|
| source | web/QA/code repos/books作为synthetic seeds；natural corpus细目不足 | varied-format raw sources；STEM/code；post含public/licensed/vendor/open/synthetic | public benchmarks、web crawl、real-world、proprietary synthesis | `unknown` |
| rights | source-level rights mix `unknown` | SFT明确出现public/licensed/procurement；P0 rights未闭合 | >130 languages；pair-level rights/consent未发布 | weights为Tencent Hy license；training rights未知 |
| cutoff | `unknown` | `unknown` | `unknown` | `unknown` |
| token accounting | 7T、1.5T synthetic；2×10B inclusion open | TurboS 16.35T stage sum；A13B≥20.3T known exposure | 454B stage sum、>200M pairs | `unknown` |
| quality | writing/educational/toxicity、privacy anonymization、category labels | URL/block/heuristic/model critique、difficulty/knowledge labels | LLM judge、hard retrieval、cross-model consistency、人审 | `unknown` |
| dedup | synthetic self-consistency；natural dedup方法不足 | URL dedup + global semantic dedup；阈值/retention未知 | 完整exact/near/semantic阈值未知 | `unknown` |
| mixture/order | synthetic约21.4%；末5% anneal；32K→256K | foundation→anneal→32K→256K；A13B long token未知 | alignment→joint→long→application SFT→RL | stage/order未知 |
| validation | MoE scaling、base/post、RULER/LV/PenguinScrolls | pretrain/long/reasoning/agent eval；部分ablation | public+internal OCR benchmark、task reward | public/internal capability；pretrain validation contract未知 |
| artifacts | base/instruct weights、code/report | A13B weights/code/report；TurboS report，生产模型资产边界较弱 | weights、inference、report；部分benchmarks | base/instruct weights、train/inference code；无corpus |

## Hunyuan-Large：synthetic instruction data进入 pretraining

### 7T中近1.5T synthetic

`[来源事实 | verified]` Hunyuan-Large在7T tokens上预训练，其中近1.5T为synthetic data，约占headline exposure的：

$$
r_{\text{syn}}=\frac{1.5\text{T}}{7\text{T}}\approx21.4\%.
$$

变量单位是报告中的training tokens；1.5T是近似值，不能解释为unique instructions，也不能推断其loss mask与natural corpus相同。

synthetic pipeline是：

```text
web / web-QA / code repositories / books as seeds
  -> instruction generation
  -> clarity, information, low-resource expansion, difficulty evolution
  -> multiple specialized models generate responses
  -> critique model + multi-answer self-consistency filtering
```

主要目标是math、code、low-resource和high-educational-value domains。`[综合判断 | supported]` 这批数据虽然呈instruction-response格式，却属于P0；不能因格式像SFT就把1.5T移到post-training。

### annealing 与 long-context accounting

`[来源事实 | verified]` 训练最后5%进入短annealing并优先使用最高质量数据。随后另做32K→256K long-context pretraining，每段约10B；long-context corpus约25%为natural books/code长文档，约75%为normal-length pretraining data。

`[未知 | open]` 报告同时写“trained tokens 7T”和“after annealing”进行2×10B长训，未明说7T是否含两段。因此保留：

$$
T_{\text{headline}}=7\text{T},\qquad
T_{\text{long,stated}}=10\text{B}+10\text{B},
$$

不擅自把总量改为7.02T。

### scaling proxy 的边界

`[来源事实 | verified]` MoE scaling实验使用10M–1B active-parameter模型和10B–100B token范围，拟合得到约58.1B optimal active parameters与5.6T optimal trained tokens；最终选择52B active和7T，因为最优附近曲线较平且希望利用更多数据。

`[综合判断 | supported]` 这是同一实验族内的compute proxy，不是7T全配方的严格最优证明；最终模型还包含1.5T synthetic、annealing、long context和架构策略。

### 10M instruction pool不等于10M SFT samples

`[来源事实 | verified]` SFT通过public web/encyclopedia extraction与instruction generalization积累>10M instructions，再做taxonomy balancing、rule/model/human filtering，最终使用>1M高质量样本训练3 epochs。model filter基于70B critique model做四档质量评分。DPO又混合预编译preference与当前policy生成的online responses。

必须保存candidate instructions、accepted pairs、epochs、policy responses、preference pairs和loss tokens六套计量。

## TurboS：公开到 stage-sum 的数据recipe

### preprocessing→extraction→post-processing

`[来源事实 | verified]` TurboS pipeline为：URL-level dedup/filter→block pruning→topic classification→content extraction model→STEM/code specialist extraction→heuristic filtering→global semantic dedup。critique models与data-mixture models又以数十个domain labels控制阶段配比。

`[综合判断 | supported]` semantic dedup发生在model-based extraction之后，这意味着extractor版本会改变dedup输入空间；只固定semantic threshold不能复现retention。

### 16.35T 已知阶段和

设各段sampled tokens为：

$$
\begin{aligned}
T_{\text{foundation}}&=16\text{T},\\
T_{\text{anneal}}&=300\text{B},\\
T_{\text{context}}&=30\text{B}+20\text{B},\\
T_{\text{known stages}}&=16.35\text{T}.
\end{aligned}
$$

报告明确以“following completion of pre-training”引出annealing，再以final phase引出context，因此该加法是`implementation relation`。它仍不是unique或loss tokens。

anneal mixture含high-quality pretraining、code、math、STEM、instruction/LongCoT、synthetic和适量web。报告还说明：

- instruction data前移到anneal，以减少后续SFT需求；
- QA-style data选择性施加target loss，防止模型把开放续写变成“出题”；
- 被4K截断的LongCoT延后到long-context阶段；
- context阶段32K用30B、256K用20B，short:long=3:1。

`[综合判断 | supported]` “instruction data前移”表明P2/A1边界不能按文件格式判断，应以训练目标、loss mask与阶段位置标注。

### post-training的分母

`[来源事实 | verified]` TurboS使用3M SFT samples；reasoning GRPO用300K，code:math:logic+science=2:2:1；general GRPO使用160K instructions并保留10% Stage I reasoning replay。它还使用adaptive long/short CoT teacher与multi-round deliberation发现能力缺口。

这些是sample/prompt counts，不是rollout或loss tokens。dynamic sampling会丢弃zero-advantage样本，因此nominal prompts与实际更新分母不同。

## A13B：20T foundation、250B STEM 与 agent data factory

`[来源事实 | verified]` A13B复用并改进TurboS pipeline，获得250B high-quality STEM pretraining corpus，并建立refined knowledge labels和multi-dimensional difficulty grading。foundation处理20T@4K，annealing处理300B@8K，之后32K→256K但未给token。

因此只能写：

$$
T_{\text{known}}=20\text{T}+300\text{B}=20.3\text{T},
\qquad T_{32K},T_{256K}=\text{unknown}.
$$

250B STEM是20T中的incorporated corpus，不是额外相加项。

`[来源事实 | verified]` reasoning RL用150K prompts，math:code:logic:science=2:2:1:1，10%与SFT重叠、90%为novel cases；排除multiple-choice、true/false和proof problems，24K→32K训练。all-scenarios数据扩展到16个subtopics和30+ scoring services。

agent SFT engine包含user、planner、tool、agent、checker五个角色，工具响应来自sandbox、MCP与synthetic tools；超过30类system instructions与20,000种tool/action/response format combinations用于泛化。

`[综合判断 | supported]` “20,000 formats”是format transformations，不是20,000 independent tasks；需要保留base task→format variant→trajectory→reward lineage。

## HunyuanOCR：为什么 pair 数和 token 数必须分开统计

### 200M pairs、130+ languages与九类场景

`[来源事实 | verified]` HunyuanOCR从public benchmarks、web-crawled real-world data和proprietary synthesis构建>200M image-text pairs，覆盖130+ languages和九类场景：street view、document、advertisement、handwriting、screenshot、card/certificate/invoice、game UI、video frame与artistic typography。

synthetic engine支持LTR/RTL、cursive scripts、font/color/orientation、lighting/shadow、fold/perspective、blur/noise/compression。QA pipeline按single-source-multiple-uses复用spotting/parsing数据，经过hard retrieval、VLM generation、multi-model cross-validation，失败难例部分转人工复核。

### 四阶段共454B sampled tokens

$$
T_{\text{OCR}}=50\text{B}_{8K}+300\text{B}_{8K}+80\text{B}_{32K}+24\text{B}_{32K}=454\text{B}.
$$

| 阶段 | 可训练参数 | 数据重点 | tokens/context |
|---|---|---|---|
| alignment | ViT + adapter | caption、synthetic parse/recognition、≤10% text | 50B@8K |
| multimodal | all | synthetic spotting/parsing/translation/VQA、≤10% text | 300B@8K |
| long context | all | long text、real auto-label、long docs、IE | 80B@32K |
| application SFT/anneal | all | human real-world、hard negatives、standard instructions | 24B@32K |

这里454B是sampled token exposure，>200M是pair pool；同一pair可跨任务复用并多次采样。`[未知 | open]` image token折算、repeat、loss mask与各语言/场景exposure没有完整manifest。

RL按task设计spotting/edit-distance+IoU等reward，依据output diversity、reward variance与pass rate去掉过易/不可解样本。该机制适合可验证OCR，不自动迁移到开放式视觉问答。

## Hy3-preview：开放训练代码不等于公开 pretraining recipe

`[来源事实 | verified]` Hy3-preview公开base/instruct权重和full/LoRA fine-tuning pipeline，295B总参数、21B active、256K context，并称在重建训练基础设施上提升reasoning、context learning、code与agent能力。

`[未知 | open]` 当前README没有给pretraining token、source、rights、cutoff、mixture、dedup、anneal/context exposure或RL task/trajectory规模。训练代码用于下游fine-tuning，不是原始pretraining reproduction package。

`[综合判断 | supported]` 最新模型可以在artifact字段为`D3`、在 data recipe 字段仍为`D1`；不能继承A13B的20T、TurboS的16.35T或Hunyuan-Large的21.4% synthetic。

## 如何验证这些结论，并检查 contamination

| 问题 | 首个可能差异 | 区分性检查 |
|---|---|---|
| 1.5T synthetic是否带来增益 | replacement、teacher、quality、domain mix | 固定7T、order、seed，对synthetic share/teacher/filter做正交ablation |
| TurboS 16T和16.35T如何报 | headline foundation vs later stages | 发布stage manifest、repeat和loss tokens |
| long context来自真实结构还是位置扩展 | natural docs、short replay、RoPE同时变化 | 固定token/LR，正交比较random concat、books/code和真实长文档 |
| 250B STEM是否改善推理 | acquisition/filter/labels与总mix耦合 | fixed-budget replacement，报告STEM与non-STEM macro/per-domain |
| OCR synthetic是否覆盖真实缺陷 | rendering support vs deployment distribution | 按场景/语言/缺陷做real-only held-out和synthetic-to-real gap |
| agent格式泛化是否等于任务泛化 | 20K formats vs独立task semantics | 按base task group split，不让format variants跨train/test |

公开材料给出大量release benchmark、long-context与部分ablation，但没有同时闭合validation source、时间边界、split unit、tokenizer、per-domain sample/token、decision exposure与semantic contamination。PenguinScrolls等内部集能补场景覆盖，不能替代可审计manifest。

## 哪些经验可以借鉴，哪些不能直接照搬

### 可迁移

- `[综合判断 | supported]` synthetic instruction data可位于P0、P2或A1；阶段应由训练位置和目标定义，不由文本格式定义。
- `[综合判断 | supported]` extractor precedes semantic dedup时，必须版本化extractor与dedup embedding/threshold。
- `[综合判断 | supported]` long-context recipe至少记录真实长文档来源、short replay ratio、每个窗口token和截断/延后规则。
- `[综合判断 | supported]` 多模态任务复用要同时报告unique pairs、task instances、sampled tokens和loss tokens。
- `[综合判断 | supported]` capability artifacts、training code与pretraining data openness应逐字段评分。

### 不可外推

- `[未知 | open]` Large、TurboS、A13B、HunyuanOCR与Hy3共享哪些具体source、snapshot、filter模型和tokenizer版本；
- `[未知 | open]` 通用模型的source rights、cutoff、language/domain mixture与unique/loss token；
- `[未知 | open]` Hunyuan-Large的2×10B是否含在7T，A13B长上下文阶段token多少；
- `[待验证假设 | plausible]` TurboS的semantic dedup与mixture model对low-resource languages无系统性损失；
- `[待验证假设 | plausible]` HunyuanOCR的proprietary synthesis可覆盖真实部署中的复杂缺陷组合。

## 读完后应该掌握什么

读完后应能：

1. 解释7T/1.5T、16T/16.35T、20T/20.3T+unknown与454B/200M分别是什么口径；
2. 复述TurboS从URL dedup到global semantic dedup的顺序，并说明extractor为何是versioned dependency；
3. 说明instruction-shaped data为什么可能属于pretraining或annealing；
4. 把A13B的150K RL prompts、20K format combinations、rollouts与loss tokens分开；
5. 解释Hy3开放base权重和fine-tuning代码为何仍不能复现pretraining数据。

## 用这些问题检查自己

1. 若TurboS摘要只写16T，comparison matrix是否应写16T还是16.35T？怎样同时保留来源事实与推导？
2. Hunyuan-Large含21.4% synthetic且表现更强。为什么这不构成synthetic share的因果最优点？
3. A13B把250B STEM加入20T。若math提升而general下降，应检查哪些replacement与macro指标？
4. HunyuanOCR同一image生成spotting、parsing和VQA。如何构建split避免跨任务image leakage？

## 来源与建议阅读位置

1. [腾讯混元初代官方发布稿](https://www.tencent.com/en-us/articles/2201685.html)
   - 为什么读：固定2023年“超过2T pretraining tokens”的历史起点。
   - 建议位置：模型规模与pre-training data段落。
2. [Hunyuan-Large Technical Report](https://arxiv.org/abs/2411.02265)
   - 为什么读：固定7T/1.5T synthetic、四步生成、MoE scaling、2×10B long context与SFT pipeline。
   - 建议位置：§2.1、§2.3、§3.1–3.2。
3. [Hunyuan-Large repository](https://github.com/Tencent-Hunyuan/Tencent-Hunyuan-Large)
   - 为什么读：核对base/instruct权重、推理代码、license与artifact边界。
   - 建议位置：README、model links、license、inference。
4. [Hunyuan-TurboS Technical Report](https://arxiv.org/abs/2505.15431)
   - 为什么读：复核16T+300B+30B+20B、data pipeline、3M SFT与两阶段GRPO。
   - 建议位置：§2.1、§2.3–2.4、§3.1、§3.5。
5. [Hunyuan-A13B repository and report](https://github.com/Tencent-Hunyuan/Hunyuan-A13B)
   - 为什么读：固定20T、250B STEM、300B anneal、150K reasoning RL和agent data engine。
   - 建议位置：report §2.1–2.3、§3.1–3.2；README model/releases/license。
6. [Hunyuan-A13B-Pretrain model card](https://huggingface.co/tencent/Hunyuan-A13B-Pretrain)
   - 为什么读：核对开放base checkpoint、config/tokenizer与模型license。
   - 建议位置：Model Introduction、Files、License。
7. [HunyuanOCR Technical Report](https://arxiv.org/abs/2511.19575)
   - 为什么读：学习>200M pairs、130+ languages、454B四阶段tokens、cross-task reuse与task-specific RL reward。
   - 建议位置：§4 Data Construction、§5 Training Recipe、evaluation split说明。
8. [HunyuanOCR repository](https://github.com/Tencent-Hunyuan/HunyuanOCR)
   - 为什么读：核对权重、推理代码和后续开放benchmark边界。
   - 建议位置：README、model download、technical report、benchmarks。
9. [Hy3-preview repository](https://github.com/Tencent-Hunyuan/Hy3-preview)
   - 为什么读：固定最新base/instruct开放资产及quantitative data recipe 缺口。
   - 建议位置：Model Introduction、Training、License、base benchmark。

## 这篇案例与主线知识的关系

```text
stage accounting --prerequisite-for--> 16T vs 16.35T interpretation
synthetic provenance --prerequisite-for--> 1.5T pretraining audit
extractor version --affects-input-to--> semantic dedup
long-document provenance --prerequisite-for--> context attribution
pair accounting --not-equivalent-to--> multimodal loss tokens
open fine-tuning code --does-not-imply--> reproducible pretraining corpus
```
