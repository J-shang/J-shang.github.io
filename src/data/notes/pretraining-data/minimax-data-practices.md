---
title: "MiniMax 模型家族的数据实践"
description: "审计数据实验功效、重复曝光、长上下文、多模态阶段与 agent-data factory 的不同统计单位。"
topic: "pretraining-data"
section: "china-cases"
slug: "minimax-data-practices"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 103
readtime: 37
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/minimax.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/china/minimax.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:76123360107a1de4009870bd5ab784a1857746bd466ff860adb76fa0c6f2cc17"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`method-family/historical trace` -->
> 主要模型/资料：MiniMax-Text-01、MiniMax-VL-01、MiniMax-M1、MiniMax-M2/M2.5/M2.7、MiniMax-M3。

## 这篇案例要回答什么

MiniMax-01 的报告公开了少见的 data-experiment protocol：用 1B activated / 8B total 的 MoE、40B tokens、sample-level metric distribution、seed variance、minimal detectable effect（MDE）和 power analysis 筛 data recipe。它还给出 quality reward labeler、nested document format、重复次数实验、358B 长上下文 curriculum，以及 512B multimodal training 的逐阶段 token 数。M1/M2 又把数据问题推进到 natural QA、可验证 reasoning、GitHub environment、agent trajectories、artifact-aligned reward 与 online reward-bias monitoring。

因此本案例围绕四个问题：

1. 小模型 data proxy 怎样成为有统计门槛的决策系统，而不是单次 benchmark 排名；
2. “低质量不全删”和“高质量可重复”如何在固定 exposure 下验证；
3. context length curriculum 怎样同时控制 length、source/domain mixture 与短域遗忘；
4. 从 problem 到 executable workspace、trajectory、artifact 和 reward 的 agent-data provenance 怎样表示。

MiniMax-M3 截至核查日只公开 native multimodal from first step、1M context 与 sparse-attention 研究锚点，没有与 MiniMax-01/M2 同等的 corpus report；本笔记只记 `D1/D3`，不从模型能力反推配方。

## 各代模型和 training stage

| 模型 | 阶段 | 已披露规模 | 数据 / context 机制 | 披露 |
|---|---|---:|---|---|
| Text-01 | architecture/data proxies | 最多 300B/模型；另有 1T 与 500B module ablations | 70M–7B architecture scaling；small MoE data experiments | `D2` |
| Text-01 | `P0` | 500-step warmup；7.2T constant + 3.2T reduced-LR + 1T fast decay；总量需 step log 才能精确闭合 | web/books/academic/code；rule clean/dedup + 60B/5B-active reward labeler；8K | `D2/D3` |
| Text-01 | `P3` 128K | 300B | 30% short(<32K)、70% medium(32K–128K)、0% long；RoPE 5M | `D2/D3` |
| Text-01 | `P3` 512K | 32B | 35% short、35% medium、30% long(>128K)；RoPE 10M | `D2/D3` |
| Text-01 | `P3` 1M | 26B | 30% short、30% medium、40% long；RoPE 10M；推理 extrapolate 4M | `D2/D3` |
| Text-01 | `A1/A2` | millions prompts；精确 tokens `unknown` | 8K SFT → 1,032,192 SFT（50% long prompts）→ 8K DPO → 1,032,192 DPO（all long）→ 8K online RL | `D2/D3` |
| VL-01 vision encoder | vision pretraining | 694M unique pairs；其中180M有 refined captions；实际 exposure 37B pairs@224 + 1.2B pairs@336 | raw/refined caption 0.5/0.5；caption截断76 tokens | `D2/D3` |
| VL-01 | modality alignment | 80B multimodal tokens | 100M image-description pool；只更新 ViT+adapter；336×336 | `D2/D3` |
| VL-01 | vision instruction tuning | 420B multimodal tokens | multimodal instruction : Text-01 post-training data = 20:1；全参数更新 | `D2/D3` |
| VL-01 | user-experience tuning | 44.8B multimodal tokens，1 epoch | common-app images + human-labeled multi-turn conversations；独立 human test set 引导构造 | `D2/D3` |
| VL-01 | preference | 40K image-text preference pairs + pure-text pairs | DPO；temperature/image weakening/error injection；LLM evaluator；不足1 epoch early stop | `D2/D3` |
| M1 | `P1/P3` | Text-01 + 7.5T | natural QA，无 synthetic；STEM/code/books/reasoning 70%；32K→1M 四阶段 | `D2/D3` |
| M1 | `A1` | samples/tokens `unknown` | long-CoT cold start；math+code约60% | `D2/D3` |
| M1-40K | `A2` | math ~50K、logic ~53K、competitive code 30K、SWE several thousand、general 25K | rule/exec/GenRM rewards；先 reasoning 后混 general | `D2/D3` |
| M1-80K | `A2` continued RL | trajectories/tokens `unknown` | 40K→48K→56K→64K→72K→80K；40K model做difficulty filter；downsample synthetic reasoning | `D2/D3` |
| M2 base | `P0` constant | 19.9T | web/papers/books/code/structured QA；model score + auxiliary classifiers；upweight code/math/STEM | `D2/D3` |
| M2 base | `P2/P3` decay | 9.3T | 8K→32K→192K；short decay + code concatenation + natural long PDF + thematic packing | `D2/D3` |
| M2→M2.7 | `A1/A2/A3` | task/trajectory/token scale `unknown` | SWE/AppDev/terminal + deep search/workspace/finance/office + reasoning/conversation；Forge agent RL | `D2/D3` |
| M3 | native multimodal `P0…A3` | total/active ~428B/~23B；training tokens `unknown` | text/image/video mixed from first step；1M context；source/mix/filter `unknown` | `D1/D3` |

### Text-01 base token 口径

论文没有给一个单独的 final total row，而是给出 learning-rate segments。保守表示为：

$$
T_{reported\ segments}=7.2\text{T}+3.2\text{T}+1.0\text{T}=11.4\text{T}
$$

另有 500 iterations warmup。若 warmup 不含在 7.2T 内，其 token 数还取决于当时 global batch schedule；报告又在 69B、790B、4.7T 改变 batch。没有 step log，不能把 11.4T 自动写成精确 final sampled tokens。之后长上下文三阶段明确为：

$$
T_{long}=300\text{B}+32\text{B}+26\text{B}=358\text{B}
$$

二者是不同阶段，且均不等于 unique corpus 或 loss tokens。

## 阅读这些结论前先确认的前提

| 项 | 本笔记的处理 |
|---|---|
| 研究对象 | Text-01 base、long-context extension、alignment、VL、M1 continued pretraining/RL、M2/M3 分开 |
| token 单位 | paper 的 trained tokens 记 sampled exposure；unique/non-padding/loss tokens 通常 `unknown` |
| pair 与 token | VL encoder 的 image-caption pairs 与 VL-LLM 的 multimodal tokens 分列；37B pair exposures 不换算成 tokens |
| epoch | Text-01 repetition experiment 的 epoch 以 global-dedup 后 pool 为参照；不等于 source snapshot 的自然重复 |
| model-mediated roles | reward labeler、caption refiner、QA/test generator、difficulty solver、GenRM、agent verifier 分开记录 |
| long context | training max、input max、output-thinking budget 分开；M1 的 40K/80K 是生成上限，不是 pretraining input length |
| cutoff | Text/VL/M1/M2/M3 的全局 data cutoff 均 `unknown`；M2 permissive-license GitHub source 无 snapshot date |
| lineage | M1 明确 continued from Text-01；M2/M3 是否继承具体 Text-01 data/checkpoint 未按默认假设填充 |
| 省略效应 | architecture、MoE routing、attention、optimizer、precision、data 与 alignment 同时变化 |

## 厂商公开了哪些 data fields

| 字段 | Text-01 / VL-01 | M1 | M2 / M3 | 置信 |
|---|---|---|---|---|
| source | academic、books、web、code；VL internet captions、Common Crawl-like images、apps、instructions | web/PDF、forums、textbooks、public math/OJ/GitHub、synthetic logic、general prompts | web/papers/books/code/structured QA；permissive GitHub PR；professional/workspace/tool data；M3 source `unknown` | 已列类别 `verified`；清单 `open` |
| rights | source-level rights `unknown`；model license需按card | public source不等于统一许可 | M2 SWE 明确 permissive-license repos；其余 rights/consent `unknown` | 部分 `verified`；总体 `open` |
| cutoff | `unknown` | `unknown` | `unknown` | `open` |
| scale | base segments 11.4T + warmup ambiguity；long 358B；VL 80B+420B+44.8B，paper概述约512B | +7.5T continued；RL problem counts已披露 | M2 19.9T + 9.3T；M3 tokens `unknown` | 报告范围 `verified`；loss tokens `open` |
| parse/extract | web/book extraction细节少 | refined web/PDF parser，高 recall math/code；natural QA extraction | M2 long PDF、code concat；M3 `unknown` | M1机制 `verified`；error rate `open` |
| dedup | rule clean + dedup；global dedup用于repeat experiments；阈值 `unknown` | semantic QA dedup；RL embedding dedup；SFT/RL separation | M2 AppDev MinHash；SWE rule filters；M3 `unknown` | 存在性 `verified`；全局协议 `open` |
| quality | 60B/5B-active reward labeler：knowledge depth/helpfulness/category；次级指标 | heuristics；pass@10 difficulty；rule/exec/GenRM | model reward + classifiers；rubric/AaaV/artifact checks | 机制 `verified`；calibration `open` |
| format | natural textbook-like text；dialogue/QA nested templates，反对 heavy uniform Markdown | natural QA + long-CoT；open-ended conversion | structured task/workspace/artifact schema；interleaved thinking | 机制 `verified`；full schema `open` |
| mixture | uniform base→upweight quality while retain low-score/categories；long length buckets；VL 20:1 | CPT 70% STEM/code/book/reasoning；SFT math/code 60%；RL dynamic curriculum | code/math/STEM upweighted；short/long decay；agent domains按bottleneck动态分配；M3 `unknown` | 已述方向/比例 `verified` |
| repetition | low-quality >2 epochs下降；high-quality到4 epochs仍有效（local proxy result） | synthetic reasoning在80K RL中下采样，因重复/同质化 | trajectory/state reuse存在，完整 repeat accounting `unknown` | local result `verified`；外推 `open` |
| decontam | `unknown` | math RL 用 n-gram + embedding 对常用 test sets；target/threshold未列全 | report未给全面 protocol | M1局部 `verified`；总体 `open` |
| long context | 128K/512K/1M length mixture + source-weight interpolation；10% long QA in each stage末20% | 32K→1M CPT；RL output 40K→80K | M2 8K→32K→192K；M3 1M | stages `verified`；data-only effect `open` |
| validation | statistical data proxies；short/long benchmark；MR-NIAH code | rule/exec reward、GenRM human benchmark、online length-bias monitor | artifact-aligned verifier、within-series eval；M3 MSA 3T proxy | anchor `verified`；full held-out contract `open` |
| artifacts | weights、report、model cards、MR-NIAH/eval code；无 corpus/order/logs | weights、report、inference code | weights、reports、inference/MSA code；无 data/trajectory manifests | `D2/D3` |

## Text-01 的 data experiment：统计决策而非单点分数

### proxy setup

比较某个候选数据 $D$ 时，团队定义：

$$
H_1:\mu_{T_D}>\mu_{T_{baseline}}
$$

其中 $T$ 是 evaluation sample 上的 metric distribution，不是一次 aggregate benchmark score。multiple-choice 评测去掉选项字母，使用 completion likelihood，并对 choice probability 按 bytes 归一化，减少 tokenizer 与选项长度的影响。

稳定性用：

$$
R_{disc}=\frac{\Delta_{obvious}}{\sigma_{seed}}
$$

衡量，其中分母是跨 seed 标准差。再以 95% confidence、80% power 和接近 training variance 的 MDE 做 power analysis，最终选择 1B activated / 8B total、40B training tokens 的 MoE；20B web baseline + 20B candidate/hypothesis data。

这是可迁移的 experiment shell，但还缺关键审计字段：候选数据是否替换或追加、weights、具体 seeds、sample-size calculation、multiple-testing correction、evaluation contamination 与 proxy→target rank correlation。

### “comparable token quantities”仍需精确定义

如果 baseline 是 20B web + 20B control，而 treatment 是 20B web + 20B $D$，则实验识别的是替换后的 mixture effect：

$$
\Delta_D=Metric(0.5W+0.5D)-Metric(0.5W+0.5C)
$$

它不是 $D$ 的内在质量分数，也不能分离 domain coverage、重复率、难度与格式。多组候选共享 benchmark 时还应控制 false discovery rate，并在独立 decision set 上复核最终 winner。

## Quality、format 与 mixture 的三组结论

### reward labeler 不是单一质量分

Text-01 以前代 60B total / 5B active MoE 作 document reward labeler，初始维度含 coherence、conciseness、educational value、helpfulness、knowledge richness、category relevance；因相关性，最终主要看 knowledge depth、practical helpfulness 与 categorical distribution，其他作为secondary indicators。

这避免将完全共线的标签重复计权，但“相关”不表示维度语义相同。需要保留原始 scores、labeler version、human calibration、language/domain strata 和 threshold，才能分析某类知识被过滤还是被降权。

### heavy formatting 的边界

报告认为网站/书籍清理后可保留自然 textbook-like 格式，dialogue/QA 使用多模板 nested document format；统一重写成 heavy Markdown 会引入固定模式、降低自然变化。这个结论的合理迁移是“格式变换也要做 ablation”，不是“Markdown 总是有害”。应固定文字内容，仅切换自然、单模板和多模板，测 format overfitting、instruction parsing 与跨模板泛化。

### 低分数据不全删

高 knowledge/helpfulness scores 一般更好，但完全消除 lower-scoring content 会伤 downstream tasks。因此最终从 base-corpus uniform distribution 出发，提升高质量权重，同时保留 categories 与部分低分内容。

这说明 quality threshold 与 coverage constraint 是双目标：

$$
\max_q\ \mathbb{E}_{x\sim q}[quality(x)]
\quad\text{s.t.}\quad
coverage_k(q)\ge c_k
$$

实际 $c_k$、language floors、score bins 和 sampling temperatures 未公开。

## Repetition-aware experiment

直接在小实验中复制 final-training distribution 会使同一 unique document 被过多重复，导致 proxy 与 full run 不同。MiniMax 的做法是先 global dedup，再在小预算内 downsample，使 repetition frequency 对齐最终 schedule。

报告的局部结论是：low-quality data 超过2 epochs 后表现显著下降，high-quality data 可有效训练到4 epochs。这里的 epoch 是对已定义 pool 的 sampled exposure：

$$
epochs=\frac{T_{sampled}}{T_{unique,pool}}
$$

它依赖 MinHash cluster、tokenizer、sampling replacement 与 pool definition。不能外推为“所有高质量数据恰好可重复4次”。更强实验应做 quality-bin × repetition-count factorial，并同时报 held-out loss、exact/fuzzy memorization、domain macro metrics 和 train/validation near-duplicate rate。

## Long-context extension：length mixture 是显式 curriculum

Text-01 的三阶段数据表为：

| max length | tokens | short <32K | medium 32K–128K | long >128K |
|---:|---:|---:|---:|---:|
| 128K | 300B | 30% | 70% | 0% |
| 512K | 32B | 35% | 35% | 30% |
| 1M | 26B | 30% | 30% | 40% |

每阶段最后20% cycles 还加入10%与 pretraining long data 长度分布相似的 high-quality long QA。source-specific weights 在过渡期线性插值：

$$
w_s(t)=(1-\alpha_t)w_s^{old}+\alpha_t w_s^{target}
$$

这比瞬间切 mix 更易稳定，但仍同时改变 max length、RoPE、length buckets、QA ratio 与 source weights。NIAH 在初期很快饱和，报告因此改用更难的 intermediate-checkpoint tasks；这是重要的 metric lifecycle：容易指标到 ceiling 后不再能指导继续训练。

post-training 又是五阶段 8K→1M→8K→1M→8K，分别对应 short SFT、long SFT、short DPO、long DPO、short online RL。training stage 的交替不说明短能力永久保留，必须在每个 checkpoint 同时报 short/mid/long per-domain metrics。

## VL-01：pair、unique image 与 multimodal token 三种计量

### vision encoder

原始 pool 含 694M unique image-caption pairs，其中180M images有 refined captions；训练时 raw/refined caption 等概率抽样。但 ViT exposure 是37B pairs@224，再1.2B pairs@336，因此重复很高：

$$
pair\ exposures\gg unique\ pairs
$$

这不意味着每张图均匀出现约55次，因为 sampling weights、filtering 与 refined-caption availability 未公开。caption 均截断76 tokens，pair exposure也不能直接换算为 VL-LLM multimodal tokens。

### description 与 instruction stages

100M Common-Crawl-like images先由 caption model合成约300-token fine-grained descriptions，再人工 refinement；Stage I 从中 sampled 80B tokens做 modality alignment。Stage II 用420B multimodal instruction tokens，并按20:1混入 Text-01 post-training data，防 language forgetting。Stage III 用44.8B multimodal tokens训练1 epoch，以 common-app images 和人工 multi-turn labels改善用户体验。

对 instruction diversity，团队从1M pairs均匀抽样，让另一VLM打 concise capability tags，得到约50K unique tags、2,817个出现超过10次，再聚为14 categories。这个图描述的是 sampled instruction slice 的 model-assigned taxonomy，不是全训练集的 ground-truth domain distribution。

### preference data

Stage IV 的40K image-text pairs来自 instruction/real-user prompts；候选响应由不同 temperatures、image weakening，以及 Text-01 故意注入 hallucination/error 生成。Text-01 又作多维 evaluator，最高/最低分成 preference pair，差距小的丢弃。这里 teacher 同时是 error generator 与 judge，可能产生 correlated blind spots；需要 human adjudication、position swap、cross-judge disagreement 与 benchmark leakage audit。DPO 因 overfitting 在不足1 epoch提前停止，这也表明 dataset-size不能脱离 update count 与 checkpoint rule解释。

## M1：natural-only continued pretraining 与 synthetic RL 并不矛盾

M1 从 Text-01 继续训练7.5T tokens：先2.5T constant LR，再5T decay；优化 web/PDF parsing和heuristic rules以提高 math/code recall，抽取 webpages/forums/textbooks中的 natural QA，对QA做 semantic dedup，并把 STEM/code/books/reasoning 增至70%。该阶段明确避免 synthetic data。

后续 RL 却包含 SynLogic 生成的约53K logic problems。这不是冲突：

```text
P1 continued pretraining: natural QA only
A2 RL: natural math/code/SWE + synthetic logic + model-judged general prompts
```

阶段、目标与 verifier 不同，必须单独记录。

### RL problem accounting

- math：hundreds-of-thousands initial → incomplete/format filter → embedding dedup → 与SFT隔离 → n-gram/embedding benchmark decontam → rule-parser validity → pass@10 in $(0,0.9)$ → nearly 50K；
- logic：41 task generators + rule verifiers；难度随强模型/当前模型 pass@10 界定并动态提升 → ~53K；
- competitive programming：public OJ/code sites；缺 tests 时由 Text-01 生成 test suites；difficulty filter → 30K；
- SWE：public GitHub issues/PR、container sandbox、tests → several thousand；
- general：25K，含有/无 ground truth，GenRM 或 pairwise reference judgment。

这些是 problem samples，不是 rollouts或loss tokens。M1还明确隔离SFT/RL题目，因为重合会降低 exploration；这是 training-stage leakage，不等于 benchmark contamination。

### GenRM length bias 是在线数据质量问题

offline 加入不同长度/来源/quality tiers和adversarial cases仍不能阻止RL中的 length seeking。团队在线监控“输出变长但task success/reasoning depth不增”的信号，触发GenRM recalibration，并用 reward shaping/clipping/normalization。由此可见 reward-model validation distribution 必须覆盖 evolving policy；静态 RM benchmark 不能保证在线不被 exploit。

### 40K→80K 的负结果

M1-80K 用40K model重新评估 pass rate，去掉easy samples并提升难math/code权重。报告还下采样 synthetic reasoning：其输出在长上下文RL中变得重复和同质，持续 exposure伤整体表现。这个结果与Text-01“高质量数据可4 epochs”对象不同：前者是 on-policy long-output stability，后者是 pretraining proxy repetition。不能合并成 synthetic 或 repetition 的普遍结论。

长度按40K→48K→56K→64K→72K→80K扩展；切换条件包含generated-sequence perplexity收敛和output-length p99接近当前窗口。应同时记录 positive/negative length distribution，因为负样本更快触顶会造成后段负梯度不平衡与pattern collapse。

## M2：pretraining decay 与 agent-data factory

### base data

M2 的19.9T constant phase使用 web、academic、books、code和structured QA，经model reward scores与auxiliary classifiers做多维质量评估，并提升code/math/STEM。9.3T decay同时混 short high-quality decay data 与 long data，context 8K→32K→192K；long source含 high-quality code concatenation、natural long PDFs和thematically related packing。

“9.3T decay”不是全为long sequences，也没有给32K/192K各自tokens和真实长文/packing比例。因此不能把29.2T总segments解释为29.2T unique或long tokens。

### SWE pipeline

M2 从 permissive-license public GitHub repos 收 merged PRs/linked issues/tests，经 rule filter；agent构建并迭代修复多语言Docker environments；按bug/feature/performance/refactor/test等route；构建F2P/P2P、event/performance tests；LLM检查problem-test consistency并补全信息；再通过merge adjacent commits、SWE-Test inversion与code review等变换扩充。最终instance至少包含：

```text
problem statement
+ runnable Docker environment
+ test-based verifiable reward
+ repository/commit provenance
```

报告只给large-scale与>10 languages，没有给raw PR、build success、retained tasks、trajectories与tokens，均为`unknown`。

### AppDev 与 artifact-aligned verification

AppDev 由专家定义meta queries、system prompts与execution/interaction/aesthetics rubrics；MinHash去重、rubric filter后采trajectory。sampling时给full expert system prompt，training时选择性删除部分指导，形成prompt distillation。

Agent-as-a-Verifier分三层：execution hard gate、Playwright interaction、visual aesthetics。它部署真实应用、多轮交互并要求evidence，比static screenshot judge更强，但aesthetics仍依赖rubric/judge且可能被style template exploit。需要保存runtime、browser version、screens/video、tool trace、rubric version与verifier model。

### reverse synthesis 与 workspace tasks

M2 的 cowork pipeline展示两种减少幻觉的构造方向：先执行financial tools拿grounded trace，再反推出task/answer；或让agent在seed workbook上执行atomic operations，将intermediate states回收为new seeds，再从trajectory反向生成answer/question。这种 `execution trace --grounds--> task` 比先写题再找证据更可验证，但会把工具覆盖和seed workbook偏差写入任务分布。

### scheduler 本身会改变 data distribution

Forge 指出 strict FIFO保留分布但被straggler阻塞，fully greedy则让短/easy trajectories先进入batch、hard tasks后聚集，造成distribution shift与gradient oscillation。这是系统级数据指标：

$$
q_{consumed}(x\mid t)\neq q_{submitted}(x)
$$

即使task manifest不变，completion-time-correlated scheduler也会改变实际训练顺序。必须按domain/difficulty/trajectory length报告submitted、completed、accepted和consumed distributions。

## M3：native multimodality 的最小可知边界

M3 model card称 text/image/video 从first step混合训练，约428B total/23B active、1M context；但未披露 token total、modality ratio、visual tokenizer accounting、source、rights、filter、dedup、cutoff、curriculum或contamination。

MSA论文用109B MoE、3T-token native multimodal proxy比较GQA与MSA，并报告下游近似持平。这个3T是architecture research model的budget，不是M3的training token count。M3只能作为“Text-01先text后VL adaptation”到“native mixed-modality”的方法变化记录，不能量化数据变化。

## 如何验证这些结论，并检查 contamination

### 有辨识力的公开锚点

- `[来源事实 | verified]` data proxy用sample-level distribution、seed noise、MDE、95% confidence/80% power，而非只比较aggregate score。
- `[来源事实 | supported]` repetition-aware proxy称比直接复刻final distribution更对齐large-compute result；未公开rank correlation与完整runs。
- `[来源事实 | verified]` Text-01 long curriculum给tokens/length buckets，并观察NIAH早期ceiling；改用更难checkpoint tasks。
- `[来源事实 | verified]` VL-01公开unique pairs、pair exposures、multimodal tokens、stage mix与DPO pair size。
- `[来源事实 | verified]` M1给出分域problem retention、SFT/RL隔离、部分decontam和online GenRM length-bias处理。
- `[来源事实 | verified]` M2给出pretraining phase tokens、agent-data schema、multi-layer verifier和scheduler distribution-shift机制。

### 主要混淆与检查

| 结论 | 首个混淆 | 区分性检查 |
|---|---|---|
| high-score upweight更好 | score与domain/category相关 | 同category/language/length内stratified resampling；固定retention和tokens |
| 低分数据不应全删 | 可能只是coverage loss | 2×2 quality threshold × category floor；micro/macro/per-domain |
| high-quality可4 epochs | pool size/dedup与quality共变 | fixed unique pool，quality-bin × repeats；memorization与held-out loss |
| long curriculum支持1M | architecture/RoPE/QA/mix共变 | checkpoint fork：length-only、data-only、both；真实长文与packing分层 |
| VL 20:1防language forgetting | 无0:1/20:1/variable对照 | 固定multimodal tokens，扫text replay ratio，报text与vision Pareto curve |
| natural-only CPT改善reasoning | parser/mix/LR/context一起变 | Text-01 fork，单独切natural QA、70% mix、context schedule |
| synthetic destabilizes80K RL | synthetic quality/teacher/repeat未知 | 同prompt difficulty与token budget，比natural、single-teacher、multi-teacher、verified synthetic |
| M2 agent pipeline推动代际gain | model/compute/scaffold/eval也变 | same checkpoint/scaffold，按task family增量RL；固定rollout/update budget |

### 缺失的 validation contract

仍缺 pretraining validation corpus的source、time cutoff、split/dedup unit、freeze date、每域tokens、micro/macro loss、decision thresholds、checkpoint rule；data-experiment benchmark contamination与multiple comparisons也未完全披露。公开weight/eval code不能替代training-data manifest。

## 看似矛盾的说法怎样区分

### 11.4T 与完整 Text-01 exposure

11.4T是三段LR文字相加；500 warmup iterations是否已含、精确batch/token accounting未知。再加358B long extension只能得到reported-segment exposure，不是unique/loss-token exact total。

### 512B VL tokens 与 544.8B stage sum

paper abstract称about 512B vision-language tokens，而详细stage为80B+420B+44.8B=544.8B，另有40K DPO pairs。可能是approximate headline、阶段归类或计数口径差异；保留`ambiguous`，优先逐阶段记录，不把544.8B改写成512B。

### 694M unique pairs 与38.2B pair exposures

前者是pool去重规模，后者是训练抽样次数；不是矛盾。需要sample order/repeat histogram才能知道哪些图被重复。

### M1“禁synthetic”与53K synthetic logic

禁用只属于7.5T continued pretraining；logic synthetic属于RL。训练阶段不同。

### M1下采样synthetic 与 Text-01高质量重复

M1观察的是long-output RL中的重复/同质pattern和stability；Text-01观察pretraining proxy的document repeats。对象、目标和指标都不同。

### M2 29.2T 与 M3 token未知

M2的19.9T+9.3T不应回填给M3。M3是不同architecture/native multimodal model，官方没有声明完整inheritance。

## 哪些经验可以借鉴，哪些不能直接照搬

### 可迁移

- `[综合判断 | supported]` data proxy应先估seed noise与MDE，再定model/token/sample size；对多个候选还要控制selection bias。
- `[综合判断 | supported]` quality filtering要与category/language coverage constraints联合优化，保留score-bin retention曲线。
- `[综合判断 | supported]` repetition实验必须对齐final schedule的unique-to-sampled ratio，而不是在小预算中盲目复制final weights。
- `[综合判断 | supported]` long-context监控需要ceiling-resistant tasks，并分short/medium/long及真实/拼接来源。
- `[综合判断 | supported]` agent data的最小artifact是task+workspace+tool trace+deliverable+verifier+reward，不只是prompt/response。
- `[综合判断 | supported]` rollout scheduler会改变实际消费分布，应像sampler一样被版本化和审计。

### 不可外推

- `[未知 | open]` 2/4 epochs、70% domain mix、20:1 text replay或具体length buckets不是跨模型最优值。
- `[未知 | open]` 不能从MiniMax benchmark或1M capability反推train source、cutoff、contamination或真实长文比例。
- `[未知 | open]` permissive-license GitHub subset不能代表全部pretraining/post-training data都具相同rights。
- `[未知 | open]` M2代际benchmark提升不能仅归因于新增data pipeline；Forge、checkpoint、scaffold与eval时间也变化。
- `[未知 | open]` M3的3T proxy architecture experiment不是M3训练规模。

## 目前仍不知道什么

- 完整source manifest、language/domain/modality比例、rights/consent、cutoff；
- unique/sample/non-padding/loss tokens、tokenizer revisions、packing/loss masks；
- reward labeler标签集、人工校准、阈值、各语言误杀和版本；
- dedup算法阈值、cluster representative、global repeat histogram；
- data proxy的seeds、weights、sample counts、multiple-testing correction与proxy→target correlation；
- long-context真实长文/拼接/QA/source比例和每阶段sample order；
- VL visual-token accounting、resolution distribution、image safety/rights与caption-refinement provenance；
- M1/M2 task raw/retained/build-success、rollout/trajectory/tool-call/loss-token规模；
- M2 verifier false positive/negative、artifact retention、scheduler-consumed distribution；
- M3训练tokens、modality mix与M2/VL-01 lineage。

## 读完后应该能回答的问题

完成后应能：

1. 用MDE、power与seed variance解释为什么40B-token proxy可能比更小的单seed run可靠；
2. 区分unique images、pair exposures、multimodal sampled tokens与loss tokens；
3. 解释为什么low-score保留可能是coverage effect，而不是低质量本身有益；
4. 重建Text-01的358B long curriculum，并指出联合干预；
5. 分清M1 natural-only CPT与synthetic logic RL；
6. 画出M2从GitHub PR到workspace/verifier/trajectory/reward的provenance；
7. 解释greedy rollout scheduling如何改变训练数据分布。

自测：

- 若candidate data只提升与其domain相同的benchmark，weighted average显著，是否应进入full mix？还需哪些macro/coverage约束？
- 40B proxy中同一高质量pool重复4次，但full run只重复1.5次，怎样重新采样才能提高transfer validity？
- VL Stage II加入5% text replay后text loss恢复、vision score下降，如何找Pareto-optimal ratio？
- agent task的test suite由同一teacher生成并验证，怎样发现共同盲点？
- greedy scheduler使short tasks提前消费，应该重排队列、importance weight还是分bucket更新？各自改变了什么estimand？

## 来源与建议阅读位置

1. [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313)：先读§4.1 data quality/format/mixture与statistical experiment，§4.2的11.4T segments/358B long curriculum，再读§6的VL data/stages。
2. [MiniMax-01官方仓库](https://github.com/MiniMax-AI/MiniMax-01)：固定Text/VL weights、model cards、MR-NIAH/eval与inference assets；不是corpus release。
3. [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)：读§2的+7.5T natural-only CPT，§4分域RL data/reward，§5的40K→80K scheduler与synthetic负结果。
4. [MiniMax-M1官方仓库](https://github.com/MiniMax-AI/MiniMax-M1)：用于固定40K/80K checkpoint与部署实现。
5. [The MiniMax-M2 Series](https://arxiv.org/abs/2605.26494)：读§3的19.9T+9.3T，§4各agent-data factory，§6 Forge scheduling与§8 within-series comparison。
6. [MiniMax-M2官方仓库](https://github.com/MiniMax-AI/MiniMax-M2)：固定M2 weights、interleaved-thinking与tool-call协议。
7. [MiniMax-M3 model card](https://huggingface.co/MiniMaxAI/MiniMax-M3)：读取native multimodal from first step、1M、参数与license；token/source字段保持unknown。
8. [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)：读取109B/3T native-multimodal architecture proxy；不要把3T回填为M3训练规模。
