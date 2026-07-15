---
title: "合成数据与蒸馏：从来源衍生到学生暴露"
description: "恢复 source、teacher、transform、verifier、accepted derivatives、采样与 student loss 的完整审计链。"
topic: "pretraining-data"
section: "cross-cutting"
slug: "synthetic-data-and-distillation"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 80
readtime: 16
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/synthetic-and-distillation.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/synthetic-and-distillation.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:087ca479d042e7954e3d62763cbc18b9ac43c1e73bb6094af4c850e9845f8f79"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`unified-framework` -->
> 主要参考案例：NVIDIA Nemotron、Apple AFM、Alibaba Qwen、Xiaomi MiMo；开放控制：AI2 OLMo/Dolma、Apple OpenELM

## 这篇专题要回答什么

“用了1T synthetic tokens”至少可能表示：对真实网页重写后的P0文本、teacher logits、SFT回答、preference pairs、RL trajectories，或同一source的多个派生版本。若只按最终token相加，会把独立信息、表面多样性和训练监督混成一个量。

最小例子：一篇网页生成8个QA和1个rewrite。它可能产生远多于原文的tokens，但独立source仍只有1篇；若train/eval都从该网页派生，document-level split也会泄漏。

```text
source document
  -> prompt/template + teacher checkpoint
  -> candidate outputs
  -> verifier/judge/rejection
  -> accepted derivatives
  -> sampler/repetition
  -> student loss
```

专题目标不是判断 synthetic “好或坏”，而是说明这条链每一段分别记录什么统计单位、版本和验证强度，以及公开资料在哪些环节仍有缺口。

## 统一对象

| 对象 | 主键/单位 | 必须保留的字段 |
|---|---|---|
| source | document/repo/problem/image/trajectory seed | URI/hash、license、cutoff、group id |
| teacher | fixed checkpoint/API snapshot | model/version、license、system prompt、decoding |
| transform | generation job | template、temperature、seed、input/output mapping |
| candidate | output/solution/trajectory | raw output、length、tool/environment version |
| verification | check/judge event | verifier版本、reward、threshold、reject reason |
| accepted derivative | record/pair/sequence | parent ids、dedup cluster、quality labels |
| exposure | sampled/loss tokens | sampler weight、repeats、packing、mask、stage |

派生数据图可写成：

$$
G=(V,E),\qquad
E\in\{\text{derived-from},\text{generated-by},\text{verified-by},\text{sampled-into}\}.
$$

只给最终文件而没有边，无法区分多source覆盖与单source扩写。

## 三类互不等价的“规模”

设独立source为 $s$，其接受的派生样本数为 $m_s$，第 $j$ 个样本token数为 $t_{sj}$，训练重复次数为 $r_{sj}$：

$$
N_{\text{source}}=|S|,
$$

$$
T_{\text{derived}}=\sum_s\sum_{j=1}^{m_s} t_{sj},
$$

$$
T_{\text{sampled}}=\sum_s\sum_{j=1}^{m_s} r_{sj}t_{sj}.
$$

三者分别回答coverage、dataset体积和student exposure，不能互换。进入loss的tokens还要扣除prompt mask、padding和无效trajectory片段。

## 方法族

| 方法 | 典型阶段 | 增加什么 | 主要风险 |
|---|---|---|---|
| rewrite/paraphrase | `P0/P2` | surface、style、可读性 | 保留原错误、伪造新source量 |
| extract/QA/list | `P0/P2/A1` | task views、knowledge access paths | 同源泄漏、teacher framing |
| textbook/solution synthesis | `P0/P2/A1` | explanation、reasoning steps | hallucination、format monoculture |
| translation | `P0/P2/A1` | language coverage | translationese、语义漂移、重复知识 |
| logit distillation | `P0/P2` | soft/hard teacher supervision | teacher bias、tokenizer/label mismatch |
| rejection sampling | `A1/A2` | 高reward responses | verifier gaming、diversity collapse |
| verified rollout | `A2/A3` | executable reasoning/agent trajectories | environment leakage、reward overfit |
| preference synthesis | `A2` | paired/ranked signals | judge self-preference、position/style bias |

## 厂商公开资料实际告诉了我们什么

| 案例 | source→derivative | verification | exposure/边界 |
|---|---|---|---|
| [Nemotron-CC](/topics/pretraining-data/nvidia-nemotron-data-practices/) | low-quality web→Wiki rewrite；HQ web→rewrite/QA/distill/extract/list | quality buckets、selected ablations；逐条factuality未闭合 | 4.4T real unique +1.9T synthetic；1.9T不是1.9T独立source |
| [Nemotron-4/3](/topics/pretraining-data/nvidia-nemotron-data-practices/) | topic/keyword seeds→SFT/preferences；task seeds→code/agent data | ground truth、program verifier、judge、reward model | alignment >98% synthetic；teacher与checkpoint迭代依赖 |
| [Apple AFM 2024](/topics/pretraining-data/apple-foundation-models-data-practices/) | math evolution、tool bootstrap、code self-instruct；teacher top-1 distill | ground truth、LLM judge、compile/tests、reward model | 12K code triplets；on-device6.3T distill与188B mask分账 |
| [Apple AFM 2025](/topics/pretraining-data/apple-foundation-models-data-practices/) | images→>5B captions；charts→code render→teacher QA；MoE teacher→dense student | CLIP/OCR、rule/code verification、RM/judge | 1T teacher branch不加进1.4T student replay；image集合有重叠 |
| [Qwen](/topics/pretraining-data/qwen-data-practices/) | code/math specialist、off/on-policy responses | execution/answer verifier、rejection与RL reward | pretraining derivative、SFT、RL rollout按阶段分列 |
| [MiMo](/topics/pretraining-data/mimo-data-practices/) | reasoning extraction、multimodal/agent trajectories、MOPD | verifier、environment、multi-teacher token supervision | 25T/27T/48T口径需识别阶段/衍生关系 |
| [OLMo/Dolma开放控制](/topics/pretraining-data/ai2-olmo-dolma-data-practices/) | 主要以可版本化自然/公共corpus为对照 | data/config/order/logs可查 | 展示没有synthetic lineage时应怎样闭合source→exposure |
| [OpenELM开放控制](/topics/pretraining-data/apple-foundation-models-data-practices/) | public named datasets；Instruct=cleaned UltraFeedback | config/logs/checkpoints、固定dataset | pool 1.8T vs sampled 1.5T，说明自然数据同样需分pool/exposure |

## 统一比较会丢掉哪些重要差异

- Nemotron-CC的rewrite与QA共享source，统一成“derivative”会隐藏transform目的；需保留transform subtype。
- Apple logit distillation不一定产生可存储文本，统一成“synthetic sample”会误报dataset size；需保留supervision representation。
- Qwen/MiMo的on-policy rollout随student checkpoint改变，统一成固定dataset会丢失policy version与环境状态。
- OLMo/OpenELM没有生产级synthetic flywheel，作为control只能校准artifact完整度，不能估计synthetic收益。

因此统一表只比较lineage字段，不比较“synthetic比例越高越先进”。

## verification 不是单一标签

验证强度取决于命题：

```text
format/schema check
  < execution or exact-answer check
  < independent proof/test suite on covered property
```

但顺序不是全局总序：程序tests只能验证覆盖到的行为；LLM judge可能覆盖开放问题却有偏差。每条验证需写：

- 被验证的property；
- false accept/false reject的估计；
- verifier是否见过seed/eval；
- teacher与judge是否同族；
- reject reason distribution；
- accepted样本在domain/difficulty上的偏移。

通过率：

$$
q_{d}=\frac{N_{\text{accepted},d}}{N_{\text{generated},d}}
$$

必须按domain/difficulty报告；全局$q$会掩盖某域几乎全被拒绝。

## diversity 与有效覆盖

文本不同不等于语义独立。可用source-group与semantic cluster双层统计：

$$
N_{\text{effective}}
=\frac{(\sum_c w_c)^2}{\sum_c w_c^2},
$$

其中 $w_c$ 是semantic cluster的sampled exposure。若大量tokens集中在少数cluster，$N_{\text{effective}}$下降，即使文件很大。

诊断至少包括：

- parent source数与每source derivative分布；
- embedding cluster concentration；
- n-gram/template复用；
- teacher/style classifier可辨识度；
- rare domain/language pass rate；
- natural-only held-out loss与clean benchmark。

## contamination 与split unit

若seed来自benchmark train、solution、concept或相邻repo，只做最终文本exact dedup不够。split必须在最早共享祖先处分组：

```text
source/benchmark/repo group
  -> train derivatives OR eval derivatives
```

不能在生成后随机split。还要审计decision-layer contamination：即使test文本未出现，反复用benchmark反馈选择prompt、teacher、filter或mixture，也可能对评测决策过拟合。

## 因果实验

合成数据ablation至少固定：base checkpoint、tokenizer、sampled/loss budget、optimizer、LR、sequence distribution、seed和eval。推荐三臂：

1. natural-only，按token budget重复；
2. natural + derivative，保持总sampled tokens；
3. natural + independent-source追加，匹配domain/quality。

这样可区分“更高质量表述”“更多独立知识”“只是更多训练token”。若transform带来长度变化，应按loss tokens而非record count匹配。

## 看似矛盾的说法怎样区分

### synthetic提高质量 vs synthetic造成collapse

- 首个差异：teacher/verifier质量、source覆盖、synthetic占比与训练阶段。
- 区分检查：按比例扫描，报告natural-only per-domain loss、effective clusters和teacher-style detectability。
- 状态：两者可同时成立，无全局结论。

### verified等于正确 vs verifier只覆盖局部性质

- execution pass证明给定tests通过，不证明spec完备或无数据泄漏。
- 区分检查：hidden tests、property-based tests、人工错误taxonomy与跨环境复现。

### distillation节省compute vs teacher branch没有成本

- student训练可能减少，但teacher训练、logit生成、storage和验证仍消耗compute。
- 区分检查：分别报告teacher amortized cost、student cost和每个accepted/loss token成本。

## 最少需要保存哪些 manifest 信息

```yaml
source_id: sha256:...
source_group: repo-or-document-family
source_license: ...
transform_id: prompt-template@revision
teacher: model@checkpoint
decoder: {temperature: ..., seed: ...}
candidate_id: ...
verifier: tests-or-judge@revision
verdict: accepted
reject_reasons: []
semantic_cluster: ...
training_stage: P2
sample_count: ...
loss_tokens: ...
```

## 读完后应该能回答的问题

读者应能：

1. 区分source count、derived tokens、sampled tokens和loss tokens；
2. 解释Nemotron 1.9T为什么不是1.9T新source；
3. 不把Apple teacher 1T加进student exposure；
4. 为code/math/agent选择合适verifier并说明盲点；
5. 以source group而非最终row做split；
6. 设计fixed-budget三臂ablation。

推理题：若1M seed各生成8个样本，通过率25%，平均512 tokens，训练重复4次，则accepted derivatives、dataset tokens和sampled tokens各是多少？哪些量仍不能说明独立知识覆盖？

## 来源与建议阅读位置

1. [NVIDIA Nemotron](/topics/pretraining-data/nvidia-nemotron-data-practices/)：读Nemotron-CC 4.4T+1.9T及teacher/prompt metadata，再读>98% synthetic alignment与MOPD。
2. [Apple Foundation Models](/topics/pretraining-data/apple-foundation-models-data-practices/)：读2024 math/tool/code factory和2025 teacher/student、image caption lineage。
3. [Alibaba Qwen](/topics/pretraining-data/qwen-data-practices/)：读specialist data与off/on-policy distillation边界。
4. [Xiaomi MiMo](/topics/pretraining-data/mimo-data-practices/)：读verifier/agent/MOPD provenance和跨阶段token冲突。
5. [AI2 OLMo/Dolma](/topics/pretraining-data/ai2-olmo-dolma-data-practices/)：作为source/config/order/logs开放控制。
6. [Data lifecycle](/topics/pretraining-data/data-lifecycle/)与[Token accounting](/topics/pretraining-data/token-accounting/)：先统一生命周期和分母，再比较案例。
