---
title: "Google Gemini / Gemma 的数据实践"
description: "区分闭源 source taxonomy、开放权重 token ladder 与 WebLI、JEST 等数据方法证据。"
topic: "pretraining-data"
section: "international-cases"
slug: "google-gemini-gemma-data-practices"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 123
readtime: 31
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/google-gemini-gemma.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/international/google-gemini-gemma.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:47b1a0a7df25cb719135759a0054ef1d4cd55eb36ced70c4a6d6e93113b14f35"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`method-family/historical trace` -->
<!-- maintenance: secondary-view=`implementation-trace`，仅用于 Gemma、PaLI/WebLI 与 JEST 的公开实现关系 -->
> 主要模型/资料：Gemini 1.5、2.5、3 Pro/3.1 Pro；Gemma 1→4；PaLI/WebLI；JEST

## 这篇案例研究什么

Google 的公开材料不能被当成一条披露强度恒定的模型谱系。至少要拆成三层：

1. **Gemini production family**：报告和 model card 能确认来源类别、模态、cutoff 与高层处理方法，但 token、mixture、重复暴露和完整 validation contract 仍未知；
2. **Gemma open-weight family**：权重可下载，Gemma 1–3 报告还给出 token budget、context、distillation、过滤与部分 ablation；开放权重不等于开放训练数据；
3. **PaLI/WebLI 与 JEST research artifacts**：能研究多语言图文采集、近重复去污染和在线 batch selection，但除非产品报告建立 lineage，否则不能写成 Gemini/Gemma 的实际训练集或生产管线。

技术关系应写成：

```text
Gemini research and technology --informs--> Gemma family
Gemma weights --artifact-of--> undisclosed training corpus and order
WebLI --used-to-train--> PaLI
WebLI-curated reference --used-by--> JEST experiments
WebLI/JEST --analogy-not-lineage--> Gemini production recipe
```

## 核心问题：同一组织的论文不能自动拼成一份生产配方

如果只按“Google 发布”聚合材料，会产生三种常见错误：

- 把 Gemini 1.5 的“最多测试到 10M token”写成训练 context 或正式 API context；
- 把 Gemma 3 的 14T token、质量重加权和去污染外推给 Gemini 2.5/3；
- 把 WebLI/JEST 的公开实验流程写成 Gemini 多模态数据管线。

本篇要回答的不是“Google 用了哪些数据”的单一问题，而是：**每一条公开事实究竟属于哪个模型、哪个训练阶段、哪个数据对象和哪一类证据。**

## 各代模型和 training stage

| 家族/代际 | 阶段 | 公开数据事实 | token/context 口径 | 披露等级 | 结论状态 |
|---|---|---|---|---|---|
| Gemini 1.5 Pro/Flash | `P0/A1/A2` | 多域、多语言、多模态；pretraining 包含 web documents、code、image、audio、video；instruction tuning 为多模态 instruction-response，再用 human preference | pretraining token、mixture、repeat、loss token `unknown`；研究测试 text 10M、audio 9.7M、video 9.9M，不等于训练窗口 | `D1/D2` | source/stage `verified`；定量 recipe `open` |
| Gemini 2.0/2.5 | `P0/A1/A2/A3` | web、code、image、audio、video；2.0 cutoff 2024-06，2.5 cutoff 2025-01；2.5 增加 filtering/dedup 方法；post-training 含 paired multimodal instruction、preference、tool-use | pretraining token/mixture `unknown`；1M input context；thinking token 为推理时预算，不是训练 token | `D2` | categories/cutoff `verified`；method details `open` |
| Gemini 3 Pro/3.1 Pro | `P0/A*` | public web/downloadable datasets、crawler、licensed、受条款与 controls 约束的 user data、business/workforce、AI synthetic；RL 含 multi-step reasoning/problem solving/theorem proving | token/mixture/order/loss `unknown`；3 Pro/3.1 Pro 1M input、64K output；cutoff 2025-01 | `D1/D2` | source classes/cutoff `verified`；3.1 data diff `open` |
| Gemma 1 2B/7B | `P0` | primarily English；web documents、mathematics、code；安全/隐私过滤与 evaluation decontamination | 3T/6T sampled training tokens；8K context | `D2/D3` | budget/categories `verified`；unique/loss `open` |
| Gemma 2 2B/9B/27B | `P0` | primarily English；web、code、science articles；knowledge distillation；过滤、evaluation decontamination | 2T/8T/13T；8K context | `D2/D3` | budget/mechanism `verified`；mixture `open` |
| Gemma 3 1B/4B/12B/27B | `P0/P3/A*` | text+image；增加 monolingual/parallel multilingual data；quality reweighting；pretraining distillation；post-training human/model/ground-truth rewards | 2T/4T/12T/14T；先 32K，4B+ 在 pretraining 末段扩至128K；1B 32K | `D2/D3` | stage/budget `verified`；modality split `open` |
| Gemma 4 | `P0/A*` | web、code、math、image，部分型号含 audio；140+ languages；cutoff 2025-01；质量/安全/PII/CSAM filtering | pretraining token、mixture、context-extension exposure `unknown`；产品 context 128K/256K | `D1/D3` | categories/cutoff/weights `verified`；quant recipe `open` |
| PaLI/WebLI | multimodal pretraining research | 10B images、12B alt-texts、109 languages、29B image-OCR pairs；top 10% 约1B image-text examples；最终 8-task mixture 1.6B examples | example/pair 统计，不是 language-model token；PaLI 一轮遍历 1.6B mixture | `D2` | construction/counts `verified`；Gemini lineage `open` |
| JEST/Flexi-JEST | online data selection research | learner/reference joint loss 从大 super-batch 选 sub-batch；reference 由小规模 curated WebLI 引导 | 最高约13× fewer iterations、9.9× fewer FLOPs 均为论文 tested setup 的 empirical association | `D2/D3` | local result `verified`；跨任务/生产迁移 `open` |

## 阅读这些结论前先确认的前提

| 维度 | 本篇约束 |
|---|---|
| 研究对象 | Gemini/Gemma 代际与公开研究方法，不假定它们共享完整 corpus |
| 统计单位 | language-model token、image-text pair、image、OCR pair、training example、context token 分开 |
| 阶段 | `P0`、context extension、distillation、SFT、RM/RL、inference thinking budget 分开 |
| “开放” | Gemma 的权重和部分代码开放；训练 corpus、manifest、顺序和完整生成 provenance 未开放 |
| context | 支持/测试长度、实际训练 seqlen、长上下文 token exposure 是三个字段 |
| cutoff | 只记录报告明确给出的 knowledge/data cutoff；不能由模型回答估计 |
| 省略效应 | 架构、tokenizer、teacher、filter、mixture、context schedule 与 post-training 同时变化 |

## 厂商公开了哪些 data fields

| 字段 | Gemini | Gemma 1–3 | Gemma 4 | WebLI/JEST |
|---|---|---|---|---|
| source | 高层 source classes | web/code/math/science/image 等类别 | web/code/math/image/audio 类别 | public web image/text/OCR；curated reference |
| rights | Gemini 3 区分 public/crawled/licensed/user/business/synthetic | 具体 source license mix `unknown` | 权重 Apache 2.0；训练数据 rights mix `unknown` | 单样例许可不等于全 corpus 可再发布 |
| cutoff | 2.0: 2024-06；2.5/3: 2025-01 | `unknown` | 2025-01 | 论文固定版本；不是产品 knowledge cutoff |
| token accounting | unique/sample/loss 全部 `unknown` | sampled budget 已知；unique/loss `unknown` | 全部 `unknown` | example/pair/FLOPs；不能与 LM tokens 相加 |
| mixture | modality/source share `unknown` | source share `unknown` | share `unknown` | PaLI 8-task coefficients未形成通用产品 recipe |
| quality | safety/quality filtering；2.5称采用新方法 | quality reweighting、敏感数据过滤 | quality/safety/PII/CSAM filtering | cross-modal similarity；curated reference model |
| dedup | 2.5/3 只披露使用 dedup | evaluation decontamination，完整阈值未知 | model card未给完整阈值 | WebLI 对68套 benchmark 做 image near-dedup；JEST不是dedup |
| order/seed | `unknown` | `unknown` | `unknown` | JEST 是动态 order/selection；论文 setup 可追踪 |
| validation | capability/safety/long-context eval；pretrain held-out contract不完整 | benchmark、ablation、memorization/safety tests | benchmark/safety；pretrain validation contract未知 | 4/8 downstream tasks；并非 production validation |
| artifacts | report/model card/API | weights、报告、推理代码 | weights、model card、工具链 | papers、部分 research code；原始 WebLI 未开放 |

## Gemini：来源类别在扩展，定量配方仍闭合不了

### Gemini 1.5：10M 是 stress test 上限，不是训练 token 说明

`[来源事实 | verified]` Gemini 1.5 报告只在数据章节确认：模型在多数据中心 TPUv4 上训练，pretraining 使用多域、多语言、多模态数据，包括 web documents、code、image、audio、video；instruction-tuning 是 paired multimodal instructions/responses，之后还有 human preference tuning。

`[来源事实 | verified]` 报告对 Gemini 1.5 Pro 的长上下文研究测试覆盖：text 最多 10M token、audio 约9.7M、video 约9.9M；正式能力表与不同模型/时期的 serving context 另有边界。

因此：

```text
10M-token stress test
  --empirically-demonstrates-->
retrieval/NLL behavior in named evaluations

10M-token stress test
  --does-not-imply-->
10M training sequences or 10M-token production guarantee
```

`[未知 | open]` 训练长 sequence 来自真实长文档、repository concat、packing 还是合成长依赖，公开报告没有给出可复现 mixture、阶段 token 或 seqlen histogram。

### Gemini 2.5：cutoff 与 post-training data 类型更清楚

`[来源事实 | verified]` Gemini 2.5 报告明确给出 2.0 的 cutoff 为2024-06，2.5 为2025-01；pretraining 来源仍是 web、code、image、audio、video，并声称相对1.5采用了新的 quality filtering 与 dedup 方法，但没有算法、阈值、retention 或分域统计。

`[来源事实 | verified]` post-training 包含：

- carefully collected/vetted multimodal instruction-response；
- human preference data；
- tool-use data；
- SFT、RM、RL 中的 model-assisted quality control；
- verifiable rewards、model-based generative rewards 与多步 tool-use environments。

`[综合判断 | supported]` 这些信息足以区分“pretraining source”与“agent/reasoning post-training provenance”，但不足以计算任何 token share 或数据效率。

### Gemini 3：source/rights 类别增加，但没有 token accounting

`[来源事实 | verified]` 2026-05 更新的 Gemini 3 Pro model card 披露了更细的来源类：公开可下载数据、crawler 数据、commercially licensed data、按相关条款/政策与 controls 使用的 user data、business/workforce 产生或获得的数据，以及 AI-generated synthetic data。

`[来源事实 | verified]` 该卡同时写明 pretraining 的 web/text/code/image/audio/video 与 post-training 的 instruction、RL、human preference；reasoning RL 可利用 multi-step reasoning、problem-solving 与 theorem-proving data。

`[未知 | open]` source classes 的存在不回答：

- 每类占比和按模态 tokenization；
- user data 的具体产品、时间窗、opt-in/opt-out 分母；
- licensed corpus 的名称和权利边界；
- synthetic 的 teacher、prompt、acceptance、lineage；
- dedup 是 exact、near、semantic 还是 cross-modal；
- 3 Pro 到3.1 Pro 的训练数据差异。

## Gemma：开放权重提供实验对象，但不等于开放数据

### Token ladder

用 $T_{g,s}$ 表示 Gemma 代际 $g$、型号 $s$ 报告中的 sampled training tokens：

$$
\begin{aligned}
T_{1,2B} &= 3\text{T}, & T_{1,7B} &= 6\text{T},\\
T_{2,2B} &= 2\text{T}, & T_{2,9B} &= 8\text{T}, & T_{2,27B} &= 13\text{T},\\
T_{3,1B} &= 2\text{T}, & T_{3,4B} &= 4\text{T}, & T_{3,12B} &= 12\text{T}, & T_{3,27B} &= 14\text{T}.
\end{aligned}
$$

变量单位是报告声称的训练 token；它不是 unique corpus token，也未证明所有 token 都进入 loss。Gemma 3 报告还说明 token 增长包含 text/image mixture，因此不同模态如何折算进总数仍是 `open`。

`[推导结论 | verified]` 这些数字不能被解释成单调的 compute-optimal law。例如 Gemma 1 2B 的3T到 Gemma 2 2B 的2T同时伴随 teacher、architecture、data recipe 变化；这只是 family recipe，不是固定条件下的 token ablation。

### Gemma 3 的 distillation 与长上下文

`[来源事实 | verified]` Gemma 3 pretraining 延续 Gemma 2 的 knowledge distillation，每个 token 从 teacher distribution 采样256个 logit、截断后重归一化，以 cross-entropy 训练 student。

`[来源事实 | verified]` 长上下文不是从头全部按128K训练：模型先用32K sequence pretrain，4B/12B/27B 在 pretraining 末段通过 RoPE rescaling 扩到128K；1B 保持32K。

这产生两个不同的数据问题：

1. **distillation accounting**：相同 source token 携带 teacher soft target，storage/throughput 与监督信息量变化；
2. **context accounting**：末段128K exposure 的 token、真实长度分布、packing 与 loss mask 未披露，产品128K不能替代这些字段。

`[来源事实 | verified]` Gemma 3 的 pretraining filtering 包括 unwanted/unsafe utterance risk、部分 PII/sensitive data、evaluation decontamination、降低 recitation 风险，以及 inspired by DataComp-LM 的 quality reweighting。post-training 另行过滤 PII、unsafe/toxic outputs、self-identification errors 与 duplicates。

### Gemma 4：最新开放权重的配方披露反而收缩

`[来源事实 | verified]` Gemma 4 model card 给出 web/code/math/image、部分型号 audio、140+ language、2025-01 cutoff、128K/256K serving context，以及 PII/CSAM/quality/safety filtering；权重可下载。

`[综合判断 | supported]` 相比 Gemma 3 技术报告，Gemma 4 当前公开卡没有可比较的 training-token ladder、distillation target、context-extension schedule、decontamination threshold 或 mixture。因此应把它记录为“artifact openness 高、recipe quantification 低”，而不是沿用 Gemma 3 字段填空。

## WebLI：pair accounting 与 benchmark near-dedup 的公开锚点

`[来源事实 | verified]` PaLI 论文中的 WebLI 从 public web 构建，原始规模约：

$$
10\text{B images},\quad 12\text{B alt-texts},\quad 29\text{B image-OCR pairs},\quad 109\text{ languages}.
$$

通过 cross-modal embedding cosine similarity 评分后保留 top 10%，约1B image-text examples；PaLI 的八任务总 mixture 为1.6B examples，并训练一轮。

`[来源事实 | verified]` 为减轻 train-to-test leakage，WebLI images 对68个视觉/视觉语言数据集的 train/validation/test split 做 near-dedup，移除约0.36%；mixture 中其他数据也针对 downstream tasks 做相同检查。

最小审计表：

| 问题 | 可回答 | 不可回答 |
|---|---|---|
| 原始数据规模 | image、alt-text、OCR pair、language | LM token 或独立语义证据数 |
| 质量过滤 | cross-modal similarity top 10% | 人类效用、文化代表性或 rights completeness |
| 去污染 | 68 benchmark 的 image near-dedup | 未列 benchmark、text-only leakage、语义改写 |
| 训练使用 | PaLI 1.6B task mixture | Gemini/Gemma 是否复用及占比 |

`[综合判断 | supported]` top 10% 是针对 image-text alignment 的 operational threshold，不是跨语言、跨任务的单一“数据质量分数”。长尾语言 coverage、OCR noise 与 rights 风险仍需独立指标。

## JEST：selection score 会继承 reference distribution

设当前 learner 参数为 $\theta$，预训练 reference 为 $\theta^*$，候选 super-batch 为 $D$，大小 $B$；选出的 sub-batch 为 $\mathcal{B}$，大小 $b$。JEST 的 learnability score 为：

$$
s_{\text{learn}}(\mathcal{B};\theta,\theta^*)
=\ell(\mathcal{B};\theta)-\ell(\mathcal{B};\theta^*),
\qquad
f=1-\frac{b}{B}.
$$

这里 $f$ 是每步 filtering ratio；高 learner loss 但低 reference loss 的 batch 被视为“尚未学会、但可学”。contrastive loss 依赖 batch 内 negative composition，因此 joint selection 与逐样本独立排序不是等价关系。

```text
score super-batch with learner and curated-data reference
  -> compute batch-conditional learnability
  -> sequentially sample chunks without replacement
  -> train on selected sub-batch
  -> update learner; repeat
```

`[来源事实 | verified]` 论文 setup 中，reference 训练自约100M strongly curated WebLI；JEST++ 另加约600M curated web pairs。Flexi-JEST 用低分辨率近似 scoring，把额外 scoring overhead 降到约10%。最佳结果相对40B-example SigLIP baseline 可用约13× fewer iterations、9.9× fewer FLOPs 达到相同平均表现。

`[综合判断 | supported]` JEST 不是“自动发现客观高质量数据”。reference model 将小 curated distribution 的偏好传播给大 pool；如果 reference 在低资源语言、文化、OCR、特殊 domain 上有盲区，selection 可能系统性放大该盲区。

必须同时报告：

- super-batch/sub-batch 大小与 $f$；
- learner/reference checkpoint 和 reference training data；
- scoring FLOPs 与实际 learning FLOPs；
- selected-language/domain/modality distribution；
- selection probability/重复 exposure；
- micro、macro 与 per-domain validation；
- 与 uniform、independent score、hard-learner、easy-reference 的对照。

## Validation、污染与可区分实验

### 已公开的检查

- Gemini 报告：long-context NLL/retrieval、capability、multilingual/multimodal、safety 和 red-team evaluations；
- Gemma：标准 benchmark、long-context ablation、memorization/recitation 与 safety tests；Gemma 3 明确 evaluation decontamination；
- WebLI：对68套视觉/视觉语言 benchmark 的 image near-dedup；
- JEST：uniform/independent/joint、filter ratio、reference quality、resolution 与 compute ablation。

### 仍缺少的 validation contract

`[未知 | open]` Gemini/Gemma 的公开材料通常没有同时给出：validation source、time boundary、split unit、domain sample/token counts、tokenizer version、冻结版本、跨代 exact/semantic decontamination 范围和训练期间 decision exposure。

不得把 release benchmark table 当成 pretraining held-out loss。建议区分性实验：

| 冲突/问题 | 首个可能差异 | 区分性检查 |
|---|---|---|
| 长 context 支持是否来自训练数据 | serving limit vs training sequence exposure | 固定模型比较真实长文档、随机 concat、synthetic dependency；按长度报告 NLL/QA |
| JEST 增益来自数据还是更多 scoring compute | selected distribution vs extra FLOPs | 同总FLOPs uniform、independent、JEST、Flexi-JEST 对照 |
| reference 是否损害低资源域 | micro gain vs macro regression | per-language/domain acceptance、exposure 与 held-out macro |
| Gemma token ladder是否解释能力增益 | token 与 teacher/architecture/mixture 联合变化 | 固定架构/teacher/tokenizer/order 的 local budget ablation |
| 去污染是否充分 | exact/image near-dedup vs semantic/text leakage | modality-specific exact/near/semantic audit + post-cutoff benchmark |

## 哪些经验可以借鉴，哪些不能直接照搬

### 可迁移

- `[综合判断 | supported]` 多模态数据必须同时保存 pair/example、modality token、loss token 三套计量；WebLI 的1B pairs不能与 Gemma 的14T tokens直接比较。
- `[综合判断 | supported]` 长上下文必须记录“基础训练长度 → 扩窗阶段 → 末段 token exposure → validation length”，Gemma 3 的32K→128K是可操作锚点。
- `[综合判断 | supported]` model-based selection 需要 reference provenance；JEST 的 score 不是独立于 curated seed 的中立质量函数。
- `[综合判断 | supported]` 开放权重、开放报告、开放数据、开放顺序是四个字段；Gemma 不能因可下载权重被记成 D4 data recipe。

### 不可外推

- `[未知 | open]` WebLI/JEST 是否、如何进入 Gemini/Gemma 的 production pipeline；
- `[未知 | open]` Gemini 的 token budget、mixture、repeat、loss mask 与 long-context curriculum；
- `[未知 | open]` Gemma 3 image token 与 text token 的精确统计关系；
- `[未知 | open]` Gemma 4 的 token ladder、teacher、context-extension exposure 与 decontamination threshold；
- `[待验证假设 | plausible]` JEST 类在线选择可提高语言模型 pretraining 效率；当前主要证据是 multimodal contrastive setup，不能无条件迁移到 autoregressive LM。

## 读完后应该掌握什么

读完后应能：

1. 解释为什么 Gemini 1.5 的10M测试不等于10M训练窗口；
2. 正确复述 Gemma 1–3 的 token ladder，并说明 sampled/unique/loss 的区别；
3. 用公式解释 JEST learnability 与逐样本 quality score 的不同；
4. 说明 WebLI top 10% 和 benchmark near-dedup各自优化什么、漏掉什么；
5. 在不知道 Gemini mixture 时保留 `unknown`，而不是用 Gemma/WebLI补齐。

## 用这些问题检查自己

1. Gemma 3 27B 训练14T，Gemini 2.5支持1M context。能否据此推断 Gemini 2.5训练量大于14T？为什么？
2. JEST 在8项任务 macro平均提升，但某低资源语言下降。应该新增哪些 selection 与 validation 分母？
3. WebLI 移除了与68套 benchmark近重复的0.36% images。为什么这不证明 text/OCR 与所有下游 benchmark 都无污染？
4. Gemma 4公开权重、cutoff和source categories，却未给token。它在 source、artifact、accounting三个字段分别是什么披露等级？

## 来源与建议阅读位置

1. [Gemini 1.5 Technical Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)
   - 为什么读：固定 Gemini 1.5 的多模态 source categories 与10M long-context测试边界。
   - 建议位置：§4 Training Infrastructure and Dataset；§5 long-context evaluations。
2. [Gemini 2.5 Technical Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
   - 为什么读：固定 2.0/2.5 cutoff、filter/dedup代际差异和 post-training 数据类型。
   - 建议位置：§2.2 Dataset、§2.4 Post-training、§5.3 safety training。
3. [Gemini 3 Pro Model Card](https://deepmind.google/models/model-cards/gemini-3-pro/)
   - 为什么读：当前 Gemini lineage 中更细的 public/licensed/user/business/synthetic source taxonomy。
   - 建议位置：Model Data；Training Data Processing；Known Limitations。
4. [Gemini 3.1 Pro Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)
   - 为什么读：确认3.1依赖3 Pro数据披露，而没有独立 quantitative recipe。
   - 建议位置：Model Dependencies 与 Model Data。
5. [Gemma 1 Technical Report](https://arxiv.org/abs/2403.08295)
   - 为什么读：固定3T/6T、8K与早期过滤/去污染边界。
   - 建议位置：Training Data 与 Memorization。
6. [Gemma 2 Technical Report](https://arxiv.org/abs/2408.00118)
   - 为什么读：固定2T/8T/13T与distillation recipe演进。
   - 建议位置：§3.1 Training Data 与 scaling/ablation。
7. [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
   - 为什么读：固定2T/4T/12T/14T、256-logit distillation、32K→128K与quality reweighting。
   - 建议位置：§2.2 Pre-training、§2.3 Post-training、§5.3 long context。
8. [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)
   - 为什么读：确认最新开放权重的source/cutoff/filter字段及quantitative recipe缺口。
   - 建议位置：Model Data、Data Preprocessing。
9. [PaLI / WebLI paper](https://arxiv.org/abs/2209.06794)
   - 为什么读：学习多语言图文数据从原始 pair、alignment filter 到 benchmark near-dedup 的可检查统计流程。
   - 建议位置：§3.2、Appendix A.2、Appendix G datasheet。
10. [JEST paper](https://arxiv.org/abs/2406.17711)
    - 为什么读：学习batch-conditional selection、reference provenance与compute accounting。
    - 建议位置：§3 Methods、§4 Experiments、Algorithm 1、Appendix A.4。

## 这篇案例与主线知识的关系

```text
token accounting --prerequisite-for--> Gemma token ladder interpretation
multimodal pair accounting --prerequisite-for--> WebLI audit
reference provenance --prerequisite-for--> JEST selection interpretation
seqlen histogram --prerequisite-for--> long-context data attribution
evaluation decontamination --does-not-imply--> production-corpus reproducibility
open weights --does-not-imply--> open training data
```
