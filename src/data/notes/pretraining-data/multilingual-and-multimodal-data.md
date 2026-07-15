---
title: "多语言与多模态数据"
description: "用 tokenizer fertility、模态统计单位、采样分母与来源权利约束解释 multilingual/multimodal mixture。"
topic: "pretraining-data"
section: "cross-cutting"
slug: "multilingual-and-multimodal-data"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 83
readtime: 16
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/multilingual-and-multimodal-data.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/multilingual-and-multimodal-data.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:7f0d192fabe17b364afc5a03c674eccc1fff10fa3fa3314934270934e3892638"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`unified-framework` -->
> 主要参考案例：Apple AFM、Google Gemini/Gemma、Meta Llama、ByteDance Seed、Tencent Hunyuan、Xiaomi MiMo、Mistral；开放控制：AI2 OLMo/Dolma、Apple OpenELM

## 这篇专题要回答什么

“训练了12T multilingual/multimodal tokens”可能同时包含text tokens、image embeddings、video frames或audio codes。不同tokenizer下，同一句低资源语言又可能消耗不同token数。若没有单位与分母，language/modality mixture无法比较。

最小例子：中文tokenizer改进30%后，同样1TB中文文本产生更少tokens。固定1T-token训练可能覆盖更多字符，却也可能被sampler重新配平。token比例既不是byte比例，也不是document比例。

## 需要分别统计哪些单位

### 语言fertility

对语言 $\ell$ 的固定规范化语料：

$$
f_\ell=\frac{N_{\text{tokens},\ell}}{N_{\text{characters},\ell}},
\qquad
c_\ell=\frac{N_{\text{bytes},\ell}}{N_{\text{tokens},\ell}}.
$$

报告training token share时同时给bytes/characters与tokenizer version。fertility低不自动表示model quality高。

### 模态positions

总sampled positions可写为：

$$
T= T_{text}+T_{image}+T_{video}+T_{audio}+T_{special},
$$

但只有在position定义相容时才可相加。必须另报原始单位：unique images、pairs、interleaved docs、frames/seconds、audio hours和text bytes。

## 比较不同厂商时要记录什么

| 维度 | 语言 | 图像/视频/音频 |
|---|---|---|
| source | web/books/code/licensed/translation | crawl pairs、interleaved docs、licensed media、synthetic |
| raw unit | bytes/docs | images/pairs/docs/frames/seconds/hours |
| model unit | tokenizer tokens | patches/embeddings/codes |
| sampling | natural/temperature/capped | pair/interleaved/task/modality weights |
| quality | LID、per-language model filter | alignment、resolution、aesthetic/OCR/safety |
| dedup | script/normalization/cross-language | exact/perceptual/video-segment/caption |
| rights | locale/source license | image/personality/face/music/voice与derivative rights |
| validation | per-language loss/macro | per-modality/task/resolution/locale |

## 厂商公开资料实际告诉了我们什么

| 案例 | 披露锚点 | 可知 | 不能推出 |
|---|---|---|---|
| [Apple AFM 2025](/topics/pretraining-data/apple-foundation-models-data-practices/) | tokenizer 100K→150K；multilingual 8%→30%；10B+/5B/6B image pairs | text continued与MM mix、144 embeddings、SFT/RL 80:20 | unique images、loss tokens、跨代等量文本 |
| [Google Gemini/Gemma](/topics/pretraining-data/google-gemini-gemma-data-practices/) | multimodal source taxonomy、Gemma token ladder、WebLI pair/decontam | open/closed差异、pair与model-token分离 | Gemini mixture/token、WebLI=production lineage |
| [Meta Llama](/topics/pretraining-data/meta-llama-data-practices/) | Llama3 15.6T multilingual/code、Llama4约22T multimodal | generation/stage headlines | 模态token定义、每语/模态loss比例 |
| [Seed1.5-VL](/topics/pretraining-data/bytedance-seed-doubao-data-practices/) | caption/interleaved/OCR/grounding/video；3.256T stage chain | text-only约5%于Stage1、task scaling | image独立数、3T headline closure |
| [HunyuanOCR](/topics/pretraining-data/tencent-hunyuan-hy-data-practices/) | >200M pairs、130+ languages、454B四阶段 | pair/stage/task accounting | image/loss accounting与group contamination |
| [MiMo-VL](/topics/pretraining-data/mimo-data-practices/) | vision-language stages、固定budget实验 | injection timing/ratio、stage boundaries | image/text token等价、完整rights |
| [Mistral NeMo/Pixtral](/topics/pretraining-data/mistral-ai-data-practices/) | Tekken>100 languages；interleaved image-text、16×16 patches | tokenizer compression、sequence shape | NeMo language mix、Pixtral pair数量 |
| [OLMo/OpenELM控制](/topics/pretraining-data/ai2-olmo-dolma-data-practices/) | named/versioned text data与tokenizer/config | text source→token exposure审计 baseline | 不代表multimodal生产pipeline |

## 统一比较会丢掉哪些重要差异

- Apple给训练mixture percentages，Hunyuan给pairs与stage tokens，不能用同一列直接排名；
- Mistral给tokenizer compression但不给model language mix；
- Meta Llama4的multimodal tokens缺模态编码定义；
- Google WebLI是研究dataset，不等于Gemini生产corpus；
- 开放text控制只能校准token/data版本，不能覆盖image/audio rights。

## 低资源语言采样

temperature sampling常写为：

$$
p_\ell=\frac{n_\ell^\alpha}{\sum_j n_j^\alpha},\qquad 0<\alpha<1,
$$

其中$n_\ell$需说明按tokens、bytes还是documents。降低$\alpha$提高低资源相对权重，但增加重复和过拟合风险。

最低报告：

- pre/post-filter raw与retained量；
- tokenizer fertility；
- unique与sampled exposure/epochs；
- cross-language translated/parallel比例；
- per-language held-out loss和macro；
- low-resource memorization/PII与safety。

Apple 8%→30%是curriculum总bucket；其内temperature sampling仍是另一层分布。两者不可合并成每语比例。

## LID、script 与cross-language dedup

语言识别对短文本、code-switching、转写和低资源语种误差更高。每条document宜保留概率分布而非单label。

dedup至少包括：

```text
within-language exact/near
cross-script normalization
translation/parallel semantic clusters
code-switched segment groups
```

翻译衍生文本增加目标语tokens，不增加独立事实source；train/eval应按共享祖先group split。

## 图像数据 lineage

```text
source image/url/license
  -> exact + perceptual dedup cluster
  -> paired alt-text / surrounding text
  -> synthetic caption(s)
  -> OCR/grounding/QA derivatives
  -> sampled sequence + image embeddings
```

Apple 的 >10B crawl pairs、>5B captions、175M interleaved docs/>550M images、>6B vision pairs 可能重叠，必须根据 lineage 判断重叠关系，不能直接相加。

对$W\times H$图像和patch $P$：

$$
N_{patch}\approx\left\lceil W/P\right\rceil\left\lceil H/P\right\rceil.
$$

实际模型可能resample为固定embeddings（Apple 144）或加入special tokens；需记录encoder version与loss mask。

## OCR、文档与图表

OCR数据不是普通caption子集。应分：

- scene/printed/handwritten；
- language/script；
- document layout/table/formula；
- transcription、QA、grounding任务；
- synthetic render vs real capture；
- exact document/revision group。

HunyuanOCR的130+ languages和Apple支持15-language OCR分别是dataset/model范围声明，不可直接比较每语exposure。

## 视频与音频

视频需同时报告unique videos、duration、sampled clips、fps、frames、visual tokens与字幕/ASR来源。相邻clips必须按source video group split。

音频需报告hours、speakers、locales、transcripts、codes/frames与consent/voice rights。2026 AFM 3 Core Advanced的audio capability只说明specialization方向，训练数量保持unknown。

## rights 与隐私

公开URL不等于可训练/可再分发许可。多模态还涉及：

- copyright与publisher opt-out；
- faces、biometric/identity与location；
- minors、medical/financial screens；
- music/performance与voice likeness；
- synthetic derivative的source/teacher rights；
- licensed asset能否进入开放weights/data。

manifest至少记录source terms snapshot、license interpretation、opt-out/delete状态和derivative lineage。

## validation

不得只报global score。最低矩阵：

```text
language × locale × task
modality × source type × resolution/duration
natural vs translated vs synthetic
seen-script vs unseen/code-switched
```

同时报告micro与macro。翻译eval需标machine translation、native refinement与translationese；image eval按source image group去污染。

## 看似矛盾的说法怎样区分

### 更低fertility=更好语言能力

- 首个差异：表示效率与建模质量。
- 检查：固定bytes/compute、per-language loss和downstream；同时扫sampling。

### 更多pairs=更多unique images

- 首个差异：一图多caption/QA与重复sampling。
- 检查：perceptual hash cluster、parent image count、derived-per-parent分布。

### 多模态训练损伤text vs replay可保持

- 首个差异：text replay、encoder初始化、mixture和loss normalization。
- 检查：fixed-token mixture sweep，报告text/mm per-domain Pareto。

### translation扩展低资源 vs translationese偏移

- 检查：natural-only held-out、native human eval、translated-source classifier与parallel ancestor split。

## 不可外推

- 不跨tokenizer直接比较language token totals；
- 不把pairs/images/docs/embeddings相加；
- 不从language list反推mixture；
- 不从支持窗口/分辨率反推训练分布；
- 不把WebLI等研究dataset自动映射到生产模型；
- 不把公开可访问当许可充分。

## 最少需要保存哪些 manifest 信息

```yaml
source_group: ...
source_uri_and_terms: ...
language_probs: {...}
raw_units: {bytes: ..., images: ..., seconds: ...}
derivative_type: alt_text|caption|ocr|qa|translation
parent_ids: [...]
encoder_or_tokenizer: ...
model_positions: ...
sampling_weight: ...
loss_positions: ...
dedup_cluster: ...
eval_overlap: ...
```

## 读完后应该能回答的问题

读者应能：

1. 同时报告fertility与language exposure；
2. 区分image、pair、document、embedding与loss token；
3. 解释Apple 10B/5B/6B不可直接相加；
4. 将8%→30%与80:20分配到P2和A1/A2；
5. 为translation、video clips和OCR选择祖先group；
6. 设计text/mm mixture Pareto实验。

推理题：同一100M images各有2个alt-text和3个synthetic captions，训练每个pair平均采样1.5次。分别给出unique images、pair records和sampled pair exposures；还缺什么才能算model/loss tokens？

## 来源与建议阅读位置

1. [Apple AFM](/topics/pretraining-data/apple-foundation-models-data-practices/)：读tokenizer、multilingual curriculum、image lineage与multimodal mix。
2. [Google Gemini/Gemma](/topics/pretraining-data/google-gemini-gemma-data-practices/)与[Meta Llama](/topics/pretraining-data/meta-llama-data-practices/)：比较closed taxonomy、open-weight token ladder和multimodal口径。
3. [ByteDance Seed](/topics/pretraining-data/bytedance-seed-doubao-data-practices/)、[Tencent Hunyuan](/topics/pretraining-data/tencent-hunyuan-hy-data-practices/)、[Xiaomi MiMo](/topics/pretraining-data/mimo-data-practices/)：读 stage、pair、token 和 task 的分列统计方法。
4. [Mistral](/topics/pretraining-data/mistral-ai-data-practices/)：用Tekken和Pixtral学习“tokenizer/sequence evidence不等于model mixture”。
5. [OLMo/Dolma](/topics/pretraining-data/ai2-olmo-dolma-data-practices/)与[OpenELM](/topics/pretraining-data/apple-foundation-models-data-practices/)：开放text控制。
6. [Data metrics](/topics/pretraining-data/data-metrics/)：统一分母、coverage和失败模式。
