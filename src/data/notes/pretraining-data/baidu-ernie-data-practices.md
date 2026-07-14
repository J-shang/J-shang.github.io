---
title: "Baidu ERNIE / 文心的数据实践"
description: "按阶段拆分知识图谱任务、确定性数据流、masked multimodal loss 与 unified codec 的公开证据。"
topic: "pretraining-data"
section: "china-cases"
slug: "baidu-ernie-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 109
readtime: 27
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/china/baidu-ernie.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/china/baidu-ernie.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:248334b0e1f63834282c9866cd6a83b02f6c60f9d273dbeeeae153317bcfef9a"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family/historical trace`
> 辅助视角：`implementation-trace`，用于 ERNIE 4.5 REEAO 与 multimodal loss accounting
> 研究锚点：ERNIE 3.0、ERNIE 4.5、ERNIE-4.5-VL Thinking、ERNIE 5.0/5.1

## 定位

ERNIE 的代际演化不能简化成“百度一直使用知识增强”：

- ERNIE 3.0 把plain text、50M+ knowledge-graph facts与knowledge-text prediction任务显式结合；
- ERNIE 4.5 的“knowledge-based data”主要表现为五级knowledge labels、key-point synthesis、data map与human-model loop，同时转向text/vision/video联合训练；
- ERNIE 5.0 从训练起点统一text/image/video/audio的理解与生成，但只披露“trillions of text tokens and multimodal instances”，无法与3.0的4TB直接换算；
- ERNIE 5.1主要通过5.0 super-network抽取sub-network继承预训练知识，低pretraining cost不等于另一个小corpus recipe。

## Motivating problem：“知识增强”究竟是数据、标签、目标还是模型结构

至少要区分：

```text
knowledge graph facts
  -> aligned triple-sentence examples
  -> knowledge-text prediction objective

natural/synthetic corpus
  -> knowledge-level classifier
  -> data labels, mixture and staged selection

structured knowledge sources
  -> ordinary next-token / unified multimodal pretraining
```

三条链都可能被称为knowledge-enhanced，但它们的可检查锚点和失败模式不同。benchmark knowledge score不能反推出采用了哪条链。

## 代际与训练阶段表

| 模型 | 阶段 | 已披露数据与规模 | objective/context | 披露 | 状态 |
|---|---|---|---|---|---|
| ERNIE 3.0 10B | continual multi-paradigm `P0` | 4TB、11类中文数据；plain text + Baidu KG 50M+ facts | autoencoding、autoregressive、knowledge extraction任务 | `D2` | bytes/sources/tasks `verified`；tokens `open` |
| ERNIE 3.0 English | `P0` | Wikipedia、BookCorpus、CC-News、OpenWebText、Stories | 同family framework；精确mix/tokens未知 | `D2` | sources `verified`；accounting `open` |
| ERNIE 4.5 LLM | text-only `P0/P3` | diverse domains的trillions text tokens；精确总量/mixture未知 | short 8K；long 32K→128K，upsample >16K docs | `D2/D3` | stage/source classes `verified`；budget `open` |
| ERNIE 4.5 VL | vision `P0` | large image-text pairs；encoder→pre-alignment→integration | 8K；冻结/解冻范围逐段变化 | `D2/D3` | stages `verified`；pair/token counts `open` |
| ERNIE 4.5 VL | joint multimodal `P1/P3` | text+image+video；interleaved/pairs/synthetic/domain data | 8K→128K；full model；image/prompt tokens masked | `D2/D3` | modality/stage/loss `verified`；mix `open` |
| ERNIE 4.5 LLM | `A1/A2` | 2.3M SFT samples、平均2 epochs；三阶段progressive RL | logic→math/code→general；online/offline UPO | `D2/D3` | SFT count `verified`；RL accounting `open` |
| ERNIE 4.5 VL | `A1/A2` | three SFT stages；>10K verified visual puzzles；visual STEM/UI2Code | text cold start→vision rejection→think/no-think fusion→RLVR/RLHF | `D2/D3` | task/pipeline `verified`；total prompts/rollouts `open` |
| ERNIE-4.5-VL Thinking | mid-training + `A2` | high-level称large premium VL reasoning corpus；数量未知 | mid-training后GSPO/IcePop、dynamic difficulty | `D1/D3` | stage existence `verified`；recipe `open` |
| ERNIE 5.0 | unified multimodal `P0/P3` | multilingual web/curated/books/papers/code/structured knowledge；image/video/audio paired+interleaved；trillions级 | from scratch；8K→128K；NTP/NFSP/NCP统一目标 | `D2` | categories/stages `verified`；exact budget `open` |
| ERNIE 5.0 | `A2/A3` | multimodal RL；prompt/rollout/loss规模未知 | U-RB、MISC/WPSM、AHRL hint schedule | `D2` | mechanisms `verified`；data counts `open` |
| ERNIE 5.1 | inherited `P0` + new `A*` | 从5.0 elastic sub-model matrix抽取；称约6% comparable pretraining cost | 非独立from-scratch recipe | `D1` | lineage `verified`；data diff `open` |

## Assumption ledger

| 维度 | 本篇约束 |
|---|---|
| 统计单位 | 4TB storage、50M facts、trillions text tokens、multimodal instances、2.3M SFT samples、RL prompts分开 |
| knowledge | KG triple、knowledge label、key point、structured source与memorized fact分开 |
| modality | text/image/video/audio的tokenizer、instance、loss mask与generation target分别记录 |
| “128K” | Stage I/III training context与产品context分开；只有报告明确训练的阶段才记exposure |
| 4.5 lineage | LLM可从joint multimodal model删除vision modules得到，但text-only post-training不等于纯text P0 |
| 5.1 lineage | 低pretraining cost来自5.0 super-network继承；不是凭空用6%数据重训 |
| rights | 模型Apache 2.0不覆盖training corpus；web/search/proprietary education等source rights仍需逐源核对 |
| cutoff | 4.5/5.0完整training cutoff未披露；source页面访问日期不替代snapshot |
| 省略效应 | architecture、objective、tokenizer、modality、context、routing、data与RL跨代同时变化 |

## 统一数据字段

| 字段 | ERNIE 3.0 | ERNIE 4.5 | ERNIE 5.0/5.1 |
|---|---|---|---|
| source | Baike/Wikipedia/feed、Baidu Search产品、web/QA/poetry、medical/legal/finance、Baidu KG | web/papers/documents/images/videos/synthetic；domain/audio transcription；open/proprietary post data | multilingual web、curated corpora、books、science、code、structured knowledge；paired/interleaved multimodal |
| rights | public与Baidu-owned/controlled sources并存；逐源license未知 | web/domain/proprietary K-12等权利分布未知 | source-level rights与consent mix未知 |
| cutoff | `unknown` | `unknown` | `unknown` |
| token accounting | 4TB、11类、50M+ facts；token exposure/loss未知 | “trillions”；阶段token、modality mix、repeat/loss未给 | trillions text tokens + multimodal instances；精确转换未知 |
| quality | task construction；corpus清洗阈值不完整 | heuristic/model filters、labels/clusters、human-model loop、recaption | heuristic/model safety filter、dedup、decontam高层说明 |
| dedup | 未给现代pipeline级阈值 | text/image/video/audio dedup；pair image+text dedup；阈值/retention未知 | extensive dedup + benchmark decontam；方法/retention未知 |
| order | continual multi-task，允许新增task | REEAO bitwise-deterministic records；三阶段multimodal schedule | 8K→128K；具体data order/seed未开放 |
| validation | 54 tasks、UKTP ablation、progressive-learning timing | base/VL/post benchmarks、router ablation；pretrain validation contract不完整 | held-out elastic validation + broad release eval；data ablation有限 |
| artifacts | papers、早期Paddle models；4TB/KG snapshot未开放 | base/post weights、ERNIEKit/FastDeploy、pretrain implementation；无corpus/order config | report；5.0 weights/data开放边界低，5.1产品披露 |

## ERNIE 3.0：知识图谱不是“高质量网页”的同义词

### 4TB、11类与50M+ facts

`[来源事实 | verified]` ERNIE 3.0中文corpus为4TB storage、11类，来源包括ERNIE 2.0语料（Baike、Wikipedia、feed等）、Baidu Search中的Baijiahao/Zhidao/Tieba/Experience、web text、long/short QA、poetry/couplet、medical/legal/finance，以及Baidu knowledge graph的50M+ facts。

这些量不能相加：

$$
S_{\text{text+KG}}=4\text{TB},\qquad N_{\text{KG facts}}>50\text{M},
\qquad T_{\text{sampled}}=\text{unknown}.
$$

$S$是storage size，$N$是fact count；tokenizer、serialization、epochs和loss mask未知。

### triple-sentence alignment与UKTP

`[来源事实 | verified]` Universal Knowledge-Text Prediction（UKTP）从百科文档中找与标题实体相关的KG triples，再保留head/tail entity同时出现在同一句的triples；训练时随机mask triple relation或sentence words。模型需利用实体mention与句子来预测关系/文本。

```text
encyclopedia document title/entity
  -> retrieve candidate KG triples
  -> keep triple when head and tail co-occur in sentence
  -> mask relation or words
  -> joint knowledge-text prediction
```

`[来源事实 | verified]` 小规模ablation中加入UKTP后，SanWen relation extraction从75.56到77.36，FinRE从58.19到59.75。

`[综合判断 | supported]` 该结果支持named tasks与setup内的empirical association；不能证明4TB corpus的所有收益来自KG，也不能外推到next-token-only LLM。

### continual multi-paradigm 的数据顺序问题

ERNIE 3.0在shared universal module上持续加入NLU、NLG、knowledge-extraction任务，并保留task-specific modules。这里order unit是`task batch`而不只是document。复现需要task scheduler、task sampling、masking seed与各task exposure；4TB source list不足以复现训练分布。

## ERNIE 4.5：data map与确定性数据消费

### 从过滤到human-model loop

`[来源事实 | verified]` 4.5 source classes包括web pages、academic papers、documents、images、videos和synthetic。pipeline包含heuristic dedup/low-quality removal、model-based quality filtering、labeling/clustering、data map和human-model-in-the-loop refinement。

data map按language、knowledge、domain、application scenario和quality组织，用于mixture、staged selection与model monitoring。它是多维manifest/feature space，不是将质量压成单一分数。

### knowledge-level labels与key-point synthesis

`[来源事实 | verified]` 4.5借用DIKW思路定义五级knowledge并训练classifier标注pretraining data；高价值数据稀少，于是从textbooks与educational websites抽取structured key points，再生成math/factual/code数据。reasoning data还按type与difficulty组织并过滤。

与3.0的关系是：

```text
ERNIE 3.0 KG triple --used-in-objective--> UKTP
ERNIE 4.5 knowledge label --used-for--> data analysis and mixture
key point --conditions-generation-of--> synthetic sample
```

这是method-family evolution，不是exact identity。

### 多模态source pipeline

- interleaved data：高质量web pages/documents + video keyframes/ASR；页内image-text dedup、低分辨率/无关内容/乱码/广告/顺序错误过滤；
- image-text pairs：CLIP-like similarity、image+text dedup、scene/table/screenshot/chart/document labels、subset recaption；
- grid understanding：拼接多图并按结构拼caption；
- domain data：progressive mining/conditional pretraining；video soundtrack/podcast经ASR、rewrite与filter形成口语数据。

`[未知 | open]` 每条管线的raw/retained pairs、阈值、language/domain distribution、source rights与交叉模态near-dedup未公开。

## REEAO：确定性顺序是数据资产的一部分

`[来源事实 | verified]` REEAO把multimodal sources切成fixed-length records，在data configuration确定后、训练开始前固定bitwise-deterministic token sequence，并独立于分布式配置记录source consumption。即使node count、parallel strategy、global batch、context或online data updates变化，也防止重复/遗漏。

最小invariant：

```text
same data config + same record mapping
  -> same global token sequence
  -> independent of worker topology and recovery path
```

`[综合判断 | supported]` 这解决的是data delivery reproducibility，不证明upstream corpus本身可公开复现。还需source snapshot、parser/filter/tokenizer、record manifest与seed。

## 4.5 三阶段multimodal recipe与token-balanced loss

### Stage I–III

```text
Stage I text-only:
  trillions text tokens @8K -> long documents upsampled, 32K -> 128K

Stage II vision-only:
  vision encoder -> adapter/vision experts pre-alignment -> vision integration

Stage III joint:
  full model, text+image+video @8K -> full model multimodal @128K
```

Stage II先用小LM+large image-text pairs训练ViT；pre-alignment冻结LLM/ViT，只训adapter/vision experts/router；integration再解冻ViT并upsample high-quality captions/alt text。

`[未知 | open]` 报告没有给每段token、pair、step、repeat或mixture，所以只能记录阶段顺序，不能计算总sampled exposure。

### loss tokens与sequence tokens不能混用

设样本$i$的masked positions为$M_i$，unmasked output text positions为$U_i$。传统per-sample loss为：

$$
L^{(i)}=-\frac{1}{|U_i|}\sum_{j\in U_i}\log P(y_j^{(i)}\mid y_{<j}^{(i)};\theta).
$$

4.5明确image tokens与prompt positions被mask，不进入cross-entropy。不同样本的$|U_i|$比例差异会使少量unmasked tokens的样本梯度被放大，因此token-balanced loss改为：

$$
L_{\text{balanced}}^{(i)}=
\frac{1}{|U_i|+|M_i|}\sum_{j\in U_i}\ell_j^{(i)}.
$$

`[推导结论 | verified]` 分母是total sequence positions，求和仍只含unmasked loss tokens。这是`implementation relation`，不是把image token变成language target。

审计时必须同时报告sequence tokens、unmasked loss tokens、modality tokens与per-sample normalization。

## 4.5 post-training：2.3M SFT与动态RL分母

`[来源事实 | verified]` LLM SFT覆盖十类domain、reasoning/non-reasoning，并为部分query保留多个不同reasoning chains；最终2.3M samples、平均2 epochs。RL按logic→math/code→general progressive stages，UPO把DPO pairwise loss整合进PPO；online pair来自当前iteration多response rejection sampling，offline pair提前生成。

prompt selection排除accuracy恒为0或1的verifiable prompts，再按intra-group reward variance去除无区分力样本，并按domain独立normalize reward。

`[综合判断 | supported]` 2.3M×2只是SFT nominal exposure近似，不能与RL prompts相加；online-UPO还需要query→responses→preference pairs→accepted trajectories→loss tokens lineage。

VLM使用text-only cold start→vision reject sampling→thinking/non-thinking fusion；RLVR涵盖visual STEM、>10K synthetic visual puzzles与UI2Code。visual STEM同时包含open-source与proprietary K-12 sources，意味着权利与decontam要按子源记录。

## ERNIE 5.0/5.1：统一目标扩大了accounting维度

### 5.0 data foundation

`[来源事实 | verified]` 5.0从头同时看到text、image、video、audio。text包括multilingual web crawl、curated corpora、books、scientific publications、code repos与structured knowledge；multimodal包括image-text、video-text、audio-text pairs和含metadata/caption的interleaved sequences。

官方只称final corpus含“trillions of text tokens and multimodal instances”，未给精确token或instance数。heuristic/model filters移除low-quality/unsafe，使用extensive dedup和benchmark decontamination，但方法、阈值、retention与benchmark列表未公开。

统一训练目标实际包含不同统计对象：

| 模态 | objective | 最小accounting |
|---|---|---|
| text | NTP + MTP | input/output/loss tokens |
| image/video | Next-Frame-and-Scale Prediction | frames、scales、visual codes、loss positions |
| audio | Next-Codec Prediction | duration、codec levels/codes、loss positions |

`[综合判断 | supported]` “shared token space”不意味着所有模态token成本/语义粒度相同。跨模态mixture必须保留原始单位和codec/tokenizer版本。

### U-RB 与 AHRL是data-order机制

U-RB在异步rollout中为query预分配iteration group，允许后续batch提前生成，但只有当前$D_t$完整后才进入training pool；它减少长response阻塞，同时避免只训练短/easy trajectories造成的difficulty drift。

AHRL把reference thinking trajectory的前$p_{\text{hint}}$比例注入query，并随训练退火：

$$
p_{\text{hint}}(x,t)=p_{\text{initial}}
\exp\left(-\gamma t\,\operatorname{pass}_{\text{initial}}(x)\right).
$$

变量$t$为iteration，$\gamma$为decay，initial pass来自SFT model。hint是training-time scaffold，需要保存source trajectory、revealed prefix、iteration与最终无hint评估，防止将答案泄露当作能力。

### 5.1不是独立小预算from-scratch对照

`[来源事实 | verified]` 5.1从5.0的multi-dimensional elastic sub-model matrix抽取sub-network，继承5.0知识，官方称约6% comparable models的pretraining cost。这个数字是compute comparison，不是“使用5.0数据的6%”。

`[未知 | open]` 5.1是否做额外continued pretraining、用了哪些post-training data与5.0 corpus的何种exposure，当前发布稿未量化。

## Validation、污染与区分性检查

| 问题 | 首个可能差异 | 区分性检查 |
|---|---|---|
| KG增强是否有效 | triple objective vs更多domain text | fixed-text/fixed-compute add/remove aligned triple-sentence task |
| knowledge-level synthesis是否改善长尾 | key-point coverage vs重复/teacher style | source-key-point group split、coverage/repetition/factuality联合报告 |
| REEAO是否真正deterministic | topology/recovery vs upstream mutable data | 固定manifest哈希，跨node/batch/restart比较global token hashes |
| multimodal joint training是否互促 | data mix、routing、loss normalization | fixed-compute text-only/vision-only/joint + per-domain macro |
| token-balanced loss改善来自何处 | gradient scale vsmodality distribution | 同data/order，对比per-loss-token、per-sequence-token与balanced normalization |
| AHRL是否只是hint依赖 | train scaffold vs autonomous reasoning | 按hint fraction/iteration，最终0-hint clean evaluation |

ERNIE 4.5公开了base/post benchmark与router loss fixed-data ablation，5.0也报告elastic submodel held-out loss；但完整validation source、time boundary、split unit、token counts、decision exposure和跨模态decontam manifest仍缺。

## 可迁移经验与不可外推部分

### 可迁移

- `[综合判断 | supported]` “知识增强”要拆成source、alignment、label、objective和mixture五层。
- `[综合判断 | supported]` distributed data manager必须保证global token order与topology/restart解耦，并保存source consumption ledger。
- `[综合判断 | supported]` masked multimodal training应报告sequence与loss分母；token-balanced loss给出明确operational anchor。
- `[综合判断 | supported]` 同一source跨caption/parsing/reasoning任务时，应按source group做validation split。
- `[综合判断 | supported]` inherited super-network compute不能解释成from-scratch data efficiency。

### 不可外推

- `[未知 | open]` 4.5/5.0的精确tokens、mixture、repeat、cutoff、source rights和loss accounting；
- `[未知 | open]` 3.0的4TB如何tokenize/serialize/采样，KG task占总update多少；
- `[未知 | open]` 4.5 VL Thinking mid-training corpus规模及与base 4.5的重叠；
- `[待验证假设 | plausible]` 五级knowledge classifier在不同语言/domain中具有可比校准；
- `[待验证假设 | plausible]` 5.0的unified codec space不会在低资源模态上形成隐性gradient underweighting。

## 掌握标准

读完后应能：

1. 解释4TB、50M facts、trillions tokens与multimodal instances为什么不可直接比较；
2. 用流程说明ERNIE 3.0 UKTP如何构造triple-sentence pairs；
3. 说明REEAO解决了什么、不解决什么；
4. 写出token-balanced loss并指出masked image/prompt仍不进入loss；
5. 解释5.1“6% pretraining cost”为何不是6% corpus。

## 推理型自测

1. 若knowledge benchmark提升，怎样区分KG objective、key-point synthetic与更多high-value text？
2. 两次4.5训练最终权重相同，但raw web snapshot不同，能否称data reproducible？为什么？
3. 一个样本有90% image/prompt tokens、另一个有10%。传统per-unmasked-token与balanced loss如何改变样本权重？
4. AHRL在有hint训练中提升很快，但0-hint下降。应怎样调整curriculum与validation？

## 来源与建议阅读位置

1. [ERNIE 3.0 paper](https://arxiv.org/abs/2107.02137)
   - 为什么读：固定4TB/11类/50M+ facts、UKTP构造、continual multi-paradigm与ablation。
   - 建议位置：§3.2–3.3、§4 corpus说明、§5 UKTP/progressive learning。
2. [ERNIE 3.0 Titan](https://arxiv.org/abs/2112.12731)
   - 为什么读：确认knowledge-enhanced framework向260B参数扩展与online distillation方向；数据定量仍不足。
   - 建议位置：model/training、online distillation、data disclosure。
3. [ERNIE 4.5 Technical Report](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)
   - 为什么读：核对data map、REEAO、三阶段multimodal、token-balanced loss、2.3M SFT与RLVR。
   - 建议位置：§3、§4、Appendix A ablations。
4. [ERNIE 4.5 repository](https://github.com/PaddlePaddle/ERNIE)
   - 为什么读：核对base/post weights、ERNIEKit pretraining implementation、training support与Apache license。
   - 建议位置：model table、ERNIEKit、release history、license。
5. [ERNIE 4.5 open-source announcement](https://ernie.baidu.com/blog/posts/ernie4.5/)
   - 为什么读：固定10个开放模型、joint multimodal与artifact边界。
   - 建议位置：highlights、model development、license。
6. [ERNIE-4.5-VL Thinking announcement](https://ernie.baidu.com/blog/posts/ernie-4.5-vl-28b-a3b-thinking/)
   - 为什么读：确认base后存在独立multimodal mid-training和RL阶段，同时暴露定量缺口。
   - 建议位置：Model Highlights、training与RL说明。
7. [ERNIE 5.0 Technical Report](https://arxiv.org/abs/2602.04705)
   - 为什么读：固定from-scratch unified multimodal source taxonomy、dedup/decontam、U-RB与AHRL。
   - 建议位置：§3.1、§4.1–4.3、elastic validation appendix。
8. [ERNIE 5.0 official overview](https://ernie.baidu.com/blog/posts/ernie5.0/)
   - 为什么读：核对trillions级data、8K→128K、NTP/NFSP/NCP和产品披露边界。
   - 建议位置：Data Foundation、Training、Post-Training。
9. [ERNIE 5.1 release](https://ernie.baidu.com/blog/posts/ernie-5.1-0508-release/)
   - 为什么读：确认5.1从5.0 elastic matrix抽取，避免把6% cost误写成data ratio。
   - 建议位置：model lineage与pretraining cost段落。

## 与主线的关系

```text
knowledge graph fact --aligned-with--> encyclopedia sentence
knowledge label --used-for--> data map and mixture
REEAO record ledger --implements--> deterministic global token order
loss mask --distinguishes--> sequence tokens vs loss tokens
source group split --prerequisite-for--> cross-task contamination control
super-network inheritance --does-not-imply--> independent low-data pretraining
```
