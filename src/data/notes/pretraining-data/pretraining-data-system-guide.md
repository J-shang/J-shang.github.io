---
title: "Pretraining Data 系统学习指南"
description: "从数据单位、处理管线和采样配方，到训练期诊断、validation 与证据边界的完整学习路线。"
topic: "pretraining-data"
section: "guide"
slug: "pretraining-data-system-guide"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
featured: true
order: 0
readtime: 24
source:
  repository: "J-shang/pt-data-learning"
  path: "README.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/README.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:4ce44ce3e80fe49ca4588a30af13cd372b33a38dd4d135e229e0cea65132c6ec"
  manifest: "pretraining-data"
  managed: true
---
> 范围：以 decoder-only 语言模型为主，系统学习预训练数据的构建、度量、训练期诊断与 validation；兼顾代码、多语言和合成数据，但不把 post-training 数据工程作为主线。
> 初始资料核查截止：**2026-07-14**
> 当前阶段：**六篇主线枢纽笔记、大厂数据实践方法、22个组织/家族案例与6个横向专题已建立；下一轮按复核队列深化概念和实验。**

## 0. 一页结论

预训练数据不是“抓取文本后喂给模型”，而是一套带反馈的数据系统：

```text
目标能力与约束
  -> 数据源/许可/时间边界
  -> 抽取与标准化
  -> 质量过滤、去重、隐私与安全处理
  -> tokenizer 与 token 预算核算
  -> domain mixture、采样与排序
  -> train/validation 冻结与去污染
  -> 小规模 proxy / ablation
  -> 正式训练中的分域指标与异常诊断
  -> downstream evaluation 与误差归因
  -> 反哺下一版数据配方
```

最重要的五个区分：

1. **数据量不等于有效数据量**：raw bytes、文档数、字符数、unique tokens、sampled tokens、non-padding loss tokens 不能混用。
2. **过滤不等于质量**：规则/分类器只能实现一个可观测的代理目标；过强过滤可能损失多样性、方言、少数语言或困难样本。
3. **去重不等于去污染**：去重处理训练数据内部冗余，decontamination 处理训练数据与 validation/test/benchmark 的交叉泄漏。
4. **训练 loss 不等于数据价值**：低 loss 可能意味着容易、重复或模板化；高 loss 可能意味着噪声，也可能是稀有且有价值的知识。
5. **一个全局 validation loss 不够**：至少需要固定的分域 held-out loss、数据管线健康指标，以及与目标能力对应的 downstream eval。

学习本主题的目标不是背诵数据集名单，而是能回答并验证：

- 每个训练 token 从哪里来，经历了什么变换，为什么被保留？
- 同样的 token/compute 预算下，改变过滤器、去重范围或 mixture 后，哪类能力改善，哪类能力退化？
- 一个指标的分母、统计单位和 tokenizer 是什么？它能发现什么，又会漏掉什么？
- validation 是否独立、冻结、可比较、没有被训练数据或调参过程污染？
- 如何用小模型/proxy 实验降低大训练前的数据决策风险？

### 本项目如何推理

不同问题使用不同的主要 reasoning path，避免把所有主题都写成“定义 + 名词列表”：

| 学习任务 | 主要 reasoning path | 核心检查 |
|---|---|---|
| Token 核算 | Constraint-driven derivation | 从 batch/mask 约束推导计数，并用人工 fixture 回算 |
| 数据生命周期 | Implementation-trace | 固定 build/version，追踪 input → state → output 与 lineage |
| Filtering/dedup/decontam | Method-family trace | 统一比较对象、操作、状态、实现和失败模式 |
| Mixture 与采样 | Unified-framework | 用目标 exposure 分布 $q$ 映射各种 sampler，并标出近似边界 |
| 数据指标诊断 | Phenomenon-to-mechanism | 从异常模式列竞争解释，用 slice/ablation 区分 |
| Validation | Constraint-driven derivation | 从独立性、可比性和决策目标推出 split/metric contract |

结论使用四级置信状态：`verified`、`supported`、`plausible`、`open`。标签只在声明的对象、版本和假设范围内有效；`supported` 不等于跨模型普遍因果规律。

## 1. 项目边界

### 主线覆盖

- 语料来源、许可、治理、provenance 和版本化。
- HTML/WARC/PDF/代码等内容的抽取、规范化、语言识别和文档边界。
- 文档/段落/句子级质量信号、规则过滤、模型过滤与拒绝分析。
- exact、MinHash/LSH、substring 等去重方法及其系统代价。
- benchmark decontamination、时间切分与泄漏诊断。
- tokenizer fertility、序列 packing、token accounting 和有效训练 token。
- 静态/动态 domain mixture、温度采样、上/下采样与 curriculum。
- 数据管线指标、训练时分域 loss、梯度/优化信号与异常监控。
- validation 集构造、micro/macro 汇总、置信区间、proxy 训练与 ablation。

### 暂不作为主线

- SFT、preference optimization 和 RL 数据（只讨论与预训练数据边界相关之处）。
- 模型架构、并行训练和优化器的完整教程（只学习其对数据指标解释的必要部分）。
- 多模态数据工程的全部细节（未来作为扩展层）。

### 独立研究支线

- [大厂 Pretraining Data 实践](/topics/pretraining-data/)：以统一字段和证据等级研究 OpenAI、Anthropic、Gemini/Gemma、Llama、DeepSeek、Qwen、GLM、Kimi、MiMo、MiniMax、StepFun、MAI/Phi、Mistral、Apple、NVIDIA 等模型家族。首轮厂商档案和六个横向专题已完成；逐字段证据卡见 [disclosure-scorecard.md](/topics/pretraining-data/data-disclosure-scorecard/)，维护队列见 [TASKS.md](https://github.com/J-shang/pt-data-learning/blob/main/industry-data-practices/TASKS.md)。

## 2. 必备知识地图

完整索引见 [knowledge-map/README.md](/topics/pretraining-data/)。

| 层级 | 关键问题 | 当前学习产出 |
|---|---|---|
| 00 度量基础 | “一个 token/样本/epoch”到底指什么？ | 能做 token 与有效训练量核算 |
| 01 数据生命周期 | 数据如何从 source 变成可训练 sequence？ | 能画出带 lineage 的 pipeline |
| 02 筛选与风险 | 如何过滤、去重、去污染且知道副作用？ | 能设计 stage metrics 与拒绝审计 |
| 03 配比与采样 | 每个 domain 被看到多少次、以什么顺序看到？ | 能推导曝光量和实现 sampler |
| 04 指标与诊断 | 数据问题如何在训练前/中/后显现？ | 能构建 data observability 面板 |
| 05 Validation | 怎样得到可信、可归因的反馈？ | 能冻结多层 validation protocol |
| 06 实证方法 | 如何用有限 compute 比较数据方案？ | 能设计 paired ablation/proxy 实验 |
| 07 治理与前沿 | 合规、隐私、合成数据和自适应选择如何处理？ | 能标证据边界与开放问题 |

最低前置知识：交叉熵与最大似然、基础概率统计、语言模型 tokenization、训练/验证/测试划分。不会时从 `00-foundations` 开始，不需要先学完全部 Transformer 架构。

## 3. 主路线图

### 阶段 A：建立共同计量语言

输入：一个数据 manifest、tokenizer 与训练配置。
学习：文档/字符/字节/token/sequence/step 的换算，unique 与 sampled 的区别，padding 与 mask。
输出：一张 token accounting 表和两条 invariant。
检查：能解释为什么“训练了 1T tokens”不足以复现实验。

入口：[Token 核算与数据单位](/topics/pretraining-data/token-accounting/)

### 阶段 B：理解端到端数据生命周期

输入：一个开放语料的 pipeline 描述（建议 FineWeb、Dolma 或 DCLM）。
学习：source snapshot、抽取、规范化、过滤、去重、分片、tokenization、packing、manifest。
输出：带 stage 输入/输出计数、版本和哈希的 pipeline DAG。
检查：任意训练 sequence 能回溯到原始 document 和处理配置。

入口：[预训练数据生命周期](/topics/pretraining-data/data-lifecycle/)

### 阶段 C：掌握数据筛选的机制与反事实

输入：raw/accepted/rejected 三类样本和 benchmark 集。
学习：heuristic 与 classifier filtering、exact/near/substr dedup、decontamination。
输出：过滤漏斗、每条规则的拒绝原因分布、重叠率和人工误差分析。
检查：能说明删除一个文档后，数据规模、多样性、memorization 风险和 eval 可信度如何变化。

入口：[质量过滤、去重与去污染](/topics/pretraining-data/filtering-dedup-decontamination/)

### 阶段 D：学习 mixture 与训练期曝光

输入：各 domain 的可用 token 数与目标权重。
学习：自然采样、温度采样、上/下采样、epoch/曝光量、静态与动态 reweighting。
输出：一个可审计的 sampler 配置与预期/实际 exposure 对照。
检查：能计算小域被重复多少次，并判断这是否引入过拟合或记忆风险。

入口：[数据配比与采样](/topics/pretraining-data/data-mixtures-and-sampling/)

### 阶段 E：把数据变成可观测系统

输入：stage logs、训练 logs、固定 validation suites。
学习：coverage、yield、重复率、fertility、分域 loss、loss gap、漂移与不确定性。
输出：最小 data observability dashboard 规范。
检查：一个全局 validation loss 异常时，能向 tokenizer/domain/source/stage 逐层定位。

入口：[训练数据指标体系](/topics/pretraining-data/data-metrics/)

### 阶段 F：用 validation 和实验闭环

输入：候选数据配方、固定模型/compute 和冻结评测协议。
学习：in-distribution held-out、per-domain validation、downstream eval、contamination audit、paired ablation。
输出：一份 validation contract 与第一个小规模实验报告。
检查：能把“结果更好”拆成差异、置信度、成本、受益域、受损域和可能机制。

入口：[Validation 设计与诊断](/topics/pretraining-data/validation-design/)；实验模板见 [experiments/README.md](https://github.com/J-shang/pt-data-learning/blob/main/experiments/README.md)。

## 4. 核心对象与公式

以下关系只在紧邻说明的定义与假设内成立。公式本身可以是 exact identity，但从某个指标变化推断“数据更好”通常只属于 empirical association，需要受控 ablation 才能加强为因果解释。

### 4.1 训练目标

对 token 序列 $x_{1:T}$，causal LM 的平均 negative log-likelihood（NLL）为：

$$
L = -\frac{1}{M}\sum_{t=1}^{T} m_t\log p_\theta(x_t\mid x_{<t}),
\qquad M=\sum_{t=1}^{T}m_t
$$

$m_t\in\{0,1\}$ 是 loss mask；$M$ 是真正进入 loss 分母的 token 数。数据管线中的 padding、文档边界 token、packing 方式和重复样本都会改变 $M$ 或每个 token 的上下文，却未必反映在名义 token budget 中。

### 4.2 数据 mixture

将数据划分为 $K$ 个 domain，目标 mixture 为 $q=(q_1,\ldots,q_K)$：

$$
q_k\ge 0,\qquad \sum_{k=1}^{K}q_k=1,\qquad E_k=T_{\text{train}}q_k
$$

$E_k$ 是 domain $k$ 的预期 sampled tokens。若它只有 $N_k$ 个 unique tokens，则近似曝光倍数为 $r_k=E_k/N_k$。这比单独报告 $q_k$ 更能暴露小域反复采样。

### 4.3 Validation 汇总

分域 NLL 为 $L_k$。micro average 按 token 数加权，macro average 给各 domain 同等权重：

$$
L_{\text{micro}}=\frac{\sum_k M_kL_k}{\sum_kM_k},
\qquad
L_{\text{macro}}=\frac{1}{K}\sum_kL_k
$$

两者回答不同问题，必须与每个 $L_k$ 一起看；否则大 web domain 会淹没代码、数学或低资源语言的退化。

## 5. 资料与实现入口

完整的分层阅读队列见 [sources/README.md](https://github.com/J-shang/pt-data-learning/blob/main/sources/README.md)。第一轮建议：

1. **数据公开与可复现基线**：FineWeb、Dolma、DataComp-LM。
2. **去重机制与影响**：Lee et al., *Deduplicating Training Data Makes Language Models Better*。
3. **数据配比**：DoReMi；先理解 group/domain weight 与 proxy model，再看动态采样前沿。
4. **token/compute 边界**：Chinchilla；把它当受实验设定约束的 scaling 结果，不当永恒配方。
5. **实现阅读**：DataTrove 的 reader → filter → dedup → writer/summary statistics 流程。

## 6. 实践与实验队列

初始化后优先完成三个低成本闭环：

1. **Token accounting audit**：用两个 tokenizer 对同一小型中英代码混合语料计算 bytes/token、tokens/document、packing waste 和 domain exposure。
2. **Dedup 方法对照**：在人工植入 exact、near、substring 重复的小语料上比较 exact hash 与 MinHash 的 precision/recall、删除量和运行成本。
3. **Mixture paired run**：固定小模型、总 token 和 seed，对 natural mixture 与温度/手工 mixture 训练，比较各域 validation loss 曲线而非只看终点平均值。

每次实验从 [experiments/README.md](https://github.com/J-shang/pt-data-learning/blob/main/experiments/README.md) 复制模板，并先冻结问题、控制变量与成功标准。

## 7. 稳定基础、经验结论与前沿

### 相对稳定

- `[verified | 定义/推导]` 固定 tokenizer、mask 和对数底时，NLL 与 perplexity 是单调变换；训练/validation/test 的角色由测量合同定义。
- `[supported | 工程与复现实践]` 数据 lineage、版本化、分域统计和随机抽样审计提高可诊断性与可复现性。
- `[supported | empirical association]` 在已研究设置中，duplication 与 memorization/evaluation leakage 风险相关；效应大小不能脱离模型、重复次数和攻击/评测协议外推。
- `[verified | 聚合恒等式]` aggregate metric 可以在权重占优时掩盖 subgroup/domain failure；是否构成业务问题取决于决策目标。

### 强经验依赖

- 某个质量过滤器的阈值、某种去重粒度、最佳 domain mixture。
- 最优 token/parameter 比例、proxy 到目标模型的迁移性。
- “educational quality”“reasoning quality”等模型打分是否跨语言/领域可靠。

这些条目默认置信状态为 `plausible` 或 setting-specific `supported`；只有固定数据、模型、预算和评测协议的实验才能提升置信度。

### 前沿与开放问题

- 在线 data selection、importance sampling 与模型当前学习状态如何耦合且不引入偏差？
- 合成数据在多轮自训练中的 diversity collapse、错误强化和 provenance 如何测量？
- 如何把数据价值从相关性提升到可靠的因果归因？
- 如何联合优化能力、公平性、隐私、版权、能耗与数据获取成本？

## 8. 推荐学习顺序

1. 读根指南和 `00-foundations`，完成 token accounting 手算。
2. 读生命周期笔记，对照一个官方 pipeline 画 DAG。
3. 读筛选/去重/去污染笔记，做人工重复 fixture。
4. 读 mixture 笔记，计算三个 domain 的曝光倍数。
5. 读 metrics 和 validation，写一份最小 validation contract。
6. 选一篇开放数据论文，按 `sources/README.md` 的问题模板精读。
7. 跑一个 paired proxy experiment，把结果回填到知识地图。

## 9. 术语表

| 术语 | 本项目中的精确定义 |
|---|---|
| document | 数据管线中可独立追踪、过滤、去重的基本内容对象，不必等同网页 |
| domain | 为采样或分析定义的组，如 source、语言、体裁、时间段或质量桶 |
| unique tokens | 去重/冻结后的语料经指定 tokenizer 得到的 token 数 |
| sampled tokens | sampler 实际抽取的 token 数；可能多次经过同一 unique token |
| loss tokens | mask 后真正进入 NLL 分母的 token 数 |
| quality signal | 与目标数据价值相关的可观测特征或模型分数，不等同“真实质量” |
| deduplication | 检测并处理训练语料内部重复 |
| decontamination | 检测并处理训练语料与 validation/test/benchmark 的重叠 |
| mixture | 训练 token 在 domain 间的目标概率分布 |
| exposure | 一个 domain、document 或 token 在训练中被采样/看到的次数或期望次数 |
| validation contract | 在看结果前冻结的数据版本、指标、聚合、去污染、checkpoint 和决策规则 |
| exact identity | 在明确的定义和假设下严格相等；条件改变后必须重新检查 |
| approximation | 在声明的 regime 内忽略部分误差项后的近似关系 |
| empirical association | 两个量在给定研究中共同变化，但尚未建立普遍因果机制 |
| verified / supported / plausible / open | 本项目从强到弱的结论置信状态，始终绑定具体对象、版本和假设 |

## 10. 当前完成定义与下一步

初始化完成表示：目录约定清楚、路线可执行、核心概念有机制与公式、来源有入口、实验有模板；**不表示这个领域已经覆盖完毕**。

下一轮最有价值的扩展：

- 增加 `tokenization-and-packing.md`，细化 tokenizer fertility、文档边界与 sequence packing。
- 增加 `data-provenance-and-governance.md`，覆盖许可、删除请求、PII 与版本 lineage。
- 为 FineWeb、Dolma、DCLM 各写一篇 pipeline walk-through，并记录具体版本/commit。
- 实现小型 metrics/dedup 实验代码，使用人工 fixture 而非下载大语料。
