---
title: "预训练数据生命周期"
description: "把 source、解析、过滤、tokenization、sharding、采样和训练反馈连接成有版本、可追溯的数据流。"
topic: "pretraining-data"
section: "pipeline"
slug: "data-lifecycle"
date: 2026-07-14
updated: 2026-07-14
order: 20
readtime: 16
source:
  repository: "J-shang/pt-data-learning"
  path: "knowledge-map/01-pipeline/data-lifecycle.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/48b6c6907a65afc718659f922895f835335be1d3/knowledge-map/01-pipeline/data-lifecycle.md"
  revision: "48b6c6907a65afc718659f922895f835335be1d3"
  syncedAt: "2026-07-14"
  contentHash: "sha256:d533079c5c6e66f9e21b8b7d2c8a75ef769095599fb1383abb3ccc13a2085f6f"
  manifest: "pretraining-data"
  managed: true
---
> 层级：01 Pipeline
> 状态：`core`
> 初始资料核查截止：2026-07-14
> 主要 reasoning path：`implementation-trace`
> 证据姿态：lineage/schema 约束是项目设计合同；具体 stage 顺序的收益属于 setting-specific `supported` 或 `plausible`

## 一句话定位

预训练数据生命周期是从目标能力和合法来源开始，到可追踪的训练 sequence、冻结 validation 和训练反馈结束的一条有版本数据流，而不是一串彼此独立的清洗脚本。

## Motivating Problem

只保留最终训练 shard 时，一旦模型指标退化，就无法判断变化来自 source snapshot、解析器、filter、dedup、sampler 还是 tokenizer。成功的数据系统必须满足：任意输出都能追溯输入与变换；任意版本差异都能缩小到具体 stage；删除或修复能传播到下游资产。

本笔记沿真实实现路径理解生命周期：固定一次 build 的 source、code、config 和 executor，再追踪每个 record 的状态变化。

## Minimal Motivating Example

两个 pipeline 最终都保留 900/1000 个文档。A 有 100 个 HTML 解析失败，B 解析全部成功但质量过滤删除 100 个。输出数量相同，原因、修复方式、内容分布与治理含义完全不同；若日志都只记作 `filtered=100`，后续无法区分。

## 核心定义

把一次数据构建表示为有向无环图 $G=(V,E)$：节点是不可变的数据快照或统计产物，边是带版本配置的变换。任意输出 document $d^{(s)}$ 都应存在 lineage：

$$
\ell(d^{(s)})=(\text{source-id},\text{snapshot},\text{parent-id},\text{stage},
\text{code-rev},\text{config-hash},\text{timestamp})
$$

这一 tuple 的具体实现可以不同，但必须能回答“它从哪来、经历了什么、为何留下/删除”。

## Assumptions and Validity

| 关系或结论 | 类型与条件 | 置信状态 |
|---|---|---|
| 一次冻结 build 可表示为 DAG | `implementation model`；对带在线反馈的长期系统，只对单次版本化执行展开成立 | `supported` |
| lineage tuple 足以审计来源与变换 | `design contract`，不是唯一 schema；还依赖 ID 稳定、日志不丢失和删除传播 | `supported` |
| $\operatorname{hash}(O_s)=f(\operatorname{hash}(I_s),\ldots)$ | deterministic stage 下的目标 `invariant`；第三方服务、并行归约或非确定模型会破坏 bitwise equality | `verified only under determinism` |
| 记录 stage funnel 能定位因果 | 只提供定位线索的 `empirical/diagnostic association`；真正因果仍需重放或 ablation | `plausible until checked` |

本文把 pipeline 顺序当作可配置实现，不主张存在跨语料普适的唯一顺序。比如“先 exact dedup 再昂贵过滤”能降低计算是可推导的成本关系，但是否改善最终模型取决于 normalization、canonical selection、数据分布与训练预算。

## 相关知识展开

### 1. 从目标反推数据，而不是从手头文件正推

第一步不是下载 Common Crawl，而是冻结目标与约束：

- 目标模型/语言/代码/知识时效/上下文长度。
- 数据 cutoff、许可和隐私约束。
- 总 token/compute/storage 预算。
- validation suites 与不可进入训练的数据边界。
- 需要进行归因的 domain taxonomy。

这些约束决定 source selection、抽取器、过滤阈值和 mixture。若 domain taxonomy 到训练后才补，很多“分域 loss”将无法可靠回溯。

### 2. 一个可审计的 stage graph

推荐将 pipeline 拆成可重放阶段：

```text
source registry / immutable snapshot
  -> acquisition + checksum
  -> parse / text extraction
  -> canonical normalization
  -> metadata + language/domain annotation
  -> safety / PII policy actions
  -> quality features and filtering
  -> exact / near / substring deduplication
  -> benchmark decontamination
  -> frozen train / validation split
  -> domain mixture and resampling manifest
  -> tokenize / pack / shard
  -> dataloader exposure logs
```

顺序不是绝对的。例如先 exact dedup 可降低昂贵 classifier 的推理量；先规范化会改变 hash；先 split 再 dedup 可能让 train/validation 近重复残留。关键是记录顺序并用实验验证，而不是假设步骤可交换。

### 3. 每阶段都要保存“漏斗 + 分布”，不只保存输出

对 stage $s$：

$$
Y_s^{(doc)}=\frac{N_{s,\text{out}}^{(doc)}}{N_{s,\text{in}}^{(doc)}},
\qquad
Y_s^{(tok)}=\frac{N_{s,\text{out}}^{(tok)}}{N_{s,\text{in}}^{(tok)}}
$$

文档 yield 与 token yield 要同时看，因为长文档和短文档的拒绝会产生不同影响。还应按 source、language、domain、time、length bucket 切片，记录：

- 输入、保留、拒绝、异常和无法解析的计数。
- reject reason（允许多标签时说明计数口径）。
- 关键 feature 的 before/after 分布和分位数。
- 随机 accepted/rejected 样本审计，但要脱敏并遵循数据使用边界。
- CPU/GPU time、峰值内存、I/O bytes 和失败重试。

只保存最终 2T tokens 会丢掉最有价值的因果线索：到底是哪个 stage 删除了哪个群体。

### 4. 文档身份与变换语义

建议区分三个 ID：

```text
source_id:  原始对象身份，如 URL + crawl timestamp / repo commit
content_id: 规范化内容的 cryptographic hash
record_id:  当前 pipeline record，含父节点和 stage
```

`source_id` 支持 provenance/删除请求，`content_id` 支持 exact duplicate 检测，`record_id` 支持一对多/多对一变换。例如一个 PDF 可抽取多个段落，一个 URL 在不同 crawl snapshot 的内容也可能不同。

### 5. 冻结、版本与可重现性 invariant

一次正式 build 至少冻结：

- source snapshot/URI/checksum；
- extractor/filter/dedup/tokenizer 的代码 revision；
- 完整配置与随机 seed；
- stage manifest 与输入输出 shard checksum；
- schema version、统计口径和失败处理策略；
- train/validation split 和 contamination reference 版本。

关键 invariant：

$$
\text{hash}(O_s)=f(\text{hash}(I_s),\text{code-rev}_s,
\text{config-hash}_s,\text{seed}_s)
$$

现实中分布式执行和第三方服务可能破坏 bitwise determinism，至少要追求 semantic reproducibility：文档集合、拒绝决策和统计在允许误差内一致，并明确非确定源。

### 6. 最小 worked example

设 1000 个 raw documents，共 2.0M tokenizer tokens：

| Stage | Documents | Tokens | 主要观察 |
|---|---:|---:|---|
| parse success | 900 | 1.80M | 100 个解析失败需单列，不应混入“低质量” |
| heuristic filter | 720 | 1.50M | doc yield 80%，token yield 83.3% |
| near dedup | 600 | 1.20M | 删除长重复簇，token 降幅更大 |
| decontamination | 594 | 1.18M | 数量小，但直接影响评测可信度 |
| packing | 580 sequences × 2048 | 1.18M | 名义容量 1.188M，packing efficiency 约 99.3% |

这个表不能证明 pipeline 更好，但能暴露异常。例如 parse failure 突然从 10% 到 40%，应先排查 source/schema/extractor 变化，而不是调整 quality filter。

### 7. 训练反馈如何回流

数据生命周期不在生成 shard 时结束。训练系统需要发回：

- 实际 per-domain sampled/loss tokens 与 target mixture 的偏差。
- per-domain held-out loss、loss slope、spike 时的 batch lineage。
- downstream error slices 与可能相关的数据 coverage。
- memorization/decontamination audit 结果。

这些反馈形成下一版配方的假设，但不能直接当因果证据。比如代码 benchmark 提升可能来自代码占比，也可能来自 tokenizer、训练步数或 checkpoint 选择；需要 paired ablation。

## 与 Pretraining Data 主线的关系

关系类型：`implemented-by` versioned pipeline stages；`prerequisite-for` source/stage-level diagnosis。

生命周期是本项目的总骨架。过滤、去重、mixture、metrics、validation 都是图上的节点或反馈边。掌握它之后，学习新论文时不再只问“用了什么过滤器”，而会问输入快照、顺序、统计、版本、反事实和训练反馈是否完整。

## 目标掌握程度

- 能为一个公开数据集画出 stage DAG，标出每个输入/输出和 lineage 字段。
- 能设计 stage-level doc/token yield 与 sliced distribution 监控。
- 能解释为什么 normalization、split、dedup、decontamination 的顺序会改变结果。
- 能写一份 build manifest，使另一台机器可语义复现。

## 常见误区

- 把 parse failure、policy removal 和 low-quality rejection 都记成 `filtered`。
- 只保留最终数据，不保留拒绝原因和中间统计，导致无法诊断版本退化。
- 用文件名当版本；同名 remote dataset 或 mutable branch 可能已变化。
- 在 tokenize/packing 后才尝试恢复 source/domain 元数据。
- pipeline 可重跑就等同可重现；外部服务、随机性和 mutable inputs 仍可改变结果。

## 自测问题

1. 若先随机切 train/validation 再全局 near-dedup，与先 dedup 再切分相比，各有什么泄漏和分布风险？
2. 某版本总 token yield 不变，但低资源语言的 yield 下降一半、英语上升。全局指标为什么失效？你需要哪些 slice？
3. 一个 URL 被多次 crawl、内容逐渐更新。如何设计 ID 和删除请求传播，使 provenance 与 dedup 都成立？

## 参考入口

- [Penedo et al., 2024, The FineWeb Datasets](https://arxiv.org/abs/2406.17557) — 适合观察大规模 web pipeline 的设计选择、ablation 与公开构建思路。
- [Soldaini et al., 2024, Dolma](https://arxiv.org/abs/2402.00159) — 适合学习开放 corpus 的组成、版本说明、pipeline 文档和中间状态分析。
- [Li et al., 2024, DataComp-LM](https://arxiv.org/abs/2406.11794) — 适合学习如何用固定训练/评测设置把数据 curation 变成受控 benchmark。
- [DataTrove official repository](https://github.com/huggingface/datatrove) — 对照 `Document`、pipeline block、executor、logging 和 dedup 示例，检查本文的 stage/manifest 思路如何落到代码。
