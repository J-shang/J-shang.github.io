---
title: "长上下文数据：能力窗口之外的暴露账本"
description: "分离 context window、长度 curriculum、自然长文、拼接方式、attention mask 与 loss exposure。"
topic: "pretraining-data"
section: "cross-cutting"
slug: "long-context-data"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 81
readtime: 15
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/long-context-data.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/long-context-data.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:67b59fc2663819a13636a9fa5b4d28bffe6d3f3df23b58ab0ccb19eada55b1c7"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`unified-framework`，即用同一组问题对照不同厂商；这不表示它们的 data recipe 等价 -->
> 主要参考案例：Meta Llama、MiniMax、Apple AFM、NVIDIA Nemotron、GLM、Hunyuan、Kimi；开放对照：AI2 OLMo/Dolma

## 这篇专题要回答什么

模型发布时常说“支持 128K、1M，甚至 10M context window”。这个数字只说明模型或服务允许输入的最大 seqlen，不能单独回答下面这些问题：

1. 模型训练时是否真的使用过这么长的 sequence？
2. 这些 long sequence 来自完整书籍、论文和代码仓库，还是由许多无关短文拼接而成？
3. 标称训练量中有多少是 padding，又有多少 token 真正参与了 loss 计算？
4. 模型只是能接收很长的输入，还是确实能利用相距很远的信息？
5. 扩展 context window 后，原有的短文本能力有没有下降？

这篇专题的目标，就是把厂商资料中的这些问题分开讨论。它不是在比较谁公布的 context window 最大，也不根据窗口大小推测模型能力。

## 先看一个具体例子

假设模型 A 和模型 B 都支持 128K context window。

| 模型 | 训练时使用的 128K sequence | 可能学到的东西 |
|---|---|---|
| A | 把 100 篇互不相关的短文章随机拼在一起 | 适应更大的 seqlen 和相应计算过程，但未必学会 long distance dependency |
| B | 使用一本完整书籍或一个完整代码仓库；sequence 末尾的问题依赖开头的信息 | 更有机会学习 long distance dependency，但效果仍需要专门评测确认 |

两者接受的最大输入长度相同，训练数据提供的学习任务却不同。因此：

> **context window 是输入容量；长上下文训练数据决定模型在这个容量里见过什么；评测才告诉我们模型最终会不会使用远处的信息。**

这三者有关联，但不是同一个概念，也不能互相替代。

## 六个常用术语

| 术语 | 本文中的含义 | 不要和什么混淆 |
|---|---|---|
| `context window` | 模型一次最多允许处理的 token 数 | 不等于训练时常用的 seqlen |
| `sequence` | 一次送入模型的 token sequence | 不等于原始 document；一个 sequence 可以包含多个 document |
| `seqlen` | 一个 sequence 包含的 token 数 | 不等于原始文档长度；短文也可以拼成很长的 sequence |
| 自然长文（`natural long document`） | 本身就具有长结构的文档，例如书籍、论文或代码仓库 | 不等于随机拼接的短文 |
| `packing` | 为减少 padding，把一个或多个样本放进固定 seqlen 的 sequence | packing 提高计算利用率，但不会自动创造有意义的 long distance dependency |
| `loss tokens` | 真正进入训练 loss 分母、对参数更新产生直接训练信号的非 padding tokens | 不等于语料池规模，也不一定等于送入模型的全部 tokens |

本文尽量使用这些常见术语。后文所说的“实际训练量”，指能够明确统计到具体训练阶段、seqlen 和 loss 口径的 token 数。

## 阅读厂商资料时需要分别回答的六个问题

| 问题 | 需要查什么 | 常见的错误推断 |
|---|---|---|
| 最多能输入多长？ | 模型或 API 的最大 context window | “支持 1M，所以一定用过大量 1M 长文训练” |
| 训练使用多大的 seqlen？ | 每个阶段的 seqlen 和 token 数 | “最终阶段是 128K，所以整个 pretraining 都是 128K” |
| 原始材料是什么？ | 书籍、论文、代码仓库、网页、对话轨迹等 | “sequence 很长，所以原始文档也很长” |
| long sequence 怎样构造？ | 完整文档、随机拼接、按主题拼接、合成长问答等 | “所有 128K sequence 都包含有用的 long distance dependency” |
| 多少 token 真正参与 loss？ | padding、loss mask、截断和丢弃情况 | “训练日志中的位置数就等于 loss tokens” |
| 最终学会了什么？ | 不同距离、文档类型和任务上的评测结果 | “通过 needle-in-a-haystack 就等于会做长文推理” |

只有把这六个问题分开，才能判断两个厂商公布的数字是否真的可比。

## 如何计算真正参与训练的 token 数

先区分三个量：

- 标称位置数（`nominal positions`）：训练配置中预留的位置总数，其中可能含 padding；
- 非 padding tokens：去掉 padding 后，模型实际读到的 tokens；
- `loss tokens`：非 padding tokens 中真正参与 loss 计算的部分。

对某个 seqlen bucket $b$，设：

- $n_b$：这个区间训练了多少个训练迭代；
- $B_b$：每个训练迭代包含多少个标称位置；
- $u_b$：其中非 padding 部分的比例；
- $q_b$：非 padding tokens 中参与 loss 的比例。

那么：

$$
T_{\text{nominal},b}=n_bB_b,
$$

$$
T_{\text{loss},b}=n_bB_bu_bq_b.
$$

例如，一个 batch 有 128K 个标称位置，其中 80% 不是 padding；非 padding tokens 中又有 90% 参与 loss：

$$
128\text{K}\times 0.8\times 0.9=92.16\text{K}.
$$

因此，这个 batch 有 92.16K 个 loss tokens。这个结果仍然不能告诉我们其中有多少来自完整长文，因为完整长文、随机拼接短文和合成任务都可能贡献这些 tokens。要回答来源问题，还需要 manifest 或 dataloader log。

### 假设与适用范围

- 上述公式是给定日志字段后的精确统计关系，不是由 context window 推算训练量的方法。
- $q_b$ 的分母是非 padding tokens；若实现使用别的统计口径，必须重新定义。
- 不同 tokenizer 会把同一文本切成不同数量的 tokens，因此跨模型数字不能直接相加或排名。
- 如果厂商没有公开 padding、loss mask 和训练迭代数，`loss tokens` 就应保持 `unknown`。

## long sequence 通常从哪里来

### 1. 本身就很长的材料

| 数据类型 | 保留的结构 | 适合训练什么 | 主要风险 |
|---|---|---|---|
| 完整书籍、论文或长文档 | 章节、论证过程和跨段语义 | 长文理解、跨章节推理 | 许可、OCR 错误、训练与评测章节泄漏 |
| 代码仓库、issue 和 pull request | 文件依赖、修改历史和测试关系 | 跨文件代码理解、软件工程（SWE）任务 | 基准测试仓库或测试集污染 |
| Agent 或工具调用轨迹 | 多步状态、动作和反馈 | 长流程执行和状态跟踪 | 环境版本不固定、失败步骤被错误 mask |

这些材料具有真实的长结构，但“文档很长”仍不保证每个 token 都需要依赖远处信息。

### 2. 把短材料组合成 long sequence

| 构造方式 | 做法 | 能解决什么 | 局限 |
|---|---|---|---|
| 随机拼接 | 把互不相关的短文连续放入同一个 sequence | 减少 padding，让模型适应更大的 seqlen | 几乎不提供跨文档依赖，模型可能学会忽略远处内容 |
| 按长度 packing | 优先组合长度合适的样本 | 提高计算利用率 | 容易优先选中“好拼”的长度，改变原始长度分布 |
| 按主题组合 | 把主题接近的文档放在一起 | 提供较弱的跨段关联 | 可能制造现实中不存在的文档关系 |

这里需要特别注意：`packing` 首先是数据装载和计算效率方法，不是长上下文能力的充分条件。

### 3. 人工生成需要长距离信息的任务

常见做法包括长文问答、多处证据检索、摘要、needle-in-a-haystack、合成工具轨迹等。它们的优点是可以精确控制答案依赖哪些位置，缺点是任务形式可能过于单一，模型也可能利用答案位置或模板等捷径。

### 4. 混入短文本以保持原有能力

context window 扩展阶段通常不会只训练 long sequence。继续混入短文本常被称为 `short replay`，目的是降低短任务能力退化。需要报告的是短文本与长文本在具体训练阶段中的采样比例，而不是笼统地说“使用了 replay”。

## 厂商公开资料实际告诉了我们什么

下表把“公开事实”和“不能据此推出的结论”放在一起。详细来源和版本见各厂商笔记。

| 模型或资料 | 公开信息（来源事实） | 可以得到的有限结论 | 仍不能确定的内容 |
|---|---|---|---|
| [Llama 3.1](/topics/pretraining-data/meta-llama-data-practices/) | 约 800B tokens、六阶段从 8K 增加到 128K，并评估长短任务 | context window 扩展不是一次跳到 128K，而是分阶段完成 | 每阶段 token 数、原始文档长度分布、完整长文与 packing 的比例 |
| [Llama 4](/topics/pretraining-data/meta-llama-data-practices/) | Scout 支持 10M、Maverick 支持 1M；提到专门的 mid-training 数据 | 最大窗口与专门训练阶段有关 | 各 seqlen 的训练量、数据来源和 long-sequence 构造比例 |
| [MiniMax Text-01](/topics/pretraining-data/minimax-data-practices/) | 披露 300B@128K、32B@512K、26B@1M 等长度区间 | 比只公布“支持 1M”多提供了各长度训练量 | 每个区间中完整长文、拼接和 loss tokens 的比例 |
| [Apple AFM 2024](/topics/pretraining-data/apple-foundation-models-data-practices/) | 披露 100B@32K，并使用获得授权的长文本和合成长问答；多数原文短于 32K | 32K sequence 不全是原生 32K 长文 | 完整长文与合成数据各占多少 |
| [Apple AFM 2025](/topics/pretraining-data/apple-foundation-models-data-practices/) | 最长训练到 65K；提到书籍、代码、合成 ICL/检索任务和 replay | 数据同时包含自然材料与合成任务 | 各阶段 token 数和长度分布 |
| [Nemotron 3 Ultra](/topics/pretraining-data/nvidia-nemotron-data-practices/) | 披露 33B tokens；92% 的训练迭代使用 1M、8% 使用 4K；另给出 46% long-category（长文本类别）与 54% phase-2 replay（第二阶段回放数据） | seqlen 比例和数据类别比例是两套不同统计 | 1M sequence 中各数据类别占多少，即二者的联合分布 |
| [GLM-5](/topics/pretraining-data/glm-data-practices/) | 披露 1T@32K、500B@128K、50B@200K，并列出书籍、论文、文档、代码仓库、SWE 和合成数据 | 可以看出长度分阶段增加且来源不止一种 | 每个长度阶段的自然/合成比例和 loss tokens |
| [Hunyuan TurboS](/topics/pretraining-data/tencent-hunyuan-hy-data-practices/) | 披露 30B@32K、20B@256K，以及 short:long=3:1 | 可以确认存在不同长度阶段和长短数据混合 | 数据进入顺序、完整长文比例和更细的筛选规则 |
| [Kimi k1.5/K2](/topics/pretraining-data/moonshot-kimi-data-practices/) | 长度从 4K 增至 32K、128K；K2 末段有 60B@32K；另披露全量/部分 attention 的 40/60 处理 | 长度、attention 方式和数据构造都可能变化 | 40/60 不能解释为自然/合成数据比例，也不能直接解释为 token 比例 |
| [OLMo/Dolma](/topics/pretraining-data/ai2-olmo-dolma-data-practices/) | 开放的数据、配置、顺序和 checkpoint 便于追踪 dataloader 的行为 | 可用来学习如何检查训练数据顺序和统计口径 | 不能代表所有前沿模型的长上下文训练方案 |

这张表的作用是帮助读者回到正确的问题，而不是把不同厂商压缩成一个“长数据百分比”排名。

## 为什么有些公开数字不能直接相乘或比较

### Nemotron 的 46/54 和 92/8

`46%/54%` 描述数据类别，`92%/8%` 描述训练迭代使用的 seqlen。除非报告同时给出“数据类别 × seqlen”的联合分布，否则不能把两组比例相乘后声称得到了“1M 自然长文占比”。

### Kimi 的 40/60

这组数字涉及全量 attention、部分 attention 以及相应的数据处理方式。它首先描述训练处理方案，不是自然长文与合成长文的比例。

### Hunyuan 的 3:1 与 Llama 的 800B

Hunyuan 的 3:1 是某个阶段内的长短数据混合比例；Llama 的 800B 是一个训练阶段的总 token 数。一个是比例，一个是总量，且分母和模型都不同，不能直接比较。

遇到类似数字，先写清楚四件事：属于哪个模型和训练阶段、统计单位是 token、文档还是训练迭代、分母是什么、是否使用同一 tokenizer。

## dataloader 也会改变模型真正看到的内容

长样本进入训练 batch 前，通常还会经过 packing、截断和丢弃。这些实现细节可能系统性地改变训练分布：

- `best-fit packing`（最佳适配拼接）会优先选择容易组合到目标长度的样本；
- `random truncation`（随机截断）可能频繁丢掉文档尾部；
- 按主题 packing 可能创造原始数据中不存在的跨文档关系；
- 过长或难以组合的样本可能更容易被丢弃。

要检查这些问题，dataloader 至少应记录：

```text
source_id, original_length, chunk_offsets
pack_id, segment_order, separator
target_length, padding_positions
attention_mask_type, loss_mask_positions
truncated_or_drop_reason
```

然后按来源、数据领域（domain）和原始长度分别统计：提交了多少样本、实际进入 batch 多少、被截断多少、被丢弃多少、最终产生多少 loss tokens。这样才能发现 dataloader 是否让某些类型的长文更难进入训练。

## 训练长度通常是分阶段增加的

context window 扩展往往同时改变多项因素：seqlen、位置编码或 attention 实现、learning rate、数据来源以及 short replay 比例。因此，最终指标变好不能自动归因于某一个因素。

一个较清楚的对照实验包含四组：

| 实验组 | 模型支持的长度 | 使用的数据 | 想回答的问题 |
|---|---|---|---|
| short-sequence baseline | 短 | 短文本 | 原始能力是多少 |
| 只扩展模型长度 | 长 | 短文本或随机拼接 | 位置编码和 long-sequence 计算本身带来什么 |
| 只更换数据来源 | 短 | 从长文档截取的短片段 | 长文档的内容分布是否有帮助 |
| 完整方案 | 长 | 保留结构的长文档和长任务 | 模型长度与长结构数据结合后的效果 |

四组仍不能消除所有混杂因素，但比只比较训练前后两个 checkpoint 更容易定位收益来源。每次阶段切换都应同时测试短、中、长输入，避免只在最终模型上测最大长度。

## 怎么判断模型真的会使用长上下文

### Needle-in-a-haystack 不够

Needle-in-a-haystack（NIAH）通常是在长文本中埋入一小段目标信息，再让模型把它找出来。它适合检查精确检索，但不能单独证明模型会做跨章节推理、整合多处证据或理解大型代码仓库。

更完整的评测至少应包含：

- 单处检索、多处检索以及干扰信息；
- 自然长文问答和摘要；
- 跨章节推理与矛盾识别；
- 跨文件代码理解或 Agent 状态跟踪；
- 短文本任务回归测试；
- 按 `length × source × task` 分别报告结果。

评测数据还应按完整书籍、代码仓库或文档分组后再切分，避免同一本书的不同章节或同一仓库的不同文件分别进入训练集和评测集。

### 测量模型能利用多远的信息

可以让正确答案依赖距离当前位置为 $d$ 的证据，并画出分数随距离变化的曲线：

$$
Q(d)=\mathbb{E}[\text{score}\mid D=d].
$$

其中，$D$ 是完成任务所需证据与当前位置之间的距离，$Q(d)$ 是 $D=d$ 时的平均任务得分。测试时还要改变答案位置、干扰信息密度和所需证据数量，防止模型只利用固定位置模式。

### 检查短文本能力是否退化

对每个数据领域 $k$，分别记录 context window 扩展前后的得分变化：

$$
\Delta_k=Q_{k,\text{after}}-Q_{k,\text{before}}.
$$

除了全局平均分，还要报告每个领域的结果，以及把各领域等权计算的 macro average。否则，少数短文本领域的明显退化可能被总体平均值掩盖。`short replay` 比例应根据长短任务的权衡曲线选择，而不是只追求最大的长文本分数。

## 三组看似矛盾的说法

### “模型的 1M 能力很好”与“1M 训练 tokens 很少”

两句话可能同时成立，因为最大输入能力、某项评测表现和训练数据量是三个不同对象。要区分具体原因，需要查看不同长度的训练量、loader 日志、完整长文比例以及 $Q(d)$ 曲线。

### “自然长文更好”与“合成长任务更可控”

两句话关注的目标不同：自然长文保留真实结构，合成任务可以精确控制需要哪些证据。较好的验证方式是在固定总 tokens、模型和训练配置的情况下，分别比较完整长文、随机拼接、按主题组合、合成长任务以及它们的不同比例。

### “长训练会损害短能力”与“加入 replay 可以保持短能力”

结果取决于 replay 比例、learning rate、阶段长度和评测分布。需要比较多个中间 checkpoint 上的长短任务曲线，不能只比较最终的两个全局平均分。

上述关系都是需要实验验证的经验问题，不是由 context window 或训练 token 数直接推出的数学结论。

## 不要从公开数字推断什么

- 不由 context window 推断模型使用过多少长上下文训练 tokens；
- 不由 128K 或 1M sequence 推断其中都是完整长文；
- 不把标称位置数当成非 padding tokens 或 loss tokens；
- 不把 attention 处理比例或数据类别比例解释成 seqlen distribution；
- 不把 NIAH 通过率外推为自然长文推理能力；
- 不把某个模型使用的长度阶段、位置编码或 replay 比例当成所有模型的最优方案。

如果资料没有给出相应字段，结论应标为 `[未知]` 或 `open`，而不是根据相邻数字补齐。

## 最小实验与日志清单

如果要自己验证一套长上下文数据方案，至少应做到：

1. 固定模型、总 token 数、优化器和 learning rate，比较完整长文、随机拼接、按主题组合和合成长任务；
2. 扫描不同的 short replay 比例；
3. 在每个 checkpoint 记录按“seqlen × 数据领域”划分的 validation loss；
4. 按完整书籍、仓库或文档进行训练集/评测集切分和去污染；
5. 保存样本进入 batch、截断、丢弃、padding 和 loss mask 的统计；
6. 同时测试检索、自然长文推理、代码/Agent 任务和短文本回归。

## 读完后应该能回答的问题

1. 为什么两个模型都支持 128K context window，并不代表它们接受过相同的长上下文训练？
2. 为什么随机拼接的 128K sequence 不等于一篇 128K 自然长文？
3. 如何从标称位置数、padding 和 loss mask 计算 loss tokens？
4. 为什么 Nemotron 的 46/54 与 92/8 不能直接相乘成“1M 长文占比”？
5. 为什么 Apple 的 100B@32K 不等于 100B 完整长文 tokens？
6. 为什么通过 NIAH 不能单独证明模型会做长文推理？
7. 如何设计实验，区分“模型能够处理更长位置”和“模型从有结构的长数据中学到能力”？

练习：某训练阶段有 20B 个标称位置，padding 占 15%；剩余非 padding tokens 中有 10% 被 loss mask 排除。请计算 loss tokens。这个数字能否说明完整长文所占比例？为什么？

## 来源与建议阅读位置

1. [Meta Llama](/topics/pretraining-data/meta-llama-data-practices/)：阅读 Llama 3.1 的约 800B tokens、六阶段长度增加和短任务恢复，以及 Llama 4 的披露边界。
2. [MiniMax](/topics/pretraining-data/minimax-data-practices/)：阅读 300B@128K、32B@512K、26B@1M 的分阶段训练量，以及 NIAH 较早饱和后为何要换更难的中间 checkpoint 任务。
3. [Apple AFM](/topics/pretraining-data/apple-foundation-models-data-practices/)：阅读 100B@32K、65K 数据来源，以及“多数原文短于最大训练长度”这一重要边界。
4. [NVIDIA Nemotron](/topics/pretraining-data/nvidia-nemotron-data-practices/)：重点区分 33B tokens、46/54 数据类别比例与 92/8 训练迭代长度比例。
5. [GLM](/topics/pretraining-data/glm-data-practices/)、[Hunyuan](/topics/pretraining-data/tencent-hunyuan-hy-data-practices/) 和 [Kimi](/topics/pretraining-data/moonshot-kimi-data-practices/)：比较自然长文、代码仓库、Agent 数据、合成长任务、packing 与 attention 处理方式。
6. [Validation design](/topics/pretraining-data/validation-design/)：学习怎样用各领域结果、干净的数据切分和中间 checkpoint 组织评测。
