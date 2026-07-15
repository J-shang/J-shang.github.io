---
title: "大厂数据实践统一调研方法"
description: "固定 organization、model family、generation、training stage 与 artifact version，再逐字段记录可审计证据。"
topic: "pretraining-data"
section: "methodology"
slug: "industry-data-methodology"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 70
readtime: 18
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/methodology.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/methodology.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:da3f01ad890971c5273acd5449ca42fbf1da670c5e8ae1cf7eae059edb852464"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`core`
> 版本：`v1.0`
> 最近复核：**2026-07-14**
<!-- maintenance: reasoning-path=`method-family/historical trace`；涉及真实配置和代码时切换为 `implementation-trace` -->

## 1. 这套方法要避免什么错误

“模型训练了 15T tokens”没有稳定含义：它可能是原始语料规模、去重后的 unique tokens、带重复采样的 sampled tokens、进入 loss 的 non-padding tokens，或多个训练阶段的累计数。厂商材料的披露粒度又不同，直接把这些数字放进一张表会制造虚假的可比性。

本方法要求先固定比较单位和训练阶段，再记录来源事实。无法统一统计对象时不做换算。

### 最小例子

假设两份报告都写“2T tokens”：

- A：2T 是去重后语料池，模型只采样了 300B；
- B：语料池只有 500B，训练时重复采样到 2T。

只有在 tokenizer、mask 和 packing 合同固定时，`sampled tokens` 才能进一步对应训练 exposure；因此 A 与 B 的“2T”既不是同一对象，也不能据此判断哪份数据更多或更好。

## 2. 一条记录代表什么

每条记录的主键为：

```text
(organization, model_family, generation, training_stage, artifact_version)
```

- `organization`：作出声明或发布资产的组织。
- `model_family`：共享主要训练谱系的家族，不等于 API 别名。
- `generation`：发生数据或训练方法变化的代际。
- `training_stage`：下表中的阶段；同一代可有多行。
- `artifact_version`：论文版本、system/model card 日期、dataset revision 或代码 commit。

产品别名只有在能证明映射到具体训练代际时才写入。持续更新的线上模型若 checkpoint 不固定，必须标记 `rolling/unknown`。

## 3. 怎样区分 training stage

阶段标签描述数据在训练中的角色，不强制所有厂商遵循同一时间顺序。

| ID | 阶段 | 操作性定义 | 常见混淆 |
|---|---|---|---|
| `P0` | base pretraining | 从初始化或早期 checkpoint 进行大规模 next-token/多模态基础训练 | 把语料池规模当成 loss tokens |
| `P1` | continued/mid-training | 在 base checkpoint 上继续训练，目标是更新领域、能力或数据分布 | 与 SFT、domain adaptation 混写 |
| `P2` | annealing / high-quality finishing | 训练末段改变 data recipe 和/或 learning rate，以高质量、目标域数据收尾 | 把效果全部归因于数据，忽略 learning rate 与 checkpoint averaging |
| `P3` | context extension | 用更长 sequence 或专门长文档 mixture 扩展 context | 把最大 context 长度当成真实长数据占比 |
| `A1` | supervised post-training | instruction/SFT 等监督训练 | 与预训练合成文本混写 |
| `A2` | preference / RL | DPO、RLHF、RLVR、reasoning RL 等偏好或强化学习 | 把 rollout tokens 加进 pretraining tokens |
| `A3` | agent/multimodal specialization | tool trajectory、computer use、模态适配等专项训练 | 与部署时 test-time compute 混写 |

`P2` 可以是 `P1` 的 special case，但只有报告明确将其作为收尾阶段时才使用 `P2`。`A2` 与 `P0` 不是可相加的同质 token budget。

## 4. 每篇案例要记录哪些字段

### 4.1 身份和时间

| 字段 | 记录要求 |
|---|---|
| `model_generation` | 固定模型代际；产品路由系统需拆出可识别组件 |
| `release_date` | 使用绝对日期 |
| `knowledge/data_cutoff` | 区分声明 cutoff、source snapshot 日期与推测的有效知识截止 |
| `checked_on` | 本仓库最后核查官方资料的日期 |
| `artifact_version` | arXiv 版本、card 更新日期、dataset revision、commit/tag |

### 4.2 数据与 token accounting

| 字段 | 允许值/要求 |
|---|---|
| `source` | web、code、paper、book、licensed/partnership、user opt-in、human、synthetic 等；保留厂商原分类 |
| `rights` | 具体许可、来源条款、授权方式或 `unknown`；“公开可访问”不等于“开放许可” |
| `raw/processed scale` | bytes、documents、images 等必须保留单位与处理阶段 |
| `unique_tokens` | 固定 tokenizer 下去重/整理后语料池规模；未给则 `unknown` |
| `sampled_tokens` | sampler 实际抽取量；若只给“trained on”但定义不清，附原文范围并标 `ambiguous` |
| `loss_tokens` | 真正进入 loss 分母的 non-padding tokens；通常为 `unknown` |
| `mixture` | 权重、自然比例或 exposure；百分比需说明按 document、sample 还是 token |
| `tokenizer` | 名称、版本、词表和特殊 token；缺失时不跨厂商换算 token 数 |
| `packing/mask` | sequence 构造和 loss mask；未披露写 `unknown` |

### 4.3 Pipeline 和实验

| 字段 | 记录要求 |
|---|---|
| `parse/normalize` | source 到 document 的主要变换和边界 |
| `quality/filter` | 规则/分类器、统计单位、阈值、适用域和失败模式 |
| `dedup` | exact/near/substr、粒度、within/cross-source 范围 |
| `decontam` | 对象、benchmark 版本、匹配规则、阈值与执行阶段 |
| `sampling/curriculum` | 静态/动态权重、重复 exposure、阶段变化和随机种子 |
| `synthetic/distillation` | teacher、生成阶段、验证/rejection 与是否进入 pretraining |
| `validation` | held-out 来源、时间边界、去重/去污染、tokenizer、每域规模与 micro/macro/per-domain 指标 |
| `ablation` | 模型、token budget、优化器、seed、checkpoint 与 eval 是否受控 |
| `artifacts` | data/manifest、processing code、config、order、logs、checkpoints、eval code |

## 5. 证据与置信度

### 5.1 内容标签

- `[来源事实]`：来源明确陈述或固定版本源码直接显示的内容。
- `[推导结论]`：由已写明的定义/公式推导，必须列假设。
- `[综合判断]`：整合多个证据后的项目判断。
- `[待验证假设]`：机制合理但尚无区分性证据。
- `[未知]`：资料没有给出、版本无法映射，或定义不足以比较。

### 5.2 置信状态

- `verified`：在声明范围内由定义、推导、源码或可复现检查直接建立。
- `supported`：多项相关证据一致，但仍有边界或未控制因素。
- `plausible`：机制合理但缺少区分性检查。
- `open`：证据不足、冲突或对象无法统一。

“官方说 X”只使“该官方资料作出 X 声明”成为 `verified` 来源事实；X 的独立真实性或跨代外推仍可能是 `supported/open`。

### 5.3 披露等级

| 等级 | 最低条件 |
|---|---|
| `D0` | 只有营销或第三方描述，关键数据字段未知 |
| `D1` | 官方高层描述，披露少量 source、cutoff 或阶段信息 |
| `D2` | 技术报告给出 token、mixture、pipeline、validation 等定量信息 |
| `D3` | 发布权重、代码、配置、训练指标或部分数据资产 |
| `D4` | data/manifest、处理代码、顺序、指标与评测协议基本可追踪 |

等级按“模型代际 × 阶段/字段”评定。`D4` 不保证低偏差、许可无争议或模型能力更强；`D1` 也不证明数据质量差。

## 6. 阅读结论前先确认哪些前提

每篇厂商笔记至少记录：

| 项 | 必答问题 |
|---|---|
| 研究对象 | 是 base model、路由系统、instruction model 还是 API snapshot？ |
| 统计单位 | token、document、sequence、image、trajectory 还是 mixture weight？ |
| 阶段 | 数字属于 `P0`、`P1/P2/P3` 还是 `A1/A2/A3`？ |
| 分布 | 自然分布、人工加权、动态 sampler 还是未知？ |
| 版本 | 哪个 dataset/card/code revision？ |
| 近似 | token 换算、epoch/exposure 或跨模型比较用了什么近似？ |
| 省略效应 | 架构、优化器、compute、checkpoint averaging、eval 变化是否混杂？ |

## 7. 同一厂商的新旧模型怎样比较

不重复抄写所有背景，只记录第一个发生变化的对象：

```text
Generation A -> Generation B
Stage:
Changed field:
Previous value/behavior:
New value/behavior:
Evidence/version:
Relation type: exact / special case / approximation / implementation / analogy / empirical association
Confounders:
Discriminating check:
Confidence:
```

若 B 只重复了组织级高层说明，应写“未披露代际差异”，而不是默认沿用 A 的 mixture、cutoff 或 tokenizer。

## 8. 冲突处理

两个来源表面冲突时按以下顺序处理：

1. 统一模型代际和训练阶段。
2. 统一统计单位、tokenizer、分母与时间窗口。
3. 对齐资料版本和发布日期；后续卡片可能描述 rolling snapshot。
4. 找到第一个不同的假设、指标、近似或实现。
5. 写出区分性检查：固定版本源码、manifest 计数、人工 fixture、受控 ablation 或向发布方确认。
6. 无法区分时保留双方声明并标 `open`。

第三方 tokenizer inference、模型自述 cutoff、benchmark 表现只可作为研究线索，不能覆盖官方/可复现证据，也不能填入事实矩阵。

## 9. 厂商笔记模板

```text
# 组织：模型家族数据实践

> 状态：draft
> 核查截止：YYYY-MM-DD
<!-- maintenance: reasoning-path=... -->
> 主要模型/资料：...

## 这篇案例要回答什么
## 各代模型和 training stage
## 阅读这些结论前先确认的前提
## 厂商公开了哪些 data fields
## Pipeline、采样与阶段变化
## Validation / contamination / ablation
## 可检查锚点（公式、shape、config、manifest 或诊断）
## 看似矛盾的说法怎样区分
## 目前仍不知道什么
## 哪些经验可以借鉴，哪些不能直接照搬
## 读完后应该能回答的问题
## 来源（为什么读、读哪里、版本/日期）
```

## 10. 比较矩阵录入规则

- 一行只能表示一个 `model_generation × training_stage`。
- 精确数字必须能回链到厂商笔记的一手来源。
- 空白不是零；统一写 `unknown`，`n/a` 只用于该阶段不适用。
- 百分比没有统计单位时写 `ambiguous`。
- 不把不同 tokenizer 的 token 数相加或排名。
- 不从某代继承另一代未再次披露的字段。
- 矩阵只显示关键字段，复杂 caveat 留在厂商笔记。

## 11. 复核周期与版本记录

- 每次编辑厂商笔记时重新打开关键一手来源，并更新 `checked_on`。
- 最新模型、system card、data policy 与许可字段每 **90 天**做一次队列复核；重大新代际发布后即时复核。
- 固定论文/代码结论仅在 arXiv 版本、dataset revision、commit 或实现路径变化时重查。
- 历史结论不被新代覆盖；新增 `Generation A -> B` diff。
- 每张表保留绝对日期，不使用“当前”“最近”等无时间锚点表述。

## 12. 一篇案例达到什么条件才进入索引

一篇案例进入索引前必须具备：

- 至少一张阶段表和一张统一字段表；
- 已记录主要分析方式，以及阅读结论所需的关键前提；
- 至少一个可检查 operational anchor；
- validation/contamination/ablation 的证据边界；
- 冲突或缺失字段的区分性检查；
- 明确的 `unknown`、披露等级与置信状态；
- “可迁移”与“不可外推”结论；
- 带阅读理由和定位建议的一手来源。

达到这些条件可标 `draft` 并进入索引；机制、实现和来源达到概念深度后才标 `core`；逐项复核后才标 `verified`。
