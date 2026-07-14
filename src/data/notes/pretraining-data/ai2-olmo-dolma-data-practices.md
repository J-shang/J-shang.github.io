---
title: "AI2 OLMo / Dolma 数据实践"
description: "用开放数据、处理工具、配置、顺序、checkpoint、日志和评测资产建立可复查的数据系统控制组。"
topic: "pretraining-data"
section: "open-controls"
slug: "ai2-olmo-dolma-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 90
readtime: 23
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/open-controls/ai2-olmo-dolma.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/open-controls/ai2-olmo-dolma.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:63334b20c7c72a077dc045ac1fdbebfcfc193d0ddb7a4994a9597b7fdae81233"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`implementation-trace`
> 研究锚点：Dolma v1.6、OLMo 1、OLMo 2 1B/7B/13B 的两阶段训练资产。

## 定位与 motivating problem

OLMo/Dolma 在本项目中的角色不是“最佳数据配方”，而是开放控制组：当语料、处理工具、配置、随机种子、训练日志和中间 checkpoint 都能访问时，数据研究能从“厂商说做了什么”推进到“哪个输入通过哪个版本的实现影响了哪个训练阶段”。

需要避免另一种误区：开放不等于无偏、无版权风险或跨规模有效。Dolma 的英文主导、source 可再分发性和 1B proxy ablation 本身构成明确边界。

## 代际与阶段表

| 模型/资产 | 阶段 | 官方披露 | 数据量语义 | 披露等级 |
|---|---|---|---|---|
| Dolma v1.6 | corpus build | 6 类 source、约 4.367B documents、3.059T LLaMA-tokenized tokens；论文基于 v1.6 | processed corpus pool，不等于单次训练 loss tokens | `D4` |
| OLMo 1 | `P0` | 初版 OLMo 使用 Dolma 的 2T-token sample；7B 评测 checkpoint 到 2.46T sampled tokens | sampled/training tokens；`loss_tokens=unknown` | `D4` |
| OLMo-7B-0424 | `P0/P2` | 模型卡给出 2.05T training tokens、4K context、staged training 与 logs | 官方未在单一字段里分解 base/anneal 的 loss-token 口径 | `D4` |
| OLMo 2 1B/7B/13B | `P0` | largely web-based `OLMo-mix-1124`；1B/7B 约 4T、13B 5T；公开 data CSV、config、checkpoint、W&B | sampled tokens；精确 non-padding loss 分母需从配置/loader 回算 | `D4` |
| OLMo 2 1B | `P2` | 3 个不同 data order，各约 50B high-quality tokens，seed 42 run 作为最终 checkpoint | 每个 run 的 sampled tokens | `D4` |
| OLMo 2 7B | `P2` | 3 个不同 data order，各 50B；最终模型为 3 个 run 权重平均 | 不能把 3×50B 当作一个模型顺序看到 150B | `D4` |
| OLMo 2 13B | `P2` | 3 个 100B run + 1 个 300B run；最终权重平均 | 各 run 独立 exposure；最终模型没有单一等价 sampled-token 数 | `D4` |

`三模型权重平均 --analogy-to--> 单模型连续训练 150B` 只能作为直觉类比，不是 exact identity。

## Assumption ledger

| 项 | 本笔记的处理 |
|---|---|
| 研究对象 | 以 base checkpoint 和公开训练 run 为主，不把 Tülu/SFT/DPO 数据并入 pretraining |
| 统计单位 | Dolma corpus 用 document/UTF-8 bytes/LLaMA tokens；OLMo run 用 sampled training tokens |
| tokenizer | Dolma v1.6 总表使用 LLaMA tokenizer；具体 OLMo run 的 tokenizer 需从固定 config 读取，二者数字不可默认等价 |
| 阶段 | OLMo 2 的 base `P0` 与 high-quality finishing `P2` 分行 |
| 分布 | Dolma corpus 构成不等于 OLMo sampler 的实际 exposure；精确 exposure 需结合 mix/config/order |
| 版本 | Dolma 论文为 v1.6 manuscript；数据卡另有 v1.5/v1.7；不能混写 |
| 省略效应 | OLMo 2 `P2` 同时涉及 data order、学习率/训练设置与权重平均，不能把最终差异只归因于高质量数据 |

## 数据字段表

| 字段 | 已核验事实 | 边界/未知 | 置信 |
|---|---|---|---|
| source | `[来源事实]` Dolma v1.6 包含 Common Crawl、GitHub、Reddit、Semantic Scholar、Project Gutenberg、Wikipedia/Wikibooks | 这不是所有 OLMo 2 mixture 的完整代际 diff | `verified` |
| corpus scale | `[来源事实]` v1.6 表 1：约 11,519 GB processed UTF-8、4.367B documents、3.059T LLaMA tokens | 原始抓取约 200TB 与 processed 11TB 不是同一 stage | `verified` |
| rights | `[来源事实]` 数据集以 ODC-BY 发布，toolkit 为 Apache-2.0；数据卡同时要求遵守原始 source 的许可/条款 | 开放发布不消除逐 source 权利审查 | `verified`（发布条款）；法律外推 `open` |
| cutoff | `[来源事实]` OLMo-7B-0424 模型卡给出 2023-10 cutoff，且多数数据来自 2023-02/03 | 不同 Dolma/OLMo 2 mix 需要独立 snapshot 审计 | `verified`（该模型卡范围） |
| filtering | `[来源事实]` Dolma 记录 language、质量/重复、PII/toxicity 等 source-specific 处理，并警告“quality”带价值判断 | 本笔记尚未逐 source 固定所有阈值 | `supported` |
| dedup | `[来源事实]` Dolma 对不同 source 使用不同去重策略；v1.7 数据卡称加入更多 fuzzy dedup | within/cross-source 的具体覆盖需逐 pipeline/config 追踪 | `supported` |
| decontam | `[来源事实]` OLMo 报告其训练数据针对 Paloma 去污染；Dolma ablation 任务选择也考虑训练数据中不存在的测试集 | 不能外推为对所有 benchmark 无污染 | `verified`（声明范围） |
| mixture | `[来源事实]` Dolma v1.6 公开 source pool 规模；OLMo 2 公开 base/anneal mix 资产和 config | corpus 自然比例、sampler 权重与实际 exposure 必须分别计算 | `supported` |
| synthetic | `[未知]` 本轮锚点没有建立 OLMo 1/2 base pretraining 中 synthetic 数据的统一定量占比 | 需逐 `OLMo-mix`/`Dolmino` 数据卡检查 | `open` |
| loss tokens | `[未知]` 已披露 training tokens 未在本轮直接等价到 non-padding loss tokens | 需固定 config、packing、mask 并回算 | `open` |

## Pipeline、采样与阶段变化

### Dolma：source-specific pipeline

Dolma 的重要方法不是“一套 filter 处理所有 source”，而是先保留 source/document 边界，再使用适合 web、code、paper、book、forum 和 encyclopedia 的不同处理路径。可检查的 lineage 最少应包含：

```text
source snapshot
  -> parse / document boundary
  -> source-specific filter + PII/safety handling
  -> exact/near dedup scope
  -> versioned shard + metadata
  -> tokenizer/config
  -> sampled order
  -> OLMo training step/checkpoint/log
```

`Dolma corpus composition --not-equivalent-to--> OLMo exposure distribution`。必须结合训练 mix、sample order 和终止 step，才能回答某 source 实际被看到多少次。

### OLMo 2：两阶段与 data order

OLMo 2 把大规模、largely web-based 的 `P0` 与较小、high-quality targeted 的 `P2` 分开。官方仓库进一步公开不同 seed/data order 的 finishing runs，并对 7B/13B 做权重平均。这使两个问题可被区分：

1. high-quality mixture 相对 base continuation 的作用；
2. 固定 mixture 下 data order 的方差，以及权重平均带来的变化。

但仅看最终 souped checkpoint 无法识别两者。需要比较每个 ingredient 的同 step checkpoint、W&B 曲线与统一 eval。

## Validation、contamination 与 ablation

### Dolma proxy ablation

`[来源事实]` Dolma 对多个中间数据选择训练 1B 级模型；跨 corpus 对照使用 1.2B 模型、每个 150B tokens，并用 Paloma 看多 source/domain perplexity，同时报告 8 个 downstream tasks。论文据此支持多 source corpus 在该设置下的 domain fit。

证据强度：给定 1.2B/150B/该 eval suite 的 controlled comparison 可标 `supported`；外推到 7B–70B、多语言或不同 tokenizer 仍为 `open`。论文自己也把“单一模型配置”和“有限任务集合”列为限制。

### OLMo training observability

OLMo 1 发布 500+ 中间 checkpoint（约每 1,000 steps）、完整训练 metrics、训练与 eval code；OLMo 2 继续提供 stage checkpoint、config 与 W&B。 operational anchor 是：

```text
(config revision, dataset/mix revision, seed, step)
  -> checkpoint
  -> train/validation metrics
  -> fixed eval result
```

缺少任一主键都可能把不同 data order、阶段或代码版本的指标错接。

## 表面冲突与区分性检查

### “Dolma 是 3T”与“OLMo 用 2T/2.46T/2.05T”

- 第一个差异：对象不同。3.059T 是 Dolma v1.6 corpus pool；2T 是初版选样描述；2.46T 是 OLMo-7B checkpoint exposure；2.05T 属于 OLMo-7B-0424。
- 区分性检查：固定 model card/config 和 dataset revision，读取 sampler manifest 与终止 step，不从 corpus 总量反推训练 exposure。
- 结论：不是数值冲突；是版本、模型和统计对象未对齐。`verified`。

### “7B stage 2 是 50B”与“三个 run 共 150B”

- 第一个差异：统计对象是单个 run 还是整个研发 compute。
- 区分性检查：追踪最终 checkpoint 是连续训练还是 ingredient 权重平均。
- 结论：最终 souped model 的每个 ingredient 各看过 50B；不能称最终模型连续看过 150B。`verified`。

## 明确未知项

- `[未知 | open]` OLMo 2 各版本的精确 non-padding `loss_tokens`，需要配置级 accounting。
- `[未知 | open]` 本轮尚未逐文件复核 `OLMo-mix-1124` 与 `Dolmino-mix-1124` 每个 source 的 tokenizer-token exposure。
- `[未知 | open]` 不同 seed/order 与 weight soup 的独立效应；需要 ingredient-level paired evaluation。
- `[未知 | open]` Dolma/OLMo 的数据选择结论对更大模型、多语言和多模态的迁移程度。

## 可迁移经验与不可外推

### 可迁移

- `[综合判断 | supported]` 发布 dataset revision、处理 code/config、sample order/seed、checkpoint 和 logs，能把数据研究从静态 corpus audit 扩展到训练轨迹分析。
- `[综合判断 | supported]` corpus pool、sampler exposure 和 loss tokens 应分别记录；同一个“3T”不能跨对象复用。
- `[综合判断 | supported]` 用小模型做 data ablation 时，应保留统一 token budget、模型配置与多域 validation，并显式记录 proxy-transfer 风险。

### 不可外推

- Dolma 在 1B proxy 上的结果不证明其过滤器/mixture 对所有规模最优。
- ODC-BY 发布状态不等价于所有原始文档都具备同一许可。
- OLMo 的开放程度不等价于训练数据没有偏差、污染、隐私或时效问题。
- OLMo 2 finishing 的提升不能只归因于“高质量数据”，除非控制 LR、order、checkpoint averaging 和 eval。

## 掌握标准与推理型自测

掌握本案例后应能：

1. 解释 Dolma 3.059T、OLMo 2T、2.46T 与 2.05T 分别是什么对象。
2. 设计一个固定 mixture、改变 data order 的 paired comparison，并说明为何只看 souped model 不够。
3. 从 `(dataset revision, config, seed, step)` 定位训练 checkpoint 和指标。
4. 说明为什么 1B/150B ablation 的结论不能直接升级为跨规模因果定律。

自测：如果两个 OLMo 2 7B ingredient 在 50B finishing 后分域 loss 不同，你至少还需要检查哪些配置、数据顺序统计和 checkpoint 指标，才能把差异归因于 order 而不是运行漂移？

## 一手来源

- [Dolma paper, arXiv:2402.00159v2](https://arxiv.org/abs/2402.00159)：为什么读：给出 v1.6 的 source 规模、设计目标、pipeline、ablation 和限制；重点读表 1、§3–§9、Limitations 与 datasheet。
- [Dolma official dataset card](https://huggingface.co/datasets/allenai/dolma)：为什么读：固定 v1.5/v1.6/v1.7 的发布关系、下载入口与 ODC-BY/原 source 条款；重点读 Versions 与 Licensing Information。
- [Dolma official toolkit](https://github.com/allenai/dolma)：为什么读：把论文方法落到 processor、tagger、dedup 与 config；实现结论需另外固定 commit 后再标 `verified`。
- [OLMo paper, arXiv:2402.00838v4](https://arxiv.org/abs/2402.00838)：为什么读：连接 Dolma、训练设置、in-loop/offline evaluation、checkpoint 和 logs；重点读 §2.2、§2.4、§3.3、§4.2、§5。
- [OLMo-7B-0424 official model card](https://huggingface.co/allenai/OLMo-7B-0424)：为什么读：核对 2.05T、context、cutoff、staged training 与 W&B 入口；重点读 Model Details、Data、Staged training/annealing。
- [OLMo official repository](https://github.com/allenai/OLMo)：为什么读：核对 OLMo 2 的两阶段 token 数、seed/data order、config、checkpoint 和 weight soup；重点读 Pretraining、Stage 1/2。仓库已提示后续新版本迁移至 OLMo-core，因此实现追踪必须固定历史 revision。
