---
title: "数据配比与采样"
description: "用目标 exposure 分布统一自然采样、温度采样、手工配比、proxy-learned 与 online selection。"
topic: "pretraining-data"
section: "mixtures"
slug: "data-mixtures-and-sampling"
date: 2026-07-14
updated: 2026-07-15
order: 40
readtime: 17
source:
  repository: "J-shang/pt-data-learning"
  path: "knowledge-map/03-mixtures/data-mixtures-and-sampling.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/knowledge-map/03-mixtures/data-mixtures-and-sampling.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:9f2054ace8dba887c5a86ecf99b6e6a0d039a64db77852de03e331c34feb29ac"
  manifest: "pretraining-data"
  managed: true
---
> 层级：03 Mixtures
> 状态：`core`
> 初始资料核查截止：2026-07-14
<!-- maintenance: reasoning-path=`unified-framework` -->
> 证据说明：exposure 的期望关系在 sampler contract 下为 `verified`；“最佳 mixture”及 proxy 迁移为 `supported`、`plausible` 或 `open`

## 这篇笔记帮助你回答什么

数据 mixture 把静态 corpus 变成训练时的 token exposure 分布；同一批 unique data 仅改变采样权重和顺序，就会改变能力、记忆风险与优化轨迹。

## 为什么需要这个概念

数据集的原始占比并不等于模型实际看到的占比；按文档采样也不等于按 token 采样。需要一个共同对象，把自然采样、温度采样、手工配比、proxy-learned 和 online selection 都映射到可审计的目标/实际 exposure，同时保留它们不等价的状态与反馈路径。

本笔记用目标 token 分布 $q$ 作为统一框架。这个统一只覆盖“期望曝光”这一共同属性，不声称各种 sampler 在 batch correlation、时序、计算成本或优化行为上等价。

## 先看一个最小例子

假设 web 有 900M unique tokens，math 有 100M；总训练预算 1B。若目标 mixture 给 math 50%，math 平均曝光约 5 次，而 web 约 0.56 次。只写“50/50 mixture”会隐藏两个 domain 完全不同的重复程度。

## 核心定义

将训练数据分为 $K$ 个 domain，目标采样分布 $q$ 满足 $q_k\ge0$、$\sum_kq_k=1$。在总预算 $T$ 下，domain $k$ 的预期 exposure 为：

$$
E_k=Tq_k,\qquad r_k=\frac{E_k}{N_k}
$$

$N_k$ 是指定 tokenizer、指定数据版本下的 unique tokens；$r_k$ 是近似 exposure/epoch 倍数。真正执行后还要用 observed counts $\hat E_k$ 验证 sampler。

## 这些结论依赖哪些前提

| 关系或结论 | 类型与条件 | 置信状态 |
|---|---|---|
| $E_k=Tq_k$ | 目标为 token-level $q$、总预算固定时的 `exact expectation` | `verified` |
| $r_k=E_k/N_k$ | 平均曝光的 `approximation`；忽略 domain 内非均匀采样、内部重复、长度和 packing correlation | `supported` |
| 温度采样 $q_k\propto p_k^\alpha$ | 给定本文参数化的 `exact definition`；其他来源可能把 temperature 写成倒数 | `verified for this convention` |
| 乘法权重更新 | group-DRO 风格的 `illustrative generalization`，不是任意具体论文实现的等价公式 | `plausible template` |
| mixture 改变能力/记忆/优化轨迹 | `empirical association`；需要等模型、token、compute 与评测协议的 ablation 才能作因果判断 | `supported in studied settings` |

本文主要讨论 token-level mixture。若实现按 document/sequence/batch 采样，必须显式推导到 observed token/loss-token share；with-replacement、without-replacement 和阶段性 sampler 只在期望曝光层共享 $q$，不在时序和方差上等价。

## Alternative Explanations or Conflicts

“高 loss 数据更值得采样”和“高 loss 数据多为噪声”研究的隐藏对象不同：前者把 loss 当尚未学习的信号，后者把 loss 当解析错误或分布外信号。第一个分歧是高 loss 样本在人工质量、reference-model excess loss、后续 loss slope 和 held-out transfer 上是否表现不同。可用质量盲审、固定 exposure 的 slice ablation、多个 checkpoint 的 learnability 曲线和 parser/error 标签区分，而不能仅凭当前 loss 决策。

## 机制与相关知识

### 1. Domain 是决策变量，不是自然真理

domain 可以按 source、语言、体裁、主题、许可、时间、质量桶或其交叉定义。分得太粗，无法发现局部退化；分得太细，validation 方差大、sampler 和分析复杂。

好的 domain taxonomy 满足：

- 与目标能力或风险有明确关系；
- 在 source → train shard → validation 全链路保持标签；
- 每组有足够 unique tokens 和 held-out samples；
- 允许层级聚合，如 `web/zh/news`；
- 版本变化时能映射旧标签，或明确 breaking change。

### 2. 四类 mixture 机制

设 raw domain token share 为 $p_k=N_k/\sum_jN_j$。

**自然采样（proportional）**：

$$
q_k=p_k
$$

保留语料分布，但大 web/source 会主导，小域学习信号弱。

**温度采样**：

$$
q_k(\alpha)=\frac{p_k^{\alpha}}{\sum_jp_j^{\alpha}}
$$

$0<\alpha<1$ 会拉平分布、提高小域比例；$\alpha=1$ 回到自然采样；$\alpha>1$ 强化大域。不同资料对 temperature 的参数化可能用 $1/T$，阅读时必须核对公式，不能只比较“温度 0.3”。

**手工/约束 mixture**：依据能力目标、数据可信度和 exposure cap 直接设 $q_k$，常带上下界：

$$
\min_{q\in\Delta^K}\;J(q)
\quad\text{s.t.}\quad l_k\le q_k\le u_k
$$

**学习型/动态 mixture**：用 proxy loss、excess loss、gradient 或 learnability 信号更新权重。以 group DRO 风格为例，可用乘法权重：

$$
\tilde q_k^{(t+1)}=q_k^{(t)}\exp(\eta z_k^{(t)}),
\qquad
q_k^{(t+1)}=\frac{\tilde q_k^{(t+1)}}{\sum_j\tilde q_j^{(t+1)}}
$$

$z_k$ 是某种分域困难/超额损失信号，$\eta$ 为更新强度。具体方法的 reference model、平滑、裁剪与最终 resampling 语义必须逐篇核对。

### 3. 最小数值例子：比例与曝光不是一回事

三个 domain 有 unique tokens：web 800M、code 150M、math 50M，总预算 $T=2B$。

自然采样 $q=(0.8,0.15,0.05)$ 时：

$$
r=(2.0,2.0,2.0)
$$

若手工 mixture $q=(0.5,0.25,0.25)$：

$$
E=(1.0B,0.5B,0.5B),\qquad
r=(1.25,3.33,10.0)
$$

“math 只占 25%”听起来不夸张，但其实每个 math unique token 平均被看到约 10 次。应同时考虑数据内部重复，某些文档的实际 exposure 可能更高。

### 4. 有限数据、without replacement 与 exhaustion

实现 sampler 时要明确：

- domain 内是 with replacement、shuffle 后循环，还是用尽即停止；
- 跨 rank/worker 是否会意外重复或遗漏；
- shard 长度不齐时如何处理 remainder；
- 短/长文档和 packing 是否改变 token-level target；
- 权重按 document、sequence 还是 token 实现。

若目标是 token share $q_k$，却按 document 权重采样，长文档分布差异会导致实际 token share 偏离。正确性测试应比较：

$$
\epsilon_k=\frac{\hat E_k}{\sum_j\hat E_j}-q_k
$$

并给出有限样本预期波动，而不是要求每个 batch 精确等比例。

### 5. 静态、阶段性与在线选择的边界

| 方案 | 权重何时决定 | 额外状态 | 主要风险 |
|---|---|---|---|
| 静态 mixture | 训练前 | manifest + RNG state | 无法适应学习进度 |
| 阶段性/curriculum | 预设阶段边界 | phase、各阶段 sampler state | 阶段混杂，很难归因 |
| proxy-learned | 大训练前由小模型学习 | proxy/reference checkpoints、scores | proxy → target 不迁移 |
| online selection | 训练中按当前模型更新 | per-sample/domain state、在线分数 | 反馈回路、吞吐成本、偏置 |

动态不自动优于静态。若 scoring 成本、数据等待或 sampler 不稳定降低训练吞吐，必须把 wall-clock/compute 纳入收益；若权重基于 noisy loss，可能不断追逐噪声域。

### 6. Loss 高低不直接等于采样价值

- 高 loss：可能是尚未学会的有效长尾，也可能是解析错误、乱码、错误语言标签。
- 低 loss：可能是已掌握的基础，也可能是重复模板/泄漏/过易数据。
- loss decrease：可能表示 learnability，但也受 batch difficulty、长度和 tokenizer 影响。
- gradient 信号：更接近优化作用，但存储/计算昂贵且依赖当前模型状态。

因此 mixture 优化需要 validation 与样本审计，不应把单一 per-sample loss 直接映射为“价值”。

### 7. 可执行 sampler invariant

```text
inputs:
  frozen domain manifests with unique_token_count
  target token weights q[k]
  total token budget T
  seed and distributed topology

runtime counters:
  sampled_tokens[k]
  loss_tokens[k]
  unique_documents_seen[k]
  repeat_count_histogram[k]

checks:
  observed share ~= target share within sampling error
  all ranks have disjoint intended samples for a logical step
  resume reproduces sampler state without skip/replay
  packing does not erase domain attribution
```

## 它怎样影响 pretraining data 工作

关系类型：mixture target `implemented-by` sampler；observed exposure 与模型结果是需要 ablation 验证的 `empirically-associated-with`。

Mixture 是数据 curation 与优化过程的接口。过滤/去重决定 $N_k$ 和样本构成，mixture 决定 $E_k$；metrics/validation 再观察每个 $q_k$ 的能力与风险结果。任何数据集比较若不控制 exposure，就可能把“数据更好”与“看得更多”混在一起。

## 读完后应该掌握什么

- 能从 unique tokens、总预算和 $q$ 计算每域 exposure 倍数。
- 能推导温度采样并解释不同参数化。
- 能实现/审计 token-level sampler 和 distributed resume invariant。
- 能设计固定 compute/token 的 mixture paired ablation，并报告 per-domain 曲线。

## 常见误区

- 只报告 mixture 百分比，不报告 unique size 与 exposure。
- 目标是 token 比例，代码实现却按文档均匀采样。
- 把一个 proxy model 学到的权重无条件迁移到不同规模/tokenizer/预算的 target model。
- 每个 batch 强制固定比例，忽略这可能改变随机性和 batch correlation。
- 动态选择只算模型 step，不算 scoring pipeline 的额外 compute/wall-clock。

## 用这些问题检查自己

1. 两个 domain 的文档均数相同，但平均长度差 10 倍。按 document 50/50 采样会得到怎样的 token share？如何修复？
2. 某小域 validation loss 持续下降，但 downstream 指标先升后降。列出至少三个数据/评测机制并设计区分实验。
3. proxy mixture 在 280M 模型有效，迁移到 8B 前你会检查哪些不变量和失败模式？

## 来源与建议阅读位置

- [Xie et al., 2023, DoReMi](https://arxiv.org/abs/2305.10429) — 重点读 domain excess loss、group DRO/proxy model 与最终权重如何用于大模型数据重采样。
- [Gao et al., 2020, The Pile](https://arxiv.org/abs/2101.00027) — 用 component weights 练习从数据规模推 exposure，并观察开放混合语料的 domain taxonomy。
- [Hoffmann et al., 2022, Chinchilla](https://arxiv.org/abs/2203.15556) — 理解总 token budget 如何与模型/compute 约束耦合；不要把经验比例脱离设置使用。
- [Li et al., 2024, DataComp-LM](https://arxiv.org/abs/2406.11794) — 观察固定训练协议如何比较 filtering/mixing 策略，适合作为 mixture ablation 设计参照。
