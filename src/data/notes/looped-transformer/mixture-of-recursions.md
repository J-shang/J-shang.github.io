---
title: "Mixture-of-Recursions：Token-Level Dynamic Recurrent Depth"
description: "拆解 token-wise routing、共享 recursion block 与 selective KV cache，并区分 isoFLOP、吞吐和 token budget 口径。"
topic: "looped-transformer"
section: "llm-pretraining"
slug: "mixture-of-recursions"
date: 2026-08-02
updated: 2026-08-04
cutoff: 2026-08-02
order: 52
source:
  repository: "J-shang/looped-transformer"
  path: "papers/16-mixture-of-recursions.md"
  url: "https://github.com/J-shang/looped-transformer/blob/eb43191df7da7f8d1b936fa6485ea21f7c8f430a/papers/16-mixture-of-recursions.md"
  revision: "eb43191df7da7f8d1b936fa6485ea21f7c8f430a"
  syncedAt: "2026-08-04"
  contentHash: "sha256:435c3c9ca43bd5602076f40b05934ad8ef9d86297d75a39b7d969fc2da045b9f"
  manifest: "looped-transformer"
  managed: true
---
> 论文：*Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*<br>
> 精确版本：[arXiv:2507.10524v3](https://arxiv.org/abs/2507.10524v3)，2025-10-25；NeurIPS 2025<br>
> 作者与机构：Sangmin Bae、Yujin Kim、Reza Bayat 等；KAIST AI、Mila、Google Cloud、Google DeepMind、Google Research、Université de Montréal<br>
> 官方 artifact：[code](https://github.com/raymin0223/mixture_of_recursions)<br>
> 证据标签：**B/C**。训练代码公开、实验覆盖 135M–1.7B；但没有现代 post-training 或旗舰通用模型证据。

## 1. 30 秒结论

MoR 同时做两件事：用 shared recursion block 降低 stored parameters，用 router 为每个 token 分配不同的 recursive depth。它还设计 recursion-wise KV cache 和 recursive KV sharing，使“跳过后续 recurrence”能转化为部分实际推理收益。

真正需要记住的不是摘要里的“new Pareto frontier”，而是 appendix 中更复杂的事实：在某些固定训练 FLOPs 配置下，1.7B MoR 略胜 vanilla；在另一些严格切分下，vanilla 平均准确率反而更高。MoR 的结果高度依赖 token budget、参数量与执行计算如何分配。

## 2. 学习目标

1. 区分 expert-choice 与 token-choice routing；
2. 理解 token-wise variable depth 对 causal attention/KV cache 的影响；
3. 比较 Cycle、Sequence、Middle-Cycle、Middle-Sequence 四种 parameter sharing；
4. 审核 isoFLOP、throughput 与 token-count 三种口径；
5. 识别论文中被主结果弱化的 negative evidence。

## 3. 符号与最小例子

| 符号 | 含义 |
|---|---|
| $N_r$ | 最大 recursion depth |
| $B$ | 每次重复的 shared layer block |
| $g_i^{(r)}$ | token $i$ 在 recursion $r$ 的 router score |
| $z_i^{(r)}$ | token $i$ 是否在该 recursion 继续计算 |
| $C_r$ | 第 $r$ 层可接收的 token capacity |

若一句话中 token depth 为：

```text
People  who  feel  comfortable ...
  1      2    3        2
```

那么第 1 次 recursion 处理全部 token，第 2 次只处理 depth 至少为 2 的 token，第 3 次只处理 depth 为 3 的 token。它比 sequence-level exit 更细，但也要求 attention 与 cache 正确处理“有些历史 token 已退出”的状态。

![MoR 架构与 token-wise routing pattern](/assets/looped-transformer/16-mixture-of-recursions/figure-1-overview.png)

*原论文 Figure 1，PDF p.1，[arXiv v3 PDF](https://arxiv.org/pdf/2507.10524v3)。看图重点：中间是共享 recursion block，右侧深浅格表示每个 token 的执行深度不同。它展示 routing 语义，不代表路由本身在 causal decoding 中零开销。*

## 4. Parameter-sharing strategies

| 策略 | 做法 | 直觉与代价 |
|---|---|---|
| Cycle | 按固定周期重复层组 | 共享最多、结构规则；不同 logical depth 接收同一参数 |
| Sequence | 按层序列分段复用 | 保留某些局部顺序，映射更复杂 |
| Middle-Cycle | 首尾层独立，中间按周期复用 | 让 embedding-side 与 readout-side 各自专门化 |
| Middle-Sequence | 首尾独立，中间按序列复用 | 同上，但中间分组方式不同 |

Middle variants 的重要性在于：最早层负责把 token/position 表示变成工作空间，最后层负责映射到 LM head，它们可能不适合与中间迭代层完全共享。这个观察也连接 Huginn 的 prelude/core/coda 和 retrofit 的 model surgery。

## 5. 两种 routing

### 5.1 Expert-choice

每个 recursion step 从全部候选 token 中选 top-$k$，容量固定、load balance 好。问题是 top-$k$ 排名需要同时看到一组 token 的 score；如果未来 token 参与决定当前 token 是否被选，就会产生 causal leakage。训练可并行，但标准 autoregressive decoding 中不能原样使用。

### 5.2 Token-choice

每个 token 在进入 recursion 前独立选择 depth。它满足 causal inference，却容易产生负载不均：某个 batch 可能大量 token 都选最深层，使实际 latency 接近 worst case。

因此没有“最好的 router”，只有不同部署约束下的选择：

| 目标 | 更自然的选择 | 主要风险 |
|---|---|---|
| 训练吞吐与固定容量 | expert-choice | causal leakage / train–infer mismatch |
| 严格 autoregressive 推理 | token-choice | imbalance、dynamic batching 困难 |

## 6. KV cache 设计

### 6.1 Recursion-wise KV cache

每个 depth 只缓存仍活跃 token 的 key/value。它与 token-wise routing 语义一致，可省内存和后续 attention，但 cache layout 变成 ragged；gather/scatter、索引和 kernel launch 可能吞掉 theoretical saving。

### 6.2 Recursive KV sharing

后续 recursion 复用第一次 recursion 的 K/V，不再重复 KV projection。优点是 memory/compute 更低；缺点是 query 在更新、K/V 却固定，模型看到的是混合时间尺度。该设计主要与 Cycle sharing 兼容，且引入额外 architecture mismatch。

![MoR 的 likelihood–throughput 与 cache 消融](/assets/looped-transformer/16-mixture-of-recursions/figure-4-throughput.png)

*原论文 Figure 4，PDF p.8，[arXiv v3 PDF](https://arxiv.org/pdf/2507.10524v3)。看图重点：MoR 在作者实现和固定 batch 假设下形成更好的部分 Pareto 点；caption 明确说明 throughput 不含 KV-cache update time，因此不能直接当端到端 serving latency。*

## 7. 实验设置

| 维度 | 设置 |
|---|---|
| model scale | 135M、360M，扩展到 1.7B |
| data | deduplicated FineWeb-Edu / SmolLM corpus |
| 最大训练量 | 约 220B tokens；不同实验切片不同 |
| context window | 2K |
| hardware | H100 / A100，4 GPUs 为主 |
| tasks | LAMBADA、HellaSwag、PIQA、WinoGrande、ARC、MMLU |
| comparisons | vanilla、fixed recursive baselines、不同 sharing/router/cache |

它属于 broad base-LM 证据，但离现代长 context、instruction/chat、tool use 和 safety model flow 仍有明显距离。

## 8. 如何读结果而不被单一表格误导

### 8.1 固定较大训练 FLOPs 的 1.7B 对照

在约 $68.5\times10^{18}$ FLOPs 的 appendix 设置中：

| 模型 | 平均准确率 |
|---|---:|
| vanilla 1.7B | 48.9 |
| MoR，$R=2$ | 48.4 |
| MoR，$R=3$ | 46.7 |

这是重要的 negative evidence：更深 recurrence 没有在这组固定预算下胜过 vanilla。

### 8.2 另一种 compute/token allocation

在约 $16.5\times10^{18}$ FLOPs 的表中：

| 模型 | seen tokens | 平均准确率 |
|---|---:|---:|
| vanilla 1.7B | 4.8B | 40.54 |
| MoR 1.7B | 6.5B | 40.97 |

MoR 略胜，但也多看了约 35% tokens。这个结果可以支持“parameter sharing 让模型在同 FLOPs 下处理更多 data”，不能支持“相同 data exposure 时 recurrence 本身更强”。

### 8.3 正确结论

MoR 的价值是展示了一个动态 token-depth 系统与若干 Pareto 改进点；当前证据不足以说明它在 1.7B 以上、full post-training 或真实 serving 中稳定优于 vanilla。

## 9. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| architecture | parameter sharing + adaptive token depth | 强 |
| routing | expert-choice 与 token-choice 两条路径 | 中强；各有明确限制 |
| systems | recursion-wise cache、KV sharing、depth-wise batching | 中；部分 throughput 未含 cache update |
| scaling | 135M–1.7B isoFLOP studies | 中；强配置结果不完全一致 |
| post-training | 无完整证据 | 缺失 |

## 10. Claim–evidence map

| Claim | Evidence | Boundary |
|---|---|---|
| token 可以分配不同 recurrence | router 与 routing map | 不等于真实请求 latency 按比例下降 |
| MoR 有部分更优 Pareto 点 | likelihood/throughput plots | 测量假设、batch size、cache update 影响结论 |
| parameter sharing 可换更多 seen tokens | isoFLOP table | data exposure 已改变，不能隔离 architecture |
| KV sharing 降低 memory/projection cost | cache design + ablation | state/KV mismatch 可能损害质量 |
| 1.7B 已证明 scaling | 单一小规模扩展 | 不能外推到 7B/70B 或 MoE |

## 11. 局限

1. context window 2K，不代表长 context cache 行为。
2. router 的训练可并行性与 causal deployment 存在结构张力。
3. throughput figure 排除了 KV-cache update，且依赖 fixed/max batch assumptions。
4. 1.7B 强配置中 vanilla 仍略优；论文结论依赖预算切法。
5. 没有统一 SFT/DPO/RLVR 后的 instruction、chat、safety、tool-use 评测。

## 12. 推荐复现实验

1. 用同一 tokenizer/data order 构造 vanilla 与 MoR；
2. 同时固定 unique tokens、seen tokens、analytical FLOPs、wall-clock 四个口径；
3. 记录 router depth histogram、per-token entropy 与 task difficulty；
4. 把 cache gather/scatter、router、KV update 都纳入端到端 latency；
5. 在 static batch、continuous batch、batch size 1 三种服务模式下复测；
6. 对 post-training 前后 router calibration 做比较。

## 13. 自测题

1. expert-choice 为什么可能破坏 causality？
2. token-choice 为什么可能在平均 FLOPs 较低时仍没有 latency 优势？
3. 4.8B 与 6.5B seen tokens 的比较中，architecture claim 应怎样改写？
4. recursion-wise KV cache 与 recursive KV sharing 的 state 语义有什么不同？
5. 如果 throughput 不计 cache update，你还需要哪些测量才能判断部署价值？

## 14. 一句话定位

MoR 是“parameter sharing + per-token adaptive depth + cache-aware execution”的关键设计论文，但它的规模实验也表明：动态 recurrence 的优势取决于真实 data/compute/system allocation，不能从一个 Pareto 图直接推广到通用 LLM。
