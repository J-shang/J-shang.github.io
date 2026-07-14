---
title: "Muon 论文与证据索引"
description: "用来源类别、结论置信度和 claim ledger 管理 Muon 主线结论与证据缺口。"
topic: "muon"
section: "research-practice"
slug: "muon-evidence-index"
legacyPaths: ["/notes/muon-evidence-index/"]
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 80
source:
  repository: "J-shang/Muon"
  path: "notes/论文与证据索引.md"
  url: "https://github.com/J-shang/Muon/blob/65164a375bd729b71f0e89b03642c67c50e624b3/notes/%E8%AE%BA%E6%96%87%E4%B8%8E%E8%AF%81%E6%8D%AE%E7%B4%A2%E5%BC%95.md"
  revision: "65164a375bd729b71f0e89b03642c67c50e624b3"
  syncedAt: "2026-07-14"
  contentHash: "sha256:631af5bc894f1286719a48ad4c9d85d7206e0a1219bd1e114937de2b47c2c0fd"
  manifest: "muon"
  managed: true
---
> 范围：Muon 主线论文、官方实现/文档和决定学习结论的代表性前沿工作。
> 信息截点：2026-07-14。
> 用法：先查“结论账本”，再按阅读路径进入原始来源；不要按论文数量投票。

逐篇精读正文见 [Muon 论文精读路线](/topics/muon/muon-paper-reading-guide/)。该目录为每个核心来源单独记录公式、证据边界、可核查锚点、实现/实验任务和推理型自测；本页继续承担跨论文 claim ledger。

## 一页结论

目前可以较有把握地说：Muon 对隐藏层二维矩阵的 momentum update 做有限步 Newton–Schulz 谱变换；现代大规模配方还依赖参数路由、shape-aware update scaling、解耦 weight decay 和分布式矩阵语义。这些属于算法合同，不是可忽略的外围实现。

大模型预训练收益已有多个规模化来源支持，但“约 2×”仍是指定 compute-optimal scaling-law 设置下的作者报告，不是跨架构定律。微调、RL、非 Transformer、不同并行布局和不同 NS kernel 仍需要分别验证。

## 证据语言

### 来源类别

| 类别 | 能直接支持什么 | 不能自动支持什么 |
|---|---|---|
| 数学推导/教材 | 恒等式、定理及其假设 | 现实训练收益 |
| 官方代码/文档 | 指定版本的参数、分支和执行行为 | 跨版本稳定性、训练效果 |
| 大规模技术报告 | 指定 recipe/硬件/数据下的结果 | 因果机制、跨设置外推 |
| 多尺度实验论文 | 受控范围内的趋势和消融 | 未测试架构或训练阶段 |
| 早期预印本 | 值得检验的机制与作者报告 | 已独立复现的共识 |
| 博客/个人说明 | 设计史、直觉、最小实现入口 | 高置信度的规模化结论 |

### 结论置信度

- `verified`：在声明范围内由推导、版本固定的代码走读或可复现实验直接确认。
- `supported`：多个相关检查一致，但范围和剩余限制已写明。
- `plausible`：解释自洽且有部分证据，仍缺少区分性检查。
- `open`：证据不足、相互冲突或尚未检查。

“来源是一手”与“结论 verified”不是一回事。例如，一篇预印本是一手来源，但它对自身大规模收益的结论仍可能只是单一团队的作者报告。

## 结论账本

| 结论 | 关系强度 | 当前判断 | 最强支持 | 仍缺什么 |
|---|---|---:|---|---|
| 精确 polar factor 为 $UV^\top$ | exact（薄 SVD 约定下） | verified | 线性代数推导 | 秩亏时需说明非唯一性 |
| 有限步低精度 NS 得到精确 polar | approximation，不是 exact | verified（否定强等价） | 迭代公式、数值对照 | 不同系数/kernel 的误差面 |
| shape scaling 与 weight decay 对规模化 Muon 很关键 | empirical + implementation | supported | *Muon is Scalable* 及其代码 | 跨架构独立消融 |
| Muon 达到 AdamW 同等表现约需一半 FLOPs | empirical，特定 operating point | supported in-scope | Moonlight scaling-law 报告 | 独立复现、其他数据/架构/预算 |
| Muon 在更大 batch 保持较好 data efficiency | empirical | supported in-scope | *Practical Efficiency of Muon* 多尺度实验 | 更长 horizon、更多训练栈复现 |
| 谱压平导致更好的表征/泛化 | causal claim | plausible | SVD entropy 与训练结果的关联 | 干预谱变换的区分性消融 |
| Muon 是“二阶优化器” | analogy / framework-dependent | open as taxonomy | Shampoo 代数联系、norm geometry | 先固定“二阶”的定义；Muon 不显式用 Hessian |
| AdamW checkpoint 可无风险切换 Muon 微调 | empirical generalization | open，且已有反例信号 | 2026 微调预印本 | 多任务、SFT/RL、更新约束的独立验证 |

## 推荐阅读路径

### 路径 A：从约束推导更新方向

1. [Muon 原始设计说明](https://kellerjordan.github.io/posts/muon/)：先拿到最小算法和设计史；这是作者博客，不承担大规模证据。
2. [Old Optimizer, New Norm](https://arxiv.org/abs/2409.20325)：建立 norm/dual norm 的统一语言。
3. [Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265)：理解 layer role、operator norm 与 dualization 的关系。
4. [Shampoo](https://proceedings.mlr.press/v80/gupta18a.html)：核对“instantaneous Shampoo”到底删掉了什么历史统计。

输出：独立推导谱范数单位球上线性目标的最陡方向，并明确它与 Shampoo 是特例/近似/类比中的哪一种关系。

### 路径 B：从小算法走到规模化配方

1. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)：读 weight decay、update RMS、分布式方案和 scaling law。
2. [Moonlight 官方仓库](https://github.com/MoonshotAI/Moonlight)：把论文公式对应到参数分组、状态分片和通信。
3. [Practical Efficiency of Muon for Pretraining](https://arxiv.org/abs/2505.02222)：这是一篇多尺度实验预印本；读 muP、critical batch 与 compute-time Pareto frontier。
4. [Kimi K2 技术报告](https://arxiv.org/abs/2507.20534)：把 MuonClip 当作超大规模训练案例，不把整套系统收益归给优化器。

输出：制作一张 claim-to-evidence 表，每条收益注明模型、token、batch、调参预算、横轴和基线。

### 路径 C：数值近似和分布式系统

1. [Beyond the Ideal: Analyzing the Inexact Muon Update](https://arxiv.org/abs/2510.19933)：研究有限步 NS 与训练超参数的耦合。
2. [本地 Megatron-LM 实现解析](/topics/muon/megatron-muon-implementation/)：追踪 global/local shape、TP mode 和 layer-wise owner rank。
3. [NVIDIA Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)：固定版本后检查实际 NS 系数、scale mode 和状态。
4. [DMuon](https://arxiv.org/abs/2606.27153)：把近 AdamW overhead 当作作者在其硬件/工作负载上的性能报告，等待独立复现。

输出：给出一个能区分 `duplicated`、`distributed` 与 `blockwise` 的多 rank correctness/performance test。

## 单篇论文记录模板

```text
source / version / date:
source class:
question and comparison axes:
model / data / tokens / hardware:
optimizer routing and full recipe:
reported fact:
authors' mechanism explanation:
my derivation or synthesis:
assumptions and omitted effects:
confidence: verified | supported | plausible | open
discriminating check:
```

把“reported fact”“mechanism explanation”和“my synthesis”分开。若两篇解释冲突，先对齐它们研究的对象、尺度、指标和近似，再找第一处分歧。

## 当前证据缺口

- 长 horizon 和数据重复下，Muon 的 batch 优势是否持续。
- 同等调参预算下，非 Transformer、dense/MoE、SFT/RL 的 Pareto frontier。
- NS 逼近误差、update scale 和训练收益之间的可干预因果关系。
- 数学等价的 distributed polar 与工程近似在真实拓扑上的成本边界。
- Muon 预训练 checkpoint 的最优继续训练、SFT 和 RL optimizer policy。
