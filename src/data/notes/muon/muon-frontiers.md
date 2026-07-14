---
title: "Muon 前沿变体与开放问题"
description: "按训练统计与 polar 实现两个主轴比较前沿变体，并给出可证伪的研究问题。"
topic: "muon"
section: "research-practice"
slug: "muon-frontiers"
legacyPaths: ["/notes/muon-frontiers/"]
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 82
source:
  repository: "J-shang/Muon"
  path: "notes/前沿变体与开放问题.md"
  url: "https://github.com/J-shang/Muon/blob/f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13/notes/%E5%89%8D%E6%B2%BF%E5%8F%98%E4%BD%93%E4%B8%8E%E5%BC%80%E6%94%BE%E9%97%AE%E9%A2%98.md"
  revision: "f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13"
  syncedAt: "2026-07-14"
  contentHash: "sha256:927a84a75cb97642a952f9c4efc4a020f4ad6ea1c52fb3d1365ad5f0936fc2d0"
  manifest: "muon"
  managed: true
---
> 信息截点：2026-07-14。
> 证据边界：本页多数条目是 2025–2026 年预印本或团队技术报告；“论文报告”不等于独立复现。
> 推理路径：先用共同轴比较方法，再定位它们改变的对象；拒绝只有名字的 taxonomy。

## 两个主轴

1. **训练统计轴**：改变进入 polar 前或离开 polar 后的历史统计、方差估计或加速项。这会改变训练动力学和持久状态。
2. **polar 实现轴**：试图用更少 GEMM、更小矩阵、更低精度或更好的分布式布局逼近同一个方向。这主要改变数值误差和系统成本，但某些近似也会反过来改变动力学。

不能因为两个方法都“让 Muon 更快”就把它们放在同一机制里。

## 方法族比较

| 方法 | 被改变的对象 | 核心操作 | 新增持久状态 | 实现/验证路径 | 来源与当前判断 |
|---|---|---|---|---|---|
| MuonClip | attention Q/K 权重，位于 optimizer update 之后 | max QK logit 超阈值时缩放 Q/K | 监控统计，通常无 Adam 式完整二阶矩 | 记录 per-head max logit、触发率、裁剪量与 loss spike | Kimi K2 大规模技术报告；`supported in-scope` |
| NorMuon | polar 后的 neuron/row update | row-wise 二阶统计归一化 | 每行/神经元统计 | 对照状态字节、row RMS 与原 Muon | 预印本；`plausible`，待独立复现 |
| AdaMuon / Adaptive Muon | polar 后 update | adaptive second-moment scale | 新增二阶统计 | state round trip + 与 Megatron adaptive branch 对齐 | 预印本/实验实现；`plausible` |
| Muon² / Muon²-F | polar 前 momentum | Adam-style full/factorized precondition，再做少步 NS | full 或 factorized second moment | 比较 3/5 步方向误差、step time 和训练 loss | 2026 预印本；`plausible` |
| Muon-NSR / Muon-VS | polar 前 momentum | noise-to-signal 或 variance scaling | 方差相关状态 | 固定 scale/NS 后做独立消融 | 2026 预印本；`plausible` |
| 8-bit Muon | momentum state 表示 | blockwise quantize/dequantize | 量化 momentum + scale metadata | state bytes、量化误差、恢复训练和端到端吞吐 | 预印本；作者报告到 2.7B，`plausible` |
| MuonAll | 原本非矩阵/非 hidden 参数 | 先指定 reshape，再做 Muon | 取决于路由 | 逐类声明 reshape 的归纳偏置和反例 | 小规模预印本；`open` |
| Riemannion | 固定秩 LoRA 参数所在流形 | Riemannian update/retraction | 流形 optimizer state | 检查 rank、tangent projection、retraction | 预印本；是 generalization，不是开关 |
| GramMuon | NS 的计算对象 | 在较小 Gram 矩阵迭代并为低精度加 restart | 通常无新增长期训练统计 | 同方向误差 + kernel/端到端 profile | 作者博客/实现；`plausible` 系统收益 |
| DMuon | 分布式 polar 和通信路径 | drop-in kernel/layout/communication 优化 | 实现相关 | 固定硬件与矩阵分布复现 step breakdown | [2026 预印本](https://arxiv.org/abs/2606.27153)；作者报告 near-Adam overhead，`plausible` |
| MONA | 进入 Muon 的梯度处理 | gradient difference EMA acceleration | 额外 EMA state | 复现 sharpness 定义、相同预算 MoE A/B | [2026 预印本](https://arxiv.org/abs/2605.26842)；作者报告最高 68B/1T，`plausible` |
| MiMuon | Muon 与 momentum-SGD 的混合方向 | 有选择地混合正交化与非正交更新 | 取决于混合规则 | 检查 generalization bound 假设和受控训练 | [2026 预印本](https://arxiv.org/abs/2605.19619)；`open` |

## 四个容易被揉在一起的争议

### 1. “更精确 polar”是否带来更好训练

- 数值问题：输出与 $UV^\top$ 的距离是否更小。
- 优化问题：在固定 token/FLOP/time 下 loss 是否更低。

第一处分歧在评价指标。可区分检查：对同一 momentum 输入生成一组不同精度但 update RMS 对齐的方向，同时测 polar error 与训练 loss；若二者排序不同，就不能用数值精度替代训练结论。

### 2. “Muon 是一阶还是二阶”

- 一阶几何解释研究：只用当前/历史梯度，在非欧氏范数下 dualize。
- 二阶/预条件解释研究：强调与 Shampoo 或矩阵预条件的代数关系。

第一处分歧在“二阶”的定义：是否必须估计 Hessian/梯度协方差，还是只要非对角矩阵变换即可。学习项目应保留两个框架，避免把 taxonomy 当成机制结论。

### 3. “局部分片 Muon 是否仍是 Muon”

- full/global 路径的对象是完整语义矩阵。
- blockwise 路径的对象是一组局部子矩阵。

这不是视角差异，而是对象不同。判别测试是同一个 global gradient 在不同 TP size 下恢复出的 global update 是否保持不变。

### 4. “预训练收益是否迁移到微调”

- 从头预训练改变整个优化轨迹。
- 从 AdamW checkpoint 切换 Muon 会在已有表示上突然改变几何与隐式偏置。

第一处分歧在初始状态和训练阶段。实验必须拆成 Muon→Muon continuation、AdamW→Muon full FT、AdamW→Muon LoRA、SFT 和 RL，不能合并报告。

## 开放问题与判别实验

| 开放问题 | 不能接受的证据 | 最小判别检查 |
|---|---|---|
| 谱压平究竟改善什么 | 只展示 singular-value entropy 与 loss 相关 | 保持 RMS/compute 相同，干预谱映射并做多 seed ablation |
| 大 batch 优势能否持续 | 只看 steps 或短 horizon | loss-vs-token/FLOP/time，覆盖数据重复和重新调参 |
| 每层是否应有不同 NS 精度 | 只比较全局 `ns_steps` | 按 shape/condition proxy 分桶，测方向误差与 layer update statistics |
| distributed approximation 是否值得 | 只报 optimizer kernel speedup | 同时报告 global-update error、整步时间、通信量、最终 loss |
| adaptive variants 是否真的需要二阶状态 | 只比最终 loss | 与状态/算力匹配的 row-wise、factorized、no-state 基线 |
| 微调 mismatch 如何缓解 | 单任务单 seed | optimizer transition、update constraint、LoRA/full FT 的交叉设计 |

## 更新规则

- 新方法先填“对象、操作、状态、实现路径、失败模式”，再决定属于哪个轴。
- 规模数字只写成“作者报告”，同时记录 active/total params、tokens、硬件和基线。
- 至少有独立复现或仓库内可重现实验后，才把 `plausible` 升为 `supported`。
- 新论文若只是换名字而没有可区分操作，不新增条目。
