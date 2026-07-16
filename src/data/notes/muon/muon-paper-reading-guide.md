---
title: "Muon 论文精读路线"
description: "沿前史、规模化证据、理论与数值边界、分布式系统四条线索精读 Muon 核心来源。"
topic: "muon"
section: "papers"
slug: "muon-paper-reading-guide"
legacyPaths: ["/notes/muon-paper-reading-guide/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
featured: true
order: 60
source:
  repository: "J-shang/Muon"
  path: "论文精读/README.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/README.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:2c6cdbbb60a6a259306e1f72a88047708962c85453c91ebf6a38ffafbc2ce9f8"
  manifest: "muon"
  managed: true
---
> 资料范围截至 **2026-07-14**；核心来源版本复核于 **2026-07-16**。
> 收录原则：优先一手论文、正式会议版本、作者技术报告和官方实现；“历史影响力”与“结论是否可靠”分开判断。
> 阅读目标：不是积累摘要，而是能重建公式、指出边界、对照代码并设计判别实验。

## 1. 为什么这份清单不是按引用数排序

Muon 在 2024 年下半年才出现，严格意义上的“经典 Muon 论文”还很少。这里把必读材料分成四层：

1. **前史与统一语言**：Shampoo、modular norm、SOAP、norm duality；它们解释 Muon 从哪里来、和什么不等价。
2. **Muon 主线与规模化证据**：原始设计说明、Moonlight、Practical Efficiency、Kimi K2。
3. **理论与数值边界**：谱范数约束、critical batch、inexact update、有限步 Newton–Schulz 收敛。
4. **重要前沿变体与系统**：Muon²、DMuon。它们很新，重要性来自问题本身和实验规模，不应提前称为“经典”。

本目录对每个核心来源建立一个文件。文件开头只保留原文入口、来源类型和阅读提醒；正文中的实验数字会写成“作者报告”，不会自动外推到其他模型或训练设置。

## 2. 核心精读清单与顺序

| 顺序 | 来源 | 为什么必读 | 来源类别 | 当前定位 |
|---:|---|---|---|---|
| 00 | [Muon 原始设计说明](/topics/muon/muon-original-design/) | 拿到算法最小合同和设计史；它不是论文 | 作者博客/官方实现入口 | 主线起点 |
| 01 | [Shampoo](/topics/muon/shampoo/) | 矩阵结构预条件的经典前史；可推导 instantaneous Shampoo 与 polar 的条件关系 | ICML 2018 同行评审 | 经典 |
| 02 | [Scalable Optimization in the Modular Norm](/topics/muon/scalable-optimization-modular-norm/) | 把层角色、operator norm 和可迁移 update scale 放进统一架构语言 | NeurIPS 2024 同行评审 | 理论地基 |
| 03 | [SOAP](/topics/muon/soap/) | 展示“旋转到结构化基底后运行 Adam”的设计空间，避免把 Muon 泛称为 Shampoo | 预印本 | 重要邻近方法 |
| 04 | [Old Optimizer, New Norm](/topics/muon/old-optimizer-new-norm/) | 从范数约束重建 Adam、Shampoo、Muon 的 steepest-descent 解释 | 理论预印本 | 核心解释 |
| 05 | [Modular Duality in Deep Learning](/topics/muon/modular-duality/) | 解释梯度为何是 dual object，以及矩阵方向如何从 dualization 得到 | 理论预印本 | 核心解释 |
| 06 | [Training Deep Learning Models with Norm-Constrained LMOs](/topics/muon/norm-constrained-lmos-scion/) | 把 polar update 放进线性最小化 oracle；区分 optimizer update 与参数约束 | 预印本 | 重要理论邻居 |
| 07 | [Muon is Scalable for LLM Training](/topics/muon/muon-is-scalable/) | weight decay、shape scale、distributed Muon 和 scaling law 的主线报告 | 大规模技术预印本 | 核心实证 |
| 08 | [Practical Efficiency of Muon for Pretraining](/topics/muon/practical-efficiency-of-muon/) | 多尺度、large batch、muP 和 compute/time Pareto | 多尺度实验预印本 | 核心实证 |
| 09 | [Muon Optimizes Under Spectral Norm Constraints](/topics/muon/muon-spectral-norm-constraints/) | 将 matrix sign、核范数与解耦 decay 的隐式谱约束连起来 | 理论预印本 | 核心理论 |
| 10 | [Convergence Bound and Critical Batch Size](/topics/muon/muon-critical-batch-size/) | 同时分析 momentum、weight decay、rank/noise 与 critical batch | 理论+实验预印本 | 核心理论 |
| 11 | [Kimi K2](/topics/muon/kimi-k2-muonclip/) | 超大规模采用 Muon 的案例，以及 MuonClip 对 QK logit spike 的系统补丁 | 大规模技术报告 | 规模案例 |
| 12 | [Beyond the Ideal](/topics/muon/inexact-muon-update/) | 把 NS 近似误差从“实现细节”提升为需与 LR/momentum 联调的变量 | 理论+小规模实验预印本 | 数值必读 |
| 13 | [Convergence of Muon with Newton–Schulz](/topics/muon/muon-convergence-newton-schulz/) | 直接研究有限步 NS，而不是用精确 SVD 替代实际算法 | ICLR 2026 论文 | 数值必读 |
| 14 | [Muon²](/topics/muon/muon2/) | 研究 polar 前二阶统计如何改善有限步 NS 输入谱 | 2026 预印本 | 前沿待复现 |
| 15 | [DMuon](/topics/muon/dmuon/) | 保持完整矩阵语义时，如何把分布式 Muon 的额外成本压到近 AdamW | 2026 系统预印本 | 前沿待复现 |

## 3. 推荐的三轮读法

### 第一轮：只回答“对象是什么”

按 `00 → 01 → 04 → 05 → 07` 阅读。每篇只记录：被变换的是 gradient、momentum、update 还是 parameter；用的范数是什么；persistent state 是什么。

完成标准：能从薄 SVD $M=U\Sigma V^\top$ 推出 $UV^\top$，并解释为什么“Muon 正交化权重”和“Muon 就是 Shampoo”都不准确。

### 第二轮：解释真实训练配方

按 `02 → 03 → 07 → 08 → 11` 阅读。建立同一张表：parameter routing、update scale、weight decay、batch、token、FLOP、wall-clock、memory、调参预算。

完成标准：任何“快 2×”的说法都能还原为明确的横轴、模型范围、数据预算和作者报告。

### 第三轮：寻找理论—实现裂缝

按 `06 → 09 → 10 → 12 → 13 → 14 → 15` 阅读。重点看 exact polar 与 finite-step NS、full matrix 与 shard/block、数值误差与训练最优点之间的差异。

完成标准：提出一个能让两种解释产生不同可观测输出的实验，而不是只复述作者直觉。

## 4. 扩展论文池

下面这些材料值得继续追踪，但截至 2026-07-14 还不足以和核心主线同级定论。它们已按“改变的对象”整理在 [前沿变体与开放问题](/topics/muon/muon-frontiers/)：

| 来源 | 改变的主要对象 | 当前处理 |
|---|---|---|
| [AdaMuon](https://arxiv.org/abs/2507.11005) | polar 后/周围的 adaptive statistics | 前沿预印本，待独立复现 |
| [8-bit Muon](https://arxiv.org/abs/2509.23106) | momentum state 的量化表示 | 先验证 state bytes、误差与恢复一致性 |
| [NorMuon](https://arxiv.org/abs/2510.05491) | neuron/row 级二阶归一化 | 与 full/factorized second moment 做状态匹配对照 |
| [MuonAll](https://arxiv.org/abs/2511.06086) | 把 Muon 扩展到原本非矩阵参数 | reshape 本身是归纳偏置，仍待验证 |
| [MiMuon](https://arxiv.org/abs/2605.19619) | 混合正交化与 momentum-SGD 方向 | 需要固定混合规则和理论假设 |
| [MONA](https://arxiv.org/abs/2605.26842) | gradient-difference/EMA acceleration | 大规模作者报告，待同预算复现 |
| [Muon in Adversarial Training](https://arxiv.org/abs/2605.26929) | 应用域与鲁棒优化动力学 | 不从预训练结论直接外推 |
| [Muown](https://arxiv.org/abs/2605.10797) | row norm / neuron update control | 需区分 exact-polar 方向与后处理 |
| [Tensorion](https://arxiv.org/abs/2606.25975) | 从矩阵 polar 推广到高阶 tensor | 属于 generalization，不是简单路由开关 |

此外还有围绕 polar polynomial、Gram kernel 和低秩/分布式路径的实现工作；只有找到稳定的一手版本并建立可比实验后，才进入清单。方法名本身不构成新的机制证据。

升级为核心精读的条件至少满足其一：有独立复现；改变了主线理论判断；在清晰披露 recipe 的更大规模上稳定成立；或已进入正式同行评审版本。

## 5. 每篇笔记怎样标明出处

每篇文件都必须区分：

- **论文报告**：作者直接声称或展示的结果；
- **本文推导**：本项目能逐步核查的数学关系；
- **跨论文比较**：把多篇来源连接起来的解释；
- **开放问题**：尚不能由现有证据裁决的结论。

判断重点不在标签本身，而在原文定位和适用边界。大规模数字即使来自一手论文，也仍写成作者在该实验范围内的报告。

## 6. 逐篇复核后最重要的校正

截至 2026-07-16，`00`–`15` 每个文件都回答三个问题：

1. 原文还有哪些值得注意但首轮遗漏的内容？
2. 它与其他论文的观点是真冲突，还是对象、假设、尺度、指标或近似不同？
3. 本文每个主要知识点来自原文、作者实验、本文推导还是跨论文比较？

本轮最重要的校正如下：

| 原先容易形成的印象 | 核查后的准确表述 |
|---|---|
| instantaneous Shampoo = polar 是 2018 Shampoo 的主结论 | 该关系由后续 *Old Optimizer, New Norm* / Muon 说明明确提出；可由 Shampoo 公式推导，但不能回溯归因给原论文 |
| *Modular Duality* 直接提出或讨论 Muon | 正文没有使用 Muon 名称；它提出 rectangular NS duality map，和 Muon 的关系是时间线与公式上的后见综合 |
| Muon 是/不是二阶法已经有共识 | 文献存在真实的分类口径分歧：*Practical Efficiency* 采用宽泛的“matrix structured = second-order”称呼，finite-NS 论文要求 curvature estimate 才算二阶 |
| “critical batch size”在各论文含义相同 | SOAP/部分实证论文用 linear-scaling break point；Sato 等定义为最小化 $bT(b)$ 的 SFO minimizer |
| 更精确 polar 必然带来更好训练 | finite-NS/inexact 理论改善的是 error/rate；Moonlight 在固定 recipe 下观察更精确未必改善 loss。模型、kernel、超参重调与指标均不同 |
| Muon 论文共享一套 weight decay recipe | Moonlight/spectral-constraint 重点是 decoupled decay；*Practical Efficiency* 明确研究 coupled weight decay 与其 $\mu$P transfer |

若正文没有对应陈述，但公式可直接推出，本目录把它写成“本文推导”；若需要连接两篇以上来源，则写成“跨论文比较”。两者都必须给出可核查路径，也不能冒充作者原话或论文结论。

## 7. 总通过检查

完成全部核心笔记后，应能交付以下可观察产出：

1. 一页从 spectral-norm trust region 到 polar direction 的推导。
2. 一张 Muon、Shampoo、SOAP、AdamW 的 object/operation/state/cost/failure-mode 对照表。
3. 一个 exact SVD polar 对不同 NS steps/dtype/shape 的数值测试。
4. 一个同一 global gradient 在不同 TP/shard 布局下的 update 一致性测试。
5. 一张同时含 loss-vs-token、FLOP、time 和 peak-memory 的实验图，而不是只画 steps。
