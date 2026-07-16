---
title: "Muon 的收敛界与 Critical Batch Size"
description: "对齐 momentum、weight decay、rank、noise 与 critical batch 的理论定义和实验口径。"
topic: "muon"
section: "papers"
slug: "muon-critical-batch-size"
legacyPaths: ["/notes/muon-critical-batch-size/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 71
source:
  repository: "J-shang/Muon"
  path: "论文精读/10-Critical-Batch-Size.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/10-Critical-Batch-Size.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:04f02c951b2911b892115884919adcf5dc66d523d561efdf80975bdd66461b5f"
  manifest: "muon"
  managed: true
---
> 原文：[arXiv:2507.01598](https://arxiv.org/abs/2507.01598)，核验版本 v5（2026-06-08）
> 来源类型：理论与实验预印本
> 阅读提醒：定理只在论文假设下成立；作者给出定性的超参数趋势，不预测普适的最佳 batch。

## 它解决什么问题

论文系统分析四种 Muon：有/无 Nesterov、以及有/无 weight decay。它给出收敛界，说明 decay 可在不先假设 bounded gradient 的情况下帮助控制参数与梯度，并从 stochastic first-order oracle（SFO）复杂度推导 critical batch size 的下界。

## critical batch 的严格对象

若 batch size 为 $b$，达到指定停止条件需要 $T(b)$ 步，则

$$
\operatorname{SFO}(b)=bT(b).
$$

critical batch $b^*$ 是使 SFO 复杂度最小的 batch。它依赖模型、数据、训练阶段、目标精度、optimizer 及超参数，不是硬件常数或模型名字的属性。

小 batch 区域常见 $T(b)$ 随 $b$ 近似下降；超过某点后 steps 的下降变慢，而每步样本数仍线性增长，于是 $bT(b)$ 出现 U 形或平台—上升结构。

## 可核查锚点：同一个 batch 可因停止指标不同而改变地位

假设两种 batch 的观测如下：

| $b$ | 达到 train-loss 阈值的 steps | SFO | 达到 gradient-norm 阈值的 steps | SFO |
|---:|---:|---:|---:|---:|
| 256 | 1000 | 256000 | 1600 | 409600 |
| 1024 | 320 | 327680 | 350 | 358400 |

按 train loss，256 更省 SFO；按 gradient norm，1024 更省。理论若使用 stationarity proxy，实验就不能用任意下游 metric 直接宣称验证。阅读时必须对齐 stopping target。

## 论文结论应怎样表述

- 作者推导的公式含 gradient variance、目标精度、effective rank 等不可直接观测量。
- 因此公式主要给出 momentum $\beta$、weight decay $\lambda$ 等对 critical-batch 下界的**定性缩放**，不是绝对 batch 预测器。
- 论文实验覆盖图像分类和 320M 语言模型，验证多项 hyperparameter-dependent ordering。
- Nesterov、decay 对 critical batch 的影响与 LR/停止条件耦合，不能抽成孤立口号。

## rank 为什么出现

Muon 把 matrix gradient/momentum 的谱结构带进 convergence bound。effective rank 影响 spectral/nuclear/Frobenius norm 间的换算，也会影响噪声项。要避免把矩阵宽度直接替换成 rank：训练中 rank proxy、momentum rank 与 gradient rank 都可能不同。

## 与规模实验的关系

- *Practical Efficiency* 观察 Muon 在较大 batch 下的经验 frontier。
- 本文尝试解释 optimizer hyperparameters 怎样移动 critical-batch 下界。
- 两者在实验上相关，但本文的单矩阵、假设化公式不能反推任意 LLM 的最佳 global batch。

## 精读后的任务

固定模型、数据和 token budget，扫描 $b$ 与 $\beta$。同时用 train loss、validation loss、gradient proxy 三种停止条件计算 $T(b)$ 和 $bT(b)$，报告 critical-batch 区间而不是单点，并记录 LR scaling rule。

## 自测

1. 为什么“更大 batch 需要更少 steps”不等于 SFO 更优？
2. 论文公式为何更适合预测相对 ordering 而不是绝对 batch？
3. 若同时随 batch 线性放大 LR，critical-batch 实验改变了哪些变量？

**掌握标准**：能把 batch、steps、SFO、wall-clock 和停止条件分开，且不把理论下界写成通用 recipe。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **四种配置被统一分析**：有/无 Nesterov × 有/无 weight decay；这比只研究“Muon”一个名字更接近真实 recipe。
2. **weight decay 的理论角色不只是 generalization**：论文证明它可在特定 LR–decay 条件下控制 parameter/gradient norms，减少对 bounded-gradient 假设的依赖。
3. **两级验证策略值得学习**：单矩阵 full-Muon toy 严格贴理论；ResNet/VGG/Llama-320M 使用 hybrid Muon，只验证 qualitative transfer，不冒充严格定理验证。
4. **作者明确列出三项限制**：single-matrix 无 layer heterogeneity、理论 full-Muon vs 实践 hybrid、Shampoo/SOAP 的同类 bound 尚未建立。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| SOAP/Practical Efficiency 的 CBS | break point/数据效率 vs $bT(b)$ 的 SFO minimizer | **定义真实不同** | 同一 stopping target 计算各定义 |
| spectral-constraint 论文 | KKT score/decay 约束 vs average gradient norm/CBS | **measure 与假设不同**；原文也警告 rate 不宜横比 | 统一 convergence measure 后才比较 |
| Moonlight 报告更大模型可用更大 batch | 本文只给含未知 variance/rank 的定性下界 | **支持趋势但不预测绝对值** | 测真实 layerwise noise/rank 再代入 |
| Practical Efficiency 说 Muon beyond AdamW critical batch 更 data-efficient | 本文讨论 optimizer/hyperparameter 如何移动 SFO-optimal batch | **相关但非同一命题** | loss target、token budget、LR scaling 对齐 |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| $\operatorname{SFO}(b)=bT(b)$ 与 CBS minimizer | §4.1–4.2 | 论文明确 |
| CBS 公式含 variance、precision、effective rank，不能预测绝对值 | Abstract、§3–4、limitations | 论文反复明确 |
| momentum/decay 影响 CBS 的定性 ordering | Proposition 4.3、Table 1、§5 | 理论下界与作者实验分别提供支持，适用范围以原文假设为准 |
| full-Muon theory 与 hybrid practice 有 gap | §5 two-level validation、Conclusion limitations | 论文明确 |
| 本笔记两种 stopping metric 的数字表 | 无原文表 | 本文反例，说明定义敏感性 |
| “报告 critical-batch 区间而非单点” | 本项目实验建议 | 本文的方法建议，不是论文定理 |
