---
title: "SOAP"
description: "理解在结构化特征基底中运行 Adam 的设计，并辨清 SOAP、Shampoo 与 Muon 的差异。"
topic: "muon"
section: "papers"
slug: "soap"
legacyPaths: ["/notes/soap/"]
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 64
source:
  repository: "J-shang/Muon"
  path: "论文精读/03-SOAP.md"
  url: "https://github.com/J-shang/Muon/blob/ae2b5f9e6ee06b411aef2220e361c75988a7d753/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/03-SOAP.md"
  revision: "ae2b5f9e6ee06b411aef2220e361c75988a7d753"
  syncedAt: "2026-07-14"
  contentHash: "sha256:be8c7af1865d50539044f4cff082d217f23e66e6a329f746dac5a3befb3a3e5a"
  manifest: "muon"
  managed: true
---
> source: [arXiv:2409.11321](https://arxiv.org/abs/2409.11321)
> source class: 实验与理论预印本
> confidence: 理想化等价 `verified under stated assumptions`；效率数字 `supported in-scope`

## 它解决什么问题

Shampoo 的 eigenbasis/矩阵根昂贵，降低更新频率会让 preconditioner 过时。SOAP 的设计是：较慢地更新结构化 eigenbasis，但每步都在该旋转坐标系里更新 Adam 式一、二阶统计。

令左右统计的特征向量为 $Q_L,Q_R$，把梯度旋转为

$$
\widehat G_t=Q_L^\top G_tQ_R.
$$

在 $\widehat G_t$ 上运行逐元素 Adam/Adafactor，再旋转回去：

$$
\Delta_t=Q_L\widehat\Delta_tQ_R^\top.
$$

这使“basis 更新频率”和“逐坐标二阶统计更新频率”解耦。

## 关键理论关系

论文在明确修改下证明：使用 $1/2$ power、layerwise scalar correction、dataset averages 的理想化 Shampoo，等价于在 Shampoo eigenbasis 中运行 Adafactor。实际算法与理想化证明之间至少有三处裂缝：running average 替代 dataset average、basis 不每步更新、Adam 状态要在变化 basis 下解释。

## 可核查锚点：旋转基底为何不是无关操作

逐元素 Adam 对坐标旋转不具有不变性。设二维梯度历史主要沿 $(1,1)$ 方向；在标准 basis 中两个坐标各自积累方差，在旋转 $45^\circ$ 后，能量集中到一个坐标。对角 second moment 因此不同，旋转回原空间的 update 也不同。

可执行检查：构造相同梯度序列，分别在 $I$ 和固定正交矩阵 $Q$ 中运行对角 Adam，再比较最终 update。若不同，就验证了 SOAP 的核心设计空间确实来自“在哪个 basis 里做对角适应”。

## 与 Muon 的第一处分歧

| 轴 | SOAP | Muon |
|---|---|---|
| 变换对象 | 在 Shampoo eigenbasis 中的 gradient/statistics | momentum matrix |
| 持久状态 | 一阶、二阶及结构化 basis/statistics | 通常一个 momentum state |
| 核心操作 | 旋转 + 对角自适应 + 旋回 | 有限步 polar 近似 |
| 主要失败模式 | basis 陈旧、state/分解成本 | NS 误差、shape scale、global matrix 语义 |

两者都尊重矩阵结构，但不是同一算法的不同实现。

## 论文报告与边界

作者在 360M/660M 语言模型和 large-batch 设置中报告：相对 AdamW，iteration 减少超过 40%、wall-clock 超过 35%；相对 Shampoo 约有 20% 改善。应保留为指定模型、batch、实现和调参范围内的作者报告，不能用于证明 Muon 的收益。

## 精读后的任务

实现固定 basis SOAP toy：每 $k$ 步重算 $Q_L,Q_R$，其余 step 只更新旋转空间 Adam。扫描 $k\in\{1,10,100\}$，同时测 loss、basis drift、step time 和 state bytes。

## 自测

1. 为什么“Adam 在旋转空间运行”不等价于原坐标 Adam？
2. SOAP 理想化等价中哪三个假设最容易在生产实现失效？
3. SOAP 和 Muon 都用矩阵结构，为什么不能据此说 Muon 是 SOAP 的无状态版本？

**掌握标准**：能把 basis、逐坐标 state、preconditioner frequency 三者分开解释。

## 二次审计：补漏、分歧与原文核查

### A. 还值得学习的点

1. **critical batch 是论文的重要实证线**：§6.3 报告 SOAP 在其设置中更接近 batch doubling/step halving 的理想线；这解释其 large-batch wall-clock 收益不只来自单步 preconditioner。
2. **one-sided SOAP 是明确的成本—质量变体**：§7.1–7.3 讨论只使用一侧 eigenbasis、Adafactor 替代 Adam和低精度 state；它们改变 state、矩阵分解次数与旋转成本。
3. **basis 更新慢、diagonal state 更新快是核心时间尺度分离**：这比“Shampoo + Adam”名称更能解释算法。
4. **作者自己承认规模边界**：§9 明确实验模型比当时大 LLM 小两个数量级，把跨规模泛化写成 hypothesis。

### B. 与其他论文或学者观点的冲突检查

| 对照观点 | 第一处分歧 | 判断 | 判别检查 |
|---|---|---|---|
| Shampoo 原论文的双侧 matrix power | SOAP 在 Shampoo eigenbasis 内每步运行 diagonal Adam/Adafactor | **generalization/变体，不冲突** | 固定 basis，关闭 diagonal state 更新 |
| Muon/Old Norm 的 accumulation-free polar | SOAP 保留慢变的结构化 basis 和快变二阶 state | **核心 state 不同**；把 SOAP reduce 到 Muon 需删除信息 | state ablation + update cosine |
| SOAP 说“rotated Adam”支持跨规模稳健性 | §9 同时承认该泛化尚未验证 | **作者内部是 hypothesis，不是已证冲突** | 更大模型和不同 domain 的独立复现 |
| critical batch 的定义（SOAP）与 Sato 等 SFO-minimizer 定义 | 前者是线性 scaling 开始失效处；后者最小化 $bT(b)$ | **定义层真实差异** | 同一曲线同时计算 break point 与 SFO minimizer |

### C. 本笔记知识核查表

| 本笔记学习项 | 原文位置 | 核查结论 |
|---|---|---|
| $\widehat G=Q_L^\top GQ_R$，在 eigenbasis 内运行 Adam 再旋回 | §4 Algorithm 3 | `论文明确` |
| idealized $1/2$-power Shampoo 等价于 eigenbasis Adafactor | §4.1 Claim 1 | `论文明确但有三个修改条件` |
| basis 不每步更新时，SOAP 每步仍更新 diagonal second moment | §4.1 实际差异讨论 | `论文明确` |
| 360M/660M、相对 AdamW iteration >40%、wall-clock >35% | Abstract、§6 | `作者报告 in-scope` |
| 本笔记 45° rotation toy | 原文无该例子 | `仓库内反例/教学推导`，用于验证 Adam 非旋转不变性 |
| SOAP 与 Muon 的 object/state/cost 表 | 跨论文整理 | `个人综合`，各单元可由两篇算法逐项核查 |
