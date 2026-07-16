---
title: "Shampoo"
description: "理解矩阵结构预条件的经典前史，并推导 instantaneous Shampoo 与 polar 的条件关系。"
topic: "muon"
section: "papers"
slug: "shampoo"
legacyPaths: ["/notes/shampoo/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 62
source:
  repository: "J-shang/Muon"
  path: "论文精读/01-Shampoo.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/01-Shampoo.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:12b7f9ee75a48233285b4ab495715e1ea247ec86ff41cb683c102f2ab759b30e"
  manifest: "muon"
  managed: true
---
> 原文：[ICML 2018 / PMLR](https://proceedings.mlr.press/v80/gupta18a.html)
> 来源类型：同行评审会议论文
> 阅读提醒：算法与代数可直接复核；训练收益只覆盖论文报告的设置。

## 它解决什么问题

全矩阵 AdaGrad 对向量化参数维护巨大 preconditioner，存储和矩阵函数都不可承受。Shampoo 利用 tensor 各轴结构，给每一维维护较小的统计量。对矩阵梯度 $G_t\in\mathbb{R}^{m\times n}$，简化写法是

$$
L_t=\epsilon I_m+\sum_{s\le t}G_sG_s^\top,
\qquad
R_t=\epsilon I_n+\sum_{s\le t}G_s^\top G_s,
$$

并以双侧矩阵幂预条件当前梯度。二维原始形式常写成

$$
\Delta_t=L_t^{-1/4}G_tR_t^{-1/4}.
$$

关键不在指数名字，而在：统计跨 step 累积、左右两个轴各有状态、矩阵函数成本随维度增长。

## 可核查锚点：何时会退化成 polar

只看单个满秩矩阵 $G=U\Sigma V^\top$，删除历史、damping 与 grafting，令

$$
L=GG^\top=U\Sigma^2U^\top,
\qquad
R=G^\top G=V\Sigma^2V^\top.
$$

于是

$$
L^{-1/4}GR^{-1/4}
=U\Sigma^{-1/2}\Sigma\Sigma^{-1/2}V^\top
=UV^\top.
$$

这条等式解释了 Muon 与 Shampoo 的代数联系，也同时列出了失效条件：

- $L_t,R_t$ 若包含历史梯度，basis 和 spectrum 不再只由当前 $G_t$ 决定；
- damping 会改变小奇异值映射；
- rank deficient 时要用伪逆/正则化并处理 null space；
- momentum、grafting、更新频率和 weight decay 会改变完整动力学。

因此准确关系是：**instantaneous、无 damping 的二维 Shampoo 在适当矩阵幂下可得到 polar direction**；不是“Muon 等于 Shampoo”。

## 状态与成本

对 $m\times n$ 矩阵，左右统计至少占 $m^2+n^2$ 个元素；Muon 的单个 momentum state 占 $mn$。但不能只比较持久状态：Shampoo 还需矩阵根/特征分解，Muon 需 NS 临时 GEMM。真实结论要看 dtype、更新频率、分块和分布式布局。

## 论文报告与边界

- **论文报告**：作者在 stochastic convex setting 给出收敛分析，并在当时的深度模型上报告较快收敛和可比较 step time。
- **今天仍稳固的贡献**：按 tensor axis 构造结构化 preconditioner 的思想。
- **不能直接外推**：2018 年工作负载的 wall-clock 结论不代表现代 LLM/加速器。
- **Muon 相关性**：它提供一个强 baseline 和精确的“删掉了哪些二阶历史”的对照。

## 知识关系

- **后续发展**：SOAP 等结构化 preconditioner 延续了这条路线。
- **特殊情形**：instantaneous 版本在上述条件下得到 polar。
- **不要混同**：实际 Muon 的 persistent state、update object 和数值路径都不同。

## 精读后的任务

实现一个 $8\times4$ 矩阵实验，比较：exact polar、instantaneous Shampoo、含 100 步历史的 Shampoo、有限步 NS。对每个方向报告 singular values、与 polar 的 cosine、状态字节和一次更新的矩阵乘/分解成本。

## 自测

1. 为什么左右各用 $-1/4$ 幂会合成对奇异值的 $-1$ 次缩放？
2. 加入 $\epsilon I$ 后，小奇异值方向会怎样变化？
3. 只凭“都有矩阵乘”能否把 Shampoo 和 Muon 都叫二阶法？需要先固定什么定义？

**掌握标准**：能现场完成上面的 SVD 代数，并说出至少四个使等价失效的实际因素。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **原论文首先是 tensor optimizer**：第 4 节用 matricization 和沿其余轴 contraction，为 order-$k$ tensor 的每个轴维护一个 $n_i\times n_i$ preconditioner；只讲二维矩阵会漏掉 Shampoo 命名与结构优势的核心。
2. **理论对象是 online convex regret**：第 2 节采用 OCO/Online Mirror Descent，第 4 节 Theorem 10 的 bound 含各轴 matricization rank。它不是现代非凸 LLM 收敛定理。
3. **rank 假设进入理论收益**：Theorem 10 通过各轴 rank $r_i$ 控制 regret；“结构感知”不仅是省内存，也进入理论量。
4. **原始算法与现代 Shampoo recipe 有版本边界**：EMA、grafting、blocking、延迟更新矩阵根等生产设计来自后续工作，不能回写成 2018 原论文事实。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| *Old Optimizer, New Norm*：无累积 Shampoo 是 spectral steepest descent | 原论文累积 $L_t,R_t$；后者把 accumulator 关闭 | **特殊情形，不冲突** | 同一 gradient history 下比较 accumulated 与 instantaneous update |
| SOAP：$1/2$ power Shampoo 等价于 eigenbasis Adafactor | SOAP 改了 exponent、scalar correction，并用 dataset averages 建立理想化等价 | **条件化重解释，不是原论文结论** | 按 SOAP Claim 1 的三个修改逐一撤销 |
| *Practical Efficiency*：可在简化假设下把 Shampoo/Soap reduce 到 Muon | 删除历史/二阶状态后的极限 | **unification 会丢掉 Shampoo 的核心 state** | 记录每种 reduction 删除的 state 与 update frequency |
| Muon 是一阶 matrix geometry | 是否估计跨 step 的 second moment/curvature proxy | **机制分类差异**；Shampoo 明确保留结构化 accumulator | state inspection |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| $L_t=\epsilon I+\sum GG^\top$、$R_t=\epsilon I+\sum G^\top G$ 的双侧累积结构 | 矩阵算法和第 4 节 tensor generalization | 论文明确 |
| 二维 update 使用左右 $-1/4$ matrix powers | 原论文矩阵 Shampoo 算法 | 论文明确 |
| general tensor 为每一轴维护 contraction preconditioner | §4.1–4.2、Algorithm 2 | 论文明确 |
| stochastic convex/OCO regret guarantee | §2、Theorem 10 | 论文明确；不可写成非凸 LLM 保证 |
| instantaneous Shampoo 等于 $UV^\top$ | **不在 2018 原论文中作为主张出现**；见 *Old Optimizer, New Norm* Story II 和 Muon 原始说明 | 后续论文 + 本文 SVD 推导，已纠正归因 |
| $m^2+n^2$ state 与 Muon $mn$ 的比较 | 由算法 shape 直接计数 | 本文字节/元素推导，未含 blocking/dtype/temporary buffers |
