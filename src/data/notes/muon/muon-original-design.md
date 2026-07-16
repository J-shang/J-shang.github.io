---
title: "Muon 原始设计说明"
description: "从作者设计说明和参考实现提取 Muon 的最小算法合同，并划清其证据边界。"
topic: "muon"
section: "papers"
slug: "muon-original-design"
legacyPaths: ["/notes/muon-original-design/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 61
source:
  repository: "J-shang/Muon"
  path: "论文精读/00-Muon原始设计说明.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/00-Muon%E5%8E%9F%E5%A7%8B%E8%AE%BE%E8%AE%A1%E8%AF%B4%E6%98%8E.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:ee745ca27775fce9013c818b60ada20ad73b1623d830cffba7d745021a912589"
  manifest: "muon"
  managed: true
---
> 原文：[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
> 来源类型：作者设计说明与参考实现入口
> 阅读提醒：算法定义可由作者说明和代码核对；效果只按原文实验范围理解。

## 为什么先读它

Muon 没有一篇 2024 年“原始会议论文”可替代这份说明。若跳过它，后续很容易把 Moonlight 的规模化 recipe、框架默认值或 2026 变体反投影成“原始 Muon”。

## 论文式问题陈述

对隐藏层矩阵参数 $W\in\mathbb{R}^{m\times n}$，普通 momentum 给出矩阵 $M_t$。原始设计问：能否保留矩阵左右奇异向量表达的方向结构，同时压平奇异值尺度，让更新更充分地使用多个方向？

典型时序是

$$
M_t=\beta M_{t-1}+G_t,
\qquad
\widetilde M_t=G_t+\beta M_t
$$

（第二式对应常见 Nesterov-style 变体），然后对 $\widetilde M_t$ 做有限步 Newton–Schulz 谱变换。

若薄 SVD 为

$$
\widetilde M_t=U\Sigma V^\top,
$$

理想目标是 polar factor

$$
O_t=UV^\top.
$$

生产实现不做 SVD，而用少量矩阵乘近似它。

## 可核查锚点：变换到底保留和删除什么

令

$$
M=U\operatorname{diag}(9,2,0.1)V^\top.
$$

精确 polar 把非零奇异值映射成 $(1,1,1)$，但 $U,V$ 不变。因此：

- 它不是把 $W$ 正交化；对象是 momentum/update matrix。
- 它不是逐元素 clipping；变换依赖整个矩阵的 singular subspaces。
- 它丢掉了奇异值相对大小，却保留左右奇异向量。

最小数值检查：随机生成 $M$，比较 `svd(M).U @ svd(M).Vh` 与不同 NS steps 的输出，报告 singular values、cosine、orthogonality error 和 RMS。

## 参数路由是算法的一部分

原始说明针对 hidden-layer 2D weights。embedding、输出头、bias 和 norm 参数通常走 AdamW。真实实现不能只写 `p.ndim == 2`：fused QKV、tied embedding/head、MoE expert 和 TP shard 都要求语义判断。

## 证据边界

- **作者报告**：在其 nanoGPT/训练记录设置中有明显优化收益。
- **本文推导**：polar 的谱作用和 shape-dependent RMS 可独立验证。
- **不能推出**：所有 LLM、SFT、RL、卷积或任意参数 reshape 都会受益。
- **版本风险**：NS 系数、step 数、scale、weight decay 和 momentum 定义在不同实现中并不统一。

## 知识关系

- **怎样实现**：用有限步 Newton–Schulz 近似 polar。
- **怎样解释**：在精确 polar、无额外状态等条件下，它是谱范数几何中的 matrix steepest direction。
- **实验证据**：原文观察到预训练 loss 更快下降，但没有证明具体机制是唯一原因。
- **不要混同**：Shampoo 会维护历史 Kronecker preconditioner，实际算法并不等价。

## 精读后的任务

写出一个 optimizer-step 伪代码，明确标记：梯度同步发生点、momentum 更新、Nesterov 组合、NS、shape scale、weight decay、参数更新和非 Muon 参数组。只要其中任何两个步骤交换会改变结果，就不能省略时序。

## 自测

1. 为什么对每个 micro-batch 分别做 polar 再相加，一般不等价于先累积 gradient 再做一次 polar？
2. 对 $m\times n$ 满秩 polar factor，元素 RMS 是多少？它如何随长边变化？
3. 若一个实现对 local TP shard 单独做 NS，它与原始算法的第一处分歧是什么？

**掌握标准**：不用“正交化梯度所以更稳定”这类口号，能画出 object/shape/state/time-order 全链路。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **NS 系数追求的不是高精度 polar**：原文“Proving…”和“Tuning…”两节明确把 tuned quintic 的长期奇异值目标放宽到约 $[0.7,1.3]$，并称这种误差在其训练里未伤害 loss。这比“用 NS 近似 SVD”更重要，因为它说明设计目标本来就是 GPU/训练效用，而非数值分析中的最高精度。
2. **momentum 的位置有历史对照**：原文比较 Orthogonal-SGDM；后者先正交化 gradient 再加 momentum，Muon 则先形成 momentum 再正交化。两者非线性次序不同，不能合并。
3. **Q/K/V 拆分是算法语义**：原文“Empirical considerations”明确报告 fused QKV 分开运行 Muon 更好；这支持本项目把 parameter routing 视作算法合同。
4. **原文自己讨论证据标准**：最后一节强调 competitive task 和 tuned baseline。这是学习 optimizer 论文时值得保留的方法学，而不是 Muon 公式的一部分。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| Shampoo / SOAP 的“结构化二阶预条件” | 是否跨 step 维护 curvature/second-moment statistics | **术语与机制不同，不是真冲突**；原文只把无累积 Shampoo 作为特殊代数联系 | 关闭/开启 Shampoo accumulator，比较 state 与 update |
| *Practical Efficiency* 称 Muon 为最简单的 second-order optimizer；NS 收敛论文称它不是二阶法 | “二阶”的定义：非对角结构变换，还是必须估计/反演曲率 | **真实的分类口径分歧**，但公式不矛盾 | 检查算法是否存储 Hessian/Fisher/gradient-covariance estimate |
| *Muon is Scalable* 认为 weight decay 对长程规模化关键 | 原始实现没有 decay，只报告短程/较小任务 | **后续证据扩展，不是反驳原始定义** | 长 horizon 同 recipe 的 decay ablation |
| 有限步 NS 理论追求逼近 exact polar | 原始 tuned quintic 接受 $[0.7,1.3]$，训练指标未随精度单调改善 | **指标分歧**：polar error 对 training loss | 对齐 RMS/LR 后同时画 polar error 与 loss-vs-time |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| Muon 面向 hidden-layer 2D 参数，其他参数通常交给 AdamW | “Definition”“Empirical considerations” | 论文/原始说明明确 |
| momentum 后运行 BF16、5-step quintic NS | “Definition”代码和“Design of Muon” | 原始说明明确 |
| ideal object 是 $UV^\top$，保留奇异向量并压平奇异值 | “Design”“Proving…” | 原始说明明确；但 tuned finite-step 输出不是精确 $UV^\top$ |
| 本笔记的 $(9,2,0.1)\mapsto(1,1,1)$ 例子 | 无对应例子 | 本文推导，由原文 SVD 公式直接验证，不是作者实验 |
| polar 的 shape-dependent RMS | 原文没有该推导；后见 *Muon is Scalable* Lemma 1 | 跨论文补充，不归因给原始博客 |
| micro-batch 先 polar 与先累积后 polar 不等价 | 原文未直接陈述 | 本文推导，来自 polar 非线性；需数值反例核查 |
