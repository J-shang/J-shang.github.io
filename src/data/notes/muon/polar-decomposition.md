---
title: "极分解"
description: "从 SVD 推出 polar factor，并区分最近正交矩阵、QR 与 Muon 近似。"
topic: "muon"
section: "linear-algebra"
slug: "polar-decomposition"
legacyPaths: ["/notes/polar-decomposition/"]
date: 2026-07-01
updated: 2026-07-16
order: 14
readtime: 8
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/线性代数/极分解.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0/%E6%9E%81%E5%88%86%E8%A7%A3.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:f61fd611548ca4cd23ad51aa43ca82e36d596a13bc6694031be148cac66327f3"
  manifest: "muon"
  managed: true
---
## 先记住什么

极分解把矩阵拆成“方向”与“拉伸”，Muon 近似的正是其中的方向因子。

## 核心定义

矩阵 $M$ 的极分解可写作

$$
M=QP,
$$

其中 $Q$ 是正交或半正交的方向因子，$P$ 是对称半正定的拉伸因子。若 $M=U\Sigma V^\top$，则常见的 polar factor 为

$$
Q=UV^\top.
$$

对非方阵或秩亏矩阵，极分解需要更谨慎地处理唯一性，但在 Muon 的核心直觉里，$UV^\top$ 就是把奇异值压到同一尺度后的矩阵方向。

## 相关知识展开

### 1. 极分解像复数的极坐标形式

复数可以写成

$$
z=re^{i\theta},
$$

其中 $e^{i\theta}$ 表示方向，$r$ 表示长度。矩阵极分解也有类似味道：

$$
M=QP.
$$

$Q$ 是方向因子，$P$ 是对称半正定的拉伸因子。对方阵满秩情形，$Q$ 是正交矩阵；对非方阵，$Q$ 是半正交或部分等距对象。

### 2. 从 SVD 推出极分解

若

$$
M=U\Sigma V^\top,
$$

则可以写成

$$
M=(UV^\top)(V\Sigma V^\top).
$$

其中

$$
Q=UV^\top,\qquad P=V\Sigma V^\top.
$$

$P$ 是对称半正定矩阵，承载奇异值尺度；$Q$ 承载方向。Muon 近似的就是 $Q$。

### 3. 极分解和“最近正交矩阵”

对方阵满秩矩阵，polar factor $Q$ 也是离 $M$ 最近的正交矩阵之一：

$$
Q=\arg\min_{R^\top R=I}\|M-R\|_F.
$$

这个性质很有直觉：如果你想把一个一般矩阵变成正交矩阵，同时尽量少改变它的方向，polar factor 是自然选择。Muon 并不是做这个投影问题本身，但这个结果能解释为什么 $UV^\top$ 是“保留方向、去掉尺度”的合理对象。

### 4. 非方阵时发生什么？

线性层权重常是 $m\times n$，不一定方。若 $m>n$，$Q$ 可能满足 $Q^\top Q=I_n$；若 $m<n$，则可能满足 $QQ^\top=I_m$。它不是方阵正交矩阵，而是半正交矩阵。Muon 的 update 因此要结合形状理解，不能机械说“更新矩阵正交”。

### 5. 秩亏时的微妙性

如果 $M$ 有零奇异值，$UV^\top$ 在零奇异子空间上的定义可能不唯一。实际 Newton–Schulz 近似也会受小奇异值、噪声和 $\epsilon$ 影响。训练中这未必是坏事，因为 optimizer update 本来就带有近似和随机性；但做理论推导时要知道满秩假设在哪里被用了。

### 6. 与 QR 分解的区别

QR 分解把矩阵写成 $M=QR$，其中 $Q$ 的列正交，$R$ 是上三角。它常用于解最小二乘或构造正交基。极分解的 $P$ 是对称半正定，和矩阵拉伸直接对应；因此极分解更适合描述“去掉奇异值尺度”的操作。

### 7. Muon 里为什么不用显式极分解？

显式极分解通常依赖 SVD 或矩阵平方根/逆平方根，成本较高。Muon 使用 Newton–Schulz 迭代，是因为它把 polar factor 近似转化成少量矩阵乘，更适合 GPU 和大模型训练。

## 和 Muon 的关系

Muon 的“orthogonalization”更准确地说是近似计算 momentum 矩阵的 polar factor。原始每步 SVD 太贵，因此实现用 Newton–Schulz 迭代近似 $UV^\top$。理解极分解可以避免一个常见误会：Muon 不是对权重做正交正则，而是对更新方向做极分解式归一化。

## 需要掌握到什么程度

- 能从 SVD 推出 $Q=UV^\top$。
- 能解释 $Q$ 保留奇异向量、丢弃奇异值尺度。
- 能理解非方阵时 $Q$ 往往是半正交矩阵。
- 能把“polar factor 精确度”与 Muon 的训练行为联系起来，但不把精确 polar 当作必然最优。

## 常见误区

- 把极分解和 QR 分解混为一谈。QR 是列空间正交化工具；极分解更接近“最近的正交/半正交方向”。
- 认为越精确近似 polar 越好。Muon 文献里存在 inexact update 反而有训练效应的讨论。
- 认为极分解只能用于方阵；非方阵也有相应形式。

## 自测问题

1. 若 $M=U\Sigma V^\top$，polar factor 是什么？
2. Muon 为什么说是对更新矩阵做正交化，而不是对权重矩阵做正交化？
3. 精确 SVD 和 Newton–Schulz 近似在工程成本上有什么差别？

## 参考入口

- [Higham, *Functions of Matrices*](https://epubs.siam.org/doi/book/10.1137/1.9780898717778) 的 polar decomposition 章节 —— 精确分解、最近正交矩阵和数值迭代的规范来源。
- [Keller Jordan, *Muon: An optimizer for hidden layers in neural networks*](https://kellerjordan.github.io/posts/muon/) —— 查看 polar factor 如何成为 optimizer update，而非 weight constraint。
- [Shulgin et al., *Beyond the Ideal: Analyzing the Inexact Muon Update*](https://arxiv.org/abs/2510.19933) —— 研究有限步 NS 偏离精确 polar 后的理论边界。
