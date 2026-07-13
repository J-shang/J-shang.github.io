---
title: "核范数"
description: "理解谱范数的对偶与 Muon 极分解方向之间的理论桥梁。"
category: "线性代数"
date: 2026-07-01
updated: 2026-07-01
order: 13
readtime: 8
source: "https://github.com/J-shang/Muon/blob/main/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0/%E6%A0%B8%E8%8C%83%E6%95%B0.md"
---
> 层次：线性代数

## 一句话定位

核范数是矩阵所有奇异值之和，是秩的凸替代，也是谱范数的对偶范数。

## 核心定义

对矩阵 $A$，核范数定义为

$$
\|A\|_*=\sum_i\sigma_i(A).
$$

它把每个奇异方向的强度线性相加。在线性代数和优化中，核范数常用于鼓励低秩结构，因为直接优化秩是非凸且离散的。更关键的是，核范数与谱范数互为对偶：

$$
\|A\|_*=\max_{\|B\|_2\le 1}\langle A,B\rangle.
$$

## 相关知识展开

### 1. 核范数是“奇异值的一范数”

向量有一范数、二范数、无穷范数。矩阵也可以先看奇异值向量，再对这个向量取范数：

- 谱范数：奇异值的无穷范数 $\max_i\sigma_i$；
- Frobenius 范数：奇异值的二范数 $\sqrt{\sum_i\sigma_i^2}$；
- 核范数：奇异值的一范数 $\sum_i\sigma_i$。

所以核范数关心所有奇异方向强度的线性总和。

### 2. 为什么核范数常和低秩联系在一起？

矩阵的 rank 是非零奇异值的个数：

$$
\operatorname{rank}(A)=|\{i:\sigma_i>0\}|.
$$

直接最小化 rank 是困难的非凸问题。核范数作为奇异值之和，是 rank 的一个常用凸替代。它会鼓励一些奇异值变小甚至为 0，因此常用于矩阵补全和低秩恢复。

Muon 中不是在最小化权重核范数，但理解“奇异值向量”这个视角，有助于看懂极分解和范数对偶。

### 3. 谱范数的对偶为什么是核范数？

矩阵内积定义为

$$
\langle A,B\rangle=\operatorname{tr}(A^\top B).
$$

如果要求 $\|B\|_2\le1$，那么 $B$ 的每个奇异方向最大不能超过 1。要让 $\langle A,B\rangle$ 最大，最好的选择就是让 $B$ 和 $A$ 的奇异向量对齐，并把允许的奇异值都用满。于是最大值变成

$$
\sum_i\sigma_i(A)=\|A\|_*.
$$

这个结论是 Muon 谱范数最陡下降解释背后的数学拼图。

### 4. 一个对角例子

设

$$
A=\begin{bmatrix}3&0\\0&1\end{bmatrix}.
$$

它的奇异值是 3 和 1，所以

$$
\|A\|_2=3,\qquad
\|A\|_F=\sqrt{10},\qquad
\|A\|_*=4.
$$

三个范数没有谁“更正确”，只是看问题的角度不同。

### 5. 核范数与 Muon 公式的连接

Muon 理想方向 $UV^\top$ 可以从一个优化问题里出现：

$$
\max_{\|B\|_2\le1}\langle M,B\rangle.
$$

当 $M=U\Sigma V^\top$，最优 $B$ 的一个选择就是 $UV^\top$。这说明 Muon 的方向可以理解为：在谱范数允许的 update 集合里，选和当前 momentum 最对齐的方向。

### 6. 不要把理论桥梁误读成训练目标

核范数在这里帮助解释为什么 $UV^\top$ 合理，但 Muon 实现不会每步计算 $\|M\|_*$，也不会显式给权重加核范数惩罚。它是理论语言，不是工程步骤。

## 和 Muon 的关系

Muon 的谱范数最陡下降解释会自然碰到核范数对偶。若要在谱范数约束下最大化与动量矩阵 $M$ 的内积，最优方向与 $M$ 的奇异向量对齐，形式上导向 $UV^\top$。所以核范数不是 Muon 实现中直接计算的量，而是理解“为什么 polar factor 是某种最优更新方向”的理论桥梁。

## 需要掌握到什么程度

- 知道核范数等于奇异值之和，不是元素绝对值之和。
- 能说出核范数和谱范数的对偶关系大意。
- 能理解它在 Muon 中更多是理论解释工具，而不是每步工程操作。
- 能把核范数和低秩正则的常见用途区分开。

## 常见误区

- 看到“nuclear”就联想到物理 μ 子；这里完全是矩阵范数。
- 认为 Muon 在最小化核范数。Muon 主要是在构造更新方向，不是在给权重加核范数正则。
- 把核范数与 Frobenius 范数混淆：一个是奇异值一范数，一个是奇异值二范数。

## 自测问题

1. 核范数、谱范数、Frobenius 范数分别对应奇异值的什么聚合？
2. “谱范数约束下的最优线性目标”为什么会和 $UV^\top$ 有关？
3. Muon 训练中是否需要每步显式计算核范数？

## 参考入口

- Recht, Fazel & Parrilo, *Guaranteed Minimum-Rank Solutions of Linear Matrix Equations via Nuclear Norm Minimization*。
- Trefethen & Bau, *Numerical Linear Algebra*。
- Chen, Li & Liu, *Muon Optimizes Under Spectral Norm Constraints*。
