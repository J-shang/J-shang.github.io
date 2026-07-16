---
title: "半正交矩阵"
description: "非方阵中的行半正交与列半正交，以及 update 和 weight 的关键区别。"
topic: "muon"
section: "linear-algebra"
slug: "semi-orthogonal-matrix"
legacyPaths: ["/notes/semi-orthogonal-matrix/"]
date: 2026-07-01
updated: 2026-07-16
order: 15
readtime: 8
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/线性代数/半正交矩阵.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0/%E5%8D%8A%E6%AD%A3%E4%BA%A4%E7%9F%A9%E9%98%B5.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:b3af2e7812d12e792ff2085c5e33c4b27c9e4eeeaf64018aaa9588b3e9249569"
  manifest: "muon"
  managed: true
---
## 先记住什么

半正交矩阵是非方阵版的“尽可能正交”：行或列两两正交且范数为 1。

## 核心定义

若 $Q\in\mathbb{R}^{m\times n}$ 且 $m\ge n$，满足

$$
Q^\top Q=I_n,
$$

则 $Q$ 的列半正交；若 $m\le n$ 且

$$
QQ^\top=I_m,
$$

则 $Q$ 的行半正交。非方阵不可能同时满足两边都是单位阵，但可以在较小维度上保持等距。SVD 中的薄 $U$ 或 $V$ 就常是半正交矩阵。

## 相关知识展开

### 1. 方阵正交和半正交的区别

方阵正交矩阵 $Q\in\mathbb{R}^{n\times n}$ 满足

$$
Q^\top Q=QQ^\top=I.
$$

这意味着它既不改变向量长度，也不改变向量之间的夹角。非方阵没有办法同时做到这两点，因为输入维度和输出维度不同。半正交矩阵只能在较小的那个空间上保持等距。

### 2. 列半正交：高矩阵

若 $Q\in\mathbb{R}^{m\times n}$ 且 $m>n$，满足

$$
Q^\top Q=I_n,
$$

则它的列向量两两正交且长度为 1。它把 $n$ 维输入嵌入到 $m$ 维空间，不改变输入向量之间的内积：

$$
\|Qx\|_2^2=x^\top Q^\top Qx=\|x\|_2^2.
$$

### 3. 行半正交：宽矩阵

若 $Q\in\mathbb{R}^{m\times n}$ 且 $m<n$，满足

$$
QQ^\top=I_m,
$$

则它的行向量两两正交且长度为 1。它从高维输入投影到低维输出，能在输出行空间上保持良好尺度，但不可能对所有输入方向保持长度。

### 4. 与奇异值的关系

半正交矩阵的非零奇异值都等于 1。若 rank 为 $r=\min(m,n)$，则

$$
\|Q\|_2=1,\qquad
\|Q\|_F=\sqrt{r}.
$$

这正是 Muon update 的理想形态：所有可用奇异方向等幅，没有某个方向拥有更大的奇异值。

### 5. 一个小例子

矩阵

$$
Q=\begin{bmatrix}
1&0\\
0&1\\
0&0
\end{bmatrix}
$$

是 $3\times2$ 的列半正交矩阵，因为 $Q^\top Q=I_2$。但 $QQ^\top$ 是一个 $3\times3$ 的投影矩阵，不是 $I_3$。它不能覆盖整个三维输出空间，只能把二维输入等距嵌入进去。

### 6. 为什么这对线性层形状重要？

Transformer 中 MLP up projection 可能是 $4d\times d$，down projection 可能是 $d\times4d$。它们的理想 Muon update 都是半正交，但一个更像列半正交，一个更像行半正交。若不考虑形状，直接比较元素 RMS 或 Frobenius norm 容易误判更新尺度。

### 7. 不要把半正交 update 当成半正交 weight

Muon 正交化的是 momentum/update 矩阵，不是把权重 $W$ 替换成半正交矩阵。参数更新后

$$
W_{t+1}=W_t-\eta Q_t,
$$

$W_{t+1}$ 一般不会满足半正交关系。Muon 是改变走路方式，不是把模型绑在正交约束流形上。

## 和 Muon 的关系

神经网络线性层权重大多是非方阵。Muon 对 $M=U\Sigma V^\top$ 取 $UV^\top$ 时，得到的方向通常不是方阵正交矩阵，而是半正交/部分等距矩阵。它让更新在可用奇异方向上等幅，但并不意味着每个元素独立同分布，也不意味着权重自身变成正交矩阵。

## 需要掌握到什么程度

- 能判断一个 $m\times n$ 矩阵应检查 $Q^\top Q$ 还是 $QQ^\top$。
- 能理解非方阵 $UV^\top$ 的形状和半正交性质。
- 能把半正交的理论 RMS 与 Muon update scaling 联系起来。
- 能读懂“row orthogonal”“column orthogonal”“semi-orthogonal”等工程注释。

## 常见误区

- 要求非方阵同时满足 $Q^\top Q=I$ 和 $QQ^\top=I$；除非方阵且满秩，否则不可能。
- 把半正交矩阵的每个元素大小想象成完全相同；正交约束是行/列级别的。
- 认为半正交更新会让模型层保持正交初始化状态；训练权重仍会自由变化。

## 自测问题

1. 一个 $4096\times 11008$ 的 MLP up projection 更新，可能满足哪一侧的半正交关系？
2. 半正交矩阵的 Frobenius 范数与较小维度有什么关系？
3. 为什么 Muon 文献中经常同时讨论矩阵形状和更新 RMS？

## 参考入口

- Golub & Van Loan, *Matrix Computations* —— 非方阵 polar factor、薄 SVD 和正交性条件的数学入口。
- [Liu et al., *Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982) —— 查看 shape-dependent update scale 为什么影响大模型 recipe。
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) —— 对照高矩阵/宽矩阵的 transpose 分支与 scale，防止检查错 $Q^\top Q$ 或 $QQ^\top$。
