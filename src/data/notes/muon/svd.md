---
title: "奇异值分解（SVD）"
description: "把矩阵看成线性变换，理解 Muon 为什么保留奇异向量并压平奇异值。"
topic: "muon"
section: "linear-algebra"
slug: "svd"
legacyPaths: ["/notes/svd/"]
date: 2026-07-01
updated: 2026-07-14
order: 10
readtime: 7
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/线性代数/SVD.md"
  url: "https://github.com/J-shang/Muon/blob/f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0/SVD.md"
  revision: "f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13"
  syncedAt: "2026-07-14"
  contentHash: "sha256:5949faf4286ec1f1737f75a566898901c17293aff6208e1575883055d4a9a660"
  manifest: "muon"
  managed: true
---
> 层次：线性代数

## 一句话定位

SVD 把任意矩阵拆成“输入方向、每个方向的拉伸强度、输出方向”，是理解 Muon 正交化的入口。

## 核心定义

对矩阵 $M\in\mathbb{R}^{m\times n}$，奇异值分解写作

$$
M=U\Sigma V^\top,
$$

其中 $U$ 和 $V$ 的列是左/右奇异向量，$\Sigma$ 的非负对角元是奇异值。几何上，$V^\top$ 先旋转输入坐标，$\Sigma$ 沿不同轴缩放，$U$ 再旋转到输出坐标。SVD 适用于非方阵，因此比特征分解更适合神经网络里的线性层权重。

## 相关知识展开

### 1. 把矩阵看成线性变换

矩阵不只是二维数字表。在线性层

$$
y=Wx
$$

里，$W$ 把输入向量 $x$ 映射成输出向量 $y$。SVD 告诉我们，这个映射可以分成三步：

$$
x \xrightarrow{V^\top} \text{换到右奇异向量坐标}
\xrightarrow{\Sigma} \text{按每个奇异值缩放}
\xrightarrow{U} \text{换到输出坐标}.
$$

所以奇异值不是抽象数字，而是矩阵在不同方向上的拉伸强度。

### 2. 薄 SVD 和完整 SVD

若 $M\in\mathbb{R}^{m\times n}$，令 $r=\operatorname{rank}(M)$。薄 SVD 常写成

$$
M=U_r\Sigma_rV_r^\top,
$$

其中 $U_r\in\mathbb{R}^{m\times r}$，$\Sigma_r\in\mathbb{R}^{r\times r}$，$V_r\in\mathbb{R}^{n\times r}$。完整 SVD 会把 $U,V$ 补成完整正交基。Muon 里通常关心的是非零奇异方向，所以薄 SVD 的心智模型更直接。

### 3. 一个 2×2 例子

设

$$
M=\begin{bmatrix}3&0\\0&1\end{bmatrix}.
$$

它已经处在奇异向量坐标中，奇异值是 3 和 1。这个矩阵把第一个坐标方向拉伸 3 倍，第二个方向保持 1 倍。它的 polar factor 是

$$
UV^\top=I.
$$

这说明 polar factor 保留“方向轴”，但去掉“哪个方向拉伸更强”。Muon 正是在更新矩阵上做类似操作。

### 4. 奇异值谱透露什么？

如果一个矩阵的奇异值高度集中，比如 $\sigma_1$ 很大、其他很小，那么这个矩阵的作用主要由少数方向主导。若奇异值更平，说明多个方向都在参与。

训练分析里有时会看 SVD entropy 或 effective rank，试图衡量权重或更新是否由少数方向控制。读这类结果时要问清楚：分析的是权重矩阵、梯度矩阵，还是 optimizer update？Muon 直接处理的是 update/momentum 矩阵。

### 5. SVD 与特征分解有什么区别？

特征分解写作 $A=P\Lambda P^{-1}$，主要适用于方阵，并且对非对称矩阵可能不稳定或没有正交特征向量。SVD 对任意矩阵都存在，奇异值非负，左右奇异向量分别描述输入和输出空间。

Transformer 里的权重常是 $d_\text{out}\times d_\text{in}$ 的非方阵。用 SVD 语言，比强行套特征值语言自然得多。

### 6. 学 Muon 时 SVD 要抓住哪一句话？

最重要的一句是：

$$
M=U\Sigma V^\top \quad\Rightarrow\quad \text{Muon 理想方向 } UV^\top.
$$

这不是为了真的每步做 SVD，而是为了理解 Muon 的目标：保留矩阵更新的奇异向量结构，抹平奇异值尺度。

## 和 Muon 的关系

理想 Muon 方向可写成 $UV^\top$：保留 $M$ 的奇异向量，抹平非零奇异值。换句话说，Muon 不是让权重矩阵本身正交，而是让更新矩阵在奇异方向上“等幅”。真实实现通常不每步做 SVD，而用 Newton–Schulz 迭代近似这个 polar factor。

## 需要掌握到什么程度

- 能从 $M=U\Sigma V^\top$ 说出每个因子的形状和意义。
- 能理解 $UV^\top$ 与 $M$ 方向相关，但去掉了奇异值尺度。
- 能解释为什么二维线性权重天然适合用 SVD 语言，而一维 bias 不适合。
- 能看懂“rank”“singular spectrum”“SVD entropy”等训练诊断词。

## 常见误区

- 把 SVD 只当作降维工具；在 Muon 中它首先是矩阵更新几何的语言。
- 认为 $UV^\top$ 总是方阵正交矩阵。非方阵时它是半正交/部分等距结构。
- 认为 Muon 每步显式做 SVD。工程实现通常用少量矩阵乘近似。

## 自测问题

1. 若 $M$ 是 $m\times n$，薄 SVD 中 $U,\Sigma,V$ 的形状分别是什么？
2. 从 $M=U\Sigma V^\top$ 到 $UV^\top$，丢掉了什么信息？
3. 为什么 embedding table 的二维形状不自动意味着应按普通隐藏层 Muon 处理？

## 参考入口

- Golub & Van Loan, *Matrix Computations* —— SVD、polar 和稳定矩阵算法的规范参考。
- Trefethen & Bau, *Numerical Linear Algebra* —— 用几何图像、低秩近似和条件数理解奇异值。
- Keller Jordan, *Muon: An optimizer for hidden layers in neural networks* —— 直接观察 Muon 如何从 $M=U\Sigma V^\top$ 保留 $UV^\top$。
