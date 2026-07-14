---
title: "Newton–Schulz 迭代"
description: "Muon 的计算核心：用少量低精度矩阵乘近似压平动量矩阵的奇异谱。"
topic: "muon"
section: "numerical-computing"
slug: "newton-schulz"
legacyPaths: ["/notes/newton-schulz/"]
date: 2026-07-01
updated: 2026-07-14
order: 20
readtime: 13
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/数值计算/Newton–Schulz 迭代.md"
  url: "https://github.com/J-shang/Muon/blob/f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97/Newton%E2%80%93Schulz%20%E8%BF%AD%E4%BB%A3.md"
  revision: "f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13"
  syncedAt: "2026-07-14"
  contentHash: "sha256:cdb41275bbc19603cd84bec3ee7a393f3f2d46e9915ae30dc2b4b1dc1024c94b"
  manifest: "muon"
  managed: true
---
> 层次：数值计算

## 一句话定位

Newton–Schulz 迭代是 Muon 用少量矩阵乘近似 polar factor 的计算核心，用来避免每一步显式 SVD。

## 核心定义

经典 Newton–Schulz 可用于求矩阵逆平方根或 polar factor。对已适当缩放的矩阵 $X_0$，一种 polar 迭代形式是

$$
X_{k+1}=\frac{1}{2}X_k(3I-X_k^\top X_k).
$$

当奇异值落在合适区间时，$X_k$ 会趋向 $UV^\top$。Muon 实现常使用少量固定步数，并可能采用调过的多项式系数，而不是追求机器精度收敛。这样做的核心收益是：主要成本变成几次 GEMM，可在 GPU 上高效执行。

## 相关知识展开

### 1. 先从目标说起：我们想要什么矩阵？

假设有一个非零矩阵

$$
M=U\Sigma V^\top,
$$

其中 $\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_r)$，$\sigma_i>0$。Muon 理想上想要的不是 $M$ 本身，而是

$$
Q=UV^\top.
$$

这件事可以理解成：保留 $M$ 的左奇异向量和右奇异向量，但把每个非零奇异值都变成 1。也就是说，$M$ 原来可能在某些方向上更新很猛、在另一些方向上更新很弱；$Q$ 则让所有已识别到的奇异方向以同等强度参与更新。

如果直接做 SVD，数学上最清楚：

1. 分解 $M=U\Sigma V^\top$；
2. 扔掉 $\Sigma$；
3. 返回 $UV^\top$。

问题是，训练中每一步、每个大线性层都做 SVD 太贵。Newton–Schulz 的价值就在这里：它不显式求 $U,\Sigma,V$，而是用几次矩阵乘把 $M$ 推向一个近似的 $UV^\top$。

### 2. 为什么一个多项式迭代能“抹平奇异值”？

理解 Newton–Schulz 最好的入口，是把矩阵迭代暂时想成“对每个奇异值做同一个标量函数”。

对迭代

$$
X_{k+1}=\frac{1}{2}X_k(3I-X_k^\top X_k),
$$

如果 $X_k=U\operatorname{diag}(s_i)V^\top$，那么在理想精确算术下，左右奇异向量保持不变，每个奇异值大致按下面的标量映射更新：

$$
s_{k+1}=f(s_k)=\frac{1}{2}s_k(3-s_k^2).
$$

这个函数在 $s=1$ 处有不动点，因为

$$
f(1)=1.
$$

当 $s$ 在合适范围内，例如 $0<s<\sqrt{3}$，迭代会把 $s$ 往 1 推。你可以用几个数感受一下：

| $s$ | $f(s)=\frac{1}{2}s(3-s^2)$ | 直觉 |
|---:|---:|---|
| 0.2 | 0.296 | 太小的奇异值被放大 |
| 0.5 | 0.6875 | 小奇异值继续变大 |
| 1.0 | 1.0 | 目标点保持不动 |
| 1.2 | 0.936 | 大于 1 的奇异值被拉回 |

所以 NS 迭代并不是“把矩阵元素归一化”，而是在奇异值层面做非线性整形：奇异向量尽量保留，奇异值被推向 1。

### 3. 为什么迭代前必须缩放？

上面的收敛直觉依赖一个前提：初始奇异值要落在合适区间。如果某个奇异值太大，标量映射可能直接跑飞。比如 $s=2$ 时，

$$
f(2)=\frac{1}{2}\cdot 2\cdot(3-4)=-1.
$$

这已经不是“往 1 收敛”的温和行为了。实际实现通常会先做类似

$$
X_0=\frac{M}{\|M\|+\epsilon}
$$

的缩放，让最大奇异值不要太大。这里的 $\|M\|$ 可以是 Frobenius 范数、谱范数估计或实现中更便宜的近似尺度。缩放的目的不是改变最终 polar factor 的方向，因为

$$
\operatorname{polar}(cM)=\operatorname{polar}(M),\quad c>0,
$$

而是把数值放进迭代舒服的区间里。

这一点很像开车进弯前先减速：目标方向没变，但如果速度太快，后面的控制律就不再稳定。

### 4. 为什么常见实现只做少量步？

数值分析教材里会关心“迭代到很高精度”。Muon 里关心的问题不完全一样。训练时我们真正需要的是一个足够好的更新方向，而不是把 polar factor 算到 $10^{-12}$ 精度。

少量 NS 步数的取舍大致是：

- **步数少**：optimizer step 更快，近似更粗，可能保留一些原始奇异值差异；
- **步数多**：更接近 $UV^\top$，但多做矩阵乘，墙钟变慢，也可能改变训练动力学；
- **调系数多项式**：有些 Muon 实现不用上面最经典的 $\frac{1}{2}x(3-x^2)$，而用经验调过的奇异值整形多项式，使少量步数下的训练效果更好。

所以 `ns_steps=5` 不应理解为“数学上只需要 5 步就收敛”，而应理解为“这个训练配方选择用 5 次左右的矩阵多项式变换来得到可用更新”。

### 5. 代码里应该怎样识别 NS？

在 PyTorch/JAX 代码里，NS 迭代常被写成几次矩阵乘。一个简化版本长这样：

```python
X = G / (G.norm() + 1e-7)
if X.shape[0] > X.shape[1]:
    X = X.T
for _ in range(ns_steps):
    A = X @ X.T
    X = a * X + b * A @ X + c * A @ A @ X
if transposed:
    X = X.T
```

这里有几个读代码时要盯住的点：

- `G` 通常不是裸梯度，而是 momentum 或经过 momentum/Nesterov 处理后的矩阵；
- `G.norm()` 之类的缩放是为了让 NS 的初始奇异值进入安全区间；
- `X @ X.T` 或 `X.T @ X` 的选择和矩阵长宽有关，通常会选择较小的一侧来减少计算；
- `a, b, c` 可能不是经典 NS 系数，而是 Muon 配方中为了训练表现调过的系数；
- 迭代结束的 `X` 只是更新方向，后面通常还会乘学习率、weight decay 和 update scaling。

如果你看到实现里先把高矩阵转置成宽矩阵，再迭代，通常不是算法含义变了，而是在减少中间矩阵的尺寸。例如对 $m\times n$ 矩阵，若 $m>n$，计算 $X^\top X$ 得到的是 $n\times n$；若 $n$ 更小，这会比构造 $m\times m$ 中间矩阵便宜得多。

### 6. 一个最小手算例子

设

$$
M=\begin{bmatrix}2&0\\0&0.5\end{bmatrix}.
$$

它的奇异向量就是标准基，奇异值是 $2$ 和 $0.5$。理想 polar factor 是

$$
Q=\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$$

但如果不缩放，$s=2$ 会让经典 NS 映射变成 $-1$，方向可能翻转。先用最大尺度粗略缩放：

$$
X_0=\frac{1}{2}M=\begin{bmatrix}1&0\\0&0.25\end{bmatrix}.
$$

第一步后两个奇异值分别变为：

$$
f(1)=1,\qquad f(0.25)=\frac{1}{2}\cdot0.25\cdot(3-0.0625)\approx0.367.
$$

第二个方向从 $0.25$ 被推到 $0.367$，继续迭代会逐步接近 1。这个例子展示了 NS 的本质：不是直接“除以范数”一次了事，而是用多次矩阵多项式把不同奇异方向拉到统一尺度。

### 7. 数值稳定性要看哪些东西？

NS 在深度学习里好用，是因为它主要由矩阵乘组成，GPU 友好；但它也有数值脾气。排查实现时重点看：

- **输入是否全零或接近全零**：需要 $\epsilon$ 避免除零；接近零的更新是否应该跳过，也要按实现判断。
- **缩放是否足够**：最大奇异值太大时，迭代可能发散或产生 NaN。
- **dtype 是否合适**：bf16/fp16 下重复矩阵乘会积累舍入误差；有些实现会在 NS 内部使用更高精度累加。
- **矩阵形状是否极端**：很扁或很高的矩阵会影响中间矩阵大小、RMS 缩放和 kernel 效率。
- **分布式分片是否改变语义**：对 shard 做 NS 不一定等于对完整权重更新做 NS。

学 Muon 时，看到 NS 不要只问“公式是什么”，还要问“输入矩阵是谁、在哪个 dtype 上算、算几步、按什么形状转置、最后怎样缩放”。这些问题共同决定了训练里的 Muon，而不只是白板上的 Muon。

## 和 Muon 的关系

理想 Muon 方向需要 $M=U\Sigma V^\top$ 的 $UV^\top$。显式 SVD 对每个参数矩阵每步都太贵，也不适合大规模分布式训练。Newton–Schulz 让 Muon 以“近似正交化”的方式落地。需要注意，NS 误差不是纯实现误差；文献已经开始讨论 inexact Muon update 本身可能改变训练动力学。

## 需要掌握到什么程度

- 知道 NS 迭代通过矩阵乘和多项式把奇异值推向 1。
- 能解释为什么迭代前通常要缩放或归一化矩阵。
- 能理解 `ns_steps` 是准确度、速度和稳定性的折中。
- 能读懂代码中 `X @ X.T @ X` 这类表达式是在近似 polar factor。

## 常见误区

- 认为 NS 步数越多一定越好。更精确的 polar 方向不必然带来更好的训练或更快墙钟。
- 忽略低精度误差。重复矩阵乘会放大舍入误差，尤其在 fp16/fp8 路径上。
- 把 NS 当成普通归一化；它是矩阵奇异值层面的非线性变换。

## 自测问题

1. 为什么 Muon 不直接对每个权重矩阵做 SVD？
2. NS 迭代中矩阵缩放的目的是什么？
3. `ns_steps` 从 5 降到 3，可能影响哪些指标？

## 参考入口

- KellerJordan/Muon 中的 Newton–Schulz 函数 —— 最小实现入口；逐行核对 transpose、归一化、系数、steps 和 dtype。
- Higham, *Functions of Matrices* —— 矩阵函数、polar decomposition 与迭代收敛的规范数学背景。
- Shulgin et al., *Beyond the Ideal: Analyzing the Inexact Muon Update* —— 研究有限步近似如何与学习率/momentum 耦合；属于前沿理论预印本。
