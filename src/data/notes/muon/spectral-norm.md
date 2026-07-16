---
title: "谱范数"
description: "从最坏方向的长度放大，到谱范数几何下的最陡下降。"
topic: "muon"
section: "linear-algebra"
slug: "spectral-norm"
legacyPaths: ["/notes/spectral-norm/"]
date: 2026-07-01
updated: 2026-07-16
order: 11
readtime: 7
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/线性代数/谱范数.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0/%E8%B0%B1%E8%8C%83%E6%95%B0.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:2d18e613aa0e9c5a49d629130c97d2797c55e0f128dabd192014a777b8c77e86"
  manifest: "muon"
  managed: true
---
## 先记住什么

谱范数衡量矩阵对向量最大能拉伸多少，是 Muon “最陡下降”解释里的关键范数。

## 核心定义

矩阵 $A$ 的谱范数定义为

$$
\|A\|_2=\max_{\|x\|_2=1}\|Ax\|_2.
$$

它等于最大奇异值 $\sigma_{\max}(A)$。如果把矩阵看作线性算子，谱范数就是最坏输入方向上的放大倍数。与逐元素范数不同，谱范数关心整体算子行为，而不只是每个元素大小。

## 相关知识展开

### 1. 谱范数是“最坏方向”的长度放大

定义

$$
\|A\|_2=\max_{\|x\|_2=1}\|Ax\|_2
$$

的意思是：把所有长度为 1 的输入方向都试一遍，找出被 $A$ 拉得最长的那个。这个最大拉伸倍数就是谱范数。

如果 $A=U\Sigma V^\top$，输入取 $V$ 中对应最大奇异值的右奇异向量时，输出长度达到 $\sigma_{\max}$。所以谱范数等于最大奇异值。

### 2. 与向量二范数的关系

谱范数是由向量二范数诱导出来的矩阵范数，因此满足

$$
\|Ax\|_2\le \|A\|_2\|x\|_2.
$$

这条不等式很重要：如果你知道某层权重的谱范数，就能上界它对输入扰动的最大放大。虽然 Muon 不是直接控制权重谱范数，但“矩阵作为算子”的语言来自这里。

### 3. 一个对角矩阵例子

设

$$
A=\begin{bmatrix}5&0\\0&0.2\end{bmatrix}.
$$

第一坐标方向被放大 5 倍，第二坐标方向被缩小到 0.2 倍。因此

$$
\|A\|_2=5.
$$

Frobenius 范数则是 $\|A\|_F=\sqrt{5^2+0.2^2}$。可以看到，谱范数只关心最大方向；Frobenius 范数会累计所有方向。

### 4. 谱范数约束下的最陡下降

普通梯度下降常被解释为：在欧氏范数约束的 update ball 中，选择让线性化损失下降最快的方向。若把约束从 Frobenius/欧氏几何换成谱范数几何，最优方向会变成与梯度矩阵奇异向量相关的对象。

这就是 Muon 理论解释的一条路：对矩阵参数，不一定要按元素欧氏几何理解更新；可以把权重看成线性算子，在谱范数约束下选方向。

### 5. 谱范数和 Lipschitz 稳定性

在深度网络里，层的谱范数常用于讨论 Lipschitz 常数、对抗鲁棒性或梯度爆炸。但这不意味着“谱范数越小越好”。过强约束可能限制表达能力；而且整体模型稳定性还取决于激活、归一化、残差路径和 attention。

Muon 用谱范数语言解释 update，不等同于给每个权重矩阵做 spectral normalization。

### 6. 快速估计：power iteration

实际中要估计谱范数，常用 power iteration：

$$
v\leftarrow \frac{A^\top u}{\|A^\top u\|},\qquad
u\leftarrow \frac{Av}{\|Av\|}.
$$

迭代后 $u^\top Av$ 近似最大奇异值。Muon 主实现通常不靠 power iteration 做每步更新，但理解它有助于区分“估计最大奇异值”和“近似整个 polar factor”这两件事。

## 和 Muon 的关系

Muon 的一个理论视角是：在谱范数约束下选择对损失下降最快的矩阵更新。这个视角会导向对动量矩阵 $M$ 的 polar factor $UV^\top$。直观地说，普通梯度可能被少数大奇异方向主导；Muon 让更新在谱范数几何下更均衡地利用多个方向。

## 需要掌握到什么程度

- 能说明谱范数等于最大奇异值。
- 能区分谱范数、Frobenius 范数、核范数分别在看什么。
- 能理解“约束更新矩阵的谱范数”和“把权重矩阵正交化”不是同一件事。
- 能读懂 Muon 文献中关于 spectral norm steepest descent 或 spectral norm constraints 的表述。

## 常见误区

- 把谱范数误认为元素最大绝对值；那是 $\ell_\infty$ 风格的元素范数，不是算子范数。
- 认为谱范数小就代表训练一定稳定。训练稳定还受 learning rate、归一化层、attention logits、数据和精度影响。
- 把理论约束视角直接等同于工程实现；实现还要处理 NS 误差、缩放和通信。

## 自测问题

1. 为什么 $\|A\|_2$ 可以解释为最大拉伸倍数？
2. 若一个矩阵只有一个很大的奇异值，谱范数和 Frobenius 范数会如何反映它？
3. Muon 的谱范数视角解释的是更新矩阵还是权重矩阵？

## 参考入口

- Trefethen & Bau, *Numerical Linear Algebra* —— 最大奇异值、operator norm 和 power iteration 的规范入口。
- [Chen, Li & Liu, *Muon Optimizes Under Spectral Norm Constraints*](https://arxiv.org/abs/2506.15054) —— 查看 decoupled WD 下的谱范数约束分析及其假设。
- [Keller Jordan, *Muon: An optimizer for hidden layers in neural networks*](https://kellerjordan.github.io/posts/muon/) —— 把谱范数 steepest direction 与最小 Muon 实现对应起来。
