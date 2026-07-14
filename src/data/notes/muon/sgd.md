---
title: "SGD"
description: "从随机梯度估计、学习率动力学到 Muon 的一阶优化起点。"
topic: "muon"
section: "optimization"
slug: "sgd"
legacyPaths: ["/notes/sgd/"]
date: 2026-07-01
updated: 2026-07-01
order: 1
readtime: 8
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/优化基础/SGD.md"
  url: "https://github.com/J-shang/Muon/blob/7458bf6dbff95ca416a8ca9069308d5cc6907f96/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E4%BC%98%E5%8C%96%E5%9F%BA%E7%A1%80/SGD.md"
  revision: "7458bf6dbff95ca416a8ca9069308d5cc6907f96"
  syncedAt: "2026-07-14"
  contentHash: "sha256:435707eda1dd42a93fd440ab99f2ceed15fe3d65f1c9d41a1cc5df11dc280998"
  manifest: "muon"
  managed: true
---
> 层次：优化基础

## 一句话定位

SGD（stochastic gradient descent）是深度学习优化器的基准动作：用一个小批量估计全数据梯度，然后沿负梯度方向更新参数。

## 核心定义

给定参数 $\theta_t$、学习率 $\eta_t$ 和 mini-batch 损失梯度 $g_t=\nabla_\theta L_{\mathcal{B}_t}(\theta_t)$，最基本的 SGD 更新为

$$
\theta_{t+1}=\theta_t-\eta_t g_t.
$$

“stochastic” 指 $g_t$ 是抽样估计，不是完整数据集的精确梯度。SGD 的噪声不是纯坏事：它能降低每步成本，也可能帮助模型离开尖锐或不稳定区域。但噪声大小会随 batch size、数据分布和训练阶段改变，因此学习率 schedule 与 batch size 通常要一起理解。

## 相关知识展开

### 1. SGD 到底在近似什么？

如果完整训练集有 $N$ 个样本，经验风险可以写成

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta).
$$

完整梯度是

$$
\nabla L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\nabla\ell_i(\theta).
$$

大模型训练里每一步都扫完整数据集不现实，所以我们抽一个 mini-batch $\mathcal{B}$，用

$$
g_{\mathcal{B}}(\theta)=\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\nabla\ell_i(\theta)
$$

近似完整梯度。若 mini-batch 是均匀抽样，$g_{\mathcal{B}}$ 通常可以看作完整梯度的无偏估计：

$$
\mathbb{E}_{\mathcal{B}}[g_{\mathcal{B}}]=\nabla L.
$$

但“无偏”不等于“每一步都很准”。batch 越小，梯度噪声越大；batch 越大，单步方向越稳定，但单位 token 提供的新信息可能递减。

### 2. 学习率不是装饰项，而是离散动力学的一部分

SGD 的连续直觉是沿负梯度下降，但真实训练是离散更新。学习率 $\eta$ 决定每次跨多远：

$$
\Delta\theta_t=-\eta_t g_t.
$$

如果 $\eta$ 太小，训练像慢慢挪；如果太大，可能跨过谷底、震荡甚至发散。对一个一维二次函数

$$
L(\theta)=\frac{a}{2}\theta^2,
$$

SGD 在无噪声时变成

$$
\theta_{t+1}=(1-\eta a)\theta_t.
$$

要稳定收敛，需要大致满足 $|1-\eta a|<1$，即 $0<\eta<2/a$。这说明曲率越大的方向允许的稳定步长越小。深度网络有很多方向，曲率差异很大，这就是 momentum、AdamW、Muon 等方法存在的土壤。

### 3. 梯度、更新、参数变化要分清

训练日志里常见三个量：

- **gradient**：反向传播得到的 $g_t$；
- **update**：优化器实际应用的 $\Delta\theta_t$；
- **parameter change**：参数从 $\theta_t$ 到 $\theta_{t+1}$ 的变化，可能还包含 weight decay、裁剪、混合精度修正等。

对纯 SGD，三者关系很直接：update 就是 $-\eta g_t$。但一旦加入 momentum、AdamW、Muon，这个关系会变复杂。比如 Muon 的更新方向来自 momentum 矩阵的正交化，不再等于裸梯度方向。

### 4. SGD 噪声为什么有时有益？

mini-batch 噪声会让每一步不完全沿最陡方向走。坏处是收敛曲线更抖，可能需要更多 step；好处是它有时能帮助训练避免陷入某些尖锐或脆弱区域。这个说法不应神秘化：噪声的实际效果取决于模型、数据、batch、学习率和训练阶段。

在 LLM 预训练中，batch 通常很大，硬件利用率也很重要。因此不能简单套用“小 batch 噪声帮助泛化”的图像。比较 Muon、AdamW 和 SGD 时，更合理的做法是同时记录 loss-vs-token、loss-vs-FLOP 和 wall-clock。

### 5. 一个最小实现

最朴素的 SGD 可以写成：

```python
for p in model.parameters():
    if p.grad is None:
        continue
    p.data.add_(p.grad, alpha=-lr)
```

这个片段故意省略了混合精度、分布式同步、梯度裁剪和 weight decay。它的好处是让你看清核心：优化器最终就是把一个 update 加到参数上。学 Muon 时也要保持这个视角：再复杂的正交化和缩放，最后都要变成某个矩阵 update。

## 和 Muon 的关系

Muon 可以从“SGD + momentum + 矩阵正交化”这条线理解。SGD 直接用梯度方向；Muon 通常先维护 momentum，再把二维矩阵参数的动量 $M$ 近似投影成 polar factor $UV^\top$，最后用这个正交化后的方向更新。也就是说，Muon 并不是丢掉一阶梯度，而是在一阶方向之后改变更新几何。

## 需要掌握到什么程度

- 能手写基本 SGD 更新式，并说明 mini-batch 梯度为什么有噪声。
- 能区分 loss gradient、实际 update、learning rate schedule 三件事。
- 能解释 batch size 增大时，梯度噪声、吞吐和泛化之间可能出现的取舍。
- 能把 SGD 当作比较基线，而不是把所有训练差异都归因于“优化器名字不同”。

## 常见误区

- 把 SGD 说成“没有状态所以一定慢”。它的每步计算便宜，但收敛速度取决于曲率、噪声和 schedule。
- 把梯度裁剪、更新裁剪和权重衰减混在一起。SGD 的核心只定义沿梯度下降，其他机制是额外改造。
- 认为小 batch 一定更好或大 batch 一定更好；实际要看临界 batch size 与硬件效率。

## 自测问题

1. 如果学习率翻倍，SGD 的更新方向和更新长度分别发生什么变化？
2. mini-batch 梯度估计无偏，是否意味着每一步都接近完整梯度？
3. 为什么比较 Muon 和 SGD/AdamW 时必须记录 batch size 与学习率 schedule？

## 参考入口

- Bottou, Curtis & Nocedal, *Optimization Methods for Large-Scale Machine Learning*。
- Goodfellow, Bengio & Courville, *Deep Learning* 第 8 章。
- Keller Jordan, *Muon: An optimizer for hidden layers in neural networks*。
