---
title: "解耦 Weight Decay"
description: "为什么权重衰减不等于 L2 正则，以及它为何是现代 Muon 配方的一部分。"
topic: "muon"
section: "optimization"
slug: "decoupled-weight-decay"
legacyPaths: ["/notes/decoupled-weight-decay/"]
date: 2026-07-01
updated: 2026-07-01
order: 5
readtime: 8
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/优化基础/解耦 weight decay.md"
  url: "https://github.com/J-shang/Muon/blob/7458bf6dbff95ca416a8ca9069308d5cc6907f96/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E4%BC%98%E5%8C%96%E5%9F%BA%E7%A1%80/%E8%A7%A3%E8%80%A6%20weight%20decay.md"
  revision: "7458bf6dbff95ca416a8ca9069308d5cc6907f96"
  syncedAt: "2026-07-14"
  contentHash: "sha256:e60f6bd70312d535f3648dbc256962e69307d8248697b56836ab110b20e0f816"
  manifest: "muon"
  managed: true
---
> 层次：优化基础

## 一句话定位

解耦 weight decay 是把“让参数变小”的操作从“沿梯度更新”里拆出来，使正则强度不再被自适应梯度缩放扭曲。

## 核心定义

在普通 SGD 中，把 L2 正则 $\frac{\lambda}{2}\|\theta\|_2^2$ 加进损失，会让梯度多出 $\lambda\theta$，更新为

$$
\theta_{t+1}=\theta_t-\eta(g_t+\lambda\theta_t).
$$

这等价于先按比例衰减参数，再沿梯度下降。但在 Adam 这类自适应优化器中，$g_t+\lambda\theta_t$ 会进入一阶/二阶矩和逐坐标归一化，导致“正则项”被不同坐标的历史方差重新缩放。AdamW 的做法是直接执行

$$
\theta\leftarrow(1-\eta\lambda)\theta
$$

再叠加优化器本身的更新。

## 相关知识展开

### 1. L2 正则和 weight decay 为什么容易混淆？

在普通 SGD 中，如果损失变成

$$
L'(\theta)=L(\theta)+\frac{\lambda}{2}\|\theta\|_2^2,
$$

梯度就是

$$
\nabla L'(\theta)=\nabla L(\theta)+\lambda\theta.
$$

SGD 更新为

$$
\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)-\eta\lambda\theta_t
=(1-\eta\lambda)\theta_t-\eta\nabla L(\theta_t).
$$

所以在纯 SGD 里，“L2 正则进损失”和“每步把参数乘小一点”看起来等价。这种等价性太深入人心，导致很多人把它误带到 Adam 里。

### 2. Adam 里为什么不等价？

Adam 的 update 会除以 $\sqrt{\hat v_t}+\epsilon$。如果把 $\lambda\theta$ 加进梯度，那么正则项也会参与二阶矩估计，并被逐坐标分母缩放。于是参数大的坐标不一定按同样比例衰减；衰减强度和历史梯度尺度纠缠在一起。

解耦 weight decay 的思想是：不要让正则项穿过 Adam 的自适应分母。直接做

$$
\theta\leftarrow(1-\eta\lambda)\theta
$$

就能保持“按比例缩小参数”的含义。

### 3. weight decay 与学习率仍然相关

“解耦”不是说 weight decay 与学习率完全无关。每一步衰减因子仍是

$$
1-\eta_t\lambda.
$$

如果学习率有 warmup 和 cosine decay，实际衰减强度也随时间变。训练日志里只写 `weight_decay=0.1` 不够，还要知道学习率 schedule 和总步数。

### 4. 哪些参数通常不做 weight decay？

大模型 recipe 里常见做法是：

- 对大多数线性层权重使用 weight decay；
- 对 bias 不使用；
- 对 LayerNorm/RMSNorm 的 scale 不使用；
- 对 embedding 和输出头是否使用，需要看具体 recipe。

原因是 bias/norm scale 的语义不是“可自由收缩的线性算子权重”。机械衰减它们可能损害训练稳定性或表达能力。

### 5. weight decay 与“稳定训练”不是同义词

weight decay 能抑制权重范数增长，但不能替代学习率、梯度裁剪、attention logit 控制、数值精度和数据清洗。Muon 文献里 weight decay 很重要，是因为正交化更新可能带来不同的权重范数演化；但如果训练出现 loss spike，不能只靠调大 weight decay 解决。

### 6. 一个配置检查清单

读一个 Muon/AdamW 配方时，至少记录：

- weight decay 的数值；
- 哪些参数组应用 decay；
- decay 是耦合还是解耦；
- decay 和 learning rate schedule 的组合；
- Muon 组与 AdamW 组是否使用相同 decay。

这些细节足以改变实验结论，不能在复现实验里省略。

## 和 Muon 的关系

现代 Muon 配方通常也需要 weight decay。大规模实验里，无 weight decay 的 Muon 可能早期下降很快，但部分权重范数持续长大，长程训练稳定性变差。这里的重点不是“Muon 也要正则化”这么简单，而是 weight decay 的尺度要和 Muon 的更新缩放约定、参数路由一起调。

## 需要掌握到什么程度

- 能区分 L2 正则、耦合 weight decay、解耦 weight decay。
- 能解释为什么 AdamW 名字里的 W 重要。
- 能在实验配置中记录 weight decay 应用于哪些参数组，哪些参数组被排除。
- 能说明 Muon 配方中的 weight decay 是稳定性和泛化的一部分，而不只是防 overfit 的装饰项。

## 常见误区

- 给 bias、LayerNorm/RMSNorm scale、embedding 等参数机械套同一个 weight decay。
- 认为 weight decay 越大越稳定；过大衰减会压制表示学习。
- 忽略学习率和 weight decay 的耦合：解耦衰减仍然含有 $\eta\lambda$。

## 自测问题

1. 为什么 Adam 中把 L2 正则塞进梯度不等价于 AdamW？
2. Muon 训练里如果去掉 weight decay，可能观察到哪些现象？
3. 配置 optimizer group 时，哪些参数常被排除出 weight decay？

## 参考入口

- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*。
- Liu et al., *Muon is Scalable for LLM Training*。
- PyTorch `AdamW` 与 `SGD` 文档中关于 weight decay 的参数说明。
