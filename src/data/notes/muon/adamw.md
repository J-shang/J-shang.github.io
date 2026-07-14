---
title: "AdamW"
description: "理解逐元素二阶矩、参数尺度和 Muon 混合优化器中的职责分工。"
topic: "muon"
section: "optimization"
slug: "adamw"
legacyPaths: ["/notes/adamw/"]
date: 2026-07-01
updated: 2026-07-14
order: 4
readtime: 8
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/优化基础/AdamW.md"
  url: "https://github.com/J-shang/Muon/blob/f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E4%BC%98%E5%8C%96%E5%9F%BA%E7%A1%80/AdamW.md"
  revision: "f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13"
  syncedAt: "2026-07-14"
  contentHash: "sha256:58573182caf40872f7497d01f9288b886e59f648f0cfa3fba552e0ce152d5864"
  manifest: "muon"
  managed: true
---
> 层次：优化基础

## 一句话定位

AdamW 是大模型训练中最常见的强基线：Adam 的逐坐标自适应更新，加上与梯度更新解耦的 weight decay。

## 核心定义

Adam 维护一阶矩和二阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

经过 bias correction 后，Adam 用

$$
\Delta_t=\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

作为逐坐标归一化方向。AdamW 的关键是把 weight decay 写成参数衰减项，而不是塞进梯度：

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\Delta_t.
$$

## 相关知识展开

### 1. Adam 的两个状态分别在做什么？

Adam 维护的一阶矩 $m_t$ 是“方向的平滑版本”，类似 momentum；二阶矩 $v_t$ 是“梯度平方的平滑版本”，用于估计每个坐标近期梯度尺度。

如果某个坐标长期梯度很大，$\sqrt{v_t}$ 会变大，这个坐标的有效步长会变小；如果某个坐标梯度长期很小，分母较小，有效步长会相对变大。Adam 因此被称为逐坐标自适应优化器。

### 2. bias correction 为什么存在？

初始时 $m_0=0,v_0=0$。在训练早期，指数滑动平均会偏向 0。例如如果梯度恒为 $g$，

$$
m_t=(1-\beta_1^t)g.
$$

所以 Adam 使用

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

做 bias correction。没有它，训练前几步的 update 会被系统性低估，尤其当 $\beta_2$ 很接近 1 时更明显。

### 3. AdamW 的“W”到底改了什么？

在 SGD 中，L2 正则和 weight decay 在形式上常常等价；但在 Adam 里，把 $\lambda\theta$ 加进梯度会让正则项也进入 $m_t$ 和 $v_t$，再被逐坐标分母缩放。这会导致不同坐标的衰减强度不一致。

AdamW 直接执行参数衰减：

$$
\theta\leftarrow(1-\eta\lambda)\theta,
$$

然后再做 Adam update。这让“优化方向”和“权重衰减”分工更清楚，也让超参数更容易解释。

### 4. 为什么 AdamW 是 LLM 的强基线？

AdamW 在 Transformer 预训练里好用，原因包括：

- 对稀疏或尺度差异大的梯度坐标有自适应步长；
- 对 embedding、norm、bias 等非矩阵参数不需要特殊几何假设；
- 工程生态成熟，fused kernel、ZeRO/FSDP、混合精度支持充分；
- 超参数经验丰富，很多模型 recipe 都以 AdamW 为默认基线。

因此评估 Muon 时，如果 AdamW 没有充分调参，结论会很脆弱。Muon 的优势需要面对一个强 AdamW，而不是默认配置的 AdamW。

### 5. AdamW 的成本在哪里？

AdamW 的状态开销通常是每个参数两个同形状张量：$m$ 和 $v$。如果状态用 fp32，一个参数本体之外还要 8 bytes 状态；加上梯度、master weights，训练显存压力会很快上来。

Muon 之所以在大模型训练中有吸引力，一部分原因就是它通常不需要逐坐标二阶矩。但代价是：Muon 需要对矩阵参数做正交化，且非矩阵参数仍经常交给 AdamW。

### 6. 最小伪代码

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad.square()
m_hat = m / (1 - beta1 ** step)
v_hat = v / (1 - beta2 ** step)
param *= 1 - lr * weight_decay
param -= lr * m_hat / (v_hat.sqrt() + eps)
```

这个伪代码清楚展示了 AdamW 与 Muon 的核心差别：AdamW 的自适应发生在逐元素分母上；Muon 的主要改造发生在二维矩阵更新的奇异方向上。

## 和 Muon 的关系

Muon 和 AdamW 的差别不是小修小补。AdamW 通过逐坐标二阶矩做自适应缩放；标准 Muon 主要维护 momentum，然后对二维矩阵更新做正交化。实践中 Muon 往往是混合优化器：隐藏层二维线性权重走 Muon，embedding、输出头、bias、norm scale 等非矩阵或语义特殊参数仍走 AdamW。

## 需要掌握到什么程度

- 能写出 Adam 的一阶矩、二阶矩和 bias correction 的目的。
- 能解释 AdamW 为什么比“Adam + L2 正则塞进梯度”更清晰。
- 能理解 optimizer state 内存：AdamW 通常为每个参数保存 $m$ 和 $v$，状态开销大。
- 能把 AdamW 当成需要认真调参的基线，而不是默认被 Muon 击败的稻草人。

## 常见误区

- 把 AdamW 的 `weight_decay` 当作损失函数里的 L2 正则；在 AdamW 中它是解耦参数衰减。
- 只比较训练步数，不比较 tokens、FLOPs、wall-clock 和状态内存。
- 直接把 AdamW 预训练 checkpoint 切到 Muon 微调，并假设不会有 optimizer mismatch。

## 自测问题

1. AdamW 的二阶矩 $v_t$ 解决了什么问题，也带来了什么内存成本？
2. 为什么 Muon 项目里通常还需要 AdamW 参数组？
3. AdamW 与 Muon 对“矩阵结构”的利用有什么根本差别？

## 参考入口

- Kingma & Ba, *Adam: A Method for Stochastic Optimization* —— 从原始算法核对一阶/二阶矩、bias correction 和逐元素分母。
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* —— 核对 AdamW 与 L2 penalty 分叉的更新顺序和实验动机。
- Liu et al., *Muon is Scalable for LLM Training* —— 查看现代 Muon 训练中 AdamW scalar group 的基线与配方边界。
