---
title: "Momentum"
description: "把动量理解成方向历史的低通滤波，并看清不同实现约定的尺度差异。"
topic: "muon"
section: "optimization"
slug: "momentum"
legacyPaths: ["/notes/momentum/"]
date: 2026-07-01
updated: 2026-07-01
order: 2
readtime: 7
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/优化基础/momentum.md"
  url: "https://github.com/J-shang/Muon/blob/7458bf6dbff95ca416a8ca9069308d5cc6907f96/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E4%BC%98%E5%8C%96%E5%9F%BA%E7%A1%80/momentum.md"
  revision: "7458bf6dbff95ca416a8ca9069308d5cc6907f96"
  syncedAt: "2026-07-14"
  contentHash: "sha256:ecb377067140e8377d83722628471513b3194ce49800b4b6c046e2f3a566dfd8"
  manifest: "muon"
  managed: true
---
> 层次：优化基础

## 一句话定位

momentum 是给 SGD 加“惯性”：不只看当前梯度，还累计过去方向，让更新在稳定方向上加速、在震荡方向上互相抵消。

## 核心定义

常见 heavy-ball momentum 写作

$$
v_t=\mu v_{t-1}+g_t,\qquad
\theta_{t+1}=\theta_t-\eta v_t,
$$

其中 $\mu$ 是动量系数。也有实现使用指数滑动平均形式

$$
m_t=\beta m_{t-1}+(1-\beta)g_t.
$$

两种写法只是尺度约定不同；关键是当前更新方向包含历史梯度。momentum 对狭长峡谷形损失特别有用：沿主下降方向持续积累，沿高曲率震荡方向被平均掉。

## 相关知识展开

### 1. momentum 是对“方向历史”的低通滤波

普通 SGD 每一步只看当前 mini-batch 梯度，所以方向会受采样噪声影响。momentum buffer 可以展开成历史梯度的加权和：

$$
v_t=g_t+\mu g_{t-1}+\mu^2 g_{t-2}+\cdots.
$$

越久以前的梯度权重越小，衰减速度由 $\mu$ 控制。若 $\mu=0.9$，十步前的梯度还保留约 $0.9^{10}\approx0.35$ 的相对权重。这个“记忆长度”解释了 momentum 为什么能让一致方向变强，也解释了它为什么可能在方向突然变化时反应慢。

### 2. 为什么它能加速峡谷里的优化？

考虑一个二维损失地形：横向曲率很大，纵向曲率较小。普通 SGD 可能在横向左右震荡，同时沿纵向慢慢下降。momentum 会把来回变号的横向梯度相互抵消，而把连续同向的纵向梯度积累起来。

用口语说：如果很多步都在说“往南走”，momentum 会越走越坚决；如果一步说“往东”、下一步说“往西”，momentum 会判断这只是震荡，不会被每个 mini-batch 牵着鼻子走。

### 3. 两种常见写法的尺度差异

有些代码写

$$
v_t=\mu v_{t-1}+g_t,
$$

有些代码写

$$
m_t=\beta m_{t-1}+(1-\beta)g_t.
$$

第二种是指数滑动平均，长期稳定梯度 $g$ 下会收敛到 $g$；第一种在稳定梯度下会收敛到 $g/(1-\mu)$。所以读实现时不能只看“用了 0.9 momentum”，还要看 buffer 是否乘了 $(1-\beta)$、学习率是否配套调整。

### 4. momentum 与曲率、学习率的相互作用

momentum 会增加有效步长。对连续同向梯度，heavy-ball 形式的稳态 buffer 约为

$$
v\approx \frac{g}{1-\mu}.
$$

当 $\mu=0.9$，这相当于把方向累积放大约 10 倍。因此 momentum 往往需要和学习率一起调。学习率原本安全，打开 momentum 后可能开始 overshoot；反过来，恰当的 momentum 又能让较小学习率获得更快进展。

### 5. optimizer state 里 momentum 长什么样？

在工程实现里，momentum buffer 通常和参数同形状。一个 $4096\times4096$ 的线性层权重，如果用 fp32 momentum，就需要额外约 $4096^2\times4\approx67$ MB 的状态。Muon 主要保存的就是这类 momentum 状态；AdamW 还会保存二阶矩，所以状态内存更重。

### 6. 一个简化实现

```python
for p in model.parameters():
    if p.grad is None:
        continue
    buf = momentum_buffer[p]
    buf.mul_(mu).add_(p.grad)
    p.data.add_(buf, alpha=-lr)
```

Muon 可以想象成在 `p.data.add_` 之前多插入一步：如果 `buf` 是适合 Muon 的二维矩阵，就先把它变成近似 $UV^\top$，再作为更新方向。

## 和 Muon 的关系

Muon 的输入通常不是裸梯度，而是 momentum 矩阵。对二维权重，Muon 先得到动量 $M_t$，再用 Newton–Schulz 近似 $M_t$ 的 polar factor，即保留奇异向量、压平奇异值。可以把 momentum 看作“时间维度的平滑”，把 Muon 正交化看作“矩阵方向维度的几何改造”。

## 需要掌握到什么程度

- 能写出 momentum 更新式，知道 $\mu$ 或 $\beta$ 越大，历史记忆越长。
- 能解释为什么 momentum 可能允许更大的有效步长，也可能在学习率过大时造成 overshoot。
- 能看懂 optimizer state 中的 momentum buffer。
- 能区分 momentum 与 Adam 的二阶矩状态：前者累计方向，后者估计逐坐标尺度。

## 常见误区

- 认为 momentum 只是“平滑梯度”。平滑是结果之一，更重要的是它改变了离散动力学。
- 忽略不同框架对 momentum 的尺度约定；同样的学习率在不同写法下未必等价。
- 把 Muon 的 momentum buffer 当作 AdamW 的一阶矩和二阶矩组合。标准 Muon 主要保留一阶动量。

## 自测问题

1. 当连续十步梯度方向相同，momentum 会怎样改变有效更新长度？
2. 当两个坐标方向的梯度交替变号，momentum 会怎样影响它们？
3. Muon 为什么通常对 momentum 而不是裸梯度做正交化？

## 参考入口

- Polyak, *Some methods of speeding up the convergence of iteration methods*。
- Sutskever et al., *On the importance of initialization and momentum in deep learning*。
- KellerJordan/Muon 参考实现中的 momentum buffer。
