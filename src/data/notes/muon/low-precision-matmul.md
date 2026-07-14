---
title: "低精度矩阵乘"
description: "输入精度、累加精度与舍入误差怎样进入优化器更新路径。"
topic: "muon"
section: "numerical-computing"
slug: "low-precision-matmul"
legacyPaths: ["/notes/low-precision-matmul/"]
date: 2026-07-01
updated: 2026-07-01
order: 22
readtime: 6
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/数值计算/低精度矩阵乘.md"
  url: "https://github.com/J-shang/Muon/blob/7458bf6dbff95ca416a8ca9069308d5cc6907f96/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97/%E4%BD%8E%E7%B2%BE%E5%BA%A6%E7%9F%A9%E9%98%B5%E4%B9%98.md"
  revision: "7458bf6dbff95ca416a8ca9069308d5cc6907f96"
  syncedAt: "2026-07-14"
  contentHash: "sha256:eb9cb00ad5544e2a5498815acc6e12999fb9c22f6014cc726c0795ccf77a3246"
  manifest: "muon"
  managed: true
---
> 层次：数值计算

## 一句话定位

低精度矩阵乘用更少位宽换吞吐和显存，但会把舍入、溢出、累加精度和数值稳定性变成优化器设计的一部分。

## 核心定义

深度学习训练常见数值格式包括 fp32、tf32、fp16、bf16、fp8 等。矩阵乘通常涉及输入精度、乘法精度和累加精度：例如输入用 bf16，累加用 fp32。低精度能显著提升 GPU Tensor Core 利用率和内存带宽效率，但也会带来更粗的量化间隔、更小或不同的动态范围，以及重复运算中的误差积累。

## 相关知识展开

### 1. 精度格式看两个核心指标

浮点格式主要看动态范围和有效精度。动态范围决定能表示多大/多小的数；有效精度决定两个相近数字之间能分多细。

fp16 有较多尾数位但指数位少，动态范围较窄；bf16 指数位接近 fp32，动态范围大，但尾数位更少。tf32 常用于 NVIDIA GPU 上的 fp32 矩阵乘加速；fp8 则进一步降低位宽，通常需要更细的缩放策略。

### 2. “输入 dtype”和“累加 dtype”不是一回事

矩阵乘 $C=AB$ 的每个元素都是很多乘积的和：

$$
C_{ij}=\sum_k A_{ik}B_{kj}.
$$

即使 $A,B$ 是 bf16，硬件也可能用 fp32 累加。累加精度非常重要，因为求和项很多，舍入误差会积累。读训练实现时要看的是：输入是什么 dtype、乘法是什么路径、累加是什么 dtype、输出又 cast 到哪里。

### 3. 为什么低精度会溢出或下溢？

如果数值超过格式能表示的最大范围，会变成 Inf；如果太小，可能被舍入成 0 或 subnormal。fp16 的动态范围较窄，所以早期混合精度训练常用 loss scaling：把 loss 放大后反传，避免梯度太小下溢，再在更新前缩回去。

bf16 因为指数范围大，通常不太需要动态 loss scaling，但它尾数更粗；所以它解决的是动态范围问题，不是所有精度问题。

### 4. 矩阵乘误差为什么会影响优化器？

普通 SGD 的 optimizer step 主要是逐元素加减乘，数值路径相对简单。Muon 的 optimizer step 内部包含多次矩阵乘，例如 Newton–Schulz 里的 $X X^\top X$。这意味着低精度不仅影响 forward/backward，也影响“如何构造更新方向”。

如果 NS 在低精度下把奇异值推歪，最终 update 的方向和尺度都会变。训练曲线可能表现为 loss 抖动、某些层 update RMS 异常，或偶发 NaN。

### 5. 一个误差积累直觉

假设一次矩阵乘已有小的舍入误差。NS 不是只做一次矩阵乘，而是多步迭代；每一步又把上一步输出作为下一步输入。误差因此可能被后续多项式继续加工。好的实现会通过缩放、选择 bf16/fp32 累加、限制步数或加入 restart 来控制这种误差。

### 6. 低精度不是只为省显存

很多人首先想到低精度是省显存，但大模型训练里更关键的常常是吞吐。Tensor Core 对 bf16/fp16/fp8 GEMM 的吞吐远高于普通 fp32。Muon 的 NS 如果能走高效 GEMM kernel，就有机会把额外正交化成本压低；如果 kernel 不匹配，理论上的 token/FLOP 优势可能被 wall-clock 吃掉。

### 7. 实验记录应该写清楚什么？

记录 Muon 实验时，至少写：参数 dtype、gradient dtype、momentum state dtype、NS 内部矩阵乘 dtype 与 accumulate dtype、是否使用 loss scaling、是否启用 fused/foreach/custom kernel、发生 NaN 时是哪一层和哪个 buffer。

## 和 Muon 的关系

Muon 的正交化核心是若干次矩阵乘，因此低精度路径直接影响 optimizer step。NS 迭代会反复计算类似 $X X^\top X$ 的项，若输入缩放不当或累加精度不足，可能出现近似偏差、NaN 或收敛变差。工程实现要决定 momentum state、NS 输入、矩阵乘和最终 update 分别用什么 dtype。

## 需要掌握到什么程度

- 能区分 fp16 与 bf16 的动态范围和精度取舍。
- 知道矩阵乘的 accumulate dtype 往往比输入 dtype 更重要。
- 能解释 mixed precision 训练中为什么需要 loss scaling 或选择 bf16。
- 能把 Muon 的 NS 步数、dtype、kernel 选择写进实验记录。

## 常见误区

- 只看参数 dtype，不看 optimizer state 和矩阵乘累加 dtype。
- 认为低精度误差只影响 forward/backward；优化器内部的矩阵乘同样受影响。
- 看到某个实现“支持 bf16/fp16”就假设所有矩阵形状和硬件上都稳定。

## 自测问题

1. 为什么 bf16 通常比 fp16 更不容易溢出，但精度更粗？
2. Muon 的 NS 迭代为什么比普通 SGD 更新更关心矩阵乘精度？
3. 如果 optimizer step 出现 NaN，你会优先检查哪些 dtype 和缩放设置？

## 参考入口

- NVIDIA Tensor Core 与混合精度训练文档。
- PyTorch AMP 文档。
- Gram Newton–Schulz / GramMuon 关于低精度 NS kernel 的讨论。
