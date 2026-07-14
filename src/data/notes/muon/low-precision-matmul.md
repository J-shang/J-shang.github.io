---
title: "低精度矩阵乘"
description: "输入精度、累加精度与舍入误差怎样进入优化器更新路径。"
topic: "muon"
section: "numerical-computing"
slug: "low-precision-matmul"
legacyPaths: ["/notes/low-precision-matmul/"]
date: 2026-07-01
updated: 2026-07-14
order: 22
readtime: 6
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/数值计算/低精度矩阵乘.md"
  url: "https://github.com/J-shang/Muon/blob/f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97/%E4%BD%8E%E7%B2%BE%E5%BA%A6%E7%9F%A9%E9%98%B5%E4%B9%98.md"
  revision: "f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13"
  syncedAt: "2026-07-14"
  contentHash: "sha256:fcf129c4c678e961af9d7a373dddbda95673515e4302f41106bd7724899f0467"
  manifest: "muon"
  managed: true
---
> 层次：数值计算
> 信息截点：2026-07-14
> 主推理路径：从误差模型推到可测诊断，再回到 Muon 的有限步 NS。

## 一句话定位

低精度矩阵乘用更少位宽换吞吐和显存，但会把舍入、溢出、累加精度和数值稳定性变成优化器设计的一部分。

## 动机问题与最小例子

Muon 的一步 NS 会把上一轮矩阵乘结果重新送入下一轮。假设某个输入只有两个奇异值 $1$ 和 $10^{-4}$：bf16 在数值 1 附近的间隔远大于 $10^{-4}$，所以“动态范围容得下 $10^{-4}$”并不表示相关细微差异能在所有运算中保留。所需性质不是单纯 `isfinite`，而是低精度路径仍在可接受误差内保持奇异值映射和 update scale。

这个例子是数值分辨率的教学构造，不表示实际 kernel 会直接把该奇异值舍入为零；矩阵基、乘加顺序、缩放和累加 dtype 都会影响结果。

## 核心定义

深度学习训练常见数值格式包括 fp32、tf32、fp16、bf16、fp8 等。矩阵乘通常涉及输入精度、乘法精度和累加精度：例如输入用 bf16，累加用 fp32。低精度能显著提升 GPU Tensor Core 利用率和内存带宽效率，但也会带来更粗的量化间隔、更小或不同的动态范围，以及重复运算中的误差积累。

## 假设与适用范围

- 以下经典舍入界假设标准浮点模型、无 overflow/underflow，并把每次基本运算的相对误差界为 unit roundoff $u$；真实 GPU kernel 还受 fused multiply-add、reduction tree、Tensor Core 模式和 flush-to-zero 影响。
- fp16/bf16/tf32/fp8 不是单一执行路径；必须记录 input、multiply、accumulate、output 和 state dtype。
- 正交误差小是数值性质，不自动表示训练 loss 更好；最终还要对齐 update RMS 和端到端预算。
- 对低精度 NS 的结论只在记录的 shape、谱、缩放、系数、steps、硬件和 kernel 上成立。

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

在经典模型下，长度为 $k$ 的 dot product 可用

$$
\gamma_k=\frac{ku}{1-ku}
$$

描述一个最坏情况相对误差因子，前提是 $ku<1$。这不是实际误差的精确预测，但明确揭示两件事：累加项数和 accumulate dtype 都会进入误差预算；只写输入是 bf16 不足以复现实验。

### 3. 为什么低精度会溢出或下溢？

如果数值超过格式能表示的最大范围，会变成 Inf；如果太小，可能被舍入成 0 或 subnormal。fp16 的动态范围较窄，所以早期混合精度训练常用 loss scaling：把 loss 放大后反传，避免梯度太小下溢，再在更新前缩回去。

bf16 因为指数范围大，通常不太需要动态 loss scaling，但它尾数更粗；所以它解决的是动态范围问题，不是所有精度问题。

### 4. 矩阵乘误差为什么会影响优化器？

普通 SGD 的 optimizer step 主要是逐元素加减乘，数值路径相对简单。Muon 的 optimizer step 内部包含多次矩阵乘，例如 Newton–Schulz 里的 $X X^\top X$。这意味着低精度不仅影响 forward/backward，也影响“如何构造更新方向”。

如果 NS 在低精度下把奇异值推歪，最终 update 的方向和尺度都会变。训练曲线可能表现为 loss 抖动、某些层 update RMS 异常，或偶发 NaN。

### 5. 一个误差积累直觉

假设一次矩阵乘已有小的舍入误差。NS 不是只做一次矩阵乘，而是多步迭代；每一步又把上一步输出作为下一步输入。误差因此可能被后续多项式继续加工。好的实现会通过缩放、选择 bf16/fp32 累加、限制步数或加入 restart 来控制这种误差。

更具体地，若第 $j$ 步实际得到

$$
\widehat X_{j+1}=p(\widehat X_j)+E_j,
$$

其中 $p$ 是 NS 多项式、$E_j$ 汇总该步矩阵乘与 cast 误差，那么下一步误差不仅包含新的 $E_{j+1}$，还会被 $p$ 在当前谱附近的局部敏感度放大或压缩。因此不能只测最终 `isfinite`；需要观察每步 singular-value range 或至少 orthogonality proxy。

### 6. 低精度不是只为省显存

很多人首先想到低精度是省显存，但大模型训练里更关键的常常是吞吐。Tensor Core 对 bf16/fp16/fp8 GEMM 的吞吐远高于普通 fp32。Muon 的 NS 如果能走高效 GEMM kernel，就有机会把额外正交化成本压低；如果 kernel 不匹配，理论上的 token/FLOP 优势可能被 wall-clock 吃掉。

### 7. 一个可判错的对照实验

构造相同奇异向量、不同条件数的矩阵，比较 fp64 SVD polar、fp32 NS 和目标低精度 NS。每步记录：

```text
shape / prescribed singular values
input, multiply, accumulate, output, state dtype
normalization / coefficients / steps
singular-value min/max after each step
orthogonality error / polar cosine / update RMS
kernel time / peak temporary memory / isfinite
```

诊断逻辑：若 cosine 尚可但 RMS 漂移，优先检查 scale/cast；若小奇异方向首先偏离，检查谱和低精度分辨率；若不同 shape 只在某些尺寸失败，检查 kernel 路径而非先归咎算法。

### 8. 实验记录应该写清楚什么？

记录 Muon 实验时，至少写：参数 dtype、gradient dtype、momentum state dtype、NS 内部矩阵乘 dtype 与 accumulate dtype、是否使用 loss scaling、是否启用 fused/foreach/custom kernel、发生 NaN 时是哪一层和哪个 buffer。

## 和 Muon 的关系

Muon 的正交化核心是若干次矩阵乘，因此低精度路径直接影响 optimizer step。NS 迭代会反复计算类似 $X X^\top X$ 的项，若输入缩放不当或累加精度不足，可能出现近似偏差、NaN 或收敛变差。工程实现要决定 momentum state、NS 输入、矩阵乘和最终 update 分别用什么 dtype。

## 需要掌握到什么程度

- 能区分 fp16 与 bf16 的动态范围和精度取舍。
- 知道矩阵乘的 accumulate dtype 往往比输入 dtype 更重要。
- 能解释 mixed precision 训练中为什么需要 loss scaling 或选择 bf16。
- 能把 Muon 的 NS 步数、dtype、kernel 选择写进实验记录。
- 能说明 $\gamma_k$ 误差模型的假设，并解释为什么它只是上界而不是实际 kernel 预测。
- 能用逐步谱/正交误差区分 overflow、scale 漂移和低精度方向误差。

## 常见误区

- 只看参数 dtype，不看 optimizer state 和矩阵乘累加 dtype。
- 认为低精度误差只影响 forward/backward；优化器内部的矩阵乘同样受影响。
- 看到某个实现“支持 bf16/fp16”就假设所有矩阵形状和硬件上都稳定。

## 自测问题

1. 两条 NS 路径都没有 NaN，但 bf16 路径的 update RMS 是 fp32 的 1.8 倍、cosine 是 0.99。你会把问题归为方向、尺度还是 overflow？下一项检查是什么？
2. 输入 bf16、fp32 accumulate 为什么仍可能在每步输出 cast 回 bf16 时丢失信息？请沿两步迭代说明误差如何进入下一步。
3. 某 shape 只在 Tensor Core kernel 上失败而普通 fp32 matmul 正常。怎样设计最小实验区分 dtype、kernel 和矩阵条件数？

## 参考入口

- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/) —— 浮点格式、loss scaling 和 Tensor Core 执行背景。
- [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) —— reduction、TF32 和 backend 精度边界；用来避免把 dtype 名称等同于执行语义。
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html) —— autocast/gradient scaling 的官方行为入口。
- [Gram Newton–Schulz / GramMuon](https://dao-lab.ai/blog/2026/gram-newton-schulz/) —— 低精度 Gram 路径与 restart 的系统动机；属于作者博客/实现说明，性能数字需本地复现。
