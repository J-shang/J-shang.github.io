---
title: "Mixed Precision"
description: "理解 autocast、loss scaling、master weights，以及 BF16 Muon 的数值边界。"
topic: "muon"
section: "engineering"
slug: "mixed-precision"
legacyPaths: ["/notes/mixed-precision/"]
date: 2026-07-01
updated: 2026-07-14
order: 30
readtime: 6
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/深度学习工程/mixed precision.md"
  url: "https://github.com/J-shang/Muon/blob/f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%B7%A5%E7%A8%8B/mixed%20precision.md"
  revision: "f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13"
  syncedAt: "2026-07-14"
  contentHash: "sha256:3129cec33c240d676d71190c8c3a2f50c9088d69f1647cda206bf90435cb4196"
  manifest: "muon"
  managed: true
---
> 层次：深度学习工程
> 信息截点：2026-07-14
> 主推理路径：实现追踪——逐一标注参数、梯度、状态、NS 输入/累加/输出和最终 update 的 dtype 与时序。

## 一句话定位

mixed precision 是把训练中的不同张量和运算放在不同精度上，以换取吞吐、显存和稳定性的平衡。

## 动机问题与最小例子

设一个 fp16 参数当前值约为 $1$，某次未缩放参数改变量为 $10^{-5}$。即使这个 update 本身可表示，把它直接加到低精度参数后也可能因参数附近的表示间隔而没有可见变化。维护 fp32 master weight 的目的之一，就是让许多小 update 能在高精度副本里累积，而不是每步都被低精度存储舍掉。

Muon 又多一层问题：即便参数更新在 fp32 完成，momentum 若以低精度持久保存，或 NS 每步输出都 cast 回低精度，构造出来的方向已经可能改变。所需性质是整条 update pipeline 的误差可控，而不只是 forward 能跑。

## 核心定义

典型混合精度训练会让 forward/backward 的大部分矩阵乘使用 fp16、bf16、tf32 或 fp8，同时保留部分 fp32 计算或 fp32 master copy。fp16 因动态范围较小常配合 loss scaling；bf16 动态范围接近 fp32，通常更易稳定，但尾数位更少。混合精度不是“把所有东西都转半精度”，而是为不同环节选择合适 dtype。

## 假设与适用范围

- dtype 名称不唯一决定实际精度；backend、硬件、autocast policy、accumulate dtype、TF32 设置和自定义 kernel 都必须纳入版本记录。
- loss scaling 主要处理反向梯度的动态范围，不修复 bf16 尾数较粗、NS 舍入误差或低精度参数加法丢失小 update。
- fp32 master weights/state 会提高稳定性但增加持久内存与 cast/带宽成本；收益需要端到端 profile。
- “bf16 通常不需要 loss scaling”是常见经验，不是所有模型/算子上的保证。

## 相关知识展开

### 1. mixed precision 的目标不是“越低越好”

训练需要同时满足三件事：算得快、放得下、数值稳定。低精度能提升吞吐和降低显存，但也可能带来溢出、下溢和舍入误差。mixed precision 的核心是把不同环节放在不同精度上，而不是一刀切。

常见策略是：大矩阵乘使用 bf16/fp16/tf32；归一化、softmax、loss、优化器关键状态保留更高精度；必要时维护 fp32 master weights。

### 2. autocast 做了什么？

PyTorch AMP 的 autocast 会根据算子类型自动选择较低精度或保留 fp32。例如 GEMM/conv 往往可以低精度，某些 reduction 或数值敏感算子则保留 fp32。但 autocast 主要覆盖 forward；optimizer state、手写矩阵乘和自定义 kernel 不一定自动安全。

Muon 实现如果在 optimizer step 里手动写 NS，需要明确内部 dtype。不能假设开了 AMP，Muon 的正交化就自动使用最佳精度。

### 3. loss scaling 为什么存在？

fp16 梯度可能太小，下溢成 0。loss scaling 把 loss 乘以一个较大系数 $S$，反传得到放大的梯度 $Sg$，在 optimizer step 前再除以 $S$。如果发现 Inf/NaN，就降低 scale。

bf16 因为指数范围大，通常不需要动态 loss scaling，但尾数更粗；所以它解决的是动态范围问题，不是所有精度问题。

### 4. master weights 与 optimizer state

有些混合精度训练会用低精度参数参与 forward/backward，但保留 fp32 master weights 用于更新。AdamW 也常把一阶/二阶矩保存在 fp32。这样稳定但占显存。

Muon 的状态主要是 momentum。这个 momentum 用 bf16 还是 fp32，会影响内存和正交化质量。若用低精度存 momentum，再用更高精度做 NS，读写与 cast 成本也要计入。

### 5. Muon dtype/state flow

一个可审计的 step 至少要画出：

```text
parameter storage dtype
  -> forward/autocast compute dtype
  -> gradient production dtype
  -> gradient unscale + finite check
  -> clipping/reduction/accumulation dtype
  -> persistent momentum read/update dtype
  -> NS input / multiply / accumulate / per-step output dtype
  -> shape scaling + weight decay compute dtype
  -> fp32 master update (if any)
  -> parameter cast-back dtype
```

关键顺序是先 unscale 再做 finite check/clip，并在最终 optimizer gradient 上更新 momentum。若 clip 发生在 unscale 前，阈值语义会随 loss scale 改变；若 NS 在不同 rank 的 cast 前后顺序不同，分布式结果可能漂移。

### 6. mixed precision 对 wall-clock 的影响

低精度不保证总时间更短。瓶颈可能在数据加载、通信、optimizer step、kernel launch 或 checkpoint。Muon 尤其要看 optimizer step：如果 NS 矩阵乘没有高效 kernel，forward/backward 变快也可能被 optimizer 抵消。

### 7. 一个检查清单

调 Muon mixed precision 时建议逐项确认：

- forward/backward 是否启用 AMP；
- attention softmax 和 norm 是否稳定；
- gradient unscale 是否在 clipping 前完成；
- momentum state dtype 是什么；
- NS 内部是否使用足够稳定的 accumulate dtype；
- update cast 回参数 dtype 前是否做了缩放和检查。

再加两个最小 A/B：

1. 固定 gradient 与 state，比较 fp32 state+fp32 NS、bf16 state+fp32 NS、bf16 state+bf16 NS 的一步 update 和连续 100 步漂移。
2. profile autocast forward/backward、optimizer NS、cast、communication 和整个 step；同时报告 peak persistent/temporary memory。

若只在“bf16 state + fp32 NS”漂移，第一嫌疑是持久状态量化；若 state 相同而 bf16 NS 漂移，第一嫌疑才是正交化路径。

## 和 Muon 的关系

Muon 除了模型 forward/backward，还在 optimizer step 内部做矩阵正交化。也就是说，mixed precision 策略要覆盖 momentum buffer、NS 迭代、权重衰减和最终参数更新。若只优化模型计算而忽略 Muon 的矩阵乘 dtype，可能出现 optimizer step 成为瓶颈，或在低精度下近似质量下降。

## 需要掌握到什么程度

- 能说明 fp16、bf16、tf32、fp32 的大致角色。
- 知道 autocast 主要影响算子选择，不等于所有状态都自动安全。
- 能解释 loss scaling 的目的，以及为什么 bf16 常可少依赖它。
- 能在 Muon 实验里记录 forward dtype、state dtype、NS dtype 和 accumulate dtype。
- 能画出完整 dtype/state flow，并说明 unscale、clip、momentum、NS 的正确相对顺序。
- 能用状态精度与计算精度的交叉 A/B 定位误差来自哪里。

## 常见误区

- 把 mixed precision 当成单一开关。不同框架和硬件默认策略差异很大。
- 忽略归一化、softmax、optimizer update 这些对数值更敏感的环节。
- 认为 Muon 节省 AdamW 二阶状态后，dtype 就不重要了；正交化矩阵乘仍然可能很贵。

## 自测问题

1. gradient finite，但连续 100 步后 bf16-momentum 路径与 fp32-momentum 路径方向明显分叉；怎样排除 NS dtype 的影响？
2. loss scale 增大 1024 倍后，如果 clipping 仍在 unscale 前执行，实际未缩放梯度阈值发生了什么变化？
3. forward/backward 快 30%，整步只快 3%。你至少需要哪四段计时和哪两类 memory 指标才能定位瓶颈？

## 参考入口

- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html) —— autocast 与 GradScaler 的官方语义和推荐时序。
- [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) —— backend、TF32、reduction 和低精度行为边界。
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/) —— Tensor Core、master weights 与 loss scaling 的系统入口。
- [Liu et al., *Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982) —— 规模化 Muon 的工程背景；具体 dtype 合同仍以所用代码版本为准。
