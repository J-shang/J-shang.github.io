---
title: "Muon 实现与代码走读"
description: "沿输入、shape、state、branch、output 与 test 把论文公式落到真实训练代码。"
topic: "muon"
section: "research-practice"
slug: "muon-code-reading"
legacyPaths: ["/notes/muon-code-reading/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 81
source:
  repository: "J-shang/Muon"
  path: "notes/实现与代码走读.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/notes/%E5%AE%9E%E7%8E%B0%E4%B8%8E%E4%BB%A3%E7%A0%81%E8%B5%B0%E8%AF%BB.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:0bc5ef122e4029f7e90ae3cb53563d3af17697c76fbfdfb82bd6b45a8e39970d"
  manifest: "muon"
  managed: true
---
> 本地分析目标：`Megatron-LM` submodule commit `0823c731ed7d793aef047b6a64f2dbbf32bf6e2c`；其 `pyproject.toml` 将 `Emerging-Optimizers` 固定到 `v0.2.0`。
> 代码复核日期：2026-07-16。本文只对页首固定的 commit 和依赖版本负责。

## 先读哪些实现

| 目标 | 文件/入口 | 要核对的不变量 |
|---|---|---|
| 最小算法 | [KellerJordan/Muon](https://github.com/KellerJordan/Muon) | momentum 顺序、NS 缩放、参数限制 |
| PyTorch 用户 API | [`torch.optim.Muon`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) | 只接收 2D hidden-layer 参数；`adjust_lr_fn` 与 `ns_steps` |
| Megatron 配置 | [`optimizer_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/0823c731ed7d793aef047b6a64f2dbbf32bf6e2c/megatron/core/optimizer/optimizer_config.py) | scale/coefficient/TP mode、scalar optimizer、adaptive state |
| Megatron 核心分支 | [`emerging_optimizers.py`](https://github.com/NVIDIA/Megatron-LM/blob/0823c731ed7d793aef047b6a64f2dbbf32bf6e2c/megatron/core/optimizer/emerging_optimizers.py) | global shape、QKV split、`orthogonalize` 与 TP 分支 |
| 参数路由 | [`optimizer/__init__.py`](https://github.com/NVIDIA/Megatron-LM/blob/0823c731ed7d793aef047b6a64f2dbbf32bf6e2c/megatron/core/optimizer/__init__.py) | 每个参数恰好一组、embedding/output/scalar 的 override |
| layer-wise 分布式 | [`layer_wise_optimizer.py`](https://github.com/NVIDIA/Megatron-LM/blob/0823c731ed7d793aef047b6a64f2dbbf32bf6e2c/megatron/core/optimizer/layer_wise_optimizer.py) | owner rank、shard-aligned layout、同步时序 |
| correctness tests | [`test_emerging_optimizers.py`](https://github.com/NVIDIA/Megatron-LM/blob/0823c731ed7d793aef047b6a64f2dbbf32bf6e2c/tests/unit_tests/test_emerging_optimizers.py) | TP modes、QKV、scale、系数、steps、adaptive variant |

更详细的逐函数说明见 [Megatron-LM Muon 实现解析](/topics/muon/megatron-muon-implementation/)。本页只保留跨实现都要检查的更新顺序和最小测试。

## 数据与控制流

```text
named parameter + metadata
  -> route to Muon or scalar optimizer exactly once
  -> accumulate/reduce final gradient for this optimizer step
  -> update persistent momentum state
  -> optional Nesterov combination
  -> recover semantic matrix blocks (for example Q/K/V)
  -> choose TP mode and global/local shape contract
  -> finite-step Newton–Schulz orthogonalization
  -> shape-aware scale + decoupled weight decay
  -> owner/local shard update
  -> parameter synchronization before next consumer
```

任何一步改变对象，后续公式的含义都会变。例如，把 fused QKV 当成一个大矩阵正交化，不是“同一算法的更快 kernel”，而是换了 polar 的对象。

## 核心不变量

### 1. 参数覆盖不变量

设可训练参数集合为 $P$，各 optimizer group 为 $P_i$。生产配置应断言

$$
\bigcup_i P_i=P,
\qquad
P_i\cap P_j=\varnothing\quad(i\ne j).
$$

测试后果：打印遗漏参数和重复参数的名字；tied embedding/output head 必须按对象 identity 去重，不能只按 name。

### 2. shape 不变量

对语义矩阵 $W\in\mathbb{R}^{m\times n}$，需要同时保存：

```text
semantic/global shape: (m, n)
physical/local shard shape: depends on TP/DP/EP layout
orthogonalization object: whole W, Q/K/V block, expert block, or declared approximation
scale shape: must match the recipe's semantic contract
```

一般地，对按列分片 $M=[M_1\;M_2]$，

$$
\operatorname{polar}(M)
\ne
[\operatorname{polar}(M_1)\;\operatorname{polar}(M_2)].
$$

因此 `blockwise` 应标成 approximation；`duplicated`/`distributed` 是否等价要由多 rank 测试确认，不能由命名推断。

### 3. state/update 时序不变量

梯度累积与数据并行归约完成后，才应形成一次 optimizer step 的 momentum：

$$
M_t=\beta M_{t-1}+G_t^{\text{final}}.
$$

若某路径对 micro-batch 梯度逐次做 NS，再累加正交化结果，因为 polar/NS 非线性，一般不等价于对最终梯度做一次 NS。

### 4. 数值不变量

至少按层抽样记录：

$$
e_{\mathrm{orth}}=
\begin{cases}
\|X^\top X-I_n\|_F/\sqrt n,&m\ge n,\\
\|XX^\top-I_m\|_F/\sqrt m,&m<n,
\end{cases}
$$

以及 `isfinite`、singular-value range、update RMS、与 fp32 SVD polar 的 cosine。公式中的 $X$ 是完成 NS 和声明的 scale 之前/之后哪个对象，日志必须写清。

## 变体分叉点

| 轴 | 选择 | 改变的是 | 必须补的测试 |
|---|---|---|---|
| momentum | plain / Nesterov | NS 输入矩阵 | 一步手算与参考实现对齐 |
| scale | spectral / unit RMS / shape scaling | 每层 update RMS | 多 shape 的解析值与实现值 |
| coefficient | simple / quintic / other | 奇异值映射多项式 | 固定谱上的误差曲线 |
| TP mode | duplicated / distributed / blockwise | global polar 的实现或近似 | 多 rank output equality + communication profile |
| QKV | split / fused | 被正交化的语义对象 | block boundary 与重拼 shape |
| optimizer | Muon / adaptive Muon | polar 前后附加统计与状态 | state_dict round trip、额外字节数 |
| layout | replicated / layer-wise owner | state/parameter 所有权与同步 | next forward 前参数一致性 |

## 最小验证套件

1. **单矩阵数学测试**：方阵、高矩阵、宽矩阵、近秩亏矩阵，对比 fp64 SVD polar 与不同 NS steps/dtype。
2. **单 rank step 测试**：固定 gradient，手算 momentum/Nesterov/WD/scale；检查一个 step 和 state。
3. **参数路由测试**：覆盖 hidden linear、embedding、tied head、bias、norm、router、fused QKV、expert。
4. **多 rank 语义测试**：同一 global matrix 比较 `duplicated`、`distributed`、`blockwise`；预期前两者在合同允许的容差内一致，`blockwise` 可不同但必须显式记录。
5. **checkpoint 测试**：保存/加载后下一步更新与不中断运行一致，并覆盖 owner rank/state repartition。
6. **性能测试**：分开计时 gradient communication、NS GEMM、parameter sync 和整个 step；只报 optimizer kernel 时间不够。

## 版本边界和开放问题

- 本地 submodule 与 NVIDIA `main` 会继续演进；代码结论只对页首 commit 有效。
- `Emerging-Optimizers` 的 API/系数集合仍属实验性依赖；本页目标固定为 `v0.2.0`，其他复现实验也必须记录实际解析版本，而不只固定 Megatron commit。
- router/gate 是否走 Muon 是 recipe 选择，不能从 `ndim == 2` 推出正确答案。
- 生产上应进一步确认：实际 NCCL 拓扑、expert parallel group、buffer reuse 和 overlap 是否满足源码层面的时序假设。
