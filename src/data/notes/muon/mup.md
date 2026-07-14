---
title: "μP"
description: "把超参数从小模型迁移到大模型时，哪些尺度规则不能靠直觉复制。"
topic: "muon"
section: "experiments"
slug: "mup"
legacyPaths: ["/notes/mup/"]
date: 2026-07-01
updated: 2026-07-14
order: 42
readtime: 6
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/LLM 实验方法/muP.md"
  url: "https://github.com/J-shang/Muon/blob/f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/LLM%20%E5%AE%9E%E9%AA%8C%E6%96%B9%E6%B3%95/muP.md"
  revision: "f6b7bd6ea9ca6a833648ad92c9f339cf56ccdf13"
  syncedAt: "2026-07-14"
  contentHash: "sha256:b9758b9b79fc6d3ea4c3f7709b0033254203b768b9c8403adb0d2a178b3734f9"
  manifest: "muon"
  managed: true
---
> 层次：LLM 实验方法
> 信息截点：2026-07-14
> 主推理路径：约束驱动——先定义“跨宽度可迁移”要保持的可观测量，再检查参数化和 optimizer recipe 是否满足。

## 一句话定位

muP（maximal update parametrization）是一套让超参数更容易跨宽度迁移的参数化方法，用来减少小模型调参到大模型时的失真。

## 动机问题与最小例子

希望用宽度 $d=256$ 的 proxy 选择学习率，再把它迁移到 $d=4096$。若普通参数化让某个 hidden matrix 的梯度或 update-to-weight ratio 随 $d$ 系统性变化，那么“大模型复用同一学习率”并不是同一实验条件。

一个不依赖具体 muP 规则、但可直接测量的诊断是参数组 $p$ 的相对更新：

$$
r_p(t)=\frac{\|\Delta p_t\|_F}{\|p_t\|_F+\varepsilon}.
$$

若 proxy 与 target 在训练初期的对应参数角色上，$r_p(t)$ 相差几个数量级，超参数迁移失败并不令人意外。muP 的目标不是强迫所有 $r_p$ 完全相等，而是通过角色相关的初始化、乘子与学习率缩放，得到良定义的宽度极限和可迁移的训练动态。

## 核心定义

普通参数化下，模型宽度变化会改变激活、梯度和更新尺度，导致在小模型上调出的学习率、初始化或 optimizer 配置不一定能迁移到大模型。muP 通过特定初始化、学习率缩放和参数分类，让不同宽度模型在无限宽极限下保持可比较的更新行为。它不是一个优化器，而是训练 recipe 的尺度约定。

## 假设与适用范围

- muP 的理论陈述针对指定网络族和宽度扩展规则；“宽度变大”必须说明哪些维度扩展、哪些保持固定。
- 参数角色（input-like、hidden、output-like、bias/norm 等）必须正确分类；base/delta shapes 只是让框架推断这些角色的工具，不是证明。
- 从小模型到大模型通常要求架构、数据处理、optimizer 语义和训练阶段足够一致。换 optimizer、scale mode、batch regime 或数值路径会引入 muP 没有自动消除的新变量。
- 超参数 transfer 是经验结果，不是任意有限宽模型上的 exact identity；仍需要中间宽度和 target width 确认点。
- 本文把 muP 与 Muon 的配合标为 **implementation + empirical relation**，不是已证明的普适等价。

## 相关知识展开

### 1. 为什么小模型调参不能直接搬到大模型？

模型宽度变大时，很多量的尺度会变：激活方差、梯度方差、logit 尺度、每层 update-to-weight ratio。若参数化方式不合适，小模型上最好的学习率到了大模型可能过大或过小。

这会让低成本 proxy 实验失效：你以为在小模型上比较的是优化器，其实比较到的是“哪个优化器更适应错误的尺度迁移”。

### 2. muP 的核心思想

muP 试图定义一套参数化，使不同宽度模型在训练初期和训练过程中保持可比较的更新尺度。它会规定不同参数类型的初始化和学习率如何随宽度变化。

关键不是“所有参数同一个缩放”，而是按参数角色分类：输入层、隐藏层、输出层、bias、norm 等可能有不同规则。

### 3. base shapes 是什么？

实践中使用 muP 常需要记录一个 base model 的参数形状，称为 base shapes。框架根据 base shape 和目标模型 shape 判断哪些维度是宽度维，并应用对应缩放规则。

如果 base shape 配错，muP 可能悄悄失效。读实验时，看到“用了 muP”还不够，要看 base model 如何定义、哪些参数被纳入 muP、哪些参数例外处理。

最小 shape 审计应输出：

```text
parameter name | role | base shape | delta/target shape
width dimensions | init multiplier | lr multiplier | optimizer route | Muon scale mode
```

如果一个 fused tensor 同时包含不同语义块，例如 Q/K/V 或 gate/up，它的单一 shape 分类可能掩盖实际需要的分块规则。

### 4. muP 不是优化器

muP 不决定 update 方向，它决定参数化和超参数如何随宽度缩放。AdamW、SGD、Muon 都可以在某种参数化下训练。问题是：某个为 AdamW 调好的 muP recipe，是否对 Muon 仍然合适？

Muon 的 update RMS、polar direction 和参数路由会引入额外尺度因素，所以需要专门验证。

### 5. muP 如何帮助 Muon 实验？

Muon 论文若想用小模型预测大模型，必须减少“宽度变化导致的超参数漂移”。muP 可以提供更稳的小模型 proxy，让你更有信心地比较不同优化器在规模上的趋势。

但 muP 不能替代 scaling sweep。它降低不确定性，不消除架构、数据、batch、数值实现和分布式系统带来的变化。

### 6. Muon 引入了哪一个额外尺度？

理想 polar factor $O\in\mathbb{R}^{m\times n}$ 的元素 RMS 为

$$
\operatorname{RMS}(O)=\frac{1}{\sqrt{\max(m,n)}}.
$$

因此 Muon 的 `spectral`、`unit_rms_norm` 或其他 shape scaling 会直接改变宽度扩展时的 update scale。即使普通 AdamW 参数已经按 muP 分类，如果 Muon scale mode 又乘入一个依赖 $m,n$ 的因子，最终有效学习率仍可能随宽度漂移。

本地 Megatron-LM 在 Muon 与 muP 同时启用时会检查 scale mode，并为不同参数角色构造 optimizer override。代码走读入口是 [`get_mup_config_overrides`](https://github.com/NVIDIA/Megatron-LM/blob/0823c731ed7d793aef047b6a64f2dbbf32bf6e2c/megatron/core/optimizer/__init__.py)；应核对实际 commit 的 warning、LR/epsilon multiplier 与参数 predicate，而不是只看命令行写了 `--use-mup`。

### 7. 可复现的 transfer 检查

对宽度 $d_0<d_1<d_2$，先在 $d_0$ 选超参数，再原样迁移到 $d_1,d_2$，至少记录：

| 检查 | 目的 |
|---|---|
| 初始 activation/logit RMS | 检查 forward 参数化 |
| 每类参数的 grad RMS 与 $r_p(t)$ | 检查 backward/update 尺度 |
| Muon polar 前/后 RMS 与 scale factor | 定位额外 width dependence |
| 固定 token 的 loss 与最佳 LR 排序 | 检查超参数是否真正迁移 |
| 一个重新调优的 target-width 对照 | 估计 transfer gap，而非只看绝对 loss |

如果 $d_0\to d_1$ 成功而 $d_1\to d_2$ 失败，先检查哪一类参数的 observable 首先漂移；不要直接把失败归因于“有限宽误差”。

### 8. 公平比较要注意什么？

公平的 Muon vs AdamW 比较最好说明：两者是否都用 muP；两者是否分别调学习率；Muon 的 update scaling 是否与 muP 规则一致；embedding、output head、norm 参数是否用相同路由；小模型上调出的超参数是否在大模型上验证过。

## 和 Muon 的关系

Muon 的核心也关心“更新尺度”与矩阵形状，因此和 muP 的相互作用很重要。若用小模型 proxy 评估 Muon，再把学习率和缩放规则迁移到大模型，muP 可以帮助降低宽度变化带来的混杂因素。但 Muon 的正交化、update RMS、参数路由可能引入额外尺度约定，不能假设 AdamW 的 muP 配方原样适用。

## 需要掌握到什么程度

- 能说明 muP 解决的是跨宽度超参数迁移问题。
- 能区分 base shapes、宽度缩放和普通超参数扫描。
- 能理解 muP 不是自动保证跨架构、跨优化器全部可迁移。
- 能在读 Muon 实验时确认是否使用 muP，以及 AdamW/Muon 是否公平调参。
- 能从 base/delta shapes 生成参数角色审计表，并用 $r_p(t)$ 定位 transfer 失败的第一层。
- 能解释 Muon scale mode 如何重新引入 width-dependent update scale。

## 常见误区

- 把 muP 当成“更好的初始化”或“新优化器”。它主要是参数化和缩放规则。
- 认为用了 muP 就不需要大模型验证。muP 减少失真，不消灭所有规模效应。
- 忽略 Muon 自身 update scaling 与 muP 学习率缩放之间可能互相影响。

## 自测问题

1. 两个宽度的 loss 曲线不同，而 activation RMS 相近、Muon update RMS 相差 8 倍。你会先怀疑 base shape、LR multiplier 还是 Muon scale mode？为什么？
2. 只在 $d=256$ 调参并在 $d=4096$ 验证一次，为什么不能区分“平滑迁移”与“偶然在两个端点都可用”？最小增加哪个宽度或对照？
3. 若 AdamW 使用 muP 的 hidden LR scaling，而 Muon 使用 `spectral` scale，怎样记录才能判断比较是在测 optimizer geometry 还是 width-dependent effective LR？

## 参考入口

- [Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*](https://arxiv.org/abs/2203.03466) —— muP 超参数迁移的主论文入口，重点读参数角色和 transfer 条件。
- [microsoft/mup](https://github.com/microsoft/mup) —— 官方参考实现；重点看 base/delta shapes 如何决定 infshape 和 multiplier。
- [Yang & Hu, *Feature Learning in Infinite-Width Neural Networks*](https://arxiv.org/abs/2011.14522) —— 理解 maximal update 参数化为什么不同于 NTK/lazy 极限。
- [*Practical Efficiency of Muon for Pretraining*](https://arxiv.org/abs/2505.02222) —— Muon + muP 的规模化实验和 telescoping transfer；把结果限定在论文设置内。
