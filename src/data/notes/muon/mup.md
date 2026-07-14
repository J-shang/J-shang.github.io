---
title: "μP"
description: "把超参数从小模型迁移到大模型时，哪些尺度规则不能靠直觉复制。"
topic: "muon"
section: "experiments"
slug: "mup"
legacyPaths: ["/notes/mup/"]
date: 2026-07-01
updated: 2026-07-01
order: 42
readtime: 6
source:
  repository: "J-shang/Muon"
  path: "必备知识地图/LLM 实验方法/muP.md"
  url: "https://github.com/J-shang/Muon/blob/7458bf6dbff95ca416a8ca9069308d5cc6907f96/%E5%BF%85%E5%A4%87%E7%9F%A5%E8%AF%86%E5%9C%B0%E5%9B%BE/LLM%20%E5%AE%9E%E9%AA%8C%E6%96%B9%E6%B3%95/muP.md"
  revision: "7458bf6dbff95ca416a8ca9069308d5cc6907f96"
  syncedAt: "2026-07-14"
  contentHash: "sha256:3c3617fe37d58ec0e6dfd16ba6f70f01028e1096907081e3f041f426b8f92c0c"
  manifest: "muon"
  managed: true
---
> 层次：LLM 实验方法

## 一句话定位

muP（maximal update parametrization）是一套让超参数更容易跨宽度迁移的参数化方法，用来减少小模型调参到大模型时的失真。

## 核心定义

普通参数化下，模型宽度变化会改变激活、梯度和更新尺度，导致在小模型上调出的学习率、初始化或 optimizer 配置不一定能迁移到大模型。muP 通过特定初始化、学习率缩放和参数分类，让不同宽度模型在无限宽极限下保持可比较的更新行为。它不是一个优化器，而是训练 recipe 的尺度约定。

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

### 4. muP 不是优化器

muP 不决定 update 方向，它决定参数化和超参数如何随宽度缩放。AdamW、SGD、Muon 都可以在某种参数化下训练。问题是：某个为 AdamW 调好的 muP recipe，是否对 Muon 仍然合适？

Muon 的 update RMS、polar direction 和参数路由会引入额外尺度因素，所以需要专门验证。

### 5. muP 如何帮助 Muon 实验？

Muon 论文若想用小模型预测大模型，必须减少“宽度变化导致的超参数漂移”。muP 可以提供更稳的小模型 proxy，让你更有信心地比较不同优化器在规模上的趋势。

但 muP 不能替代 scaling sweep。它降低不确定性，不消除架构、数据、batch、数值实现和分布式系统带来的变化。

### 6. 公平比较要注意什么？

公平的 Muon vs AdamW 比较最好说明：两者是否都用 muP；两者是否分别调学习率；Muon 的 update scaling 是否与 muP 规则一致；embedding、output head、norm 参数是否用相同路由；小模型上调出的超参数是否在大模型上验证过。

## 和 Muon 的关系

Muon 的核心也关心“更新尺度”与矩阵形状，因此和 muP 的相互作用很重要。若用小模型 proxy 评估 Muon，再把学习率和缩放规则迁移到大模型，muP 可以帮助降低宽度变化带来的混杂因素。但 Muon 的正交化、update RMS、参数路由可能引入额外尺度约定，不能假设 AdamW 的 muP 配方原样适用。

## 需要掌握到什么程度

- 能说明 muP 解决的是跨宽度超参数迁移问题。
- 能区分 base shapes、宽度缩放和普通超参数扫描。
- 能理解 muP 不是自动保证跨架构、跨优化器全部可迁移。
- 能在读 Muon 实验时确认是否使用 muP，以及 AdamW/Muon 是否公平调参。

## 常见误区

- 把 muP 当成“更好的初始化”或“新优化器”。它主要是参数化和缩放规则。
- 认为用了 muP 就不需要大模型验证。muP 减少失真，不消灭所有规模效应。
- 忽略 Muon 自身 update scaling 与 muP 学习率缩放之间可能互相影响。

## 自测问题

1. 为什么同一个学习率在不同宽度模型上可能不等价？
2. muP 试图保持哪类“更新行为”跨宽度一致？
3. 如果 Muon 和 AdamW 使用不同参数化或缩放规则，公平性会受到什么影响？

## 参考入口

- Yang et al., *Tensor Programs* 系列论文。
- Yang & Hu, *Feature Learning in Infinite-Width Neural Networks*。
- *Practical Efficiency of Muon for Pretraining* 中关于 muP 的实验设置。
