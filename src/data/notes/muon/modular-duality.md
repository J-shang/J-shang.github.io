---
title: "Modular Duality in Deep Learning"
description: "理解梯度为何是 dual object，以及矩阵更新方向如何由 dualization 得到。"
topic: "muon"
section: "papers"
slug: "modular-duality"
legacyPaths: ["/notes/modular-duality/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 66
source:
  repository: "J-shang/Muon"
  path: "论文精读/05-Modular-Duality.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/05-Modular-Duality.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:229a019cefa0477ba74c5bc4ef5fef1163c9a7a6cf4ab2415bf996b35e85cb75"
  manifest: "muon"
  managed: true
---
> 原文：[arXiv:2410.21265](https://arxiv.org/abs/2410.21265)，核验版本 v2（2024-12-06）
> 来源类型：理论预印本
> 阅读提醒：形式关系在论文定义下成立；与 Muon 的联系包含后见的跨论文比较。

## 它解决什么问题

反向传播给出的 gradient 不是一个不带几何的“更新数组”。它是作用在参数扰动上的线性泛函：

$$
\mathrm{d}L_W[\Delta W]=\langle \nabla_WL,\Delta W\rangle.
$$

因此 gradient 自然位于参数空间的 dual；要得到可执行 update，需要结合参数/模块的 norm，把 dual object 映回 primal object。论文把这个过程称为 modular dualization，并让网络的组合结构参与定义几何。

## 从 linear layer 到 matrix dualization

对 linear map $x\mapsto Wx$，若参数扰动用 spectral/operator norm 度量，则其 dual norm 是 nuclear norm。给定 gradient $G=U\Sigma V^\top$，unit spectral ball 上最优更新方向为

$$
\mathcal{D}(G)=-UV^\top.
$$

这与 exact-polar Muon 的核心方向相同。论文进一步强调，linear、convolution、embedding、normalization 等模块的自然范数不同，完整网络的 dualization 应尊重模块角色，而不是把所有参数 flatten 后使用一个 Euclidean norm。

## 可核查锚点：尺度不变性与它的代价

对任意 $c>0$，

$$
\operatorname{polar}(cG)=\operatorname{polar}(G).
$$

证明只需把 $c$ 吸收到奇异值：$cG=U(c\Sigma)V^\top$。这说明 exact polar 忽略 gradient 的整体幅度。它带来归一化，也意味着：

- batch/noise 导致的幅度信息不会直接进入方向尺度；
- learning rate 和外部 shape scale 承担更大责任；
- $G\approx0$ 或秩变化时，方向连续性和数值近似需要单独处理。

这条性质是一个比“正交化更稳定”更精确的机制陈述。

## 与 modular norm 论文的分工

- *Scalable Optimization in the Modular Norm* 更侧重完整架构的自然 norm、Lipschitz 控制与 LR transfer。
- *Modular Duality* 更直接强调从 dual gradient 到 primal update 的映射，以及不同 layer role 的 optimizer 方向。
- Muon 是其中 linear/matrix dualization 的一个实际近似路径，但完整 Muon recipe 还包含 momentum、有限步 NS、scale、decay 和 routing。

## 边界与冲突定位

若有人说“Muon 是一阶法”，他可能按信息来源分类：只使用 gradient/momentum。若另一个人说“Muon 类似二阶法”，他可能按非对角矩阵变换分类。第一处分歧是“二阶”的定义，不是公式本身。判别问题应是：算法是否估计 Hessian/Fisher/gradient covariance 的历史状态？标准 Muon 通常不估计。

## 精读后的任务

对 Transformer 的 Q projection、embedding、LayerNorm scale 各写一个 dual-to-primal map 候选，并解释为什么不能仅凭 tensor rank 复用同一个 norm。把无法确定的模块标为 open，不要强行统一。

## 自测

1. 为什么反向传播的输出不能在没有 norm 的情况下唯一决定“最陡”更新？
2. `polar(cG)=polar(G)` 对 LR 调参和噪声有什么含义？
3. modular dualization 与实际 Muon 之间还差哪些算法步骤？

**掌握标准**：能从微分泛函出发解释 dual/primal，而不是把 duality 当作 SVD 的另一个名字。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **论文提出的是 type system 候选**：§6.1 主张 activation space 应带 intended norm/size、gradient 显式标成 dual vector，duality map 负责把 dual type 翻成 primal type。
2. **不仅处理 Linear**：正文给出 Linear、Embedding、Conv2D 的 module attributes 和 dualization；Conv 的 norm/reshape 选择不是“任意 flatten”。
3. **给出三类 GPU-friendly dualization 路径**：sketching、inverse matrix roots、rectangular Newton–Schulz，并指出优劣依赖 condition number 和资源。
4. **低秩边界是 NS 的卖点**：§5.3 说明 rectangular NS 不显式求 inverse root，低秩时仍 well-behaved；这是实现选择的重要数值点。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| Muon 原始说明的 tuned quintic | 本文给出一般/经典 rectangular NS 与可调 polynomial 家族 | **实现实例化，不冲突** | 对齐初始化缩放和 polynomial 后比较 scalar map |
| Scion 的 LMO | sharp/duality map 带 gradient magnitude；LMO 固定在 norm ball、scale-invariant | **尺度定义差异** | 检查是否显式计算 dual norm |
| 原始 Muon 对 output head 用 AdamW | modular/type 理论没有直接推出所有 empirical routing | **覆盖边界**，博客也承认 output-head 规则主要来自经验 | role-specific routing ablation |
| “Muon 是论文直接提出的算法” | 正文没有使用 Muon 名称，只提 Newton–Schulz duality map 的 speedrun | **历史归因纠正** | 核对时间线和术语出现位置 |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| gradient 是 dual vector，需通过 layer/module norm dualize 成 primal update | Abstract、§2–4、§6.1 | 论文明确 |
| Linear 的 spectral/operator geometry 产生 SVD/polar 型 duality map | Linear module 和 dualization 章节 | 论文明确 |
| rectangular NS 是 GPU-friendly approximation | §5.3 | 论文明确 |
| $\operatorname{polar}(cG)=\operatorname{polar}(G)$ | 可由论文的 SVD 与 normalization 公式推出 | 本文精确推导；不是论文单独主张的训练结论 |
| “与 exact-polar Muon 核心方向相同” | 论文未使用 Muon 名称 | 后见的跨论文比较；公式一致，但历史表述已纠正 |
| 对 Q projection、embedding、LayerNorm 候选 map 的任务 | 论文 type-system 思路启发 | 本文学习任务与开放设计，不是作者给出的完整 Transformer recipe |
