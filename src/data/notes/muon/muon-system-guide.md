---
title: "Muon 优化器系统学习指南"
description: "从更新几何、Newton–Schulz 数值近似到分布式实现，建立一套可验证的 Muon 心智模型。"
topic: "muon"
section: "guide"
slug: "muon-system-guide"
legacyPaths: ["/notes/muon-system-guide/"]
date: 2026-07-06
updated: 2026-07-14
cutoff: 2026-07-01
featured: true
order: 0
readtime: 30
source:
  repository: "J-shang/Muon"
  path: "MUON_LEARNING_GUIDE.md"
  url: "https://github.com/J-shang/Muon/blob/65164a375bd729b71f0e89b03642c67c50e624b3/MUON_LEARNING_GUIDE.md"
  revision: "65164a375bd729b71f0e89b03642c67c50e624b3"
  syncedAt: "2026-07-14"
  contentHash: "sha256:9d5b5a21c498d2f23a41aca388332002a6b9f3218d5b5b634ae30d515c6b41d1"
  manifest: "muon"
  managed: true
---
> 范围：深度学习优化器 **Muon（MomentUm Orthogonalized by Newton–Schulz）**，不是粒子物理中的 μ 子。
> 信息截点：**2026-07-14**。
> 学习对象：算法几何、有限精度实现、参数路由、分布式系统和公平实验；不把未经复现的前沿变体当作稳定配方。

## 0. 一页结论

Muon 的核心不是修改 Adam 的一个超参数，而是改变隐藏层矩阵参数的更新几何：

1. 对梯度形成 SGD momentum 或 Nesterov-style momentum；
2. 对这个语义矩阵做有限步 Newton–Schulz（NS）谱变换，近似其 polar factor；
3. 按矩阵形状缩放 update，并施加解耦 weight decay；
4. embedding、LM head、bias、norm scale 等通常仍交给 AdamW 或其他 scalar optimizer。

若动量矩阵的薄 SVD 为

$$
M=U\Sigma V^\top,
$$

理想方向是

$$
O=UV^\top.
$$

这个变换保留左右奇异向量，并把非零奇异值压到 1。它可以被严格地理解为谱范数单位球上线性化目标的最陡下降方向；也可与去掉历史统计的 Shampoo 建立代数联系。后者是带条件的特例/框架解释，不表示两者训练动力学等价。

截至信息截点，最稳妥的判断是：

- 大模型 **从头预训练** 是证据最强的应用范围；“约 2×”只是在特定 compute-optimal scaling-law 设置中的论文报告。
- update scaling、weight decay、参数路由、NS 精度与分布式矩阵语义都属于算法合同，不是无关紧要的工程细节。
- 有限步低精度 NS 是近似；数值上更接近 $UV^\top$ 不自动推出固定训练预算下 loss 更好。
- Muon 的持久状态通常比 AdamW 少，但 NS 临时 buffer、通信和 scalar AdamW 组仍要计入峰值内存与 wall-clock。
- AdamW 预训练 checkpoint 切换 Muon 做全量微调存在 optimizer mismatch 风险；SFT、RL、LoRA 和 continuation 必须分开验证。

当前最重要的学习产出不是背诵方法名，而是能够：推导方向、实现数值对照、追踪真实代码的 shape/state flow，并设计同时报告 token/FLOP/time/memory 的公平实验。

## 1. 证据和关系语言

本项目把两件事分开记录：

- **来源类别**：定理/教材、官方代码或文档、大规模技术报告、多尺度论文、早期预印本、博客、仓库内实验。
- **结论置信度**：`verified`、`supported`、`plausible`、`open`。

一手预印本仍然是预印本；“作者报告”不自动成为跨架构事实。数学与知识地图关系使用以下强度：exact identity、special case、approximation、generalization、implementation、analogy、empirical association。

完整的 claim ledger、阅读入口和证据缺口见 [论文与证据索引](/topics/muon/muon-evidence-index/)。

## 2. 必备知识地图

| 层次 | 核心概念 | 最小学习产出 |
|---|---|---|
| 优化基础 | SGD、momentum、Nesterov、AdamW、解耦 weight decay | 手写更新顺序，区分 gradient、update、decay 与 optimizer state |
| 线性代数 | SVD、谱/Frobenius/核范数、极分解、半正交矩阵 | 从薄 SVD 推出 polar factor 和谱范数 steepest direction |
| 数值计算 | Newton–Schulz、条件数、低精度矩阵乘 | 比较 SVD polar 与不同 steps/dtype 的 NS 误差和成本 |
| 深度学习工程 | mixed precision、state、ZeRO/FSDP、TP、Megatron | 画出 global/local shape、state、通信和参数路由 |
| LLM 实验方法 | scaling law、critical batch、muP、公平比较 | 写出带调参预算和失败判据的 AdamW/Muon 实验卡 |

逐篇笔记、阅读顺序、通过检查和 typed relations 见 [必备知识地图](/topics/muon/)。

## 3. 核心算法合同

### 3.1 momentum 的对象和时序

对语义矩阵参数 $W_t\in\mathbb{R}^{m\times n}$，最终累积并同步后的梯度记为 $G_t$：

$$
M_t=\beta M_{t-1}+G_t.
$$

若启用 Nesterov，常见实现再构造

$$
\widetilde M_t=G_t+\beta M_t.
$$

不同库的 momentum 归一化和更新顺序并不统一。比较实现时必须固定公式；对每个 micro-batch 分别正交化后相加，一般不等价于对最终 $G_t$ 做一次正交化。

### 3.2 精确 polar 与有限步 NS

精确 polar 方向为 $UV^\top$。生产实现通常先缩放：

$$
X_0=\frac{M}{\|M\|_F+\varepsilon},
$$

再执行少量多项式迭代：

$$
\begin{aligned}
A_k&=X_kX_k^\top,\\
X_{k+1}&=aX_k+bA_kX_k+cA_k^2X_k.
\end{aligned}
$$

原始常见 quintic 系数为

$$
(a,b,c)=(3.4445,-4.7750,2.0315),
\qquad N_{NS}=5.
$$

它追求 BF16/GPU 上快速得到有用的谱变换，不是高精度矩阵分解。假设、奇异值映射、手算例子和诊断见 [Newton–Schulz 迭代](/topics/muon/newton-schulz/)。

### 3.3 shape-aware update scale

对满秩 $m\times n$ polar factor，非零奇异值为 1，因此

$$
\|O\|_F^2=\min(m,n),
$$

从而元素 RMS 为

$$
\operatorname{RMS}(O)
=\frac{\|O\|_F}{\sqrt{mn}}
=\frac{1}{\sqrt{\max(m,n)}}.
$$

这说明统一学习率下，不同 shape 的原始 polar update 尺度天然不同。常见 recipe 会使用 spectral/original scale、unit-RMS 或 match-RMS 约定；它们不是同一公式的不同名字。完整更新可概括为

$$
W_{t+1}=(1-\eta\lambda)W_t-\eta\,s(m,n)\,O_t.
$$

其中 $s(m,n)$ 的定义、矩阵朝向和是否使用 global shape 必须跟实现一同记录。

### 3.4 参数路由

推荐起始合同：

| 参数角色 | 默认 optimizer | 必查边界 |
|---|---|---|
| attention Q/K/V/O projection | Muon | fused QKV 是否按语义块拆分 |
| MLP up/gate/down | Muon | up/down shape 与 scale |
| MoE expert matrices | Muon | expert/TP/EP group 与 batched NS |
| embedding、LM head | AdamW | tied weight 不能重复持有 |
| bias、norm scale | AdamW/其他 scalar optimizer | decay 规则与 state dtype |
| router/gate、卷积、高阶 tensor | recipe-specific | reshape/路由本身定义了优化几何 |

`p.ndim == 2` 只是候选条件，不是充分语义。生产代码应断言每个 trainable parameter 恰好属于一个 optimizer group。

## 4. 不能混淆的邻近概念

| 常见说法 | 更准确的关系 |
|---|---|
| Muon 把权重正交化 | 错；标准 Muon 近似正交化的是 momentum/update matrix |
| Muon 就是 Shampoo | 错；存在带条件的 instantaneous algebraic view，但 Shampoo 累积历史二阶统计 |
| 有限步 NS 等于 polar decomposition | 错；它是依赖谱、缩放、系数、steps 和 dtype 的 approximation |
| shard 上分别做 polar 等于 global polar | 一般错；blockwise 必须标成算法近似 |
| 状态少一半所以墙钟更快 | 不成立；还要计 NS、临时 buffer、通信与 kernel 利用率 |
| loss-vs-step 更快就是更高效 | 不成立；至少拆成 token、FLOP、time 和 memory |
| 使用 muP 就能跨规模无损迁移 | 不成立；它支持宽度迁移，但不覆盖架构、optimizer、batch 和数值实现变化 |

## 5. 学习与实现路线

### 阶段 A：约束驱动的数学推导

输入：SVD、对偶范数、极分解。

输出：从

$$
\arg\min_{\|\Delta\|_2\le1}\langle G,\Delta\rangle
$$

推出方向 $-UV^\top$，并写出秩亏/非方阵边界。

检查：能解释 spectral/Frobenius/nuclear norm 分别扮演什么角色，而不借助“像二阶法”类比。

### 阶段 B：数值实现

输入：polar target、NS 多项式、dtype。

输出：完成 [实验 0](/topics/muon/muon-experiment-roadmap/#实验-0数学与数值合同)。

检查：方/高/宽/近秩亏矩阵均报告 singular values、orthogonality error、cosine、RMS、time 和 memory。

### 阶段 C：真实代码走读

输入：固定 commit 的实现。

输出：完成 [实现与代码走读](/topics/muon/muon-code-reading/) 中的 shape/state/control-flow 图和最小验证套件。

检查：能指出 momentum、QKV split、scale、TP mode、owner rank 和 parameter sync 的准确分叉点。

### 阶段 D：公平训练实验

输入：tuned AdamW、明确 Muon recipe、相同数据与预算。

输出：完成 [小模型 A/B、scale/batch sweep 和分布式实验](/topics/muon/muon-experiment-roadmap/)。

检查：区分 drop-in 迁移公平与各自调优后的上限公平，并同时画 loss-vs-token/FLOP/time。

### 阶段 E：前沿研究

输入：已验证的主线和可复现实验底座。

输出：从 [前沿变体与开放问题](/topics/muon/muon-frontiers/) 选择一个区分性问题。

检查：先写被改变的对象、操作、state、实现路径和失败模式，再决定是否值得复现。

## 6. 专题索引

- [必备知识地图](/topics/muon/)：概念笔记、typed relations、学习产出。
- [Muon 论文精读](/topics/muon/muon-paper-reading-guide/)：按历史谱系、规模化实证、理论/数值与系统四层组织的一手来源；核心来源一篇一文件。
- [论文与证据索引](/topics/muon/muon-evidence-index/)：来源类别、置信度、claim ledger、阅读路径。
- [实现与代码走读](/topics/muon/muon-code-reading/)：固定版本、shape/state flow、实现不变量和测试。
- [Megatron-LM Muon 实现解析](/topics/muon/megatron-muon-implementation/)：本地 submodule 的逐函数走读。
- [前沿变体与开放问题](/topics/muon/muon-frontiers/)：机制族、争议的第一处分歧和判别实验。
- [实验路线与记录规范](/topics/muon/muon-experiment-roadmap/)：可复现实验卡、指标和完成定义。

## 7. 核心来源和实现入口

| 入口 | 来源类别 | 读取目的 |
|---|---|---|
| [Muon 原始说明](https://kellerjordan.github.io/posts/muon/) | 作者设计说明 | 最小算法、NS 与设计史 |
| [Old Optimizer, New Norm](https://arxiv.org/abs/2409.20325) | 理论预印本 | norm/dual norm 统一视角 |
| [Shampoo](https://proceedings.mlr.press/v80/gupta18a.html) | 同行评审论文 | 矩阵预条件与状态成本 |
| [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) | 大规模技术预印本 | WD、scale、Moonlight、distributed Muon |
| [Practical Efficiency of Muon](https://arxiv.org/abs/2505.02222) | 多尺度实验预印本 | muP、large batch、compute-time frontier |
| [KellerJordan/Muon](https://github.com/KellerJordan/Muon) | 官方参考实现 | 最小 PyTorch 代码 |
| [`torch.optim.Muon`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) | 官方框架文档 | 当前 API、参数限制与默认值 |
| [Moonlight](https://github.com/MoonshotAI/Moonlight) | 官方规模化实现 | 状态分片、通信、完整 recipe |
| [NVIDIA Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) | 实验性官方仓库 | Megatron 依赖、系数和 GPU 实现 |

## 8. 术语速查

- **polar factor**：薄 SVD $M=U\Sigma V^\top$ 中的半正交方向 $UV^\top$。
- **semi-orthogonal**：非方阵只能满足 $Q^\top Q=I$ 或 $QQ^\top=I$ 中与短边对应的一式。
- **spectral norm**：最大奇异值；其对偶范数是 nuclear norm。
- **update RMS**：update 元素平方均值的平方根，不等于 grad norm。
- **orthogonalization quality**：需要声明对象，可用正交误差、奇异值分布或 polar cosine 衡量。
- **optimizer state**：跨 step 持久保存的历史；不要和 NS 临时 buffer 混为一谈。
- **critical batch size**：给定模型、数据、训练阶段和 optimizer 下，继续增大 batch 的边际优化收益开始明显下降的区域，不是普适常数。
- **optimizer mismatch**：训练阶段切换 optimizer 几何/隐式偏置后，已有表示与新 update 不匹配的风险。

## 9. 维护与下一步

- 顶层指南只保留稳定结论、核心合同、路线图和链接；大段论文、实现、实验或前沿内容进入专题文件。
- 时间敏感来源按绝对日期更新。先更新专题的 claim ledger，再决定是否提升顶层结论。
- 新知识地图边必须标 relation type 和条件；新概念笔记必须包含可核查锚点与推理型自测。
- 默认不提交 checkpoint、数据集、下载论文、缓存或完整训练输出。

推荐下一步：先完成实验 0 的小型数值脚本和测试，再进入任何训练 A/B。它成本最低，却能最快暴露 transpose、shape、dtype、scale 和“近似等于精确”的理解错误。
