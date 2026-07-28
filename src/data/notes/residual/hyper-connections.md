---
title: "Hyper-Connections：把固定 residual path 扩展为可学习的多流状态系统"
description: "从 read、mix、write 三条路径复原 HC 的多流 residual state，核查初始化、深层组合、实验收益与资源成本。"
topic: "residual"
section: "methods"
slug: "hyper-connections"
date: 2026-07-27
updated: 2026-07-27
cutoff: 2026-07-23
featured: true
order: 10
readtime: 34
source:
  repository: "J-shang/residual"
  path: "papers/01-hyper-connections.md"
  url: "https://github.com/J-shang/residual/blob/c55583707ab645bef7408c588fc33cbf91b809a5/papers/01-hyper-connections.md"
  revision: "c55583707ab645bef7408c588fc33cbf91b809a5"
  syncedAt: "2026-07-28"
  contentHash: "sha256:7964725c8282d995d630bb162809035c6bb3177c62a939c8409cb5780cecdc7c"
  manifest: "residual"
  managed: true
---
<!-- paper-order: 01 -->

> **论文**：Defa Zhu et al., *Hyper-Connections*<br>
> **版本**：arXiv:2409.19606v3，2025-03-18；ICLR 2025 Poster<br>
> **主来源**：[arXiv 摘要](https://arxiv.org/abs/2409.19606v3) · [PDF](https://arxiv.org/pdf/2409.19606v3) · [OpenReview](https://openreview.net/forum?id=9FqARW7dwB)<br>
> **阅读范围**：正文与 Appendix A–L，包括算法伪代码、资源消耗、视觉实验、展开推导和 $n=1$ 讨论<br>
> **信息截止日期**：2026-07-23<br>
> **本文定位**：先忠实复原论文，再给出可验证的推导、工程判断和研究延伸

## 证据标签

- **[论文事实]**：论文正文、表格、图或附录直接报告的内容。
- **[复原推导]**：由论文公式推导出的结论，不等同于作者原话。
- **[综合判断]**：结合结果、实现和系统约束形成的解释。
- **[后续工作]**：论文发表之后出现的相关证据。
- **[开放问题]**：当前证据不足，仍需实验或实现验证。

---

## 先给结论

### 一句话

Hyper-Connections（HC）把 Transformer 中单一、固定的 residual stream，推广为 $n$ 条并行的 residual streams，并通过可学习矩阵决定每层“从哪些流读取、旧状态如何跨流传递、层输出写回哪些流”。

### 这篇论文真正贡献了什么

1. **把 residual connection 从固定加法改写成可学习的状态路由。**
2. **用一个统一矩阵同时表达深度方向和宽度方向的连接。**
3. **提出静态 SHC 与 token-dependent 的动态 DHC。**
4. **给出与普通 PreNorm Transformer 兼容的初始化，使训练能从熟悉的计算图出发。**
5. **在 1B、7B dense LLM、7B MoE、DiT 和 ViT 上报告了一致收益。**
6. **观察到训练后连接呈现“近期层 + 底部锚点”的 $\Lambda$-形结构，为后续稀疏化和稳定化提供线索。**

### 最重要的保留意见

- 论文证明了“可学习 residual topology 值得做”，但没有证明原始 HC 已经是稳定、可扩展的最终形式。
- 参数量和 FLOPs 增幅确实很小；训练显存却增加约 $9.7\%$–$28.3\%$，不能笼统称为“几乎无成本”。
- “1.8× faster convergence”按达到相同 loss 所需 token 数衡量，不是 wall-clock throughput 加速。
- 大模型实验没有多随机种子、误差条或显著性检验；机制可视化也是相关性证据，不是因果证明。
- 论文没有完整官方训练仓库，只有附录中的 PyTorch 风格伪代码；分布式布局和性能仍需独立复现。

---

## 阅读前：符号与约定

后文会在公式首次出现时再次解释新符号；本表先给出贯穿全文的统一词典。论文原文会在不同段落省略层号或改变矩阵朝向，本文固定采用“**旧 stream 是 source，新 stream 是 target**”的语义。

| 符号 | 含义 | 类型或 shape | 作用域 | 身份 |
|---|---|---:|---|---|
| $k,\ell$ | Transformer 子层编号；两者都表示 depth index | 整数 | $0,\ldots,L-1$ | 索引；$\ell$ 用于 DHC 小节以免与其他角标混淆 |
| $L$ | Attention/FFN 子层总数 | 正整数 | 全模型 | 固定超参数 |
| $N$ / $B_{\mathrm{batch}}$ | batch size | 正整数 | 一个 microbatch/batch | 运行配置；后文系统公式有时简写为 $B$ |
| $T$ / $S$ | sequence length | 正整数 | 一个 batch | 运行配置；后文系统公式有时简写为 $S$ |
| $n$ | residual stream 数，也叫 expansion rate | 正整数 | 通常全模型固定 | 架构超参数 |
| $d$ | 每条 stream 的 hidden width，即 `d_model` | 正整数 | 全模型或当前 stage | 架构超参数 |
| $x_k$ | 普通单流 residual state | $[d]$ 或 $[N,T,d]$ | 第 $k$ 个子层入口 | 运行时张量 |
| $H_k$ | HC 的全部 residual streams | $[n,d]$ 或 $[N,T,n,d]$ | 第 $k$ 个状态边界 | 运行时张量，不是 optimizer 参数 |
| $h_k^{(i)}$ | $H_k$ 的第 $i$ 条 stream | $[d]$ | $i=1,\ldots,n$ | 运行时张量；上标 $(i)$ 不是幂 |
| $\mathcal T_k$ | 第 $k$ 个 Attention 或 FFN branch | $[d]\to[d]$ | 每个子层一份 | 含普通 Transformer 可训练参数 |
| $\operatorname{Norm}$ | branch 的 PreNorm；$\operatorname{Norm}_{\mathrm{route}}$ 专用于生成 DHC 路由 | 保持输入 shape | 每 token | 可微算子；affine scale 若启用则可训练 |
| $o_k$ | $\mathcal T_k$ 产生的 branch output | $[d]$ 或 $[N,T,d]$ | 每个子层、token | 运行时张量 |
| $A_{m,k}$ | 从 $n$ 条 streams 读出 branch input 的权重 | $[n,1]$ | 每层；DHC 中每 token 可不同 | SHC 中是参数；DHC 中是运行时 mapping |
| $A_{r,k}$ | 旧 streams 到新 streams 的 carry/mixing 权重 | $[n,n]$ | 每层；DHC 中每 token 可不同 | SHC 中是参数；DHC 中是运行时 mapping |
| $B_k$ | branch output 写回 $n$ 条 streams 的权重 | $[1,n]$ | 每层；DHC 中每 token 可不同 | SHC 中是参数；DHC 中是运行时 mapping；不是 batch size |
| $A^{\mathrm{base}}_{m,\ell},A^{\mathrm{base}}_{r,\ell},B^{\mathrm{base}}_\ell$ | DHC 的输入无关基础拓扑 | 分别为 $[n,1],[n,n],[1,n]$ | 每层 | optimizer 直接更新的参数 |
| $W_{m,\ell},W_{r,\ell},W_{\beta,\ell}$ | 从 normalized streams 生成动态路由增量的投影 | 分别为 $[d,1],[d,n],[d,1]$ | 每层 | optimizer 直接更新的参数 |
| $s_{\alpha,\ell},s_{\beta,\ell}$ | 动态路由增量的幅度 | 标量 | 每层 | optimizer 直接更新的参数 |
| $I_n,\mathbf1,e_i$ | identity、全 1 对象、one-hot basis | 由上下文确定 | 初始化/推导 | 固定数学对象 |
| $\tanh$ | 对动态路由增量逐元素施加的双曲正切 | $\mathbb R\to(-1,1)$ | 每个路由 logit | 固定可微算子，不是参数 |
| $\prod_t A_{r,t}$ | 按 depth 顺序复合的 residual transport | $[n,n]$ | 多层展开 | 推导出的矩阵乘积 |

最重要的身份区别是：在 DHC 中，$A_m(H)$、$A_r(H)$、$B(H)$ 虽然参与反向传播，却是本次输入生成的中间张量；optimizer 实际更新的是它们背后的 base matrices、$W$ 投影、尺度参数以及 Transformer 主干参数。

---

## 1. 论文要解决的矛盾

标准 residual connection 的核心价值是保留 identity path：

1. 深层网络中的信息可以不经复杂变换直接传播；
2. 梯度拥有短路径，优化更稳定；
3. 新增 block 只需学习对已有表示的增量修正。

以 PreNorm Transformer 为例：

$$
x_{k+1}
=
x_k + \mathcal{T}_k\!\left(\operatorname{Norm}(x_k)\right),
$$

其中 $k$ 是子层编号，$x_k\in\mathbb{R}^{d}$ 是进入该子层的运行时 residual state，$\operatorname{Norm}$ 是 PreNorm，$\mathcal T_k:\mathbb R^d\to\mathbb R^d$ 是第 $k$ 个 Attention 或 FFN branch。$\mathcal T_k$ 内的权重由主任务 loss 正常训练；$x_k$ 本身是 activation，不是 optimizer 参数。

这个结构稳定，但也非常刚性：

- 每层只能读取当前唯一的 residual state；
- 每层输出只能写回同一条流；
- 所有历史信息都在一个向量里反复叠加；
- 深度方向的连接拓扑由人工预先固定。

**[论文事实]** HC 的出发点是：既然线性层、Attention、MoE routing 都可以学习，为什么 residual connection 本身不能学习？

---

## 2. 从单流到多流：最小直觉

普通 residual block 只保存一个状态 $x$。HC 保存 $n$ 个状态：

$$
H_k =
\begin{bmatrix}
h_k^{(1)}\\
h_k^{(2)}\\
\vdots\\
h_k^{(n)}
\end{bmatrix}
\in \mathbb{R}^{n\times d}.
$$

在批量序列实现中，典型 shape 是：

$$
H_k\in\mathbb{R}^{B\times S\times n\times d},
$$

其中 $B$ 是 batch size，$S$ 是 sequence length，$n$ 是 stream 数，$d$ 是 hidden size。

每个 HC-wrapped 子层要完成三件事：

1. **Read**：从 $n$ 条流组合出一个 $d$-维输入；
2. **Carry/Mix**：决定旧状态怎样保留和跨流混合；
3. **Write**：决定子层输出写回哪些流、写多少。

这使 residual connection 不再只是“加不加”，而成为一个小型、可学习的状态路由器。

---

## 3. 核心方法

### 3.1 先把容易混淆的角标拆开

**[论文事实]** 下面的符号对应论文 §2.1 的 Eq. 1–7、Fig. 2，以及 §2.2 的 Eq. 8–13。论文先用 $k$ 表示层号，随后为了简化公式又省略了层号；同时，$A_m$ 里的 $m$ 是名称的一部分而不是求和角标，因此很容易误读。

为避免 $B$ 同时表示 batch size 和写回矩阵，本节暂用：

$$
N=\text{batch size},
\qquad
T=\text{sequence length},
\qquad
n=\text{stream 数},
\qquad
d=\text{hidden size}.
$$

| 论文符号 | 类型或 shape | 应该怎样读 |
|---|---:|---|
| $k$ | 整数层号 | 第 $k$ 个 Attention 或 FFN 子层；**不是矩阵维度** |
| $i,j\in\{1,\ldots,n\}$ | stream 角标 | $i$ 是旧状态的 source stream，$j$ 是新状态的 target stream |
| $n$ | 正整数 | expansion rate，也就是 residual stream 数；`DHC×4` 就是 $n=4$ |
| $d$ | 正整数 | 每条 stream 中 hidden vector 的宽度，即常见的 `d_model` |
| $H_k$ | $[N,T,n,d]$ | 第 $k$ 个状态边界上的全部 residual streams |
| $h_k^{(i)}$ | $[d]$ | $H_k$ 的第 $i$ 条 stream；上标 $(i)$ 是 stream 编号，不是幂 |
| $B_k$ | $[1,n]$ | branch output 写回 $n$ 条 streams 的权重行；这里的 $B$ **不是 batch size** |
| $\beta_{k,j}$ | 标量 | $B_k$ 的第 $j$ 项：当前 branch output 写入 target stream $j$ 的系数 |
| $A_{m,k}$ | $[n,1]$ | 从 $n$ 条 streams 读出 branch input；$m$ 是功能标签，不是维度或游标 |
| $A_{r,k}$ | $[n,n]$ | 旧 residual states 的 carry/mixing 矩阵；$r$ 也是功能标签 |
| $\alpha_{k,i,0}$ | 标量 | source stream $i$ 读入当前 branch 的权重，是 $A_{m,k}$ 的元素 |
| $\alpha_{k,i,j}$ | 标量 | source stream $i$ 传到 target stream $j$ 的权重，是 $A_{r,k}$ 的元素 |

因此，像 $A_{m,k}$ 这样的写法不能理解成“第 $m$ 行、第 $k$ 列”：逗号右侧的 $k$ 才是层号，$m$ 只是说明这个矩阵负责生成当前 module/branch 的输入。论文没有正式展开字母 $m$ 的全称；把它记成 **merge/mix-to-module** 只是帮助记忆，不是新的数学定义。

论文 Fig. 2 和 Eq. 1 的逐元素写法是：

$$
\mathrm{HC}_k=
\begin{bmatrix}
0 & \beta_{k,1} & \cdots & \beta_{k,n}\\
\alpha_{k,1,0} & \alpha_{k,1,1} & \cdots & \alpha_{k,1,n}\\
\vdots & \vdots & \ddots & \vdots\\
\alpha_{k,n,0} & \alpha_{k,n,1} & \cdots & \alpha_{k,n,n}
\end{bmatrix}.
$$

把它看成“**行是 source，列是 target**”最不容易出错：

- 第 $0$ 行代表 branch output，$\beta_{k,j}$ 把它写到 target stream $j$；
- 第 $i$ 行、第 $0$ 列的 $\alpha_{k,i,0}$ 把 source stream $i$ 读入 branch；
- 第 $i$ 行、第 $j$ 列的 $\alpha_{k,i,j}$ 把旧 stream $i$ 传到新 stream $j$。

本文前后采用 $H_k\rightarrow H_{k+1}$ 表示一次更新；原论文正文采用“第 $k$ 层读取 $H_{k-1}$、产生 $H_k$”。两者只差一个整体平移，不影响公式，但阅读时必须固定一种 convention。

### 3.2 统一连接矩阵

**[论文事实]** 论文把一层的 Hyper-Connection 写为：

$$
\mathrm{HC}_k =
\begin{bmatrix}
0 & B_k\\
A_{m,k} & A_{r,k}
\end{bmatrix}
\in\mathbb{R}^{(n+1)\times(n+1)},
$$

其中：

- $B_k\in\mathbb{R}^{1\times n}$：branch output 写回各 stream 的权重；
- $A_{m,k}\in\mathbb{R}^{n\times1}$：多流状态读入当前 branch 的权重；
- $A_{r,k}\in\mathbb{R}^{n\times n}$：旧 residual states 的保留与跨流混合。

给定 $H_k\in\mathbb{R}^{n\times d}$，计算分为三步。

#### Read

$$
h_k^{\text{read}}
=
H_k^\top A_{m,k}
\in\mathbb{R}^{d}.
$$

#### Transform

$$
o_k
=
\mathcal{T}_k\!\left(
\operatorname{Norm}(h_k^{\text{read}})
\right)
\in\mathbb{R}^{d}.
$$

#### Carry + Write

$$
H_{k+1}
=
A_{r,k}^{\top}H_k
+
B_k^\top o_k^\top
\in\mathbb{R}^{n\times d}.
$$

第一项传播和混合旧状态，第二项把新 branch output 注入多条流。

> 在本文的 row-major contract 下，$A_r[i,j]$ 表示 source $i\rightarrow$ target $j$，所以更新时出现 $A_r^\top H$。如果代码采用另一种权重存储约定，转置位置会变化；语义不应变化。

### 3.3 一个 $n=2$ 的静态最小例子

令

$$
A_m =
\begin{bmatrix}1\\0\end{bmatrix},
\qquad
A_r = I_2,
\qquad
B =
\begin{bmatrix}1&0\end{bmatrix}.
$$

则

$$
h^{\text{read}} = h^{(1)},
$$

$$
H_{\text{new}}
=
\begin{bmatrix}
h^{(1)} + \mathcal{T}(h^{(1)})\\
h^{(2)}
\end{bmatrix}.
$$

这就是普通 residual update 被放进第一条 stream，第二条 stream 原样保存。

如果改成

$$
B=\begin{bmatrix}1&1\end{bmatrix},
$$

同一个 branch output 会同时写入两条流。如果 $A_r$ 出现非对角元素，旧状态也可以从一条 stream 转移到另一条。

### 3.4 论文所说的 depth 与 width connections

**[论文事实]**

- $B$ 与 $A_r$ 的对角项主要控制深度方向的信息传播；
- $A_m$ 与 $A_r$ 共同形成宽度方向的跨流组合。

这里不是互斥分解：$A_r$ 的对角与非对角部分承担不同角色，而整个 $A_r$ 又参与 stream mixing。

---

## 4. SHC 与 DHC

### 4.1 Static Hyper-Connections（SHC）

SHC 中的 $B_k$、$A_{m,k}$、$A_{r,k}$ 是每层可学习参数，但不随 token 内容变化。

特点：

- 同一层对所有 token 使用相同拓扑；
- 容易解释和可视化；
- 额外计算较少；
- 可能在训练后进行稀疏化或静态图重排。

### 4.2 Dynamic Hyper-Connections（DHC）

DHC 在静态基值上增加由当前状态生成的 token-dependent 修正。下面先给无歧义的 shape contract，再做一次完整手算。证据位置是论文 §2.2 的 Eq. 8–13，以及 Appendix J 的 Algorithm 2–3。

先明确参数身份和训练方式：

| 对象 | 是否由 optimizer 直接更新 | 梯度来源 |
|---|---|---|
| $B_\ell^{\mathrm{base}},A_{m,\ell}^{\mathrm{base}},A_{r,\ell}^{\mathrm{base}}$ | 是 | 主任务 loss 经 HC state update 反传 |
| $W_{\beta,\ell},W_{m,\ell},W_{r,\ell}$ | 是 | 主任务 loss 经 $\tanh$ 和动态 mapping 反传 |
| $s_{\beta,\ell},s_{\alpha,\ell}$ | 是 | 主任务 loss；控制动态增量幅度 |
| route norm 的 affine scale（若实现启用） | 是 | 主任务 loss |
| $B_{\ell,b,t}(H),A_{m,\ell,b,t}(H),A_{r,\ell,b,t}(H)$ | 否 | 它们是每次 forward 生成的运行时张量 |
| $H_\ell,h_\ell^{\mathrm{read}},o_\ell$ | 否 | activation；只在计算图中承接梯度 |

**[复原推导]** 论文没有为路由增加独立监督或 auxiliary loss；这些参数与 Transformer 主干一起由语言建模或视觉任务的主 loss 端到端训练。optimizer 分组上，论文的 static HC component 不使用 weight decay，dynamic component 使用 weight decay；这只是正则化配置不同，不代表使用不同目标函数（论文 §4 “Implementation”）。

为避免层号 $k$ 和 stream 角标混在一起，这一小节改用 $\ell$ 表示层号。对某个 batch $b$、token position $t$，先取：

$$
\bar H_{\ell,b,t}
=
\operatorname{Norm}_{\mathrm{route}}(H_{\ell,b,t})
\in\mathbb{R}^{n\times d}.
$$

这里的 `route norm` 是为了生成动态 routing weights；branch 自身的 PreNorm 仍会在读出 $h^{\mathrm{read}}$ 后再执行一次。

每个 DHC module 的静态基值和动态投影是：

$$
B_\ell^{\mathrm{base}}\in\mathbb{R}^{1\times n},
\quad
A_{m,\ell}^{\mathrm{base}}\in\mathbb{R}^{n\times1},
\quad
A_{r,\ell}^{\mathrm{base}}\in\mathbb{R}^{n\times n},
$$

$$
W_{\beta,\ell}\in\mathbb{R}^{d\times1},
\quad
W_{m,\ell}\in\mathbb{R}^{d\times1},
\quad
W_{r,\ell}\in\mathbb{R}^{d\times n},
\quad
s_{\beta,\ell},s_{\alpha,\ell}\in\mathbb{R}.
$$

因此每个 token 的三条动态生成路径是：

$$
\underbrace{\bar H_{\ell,b,t}}_{[n,d]}
\underbrace{W_{\beta,\ell}}_{[d,1]}
\rightarrow[n,1]
\xrightarrow{\text{transpose}}[1,n],
$$

$$
\underbrace{\bar H_{\ell,b,t}}_{[n,d]}
\underbrace{W_{m,\ell}}_{[d,1]}
\rightarrow[n,1],
$$

$$
\underbrace{\bar H_{\ell,b,t}}_{[n,d]}
\underbrace{W_{r,\ell}}_{[d,n]}
\rightarrow[n,n].
$$

加上静态基值后：

$$
B_{\ell,b,t}(H)
=
B_\ell^{\mathrm{base}}
+
s_{\beta,\ell}
\tanh\!\left(
\bar H_{\ell,b,t}W_{\beta,\ell}
\right)^\top
\in\mathbb{R}^{1\times n},
$$

$$
A_{m,\ell,b,t}(H)
=
A_{m,\ell}^{\mathrm{base}}
+
s_{\alpha,\ell}
\tanh\!\left(
\bar H_{\ell,b,t}W_{m,\ell}
\right)
\in\mathbb{R}^{n\times1},
$$

$$
A_{r,\ell,b,t}(H)
=
A_{r,\ell}^{\mathrm{base}}
+
s_{\alpha,\ell}
\tanh\!\left(
\bar H_{\ell,b,t}W_{r,\ell}
\right)
\in\mathbb{R}^{n\times n}.
$$

恢复完整 batch 和 sequence 维后，运行时 tensor shapes 是：

| Tensor | Shape |
|---|---:|
| $H_\ell,\bar H_\ell$ | $[N,T,n,d]$ |
| $B_\ell(H)$ | $[N,T,1,n]$ |
| $A_{m,\ell}(H)$ | $[N,T,n,1]$ |
| $A_{r,\ell}(H)$ | $[N,T,n,n]$ |
| $h_\ell^{\mathrm{read}}$ | $[N,T,d]$ |
| $o_\ell=\mathcal T_\ell(\operatorname{Norm}(h_\ell^{\mathrm{read}}))$ | $[N,T,d]$ |
| $H_{\ell+1}$ | $[N,T,n,d]$ |

逐元素地看，$\beta$ 和 $\alpha$ 都只是标量 routing coefficients：

$$
\beta_{\ell,b,t,j}(H)
=
\beta_{\ell,j}^{\mathrm{base}}
+
s_{\beta,\ell}
\tanh\!\left(
\bar h_{\ell,b,t}^{(j)}\cdot w_{\beta,\ell}
\right),
$$

$$
\alpha_{\ell,b,t,i,0}(H)
=
\alpha_{\ell,i,0}^{\mathrm{base}}
+
s_{\alpha,\ell}
\tanh\!\left(
\bar h_{\ell,b,t}^{(i)}\cdot w_{m,\ell}
\right),
$$

$$
\alpha_{\ell,b,t,i,j}(H)
=
\alpha_{\ell,i,j}^{\mathrm{base}}
+
s_{\alpha,\ell}
\tanh\!\left(
\bar h_{\ell,b,t}^{(i)}\cdot w_{r,\ell}^{(j)}
\right).
$$

这里 $w_{\beta,\ell},w_{m,\ell},w_{r,\ell}^{(j)}\in\mathbb{R}^{d}$。也就是说，DHC 不会用一个 $d\times d$ 大矩阵生成主干特征；它只把每条 $d$ 维 stream 压成少量 routing scalars。

#### 论文代码为什么只看到一个 `alpha`

Appendix J 的 Algorithm 2 把

$$
A_\ell
=
\left[A_{m,\ell}\;\;A_{r,\ell}\right]
\in\mathbb{R}^{n\times(n+1)}
$$

打包为 `static_alpha`，同时把

$$
W_{\alpha,\ell}
=
\left[W_{m,\ell}\;\;W_{r,\ell}\right]
\in\mathbb{R}^{d\times(n+1)}
$$

打包为 `dynamic_alpha_fn`。所以代码中的：

```text
norm_h:           [N, T, n, d]
dynamic_alpha_fn: [d, n+1]
alpha:            [N, T, n, n+1]
mix_h = alpha.transpose(-1, -2) @ h
mix_h:            [N, T, n+1, d]
```

`mix_h[..., 0, :]` 是读给 branch 的 $h^{\mathrm{read}}$，其余 $n$ 个 slots 是 $A_r^\top H$ 得到的 carry states。`beta:[N,T,n]` 再把 branch output 写回这些 carry states。

#### 一个完整的 DHC 手算：$N=T=1,\ n=d=2$

**[复原推导]** 省略长度为 $1$ 的 batch/token 维。令当前两条 stream 为：

$$
H=
\begin{bmatrix}
1&-1\\
-1&1
\end{bmatrix}
\in\mathbb{R}^{2\times2}.
$$

为方便手算，忽略 LayerNorm 的 $\epsilon$ 和 affine 参数；每行均值为 $0$、方差为 $1$，因此这里有 $\bar H=H$。

取静态基值为论文的 PreNorm-compatible 形式：

$$
B^{\mathrm{base}}=
\begin{bmatrix}1&1\end{bmatrix},
\qquad
A_m^{\mathrm{base}}=
\begin{bmatrix}1\\0\end{bmatrix},
\qquad
A_r^{\mathrm{base}}=
\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$$

为了让动态项可见，假设训练后投影参数为：

$$
W_\beta=
\begin{bmatrix}0.2\\-0.1\end{bmatrix},
\qquad
W_m=
\begin{bmatrix}-0.2\\0.1\end{bmatrix},
\qquad
W_r=
\begin{bmatrix}
0.1&-0.2\\
0.3&0
\end{bmatrix},
\qquad
s_\alpha=s_\beta=0.5.
$$

先检查三个矩阵乘法的维度和值：

$$
\underbrace{\bar H}_{[2,2]}
\underbrace{W_\beta}_{[2,1]}
=
\begin{bmatrix}0.3\\-0.3\end{bmatrix}_{[2,1]},
$$

$$
\underbrace{\bar H}_{[2,2]}
\underbrace{W_m}_{[2,1]}
=
\begin{bmatrix}-0.3\\0.3\end{bmatrix}_{[2,1]},
$$

$$
\underbrace{\bar H}_{[2,2]}
\underbrace{W_r}_{[2,2]}
=
\begin{bmatrix}
-0.2&-0.2\\
0.2&0.2
\end{bmatrix}_{[2,2]}.
$$

使用 $\tanh(0.3)\approx0.2913$、$\tanh(0.2)\approx0.1974$，得到这个 token 专属的 routing weights：

$$
B(H)
\approx
\begin{bmatrix}
1.1457&0.8543
\end{bmatrix}_{[1,2]},
$$

$$
A_m(H)
\approx
\begin{bmatrix}
0.8543\\
0.1457
\end{bmatrix}_{[2,1]},
$$

$$
A_r(H)
\approx
\begin{bmatrix}
0.9013&-0.0987\\
0.0987&1.0987
\end{bmatrix}_{[2,2]}.
$$

第一步，读出 branch input：

$$
\underbrace{h^{\mathrm{read}}}_{[2,1]}
=
\underbrace{H^\top}_{[2,2]}
\underbrace{A_m(H)}_{[2,1]}
\approx
\begin{bmatrix}
0.7087\\
-0.7087
\end{bmatrix}.
$$

这说明当前 branch 不是只读第一条流，而是读：

$$
0.8543\,h^{(1)}+0.1457\,h^{(2)}.
$$

第二步，假设经过 `Norm → Attention/FFN` 后，branch 输出为：

$$
o=
\begin{bmatrix}
0.5\\-0.25
\end{bmatrix}_{[2,1]}.
$$

第三步，分别计算 carry 和 write：

$$
\underbrace{A_r(H)^\top H}_{[2,2]}
\approx
\begin{bmatrix}
0.8026&-0.8026\\
-1.1974&1.1974
\end{bmatrix},
$$

$$
\underbrace{B(H)^\top o^\top}_{[2,2]}
\approx
\begin{bmatrix}
0.5728&-0.2864\\
0.4272&-0.2136
\end{bmatrix}.
$$

所以最终：

$$
H_{\mathrm{new}}
=
A_r(H)^\top H+B(H)^\top o^\top
\approx
\begin{bmatrix}
1.3755&-1.0890\\
-0.7702&0.9838
\end{bmatrix}_{[2,2]}.
$$

这个例子里三类系数的职责完全不同：

1. $A_m:[n,1]$ 在 **stream 维**提供 $n$ 个加权系数，把 $n$ 条 streams 加权求和成一个 $d$ 维 branch input；本例 $n=2$，所以 $A_m:[2,1]$；
2. $A_r:[n,n]$ 把 $n$ 条旧 streams 变成 $n$ 条新 carry streams；本例为 $[2,2]$；
3. $B:[1,n]$ 用 $n$ 个 $\beta_j$ 把同一个 branch output 分别写回 $n$ 条 streams；本例为 $[1,2]$。

这里并没有把 hidden width 从 $2$ 压成 $1$。被消去的是 stream 维 $n$，hidden width $d$ 保持不变：

$$
\underbrace{H^\top}_{[d,n]}
\underbrace{A_m}_{[n,1]}
\longrightarrow
\underbrace{h^{\mathrm{read}}}_{[d,1]}.
$$

若把 hidden vector 写成行向量，同一个运算也可写成：

$$
\underbrace{A_m^\top}_{[1,n]}
\underbrace{H}_{[n,d]}
\longrightarrow
\underbrace{(h^{\mathrm{read}})^\top}_{[1,d]}.
$$

本例恰好取了 $n=d=2$，所以两种不同含义的维度都显示成数字 $2$；这正是原句容易让人误以为 shape 写反的原因。

直观上：

- 静态矩阵提供全局默认拓扑；
- 动态项允许不同 token 采用不同的读、传、写策略；
- $\tanh$ 限制动态修正幅度，但论文消融显示它不是取得收益的必要条件。

**[综合判断]** DHC 类似一个沿“网络深度”运行的轻量 hypernetwork：它不生成主干大权重，而是生成 residual routing coefficients。

---

## 5. 初始化：如何退化回熟悉的 PreNorm

### 5.1 初始化规则

**[论文事实]** 对第 $k$ 个 HC 层，论文使用：

$$
B_k=\mathbf{1}_{1\times n},
$$

$$
A_{m,k}=e_{k\bmod n},
$$

$$
A_{r,k}=I_n,
$$

并把动态投影矩阵初始化为零：

$$
W_{\beta,k}=W_{m,k}=W_{r,k}=0.
$$

这里 $e_i$ 是 one-hot basis vector。初始化时：

- $A_r=I$：每条 stream 保持 identity；
- $A_m$ 轮流选择一条 stream 供当前 branch 读取；
- $B=\mathbf{1}$：branch output 写入所有 streams；
- 动态分支从零修正开始。

论文还把 Attention output projection 和 FFN 第二个线性层的初始化标准差乘以 $\sqrt n$，以补偿末端多流求和带来的尺度变化。

### 5.2 “普通 PreNorm 特例”应该怎样理解

**[论文事实]** 作者把上述初始化称为 standard PreNorm architecture 的特例。

**[复原推导]** 更精确地说，这种等价性依赖：

1. 指定的轮转读取与全流写入规则；
2. 最终把各 stream 求和；
3. branch output 的初始化尺度补偿；
4. Attention 与 FFN 都按论文方式包装。

因此验证代码时，不应只检查 $A_r=I$，而应比较完整网络在初始化时的端到端输出与梯度。

---

## 6. 一张实现 contract 表

| 项目 | HC 的定义 | 实现时要核对 |
|---|---|---|
| Stored state | $H\in\mathbb{R}^{B\times S\times n\times d}$ | stream 维位置、contiguous/layout、dtype |
| Read rule | $h^{read}=H^\top A_m$ | 输出必须回到 $[B,S,d]$ |
| Transform | $o=\mathcal T(\operatorname{Norm}(h^{read}))$ | Norm 在读出后还是逐 stream |
| Carry rule | $A_r^\top H$ | 左右乘方向、batch/token 广播 |
| Write rule | $B^\top o^\top$ | branch output 如何广播到 stream |
| State update | $H'=A_r^\top H+B^\top o^\top$ | 加法两项 shape/dtype 一致 |
| Weight granularity | SHC：per-layer；DHC：per-layer base + per-token delta | DHC 是否为每个 token 生成完整 $n\times n$ |
| Init special case | $B_k=\mathbf{1}_{1\times n}$<br>$A_{m,k}=e_{k\bmod n}$<br>$A_{r,k}=I_n$<br>$W_{\beta,k}=W_{m,k}=W_{r,k}=0$ | 与 PreNorm 端到端等价性 |
| Deep composition | 多层 $A_r$ 的矩阵乘积形成跨深度路径系数 | 顺序和转置不能错 |
| Final collapse | 最后一层后对 stream 求和，再做 final norm/unembedding | 不要提前 collapse |
| Primary cost | 多流 activation 与状态搬运 | FLOPs 小不代表带宽和显存小 |

---

## 7. 多层展开：HC 为什么不仅是“多复制几份 hidden state”

把多个 HC 层展开后，第 $j$ 层 branch output 对第 $k$ 层 branch input 的有效系数可写成：

$$
c_{k,j}^{(0)}
=
B_j
\left(
\prod_{t=j+1}^{k-1}A_{r,t}
\right)
A_{m,k}.
$$

具体转置方向随符号约定变化，但结构含义不变：

- $B_j$：第 $j$ 层把新信息注入 residual state 的方式；
- $\prod A_r$：信息在中间各层之间如何被保存、衰减和换流；
- $A_{m,k}$：第 $k$ 层最终怎样读取这些历史信息。

**[复原推导]** 这让 HC 可以被看成沿网络深度运行的线性状态系统：

$$
\text{state transition}=A_r,\qquad
\text{input injection}=B,\qquad
\text{readout}=A_m.
$$

非线性主干 $\mathcal T_k$ 产生新信息，HC 则控制这些信息在深度轴上的记忆与路由。

这个视角解释了为什么 $n>1$ 有意义：多维状态允许不同历史分量沿不同动力学传播，而单一 residual stream 只能让所有历史共享同一个标量式更新链。

---

## 8. 为什么 $n=1$ 不够

**[论文事实]** 论文报告 DHC×1 表现反而略低于普通 baseline，并在附录专门讨论该现象。

当 $n=1$ 时：

$$
H_k\in\mathbb{R}^{1\times d},
\quad
A_r\in\mathbb{R}^{1\times1}.
$$

所有历史 branch outputs 都通过同一串标量乘积传播。模型无法同时实现：

- 一部分信息长时间近似无损保留；
- 另一部分信息快速衰减；
- 某些 stream 专门保存早期 anchor；
- 某些 stream 专门服务于局部相邻层。

**[综合判断]** HC 的主要收益不是“把 residual coefficient 学出来”，而是把单一 memory channel 扩展为多维 memory state。只有 $n>1$ 才获得真正的状态分工能力。

---

## 9. 顺序结构与并行结构的统一

**[论文事实]** 论文展示：通过设定特殊的 HC 矩阵，若干 residual blocks 可以表达为顺序组合，也可以表达为共享输入的并行组合。

这揭示 HC 矩阵不仅控制“残差强弱”，还控制 block 之间的函数依赖图。

但需要区分两件事：

- **函数表达上的并行**：特定矩阵使两个 block 的输入不再相互依赖；
- **硬件执行上的并行**：实际 runtime 是否能并发调度、是否减少 latency。

**[综合判断]** 对 SHC，训练后若矩阵形成稳定、精确或可阈值化的模式，存在静态重排机会；对 token-dependent DHC，动态依赖通常更难直接转成固定执行图。论文没有报告 wall-clock 并行加速。

---

## 10. 实验结果

### 10.1 实验边界

**[论文事实]**

- Dense LLM：OLMo 风格 1B 与 7B 模型；
- MoE：约 7B 总参数、1.3B active parameters；
- 所有主要 LLM 训练使用 500B tokens；
- Dense 数据使用 dolma-v1.5-sample，MoE 使用 OLMOE-MIX；
- 还测试 DiT 图像生成与 ViT 图像分类。

因此下面的 `baseline` 特指同一表格内、论文给定模型配置、数据、训练 token budget 与评估任务下的原始架构。

### 10.2 1B Dense LLM

| 模型 | V2 loss | V2 PPL | V3 loss | V3 PPL | 下游平均 |
|---|---:|---:|---:|---:|---:|
| OLMo-1B baseline | 2.811 | 18.023 | 2.544 | 14.229 | 62.5 |
| SHC×4 | 2.791 | — | 2.528 | — | 63.6 |
| DHC×4（默认） | 2.781 | 17.509 | 2.514 | 13.826 | 63.8 |
| DHC×4（无 $\tanh$） | 2.779 | 17.451 | 2.516 | 13.844 | 64.4 |
| DHC×1 | 2.819 | — | — | — | 62.3 |

**读法：**

- DHC×4 相对 baseline 的语言模型 loss 与下游平均都有改善；
- DHC×1 不改善，支持“多流状态才是关键”；
- 去掉 $\tanh$ 后结果没有恶化，说明有界动态偏移不是核心贡献；
- 论文中更大的 $n$ 往往继续降低 loss，但下游收益会饱和甚至回落，$n$ 不是越大越好。

论文还比较了 ResiDual 与 AltUp。作者实现的 DHC×2 优于这些对照，但这些方法均由作者侧复现，调参公平性和实现成熟度仍需谨慎看待。

### 10.3 7B Dense LLM

| 模型 | 参数量 | FLOPs | V2 loss | V2 PPL | V3 loss | V3 PPL | 下游平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLMo-7B baseline | 6.9B | 13.36G | 2.581 | 14.316 | 2.322 | 11.324 | 70.1 |
| DHC×4 | 6.9B | 13.38G | 2.559 | 14.023 | 2.304 | 11.120 | 71.0 |

**[论文事实]** 在相同 500B-token 预算下，DHC×4 的参数规模基本不变，FLOPs 增幅很小，下游平均提高 0.9 个百分点。

### 10.4 7B MoE

| 任务 | baseline | DHC×4 |
|---|---:|---:|
| MMLU Var | 38.5 | 39.7 |
| HellaSwag | 69.5 | 70.2 |
| ARC-C | 41.8 | 47.8 |
| ARC-E | 72.8 | 76.7 |
| PIQA | 77.6 | 78.2 |
| WinoGrande | 64.4 | 64.6 |
| BoolQ | 65.4 | 68.5 |

论文还报告训练 loss 改善 0.027、C4 validation loss 改善 0.028。

**[论文事实]** “1.8× faster convergence”指 DHC 达到指定 loss 所需 token 数更少。

**[综合判断]** 这支持 sample efficiency 改善，但不能直接推出训练时间缩短 1.8 倍；wall-clock 还受 activation memory、kernel、通信和吞吐影响。

### 10.5 视觉任务

#### DiT 图像生成

| 模型 | 参数量 | FID（越低越好） |
|---|---:|---:|
| DiT baseline | 675M | 2.36 |
| DiT + SHC×2 | 675M 附近 | 2.18 |
| 更大 DiT baseline | 983M | 2.13 |

在论文设置下，较小的 SHC 模型取得接近更大 baseline 的 FID。

#### ViT 图像分类

| 模型 | baseline | SHC | DHC |
|---|---:|---:|---:|
| ViT-Base | 76.38 | 77.60 | 77.26 |
| ViT-Large | 77.25 | 78.38 | 79.94 |

论文把这些提升称为百分比改善，但表中数值直接相减更准确地说是：

- ViT-Base：SHC $+1.22$ 个百分点，DHC $+0.88$ 个百分点；
- ViT-Large：SHC $+1.13$ 个百分点，DHC $+2.69$ 个百分点。

---

## 11. 资源成本：算力小，状态成本不小

| 设置 | 参数增幅 | FLOPs 增幅 | 训练显存 | 显存增幅 |
|---|---:|---:|---:|---:|
| 1B DHC×4 | 0.03349% | 0.200% | 41.11 → 51.86 GB | 约 26.1% |
| 7B DHC×4 | 0.02286% | 0.147% | 26.27 → 33.70 GB | 约 28.3% |
| 7B MoE DHC×4 | 0.00570% | 0.208% | 31.59 → 34.65 GB | 约 9.7% |

**[论文事实]** HC 新增参数和算术 FLOPs 很少。

**[复原推导]** 主要成本来自：

$$
\text{activation elements}
\propto
B S n d L,
$$

而普通单流 state 对应 $BSD L$。如果训练需要保存多层 $H_k$，额外 activation 近似随 $n$ 与层数增长。

论文给出的额外 activation 量级为：

$$
2nSBd_{\text{model}}L,
$$

并讨论通过 recomputation 把保存量降到约：

$$
nSBd_{\text{model}}.
$$

这是一种 memory–compute trade-off：减少保存的中间状态，但反向时重新计算。

**[综合判断]** HC 更像 bandwidth/activation-heavy 而不是 compute-heavy 方法。评估它时至少要同时报告：

- tokens/s；
- step time；
- peak memory；
- activation bytes；
- pipeline payload；
- collective time；
- recomputation 开销。

---

## 12. 学到的连接模式提供了什么 insight

### 12.1 $\Lambda$-形连接

**[论文事实]** 可视化显示，训练后的连接倾向于同时依赖：

- 最近若干层；
- 网络底部的一些长期 anchor；
- 较少依赖中间大量层。

连接图因此呈现近似 $\Lambda$ 形。

**[综合判断]** 深层 Transformer 可能同时需要两种 memory：

1. **local working memory**：服务于相邻 block 的短期组合；
2. **persistent anchor memory**：保存较少被中间变换污染的早期表示。

普通 residual stream 把两者混在同一个向量中；HC 用多流状态把它们部分解耦。

### 12.2 Attention 与 FFN 可能承担不同深度角色

**[论文事实]** 论文的可视化观察到：

- FFN 输出更常沿深度传播较远；
- Attention 输出往往更强地服务于紧邻的后续 FFN；
- 某些 stream 出现近似正负成对的存储。

**[综合判断]** 这与“Attention 负责上下文交互、FFN 负责写入长期特征”的直觉一致，但当前只是权重模式与功能之间的经验关联。

要证明因果关系，需要干预实验，而不只是画热力图。

---

## 13. 论文证据没有覆盖什么

### 13.1 统计证据

- 大模型训练没有报告多随机种子均值、方差或误差条；
- 没有显著性检验；
- 下游分数可能存在 checkpoint 波动；
- 论文排除了波动超过 20% 的任务，这有合理性，但也会改变最终平均分的敏感性。

### 13.2 泛化边界

- LLM 实验集中于同一模型家族；
- 没覆盖现代长上下文、不同 normalization、不同位置编码或更多训练 recipe；
- 没有展示更大数量级模型是否仍保持相同收益/成本比。

### 13.3 机制证据

- $\Lambda$-形、Attention/FFN 分工和正负 stream 都是观察性证据；
- 没有通过冻结、打乱、移植或删除指定连接来做因果验证。

### 13.4 工程证据

- 没有完整官方训练代码；
- 没有 wall-clock throughput 或 latency；
- 没有 TP、sequence/context parallel、PP、FSDP 的实现说明；
- 没有通信量和 kernel efficiency 分解；
- 没有明确的混合精度数值 contract。

### 13.5 稳定性

$A_r$ 在深度上反复相乘：

$$
\prod_t A_{r,t}.
$$

原始 HC 没有对其谱半径、奇异值或矩阵范数给出强约束。

**[复原推导]** 如果这些乘积持续放大或压缩，可能引起 activation/gradient scale 漂移。PreNorm-compatible 初始化只能保证起点附近，不能自动保证整个训练过程中的稳定性。

---

## 14. 后续工作：mHC 如何回应原始 HC 的隐患

**[后续工作]** 后续论文 [*mHC: Manifold-Constrained Hyper-Connections*](https://arxiv.org/abs/2512.24880) 明确指出，原始 HC 可能损害 identity mapping、训练稳定性和 scale invariance，并把相关连接矩阵约束到 Birkhoff polytope。

这不应倒推成“原论文结论无效”，而应理解为：

- HC 发现了重要设计空间；
- 原始参数化验证了可学习拓扑的潜力；
- 之后的工作继续补足稳定性与可扩展性约束。

**[综合判断]** “多流 residual state + 可学习路由”可能是长期有效的抽象，而无约束 dense $A_r$ 未必是最终实现。

---

## 15. 论文之外的可验证扩展

### 15.1 稳定化的 $A_r$ 参数化

**提案**

让 $A_r$ 满足 row/column stochastic、近正交、谱归一化或 identity-plus-small-update 约束。

**推理链**

1. 深层路径系数包含多个 $A_r$ 的乘积；
2. 无约束矩阵乘积可能导致尺度漂移；
3. 保结构参数化可让 identity-like transport 更持久；
4. 后续 mHC 为这一方向提供了外部支持。

**可观测预测**

- 更深模型的 activation RMS 与 gradient norm 更稳定；
- 对 learning rate、初始化尺度和混合精度更不敏感；
- 相同 loss 下出现更少 spike 或 divergence。

**证伪条件**

若在相同模型、数据和计算预算下，约束版本并未改善稳定性，且显著损伤最终质量，则该假设不成立。

**最小实验**

在 CPU 可运行的小型 Transformer 上比较：

- unconstrained $A_r$；
- $I+\epsilon\Delta$；
- row-stochastic；
- doubly stochastic。

记录 24、48、96 层下的 activation RMS、Jacobian norm proxy、loss 与 grad norm。

**成本与风险**

归一化或投影会增加额外算子；过强约束可能限制有用的负权重和跨流变换。

### 15.2 稀疏的 local + anchor HC

**提案**

根据 $\Lambda$-形模式，把连接限制为：

- 少量 recent/local paths；
- 少量 persistent anchor streams；
- 其余连接剪枝或不实例化。

**推理链**

1. 学到的 dense 矩阵并非均匀使用所有连接；
2. 主要质量可能来自局部上下文与底部锚点的组合；
3. 显式稀疏结构有机会降低 activation traffic 和矩阵开销。

**可观测预测**

- 在保留约 $20\%$–$40\%$ 有效连接时，quality 接近 dense HC；
- peak memory 和 step time 优于 dense HC；
- 稀疏 mask 在不同 seed 或相邻模型规模间具有一定一致性。

**证伪条件**

若轻度剪枝就显著降低 loss/下游表现，或不同运行学到完全不一致的拓扑，则静态稀疏归纳不成立。

**最小实验**

先训练 SHC，再按权重重要度剪枝并短暂 finetune；与同参数量的随机 mask、banded mask、local+anchor mask 比较。

**成本与风险**

非结构化稀疏未必带来真实硬件加速；需要优先研究结构化 block sparsity。

### 15.3 用干预而不是热力图验证机制

**提案**

对训练后的 HC 做以下干预：

- 冻结某类矩阵；
- 跨 checkpoint 移植 $A_m,A_r,B$；
- 打乱层序；
- 删除 bottom anchors；
- 删除 local connections；
- 交换 Attention 与 FFN 的连接矩阵。

**可观测预测**

- 删除 bottom anchors 更伤害需要保留早期词法/位置信息的任务；
- 删除 local paths 更伤害局部组合和训练 loss；
- 交换 Attention/FFN routing 会破坏论文观察到的功能分工。

**证伪条件**

若这些结构化干预与随机同规模扰动无差别，则当前机制解释缺乏因果支持。

**最小实验**

在一个训练完成的小模型上只修改 HC 参数，不更新 backbone，比较 validation loss 与按任务切分的性能变化。

---

## 16. 分布式实现推演

以下均为 **[复原推导]**，不是论文报告的实现结论。实际行为必须依据固定版本代码和具体 sharding layout 验证。

### 16.1 Tensor Parallel（TP）

如果 hidden dimension $d$ 按 TP rank 切分，而 stream 维 $n$ 在每个 rank 本地复制：

$$
H^{(r)}
\in
\mathbb{R}^{B\times S\times n\times d_{\text{local}}}.
$$

$A_m,A_r,B$ 只在 stream 维操作，理论上可在本地完成，不必新增跨 TP collective。主干 Attention/FFN 仍使用原有 collectives。

需要核对：

- 动态权重生成是否只依赖本地 hidden shard；
- Norm 是否需要跨 TP reduction；
- 动态投影的输出是否在 rank 间一致；
- final stream collapse 位于哪个 collective 前后。

### 16.2 Sequence/Context Parallel

DHC 是 token-wise routing。如果每个 rank 持有完整 hidden、不同 token shard，理论上每个 token 的 HC 更新可以本地完成。

但仍需核对：

- Norm 的统计维；
- Attention 前后的 sequence layout；
- all-to-all/all-gather 之前是否需要携带 $n$ 条流；
- recomputation 是否能复现完全一致的动态权重。

### 16.3 Pipeline Parallel（PP）

如果 stage boundary 直接传递多流状态，payload 从：

$$
B S d
$$

增加到：

$$
B S n d.
$$

忽略 dtype 与 padding 后，通信量可能近似放大 $n$ 倍。这可能成为 HC 在大规模系统中的主要障碍。

可研究：

- stage boundary 临时 collapse、下一 stage 再展开；
- 只传 anchor + compressed local streams；
- 把 HC group 完整放在单个 stage 内；
- 对 streams 做低秩或量化通信。

这些方案都会改变精确数值语义，必须建立 reference contract。

### 16.4 Data Parallel / FSDP

HC 参数量极小，因此仅对新增参数做 sharding 收益有限。主要问题是 activation，而不是 parameter memory。

FSDP 评估应分开记录：

- backbone 参数/optimizer state；
- HC 参数；
- multi-stream activations；
- recomputation 后的 peak memory；
- all-gather 与 reduce-scatter 时间。

### 16.5 混合精度与数值边界

建议至少验证：

- $A_r$ 矩阵乘法的 accumulation dtype；
- dynamic logits 经 $\tanh$ 前的范围；
- stream 求和是否用 FP32 accumulation；
- recomputation 是否保持相同 autocast 路径；
- 长深度矩阵乘积的 singular value 漂移。

---

## 17. 最小 CPU 复现与正确性测试

在进入 GPU 训练前，可以先做一个无 fused kernel 的参考实现。

### 17.1 Shape tests

覆盖：

$$
n\in\{1,2,4\},
\quad
B\in\{1,2\},
\quad
S\in\{1,7\}.
$$

检查每一步：

- 输入 $H:[B,S,n,d]$；
- 读出 $h^{read}:[B,S,d]$；
- 动态 $B:[B,S,1,n]$；
- 动态 $A_m:[B,S,n,1]$；
- 动态 $A_r:[B,S,n,n]$；
- 输出 $H':[B,S,n,d]$。

### 17.2 初始化等价性

固定随机输入和 backbone 权重，比较：

1. 普通 PreNorm block stack；
2. 按论文初始化的 HC stack；
3. final stream collapse 后的输出；
4. 对输入和 backbone 权重的梯度。

应使用明确的 absolute/relative tolerance，并分别测试 FP64 与 FP32。

### 17.3 手算例

使用 $B=S=1,n=2,d=2$ 的小张量，手工计算 read、carry、write，避免 `einsum` 下标写反却仍通过 shape test。

### 17.4 `gradcheck`

用 double precision 对 SHC 和 DHC 的 reference path 做 `torch.autograd.gradcheck`，重点覆盖：

- $A_r$；
- $A_m$；
- $B$；
- 动态生成权重；
- final stream collapse。

### 17.5 展开系数验证

在线性 branch 下，显式展开 3–4 层 HC，检查端到端输出是否等于由

$$
B_j\left(\prod A_{r,t}\right)A_{m,k}
$$

重构的路径和。这是验证矩阵方向和深度组合最直接的测试。

---

## 18. 证据账本

| 结论 | 证据位置 | 证据强度 | 备注 |
|---|---|---|---|
| HC 用多流和可学习矩阵推广 residual connection | §2，Eq. 1–4 | 高 | 方法定义 |
| SHC/DHC 均改善 1B baseline | Table 1–3 | 中 | 单次大模型训练 |
| DHC×4 改善 7B dense | Table 5 | 中 | 无多 seed |
| DHC 改善 7B MoE | §3.2，Table 6 | 中 | 任务间提升幅度差异大 |
| 1.8× convergence | §3.2 与训练曲线 | 中 | token efficiency，不是 wall-clock |
| 视觉任务也有收益 | Appendix B | 中 | 跨模态支持 |
| 参数/FLOPs 增幅很小 | Appendix A | 高 | 表格直接报告 |
| 显存显著增加 | Appendix A | 高 | 约 9.7%–28.3% |
| 学到 $\Lambda$-形模式 | §3.3 与 Appendix figures | 中 | 观察性机制证据 |
| $n=1$ 不足 | Table 2、Appendix G | 中 | 结果与理论解释一致 |
| 原始 HC 存在稳定性风险 | 矩阵乘积推导；后续 mHC | 中 | 原论文未做完整稳定性分析 |
| PP payload 可能近似放大 $n$ 倍 | 本文 shape 推导 | 待验证 | 依赖具体 stage layout |

---

## 19. 怎样阅读这篇论文最省时间

如果只用 15 分钟：

1. 看 §2 的 Eq. 1–4，弄清 $B,A_m,A_r$；
2. 看初始化图和 SHC/DHC 区别；
3. 看 1B、7B、MoE 三组主结果；
4. 看 Appendix A 的显存表；
5. 看连接可视化，理解 $\Lambda$-形；
6. 最后读 Appendix G 的 $n=1$ 解释。

如果要实现：

1. 先写 `[B,S,n,d]` CPU reference；
2. 用手算例检查矩阵方向；
3. 验证初始化等价性和 `gradcheck`；
4. 再接入 Attention/FFN；
5. 最后评估 recomputation 与分布式 layout。

---

## 20. 最终评价

Hyper-Connections 最有价值的地方，不是某一张 benchmark 表，而是它改变了 residual connection 的问题定义：

> residual stream 不一定只能是一条固定的 identity path；它也可以是一个可学习的、多维、沿深度演化的状态系统。

论文的实验足以说明这个方向有潜力：跨 dense、MoE 和视觉模型都观察到收益，且新增参数和 FLOPs 极少。

但原始 HC 仍留下三个关键问题：

1. 多流 activation 是否能在大规模训练中以可接受的显存、带宽和 PP 通信成本实现；
2. 无约束深度矩阵乘积是否能长期保持 identity mapping 与数值稳定；
3. 学到的 $\Lambda$-形拓扑究竟是因果机制，还是与有效训练同时出现的相关现象。

因此，一个稳妥的结论是：

**HC 是一个重要且有启发性的架构抽象；它已经证明“学习 residual topology”值得研究，但工程上更稳定、更稀疏、更适合分布式系统的版本，才可能成为真正可规模化的答案。**
