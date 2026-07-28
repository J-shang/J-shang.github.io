---
title: "Attention Residuals：把 residual stream 从固定累加改成沿深度检索"
description: "复原 Full 与 Block AttnRes 的 depth-wise routing，追踪候选状态、softmax 权重、训练证据和 pipeline cache 代价。"
topic: "residual"
section: "methods"
slug: "attention-residuals"
date: 2026-07-27
updated: 2026-07-27
cutoff: 2026-07-23
featured: true
order: 12
readtime: 46
source:
  repository: "J-shang/residual"
  path: "papers/03-attention-residuals.md"
  url: "https://github.com/J-shang/residual/blob/c55583707ab645bef7408c588fc33cbf91b809a5/papers/03-attention-residuals.md"
  revision: "c55583707ab645bef7408c588fc33cbf91b809a5"
  syncedAt: "2026-07-28"
  contentHash: "sha256:1b9ad71c6832377c09f8595800a487f27da9bbf49e2698ab6c899457d265eb60"
  manifest: "residual"
  managed: true
---
<!-- paper-order: 03 -->

> **论文**：Kimi Team（Guangyu Chen、Yu Zhang、Jianlin Su et al.），*Attention Residuals*<br>
> **机构**：Moonshot AI / Kimi Team<br>
> **版本**：arXiv:2603.15031v1，2026-03-16<br>
> **状态**：technical report<br>
> **主来源**：[arXiv 摘要](https://arxiv.org/abs/2603.15031v1) · [PDF](https://arxiv.org/pdf/2603.15031v1)<br>
> **官方 artifact**：[MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals/tree/85e22310fe5ee860b4a023de312d791de8a5a5e6)，`master` commit `85e22310fe5ee860b4a023de312d791de8a5a5e6`<br>
> **阅读范围**：完整 21 页，包括 Eq. 1–17、Figure 1–9、Table 1–5、Algorithm 1、Appendix A–B<br>
> **后续信息截止日期**：2026-07-23<br>
> **前置阅读**：[Hyper-Connections](/topics/residual/hyper-connections/) · [mHC](/topics/residual/mhc/)

## 证据标签

- **[论文报告]**：论文正文、公式、图表或附录直接给出的内容。
- **[复原推导]**：由论文定义独立推导出的结论。
- **[综合判断]**：结合数学、实验和系统约束形成的解释。
- **[后续证据]**：论文之后出现的 primary source 或官方 artifact。
- **[扩展假设]**：论文之外、需要新实验验证的设计。
- **[待验证]**：当前来源不足，不能视为已经建立的事实。

---

## 先给结论

### 30 秒版

标准 PreNorm residual 把 embedding 和所有历史子层输出按固定系数 $1$ 相加：

$$
h_l
=
\sum_{i=0}^{l-1} v_i.
$$

Attention Residuals（AttnRes）把这个固定累加器改成一个沿网络深度工作的 softmax attention：

$$
h_l
=
\sum_{i=0}^{l-1}
\alpha_{i\to l}v_i,
\qquad
\alpha_{i\to l}
=
\operatorname{softmax}_i
\left(
w_l^\top \operatorname{RMSNorm}(v_i)
\right).
$$

这里 $l$ 是当前 Attention/MLP 子层编号，$v_i\in\mathbb R^d$ 是第 $i$ 个历史 depth source，$h_l\in\mathbb R^d$ 是当前子层读到的运行时 hidden state，$\alpha_{i\to l}$ 是从 source $i$ 到当前层 $l$ 的运行时标量权重，$w_l\in\mathbb R^d$ 是 optimizer 更新的 layer-specific pseudo-query。这里不是让 token 在 sequence 维互相 attention，而是让**每个 token 在 depth 维选择历史表示**。每个子层只增加一个可训练 pseudo-query 和一个 RMSNorm；query 本身与输入无关，但 key 来自当前样本的历史表示，因此最终权重仍随 token 和输入变化。

Full AttnRes 保存并访问所有历史 source，表达力强，但在 activation recomputation 和 pipeline parallel 场景下需要维护 $O(Ld)$ 的历史。Block AttnRes 把 $L$ 个子层分成 $N$ 个 block，只在 block 级表示之间做 attention，把保存状态和跨 stage 通信降到 $O(Nd)$。论文经验上取大约 8 个 block。

最强的正面证据是：

- 五个小规模 MoE 配置中，Full 和 Block AttnRes 的 validation loss 均低于对应 PreNorm baseline；
- 最大 scaling 点上，Full 为 1.692、Block 为 1.693、baseline 为 1.719；
- 拟合曲线声称 Block AttnRes 达到 baseline 相同 loss 时可少用约 20% compute，等价表述为 baseline 需要 $1.25\times$ compute；
- 48B total / 3B activated 的 Kimi Linear 模型训练 1.4T tokens 后，在论文列出的 15 个 benchmark 上 14 项提高、1 项持平；
- 在论文测量的系统设置中，Block AttnRes 的 pipeline training overhead 小于 4%，典型 inference latency overhead 小于 2%。

### 这篇论文真正贡献了什么

1. **问题重构**：residual connection 不只是 gradient highway，也是一个固定的 depth aggregation rule。
2. **方法**：用 token-dependent、softmax-normalized 的 depth attention 替代固定单位权重。
3. **可扩展变体**：Block AttnRes 用 block summary 压缩历史 source。
4. **系统方法**：cross-stage cache、two-phase computation、online-softmax merge 和 sequence-sharded prefill。
5. **统一视角**：用 depth mixing matrix 和 semiseparable rank 解释 standard residual、Highway、HC/mHC 与 AttnRes 的关系。
6. **规模证据**：从 194M 到 528M activated 参数的 scaling sweep，以及 48B/3B activated、1.4T-token 的主实验。

### 最重要的精确化

AttnRes 不是“给普通 residual add 再乘一个 attention 权重”。论文的核心公式是**用 convex combination 替换原来的累加结果**：

$$
\sum_i v_i
\quad\longrightarrow\quad
\sum_i \alpha_i v_i,
\qquad
\alpha_i\ge 0,
\qquad
\sum_i\alpha_i=1.
$$

因此它同时改变了：

- 哪些历史 source 被选中；
- 聚合系数是否依赖输入；
- residual stream 的总尺度；
- 直接梯度路径的系数；
- pipeline payload 和 inference cache contract。

把它称作“drop-in replacement”只表示模型接口可以替换，不表示函数、初始化或分布式实现都与普通 residual 完全等价。

### 最重要的保留意见

- zero-initialized query 产生的是**均匀平均**，不是普通 residual 的单位权重求和，因此初始化并不恢复 baseline 函数。
- softmax convex combination 能约束一次 read 的尺度，但论文没有给出全网络 hidden-state 或 gradient stability theorem；Figure 5 是经验现象。
- 主要实验没有多 seed、误差条、置信区间或显著性检验。
- 训练数据组成没有公开，benchmark contamination 和数据公平性无法独立审计。
- 论文的系统 overhead 缺少硬件拓扑、pipeline 配置、吞吐、batch、kernel 和 profiler 明细。
- 官方仓库截至 2026-07-23 只有报告、README、assets 和 PyTorch-style pseudocode，没有完整训练实现、optimized kernel、checkpoint 或复现实验脚本。
- 论文公式与公开 Block pseudocode 对 source/partial state 的语义存在容易误读的边界；后续论文已经给出不同解释，当前缺少官方实现来裁决。

---

## 阅读前：符号与约定

AttnRes 同时讨论模型数学、block 压缩和 pipeline 调度，原文会复用 $V$、$N$ 等字母。后文在公式旁仍会重新解释；本表先固定全文的 reader-facing convention。

| 符号 | 含义 | 类型或 shape | 作用域 | 身份 |
|---|---|---:|---|---|
| $l,i,j,L$ | 当前子层、source/求和索引、全模型子层总数 | 整数 | depth 维 | 索引/架构超参数 |
| $L_b$ | Transformer block 数；每个 block 含 Attention 与 MLP 两个子层 | 正整数 | 全模型 | 架构超参数，本文通常有 $L=2L_b$ |
| $B,T,d$ | batch size、sequence length、hidden width | 正整数 | batch/模型 | 运行配置或架构超参数 |
| $x$ | 原始模型输入 | 依任务而定 | 一个样本/token sequence | 固定输入，不是参数 |
| $h_l$ | 第 $l$ 个子层实际读到的 hidden state | $[d]$ 或 $[B,T,d]$ | 每层、每 token | 运行时张量 |
| $f_l$ | 第 $l$ 个 PreNorm Attention/MLP branch | $[d]\to[d]$ | 每个子层 | 含普通 Transformer 可训练参数 |
| $v_0=h_1$ | embedding/首个 hidden source | $[d]$ 或 $[B,T,d]$ | 全 depth 可访问 | 运行时 value/key source |
| $v_i=f_i(h_i)$ | 第 $i$ 个子层产生的 source/delta | $[d]$ 或 $[B,T,d]$ | Full AttnRes 历史 | 运行时 value/key source |
| $w_l$ | 第 $l$ 层 pseudo-query | $[d]$ | 每层一份 | optimizer 直接更新的参数，初始化为 0 |
| $q_l$ | query；默认定义为 $q_l=w_l$ | $[d]$ | 每层 | 运行时对参数 $w_l$ 的引用 |
| $k_i$ | source key；默认定义为 $k_i=v_i$，打分前做 RMSNorm | $[d]$ 或 $[B,T,d]$ | 每 source/token | 运行时张量 |
| $\phi(q,k)$ | positive attention kernel $\exp(q^\top\operatorname{RMSNorm}(k))$ | 标量输出 | query–source pair | 固定可微函数 |
| $\alpha_{i\to l}$ | source $i$ 到层 $l$ 的 depth-attention weight | 标量；对固定 $(l,b,t)$ 沿 $i$ 求和为 1 | 每 token、每 source | softmax 生成的运行时张量，不是参数 |
| $\operatorname{RMSNorm}$ | 只用于形成 routing key，value 保持未归一化 | 保持 shape | 每个 source | 可微算子；其 affine weight 若启用则可训练 |
| $J$ | 当前可访问 source 数 | 正整数 | 某一次 Full read | 运行时长度 |
| $N$ | Block AttnRes 的 block 数 | 正整数 | 全模型 | 架构超参数 |
| $S=L/N$ | 每个 AttnRes block 的子层数 | 正整数 | 全模型；末块可更短 | 架构超参数 |
| $\mathcal B_n$ | 第 $n$ 个 AttnRes block 的子层索引集合 | 集合 | $n=1,\ldots,N$ | 固定分块；不要与 mHC 的 Birkhoff polytope $\mathcal B_n$ 混淆 |
| $b_n,b_n^{(i)}$ | 已完成的第 $n$ 个 block summary、该 block 前 $i$ 层的 partial sum | $[d]$ 或 $[B,T,d]$ | Block AttnRes | 运行时状态 |
| $Q,K,V_{\mathrm{src}}$ | batched queries、normalized keys、source values | 例如 $[S,d],[n,d],[n,d]$，另带 batch/token 维 | two-phase attention | 运行时张量 |
| $m^{(g)},\ell^{(g)},o^{(g)}$ | source group $g$ 的最大 logit、shifted-exp sum、未归一化 value numerator | 标量、标量、$[d]$，另带 query/token 维 | online softmax | 运行时统计量 |
| $P,V_{\mathrm{pp}},C_{\mathrm{pp}}=PV_{\mathrm{pp}}$ | physical pipeline stages、每 stage 的 virtual stages、pipeline chunks | 正整数 | PP 调度 | 系统配置；$V_{\mathrm{pp}}$ 不等于 value matrix |
| $N_p$ | 每个 physical stage 平均产生的 AttnRes blocks | 正数/整数 | PP 通信模型 | 系统配置或近似计数 |
| $\|\cdot\|$ | 任意向量 norm；写 $\|\cdot\|_2$ 时特指 Euclidean norm | 标量输出 | 尺度推导 | 固定数学运算 |

关键身份链条是：

$$
\left\{w_l,\text{RMSNorm affine weight},\text{branch parameters}\right\}_{\text{可训练参数}}
\longrightarrow
\left\{v_i,k_i,\alpha_{i\to l},h_l\right\}_{\text{运行时张量}}.
$$

**[复原推导]** 所有新增参数都由原来的预训练主 loss 端到端更新；论文没有额外的 routing supervision 或 auxiliary routing loss。softmax 约束的是每次 forward 产生的 $\alpha$，不是 optimizer 直接更新一个自由的 attention-weight table。

---

## 1. 论文要解决的不是“有没有 skip connection”

### 1.1 标准 residual 有两个角色

为与论文一致，把每个 self-attention 或 MLP 都视为一个独立子层。令：

$$
h_l\in\mathbb R^d
$$

是进入第 $l$ 个子层的单 token hidden state，$f_l$ 包含该子层的 PreNorm 和变换。标准 residual 为：

$$
h_l
=
h_{l-1}
+
f_{l-1}(h_{l-1}).
$$

它的第一个角色是熟悉的 gradient highway。局部 Jacobian 为：

$$
\frac{\partial h_l}{\partial h_{l-1}}
=
I
+
\frac{\partial f_{l-1}}{\partial h_{l-1}}.
$$

第二个角色容易被忽视：展开递推后，

$$
h_l
=
h_1
+
\sum_{i=1}^{l-1}f_i(h_i).
$$

定义：

$$
v_0=h_1,
\qquad
v_i=f_i(h_i)\quad(i\ge 1),
$$

则：

$$
h_l
=
\sum_{i=0}^{l-1}v_i.
$$

也就是说，标准 residual 同时规定了一个 depth mixing policy：所有历史 source 的系数固定为 1。

### 1.2 论文认为固定累加带来三类压力

**[论文报告，§2.1]**

1. **不能选择性访问**：attention 子层与 MLP 子层只能接收同一个累计状态，不能分别偏好不同深度的 source。
2. **聚合不可逆**：历史表示一旦被压进一个累计状态，后层不能单独取回某个早期 source。
3. **PreNorm dilution**：累计状态随深度增大，固定尺度的单层更新在总状态中的相对占比变小。

论文把这个问题类比为 RNN 在 time 维把历史压缩到单一 recurrent state，而 Transformer attention 允许当前位置直接访问历史 token。

### 1.3 类比成立到什么程度

类比有启发性，但不是数学同一：

| Sequence attention | Depth attention |
|---|---|
| source 是不同 token/position | source 是不同子层或 block 的表示 |
| query 通常由当前 token state 生成 | 默认 query 是每层固定的学习向量 $w_l$ |
| sequence length 可达 $10^5$–$10^6$ | depth 通常低于 $10^3$ |
| 改变 token 间信息流 | 改变 residual path |
| 常是模型主计算 | 在论文设置中是轻量 residual-side 运算 |

AttnRes 借用了 attention 的归一化检索形式，不代表网络深度真的等价于自然语言时间轴。

---

## 2. Full AttnRes：沿 depth 做单 query softmax attention

### 2.1 核心公式

论文定义 kernel：

$$
\phi(q,k)
=
\exp
\left(
q^\top\operatorname{RMSNorm}(k)
\right).
$$

第 $l$ 个子层的 query、key、value 为：

$$
q_l=w_l,
\qquad
k_i=v_i,
\qquad
v_i=
\begin{cases}
h_1,&i=0,\\
f_i(h_i),&1\le i\le l-1.
\end{cases}
$$

attention weight 为：

$$
\alpha_{i\to l}
=
\frac{
\exp\left(w_l^\top\operatorname{RMSNorm}(v_i)\right)
}{
\sum_{j=0}^{l-1}
\exp\left(w_l^\top\operatorname{RMSNorm}(v_j)\right)
}.
$$

第 $l$ 个子层的输入为：

$$
h_l
=
\sum_{i=0}^{l-1}
\alpha_{i\to l}v_i.
$$

论文没有在 dot product 中写 $1/\sqrt d$ temperature。RMSNorm 控制 key 的尺度，query norm 则由训练决定。

### 2.2 实际 tensor shapes

对批量序列：

$$
v_i\in\mathbb R^{B\times T\times d},
\qquad
w_l\in\mathbb R^d.
$$

假设当前有 $J$ 个可用 source：

| Tensor | Shape | 含义 |
|---|---:|---|
| `V` | $[J,B,T,d]$ | 历史 source stack |
| `K = RMSNorm(V)` | $[J,B,T,d]$ | 归一化 routing keys |
| `w_l` | $[d]$ | 当前子层的 pseudo-query |
| `logits` | $[J,B,T]$ | 每个 token 对每个 depth source 的分数 |
| `alpha` | $[J,B,T]$ | 沿 $J$ 维 softmax |
| `h_l` | $[B,T,d]$ | 加权后的当前子层输入 |

对应的最小运算是：

```python
V = torch.stack(sources, dim=0)              # [J, B, T, D]
K = rms_norm(V)                              # [J, B, T, D]
logits = torch.einsum("d,jbtd->jbt", w, K)  # [J, B, T]
alpha = logits.softmax(dim=0)                # depth/source axis
h = torch.einsum("jbt,jbtd->btd", alpha, V)
```

这里的 routing：

- 对不同 token 可以不同；
- 对同一 token 的所有 hidden channels 共用一个标量权重；
- 默认不是 multihead；
- 不在 sequence 维混合 token；
- query 不看当前 hidden state，但 key 看输入产生的历史表示。

### 2.3 为什么叫 input-dependent

虽然：

$$
q_l=w_l
$$

对所有样本固定，但：

$$
k_i=v_i(x)
$$

随输入 $x$ 变化，所以：

$$
\alpha_{i\to l}(x)
=
\operatorname{softmax}_i
\left(
w_l^\top\operatorname{RMSNorm}(v_i(x))
\right)
$$

仍然是 input-dependent。

更准确的命名是：

> **static query + dynamic key 的 token-wise depth routing**。

论文消融显示，从当前 hidden state 生成 dynamic query 可把 loss 从 1.737 进一步降到 1.731，但需要每层额外的 $d\times d$ projection，并破坏 block 内 query 预计算，因此没有作为默认方案。

### 2.4 一个三 source 的最小例子

设：

$$
v_0=
\begin{bmatrix}
2\\0
\end{bmatrix},
\quad
v_1=
\begin{bmatrix}
0\\3
\end{bmatrix},
\quad
v_2=
\begin{bmatrix}
2\\1
\end{bmatrix}.
$$

标准 residual 聚合为：

$$
h_{\mathrm{res}}
=
v_0+v_1+v_2
=
\begin{bmatrix}
4\\4
\end{bmatrix}.
$$

若 AttnRes query 初始化为零，则三个 logit 都是零：

$$
\alpha_0=\alpha_1=\alpha_2=\frac13,
$$

于是：

$$
h_{\mathrm{attnres}}
=
\frac13(v_0+v_1+v_2)
=
\begin{bmatrix}
4/3\\4/3
\end{bmatrix}.
$$

这说明 zero query 初始化带来的是 equal-weight average，不是 ordinary residual sum。

### 2.5 softmax 给出的局部尺度性质

对任意 norm，由 $\alpha_i\ge 0$ 且 $\sum_i\alpha_i=1$：

$$
\left\|
\sum_i\alpha_i v_i
\right\|
\le
\sum_i\alpha_i\|v_i\|
\le
\max_i\|v_i\|.
$$

**[复原推导]** 一次 AttnRes read 不会超过现有 source 的最大 norm。

但这不等于全网络有统一上界：

- source $v_i=f_i(h_i)$ 自身可能增长；
- Block source 是多个子层输出之和；
- attention 参数和 branch 参数共同训练；
- gradient 还包含 softmax 对 key 的导数。

所以 Figure 5 的“bounded periodic pattern”是经验结果，不是由上式直接推出的全局 theorem。

---

## 3. Block AttnRes：用 block summary 压缩历史

### 3.1 状态定义

把 $L$ 个子层分为 $N$ 个 block，每个 block 约有：

$$
S=\frac{L}{N}
$$

个子层。最后一个 block 可以更短。

令 $\mathcal B_n$ 是第 $n$ 个 block 的子层集合。论文的形式化定义为：

$$
b_0=h_1,
$$

$$
b_n
=
\sum_{j\in\mathcal B_n}
f_j(h_j).
$$

当前 block 内前 $i$ 个子层的 partial sum 为：

$$
b_n^{(i)}
=
\sum_{j\in\text{first }i\text{ layers of }\mathcal B_n}
f_j(h_j).
$$

### 3.2 Read rule

对 block $n$ 的第一个子层，source 是：

$$
V
=
\begin{bmatrix}
b_0&b_1&\cdots&b_{n-1}
\end{bmatrix}^{\mathsf T}.
$$

对当前 block 的后续子层，再加入 partial block：

$$
V
=
\begin{bmatrix}
b_0&b_1&\cdots&b_{n-1}&b_n^{(i-1)}
\end{bmatrix}^{\mathsf T}.
$$

然后仍使用：

$$
h_l
=
\operatorname{softmax}
\left(
w_l^\top\operatorname{RMSNorm}(V)
\right)^{\mathsf T}
V.
$$

### 3.3 Write rule

当前子层输出：

$$
y_l=f_l(h_l)
$$

不直接加回 $h_l$，而是写进当前 partial block：

$$
b_n^{(i)}
=
b_n^{(i-1)}
+
y_l.
$$

block 结束时，$b_n^{(S)}$ 成为一个完成的历史 source；下一 block 重新开始 partial accumulation。

### 3.4 Operational contract

| 问题 | Full AttnRes | Block AttnRes |
|---|---|---|
| stored state | embedding + 所有历史子层 source | embedding + 完成的 block source + 当前 partial |
| read | 对所有历史 source 做 softmax attention | 对历史 block 和当前 partial 做 softmax attention |
| transform | $y_l=f_l(h_l)$ | $y_l=f_l(h_l)$ |
| write | 保存 $y_l$ 为新 source | $b_n^{(i)}\leftarrow b_n^{(i-1)}+y_l$ |
| residual update | 由下一层重新 attention 聚合 | 当前 block 内 additive partial update |
| weight granularity | 每 token、每 source、全 channel 共享 | 每 token、每 block/partial、全 channel 共享 |
| initialization | $w_l=0$，均匀 attention | $w_l=0$，均匀 attention |
| history cost | $O(Ld)$ | $O(Nd)$ |
| depth arithmetic | $O(L^2d)$ | 论文概括为 $O(N^2d)$，逐层直接实现则为 $O(LNd)$ |

最后一行要区分两个统计口径：

- 全模型有 $L$ 个 query，每个大约看 $N$ 个 source，直接算是 $O(LNd)$；
- 若把 $L=NS$ 且 $S$ 视为固定 block width，随 block 数增长可写成 $O(N^2d)$；
- two-phase batching 改变的是 memory traffic 和执行方式，不改变模型定义。

### 3.5 两个极端

论文称：

- $N=L$ 恢复 Full AttnRes；
- $N=1$ 退化为“embedding 独立为 $b_0$ 的 standard residual”。

第一条在 source 粒度上清楚：每个 block 只有一个子层。

第二条需要谨慎。$N=1$ 时 block 内 write 确实是 additive partial sum，但后续子层的 read 仍可能在 $b_0$ 与 partial sum 之间做 softmax replacement：

$$
h_l
=
\alpha_0 b_0
+
\alpha_1 b_1^{(i-1)}.
$$

这通常不等于：

$$
b_0+b_1^{(i-1)}.
$$

**[综合判断]** 因此 $N=1$ 更适合解释为“只保留一个 additive block 的拓扑极端”，不应不加条件地当成与普通 PreNorm 完全相同的函数。

---

## 4. 初始化：稳定起步，但不是 identity-preserving

### 4.1 论文的初始化

所有 pseudo-query 都初始化为：

$$
w_l=0.
$$

于是每个 source 的 score 都是 0：

$$
w_l^\top\operatorname{RMSNorm}(v_i)=0,
$$

softmax 产生均匀权重：

$$
\alpha_{i\to l}
=
\frac{1}{l}.
$$

论文报告 zero initialization 能避免训练波动。

### 4.2 它恢复的 special case

它恢复的是：

> equal-weight normalized aggregation。

不是：

> standard residual 的 equal-weight unnormalized sum。

两者相差随 source 数变化的系数：

$$
h_l^{\mathrm{AttnRes,init}}
=
\frac1l
h_l^{\mathrm{standard}}.
$$

即使下一个子层前有 RMSNorm，使统一 scale 的一部分影响被消除，也不能据此宣布两种网络完全相同，因为：

- final output path 仍可能受 scale 影响；
- RMSNorm 不是对所有数值和反向路径都无影响；
- block 内 source 数变化；
- attention 对 key 的梯度路径不同；
- pretrained checkpoint 转换会发生函数突变。

### 4.3 gradient highway 发生了什么

标准 residual 有显式 identity Jacobian 项。AttnRes 中，某个 source $v_i$ 到未来 read $h_l$ 的梯度包含：

$$
\frac{\partial h_l}{\partial v_i}
=
\alpha_{i\to l}I
+
\text{softmax/key-dependent terms}.
$$

直接项系数是 $\alpha_{i\to l}$，不再固定为 1。

**[复原推导]** AttnRes 保留了从未来层到历史 source 的直接路径，但没有保留单位系数 identity path。某个 source 若长期获得极小权重，其直接梯度路径也会变弱。

这使论文 Figure 5(c) 中“梯度分布更均匀”的观察很有价值，但它仍是训练结果，不是结构上无条件保证。

---

## 5. Two-phase computation 为什么与逐层 attention 等价

### 5.1 为什么可以预计算

默认 query：

$$
w_l
$$

与当前层 forward output 无关，所以同一 block 内所有 $S$ 个 query 可以预先组成：

$$
Q
\in
\mathbb R^{S\times d}.
$$

对已经完成的历史 block：

$$
K,V_{\mathrm{src}}
\in
\mathbb R^{n\times d},
$$

其中 $K$ 是 normalized routing keys，$V_{\mathrm{src}}$ 是未归一化的 source values；下标 `src` 是本文为消除歧义增加的 reader-facing 标记，原论文把 value matrix 简写成 $V$。可以一次 batched matmul 计算所有 query 的 inter-block attention。

当前 block 的 partial sum 必须顺序产生，因此留到 Phase 2。

### 5.2 Phase 1

对 block 内所有 query 一次计算：

$$
\{o_l^{(1)},m_l^{(1)},\ell_l^{(1)}\}_{l\in\mathcal B_n}
=
\operatorname{AttnWithStats}(Q,K,V_{\mathrm{src}}).
$$

其中对每个 query：

- $m^{(1)}$ 是历史 block logits 的最大值；
- $\ell^{(1)}$ 是减去最大值后的 exp sum；
- $o^{(1)}$ 是相同权重下尚未除以 denominator 的 value numerator。

### 5.3 Phase 2

顺序生成当前 partial source，得到第二组：

$$
o_l^{(2)},m_l^{(2)},\ell_l^{(2)}.
$$

令：

$$
m_l
=
\max
\left(
m_l^{(1)},m_l^{(2)}
\right).
$$

两个 source group 合并后的 denominator 为：

$$
\ell_l
=
e^{m_l^{(1)}-m_l}\ell_l^{(1)}
+
e^{m_l^{(2)}-m_l}\ell_l^{(2)}.
$$

numerator 为：

$$
o_l
=
e^{m_l^{(1)}-m_l}o_l^{(1)}
+
e^{m_l^{(2)}-m_l}o_l^{(2)}.
$$

最终：

$$
h_l
=
\frac{o_l}{\ell_l}.
$$

这是 online softmax 的精确分块合并，不是近似。

### 5.4 正确性 contract

reference 与 optimized path 应满足：

$$
\operatorname{Attn}
\left(
Q,[K_1;K_2],[V_1;V_2]
\right)
=
\operatorname{OnlineMerge}
\left(
\operatorname{AttnStats}(Q,K_1,V_1),
\operatorname{AttnStats}(Q,K_2,V_2)
\right)
$$

在同一 dtype 和相同 reduction order 下应数值接近；混合精度实现需要明确：

- logits accumulation dtype；
- max/LSE dtype；
- numerator accumulation dtype；
- merge 前后是否 cast；
- recomputation 是否使用相同 RMSNorm 和 source。

---

## 6. Pipeline parallel：payload 从单状态变成历史 cache

### 6.1 naïve communication

设：

- $P$：physical pipeline stages；
- $V_{\mathrm{pp}}$：每个 physical stage 的 virtual stages；
- $C_{\mathrm{pp}}=PV_{\mathrm{pp}}$：总 pipeline chunks；
- $N_p$：每个 physical stage 平均产生的 AttnRes block 数；
- $d$：hidden dimension。

原论文在这段把 virtual-stage 数和 pipeline chunks 分别简写为 $V$、$C$；本文增加 `pp` 下标，避免与 attention value matrix 以及 mHC 的 hidden width $C$ 混淆。

若每次 stage transition 都重新发送全部累计 block，论文给出每 token 通信量：

$$
\operatorname{Comm}_{\mathrm{naive}}
=
\sum_{j=1}^{C_{\mathrm{pp}}-1}
jN_p d
=
\frac{C_{\mathrm{pp}}(C_{\mathrm{pp}}-1)}{2}N_p d.
$$

它随 virtual chunks 的总数近似二次增长。

### 6.2 cross-stage cache

接收方把早先 virtual stage 收到的 block 留在本地。之后只发送相对该接收方 cache 新增加的 blocks：

$$
\operatorname{Comm}_{\mathrm{cached}}
=
\frac{P(P-1)}{2}N_p d
+
(V_{\mathrm{pp}}-1)P^2N_p d.
$$

论文强调的主要收益是 peak transition payload：

$$
O(C_{\mathrm{pp}})
\quad\longrightarrow\quad
O(P),
$$

即约 $V_{\mathrm{pp}}\times$ 改善，从而更容易与 steady-state 1F1B compute overlap。

Figure 3 的 $P=4,V_{\mathrm{pp}}=2$ 例子中，第二个 virtual stage 因 cache 避免了 6 次重复 block transmission。

### 6.3 pipeline state ownership

一个可实现的 ownership contract 是：

| State | Ownership | 生命周期 |
|---|---|---|
| completed local block | 产生它的 stage 首先拥有 | forward 后进入下游 cache；backward 完成后释放 |
| received block cache | 每个需要读取它的 stage replica | micro-batch 对应 forward/backward 窗口 |
| current partial block | 当前计算 stage | block 边界前持续更新 |
| pseudo-query | 所属子层的 parameter shard | 与子层参数一致 |
| softmax stats | 当前计算 stage | optimized forward 或 checkpoint recomputation |

实现时必须给 cache key 加上：

$$
(\text{microbatch},\text{virtual stage},\text{block id},\text{direction})
$$

而不能只按 block id 缓存，否则 interleaved schedule 会读错 micro-batch。

### 6.4 “小于 4% overhead”的边界

**[论文报告，§4.1]**：

- 无 pipeline parallel 时，Block AttnRes training overhead 被描述为 negligible；
- 启用 pipeline parallel 后，端到端 overhead 小于 4%。

论文没有给出：

- GPU/互联型号；
- $P$、$V$、micro-batch 数；
- model/tensor/context parallel 组合；
- baseline 与 AttnRes 的 tokens/s；
- overlap timeline；
- cache bytes；
- 是否包含 optimizer 和 data pipeline；
- variance 或多次测量。

所以该数字只能解释为作者内部配置中的实测点，不能当作跨系统常数。

---

## 7. Tensor/context parallel 与长上下文 prefill

### 7.1 block cache bytes

若 block source 使用 $b$ bytes/element，则：

$$
\operatorname{Bytes}_{\mathrm{cache}}
=
N T d b.
$$

论文给出一个 128K context、$N=8$ 的例子，总 cache 约 15 GB。

若 sequence 维在 $P$ 个设备上分片：

$$
\operatorname{Bytes}_{\mathrm{device}}
=
N\frac{T}{P}db.
$$

论文报告该例约降到 1.9 GB/device；再配合 16K chunked prefill，降到 0.3 GB/device 以下。

### 7.2 为什么 sequence sharding 可行

AttnRes 的 depth attention 对每个 token 独立：

$$
\alpha_{i\to l,b,t}
=
\operatorname{softmax}_i
\left(
w_l^\top K_{i,b,t}
\right).
$$

不同 $t$ 之间没有 reduction，因此 Phase 1 可以在本地 sequence shard 上计算。

论文建议 Phase 2 merge 融入标准 TP communication path：

1. 输出 `reduce-scatter`；
2. 本地 online-softmax merge；
3. `all-gather` 重建；
4. 与 RMSNorm 等操作 fusion。

这要求 optimized path 明确：

- 哪些 tensor 在 hidden 维 sharded；
- query 是否 replicated；
- RMSNorm 的平方和 collective 在哪里；
- merge 发生在 reduce-scatter 前还是后；
- output projection 是 row-parallel 还是 column-parallel；
- sequence/context parallel 的布局是否与 TP 相容。

### 7.3 不能假设一种通用 sharding

论文给的是设计方向，不是完整实现。实际 tensor ownership 必须从具体 framework 和 pinned commit 推导。

在没有官方训练代码时，不能直接断言：

- block cache 一定沿 sequence shard；
- pseudo-query 一定 replicated；
- RMSNorm 一定使用某种 fused kernel；
- Phase 2 一定不新增 collective；
- cache 能与任意 context-parallel algorithm 直接兼容。

---

## 8. Inference I/O：算术不大，HBM traffic 才是焦点

### 8.1 论文的典型设置

Table 1 使用：

$$
L=128,\qquad
N=8,\qquad
S=\frac LN=16,\qquad
m=4.
$$

只统计 residual mechanism 的 memory access，不含子层 $f_l$ 内部 I/O。

| 方法 | 每 token、每子层 residual-side I/O | 典型值 |
|---|---:|---:|
| Standard residual | $3d$ | $3d$ |
| mHC，$m$ streams | $(8m+2)d+2m^2+4m$ | 约 $34d$ |
| Full AttnRes，two-phase | $(S+N)d$ | $24d$ |
| Block AttnRes，two-phase | $(N/S+5)d$ | $5.5d$ |

Block AttnRes 的 Phase 1 历史读取被 $S$ 个 query 分摊；Phase 2 只处理当前 partial source。

### 8.2 Full AttnRes Appendix B 推导

把 Full AttnRes 仅为调度目的切成 $N$ 组，每组 $S=L/N$ 层，不改变 source 粒度。

Phase 1 总 read：

$$
\operatorname{Read}_{\mathrm{inter}}
=
dL(N-1).
$$

Phase 2 总 read：

$$
\operatorname{Read}_{\mathrm{intra}}
=
NS(S-1)d.
$$

总 write：

$$
\operatorname{Write}_{\mathrm{total}}
=
2Ld.
$$

除以 $L$：

$$
\operatorname{Read/layer}
=
(S+N-2)d,
$$

$$
\operatorname{Write/layer}
=
2d,
$$

因此：

$$
\operatorname{I/O/layer}
=
(S+N)d.
$$

若只优化这个上界，可令 $S=N=\sqrt L$，得到：

$$
\operatorname{I/O/layer}
\approx
2\sqrt L\,d.
$$

**[复原推导]** 这是调度参数的 I/O 最优点，不表示模型 architecture 的 block 数也必然应取 $\sqrt L$。

### 8.3 latency claim

**[论文报告，§4.2]** 典型 inference workload 上 latency overhead 小于 2%。

缺少的审计信息包括：

- decode 与 prefill 的分开结果；
- batch size、context length、generated length；
- GPU、memory bandwidth 和 kernel 版本；
- Full 与 Block 的各自 latency；
- eager、CUDA Graph 或 speculative decoding；
- TP/PP degree；
- p50/p95 latency。

因此应把“小于 2%”理解为 feasibility signal，而不是部署承诺。

---

## 9. 实验设置

### 9.1 架构 baseline

论文使用 Kimi Linear 风格 MoE：

- Kimi Delta Attention（KDA）与 Multi-Head Latent Attention（MLA）按 3:1 交替；
- 每个 attention 子层后接 MoE feed-forward；
- 设计继承 Moonlight / DeepSeek-V3 路线；
- AttnRes 之外，模型 depth、hidden dimension、expert routing 和 MLP 结构保持不变。

论文中的“layer”通常指 attention 或 MLP 子层：

$$
L
=
2L_b,
$$

其中 $L_b$ 是 Transformer block 数。

### 9.2 新增参数

每个子层新增：

- 一个 pseudo-query $w_l\in\mathbb R^d$；
- 一个用于 key 的 RMSNorm。

参数增量约为：

$$
O(Ld),
$$

相对 LLM 主参数量很小。

参数与运行时量应明确区分：

| 对象 | optimizer 是否直接更新 | 怎样获得梯度 |
|---|---|---|
| $w_l$ | 是 | 主任务 loss 经 depth softmax 的 query–key dot product 反传 |
| key RMSNorm 的 affine weight（若启用） | 是 | 主任务 loss 经 routing logits 反传 |
| Attention/MLP、embedding、unembedding 等主干参数 | 是 | 与 baseline 相同的主任务 loss |
| $\alpha_{i\to l}$ | 否 | 当前 input 的 softmax 输出，只是运行时张量 |
| $v_i,k_i,h_l,b_n,b_n^{(i)}$ | 否 | activation/state；在计算图中承接梯度 |

**[复原推导]** 因此训练代码仍是普通的：

```python
loss = language_model_loss(model(tokens), targets)
loss.backward()
optimizer.step()
```

不需要给“正确 source”提供标签，也没有额外 routing loss。$w_l=0$ 只是初始化约束；训练开始后，主 loss 可以沿上述可微链更新每层 pseudo-query，而不需要另设一个启动路由的目标函数。

### 9.3 scaling sweep

五个模型都使用：

- context length 8192；
- cosine learning-rate schedule；
- 同一规模内所有变体共享由 baseline 选出的 hyperparameters；
- 比较 PreNorm、Block AttnRes、Full AttnRes 和 mHC(-lite)。

论文没有报告：

- 数据集名称与组成；
- validation set 定义；
- optimizer 细节；
- weight decay；
- warmup；
- seeds；
- loss variance；
- 每个点是否只训练一次。

### 9.4 48B/3B activated 主实验

配置：

- 48B total parameters；
- 3B activated parameters；
- 27 Transformer blocks，即 54 attention/MLP 子层；
- 256 routed experts，每 token 选 8 个；
- 1 个 shared expert；
- 每 6 个子层一个 AttnRes block；
- 共 9 个 block，加 embedding 后是 10 个 depth source。

训练：

- 4096 context；
- Muon optimizer；
- WSD schedule；
- global batch 8M tokens；
- 1T-token pre-training；
- 约 400B high-quality-token mid-training；
- 随后扩展到 32K context。

总计约 1.4T tokens。

---

## 10. Scaling law 结果

### 10.1 原始结果

| Activated params | Tokens | $L_b$ | $H$ | $d_{\mathrm{model}}$ | Baseline | Block | Full | mHC(-lite) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 194M | 38.7B | 12 | 12 | 896 | 1.931 | 1.909 | 1.899 | 1.906 |
| 241M | 45.4B | 13 | 13 | 960 | 1.895 | 1.875 | 1.874 | 1.869 |
| 296M | 62.1B | 14 | 14 | 1024 | 1.829 | 1.809 | 1.804 | 1.807 |
| 436M | 87.9B | 16 | 16 | 1168 | 1.766 | 1.746 | 1.737 | 1.747 |
| 528M | 119.0B | 17 | 17 | 1264 | 1.719 | 1.693 | 1.692 | 1.694 |

### 10.2 可以支持的结论

**[论文报告，Table 2]**

- Full AttnRes 在五个规模都优于 baseline。
- Block AttnRes 在五个规模都优于 baseline。
- 最大规模时 Full 与 Block 只差 0.001。
- mHC(-lite) 并非每个规模都落后：241M 点的 1.869 优于 Full 的 1.874。
- 最大点 Full 1.692、Block 1.693、mHC(-lite) 1.694，差距都很小。

### 10.3 简单 delta

最大规模上：

$$
\Delta_{\mathrm{Block}}
=
1.693-1.719
=
-0.026,
$$

$$
\Delta_{\mathrm{Full}}
=
1.692-1.719
=
-0.027.
$$

相对 loss 降幅约：

$$
\frac{0.026}{1.719}
\approx
1.51\%,
$$

$$
\frac{0.027}{1.719}
\approx
1.57\%.
$$

这是 loss 的相对差，不是 perplexity 改善，也不能直接转换为 benchmark accuracy。

### 10.4 scaling fit

论文拟合：

$$
\mathcal L_{\mathrm{base}}
=
1.891C^{-0.057},
$$

$$
\mathcal L_{\mathrm{block}}
=
1.870C^{-0.058},
$$

$$
\mathcal L_{\mathrm{full}}
=
1.865C^{-0.057}.
$$

在 $C=5.6$ PFLOP/s-days 附近，论文给出：

$$
\mathcal L_{\mathrm{block}}=1.692,
\qquad
\mathcal L_{\mathrm{base}}=1.714,
$$

并把差距解释为 $1.25\times$ compute advantage。

### 10.5 保留意见

- 只有五个规模点；
- 每种方法的 exponent 非常接近；
- 没有 fit uncertainty、residual plot 或 held-out scale；
- “$1.25\times$”来自拟合曲线，不是直接训练一对相同 loss、不同 compute 的模型；
- 数据和 validation protocol 未公开。

所以可以说“在这五个点和该拟合假设下存在稳定 offset”，不能说已经建立跨架构的通用 scaling law。

---

## 11. 48B/3B activated 下游结果

### 11.1 论文表格

| 类别 | Benchmark | Baseline | Block AttnRes | 绝对变化 |
|---|---|---:|---:|---:|
| General | MMLU | 73.5 | 74.6 | +1.1 |
| General | MMLU-Pro | 52.2 | 52.2 | 0.0 |
| General | GPQA-Diamond | 36.9 | 44.4 | +7.5 |
| General | BBH | 76.3 | 78.0 | +1.7 |
| General | ARC-Challenge | 64.6 | 65.7 | +1.1 |
| General | HellaSwag | 83.2 | 83.4 | +0.2 |
| General | TriviaQA | 69.9 | 71.8 | +1.9 |
| Math & Code | GSM8K | 81.7 | 82.4 | +0.7 |
| Math & Code | MGSM | 64.9 | 66.1 | +1.2 |
| Math & Code | Math | 53.5 | 57.1 | +3.6 |
| Math & Code | CMath | 84.7 | 85.1 | +0.4 |
| Math & Code | HumanEval | 59.1 | 62.2 | +3.1 |
| Math & Code | MBPP | 72.0 | 73.9 | +1.9 |
| Chinese | CMMLU | 82.0 | 82.9 | +0.9 |
| Chinese | C-Eval | 79.6 | 82.5 | +2.9 |

### 11.2 最稳妥的结论

**[论文报告，Table 3]** 在作者使用的 Kimi Linear evaluation protocol 下，Block AttnRes：

- 14 项提高；
- 1 项持平；
- 没有表中任务下降；
- 最大绝对提升是 GPQA-Diamond 的 +7.5。

### 11.3 不能直接推出的结论

论文把 reasoning/code 上较大提升与更好的 depth-wise information flow 联系起来。这是合理 hypothesis，但 Table 3 本身不能排除：

- optimization 更容易；
- scale/normalization 改变；
- benchmark variance；
- decoding 或 checkpoint selection 差异；
- 训练数据中的任务分布差异；
- 单次评估噪声。

没有 task-level intervention 能证明模型确实因为“取回了某个早期 representation”而答对。

---

## 12. Ablation：哪些设计真的有证据

### 12.1 Table 4

| Variant | Validation loss | 相对 Full |
|---|---:|---:|
| Baseline PreNorm | 1.766 | +0.029 |
| DenseFormer | 1.767 | +0.030 |
| mHC | 1.747 | +0.010 |
| Full AttnRes | 1.737 | 0 |
| input-dependent query | 1.731 | -0.006 |
| input-independent mixing | 1.749 | +0.012 |
| sigmoid instead of softmax | 1.741 | +0.004 |
| Full without RMSNorm | 1.743 | +0.006 |
| sliding window, embedding + 8 recent | 1.764 | +0.027 |
| Block, $S=4$ | 1.746 | +0.009 |
| Block multihead, $H=16$ | 1.752 | +0.015 |
| Block without RMSNorm | 1.750 | +0.013 |

论文 caption 称这是“16-layer model”，正文称“16-head model from Table 2”；Table 2 对应 $L_b=16,H=16$。这里存在 layer/head 术语歧义，应以完整配置表为准。

### 12.2 input dependence

DenseFormer 1.767 与 input-independent mixing 1.749 表明：

- 仅增加 cross-layer access 不一定足够；
- learned static mixing 在该实验里弱于 dynamic key routing。

但 input-independent mixing 仍优于 baseline 0.017，而 DenseFormer 没有提升，说明“是否 input-dependent”不是唯一因素；具体参数化、归一化和训练条件也重要。

### 12.3 query 设计

dynamic query 达到最优 1.731，但论文选择 static learned query 1.737。

这是明确的 quality–system trade-off：

| Query | Quality | 系统影响 |
|---|---|---|
| learned static $w_l$ | 1.737 | 可提前 batch block 内 query |
| projected dynamic query | 1.731 | 每层 $d\times d$ projection；必须等当前 state |

默认方法不是 ablation 中质量最强的方法，而是作者认为最容易规模化的方法。

### 12.4 softmax 的作用

sigmoid 为 1.741，softmax 为 1.737。

论文解释为 softmax 的竞争性归一化促使 source selection 更尖锐。证据支持：

> 在这一配置中 softmax 优于所测 sigmoid 实现。

它不能单独证明：

> 所有 normalized competition 都优于独立 gating。

还需要控制 temperature、gate bias、scale 和稀疏化策略。

### 12.5 RMSNorm 的作用

去掉 RMSNorm：

- Full 从 1.737 变成 1.743；
- Block 从 1.746 变成 1.750。

key normalization 防止大 magnitude source 仅凭 norm 获得高 logit。

这也说明 attention heatmap 不能简单解释成“语义相关性”：没有 normalization 时，它可能主要是 magnitude detector。

### 12.6 block size

Figure 6：

| Block size $S$ | 近似 validation loss |
|---:|---:|
| 1，Full | 1.737 |
| 2 | 1.746 |
| 4 | 1.746 |
| 8 | 1.748 |
| 16 | 1.753 |
| 32 | 1.757 |
| Baseline | 1.766 |

性能随 block 变粗而平滑退化，没有突然崩坏。$S=2$ 到 $8$ 形成较平的平台，使固定约 8 个 block 成为可行工程折中。

### 12.7 远程 source 是否重要

sliding-window variant 保存 embedding 和最近 8 个 source，loss 为 1.764，接近 baseline；Block $S=4$ 为 1.746。

这支持：

> 仅扩大局部 receptive field 不足；保留跨长 depth range 的 anchor/source 更重要。

但 sliding window 与 block summary 同时改变 source 覆盖、source 语义和归一化集合，不能把差距唯一归因于“远距离”。

---

## 13. 学到的 routing pattern

### 13.1 Figure 8 的三点观察

论文把 attention weight 在 token 上取平均后画 heatmap：

1. **locality**：大部分层仍偏向最近 source；
2. **embedding persistence**：embedding source 在深层仍获得非零权重，pre-attention 尤其明显；
3. **layer-type specialization**：pre-MLP 更依赖邻近 source，pre-attention 的 depth receptive field 更宽。

Block AttnRes 保留了这些大结构，同时权重更尖锐。

### 13.2 heatmap 能证明什么

它证明模型学到的平均权重不是完全均匀，也不是纯对角。

它不能证明：

- 不同样本真的采用不同 routing；
- off-diagonal source 对 loss 有因果贡献；
- embedding 高权重一定是在恢复被稀释的信息；
- 某个 attention sink 是有用 anchor，而不是 query/key bias；
- routing 对特定 reasoning step 有可解释对应。

### 13.3 应补的统计量

至少需要：

$$
H_l(x)
=
-\sum_i
\alpha_{i\to l}(x)
\log\alpha_{i\to l}(x)
$$

的 token/sample 分布，以及：

$$
\operatorname{Var}_x
\left[
\alpha_{i\to l}(x)
\right].
$$

若只看：

$$
\mathbb E_x[\alpha_{i\to l}(x)],
$$

无法区分：

- 每个 token 都用同一组平均权重；
- 不同 token 选择完全不同 source，但平均后相同。

---

## 14. AttnRes 是否偏好更深的模型

### 14.1 控制实验

论文固定：

$$
\text{training FLOPs}
\approx
6.5\times10^{19},
$$

$$
\text{activated parameters}
\approx
2.3\times10^8,
$$

并在 25 个 depth/width/head 配置上 sweep：

$$
\frac{d_{\mathrm{model}}}{L_b}
\in
\{15,30,45,60,75\},
$$

$$
\frac{H}{L_b}
\in
\{0.3,0.4,0.5,0.6,0.7\}.
$$

### 14.2 结果

- AttnRes 在 25 个配置上全部优于 baseline，loss 低 0.019–0.063。
- 两者最优 $H/L_b$ 都约为 0.3。
- baseline 最优点约在 $d_{\mathrm{model}}/L_b=60$，loss 1.847。
- AttnRes 最优点约在 $d_{\mathrm{model}}/L_b=45$，loss 1.802。

在固定参数预算下，更低 $d_{\mathrm{model}}/L_b$ 对应更深、更窄的模型。

### 14.3 正确解释

证据支持：

> AttnRes 改变了该受控实验下的 quality-optimal depth/width allocation。

不支持：

> 部署时应该无条件把模型做得更深。

更深模型通常增加 sequential latency。论文自己也明确把 Figure 7 定位为诊断，而不是 deployment recommendation。

---

## 15. 用 depth mixing matrix 统一看 residual 方法

### 15.1 统一定义

定义：

$$
h_l
=
\sum_{i=0}^{l-1}
M_{i\to l}v_i,
$$

其中：

$$
M\in\mathbb R^{L\times L}
$$

是 causal lower-triangular depth mixing matrix。

### 15.2 方法对照

| 方法 | $M_{i\to l}$ 如何产生 | source access | 结构 |
|---|---|---|---|
| Standard residual | 固定为 1 | 通过累计 state 间接访问 | rank-1 semiseparable |
| Highway | gate 连乘 | immediate recurrent state | rank-1 semiseparable，dynamic |
| HC/mHC | $\beta_i^\top A_{i+1\to l}^{\times}\alpha_l$ | $m$ 条 recurrent streams | rank-$m$ semiseparable |
| DenseFormer | learned static scalar | 历史 layer outputs | dense but input-independent |
| Full AttnRes | $\alpha_{i\to l}(x)$ | 所有历史 source | generic dense / rank-$L$ |
| Block AttnRes | block 内共享 source/weight | 历史 blocks + partial | effective rank 介于 $N$ 与 $N+S$ |

### 15.3 HC/mHC 的展开

对多流状态：

$$
H_l
=
H_{l-1}A_l
+
f_{l-1}
\left(
H_{l-1}\alpha_{l-1}
\right)
\beta_{l-1}^{\mathsf T},
$$

历史 source $i$ 到层 $l$ 的有效权重为：

$$
M_{i\to l}
=
\beta_i^{\mathsf T}
A_{i+1\to l}^{\times}
\alpha_l.
$$

其中：

$$
A_{i+1\to l}^{\times}
=
\prod_{k=i+1}^{l}A_k.
$$

论文据此把：

- $\alpha_l$ 类比为 depth query；
- $\beta_i$ 类比为 source key；
- 中间 transition product 类比为 depth-relative positional operator；
- $m$ 条 streams 类比为 depth recurrence 的 state expansion。

HC/mHC 因此被解释为 matrix-valued state 的 depth-wise linear attention；AttnRes 是显式 depth-wise softmax attention。

### 15.4 这套视角的价值

它把“连接方法”的比较从名称变成三个问题：

1. depth mixing matrix 的可表达集合是什么？
2. 权重是否依赖输入？
3. dense access 的 state/I/O/communication 怎样实现？

它也揭示：

> 扩大 state rank、允许 dense source access、使用 softmax competition 是三个不同设计轴。

mHC 改第一轴并约束 transition；AttnRes 主要改第二、第三轴。

---

## 16. 贡献账本

| 可检验 claim | 类型 | 论文证据 | 强度 | 没有建立什么 |
|---|---|---|---|---|
| residual 可视为 depth aggregation | framing / synthesis | §2、§6、Fig. 9 | strong | 不是唯一可能的理论视角 |
| Full AttnRes 用 softmax depth routing 改善 loss | mechanism + empirical | Eq. 1–4、Table 2、Table 4 | moderate–strong | 未跨公开数据、多 seed 验证 |
| Block 约 8 个 source 保留大部分收益 | system/architecture | §3.2、Fig. 6、Table 2 | moderate | 不是所有 depth/架构的通用最优值 |
| content dependence 很重要 | empirical mechanism | Table 4 | moderate | ablation 仍混有参数化差异 |
| RMSNorm 防止 magnitude bias | design + empirical | Eq. 2、Table 4 | moderate | 没直接测 logit 的 norm/semantic 分解 |
| AttnRes 缓解 PreNorm dilution | empirical association | Fig. 5(b) | moderate | 没有全局 stability theorem |
| gradient 更均匀 | empirical association | Fig. 5(c) | moderate | 没有因果归因或多 run 统计 |
| Block 等价于 baseline 的 $1.25\times$ compute | scaling extrapolation | Fig. 4 | moderate–weak | 缺 fit uncertainty 和直接配对实验 |
| PP training overhead 小于 4% | system measurement | §4.1 | weakly auditable | 硬件和配置不完整 |
| inference overhead 小于 2% | system measurement | §4.2 | weakly auditable | workload 和 latency 分解不完整 |
| 48B/3B activated 全任务不降 | empirical | Table 3 | moderate | 无 error bar、污染审计或公开 checkpoint |

---

## 17. 这篇论文最有价值的 insight

### 17.1 Residual path 本质上是一个 memory policy

**[综合判断]**

标准 residual 的：

$$
h_l=h_{l-1}+v_{l-1}
$$

看似只是一次 add，展开后却规定了：

- memory 写入：每个新 delta 永久进入累计 state；
- memory 读取：所有历史 delta 固定权重 1；
- eviction：没有；
- retrieval：不能按内容选择；
- compression：所有历史被压进一个 $d$ 维向量。

AttnRes 的贡献不是“把加法换成注意力”这么表面，而是把 residual path 明确成可学习的 memory read policy。

### 17.2 softmax 同时完成选择与尺度归一化

**[综合判断]**

softmax 带来两个不可分的变化：

1. input-dependent relative selection；
2. 权重和从 $l$ 变成 1。

因此 loss 和 magnitude 改善不能只归因于“检索更聪明”。可能还有 normalized aggregation 本身的作用。

一个关键缺失 ablation 是：

$$
h_l
=
c_l
\sum_i\alpha_{i\to l}v_i,
$$

分别比较：

- $c_l=1$；
- $c_l=l$；
- learned $c_l$；
- 与 baseline RMS 匹配的 $c_l$。

这能拆开 routing 与 scale control。

### 17.3 “表达力最强”与“系统最优”不是同一个点

Full AttnRes loss 最好；dynamic query 更好；但作者最终选择 Block + static query。

这说明方法真实目标是：

$$
\text{quality}
\times
\text{memory feasibility}
\times
\text{pipeline compatibility}
\times
\text{inference latency}.
$$

论文最成熟的部分不是单独的 attention 公式，而是承认大模型连接方法必须与 memory hierarchy 和 pipeline schedule 一起设计。

### 17.4 block summary 是一种结构化压缩

Block AttnRes 把同一 block 内的多个 source 映射成：

$$
b_n=\sum_{j\in\mathcal B_n}v_j.
$$

这不是简单丢掉历史，而是施加“同一 block 内共享一个 routing coefficient”的结构约束。

在 depth mixing matrix 中，相同 block 的列被绑定。它既降低 rank/自由度，也可能成为 regularization。Figure 6 的平滑退化与 Figure 8 的更尖锐权重都与这个解释相容，但还不能证明因果。

### 17.5 连接方法应同时记录 source semantics

仅写“保存 $N$ 个 states”不够。必须回答：

- state 是 cumulative hidden、branch delta、block delta 还是 normalized key？
- read 是 replacement 还是 additive augmentation？
- block boundary 是否 reset？
- final output 怎样 collapse？

后续工作对原 AttnRes 的不同解释正说明：如果 source semantics 没有被 implementation contract 固定，公式、伪代码和复现可能实现成不同算法。

---

## 18. 论文没有充分回答的问题

### 18.1 初始化不是 baseline-preserving

zero query 给均匀平均而非单位和。论文只报告从 scratch pretraining，没有评估：

- 从普通 PreNorm checkpoint 转换的 loss spike；
- 是否能 exact function-preserving 初始化；
- average scale 对 gain 的独立贡献；
- final read 的尺度处理。

### 18.2 Full 公式与 Block pseudocode 的 source 语义需要官方代码确认

Full AttnRes 的 Eq. 3 明确写：

$$
v_i=f_i(h_i),
$$

即 branch/sub-layer output。

Block 的 Eq. 5 写：

$$
b_n=\sum_{j\in\mathcal B_n}f_j(h_j),
$$

即 block delta。

但 Figure 2 pseudocode 使用：

```python
partial_block = hidden_states
...
blocks.append(partial_block)
partial_block = None
...
partial_block = partial_block + attn_out if partial_block is not None else attn_out
```

`hidden_states` 在调用边界上究竟表示完整 residual、block-local partial，还是已经 replacement 后的 state，需要外部模型调用链才能完全确定。

官方仓库没有可运行实现，因此当前只能把公式作为 formal contract，把 pseudocode 作为 operational sketch。

### 18.3 没有 routing 的因果验证

Figure 8 是平均 heatmap。缺少：

- 强制屏蔽高权重 source；
- 用低权重 source 替换；
- source permutation；
- routing weight shuffle；
- per-token entropy 与 task difficulty 的关系；
- pre-attention 与 pre-MLP routing 的 controlled intervention。

### 18.4 stability 只有观察，没有 theorem

论文展示 output magnitude 和 gradient magnitude 更均匀，但没有给：

- Jacobian spectral analysis；
- deep composition bound；
- softmax temperature/norm 上界；
- source norm 的递推界；
- block size 对稳定性的理论关系；
- mixed-precision error 分析。

### 18.5 scaling evidence 的外部有效性窄

所有核心点都基于同一 Kimi Linear / MoE family。尚未建立：

- dense Transformer；
- encoder-only / encoder-decoder；
- ViT / diffusion；
- 不同 optimizer；
- 不同 normalization；
- post-training；
- long-context 从 scratch；
- serving workload。

### 18.6 统计报告不足

Tables 2–4 都没有：

- seeds；
- standard deviation；
- confidence interval；
- significance test；
- checkpoint selection protocol。

像 Full 1.692、Block 1.693、mHC 1.694 这种 0.001 量级差距，在没有 run-to-run variance 时不能排序得过于确定。

### 18.7 数据与评测不可完全审计

论文沿用 Kimi Linear 数据 recipe，但未公开训练数据组成。无法检查：

- benchmark contamination；
- 语言比例；
- reasoning/code 数据权重；
- 1T 与 400B 阶段的分布变化；
- baseline 和 AttnRes 是否使用完全相同 token stream。

### 18.8 system claims 缺复现 artifact

官方仓库没有：

- PP cache implementation；
- two-phase kernel；
- online merge kernel；
- TP prefill integration；
- profiler trace；
- benchmark command；
- hardware manifest。

因此论文系统设计可用于写 implementation proposal，还不能当作已公开可复现的 production implementation。

---

## 19. 后续证据：论文之后出现了什么

### 19.1 Delta Attention Residuals 提出的挑战

**[后续证据]** [Delta Attention Residuals](https://arxiv.org/abs/2605.18855v1)（2026-05-13）把原 AttnRes 解释为对 cumulative hidden states 做 replacement routing，并报告：

- deep layer routing weight 变得低对比、接近均匀；
- Qwen3-style 实验中其 AttnRes reimplementation 在更大规模可能退化；
- additive delta routing 在其 220M–7.6B 实验中优于 baseline 和该 reimplementation。

但这项后续证据有一个重要边界：

- 原 AttnRes Eq. 3 把 Full source 写成 $f_i(h_i)$，形式上已经是 sublayer delta；
- Delta AttnRes 作者明确说明原实现未公开，他们使用的是基于论文描述的 faithful reimplementation；
- 分歧可能来自 Block pseudocode 的 `partial_block`/reset 语义，而不是 Full 公式本身。

**[综合判断]** 后续论文不能直接推翻原论文的所有结果，却暴露了一个非常实际的问题：formal source definition 与 released pseudocode 必须由官方调用链和 tests 固定，否则“复现 AttnRes”可能复现成不同算法。

### 19.2 Low-Rank Attention Residuals 的方向

**[后续证据]** [Low-Rank Attention Residuals](https://arxiv.org/abs/2607.09694v1)（2026-06-19）把 routing key 与 residual value 解耦：

$$
k_i\in\mathbb R^r,
\qquad
v_i\in\mathbb R^d,
\qquad
r\ll d.
$$

它的核心主张是 depth routing 不需要用完整 $d$ 维表示计算 score。Projected 与 sliced key 设计表明 routing bandwidth/compute 可能进一步下降。

这正对应原论文未拆开的设计轴：

- source 内容需要 full width；
- routing key 未必需要 full width；
- value cache 与 key cache 可以有不同 layout/dtype。

### 19.3 当前应怎样看原论文

截至 2026-07-23：

- 原论文建立了 depth softmax routing 的强 baseline 和系统问题定义；
- 后续工作已经开始质疑 source semantics、replacement update 和 full-width key；
- 这些后续结果仍是较新的 preprint，架构和数据也不同；
- 最可靠的结论不是“原方案已经最终定型”，而是“depth routing 已成为一个可拆分、可实验的设计空间”。

---

## 20. 论文之外的高信息量扩展

### 20.1 扩展一：source-semantics 四象限实验

**标签：[扩展假设]**

同时控制两个轴：

| | replacement read | additive read |
|---|---|---|
| cumulative hidden source | A | B |
| sublayer/block delta source | C | D |

统一：

- model/data/compute；
- query/key parameterization；
- block size；
- normalization；
- initialization；
- source 数。

**预测**：

- additive update 更容易保持 baseline initialization；
- delta source 可能有更高 routing contrast；
- 原论文 Full Eq. 3 更接近 C，而公开 Block pseudocode 的复现可能落在 A/C 的边界。

**证伪条件**：

- 在多个 scale 上，四种方案差异均小于 run variance；
- routing entropy 与质量没有稳定关系。

**最小验证**：

- 100M–300M dense Transformer；
- 3 seeds；
- public data；
- 记录 loss、PPL、source cosine similarity、routing entropy、gradient RMS；
- 在相同 source budget 下比较。

### 20.2 扩展二：exact baseline-preserving router

**标签：[扩展假设]**

保留标准 residual state：

$$
\tilde h_l
=
h_{l-1}
+
f_{l-1}(h_{l-1}),
$$

再加 zero-initialized depth retrieval：

$$
h_l
=
\tilde h_l
+
g_l
\sum_i\alpha_{i\to l}v_i,
\qquad
g_l(0)=0.
$$

初始化时：

$$
h_l=\tilde h_l
$$

exactly。

**预测**：

- 从 pretrained checkpoint 转换无初始 loss jump；
- from-scratch 训练可能不如 replacement softmax 那样强地控制 magnitude；
- gate 学习速度决定 router 是否真正启用。

**证伪条件**：

- gate 长期接近 0；
- 或 exact initialization 虽稳定但最终质量显著弱于 replacement。

**最小验证**：

- 对同一 checkpoint 比较 conversion step 0 loss；
- 记录 gate、router contribution RMS 和后续 10K-step recovery。

### 20.3 扩展三：低维 key、全维 value

**标签：[扩展假设，已有后续 primary evidence]**

令：

$$
k_i=P_kv_i,
\qquad
P_k\in\mathbb R^{r\times d},
\qquad
r\ll d,
$$

但仍保存：

$$
v_i\in\mathbb R^d.
$$

**预测**：

- routing score compute 从 $O(d)$ 降到 $O(r)$；
- key cache bytes 降低；
- value cache 仍是主要 memory；
- 适当 $r$ 下质量接近 full-width key。

**证伪条件**：

- 小 $r$ 导致 routing 接近静态或 uniform；
- key projection I/O 抵消 compute 收益；
- routing 与 value semantic 严重错位。

**最小验证**：

$$
r\in\{8,16,32,64,128,d\},
$$

同时报告 validation loss、routing entropy、key/value cache bytes 和 end-to-end latency。

### 20.4 扩展四：用干预验证 depth routing

**标签：[扩展假设]**

对训练好的模型执行：

1. mask top-1 source；
2. mask random source；
3. shuffle weights but preserve histogram；
4. replace per-token routing with token-average routing；
5. freeze routing to diagonal；
6. freeze routing to embedding + local source。

若 input-dependent skip 真有因果作用，应观察到：

$$
\Delta\mathcal L_{\mathrm{top1-mask}}
>
\Delta\mathcal L_{\mathrm{random-mask}},
$$

且：

$$
\Delta\mathcal L_{\mathrm{token-average}}
>
0.
$$

这比 heatmap 更能区分：

- 真正的 conditional retrieval；
- 静态 learned topology；
- attention sink；
- 单纯尺度正则化。

### 20.5 扩展五：quality–bytes Pareto frontier

**标签：[扩展假设]**

不要只 sweep block 数，应联合 sweep：

$$
(N,S,r,\text{source dtype},\text{cache policy}).
$$

优化目标：

$$
\min
\left(
\mathcal L,
\operatorname{activation\ bytes},
\operatorname{PP\ payload},
\operatorname{HBM\ I/O},
\operatorname{latency}
\right).
$$

报告 Pareto frontier，而不是只给单个“约 8 blocks”结论。

---

## 21. 最小 CPU reference 实现路线

### 21.1 第一阶段：Full AttnRes exact math

实现：

```python
def full_attn_res(
    sources: list[Tensor],  # each [B, T, D]
    query: Tensor,         # [D]
    eps: float,
) -> Tensor:
    ...
```

检查：

- stack axis 是 depth/source；
- softmax axis 是 depth/source；
- RMSNorm 只用于 key，value 保留原值；
- query shape 不含 batch/token；
- output 是 `[B,T,D]`。

### 21.2 Shape tests

覆盖：

$$
B\in\{1,2\},
\quad
T\in\{1,3,7\},
\quad
d\in\{2,8\},
\quad
J\in\{1,2,5\}.
$$

不变量：

$$
\alpha.shape=[J,B,T],
$$

$$
\sum_{j=1}^{J}\alpha_{j,b,t}=1,
$$

$$
h.shape=[B,T,d].
$$

### 21.3 zero-query test

若：

$$
w=0,
$$

应有：

$$
\alpha_j=\frac1J,
$$

$$
h=\frac1J\sum_jv_j.
$$

同时写一个 negative test，确认它通常不等于 standard residual sum。

### 21.4 convex-hull test

对 scalar source：

$$
\min_i v_i
\le
h
\le
\max_i v_i.
$$

对 vector source，验证：

$$
\|h\|_2
\le
\max_i\|v_i\|_2
+
\epsilon.
$$

### 21.5 Block state-machine test

显式状态：

```text
completed_blocks: list[[B,T,D]]
partial_block: Optional[[B,T,D]]
layer_in_block: int
```

逐层记录：

```text
read sources
→ attention read
→ sublayer output
→ partial update
→ optional boundary commit/reset
```

对 $L=6,N=2,S=3$ 手工列出每层 source list，防止 boundary off-by-one。

### 21.6 two-phase equivalence

随机切分 source：

$$
V=[V_1;V_2].
$$

比较：

$$
h_{\mathrm{direct}}
=
\operatorname{softmax}(qK^\top)V
$$

与 online merge。

测试：

- `float64` exactness；
- `float32` tolerance；
- 极大正负 logits；
- source group 为空；
- 单 source；
- unequal group sizes。

### 21.7 gradcheck

对：

- query；
- source values；
- RMSNorm weight；
- online merge；

使用 `torch.autograd.gradcheck`。

还应比较：

$$
\nabla h_{\mathrm{direct}}
\approx
\nabla h_{\mathrm{two-phase}}.
$$

### 21.8 formula–pseudocode ambiguity test

分别实现：

1. Eq. 3/5 的 delta-source contract；
2. released pseudocode 的 `partial_block` contract；

在相同小例子上打印 source contents。若两者不同，不应让同一个类名静默代表两种语义。

---

## 22. 分布式实现 contract

### 22.1 Tensor Parallel

若 hidden 维按 TP sharded：

- value source 可能是 `[B,T,d/P]` local shard；
- RMSNorm 需要全 hidden 维平方和 reduction；
- query 需按相同 hidden layout sharded 或 replicated；
- logit dot product 需要跨 shard sum；
- depth softmax 本身不跨 TP，因为 source 维通常 replicated；
- weighted value output 可保持 hidden-sharded。

最小 collective contract：

```text
local RMS squares
→ all-reduce hidden statistic
→ local normalized key
→ local query-key dot
→ all-reduce logits
→ source-axis softmax locally
→ local weighted value
```

若 framework 已把 RMSNorm 与 projection 融合，collective placement 可能不同，不能从论文直接假设。

### 22.2 Sequence/Context Parallel

depth routing 对 token 独立，因此 source tensor 可以沿 $T$ sharded。

但必须确认：

- attention/MLP branch output 回到 residual state 时的 layout；
- block cache 与当前 partial 是否同 layout；
- context-parallel attention 输出是否已 reduce-scatter；
- activation checkpoint 重算时 source shard 是否仍可访问。

### 22.3 Pipeline Parallel

pipeline payload 不再只是：

$$
[B,T,d],
$$

而可能是：

$$
\text{new partial}
+
\text{incremental completed blocks}.
$$

应为 payload 写 schema：

```text
microbatch_id
source_block_ids
source_tensors
partial_block
cache_generation
dtype/layout
```

并测试：

- non-interleaved 1F1B；
- interleaved 1F1B；
- uneven block/stage boundaries；
- last short block；
- forward/backward cache lifetime；
- pipeline flush；
- virtual stage wraparound。

### 22.4 Data Parallel / FSDP

pseudo-query 和 RMSNorm 参数随层 sharding。

需要确认：

- query 在 forward 前是否 all-gather；
- completed block activation 不应被当作 parameter；
- FSDP activation checkpoint 与 block cache 是否重复保存；
- no-sync / gradient accumulation 不影响 cache key；
- optimizer state 增量虽小，但参数注册必须稳定。

### 22.5 Mixed precision

建议 contract：

| Operation | 推荐 accumulation |
|---|---|
| source/value storage | BF16/FP16，按模型策略 |
| RMS square sum | FP32 |
| query-key dot | FP32 accumulate |
| softmax max/LSE | FP32 |
| online merge stats | FP32 |
| weighted value output | FP32 accumulate 后 cast |
| block partial accumulation | 至少验证 BF16 与 FP32 accumulator 差异 |

这不是论文规定，而是 correctness-first reference 应明确验证的边界。

### 22.6 Recomputation

必须区分：

- sublayer activations 可重算；
- 历史 source 是否被后续层依赖；
- source 是保存还是重算；
- completed block 是否跨 PP stage；
- softmax intermediates 是否 checkpoint。

Full AttnRes 在 vanilla training “没有额外 activation memory”的说法，只在原本就保存所有层输出时成立。启用 selective recomputation 后，历史 source 不能像普通 residual 那样随意释放。

---

## 23. 官方 artifact 能提供什么

截至 commit `85e22310fe5ee860b4a023de312d791de8a5a5e6`，官方仓库提供：

- technical report PDF；
- README overview；
- Block AttnRes PyTorch-style pseudocode；
- figures/assets。

没有：

- package/code implementation；
- config；
- training loop；
- custom kernel；
- distributed integration；
- checkpoint；
- evaluation script；
- reproducibility manifest。

因此：

- 论文公式可以作为 reference math；
- pseudocode 可以作为接口草图；
- 不能从该仓库 pin 出 production forward/backward 调用链；
- 不能声称论文系统优化已经以开源代码验证。

---

## 24. 证据账本

| 主题 | 主要来源 | 最可靠结论 | 证据缺口 |
|---|---|---|---|
| 方法定义 | Eq. 1–6、Fig. 1–2 | depth softmax 替代固定累加；Block 压缩 source | pseudocode 调用边界不完整 |
| two-phase exactness | Algorithm 1、§4.2 | online softmax 可精确合并两组 source | 无公开 kernel/test |
| PP cache | Eq. 7–8、Fig. 3 | incremental cache 可减少重复传输 | 无硬件和 trace |
| inference I/O | Table 1、Appendix B | 典型符号模型下 Block 为 5.5d | 不等于端到端 latency |
| scaling | Table 2、Fig. 4 | 五个点中 AttnRes 都优于 baseline | 无 seed/fit uncertainty/data |
| 48B quality | Table 3、Fig. 5 | 表中任务不降，部分 reasoning/code 提升大 | 无公开 checkpoint/数据 |
| component ablation | Table 4、Fig. 6 | dynamic key、softmax、RMSNorm、远程 source 均有局部证据 | 单规模、无方差 |
| architecture allocation | Fig. 7 | 受控 sweep 中 optimum 更深窄 | 不含 deployment latency |
| learned routing | Fig. 8 | 平均 pattern 有 locality、embedding anchor、layer-type 差异 | 无 per-token/causal analysis |
| unified theory | Table 5、Fig. 9、§6 | residual variants 可写成 depth matrix | 主要是解释框架 |
| 后续挑战 | Delta AttnRes v1 | replacement/cumulative 复现可能 routing collapse | 与原 Eq. 3 source 定义有分歧 |
| key compression | LR-AttnRes v1 | routing key 可尝试低维化 | 独立后续 preprint |

---

## 25. 推荐阅读顺序

若只花 20 分钟：

1. Abstract + Figure 1：理解 standard / Full / Block 的差异。
2. Eq. 1–6：确认 source、query、key、value 和 block partial。
3. Table 2 + Table 4：看 scaling 与关键消融。
4. Figure 5 + Figure 8：看作者对 magnitude、gradient 和 routing pattern 的解释。
5. §4 + Appendix B：理解为什么模型公式简单但系统实现不简单。

若要实现：

1. Eq. 2–4；
2. Figure 2 pseudocode；
3. Algorithm 1；
4. Eq. 7–8；
5. Table 1；
6. Appendix B；
7. 再对照后续 Delta AttnRes 对 source/reset 的不同解释。

若要评审：

1. zero-query 是否 baseline-preserving；
2. Table 2 是否有统计可靠性；
3. Figure 8 是否足以支持机制解释；
4. system overhead 是否可审计；
5. formal equations 与 pseudocode 是否定义同一个 state machine。

---

## 26. 最终评价

Attention Residuals 最重要的价值，是把 residual connection 从“默认存在的一次加法”提升为一个可分析的 depth memory/routing mechanism。

它给出了一个极其直接的设计：

$$
\text{fixed depth sum}
\quad\longrightarrow\quad
\text{token-dependent depth softmax}.
$$

Full AttnRes 展示了表达力上限，Block AttnRes 展示了系统折中，two-phase computation 和 cross-stage cache 则说明作者确实面对了大模型训练中的 memory/communication 约束。五规模 scaling、48B/3B activated 主实验和组件消融共同构成了相当完整的第一轮证据。

但论文还没有把方法闭环到“可独立复现的成熟实现”：

- 初始化不是 baseline function；
- stability 主要是 empirical；
- source semantics 在公式与 pseudocode 间需要更严格 contract；
- 统计和数据透明度不足；
- system 数字缺少公开 artifact；
- 后续工作已经暴露 replacement/additive、cumulative/delta、full-width/low-rank key 等未定设计轴。

因此当前最合适的判断是：

> **AttnRes 不是 residual connection 的最终答案，而是把 depth routing 变成一个清晰研究对象的强起点。**

对于本项目，下一步不应直接追求训练复现，而应先在 CPU reference 中固定：

1. source 到底存什么；
2. read 是 replacement 还是 additive；
3. block boundary 怎样 commit/reset；
4. zero initialization 恢复什么 special case；
5. two-phase path 是否与 direct path 前后向一致；
6. TP/PP payload 与 cache lifetime 是什么。

这些 contract 明确后，才适合进入真实 framework 的 distributed implementation 设计。
