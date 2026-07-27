---
title: "从固定加法到深度记忆"
slug: "residual-topology-four-papers"
description: "把 residual path 视为 depth memory，以统一 operational contract 比较 HC、mHC、AttnRes 与 xHC 的状态、检索、稳定性和系统成本。"
topic: "residual"
status: "published"
date: "2026-07-25"
updated: "2026-07-27"
cutoff: "2026-07-25"
audience: "熟悉 Transformer、PreNorm、Attention/MLP block 与基本线性代数的研究和工程同学"
duration: 50
slideCount: 33
source:
  repository: "J-shang/J-shang.github.io"
  path: "src/data/slides/residual/residual-topology-four-papers/index.md"
  url: "https://github.com/J-shang/J-shang.github.io/blob/main/src/data/slides/residual/residual-topology-four-papers/index.md"
  revision: "main"
  syncedAt: "2026-07-27"
  contentHash: "sha256:101ad820ca872011818fb69576cf5ac72d5f9859830726753e692390635c6ea4"
  manifest: "local-residual-slides"
  dirty: false
  managed: false
---

<!-- layout: title -->
<!-- section-header: cover || ACADEMIC SHARING · 60 MIN || 01 / 29 -->

# 从固定加法到深度记忆

Hyper-Connections · Manifold-Constrained HC · Attention Residuals · Expanded HC

![Hyper-Connections 机制总览](./assets/hc-fig2-overview.png)
![mHC 机制总览](./assets/mhc-fig1-overview.png)
![Attention Residuals 机制总览](./assets/attnres-fig1-overview.png)
![xHC 机制总览](./assets/xhc-fig3-overview.png)

<!-- notes:
0.5 分钟。只报题目、四篇论文与分享目标：把 residual connection 当作沿 depth 运行的 memory policy。

[Sources]
- https://arxiv.org/abs/2409.19606v3
- https://arxiv.org/abs/2512.24880v2
- https://arxiv.org/abs/2603.15031v1
- https://arxiv.org/abs/2607.14530v1
-->

---

<!-- layout: figure -->
<!-- section-header: depth || 01 · 深度记忆 / DEPTH MEMORY | MEMORY POLICY · 1/4 || 02 / 29 -->

## 一次加法其实规定了 memory policy

令 $v_0=h_1,\ v_i=f_i(h_i)$：

$$
h_l=\sum_{i=0}^{l-1}v_i
$$

<div class="slide-cards slide-cards--2">
  <div class="slide-card slide-card--state"><strong>state</strong>一条累计 hidden state</div>
  <div class="slide-card slide-card--read"><strong>read</strong>下一层只能读取整条 state</div>
  <div class="slide-card slide-card--write"><strong>write</strong>加上 Attention/MLP 新增量</div>
  <div class="slide-card slide-card--caveat"><strong>forget</strong>没有显式遗忘或 source retrieval</div>
</div>

![Standard Residuals 把 embedding 与历史 branch delta 累加到同一条 residual path](./assets/attnres-fig1-standard.png)

> **Derived** · Attention Residuals, arXiv:2603.15031v1, Fig. 1(a), PDF p.1；Eq. 1–2

<!-- notes:
1.5 分钟。先问“如果 residual 是 memory，它的 eviction policy 是什么？”答案是没有显式 eviction，也不能单独检索 source。

[Sources]
- https://arxiv.org/abs/2603.15031v1
- https://github.com/J-shang/residual
-->

---

<!-- layout: comparison -->
<!-- section-header: depth || 01 · 深度记忆 / DEPTH MEMORY | FIXED DEPTH MIXING · 2/4 || 03 / 29 -->

## 标准 residual 是固定 depth mixing

![Attention Residuals 论文中的 Standard Residuals 基线](./assets/attnres-fig1-standard.png)
![标准 residual 的 causal depth mixing 矩阵在可见区域全部使用固定系数 1](./assets/fixed-depth-mixing.svg)

$$
h_l=\sum_{i=0}^{l-1}\color{#1258ff}{1}\cdot v_i
$$

`1 = 每层以相同权重读取该历史 source`。direct path 帮助信息与梯度跨层传播，也限制了按内容选择历史 source 的能力。

> **Reported + Derived** · Attention Residuals, arXiv:2603.15031v1, Fig. 1(a), PDF p.1；Eq. 1–2

<!-- notes:
1.5 分钟。identity path、identity initialization 和训练后学成 identity 是三件事；本页只建立共同起点。

[Sources]
- https://arxiv.org/abs/2603.15031v1
-->

---

<!-- layout: figure -->
<!-- section-header: depth || 01 · 深度记忆 / DEPTH MEMORY | OPERATIONAL CONTRACT · 3/4 || 04 / 29 -->

## 用 operational contract 比较方法

![连接方法的八字段 operational contract：state、read、branch、transport、write、granularity、initialization、system cost](./assets/operational-contract-fit.svg)

<div class="slide-flow" aria-label="连接方法的运行流程">
  <span class="slide-flow__step">stored state</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">read</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">branch</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">write</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">next state</span>
</div>

同一张 contract 必须能落到 reference code、correctness test 与 distributed layout。

<!-- notes:
1.5 分钟。后续每种方法都回答同一组问题：state、read、write、transport、granularity、initialization、composition、cost。

[Sources]
- https://github.com/J-shang/residual
-->

---

<!-- layout: comparison -->
<!-- section-header: depth || 01 · 深度记忆 / DEPTH MEMORY | RESEARCH MAP · 4/4 || 05 / 29 -->

## 四篇论文是一条设计压力链

<div class="slide-pressure-map" aria-label="四篇论文的问题驱动关系，不是发表时间线">
  <div class="slide-pressure-card slide-pressure-card--baseline"><strong>Standard residual</strong>固定单流累计 state</div>
  <span class="slide-pressure-arrow slide-pressure-arrow--hc">→</span>
  <div class="slide-pressure-card slide-pressure-card--hc"><strong>HC</strong>怎样扩大 state topology？</div>
  <span class="slide-pressure-arrow slide-pressure-arrow--mhc">→</span>
  <div class="slide-pressure-card slide-pressure-card--mhc"><strong>mHC</strong>怎样让深层 transport 稳定？</div>
  <span class="slide-pressure-arrow slide-pressure-arrow--xhc">→</span>
  <div class="slide-pressure-card slide-pressure-card--xhc"><strong>xHC</strong>怎样扩大 N 而不让更新成本失控？</div>
  <span class="slide-pressure-arrow slide-pressure-arrow--attnres">↘</span>
  <div class="slide-pressure-card slide-pressure-card--attnres"><strong>AttnRes</strong>怎样检索指定历史 source？</div>
</div>

<div class="slide-boundary">AttnRes 是 baseline 的另一条问题分支；这里表达设计压力关系，不是代码继承或发表时间线。</div>

> **Derived** · HC Fig. 2 · mHC Fig. 1 · Attention Residuals Fig. 1 · xHC Fig. 3

<!-- notes:
0.5 分钟。快速预告四个章节，告诉听众后面始终回到同一个 contract。

[Sources]
- https://arxiv.org/abs/2409.19606v3
- https://arxiv.org/abs/2512.24880v2
- https://arxiv.org/abs/2603.15031v1
- https://arxiv.org/abs/2607.14530v1
-->

---

<!-- layout: figure -->
<!-- section-header: hc || 02 · HC / MULTI-STREAM | STATE EXPANSION · 1/6 || 06 / 29 -->

## HC 扩张的是 residual state

![Hyper-Connections Figure 2：从标准 residual 到 hyper-connections](./assets/hc-fig2-overview.png)

$$
[B,S,d]\longrightarrow[B,S,n,d],
\qquad \text{branch}:d\to d
$$

`B=batch` · `S=sequence` · `n=streams` · `d=hidden size`

> **Reported** · Hyper-Connections, arXiv:2409.19606v3, Fig. 2, PDF p.2；§2

<!-- notes:
2 分钟。用 n=2 的最小例子说明：扩的是跨 depth 持久 state，不是 FFN hidden size。

[Sources]
- https://arxiv.org/abs/2409.19606v3
-->

---

<!-- layout: figure -->
<!-- section-header: hc || 02 · HC / MULTI-STREAM | READ · TRANSPORT · WRITE · 2/6 || 07 / 29 -->

## 三个矩阵分别回答读、传输和写

![Hyper-Connections Figure 2(b)：两流 read、transport 与 write](./assets/hc-fig2b-hyper-connections.png)

<div class="slide-cards slide-cards--3">
  <div class="slide-card slide-card--read"><strong><code>A_m:[n,1]</code></strong><code>n→1</code>，把多流读给 branch</div>
  <div class="slide-card slide-card--transport"><strong><code>A_r:[n,n]</code></strong>旧 state 从 source 路由到 target</div>
  <div class="slide-card slide-card--write"><strong><code>B:[1,n]</code></strong><code>1→n</code>，把 branch output 写回</div>
</div>

$$
u_l=H_l^\top A_m,\qquad
H_{l+1}=A_r^\top H_l+B^\top y_l^\top
$$

`A_r[i,j] = source i → target j`

> **Reported** · Hyper-Connections, arXiv:2409.19606v3, Fig. 2(b), PDF p.2；Eq. 1–8

<!-- notes:
1.5 分钟。先说 shape，再说 entry 的 source→target 含义，最后解释公式里的转置。DHC 只改变系数如何生成，不改变三个 operational roles。

[Sources]
- https://arxiv.org/abs/2409.19606v3
-->

---

<!-- layout: figure -->
<!-- section-header: hc || 02 · HC / MULTI-STREAM | ONE-LAYER UPDATE · 3/6 || 08 / 29 -->

## HC 的一层状态更新

![Hyper-Connections Figure 2(b)：围绕同一 branch 的多流状态更新](./assets/hc-fig2b-hyper-connections.png)

<div class="slide-flow" aria-label="HC 一层状态更新时间顺序">
  <span class="slide-flow__step">① read one d-vector</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">② Attention / MLP</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">③ carry old state + write output</span>
</div>

$$
\begin{aligned}
u_l &= H_l^\top A_m,\\
y_l &= F_l(\operatorname{Norm}(u_l)),\\
H_{l+1} &= A_r^\top H_l+B^\top y_l^\top
\end{aligned}
$$

`input n streams → output n streams`

> **Reported** · Hyper-Connections, arXiv:2409.19606v3, Fig. 2(b), PDF p.2；Eq. 1–8

<!-- notes:
1.5 分钟。A_m、A_r 与 B 不是三块独立装饰，而是围绕同一个主 branch 的完整 state machine。

[Sources]
- https://arxiv.org/abs/2409.19606v3
-->

---

<!-- layout: figure -->
<!-- section-header: hc || 02 · HC / MULTI-STREAM | DYNAMIC ROUTING · 4/6 || 09 / 29 -->

## DHC 先恢复熟悉路径再学习 token 路由

![Dynamic Hyper-Connections 的两流结构锚点](./assets/hc-fig2b-hyper-connections.png)

| 初始化时 | 训练后 |
|---|---|
| `dynamic projection = 0` | mapping 由当前 token 生成 |
| $A_r=I$ | $A_r(x)$ 学 transport |
| $B=\mathbf 1$ | $B(x)$ 学 write |
| $A_m$ 按层轮换 one-hot | $A_m(x)$ 学 read |

这是一种 **PreNorm-compatible 多流函数**；不等于每层 hidden 都与普通 PreNorm 完全相同。

> **Reported** · Hyper-Connections, arXiv:2409.19606v3, Fig. 2(b), PDF p.2；§2 与 Appendix

<!-- notes:
1.5 分钟。初始化的目的，是先给优化器一个熟悉的多流起点；不要把它误说成逐层 hidden 与 baseline 完全相同。

[Sources]
- https://arxiv.org/abs/2409.19606v3
-->

---

<!-- layout: figure -->
<!-- section-header: hc || 02 · HC / MULTI-STREAM | QUALITY & MEMORY · 5/6 || 10 / 29 -->

## HC 的算术很轻，状态并不轻

![HC 论文报告的 Training Loss 与 C4 validation loss](./assets/hc-fig1-loss-panels.png)

<div class="slide-cards slide-cards--3">
  <div class="slide-card slide-card--state"><strong>1B · DHC×4</strong>41.11→51.86 GB<br><code>+26.1%</code> · microbatch/GPU 16,384</div>
  <div class="slide-card slide-card--state"><strong>7B · DHC×4</strong>26.27→33.70 GB<br><code>+28.28%</code> · microbatch/GPU 2,048</div>
  <div class="slide-card slide-card--state"><strong>MoE · DHC×4</strong>31.59→34.65 GB<br><code>+9.7%</code> · microbatch/GPU 4,096</div>
</div>

7B dense DHC×4：V2 loss `2.581 → 2.559`，下游平均 `70.1 → 71.0`。

> **Reported** · Hyper-Connections, arXiv:2409.19606v3, Fig. 1(a–b), PDF p.1；Table 9, PDF p.16。横轴是 tokens；token efficiency ≠ wall-clock speedup。

<!-- notes:
2 分钟。三张显存卡的模型和 microbatch 设置不同，不能画进同一 Pareto curve。“1.8× faster convergence”是达到同 loss 的 token efficiency。

[Sources]
- https://arxiv.org/abs/2409.19606v3
-->

---

<!-- layout: figure -->
<!-- section-header: hc || 02 · HC / MULTI-STREAM | DEEP PRODUCT RISK · 6/6 || 11 / 29 -->

## HC 的深层风险来自 transport 连乘

![HC 的 composite forward 和 backward Amax gain 随深度放大](./assets/mhc-fig3b-composite-instability.png)

`direct path = 不经过 Attention/MLP、只搬运旧 state 的路径`

<div class="slide-flow" aria-label="深层 transport 风险的机制链">
  <span class="slide-flow__step">每层 <code>A_r</code> 无约束</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">沿 depth 连续相乘</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">signal / gradient 放大或衰减</span>
</div>

$$
P_{l\to L}=A_{r,L-1}^{\top}\cdots A_{r,l}^{\top},
\qquad (1.02)^{60}\approx3.28
$$

<div class="slide-boundary">Amax 是最大绝对行和/列和的 propagation diagnostic，不等于完整 spectral norm。曲线是现象证据，乘积才是原因解释。</div>

> **Reported + Derived** · mHC, arXiv:2512.24880v2, Fig. 3(b), PDF p.7

<!-- notes:
1.5 分钟。先用标量特例建立直觉，再说明矩阵中是 singular directions 被反复作用。上界不代表每次都取等号。

[Sources]
- https://arxiv.org/abs/2409.19606v3
- https://arxiv.org/abs/2512.24880v2
-->

---

<!-- layout: figure -->
<!-- section-header: mhc || 03 · mHC / STABLE TRANSPORT | CONSTRAINT · 1/4 || 12 / 29 -->

## mHC 用双随机 transport 控制深层乘积

![mHC Figure 1：Residual、HC 与 manifold-constrained HC 的结构对照](./assets/mhc-fig1-overview.png)

$$
\mathcal B_n=\{H\ge0\mid H\mathbf1=\mathbf1,\ \mathbf1^\top H=\mathbf1^\top\}
$$

<div class="slide-flow" aria-label="mHC 运行时 mapping 的生成">
  <span class="slide-flow__step">free logits</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">exp</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">20-step Sinkhorn</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">near <code>B_n</code></span>
</div>

`doubly stochastic = 元素非负，且每行、每列之和都是 1`

> **Reported** · mHC, arXiv:2512.24880v2, Fig. 1, PDF p.1；Eq. 6–9。$\mathcal B_n$ 是所有 $n\times n$ 双随机矩阵形成的凸多面体。

<!-- notes:
2 分钟。沿用论文 mHC 名称，但不宣称 Birkhoff polytope 处处是光滑 manifold；20 次 Sinkhorn 也是有限步近似。

[Sources]
- https://arxiv.org/abs/2512.24880v2
-->

---

<!-- layout: figure -->
<!-- section-header: mhc || 03 · mHC / STABLE TRANSPORT | GUARANTEES · 2/4 || 13 / 29 -->

## 双随机约束保的是共同模式与均值

![mHC Figure 1(c)：受约束的 pre、res 与 post mappings](./assets/mhc-fig1c-constrained.png)

<div class="slide-cards slide-cards--3">
  <div class="slide-card slide-card--transport"><strong><code>H1 = 1</code></strong>所有 streams 相同的 common mode 不变</div>
  <div class="slide-card slide-card--state"><strong><code>1ᵀH = 1ᵀ</code></strong>stream 总和与均值不漂移</div>
  <div class="slide-card slide-card--read"><strong><code>||H||₂ ≤ 1</code> + closure</strong>2-norm 不放大，乘积仍双随机</div>
</div>

<div class="slide-boundary">没有保证 <code>H = I</code>；stream differences 可以收缩；finite Sinkhorn 只近似满足约束。</div>

> **Derived** · mHC, arXiv:2512.24880v2, Fig. 1(c), PDF p.1；Eq. 6–9

<!-- notes:
2 分钟。用 common mode 和 stream mean 两个最小向量例子解释，不只读公式。non-expansive 的证明放在备份页。

[Sources]
- https://arxiv.org/abs/2512.24880v2
-->

---

<!-- layout: comparison -->
<!-- section-header: mhc || 03 · mHC / STABLE TRANSPORT | PROPAGATION EVIDENCE · 3/4 || 14 / 29 -->

## mHC 的强证据是稳定性链条

![HC 的 composite Amax gain 接近 3000，使用 log y-axis](./assets/mhc-fig3b-composite-instability.png)
![mHC 的 composite Amax gain 约维持在 1–1.6，使用 linear y-axis](./assets/mhc-fig7b-composite-stability.png)

<div class="slide-cards slide-cards--3">
  <div class="slide-card slide-card--transport"><strong>composite Amax</strong>约 <code>3000 → 1.6</code></div>
  <div class="slide-card slide-card--read"><strong>final loss gap</strong><code>-0.021</code> vs baseline</div>
  <div class="slide-card slide-card--branch"><strong>reported tasks</strong><code>8 / 8</code> 提高</div>
</div>

<div class="slide-boundary">左图是 log y-axis；右图是 0–2 linear y-axis。Amax 不是完整 operator norm；结果来自作者的单次 27B MoE 运行。</div>

> **Reported** · mHC, arXiv:2512.24880v2, Fig. 3(b), 5, 7(b), PDF p.7/12/14

<!-- notes:
2 分钟。先定义 Amax，再提醒两图 y 轴不同，最后比较各自报告的绝对峰值。单次运行不建立统计显著性。

[Sources]
- https://arxiv.org/abs/2512.24880v2
-->

---

<!-- layout: figure -->
<!-- section-header: mhc || 03 · mHC / STABLE TRANSPORT | SYSTEM CONTRACT · 4/4 || 15 / 29 -->

## mHC 把系统优化变成方法的一部分

![mHC 的 communication–computation overlap schedule](./assets/mhc-fig4-system-overlap.png)

<div class="slide-cards slide-cards--3">
  <div class="slide-card slide-card--read"><strong>HBM I/O</strong><code>n=4</code> residual-side 静态 accounting 约 <code>34C</code></div>
  <div class="slide-card slide-card--state"><strong>activation</strong>多流 state 需要保存或 recompute</div>
  <div class="slide-card slide-card--transport"><strong>PP payload</strong>完整 state 跨 boundary 时近似 <code>n×</code></div>
</div>

`F/B/W = forward / backward / weight-gradient` · `A/M = Attention / MLP` · `PP = pipeline parallelism`

<div class="slide-boundary">timeline block length 只作示意；作者报告 6.7% overhead，但未给完整硬件、并行布局与 tokens/s 分解。</div>

> **Reported + Derived** · mHC, arXiv:2512.24880v2, Fig. 4, PDF p.12；Table 2

<!-- notes:
1.5 分钟。state 变宽先带来 bytes、activation 与 PP payload，fusion/recompute/overlap 是对这条因果链的响应。

[Sources]
- https://arxiv.org/abs/2512.24880v2
- https://github.com/deepseek-ai/TileKernels/tree/36d9e45d38e204ebb87e6f6e833821eee0482fe5
-->

---

<!-- layout: figure -->
<!-- section-header: attnres || 04 · ATTNRES / DEPTH RETRIEVAL | ROUTE SWITCH · 1/6 || 16 / 29 -->

## AttnRes 把固定求和改成 depth softmax

![Attention Residuals Figure 1：Standard、Full 与 Block residual structures](./assets/attnres-fig1-overview.png)

<div class="slide-cards slide-cards--2">
  <div class="slide-card slide-card--transport"><strong>HC / mHC</strong>携带不断混合的多流 state</div>
  <div class="slide-card slide-card--read"><strong>AttnRes</strong>保留有名字的历史 sources</div>
</div>

$$
\begin{aligned}
h_l &= \sum_i \alpha_{i\to l}v_i,\\
\alpha_{\cdot\to l} &= \operatorname{softmax}_i(s_{i,l}),\\
s_{i,l} &= w_l^\top\operatorname{RMSNorm}(v_i)
\end{aligned}
$$

softmax 轴是 **historical depth/source**，不是 sequence token。

> **Reported** · Attention Residuals, arXiv:2603.15031v1, Fig. 1, PDF p.1；Eq. 1–4

<!-- notes:
1.5 分钟。query 是 layer-specific parameter；key/value 来自输入相关历史 source。同一 token 的 channels 默认共享 source 权重。

[Sources]
- https://arxiv.org/abs/2603.15031v1
-->

---

<!-- layout: comparison -->
<!-- section-header: attnres || 04 · ATTNRES / DEPTH RETRIEVAL | FULL RETRIEVAL · 2/6 || 17 / 29 -->

## Full AttnRes 检索 depth source

`[embedding v₀, layer-1 output v₁, layer-2 output v₂] -- α₀,α₁,α₂ --> current read`

![Full Attention Residuals 保存并检索逐层 source](./assets/attnres-fig1b-full.png)
![论文报告的 learned depth-routing heatmap](./assets/attnres-fig8-routing.png)

source stack 为 $[J,B,T,d]$，logits / weights 为 $[J,B,T]$，softmax 沿 $J$；不同 token 可选择不同 depth source，但默认不是 multihead。

> **Reported** · Attention Residuals, arXiv:2603.15031v1, Fig. 1(b)/8, PDF p.1/13；Eq. 3–4

<!-- notes:
2 分钟。先指明 embedding 和每个历史 branch output 都是可单独寻址的 source，再用三个 source 手算。

[Sources]
- https://arxiv.org/abs/2603.15031v1
-->

---

<!-- layout: comparison -->
<!-- section-header: attnres || 04 · ATTNRES / DEPTH RETRIEVAL | INITIALIZATION · 3/6 || 18 / 29 -->

## zero query 恢复 uniform average

**zero query 能恢复 standard residual 吗？不能。**

![Standard Residuals 使用单位权重求和](./assets/attnres-fig1-standard.png)
![Full AttnRes 使用 depth softmax](./assets/attnres-fig1b-full.png)

<div class="slide-cards slide-cards--2">
  <div class="slide-card slide-card--state"><strong>forward</strong><code>Σvᵢ</code> vs <code>(1/J)Σvᵢ</code></div>
  <div class="slide-card slide-card--read"><strong>direct gradient</strong><code>I</code> vs <code>αᵢI = (1/J)I</code></div>
</div>

`J = 当前可读 source 数`。RMSNorm 可能减弱统一 scale 的影响，但不能推出函数与 Jacobian 完全等价。

> **Derived** · Attention Residuals, arXiv:2603.15031v1, Fig. 1(a–b), PDF p.1；Eq. 3–4

<!-- notes:
1.5 分钟。zero query 是明确定义的 special case，不是 baseline-preserving identity initialization。

[Sources]
- https://arxiv.org/abs/2603.15031v1
-->

---

<!-- layout: comparison -->
<!-- section-header: attnres || 04 · ATTNRES / DEPTH RETRIEVAL | BLOCK COMPRESSION · 4/6 || 19 / 29 -->

## Block AttnRes 用结构约束压缩历史

`假设每 4 个 sublayers 分一组：4 个 layer sources → 1 个 completed-block summary；当前 block 另保留 partial`

![Full AttnRes 保存所有 layer sources](./assets/attnres-fig1b-full.png)
![Block AttnRes 保存 completed blocks 与 current partial](./assets/attnres-fig1c-block.png)

`all layer sources → embedding + completed block summaries + current partial`

<div class="slide-boundary">这是 depth mixing columns 的结构化压缩，不宣称无损；实际 block size 以论文配置为准。</div>

> **Reported** · Attention Residuals, arXiv:2603.15031v1, Fig. 1(b–c), PDF p.1；Eq. 5–6

<!-- notes:
2 分钟。completed block 保存 branch delta 的和；当前 block 的 branch output 累加到 partial。

[Sources]
- https://arxiv.org/abs/2603.15031v1
-->

---

<!-- layout: figure -->
<!-- section-header: attnres || 04 · ATTNRES / DEPTH RETRIEVAL | EXACT EXECUTION & CACHE · 5/6 || 20 / 29 -->

## two-phase 是精确执行，PP cache 是状态协议

![Block AttnRes 的 pipeline cache 传递](./assets/attnres-fig3-pp-cache.png)

<div class="slide-cards slide-cards--2">
  <div class="slide-card slide-card--read"><strong>Math contract</strong>历史 blocks 与 current partial 的 softmax statistics 可精确 merge</div>
  <div class="slide-card slide-card--transport"><strong>State contract</strong>cache key 至少包含 microbatch / virtual stage / block / direction</div>
</div>

<div class="slide-boundary">数学等价 ≠ 系统状态自动正确；精确算法也不代表实现无开销。</div>

> **Reported** · Attention Residuals, arXiv:2603.15031v1, Fig. 3, PDF p.6；Algorithm 1、Appendix B

<!-- notes:
1.5 分钟。需要时口头解释 (max, exp-sum, numerator) 的 online-softmax merge；主页面只区分数学等价和状态 ownership。

[Sources]
- https://arxiv.org/abs/2603.15031v1
- https://github.com/MoonshotAI/Attention-Residuals/tree/85e22310fe5ee860b4a023de312d791de8a5a5e6
-->

---

<!-- layout: figure -->
<!-- section-header: attnres || 04 · ATTNRES / DEPTH RETRIEVAL | SCALING EVIDENCE · 6/6 || 21 / 29 -->

## Block 版本保留了大部分 scaling 收益

![Baseline、Full 与 Block AttnRes 的五点 scaling-law fit](./assets/attnres-fig4-scaling.png)

<div class="slide-cards slide-cards--3">
  <div class="slide-card slide-card--neutral"><strong>Baseline</strong><code>1.719</code></div>
  <div class="slide-card slide-card--transport"><strong>Block</strong><code>1.693</code></div>
  <div class="slide-card slide-card--read"><strong>Full</strong><code>1.692</code></div>
</div>

`1.25× = 五点拟合的 compute-equivalence estimate`，不是 measured wall-clock speedup。

<div class="slide-boundary">single run · no fit uncertainty · data / validation protocol not public</div>

> **Reported** · Attention Residuals, arXiv:2603.15031v1, Fig. 4, PDF p.9

<!-- notes:
2 分钟。先读同一规模的 validation loss，再解释 fitted compute-equivalence 的边界。

[Sources]
- https://arxiv.org/abs/2603.15031v1
-->

---

<!-- layout: figure -->
<!-- section-header: xhc || 05 · xHC / LARGE-N | TWO BOTTLENECKS · 1/5 || 22 / 29 -->

## xHC 从大 N 的两个瓶颈出发

![mHC 与 xHC 的 loss 和 training FLOPs 随 expansion rate N 的变化](./assets/xhc-fig1-expansion-efficiency.png)

<div class="slide-cards slide-cards--2">
  <div class="slide-card slide-card--write"><strong>write-back information supply</strong>同一个 branch output 只能提供一个新方向</div>
  <div class="slide-card slide-card--transport"><strong>mapping generator cost</strong>从 <code>NC</code> state 生成 <code>N²</code> logits，规模 <code>O(N³C)</code></div>
</div>

`N = persistent stream 数`。信息供给不足是有实验支持的 mechanism hypothesis，不是已证明的容量定理。

> **Reported** · xHC, arXiv:2607.14530v1, Fig. 1, PDF p.1；§3

<!-- notes:
1.5 分钟。先指出 N>4 后 mHC 收益趋缓，再解释两个候选机制。

[Sources]
- https://arxiv.org/abs/2607.14530v1
-->

---

<!-- layout: figure -->
<!-- section-header: xhc || 05 · xHC / LARGE-N | RICHER WRITE · 2/5 || 23 / 29 -->

## Temporal augmentation 扩展写回

![xHC Figure 3(c)：dense read、sparse update 与 temporal feature augmentation](./assets/xhc-fig3c-expanded.png)

<div class="slide-flow" aria-label="xHC 的四个 temporal write candidates">
  <span class="slide-flow__step">one MLP output sequence</span><span class="slide-flow__arrow">→</span>
  <span class="slide-flow__step">current</span>
  <span class="slide-flow__step">DWConv-4</span>
  <span class="slide-flow__step">DWConv-8</span>
  <span class="slide-flow__step">DWConv-12</span>
</div>

默认 kernel sizes 为 $\{4,8,12\}$，得到 $K_r=4$ 个 write candidates；新增的是 $24C$ depthwise-conv 参数，**不是计算 4 次 MLP**。

<div class="slide-boundary">causal convolution 沿 token sequence，只使用当前位置及更早 token；不混合 channel，也不是 token self-attention。</div>

> **Reported** · xHC, arXiv:2607.14530v1, Fig. 3(c), PDF p.4；Eq. 7–10

<!-- notes:
1.5 分钟。components 仍来自同一 sequence；Gram–Schmidt 只消除局部共线，不保证语义独立或 unit norm。

[Sources]
- https://arxiv.org/abs/2607.14530v1
-->

---

<!-- layout: figure -->
<!-- section-header: xhc || 05 · xHC / LARGE-N | DENSE READ · SPARSE UPDATE · 3/5 || 24 / 29 -->

## Dense read 解耦 $N$ 与 $k$

![xHC Figure 3(c)：全部 streams 参与 read，active subset 执行 mutation](./assets/xhc-fig3c-expanded.png)

<div class="slide-cards slide-cards--2">
  <div class="slide-card slide-card--read"><strong>① dense read</strong>全部 <code>N=16</code> streams 都影响 branch input</div>
  <div class="slide-card slide-card--write"><strong>② sparse mutation</strong>fixed-2 + routed Top-2 只更新 <code>k=4</code> streams</div>
</div>

`N = memory capacity` · `k = mutation budget` · `inactive streams = exact carry`

> **Reported** · xHC, arXiv:2607.14530v1, Fig. 3(c), PDF p.4；Algorithm 1。只更新四条 ≠ 只读取四条。

<!-- notes:
2 分钟。hard Top-k 的 indices 不可微，只有 selected routed scores 收到本次 routing gradient。

[Sources]
- https://arxiv.org/abs/2607.14530v1
-->

---

<!-- layout: figure -->
<!-- section-header: xhc || 05 · xHC / LARGE-N | ABLATION · 4/5 || 25 / 29 -->

## ablation 分开了质量与成本角色

![xHC Table 2 前五行与 Figure 5 的聚焦裁切](./assets/xhc-table2-fig5-ablation-focus.png)

<div class="slide-flow" aria-label="xHC 核心 ablation">
  <span class="slide-flow__step">mHC N=16<br>1.998 · +18.8%</span><span class="slide-flow__arrow">→ quality</span>
  <span class="slide-flow__step">+ TempAug<br>1.984 · +20.1%</span><span class="slide-flow__arrow">→ cost</span>
  <span class="slide-flow__step">full xHC<br>1.983 · +3.3%</span>
</div>

第一段主要改善 validation loss；第二段主要回收 dense large-N 的额外 FLOPs。

<div class="slide-boundary">Val. Loss 越低越好；括号为论文 counting boundary 下的 training FLOPs overhead，不等于 wall-clock。</div>

> **Reported** · xHC, arXiv:2607.14530v1, Table 2 & Fig. 5, PDF p.10

<!-- notes:
2 分钟。沿三行一次只比较一个新增组件；单一 in-house setting、无多 seed。

[Sources]
- https://arxiv.org/abs/2607.14530v1
-->

---

<!-- layout: figure -->
<!-- section-header: xhc || 05 · xHC / LARGE-N | FLASH BOUNDARY · 5/5 || 26 / 29 -->

## xHC-Flash 局部精确，整体是近似架构

![xHC、xHC-Flash 与 Flash-4sub 的 validation loss 和静态 residual-side I/O](./assets/xhc-table5-flash.png)

`window = 一组复用同一 routing / pre-mapping 的 sublayers`

<div class="slide-cards slide-cards--2">
  <div class="slide-card slide-card--read"><strong>window 内</strong>给定 routing 后，dense-read correction 与逐步更新代数等价</div>
  <div class="slide-card slide-card--caveat"><strong>与 full xHC 比</strong>routing 刷新频率、依赖状态和 mixing 位置改变，因此整体架构近似</div>
</div>

<div class="slide-boundary">静态 accounting 从 full xHC <code>73.5C</code> 降到 Flash-4sub <code>40C</code>；不是实际 HBM bytes 或 wall-clock speedup。</div>

> **Reported + Derived** · xHC, arXiv:2607.14530v1, Table 5, PDF p.14；§4

<!-- notes:
1.5 分钟。exact correction 与 architectural approximation 的比较对象不同；不要混成一个“完全等价”结论。

[Sources]
- https://arxiv.org/abs/2607.14530v1
-->

---

<!-- layout: comparison -->
<!-- section-header: synthesis || 06 · 综合 / DESIGN & VALIDATION | METHOD CHOICE · 1/3 || 27 / 29 -->

## 没有一种方法统治所有设计轴

| 方法 | 并行 state capacity | 直接选择历史 source | 深层 transport 约束 | 主要系统负担 |
|---|---|---|---|---|
| HC | 中：$n$ streams | 低 | 低：unconstrained | activation / 深层乘积 |
| mHC | 中：$n$ streams | 低 | 高：near doubly stochastic | I/O / Sinkhorn / PP |
| AttnRes | 高：历史 sources | 高：depth softmax | 不适用：sources persist | history / cache |
| xHC | 高：$N$ streams | 中：dense read | 中：active mixing | persistent state / routing |

选择方法，本质是在 state capacity、retrieval expressivity、transport stability 与 system cost 上选 Pareto 点。

<div class="slide-boundary">定性综合，不是统一实验排名；“低/中/高”只相对同一列的比较对象。</div>

<!-- notes:
2 分钟。逐列比较，不逐方法重讲一遍。要可组合多流 transport 看 mHC；要直接取历史看 AttnRes；要扩大 N 并限制 mutation 看 xHC。

[Sources]
- https://arxiv.org/abs/2409.19606v3
- https://arxiv.org/abs/2512.24880v2
- https://arxiv.org/abs/2603.15031v1
- https://arxiv.org/abs/2607.14530v1
- https://github.com/J-shang/residual
-->

---

<!-- layout: comparison -->
<!-- section-header: synthesis || 06 · 综合 / DESIGN & VALIDATION | TEST AGENDA · 2/3 || 28 / 29 -->

## 开放问题应该被写成测试

**论文未闭合的问题，不用先训练大模型也能部分验证。**

| 问题 | 最小 CPU check | later distributed check |
|---|---|---|
| 深层 transport 会不会漂移？ | compose 60 层，记录 singular values 与 row/column error | mixed precision + recompute |
| 初始化是否等价？ | hand calculation + output/JVP + float64 gradcheck | checkpoint migration |
| reference / fused 是否同一函数？ | direct/recompute/two-phase 对照 | fallback + determinism |

先固定 correctness contract，再进入 GPU scaling。

<!-- notes:
2 分钟。逐行回指 S11、S18、S20/S26 的未闭合边界，把研究问题改写成可证伪测试。

[Sources]
- https://github.com/J-shang/residual
-->

---

<!-- layout: statement -->
<!-- section-header: synthesis || 06 · 综合 / DESIGN & VALIDATION | TAKEAWAY · 3/3 || 29 / 29 -->

## Residual path 是 depth memory

$$
\boxed{
\text{stored state}
+\text{read}
+\text{transport}
+\text{write}
+\text{system contract}
}
$$

<div class="slide-cards slide-cards--4">
  <div class="slide-card slide-card--read"><strong>HC</strong>multi-stream state</div>
  <div class="slide-card slide-card--transport"><strong>mHC</strong>stable transport</div>
  <div class="slide-card slide-card--state"><strong>AttnRes</strong>depth retrieval</div>
  <div class="slide-card slide-card--write"><strong>xHC</strong>sparse mutation</div>
</div>

连接方法的核心不是多加一个算子，而是重新定义信息沿 depth 的**保存、访问、变换与消亡方式**。

`Q&A · 10 MIN`

<!-- notes:
1.5 分钟。按公式五个字段回收全场，不再引入新术语。结束句：连接方法的核心不是多加一个算子，而是重新定义 depth 上的信息生命史。

[Sources]
- https://arxiv.org/abs/2409.19606v3
- https://arxiv.org/abs/2512.24880v2
- https://arxiv.org/abs/2603.15031v1
- https://arxiv.org/abs/2607.14530v1
- https://github.com/J-shang/residual
-->

---

<!-- layout: comparison -->
<!-- section-header: backup || BACKUP / PROOF & IMPLEMENTATION | NON-EXPANSIVE PROOF · 1/4 || B1 / B4 -->

## 备份：双随机为何 non-expansive

<div class="slide-cards slide-cards--3">
  <div class="slide-card slide-card--transport"><strong>条件</strong><code>H ≥ 0, H1 = 1, 1ᵀH = 1ᵀ</code></div>
  <div class="slide-card slide-card--state"><strong>induced norms</strong><code>||H||₁ = 1, ||H||∞ = 1</code></div>
  <div class="slide-card slide-card--read"><strong>spectral bound</strong><code>||H||₂ ≤ √(||H||₁||H||∞) ≤ 1</code></div>
</div>

<div class="slide-boundary">只做 row-stochastic 不足以得到相同结论；finite Sinkhorn 还需报告 row/column error 与深层 product drift。</div>

> **Derived** · mHC, arXiv:2512.24880v2, Eq. 6–9

<!-- notes:
备份页。只在听众追问 mHC 的 norm guarantee 时展开；结论依赖非负、行和与列和同时成立。

[Sources]
- https://arxiv.org/abs/2512.24880v2
-->

---

<!-- layout: comparison -->
<!-- section-header: backup || BACKUP / PROOF & IMPLEMENTATION | SHAPE & DTYPE · 2/4 || B2 / B4 -->

## 备份：reference shape 与 dtype

`B=batch` · `S/T=sequence` · `n/N=streams` · `J=history sources` · `C/d=hidden size`

| 方法 | logical state | read / weights | transport / cache | write / active state |
|---|---|---|---|---|
| HC | `[B,S,n,C]` | `A_m:[B,S,n,1]` | `A_r:[B,S,n,n]` | `B:[B,S,1,n]` |
| mHC | `[B,S,n,C]` | `H_pre:[B,S,1,n]` | `H_res:[B,S,n,n]` | `H_post:[B,S,1,n]` |
| AttnRes | `[J,B,T,d]` | weights `[J,B,T]` | cache owns source lifetime | read `[B,T,d]` |
| xHC | `[B,S,N,C]` | dense read over `N` | active mix `[B,S,k,k]` | active `[B,S,k,C]` |

correctness-first：Sinkhorn、softmax statistics、routing logits 与 direct-vs-optimized 对照先用 FP32 / float64 边界验证。

<!-- notes:
备份页。dtype 是 correctness-first 建议，不全部是论文规定；logical state、local shard、layout 与 ownership 要分开写。

[Sources]
- https://arxiv.org/abs/2409.19606v3
- https://arxiv.org/abs/2512.24880v2
- https://arxiv.org/abs/2603.15031v1
- https://arxiv.org/abs/2607.14530v1
- https://github.com/deepseek-ai/TileKernels/tree/36d9e45d38e204ebb87e6f6e833821eee0482fe5
-->

---

<!-- layout: comparison -->
<!-- section-header: backup || BACKUP / PROOF & IMPLEMENTATION | EVIDENCE AUDIT · 3/4 || B3 / B4 -->

## 备份：四篇论文的证据成熟度不同

`强 / 中 / 弱 = 当前公开材料的可审计程度，不是方法质量评分`

| 来源 | method / math | quality | system | artifact | statistics |
|---|---|---|---|---|---|
| HC | 强 | 中 | 中 | 弱 | 弱 |
| mHC | 强 | 中 | 中 | 中 | 弱 |
| AttnRes | 强 | 中 | 弱 | 弱 | 弱 |
| xHC | 强 | 中 | 弱 | 弱 | 弱 |

“弱”表示公开材料难以完整审计，**不表示方法无效**；四篇论文的大模型结果普遍缺多 seed 与 error bar。

<!-- notes:
备份页。mHC 有后续 TileKernels artifact；AttnRes/xHC artifact 仍缺完整训练实现或 kernel。

[Sources]
- https://arxiv.org/abs/2409.19606v3
- https://arxiv.org/abs/2512.24880v2
- https://arxiv.org/abs/2603.15031v1
- https://arxiv.org/abs/2607.14530v1
- https://github.com/MoonshotAI/Attention-Residuals/tree/85e22310fe5ee860b4a023de312d791de8a5a5e6
- https://github.com/aHapBean/xHC/tree/7890266d5cd648811b6783029ee6b5031cd209db
-->

---

<!-- layout: comparison -->
<!-- section-header: backup || BACKUP / PROOF & IMPLEMENTATION | COUNTING BOUNDARY · 4/4 || B4 / B4 -->

## 备份：系统成本数字不可直接横比

`counting boundary = 一个数字具体统计了哪些 operation、bytes 或时间`

| 类型 | 数字示例 | counting boundary |
|---|---|---|
| residual-side 静态 I/O | mHC $\approx34C$ | mapping generation / application |
| residual-side 静态 I/O | AttnRes $24d / 5.5d$ | two-phase residual mechanism |
| residual-side 静态 I/O | xHC $73.5C / 40C$ | 抽象 element traffic |
| 端到端 memory | HC `+9.7%–28.3%` | 论文具体训练设置 |
| 端到端 overhead | mHC `6.7%`、AttnRes `<4%` | 硬件与并行分解不完整 |

当前公开证据不足以给出跨论文 wall-clock 排名。

<!-- notes:
备份页。每个成本数字必须保留 setup、是否实测、是否含 branch I/O，以及 resident cache/payload 的边界。

[Sources]
- https://arxiv.org/abs/2409.19606v3
- https://arxiv.org/abs/2512.24880v2
- https://arxiv.org/abs/2603.15031v1
- https://arxiv.org/abs/2607.14530v1
-->
