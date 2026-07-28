---
title: "mHC：用双随机约束把可学习 residual topology 拉回稳定区"
description: "解释 mHC 如何用 Sinkhorn 投影约束 residual mixing，并区分结构保证、训练稳定性证据与系统实现成本。"
topic: "residual"
section: "methods"
slug: "mhc"
date: 2026-07-27
updated: 2026-07-27
cutoff: 2026-07-23
featured: true
order: 11
readtime: 37
source:
  repository: "J-shang/residual"
  path: "papers/02-mhc.md"
  url: "https://github.com/J-shang/residual/blob/c55583707ab645bef7408c588fc33cbf91b809a5/papers/02-mhc.md"
  revision: "c55583707ab645bef7408c588fc33cbf91b809a5"
  syncedAt: "2026-07-28"
  contentHash: "sha256:d023f537ee7cfbf1687b191bc1a67a283ff7f4236f0687b699b2baf6f99a566e"
  manifest: "residual"
  managed: true
---
<!-- paper-order: 02 -->

> **论文**：Zhenda Xie, Yixuan Wei, Huanqi Cao et al., *mHC: Manifold-Constrained Hyper-Connections*<br>
> **机构**：DeepSeek-AI<br>
> **版本**：arXiv:2512.24880v2，2026-01-05<br>
> **状态**：arXiv preprint<br>
> **主来源**：[arXiv 摘要](https://arxiv.org/abs/2512.24880v2) · [HTML](https://arxiv.org/html/2512.24880v2) · [PDF](https://arxiv.org/pdf/2512.24880v2)<br>
> **阅读范围**：完整 19 页，包括 Eq. 1–20、Figure 1–8、Table 1–5 与 Appendix A.1<br>
> **后续信息截止日期**：2026-07-23<br>
> **前置阅读**：[Hyper-Connections 论文解读](/topics/residual/hyper-connections/)

## 证据标签

- **[论文报告]**：论文正文、公式、图表或附录直接给出的内容。
- **[复原推导]**：由论文定义独立推导出的结论。
- **[综合判断]**：结合理论、实验和系统约束形成的解释。
- **[后续证据]**：论文之后出现的 primary source 或官方 artifact。
- **[扩展假设]**：论文之外、需要新实验验证的提案。
- **[待验证]**：当前信息不足，不能视为已建立的事实。

---

## 先给结论

### 30 秒版

原始 Hyper-Connections（HC）用 $n$ 条 residual streams 和可学习矩阵增加了跨层拓扑表达能力，但核心 residual mixing matrix $H_l^{\mathrm{res}}$ 没有约束。深层连续相乘后，这条“直通路径”可能把信号和梯度放大或压缩，失去 residual connection 最关键的稳定性。

mHC 的核心修正是：

$$
H_l^{\mathrm{res}}
\in
\mathcal B_n,
$$

这里 $l$ 是子层编号，$H_l^{\mathrm{res}}\in\mathbb R^{n\times n}$ 是该层 forward 时使用的 residual-stream mixing matrix，$\mathcal B_n$ 是由非负、行和与列和均为 1 的 doubly stochastic matrices 构成的 Birkhoff polytope。$H_l^{\mathrm{res}}$ 是输入相关的运行时 mapping，不是 optimizer 直接保存的自由矩阵；论文用 20 次 Sinkhorn-Knopp 迭代从无约束 logits 近似生成它。

精确双随机矩阵具有三个关键性质：

1. 对 stream 维的共同分量和均值守恒；
2. $\ell_2$ operator norm 不超过 1，不会放大 residual transport；
3. 多层乘积仍是双随机矩阵，稳定性质可以跨深度组合。

27B MoE 实验中，原始 HC 的 composite gain 峰值接近 3000，而 mHC 的近似投影把峰值控制到约 1.6；mHC 相对普通 baseline 最终 training loss 降低 0.021，并在 8 个下游任务上全部超过 baseline。

### 最重要的精确化

论文使用“restore identity mapping property”的表述，但 mHC 通常并不满足：

$$
H_l^{\mathrm{res}}=I.
$$

更准确地说，它恢复的是：

- common-mode identity；
- stream mean conservation；
- residual transport 的 non-expansiveness；
- 这些性质在矩阵复合下的闭包。

不同 streams 之间仍会被混合，因此不是“每条 stream 原样穿过”。

### 最重要的保留意见

- 20 次 Sinkhorn 只是近似投影，论文实测 composite backward gain 仍可到约 1.6，而不是严格等于 1。
- mHC 同时改变了 $H^{\mathrm{pre}}$、$H^{\mathrm{post}}$、$H^{\mathrm{res}}$ 的参数化和动态权重生成方式；论文没有做足够消融，把质量与稳定性收益唯一归因于双随机约束。
- 27B 结果没有多 seed、误差条或统计检验。
- 训练数据名称、组成与质量没有公开。
- “6.7% additional time overhead”没有给出硬件、集群规模、tokens/s、step time 或对照表，属于难以独立审计的 in-house system claim。
- 所有主要实验都是 DeepSeek-V3 风格 MoE，尚不能直接外推到 dense LLM、视觉模型或其他训练 recipe。

---

## 阅读前：符号与约定

后文会在关键公式旁重新定义新符号；本表用于先建立整体对象关系。默认把 stream 作为矩阵第一维、hidden channel 作为第二维。

| 符号 | 含义 | 类型或 shape | 作用域 | 身份 |
|---|---|---:|---|---|
| $l,i,L$ | 当前子层、乘积中的层索引、最深子层 | 整数 | depth 维 | 索引 |
| $B,S$ | batch size、sequence length | 正整数 | 一个 batch | 运行配置 |
| $n$ | residual stream expansion rate | 正整数 | 通常全模型固定；论文默认 $n=4$ | 架构超参数 |
| $C$ | 每条 stream 的 hidden dimension | 正整数 | 全模型或当前 stage | 架构超参数 |
| $\mathbf x_l$ | 第 $l$ 个子层入口的多流 residual state | $[n,C]$ 或 $[B,S,n,C]$ | 每层、每 token | 运行时张量 |
| $\mathcal F(\cdot,W_l)$ | Attention/FFN branch；$W_l$ 是其主干权重 | $[C]\to[C]$ | 每个子层 | $W_l$ 是普通可训练参数 |
| $H_l^{\mathrm{pre}}$ | 从 $n$ 条 streams 读取 branch input | $[1,n]$ | 每层、每 token | 受约束的运行时 mapping |
| $H_l^{\mathrm{post}}$ | 把 branch output 写回 $n$ 条 streams | $[1,n]$ | 每层、每 token | 受约束的运行时 mapping |
| $H_l^{\mathrm{res}}$ | 传播和混合旧 streams | $[n,n]$ | 每层、每 token | 近似双随机的运行时 mapping |
| $\widetilde H_l^{(\cdot)}$ | 施加 Sigmoid/Sinkhorn 前的自由 logits | 与对应 $H_l^{(\cdot)}$ 相同 | 每层、每 token | 运行时张量，不是 leaf parameter |
| $\operatorname{vec},\operatorname{mat}$ | 把 $[n,C]$ 展平为 $[1,nC]$，以及把 $n^2$ 项恢复为 $[n,n]$ | reshape | 每 token | 固定无参数算子 |
| $\varphi_l^{\mathrm{pre}},\varphi_l^{\mathrm{post}},\varphi_l^{\mathrm{res}}$ | 生成三组动态 logits 的线性投影 | $[nC,n],[nC,n],[nC,n^2]$ | 每层 | optimizer 直接更新的参数 |
| $\alpha_l^{\mathrm{pre}},\alpha_l^{\mathrm{post}},\alpha_l^{\mathrm{res}}$ | 三条动态生成路径的幅度 gate | 标量 | 每层 | optimizer 直接更新的参数；不要与 AttnRes attention weight 混淆 |
| $b_l^{\mathrm{pre}},b_l^{\mathrm{post}},b_l^{\mathrm{res}}$ | 三组输入无关的 static logits/bias | $[1,n],[1,n],[n,n]$ | 每层 | optimizer 直接更新的参数 |
| $\sigma(z)$ | logistic Sigmoid，$\sigma(z)=1/(1+e^{-z})$ | $\mathbb R\to(0,1)$，逐元素 | pre/post mapping | 固定可微函数；$\sigma$ 本身没有可训练参数 |
| $\operatorname{SK}$ / $\operatorname{SinkhornKnopp}$ | `exp` 后交替归一化行列的映射 | $[n,n]\to[n,n]$ | residual mapping | 固定可微重参数化，不是 loss |
| $M^{(t)}$ | 第 $t$ 次 Sinkhorn 迭代后的正矩阵 | $[n,n]$ | 一个 forward 内 | 运行时中间量 |
| $\mathcal T_r,\mathcal T_c$ | row/column normalization | $[n,n]\to[n,n]$ | Sinkhorn 内部 | 固定无参数算子 |
| $t_{\max}$ | Sinkhorn 迭代次数，论文取 20 | 整数 | 训练/推理配置 | 超参数 |
| $\mathcal B_n$ | $n$ 阶 Birkhoff polytope | 矩阵集合 | 理论约束 | 固定集合，不是单个参数 |
| $\mathbf1_n,I_n$ | $n$ 维全 1 向量、identity matrix | $[n]$、$[n,n]$ | 推导/测试 | 固定数学对象 |
| $\|\cdot\|_2,\|\cdot\|_F,\|\cdot\|_1,\|\cdot\|_\infty$ | spectral/operator norm、Frobenius norm、induced 1/∞ norms | 标量输出 | 稳定性分析 | 固定数学运算 |

核心身份链条是：

$$
\left\{\varphi,\alpha,b\right\}_{\text{可训练参数}}
\longrightarrow
\widetilde H_{\text{运行时 logits}}
\xrightarrow{\ \sigma\ /\ \operatorname{SK}\ }
H_{\text{受约束运行时 mapping}}.
$$

**[复原推导]** 主任务 loss 沿这条可微链反传到 $\varphi,\alpha,b$ 和 Transformer 主干；论文没有另加一个“违反双随机约束的惩罚 loss”。

---

## 1. mHC 在 HC 上修了什么

### 1.1 HC 的基本状态更新

论文把第 $l$ 个子层的多流状态记为：

$$
\mathbf x_l
\in
\mathbb R^{n\times C},
$$

其中：

- $n$：residual stream expansion rate；
- $C$：单个 Transformer 子层的 hidden dimension。

实际批量序列实现通常对应：

$$
\mathbf x_l
\in
\mathbb R^{B\times S\times n\times C}.
$$

HC 的更新为（论文 Eq. 3）：

$$
\mathbf x_{l+1}
=
H_l^{\mathrm{res}}\mathbf x_l
+
\left(H_l^{\mathrm{post}}\right)^\top
\mathcal F
\left(
H_l^{\mathrm{pre}}\mathbf x_l,
W_l
\right),
$$

其中：

$$
H_l^{\mathrm{pre}},H_l^{\mathrm{post}}
\in
\mathbb R^{1\times n},
\qquad
H_l^{\mathrm{res}}
\in
\mathbb R^{n\times n}.
$$

三者分别完成：

| Mapping | 作用 | Shape 变化 |
|---|---|---|
| $H^{\mathrm{pre}}$ | 从 $n$ 条 streams 读取子层输入 | $[n,C]\to[1,C]$ |
| $H^{\mathrm{res}}$ | 传播并混合旧 residual state | $[n,C]\to[n,C]$ |
| $H^{\mathrm{post}}$ | 把子层输出写回 $n$ 条 streams | $[1,C]\to[n,C]$ |

### 1.2 深层展开后的危险项

从浅层 $l$ 展开到深层 $L$，旧状态的直接传播项包含（论文 Eq. 4）：

$$
\left(
\prod_{i=1}^{L-l}
H_{L-i}^{\mathrm{res}}
\right)
\mathbf x_l.
$$

普通 residual connection 的对应系数是 identity。原始 HC 中，每个
$H_i^{\mathrm{res}}$ 都是无约束矩阵，连续相乘可能：

- 放大某些方向；
- 压缩某些方向；
- 改变 stream mean；
- 在 backward 中产生类似的梯度放大/衰减。

**[论文报告]** 在 27B HC 模型中，loss 在约 12k step 出现异常抬升，并与 gradient norm spike 高度同步（Figure 2）。由 composite mapping 计算的 Amax Gain Magnitude 峰值接近 3000（Figure 3）。

### 1.3 mHC 的最小方法变化

mHC 不把 $H^{\mathrm{res}}$ 固定为 identity，因为那会禁止 stream exchange。它把搜索空间从所有实矩阵：

$$
\mathbb R^{n\times n}
$$

缩小到 doubly stochastic matrices：

$$
\mathcal B_n
=
\left\{
H\in\mathbb R^{n\times n}
\;\middle|\;
H\mathbf1=\mathbf1,\;
\mathbf1^\top H=\mathbf1^\top,\;
H\ge0
\right\}.
$$

这是论文 Eq. 6 所定义的核心约束。

---

## 2. 一张方法差异表

| 维度 | 普通 residual | 原始 HC | mHC |
|---|---|---|---|
| Stored state | $[B,S,C]$ | $[B,S,n,C]$ | $[B,S,n,C]$ |
| Residual transport | $I$ | 无约束 $H^{res}$ | 近似双随机 $H^{res}$ |
| Read | 固定 identity | 可学习 $H^{pre}$ | positive $H^{pre}=\sigma(\cdot)$ |
| Write | 固定加法 | 可学习 $H^{post}$ | positive $H^{post}=2\sigma(\cdot)$ |
| Stream exchange | 无 | 有 | 有 |
| 精确 common-mode identity | 是 | 不保证 | 精确投影时是 |
| Mean conservation | 是 | 不保证 | 精确投影时是 |
| Residual transport non-expansive | 是 | 不保证 | 精确投影时是 |
| 深度复合闭包 | identity | 不保证 | 精确投影时仍双随机 |
| 实际投影 | 不需要 | 不需要 | 20 次 Sinkhorn 近似 |
| 主要系统成本 | baseline | 多流 I/O、activation、PP payload | 同类成本；用 fusion/recompute/overlap 缓解 |

---

## 3. 双随机矩阵到底保证了什么

### 3.1 Common-mode identity

对任意 feature vector $z\in\mathbb R^C$，如果所有 streams 保存相同状态：

$$
\mathbf x
=
\mathbf1_n z^\top,
$$

则：

$$
H^{\mathrm{res}}\mathbf x
=
H^{\mathrm{res}}\mathbf1_n z^\top
=
\mathbf1_n z^\top
=
\mathbf x.
$$

因为双随机矩阵满足：

$$
H^{\mathrm{res}}\mathbf1_n=\mathbf1_n.
$$

**[复原推导]** mHC 对“所有 streams 共同拥有的分量”实现了精确 identity mapping。它没有对 streams 之间的差异分量保持 identity。

### 3.2 Stream mean conservation

定义 stream mean：

$$
\bar x
=
\frac1n\mathbf1_n^\top \mathbf x
\in
\mathbb R^{1\times C}.
$$

经过 residual transport：

$$
\frac1n\mathbf1_n^\top
H^{\mathrm{res}}\mathbf x
=
\frac1n\mathbf1_n^\top\mathbf x
=
\bar x,
$$

因为：

$$
\mathbf1_n^\top H^{\mathrm{res}}
=
\mathbf1_n^\top.
$$

所以 exact doubly stochastic transport 不改变各 feature channel 在 streams 上的平均值。

### 3.3 Forward 与 backward 的共同守恒

若 forward 使用：

$$
y=Hx,
$$

则 backward 为：

$$
g_x=H^\top g_y.
$$

双随机性对转置封闭：

$$
H\in\mathcal B_n
\Rightarrow
H^\top\in\mathcal B_n.
$$

因此 common-mode 与 mean conservation 同时适用于 forward signal 和 backward gradient。

### 3.4 Non-expansiveness

根据 Birkhoff-von Neumann theorem，每个双随机矩阵都可写为 permutation matrices 的凸组合：

$$
H
=
\sum_k\lambda_kP_k,
\qquad
\lambda_k\ge0,
\qquad
\sum_k\lambda_k=1.
$$

每个 permutation matrix 满足：

$$
\lVert P_k\rVert_2=1.
$$

因此：

$$
\lVert H\rVert_2
\le
\sum_k\lambda_k\lVert P_k\rVert_2
=
1.
$$

另一方面 $H\mathbf1=\mathbf1$，所以 $1$ 是 singular value，下界也是 1。对 exact doubly stochastic matrix：

$$
\boxed{\lVert H\rVert_2=1}.
$$

对任意多流状态矩阵：

$$
\lVert H\mathbf x\rVert_F
\le
\lVert\mathbf x\rVert_F.
$$

**[复原推导]** 论文称这一性质为 “Norm Preservation”，更精确的叫法是 **non-expansion**：common-mode 的 norm 保持，但一般差异分量可以收缩。

### 3.5 深度复合闭包

若：

$$
H_1,H_2\in\mathcal B_n,
$$

则：

$$
(H_2H_1)\mathbf1
=
H_2\mathbf1
=
\mathbf1,
$$

$$
\mathbf1^\top(H_2H_1)
=
\mathbf1^\top H_1
=
\mathbf1^\top,
$$

且两个非负矩阵的乘积仍非负，所以：

$$
H_2H_1\in\mathcal B_n.
$$

由此：

$$
\prod_l H_l^{\mathrm{res}}
\in
\mathcal B_n.
$$

这是 mHC 相比“只在初始化时接近 identity”更重要的性质：约束可以跨任意深度复合。

---

## 4. 它没有保证什么

### 4.1 不是完整 identity

除非：

$$
H^{\mathrm{res}}=I,
$$

否则一般有：

$$
H^{\mathrm{res}}\mathbf x\ne\mathbf x.
$$

mHC 保持 common-mode 和均值，但允许差异分量在 streams 之间重新分配。

### 4.2 不保证所有方向 norm 不变

双随机矩阵可以强烈收缩 stream differences。例如：

$$
H
=
\frac12
\begin{bmatrix}
1&1\\
1&1
\end{bmatrix}.
$$

对：

$$
x=
\begin{bmatrix}
1\\-1
\end{bmatrix},
$$

有：

$$
Hx=0.
$$

所以它不会爆炸，但可能把 streams 的差异完全抹掉。

### 4.3 不保证总状态不会增长

mHC 的完整更新仍包含 branch injection：

$$
\mathbf x_{l+1}
=
H_l^{\mathrm{res}}\mathbf x_l
+
\left(H_l^{\mathrm{post}}\right)^\top
\mathcal F(\cdot).
$$

双随机约束只控制第一项。第二项与普通 residual branch 一样，可以改变 activation scale。

### 4.4 “反复应用会单调增加 mixing”不是无条件成立

Permutation matrix 本身就是双随机矩阵，但反复应用只会重排 streams，并不会趋向平均。

要收敛到 uniform mixing，通常还需要：

- irreducibility；
- aperiodicity；
- 足够的正元素；
- 或对 time-varying products 的共同连通性条件。

**[综合判断]** Birkhoff polytope 提供稳定搜索空间，但“最终是否充分利用多流容量”仍是独立问题。

### 4.5 Birkhoff polytope 严格说不是处处光滑的 manifold

论文名称和正文使用 manifold 一词，但 Birkhoff polytope 是带边界和顶点的 convex polytope。它的相对内部可以视为光滑约束空间的一部分，整个集合则不是普通意义下无边界的 smooth manifold。

这不影响约束与算法本身，但在讨论 “manifold optimization” 时应避免把术语理解得过强。

---

## 5. 参数化与 Sinkhorn-Knopp 投影

### 5.1 生成动态 mappings

给定：

$$
\mathbf x_l\in\mathbb R^{n\times C},
$$

mHC 先 flatten：

$$
\vec x_l
=
\operatorname{vec}(\mathbf x_l)
\in
\mathbb R^{1\times nC}.
$$

归一化后：

$$
\vec x'_l
=
\operatorname{RMSNorm}(\vec x_l).
$$

论文 Eq. 7 定义 unconstrained logits：

$$
\widetilde H_l^{\mathrm{pre}}
=
\alpha_l^{\mathrm{pre}}
\left(
\vec x'_l\varphi_l^{\mathrm{pre}}
\right)
+
b_l^{\mathrm{pre}},
$$

$$
\widetilde H_l^{\mathrm{post}}
=
\alpha_l^{\mathrm{post}}
\left(
\vec x'_l\varphi_l^{\mathrm{post}}
\right)
+
b_l^{\mathrm{post}},
$$

$$
\widetilde H_l^{\mathrm{res}}
=
\alpha_l^{\mathrm{res}}
\operatorname{mat}
\left(
\vec x'_l\varphi_l^{\mathrm{res}}
\right)
+
b_l^{\mathrm{res}}.
$$

参数 shape 为：

$$
\varphi_l^{\mathrm{pre}},
\varphi_l^{\mathrm{post}}
\in
\mathbb R^{nC\times n},
$$

$$
\varphi_l^{\mathrm{res}}
\in
\mathbb R^{nC\times n^2}.
$$

其余 optimizer-owned 参数为：

$$
\alpha_l^{\mathrm{pre}},
\alpha_l^{\mathrm{post}},
\alpha_l^{\mathrm{res}}
\in\mathbb R,
$$

$$
b_l^{\mathrm{pre}},
b_l^{\mathrm{post}}
\in\mathbb R^{1\times n},
\qquad
b_l^{\mathrm{res}}
\in\mathbb R^{n\times n}.
$$

这里三个 $\alpha_l$ 是可训练标量 gate，三个 $b_l$ 是可训练的 input-independent static logits/bias；若 routing RMSNorm 使用 affine weight，该 scale 也是可训练参数，并可在优化 kernel 中吸收到 $\varphi_l$。相反，$\widetilde H_l^{(\cdot)}$ 和约束后的 $H_l^{(\cdot)}$ 都是由当前 token state 生成的中间张量，不会作为独立条目交给 optimizer。

**[复原推导]** 训练时直接使用原模型的主任务 loss：

$$
\mathcal L_{\mathrm{main}}
\longrightarrow
H_l^{(\cdot)}
\longrightarrow
\widetilde H_l^{(\cdot)}
\longrightarrow
\left\{\varphi_l^{(\cdot)},\alpha_l^{(\cdot)},b_l^{(\cdot)}\right\}.
$$

Sigmoid、20 步 Sinkhorn 及其 custom backward/recomputation 都位于这条计算图中；不需要额外 routing label，也不需要把约束残差作为 auxiliary loss。

**[复原推导]** 三个 projection 合并后每层主要新增：

$$
nC(n^2+2n)
$$

个线性权重。对 $n=4$：

$$
4C(16+8)=96C,
$$

相对主干中 $O(C^2)$ 的 Attention/FFN 权重通常很小。

### 5.2 对三个 mappings 施加不同约束

论文 Eq. 8：

$$
H_l^{\mathrm{pre}}
=
\sigma
\left(
\widetilde H_l^{\mathrm{pre}}
\right),
$$

$$
H_l^{\mathrm{post}}
=
2\sigma
\left(
\widetilde H_l^{\mathrm{post}}
\right),
$$

$$
H_l^{\mathrm{res}}
=
\operatorname{SinkhornKnopp}
\left(
\widetilde H_l^{\mathrm{res}}
\right).
$$

这里的 $\sigma$ 是逐元素 logistic Sigmoid：

$$
\sigma(z)
=
\frac{1}{1+e^{-z}},
\qquad
0<\sigma(z)<1.
$$

它不是标准差，也不是可训练参数。由此得到的约束是：

$$
H_l^{\mathrm{pre}}\in(0,1)^{1\times n},
$$

$$
H_l^{\mathrm{post}}\in(0,2)^{1\times n}.
$$

因此 $H_l^{\mathrm{pre}}$ 每个元素严格位于 $(0,1)$，$H_l^{\mathrm{post}}$ 因前面的系数 2 而逐元素位于 $(0,2)$。两者只受到**逐元素正值范围**约束，并不要求各元素之和为 1；也就是说，$\sigma$ 没有把它们变成 probability simplex。只有 $H^{\mathrm{res}}$ 经 Sinkhorn 被约束为近似双随机矩阵。

### 5.3 Sinkhorn-Knopp 过程

先把任意 logits 变为正矩阵：

$$
M^{(0)}
=
\exp
\left(
\widetilde H_l^{\mathrm{res}}
\right).
$$

这里 $\exp$ 逐元素作用，$M^{(0)}\in\mathbb R_{>0}^{n\times n}$ 是运行时起始矩阵。

然后交替做 column normalization 与 row normalization（论文 Eq. 9）：

$$
M^{(t)}
=
\mathcal T_r
\left(
\mathcal T_c
\left(
M^{(t-1)}
\right)
\right).
$$

其中 $t=1,\ldots,t_{\max}$ 是迭代编号，$\mathcal T_c$ 把每一列除以该列元素和，$\mathcal T_r$ 把每一行除以该行元素和；二者都是固定、可微、无参数的归一化算子。

当：

$$
t_{\max}\to\infty
$$

且矩阵满足适当条件时，迭代收敛到双随机矩阵。

论文实际使用：

$$
t_{\max}=20.
$$

### 5.4 有限迭代留下的误差

论文实现的最后一步是 row normalization，因此 row sums 更接近精确的 1；column sums 只有近似保证。

**[论文报告]** 在 27B 模型中：

- single-layer forward gain 基本保持 1；
- backward gain 对 1 有轻微偏离；
- 多层 composite backward gain 的峰值约 1.6；
- 仍比原始 HC 接近 3000 的峰值低约三个数量级。

**[综合判断]** 理论保证对应 exact projection；实际系统对应 finite-iteration approximation。正确实现应该同时记录：

$$
\epsilon_{\mathrm{row}}
=
\lVert H\mathbf1-\mathbf1\rVert_\infty,
$$

$$
\epsilon_{\mathrm{col}}
=
\lVert \mathbf1^\top H-\mathbf1^\top\rVert_\infty,
$$

而不是只检查输出是否有限。

---

## 6. 一个 $n=2$ 的最小例子

### 6.1 可交换但稳定的 residual mixing

令：

$$
H^{\mathrm{res}}
=
\begin{bmatrix}
0.8&0.2\\
0.2&0.8
\end{bmatrix}.
$$

它满足：

$$
H^{\mathrm{res}}\mathbf1=\mathbf1,
\qquad
\mathbf1^\top H^{\mathrm{res}}=\mathbf1^\top,
\qquad
H^{\mathrm{res}}\ge0.
$$

对两条 stream：

$$
\mathbf x=
\begin{bmatrix}
x^{(1)}\\
x^{(2)}
\end{bmatrix},
$$

传播后：

$$
\mathbf x'
=
\begin{bmatrix}
0.8x^{(1)}+0.2x^{(2)}\\
0.2x^{(1)}+0.8x^{(2)}
\end{bmatrix}.
$$

均值保持：

$$
\frac{x'^{(1)}+x'^{(2)}}2
=
\frac{x^{(1)}+x^{(2)}}2.
$$

差值则收缩：

$$
x'^{(1)}-x'^{(2)}
=
0.6
\left(
x^{(1)}-x^{(2)}
\right).
$$

这正好展示 mHC 的取舍：

- common information 原样传播；
- stream-specific difference 可以被稳定地混合或遗忘。

### 6.2 与 identity 和 uniform mixing 的边界

当：

$$
H^{\mathrm{res}}=I,
$$

两条 streams 完全独立保留。

当：

$$
H^{\mathrm{res}}
=
\frac12\mathbf1\mathbf1^\top,
$$

一次传播后两条 streams 完全相同。

mHC 的学习问题不是简单选择“稳不稳定”，而是在 Birkhoff polytope 内选择：

> 更接近 permutation/identity 的信息保留，还是更接近 uniform averaging 的信息融合。

---

## 7. 对论文稳定性指标的精确解读

### 7.1 Amax Gain Magnitude

论文使用两类指标：

- forward：row sums 的最大绝对值；
- backward：column sums 的最大绝对值。

这些指标直接检查 constant/common-mode signal 在 stream mixing 中的增益。

### 7.2 它不是完整 operator norm

“最大绝对 row sum”可能有两种不同定义：

$$
\max_i
\left|
\sum_j H_{ij}
\right|
$$

和 induced infinity norm：

$$
\lVert H\rVert_\infty
=
\max_i
\sum_j|H_{ij}|.
$$

论文文字与 Figure 8 标注对应前者，即先求带符号 row sum，再取绝对值。对含正负系数的原始 HC，它不能检测所有最坏方向，因为正负项可能抵消。

**[综合判断]** Amax gain 对“identity/common-mode 是否守恒”很有诊断力，但不等价于完整的 worst-case vector amplification。

建议复现时同时记录：

- $\lVert H\rVert_2$；
- $\lVert H\rVert_\infty$；
- $\lVert H\rVert_1$；
- 最大/最小 singular value；
- common-mode gain；
- stream-difference subspace 的 contraction。

---

## 8. 系统问题：HC 的 FLOPs 小不等于便宜

### 8.1 单层 forward I/O

论文 Table 2 只统计 residual stream maintenance，不包含主干
$\mathcal F$ 内部 I/O。

普通 residual connection 每 token：

| 操作 | Read | Write |
|---|---:|---:|
| Residual merge | $2C$ | $C$ |

HC 每 token：

| 操作 | Read | Write |
|---|---:|---:|
| 生成 $H^{pre},H^{post},H^{res}$ | $nC$ | $n^2+2n$ |
| 应用 $H^{pre}$ | $nC+n$ | $C$ |
| 应用 $H^{post}$ | $C+n$ | $nC$ |
| 应用 $H^{res}$ | $nC+n^2$ | $nC$ |
| Residual merge | $2nC$ | $nC$ |
| **合计** | $(5n+1)C+n^2+2n$ | $(3n+1)C+n^2+2n$ |

忽略 $n^2$ 小项，总 traffic 近似：

$$
(8n+2)C.
$$

相对普通 residual 的 $3C$：

$$
\text{I/O ratio}
\approx
\frac{8n+2}{3}.
$$

对 $n=4$：

$$
\frac{34}{3}
\approx
11.3.
$$

**[复原推导]** 论文概括为“约随 $n$ 增长”是对渐近关系的描述；具体常数并不小。多流方法的主战场是 memory traffic，而不是 arithmetic FLOPs。

### 8.2 Pipeline Parallel payload

单流 stage boundary activation：

$$
BSC.
$$

$n$-stream state：

$$
BSnC.
$$

论文明确指出，若直接传递完整多流状态，PP communication 近似增加 $n$ 倍。

这不是新增参数 sharding 能解决的问题，因为 payload 来自 activation。

---

## 9. 三组 infrastructure optimization

### 9.1 Kernel fusion

**[论文报告]** mHC 的 optimized path 使用混合精度：

| 对象 | 论文 dtype |
|---|---|
| Flattened hidden $\vec x_l$ | bfloat16 |
| Combined projection $\varphi_l$ | tfloat32 |
| Gating scalars $\alpha_l$ | float32 |
| Bias $b_l$ | float32 |
| Mapping logits/coefficients | float32 |

论文把 RMSNorm 的除法移动到 matrix multiplication 之后：

$$
\operatorname{RMSNorm}(x)\varphi
=
\frac{x}{r}\varphi
=
\frac{x\varphi}{r},
$$

并把 RMSNorm weight 吸收到 $\varphi$ 中，以减少高维状态上的独立 memory pass。

实现分为：

1. 生成三个 mapping logits；
2. 对小系数做 Sigmoid、scale 与 Sinkhorn；
3. 应用 $H^{pre}$；
4. 融合 $H^{res}$、$H^{post}$ 与 residual merge；
5. backward 中对 Sinkhorn intermediates on-chip recompute。

对 post/residual application 的 fusion，论文报告：

$$
\text{read}:
(3n+1)C
\rightarrow
(n+1)C,
$$

$$
\text{write}:
3nC
\rightarrow
nC.
$$

大部分相关 kernels 使用 TileLang 实现。

### 9.2 Selective recomputation

对连续 $L_r$ 个子层，论文不保存每层所有 mHC intermediate，而只长期保存 block 首个输入：

$$
\mathbf x_{l_0}\in\mathbb R^{n\times C}.
$$

主干函数：

$$
\mathcal F(H_l^{pre}\mathbf x_l,W_l)
$$

的输出仍按层保存；backward 时只重算相对轻量的 mHC path，不重算重型 Attention/FFN。

对于总共 $L$ 个子层，常驻与瞬时 activation 的近似和为（论文 Eq. 20）：

$$
M(L_r)
=
nC
\left\lceil
\frac{L}{L_r}
\right\rceil
+
(n+2)CL_r.
$$

忽略取整，最优 block size：

$$
L_r^\star
\approx
\sqrt{
\frac{nL}{n+2}
}.
$$

例如 $n=4,L=60$：

$$
L_r^\star
\approx
\sqrt{40}
\approx
6.3.
$$

**[复原推导]** 这解释了论文为什么说理论最优值常与每个 PP stage 的层数接近。

### 9.3 与 Pipeline Parallel 边界对齐

Recompute block 不能跨 PP stage，因为下一 stage 的中间状态不在本 rank。

论文选择：

- recompute boundary 与 PP stage boundary 对齐；
- 在本地缓存 stage 初始 activation；
- 让 stage 内 mHC recompute 不依赖再次进行 pipeline communication。

### 9.4 扩展 DualPipe overlap

论文基于 DualPipe 调度：

- 把 MLP 的 post/res mapping kernel 放在 dedicated high-priority compute stream；
- 避免对长时间运行的 Attention 操作使用 persistent kernel；
- 允许通信期间的 Attention compute 被更灵活地抢占/调度；
- 把多流 PP communication 与其他 forward/backward/weight-gradient 工作重叠。

### 9.5 6.7% 开销应该怎样解读

**[论文报告]** 论文称 $n=4$ 的 large-scale mHC training 只增加 6.7% time overhead。

但论文没有报告：

- GPU 型号；
- GPU 数量；
- 节点和网络拓扑；
- parallelism layout；
- microbatch 数；
- baseline tokens/s；
- mHC tokens/s；
- step-time breakdown；
- peak memory；
- 未优化、只 fusion、再 recompute、再 overlap 的逐项消融。

**[综合判断]** 6.7% 是有价值的 engineering outcome，但证据可复现性弱。它只能视为作者内部系统在未公开条件下的测量，不能直接当成跨硬件通用常数。

---

## 10. 实验设置

### 10.1 模型家族

**[论文报告]** 所有主要模型都是受 DeepSeek-V3 启发的 MoE，HC 与 mHC 都使用：

$$
n=4.
$$

模型配置：

| 配置 | Total params | Active params | Layers | Hidden $C$ | Training tokens |
|---|---:|---:|---:|---:|---:|
| 3B compute-scaling | 2.97B | 612M | 12 | 1280 | 39.3B |
| 9B compute-scaling | 9.18B | 1.66B | 18 | 1920 | 105B |
| 27B main | 27.0B | 4.14B | 30 | 2560 | 262B |
| 3B token-scaling | 2.97B | 612M | 12 | 1280 | 1.05T |

共有设置：

- sequence length：4096；
- vocabulary size：129280；
- Attention：MLA；
- LayerNorm：RMSNorm；
- gating factor init：0.01；
- Sinkhorn iterations：20；
- optimizer：AdamW；
- betas：$(0.9,0.95)$；
- weight decay：0.1；
- warmup：2000 steps。

### 10.2 各规模训练预算

| 配置 | Batch size | Steps | Base LR |
|---|---:|---:|---:|
| 3B / 39.3B tokens | 320 | 30,000 | $8.6\times10^{-4}$ |
| 9B / 105B tokens | 512 | 50,000 | $5.9\times10^{-4}$ |
| 27B / 262B tokens | 1280 | 50,000 | $4.0\times10^{-4}$ |
| 3B / 1.05T tokens | 2560 | 100,000 | $9.0\times10^{-4}$ |

### 10.3 未报告但影响复现的信息

- 预训练数据集名称、配比、去重和质量过滤；
- tokenizer 来源；
- dropout、gradient clipping 和 precision recipe 的完整设置；
- GPU/cluster 与 parallelism 配置；
- 随机种子与重复运行次数；
- checkpoint selection protocol；
- 下游 evaluation harness 版本。

因此实验可用于判断作者系统内的相对趋势，但还不足以做独立的端到端严格复现。

---

## 11. 实验一：HC 哪个 mapping 真正有用

论文 Table 1 逐步打开 HC mappings：

| $H^{res}$ | $H^{pre}$ | $H^{post}$ | Absolute loss gap |
|---:|---:|---:|---:|
| 关闭 | 关闭 | 关闭 | 0.000 |
| 开启 | 关闭 | 关闭 | -0.022 |
| 开启 | 开启 | 关闭 | -0.025 |
| 开启 | 开启 | 开启 | -0.027 |

关闭 mapping 时使用固定替代：

- $H^{pre}$：每项 $1/n$；
- $H^{post}$：全 1；
- $H^{res}$：identity。

**支持的结论**

- 主要收益来自可学习 residual stream mixing；
- learnable read/write 继续带来小幅增益；
- 只增加 streams 而不学习 $H^{res}$ 不能解释完整收益。

**证据边界**

- 表格没有给出具体模型规模和重复运行；
- 这是 HC component ablation，不是 mHC constraint ablation；
- 没有比较 unconstrained $H^{res}$、只 row-stochastic、只 column-stochastic、exact/approx doubly stochastic。

---

## 12. 实验二：27B 的质量与稳定性

### 12.1 Training loss 与 gradient norm

**[论文报告]**

- 原始 HC 在约 12k step 出现 loss surge；
- 同期 gradient norm 明显异常；
- mHC 的 gradient norm 与普通 baseline 类似；
- mHC 最终 training loss 相对 baseline 降低 0.021。

这是论文最有诊断力的一组结果，因为它同时对齐了：

$$
\text{unconstrained matrix products}
\rightarrow
\text{composite gain explosion}
\rightarrow
\text{gradient spike}
\rightarrow
\text{loss instability}.
$$

### 12.2 下游结果

论文 Table 4：

| Benchmark | Metric | Shots | Baseline | HC | mHC |
|---|---|---:|---:|---:|---:|
| BBH | EM | 3 | 43.8 | 48.9 | **51.0** |
| DROP | F1 | 3 | 47.0 | 51.6 | **53.9** |
| GSM8K | EM | 8 | 46.7 | 53.2 | **53.8** |
| HellaSwag | Accuracy | 10 | 73.7 | 74.3 | **74.7** |
| MATH | EM | 4 | 22.0 | **26.4** | 26.0 |
| MMLU | Accuracy | 5 | 59.0 | 63.0 | **63.4** |
| PIQA | Accuracy | 0 | 78.5 | 79.9 | **80.5** |
| TriviaQA | EM | 5 | 54.3 | 56.3 | **57.6** |

**直接读法**

- mHC 在 8/8 tasks 上超过 baseline；
- mHC 在 7/8 tasks 上超过 HC；
- 唯一例外是 MATH：mHC 26.0，低于 HC 26.4；
- 相对 HC，BBH 与 DROP 分别提高 2.1 和 2.3 个百分点。

论文将后二者写为 2.1% 与 2.3% gains。由于表中是 0–100 scale 的 benchmark score，更准确的表述是 **percentage points**。

**[复原推导]** 若仅为概览、对八种不同 metric 做未加权平均：

$$
\text{Baseline}=53.13,
\qquad
\text{HC}=56.70,
\qquad
\text{mHC}=57.61.
$$

该平均不是论文正式指标，且不同 benchmark metric 不完全可比，只能帮助观察整体方向。

### 12.3 能支持什么

结果支持：

- 多流 HC 的质量收益在 27B MoE 上仍存在；
- mHC 没有因为稳定约束而丢掉总体性能；
- 在这一运行中，mHC 比 HC 更稳定且多数下游任务更好。

结果不支持：

- mHC 在所有任务都优于 HC；
- 双随机约束是质量提升的唯一原因；
- 对其他模型家族或更大规模一定保持相同幅度；
- 结果具有统计显著性。

---

## 13. 实验三：Scaling evidence

### 13.1 Compute scaling

论文 Figure 6(a) 比较 3B、9B、27B compute-optimal 配置。

**[论文报告]** mHC 相对 baseline 的 loss gap 在三个规模上都保持为负，说明收益没有在 27B 时消失。

从图中还能看到：

- 绝对 loss gap 随 compute 增大有所收窄；
- relative loss ratio 始终低于 100%；
- 只有三个 model-size/data-budget 点。

**[综合判断]** 这支持“在测量范围内没有明显失效”，但不足以拟合新的 asymptotic scaling law，也不能证明任意更大规模仍保持相同收益。

### 13.2 Token scaling

论文 Figure 6(b) 跟踪同一个 3B 模型训练到约 1T tokens。

结果显示：

- mHC 在多个训练阶段都优于 baseline；
- 随训练继续，绝对 loss gap 逐渐缩小但未消失。

**[综合判断]** mHC 更像改变有限预算下的 optimization/representation efficiency，而不是让 loss gap 随 token 无限扩大。

### 13.3 “In-house large-scale”不是公开证据点

论文称额外内部大规模训练进一步支持结论，但没有给出：

- 模型规模；
- token 数；
- loss；
- stability curve；
- benchmark；
- hardware。

因此不能把它计入可独立审计的 scaling evidence。

---

## 14. 实验四：约束是否真的控制传播

### 14.1 原始 HC

Figure 3 与 Figure 8 显示：

- 单层 $H_l^{res}$ 已可能偏离 gain 1；
- 多层乘积会累积放大；
- composite gain 峰值接近 3000；
- 代表性 composite matrix 出现数百量级的正负元素。

### 14.2 mHC

Figure 7 与 Figure 8 显示：

- 单层 row/column gains 接近 1；
- forward common-mode gain 基本保持 1；
- finite Sinkhorn 导致 backward gain 轻微偏离；
- composite 最大偏离约 1.6；
- composite matrix 元素保持非负且规模受控。

### 14.3 证据强度

这是对 mHC 核心机制的直接测量，比单独比较 benchmark 更有说服力。

但仍有两个边界：

1. 使用的是 Amax common-mode gain，而不是完整 singular spectrum；
2. 只展示一个选定 sequence 的 token average，未报告不同样本、step 和 seed 的分布。

---

## 15. 贡献账本

| Contribution | 类型 | 论文位置 | 实际新意 | 未建立的结论 |
|---|---|---|---|---|
| 诊断 HC 深层不稳定 | mechanism + empirical finding | §3.1, Fig. 2–3 | 把 loss/grad spike 与 composite residual mapping 联系起来 | 未穷尽其他 instability 来源 |
| 用 Birkhoff polytope 约束 $H^{res}$ | mechanism | §4.1, Eq. 6 | 在可学习 stream mixing 与稳定 transport 之间取中间点 | 不等于完整 identity |
| Sinkhorn 动态参数化 | algorithm | §4.2, Eq. 7–9 | 为每 token 生成近似双随机 residual mapping | 没有迭代次数消融 |
| Positive pre/post mappings | mechanism | §4.1–4.2 | 避免正负权重直接抵消 | 没证明这是必要条件 |
| Fusion + mixed precision kernels | system | §4.3.1, Eq. 10–19 | 针对 mHC I/O path 的专用实现 | 缺少公开端到端 benchmark |
| Selective recomputation | system | §4.3.2, Table 3, Eq. 20 | 只重算轻量 mHC path，并推导最优 block size | 没给 peak-memory 实测表 |
| DualPipe overlap | system | §4.3.3, Fig. 4 | 针对多流 PP payload 和 stage-boundary recompute 调度 | 未单独量化收益 |
| 3B–27B MoE 实证 | empirical finding | §5, Fig. 5–8, Table 4–5 | 显示稳定性、质量和有限范围 scaling | 未覆盖 dense/vision/更多 recipe |

---

## 16. 这篇论文最有价值的 insight

### 16.1 Residual connection 的本质不是“加法”，而是受控的状态传输

普通 residual：

$$
x_{l+1}=x_l+F_l(x_l)
$$

之所以稳定，不只是因为有一个加号，而是因为旧状态的 transport operator 是：

$$
I.
$$

HC 改变这个 operator 后，必须重新回答：

- 什么量守恒；
- 哪个 norm 受控；
- 多层复合后是否仍受控；
- forward/backward 是否对称稳定。

mHC 的核心设计原则可以抽象为：

> 如果要学习 residual topology，应先定义一个对深度复合封闭的稳定算子集合。

### 16.2 Stability 与 plasticity 是集合设计问题

固定 identity：

- 最稳定；
- 不允许 stream exchange。

无约束矩阵：

- 最灵活；
- 可能深度爆炸。

Birkhoff polytope：

- 允许 permutation 与 convex mixing；
- 保留 common-mode；
- 禁止 transport expansion；
- 但可能收缩 stream differences。

这不是单个 initialization trick，而是对整个 optimization search space 的重构。

### 16.3 多流架构的瓶颈首先是 memory system

HC/mHC 的 coefficient 计算 FLOPs 小，但 state width 变成 $nC$ 后：

- HBM traffic 增长；
- saved activations 增长；
- PP payload 增长；
- kernel launch 与小矩阵操作变多；
- recomputation 和 overlap 变成架构可用性的组成部分。

因此宏观架构论文不能只报参数量和 FLOPs；必须把 memory traffic 与 distributed schedule 纳入方法本身。

### 16.4 “稳定”与“streams 有效分工”是两个不同目标

双随机约束能防止爆炸，却不能保证：

- 每条 stream 都被使用；
- streams 学到不同功能；
- mixing 不会接近 identity；
- mixing 不会趋向 uniform collapse；
- 增加 $n$ 会继续带来收益。

稳定性是多流 scaling 的必要条件，但不是充分条件。

---

## 17. 论文没有充分回答的问题

### 17.1 方法归因混杂

mHC 相比原始 HC 不只增加 Sinkhorn：

- 把完整 $nC$ state flatten 后生成动态 mappings；
- 改变 projection 参数 shape；
- 去掉原始 HC 公式中的 $\tanh$；
- 对 $H^{pre}$ 用 Sigmoid；
- 对 $H^{post}$ 用 $2\operatorname{Sigmoid}$；
- 对 $H^{res}$ 用 exponentiation + Sinkhorn。

论文没有提供逐项消融，因此：

$$
\text{mHC}-\text{HC}
$$

的质量差异不能唯一归因于 doubly stochastic constraint。

### 17.2 缺少 constraint ablation

理想对照至少包括：

- unconstrained；
- non-negative only；
- row-stochastic only；
- column-stochastic only；
- doubly stochastic；
- orthogonal；
- identity-plus-small-update；
- different Sinkhorn iterations。

这些对照能分辨究竟是：

- 非负性；
- mean conservation；
- backward conservation；
- spectral control；
- 还是更平滑的参数化

在起主要作用。

### 17.3 缺少统计可靠性

- 没有多 seed；
- 没有 error bars；
- 没有显著性检验；
- 27B 下游增益中有些只有 0.4–0.6 个百分点；
- 没有说明 checkpoint selection。

### 17.4 数据不可审计

论文只报告 token 数，没有公开 corpus。这使读者无法评估：

- 数据质量差异；
- benchmark contamination；
- 训练语言分布；
- 3B/9B/27B 是否使用严格同分布数据。

### 17.5 System claim 不完整

6.7% overhead 没有硬件和分解数据。没有：

- reference PyTorch；
- unfused mHC；
- fused-only；
- fused + recompute；
- fused + recompute + DualPipe

的阶梯式 benchmark。

因此无法判断每项优化的边际贡献，也无法知道 bottleneck 是否转移到了 PP communication 或主干 kernel。

### 17.6 外部有效性窄

论文只测试 DeepSeek-V3-like MoE。没有：

- dense Transformer；
- encoder-only；
- vision；
- diffusion；
- long-context；
- 小 batch 或 inference latency；
- 不同 $n$。

---

## 18. 后续证据与当前 artifact

### 18.1 Stream collapse

**[后续证据]** [*Analyzing Stream Collapse in Hyper-Connections*](https://arxiv.org/abs/2606.03483)（2026-06-02）报告：

- residual mixing 往往保持接近 identity；
- signal 与可解释特征可能集中到 dominant stream；
- nominal multi-stream architecture 可能退化得更像 single stream；
- 打破 stream initialization symmetry 可以缓解 dominant behavior。

这补充了 mHC 没回答的问题：

> 把系统稳定在 Birkhoff polytope 内以后，如何保证它真正使用这个空间？

### 18.2 扩大 $n$ 的瓶颈

**[后续证据]** [*xHC: Expanded Hyper-Connections*](https://arxiv.org/abs/2607.14530)（2026-07-16）报告：

- mHC 在 $n>4$ 后收益递减、成本快速上升；
- residual mixing generation 对 $n$ 的成本可达 cubic scaling；
- $n=4$ mHC 的每子层 memory traffic 约为 $34C$，与本文由 Table 2 得到的 $(8n+2)C$ 一致；
- xHC 用稀疏 stream update 与更丰富 write-back 尝试扩展到 $N=16$。

这说明 mHC 解决了稳定性，但没有解决 residual-stream scaling 的所有信息与系统瓶颈。

### 18.3 官方 kernel artifact

**[后续证据]** 截至 2026-07-23，DeepSeek 官方
[TileKernels repository](https://github.com/deepseek-ai/TileKernels/tree/36d9e45d38e204ebb87e6f6e833821eee0482fe5)
在 commit `36d9e45d38e204ebb87e6f6e833821eee0482fe5` 中公开了：

- `tile_kernels/mhc/sinkhorn_kernel.py`；
- pre/post/mix split 与 apply kernels；
- `multilayer_recompute_kernel.py`；
- `tile_kernels/torch/mhc.py` reference；
- 高层 autograd modeling wrappers。

该 artifact 发布于论文之后，可用于实现对照，但不等同于论文发布时提供了完整 27B training stack。

仓库当前说明的主要硬件/软件边界包括：

- NVIDIA SM90 或 SM100；
- CUDA 13.1+；
- PyTorch 2.10+；
- TileLang 0.1.9+。

---

## 19. 分布式实现 contract

以下除论文明确部分外均为 **[复原推导]**。最终 layout 必须以固定代码版本验证。

### 19.1 State ownership

逻辑状态：

$$
\mathbf x
\in
\mathbb R^{B\times S\times n\times C}.
$$

必须明确：

- $B$ 是否含 microbatch；
- $S$ 是否 local sequence shard；
- $n$ 是否每 rank 完整复制；
- $C$ 是否 TP-sharded；
- stream 维与 hidden 维的物理 layout。

### 19.2 Tensor Parallel

若 $C$ 按 TP 切分：

$$
\mathbf x^{(r)}
\in
\mathbb R^{B\times S\times n\times C_{\mathrm{local}}},
$$

且 $n$ 在每个 rank 本地完整保存，则 $H^{res}$ 的 stream mixing 可以本地执行。

但 mHC 的 dynamic coefficients 依赖 flatten 后的完整 $nC$ vector：

$$
\vec x_l\varphi_l.
$$

如果 $\varphi_l$ 与 $C$ 一起 column/row parallel，需要明确：

- partial GEMM 输出是否需要 all-reduce；
- RMSNorm 的 sum-of-squares 是否跨 TP；
- 每个 rank 生成的 $H^{pre},H^{post},H^{res}$ 是否完全一致。

与原始 HC 相比，这个 full-$nC$ dynamic projection 可能引入新的 TP reduction requirement。

### 19.3 Sequence/Context Parallel

Mapping 是 token-dependent，但不同 tokens 之间不直接混合，所以 token shard 可以独立生成 coefficients。

需要验证：

- RMSNorm 是否只跨 hidden/stream 维；
- CP attention 的 gather/scatter 前后保持哪种 state layout；
- recompute 是否能在相同 autocast 和 RNG state 下重现 coefficients。

### 19.4 Pipeline Parallel

论文明确指出完整多流 payload 近似是 baseline 的 $n$ 倍。

应记录：

$$
\text{payload bytes}
=
B_{\mu}
\times
S
\times
n
\times
C
\times
\text{bytes(dtype)}.
$$

例如 BF16：

$$
\text{payload bytes}
=
2B_{\mu}SnC.
$$

还应区分：

- forward send/recv；
- backward gradient send/recv；
- 双向 DualPipe overlap；
- stage-boundary resident activation；
- communication stream 与 high-priority compute stream 的同步事件。

### 19.5 Data Parallel / FSDP

mHC 参数相对主干很小，FSDP 对新增参数的节省有限。主要内存来自：

- multi-stream state；
- saved branch outputs；
- stage-boundary activation；
- transient recompute block。

因此 parameter memory 与 activation memory 必须单独报告。

### 19.6 混合精度 contract

Reference 与 fused path 至少应固定：

| 环节 | 建议 contract |
|---|---|
| Hidden storage | BF16 或训练主 dtype |
| RMS sum-of-squares | FP32 accumulation |
| Dynamic projection accumulation | 与论文 TF32/FP32 语义对齐 |
| Mapping logits | FP32 |
| `exp`/Sinkhorn normalization | FP32 |
| Mapping application | 明确乘法与 accumulation dtype |
| Stream collapse | 建议 FP32 accumulation 后 cast |
| Backward recompute | 与 forward 完全相同的 iteration/order |

---

## 20. 最小 CPU reference 实现路线

### 20.1 第一阶段：只实现 exact math

使用：

$$
B=2,\quad S=3,\quad n\in\{1,2,4\},\quad C=8
$$

实现：

1. flatten + RMSNorm；
2. mapping logits；
3. Sigmoid pre/post；
4. Sinkhorn residual mapping；
5. read；
6. branch function；
7. carry + write。

不要一开始实现 fusion 或 custom backward。

### 20.2 Shape invariants

| Tensor | Shape |
|---|---|
| $\mathbf x_l$ | $[B,S,n,C]$ |
| $\vec x_l$ | $[B,S,nC]$ |
| $H^{pre}$ | $[B,S,1,n]$ |
| $H^{post}$ | $[B,S,1,n]$ |
| $H^{res}$ | $[B,S,n,n]$ |
| Branch input | $[B,S,C]$ |
| Branch output | $[B,S,C]$ |
| $\mathbf x_{l+1}$ | $[B,S,n,C]$ |

### 20.3 Sinkhorn correctness

对不同 iteration 数：

$$
t\in\{1,2,5,10,20,50\},
$$

记录：

$$
\epsilon_{\mathrm{row}},
\quad
\epsilon_{\mathrm{col}},
\quad
\lVert H\rVert_2,
\quad
\min(H).
$$

预期：

- 所有元素为正；
- row/column errors 随迭代下降；
- exact tolerance 下 $\lVert H\rVert_2$ 接近 1；
- finite iteration 的 column error 可能大于 row error。

### 20.4 Special cases

#### $n=1$

Sinkhorn 后必须：

$$
H^{res}=[1].
$$

这是论文声称恢复普通 identity mapping 的最小检查。

#### Equal streams

若：

$$
x^{(1)}=\cdots=x^{(n)},
$$

则 residual transport 后应保持完全相等。

#### Mean preservation

比较：

$$
\frac1n\sum_i x_i
$$

和：

$$
\frac1n\sum_i(H^{res}x)_i.
$$

### 20.5 Gradient checks

用 FP64 reference 做：

- `torch.autograd.gradcheck`；
- autograd Sinkhorn 与显式 Jacobian-vector product 对比；
- finite iterations 的 backward；
- recompute path 与 save-all path 的 input/parameter gradients 对比。

### 20.6 Deep composition

随机生成 60 个 projected matrices：

$$
P_{60}=H_{60}\cdots H_1.
$$

检查：

- exact/高迭代版本的 row/column sums；
- finite-20 误差随深度如何累积；
- spectral norm；
- second singular value；
- stream variance。

最后一项用于检测“稳定但 collapse”。

---

## 21. 高信息量的后续实验

### 21.1 约束因素拆分

**标签：[扩展假设]**

**提案**

比较：

1. unconstrained；
2. positive only；
3. row-stochastic；
4. column-stochastic；
5. doubly stochastic；
6. orthogonal；
7. identity-plus-doubly-stochastic update。

**推理链**

当前 mHC 同时改变多个因素，无法知道收益来自哪里。逐个约束能分离：

- forward conservation；
- backward conservation；
- non-negativity；
- norm control；
- mixing capacity。

**预测**

- row-only 对 forward common-mode 更稳；
- column-only 对 backward common-mode 更稳；
- doubly stochastic 同时最稳；
- orthogonal 保留 norm 但可能允许负系数和 cancellation；
- identity-plus-mixing 可能减轻 stream collapse。

**证伪条件**

若这些约束在相同参数化下没有可区分的 propagation/optimization 行为，则当前机制解释不完整。

**最小实验**

24–96 层、$n=4$ 的小型语言模型；相同 seed、参数量与 token budget，报告 loss、grad norm、singular spectrum、stream diversity。

**风险**

不同 projection 的优化难度和 kernel 成本不同，需要同时报告 quality 与 system cost。

### 21.2 自适应 Sinkhorn iteration

**标签：[扩展假设]**

**提案**

不用固定 20 次，而根据：

$$
\max
\left(
\epsilon_{\mathrm{row}},
\epsilon_{\mathrm{col}}
\right)
$$

达到阈值后提前停止；或者按层深、训练阶段调整 iteration budget。

**推理链**

不同 token/logit matrix 的 condition 不同。固定 20 次可能：

- 对容易矩阵浪费计算；
- 对困难矩阵约束不足。

**预测**

在相同最大误差下减少平均迭代次数，或在相同时间成本下降低 composite gain。

**证伪条件**

动态控制流导致 kernel divergence/launch overhead，端到端速度反而下降。

**最小实验**

先离线统计每层每 token 达到 $10^{-3},10^{-4},10^{-5}$ tolerance 所需迭代分布，再决定是否值得写 fused adaptive kernel。

**风险**

GPU SIMT 对不规则迭代不友好，可能需要分桶而不是逐 token early stop。

### 21.3 稳定性与 stream diversity 的双目标

**标签：[扩展假设]**

**提案**

在双随机约束之外，增加轻量 diversity objective，例如：

- 限制 second singular value 过快趋零；
- 最大化 streams 间的有效 rank；
- 对 stream usage entropy 设下界；
- 用非对称 initialization 打破 permutation symmetry。

**推理链**

双随机约束防爆炸，但 uniform averaging 也位于 Birkhoff polytope，可能让 streams collapse。

**预测**

- stream covariance effective rank 提高；
- dominant-stream ratio 降低；
- 在较大 $n$ 时收益更明显；
- 不破坏 composite gain 稳定性。

**证伪条件**

diversity 指标提高但 validation loss/下游性能无改善，说明表征差异不等于有用分工。

**最小实验**

在 $n=4,8$ 上比较 symmetric init、asymmetric init 与 diversity regularization，记录 stream cosine、usage、effective rank 和 loss。

**风险**

过度追求差异可能阻止有益的信息共享。

### 21.4 PP boundary compression

**标签：[扩展假设]**

**提案**

只在 PP stage boundary 对多流 state 做可逆或近似压缩：

$$
[B,S,n,C]
\rightarrow
[B,S,k,C],
\qquad
k<n.
$$

候选包括：

- learned low-rank stream basis；
- anchor + delta representation；
- quantized stream differences；
- 只通信被下一 stage 高权重读取的 streams。

**推理链**

PP payload 是 $n$ 倍增长；stage 内保留完整 state、边界压缩可能改善跨节点扩展。

**预测**

在小于 0.1% loss degradation 下显著降低 PP communication bytes 与 bubble。

**证伪条件**

边界压缩误差跨 stage 累积，或 encode/decode kernel 抵消通信收益。

**最小实验**

先用离线 activation trace 测 rank/quantization error，再做两 stage Gloo/CUDA microbenchmark。

**风险**

改变数值 contract，可能无法与原始 checkpoint 兼容。

---

## 22. 证据账本

| Claim | 证据 | 强度 | 主要缺口 |
|---|---|---|---|
| 原始 HC 在 27B 出现训练不稳定 | Fig. 2 | 中–强 | 单次运行、无 seed 分布 |
| HC composite gain 可接近 3000 | Fig. 3, Fig. 8 | 强 | 指标不是完整 operator norm |
| 双随机 $H^{res}$ 对 common-mode/mean 守恒 | Eq. 6 + 数学推导 | 强 | 实现只有近似投影 |
| 双随机矩阵 non-expansive 且复合闭包 | §4.1 | 强 | “norm preservation”措辞偏强 |
| 20 次 Sinkhorn 把 composite gain 控制到约 1.6 | Fig. 7–8, §5.4 | 中–强 | 单 sequence average |
| mHC 最终 loss 比 baseline 低 0.021 | Fig. 5 | 中 | 无多 seed、无 exact loss table |
| mHC 在 8/8 下游任务超过 baseline | Table 4 | 中 | 数据与 eval harness 未公开 |
| mHC 在 7/8 任务超过 HC | Table 4 | 中 | 部分差值小、无方差 |
| 收益从 3B 保持到 27B | Fig. 6(a) | 中 | 仅三个点 |
| 3B 训练到 1T tokens 仍有收益 | Fig. 6(b) | 中 | 仅一个规模与一次运行 |
| Fusion 降低 mapping application I/O | §4.3.1 | 强（解析） | 无独立实测 bandwidth |
| Recompute 最优 block size Eq. 20 | §4.3.2 | 强（模型内） | 无实测 memory curve |
| 大规模训练 overhead 仅 6.7% | §4.3 | 弱–中 | 无硬件与 benchmark 表 |
| mHC 可能出现 stream underutilization | 后续 stream-collapse paper | 中 | 不属于原论文证据 |
| $n>4$ 有信息与系统瓶颈 | 后续 xHC paper | 中 | 不属于原论文证据 |

---

## 23. 推荐阅读顺序

如果只用 20 分钟：

1. Eq. 3–4：看懂 HC 的多层危险项；
2. Figure 2–3：把 loss spike、grad norm 与 composite gain 对齐；
3. Eq. 6：理解 doubly stochastic constraint；
4. Eq. 8–9：看 Sinkhorn 参数化；
5. Figure 7–8：确认 finite projection 的实际稳定性；
6. Table 4：看质量是否保留；
7. Table 2、Eq. 20：理解为什么必须做系统优化。

如果准备实现：

1. 先写 CPU reference；
2. 验证 $n=1$、equal-stream 与 mean conservation；
3. 对 finite Sinkhorn 做 row/column error 与 `gradcheck`；
4. 验证 60 层 composite；
5. 再实现 save-all 与 recompute 等价性；
6. 最后阅读 pinned TileKernels artifact 的 reference 和 fused path。

---

## 24. 最终评价

mHC 对原始 HC 的贡献不只是“加一个 Sinkhorn”。它真正补上了可学习 residual topology 必须面对的两类问题：

1. **数学问题**：深层连续复合时，什么结构可以同时允许 mixing、保持关键守恒量并控制 operator gain；
2. **系统问题**：当 residual state 扩成 $n$ 倍后，如何用 fusion、recompute 和 communication overlap 对抗 memory wall。

最强证据是稳定性链条：

$$
\text{HC 无约束}
\rightarrow
\text{composite gain}\approx3000
\rightarrow
\text{gradient/loss instability},
$$

$$
\text{mHC 近似双随机}
\rightarrow
\text{composite gain}\lesssim1.6
\rightarrow
\text{baseline-like gradient profile}.
$$

这比单纯的 benchmark 增益更有解释力。

但 mHC 的结论也需要准确收窄：

- 它恢复的是 common-mode identity 与 mean conservation，不是完整 identity matrix；
- 它保证 transport non-expansion，不保证所有 stream differences 被保留；
- 它解决了爆炸风险，不保证多条 streams 真正形成有效分工；
- 它展示了 27B MoE 可行性，但 6.7% overhead 和更大规模扩展仍缺少公开、可复现的系统证据。

因此最稳妥的判断是：

**mHC 把 Hyper-Connections 从一个有潜力但不受控的架构想法，推进成了具有明确稳定性约束和系统实现路径的方法；同时，它也把下一阶段的问题暴露得更清楚——如何在“稳定、不 collapse、可扩大 $n$、可高效分布式训练”之间取得共同成立的解。**
