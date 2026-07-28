---
title: "xHC：让 Hyper-Connections 的 residual streams 从 4 条扩到 16 条"
description: "拆解 xHC 的 dense read、active-stream mixing 与 sparse write，核对扩展率、I/O、Flash kernel 报告和证据边界。"
topic: "residual"
section: "methods"
slug: "xhc-expanded-hyper-connections"
date: 2026-07-27
updated: 2026-07-27
cutoff: 2026-07-24
order: 13
readtime: 49
source:
  repository: "J-shang/residual"
  path: "papers/04-xhc-expanded-hyper-connections.md"
  url: "https://github.com/J-shang/residual/blob/c55583707ab645bef7408c588fc33cbf91b809a5/papers/04-xhc-expanded-hyper-connections.md"
  revision: "c55583707ab645bef7408c588fc33cbf91b809a5"
  syncedAt: "2026-07-28"
  contentHash: "sha256:72938eea430445872e7a6d7c2159e787a3c79c265043928b8788f314300b63c5"
  manifest: "residual"
  managed: true
---
<!-- paper-order: 04 -->

> **论文**：Xiangdong Zhang、Xiaohan Qin、Sunan Zou et al., *xHC: Expanded Hyper-Connections*<br>
> **机构**：上海交通大学、Dots Studio / 小红书、USTC、北京大学、CUHK 等<br>
> **版本**：arXiv:2607.14530v1，2026-07-16<br>
> **状态**：technical report / arXiv preprint<br>
> **主来源**：[arXiv HTML](https://arxiv.org/html/2607.14530v1) · [PDF](https://arxiv.org/pdf/2607.14530v1)<br>
> **官方 artifact**：[aHapBean/xHC](https://github.com/aHapBean/xHC/tree/7890266d5cd648811b6783029ee6b5031cd209db)，`main` commit `7890266d5cd648811b6783029ee6b5031cd209db`<br>
> **artifact 状态**：截至核对日只有 README、图片和 technical report PDF；没有训练代码、Triton kernel source、checkpoint、配置文件或 license<br>
> **阅读范围**：完整 27 页，包括 Eq. 1–37、Algorithm 1–2、Figure 1–5、Table 1–12、Appendix A–E 和 arXiv LaTeX source<br>
> **后续信息截止日期**：2026-07-24<br>
> **前置阅读**：[Hyper-Connections](/topics/residual/hyper-connections/) · [mHC](/topics/residual/mhc/)

## 证据标签

- **[论文报告]**：论文正文、公式、图表或附录直接给出的内容。
- **[复原推导]**：从论文定义可以独立推出、但原文没有直接写出的结论。
- **[综合判断]**：结合数学、实验和系统约束形成的解释。
- **[后续证据]**：论文发布后的 primary source 或官方 artifact。
- **[扩展假设]**：论文之外、需要新实验验证的设计。
- **[待验证]**：当前来源不足，不能当作已经建立的事实。

---

## 先给结论

### 30 秒版

HC / mHC 把单条 residual stream 扩成 $N$ 条并行 streams。mHC 在 $N=4$ 时有效且稳定，但直接继续增大 $N$ 会遇到两个问题：

1. **信息供给不足**：一次子层计算只产生一个输出向量 $\mathrm{out}$，所有 streams 只是以不同标量写入同一个方向；
2. **动态 mixing 太贵**：从 $NC$ 维状态生成 $N\times N$ residual mapping，需要 $O(N^3C)$ 的 projection。

xHC 分别处理这两个瓶颈：

$$
\boxed{
\text{xHC}
=
\text{temporal feature augmentation}
+
\text{sparse residual-stream update}
}
$$

- MLP 输出经 3 个 causal depthwise convolutions 形成 4 个 write-back components；
- 模型仍保存并读取全部 $N=16$ 条 streams；
- 每个子层只选择 $k=4$ 条 active streams 做 residual mixing 和 write-back；
- active residual mapping 只有 $k\times k$，其生成成本从 $O(N^3C)$ 降为 $O(k^3C)$。

默认更新可以概括为：

对某个 token，先把 router 选出的 $k$ 条 active stream indices 明确定义为：

$$
\mathcal I
=
(\mathcal I_1,\ldots,\mathcal I_k),
\qquad
\mathcal I_j\in\{1,\ldots,N\},
\qquad
|\mathcal I|=k.
$$

按这些 indices 从完整状态中取出 active state，并记对应的 write-back routing weights 为：

$$
X_{\mathrm{active}}
=
\operatorname{Gather}(X_l,\mathcal I)
\in
\mathbb R^{k\times C},
\qquad
p=(p_1,\ldots,p_k)
\in
\mathbb R^k.
$$

再令：

$$
\mathcal H_l^{\mathrm{pre}}
=
(h_{l,1}^{\mathrm{pre}},\ldots,h_{l,N}^{\mathrm{pre}})
\in
\mathbb R^{1\times N}
$$

表示 dense-read mapping，则：

$$
\mathrm{input}_l
=
\sum_{i=1}^{N}
h_{l,i}^{\mathrm{pre}}x_{l,i},
$$

$$
\mathrm{out}_l
=
\mathcal F_l(\mathrm{input}_l),
$$

$$
X_{\mathrm{active}}^{\mathrm{new}}
=
\mathcal H_l^{\mathrm{res}}X_{\mathrm{active}}
+
p\odot
\left(
\mathcal H_l^{\mathrm{post}}\,
\mathrm{out}_{\mathrm{aug}}
\right).
$$

这里：

- $X_l\in\mathbb R^{N\times C}$ 是某个 token 的完整多流状态；
- $\mathcal I_j$ 是第 $j$ 条 active stream 在完整 $N$-stream state 中的索引；
- $X_{\mathrm{active}}$ 是按 $\mathcal I$ gather 后的 active state；
- $h_{l,i}^{\mathrm{pre}}$ 是第 $i$ 条 stream 的 dense-read coefficient；
- $\mathcal F_l$ 是当前 Attention 或 MLP/MoE branch；
- $\mathcal H_l^{\mathrm{res}}\in\mathbb R^{k\times k}$ 只混合 active streams；
- $\mathcal H_l^{\mathrm{post}}\in\mathbb R^{k\times K_r}$ 把 $K_r$ 个 write-back components 写入 active streams；
- $p\in\mathbb R^k$ 是 fixed-plus-routed routing weight；
- $\mathrm{out}_{\mathrm{aug}}\in\mathbb R^{K_r\times C}$ 是当前 branch 的 augmented write-back components；
- 未被选中的 $N-k$ 条 streams 在这一子层保持完全不变。

### 最重要的实验结论

**[论文报告]**

- 18B-total / 1.7B-activated MoE：最终 training loss 为 xHC 1.758、mHC 1.776、vanilla 1.799；12 项下游平均分为 48.8、44.8、40.6。
- 28B-total / 2.7B-activated MoE：12 项下游平均分为 xHC 53.6、mHC 50.5、vanilla 47.8。
- 2.5B MoE 的 $N$-sweep 中，从 $N=4$ 增到 $N=16$：
  - mHC：loss 只降 0.006，training FLOPs 增加 32%；
  - xHC：loss 降 0.012，training FLOPs 只增加 4%。
- 四个 compute scale 的拟合声称：达到相同 loss 时，vanilla 和 mHC 分别需要 xHC 的 $1.50\times$ 和 $1.19\times$ compute。
- 10B ablation 中，dense mHC $N=16$ 的 validation loss 为 1.998；加入 temporal augmentation 后为 1.984；再换成完整 sparse xHC 后为 1.983，同时 FLOPs overhead 从 20.1% 降到 3.3%。

### 这篇论文真正贡献了什么

1. **诊断**：把 mHC 的 large-$N$ 饱和拆成 information-supply 与 cost 两个瓶颈。
2. **写回基扩展**：不用重复计算 $N$ 次大子层，而用低成本 causal depthwise convolution 把一个输出扩成多个 temporal components。
3. **非对称稀疏化**：read 保持 dense，residual mixing 和 write-back 变成 sparse。
4. **大 $N$ 参数化**：默认 $N=16,k=4,m=2,K_r=4$，保留 active-set 上的 Sinkhorn mixing。
5. **部署变体**：xHC-Flash 跨相邻子层共享 routing 和 full-state read，并用精确 correction 恢复后续子层输入。
6. **系统设计**：BF16/FP32 边界、router/pre projection fusion、active gather、Sinkhorn recomputation、mapping application fusion 和 temporal-conv fusion。
7. **规模证据**：18B、28B MoE 主结果，2.5B $N$-sweep，10B ablation，以及 Muon compatibility。

### 最重要的精确化

xHC 所谓“只更新 4 条 streams”，不是只使用 4 条 streams：

$$
\underbrace{\text{read all }N}_{\text{dense information access}}
\qquad
\underbrace{\text{update only }k}_{\text{sparse state mutation}}.
$$

如果把 read 也改成 Top-$k$，上一层写进某条 stream 的信息可能在下一层完全不可见。论文的 ablation 支持 dense read 是质量所需，而不仅是实现细节。

第二个精确化是：$N=16$ 不表示 Attention/MLP 的 hidden width 也变成 $16C$。大子层仍接收和输出 $C$ 维向量；扩张的是跨 depth 持久保存的 state：

$$
[B,S,C]
\quad\longrightarrow\quad
[B,S,N,C].
$$

这解释了为什么主干 matmul FLOPs 增幅不大，但 activation storage、memory traffic 和 pipeline payload 可能显著增大。

第三个精确化是：xHC-Flash 的 dense-read correction 在它自己的约束下是精确的，但 xHC-Flash **不与 full xHC 函数等价**。相对 full xHC，它改变了：

- routing decision 的刷新频率；
- pre-mapping 依赖的状态；
- residual mixing 出现的位置。

### 最重要的保留意见

- 论文没有公开训练数据集名称、配比、去重、training-token 数；主表中 training tokens 全部是 `--`。
- 18B 的 global batch size 也未给出。
- 没有多 seed、误差条、置信区间或显著性检验。
- scaling law 每种方法只有 4 个点，固定 $E=0.72$，没有公开 raw loss、fit code、残差或参数不确定性。
- 对“额外 streams 冗余”的证据主要是 loss ablation，没有直接测量 stream similarity、effective rank、互信息或 utilization。
- 官方仓库尚未发布论文所述训练实现和 fused Triton kernels，无法独立核对 reference/fused numerical contract。
- 论文给出相对 wall-clock overhead，但没有 GPU 型号、节点数、并行布局、batch、tokens/s 或 profiler 明细。
- 所有主要质量实验都是 DeepSeekMoE-style MoE；不能直接外推到 dense LLM、视觉模型或其他 optimizer/normalization recipe。
- temporal convolution 的 decode-state、逐 token decode latency 和 sequence-parallel halo communication 没有被分析。

---

## 阅读前：符号、shape 与身份

论文在单 token 公式中省略 batch 和 sequence 维。实现时建议固定：

$$
X_l
\in
\mathbb R^{B\times S\times N\times C}.
$$

后文会在关键公式旁再次解释新符号；本表先给出整体对象关系。

| 符号 | 含义 | 类型或 shape | 作用域 | 身份 |
|---|---|---:|---|---|
| $l,L$ | 当前子层、Transformer layer 总数 | 整数 | depth 维 | 索引/架构配置 |
| $B,S,C$ | batch size、sequence length、每条 stream 的 hidden width | 正整数 | batch/模型 | 运行配置或架构超参数 |
| $N$ | residual stream expansion rate | 正整数 | 通常全模型固定 | 架构超参数；主设置为 16 |
| $k$ | 每个 token、每个子层被更新的 active stream 数 | 正整数，$k\le N$ | routing | 架构超参数；主设置为 4 |
| $m$ | $k$ 条 active streams 中始终 active 的数量 | 整数，$0\le m\le k$ | routing | 架构超参数；主设置为 2 |
| $X_l$ | 第 $l$ 个子层入口的完整多流 state | $[N,C]$ 或 $[B,S,N,C]$ | 每层、每 token | 运行时张量 |
| $x_{l,i}$ | $X_l$ 的第 $i$ 条 stream | $[C]$ 或 $[B,S,C]$ | 每层、每 stream | 运行时张量 |
| $\mathcal F_l(\cdot,\mathcal W_l)$ | Attention 或 MLP/MoE branch | $[C]\to[C]$ | 每个子层 | $\mathcal W_l$ 是普通主干参数 |
| $\mathcal H_l^{\mathrm{pre}}$ | 从全部 $N$ 条 streams 读出一个 branch input | $[1,N]$ | 每层、每 token | input-dependent 运行时 mapping |
| $\mathcal H_l^{\mathrm{res}}$ | active streams 之间的 residual mixing | $[k,k]$ | 每层、每 token | Sinkhorn 生成的运行时 mapping |
| $\mathcal H_l^{\mathrm{post}}$ | 把 augmented branch outputs 写回 active streams | $[k,K_r]$ | 每层、每 token | input-dependent 运行时 mapping |
| $\mathrm{out}$ | Attention/MLP 的原始输出 | $[C]$ 或 $[B,S,C]$ | 当前子层 | 运行时张量 |
| $r$ | temporal convolution branch 数 | 非负整数 | MLP side | 超参数；主设置为 3 |
| $\kappa_j$ | 第 $j$ 个 causal depthwise-conv kernel size | 正整数 | MLP side | 超参数；主设置为 $4,8,12$ |
| $K_r=r+1$ | write-back components 数，含原始 $\mathrm{out}$ | 正整数 | 当前子层 | Attention 为 1，MLP 为 4 |
| $g_j$ | 第 $j$ 个 depthwise-conv 输出 | 与 $\mathrm{out}$ 相同 | temporal augmentation | 运行时张量 |
| $v_j$ | Gram–Schmidt 后第 $j$ 个 write-back component | 与 $\mathrm{out}$ 相同 | temporal augmentation | 运行时张量 |
| $\mathrm{out}_{\mathrm{aug}}$ | 堆叠后的 write-back basis | $[K_r,C]$ 或 $[B,S,K_r,C]$ | 当前子层 | 运行时张量 |
| $s$ | router 为全部 $N$ 条 streams 生成的 sigmoid score | $[N]$ 或 $[B,S,N]$ | 每层、每 token | 运行时张量，不是参数 |
| $\mathcal I=(\mathcal I_1,\ldots,\mathcal I_k)$ | 当前 active stream indices | $k$ 个整数 | 每层、每 token | hard Top-$k$ 运行时结果 |
| $p_j$ | 第 $j$ 条 active stream 的 write-back gate | 标量 | 每层、每 token | fixed stream 为 1；routed stream 为 sigmoid score |
| $X_{\mathrm{active}}$ | 按 $\mathcal I$ 从 $X_l$ gather 的 state | $[k,C]$ 或 $[B,S,k,C]$ | 当前子层 | 运行时张量 |
| $W^r$ | full-state router projection | $[NC,N]$ | 每个子层 | optimizer 直接更新的参数 |
| $W^{\mathrm{pre}}$ | full-state pre-mapping projection | $[NC,N]$ | 每个子层 | optimizer 直接更新的参数 |
| $W^{\mathrm{res}}$ | active-state residual-mapping projection | $[kC,k^2]$ | 每个子层 | optimizer 直接更新的参数 |
| $W^{\mathrm{post}}$ | active-state post-mapping projection | $[kC,kK_r]$ | 每个子层 | optimizer 直接更新的参数 |
| $\alpha^{\mathrm{pre/res/post}}$ | input-dependent mapping path 的幅度 gate | 标量 | 每个 mapping、每个子层 | 可训练参数；初始化为 0.01 |
| $b^{\mathrm{pre}}$ | pre-mapping 的 static logits | $[N]$ | 每个子层 | 可训练参数 |
| $b^{\mathrm{res}}$ | residual-mapping 的 static logits | $[k,k]$ | 每个子层 | 可训练参数 |
| $b^{\mathrm{post}}$ | post-mapping 的 static logits | $[k,K_r]$ | 每个子层 | 可训练参数 |
| $w_j$ | 第 $j$ 个 causal depthwise-conv kernel | $[\kappa_j,C]$ | 每个 MLP sublayer | 可训练参数 |
| $\operatorname{SK}$ | Sinkhorn–Knopp normalization | $[k,k]\to[k,k]$ | residual mapping | 固定可微重参数化 |
| $\sigma$ | logistic sigmoid | 标量逐元素 | router/pre/post | 固定可微函数，不是参数 |
| $\operatorname{LN}$ / $\operatorname{RMSNorm}$ | router / mapping generator 的 normalization | shape 不变 | 每 token | 算子；affine weight 若启用则可训练 |
| $\mu,\nu$ | full-state feature mean、second moment | 标量/每 token | fused normalization | FP32 运行时统计量 |
| $\alpha$（Flash） | Attention write-back 对预计算 MLP readout 的 correction coefficient | 每 token 标量 | xHC-Flash | 运行时张量；不要与 mapping gate $\alpha^{(\cdot)}$ 混淆 |
| $C_{\mathrm{train}}$ | scaling law 中的 training FLOPs | 正实数 | scaling experiment | 自变量；不要与 hidden width $C$ 混淆 |

参数与运行时 mapping 的关系是：

$$
\left\{
W^r,
W^{\mathrm{pre}},
W^{\mathrm{res}},
W^{\mathrm{post}},
\alpha^{(\cdot)},
b^{(\cdot)},
w_1,\ldots,w_r,
\mathcal W_l
\right\}_{\text{optimizer-owned parameters}}
$$

$$
\Downarrow
$$

$$
\left\{
s,\mathcal I,p,
\mathcal H^{\mathrm{pre}},
\mathcal H^{\mathrm{res}},
\mathcal H^{\mathrm{post}},
\mathrm{out}_{\mathrm{aug}},
X_{\mathrm{active}}
\right\}_{\text{input-dependent runtime tensors}}.
$$

**[复原推导]** $\mathcal H^{(\cdot)}$ 不是 optimizer 直接保存的一组自由矩阵；它们每次 forward 都由当前 token 的 residual state 动态生成。

---

## 1. 从最小例子理解论文诊断

### 1.1 mHC 为什么可能在大 $N$ 时浪费 streams

假设某个 MLP 输出：

$$
\mathrm{out}
\in
\mathbb R^C.
$$

mHC 写入第 $i$ 条 stream 的增量是：

$$
\Delta x_{l,i}
=
h_{l,i}^{\mathrm{post}}\,
\mathrm{out}.
$$

即使 $h_{l,i}^{\mathrm{post}}$ 随输入、token 和 stream 改变，所有新信息仍位于同一个一维子空间：

$$
\operatorname{span}
\left\{
\Delta x_{l,1},
\ldots,
\Delta x_{l,N}
\right\}
\subseteq
\operatorname{span}\{\mathrm{out}\}.
$$

假设 $N=4$，四条 streams 的系数分别是：

$$
(0.2,\ 0.8,\ 1.1,\ 1.7).
$$

它们写入的是：

$$
0.2\,\mathrm{out},
\quad
0.8\,\mathrm{out},
\quad
1.1\,\mathrm{out},
\quad
1.7\,\mathrm{out}.
$$

数值不同，但方向完全共线。增大 $N$ 会增加不同历史权重的存储槽位，却没有同步增加每层可注入的新方向数量。

### 1.2 这是“诊断假说”，不是已证明的容量定理

**[论文报告]** 论文用三类现象支持 information-supply bottleneck：

- mHC 的 $N$-sweep 在 $N>4$ 后迅速饱和；
- temporal augmentation 的收益随 $N$ 墑大；
- dense mHC $N=16$ 加入多尺度 temporal components 后，Pile validation loss 从 1.998 降到 1.984。

**[综合判断]** 这些证据与诊断一致，但没有直接证明额外 streams 已经冗余。更直接的证据应包括：

- stream-to-stream cosine similarity；
- residual state 在 stream 维的 singular-value spectrum；
- write-back matrix 的 effective rank；
- router utilization 和 stream occupancy；
- 不同 streams 对最终 prediction 的 causal contribution。

论文没有报告这些量，因此“信息供给不足”应理解为有实验支持的 mechanism hypothesis，而不是严格定理。

### 1.3 为什么动态 residual mapping 是 $O(N^3C)$

mHC 从 flattened state：

$$
x_l'
\in
\mathbb R^{NC}
$$

生成 $N^2$ 个 residual-mapping logits：

$$
W_l^{\mathrm{res}}
\in
\mathbb R^{NC\times N^2}.
$$

一次 projection 的乘加规模为：

$$
(NC)(N^2)
=
N^3C.
$$

因此这里的 cubic cost 不是计算：

$$
H^{\mathrm{res}}X
$$

本身造成的。该矩阵乘法每 token 约为 $O(N^2C)$；主导项是**从输入状态动态生成 $N^2$ 个系数**。

---

## 2. HC 与 mHC baseline

令某 token 在第 $l$ 个子层入口的状态为：

$$
X_l
=
\begin{bmatrix}
x_{l,1}\\
\vdots\\
x_{l,N}
\end{bmatrix}
\in
\mathbb R^{N\times C}.
$$

HC 的统一更新是：

$$
X_{l+1}
=
\mathcal H_l^{\mathrm{res}}X_l
+
\mathcal H_l^{\mathrm{post}}
\mathcal F_l
\left(
\mathcal H_l^{\mathrm{pre}}X_l,
\mathcal W_l
\right).
$$

其中：

$$
\mathcal H_l^{\mathrm{pre}}
\in
\mathbb R^{1\times N},
\qquad
\mathcal H_l^{\mathrm{post}}
\in
\mathbb R^{N\times1},
\qquad
\mathcal H_l^{\mathrm{res}}
\in
\mathbb R^{N\times N}.
$$

mHC 把 residual mapping 约束到 Birkhoff polytope 附近：

$$
\mathcal H_l^{\mathrm{res}}
=
\operatorname{SK}
\left[
\exp
\left(
\alpha_l^{\mathrm{res}}
(x_l'W_l^{\mathrm{res}})
+
b_l^{\mathrm{res}}
\right)
\right].
$$

理想的 doubly stochastic mapping 满足：

$$
\mathcal H_l^{\mathrm{res}}\mathbf1
=
\mathbf1,
\qquad
\mathbf1^\top \mathcal H_l^{\mathrm{res}}
=
\mathbf1^\top,
\qquad
\mathcal H_{l,ij}^{\mathrm{res}}\ge0.
$$

它恢复的是 common-mode identity、stream mean conservation 和 non-expansive transport，不要求：

$$
\mathcal H_l^{\mathrm{res}}=I.
$$

xHC 保留这个思想，但只在 active subset 上生成 $k\times k$ mapping。

---

## 3. 方法一：Temporal Feature Augmentation

### 3.1 从一个 branch output 构造多个 write-back components

为讨论 token 邻域，恢复 sequence 维：

$$
\mathrm{out}
\in
\mathbb R^{S\times C}.
$$

对它施加 $r$ 个 causal depthwise 1D convolutions：

$$
g_j
=
\operatorname{DWConv}_{\kappa_j}(\mathrm{out}),
\qquad
j=1,\ldots,r.
$$

第 $j$ 个 kernel 的逐 channel 计算是：

$$
g_j[t]
=
\sum_{i=0}^{\kappa_j-1}
w_j[i]
\odot
\mathrm{out}[t-i],
$$

其中：

- $t$ 是 token position；
- $w_j\in\mathbb R^{\kappa_j\times C}$ 是可训练 depthwise kernel；
- $\odot$ 表示逐 channel 乘法；
- 只访问 $t,t-1,\ldots$，因此保持 causal。

把原始输出与卷积分支堆叠：

$$
\mathrm{out}_{\mathrm{aug}}
=
\left[
\mathrm{out};
\operatorname{DWConv}_{\kappa_1}(\mathrm{out});
\ldots;
\operatorname{DWConv}_{\kappa_r}(\mathrm{out})
\right]
\in
\mathbb R^{S\times K_r\times C},
$$

$$
K_r=r+1.
$$

默认配置为：

$$
r=3,
\qquad
\{\kappa_1,\kappa_2,\kappa_3\}
=
\{4,8,12\},
\qquad
K_r=4.
$$

新增 convolution parameters 为：

$$
C
\sum_{j=1}^{r}\kappa_j
=
24C
$$

每个 MLP sublayer。

### 3.2 它增加的是低成本 write-back basis，不是四次 MLP

如果直接为每条 stream 单独计算一个 MLP 输出，主干成本会随 $N$ 成倍增长。xHC 只计算一次：

$$
\mathrm{out}
=
\operatorname{MLP}(\mathrm{input}),
$$

再用轻量 depthwise convolution 构造局部时序变体。因此 $K_r=4$ 不表示 MLP FLOPs 变成四倍。

**[综合判断]** 这些 components 也不能自动视为四个完全独立的 semantic features：

- 它们都线性来源于同一条 $\mathrm{out}$ sequence；
- depthwise convolution 不跨 channel 混合；
- kernel receptive fields 重叠；
- 最终有效维数取决于训练后的 kernel 和数据。

### 3.3 Modified Gram–Schmidt

卷积分支可能与原始输出高度共线。论文令：

$$
v_1=\mathrm{out},
\qquad
g_j=\operatorname{DWConv}_{\kappa_j}(\mathrm{out}),
$$

并对 $j=1,\ldots,r$ 计算：

$$
v_{j+1}
=
g_j
-
\sum_{i=1}^{j}
\frac{
\langle g_j,v_i\rangle
}{
\langle v_i,v_i\rangle
}
v_i.
$$

最后重定义：

$$
\mathrm{out}_{\mathrm{aug}}
=
[v_1;\ldots;v_{K_r}].
$$

这里 inner product 对每个 token 的 $C$ 维 channel vector 计算，所以理想情况下：

$$
\langle v_a,v_b\rangle=0
\qquad
(a\ne b).
$$

### 3.4 Gram–Schmidt 能保证什么、不能保证什么

它能在精确算术且 denominator 非零时去除前序 components 的平行投影，但不能保证：

- 每个 $v_i$ 的 norm 为 1；
- 各分支尺度一致；
- 跨 token、跨 batch 或跨 layer 正交；
- $K_r$ 个 components 都有足够大且有语义的信息量。

论文公式的 denominator 没有显式 $\epsilon$：

$$
\langle v_i,v_i\rangle.
$$

**[待验证]** 如果某个 $v_i$ 接近零，实际 kernel 如何避免除零或极大 projection coefficient，没有在 paper 或公开代码中说明。

### 3.5 为什么只放在 MLP 后

**[论文报告]**

- Attention 已经做 token mixing；
- MLP 对每个 token 独立处理，适合补充低成本 local temporal context；
- 10B ablation 中，默认 MLP-only 为 1.983，额外在 Attention 后也加 augmentation 为 1.985。

正文早先用“destabilizes training”描述 Attention-side augmentation；附录给出的直接证据却只是 validation loss 差 0.002，并强调 extra computation/complexity。

**[综合判断]** 因此更稳妥的结论是：在报告的 10B 设置里，Attention-side augmentation 没有额外收益；不能只根据 Table 11 断言它普遍导致训练发散。

---

## 4. 方法二：Sparse Residual-Stream Architecture

### 4.1 Routing：fixed paths 加动态 Top-$k$

先把某 token 的完整 state 展平并做 LayerNorm：

$$
\widetilde x_l
=
\operatorname{LN}
\left(
\operatorname{flatten}(X_l)
\right)
\in
\mathbb R^{NC}.
$$

router 生成全部 $N$ 条 streams 的 score：

$$
s
=
\sigma
\left(
\widetilde x_lW^r
\right)
\in
\mathbb R^N,
\qquad
W^r\in\mathbb R^{NC\times N}.
$$

默认 $m=2$ 条 fixed streams 始终 active，剩余 $k-m=2$ 条从 non-fixed streams 中选出：

$$
\mathcal I
=
\operatorname{FixedIdx}
\cup
\operatorname{TopK}_{i>m}
(s_i,k-m).
$$

对 active position $j=1,\ldots,k$：

$$
p_j
=
\begin{cases}
1,
&
\mathcal I_j\in\operatorname{FixedIdx},
\\
s_{\mathcal I_j},
&
\mathcal I_j\notin\operatorname{FixedIdx}.
\end{cases}
$$

需要注意：

- router 虽然忽略 fixed-stream scores，仍投影出全部 $N$ 个 scores，以便与 pre-mapping projection 融合；
- sigmoid scores 不沿 stream 维归一化，不要求和为 1；
- hard Top-$k$ indices 不可微；
- routed scores $p_j$ 只 gate 新 write-back，不 gate residual mixing。

### 4.2 Dense Read：所有 streams 都能影响当前 branch

pre-mapping 由完整 state 动态生成：

$$
\mathcal H_l^{\mathrm{pre}}
=
f_{\mathrm{pre}}(X_l)
\in
\mathbb R^{1\times N}.
$$

当前子层输入是：

$$
\mathrm{input}_l
=
\sum_{i=1}^{N}
h_{l,i}^{\mathrm{pre}}
x_{l,i}
\in
\mathbb R^C.
$$

即使 stream $i$ 当前没有被选中写入：

$$
i\notin\mathcal I,
$$

它仍可通过 $h_{l,i}^{\mathrm{pre}}x_{l,i}$ 影响当前 Attention/MLP。

### 4.3 Sparse residual mixing 与 write-back

按 indices gather：

$$
X_{\mathrm{active}}
=
\operatorname{Gather}(X_l,\mathcal I)
\in
\mathbb R^{k\times C}.
$$

只从 active state 生成：

$$
\mathcal H_l^{\mathrm{res}}
\in
\mathbb R^{k\times k},
\qquad
\mathcal H_l^{\mathrm{post}}
\in
\mathbb R^{k\times K_r}.
$$

对第 $j$ 条 active stream：

$$
\Delta X_{\mathrm{active},j}
=
p_j
\sum_{q=1}^{K_r}
\mathcal H_{l,j,q}^{\mathrm{post}}
\mathrm{out}_{\mathrm{aug},q}.
$$

然后：

$$
X_{\mathrm{active}}^{\mathrm{new}}
=
\mathcal H_l^{\mathrm{res}}
X_{\mathrm{active}}
+
\Delta X_{\mathrm{active}}.
$$

最后 scatter：

$$
X_{l+1,\mathcal I}
=
X_{\mathrm{active}}^{\mathrm{new}},
$$

$$
X_{l+1,i}
=
X_{l,i}
\qquad
(i\notin\mathcal I).
$$

### 4.4 一张 connection-method contract 表

| 问题 | xHC 的答案 |
|---|---|
| Stored state | $X_l\in[B,S,N,C]$ 的 $N$ 条 persistent residual streams |
| Read rule | 每 token、每子层通过 $\mathcal H^{pre}$ dense read 全部 $N$ 条 streams |
| Branch | 标准 $C\to C$ Attention 或 MLP/MoE |
| Write rule | 只向 $k$ 条 active streams 写入；MLP 使用 $K_r=4$ 个 components，Attention 使用 $K_r=1$ |
| Residual-state update | active subset 做 $\mathcal H^{res}X_{\mathrm{active}}$；inactive streams 是 exact carry |
| Weight granularity | mapping coefficients 和 routing 是 per token、per sublayer 的标量；同一 coefficient 作用于全部 $C$ channels |
| Fixed parameters | $N,k,m,r,\kappa_j$ 和 fixed stream indices |
| Trainable parameters | 主干、router、pre/res/post generators、gates、biases、temporal kernels、可能的 norm affine weights |
| 初始化 | mapping gate $\alpha=0.01$；论文没有完整给出 static biases 和 multi-stream state initialization |
| 深度组合 | 理想 active mixing 可嵌入为全局 doubly stochastic block；实际使用有限步 Sinkhorn 和 row-sum clamp |
| 最终 collapse | 全部 $N$ 条 streams 求和，再接 final RMSNorm 和 unembedding |
| 主要资源成本 | full-state read/router 为 $N$ 相关；active mapping 为 $k$ 相关；persistent state 与 PP payload 为 $NC$ |

### 4.5 默认 shape trace

取：

$$
B=2,\quad S=8192,\quad N=16,\quad k=4,\quad C=2112.
$$

| 操作 | 输入 shape | 输出 shape |
|---|---:|---:|
| full residual state | — | $[2,8192,16,2112]$ |
| flatten stream/channel | $[2,8192,16,2112]$ | $[2,8192,33792]$ |
| router projection | $[2,8192,33792]$ | $[2,8192,16]$ |
| fixed + Top-2 route | scores | indices/weights $[2,8192,4]$ |
| active gather | full state + indices | $[2,8192,4,2112]$ |
| pre mapping | full state | $[2,8192,16]$ |
| dense read | state + pre mapping | $[2,8192,2112]$ |
| residual mapping | active state | $[2,8192,4,4]$ |
| Attention post mapping | active state | $[2,8192,4,1]$ |
| MLP post mapping | active state | $[2,8192,4,4]$ |
| MLP temporal augmentation | MLP output | $[2,8192,4,2112]$ |
| active update | mappings + outputs | $[2,8192,4,2112]$ |
| sparse scatter | full + active update | $[2,8192,16,2112]$ |
| final stream sum | $[2,8192,16,2112]$ | $[2,8192,2112]$ |

这个 trace 也说明 active indices 理论上可以随 batch item 和 token position 改变，而不是全 batch 共用一组 4 条 streams。

---

## 5. 具体参数化：到底哪些是可训练参数

### 5.1 Dense pre-mapping

$$
\mathcal H_l^{\mathrm{pre}}
=
\sigma
\left[
\alpha^{\mathrm{pre}}
\operatorname{RMSNorm}
\left(
\operatorname{flatten}(X_l)
\right)
W^{\mathrm{pre}}
+
b^{\mathrm{pre}}
\right],
$$

$$
W^{\mathrm{pre}}
\in
\mathbb R^{NC\times N}.
$$

因此：

$$
0
<
\mathcal H_{l,i}^{\mathrm{pre}}
<
1,
$$

但不要求：

$$
\sum_i
\mathcal H_{l,i}^{\mathrm{pre}}
=1.
$$

它是 positive weighted sum，不是 probability-simplex convex combination。

### 5.2 Active residual mapping

令：

$$
\widetilde x_{\mathrm{active}}
=
\operatorname{RMSNorm}
\left(
\operatorname{flatten}
(X_{\mathrm{active}})
\right)
\in
\mathbb R^{kC}.
$$

则：

$$
\mathcal H_l^{\mathrm{res}}
=
\operatorname{SK}
\left\{
\exp
\left[
\alpha^{\mathrm{res}}
\operatorname{mat}_{k\times k}
\left(
\widetilde x_{\mathrm{active}}
W^{\mathrm{res}}
\right)
+
b^{\mathrm{res}}
\right]
\right\},
$$

$$
W^{\mathrm{res}}
\in
\mathbb R^{kC\times k^2}.
$$

### 5.3 Active post-mapping

$$
\mathcal H_l^{\mathrm{post}}
=
2\sigma
\left[
\alpha^{\mathrm{post}}
\operatorname{mat}_{k\times K_r}
\left(
\widetilde x_{\mathrm{active}}
W^{\mathrm{post}}
\right)
+
b^{\mathrm{post}}
\right],
$$

$$
W^{\mathrm{post}}
\in
\mathbb R^{kC\times kK_r}.
$$

所以：

$$
0
<
\mathcal H_{l,j,q}^{\mathrm{post}}
<
2.
$$

论文把这个范围解释为允许 attenuation 和 mild amplification。需要注意，系数为正，因此单个 post coefficient 不能直接反转某个 component 的符号；但 component 本身和不同 components 的线性方向仍可含正负元素。

### 5.4 完整的 optimizer-owned 参数

每个 xHC sublayer 的主要新增参数是：

1. router：

$$
W^r\in\mathbb R^{NC\times N};
$$

2. full-state pre generator：

$$
W^{\mathrm{pre}}\in\mathbb R^{NC\times N};
$$

3. active-state residual generator：

$$
W^{\mathrm{res}}\in\mathbb R^{kC\times k^2};
$$

4. active-state post generator：

$$
W^{\mathrm{post}}\in\mathbb R^{kC\times kK_r};
$$

5. 三类 mapping 的：

$$
\alpha^{\mathrm{pre}},
\alpha^{\mathrm{res}},
\alpha^{\mathrm{post}},
\quad
b^{\mathrm{pre}},
b^{\mathrm{res}},
b^{\mathrm{post}};
$$

6. MLP side 的 temporal kernels：

$$
w_j\in\mathbb R^{\kappa_j\times C};
$$

7. normalization 若启用 affine weight，则相应 affine 参数；
8. 原有 Attention、MLP/MoE、embedding、unembedding、norm 和 MoE router 参数。

### 5.5 是不是跟着主 loss 正常训练

是。**[复原推导]** 默认 AdamW 训练中的梯度链是：

$$
\mathcal L_{\mathrm{LM}}
\longrightarrow
X_{l+1}
\longrightarrow
\left\{
\mathcal H^{\mathrm{pre}},
\mathcal H^{\mathrm{res}},
\mathcal H^{\mathrm{post}},
p,
\mathrm{out}_{\mathrm{aug}}
\right\}
$$

$$
\longrightarrow
\left\{
W^r,
W^{\mathrm{pre}},
W^{\mathrm{res}},
W^{\mathrm{post}},
\alpha^{(\cdot)},
b^{(\cdot)},
w_j,
\mathcal W_l
\right\}.
$$

论文没有报告：

- router load-balancing loss；
- stream-utilization auxiliary loss；
- doubly-stochastic violation penalty；
- temporal-component diversity loss。

Sinkhorn 和 Gram–Schmidt 是 forward graph 中的重参数化/算子，不是额外 loss。

### 5.6 Hard Top-$k$ 的梯度边界

Top-$k$ 选择 indices：

$$
\mathcal I
=
\operatorname{TopK}(s)
$$

是离散操作。论文的 fused backward 明确只把 routing gradient 传给两条被动态选中的 logits。

因此：

- selected routed scores 可通过 $p_j$ 收到梯度；
- unselected scores 在这次 forward 中没有 routing gradient；
- fixed streams 的 score 被忽略；
- indices 变化本身没有普通导数；
- score 接近选取边界时，函数对参数是 piecewise smooth。

论文没有使用 straight-through estimator、Gumbel-Softmax 或 differentiable sorting。

### 5.7 Muon 设置下的 optimizer 分组

**[论文报告]**

- Muon 用于 backbone 的 2D matrices：Attention、MLP/MoE 和 MoE router projections；
- AdamW 用于 embedding、normalization 和全部 xHC-specific parameters；
- Muon momentum $\beta=0.95$；
- 5 次 Newton–Schulz iterations；
- matched-AdamW-RMS target 为 0.2；
- Muon+xHC 实验移除 Gram–Schmidt。

论文的理由是 xHC matrices 极不方正，例如：

$$
NC\to N,
\qquad
kC\to k^2,
\qquad
kC\to kK_r,
$$

不适合直接使用相同的 matrix orthogonalization recipe。

---

## 6. 初始化、约束与深层组合

### 6.1 论文明确给出的初始化

三个 mapping generator 的 input-dependent gate 初始化为：

$$
\alpha^{(\cdot)}=0.01.
$$

这让运行时 mapping 初期主要受 static bias 控制，但它不是：

$$
\alpha^{(\cdot)}=0.
$$

所以输入相关分支从第一步起就不是严格关闭。

### 6.2 论文没有完整给出的初始化

论文没有明确列出：

- $b^{\mathrm{pre}}$、$b^{\mathrm{res}}$、$b^{\mathrm{post}}$ 的初始数值；
- $W^r$ 和各 mapping projection 的 initializer；
- $N$ 条初始 residual streams 如何从 embedding 构造；
- temporal convolution kernel 的 initializer；
- 如何在 default $N=16,k=4$ 下恢复普通 PreNorm 的完整函数；
- branch output projection 是否做与 HC 相同的 $\sqrt N$ 尺度补偿。

它只说未特别说明的实现细节 follow mHC。由于官方代码尚未公开，这些项目不能从 artifact 中验证。

**[综合判断]** 因此不能仅凭 $\alpha=0.01$ 声称 xHC 在 initialization 时与 vanilla Transformer 精确等价。

### 6.3 active-set mixing 能否继承 mHC 的约束

若 $\mathcal H^{\mathrm{res}}\in\mathbb R^{k\times k}$ 是精确 doubly stochastic，把它嵌入完整 $N$-stream update，而 inactive streams 使用 identity，则得到一个置换后 block-diagonal matrix：

$$
\widehat H
=
P^\top
\begin{bmatrix}
\mathcal H^{\mathrm{res}} & 0\\
0 & I_{N-k}
\end{bmatrix}
P.
$$

这里 $P$ 只是把 active streams 排到前面的 permutation matrix。

**[复原推导]** 若 active mapping 精确双随机，则：

$$
\widehat H\mathbf1_N
=
\mathbf1_N,
\qquad
\mathbf1_N^\top\widehat H
=
\mathbf1_N^\top.
$$

即使 active set 随层改变，每层的 $\widehat H_l$ 仍是全局 doubly stochastic，乘积仍留在 Birkhoff polytope：

$$
\widehat H_{L:1}
=
\widehat H_L\cdots\widehat H_1.
$$

这说明 sparse active-set mixing 与 mHC 的 common-mode stability 原则在理想精确约束下兼容。

### 6.4 实际 row-sum clamp 改变了什么

论文报告：极端 activation 有时会让有限步 Sinkhorn 后某些 row sum 大于 1，因此对每行执行：

$$
\mathcal H_{i:}^{\mathrm{res}}
\leftarrow
\frac{
\mathcal H_{i:}^{\mathrm{res}}
}{
\max
\left(
\sum_j\mathcal H_{ij}^{\mathrm{res}},
1
\right)
}.
$$

这保证：

$$
\sum_j
\mathcal H_{ij}^{\mathrm{res}}
\le1.
$$

但如果只缩放部分 rows，通常不再保证：

$$
\sum_i
\mathcal H_{ij}^{\mathrm{res}}
=1.
$$

**[复原推导]** row-sum clamp 可能把近似 doubly stochastic matrix 变成 row-substochastic matrix。因此：

- nonnegativity 仍在；
- row-wise forward amplification 得到限制；
- exact common-mode identity 和 stream-mean conservation 可能被破坏；
- 仅有 row sum $\le1$ 也不足以单独推出 spectral norm $\le1$。

这是一个实用稳定化补丁，不应与精确 Birkhoff projection 混为一谈。论文没有报告 clamp 的触发频率、偏离 column sum 的幅度或对最终 transport spectrum 的影响。

---

## 7. 参数量与 FLOPs：为什么 $N=16$ 仍可负担

### 7.1 每个 Transformer layer 的参数公式

一个 Transformer layer 含 Attention 和 MLP 两个 sublayers。论文统计 dominant projection 和 temporal-conv weights：

$$
P_{\mathrm{xHC}}
=
\left(
4N^2
+
2k^3
+
k^2
+
k^2K_r
+
\sum_{i=1}^{r}\kappa_i
\right)C.
$$

逐项来源是：

| 项 | 来源 |
|---:|---|
| $4N^2C$ | 两个 sublayers 各有 router $N^2C$ 和 pre generator $N^2C$ |
| $2k^3C$ | Attention、MLP 各一个 $kC\to k^2$ residual generator |
| $k^2C$ | Attention 的 $kC\to k$ post generator |
| $k^2K_rC$ | MLP 的 $kC\to kK_r$ post generator |
| $C\sum_i\kappa_i$ | MLP temporal depthwise kernels |

对 dense mHC：

$$
P_{\mathrm{mHC}}
=
\left(
4N^2+2N^3
\right)C.
$$

默认参数代入：

$$
N=16,
\qquad
k=4,
\qquad
K_r=4,
\qquad
\sum_i\kappa_i=24,
$$

得到：

$$
P_{\mathrm{xHC}}
=
1256C,
$$

$$
P_{\mathrm{mHC},N=16}
=
9216C.
$$

xHC 的 per-layer 新增参数约少：

$$
\frac{9216}{1256}
\approx
7.34\times.
$$

### 7.2 论文参数公式的边界

**[综合判断]** 上述公式统计主要 matrix 和 conv weights，没有显式计入：

- bias；
- $\alpha$ scalars；
- normalization affine parameters；
- 可能的 padding/cache metadata。

这些项相对 projection 很小，但这仍是 dominant-term parameter accounting，不是逐 parameter object 的精确 state-dict count。

### 7.3 Training FLOPs 公式

论文用：

$$
F_{\mathrm{HC}}
=
6P_{\mathrm{HC}}L
$$

近似 parametric forward+backward FLOPs。factor 6 来自训练线性层常用近似。

它另外说明以下 non-parametric operations 不计入这个公式：

- Sinkhorn；
- residual mixing；
- multi-stream LayerNorm/RMSNorm；
- Gram–Schmidt。

这些操作主要在 memory-access 和 infrastructure 部分讨论。因此“xHC 只增加 3.0% FLOPs”不是对每个算术操作逐项计数后的完整 FLOP audit。

### 7.4 18B / 28B overhead

| 方法 | 18B 每层参数 | 18B 参数 overhead | 18B train FLOPs overhead | 28B 每层参数 | 28B 参数 overhead | 28B train FLOPs overhead |
|---|---:|---:|---:|---:|---:|---:|
| mHC $N=4$ | 405K | 0.5% | 0.7% | 492K | 0.6% | 0.5% |
| mHC $N=16$ | 19.5M | 26.3% | 18.9% | 23.6M | 30.2% | 22.3% |
| xHC $N=16,k=4$ | 2.65M | 3.5% | 4.1% | 3.22M | 4.1% | 3.0% |

这里参数 overhead 以 vanilla backbone 的 **activated parameters** 为 denominator，不是 total MoE parameters。

---

## 8. xHC-Flash：什么是精确复用，什么是架构近似

### 8.1 full xHC 的主要 I/O 问题

默认 full xHC 每个 sublayer 至少有两次 full-state pass：

1. 生成 router 和 pre mapping；
2. 用 pre mapping dense read 全部 $N$ 条 streams。

即使 active update 只有 $k=4$，每次仍访问：

$$
NC=16C
$$

大小的 persistent state。

论文按每 token、每 sublayer 的 residual-stream operations 估算：

| 方法 | Reads | Writes | Total I/O |
|---|---:|---:|---:|
| mHC $N=4$ | $21C$ | $13C$ | $34C$ |
| mHC $N=16$ | — | — | $130C$ |
| full xHC $N=16,k=4$ | $55C$ | $18.5C$ | $73.5C$ |
| xHC-Flash | $36C$ | $15C$ | $51C$ |
| xHC-Flash-4sub | $26.5C$ | $13.5C$ | $40C$ |

这些数：

- 不包含 Attention/MLP branch 自身 I/O；
- 以抽象 element count $C$ 计量，不是实际 HBM bytes；
- 是按 Attention/MLP asymmetry amortize 的估算；
- 不能直接等同于 wall-clock speedup。

### 8.2 一个 block 内共享 routing 和 precomputation

xHC-Flash 在 block-entry state $X^{(0)}$ 上一次生成：

$$
s,
\qquad
\mathcal H^{\mathrm{pre,Attn}},
\qquad
\mathcal H^{\mathrm{pre,MLP}}.
$$

两个 pre mappings 使用独立权重，但共享 full-state projection/read schedule。预计算：

$$
\mathrm{inp}_{\mathrm A}
=
\sum_{i=1}^{N}
\mathcal H_i^{\mathrm{pre,Attn}}
x_i^{(0)},
$$

$$
\mathrm{inp}_{\mathrm M}
=
\sum_{i=1}^{N}
\mathcal H_i^{\mathrm{pre,MLP}}
x_i^{(0)}.
$$

### 8.3 为什么 Attention side 去掉 residual mixing

Attention active update 改为：

$$
X_{\mathrm{active}}^{(\mathrm A)}
=
X_{\mathrm{active}}^{(0)}
+
p\odot
\left(
\mathcal H^{\mathrm{post,Attn}}
\mathrm{out}_{\mathrm{Attn}}
\right).
$$

即：

$$
\mathcal H^{\mathrm{res,Attn}}
=
I_k.
$$

这让每条 active stream 的变化都是同一个 Attention output 的标量倍数，后续 MLP dense read 可以低成本修正。

### 8.4 MLP input correction 的完整推导

active stream $\mathcal I_j$ 经 Attention 后：

$$
x_{\mathcal I_j}^{(\mathrm A)}
=
x_{\mathcal I_j}^{(0)}
+
p_j
\mathcal H_j^{\mathrm{post,Attn}}
\mathrm{out}_{\mathrm{Attn}}.
$$

inactive streams 不变。把固定在 block entry 生成的 MLP pre mapping 应用到更新后状态：

$$
\mathrm{input}_{\mathrm{MLP}}
=
\sum_{i\notin\mathcal I}
\mathcal H_i^{\mathrm{pre,MLP}}
x_i^{(0)}
+
\sum_{j=1}^{k}
\mathcal H_{\mathcal I_j}^{\mathrm{pre,MLP}}
x_{\mathcal I_j}^{(\mathrm A)}.
$$

代入 active update：

$$
\begin{aligned}
\mathrm{input}_{\mathrm{MLP}}
&=
\sum_{i=1}^{N}
\mathcal H_i^{\mathrm{pre,MLP}}
x_i^{(0)}
\\
&\quad+
\sum_{j=1}^{k}
\mathcal H_{\mathcal I_j}^{\mathrm{pre,MLP}}
p_j
\mathcal H_j^{\mathrm{post,Attn}}
\mathrm{out}_{\mathrm{Attn}}.
\end{aligned}
$$

第一项正是预计算的 $\mathrm{inp}_{\mathrm M}$。把第二项的标量系数记为：

$$
\alpha
=
\sum_{j=1}^{k}
\mathcal H_{\mathcal I_j}^{\mathrm{pre,MLP}}
p_j
\mathcal H_j^{\mathrm{post,Attn}},
$$

得到：

$$
\boxed{
\mathrm{input}_{\mathrm{MLP}}
=
\mathrm{inp}_{\mathrm M}
+
\alpha\,
\mathrm{out}_{\mathrm{Attn}}
}
$$

这就是 dense-read reuse。它只需一个 token-wise scalar-vector multiply 和 addition，不必重新读取 $NC$ state。

### 8.5 “精确”的适用条件

论文 Appendix E 明确要求：

1. routing decision 在 sharing window 内固定；
2. sublayer-specific pre mappings 从 window-entry state 生成后固定；
3. intermediate sublayer 前不做 residual mixing；
4. write-back 只改变 active streams。

在这些条件下，correction 精确恢复的是：

> 用这组固定 pre mapping 对已经发生 sparse write-back 的 state 做 dense read 所得到的输入。

### 8.6 相对 full xHC 的近似

xHC-Flash 相对 full xHC 近似的是 dynamic control schedule：

- full xHC 每个 sublayer 重新 routing；Flash 在一个或两个 blocks 内共享；
- full xHC 的 pre mapping 依赖每个 sublayer 最新 state；Flash 依赖 sharing-window entry state；
- full xHC 每个 sublayer 都可 residual-mix active streams；Flash 把 mixing 推迟到 MLP 或最后一个 MLP。

所以：

$$
\text{exact correction}
\ne
\text{exact equivalence to full xHC}.
$$

### 8.7 四子层版本

xHC-Flash-4sub 在两个 Transformer blocks、四个 sublayers 内：

- 共享一组 routing indices 和 weights；
- 联合生成四组独立 pre mappings；
- intermediate sublayers 只 sparse write；
- 只在最后一个 MLP 做 residual mixing；
- 最后只 scatter 一次。

MLP write-back 有 $K_r=4$ 个 components，不能再压成单个：

$$
\alpha\,\mathrm{out}.
$$

因此论文改用：

$$
\mathrm{input}^{(t)}
=
\underbrace{
\sum_{i\notin\mathcal I}
\mathcal H_i^{\mathrm{pre},t}
x_i^{(0)}
}_{\text{non-active base}}
+
\underbrace{
\sum_{j=1}^{k}
\mathcal H_{\mathcal I_j}^{\mathrm{pre},t}
x_{\mathcal I_j}^{(t)}
}_{\text{current active contribution}}.
$$

active state 本来就在四个 sublayers 之间携带，因此不必额外保存 accumulated-delta buffer。

### 8.8 Flash 质量–I/O trade-off

10B MoE、Pile test：

| 方法 | Validation loss | Estimated I/O / sublayer |
|---|---:|---:|
| Vanilla | 2.029 | $3C$ |
| mHC $N=4$ | 2.004 | $34C$ |
| full xHC | **1.983** | $73.5C$ |
| xHC-Flash | **1.983** | $51C$ |
| xHC-Flash-4sub | 1.984 | $40C$ |

这表明短窗口共享在这个 10B setting 中几乎不损失 validation loss。它不能证明：

- 18B/28B 下游结果也完全保持；
- 所有 sequence lengths、tasks 和 model depths 下都保持；
- sharing window 可以继续无损增大。

主结果 Table 1 使用 full xHC；论文没有给出 Flash 版本的 18B/28B 全套 downstream table。

---

## 9. Infrastructure contract

### 9.1 Mixed precision

**[论文报告]**

| 对象 | dtype |
|---|---|
| persistent residual state | BF16 |
| projection operands | BF16 |
| normalization statistics | FP32 |
| routing scores/coefficients | FP32 |
| mapping coefficients | FP32 |
| Sinkhorn iterations | FP32 |

这意味着 reference 实现和 fused 实现至少要约定：

- BF16 state 何时升到 FP32；
- mapping application 的 accumulation dtype；
- 输出何时 cast 回 BF16；
- recomputation 是否逐 bit 或只在 tolerance 内一致。

### 9.2 Router 与 pre projection fusion

router 使用 LayerNorm，pre mapping 使用 RMSNorm。令 flattened full state 为：

$$
x\in\mathbb R^{NC},
$$

feature mean 和 second moment 为：

$$
\mu
=
\frac1{NC}
\sum_a x_a,
\qquad
\nu
=
\frac1{NC}
\sum_a x_a^2.
$$

论文不显式 materialize normalized state，而先计算 raw projection，再修正：

$$
Z^r
=
\frac{
xW^r
-
\mu\,\mathbf1^\top W^r
}{
\sqrt{\nu-\mu^2+\epsilon}
},
$$

$$
Z^{\mathrm{pre}}
=
\frac{
xW^{\mathrm{pre}}
}{
\sqrt{\nu+\epsilon}
}.
$$

并把：

$$
[W^r,W^{\mathrm{pre}}]
$$

沿 output dimension 拼接，用一次 matmul 生成 raw projections。

### 9.3 Active mapping fusion

gather active state 时同时计算 flattened RMS statistic，然后一次 projection 联合生成：

$$
[Z^{\mathrm{post}},Z^{\mathrm{res}}]
=
\operatorname{RMSNorm}
\left(
\operatorname{vec}
(X_{\mathrm{active}})
\right)
[W^{\mathrm{post}},W^{\mathrm{res}}].
$$

默认 $k=4$，所以 Sinkhorn 只处理每 token 一个 $4\times4$ matrix。

论文的 backward 不保存所有 Sinkhorn intermediate states，而在 backward recompute。

### 9.4 Mapping application fusion

子层前联合计算：

$$
\mathrm{input}
=
\sum_{i=1}^{N}
\mathcal H_i^{\mathrm{pre}}x_i,
\qquad
X_{\mathrm{mixed}}
=
\mathcal H^{\mathrm{res}}X_{\mathrm{active}}.
$$

子层后联合计算：

$$
X_{\mathcal I}^{\mathrm{new}}
=
X_{\mathrm{mixed}}
+
p\odot
\left(
\mathcal H^{\mathrm{post}}
\mathrm{out}_{\mathrm{aug}}
\right),
$$

并直接 scatter，避免 materialize $kC$ write-back buffer。

### 9.5 Temporal-conv fusion

三个 kernels $\{4,8,12\}$ 在一个 specialized kernel 中计算，并让 post-mapping 直接消费：

- 原始 MLP output；
- 三个 convolution outputs。

不显式创建：

$$
[S,B,4,C]
$$

concatenated tensor。

### 9.6 公开 artifact 与实现成熟度

截至 2026-07-24，官方仓库 commit `7890266...` 只有：

- `README.md`；
- `assets/`；
- `paper/xHC_tech_report.pdf`。

不存在可检查的：

- reference `nn.Module`；
- custom autograd；
- Triton forward/backward kernels；
- optimizer parameter groups；
- distributed parallel implementation；
- numerical tests；
- benchmark scripts。

因此本节只能复原 paper-level contract，不能完成真实代码调用链 walkthrough。

---

## 10. 实验设置

### 10.1 Backbone

所有主要模型使用 DeepSeekMoE-style architecture：

- 1 个 leading dense layer；
- 之后为 MoE layers；
- 144 routed experts；
- 1 shared expert；
- sigmoid Top-8 expert routing；
- GQA；
- QK Norm；
- SwiGLU；
- context length 8192；
- Qwen2 tokenizer；
- vocabulary size 152,064；
- RoPE $\theta=50{,}000$；
- RMSNorm $\epsilon=10^{-5}$。

### 10.2 模型规模

| Setting | Total params | Activated params | Layers | Hidden $C$ | 用途 |
|---|---:|---:|---:|---:|---|
| 2.5B | 2.5B | 0.5B | 15 | 1024 | $N$-sweep |
| 10B | 10B | 1.4B | 15 | 2080 | ablation |
| 18B | 18B | 1.7B | 28 | 2112 | main results / Muon |
| 28B | 28B | 2.7B | 32 | 2560 | main results |

### 10.3 默认 xHC

$$
N=16,
\quad
k=4,
\quad
m=2,
\quad
r=3,
\quad
\kappa=\{4,8,12\},
\quad
K_r=4.
$$

另外：

- MLP-only temporal augmentation；
- 20 次 Sinkhorn；
- mapping gate $\alpha=0.01$；
- fixed-2 + routed-Top-2；
- sigmoid stream router。

### 10.4 AdamW recipe

$$
\beta_1=0.9,
\qquad
\beta_2=0.95,
\qquad
\epsilon=10^{-15},
$$

$$
\text{weight decay}=0.1,
\qquad
\text{gradient clipping}=1.0.
$$

学习率采用 warmup–stable–decay：

- 500 warmup steps；
- exponential decay；
- minimum/peak LR ratio 0.1。

### 10.5 Final stream collapse

进入 final RMSNorm 和 unembedding 前：

$$
h_{\mathrm{final}}
=
\sum_{i=1}^{N}
x_{L,i}
\in
\mathbb R^C.
$$

这一步不是 average。若比较不同 $N$ 的 activation scale，必须把该 sum 和前面 mapping/branch initialization 一起审计。

### 10.6 Evaluation protocol

Perplexity-based multiple choice：

- MMLU 5-shot；
- MMLU-Redux 5-shot；
- CMMLU 5-shot；
- CEval 5-shot；
- ARC-Challenge 25-shot；
- C3 3-shot。

Generation-based：

- MMLU-Pro 5-shot；
- BBH 3-shot；
- CommonsenseQA 7-shot；
- GSM8K 4-shot；
- HumanEval 0-shot pass@1；
- LCBench 5-shot。

使用 modified OpenCompass，但论文没有提供 fork、commit、配置或 parsing code。

---

## 11. 主结果：18B 与 28B

### 11.1 完整 downstream table

所有值为百分数，越高越好。

| Benchmark | 18B Vanilla | 18B mHC | 18B xHC | 28B Vanilla | 28B mHC | 28B xHC |
|---|---:|---:|---:|---:|---:|---:|
| MMLU | 48.9 | 54.7 | **57.2** | 54.6 | 56.8 | **60.5** |
| MMLU-Pro | 21.1 | 27.4 | **29.7** | 30.1 | 34.9 | **36.0** |
| MMLU-Redux | 46.4 | 49.9 | **52.8** | 50.6 | 53.9 | **56.4** |
| BBH | 32.4 | 33.7 | **39.5** | 41.7 | **43.6** | 43.4 |
| CommonsenseQA | 54.6 | 56.6 | **60.9** | 60.5 | 63.9 | **69.6** |
| ARC-Challenge | 55.7 | 66.3 | **72.2** | 70.8 | 74.9 | **77.7** |
| GSM8K | 37.7 | 44.5 | **48.4** | 50.3 | 56.3 | **59.2** |
| HumanEval | 25.6 | 23.2 | **29.3** | 27.4 | 26.8 | **31.1** |
| LCBench | 9.9 | 12.2 | **14.6** | 15.1 | 14.8 | **17.9** |
| CMMLU | 42.7 | 47.6 | **50.4** | 47.6 | 50.1 | **53.4** |
| CEval | 44.5 | 48.8 | **52.4** | 50.2 | 51.2 | **54.9** |
| C3 | 67.1 | 72.7 | **78.3** | 75.2 | 78.7 | **82.5** |
| **Average** | 40.6 | 44.8 | **48.8** | 47.8 | 50.5 | **53.6** |

### 11.2 支持的结论

- xHC 在两个规模的 average 都高于 mHC 和 vanilla。
- 18B 的 12 项 benchmark 中，xHC 全部高于两者。
- 28B 时，xHC 高于 vanilla 的 12 项；相对 mHC 为 11 胜 1 负。
- 唯一相对 mHC 略低的是 28B BBH：

$$
43.4
\quad\text{vs.}\quad
43.6.
$$

因此不能写成“xHC 在所有规模、所有 benchmark 都严格优于 mHC”。

### 11.3 还不能推出什么

- benchmark average 是不同任务百分数的未加权平均，不是统一统计量；
- 无多 seed 时，0.2–1.0 point 的差异可能处于 run variance 范围；
- 训练数据不透明时，不能审计 benchmark contamination；
- mHC $N=4$ 与 xHC $N=16,k=4$ 比较的是完整 architecture package，不是只改变一个变量。

---

## 12. Scaling law 证据

### 12.1 拟合形式

每种方法训练 4 个模型，compute 范围约：

$$
1.7\times10^{19}
\quad\text{到}\quad
4.0\times10^{20}
\ \text{FLOPs}.
$$

拟合：

$$
\mathcal L(C_{\mathrm{train}})
=
A
C_{\mathrm{train}}^{-\alpha}
+
E,
$$

其中固定：

$$
E=0.72.
$$

报告的参数为：

| 方法 | $A$ | $\alpha$ |
|---|---:|---:|
| Vanilla | 109.303 | 0.0936 |
| mHC $N=4$ | 99.139 | 0.0920 |
| xHC $N=16,k=4$ | 97.703 | 0.0919 |

论文在：

$$
\log_2(\mathcal L-E)
$$

与：

$$
\log_{10}C_{\mathrm{train}}
$$

之间做 linear regression，再转换回 power law。

### 12.2 论文报告的 matched-loss 结果

在最大 compute point：

- xHC loss 相对 mHC 约低 1.1%；
- xHC loss 相对 vanilla 约低 2.4%。

用最大 vanilla/mHC model 的 loss 作为 target，再从 xHC fitted curve 读取 matching compute，得到：

$$
\frac{
C_{\mathrm{vanilla}}
}{
C_{\mathrm{xHC}}
}
\approx
1.50,
$$

$$
\frac{
C_{\mathrm{mHC}}
}{
C_{\mathrm{xHC}}
}
\approx
1.19.
$$

### 12.3 证据强度审计

这个实验比只报告一个模型规模更强，但仍有明显边界：

1. 每条曲线只有 4 个 points；
2. $E=0.72$ 是固定估计，不是与 $A,\alpha$ 联合拟合；
3. 没有 raw final losses 表；
4. 没有 fit residual、置信区间、参数 covariance 或 held-out validation；
5. “following our fitting code”被写入论文，但官方仓库没有 fitting code；
6. 四个 points 同时改变 width、depth、expert FFN 和 token budget，不能解释为只沿单一 model-size 轴变化；
7. training-budget 列在附录仍为 `--`。

所以 $1.50\times$ 与 $1.19\times$ 应称为**论文拟合得到的 compute-equivalence estimate**，不是直接成对测量的硬件加速比。

---

## 13. $N$-sweep：xHC 是否真的让 expansion rate 可扩

### 13.1 xHC active-stream 配置

| $N$ | $k$ | $m$ |
|---:|---:|---:|
| 2 | 1 | 0 |
| 4 | 2 | 1 |
| 8 | 4 | 2 |
| 16 | 4 | 2 |

因此这个 sweep 不只改变 $N$：

- $N=2\to4$ 时 $k$ 从 1 变 2；
- $N=4\to8$ 时 $k$ 从 2 变 4；
- 只有 $N=8\to16$ 保持 $k=4,m=2$。

**[综合判断]** 从 $N=2$ 到 8 的收益不能完全归因于增加 inactive memory capacity，也包含 active budget 增加。

### 13.2 关键结果

从 $N=4$ 到 $N=16$：

| 方法 | Loss improvement | Training FLOPs increase |
|---|---:|---:|
| mHC | 0.006 | 32% |
| xHC | 0.012 | 4% |

Figure 1 还显示 mHC $N=32$ 的点，继续增加大量 FLOPs而 loss 改善有限。

### 13.3 论文内部的可复现性缺口

正文 §3.2 写：

$$
N\in\{2,4,8,16,32\}
$$

用于 mHC sweep；Figure 1 也画出 $N=32$。

但 Appendix A 写 mHC sweep 为：

$$
N\in\{2,4,8,16\}.
$$

另外 Figure 1 caption 指向 Table 6 “details”，Table 6 只列默认 xHC 配置，Table 7 只列 xHC 的 $N,k,m$，没有 mHC $N=32$ 的 hyperparameters 或 raw numeric loss/FLOPs。

这不推翻趋势，但使 $N=32$ 点和整张曲线难以独立复现。

---

## 14. Ablation：两个设计分别贡献什么

10B MoE，Pile test validation loss：

| Variant | $N$ | Temp Aug | Sparse | Dense Read | $k$ | Fixed | Router | FLOPs overhead | Val. loss |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Vanilla | — | — | — | — | — | — | — | 0 | 2.029 |
| mHC | 4 | — | — | — | — | — | — | 0.6% | 2.004 |
| mHC | 16 | — | — | — | — | — | — | 18.8% | 1.998 |
| mHC + Temp Aug | 16 | ✓ | — | — | — | — | — | 20.1% | 1.984 |
| **xHC** | 16 | ✓ | ✓ | ✓ | 4 | 2 | Sigmoid | 3.3% | **1.983** |
| w/o Dense Read & Fixed | 16 | ✓ | ✓ | ✗ | 4 | 0 | Sigmoid | — | 1.997 |
| w/o Dense Read | 16 | ✓ | ✓ | ✗ | 4 | 2 | Sigmoid | — | 1.985 |
| w/o Fixed | 16 | ✓ | ✓ | ✓ | 4 | 0 | Sigmoid | — | 1.986 |
| $k=2$ | 16 | ✓ | ✓ | ✓ | 2 | 1 | Sigmoid | — | 1.991 |
| $k=8$ | 16 | ✓ | ✓ | ✓ | 8 | 2 | Sigmoid | — | 1.982 |
| Softmax router | 16 | ✓ | ✓ | ✓ | 4 | 2 | Softmax | — | 1.988 |

### 14.1 最有信息量的结论

1. temporal augmentation 是 $N=16$ 质量提升的主要来源：

$$
1.998\to1.984;
$$

2. sparse architecture 的主要价值是保住质量并显著降低 FLOPs：

$$
20.1\%\to3.3\%;
$$

3. dense read 和 fixed streams 都有帮助，但各自效果较小；
4. $k=8$ 只比 $k=4$ 好 0.001，说明默认点在该设置中有合理 cost–quality trade-off；
5. sigmoid 比 softmax 好 0.005，但论文没有报告 router entropy/utilization，winner-take-all 解释仍是间接的。

### 14.2 Temporal augmentation 的额外 ablation

| Variant | Val. loss |
|---|---:|
| xHC default | 1.983 |
| w/o Gram–Schmidt | 1.984 |
| + Attention-side Temp Aug | 1.985 |

dense mHC $N=16$、不使用 sparse update：

| Temporal branches | Val. loss |
|---:|---:|
| 0 | 1.998 |
| 1 | 1.989 |
| 3 | 1.984 |

多尺度比单尺度进一步改善 0.005。

### 14.3 Gram–Schmidt 的证据边界

10B loss 几乎不受影响，但论文报告：

- 18B 时 conv branch 与 main branch cosine similarity 可超过 0.7；
- 移除 GS 会导致 training instability；
- Muon 设置下又移除 GS。

论文没有给出：

- instability curve；
- spike frequency；
- affected seeds；
- GS 与 no-GS 的 gradient/activation statistics table。

因此它是作者观察到的 scale/optimizer-dependent stability mechanism，尚不是具有普适保证的必要组件。

---

## 15. Muon 结果

18B MoE：

| Benchmark | AdamW Vanilla | Muon Vanilla | Muon + xHC |
|---|---:|---:|---:|
| MMLU | 48.9 | 51.3 | **56.6** |
| MMLU-Pro | 21.1 | 27.9 | **32.0** |
| MMLU-Redux | 46.4 | 49.0 | **52.2** |
| BBH | 32.4 | 36.1 | **42.2** |
| CommonsenseQA | 54.6 | 49.1 | **61.3** |
| ARC-Challenge | 55.7 | 63.6 | **75.3** |
| GSM8K | 37.7 | 41.8 | **52.8** |
| HumanEval | 25.6 | 21.3 | **28.7** |
| LCBench | 9.9 | 11.0 | **16.0** |
| CMMLU | 42.7 | 45.2 | **50.2** |
| CEval | 44.5 | 47.2 | **54.3** |
| C3 | 67.1 | 73.7 | **77.1** |
| **Average** | 40.6 | 43.1 | **49.9** |

这支持 xHC 的收益不局限于 AdamW。它不是干净的“optimizer-only”ablation，因为 Muon+xHC 同时：

- 增加 xHC architecture；
- 对不同 parameter groups 使用不同 optimizer；
- 移除 Gram–Schmidt。

因此不能从这张表单独判断 GS 与 Muon 的因果交互。

---

## 16. Wall-clock 与 inference 证据

### 16.1 Training

18B MoE、关闭 pipeline-communication overlap：

- 作者重实现的 fused mHC $N=4$ 相对 vanilla 增加约 15% training time；
- xHC-Flash-4sub 在 mHC 之上再增加约 11%。

若“on top of”按乘法解释，xHC-Flash-4sub 相对 vanilla 的总 overhead 约为：

$$
1.15\times1.11-1
\approx
27.7\%.
$$

这是**[复原推导]**，论文没有直接写 27.7%。

### 16.2 Prefill

2K-token prefill：

- mHC 相对 vanilla：+11.4%；
- xHC-Flash-4sub 相对 vanilla：+12.9%。

因此 xHC-Flash-4sub 相对 mHC：

$$
\frac{1.129}{1.114}-1
\approx
1.35\%,
$$

与论文所写约 1.3% 一致。

### 16.3 缺失信息

论文没有给出：

- GPU/accelerator 型号；
- 节点与 device 数；
- TP/PP/EP/DP 配置；
- micro/global batch；
- sequence parallel 设置；
- absolute step time 或 tokens/s；
- kernel-level profiler；
- decode latency；
- KV/temporal cache bytes。

所以这些百分比适合说明作者环境中的相对趋势，不适合直接拿来做部署预算。

---

## 17. 分布式实现推导

本节是 **[复原推导/综合判断]**，不是论文已实现并公开验证的并行方案。

### 17.1 Tensor Parallel

若 $C$ 在 TP ranks 间切分：

$$
X_l^{(r)}
\in
\mathbb R^{B\times S\times N\times C_{\mathrm{local}}}.
$$

router/pre projections 的输入也是按 $C$ sharded，但输出只有 $N$ 维：

$$
NC\to N.
$$

常见 contract 是每个 rank 计算 partial logits/readout，再做 reduce：

- router logits：对 hidden-sharded partial projection 做 all-reduce；
- pre logits：同样做 all-reduce；
- dense read 的 $C$ 输出仍可保持 hidden-sharded；
- $k\times k$ mappings 很小，适合 replicated；
- active indices 必须在 TP ranks 一致。

若不同 ranks 的 Top-$k$ tie-breaking 不一致，gather/scatter 会破坏数值正确性。

### 17.2 是否应该 shard stream 维

把 $N=16$ 按 ranks 切分会使：

- global Top-$k$ 需要 all-gather scores；
- dense read 需要跨 stream reduction；
- active streams 可能跨 ranks；
- $k\times k$ residual mixing 需要重新分发 active state。

由于 $N$ 小而 $C$ 大，默认更自然的设计是 replicated stream dimension、sharded hidden dimension。真实最优布局仍需根据框架和并行规模 benchmark。

### 17.3 Sequence / Context Parallel

router、dense read 和 stream mixing 对每 token 独立，容易沿 sequence 分片。temporal convolution 不独立：

$$
g_j[t]
=
\sum_{i=0}^{\kappa_j-1}
w_j[i]\odot\mathrm{out}[t-i].
$$

最大 kernel 为 12，因此 sequence shard 边界需要最多：

$$
\kappa_{\max}-1
=
11
$$

个左侧 token 的 halo，或等价的 causal state exchange。

正确性测试必须覆盖：

- shard boundary；
- sequence start padding；
- packed sequences 之间不能互相卷积；
- document boundary reset；
- recomputation 与 forward 使用相同 halo。

### 17.4 Pipeline Parallel

若 stage boundary 直接传 persistent state，payload 从：

$$
BSC
$$

增为：

$$
BSNC.
$$

默认 $N=16$ 时是 16 倍 element count。xHC-Flash 降低 stage 内 full-state reread，却不会自动压缩跨 stage 的 persistent state。

Flash sharing window 还要求：

- 两子层或四子层 group 不被 PP stage boundary 随意切开；
- 若切开，必须传递 active state、routing、pre mappings/base readouts 或重新计算；
- shared routing 的生命周期与 activation checkpoint schedule 对齐。

论文只提到可用 DualPipe overlap 降低有效 overhead，没有给出这些 payload 和 schedule contract。

### 17.5 Expert Parallel

xHC temporal augmentation 位于 MoE FFN 输出之后，所以输入应已完成 expert output combine。它本身不改变 expert routing，但会：

- 增加 MoE output 后的 local work；
- 延长下一子层前 residual update 的 critical path；
- 需要明确 conv 在 EP all-to-all/combine 之前还是之后。

按论文公式，应在合并得到完整 $C$ 维 MLP output 后执行。

### 17.6 Activation recomputation

可选择保存或重算：

- full state normalization statistics；
- router scores 和 Top-$k$ indices；
- active state；
- pre/res/post mappings；
- Sinkhorn intermediates；
- temporal convolution outputs；
- Gram–Schmidt projections。

论文明确 Sinkhorn intermediates 在 backward 重算。其他对象的 checkpoint policy 没有公开。

对 hard Top-$k$，重算需要 deterministic routing。即使 scores 只发生很小的 BF16/FP32 差异，处于 boundary 的 stream 也可能改变 selected set。

### 17.7 Autoregressive decode state

**[复原推导]** causal depthwise convolution 在单 token decode 时需要保存此前 MLP outputs。对最大 kernel 12，至少要维护最近 11 个 positions 的 per-layer history：

$$
O
\left(
L(\kappa_{\max}-1)C
\right)
$$

elements，具体能否在多个 branches 间共享 ring buffer 取决于实现。

论文只报告 2K prefill latency，没有报告：

- temporal cache layout；
- per-token decode latency；
- continuous batching 行为；
- sequence reset；
- beam/speculative decoding 下的 cache fork。

这是 xHC 从 pretraining/prefill 走向 serving 时最需要补的系统证据之一。

---

## 18. CPU reference implementation 的最小 contract

当前项目主线不需要 GPU。一个 correctness-first reference 应先实现清晰版本，再考虑 fusion。

### 18.1 伪代码

```python
def xhc_sublayer(x, branch, is_mlp):
    # x: [B, S, N, C]
    flat = x.flatten(-2)                     # [B, S, N*C]

    scores = sigmoid(layer_norm(flat) @ w_router)
    h_pre = sigmoid(
        alpha_pre * (rms_norm(flat) @ w_pre) + b_pre
    )

    indices, route_weight = fixed_plus_topk(scores, m, k)
    x_active = gather_streams(x, indices)    # [B, S, k, C]

    active_flat = rms_norm(x_active.flatten(-2))
    h_res = sinkhorn(exp(
        alpha_res * reshape(active_flat @ w_res, [k, k]) + b_res
    ))
    h_res = row_sum_clamp(h_res)

    k_r = 4 if is_mlp else 1
    h_post = 2 * sigmoid(
        alpha_post
        * reshape(active_flat @ w_post, [k, k_r])
        + b_post
    )

    branch_input = einsum("bsn,bsnc->bsc", h_pre, x)
    out = branch(branch_input)
    out_aug = temporal_augment(out) if is_mlp else out[..., None, :]

    mixed = einsum("bskq,bsqc->bskc", h_res, x_active)
    write = einsum("bskq,bsqc->bskc", h_post, out_aug)
    x_active_new = mixed + route_weight[..., None] * write

    return scatter_streams(x, indices, x_active_new)
```

### 18.2 必须有的单元测试

1. **Shape**

$$
[B,S,N,C]\to[B,S,N,C].
$$

2. **Inactive exact carry**

$$
i\notin\mathcal I
\Longrightarrow
X_{l+1,i}=X_{l,i}.
$$

3. **Dense-read reachability**

修改任一 inactive stream，在 $h_i^{pre}\ne0$ 时 branch input 应变化。

4. **Route-weight placement**

$p_j$ 只缩放 write-back，不缩放：

$$
\mathcal H^{res}X_{\mathrm{active}}.
$$

5. **Causality**

改变未来 token 的 $\mathrm{out}[t+1:]$ 不得改变当前：

$$
\mathrm{out}_{aug}[t].
$$

6. **Packed-sequence isolation**

前一文档末尾不得影响后一文档开头。

7. **Gram–Schmidt**

对非退化输入：

$$
\left|
\langle v_i,v_j\rangle
\right|
<\varepsilon
\qquad
(i\ne j).
$$

还要构造 near-zero component 测 denominator policy。

8. **Sinkhorn**

clamp 前检查近似 row/column sums；clamp 后单独检查：

$$
\text{row sum}\le1
$$

并记录 column deviation，不能仍断言精确双随机。

9. **Parameter-count formula**

state-dict dominant weights 应匹配：

$$
\left(
4N^2+2k^3+k^2+k^2K_r+\sum\kappa_i
\right)C.
$$

10. **Final collapse**

$$
[B,S,N,C]
\xrightarrow{\sum_N}
[B,S,C].
$$

11. **Gradcheck**

用固定 indices 或远离 Top-$k$ boundary 的 scores，对 smooth path 做 float64 `gradcheck`。

12. **Selected routing gradient**

动态 selected logits 有梯度；unselected/fixed scores 的行为与 contract 一致。

13. **Flash correction**

显式更新 full state 后重新 dense read，与：

$$
\mathrm{inp}_{M}
+
\alpha\mathrm{out}_{A}
$$

在 tolerance 内相等。

14. **Reference–fused parity**

若未来实现 kernel，分别检查 forward、input grad、mapping grad、router grad 和 conv-kernel grad。

---

## 19. 论文之外的扩展与新实验

### 19.1 直接验证 information bottleneck

**[扩展假设]** 对每层记录 stream matrix：

$$
X_l[t]
\in
\mathbb R^{N\times C}
$$

的 singular values，定义 effective rank：

$$
r_{\mathrm{eff}}
=
\exp
\left(
-
\sum_i
p_i\log p_i
\right),
$$

$$
p_i
=
\frac{\sigma_i}{\sum_j\sigma_j}.
$$

比较：

- mHC $N=4,8,16,32$；
- xHC without TempAug；
- xHC single-scale；
- xHC multi-scale。

如果 diagnosis 正确，temporal augmentation 应随 $N$ 墑大更明显地提高 stream-space effective rank，而不只是降低 loss。

### 19.2 Router utilization

记录每条 non-fixed stream：

$$
u_i
=
\frac{
\#\{(b,t,l):i\in\mathcal I_{b,t,l}\}
}{
\#\{(b,t,l)\}
}.
$$

还应看：

- entropy；
- layer-wise utilization；
- token-domain dependence；
- longest inactive streak；
- fixed streams 与 routed streams 的 norm/gradient ratio。

这可以检验 sigmoid 是否真的避免 persistent starvation。

### 19.3 稳定的正交化替代

**[扩展假设]**

- denominator 加显式 $\epsilon$；
- normalize 后做 modified GS；
- Householder/QR；
- learned decorrelation penalty；
- 只在 cosine similarity 超阈值时 orthogonalize。

需要同时比较 loss、activation spikes、gradient spikes、FLOPs 和 kernel complexity。

### 19.4 Adaptive $k$

当前所有 token 大多使用固定 $k=4$。可以令 easy token 用更小 $k$，复杂 token 用更大 $k$：

$$
k_{b,t,l}
\in
\{2,4,8\}.
$$

但这会带来：

- dynamic shape；
- load imbalance；
- kernel specialization 困难；
- PP/TP routing metadata；
- 新的 budget regularization。

因此它不是无成本扩展。

### 19.5 降低 dense read 的 $O(N^2C)$ generator

xHC 消除了 $N^3C$ active mapping bottleneck，但 router 和 pre generator 仍为：

$$
NC\to N
\quad\Rightarrow\quad
O(N^2C).
$$

若继续扩到 $N=32,64$，这会成为新瓶颈。候选方案：

- factorized projection；
- shared low-rank stream descriptors；
- grouped streams；
- periodic full routing + cheap intermediate update；
- state summary 生成 routing，full state 只用于 value read。

### 19.6 xHC 与 AttnRes 的关系

xHC 保存固定大小：

$$
N\times C
$$

的 learned recurrent depth state；AttnRes 保存或压缩历史 depth sources，再沿 depth attention。

二者分别类似：

- xHC：有限槽位的可学习 recurrent memory；
- AttnRes：对历史 source 的 content-addressable retrieval。

潜在 hybrid 可以：

- 用 AttnRes 选择 block-level history；
- 用 xHC 在 block 内维护多流 state。

但这会叠加 state、routing、pipeline payload 和 normalization 问题，必须先有明确的 resource budget。

### 19.7 serving 必做实验

至少补：

1. batch-1 decode latency；
2. continuous batching throughput；
3. temporal-conv cache bytes；
4. context length 2K/8K/32K prefill；
5. TP/PP/EP 分解；
6. xHC-Flash sharing window 与 PP stage alignment；
7. BF16 reference vs fused error；
8. end-to-end quality of Flash at 18B/28B。

---

## 20. 证据账本

| Claim | 证据 | 支持强度 | 主要边界 |
|---|---|---|---|
| mHC 大 $N$ 收益饱和 | Figure 1，2.5B $N$-sweep | 中 | 无 raw table；$N=32$ 配置前后不一致 |
| 单一 write-back component 是瓶颈 | Eq. 3 + TempAug ablation | 中 | 没有直接 stream-rank/utilization 测量 |
| temporal augmentation 提升大 $N$ 质量 | Table 2、12、Figure 5 | 较强 | 只在 in-house MoE/Pile setting；无多 seed |
| sparse update 保持质量并降 FLOPs | Table 2 | 较强 | validation loss 单点；non-param FLOPs 未完整计入 |
| dense read 必要 | Table 2 rows 5–8 | 中 | 差异较小，无统计误差 |
| $k=4$ 是合理 trade-off | Table 2 rows 9–10 | 中 | 只测 $k=2,4,8$，无 wall-clock |
| xHC 提升 18B/28B downstream | Table 1 | 较强 | 无 seed/CI；数据不透明 |
| compute efficiency 更高 | Figure 4 + Table 9 | 中 | 每方法 4 点；固定 $E$；无 fit uncertainty |
| Muon 下仍有效 | Table 3 | 中–较强 | optimizer 分组与 GS 同时改变 |
| Flash correction 精确 | Eq. 28–37 的代数 | 强，限定义内 | 不等价于 full xHC |
| Flash 保留 10B loss | Table 5 | 中 | 没有大模型 downstream |
| I/O 降到 $40C$ | Table 4 的静态 accounting | 中 | 非实际 bytes/throughput；排除 branch I/O |
| fused runtime 可接受 | §5.3 相对 overhead | 弱–中 | 缺硬件、并行配置和绝对吞吐 |

---

## 21. 最终评价

### 方法层

xHC 的设计逻辑很完整：

$$
\text{more streams need more write-back diversity}
$$

与：

$$
\text{more streams cannot afford dense dynamic mixing}
$$

分别由 temporal augmentation 和 sparse active-set update 处理。最值得肯定的是，它没有把 read 和 write 一起稀疏化，而是明确保留：

$$
\text{dense read}
+
\text{sparse mutation}.
$$

这使状态容量和每步更新成本被部分解耦。

### 数学层

关键公式、shape 和参数化足够清晰，Flash correction 也在附录中给出了可检查推导。但稳定性论证仍主要继承 mHC 的 active-set Sinkhorn 直觉：

- finite Sinkhorn 只是近似；
- row-sum clamp 会破坏精确双随机；
- hard routing 使全模型是 piecewise-smooth dynamic system；
- temporal write-back 和 final stream sum 的整体尺度没有 theorem。

### 实验层

18B/28B 的提升幅度大，ablation 也能对应论文提出的两个瓶颈，是这篇工作的强项。最需要补的是：

- data/training-token transparency；
- repeated runs；
- stream diversity 的直接度量；
- scaling fit uncertainty；
- dense model 与非语言任务；
- Flash 大模型 downstream。

### 系统层

论文对 fusion boundary、dtype 和 I/O decomposition 的描述比一般 architecture paper 详细。但没有公开 code 和硬件配置，使“可部署”仍是 paper-level evidence，而不是可独立复现的工程结论。

### 一句话 takeaway

> xHC 的核心不是简单把 mHC 的 $N$ 从 4 改成 16，而是把多流 residual system 重构成“全量读取、稀疏更新、富基写回”的状态机；它用 $k$ 控制每层 mutation 成本，用 $N$ 控制持久 residual memory 容量，再用 temporal components 避免更多槽位只存同一 write-back 方向的不同倍数。

---

## 22. 来源

### Primary sources

- Zhang et al., [*xHC: Expanded Hyper-Connections* — arXiv HTML v1](https://arxiv.org/html/2607.14530v1)
- Zhang et al., [PDF v1](https://arxiv.org/pdf/2607.14530v1)
- 官方项目页：[aHapBean/xHC](https://github.com/aHapBean/xHC)
- 本文核对的固定 artifact：[commit `7890266d5cd648811b6783029ee6b5031cd209db`](https://github.com/aHapBean/xHC/tree/7890266d5cd648811b6783029ee6b5031cd209db)

### 前置论文

- Zhu et al., [*Hyper-Connections*](https://arxiv.org/abs/2409.19606)
- Xie et al., [*mHC: Manifold-Constrained Hyper-Connections*](https://arxiv.org/abs/2512.24880)

### 本项目相关解读

- [01 — Hyper-Connections](/topics/residual/hyper-connections/)
- [02 — mHC](/topics/residual/mhc/)
- [03 — Attention Residuals](/topics/residual/attention-residuals/)
