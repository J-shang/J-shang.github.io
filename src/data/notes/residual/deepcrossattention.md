---
title: "DeepCrossAttention：让 Q、K、V 分别动态选择历史深度"
description: "解释 GRN 与 DCA 如何为 Q、K、V 建立 feature-aware depth aggregation，并审计理论保证、参数效率和评测边界。"
topic: "residual"
section: "methods"
slug: "deepcrossattention"
date: 2026-07-27
updated: 2026-07-27
cutoff: 2026-07-27
order: 15
readtime: 16
source:
  repository: "J-shang/residual"
  path: "papers/06-deepcrossattention.md"
  url: "https://github.com/J-shang/residual/blob/c55583707ab645bef7408c588fc33cbf91b809a5/papers/06-deepcrossattention.md"
  revision: "c55583707ab645bef7408c588fc33cbf91b809a5"
  syncedAt: "2026-07-28"
  contentHash: "sha256:96744bffa135cb2fd84a74e03ab46c6c0b45da37e3ed6b9c1d954b46fcf6f4f0"
  manifest: "residual"
  managed: true
---
<!-- paper-order: 06 -->

> **论文**：Mike Heddes、Adel Javanmard、Kyriakos Axiotis、Gang Fu、Mohammadhossein Bateni、Vahab Mirrokni, *DeepCrossAttention: Supercharging Transformer Residual Connections*<br>
> **机构**：University of California, Irvine；University of Southern California；Google Research<br>
> **版本**：ICML 2025 / PMLR 267；arXiv:2502.06785v2，2025-07-23<br>
> **状态**：ICML 2025 正式发表，不是 Independent Researcher 投稿<br>
> **主来源**：[PMLR 正式页面](https://proceedings.mlr.press/v267/heddes25a.html) · [arXiv PDF](https://arxiv.org/pdf/2502.06785v2) · [arXiv](https://arxiv.org/abs/2502.06785)<br>
> **官方 artifact**：论文与正式补充材料可用；截至核对日未在论文/PMLR 页面发现作者公开的训练代码仓库<br>
> **阅读范围**：完整 23 页，包括 GRN 三种参数化、DCA、理论、全部实验、证明和附录分析<br>
> **信息核对日期**：2026-07-27<br>
> **前置阅读**：[DenseFormer](/topics/residual/denseformer/)

## 证据标签

- **[论文报告]**：论文直接报告。
- **[复原推导]**：由论文定义直接推出。
- **[综合判断]**：对论文证据的解释或系统推论。
- **[扩展假设]**：需要新实验的设计。
- **[待验证]**：来源不足。

---

## 先给结论

### 30 秒版

DCA 的核心并不是修改 token attention 的 softmax，而是修改 Attention 之前的输入来源。

它先把标准 residual sum 写成历史 branch outputs 的 stack：

$$
G_t=[f_{t-1}(g_{t-1}(x)),\ldots,f_0(g_0(x))]\in\mathbb R^{D\times t}.
$$

标准 residual 相当于：

$$
g_t(x)=G_t\mathbf1.
$$

DCA 用 input-dependent 的 Generalized Residual Network（GRN-v3）分别生成三份历史聚合：

$$
Q_t=\operatorname{GRN}_t^Q(G_t),\quad
K_t=\operatorname{GRN}_t^K(G_t),\quad
V_t=\operatorname{GRN}_t^V(G_t),
$$

再送入普通 self-attention。于是第 $t$ 层可以让 Q 来自某些深度、K 来自另一些深度、V 再来自另一组深度。

它与后来 AttnRes 的差别是：DCA 的 depth mixing 权重不是标准 query–key dot-product + softmax；其 GRN-v3 使用 learned feature-wise bias，再加由当前 stack 投影得到的 ReLU 动态项。

### 论文最强结果

**[论文报告]**

- LM1B、24-layer：
  - Transformer 最终 PPL 15.14；
  - 2-DCA 最终 PPL 14.41；
  - 2-DCA 每步更慢，但达到 Transformer 最终 PPL 所需归一化训练时间是 0.33，即论文所称“约 3× faster to same quality”。
- C4 从 75M 到 449M，DCA 均优于 Transformer，但绝对 PPL gap 从 1.443 缩到 0.402。
- 6-layer LM1B：Transformer 18.98，DCA 18.06；论文重实现的 HC 为 18.65、DenseFormer 为 18.80。
- ViT-S/16 ImageNet：accuracy 从 76.4 提升到 77.1。
- 从 500k-step Transformer function-preserving 地加入 DCA，再训练 60k steps，PPL 从 18.98 降到 18.79；继续训练原模型约 18.96。

### 我的总判断

**[综合判断]** DCA 是四篇里理论最重、也最容易被标题误导的一篇。

其可靠经验结论是：

> 在论文的 LM1B/C4 与 ViT 设置中，动态、分路的 depth aggregation 能改善参数效率和达到目标 PPL 的时间。

但其理论并没有证明完整 Transformer DCA 一定优于标准 Transformer。理论针对的是 stylized low-rank linear residual model；非线性部分给出 bottleneck-rank 的类包含关系，而非 DCA 训练动力学或泛化定理。

---

## 5 分钟方法地图

| 层面 | DCA 的答案 |
|---|---|
| motivating failure | 等权 residual sum 可能稀释少数关键层的信息 |
| stored state | embedding 与历史 residual branch outputs 的 stack |
| GRN-v1 | source-depth scalar |
| GRN-v2 | feature × source-depth 权重 |
| GRN-v3 | GRN-v2 bias + token/input-dependent depth term |
| DCA | 每个 decoder block 放 3 个独立 GRN-v3，生成 Q/K/V |
| 稀疏近似 | 保留 model input、聚合后的中间状态、最近 $k$ 个 outputs |
| 初始化 | 静态 bias 全 1，动态 projection 全 0；恢复 residual sum |
| 权重约束 | ReLU 只施加于动态项；最终 bias + 动态项不受 simplex 约束 |
| 理论对象 | low-rank linear ResNet / bottleneck rank |
| 主要成本 | 历史 activations、3 路 depth aggregation、memory traffic |

### 贡献账本

| 类型 | 贡献 | 新颖性与证据 |
|---|---|---|
| conceptual | 将 fixed residual sum 解释为对 history stack 乘全 1 向量 | 为 GRN 参数化提供统一起点 |
| methodological | GRN-v1/v2/v3：depth scalar → feature-aware → input-dependent | 逐级扩大 residual aggregation 函数类 |
| architectural | 用 3 个独立 GRN-v3 为 Attention 构造 Q/K/V | DCA 核心 |
| theoretical | low-rank linear model 的函数类与 risk–size trade-off | stylized setting 内成立 |
| empirical | LM1B/C4 depth/width/size sweep、retrofit、ViT | 中等规模、多角度证据 |
| systems | first + last-$k$ history 压缩与 time-to-target 测量 | 支持质量–效率 trade-off |
| artifact | 正式论文与证明公开 | 未识别到官方训练代码 |

### 符号、shape 与身份

| 符号 | 含义 | 教学 shape | 身份 |
|---|---|---:|---|
| $f_t$ | 第 $t$ 个 residual branch | $\mathbb R^D\to\mathbb R^D$ | 普通模型函数 |
| $g_t(x)$ | 第 $t$ 个 residual state | `[D]` 或 `[B,T,D]` | 运行时张量 |
| $G_t$ | 历史 branch outputs 的 stack | `[D,t]` 或 `[B,T,D,t]` | 保存状态 |
| $b_t$ | 静态聚合权重 | `[t]` 或 `[D,t]` | 可训练参数 |
| $w_t$ | 动态 score projection | `[D,1]` | 可训练参数 |
| $\bar w_t$ | input-dependent depth scores，沿 feature 广播 | `[D,t]` | 运行时张量 |
| $k$ | first + last-$k$ 中显式保留的最近输出数 | integer | 架构超参数 |
| $r_t,r_*$ | 单层 rank 与 collective rank | integer | 理论对象 |

---

## 1. 从 residual sum 到显式 history stack

论文定义 residual branch：

$$
f_t:\mathbb R^D\to\mathbb R^D,
$$

标准递归：

$$
g_{t+1}(x)=f_t(g_t(x))+g_t(x).
$$

设：

$$
g_0(x)=0,\qquad f_0(g_0(x))=x.
$$

则：

$$
g_{T+1}(x)=\sum_{t=0}^{T}f_t(g_t(x)).
$$

把各 branch output 堆为列：

$$
G_t=
\left[
f_{t-1}(g_{t-1}(x)),
\ldots,
f_0(g_0(x))
\right]
\in\mathbb R^{D\times t}.
$$

标准 residual 变成：

$$
g_t(x)=G_t\mathbf1.
$$

这一步揭示 DCA 的真正设计空间：把固定 $\mathbf1$ 换成可学习、feature-dependent、input-dependent 的权重。

在 sequence model 中，每个 token 都有一个对应的 $G_t$。实现 shape 更接近：

$$
G_t\in\mathbb R^{B\times T\times D\times t}.
$$

---

## 2. 三种 GRN

### 2.1 GRN-v1：depth scalar

$$
g_t(x)=G_tb_t,
\qquad b_t\in\mathbb R^{t}.
$$

$b_t$ 初始化为全 1。它对所有 feature 共用 depth 权重，本质上接近 DenseFormer 的静态标量聚合。

差别主要来自记号：DenseFormer 存 block outputs，DCA 理论部分存显式 residual branch outputs。做代码比较时不能只凭公式外形判断张量身份相同。

### 2.2 GRN-v2：feature-dependent depth weights

$$
g_t(x)
=
(G_t\odot b_t)\mathbf1,
\qquad
b_t\in\mathbb R^{D\times t}.
$$

对第 $d$ 个 feature：

$$
[g_t(x)]_d
=
\sum_{j=1}^{t}G_t[d,j]\,b_t[d,j].
$$

这等价于每个 feature channel 拥有不同 depth filter，但对不同 token/input 仍是静态的。

### 2.3 GRN-v3：再加入 input-dependent 项

论文写作：

$$
\bar w_t
=
\mathbf1_D\,\sigma(w_t^\top G_t),
\qquad
w_t\in\mathbb R^{D\times1},
$$

其中 $\sigma=\operatorname{ReLU}$，$\bar w_t\in\mathbb R^{D\times t}$ 通过广播在 feature 维重复。聚合为：

$$
g_t(x)
=
\left(
G_t\odot(b_t+\bar w_t)
\right)\mathbf1.
$$

含义：

- $b_t[d,j]$：第 $d$ 个 feature 对第 $j$ 个 depth 的静态偏置；
- $w_t^\top G_t[:,j]$：当前输入下，第 $j$ 个 source 的 scalar score；
- ReLU 后的动态 score 对所有 feature 广播；
- 最终权重是 $b_t+\bar w_t$，不是 softmax。

因此 DCA 的 “attention across depth” 是功能类比，不是标准 attention 公式。

### 2.4 初始化

论文令：

$$
b_t=\mathbf1,\qquad w_t=0.
$$

则：

$$
\bar w_t=0,\qquad
g_t(x)=G_t\mathbf1.
$$

**[复原推导]** GRN-v3 初始化时精确恢复标准 residual sum。这个性质也使作者可以把 DCA 插入预训练 residual 模型而不改变初始函数。

---

## 3. 从 GRN 到 DeepCrossAttention

### 3.1 三路 depth aggregation

对第 $l$ 个 Transformer block，DCA 实例化三个独立 GRN-v3：

$$
\begin{aligned}
\widetilde X_l^Q&=\operatorname{GRN}_l^Q(G_l),\\
\widetilde X_l^K&=\operatorname{GRN}_l^K(G_l),\\
\widetilde X_l^V&=\operatorname{GRN}_l^V(G_l).
\end{aligned}
$$

然后：

$$
A_l
=
\operatorname{MHA}
\left(
\operatorname{Norm}(\widetilde X_l^Q),
\operatorname{Norm}(\widetilde X_l^K),
\operatorname{Norm}(\widetilde X_l^V)
\right).
$$

论文强调 underlying attention mechanism 不变；改变的是 Attention 的 Q/K/V 输入表示。

### 3.2 residual placement

论文 Figure 4 说明，普通“把 block input 再加到 block output”的路径被重新安排：model input/history 已经包含在 Attention 和 FFN 的 GRN inputs 中，不能机械地在外面再加一次，否则会改变函数与初始化等价性。

**[综合判断]** 实现时最危险的错误不是 GRN einsum，而是 residual add 的重复或遗漏。必须以计算图而不是“DCA 是一个插件层”的直觉来插入。

### 3.3 First + last-$k$

full history 随深度增长。论文给出压缩版本：

- 显式保留 model input；
- 显式保留最近 $k$ 个 layer outputs；
- 更早的中间层继续用普通 residual sum 聚成一个 state。

当 $k=2$ 时，GRN 最多看到 4 个 vectors：

1. model input；
2. 中间历史的累计和；
3. 倒数第 2 个 output；
4. 倒数第 1 个 output。

论文记作 `k-DCA`。这不是普通 sliding window，因为 model input 和累计中间 history 仍然存在。

---

## 4. 理论部分应该怎样读

### 4.1 stylized low-rank model

论文考虑：

$$
y=Ax+\epsilon,\qquad
A\in\mathbb R^{D\times D},
$$

并令每层为低秩线性映射，collective rank：

$$
r_*=\sum_{t=1}^{T}r_t<D.
$$

Theorem 4.1 给出的函数类核心是：

$$
\begin{aligned}
\mathcal C_{\text{base}}
&=
\{x\mapsto Mx:\operatorname{rank}(M)\le\min_t r_t\},\\
\mathcal C_{\text{res}}
&=
\{x\mapsto(I+M)x:\operatorname{rank}(M)\le r_*\},\\
\mathcal C_{\text{GRN-v1}}
&=
\{x\mapsto(\alpha I+M)x:\operatorname{rank}(M)\le r_*\},\\
\mathcal C_{\text{GRN-v2}}
&=
\{x\mapsto(D_{\mathrm{diag}}+M)x:
\operatorname{rank}(M)\le r_*\}.
\end{aligned}
$$

直觉是：

- 标准 ResNet 固定提供 $I$；
- GRN-v1 可以学习 $\alpha I$；
- GRN-v2 可以学习任意 diagonal map；
- 这些自由度不需要用低秩 branches 去“浪费” collective rank。

### 4.2 risk–model-size trade-off

在 isotropic features 假设：

$$
\mathbb E[xx^\top]=I,
$$

线性模型的 excess risk 化为：

$$
\operatorname{ER}(\widehat A)
=
\|A-\widehat A\|_F^2.
$$

论文用奇异值近似与参数计数，给出 GRN 在某些 $r_*/D$ 阈值下优于 ResNet 的充分条件。

### 4.3 理论支持与不支持

**支持：**

- 固定 identity shortcut 会限制低秩 residual branches 如何分配表达能力；
- 学习 scalar / diagonal shortcut 可以在相同 collective rank 下扩大线性函数类；
- 当 collective rank 相对 ambient dimension 较小时，额外 shortcut 自由度更有价值。

**不支持：**

- 完整 DCA 的 Q/K/V 三路结构一定获得同样定理；
- Transformer feature 真满足 isotropic；
- $r_t$ 等同于真实 learned representation rank；
- 优化一定找到理论最优近似；
- 大模型越宽，DCA 就一定无用。

论文对非线性模型使用 bottleneck rank：

$$
\operatorname{rank}_{BN}(f\circ g)
\le
\min\{\operatorname{rank}_{BN}(f),\operatorname{rank}_{BN}(g)\},
$$

$$
\operatorname{rank}_{BN}(f+g)
\le
\operatorname{rank}_{BN}(f)+\operatorname{rank}_{BN}(g).
$$

这提供函数类层面的延伸，但不是完整的泛化或训练稳定性证明。

---

## 5. 实验审计

### 5.1 主训练设置

**[论文报告]**

- LM1B 与 C4；
- 默认 $D=512$，MLP width $4D$；
- sequence length 128；
- batch size 2048；
- 64 TPUs；
- 500k steps，总计约 131B tokens；
- AdamW，$\beta_1=0.9,\beta_2=0.98$，weight decay 0.1；
- learning rate 0.0016，1000 warmup，inverse-square-root schedule。

论文多处报告均值与误差，但没有公开代码，独立复现门槛高于 DenseFormer 和 MUDDFormer。

### 5.2 depth scaling

LM1B 中，作者比较 6–42 layers。30-layer DCA 的 PPL 优于 42-layer Transformer。

**支持：** 在该 width 和 recipe 下，DCA 的参数效率优于单纯加深。

**缺口：** 图中没有给出所有点的表格数值；“更深模型收益保持”只覆盖该实验尺度。

### 5.3 first + last-$k$ 效率

24-layer：

| 方法 | speed (batches/s) ↑ | 达到 Transformer 最终 PPL 的时间 ↓ | final PPL ↓ |
|---|---:|---:|---:|
| Transformer | 8.14 | 1.00 | 15.14 |
| 1-DCA | 5.62 | 0.33 | 14.48 |
| 2-DCA | 5.39 | 0.33 | 14.41 |
| 4-DCA | 5.01 | 0.37 | 14.50 |
| 8-DCA | 4.35 | 0.47 | 14.49 |
| 16-DCA | 3.86 | 0.40 | 14.35 |
| 24-DCA | 3.72 | 0.39 | 14.35 |

最精确的结论是：

> DCA 每 step 更慢，但 loss 下降更快；在该设置中，small-$k$ DCA 达到 baseline 最终 PPL 的 wall-clock 约为 baseline 的三分之一。

不能把“3× faster”理解成 step throughput 或 inference 都快 3×。表中 2-DCA 的 batches/s 实际低于 Transformer。

### 5.4 width scaling

12-layer LM1B：

| width | Transformer PPL | DCA PPL | gap |
|---:|---:|---:|---:|
| 64 | 45.75 | 42.94 | -2.82 |
| 192 | 25.49 | 23.92 | -1.57 |
| 384 | 18.86 | 17.83 | -1.03 |
| 768 | 14.70 | 14.11 | -0.59 |
| 1024 | 13.61 | 13.22 | -0.39 |

gap 随 width 增大而缩小，与 low collective-rank 理论的方向一致。

**[综合判断]** 这是 theory-aligned evidence，不是理论验证。width 改变还会同时改变优化、容量与 loss 区间。

### 5.5 C4 model scaling

| depth | width | params | Transformer | 8-DCA |
|---:|---:|---:|---:|---:|
| 9 | 771 | 75M | 27.876 | 26.443 |
| 18 | 771 | 124M | 23.013 | 21.810 |
| 13 | 1111 | 179M | 21.570 | 20.461 |
| 18 | 1111 | 234M | 19.756 | 18.824 |
| 18 | 1600 | 449M | 17.166 | 16.764 |

所有点都改善，但 gap 确实在最大点缩小。论文没有超过 449M 的 DCA 主模型。

### 5.6 related-work comparison

6-layer LM1B、同一作者代码框架：

| 方法 | params | PPL ↓ |
|---|---:|---:|
| Transformer | 49.65M | 18.98 |
| LAuReL-PA | 49.75M | 18.99 |
| DenseFormer | 49.65M | 18.80 |
| Hyper-Connections | 49.68M | 18.65 |
| DCA | 49.73M | 18.06 |

C4 179M：

| 方法 | PPL ↓ |
|---|---:|
| Transformer | 21.534 |
| LAuReL-PA | 20.951 |
| DenseFormer | 21.168 |
| HC, stack 4 | 21.077 |
| HC, stack 10 | 20.718 |
| 8-DCA | 20.392 |

这些是有价值的同框架比较，但应注意：

- 它们是 DCA 作者的 reimplementation；
- 各方法最佳超参数搜索预算未必等价；
- LAuReL 原论文推荐优先尝试 RW/RW+LR，而这里主要列 PA；
- HC 的 stack size 会同时改变状态容量和成本。

### 5.7 ablation

6-layer LM1B：

| 方法 | PPL ↓ |
|---|---:|
| Transformer | 18.98 |
| GRN-v1 | 18.80 |
| GRN-v2 | 18.43 |
| GRN-v3 | 18.41 |
| DCA（三路 GRN-v3） | 18.06 |

动态项从 v2 到 v3 的增益很小，三路 Q/K/V decoupling 的增益更明显。

**[综合判断]** 这提示 DCA 的优势不能只归因于“input-dependent”；feature-wise static mixing 与多路输入解耦都贡献很大。

### 5.8 训练稳定性

论文在若干 Transformer loss curve 中看到 spikes，而 DCA 没有明显 spikes。

这是经验观察，不是稳定性保证。缺少：

- spike 定义；
- 跨 seeds 的发生率；
- gradient / activation norm 诊断；
- 恢复策略与 optimizer-state 分析。

---

## 6. Claim–evidence 对照

| 主张 | 证据 | 强度 | 保留意见 |
|---|---|---|---|
| 防止 information dilution | toy low-rank task、理论函数类、LM 结果 | 中等 | “dilution”不是直接测量量 |
| 同质量最多 3× 更快 | 24-layer LM1B time-to-target | 强，限该设置 | 不是吞吐 3× |
| 参数增量 negligible | 6-layer 约 0.16%；PMLR 摘要举例 0.2% | 强 | activation 与 traffic 仍显著 |
| 越窄越受益 | width sweep + 理论方向 | 中等 | 不能直接外推真实 effective rank |
| 更稳定 | loss curves | 偏弱 | 缺少系统统计 |
| 优于相关残差方法 | LM1B/C4 同框架表 | 中等 | reimplementation 与调参公平性 |
| 可 retrofit | 500k checkpoint continuation | 中等 | 只测 6-layer LM1B |

---

## 7. 系统实现含义

### 7.1 状态与计算

full DCA 需要历史 stack：

$$
G_l\in\mathbb R^{B\times T\times D\times m_l},
$$

其中 $m_l$ 随 depth 增长。每个 block 做三路聚合。即使参数很少，读取量近似：

$$
O(3BTD\,m_l).
$$

全深度求和是 $O(BTDL^2)$ 级的 depth-wise elementwise 工作。first+last-$k$ 把 $m_l$ 限制为约 $k+2$，将其降到 $O(BTDLk)$。

### 7.2 TP 与 sequence parallel

- feature 维切分时，GRN-v2 的 $b_t$ 随 feature shard；
- $w_t^\top G_t[:,j]$ 是对 hidden dimension 的 reduction，TP 下可能需要 all-reduce；
- Q/K/V 三个 GRN 各有投影，可合并计算以减少 collectives；
- sequence shard 下，每个 token 独立做 depth routing，通常可本地执行。

DenseFormer 的 scalar $\alpha$ 不需要 hidden reduction；DCA 的动态 score 会带来新的 TP 设计点。

### 7.3 PP

与 full DenseFormer 相同，任意历史读取和 pipeline stages 天然冲突。first+last-$k$ 仍需明确：

- “model input”在哪些 stages 复制；
- 中间累计 state 如何跨 stage 传递；
- 最近 $k$ 个 outputs 是否跨 stage boundary；
- backward/recompute 时谁拥有 history。

### 7.4 fused path contract

一个 fused GRN-v3 至少要与 reference 满足：

$$
\operatorname{reduce}_{depth}
\left[
G\odot
\left(
b+\operatorname{broadcast}_D(\operatorname{ReLU}(G^\top w))
\right)
\right].
$$

数值测试要覆盖：

- ReLU 边界；
- static bias 可为负；
- BF16 input、FP32 accumulation；
- Q/K/V 参数独立；
- first+last-$k$ source ordering；
- backward 对 history、$b$、$w$ 的梯度。

---

## 8. CPU reference 学习任务

建议先实现单 token GRN，再扩到 `[B,T,D,M]`：

```python
dynamic = torch.relu(torch.einsum("...dm,d->...m", history, w))
weights = bias + dynamic.unsqueeze(-2)
output = (history * weights).sum(dim=-1)
```

必须核对具体维序；上面只是教学记号。

测试清单：

1. v1/v2/v3 shape；
2. `bias=1,w=0` 恢复历史求和；
3. v1 是 v2 的 feature-shared special case；
4. 动态项对 feature 广播正确；
5. 三个 GRN 参数互不共享；
6. DCA residual placement 不重复 add；
7. full history 与 first+last-$k$ 的 source identity；
8. `gradcheck`；
9. 与明确 for-loop oracle 对齐；
10. 两进程 Gloo 模拟 TP hidden reduction。

---

## 9. 主要局限

- 最大 LM 主结果为 449M，离当前大模型规模仍远；
- sequence length 128，不能说明长上下文行为；
- 未发现官方训练代码，复现成本高；
- 理论模型与实际 DCA 结构之间有明显抽象距离；
- “information dilution”主要由 toy task和结果间接支持；
- loss-spike 稳定性缺少统计定义和机制诊断；
- 参数 overhead 很小，但 full-history activation / traffic 并不小；
- related-work comparison 由作者重实现，方法间调参预算不透明；
- 没有完整 downstream LM benchmark；
- 没有真实 TP/PP/CP 或 serving stack 结果。

---

## 10. 与 DenseFormer、MUDDFormer、AttnRes 的边界

### 对 DenseFormer

DenseFormer：

$$
\alpha_{l,j}\quad\text{是 source-depth scalar，静态。}
$$

DCA：

$$
b_{l,d,j}
+
\operatorname{ReLU}(w_l^\top X_j)
\quad\text{同时含 feature-wise static 与 token-dependent depth term。}
$$

### 对 MUDDFormer

两者都分开构造 Q/K/V 的 depth mixtures。MUDDFormer 还加入 residual stream $R$ 第四路，并使用小 MLP 从当前 hidden state 一次生成各路、各 source 的 token-wise 权重。

### 对 AttnRes

AttnRes 更接近规范的 depth attention：

$$
\operatorname{softmax}(q^\top K_{\text{depth}})V_{\text{depth}}.
$$

DCA 的动态权重没有 depth softmax 归一化，且最终权重含 feature-wise bias。二者都可称 depth-wise selection，但 numerical contract 不同。

---

## 11. 最终评价

### 可信度

- **论文身份**：高；ICML 2025 正式发表。
- **作者/机构**：UCI、USC、Google Research，身份清楚。
- **理论严谨度**：stylized setting 内较强，外推需要克制。
- **实验广度**：中等；两种 LM 数据、规模/宽度/深度 sweep、ViT。
- **复现性**：方法公式充分，但缺少已识别的官方代码。
- **大规模系统证据**：有限。

### 一句话定位

> DCA 把“历史层选择”推进到 feature-aware、input-dependent、Q/K/V 分路；它的重要性在于揭示多路 depth routing 的价值，而不是已经以理论证明了完整 Transformer 的最优性。
