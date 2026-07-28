---
title: "DenseFormer：用静态 Depth-Weighted Average 重写跨层信息流"
description: "从标准 residual sum 出发理解静态 Depth-Weighted Average，并核查 dilation、period、参数成本与中等规模语言模型证据。"
topic: "residual"
section: "methods"
slug: "denseformer"
date: 2026-07-27
updated: 2026-07-27
cutoff: 2026-07-27
order: 14
readtime: 16
source:
  repository: "J-shang/residual"
  path: "papers/05-denseformer.md"
  url: "https://github.com/J-shang/residual/blob/c55583707ab645bef7408c588fc33cbf91b809a5/papers/05-denseformer.md"
  revision: "c55583707ab645bef7408c588fc33cbf91b809a5"
  syncedAt: "2026-07-28"
  contentHash: "sha256:47fbd79ea1751931aa556103ee97a320361d30b0c77a7a130bb477a98a4be43b"
  manifest: "residual"
  managed: true
---
<!-- paper-order: 05 -->

> **论文**：Matteo Pagliardini、Amirkeivan Mohtashami、Francois Fleuret、Martin Jaggi, *DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging*<br>
> **机构**：EPFL、University of Geneva<br>
> **版本**：NeurIPS 2024 conference paper；对应 arXiv:2402.02622<br>
> **状态**：NeurIPS 2024 正式发表，不是 Independent Researcher 投稿<br>
> **主来源**：[NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f67449c7ab72f441d3a713b046c6818c-Abstract-Conference.html) · [正式 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/f67449c7ab72f441d3a713b046c6818c-Paper-Conference.pdf) · [arXiv](https://arxiv.org/abs/2402.02622)<br>
> **官方 artifact**：[epfml/DenseFormer](https://github.com/epfml/DenseFormer)；论文附录也给出 naive 与优化版 PyTorch 伪实现<br>
> **阅读范围**：正式 PDF 全文 30 页，包括正文、实现、超参数、补充实验和 NeurIPS checklist<br>
> **信息核对日期**：2026-07-27<br>
> **建议前置**：[路线图中的 residual / identity mapping 阶段](/topics/residual/)

## 证据标签

- **[论文报告]**：论文正文、公式、表格或附录直接报告。
- **[复原推导]**：由论文定义可以直接推出，但原文没有逐句写出的结论。
- **[综合判断]**：结合公式、实验和系统约束形成的解释。
- **[扩展假设]**：论文之外、需要新实验验证的想法。
- **[待验证]**：现有证据不足，不能当作既定事实。

---

## 先给结论

### 30 秒版

标准 residual stream 把历史更新压进一条累计状态：

$$
x_i = x_0 + \sum_{j=1}^{i} \Delta_j.
$$

后续 block 能看到累计和，却不能直接选择“我要第 3 层而不是第 17 层的表示”。DenseFormer 保存各层 block output，并在每个 block 后加入一个静态、可训练的 Depth-Weighted Average（DWA）：

$$
Y_i=\sum_{j=0}^{i}\alpha_{i,j}X_j,
\qquad
X_i=B_i(Y_{i-1}).
$$

它的关键特点是：

- mixing 沿 **depth** 发生，不沿 token 维发生；
- 每个深度对所有 token、所有 feature 共用同一组标量；
- 权重不经过 softmax，因此可以为负，也不要求和为 1；
- 初始化为 $\alpha_{i,i}=1$、其余为 $0$，所以初始函数精确恢复原 Transformer；
- 参数增量只有 $d(d+3)/2$，但 activation 读取和 memory traffic 并不为零。

DenseFormer 最适合作为跨层方法的“最小基线”：它已经证明静态 depth mixing 有价值，但还没有 input-dependent routing、Q/K/V 分流或多 residual streams。

### 最值得记住的实验

**[论文报告]**

- OpenWebText2、48 blocks：
  - Transformer：PPL 18.61，inference 5.94 batches/s；
  - full 1×1 DenseFormer：PPL 17.84，4.65 batches/s；
  - sparse 4×5 DenseFormer：PPL 17.87，5.72 batches/s。
- 48-block 4×5 DenseFormer 的 PPL 与 72-block Transformer（17.82）接近，但论文测得推理快约 $1.4\times$。
- 同为 72 blocks，full DenseFormer 从 Transformer 的 17.82 改善到 17.12。
- 相同约 8 小时训练预算，48-block Transformer 为 18.33，4×5 DenseFormer 为 17.87。
- PG-19 上，24/48-block DenseFormer 分别从 20.13/18.94 改善到 19.60/18.43。

### 我的总判断

**[综合判断]** 这是一篇可信、设计干净、对照较扎实的架构论文。其最可靠结论是：

> 在论文测试的中等规模 decoder-only LM 上，允许 block 直接线性复用历史表示，确实优于只让信息沿单条累计 residual stream 传播。

不应扩大成：

> DenseFormer 已被证明是大规模 LLM 的通用、更快替代品。

论文最大模型约 676M 参数，主要数据集是 OpenWebText2，训练使用最多 4 张 A100；它没有验证数十亿到数百亿参数、多轴并行、长上下文或现代 fused stack。

---

## 5 分钟结构图

| 层面 | DenseFormer 的答案 |
|---|---|
| motivating pressure | 深度增加收益递减；累计 residual state 不便直接复用远层表示 |
| stored state | embedding 与所有历史 block outputs：$\{X_0,\ldots,X_i\}$ |
| read rule | $B_i$ 读取上一次 DWA 输出 $Y_{i-1}$ |
| write rule | 当前 block 产生 $X_i$，追加到 history |
| state update | $Y_i=\sum_j\alpha_{i,j}X_j$ |
| 权重粒度 | per-destination-depth × per-source-depth scalar |
| 是否 input-dependent | 否 |
| normalization | 无 softmax；允许负权重和任意总和 |
| 初始化 | 当前层系数 1，其余 0；精确恢复 Transformer |
| 稀疏化 | dilation $k$ 与 period $p$ |
| 深层组合性质 | 没有 stochastic / norm-preserving 结构保证 |
| 参数成本 | $d(d+3)/2$ |
| 主要系统成本 | 对历史 `[B,T,D]` activations 的读取、乘加与数据移动 |

### 贡献账本

| 类型 | 贡献 | 新颖性与证据 |
|---|---|---|
| conceptual | 把 residual history 的“可寻址复用”作为独立设计轴 | 受 DenseNet 启发；在 decoder-only Transformer 上系统化 |
| methodological | 在每个 block 后加入静态 DWA，并提出 dilation / period | 核心新方法 |
| empirical | OWT2 多深度、同时间效率、PG-19、3 seeds | 本文最强证据 |
| interpretive | learned-weight heatmap 与 embedding cosine analysis | 支持性分析，不是因果证明 |
| systems | 优化版 history buffer / autograd 实现与速度测试 | 中型单机/DP 环境证据 |
| artifact | 官方 PyTorch 代码与附录伪实现 | 有助复现，但 OWT2 数据获取受限 |

---

## 1. 论文到底想解决什么

### 1.1 标准 residual 并非“没有跨层连接”

标准 residual recursion 可写为：

$$
z_i=z_{i-1}+\Delta_i(z_{i-1}).
$$

展开后：

$$
z_i=z_0+\sum_{j=1}^{i}\Delta_j.
$$

因此所有过去增量都能影响当前层。DenseFormer 的批评不是“Transformer 看不到历史”，而是：

> 历史贡献在进入后续层前已经被固定系数累加；后续层不能低成本地重新选择某个具体深度的完整表示。

这一区分很重要。DenseFormer 增加的是 **addressability**，而不是从零创造跨层依赖。

### 1.2 最小例子

三层标准链：

$$
X_1=B_1(X_0),\quad
X_2=B_2(X_1),\quad
X_3=B_3(X_2).
$$

第三层只显式接收 $X_2$。DenseFormer 改成：

$$
\begin{aligned}
Y_1 &= \alpha_{1,0}X_0+\alpha_{1,1}X_1,\\
X_2 &= B_2(Y_1),\\
Y_2 &= \alpha_{2,0}X_0+\alpha_{2,1}X_1+\alpha_{2,2}X_2,\\
X_3 &= B_3(Y_2).
\end{aligned}
$$

现在 $B_3$ 的输入可以直接重用、抵消或放大 $X_0$、$X_1$、$X_2$。

---

## 2. 符号、shape 与参数身份

论文公式省略 batch 和 sequence 维。实现时可固定：

$$
X_i,Y_i\in\mathbb R^{B\times T\times D}.
$$

| 符号 | 含义 | shape / 类型 | 身份 |
|---|---|---:|---|
| $d$ | Transformer block 数 | integer | 架构超参数 |
| $B_i$ | 第 $i$ 个 Transformer block | `[B,T,D] -> [B,T,D]` | 普通模型参数 |
| $X_0$ | token embedding output | `[B,T,D]` | 运行时张量 |
| $X_i$ | 第 $i$ 个 block 的原始输出 | `[B,T,D]` | 保存的历史状态 |
| $Y_i$ | 第 $i$ 个 DWA 输出 | `[B,T,D]` | 下一 block 的输入 |
| $\alpha_{i,j}$ | destination $i$ 对 source $j$ 的权重 | scalar | 可训练参数 |
| $k$ | DWA dilation | integer | 稀疏化超参数 |
| $p$ | DWA period | integer | 稀疏化超参数 |

需要特别避免两个混淆：

1. $X_i$ 是 block 原始输出，$Y_i$ 才是 mixing 后状态；
2. “weighted average”只是论文命名，$\alpha$ 不受 simplex 约束，数学上是一般线性组合。

---

## 3. 方法复原

### 3.1 Full DenseFormer

论文定义：

$$
\begin{aligned}
X_0&=\operatorname{Embedding}(X),\\
Y_0&=X_0,\\
X_i&=B_i(Y_{i-1}),\\
Y_i&=\operatorname{DWA}_i(X_0,\ldots,X_i)
=\sum_{j=0}^{i}\alpha_{i,j}X_j,\\
\operatorname{DenseFormer}(X)&=Y_d.
\end{aligned}
$$

如果把所有 $\alpha_{i,j}$ 排成下三角矩阵，depth mixing topology 一目了然。第 $i$ 行决定第 $i$ 个 DWA 从哪些过去 depth 读取。

### 3.2 初始化为何是精确 special case

令：

$$
\alpha_{i,j}=
\begin{cases}
1,&j=i,\\
0,&j\ne i.
\end{cases}
$$

则：

$$
Y_i=X_i,
$$

递归立即退化为：

$$
X_i=B_i(X_{i-1}).
$$

**[复原推导]** 这不是近似 identity，也不是“期望上相同”，而是在无 dropout 随机差异等外部因素时逐元素函数相等。因此 DenseFormer 可以从标准 Transformer function-preserving 地开始训练。

### 3.3 参数量

深度 $i$ 的 DWA 有 $i+1$ 个 scalar。总增量：

$$
\sum_{i=1}^{d}(i+1)
=
\frac{d(d+3)}{2}.
$$

例如 $d=72$：

$$
\frac{72\cdot75}{2}=2700
$$

个 scalar。参数量确实可忽略。

但“参数少”不等于“算子免费”。full DWA 的 elementwise multiply-add 量近似：

$$
O\left(BTD\sum_{i=1}^{d}(i+1)\right)
=O(BTDd^2).
$$

它通常不如 Attention/MLP matmul 的 FLOPs 大，却可能受 memory bandwidth 和 kernel launch 支配。

### 3.4 Dilated DenseFormer

dilation $k$ 只保留与当前 depth 同余的历史：

$$
\mathcal S_i^{(k)}
=
\{j\mid 0\le j\le i,\ j\equiv i\pmod{k}\}.
$$

于是：

$$
Y_i
=
\sum_{j\in\mathcal S_i^{(k)}}\alpha_{i,j}X_j.
$$

连接数和 DWA 计算约缩小 $1/k$。

这不是“只看最近 $k$ 层”。它保留跨越整个网络的等间隔长程连接。

### 3.5 Periodic DenseFormer

period $p$ 表示只在每 $p$ 个 block 后执行 DWA：

$$
Y_i=
\begin{cases}
\operatorname{DWA}_i(\{X_j:j\in\mathcal S_i^{(k)}\}),& p\mid i,\\
X_i,&p\nmid i.
\end{cases}
$$

论文把组合记为 `k×p-DenseFormer`：

- `1×1`：full DenseFormer；
- `4×1`：每层做 DWA，但只连每 4 层；
- `4×5`：每 5 层做一次，并在该 DWA 中使用 dilation 4。

总 DWA density 约为 full 版本的 $1/(kp)$。

---

## 4. 它与普通 residual 的真实关系

### 4.1 DenseFormer 保存的是 block outputs，不是多条 residual streams

DenseFormer 的 history 维是不断增长的 depth list：

$$
\{X_0,X_1,\ldots,X_i\}.
$$

HC / mHC 保存的是固定 $N$ 条并行 streams：

$$
S_i\in\mathbb R^{B\times T\times N\times D}.
$$

前者的 state count 随 depth 增长；后者的 stream count 固定。这会导致不同的 activation、pipeline 和 kernel 设计。

### 4.2 静态标量仍然可以做非平凡重参数化

每个 $\alpha_{i,j}$ 对所有 token 和 feature 共用，但不同 destination depth 有不同权重。因此它能学习：

- 直接复用早层 feature；
- 对某些历史层做负向抵消；
- 跳过中间层；
- 改变不同深度的有效贡献尺度。

它不能学习：

- token A 读取第 3 层、token B 读取第 8 层；
- Q、K、V 各自选择不同 history；
- feature channel 级别的独立 depth routing。

这些正是 DCA 与 MUDDFormer 的后续扩展方向。

### 4.3 权重无归一化约束

论文明确允许负权重，并观察到有意义的负值。因而：

$$
\sum_j\alpha_{i,j}\ne1
$$

完全允许。

**[综合判断]** 好处是表达力与“删除/抵消”能力；代价是没有凸组合、row-stochastic 或 operator-norm 的结构性稳定保证。深层组合是否稳定主要靠初始化、优化和 surrounding Transformer normalization，而非 DWA 本身的数学约束。

---

## 5. 实验逐项审计

### 5.1 主设置

**[论文报告]**

- 主数据：OpenWebText2，约 17B tokens；
- 补充数据：PG-19；
- decoder-only LM；
- hidden size $D=512$，8 heads，每头 64 维；
- sequence length 256；
- 主 batch：400 sequences；
- AdamW，$\beta_1=0.9,\beta_2=0.95$，weight decay 0.1；
- max learning rate 0.001，cosine decay，2000 warmup steps；
- 主实验 40k steps；
- A100 80GB，最多 4 GPU，主要采用 data parallel；
- 主表 3 个随机种子，报告均值与 standard error。

这套设置对“方法在中型 LM 上有效”是不错的证据，但不代表现代 frontier LLM recipe。

### 5.2 同深度和同推理速度

| 模型 | depth | PPL ↓ | inference BPS ↑ |
|---|---:|---:|---:|
| Transformer | 48 | 18.61 | 5.94 |
| Skips with Gains | 48 | 18.45 | 5.72 |
| DenseFormer 1×1 | 48 | 17.84 | 4.65 |
| DenseFormer 4×1 | 48 | 17.86 | 5.31 |
| DenseFormer 4×5 | 48 | 17.87 | 5.72 |
| Transformer | 72 | 17.82 | 4.08 |
| DenseFormer 1×1 | 72 | 17.12 | 2.93 |
| DenseFormer 4×5 | 72 | 17.21 | 3.90 |

**支持什么：**

- 同 depth 下 DWA 明显改善 PPL；
- 4×5 保留了 full DWA 的大部分收益；
- 简单给现有 skip 加 scalar gain 不足以解释全部收益；
- 浅 DenseFormer 可在相似 PPL 下快于更深 Transformer。

**不支持什么：**

- 没有证明任何模型规模都保持同样 gap；
- 没有证明真实 serving stack 的 batch-1 decode、prefill、KV cache 或 quantization 下仍有相同比例；
- BPS 依赖具体硬件、batch、实现和 kernel。

### 5.3 相同训练时间

论文让 Transformer 多训练 1500 steps，以补偿 DenseFormer 的每步 overhead：

| 模型 | steps | wall time | PPL ↓ |
|---|---:|---:|---:|
| 48-layer Transformer | 41,500 | 8.09 h | 18.33 |
| 48-layer 4×5 DenseFormer | 40,000 | 8.04 h | 17.87 |

**[综合判断]** 这是比“相同步数”更有价值的效率对照。不过它仍是一次特定规模与实现上的 wall-clock 结果，不等价于 scaling-law 级 compute efficiency 证明。

### 5.4 PG-19 外部数据验证

| 模型 | depth | PPL ↓ |
|---|---:|---:|
| Transformer | 24 | 20.13 |
| DenseFormer 1×1 | 24 | 19.60 |
| Transformer | 48 | 18.94 |
| DenseFormer 1×1 | 48 | 18.43 |
| Transformer | 72 | 18.44 |

这说明效果不只出现在 OpenWebText2，但仍局限于自回归语言建模。

### 5.5 learned weights 告诉了我们什么

论文在 48/72 层、3 个 seeds 上观察到较稳定的 DWA pattern：

- 当前层附近通常权重大；
- input embedding 和远层仍被使用；
- 某些连接为负；
- dilation 后仍出现相似长程结构；
- “只连最近 $K$ 层”或“只在最后做一次聚合”都弱于分布在网络中的 DWA。

**[论文报告]** 这些图支持“模型确实使用跨层连接”。

**[综合判断]** 它们不证明每条连接具有明确语义，也不证明权重绝对值等价于因果重要性。activation norm、后续非线性和连接间协同都会影响真实贡献。

---

## 6. Claim–evidence 对照

| 论文主张 | 主要证据 | 支持强度 | 主要缺口 |
|---|---|---|---|
| DWA 改善同深度 LM 质量 | OWT2 多深度、3 seeds | 强，限本文规模 | 未到 billion-scale |
| 比单纯 residual gain 更有效 | 48-layer `Skips with Gains` | 中等 | baseline 类型仍有限 |
| sparse DWA 保留大部分收益 | dilation / period sweep | 较强 | 最佳 $k,p$ 可能依 recipe 改变 |
| 同 PPL 下更快、更小 | 48 DenseFormer vs 72 Transformer | 中等 | wall-clock 与部署栈相关 |
| 更 data-efficient | 固定 steps/tokens 下更低 PPL | 中等 | “data efficiency”未做完整 token scaling fit |
| 长程 history 有价值 | learned weights 与 sparsity ablation | 中等 | 缺少 causal intervention |
| memory overhead negligible | 训练 activation 已保存、decode 仅当前 token | 偏弱 | checkpoint、PP、allocator 与 memory traffic 未充分测量 |

---

## 7. 系统与分布式实现含义

### 7.1 Activation bytes

若每层历史 activation 都以 $s$ bytes/element 保存，完整 history 的张量体积为：

$$
M_{\text{history}}
\approx
(d+1)BTD\,s.
$$

标准训练也常为 backward 保存 layer outputs，但在 activation checkpointing 下，标准 Transformer 可以选择只保存 checkpoint boundaries。DenseFormer 的 DWA 需要直接访问历史，可能改变最佳 checkpoint 粒度。

因此论文的“negligible memory overhead”应精确理解为：

> 在其保存策略和实现下，DWA 复用了许多本就为 backward 保留的张量。

它不是任意训练系统中的结构性零开销保证。

### 7.2 Tensor parallel

$\alpha_{i,j}$ 是 scalar，因此若 hidden dimension 按 TP 切分：

- 每个 rank 可对本地 $D/P$ shard 独立做相同 DWA；
- 不需要因为 DWA 本身新增 hidden-dimension all-reduce；
- scalar weights 应复制到各 rank。

前提是所有 source activations 采用相同 TP layout。

### 7.3 Sequence / context parallel

DWA 不跨 token mixing，所以 sequence shard 可本地计算。只要所有历史层保持相同 sequence ownership，DWA 本身不需要 token collective。

### 7.4 Pipeline parallel

这是最麻烦的轴。若 destination DWA 需要读取跨 PP stage 的任意历史 $X_j$，则必须：

1. 把历史 activations 一路携带过 stage boundary；
2. 在每个 stage 复制/缓存远层 activations；
3. 或限制 DWA 只连接 stage-local / block-local history。

Full DenseFormer 的 pipeline payload 可能从一个 `[B,T,D]` 累计状态变成多个历史 tensors。论文没有给出大型 PP 设计，这也是作者明确留下的 future work。

### 7.5 Inference

对自回归 decode 的当前 token，DWA 需要该 token 在各 depth 的当前表示。它不增加跨时间的 KV cache 维度，但增加：

- 当前 token 的 depth-history scratch；
- depth-wise elementwise reads；
- 若未融合，额外 kernel launches；
- 对 continuous batching / CUDA graph 的静态 buffer 设计要求。

参数 overhead 小，不能替代真实 TTFT、TPOT 和峰值显存测量。

---

## 8. CPU reference implementation 应该验证什么

一个 correctness-first 版本可以维护 Python list：

```python
history = [embedding(tokens)]
y = history[0]

for i, block in enumerate(blocks, start=1):
    x = block(y)
    history.append(x)
    sources = select_by_dilation(history, i, dilation)
    y = dwa[i](sources) if i % period == 0 else x
```

至少需要以下 tests：

1. **shape**：不同 `B,T,D,d,k,p` 都输出 `[B,T,D]`；
2. **exact baseline**：one-hot-current 初始化逐元素等于原 Transformer；
3. **gradient**：`float64` 小 tensor 做 `gradcheck`；
4. **negative weights**：确认实现没有误加 softmax；
5. **dilation index**：严格检查 $j\equiv i\pmod{k}$，避免从 list 尾部计数造成 off-by-one；
6. **period**：非 DWA 层必须令 $Y_i=X_i$；
7. **reference vs vectorized**：list 实现与 stacked/einsum 实现前后向一致；
8. **checkpoint**：若加入重计算，检查 history ownership 与 version counter。

---

## 9. 主要局限与可能误读

### 9.1 论文没有证明的内容

- 没有 billion-scale 或现代长上下文 LLM 主结果；
- 没有 downstream task、few-shot 或 instruction-following 评估；
- 没有理论保证 DWA 的深层 composite operator 稳定；
- 没有 TP/PP/CP/FSDP 的实测；
- 没有与后来出现的 DCA、MUDDFormer、HC/mHC、AttnRes 做同 recipe 比较；
- OpenWebText2 已不再像论文实验时那样容易公开获取，完整复现存在数据障碍。

### 9.2 “average”不表示凸组合

如果读者把 DWA 当 softmax attention，会错过负权重和总尺度变化。正确写法是：

$$
Y_i=\sum_j\alpha_{i,j}X_j,
\qquad \alpha_{i,j}\in\mathbb R.
$$

### 9.3 “memory negligible”要绑定实现条件

训练时是否已经保存全部 block outputs，取决于 checkpoint/recompute 策略。大型训练中，额外历史的 lifetime、PP 传输和 allocator fragmentation 都可能变得显著。

### 9.4 比更深模型快，不等于比同 FLOPs 模型总是好

论文展示的是特定 Pareto 点。更严格的架构比较还应固定：

- 参数；
- 训练 tokens；
- training FLOPs；
- wall-clock；
- inference batch / sequence；
- kernel maturity；
- validation 和 downstream protocol。

---

## 10. 与另外三篇的关系

| 方法 | history 范围 | weight 粒度 | input-dependent | Q/K/V 分流 | 低秩 residual transform |
|---|---|---|---|---|---|
| DenseFormer | 所有历史 block outputs | source-depth scalar | 否 | 否 | 否 |
| DCA | all 或 first+last-$k$ | depth × feature bias + token-dependent depth term | 是 | 是，Q/K/V 三路 | 否 |
| MUDDFormer | 所有历史 block outputs | token × source-depth × Q/K/V/R | 是 | 是，四路 | 否 |
| LAuReL | 当前或最近 $k$ 个状态 | layer scalar / low-rank linear map | 基本版本否 | 否 | 是 |

DenseFormer 是这条研究线最清楚的起点：

$$
\text{static depth scalar}
\rightarrow
\text{dynamic depth routing}
\rightarrow
\text{multiway dynamic routing}.
$$

---

## 11. 可检验的后续问题

### 11.1 [扩展假设] block-local DenseFormer 更适合 PP

只在每个 PP stage 内做 DWA，可避免跨 stage 携带完整 history。需要测：

- 质量相对 full DWA 的损失；
- stage-local history bytes；
- pipeline payload；
- 每 stage 的 kernel fusion 机会。

### 11.2 [扩展假设] 对 DWA composite matrix 加约束

可比较：

- unconstrained $\alpha$；
- row-softmax；
- signed normalized weights；
- row-stochastic / doubly stochastic depth mixing；
- spectral regularization。

目标是区分“负权重表达力”和“深层稳定性”之间的真实 trade-off。

### 11.3 [扩展假设] 用 learned pattern 做结构化剪枝

训练 full DWA 后，以 causal ablation 而非只看绝对值决定连接：

1. 单连接置零；
2. 分组置零；
3. 重新微调；
4. 测 PPL、downstream、wall-clock 和 history bytes。

这样才能判断稳定 heatmap 是否能转化为部署稀疏结构。

---

## 最终评价

### 可信度

- **论文身份**：高；NeurIPS 2024 正式发表。
- **机构信息**：清楚；EPFL 与 University of Geneva。
- **方法可复原性**：高；公式简单，附录给出实现与超参数，官方代码公开。
- **中型 LM 实验证据**：较强；多深度、效率对照、3 seeds、第二数据集。
- **大模型与分布式外推**：有限。

### 一句话定位

> DenseFormer 证明了“让每层重新线性选择历史表示”本身就有价值；它是静态、低参数、实现友好的跨 depth aggregation baseline，而不是已经解决动态选择和大规模系统问题的终点。
