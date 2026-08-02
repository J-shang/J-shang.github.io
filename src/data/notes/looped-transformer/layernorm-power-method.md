---
title: "Looped Transformers with Layer Normalization Provably Learn the Power Method：逐篇解析"
description: "分析受控线性模型中 normalization 如何选择 power method，并区分定理、有限样本实验与完整 LLM 外推。"
topic: "looped-transformer"
section: "frontiers"
slug: "layernorm-power-method"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
order: 40
source:
  repository: "J-shang/looped-transformer"
  path: "papers/11-layernorm-power-method.md"
  url: "https://github.com/J-shang/looped-transformer/blob/9ab82eeb3178ddd627b592ac2cba22de91e7be66/papers/11-layernorm-power-method.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-08-02"
  contentHash: "sha256:0ad6073009b3f98ecb1f5495ef685c78846e1b653ea8808faaba9c69ea3d72f6"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份与证据范围

- 论文：*Looped Transformers with Layer Normalization Provably Learn the Power Method*
- 作者：Lyumin Wu、Chenyang Zhang、Yuan Cao
- 分析版本：arXiv:2606.00605v1，2026-05-30
- 发表状态：arXiv preprint；截至信息截止日未用同行评审版本替换
- 主来源：[arXiv](https://arxiv.org/abs/2606.00605)、[HTML 全文](https://arxiv.org/html/2606.00605)、[PDF](https://arxiv.org/pdf/2606.00605)
- 阅读范围：正文 §1–6、Theorem 4.1/4.2/4.3/4.6/4.7、Figure 1–2、Appendix A–G 的证明结构与额外实验
- 信息截止：2026-07-24

## 30 秒结论

**[论文报告]** 在一个简化的 principal-component prediction 任务上，带 column-wise normalization 的 looped linear Transformer 经 population gradient descent 后，极限模型精确等价于执行固定 $L$ 步 power method；训练只监督最终 principal subspace，并未逐层提供 power-iteration target（Theorem 4.1）。在单层、逐层监督对照中，无 normalization 的模型仍有严格正的不可约 loss，而带 normalization 的模型能把 loss 推到任意小（Theorem 4.2–4.3）。

**[综合判断]** 论文的核心不是泛泛的“LayerNorm 稳定训练”，而是 normalization 改变了可学习更新规则的函数形式和优化极限。不过论文实际使用的是无均值中心化、结构上更接近 RMSNorm 的 column normalizer，结论不能直接外推到完整 LLM 的 LayerNorm/RMSNorm。

## 5 分钟论文地图

```text
现有算法学习理论常删掉 normalization
  → 选择天然含“乘矩阵后归一化”的 power method
  → 分析共享 linear-attention + residual + column normalization
  → 用对称性把全矩阵 gradient descent 降成低维 scalar dynamics
  → 证明极限模型为 L-step power method
  → 用无 norm 对照和有限样本实验检验机制
```

阅读顺序：

1. §3.1：PCA 任务与数据分布假设。
2. §3.2：被分析的 simplified Transformer。
3. Theorem 4.1：end-to-end looped training。
4. Theorem 4.2–4.3/4.6：有无 normalization 的可辨识差异。
5. §5、Figure 1–2：finite-sample sanity checks。
6. Appendix A：invariant manifold 与 scalar recursion。

前置知识：eigendecomposition、principal subspace、power method、rotation invariance、population loss。最小例子是 $\Sigma=\operatorname{diag}(4,1)$、$q_0=(1,1)/\sqrt2$：每次乘 $\Sigma$ 并归一化，第一个坐标相对第二个扩大 4 倍，方向快速靠近主特征向量。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $d,n$ | data dimension 与样本列数 | 正整数 | 超参数 |
| $X$ | 数据矩阵，列为 data vectors | $d\times n$ | sampled runtime input |
| $\Sigma=XX^\top$ | 未归一化 sample covariance/Gram operator | $d\times d$ 对称半正定 | 派生量 |
| $U_1$ | 最大特征值对应 principal subspace | $\mathbb{R}^d$ 的子空间 | 由 $X$ 派生 |
| $q_0$ | 初始 query | $d$ 维单位向量 | 随机 runtime input |
| $q_\ell$ | 第 $\ell$ 次 power iterate | $d$ 维单位向量 | runtime state |
| $L$ | looped layer/iteration 数 | 正整数 | 超参数 |
| $W,V$ | simplified attention 的合并参数矩阵 | block matrices | 可训练参数 |
| $\operatorname{Norm}$ | 逐列除以 $\ell_2$ norm 的算子 | column-wise map | 固定算子；RMSNorm-like |
| $\mathcal L$ | 输出到 principal subspace 的 population squared distance | 标量 | 训练目标 |
| $\eta$ | gradient-descent learning rate | 正标量 | 优化超参数 |

论文中训练 step 与 loop index 都可能用相邻字母表示；本笔记用 $r$ 表示 optimizer step、$\ell=0,\ldots,L$ 表示模型内 power/loop step。$X,W,V$ 的具体 augmented block shape 见 §3.2；核心算法状态 $q_\ell$ 始终在 $\mathbb{R}^d$。

## 贡献账本

| 可检查贡献 | 类型与最近基线增量 | 支持位置 | 没有建立 |
|---|---|---|---|
| end-to-end 训练选出 $L$-step power method | 训练动力学定理 | Theorem 4.1、Appendix A | 任意初始化/optimizer 都选同一算法 |
| 有无 normalization 的严格性能分离 | 理论对照 | Theorem 4.2–4.3/4.6 | 现实 LayerNorm 是唯一原因 |
| 对 input-dependent norm 的证明工具 | 理论工具 | Schur’s lemma、dominated convergence，Appendix A/C/F | 可直接复用于完整 softmax LLM |
| finite-sample SGD 现象与定理相容 | 数值实验 | §5、Figure 1–2、Appendix G | 经验图就是定理证明 |

## 方法与定理复原

### 1. Baseline algorithm：power method

对 $\Sigma=XX^\top$，power iteration 为：

$$
q_{\ell+1}
=\frac{\Sigma q_\ell}{\lVert\Sigma q_\ell\rVert_2},
\qquad \ell=0,\ldots,L-1.
$$

这里 $\Sigma\in\mathbb{R}^{d\times d}$ 是固定输入派生矩阵，$q_\ell\in\mathbb{R}^d$ 是 runtime state，分母是固定 $\ell_2$ normalization，不含 learned parameter。若 $q_0$ 在主子空间上投影非零且有 eigengap，方向误差按特征值比指数衰减。

**二维算例。** $\Sigma=\operatorname{diag}(4,1)$，$q_0=(1,1)/\sqrt2$。未归一化一步是 $(4,1)/\sqrt2$；归一化后方向为 $(4,1)/\sqrt{17}$。第二步方向比例变为 $16:1$。normalization 不改变方向，却防止 norm 按 $4^\ell$ 爆炸。

### 2. 被分析的 Transformer

论文把 $X$ 与独立随机单位 query $q_0$ 组成 augmented input；使用：

- linear self-attention，无 softmax；
- residual connection；
- 合并后的矩阵 $W,V$；
- 对每个 token/column 独立做 $\ell_2$ normalization；
- 相同 $W,V$ 跨 $L$ 层共享。

§3.2 明确说明，这个 normalizer 去掉标准 LayerNorm 的 mean centering，结构上等价于 RMSNorm（差一个标量常数）。

**Shape trace。** 数据列和 query 列进入同一 augmented matrix；attention 让 query column 读取 $X$ 形成的二阶统计；特定 block 结构下，数据 columns 保持为数据载体，最后 query block 按 $\Sigma q_\ell$ 更新；column normalization 把它重新映射到单位球。

### 3. 为什么最终 supervision 能选出 power method

训练目标测量最终输出到 principal subspace 的 squared distance，并对 $X,q_0$ 分布取期望（§4.1）：

$$
\mathcal L(W,V)
=\mathbb E_{X,q_0}\!
\left[\operatorname{dist}\!\left(
\widehat q_L(W,V;X,q_0),U_1(X)
\right)^2\right].
$$

这里 $\widehat q_L$ 是 $L$ 次共享 layer 后的 runtime output，$U_1(X)$ 是输入 $X$ 的 principal subspace，只有 $W,V$ 是 optimizer-owned parameters。监督没有提供中间 $q_1,\ldots,q_{L-1}$。

Theorem 4.1 依赖：

- $X$ 的每列单位 norm；
- 数据分布 rotation-invariant；
- 技术性 integrability 条件；
- 指定的稀疏 block initialization；
- 足够小的 gradient-descent step。

**证明主线（Appendix A）。**

1. 特定 $W,V$ block manifold 上，layer update 化成 normalized power iteration。
2. 由 rotation symmetry/Schur’s lemma，population gradient 不会激活其他 blocks；manifold 对 gradient descent 不变。
3. 全矩阵 dynamics 因而降成少量 scalar coefficients 的 recursion。
4. active coefficients 随训练增长，residual 中的 identity/初始项相对消失。
5. $r\to\infty$ 时，每层正好实现一次 power update。

![Learned parameter structure and coefficient growth](/assets/looped-transformer/11-layernorm-power-method/figure-1-parameter-structure.png)

*原图：Figure 1，PDF p. 12；来源：arXiv:2606.00605v1。看图重点：两组任务中，$W$ 与 $V$ 只有理论预测的对角 block 保持显著；右侧 log-log 曲线显示 active coefficients 近似按 $t^{1/4}$ 增长。它是 finite-sample SGD 与 population symmetry 分析的定性一致性证据；无误差条和单一小型设定意味着它不能替代定理。*

### 4. 正 limiting loss 不表示算法没学准

**[论文报告]** Theorem 4.1 的训练 loss 收敛到严格正值，但极限模型与 $L$-step power method 的输出差距为零（§4.1）。

**[复原推导]** 原因是有限 $L$ 的 power iterate 一般仍未完全落到 $U_1$；正 loss 是有限算法步数相对精确 PCA 的 approximation error，不是模型相对算法的 residual。

### 5. 有无 normalization 的可辨识对照

论文进一步直接监督单层去拟合“一步 power iteration”：

- 无 normalization：不论 target 使用 normalized 还是 unnormalized 版本，训练 loss 收敛到严格正值（Theorem 4.2）。
- 有 normalization：population loss 可在有限训练后任意接近 0（Theorem 4.3）。
- 推理时重复调用：两者 angular error 都可指数下降，但 normalized 模型的 contraction factor 随训练接近 power method 的最优特征值比；unnormalized 模型停在固定、次优因子（Theorem 4.6）。

这个对照比“加 norm loss 更低”更强，因为它给出了两个机制在训练极限下不同的可达算法类。

![Loss and repeated-loop angular error with and without normalization](/assets/looped-transformer/11-layernorm-power-method/figure-2-layernorm-loss.png)

*原图：Figure 2，PDF p. 13；来源：arXiv:2606.00605v1。看图重点：蓝线（有 normalization、normalized target）在 train/test loss 上继续下降，橙/绿线 plateau；右列重复调用时蓝线的角误差也衰减更快。图验证的是简化 linear Transformer 中 RMSNorm-like column normalization 的机制分离，不能直接归纳为完整 LLM 中标准 LayerNorm 的唯一作用。*

## 实验证据：问题—设置—结果—边界

| 实验问题 | 设置与观察 | 支持结论 | 剩余不确定性 |
|---|---|---|---|
| finite-sample SGD 是否复现结构动力学？ | 10,000 train / 2,000 test samples，2,000 epochs；Figure 1 中只有预测的 active matrix blocks 保持显著，系数呈理论预期的 polynomial trend（§5） | population 结构在有限样本下定性可见 | 主要是 heatmap/曲线，未给置信区间 |
| norm 是否消除一步算法误差？ | normalized model train/test loss 近 0；两种 unnormalized 模型 plateau 于正 loss（Figure 2） | 与 Theorem 4.2–4.3 的分离一致 | 合成数据、指定初始化 |
| 反复调用是否产生正确收敛率？ | Figure 2/Appendix G：normalized angular error 衰减快于 unnormalized | 与 Theorem 4.6 的 contraction-factor 预测一致 | 不是大规模任务 |
| OOD 是否只记住训练矩阵？ | Theorem 4.6/4.7 覆盖 inference-time unit-column matrices 和非正交初始 query | 学得更新具有一定分布外算法解释 | OOD 仍保留 unit-norm column 等结构 |

### Claim–evidence map

| Claim | 证据 | 强度 | Gap |
|---|---|---|---|
| end-to-end GD 选择 power method | Theorem 4.1 + Appendix A | strong（假设内） | 特殊 symmetry/init/population |
| normalization 改变 algorithmic implicit bias | Theorem 4.2–4.3/4.6 | strong（比较域内） | norm 与参数化共同限定函数类 |
| 现实 Transformer 的 LayerNorm 具有同作用 | 动机与结构类比 | weak | 无 softmax、MLP、mean centering、finite-data theory |
| finite-sample 结果支持理论机制 | Figure 1–2 | moderate | 小型合成、方差未报告 |

## 关键洞察

1. **[论文报告] normalization 是算法操作，不只是优化辅助。** Power method 每一步本来就需要重新归一化，结构吻合使共享 layer 能成为可重复更新。
2. **[复原推导] 参数发散与函数收敛可以并存。** active parameter scale 增大时，column normalization 消除共同尺度，模型输出趋于稳定算法。
3. **[综合判断] “LayerNorm 证明”应读成“RMSNorm-like column normalizer 证明”。** 论文自己明确没有 mean centering。
4. **[综合判断] 这是算法识别强于 benchmark correlation 的例子。** 定理不仅说 loss 下降，还唯一刻画了极限 update path。

## 局限与开放问题

作者明确承认（§6）：

- simplified linear Transformer；
- population-loss training；
- 尚未包含 MLP；
- finite-sample guarantee 留待未来。

进一步边界：

- 无 softmax、无现代多头/GQA/MoE；
- 特殊 rotation-invariant 数据与 unit-column 条件；
- 指定 block initialization 和 symmetry 对 proof 至关重要；
- principal-component prediction 与语言建模差异大；
- 数值实验未报告多 seed variance；
- “first theoretical analysis”是作者截至 v1 的定位性主张，应等待同行评审与后续文献核对。

## 超出论文：哪种 normalization 成分真正关键

**[扩展假设] Proposal：** 在同一 PCA task 上比较 pure $\ell_2$ norm、RMSNorm+learned gain、mean-centering LayerNorm、QK norm 与只做 weight normalization。

- Reasoning chain：原定理只需要按输入状态归一化；learned affine、mean subtraction 或仅规范参数可能改变 invariant manifold。
- Predicted observation：所有逐状态正尺度不变的 normalizer 都能近似 power update；只规范权重不能消除 iterate norm。
- Falsification condition：去掉状态归一化但保留适当参数重参数化后仍精确学到同一 update。
- Minimum experiment：相同 initialization/data/SGD，测一步 imitation loss、loop angular decay、有效 update 与 power iterate 的 cosine error。
- Cost/risk：不同 normalizer 的 learning-rate scale 不同，必须等额调参。

## 复现与阅读路径

1. §3.1 → 手算二维 power method。
2. §3.2 → 标出哪些是 parameter、input、query 和 normalizer。
3. Theorem 4.1 → Appendix A 四步证明图。
4. Theorem 4.2–4.3 → Figure 2，理解对照为何可辨识。
5. 最小复现必须记录每个 loop 的 angular error，而不仅是最终 subspace loss。
6. sanity checks：输出列 norm 恒为 1、active blocks 与理论结构匹配、finite-$L$ algorithm gap 与 model-to-algorithm gap 分开。

## 一句话带走

**在受控 PCA 世界中，normalization 把共享 linear-attention 层变成了可训练的 power-iteration 算子；它证明了 normalization 能决定学到哪种算法，但还不是完整 LLM 的普遍定理。**
