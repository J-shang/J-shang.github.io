---
title: "Looped Transformers are Better at Learning Learning Algorithms：逐篇解析"
description: "分析 input injection、truncated loss 与 loop schedule 如何诱导逐步更新，并核查训练范围外的稳定性。"
topic: "looped-transformer"
section: "core"
slug: "looped-transformers-learning-algorithms"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
featured: true
order: 21
source:
  repository: "local/looped-transformer"
  path: "papers/05-looped-transformers-learning-algorithms.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-07-29"
  contentHash: "sha256:f6725dc501db8da243b75307f8df2a78aba943b533a933fdee929bbf69906fc8"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份

- 论文：*Looped Transformers are Better at Learning Learning Algorithms*
- 作者：Liu Yang、Kangwook Lee、Robert D. Nowak、Dimitris Papailiopoulos
- 版本：arXiv:2311.12424v3，2024-03-16
- 发表状态：ICLR 2024
- 主来源：[arXiv](https://arxiv.org/abs/2311.12424)、[HTML 全文](https://arxiv.org/html/2311.12424)
- 官方实现：[Leiay/looped_transformer](https://github.com/Leiay/looped_transformer)
- 阅读范围：正文、Figure 1–9、主表与 Appendix B–G；代码仅作为复现入口
- 信息截止：2026-07-24

## 30 秒结论

**[论文报告]** 作者把一个浅层 GPT-2 decoder block 重复执行，用每步 input injection、后段 loop loss 和 loop-depth schedule 训练 in-context function learning。在线性回归等任务上，looped 模型能以少于普通 12 层 Transformer 10% 的独立参数达到相近或更好表现，并在合适训练下呈现迭代收敛曲线。

**[综合判断]** 这是从“可编程构造”走向“从数据学会迭代”的关键实证论文。最重要的证据不是最终 MSE，而是：更多 loop 是否继续改进、训练范围外是否稳定、OOD 时是否仍像真实算法。

## 5 分钟论文地图

1. §3：in-context function learning 任务。
2. §4、Equation 1：looped 架构与 truncated loss。
3. §4.1、Figure 3：input injection。
4. §4.2、Figure 4：最大 loop $b$ 与 loss window $T$。
5. §5.2、Figure 2、5、6：线性回归、样本效率和 OOD。
6. §5.3、Figure 7–8：层数、head、width 与 probing。
7. §5.4、Figure 9 等：更复杂函数类。
8. Appendices B–G：loop 选择、seed/数据量、外推、schedule、OpenML。

前置知识：in-context learning、least squares、gradient descent、GPT-2 decoder block。最小例子是在线性回归中从 $w_0=0$ 开始，每轮用同一个梯度规则修正 $w_t$，而不是用不同规则一步得到答案。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $x_i,y_i$ | 第 $i$ 个 in-context 输入与标签 | $x_i\in\mathbb{R}^d,y_i\in\mathbb{R}$ | sampled data |
| $x$ | 整个 prompt 的固定 embedding | sequence tensor，shape 由 tokenization/width 决定 | runtime input |
| $z_t$ | 第 $t$ 个 loop 的 hidden state | 与 $x$ 同主 shape | activation |
| $f_\theta$ | 共享 GPT-2 block | sequence-to-sequence map | $\theta$ 可训练 |
| $b$ | 一次训练展开的最大 loop 数 | 正整数 | schedule/hyperparameter |
| $T$ | 参与 loss 的最后状态数 | $1\le T\le b$ | hyperparameter |
| $\hat y^{(t)}$ | 第 $t$ 步 query prediction | 标量或 batch 向量 | activation |
| $\ell$ | 单步 prediction loss | 标量，例如 squared error | 固定损失函数 |

不要混淆：论文的 $T$ 是 loss window，不是总 loop；本笔记用 $b$ 表示最大 loop。输入 $x$ 每步重复注入但不是可训练参数。

## 贡献账本与论证链

```text
已知 looped Transformer 能被手工编程
  → 问随机初始化能否从任务 loss 学到迭代
  → 加 input injection、后段 loop loss、深度 schedule
  → 检查更多 loops 是否继续改进及 OOD 是否像算法
  → 得到“可学习但分布相关”的实证结论
```

| 可检查贡献 | 类型与相对基线的增量 | 支持位置 | 没有建立 |
|---|---|---|---|
| 面向迭代学习的 loop 训练配方 | 机制组合 | §4、Eq. 1、Figure 3–4 | 每个组件对所有任务都必要 |
| 少独立参数匹配深 untied 模型 | 经验发现 | Figure 2/5 | FLOPs 同比例下降 |
| 循环轨迹呈逐步改进 | 经验发现 | Figure 2/8 与附录曲线 | 内部严格执行某个已知算法 |
| OOD 成功与失败边界 | 负面/诊断证据 | Figure 6 | 分布无关通用求解器 |

## 问题设定

一个 prompt 包含若干 in-context 样本：

$$
(x_1,y_1),\ldots,(x_n,y_n),x_{\text{query}},
$$

这里 $x_i,x_{\text{query}}\in\mathbb{R}^d$ 是输入向量，$y_i=f(x_i)$ 是 label；整段序列是 runtime data，模型参数不在 prompt 中。目标是预测 $f(x_{\text{query}})$。在线性回归中，一次性求 least-squares 涉及矩阵逆，而 gradient descent 可通过简单更新反复逼近。作者假设共享 block 更适合学习这种“同一更新规则多次调用”的结构。

![Learned Transformer solver versus a looped iterative solver](/assets/looped-transformer/05-looped-transformers-learning-algorithms/figure-1-iterative-solver.png)

*原图：Figure 1，PDF p. 2；来源：arXiv:2311.12424v3。看图重点：普通 Transformer 被画成一次性 learned solver，looped Transformer 则重复同一 block，结构上对应 gradient descent 这类迭代算法。它说明研究假设和比较对象，不是“模型内部已经被证明等于 GD”的证据。*

## 架构与训练机制

### 1. Looped state 与 input injection

基础 recurrence 可写为：

$$
z_{t+1}=f_\theta(z_t+x),
$$

其中 $z_t$ 与 $x$ 是同形或可相加的 activation，$f_\theta$ 的 $\theta$ 是所有 $t$ 共享的可训练参数。每步重新注入固定的 prompt embedding $x$，确保固定点仍依赖问题输入。

**两步 trace。** 从 $z_0=0$ 出发，$z_1=f_\theta(x)$，$z_2=f_\theta(f_\theta(x)+x)$。第二步既看到第一步结果，也重新看到原问题；无 injection 时第二步只有 $f_\theta(f_\theta(x))$，输入信息只能靠 hidden state 保存。

**[论文报告]** Figure 3 显示，不注入输入的普通 weight tying 在超过训练 loop 后快速恶化；input injection 版本更稳定（§4.1）。

### 2. Truncated loss window

模型执行最多 $b$ 次 loop，但只对最后 $T$ 个状态的预测计算 loss：

$$
\mathcal L(\theta)
=\frac{1}{T}
\sum_{t=b-T+1}^{b}
\ell\!\left(\hat y^{(t)},y\right).
$$

这里 $b$ 是最大 loop，$T$ 是末尾受监督步数，$\hat y^{(t)}$ 是从 $z_t$ 读出的预测，$y$ 是 query label；只有 $\theta$ 和 prediction head 参数被 optimizer 更新。**[复原推导]** 如果只监督最后一步，早期 transition 可能学到脆弱捷径；如果监督所有步，又会强迫前几步过早达到答案，并增加内存。后段 window 在二者之间折中。

### 3. Scheduled training

作者逐渐增加训练 loop，使深展开优化有一个较容易的起点。论文强调这更像 kick-start，而非严格意义上从简单样本到困难样本的 curriculum。

![Loop-count extrapolation under different training schedules](/assets/looped-transformer/05-looped-transformers-learning-algorithms/figure-4-loop-extrapolation.png)

*原图：Figure 4，PDF p. 5；来源：arXiv:2311.12424v3。看图重点：虚线是训练时最大 loop 数 $b$，实线继续画到更长推理 loops；只有部分 $b,T$ 组合在训练范围外保持低误差，另一些会回升或发散。因此“共享 block”本身不保证无限迭代稳定，训练窗口与 schedule 是结论的一部分。*

## 实验证据：问题—结果—边界

| 实验问题 | 设置与观察 | 支持结论 | 剩余不确定性 |
|---|---|---|---|
| 共享 block 能否学出逐步回归？ | 训练 30 loops 的 1-layer looped 模型随推理 loops 收敛，匹配 12-layer baseline 并接近 least squares；独立 block 参数约 $1/12$（Figure 2） | 随机初始化可学到迭代式行为 | block calls/FLOPs 并未减少到 $1/12$ |
| 参数共享会怎样影响样本效率？ | 有限 distinct prompts/functions 时，looped 模型用更少独立训练函数学好（Figure 5） | 共享带来有用归纳偏置 | 设置较合成，和自然预训练不同 |
| 行为是否 OOD 算法化？ | skew covariance/noise 下有时像轻度正则；input scale 大变时可能更差（Figure 6） | 不是分布无关线性求解器 | 仍不知具体学到了哪种更新 |
| hidden state 是否逐步细化？ | probe 在 looped 模型中随迭代持续改善（Figure 8） | 与 iterative refinement 相容 | probe 可读性不是因果使用证明 |
| 是否适用于其他函数族？ | sparse linear、tree、2-layer ReLU、OpenML 结果见 §5.4/附录 | 现象不限普通线性回归 | 最佳 $b,T$ 随任务改变 |

**[综合判断]** probe 只说明信息可被辅助 MLP 读出，不等同于模型因果地使用了该变量。但“随 loop 单调细化”与迭代解释相容。

### Claim–evidence map

| Claim | 证据 | 强度 | Gap |
|---|---|---|---|
| looped 模型更容易学习 learning algorithms | Figure 2/5/9 与附录 | moderate–strong（实验域内） | “算法”身份没有被唯一识别 |
| input injection 改善长循环稳定性 | Figure 3 | moderate | 与其他稳定化手段未全面比较 |
| OOD 不是普遍成功 | Figure 6 | strong（反例） | 失败机制仍未定位 |

## 贡献分级

- **表达能力**：不是主贡献，已有前作覆盖。
- **经验可学习性**：强；从随机初始化训练并观察循环轨迹。
- **优化定理**：无；不能保证一般 SGD 一定学到算法。
- **OOD 算法性**：有限；尺度变化提供了明确反例。

## 局限与易错解读

- 少参数不代表少 FLOPs：1 层 × 12 loops 与 12 层 baseline 的 block 执行量接近。
- “固定点”主要按输出曲线观察，并非 DEQ 的严格 equilibrium 求解。
- $b,T$、输入注入和 schedule 都是关键设计，不能只复现“共享一层”。
- 合成 function-class learning 与自然语言预训练之间仍有巨大尺度与分布差距。
- OOD 结果说明模型学到的是训练分布中的迭代偏置，不是完全等价于数值求解器。

## 超出论文：识别“学到哪种算法”

**[扩展假设] Proposal：** 用同一训练分布下预测不同但 in-distribution 表现接近的候选更新规则，设计干预点区分它们。

- Reasoning chain：最终 MSE 无法区分 GD、preconditioned GD 与直接 shrinkage；构造协方差谱和尺度干预可使轨迹预测分叉。
- Predicted observation：若近似 GD，误差在协方差特征方向上的衰减率应与特征值相关。
- Falsification condition：任何固定 step-size/preconditioner 都无法解释多步轨迹。
- Minimum experiment：记录每 loop 的 weight probe，在对角 covariance 的多个 eigenvalue 上拟合候选更新；使用 held-out intervention 验证。
- Cost/risk：probe 误差会污染算法识别，应同时做 activation patching 或 causal intervention。

## 推荐复现矩阵

| 变量 | 建议取值 |
|---|---|
| 架构 | 12-layer untied；1-layer looped；2-layer block looped |
| 注入 | 无；additive input injection |
| loss | final-only；all-step；last-$T$ |
| 推理 loop | 训练内、训练边界、2×训练上限 |
| OOD | covariance、noise、input scale |
| 报告 | MSE-loop 曲线、state delta、梯度范数、参数/FLOPs/显存 |

## 阅读后应回答

1. 为什么固定点若没有输入注入，可能逐渐忘记原问题？
2. $b$ 与 $T$ 分别控制计算深度和监督/显存，为什么不能混为一个量？
3. 哪些曲线支持“迭代”，哪些 OOD 结果反驳“已学到通用算法”？

## 一句话带走

**共享 block 确实能从数据中学出逐步改进的行为，但这种行为依赖专门训练机制，并且首先是训练分布内的迭代偏置，而非无条件的通用求解器。**
