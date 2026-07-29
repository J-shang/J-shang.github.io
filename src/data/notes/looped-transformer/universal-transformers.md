---
title: "Universal Transformers：逐篇解析"
description: "复原共享 transition、depth recurrence 与 per-position ACT，理解 Looped Transformer 的直接架构前身。"
topic: "looped-transformer"
section: "foundations"
slug: "universal-transformers"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
featured: true
order: 11
source:
  repository: "local/looped-transformer"
  path: "papers/02-universal-transformers.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-07-29"
  contentHash: "sha256:4e73b252633ab11d51fc60dfdeb864857e44fb6c6d960b69f664b299312e2a05"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份

- 论文：*Universal Transformers*
- 作者：Mostafa Dehghani、Stephan Gouws、Oriol Vinyals、Jakob Uszkoreit、Łukasz Kaiser
- 版本：arXiv:1807.03819v3，2019-03-05
- 发表状态：ICLR 2019
- 主来源：[arXiv 摘要](https://arxiv.org/abs/1807.03819)、[HTML 全文](https://arxiv.org/html/1807.03819)
- 阅读范围：正文、Figure 1–4、主表、Appendix B/C/E；未把后续 looped 工作的结果归给本论文
- 信息截止：2026-07-24（用于与本项目其他论文建立关系）

## 30 秒结论

**[论文报告]** Universal Transformer（UT）把同一个 self-attention + transition block 沿 depth 反复应用；token 之间仍并行，depth step 之间递归。它还可用 ACT 让每个位置动态决定停止时间。

**[综合判断]** 这是现代 Looped Transformer 最直接的架构祖先。真正值得学习的不是“共享权重”四个字，而是：循环状态、输入/时间编码、每位置 halting 以及训练深度与推理深度之间的关系。

## 5 分钟论文地图

1. §1：为什么固定深度 Transformer 在算法外推上可能不足。
2. §2、Figure 2：UT 的 recurrent transition。
3. §2.1：动态 ACT。
4. §3：bAbI、subject–verb agreement、算法任务、翻译。
5. §4：与 RNN、Transformer、计算完备性的关系。
6. Appendix B：计算 universality 的条件。
7. Appendix C：ACT 算法。
8. Appendix E：bAbI 多随机种子结果。

前置知识：Transformer block、recurrence、halting probability。最小例子是：固定深度 Transformer 用 6 组参数做 6 次变换；UT 用同一组参数做 6 次变换，并让状态携带“当前是第几步”的信息。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $n,d$ | token 数与 hidden width | 正整数 | 输入决定 / 超参数 |
| $H^{(t)}$ | recurrent depth 第 $t$ 步的全序列状态 | $n\times d$，省略 batch 轴 | activation |
| $t$ | recurrent depth step | $1,\ldots,T$ | 运行时索引 |
| $F_\theta$ | 共享的 attention + transition block | $\mathbb{R}^{n\times d}\to\mathbb{R}^{n\times d}$ | $\theta$ 可训练 |
| $P_i^t$ | token position $i$ 与 step $t$ 的二维位置/时间信号 | $d$ 维向量 | 固定编码 |
| $p_i^t$ | 位置 $i$ 在 step $t$ 的 halting probability | $[0,1]$ 标量 | activation |
| $N_i$ | 位置 $i$ 的实际停止步 | 正整数 | 运行时派生量 |
| $\epsilon$ | ACT 剩余概率阈值 | 标量 | 超参数 |

token 为行、feature 为列；$i\in\{1,\ldots,n\}$，$t$ 沿深度而不是序列位置推进。

## 贡献账本与论证链

```text
固定深度、每层独立参数不天然适合反复算法
  → 把一个全局 self-attention transition 沿深度递归
  → 可选 ACT 为不同 token 分配不同步数
  → 测试算法任务、语言任务与翻译
  → 支持共享 recurrent depth 是可行归纳偏置
```

| 可检查贡献 | 类型与最小增量 | 支持位置 | 没有建立 |
|---|---|---|---|
| Universal Transformer | 机制：Transformer 内加入 depth recurrence 与共享参数 | §2、Figure 2 | 训练外无限步稳定 |
| per-position ACT | 条件计算机制 | §2.1、Appendix C | 同比例硬件加速 |
| 特定条件下 computational universality | 理论 | §4、Appendix B | 有限模型会自动学到任意程序 |
| 多任务经验改进 | 经验发现 | §3、各结果表 | 现代大模型规模仍保持同样优势 |

## 核心状态更新

论文 §2 将每个 depth step 写成“全序列并行 attention，再做逐位置 transition”：

$$
H^{(t+1)}
=\operatorname{Transition}\left(
\operatorname{SelfAttention}(H^{(t)})
\right).
$$

这里 $H^{(t)}\in\mathbb{R}^{n\times d}$ 是运行时状态；$\operatorname{SelfAttention}$ 与 $\operatorname{Transition}$ 内的参数属于同一个 $\theta$，跨 $t$ 共享。实际模型还含 residual 与 LayerNorm，并把 $P_i^t$ 加入位置 $i$ 的状态。这样，同一参数知道“我在处理哪个 token”和“这是第几次迭代”。

**三步 shape trace。** 若 batch 为 2、$n=5,d=16$，则 $H^{(0)},H^{(1)},H^{(2)},H^{(3)}$ 都是 $2\times5\times16$；参数不会随 $T=3$ 而保存三份，activation 路径却仍有 3 个串行 step。

![Universal Transformer iterative refinement](/assets/looped-transformer/02-universal-transformers/figure-1-iterative-refinement.png)

*原图：Figure 1，PDF p. 2；来源：arXiv:1807.03819v3。看图重点：每个 recurrent time step 内，所有 token 位置通过 self-attention 并行交换信息；沿横向 time/depth 方向重复的是同一组 self-attention 与 transition 参数。因此“token 并行”与“depth 串行”可以同时成立。*

### 与 RNN 的关键差异

- RNN 通常沿 token/time 逐个更新；
- UT 在一个 depth step 内同时更新所有位置；
- 每个位置在每次 step 都能通过 attention 读取完整的上一层序列表示。

所以 UT 同时保留了 Transformer 的全局感受野和 recurrence 的重复规则偏置。

## Adaptive Computation Time

ACT 为位置 $i$ 在 step $t$ 产生 $p_i^t\in[0,1]$。当累计概率首次达到 $1-\epsilon$ 时得到停止步 $N_i$，最终表示是截至 $N_i$ 的若干 $H_i^{(t)}$ 的加权和；$p_i^t$ 是网络输出，$\epsilon$ 是固定超参数。训练目标还加入 ponder cost，使更长计算付出代价（§2.1、Appendix C）。

**最小例子。** 若某位置连续产生 $(0.4,0.35,0.3)$，阈值为 0.9，则前两步累计 0.75，第三步越过阈值；最后一步只使用剩余质量 0.25，使权重总和为 1。该位置的逻辑停止步为 3。

**[论文报告]** bAbI 可视化显示，需要更多 supporting facts 的位置往往使用更多 ponder steps（§3、相应 halting 图）。

**[综合判断]** ACT 提供的是“条件计算接口”，不保证真实硬件加速。一个 batch 中不同位置停止时间不同，若实现仍计算到最大 step，节省的只是逻辑计算而非同等比例的墙钟时间。

![Adaptive ponder time on a bAbI example](/assets/looped-transformer/02-universal-transformers/figure-3-adaptive-ponder-time.png)

*原图：Figure 3，PDF p. 6；来源：arXiv:1807.03819v3。看图重点：不同事实与问题 token 的 ponder time 并不相同，少数位置获得更多 recurrent steps。这是 ACT 学会非均匀逻辑计算的实例证据，但单个可视化既不证明普遍 halting 可靠，也不等价于真实设备吞吐按相同比例提升。*

## 理论贡献怎样理解

**[论文报告]** Appendix B 论证 UT 在特定假设下具有计算 universality。关键条件包括：

- recurrent steps 可以随输入长度增长；
- 使用有限精度但足够的状态表示；
- 结论不是固定常数步数模型对任意长度输入都通用。

**[综合判断]** 所以“Turing-complete”不能被缩写成“这个有限步、有限上下文的已训练模型会解决任意程序”。这是模型族的表达能力结论。

## 实验证据：问题—结果—边界

| 实验问题 | 设置与观察 | 支持结论 | 剩余不确定性 |
|---|---|---|---|
| recurrence 是否改善长依赖 agreement？ | 总准确率：Transformer 0.962，UT 0.992，UT+ACT 0.992；5 attractors 时约 0.883/0.892/0.907（Table 2） | UT 在该合成协议更强 | ACT 相对固定步 UT 的总分增益不明显 |
| 能否执行算法？ | 程序执行与记忆类任务接近完美并优于对照（§3.5） | recurrent depth 适合部分算法任务 | 任务短小、与自然数据差距大 |
| 普通翻译是否受益？ | WMT14 En–De：UT Base 28.9，Transformer Base 28.0，weighted Transformer 28.4 BLEU（Table 7） | 不只在合成任务有效 | 搜索预算、训练方差和系统代价未充分归一 |
| 结果是否稳定？ | bAbI 最佳 seed 与 10-seed 平均明显不同（Appendix E） | 最佳 seed 不代表典型表现 | 缺少更系统的方差分析 |

最后一点很重要：只看最佳 seed 会高估方法稳定性。

### Claim–evidence map

| Claim | 证据 | 强度 | Gap |
|---|---|---|---|
| 共享 recurrent depth 是有效架构 | 多任务结果 | moderate–strong | 规模较小、协议异质 |
| ACT 学会按输入分配计算 | halting 可视化与任务结果 | moderate | 未隔离真实吞吐收益 |
| 模型族计算 universal | Appendix B 构造 | strong（条件性理论结论） | 不等于优化可达性 |

## 与后续工作的连接

- ALBERT：保留跨层共享，但重点转为预训练参数效率。
- Learning Learning Algorithms：加入每步 input injection 与专门的 loop loss。
- Length Generalization：让 step 数显式依赖输入长度。
- DEQ：不再显式固定展开次数，而直接求共享变换的 equilibrium。
- Latent Thoughts：把 recurrent depth 当作 reasoning 的计算预算。

## 局限与证据边界

- **[论文报告]** 语言与算法实验规模远小于现代 LLM。
- **[论文报告]** 一些任务报告存在 seed sensitivity。
- **[综合判断]** 动态 halting 的准确率收益、平均步数下降和真实吞吐提升是三个不同指标。
- **[综合判断]** 共享 transition 可能学到可复用算法，也可能只形成固定深度的共享特征变换；必须做超训练深度测试才能区分。

## 超出论文：可证伪扩展

**[扩展假设] Proposal：** 用“训练步数随机化 + residual-norm 终止”替代固定步数或 ACT。

- Reasoning chain：若共享 block 真在逼近一个迭代算法，随机训练深度应减少对绝对 step 的记忆；状态改变量应能提示收敛。
- Predicted observation：在训练未见的更长输入与更多 steps 上，准确率下降更慢。
- Falsification condition：随机深度不改善外推，或 residual 很小但答案仍错误。
- Minimum experiment：copy、addition、bAbI 各 5 seeds；对照固定 $T$、ACT、随机 $T$，报告准确率、平均执行步、wall-clock。
- Cost/risk：residual 变小可能只是表示坍缩，并非任务收敛。

## 推荐复现

1. 在 copy 或 parity 上比较固定 6 层 Transformer 与 1 层 UT × 6 steps。
2. 分别加入 token position、time-step encoding，做四组消融。
3. 画每个位置的 halting step 热力图。
4. 用至少 5 个 seed 报告均值、标准差和失败率。
5. 推理时测试训练范围外的 step，但同时监控过度循环导致的退化。

## 一句话带走

**UT 首次把“Transformer 深度”显式改造成可重复、可自适应的计算时间；后续 Looped Transformer 的大多数问题，都能在这里找到原型。**
