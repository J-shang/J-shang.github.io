---
title: "Looped Transformers for Length Generalization：逐篇解析"
description: "解释 loop 数如何随任务长度增长，审视 step supervision、length extrapolation 与 stopping criterion 的证据边界。"
topic: "looped-transformer"
section: "core"
slug: "looped-transformers-length-generalization"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
order: 23
source:
  repository: "local/looped-transformer"
  path: "papers/07-looped-transformers-length-generalization.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-07-29"
  contentHash: "sha256:0bca45c8d3d5f508ed4b7374307fb2eedf28360538618c82c00b5ae4871bf08a"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份

- 论文：*Looped Transformers for Length Generalization*
- 作者：Ying Fan、Yilun Du、Kannan Ramchandran、Kangwook Lee
- 版本：arXiv:2409.15647v5，2025-05-12
- 发表状态：ICLR 2025
- 主来源：[arXiv](https://arxiv.org/abs/2409.15647)、[HTML 全文](https://arxiv.org/html/2409.15647)
- 官方实现：[UW-Madison-Lee-Lab/looped-tf](https://github.com/UW-Madison-Lee-Lab/looped-tf)
- 阅读范围：正文、Figure 1、Figure 3–7、Definition 3.1、Appendix A/B/E/F；代码仅作复现入口
- 信息截止：2026-07-24

## 30 秒结论

**[论文报告]** 对需要随输入长度重复同一 RASP-L 操作的算法任务，作者训练 input-injected、NoPE 的 decoder-only looped Transformer，并让训练样本携带所需 step 数。推理时随长度增加 loops，可显著改善 copy、parity、addition 等任务的 length generalization。

**[综合判断]** 论文揭示的关键不是“共享参数能处理任意长文本”，而是“当任务计算深度本来就随长度增长时，模型深度也必须允许随长度增长”。

## 5 分钟论文地图

1. §2：NTP 与 full-output prediction（FOP）。
2. §3、Definition 3.1：$n$-RASP-L。
3. Figure 3、Appendix A：parity/copy/addition 的循环程序。
4. §4.2：looped architecture、input injection、step supervision。
5. §4.3：oracle 与 maximum-confidence stopping。
6. §6、Figure 4–7：六个任务、baseline、消融。
7. §7：限制。
8. Appendix E/F：复杂度、模型与训练细节。

前置知识：RASP-L、length generalization、next-token prediction（NTP）、full-output prediction（FOP）。最小例子是 parity：每一轮把相邻部分结果合并，输入越长就需要越多轮，而不是只需要更多并行 token 槽位。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $n$ | 输入长度 | 正整数，token/digit 数 | input size |
| $T(n)$ | 长度 $n$ 所需的算法 step 数 | 正整数函数 | 由任务/程序决定 |
| $L$ | 每个 recurrent step 内的物理层数 | 正整数 | 架构超参数 |
| $x$ | 原始输入 embedding sequence | $n\times d$，省略 batch | runtime input |
| $h_t$ | 第 $t$ 个 recurrent step 状态 | $n\times d$ | activation |
| $F_\theta$ | 共享的 $L$-layer Transformer map | sequence-to-sequence | $\theta$ 可训练 |
| $T_i$ | 第 $i$ 个训练样本的监督 step | 正整数 | 数据附带标签 |
| $\hat y_t$ | 第 $t$ 步全输出预测 | token logits/sequence | activation |

本文的 length 是 token/digit 长度；“adaptive depth”是改变 recurrent step 数，不是改变 hidden width。NoPE 指不加入显式 positional embedding。

## 贡献账本与论证链

```text
某些算法的串行 step 数随输入长度增长
  → 定义可由重复 RASP-L 操作表示的 n-RASP-L 类
  → 用 input-injected FOP loop 执行长度相关步数
  → 用 oracle/置信度规则决定推理停止
  → 在六个合成任务上验证 length extrapolation
```

| 可检查贡献 | 类型与最近基线增量 | 支持位置 | 没有建立 |
|---|---|---|---|
| $n$-RASP-L 任务类 | 问题框架/形式化 | §3、Definition 3.1 | 覆盖所有 length-generalization 任务 |
| 随长度扩展 recurrent depth 的训练法 | 机制 | §4.2 | 不需要 step label |
| oracle/maximum-confidence stopping | inference procedure | §4.3、Appendix B | 通用可靠 learned halting |
| 六任务 length-generalization 增益 | 经验发现 | §6、Figure 4–7 | 可直接外推自然语言长上下文 |

## 为什么普通固定深度会失败

若长度为 $n$ 的输入需要执行 $T(n)$ 步相同局部规则，而模型固定为 $D$ 层，当测试 $n$ 增大到 $T(n)>D$ 时，模型没有足够串行深度。更多 token 位置提供的是并行空间，未必能替代顺序迭代。

## $n$-RASP-L

论文定义：若任务可以分解为预处理、重复 $T(n)$ 次同一有限 RASP-L 操作、后处理，则属于 $n$-RASP-L 类。每个 RASP-L step 可由固定大小 Transformer 表示，所以 loop 次数承担与长度相关的计算量。

**[论文报告]** 作者给出 parity、copy、forward-order addition 的显式程序。以 addition 为例，每步用 shift、XOR、AND 传播 carry。

## 训练形式

### Full-output prediction

与 autoregressive NTP 不同，FOP 在若干内部 step 后一次预测整个输出序列。这样内部循环用于计算，而不是被迫每次生成一个可见 token。

### Input injection

$$
h_{t+1}=F_\theta(h_t+x).
$$

这里 $h_t,x\in\mathbb{R}^{n\times d}$ 是 runtime tensor，$F_\theta$ 的参数 $\theta$ 在所有 $t$ 共享。原始输入每步加入，防止问题信息随长循环消失。

**三步 trace。** $h_0$ 初始化后，$h_1=F_\theta(h_0+x)$，$h_2=F_\theta(h_1+x)$，$h_3=F_\theta(h_2+x)$。若每一步只把 carry 向左传播一位，3 个 steps 最多处理跨 3 位依赖；测试更长加法时必须增加 $t$。

### Step-dependent supervision

每个样本给定所需步数 $T_i$，只在对应 step 对最终答案做交叉熵。不同长度样本提供不同 step 的监督，因此同一 block 在训练集整体上获得多个深度的信号，而无需显式中间 scratchpad。

### Position encoding

**[论文报告]** 作者使用 NoPE，以隔离 looped training 并贴近 RASP-L 不依赖绝对位置编码的假设。

![Method overview for length-adaptive looping](/assets/looped-transformer/07-looped-transformers-length-generalization/figure-1-method-overview.png)

*原图：Figure 1，PDF p. 2；来源：arXiv:2409.15647v5。看图重点：同一 decoder block 被重复 $n$ 次，训练只在与样本所需步数对齐的最终输出上监督；灰色 block 共享参数。图没有显式中间 CoT，因此方法依靠终态监督诱导可复用的内部 steps。*

## 实验设计

任务：

- Copy
- Parity
- Addition
- Binary Sum
- Binary Multiplication
- Unique Set

Baseline 包括 vanilla NTP、NTP + pause tokens、NTP + weight-tied layers、vanilla FOP、FOP + pause，以及无 input injection 的 looped FOP。

评估使用整个输出 exact match，而非逐 token 平均准确率。

## 实验证据：问题—结果—边界

| 实验问题 | 设置与观察 | 支持结论 | 剩余不确定性 |
|---|---|---|---|
| recurrent depth 能否随 length 外推？ | Parity 训练至长度 20，对 40+ digits 近乎完美（§6.2）；Addition/Copy 也明显优于固定基线 | 对所选算法任务，增长 steps 有效 | 数值多由曲线读取，精确值应以原图为准 |
| 增益是否只是 pause token 或 tied NTP？ | NTP-loop/pause 偶有提升，但整体不及 adaptive FOP（Figure 4–6） | 输出协议与 depth schedule 都重要 | baseline 调参公平性仍需复验 |
| input injection 是否必要？ | 去掉 injection、改 pause 或固定 FOP 整体变差（Figure 7） | injection 是当前配方关键组件 | 消融只在此架构/任务局部成立 |
| 正确后还能否稳定？ | Addition/Copy/Multiplication/Unique Set 有一定收敛；Parity/Binary Sum 不同样稳定 | stopping 行为任务相关 | 不能用统一 residual threshold 保证正确 |

![Length-generalization results across six algorithmic tasks](/assets/looped-transformer/07-looped-transformers-length-generalization/figure-4-length-generalization.png)

*原图：Figure 4，PDF p. 8；来源：arXiv:2409.15647v5。看图重点：竖虚线给出最大训练长度；蓝色 adaptive-loop FOP 在多数任务上跨过该边界后仍保持高准确率，而 NTP、pause-token 和 tied NTP 基线不同程度退化。图支持所选六类任务内的 length extrapolation，不支持所有长上下文任务。*

### Claim–evidence map

| Claim | 证据 | 强度 | Gap |
|---|---|---|---|
| 计算深度应随长度增长 | $n$-RASP-L 构造 + 六任务结果 | strong（选定任务类内） | 类外任务未知 |
| adaptive loop 优于固定计算基线 | Figure 4–7 | moderate–strong | oracle step 提供额外信息 |
| confidence 可替代 oracle | §4.3、Appendix B | moderate | per-sample 非收敛任务明显变差 |

## Stopping criterion 的真实含义

作者报告两类：

1. **Oracle**：知道数据生成规则给出的正确 step 数；
2. **Maximum confidence**：运行到上限，选交叉熵最低/置信度最高的 step。

**[论文报告]** batch-level maximum confidence 在多任务上有效；per-sample criterion 对收敛任务接近 oracle，但对非收敛任务明显较弱（Appendix B）。

**[综合判断]** 主结果大量依赖 oracle step，部署含义必须与 learned/observable stopping 分开报告。

![Stopping behavior across six tasks](/assets/looped-transformer/07-looped-transformers-length-generalization/figure-5-stopping-criterion.png)

*原图：Figure 5，PDF p. 10；来源：arXiv:2409.15647v5。看图重点：蓝线是随 step 变化的 accuracy，绿线是 log loss，竖线是按 batch-level confidence 选出的停止点。Addition、Copy 等任务出现较宽的正确区间，Parity 与 Binary Sum 更不稳定；这正是统一 halting 规则难以成立的视觉证据。该图使用完整测试集统计，不能当作逐样本 learned halting 的保证。*

## 复杂度与成本

若每 step 有 $L$ 层、测试长度 $n$ 需 $T(n)$ 步，显式 loop 计算随 $L\,T(n)$ 增长。Appendix E 还对 NTP KV cache、FOP 与不同 stopping 上限进行复杂度比较。

参数量可以固定，但推理延迟不会固定。

## 局限

- 任务被选为存在单循环 $n$-RASP-L 解；
- 当前定义不支持多个先后嵌套/串联 loop；
- 训练需要 ground-truth step 数；
- 直接 BPTT 很多 steps 成本高；
- 仅在有限训练长度、有限算力下测试；
- NoPE 与真实 LLM 常用 RoPE/其他 position scheme 有差异；
- 合成 exact-match 任务不能直接外推到自然语言长上下文。

## 超出论文：无 oracle 的 halting

**[扩展假设] Proposal：** 训练一个只读当前状态改变量与输出 margin 的 halting head，并对过早/过晚停止共同计费。

- Reasoning chain：oracle $T_i$ 不可部署；单纯最大置信度会把错误过度自信当收敛；联合 residual 与校准 margin 可能更可辨识。
- Predicted observation：收敛型任务接近 oracle，Parity/Binary Sum 仍需任务相关信号。
- Falsification condition：halting head 在长度外推区系统性早停，且不优于 batch maximum-confidence。
- Minimum experiment：训练长度内校准、2×长度测试；报告 exact match、平均/尾部 steps、过早停止率、wall-clock。
- Cost/risk：halting labels 若来自 oracle，会把原假设重新注入；应比较监督与自监督版本。

## 推荐复现

1. 先做 Copy、Parity，再做 Addition。
2. 同时报告 oracle、batch confidence、per-sample confidence。
3. 测试长度与 loops 做二维网格，而不是只给一条最佳曲线。
4. 记录正确后继续 loop 是否 overthink。
5. 消融 input injection、NoPE/RoPE、FOP/NTP、fixed/adaptive depth。

## 一句话带走

**长度泛化的核心不只是接受更长 sequence，而是让串行计算深度也随问题长度增长；Looped Transformer 为这种增长提供了参数共享的实现。**
