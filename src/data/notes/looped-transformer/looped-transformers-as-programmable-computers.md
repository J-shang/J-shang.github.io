---
title: "Looped Transformers as Programmable Computers：逐篇解析"
description: "还原 Looped Transformer 执行可编程计算的构造，区分表达能力、可训练性与真实语言任务能力。"
topic: "looped-transformer"
section: "core"
slug: "looped-transformers-as-programmable-computers"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
order: 20
source:
  repository: "J-shang/looped-transformer"
  path: "papers/04-looped-transformers-as-programmable-computers.md"
  url: "https://github.com/J-shang/looped-transformer/blob/9ab82eeb3178ddd627b592ac2cba22de91e7be66/papers/04-looped-transformers-as-programmable-computers.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-08-02"
  contentHash: "sha256:7d4a24d72965649d64cb7158112b947d4ada74351ec7489d6ded9f212ba3eb69"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份

- 论文：*Looped Transformers as Programmable Computers*
- 作者：Angeliki Giannou、Shashank Rajput、Jy-Yong Sohn、Kangwook Lee、Jason D. Lee、Dimitris Papailiopoulos
- 版本：arXiv:2301.13196v1，2023-01-30
- 发表状态：ICML 2023
- 主来源：[PMLR 论文页](https://proceedings.mlr.press/v202/giannou23a.html)、[arXiv](https://arxiv.org/abs/2301.13196)
- 阅读范围：PMLR 正文、Figure 1–6、Algorithm 1、Table 1、定理与构造附录；PMLR 版本作为发表版主来源
- 信息截止：2026-07-24（后续论文只用于路线关系，不用于增强原定理）

## 30 秒结论

**[论文报告]** 作者手工构造 Transformer 权重，把输入 sequence 组织成“指令 + 内存 + scratchpad”，并在外层反复调用固定 Transformer。attention 负责寻址与搬运，MLP 负责逻辑和数值操作，从而执行条件分支、函数调用、线性代数和带 backpropagation 的 in-context learning 程序。

**[综合判断]** 这是一篇 expressivity 论文：它证明“存在这样的权重”，没有证明 SGD 会从普通数据中学到这台计算机。

## 5 分钟论文地图

1. Abstract、Figure 1：punchcard、memory、instruction 的整体概念。
2. §3、Algorithm 1：looped Transformer 形式。
3. §4：attention 如何实现 read/write，MLP 如何实现条件逻辑。
4. §5、Theorem 2：FLEQ 指令集和通用执行器。
5. Table 1：不同算法所需层数和 head 数。
6. 后续章节：矩阵运算、power iteration、SGD/backprop 构造。
7. Appendices：各模块的完整权重与误差控制。

前置知识：attention 寻址、ReLU MLP、RAM/汇编式程序、构造性存在证明。最小例子是让一个 attention head 按 address 从两个 memory token 中选一个，再由 MLP 把值减一。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $X^{(t)}$ | 第 $t$ 次循环时的完整“程序 + 内存”序列 | $d\times n$（本文沿论文的列 token 直觉；实现可转置） | runtime state |
| $t,T$ | 当前调用与总程序步数 | $t=0,\ldots,T$ | 运行时索引 / 预算 |
| $\operatorname{TF}_W$ | 固定 Transformer 执行器 | sequence-to-sequence map | $W$ 为手工设定参数 |
| $W$ | attention/MLP 的全部权重 | 若干矩阵集合 | 固定构造，不由实验 SGD 学得 |
| $a,b,c,p$ | 两个源地址、目标地址、跳转地址 | 离散地址，编码进 token 字段 | 程序输入 |
| $f_m$ | 第 $m$ 个可调用 function block | 向量到向量的固定/构造映射 | 指令选择 |
| $l_i$ | 第 $i$ 个 function block 所需层数 | 正整数 | 构造复杂度 |

本论文中的“constant depth”指每次外层调用的 Transformer 层数不随程序长度 $T$ 增长；总执行时间仍随 $T$ 增长。存在性证明中的 $W$ 不是普通训练得到的模型 checkpoint。

## 贡献账本与论证链

```text
Transformer 已能内容寻址，但固定 forward 的程序长度受深度约束
  → 把程序、内存、counter 编入序列
  → 用 attention 读写、MLP 运算、外层 loop 推进 counter
  → 构造 FLEQ 通用执行器与数值算法
  → 得到表达能力存在性结论
```

| 可检查贡献 | 类型与最小增量 | 证据位置 | 没有建立 |
|---|---|---|---|
| 结构化 Transformer “计算机”表示 | 机制/构造 | §3–4、Figure 1、Algorithm 1 | 自然语言会自动形成同样表示 |
| FLEQ 通用执行器 | 理论/系统构造 | §5、Theorem 2 | SGD 可找到这组权重 |
| 矩阵算法与 backprop/SGD 程序 | 构造性实例 | Table 1、后续章节与附录 | 这是数值稳定或工程高效的实现 |
| 程序长度移到外层循环 | 理论设计原则 | Algorithm 1、Theorem 2 | 总计算量变成常数 |

## 模型怎样成为“计算机”

输入不是普通自然语言，而是结构化向量序列。不同 token/embedding 子空间承担：

- program memory；
- data memory；
- program counter；
- address pointers；
- function selector；
- scratchpad；
- condition flag。

外层循环执行：

$$
X^{(t+1)}=\operatorname{TF}_W(X^{(t)}).
$$

这里 $X^{(t)}\in\mathbb{R}^{d\times n}$ 是 input-dependent runtime state；$W$ 是一次构造后在所有 $t$ 共享的 attention/MLP 权重。程序与数据主要编码在 $X$ 中。每次调用形状不变，但 program counter、memory 与 scratchpad 字段会变化。

![Structured input layout for executing FLEQ commands](/assets/looped-transformer/04-looped-transformers-as-programmable-computers/figure-5-input-layout.png)

*原图：Figure 5，PDF p. 17；来源：ICML 2023 PMLR 发表版。看图重点：列方向分为 scratchpad、memory、instructions，行方向显式留出 program counter、current instruction、scratchpad memory、embedding 与 function blocks。模型之所以能像计算机，是因为运行时状态先被人为组织成这些可寻址字段；这不是普通自然语言 token 会自动形成的布局。*

### Attention 是寻址器

通过精心构造 query/key，attention 近似选择某个 address 的 token；value 投影把数据写到指定 scratchpad。高温 softmax 近似 hard selection，因此很多结论是“精确或任意精度近似”，取决于构造条件。

### MLP 是局部算术与控制器

ReLU MLP 实现条件门控、比较、非线性函数以及 embedding 字段修改。attention 先取数，MLP 再运算，之后 attention 写回。

### Program counter 与条件跳转

论文定义 FLEQ 风格指令：

$$
\text{mem}[c]
=f_m(\text{mem}[a],\text{mem}[b]);
\quad
\text{if }\text{mem}[\text{flag}]\le 0\text{ goto }p.
$$

这里 $a,b$ 是两个源地址，$c$ 是写回地址，$p$ 是条件跳转地址，$m$ 选择 function block；它们都是程序 token 中的离散字段，不是模型参数。同一 Transformer 每个 loop 取一条指令、执行、写回并更新 program counter，由此支持任意长程序。

**最小三指令 trace。** 假设 memory 中 `mem[1]=3, mem[2]=1`，当前指令计算 `mem[3]=mem[1]-mem[2]`。attention 先按地址 1/2 读出 3 和 1；MLP 得 2；写回 head 把 2 放入地址 3；counter 再跳到下一条指令。外层第二次调用使用同一 $W$，但读取的是更新后的 $X^{(1)}$。

![OISC instruction implemented as Transformer building blocks](/assets/looped-transformer/04-looped-transformers-as-programmable-computers/figure-3-oisc-building-blocks.png)

*原图：Figure 3，PDF p. 5；来源：ICML 2023 PMLR 发表版。看图重点：一次指令被拆成 Read Command → Read Data → Subtract → Write → If Goto，恰好对应“寻址、局部运算、写回、控制流”的模块化构造。图展示的是存在性设计蓝图，不是从随机初始化训练出来的内部电路。*

## 主要理论结论

**[论文报告]** Theorem 2 表明，给定若干 Transformer function blocks，可以构造常数层数的执行器；层数约为 $9+\max_i l_i$，宽度随函数数目、数据维度和地址编码增长。每个 recurrent call 执行一条指令。

**[论文报告]** Table 1 给出的代表性构造：

| 程序 | 层数 | attention heads |
|---|---:|---:|
| SUBLEQ/FLEQ 类执行 | 9 | 2 |
| 矩阵求逆 | 13 | 1 |
| Power iteration | 13 | 1 |
| SGD | 13 | 1 |

层数保持常数，程序运行时间转移到外层 loop 次数。

## 它真正证明了什么

- **[论文报告]** 常数深度、共享权重 Transformer 家族具备执行通用程序的表达能力。
- **[论文报告]** attention 可以显式实现地址读取、写入和控制流。
- **[论文报告]** 程序长度可由 loop 次数扩展，而不需为每一步新增参数。

### Claim–evidence map

| Claim | 论文证据 | 强度 | Gap |
|---|---|---|---|
| 常数物理深度可执行任意长 FLEQ 程序 | Theorem 2 + 构造附录 | strong（条件性存在证明） | 宽度、精度、循环时间仍增长 |
| attention 可做地址读写 | 模块构造与权重证明 | strong（构造域内） | 对噪声和有限精度的工程鲁棒性有限 |
| 可执行 power iteration/SGD 等 | Table 1 + 对应构造 | strong（表达性） | 不代表学习性或效率 |

## 它没有证明什么

- **[论文报告]** 论文自己明确提醒，这些构造与现实语言模型训练没有相似性。
- **[综合判断]** 没有从随机初始化训练执行器。
- **[综合判断]** 没有证明自然语言 prompt 会自动形成可靠的离散 memory 和 program counter。
- **[综合判断]** 复杂度上常数层不等于低成本；每个 loop 仍需完整 attention，标准 dense attention 对 sequence length 为二次复杂度。

## 与后续可学习性论文的关系

这篇工作回答：

> “Looped Transformer 能否表示算法？”

`Learning Learning Algorithms` 用实验追问：

> “从任务 loss 出发能否学到逐步更新？”

`Multi-step Gradient Descent` 再追问：

> “能否证明优化会收敛到算法解？”

三篇必须按 expressivity → empirical learnability → provable learnability 的顺序理解。

## 复原推导：为什么 loop 很关键

**[复原推导]** 若每次只执行一条 FLEQ 指令，则长度为 $T$ 的程序需要：

$$
X^{(T)}=\operatorname{TF}_W^{\circ T}(X^{(0)}).
$$

不循环时，要么把 $T$ 条指令对应的执行逻辑展开为深度，要么让一次 forward 直接近似整个程序。循环把“程序长度”从参数深度移到运行时间，这是 recurrent depth 的核心交换。

## 局限

- 结构化输入、专门 embedding 分区和高精度权重假设很强；
- 部分操作依赖 softmax 温度带来的近 hardmax；
- scratchpad 为不同函数预留独立空间，参数/宽度构造并不追求工程效率；
- 没有真实数据、随机初始化训练或大模型 benchmark；
- sparse attention 可降低构造复杂度，但不是论文所有实验的现成硬件收益。

## 超出论文：从存在性到可学习性的桥接实验

**[扩展假设] Proposal：** 固定输入编码和执行架构，只把手工权重替换为“手工权重加噪后训练恢复”，逐步增加噪声。

- Reasoning chain：纯随机初始化失败无法区分优化困难与表示方案错误；从已知解附近出发可直接测量解的 basin。
- Predicted observation：读写模块比多步控制流更容易恢复，容许噪声随程序长度下降。
- Falsification condition：即使极小权重扰动也无法通过程序 I/O loss 恢复。
- Minimum experiment：1 次 read/write、3 条 FLEQ、10 条带分支程序；5 seeds；报告 exact execution、softmax margin、梯度与累计误差。
- Cost/risk：局部恢复成功仍不能说明从随机初始化全局可学。

## 推荐阅读与复现

1. 先只复原一个 `read(address)` attention head。
2. 再做 `write(address, value)` 与条件赋值。
3. 实现 2–3 条指令的小程序，不必一开始复现矩阵求逆。
4. 对 softmax temperature 做误差曲线，验证 hard selection 近似。
5. 把“手工构造成功”与“从随机初始化训练成功”分成两个实验。

## 一句话带走

**循环让一个固定 Transformer 从“函数近似器”变成可重复调用的指令执行器；但这篇论文证明的是可编程性存在，不是训练会自然发现程序。**
