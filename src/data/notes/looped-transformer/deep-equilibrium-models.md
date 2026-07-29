---
title: "Deep Equilibrium Models：逐篇解析"
description: "从显式 weight tying 走向 fixed point、Broyden 求根与隐式微分，并区分终态与有限循环轨迹。"
topic: "looped-transformer"
section: "adjacent"
slug: "deep-equilibrium-models"
date: 2026-07-29
updated: 2026-07-29
cutoff: 2026-07-24
order: 30
source:
  repository: "local/looped-transformer"
  path: "papers/09-deep-equilibrium-models.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66"
  syncedAt: "2026-07-29"
  contentHash: "sha256:76eb41b4aaedfaec5ba8a1ebbfc3d01260cfcbd8f77e6086b0a3d95fefbe0748"
  manifest: "looped-transformer"
  managed: true
---
## 论文身份与证据范围

- 论文：*Deep Equilibrium Models*
- 作者：Shaojie Bai、J. Zico Kolter、Vladlen Koltun
- 分析版本：arXiv:1909.01377v2，2019-10-28
- 发表状态：NeurIPS 2019 Spotlight Oral
- 主来源：[NeurIPS 论文页](https://papers.nips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html)、[发表版 PDF](https://papers.nips.cc/paper/2019/file/01386bd6d8e091c2ab4c7c7de644d37b-Paper.pdf)、[arXiv](https://arxiv.org/abs/1909.01377)
- 官方实现：[locuslab/deq](https://github.com/locuslab/deq)
- 阅读范围：正文 §1–6、Theorem 1、Figure 1–3、Table 1–4 及论文附录引用；数值以发表版 PDF 为准
- 信息截止：2026-07-24；本笔记不把后续 DEQ 稳定化方法当成原论文贡献

## 30 秒结论

**[论文报告]** DEQ 不再显式展开固定数量的共享层，而把模型输出定义为

$$
z^\star=f_\theta(z^\star;x)
$$

的 fixed point。forward 用 Broyden 等 root solver 找 $z^\star$；backward 用 implicit differentiation 解一个线性系统，因此无需保存所有 forward 迭代的 activation。论文在 WikiText-103、Penn Treebank 与 copy-memory 上展示了接近或优于相似规模显式网络的结果，并在其受控测量中把非 embedding activation memory 降低 80% 以上、最高 88%（Abstract、§5、Table 2–3）。

**[综合判断]** DEQ 是“共享深度趋于平衡”路线，而多数 Looped Transformer 是“保留有限迭代轨迹”路线。前者适合只关心终态的任务；若中间 step 本身承载算法进度或 test-time compute，强行求 equilibrium 可能改变问题。

## 5 分钟论文地图

```text
显式深层网络的 activation memory 随深度增长
  → 观察 weight-tied sequence model 可能趋于 fixed point
  → 直接求 gθ(z;x)=fθ(z;x)-z 的根
  → 用隐式函数定理得到不依赖 solver 轨迹的梯度
  → 在 Transformer/TrellisNet 上验证性能、显存与运行时
  → 结论限于能稳定找到任务相关 equilibrium 的模型
```

阅读定位：

1. §3.1.1：forward root finding。
2. Theorem 1、§3.1.2–3.1.3：implicit gradient 与 backward solver。
3. §4：DEQ-TrellisNet、DEQ-Transformer 两个实例。
4. Table 2–4、Figure 2–3：性能、显存与速度代价。
5. Appendix A：Theorem 1 证明。

前置知识：fixed point、Jacobian、vector–Jacobian product、implicit function theorem。最小例子是标量映射 $z=\tanh(az+x)$：DEQ 输出不是第 10 次迭代，而是满足等式的 $z^\star$。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $T,d,p,q$ | 序列长度、hidden/input/output width | 正整数 | 输入规模 / 超参数 |
| $x_{1:T}$ | 输入序列 | $T\times p$ | fixed runtime input |
| $z^{[i]}_{1:T}$ | solver/显式展开第 $i$ 步状态 | $T\times d$ | activation |
| $z^\star_{1:T}$ | equilibrium state | $T\times d$ | solver 派生的 activation |
| $f_\theta$ | 带 input injection 的共享 sequence map | $(T\times d,T\times p)\to T\times d$ | $\theta$ 可训练 |
| $g_\theta(z;x)$ | root residual，$f_\theta(z;x)-z$ | $T\times d$ | 派生函数 |
| $J_g(z^\star)$ | $g$ 对 $z$ 的 Jacobian | $(Td)\times(Td)$ 的线性算子 | 运行时导数 |
| $\epsilon$ | root residual tolerance | 非负标量 | solver 超参数 |
| $\ell$ | task loss | 标量 | 训练目标 |

本文省略 batch 轴。$T$ 是序列长度，不是 loop 次数；上标 $[i]$ 才是 solver/层迭代索引。实现不显式构造 $(Td)\times(Td)$ Jacobian，而通过 autograd 计算向量–Jacobian product。

## 贡献账本

| 可检查贡献 | 类型与最小增量 | 支持位置 | 没有建立 |
|---|---|---|---|
| 用 equilibrium 替代显式深度 | 模型框架；相对 weight-tied unroll 改为 implicit layer | §3、Figure 1 | 所有共享网络都存在唯一平衡点 |
| 对 equilibrium 直接反传 | 数学/算法机制 | Theorem 1、Eq. 8–11 | backward 不需要额外计算 |
| Broyden forward/backward 实现 | 系统算法 | §3.1.3 | 它是所有任务的最优 solver |
| Transformer/TrellisNet 实例与 LM 结果 | 系统 + 经验发现 | §4–5、Table 1–4 | 现代大模型规模仍保持相同速度/稳定性 |
| 常数 activation-memory 路径 | 资源性质 | Theorem 1、Table 2–3 | 参数、embedding、solver workspace 都是常数或被消除 |

## 方法复原

### 1. 从显式 weight tying 到 root

显式共享网络为（§3、Eq. 3）：

$$
z^{[i+1]}=f_\theta(z^{[i]};x),\qquad z^{[0]}=0.
$$

这里 $z^{[i]}\in\mathbb{R}^{T\times d}$ 是第 $i$ 次 activation，$x\in\mathbb{R}^{T\times p}$ 每步注入，$\theta$ 是跨 $i$ 共享且由 optimizer 更新的参数。显式 BPTT 需要保存或重算这条轨迹。

若极限存在，则（§3、Eq. 4）：

$$
z^\star=f_\theta(z^\star;x).
$$

定义

$$
g_\theta(z;x)=f_\theta(z;x)-z,
$$

则目标变成求 $g_\theta(z^\star;x)=0$。$g_\theta$ 没有新参数，只是把 fixed-point equation 改写成 root problem。

**标量算例。** 若 $f(z;x)=0.5z+x$，则 $z^\star=2x$。从 0 显式迭代得到 $x,1.5x,1.75x,\ldots$；root solver 直接寻找 $g(z)=x-0.5z=0$。两条计算路径的终点相同，中间轨迹不同。

![Deep equilibrium model versus a conventional weight-tied deep network](/assets/looped-transformer/09-deep-equilibrium-models/figure-1-deq-vs-weight-tied.png)

*原图：Figure 1，PDF p. 6；来源：NeurIPS 2019 发表版。看图重点：左侧把显式共享层的轨迹与直接求 fixed point 区分开；右侧显示普通展开必须保留多层 activation，而 DEQ 在 equilibrium 处通过隐式 backward 绕过整条轨迹。彩色块只表示与 effective depth 相关的训练内存，不表示参数、embedding 和 solver workspace 都消失。*

### 2. Forward：Broyden 求根

论文用 quasi-Newton update（§3.1.1、Eq. 6）：

$$
z^{[i+1]}=z^{[i]}-\alpha B_i g_\theta(z^{[i]};x).
$$

这里 $\alpha$ 是 solver step size，$B_i$ 是 $J_g(z^{[i]})^{-1}$ 的低秩近似，二者都不是模型的 learned Transformer 参数。迭代在 $\lVert g_\theta(z^{[i]};x)\rVert<\epsilon$ 或达到最大步数时停止（§3.1.3）。

操作路径：

```text
x, z[0]=0
  → evaluate fθ(z[i];x)
  → residual g=f-z
  → Broyden updates inverse-Jacobian approximation
  → stop on tolerance/max-iterations
  → z*
```

### 3. Backward：隐式微分

由 $g_\theta(z^\star;x)=0$ 对参数微分，[复原推导]：

$$
J_g(z^\star)\frac{\partial z^\star}{\partial\theta}
+\frac{\partial f_\theta(z^\star;x)}{\partial\theta}=0,
$$

因此

$$
\frac{\partial z^\star}{\partial\theta}
=-J_g(z^\star)^{-1}
\frac{\partial f_\theta(z^\star;x)}{\partial\theta}.
$$

这里 $J_g=\partial g/\partial z$，$\partial f/\partial\theta$ 把参数扰动映射到 hidden-space；两者都在当前样本的 $z^\star$ 处求值。对 loss $\ell$ 实际需要的是 vector–Jacobian product，不会显式生成逆矩阵。论文把它改写为 Eq. 11 的线性系统，再用 Broyden 类方法求解（Theorem 1、§3.1.2–3.1.3）。

**关键 invariant。** backward 只需要 $z^\star$、$x$、$\theta$ 和 solver 的工作向量，不需要保存 forward 中每一个 $z^{[i]}$；因此 activation memory 不随等效展开深度线性增长。

### 4. 计算和内存账本

- 显式 $L$ 步 unroll：普通 BPTT activation memory 为 $O(L)$；checkpointing 可降到约 $O(\sqrt L)$，但增加重算（§2）。
- DEQ：关于 effective depth 的 activation memory 为 $O(1)$，但一次模型计算仍需多次 $f_\theta$ evaluation。
- Forward cost 取决于 root iterations；backward cost取决于线性系统 iterations。
- tolerance 越松或 iteration cap 越低通常更快，但 fixed point/gradient 误差增大（Figure 3）。

所以“infinite depth”是模型定义，不是零成本地执行无穷次 block。

![Broyden iterations and fixed-point convergence during DEQ training](/assets/looped-transformer/09-deep-equilibrium-models/figure-2-broyden-iterations.png)

*原图：Figure 2，PDF p. 8；来源：NeurIPS 2019 发表版。看图重点：左图显示 forward/backward Broyden 迭代数会随训练增长；右图显示 DEQ 能把 residual 压到很小，而显式 weight-tied Transformer 可能在 fixed point 附近振荡。图直接反驳“常数 activation memory 就等于计算更便宜”，同时也说明 solver tolerance 是模型行为的一部分。*

## 实验证据：问题—设置—结果—边界

| 实验问题 | 设置与准确结果 | 支持结论 | 剩余不确定性 |
|---|---|---|---|
| equilibrium 模型能否保留长记忆？ | copy-memory，$T=400$：DEQ-Transformer loss $3.5\times10^{-6}$，TCN $2.7\times10^{-5}$（Table 1） | 小型压力测试中可找到有用 equilibrium | 与真实语言建模不同 |
| 能否匹配显式 sequence model？ | PTB：DEQ-TrellisNet PPL 57.1、1.2GB；60-layer TrellisNet 57.0、8.5GB（Table 2） | 几乎同 PPL 下显存显著降低 | 内存测量排除 word embeddings |
| Transformer 实例是否有效？ | WT103：medium TXL 24.3 PPL/8.5GB，DEQ 24.2/2.7GB；adaptive-embed TXL 23.6/9.0GB，DEQ 23.2/3.7GB；small tied TXL 34.9/6.8GB，DEQ 32.4/1.1GB（Table 3） | 在该协议下性能竞争且内存低 | 参数数不完全相同；2019 配方 |
| 显存节省是否带来速度收益？ | DEQ/18-layer Transformer：训练 2.82×、推理 1.76×；DEQ/70-layer TrellisNet：2.40×/1.64×，大于 1 表示 DEQ 更慢（Table 4） | memory 与 wall-clock 是不同维度 | solver/kernel 与硬件依赖强 |
| solver 误差是否重要？ | Figure 2–3：迭代数随训练增加；过松 tolerance 会使 PPL 快速恶化 | equilibrium 精度是实质超参数 | 未给通用最佳 stopping rule |

### Claim–evidence map

| Claim | 证据 | 强度 | Gap |
|---|---|---|---|
| 可通过 implicit differentiation 常数内存训练 | Theorem 1 + 实现 | strong | solver workspace/embedding 不包含在“depth-constant”主张内 |
| 实际 LM 性能可竞争 | Table 2–3 | moderate–strong | 模型/调参并非所有项完全相同 |
| 计算需求“similar” | Table 4 | moderate | 明确比对应显式网络慢 1.64–2.82× |
| equilibrium 是共享深度的合理终态 | Figure 2 + 结果 | moderate | 存在/唯一/易求未普遍保证 |

## 关键洞察

1. **[复原推导] 终态与轨迹是两个建模选择。** DEQ 的 gradient 对选用哪种 forward root solver 理论上不敏感，但有限容差下仍受数值解影响；显式 loop 则把每一步都放进 computational graph。
2. **[综合判断] 常数显存换来的主要是 solver 计算。** Table 4 是防止“更省显存等于更快”的关键负面证据。
3. **[综合判断] input injection 是 equilibrium 可依赖问题的基础。** 若 $f_\theta$ 不持续读 $x$，稳定终态可能更多由 autonomous dynamics 决定。
4. **[综合判断] reasoning 不一定适合只读 fixed point。** 若答案依赖有限步计数、奇偶振荡或中间 scratchpad，终态化会丢失有用 trajectory information。

## 局限与开放问题

作者证据中可直接看到：

- root tolerance 和 iteration cap 影响性能，且迭代次数会随训练增长（Figure 2–3）；
- DEQ 在 Table 4 中比对应显式网络慢；
- Table 2–3 的 memory 不含 word embeddings；
- 实例集中于 2019 年 Transformer/TrellisNet 与语言建模。

分析补充：

- fixed point 可能不存在、多解或 solver 对初始化敏感；
- 任务 loss 最优不等价于小 residual，也不保证 Jacobian 条件良好；
- “infinite depth equivalence”要求显式 tied iteration 的极限与所求 root 对应；
- 论文没有现代 decoder-only、MoE、RL reasoning 或超长上下文证据。

## 超出论文：区分 equilibrium 与 useful finite trajectory

**[扩展假设] Proposal：** 对同一共享 block 比较 fixed-$L$ unroll、random-$L$ unroll 与 DEQ 三种训练/readout。

- Reasoning chain：三者共享参数化，只改变“读有限状态还是平衡态”及梯度路径，可识别任务需要终态还是轨迹。
- Predicted observation：稳态回归任务中 DEQ 更省 activation memory；parity/有限步程序中显式 loop 更可靠。
- Falsification condition：所有任务中三者在等 wall-clock 和调参预算下无稳定差异。
- Minimum experiment：linear regression、copy、parity；报告 task metric、root residual、Jacobian spectral radius、function evaluations、peak memory、step time。
- Cost/risk：solver 与 unroll 的 wall-clock 公平匹配较难，必须同时给 block calls 和真实时间。

## 复现与阅读路径

1. Figure 1 → §3.1 → Theorem 1：先理解 implicit layer。
2. Table 3 → Table 4 → Figure 3：同时看性能、显存和速度。
3. 最小复现先用 $z=\tanh(Wz+Ux)$，检查 forward residual 与 implicit gradient 的 finite-difference 误差。
4. 再复现小型 DEQ-Transformer，记录 forward/backward solver iterations。
5. sanity checks：$\lVert f(z^\star;x)-z^\star\rVert$、implicit gradient vs. long-unroll gradient、tolerance–accuracy 曲线。

## 一句话带走

**DEQ 把“重复多少层”改写成“求哪个平衡点”，用隐式梯度换取关于深度的常数 activation memory；它与显式 loop 同源，但服务于终态而不是迭代轨迹。**
