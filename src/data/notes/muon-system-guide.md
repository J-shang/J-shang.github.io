---
title: "Muon 优化器系统学习指南"
description: "从更新几何、Newton–Schulz 数值近似到分布式实现，建立一套可验证的 Muon 心智模型。"
category: "Muon 专题"
date: 2026-07-06
updated: 2026-07-06
cutoff: 2026-07-01
featured: true
order: 0
readtime: 30
source: "https://github.com/J-shang/Muon/blob/main/MUON_LEARNING_GUIDE.md"
---
> 范围：本文讨论深度学习优化器 **Muon（MomentUm Orthogonalized by Newton–Schulz）**，不是粒子物理中的 μ 子。<br>
> 信息截点：**2026-07-01**。Muon 仍处于快速演进期；本文把已被大规模训练采用的主线，与尚待独立复现的前沿预印本分开记录。

## 0. 一页结论

Muon 的核心不是“把 Adam 的某个超参数改一下”，而是改变矩阵参数的更新几何：

1. 对梯度做 SGD momentum / Nesterov momentum；
2. 把动量矩阵的奇异值压到近似相同，得到近似极分解因子（polar factor）；
3. 按矩阵形状缩放更新，再施加解耦 weight decay；
4. 只用它更新隐藏层矩阵；embedding、LM head、bias、norm scale 等通常仍交给 AdamW。

若动量矩阵为

$$M=U\Sigma V^\top,$$

理想的 Muon 方向是

$$\operatorname{polar}(M)=UV^\top.$$

所以 Muon 保留奇异向量，抹平奇异值。它可以被理解成：

- 对矩阵更新做近似极分解/正交化；
- 谱范数几何下的最陡下降方向；
- 不累积预条件器时的“instantaneous Shampoo”；
- 一种只依赖一阶梯度、但利用矩阵结构的几何优化器。

最后一点很重要：有论文把 Muon 称为“最简单的二阶优化器”，但它不计算 Hessian，也没有 Adam 的二阶矩状态。更准确的说法是：**它具有预条件/二阶方法的味道，却也可以被完整地解释为非欧氏范数下的一阶法。**

截至本文时间，最稳妥的判断是：

- Muon 在 Transformer **从头预训练**上的证据最强，尤其值得在大 batch、长训练和优化器状态内存紧张时评估。
- 它不是无条件的 AdamW 替代品。更新缩放、参数路由、weight decay、NS 精度和分布式实现都属于算法本身，而不只是工程细节。
- 对 AdamW 预训练模型直接切换 Muon 做全量微调，存在 optimizer mismatch 风险；应单独验证，不能把预训练结论外推到 SFT/RL。
- “约 2× 计算效率”是特定 scaling-law 设置下的论文结果，不等于任何模型都能获得 2× 墙钟加速。

## 1. 必备知识地图

建议先补齐以下概念，再读 Muon 论文：

| 层次 | 必备概念 | 学到什么程度 |
|---|---|---|
| 优化基础 | SGD、momentum、Nesterov、AdamW、解耦 weight decay | 能手写更新式，区分梯度裁剪、更新裁剪与权重衰减 |
| 线性代数 | SVD、谱范数、Frobenius 范数、核范数、极分解、半正交矩阵 | 能从 $M=U\Sigma V^\top$ 推出 $UV^\top$ |
| 数值计算 | Newton–Schulz 迭代、条件数、低精度矩阵乘 | 明白为什么只迭代少量步，以及近似误差如何进入优化 |
| 深度学习工程 | mixed precision、optimizer state、ZeRO/FSDP、张量并行 | 能解释 Muon 的内存收益和分布式通信代价 |
| LLM 实验方法 | scaling law、critical batch size、muP、token/FLOP/wall-clock 公平比较 | 不把“步数更少”误读成“训练更便宜” |

每个概念的展开笔记见：[必备知识地图/README.md](/notes/)。

## 2. 从 SGD 到 Muon

### 2.1 动量

对某个二维权重 $W_t\in\mathbb{R}^{A\times B}$，梯度为 $G_t$。简化的 momentum 为

$$M_t=\beta M_{t-1}+G_t.$$

使用 Nesterov 时，各实现的记号略有不同；PyTorch 当前文档写作

$$\widetilde M_t=G_t+\beta M_t.$$

后续正交化的对象是 $M_t$ 或 $\widetilde M_t$，因此比较实现时必须先核对 momentum 约定。

### 2.2 理想正交化

若 $M=U\Sigma V^\top$，则

$$O=UV^\top.$$

当矩阵满秩时，也可写成

$$O=(MM^\top)^{-1/2}M$$

或

$$O=M(M^\top M)^{-1/2},$$

具体选择取决于哪一边维度更小。$O$ 是离 $M$ 最近的半正交矩阵之一；它的非零奇异值都为 1。

### 2.3 谱范数最陡下降视角

在线性化目标下求

$$\arg\min_{\|\Delta\|_2\le 1}\langle G,\Delta\rangle,$$

其方向与 $-UV^\top$ 对齐。原因是谱范数的对偶范数是核范数。于是：

- SGD 在欧氏/Frobenius 几何里直接用梯度；
- Adam 近似做逐坐标尺度适配；
- Muon 把二维权重当作线性算子，在谱范数几何里选择更新。

这解释了 Muon 为什么必须知道参数的“语义”：同样是二维张量，hidden linear weight 与 embedding table 并不扮演相同的算子角色。

### 2.4 为什么可能更有效

直觉上，普通梯度或动量可能由少数大奇异方向主导；Muon 把奇异谱压平，让一次更新同时利用更多方向。Moonlight 实验观察到 Muon 训练所得权重的 SVD entropy 往往高于 AdamW。但需要区分：

- “谱更平、方向更丰富”是可测的经验现象；
- “它一定导致更好的特征学习/泛化”仍是研究假设，不是普遍定理；
- Muon 处理的是**更新矩阵**，不是强制权重矩阵本身正交。

## 3. Newton–Schulz：Muon 的计算核心

直接每步做 SVD 太贵。Muon 用少量 Newton–Schulz（NS）迭代近似 $UV^\top$。

先归一化：

$$X_0=\frac{M}{\|M\|_F+\varepsilon}.$$

若把较短维放在行方向，常见五次迭代写作

$$A_k=X_kX_k^\top,$$

$$X_{k+1}=aX_k+bA_kX_k+cA_k^2X_k.$$

原始 Muon 常用：

$$ (a,b,c)=(3.4445,-4.7750,2.0315),\qquad N_{NS}=5.$$

需要牢记三件事：

1. 这组经验系数追求的是在 BF16/GPU 上快速压平奇异谱，并非高精度求极分解。
2. 因而“orthogonalized”在生产实现中通常是**近似正交化**；输出奇异值不会严格等于 1。
3. NS 步数、系数、归一化、学习率和 momentum 会耦合。改变 `ns_steps` 后沿用原超参数，并不构成公平消融。

若 $m=\min(A,B)$、$n=\max(A,B)$，主要计算复杂度约为 $O(nm^2)$ 每次迭代。方阵最贵，高度长方形矩阵相对更便宜。实际墙钟开销还取决于矩阵尺寸、GPU GEMM 利用率、是否批处理/并行，以及通信能否与计算重叠。

### 数值检查

实现或调试时至少记录：

- `isfinite(X)` 与更新 RMS；
- $\|XX^\top-I\|_F$ 或 $\|X^\top X-I\|_F$；
- 相对精确 SVD polar factor 的 cosine similarity；
- 输入动量的奇异值范围、条件数和 effective rank；
- BF16 与 FP32 正交化结果差异。

## 4. 完整更新与两种缩放约定

极分解方向的元素 RMS 与形状有关。对满秩 $A\times B$ 矩阵，理想 Muon 方向的理论 RMS 为

$$\operatorname{RMS}(O)=\frac{1}{\sqrt{\max(A,B)}}.$$

因此只写 $W\leftarrow W-\eta O$ 会让不同形状的层获得不同更新尺度。常见约定有：

| 约定 | 形状缩放 | 常见目的 |
|---|---|---|
| Keller 原始约定 | $\sqrt{\max(1,A/B)}$ | 保持早期 Muon/nanoGPT 配方的尺度 |
| Moonshot `match_rms_adamw` | $0.2\sqrt{\max(A,B)}$ | 把各矩阵 update RMS 调到约 0.2，并复用 AdamW 的 LR/WD |

不同矩阵朝向会影响第一种写法中的 $A/B$。必须按所用代码的 shape 定义核对，不要只复制公式。

加入解耦 weight decay 后，可概括为

$$W_{t+1}=(1-\eta\lambda)W_t-\eta\,s(A,B)\,O_t.$$

Moonlight 的长程实验表明：无 weight decay 的 vanilla Muon 前期很快，但部分权重会持续长大；加入 weight decay 后在 over-training 区间更稳定。这也是现代大规模配方不可省略的一环。

## 5. 参数路由：Muon 实际上通常是混合优化器

推荐的起始规则：

| 参数 | 默认优化器 | 原因 |
|---|---|---|
| attention 的 Q/K/V/O projection | Muon | 隐藏层二维线性算子；Q、K、V 最好按语义分开正交化 |
| MLP up/gate/down projection | Muon | 隐藏层二维线性算子 |
| MoE expert linear weights | Muon | 同上，但需处理 expert/并行分片 |
| embedding、LM head | AdamW | 虽是二维张量，但语义不是普通 hidden-to-hidden operator |
| bias、LayerNorm/RMSNorm scale | AdamW | 一维，标准 Muon 没有自然矩阵结构 |
| 卷积核、高阶张量 | 先用 AdamW；做专项实验后再 reshape/指定维度 | reshape 方式本身定义了优化几何 |

常见坑：

- 仅按 `p.ndim == 2` 自动选择会错误地包含 embedding/LM head。
- fused QKV 若作为一个大矩阵整体正交化，与分别处理 Q/K/V 并不等价。
- tied embedding/LM head 不能被两个优化器重复持有。
- tensor parallel shard 上局部正交化通常不等于对完整矩阵正交化。
- 梯度累积应先得到本次 optimizer step 的最终梯度，再更新 momentum 和做 NS。

## 6. 与相关优化器的关系

| 方法 | 状态/核心操作 | 与 Muon 的关系 |
|---|---|---|
| SGD + momentum | 一阶动量；直接更新 | Muon 在它之后加入矩阵正交化 |
| AdamW | 一阶、逐元素二阶矩；坐标自适应 | Muon 状态更少，但缺少逐坐标方差适配；非矩阵参数常由 AdamW 补位 |
| Shampoo | 累积行/列二阶统计并做逆矩阵幂 | 去掉统计累积时，其更新可化为 $UV^\top$；Muon 可看作便宜的 instantaneous Shampoo |
| SOAP | Shampoo 特征基 + Adam | 与 Muon 同属矩阵结构预条件路线，但状态和机制不同 |
| Scion | 范数约束 LMO | 与 Muon 共享非欧氏/谱范数优化视角，不是 Muon 的简单别名 |
| Dion | 低秩 power iteration 的分布式正交更新 | 目标相近，重点解决正交化与分布式成本 |

Muon 与 Shampoo 的代数联系不意味着两者训练动力学相同。Shampoo 累积历史二阶统计；Muon 通常只保存 momentum，并对当下动量做 polar 近似。

## 7. 关键论文与推荐阅读顺序

### 第一层：先形成正确心智模型

1. [Keller Jordan, *Muon: An optimizer for hidden layers in neural networks*](https://kellerjordan.github.io/posts/muon/)<br>
   原始设计说明；读算法、Newton–Schulz、与 Shampoo 的关系及运行开销。
2. [Bernstein & Newhouse, *Old Optimizer, New Norm: An Anthology*](https://arxiv.org/abs/2409.20325)<br>
   从 norm/steepest descent 视角理解 Adam、Shampoo 与新优化器设计空间。
3. [Bernstein & Newhouse, *Modular Duality in Deep Learning*](https://arxiv.org/abs/2410.21265)<br>
   理解为什么不同层应使用与其功能相符的范数，以及 Newton–Schulz 如何成为 GPU 友好的 dualization。
4. [Gupta, Koren & Singer, *Shampoo*](https://proceedings.mlr.press/v80/gupta18a.html)<br>
   Muon 的重要前史；重点读矩阵预条件器与状态/计算复杂度。

### 第二层：规模化与工程证据

5. [Liu et al., *Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982)<br>
   必读。关键贡献是 weight decay、consistent update RMS、分布式 Muon。论文在其 compute-optimal scaling-law 设置中报告：达到 AdamW 同等表现约需 52% 训练 FLOPs，并训练了 3B active / 16B total MoE Moonlight。
6. [Essential AI, *Practical Efficiency of Muon for Pretraining*](https://arxiv.org/abs/2505.02222)<br>
   最高到 4B 参数，重点研究大 batch、critical batch size、compute-time Pareto frontier 与 muP。
7. [Kimi Team, *Kimi K2: Open Agentic Intelligence*](https://arxiv.org/abs/2507.20534)<br>
   MuonClip/QK-Clip 的来源；报告在 1T total、32B active MoE 上用 15.5T tokens 完成无 loss spike 预训练。
8. [Shulgin et al., *Beyond the Ideal: Analyzing the Inexact Muon Update*](https://arxiv.org/abs/2510.19933)<br>
   把 NS 近似误差纳入理论，说明精度、学习率和 momentum 必须联合考虑。

### 第三层：理论深化

9. [Chen, Li & Liu, *Muon Optimizes Under Spectral Norm Constraints*](https://arxiv.org/abs/2506.15054)<br>
   从 Lion-$\mathcal K$ 视角分析 decoupled weight decay 下的隐式谱范数约束。
10. [Sato et al., *Convergence Bound and Critical Batch Size of Muon Optimizer*](https://arxiv.org/abs/2507.01598)<br>
    覆盖 Nesterov/weight decay 的组合，并分析 critical batch size。

阅读论文时，把结论分别记到四栏：`方向/几何`、`数值近似`、`训练统计`、`系统成本`。Muon 文献最容易犯的错误，就是用某一栏的改善替另一栏做结论。

## 8. 工程实现索引

| 实现 | 适用场景 | 阅读重点 |
|---|---|---|
| [KellerJordan/Muon](https://github.com/KellerJordan/Muon) | 最小 PyTorch 参考、理解算法 | NS 函数、参数分组、单机与分布式类 |
| [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | 端到端 speedrun 配方 | Muon 与架构、数据、kernel 优化如何共同作用；不要把全部收益归因于优化器 |
| [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) | 大规模分布式实现与模型检查点 | ZeRO 风格状态分片、all-gather 完整矩阵后正交化、通信/计算重叠 |
| [`torch.optim.Muon`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) | 当前 PyTorch 用户的首选起点 | `adjust_lr_fn`、`ns_steps`、参数限制；当前实现仍无 fused/foreach 快速路径 |
| [`optax.contrib.muon`](https://optax.readthedocs.io/en/stable/_collections/examples/contrib/muon.html) | JAX/Flax | mask、PyTree 参数路由、高阶张量 dimension specification |
| [本地 Megatron-LM Muon 实现解析](/notes/megatron-muon-implementation/) | 本仓库 submodule 代码走读 | 参数路由、TensorParallelMuon、LayerWise 分布式布局、DDP buffer sync、可改进点 |
| [Megatron-Core emerging optimizers](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.emerging_optimizers.html) | TP/DP 大模型训练 | TensorParallelMuon/AdaptiveMuon、split QKV、TP mode、非线性参数自动路由 |
| [Megatron-Core QK-clip](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.qk_clip.html) | MuonClip 稳定性机制 | attention max logit 监控与参数裁剪 |
| [Gram Newton–Schulz / GramMuon](https://dao-lab.ai/blog/2026/gram-newton-schulz/) | 优化 NS kernel 成本 | 在较小 Gram 矩阵上迭代、半精度数值不稳定与 restart；作者报告正交化阶段 40–50% 加速 |

### 最小 PyTorch 使用骨架

以下代码只表达参数路由，不是完整训练配方：

```python
hidden_matrix_params = []
other_params = []

for name, p in model.named_parameters():
    is_hidden_matrix = (
        p.ndim == 2
        and "embed" not in name
        and "lm_head" not in name
    )
    (hidden_matrix_params if is_hidden_matrix else other_params).append(p)

muon = torch.optim.Muon(
    hidden_matrix_params,
    lr=muon_lr,
    momentum=0.95,
    nesterov=True,
    weight_decay=weight_decay,
    ns_steps=5,
    adjust_lr_fn="match_rms_adamw",
)
adamw = torch.optim.AdamW(
    other_params,
    lr=adamw_lr,
    betas=(0.9, 0.95),
    weight_decay=weight_decay,
)
```

生产代码应按模块类型而非字符串可靠地分组，并断言：每个 trainable parameter 恰好属于一个 optimizer group。通常还要把 AdamW 组继续拆成“需要 weight decay”和“bias/norm 等不衰减”两组；示例为了只突出 Muon 路由而省略了这层配置。

## 9. 变体与演进路线

下表的“结果”均是各自论文报告，不应被视为跨代码库、跨模型的直接排名。

| 方向 | 代表工作 | 改了什么 | 当前判断 |
|---|---|---|---|
| 超大规模稳定性 | [MuonClip / Kimi K2](https://arxiv.org/abs/2507.20534) | 监控每个 attention head 的最大 QK logit，超阈值时在 optimizer update 后缩放 Q/K 权重 | 已有 1T MoE 长程训练证据；它是 Muon + 模型感知稳定机制，不只是优化器公式 |
| 神经元级自适应 | [NorMuon](https://arxiv.org/abs/2510.05491) | 正交化后维护 neuron/row-wise 二阶统计并归一化 | 论文在 1.1B 预训练上报告相对 Muon 11.31% 训练效率改善；仍需更多独立大规模复现 |
| 自适应更新 | [AdaMuon](https://arxiv.org/abs/2507.11005) | 对正交化后的更新引入自适应二阶尺度 | 与 NorMuon 同属“polar 后适配”路线 |
| polar 前预条件 | [Muon² / Muon²-F](https://arxiv.org/abs/2604.09967) | 在 NS 前用 Adam-style 二阶矩改善动量矩阵条件；F 为因子化低内存版 | 60M–1.3B GPT/LLaMA 实验报告可把 NS 从 5 步降到 3 步，并改善方向逼近；前沿预印本 |
| 方差自适应 | [Muon-NSR / Muon-VS](https://arxiv.org/abs/2601.14603) | 正交化前对 momentum 做 noise-to-signal 或 variance scaling | 连接 Adam 的 variance adaptation 与 Muon；前沿预印本 |
| 状态低精度 | [8-bit Muon](https://arxiv.org/abs/2509.23106) | blockwise 量化 momentum state | 最高 2.7B 实验报告与 Muon 接近，同时最多减少 62% optimizer-state footprint |
| 全参数/微调 | [MuonAll](https://arxiv.org/abs/2511.06086) | 把包括一维参数在内的参数 reshape 成二维后纳入 Muon | 实验规模到约 0.5B，更多是可行性信号；“如何 reshape”本身是强归纳偏置 |
| 低秩适配器 | [Riemannion](https://arxiv.org/abs/2507.12142) | 在固定秩流形上推广 Muon，用于 LoRA | 更接近 Riemannian optimizer，不宜简单当作标准 Muon 开关 |
| 正交化 kernel | [GramMuon](https://dao-lab.ai/blog/2026/gram-newton-schulz/) | 把大部分迭代移到较小 Gram 矩阵，并为低精度加入 restart | 算法方向近似等价，主要收益是系统加速；很值得工程跟踪 |
| 分布式系统 | [DMuon](https://arxiv.org/abs/2606.27153) | drop-in 分布式 Muon、优化正交化与通信 | 很新的预印本；作者报告 optimizer step 接近 AdamW 开销，需结合代码与硬件复现 |

可以用两个轴理解这些工作：

1. **更新统计轴**：momentum → row-wise variance → full/factorized second moment；
2. **polar 实现轴**：标准 quintic NS → 更好的多项式/少步数 → Gram/restart → 分布式/缓存/低秩近似。

不要把这两个轴混为一谈。前者改变训练动力学，后者有时只想更便宜地逼近同一个方向。

## 10. 已知边界、争议与开放问题

### 10.1 预训练收益不自动迁移到微调

[Can Muon Fine-tune Adam-Pretrained Models?](https://arxiv.org/abs/2605.10468) 报告，Adam 预训练后直接切换 Muon 全量微调可能因隐式偏置不同而退化；限制更新强度（例如 LoRA）可缓解差距。因此实验必须区分：

- Muon 从头预训练；
- Muon 预训练模型继续训练；
- AdamW 预训练模型切换 Muon；
- full fine-tuning、LoRA、SFT、RL。

### 10.2 精确 polar 不一定是最好训练更新

原始五次多项式的目标是快速产生有用的谱变换，而不是数值上最精确的 polar factor。更精确的 SVD 方向未必在固定训练预算下更好。应把“正交误差”和“最终 loss/time”同时测量。

### 10.3 token efficiency、FLOP efficiency 与 wall-clock efficiency 不同

- 更少 token 达到目标 loss：data/token efficient；
- 更少训练 FLOPs：compute efficient；
- 更短现实时间：wall-clock efficient；
- 更少 optimizer state：memory efficient。

Muon 可能在第一项赢、却因 NS/通信在第三项丢。任何“加速 X%”都必须先问分母是什么。

### 10.4 公平基线很难

Muon 与 AdamW 的最佳 LR、beta、weight decay、batch size 和 schedule 可能不同。只替换 optimizer 而完全不调参，测到的是“配方迁移能力”；分别充分调参，测到的才更接近“方法上限”。两种实验都合理，但回答的问题不同。

### 10.5 仍未解决的问题

- 哪些架构/数据分布真正受益，哪些只是 Transformer 预训练特例？
- polar 的关键是方向、谱压平程度，还是由此产生的隐式正则化？
- 每层/每种矩阵是否应该使用不同范数、不同 NS 精度和不同更新 RMS？
- Muon 的大 batch 优势是否在更长 horizon、更强数据重复下持续？
- QK-Clip 是 Muon 特有补丁，还是任何高效大模型训练都应具备的安全护栏？
- 如何在 tensor/expert/sequence parallel 下做到数学等价且通信近似 AdamW？
- Muon 预训练 checkpoint 的 SFT/RL 最佳延续策略是什么？

## 11. 推荐复现实验

### 实验 A：先验证数学与实现

随机生成方阵和长方形矩阵，对比：

1. `torch.linalg.svd` 得到的 $UV^\top$；
2. FP32 quintic NS，1/3/5/7 步；
3. BF16 quintic NS；
4. Gram Newton–Schulz（若硬件/依赖允许）。

输出 singular values、orthogonality error、cosine similarity、时间和峰值内存。

### 实验 B：小模型可控 A/B

用同一 GPT 配置、数据顺序与 token budget 比较：

- tuned AdamW；
- original-scale Muon + AdamW；
- match-RMS Muon + AdamW；
- Muon 去掉 weight decay；
- NS 3/5 steps；
- QKV 合并与拆分。

至少报告：train/validation loss vs token、FLOPs、wall time，tokens/s，optimizer-step time，峰值显存，更新 RMS、权重 RMS、grad norm、attention max logit、checkpoint 下游评测。

### 实验 C：规模和 batch 扫描

在 2–3 个模型宽度和多个 global batch size 上复用 muP 或严格做小规模调参，检查：

- critical batch size 是否右移；
- 最优 LR/WD 能否转移；
- NS/通信占 step time 的比例怎样随规模变化；
- 达到目标 loss 的总 GPU-hours 是否真的降低。

### 实验记录模板

```text
commit / environment:
model / parameterization:
dataset / tokenizer / exact token count:
optimizer routing:
Muon: lr, momentum, nesterov, wd, scaling, ns coefficients, ns steps, dtype
AdamW: lr, betas, eps, wd
schedule / warmup / batch / grad accumulation:
parallelism / hardware:
result: final loss, target-loss tokens, FLOPs, wall time, peak memory
stability: non-finite, loss spike, grad norm, max QK logit
```

## 12. 建议学习路线

### 第 1 阶段：算法本体（2–3 天）

- 手推 SVD → polar factor → 谱范数最陡方向。
- 阅读 Keller 原文与 `KellerJordan/Muon` 的 NS 函数。
- 完成实验 A，确认自己理解“近似正交”而非只会调用 optimizer。

### 第 2 阶段：规模化配方（3–5 天）

- 精读 *Muon is Scalable for LLM Training* 的第 2 节、分布式算法和 update RMS 附录。
- 读 *Practical Efficiency of Muon for Pretraining*，区分 token/FLOP/time。
- 完成实验 B。

### 第 3 阶段：理论与邻近方法（1 周）

- 读 *Old Optimizer, New Norm*、*Modular Duality*、Shampoo。
- 再读谱范数约束、critical batch size、inexact Muon 三篇理论文。
- 用一页纸回答：“Muon 到底是一阶、二阶，还是预条件法？”

### 第 4 阶段：前沿与系统（持续）

- 根据目标选一条：MuonClip 稳定性、NorMuon/Muon² 自适应、8-bit 内存、GramMuon/DMuon 系统。
- 在自己的训练栈中做端到端 profile，不以 nanoGPT speedrun 单项结果代替生产判断。
- 每月按 `arXiv: Muon optimizer`、PyTorch/Megatron release note 和核心仓库 commit 更新本文。

## 13. 术语速查

- **polar factor**：$M=UH$ 或 SVD $M=U\Sigma V^\top$ 中的半正交方向 $UV^\top$。
- **semi-orthogonal**：长方形矩阵不可能两边都等于单位阵；只有行或列正交。
- **spectral norm**：最大奇异值 $\|M\|_2=\sigma_{max}(M)$。
- **nuclear norm**：奇异值之和，是谱范数的对偶范数。
- **Schatten-$p$ norm**：对奇异值向量取 $\ell_p$ 范数；$p=2$ 为 Frobenius，$p=\infty$ 为谱范数。
- **update RMS**：更新张量元素平方均值的平方根，不等同于 grad norm。
- **orthogonalization quality**：可用正交误差、奇异值分布或与精确 polar 的方向相似度衡量。
- **optimizer state**：参数之外为更新维护的状态；标准 Muon 主要是一阶 momentum，AdamW 通常有一阶和二阶矩。
- **critical batch size**：继续增大 batch 后，减少优化步数的收益开始明显饱和的尺度。
- **optimizer mismatch**：预训练和微调优化器的几何/隐式偏置不同，切换时可能破坏已有表示。

## 14. 参考入口与证据分级

优先级约定：A = 官方实现或超大规模技术报告；B = 多尺度系统实验；C = 新预印本/待独立复现。

| 资料 | 级别 | 用途 |
|---|---:|---|
| [Muon 原始说明](https://kellerjordan.github.io/posts/muon/) | A | 定义、设计史、NS 与 Shampoo 关系 |
| [KellerJordan/Muon](https://github.com/KellerJordan/Muon) | A | 最小参考实现 |
| [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) | A | scaling law、Moonlight、分布式 Muon |
| [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) | A | 分布式代码与模型资产 |
| [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) | A | MuonClip 的超大规模训练案例 |
| [PyTorch Muon API](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) | A | 生产 API 的参数与当前限制 |
| [Practical Efficiency of Muon](https://arxiv.org/abs/2505.02222) | B | 多规模、大 batch、muP |
| [NorMuon](https://arxiv.org/abs/2510.05491) | B/C | neuron-wise adaptive variant |
| [8-bit Muon](https://arxiv.org/abs/2509.23106) | B/C | optimizer state 量化 |
| [Muon²](https://arxiv.org/abs/2604.09967) | C | polar 前二阶预条件与少步 NS |
| [DMuon](https://arxiv.org/abs/2606.27153) | C | 最新分布式工程方向 |

---

维护建议：后续不要把所有内容继续堆在单文件里。完成第一轮学习后，可拆成 `notes/math.md`、`notes/papers.md`、`notes/implementations.md`、`notes/experiments.md`，本文保留为总索引。
