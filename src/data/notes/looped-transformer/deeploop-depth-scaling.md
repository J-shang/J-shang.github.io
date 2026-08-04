---
title: "DeepLoop: Depth Scaling for Looped Transformers：逐篇解析"
description: "追踪共享参数重复访问带来的 residual scaling 问题，复原 visit alignment、DeepLoop 配方与单种子证据。"
topic: "looped-transformer"
section: "frontiers"
slug: "deeploop-depth-scaling"
date: 2026-07-29
updated: 2026-08-04
cutoff: 2026-07-24
order: 41
source:
  repository: "J-shang/looped-transformer"
  path: "papers/12-deeploop-depth-scaling.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-04"
  contentHash: "sha256:9f62fd04381c1f8d7d324e08281aabae738fb83ad5dd33ded716ba9be14f4b4b"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
## 论文身份与证据范围

- 论文：*DeepLoop: Depth Scaling for Looped Transformers*
- 作者：Shuzhen Li、Yifan Zhang、Jiacheng Guo、Quanquan Gu、Mengdi Wang
- 分析版本：arXiv:2607.13491v1，2026-07-15
- 发表状态：arXiv preprint
- 主来源：[arXiv](https://arxiv.org/abs/2607.13491)、[HTML 全文](https://arxiv.org/html/2607.13491)、[PDF](https://arxiv.org/pdf/2607.13491)
- 官方项目：[lszshu/DeepLoop](https://github.com/lszshu/DeepLoop)
- 阅读范围：正文、Proposition 3.1、Figure 1–3、Table 1–3、Appendix A–D；HTML 丢失的公式符号按 PDF 语义与正文定义交叉核对
- 信息截止：2026-07-24；论文发布仅 9 天，外部复验仍不足

## 30 秒结论

**[论文报告]** 普通 DeepNorm 的 residual scaling 假设每个深度位置有自己的参数。Looped Transformer 中，同一物理参数被 $R$ 次 visit 的梯度共同写入，更新后又被 $R$ 次 visit 共同读取，形成“写入 × 读取”的双重聚合。论文用 visit-alignment coefficient 描述这些项的相关性：decorrelated 时恢复 DeepNorm 的 $p=1/4$；固定物理深度、fully aligned 时需要 $p=1/2$。由此给出 Post-LN/RMSNorm 的 DeepLoop：

$$
\alpha=(2N)^{1/2},
\qquad
\beta=(8N)^{-1/2},
$$

其中 $N$ 是 unrolled block depth，$\alpha$ 缩放 skip path，$\beta$ 是指定 residual-branch matrices 的 initialization gain。

**[综合判断]** 这是 loop-specific initialization/residual parameterization，不是新的 loop architecture。理论给的是一阶、最坏对齐条件下的充分稳定性界；实验证据支持 $p=1/2$ 是论文设定中的可靠默认值，但尚未直接测量核心的跨 visit alignment。

## 5 分钟论文地图

```text
DeepNorm 按“展开深度”缩放，但默认每层参数独立
  → weight tying 让一次 optimizer update 汇总多个 visits
  → 更新后的同一 tensor 又被多个 visits 读取
  → 用 κR 量化 double sum 的对齐程度
  → 推导 p 的稳定阈值并选保守 p=1/2
  → GPT-2 LM、8-task eval、HRM/ARC-AGI 检验
```

阅读顺序：

1. Figure 1、§2.1：physical depth 与 unrolled depth。
2. §2.3、§3.4：双重聚合与 $\kappa_R$。
3. Proposition 3.1：exponent threshold。
4. §5.1/Table 1：最干净的 controlled LM evidence。
5. Appendix C/Figure 6：多 seed $p$-sweep。
6. §5.3/Table 3：迁移到 hierarchical reasoner。

前置知识：Post-LN/RMSNorm、residual initialization、first-order perturbation、gradient alignment。最小例子是一个参数 $\theta$ 在两次 loop 中使用：更新为 $g_1+g_2$，下一 forward 的两个位置都对这同一个更新敏感，输出变化含四项 $S_i g_j$，而 untied 深度只有两项 $S_i g_i$。

## 符号与约定

| 符号 | 含义 | 类型、形状与范围 | 状态 |
|---|---|---|---|
| $B$ | 保存的 physical Transformer blocks | 正整数 | 架构超参数 |
| $R$ | 对整个 block stack 的 rounds/visits | 正整数 | loop 超参数 |
| $N=BR$ | unrolled block depth | block calls | 派生量 |
| $2N$ | attention/MLP residual-sublayer visits | 正整数 | 派生量 |
| $\theta_b$ | 第 $b$ 个物理 residual branch 参数 | matrix/tensor | 可训练且跨 rounds 共享 |
| $g_{b,r}$ | visit $(b,r)$ 对 $\theta_b$ 的有效更新贡献 | 与 $\theta_b$ 同 shape | gradient-derived |
| $S_{b,r}$ | visit 对参数扰动到最终输出的局部 sensitivity | 线性算子 | runtime derivative |
| $\kappa_R$ | 同一物理 branch 跨 visits 的 alignment coefficient | 无量纲；decorrelated $O(1)$，aligned $\Theta(R)$ | 派生诊断量 |
| $\alpha$ | skip/residual-stream scale | 正标量 | architecture constant |
| $\beta$ | 指定 branch matrices 的 initialization gain | 正标量 | initialization constant |
| $p$ | scaling exponent | 非负标量 | 设计超参数 |

“visit”是同一物理参数的一次 unrolled 调用。$\beta$ 只在初始化时应用到 DeepNorm 指定矩阵，不是每次 forward 再乘一次的 runtime coefficient（§2.2、§3.1）。

## 贡献账本

| 可检查贡献 | 类型与最近基线增量 | 支持位置 | 没有建立 |
|---|---|---|---|
| tied-depth 双重聚合机制 | 理论 framing/mechanism | §2.3、Eq. 11 | 所有现实 visits 都 fully aligned |
| visit-alignment coefficient 与一阶 bound | 理论工具 | §3.4、Prop. A.5 | 必要条件或非线性全程稳定保证 |
| $p=1/2$ aligned threshold | 条件性定理 | Proposition 3.1、Corollary A.6 | Pre-LN/任意 norm 同样适用 |
| DeepLoop one-line scaling | 参数化方法 | §3.5 | 无需单独调学习率 |
| LM 与 HRM 改进 | 经验发现 | Table 1–3、Appendix C | 大规模/多架构普遍收益 |

## 方法与推导复原

### 1. Baseline：DeepNorm

对 decoder-only 的 $N$ 个 unrolled blocks、$2N$ 个 residual sublayers，论文采用 DeepNorm 风格 Post-LN：

$$
h^+=\operatorname{RMSNorm}\!\left(\alpha h+F_{\theta}(h)\right).
$$

这里 $h$ 是 runtime residual state，$F_\theta$ 是 attention 或 MLP branch，$\theta$ 是 learned matrix set，$\alpha$ 是固定 skip scale。

DeepNorm 采用 $p=1/4$ 的 scaling family；DeepLoop 保持同一架构和常数，只把 exponent 改为 $1/2$。用统一记法：

$$
\alpha=(2N)^p,\qquad \beta=(8N)^{-p}.
$$

这里 $\beta$ 乘到指定 branch matrices 的初始权重；不在每个 visit 重复乘。$p=1/4$ 是 untied DeepNorm，$p=1/2$ 是论文的 conservative tied-depth choice。

### 2. 为什么 weight sharing 不只是把 $N$ 代入公式

Untied depth 的一阶输出变化示意为：

$$
\Delta y_{\text{untied}}
\approx\sum_{v=1}^{2N}S_v g_v.
$$

每个 visit $v$ 有自己的参数，所以只读取自己的 update。$S_v$ 是参数扰动到最终输出的 local sensitivity，$g_v$ 是该 visit 对参数的 optimizer update contribution。

对一个被访问 $R$ 次的共享参数：

$$
\Delta\theta_b\propto\sum_{r'=1}^{R}g_{b,r'},
$$

而该更新被全部 $R$ 个 visits 读取：

$$
\Delta y_{\text{tied}}
\approx\sum_{r=1}^{R}
S_{b,r}
\left(\sum_{r'=1}^{R}g_{b,r'}\right).
$$

这就是 double sum。若项互相抵消/正交，总量接近随机和；若同向，则可接近 $R^2$ 个 coherent pair 的累积。

**两次 loop 算例。** $R=2$ 时 untied 只有 $S_1g_1+S_2g_2$；tied 有 $S_1g_1+S_1g_2+S_2g_1+S_2g_2$。新增 cross terms 正是普通 depth analysis 没有的部分。

### 3. Visit alignment

论文用 $\kappa_R$ 压缩 double sum 的对齐程度（§3.4、Eq. 12–14）：

- nearly orthogonal / decorrelated visits：$\kappa_R=O(1)$；
- fully aligned visits：$\kappa_R=\Theta(R)$。

在论文的 DeepNorm-scale local-sensitivity assumption 下，bound 导出 scaling family 的 exponent threshold（Proposition 3.1）：

- decorrelated：$p\ge 1/4$；
- fixed physical depth、aligned $R\to\infty$：$p\ge 1/2$。

**证据边界。** 这是 first-order sufficient bound。它不证明 $p<1/2$ 必然发散，也不证明 $p=1/2$ 在所有 optimizer、norm placement 与训练阶段都最优。

### 4. DeepLoop 配方

取最小 conservative exponent $p=1/2$：

$$
\alpha=(2N)^{1/2},\qquad
\beta=(8N)^{-1/2}.
$$

实现步骤：

```text
N = physical_blocks × rounds
for every DeepNorm-designated attention/MLP matrix:
    initialize W = beta × W_base
for every Post-LN residual sublayer:
    output = RMSNorm(alpha × skip + branch_W(input))
```

它不添加 gate、learned residual scalar、auxiliary loss 或新的 runtime branch（§3.5）。

![DeepLoop physical blocks, unrolled visits, residual rule, and scaling rule](/assets/looped-transformer/12-deeploop-depth-scaling/figure-1-deeploop-overview.png)

*原图：Figure 1，PDF p. 2；来源：arXiv:2607.13491v1。看图重点：(a) 物理 block 只存一份；(b) 每轮重复访问形成更深 unrolled execution；(c) 每个 residual sublayer 使用同一 $\alpha,\beta$ 参数化；(d) DeepLoop 取 $p=1/2$。图把 stored depth、effective depth 和 residual scaling 放在同一账本中，但 $p=1/2$ 的合理性仍来自条件性一阶推导，不是示意图本身。*

### 5. Hierarchical recurrence

§4 把同一推导扩展到高/低层模块和 one-step gradient truncation。关键变化是使用 gradient-visible visits，而不是所有 forward-visible visits。每个模块可有不同的 alignment regime；共享的 inner recurrent module仍落到 aligned $p=1/2$ 约束。

**[综合判断]** 这个扩展的重要性在于：residual scaling 应按“哪些 visits 真正写入 gradient”核算，而不只是画 forward 展开图。

## 实验证据：问题—设置—结果—边界

### 1. FineWeb-Edu validation loss

相同 GPT-MHA-RoPE backbone、optimizer、data pipeline、seed；50B tokens、100K steps、context 1024、global batch 480。small 用 4×H200 141GB，medium 用 8×H200 141GB（§5.1）。Table 1 的四列依次为 $R=1,3,5,7$：

| Backbone | 方法 | $R=1$ | $R=3$ | $R=5$ | $R=7$ |
|---|---|---:|---:|---:|---:|
| GPT-2 small | baseline | 2.8627 | 2.8077 | 2.7910 | 2.7700 |
| GPT-2 small | DeepLoop | 2.8631 | 2.7917 | 2.7679 | 2.7514 |
| GPT-2 medium | baseline | 2.6253 | 2.5779 | 2.5640 | 2.5558 |
| GPT-2 medium | DeepLoop | 2.6264 | 2.5627 | 2.5444 | 2.5280 |

**支持结论：** $R=1$ 时基本中性，启用共享 revisit 后 DeepLoop 在两规模的单 seed run 中更低。

**剩余不确定性：** 主矩阵是 single-seed；论文明确说需要 multi-seed 才能量化方差（§5.1）。

![Validation loss versus loop count for baseline and DeepLoop](/assets/looped-transformer/12-deeploop-depth-scaling/figure-2-loop-count-validation-loss.png)

*原图：Figure 2，PDF p. 12；来源：arXiv:2607.13491v1。看图重点：在 small 与 medium backbone 上，$R=1$ 时两者近似持平，$R\ge3$ 后 DeepLoop 的 validation loss 更低且差距随 loop count 扩大。这与“问题来自共享 revisit”相容；caption 明确这些是 single-seed runs，因此不能从曲线间距推断统计显著性。*

### 2. Downstream transfer

medium checkpoint 在 8-task lm-eval-harness、default `acc`、0/1-shot、bf16、context 1024 上评估（§5.2）。$R=7$：

- 0-shot Avg：baseline 52.95，DeepLoop 53.88；
- 1-shot Avg：baseline 54.62，DeepLoop 55.20（Table 2）。

这支持 validation-loss advantage 能部分迁移，但 1-shot 在某些较小 $R$ 上排名并不单调，不能概括为每项都赢。

### 3. ARC-AGI / HRM

只改 residual parameterization，保持 backbone、halting、AdamATan2、data、100K epochs 与 voting protocol（§5.3）。Table 3 的 voting ladder：

| 方法 | budget 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Vanilla HRM | 31.50 | 36.50 | 41.50 | 47.50 | 50.75 |
| DeepLoop | 35.50 | 39.75 | 44.25 | 49.75 | 51.50 |

它支持方法跨出普通 LM backbone；但投票预算会改变 test-time compute，不能只取最好列与别的协议横比。

### 4. $p$-sweep

Appendix C 在单一 GPT-2 small、固定 loop depth/短 budget 上，对 7 个 $p$ 值最多 5 seeds 扫描。论文报告阈值附近存在“部分 seed 训练、部分失败”，$p=1/2$ 是最小可靠稳定值；更大 $p$ 虽稳定但学习更保守。作者同时明确该 sweep 只有一个规模、一个 loop depth、一个 step budget。

### Claim–evidence map

| Claim | 证据 | 强度 | Gap |
|---|---|---|---|
| weight tying 改变一阶 residual bound | double-sum derivation、Prop. A.5 | strong（假设内） | local-sensitivity assumption 未在模型中测量 |
| aligned regime 阈值为 $p=1/2$ | Proposition 3.1 | strong（asymptotic sufficient condition） | 非必要性、有限深度最优性 |
| DeepLoop 在 recurrence 开启时有益 | Table 1–2 | moderate | 主 LM 矩阵 single-seed |
| 可迁移 hierarchical reasoner | Table 3、有限 seed control | moderate–strong | 单架构/数据/任务 |
| visits 实际 aligned | 架构动机与效果 | weak–moderate | 未直接测 $\kappa_R$ |

## 关键洞察

1. **[论文报告] 参数共享同时改变 backward 写入与 next-forward 读取。** 只按 unrolled depth 套 untied 公式会漏掉 cross-visit terms。
2. **[复原推导] 更保守 scaling 是稳定—学习速度交换。** 更大的 $p$ 缩小 branch initialization；Appendix C 的已收敛 runs 中，更保守可能 loss 略高。
3. **[综合判断] 最关键的缺失诊断是实测 $\kappa_R$。** 若不同层、训练阶段、任务的 alignment 差异很大，统一 $p=1/2$ 可能过于保守。
4. **[综合判断] $N$ 不是唯一深度。** physical depth、forward visits、gradient-visible visits 与 optimizer reuse pattern 都应分别核算。

## 局限与开放问题

- 理论是一阶 local perturbation bound，不是完整非线性训练收敛定理；
- DeepNorm-scale local-sensitivity 是假设；
- $\kappa_R$ 没有在主实验直接测量；
- 主 LM 表格 single-seed，尺度止于 GPT-2 small/medium；
- $p$-sweep 只有一个小规模、一个 loop depth、一个短预算（Appendix C）；
- 架构限于 Post-LN/RMSNorm 风格；Pre-LN、其他 parameterization 未覆盖；
- 50B-token training 与 H200 环境仍不足以代表旗舰 LLM；
- 论文发布极新，独立复验与同行评审尚缺。

## 超出论文：直接测 alignment 并自适应选 $p$

**[扩展假设] Proposal：** 训练时周期性记录同一物理 branch 各 visit 的 gradient/sensitivity sketch，估计 $\kappa_R$，再选择 per-block $p_b$。

- Reasoning chain：理论阈值由 alignment 决定；若早层/晚层或训练早晚 alignment 不同，统一最坏值会浪费学习信号。
- Predicted observation：高 alignment blocks 需要接近 $1/2$，低 alignment blocks 用 $1/4$–$1/2$ 间值可更快收敛。
- Falsification condition：估计 alignment 与稳定边界/最终 loss 无预测关系。
- Minimum experiment：GPT-2 small，$R=3,5,7$，对每 block 记录 visit gradient cosine 与低秩 sensitivity proxy；预注册 $p_b$ rule 再测 held-out seeds。
- Cost/risk：per-visit gradient 收集昂贵且改变显存；应使用随机投影 sketch，避免形成新的大开销方法。

## 复现与阅读路径

1. Figure 1：先写出 $B,R,N,2N$。
2. §2.3/§3.4：手推 $R=2$ 的四个 cross terms。
3. Proposition 3.1：明确是 sufficient exponent threshold。
4. Table 1：先复现 $R=1$ neutral check，再跑 $R\in\{3,5,7\}$。
5. Appendix C：至少 5 seeds 画“失败率 + 条件于收敛的 loss”，不要只报均值。
6. sanity checks：确认 $\beta$ 只用于初始化、$\alpha$ 用在 skip、baseline 共享拓扑完全相同。

## 一句话带走

**DeepLoop 的核心发现是：共享参数被多次写入又多次读取，残差稳定性取决于 visit alignment；$p=1/2$ 是 aligned Post-LN loop 的保守充分缩放，而不是所有 looped 模型的普遍常数。**
