---
title: "Kimi K3：把 3T-class MoE、1M context 与长时程 agent RL 做成一个系统"
description: "从 hybrid attention、Block AttnRes 与 Stable LatentMoE 出发，审计 Kimi K3 的训练、agent RL、基础设施、评测证据和披露边界。"
topic: "tech-reports"
section: "report-readings"
slug: "kimi-k3"
date: 2026-07-28
cutoff: 2026-07-28
featured: true
order: 10
readtime: 40
---

> 论文：**Kimi K3: Open Frontier Intelligence**<br>
> 作者：Kimi Team<br>
> 状态：Moonshot AI 官方 technical report；PDF 未注明会议或期刊<br>
> 模型发布日期：2026-07-16<br>
> Technical report 版本：官方仓库 `main` 上的 47 页 PDF；PDF metadata 创建日期为 2026-07-27<br>
> 本次分析版本锚点：用户提供的 `k3_tech_report.pdf`，SHA-256 `38621eb5be601a5dcd5c795fc10b692d124430014ff9eb035b0ce38c72ec2eaf`<br>
> 一手来源：[官方 PDF](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) · [官方模型仓库](https://huggingface.co/moonshotai/Kimi-K3) · [官方配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)<br>
> 阅读范围：完整 47 页，包括 Figure 1–16、Table 1–5、方法、系统、评测、case studies 与 Appendix B–F；未做模型权重下载、代码执行或 benchmark 复现<br>
> 视觉证据：选择性嵌入 Figure 2、3、5、7、8、12、13；均从上述版本 PDF 以 240 DPI 裁剪，保留原始图例、坐标和 caption，生成记录见 [Kimi K3 visual evidence assets](/assets/tech-reports/kimi-k3-2026/README.txt)<br>
> 外部信息核验截止：2026-07-28<br>
> 阅读状态：`complete`

## 30 秒结论

Kimi K3 不是靠一个新模块把 Kimi K2 放大到 2.8T parameters，而是同时重做三条信息流：

- **沿 sequence**：用 3:1 的 KDA–Gated MLA hybrid attention，让大部分层以固定状态处理长 context，同时周期性保留全局 softmax attention；
- **沿 depth**：用 Attention Residuals（AttnRes）让层从此前 block 表示中选择信息，而不是只接收逐层累加后的一个 residual state；
- **沿 width**：用 Stable LatentMoE 把 896 个 routed experts 放进较窄的 latent space，每个 token 激活 16 个，同时用 SiTU-GLU 和 Quantile Balancing 解决极稀疏 MoE 的数值与负载问题。

模型是 2.78T total parameters、104.2B activated parameters、93 层、原生视觉、最大 1M-token context window；post-training 再把 3 个任务 domain × 3 个 reasoning-effort level 的 9 个 RL experts 蒸馏回一个模型。（Table 1，PDF pp. 11–14）

**最值得学习的地方**不是 “2.8T” 本身，而是 architecture、training、RL infrastructure 与 serving 被设计成同一套约束闭环：KDA 的 recurrence 决定了 context parallelism 和 prefix cache 的形状；极稀疏 MoE 决定了 load balancing、expert placement 和 kernel；million-token agent trajectory 又决定了 partial rollout、外部 cache pool 和 resumable sandbox。

**最强证据**是几个可以独立检查的机制：KDA 的 bounded decay、Quantile Balancing 的 balanced-assignment 推导、MoonEP 的 redundant-expert 上界，以及官方配置与论文 Table 1 的对应。

**最大证据缺口**是：report 没有披露 pretraining sampled/loss tokens、总训练 FLOPs、集群规模、data mixture、scaling-law 拟合参数，也没有把 “2.5× scaling efficiency” 分解到各项改动。因此这个数字应理解为整套 K3 recipe 相对 K2 的等 validation loss 横向 compute 比，不是某个模块单独带来的 2.5× throughput 或 benchmark 提升。

## 五分钟论文地图

```text
开放模型的 pretraining scale 停留在约 1T-class
且 reasoning / agent RL 开始产生越来越长的 test-time trajectory
    ↓
K2-style MLA + conventional residual + conventional MoE
在 3T parameters、1M context 和 16-of-896 routing 下出现
长 context 成本、跨层信息压缩、activation explosion、负载失衡和系统状态问题
    ↓
Architecture：
3 KDA : 1 Gated MLA
+ Block AttnRes
+ Stable LatentMoE
+ from-scratch MoonViT-V2
    ↓
Training：
8K → 64K → 256K → 1M context curriculum
+ multimodal next-token prediction
+ 9 个 domain/effort RL experts
+ Multi-Teacher On-Policy Distillation
    ↓
Systems：
KDA Context Parallelism
+ MoonEP
+ memory/offload co-design
+ persistent rollout/sandbox state
+ KDA-aware prefix cache
    ↓
Evidence：
scaling-law curve声称等 loss 下约 2.5× compute efficiency
+ 大规模 public / internal / third-party benchmark suite
+ 若干 kernel、compiler、chip 与 knowledge-work case studies
    ↓
可支持的结论：
K3 展示了一条把超稀疏 MoE、hybrid linear attention 和长时程 agent
联合做成可训练、可部署 open-weight model 的完整路线

不能直接支持的结论：
每个新组件各自贡献多少、1M context 是否在所有 long distance dependency
任务上稳定有效、或 K3 是否脱离具体 harness 和预算后总体优于其他模型
```

### 最小前置知识

读这篇 report 前，只需先掌握：

1. causal attention 中 query、key、value 与 KV cache 的作用；
2. MoE 的 Top-$k$ routing、total parameters 与 active parameters；
3. residual connection 与 pipeline/expert/context parallelism；
4. SFT、RL、on-policy distillation 和 speculative decoding 的基本区别；
5. benchmark score 必须连同 prompt、sampling、tools、harness 和预算一起解释。

## 论文真正贡献了什么

| 可检查贡献 | 类型 | 相对 baseline 的最小变化 | 主要证据 | 不能证明什么 |
|---|---|---|---|---|
| 训练并开放 2.78T total / 104.2B active 的原生多模态 MoE | artifact + system integration | 从 K2 的 1.04T / 32.6B 扩到 K3，并把 context window 从 128K 扩到 1M | Table 1；官方 weights/config | 模型参数更大本身不证明更高 compute efficiency |
| Hybrid KDA–MLA + bounded KDA decay | mechanism + implementation | 每 block 3 KDA + 1 Gated MLA；log-decay 下界固定为 $-5$ | Eq. 1–7、Figure 3，pp. 4–6 | report 没给单独的 quality/throughput ablation |
| Block AttnRes | implementation of prior mechanism | 从逐层 residual accumulation 改为对 block 表示的 depth attention | Eq. 8–10，p. 6；AttnRes 来自前序工作 | K3 report 没单独量化它在 K3 上的增益 |
| Stable LatentMoE | mechanism combination | LatentMoE + routed-path RMSNorm + SiTU-GLU + Quantile Balancing | Eq. 11–14、Figure 4–5、Appendix B–D | 2.5× 总体提升不能全部归因给 MoE |
| Quantile Balancing（QB） | optimization mechanism | 用 balanced-assignment dual 的 quantile coordinate minimizer 取代固定步长 bias update | Eq. 14、20–27；Algorithm 1 | “完美负载”依赖 batch-level target、无 ties 与 histogram 近似 |
| Multi-effort agentic RL + MOPD | training system | 3 domains × 3 effort experts，经 dense on-policy reward 合并 | Figure 8、Eq. 15，pp. 12–14 | Figure 8 无绝对 FLOPs/score 轴值，无法量化 scaling law |
| KDA Context Parallelism（KCP） | distributed-system mechanism | 把 segment 表示为对 incoming state 的 affine map，并用 prefix scan 合成 | Eq. 17，pp. 17–18 | 没有端到端 scaling curve 或通信量实测表 |
| MoonEP | distributed-system mechanism + proof | dynamic redundant experts 保证每 rank 精确等负载 | §5.2.1、Appendix E | report 没提供对 DeepEP/ECHO 的吞吐 benchmark |
| Million-token RL/serving state management | system | 外部 cache pool、partial rollout、resumable microVM、hybrid prefix cache | §5.3–5.4，pp. 21–25 | 多数收益是设计说明，缺少 controlled baseline |

其中 KDA、AttnRes、LatentMoE、Muon、partial rollout 等都有前序工作。K3 的主要新意更接近“在新的 scale 与约束下，修改关键机制并完成系统闭环”，而不是从零提出所有组成部分。

## 符号与约定

论文在不同小节复用了 $\alpha$、$q$、$p$ 等字母。下面保留论文符号，同时给出 reader-facing role，避免把不同对象混在一起。

| 符号 | 含义 | 类型、shape 或单位 | 状态 |
|---|---|---|---|
| $t$ | token position | $t=1,\ldots,T$ | index |
| $x_t$ | 第 $t$ 个位置的 hidden state | runtime vector，$\mathbb{R}^{d}$ | activation |
| $q_t,k_t$ | KDA 单头 query/key | runtime vector，$\mathbb{R}^{d_k}$ | activation |
| $v_t$ | KDA 单头 value | runtime vector，$\mathbb{R}^{d_v}$ | activation |
| $S_t$ | KDA 在位置 $t$ 更新后的 recurrent state | runtime matrix，$\mathbb{R}^{d_k\times d_v}$ | persistent state |
| $\alpha_t^{\mathrm{KDA}}$ | KDA 每个 key channel 的一步 retention | runtime vector，$(0,1)^{d_k}$ | activation |
| $\beta_t$ | delta-rule write strength | runtime scalar，$(0,1)$ | activation |
| $g_t$ | log-decay，$\alpha_t^{\mathrm{KDA}}=\exp(g_t)$ | runtime vector | activation |
| $A_h$ | KDA 每个 head 的 decay log-scale | scalar per head | trainable parameter |
| $g_{\min}$ | log-decay 下界，K3 固定为 $-5$ | scalar | fixed hyperparameter |
| $L$ | backbone layer 数；K3 为 93 | count | fixed model shape |
| $w_l$ | AttnRes 第 $l$ 层的 pseudo-query | $\mathbb{R}^{d}$ | trainable parameter |
| $\alpha_{i\to l}^{\mathrm{depth}}$ | 第 $l$ 层从第 $i$ 个 depth source 读取的权重 | scalar，沿 source 归一化 | runtime activation |
| $d$ | full hidden width；K3 为 7168 | channels | fixed model shape |
| $\ell$ | routed latent width；K3 为 3584 | channels | fixed model shape |
| $m,n,k$ | batch tokens、routed experts、每 token 选择数 | counts；K3 $n=896,k=16$ | runtime/fixed |
| $s_{i,j}$ | token $i$ 对 expert $j$ 的 raw sigmoid router score | scalar in $(0,1)$ | runtime activation |
| $b_j$ | 只用于 Top-$k$ selection 的 expert bias | scalar per expert | training-updated buffer，inference 冻结 |
| $\mathcal{T}_i$ | token $i$ 选中的 expert index set | $\lvert\mathcal{T}_i\rvert=k$ | runtime indices |
| $p_{i,j}$ | 选中 expert 的 mixture weight | scalar | runtime activation |
| $q_{\mathrm{load}}=mk/n$ | 每个 expert 的目标 token load | token count | derived quantity |
| $\pi_\theta$ | 被合并的 student policy | token distribution | trainable model |
| $\pi_{\mathrm{teacher}}^{(d,e)}$ | domain $d$、effort $e$ 的 teacher policy | token distribution | frozen teacher |
| $R_{\max}$ | MOPD reward clipping threshold | scalar | hyperparameter，数值未披露 |
| $M$、$\widetilde S$ | KCP segment 的 transition 与 zero-state contribution | $\mathbb{R}^{d_k\times d_k}$、$\mathbb{R}^{d_k\times d_v}$ | runtime state fragment |

约定：

- 所有向量在公式中按 column vector 理解。
- KDA 的 $\alpha_t^{\mathrm{KDA}}$ 与 AttnRes 的 $\alpha_{i\to l}^{\mathrm{depth}}$ 不是同一个对象。
- QB 中的 $q_{\mathrm{load}}$ 是目标负载；Eq. 16 speculative decoding 中的 $q(x)$ 是 draft distribution，二者无关。
- “1M”在模型配置中是 $1{,}048{,}576$ positions；正文通常用近似的 one million tokens。

## Architecture：三条信息流如何拼在一起

[![Kimi K3 整体架构：token、depth、width 三条信息流与原生视觉入口](/assets/tech-reports/kimi-k3-2026/figure-02-k3-architecture.png)](/assets/tech-reports/kimi-k3-2026/figure-02-k3-architecture.png)

> **原图定位：** Kimi K3 Technical Report，Figure 2，p. 3；由官方 PDF 原图裁剪。
>
> **看图重点：** 右侧是重复 block 与 AttnRes 的 depth mixing；左下是 KDA 的 token mixing；左上是 Stable LatentMoE 的 width mixing；右下是 MoonViT-V2 视觉入口。整篇 architecture 可以先按这四块阅读。

### 模型形状

| Field | Kimi K3 | 来源 |
|---|---:|---|
| Total parameters | 2.78T | Table 1，p. 11 |
| Activated parameters | 104.2B | Table 1，p. 11 |
| Layers | 93 | Table 1，p. 11 |
| Hidden dimension | 7168 | Table 1，p. 11 |
| Attention heads | 96 | Table 1，p. 11 |
| Attention composition | 69 KDA + 24 Gated MLA | Table 1，p. 11 |
| Routed experts | 896 | Table 1，p. 11 |
| Active experts per token | 16 | Table 1，p. 11 |
| Shared experts | 2 | Table 1，p. 11 |
| Latent MoE dimension | 3584 | Table 1，p. 11 |
| Vocabulary | 160K | Table 1，p. 11 |
| Vision encoder | MoonViT-V2，27 layers，401M parameters | pp. 9–11 |
| context window | up to 1M tokens | pp. 1–2、12 |

每个重复 block 是：

```text
KDA + Stable LatentMoE
KDA + Stable LatentMoE
KDA + Stable LatentMoE
Gated MLA + Stable LatentMoE
```

backbone 末尾额外放一个 Gated MLA，使最后一层一定执行 global attention。（§2.1，p. 4）

### KDA：固定状态的长 context mixing

KDA 的单头 recurrence 是：

$$
S_t
=
\left(I-\beta_t k_tk_t^\top\right)
\operatorname{Diag}\!\left(\alpha_t^{\mathrm{KDA}}\right)S_{t-1}
+\beta_t k_tv_t^\top,
\qquad
\widetilde o_t=S_t^\top q_t.
$$

这里 $S_t\in\mathbb{R}^{d_k\times d_v}$ 是 runtime state；$q_t,k_t\in\mathbb{R}^{d_k}$，$v_t\in\mathbb{R}^{d_v}$；$\alpha_t^{\mathrm{KDA}}$ 先按 channel 衰减旧状态，$\beta_t$ 决定当前 key/value 写入强度。（Eq. 1，p. 4）

一步直觉：

1. $\operatorname{Diag}(\alpha_t)S_{t-1}$ 对旧状态按 key channel 做遗忘；
2. $-\beta_t k_tk_t^\top$ 删除旧状态在当前 key 方向上已经存有的内容；
3. $+\beta_t k_tv_t^\top$ 写入当前 value；
4. 用 query $q_t$ 从更新后的 state 读取 $\widetilde o_t$。

这不是普通 attention weight matrix；它把整个 prefix 压进固定 shape 的 recurrent state。

**[复原推导] 标量最小例子。** 令 $d_k=d_v=1$、$k_t=q_t=1$、$\alpha_t=0.9$、$\beta_t=0.5$、$S_{t-1}=2$、$v_t=4$：

$$
S_t=(1-0.5)\times0.9\times2+0.5\times4=2.9,
\qquad
\widetilde o_t=2.9.
$$

旧记忆先衰减并被当前 key 方向部分覆盖，再写入新值。这个例子只解释 recurrence，不代表真实多维 head 的行为。

#### 为什么要 lower-bounded decay

chunkwise parallel form 会用累计 retention

$$
\gamma_{i\to j}=\prod_{r=i}^{j}\alpha_r^{\mathrm{KDA}}
$$

重标定 key。若 $\alpha$ 可以无限接近 0，$1/\gamma$ 会溢出。Kimi Linear 使用 unbounded negative-Softplus 产生 log-decay；K3 改成：

$$
g_t^h
=
g_{\min}\operatorname{Sigmoid}\!\left(e^{A_h}z_t^h\right)
\in(g_{\min},0),
\qquad
\alpha_t^h=\exp(g_t^h),
\qquad
g_{\min}=-5.
$$

$A_h$ 是 trainable per-head scalar，$z_t^h$ 是输入相关的 decay logit。于是每步 $\alpha_t^h>e^{-5}\approx6.7\times10^{-3}$；16-token tile 的累计 log-decay 大于 $-80$。（Eq. 5、Figure 3，p. 5）

[![Kimi K3 的 lower-bounded log-decay 及其对 KDA diagonal-tile 计算路径的影响](/assets/tech-reports/kimi-k3-2026/figure-03-bounded-kda-decay.png)](/assets/tech-reports/kimi-k3-2026/figure-03-bounded-kda-decay.png)

> **原图定位：** Kimi K3 Technical Report，Figure 3，p. 5；由官方 PDF 原图裁剪。
>
> **看图重点：** 左图把无下界的 negative-Softplus 改为下界为 $g_{\min}=-5$ 的 scaled sigmoid；右图给出实现后果——原本需要 position-pair 特殊处理的 diagonal tiles 可以改走 dense Tensor Core 路径。

**[复原推导] 数值边界。** $e^{80}\approx5.54\times10^{34}$，仍低于 BF16 最大有限值约 $3.39\times10^{38}$。因此 diagonal tile 不再需要逐 position-pair 的特殊路径，可以与 off-diagonal tile 一样交给 dense Tensor Core matmul。这个改动首先是数值范围与 kernel regularity 的共同设计，不应只理解成一种新的忘记门。

K3 还把 KDA output gate 改成 input-dependent full-rank projection：

$$
y_t
=
W_o\left[
\operatorname{Sigmoid}(W_gx_t)
\odot
\operatorname{RMSNorm}(\widetilde o_t)
\right].
$$

$W_g,W_o$ 是 trainable matrices，$\widetilde o_t$ 是 recurrent output。（Eq. 6，p. 5）

### 为什么还要保留 Gated MLA

KDA 的优势是 state 大小不随 prefix 长度增长，但它把历史压缩进固定 state。Gated MLA 周期性提供不受该固定状态压缩约束的 global token-to-token content interaction，同时用 latent KV 表示降低 cache。（§2.1.2，pp. 5–6）

K3 的 MLA 不使用显式 positional encoding（NoPE）。论文的设计解释是：

```text
KDA：提供 position-sensitive、recency-aware mixing
MLA：提供 unrestricted global content interaction
```

这使 1M context 扩展不需要 RoPE rescaling 或 interpolation。但“NoPE 不需要改参数”只说明扩展机制，不自动证明 1M 位置上的所有 long distance dependency 都能解决；能力仍取决于 long-context training data 与评测。

### AttnRes：把 depth 也变成一次选择

标准 residual path 把此前所有层压进一个 $h_l$。Full AttnRes 让第 $l$ 层用 trainable pseudo-query $w_l$ 对 embedding 与此前 layer outputs 做 depth attention：

$$
\alpha_{i\to l}^{\mathrm{depth}}
=
\frac{
\exp\!\left(w_l^\top\operatorname{RMSNorm}(k_i)\right)
}{
\sum_{j=0}^{l-1}
\exp\!\left(w_l^\top\operatorname{RMSNorm}(k_j)\right)
},
\qquad
h_l
=
\sum_{i=0}^{l-1}\alpha_{i\to l}^{\mathrm{depth}}v_i.
$$

$w_l\in\mathbb{R}^d$ 是每层 trainable parameter；$k_i=v_i$ 来自 embedding 或第 $i$ 层输出，是 runtime activation。RMSNorm 防止大幅值 layer output 只靠 norm 主导权重。（Eq. 8–9，p. 6）

Full form 的 arithmetic 是 $O(L^2d)$，保存所有 layer output 的 memory/communication 是 $O(Ld)$。K3 使用 Block AttnRes：block 内做 partial sum，block 间对 block representation 做 attention，把长期保存状态降到 $O(Nd)$。93 层按 12-layer block 划分为 8 个 layer blocks；再把 embedding 作为 source 计入，共 9 个 depth sources。（Eq. 10，p. 6）

这里的关系是：

```text
Block AttnRes --approximates[block-level compression]--> Full AttnRes
Full AttnRes --generalizes[learned depth weights]--> uniform residual accumulation
```

论文称 $N\approx8$ 在不同 scale 上恢复大部分收益，但该证据来自前序 AttnRes 工作；K3 report 没给本模型上的独立 ablation。

### Stable LatentMoE：先压窄，再做大量专家

K3 的 routed path 是：

$$
z=W^\downarrow x\in\mathbb{R}^{\ell},
\qquad
u
=
\sum_{i\in\mathcal{T}_k(x)}
p_i E_i^{\mathrm{routed}}(z),
$$

$$
y
=
\sum_{j=1}^{N_s}E_j^{\mathrm{shared}}(x)
+
W^\uparrow\operatorname{RMSNorm}(u),
\qquad
N_s=2.
$$

$W^\downarrow\in\mathbb{R}^{\ell\times d}$ 与 $W^\uparrow\in\mathbb{R}^{d\times\ell}$ 是 trainable projections；K3 的 $d=7168,\ell=3584$。shared experts 走 full-width path，896 个 routed experts 在 half-width latent space 工作，每 token 选 16 个。（Eq. 11，pp. 6–7）

这使“更多 active experts”不必把 full-width representation 发给所有专家，但产生两个新问题：

1. down projection → gated expert FFN → up projection 形成近四个连续 matmul，2.8T scale 下出现 routed activation explosion；
2. 896 experts 让固定步长的 loss-free load-balancing bias 难以快速稳定。

Stable LatentMoE 的三个修正分别对应这些问题。

#### Routed-path RMSNorm

在 aggregate $u$ 与 $W^\uparrow$ 之间加 RMSNorm，使 expert 选择与 mixture weight 引起的 scale variation 不直接进入 up projection。论文称 validation loss 和 downstream benchmark 均有改善，但没有给出数值 ablation。（§2.3.1，p. 7）

#### SiTU-GLU

SiTU-GLU 对 gate 与 up branch 分别 smooth-cap：

$$
\operatorname{SiTU\text{-}GLU}(x)
=
\left[
\beta_1\tanh\!\left(\frac{W_gx}{\beta_1}\right)
\odot
\operatorname{Sigmoid}(W_gx)
\right]
\odot
\left[
\beta_2\tanh\!\left(\frac{W_ux}{\beta_2}\right)
\right],
$$

其中 $\beta_1=4,\beta_2=25$ 是 fixed hyperparameters，$W_g,W_u$ 是 trainable expert matrices。（Eq. 12，pp. 7–8）

局部有

$$
\beta\tanh(z/\beta)=z+O(z^3/\beta^2),
$$

所以它在原点附近一阶匹配 SwiGLU；当 $\beta_1,\beta_2\to\infty$ 时逐点恢复 SwiGLU。每个输出 coordinate 满足

$$
\left\|\operatorname{SiTU\text{-}GLU}(x)\right\|_\infty
\le \beta_1\beta_2=100.
$$

这给出了清楚的数值稳定性边界。（Appendix B，p. 43）尚缺的是：与 SwiGLU、hard clamp 及其他 bounded activation 在 K3 scale 上的完整 quality–stability tradeoff。

#### Quantile Balancing

router 先算

$$
s_i=\operatorname{Sigmoid}(W_rx_i),
\qquad
\mathcal{T}_i=\operatorname{argtopk}(s_i+b),
$$

但 mixture weight 不使用 bias：

$$
p_{i,j}
=
\frac{s_{i,j}}{\sum_{r\in\mathcal{T}_i}s_{i,r}},
\qquad j\in\mathcal{T}_i.
$$

因此 $b_j$ 只改变 dispatch，不直接改变被选 expert 的 mixture weight，也不通过主 objective 训练 router。（Eq. 13，p. 8）

设一个 global batch 有 $m$ tokens、$n$ experts、每 token 选 $k$，目标 load 为

$$
q_{\mathrm{load}}=\frac{mk}{n}.
$$

QB 对每个 token 额外取 Top-$(k+1)$，把第 $k+1$ 个 biased score 作为 cutoff $\alpha_i$。expert $j$ 若想恰好让 $q_{\mathrm{load}}$ 个 token 越过 cutoff，其新 bias 应取：

$$
\widehat b_j^{(t+1)}
=
-
\operatorname{quantile}_{1-k/n}
\left(s_{:,j}-\alpha^{(t)}\right),
$$

$$
b^{(t+1)}
=
\widehat b^{(t+1)}
-
\operatorname{mean}\!\left(\widehat b^{(t+1)}\right)\mathbf 1.
$$

均值中心化不改变 Top-$k$。新 bias 到下一 step 才生效，inference 时冻结。（Eq. 14，pp. 8–9）

Figure 5 的最小例子是 $m=8,n=4,k=1$，所以每 expert 目标 load 为 2。原 Top-$k$ load 是 $(4,3,1,0)$，quantile threshold 调整后变为 $(2,2,2,2)$。

[![Quantile Balancing 把 8 个 token 从不均衡路由调整为每个 expert 负载为 2](/assets/tech-reports/kimi-k3-2026/figure-05-quantile-balancing.png)](/assets/tech-reports/kimi-k3-2026/figure-05-quantile-balancing.png)

> **原图定位：** Kimi K3 Technical Report，Figure 5，p. 8；由官方 PDF 原图裁剪。
>
> **看图重点：** 左侧先显示 $(4,3,1,0)$ 的过热与 dying experts；中间对每列设置 quantile threshold；右侧只有红边对应的 token–expert assignment 发生改变，最终得到 $(2,2,2,2)$。图中的星号是更新 bias 后的 Top-$k$ 选择。

Appendix C 更进一步说明：QB 是 maximum-score balanced bipartite assignment 的 LP dual 上 alternating coordinate minimization；固定步长 sign update 只是同一 objective 的 SignSGD 近似，而 QB 直接跳到 coordinate minimizer。（Eq. 20–27、Algorithm 1，pp. 43–44）

实际训练不收集 $O(mn)$ raw margins，而是每 expert 维护 1000-bin histogram，跨 rank 对 bin counts 做一次 integer all-reduce。论文给出的估计误差上界是一个 bin width，实际为几 $10^{-3}$；通信量低于每 micro-batch 交换 raw margins 方案的 1%。（Appendix D，pp. 44–45）

## Native vision 与 optimizer

MoonViT-V2 有约 401M parameters、27 层，图像和视频共享参数；spatial/temporal attention 分解，temporal pooling 压缩视频 token，2×2 pixel shuffle 把视觉 token 数降到四分之一，支持最高 $3584\times3584$ 输入。（§2.4，pp. 9–10）

与先做 SigLIP contrastive pretraining 再接 LLM 不同，MoonViT-V2 从随机初始化开始，与文本一起只用 next-token prediction。Figure 6 显示它比 SigLIP-initialized MoonViT-3D 有更低、更少 spike 的 vision-tower gradient norm；论文称 vision evaluation 与该 baseline 相当。

这项证据支持“from-scratch 在该 recipe 下更稳定且没有明显损失”，但不够回答：

- 对比模型是否拥有相同 compute、data 与 tuning budget；
- vision data 规模与 mixture；
- 每项 vision benchmark 的等预算 ablation；
- stability 改善来自初始化、architecture 还是 data objective。

矩阵参数继续使用 Muon；attention Q/K/V momentum 按 head 独立做 Newton–Schulz orthogonalization。论文解释这样可以避免大 gradient head 主导 full-matrix update，并称提升大 scale 稳定性，但没有给单独曲线或数值。（§2.5，p. 10）

## Pretraining：2.5× 到底是什么意思

### 公开的 data recipe

文本覆盖 Web Text、Code、Mathematics、Knowledge；视觉覆盖 captions、interleaved image–text、OCR、perception、video 与 visual coding。文本使用 rule、classifier quality、dedup，domain sampling rate 由小模型 ablation 决定；knowledge/math 还会 rephrase 并做 fidelity verification。视觉数据包含 open collections 与 in-house filtering/synthesis/dedup，并增加 code 与 SVG、3D、webpage、game、CAD rendering 的成对数据。（§3.1，p. 10）

未披露：

- sampled tokens、loss tokens 与重复 exposure；
- 文本/视觉及各 domain mixture；
- dataset 名单、license、cutoff 和 decontamination 细节；
- tokenizer 之外的数据计量口径；
- 训练 steps、global batch、peak learning rate；
- 总 FLOPs、accelerator 型号/数量与 wall-clock time。

因此目前无法做完整 token accounting 或 compute accounting。

### Scaling-law claim

论文对 K2 与 K3 model family 分别重新搜索 batch、learning rate、tokens-per-parameter 和 model shape，在 held-out OOD validation loss 上拟合曲线；Figure 7 声称 K3 获得约 2.5× overall scaling efficiency。（§3.2、Figure 7，pp. 10–11）

[![Kimi K2 与 Kimi K3 的 fitted scaling-law curves](/assets/tech-reports/kimi-k3-2026/figure-07-scaling-efficiency.png)](/assets/tech-reports/kimi-k3-2026/figure-07-scaling-efficiency.png)

> **原图定位：** Kimi K3 Technical Report，Figure 7，p. 11；由官方 PDF 原图裁剪。
>
> **看图重点：** `2.5×` 标在相同 validation loss 下两条拟合曲线的水平 FLOPs 距离上；纵轴刻度、拟合参数和置信区间没有公开，因此这张图支持相对 compute-efficiency claim，不支持恢复绝对 loss 或统计不确定性。

**[复原推导] 正确读法**是：在某个共同 target validation loss $L^{\ast}$ 上，图示的水平距离近似表示

$$
\frac{
C_{\mathrm{K2}}(L^{\ast})
}{
C_{\mathrm{K3}}(L^{\ast})
}
\approx2.5,
$$

其中 $C$ 是 scaling study 使用的 training FLOPs。

它不是：

- 训练 throughput 提高 2.5×；
- inference 便宜 2.5×；
- KDA 单独提高 2.5×；
- benchmark 平均分提高 2.5×；
- 2.78T K3 的总训练预算只有 K2 的 $1/2.5$。

Figure 7 没有给 loss 数值、拟合方程、模型点规模、置信区间或 compute 范围；Table 1 同时改变 layers、parameters、active parameters、experts、attention、activation、vision、context、data 和 schedule。因此 2.5× 是 **whole-family empirical association**，不是 component-level causal estimate。

### Training recipe 与 long context

公开 recipe 包括：

- native multimodal next-token prediction；
- Per-Head Muon、K2 weight clipping、QB；
- cosine learning rate schedule，1% linear warmup；
- weight decay 0.1；
- pretraining context 从 8K 扩到 64K；
- cooldown 再从 256K 扩到 1M，形成四阶段 curriculum。（§3.3–3.4，pp. 11–12）

long-context data 经过 exact/fuzzy dedup、video frame perceptual hash、quality filter 和 structural validation；自然长文/视频会 upsample，还合成需要跨 1M positions 收集分散信息才能完成的 multimodal task。

这比“只把 context window 配成 1M”强，因为训练目标确实包含 long distance dependency；但 report 没给：

- 每个 seqlen stage 的 tokens 或 steps；
- 64K 到 256K 的具体切换规则；
- long-context benchmark 的完整曲线；
- 不同位置、needle 数、干扰密度和 generation length 下的失败边界。

## Post-training：九个 experts 如何合回一个模型

### 三阶段 pipeline

```text
SFT cold start
→ 3 domains × 3 reasoning-effort levels 的 RL
→ Multi-Teacher On-Policy Distillation
```

三个 RL domains 是 general tasks、general agents、coding agents；effort 是 low、high、max，共 9 个 expert policies。（§4.1，pp. 12–14）

SFT trajectory 由此前 Kimi domain models 合成，经多阶段 verification 与 human-in-the-loop annotation；数据用 XTML chat template 序列化。数量、人员规模、质量分布和 inter-annotator agreement 未披露。

### Partial rollout 与 reasoning budget

每个 iteration 对 $N$ prompts 各采 $K$ trajectories；当 $\lambda NK$ 个完成时就暂停 generation，未完成 trajectory 排队到下一轮恢复，以避免长尾 straggler。跨 iteration 会产生 stale rollout，论文称沿用前序工作中的 per-token regularization 来容忍 off-policy data，但本 report 没重述 objective 细节。（§4.1.2，p. 13）

reasoning-effort control 给每个 problem $x$ 一个 cold-start budget $b_0(x)$。若 trajectory 总预算 $T(y)$ 超过 $\tau b_0(x)$，task reward 被覆盖为 $-1$：

```text
general task: T(y) = thinking tokens
agentic task: T(y) = reasoning traces + tool-call arguments 的累计 output tokens
```

先训练较大 $\tau$ 的 max expert，再逐步降低 $\tau$ 得到 high/low。$\tau$ 的具体数值与各 domain 配置未披露。

Figure 8 显示 RL FLOPs 增长时多个 internal/public task score 与平均 assistant steps 大体上升。由于图轴没有绝对 FLOPs、score、样本数或误差条，这只能支持定性趋势，不能形成可复用的 RL scaling law。

[![RL FLOPs 增长时八类任务的 score 与平均 assistant steps 变化](/assets/tech-reports/kimi-k3-2026/figure-08-rl-scaling.png)](/assets/tech-reports/kimi-k3-2026/figure-08-rl-scaling.png)

> **原图定位：** Kimi K3 Technical Report，Figure 8，p. 13；由官方 PDF 原图裁剪。
>
> **看图重点：** 蓝线是 score、红色虚线是平均 assistant steps。多数面板随 RL FLOPs 上升而共同增加，但 Coding Experience、Web Development、Professional Workflows 等存在明显非单调区间；这也是不能把它概括成严格 scaling law 的原因。

### Multi-Teacher On-Policy Distillation

对 domain $d$、effort $e$，student 自己生成 token $y_t$，teacher 与 student 的 log-probability ratio 形成 dense reward：

$$
r_{\mathrm{opd}}^d(y_t\mid e,x,y_{<t})
=
\operatorname{clip}
\left(
\operatorname{sg}
\left[
\log
\frac{
\pi_{\mathrm{teacher}}^{(d,e)}(y_t\mid x,y_{<t})
}{
\pi_\theta(y_t\mid e,x,y_{<t})
}
\right],
-R_{\max},
R_{\max}
\right).
$$

$\operatorname{sg}$ 是 stop-gradient；ratio 作为 RL reward，不直接反传进 teacher/student log-prob expression；$\pi_\theta$ 仍通过 policy optimization 更新。（Eq. 15，pp. 13–14）

直觉是：若 teacher 比 student 更认可 student 当前采到的 token，reward 为正；反之为负。它让不同 domain/effort teacher 在同一 RL infrastructure 中提供 dense signal。

论文还报告一个负结果：更细粒度的 top-$k$ distillation objective 在 convergence 或 final performance 上没有明显优势。这是少数明确写出的 negative evidence。

### Deployment-aware post-training

1. **QAT**：从 SFT 起，MoE expert weights 用 MXFP4、activations 用 MXFP8；attention、latent projections、shared experts、router 保持 higher precision。rollout 与 training 使用同一量化配置，避免 train–inference mismatch。（§4.1.4，p. 14）
2. **EAGLE-3 draft**：把 pretraining 的 MTP layer fine-tune 成 draft model，target frozen；融合第 1、第 4 和最后 AttnRes block features，projection 初始化为 $[0\ 0\ I]$，开始时只等于 high-level feature。
3. **直接优化 acceptance**：

   $$
   \mathcal{L}_{\mathrm{LK}}
   =
   -\log
   \sum_{x\in\mathcal V}
   \min\!\left(p_{\mathrm{target}}(x),p_{\mathrm{draft}}(x)\right).
   $$

   $\sum_x\min(p,q)$ 是 lossless speculative sampling 的 per-token acceptance probability，因此这个 objective 比 generic KL 更直接。（Eq. 16，p. 14）

report 没有给 QAT quality loss、draft acceptance rate 或 speculative decoding speedup，所以这里的证据主要是方法设计而非收益量化。

## RL environment：能力来自哪里

论文的 agent capability 不只是模型 objective，还依赖 environment distribution：

- unified white-box harness 把 tools、system prompt、context management、skills、memory、subagents 组合成不同 agent scaffold；
- self-evolving knowledge graph 用 web retrieval 产生 knowledge/coding/vision task；
- verifiable search、professional workflow 与 vision-in-the-loop Python task；
- GPU kernel task 同时检查 correctness、performance 和 reward hacking；
- Gmail、Notion、Slack、Canvas 等 mock apps 支持 persistent multi-day personal-assistant workflow；
- Autonomous Execution Tasks 用 hidden verifier 检查 final environment state；
- web development 同时用 deterministic test、pixel/structure similarity 和 model judge。（§4.2，pp. 14–17）

**[综合判断]** 这说明 K3 的 agent 能力是：

```text
base model
+ environment/task distribution
+ verifier/reward design
+ harness diversity
+ long-lived state infrastructure
```

的联合结果。只看 base model parameters 无法解释它，也不能把特定 harness 下的 benchmark gain 自动归因给 architecture。

## Infrastructure：模型公式如何变成可训练系统

### KDA Context Parallelism

一个 KDA segment 对 incoming state 的作用可写成 affine map：

$$
S_{\mathrm{out}}=MS_{\mathrm{in}}+\widetilde S,
$$

$M\in\mathbb{R}^{d_k\times d_k}$ 是该 segment 的 cumulative transition，$\widetilde S\in\mathbb{R}^{d_k\times d_v}$ 是从 zero state 出发由本地 tokens 生成的 state。

**[复原推导]** 两个连续 segments 的组合是：

$$
(M_2,\widetilde S_2)\circ(M_1,\widetilde S_1)
=
\left(
M_2M_1,
\widetilde S_2+M_2\widetilde S_1
\right).
$$

这个 operator 是 associative，所以各 context-parallel rank 可先独立算本地 $(M,\widetilde S)$，all-gather fixed-size fragments 后用 prefix scan 精确恢复每个 rank 的 incoming state。（Eq. 17，pp. 17–18）

与 softmax attention CP 需要交换随 seqlen 增长的 KV blocks 相比，KCP 交换的 state fragment shape 不随 local sequence length 增长；代价是 $M$ 的计算与 fixed-size state communication。论文称 linear compute scaling，但没有给设备数–吞吐曲线。

### MoonEP

MoonEP 观察到 expert-parallel rank 的 token load 与 activation shape 会动态失衡，于是：

- 按当前 micro-batch/router output 在线规划 redundant experts；
- 保证每 rank 恰好接收 $S\times K$ token routes；
- 让 shape 静态，从而消除每层 host/device shape synchronization；
- 直接把 token 发送到 remote expert-grouped position，避免中间 copy；
- backward 后把 redundant expert gradient reduce 回 home rank。

Appendix E 证明：对 $E$ experts、EP size $R$，任意 router output 都存在每 rank 最多放 $E/R$ redundant experts 的 perfect-balance plan；并构造了约 $\lceil E(R-1)/R^2\rceil\approx E/R$ 的近 tight case。（Eq. 28，pp. 45–46）

这是很强的 feasibility 证据，但不是性能证据。report 没给 planner overhead、extra weight memory、network bytes 或与 DeepEP/ECHO/UltraEP 的端到端 throughput。

### Million-token RL state

K3 的 partial rollout 需要跨 iteration 保存模型 prefix 与 sandbox world state。系统使用：

- GPU 上保留 active KV blocks；
- idle reusable prefix 被 eviction 时 write back 到 CPU DRAM external cache pool；
- KDA state 与 MLA KV blocks 同生命周期 offload/prefetch；
- training iteration 结束后把 weights/optimizer state 放到 NVMe，为 rollout external cache pool 释放 DRAM；
- runtime signal 驱动 auto-throttling，随 KV pressure 上升降低 concurrency；
- non-policy model weights 复用 policy FP32 gradient-buffer storage。

AgentENV 基于 Firecracker microVM。论文报告 incremental checkpoint/resume 最低 133 ms/49 ms、真实 workload 最高 6.5× memory overcommit，以及整个 K3 training/evaluation 创建 51,219,741 sandboxes、覆盖 1,505,678 images。（§5.3，pp. 21–22）

这些是 source-reported system numbers；没有 latency distribution、硬件环境、对照 sandbox 或 failure rate，外推需谨慎。

### KDA-aware prefix cache

Hybrid architecture 同时有：

```text
MLA KV cache：随 prefix length 增长，per-token paged
KDA state：固定大小，per request / per checkpoint boundary
```

K3 把两者放进统一 byte-size page pool，但把 storage 与 hash granularity 分开：

- physical block 可以是 1024–6144 tokens；
- prefix hash block 可细到 512 tokens；
- KDA checkpoint 只在可被 hash lookup 引用的边界稀疏保存；
- hit 必须在所有 KDA cache groups 都有一致 checkpoint。

Figure 12 的例子中，physical block 是 6144 tokens；请求前 2800 tokens 匹配，最长可恢复边界是 $2560=5\times512$，于是无需重算 $[0,2560)$。（pp. 23–24）

[![KDA-aware prefix cache 在 6144-token physical block 内恢复 2560-token prefix](/assets/tech-reports/kimi-k3-2026/figure-12-prefix-cache.png)](/assets/tech-reports/kimi-k3-2026/figure-12-prefix-cache.png)

> **原图定位：** Kimi K3 Technical Report，Figure 12，p. 23；由官方 PDF 原图裁剪。
>
> **看图重点：** MLA KV 可以按 512-token hash blocks 命中，但只有同时保存 KDA checkpoint 的边界才能恢复 recurrent state。橙点 $B=2560$ 是两种 cache 都满足的最长共同边界；恢复后对 partial MLA block 做 copy-on-write。

这解决的是“固定大状态不能每 token snapshot”与“MLA 想细粒度复用”的冲突。report 详细解释了一致性 failure modes，但没有 cache hit-rate 或 TTFT speedup benchmark。

## 实验到底证明了什么

### 主要评测协议

Kimi K3 默认使用 `reasoning_effort=max`、temperature 1.0；single-step reasoning/knowledge 和无工具 vision 通常 top-$p=0.95$，agentic task top-$p=1.0$。（§6.1.3，p. 26）

对比模型主要是 Claude Fable 5、GPT-5.6 Sol、Claude Opus 4.8、GPT-5.5 与 open-weight GLM-5.2。论文明确披露了若干重要不一致：

- Claude Fable 5 results 可能发生 fallback，GPT-5.6 Sol 可能触发 cyberguards；
- coding benchmark 使用 Kimi Code、Claude Code 或 Codex，不总是同一 harness；
- Terminal-Bench 2.1 对所有模型取各自最佳 harness；
- SWE-Marathon 使用 2026-07-09 的 H20-calibrated pre-v1.1 branch，且 Fable 5 在 35% tasks fallback；
- BrowseComp 主表结果使用 300K context-compaction；完整 1M、无 context management 时 K3 为 90.4；
- 大部分 vision 分数平均 3 runs，ZeroBench 按官方设置 5 runs；许多其他 benchmark 未给 seeds/variance。

### 一组有诊断价值的结果

| 问题 | K3 结果 | 最有意义的比较 | 支持的结论 | 主要边界 |
|---|---:|---|---|---|
| Graduate reasoning | GPQA Diamond 93.5 | GPT-5.6 Sol 94.1 | 与 frontier result 接近 | 高分 benchmark，未给 variance |
| Research-level reasoning | HLE-Full 43.5 / 56.0（无/有工具） | Fable 5 53.3 / 63.0；Sol 44.5 / 58.0 | K3 仍有明显上限 | tool setting 只按表中大类说明 |
| Agentic coding | ProgramBench 77.8 | Sol 77.6 | 在该 protocol 下最高 | 0.2 point 差距未给不确定性 |
| Terminal task | Terminal-Bench 2.1 88.3 | Sol 88.8 | 接近最高结果 | 各模型取最佳 harness |
| GPU-oriented SWE | SWE-Marathon 42.0 | Fable 35.0、Sol 39.0、Opus 40.0 | 在该 H20 branch 上领先 | pre-v1.1、hardware recalibration、fallback |
| Web search agent | BrowseComp 91.2 | Sol 90.4、Fable 88.0 | 强 agentic search 表现 | K3 使用 300K compaction；不同来源/系统 |
| Spreadsheet agent | SpreadsheetBench 2 34.8 | Fable 34.7 | 近似持平最高 | 0.1 point，无误差条 |
| Document vision | OmniDocBench 91.1 | Fable 89.8 | 强 OCR/document capability | 3-run average，但方差未给 |
| Visual reasoning | Math-Vision 94.3 / 97.8 | Sol 95.8 / 97.8 | tool 后追平最强值 | Python tool 是 system component |
| Hard visual reasoning | ZeroBench 23.0 / 41.0 pass@5 | Fable 23.0 / 46.0 | tools 带来大幅 gain | pass@5，不是 single-sample |

以上均来自 Table 2（p. 27）。不能从这张表推出统一“总体排名”，因为 harness、fallback、tool、pass@$k$ 与评测来源不同。

### Internal evaluation

Table 3–4 显示 K3 在 Swarm Bench 76.3、Deep Research Bench 90.0、Kimi Webdev Bench 对 Opus 4.8 的 blind expert judging 为 58.6% win / 13.8% tie / 27.6% lose；但在 MIRA、Agent Behavior、24/7 Claw、Agentic Vision 等 internal suite 并非领先。（pp. 28–30）

这些 internal benchmarks 能紧贴产品 failure modes，但 dataset、sample size、rubric、judge reliability、版本维护和 contamination boundary 未公开，外部可复现性弱。它们更适合回答“团队用什么反馈训练产品”，不适合独立确认普遍 superiority。

### Cyber evaluation

论文报告：

- Tier 1 人工 review 的候选中约 70% 被确认是真漏洞，并含 6 个项目中的 16 个此前未知漏洞；
- Tier 2 exploit suite 上 K3 解出 14/36（38.9%），GLM-5.2 为 8/36（22.2%）；
- 14 个成功中有 10 个来自 user-space；kernel track 中两模型均有四分之三任务未解；
- paper 引述 UK AISI/NIST 联合评估：K3 在 41 个 end-to-end tasks 上 arbitrary code execution 为 0。（§6.2.2，pp. 30–31）

这支持“已有真实 vulnerability discovery 与部分 exploit capability，但 hardened end-to-end exploitation 仍显著低于 human expert”。70% 的 denominator、review selection rule 和完整 task artifact 未公开，因此不能估计总体 vulnerability precision。

### Third-party 与 cost

Table 5 汇总的是截至 2026-07-23 的 third-party snapshot，例如 Artificial Analysis Intelligence Index 57.1（#4/580）、Vals Index 74.7（#2/39）、WebDev Arena 1678 Elo（#1/99）。Elo 会随 match 漂移，不能当成 2026-07-28 的固定现状。（p. 32）

Figure 13 报告 BrowseComp 上 K3 为 91.2、USD 2.03/task；相对 GPT-5.6 Sol 成本约一半，相对 Claude max effort 低一个数量级。这里的 cost 来源混合了 K3 自有 run、其他厂商 published chart 与 third-party pay-per-token pricing；Kimi Code Bench 又使用不同 harness。因此它支持“在这些计价和 workload 假设下处于 cost frontier”，不支持统一 serving TCO。

[![Kimi K3 在四项 agent benchmark 上的 score 或 Elo 与每任务推理成本](/assets/tech-reports/kimi-k3-2026/figure-13-score-cost.png)](/assets/tech-reports/kimi-k3-2026/figure-13-score-cost.png)

> **原图定位：** Kimi K3 Technical Report，Figure 13，p. 32；由官方 PDF 原图裁剪。
>
> **看图重点：** 每个 panel 的纵轴和 harness 都不同，只能在单个 panel 内看 score–cost trade-off。K3 用红色星号表示；图中的美元成本是特定 workload 与计价假设下的 per-task cost，不是统一硬件上的 serving TCO。

## Claim–evidence map

| Claim | Paper evidence | Strength | Gap |
|---|---|---|---|
| K3 是 2.78T total / 104.2B active、16-of-896 MoE | Table 1；官方 config/model card | strong / verified | weights 未在本次分析中实际加载 |
| bounded KDA decay 避免 16-token tile reciprocal 超出 BF16 range | Eq. 5、Figure 3；数值可重算 | strong / verified mechanism | 无 kernel speedup table |
| QB 对 balanced-assignment dual 做 exact coordinate minimization | Eq. 20–27、Algorithm 1 | strong / verified derivation | training histogram 是近似 quantile |
| MoonEP 每 rank $E/R$ redundant slots 足以保证 perfect balance | Appendix E theorem | strong / conditional proof | 不等于端到端性能更优 |
| K3 family 比 K2 有约 2.5× scaling efficiency | Figure 7 | moderate / supported | 无 fit 参数、范围、置信区间和 component ablation |
| from-scratch MoonViT-V2 更稳定且 quality 相当 | Figure 6 + prose | moderate | quality table 与等预算控制不足 |
| RL FLOPs 增加带来更强能力和更长 tool trajectory | Figure 8 | weak-to-moderate trend | 无绝对轴、variance 和 task details |
| K3 在广泛 benchmark 上接近最强 proprietary systems | Table 2、5 | moderate | protocol/harness/tool/fallback 混杂 |
| 1M agent RL 与 serving infrastructure 可行 | §5.3–5.4 system descriptions | moderate system evidence | 缺少 end-to-end baseline、SLO 与利用率曲线 |
| case studies 展示长时程技术执行能力 | §7、Figure 14–15 | qualitative | selected cases、缺少样本分布与独立复现 |

## 论文最重要的三个 insight

### 1. Hybrid attention 的价值不只是复杂度，而是状态分工

**[综合判断]** KDA 与 MLA 的关系不是简单“linear attention 近似 softmax attention”：

```text
KDA state
--fixed-size compression--> long prefix

periodic MLA
--global content interaction--> compensate fixed-state bottleneck

NoPE
--delegates position sensitivity to--> KDA recurrence
```

这是一种 state representation 分工。它的风险也由此产生：若 KDA state 丢失了需要的细节，后续 MLA 只能访问 token representations 已经保留下来的内容，不能从不存在的历史 KV 中恢复原始细节。

### 2. 超稀疏 MoE 的算法问题会立即变成系统问题

**[综合判断]** 16-of-896 不只是 router 公式改变：

```text
更多 experts
→ 更难平衡的 token load
→ dynamic shapes / fragmentation
→ host sync 与 rank straggler
→ MoonEP redundant placement
→ static shapes 与 zero-copy path
```

QB 解决“每个 expert 应该吸收多少 token”，MoonEP 解决“这些 token 到哪个 rank 执行”。前者是 optimization/routing，后者是 placement/communication，两者不能互相替代。

### 3. 长时程 agent 的真正 state 跨越模型与环境

**[综合判断]** 一个暂停的 million-token trajectory 至少包含：

- model-side MLA KV cache；
- model-side KDA recurrent states；
- rollout policy/version 与 stale-data metadata；
- sandbox memory/disk/process state；
- tool results 与 external world state。

只保存文本 transcript 会导致昂贵 re-prefill，也无法恢复环境。K3 的 external cache pool、partial rollout 与 AgentENV checkpoint 是同一个“persistent trajectory state”问题的不同切面。

## 局限与容易误读之处

### 作者明确承认

- overall performance 仍落后于最强 proprietary systems；HLE/CritPt 等 research-level reasoning 是明显短板。（pp. 1、26）
- cyber exploit completion 对 hardened targets 仍远低于 human experts。（pp. 30–31）
- top-$k$ distillation objective 没有显示出优势。（p. 14）

### 本次分析识别的证据缺口

1. **训练透明度不足**：无 total tokens、loss tokens、FLOPs、硬件集群、wall time、data mixture、具体 batch/LR。
2. **2.5× 不可分解**：architecture、model shape、data、optimizer、schedule、context 和 multimodal 同时变化。
3. **系统章节缺少性能表**：KCP、MoonEP、offload、prefix cache 和 kernel 都缺少统一 baseline 的吞吐/延迟/显存/网络数据。
4. **long-context capability 证据不完整**：有 training recipe 和 agent workload，但缺少 systematized length/generalization curve。
5. **大量评测条件异质**：harness、fallback、tools、pass@$k$、context management 与来源不同。
6. **internal benchmark 不可复现**：缺 dataset、sample size、judge agreement、完整 prompt 和版本。
7. **case study 有 selection bias**：令人印象深刻，但不提供成功率、失败样本或同预算分布。
8. **安全材料不是完整 system card**：有 cyber capability，但对 broader misuse、bias、privacy、deployment mitigation 的系统覆盖有限。

### 四个常见误读

- “1M context window”不等于模型在所有 1M-token task 上都能利用 long distance dependency。
- “2.8T parameters”不能与 dense model 参数量直接比较 compute；每 token active parameters 是 104.2B。
- “2.5× scaling efficiency”不是单个 KDA/MoE module 的 speedup。
- “benchmark 第一”不等于 base model 第一；许多结果包含 harness、tool、context management 或多次 sampling。

## 可验证的后续实验

### 1. 分解 2.5× scaling efficiency

**Label：** `[扩展假设]`

**Proposal：** 在同一小规模 model family、同一 tokens/compute/data 上逐步加入 hybrid KDA–MLA、AttnRes、LatentMoE norm、SiTU-GLU、QB、Per-Head Muon。

**Predicted observation：** stability-related components 主要减少 divergence/outlier；attention/AttnRes 改变 validation loss slope；data/recipe 贡献可能与 architecture 同量级。

**Falsification：** 若控制 data/shape 后各结构改动没有稳定 improvement，则 2.5× 主要来自 data、shape 或 tuning，而非 architecture。

**Minimum experiment：** 3 个 scale points × 每项至少 3 seeds，公开 loss–FLOPs curve、fit residual 与 failure rate。

### 2. 检查 KDA 固定 state 的 information bottleneck

**Label：** `[扩展假设]`

**Proposal：** 构造相同 1M seqlen、但 required facts 数、间隔、干扰密度和 query composition 不同的任务；对比 3:1、1:1 KDA–MLA 与 full attention。

**Predicted observation：** KDA 在低-rank、可递推信息上稳定；当需要保留大量独立细节时，periodic MLA 频率会成为关键变量。

**Falsification：** 若三种 density 下 KDA hybrid 与 full attention gap 不随信息量增加，则固定 state bottleneck 不是主导因素。

**Minimum experiment：** 固定 model/compute，报告 accuracy–information-density–length surface，而非单一 needle score。

### 3. 分开验证 QB 与 MoonEP

**Label：** `[扩展假设]`

**Proposal：** 用同一 router trace 比较 sign bias vs QB 的 load convergence；再固定路由，比较 DeepEP-like placement vs MoonEP 的 rank makespan。

**Predicted observation：** QB 减少 expert load variance；MoonEP 把剩余 token/expert placement skew 转换成 bounded redundant memory，并降低 rank tail。

**Falsification：** 若 QB 已使 rank makespan 接近一致，MoonEP 的 redundant expert cost 可能大于收益。

**Minimum experiment：** 同时报 load CV、max/mean rank tokens、redundant weight bytes、all-to-all bytes、step time P50/P99。

### 4. 把 agent capability 与 scaffold 分离

**Label：** `[待验证]`

**Proposal：** 对 K3 与对比模型使用完全相同的 tool schema、context manager、budget、retry、judge 和 sandbox。

**Predicted observation：** 一部分 gain 保留，另一部分会随 Kimi-specific environment/harness 优势缩小。

**Falsification：** 若换 scaffold 后相对排序和 effect size 基本不变，则 base policy 是主要来源。

**Minimum experiment：** 至少两个 in-distribution 和两个 OOD harness，报告 single-run、pass@$k$、tokens、tool calls、latency 与 dollar cost。

## Reproduction 与代码阅读入口

### 先读论文哪几处

1. **Figure 2 + §2.1–2.3（pp. 3–9）**：理解三条信息流和稳定性修改。
2. **Table 1 + §3.2–4.1（pp. 10–14）**：区分 model scale、scaling claim 与 post-training。
3. **§5.1、§5.2.1、§5.3–5.4（pp. 17–25）**：看公式如何约束系统。
4. **Table 2 + evaluation configuration（pp. 25–28）**：先读条件，再读分数。
5. **Appendix C–E（pp. 43–46）**：QB 与 MoonEP 最可审计的形式化证据。

### 官方 artifact

- [Kimi K3 Hugging Face repository](https://huggingface.co/moonshotai/Kimi-K3)：weights、config、modeling files 与 model card。
- [Kimi K3 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)：核对层数、attention composition、$g_{\min}$、AttnRes block size 与 SiTU parameters。
- [Flash Linear Attention PR #691](https://github.com/fla-org/flash-linear-attention/pull/691)：论文指向的 KDA implementation。
- [MoonEP](https://github.com/MoonshotAI/MoonEP)：expert-parallel implementation。
- [AgentENV](https://github.com/kvcache-ai/AgentENV)：microVM sandbox。
- [MiniTriton](https://github.com/MoonshotAI/minitriton) 与 [nano-kpu](https://github.com/MoonshotAI/nano-kpu)：case-study artifacts。

### 最小 sanity checks

- 从 config 重建 69 KDA + 24 MLA 的 layer index；
- 验证 $g_{\min}=-5$ 时 16-token tile 的 reciprocal bound；
- 用 Figure 5 的 $m=8,n=4,k=1$ scores 重现 QB 的 $(4,3,1,0)\to(2,2,2,2)$；
- 用 synthetic segment 验证 KCP affine-map composition 与 sequential recurrence 数值一致；
- 比较官方 weights 的 total parameters、active path 与 Table 1；
- 在任何 benchmark 重跑前固定 harness、effort、temperature、top-$p$、tool、pass@$k$ 与 context management。

## Comparison Extract

| Axis | Normalized value | Conditions | Comparable now? |
|---|---|---|---|
| Model identity | Kimi K3，official open weights/API `kimi-k3` | official report/model repo，核至 2026-07-28 | yes |
| Architecture | native multimodal MoE；69 KDA + 24 Gated MLA；Block AttnRes | 93 layers，$d=7168$ | yes, if peer fields disclosed |
| Parameters | 2.78T total / 104.2B activated | 896 routed、16 active、2 shared | only against models separating total/active |
| Data | Web/Code/Math/Knowledge + vision categories | token counts、mixture、cutoff not disclosed | no quantitative comparison |
| Training compute | not disclosed | 2.5× is scaling-study relative claim | no |
| Training seqlen | 8K→64K pretraining；256K→1M cooldown | per-stage tokens not disclosed | partial |
| context window | 1,048,576 | capacity plus targeted long-context data | capacity comparable; capability conditional |
| Post-training | SFT → 9 RL experts → MOPD；QAT throughout | 3 domains × low/high/max | mechanism comparable; budget not |
| Inference-time compute | effort low/high/max；main eval max | tokens/tool calls mostly not reported | no cost-normalized global comparison |
| Main evaluation | Table 2 public + mixed third-party results | harness/tools/fallback differ | benchmark-by-benchmark only |
| Safety evidence | internal cyber evaluation + cited external assessment | broader system-card coverage limited | partial |
| Disclosure gaps | tokens、FLOPs、hardware、data mixture、many ablations | material to scale/cost conclusions | explicit |

## Self-test

1. 为什么 KDA 的 fixed-size state 使 context parallelism 的通信对象与 softmax attention 不同？
2. $g_{\min}=-5$ 如何同时改变数值稳定性与 GPU kernel 路径？
3. Gated MLA 在 3:1 hybrid 中补偿了 KDA 的什么表示边界？
4. AttnRes 的 trainable parameter、runtime source state 和 attention weight 分别是什么？
5. 为什么 SiTU-GLU 的 $\ell_\infty$ bound 是 100，它与 hard clamp 有什么不同？
6. QB 与 sign-based loss-free bias update 优化的是哪个共同 objective？为什么 bias 不进入 mixture weight？
7. QB 解决 load distribution 后，MoonEP 还解决什么问题？
8. “2.5× scaling efficiency”可以写成什么等 loss 关系？为什么不能归因给 KDA？
9. MOPD 的 reward 为什么是 on-policy dense signal，而不只是离线 teacher forcing？
10. Table 2 中至少列出三种让两个分数不可直接比较的条件。
11. 1M context window、训练 seqlen、long-context data exposure 和 long distance dependency evidence 有什么区别？
12. 若要判断 K3 的 agent gain 来自 base model 还是 scaffold，最小 discriminating experiment 是什么？

## Open questions

- **Open：** K3 的总 sampled/loss tokens、训练 FLOPs、集群规模和各阶段预算是多少？
- **Open：** Figure 7 的 scale points、拟合方程、target loss 范围与置信区间是什么？
- **Open：** KDA、AttnRes、Stable LatentMoE、data 与 optimizer 各自贡献多少？
- **Open：** KCP、MoonEP 与 hybrid prefix cache 在统一硬件上的吞吐、延迟、显存与通信收益是多少？
- **Open：** 1M context 下，能力随 required information density 和 output length 如何退化？
- **Open：** internal agent benchmarks 的 dataset、rubric、judge agreement、版本和 contamination control 能否公开？
- **Open：** MXFP4/MXFP8 QAT 相对 higher-precision checkpoint 的逐能力 loss 与 serving gain 是多少？

## References

- Kimi Team. [Kimi K3: Open Frontier Intelligence](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf). Technical report, 2026. 本文分析的主要一手来源。
- Moonshot AI. [Kimi K3 model repository](https://huggingface.co/moonshotai/Kimi-K3). 权重、model card、配置和官方实现入口。
- Moonshot AI. [Kimi K3 release entry](https://www.kimi.com/code/docs/en/kimi-code/whats-new.html). 用于核验 2026-07-16 模型发布日期；technical report 本身未在封面注明日期。
- 论文 References 中的 Kimi Linear、Attention Residuals、Kimi K2/K2.5、LatentMoE 等，是理解各组件“前序 baseline 与 K3 delta”的首选入口；本笔记没有用这些前序论文替代 K3 report 的证据。
