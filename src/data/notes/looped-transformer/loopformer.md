---
title: "LoopFormer：用 Shortcut Consistency 训练 Elastic-Depth Trajectories"
description: "还原 time/step-size conditioning 与 shortcut consistency，核查多预算推理的平滑性及额外训练计算成本。"
topic: "looped-transformer"
section: "llm-posttraining"
slug: "loopformer"
date: 2026-08-02
updated: 2026-08-02
cutoff: 2026-08-02
featured: true
order: 72
source:
  repository: "J-shang/looped-transformer"
  path: "papers/21-loopformer.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-02"
  contentHash: "sha256:0dcd421ccfd498f578166c1b41a70a08b0b21123cbed36414f737d5d9e443e36"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
> 论文：*LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation*<br>
> 精确版本：[arXiv:2602.11451v1](https://arxiv.org/abs/2602.11451v1)，2026-02-11；ICLR 2026<br>
> 作者与机构：Ahmadreza Jeddi、Marco Ciccone、Babak Taati；University of Toronto、Vector Institute、University Health Network<br>
> 官方 artifact：[project page](https://loopformer.github.io/)<br>
> 证据标签：**C**。论文为正式会议版本，但模型约 1B、data 为 25B-token Pile subset，未覆盖现代 post-training。

## 1. 30 秒结论

普通 looped model 往往只适应一个固定 loop count；提前退出会得到未成熟 state，超过训练深度又可能 drift。LoopFormer 把 recurrence 看作从 $t=0$ 到 $t=1$ 的 trajectory，每个 step 同时输入 normalized time $t$ 与 step size $\Delta t$。训练时既跑最长 trajectory，也随机跑一条较短 shortcut，并让短路线 final state 匹配长路线。

结果是一个用户可指定 global budget 的 elastic-depth model，在不同 loops 下比其他 looped baselines 更平滑。但它仍略逊于相同执行深度的 non-looped Transformer；训练还因双 trajectory 增加约 1.5× FLOPs、1.3× wall-clock。它解决的是 trajectory robustness，不是已经实现 per-token adaptive serving。

## 2. 学习目标

1. 理解 time/step-size conditioning 与普通 loop index embedding 的区别；
2. 推导 full/shortcut/consistency 三项 loss；
3. 区分 user-selected global budget 与 learned early exit；
4. 审核 token-matched 与 training-FLOP-matched 结果；
5. 判断 representation trajectory analysis 的证据等级。

## 3. 符号表

| 符号 | 含义 |
|---|---|
| $k$ | 每次 loop 内 stored Transformer blocks 数 |
| $L$ | 最大 loops |
| $M$ / $S$ | inference / sampled shortcut budget，$\le L$ |
| $t_i$ | normalized trajectory time |
| $\Delta_i$ | 第 $i$ 次 step size，$\sum_i\Delta_i=1$ |
| $\Phi_k$ | 由 $k$ 个 shared blocks 构成的 update |
| $h^{(L)},h^{(S)}$ | full 与 shortcut 的 final states |

作者记 $(k\otimes L)$ 为 $k$ blocks 重复 $L$ 次，近似执行深度 $kL$。

## 4. Trajectory conditioning

每次 update：

$$
h^{(i)}=\Phi_k\left(h^{(i-1)};t_{i-1},\Delta_i\right),
\qquad \sum_{i=1}^{M}\Delta_i=1.
$$

$t$ 与 $\Delta$ 经过 sinusoidal embeddings 和 MLP，生成 RMSNorm scales 与 residual gates。相同 shared block 因“当前走到哪里、这一步跨多大”而改变行为。

这与 diffusion/ODE shortcut 的类比是：长路线用许多小步，短路线用少数大步，都应接近 $t=1$ endpoint。类比提供设计灵感，不说明 hidden-state dynamics 真的是某个已知 ODE 的数值解。

![LoopFormer 的 shortcut modulation 与不同预算 trajectories](/assets/looped-transformer/21-loopformer/figure-1-architecture.png)

*原论文 Figure 1，PDF p.4，[arXiv v1 PDF](https://arxiv.org/pdf/2602.11451v1)。看图重点：每次 loop 接收 $t$、$\Delta t$ conditioning；右图表示不同数量 steps 都要走到同一 normalized endpoint。*

## 5. Shortcut-consistency training

每个 batch 同时计算：

1. 最大长度 $L$ 的 uniform trajectory；
2. 随机 $S\sim U\{1,\ldots,L-1\}$ 的 shortcut trajectory；
3. shortcut schedule $\Delta^S$ 满足总步长为 1。

两条路线各自计算 next-token loss：

$$
\mathcal L_L=\operatorname{CE}(\operatorname{LMHead}(h^{(L)}),Y),
$$

$$
\mathcal L_S=\operatorname{CE}(\operatorname{LMHead}(h^{(S)}),Y).
$$

短路线还匹配 stop-gradient full state：

$$
\mathcal L_{\rm cons}
=\left\|\operatorname{stopgrad}(h^{(L)})-h^{(S)}\right\|_2^2.
$$

总目标：

$$
\mathcal L=\mathcal L_L+0.1\mathcal L_S+0.1\mathcal L_{\rm cons}.
$$

full trajectory 是 batch 内 teacher，shortcut 自蒸馏其 endpoint。stop-gradient 避免 full route 为迁就 short route 而被同时拉动。

## 6. 与 early exit 的区别

LoopFormer 没有为每个 token 或 sequence 学一个 halting policy。用户在 inference 前选择 $M$ 和 schedule；整个 batch 使用同一 global budget。

| 方法 | budget 决策者 | 粒度 | 主要优势 |
|---|---|---|---|
| Ouro/LoopUS gate | 模型 | sequence 或 token proxy | 自动分配 |
| MoR router | 模型 | token | 最细粒度 |
| LoopFormer | 用户/系统 | 整个请求或 batch | 可控、无需 gate calibration |

所以 “elastic” 是多预算可用，不等于 “adaptive” 已学会按难度分配。

## 7. 训练与 baseline

| 项目 | 设置 |
|---|---|
| model | NanoGPT-style，约 1B |
| data | deduplicated Pile subset，25B tokens |
| context window | 1024 |
| hardware | 4×H100 80GB |
| loop configs | $k\in\{1,2,3\}$，$L\in\{8,12,24\}$ |
| non-loop baseline | 24-layer Transformer |
| tasks | Pile/FineWeb-Edu/OpenWebText perplexity + 10 zero-shot tasks |

论文用 $kL$ 近似 inference FLOPs，忽略 embedding/unembedding cost。小模型时这些固定成本占比不一定可忽略，因此最终还要看实测 latency。

## 8. 主要结果

在 $(3\otimes8)$、24× execution budget：

| 模型 | Pile PPL ↓ | avg zero-shot acc ↑ |
|---|---:|---:|
| Base $(24\otimes1)$ | 9.49 | 45.27 |
| TMLT $(3\otimes8)$ | 10.38 | 44.69 |
| LoopFormer $(3\otimes8)$ | 10.28 | 44.81 |

LoopFormer 是 looped variants 中最好，但仍略逊于 non-looped base。12× budget 时，LoopFormer PPL 11.12、avg 43.73；Base $(12\otimes1)$ 为 9.98、44.93。parameter sharing 对 memorization/perplexity 的差距没有消失。

![不同 block/loop 配置的 perplexity 与 reasoning 曲线](/assets/looped-transformer/21-loopformer/figure-2-3-schedule-results.png)

*原论文 Figures 2–3，PDF p.7，[arXiv v1 PDF](https://arxiv.org/pdf/2602.11451v1)。看图重点：同一 $k$ 下增加 budget 通常平滑改善，LoopFormer 比其他 elastic loop baselines 稳；non-looped base 点仍是关键上界/对照。*

不同 step schedules 在相同 budget 下仍可能造成约 1.4 PPL、近 3 PPL 或约 1.3 accuracy points 的 spread（依配置而变）。因此只给 budget $M$ 不足以复现，还要给完整 $\Delta$ schedule。

## 9. 训练 overhead 与 FLOP-matched 结果

每个 batch 同时执行 $L$ 与随机 $S$ 两条 trajectory。论文估计训练 FLOPs 约是 fixed-loop/vanilla 的 1.5×，作者 4×H100 设置下 wall-clock 约慢 1.3×。

FLOP-matched 时，LoopFormer 只训练 34k iterations，baselines 为 50k，因此少看约 8B tokens：

| 模型 | avg zero-shot acc |
|---|---:|
| Base | 45.27 |
| TMLT | 44.69 |
| LoopFormer | 44.21 |

这张表校正了“同 tokens 训练”对 LoopFormer 的额外计算优势。它表明 elastic training 的 robustness 有成本，并没有在该规模下总体超过 non-looped base。

## 10. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| conditioning | time + step-size modulation | 强 |
| training | full/shortcut consistency self-distillation | 强 |
| elasticity | 单 checkpoint 支持 $M\le L$ 多预算 | 中强：完整 curves |
| efficiency | user-controllable depth | 中：无 learned per-token allocation |
| scaling | 1B/25B Pile | 中低：规模与 data recipe 较旧 |
| systems | overhead accounting | 中强：FLOPs 与 wall-clock 都报告 |

## 11. Claim–evidence map

| Claim | Evidence | Boundary |
|---|---|---|
| shortcut consistency 防止短路线 collapse | 多 budget curves/ablations | 不证明任意 schedule 都稳定 |
| 额外 loops 平滑改善 | Figures 2–3 | non-looped base 仍略强 |
| 单模型支持用户指定 budget | inference algorithm | 不是按样本自动 adaptive |
| representation trajectory 更稳定 | distance/probe analysis | correlational，非 causal mechanism |
| elastic training 值得计算成本 | token-matched results | FLOP-matched 后优势减弱 |

## 12. 局限

1. 约 1B、25B Pile、context 1024，不代表 modern multi-trillion-token LLM。
2. 不含 SFT/DPO/RLVR、chat、tools、safety。
3. 双 trajectory 训练增加约 1.5× FLOPs。
4. schedule selection 仍需人工/系统设定，且相同 budget 下结果差异显著。
5. looped model 仍落后同 execution depth 的 non-looped perplexity/平均准确率。
6. representation analysis 是相关性证据，不能说明模型在执行具体 reasoning algorithm。

## 13. 推荐扩展

- 把 LoopFormer conditioning 加到 pretrained-model retrofit，减少从头训练成本。
- 用 learned scheduler 在若干候选 $M,\Delta$ 间选择，再测 calibration 与 batching。
- 比较 hidden-state consistency、logit consistency、task-success consistency 三种 target。
- 在相同 data order 下给 token-matched、FLOP-matched、wall-clock-matched 三套曲线。
- 经过相同 SFT/DPO/RLVR 后检查 elastic range 是否保持。

## 14. 自测题

1. $t$ 与 $\Delta t$ 分别告诉 shared block 什么信息？
2. 为什么 consistency target 对 full route 使用 stop-gradient？
3. elastic depth 与 adaptive depth 的区别是什么？
4. token-matched 领先、FLOP-matched 落后能否同时成立？
5. 为什么 schedule 也必须作为 inference hyperparameter 报告？

## 15. 一句话定位

LoopFormer 给 Looped LM 增加了“同一 checkpoint、多个可控 compute budgets”的 trajectory training 方法；它最可信的贡献是稳定的深度弹性，而不是在当前规模上击败 vanilla LLM。
