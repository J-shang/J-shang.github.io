---
title: "LoopRPT：把 Reinforcement Signal 放进 Latent Recurrent Steps"
description: "复原 hard-token selection、EMA teacher、step reward 与 exit policy，明确小规模数学 continued training 的证据边界。"
topic: "looped-transformer"
section: "llm-posttraining"
slug: "looprpt"
date: 2026-08-02
updated: 2026-08-04
cutoff: 2026-08-02
order: 71
source:
  repository: "J-shang/looped-transformer"
  path: "papers/20-looprpt.md"
  url: "https://github.com/J-shang/looped-transformer/blob/eb43191df7da7f8d1b936fa6485ea21f7c8f430a/papers/20-looprpt.md"
  revision: "eb43191df7da7f8d1b936fa6485ea21f7c8f430a"
  syncedAt: "2026-08-04"
  contentHash: "sha256:9543b7770189361d71fea9717537b9a6b15f0379b7769e79c15fc1f8471eeeda"
  manifest: "looped-transformer"
  managed: true
---
> 论文：*LoopRPT: Reinforcement Pre-Training for Looped Language Models*<br>
> 精确版本：[arXiv:2603.19714v1](https://arxiv.org/abs/2603.19714v1)，2026-03-20<br>
> 作者与机构：Guo Tang、Shixin Jiang、Heng Chang 等；Harbin Institute of Technology、Tsinghua University、HKUST (Guangzhou)<br>
> 官方 artifact：论文未给出独立 code/model release<br>
> 证据标签：**C**。方法和 appendix 较完整，但实验仅从 Ouro-1.4B/2.6B 出发，在 4,428 条 OMNI-MATH 上短程训练。

## 1. 30 秒结论

传统 RLVR 奖励最终生成答案，LoopRPT 则把每个 latent loop step 当成 action/value-improvement 的对象。它用 EMA teacher 定义参考 exit step，用 next-token log-prob gain 减去 difficulty-aware time penalty 得到逐步 reward；再通过 Gaussian latent rollouts 训练 exit policy，并用正 advantage 加权 intermediate representation。

在 Ouro-2.6B 上，它让 hard-token peak accuracy 从 34.52 提到 38.10，并把 adaptive accuracy 从 34.35 提到 37.24、平均 exit 从 3.51 降到 2.28。方法有效，但“reinforcement pre-training”这个名字容易误导：实质是一个 math-specialized、极小数据集上的 reinforcement continued-training stage，不是 foundation pretraining。

## 2. 学习目标

1. 区分 output-token RLVR 与 latent-step reinforcement；
2. 推导 hard-token selection、teacher reference、accuracy gain 与 time penalty；
3. 理解 Gaussian rollouts 怎样为离散 exit policy 提供 on-policy variation；
4. 分开 exit-policy learning 与 representation learning；
5. 审核小数据短训练对“general-purpose RPT”主张的限制。

## 3. 符号表

| 符号 | 含义 |
|---|---|
| $\theta,\bar\theta$ | student 与 EMA teacher parameters |
| $K$ | 最大 latent steps，Ouro 设置为 4 |
| $\pi_\theta(k)$ | 在 step $k$ 退出的概率 |
| $\tau$ | cumulative exit threshold |
| $H_t$ | teacher 对 token position $t$ 的 entropy |
| $\rho$ | 选为 hard tokens 的比例，设置为 0.2 |
| $t_{\rm ref}$ | teacher 认为足够的 reference step |
| $\ell_\theta^{(k)}$ | step $k$ 对 ground-truth token 的 log-prob |
| $R(k)$ | step-wise reward |
| $G$ | noisy rollouts 数，设置为 8 |

## 4. Hard-token selection

对位置 $t$，EMA teacher 的 entropy：

$$
H_t=-\sum_v p_{\bar\theta}(v\mid x_{<t})
\log p_{\bar\theta}(v\mid x_{<t}).
$$

每个 example 内只选择 top-$\rho$ 高 entropy token。这样把训练集中在 teacher 不确定的位置，减少大量标点、模板 token 的弱信号。

边界：高 entropy 不等于真正需要 reasoning。多义词、罕见实体、噪声 label 也会高 entropy；而某些困难题的关键 token 可能被模型自信地答错。

## 5. EMA teacher 与 step reward

teacher reference step 由 cumulative exit distribution 决定：

$$
t_{\rm ref}=\min\left\{k:\sum_{j\le k}\pi_{\bar\theta}(j)\ge\tau\right\}.
$$

以 teacher 在该 step 的正确 token log-prob 为 baseline：

$$
b_{\rm ref}=\ell_{\bar\theta}^{(t_{\rm ref})},\qquad
\Delta_{\rm acc}(k)=\ell_\theta^{(k)}-b_{\rm ref}.
$$

时间惩罚为：

$$
C(k)=\lambda_t(k-t_{\rm ref}),
$$

$$
d_t=\operatorname{Clamp}\left(\frac{H_t}{\log|V|},0,1\right),\qquad
\lambda_t=\lambda_{\rm base}\left[1+\lambda_{\rm scale}(1-d_t)\right].
$$

容易 token 的 $d_t$ 小，所以 $\lambda_t$ 大，更强迫早退；困难 token 得到较宽松的额外计算。总 reward：

$$
R(k)=\Delta_{\rm acc}(k)-C(k).
$$

![LoopRPT 的 hard-token、EMA 与 reward 计算](/assets/looped-transformer/20-looprpt/figure-2-training-framework.png)

*原论文 Figure 2，PDF p.4，[arXiv v1 PDF](https://arxiv.org/pdf/2603.19714v1)。看图重点：student 与 EMA teacher 在相同 token 上产生不同 step log-prob；reward 同时包含 predictive gain 与相对 reference 的时间成本。*

## 6. Noisy latent rollouts 与 exit policy

对 latent state 注入 Gaussian noise：

$$
h^{(k)}\leftarrow h^{(k)}+\epsilon^{(k)},qquad
\epsilon^{(k)}\sim\mathcal N(0,\sigma^2I),quad \sigma=0.1.
$$

每个 token 产生 $G=8$ 条 rollout，各自从 $\pi_\theta^{(g)}$ 采样 exit step $t^{(g)}$，并查表得到 $r^{(g)}=R(t^{(g)})$。组内标准化 advantage：

$$
A^{(g)}=\frac{r^{(g)}-\operatorname{mean}_g r^{(g)}}
{\operatorname{std}_g r^{(g)}+\epsilon}.
$$

policy-gradient loss：

$$
\mathcal L_{\rm PG}=-\mathbb E_g
\left[A^{(g)}\log\pi_\theta^{(g)}(t^{(g)})\right].
$$

noise 的角色是创造局部 trajectory variation，使 exit policy 有可比较的 actions；它优化的是 Gaussian-smoothed objective。若 noise distribution 与真实 state uncertainty 不匹配，robustness 结论也只在局部成立。

## 7. Representation 与 regularization

除 exit policy 外，论文用跨 step 标准化的正 advantage 加权 representation learning，让有较高 $R(k)$ 的 latent states 更能预测正确 token。总 loss 组合：

$$
\mathcal L=alpha\mathcal L_{\rm PG}
+\beta\mathcal L_{\rm rep}
+\gamma\mathcal L_{\rm ent}
+\delta\mathcal L_{\rm KL}.
$$

- $\mathcal L_{\rm ent}$ 防止 exit distribution 过早塌缩；
- $\mathcal L_{\rm KL}$ 把 student 限制在 EMA teacher 邻域；
- EMA momentum 0.995，让 reference 缓慢移动。

它比只训练 gate 更深：backbone latent states 也被 reward 塑造。但多 loss 的 improvement attribution 需要消融，且 reward 仍以 next-token log-prob 为核心，不等于 sequence-level mathematical correctness。

## 8. 训练规模的真实含义

| 项目 | 设置 |
|---|---|
| starting checkpoints | Ouro-1.4B、Ouro-2.6B |
| data | OMNI-MATH 4,428 problems with solutions |
| validation | 同数据集中 held-out 200 examples |
| epochs | 3 |
| max sequence length | 4096 |
| hard-token ratio | 0.2 |
| noisy rollouts | 8 |
| hardware | 8×A100 80GB |
| wall-clock | 约 2h（1.4B）/ 4h（2.6B） |

因此名称中的 “pre-training” 是作者方法名，不能与 Chinchilla/Llama/Qwen 式 broad-corpus pretraining 混用。更准确的 stage 定位：

```text
Ouro base checkpoint
  → small math-domain reinforcement continued training
  → improved latent exit/representation
```

## 9. 结果

### 9.1 Hard-token next-token reasoning

| Ouro-2.6B | Peak acc | Adaptive acc | Avg step |
|---|---:|---:|---:|
| base | 34.52 | 34.35 | 3.51 |
| + LoopRPT | 38.10 | 37.24 | 2.28 |

hard tokens 上 quality 与 compute 同时改善，是论文最有力的证据。easy/medium 同样上升，但 harder bucket gain 更大。

![不同 token 难度的 accuracy–compute 与逐步表现](/assets/looped-transformer/20-looprpt/figure-5-6-performance.png)

*原论文 Figures 5–6，PDF p.8，[arXiv v1 PDF](https://arxiv.org/pdf/2603.19714v1)。看图重点：LoopRPT 把提升前移到较早 recurrence，并改善 adaptive Pareto；forced deeper 的曲线并不始终单调。*

### 9.2 Downstream transfer

2.6B：GSM8K 81.76→85.36，HumanEval+ 70.13→71.95。general tasks 增益多小于 1 point，例如 MMLU +0.06、MMLU-Pro +0.12、BBH +0.26、ARC-C +0.77。

这些结果与“小 math-domain stage 主要改善 math/code”一致。general tasks 未明显崩溃是好信号，但小幅上升不能证明广泛通用能力增强。

## 10. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| credit assignment | reward 直接作用于 latent steps | 强：方法定义清楚 |
| adaptive compute | difficulty-aware time penalty + exit policy | 中强：hard-token Pareto 改善 |
| optimization | noisy rollouts、EMA、entropy/KL | 中：多组件消融有效 |
| transfer | math/code 下游提升 | 中：小数据且 domain-adjacent |
| general-purpose claim | general task 小幅保持/提升 | 弱到中：非 broad training |
| reproducibility | appendix 较全 | 中：无官方 code release |

## 11. Claim–evidence map

| Claim | Evidence | Boundary |
|---|---|---|
| latent-step RL 可改善 Ouro | controlled Ouro base comparison | 只测两种小模型与 math data |
| gain 不只是提前退出 | peak acc 与 early-step acc 都提高 | reward 仍基于 token log-prob |
| hard tokens 受益更大 | entropy buckets | entropy 不是完美 difficulty measure |
| compute 压缩到更少 loops | adaptive 3.51→2.28 | 未给 production serving latency |
| 是 general-purpose reinforcement pretraining | 方法可作用于 text tokens | 实验不足以支持广泛 general-purpose 结论 |

## 12. 局限与风险

1. 仅 4,428 个数学问题，可能有 domain overfit 与 contamination 风险。
2. teacher/student 同源，EMA baseline 可能强化已有错误与 calibration bias。
3. token entropy 与 task reasoning difficulty 不完全等价。
4. next-token reward 不直接验证最终 proof/answer 正确。
5. 无官方 code/model，复现多 loss 和 rollout details 成本较高。
6. forced deeper 仍可能退化，方法没有得到全局单调改进保证。

## 13. 推荐扩展

- 在 broad FineWeb/Dolma token stream 上做 truly general RPT，并保留 data contamination audit。
- 用 verifier-based sequence reward 与 token reward 混合，比较 latent credit placement。
- 将 high entropy、loss improvement、verifier uncertainty 三种 hard-token selector 做对照。
- 与 gate-only training、ordinary continued LM、output-only RLVR 做相同 tokens/FLOPs/wall-clock comparison。
- 在 SFT/DPO/RLVR 前后检查 exit calibration 是否漂移。

## 14. 自测题

1. 为什么容易 token 的 time penalty 更大？
2. EMA teacher 同时承担哪两个角色？
3. Gaussian noise 如何产生 policy-gradient 所需 variation？
4. 4,428 条 OMNI-MATH 为什么不足以支撑“foundation pretraining”结论？
5. Peak 与 Adaptive 同时提高能排除哪一种伪解释，又不能排除什么？

## 15. 一句话定位

LoopRPT 是当前最直接的 latent-step reinforcement 方案之一，证明 reward placement 可以改善 Ouro 的早期 state 与 exit；但它仍是 math-specialized proof of concept，而不是通用 LLM reinforcement pretraining 的完成形态。
