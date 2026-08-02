---
title: "LoopUS：把标准 Pretrained LLM 重组为 Latent Refinement Model"
description: "研究 encoder–loop–decoder 重组、selective gate、随机深度监督与 confidence exit，同时保留能力退化的负面证据。"
topic: "looped-transformer"
section: "llm-retrofit"
slug: "loopus"
date: 2026-08-02
updated: 2026-08-02
cutoff: 2026-08-02
order: 61
source:
  repository: "J-shang/looped-transformer"
  path: "papers/18-loopus.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-02"
  contentHash: "sha256:c2c94ec70472eb8be96dd56d1fa96ab7c57fa28e65128049061b7945d428f76e"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
> 论文：*LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models*<br>
> 精确版本：[arXiv:2605.11011v1](https://arxiv.org/abs/2605.11011v1)，2026-05-10<br>
> 作者与机构：Taekyhun Park、Yongjae Lee、Dohee Kim、Hyerim Bae；Pusan National University、Changwon National University<br>
> 官方 artifacts：[project](https://thrillcrazyer.github.io/LoopUS) · [code](https://github.com/Thrillcrazyer/LoopUS) · [models](https://huggingface.co/collections/Thrillcrazyer/loopus)<br>
> 证据标签：**B**。多种 pretrained model、code 与 weights 可用；论文仍是极新的 preprint。

## 1. 30 秒结论

LoopUS 不删除大段层并重训，而是根据层间表示变化，把 pretrained LLM 分成 encoder $E$、可重复 middle block $M$ 和 decoder $D$。在 $M$ 每轮输出上加入 selective gate，训练时从 20 个可能 steps 中随机挑 5 个做 deep supervision，推理时由 confidence head early exit。

它在 Qwen3 1.7B/4B/8B、Phi-4 14B 等模型上用 3B FineWeb-Edu tokens 完成结构适配，平均 reasoning-oriented benchmark 提升约 1.6–2.2 points。与此同时，部分 MMLU、HellaSwag 会下降；所以它证明的是“较低 token budget 可形成 latent loop”，不是“预训练能力完全无损”。

## 2. 学习目标

1. 理解 encoder–loop–decoder split 的依据；
2. 推导 selective gate 为什么能阻尼 state update；
3. 理解 random deep supervision 如何节省 memory；
4. 区分 LM loss、monotonicity loss 和 confidence loss；
5. 用 retained-capability 表而非单一平均分判断 retrofit 是否成功。

## 3. 符号表

| 符号 | 含义 |
|---|---|
| $E,M,D$ | encoder、recurrent middle block、decoder |
| $h_b$ | 第 $b$ 次 loop 前的 hidden state |
| $\delta_b=M(h_b)-h_b$ | 当前 block 提议的 state update |
| $\Delta_b$ | input-dependent positive step scale |
| $A<0$ | 可学习的衰减参数 |
| $\alpha_b$ | 保留 block 输出的 gate |
| $B$ | 最大训练 loop，论文设为 20 |
| $K$ | 每个 batch 实际监督的随机 steps，论文设为 5 |

## 4. 从层间表示选择 loop block

论文先观察 pretrained LLM 的相邻层 cosine distance：早期层把 embedding 转为 contextual state，中间层的表示演化相对平缓，末层更接近输出空间。于是把中间连续层作为 $M$：

$$
h_0=E(x),\qquad h_{b+1}=G\bigl(h_b,M(h_b)\bigr),\qquad
p(y)=D(h_B).
$$

这是一种 empirical layer-role hypothesis。cosine similarity 能说明方向相似度，不能证明某组中间层天然是收敛算子；因此作者还需要 gate 与多 step supervision。

![LoopUS 架构、随机监督与 confidence exit](/assets/looped-transformer/18-loopus/figure-2-selective-loop.png)

*原论文 Figure 2，PDF p.4，[arXiv v1 PDF](https://arxiv.org/pdf/2605.11011v1)。看图重点：左图是 encoder–loop–decoder 重组，右图同时显示随机选择的 supervised steps、monotonic loss 和 confidence head。不是每个 step 都保留完整梯度图。*

## 5. Selective gate

首先计算 middle block 的更新提议：

$$
\delta_b=M(h_b)-h_b.
$$

由该更新本身预测正的步长，再映射成衰减 gate：

$$
\Delta_b=\operatorname{softplus}(W_\Delta\delta_b+b_\Delta),
$$

$$
\alpha_b=\exp(\Delta_b\odot A),\qquad A<0.
$$

最终更新是：

$$
h_{b+1}=\alpha_b\odot M(h_b)+(1-\alpha_b)\odot h_b.
$$

由于 $\Delta_b>0$、$A<0$，有 $0<\alpha_b<1$，所以每次更新是旧 state 与新提议的插值。它能降低剧烈 state drift，但不等价于 contraction：$M$ 的 Jacobian、gate 对 $h$ 的导数和多维耦合仍可能使整体 Lipschitz constant 大于 1。

### 最小例子

若某维度 $h_b=2$、$M(h_b)=6$：

- $\alpha=0.8$ 时，$h_{b+1}=5.2$，大幅接受更新；
- $\alpha=0.1$ 时，$h_{b+1}=2.4$，接近保持原 state。

gate 学习的是“这次 block update 应接受多少”，不是直接判断答案正确与否。

## 6. Random deep supervision

最大循环 $B=20$，但每个 batch 随机选择 $K=5$ 个 steps 保存 gradient；其余 steps 在 `no_grad` / detach 下推进 state。这样计算仍执行，但 activation memory 与 backward cost 下降。

优点：模型在多个 depth 都收到监督，不只适应最后一步。风险：未被选择的早期 update 不接收端到端梯度；训练目标优化的是随机稀疏路径，和完整 20-step gradient 不同。

## 7. 三类 loss

### 7.1 每步 next-token loss

被选中的 step $b$ 经 decoder/head 计算：

$$
\mathcal L_{\mathrm{LM}}^{(b)}
=-\sum_i\log p_\theta(x_{i+1}\mid x_{\le i},h_b).
$$

### 7.2 Monotonicity loss

作者对相邻 step 的 loss 变化加惩罚，形式包含：

$$
\operatorname{SiLU}\left(\mathcal L_{\mathrm{LM}}^{(b)}-
\mathcal L_{\mathrm{LM}}^{(b-1)}\right).
$$

它鼓励后一步不劣于前一步。由于这是 soft empirical objective，而不是对 state operator 的数学约束，不能保证所有 token、所有 inference depths 单调改善。

### 7.3 Confidence loss

confidence head 预测当前 step 是否已足够好，target 来自 token accuracy，再用 binary cross-entropy 训练。推理时达到阈值即可退出；论文常用 threshold 0.6、最大 8 steps。

这里的 calibration 很关键：token-level correctness proxy 与 sequence-level task success 不完全一致。高 confidence 可能只表示 next-token 容易，不代表整个 reasoning trajectory 已完成。

## 8. 训练与评测

| 项目 | 设置 |
|---|---|
| starting models | Qwen3 1.7B/4B/8B、Phi-4 14B、TinyLlama 等 |
| adaptation data | 3B FineWeb-Edu tokens |
| context window | 1024 |
| optimizer | AdamW |
| train max loops | 20 |
| supervised loops / batch | 5 |
| inference max loops | 8 |
| early-exit threshold | 0.6 |

这个阶段在论文中叫 post-training framework，但从 objective 与 data 看，更准确的通用术语是**architecture adaptation / continued LM training**，不是 instruction preference alignment。

## 9. 结果怎样解读

七个 benchmark 平均分相对 base 提升：Qwen3 1.7B +1.6、4B +1.8、8B +2.2、Phi-4 14B +1.7。它说明 3B-token adaptation 能让多个 backbone 使用 loops。

但 retained capability 不是全正向：

| 模型 | 指标 | Base | LoopUS |
|---|---|---:|---:|
| Qwen3-4B | MMLU | 68.3 | 67.7 |
| Qwen3-4B | HellaSwag | 52.1 | 51.4 |
| Qwen3-8B | MMLU | 72.8 | 71.5 |
| Qwen3-8B | HellaSwag | 57.2 | 56.0 |
| Phi-4-14B | HellaSwag | 63.1 | 60.5 |

因此“平均分上升”可能同时包含 reasoning gains 与 knowledge/commonsense regressions。研究通用 LLM 时必须保留每项结果和 post-training 后能力。

## 10. Adaptive scaling

![LoopUS 的 recurrence-depth 与 adaptive exit](/assets/looped-transformer/18-loopus/figure-5-adaptive-scaling.png)

*原论文 Figure 5，PDF p.8，[arXiv v1 PDF](https://arxiv.org/pdf/2605.11011v1)。看图重点：不同任务最佳 depth 不同，adaptive mode 平均约 3.39 次 recurrence；更深后曲线趋于平稳。该图支持 task-dependent stopping，不等于每个样本都被正确分配。*

平均 3.39 steps 相比固定 8 steps 有 theoretical saving，但真实部署还要考虑同一 batch 内不同退出时间、cache 整理和 GPU utilization。

## 11. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| retrofit | 多种标准 pretrained LLM → looped model | 强：多 family + artifacts |
| stability | selective damped gate | 中强：有消融；无 contraction guarantee |
| efficient training | random deep supervision | 中：省 backward memory，仍执行 forward |
| adaptive inference | confidence early exit | 中：平均 depth 可测，部署收益待验证 |
| generality | 多类 base benchmark | 中：部分 retained capabilities 下滑 |

## 12. Claim–evidence map

| Claim | Evidence | Boundary |
|---|---|---|
| 3B tokens 可适配多个 pretrained families | Qwen/Phi/TinyLlama experiments | context 仅 1024，非完整 model flow |
| selective gate 稳定多步循环 | ablations/curves | 不证明数学收敛 |
| adaptive exit 减少平均 loops | mean 3.39 | 不等于 wall-clock 同比减少 |
| reasoning-oriented average 提升 | seven-task average | 个别 MMLU/HellaSwag 退化 |
| LoopUS 保持 pretrained capabilities | 多数任务尚可 | “完全无损”不成立 |

## 13. 局限

1. adaptation 只用 3B FineWeb-Edu、context 1024，data 与长 context 范围有限。
2. 没有统一 instruction SFT、DPO/RLVR、multi-turn chat、tool use 与 safety 评测。
3. confidence target 是 token-level proxy。
4. 跨论文 TinyLlama 对比并非 controlled comparison，作者也只作为 reference。
5. 真实 serving 中 dynamic depth 的 scheduler/KV overhead 未充分测量。

## 14. 推荐实验

1. 选择 fully-open base，同 data 做 vanilla continued training control；
2. 对 split location、gate、random supervision、monotonic loss 做全因子消融；
3. 画每任务、每 difficulty bucket 的 quality–depth curve；
4. 测 confidence calibration、overthinking false-positive/false-negative；
5. 在 SFT/DPO/RLVR 每 stage 复测 retained capabilities；
6. 在 batch size 1 与 continuous batching 下报告真实 latency。

## 15. 自测题

1. 为什么 $0<\alpha<1$ 仍不能证明整个 update 是 contraction？
2. random deep supervision 节省的是 forward 还是 backward 资源？
3. token accuracy 为什么不是完美的 exit target？
4. 平均 benchmark +2 与若干 MMLU/HellaSwag 下降能否同时成立？
5. 为什么把 3B FineWeb-Edu adaptation 直接称为 alignment post-training 会混淆？

## 16. 一句话定位

LoopUS 提供了目前较轻量、跨多种 pretrained LLM 的 latent-loop 改造方案；它的主要研究价值是 retrofit 工程与稳定性，而其“通用能力保持”和部署加速仍需更严格验证。
