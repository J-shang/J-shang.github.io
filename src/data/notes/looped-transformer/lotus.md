---
title: "LOTUS：用 Gold-CoT Supervision 对齐 Looped Latent States"
description: "分析并行 latent workspace、post-loop token supervision 与 answer loss，校准 3B 数学任务上的质量和延迟结论。"
topic: "looped-transformer"
section: "llm-posttraining"
slug: "lotus"
date: 2026-08-02
updated: 2026-08-02
cutoff: 2026-08-02
order: 70
source:
  repository: "J-shang/looped-transformer"
  path: "papers/19-lotus.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-02"
  contentHash: "sha256:6f45af514437341edbf9043825048f248a9d21d4155e25116eb3187d92e11a42"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
> 论文：*Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers*<br>
> 精确版本：[arXiv:2606.31779v2](https://arxiv.org/abs/2606.31779v2)，2026-07-13<br>
> 作者与机构：Ying Fan、Anej Svete、Kangwook Lee；Microsoft Research、ETH Zürich、KRAFTON、Ludo Robotics<br>
> 官方 artifacts：[code](https://github.com/yingfan-bot/lotus) · [models](https://huggingface.co/collections/yingfanbot/looped-padded)<br>
> 证据标签：**B**。code/models 可用，实验集中在 3B math reasoning 与 gold-CoT supervision。

## 1. 30 秒结论

LOTUS 的核心想法很直接：预留 $K$ 个 latent blocks，让它们并行存在于 context 中；对整个 padded sequence 执行 $R$ 次 loop；训练结束时把每个 latent position 通过原 LM head 直接预测对应 gold CoT token。它把显式 CoT 的监督信号迁移到 latent space，又避免逐 token 生成整个 thought trace。

在 Llama-3.2-3B 上，LOTUS 的 GSM8K accuracy 接近 explicit CoT，thought-phase latency 降低 2.5×；自然语言 CoT 设置约 6.9×。但这种效率来自固定大小的并行 latent workspace、gold reasoning traces 和 math-specific curriculum，不能直接外推到开放式 chat 或任意长 reasoning。

## 2. 学习目标

1. 区分“并行 latent positions”与 autoregressive latent tokens；
2. 复原 $K$ blocks、每 block $c$ positions、$R$ loops 的计算图；
3. 理解 direct post-loop supervision 为什么优于只监督答案；
4. 判断 latent readout 是何种程度的 interpretability evidence；
5. 比较 latent-loop latency 与 explicit CoT 时统一 output budget。

## 3. 符号表

| 符号 | 含义 |
|---|---|
| $Q$ | question tokens |
| $K$ | latent blocks 数，Llama 设置为 6 |
| $c$ | 每个 block 的 latent positions，设置为 25 |
| $R$ | loop iterations，设置为 6 |
| $h_{ij}^{(R)}$ | 第 $i$ 个 block、第 $j$ 个位置在最终 loop 后的 state |
| $T_{ij}$ | 与该 latent position 对齐的 gold CoT token |
| $f_{\text{head}}$ | pretrained LM head |
| $A$ | answer suffix |

固定 latent workspace 为 $Kc=150$ positions。它不是把 150 个 thought tokens 逐个生成，而是让这 150 个位置在每次 full forward 中并行更新。

## 4. 计算图

输入模板可写成：

```text
[Q] [BoT] [lat × c] ... [lat × c] [EoT] [answer]
           block 1          block K
```

首先只对 question + latent prefix 执行 $R$ 次 loop；然后冻结最终 latent states 作为 prefix，执行一次 final forward 生成 answer。question prefix 的 KV 可复用，以降低重复成本。

![LOTUS 的 looped forward 与 final answer forward](/assets/looped-transformer/19-lotus/figure-2-architecture.png)

*原论文 Figure 2，PDF p.4，[arXiv v2 PDF](https://arxiv.org/pdf/2606.31779v2)。看图重点：所有 latent positions 在每轮并行更新；answer 不在 loop 内反复生成，而是在 post-loop prefix 上 autoregressive decode。*

## 5. 两个训练目标

### 5.1 Post-loop latent-step supervision

只在最终 loop 的 latent states 上，经 base LM head 预测 gold CoT tokens：

$$
\mathcal L_{\mathrm{step}}
=\frac{1}{N}\sum_{i,j}
\operatorname{CE}\left(f_{\mathrm{head}}(h_{ij}^{(R)}),T_{ij}\right).
$$

重要的是：监督作用在 $R$ 轮之后，而不是强迫每一轮都输出下一步。这给 recurrent dynamics 留出自由度，同时让最终 workspace 可用原 vocabulary 读出。

### 5.2 Answer loss

final forward 在 question + final latent prefix 上生成答案：

$$
\mathcal L_{\mathrm{ans}}
=-\sum_t\log p(A_t\mid Q,h^{(R)},A_{<t}).
$$

总目标：

$$
\mathcal L=\mathcal L_{\mathrm{ans}}+\lambda\mathcal L_{\mathrm{step}}.
$$

只用 answer loss 时，latent states 缺少局部 credit assignment；direct step supervision 显著提高训练可学性。

## 6. Curriculum：从 explicit CoT 逐步转 latent

训练从一个 explicit-CoT checkpoint 起步，并在多个 epoch 中逐步把显式 reasoning steps 替换成 latent blocks。论文在 GSM8K-Aug 约 385k examples 上训练 30 epochs。

这个 curriculum 同时解决：

- 初始 state 无法承载推理步骤的问题；
- 一次性移除全部 CoT 导致的 distribution shift；
- gold step 与 latent block 的对齐。

它也引入强 supervision dependence：如果 gold traces 错误、风格单一或长度超过 $Kc$，latent workspace 会继承这些偏差。

## 7. 核心结果

![LOTUS 与 explicit/latent CoT 的结果及 latency](/assets/looped-transformer/19-lotus/figure-1-main-results.png)

*原论文 Figure 1，PDF p.2，[arXiv v2 PDF](https://arxiv.org/pdf/2606.31779v2)。左图显示 LOTUS 在模型扩到 3B 时缩小 latent–explicit CoT accuracy gap；右图是自然语言 CoT。看图重点：比较对象的 supervision、模型和 reasoning format 需一致。*

### 7.1 Accuracy 与 OOD

Llama-3.2-3B、三个 seeds：

| 方法 | GSM8K | OOD average |
|---|---:|---:|
| Explicit CoT | 71.5 | 62.1 |
| LOTUS | $70.0\pm0.9$ | 63.9 |
| LOTUS + CODI | 70.6 | — |
| prior CODI + SIM | 62.3 | — |

这支持 LOTUS 在该 3B math setup 中接近 explicit CoT，同时并未明显牺牲所测 OOD math datasets。它不是 broad-domain OOD：评测仍主要围绕数学推理。

### 7.2 Latency

H100、batch size 1、greedy decoding 的 3B 结果：

| 设置 | thought phase | total |
|---|---:|---:|
| explicit math-expression CoT | 338.8 ms | 384.2 ms |
| LOTUS | 133.0 ms | 181.2 ms |

thought phase 约 2.5×，total 约 2.1×。自然语言 CoT 中，thought latency 约 963.6 ms vs 140.8 ms，即 6.9×；accuracy 68.41 vs 68.13。

优势来自显式 CoT 越长，sequential decoding 越贵；LOTUS 的固定 $Kc$ parallel workspace 对较长 traces 更有利。但如果问题需要超过固定 workspace 的推理，它会退回 autoregressive extension，速度优势会减弱。

## 8. 消融告诉了什么

| 训练方式 | GSM8K |
|---|---:|
| only answer supervision | 63.3 |
| CODI-style | 64.4 |
| direct per-iteration supervision | 68.2 |
| direct post-loop supervision | 70.0 |

post-loop direct supervision 最好，说明：

1. latent state 需要 step-level target；
2. 不必规定每一轮对应哪一个 reasoning step；
3. 让 $R$ 轮共同形成最终并行 workspace 比逐轮强制对齐更合适。

训练 recurrence 同样关键：train $R=2$ 只有 14.6，train $R=6$ 达 70.0。对 train-$R=6$ 模型，inference 从 $R=1$ 的 22.7 升到 $R=6$ 的 70.0，$R=7$ 略降到 69.3。它支持训练范围内 refinement，也再次否定“越深永远越好”。

## 9. Latent readout 应怎样解释

把最终 latent positions 经 LM head 投影，gold CoT token retrieval top-1 为 70.9%、top-5 为 85.8%；作者还报告 unseen-but-valid chains 的 top-1 15.3%、top-5 64%。

这是有价值的可读性证据，但必须保留三点：

1. states 被 $\mathcal L_{\mathrm{step}}$ 直接训练成可由 LM head 读出；
2. top-$k$ token 对齐不说明整个 hidden vector 只编码这个 token；
3. alternative valid chain 是观察性证据，不是 causal faithfulness 证明。

正确表述是“latent workspace 与 CoT token space 对齐且部分可读”，而不是“读出了模型真实、完整、忠实的思维过程”。

## 10. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| architecture | parallel padded latent blocks + looped updates | 强 |
| supervision | post-loop gold-CoT token readout | 强：关键消融 |
| efficiency | 减少 thought-phase sequential passes | 中强：H100 b1 measurement |
| capability | 3B latent CoT 接近 explicit CoT | 中强：3 seeds，math domain |
| interpretability | LM-head readout 与 alternative chains | 中：受直接监督影响 |
| generality | broad LLM post-training | 弱：尚未覆盖 |

## 11. Claim–evidence map

| Claim | Evidence | Boundary |
|---|---|---|
| LOTUS 在 3B GSM8K 缩小 latent–explicit gap | controlled table + seeds | 不等于所有 reasoning/domain |
| parallel latent workspace 降低 latency | H100 b1 measurements | hardware、batch、trace length 依赖 |
| post-loop supervision 优于 only-answer/CODI | ablation | 依赖 gold CoT availability/quality |
| latent states 对齐 CoT token | LM-head retrieval | 不构成 causal faithfulness |
| $R=6$ refinement 有效 | train/infer depth sweep | $R>6$ 不再稳定提高 |

## 12. 局限

1. 数据与任务集中于 GSM8K-style math reasoning。
2. 依赖 gold CoT，并从 explicit-CoT checkpoint 与 curriculum 起步。
3. $K$、$c$、$R$ 固定，难度和 trace length 自适应不足。
4. 不是通用 chat/tool/safety post-training 方案。
5. 对超过 latent capacity 的长推理需 fallback，效率边界未完全展开。
6. latent readout 经过直接监督，不应被当作自然涌现的可解释性。

## 13. 与显式 CoT/RL 的关系

LOTUS 没有消除显式 reasoning data：它在训练时消费 gold CoT，只在 inference 把 sequential visible trace 压入 latent workspace。它与 DeepSeek-R1 式 reasoning RL 的关系应这样理解：

```text
reasoning RL / rejection sampling 产生高质量 traces
  → LOTUS curriculum 把 trace supervision 压进 latent blocks
  → inference 用较少 sequential passes 生成 final answer
```

这是一条潜在的 distillation/compression 路线，而不是显式 reasoning data 的替代品。

## 14. 推荐扩展实验

- **宽领域**：加入 code、knowledge-intensive QA、instruction following，测是否只学到 math trace template。
- **自适应 workspace**：让 $K$ 或 $R$ 随样本退出，并测 continuous batching latency。
- **trace robustness**：对同一题提供不同正确 CoT、含噪 CoT、最短 proof，测 latent representation 是否过拟合表面措辞。
- **causal intervention**：替换/删除某个 latent block，检查对应 reasoning step 与 final answer 的因果变化。
- **post-training chain**：在相同 base 上比较 explicit SFT、LOTUS、LOTUS + DPO/RLVR。

## 15. 自测题

1. 为什么 $Kc=150$ latent positions 不等于生成 150 个 latent tokens？
2. post-loop supervision 与 per-iteration supervision 的约束区别是什么？
3. latency 优势为什么随 explicit CoT 长度增大？
4. LM-head top-1 70.9% 为什么不能证明 faithful reasoning？
5. LOTUS 的训练仍然在哪一步依赖显式 CoT？

## 16. 一句话定位

LOTUS 是 looped architecture 与 reasoning post-training 之间最清楚的桥梁之一：它证明 gold-CoT step supervision 可以塑造并行 latent workspace，但现阶段应把它视为数学推理 trace 的压缩方案，而非通用 LLM 的完整 latent replacement。
