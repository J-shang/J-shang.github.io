---
title: "Muon is Scalable for LLM Training"
description: "核查 weight decay、shape scale、distributed Muon 与 scaling law 的规模化证据。"
topic: "muon"
section: "papers"
slug: "muon-is-scalable"
legacyPaths: ["/notes/muon-is-scalable/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 68
source:
  repository: "J-shang/Muon"
  path: "论文精读/07-Muon-is-Scalable.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/07-Muon-is-Scalable.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:a3a69cd35704a1e69cfeffaad0552ebc01a7e1646230b8b0d689a151fc841d50"
  manifest: "muon"
  managed: true
---
> 原文：[arXiv:2502.16982](https://arxiv.org/abs/2502.16982)，核验版本 v1（2025-02-24）；[Moonlight 官方仓库](https://github.com/MoonshotAI/Moonlight)
> 来源类型：大规模技术预印本与官方实现
> 阅读提醒：配方和代码事实必须绑定版本；规模收益只覆盖论文的模型、数据和拟合设置。

## 为什么它是 Muon 主线论文

原始 Muon 说明给出小而清晰的算法，这篇工作回答“怎样让它在 LLM 规模真正工作”。关键贡献不是简单把模型放大，而是识别三个容易被忽略的合同：解耦 weight decay、按 shape 控制 update RMS、在分布式环境中保持完整语义矩阵的 polar。

## 1. weight decay 不是附属项

论文观察到 vanilla Muon 在长训练中 weight/output RMS 可能持续增长，加入 decoupled weight decay 后更稳定。完整更新可写成

$$
W_{t+1}=(1-\eta\lambda)W_t-\eta s(m,n)O_t.
$$

这里 decay 与 Muon direction 分开；把 $\lambda W$ 加进 gradient 后再 polar，不是同一算法。

## 2. 可核查锚点：polar update 的原生 RMS

若 $O=UV^\top\in\mathbb{R}^{m\times n}$，秩 $r=\min(m,n)$，则

$$
\|O\|_F^2=r,
$$

所以

$$
\operatorname{RMS}(O)
=\frac{\sqrt{r}}{\sqrt{mn}}
=\frac{1}{\sqrt{\max(m,n)}}.
$$

这意味着统一 $\eta$ 会让不同 shape 获得不同元素 RMS。论文据此讨论 shape adjustment，并用与 AdamW update RMS 匹配的经验尺度建立可训练 recipe。注意：match-RMS、unit-RMS、spectral scale 是不同约定，必须记录实际公式。

## 3. 分布式 Muon 的语义

polar 是非线性全矩阵操作。对 shards 分别做

$$
\operatorname{polar}([G_1;G_2])
\ne
[\operatorname{polar}(G_1);\operatorname{polar}(G_2)]
$$

一般成立。论文的 distributed 路径收集完整 gradient matrix、对完整矩阵运行 NS，再只保留对应 local update shard，并同步参数。通信与临时内存因此是算法成本的一部分。

## 4. 规模化结果怎样读

- **作者报告**：在其 scaling-law/compute-optimal 设置下，Muon 达到相同性能所需训练 FLOPs 约为 AdamW 的 52%。
- **模型案例**：Moonlight 为 3B active / 16B total MoE，报告约 5.7T tokens。
- **能支持**：存在一套可扩展 Muon recipe，在该团队设置中形成更优 compute frontier。
- **不能支持**：所有模型“固定快约 2×”；收益完全来自 polar；相同超参 drop-in 即可复现。

还要特别记住论文观察：增加 NS steps、让输出数值上更接近 exact polar，并不总让训练更好。这把“polar accuracy”和“training utility”分成两个问题。

## 关系图谱

- **规模化扩展**：把原始 Muon 发展成大规模 LLM recipe。
- **实现入口**：Moonlight distributed optimizer。
- **实验证据**：作者报告 compute-optimal frontier 改善，但只覆盖其设置。
- **后续问题**：推动了 spectral constraint、critical batch、MuonClip、DMuon 等研究。

## 精读后的任务

从论文和固定 commit 代码制作“公式—代码”对照表：每条写公式、参数名、默认值、global/local shape、通信 collective 和 state dtype。另做一张收益来源卡，记录模型、tokens、batch、baseline 调参预算、横轴和硬件。

## 自测

1. 为什么同一 LR 下 $4096\times4096$ 与 $4096\times11008$ 的 polar update RMS 不同？
2. 为什么“在每个 shard 上运行相同 NS 代码”仍可能改变算法？
3. 52% FLOPs 是什么范围的结论，缺哪些独立证据才能升级为通用结论？

**掌握标准**：能把算法方向、shape scale、decay、parameter routing、distributed semantics 五项作为一个完整 recipe 复述。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **SFT optimizer mismatch 是主文结果，不只是展望**：§3.5 显示 Muon-pretrain + Muon-SFT 在其 ablation 中最好，但当 pretrain/SFT optimizer 不一致时，Muon-SFT 没有显著优势；Qwen2.5-7B 的 Muon-SFT 与 Adam-SFT 大体相当，且部分指标更低。
2. **谱熵证据只是 association**：§3.4 观察 Muon 权重矩阵 SVD entropy 更高，并用它支持“更多方向”直觉；没有干预谱熵来证明因果。
3. **update scale 有两个层次**：先消除 shape-dependent RMS，再把 Muon RMS 经验匹配到 AdamW 约 0.2–0.4，目的是共享 LR/WD。不要把两步合成一个理论常数。
4. **Distributed Muon 的通信流程写得很具体**：算法包含 DP gather full gradient、full NS、discard nonlocal update、all-gather updated parameters；这比笼统写“支持 ZeRO”更有用。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| 原始博客把 >20B/1T、分布式、finetuning 列作 open | 本报告给 16B/5.7T、distributed implementation 和 SFT ablation | **时间上的问题推进**，不是同时冲突 | 按发布日期和 scale 记录证据 |
| *Practical Efficiency* 使用“coupled weight decay”并做 $\mu$P transfer | 本文引入 AdamW-style decoupled decay，并复用 AdamW 超参 | **recipe 与术语真实不同** | 逐式比较 decay 是进 polar 前还是参数 update 项 |
| Inexact/NS 理论：更准 polar 改善 bound | 本文观察更多 NS steps 更准确却不提升训练表现 | **指标和调参条件有张力** | 每个 precision 独立调 LR/momentum，再比较 loss-vs-time |
| Scion：实践 Muon 是 scale-invariant LMO | 本文显式添加 shape 和 match-RMS scale | **方向层一致，完整 step 尺度不同** | 去掉外部 scale 的 ablation |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| weight decay 解决长程 weight/output RMS growth | §2.2 “Weight Decay”、Figure 2 | 论文报告，机制措辞仍是作者解释 |
| 满秩 $m\times n$ polar update RMS 为 $1/\sqrt{\max(m,n)}$ | Lemma 1 与 Appendix A | 论文明确/可独立推导 |
| match AdamW RMS 约 0.2–0.4，并复用 LR/WD | §2.2 “Matching update RMS” | 论文中的经验 recipe，不是普适常数 |
| full gather → NS → discard shard 的 distributed 流程 | §2.3 Algorithm 1 | 论文明确 |
| 约 52% FLOPs | §3.2 fitted scaling-law curves | 作者在该实验范围内报告，不是逐训练 run 的普适比例 |
| 本笔记 full-vs-shard 不等式 | 论文用“requires full gradient matrix”表述 | 本文非线性综合，应由具体矩阵反例验证 |
