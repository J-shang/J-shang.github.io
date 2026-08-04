---
title: "Looped Transformer 论文解析索引"
description: "组织 21 篇 Looped Transformer 论文，从机制背景推进到通用 LLM 预训练、checkpoint 改造与 latent 后训练。"
topic: "looped-transformer"
section: "guide"
slug: "paper-reading-guide"
date: 2026-07-29
updated: 2026-08-04
cutoff: 2026-08-02
featured: true
order: 0
source:
  repository: "J-shang/looped-transformer"
  path: "papers/README.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-04"
  contentHash: "sha256:d5e3b9168850c0a8f45cf4d094c4547c82ba515f8d7432e9401d2255081fa8f4"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
> **定位更新（2026-08-02）：** 01–13 保留为机制与历史背景，14–21 是面向通用 LLM pretraining、retrofit、latent supervision 与 adaptive compute 的新主线。先读[系统学习课程](/topics/looped-transformer/)，查一手来源时使用[通用 LLM 资料地图](/topics/looped-transformer/)。
>
> 共 21 篇，全部按 `analyze-research-paper` 最新规范处理：明确版本与证据范围，提供早期符号表、贡献账本、方法复原、最小算例、实验问题—证据链、claim–evidence map、局限和可证伪扩展。共选 45 张真正承担解释作用的论文原图；图片下方均标注图号、PDF 页码、精确来源、看图重点与证据边界。
>
> 01–13 信息截止 2026-07-24，视觉复核 2026-07-29；14–21 版本与数值复核截止 2026-08-03，视觉复核截止 2026-08-02。图源与裁切说明见[视觉证据资产清单](/topics/looped-transformer/visual-evidence-sources/)。

八项核心工作已有一份按 Paper 时间、architecture、loop、training recipe、performance 与 insight 统一归一化的[横向综述](/topics/looped-transformer/looped-transformer-eight-paper-survey/)；需要公式、原图、表格与逐条证据时再进入下方单篇解析。DeepLoop 因模型规模较小，只保留在背景论文索引中。

跨论文的作者、机构、引用量与重要性比较见：[论文重要性评估](/topics/looped-transformer/paper-importance-analysis/)（引用快照：2026-07-28）。

## 论文总表

| # | 论文解析 | 在知识链中的角色 | 最强结论 | 最重要边界 |
|---:|---|---|---|---|
| 01 | [Attention Is All You Need](/topics/looped-transformer/attention-is-all-you-need/) | 标准 Transformer baseline | self-attention 去掉 token 方向 recurrence | 没有 depth loop 或参数共享 |
| 02 | [Universal Transformers](/topics/looped-transformer/universal-transformers/) | depth recurrence 祖先 | 共享 transition 与 per-position ACT 可行 | 小规模、halting 不等于硬件加速 |
| 03 | [ALBERT](/topics/looped-transformer/albert/) | 参数共享对照 | 参数量与执行深度可解耦 | 少参数不等于少 FLOPs/算法学习 |
| 04 | [Looped Transformers as Programmable Computers](/topics/looped-transformer/looped-transformers-as-programmable-computers/) | 表达能力 | 构造性权重可执行通用程序 | 存在性不等于 SGD 可学 |
| 05 | [Learning Learning Algorithms](/topics/looped-transformer/looped-transformers-learning-algorithms/) | 经验可学习性 | input injection 等配方可学逐步更新 | OOD scale 变化暴露非通用算法 |
| 06 | [Multi-step Gradient Descent](/topics/looped-transformer/looped-transformers-multistep-gradient-descent/) | 理论可学习性 | 简化设定中训练收敛到 preconditioned GD | linear/Gaussian/population/gradient flow |
| 07 | [Length Generalization](/topics/looped-transformer/looped-transformers-length-generalization/) | 深度—长度联动 | tasks 所需 step 随长度增长时应增加 loop | 依赖所选 $n$-RASP-L 类与 step label |
| 08 | [Reasoning with Latent Thoughts](/topics/looped-transformer/reasoning-with-latent-thoughts/) | latent reasoning | reasoning 可能更依赖 effective depth | 不是免费 FLOPs，也非可读 hidden CoT |
| 09 | [Deep Equilibrium Models](/topics/looped-transformer/deep-equilibrium-models/) | 隐式 fixed point | equilibrium + implicit gradient 可省 activation memory | solver 更慢，终态不等于有限轨迹 |
| 10 | [Block-Recurrent Transformers](/topics/looped-transformer/block-recurrent-transformers/) | sequence recurrence 对照 | recurrent states 改善长文本记忆 | 循环沿 token blocks，不沿 depth |
| 11 | [LayerNorm Provably Learns the Power Method](/topics/looped-transformer/layernorm-power-method/) | normalization 与算法偏置 | 受控设定中训练选出 power method | 实际是 RMSNorm-like linear model |
| 12 | [DeepLoop](/topics/looped-transformer/deeploop-depth-scaling/) | residual scaling | aligned tied visits 的保守 exponent 为 $1/2$ | 一阶充分界、核心 alignment 未直接测 |
| 13 | [Loop the Loopies!](/topics/looped-transformer/loop-the-loopies/) | 大规模 MoE 与系统共设计 | layer-loop + measured wall-clock recipe 胜过 matched vanilla | 非 analytical-FLOP matched，v2 极新 |
| 14 | [Huginn / Recurrent Depth](/topics/looped-transformer/huginn-recurrent-depth/) | 从头 recurrent pretraining | 3.5B/800B 上建立 task-dependent latent depth scaling | proof of concept；无完整 post-training |
| 15 | [Ouro / Looped Language Models](/topics/looped-transformer/ouro-looped-language-models/) | multi-trillion-token model-loop | 7.7T staged training + adaptive gate 得到强 1.4B/2.6B base/Thinking | 超过 train depth 常退化；code/data 未完全开放 |
| 16 | [Mixture-of-Recursions](/topics/looped-transformer/mixture-of-recursions/) | token-level dynamic compute | parameter sharing、routing 与 selective KV cache 可联合设计 | 1.7B 强配置并非稳定胜 vanilla；吞吐口径有限 |
| 17 | [Retrofitted Recurrence](/topics/looped-transformer/retrofitted-recurrence/) | pretrained-model surgery | pretrained init + curriculum/healing 可把 vanilla base 改成 recurrent | 约 1B、~50B tokens、math-heavy |
| 18 | [LoopUS](/topics/looped-transformer/loopus/) | 轻量 pretrained LLM adaptation | selective gate + random deep supervision 跨 Qwen/Phi 有效 | 部分 retained benchmarks 下滑；context 1K |
| 19 | [LOTUS](/topics/looped-transformer/lotus/) | latent CoT supervision | parallel latent blocks 在 3B math 上接近 explicit CoT 并降 thought latency | gold-CoT、固定 workspace、math domain |
| 20 | [LoopRPT](/topics/looped-transformer/looprpt/) | latent-step reinforcement | hard-token quality 提升且平均 exit steps 下降 | 仅 4,428 条 OMNI-MATH；不是 broad pretraining |
| 21 | [LoopFormer](/topics/looped-transformer/loopformer/) | elastic-depth trajectory | shortcut consistency 让单 checkpoint 支持多个 user budgets | 训练约 1.5× FLOPs，仍略逊 non-looped base |

## 通用 LLM 主线

1. 从头训练：[Huginn](/topics/looped-transformer/huginn-recurrent-depth/) → [Ouro](/topics/looped-transformer/ouro-looped-language-models/) → [Mixture-of-Recursions](/topics/looped-transformer/mixture-of-recursions/) → [Loop the Loopies!](/topics/looped-transformer/loop-the-loopies/)。
2. 复用 checkpoint：[Retrofitted Recurrence](/topics/looped-transformer/retrofitted-recurrence/) → [LoopUS](/topics/looped-transformer/loopus/)。
3. Post-training 与 compute control：[LOTUS](/topics/looped-transformer/lotus/) → [LoopRPT](/topics/looped-transformer/looprpt/) → [LoopFormer](/topics/looped-transformer/loopformer/)。

跨论文统一比较、实验模板与自测见[Looped LLM 研究手册](/topics/looped-transformer/)。

## 四条机制背景阅读路线

### 最短机制路线

1. [Universal Transformers](/topics/looped-transformer/universal-transformers/)
2. [Learning Learning Algorithms](/topics/looped-transformer/looped-transformers-learning-algorithms/)
3. [Length Generalization](/topics/looped-transformer/looped-transformers-length-generalization/)
4. [Reasoning with Latent Thoughts](/topics/looped-transformer/reasoning-with-latent-thoughts/)
5. [Loop the Loopies!](/topics/looped-transformer/loop-the-loopies/)

这条路线从架构原型走到 toy algorithms、泛化、language reasoning，再到大规模 MoE。若目标是研究通用 LLM 的完整训练流程，应先走[新资料地图的阅读路线](/topics/looped-transformer/#6-建议的阅读顺序)，再按问题回到这里补机制。

### 理论主线

1. [Programmable Computers](/topics/looped-transformer/looped-transformers-as-programmable-computers/)：能表示。
2. [Multi-step Gradient Descent](/topics/looped-transformer/looped-transformers-multistep-gradient-descent/)：在简化设定中能优化得到。
3. [LayerNorm Power Method](/topics/looped-transformer/layernorm-power-method/)：normalization 如何选择算法。
4. [DeepLoop](/topics/looped-transformer/deeploop-depth-scaling/)：共享 visits 如何改变深度稳定性。

### 动力系统主线

1. [Universal Transformers](/topics/looped-transformer/universal-transformers/)：显式有限 recurrence。
2. [Learning Learning Algorithms](/topics/looped-transformer/looped-transformers-learning-algorithms/)：input-injected fixed-point-like behavior。
3. [Deep Equilibrium Models](/topics/looped-transformer/deep-equilibrium-models/)：直接求 equilibrium。
4. [DeepLoop](/topics/looped-transformer/deeploop-depth-scaling/)：超深 loop 的 residual parameterization。

### 工程主线

1. [ALBERT](/topics/looped-transformer/albert/)：参数、FLOPs、速度分开核算。
2. [Block-Recurrent Transformers](/topics/looped-transformer/block-recurrent-transformers/)：state/cache/step-time。
3. [DeepLoop](/topics/looped-transformer/deeploop-depth-scaling/)：训练稳定性。
4. [Loop the Loopies!](/topics/looped-transformer/loop-the-loopies/)：stored depth、checkpointing、microbatch、parallelism 和 wall-clock。

## 统一审稿问题

读任意一篇时都回答：

1. 循环索引沿 depth、sequence block，还是 root-solver iteration？
2. 保存多少组不同参数，实际执行多少次 block？
3. state 的 shape 是什么；原输入是否每步注入？
4. 训练监督在哪些 steps；推理能否改变 loop 数？
5. baseline 固定的是参数、block calls、analytical FLOPs，还是实测 wall-clock？
6. 论文建立的是表达性、训练收敛、经验关联，还是系统测量？
7. 数值是否有 seeds/error bars；外部 benchmark 是否数据和 posttraining 可比？
8. 超过训练 loop、length、scale 或硬件环境后，哪个假设最先失效？

## 两个需要特别保留的勘误式提醒

- [Deep Equilibrium Models](/topics/looped-transformer/deep-equilibrium-models/) 同时报告显存大降与 wall-clock 更慢，不能只引用“最高 88% memory reduction”。
- [Loop the Loopies!](/topics/looped-transformer/loop-the-loopies/) v2 的 Stage 1 有 “3T” 与 “570B × 4 = 2.28T” 两种口径；后者加 Stage 2 的 1.263T 才与 Table 3 的 3.5T 总量一致，复现前需要作者/config 澄清。
