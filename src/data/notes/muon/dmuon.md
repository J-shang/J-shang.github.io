---
title: "DMuon"
description: "理解保持完整矩阵语义时，分布式 Muon 如何把额外开销压到接近 AdamW。"
topic: "muon"
section: "papers"
slug: "dmuon"
legacyPaths: ["/notes/dmuon/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 76
source:
  repository: "J-shang/Muon"
  path: "论文精读/15-DMuon.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/15-DMuon.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:4fb92d15a79f91274f762419aff3f813403e42034475f14b61736bb2af86799d"
  manifest: "muon"
  managed: true
---
> 原文：[arXiv:2606.27153](https://arxiv.org/abs/2606.27153)，核验版本 v1（2026-06-25）；[官方代码](https://github.com/X-Square-Robot/dmuon)
> 来源类型：2026 年系统预印本与官方实现
> 阅读提醒：设计和代码可核对；性能数字是作者硬件与工作负载上的报告，仍需独立复现。

## 它解决什么问题

Muon 的 NS 需要完整矩阵级耦合，而 ZeRO/FSDP/TP 通常围绕 elementwise optimizer 设计。朴素实现可能重复计算 NS、频繁聚合完整矩阵，并让最大矩阵成为 straggler。DMuon 的目标是在不改变 Muon update 语义的前提下优化 owner、通信、Gram kernel、调度与 batching。

## 系统设计主线

1. **owner-centric**：每个完整语义矩阵指定 owner，避免多个 rank 重复做 NS。
2. **fine-grained layout/overlap**：让参数/梯度通信与 forward/backward 尽可能重叠。
3. **symmetric Gram kernel**：利用 $XX^\top$ 或 $X^\top X$ 对称性减少 dominant GEMM 工作。
4. **computation-aware load balance**：按矩阵 shape/NS cost 分配 owner，避免大矩阵集中。
5. **auto-tuning 与 batching**：合并小矩阵 kernel launch，按 shape 选择配置。

## 可核查锚点：full 与 blockwise 不是同一算法

对 global momentum

$$
M=\begin{bmatrix}2&0\\1&1\end{bmatrix},
$$

full polar 是 $M(M^\top M)^{-1/2}$。若按行分成两个 $1\times2$ blocks，各自 polar 只是把每行归一化：

$$
\begin{bmatrix}1&0\\1/\sqrt2&1/\sqrt2\end{bmatrix}.
$$

该 blockwise 结果的列通常不满足 full polar 的正交关系，因此一般不等于 global polar。分布式优化若宣称“保持精确语义”，就应在不同 world size/partition 下重建同一 global update。

## 论文报告怎样读

作者在 embodied foundation model 与 LLM 工作负载、最高 256 张 A800 上报告接近 AdamW step latency。分解实验称：symmetric Gram kernel 贡献约 48% optimizer-time reduction，owner scheduling/load balance 约 32%，auto-tuning/NS batching 约 16%；平均 end-to-end step-time 只比 AdamW 高约 2%。

这些数字依赖矩阵 shape 分布、硬件、并行策略、batch 和 baseline kernel。论文也明确指出不可消除的临界路径：最大 owner-side NS 至少仍要计算一次；单卡收益较弱。

## 正确性先于性能

性能测试前必须通过：

- world size 1/2/4 的 global update 对照；
- 不同 owner assignment 得到相同参数；
- odd/rectangular/fused QKV/MoE matrices；
- BF16/FP32 容差与 deterministic seed；
- parameter、momentum、update shard round trip；
- optimizer state checkpoint 恢复后下一步一致。

只有正确性相同，kernel speedup 才能归为 `implementation`；若 blockwise/low-rank 改变方向，应标为 `algorithmic approximation`。

## 与 Moonlight distributed Muon 的关系

- **优化对象**：full-matrix distributed Muon 的实现路径。
- **保持不变**：论文目标是数学等价 update，而不是提出新 optimizer。
- **性能上限**：受最大矩阵 NS 临界路径和硬件拓扑限制。

## 精读后的任务

制作每步 timeline：gradient ready、collective start/end、owner NS、update scatter、parameter sync。分别报告 optimizer kernel time、通信、整步时间和 overlap ratio；在相同 global batch 下比较 AdamW、朴素 Muon、DMuon。

## 自测

1. 为什么 blockwise polar 不能仅靠“每块都正交”证明等于 full polar？
2. owner-centric 方案减少了什么重复，又引入什么负载均衡问题？
3. 2% step overhead 能否推出 2% training overhead？还取决于什么？

**掌握标准**：能同时给出 distributed correctness invariant 和 end-to-end performance breakdown。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **DMuon 的创新是等价重排，不是新 optimizer**：§5.3 明确称其为 mathematically equivalent reformulations，保留 exact optimizer semantics，只消除系统开销。
2. **symmetric Gram kernel 是最大单项收益**：§5.2 报告其占 optimizer-time reduction 的 48%；owner scheduling/load balance 32%，autotuning/batching 16%。这给出可复现的 profile 假设。
3. **不可消除的 critical path 被明确指出**：最大 owner-side matrix 的 NS 至少做一次；规模继续增大时，剩余 overhead 由该矩阵主导。
4. **性能结论依赖 workload/GPU count**：Table 1 覆盖四类 workload、8–256 A800；单 GPU 缺乏 distributed overlap/ownership收益，作者在限制中明确承认。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| Moonlight distributed Muon：DP gather full matrix后计算 | DMuon 分配 matrix owner、overlap并优化 layout/kernel | **同语义的系统替代方案**，不是算法冲突 | world-size invariant global update |
| blockwise/low-rank distributed variants | DMuon 声称 exact full-matrix semantics；其他方法可能改 target | **算法近似 vs 实现等价的真边界** | exact SVD/full NS reference comparison |
| Muon blog 的“FLOP overhead <1%”估算 | DMuon 显示朴素 distributed 实现可比 fwd/bwd 更贵 | **FLOPs 与 realized latency/communication 的指标冲突** | kernel FLOPs、launch、comm、end-to-end 分解 |
| “within 2% of AdamW”作为通用结论 | 论文是受测 workload 的平均 step-time overhead | **范围外外推不成立** | 新硬件/shape/topology 独立 profile |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| owner-centric communication、symmetric Gram、load balance、batching | §3 System Design | 论文明确 |
| 保持 exact optimizer semantics | contributions、§5.3 Limitations | 作者明确主张，仍应以 bit/tolerance test 验证实现 |
| 平均 end-to-end step cost 距 AdamW 约 2% | §5.1–5.2、Table 1/2 | 作者硬件/工作负载报告，非 universal |
| 48%/32%/16% speedup breakdown | §5.2 Table 2 | 作者 ablation 报告 |
| 本笔记 $2\times2$ blockwise 反例 | 原文无此矩阵 | 本文数学反例，用来解释 full 与 blockwise 不等价 |
| “DMuon inherits Muon convergence exactly” | §5.3 作者表述 | 只在数值容差内确实保持同一 update 时成立 |
