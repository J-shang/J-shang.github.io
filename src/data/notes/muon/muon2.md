---
title: "Muon²"
description: "研究 polar 之前的二阶统计如何改善有限步 Newton–Schulz 的输入谱。"
topic: "muon"
section: "papers"
slug: "muon2"
legacyPaths: ["/notes/muon2/"]
date: 2026-07-14
updated: 2026-07-16
cutoff: 2026-07-14
order: 75
source:
  repository: "J-shang/Muon"
  path: "论文精读/14-Muon2.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB/14-Muon2.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:4c2278b3d0bd69dfe114fc5c2f7e5920bd12fa457a88a22de961185c8de47f3f"
  manifest: "muon"
  managed: true
---
> 原文：[arXiv:2604.09967](https://arxiv.org/abs/2604.09967)，核验版本 v2（2026-06-05）
> 来源类型：2026 年前沿预印本
> 阅读提醒：算法定义可按原文复核；最高 13B 实验是作者报告，独立复现仍待完成。

## 它改变了 Muon 的哪个对象

Muon² 在 **polar 之前** 对 momentum 做 Adam-style elementwise second-moment preconditioning：

$$
M_t=\beta_1M_{t-1}+G_t,
$$

$$
V_t=\beta_2V_{t-1}+(1-\beta_2)G_t\odot G_t,
$$

$$
\widehat M_t=M_t\oslash(\sqrt{V_t}+\epsilon),
$$

然后对 $\widehat M_t$ 做有限步 NS。Muon²-F 用行列 factorized second moment 降低持久状态。

这与“polar 后按行归一化”的方法不是同一族，因为 preconditioning 会改变进入 NS 的 singular vectors/spectrum。

## 论文机制主张

作者认为原 momentum 往往 ill-conditioned，小奇异值在少量 NS steps 内难以接近目标区间。second-moment scaling 改善输入谱，因此 3-step Muon² 可达到或超过 5-step Muon 的实用方向质量。作者报告最高 13B 的 GPT/LLaMA/MoE 预训练、NS steps 减少 40%，以及达到相同 loss 时最多节省约四分之一训练时间。

这些是作者设置内结果，不应在独立复现前升级为稳定配方。

## 可核查锚点：exact polar 与 finite-step benefit 必须分开

取满秩对角矩阵

$$
M=\operatorname{diag}(10,0.1),
$$

以及任意正的逐元素对角 rescaling 后

$$
\widehat M=\operatorname{diag}(a,b),\qquad a,b>0.
$$

二者的 exact polar 都是 $I$。因此在这个特殊例子中，preconditioning 不改变 exact-polar direction，却可能显著改善有限步 NS 收敛速度。这个例子把论文最强的机制解释定位为：**conditioning finite-step approximation**，而不是笼统说“second moment 给了更好的 exact direction”。对非对角/一般 elementwise scaling，exact polar 也可能改变，需实测。

## 状态与系统权衡

标准 Muon 持久保存约一份 $mn$ momentum；full Muon² 再增加 $mn$ second moment，接近 Adam 的两份状态。Muon²-F 保存 $m+n$ 级行列统计，但重构是近似。节省 NS GEMM 是否抵消 state memory、bandwidth 与额外 elementwise kernel，必须看 end-to-end profile。

## 与 SOAP/AdaMuon 的关系

- **组合内容**：Adam 式二阶统计与 Muon polar；操作顺序决定算法。
- **与 SOAP 的相似处**：都用二阶统计改善结构化变换，但 basis/state 位置不同。
- **不要混同**：它不等于 polar 后 adaptive scaling，因为 nonlinear operations 不可交换。

## 精读后的任务

保存真实 momentum snapshots，对比 Muon/Muon² 输入 condition proxy、3/5-step polar cosine、LMO gap、step time、peak memory。加入 exact SVD 路径，判断收益来自改变 exact direction、加速 approximation，还是两者都有。

## 自测

1. 为什么 precondition 与 polar 一般不可交换？
2. 上面对角例子说明了什么，又没有说明什么？
3. Muon²-F 的 $m+n$ 状态节省会引入哪种结构近似？

**掌握标准**：能把 exact direction、finite-step conditioning、persistent state 和 wall-clock 四个效应拆开验证。

## 补充阅读：遗漏、分歧与出处

### 还值得注意什么

1. **directional alignment 是作者新定义的实用质量指标**：§3.3 用与 exact orthogonalized direction 的 cosine，刻意消除 global scaling，区别于只看 orthogonality error。
2. **Muon² 同时可能改变 target 与 approximation**：elementwise second-moment scaling 一般会改变 singular vectors，所以 exact polar 本身可能不同；论文主要强调 conditioning/alignment，但不能把收益全归于“更快逼近原 Muon target”。
3. **Muon²-F 的价值需结合实测 memory**：Table 6 报告 1B/7B 上内存接近 Muon，而 full Muon² 接近/超过 AdamW；它不是只看 $m+n$ 渐近式。
4. **作者明确承认 scale 限制**：§6 称最高 13B 仍不是 frontier industrial scale，当前结果不构成生产规模共识。

### 与其他论文哪里不同

| 对照来源或观点 | 分歧从哪里开始 | 如何理解 | 怎样验证 |
|---|---|---|---|
| Moonlight：更精确 orthogonalization 未必更好训练 | Muon² 增加 second-moment history，且可能改变 exact direction，不只是加 NS steps | **看似冲突但 intervention 不同** | exact SVD + precondition-on/off 2× steps factorial |
| SOAP：在 Shampoo eigenbasis 运行 Adam | Muon² 在原坐标 elementwise precondition 后做 polar | **结构相似但 basis/state 位置不同** | 记录 second moment 所在 basis 和旋转时序 |
| NorMuon/AdaMuon 的 adaptive scaling | pre-polar vs post-polar | **非交换操作产生真实算法差异** | 比较 $\operatorname{polar}(D(M))$ 与 $D(\operatorname{polar}(M))$ |
| finite-NS theory 把 target 固定为原 momentum polar | Muon² 改了进入 NS 的矩阵 | **理论对象不同** | 分别以 $\operatorname{polar}(M)$ 和 $\operatorname{polar}(\widehat M)$ 作 target |

### 本文内容从哪里来

| 本文讲到的内容 | 原文位置 | 来源说明 |
|---|---|---|
| second-moment preconditioning 发生在 orthogonalization 前 | §3.1 Algorithm 1 | 论文明确 |
| Muon²-F 使用 Adafactor-style row/column factorization | §3.6 | 论文明确 |
| 最多 13B、NS steps 减少 40%、最高节省约 25% GPU-hours | Abstract、Tables 3/5/6、§4.6 | 作者在该实验范围内报告 |
| conditioning 改善与 directional alignment | §3.3–3.5 | 作者机制分析 + snapshot 证据，因果仍需隔离 |
| 本笔记对角矩阵 exact-polar 不变例子 | 原文无此例 | 本文反例，只对正对角 special case 成立 |
| “收益可能来自 target 改变与近似加速两者” | 论文重点偏后者 | 本文比较，仍待验证；需 exact-SVD factorial 实验 |
