---
title: "Looped Transformer 论文视觉证据来源"
description: "记录 45 张论文原图裁切对应的正式 PDF、版本、页码与处理边界，便于回查图像证据。"
topic: "looped-transformer"
section: "supplements"
slug: "visual-evidence-sources"
date: 2026-07-29
updated: 2026-08-02
cutoff: 2026-08-02
order: 81
source:
  repository: "J-shang/looped-transformer"
  path: "papers/assets/README.md"
  revision: "9ab82eeb3178ddd627b592ac2cba22de91e7be66+working-tree"
  syncedAt: "2026-08-02"
  contentHash: "sha256:7b26eaf4b332188466bfec328b6621ff34489165bebebf230d428cd752953f6b"
  manifest: "looped-transformer"
  dirty: true
  managed: true
---
本目录只保存论文原图的忠实裁切，用于相邻 Markdown 解析的自包含阅读。01–13 于 2026-07-29 以 240 DPI 处理，14–21 于 2026-08-02 以 216 DPI 处理；未重绘、未生成式补全，也未改变图中的数据、坐标轴、图例或 caption。原始 PDF 仅用于本地处理，不纳入仓库。14–21 的重复提取入口是 [`scripts/extract_new_paper_figures.py`](https://github.com/J-shang/looped-transformer/blob/main/scripts/extract_new_paper_figures.py)。

| 解析 | PDF 版本与来源 | 收录图（PDF 页码） |
|---|---|---|
| 01 Attention Is All You Need | [arXiv:1706.03762v7](https://arxiv.org/pdf/1706.03762v7) | Figure 1（p. 3）、Figure 2（p. 4） |
| 02 Universal Transformers | [arXiv:1807.03819v3](https://arxiv.org/pdf/1807.03819v3) | Figure 1（p. 2）、Figure 3（p. 6） |
| 03 ALBERT | [arXiv:1909.11942v6](https://arxiv.org/pdf/1909.11942v6) | Figure 1（p. 4）、Figure 2（p. 9） |
| 04 Looped Transformers as Programmable Computers | [ICML 2023 PMLR 发表版](https://proceedings.mlr.press/v202/giannou23a/giannou23a.pdf) | Figure 3（p. 5）、Figure 5（p. 17） |
| 05 Learning Learning Algorithms | [arXiv:2311.12424v3](https://arxiv.org/pdf/2311.12424v3) | Figure 1（p. 2）、Figure 4（p. 5） |
| 06 Multi-step Gradient Descent | [arXiv:2410.08292v1](https://arxiv.org/pdf/2410.08292v1) | Figure 2（p. 8）、Figure 3（p. 9） |
| 07 Length Generalization | [arXiv:2409.15647v5](https://arxiv.org/pdf/2409.15647v5) | Figure 1（p. 2）、Figure 4（p. 8）、Figure 5（p. 10） |
| 08 Reasoning with Latent Thoughts | [arXiv:2502.17416v1](https://arxiv.org/pdf/2502.17416v1) | Figure 1（p. 2）、Figure 3（p. 9） |
| 09 Deep Equilibrium Models | [NeurIPS 2019 发表版](https://papers.nips.cc/paper/2019/file/01386bd6d8e091c2ab4c7c7de644d37b-Paper.pdf) | Figure 1（p. 6）、Figure 2（p. 8） |
| 10 Block-Recurrent Transformers | [NeurIPS 2022 camera-ready](https://proceedings.neurips.cc/paper_files/paper/2022/file/d6e0bbb9fc3f4c10950052ec2359355c-Paper-Conference.pdf) | Figure 1（p. 2）、Figure 3（p. 8） |
| 11 LayerNorm / Power Method | [arXiv:2606.00605v1](https://arxiv.org/pdf/2606.00605v1) | Figure 1（p. 12）、Figure 2（p. 13） |
| 12 DeepLoop | [arXiv:2607.13491v1](https://arxiv.org/pdf/2607.13491v1) | Figure 1（p. 2）、Figure 2（p. 12） |
| 13 Loop the Loopies! | [arXiv:2607.16051v2](https://arxiv.org/pdf/2607.16051v2) | Figure 1（p. 4）、Figure 3（p. 9）、Figure 4（p. 10） |
| 14 Huginn / Recurrent Depth | [arXiv:2502.05171v2](https://arxiv.org/pdf/2502.05171v2) | Figure 2（p. 3）、Figure 7（p. 8） |
| 15 Ouro / Looped Language Models | [arXiv:2510.25741v5](https://arxiv.org/pdf/2510.25741v5) | Figure 3（p. 4）、Figure 4（p. 8） |
| 16 Mixture-of-Recursions | [arXiv:2507.10524v3](https://arxiv.org/pdf/2507.10524v3) | Figure 1（p. 1）、Figure 4（p. 8） |
| 17 Retrofitted Recurrence | [arXiv:2511.07384v1](https://arxiv.org/pdf/2511.07384v1) | Figure 1（p. 1）、Figure 7（p. 9）、Figure 8（p. 10） |
| 18 LoopUS | [arXiv:2605.11011v1](https://arxiv.org/pdf/2605.11011v1) | Figure 2（p. 4）、Figure 5（p. 8） |
| 19 LOTUS | [arXiv:2606.31779v2](https://arxiv.org/pdf/2606.31779v2) | Figure 1（p. 2）、Figure 2（p. 4） |
| 20 LoopRPT | [arXiv:2603.19714v1](https://arxiv.org/pdf/2603.19714v1) | Figure 2（p. 4）、Figures 5–6（p. 8） |
| 21 LoopFormer | [arXiv:2602.11451v1](https://arxiv.org/pdf/2602.11451v1) | Figure 1（p. 4）、Figures 2–3（p. 7） |

每张图片的具体解释、证据边界和“看图重点”均写在对应论文解析的图片下方。页码指 PDF 文件页序，不是论文正文印刷页码。
