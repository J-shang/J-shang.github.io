# Kimi K3 visual evidence assets

本目录保存 `kimi-k3-2026.md` 选择性嵌入的论文原图。图片用于让阅读笔记在不反复切换 PDF 的情况下仍能检查关键机制和实验图。

## Source identity

- Source: `k3_tech_report.pdf`
- Official source: <https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf>
- SHA-256: `38621eb5be601a5dcd5c795fc10b692d124430014ff9eb035b0ce38c72ec2eaf`
- PDF metadata creation date: `2026-07-27`
- Extraction date: `2026-07-28`
- Source class: Moonshot AI official technical report

所有 PNG 均直接从上述 PDF 以 240 DPI 渲染并裁剪；未重绘、未生成、未修改坐标、图例、数据或 caption。

## Asset map

| Asset | Paper location | Role in note | Crop `(x, y, W, H)` |
|---|---|---|---|
| `figure-02-k3-architecture.png` | Figure 2, p. 3 | architecture overview | `(220, 165, 1600, 1500)` |
| `figure-03-bounded-kda-decay.png` | Figure 3, p. 5 | KDA numerical and kernel mechanism | `(220, 160, 1600, 720)` |
| `figure-05-quantile-balancing.png` | Figure 5, p. 8 | QB worked example | `(220, 170, 1600, 830)` |
| `figure-07-scaling-efficiency.png` | Figure 7, p. 11 | pretraining scaling claim | `(220, 150, 1600, 810)` |
| `figure-08-rl-scaling.png` | Figure 8, p. 13 | RL trend and non-monotonicity | `(220, 150, 1600, 700)` |
| `figure-12-prefix-cache.png` | Figure 12, p. 23 | hybrid-cache state boundary | `(220, 1100, 1600, 675)` |
| `figure-13-score-cost.png` | Figure 13, p. 32 | score–cost evidence | `(220, 1040, 1600, 1240)` |

Crop coordinates are pixels in the 240-DPI Poppler render.

## Reproduction

Each asset uses the same command shape:

```bash
pdftoppm \
  -f <page> -l <page> -singlefile \
  -r 240 -png \
  -x <x> -y <y> -W <width> -H <height> \
  k3_tech_report.pdf <output-prefix>
```

The rendered images were visually checked for complete axes, legends, annotations, captions, and absence of adjacent page text.
