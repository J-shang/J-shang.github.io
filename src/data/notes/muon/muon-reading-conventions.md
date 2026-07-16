---
title: "Muon 术语与阅读约定"
description: "统一常用术语，并区分论文明确、作者报告、本文推导、跨论文比较与仍待验证的解释。"
topic: "muon"
section: "research-practice"
slug: "muon-reading-conventions"
legacyPaths: ["/notes/muon-reading-conventions/"]
date: 2026-07-16
updated: 2026-07-16
order: 79
source:
  repository: "J-shang/Muon"
  path: "notes/术语与阅读约定.md"
  url: "https://github.com/J-shang/Muon/blob/97e51478028b351af529830cf14917daff8dd5ef/notes/%E6%9C%AF%E8%AF%AD%E4%B8%8E%E9%98%85%E8%AF%BB%E7%BA%A6%E5%AE%9A.md"
  revision: "97e51478028b351af529830cf14917daff8dd5ef"
  syncedAt: "2026-07-16"
  contentHash: "sha256:831e5f9ab43d19953f1fbe1ff645c22221d01b900c7964db03f90acbc28b655c"
  manifest: "muon"
  managed: true
---
这份短表只解决两件事：让不同笔记使用同一套自然术语；让读者一眼看出一句话来自论文、代码、本文推导，还是尚未验证的解释。

## 常用术语

正文以中文解释为主，但保留下列已经约定俗成、且在代码和论文中更容易检索的英文词：

| 推荐写法 | 不优先使用 | 原因 |
|---|---|---|
| learning rate | 学习率、步长率 | 与论文、配置和 API 一致 |
| baseline | 基线 | 避免与几何“基”混淆 |
| schedule | 调度表、学习率策略（泛称时） | 与训练配置一致 |
| sequence length | 序列长度（作为配置字段时） | 与实验表和代码字段一致 |
| optimizer state | 优化器状态（首次出现可并写） | 便于和 temporary buffer 区分 |
| update | 更新量、更新方向（需看语境） | 不与 gradient 或 parameter change 混用 |
| polar factor | 极分解方向因子 | 公式附近可并写，后文保留英文 |

中文句子仍应完整自然，不堆叠同义词。例如写“记录 learning rate、schedule 和总步数”，不写“记录学习率（learning rate / LR）调度策略”。

## 一句话从哪里来

- **论文明确**：原文直接给出定义、算法、定理、实验设置或限制，并附 section、equation、table 或代码入口。
- **作者报告**：原文展示了结果，但只覆盖作者的模型、数据、硬件、实现和调参范围。
- **本文推导**：由已写出的公式、shape、字节数、代码分支或数值例子直接得到；不是作者原话。
- **跨论文比较**：把多个来源放到同一对象、假设和指标下比较；需要列出参与比较的来源。
- **仍待验证**：当前证据不足、定义不一致或存在竞争解释。

来源类型和结论边界是两件事。同行评审论文也可能只报告特定实验；预印本中的恒等式也可以直接复核。正文优先写清范围，不用标签代替解释。

## 冲突怎样写

先找两篇工作最早不同的地方：研究对象、假设、尺度、指标、近似或实现版本。只有在这些条件对齐后仍给出不兼容结论，才称为冲突；否则应写成定义差异、条件差异或后续证据扩展。
