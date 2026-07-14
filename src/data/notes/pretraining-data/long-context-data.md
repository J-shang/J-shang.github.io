---
title: "长上下文数据：能力窗口之外的暴露账本"
description: "分离 context window、长度 curriculum、自然长文、拼接方式、attention mask 与 loss exposure。"
topic: "pretraining-data"
section: "cross-cutting"
slug: "long-context-data"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 81
readtime: 15
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/long-context-data.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/long-context-data.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:bf8888c64cd56ece1c1be1e035c8562b14a58b69a871ede67c91ebed3037c898"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`unified-framework`
> 案例锚点：Meta Llama、MiniMax、Apple AFM、NVIDIA Nemotron、GLM、Hunyuan、Kimi；开放控制：AI2 OLMo/Dolma

## Motivating problem

“支持1M context”是服务/架构能力，不回答模型训练时见过多少1M序列、其中多少是真实长文、多少是短文拼接或synthetic retrieval，也不回答多少positions真正进入loss。

最小反例：两个模型都接受128K输入。

- A只改变RoPE并用短文随机拼接；
- B用books/repos等自然长结构，混入short replay并做分阶段长度curriculum。

相同窗口不能推出相同长依赖能力、source coverage或短任务保持。

## 六个正交轴

```text
serving/model max context
sequence-length schedule
source-document length/structure
construction: natural / concat / thematic pack / synthetic
attention mask + position scheme
loss mask + non-padding exposure
```

任一横向表若只保留第一轴，信息损失过大。

## 核算

对长度bucket $b$，令iterations为 $n_b$、每iteration nominal positions为 $B_b$、non-padding率为 $u_b$、loss mask率为 $m_b$：

$$
T_{\text{sampled},b}=n_bB_b,
$$

$$
T_{\text{loss},b}=n_bB_bu_bm_b.
$$

另定义自然长文占比：

$$
p_{\text{natural-long},b}
=\frac{T_{\text{sampled from intact long sources},b}}
{T_{\text{sampled},b}}.
$$

没有packer日志、padding和mask，不能由max length反推loss exposure。

## 长数据构造 taxonomy

| 类型 | 保留的结构 | 典型能力 | 风险 |
|---|---|---|---|
| intact book/paper/PDF | section与远距语义 | narrative/document reasoning | rights、OCR、章节泄漏 |
| repository/issue/PR | 文件与变更图 | repo reasoning/SWE | commit/test contamination |
| random concat | 几乎无跨文档关系 | position/throughput适配 | 学到忽略远距内容 |
| thematic/best-fit packing | topic或长度近似 | 减少padding、弱跨段关联 | packer诱导伪关系 |
| synthetic QA/needle | 可控依赖 | retrieval/ICL诊断 | task monoculture、答案位置捷径 |
| agent/tool trajectory | state/action长依赖 | multi-step execution | 环境版本、失败轨迹mask |
| short replay | 短域能力保持 | anti-forgetting | 稀释长结构exposure |

## 案例恢复

| 案例 | 长度与tokens | source/construction | 仍未知 |
|---|---|---|---|
| [Llama 3.1](/topics/pretraining-data/meta-llama-data-practices/) | 约800B，六阶段8K→128K | long data细节有限；长/短eval恢复 | 每阶段tokens、length histogram、natural/pack ratio |
| [Llama 4](/topics/pretraining-data/meta-llama-data-practices/) | Scout 10M/Maverick 1M capability | specialized mid-training data | tokens/curriculum/source比例 |
| [MiniMax Text-01](/topics/pretraining-data/minimax-data-practices/) | 300B@128K、32B@512K、26B@1M | short/medium/long buckets与source interpolation | 各类natural/packed比例、loss tokens |
| [Apple AFM 2024](/topics/pretraining-data/apple-foundation-models-data-practices/) | 100B@32K | continued replay + licensed long + synthetic long QA；多数原文短于32K | natural/synthetic比例 |
| [Apple AFM 2025](/topics/pretraining-data/apple-foundation-models-data-practices/) | up to65K，token未知 | books/code、synthetic ICL/retrieval、continued replay | stage budget/length distribution |
| [Nemotron 3 Ultra](/topics/pretraining-data/nvidia-nemotron-data-practices/) | 33B；92% iterations@1M、8%@4K | 46% long-category/54% phase2 replay | length×category联合分布 |
| [GLM-5](/topics/pretraining-data/glm-data-practices/) | 1T@32K→500B@128K→50B@200K | books/papers/docs、repo/SWE、synthetic dependency/agents | natural/synthetic与loss比例 |
| [Hunyuan TurboS](/topics/pretraining-data/tencent-hunyuan-hy-data-practices/) | 30B@32K+20B@256K | curated natural long；short:long=3:1 | inclusion/order细节 |
| [Kimi k1.5/K2](/topics/pretraining-data/moonshot-kimi-data-practices/) | 4K→32K→128K；K2末段60B@32K | natural long+synthetic QA/summary；full/partial attention 40/60 | 40/60不是source/token比例 |
| [OLMo/Dolma控制](/topics/pretraining-data/ai2-olmo-dolma-data-practices/) | 固定data/config/order/checkpoints可追 | 开放loader与日志校准核算字段 | 不能代表frontier long recipe |

## 恢复案例时的框架损失

- MiniMax直接给length buckets；Nemotron分别给category mix和iteration-length mix，不能压成同一“long%”。
- Kimi的40/60同时含attention treatment和source construction，不能当natural/synthetic比例。
- Hunyuan的short:long是stage mixture，而Llama的800B是阶段total；分母不同。
- OLMo控制说明怎样追sample order，不提供所有长构造方法的效果基线。

统一框架只保留可对齐字段，原始定义必须回链厂商笔记。

## packing、truncation 与admission bias

长样本能否进入batch取决于packer。best-fit packing减少padding，却可能优先接纳容易组合的长度；random truncation会系统性丢弃document tail；thematic packing可能制造非自然跨文档依赖。

最小日志：

```text
source_id, original_length, chunk_offsets
pack_id, segment_order, separator
target_length, padding_positions
attention_mask_type, loss_mask_positions
truncated/drop reason
```

按source/domain/length报告submitted、admitted、truncated、sampled与loss distributions，才能发现admission bias。

## curriculum

长度扩展通常与RoPE/attention、LR和mixture共变。推荐factorial fork：

| fork | length | data | 目的 |
|---|---|---|---|
| baseline | short | short | 参考 |
| position-only | long | short/concat | 架构位置适配 |
| data-only proxy | short | long chunks | source内容收益 |
| full | long | long structured | 交互效应 |

每次切换同时评估short/mid/long，不能只在最终checkpoint测最大长度。

## validation

### 不能只用needle

needle-in-a-haystack容易ceiling，并可能奖励字符串检索。MiniMax报告在早期很快饱和后换更难的intermediate-checkpoint tasks；Apple同时用NIAH、RULER和13类synthetic tasks。

最低suite：

- exact retrieval与多needle；
- natural document QA/summarization；
- cross-section inference与contradiction；
- repo/agent state tracking；
- distractor robustness；
- short-context regression；
- 按length×source×task per-domain结果。

评测集必须按完整book/repo/document group split，并与训练derivative一起去污染。

## 关键诊断

### effective used context

最大窗口不等于有效窗口。可画性能随依赖距离 $d$ 的曲线：

$$
Q(d)=\mathbb{E}[\text{score}\mid\text{required evidence distance}=d].
$$

同时改变答案位置、distractor密度和所需证据数，避免位置捷径。

### 长短能力保持

对每个domain $k$：

$$
\Delta_k=Q_{k,\text{after P3}}-Q_{k,\text{before P3}}.
$$

同时报告micro与macro；全局均值可能掩盖小域退化。short replay比例应通过Pareto曲线选择，而不是只最大化long score。

## 表面冲突与区分性检查

### 1M能力强 vs long tokens很少

- 首个差异：能力窗口与训练exposure不是同一对象。
- 检查：发布length histogram、packer日志、distance curve和long-source比例。

### natural long优于synthetic vs synthetic更可控

- 首个差异：目标任务、噪声与验证方式。
- 检查：固定tokens比较intact、random concat、thematic pack、synthetic QA，并做混合比例扫描。

### long训练导致短能力退化 vs replay可保持

- 首个差异：replay比例、LR、阶段时长和评测分布。
- 检查：checkpoint-level short/long Pareto与per-domain loss。

## 明确未知项与不可外推

- 不由context size推断long tokens或真实长文比例；
- 不把sequence positions当non-padding loss tokens；
- 不把40/60 attention treatment或46/54 category直接解释成length distribution；
- 不把1M needle通过率外推到1M natural reasoning；
- 不把某模型的长度bucket、RoPE或replay比例当通用最优值。

## 最小实验清单

1. 固定checkpoint/token/LR，比较四种construction；
2. 扫描short replay比例；
3. 每个checkpoint记录length×domain loss；
4. 按完整source group去污染；
5. 报告admission/truncation/padding/mask；
6. 同测retrieval、natural reasoning、agent与short regression。

## 掌握标准与自测

读者应能解释：

1. Nemotron的46/54与92/8为何不能相乘成“1M长文占比”；
2. Apple 100B@32K为何不等于100B natural long tokens；
3. MiniMax length buckets比单一1M标签多提供什么；
4. Kimi 40/60为何首先是attention/data treatment；
5. 如何用source-group split防止章节/repo泄漏；
6. 如何设计length-only/data-only/full forks。

推理题：某P3有20B nominal positions，padding率15%，其中prompt mask覆盖10%的non-padding positions。若其余均进入loss，loss tokens是多少？这个数字能否说明natural long比例？

## 来源与阅读路径

1. [Meta Llama](/topics/pretraining-data/meta-llama-data-practices/)：读800B六阶段、short recovery与Llama 4披露边界。
2. [MiniMax](/topics/pretraining-data/minimax-data-practices/)：读300B/32B/26B length curriculum、NIAH ceiling与短长交替post-training。
3. [Apple AFM](/topics/pretraining-data/apple-foundation-models-data-practices/)：读100B@32K、65K source taxonomy和多数原文短于窗口的边界。
4. [NVIDIA Nemotron](/topics/pretraining-data/nvidia-nemotron-data-practices/)：读33B、46/54与92/8的正交核算。
5. [GLM](/topics/pretraining-data/glm-data-practices/)、[Hunyuan](/topics/pretraining-data/tencent-hunyuan-hy-data-practices/)、[Kimi](/topics/pretraining-data/moonshot-kimi-data-practices/)：比较自然长文、repo/agent、synthetic dependency、packing和attention treatment。
6. [Validation design](/topics/pretraining-data/validation-design/)：用per-domain、clean split和checkpoint诊断组织评测。
