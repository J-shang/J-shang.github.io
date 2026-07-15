---
title: "Microsoft MAI 与 Phi 家族的数据实践"
description: "审计 model ladder、unique/sample/epoch、synthetic textbook、蒸馏与阶段边界。"
topic: "pretraining-data"
section: "international-cases"
slug: "microsoft-mai-phi-data-practices"
date: 2026-07-14
updated: 2026-07-15
cutoff: 2026-07-14
order: 124
readtime: 34
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/microsoft-mai-phi.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/67a4f4c4f8a4c5793a56d3050c61a7ca54971678/industry-data-practices/international/microsoft-mai-phi.md"
  revision: "67a4f4c4f8a4c5793a56d3050c61a7ca54971678"
  syncedAt: "2026-07-15"
  contentHash: "sha256:e264a4a25140839e0580587fd59ed73ae0e7fd64b5b3cf7a045957ecb394586d"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
<!-- maintenance: reasoning-path=`method-family/historical trace` -->
> 主要模型/资料：MAI-Thinking-1 / MAI-Base-1 与 Phi-1→Phi-4；MAI-1-preview 作为闭源历史边界。

## 这篇案例要回答什么

Microsoft 内部存在两条不能互相代填的研究线：

- MAI 是 Microsoft AI 的 frontier family。2025 年 MAI-1-preview 只披露 end-to-end MoE 与约 15K H100；2026 年 MAI-Thinking-1 报告则公开 MAI-Base-1 的 30T pretraining、3.55T mid-training、29.2T unique-token pool、source cutoff、mixture ladder 与 RL climb。
- Phi 是 compact/open-weight family，从 phi-1 的 7B-token coding textbooks，发展到 phi-3 的两阶段 data-optimal framing，再到 phi-4 的 10T、40% synthetic 与 250B long-context mid-training。

两条线恰好构成受控对照：MAI-Base-1 的 pretraining 明确不使用 LM-generated synthetic data，Phi-4 则让 synthetic 成为最大单一 source。这个差异不是“谁更好”的实验，因为模型规模、目标、数据、token budget 与训练年代都不同；它用于检验本仓库的 synthetic provenance、unique/sample token、model ladder 与 validation 定义能否容纳相反策略。

## 各代模型和 training stage

| 模型 | 阶段 | 已披露数据规模 | mixture / context | 披露 |
|---|---|---:|---|---|
| MAI-1-preview | `P0/A*` | token/source/mix `unknown`；约 15K H100 为计算资源，不是数据量 | first end-to-end in-house MoE；pre/post-trained | `D1` |
| MAI-Base-1 | `P0` | 30T sampled；29.2T dedup unique pool | 54.6% code、15.8% STEM、5.4% math、3.1% books/journals、4.7% PDF、14.9% web、1.6% other multilingual；16K | `D2` |
| MAI-Base-1 | `P1/P3` mid 1 | 3.4T | 55% code、35% STEM/math、10% background；64K；只重配 pretraining corpus | `D2` |
| MAI-Base-1 | `P1/P3` mid 2 | 150B | 同 mixture weight、longer packing；256K | `D2` |
| MAI-Thinking-1 | `A1/A2/A3` | prompt/rollout/loss totals `unknown` | reasoning、SWE、tool、IF、general；RL length 8K→128K；self-distillation/consolidation | `D2` |
| phi-1-base | `P0` | 6B filtered web +约1B GPT-3.5 synthetic textbooks/exercises | Python/code；base vs exercise finetune分开 | `D2/D3` |
| phi-1 | specialization | exercise data precise token count `unknown` | coding exercises；形成 phi-1 vs phi-1-base ablation | `D2/D3` |
| phi-1.5 / phi-2 | `P0` | exact sampled/loss tokens未统一披露于本轮主锚点 | phi-1 sources + NLP synthetic；phi-2 另做 knowledge transfer | `D1-D3` |
| phi-3-mini | `P0/P1` | 3.3T | Phase 1 filtered web；Phase 2 ultra-filtered web + synthetic reasoning | `D2/D3` |
| phi-3-small/medium | `P0/P1` | 各 4.8T | 7B/14B model-size ladder；同 family data | `D2/D3` |
| Phi-3.5-Vision | multimodal `P0/A1` | pre 500B mixed text/visual tokens；SFT 33B | image-text/interleaved/OCR/PDF/chart/table/text；image tokens不计 LM loss | `D2/D3` |
| phi-4 | `P0` | 约10T sampled | 15% web、15% web rewrite、40% synthetic、20% code、10% acquired；4K | `D2/D3` |
| phi-4 | `P3` | 250B | 30% newly curated long data +70% recall；4K→16K | `D2/D3` |
| phi-4 | `A1/A2` | SFT约8B；DPO sample/token totals未完整闭合 | SFT→pivotal-token DPO→judge-guided DPO | `D2/D3` |
| Phi-4-reasoning | `A1/A2` | 数量 `unknown` | teachable prompts + o3-mini demonstrations；reasoning-plus短RL | `D2/D3` |

MAI 的总 base exposure：

$$
T_{base}=30\text{T}+3.4\text{T}+0.15\text{T}=33.55\text{T}
$$

mid-training 没有引入新 source 或 synthetic，只从 pretraining corpus 过滤、重权重、重打包；因此 3.55T 是追加 sampled exposure，不是新增 3.55T unique corpus。

## 阅读这些结论前先确认的前提

| 项 | 本笔记的处理 |
|---|---|
| 家族边界 | MAI 与 Phi 独立记录；不因同属 Microsoft 而推断共享 corpus、tokenizer 或 data policy |
| token 单位 | MAI 表同时给 unique/training tokens；Phi 表给 unique、fraction、epochs；统一按原始分母保留 |
| synthetic | MAI pretraining 不用 LM-generated synthetic；其 RL/post-training包含 synthetic。Phi 的 web rewrites 也是 synthetic 子类 |
| cutoff | MAI 按 source family 给绝对日期；Phi 未给同等完整 cutoff |
| “data optimal” | Phi-3 报告明确是 aspirational term，不表示找到数学全局最优 |
| ladder transfer | MAI 记录 rank reversal；Phi-4 声称 7B↔14B 在足够大 mixture gap 上相关。二者适用域不同，不判为逻辑矛盾 |
| multimodal token | Phi-3.5-Vision 的 500B 含 visual/text elements，image token 不计 LM loss；不能当 500B text loss tokens |
| knowledge cutoff vs collection date | MAI 表是 source knowledge cutoff；不自动等于所有文件抓取/授权/处理完成日 |

## 厂商公开了哪些 data fields

| 字段 | MAI-Base-1 / Thinking-1 | Phi | 置信 |
|---|---|---|---|
| source | proprietary web crawl、licensed third-party、public GitHub、books/journals、PDF、news、multilingual/domain materials | filtered web、Stack/StackOverflow/contests、books/academic/forums、teacher-generated textbooks/QA/rewrite、multimodal open/synthetic | 类别 `verified`；清单 `open` |
| rights | public controls + commercial agreements/diligence；无 private customer/product data，除 explicit opt-in/agreement | acquired/open/web/synthetic；逐 source rights 未统一披露 | MAI high-level `verified`；逐项 `open` |
| cutoff | HTML 2025-09；PDF 2025-12；GitHub 2025-06；books/journals 2026-03 | `unknown` | MAI `verified` |
| token scale | 29.2T unique pool→30T sampled；+3.55T mid | phi-1 7B；phi-3 3.3/4.8T；phi-4 10T+250B | headline `verified`；loss tokens部分 `open` |
| parse/extract | source-specific parser、hand extractors、keep/remove-only LLM agent、raw wikitext；PDF/OCR | web/PDF/code filters；synthetic generation/rewrite；实现细节代际不均 | `verified/supported` |
| dedup | boilerplate/exact/MinHash 0.8/template skeleton/semantic Qwen3-Embed；global drop-order；20-gram eval threshold 80% | phi-1 decontam；phi-4 13/7-gram test filters；web/synthetic self filtering | MAI `verified`；Phi部分 `verified` |
| quality | metadata/heuristics/classifier/LLM/manual；bucket + ladder ablation | educational-value web、seed curation、plurality/rejection/test validation | 机制 `verified`；阈值/labels部分 `open` |
| mixture | explicit unique/sample/epoch table；hierarchical local/global search；rank reversal | two-phase web→synthetic；phi-4 explicit 15/15/40/20/10 | `verified` |
| synthetic | pretraining明确排除/移除 AI-generated；post/RL含 synthetic problems/environments | central：textbooks、solutions、rewrites、QA、code/OCR、teacher reasoning | `verified`；完整 provenance `open` |
| decontam | remove HF/mirrors；universal 20-gram fuzzy；private eval；RL exact/MinHash/vector | benchmark-specific 13/7-gram；Phi-3.5/Phi-4 SFT/DPO filters | 局部 `verified`；全局完整性 `open` |
| validation | 近40 NLL tasks、BPB、private eval、scaling ladder、EG、61 mixtures、scale-up | model ladder、phase ablations、unique-vs-repeat、HELMET、internal benchmark | `verified`；跨目标因果 `open` |
| artifacts | detailed report；MAI weights/data/order/logs未公开 | weights、reports、部分 code；synthetic corpus/manifest未全公开 | MAI `D2`；Phi `D2/D3` |

## MAI：source、rights 与 cutoff 的可审计边界

MAI-Base-1 的 30T 只使用 publicly available 与 licensed human-generated data，并明确尝试移除 AI-generated content；不使用 open-source training datasets，而是从原始 HTML/PDF、books/journals 与 public GitHub 自行处理。报告还声明：不使用 private customer data 或 Microsoft products/services data，除非用户 explicit opt-in 或 applicable agreement 覆盖，并尊重 opt-out。

这组声明的关系需精确：

```text
publicly accessible --does-not-imply--> unrestricted license
robots/meta controls --implements--> access/use controls
commercial agreement --supports--> licensed third-party data
human-generated-only pretraining --does-not-apply-to--> RL/post-training synthetic data
```

source cutoff 是少见的绝对日期表，但完整 provider、URL、document hash 因 privacy/legal/safety/competitive reasons 未公开。因此可以验证声明和时间边界，不能复现 corpus。

## MAI web extraction 与多层 dedup

HTML 根据 domain 采用 structured parser、BeautifulSoup 类定制 extractor、LLM/agent keep-or-remove processing，或直接使用 raw content。LLM extractor 被限制为只能保留/删除原文，不能添加 synthetic content；raw wikitext 虽约冗长 3×，但为保留 infobox 等结构被认为优于 stripped conversion。

dedup 分层为：

1. boilerplate line-occurrence removal；
2. byte/hash exact dedup；
3. MinHash LSH，similarity 0.8；
4. page skeleton/template fuzzy dedup；
5. Qwen3-Embedding-0.6B semantic clusters，每 cluster 限制代表数；
6. cross-dataset global drop-order。

global drop-order 是关键 implementation invariant：若两个 dataset 重叠，样本归属优先级高者。修改一个 dataset 会使相同内容在 source accounting 中“迁移”，即便全局 unique content 没变。因此 source ablation 必须固定 drop-order 和所有交叉集合，不能只 diff 一个目录。

## MAI 29.2T unique → 30T sampled：不把 mixture 百分比误当质量排名

| source family | unique T | training T | mix | avg epochs |
|---|---:|---:|---:|---:|
| Code | 7.4 | 16.4 | 54.6% | 2.22× |
| STEM | 2.2 | 4.7 | 15.8% | 2.17× |
| Math | 0.3 | 1.6 | 5.4% | 5.28× |
| Books/journals | 0.6 | 0.9 | 3.1% | 1.65× |
| PDFs | 2.7 | 1.4 | 4.7% | 0.53× |
| Web text | 8.1 | 4.5 | 14.9% | 0.55× |
| Multilingual other | 8.1 | 0.5 | 1.6% | 0.06× |
| total | 29.2 | 30.0 | 100% | 1.03× |

math 的 5.28× 是最强 repeat，multilingual other 的 0.06× 是强 downsample；这不等价于二者各自“质量”高低。mix 是面向 code/reasoning 产品目标的 constrained allocation，也受 unique supply、repeat saturation 与 taxonomy first-match rule 影响。

## MAI mixture search、NLL objective 与 rank reversal

### 预训练指标

近 40 个 NLL benchmarks 分 Code、STEM、Math、General、Multilingual，聚合目标：

$$
J=0.5L_{code}+0.175L_{STEM}+0.175L_{math}+0.1L_{general}+0.05L_{multi}
$$

权重显式编码目标偏好，不是自然规律。所有 model 使用同 tokenizer，NLL 可直接比较；对跨 tokenizer base model 报 bits-per-byte。private NLL datasets 来自内部项目、人机交互、vendor/domain experts 与去重 public sources，降低污染但不能被外部复核。

### scaling ladder 与 efficiency gain

ladder 在固定 active-parameter TPP 下训练不同 size；architecture ablation 常用 100–200 TPP，生产 main run 为推理效率而 overtrain 到 500–1,000 TPP。候选的 Efficiency Gain：

$$
EG=\frac{f^{-1}(L')}{C'},\qquad f(C)=AC^{-\alpha}+E
$$

含义是 baseline 要用多少倍 cost 才匹配 candidate loss；cost 可取 FLOPs 或 wall time。它是 relative scaling metric，不是 data quality score。

### 小模型 mixture 排名会反转

团队用 760M–4B active 的数千模型搜索 mixture，又在更大 scale 复核。一个明确反例中，stem-heavy mix 在小模型早期更好，但两个约 23B-active 模型训练约 20T 后，code-heavy mix 的 held-out STEM NLL 反超。检查发现 stem-heavy 中两个高质量 STEM sources 权重 11.8%，但 fuzzy duplicate 多、diversity 低；code-heavy 中只占 0.3%。

这是 `[来源事实 | verified]` 的 rank non-invariance 案例。它说明：

```text
small-proxy score --is-not-order-preserving-in-general--> target-horizon score
unique/diversity saturation --can-change--> mixture ranking over scale
```

后续采用约 10 个高层 category 的 local/global hierarchical search，并以约 2.8× global-mixing compute 做 scale-up validation。

## MAI mid-training：从同一 corpus 重配，而不是新增 synthetic

mid mixture 为 code 55%、STEM/math 35%、background 10%。新增处理包括：

- STEM/PDF：Bloom `Analyze` 及以上、technical correctness、reasoning depth、移除 extraction artifacts；
- code：repo-quality 三档下按 file extension filter，并同时使用 repo-level 与 file-level formatting；
- memorization-aware epoch cap：用两 checkpoint 之间 validation loss 改善中来自近确定 token（NLL<0.01）的比例做代理；
- 64K→256K 只 re-pack 为更长 sequences，不改变 mixture weights，以减少 distribution shift 和 truncation。

该 memorization proxy 是 empirical diagnostic：近确定 token 也可能是语言模板、容易事实或重复结构，不等于逐例记忆证明。应用时需配合 exact/fuzzy extraction 与 held-out canary。

## MAI RL 数据：synthetic 从 pretraining 的“零”变成明确组成

MAI-Thinking-1 从 mid-trained checkpoint 做 reasoning/agent/general climbs，output length 从 8K 按 2 倍扩到 128K。self-distillation 时会混入 mid-training data，防止遗忘 long-context behavior。报告称固定 token budget 下增加 prompt diversity 比增加每题 traces 更有价值；这是其训练域内的 empirical result。

SWE environment 从 102M public GitHub PR 起步，按 issue linkage、repo/build/test、specification clarity、test quality、leakage risk 与 feasibility 多层过滤；失败但可执行的环境可通过 problem rewrite/test augmentation 转成 synthetic tasks。reasoning 数据又做 SHA-256 exact、character n-gram MinHash 与 vector dedup/decontam。

所以“MAI 不用 synthetic”只适用于 pretraining。跨阶段正确表示为：

```text
P0/P1: human-generated source only
A1/A2/A3: organic prompts + synthetic problems/environments + human preference
```

## Phi-1→3：textbook quality 与 model ladder

phi-1-base 用 6B filtered “textbook-quality” web tokens 和约 1B GPT-3.5-generated textbooks/exercises；phi-1 再加 coding-exercise specialization。source 包括 The Stack v1.2 Python、StackOverflow、code contests 与 `gpt-3.5-turbo-0301` generations。其强结果证明该 recipe 在 350M/1.3B coding setting 有效，不证明 7B unique tokens 足以覆盖通用知识。

phi-1.5 加 NLP synthetic texts，并为安全研究排除 generic Common Crawl；phi-2 将 1.3B 的知识 transfer 到 2.7B，同时混合 synthetic common-sense/general knowledge 与 educational-value web。公开说明没有形成像 MAI 表 5 那样的统一 unique/sample/loss ledger。

phi-3-mini（3.8B）3.3T；small（7B）与 medium（14B）各 4.8T。Phase 1 以 filtered web 教 general knowledge/language，Phase 2 用更严格 web subset + synthetic reasoning/niche skills。报告称 14B 相对 7B 的部分 benchmark gain 小于 7B 相对 3.8B，提示同一 mixture 对更大模型可能不再 “data optimal”。报告脚注明确：`data optimal` 是 aspirational，不声称找到 provable optimum。

## Phi-4：10T synthetic-heavy mixture 与 unique/sample 对账

Phi-4 生成约 50 类 synthetic datasets，总计约 400B unweighted tokens；训练最终表：

| source | training fraction | unique tokens | epochs | implied sampled |
|---|---:|---:|---:|---:|
| filtered web | 15% | 1.3T | 1.2× | 1.5T |
| web rewrites | 15% | 290B | 5.2× | 1.5T |
| synthetic | 40% | 290B | 13.8× | 4.0T |
| code（raw+synthetic） | 20% | 820B | 2.4× | 2.0T |
| acquired（academic/books等） | 10% | 580B | 1.7× | 1.0T |

`web rewrites` 是 synthetic 子类，因此若问“所有 synthetic-derived training share”，不能只报 40%；code 也含 synthetic/raw 未拆比例。官方 40% 是表中 `Synthetic` bucket，不是全体 model-generated content 的 exhaustive share。

### generation 与 validation

synthetic 由 curated seeds 出发，进行 rewrite/augment、multi-step prompting、self-revision、teacher answers；math/science/coding使用 plurality、rejection、test execution 或 LLM checks。报告强调 seed 中的小错会在衍生数据中放大，因而先严格 curate organic web/Q&A。

13B synthetic-only ablation 重复每 source >20 次，大多数 reasoning/code benchmark继续提高，但 TriviaQA 大幅退化且 hallucination 增加。4 vs 12 epoch experiment 也显示在相同 token horizon，更多 synthetic repetition 优于更多 fresh web，限于测试 mixture/model/benchmark。

这两项不是矛盾：synthetic repetition 对 reasoning loss 有益，同时 organic web 对 long-tail knowledge/calibration 有不可替代作用。

### 1T proxy 与 transfer 边界

mixture search 使用 1T horizon、7B proxy，并观察在 mixture gap 足够大时 7B/14B rank correlation。最终仍选择较均衡的 web/acquired share，因为 pure synthetic-heavy 在 TQA/knowledge 较弱，且 post-training 会缩小部分 pretraining difference。

这与 MAI rank reversal 的统一解释是：rank transfer 取决于 model sizes、horizon、mixture distance、unique/repeat saturation 和 objective；Phi 的局部相关性是 special case，MAI 的反例否定 universal rank-invariance。

## Phi-4 long context 与 Phi-3.5-Vision

Phi-4 mid-training 250B，把 30% 分给新筛的 naturally long/synthetic >4K data，70% recall pretraining；自然长 context 在报告实验中优于人工 pad/concat。context 4K→16K 同时改变 RoPE，因此 data-only effect仍需 matched control。

Phi-3.5-Vision pretraining 500B mixed visual/text tokens：interleaved docs、image-text pairs、OCR-from-PDF synthetic、chart/table、text-only。LM objective只在 text tokens计算，image token loss masked：

$$
T_{sampled}=T_{text}+T_{vision},\qquad T_{LM-loss}\le T_{text}<T_{sampled}
$$

SFT 约33B，并联合 text/multimodal tasks 以降低 language regression。没有 visual tokenizer equivalence、padding/mask ledger 时，500B 不能与 phi-4 10T直接做 data-efficiency ratio。

## 如何验证这些结论，并检查 contamination

| 公开结论 | 首个混淆 | 有辨识力的检查 |
|---|---|---|
| MAI dedup改善scaling/reasoning | novelty、source attribution、mix同时变化 | fixed mix/exposure，分exact/template/semantic；测unique saturation、canary、per-domain NLL |
| MAI small-mixture proxy可筛选 | rank reversal随horizon/scale发生 | 多scale×多horizon交叉，报告ranking uncertainty/crossing point |
| memorization-aware epoch cap有效 | NLL<0.01不专属于memorization | 与exact exposure、canary、paraphrase generalization联合标注 |
| phi synthetic多epoch仍提升 | unique prompts/teacher diversity/knowledge缺口不明 | fixed unique/sample tokens，重复same response vs diverse generations vs fresh web |
| phi synthetic优于web reasoning | benchmark与teacher偏好/污染可能同源 | source-group split、teacher-independent eval、knowledge/calibration counter-metrics |
| natural long优于padded long | document quality/length/task不同 | length/quality matched natural vs concat vs synthetic dependency |
| Phi vision 500B高效 | visual/text/loss token口径不明 | 按modality报告sample、tokens、loss mask与compute；matched-text replay |

## 看似矛盾的说法怎样区分

1. **MAI no synthetic vs Phi synthetic-heavy**：不同家族/目标；MAI 限定 pretraining，post-training明确含 synthetic。
2. **MAI 29.2T unique vs 30T training**：source-level oversample/downsample抵消后总 epochs 1.03×，不表示每个 token约一次。
3. **3.55T mid vs no new data**：是旧 corpus 的追加 exposure与新 packing，不是 unique data 增长。
4. **Phi synthetic 40% vs synthetic-derived share**：web rewrites 15% 也是 synthetic；code bucket又混合 raw/synthetic。
5. **Phi rank correlation vs MAI rank reversal**：Phi 是给定 7B/14B、1T/mixture-gap 的局部观察；MAI 提供 23B/20T crossing 反例。
6. **15K H100 vs 8,192 GB200**：一个是 MAI-1-preview 的高层 compute claim，一个是 MAI-Base-1 phase table；均不是 token count。

## 哪些经验可以借鉴，哪些不能直接照搬

### 可迁移

- 同时发布 source unique tokens、sampled training tokens、mix% 与 avg epochs，能暴露 repeat saturation。
- mixture optimizer 必须显式写 objective weights、proxy sizes/horizons、rank uncertainty 与 scale-up gate。
- cross-dataset dedup 必须版本化 global drop-order，否则 source attribution会漂移。
- synthetic provenance 至少含 seed、generator/version、prompt、rewrite、validator、repeat count 与 source group split。
- pretraining “不含 synthetic”必须限定阶段，不能延伸到 SFT/RL/environment generation。
- multimodal总 token必须补 modality 与 loss-mask ledger。

### 不可外推

- 不把 MAI 54.6% code、5.28× math 或 mid 55/35/10 当通用配比。
- 不把 Phi-4 40% bucket、13.8× synthetic 或 7B proxy 视为跨规模最优。
- 不由 robots/terms compliance 推断所有版权与用途争议已经解决。
- 不把 teacher-generated synthetic 的 benchmark gain自动解释为超越 teacher 的通用知识；需 teacher-independent domains。
- 不用 MAI/Phi benchmark差异比较两种数据哲学，除非固定模型、compute、tokenizer与post-training。

## 目前仍不知道什么

- MAI 公开 corpus/manifest、provider list、per-document rights、order、seed、packing/loss tokens与训练日志；
- MAI mixture ladder 的所有 candidate configs、NLL private eval复现资产与最终 30T sample order；
- MAI RL prompt/environment/rollout/tool-call/loss-token完整统计；
- Phi-1.5/2 的统一 unique/sample/loss ledger与 cutoff；
- Phi synthetic 50 families 的完整 seed→generation manifest、teacher revisions、accept/reject counts；
- Phi-4 code bucket 内 raw/synthetic proportion、完整 source rights/cutoff；
- Phi-3.5-Vision 500B 的 text/visual/padding/masked token breakdown。

## 读完后应该能回答的问题

应能：

1. 从 MAI table 解释 unique、sampled、mix、epochs 四列；
2. 重建 MAI 30T→3.4T→150B 与 Phi-4 10T→250B；
3. 说明 MAI/Phi synthetic 策略为何不是矛盾实验；
4. 用 rank reversal 反驳无条件 proxy rank invariance；
5. 为 synthetic repetition、dedup/drop-order 和 multimodal mask各设计审计。

自测：

1. MAI multilingual other 有 8.1T unique、只采0.5T；为什么不能说83亿/万亿未采token“质量低”？
2. Phi-4 表中明确可归入 synthetic-derived 的最小 training share是多少？为什么不是完整上界？
3. 若 semantic dedup 修改导致样本从低优先 dataset移到高优先 dataset，如何保持ablation可解释？
4. MAI 的 NLL objective 若把 multilingual权重从0.05升到0.2，mixture结论为何可能整体变化？
5. 怎样用同一实验同时检查 synthetic repeat收益与TriviaQA知识退化？

## 来源与建议阅读位置

- [MAI-Thinking-1 technical report](https://microsoft.ai/pdf/mai-thinking-1.pdf)：MAI 主锚点；读 §2.2–2.6、Appendix B/C 与 §3 data/RL。
- [Microsoft AI: seven new MAI models](https://microsoft.ai/news/building-a-hillclimbing-machine-launching-seven-new-mai-models/)：固定 2026 MAI family 与产品边界；不用于补 token 配方。
- [Microsoft AI: MAI-1-preview announcement](https://microsoft.ai/news/two-new-in-house-models/)：确认首个 end-to-end in-house model 与约15K H100；其数据字段仍为 `unknown`。
- [Textbooks Are All You Need](https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need/)：读 phi-1 的 6B+1B、350M/1.3B ladder 与 source。
- [Textbooks Are All You Need II](https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need-ii-phi-1-5-technical-report/)：读 phi-1.5 NLP synthetic 和排除 generic web-crawl 的边界。
- [Phi-2: surprising power of small language models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)：读 educational filter、synthetic categories 与 knowledge transfer；定量披露有限。
- [Phi-3 technical report](https://arxiv.org/abs/2404.14219)：读 3.3T/4.8T ladder、two-phase data-optimal framing 与 Phi-3.5-Vision 500B/33B。
- [Phi-4 technical report](https://arxiv.org/abs/2412.08905)：读 synthetic generation、10T mixture/epochs、1T proxy、250B long-context mid-training 与 decontamination。
- [Phi-4-reasoning technical report](https://arxiv.org/abs/2504.21318)：读 teachable prompt selection、o3-mini reasoning demonstrations 与短RL增量。

## 这篇案例与主线知识的关系

```text
source-governance --bounds--> rights + cutoff claims
dedup-drop-order --changes--> source attribution
mixture-objective --selects--> unique/sample/repeat allocation
scaling-ladder --tests-but-does-not-guarantee--> target ranking
synthetic-seed --generates--> rewrite/solution --validated-by--> verifier
multimodal-context --contains--> visual tokens --masked-from--> LM loss
```
