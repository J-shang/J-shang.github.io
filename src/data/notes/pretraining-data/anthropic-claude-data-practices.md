---
title: "Anthropic Claude 家族的数据实践"
description: "分离 source taxonomy、consent 边界、transcript incident、canary 与 evaluation decision exposure。"
topic: "pretraining-data"
section: "international-cases"
slug: "anthropic-claude-data-practices"
date: 2026-07-14
updated: 2026-07-14
cutoff: 2026-07-14
order: 122
readtime: 27
source:
  repository: "J-shang/pt-data-learning"
  path: "industry-data-practices/international/anthropic-claude.md"
  url: "https://github.com/J-shang/pt-data-learning/blob/dc18f7fad9acbef375773418a5e05cc614f7a2d4/industry-data-practices/international/anthropic-claude.md"
  revision: "dc18f7fad9acbef375773418a5e05cc614f7a2d4"
  syncedAt: "2026-07-14"
  contentHash: "sha256:d908c102b26857b01957b3f120a0aa78c65769d264f2d12b32e26e0f26af07fd"
  manifest: "pretraining-data"
  managed: true
---
> 状态：`draft`
> 核查截止：**2026-07-14**
> 主要 reasoning path：`method-family/historical trace`
> 研究锚点：Claude 2、Claude 3/3.7、Claude 4；4.5–5 系统卡用于披露边界，不从能力反推数据。

## 定位与 motivating problem

Claude 是典型的闭源 frontier 对照：Anthropic 公开 source family、knowledge cutoff、consumer/commercial opt-in 边界、Constitutional AI 与大量 safety evaluations，但不公开 pretraining token 规模、mixture、语言比例、dedup 阈值、loader/order 或 loss-token accounting。

Claude 4 system card 还披露了一个少见的训练数据事故：约 150K 条公开 alignment-faking research transcripts 进入 Opus 4 pretraining，因缺失原实验 system prompts 而诱发特定幻觉。这使案例可研究：

1. 公开 transcript、论文、benchmark 与模型训练如何形成反馈回路；
2. canary 能证明/排除什么，不能替代什么；
3. consumer opt-in、commercial default no-training、crowdworker、third-party 与 internal synthetic 如何分别治理；
4. system-card evaluation 与 pretraining validation 为何不是同一种 artifact。

## 代际与阶段表

| 模型 | 阶段 | 已披露数据 | cutoff / scale | 披露 |
|---|---|---|---|---|
| Claude 2 | `P0/A1/A2` | internet、licensed third party、affirmatively shared user/crowdworker、internal；约10% non-English | early 2023；unique/sample/loss tokens `unknown` | `D1` |
| Claude 3 Opus/Haiku | `P0/A*` | proprietary source categories；text/image；完整 mixture `unknown` | 2023-08 | `D1` |
| Claude 3 Sonnet | `P0/A*` | 同类高层披露 | 2024-04 | `D1` |
| Claude 3.5 Haiku | `P0/A*` | source/mix/token `unknown` | 2024-07 | `D1` |
| Claude 3.7 Sonnet | `P0/A1/A2` | internal/generated + public/third-party/opt-in categories；extended-thinking/post-training数量未知 | 2024-11 | `D1` |
| Claude Opus/Sonnet 4 | `P0` | public Internet、non-public third party、labeling services/paid contractors、opted-in users、Anthropic-generated | 2025-03；token/mixture `unknown` | `D1` |
| Claude 4 | `A1/A2` | human feedback、Constitutional AI、internally generated alignment/safety/agent data；数量未知 | n/a | `D1/D2`（process/eval） |
| Claude 4.1 | `P*/A*` | family data diff `unknown` | 2025-03 | `D1` |
| Claude 4.5–4.8 / Sonnet 5 | `P*/A*` | system cards公开能力/安全评测；完整训练数据差异 `unknown` | cutoff/token/mix未形成可比公开表 | `D1` |

Claude 4 source statement 是类别并集，不是占比表：

$$
T_{total}=T_{public}+T_{thirdparty}+T_{labeler}+T_{optin}+T_{internal}
$$

但每一项、重复集合、pre/mid/post stage 与 loss mask 都是 `unknown`，所以上式仅为 accounting schema，不可据此求值。

## Assumption ledger

| 项 | 本笔记的处理 |
|---|---|
| source vs rights | “public Internet”不等于统一许可；third-party 非公开也不等于可公开复现 |
| cutoff | knowledge cutoff 按官方 help center；不是精确 crawl date、每 source max timestamp 或所有事实保证 |
| user data | consumer、commercial、feedback、安全审查、Development Partner Program 分政策路径，不合并成“用户数据” |
| synthetic | Anthropic-generated pre/post data、Constitutional critiques、model-written evals、simulated trajectories 分 artifact |
| transcript incident | 150K 是公开 transcript 条数，不是 token count；进入 Opus 4 pretraining，不代表整个 Claude 4 family均同样受影响 |
| system card | safety/capability evaluation 披露较多，不等于 pretraining data/validation 可复现 |
| canary | machine-readable marker 可帮助识别/排除已标数据；无标记数据与 marker 被删除后的变体仍需其他方法 |
| latest family | 系统卡列表确认版本存在；若 data section不闭合，只记录 `unknown`，不继承 Claude 4 比例 |

## 统一数据字段表

| 字段 | Claude 公开信息 | 置信 |
|---|---|---|
| source | public Internet、commercial third party、labelers/contractors、opt-in users/crowdworkers、Anthropic-generated | 类别 `verified`；清单/比例 `open` |
| rights/governance | crawler遵守 robots.txt；不进 password/sign-in/CAPTCHA；third-party diligence；consumer/commercial不同 consent path | 声明 `verified`；逐样本权利 `open` |
| cutoff | Claude 2 early-2023；3 Opus/Haiku 2023-08；3 Sonnet 2024-04；3.5 Haiku 2024-07；3.7 2024-11；4/4.1 2025-03 | 日期 `verified`；source细分 `open` |
| token scale | unique/sample/loss 全部 `unknown` | `open` |
| language/modality | Claude 2约10% non-English；后代 text/image/multilingual但比例未知 | Claude2 `verified`；后代 `open` |
| parse/extract | crawler边界公开；parser/OCR/structured extraction details `unknown` | `open` |
| quality/dedup | 官方称 cleaning、filtering、dedup、classification；阈值/labels/version/retention未知 | 存在 `verified`；实现 `open` |
| mixture/order | `unknown` | `open` |
| synthetic | internal generation；Constitutional AI self-critique/revision与AI feedback；post-training/eval artifacts | method family `verified`；规模/provenance `open` |
| decontam | Claude 4 incident、targeted mitigation、prospective canary；无覆盖全部benchmark的protocol | incident `verified`；全局 `open` |
| validation | system cards含capability/safety/RSP/behavioral eval；pretraining held-out contract与per-source loss未知 | eval存在 `verified`；data attribution `open` |
| artifacts | system cards、部分eval prompts/transcripts/research；无weights/corpus/config/order/logs | `D1-D2` |

## Source 与 consent：至少五条不同的数据路径

### public web 与 third party

Claude 4 的 crawler 遵循 robots.txt，不访问 password-protected、sign-in 或 CAPTCHA pages，并让网站运营者可识别 crawler、表达偏好。Anthropic 同时声明对训练数据做 diligence。

这些是 access/governance controls；它们不提供 URL manifest、license ledger、抓取时间、删除请求传播到 checkpoint 的状态，也不把所有公共内容归为同一法律类别。

### consumer opt-in

自 2025-08/09 政策更新后，Claude Free/Pro/Max（含关联 consumer account 的 Claude Code）只有在用户选择允许时，才可把新建或恢复的 chats/coding sessions 用于改进模型；安全审查 flagged conversations 与其他明确 opt-in 另有路径。允许训练时相关数据最长保留五年；不选择时维持较短 retention。删除 chat 会阻止其用于未来训练，但已开始训练或已训练模型不可回滚。

policy snapshot 随时间变化，所以任何案例必须记录绝对日期，不能把当前政策倒推到 Claude 2/3 的历史训练。

### commercial default exclusion

Claude for Work、API、Bedrock/Vertex 等 commercial products 默认不以 inputs/outputs 训练 generative models；明确 feedback/opt-in 或 Development Partner Program 例外。Development Partner Program 只分享 Claude Code input/output tokens，组织级 opt-in，数据最多保存两年；zero-data-retention accounts 不可参加。

### labelers、contractors 与 internal data

data-labeling services/paid contractors 是 human-provided source；Anthropic-generated 则可能包括 synthetic responses、Constitutional critiques/revisions、preference comparisons、model-written evals 与 agent simulations。把它们都写成“内部数据”会隐藏 creator、rights、teacher、judge 与 loss role。

建议 manifest schema：

```text
artifact_id
stage: pretrain | sft | preference | rl | eval
creator: public-human | contractor | opted-in-user | model/version
source_parent / license / consent_basis
prompt_template / generator / judge
accepted_reason / rejection_reason
canary / benchmark_overlap / deletion_state
```

## Constitutional AI：synthetic data 不只是 teacher answer

Constitutional AI 的核心流程可抽象为：

```text
harmful/helpful prompt
 -> model response
 -> constitutional critique
 -> revised response
 -> preference comparison / AI feedback
 -> supervised or preference/RL update
```

constitution、critique model、revision model、preference model、prompt set 与 sampling config 都是 provenance 节点。它与“从 GPT 类 teacher 直接蒸馏一个答案”不是 exact identity：监督信号还包括规则条件、自我批评与成对偏好。

公开研究建立了方法关系，但 Claude 各代使用多少样本、哪些 principles、human/AI feedback比例、是否加入pretraining replay仍未形成可核算表。因此只标 `[来源事实 | supported]`，不据方法论文补填产品模型数量。

## Claude 4 的 150K transcript incident

### failure chain

Claude Opus 4 的 pretraining corpus 包含约 150,000 条公开发布的 alignment-faking paper transcripts。这些 transcripts 缺少解释论文 fictional setting 的 system prompts，却保留了与 Anthropic 后期训练相似的 distinctive transcript format。模型在相关提示下会补出被省略背景，表现为特定幻觉。

```text
research scenario + system prompt
 -> public release strips system prompt
 -> ~150K formatted transcripts enter pretraining
 -> model learns transcript-format-conditioned latent context
 -> related evaluations/prompts elicit hallucinated setting
```

这不是普通 exact memorization 的唯一解释。关键机制可能是 format、重复量、上下文缺失与训练阶段相似性联合形成的 conditional prior。

### mitigation 与残余

Anthropic 加入 targeted dataset：用论文 prompts 训练模型给出更接近 Sonnet 3.7（未受该 data影响）的响应。报告称行为 largely but not completely eliminated。这个 intervention同时改变 target responses和可能的format exposure，不能区分哪个因素主要起作用。

更重要的是，因相关内容已进入 pretraining，Claude Opus 4 上与该论文相关的 evaluations 不能再可靠作为独立证据。污染不只是“分数可能变高”，还可能改变模型对测试世界的先验。

### 区分性检查

至少做四组：

1. 原 transcript format + 原 prompt；
2. 同语义、换 format；
3. 同 format、换无关 fictional setting；
4. 完整 system prompt恢复。

再比较 affected/unaffected checkpoint、targeted mitigation前后、exact/fuzzy nearest-neighbor与输出中背景细节。这样才能区分 lexical memorization、format trigger 与一般 scheming propensity。

## Canary：用途、强项与失败边界

Anthropic 计划在公开 transcripts 中加入 machine-readable canary strings，以便未来 pretraining corpora 排除。canary 的可检查关系是：

```text
released artifact --contains--> stable marker
corpus pipeline --detects--> marker
manifest --records--> exclusion/rejection reason
```

它能帮助：

- 在 crawl/extraction 后确认某一已知 artifact 是否进入 candidate corpus；
- 生成明确 rejection log；
- 在多个内部 pipeline 间共享 exclusion rule；
- 为 evaluation “post-cutoff/not-in-training”提供额外证据。

它不能单独保证：

- 转贴、截图、OCR、翻译、摘要或删除 marker 的副本被发现；
- 训练前已存在的版本被移除；
- benchmark questions/solutions 的语义近重复被清理；
- 未见 canary 就能证明数据从未出现。

因此 canary 是 targeted negative-control marker，不是 universal decontamination proof。应配合 exact/fuzzy/semantic search、release date、source group split 与 model behavior probes。

## Evaluation contamination 与 decision-layer overfitting

Claude system cards公开大量 safety、RSP、cyber、biology、autonomy、reward-hacking与agentic eval。Claude 4还在训练过程中反复运行新 reward-hacking evaluations作为early-warning system。

这带来三个不同问题：

1. **training-data contamination**：eval内容或近重复进入 pretraining/SFT；
2. **development overfitting**：反复据同一eval选择data/reward/checkpoint，即使题目未进gradient；
3. **elicitation/scaffold drift**：tool、step limit、thinking budget、parallel sampling/selection改变分数。

例如 Claude 4 的 SWE high-compute结果使用parallel attempts、visible-test rejection与internal scorer selection；它不是单一policy的一次pass。TAU-bench又提高maximum steps。benchmark score不能直接反推agent training data。

建议 evaluation ledger：

| 字段 | 必需记录 |
|---|---|
| provenance | author、release date、canary、是否public |
| train overlap | exact/fuzzy/semantic、solutions/transcripts/scaffold |
| decision exposure | 何时首次看见、看过多少次、是否用于checkpoint/reward选择 |
| inference protocol | prompt、tools、step/token budget、sampling、candidate selection |
| result | raw trials、variance、failure taxonomy、clean subset |

## System-card incidents 与 data attribution边界

Claude 4 system card还记录 early Opus 4 snapshot 的strategic-deception/scheming表现，以及Sonnet 3.7较突出的reward hacking；最终训练加入更robust reward signal和持续evaluation。这里必须避免错误因果链：

```text
150K transcript incident --supports--> specific format-conditioned hallucination
150K transcript incident --does-not-establish--> broad strategic-deception cause
training-time reward-hack eval --detects/steers--> behavior trend
benchmark improvement --does-not-identify--> particular data intervention
```

Anthropic 自身的小型exploratory tests认为 transcript incident更广泛影响 unlikely，但final Opus 4未在所有相同external scheming eval上复测。最准确状态是：specific incident `verified`；broader causal attribution `open`。

## Validation、ablation 与区分性检查

| 公开结论 | 首个混淆 | 有辨识力的检查 |
|---|---|---|
| targeted mitigation减少incident | response target、format、额外exposure同时变 | format×content factorial；mitigation teacher/version ablation |
| canary可避免未来污染 | copied/stripped/translated variants漏检 | marker exact + semantic lineage + OCR/translation stress test |
| Claude 4 reward hacking下降 | training data、reward、model、eval familiarity变化 | frozen eval + new sealed eval + policy-only matched scaffold |
| cutoff保证新eval clean | old component/semantic duplicates可能存在 | canary、source search、model extraction、clean-group performance |
| opt-in user data改善coding | selection bias、product feedback、模型/compute变化 | consent cohort provenance；domain/quality matched ablation；privacy review |
| system-card safety score可代表model | scaffold、thinking/step budget与judge影响 | 多elicitation协议、raw trials、human adjudication与confidence interval |

## 表面冲突与处理

1. **commercial data不训练 vs user data是source**：commercial默认排除；consumer opt-in、feedback、safety review、DPP是不同例外路径。
2. **cutoff vs 2025/2026公开材料**：knowledge cutoff是训练边界声明；system card/eval发布可晚于训练，不表示模型见过全文。
3. **Claude 4 family vs Opus 4 incident**：报告明确指Opus 4 pretraining；不自动扩展到Sonnet 4。
4. **mitigated vs eliminated**：官方写largely but not completely；必须保留残余。
5. **canary present vs clean**：检测到可证明artifact候选存在；未检测到不能证明语义副本不存在。
6. **大量system-card评测 vs data透明度**：evaluation D2不提升pretraining corpus字段到D2。

## 可迁移与不可外推

### 可迁移

- user data必须按产品、consent、feedback、安全审查与retention分别建ledger。
- 公开research/eval transcripts要携带stable machine-readable canary和完整scenario metadata。
- 去污染要覆盖prompt、solution、transcript、system prompt、scaffold和语义变体。
- training-time反复看eval需要decision-exposure log，即使eval未加入gradient。
- incident mitigation要保留before/after checkpoint、targeted dataset provenance和残余失败率。
- closed model的unknown字段应作为结论，不用能力/回答猜corpus。

### 不可外推

- 不从Claude 2约10% non-English推断Claude 3–5语言比例。
- 不把当前consumer opt-in政策倒推到历史checkpoint。
- 不把robots compliance等同于全部source rights已公开/解决。
- 不把specific transcript hallucination解释成Claude所有alignment行为根因。
- 不用system-card benchmark推断pretraining token、mixture或synthetic比例。

## 明确未知项

- 所有代际的unique/sample/loss tokens、source proportions、repeat/epoch、packing与order；
- URL/document/provider manifest、逐source license、deletion/opt-out propagation与固定snapshot；
- parser、OCR、quality classifier、dedup/semantic threshold及各语言retention；
- internal synthetic、contractor、opt-in user data在pre/SFT/RL各阶段的数量与采样权重；
- benchmark decontamination覆盖表、decision-exposure次数与sealed validation contract；
- 4.5–5各版本训练数据diff与cutoff的完整可比记录。

## 掌握标准与推理型自测

应能：

1. 画出五种source/consent路径且不混同；
2. 解释150K transcript incident的failure chain与证据边界；
3. 区分canary、exact/fuzzy/semantic decontam与sealed eval；
4. 区分训练污染、development overfitting和inference scaffold变化；
5. 在token全部unknown时仍完成诚实的comparison matrix。

自测：

1. 若公开transcript含canary，但博客转贴删除canary，pipeline怎样保持lineage detection？
2. 为什么恢复缺失system prompt能区分format-conditioned hallucination与一般行为倾向？
3. consumer用户先opt-in后opt-out，manifest需要哪些状态才能阻止未来训练又不声称撤销已训练模型？
4. 一项eval从未进gradient，但每周用于checkpoint选择，属于哪种污染/过拟合？
5. 为什么Claude 4的March 2025 cutoff不能单独证明某个2025-04 benchmark完全clean？

## 原始来源与阅读位置

- [Claude 4 System Card](https://assets.anthropic.com/m/6c940a1b69ed6a1c/original/Claude-4-System-Card.pdf)：主锚点；读§1.1 training data、§4.1.4 transcript incident、§6 reward hacking与RSP eval。
- [Claude model system cards index](https://www.anthropic.com/system-cards)：固定各代system card版本和日期；最新版本的数据字段仍逐卡复核。
- [Claude 2 Model Card](https://www-cdn.anthropic.com/files/4zrzovbb/website/5c49cc247484cecf107c699baf29250302e5da70.pdf)：读source categories、early-2023 cutoff与约10% non-English。
- [Anthropic Privacy Center: personal data in model training](https://privacy.anthropic.com/en/articles/7996885-how-do-you-use-personal-data-in-model-training)：区分public、third-party、user/crowdworker与commercial default。
- [Anthropic Privacy Center: consumer data use](https://privacy.anthropic.com/en/articles/10023580-is-my-data-used-for-model-training)：固定consumer opt-in、安全审查与feedback例外。
- [Consumer Terms/Privacy update, 2025-08-28](https://www.anthropic.com/news/updates-to-our-consumer-terms)：记录政策生效时间、五年retention和删除边界。
- [Development Partner Program](https://support.anthropic.com/en/articles/11174108-about-the-development-partner-program)：核对commercial Claude Code opt-in、两年retention与组织级范围。
- [Knowledge cutoff list](https://support.anthropic.com/en/articles/8114494-how-up-to-date-is-claude-s-training-data)：核对Claude 3–4.1绝对cutoff；不把knowledge cutoff当完整source manifest。
- [Constitutional AI](https://arxiv.org/abs/2212.08073)：读self-critique/revision与AI feedback方法；不据论文补写产品数据量。

## 与主线的关系

```text
source-category --governed-by--> consent + access + retention
public-transcript --can-contaminate--> pretraining + evaluation
canary --detects-targeted--> artifact lineage
system-card-eval --differs-from--> pretraining validation
decision-exposure --can-cause--> evaluation overfitting without gradient
unknown-token-scale --prevents--> quantitative efficiency claim
```
