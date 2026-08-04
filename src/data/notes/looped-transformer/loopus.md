---
title: "LoopUS：把标准 Pretrained LLM 重组为 Latent Refinement Model"
description: "研究 encoder–loop–decoder 重组、selective gate、随机深度监督与 confidence exit，同时保留能力退化的负面证据。"
topic: "looped-transformer"
section: "llm-retrofit"
slug: "loopus"
date: 2026-08-02
updated: 2026-08-04
cutoff: 2026-08-02
order: 61
source:
  repository: "J-shang/looped-transformer"
  path: "papers/18-loopus.md"
  url: "https://github.com/J-shang/looped-transformer/blob/eb43191df7da7f8d1b936fa6485ea21f7c8f430a/papers/18-loopus.md"
  revision: "eb43191df7da7f8d1b936fa6485ea21f7c8f430a"
  syncedAt: "2026-08-04"
  contentHash: "sha256:9ac28d2ec9cca3a5c73cb99b3d7fddee2d5c6c736d0e334d5ff0807b48ae2bba"
  manifest: "looped-transformer"
  managed: true
---
> 论文：*LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models*<br>
> 精确版本：[arXiv:2605.11011v1](https://arxiv.org/abs/2605.11011v1)，2026-05-10<br>
> 作者与机构：Taekhyun Park、Yongjae Lee、Dohee Kim、Hyerim Bae；Pusan National University、Changwon National University<br>
> 官方 artifacts：[project](https://thrillcrazyer.github.io/LoopUS) · [code](https://github.com/Thrillcrazyer/LoopUS) · [models](https://huggingface.co/collections/Thrillcrazyer/loopus)<br>
> 官方代码核对截止：2026-08-03<br>
> 证据标签：**B**。多种 pretrained model、code 与 weights 可用；论文仍是极新的 preprint。

## 1. 30 秒结论

LoopUS 不删除大段层并重训，而是根据层间表示变化，把 pretrained LLM 分成 encoder $E$、可重复 middle block $M$ 和 decoder $D$。在 $M$ 每轮输出上加入 selective gate，训练时从 20 个可能 steps 中随机挑 5 个做 deep supervision，推理时由 confidence head early exit。

它在 Qwen3 1.7B/4B/8B、Phi-4 14B 等模型上用 3B FineWeb-Edu tokens 完成结构适配，平均 reasoning-oriented benchmark 提升约 1.6–2.2 points。与此同时，部分 MMLU、HellaSwag 会下降；所以它证明的是“较低 token budget 可形成 latent loop”，不是“预训练能力完全无损”。

## 2. 学习目标

1. 理解 encoder–loop–decoder split 的依据；
2. 推导 selective gate 为什么能阻尼 state update；
3. 理解 random deep supervision 如何节省 memory；
4. 区分 LM loss、monotonicity loss 和 confidence loss；
5. 用 retained-capability 表而非单一平均分判断 retrofit 是否成功。

## 3. 符号表

| 符号 | 含义 |
|---|---|
| $E,M,D$ | encoder、recurrent middle block、decoder |
| $h_b$ | 第 $b$ 次 loop 前的 hidden state |
| $\delta_b=M(h_b)-h_b$ | 当前 block 提议的 state update |
| $\Delta_b$ | input-dependent positive step scale |
| $A=-\exp(a_{\log})\in\mathbb R_{<0}^d$ | 由可训练向量 $a_{\log}$ 得到的 channel-wise negative decay vector |
| $\alpha_b$ | 接受 middle block 新提议的比例，逐 token、逐 channel 变化 |
| $B$ | 最大训练 loop，论文设为 20 |
| $K$ | 每个 batch 实际监督的随机 steps，论文设为 5 |

## 4. 从层间表示选择 loop block

论文先观察 pretrained LLM 的相邻层 cosine distance：早期层把 embedding 转为 contextual state，中间层的表示演化相对平缓，末层更接近输出空间。于是把中间连续层作为 $M$：

$$
h_0=E(x),\qquad h_{b+1}=G\bigl(h_b,M(h_b)\bigr),\qquad
p(y)=D(h_B).
$$

这是一种 empirical layer-role hypothesis。cosine similarity 能说明方向相似度，不能证明某组中间层天然是收敛算子；因此作者还需要 gate 与多 step supervision。

![LoopUS 架构、随机监督与 confidence exit](/assets/looped-transformer/18-loopus/figure-2-selective-loop.png)

*原论文 Figure 2，PDF p.4，[arXiv v1 PDF](https://arxiv.org/pdf/2605.11011v1)。看图重点：左图是 encoder–loop–decoder 重组，右图同时显示随机选择的 supervised steps、monotonic loss 和 confidence head。不是每个 step 都保留完整梯度图。*

## 5. Selective gate

先只记住一行：

$$
h_{b+1}=h_b+\alpha_b\odot\delta_b,
\qquad
\delta_b=M(h_b)-h_b.
$$

middle block $M$ 先提出一个新 state，$\delta_b$ 是“新提议相对旧 state 想移动多少”；selective gate 再决定这个位移的每个坐标实际放行多少。它是一个**逐坐标的 learned damping mechanism**，不是另一个 reasoning block。

### 5.1 为什么直接重复 $M$ 容易出问题

pretrained model 原来只会沿 depth 使用中间层一次。某组中间层 $M$ 在 pretraining 时见到的是特定深度的 hidden-state distribution，却没有被训练成可以对自己的输出反复作用的 recurrent operator。若直接循环：

$$
h_{b+1}=M(h_b)=h_b+\delta_b,
$$

每轮都会完整接受 $\delta_b$。第一轮之后，$h_{b+1}$ 可能已经偏离 $M$ 熟悉的输入分布；下一轮的误差可能继续累积，形成 hidden-state drift。selective gate 的目标不是判断 $M(h_b)$ 是否“正确”，而是给这个未经 recurrent pretraining 的 operator 加一个可学习的刹车。

### 5.2 三步计算

第一步，middle block 给出 candidate state $\tilde h_b$，并计算它与旧 state 的差：

$$
\tilde h_b=M(h_b),\qquad \delta_b=\tilde h_b-h_b.
$$

第二步，用这个差值预测正数 $\Delta_b$：

$$
\Delta_b=\operatorname{softplus}(W_\Delta\delta_b+b_\Delta),
$$

其中 softplus 保证 $\Delta_b>0$。论文再令每个 hidden channel 有一个负的 learned coefficient $A<0$。把 $A=-\lambda$、$\lambda>0$ 代入后，gate 可写得更直观：

$$
\alpha_b
=\exp(\Delta_b\odot A)
=\exp(-\Delta_b\odot\lambda),
\qquad 0<\alpha_b<1.
$$

第三步，在旧 state 和 candidate 之间插值：

$$
\begin{aligned}
h_{b+1}
&=\alpha_b\odot \tilde h_b+(1-\alpha_b)\odot h_b\\
&=h_b+\alpha_b\odot(\tilde h_b-h_b)\\
&=h_b+\alpha_b\odot\delta_b.
\end{aligned}
$$

这也澄清了一个很容易读反的命名：论文把 $\alpha_b$ 称为 decay factor，但在最终更新式和[官方实现](https://github.com/Thrillcrazyer/LoopUS/blob/main/models/modeling_lds.py#L2789-L2808)中，它乘的是 `h_new`，不是 `h_old`。因此：

- $\Delta_b\lambda\to 0$ 时，$\alpha_b\to 1$：几乎完整接受 candidate；
- $\Delta_b\lambda$ 较大时，$\alpha_b\to 0$：几乎拒绝 candidate，保留旧 state；
- 真正表示“放行比例”的是 $\alpha_b$；$\Delta_b$ 虽被论文称作 step size，但单独看它会得到相反直觉。更准确地说，$\Delta_b\lambda$ 是 damping strength。

### 5.3 “Selective” 到底选择什么

省略 batch 维时，$h_b,\tilde h_b,\delta_b,\Delta_b,\alpha_b\in\mathbb R^{T\times d}$。也就是说，gate 不是给整个 sample 一个 scalar，也不是每个 token 一个 scalar，而是为**每个 token 的每个 hidden channel**产生一个 gate 值：

```text
hidden state / candidate / residual: [batch, T, d]
gate alpha:                         [batch, T, d]
learned decay A:                    [d]  → 向 batch、token 维广播
```

其中 $A$ 是跨 input、token 和 loop step 共享的 channel-wise parameter；$\Delta_b$ 由当前 $\delta_b$ 产生，所以随 input、token、channel 和 loop step 改变。这就是 input-dependent selective：不同位置、不同 feature 可以选择不同的更新幅度。

论文把 $W_\Delta$ 写成一个投影；官方代码实际用低秩路径降低参数与计算量：

$$
\delta_b\in\mathbb R^d
\xrightarrow{W_{\text{down}}}
\mathbb R^r
\xrightarrow{W_{\text{up}}}
\mathbb R^d
\xrightarrow{\text{softplus}}
\Delta_b,
\qquad r=\left\lceil\frac d{16}\right\rceil.
$$

注意它**不选择是否执行 middle block**：$M(h_b)$ 已经算完，gate 才混合新旧 state。因此 selective gate 本身不节省这一轮 Transformer FLOPs；决定是否停止后续 loops 的是 confidence head / early exit。

### 5.4 $A$ 的 shape，以及哪些量可以训练

$A$ 不是一个 dense $d\times d$ matrix。实现实际保存的是一个长度为 hidden size 的 parameter vector：

$$
a_{\log}\in\mathbb R^d,
\qquad
A=-\exp(a_{\log})\in\mathbb R_{<0}^d.
$$

使用 $-\exp(\cdot)$ 的目的，是让 optimizer 可以自由更新 $a_{\log}$，同时保证 forward 中真正使用的每个 $A_j$ 永远为负。若一定要写成 matrix operator，它等价于 diagonal matrix：

$$
\operatorname{Diag}(A)\in\mathbb R^{d\times d},
$$

但代码没有存储或学习一个完整的 $d\times d$ matrix；不同 channel 有独立 decay rate，channel 之间的混合由前面的 projection 完成。

设 $r=\lceil d/16\rceil$。selective gate 内的 parameter/activation 应严格区分：

| 对象 | shape | 是否 trainable parameter | 作用 |
|---|---:|---|---|
| $W_{\mathrm{down}}$ | $r\times d$ | 是 | 把 proposed update 从 $d$ 维压到低秩空间 |
| $W_{\mathrm{up}}$ | $d\times r$ | 是 | 从低秩空间产生每个 channel 的 $\Delta$ |
| $b_\Delta$ | $d$ | 是 | $\Delta$ projection 的 bias |
| $a_{\log}$ | $d$ | 是 | unconstrained parameter，间接决定负值 $A$ |
| $A=-\exp(a_{\log})$ | $d$ | 否，derived tensor | forward 中使用的负 decay vector |
| $\delta_b,\Delta_b,\alpha_b$ | $N\times T\times d$ | 否，runtime activations | 随 batch/input/step 动态计算 |

因此 gate 的可训练计算是：

$$
\begin{aligned}
z_b &= \delta_b W_{\mathrm{down}}^\top,\\
\Delta_b &= \operatorname{softplus}
\left(z_bW_{\mathrm{up}}^\top+b_\Delta\right),\\
A &= -\exp(a_{\log}),\\
\alpha_b &= \exp(\Delta_b\odot A).
\end{aligned}
$$

这里真正由 optimizer 直接更新的是
$\{W_{\mathrm{down}},W_{\mathrm{up}},b_\Delta,a_{\log}\}$；$A$ 和 $\alpha_b$ 都是从这些 parameter 算出来的，不能作为独立参数直接修改。

#### 整个 LoopUS 中还有哪些 parameter

除了 selective gate，模型还有 pretrained encoder $E$、middle block $M$、decoder $D$，以及 confidence head。confidence head 在官方代码中是：

```text
LayerNorm(d) → Linear(d, 1)
```

所以其 LayerNorm scale/bias、Linear weight/bias 也是 trainable parameters。当前公开训练代码把 `combined_model.parameters()` 整体交给 AdamW，且没有显式 `requires_grad=False`；从 optimizer membership 看，$E/M/D$ 的 pretrained weights、gate 和 confidence head 都被列为可训练参数。

但“被放进 optimizer”不等于“每一步都得到 gradient”。random deep supervision 会对未监督 steps 使用 `no_grad` / detach；在 supervised step，当前 $M$ 调用、gate、decoder 和 confidence head 获得局部 gradient，而之前已经 detach 的 steps 不会收到跨 step gradient。当前公开 helper 还会在监督前 detach 输入 hidden state，因此该路径也不会继续反传到更早的 loop 或 encoder。理解训练范围时，应同时检查 optimizer parameter list 与实际 autograd path，不能只看 `requires_grad=True`。

### 5.5 两维数值例子

设旧 state、candidate 和 proposed update 为：

$$
h_b=(2,-1),\qquad
\tilde h_b=(6,3),\qquad
\delta_b=(4,4).
$$

假设两个 channel 的 damping strength 分别为：

$$
\Delta_b\lambda=(0.223,2.303),
$$

则：

$$
\alpha_b=e^{-\Delta_b\lambda}\approx(0.8,0.1).
$$

最终：

$$
h_{b+1}
=(2,-1)+(0.8,0.1)\odot(4,4)
=(5.2,-0.6).
$$

同样是 $+4$ 的 proposed update，第一个 channel 放行 $80\%$，第二个只放行 $10\%$。这就是“selective”：它选择的是 update 的坐标和幅度，不是选择一个离散的 reasoning branch。

### 5.6 为什么它像 fixed-point iteration，但不保证收敛

把一轮整体写成：

$$
F(h)=h+\alpha(h)\odot\bigl(M(h)-h\bigr).
$$

只要 $\alpha$ 严格大于 0，$F(h^\star)=h^\star$ 仍要求：

$$
M(h^\star)=h^\star.
$$

所以 gate 主要改变到达 fixed point 的**轨迹和速度**，并不会凭空创造一个正确 fixed point。用一维、在 fixed point 附近把 $\alpha$ 看作常数的最小分析更容易看出边界：

$$
F'(h^\star)=1+\alpha\bigl(M'(h^\star)-1\bigr).
$$

- 若 $M'(h^\star)=-3$，naive iteration 的局部斜率为 $-3$，会振荡发散；取 $\alpha=0.25$ 后，$F'(h^\star)=0$，damping 可以稳定 overshoot。
- 若 $M'(h^\star)=3$，则 $F'(h^\star)=1+2\alpha>1$；无论怎样选择正的 $\alpha$，这个方向仍不收敛。

真实 Transformer 是高维非线性 operator，$\alpha$ 还依赖 $h$；其 Jacobian 包含 $M$、gate 以及 channel coupling。$0<\alpha<1$ 只保证新 state 的每个坐标位于旧 state 与 candidate 的对应坐标之间，限制单步位移；它不等价于证明 $F$ 是 contraction。论文展示的是 contraction-like 的经验行为和 loss ablation，不是数学收敛保证。

### 5.7 gate 是怎样学会“开多少”的

gate 没有直接的“答案对不对”标签。它与 $M$ 一起通过 sampled steps 上的 next-token loss 和 monotonicity loss 接收梯度：如果某类 $\delta_b$ 经常改善预测，训练会倾向于放大相应 $\alpha_b$；如果某类 update 经常使 loss 变差，则倾向于减小相应 $\alpha_b$。

这也有两个需监控的饱和边界：

- $\alpha_b\approx 0$：state 很稳定，但 middle block 的 update 几乎被关闭，相关梯度也会变弱；
- $\alpha_b\approx 1$：充分使用新计算，但几乎失去 damping，重新接近 naive loop。

论文的 component ablation 显示，移除 selective gate 或换成 sigmoid gate 后，训练曲线更不稳定且最终 LM loss 更高；这支持该设计在论文设置中有用，但仍不能单独证明 gate 是 benchmark 增益的唯一原因。

一句话总结：**$M$ 负责提出“往哪里改”，selective gate 负责决定“每个 token 的每个 feature 这次改多少”，confidence head 才负责决定“还要不要再循环一次”。**

## 6. Random deep supervision

最大循环 $B=20$，但每个 batch 随机选择 $K=5$ 个 steps 做 supervision。这里必须区分两件事：

1. **为了继续 forward 而使用当前 hidden state**；
2. **为了以后 backward 而保存整套 autograd activations**。

用户直觉中的“中间 activation 也要用”对第一点完全正确：20 次 $M(h_b)$ 都必须执行，$h_{b+1}$ 也必须作为下一轮输入。Random deep supervision 没有省掉这 20 次 middle-block forward。

它省的是第二点。Transformer 一次 forward 不只产生最终 $h_{b+1}$；为了 backward，autograd 通常还要保存每层的 Q/K/V、attention 中间结果、normalization statistics、MLP expansion activation、gate 输入等 tensors。若完整 unroll 20 steps 并在最后统一 backward，这些 saved tensors 会沿 20 轮 graph 累积。

### 6.1 一个具体执行轨迹

假设抽到的 supervised steps 是 $\{3,7,12,16,20\}$：

```text
step 1:  no_grad: h0 → M/Gate → h1；释放 M 内部 tensors，只留下 h1
step 2:  no_grad: h1 → M/Gate → h2；释放 M 内部 tensors，只留下 h2
step 3:  grad:    detach(h2) → M/Gate → h3 → Decoder → loss
                    backward 后释放当前局部 graph，再把 h3 detach
step 4–6: no_grad，只滚动当前 h
step 7:  对 detach(h6) 做一次局部 supervised update
...
```

这里 $h_1,h_2$ 当然被依次使用过，但不需要同时保存 $h_0,h_1,h_2$，也不需要保留 step 1–2 内部的 attention/MLP activations。它像一个覆盖更新的 accumulator：

```text
h = next_h(h)
```

而不是把每步连着 graph 放入 list：

```text
all_h.append(next_h(all_h[-1]))
```

### 6.2 到底节省哪一项

设一次 middle block forward 成本为 $C_F$，对应 backward 成本为 $C_B$，保存一轮 backward 所需 tensors 的 memory 为 $M_{\mathrm{act}}$。忽略 decoder、parameter、optimizer state 等共同项：

| 方案 | middle-block forward | middle-block backward | 主要 activation-memory 量级 |
|---|---:|---:|---:|
| full 20-step BPTT | $20C_F$ | 约 $20C_B$ | 约 $20M_{\mathrm{act}}$ |
| LoopUS，$K=5$ | $20C_F$ | 约 $5C_B$ | 逐 supervised step backward 时接近 $M_{\mathrm{act}}+O(NTd)$ |

若实现先累计 5 个 loss、最后统一 backward，peak graph memory 会更接近 $5M_{\mathrm{act}}$；若像论文 pseudocode 一样在每个 selected step 后立即 backward 并 detach，则 peak 不需要随 $K$ 线性增长。无论哪种实现，**$20C_F$ 都存在**。

此外，每个 supervised step 还会执行 decoder、LM loss、monotonicity loss 和 confidence loss，所以真实 wall-clock 并不是简单地从 20 降到 5。更准确的成本式是：

$$
C_{\mathrm{LoopUS}}
\approx 20C_F+5C_B+C_{\mathrm{5\ supervised\ heads}}.
$$

因此该方法最可靠的表述是：

- 显著降低 activation memory；
- 减少 middle block 的 backward 数量；
- 不减少训练时的 20 次 recurrent forward；
- 不应把 $K/B=5/20$ 解释成训练 FLOPs 或 wall-clock 只剩 $25\%$。

### 6.3 节省的代价：gradient 不再穿过完整轨迹

full BPTT 可以让 step 20 的 loss 通过链式法则影响 step 1：

$$
\frac{\partial \mathcal L_{20}}{\partial h_1}
=
\frac{\partial \mathcal L_{20}}{\partial h_{20}}
\prod_{b=1}^{19}
\frac{\partial h_{b+1}}{\partial h_b}.
$$

LoopUS 在 `no_grad` / detach 处把这条乘积截断。step 7 的 supervision 主要训练的是：“给定已经得到但视为常量的 $h_6$，当前共享 block 怎样做一次有用修改”，而不是“step 1 怎样改变才能让 step 20 最好”。

所以 random deep supervision 优化的是随机抽取深度上的**局部 correction objective**，不是完整 20-step BPTT 的无偏近似。它换取 memory 与 backward 成本的方式，正是放弃跨长 loop 的 end-to-end credit assignment。

一句话总结：**中间 hidden state 必须算、必须用，但无需为 backward 保留它背后的整条计算历史；LoopUS 省 graph，不省 20 次 forward。**

## 7. 三类 loss

### 7.1 每步 next-token loss

被选中的 step $b$ 经 decoder/head 计算：

$$
\mathcal L_{\mathrm{LM}}^{(b)}
=-\sum_i\log p_\theta(x_{i+1}\mid x_{\le i},h_b).
$$

### 7.2 Monotonicity loss

先把 decoder 定义成一个衡量 hidden state “离正确 next-token prediction 有多远”的标尺：

$$
E_x(h)=\operatorname{CE}\!\left(\mathcal D(h),x_{2:T}\right).
$$

这里 $x$ 是当前训练序列，$h$ 是某个 loop depth 的 hidden state，$\mathcal D$ 是 decoder，$E_x(h)$ 是对 batch 中有效 next-token positions 求平均得到的标量 cross-entropy。它不是独立训练出来的 energy model，只是借用现有 decoder 与 gold tokens 定义的 **task-aligned surrogate energy**：越小表示当前 state 越有利于预测正确 token。

对一次局部 transition $h^{(b-1)}\rightarrow h^{(b)}$，先计算 loss 的变化量：

$$
z_b=E_x(h^{(b)})-E_x(h^{(b-1)}),
$$

再定义：

$$
\mathcal L_{\mathrm{mono}}^{(b)}
=\operatorname{SiLU}(z_b)
=z_b\,\sigma(z_b).
$$

其中 $\sigma$ 是 sigmoid。它比较的是同一 batch、同一 target 在一次 loop 更新前后的预测 loss：

| 情况 | $z_b$ | $\mathcal L_{\mathrm{mono}}^{(b)}$ | 含义 |
|---|---:|---:|---|
| 后一步更差 | $>0$ | $>0$，大正数时约等于 $z_b$ | 给退步加 penalty |
| 一样好 | $=0$ | $0$ | 不加不减 |
| 后一步稍好 | $<0$ | 小负数 | 给改善一个有限的 soft reward |
| 后一步好非常多 | $\ll0$ | 从负侧趋近 $0$ | 不让 reward 随改善幅度无限增长 |

例如前一步 CE 为 $2.0$：

- 更新后变成 $2.4$：$z_b=0.4$，mono loss 约为 $0.4\sigma(0.4)=0.239$，明确惩罚这次退步；
- 更新后变成 $1.8$：$z_b=-0.2$，mono loss 约为 $-0.090$，总 objective 略微降低；
- 如果 $z_b=-8$，mono loss 只有约 $-0.0027$。所以它只提供有界的 soft reward，不会靠制造极大的 step-to-step loss gap 来无限降低 objective。

这也是作者选择 SiLU 而非普通 hinge/ReLU 的原因。若写成

$$
\operatorname{ReLU}(z_b)=\max(0,z_b),
$$

它只能惩罚退步；一旦 $z_b<0$，该项就完全为零。SiLU 在小幅改善区仍给出负值，同时其最小值大约只有 $-0.278$，所以辅助项不容易压过主 LM loss。注意 `monotonicity loss` 实际上可以为负，它更准确地说是一个 signed regularizer，而不是通常要求非负的误差度量。

官方训练实现实际使用：

$$
\mathcal L^{(b)}
=E_x(h^{(b)})
+\beta\,\mathcal L_{\mathrm{mono}}^{(b)}
+\mathcal L_Q^{(b)},
$$

当前代码的默认值是 $\beta=1$。最重要的 autograd 路径是：

```text
                              有梯度
detach(h_{b-1}) -> M/Gate -> h_b -> Decoder -> E_b -----------+
       |                                                        |
       +-> Decoder -> E_{b-1}   [no_grad，只有比较值]           |
                                                                v
                                      SiLU(E_b - stopgrad(E_{b-1}))
```

因此，前一步的 $E_x(h^{(b-1)})$ 是一个停止梯度的 baseline。monotonicity term 的梯度只经当前分支流向当前调用的 reasoning block、selective gate 和 decoder；不会通过 $h^{(b-1)}$ 回到更早的 loops 或 encoder。用公式写：

$$
\nabla_\theta\mathcal L_{\mathrm{mono}}^{(b)}
=\operatorname{SiLU}'(z_b)\,
\nabla_\theta E_x(h^{(b)}),
$$

而没有 $-\nabla_\theta E_x(h^{(b-1)})$ 那条分支。它本质上是在根据“这次更新相对基准是进步还是退步”，动态调整降低当前 CE 的训练力度，不是让两个相邻 state 组成一条跨 step 的反向传播链。

这也回答了它如何配合 random deep supervision。假设只抽中 steps $\{3,7,12,16,20\}$：

- step 7 的 mono loss 比较 $h_6$ 与 $h_7$，不是比较 $h_3$ 与 $h_7$；
- 即使 step 6 没有 supervision，$h_6$ 仍可作为 detached pre-update baseline；
- 为得到 $E_x(h_6)$，每个 supervised step 会额外执行一次无梯度 decoder forward，但不保存该 decoder 的 backward activations；
- 所以它增加了计算量，却没有恢复完整 20-step BPTT 的 activation memory。

普通 $\mathcal L_{\mathrm{LM}}^{(b)}$ 只说“当前输出要好”，monotonicity loss 额外说“共享 block 的这一次应用不能把已有 state 改坏”。由于同一个 $M$ 会被重复使用，这个 local before/after constraint 很重要：它把 loop 从“每个抽样深度各自能预测”进一步推向“每次局部 correction 大体沿着 decoder CE 下降的方向走”。

但名字中的 monotonicity 不能理解成数学保证。它最多鼓励训练分布上的**近似单调**：

1. $E_x$ 是对 token/batch 聚合的 CE；平均值下降时，个别 token 仍可能变差；
2. 只在随机抽中的 transitions 上施加该项，不是每次训练都检查全部 20 步；
3. CE 下降不等于下游 benchmark accuracy 或完整 reasoning quality 必然提高；
4. SiLU 是 soft penalty，且权重 $\beta$ 有限，优化器可以为了其他 loss 接受局部退步；
5. 推理时遇到分布外输入或使用更深的未见 recursion depth，仍可能出现非单调变化。

一句话总结：**LM loss 教当前 state 预测正确答案；monotonicity loss 把更新前的 state 当作无梯度基线，要求这一次 loop update 至少别把 decoder CE 变差。**论文公式见 [Section 3.2, Eq. 12](https://arxiv.org/html/2605.11011v1#S3.SS2)，梯度实现见官方 [`training_runtime.py`](https://github.com/Thrillcrazyer/LoopUS/blob/main/training_runtime.py)。

### 7.3 Confidence loss

Confidence loss 的任务不是让模型生成 token，而是训练一个很小的 confidence head 回答：**“当前 latent state 已经有多可靠，是否可以停止继续 loop？”**

#### 7.3.1 Confidence head 输出什么

论文把第 $b$ 个 refinement step 的输出写成：

$$
\tilde q_n^{(b)}=q_\phi\!\left(\bar h_n^{(b)}\right),
\qquad
q_n^{(b)}=\sigma\!\left(\tilde q_n^{(b)}\right).
$$

这里：

- $n$ 是 batch 中的样本编号；
- $\bar h_n^{(b)}\in\mathbb R^d$ 是用于判断停止的 pooled/last-token representation；
- $q_\phi$ 是参数为 $\phi$ 的 confidence head；
- $\tilde q_n^{(b)}\in\mathbb R$ 是未归一化 logit；
- $q_n^{(b)}\in(0,1)$ 是 sigmoid 后的 confidence score。

当前官方模型中的 head 是：

```text
LayerNorm(d) -> Linear(d, 1)
```

所以它每个样本只产生一个标量，trainable parameters 是 LayerNorm 的 scale/bias 和 Linear 的 weight/bias，共 $3d+1$ 个参数。它比再运行一次 Transformer reasoning block 轻得多。

#### 7.3.2 监督标签从哪里来

训练时有 gold next tokens，因此可以先让当前 state 经过 decoder：

$$
\hat x_{n,j}^{(b)}
=\arg\max_v \ell_{n,j,v}^{(b)},
\qquad
\ell^{(b)}=\mathcal D(h^{(b)}).
$$

然后统计样本 $n$ 的有效 token positions 中预测正确的比例：

$$
q_{\mathrm{target},n}^{(b)}
=\frac{1}{T_{\mathrm{valid},n}}
\sum_{j=1}^{T-1}
\mathbf 1\!\left[\hat x_{n,j}^{(b)}=x_{n,j+1}\right].
$$

padding 或 label 为 `-100` 的位置不计入分母。这得到的是每个样本一个 $[0,1]$ 软标签，而不是人工标注的 binary “停/不停”标签：

```text
样本 A：10 个有效 token，8 个 argmax 正确 -> q_target = 0.8
样本 B：10 个有效 token，3 个 argmax 正确 -> q_target = 0.3
```

因此更准确的说法是：confidence head 在拟合**当前 state 的 per-sample teacher-forced token accuracy**。

#### 7.3.3 为什么用 BCEWithLogits

Confidence loss 是 soft-target binary cross-entropy：

$$
\mathcal L_{Q,n}^{(b)}
=-q_{\mathrm{target},n}^{(b)}\log q_n^{(b)}
-\left(1-q_{\mathrm{target},n}^{(b)}\right)
\log\left(1-q_n^{(b)}\right).
$$

代码使用 `BCEWithLogits`，直接输入 $\tilde q$，在数值上比先手动 sigmoid 再取 log 更稳定。它允许 target 是 $0.3、0.8$ 这类软值。对 logit 的导数特别直观：

$$
\frac{\partial\mathcal L_Q}{\partial\tilde q_n^{(b)}}
=q_n^{(b)}-q_{\mathrm{target},n}^{(b)}.
$$

例如真实 token accuracy 是 $0.75$：

- head 只输出 $q=0.40$，梯度为 $-0.35$，优化会把 logit 和 $q$ 往上推；
- head 输出 $q=0.90$，梯度为 $+0.15$，优化会把它往下压；
- 给定相同 representation，BCE 的最优预测就是相应 target 的条件期望。

`argmax`、相等比较和 indicator 都不可导，官方代码也不会通过 `q_target` 反传。因此 $\mathcal L_Q$ 不会借 target 分支训练 decoder；它的主要梯度流向 confidence head 和产生 head input 的当前 reasoning computation。decoder 由 LM/monotonicity losses 训练。

#### 7.3.4 训练后如何 early exit

推理时没有 gold token，不能现场计算 $q_{\mathrm{target}}$；此时只运行已训练好的 confidence head：

```text
h_b -> confidence head -> q_b

if q_b >= q_threshold:
    stop looping
else:
    run another reasoning step
```

论文的标准评测设置是最大 8 个 recursion steps、$q_{\mathrm{th}}=0.6$。如果到最大深度仍未越过阈值，就强制使用最后一步。论文报告 adaptive mode 平均使用约 3.39 steps。

这和 random deep supervision 正好配合：每个 batch 随机监督不同 loop depths，confidence head 因而会看到浅层、中层、深层 state，以及它们各自的 token accuracy，学习一个跨深度共享的 readiness estimator。

有 KV cache 的 batched inference 中，论文实现采用保守的同步停止：

$$
\min_{n\in\mathcal B}q_n^{(b)}\ge q_{\mathrm{th}}
$$

时整个 batch 才停止。也就是说，只要 batch 中还有一个样本 confidence 不够，其他样本通常也要陪它继续计算。这会让理论上的 sample-wise adaptive saving 在实际 GPU batching 中打折。

#### 7.3.5 它和另外两个 loss 的分工

| Loss | 直接问题 | 训练对象 | 推理时是否直接使用 |
|---|---|---|---|
| $\mathcal L_{\mathrm{LM}}$ | 当前 state 能否预测 gold next tokens？ | reasoning/gate/decoder | 否，训练 objective |
| $\mathcal L_{\mathrm{mono}}$ | 这次 update 是否比 update 前更差？ | 当前 reasoning/gate/decoder | 否，训练 regularizer |
| $\mathcal L_Q$ | head 能否从 state 估计当前 token accuracy？ | confidence head 与其输入路径 | 是，用于 early exit |

所以 confidence loss 不负责直接提高答案质量；它负责让模型学会估计“当前答案质量 proxy”。答案质量仍主要由 LM loss 和 monotonicity loss 塑造。

#### 7.3.6 为什么它不等于“答案正确概率”

这个 confidence 的语义必须严格限制：

1. $q=0.6$ 更接近“模型预计 teacher-forced positions 中约 60% 的 argmax token 正确”，不是“完整答案有 60% 概率正确”；
2. target 是整段序列的平均 token accuracy，容易被大量简单 token 稀释；关键 reasoning token 错误时平均值仍可能很高；
3. 它监督的是当前 accuracy，不是“再多算一步能改善多少”。用它停止隐含假设：当前 accuracy 高通常意味着未来 refinement 的边际收益小；该假设没有直接写进 target；
4. BCE 让 score 拟合训练分布上的 soft label，不自动保证分布外 calibration；threshold 仍需验证和调节；
5. teacher forcing 下的 token accuracy 与自由生成时的 exposure errors、sequence-level task success 并不等价。

因此更精确的名称其实是 **token-accuracy prediction loss**。它被用作 early-exit confidence，是一种实用 proxy，而不是严格的 optimal-stopping objective。

#### 7.3.7 公开实现中的输入口径差异

`[代码核对，截止 2026-08-03]` 论文正文与 Appendix E 把 confidence head 写成读取 post-update/last-token state $\bar h^{(b)}$。但当前官方代码的训练路径中，`ReasoningBlock.forward(q_head=...)` 实际构造：

```text
q_hidden_states = hidden_states_new - hidden_states_old
pooled = q_hidden_states[last_valid_token]
q_logit = q_head(pooled)
```

即训练时 head 读取的是 gate 前 proposed update 的最后有效 token delta；这也意味着当前 $\mathcal L_Q$ 的 input branch 会训练 reasoning block，却不经过 selective gate。当前 `LDSForCausalLM.forward` 的 threshold inference 路径则直接对 post-gate `hidden_states[:, -1, :]` 调用同一个 `q_head`。这形成了一个值得复现实验时检查的 train–inference input mismatch。它不改变论文中 confidence loss 的定义，但可能影响已发布实现的 calibration 与 threshold transfer。

一句话总结：**confidence loss 用当前 step 的真实 token accuracy 生成软标签，训练一个小 head 在没有 gold answer 时预测这个 accuracy proxy；推理阶段一旦预测值超过阈值，就停止继续循环。**论文公式见 [Section 3.2, Eq. 13](https://arxiv.org/html/2605.11011v1#S3.SS2)，halting 规则见 [Appendix E](https://arxiv.org/html/2605.11011v1#Sx5)，实现见官方 [`metrics.py`](https://github.com/Thrillcrazyer/LoopUS/blob/main/utils/metrics.py)、[`modeling_lds.py`](https://github.com/Thrillcrazyer/LoopUS/blob/main/models/modeling_lds.py) 与 [`training_runtime.py`](https://github.com/Thrillcrazyer/LoopUS/blob/main/training_runtime.py)。

## 8. 训练与评测

| 项目 | 设置 |
|---|---|
| starting models | Qwen3 1.7B/4B/8B、Phi-4 14B、TinyLlama 等 |
| adaptation data | 3B FineWeb-Edu tokens |
| context window | 1024 |
| optimizer | AdamW |
| train max loops | 20 |
| supervised loops / batch | 5 |
| inference max loops | 8 |
| early-exit threshold | 0.6 |

这个阶段在论文中叫 post-training framework，但从 objective 与 data 看，更准确的通用术语是**architecture adaptation / continued LM training**，不是 instruction preference alignment。

## 9. 结果怎样解读

七个 benchmark 平均分相对 base 提升：Qwen3 1.7B +1.6、4B +1.8、8B +2.2、Phi-4 14B +1.7。它说明 3B-token adaptation 能让多个 backbone 使用 loops。

但 retained capability 不是全正向：

| 模型 | 指标 | Base | LoopUS |
|---|---|---:|---:|
| Qwen3-4B | MMLU | 68.3 | 67.7 |
| Qwen3-4B | HellaSwag | 52.1 | 51.4 |
| Qwen3-8B | MMLU | 72.8 | 71.5 |
| Qwen3-8B | HellaSwag | 57.2 | 56.0 |
| Phi-4-14B | HellaSwag | 63.1 | 60.5 |

因此“平均分上升”可能同时包含 reasoning gains 与 knowledge/commonsense regressions。研究通用 LLM 时必须保留每项结果和 post-training 后能力。

## 10. Adaptive scaling

![LoopUS 的 recurrence-depth 与 adaptive exit](/assets/looped-transformer/18-loopus/figure-5-adaptive-scaling.png)

*原论文 Figure 5，PDF p.8，[arXiv v1 PDF](https://arxiv.org/pdf/2605.11011v1)。看图重点：不同任务最佳 depth 不同，adaptive mode 平均约 3.39 次 recurrence；更深后曲线趋于平稳。该图支持 task-dependent stopping，不等于每个样本都被正确分配。*

平均 3.39 steps 相比固定 8 steps 有 theoretical saving，但真实部署还要考虑同一 batch 内不同退出时间、cache 整理和 GPU utilization。

## 11. Contribution ledger

| 类型 | 贡献 | 证据强度 |
|---|---|---|
| retrofit | 多种标准 pretrained LLM → looped model | 强：多 family + artifacts |
| stability | selective damped gate | 中强：有消融；无 contraction guarantee |
| efficient training | random deep supervision | 中：省 backward memory，仍执行 forward |
| adaptive inference | confidence early exit | 中：平均 depth 可测，部署收益待验证 |
| generality | 多类 base benchmark | 中：部分 retained capabilities 下滑 |

## 12. Claim–evidence map

| Claim | Evidence | Boundary |
|---|---|---|
| 3B tokens 可适配多个 pretrained families | Qwen/Phi/TinyLlama experiments | context 仅 1024，非完整 model flow |
| selective gate 稳定多步循环 | ablations/curves | 不证明数学收敛 |
| adaptive exit 减少平均 loops | mean 3.39 | 不等于 wall-clock 同比减少 |
| reasoning-oriented average 提升 | seven-task average | 个别 MMLU/HellaSwag 退化 |
| LoopUS 保持 pretrained capabilities | 多数任务尚可 | “完全无损”不成立 |

## 13. 局限

1. adaptation 只用 3B FineWeb-Edu、context 1024，data 与长 context 范围有限。
2. 没有统一 instruction SFT、DPO/RLVR、multi-turn chat、tool use 与 safety 评测。
3. confidence target 是 token-level proxy。
4. 跨论文 TinyLlama 对比并非 controlled comparison，作者也只作为 reference。
5. 真实 serving 中 dynamic depth 的 scheduler/KV overhead 未充分测量。

## 14. 推荐实验

1. 选择 fully-open base，同 data 做 vanilla continued training control；
2. 对 split location、gate、random supervision、monotonic loss 做全因子消融；
3. 画每任务、每 difficulty bucket 的 quality–depth curve；
4. 测 confidence calibration、overthinking false-positive/false-negative；
5. 在 SFT/DPO/RLVR 每 stage 复测 retained capabilities；
6. 在 batch size 1 与 continuous batching 下报告真实 latency。

## 15. 自测题

1. 为什么 $0<\alpha<1$ 仍不能证明整个 update 是 contraction？
2. random deep supervision 节省的是 forward 还是 backward 资源？
3. token accuracy 为什么不是完美的 exit target？
4. 平均 benchmark +2 与若干 MMLU/HellaSwag 下降能否同时成立？
5. 为什么把 3B FineWeb-Edu adaptation 直接称为 alignment post-training 会混淆？

## 16. 一句话定位

LoopUS 提供了目前较轻量、跨多种 pretrained LLM 的 latent-loop 改造方案；它的主要研究价值是 retrofit 工程与稳定性，而其“通用能力保持”和部署加速仍需更严格验证。
