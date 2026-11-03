---
title: 05 Advantage and Policy Loss
created: 2026-09-02
published: 2026-09-02
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - advantage
  - policy-loss
---

上一章结束时，128 道题已经生成 512 条 trajectory，回答和奖励保存在 TransferQueue 中。对于问题 P0——“盒中有 3 袋铅笔，每袋 4 支，一共多少支？”——四条回答得分为 `[1,0,1,0]`。现在要让参数更新利用这个差别：在这些已经生成的上下文中，提高较好回答中 token 的概率，降低较差回答中 token 的概率。

得分本身还不能直接调用 `backward()`。它来自外部评分，既没有模型的计算图，也没有说明每个 token 应获得多大的更新信号。verl 接下来先补齐概率，再计算 advantage，最后用当前 actor 的一次新 forward 建立可求导的 loss。

沿用 [[projects/verl-source-reading/01-sync-grpo-run-overview|01 的完整案例]]：两机八卡、FSDP DP=8、`B=128、G=4、M=32`，一个 PPO epoch，静态 micro-batch 为每卡 4 条；GRPO 使用组内标准差归一化，actor KL 系数为 `0.001`，reward KL 和 rollout correction 关闭。本文的短 token 序列和概率数值用于手算，不是 Qwen tokenizer 或 GPU 运行的实测输出。

## 先给已经生成的回答补上概率

`_step_once` 的顺序决定了这一章应该从哪里开始：

```text
512 条带奖励的 trajectory，已经按 workload 重排
  → actor forward：old_log_probs
  → reference forward：ref_log_prob
  → 全局计算 advantages、returns
  → actor forward + backward：current log_probs 和 loss
  → optimizer update
```

源码中的 [训练步骤](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L540) 在 advantage 前调用 `_compute_old_log_prob` 和 `_compute_ref_log_prob`。old policy 是本轮 actor 更新开始之前的概率基准；reference 是固定的参照模型。两者都对已经确定的回答 token 计算概率，不会重新生成另一份回答。

例如，`P0_0_0` 有三个示意 response token `r0,r1,r2`。它们分别对应三个不同的条件概率：

```text
π(r0 | prompt)
π(r1 | prompt, r0)
π(r2 | prompt, r0, r1)
```

因此同一句回答进入 old actor、reference 和 current actor 后，token ID 保持相同，变化的是各模型状态给这些 token 分配的概率。

[`_compute_old_log_prob`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1541) 的关键调用是：

```python
output = self.actor_rollout_wg.compute_log_prob(batch)
# 从 TQ 取回 engine 写出的全序列 log_probs 后：
data["old_log_probs"] = response_from_nested(
    data.pop("log_probs"), data["response_mask"]
)
```

第一行的 `batch` 是 512 个 key 的 metadata。WorkerGroup 把 key 分到八个 DP rank，每个 rank 取得自己的 64 条真实数据，再以每次 4 条执行 forward。结果先是全序列 log-prob，第二处调用才从中抽出与 response 对齐的部分，命名为 `old_log_probs` 并写回同一批 key。

reference 随后走相同的输入与输出对齐过程，产出 `ref_log_prob`。reference forward 不更新模型；计算 old log-prob 的 forward 也不更新 actor。本例没有使用 bypass mode，因此即使 rollout 另外保存了 `rollout_log_probs`，也不会直接拿它替代 `old_log_probs`。生成引擎和训练引擎的数值实现可能不同，重新计算为后面的 ratio 提供了训练引擎侧的固定分母。

## 一次 forward 的输入为什么是 `[1,38]`

这里先放大 rank 0 的一次 micro-batch。沿用 04 的教学排列，以下四条位于重排后全批的下标 `16、17、18、19`，也就是 rank 0 收到的 64 条中的本地下标 `16…19`。所有下标从 0 开始。

| 本次 micro-batch 的行 | trajectory key | prompt 长度 | response 长度 | 完整序列长度 |
| --------------------- | -------------- | ----------: | ------------: | -----------: |
| 0                     | `P0_0_0`       |           6 |             3 |            9 |
| 1                     | `P17_2_0`      |           6 |             5 |           11 |
| 2                     | `P42_1_0`      |           6 |             4 |           10 |
| 3                     | `P85_3_0`      |           6 |             2 |            8 |

这是一组固定沿用的教学布局，不是实际运行的 workload-balancing 记录。old-log-prob 阶段，rank 0 以每次 4 条遍历 64 条，因此它是第 5 个 inference micro-batch；训练阶段保留基线 `actor.shuffle=false`，本地 mini-batch 每次 16 条，因此同样四条成为第二个 mini-batch 的第一个 micro-batch，即 `mini=1、micro=0`。

P0 的另外三个回答位于 rank 2、4、6，各自也在本地下标 16。GRPO 的四条同题回答没有必须一起做 forward 的要求；只是这个教学排列恰好让它们参与同一次全局 mini-batch 更新。

TQ 取出的 `input_ids` 是 jagged nested tensor，逻辑上有四行，物理上可以理解为连续 values 和行边界：

```text
input_ids.values():  38 个 token
input_ids.offsets(): [0, 9, 20, 30, 38]

第 0 行：values[ 0: 9] = 6 个 prompt token + 3 个 response token
第 1 行：values[ 9:20] = 6 个 prompt token + 5 个 response token
第 2 行：values[20:30] = 6 个 prompt token + 4 个 response token
第 3 行：values[30:38] = 6 个 prompt token + 2 个 response token
```

本例启用 `use_remove_padding=True`，SP=1，且不额外补 packed bucket padding。[`prepare_model_inputs`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L1178) 直接使用有效 token：

```python
input_ids_rmpad = input_ids.values().unsqueeze(0)
position_ids_rmpad = position_ids.values().unsqueeze(0)
input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
```

于是输入由“4 条变长序列”变成模型接口上的 `[1,38]`。其中 `1` 是 packed 输入的外层维度，不表示本次只有一个样本；本次仍有四条相互独立的序列。

对应的纯文本 `position_ids` 在每个样本起点重新从 0 开始：

```text
0 1 … 8 | 0 1 … 10 | 0 1 … 9 | 0 1 … 7
```

模型适配后的变长 attention 利用序列边界处理这些 packed tokens。这里不能把 `attention_mask=None` 理解成允许四条回答相互注意：remove-padding 路径依赖兼容的模型 attention 实现。[attention patch](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/models/transformers/monkey_patch.py#L135) 把重置后的 `position_ids` 传给 FlashAttention 路径；适用的模型还会接收 cumulative sequence lengths。完整的模型适配在 [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]] 展开。

## logit 的位置比它预测的 token 早一格

先只看 `P0_0_0`，把 prompt 六个 token 记为 `p0…p5`：

```text
输入位置        0   1   2   3   4   5   6   7   8
input_ids      p0  p1  p2  p3  p4  p5  r0  r1  r2
该位置预测      p1  p2  p3  p4  p5  r0  r1  r2  下一样本首token
response所需                        ↑   ↑   ↑
logit位置                           5   6   7
```

`torch.roll(..., shifts=-1)` 把下一个 token ID 作为当前位置的 label。普通、非 fused 的输出路径先得到 logits `[1,38,V]`，去掉外层维度后为 `[38,V]`，其中 `V` 是模型词表大小。源码先按 temperature 缩放 logits，本例 temperature=1.0，数值保持不变。随后 `logprobs_from_logits` 从每个位置的词表分布中选出对应 label，得到 `[38]` 的 log-prob，再按原 offsets 还原为四行 nested 输出。

第一条 response 的三个概率来自位置 `5,6,7`，不是 `6,7,8`。最后一个位置预测到了下一条样本的开头，但这个跨样本 label 不属于任何 response 所需的概率，因此后面的切片会排除它；attention 本身仍必须正确隔离样本。

[`response_from_nested`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/utils/padding.py#L194) 正是在做这个对齐：

```python
for resp_len, seq_offset in zip(response_lens, offsets[1:], strict=True):
    response_list.append(
        values[seq_offset - resp_len - 1 : seq_offset - 1]
    )
```

代入四条序列的结束位置 `[9,20,30,38]`：

```text
P0_0_0：   [9 - 3 - 1 : 9 - 1]   = [ 5: 8]，3 个概率
P17_2_0： [20 - 5 - 1 : 20 - 1] = [14:19]，5 个概率
P42_1_0： [30 - 4 - 1 : 30 - 1] = [25:29]，4 个概率
P85_3_0： [38 - 2 - 1 : 38 - 1] = [35:37]，2 个概率
```

`old_log_probs` 因而只有 14 个 response 位置，offsets 为 `[0,3,8,12,14]`。模型需要完整 prompt+response 才能计算条件概率，后续 PPO loss 只需要 response 对应的部分。这两种长度不同，不能用一个笼统的 `seq_len` 代替。

current actor 训练时会重复相同的 forward 和对齐过程；不同之处是保留梯度，并由 [`no_padding_2_padding`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/utils/padding.py#L97) 直接得到本次 micro-batch 的 padded response 概率，例如 `[4,5]`。形状相同的 old/current 概率才能逐 token 相除。

## 回到 P0：四个得分怎样变成 advantage

old/ref 概率都写回后，controller 的 [`_compute_advantage`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1650) 一次取出本轮 512 条 trajectory 的相关字段，并调用 `to_padded_tensor()`。

此处是算法阶段的全局矩阵，response 字段形状为 `[512,R_max]`；`R_max` 是本轮这些 response 的最大存储长度，不必等于配置的最大生成长度。controller 不需要完整词表 logits，只需要 reward、mask、uid 和相应的 log-prob 等字段。

为了手算，现在从 512 行中找到 `uid=P0` 的四行，按 session 顺序单独列出。沿用 response 长度 `[3,5,4,2]`，局部演算补齐到宽度 5：

```text
              response_mask [4,5]       rm_scores [4,5]
P0_0_0        1 1 1 0 0                0 0 1 0 0
P0_1_0        1 1 1 1 1                0 0 0 0 0
P0_2_0        1 1 1 1 0                0 0 0 1 0
P0_3_0        1 1 0 0 0                0 0 0 0 0
```

评分分别是正确 `12`、错误 `7`、正确 `12`、错误 `16`。正确性得分放在最后一个有效 response 位置；零奖励回答仍然有有效 token。不能拿 `rm_scores != 0` 充当有效位置的 mask。

由于本例关闭 reward KL，代码直接令 `token_level_rewards=token_level_scores=rm_scores`。[`compute_grpo_outcome_advantage`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L268) 先执行：

```python
scores = token_level_rewards.sum(dim=-1)
```

它将 `[4,5]` 压成四个回答分数 `[1,0,1,0]`。随后按照 `uid` 收集同一道题的得分，计算均值和标准差；这里的 group 由身份关联，既不由相邻四行定义，也不由 GPU 编号定义。

```text
mean = (1 + 0 + 1 + 0) / 4 = 0.5

std = sqrt(((1-0.5)² + (0-0.5)² + (1-0.5)² + (0-0.5)²) / 3)
    = sqrt(1/3)
    ≈ 0.577350

a = 0.5 / (std + 1e-6) ≈ 0.866024
归一化分数 = [+a, -a, +a, -a]
```

分母中的 `3` 来自源码 `torch.std(scores_tensor)` 的样本标准差约定，即 `correction=1`。若按总体标准差除以 `4`，就会算出 `±1`，与当前实现不同。

下一步广播到每个 action token：

```python
scores = scores.unsqueeze(-1) * response_mask
```

```text
[4] → [4,1] → [4,5]

advantages：
 +a +a +a  0  0
 -a -a -a -a -a
 +a +a +a +a  0
 -a -a  0  0  0
```

这样，回答级别的奖励差异变成了逐 token 的优化信号。它没有判断某一个具体 token 是否造成最终答错，而是把同一回答的相对得分赋给该回答的所有有效 action token。本例为单轮纯文本，没有工具 observation；工具场景中，属于 observation 的 response 位置也会被 `response_mask=0` 排除。

V1 先经过 [`compute_advantage_for_multi_trajectories`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/utils.py#L148)：它从 key 解析 `uid/session/index`，每个 session 只选最大 index 的 output 比较分数，再把得到的 scalar advantage 广播给同一 session 的 outputs。本例每个 session 只有一条 output，因此正好退化为上面的四行计算。

GRPO 返回的 `returns` 与 `advantages` 相同，这是该 estimator 的输出接口，不表示这里还需要训练 critic。最终 `response_to_nested` 按原 response 长度截去 padded 尾部，把 `advantages/returns` 写回原来的 trajectory key。P0 四条的有效 advantage 共 14 个位置，之后无论 key 怎样重排，都仍能取到各自的值。

若四个回答全得 1 或全得 0，中心化后都为 0，该组不贡献 GRPO policy gradient；本例仍可能有 reference KL 的梯度。若关闭标准差归一化，四个值变为 `±0.5`。若只剩一个 session，源码使用 mean=0、std=1 的特殊处理，此时已经没有真正的组内相对比较。

## 512 条样本怎样成为四次参数更新

advantage 写回后，所有学习信号都已准备好。`_update_actor` 并不会先在 controller 上切出四批再发四次远程调用；它向 worker 传递整批 512 个 key，并设置：

```python
ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
ppo_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
# extra_info 中 global_batch_size、mini_batch_size 都设为此值。
```

本例 `32×4=128` 是每次 optimizer update 覆盖的全局 trajectory 数。dispatch 先按 DP=8 分发，随后每个 worker 的 [`train_mini_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L241) 才计算本地大小：

```python
mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()
```

```text
一次 update_actor RPC：512 条
  → 每 rank 64 条
  → 每 rank 4 个 mini-batch，每个 16 条
  → 每个本地 mini-batch 再分 4 个 micro-batch，每个 4 条

同一次 optimizer update：8 ranks × 16 条 = 128 条
本轮 optimizer update 数：512 / 128 = 4
```

八张卡各执行一次 `optimizer.step()` 是同一次分布式更新，不是八次连续更新。每卡的四个 micro-batch 分别执行 forward/backward，梯度累计到同一个本地 mini-batch 的更新中。

`M=32` 决定全局更新量和更新次数；GPU 数决定这一更新量怎样分到各 rank。仅改变 DP 数、保持全局 mini-batch 不变，不会自动把四次更新变成八次。若改为两个 PPO epoch，则对同一份数据再遍历一次，共八次更新。

## `ppo_loss` 计算的是本次 micro-batch

现在来到包含 `P0_0_0` 的那次训练 forward：rank 0 的第二个 mini-batch、第一个 micro-batch。第一个全局 mini-batch 已把 actor 从 `θ0` 更新为 `θ1`，所以本次 current probability 由 `θ1` 计算，old probability 仍来自先前的 `θ0`。与 old forward 相同，它先对完整序列计算概率，再提取 response 部分。[`ppo_loss`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/utils/losses.py#L57) 将 old probability、advantage 和 mask 转为 padded tensor，与 current probability 对齐。

必须区分三个外观相似的矩阵：

| 所在位置                      | response 矩阵形状 | 行表示什么                             |
| ----------------------------- | ----------------- | -------------------------------------- |
| controller 计算本轮 advantage | `[512,R_max]`     | 本轮全部 trajectory                    |
| 单独演算 P0                   | `[4,5]`           | 同题的四个 session                     |
| 一个 rank 的一次 `ppo_loss`   | `[4,R_micro]`     | 该 rank 本次 forward 的四条 trajectory |

教学 micro-batch 的 `R_micro=5`，但它的另外三行来自 P17、P42、P85。相同形状不意味着相同的一组数据。实际本地 `ppo_loss` 不会突然看到全局 128 条，更不会看到完整 512 条。

对于每个有效 response token，vanilla policy loss 先求概率比值：

$$
r_t(\theta)=\exp\bigl(\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t)\bigr).
$$

这里 `old_log_probs` 是固定输入，只有当前 `log_probs` 连接本次 forward 的计算图。若某 token 的 old probability 为 `0.2`，current probability 为 `0.28`，则 `ratio=1.4`。源码先把 log-ratio 截到 `[-20,20]` 防止指数溢出，再做 PPO clip。

基线使用左右 clip `0.2`，因此：

$$
\ell_1=-A_t r_t,\qquad
\ell_2=-A_t\operatorname{clip}(r_t,0.8,1.2),\qquad
\ell_{clip}=\max(\ell_1,\ell_2).
$$

将 P0 的正负 advantage 分别代入，就能看出两个方向的作用：

| 示例 token        | ratio | `ell_1` | `ell_2` | 选中的 loss | 当前含义                               |
| ----------------- | ----: | ------: | ------: | ----------: | -------------------------------------- |
| 正 advantage `+a` |   1.1 | `-1.1a` | `-1.1a` |     `-1.1a` | 增加该 token 概率仍降低 loss           |
| 正 advantage `+a` |   1.4 | `-1.4a` | `-1.2a` |     `-1.2a` | 概率已提高较多，这一项不再奖励继续提高 |
| 负 advantage `-a` |   0.6 | `+0.6a` | `+0.8a` |     `+0.8a` | 概率已降低较多，这一项不再奖励继续降低 |
| 负 advantage `-a` |   1.4 | `+1.4a` | `+1.2a` |     `+1.4a` | 差回答概率反而提高，需要惩罚           |

clip 限制的是目标函数的部分梯度，不是执行完 optimizer 后把所有 token 的 ratio 强行限制在 `[0.8,1.2]`。同一组共享参数会同时影响很多 token，KL 项也会贡献梯度，实际 ratio 仍可能越界。

当前源码的 [`compute_policy_loss_vanilla`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L1286) 还包含 dual-clip，不能只写到上面两项为止。基线 `clip_ratio_c=3`，负 advantage 时再取：

```python
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

例如 `A=-a、ratio=4`，普通 clipped loss 是 `4a`，dual-clip 把它限制为 `3a`。上表的 ratio 都未触发这个分支，因此前面的手算保持成立。最后才用 `response_mask` 排除 padding/observation，并汇总有效 token 的 loss。

## 一个 P0 token 的 loss 怎样回传到 logits

现在只看 `P0_0_0` 的第一个 response token `r0`。它由 prompt 末尾的模型输出预测，对应前面 packed 图中的 logit 位置 5；本次训练使用 `θ1`，old probability 仍来自 `θ0`。

设它的 advantage 为 `A=+a≈0.866024`，old probability 为 `0.20`，current probability 为 `0.22`，所以 `ratio=1.1`。这一点位于 clip 区间内部，log-ratio 也没有触发数值截断，因此该 token 的 policy loss 等于：

$$
\ell=-A\exp(u-u_{\mathrm{old}}),\qquad
u=\log\pi_\theta(r_0\mid\mathrm{prompt}).
$$

`A` 和 `u_old` 都是从 TQ 取出的固定数值，不参与本次求导。对 current log-prob 求导得到：

$$
\frac{\partial\ell}{\partial u}=-A\exp(u-u_{\mathrm{old}})
=-A\,r\approx-0.952626.
$$

这个负导数说明：只考察这一项时，提高 `u` 会降低 loss。但模型优化的参数不是一个独立的 `u`；它先输出整张词表的 logits，再由 log-softmax 得到已采样 token 的概率。

用 $z_j$ 表示当前位置第 $j$ 个候选的 logit，$q_j=\operatorname{softmax}(z)_j$，已采样 token 的索引为 $s$。在本例 temperature=1 时：

$$
u=z_s-\log\sum_j e^{z_j},\qquad
\frac{\partial u}{\partial z_j}=\mathbf{1}[j=s]-q_j.
$$

沿链式法则：

$$
\frac{\partial\ell}{\partial z_j}
=-A\,r\bigl(\mathbf{1}[j=s]-q_j\bigr).
$$

为了把这个词表维度也算出来，单独构造一个三候选的局部演算：`r0` 的 current probability 是 0.22，另外两个候选是 0.30、0.48。这只是缩小词表的数学例子；真实 Qwen 词表中，其余概率分散在更多候选上。

| 候选        | current probability | `∂u/∂z` | `∂ell/∂z`，约值 |
| ----------- | ------------------: | ------: | --------------: |
| 已采样 `r0` |                0.22 |    0.78 |       -0.743049 |
| 另一个候选  |                0.30 |   -0.30 |       +0.285788 |
| 第三个候选  |                0.48 |   -0.48 |       +0.457261 |

若暂时把这三个 logits 当作独立变量沿负梯度移动，第一项增加，另两项减小，`r0` 的概率就上升。三个 logit 导数之和为零，也符合“给全部 logits 加同一个常数不改变 softmax”的性质。

这些导数接着经过输出投影和 Transformer 层回到参数。实际一次更新还汇总其他 token、其他样本以及 reference KL 的梯度，optimizer 也会使用自己的状态，所以不能由一个 token 的正 advantage 保证它在整次更新后概率一定上升。这里证明的是这一个未裁剪 policy 项提供的局部方向。

回到模型计算的形状，当前 micro-batch 的梯度路径为：

```text
packed logits [1,38,V]
  → 抽取每个已给定 next-token label 的 log-prob
  → 提取四条 response，形成 [4,5] 概率和 loss
  → mask、全局归一化 → scalar loss
  → backward：梯度沿相同的 gather / slice 回到 logits 和参数
```

虽然只抽取了 sampled token 的 log-prob，log-softmax 的分母仍涉及整个词表，因此一个有效预测位置通常会向其全部候选 logits 传递梯度。prompt token 本身不作为 response policy loss 的预测目标，但 prompt 末位的 logits 恰好用于预测第一个 response token；更早的 prompt 表示也通过 attention 影响回答。mask 为零不能理解成这段上下文与参数梯度完全无关。

再比较两个位于 clip 区间外、且远离分支边界的点：

- `A=+a、ratio=1.4` 时，选中常数分支 `ell=-1.2a`，这一 policy 项对 current log-prob 的导数为 0。
- `A=-a、ratio=0.6` 时，选中 `ell=+0.8a`，同样不再奖励继续降低该 token 的概率。

负 advantage 但 `ratio=1.4` 时，选中的仍是未裁剪项 `ell=+1.4a`，导数为正，会惩罚概率反而上升；达到 dual-clip 的常数分支后，这一项的局部梯度也会归零。在裁剪阈值处，PyTorch 按具体算子的边界导数规则处理，上述示例特意避开了阈值。reference KL 是另一个 loss 项，policy clip 不会将它的梯度一并清零。

实际代码无需手写这些导数：[`compute_policy_loss_vanilla`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L1286) 构造 tensor 运算，autograd 保留 current forward 的计算图；FSDP engine 随后调用 [`loss.backward()`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L755)。下面再把这个局部导数放进真实的全局 loss 归一化中。

## 四次 micro-batch 累加，为什么不用再除以四

每个 micro-batch 的有效 token 数不同。如果分别取各自的平均再等权累加，短回答所在 micro-batch 的单 token 权重就会更大。当前 FSDP 路径在切 micro-batch 之前，先为这次全局 mini-batch 计算统一分母。

[`forward_backward_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L719) 先对本地 16 条的 `loss_mask` 求和，再跨八个 DP rank 求和：

```python
batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
torch.distributed.all_reduce(
    batch_num_tokens, op=torch.distributed.ReduceOp.SUM,
    group=self.get_data_parallel_group()
)
```

本例 `loss_mask` 只标记有效 response action，所以分母是本次全局 128 条 response 的有效 token 数，记作 `T_global`。prompt 参与前向计算，却不增加这个 loss 分母。不要与性能统计中包括 prompt 的 `global_token_num` 混淆。

`loss_agg_mode=token-mean` 时，[`agg_loss`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L1140) 对每个 micro-batch 计算：

$$
L_{d,m}=\frac{8}{T_{global}}\sum_{t\in\text{rank }d,\text{ micro }m}\!\mathrm{mask}_t\ell_t.
$$

如果本次全局 mini-batch 有 640 个有效 response token，而某次 micro-batch 的有效 token loss 之和是 7，则该次 backward 用的标量为 `7×8/640=0.0875`。它不是 `7/14`，也不需要再除以四个 micro-batch。

当前教学基线加载的 [FSDP YAML](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/engine/fsdp.yaml#L61) 设置 `use_no_sync_for_gradient_accumulation=false`，因此每个 micro-batch backward 都保留正常的梯度同步，再把贡献累积到本次 mini-batch；四次完成后才调用 optimizer。将每次同步按 DP 平均语义表示，累计结果为：

$$
\sum_{m=0}^{3}\left(\frac{1}{8}\sum_{d=0}^{7}\nabla L_{d,m}\right)
=\frac{1}{T_{global}}\sum_{\text{全部有效 token}}\nabla\ell_t.
$$

乘 `dp_size=8` 抵消了跨 rank 的平均，所有 micro-batch 共用全局分母保证拆分不改变目标的尺度。在精确算术下，先局部求和再平均也得到同一表达式，但不能把这种代数等价直接当成本配置的通信时间顺序。显式开启 no-sync 变体后，前三个 micro-batch 才延后梯度同步；浮点运算顺序也可能带来舍入差异。640 和 7 是算式示例；实际 `T_global` 每个 mini-batch 重新计算。

其他聚合方式会改变样本权重：`seq-mean-token-mean` 先在每条回答内平均，再用全局回答数归一化；`seq-mean-token-sum` 先在回答内求和，再除全局回答数。前者让长短回答更接近等权，后者保留长回答包含更多 token 的影响。这是目标定义的变化，不能当作纯粹的显存设置。

## reference KL 加到哪里

到这里得到的是 policy loss。基线还开启 `actor.use_kl_loss=true`，因此 `ppo_loss` 对相同 response token 比较 current 和 reference probability：

```python
kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob,
                 kl_penalty=config.kl_loss_type)
kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                   loss_agg_mode=config.loss_agg_mode,
                   **config.global_batch_info)
policy_loss += kl_loss * config.kl_loss_coef
```

本例的 `low_var_kl` 在 [`kl_penalty_forward`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L2216) 中令 `d=logπ_ref-logπ_current`，计算 `exp(d)-d-1`，并做数值截断。若 current probability 为 `0.28`、reference 为 `0.2`，则 `d≈-0.33647`，该位置的估计值约为 `0.05076`，之后还需经过 mask、全局聚合和 `0.001` 系数。

它是基于 sampled token 的估计量，不是对整个词表精确求和的 KL。梯度经 current log-prob 回到当前 actor；reference probability 是固定输入。`entropy_coeff=0`，所以本案例的更新目标为：

$$
L_{actor}=L_{policy}+0.001L_{KL}.
$$

old forward 虽然计算 entropy 作为统计，也不意味着 actor 更新中启用了 entropy regularization。若开启 entropy 项，它以 `-entropy_coeff×entropy` 加入最小化目标。

reward KL 是另一个位置的机制。本例关闭它；若打开 `algorithm.use_kl_in_reward`，`_compute_advantage` 会实际调用 [`apply_kl_penalty`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/ray_trainer.py#L78)：

```python
kld = core_algos.kl_penalty(
    data.batch["old_log_probs"], data.batch["ref_log_prob"],
    kl_penalty=kl_penalty
)
kld = kld * response_mask
token_level_rewards = token_level_scores - kl_ctrl.value * kld
```

这次比较的是 old 与 reference，发生在 advantage 之前，还会更新 KL controller。P0 的四个 group score 将包含各自的 token KL 惩罚，后面的 advantage 也随之改变。actor KL 则在当前 loss 中比较 current 与 reference，四次参数更新都会重新计算，却不回头重算已保存的 advantage。两个开关的位置和作用由此区分开来。

## 一批数据用完时，哪些东西已经变化

把本轮开始的 actor 参数记为 `θ0`。四次全局 mini-batch 更新对应：

```text
开始：由 θ0 计算整批 old_log_probs；随后计算整批 advantages

mini-batch 0：current=θ0 → backward → optimizer → θ1
mini-batch 1：current=θ1 → backward → optimizer → θ2
mini-batch 2：current=θ2 → backward → optimizer → θ3
mini-batch 3：current=θ3 → backward → optimizer → θ4

整轮保持：已生成 token、reward、old_log_probs、ref_log_prob、advantages
每次变化：current log_probs、loss、梯度、actor 参数、optimizer state
```

同一个 PPO epoch 中，各 mini-batch 包含不同的 trajectory，但都用 `θ0` 时计算的 old probability 作分母。若开启第二个 epoch，会重新遍历这些 trajectory，current actor 从 `θ4` 继续变化；old probability 和 advantage 仍然不刷新。

在同步路径中，这 512 条数据完成更新后，新的 actor 权重才交给下一轮 rollout。具体由谁接收 RPC、梯度怎样跨两台机器汇总，接着看 [[projects/verl-source-reading/06-single-controller-and-ray|06 Single Controller and Ray]]；权重同步与恢复另见 [[projects/verl-source-reading/08-hybrid-engine-and-weight-sync|08 Hybrid Engine and Weight Sync]]。

## 换成 GAE 时，仍从同一条回答出发

GRPO 的主线已经闭合，可以在同一份输入上比较 PPO+GAE。这需要把 `algorithm.adv_estimator` 改为 `gae`，并把基线显式关闭的 `critic.enable=false` 改为 `true`，或者恢复为 `null` 让 `need_critic` 根据 estimator 决定。只改 estimator 不会覆盖显式关闭 critic 的设置。正确启用后，训练会在 advantage 前算出 `values`，在 actor update 前增加 critic update。

取 `P0_0_0` 的三个有效 response 位置，令 reward 为 `[0,0,1]`，示意 old values 为 `[0.2,0.4,0.6]`，`γ=λ=1`，末尾 bootstrap value 为 0。沿 [`compute_gae_advantage_return`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L215) 逆序计算：

```text
t=2：delta=1+0-0.6=0.4，A2=0.4
t=1：delta=0+0.6-0.4=0.2，A1=0.2+A2=0.6
t=0：delta=0+0.4-0.2=0.2，A0=0.2+A1=0.8

未whiten的 advantages = [0.8,0.6,0.4]
returns = advantages + old_values = [1,1,1]
```

源码先由未 whiten 的 advantage 得到 `returns`，再用 `masked_whiten` 标准化 actor 使用的 advantages；因此写回后的 `advantages` 不能直接与 values 相加，重新推算 returns。padding 和 observation 位置在逆序递推中会跳过，不是把 padding 当成新的环境时间步。

critic 用预测 value 拟合 returns，并通过 clipped value loss 约束偏离 old value 的更新。actor 仍然消费 old/current log-prob、advantages 和 response mask，只是 advantage 的来源从“同题回答的相对得分”变成了“reward 与 value 的时序估计”。这也是这两种训练方式能够复用后半段 worker、model engine 和 policy loss 接口的原因。

## 延伸阅读

- [[training/post-training/grpo|GRPO]]：组内相对 advantage 的算法含义。
- [[training/post-training/ppo|PPO]]：old policy、ratio 和 clipped objective。
- [[projects/verl-source-reading/12-on-policy-distillation-and-mopd|12 On-Policy Distillation and MOPD]]：同一条 sampled trajectory 如何增加 teacher 信号。
- [[projects/verl-source-reading/labs/03-compare-grpo-ppo|Lab 03 Compare GRPO and PPO]]：比较本章两种 advantage 路径实际需要的字段和计算。
