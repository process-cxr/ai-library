---
title: 03 From Prompts to Trajectories
created: 2026-09-02
published: 2026-09-02
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - dataflow
  - rollout
---

上一章结束时，trainer 已经取出 128 道题，为它们分配 `uid`，并把整批 prompt 交给 AgentLoopManager。TransferQueue 中的 `P0` 还只是一个标记为 `pending` 的 prompt key，没有回答，也没有训练用的 reward。

这一章让 P0 真正生成回答。我们依次进入接收 prompt 的 Ray worker、四个独立 session、负责 tokenization 的 AgentLoop，再跟着一次请求走到 vLLM，最后拿回 `AgentLoopOutput`。到章末，P0 已经有四条长度不同的回答；下一章再给它们评分并组成训练批次。

继续使用 [[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]] 中的固定案例：两台机器，每台 4 张 GPU；rollout 使用 vLLM，`TP=2、DP=1、PP=1`，共 4 个生成副本；actor 使用 FSDP，训练 DP 为 8。`train_batch_size=128`、`rollout.n=4`。这里的调用和字段来自固定源码 revision；回答内容与小矩阵是教学构造，没有进行这套多机训练。

## P0 到达生成侧时，还保留着一道完整的题

把其中一个真实 UUID 简写为 `P0`，它携带的主要信息可以写成下面这样。题目是教学自造的，`data_source` 选择 `openai/gsm8k`，是为了使用仓库内的 GSM8K 规则评分器。

```python
{
    "uid": "P0",
    "raw_prompt": [{
        "role": "user",
        "content": "盒中有3袋铅笔，每袋4支，一共多少支？请把最终答案写在 #### 后面。",
    }],
    "data_source": "openai/gsm8k",
    "reward_model": {"style": "rule", "ground_truth": "12"},
    # 另有 index、global_steps、extra_info 等字段
}
```

`raw_prompt` 是一组 chat messages。模型不能直接计算这些 Python 字典，后面还需要应用 chat template 并编码成 token。`ground_truth` 则留给下一章的评分器；SingleTurnAgentLoop 构造模型输入时读取的是 `raw_prompt`，不会把整份样本字典拼进 prompt。因此把答案保存在 `reward_model` 中，不等于把答案提供给模型。

现在先区分两个容易都被称为 worker 的对象。**AgentLoopWorker 调度会话，rollout replica 执行模型推理。** 前者会处理 messages、调用服务和接收结果；后者持有 vLLM 模型并在 GPU 上运行。AgentLoop worker 的数量不等于 GPU 数，也不等于模型副本数。

[`AgentLoopManagerTQ.generate_sequences`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L243) 首先按 AgentLoop worker 数量切分 prompt batch：

```python
chunkes = prompts.chunk(len(self.agent_loop_workers))
ray.get([
    worker.generate_sequences.remote(chunk)
    for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
])
```

保持当前配置的 8 个 AgentLoop worker 时，128 道题分成 8 份，每份 16 道。这里切的是 **prompt 数**，还没有复制成 512 条回答。假设 P0 落到第一个 worker，它随后由该 worker 的事件循环处理；不能由此推断 P0 的模型 forward 就运行在 GPU 0 上。

这一处 `ray.get` 等待远端 `generate_sequences` 返回。远端方法的工作是创建后台任务，因此它返回时，实际回答可能仍在生成。同步训练中的“同步”，体现在后面必须收齐本批可采样 group 才能更新，并不要求每个提交函数都一直阻塞到生成结束。

## `rollout.n=4` 在哪里变成四次独立生成

[`AgentLoopWorkerTQ.generate_sequences`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L59) 从 rollout 配置取出 temperature、top-p、top-k 等采样参数。如果输入没有 `agent_name`，它使用默认的 `single_turn_agent`，随后逐行把 TensorDict 中的张量和非张量字段还原为 prompt 字典，为每一道题创建 `_run_prompt` 后台任务。

P0 的任务开始后，先把 TQ 中 `P0` 的状态更新为 `running`。再看 [`_run_prompt`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L107) 的关键部分：

```python
n = prompt.pop(
    "__rollout_n__",
    config.n if not trajectory["validate"] else config.val_kwargs.n,
)

for i in range(n):
    task = asyncio.create_task(
        self._run_agent_loop(
            run_sampling_params,
            trajectory=trajectory,
            trace=trace,
            session_id=i,
            **prompt,
        )
    )
    tasks.append(task)
```

主线没有逐样本覆盖 `__rollout_n__`，也不是 validation，所以 `n=4`。循环创建四个 session，它们都拿到 `uid=P0` 和相同的题目，却拥有不同的 `session_id`：

| 本次调用  | 共享的信息                    | 独立的信息                 | 单输出 session 结束后使用的 key |
| --------- | ----------------------------- | -------------------------- | ------------------------------- |
| session 0 | P0、题目、标准答案、训练 step | 生成请求、token 序列、耗时 | `P0_0_0`                        |
| session 1 | 同上                          | 同上                       | `P0_1_0`                        |
| session 2 | 同上                          | 同上                       | `P0_2_0`                        |
| session 3 | 同上                          | 同上                       | `P0_3_0`                        |

最后一个 `_0` 是 session 内 output 的编号。SingleTurnAgentLoop 返回一个 output，所以本例总是 0；它不表示 GPU rank 或 mini-batch 编号。此时这些 trajectory key 尚未写回，表中列的是下一章后处理将使用的名字。

由此可见，四次采样由 AgentLoop 外层显式创建。不能把这一行配置理解成“给一个 vLLM request 设置 `n=4`，返回四条候选”。这里是四个 session 各自调用一次生成服务，四条回答也不要求互不相同。

## session 0 创建自己的 SingleTurnAgentLoop

接下来只跟踪 session 0，其他三个 session 独立执行同样的代码。

[`AgentLoopWorker._run_agent_loop`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L631) 根据 `agent_name` 查找注册项，实例化对应的 AgentLoop，并调用：

```python
output = await agent_loop.run(sampling_params, **kwargs)
return await self._agent_loop_postprocess(
    output, trajectory["validate"], **kwargs
)
```

这里有两个阅读源码时很有用的细节。首先，每个 session 获得自己的 AgentLoop 对象，所以它的会话状态不会和 P0 的另外三次采样混在一起。其次，代码虽然写在基类中，当前 `self` 是 `AgentLoopWorkerTQ`，因此最后调用的是 V1 的 TQ 后处理实现。基类里另一份把全部输出 padding 成大 batch 的后处理，不是本篇的主路径。

实例化参数中的 `server_manager=self.llm_client` 也容易误导。SingleTurnAgentLoop 后面调用的 `self.server_manager.generate`，在这里实际指向 `LLMServerClient`。这个对象负责选择推理 server，自己不执行 Transformer forward。

## messages 怎样变成模型真正读入的 token

[`SingleTurnAgentLoop.run`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/single_turn_agent_loop.py#L38) 先取 `messages = list(kwargs["raw_prompt"])`。当前样本只有文本，多模态提取不会为它增加图片或音频输入，然后进入：

```python
prompt_ids = await self.ct_build_initial_tokens(
    messages,
    images=images,
    videos=videos,
    audios=audios,
)
```

`ct_build_initial_tokens` 委托 Continuous Token builder 应用 chat template。文本 builder 的底层操作包含 `tokenize=True` 和 `add_generation_prompt=True`：题目前后的 role 标记、模型要求的对话分隔符、等待 assistant 续写的前缀都会影响最终 token 序列。实际 token 数不能用题目的字数直接估算。

为了把后面的矩阵写得可手算，全套文章把 P0 的 prompt **缩尺画成 6 个 token**：

```text
messages                   角色与文本组成的 Python 对象
    ↓ chat template + tokenizer
prompt_ids                 [p0, p1, p2, p3, p4, p5]
逻辑长度                    6
此处的类型                  list[int]，还不是 GPU 上的 [batch, length] 张量
```

`p0 … p5` 是位置示意，既不是实测 token id，也不表示一个 token 对应一个字。真实运行保留 `prompt_length=1024` 的上限，模型与 tokenizer 仍是固定案例的 Qwen3-4B。小图只缩小画出的有效长度，不修改全套训练配置。

[`ct_build_initial_tokens`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L315) 还会检查最终长度：纯文本超过上限时保留末尾的 `prompt_length` 个 token；带图像、视频或音频的过长 prompt 会报错，避免直接切 token 破坏特征对齐。此前 dataset 的过滤与这里的检查发生在不同位置，不能因为 dataset 已过滤，就忽略 AgentLoop 最终 template 产生的长度。

本例不触发截断。session 0 带着完整的六位置示意 prompt 继续请求生成。

## 一次生成请求怎样抵达两张 GPU

SingleTurnAgentLoop 的下一段调用为：

```python
output = await self.server_manager.generate(
    request_id=request_id,
    prompt_ids=prompt_ids,
    sampling_params=sampling_params,
    # 纯文本案例的多模态参数为空
)
```

普通采样时 `request_id` 是新生成的 UUID。它用于服务路由，与 `uid=P0`、`session_id=0` 承担不同用途：P0 标识训练问题，session 标识该问题的一次尝试，request id 标识服务请求。不要把它们当作同一个字段的不同写法。

[`LLMServerClient.generate`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/llm_server.py#L98) 做的核心工作是申请 server、发起 Ray RPC，并在 `finally` 中释放负载计数：

```python
server_id, server = await self._acquire_server(
    request_id,
    prompt_ids=prompt_ids,
    sampling_params=sampling_params,
    # 省略纯文本案例不需要的附加参数
)
try:
    output = await server.generate.remote(
        request_id=self._vllm_request_id(request_id),
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        # 省略附加参数
    )
    return output
finally:
    self._release_server(server_id, request_id=request_id)
```

这是省略附加参数的调用节选。实际函数还会把 server 返回的权重版本补成 `min_global_steps` 和 `max_global_steps`，随输出保留。关闭 full determinism 时，发给 vLLM 的 request id 还会重新生成；用于 sticky routing 的 id 和引擎请求 id 不必相同。

现在再看请求可选的四个副本。它们在初始化阶段已经建好；根据 [`_initialize_llm_servers`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/llm_server.py#L399)，关闭 prefill/decode 分离时：

```text
每个 rollout replica 所需 GPU = TP × DP × PP = 2 × 1 × 1 = 2
rollout replica 数量           = 8 / 2 = 4
```

一种与此配置相符的布局是 node 0 的 GPU 0–1、2–3 各组成一个副本，node 1 同样组成两个副本。两张 TP GPU 共同计算同一个副本处理的请求；P0 的一条回答不会因此变成两条训练样本。实际 rank 到物理设备的映射还取决于资源放置，不能把示意布局当作运行日志。

负载均衡器在副本之间选择 server；服务内部再安排请求的执行。`rollout.n=4` 恰好等于副本数，只是这个例子的数值巧合，不保证 P0 的四个 session 各落一个副本。它们还会与其余 127 道题的请求交错，vLLM 的运行 batch 也不等同于训练时的 PPO mini-batch。

## vLLM 返回 token，而不是一整块训练矩阵

在 [`vLLMAsyncServer.generate`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L585) 中，prompt token 被装进 `TokensPrompt`，采样配置被装进 `SamplingParams`，然后交给引擎。server 依据剩余上下文空间限制 `max_tokens`；本例 response 上限为 1024，但上限不是实际生成长度。

假设 session 0 返回文本“`3 × 4 = 12。#### 12`”。继续采用缩尺示意，把其生成 token 画成三个位置：

```text
TokenOutput.token_ids     [a0, a1, a2]            长度 3
TokenOutput.log_probs     [ℓ0, ℓ1, ℓ2]            若开启，长度同为 3
TokenOutput.extra_fields  包含生成时的 global_steps 等信息
```

`ℓj` 是生成服务对实际采样 token 给出的 log-prob。它不会因为这个 session 的答案正确就变大，也不是 reward。后面训练端还会重算 `old_log_probs`；这两个概率来源的区别留到 05 展开。

这份输出还没有 `input_ids` 的训练 TensorDict，没有 `advantages`，也没有评分结果。它先回到 SingleTurnAgentLoop，完成 token 与 mask 的对齐。

## 把 token 与 mask 一起接回 session

SingleTurnAgentLoop 调用 `ct_merge_assistant_token`，传入原 prompt、刚生成的 token，以及初始为空的 response mask。对当前纯文本单轮路径，builder 中的合并就是：

```python
merged_token_ids = list(runtime_token_ids) + list(assistant_token_ids)
```

同时 [`align_response_metadata`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/tokenizer/continuous_token.py#L333) 为这些 assistant token 追加 mask 1，并按同样的顺序追加 log-probs。把这几步一起画出，才能看清字段怎样对齐：

```text
合并前 runtime tokens     p0 p1 p2 p3 p4 p5                 长度 6
新生成 assistant tokens                     a0 a1 a2       长度 3
合并后 runtime tokens     p0 p1 p2 p3 p4 p5 a0 a1 a2       长度 9

response_mask                                1  1  1       长度 3
response_logprobs                           ℓ0 ℓ1 ℓ2       长度 3
```

mask 没有为六个 prompt token 分配位置，它只覆盖 response 区域。三个 1 表示三个位置都是模型生成的 action。稍后补齐矩阵时产生的 padding 不应参与 loss；在工具场景，环境返回的 observation 也不应作为模型采样的 action。两者都需要 mask，但原因不同。

Continuous Token 在本例看起来只是拼接，仍然有必要从这里认识它：后续工具调用会在已经生成的 token 后追加非 assistant 内容，有些模型还需要补边界 token。如果重新把整段对话 decode 再 encode，既有 token 可能变化，已记录的采样概率就会失去原来的对应位置。工具路径怎样保持对齐，见 [[projects/verl-source-reading/11-agentic-rl-extension|11 Agentic RL Extension]]。

## 返回的 `AgentLoopOutput` 已经可以描述一条回答

合并后，代码按 response mask 的长度重新切开 prompt 与 response：

```python
response_ids = merge_result.token_ids[-len(response_mask):] if response_mask else []
prompt_ids = merge_result.token_ids[:len(merge_result.token_ids) - len(response_mask)]
```

在本例中是从长度 9 的序列中取出最后 3 个位置，得到 `[a0,a1,a2]`；前 6 个位置恢复为 prompt。随后创建 `AgentLoopOutput`，将 response、mask 和可选 log-probs 同步裁剪到 `response_length` 上限：

| session 0 的输出字段 | 缩尺示意内容         | 下一步为何需要                             |
| -------------------- | -------------------- | ------------------------------------------ |
| `prompt_ids`         | `[p0,…,p5]`          | 与回答拼成训练模型输入                     |
| `response_ids`       | `[a0,a1,a2]`         | decode 评分，也作为预测目标                |
| `response_mask`      | `[1,1,1]`            | 指出哪些 response 位置参与 policy loss     |
| `response_logprobs`  | `[ℓ0,ℓ1,ℓ2]`，若开启 | 保留 rollout 概率，与训练端计算比较        |
| `reward_score`       | 尚未计算             | 下一章由规则函数产生                       |
| `num_turns`          | `2`                  | 单轮路径固定记为一次 user 与一次 assistant |
| `extra_fields`       | 版本与生成统计等     | 后续状态检查和指标计算                     |

这里的字段主体仍是 Python list。不能在这个阶段写成“全局 response tensor 形状 `[512,1024]`”：其他 session 可能还没生成完，每条 response 长度也不同，而且 V1 不要求先把全局回答补齐后才写入 TQ。

把视野拉回 P0 的四个 session，教学中使用下面四条回答。长度列继续是人工缩尺，**不对应这些文本的实际 tokenizer 结果**：

| 后续 trajectory key | 回答文本                  | 示意 response 长度 |
| ------------------- | ------------------------- | -----------------: |
| `P0_0_0`            | `3 × 4 = 12。#### 12`     |                  3 |
| `P0_1_0`            | `3 + 4 = 7。#### 7`       |                  5 |
| `P0_2_0`            | `4 + 4 + 4 = 12。#### 12` |                  4 |
| `P0_3_0`            | `4 × 4 = 16。#### 16`     |                  2 |

每条回答返回后，会立即进入自己那次 `_agent_loop_postprocess`；它不需要等 P0 的四条回答都返回才开始评分。外层 `_run_prompt` 仍然等待全部 session 完成，包括各自的后处理和写回。

[[projects/verl-source-reading/04-rollout-reward-and-agent-loop|04 Reward and Training Batch]] 从这个精确位置接续：`AgentLoopWorkerTQ._agent_loop_postprocess` 已经收到 session 0 的 output，正要执行 `await self._compute_score(outputs, kwargs=kwargs)`。那段文字回答会先变成分数 `1`，再变成 `[0,0,1]` 的 token reward，最终连同原始 `uid=P0` 写入 `P0_0_0`。
