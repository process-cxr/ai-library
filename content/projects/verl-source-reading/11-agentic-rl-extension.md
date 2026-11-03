---
title: 11 Agentic RL Extension
created: 2026-09-02
published: 2026-09-02
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - agentic-rl
  - tool-use
---

[[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 From Prompts to Trajectories]] 的单轮 session 只调用一次模型；模型返回回答，session 就可以交给评分器。现在保持同一道 P0 铅笔题，把其中一个 session 改为允许调用计算器：模型先生成工具调用，工具返回 `12`，模型再生成“`#### 12`”。这一次尝试仍然属于 P0，同一批问题仍然可以每题采样 4 次，但一次尝试中已经包含两次模型生成。

本篇解释这些 turn 怎样接成一条训练 trajectory。沿用固定 revision 的 `ToolAgentLoop`，并假设注册了一个返回文本结果的计算器工具；工具名称与返回内容是教学示意，不表示仓库默认附带该函数，也没有实际运行工具实验。2 节点 × 4 GPU、rollout TP=2、actor DP=8 的资源基线可以保留，但需要切换 `agent_name`、配置工具和对应 parser，才能进入本篇路径。

## 一个 session 内增加了什么状态

AgentLoopWorker 仍然调用 `_run_agent_loop`，注册名从 `single_turn_agent` 变为 `tool_agent`。外层的 `uid=P0`、`session_id=0`、最终 key `P0_0_0` 都沿用原协议；变化发生在 `ToolAgentLoop.run` 内部。

[`AgentData`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/tool_agent_loop.py#L55) 保存 messages、当前 token 序列、response mask、log-probs、工具调用、turn 计数以及附加状态。其中 `prompt_ids` 这个名字需要特别留意：循环开始时它是初始 prompt，之后会逐步增长为**下一次模型请求使用的全部上下文**。因此它后面会包含先前 assistant 和 tool 的 token，而不是永远只包含原始题目。

可以先在文本层看一次完整交互：

```text
user        盒中有3袋铅笔，每袋4支，一共多少支？
assistant   调用 calculator，参数 {"expression": "3 * 4"}
tool        12
assistant   3 × 4 = 12。#### 12
```

两段 assistant 都是 policy action；工具消息是环境 observation。下次模型生成时它们都属于上下文，训练时却不能把所有位置都当成 policy 采样出来的目标。

ToolAgentLoop 在一个 session 中保持同一个 request id。`LLMServerClient` 可以据此做 sticky routing，使多个 turn 尽量复用同一 server 的前缀缓存；关闭 full determinism 时，它发给 vLLM 的每个 turn 又有新的引擎请求 id。session 连续性与单次引擎请求的身份是两层概念。

## 沿 P0 的计算器调用走一遍状态机

[`ToolAgentLoop.run`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/tool_agent_loop.py#L127) 使用四种状态。对刚才的具体交互，经过的顺序是：

```text
PENDING
  以题目和 tool schema 构建初始 tokens
     ↓
GENERATING
  模型生成 calculator 调用
     ↓
PROCESSING_TOOLS
  解析参数，执行计算器，把结果接到上下文
     ↓
GENERATING
  模型基于工具结果生成最终答案，没有新 tool call
     ↓
TERMINATED
  把累计的 token、mask、log-probs 切成 AgentLoopOutput
```

`PENDING` 调用 `ct_build_initial_tokens`，除了原始 messages，还传入 tool schemas。因此这个变体的 prompt 长度通常不同于单轮案例；不能把单轮的六 token 缩尺图当作加入工具后的真实长度。样本可通过 `extra_info.tool_selection` 限制可见工具，实际 parser 和工具查找也使用筛选后的工具集合。

`GENERATING` 首先将 parser 要求的 stop token 合并进采样参数，再把当前全部 `prompt_ids` 交给 server。返回 token 后，`assistant_turns` 加一，Continuous Token builder 将新 token 与 mask、log-probs 一起接到累计序列中。

随后代码检查 response 总长度、assistant turn 上限与 user turn 上限，未终止才解析 tool calls。解析对象是**本轮生成的 token**，而不是已经拼接好的全部历史。如果找到工具调用，状态进入 `PROCESSING_TOOLS`；没有则终止。具体顺序见 [`_handle_generating_state`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/tool_agent_loop.py#L227)。

因此“输出了工具格式”并不无条件保证执行工具：达到长度或 turn 上限时，函数可能先返回 `TERMINATED`。阅读训练样本时，要区分模型产生的调用文本与工具确实执行过的 observation。

## 工具怎样执行，返回值怎样进入上下文

[`_handle_processing_tools_state`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/tool_agent_loop.py#L302) 从当前 tool calls 中取前 `max_parallel_calls` 个，为每个创建调用任务：

```python
tasks = []
for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
    tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))

responses = await asyncio.gather(*tasks)
```

这里是截取并发处理的调用数量，不是让超出数量的调用排队以后再执行。返回结果按调用列表对应回各个 tool call，并转换成 `role="tool"` 的消息；存在 `tool_call_id` 时也保留，方便与 assistant 的调用关联。

实际执行位于 [`_call_tool`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/tool_agent_loop.py#L435)。先检查工具名，再用 `json.loads` 解析参数。随后分为两种契约：

| 工具类型        | 一次调用的主要步骤                                                                     | 状态的去向                                                          |
| --------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `FunctionTool`  | `tool.call(tool_args)`，规范化返回值                                                   | 不执行 create/release；不通过 `tools_kwargs` 注入逐轨迹生命周期状态 |
| `BaseTool` 子类 | `create(...) → execute(instance_id, tool_args, agent_data=...) → release(instance_id)` | create 返回实例标识；execute 可以读取 session 的 AgentData          |

计算器的返回文本 `12` 随后成为 tool message，通过 `ct_merge_non_assistant_msg` 接到 runtime tokens。更新成功后，代码增加 `user_turns` 并返回 `GENERATING`。这里的 `user_turns` 是实现中的交互轮次计数，工具反馈也会使它增加；不能按名字理解为必须有真人又发送了一条 user 消息。

模型第二次看到的输入已经包含题目、自己之前的调用和工具结果。它生成最终答案后，没有新 tool call，session 结束。完成这一系列动作通常仍返回一个 `AgentLoopOutput`，并不会因为有两个 assistant turn 自动产生两条 TQ trajectory。

## token、mask 与 log-prob 必须同步增长

为了看清实际对齐，单独构造一段缩尺序列。假设初始 prompt 长度仍画成 6，第一个 assistant 段占 3 个位置，工具及其模板边界占 2 个位置，最终 assistant 段占 3 个位置。这些都是位置示意，不是计算器 schema 的真实 tokenization。

```text
阶段                    完整输入长度     response_mask
初始 prompt                 6           []
追加 assistant 调用         9           [1,1,1]
追加 tool observation      11           [1,1,1,0,0]
追加 assistant 最终答案     14           [1,1,1,0,0,1,1,1]
```

展开最终字段：

```text
input_ids     p0 p1 p2 p3 p4 p5 | a0 a1 a2 | o0 o1 | b0 b1 b2
responses                        a0 a1 a2 | o0 o1 | b0 b1 b2
response_mask                     1  1  1 |  0  0 |  1  1  1
rollout log-probs                 ℓa0 ℓa1 ℓa2 | 0 0 | ℓb0 ℓb1 ℓb2
```

`responses` 长度为 8，包括工具反馈；`response_mask.sum()` 为 6，只统计 policy 生成位置。完整序列中这 14 个 token 都是真实上下文，attention mask 可以全为 1。工具位置 log-prob 中的 0 是对齐占位值，不表示模型真的以概率 1 生成了 observation。

如果最终 task reward 为 1，当前 V1 的 `AgentLoopOutput.as_dict` 会把它写在 response 最后一项，形成 `[0,0,0,0,0,0,0,1]`。GRPO 再按 uid 比较不同 session 的最终分数，并把组内 advantage 乘到 response mask 上。假设本条得到正 advantage `a`，最终是：

```text
advantages = [a, a, a, 0, 0, a, a, a]
```

这意味着先前的工具调用与最后的答案都得到同一条 session 的相对好坏信号。普通 outcome GRPO 没有因为出现多个 turn，就自动判断哪一步工具调用更重要。若需要逐步 credit assignment，需要对应的 reward/advantage 设计，不能仅靠新增一个 `tool_rewards` 字段实现。

工具位置 mask 为 0，表示这些位置不直接作为 policy loss 的目标；它们仍然影响后续 token 的条件概率。不能把“没有针对 observation 的直接 loss 项”误写成“observation 对训练没有影响”。

## 为什么需要 Continuous Token，而不直接重新编码 messages

工具返回后，messages 会多出新消息。最直觉的做法是把完整 messages 重新应用 chat template，获得新的全量 token；问题在于 chat template 的边界、assistant 内容格式化和重新 tokenization 可能改变前面已经采样的 token。先前记录的 log-prob 对应的是旧 token 序列，不能直接套到修改过的序列上。

Continuous Token 保留运行时已经生成的 token，仅编码新增的非 assistant 消息，再通过 `MergeResult` 描述必要的边界修改。[`align_response_metadata`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/tokenizer/continuous_token.py#L333) 根据同一份结果同步改 mask 和 log-probs：

```text
删除 prefix 末尾 k 个 token  → 同时删去末尾 k 个 response 元数据位置
插入 builder 边界 token     → mask 追加 0，log-prob 追加 0
追加 assistant token        → mask 追加 1，追加对应生成 log-prob
追加 tool/user/system token → mask 追加 0，log-prob 追加 0
```

这份快照对 Qwen 有一个具体实现：模板在一个 turn 后渲染 `<|im_end|>\n`，但生成可能停在 `<|im_end|>`。当后面需要接 tool 消息时，[`QwenContinuousTokenBuilder`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/tokenizer/continuous_token.py#L470) 补上缺少的换行 token。该换行由 builder 插入，mask 为 0；它不应伪装成模型采样出的 token。

所以前面缩尺图中的 observation 区域，不一定只是工具文本 `12` 的 token。它还可能包含 tool role 标记、分隔符、assistant generation prefix 以及 builder 插入的边界。mask 的依据是实际 token 构建过程，不能只对自然语言的“工具内容”几个字置零。

## `tool_reward` 与最终训练 reward 之间隔着评分器

工具执行可以返回 `tool_reward`。ToolAgentLoop 将非空值追加到 `agent_data.tool_rewards`，最终放入 `output.extra_fields`；这一步没有把它直接写成 `output.reward_score`。

沿 [[projects/verl-source-reading/04-rollout-reward-and-agent-loop|04 Reward and Training Batch]] 的调用继续，NaiveRewardManager 会把 `tool_extra_fields` 合并进 reward 函数可见的 extra info。自定义 reward function 因而可以读取 tool rewards，选择求和、惩罚无效调用，或结合最终正确率评分。

但是默认 GSM8K 规则函数主要解析最终答案字符串，不会因为工具返回 `tool_reward=1` 就额外加一分。一个工具可以成功返回 `12`，模型却在最终答案写成 `16`；也可以工具执行失败，模型仍然自行解出正确答案。工具操作是否成功与任务回答是否正确是两个信号，怎样组合必须由实际评分函数决定。

此外，默认 naive 路径 decode 的 response 容器会包含工具相关 token。对复杂工具任务，评分器应明确自己检查的是哪个最终产物，避免把 observation 中恰好出现的答案标记当成最终答案。当前源码接口提供了定制能力，具体任务仍需验证评分行为。

## 长度上限具体在哪一步生效

多轮轨迹的长度来自 assistant 与 observation 的累计，因此限制也分布在几个位置。它们不能笼统地归结为一个 `max_tokens`：

| 限制位置          | 源码处理                                             | 对样本的含义                                         |
| ----------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| 初始 prompt 超限  | 纯文本左截断；不允许直接切坏多模态对齐               | 可能丢失问题开头，应检查实际模板后的输入             |
| vLLM 单次请求     | 根据剩余上下文和 response 配置限制新生成长度         | 每个 turn 的生成预算会随累计上下文变化               |
| assistant 合并后  | response mask 长度、assistant/user turn 上限检查     | 可能在解析和执行工具前终止                           |
| tool 文本过长     | `_call_tool` 按 `len(text)` 截断                     | 此实现按字符数切文本，不是 tokenizer token 数        |
| tool token 合并后 | 若新 mask 长度达到 response 上限，直接终止           | 该次 candidate token 合并结果不会赋回 runtime tokens |
| 最终 output       | response、mask、log-probs 同步切到 `response_length` | 返回训练字段长度一致，但不保证语义回答完整           |

工具文本截断支持保留尾部、保留头部或保留两端，中间插入 truncation 提示。由于额外加了提示文本，最终字符数也不必严格等于配置数值。实现见 [`_call_tool` 的截断段](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/tool_agent_loop.py#L490)。

tool 合并后的检查尤其值得看原代码：

```python
merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(...)
if len(response_mask) >= self.response_length:
    return AgentState.TERMINATED
agent_data.prompt_ids = merge_result.token_ids
agent_data.response_mask = response_mask
```

检查发生在赋回 `agent_data.prompt_ids` 之前。因此如果观察文本把长度推到上限，工具可能已经执行，messages 也已追加，但最终 runtime token 输出仍停在先前状态。不能笼统写成“工具输出超长后总会截断存入 trajectory”。训练信号最终依赖实际返回的 tokens，而不是 Python messages 的预期内容。

多模态工具还有额外约束：图像结果需要兼容 processor 和 VL builder；这个 handler 对工具返回视频显式抛出 `NotImplementedError`。支持多模态输入不代表每一种工具输出都已支持。

## 工具报错怎样影响整条 trajectory

`_call_tool` 对未知工具名、JSON 解析失败和多数执行异常，会返回一条包含错误信息的 `ToolResponse`，并给出 `tool_reward=0.0`。这样错误通常成为下一轮模型可以读到的 observation，session 仍可能继续生成。因此“工具失败”不能直接等同于外层 prompt 的 `status="failure"`。

外层 failure 表示 session 任务有异常传播出来。比如未被局部捕获的处理错误或释放资源阶段异常，可能使 `_run_prompt` 最终标记失败；它会先等同组其他 session settle，之后才发布 group 终态。失败组是否进入训练、是否补采，沿用 04 中的 ReplayBuffer 策略。

这段通用 `_call_tool` 没有为所有工具统一包一个 `asyncio.wait_for` 超时。具体工具实现可能自行控制超时，但不能仅看到异步调用就宣称系统具备统一 timeout 和 retry 机制。出现等待过久时，应检查具体工具与环境的契约。

## validation 与训练如何共享这条路径

validation 仍可以使用同一个 ToolAgentLoop，但外层采样参数改用 `rollout.val_kwargs`，采样次数取 validation 的 `n`，结果写入 `val` partition。单个 session 中 tool schema、token 拼接和 mask 语义保持相同，消费方式却是评估指标汇总，不会因为经过同样的 AgentLoop 就自动参与 actor update。

因此判断一条轨迹的用途，需要同时看 `validate`、partition、调用者与后续消费路径；不能只看它包含 `response_mask` 或 `rm_scores`。源码中的这条数据契约同时服务训练和验证，而训练循环决定何时计算 advantage 和反向传播。

到这里，工具扩展与前面单轮路径重新汇合：最终 `AgentLoopOutput` 经评分与后处理写入原 trajectory key。多轮新增了环境交互和 token 来源的区别；TQ 按 key 保存字段、GRPO 按 uid 组织 group、actor 按 mask 选择直接训练目标的逻辑仍然可以沿用。真正运行时，应在 [[projects/verl-source-reading/labs/index|Labs]] 中记录最终 tokens 与 masks，以核对模板边界、截断和工具异常是否符合具体任务预期。
