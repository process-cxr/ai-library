---
title: Chat Template
created: 2026-03-01
published: 2026-03-01
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - post-training
  - chat-template
---

Chat Template 是把结构化对话消息转换成模型 token 序列的规则。它定义 system、user、assistant、tool 等角色如何排列，特殊 token 如何插入，assistant 应该从哪里开始生成，以及哪些 token 在训练时参与 loss。对 chat model 来说，chat template 不是展示层格式，而是模型行为边界的一部分。

很多后训练问题表面上像“模型不听话”“工具调用格式不稳定”“多轮对话串角色”，实际根因可能是 template 错配：训练时模型学习了一套角色标记，推理时 serving 系统却使用另一套序列化方式。

## 目标与问题

Base language model 接收的是一串 tokens，并不知道 JSON 里的 `role=user` 或 `role=assistant` 是什么。Chat template 的作用是把高层 conversation schema 映射到模型可学习的文本/特殊 token 序列，使模型能够区分：

- system 指令和普通用户请求；
- 用户输入和 assistant 输出；
- tool call 请求和 tool observation；
- 当前轮应该生成的位置；
- 对话何时结束。

因此，chat template 同时承担三个任务：角色编码、生成边界定义、训练与推理协议。

## 角色结构

常见角色包括：

| Role | 含义 | 训练/推理中的作用 |
|---|---|---|
| `system` | 全局行为约束、身份、政策、输出风格 | 优先级通常高于 user |
| `user` | 用户请求、上下文、输入数据 | 作为模型生成 assistant 回复的条件 |
| `assistant` | 模型应生成的回答 | SFT loss 通常只作用于 assistant tokens |
| `tool` | 外部工具返回结果 | 作为上下文供 assistant 整合 |
| `developer` | 开发者级约束或应用策略 | 一些系统中位于 system 与 user 之间 |

不同模型家族对这些角色的支持不同。某些模型只显式支持 system/user/assistant；某些模型为 tool call 设计了专门 special tokens 或 JSON schema。

## 序列化示例

结构化消息：

```json
[
  {"role": "system", "content": "You are a precise assistant."},
  {"role": "user", "content": "Return a JSON object with name and age."},
  {"role": "assistant", "content": "{\"name\":\"Ada\",\"age\":36}"}
]
```

可能被序列化为：

```text
<bos><|system|>
You are a precise assistant.
<|user|>
Return a JSON object with name and age.
<|assistant|>
{"name":"Ada","age":36}<eos>
```

也可能使用 `[INST]...[/INST]`、`<start_of_turn>`、`<|im_start|>` 等完全不同格式。重要的不是某一种格式更“自然”，而是模型训练时见过哪一种格式。

## 特殊 Token

Chat template 常用 special tokens 表示：

- beginning-of-sequence / end-of-sequence；
- role boundary；
- assistant generation prompt；
- tool call start/end；
- tool result start/end；
- stop tokens；
- message separator。

特殊 token 的语义来自训练。一个 token 只有在 tokenizer 中存在且训练数据持续使用，才会成为稳定控制信号。随意更换特殊 token 或把它们当普通字符串处理，会造成模型无法识别边界。

## Loss Masking

SFT 时通常只在 assistant response 上计算 loss：

```text
<system> ...        mask = 0
<user> ...          mask = 0
<assistant> answer  mask = 1
```

原因是 system/user/tool tokens 是条件，不是模型应学习生成的目标。如果对 user tokens 也计算 loss，模型会被训练去复述或模拟用户；如果 assistant 起始 token mask 错误，模型可能学不到何时开始回答。

对于 tool-use 数据，mask 更复杂：

- assistant 的自然语言和 tool call 参数通常需要训练；
- tool observation 通常不训练，因为它来自外部系统；
- 最终 assistant 总结需要训练；
- 如果包含隐藏 reasoning trace，需要决定 trace 是否作为可见输出学习。

## 训练与推理一致性

Chat template 最重要的工程原则是：**训练、评估、推理、数据生成必须使用同一套语义等价的模板**。不一致常见于：

- 训练数据使用 `<|assistant|>`，推理系统使用 `Assistant:`；
- 训练时有 system prompt，评估时没有；
- SFT 时添加 `<eos>`，推理 stop sequence 没有包含它；
- tool call 训练为 JSON，部署时使用函数调用专用 wrapper；
- tokenizer 版本变更导致 special token id 不一致；
- 多轮对话拼接时遗漏上一轮 assistant 结束标记。

这些问题会显著影响模型行为，却不一定反映在通用 benchmark 上。

## 与后训练方法的关系

- [[training/post-training/sft|SFT]]：chat template 决定输入输出序列和 loss mask。
- [[training/post-training/instruction-tuning|Instruction Tuning]]：instruction 数据需要被 template 包装成模型认识的对话形式。
- [[training/post-training/rlhf|RLHF]]：policy rollout、reward model 输入和 reference model 输入必须使用一致模板。
- [[training/post-training/dpo|DPO]]：chosen/rejected responses 必须在相同 prompt template 下计算 log-prob。
- [[training/post-training/rejection-sampling|Rejection Sampling]]：采样候选时的 template 会影响候选质量和长度。
- Tool-use training：模板就是函数调用协议的一部分。

## 常见失败模式

### Role Leakage

模型输出类似 `<user>`、`System:` 或继续生成下一轮用户消息，通常说明角色边界学习不稳或 stop token 配置错误。

### Prompt Injection 边界弱

如果 system 与 user 在模板中没有清晰区分，或者训练数据没有强化优先级，模型更容易把用户文本中的“忽略前面指令”当作高优先级命令。

### Tool Call 格式漂移

工具调用需要严格 schema。若训练中工具参数格式多样但推理系统只接受一种格式，模型会产生不可解析调用。

### EOS / Stop Token 错误

遗漏 EOS 会导致模型继续生成多轮对话；stop sequence 过宽会截断合法输出；stop sequence 过窄会让模型泄露模板 token。

### 评估模板错配

同一个模型在不同 benchmark harness 中分数差异很大，有时不是能力差异，而是 chat template、few-shot 包装或 generation prompt 不同。

## 实践检查清单

- tokenizer 中 special tokens 是否与模板一致；
- 训练样本是否全部通过同一个 template 函数生成；
- assistant 起始位置和 loss mask 是否正确；
- 多轮数据是否包含清晰轮次边界；
- 推理 serving 是否调用同一个 `apply_chat_template` 逻辑；
- reward model / DPO / RL rollout 是否复用同一 prompt 格式；
- stop tokens 是否覆盖 assistant 结束但不会误伤正文；
- tool call 与 tool result 是否有可解析边界。

## 相关概念

- [[training/post-training/sft|SFT]]
- [[training/post-training/instruction-tuning|Instruction Tuning]]
- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/dpo|DPO]]
