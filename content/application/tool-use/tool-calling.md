---
title: Tool Calling
created: 2026-05-10
published: 2026-05-10
modified: 2026-06-30
type: topic
status: growing
area: application
tags:
  - tool-use
  - tool-calling
---

Tool Calling 指模型在对话或 agent 过程中选择外部工具、生成符合 schema 的参数、接收执行结果，并把结果纳入后续推理和响应的能力。它是 [[application/agents/agent|Agent]]、RAG、代码助手、浏览器代理和自动化工作流的基础机制。

在最简单的 function calling 场景中，模型只需要根据用户请求选择函数并生成 JSON 参数。但在真实 agent 系统中，tool calling 往往更复杂：工具定义可能出现在 system prompt、MCP schema、IDE scaffold 或自定义 XML/JSON wrapper 中；工具结果可能是文件内容、命令输出、网页 DOM、截图分析、数据库查询结果或错误日志；模型还需要在多轮 action-observation loop 中决定是否继续调用工具、修改参数、恢复失败或结束任务。

## 核心组成

| 组成 | 说明 |
|---|---|
| Tool definition | 工具名称、用途、参数 schema、约束和可用范围 |
| Tool selection | 判断当前任务是否需要工具，以及选择哪个工具 |
| Argument generation | 生成类型正确、语义 grounded、可执行的参数 |
| Execution feedback | 接收工具返回、错误、状态变化或环境 observation |
| State integration | 将工具结果纳入后续规划、回答或下一步 action |
| Format adherence | 严格遵循当前 scaffold 要求的调用格式 |

工具调用不是单纯的结构化输出任务。格式合法只说明模型能生成可解析的 call；真正有效的 tool calling 还要求工具选择合理、参数与上下文一致、执行结果被正确解释，并能根据失败反馈调整策略。

## Tool Template Diversity

不同系统对工具调用的表示差异很大。常见形式包括 JSON function call、XML tag、Python-style call、TypeScript interface、自然语言工具说明，以及 IDE / CLI agent 自定义的 tool response wrapper。即使底层工具语义相同，模型也必须适应不同的 tool definition、invocation syntax 和 response boundary。

只在单一工具模板上训练，模型容易过拟合表层格式：在熟悉 scaffold 中表现良好，但迁移到另一个 IDE、CLI 或 agent runtime 时出现 malformed call、参数转义错误、response 边界混淆或不必要的重试。因此，tool-use 数据设计应把模板多样性作为训练变量，而不是只把工具调用看成固定 JSON schema。

对 code agent 尤其如此。代码片段、patch、shell command 和长字符串参数常常包含换行、引号和特殊字符，JSON 格式会带来较重 escaping 负担。XML-style 或其他长字符串友好的格式可能降低转义复杂度，但也需要模型学习新的边界规则。[[sources/papers/2026-qwen3-coder-next-technical-report|Qwen3-Coder-Next]] 的工具模板实验说明，在数据量和训练配置不变时，增加 tool chat templates 可以提升跨 scaffold 的工具格式鲁棒性。

## 评测

Tool calling 至少需要分开评估：

- 工具选择是否正确；
- 参数 schema 是否合法；
- 参数是否由当前上下文支持；
- 工具执行是否成功；
- 模型是否正确使用 tool result；
- 失败后是否能恢复；
- 未见模板下的 format adherence；
- 多轮任务中的最终成功率。

只看 JSON validity 或 schema validity 会高估能力。对 agent 任务，更应同时观察有效工具调用率、无效调用重试率、环境反馈利用率、episode 成功率和平均交互轮数。

## 相关概念

- [[application/agents/agent|Agent]]
- [[application/agents/planning|Planning]]
- [[application/evaluation/benchmark|Benchmark]]
- [[training/post-training/sft|SFT]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
