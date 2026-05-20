# application

`application/` 负责整理大模型应用模式、能力编排和评测方法，回答模型“如何被放进真实任务中使用并判断效果”。当前采用五个二级目录：prompting、rag、tool-use、agents、evaluation。

## 当前模块职责

- 整理提示词、RAG、工具调用、Agent 和评测五条应用主线。
- 关注系统设计、交互方式、工具调用、知识注入和效果度量。
- 将底层模型能力与上层产品或研究任务连接起来。

## 直接子项

- `index.md` — Application 网页入口与模块索引。
- `prompting/` — Prompt Engineering、system prompt、few-shot、CoT 和结构化输出。
- `rag/` — 检索增强生成，包括 retrieval、chunking、embedding、reranking、hybrid search 等。
- `tool-use/` — Function Calling、Tool Calling、MCP 和结构化 API。
- `agents/` — Agent、planning、memory、reflection、多 Agent 和 workflow agent。
- `evaluation/` — Benchmark、LLM-as-a-Judge、人评、线上评测和幻觉评测。

## 文件契约

- 二级目录的 `index.md` 负责该应用模块入口和直接子项索引。
- 具体应用笔记可以先以 TODO 占位，后续再按知识写作规范扩展。
- 每篇笔记说明应用目标、输入输出、关键组件、评测方式和失败边界。
- 涉及模型能力来源时，应链接到底层架构、训练或推理笔记。

## 边界

- 数学基础放入 `../fundamentals/`。
- 模型结构放入 `../architecture/`。
- 训练方法放入 `../training/`。
- 推理部署和性能优化放入 `../inference/`。

## Direct-Call 信息

- 目录入口链接：`[[application/|Application]]`。
- 典型入口：`[[application/rag/|RAG]]`、`[[application/agents/|Agents]]`、`[[application/evaluation/|Evaluation]]`。
- 典型外部调用：读者从 `../index.md` 或搜索入口进入具体应用主题。
