# inference

`inference/` 负责整理大模型推理优化、部署性能和模型压缩相关知识，回答模型“如何更快、更省、更稳定地服务请求”。当前采用二级目录结构，按推理链路和工程主题组织。

## 当前模块职责

- 整理解码、KV Cache、Attention 加速、量化、Serving、压缩和性能评测。
- 连接模型架构与实际部署指标，例如延迟、吞吐、显存占用和成本。
- 关注推理时行为，不展开训练算法本身。

## 直接子项

- `index.md` — Inference 网页入口与模块索引。
- `decoding/` — 自回归解码、采样、beam search 和投机解码。
- `kv-cache-and-memory/` — KV Cache、PagedAttention、prefix cache、cache eviction 和显存碎片。
- `attention-acceleration/` — FlashAttention、FlashDecoding 和 attention kernel 优化。
- `quantization/` — 量化，包括 weight-only、AWQ、GPTQ、FP8、KV Cache 量化。
- `serving-systems/` — Serving 系统，包括 vLLM、continuous batching、请求调度和 PD 分离。
- `compression/` — 模型压缩，包括剪枝、蒸馏和低秩压缩。
- `performance/` — 性能指标，包括 latency、throughput、TTFT、TPOT 和 benchmark。

## 文件契约

- 二级目录的 `index.md` 负责该推理模块入口和直接子项索引。
- 具体推理笔记可以先以 TODO 占位，后续再按知识写作规范扩展。
- 每篇笔记说明优化对象、核心机制、收益指标、代价和适用场景。
- 涉及架构假设时，应链接回对应架构笔记；涉及训练后处理或精度损失时，应链接到训练或评测模块。

## 边界

- 模型结构放入 `../architecture/`。
- 训练和对齐阶段放入 `../training/`。
- 应用编排、RAG、Agent 和评测入口放入 `../application/`。

## Direct-Call 信息

- 目录入口链接：`[[inference/|Inference]]`。
- 推理链路入口：`[[inference/decoding/|Decoding]]`、`[[inference/kv-cache-and-memory/|KV Cache and Memory]]`。
- Serving 入口：`[[inference/serving-systems/|Serving Systems]]`。
