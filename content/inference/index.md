---
title: "Inference"
created: 2025-12-27
published: 2025-12-27
modified: 2026-01-03
---

大模型推理优化与部署，从解码链路、KV Cache、Attention 加速到量化、Serving、压缩和性能评测。

## Inference Pipeline

- [[inference/decoding/|Decoding]] — 自回归解码、采样、beam search 和投机解码。
- [[inference/kv-cache-and-memory/|KV Cache and Memory]] — KV Cache、PagedAttention、prefix cache 和显存管理。
- [[inference/attention-acceleration/|Attention Acceleration]] — FlashAttention、FlashDecoding 和 attention kernel。

## Deployment and Optimization

- [[inference/quantization/|Quantization]] — weight-only、AWQ、GPTQ、FP8 和 KV Cache 量化。
- [[inference/serving-systems/|Serving Systems]] — vLLM、continuous batching、请求调度和 PD 分离。
- [[inference/compression/|Compression]] — 模型压缩、剪枝、蒸馏和低秩压缩。
- [[inference/performance/|Performance]] — 延迟、吞吐、TTFT、TPOT 和 benchmark。
