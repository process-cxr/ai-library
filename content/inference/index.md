---
title: "Inference"
created: 2026-03-29
published: 2026-03-29
modified: 2026-03-29
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
- [[inference/multimodal/|Multimodal Inference]] — visual token、multimodal context、video streaming 和多模态 serving。
- [[inference/world-models/|World Model Inference]] — rollout、simulation、imagination 和 planning-time inference。
