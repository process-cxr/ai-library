---
title: Rate-Distortion
created: 2026-01-03
published: 2026-01-03
modified: 2026-01-03
type: topic
status: seed
area: fundamentals
tags:
  - information-theory
  - compression
  - quantization
aliases:
  - Rate Distortion
  - Rate-Distortion Theory
---

## 概念界定

率失真思想研究在允许一定信息损失的情况下，如何用更少的编码成本表示数据。它提供了一种理解压缩、量化和表示瓶颈的基础视角。

## 背景与问题

大模型部署时常需要压缩模型权重、激活或 KV Cache。压缩通常会减少存储和计算成本，但也可能带来精度损失。率失真思想关注的正是“压缩率”和“失真程度”之间的权衡。

## 定义与记号

直观上，率失真问题关心：

```text
在失真 D 可接受的条件下，最少需要多少编码率 R？
```

或者反过来：

```text
在编码率 R 固定时，能达到多小的失真 D？
```

这里：

- `Rate`：编码成本、比特数或表示容量。
- `Distortion`：压缩后与原始对象之间的误差。

## 直观解释

如果把模型量化到更低 bit，存储和计算更省，但权重表示更粗糙，可能带来性能下降。率失真思想就是用信息论方式理解这种取舍。

## 基本性质

- 压缩率越高，通常失真风险越大。
- 允许少量失真可以显著降低存储或通信成本。
- 不同对象对失真的敏感性不同，例如某些层或某些通道可能更关键。
- 工程中的量化策略可以看作在成本和误差之间寻找平衡。

## 常见误解

- 误解：压缩只要看模型大小。
  - 正确理解：还要看失真如何影响下游质量、稳定性和延迟。
- 误解：低 bit 一定导致不可接受的性能下降。
  - 正确理解：合理校准、分组量化和混合精度可以在较小失真下显著压缩。

## 相关概念

- [[inference/quantization/quantization|量化]] — 率失真思想在低精度表示中的应用。
- [[inference/compression/model-compression|模型压缩]] — 压缩方法需要权衡成本和失真。
- [[fundamentals/linear-algebra/numerical-stability|数值稳定性]] — 低精度表示带来的误差问题。
