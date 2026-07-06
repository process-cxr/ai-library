---
title: Capability Injection
created: 2026-02-28
published: 2026-02-28
modified: 2026-06-30
type: topic
status: mature
area: training
tags:
  - mid-training
  - capability
---

Capability Injection 指在已有 base model 或 intermediate checkpoint 上，通过 targeted data、continued pretraining、SFT-like data、verifier filtering 或合成数据增强某类能力。它常发生在 mid-training 阶段，因为此时模型已有通用表示，继续训练可以更高效地塑造特定能力。

它与 [[training/mid-training/domain-adaptation|Domain Adaptation]] 相关，但不完全相同。Domain adaptation 关注分布或领域，例如法律、医疗、金融；capability injection 关注能力维度，例如数学推理、代码生成、工具使用、长上下文、多语言、格式遵循。

## 常见能力

| 能力 | 数据形态 | 训练信号 |
|---|---|---|
| Math reasoning | 题目、解答、证明、CoT、verifier 通过样本 | NTP、SFT、RLVR |
| Code | 仓库、函数、测试、issue、patch、执行结果 | NTP、SFT、unit-test filtering |
| Long context | 长文档、多文档 QA、代码仓库、长对话 | NTP、retrieval tasks |
| Tool use | tool traces、API docs、函数调用样本 | SFT、schema validation、RL |
| Multilingual | 多语言高质量文档、翻译对、跨语言 QA | NTP、SFT |
| Structured output | JSON/XML/YAML/schema examples | SFT、schema validation |
| Scientific reasoning | 论文、教材、实验报告、公式推导 | NTP、SFT、verifier |

能力注入的关键不是简单增加某类数据，而是让训练信号与目标能力匹配。例如，代码仓库 NTP 能增强代码分布建模，但不一定让模型学会根据测试修 bug；数学题解 NTP 能增强解题模式，但可验证任务可能需要 verifier 或 RLVR。

## 数据设计

能力注入通常需要专门数据工程：

- 选择高质量 seed data；
- 构造目标能力任务；
- 用 teacher model 或程序生成候选；
- 使用 verifier / unit test / schema validator 筛选；
- 做 semantic dedup 和 contamination audit；
- 与通用 replay data 混合；
- 记录能力数据比例和阶段。

数据质量比数据量更关键。低质 CoT、错误代码、不可执行工具调用或污染题库会直接塑造错误能力。

## 训练方式

常见方式包括：

1. **CPT-style injection**：继续语言建模，适合代码、数学文本、科学文档、长文档。
2. **SFT-style injection**：用 prompt-response 或 tool trace，适合工具使用、结构化输出和任务格式。
3. **Verifier-filtered distillation**：生成多个候选，只保留通过测试或验证的样本。
4. **RLVR / reward-based training**：对可验证任务使用 reward 进一步优化。
5. **Curriculum injection**：从简单能力样本逐步过渡到复杂样本。

不同方式可以叠加。例如，代码能力可先用仓库 CPT，再用 issue/patch SFT，最后用 unit-test reward 或 rejection sampling。

## Agentic Capability Injection

Agentic capability injection 指在后训练之前，提前注入工具使用、规划、环境反馈利用和长程任务推进等 agent 行为倾向。它与普通代码或领域能力注入的区别在于，训练数据需要保留 action-observation loop，而不仅是最终产物。

在 software engineering agent 中，更合适的数据形态不是孤立的代码文件、单独的 bug localization 样本或 oracle-file patch，而是把 issue、相关文件、编辑序列、测试输出和修正过程组织成连续样本。PR 数据可以提供大规模 contextually-native supervision；可执行环境 rollout 可以提供 environmentally-native feedback。相关案例见 [[sources/papers/2026-davinci-dev-agent-native-mid-training-for-software-engineering|daVinci-Dev: Agent-native Mid-training for Software Engineering]]。

另一种路径是先用 workflow / Agentless training 注入原子技能，再迁移到更开放的 agent 框架。固定 workflow 可以将软件工程任务拆成 localization、code edit、test writing、verification 和 self-reflection 等可验证子问题，使 SFT 或 RL 更容易获得稳定信号。这类训练得到的不是完整 agent policy，而是后续多轮 agent 适配所需的 skill priors。相关案例见 [[sources/papers/2025-kimi-dev-agentless-training-as-skill-prior-for-swe-agents|Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents]]。

## 与 Post-training 的边界

能力注入如果发生在 base 或 intermediate checkpoint 上，通常更关注能力表示；post-training 更关注交互行为和用户可控性。

例如：

- 数学 CPT 让模型更熟悉数学文本和推理轨迹；
- 数学 SFT 让模型学会按题目输出解答；
- RLVR 让模型优化可验证答案；
- final alignment 让模型在用户对话中安全、简洁、可控地展示能力。

不要期待单一阶段完成所有目标。能力形成和行为对齐应分层设计。

## 评测

能力注入需要同时评估目标能力和副作用：

- target benchmark；
- held-out domain validation loss；
- contamination 检查；
- 通用 benchmark；
- 相关能力迁移；
- 无关能力退化；
- 输出格式和安全行为；
- 后续 SFT/RL 后能力是否保留。

尤其对数学和代码，benchmark contamination 很容易造成虚高。对工具使用，schema 合法不等于任务成功；对多语言，平均分不代表低资源语言改善。

## 风险

- **能力过拟合**：模型只擅长训练格式，而非真实任务。
- **负迁移**：目标能力提升，其他能力下降。
- **风格污染**：模型生成过多题解式、代码式或工具式文本。
- **Verifier bias**：模型学会取悦验证器，而不是真正解决问题。
- **合成数据偏差**：teacher 错误被系统性蒸馏。
- **阶段混淆**：把 SFT 行为数据过早混入 base CPT，改变 base 分布。

## 相关概念

- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/mid-training/domain-adaptation|Domain Adaptation]]
- [[training/mid-training/long-context-training|Long Context Training]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
- [[training/post-training/sft|SFT]]
- [[training/post-training/grpo|GRPO]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
