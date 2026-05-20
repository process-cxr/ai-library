# AI Library

这是一个面向 AI / 大模型学习、研究和工程实践的长期知识库项目，用于沉淀学习笔记、论文阅读、源码分析、实验复现和工程经验。项目使用 Quartz 将 Markdown 内容发布为可搜索、可双链跳转、可图谱浏览的数字花园。

## 项目定位

- 面向大模型全栈学习：从基础理论、模型架构、训练体系、推理系统到应用工程。
- 面向长期维护：知识内容按主题沉淀，资料阅读和实践记录作为持续输入。
- 面向网页阅读：`content/` 是最终发布到站点的知识正文，根目录文件主要服务项目维护。
- 面向协作整理：README、目录约定和写作规范用于保持结构清晰、边界稳定。

## 内容组织

- `content/fundamentals/` — 基础理论：线性代数、概率、信息论、神经网络基础、优化。
- `content/architecture/` — 模型架构：Transformer、Attention、位置编码、模型家族、稀疏与高效架构、多模态。
- `content/training/` — 训练体系：预训练、中训练、后训练、数据工程、训练优化、分布式训练、Scaling。
- `content/inference/` — 推理系统：解码、KV Cache、Attention 加速、量化、Serving、压缩和性能分析。
- `content/application/` — 应用工程：Prompting、RAG、Tool Use、Agents、Evaluation。
- `content/sources/` — 资料分析：论文、博客、课程、报告和官方文档阅读。
- `content/projects/` — 实践记录：复现、Demo、部署实验、源码阅读和工程排障。

## 维护规则

- 稳定知识放入五个主题目录；外部资料分析放入 `sources/`；实践、源码阅读和实验记录放入 `projects/`。
- 根目录 `README.md` 只负责项目定位、全局规则、一级目录索引和常用入口。
- 父目录 README 只总结直接子项，不展开孙级目录细节。
- 关键子目录 README 才写模块职责、文件、契约、边界和 direct-call 信息。
- 架构类 README 不记录阶段性进度、迁移流水账或临时计划。
- 网站入口使用各级 `index.md`；维护说明使用 README，并通过 Quartz 配置隐藏在网站外。

## 项目结构

- `content/` — 个人 AI 知识正文，发布到 XR Lab 网站。
- `quartz/` — Quartz 静态站点生成器源码，用于构建和渲染知识库。
- `docs/` — Quartz 上游文档，用于查阅配置、插件、部署和维护方式。
- `.comate/skills/` — 项目级 AI 写作和维护技能说明。
- `.github/` — GitHub 工作流、Issue/PR 模板和依赖更新配置。

## 根目录关键文件

- `quartz.config.ts` — 站点标题、主题、语言、发布地址、忽略规则和插件管线。
- `quartz.layout.ts` — 页面布局组件编排。
- `package.json` — npm 脚本、依赖、Node 版本要求和 Quartz CLI 入口。
- `package-lock.json` — npm 依赖锁定文件。
- `LICENSE.txt` — 本仓库许可说明与 Quartz 上游许可保留。
- `CODE_OF_CONDUCT.md` — 本知识库协作行为准则。

## 常用命令

- 预览个人知识库：`npx quartz build --serve -d content`。
- Quartz CLI：`npm run quartz -- <command>`。
- 当前脚本预览文档：`npm run docs`，对应 `npx quartz build --serve -d docs`。
- 类型与格式检查：`npm run check`。
- 自动格式化：`npm run format`。
- 测试：`npm test`。
