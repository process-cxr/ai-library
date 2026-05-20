# AI Library

这是一个面向个人学习、长期整理和网页托管的 AI 知识库项目。项目基于 Quartz，将 Markdown 笔记发布为可搜索、可双链跳转、可图谱浏览的数字花园。

## 总体规则

- 根目录 `README.md` 只负责项目定位、全局规则、一级目录索引和常用入口。
- 父目录 README 只总结直接子项，不展开孙级目录细节。
- 关键子目录 README 才写模块职责、文件、契约、边界和 direct-call 信息。
- 架构类 README 不记录阶段性进度、TODO、迁移流水账或临时计划。
- 学习正文统一放入 `content/`；Quartz 框架源码、上游文档和配置文件与正文分层维护。

## 一级目录索引

- `content/` — 个人 AI 知识正文，按基础、架构、训练、推理、应用组织。
- `docs/` — Quartz 上游文档，用于理解配置、功能、插件、部署和维护方式。
- `quartz/` — Quartz 静态站点生成器源码，负责把 Markdown 构建为网页。
- `.github/` — GitHub 工作流、Issue/PR 模板和依赖更新配置。

## 根目录关键文件

- `package.json` — npm 脚本、依赖、Node 版本要求和 Quartz CLI 入口。
- `package-lock.json` — npm 依赖锁定文件。
- `quartz.config.ts` — 站点标题、主题、语言、发布地址和插件管线。
- `quartz.layout.ts` — 页面布局组件编排。
- `tsconfig.json` — TypeScript 编译配置。
- `.prettierrc` / `.prettierignore` — 格式化规则和忽略范围。
- `.node-version` / `.npmrc` — Node 与 npm 环境约束。
- `Dockerfile` — 容器化构建入口。
- `globals.d.ts` / `index.d.ts` — 类型声明入口。
- `LICENSE.txt` / `CODE_OF_CONDUCT.md` — 上游许可和社区行为准则。

## Direct-Call 信息

- Quartz CLI：`npm run quartz -- <command>`。
- 当前脚本预览文档：`npm run docs`，对应 `npx quartz build --serve -d docs`。
- 预览个人知识库：`npx quartz build --serve -d content`。
- 类型与格式检查：`npm run check`。
- 自动格式化：`npm run format`。
- 测试：`npm test`。
