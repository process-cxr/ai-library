# docs

`docs/` 是 Quartz 上游文档目录，用于理解和维护站点生成器能力。它不是个人 AI 学习正文目录。

## 当前模块职责

- 保存 Quartz 的功能说明、插件文档、组件说明、部署指南和图片资源。
- 为修改 `../quartz.config.ts`、`../quartz.layout.ts`、`../quartz/` 源码提供参考。
- 帮助区分“站点生成器文档”和“个人知识库正文”。

## 直接子项

- `advanced/` — Quartz 架构、组件创建、插件开发和路径机制等高级主题。
- `features/` — Backlinks、Graph、Search、Darkmode、RSS、i18n 等功能说明。
- `plugins/` — Quartz 插件清单与单插件配置说明。
- `images/` — 文档示例图片与站点说明图。
- `tags/` — 文档标签页。
- `index.md` — Quartz 文档首页。
- 根级 `.md` — 作者内容、构建、配置、托管、布局、迁移、升级等说明。

## 文件契约

- 保持 Quartz 文档性质，优先描述框架功能和使用方式。
- 修改站点配置、布局或插件前，可先查本目录对应说明。
- 个人 AI 笔记应写入 `../content/`，不要混入本目录。

## 边界

- 不放个人学习正文。
- 不放 Quartz 运行时代码。
- 不记录项目改造进度或部署流水账。

## Direct-Call 信息

- 当前预览脚本：`npm run docs`，对应 `npx quartz build --serve -d docs`。
- 发布个人知识库时，应改用 `content/` 作为构建源。
