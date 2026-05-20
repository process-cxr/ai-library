# quartz

`quartz/` 是 Quartz 静态站点生成器源码目录，负责把 Markdown 内容转换成可部署网页。它是框架层，不是个人 AI 学习正文目录。

## 当前模块职责

- 提供 CLI、构建管线、页面组件、插件、样式、国际化、工具函数和静态资源。
- 被根目录 `package.json`、`quartz.config.ts`、`quartz.layout.ts` 调用。
- 支撑 `content/` 或 `docs/` 被构建为网页。

## 直接子项

- `cli/` — 命令行参数、命令处理和辅助逻辑。
- `components/` — Preact 页面组件、布局组件、客户端脚本和组件样式。
- `i18n/` — 国际化入口与本地化文案。
- `plugins/` — 内容转换、过滤和输出插件。
- `processors/` — parse、filter、emit 构建阶段处理器。
- `static/` — 运行时静态资源。
- `styles/` — 全局样式、变量、语法高亮和自定义样式。
- `util/` — 路径、日志、主题、资源、性能和文件树等通用工具。
- `bootstrap-cli.mjs` — CLI 启动入口。
- `bootstrap-worker.mjs` — Worker 启动入口。
- `build.ts` — 构建流程入口。
- `cfg.ts` — 配置类型与默认配置。
- `worker.ts` — 并行构建 worker 逻辑。

## 文件契约

- 构建阶段保持 `parse -> filter -> emit` 的边界。
- 组件、插件、工具函数分别放入对应目录，不在框架层写知识正文。
- 定制 Quartz 行为时优先从配置入口开始，确需改源码再进入本目录。

## 边界

- 个人知识正文放入 `../content/`。
- Quartz 使用文档放入 `../docs/`。
- 站点配置放在 `../quartz.config.ts` 和 `../quartz.layout.ts`。

## Direct-Call 信息

- CLI 入口：`../package.json` 的 `scripts.quartz` 和 `bin.quartz` 指向 `./quartz/bootstrap-cli.mjs`。
- 构建入口：`bootstrap-cli.mjs` 经 CLI 处理后进入 `build.ts`。
- 配置入口：`../quartz.config.ts` 导入 `./quartz/cfg` 和 `./quartz/plugins`。
- 布局入口：`../quartz.layout.ts` 导入 `./quartz/components`。
