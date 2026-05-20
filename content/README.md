# content

`content/` 是个人 AI 知识库正文根目录，存放会被 Quartz 构建成网页的学习笔记、目录页和概念链接。

## 当前模块职责

- 维护公开阅读的 AI 学习笔记和主题索引。
- 使用 Markdown、frontmatter、wikilink、callout 等 Quartz 语法组织知识。
- 通过 `index.md` 和各主题目录形成稳定阅读入口。

## 直接子项

- `index.md` — 知识库首页，负责导航到各学习模块。
- `fundamentals/` — 数学、机器学习和信息论基础。
- `architecture/` — 模型架构与核心模块。
- `training/` — 预训练、后训练、对齐和分布式训练。
- `inference/` — 推理优化、部署性能和模型压缩。
- `application/` — RAG、Agent、Prompt、工具调用和评测。
- `sources/` — 外部资料阅读分析，包括论文、博客、课程、报告和官方文档。
- `projects/` — 个人实验、复现、Demo、部署记录和工程踩坑。

## 文件契约

- 每个主题目录保留 `index.md` 作为网页入口。
- 普通笔记使用 `.md`，文件名优先使用稳定的英文 kebab-case。
- 页面标题优先写入 frontmatter 的 `title`。
- 跨笔记引用优先使用 Quartz wikilink，例如 `[[training/post-training/sft|SFT]]`。
- 如果笔记回答“这个概念是什么、怎么工作、和其他概念什么关系”，放入五个主题目录。
- 如果笔记分析“某个资料讲了什么、贡献和局限是什么”，放入 `sources/`。
- 如果笔记记录“我做了什么实验、结果如何、踩了什么坑”，放入 `projects/`。

## 边界

- 不放 Quartz 框架源码、组件、插件或构建脚本。
- 不放上游 Quartz 使用文档。
- 不按论文、博客、课程拆多个顶层阅读目录，统一收敛到 `sources/`。
- 不记录阶段性进度、临时 TODO 或迁移流水账。

## Direct-Call 信息

- 网页构建：`npx quartz build -d content`。
- 本地预览：`npx quartz build --serve -d content`。
- 首页入口：`index.md`。
