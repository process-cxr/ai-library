---
name: blog-reading-to-ai-library
description: 将英文个人 AI 技术博客整理为 AI Library 的中文翻译或来源笔记。Use when the user asks to read, translate, archive, or study an English AI blog, especially a personal research blog, while preserving the original structure, figures, links, formulas, tables, and first-person voice.
---

# Blog Reading To AI Library

Use this skill for English personal AI and machine learning blogs that the user wants to read in Chinese and archive under `content/sources/blogs/`.

The source may be a URL, pasted plain text, Markdown, or a saved HTML page. For HTML, use the article body (`<article>` or the page's structured article content) rather than translating navigation, scripts, styles, metadata boilerplate, or duplicate JSON-LD text. Resolve relative image URLs against the page's canonical URL and keep the article's original asset links.

## Default translation mode

When the user provides the article text or has the right to provide it for transformation, produce a faithful Chinese translation rather than a summary or commentary.

- Keep the original first-person voice. Translate `I`, `we`, `my`, and `our` as the original author uses them; do not rewrite them as “作者认为” or “本文认为”.
- Preserve the original order and hierarchy of headings, paragraphs, lists, tables, code blocks, equations, citations, links, figures, captions, and footnotes.
- Translate prose accurately and completely. Do not compress, merge, omit, reorder, or add claims.
- Preserve modality and epistemic strength: distinguish claims, hypotheses, predictions, caveats, uncertainty, examples, and personal opinions.
- Keep standard technical terms in English on first use when useful, with a Chinese explanation in parentheses; keep later terminology consistent.
- Keep paper titles, project names, model names, benchmark names, proper nouns, code, identifiers, URLs, and citation keys unchanged unless the source already provides a translation.
- Keep images in their original positions. Prefer the original image URL, preserve the alt text and caption, and retain the image source link.
- Keep formulas mathematically equivalent and preserve variable names. Use fenced code blocks for code and avoid translating code, CLI commands, filenames, or API identifiers.
- Do not add a summary, “我的理解”, critique, or topic-note backfill inside a pure translation unless the user asks for it separately.

## Copyright boundary for URL-only requests

An external URL alone is enough to inspect and cite a blog, but it is not a request to reproduce a complete location-based translation of the entire third-party article. For a long article available only by URL:

- Do not create a full sentence-by-sentence substitute for the source.
- Explain briefly that a complete translation requires user-provided text or authorization to transform it.
- Offer a source-linked Chinese reading note, section map, or concise summary while preserving headings, image links, citations, and the author’s claims accurately.
- If the user supplies the article text in chunks, translate those chunks faithfully and preserve their local structure. Do not use the supplied text as permission to fetch and reproduce unrelated portions of the article.

## Note header

For every archived blog note, put this block immediately after the title or frontmatter. Keep it factual and concise:

```markdown
## 基本信息

- 标题：[Original title](SOURCE_URL)
- 作者：Author
- 来源：Blog or publisher
- 日期：YYYY-MM-DD
- 类型：个人技术博客 / 研究综述
- 标签：Language Model、Agent、...
- 阅读时间：约 N 分钟
```

Use `source_url`, `source_type: personal-blog`, `publisher`, `author`, and `source_date` in frontmatter. Set `type: source`, `area: sources`, and use a stable English kebab-case filename.

## Structure and assets

Keep the original article’s structure as the primary document shape. Do not reorganize a translation into the project’s concept-note template. Add Quartz wikilinks only in a separate related-links block when the user requests cross-linking or when the note is explicitly being converted into a topic note.

For each figure, preserve this pattern:

```markdown
![Translated alt text](ORIGINAL_IMAGE_URL)

_Figure N: Translated caption. Image source: [Original source](SOURCE_URL)._
```

Do not silently replace remote images with generated diagrams. If an image URL is unavailable, keep the original caption and link and mark the missing asset clearly.

## Workflow

1. Establish the canonical article URL, title, author, date, reading time, and available figures.
2. Determine whether the user supplied the text for full translation or only a URL for a reading note.
3. Create or update `content/sources/blogs/<slug>.md` using the metadata block above.
4. Translate or summarize according to the applicable mode without changing the source’s structure.
5. Preserve source links, image URLs, citations, formulas, tables, and code.
6. Add the note to `content/sources/index.md` when it is a new public source.
7. Run Prettier on changed Markdown files and validate with `npm run quartz -- build -d content`.

## Quality checks

- The first-person perspective matches the source.
- No paragraph was summarized or silently dropped in full-translation mode.
- Headings, figures, tables, equations, citations, and links remain in source order.
- Technical terms and proper nouns are consistent throughout the note.
- The original source URL is visible in `## 基本信息` and frontmatter.
- The generated Quartz page builds successfully.
