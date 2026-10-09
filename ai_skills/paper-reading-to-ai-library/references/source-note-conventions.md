# Source Note Conventions

Shared by the paper and technical-report reading skills. These rules concern metadata, writing, evidence, and repository integration, not the analytical outline.

## Basic Information

Use this field order in paper and technical-report source notes:

```markdown
## 基本信息

- 标题：[Full Title](canonical public source)
- 作者：Author names or author team
- 机构/来源：Verified affiliations or publishing organization
- 发表：arXiv identifier, conference/year, journal, or official publication channel
- 日期：YYYY-MM or YYYY-MM-DD
- 代码/数据：[Official resource](public URL)
- 相关主题：[[path/to/topic|Topic]]
```

Omit `机构/来源` or `代码/数据` when unavailable; do not print placeholders or explanations about unsuccessful searches. Prefer PDF author affiliations and footnotes over author-profile searches; email domains may corroborate an affiliation but must not justify an invented lab or department. Use the affiliation at the time of the work, including an explicit internship note when relevant. Keep the field concise; group labs under their parent institutions.

Keep author affiliation separate from venue. Combine former `来源`, `版本`, and `发表线索` fields into `发表`. Identify the analyzed version there when results depend on it, or in the report scope. Do not place local file paths, model configuration, sorting metadata, or the reading process in basic information.

## Frontmatter and Integration

- Preserve established page paths and aliases. Do not migrate or relabel old notes merely because a different reading workflow fits them better.
- Preserve `created` and `published` when revising; update `modified` to the actual edit date. These lifecycle dates are not the paper date.
- Preserve the canonical public `source_url`; cite a versioned primary source in the body when useful.
- Use quoted `paper_date: "YYYY-MM"` for the original publication month. For arXiv papers, use the identifier's YYMM prefix under this library's coarse chronology convention.
- Use quoted `paper_order: "NNNNN"` for the arXiv submission sequence. It is frontmatter-only, not a date or a body field. Omit it for non-arXiv sources; use the verified conference/publication month instead.
- Skill selection does not determine `source_type`. This repository's `RecentPapers` currently selects `paper` plus `processed`; metadata migrations require explicit checking of homepage eligibility, sorting, and PDF reader behavior.
- A local PDF is reading input, not a public repository artifact unless explicitly requested. Never include the user's absolute local path in published notes.
- For detailed source reading, prefer a PDF or document explicitly provided by the user. Do not proactively download remote PDFs or large files; when only a URL is available, ask the user to provide the source file before making claims that depend on the full text. Public HTML pages may establish basic metadata but do not replace the source document.

## Writing and Evidence

Write in Chinese with an English page title and conventional technical terms. Use precise, self-contained prose, not dialogue, recommendations addressed to the user, or progress logs. Do not force comparisons because two sources were read consecutively.

Expand abbreviations at first substantive use with their English full name and Chinese meaning or role, including algorithms, datasets, metrics, and evaluation settings. Metadata alone does not count as an explanation. Do not invent expansions for proper names; describe their purpose instead. For example, IID means independent and identically distributed (独立同分布), not simply in-distribution; TPP means tokens per parameter (每个参数对应的训练 token 数).

Keep source claims, controlled evidence, engineering disclosures, and the reader's interpretation distinguishable. Place section/table/figure references next to consequential quantitative claims. Preserve units, denominators, model variants, and evaluation conditions. Leave undisclosed settings unknown.

Use Markdown, formulas, compact tables, and diagrams only where they aid understanding. Establish variable meanings and distinguish mathematical abstractions from actual trainable modules. Explain the final artifact and its inference path when that is part of the method.

Link related topic notes naturally. Promote only durable, supported conclusions to topic pages, with provenance. Do not move source-specific benchmark results into generic concept definitions. A closing references section belongs at the end.

## Validation

Check the basic-information field order, frontmatter parsing, internal links, abbreviation use, claims against source evidence, and formula rendering. Run `git diff --check` and a Quartz build for note changes. Validate homepage filtering and chronological order if any source metadata changes. Do not stage, commit, or modify unrelated local files such as `.comate/`.
