import assert from "node:assert/strict"
import test from "node:test"
import { Element, Root } from "hast"
import { QuartzPluginData } from "../plugins/vfile"
import { preparePageTitle } from "./pageTitle"

const heading = (tagName: string, text: string, id: string): Element => ({
  type: "element",
  tagName,
  properties: { id },
  children: [{ type: "text", value: text }],
})

test("opening title moves to the header without changing source metadata or anchors", () => {
  const tree: Root = {
    type: "root",
    children: [
      { type: "text", value: "\n" },
      { type: "comment", value: "Introduction" },
      heading("h1", "Paper Title", "paper-title"),
      heading("h2", "Method", "method"),
      heading("h3", "Details", "details"),
    ],
  }
  const fileData: QuartzPluginData = {
    frontmatter: { title: "Paper Title" },
    toc: [
      { text: "Paper Title", slug: "paper-title", depth: 0 },
      { text: "Method", slug: "method", depth: 1 },
      { text: "Details", slug: "details", depth: 2 },
    ],
  }
  const original = structuredClone({ tree, fileData })
  const result = preparePageTitle(tree, fileData)

  assert.deepEqual(result.articleTitle, { text: "Paper Title", id: "paper-title" })
  assert.equal(result.tree.children.length, 4)
  assert.deepEqual(result.fileData.toc, [
    { text: "Method", slug: "method", depth: 0 },
    { text: "Details", slug: "details", depth: 1 },
  ])
  assert.deepEqual({ tree, fileData }, original)
})

test("short and full titles display once while navigation keeps the metadata title", () => {
  for (const [metadata, opening, expected] of [
    ["GPipe", "GPipe: Easy Scaling", "GPipe: Easy Scaling"],
    [
      "RepoCoder: Repository-Level Completion",
      "RepoCoder",
      "RepoCoder: Repository-Level Completion",
    ],
    ["A Pretrainer's Guide", "A Pretrainer’s Guide", "A Pretrainer's Guide"],
    ["Research Report", "研究报告", "研究报告"],
  ]) {
    const fileData: QuartzPluginData = { frontmatter: { title: metadata } }
    const result = preparePageTitle(
      { type: "root", children: [heading("h1", opening, "title")] },
      fileData,
    )
    assert.equal(result.articleTitle?.text, expected)
    assert.equal(result.fileData.frontmatter?.title, metadata)
  }
})

test("hero headings, later H1 sections, and pages without an opening title stay intact", () => {
  for (const first of [
    heading("h2", "Introduction", "intro"),
    {
      type: "element",
      tagName: "div",
      properties: { className: ["hero"] },
      children: [heading("h1", "A different hero heading", "hero")],
    } satisfies Element,
  ]) {
    const tree: Root = {
      type: "root",
      children: [first, heading("h1", "A later section", "section")],
    }
    const fileData: QuartzPluginData = { frontmatter: { title: "Page Title" } }
    const result = preparePageTitle(tree, fileData)
    assert.equal(result.tree, tree)
    assert.equal(result.fileData, fileData)
    assert.equal(result.articleTitle, undefined)
  }
})

test("inline formatting is read as title text and a title-only TOC is removed", () => {
  const title = heading("h1", "", "formatted-title")
  title.children = [
    { type: "text", value: "A " },
    heading("em", "Formatted", "emphasis"),
    { type: "text", value: " Title" },
  ]
  const result = preparePageTitle(
    { type: "root", children: [title] },
    {
      frontmatter: { title: "A Formatted Title" },
      toc: [{ text: "A Formatted Title", slug: "formatted-title", depth: 0 }],
    },
  )
  assert.equal(result.articleTitle?.text, "A Formatted Title")
  assert.equal(result.tree.children.length, 0)
  assert.equal(result.fileData.toc, undefined)
})
