import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { FileTrieNode } from "./quartz/util/fileTrie"

const learningRootOrder = ["fundamentals", "architecture", "training", "inference", "application"]
const learningRoots = new Set(learningRootOrder)

const explorerSort = (a: FileTrieNode, b: FileTrieNode): number => {
  const rootOrder = ["fundamentals", "architecture", "training", "inference", "application"]
  const aIsRootFolder = a.isFolder && a.slug.split("/").length === 2
  const bIsRootFolder = b.isFolder && b.slug.split("/").length === 2
  const aRank = aIsRootFolder ? rootOrder.indexOf(a.slugSegment.toLowerCase()) : -1
  const bRank = bIsRootFolder ? rootOrder.indexOf(b.slugSegment.toLowerCase()) : -1

  if (aRank !== bRank && (aRank >= 0 || bRank >= 0)) {
    return (aRank < 0 ? rootOrder.length : aRank) - (bRank < 0 ? rootOrder.length : bRank)
  }

  if ((!a.isFolder && !b.isFolder) || (a.isFolder && b.isFolder)) {
    return a.displayName.localeCompare(b.displayName, undefined, {
      numeric: true,
      sensitivity: "base",
    })
  }

  return a.isFolder ? -1 : 1
}

const researchSort = (a: FileTrieNode, b: FileTrieNode): number => {
  const rootOrder = ["projects", "papers", "blogs", "reports"]
  const aPath = a.slug.endsWith("/index") ? a.slug.slice(0, -"/index".length) : a.slug
  const bPath = b.slug.endsWith("/index") ? b.slug.slice(0, -"/index".length) : b.slug
  const aRank =
    aPath === "sources"
      ? rootOrder.length
      : aPath === "projects"
        ? rootOrder.indexOf("projects")
        : aPath.startsWith("sources/")
          ? rootOrder.indexOf(aPath.split("/")[1])
          : -1
  const bRank =
    bPath === "sources"
      ? rootOrder.length
      : bPath === "projects"
        ? rootOrder.indexOf("projects")
        : bPath.startsWith("sources/")
          ? rootOrder.indexOf(bPath.split("/")[1])
          : -1
  if (aRank !== bRank && (aRank >= 0 || bRank >= 0)) {
    return (aRank < 0 ? rootOrder.length : aRank) - (bRank < 0 ? rootOrder.length : bRank)
  }

  if ((!a.isFolder && !b.isFolder) || (a.isFolder && b.isFolder)) {
    return a.displayName.localeCompare(b.displayName, undefined, {
      numeric: true,
      sensitivity: "base",
    })
  }

  return a.isFolder ? -1 : 1
}

const learningExplorerOptions = {
  title: "学习",
  sortFn: explorerSort,
  filterFn: (node: FileTrieNode): boolean => {
    const rootOrder = ["fundamentals", "architecture", "training", "inference", "application"]
    const path = node.slug.endsWith("/index") ? node.slug.slice(0, -"/index".length) : node.slug
    return rootOrder.some((root) => path === root || path.startsWith(`${root}/`))
  },
  stateKey: "fileTree-learning",
  homeDefaultOpenPaths: [...learningRoots],
}

const researchExplorerOptions = {
  title: "探索",
  sortFn: researchSort,
  filterFn: (node: FileTrieNode): boolean => {
    const roots = ["projects", "sources/papers", "sources/blogs", "sources/reports"]
    const path = node.slug.endsWith("/index") ? node.slug.slice(0, -"/index".length) : node.slug
    return path === "sources" || roots.some((root) => path === root || path.startsWith(`${root}/`))
  },
  stateKey: "fileTree-research",
  flattenPaths: ["sources"],
  homeDefaultOpenPaths: [],
}

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {},
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
    Component.ReaderMode(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer(learningExplorerOptions),
    Component.Explorer(researchExplorerOptions),
  ],
  right: [
    Component.Graph(),
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Backlinks(),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.ReaderMode(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer(learningExplorerOptions),
    Component.Explorer(researchExplorerOptions),
  ],
  right: [],
}
