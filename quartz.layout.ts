import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { FileTrieNode } from "./quartz/util/fileTrie"

const explorerSort = (a: FileTrieNode, b: FileTrieNode): number => {
  const rootOrder = [
    "fundamentals",
    "architecture",
    "training",
    "inference",
    "application",
    "projects",
    "sources",
  ]
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

const explorerOptions = {
  sortFn: explorerSort,
  homeDefaultOpenPaths: ["fundamentals", "architecture", "training", "inference", "application"],
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
    Component.Explorer(explorerOptions),
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
    Component.Explorer(explorerOptions),
  ],
  right: [],
}
