import { Root } from "hast"
import { toString } from "hast-util-to-string"
import { QuartzPluginData } from "../plugins/vfile"

function normalizeTitle(title: string): string {
  return title
    .normalize("NFKC")
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase()
}

export function preparePageTitle(tree: Root, fileData: QuartzPluginData) {
  const index = tree.children.findIndex(
    (node) => node.type !== "comment" && !(node.type === "text" && node.value.trim() === ""),
  )
  const heading = tree.children[index]
  if (heading?.type !== "element" || heading.tagName !== "h1") {
    return { tree, fileData }
  }

  const headingText = toString(heading).trim()
  if (!headingText) return { tree, fileData }

  const metadataTitle = fileData.frontmatter?.title ?? ""
  const normalizedHeading = normalizeTitle(headingText)
  const normalizedMetadata = normalizeTitle(metadataTitle)
  const hasFullMetadataTitle =
    normalizedMetadata === normalizedHeading ||
    [":", "!", "?", " - "].some((separator) =>
      normalizedMetadata.startsWith(normalizedHeading + separator),
    )
  const title = hasFullMetadataTitle ? metadataTitle : headingText
  const id = typeof heading.properties.id === "string" ? heading.properties.id : undefined
  const entries = fileData.toc?.filter((entry) => !id || entry.slug !== id)
  const minDepth = entries?.length ? Math.min(...entries.map((entry) => entry.depth)) : 0

  // Move the opening H1 into the page header, keeping its existing hash target.
  return {
    tree: { ...tree, children: tree.children.filter((_, childIndex) => childIndex !== index) },
    fileData: {
      ...fileData,
      toc: entries?.length
        ? entries.map((entry) => ({ ...entry, depth: entry.depth - minDepth }))
        : undefined,
    },
    articleTitle: { text: title, id },
  }
}
