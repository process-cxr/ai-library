import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { SimpleSlug, resolveRelative } from "../util/path"
import { QuartzPluginData } from "../plugins/vfile"
import { classNames } from "../util/lang"
import style from "./styles/recentPapers.scss"

interface Options {
  title?: string
  limit: number
  linkToMore: SimpleSlug | false
}

const defaultOptions: Options = {
  title: "Recent Paper Reading",
  limit: 8,
  linkToMore: "sources/papers",
}

function paperDate(page: QuartzPluginData): string {
  const value = page.frontmatter?.paper_date
  return typeof value === "string" ? value : ""
}

function paperOrder(page: QuartzPluginData): number {
  const value = page.frontmatter?.paper_order
  return typeof value === "string" || typeof value === "number" ? Number(value) || 0 : 0
}

function sortByPaperDate(first: QuartzPluginData, second: QuartzPluginData): number {
  const dateDifference = paperDate(second).localeCompare(paperDate(first))
  if (dateDifference !== 0) return dateDifference

  const orderDifference = paperOrder(second) - paperOrder(first)
  if (orderDifference !== 0) return orderDifference

  const firstTitle = String(first.frontmatter?.title ?? "")
  const secondTitle = String(second.frontmatter?.title ?? "")
  return firstTitle.localeCompare(secondTitle)
}

export default ((userOpts?: Partial<Options>) => {
  const RecentPapers: QuartzComponent = ({
    allFiles,
    fileData,
    displayClass,
  }: QuartzComponentProps) => {
    const options = { ...defaultOptions, ...userOpts }
    const papers = allFiles
      .filter(
        (page) =>
          page.frontmatter?.source_type === "paper" && page.frontmatter?.status === "processed",
      )
      .sort(sortByPaperDate)

    return (
      <section class={classNames(displayClass, "recent-papers")}>
        <div class="recent-papers-heading">
          <div>
            <p class="recent-papers-kicker">Recent paper notes</p>
          </div>
          {options.linkToMore && (
            <a
              class="recent-papers-more internal"
              href={resolveRelative(fileData.slug!, options.linkToMore)}
            >
              Browse paper archive
            </a>
          )}
        </div>
        <ol class="recent-papers-list">
          {papers.slice(0, options.limit).map((page, index) => {
            const title = page.frontmatter?.title ?? "Untitled paper"
            const date = paperDate(page)
            return (
              <li class="recent-paper-item" key={page.slug}>
                <span class="recent-paper-index" aria-hidden="true">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <a href={resolveRelative(fileData.slug!, page.slug!)} class="recent-paper-link">
                  <span class="recent-paper-copy">
                    {date && <time datetime={`${date}-01`}>{date}</time>}
                    <span class="recent-paper-title">{title}</span>
                  </span>
                  <span class="recent-paper-arrow" aria-hidden="true">
                    -&gt;
                  </span>
                </a>
              </li>
            )
          })}
        </ol>
      </section>
    )
  }

  RecentPapers.css = style
  return RecentPapers
}) satisfies QuartzComponentConstructor
