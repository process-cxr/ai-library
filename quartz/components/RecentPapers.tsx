import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { SimpleSlug, resolveRelative } from "../util/path"
import { QuartzPluginData } from "../plugins/vfile"
import { classNames } from "../util/lang"
import style from "./styles/recentPapers.scss"

interface Options {
  linkToMore: SimpleSlug | false
}

interface RecommendationGroup {
  key: string
  label: string
  pages: QuartzPluginData[]
}

const defaultOptions: Options = {
  linkToMore: "sources" as SimpleSlug,
}

function frontmatterDate(page: QuartzPluginData): string {
  const frontmatter = page.frontmatter
  const value =
    frontmatter?.paper_date ??
    frontmatter?.source_date ??
    frontmatter?.modified ??
    frontmatter?.published ??
    frontmatter?.created
  return typeof value === "string" ? value : ""
}

function sortByDate(first: QuartzPluginData, second: QuartzPluginData): number {
  const dateDifference = frontmatterDate(second).localeCompare(frontmatterDate(first))
  if (dateDifference !== 0) return dateDifference

  const firstTitle = String(first.frontmatter?.title ?? "")
  const secondTitle = String(second.frontmatter?.title ?? "")
  return firstTitle.localeCompare(secondTitle)
}

function isTopLevelProjectIndex(page: QuartzPluginData): boolean {
  const segments = page.slug?.split("/") ?? []
  return segments.length === 3 && segments[0] === "projects" && segments[2] === "index"
}

function recommendationGroups(allFiles: QuartzPluginData[]): RecommendationGroup[] {
  const processedSources = allFiles.filter((page) => page.frontmatter?.status === "processed")
  const papers = processedSources
    .filter((page) => page.frontmatter?.source_type === "paper")
    .sort(sortByDate)
  const blogs = allFiles
    .filter(
      (page) =>
        page.frontmatter?.source_type === "personal-blog" ||
        page.frontmatter?.source_type === "official-blog",
    )
    .sort(sortByDate)
  const reports = processedSources
    .filter(
      (page) =>
        page.frontmatter?.source_type === "report" ||
        page.frontmatter?.source_type === "technical-report",
    )
    .sort(sortByDate)
  const activeProjects = allFiles
    .filter(
      (page) =>
        page.frontmatter?.type === "project" &&
        page.frontmatter?.status === "growing" &&
        isTopLevelProjectIndex(page),
    )
    .sort(sortByDate)

  return [
    { key: "papers", label: "Paper notes", pages: papers.slice(0, 4) },
    { key: "blogs", label: "Blog notes", pages: blogs.slice(0, 2) },
    { key: "reports", label: "Technical reports", pages: reports.slice(0, 2) },
    { key: "projects", label: "Active engineering", pages: activeProjects.slice(0, 2) },
  ]
}

export default ((userOpts?: Partial<Options>) => {
  const RecentPapers: QuartzComponent = ({
    allFiles,
    fileData,
    displayClass,
  }: QuartzComponentProps) => {
    const options = { ...defaultOptions, ...userOpts }
    const groups = recommendationGroups(allFiles)

    return (
      <section class={classNames(displayClass, "recent-papers")}>
        <div class="recent-papers-heading">
          <p class="recent-papers-kicker">Recent research notes</p>
          {options.linkToMore && (
            <a
              class="recent-papers-more internal"
              href={resolveRelative(fileData.slug!, options.linkToMore)}
            >
              Browse research archive
            </a>
          )}
        </div>
        <div class="recent-papers-groups">
          {groups.map((group) => (
            <section class={`recent-papers-group recent-papers-group-${group.key}`} key={group.key}>
              <h3>{group.label}</h3>
              <ol class="recent-papers-list">
                {group.pages.map((page, index) => {
                  const title = page.frontmatter?.title ?? "Untitled note"
                  const date = frontmatterDate(page)
                  return (
                    <li class="recent-paper-item" key={page.slug}>
                      <span class="recent-paper-index" aria-hidden="true">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <a
                        href={resolveRelative(fileData.slug!, page.slug!)}
                        class="recent-paper-link"
                      >
                        <span class="recent-paper-copy">
                          {date && <time datetime={date}>{date}</time>}
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
          ))}
        </div>
      </section>
    )
  }

  RecentPapers.css = style
  return RecentPapers
}) satisfies QuartzComponentConstructor
