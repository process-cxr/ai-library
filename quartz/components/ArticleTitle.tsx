import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

const ArticleTitle: QuartzComponent = ({
  fileData,
  displayClass,
  articleTitle,
}: QuartzComponentProps) => {
  const title = articleTitle?.text ?? fileData.frontmatter?.title
  if (title) {
    return (
      <h1 id={articleTitle?.id} class={classNames(displayClass, "article-title")}>
        {title}
      </h1>
    )
  } else {
    return null
  }
}

ArticleTitle.css = `
.article-title {
  margin: 2rem 0 0 0;
}
`

export default (() => ArticleTitle) satisfies QuartzComponentConstructor
