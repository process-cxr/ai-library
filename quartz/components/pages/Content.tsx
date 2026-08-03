import { ComponentChildren } from "preact"
import { htmlToJsx } from "../../util/jsx"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import sourcePanelStyle from "../styles/sourcePanel.scss"
// @ts-ignore
import sourcePanelScript from "../scripts/sourcePanel.inline"

function resolvePaperPdfUrl(
  frontmatter: QuartzComponentProps["fileData"]["frontmatter"],
): string | null {
  const explicitPdfUrl =
    typeof frontmatter?.pdf_url === "string"
      ? frontmatter.pdf_url
      : typeof frontmatter?.pdfUrl === "string"
        ? frontmatter.pdfUrl
        : typeof frontmatter?.pdf === "string"
          ? frontmatter.pdf
          : null
  if (explicitPdfUrl) {
    return explicitPdfUrl
  }

  const sourceUrl =
    typeof frontmatter?.source_url === "string"
      ? frontmatter.source_url
      : typeof frontmatter?.sourceUrl === "string"
        ? frontmatter.sourceUrl
        : null
  if (!sourceUrl) {
    return null
  }

  try {
    const url = new URL(sourceUrl)
    if (url.hostname === "arxiv.org") {
      if (url.pathname.startsWith("/abs/")) {
        const arxivId = url.pathname.slice("/abs/".length)
        return `https://arxiv.org/pdf/${arxivId}.pdf`
      }

      if (url.pathname.startsWith("/pdf/")) {
        return url.pathname.endsWith(".pdf") ? url.toString() : `${url.toString()}.pdf`
      }
    }

    if (url.pathname.endsWith(".pdf")) {
      return url.toString()
    }

    return sourceUrl
  } catch {
    return sourceUrl
  }
}

function formatPdfLabel(pdfUrl: string): string {
  try {
    const url = new URL(pdfUrl)
    return `${url.hostname}${url.pathname}`
  } catch {
    return pdfUrl
  }
}

const Content: QuartzComponent = ({ fileData, tree }: QuartzComponentProps) => {
  const content = htmlToJsx(fileData.filePath!, tree) as ComponentChildren
  const classes: string[] = fileData.frontmatter?.cssclasses ?? []
  const classString = ["popover-hint", ...classes].join(" ")
  const isPaper = fileData.frontmatter?.source_type === "paper"
  const pdfUrl = isPaper ? resolvePaperPdfUrl(fileData.frontmatter) : null
  const paperLayoutClass = "content-paper-layout"

  if (isPaper) {
    return (
      <div class={paperLayoutClass}>
        <article class={classString}>
          <div class="paper-panel-toolbar">
            <button class="paper-panel-toggle" type="button">
              Show PDF
            </button>
          </div>
          {content}
        </article>
        <section class="paper-panel" data-pdf-url={pdfUrl ?? undefined} hidden>
          <div class="paper-panel-header">
            <div class="paper-panel-title">
              <h3>PDF</h3>
              <p class="paper-panel-path">
                {pdfUrl ? formatPdfLabel(pdfUrl) : "No PDF URL detected"}
              </p>
            </div>
            <div class="paper-panel-actions">
              {pdfUrl ? (
                <a class="paper-panel-open" href={pdfUrl} target="_blank" rel="noopener noreferrer">
                  Open PDF
                </a>
              ) : null}
            </div>
          </div>
          <div class="paper-panel-status">
            {pdfUrl ? "PDF preview" : "Add a pdf_url or source_url to enable the PDF panel."}
          </div>
          {pdfUrl ? (
            <iframe class="paper-panel-frame" src={pdfUrl} title="Paper PDF" loading="eager" />
          ) : null}
        </section>
      </div>
    )
  }

  return (
    <div class="content-source-layout">
      <article class={classString}>{content}</article>
      <section class="source-panel" hidden>
        <div class="source-panel-header">
          <div class="source-panel-title">
            <h3>Source</h3>
            <p class="source-panel-path">Select a source link</p>
          </div>
          <div class="source-panel-actions">
            <a class="source-panel-open" href="#" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
            <button class="source-panel-close" type="button">
              Hide
            </button>
          </div>
        </div>
        <div class="source-panel-status">Select a GitHub source link in the note.</div>
        <pre class="source-panel-code" tabindex={0}>
          <code></code>
        </pre>
      </section>
    </div>
  )
}

Content.css = sourcePanelStyle
Content.afterDOMLoaded = sourcePanelScript

export default (() => Content) satisfies QuartzComponentConstructor
