import { ComponentChildren } from "preact"
import { htmlToJsx } from "../../util/jsx"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import sourcePanelStyle from "../styles/sourcePanel.scss"
// @ts-ignore
import sourcePanelScript from "../scripts/sourcePanel.inline"

const Content: QuartzComponent = ({ fileData, tree }: QuartzComponentProps) => {
  const content = htmlToJsx(fileData.filePath!, tree) as ComponentChildren
  const classes: string[] = fileData.frontmatter?.cssclasses ?? []
  const classString = ["popover-hint", ...classes].join(" ")
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
