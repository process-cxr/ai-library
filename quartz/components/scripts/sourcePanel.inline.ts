type SourceLink = {
  url: URL
  rawUrl: string
  repoPath: string
  lineStart?: number
  lineEnd?: number
}

type PaperPanelRefs = {
  panel: HTMLElement
  layout: HTMLElement | null
  toggle: HTMLButtonElement | null
  status: HTMLElement | null
}

const githubBlobPattern = /^\/([^/]+)\/([^/]+)\/blob\/([^/]+)\/(.+)$/
const sourceCache = new Map<string, string>()
function setSourcePanelMode(mode: "on" | "off") {
  document.documentElement.setAttribute("source-panel-mode", mode)
  document.dispatchEvent(new CustomEvent("sourcepanelchange", { detail: { mode } }))
}

function setPaperPageMode(isPaperPage: boolean) {
  document.documentElement.toggleAttribute("data-paper-page", isPaperPage)
}

function getPaperPanel() {
  const panel = document.querySelector<HTMLElement>(".paper-panel")
  if (!panel) {
    return null
  }

  return {
    panel,
    layout: document.querySelector<HTMLElement>(".content-paper-layout"),
    toggle: document.querySelector<HTMLButtonElement>(".paper-panel-toggle"),
    status: panel.querySelector<HTMLElement>(".paper-panel-status"),
  } satisfies PaperPanelRefs
}

function parseLineHash(hash: string): Pick<SourceLink, "lineStart" | "lineEnd"> {
  const range = hash.match(/^#L(\d+)(?:-L?(\d+))?$/)
  if (!range) {
    return {}
  }

  const lineStart = Number.parseInt(range[1], 10)
  const lineEnd = range[2] ? Number.parseInt(range[2], 10) : lineStart
  return {
    lineStart,
    lineEnd: Math.max(lineStart, lineEnd),
  }
}

function parseGithubSourceUrl(href: string): SourceLink | null {
  let url: URL
  try {
    url = new URL(href)
  } catch {
    return null
  }

  if (url.hostname !== "github.com") {
    return null
  }

  const match = url.pathname.match(githubBlobPattern)
  if (!match) {
    return null
  }

  const [, owner, repo, branch, filePath] = match
  return {
    url,
    rawUrl: `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${filePath}`,
    repoPath: `${owner}/${repo}/${filePath}`,
    ...parseLineHash(url.hash),
  }
}

function getSourcePanel() {
  const panel = document.querySelector<HTMLElement>(".source-panel")
  if (!panel) {
    return null
  }

  return {
    panel,
    status: panel.querySelector<HTMLElement>(".source-panel-status"),
    path: panel.querySelector<HTMLElement>(".source-panel-path"),
    open: panel.querySelector<HTMLAnchorElement>(".source-panel-open"),
    close: panel.querySelector<HTMLButtonElement>(".source-panel-close"),
    pre: panel.querySelector<HTMLPreElement>(".source-panel-code"),
    code: panel.querySelector<HTMLElement>(".source-panel-code > code"),
    layout: document.querySelector<HTMLElement>(".content-source-layout"),
  }
}

function getPaperPanelStorageKey() {
  const slug = document.body.dataset.slug ?? window.location.pathname
  return `paper-panel:${slug}`
}

function closePanelsForNavigation() {
  setSourcePanelMode("off")
  setPaperPageMode(false)

  const paperRefs = getPaperPanel()
  if (paperRefs) {
    paperRefs.panel.hidden = true
    paperRefs.panel.setAttribute("hidden", "")
    paperRefs.layout?.classList.remove("has-paper-panel")
    if (paperRefs.toggle) {
      paperRefs.toggle.textContent = "Show PDF"
      paperRefs.toggle.setAttribute("aria-pressed", "false")
    }
    if (paperRefs.status) {
      paperRefs.status.textContent = "PDF hidden"
    }
  }

  const sourceRefs = getSourcePanel()
  if (sourceRefs) {
    sourceRefs.panel.hidden = true
    sourceRefs.layout?.classList.remove("has-source-panel")
  }
}

function setPaperPanelVisibility(refs: PaperPanelRefs, hidden: boolean) {
  refs.panel.hidden = hidden
  refs.panel.toggleAttribute("hidden", hidden)
  refs.layout?.classList.toggle("has-paper-panel", !hidden)
  if (refs.toggle) {
    refs.toggle.textContent = hidden ? "Show PDF" : "Hide PDF"
    refs.toggle.setAttribute("aria-pressed", String(!hidden))
  }
  if (refs.status) {
    refs.status.textContent = hidden ? "PDF hidden" : "PDF preview"
  }
}

function setupPaperPanel() {
  const refs = getPaperPanel()
  if (!refs) {
    setPaperPageMode(false)
    return false
  }

  setPaperPageMode(true)
  const pdfUrl = refs.panel.dataset.pdfUrl ?? null

  if (!pdfUrl) {
    refs.panel.hidden = true
    refs.layout?.classList.remove("has-paper-panel")
    setSourcePanelMode("off")
    if (refs.toggle) {
      refs.toggle.textContent = "No PDF"
      refs.toggle.disabled = true
    }
    if (refs.status) {
      refs.status.textContent = "Add a pdf_url or source_url to enable the PDF panel."
    }
    return true
  }

  refs.layout?.classList.remove("has-paper-panel")
  refs.panel.hidden = true
  refs.panel.setAttribute("hidden", "")
  if (refs.toggle) {
    refs.toggle.textContent = "Show PDF"
    refs.toggle.setAttribute("aria-pressed", "false")
    refs.toggle.disabled = false
  }
  if (refs.status) {
    refs.status.textContent = "PDF hidden"
  }
  setSourcePanelMode("off")

  const onToggle = () => {
    const nextHidden = !refs.panel.hidden
    setPaperPanelVisibility(refs, nextHidden)
    setSourcePanelMode(nextHidden ? "off" : "on")
  }

  refs.toggle?.addEventListener("click", onToggle)
  window.addCleanup(() => refs.toggle?.removeEventListener("click", onToggle))
  return true
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
}

function renderSourceLines(source: string, lineStart?: number, lineEnd?: number): string {
  const lines = source.split("\n")
  const maxDigits = String(lines.length).length
  return lines
    .map((line, index) => {
      const lineNumber = index + 1
      const highlighted =
        lineStart !== undefined && lineNumber >= lineStart && lineNumber <= (lineEnd ?? lineStart)
      return `<span id="source-line-${lineNumber}" class="source-line${
        highlighted ? " is-highlighted" : ""
      }" data-line="${lineNumber}"><span class="source-line-number">${String(lineNumber).padStart(
        maxDigits,
        " ",
      )}</span><span class="source-line-content">${escapeHtml(line) || " "}</span></span>`
    })
    .join("")
}

async function fetchSource(rawUrl: string): Promise<string> {
  const memoryHit = sourceCache.get(rawUrl)
  if (memoryHit !== undefined) {
    return memoryHit
  }

  const storageKey = `source-panel:${rawUrl}`
  const storageHit = sessionStorage.getItem(storageKey)
  if (storageHit !== null) {
    sourceCache.set(rawUrl, storageHit)
    return storageHit
  }

  const response = await fetch(rawUrl)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }

  const source = await response.text()
  sourceCache.set(rawUrl, source)
  try {
    sessionStorage.setItem(storageKey, source)
  } catch {
    // Ignore quota failures. The in-memory cache still avoids duplicate fetches on this page.
  }
  return source
}

async function loadSourceIntoPanel(sourceLink: SourceLink) {
  const refs = getSourcePanel()
  if (!refs || !refs.code || !refs.pre || !refs.path || !refs.open || !refs.status) {
    return
  }

  refs.panel.hidden = false
  refs.layout?.classList.add("has-source-panel")
  setSourcePanelMode("on")
  refs.panel.classList.add("is-loading")
  refs.panel.classList.remove("has-error")
  refs.status.textContent = "Loading source..."
  refs.path.textContent = sourceLink.repoPath
  refs.open.href = sourceLink.url.toString()

  try {
    const source = await fetchSource(sourceLink.rawUrl)
    refs.code.innerHTML = renderSourceLines(source, sourceLink.lineStart, sourceLink.lineEnd)
    refs.status.textContent = sourceLink.lineStart
      ? `Showing line ${sourceLink.lineStart}${sourceLink.lineEnd !== sourceLink.lineStart ? `-${sourceLink.lineEnd}` : ""}`
      : "Showing full file"

    const line = sourceLink.lineStart
      ? refs.code.querySelector<HTMLElement>(`#source-line-${sourceLink.lineStart}`)
      : null
    if (line) {
      requestAnimationFrame(() => {
        refs.pre?.scrollTo({
          top: Math.max(line.offsetTop - refs.pre.clientHeight * 0.28, 0),
          behavior: "smooth",
        })
      })
    } else {
      refs.pre.scrollTop = 0
    }
  } catch (error) {
    refs.code.innerHTML = ""
    refs.panel.classList.add("has-error")
    refs.status.textContent =
      error instanceof Error ? `Failed to load source: ${error.message}` : "Failed to load source."
  } finally {
    refs.panel.classList.remove("is-loading")
  }
}

function markSourceLinks(links: HTMLAnchorElement[]) {
  for (const link of links) {
    const sourceLink = parseGithubSourceUrl(link.href)
    if (!sourceLink) {
      continue
    }

    link.dataset.sourcePanel = "true"
    link.title = link.title || "Click to preview source; Cmd/Ctrl-click opens GitHub"

    const onClick = (event: MouseEvent) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) {
        return
      }
      event.preventDefault()
      loadSourceIntoPanel(sourceLink)
    }

    link.addEventListener("click", onClick)
    window.addCleanup(() => link.removeEventListener("click", onClick))
  }
}

function setupSourcePanel() {
  const refs = getSourcePanel()
  if (!refs) {
    setPaperPageMode(false)
    setSourcePanelMode("off")
    return
  }

  setPaperPageMode(false)
  const links = Array.from(document.querySelectorAll<HTMLAnchorElement>("article a[href]"))
  const sourceLinks = links.filter((link) => parseGithubSourceUrl(link.href) !== null)

  refs.panel.hidden = true
  refs.layout?.classList.remove("has-source-panel")
  setSourcePanelMode("off")

  if (sourceLinks.length === 0) {
    return
  }

  markSourceLinks(sourceLinks)

  const onClose = () => {
    refs.panel.hidden = true
    refs.layout?.classList.remove("has-source-panel")
    setSourcePanelMode("off")
  }

  refs.close?.addEventListener("click", onClose)
  window.addCleanup(() => refs.close?.removeEventListener("click", onClose))
}

function setupPanels() {
  setSourcePanelMode("off")
  if (setupPaperPanel()) {
    return
  }

  setupSourcePanel()
}

document.addEventListener("prenav", closePanelsForNavigation)
setupPanels()
document.addEventListener("nav", setupPanels)
