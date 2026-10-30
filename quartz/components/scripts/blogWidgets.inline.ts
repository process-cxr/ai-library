// Widgets live in script-only sandboxed frames; match their window before
// accepting size messages because sandboxed documents have an opaque origin.
document.addEventListener("nav", () => {
  const frames = Array.from(document.querySelectorAll<HTMLIFrameElement>("iframe.blog-widget"))
  if (frames.length === 0) return

  const syncTheme = (frame: HTMLIFrameElement) => {
    frame.contentWindow?.postMessage(
      {
        type: "ai-library:widget-theme",
        theme: document.documentElement.getAttribute("saved-theme") ?? "light",
      },
      "*",
    )
  }
  const onMessage = (event: MessageEvent) => {
    const frame = frames.find((candidate) => candidate.contentWindow === event.source)
    if (!frame || event.data?.type !== "ai-library:widget-size") return
    const height = event.data.height
    if (typeof height === "number" && Number.isFinite(height) && height > 0) {
      frame.style.height = `${Math.min(2000, Math.max(240, Math.ceil(height)))}px`
    }
  }
  const onThemeChange = () => frames.forEach(syncTheme)

  window.addEventListener("message", onMessage)
  document.addEventListener("themechange", onThemeChange)
  window.addCleanup(() => {
    window.removeEventListener("message", onMessage)
    document.removeEventListener("themechange", onThemeChange)
  })

  for (const frame of frames) {
    const onLoad = () => syncTheme(frame)
    frame.addEventListener("load", onLoad)
    window.addCleanup(() => frame.removeEventListener("load", onLoad))
    syncTheme(frame)
  }
})
