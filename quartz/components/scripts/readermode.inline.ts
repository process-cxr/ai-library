let isReaderMode = false

function setReaderMode(mode: "on" | "off") {
  isReaderMode = mode === "on"
  document.documentElement.setAttribute("reader-mode", mode)
  for (const readerModeButton of document.getElementsByClassName("readermode")) {
    readerModeButton.setAttribute("aria-pressed", isReaderMode ? "true" : "false")
    readerModeButton.setAttribute("aria-label", isReaderMode ? "Show sidebars" : "Hide sidebars")
    readerModeButton.setAttribute("title", isReaderMode ? "Show sidebars" : "Hide sidebars")
  }
  emitReaderModeChangeEvent(mode)
}

const emitReaderModeChangeEvent = (mode: "on" | "off") => {
  const event: CustomEventMap["readermodechange"] = new CustomEvent("readermodechange", {
    detail: { mode },
  })
  document.dispatchEvent(event)
}

document.addEventListener("nav", () => {
  if (!document.querySelector(".source-panel") && !document.querySelector(".paper-panel")) {
    document.documentElement.setAttribute("source-panel-mode", "off")
  }

  const switchReaderMode = () => {
    setReaderMode(isReaderMode ? "off" : "on")
  }

  const exitOnEscape = (event: KeyboardEvent) => {
    if (event.key === "Escape" && isReaderMode) {
      setReaderMode("off")
    }
  }

  for (const readerModeButton of document.getElementsByClassName("readermode")) {
    readerModeButton.addEventListener("click", switchReaderMode)
    window.addCleanup(() => readerModeButton.removeEventListener("click", switchReaderMode))
  }
  document.addEventListener("keydown", exitOnEscape)
  window.addCleanup(() => document.removeEventListener("keydown", exitOnEscape))

  // Set initial state
  setReaderMode(isReaderMode ? "on" : "off")
})
