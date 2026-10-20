import fs from "fs"
import { BuildCtx } from "../../util/ctx"
import { FullSlug, joinSegments, QUARTZ } from "../../util/path"
import { QuartzEmitterPlugin } from "../types"
import { write } from "./helpers"

const verificationFile = "googlef5cd7a2c1b1eac27.html"

export const SiteVerification: QuartzEmitterPlugin = () => ({
  name: "SiteVerification",
  async *emit(ctx) {
    const source = joinSegments(QUARTZ, "static", verificationFile)
    const content = await fs.promises.readFile(source)
    yield write({
      ctx: ctx as BuildCtx,
      content,
      slug: verificationFile.replace(/\.html$/, "") as FullSlug,
      ext: ".html",
    })
  },
  async *partialEmit() {},
})
