/**
 * Render the pages the recorder films: terminal output, title cards, and slides.
 *
 * (In the pipelines this generalizes, this file was called `terminal.mjs` and rendered only
 * terminal output. It grew a title card, and the name stopped being true.)
 *
 * The discipline this encodes is the reason it exists: the tool is run for real, every byte
 * it printed is kept, and this file chooses only a font and a background. If a command
 * prints something awkward, the film shows it. Nothing here composes output by hand.
 *
 * ─── PROGRESSIVE REVEAL ───────────────────────────────────────────────────────────────
 * A CLI demo reads far better when its lines arrive as the narrator reaches them than when
 * a finished wall of text sits on screen for twelve seconds. `window.__reveal(seconds)`
 * walks the content over the scene's MEASURED narration length — so the pictures are still
 * paced to the words, which is the rule the whole pipeline exists to serve.
 *
 * Every page defines the same three globals so the recorder does not care which kind it is:
 *   window.__reveal(seconds)  → pace the content across that many seconds, returns the count
 *   window.__revealAll()      → show everything at once
 *   window.__lineCount        → how many revealable elements exist
 */
import { execFileSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'

import { THEME } from './film.config.mjs'

// Matches every CSI escape, not just the colour ones: a demo that repaints a progress bar
// emits cursor moves too, and a browser renders those as literal garbage.
const ANSI = new RegExp(String.fromCharCode(27) + '\\[[0-9;?]*[A-Za-z]', 'g')
const CR_PROGRESS = /\r(?!\n)/g

/** Strip ANSI escapes and carriage-return progress redraws. */
export function plain(text) {
  return String(text).replace(ANSI, '').replace(CR_PROGRESS, '\n')
}

/**
 * Run a command and keep everything it said, including on failure.
 *
 * A non-zero exit is captured rather than thrown — a tool that exits non-zero to report
 * something (a gate refusing an export, an API returning 402) is exactly the run worth
 * filming.
 */
export function capture(command, args, options = {}) {
  try {
    return plain(execFileSync(command, args, {
      encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, ...options,
    }))
  } catch (error) {
    return plain(`${error.stdout ?? ''}${error.stderr ?? ''}`)
  }
}

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }
const esc = s => String(s).replace(/[&<>"]/g, ch => ESCAPES[ch])

/**
 * The bundled webfont, served from the same origin by serve.mjs.
 *
 * `font-display: block` on purpose. The default, `swap`, paints a fallback first and
 * restyles when the font arrives — which in a video means the first frames of a scene are
 * in the wrong typeface. Blocking is invisible here because the file is on loopback.
 *
 * If setup.sh could not fetch the font, the @font-face simply fails to load and the stack
 * falls through to the system default. The film still builds; it is just no longer
 * pixel-identical across platforms.
 */
const FONT_FACE = `
  @font-face { font-family: "JetBrains Mono"; font-display: block; font-weight: 400;
               src: url("/assets/fonts/JetBrainsMono-Regular.woff2") format("woff2") }
  @font-face { font-family: "JetBrains Mono"; font-display: block; font-weight: 700;
               src: url("/assets/fonts/JetBrainsMono-Bold.woff2") format("woff2") }`

/** The reveal machinery, identical on every page kind. */
const revealScript = (staticFirst) => `
<script>
  const lines = [...document.querySelectorAll('.ln')]
  // Static by default so a still frame is never blank; the recorder opts into pacing.
  if (${staticFirst ? 'true' : 'false'}) lines.forEach(l => l.classList.add('on'))
  window.__reveal = (seconds) => {
    // Finish a beat early so the last line is readable before the scene cuts.
    const span = Math.max(${THEME.revealMinSeconds}, seconds * ${THEME.revealFraction}) * 1000
    const step = lines.length > 1 ? span / (lines.length - 1) : 0
    lines.forEach((l, i) => setTimeout(() => l.classList.add('on'), i * step))
    return lines.length
  }
  window.__revealAll = () => { lines.forEach(l => l.classList.add('on')); return lines.length }
  window.__lineCount = lines.length
</script>`

const BASE_CSS = `
  :root { color-scheme: dark }
  ${FONT_FACE}
  html, body { margin: 0; height: 100%; background: ${THEME.background}; overflow: hidden }
  .ln { display: block; opacity: 0; transition: opacity .18s ease }
  .ln.on { opacity: 1 }`

/**
 * Write a terminal-looking page.
 *
 * `prompt` is shown above the output so a viewer can see WHICH command produced it — a wall
 * of text with no command above it asks the audience to take the source on trust.
 */
export function writeTerminal(path, {
  prompt, output, highlight = [], dim = [], header = THEME.defaultHeader, reveal = true,
}) {
  const lines = plain(output).replace(/\s+$/, '').split('\n')
  const rendered = lines.map(line => {
    const cls = highlight.some(n => line.includes(n)) ? 'hi'
      : dim.some(n => line.includes(n)) ? 'dim' : ''
    // A blank line with no content collapses to zero height and the output jumps.
    return `<span class="ln ${cls}">${esc(line) || '&nbsp;'}</span>`
  // JOINED WITH NOTHING, not with a newline. The spans are `display:block` so they stack
  // on their own, and the container is `white-space: pre-wrap` so that leading indentation
  // inside a line survives — which means a newline BETWEEN them is preserved too and
  // renders as an extra blank line. The pipelines this generalizes both double-spaced
  // every terminal shot for exactly this reason, halving how much output fits in a frame.
  }).join('')

  const dots = THEME.dots
    .map((c, i) => `<span class="dot" style="background:${c}"></span>`)
    .join('')

  writeFileSync(path, `<!doctype html>
<meta charset="utf-8">
<title>${esc(header)}</title>
<style>
  ${BASE_CSS}
  body { display: flex; flex-direction: column;
         font: ${THEME.fontSize}/${THEME.lineHeight} ${THEME.fontStack} }
  header { padding: 13px 22px; background: ${THEME.headerBackground}; color: ${THEME.headerText};
           border-bottom: 1px solid ${THEME.border}; display: flex; gap: 9px; align-items: center;
           font-size: 13px; letter-spacing: .02em; flex: 0 0 auto }
  .dot { width: 11px; height: 11px; border-radius: 50% }
  main { padding: 20px 26px; flex: 1; overflow: hidden }
  .prompt { color: ${THEME.prompt}; white-space: pre-wrap; margin: 0 0 14px }
  .prompt::before { content: ${JSON.stringify(THEME.promptPrefix)}; color: ${THEME.promptSigil} }
  pre { margin: 0; color: ${THEME.text}; white-space: pre-wrap; word-break: break-word }
  .hi { color: ${THEME.highlight} }
  .dim { color: ${THEME.dim} }
</style>
<header>${dots}<span>${esc(header)}</span></header>
<main>${prompt ? `<p class="prompt">${esc(prompt)}</p>` : ''}<pre id="out">${rendered}</pre></main>
${revealScript(!reveal)}
`, 'utf8')
  return path
}

/** A full-bleed title card, for the opening beat of a film. */
export function writeCard(path, { title, subtitle, note }) {
  writeFileSync(path, `<!doctype html>
<meta charset="utf-8"><title>${esc(title)}</title>
<style>
  ${BASE_CSS}
  body { display:flex; flex-direction:column; justify-content:center; padding:0 92px;
         color:${THEME.cardTitle}; font-family:${THEME.uiFontStack} }
  h1 { font-size:${THEME.cardTitleSize}; line-height:1.1; margin:0 0 18px;
       font-weight:650; letter-spacing:-.02em }
  h2 { font-size:${THEME.cardSubtitleSize}; line-height:1.35; margin:0;
       font-weight:400; color:${THEME.cardSubtitle} }
  .note { margin-top:34px; font:14px/1.6 ${THEME.fontStack}; color:${THEME.cardNote} }
  .rule { width:64px; height:3px; background:${THEME.cardAccent}; margin:0 0 30px; border-radius:2px }
  /* A card is one beat, not a list — it is shown whole rather than revealed. */
  .ln { opacity: 1 }
</style>
<div class="rule"></div>
<h1>${esc(title)}</h1>
${subtitle ? `<h2>${esc(subtitle)}</h2>` : ''}
${note ? `<div class="note">${esc(note)}</div>` : ''}
${revealScript(true)}
`, 'utf8')
  return path
}

/**
 * A slide: a heading, and either bullets revealed one at a time or a block of body text.
 *
 * The bullets are the revealable units, so `hold()`/`__reveal()` paces a slide exactly the
 * way it paces terminal output — one mechanism, three page kinds.
 *
 * `image` is a path RELATIVE TO THE PAGES DIRECTORY. Copy the file there in a prepare step;
 * an absolute path will not resolve, because the page is served over loopback rather than
 * opened from disk.
 */
export function writeSlide(path, {
  title, subtitle, bullets = [], body = '', image = null, note = null, reveal = true,
}) {
  const items = bullets.map(b => `<li class="ln">${esc(b)}</li>`).join('\n')
  const bodyLines = body
    ? plain(body).trim().split('\n').map(l => `<p class="ln">${esc(l) || '&nbsp;'}</p>`).join('\n')
    : ''

  writeFileSync(path, `<!doctype html>
<meta charset="utf-8"><title>${esc(title)}</title>
<style>
  ${BASE_CSS}
  body { display:flex; flex-direction:column; justify-content:center; padding:56px 92px;
         color:${THEME.cardTitle}; font-family:${THEME.uiFontStack} }
  h1 { font-size:40px; line-height:1.15; margin:0 0 10px; font-weight:650; letter-spacing:-.02em }
  h2 { font-size:21px; line-height:1.35; margin:0 0 30px; font-weight:400; color:${THEME.cardSubtitle} }
  .rule { width:64px; height:3px; background:${THEME.cardAccent}; margin:0 0 26px; border-radius:2px }
  ul { margin:0; padding:0; list-style:none; font-size:25px; line-height:1.55 }
  li { margin:0 0 16px; padding-left:30px; position:relative; color:${THEME.text} }
  li::before { content:"—"; position:absolute; left:0; color:${THEME.cardAccent} }
  p { font-size:21px; line-height:1.6; margin:0 0 12px; color:${THEME.text} }
  img { max-width:100%; max-height:52vh; object-fit:contain; margin-top:22px; border-radius:6px }
  .note { margin-top:30px; font:14px/1.6 ${THEME.fontStack}; color:${THEME.cardNote} }
</style>
<div class="rule"></div>
<h1>${esc(title)}</h1>
${subtitle ? `<h2>${esc(subtitle)}</h2>` : ''}
${items ? `<ul>${items}</ul>` : ''}
${bodyLines}
${image ? `<img class="ln" src="${esc(image)}" alt="">` : ''}
${note ? `<div class="note">${esc(note)}</div>` : ''}
${revealScript(!reveal)}
`, 'utf8')
  return path
}
