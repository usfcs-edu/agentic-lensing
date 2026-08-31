/**
 * Mechanism: turn a scene's shot spec into a page the recorder can film.
 *
 * `script.mjs` holds the words and no visuals at all, deliberately — it is reviewable by
 * somebody who will never read this file, and the CLAIMS are the part that needs reviewing.
 * `recipes.mjs` is the other half: which span of which real run appears on screen while
 * each sentence is spoken. This file is the machinery joining them, and holds no content.
 *
 * ─── EVERYTHING TERMINAL IS A SLICE OF A REAL RUN ─────────────────────────────────────
 * `capture.mjs` runs each target once and keeps every byte. A scene names a SECTION of that
 * capture; this file cuts it out and renders it. No scene composes output by hand, and no
 * scene re-runs a tool to get a nicer answer. If a number in the narration is wrong, the
 * frame behind it disagrees — which is the property that makes the narration checkable
 * rather than merely confident.
 */
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { writeCard, writeSlide, writeTerminal } from './pages.mjs'
import { CAPTURE, THEME } from './film.config.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const CAPTURES = join(HERE, 'out', 'captures')

function loadCapture(name) {
  const p = join(CAPTURES, `${name}.txt`)
  if (!existsSync(p)) {
    throw new Error(
      `No capture at ${p}. The films are cut from real runs — produce it first:\n` +
      `    node capture.mjs ${name}`)
  }
  return readFileSync(p, 'utf8').split('\n')
}

/**
 * Cut out one section of a capture.
 *
 * `from` is a substring of the section's heading line. The section runs to the next heading
 * (see CAPTURE.heading in film.config.mjs), or to `to` if given. Decorative rules at either
 * end are trimmed so the frame opens on content rather than on a line of box characters.
 */
function section(lines, from, { to, max = 30, keepHeading = true } = {}) {
  const start = lines.findIndex(l => l.includes(from))
  if (start < 0) throw new Error(`capture has no section matching ${JSON.stringify(from)}`)
  let end = lines.length
  for (let i = start + 1; i < lines.length; i += 1) {
    if (to ? lines[i].includes(to) : CAPTURE.heading.test(lines[i])) { end = i; break }
  }
  let out = lines.slice(keepHeading ? start : start + 1, end)
  while (out.length && CAPTURE.rule.test(out[0])) out = out.slice(1)
  while (out.length && CAPTURE.rule.test(out[out.length - 1])) out = out.slice(0, -1)
  // Very long sections read as a wall; keep the head, which is where the claim lives.
  if (out.length > max) out = [...out.slice(0, max - 1), `  … ${out.length - max + 1} more lines`]
  return out.join('\n')
}

/** Lines matching any needle, in capture order — for a scene that quotes scattered figures. */
function grepLines(lines, needles, { max = 24, context = 0 } = {}) {
  const keep = new Set()
  lines.forEach((l, i) => {
    if (needles.some(n => l.includes(n))) {
      for (let k = Math.max(0, i - context); k <= Math.min(lines.length - 1, i + context); k += 1) {
        keep.add(k)
      }
    }
  })
  const idx = [...keep].sort((a, b) => a - b).slice(0, max)
  if (!idx.length) throw new Error(`capture has no lines matching ${JSON.stringify(needles)}`)
  const out = []
  let prev = -2
  for (const i of idx) {
    // An ellipsis where lines were skipped, so the frame does not imply the output was
    // contiguous when it was not.
    if (prev >= 0 && i > prev + 1) out.push('  …')
    out.push(lines[i]); prev = i
  }
  return out.join('\n')
}

// ── the registries ───────────────────────────────────────────────────────────────────────
//
// Filled by recipes.mjs. Each scene is one of:
//   { kind: 'card',     title, subtitle, note }
//   { kind: 'slide',    title, subtitle, bullets|body, image, note }
//   { kind: 'terminal', prompt, from|needles, to, highlight, dim, max, context, header }
//   { kind: 'web',      url, ready }        — filmed live against a running application
// `kind` defaults to 'terminal'.
export const RECIPES = {}

/** Register a film's shots. Called from recipes.mjs so this file stays mechanism-only. */
export function defineShots(filmId, spec) { RECIPES[filmId] = spec }

/**
 * Per-scene browser choreography, keyed [filmId][sceneId].
 *
 * The value is a BARE FUNCTION EXPRESSION as a string — `async (page, hold, inFrame) => {…}`
 * — because it is serialized into the browser, not run here. See record.mjs and
 * references/AUTHORING.md. Most scenes need none of this; it is the escape hatch for
 * driving a live UI.
 */
export const ACTS = {}
export function defineAction(filmId, sceneId, source) {
  ACTS[filmId] = ACTS[filmId] || {}
  ACTS[filmId][sceneId] = source
}

/**
 * Build every page a film needs.
 * Returns { [sceneId]: { page | url, reveal, ready } } for the recorder.
 */
export async function prepareFilm(film, { pagesDir }) {
  const recipe = RECIPES[film.id]
  if (!recipe) throw new Error(`No shots defined for film \`${film.id}\` (see recipes.mjs)`)

  // Loaded lazily: a film made entirely of slides and live pages needs no capture at all,
  // and demanding one would make the slide path depend on a CLI it never films.
  let lines = null
  const capture = () => {
    if (lines === null) {
      if (!recipe.capture) {
        throw new Error(`${film.id}: a terminal shot needs \`capture\` on the film's recipe.`)
      }
      lines = loadCapture(recipe.capture)
    }
    return lines
  }

  const prepared = {}
  for (const scene of film.scenes) {
    const spec = recipe.scenes[scene.id]
    // A scene with words and no picture is a hard failure, by design: the alternative is a
    // film that silently drops a sentence somebody wrote and reviewed.
    if (!spec) throw new Error(`${film.id}/${scene.id}: no shot in recipes.mjs`)

    const page = `${scene.id}.html`
    const path = join(pagesDir, page)

    if (spec.kind === 'web') {
      if (!spec.url) throw new Error(`${film.id}/${scene.id}: a web shot needs a \`url\`.`)
      prepared[scene.id] = { url: spec.url, reveal: false, ready: spec.ready }
      continue
    }

    if (spec.kind === 'card') {
      writeCard(path, spec)
      prepared[scene.id] = { page, reveal: false }
      continue
    }

    if (spec.kind === 'slide') {
      writeSlide(path, { ...spec, reveal: spec.reveal !== false })
      prepared[scene.id] = { page, reveal: spec.reveal !== false }
      continue
    }

    const body = spec.needles
      ? grepLines(capture(), spec.needles, { max: spec.max ?? 24, context: spec.context ?? 0 })
      : section(capture(), spec.from, { to: spec.to, max: spec.max ?? 30 })
    writeTerminal(path, {
      prompt: spec.prompt,
      output: body,
      highlight: spec.highlight ?? [],
      dim: spec.dim ?? [],
      header: spec.header ?? recipe.header ?? THEME.defaultHeader,
      reveal: spec.reveal !== false,
    })
    prepared[scene.id] = { page, reveal: spec.reveal !== false }
  }
  return prepared
}

export { loadCapture, section, grepLines }
