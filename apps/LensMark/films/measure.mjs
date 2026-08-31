/**
 * Measure a film, so that pacing and claims are checked rather than estimated.
 *
 *   node measure.mjs segments <film> <scene>
 *       Every speech segment in that scene's narration, with the `hold(f)` fraction each
 *       one starts at. This is where act fractions come from. Do not guess them.
 *
 *   node measure.mjs fraction <film> <scene> "<phrase>"
 *       The fraction at which a phrase is spoken, from its CHARACTER position in the text.
 *       Character position is a better proxy for time than word position, because words
 *       differ in length and the model speaks characters.
 *
 *   node measure.mjs evidence <film> <scene> "<phrase>" [out.png]
 *       Cut the frame from the FINISHED film at the moment that phrase is spoken. This is
 *       the check for copy rule 1 — is the figure actually on screen when it is said? A
 *       screenshot taken beside the take proves nothing about the film that shipped.
 *
 *   node measure.mjs frame <film> <seconds> [out.png]
 *       A frame at an absolute position in the finished film.
 */
import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { FILMS } from './script.mjs'
import { VIDEO } from './film.config.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, 'out')

const [mode, filmId, ...rest] = process.argv.slice(2)
if (!mode) { console.log(readFileSync(new URL(import.meta.url)).toString().split('*/')[0]); process.exit(1) }

const film = FILMS.find(f => f.id === filmId)
if (!film) throw new Error(`No film called \`${filmId}\`. Known: ${FILMS.map(f => f.id).join(', ')}`)

const timing = JSON.parse(readFileSync(join(OUT, 'timing.json'), 'utf8'))[film.id]
if (!timing) throw new Error(`No timing for ${film.id}. Run: node narrate.mjs ${film.id}`)

const sceneOf = id => {
  const s = film.scenes.find(x => x.id === id)
  if (!s) throw new Error(`No scene \`${id}\` in ${film.id}. Known: ${film.scenes.map(x => x.id).join(', ')}`)
  return s
}
const secondsOf = id => timing.scenes.find(s => s.id === id)?.seconds
  ?? (() => { throw new Error(`${id} has no measured length`) })()

/**
 * Where a scene STARTS in the finished film.
 *
 * NOT the sum of narration lengths. Every scene carries an extra `tailSeconds` of air, so
 * by scene ten the offset from timing.json alone is five seconds wrong — which is enough to
 * cut a frame from the wrong shot entirely and conclude the film is fine.
 */
function offsetOf(sceneId) {
  let elapsed = 0
  for (const s of film.scenes) {
    if (s.id === sceneId) return elapsed
    const secs = timing.scenes.find(x => x.id === s.id)?.seconds
    if (secs) elapsed += secs + VIDEO.tailSeconds
  }
  throw new Error(`${sceneId} is not in ${film.id}`)
}

function cutFrame(atSeconds, dest) {
  const source = join(OUT, `${film.id}.mp4`)
  if (!existsSync(source)) throw new Error(`No ${source}. Run: node assemble.mjs ${film.id}`)
  mkdirSync(dirname(dest), { recursive: true })
  // `-ss` BEFORE `-i` so ffmpeg SEEKS rather than decoding everything up to the offset.
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', atSeconds.toFixed(3),
                          '-i', source, '-frames:v', '1', dest])
  return dest
}

if (mode === 'segments') {
  const sceneId = rest[0]
  const wav = join(OUT, film.id, 'audio', `${sceneId}.wav`)
  if (!existsSync(wav)) throw new Error(`No ${wav}. Run: node narrate.mjs ${film.id}`)
  const total = secondsOf(sceneId)

  const run = spawnSync('ffmpeg',
    ['-hide_banner', '-i', wav, '-af', 'silencedetect=noise=-40dB:d=0.20', '-f', 'null', '-'],
    { encoding: 'utf8' })
  const said = `${run.stdout ?? ''}${run.stderr ?? ''}`

  // silencedetect reports the GAPS; speech is what is left between them.
  const gaps = []
  const re = /silence_start:\s*(-?[\d.]+)[\s\S]*?silence_end:\s*([\d.]+)/g
  let m
  while ((m = re.exec(said)) !== null) gaps.push([Number(m[1]), Number(m[2])])

  const speech = []
  let cursor = 0
  for (const [gs, ge] of gaps) {
    if (gs > cursor + 0.05) speech.push([cursor, gs])
    cursor = ge
  }
  if (cursor < total - 0.05) speech.push([cursor, total])

  console.log(`${film.id}/${sceneId} — ${total.toFixed(2)}s\n`)
  console.log(`  ${sceneOf(sceneId).text}\n`)
  console.log('  start     end       hold(f)')
  for (const [a, b] of speech) {
    console.log(`  ${a.toFixed(2).padStart(6)}s  ${b.toFixed(2).padStart(6)}s   ${(a / total).toFixed(3)}`)
  }
  console.log('\n  Use the start fractions. End an act on a deliberate hold, ceiling ~0.86.')

} else if (mode === 'fraction' || mode === 'evidence') {
  const [sceneId, phrase, dest] = rest
  const scene = sceneOf(sceneId)
  const at = scene.text.toLowerCase().indexOf(String(phrase).toLowerCase())
  if (at < 0) {
    throw new Error(`${sceneId} does not say ${JSON.stringify(phrase)}.\n\n  ${scene.text}`)
  }
  const fraction = at / scene.text.length
  const seconds = secondsOf(sceneId)
  console.log(`${film.id}/${sceneId}`)
  console.log(`  phrase    ${JSON.stringify(phrase)}`)
  console.log(`  at        character ${at} of ${scene.text.length}`)
  console.log(`  fraction  ${fraction.toFixed(3)}   (${(fraction * seconds).toFixed(2)}s into the scene)`)

  if (mode === 'evidence') {
    const abs = offsetOf(sceneId) + fraction * seconds
    const path = cutFrame(abs, dest || join(OUT, 'evidence', `${film.id}-${sceneId}.png`))
    console.log(`  film      ${abs.toFixed(2)}s`)
    console.log(`  frame     ${path}`)
    console.log('\n  Open it. Is every figure in that sentence visible in that frame?')
  }

} else if (mode === 'frame') {
  const [secondsArg, dest] = rest
  const at = Number(secondsArg)
  if (!Number.isFinite(at)) throw new Error(`\`${secondsArg}\` is not a number of seconds`)
  console.log(cutFrame(at, dest || join(OUT, 'evidence', `${film.id}-${at}s.png`)))

} else {
  throw new Error(`Unknown mode \`${mode}\`. One of: segments, fraction, evidence, frame`)
}
