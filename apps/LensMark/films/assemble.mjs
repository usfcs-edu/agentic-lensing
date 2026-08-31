/**
 * Pair each scene's video with its narration, then concatenate into one film.
 *
 * ─── WHY THE AUDIO IS PADDED AND THE VIDEO IS NOT TRIMMED ─────────────────────────────
 *
 * The visuals were recorded to the measured narration, so a scene's video should already
 * be at least as long as its audio. Where it is not — the browser flushes a capture a beat
 * early — the AUDIO is padded with silence and the video held on its last frame. A pause
 * reads as pacing; a jump cut reads as a glitch.
 *
 * ─── SUBTITLES ARE A SOFT TRACK, NOT A FILTER ─────────────────────────────────────────
 *
 * `mov_text` is a CODEC rather than a filter, so it works on an ffmpeg built without
 * libass and without libfreetype — which is the build these films keep meeting, and which
 * makes `drawtext` and the `subtitles` filter both unavailable. The film carries a soft
 * subtitle track a player can switch on, and an `.srt` is written beside it for anyone who
 * wants the words without the film.
 *
 * ─── EVERY FILM IS MEASURED BEFORE IT IS CALLED DONE ──────────────────────────────────
 *
 * A concat can succeed and produce something silent, or something whose audio drifted out
 * of sync with its picture. So the finished file is probed for duration against the sum of
 * its scenes, and for mean volume, and refused if either is wrong.
 *
 *   node assemble.mjs [filmId]
 */
import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { FILMS } from './script.mjs'
import { LIMITS, VIDEO } from './film.config.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, 'out')

function ffmpeg(args) {
  return execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...args], { encoding: 'utf8' })
}

function durationOf(path) {
  return Number(
    execFileSync(
      'ffprobe',
      ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
      { encoding: 'utf8' },
    ).trim(),
  )
}

/** Reads BOTH streams — volumedetect reports on stderr. See narrate.mjs. */
function meanVolume(path) {
  const run = spawnSync(
    'ffmpeg',
    ['-hide_banner', '-i', path, '-af', 'volumedetect', '-f', 'null', '-'],
    { encoding: 'utf8' },
  )
  const said = `${run.stdout ?? ''}${run.stderr ?? ''}`
  return Number(/mean_volume:\s*(-?\d+(?:\.\d+)?) dB/.exec(said)?.[1] ?? NaN)
}

/** `HH:MM:SS,mmm`, which is the only timestamp format an `.srt` accepts. */
function srtTime(seconds) {
  const ms = Math.round(seconds * 1000)
  const h = String(Math.floor(ms / 3_600_000)).padStart(2, '0')
  const m = String(Math.floor((ms % 3_600_000) / 60_000)).padStart(2, '0')
  const s = String(Math.floor((ms % 60_000) / 1000)).padStart(2, '0')
  return `${h}:${m}:${s},${String(ms % 1000).padStart(3, '0')}`
}

const timingPath = join(OUT, 'timing.json')
if (!existsSync(timingPath)) throw new Error(`No ${timingPath}. Run: node narrate.mjs`)
const timing = JSON.parse(readFileSync(timingPath, 'utf8'))

const only = process.argv[2]
const films = only ? FILMS.filter(f => f.id === only) : FILMS
if (films.length === 0) throw new Error(`No film called \`${only}\`.`)

for (const film of films) {
  if (!timing[film.id]) throw new Error(`No narration timing for ${film.id}. Run narrate.mjs first.`)

  const dir = join(OUT, film.id)
  const scenesDir = join(dir, 'scenes')
  mkdirSync(scenesDir, { recursive: true })

  const parts = []
  const srt = []
  let elapsed = 0
  let index = 0

  console.log(`${film.title}`)

  for (const scene of film.scenes) {
    const video = join(dir, 'video', `${scene.id}.webm`)
    const audio = join(dir, 'audio', `${scene.id}.wav`)
    if (!existsSync(video)) {
      console.log(`  ${scene.id.padEnd(20)} — not recorded, skipped`)
      continue
    }
    if (!existsSync(audio)) throw new Error(`${scene.id}: recorded but not narrated.`)

    const spoken = durationOf(audio)
    const filmed = durationOf(video)
    const part = join(scenesDir, `${scene.id}.mp4`)

    // `tpad` holds the last frame when the capture came up short; `apad` adds the silence.
    // Together they make every scene exactly as long as its narration plus a little air,
    // without ever cutting a frame the recorder produced.
    const target = spoken + VIDEO.tailSeconds
    ffmpeg([
      '-i', video,
      '-i', audio,
      '-filter_complex',
      `[0:v]tpad=stop_mode=clone:stop_duration=${Math.max(0, target - filmed).toFixed(2)},` +
        `fps=${VIDEO.fps},scale=${VIDEO.width}:${VIDEO.height}[v];` +
        `[1:a]apad=pad_dur=${Math.max(0, target - spoken).toFixed(2)}[a]`,
      '-map', '[v]', '-map', '[a]',
      '-t', target.toFixed(2),
      '-c:v', VIDEO.videoCodec, '-preset', VIDEO.preset, '-crf', String(VIDEO.crf),
      '-pix_fmt', VIDEO.pixelFormat,
      '-c:a', VIDEO.audioCodec, '-b:a', VIDEO.audioBitrate, '-ar', String(VIDEO.audioSampleRate),
      part,
    ])

    index += 1
    // Timed against `spoken`, not `target`, so a caption is gone before the air at the end
    // of its scene rather than hanging over the start of the next one.
    srt.push(`${index}\n${srtTime(elapsed)} --> ${srtTime(elapsed + spoken)}\n${scene.text}\n`)
    elapsed += target
    parts.push(part)
    console.log(`  ${scene.id.padEnd(20)} ${target.toFixed(2).padStart(6)}s`)
  }

  if (parts.length === 0) throw new Error(`${film.id}: nothing was assembled.`)

  // Concat of independently encoded H.264 by stream copy. Legal ONLY because every part was
  // encoded with identical parameters above — which is why those parameters live in one
  // config object rather than being written out per call site.
  const listPath = join(dir, 'concat.txt')
  writeFileSync(listPath, `${parts.map(p => `file '${p}'`).join('\n')}\n`, 'utf8')

  const srtPath = join(dir, `${film.id}.srt`)
  writeFileSync(srtPath, srt.join('\n'), 'utf8')

  const silent = join(dir, '_joined.mp4')
  ffmpeg(['-f', 'concat', '-safe', '0', '-i', listPath, '-c', 'copy', silent])

  const finalPath = join(OUT, `${film.id}.mp4`)
  ffmpeg([
    '-i', silent, '-i', srtPath, '-c', 'copy',
    '-c:s', 'mov_text', '-metadata:s:s:0', 'language=eng', finalPath,
  ])

  // The sidecar is written BESIDE THE FILM, in the same step that makes the film.
  //
  // It used to live only in the per-film directory, and a re-cut therefore left whatever
  // `.srt` happened to be next to the `.mp4` untouched. That is exactly what happened once:
  // a re-cut corrected a line of narration, the embedded `mov_text` track carried the
  // correction, and the sidecar sat there three hours stale still carrying the words the
  // re-cut existed to remove — right next to the film, with the obvious name.
  //
  // A subtitle file that disagrees with its film is worse than none, because it is a
  // transcript: somebody quoting it is quoting the film. Writing both here means they
  // cannot drift.
  writeFileSync(join(OUT, `${film.id}.srt`), srt.join('\n'), 'utf8')

  const length = durationOf(finalPath)
  const db = meanVolume(finalPath)
  const drift = Math.abs(length - elapsed)

  if (!Number.isFinite(db) || db < LIMITS.silenceDb) {
    throw new Error(`${film.id}: the finished film measures ${db} dBFS. That is silence.`)
  }
  if (drift > LIMITS.driftSeconds) {
    throw new Error(
      `${film.id}: finished at ${length.toFixed(2)}s but its scenes sum to ${elapsed.toFixed(2)}s. ` +
        'A concat that loses or repeats a scene does not fail — it just produces the wrong film.',
    )
  }

  console.log(`  ${'—'.repeat(34)}`)
  console.log(`  ${length.toFixed(1)}s, ${db} dBFS, ${parts.length} scene(s)`)
  console.log(`  film      ${finalPath}`)
  console.log(`  subtitles ${join(OUT, `${film.id}.srt`)}\n`)
}
