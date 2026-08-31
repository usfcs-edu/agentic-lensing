/**
 * Everything about a film that is a choice rather than a mechanism.
 *
 * The two pipelines this generalizes had these values scattered as literals across five
 * files — 720p was stated three times in two files with no shared constant, and the theme
 * was seventeen hex codes buried in a template string. Porting to a new product meant
 * hunting them down. They live here now, and nothing else in the pipeline hardcodes one.
 *
 * Nothing in this file needs to change to make the pipeline WORK. Change it to make the
 * film look and sound like yours.
 */
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))

// ── the voice ────────────────────────────────────────────────────────────────────────────
//
// Kokoro-82M, Apache-2.0, running locally on the CPU under onnxruntime. Nothing is sent
// anywhere: the weights are on disk and the narration text never leaves the machine, which
// matters when the script reads out a customer's figures.
//
// `langCode` selects the phonemizer and MUST agree with the voice prefix — `af_`/`am_` are
// American ('a'), `bf_`/`bm_` British ('b'). Mismatch them and you get a voice reading with
// the wrong accent's vowel set, which sounds like a bad take rather than a bug.
//
// The pack holds 54 voices, 28 of them English and reachable here. `node audition.mjs list`
// prints them from voices-v1.0.bin itself, rather than from a copy in this comment that
// would go stale.
//
// AUDITION BEFORE COMMITTING — `node audition.mjs` speaks one line of your own script in
// several voices and reports what each does to the running time. A voice is hard to change
// once a film is cut, because every scene's timing is measured from it.
//
// The three fields can be overridden for one run without editing this file, which is how a
// second cut of the same film gets a different voice:
//
//     FILM_VOICE=bm_george FILM_LANG=b node narrate.mjs <film>
//
// FILM_LANG must move WITH FILM_VOICE, for the reason above.
export const VOICE = {
  voice: process.env.FILM_VOICE || 'am_michael',
  // 1.0 is a touch brisk for technical narration
  speed: Number(process.env.FILM_VOICE_SPEED || 1.0),
  langCode: process.env.FILM_LANG || 'a',
}

// ── the picture ───────────────────────────────────────────────────────────────────────────
//
// `width`/`height` are load-bearing in two places at once: the browser viewport and the
// video size. They must match, or an element that `inFrame()` reports as visible sits
// outside the recorded picture. Change them here and both follow.
//
// 720p is deliberate — large enough to read a form field, small enough to email.
export const VIDEO = {
  width: 1920,
  height: 1080,
  fps: 25,
  // libx264 at CRF 20 / preset medium: visually lossless for flat UI and terminal text,
  // and small. Every scene MUST be encoded identically, because the final concat is a
  // stream copy and a copy across differing parameters produces a broken file.
  videoCodec: 'libx264',
  preset: 'medium',
  crf: 20,
  pixelFormat: 'yuv420p',
  audioCodec: 'aac',
  audioBitrate: '160k',
  audioSampleRate: 48000,
  // A half-second of air after each scene's narration. Without it every cut lands on the
  // last syllable and the film feels rushed.
  tailSeconds: 0.5,
}

// ── the checks that refuse a bad take ───────────────────────────────────────────────────────
//
// Both of these exist because of a specific failure, not as a precaution.
export const LIMITS = {
  // A model can load, report success, and emit silence. Without this a four-minute film
  // ships saying nothing, and nothing upstream reports an error.
  silenceDb: -50,
  // A concat that loses or repeats a scene does not fail — it produces the wrong film. So
  // the finished duration is compared against the sum of its parts.
  driftSeconds: 1.5,
  // A capture flushed early is padded by assemble.mjs, but a gross mismatch is a bad take
  // and should be visible in the recorder's output.
  shortTakeRatio: 0.6,
}

// ── runtime ────────────────────────────────────────────────────────────────────────────────
export const RUNTIME = {
  // playwright-cli keeps one browser per session name. Give each project its own, or two
  // pipelines running at once will fight over the same browser.
  session: 'films',
  // The loopback port the generated pages are served on. Change it if 8099 is taken.
  port: Number(process.env.FILM_PORT || 8099),
  // The interpreter holding kokoro-onnx. setup.sh builds this; FILM_PYTHON overrides it.
  python: process.env.FILM_PYTHON || join(HERE, '.venv', 'bin', 'python'),
  // Where kokoro-v1.0.onnx and voices-v1.0.bin live. setup.sh symlinks this at a
  // machine-wide cache so several projects share one 337 MB download.
  models: process.env.FILM_MODELS || join(HERE, 'models'),
  // Left empty, record.mjs looks for a local install then falls back to PATH.
  playwrightCli: process.env.FILM_PLAYWRIGHT_CLI || '',
}

// ── the look ────────────────────────────────────────────────────────────────────────────────
//
// `fontStack` puts a bundled woff2 first on purpose. "SF Mono" and Menlo are macOS-only, so
// a stack that leads with them renders one film on a laptop and a different one in CI. The
// font is served from `assets/fonts/` by serve.mjs; if setup.sh could not fetch it, the
// stack degrades to the system default and everything still works — it is just no longer
// reproducible across platforms.
export const THEME = {
  fontStack: '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, "DejaVu Sans Mono", monospace',
  uiFontStack: '-apple-system, "SF Pro Display", "Segoe UI", Inter, system-ui, sans-serif',
  fontSize: '22px',
  lineHeight: 1.6,

  background: '#0f1117',
  headerBackground: '#171a23',
  headerText: '#9aa4bf',
  border: '#242938',

  // The three window dots. Set `dots: []` for a plain title bar.
  dots: ['#e0575b', '#e0a34a', '#4fb477'],

  prompt: '#7fd1ff',
  promptSigil: '#4a5570',
  promptPrefix: '$ ',
  text: '#d6dbe8',
  highlight: '#4fe08a',   // lines named in a shot's `highlight`
  dim: '#6b748c',         // lines named in a shot's `dim`

  // Title cards.
  cardTitle: '#e8ecf5',
  cardSubtitle: '#8b95ad',
  cardNote: '#5c657d',
  cardAccent: '#4fe08a',
  cardTitleSize: '68px',
  cardSubtitleSize: '32px',

  // Shown in the window title bar of every terminal shot unless a film overrides it.
  defaultHeader: 'terminal',

  // Progressive reveal. Lines arrive across this fraction of the scene's narration, so the
  // last line is on screen and readable before the cut. Set `reveal: false` on a shot to
  // show everything at once.
  revealFraction: 0.82,
  revealMinSeconds: 0.3,
}

// ── what gets captured ──────────────────────────────────────────────────────────────────────
//
// Only used by terminal films. `capture.mjs` runs each target ONCE and keeps every byte;
// `recipes.mjs` then names which span of that output each scene shows. Nothing between the
// run and the frame can edit the text, which is the property that makes a spoken figure
// checkable rather than merely confident.
export const CAPTURE = {
  // Commands run relative to this. Default: the directory holding the films dir.
  root: join(HERE, '..'),

  // REPLACE THIS. It exists so that a freshly scaffolded project produces a complete,
  // real film on the first run — which is the most convincing proof the install works —
  // and so the example recipe has something true to slice.
  targets: [
    { name: 'lensmark', dir: '.', cmd: 'bash', args: ['films/capture-demo.sh'] },
  ],

  // Handed to every captured command. Demos often sleep for human readability; a film
  // paces itself from the narration, so the theatrical pauses come out and the colour
  // codes are suppressed at source rather than stripped afterwards.
  env: { NO_COLOR: '1', FORCE_COLOR: '0', DEMO_PAUSE: '0', DEMO_HOLD: '0' },

  // How `section()` recognizes where one part of the output ends and the next begins.
  // Widen it to match whatever your tools actually print — the alternative is editing
  // their output to fit the regex, which forfeits the whole point of capturing it.
  //   `3 · TITLE`  ·  `STEP 3  TITLE`  ·  `=== TITLE ===`  ·  `## TITLE`
  heading: /^\s*(?:\d+(?:[.\-]\d+)?\s+[·.]\s+\S|STEP\s+\d+\b|={3,}\s+\S.*={3,}\s*$|#{2,}\s+\S)/,
  // Box-drawing rules printed around headings, trimmed off so a frame opens on content.
  rule: /^[\s═─━=_.-]*$/,
}
