// LensMark — the film. Narration only: no selectors, no URLs, no colors.
// Trimmed to ~450 words for a ~3-minute finished film at am_michael 1.0x.
//
// FALLBACK LINES (only if a scene drops to the fixture engine; swapping means re-running
// narrate -> measure segments -> record for that scene AND a12):
//   a7-alt:  'Now the model's turn, on the image we left blank. Pick a model and an effort
//             level and press Propose — the stream you see here is a canned stand-in, but
//             the live engine takes exactly the same path, phase by phase.'
//   a12-alt: 'Hand, model, and voice — three ways in, one accountable format out. Every
//             frame in this film is a real screen. LensMark.'

export const LENSMARK = {
  id: 'lensmark',
  title: 'LensMark',
  scenes: [
    { id: 'a1-card', text:
      'LensMark is a bench for marking gravitational lens candidates — by hand, by model, ' +
      'and by voice — with every mark accountable to a file on disk.' },

    { id: 'a2-tour', text:
      'A campaign is a folder of cutouts — nine of them here, listed on the left with ' +
      'their annotation state, the stage in the centre, and the working panels on the ' +
      'right. Two images are blank on purpose: one for us to draw by hand, and one, ' +
      'later, for the model.' },

    { id: 'a3-arrow', text:
      'Tools live on single keys. A is the arrow: drag from tail to head, then label it — ' +
      'this tight blue arc is why the system is a candidate. G drags out a galaxy mask, ' +
      'and the dashed circle tells everyone downstream to ignore that neighbour.' },

    { id: 'a4-ring', text:
      'S drops a dotted star mask. R places the Einstein ring — its radius comes from ' +
      'theta E, one and a half arcseconds here, written on the plot. Save commits it all: ' +
      'one JSON file, a log line, and a freshly rendered PNG.' },

    { id: 'a5-render', text:
      'The browser preview is live S V G, but not the deliverable. Toggle Rendered, and ' +
      'this is the canonical PNG — drawn by a separate deterministic renderer from the ' +
      'same geometry, pixel for pixel reproducible.' },

    { id: 'a6-files', text:
      'On disk, an annotated image is three artefacts: the cutout, the JSON, and the ' +
      'rendered overlay, which carries a checksum of the exact JSON that produced it. ' +
      'Edit the file behind its back, and check calls the render stale.' },

    { id: 'a7-propose', text:
      'Now the model’s turn, on the image we left blank. Pick a model and an effort ' +
      'level, and Propose makes a live call to Claude — not a recording, not a mock. ' +
      'The log is the real stream, phase by phase.' },

    { id: 'a8-review', text:
      'Nothing a model draws is committed. Its marks arrive as ghosts, dashed and half ' +
      'opacity, and each takes a human verdict: accept what is right, reject the rest ' +
      'with a named failure mode — spurious, in this case. Submitting the critique ' +
      'writes a review file beside the proposal — the raw material for the numbers.' },

    { id: 'a9-eval', text:
      'Eval folds every critique into one table, grouped by model and effort: precision, ' +
      'recall, spurious mask rate, and mean cost. These rows are real runs on this ' +
      'campaign; every number traces to a submitted critique.' },

    { id: 'a10-voice', text:
      'There is also a voice lane. The mic rides the browser’s speech engine — a film ' +
      'set is a poor room for dictation, so we type the transcript; everything ' +
      'downstream is identical. Send turns the sentence into typed operations, each ' +
      'with a rationale and a confidence, and nothing lands until you apply them.' },

    { id: 'a11-export', text:
      'A reviewed campaign leaves in standard dress: COCO instances for training, D S ' +
      'nine regions for astronomers, binary masks, and a few-shot pack that seeds the ' +
      'next round of proposals.' },

    { id: 'a12-close', text:
      'Hand, model, and voice — three ways in, one accountable format out. Every call in ' +
      'this film was live, and every frame is a real screen. LensMark.' },
  ],
}

export const FILMS = [LENSMARK]
