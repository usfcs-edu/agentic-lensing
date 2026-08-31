"""
Render the narration with Kokoro-82M — a local neural voice.

Reads one JSON request on stdin, writes one WAV per scene, and prints one JSON line per
scene giving its measured length. `narrate.mjs` owns the pipeline; this file owns nothing
but the synthesis.

─── THE PROTOCOL IS THE SEAM ───────────────────────────────────────────────────────────
stdin:   {"voice": str, "speed": float, "langCode": "a"|"b", "items": [{id, text, path}]}
stdout:  one JSON object per line — {"id", "seconds", "path"} or {"id", "error"}

One side door: `speak.py --list-voices` reads nothing and prints {"voices": [...]} — the
names in the pack, so a caller can offer them and reject a typo without loading 337 MB of
weights first. It lives here because this file already owns the pack's format; a second
reader of it elsewhere would be a second thing to keep true.

Everything Kokoro-specific is below that line and nothing above it. When the original
pipeline swapped the PyTorch runtime for this one, `narrate.mjs` needed no change at all;
that is what a seam is for. To use a different engine, rewrite this file and keep the
protocol.

─── WHY kokoro-onnx AND NOT THE PYTORCH `kokoro` PACKAGE ───────────────────────────────
Same model, same voices, different runtime. The PyTorch package reaches `kokoro ->
misaki[en] -> spacy -> thinc -> blis`, and blis has no wheel for recent Pythons, so pip
tries to compile it and dies inside Cython. `kokoro-onnx` runs the same 82M weights under
onnxruntime with an espeak-based phonemizer and installs clean on 3.10 through 3.13.

─── WHY A PYTHON SIDECAR AND NOT A NODE LIBRARY ────────────────────────────────────────
There is no usable neural TTS in the Node ecosystem that runs offline. Node shells out to
this exactly ONCE per run, not once per scene: loading the model costs about a second, and
a film of twenty scenes would otherwise pay it twenty times.

Nothing leaves the machine. The weights are on disk and no text is sent anywhere — which
matters, because the narration names a customer's documents and figures.

─── THERE IS NO `say` FALLBACK, DELIBERATELY ───────────────────────────────────────────
macOS `say` is concatenative: it stitches recorded units together, has no prosody model,
and lands every sentence with the same shape. A film narrated by it sounds like a phone
tree. A fallback nobody is allowed to use is better expressed as an error.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

# setup.sh points this at a machine-wide cache so several projects share one download.
MODELS = Path(os.environ.get("FILM_MODELS") or (Path(__file__).resolve().parent / "models"))


def find_weights():
    """
    Locate the model and voice pack.

    Globbed rather than named so the fp16 and int8 builds work unchanged — they are the
    same voices at a third of the size, and worth reaching for on a machine where the
    337 MB full model is a problem.
    """
    if not MODELS.is_dir():
        return None, None
    models = sorted(p for p in MODELS.glob("*.onnx"))
    voices = sorted(p for p in MODELS.glob("voices*.bin"))
    # Prefer the full-precision model when several are present; it is the one the voice
    # was tuned against and the quantized builds are audibly rougher on sibilants.
    full = [p for p in models if ".fp16" not in p.name and ".int8" not in p.name]
    return (full or models or [None])[0], (voices or [None])[0]


def list_voices(voices) -> int:
    """
    Every voice name in the pack, without instantiating the model.

    An npz is a zip of `<name>.npy`, so numpy — already a dependency — reads the names
    from the archive index alone. That is the whole point: the answer costs a moment,
    not the second-plus a model load costs, so a caller can check a name up front.
    """
    with np.load(voices, allow_pickle=True) as pack:
        names = sorted(pack.files)
    print(json.dumps({"voices": names}), flush=True)
    return 0


def main() -> int:
    model, voices = find_weights()
    if model is None or voices is None:
        print(json.dumps({"id": "*", "error":
            f"no Kokoro weights under {MODELS}. Run scripts/setup.sh, or set FILM_MODELS "
            f"to a directory holding kokoro-v1.0.onnx and voices-v1.0.bin."}), flush=True)
        return 1

    if "--list-voices" in sys.argv[1:]:
        return list_voices(voices)

    request = json.load(sys.stdin)
    voice = request["voice"]
    speed = float(request.get("speed", 1.0))
    # kokoro-onnx takes a BCP-47-ish language tag rather than the PyTorch package's
    # single-letter lang_code. 'a' (American) maps to 'en-us', 'b' (British) to 'en-gb'.
    lang = {"a": "en-us", "b": "en-gb"}.get(request.get("langCode", "a"), "en-us")

    kokoro = Kokoro(str(model), str(voices))

    for item in request["items"]:
        # A per-item voice override exists only for auditioning: speaking one sentence in
        # several voices so a human can choose. Films use the single configured voice.
        item_voice = item.get("voice", voice)
        try:
            samples, sr = kokoro.create(item["text"], voice=item_voice, speed=speed, lang=lang)
        except Exception as exc:  # noqa: BLE001 — the id has to reach the caller
            print(json.dumps({"id": item["id"], "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            return 1
        if samples is None or len(samples) == 0:
            print(json.dumps({"id": item["id"], "error": "the model returned no audio"}), flush=True)
            return 1
        # The sample rate comes from the MODEL, never from a constant here. Writing a
        # hardcoded rate would silently resample the model's own output.
        sf.write(item["path"], np.asarray(samples), sr)
        print(
            json.dumps({"id": item["id"], "seconds": round(len(samples) / sr, 3), "path": item["path"]}),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
