"""Server-side speech-to-text (tier 2). Tier 1 is the browser's SpeechRecognition + a typed transcript
box, which needs nothing here. ``mlx_whisper`` is the only backend today (optional dependency
``lensmark[voice]``); without it ``transcribe`` raises ``NotImplementedError`` and the server answers 501."""
from __future__ import annotations

import importlib.util
import os
import tempfile

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
LENS_VOCABULARY = ("LensMark strong-lens annotation: deflector, Einstein ring, theta E, arc, counter-image, "
                   "arc knot, giant arc, galaxy mask, star mask, dashed circle, dotted circle, magenta arrow, "
                   "cyan arrow, green arrow, yellow arrow, upper left, lower right, arcsec, north, east.")
_MIME_SUFFIX = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/x-wav": ".wav",
                "audio/wave": ".wav", "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a",
                "audio/x-m4a": ".m4a", "audio/aac": ".aac", "audio/flac": ".flac"}


def available_backends() -> list[str]:
    out: list[str] = []
    try:
        if importlib.util.find_spec("mlx_whisper") is not None:
            out.append("mlx_whisper")
    except (ImportError, ValueError):
        pass
    return out


def _suffix(mime: str) -> str:
    return _MIME_SUFFIX.get((mime or "").split(";")[0].strip().lower(), ".bin")


def transcribe(audio: bytes, mime: str) -> tuple[str, str]:
    """(transcript, backend). Raises ``NotImplementedError`` when no STT backend is installed."""
    backends = available_backends()
    if "mlx_whisper" not in backends:
        raise NotImplementedError("no STT backend installed (pip install mlx-whisper)")
    import mlx_whisper  # type: ignore[import-not-found]

    fd, path = tempfile.mkstemp(suffix=_suffix(mime), prefix="lensmark-stt-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio)
        result = mlx_whisper.transcribe(path, path_or_hf_repo=WHISPER_MODEL, language="en",
                                        initial_prompt=LENS_VOCABULARY)
        text = str(result.get("text", "")).strip() if isinstance(result, dict) else str(result).strip()
        return text, "mlx_whisper"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
