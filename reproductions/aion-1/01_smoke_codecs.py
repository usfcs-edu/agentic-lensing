"""
01 -- Codec round-trip smoke test (Milestone M0).

The AION repo ships per-modality reference fixtures in tests/test_data/
(`*_codec_input_batch.pt`, `*_encoded_batch.pt`, `*_decoded_batch.pt`) and a
pytest suite that validates encode/decode against them. The authoritative codec
check is therefore the repo's own suite -- we run it through our venv to confirm
the codecs download from the HF Hub and produce bit-faithful tokens on the
TITAN RTX. Writes data/results/smoke_codecs.json.

Run: /home2/benson/.venvs/aion/bin/python 01_smoke_codecs.py
"""

import json
import os
import subprocess
import sys

import _config as C

AION_REPO = "/home2/benson/lensing-repos/AION"


def main():
    env = dict(os.environ)
    env["HF_HOME"] = C.HF_HOME
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/codecs/", "--no-header",
           "-p", "no:cacheprovider"]
    proc = subprocess.run(cmd, cwd=AION_REPO, env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    tail = "\n".join(out.strip().splitlines()[-12:])
    print(tail)
    result = {"returncode": proc.returncode, "passed": proc.returncode == 0, "tail": tail}
    (C.RESULTS / "smoke_codecs.json").write_text(json.dumps(result, indent=2))
    if proc.returncode != 0:
        print("CODEC_SMOKE_FAIL")
        sys.exit(1)
    print("CODEC_SMOKE_OK")


if __name__ == "__main__":
    main()
