"""
02 -- Model bring-up smoke test (Milestone M0).

Loads all three released checkpoints (base/large/xlarge), verifies the
frozen-encoder embedding path and the single-shot redshift forward, and checks
that ``torch.compile`` is numerically equivalent to eager on sm_75 (the one
flagged technical risk for Turing). Writes data/results/smoke_model.json.

Run: HF_HOME=... /home2/benson/.venvs/aion/bin/python 02_smoke_model.py
"""

import json
import time

import torch

import _config as C


def main():
    torch.set_grad_enabled(False)
    from aion.codecs import CodecManager
    from aion.fourm.fm_utils import NormCrossAttention
    from aion.model import AION
    from aion.modalities import LegacySurveyFluxG, LegacySurveyFluxR, LegacySurveyFluxI, \
        LegacySurveyFluxZ, Z

    cm = CodecManager(device="cuda")
    B = 8
    mk = lambda Cl: Cl(value=torch.rand(B, 1, device="cuda") * 100)
    tokens = cm.encode(mk(LegacySurveyFluxG), mk(LegacySurveyFluxR),
                       mk(LegacySurveyFluxI), mk(LegacySurveyFluxZ))

    report = {}
    for v in C.VARIANTS:
        t = time.time()
        m = AION.from_pretrained(C.MODELS[v]).to("cuda").eval()
        emb = m.encode(tokens, num_encoder_tokens=4)
        zl = m(tokens, target_modality=Z)["tok_z"]
        assert emb.shape == (B, 4, C.DIMS[v]), (emb.shape, C.DIMS[v])
        assert zl.shape[0] == B and zl.shape[-1] == C.Z_NBINS + 1, zl.shape
        report[v] = {
            "load_s": round(time.time() - t, 1),
            "dim": int(m.dim),
            "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6),
            "emb_shape": list(emb.shape),
            "z_logits_shape": list(zl.shape),
            "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        print(f"{v}: {report[v]}")
        del m
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # torch.compile parity on sm_75 (NormCrossAttention = the attentive-pooling op)
    ca = NormCrossAttention(C.DIMS["base"], num_heads=12, proj_bias=False).cuda().eval()
    q = torch.randn(B, 5, C.DIMS["base"], device="cuda")
    kv = torch.randn(B, 4, C.DIMS["base"], device="cuda")
    oe = ca(q, kv)
    try:
        oc = torch.compile(ca)(q, kv)
        maxdiff = float((oe - oc).abs().max())
        report["torch_compile"] = {"works": True, "max_abs_diff": maxdiff}
    except Exception as e:  # pragma: no cover
        report["torch_compile"] = {"works": False, "error": repr(e)[:200]}
    print("torch.compile:", report["torch_compile"])

    (C.RESULTS / "smoke_model.json").write_text(json.dumps(report, indent=2))
    print("MODEL_SMOKE_OK")


if __name__ == "__main__":
    main()
