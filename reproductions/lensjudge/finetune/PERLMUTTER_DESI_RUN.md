# Perlmutter DESI run — Phase C2 (v5 run-3 recipe transplanted to DESI resolution)

Train the Qwen3.5-27B student on the DESI-resolution corpus built by
`finetune/build_corpus_desi.py` from `outputs/parity_train_pool.csv` (human grades +
soft targets; NO Claude distillation — the v4 lesson). Recipe = v5 run-3 verbatim
(`$SCRATCH/ljv5_train27b_r3.slurm` on Perlmutter): 27B bf16 LoRA (no quantization) +
UNFROZEN ViT, dihedral-augmented train file, 2 full epochs, 4-GPU DDP on one hbm80g
node, AUC checkpoint selection on valsel, then ONE grade-token-logprob evaluation on
the frozen GATE split (never used for selection).

> **ACCESS**: Perlmutter login needs a fresh sshproxy key —
> `sshproxy.sh -u <nersc_user>` (24 h) before any `ssh`/`rsync` below works.
> Compute nodes have NO internet: weights pre-staged in `$HF_HOME`, `HF_HUB_OFFLINE=1`
> (Qwen/Qwen3.5-27B is already in `$SCRATCH/claudenet/hf`, 52 GB, from run-3).

Conventions below: `$LJ=$SCRATCH/ljdesi` (= `/pscratch/sd/g/gdbenson/ljdesi`); repo
code lives at `$SCRATCH/ljv5/reproductions/lensjudge` (rsync it if stale). Route ALL
caches to `$SCRATCH` (the #1 gotcha — see `serving/perlmutter_vllm.slurm`).

## 1. Ship the corpus + eval cutouts (Mac -> Perlmutter)

```bash
# corpus (jsonl + PNGs, ~2.4 GB); resumable
rsync -a --partial reproductions/lensjudge/finetune/corpus_desi/ perlmutter:$LJ/corpus_desi/
# code + the pool CSV (needed by the label/gate eval stages)
rsync -a reproductions/lensjudge/ perlmutter:$SCRATCH/ljv5/reproductions/lensjudge/ --exclude cache --exclude outputs --exclude finetune/corpus_desi
rsync -a reproductions/lensjudge/outputs/parity_train_pool.csv perlmutter:$SCRATCH/ljv5/reproductions/lensjudge/outputs/

# the jsonls reference ABSOLUTE Mac image paths -> rewrite in place on Perlmutter
# (mirrors the v5 sft_v5 jsonls, which carry absolute /pscratch/... image paths;
#  verify with `head -1 ... | grep -o '/[^"]*png' | head -1` afterwards)
ssh perlmutter "sed -i 's#/Users/benson/sync/research/agentic-lensing/reproductions/lensjudge/finetune/corpus_desi#'\$SCRATCH'/ljdesi/corpus_desi#g' \$SCRATCH/ljdesi/corpus_desi/*.jsonl"
```

Stage the valsel + gate FITS cutouts (grading renders from cubes at eval time; all are
already cached on the Mac). On the Mac:

```bash
cd reproductions && ~/.venvs/lensjudge/bin/python - <<'EOF'
import shutil, sys
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd
from lensjudge import config
from lensjudge.common import fetch
stage = Path("/tmp/cutouts_parity_eval"); stage.mkdir(parents=True, exist_ok=True)
pool = pd.read_csv(config.OUT / "parity_train_pool.csv", dtype={"grade": str})
names = pd.concat([pool[pool.split == "gate"]["name"],
                   pd.read_csv(config.HERE / "finetune/corpus_desi/valsel_manifest.csv")["name"]])
miss = 0
for n in names.astype(str):
    src = fetch.on_disk_path(n) or (config.CACHE / "cubes" / f"{n}.fits")
    if Path(src).exists(): shutil.copy(src, stage / f"{n}.fits")
    else: miss += 1
print(f"staged {len(names) - miss}/{len(names)} (missing {miss})")
EOF
rsync -a /tmp/cutouts_parity_eval/ perlmutter:$LJ/cutouts_parity_eval/
```

## 2. Training job — 27B bf16 LoRA + unfrozen ViT, 2 epochs, 4-GPU DDP

Transplanted VERBATIM from the proven r3 job: bf16 LoRA (`QUANT_BITS=""` — NO
quantization; the hbm80g A100-80s hold the 52 GB bf16 weights), effective batch 32
(4 GPUs x bs 1 x grad-accum 8 via `train_lora.sh`), LR 5e-5, warmup 0.1, VIT_LR 1e-5,
LoRA r16/a32/dropout 0.05, MAX_LENGTH 16384, sdpa, SAVE_STEPS 75 / SAVE_LIMIT 30
(keep ALL checkpoints — AUC selection needs them; v5 lesson: loss/token-acc selection
shipped an inverted model once, and AUC selection caught run-2's ckpt-750 recovery
collapse). With the 12,250-row augmented train file: ~383 steps/epoch, ~766 total
(r3: 12,370 rows -> 774 steps, 5.5 h; valsel plateau was ckpts 450–774).

The submitted job file `$SCRATCH/ljdesi_train27b.slurm` differs from r3 ONLY in the
header comment, job name/log path, TRAIN_JSONL/VAL_JSONL, and OUT:

```bash
#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 480
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -J ljdesi
#SBATCH -o /pscratch/sd/g/gdbenson/ljdesi_train27b_%j.out
set -x
export HF_HOME="$SCRATCH/claudenet/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_HF=1
export TMPDIR="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/.cache"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.cache/inductor"
export MODELSCOPE_CACHE=/tmp/modelscope
export NPROC_PER_NODE=4
mkdir -p /tmp/modelscope "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$SCRATCH/ljv5"
MODEL=Qwen/Qwen3.5-27B GPU=0,1,2,3 EPOCHS=2 SAVE_STEPS=75 SAVE_LIMIT=30 MAX_LENGTH=16384 \
  DTYPE=bfloat16 QUANT_BITS="" LR=5e-5 WARMUP=0.1 FREEZE_VIT=false VIT_LR=1e-5 \
  TRAIN_JSONL="$SCRATCH/ljdesi/corpus_desi/sft_desi_train_aug.jsonl" \
  VAL_JSONL="$SCRATCH/ljdesi/corpus_desi/sft_desi_val.jsonl" \
  OUT="$SCRATCH/ljdesi/ckpt_desi_27b" \
  SWIFT="$SCRATCH/venvs/ljtrain/bin/swift" \
  bash "$SCRATCH/ljv5/reproductions/lensjudge/finetune/train_lora.sh"
echo "TRAINDESI_RC=$?"
```

## 3. Checkpoint selection — merge -> vLLM serve -> grade valsel -> AUC

Do NOT use `swift infer` per-checkpoint (80 min/ckpt at 27B in v5; the serve path did
everything in <2 h). For each candidate checkpoint N (the plateau ckpts, e.g. the last
5–6), inside one shared-QOS job (TP2 on 2x A100-40; NERSC shared rule: 32 cores/GPU):

```bash
# merge the adapter (drops the mtp.* speculative head — expected, see wise_ft.py)
$SCRATCH/venvs/ljtrain/bin/swift export \
  --adapters $LJ/ckpt_desi_27b/v0-*/checkpoint-N --merge_lora true \
  --output_dir $LJ/ckpt_desi_27b/ckptN-merged

# serve (flags from serving/perlmutter_vllm.slurm; cache-route exports as above)
$SCRATCH/venvs/vllm/bin/vllm serve $LJ/ckpt_desi_27b/ckptN-merged \
  --port 8000 --dtype bfloat16 --tensor-parallel-size 2 --enforce-eager \
  --max-model-len 16384 --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":8}' \
  --mm-processor-kwargs '{"min_pixels":160000,"max_pixels":1003520}' \
  --served-model-name student &   # wait for /v1/models to answer

# grade the 1,174-row valsel selection set (logprobs are ON by default -> gp_A..gp_D)
LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL=http://localhost:8000/v1 \
LENSJUDGE_MODEL_GRADER=student LENSJUDGE_CUTOUT_DIR=$LJ/cutouts_parity_eval \
PYTHONPATH=$SCRATCH/ljv5/reproductions $SCRATCH/venvs/lensjudge/bin/python \
  -m lensjudge.finetune.build_corpus_desi label \
  --manifest $LJ/corpus_desi/valsel_manifest.csv \
  --out $LJ/preds_valsel_ckptN.parquet --concurrency 8

# valsel AUC (logprob scorer; lens A/B vs nonlens D+random, softmid-C excluded)
PYTHONPATH=$SCRATCH/ljv5/reproductions $SCRATCH/venvs/lensjudge/bin/python \
  -m lensjudge.finetune.build_corpus_desi gate \
  --preds $LJ/preds_valsel_ckptN.parquet --split valsel
```

Pick the max-AUC checkpoint. Selection happens HERE and only here.

(Fallback: `corpus_desi/valsel_desi.jsonl` + `valsel_desi_labels.csv` are the
swift-infer-format evalset for `finetune/eval_checkpoints.py` — but note that script
scores generated p_lens with softmid counted as negative; the serve path above is the
v5 run-3 endgame and is both faster and logprob-scored.)

## 4. Frozen GATE evaluation (grade-token logprob) — run ONCE, after selection

The gate split (259 rows: 139 graded incl. 79 A/B lenses + 60 grade-D + 60 random)
is the frozen Phase-C evidence gate — it was never in any corpus (the builder asserts
this) and must never influence selection. With the SELECTED merged checkpoint served
exactly as above:

```bash
LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL=http://localhost:8000/v1 \
LENSJUDGE_MODEL_GRADER=student LENSJUDGE_CUTOUT_DIR=$LJ/cutouts_parity_eval \
PYTHONPATH=$SCRATCH/ljv5/reproductions $SCRATCH/venvs/lensjudge/bin/python \
  -m lensjudge.finetune.build_corpus_desi label --split gate \
  --out $LJ/preds_gate_student.parquet --concurrency 8

PYTHONPATH=$SCRATCH/ljv5/reproductions $SCRATCH/venvs/lensjudge/bin/python \
  -m lensjudge.finetune.build_corpus_desi gate \
  --preds $LJ/preds_gate_student.parquet --split gate
```

`label` reads the gate rows straight from `outputs/parity_train_pool.csv` (no gate
manifest file is ever written into the corpus). Scoring is the pre-registered A0
logprob method: `p_lens_logprob = P(A)+P(B)` from `extract_grade_probs` at the
`"grade":"X"` token (v5: +0.11–0.20 AUC over generated p_lens). The same two commands
with the ZERO-SHOT base model served give the training-delta control (v5 protocol).

## 5. Optional run-3 followers

- WiSE-FT dial: `python -m lensjudge.finetune.wise_ft --base Qwen/Qwen3.5-27B
  --student .../ckptN-merged --alpha 0.7 --out .../wise_a70` — v5 found the trade
  monotone (pure student best on AUC/recovery; alpha~0.7 buys near-miss rejection).
- The corpus builder's dihedral copies are cube-level re-renders (not PNG transposes)
  because the DESI residual view is a g|r|z montage — see build_corpus_desi.py docstring.
