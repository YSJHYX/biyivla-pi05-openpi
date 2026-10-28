# Portable stock PyTorch π0.5 formal training

For the corrected 117-episode / 77,975-frame Stage-1 snapshot, use
[README_stage1_117ep.md](README_stage1_117ep.md). It pins the corrected dataset
and batch-8 / 20k recipe while preserving the legacy 100-episode lifecycle.

`biyivla-pi05-openpi` remains the only training code repository. AutoDL, Colab,
local servers and other GPU hosts use the same `train_pi05_stage1.sh` launcher,
`pi05_biyi_stage1` configuration and `scripts/train_pytorch.py`. There is no
platform-specific trainer. AHFE/LoRA, model, optimizer and LR recipe are unchanged.
Keep the Python 3.11 / `uv sync --frozen` runtime and existing Transformers
replacement setup from `README_stage1.md`; do not regenerate `uv.lock` per platform.

## Three storage layers

1. GitHub: code, versioned recipe, tests and storage tools; no weights/data/secrets.
2. Operator-declared persistent disk, or a runtime disk plus independent persistent
   backup: complete model + optimizer + metadata + norm stats + run manifest and
   W&B ID. Dataset revision, source Git commit, base weights, norm stats and lockfile
   hashes remain traceable. No automatic checkpoint pruning is added.
3. Private **Model** repo `PPPPPilot/BiyiVLA-pi05-checkpoints`: accepted inference
   releases via the existing explicit `checkpoint_store pack/verify/upload` gate;
   full optimizer resume bundles only when requested. Smoke checkpoints never
   qualify as formal archives. Synchronous durability backups do not upload to HF
   and do not assert inference/model-quality acceptance.

A directory name cannot prove durability. The operator must verify that the
chosen disk/mount survives runtime loss, that it has sufficient quota, and that
it is independent of a temporary VM's lifetime. `BIYI_STORAGE_MODE` is an explicit
operator declaration. The gate checks path separation, write/read/copy checksums,
file flushes and the destination filesystem device identity during the run.
Unmount/change failures stop saving rather than silently accepting an unverified
backup. It cannot prove provider retention, available future quota or recovery
from deletion of the persistent service itself.

## Common configuration

All paths must be absolute and outside the Git checkout. These are examples:
replace the mount paths and the dataset commit with actual verified values.
There is no required `/root/autodl-tmp` or `/data` path.

```bash
export BIYI_PYTHON="$PWD/.venv/bin/python"   # or the frozen environment's interpreter
export BIYI_RUN_ROOT=/mnt/persistent/biyivla
export BIYI_PERSISTENT_ROOT="$BIYI_RUN_ROOT" # checkpoint_store freeze/download/restore CLI
export BIYI_STORAGE_MODE=persistent
export BIYI_DATASET_REVISION=REPLACE_WITH_FULL_40_CHARACTER_DATASET_COMMIT
export BIYI_DATASET_REPO_ID=PPPPPilot/BiyiVLA
export BIYI_DATASET_LOCAL_ROOT="$BIYI_RUN_ROOT/datasets/$BIYI_DATASET_REVISION"
export BIYI_TRAIN_EPISODES="$(seq -s, 0 99)"
export BIYI_BASE_CHECKPOINT_DIR=/mnt/persistent/base/pi05_base_pytorch
export BIYI_ASSETS_BASE_DIR="$BIYI_RUN_ROOT/assets"
export BIYI_CHECKPOINT_BASE_DIR="$BIYI_RUN_ROOT/checkpoints"
export BIYI_FORMAL_RECIPE="$PWD/scripts/biyi/configs/pi05_biyi_100ep_fft_v1.json"
export BIYI_EXP_NAME=pi05_biyi_100ep_fft_v1
export BIYI_NUM_GPUS=1
export BIYI_BATCH_SIZE=1
export BIYI_NUM_WORKERS=0
export BIYI_NUM_TRAIN_STEPS=10000
export BIYI_SAVE_INTERVAL=1000
```

Freeze the immutable **100-episode** dataset and compute train-only stats using
`README_checkpoint_storage.md` (`freeze-dataset`, then `compute-stats`). Do not
reuse the five-episode smoke revision or its norm stats. Frozen dataset checksums
and the stats fingerprint must pass before training. Authenticate HF and W&B
interactively in the target environment; never put tokens in Git/config files.

`BIYI_NUM_GPUS=auto` selects all GPUs visible to the configured interpreter,
including `CUDA_VISIBLE_DEVICES` restrictions. A positive explicit count selects
that many visible devices; execution rejects unavailable counts. Global batch
must be positive and divisible by that count. One GPU uses direct Python; N GPUs
use the same interpreter's `torch.distributed.run` (torchrun DDP). Optional
`BIYI_TORCHRUN` retains the existing executable override. Dry-runs require an
explicit count and never import torch, probe CUDA, create directories, authenticate
or launch training.

## Actual hardware memory smoke is mandatory

Prefer trying an A100 **80 GB** if the provider offers it; this is a candidate,
not a verified fit or a Colab entitlement promise. A100 40 GB, L4 and T4 have
substantially different capacity. Use the actual allocated hardware and run the
same GPU count, global batch, dataset, stats and committed source as the formal
run. The stock FFT optimizer state must materialize before accepting memory fit.

The following commands are instructions for the GPU host; they were not run by
this code change:

```bash
# CPU-only command preview first:
bash scripts/biyi/train_pi05_stage1.sh train --dry-run

# 3-step stock forward/backward/optimizer/save smoke; W&B disabled, separate run:
BIYI_EXP_NAME=gpu_memory_smoke_v1 BIYI_NUM_TRAIN_STEPS=3 \
  BIYI_SAVE_INTERVAL=2 bash scripts/biyi/train_pi05_stage1.sh gpu-smoke
export BIYI_MEMORY_SMOKE_RUN_DIR="$BIYI_CHECKPOINT_BASE_DIR/pi05_biyi_stage1/gpu_memory_smoke_v1"

# Inspects actual per-rank completed memory receipts, storage, data, stats, base,
# effective schedule and HF/W&B authentication; does not start training:
bash scripts/biyi/train_pi05_stage1.sh preflight

# Only after the above succeeds, explicitly launch formal training:
bash scripts/biyi/train_pi05_stage1.sh train
# Same committed recipe/provenance and existing W&B ID:
BIYI_RESUME=1 bash scripts/biyi/train_pi05_stage1.sh train
```

`gpu-smoke` is a generic 2–10 step profile, not a Colab trainer. Existing
`local-smoke` and `h20-smoke` profiles remain available. A successful smoke writes
`memory_smoke_rank<N>.json` for every rank with actual GPU name, total memory,
peak allocated/reserved memory and completed steps. Formal preflight requires
matching batch/count/source/data/stats and hardware; a log assertion or a hand-set
"passed" variable is insufficient. Receipts are local operational evidence, not
HF release acceptance or tamper-proof attestations. A different GPU type, batch,
code commit, dataset or norm stats requires another smoke. A short smoke does
not guarantee a long run's memory or storage headroom.

The direct module remains supported and shares the same launcher and I/O gate:

```bash
"$BIYI_PYTHON" -m scripts.biyi.train_pi05_100ep plan
"$BIYI_PYTHON" -m scripts.biyi.train_pi05_100ep preflight
"$BIYI_PYTHON" -m scripts.biyi.train_pi05_100ep train --resume
```

The direct module defaults to the versioned recipe and root layout. Use
`--from-stage1-env` to honor the shell path/recipe environment overrides, or
`--recipe`, `--num-gpus`, `--batch-size` for explicit choices. The shell passes
`--from-stage1-env` automatically for formal recipe/name runs. Effective settings
are frozen in `run_manifest.json`; changing them on resume is rejected. Keep the
same source commit, GPU count and global batch when continuing the same run.
A deliberate new recipe/experiment is needed for other settings. Stock resume
restores weights, optimizer and global step, but does not save RNG/DataLoader
position; continuation is not promised to be bitwise identical.

Generic older Stage-1 datasets may still use `train` without a formal recipe;
100-episode runs are required to use the strict formal lifecycle. Generic formal
runs also require immutable dataset revision/file-lock provenance, a memory smoke
and durable storage, and cannot be uploaded as accepted 100-episode releases.

## Temporary runtime plus durable backups (including optional Colab)

On a temporary VM, keep datasets, assets and training checkpoints on fast local
storage if desired. Declare a separately mounted persistent destination:

```bash
export BIYI_STORAGE_MODE=backup
export BIYI_RUN_ROOT=/content/biyivla-runtime  # example temporary location
export BIYI_DATASET_LOCAL_ROOT="$BIYI_RUN_ROOT/datasets/$BIYI_DATASET_REVISION"
export BIYI_ASSETS_BASE_DIR="$BIYI_RUN_ROOT/assets"
export BIYI_CHECKPOINT_BASE_DIR="$BIYI_RUN_ROOT/checkpoints"
export BIYI_BACKUP_ROOT=/content/drive/MyDrive/BiyiVLA-resume-backups
# BIYI_PERSISTENT_ROOT remains the real persistent root for checkpoint_store CLI.
```

Mount Drive yourself before using that example; an ordinary directory named
`/content/drive` is rejected. `/content`, `/tmp` and shared-memory paths alone do
not satisfy persistence. Prepare/freeze the dataset on the persistent root, then
copy the full frozen dataset (including `dataset_lock.json`) to the runtime root.
Recompute stats for the new local path, or use the existing verified bundle
restore procedure that rebinds only the local stats fingerprint path.

Every completed stock save invokes synchronous allowlisted backup, hashes the
copy and reads it back before returning success. Backups include optimizer and
metadata even before inference acceptance, since a temporary runtime can vanish
at any time. A failed/full/disconnected destination propagates an error to the
trainer (torchrun terminates sibling ranks); published backups are immutable.
The interval bounds possible lost work: only a **verified completed backup** is
recoverable after abrupt runtime loss. Unsaved steps and interrupted copies can
be lost. DDP backup waits use a default 3600-second process-group timeout; optional
`BIYI_DDP_TIMEOUT_SECONDS` can be set for slower storage. Allow capacity for all
checkpoints plus temporary copy space; no automatic pruning is performed.

Example complete backup:

```text
$BIYI_BACKUP_ROOT/pi05_biyi_stage1/pi05_biyi_100ep_fft_v1/step-1000/
  backup_manifest.json
  run_manifest.json
  train_stats_manifest.json
  wandb_id.txt
  1000/{model.safetensors,optimizer.pt,metadata.pt,checkpoint_manifest.json}
  1000/assets/biyi_stage1/norm_stats.json
```

Restore to a **new** run directory, after restoring the same committed code,
base weights, frozen dataset and matching/rebound stats. The command never
unpickles optimizer/metadata or overwrites an existing run:

```bash
"$BIYI_PYTHON" -m scripts.biyi.training_runtime restore-backup \
  --backup "$BIYI_BACKUP_ROOT/pi05_biyi_stage1/$BIYI_EXP_NAME/step-1000" \
  --run-dir "$BIYI_CHECKPOINT_BASE_DIR/pi05_biyi_stage1/$BIYI_EXP_NAME" \
  --assets-base-dir "$BIYI_ASSETS_BASE_DIR" --dataset-root "$BIYI_DATASET_LOCAL_ROOT"
# Re-run gpu-smoke on the destination hardware; set its memory receipt path.
BIYI_RESUME=1 bash scripts/biyi/train_pi05_stage1.sh preflight
BIYI_RESUME=1 bash scripts/biyi/train_pi05_stage1.sh train
```

For 100-episode backups, supplying both assets and dataset paths also validates
all frozen dataset files, restores norm stats and rebinds only `local_root` in
the loader's stats fingerprint. The original run manifest and archived stats
provenance remain byte-for-byte unchanged. Conflicting assets are rejected.
For generic older Stage-1 backups, restore the run without these two options
and prepare its matching stats separately.

The existing accepted full HF resume bundle and `checkpoint_store restore`
procedure remain available. No Colab session is connected or provisioned by
these scripts. In VS Code/Colab, use a notebook shell cell to execute the same
commands after mounting persistent storage and preparing the frozen environment.

## CPU validation

```bash
bash -n scripts/biyi/train_pi05_stage1.sh
python -m unittest scripts.biyi.checkpoint_store_test scripts.biyi.training_runtime_test -q
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu pytest scripts/biyi/stage1_hardening_test.py -q
```

The tests include 1/2/4-GPU command plans, invalid batch/count rejection, failed
storage/smoke gates, corrupted/partial backups, no-overwrite restores, accepted-only
HF archive tests and an actual tiny CPU model/AdamW stock save→backup→restore/load
round trip. They do not establish real π0.5 GPU fit or model-quality acceptance.
