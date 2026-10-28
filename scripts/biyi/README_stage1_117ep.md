# Frozen 117-episode Stage-1 FFT

This recipe uses all 117 episodes (0..116), 77,975 frames, 30 FPS and the
single air-fryer basket task. There is no held-out validation split. Training
loss alone cannot establish policy quality. The immutable corrected dataset is
`PPPPPilot/BiyiVLA@5ffc4e58e4307fab8a02a9f34aa4d975008ffa5d`.

All 117 Parquet files / 77,975 rows were checked for episode IDs, contiguous
global index, frame_index and task_index. Only nested bookkeeping statistics in
`meta/episodes_stats.jsonl` required repair (episodes 57..116). Parquet, actions,
measured states and videos were unchanged in that HF commit.

## Recipe and hardware status

`configs/pi05_biyi_117ep_fft_v1.json` freezes one GPU, global batch 8,
20,000 optimizer steps, 2,000 warmup steps and saves every 2,000 steps. The
schedule retains Stage-1 LR 2.5e-5 -> 2.5e-6, AdamW, BF16, seed 42,
stock 32D model / 17D physical adapter and horizon 50. AHFE/LoRA remain off.
Nominal starting-frame exposure is 160,000 / 77,975 = 2.052 passes.

**A800 80GB batch-8 fit remains unverified.** The preparation computers have no
usable A800; a short run on another GPU cannot validate this claim. On the actual
A800, run the existing separate `gpu-smoke` profile with exactly the committed
source, data revision, stats, GPU count and batch. Formal preflight still requires
actual completed optimizer/save memory receipts. No formal run is authorized by
this document or by generation of norm stats.

At the project's approximate 22 GB / full checkpoint estimate, ten retained
checkpoints need about 220 GB, excluding dataset, environment, temporary copy
space and any independent backup. Measure actual capacity and first save size.
No automatic pruning is implemented.

## Same portable launcher and strict lifecycle

Use the storage/freeze/auth setup in `README_portable_training.md`, then set:

```bash
export BIYI_FORMAL_RECIPE="$PWD/scripts/biyi/configs/pi05_biyi_117ep_fft_v1.json"
export BIYI_EXP_NAME=pi05_biyi_117ep_fft_v1
export BIYI_DATASET_REVISION=5ffc4e58e4307fab8a02a9f34aa4d975008ffa5d
export BIYI_TRAIN_EPISODES="$(seq -s, 0 116)"
export BIYI_NUM_GPUS=1
export BIYI_BATCH_SIZE=8
export BIYI_NUM_TRAIN_STEPS=20000
export BIYI_SAVE_INTERVAL=2000
bash scripts/biyi/train_pi05_stage1.sh train --dry-run
```

The new `python -m scripts.biyi.train_pi05_fft plan` entry point defaults to
117 episodes; the old `train_pi05_100ep` module and 100-episode JSON retain
their original defaults. The shell selects the matching recipe by versioned
name, or an explicit `BIYI_FORMAL_RECIPE`. Count/name/kind/revision must agree;
100 and 117 cannot be mixed. Both use the same stock trainer.

Full dataset freeze, train stats manifest, source/base/lock hashes, durable
checkpoint backups, accepted inference bundles, full resume bundles, download
and restore all retain their integrity gates for the selected episode count.
Changing the frozen recipe/source/data/stats on resume is rejected. Existing
checkpoints require their original pinned code; this update does not migrate
old optimizer state or pretend continuation is bitwise identical.

## CPU numeric stats without video decoding

The original `compute_stage1_norm_stats` path remains available. For this
absolute-action adapter, `compute_stage1_parquet_norm_stats` can calculate the
same numeric inputs without decoding RGB: it uses stock `RunningStats`, the
actual LeRobot v2.1 episode-clamped 50-step action query rule, batch-8 order,
and individually includes final partial-batch frames. It keeps 17D state and
action semantics unchanged and does not normalize 32D padded dimensions.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  "$BIYI_PYTHON" -m scripts.biyi.compute_stage1_parquet_norm_stats \
  --dataset-root /absolute/path/to/corrected-snapshot \
  --source-inventory /absolute/path/to/pinned-hf-file-inventory.json \
  --assets-base-dir /absolute/path/to/new-assets
```

The inventory schema is `{repo_id, revision, files}`; each file entry includes
`size`, `blob_id` and optional HF `lfs.sha256`. Every consumed metadata/Parquet
file is checked against those pinned identities before accumulation. Outputs
include `norm_stats.json`, the normal train fingerprint, and a separate numeric
source receipt. This is numeric normalization evidence, not video decode, full
dataset freeze or GPU readiness evidence. Later formal preflight still requires
the complete frozen snapshot, checks media presence and real loader samples,
and rejects stale nested bookkeeping statistics. Relocating numeric files needs
the same explicit verified source/path rebinding as other stats manifests.

## Validation

The 100-episode lifecycle regressions are preserved. Additional 117-episode tests
cover count/kind/name/revision separation, numeric query equivalence, accepted
inference and optimizer bundles, synchronous backup sealing and restore. CPU
tests and plans do not establish A800 memory fit or learned policy quality.
