# Biyi π0.5 Stage-1 PyTorch template

## Stage-1 scope and verified baseline

`pi05_biyi_stage1` fine-tunes the stock PyTorch π0.5 model: `pi05=True`,
`action_dim=32`, `action_horizon=50`, BF16, AdamW, and all parameters trainable.
The Stage-1 experiment seed is 42; a future Stage-2 multi-seed study may use
42 / 123 / 3407, but it is not part of this launcher. Stage-2 model selection
must include validation and real-robot metrics, not train loss alone.
There is no AHFE, auxiliary/dual head, LoRA, residual RL, or delta-action
transform in this path. RTC is OFF; RTC training/inference belongs to a separate
stage after Stage-1 model validation and is not implemented here.

The local base is `/data/openpi_checkpoints/pi05_base_pytorch/model.safetensors`:
7,233,650,408 bytes (about 6.74 GiB), SHA256
`dbcbe3ea978ce1d6f791582fa034e7ece8a41feaad2d3c9a5b014c1938610e2a`.
Its SafeTensors header has 812 BF16 tensors, including
`action_in_proj.weight [1024,32]`, `action_in_proj.bias [1024]`,
`action_out_proj.weight [32,1024]`, and `action_out_proj.bias [32]`;
there are no AHFE, auxiliary, dual-head, or LoRA keys. This is a converted
PyTorch base checkpoint, not a trainer resume checkpoint. Set
`BIYI_BASE_CHECKPOINT_DIR` to the directory containing the same verified file
on a cloud host. Preflight checks its size and SHA256 before training.

## Frozen physical data contract

This uses the stock π0.5 model and existing Biyi 17D adapter. The raw LeRobot
state and action stay `float32[17]`: indices 0–5 are RM65 J1–J6 and 6–16 are
SG100. RM65 state is measured qpos; SG100 state is measured feedback. RM65
action is the actual sent absolute joint command; SG100 action is the final,
effective canonical absolute command. The adapter maps head and wrist RGB,
masks the absent third view, pads
state and action to 32D after normalization, and crops inference action to 17D.
The task prompt is the collection-time instruction. RM65 J6 and all 11 SG100
dimensions remain present. An intentionally locked SG100 DOF (including index
side sway in its locked gripper morphology) can legitimately have zero or low
variance; do not remove it. Do not interpret `action - state` as SG100 tracking
error, apply a manual temporal shift, or convert absolute actions to deltas.

## Training configuration and W&B contract

The launcher selects `pi05_biyi_stage1` and passes an explicit W&B setting:

```text
LOCAL_SMOKE_WANDB    = OFF
H20_SMOKE_WANDB      = OFF
STAGE1_TRAIN_WANDB   = ON
```

Formal runs use project `BiyiVLA-Stage1`; `BIYI_EXP_NAME` becomes the run name
and checkpoint directory name, for example `pi05_biyi_stage1_bs32_10k_v1`.
The provisional formal recipe is 10k optimizer steps, 10% warmup, and cosine
decay ending at step 10k: peak LR 2.5e-5, end LR 2.5e-6, AdamW
`(b1=0.9, b2=0.95, eps=1e-8)`, weight decay 0.05, gradient clip 1.0,
BF16 dominant mixed precision, seed 42, and no EMA. Stage-1 AdamW applies
weight decay to trainable matrix weights, including embedding matrices and
AdaRMS conditioning projections. Bias and scalar/1D parameters, including
normalization scales and AdaRMS projection biases, use zero decay.
CPU meta-model inspection found 812/812 tensors trainable:
460 decay tensors (3,616,163,328 scalar parameters) and 352 no-decay tensors
(594,192 scalar parameters). All 37 AdaRMS `dense.weight` matrices decay.
Formal preflight checks `BIYI_EXP_NAME`, W&B project/enabled state, online W&B
authentication, HF authentication, private dataset access, and the effective
steps/warmup/decay/LR/weight-decay/seed/BF16/EMA recipe before training.
`auth-check` queries the SDKs without starting a W&B run. Only DDP rank 0 initializes and
logs W&B. The trainer already logs loss, learning rate, gradient norm, step
time, and global step; W&B can collect system GPU metrics. The added trainer
code prints allocated, reserved, and peak PyTorch allocator VRAM per rank at
training completion. Do not put credentials in repository files.

## Cloud authentication and startup order

1. Start the cloud instance and run `nvidia-smi`.
2. Choose writable cache paths, for example:

   ```bash
   export HF_HOME=/data/hf_cache
   export TORCH_HOME=/data/torch_cache
   export XDG_CACHE_HOME=/data/cache
   export TMPDIR=/data/tmp
   ```

3. Set `BIYI_DATASET_REPO_ID` to the frozen Hugging Face dataset repo ID.
   Inject credentials through cloud secrets/environment variables (preferred
   for temporary instances), or use the installed official CLIs:

   ```bash
   export HF_TOKEN="<your_hf_token>"
   export WANDB_API_KEY="<your_wandb_api_key>"
   # Alternatively: hf auth login
   # Alternatively: wandb login
   scripts/biyi/train_pi05_stage1.sh auth-check
   ```

   This checkout has `hf auth login` and `wandb login` available. On a different
   image, confirm with `hf auth --help` and `wandb login --help`. The check uses
   the HF and W&B SDKs to verify account access without creating a run. If the
   dataset repo ID is set, it also verifies dataset read access. Optional
   `BIYI_HF_CHECKPOINT_REPO_ID` verifies read access to an existing model repo;
   this template has no checkpoint upload workflow or write check.

   Require `HF_AUTH = PASS`, `HF_DATASET_ACCESS = PASS`, and
   `WANDB_AUTH = PASS` before the formal launch. Do not put credentials in Git
   files, `config.py`, this README, launcher scripts, W&B config, or commit
   messages. Use interactive login or secret injection rather than passing a
   real token as a CLI argument.

4. Download/synchronize the frozen dataset and base checkpoint if needed.
5. Run `compute-stats`, then `preflight` with a formal `BIYI_EXP_NAME`.
6. Run the H20 memory smoke (W&B explicitly off), then the formal train profile
   (W&B explicitly on in project `BiyiVLA-Stage1`).

## Frozen train dataset and stats

Set these values in the shell. Use a local, frozen LeRobot v2.1 train snapshot.
`BIYI_TRAIN_EPISODES` is optional only when `meta/info.json` says its `train`
split covers every episode. If a train/eval split shares the snapshot, give the
original train episode IDs explicitly (for example `0,1,2`).

```bash
export BIYI_DATASET_REPO_ID='your-org/your-frozen-train-dataset'
export BIYI_DATASET_LOCAL_ROOT='/path/to/frozen/lerobot-dataset'
export BIYI_TRAIN_EPISODES='0,1,2' # omit if the entire snapshot is train
export BIYI_ASSETS_BASE_DIR='/path/to/new/stage1/assets'
export BIYI_CHECKPOINT_BASE_DIR='/path/to/new/stage1/checkpoints'
export HF_DATASETS_CACHE='/path/to/writable/hf-datasets-cache'
scripts/biyi/train_pi05_stage1.sh compute-stats
export BIYI_EXP_NAME='pi05_biyi_stage1_bs32_10k_v1'
export BIYI_NUM_TRAIN_STEPS=10000 # provisional; freeze after dataset sizing
scripts/biyi/train_pi05_stage1.sh preflight
```

`compute-stats` uses the stock `RunningStats` and data transforms on every
selected train frame, including the final partial batch, then records the repo ID, local root, episode
selection, metadata hashes and stats hash. The launcher's preflight requires
these exact values to match. Recompute after any dataset freeze or split change.
Metadata hashes do not prove that media and Parquet contents are immutable;
freeze the snapshot externally before this step. Never point Stage-1 assets at
the older Task 3B smoke stats.

Stage-1 uses the fixed asset ID `biyi_stage1`, independent of the dataset repo
ID. Stats and their source manifest are written under
`<BIYI_ASSETS_BASE_DIR>/pi05_biyi_stage1/biyi_stage1/`; the stats file is
`norm_stats.json`. Each saved checkpoint contains
`assets/biyi_stage1/norm_stats.json`, so deployment resolves the same asset ID
even when `BIYI_DATASET_REPO_ID` is unset.

## Smoke and training profiles

Local smoke (one GPU; 3 optimizer steps by default):

```bash
export BIYI_EXP_NAME='local_smoke_001'
BIYI_BATCH_SIZE=1 scripts/biyi/train_pi05_stage1.sh local-smoke
```

H20 memory smoke (two GPUs, begin at global batch 8, 30 steps by default):

```bash
export BIYI_EXP_NAME='h20_memory_smoke_bs8_001'
BIYI_BATCH_SIZE=8 scripts/biyi/train_pi05_stage1.sh h20-smoke
```

This profile runs to step 20, saves model plus AdamW optimizer state, resumes
the same experiment to step 30, and prints checkpoint directory sizes. The
trainer logs allocated, reserved and peak VRAM on each rank after the last
optimizer step. Use the logged step times after warmup for stable timing. A
forward/backward pass alone does not establish memory fit: the first AdamW
`optimizer.step()` allocates optimizer state.

Formal two GPU launch, only after dataset freeze, preflight, and memory smoke:

```bash
export BIYI_EXP_NAME='pi05_biyi_stage1_bs32_10k_v1'
export BIYI_NUM_TRAIN_STEPS=10000 # provisional; freeze before the first formal step
BIYI_BATCH_SIZE=32 scripts/biyi/train_pi05_stage1.sh train
# To resume an interrupted run with the same frozen schedule:
BIYI_RESUME=1 BIYI_BATCH_SIZE=32 scripts/biyi/train_pi05_stage1.sh train
```

The launcher passes current Tyro options to `scripts/train_pytorch.py`, using
`torchrun --standalone --nnodes=1 --nproc_per_node=2` on H20. Override
`BIYI_NUM_WORKERS`, `BIYI_SAVE_INTERVAL`, `BIYI_KEEP_PERIOD`, and
`BIYI_LOG_INTERVAL` as needed. Formal launch computes warmup as
`(num_train_steps + 5) / 10` using integer division (nearest 10%, halves up)
and sets `decay_steps=num_train_steps`. The 10k default resolves to 1k/10k;
15k and 20k resolve to 1.5k/15k and 2k/20k. Smoke instead uses one warmup
step and a schedule spanning its full short run, including H20 resume.
`BIYI_BATCH_SIZE` is global batch, split evenly
across DDP ranks. This trainer has no gradient accumulation; if batch 32 does
not fit, lower the global batch and recompute the steps/effective epochs.

Use distinct experiment names/directories for local smoke (for example
`local_smoke_001`), H20 smoke (`h20_memory_smoke_001`) and formal training.
The launcher rejects formal resume unless that directory already contains a
`wandb_id.txt`. Stock trainer writes that ID on the first W&B-enabled run and
uses `resume="must"` for later formal resumes. Smoke runs do not create the ID.

The default save interval is 1000 for the formal run. PyTorch checkpoints
include `model.safetensors`, `optimizer.pt`, `metadata.pt`, and norm assets.
The PyTorch trainer does not implement rotation or `keep_period` retention,
despite accepting that CLI option. Planned roles: 1k early optimization check,
3k early behavior, 5k intermediate real-robot checkpoint, 8k late checkpoint,
and 10k provisional primary endpoint. These remain on disk unless an operator
manages storage. `--resume` loads the latest numbered checkpoint and its
optimizer state. No long run is started by preparation.
With 10,000 steps and save interval 1000, saves occur at 1000, 2000, ..., 10000,
without an extra 9999 save. The final step is also saved when it is not a
multiple of the interval, for example 10500.

## AutoDL 2×H20 Cloud Bring-up Checklist

### Target instance and software boundary

Planned instance: 2× H20-NVLink, AMD EPYC 9K84 (32 cores), 300 GB RAM,
30 GB system disk, 50 GB included SSD data disk, and 400 GB paid data disk.
Verify actual usable capacity and mounts after boot. Expected driver is
580.82.07; `nvidia-smi` may report host CUDA 13.0. That is a driver capability,
not the PyTorch runtime requirement. Use Python 3.11, Torch 2.7.1, and
Transformers 4.53.2; do not upgrade them or pull upstream OpenPI for this run.

The Stage-1 standard environment is Python 3.11, Torch 2.7.1, Transformers
4.53.2, and the repository's frozen `uv.lock`. On x86 Linux this lock selects
Torch `2.7.1+cu126` with the CUDA 12.6 runtime. Use `uv sync --frozen` to
reproduce it; cu126 is the selected baseline and is not a pre-training blocker.
Record the actual cloud runtime and validate it through H20 smoke. Reassess
cu128 only if that smoke demonstrates a compatibility problem with cu126;
do not upgrade preemptively or use cu130. The host's `nvidia-smi` CUDA 13.x
version describes driver capability and does not require a CUDA 13 PyTorch
runtime. Keep `uv.lock` unchanged for this run.

### Step 1 — SSH login

Use the host, SSH port, and user from the AutoDL console. Immediately record:

```bash
hostname
uname -a
date
whoami
```

### Step 2 — Inspect storage

```bash
df -h
lsblk
mount
```

Identify the actual large data disk mount; `/root/autodl-tmp` is only an
example. Put `BIYI_ROOT` on that mount, not on the 30 GB system disk:

```bash
export BIYI_ROOT=/root/autodl-tmp/biyivla  # replace with the verified mount
mkdir -p "$BIYI_ROOT"/{openpi,datasets,models,checkpoints,assets,hf_cache,torch_cache,cache,tmp,envs,logs}
df -h /
df -h "$BIYI_ROOT"
```

### Step 3 — Redirect large caches and Python environment

```bash
export HF_HOME="$BIYI_ROOT/hf_cache"
export HF_DATASETS_CACHE="$BIYI_ROOT/hf_cache/datasets"
export TORCH_HOME="$BIYI_ROOT/torch_cache"
export XDG_CACHE_HOME="$BIYI_ROOT/cache"
export OPENPI_DATA_HOME="$BIYI_ROOT/cache/openpi"
export UV_CACHE_DIR="$BIYI_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$BIYI_ROOT/envs/openpi-stage1"
export TMPDIR="$BIYI_ROOT/tmp"
```

Keep these exports in the training shell/session. Monitor both filesystems:
`df -h / "$BIYI_ROOT"`. PyTorch, HF, uv, checkpoints, and temporary files
must not silently consume the system disk.
OpenPI's `src/openpi/shared/download.py` uses `OPENPI_DATA_HOME` for its asset
cache, defaulting to `~/.cache/openpi`. Keep that cache on the data disk as well;
the AutoDL system disk has only about 30 GB.

### Step 4 — Validate GPUs and topology

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
```

Confirm two H20 GPUs, actual driver, and expected NVLink/high-bandwidth path.
If NCCL tests are installed, run all-reduce; otherwise, the later two-rank
PyTorch smoke must exercise distributed training. Do not rely on the product
listing alone.

### Step 5 — Check network and choose a transfer route

```bash
curl -I https://github.com
curl -I https://huggingface.co
curl -I https://api.wandb.ai
```

Use, in order: direct internet, provider-supported network/proxy, SSH reverse
proxy, local `scp`/`rsync`, or OSS/S3/object storage. Do not install a VPN by
default or add proxy creation to the Stage-1 launcher. For an existing local
HTTP proxy on port 7890, an infrastructure-only example is:

```bash
# Run on the local computer, with values from the AutoDL console:
ssh -N -R 17890:127.0.0.1:7890 -p <AUTODL_PORT> root@<AUTODL_HOST>
# Run on the cloud host while that tunnel is alive:
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890
```

A local-computer disconnect breaks this tunnel. Finish large HF transfers
before training, train from local SSD, and prefer direct cloud connectivity
for live W&B logging.

### Step 6 — Clone the archived private Stage-1 source

After this Stage-1 version is committed and pushed to `private/main`, clone
the private repository, including submodules, rather than upstream/main.
Checkout the exact archived commit reported with that push:

```bash
git clone --recurse-submodules https://github.com/YSJHYX/biyivla-pi05-openpi.git "$BIYI_ROOT/openpi"
cd "$BIYI_ROOT/openpi"
git checkout <archived-stage1-commit>
git submodule update --init --recursive
git rev-parse HEAD
git status --short --branch
```

Record this commit in the run notes and W&B run config/metadata. Never put a
private Git credential in the clone URL or repository files.

### Step 7 — Create and verify the π0.5 environment

Use the repository's `uv.lock` and README installation flow. The environment
and uv cache above are on the data disk; `GIT_LFS_SKIP_SMUDGE=1` follows the
upstream LeRobot dependency instruction:

```bash
uv venv --python 3.11 "$UV_PROJECT_ENVIRONMENT"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e .
export BIYI_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export BIYI_TORCHRUN="$UV_PROJECT_ENVIRONMENT/bin/torchrun"
"$BIYI_PYTHON" --version
"$BIYI_PYTHON" - <<'PY'
import torch, transformers
print('torch:', torch.__version__, 'runtime CUDA:', torch.version.cuda)
print('transformers:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available(), 'GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Require Python 3.11, Torch 2.7.1, Transformers 4.53.2, and two visible H20s.
Record the CUDA runtime exactly; the frozen x86 Linux lock selects
`2.7.1+cu126` / CUDA 12.6. Reassess cu128 only if H20 smoke demonstrates a
compatibility problem. Do not change packages solely because the host reports
CUDA 13.x.

Apply the repository's `transformers_replace` files next. PyTorch π0.5 depends
on patched Transformers 4.53.2 for AdaRMS, activation precision, and KV-cache
behavior; `uv sync --frozen` does not copy these replacements. Resolve the
installed directory through `BIYI_PYTHON`, which points to the data-disk
environment:

```bash
TRANSFORMERS_DIR="$("$BIYI_PYTHON" - <<'PY'
import pathlib
import transformers

print(pathlib.Path(transformers.__file__).parent)
PY
)"

cp -r \
  src/openpi/models_pytorch/transformers_replace/* \
  "$TRANSFORMERS_DIR/"

"$BIYI_PYTHON" - <<'PY'
import transformers
from transformers.models.siglip import check

print("transformers:", transformers.__version__)
print(
    "transformers_replace:",
    check.check_whether_transformers_replace_is_installed_correctly(),
)

assert transformers.__version__ == "4.53.2"
assert check.check_whether_transformers_replace_is_installed_correctly()

print("TRANSFORMERS_PATCH = PASS")
PY
```

Require `transformers: 4.53.2`, `transformers_replace: True`, and
`TRANSFORMERS_PATCH = PASS`; stop before H20 smoke if verification fails.

Prefetch the PaliGemma tokenizer before proceeding to authentication, dataset,
checkpoint, or smoke steps:

```bash
"$BIYI_PYTHON" - <<'PY'
from openpi.models.tokenizer import PaligemmaTokenizer

tokenizer = PaligemmaTokenizer(200)

print("PALIGEMMA_TOKENIZER = PASS")
PY
```

Require `PALIGEMMA_TOKENIZER = PASS`. This fetches
`gs://big_vision/paligemma_tokenizer.model` into
`$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`, confirming the GCS
asset is accessible and cached on the data disk before training starts. If
prefetch fails, resolve the reported network/cache error before H20 smoke.
Keep the tokenizer asset out of Git.

### Step 8 — Authenticate Hugging Face and W&B

Set `BIYI_DATASET_REPO_ID` to the frozen private dataset and inject secrets
through the cloud secret manager or a temporary shell environment. Alternatively
use `hf auth login` and `wandb login` with their credential stores:

```bash
export BIYI_DATASET_REPO_ID='your-org/your-frozen-biyivla-dataset'
export HF_TOKEN='<your_hf_token>'
export WANDB_API_KEY='<your_wandb_api_key>'
scripts/biyi/train_pi05_stage1.sh auth-check
```

Require `HF_AUTH = PASS`, `HF_DATASET_ACCESS = PASS`, `WANDB_AUTH = PASS`,
`WANDB_PROJECT = BiyiVLA-Stage1`, and `FORMAL_AUTH_GATE = PASS`. The check
creates no W&B run. Do not save real tokens in Git, scripts, config, README,
commit messages, or a shared shell history.

### Step 9 — Transfer the verified π0.5 base checkpoint

Transfer the locally verified `model.safetensors` to
`$BIYI_ROOT/models/pi05_base_pytorch/model.safetensors` using the selected
transfer route. Then:

```bash
export BIYI_BASE_CHECKPOINT_DIR="$BIYI_ROOT/models/pi05_base_pytorch"
stat -c '%s bytes' "$BIYI_BASE_CHECKPOINT_DIR/model.safetensors"
sha256sum "$BIYI_BASE_CHECKPOINT_DIR/model.safetensors"
```

Require 7,233,650,408 bytes and SHA256
`dbcbe3ea978ce1d6f791582fa034e7ece8a41feaad2d3c9a5b014c1938610e2a`.
Stop on mismatch. The config reads `BIYI_BASE_CHECKPOINT_DIR`; do not point
cloud training at the development machine's `/data/openpi_checkpoints` path.

### Step 10 — Synchronize the frozen dataset

Once the private BiyiVLA dataset is frozen, authenticate to its HF repo,
download the complete revision to SSD, and train from that local snapshot:

```bash
export BIYI_DATASET_LOCAL_ROOT="$BIYI_ROOT/datasets/biyivla-frozen"
export BIYI_ASSETS_BASE_DIR="$BIYI_ROOT/assets/stage1"
export BIYI_CHECKPOINT_BASE_DIR="$BIYI_ROOT/checkpoints/stage1"
```

Use a pinned dataset revision with the HF CLI/SDK or a verified transfer;
record repo ID, revision, and local root. Check `meta/info.json`,
`meta/episodes.jsonl`, `meta/tasks.jsonl`, all Parquet files, and video/image
files. Preflight checks episode metadata, finite 17D state/action, video
presence, and representative transforms. It cannot prove immutable Parquet or
media content; keep the frozen snapshot unchanged after stats computation.

### Step 11 — Freeze the train/validation split

For about 500 validated episodes, 450 train / 50 validation is a starting
proposal; use the actual validated count at Dataset Freeze. Record the exact
original train episode IDs, for example:

```bash
export BIYI_TRAIN_EPISODES='0,1,2'  # replace with the full frozen train list
```

No validation episode may enter normalization statistics. The full-dataset
train-split shortcut is allowed only if `meta/info.json` declares every episode
as train. Keep the split identifier and revision in run notes/W&B config.

### Step 12 — Compute production normalization statistics

```bash
scripts/biyi/train_pi05_stage1.sh compute-stats
```

Inspect the printed source manifest path, train episode list, train frame
count, and norm-stats SHA256. The manifest records repo ID, local root, selected
original IDs, train frame count, and source metadata hashes. Preflight compares
that manifest to the current dataset, split, and stats hash and fails on a
mismatch. Do not reuse episode 9, VLADexData smoke, pilot, or 18D stats.

### Step 13 — Run formal preflight

```bash
export BIYI_EXP_NAME='pi05_biyi_stage1_bs32_10k_v1'
export BIYI_NUM_TRAIN_STEPS=10000
scripts/biyi/train_pi05_stage1.sh preflight
```

Require `DATASET_CONTRACT`, `TRAIN_SPLIT`, `NORM_STATS`, `BASE_CHECKPOINT`,
`FULL_FINE_TUNING`, `RTC_DISABLED`, `WANDB_ENABLED`, `WANDB_PROJECT`,
`HF_AUTH`, `HF_DATASET_ACCESS`, `WANDB_AUTH`, and `TRAINING_RECIPE_GATE` to pass.
The recipe gate prints `NUM_TRAIN_STEPS`, `WARMUP_STEPS`, `WARMUP_RATIO`,
`DECAY_STEPS`, `PEAK_LR`, `END_LR`, `WEIGHT_DECAY`, and `SEED` checks.
Formal preflight requires online credentials; smoke preflight explicitly disables W&B and does
not check cloud credentials.

### Step 14 — Run H20 memory smoke at increasing global batches

Use distinct new experiment names for each attempt. The current H20 profile
runs 20–50 steps, saves a checkpoint, resumes it, and continues optimizer
steps; W&B is explicitly OFF. Test global batch 8, then 16, then 32:

| Global batch | Per rank on 2 GPUs |
| ---: | ---: |
| 8 | 4 |
| 16 | 8 |
| 32 | 16 |

```bash
BIYI_EXP_NAME=h20_memory_smoke_bs8 BIYI_BATCH_SIZE=8 scripts/biyi/train_pi05_stage1.sh h20-smoke
BIYI_EXP_NAME=h20_memory_smoke_bs16 BIYI_BATCH_SIZE=16 scripts/biyi/train_pi05_stage1.sh h20-smoke
BIYI_EXP_NAME=h20_memory_smoke_bs32 BIYI_BATCH_SIZE=32 scripts/biyi/train_pi05_stage1.sh h20-smoke
```

Each run must cover forward, backward, first AdamW `optimizer.step()`, and
multiple subsequent steps. The final candidate must save, resume, and keep
stepping. Record peak/allocated/reserved VRAM per GPU, stable step time after
warmup, checkpoint size, CPU RAM, and disk use. Select a batch only after no
OOM, AdamW state allocation, checkpoint save/resume, and reasonable VRAM
headroom. A single forward/backward pass is insufficient.

### Step 15 — Choose the final training budget

Record train episode count, train frame count, validated global batch, and
actual `N_train_samples`. Use
`samples_seen = global_batch × num_train_steps` and
`effective_exposure = samples_seen / N_train_samples` before freezing the
formal budget. At global batch 32, 5k/10k/15k/20k steps represent
160k/320k/480k/640k samples seen. The provisional primary endpoint is 10k,
not a hard maximum. If the frozen dataset makes 10k exposure too low, choose
15k or 20k **before** the first formal step; warmup remains 10% and decay ends
at the chosen budget. Do not change the schedule horizon during resume.

Checkpoint roles: 1k early optimization check, 3k early behavior, 5k
intermediate real-robot checkpoint, 8k late checkpoint, and 10k provisional
primary endpoint. Interpret checkpoints with validation and real-robot metrics,
not train loss alone.

### Step 16 — Start formal Stage-1 training

Only after all cloud and smoke gates pass, use a new formal experiment name:

```bash
BIYI_EXP_NAME=pi05_biyi_stage1_bs32_10k_v1 \
BIYI_BATCH_SIZE=32 \
BIYI_NUM_TRAIN_STEPS=10000 \
scripts/biyi/train_pi05_stage1.sh train
```

Replace batch 32 with the validated global batch and adjust the name. Formal
W&B is explicitly ON in `BiyiVLA-Stage1`. Record Git HEAD, config name,
dataset repo/revision/local root, split/manifest and train frame count,
global/per-rank batch, world size, learning rate, steps, action dimensions,
base path/SHA256, and norm-stats hash in run notes or W&B config. The stock
trainer sends its `TrainConfig` to W&B, but does not automatically include Git
HEAD, train frame count, split/manifest hash, or base SHA256 there; record those
fields manually with the run until a separate metadata change is reviewed.
The trainer logs training metrics; do not alter loss, optimizer, schedule, or
DDP to add metadata. Never reuse a smoke experiment directory for formal
training.

### Step 17 — Resume the same formal run

Use the original `BIYI_EXP_NAME`, frozen `BIYI_NUM_TRAIN_STEPS`, and checkpoint
base directory; confirm `wandb_id.txt` exists. For an interrupted 10k run:

```bash
BIYI_EXP_NAME=pi05_biyi_stage1_bs32_10k_v1 \
BIYI_BATCH_SIZE=32 \
BIYI_NUM_TRAIN_STEPS=10000 \
BIYI_RESUME=1 \
scripts/biyi/train_pi05_stage1.sh train
```

The stock W&B resume uses the saved run ID with `resume="must"`; training
loads the latest numbered checkpoint and optimizer state. Never resume a
formal run from a smoke directory.
The revised two-group Stage-1 AdamW state is incompatible with an older
single-group `optimizer.pt` or the previous 423/389 two-group layout. Start
formal training in a new experiment directory; do not resume either older
optimizer state. Loading the base `model.safetensors` weights is unaffected.

### Step 18 — Plan checkpoint storage explicitly

Estimate about 22 GB per complete model/optimizer checkpoint, then measure
the first H20 save. The PyTorch trainer accepts `keep_period` but does not
rotate checkpoints. Plan to keep the 1k, 3k, 5k, 8k, and 10k role checkpoints
on the data disk, subject to measured available space. Only consider
archiving/removing nondecision checkpoints such as 2k or 4k after
verifying a complete, readable save and any required remote copy. Do not automate deletion before
the save/resume smoke. Monitor `du -sh "$BIYI_CHECKPOINT_BASE_DIR"` and
`df -h "$BIYI_ROOT"` throughout training.

### Step 19 — Stage-1 PASS conditions and intentional gates

Cloud readiness requires: two H20s and topology confirmed, reproducible
Python/Torch/Transformers runtime, base SHA PASS, HF authentication and private
dataset access PASS, W&B authentication PASS, train-only production norm stats
PASS, formal preflight PASS, first AdamW step PASS, and checkpoint save/resume
PASS. The template is prepared, but formal training must wait for the dataset
freeze, exact train/val split, production stats, H20 memory smoke, and final
batch/step choice. These are intentional runtime gates, not code defects.

## Troubleshooting

- Formal W&B/HF failure: run `auth-check`; use online mode and inject valid
  credentials or official CLI login. Smoke runs do not require W&B credentials.
- Base mismatch: check `BIYI_BASE_CHECKPOINT_DIR`, size, and SHA256. Stop on
  mismatch; do not substitute a different checkpoint silently.
- Stats mismatch: verify dataset repo/revision/local root and train IDs, then
  recompute stats from the frozen train split.
- OOM or disk pressure: lower the global batch, repeat H20 smoke, and measure
  the first checkpoint before setting formal save/retention plans.
- W&B network drops: prefer direct cloud connectivity; an SSH reverse tunnel
  is dependent on the local computer remaining connected.

## 正式 100-episode checkpoint 三层存储

正式运行使用新增的可移植入口，默认实验名 `pi05_biyi_100ep_fft_v1`。
参见 [checkpoint 存储与恢复操作手册](README_checkpoint_storage.md)，包含持久盘设置、
冻结 revision、离线校验、私有 HF Model 归档门禁和跨机 optimizer 恢复。
原 smoke 流程保持原样；本入口默认只输出计划，不能把 smoke 结果当成正式归档。
