#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

profile="${1:-}"
if [[ ! "$profile" =~ ^(auth-check|preflight|compute-stats|local-smoke|h20-smoke|train)$ ]]; then
  echo "Usage: $0 {auth-check|preflight|compute-stats|local-smoke|h20-smoke|train}" >&2
  exit 2
fi

python_bin="${BIYI_PYTHON:-.venv/bin/python}"
torchrun_bin="${BIYI_TORCHRUN:-.venv/bin/torchrun}"

if [[ "$profile" == "auth-check" ]]; then
  "$python_bin" -m scripts.biyi.check_cloud_auth
  exit
fi

: "${BIYI_DATASET_REPO_ID:?Set the exact frozen LeRobot repo ID}"
: "${BIYI_DATASET_LOCAL_ROOT:?Set the local frozen train dataset path}"
: "${BIYI_ASSETS_BASE_DIR:?Set a dedicated Stage-1 stats directory}"
: "${BIYI_CHECKPOINT_BASE_DIR:?Set a dedicated Stage-1 checkpoint directory}"

if [[ "$profile" == "compute-stats" ]]; then
  "$python_bin" -m scripts.biyi.compute_stage1_norm_stats
  exit
fi

peak_lr=2.5e-5
end_lr=2.5e-6

set_formal_schedule() {
  if [[ ! "$num_steps" =~ ^[1-9][0-9]*$ ]] || (( num_steps < 10 )); then
    echo "Formal training requires an integer BIYI_NUM_TRAIN_STEPS >= 10" >&2
    exit 2
  fi
  # Nearest integer for positive steps, with half steps rounded upward.
  warmup_steps=$(( (num_steps + 5) / 10 ))
  decay_steps="$num_steps"
}

run_formal_preflight() {
  "$python_bin" -m scripts.biyi.preflight_pi05_stage1 \
    --formal --wandb-enabled --project-name BiyiVLA-Stage1 --exp-name "$BIYI_EXP_NAME" \
    --num-train-steps "$num_steps" --warmup-steps "$warmup_steps" --decay-steps "$decay_steps" \
    --peak-lr "$peak_lr" --end-lr "$end_lr"
}

if [[ "$profile" == "preflight" ]]; then
  : "${BIYI_EXP_NAME:?Set the formal experiment name}"
  num_steps="${BIYI_NUM_TRAIN_STEPS:-10000}"
  set_formal_schedule
  run_formal_preflight
  exit
fi

case "$profile" in
  local-smoke)
    exp_name="${BIYI_EXP_NAME:-local_smoke}"
    batch_size="${BIYI_BATCH_SIZE:-1}"
    num_workers="${BIYI_NUM_WORKERS:-0}"
    num_steps="${BIYI_NUM_TRAIN_STEPS:-3}"
    save_interval="${BIYI_SAVE_INTERVAL:-2}"
    if (( num_steps < 2 || num_steps > 10 )); then
      echo "local-smoke requires 2-10 steps" >&2
      exit 2
    fi
    launcher=("$python_bin")
    ;;
  h20-smoke)
    exp_name="${BIYI_EXP_NAME:-h20_memory_smoke}"
    batch_size="${BIYI_BATCH_SIZE:-32}"
    num_workers="${BIYI_NUM_WORKERS:-2}"
    num_steps="${BIYI_NUM_TRAIN_STEPS:-30}"
    save_interval="${BIYI_SAVE_INTERVAL:-10}"
    if (( num_steps < 20 || num_steps > 50 || num_steps % 2 != 0 )); then
      echo "h20-smoke requires an even total of 20-50 steps" >&2
      exit 2
    fi
    launcher=("$torchrun_bin" --standalone --nnodes=1 --nproc_per_node=2)
    ;;
  train)
    : "${BIYI_EXP_NAME:?Set a unique formal run name}"
    : "${BIYI_NUM_TRAIN_STEPS:?Set steps after dataset and memory sizing}"
    exp_name="$BIYI_EXP_NAME"
    batch_size="${BIYI_BATCH_SIZE:-32}"
    num_workers="${BIYI_NUM_WORKERS:-2}"
    num_steps="$BIYI_NUM_TRAIN_STEPS"
    save_interval="${BIYI_SAVE_INTERVAL:-1000}"
    if [[ "$exp_name" == local_smoke* || "$exp_name" == h20_memory_smoke* ]]; then
      echo "Formal experiment name must differ from smoke experiment names" >&2
      exit 2
    fi
    launcher=("$torchrun_bin" --standalone --nnodes=1 --nproc_per_node=2)
    ;;
esac

if [[ "$profile" == "train" ]]; then
  set_formal_schedule
else
  # Short smoke runs exercise optimizer and checkpoint behavior independently
  # of the formal 10% warmup recipe, including across H20 save/resume.
  warmup_steps=1
  decay_steps="$num_steps"
fi

if (( batch_size < 1 )) || { [[ "$profile" != "local-smoke" ]] && (( batch_size % 2 != 0 )); }; then
  echo "Global batch must be positive and divisible by 2 for DDP" >&2
  exit 2
fi

if [[ "$profile" == "train" ]]; then
  run_formal_preflight
else
  "$python_bin" -m scripts.biyi.preflight_pi05_stage1 --smoke --no-wandb-enabled
fi

run_dir="$BIYI_CHECKPOINT_BASE_DIR/pi05_biyi_stage1/$exp_name"
if [[ "$profile" == "train" && "${BIYI_RESUME:-0}" == "1" ]]; then
  if [[ ! -s "$run_dir/wandb_id.txt" ]]; then
    echo "Formal resume requires an existing wandb_id.txt; smoke runs cannot become formal runs" >&2
    exit 2
  fi
else
  if [[ -e "$run_dir" ]]; then
    echo "Run directory already exists: $run_dir" >&2
    exit 2
  fi
fi

args=(
  scripts/train_pytorch.py pi05_biyi_stage1
  --exp-name "$exp_name"
  --batch-size "$batch_size"
  --num-workers "$num_workers"
  --num-train-steps "$num_steps"
  --lr-schedule.warmup-steps "$warmup_steps"
  --lr-schedule.decay-steps "$decay_steps"
  --lr-schedule.peak-lr "$peak_lr"
  --lr-schedule.decay-lr "$end_lr"
  --save-interval "$save_interval"
  --keep-period "${BIYI_KEEP_PERIOD:-5000}"
  --assets-base-dir "$BIYI_ASSETS_BASE_DIR"
  --checkpoint-base-dir "$BIYI_CHECKPOINT_BASE_DIR"
  --log-interval "${BIYI_LOG_INTERVAL:-5}"
)

if [[ "$profile" == "train" ]]; then
  args+=(--wandb-enabled --project-name BiyiVLA-Stage1)
else
  args+=(--no-wandb-enabled)
fi

if [[ "$profile" == "h20-smoke" ]]; then
  first_steps=$((num_steps - 10))
  args[9]="$first_steps" # --num-train-steps value; schedule keeps the full smoke horizon.
  "${launcher[@]}" "${args[@]}"
  args[9]="$num_steps"
  "${launcher[@]}" "${args[@]}" --resume
  du -sh "$run_dir"/*
elif [[ "$profile" == "train" && "${BIYI_RESUME:-0}" == "1" ]]; then
  "${launcher[@]}" "${args[@]}" --resume
else
  "${launcher[@]}" "${args[@]}"
fi
