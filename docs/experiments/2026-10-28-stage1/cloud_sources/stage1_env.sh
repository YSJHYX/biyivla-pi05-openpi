# BiyiVLA A800 Stage-1 preparation. No credentials belong in this file.
export BIYI_ROOT=/root/autodl-tmp/biyivla
export HF_HOME="$BIYI_ROOT/hf_cache"
export HF_DATASETS_CACHE="$BIYI_ROOT/hf_cache/datasets"
export TORCH_HOME="$BIYI_ROOT/torch_cache"
export XDG_CACHE_HOME="$BIYI_ROOT/cache"
export OPENPI_DATA_HOME="$BIYI_ROOT/cache/openpi"
export UV_CACHE_DIR="$BIYI_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$BIYI_ROOT/envs/openpi-stage1"
export UV_PYTHON_INSTALL_DIR="$BIYI_ROOT/envs/uv-python"
export UV_TOOL_DIR="$BIYI_ROOT/envs/uv-tools"
export UV_TOOL_BIN_DIR="$BIYI_ROOT/bin"
export TMPDIR="$BIYI_ROOT/tmp"
export BIYI_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export BIYI_TORCHRUN="$UV_PROJECT_ENVIRONMENT/bin/torchrun"
export BIYI_BASE_CHECKPOINT_DIR="$BIYI_ROOT/models/pi05_base_pytorch"
export BIYI_DATASET_REPO_ID=PPPPPilot/BiyiVLA
export BIYI_DATASET_LOCAL_ROOT="$BIYI_ROOT/datasets/BiyiVLA"
export BIYI_TRAIN_EPISODES=0,1,2,3,4
export BIYI_ASSETS_BASE_DIR="$BIYI_ROOT/assets"
export BIYI_CHECKPOINT_BASE_DIR="$BIYI_ROOT/checkpoints"
export BIYI_EXP_NAME=local_smoke_a800_001
export BIYI_BATCH_SIZE=1
export BIYI_NUM_WORKERS=0
export BIYI_NUM_TRAIN_STEPS=2
export BIYI_SAVE_INTERVAL=2
export BIYI_LOG_INTERVAL=1
export WANDB_DIR="$BIYI_ROOT/logs"
export WANDB_CACHE_DIR="$BIYI_ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$BIYI_ROOT/cache/wandb-config"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$BIYI_ROOT/bin:$PATH"

# FFmpeg shared runtime on the data disk (Python lock unchanged).
for biyi_ffmpeg_lib in "$BIYI_ROOT/runtime/ffmpeg/usr/lib/x86_64-linux-gnu/pulseaudio" "$BIYI_ROOT/runtime/ffmpeg/lib/x86_64-linux-gnu" "$BIYI_ROOT/runtime/ffmpeg/usr/lib/x86_64-linux-gnu"; do
  if [ -d "$biyi_ffmpeg_lib" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$biyi_ffmpeg_lib:"*) ;;
      *) export LD_LIBRARY_PATH="$biyi_ffmpeg_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
done
unset biyi_ffmpeg_lib

# W&B credential is loaded from a separate root-only file.
source /root/autodl-tmp/biyivla/secrets/wandb.env
