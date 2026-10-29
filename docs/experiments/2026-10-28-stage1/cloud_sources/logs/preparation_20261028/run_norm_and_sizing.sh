#!/usr/bin/env bash
set -euo pipefail
out=/root/autodl-tmp/biyivla/logs/preparation_20261028
trap 'status=$?; printf "%s\n" "$status" > "$out/norm_and_sizing.exit"' EXIT
source "$out/preparation_env.sh"
cd /root/autodl-tmp/biyivla/openpi
"$BIYI_PYTHON" -m scripts.biyi.compute_stage1_norm_stats --numeric-only --source-inventory "$out/numeric_source_inventory.json" > "$out/norm_stats.log" 2>&1
"$BIYI_PYTHON" "$out/biyivla_verify_norm_and_loader.py" > "$out/norm_verification.log" 2>&1
"$BIYI_PYTHON" "$out/biyivla_gpu_sizing.py" > "$out/gpu_sizing.log" 2>&1
