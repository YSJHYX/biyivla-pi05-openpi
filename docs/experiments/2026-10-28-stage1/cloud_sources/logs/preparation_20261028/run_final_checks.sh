#!/usr/bin/env bash
set -euo pipefail
out=/root/autodl-tmp/biyivla/logs/preparation_20261028
trap 'status=$?; printf "%s\n" "$status" > "$out/final_checks.exit"' EXIT
source "$out/preparation_env.sh"
cd /root/autodl-tmp/biyivla/openpi
"$BIYI_PYTHON" "$out/biyivla_summarize_experiment.py" > "$out/training_summary.log" 2>&1
"$BIYI_PYTHON" "$out/biyivla_final_technical_check.py" > "$out/final_technical_check.log" 2>&1
"$BIYI_PYTHON" "$out/biyivla_wandb_metric_audit.py" > "$out/wandb_final_audit.log" 2>&1
