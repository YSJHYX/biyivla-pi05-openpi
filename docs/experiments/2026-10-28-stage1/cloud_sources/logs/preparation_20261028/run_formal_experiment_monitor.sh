#!/usr/bin/env bash
set -euo pipefail
out=/root/autodl-tmp/biyivla/logs/preparation_20261028
trap 'status=$?; printf "%s\n" "$status" > "$out/formal_monitor.exit"' EXIT
source "$out/preparation_env.sh"
cd /root/autodl-tmp/biyivla/openpi
"$BIYI_PYTHON" "$out/biyivla_formal_experiment_monitor.py" > "$out/formal_monitor.log" 2>&1
