#!/usr/bin/env bash
set -euo pipefail
out=/root/autodl-tmp/biyivla/logs/preparation_20261028
trap 'status=$?; printf "%s\n" "$status" > "$out/optional_batch8.exit"' EXIT
while ! test -f "$out/norm_and_sizing.exit"; do sleep 5; done
test "$(cat "$out/norm_and_sizing.exit")" = 0
source "$out/preparation_env.sh"
cd /root/autodl-tmp/biyivla/openpi
"$BIYI_PYTHON" - <<'PY'
from pathlib import Path
import json
out=Path('/root/autodl-tmp/biyivla/logs/preparation_20261028')
rows=json.loads((out/'gpu_sizing.json').read_text());row=next(x for x in rows if x['batch_size']==4)
assert row['pass'] and row['initial_memory']['peak_reserved'] < .75*row['initial_memory']['total_memory'], 'Batch4 lacks sufficient headroom for optional batch8'
PY
export BIYI_SIZING_BATCHES=8
export BIYI_CHECKPOINT_BASE_DIR=/autodl-fs/data/biyivla_smoke_records_20261028
"$BIYI_PYTHON" "$out/biyivla_gpu_sizing_optional8.py" > "$out/gpu_sizing_optional8.log" 2>&1
