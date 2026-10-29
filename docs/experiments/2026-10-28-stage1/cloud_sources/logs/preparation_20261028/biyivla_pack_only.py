"""Archive only the completed reviewed formal experiment, then full HF readback."""
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

from scripts.biyi import checkpoint_store as store

BASE = Path('/root/autodl-tmp/biyivla')
OUT = BASE / 'logs/preparation_20261028'
ARCHIVE = Path('/autodl-fs/data/biyivla_stage1_archives/pi05_biyi_117ep_fft_v2/step-9747-resume')
REPO = 'PPPPPilot/BiyiVLA-pi05-checkpoints'
PREFIX = 'resume/pi05_biyi_117ep_fft_v2/step-9747'


def status(phase, **extra):
    row = dict(phase=phase, at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    store.write_json(OUT / 'hf_archive_status.json', row)
    print(json.dumps(row), flush=True)


def main():
    logging.disable(logging.CRITICAL)  # do not log credential-bearing HTTP URLs
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    bypass = 'localhost,127.0.0.1,.cdn.hf.co,.xethub.hf.co,.amazonaws.com'
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = bypass
    assert (OUT / 'final_checks.exit').read_text().strip() == '0'
    assert store.read_json(OUT / 'wandb_metric_audit.json')['state'] == 'finished'
    params = store.read_json(OUT / 'formal_parameters.json')
    evidence = OUT / 'final_evidence'
    acceptance = store.read_json(evidence / 'acceptance.json')
    additions = ['pi05_biyi_117ep_fft_v2.json', 'formal_plan.json', 'FINAL_REPORT_CN.md']
    for name in additions:
        shutil.copyfile(OUT / name, evidence / name)
    # Only an explicit whitelist of non-secret values is exported.
    names = ['BIYI_ROOT', 'BIYI_PYTHON', 'BIYI_TORCHRUN', 'BIYI_BASE_CHECKPOINT_DIR',
             'BIYI_DATASET_REPO_ID', 'BIYI_DATASET_LOCAL_ROOT', 'BIYI_DATASET_REVISION',
             'BIYI_TRAIN_EPISODES', 'BIYI_ASSETS_BASE_DIR', 'BIYI_CHECKPOINT_BASE_DIR',
             'OPENPI_DATA_HOME', 'HF_HOME', 'HF_DATASETS_CACHE', 'PYTHONPATH', 'JAX_PLATFORMS',
             'WANDB_ENTITY', 'WANDB_PROJECT']
    runtime = {key: os.environ[key] for key in names if key in os.environ}
    runtime.update(BIYI_FORMAL_RECIPE=str(OUT / additions[0]),
                   BIYI_EXP_NAME='pi05_biyi_117ep_fft_v2', BIYI_BATCH_SIZE='8',
                   BIYI_NUM_TRAIN_STEPS='9747', BIYI_SAVE_INTERVAL='3249',
                   BIYI_LOG_INTERVAL='1', BIYI_NUM_WORKERS='0', BIYI_NUM_GPUS='1',
                   BIYI_MEMORY_SMOKE_RUN_DIR=params['chosen_smoke_dir'],
                   BIYI_STORAGE_MODE='backup', BIYI_BACKUP_ROOT=params['backup_root'],
                   WANDB_MODE='online', TOKENIZERS_PARALLELISM='false')
    store.write_json(evidence / 'runtime_environment_without_secrets.json', runtime)
    additions.append('runtime_environment_without_secrets.json')
    for name in additions:
        if name not in acceptance['evidence']:
            acceptance['evidence'].append(name)
        acceptance['evidence_descriptions'][name] = name
    with (evidence / 'EXPERIMENT_CHECKLIST.md').open('a') as stream:
        stream.write('''

## Exact invocation and credential-free runtime

The bundled `pi05_biyi_117ep_fft_v2.json` is the exact external recipe.
`runtime_environment_without_secrets.json` contains the explicit non-secret
runtime and original path bindings, including the required completed memory smoke.
`formal_plan.json` records the resolved command and model/optimizer flags.
On the retained server, after secure HF/W&B authentication:

```bash
source /root/autodl-tmp/biyivla/logs/preparation_20261028/preparation_env.sh
cd /root/autodl-tmp/biyivla/openpi
# Export all entries of runtime_environment_without_secrets.json into this shell.
# For a fresh repeat, first select new versioned recipe/experiment/output roots.
bash scripts/biyi/train_pi05_stage1.sh preflight
bash scripts/biyi/train_pi05_stage1.sh train
```

The original run already exists and must not be launched over. The source/base/
data/norm hashes, all 117 selected episodes, measured memory-smoke signature,
backup storage probe, and strict formal preflight must pass before a repeat.
Changing paths requires the project's explicit verified restore/rebinding flow.
Raw data, secrets, shell history and environment dumps are excluded from this
archive; only the declared credential-free runtime whitelist is included.
''')
    store.write_json(evidence / 'acceptance.json', acceptance)
    assert shutil.disk_usage(ARCHIVE.parent.parent.parent).free > 45 * 2**30
    status('packing')
    store.pack(params['run_dir'], 9747, evidence / 'acceptance.json', ARCHIVE, include_optimizer=True)
    manifest = store.verify_bundle(ARCHIVE)
    status('packed_locally', files=len(manifest['files']) + 1, bytes=sum(row['bytes'] for row in manifest['files'].values()), path=str(ARCHIVE))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed_local_pack', error_type=type(error).__name__)
        raise SystemExit(1) from None
