"""Create reviewed text/JSON evidence after completed numerical checks."""
import datetime
from pathlib import Path
import shutil

from scripts.biyi import checkpoint_store as store

BASE = Path('/root/autodl-tmp/biyivla')
OUT = BASE / 'logs/preparation_20261028'


def main():
    params = store.read_json(OUT / 'formal_parameters.json')
    summary = store.read_json(OUT / 'training_summary.json')
    offline = store.read_json(OUT / 'final_offline_inference.json')
    source = store.read_json(OUT / 'source_reproducibility.json')
    assert all(offline['checks'].values()) and source['exact_commit_reproduced']
    assert summary['final_global_step'] == params['num_train_steps']
    assert summary['loss']['finite'] and summary['grad_norm_before_clip']['finite']
    run = Path(params['run_dir'])
    step = params['num_train_steps']
    checkpoint = run / str(step)
    store.seal_checkpoint(run, step)
    evidence = OUT / 'final_evidence'
    evidence.mkdir(exist_ok=True)
    files = ['final_offline_inference.json', 'training_summary.json', 'formal_parameters.json', 'norm_and_loader_verification.json', 'source_reproducibility.json', 'source_changes.txt', 'environment_audit.json', 'training_numeric_history.json', 'TRAINING_REPORT.md', 'gpu_sizing.json', 'wandb_metric_audit.json', 'numeric_audit.json', 'video_audit.json']
    for name in files:
        shutil.copyfile(OUT / name, evidence / name)
    shutil.copyfile(BASE / 'datasets/frozen' / params['dataset_revision'] / 'dataset_lock.json', evidence / 'dataset_lock_for_archive.json')
    files.append('dataset_lock_for_archive.json')
    lines = [
        '# BiyiVLA Stage-1 FFT reproducibility checklist', '',
        '## Immutable identity', '',
        f"- Experiment: `{run.name}`; source `{params['source_commit']}`.",
        '- GitHub base: `897bd7f8822e011cb970bb93b312f93824105353` in `https://github.com/YSJHYX/biyivla-pi05-openpi`.',
        f"- Dataset: `PPPPPilot/BiyiVLA@{params['dataset_revision']}`;117episodes0..116,77,975eligible starting frames; all training, no held-out split.",
        f"- Base weights SHA256: `{params['base_model_sha256']}`. Use the retained `/root/autodl-tmp/biyivla/models/pi05_base_pytorch/model.safetensors` or an exact checksum-matching copy. A different external base is not interchangeable.",
        f"- Norm SHA256: `{params['norm_stats_sha256']}`. Statistics use every train state and the episode-clamped50-step action chunks. action[9] is intentionally constant0.",
        '- Full effective parameters: `formal_parameters.json`; exact source reconstruction and dependency-lock checksum: `source_reproducibility.json`.', '',
        '## Exact source reconstruction', '',
        'Clone the GitHub repository, checkout the pinned base, and apply the reviewed patch:', '',
        '```bash',
        'git checkout --detach 897bd7f8822e011cb970bb93b312f93824105353',
        'git -c user.name=Codex -c user.email=codex@openai.com am --committer-date-is-author-date source_changes.txt',
        'git rev-parse HEAD',
        '```', '',
        f"The result must be `{params['source_commit']}`; this was actually tested in a temporary checkout. Keep uv.lock unchanged and use the environment manifest and project frozen setup/Transformers replacement instructions.", '',
        '## Restore inputs and train', '',
        '1. Authenticate HF normally; never include credentials in code/logs/artifacts.',
        f"2. Download the exact dataset revision and verify every file against dataset_lock_for_archive.json. The original frozen root is `/root/autodl-tmp/biyivla/datasets/frozen/{params['dataset_revision']}`. Keep download bookkeeping outside the frozen root. Formal preflight checks the full lock.",
        '3. Restore assets/biyi_stage1/norm_stats.json and the train stats manifest under the declared assets root. For a different path, use the existing explicit verified restore/rebinding workflow; do not silently change provenance.',
        '4. Check the exact base SHA above; preserve the stock32D model/17D physical adapter, horizon50 and head/wrist mapping.',
        '5. Inject W&B with export WANDB_API_KEY in a secure environment. The frozen0.19.11 CLI40-character writer is incompatible with this account key; no credential is included here. Use online mode, entity y8hong-heymandex, project BiyiVLA-Stage1.',
        f"6. Use the existing `bash scripts/biyi/train_pi05_stage1.sh train` and versioned117-episode recipe: batch{params['global_batch_size']}, oneA80080GB, {step}steps, warmup{params['warmup_steps']}, LR2.5e-5 to2.5e-6 cosine, AdamW beta(.9,.95), epsilon1e-8, matrix WD.05 with bias/0D/1D exclusions, clip1, stock BF16 dtype policy, gradient checkpointing, horizon50, seed42, workers0, log every step. All117episodes must remain selected.",
        '7. Use a fresh run/checkpoint root or new versioned experiment name. Repeat the existing exact-source/data/stats/batch GPU smoke and formal preflight; never overwrite this experiment.', '',
        'DataLoader shuffle seed42 and drop_last=True were used. A batch8 pass drops7starting frames; the next shuffled pass supplies the final batch. Total optimizer samples77,976 /77,975 =1.000013nominal epochs. Horizon50vectors overlap and are not50independent epochs.', '',
        '## Save, resume and evidence', '',
        f"- Complete saves: {params['expected_save_steps']}; all have model, optimizer, metadata, norm stats and independent checksum-verified backups.",
        f"- Local run: `{run}`.",
        f"- Independent backups: `/autodl-fs/data/biyivla_stage1_backups/pi05_biyi_stage1/{run.name}`.",
        '- Resume through the existing versioned launcher and original W&B ID6zpgxsbv; source/recipe/data/base/norm identities must match. Stock saves do not preserve every RNG/data-iterator state, so resume is not guaranteed bitwise identical.',
        '- This is a full resume archive, not a smoke model or a hardware deployment release. The immutable HF commit/upload/readback receipts are produced separately after verified upload.', '',
        'Automated technical checks cover all saved weights finite, optimizer step/state, actual stock offline policy load and three fixed real observations returning finite50x17actions. No robot commands were sent. These are training observations: no held-out generalization or physical task-success claim is made. Predicted range violations, if any, are retained without clipping in the inference receipt.',
        'training_numeric_history.json contains every step; training_summary.json records loss/gradient/system peaks and separates observed timing where available. GPU/system/provenance fields were observed in the live W&B service. Any W&B synchronization wait does not add optimizer steps.'
    ]
    (evidence / 'EXPERIMENT_CHECKLIST.md').write_text('\n'.join(lines) + '\n')
    files.append('EXPERIMENT_CHECKLIST.md')
    acceptance = {
        'accepted': True,
        'reviewer': 'Codex automated offline technical review under user-delegated archival authorization',
        'reviewed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'checkpoint_manifest_sha256': store.digest(checkpoint / 'checkpoint_manifest.json'),
        'checks': offline['checks'],
        'scope': 'Offline numerical/shape/provenance/completed-training integrity only; no physical task-success acceptance or deployment release.',
        'evidence': files,
        'evidence_descriptions': {name: name for name in files},
    }
    store.write_json(evidence / 'acceptance.json', acceptance)
    print('ARCHIVE_EVIDENCE_READY', evidence, 'files', len(files), flush=True)


if __name__ == '__main__':
    main()
