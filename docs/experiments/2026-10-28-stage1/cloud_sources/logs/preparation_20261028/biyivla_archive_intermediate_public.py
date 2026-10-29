"""Archive both user-requested intermediate checkpoints; serialize HF commits."""
import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess

from scripts.biyi import checkpoint_store as store

BASE = Path('/root/autodl-tmp/biyivla')
OUT = BASE / 'logs/preparation_20261028'
ROOT = Path('/autodl-fs/data/biyivla_stage1_archives/pi05_biyi_117ep_fft_v2')
PARAMS = store.read_json(OUT / 'formal_parameters.json')
RUN = Path(PARAMS['run_dir'])


def status(phase, **extra):
    value = dict(phase=phase, at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    store.write_json(OUT / 'intermediate_archive_status.json', value)
    print(json.dumps(value), flush=True)


def prepare(step):
    archive = ROOT / f'step-{step}-resume'
    if archive.exists():
        report = store.read_json(OUT / f'checkpoint_{step}_offline_inference.json')
        assert report['checkpoint_step'] == step and all(report['checks'].values())
        assert store.read_json(archive / 'bundle_manifest.json')['step'] == step
        status('reusing_verified_local_bundle', step=step, path=str(archive))
        return
    status('offline_checkpoint_checks', step=step)
    with (OUT / f'checkpoint_{step}_technical.log').open('w') as log:
        subprocess.run([os.environ['BIYI_PYTHON'], str(OUT / 'biyivla_checkpoint_technical_check.py'),
                        '--step', str(step)], cwd=BASE / 'openpi', stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    report = store.read_json(OUT / f'checkpoint_{step}_offline_inference.json')
    assert report['checkpoint_step'] == step and all(report['checks'].values())
    assert report['optimizer_steps'] == [step] and report['optimizer_state_count'] == 805
    assert report['finite_model_tensor_count'] == 812
    store.seal_checkpoint(RUN, step)
    evidence = OUT / f'checkpoint_{step}_evidence'
    evidence.mkdir(exist_ok=False)
    shutil.copyfile(OUT / f'checkpoint_{step}_offline_inference.json', evidence / 'checkpoint_offline_inference.json')
    store.write_json(evidence / 'checkpoint_scope.json', dict(
        checkpoint_step=step, training_recipe_total_steps=9747, kind='intermediate_resume_checkpoint',
        experiment=RUN.name, final_training_completed=True,
        acceptance_scope='Offline numerical/shape/provenance integrity only; not task success or deployment release.',
        full_hf_payload_readback='Not requested; explicitly declined by user. Metadata/hash and small manifest checks only.'))
    names = ['checkpoint_offline_inference.json', 'checkpoint_scope.json']
    for name in ['formal_parameters.json', 'source_changes.txt', 'source_reproducibility.json',
                 'source_sync_receipt.json', 'environment_audit.json', 'norm_and_loader_verification.json',
                 'numeric_source_inventory.json', 'training_summary.json', 'training_numeric_history.json',
                 'gpu_sizing.json', 'numeric_audit.json', 'video_audit.json', 'pi05_biyi_117ep_fft_v2.json']:
        shutil.copyfile(OUT / name, evidence / name)
        names.append(name)
    for name in ['dataset_lock_for_archive.json', 'EXPERIMENT_CHECKLIST.md', 'runtime_environment_without_secrets.json']:
        shutil.copyfile(OUT / 'final_evidence' / name, evidence / name)
        names.append(name)
    acceptance = dict(accepted=True,
        reviewer='Codex automated offline technical review under explicit user archive authorization',
        reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint_manifest_sha256=store.digest(RUN / str(step) / 'checkpoint_manifest.json'),
        checks=report['checks'], scope=report['scope'], evidence=names,
        evidence_descriptions={name: name for name in names})
    store.write_json(evidence / 'acceptance.json', acceptance)
    status('packing_local_checkpoint', step=step)
    archive = ROOT / f'step-{step}-resume'
    store.pack(RUN, step, evidence / 'acceptance.json', archive, include_optimizer=True)
    status('local_checkpoint_bundle_complete', step=step, path=str(archive))


def start_upload(step):
    log = (OUT / f'hf_upload_{step}.log').open('w')
    process = subprocess.Popen([os.environ['BIYI_PYTHON'], str(OUT / 'biyivla_upload_metadata_public_step.py'),
                                '--step', str(step), '--allow-public', '--confirm',
                                f'{store.MODEL_REPO}:{RUN.name}:{step}:resume:public'], cwd=BASE / 'openpi', stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT)
    status('uploading_checkpoint', step=step, upload_pid=process.pid)
    return process, log


def wait_upload(step, process, log):
    result = process.wait()
    log.close()
    assert result == 0, f'Upload process failed for {step}'
    verified = store.read_json(OUT / f'hf_archive_verification_{step}.json')
    assert verified['passed'] and not verified['full_readback_verified']
    status('checkpoint_hf_verified', step=step, revision=verified['revision'])


def main():
    logging.disable(logging.CRITICAL)
    assert all((ROOT / f'step-{step}-resume').exists() for step in [3249,6498])
    assert (OUT / 'final_checks.exit').read_text().strip() == '0'
    assert store.read_json(OUT / 'source_sync_receipt.json')['source_commit'] == PARAMS['source_commit']
    for step in [3249,6498]:
        receipt = OUT / f'hf_archive_verification_{step}.json'
        if receipt.exists() and store.read_json(receipt).get('passed') is True:
            status('reusing_verified_hf_checkpoint', step=step, revision=store.read_json(receipt)['revision'])
            continue
        prepare(step)
        process, log = start_upload(step)
        wait_upload(step, process, log)
    from huggingface_hub import HfApi
    api = HfApi()
    head = api.repo_info(store.MODEL_REPO, repo_type='model', files_metadata=True)
    assert head.private is False, 'User requested that the repository remain Public'
    remote = {entry.rfilename: entry for entry in head.siblings}
    checkpoints = []
    for step in [3249, 6498, 9747]:
        path = ROOT / f'step-{step}-resume'
        manifest = store.read_json(path / 'bundle_manifest.json')
        prefix = f'resume/{RUN.name}/step-{step}'
        names = set(manifest['files']) | {'bundle_manifest.json'}
        assert {n[len(prefix) + 1:] for n in remote if n.startswith(prefix + '/')} == names
        for name in names:
            entry = remote[prefix + '/' + name]
            local = path / name
            assert entry.size == local.stat().st_size
            if entry.lfs:
                sha = entry.lfs.sha256 if hasattr(entry.lfs, 'sha256') else entry.lfs['sha256']
                expected = manifest['files'][name]['sha256'] if name in manifest['files'] else store.digest(local)
                assert sha == expected
        verification = store.read_json(OUT / (f'hf_archive_verification_{step}.json' if step != 9747 else 'hf_archive_verification.json'))
        assert verification['passed']
        checkpoints.append(dict(step=step, upload_revision=verification['revision'],
                                present_at_final_head=True, metadata_and_manifest_verified=True,
                                path=prefix, local_checkpoint=str(RUN / str(step)),
                                url=f'https://huggingface.co/{store.MODEL_REPO}/tree/{head.sha}/{prefix}'))
    result = dict(passed=True, private=head.private, repo_id=store.MODEL_REPO, final_head=head.sha,
                  public_upload_explicitly_authorized_by_user=True,
                  full_readback_verified=False, full_readback_status='Explicitly declined by user.',
                  checkpoints=checkpoints)
    store.write_json(OUT / 'hf_all_checkpoints_verification.json', result)
    status('completed', **result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error_type=type(error).__name__)
        raise SystemExit(1) from None
