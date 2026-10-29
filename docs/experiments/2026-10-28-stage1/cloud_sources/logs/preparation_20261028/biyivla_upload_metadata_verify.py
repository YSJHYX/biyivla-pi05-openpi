"""Upload reviewed archive; user explicitly declined full payload readback."""
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi, hf_hub_download
from scripts.biyi import checkpoint_store as store

OUT = Path('/root/autodl-tmp/biyivla/logs/preparation_20261028')
ARCHIVE = Path('/autodl-fs/data/biyivla_stage1_archives/pi05_biyi_117ep_fft_v2/step-9747-resume')
REPO = 'PPPPPilot/BiyiVLA-pi05-checkpoints'
PREFIX = 'resume/pi05_biyi_117ep_fft_v2/step-9747'


def status(phase, **extra):
    row = dict(phase=phase, at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    store.write_json(OUT / 'hf_upload_status.json', row)
    print(json.dumps(row), flush=True)


def main():
    logging.disable(logging.CRITICAL)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    bypass = 'localhost,127.0.0.1,.cdn.hf.co,.xethub.hf.co,.amazonaws.com'
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = bypass
    assert (OUT / 'final_checks.exit').read_text().strip() == '0'
    assert store.read_json(OUT / 'wandb_metric_audit.json')['state'] == 'finished'
    status('waiting_for_local_pack')
    for _ in range(180):
        state = store.read_json(OUT / 'hf_archive_status.json')
        if state['phase'] == 'packed_locally':
            break
        assert state['phase'] != 'failed_local_pack'
        time.sleep(5)
    else:
        raise TimeoutError('LocalPackTimeout')
    manifest = store.read_json(ARCHIVE / 'bundle_manifest.json')
    assert manifest['experiment'] == 'pi05_biyi_117ep_fft_v2' and manifest['step'] == 9747
    assert manifest['kind'] == 'resume'
    status('uploading', files=len(manifest['files']) + 1,
           bytes=sum(row['bytes'] for row in manifest['files'].values()))
    store.upload(ARCHIVE, REPO + ':pi05_biyi_117ep_fft_v2:9747:resume')
    receipt_path = ARCHIVE.with_name(ARCHIVE.name + '.hf-receipt.json')
    receipt = store.read_json(receipt_path)
    shutil.copyfile(receipt_path, OUT / 'hf_upload_receipt.json')
    oid = receipt['revision']
    status('remote_metadata_verification', revision=oid)
    api = HfApi()
    info = api.repo_info(REPO, repo_type='model', revision=oid, files_metadata=True)
    assert info.private is True and info.sha == oid
    remote = {row.rfilename: row for row in info.siblings}
    expected = set(manifest['files']) | {'bundle_manifest.json'}
    assert {name[len(PREFIX) + 1:] for name in remote if name.startswith(PREFIX + '/')} == expected
    lfs_count = 0
    blob_count = 0
    for name in sorted(expected):
        local = ARCHIVE / name
        entry = remote[PREFIX + '/' + name]
        assert entry.size == local.stat().st_size
        local_sha = manifest['files'][name]['sha256'] if name in manifest['files'] else receipt['bundle_manifest_sha256']
        if entry.lfs:
            sha = entry.lfs.sha256 if hasattr(entry.lfs, 'sha256') else entry.lfs['sha256']
            assert sha == local_sha
            lfs_count += 1
        else:
            data = local.read_bytes()
            sha = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            assert sha == entry.blob_id
            blob_count += 1
    # Only this small JSON manifest is downloaded. No model or optimizer readback.
    downloaded = Path(hf_hub_download(REPO, PREFIX + '/bundle_manifest.json',
                                     repo_type='model', revision=oid,
                                     local_dir=OUT / 'hf_manifest_readback'))
    assert store.digest(downloaded) == receipt['bundle_manifest_sha256']
    assert store.read_json(downloaded) == manifest
    latest = api.repo_info(REPO, repo_type='model')
    result = dict(passed=True, repo_id=REPO, private=info.private, revision=oid,
                  latest_head=latest.sha, prefix=PREFIX, files=len(expected),
                  full_readback_verified=False,
                  full_readback_status='Not performed: user explicitly requested upload without full readback.',
                  metadata_size_and_hashes_verified=True,
                  remote_manifest_download_and_sha256_verified=True,
                  lfs_hash_files=lfs_count, git_blob_hash_files=blob_count,
                  bytes=sum((ARCHIVE / name).stat().st_size for name in expected),
                  bundle_manifest_sha256=receipt['bundle_manifest_sha256'],
                  local_archive=str(ARCHIVE),
                  checkpoint_path=str(Path(store.read_json(OUT / 'formal_parameters.json')['run_dir']) / '9747'),
                  url=f'https://huggingface.co/{REPO}/tree/{oid}/{PREFIX}')
    store.write_json(OUT / 'hf_archive_verification.json', result)
    status('completed', **result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error_type=type(error).__name__)
        raise SystemExit(1) from None
