"""Upload reviewed archive; user explicitly declined full payload readback."""
import argparse
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

# Use the already-installed official concurrent multipart backend for new uploads.
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub import constants
from scripts.biyi import checkpoint_store as store

parser=argparse.ArgumentParser()
parser.add_argument('--step',type=int,choices=[3249,6498],required=True)
parser.add_argument('--allow-public',action='store_true')
parser.add_argument('--confirm',required=True)
ARGS=parser.parse_args();STEP=ARGS.step

OUT = Path('/root/autodl-tmp/biyivla/logs/preparation_20261028')
ARCHIVE = Path(f'/autodl-fs/data/biyivla_stage1_archives/pi05_biyi_117ep_fft_v2/step-{STEP}-resume')
REPO = 'PPPPPilot/BiyiVLA-pi05-checkpoints'
PREFIX = f'resume/pi05_biyi_117ep_fft_v2/step-{STEP}'


def status(phase, **extra):
    row = dict(phase=phase, at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    store.write_json(OUT / f'hf_upload_status_{STEP}.json', row)
    print(json.dumps(row), flush=True)


def upload_explicitly_authorized_public_bundle():
    """User explicitly overrode the original private-only destination policy."""
    manifest = store.verify_bundle(ARCHIVE)
    phrase = f'{REPO}:pi05_biyi_117ep_fft_v2:{STEP}:resume:public'
    store.require(ARGS.allow_public and ARGS.confirm == phrase, 'Explicit public upload authorization required')
    store.require(manifest['step'] == STEP and manifest['kind'] == 'resume', 'Wrong requested checkpoint')
    api = HfApi()
    info = api.repo_info(REPO, repo_type='model')
    existing = api.list_repo_files(REPO, repo_type='model', revision=info.sha)
    store.require(not any(name.startswith(PREFIX + '/') for name in existing), 'Remote release already exists')
    names = [*manifest['files'], 'bundle_manifest.json']
    operations = [CommitOperationAdd(path_in_repo=f'{PREFIX}/{name}',
                  path_or_fileobj=str(store.file_at(ARCHIVE, name))) for name in sorted(names)]
    result = api.create_commit(REPO, repo_type='model', operations=operations,
                parent_commit=info.sha,
                commit_message=f'Archive user-authorized public resume {manifest["experiment"]} step {STEP}')
    receipt = dict(repo_id=REPO, repo_type='model', revision=result.oid, path=PREFIX,
                   bundle_manifest_sha256=store.digest(ARCHIVE / 'bundle_manifest.json'),
                   destination_private_at_upload=info.private,
                   public_upload_explicitly_authorized_by_user=True)
    store.write_json(ARCHIVE.with_name(ARCHIVE.name + '.hf-receipt.json'), receipt)
    print(json.dumps(receipt), flush=True)


def main():
    logging.disable(logging.CRITICAL)
    # Limit the existing official transport per file; preserve SDK LFS ETag validation.
    import hf_transfer
    import huggingface_hub.lfs as hub_lfs

    def bounded_multipart(operation, sorted_parts_urls, chunk_size):
        return hf_transfer.multipart_upload(
            file_path=str(operation.path_or_fileobj), parts_urls=sorted_parts_urls,
            chunk_size=chunk_size, max_files=8, parallel_failures=7, max_retries=5)

    hub_lfs._upload_parts_hf_transfer = bounded_multipart
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    bypass = 'localhost,127.0.0.1,.cdn.hf.co,.xethub.hf.co,.amazonaws.com'
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = bypass
    assert (OUT / 'final_checks.exit').read_text().strip() == '0'
    assert store.read_json(OUT / 'wandb_metric_audit.json')['state'] == 'finished'
    assert constants.HF_HUB_ENABLE_HF_TRANSFER
    manifest = store.read_json(ARCHIVE / 'bundle_manifest.json')
    assert manifest['experiment'] == 'pi05_biyi_117ep_fft_v2' and manifest['step'] == STEP
    assert manifest['kind'] == 'resume'
    status('uploading', transport='existing_hf_transfer_multipart_max_files_8', files=len(manifest['files']) + 1,
           bytes=sum(row['bytes'] for row in manifest['files'].values()))
    upload_explicitly_authorized_public_bundle()
    receipt_path = ARCHIVE.with_name(ARCHIVE.name + '.hf-receipt.json')
    receipt = store.read_json(receipt_path)
    shutil.copyfile(receipt_path, OUT / f'hf_upload_receipt_{STEP}.json')
    oid = receipt['revision']
    status('remote_metadata_verification', revision=oid)
    api = HfApi()
    info = api.repo_info(REPO, repo_type='model', revision=oid, files_metadata=True)
    assert info.sha == oid
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
                  public_upload_explicitly_authorized_by_user=True,
                  transport='existing_official_hf_transfer_multipart_max_files_8',
                  latest_head=latest.sha, prefix=PREFIX, files=len(expected),
                  full_readback_verified=False,
                  full_readback_status='Not performed: user explicitly requested upload without full readback.',
                  metadata_size_and_hashes_verified=True,
                  remote_manifest_download_and_sha256_verified=True,
                  lfs_hash_files=lfs_count, git_blob_hash_files=blob_count,
                  bytes=sum((ARCHIVE / name).stat().st_size for name in expected),
                  bundle_manifest_sha256=receipt['bundle_manifest_sha256'],
                  local_archive=str(ARCHIVE),
                  checkpoint_path=str(Path(store.read_json(OUT / 'formal_parameters.json')['run_dir']) / str(STEP)),
                  url=f'https://huggingface.co/{REPO}/tree/{oid}/{PREFIX}')
    store.write_json(OUT / f'hf_archive_verification_{STEP}.json', result)
    status('completed', **result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        chain = []
        current = error
        for _ in range(6):
            if current is None:
                break
            message = str(current).lower()
            categories = [key for key in ['timeout','connection reset','proxy','broken pipe','502','503','429','403','401'] if key in message]
            chain.append(dict(error_type=type(current).__name__, categories=categories,
                              http_status=getattr(getattr(current,'response',None),'status_code',None)))
            current = current.__cause__ or current.__context__
        status('failed', error_type=type(error).__name__, error_chain=chain)
        raise SystemExit(1) from None
