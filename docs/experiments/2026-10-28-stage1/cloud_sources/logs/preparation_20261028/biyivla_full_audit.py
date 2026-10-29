"""Read and verify one pinned snapshot; write receipts only in the audit directory."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import av
import numpy as np
import pyarrow.parquet as pq

BASE = Path('/root/autodl-tmp/biyivla')
OUT = BASE / 'logs/preparation_20261028'
REV = '5ffc4e58e4307fab8a02a9f34aa4d975008ffa5d'
ROOT = BASE / 'datasets/frozen' / REV


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    start = time.monotonic()
    remote = json.loads((OUT / 'hf_remote_file_manifest.json').read_text())
    entries = {f['rfilename']: f for f in remote['files']}
    inventory = {}
    failures = []
    for name, entry in entries.items():
        p = ROOT / name
        if not p.is_file() or p.stat().st_size != entry['size']:
            failures.append({'file': name, 'reason': 'missing_or_size_mismatch'})
            continue
        checksum = sha(p)
        if entry.get('lfs'):
            expected = entry['lfs']['sha256']
            actual = checksum
        else:
            data = p.read_bytes()
            expected = entry['blob_id']
            actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if actual != expected:
            failures.append({'file': name, 'reason': 'hub_identity_mismatch'})
        inventory[name] = {'sha256': checksum, 'bytes': p.stat().st_size}
    write('file_integrity.json', {'revision': REV, 'files': inventory, 'failures': failures})
    if failures:
        print('FILE_INTEGRITY_FAIL', failures, flush=True)
        return 2
    write('numeric_source_inventory.json', {'repo_id': remote['repo_id'], 'revision': REV, 'files': entries})
    cache = ROOT / '.cache'
    if cache.exists():
        cache.rename(OUT / 'download_metadata_cache')
    lock = ROOT / 'dataset_lock.json'
    locked = {'repo_id': remote['repo_id'], 'revision': REV, 'files': inventory}
    if lock.exists():
        assert json.loads(lock.read_text()) == locked, 'never overwrite a different freeze lock'
    else:
        lock.write_text(json.dumps(locked, indent=2, sort_keys=True) + '\n')
    print('FILE_INTEGRITY_PASS', len(inventory), sum(x['bytes'] for x in inventory.values()), sha(lock), flush=True)
    info = json.loads((ROOT / 'meta/info.json').read_text())
    episodes = [json.loads(x) for x in (ROOT / 'meta/episodes.jsonl').read_text().splitlines()]
    declared = [json.loads(x) for x in (ROOT / 'meta/episodes_stats.jsonl').read_text().splitlines()]
    tasks = {x['task_index']: x['task'] for x in map(json.loads, (ROOT / 'meta/tasks.jsonl').read_text().splitlines())}
    assert len(episodes) == len(declared) == info['total_episodes'] == 117
    assert sum(x['length'] for x in episodes) == info['total_frames'] == 77975
    assert info['splits'] == {'train': '0:117'}
    assert all(isinstance(t, str) and t.strip() for t in tasks.values())
    states, actions, weights = [], [], []
    rows, media = [], []
    offset = 0
    max_timestamp_error = 0.
    for i, episode in enumerate(episodes):
        assert episode['episode_index'] == declared[i]['episode_index'] == i
        fields = {'episode_index': i, 'episode_chunk': i // info['chunks_size']}
        t = pq.read_table(ROOT / info['data_path'].format(**fields))
        n = episode['length']
        assert t.num_rows == n
        expected_bookkeeping = {'index': np.arange(offset, offset+n), 'frame_index': np.arange(n), 'episode_index': np.full(n, i)}
        for k, expected in expected_bookkeeping.items():
            actual = np.asarray(t[k].to_pylist()).reshape(-1)
            assert np.array_equal(actual, expected), (i, k)
            for stat, value in [('min', expected.min()), ('max', expected.max()), ('mean', expected.mean())]:
                assert np.allclose(declared[i]['stats'][k][stat], value, atol=1e-5, rtol=1e-6), (i, k, stat)
        ids = set(t['task_index'].to_pylist())
        assert ids <= tasks.keys() and set(episode['tasks']) == {tasks[x] for x in ids}
        timestamps = np.asarray(t['timestamp'].to_pylist()).reshape(-1)
        err = float(np.max(np.abs(timestamps - np.arange(n) / info['fps'])))
        assert np.isfinite(timestamps).all() and err < 2e-5 and np.all(np.diff(timestamps) > 0), (i, err)
        max_timestamp_error = max(max_timestamp_error, err)
        physical = {}
        for k in ['observation.state', 'action']:
            v = np.asarray(t[k].to_pylist(), dtype=np.float32)
            assert v.shape == (n, 17) and np.isfinite(v).all(), (i, k)
            assert all(np.allclose(declared[i]['stats'][k][s], f(v, axis=0), atol=1e-5, rtol=1e-5) for s,f in [('min',np.min),('max',np.max)]), (i,k,'declared_min_max')
            physical[k] = v
        states.append(physical['observation.state'])
        actions.append(physical['action'])
        weight = np.zeros(n, dtype=np.int64)
        for delta in range(50):
            np.add.at(weight, np.minimum(np.arange(n) + delta, n-1), 1)
        assert weight.sum() == n * 50
        weights.append(weight)
        for cam in ['observation.images.head', 'observation.images.wrist']:
            media.append((i, cam, n, info['video_path'].format(**fields, video_key=cam)))
        rows.append({'episode': i, 'frames': n, 'timestamp_max_error_s': err, 'task_indices': sorted(ids)})
        offset += n
    limits = json.loads((OUT / 'joint_limits_source.json').read_text())
    lo, hi = np.array(limits['lower']), np.array(limits['upper'])
    numeric = {}
    for k,v,w in [('state', np.concatenate(states), None), ('actions_raw', np.concatenate(actions), None), ('actions_training_chunks', np.concatenate(actions), np.concatenate(weights))]:
        x = v.astype(np.float64)
        mean = np.average(x, axis=0, weights=w)
        std = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
        if w is None:
            quantiles = np.quantile(x, [0.01,0.99], axis=0)
        else:
            qs=[]
            for j in range(17):
                order=np.argsort(x[:,j]);cum=np.cumsum(w[order]);n=int(cum[-1]);vals=[]
                for q in [0.01,0.99]:
                    pos=q*(n-1);a=int(np.floor(pos));b=int(np.ceil(pos))
                    left=x[order[np.searchsorted(cum,a,side='right')],j]
                    right=x[order[np.searchsorted(cum,b,side='right')],j]
                    vals.append(float(left+(right-left)*(pos-a)))
                qs.append(vals)
            quantiles=np.array(qs).T
        bad=(x<lo-limits['tolerance_rad']) | (x>hi+limits['tolerance_rad'])
        numeric[k]={'count':len(x) if w is None else int(w.sum()),'mean':mean.tolist(),'std':std.tolist(),'min':x.min(0).tolist(),'max':x.max(0).tolist(),'q01_exact':quantiles[0].tolist(),'q99_exact':quantiles[1].tolist(),'range_violation_counts':bad.sum(0).tolist(),'zero_variance_dimensions':np.flatnonzero(std < 1e-8).tolist(),'near_zero_std_dimensions':np.flatnonzero(std < 1e-4).tolist()}
    write('numeric_audit.json', {'revision': REV, 'episodes':rows, 'tasks':tasks,'max_timestamp_error_s':max_timestamp_error,'joint_limits':limits,'statistics':numeric})
    print('NUMERIC_AUDIT_PASS', 'frames', offset, 'max_timestamp_error',max_timestamp_error, 'range_counts', {k:v['range_violation_counts'] for k,v in numeric.items()}, flush=True)
    def video_check(item):
        i,cam,n,name = item
        try:
            with av.open(str(ROOT/name)) as container:
                stream=container.streams.video[0]
                stream.codec_context.thread_count=1
                s={'codec':stream.codec_context.name,'width':stream.width,'height':stream.height,'avg_frame_rate':str(stream.average_rate)}
                pts=np.array([float(frame.pts*frame.time_base) for frame in container.decode(stream)])
        except Exception as e:
            return {'episode':i,'camera':cam,'pass':False,'reason':'decode_error','error_type':type(e).__name__}
        error=float(np.max(np.abs(pts-np.arange(len(pts))/30))) if len(pts) else 1e9
        ok=len(pts)==n and s['width']==640 and s['height']==480 and error<2e-5 and s['avg_frame_rate'] in ('30','30/1')
        return {'episode':i,'camera':cam,'pass':ok,'decoded_frames':len(pts),'expected_frames':n,'timestamp_max_error_s':error,'stream':s}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for row in pool.map(video_check,media):
            results.append(row)
            if len(results)%30==0:print('VIDEOS_CHECKED',len(results),flush=True)
    write('video_audit.json',{'videos':results,'passed':sum(x['pass'] for x in results),'total':len(results),'method':'PyAV decodes every frame; compare all presentation timestamps to frame_index/30','seconds':time.monotonic()-start})
    print('VIDEO_AUDIT',sum(x['pass'] for x in results),len(results),flush=True)
    return 0 if all(x['pass'] for x in results) else 3


if __name__=='__main__':
    raise SystemExit(main())
