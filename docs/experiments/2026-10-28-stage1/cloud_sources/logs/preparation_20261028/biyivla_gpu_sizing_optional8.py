"""Measure short runs of the existing stock launcher; never publish or run FFT."""
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

import psutil

BASE=Path('/root/autodl-tmp/biyivla')
OUT=BASE/'logs/preparation_20261028'
REPO=BASE/'openpi'
RESULTS=[]


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def monitor(command,env,label):
    path=OUT/f'{label}.log'
    cache={}
    rows=[]
    start=time.monotonic()
    with path.open('w') as log:
        proc=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        root=psutil.Process(proc.pid)
        while proc.poll() is None:
            row={'elapsed_s':time.monotonic()-start}
            try:
                children=[root,*root.children(recursive=True)]
                rss=0;cpu=0.
                for child in children:
                    try:
                        p=cache.setdefault(child.pid,child)
                        rss+=p.memory_info().rss;cpu+=p.cpu_percent()
                    except psutil.Error:pass
                row.update(rss_sum_bytes=rss,cpu_percent_sum=cpu,cgroup_memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()))
            except psutil.Error:pass
            query=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            try:
                util,mem,temp=map(float,query.stdout.strip().split(','));row.update(gpu_util_percent=util,gpu_memory_used_mib=mem,gpu_temperature_c=temp)
            except ValueError:pass
            rows.append(row)
            time.sleep(0.5)
        code=proc.wait()
    write(f'{label}_system_samples.json',rows)
    text=path.read_text(errors='replace')
    steps=[{'step':int(s),'seconds':float(t)} for s,t in re.findall(r'step=(\d+) loss=.*? time=([0-9.]+)s',text)]
    stable=[x['seconds'] for x in steps if x['step']>=1]
    result={'returncode':code,'wall_seconds':time.monotonic()-start,'log':str(path),'logged_step_times':steps,'stable_median_seconds':statistics.median(stable) if stable else None,'cpu_peak_rss_sum_bytes':max((r.get('rss_sum_bytes',0) for r in rows),default=0),'gpu_util_peak_percent':max((r.get('gpu_util_percent',0) for r in rows),default=0),'gpu_used_peak_mib':max((r.get('gpu_memory_used_mib',0) for r in rows),default=0),'gpu_temperature_peak_c':max((r.get('gpu_temperature_c',0) for r in rows),default=0),'oom':'out of memory' in text.lower(),'forward_backward_optimizer_complete':code==0 and 'training_complete' in text,'gradient_checkpointing_enabled':'Enabled gradient checkpointing' in text}
    return result


def main():
    assert json.loads((OUT/'video_audit.json').read_text())['passed']==234
    assert not json.loads((OUT/'file_integrity.json').read_text())['failures']
    stats=Path(os.environ['BIYI_ASSETS_BASE_DIR'])/'pi05_biyi_stage1/biyi_stage1/norm_stats.json'
    assert stats.is_file()
    python=os.environ['BIYI_PYTHON']
    batches=tuple(int(x) for x in os.environ.get('BIYI_SIZING_BATCHES','1,2,4').split(','))
    assert all(x in (1,2,4,8) for x in batches)
    if batches!=(1,2,4) and (OUT/'gpu_sizing.json').exists():
        RESULTS.extend(json.loads((OUT/'gpu_sizing.json').read_text()))
    for batch in batches:
        env={**os.environ,'BIYI_EXP_NAME':f'gpu_memory_smoke_20261028_b{batch}','BIYI_BATCH_SIZE':str(batch),'BIYI_NUM_GPUS':'1','BIYI_NUM_WORKERS':'0','BIYI_NUM_TRAIN_STEPS':'5','BIYI_SAVE_INTERVAL':'5','BIYI_LOG_INTERVAL':'1','BIYI_MEMORY_SMOKE':'1','WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'}
        env.pop('BIYI_CHECKPOINT_IO',None)
        run=Path(env['BIYI_CHECKPOINT_BASE_DIR'])/'pi05_biyi_stage1'/env['BIYI_EXP_NAME']
        assert not run.exists(),f'never overwrite smoke directory {run}'
        assert shutil.disk_usage(run.parent.parent.parent).free>55_000_000_000,'insufficient save/resume headroom'
        print('GPU_SMOKE_START',batch,flush=True)
        initial=monitor(['bash','scripts/biyi/train_pi05_stage1.sh','gpu-smoke'],env,f'b{batch}_initial')
        result={'batch_size':batch,'initial':initial,'run_dir':str(run),'schedule':{'warmup':1,'decay':5,'peak_lr':2.5e-5,'end_lr':2.5e-6},'steps_initial':5,'steps_resume_target':6}
        RESULTS.append(result)
        if initial['returncode']!=0:
            result['pass']=False;write('gpu_sizing.json',RESULTS)
            print('GPU_SMOKE_FAIL',batch,'oom',initial['oom'],flush=True)
            if not initial['oom']:return 2
            continue
        receipt=json.loads((run/'memory_smoke_rank0.json').read_text())
        result['initial_memory']=receipt
        write(f'b{batch}_initial_memory.json',receipt)
        result['checkpoint_5_files']={p.relative_to(run/'5').as_posix():p.stat().st_size for p in (run/'5').rglob('*') if p.is_file()}
        result['checkpoint_5_bytes']=sum(result['checkpoint_5_files'].values())
        print('GPU_SMOKE_INITIAL_PASS',batch,'peak',receipt['peak_allocated'],'checkpoint_bytes',result['checkpoint_5_bytes'],flush=True)
        command=[python,'scripts/train_pytorch.py','pi05_biyi_stage1','--exp-name',env['BIYI_EXP_NAME'],'--batch-size',str(batch),'--num-workers','0','--num-train-steps','6','--lr-schedule.warmup-steps','1','--lr-schedule.decay-steps','5','--lr-schedule.peak-lr','2.5e-5','--lr-schedule.decay-lr','2.5e-6','--save-interval','5','--assets-base-dir',env['BIYI_ASSETS_BASE_DIR'],'--checkpoint-base-dir',env['BIYI_CHECKPOINT_BASE_DIR'],'--log-interval','1','--no-wandb-enabled','--resume']
        resume=monitor(command,env,f'b{batch}_resume')
        result['resume']=resume
        result['resume_log_confirmed']='Successfully loaded all checkpoint components from step 5' in (OUT/f'b{batch}_resume.log').read_text(errors='replace')
        if resume['returncode']==0:
            result['resume_memory']=json.loads((run/'memory_smoke_rank0.json').read_text())
            result['checkpoint_6_bytes']=sum(p.stat().st_size for p in (run/'6').rglob('*') if p.is_file())
            import torch
            metadata=torch.load(run/'6/metadata.pt',map_location='cpu',weights_only=False)
            state=torch.load(run/'6/optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
            steps=sorted({int(v['step'].item()) for v in state['state'].values()})
            norm=run/'6/assets/biyi_stage1/norm_stats.json'
            norm_equal=hashlib.sha256(norm.read_bytes()).digest()==hashlib.sha256(stats.read_bytes()).digest()
            result['resume_state_verification']={'metadata_global_step':metadata['global_step'],'optimizer_state_count':len(state['state']),'optimizer_step_values':steps,'norm_stats_identical':norm_equal}
            assert metadata['global_step']==6 and steps==[6] and norm_equal
            del state,metadata
            gc.collect()
        result['pass']=initial['forward_backward_optimizer_complete'] and resume['forward_backward_optimizer_complete'] and result['resume_log_confirmed']
        write('gpu_sizing.json',RESULTS)
        print('GPU_SMOKE_RESULT',batch,result['pass'],flush=True)
        if not result['pass']:return 3
        # Both directories were created by this sizing job in its unique run.
        # Retain the complete restored checkpoint; only the verified temporary
        # pre-resume save is removed, with its sizes and successful load logged.
        assert run.resolve().is_relative_to(Path(env['BIYI_CHECKPOINT_BASE_DIR']).resolve())
        assert not (run/'5').is_symlink() and (run/'6').is_dir()
        shutil.rmtree(run/'5')
        result['retention']='keep complete step6; remove own temporary step5 only after verified resume'
        write('gpu_sizing.json',RESULTS)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
