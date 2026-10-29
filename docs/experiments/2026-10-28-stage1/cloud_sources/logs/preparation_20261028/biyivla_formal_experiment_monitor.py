"""Select from measured candidates and monitor the existing formal entry."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import psutil

BASE=Path('/root/autodl-tmp/biyivla')
OUT=BASE/'logs/preparation_20261028'
REPO=BASE/'openpi'


def write(name,obj):
    path=OUT/name;temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert (OUT/'norm_and_sizing.exit').read_text().strip()=='0'
    assert (OUT/'optional_batch8.exit').read_text().strip()=='0'
    verified=json.loads((OUT/'norm_and_loader_verification.json').read_text())
    assert verified['passed'] and verified['usable_start_samples']==77975
    assert json.loads((OUT/'video_audit.json').read_text())['passed']==234
    assert not json.loads((OUT/'file_integrity.json').read_text())['failures']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    candidates=json.loads((OUT/'gpu_sizing.json').read_text())
    usable=[]
    for row in candidates:
        if not row.get('pass'):continue
        assert row['resume_state_verification']['optimizer_step_values']==[6]
        for key in ['initial_memory','resume_memory']:
            receipt=row[key]
            assert receipt['signature']['source_commit']==commit
            assert receipt['signature']['norm_stats_sha256']==verified['norm_stats_sha256']
        peak=max(row[k]['peak_reserved'] for k in ['initial_memory','resume_memory'])
        if peak<=.85*row['initial_memory']['total_memory']:
            usable.append(row)
    assert usable,'No measured candidate has adequate memory margin'
    chosen=min(usable,key=lambda row:row['initial']['stable_median_seconds']/row['batch_size'])
    batch=chosen['batch_size'];frames=77975;steps=math.ceil(frames/batch);warmup=(steps+5)//10
    config=json.loads((REPO/'scripts/biyi/configs/pi05_biyi_117ep_fft_v1.json').read_text())
    config.update(experiment='pi05_biyi_117ep_fft_v2',batch_size=batch,num_train_steps=steps,save_interval=math.ceil(steps/3),num_workers=0,num_gpus=1,log_interval=1)
    recipe=OUT/'pi05_biyi_117ep_fft_v2.json'
    assert not recipe.exists(),'Never silently replace an existing recipe'
    recipe.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
    env=dict(os.environ)
    env.update(BIYI_FORMAL_RECIPE=str(recipe),BIYI_EXP_NAME=config['experiment'],BIYI_BATCH_SIZE=str(batch),BIYI_NUM_TRAIN_STEPS=str(steps),BIYI_SAVE_INTERVAL=str(config['save_interval']),BIYI_LOG_INTERVAL='1',BIYI_NUM_WORKERS='0',BIYI_NUM_GPUS='1',BIYI_MEMORY_SMOKE_RUN_DIR=chosen['run_dir'],BIYI_STORAGE_MODE='backup',BIYI_BACKUP_ROOT='/autodl-fs/data/biyivla_stage1_backups',WANDB_MODE='online',TOKENIZERS_PARALLELISM='false',NO_PROXY='localhost,127.0.0.1,.cdn.hf.co,.xethub.hf.co,.amazonaws.com',no_proxy='localhost,127.0.0.1,.cdn.hf.co,.xethub.hf.co,.amazonaws.com')
    env.pop('WANDB_DISABLED',None);env.pop('BIYI_RESUME',None)
    run=Path(env['BIYI_CHECKPOINT_BASE_DIR'])/'pi05_biyi_stage1'/config['experiment']
    assert not run.exists(),'Never overwrite a prior formal run'
    stats=Path(env['BIYI_ASSETS_BASE_DIR'])/'pi05_biyi_stage1/biyi_stage1/norm_stats.json'
    assert digest(stats)==verified['norm_stats_sha256']
    full_bytes=chosen['checkpoint_6_bytes']
    assert shutil.disk_usage(BASE).free>4*full_bytes+10*2**30,'Insufficient primary save headroom'
    assert shutil.disk_usage('/autodl-fs/data').free>5*full_bytes+20*2**30,'Insufficient backup/archive headroom'
    exposures={str(row['batch_size']):{str(epoch):{'steps':math.ceil(epoch*frames/row['batch_size']),'compute_hours':math.ceil(epoch*frames/row['batch_size'])*row['initial']['stable_median_seconds']/3600} for epoch in [.5,1,2,3]} for row in usable}
    parameters={'selected_at':now(),'source_commit':commit,'dataset_revision':config['dataset_revision'],'train_episodes':117,'train_frames':frames,'usable_start_samples':frames,'global_batch_size':batch,'num_gpus':1,'num_train_steps':steps,'warmup_steps':warmup,'peak_lr':2.5e-5,'end_lr':2.5e-6,'lr_schedule':'linear warmup then cosine decay','weight_decay':.05,'weight_decay_policy':'trainable matrices; bias and 0D/1D excluded','betas':[.9,.95],'eps':1e-8,'gradient_clip':1.,'precision':'bfloat16, stock parameter dtype policy','action_horizon':50,'model_action_dim':32,'physical_action_dim':17,'seed':42,'num_workers':0,'log_interval':1,'save_interval':config['save_interval'],'expected_save_steps':list(range(config['save_interval'],steps,config['save_interval']))+[steps],'equivalent_epochs':steps*batch/frames,'median_smoke_step_seconds':chosen['initial']['stable_median_seconds'],'estimated_compute_hours':steps*chosen['initial']['stable_median_seconds']/3600,'norm_stats_sha256':verified['norm_stats_sha256'],'base_model_sha256':'dbcbe3ea978ce1d6f791582fa034e7ece8a41feaad2d3c9a5b014c1938610e2a','recipe_path':str(recipe),'run_dir':str(run),'backup_root':env['BIYI_BACKUP_ROOT'],'chosen_smoke_dir':chosen['run_dir'],'retention':'retain all three formal saves locally and verified independent backups; smoke weights excluded from HF','exposure_comparison':exposures,'selection_reason':'fastest measured seconds per sample among full optimizer/save/resume PASS candidates below85% peak reserved memory; one nominal starting-frame exposure selected because all117episodes are training-only and there is no independent validation split'}
    write('formal_parameters.json',parameters)
    command=['bash','scripts/biyi/train_pi05_stage1.sh']
    plan=subprocess.run([*command,'train','--dry-run'],cwd=REPO,env=env,capture_output=True,text=True,check=True)
    parsed=json.loads(plan.stdout);assert '--wandb-enabled' in parsed['train'] and '--overwrite' not in parsed['train']
    assert parsed['environment']['BIYI_MEMORY_SMOKE']=='0'
    write('formal_plan.json',parsed)
    print('FORMAL_PARAMETERS_READY',json.dumps({k:parameters[k] for k in ['global_batch_size','num_train_steps','warmup_steps','save_interval','equivalent_epochs','estimated_compute_hours']}),flush=True)
    with (OUT/'formal_preflight.log').open('w') as log:
        subprocess.run([*command,'preflight'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write('formal_status.json',{'stage':'preflight_passed','updated_at':now(),**{k:parameters[k] for k in ['run_dir','num_train_steps','global_batch_size']}})
    started=time.monotonic();log=(OUT/'formal_training.log').open('w')
    proc=subprocess.Popen([*command,'train'],cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    root=psutil.Process(proc.pid);cache={};latest={};offset=0;peak_rss=0;peak_gpu=0;peak_cpu=0
    metrics=run/'training_metrics.jsonl';samples=(OUT/'formal_system_samples.jsonl').open('w')
    while proc.poll() is None:
        row={'at':now(),'elapsed_seconds':time.monotonic()-started}
        try:
            processes=[root,*root.children(recursive=True)];rss=0;cpu=0.
            for child in processes:
                try:p=cache.setdefault(child.pid,child);rss+=p.memory_info().rss;cpu+=p.cpu_percent()
                except psutil.Error:pass
            row.update(process_tree_rss_bytes=rss,cpu_percent_raw=cpu,cpu_percent_of_18_cores=cpu/18,cgroup_memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()))
            peak_rss=max(peak_rss,rss);peak_cpu=max(peak_cpu,cpu)
        except psutil.Error:pass
        response=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        try:
            util,memory,temp=map(float,response.stdout.strip().split(','));row.update(gpu_util_percent=util,gpu_used_mib=memory,gpu_temperature_c=temp);peak_gpu=max(peak_gpu,memory)
        except ValueError:pass
        if metrics.exists():
            with metrics.open() as stream:
                stream.seek(offset)
                for line in stream:
                    if line.endswith('\n'):
                        latest=json.loads(line);offset+=len(line.encode())
        samples.write(json.dumps(row)+'\n');samples.flush()
        write('formal_status.json',{'stage':'formal_entry_running','pid':proc.pid,'started_at':parameters['selected_at'],'updated_at':now(),'elapsed_seconds':time.monotonic()-started,'latest_metrics':latest,'peak_process_tree_rss_bytes':peak_rss,'peak_gpu_smi_used_mib':peak_gpu,'peak_cpu_percent_raw':peak_cpu,'system_sample':row,'run_dir':str(run),'target_steps':steps})
        time.sleep(5)
    code=proc.wait();log.close();samples.close()
    status={'stage':'completed' if code==0 else 'failed','returncode':code,'finished_at':now(),'wall_seconds':time.monotonic()-started,'run_dir':str(run),'target_steps':steps,'peak_process_tree_rss_bytes':peak_rss,'peak_gpu_smi_used_mib':peak_gpu,'peak_cpu_percent_raw':peak_cpu}
    write('formal_completion.json',status);write('formal_status.json',status)
    print('FORMAL_ENTRY_FINISHED',json.dumps(status),flush=True)
    return code


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        write('formal_controller_failure.json',{'error_type':type(e).__name__,'at':now()})
        print('FORMAL_CONTROLLER_FAILURE',type(e).__name__,flush=True)
        raise SystemExit(1) from None
