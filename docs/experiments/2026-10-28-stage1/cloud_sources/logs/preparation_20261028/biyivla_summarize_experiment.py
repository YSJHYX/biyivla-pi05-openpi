"""Summarize completed local training receipts without inferring task success."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np

BASE=Path('/root/autodl-tmp/biyivla')
OUT=BASE/'logs/preparation_20261028'


def main():
    params=json.loads((OUT/'formal_parameters.json').read_text())
    completion=json.loads((OUT/'formal_completion.json').read_text())
    assert completion['returncode']==0
    run=Path(params['run_dir'])
    rows=[json.loads(x) for x in (run/'training_metrics.jsonl').read_text().splitlines()]
    assert [r['global_step'] for r in rows]==list(range(1,params['num_train_steps']+1))
    assert rows[-1]['samples_seen']==params['num_train_steps']*params['global_batch_size']
    checkpoints=[json.loads(x) for x in (run/'checkpoint_metrics.jsonl').read_text().splitlines()]
    assert [x['checkpoint_step'] for x in checkpoints]==params['expected_save_steps']
    assert all(x['checkpoint/save_complete']==x['checkpoint/backup_verified']==1 for x in checkpoints)
    def statistics(key):
        values=np.array([x[key] for x in rows],dtype=np.float64)
        assert np.isfinite(values).all()
        window=min(200,len(values));post=np.array([r[key] for r in rows if r['global_step']>params['warmup_steps']])
        return {'count':len(values),'finite':True,'min':float(values.min()),'max':float(values.max()),'mean':float(values.mean()),'std':float(values.std()),'p05':float(np.quantile(values,.05)),'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'first_200_mean':float(values[:window].mean()),'last_200_mean':float(values[-window:].mean()),'first_200_after_warmup_mean':float(post[:window].mean())}
    systems=[json.loads(x) for x in (OUT/'formal_system_samples.jsonl').read_text().splitlines()]
    sample=lambda key:[x[key] for x in systems if key in x]
    peak_alloc=max(x['gpu/peak_allocated_bytes'] for x in rows)
    peak_reserved=max(x['gpu/peak_reserved_bytes'] for x in rows)
    peak_main_rss=max(x['system/cpu_peak_rss_bytes'] for x in rows)
    report={'experiment':'pi05_biyi_117ep_fft_v2','parameters':params,'completion':completion,'loss':statistics('loss'),'grad_norm_before_clip':statistics('grad_norm'),'grad_norm_over_clip_threshold_fraction':sum(r['grad_norm']>params['gradient_clip'] for r in rows)/len(rows),'final_global_step':rows[-1]['global_step'],'final_samples_seen':rows[-1]['samples_seen'],'final_equivalent_epochs':rows[-1]['equivalent_epochs'],'peak_allocated_bytes':peak_alloc,'peak_reserved_bytes':peak_reserved,'peak_main_process_rss_bytes':peak_main_rss,'peak_process_tree_rss_bytes':completion['peak_process_tree_rss_bytes'],'gpu_smi_peak_used_mib':max(sample('gpu_used_mib')),'gpu_util_mean_sampled_percent':float(np.mean(sample('gpu_util_percent'))),'gpu_util_peak_sampled_percent':max(sample('gpu_util_percent')),'gpu_peak_temperature_c':max(sample('gpu_temperature_c')),'checkpoints':checkpoints,'disk_after_training':{str(p):dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in [BASE,Path('/autodl-fs/data')]},'limitations':['All117episodes are training-only; no independent validation set or real-robot task-success test.','Stock flow-matching loss averages over32model action dimensions including padding; it is not physical robot-position RMSE.','Gradient norm is reported before clipping.','Equivalent epochs count sampled starting frames; horizon50chunks overlap and repeat final actions at episode ends.','This records a reproducible configuration and pinned artifacts; bitwise resume is not guaranteed because stock checkpoints do not preserve all RNG/data-iterator state.']}
    (OUT/'training_summary.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (OUT/'training_numeric_history.json').open('w') as f:json.dump(rows,f,allow_nan=False)
    with (OUT/'training_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    loss=report['loss'];grad=report['grad_norm_before_clip']
    text=['# BiyiVLA Stage-1 FFT experiment report','','- Experiment: `pi05_biyi_117ep_fft_v2`','- Status: completed all '+str(report['final_global_step'])+' optimizer steps.','- Data:117training episodes /77,975eligible starting frames; dataset revision `'+params['dataset_revision']+'`.','- Batch:'+str(params['global_batch_size'])+'; exposure:'+str(round(report['final_equivalent_epochs'],6))+' nominal epochs.','- Wall time:'+str(round(completion['wall_seconds']/3600,3))+'h.','','| Metric | Result |','|---|---|',f"| Loss first200 / last200 mean | {loss['first_200_mean']:.6g} / {loss['last_200_mean']:.6g} |",f"| Loss range / standard deviation | {loss['min']:.6g}..{loss['max']:.6g} / {loss['std']:.6g} |",f"| Pre-clip gradient norm median /p95 /max | {grad['median']:.6g} / {grad['p95']:.6g} / {grad['max']:.6g} |",f"| Gradient norm>1 fraction | {100*report['grad_norm_over_clip_threshold_fraction']:.2f}% |",f"| GPU peak allocated / reserved | {peak_alloc/2**30:.3f} / {peak_reserved/2**30:.3f}GiB |",f"| CPU peak main / process tree RSS | {peak_main_rss/2**30:.3f} / {completion['peak_process_tree_rss_bytes']/2**30:.3f}GiB |",f"| GPU sampled mean / peak utilization | {report['gpu_util_mean_sampled_percent']:.2f}% / {report['gpu_util_peak_sampled_percent']:.1f}% |",'','Three complete saves and independent checksum-verified backups: '+', '.join(map(str,params['expected_save_steps']))+'.','','Training-loss changes alone do not establish held-out generalization or robot task success. Offline load/finite/shape acceptance is a separate technical check.']
    (OUT/'TRAINING_REPORT.md').write_text('\n'.join(text)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        step=np.array([x['global_step'] for x in rows]);window=101
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for axis,key,title in [(axes[0],'loss','Flow-matching loss'),(axes[1],'grad_norm','Gradient norm before clipping')]:
            values=np.array([x[key] for x in rows]);axis.plot(step,values,color='#9ca3af',alpha=.3,linewidth=.5,label='per step')
            smooth=np.convolve(values,np.ones(window)/window,mode='valid');axis.plot(step[window-1:],smooth,color='#2563eb',linewidth=1.2,label='101-step mean')
            axis.axvline(params['warmup_steps'],linestyle='--',color='#f59e0b',linewidth=.8,label='warmup end');axis.set_ylabel(title);axis.grid(alpha=.2);axis.legend(loc='upper right')
        axes[1].axhline(params['gradient_clip'],color='#dc2626',linestyle=':',linewidth=.9,label='clip threshold')
        axes[1].set_xlabel('Completed optimizer steps');fig.suptitle('BiyiVLA Stage-1 FFT —117training episodes');fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=180);plt.close(fig)
    except ImportError:
        (OUT/'plot_status.txt').write_text('Matplotlib unavailable in frozen runtime; complete numeric history and CSV are provided.\n')
    print('TRAINING_SUMMARY_PASS','steps',len(rows),'checkpoints',len(checkpoints),flush=True)


if __name__=='__main__':
    main()
