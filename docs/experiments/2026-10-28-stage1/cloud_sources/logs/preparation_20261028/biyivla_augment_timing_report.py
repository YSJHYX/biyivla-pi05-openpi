"""Separate optimizer-loop time from W&B final synchronization waiting."""
import datetime as dt
import json
from pathlib import Path
import re

OUT=Path('/root/autodl-tmp/biyivla/logs/preparation_20261028')
summary=json.loads((OUT/'training_summary.json').read_text())
params=summary['parameters'];tz=dt.datetime.fromisoformat(params['selected_at']).tzinfo
day=dt.datetime.fromisoformat(params['selected_at']).date();previous=None;records=[]
text=(OUT/'formal_training.log').read_text(errors='replace')
for match in re.finditer(r'(\d{2}:\d{2}:\d{2}\.\d{3}) \[I\] ([^\r\n]*)',text):
    clock=dt.datetime.strptime(match[1],'%H:%M:%S.%f').time()
    stamp=dt.datetime.combine(day,clock,tzinfo=tz)
    if previous is not None and (previous-stamp).total_seconds()>12*3600:
        day+=dt.timedelta(days=1);stamp=dt.datetime.combine(day,clock,tzinfo=tz)
    records.append((stamp,match[2]));previous=stamp
first=next(stamp for stamp,message in records if re.match(r'step=0 loss=',message))
last=next(stamp for stamp,message in reversed(records) if re.match(r'step='+str(params['num_train_steps']-1)+r' loss=',message))
complete=next(stamp for stamp,message in reversed(records) if '(training_complete)' in message)
saved=next(stamp for stamp,message in reversed(records) if 'Saved checkpoint at step '+str(params['num_train_steps'])+' ' in message)
history=[json.loads(x) for x in (Path(params['run_dir'])/'training_metrics.jsonl').read_text().splitlines()]
loop_start=first-dt.timedelta(seconds=history[0]['time_per_step'])
finished=dt.datetime.fromisoformat(summary['completion']['finished_at'])
timing={'optimizer_loop_started_at_approx':loop_start.isoformat(),'last_optimizer_step_logged_at':last.isoformat(),'final_save_and_backup_verified_at':saved.isoformat(),'optimizer_loop_completed_at':complete.isoformat(),'process_finished_after_wandb_sync_at':finished.isoformat(),'optimizer_loop_including_three_saves_seconds':(complete-loop_start).total_seconds(),'wandb_finish_wait_seconds':(finished-complete).total_seconds(),'formal_entry_wall_seconds':summary['completion']['wall_seconds']}
systems=[json.loads(x) for x in (OUT/'formal_system_samples.jsonl').read_text().splitlines()]
active=[x['gpu_util_percent'] for x in systems if 'gpu_util_percent' in x and loop_start<=dt.datetime.fromisoformat(x['at'])<=complete]
summary['timing']=timing
summary['gpu_util_mean_full_process_sampled_percent']=summary['gpu_util_mean_sampled_percent']
summary['gpu_util_mean_training_loop_sampled_percent']=sum(active)/len(active)
summary['timing_note']='Formal-entry wall time includes a long W&B finish synchronization wait after all optimizer updates and all three verified backups had completed; no additional training steps ran during that wait.'
(OUT/'training_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
suffix=['','## Timing separation','',f"- Optimizer loop including all three saves/backups: {timing['optimizer_loop_including_three_saves_seconds']/3600:.3f}h.",f"- W&B finish synchronization wait: {timing['wandb_finish_wait_seconds']/3600:.3f}h.",f"- Full formal-entry wall time: {timing['formal_entry_wall_seconds']/3600:.3f}h.",'No optimizer steps were added during synchronization waiting.']
with (OUT/'TRAINING_REPORT.md').open('a') as f:f.write('\n'.join(suffix)+'\n')
loss=summary['loss'];grad=summary['grad_norm_before_clip']
cn=['# BiyiVLA Stage-1 FFT 实验简报','',f"已完成 **{summary['final_global_step']:,} steps**，Batch8，117条训练Episode，约1.000013个等效Epoch。",'', '| 项目 | 实测结果 |','|---|---|',f"| Loss：前200 / 后200步均值 | {loss['first_200_mean']:.6f} / {loss['last_200_mean']:.6f} |",f"| Loss：全程范围 / 标准差 | {loss['min']:.6f}–{loss['max']:.6f} / {loss['std']:.6f} |",f"| 裁剪前梯度范数：中位数 / P95 / 最大值 | {grad['median']:.6f} / {grad['p95']:.6f} / {grad['max']:.6f} |",f"| 梯度范数超过1的比例 | {summary['grad_norm_over_clip_threshold_fraction']*100:.2f}% |",f"| GPU峰值allocated / reserved | {summary['peak_allocated_bytes']/2**30:.3f} / {summary['peak_reserved_bytes']/2**30:.3f} GiB |",f"| 进程组RAM峰值 | {summary['peak_process_tree_rss_bytes']/2**30:.3f} GiB |",f"| 训练循环含3次保存备份 | {timing['optimizer_loop_including_three_saves_seconds']/3600:.3f}小时 |",f"| W&B收尾同步等待 | {timing['wandb_finish_wait_seconds']/3600:.3f}小时 |",'', '三次完整保存及独立回读校验均通过：3,249、6,498、9,747步。W&B网络同步等待期间没有继续更新权重。','', '全部117条用于训练；本轮没有独立验证集或真机任务成功率测试。训练loss变化不能作为泛化或部署成功的证明。','', '参数、环境、固定源码/数据/base/norm哈希及复现步骤见实验清单；HF归档的实际commit和回验结果将在归档完成后追加。']
(OUT/'FINAL_REPORT_CN.md').write_text('\n'.join(cn)+'\n')
print('TIMING_AND_REPORT_READY',json.dumps(timing),flush=True)
