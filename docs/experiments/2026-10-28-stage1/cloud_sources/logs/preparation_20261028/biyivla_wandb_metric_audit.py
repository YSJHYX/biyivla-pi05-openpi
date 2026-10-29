"""Read observed online keys and selected numerical provenance only."""
import json
import logging
from pathlib import Path
import wandb

logging.disable(logging.CRITICAL)
out=Path('/root/autodl-tmp/biyivla/logs/preparation_20261028')
report={'run_path':'y8hong-heymandex/BiyiVLA-Stage1/6zpgxsbv','read_only':True}
try:
    run=wandb.Api(timeout=15).run(report['run_path'])
    summary=dict(run.summary)
    events=run.history(stream='events',samples=5,pandas=False)
    report.update(state=run.state,run_url=run.url,summary_metric_keys=sorted(summary.keys()),latest_numeric={key:summary[key] for key in ['loss','grad_norm','global_step','samples_seen','equivalent_epochs','learning_rate','gpu/peak_allocated_bytes','gpu/peak_reserved_bytes','checkpoint_step','checkpoint/backup_verified'] if key in summary})
    report['observed_system_metric_keys']=sorted({key for row in events for key in row})
    repro=dict(run.config).get('reproducibility',{})
    report['observed_provenance']={'source_commit':repro.get('source_commit'),'dataset_revision':repro.get('dataset',{}).get('revision'),'episode_count':repro.get('recipe',{}).get('dataset_episodes'),'train_frames':repro.get('recipe',{}).get('dataset_frames'),'norm_stats_sha256':repro.get('norm_stats_sha256'),'base_model_sha256':repro.get('base_model_sha256')}
except Exception as e:
    report['error_type']=type(e).__name__
(out/'wandb_metric_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
