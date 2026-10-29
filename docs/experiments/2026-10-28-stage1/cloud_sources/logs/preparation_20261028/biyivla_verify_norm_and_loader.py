"""Independently check train-weighted moments and actual LeRobot boundaries."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openpi import transforms
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import data_loader

BASE=Path('/root/autodl-tmp/biyivla')
OUT=BASE/'logs/preparation_20261028'


def main():
    config=training_config.get_config('pi05_biyi_stage1')
    data=config.data.create(config.assets_dirs,config.model)
    path=config.assets_dirs/data.asset_id/'norm_stats.json'
    stats=normalize.load(path.parent)
    numeric=json.loads((OUT/'numeric_audit.json').read_text())
    report={'norm_stats_path':str(path),'norm_stats_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'moments':{}}
    for key,reference in [('state','state'),('actions','actions_training_chunks')]:
        expected=numeric['statistics'][reference];actual=stats[key]
        for field in ['mean','std','q01','q99']:
            value=np.asarray(getattr(actual,field));assert value.shape==(17,) and np.isfinite(value).all()
        errors={field:np.abs(np.asarray(getattr(actual,field))-np.asarray(expected[field])).tolist() for field in ['mean','std']}
        assert max(errors['mean'])<5e-5,(key,'mean',errors)
        assert max(errors['std'])<3e-4,(key,'std',errors)
        qerrors={field:np.abs(np.asarray(getattr(actual,field))-np.asarray(expected[field+'_exact'])).tolist() for field in ['q01','q99']}
        span=np.array(expected['max'])-np.array(expected['min'])
        for field,errors_q in qerrors.items():
            assert np.all(np.asarray(errors_q)<=np.maximum(0.005,span*0.01)),(key,field,qerrors)
        report['moments'][key]={'true_vector_count':expected['count'],'mean_absolute_errors':errors['mean'],'std_absolute_errors':errors['std'],'stock_histogram_quantile_absolute_errors':qerrors,'zero_variance_dimensions':expected['zero_variance_dimensions']}
    values=np.zeros((2,50,17),dtype=np.float32)
    normalized=transforms.Normalize({'actions':stats['actions']},use_quantiles=data.use_quantile_norm)({'actions':values.copy()})['actions']
    restored=transforms.Unnormalize({'actions':stats['actions']},use_quantiles=data.use_quantile_norm)({'actions':normalized})['actions']
    assert np.isfinite(normalized).all() and np.allclose(restored[...,9],0,atol=1e-9)
    assert stats['actions'].std[9]==0 and stats['actions'].q01[9]==stats['actions'].q99[9]==0
    report['expected_constant_action9']={'physical':0,'normalized':float(normalized[0,0,9]),'restored':float(restored[0,0,9]),'project_constraint_confirmed_by_user':True}
    raw=data_loader.create_torch_dataset(data,config.model.action_horizon,config.model)
    assert len(raw)==77975
    processed=data_loader.transform_dataset(raw,data)
    # Unwrap the real selected dataset and prompt transform, preserving LeRobot.
    base=raw
    while not hasattr(base,'_get_query_indices'):
        if hasattr(base,'dataset'):base=base.dataset
        else:base=base._dataset
    episodes=[json.loads(x) for x in (Path(data.local_root)/'meta/episodes.jsonl').read_text().splitlines()]
    offset=0;checked=0
    for i,ep in enumerate(episodes):
        n=ep['length']
        for within in sorted({0,max(0,n-50),n-1}):
            index=offset+within
            indices,padding=base._get_query_indices(index,i)
            expected=np.minimum(np.arange(index,index+50),offset+n-1)
            assert np.array_equal(np.asarray(indices['action']),expected),(i,within)
            checked+=1
        offset+=n
    samples=[]
    for index in [0,len(raw)//2,len(raw)-1]:
        source=raw[index];sample=processed[index]
        assert np.asarray(source['action']).shape==(50,17)
        assert sample['state'].shape==(32,) and sample['actions'].shape==(50,32)
        assert np.isfinite(sample['state']).all() and np.isfinite(sample['actions']).all()
        assert np.all(sample['state'][17:]==0) and np.all(sample['actions'][:,17:]==0)
        assert np.allclose(sample['actions'][:,9],-1,atol=1e-6)
        samples.append({'index':index,'state_shape':list(sample['state'].shape),'action_shape':list(sample['actions'].shape),'padding_is_zero':True,'action9_normalized':float(sample['actions'][0,9])})
    report.update(passed=True,usable_start_samples=len(raw),boundary_queries_checked=checked,episode_boundary_rule='repeat final action; do not remove last49 frames',sampler_drop_last_per_pass={str(b):len(raw)%b for b in [1,2,4]},real_model_samples=samples)
    (OUT/'norm_and_loader_verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('NORM_AND_LOADER_PASS',report['norm_stats_sha256'],'start_samples',len(raw),'boundary_queries',checked,flush=True)


if __name__=='__main__':
    main()
