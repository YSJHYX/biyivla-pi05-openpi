"""Run offline checks on this experiment's completed checkpoint only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open
import torch

from openpi.policies import policy_config
from openpi.training import config as training_config
from openpi.training import data_loader
from scripts.biyi import checkpoint_store as store

BASE=Path('/root/autodl-tmp/biyivla')
OUT=BASE/'logs/preparation_20261028'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=[3249,6498],required=True);args=parser.parse_args()
    completion=store.read_json(OUT/'formal_completion.json')
    assert completion['returncode']==0 and completion['stage']=='completed'
    params=store.read_json(OUT/'formal_parameters.json')
    run=Path(params['run_dir']);step=args.step;checkpoint=run/str(step)
    record=store.read_json(run/'run_manifest.json')
    assert record['source_commit']==store.git('rev-parse','HEAD')==params['source_commit']
    assert not store.git('status','--porcelain')
    source=store.dataset_identity(Path(record['dataset'].get('local_root',str(BASE/'datasets/frozen'/params['dataset_revision']))),params['dataset_revision'],117)
    assert source['lock_sha256']==record['dataset']['lock_sha256']
    assert store.digest(checkpoint/'assets/biyi_stage1/norm_stats.json')==params['norm_stats_sha256']
    metadata=torch.load(checkpoint/'metadata.pt',map_location='cpu',weights_only=False)
    optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
    optimizer_steps=sorted({int(value['step'].item()) for value in optimizer['state'].values()})
    assert metadata['global_step']==step and optimizer_steps==[step]
    optimizer_count=len(optimizer['state']);del optimizer,metadata
    tensors=0;elements=0;dtypes={}
    with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as file:
        for name in file.keys():
            value=file.get_tensor(name)
            assert torch.isfinite(value).all(),name
            tensors+=1;elements+=value.numel();dtypes[str(value.dtype)]=dtypes.get(str(value.dtype),0)+value.numel()
            del value
    torch.manual_seed(42)
    config=training_config.get_config('pi05_biyi_stage1')
    data=config.data.create(config.assets_dirs,config.model)
    raw=data_loader.create_torch_dataset(data,50,config.model)
    policy=policy_config.create_trained_policy(config,checkpoint,pytorch_device='cuda:0',sample_kwargs={'num_steps':10})
    limits=store.read_json(OUT/'joint_limits_source.json');lo=np.array(limits['lower']);hi=np.array(limits['upper'])
    results=[]
    for ordinal,index in enumerate([0,len(raw)//2,len(raw)-1]):
        source=raw[index]
        obs={'state':np.asarray(source['observation.state']),'images':{'head':np.asarray(source['observation.images.head']),'wrist':np.asarray(source['observation.images.wrist'])},'prompt':str(source['prompt'])}
        noise=np.random.default_rng(42+ordinal).normal(size=(50,32)).astype(np.float32)
        started=time.monotonic();output=policy.infer(obs,noise=noise);torch.cuda.synchronize()
        actions=np.asarray(output['actions']);assert actions.shape==(50,17) and np.isfinite(actions).all()
        violations=((actions<lo-1e-6)|(actions>hi+1e-6)).sum(0)
        results.append({'start_index':index,'noise_seed':42+ordinal,'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),'action_shape':list(actions.shape),'finite':True,'wall_ms':1000*(time.monotonic()-started),'min':actions.min(0).tolist(),'max':actions.max(0).tolist(),'command_range_violation_counts':violations.tolist(),'action9_max_absolute':float(np.max(np.abs(actions[:,9]))),'actions':actions.tolist()})
    report={'source_commit':params['source_commit'],'dataset_revision':params['dataset_revision'],'checkpoint_step':step,'checkpoint_path':str(checkpoint),'norm_stats_sha256':params['norm_stats_sha256'],'optimizer_state_count':optimizer_count,'optimizer_steps':optimizer_steps,'finite_model_tensor_count':tensors,'model_parameter_elements':elements,'dtype_element_counts':dtypes,'checks':{'offline_policy_load':True,'finite_actions':True,'action_shape_50_17':True},'samples':results,'scope':'Automated offline numerical/shape/provenance checks only. These training observations are not held-out validation or real-robot task-success evidence. No hardware commands were issued.'}
    store.write_json(OUT/f'checkpoint_{step}_offline_inference.json',report)
    print('FINAL_OFFLINE_TECHNICAL_CHECK_PASS','step',step,'optimizer_states',optimizer_count,'model_tensors',tensors,flush=True)


if __name__=='__main__':
    main()
