import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import dataclasses, sys, json, time, hashlib, logging, traceback, subprocess, gc
from pathlib import Path
from collections import Counter, defaultdict
from unittest.mock import patch
import numpy as np
import torch
import transformers
from safetensors import safe_open
import safetensors.torch as st
from openpi.training import config as training_config, data_loader, checkpoints
from openpi.training.biyi_optimizer_groups import stage1_adamw_groups
from openpi.policies import policy_config, biyivla_policy
from openpi import transforms
from transformers.models.siglip import check

ROOT=Path(os.environ['BIYI_ROOT'])
OUT=ROOT/'logs/checkpoint_reload_inference'
CKPT=Path(os.environ['BIYI_CHECKPOINT_BASE_DIR'])/'pi05_biyi_stage1/local_smoke_a800_001/2'
BASE=Path(os.environ['BIYI_BASE_CHECKPOINT_DIR'])/'model.safetensors'
COMMIT='bd34466dbecb8c8928e77b3b028fc59ef6e5ec56'
PROMPT='Pull the inner basket out of the air fryer and place it on the table.'
START=time.perf_counter()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',force=True)
torch.set_num_threads(8)
R={'CHECKPOINT_RELOAD_STATUS':'FAIL','INFERENCE_STATUS':'FAIL','OPTIMIZER_STATE_COVERAGE':'REVIEW','CHECKPOINT_WEIGHT_UPDATE':'REVIEW','BiyiVLA_STAGE1_CHECKPOINT_INFERENCE':'FAIL','exceptions':[],'checkpoint_path':str(CKPT),'observations':[],'pid':os.getpid(),'training_executed':False,'backward_executed':False,'optimizer_step_executed':False}
AUD={}; COMP={}; fingerprints={}

def emit(msg): logging.info(msg)
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
    return h.hexdigest()
def arr(x):
    return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
def finite_tensor(t):
    flat=t.reshape(-1)
    return all(bool(torch.isfinite(flat[i:i+8_000_000]).all()) for i in range(0,flat.numel(),8_000_000))
def clone_obs(obs):
    return {'images':{k:v.copy() for k,v in obs['images'].items()},'state':obs['state'].copy(),'prompt':obs['prompt']}

try:
    R['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert R['git_commit']==COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    assert sys.version_info[:2]==(3,11)
    assert torch.__version__=='2.7.1+cu126'
    assert transformers.__version__=='4.53.2'
    assert check.check_whether_transformers_replace_is_installed_correctly()
    assert torch.cuda.device_count()==1
    R['environment']={'python':sys.version.split()[0],'torch':torch.__version__,'transformers':transformers.__version__,'transformers_patch':'PASS','gpu':torch.cuda.get_device_name(0),'offline':True}
    files=['model.safetensors','optimizer.pt','metadata.pt','assets/biyi_stage1/norm_stats.json']
    for fn in files:
        p=CKPT/fn
        assert p.is_file() and p.stat().st_size>0
        fingerprints[fn]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':digest(p)}
    R['checkpoint_before']=fingerprints
    meta=torch.load(CKPT/'metadata.pt',map_location='cpu',weights_only=False)
    assert meta['global_step']==2
    R['metadata_global_step']=meta['global_step']
    cfg=training_config.get_config('pi05_biyi_stage1')
    cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    R['in_memory_inference_override']={'pytorch_compile_mode':None}
    load_calls=[]; norm_calls=[]
    real_load=st.load_model; real_norm=checkpoints.load_norm_stats
    def trace_load(*args,**kwargs):
        result=real_load(*args,**kwargs)
        missing,unexpected=result
        load_calls.append({'path':str(args[1] if len(args)>1 else kwargs['filename']),'missing_keys':list(missing),'unexpected_keys':list(unexpected),'strict':kwargs.get('strict',True)})
        assert not missing and not unexpected
        return result
    def trace_norm(*args,**kwargs):
        result=real_norm(*args,**kwargs)
        norm_calls.append({'path':str(Path(args[0])/args[1]/'norm_stats.json'),'object':result})
        return result
    emit('CHECKPOINT_RELOAD_BEGIN')
    torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter()
    with patch.object(st,'load_model',trace_load),patch.object(checkpoints,'load_norm_stats',trace_norm):
        policy=policy_config.create_trained_policy(cfg,CKPT,pytorch_device='cuda')
    torch.cuda.synchronize()
    R['checkpoint_reload_seconds']=time.perf_counter()-t
    assert len(load_calls)==1 and Path(load_calls[0]['path']).resolve()==(CKPT/'model.safetensors').resolve()
    assert len(norm_calls)==1 and Path(norm_calls[0]['path']).resolve()==(CKPT/'assets/biyi_stage1/norm_stats.json').resolve()
    model=policy._model
    assert not model.training and all(not m.training for m in model.modules())
    assert {str(p.device) for p in model.parameters()}=={'cuda:0'}
    R['reload']={'model_loaded':True,'eval_mode_all_modules':True,'device':'cuda:0','weight_load_calls':load_calls,'norm_stats_path':norm_calls[0]['path'],'base_fallback':False}
    source_norm=Path(os.environ['BIYI_ASSETS_BASE_DIR'])/'pi05_biyi_stage1/biyi_stage1/norm_stats.json'
    if not source_norm.is_file():
        matches=list(Path(os.environ['BIYI_ASSETS_BASE_DIR']).rglob('biyi_stage1/norm_stats.json'))
        assert len(matches)==1,matches
        source_norm=matches[0]
    assert json.loads(source_norm.read_text())==json.loads((CKPT/'assets/biyi_stage1/norm_stats.json').read_text())
    R['norm_stats_source_path']=str(source_norm)
    R['norm_stats_match_predeployed']=True
    ins=policy._input_transform.transforms; outs=policy._output_transform.transforms
    R['input_transform_order']=[type(t).__name__ for t in ins]
    R['output_transform_order']=[type(t).__name__ for t in outs]
    normalizer=next(t for t in ins if isinstance(t,transforms.Normalize))
    unnormalizer=next(t for t in outs if isinstance(t,transforms.Unnormalize))
    assert normalizer.norm_stats is norm_calls[0]['object'] and unnormalizer.norm_stats is norm_calls[0]['object']
    assert normalizer.use_quantiles and unnormalizer.use_quantiles
    assert sum(isinstance(t,transforms.Normalize) for t in ins)==1
    assert not any(isinstance(t,(transforms.DeltaActions,transforms.AbsoluteActions)) for t in (*ins,*outs))
    R['CHECKPOINT_RELOAD_STATUS']='PASS'
    emit(f'CHECKPOINT_RELOAD_PASS seconds={R["checkpoint_reload_seconds"]:.3f}')

    emit('OPTIMIZER_STATE_AUDIT_BEGIN')
    groups,group_summary=stage1_adamw_groups(model,weight_decay=meta['config']['optimizer']['weight_decay'])
    optim=torch.load(CKPT/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
    names=defaultdict(list); params={}
    for name,p in model.named_parameters(remove_duplicate=False): names[id(p)].append(name); params[id(p)]=p
    all_ids=set(params); trainable_ids={i for i,p in params.items() if p.requires_grad}
    group_ids=[id(p) for g in groups for p in g['params']]
    mapping={}; missing=[]; state_steps=Counter()
    assert len(groups)==len(optim['param_groups'])
    for gi,(live,saved) in enumerate(zip(groups,optim['param_groups'],strict=True)):
        assert len(live['params'])==len(saved['params'])
        assert live['weight_decay']==saved['weight_decay']
        for pos,(p,pid) in enumerate(zip(live['params'],saved['params'],strict=True)):
            assert pid not in mapping
            entry={'optimizer_parameter_id':pid,'group':gi,'position':pos,'name':names[id(p)][0],'aliases':names[id(p)],'shape':list(p.shape),'numel':p.numel(),'requires_grad':p.requires_grad,'module':names[id(p)][0].rpartition('.')[0]}
            state=optim['state'].get(pid)
            if state is None:
                missing.append(entry)
            else:
                step=int(state['step'].item()) if hasattr(state['step'],'item') else int(state['step'])
                assert state['exp_avg'].shape==p.shape and state['exp_avg_sq'].shape==p.shape
                assert state['exp_avg'].numel()>0 and state['exp_avg_sq'].numel()>0
                assert step==2
                state_steps[step]+=1
                entry['state_step']=step
            mapping[pid]=entry
    assert set(optim['state']).issubset(mapping)
    uncovered=trainable_ids-set(group_ids)
    assert not uncovered,'Trainable parameters omitted from groups'
    assert len(group_ids)==len(set(group_ids))
    AUD={'OPTIMIZER_STATE_COVERAGE':'PASS','unique_parameter_tensors':len(all_ids),'parameter_name_entries_including_aliases':sum(map(len,names.values())),'requires_grad_true_tensors':len(trainable_ids),'requires_grad_false_tensors':len(all_ids-trainable_ids),'optimizer_group_parameter_tensors':len(group_ids),'optimizer_state_count':len(optim['state']),'state_step_counts':dict(state_steps),'group_summary':group_summary,'uncovered_trainable_parameters':[names[i] for i in uncovered],'missing_optimizer_states':missing,'parameter_id_mapping':list(mapping.values()),'shared_parameter_aliases':[v for v in names.values() if len(v)>1],'interpretation':'All distinct trainable parameters are covered exactly once. Missing state means no gradient-generated AdamW state during the completed two steps; absence alone does not establish permanently unused parameters. No new backward or optimizer.step was run.'}
    assert len(optim['state'])==805 and len(all_ids)==812
    R['OPTIMIZER_STATE_COVERAGE']='PASS'
    save('optimizer_state_audit.json',AUD)
    emit('OPTIMIZER_STATE_AUDIT_PASS '+json.dumps({'unique':len(all_ids),'state':len(optim['state']),'missing':[e['name'] for e in missing]}))
    del optim, groups, mapping
    gc.collect()

    emit('CHECKPOINT_WEIGHT_COMPARISON_BEGIN')
    comparisons=[]; nonfinite=[]; shape_mismatch=[]
    with safe_open(str(BASE),framework='pt',device='cpu') as bf,safe_open(str(CKPT/'model.safetensors'),framework='pt',device='cpu') as sf:
        bk=set(bf.keys()); sk=set(sf.keys())
        assert bk==sk
        for idx,k in enumerate(sorted(sk)):
            a=bf.get_tensor(k); b=sf.get_tensor(k)
            if a.shape!=b.shape: shape_mismatch.append(k)
            if not finite_tensor(b): nonfinite.append(k)
            if k in ['action_in_proj.weight','action_out_proj.weight','time_mlp_in.weight','time_mlp_out.weight']:
                d=b.float()-a.float()
                comparisons.append({'name':k,'shape':list(b.shape),'base_dtype':str(a.dtype),'step2_dtype':str(b.dtype),'delta_l2_norm':float(torch.linalg.vector_norm(d.double())),'max_absolute_difference':float(d.abs().max()),'changed_elements':int(torch.count_nonzero(d)),'numel':b.numel()})
                assert torch.equal(model.state_dict()[k].detach().cpu(),b.to(model.state_dict()[k].dtype))
            del a,b
        assert not shape_mismatch and not nonfinite
    COMP={'CHECKPOINT_WEIGHT_UPDATE':'PASS' if any(c['changed_elements'] for c in comparisons) else 'REVIEW','base_path':str(BASE),'step2_path':str(CKPT/'model.safetensors'),'base_file_bytes':BASE.stat().st_size,'step2_file_bytes':(CKPT/'model.safetensors').stat().st_size,'matching_tensor_keys':len(sk),'all_tensor_shapes_match':not shape_mismatch,'step2_all_weight_tensors_finite':not nonfinite,'nonfinite_tensor_names':nonfinite,'shape_mismatch_names':shape_mismatch,'sampled_projections':comparisons,'direct_copy':False if any(c['changed_elements'] for c in comparisons) else 'not determined'}
    R['CHECKPOINT_WEIGHT_UPDATE']=COMP['CHECKPOINT_WEIGHT_UPDATE']
    save('checkpoint_weight_comparison.json',COMP)
    emit('CHECKPOINT_WEIGHT_COMPARISON_DONE '+json.dumps(comparisons))

    emit('REAL_DATASET_LOADING_BEGIN')
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    dc=dataclasses.replace(dc,episode_indices=None)
    dataset=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
    info=json.loads((Path(dc.local_root)/'meta/info.json').read_text())
    episodes=[json.loads(s) for s in (Path(dc.local_root)/'meta/episodes.jsonl').read_text().splitlines() if s.strip()]
    eps={int(e['episode_index']):e for e in episodes}
    offsets={}; offset=0
    for e in sorted(episodes,key=lambda x:x['episode_index']): offsets[e['episode_index']]=offset; offset+=int(e['length'])
    R['dataset']={'repo':dc.repo_id,'root':str(dc.local_root),'length':len(dataset),'total_frames':info['total_frames'],'episode_lengths':{str(i):e['length'] for i,e in eps.items()},'pinned_revision':'3eb637925fef4be6100e5acc1c3a203fa6ae91b8','revision_basis':'Existing deployment pinned snapshot; offline local files only','loader':'create_torch_dataset; no transform_dataset','frame_selection':'first frame or floor(length/2)'}
    assert len(dataset)==info['total_frames']
    captures=[]; real_sample=policy._sample_actions
    def trace_sample(*args,**kwargs):
        result=real_sample(*args,**kwargs)
        captures.append(arr(result[0]).copy())
        assert tuple(result.shape)==(1,50,32)
        return result
    policy._sample_actions=trace_sample
    observations={}
    for label,episode,frame in [('A',0,0),('B',0,int(eps[0]['length'])//2),('C',4,int(eps[4]['length'])//2)]:
        index=offsets[episode]+frame
        sample=dataset[index]
        assert int(arr(sample['episode_index']))==episode
        assert int(arr(sample['frame_index']))==frame
        obs={'images':{'head':arr(sample['observation.images.head']),'wrist':arr(sample['observation.images.wrist'])},'state':arr(sample['observation.state']).astype(np.float32),'prompt':sample['prompt']}
        assert obs['state'].shape==(17,) and np.isfinite(obs['state']).all()
        assert obs['prompt']==PROMPT,(episode,frame,obs['prompt'])
        assert all(im.ndim==3 and np.isfinite(im).all() for im in obs['images'].values())
        observations[label]=clone_obs(obs)
        mapped=biyivla_policy.BiyiVLAInputs()(clone_obs(obs))
        assert np.array_equal(mapped['image']['base_0_rgb'],biyivla_policy._image_to_hwc_uint8(obs['images']['head']))
        assert np.array_equal(mapped['image']['left_wrist_0_rgb'],biyivla_policy._image_to_hwc_uint8(obs['images']['wrist']))
        assert not mapped['image_mask']['right_wrist_0_rgb'] and not mapped['image']['right_wrist_0_rgb'].any()
        processed=policy._input_transform(clone_obs(obs))
        expected=normalizer(biyivla_policy.BiyiVLAInputs()(clone_obs(obs)))['state']
        assert processed['state'].shape==(32,) and np.array_equal(processed['state'][:17],expected)
        assert not processed['state'][17:].any()
        torch.cuda.synchronize(); t=time.perf_counter()
        with torch.no_grad(): output=policy.infer(clone_obs(obs))
        torch.cuda.synchronize(); elapsed=time.perf_counter()-t
        actions=np.asarray(output['actions'])
        assert actions.shape==(50,17) and np.issubdtype(actions.dtype,np.floating) and np.isfinite(actions).all()
        raw=captures[-1]
        expected_physical=unnormalizer({'state':processed['state'],'actions':raw})['actions'][:,:17]
        assert np.array_equal(actions,expected_physical)
        np.savez(OUT/f'actions_{label}.npz',actions=actions,model_actions_32=raw)
        row={'label':label,'episode':episode,'frame_index':frame,'global_frame_index':index,'timestamp_seconds':float(arr(sample['timestamp'])),'frame_source':{'parquet':info['data_path'].format(episode_chunk=episode//info['chunks_size'],episode_index=episode),'head_video':info['video_path'].format(episode_chunk=episode//info['chunks_size'],episode_index=episode,video_key='observation.images.head'),'wrist_video':info['video_path'].format(episode_chunk=episode//info['chunks_size'],episode_index=episode,video_key='observation.images.wrist')},'prompt':obs['prompt'],'raw_image_shapes':{k:list(v.shape) for k,v in obs['images'].items()},'raw_image_dtypes':{k:str(v.dtype) for k,v in obs['images'].items()},'raw_state_dtype':str(obs['state'].dtype),'raw_state_shape':[17],'model_state_shape':[32],'model_actions_shape':list(raw.shape),'actions_shape':list(actions.shape),'actions_dtype':str(actions.dtype),'finite':bool(np.isfinite(actions).all()),'output_min':float(actions.min()),'output_max':float(actions.max()),'inference_wall_seconds_synchronized':elapsed,'policy_reported_infer_ms':output['policy_timing']['infer_ms'],'physical_unnormalization_verified':True,'output_space':'absolute joint actions; unchanged first 17 dims, RM65[0:6] + SG100[6:17]'}
        R['observations'].append(row)
        emit('REAL_INFERENCE_PASS '+json.dumps(row,ensure_ascii=False))
    R['transform_validation']={'checkpoint_norm_stats_object_used_by_normalize_and_unnormalize':True,'quantile_normalization':True,'normalization_applied_once':True,'head_camera':'base_0_rgb','wrist_camera':'left_wrist_0_rgb','third_camera':'right_wrist_0_rgb','third_camera_mask':False,'state_dimensions':[17,32],'action_dimensions':[32,17],'physical_actions':True,'action_order':'RM65 6D + SG100 11D','no_delta_action_transform':True}
    # Separate fixed-noise repeatability; it does not replace the three random-noise runs.
    noise=np.random.default_rng(42).standard_normal((50,32)).astype(np.float32)
    repeats=[]; times=[]
    for _ in range(2):
        torch.cuda.synchronize(); t=time.perf_counter()
        with torch.no_grad(): result=policy.infer(clone_obs(observations['A']),noise=noise)
        torch.cuda.synchronize(); times.append(time.perf_counter()-t); repeats.append(result['actions'].copy())
    diff=np.abs(repeats[0]-repeats[1])
    R['fixed_noise_repeatability']={'observation':'A','noise_shape':[50,32],'seed':42,'bitwise_equal':bool(np.array_equal(*repeats)),'max_absolute_difference':float(diff.max()),'allclose_rtol_1e5_atol_1e6':bool(np.allclose(*repeats,rtol=1e-5,atol=1e-6)),'latencies_seconds':times,'finite':all(np.isfinite(x).all() for x in repeats)}
    assert R['fixed_noise_repeatability']['finite']
    R['INFERENCE_STATUS']='PASS'
    R['BiyiVLA_STAGE1_CHECKPOINT_INFERENCE']='PASS'
    policy._sample_actions=real_sample
    R['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    R['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    R['gpu_peak_allocated_GB']=torch.cuda.max_memory_allocated()/1e9
    R['gpu_peak_reserved_GB']=torch.cuda.max_memory_reserved()/1e9
    R['gpu_oom']=False
    del policy,model,real_sample,trace_sample,params,live,p
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    R['gpu_allocated_after_cleanup_bytes']=torch.cuda.memory_allocated()
    # Verify original checkpoint bytes and metadata remained unchanged.
    after={}
    for fn,before in fingerprints.items():
        p=CKPT/fn
        after[fn]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':digest(p)}
    assert after==fingerprints
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    R['checkpoint_unchanged_sha256_and_mtime']=True
    R['repository_unchanged']=True
except Exception as e:
    tb=traceback.format_exc()
    R['exceptions'].append({'type':type(e).__name__,'message':str(e),'traceback':tb})
    R['BiyiVLA_STAGE1_CHECKPOINT_INFERENCE']='FAIL'
    emit(tb)
finally:
    R['total_runtime_seconds']=time.perf_counter()-START
    R['output_directory']=str(OUT)
    if not AUD: AUD={'OPTIMIZER_STATE_COVERAGE':'REVIEW','exceptions':R['exceptions']}
    if not COMP: COMP={'CHECKPOINT_WEIGHT_UPDATE':'REVIEW','exceptions':R['exceptions']}
    save('optimizer_state_audit.json',AUD)
    save('checkpoint_weight_comparison.json',COMP)
    save('inference_validation.json',R)
    emit('FINAL_STATUS '+json.dumps({k:v for k,v in R.items() if k.endswith('STATUS') or k in ['OPTIMIZER_STATE_COVERAGE','CHECKPOINT_WEIGHT_UPDATE','BiyiVLA_STAGE1_CHECKPOINT_INFERENCE','total_runtime_seconds']},ensure_ascii=False))
if R['BiyiVLA_STAGE1_CHECKPOINT_INFERENCE']!='PASS': sys.exit(1)
