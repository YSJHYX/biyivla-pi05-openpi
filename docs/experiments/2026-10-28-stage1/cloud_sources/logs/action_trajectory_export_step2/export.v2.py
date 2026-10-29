import os
os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['JAX_PLATFORMS']='cpu'; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys,json,time,hashlib,logging,traceback,subprocess,csv,gc,dataclasses,importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch,transformers
import pyarrow.parquet as pq
import safetensors.torch as st
from transformers.models.siglip import check
from openpi.training import config as training_config,checkpoints
from openpi.training.data_loader import create_torch_dataset
from openpi.policies import policy_config
from openpi import transforms

ROOT=Path(os.environ['BIYI_ROOT']); OUT=ROOT/'logs/action_trajectory_export_step2'
CKPT=Path(os.environ['BIYI_CHECKPOINT_BASE_DIR'])/'pi05_biyi_stage1/local_smoke_a800_001/2'
DATA=Path(os.environ['BIYI_DATASET_LOCAL_ROOT'])
REV='3eb637925fef4be6100e5acc1c3a203fa6ae91b8'
COMMIT='bd34466dbecb8c8928e77b3b028fc59ef6e5ec56'
PROMPT='Pull the inner basket out of the air fryer and place it on the table.'
START=time.perf_counter(); torch.set_num_threads(8)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',force=True)
M={'ACTION_TRAJECTORY_EXPORT':'FAIL','BiyiVLA_STEP2_TRAJECTORY_EXPORT':'FAIL','exceptions':[],'samples':[],'pid':os.getpid(),'training_executed':False,'backward_executed':False,'optimizer_step_executed':False,'robot_controller_started':False,'output_directory':str(OUT)}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
 return h.hexdigest()
def arr(x): return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
def stat_record(p): return {'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}
def describe(x): return {'shape':list(x.shape),'dtype':str(x.dtype),'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),'finite':bool(np.isfinite(x).all())}
def diffmax(x,pairs):
 d=np.abs(np.diff(x,axis=0))[pairs]
 return {'overall':float(d.max()),'RM65':float(d[:,:6].max()),'SG100':float(d[:,6:].max())} if len(d) else None

try:
 prior=OUT/'export_metadata.attempt1.json'
 if prior.is_file():
  prev=json.loads(prior.read_text())
  M['prior_attempts']=[{'status':prev['ACTION_TRAJECTORY_EXPORT'],'failure_stage':'overstrict temporary exporter post-model-delete allocator assertion before serialization','exceptions':prev['exceptions'],'log_path':str(OUT/'action_trajectory_export.attempt1.log')}]
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==COMMIT
 assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 assert sys.version_info[:2]==(3,11) and torch.__version__=='2.7.1+cu126'
 assert transformers.__version__=='4.53.2' and check.check_whether_transformers_replace_is_installed_correctly()
 assert torch.cuda.device_count()==1
 baseline=json.loads((ROOT/'logs/checkpoint_reload_inference/inference_validation.json').read_text())
 cp_before=baseline['checkpoint_before']
 for name,record in cp_before.items():
  p=CKPT/name
  assert p.stat().st_size==record['bytes'] and p.stat().st_mtime_ns==record['mtime_ns']
 source_norm=Path(baseline['norm_stats_source_path']); source_norm_before=stat_record(source_norm)
 data_files=sorted(p for p in DATA.rglob('*') if p.is_file() and '.cache' not in p.parts and p.suffix in {'.json','.jsonl','.parquet','.mp4'})
 data_before={str(p.relative_to(DATA)):stat_record(p) for p in data_files}
 revision_evidence=[]
 for p in (DATA/'.cache/huggingface/download/meta').glob('*.metadata'):
  revision=p.read_text().splitlines()[0]; assert revision==REV
  revision_evidence.append({'path':str(p),'revision':revision})
 assert revision_evidence
 meta=torch.load(CKPT/'metadata.pt',map_location='cpu',weights_only=False)
 assert meta['global_step']==2
 info=json.loads((DATA/'meta/info.json').read_text())
 assert info['total_episodes']==5 and info['total_frames']==4468 and info['fps']==30
 episodes=[json.loads(s) for s in (DATA/'meta/episodes.jsonl').read_text().splitlines() if s.strip()]
 lengths={int(e['episode_index']):int(e['length']) for e in episodes}
 offsets={}; offset=0
 for episode in sorted(lengths): offsets[episode]=offset; offset+=lengths[episode]
 cfg=training_config.get_config('pi05_biyi_stage1')
 cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
 dc=cfg.data.create(cfg.assets_dirs,cfg.model)
 dc=dataclasses.replace(dc,episode_indices=None)
 dataset=create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
 raw=dataset
 while hasattr(raw,'_dataset'): raw=raw._dataset
 assert len(dataset)==4468
 expected_dt=[k/30 for k in range(50)]
 assert raw.delta_timestamps['action']==expected_dt
 assert list(raw.delta_indices['action'])==list(range(50))
 assert raw.image_transforms is None
 samples=[]; parquet_cache={}
 for label,episode,frame,seed in [('A',0,0,42),('B',0,431,43),('C',4,443,44)]:
  global_index=offsets[episode]+frame
  sample=dataset[global_index]
  assert int(arr(sample['episode_index']))==episode and int(arr(sample['frame_index']))==frame
  state=arr(sample['observation.state']); gt=arr(sample['action'])
  assert state.shape==(17,) and state.dtype==np.float32 and np.isfinite(state).all()
  assert gt.shape==(50,17) and gt.dtype==np.float32 and np.isfinite(gt).all()
  assert sample['prompt']==PROMPT
  valid=(frame+np.arange(50)<lengths[episode])
  assert 'action_is_pad' in sample
  assert np.array_equal(~arr(sample['action_is_pad']).astype(bool),valid)
  query,padding=raw._get_query_indices(global_index,episode)
  source_frames=np.minimum(frame+np.arange(50),lengths[episode]-1)
  assert np.array_equal(arr(query['action']),offsets[episode]+source_frames)
  assert np.array_equal(~arr(padding['action_is_pad']),valid)
  parquet=DATA/info['data_path'].format(episode_chunk=episode//info['chunks_size'],episode_index=episode)
  if episode not in parquet_cache:
   table=pq.read_table(parquet,columns=['action','frame_index','episode_index','timestamp']).to_pydict()
   parquet_cache[episode]=table
  table=parquet_cache[episode]
  assert len(table['action'])==lengths[episode]
  assert all(int(x)==episode for x in table['episode_index'])
  assert np.array_equal(np.asarray(table['frame_index']).reshape(-1),np.arange(lengths[episode]))
  recorded=np.asarray(table['action'],dtype=np.float32)[source_frames]
  assert np.array_equal(gt,recorded),'Raw action chunk does not equal original Parquet a[t+k]'
  timestamps=np.asarray(table['timestamp']).reshape(-1)[source_frames]
  assert np.allclose(timestamps[valid]-timestamps[0],np.arange(50)[valid]/30,atol=1e-4,rtol=0)
  assert float(arr(sample['timestamp']))==float(timestamps[0])
  obs={'images':{'head':arr(sample['observation.images.head']),'wrist':arr(sample['observation.images.wrist'])},'state':state.copy(),'prompt':sample['prompt']}
  assert all(im.shape==(3,480,640) and im.dtype==np.float32 and np.isfinite(im).all() and im.min()>=0 and im.max()<=1 for im in obs['images'].values())
  noise=np.random.default_rng(seed).standard_normal((50,32)).astype(np.float32)
  row={'label':label,'episode_index':episode,'frame_index':frame,'global_index':global_index,'episode_length':lengths[episode],'noise_seed':seed,'valid_gt_steps':int(valid.sum()),'input_state':state.tolist(),'prompt':sample['prompt'],'action_source':'raw_sample["action"], independently matched to original Parquet rows','parquet_path':str(parquet),'alignment':'o_t -> predicted a_(t+k) and recorded raw action a_(t+k), k=0..49','gt_source_frame_indices':source_frames.tolist(),'requested_horizon_frame_indices':(frame+np.arange(50)).tolist(),'gt_source_timestamps_seconds':timestamps.tolist(),'lerobot_padding_matches_valid_mask':True,'gt_equals_original_parquet':True,'input_image_format':'float32 CHW RGB [0,1], single frame','head_video':str(DATA/info['video_path'].format(episode_chunk=episode//info['chunks_size'],video_key='observation.images.head',episode_index=episode)),'wrist_video':str(DATA/info['video_path'].format(episode_chunk=episode//info['chunks_size'],video_key='observation.images.wrist',episode_index=episode))}
  samples.append({'obs':obs,'gt':gt.copy(),'valid':valid,'noise':noise,'row':row})
  logging.info('RAW_SAMPLE_ALIGNMENT_PASS '+json.dumps(row))
 # No trajectory export occurs before all three index/time alignment checks pass.
 calls=[]; norms=[]; real_load=st.load_model; real_norm=checkpoints.load_norm_stats
 def load_spy(*args,**kwargs):
  result=real_load(*args,**kwargs); missing,unexpected=result
  calls.append({'path':str(args[1]),'strict':kwargs.get('strict',True),'missing_keys':list(missing),'unexpected_keys':list(unexpected)})
  assert not missing and not unexpected
  return result
 def norm_spy(*args,**kwargs):
  result=real_norm(*args,**kwargs); norms.append((Path(args[0])/args[1]/'norm_stats.json',result)); return result
 logging.info('STEP2_POLICY_RELOAD_BEGIN')
 torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
 with patch.object(st,'load_model',load_spy),patch.object(checkpoints,'load_norm_stats',norm_spy):
  policy=policy_config.create_trained_policy(cfg,CKPT,pytorch_device='cuda')
 torch.cuda.synchronize(); reload_seconds=time.perf_counter()-t
 assert len(calls)==1 and Path(calls[0]['path']).resolve()==(CKPT/'model.safetensors').resolve()
 assert all(not module.training for module in policy._model.modules())
 assert {str(p.device) for p in policy._model.parameters()}=={'cuda:0'}
 assert len(norms)==1 and norms[0][0].resolve()==(CKPT/'assets/biyi_stage1/norm_stats.json').resolve()
 for transforms_list,kind in [(policy._input_transform.transforms,transforms.Normalize),(policy._output_transform.transforms,transforms.Unnormalize)]:
  stats_transforms=[t for t in transforms_list if isinstance(t,kind)]
  assert len(stats_transforms)==1 and stats_transforms[0].norm_stats is norms[0][1]
 M.update({'git_commit':COMMIT,'dataset_repo_id':dc.repo_id,'dataset_revision':REV,'dataset_revision_evidence':revision_evidence,'checkpoint_path':str(CKPT),'checkpoint_global_step':2,'task_prompt':PROMPT,'fps':30,'action_horizon':50,'physical_action_dim':17,'internal_action_dim':32,'action_type':'absolute_joint_position','joint_dim':{'0:6':'RM65','6:17':'SG100'},'joint_order':'raw action order preserved; no joint reordering','official_joint_names':None,'official_joint_names_note':'meta/info.json only provides generic action name, not per-joint names; CSV uses physical dimension indices','joint_dimension_index_map':[{'physical_index':i,'subsystem':'RM65' if i<6 else 'SG100','subsystem_index':i if i<6 else i-6} for i in range(17)],'delta_timestamps_action':raw.delta_timestamps['action'],'delta_indices_action':list(raw.delta_indices['action']),'padding_behavior':'LeRobot clamps to episode boundary and sets action_is_pad=True; no cross-episode concatenation','policy_reload':{'seconds':reload_seconds,'weight_load_calls':calls,'checkpoint_norm_stats_path':str(norms[0][0]),'eval_mode':True,'device':'cuda:0','compile_mode':None},'environment':{'python':sys.version.split()[0],'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'transformers':transformers.__version__,'patch':'PASS','gpu':torch.cuda.get_device_name(0)},'checkpoint_baseline':cp_before})
 for sample in samples:
  torch.cuda.synchronize(); t=time.perf_counter()
  with torch.no_grad(): output=policy.infer(sample['obs'],noise=sample['noise'])
  torch.cuda.synchronize(); ms=(time.perf_counter()-t)*1000
  pred=np.asarray(output['actions'])
  assert pred.shape==(50,17) and np.issubdtype(pred.dtype,np.floating) and np.isfinite(pred).all()
  assert np.array_equal(sample['obs']['state'],np.asarray(sample['row']['input_state'],dtype=np.float32))
  gt=sample['gt']; valid=sample['valid']; error=np.abs(pred[valid]-gt[valid])
  row=sample['row']; row.update({'inference_latency_ms':ms,'predicted_actions_dtype':str(pred.dtype),'ground_truth_actions_dtype':str(gt.dtype),'predicted':describe(pred),'ground_truth':describe(gt),'MAE':{'RM65':float(error[:,:6].mean()),'SG100':float(error[:,6:].mean()),'overall_descriptive_only':float(error.mean())},'prediction_smoothness_max_adjacent_absolute_difference':diffmax(pred,np.ones(49,dtype=bool)),'gt_smoothness_max_adjacent_absolute_difference':diffmax(gt,valid[:-1]&valid[1:])})
  sample['pred']=pred.copy(); M['samples'].append(row)
  logging.info('FIXED_NOISE_INFERENCE_PASS '+json.dumps(row))
 M['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated(); M['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
 # Release model before serialization, plotting and readback validation.
 del policy
 gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
 M['gpu_allocated_after_model_release_bytes']=torch.cuda.memory_allocated()
 logging.info('GPU_ALLOCATED_AFTER_MODEL_DELETE_BYTES '+str(M['gpu_allocated_after_model_release_bytes']))
 M['resource_release_check']='Record in-process allocator residual; verify no GPU process after interpreter exits'
 export_arrays={'predicted_actions':np.stack([s['pred'] for s in samples]),'ground_truth_actions':np.stack([s['gt'] for s in samples]),'observation_state':np.stack([s['obs']['state'] for s in samples]),'valid_action_mask':np.stack([s['valid'] for s in samples]),'inference_noise':np.stack([s['noise'] for s in samples]),'episode_indices':np.array([s['row']['episode_index'] for s in samples],dtype=np.int64),'frame_indices':np.array([s['row']['frame_index'] for s in samples],dtype=np.int64),'global_indices':np.array([s['row']['global_index'] for s in samples],dtype=np.int64),'noise_seeds':np.array([s['row']['noise_seed'] for s in samples],dtype=np.int64),'horizon_frame_indices':np.stack([s['row']['requested_horizon_frame_indices'] for s in samples]).astype(np.int64),'gt_source_frame_indices':np.stack([s['row']['gt_source_frame_indices'] for s in samples]).astype(np.int64),'gt_source_timestamps_seconds':np.stack([s['row']['gt_source_timestamps_seconds'] for s in samples])}
 for sample in samples:
  label=sample['row']['label']; pred=sample['pred']; gt=sample['gt']
  np.save(OUT/f'sample_{label}_predicted_actions.npy',pred,allow_pickle=False)
  np.save(OUT/f'sample_{label}_ground_truth_actions.npy',gt,allow_pickle=False)
  headers=['step','valid_gt']+[f'pred_arm_{i}' for i in range(6)]+[f'gt_arm_{i}' for i in range(6)]+[f'pred_hand_{i}' for i in range(11)]+[f'gt_hand_{i}' for i in range(11)]
  csvpath=OUT/f'sample_{label}_action_comparison.csv'
  with csvpath.open('w',newline='') as f:
   writer=csv.writer(f); writer.writerow(headers)
   for k in range(50): writer.writerow([k,int(sample['valid'][k]),*pred[k,:6].tolist(),*gt[k,:6].tolist(),*pred[k,6:].tolist(),*gt[k,6:].tolist()])
  with csvpath.open() as f: rows=list(csv.DictReader(f))
  assert len(rows)==50
  for k,row in enumerate(rows):
   assert int(row['step'])==k and bool(int(row['valid_gt']))==bool(sample['valid'][k])
   assert np.array_equal(np.asarray([float(row[f'pred_arm_{i}']) for i in range(6)]+[float(row[f'pred_hand_{i}']) for i in range(11)]),pred[k])
   assert np.array_equal(np.asarray([float(row[f'gt_arm_{i}']) for i in range(6)]+[float(row[f'gt_hand_{i}']) for i in range(11)],dtype=np.float32),gt[k])
 npzpath=OUT/'biyi_step2_action_comparison.npz'
 np.savez_compressed(npzpath,**export_arrays)
 with np.load(npzpath,allow_pickle=False) as saved:
  shapes={'predicted_actions':(3,50,17),'ground_truth_actions':(3,50,17),'observation_state':(3,17),'valid_action_mask':(3,50),'inference_noise':(3,50,32)}
  for key,shape in shapes.items(): assert saved[key].shape==shape
  for key,value in export_arrays.items(): assert np.array_equal(saved[key],value) and np.isfinite(saved[key]).all()
  for j,sample in enumerate(samples):
   label=sample['row']['label']
   for key in ['predicted_actions','ground_truth_actions']:
    assert np.array_equal(saved[key][j],np.load(OUT/f'sample_{label}_{key}.npy',allow_pickle=False))
   assert int(saved['episode_indices'][j])==sample['row']['episode_index'] and int(saved['frame_indices'][j])==sample['row']['frame_index']
   assert np.array_equal(saved['inference_noise'][j],np.random.default_rng(int(saved['noise_seeds'][j])).standard_normal((50,32)).astype(np.float32))
 M['npz_path']=str(npzpath); M['npz_sha256']=sha(npzpath); M['disk_roundtrip_checks']='PASS'; M['csv_roundtrip_checks']='PASS'
 M['plots']={'status':'SKIPPED','paths':[]}
 if importlib.util.find_spec('matplotlib'):
  try:
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   plot_paths=[]
   for sample in samples:
    label=sample['row']['label']; row=sample['row']
    for subsystem,start,count,nrows,ncols in [('arm',0,6,3,2),('hand',6,11,4,3)]:
     fig,axes=plt.subplots(nrows,ncols,figsize=(12,9 if subsystem=='arm' else 11),sharex=True,constrained_layout=True)
     for j,ax in enumerate(axes.flat):
      if j>=count: ax.set_visible(False); continue
      dim=start+j
      ax.plot(np.arange(50),sample['pred'][:,dim],color='tab:blue',linestyle='-',label='Predicted')
      masked=np.where(sample['valid'],sample['gt'][:,dim],np.nan)
      ax.plot(np.arange(50),masked,color='tab:orange',linestyle='--',label='Recorded GT')
      ax.set_title(f'{subsystem}_{j} (physical dim {dim})',fontsize=10)
      ax.set_xlim(0,49); ax.grid(alpha=.25); ax.set_xlabel('Horizon step'); ax.set_ylabel('Absolute joint position')
      if j==0: ax.legend(fontsize=8)
     fig.suptitle(f'Sample {label} | Episode {row["episode_index"]}, frame {row["frame_index"]} | Step-2 checkpoint | seed {row["noise_seed"]}',fontsize=12)
     png=OUT/f'sample_{label}_{subsystem}_trajectory.png'; fig.savefig(png,dpi=140); plt.close(fig); plot_paths.append(str(png))
   M['plots']={'status':'PASS','paths':plot_paths}
  except Exception:
   M['plots']={'status':'OPTIONAL_PLOT_FAILED','paths':[],'traceback':traceback.format_exc()}
   logging.warning(M['plots']['traceback'])
 # Final preservation checks against prior validated checkpoint hashes and fresh input hashes.
 cp_after={name:stat_record(CKPT/name) for name in cp_before}
 assert cp_after==cp_before,'Original checkpoint changed'
 assert stat_record(source_norm)==source_norm_before
 assert {str(p.relative_to(DATA)):stat_record(p) for p in data_files}==data_before,'Original dataset changed'
 assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
 M['preservation_checks']={'checkpoint_sha256_size_mtime_unchanged':True,'predeployed_norm_stats_sha256_size_mtime_unchanged':True,'dataset_sha256_size_mtime_unchanged':True,'dataset_files_checked':len(data_files),'repository_clean':True}
 M['output_files']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.suffix in {'.npz','.npy','.csv','.png'}}
 M['ACTION_TRAJECTORY_EXPORT']='PASS'; M['BiyiVLA_STEP2_TRAJECTORY_EXPORT']='PASS'; M['gpu_oom']=False
 M['interpretation']='Descriptive pipeline validation only. MAE is computed on valid GT timesteps; RM65 and SG100 are reported separately. No convergence, generalization, task success or robot safety conclusion is drawn from a two-step checkpoint.'
except Exception as e:
 M['exceptions'].append({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
 logging.error(M['exceptions'][-1]['traceback'])
finally:
 M['total_validation_seconds']=time.perf_counter()-START
 (OUT/'export_metadata.json').write_text(json.dumps(M,indent=2,ensure_ascii=False)+'\n')
 logging.info('FINAL_STATUS '+json.dumps({k:M[k] for k in ['ACTION_TRAJECTORY_EXPORT','BiyiVLA_STEP2_TRAJECTORY_EXPORT','total_validation_seconds']}))
if M['ACTION_TRAJECTORY_EXPORT']!='PASS': sys.exit(1)
