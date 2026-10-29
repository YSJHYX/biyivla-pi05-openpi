import os,sys,json,time,stat,re,hashlib,subprocess,platform,importlib.metadata as md,struct
from pathlib import Path
from collections import defaultdict
import torch,transformers
from transformers.models.siglip import check
ROOT=Path(os.environ['BIYI_ROOT']); OUT=ROOT/'logs/archive_preflight_20261020'
COMMIT='bd34466dbecb8c8928e77b3b028fc59ef6e5ec56'; REV='3eb637925fef4be6100e5acc1c3a203fa6ae91b8'
START=time.time(); R={'STAGE1_ARCHIVE_PREFLIGHT':'FAIL','backup_executed':False,'original_files_deleted_or_moved':False,'instance_shutdown_or_release_executed':False,'gpu_training_or_inference_executed':False,'root':str(ROOT),'report_directory':str(OUT),'checks':{},'exceptions':[]}
def cmd(args): return subprocess.check_output(args,text=True,stderr=subprocess.DEVNULL).strip()
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def safeinfo(p):
 s=p.lstat(); return {'path':str(p),'mode':oct(stat.S_IMODE(s.st_mode)),'bytes':s.st_size,'kind':'symlink' if p.is_symlink() else 'regular' if p.is_file() else 'other'}
def collect(root):
 files=[]; dirs=[]; links=[]; special=[]
 for here,ds,fs in os.walk(root,followlinks=False):
  ds[:]=[d for d in ds if not (Path(here)/d==OUT)]
  for name in list(ds):
   p=Path(here)/name
   if p.is_symlink(): links.append(p); ds.remove(name)
   else: dirs.append(p)
  for name in fs:
   p=Path(here)/name
   if p.is_symlink(): links.append(p)
   elif p.is_file(): files.append(p)
   else: special.append(p)
 return files,dirs,links,special
try:
 assert cmd(['git','rev-parse','HEAD'])==COMMIT
 assert not cmd(['git','status','--porcelain'])
 gpu_apps=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
 assert not gpu_apps
 processes=[]; env_credentials=[]
 envkeys=re.compile(r'(^|_)(TOKEN|API_KEY|SECRET|PASSWORD|ACCESS_KEY_ID|SECRET_ACCESS_KEY)$')
 jobs=re.compile(r'(^|/)(train_pytorch\.py|torchrun|serve_policy\.py|validate\.py|export\.v2\.py|export\.py)$')
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit(): continue
  try:
   args=(proc/'cmdline').read_bytes().split(b'\x00')
   if any(jobs.search(a.decode(errors='replace')) for a in args): processes.append({'pid':int(proc.name),'executable':str(Path(os.readlink(proc/'exe'))),'comm':(proc/'comm').read_text().strip()})
   names=[]
   for item in (proc/'environ').read_bytes().split(b'\x00'):
    if b'=' not in item: continue
    key,value=item.split(b'=',1); key=key.decode(errors='replace')
    if value and envkeys.search(key): names.append(key)
   if names: env_credentials.append({'pid':int(proc.name),'comm':(proc/'comm').read_text().strip(),'variable_names':sorted(names)})
  except (PermissionError,FileNotFoundError,ProcessLookupError): pass
 assert not processes
 R['process_checks']={'gpu_compute_processes':[],'training_or_inference_processes':processes,'nvidia_smi_gpu':cmd(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.total','--format=csv,noheader']),'checked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 R['git_commit']=COMMIT;R['git_clean']=True
 mounts={}
 for label,path in [('system','/'),('data',str(ROOT))]:
  entry=json.loads(cmd(['findmnt','-J','-T',path,'-o','TARGET,SOURCE,FSTYPE']))['filesystems'][0]
  v=os.statvfs(path); entry.update({'capacity_bytes':v.f_blocks*v.f_frsize,'available_bytes':v.f_bavail*v.f_frsize,'used_bytes':(v.f_blocks-v.f_bfree)*v.f_frsize}); mounts[label]=entry
 assert mounts['system']['target']=='/' and mounts['data']['target']=='/root/autodl-tmp'
 R['mounts']=mounts
 files,dirs,links,special=collect(ROOT)
 R['file_counts']={'regular':len(files),'directories':len(dirs),'symlinks':len(links),'special':len(special)}
 # Independent per-directory sizes and union sizes; hardlink aliases never count twice in the union.
 by_top=defaultdict(list)
 for p in files: by_top[p.relative_to(ROOT).parts[0]].append(p)
 sizes=[]
 for p in sorted(ROOT.iterdir()):
  if p==OUT: continue
  if p.is_dir() and not p.is_symlink():
   allocation=int(cmd(['du','-x','-B1','-s',str(p)]).split()[0]); apparent=int(cmd(['du','-x','--apparent-size','-B1','-s',str(p)]).split()[0])
  else: allocation=p.lstat().st_blocks*512; apparent=p.lstat().st_size
  sizes.append({'name':p.name,'allocated_bytes':allocation,'apparent_bytes':apparent,'regular_file_entries':len(by_top.get(p.name,[]))})
 inodes={}; inode_paths=defaultdict(list)
 for p in files:
  s=p.stat(); key=(s.st_dev,s.st_ino); inodes[key]=s; inode_paths[key].append(p)
 total_alloc=int(cmd(['du','-x','-B1','-s',str(ROOT)]).split()[0])
 total_apparent=int(cmd(['du','-x','--apparent-size','-B1','-s',str(ROOT)]).split()[0])
 R['directory_sizes']=sizes
 R['size_totals']={'du_allocated_bytes':total_alloc,'du_apparent_bytes_hardlink_deduplicated':total_apparent,'regular_file_bytes_unique_inodes':sum(s.st_size for s in inodes.values()),'regular_file_bytes_counting_every_path':sum(p.stat().st_size for p in files),'hardlink_groups_with_multiple_paths':sum(len(paths)>1 for paths in inode_paths.values()),'hardlink_paths_beyond_first':sum(max(0,len(paths)-1) for paths in inode_paths.values()),'note':'Measured before writing report files. du uses allocated blocks; apparent counts logical bytes. Proposed tar/rsync preserves links.'}
 linkrows=[]
 for p in links:
  dest=p.resolve(); linkrows.append({'path':str(p.relative_to(ROOT)),'target':os.readlink(p),'resolved':str(dest),'exists':dest.exists(),'outside_biyi_root':not dest.is_relative_to(ROOT)})
 save('symlinks.json',linkrows)
 # Credentials: emit locations/categories only, never matching contents or values.
 findings={}
 def flag(p,kind):
  key=str(p); entry=findings.setdefault(key,{**safeinfo(p),'categories':[]})
  if kind not in entry['categories']:entry['categories'].append(kind)
 known_names={'token','stored_tokens','.netrc','.git-credentials','credentials','credentials.json','application_default_credentials.json','id_rsa','id_ed25519','id_ecdsa','id_dsa','.env'}
 candidates=set(files)
 for base in ['/root/.ssh','/root/.config','/root/.cache/huggingface','/root/.aws','/root/.azure','/root/.kube','/root/.jupyter','/etc/profile.d']:
  p=Path(base)
  if p.exists():
   for q in p.rglob('*'):
    if q.is_file() and not q.is_symlink(): candidates.add(q)
 for name in ['.bashrc','.bash_history','.profile','.gitconfig','.netrc','.git-credentials']:
  p=Path('/root')/name
  if p.is_file(): candidates.add(p)
 for p in [Path('/etc/environment'),Path('/etc/profile')]:
  if p.is_file(): candidates.add(p)
 patterns={
  'private_key_material':re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
  'huggingface_token_literal':re.compile(rb'\bhf_[A-Za-z0-9]{25,}\b'),
  'cloud_access_key_literal':re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
  'credential_in_url':re.compile(rb'https?://[^\s/:@]+:[^\s/@]+@'),
  'api_key_or_token_assignment':re.compile(rb'''(?im)(?:WANDB_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|AWS_SECRET_ACCESS_KEY|API_KEY|api_key|access_token|password)["']?\s*[:=]\s*["']?([A-Za-z0-9_\-/.+=]{25,})''')
 }
 scanned=0
 for p in sorted(candidates):
  try:
   named=p.name in known_names or p.name.startswith('.env.') or p.name in {'authorized_keys','known_hosts'}
   if named: flag(p,'credential_location' if p.name not in {'authorized_keys','known_hosts'} else 'ssh_access_metadata_public_keys_or_hosts')
   # Avoid scanning installed dependency test fixtures and caches as if they were deployed user secrets.
   rel=p.relative_to(ROOT) if p.is_relative_to(ROOT) else None
   if rel and rel.parts[0] in {'envs','models','checkpoints','datasets','runtime','bin'}: scan=named
   elif rel and rel.parts[:2]==('cache','uv'): scan=named
   elif rel and '.git' in rel.parts and ('objects' in rel.parts): scan=False
   else: scan=True
   if not scan or p.stat().st_size>4*1024*1024: continue
   data=p.read_bytes()
   if b'\x00' in data[:8192] and not named:continue
   scanned+=1
   for category,pattern in patterns.items():
    if pattern.search(data):flag(p,category)
  except (PermissionError,OSError):pass
 cred={'file_findings':list(findings.values()),'process_environment_credential_locations':env_credentials,'text_files_scanned':scanned,'scope':'Biyi root; known root credential/config locations; process environment variable names only. Dependency code/large binaries excluded from content scan. This is a location/signature scan, not a guarantee that arbitrary or encrypted secrets are absent. No credential contents or values logged.'}
 save('credential_locations.json',cred)
 excluded={Path(k) for k,v in findings.items() if Path(k).is_relative_to(ROOT) and v['categories']!=['ssh_access_metadata_public_keys_or_hosts']}
 # A candidate credential file is excluded from the ordinary dataset/environment backup, never deleted.
 exclude_lines=[]
 for p in sorted(excluded):exclude_lines.append('/'+str(p.relative_to(ROOT)))
 for p in special:exclude_lines.append('/'+str(p.relative_to(ROOT)))
 (OUT/'backup-exclude.txt').write_text('\n'.join(exclude_lines)+'\n')
 R['credentials']={'report':'credential_locations.json','file_findings_count':len(findings),'within_biyi_root_excluded_paths':[str(p) for p in sorted(excluded)],'outside_biyi_root_locations':[str(p) for p in findings if not Path(p).is_relative_to(ROOT)],'environment_variable_locations_count':len(env_credentials),'scan_scope':cred['scope']}
 # Runtime manifest contains distribution names/versions only, never pip URLs or credentials.
 assert sys.version_info[:2]==(3,11) and torch.__version__=='2.7.1+cu126' and transformers.__version__=='4.53.2'
 assert check.check_whether_transformers_replace_is_installed_correctly()
 realpy=Path(sys.executable).resolve(); assert realpy.is_relative_to(ROOT) and Path(sys._base_executable).exists()
 packages=sorted([{'name':d.metadata.get('Name','unknown'),'version':d.version} for d in md.distributions()],key=lambda x:x['name'].lower())
 save('python_packages.json',packages)
 (OUT/'python_packages.txt').write_text('\n'.join(f"{d['name']}=={d['version']}" for d in packages)+'\n')
 R['environment']={'python':sys.version.split()[0],'python_executable':sys.executable,'python_real_executable':str(realpy),'python_base_prefix':sys.base_prefix,'python_base_executable':sys._base_executable,'torch':torch.__version__,'torch_cuda_runtime':torch.version.cuda,'transformers':transformers.__version__,'transformers_patch':'PASS','platform':platform.platform(),'machine':platform.machine(),'glibc':platform.libc_ver(),'os_release':Path('/etc/os-release').read_text(),'kernel':platform.release(),'driver':cmd(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']),'package_count':len(packages),'safe_environment_paths':{k:os.environ.get(k) for k in ['BIYI_ROOT','BIYI_PYTHON','BIYI_BASE_CHECKPOINT_DIR','BIYI_DATASET_REPO_ID','BIYI_DATASET_LOCAL_ROOT','BIYI_ASSETS_BASE_DIR','BIYI_CHECKPOINT_BASE_DIR','OPENPI_DATA_HOME','HF_HOME','LD_LIBRARY_PATH','UV_CACHE_DIR','UV_PYTHON_INSTALL_DIR']},'python_ldd':cmd(['ldd',str(realpy)]),'venv_pyvenv_cfg':Path(sys.prefix,'pyvenv.cfg').read_text(),'system_packages':cmd(['dpkg-query','-W','-f=${Package}\t${Version}\n','libc6','libstdc++6','ca-certificates','libglib2.0-0'])}
 assert 'not found' not in R['environment']['python_ldd']
 editable=[]
 for p in Path(sys.prefix).glob('lib/python*/site-packages/*.pth'):
  if p.stat().st_size<65536:editable.append({'path':str(p),'sha256':digest(p),'local_path_references':re.findall(r'/root/autodl-tmp/biyivla[^\s\"\']*',p.read_text(errors='replace'))})
 R['environment']['editable_pth_files']=editable
 data=Path(os.environ['BIYI_DATASET_LOCAL_ROOT']); info=json.loads((data/'meta/info.json').read_text())
 episodes=[json.loads(s) for s in (data/'meta/episodes.jsonl').read_text().splitlines() if s.strip()]
 assert info['total_episodes']==5 and info['total_frames']==4468 and info['fps']==30
 assert sum(e['length'] for e in episodes)==4468
 revisions=[]
 for p in (data/'.cache/huggingface/download/meta').glob('*.metadata'):
  r=p.read_text().splitlines()[0];assert r==REV;revisions.append({'path':str(p.relative_to(ROOT)),'revision':r})
 assert revisions
 for e in episodes:
  ep=e['episode_index'];chunk=ep//info['chunks_size']
  assert (data/info['data_path'].format(episode_chunk=chunk,episode_index=ep)).stat().st_size>0
  for camera in ['observation.images.head','observation.images.wrist']:assert (data/info['video_path'].format(episode_chunk=chunk,video_key=camera,episode_index=ep)).stat().st_size>0
 R['dataset']={'repo_id':os.environ['BIYI_DATASET_REPO_ID'],'revision':REV,'metadata_revision_evidence':revisions,'episodes':5,'frames':4468,'fps':30,'parquet_files':len(list(data.rglob('*.parquet'))),'video_files':len(list(data.rglob('*.mp4')))}
 assert R['dataset']['parquet_files']==5 and R['dataset']['video_files']==10
 # Hash immutable key artifacts, all dataset content, runtime assets and small launch configuration.
 critical=[]
 for p in files:
  rel=p.relative_to(ROOT);parts=rel.parts
  if p in excluded:continue
  if parts[0] in {'models','checkpoints','datasets','assets','runtime'} or parts[:2]==('cache','openpi'):
   critical.append(p)
  elif len(parts)==1 and p.suffix=='.sh': critical.append(p)
  elif parts[0]=='logs' and p.suffix in {'.json','.log','.npz','.npy','.csv','.png','.md','.sha256'}: critical.append(p)
 for p in [ROOT/'openpi/uv.lock',ROOT/'openpi/pyproject.toml',realpy,Path(sys.prefix)/'pyvenv.cfg']:
  if p not in excluded:critical.append(p)
 for p in (ROOT/'openpi/src/openpi/models_pytorch/transformers_replace').rglob('*.py'):critical.append(p)
 critical=sorted(set(critical));manifest=[]
 hash_path=OUT/'SHA256SUMS.critical'
 with hash_path.open('w') as f:
  for p in critical:
   s=p.stat(); h=digest(p); assert (p.stat().st_size,p.stat().st_mtime_ns)==(s.st_size,s.st_mtime_ns)
   row={'path':str(p.relative_to(ROOT)),'bytes':s.st_size,'sha256':h};manifest.append(row);f.write(h+'  '+row['path']+'\n')
 save('critical_files.json',manifest)
 bypath={e['path']:e for e in manifest}
 baseline=json.loads((ROOT/'logs/checkpoint_reload_inference/inference_validation.json').read_text())
 cp=Path(os.environ['BIYI_CHECKPOINT_BASE_DIR'])/'pi05_biyi_stage1/local_smoke_a800_001/2'
 for name,entry in baseline['checkpoint_before'].items():assert bypath[str((cp/name).relative_to(ROOT))]['sha256']==entry['sha256']
 base=Path(os.environ['BIYI_BASE_CHECKPOINT_DIR'])/'model.safetensors'
 assert bypath[str(base.relative_to(ROOT))]['sha256']=='dbcbe3ea978ce1d6f791582fa034e7ece8a41feaad2d3c9a5b014c1938610e2a'
 with (cp/'model.safetensors').open('rb') as f:hlen=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(hlen))
 tensors={k:v for k,v in header.items() if k!='__metadata__'};assert len(tensors)==812
 meta=torch.load(cp/'metadata.pt',map_location='cpu',weights_only=False);assert meta['global_step']==2
 opt=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
 assert len(opt['state'])==805 and all(int(s['step'])==2 for s in opt['state'].values())
 assert all(s['exp_avg'].numel()>0 and s['exp_avg_sq'].numel()>0 for s in opt['state'].values()); del opt
 normpath=cp/'assets/biyi_stage1/norm_stats.json'; assert json.loads(normpath.read_text())==json.loads(Path(baseline['norm_stats_source_path']).read_text())
 for path,key in [('logs/local_smoke_a800_001.result.json','SMOKE_STATUS'),('logs/checkpoint_reload_inference/inference_validation.json','BiyiVLA_STAGE1_CHECKPOINT_INFERENCE'),('logs/action_trajectory_export_step2/export_metadata.json','BiyiVLA_STEP2_TRAJECTORY_EXPORT')]:assert json.loads((ROOT/path).read_text())[key]=='PASS'
 R['integrity']={'critical_file_count':len(manifest),'critical_hashed_bytes':sum(e['bytes'] for e in manifest),'base_checkpoint':bypath[str(base.relative_to(ROOT))],'step2_files':[bypath[str((cp/n).relative_to(ROOT))] for n in baseline['checkpoint_before']],'metadata_global_step':2,'optimizer_nonempty_states':805,'all_optimizer_state_steps':2,'safetensors_tensor_count':812,'norm_stats_match':True,'smoke_reload_export_reports_pass':True}
 # Inventory excludes credentials content; hashes above do not include any flagged credential files.
 with (OUT/'file_inventory.jsonl').open('w') as f:
  for p in files:
   s=p.stat();f.write(json.dumps({'path':str(p.relative_to(ROOT)),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mode':oct(stat.S_IMODE(s.st_mode)),'uid':s.st_uid,'gid':s.st_gid,'mtime_ns':s.st_mtime_ns,'nlink':s.st_nlink,'exclude_from_ordinary_backup':p in excluded})+'\n')
 included=[p for p in files if p not in excluded]; unique={}
 for p in included:
  s=p.stat();unique[(s.st_dev,s.st_ino)]=s
 payload=sum(s.st_size for s in unique.values()); entries=len(included)+len(dirs)+len(links)
 # Tar logical payload padded to 512, plus conservative 2 KiB/header for long paths and extensions.
 raw_tar_est=sum(((s.st_size+511)//512)*512 for s in unique.values())+entries*2048+10240
 R['archive_estimate']={'included_regular_file_paths':len(included),'logical_payload_bytes_preserving_hardlinks':payload,'uncompressed_tar_estimate_bytes':raw_tar_est,'uncompressed_tar_estimate_note':'Planning estimate with 512-byte padding and 2 KiB per pathname metadata; archive not created, compression ratio not measured.','recommended_minimum_destination_free_bytes':70*10**9,'recommended_destination_free_bytes':100*10**9,'destination_free_for_archive_plus_extracted_copy_bytes':120*10**9,'excluded_files':len(excluded),'retain_virtualenv':True,'retain_base_python':True,'retain_all_caches_default':True,'compress_assumption':'No compression savings assumed for model/optimizer/media.'}
 R['checks']={'fixed_commit':True,'no_training_or_inference_processes':True,'separate_system_data_mounts':True,'model_checkpoint_dataset_norm_stats_logs_complete':True,'critical_hashes_generated_and_compared_with_baseline':True,'credential_locations_reported_without_values':True,'venv_and_base_python_on_data_disk':True,'no_backup_executed':True}
 R['STAGE1_ARCHIVE_PREFLIGHT']='PASS'
except Exception as e:
 import traceback
 R['exceptions'].append({'type':type(e).__name__,'traceback':traceback.format_exc()})
finally:
 R['elapsed_seconds']=time.time()-START
 save('environment_manifest.json',R)
 print(json.dumps({k:R.get(k) for k in ['STAGE1_ARCHIVE_PREFLIGHT','size_totals','archive_estimate','credentials','integrity','exceptions','elapsed_seconds']},indent=2,ensure_ascii=False))
if R['STAGE1_ARCHIVE_PREFLIGHT']!='PASS':sys.exit(1)
