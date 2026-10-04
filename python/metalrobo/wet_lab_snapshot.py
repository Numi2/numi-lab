"""Consistent filesystem snapshots for the existing local workspace.

Sealed records are copied byte-for-byte. Relocations live in a separate map.
No worker is restarted and no session token is exported.
"""
import hashlib,json,os,re,shutil,tarfile,time,uuid
from pathlib import Path
from wet_lab_shared import read,write

HOME=Path.home()/'.numi/laboratories'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def inventory(root):
 result={}
 for p in sorted(Path(root).rglob('*')):
  if p.is_symlink():raise ValueError('Snapshot rejects symlink: '+str(p))
  if p.is_file():result[str(p.relative_to(root))]=digest(p)
 return result

def snapshot(store,expected_revision):
 store.reconcile()
 with store.lock():
  state=read(store.path)
  if state['revision']!=expected_revision:raise ValueError('Workspace changed; read context before snapshot')
  if any(store.alive(op) for op in state['operations']):raise ValueError('Stop or finish the live operation before snapshot; changing output cannot be captured')
  identifier=uuid.uuid4().hex;dest=HOME/'snapshots'/identifier;dest.mkdir(parents=True);os.chmod(dest,0o700)
  try:
   shutil.copytree(store.root,dest/'workspace',ignore=shutil.ignore_patterns('workspace.lock','*.tmp'))
   # Retain executable scientific and service owners independently of disposable checkouts.
   owners=dest/'owners';shutil.copytree(Path(store.config['vivoRoot'])/'Tools/VirtualWetLab',owners/'vivo/Tools/VirtualWetLab',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
   service=Path(__file__).parent;lab=owners/'lab';lab.mkdir(parents=True)
   for name in ('virtual_wet_lab.py','wet_lab_shared.py','wet_lab_snapshot.py','wet_lab_cli.py'):shutil.copy2(service/name,lab/name)
   shutil.copytree(service/'wet_lab',lab/'wet_lab',ignore=shutil.ignore_patterns('__pycache__'))
   mappings={};dependencies={};assays={}
   for aid,source in store.config['assays'].items():
    src=Path(source).resolve()
    if src.is_relative_to(store.root.resolve()):target='workspace/'+str(src.relative_to(store.root.resolve()))
    else:
     # Existing spatial artifacts retain external source/runtime paths. Bind those
     # dependencies explicitly; never claim they were made self-contained.
     target='artifacts/'+aid+'/assay.json';shutil.copytree(src.parent,dest/Path(target).parent)
    assays[aid]=target;mappings[str(src)]=target
    a=read(src)
    for v in a.get('specimens',[]):
     if v.get('source'):dependencies[v['source']]=v['sourceSHA256']
    binary=a.get('runtime',{}).get('binary')
    if binary and Path(binary).is_absolute():
     dependencies[binary]=a['runtime']['sha256']
     bundle=Path(binary).parent/'mlx-swift_Cmlx.bundle'
     if bundle.exists():
      for p,h in inventory(bundle).items():dependencies[str(bundle/p)]=h
   for path,sha in dependencies.items():
    if digest(path)!=sha:raise ValueError('External dependency changed: '+path)
   location={'assays':assays,'vivoRoot':'owners/vivo','service':'owners/lab/virtual_wet_lab.py','originalWorkspace':str(store.root),'mappings':mappings,'externalDependencies':dependencies,'binary':store.config.get('binary')}
   write(dest/'location-map.json',location)
   import importlib.metadata,platform
   environment={'platform':platform.platform(),'machine':platform.machine(),'packages':{p:importlib.metadata.version(p) for p in ('numpy','safetensors','anndata','h5py','scipy','scikit-learn')}}
   manifest={'environment':environment,'format':'numilab-workspace-snapshot/v1','id':identifier,'createdAt':time.time(),'workspaceRevision':state['revision'],'files':inventory(dest),'externalDependencies':dependencies,'credentials':'excluded; fresh service credentials required','runningOperations':'restore as interrupted; never automatically restart'}
   write(dest/'manifest.json',manifest)
  except Exception:
   # Failed staging is retained and explicitly not returned as a complete snapshot.
   write(dest/'incomplete.json',{'status':'incomplete'});raise
  return {'id':identifier,'revision':state['revision'],'manifestSHA256':digest(dest/'manifest.json'),'dependencies':dependencies,'status':'snapshot-complete'}

def verify(folder):
 folder=Path(folder);m=read(folder/'manifest.json')
 if m['format']!='numilab-workspace-snapshot/v1':raise ValueError('Unsupported snapshot')
 expected=set(m['files'])|{'manifest.json'}
 if set(inventory(folder))!=expected:raise ValueError('Snapshot contains missing or unregistered files')
 for p,h in m['files'].items():
  rel=Path(p)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid manifest path')
  if digest(folder/rel)!=h:raise ValueError('Snapshot hash mismatch: '+p)
 import importlib.metadata,platform
 if m.get('environment',{}).get('machine',platform.machine())!=platform.machine():raise ValueError('Native architecture differs from snapshot')
 for p,v in m.get('environment',{}).get('packages',{}).items():
  if importlib.metadata.version(p)!=v:raise ValueError('Snapshot Python package version differs: '+p)
 for p,h in m['externalDependencies'].items():
  if digest(p)!=h:raise ValueError('External dependency unavailable or changed: '+p)
 return m

def export(identifier):
 folder=HOME/'snapshots'/safe_id(identifier);verify(folder);out=HOME/'exports';out.mkdir(parents=True,exist_ok=True);archive=out/(identifier+'.tar.gz')
 if not archive.exists():
  temp=archive.with_suffix('.partial')
  with tarfile.open(temp,'w:gz') as tar:tar.add(folder,arcname=identifier)
  os.replace(temp,archive)
 return {'id':identifier,'archive':str(archive),'sha256':digest(archive)}

def safe_id(value):
 if not re.fullmatch('[0-9a-f]{32}',value):raise ValueError('Expected snapshot identifier')
 return value

def restore(identifier,name):
 safe_id(identifier)
 if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):raise ValueError('Use a short laboratory name')
 source=HOME/'snapshots'/identifier
 if not source.exists():
  archive=HOME/'exports'/(identifier+'.tar.gz')
  staging=HOME/'snapshots'/('.import-'+uuid.uuid4().hex);staging.mkdir(parents=True)
  with tarfile.open(archive,'r:gz') as tar:
   for member in tar.getmembers():
    path=Path(member.name)
    if path.is_absolute() or '..' in path.parts or path.parts[0]!=identifier or not (member.isfile() or member.isdir()):raise ValueError('Unsafe snapshot archive entry')
   tar.extractall(staging,filter='data')
  verify(staging/identifier);os.replace(staging/identifier,source);staging.rmdir()
 m=verify(source);dest=HOME/name
 if dest.exists():raise ValueError('Restore destination already exists')
 shutil.copytree(source,dest);location=read(dest/'location-map.json');workspace=dest/'workspace';state=read(workspace/'shared/workspace.json')
 for op in state['operations']:
  op.pop('pid',None)
  if op['status']=='running':
   op.update(status='interrupted',progress='Restored operation requires explicit recovery')
   next(d for d in state['drafts'] if d['id']==op['draftID'])['status']='interrupted'
 state['revision']+=1;state['history'].append({'action':'restored','snapshotID':identifier,'revision':state['revision'],'time':time.time()});write(workspace/'shared/workspace.json',state)
 config={'vivoRoot':str(dest/location['vivoRoot']),'binary':location.get('binary'),'assays':{k:str(dest/p) for k,p in location['assays'].items()}}
 write(workspace/'shared/service.json',config)
 # Interrupted jobs also use relocated config. Their historical registrations and
 # sealed artifacts stay immutable; recovery writes a new job from current state.
 write(dest/'restore.json',{'snapshotID':identifier,'manifestSHA256':digest(source/'manifest.json'),'workspace':str(workspace),'service':str(dest/location['service']),'configuration':config,'freshSessionRequired':True})
 return {'name':name,'snapshotID':identifier,'workspaceRevision':state['revision'],'status':'restored','freshSessionRequired':True}
