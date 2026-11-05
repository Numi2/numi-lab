from pathlib import Path
import json,subprocess,os,hashlib,time,shutil
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');C=W/'joint-boundary-rebuild-64c/compiler'
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
B=H/'regional-compensation-1061-root'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
support=json.loads((C/'mass-support/receipt.json').read_text())
for e in support['outputs'].values():assert sha(e['path'])==e['sha256']
old=json.loads((B/'full-allocation-003/run-manifest.json').read_text());argv=old['argv'];cmdstart=next(i for i,x in enumerate(argv) if x.endswith('/mass-cli'));cmd=argv[cmdstart:]
env=dict(x.split('=',1) for x in argv[1:cmdstart])
changes={}
for role in ['skin','fat']:
 R=role.upper()
 for key,path in [('P1_'+R+'_NPZ',C/f'p1-{role.title()}/{role}-current-p1-143-support.npz'),('CARRIERS_'+R+'_NPZ',C/f'mass-support/{role}-source-carriers.npz'),('CARRIER_WITNESS_'+R+'_CSV',C/f'mass-support/{role}-carrier-witnesses.csv.gz')]:
  changes['NUMI_H706_FRACTIONAL_'+key]=str(path)
  changes['NUMI_H706_FRACTIONAL_'+key.rsplit('_',1)[0]+'_SHA256']=sha(path)
 for key in ['GEOMETRY_'+R+'_SHA256','VOLUME_'+R+'_SHA256']:changes['NUMI_H706_FRACTIONAL_'+key]=sha(C/f'canonical/{role}-canonical-parent-union.npz')
for stem,filename in [('EXACT_PROJECTION','exact-projection-support.csv'),('SOURCE_CENTROID','source-centroid-slots.csv')]:
 changes['NUMI_H706_FRACTIONAL_'+stem+'_CSV']=str(C/'mass-support'/filename);changes['NUMI_H706_FRACTIONAL_'+stem+'_SHA256']=sha(C/'mass-support'/filename)
changes['NUMI_H706_FRACTIONAL_STATE_SHA256']=sha(C/'volume-prefix/verified-state.json')
changes['NUMI_H706_FRACTIONAL_OUTPUT_DIR']=str(C/'mass-allocation/allocator-output')
for flag,file in [('--child-node-file','mesh.node'),('--child-ele-file','mesh.ele'),('--child-labels-csv','child.labels.csv')]:cmd[cmd.index(flag)+1]=str(C/'volume-export/volume'/file)
out=C/'mass-allocation';out.mkdir(exist_ok=False)
config={'objective':'Conservative physical skin/fat transfer of candidate64c','controls':'Source mass/quadrature weights/barycentric sites/density bounds/transport radius unchanged; receiver geometry and witness references rebuilt','source_manifest':pin(B/'full-allocation-003/run-manifest.json'),'support_receipt':pin(C/'mass-support/receipt.json'),'current_state':pin(C/'volume-prefix/verified-state.json'),'command':cmd,'environment_changes':changes,'qualification':'Preparation only; complete native scene and comparison remain required'}
(out/'run-config.json').write_text(json.dumps(config,indent=2)+'\n');changes['NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256']=sha(out/'run-config.json');env.update(changes)
(out/'run-manifest.json').write_text(json.dumps({'argv':['/usr/bin/env',*[k+'='+v for k,v in sorted(env.items())],*cmd],'binary':pin(cmd[0]),'source_manifest':config['source_manifest'],'current_run_config':pin(out/'run-config.json'),'physical_admission':False,'native_steps':0},indent=2)+'\n')
assert shutil.disk_usage(C).free>=12*2**30,'restore 12 GiB before allocation/native'
start=time.time()
with (out/'stdout.log').open('w') as o,(out/'stderr.log').open('w') as e:r=subprocess.run(['/usr/bin/time','-l',*cmd],env={**os.environ,**env},stdout=o,stderr=e)
(out/'execution.json').write_text(json.dumps({'returncode':r.returncode,'wall_s':time.time()-start,'stdout':pin(out/'stdout.log'),'stderr':pin(out/'stderr.log'),'physical_admission':False,'native_steps':0},indent=2)+'\n')
print('MASS_ALLOCATION',r.returncode,time.time()-start,flush=True)
print((out/'stdout.log').read_text()[-3500:]);print((out/'stderr.log').read_text()[-2500:])
raise SystemExit(r.returncode)
