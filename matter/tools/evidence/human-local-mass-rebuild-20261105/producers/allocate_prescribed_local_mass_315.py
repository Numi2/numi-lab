
from pathlib import Path
import copy,gzip,hashlib,json,os,re,shlex,shutil,subprocess,threading,time
import numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');L=Path('/Users/numi/Work/current-fractional-020-publication-20261101')
C=W/'joint-boundary-rebuild-64c/compiler';B=W/'prescribed-local-mass-315';T=B/'targets-direct-005';O=B/'allocation'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
assert shutil.disk_usage(W).free>=12*2**30
assert not O.exists();O.mkdir()
receipt=json.loads((T/'receipt.json').read_text());target=Path(receipt['output']['path']);assert sha(target)==receipt['output']['sha256']
# Prove target row IDs address the exact TetGen cells the owning CLI reads.
with np.load(C/'rich433/candidate-current-typed-volume.npz') as z:x=z['points_reference_world_m_f64'];t=z['volume_tetrahedra'];m=z['material_index']
nodes=np.loadtxt(C/'volume-export/volume/mesh.node',skiprows=1)
ele=np.loadtxt(C/'volume-export/volume/mesh.ele',skiprows=1,dtype=np.int64)
assert np.array_equal(nodes[:,0],np.arange(len(x))+1) and np.array_equal(nodes[:,1:].astype(np.float32),x.astype(np.float32))
assert np.array_equal(ele[:,0],np.arange(len(t))+1)
assert np.array_equal(np.sort(ele[:,1:5]-1,axis=1),np.sort(t,axis=1)) and np.array_equal(ele[:,5],m)
save(O/'target-row-domain.json',{'status':'PASS_ALL_EMITTED_F32_NODES_AND_MATERIAL_TETRAHEDRON_ROWS','nodes':len(x),'cells':len(t),'target':pin(target),'mesh_node':pin(C/'volume-export/volume/mesh.node'),'mesh_ele':pin(C/'volume-export/volume/mesh.ele'),'scope':'row-domain check, not anatomical or physical admission'})
del x,t,m,nodes,ele
old=json.loads((C/'mass-admission/run-manifest.json').read_text());ctrl=copy.deepcopy(json.loads(Path(old['source_control_manifest']['path']).read_text()))
compile=json.loads((B/'compile.json').read_text());assert compile['exit_code']==0
tests=json.loads((B/'tests.json').read_text());assert len(tests['tests'])==9 and all(t['compile_exit']==0 and t['test_exit']==0 for t in tests['tests'])
deps=shlex.split((B/'build/mass-cli.d').read_text().replace('\\\n',' ').split(':',1)[1]);deps=list(dict.fromkeys(deps))
snapshot=B/'source-snapshot';snapshot.mkdir()
inventory=[]
for i,path in enumerate(deps):
 p=Path(path);dest=snapshot/(str(i)+'-'+p.name);shutil.copy2(p,dest);inventory.append({'compiled':pin(p),'retained':pin(dest)})
save(B/'source-snapshot.json',{'dependencies':inventory,'compile_execution':pin(B/'compile.json')})
ctrl.update(source=pin(L/'apps/NumiHumanCurrentFractionalMassMappingCLI.cpp'),binary=pin(B/'build/mass-cli'),dependencies=[pin(p) for p in deps],compile_command=compile['argv'],compile_execution=pin(B/'compile.json'),focused_controls=pin(B/'tests.json'),prescribed_child_mass={'input':pin(target),'construction':pin(T/'receipt.json'),'relative_only_tolerance':1e-10,'density_radius_source_semantic_and_closure_checks':'UNCHANGED','source_snapshot':pin(B/'source-snapshot.json')})
save(B/'source-control.json',ctrl)
argv=old['argv'];start=next(i for i,a in enumerate(argv) if a.endswith('/mass-cli'));cmd=list(argv[start:]);env=dict(a.split('=',1) for a in argv[1:start])
cmd[0]=str(B/'build/mass-cli');cmd+=['--child-mass-target-csv',str(target),'--child-mass-target-sha256',sha(target)]
env['NUMI_H706_FRACTIONAL_OUTPUT_DIR']=str(O/'allocator-output');env['NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256']=sha(B/'source-control.json')
plan={'question':'Does preserving masses of unchanged material/rest cells remove the global mass redistribution caused by fresh allocation?','prediction':'The allocator will preserve 1405911 prescribed cell masses (97.29 percent of candidate cells, 13.90335 kg) within1e-10 relative while satisfying all existing source-carrier, semantic, donor, density, radius and mass/measure closure checks. Native performance/fidelity is not predicted from compiler tests.','controls':'Same native310 geometry, materials, source mass/quadrature, body poses, source witnesses, compensation scope and transport radius. Only per-child capacities are additionally restricted outside changed cell stars. Primary-label differences in24 untouched cells remain explicit and unqualified.','stop':'900 seconds, disk below12GiB, swap growth above3GiB, or RSS above8GiB. One heavy worker. No source/native promotion on failure.','command':cmd,'source_control_manifest':pin(B/'source-control.json'),'target_row_check':pin(O/'target-row-domain.json'),'resource_policy':{'minimum_free_bytes':12*2**30,'maximum_swap_growth_bytes':3*2**30,'maximum_rss_bytes':8*2**30,'maximum_wall_s':900},'physical_admission':False,'native_steps':0}
save(O/'run-config.json',plan);env['NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256']=sha(O/'run-config.json')
run=copy.deepcopy(old);run.update(argv=['/usr/bin/env',*[k+'='+v for k,v in sorted(env.items())],*cmd],source=ctrl['source'],binary=ctrl['binary'],source_control_manifest=pin(B/'source-control.json'),source_closure=pin(B/'source-control.json'),run_config=pin(O/'run-config.json'),output_dir=env['NUMI_H706_FRACTIONAL_OUTPUT_DIR'],stdout_log=str(O/'stdout.log'),stderr_log=str(O/'stderr.log'),prescribed_child_mass=pin(target),physical_admission=False,native_steps=0)
save(O/'run-manifest.json',run)
def swap():
 text=subprocess.check_output(['sysctl','vm.swapusage'],text=True)
 return int(float(re.search(r'used = ([0-9.]+)M',text)[1])*2**20)
swap0=swap();begin=time.time();reason=None;clean={k:v for k,v in os.environ.items() if not k.startswith('NUMI_')};clean.update(env)
with (O/'stdout.log').open('wb') as out:
 p=subprocess.Popen(cmd,env=clean,stdout=out,stderr=subprocess.PIPE)
 save(O/'active.json',{'pid':p.pid,'started':begin,'command':cmd})
 def drain():
  with gzip.open(O/'stderr.log.gz','wb',compresslevel=1) as g:shutil.copyfileobj(p.stderr,g,64*1024)
 thread=threading.Thread(target=drain);thread.start();iteration=0
 with (O/'resources.jsonl').open('w') as log:
  while p.poll() is None:
   try:rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(p.pid)],text=True).strip())*1024
   except (ValueError,subprocess.CalledProcessError):rss=0
   row={'elapsed_s':time.time()-begin,'free_bytes':shutil.disk_usage(W).free,'swap_bytes':swap(),'rss_bytes':rss}
   log.write(json.dumps(row)+'\n');log.flush()
   if row['free_bytes']<12*2**30:reason='disk_below_12GiB'
   elif row['swap_bytes']-swap0>3*2**30:reason='swap_growth_above_3GiB'
   elif rss>8*2**30:reason='rss_above_8GiB'
   elif row['elapsed_s']>900:reason='wall_above_900s'
   if reason:
    p.terminate()
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill()
    break
   if iteration%15==0:print(json.dumps(row),flush=True)
   iteration+=1;time.sleep(2)
 p.wait();thread.join()
with gzip.open(O/'stderr.log.gz','rb') as g:preview=g.read(2**20+1)
if len(preview)<=2**20:(O/'stderr.log').write_bytes(preview)
else:(O/'stderr.preview.txt').write_bytes(preview[:16384])
save(O/'execution.json',{'exit_code':p.returncode,'returncode':p.returncode,'stop_reason':reason,'wall_s':time.time()-begin,'manifest_sha256':sha(O/'run-manifest.json'),'stdout':pin(O/'stdout.log'),'stderr_gzip':pin(O/'stderr.log.gz'),'physical_admission':False,'native_steps':0})
print('FINISHED',p.returncode,reason,time.time()-begin,flush=True)
print((O/'stdout.log').read_text()[-5000:])
print(preview[:3000].decode(errors='replace') if p.returncode else preview[-2000:].decode(errors='replace'))
