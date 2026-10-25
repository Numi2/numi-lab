from pathlib import Path
import subprocess,json,hashlib,time,importlib.util,os,sys,numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170'
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
OUT=H/'joint-clearance-continuation-234-run001';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def run(cmd,log,slot=True):
 cmd=([PY,'/tmp/numi-human-geometry-slot.py']if slot else[])+list(map(str,cmd));st=time.monotonic()
 with Path(log).open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 write(str(log)+'.execution.json',dict(command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-st,log_sha256=sha(log)));assert r.returncode==0,(log,r.returncode)
sp=importlib.util.spec_from_file_location('prep185',H/'prepare_refined_proposal_185.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
state=H/'joint-clearance-continuation-232-run001/selected-result.json'
pairs=set(tuple(map(int,x))for x in np.load(H/'joint-minimum-movement-229-run001/joint-001/self-pairs.npy'))
for pp in sorted((H/'joint-minimum-movement-229-run001').glob('joint-*/local-backtrack/pass-*/audit-130/surface-self-pairs.npy')):
 pairs.update(tuple(map(int,x))for x in np.load(pp))
for pp in sorted((H/'joint-clearance-continuation-232-run001').glob('round-*/joint/joint-*/local-backtrack/pass-*/audit-130/surface-self-pairs.npy')):
 pairs.update(tuple(map(int,x))for x in np.load(pp))
template=(H/'run_joint_minimum_movement_229.py').read_text()
solver=D/'D/connected-local-witness-proposal-222/propose_joint_clearance_witnesses_222.py'
assert sha(solver)=='13171cd5559ea3644aaea7f98bf37bbbd4298168434cdffdc98aa729dde08564'
for i in range(6):
 free=os.statvfs('/').f_bavail*os.statvfs('/').f_frsize
 if free<100*1024**2:
  write(OUT/'paused.json',dict(status='DISK_HEADROOM_PAUSE',round=i+1,state=str(state),free_bytes=free));break
 rd=OUT/f'round-{i+1:03d}';rd.mkdir();p=rd/'prep';p.mkdir();s=read(state);candidate=Path(s['candidate'])
 write(p/'input-state.json',s)
 run([PY,D/'H/nearest-face-union-190/build_nearest_face_union_targets_198.py','--candidate',candidate,'--surface-report',s['surface_report'],'--output-dir',p/'targets'],rd/'face-targets.log')
 point=Path(s['point_summary']);ip=read(point);assert ip['inside_nodes']>0,'surface-only target preparation required'
 query=point.parent/'query';target=p/'point-targets';target.mkdir()
 expected=dict(candidate=sha(candidate),**{'query_'+k:sha(query/(k+'.csv'))for k in('points','pairs','nodes')})
 targetpy=target/'build_targets.py'
 m.emit(m.TARGET,targetpy,dict(CAND=candidate,QUERY=query,OUT=target,RPT=target/'nearest-union-exits.json',NPZ=target/'nearest-union-exits.npz',EXPECTED_QUERY_HASHES=expected),'target')
 run([PY,targetpy],rd/'point-targets.log')
 tj=read(target/'nearest-union-exits.json');assert tj['status']=='PASS_FIRST_UNION_EXIT_TARGETS_ONLY'and tj['passing_targets']==ip['inside_nodes']
 ps=p/'self-pairs.npy';np.save(ps,np.asarray(sorted(pairs),np.int64).reshape(-1,2))
 write(p/'prepared.json',dict(state=s,raw=str(candidate),raw_sha256=sha(candidate),pairs=str(ps),parent_fit=s['parent_fit'],native_admission=False,accepted_physical_steps=0,preparation='existing exact targets only; unnecessary ADMM proposal and proposal-self audit omitted'))
 out=rd/'joint'
 text=template.replace("OUT=H/'joint-minimum-movement-229-run001'","OUT=Path("+repr(str(out))+")").replace("prepdir=H/'whole-connected-prep-225-run001'","prepdir=Path("+repr(str(p))+")")
 # Prepared seed already includes all inherited actual pairs; no old payloads needed.
 start=text.index("for pp in [H/'whole-witness-local-210-run001")
 end=text.index("solver=Path(sys.argv[1])",start)
 text=text[:start]+text[end:]
 runner=rd/'run_joint.py';compile(text,str(runner),'exec');runner.write_text(text)
 extra=['--point-target-npz',str(target/'nearest-union-exits.npz'),'--point-target-report',str(target/'nearest-union-exits.json'),'--face-target-npz',str(p/'targets/targets.npz'),'--face-target-report',str(p/'targets/report.json')]
 run([PY,runner,solver,sha(solver),json.dumps(extra)],rd/'joint.log',False)
 selected=out/'selected-result.json'
 if not selected.exists():
  write(rd/'result.json',dict(status='NO_COMBINED_PROGRESS',native_admission=False,accepted_physical_steps=0));break
 state=selected;r=read(state);write(OUT/'selected-result.json',r)
 print(json.dumps(dict(round=i+1,joint_merit=r.get('joint_merit'),depth_merit=r.get('depth_merit'))),flush=True)
 for pp in sorted(out.glob('joint-*/local-backtrack/pass-*/audit-130/surface-self-pairs.npy')):
  pairs.update(tuple(map(int,x))for x in np.load(pp))
write(OUT/'completed.json',dict(status='BOUNDED_CONTINUATION_FINISHED',final_state=str(state),native_admission=False,accepted_physical_steps=0))
