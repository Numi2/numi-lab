import argparse
from pathlib import Path
import subprocess,json,hashlib,time,importlib.util,os,sys,numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170'
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
parser=argparse.ArgumentParser(description='Existing exact-target connected-surface construction continuation; no native admission.')
parser.add_argument('--state',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--solver',type=Path,required=True);parser.add_argument('--solver-sha256',required=True)
parser.add_argument('--rounds',type=int,default=4);parser.add_argument('--alpha',type=float,choices=[0.5,1.0],default=0.5)
parser.add_argument('--max-edge-gradient',type=float,default=0.0);parser.add_argument('--slack-tolerance',type=float,default=.05)
parser.add_argument('--face-builder',type=Path,required=True);parser.add_argument('--face-builder-sha256',required=True)
parser.add_argument('--first-target-dir',type=Path)
parser.add_argument('--time-limit-s',type=float,default=60)
parser.add_argument('--attempts',type=int,default=2)
parser.add_argument('--face-direction',choices=['nearest-boundary','source-outward','outward-nearest'],default='source-outward')
parser.add_argument('--learned-root',type=Path,action='append',default=[])
args=parser.parse_args();assert 1<=args.attempts<=4 and 1<=args.rounds<=12 and 0<=args.slack_tolerance<=.1 and args.max_edge_gradient>=0
OUT=args.out;OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def run(cmd,log,slot=True):
 cmd=([PY,'/tmp/numi-human-geometry-slot.py']if slot else[])+list(map(str,cmd));st=time.monotonic()
 with Path(log).open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 write(str(log)+'.execution.json',dict(command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-st,log_sha256=sha(log)));assert r.returncode==0,(log,r.returncode)
sp=importlib.util.spec_from_file_location('prep185',H/'prepare_refined_proposal_185.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
state=args.state
pairs=set(tuple(map(int,x))for x in np.load(H/'joint-minimum-movement-229-run001/joint-001/self-pairs.npy'))
for pp in sorted((H/'joint-minimum-movement-229-run001').glob('joint-*/local-backtrack/pass-*/audit-130/surface-self-pairs.npy')):
 pairs.update(tuple(map(int,x))for x in np.load(pp))
for pp in sorted((H/'joint-clearance-continuation-232-run001').glob('round-*/joint/joint-*/local-backtrack/pass-*/audit-130/surface-self-pairs.npy')):
 pairs.update(tuple(map(int,x))for x in np.load(pp))
for root in args.learned_root:
 for pp in sorted(root.rglob('surface-self-pairs.npy')):
  pairs.update(tuple(map(int,x))for x in np.load(pp))
template=(H/'run_joint_minimum_movement_229.py').read_text().replace('for i in range(1):','for i in range('+str(args.attempts)+'):')
template=template.replace('.5*delta',repr(args.alpha)+'*delta').replace('alpha=.5','alpha='+repr(args.alpha)).replace("extra=json.loads(sys.argv[3])+['--slack-tolerance','0.05']","extra=json.loads(sys.argv[3])")
solver=args.solver
assert sha(solver)==args.solver_sha256
assert sha(args.face_builder)==args.face_builder_sha256
assert sha(H/'run_local_witness_backtracking_262.py')=='e856876f75a28f9bf1f11c3e5bd953a235cae053778a613f8c5d6042eb2247f1'
template=template.replace('initial_scale=1.,zero_below=1/16)','initial_scale=1.,zero_below=1/16,maximum_exact_audits=16,maximum_construction_passes=64)')
template=template.replace('run_local_witness_backtracking_211.py','run_local_witness_backtracking_262.py').replace("(rd/'local-backtrack').glob('pass-*/audit-130/surface-self-pairs.npy')","(rd/'local-backtrack').rglob('surface-self-pairs.npy')")
emitter=D/'connected-local-witness-proposal-235/emit_empty_point_targets_235.py'
assert sha(emitter)=='4367ecca6a2506d61d83a76ede4ea6449606c1b792fbb45fab9182a4a5a1fd43'
write(OUT/'configuration.json',dict(attempts_per_round=args.attempts,face_builder=str(args.face_builder),face_builder_sha256=sha(args.face_builder),face_direction=args.face_direction,state=str(state),state_sha256=sha(state),solver=str(solver),solver_sha256=sha(solver),alpha=args.alpha,max_edge_gradient=args.max_edge_gradient,slack_tolerance=args.slack_tolerance,rounds=args.rounds,emitter_sha256=sha(emitter),native_admission=False,accepted_physical_steps=0))
for i in range(args.rounds):
 free=os.statvfs('/').f_bavail*os.statvfs('/').f_frsize
 if free<200*1024**2:
  write(OUT/'paused.json',dict(status='DISK_HEADROOM_PAUSE',round=i+1,state=str(state),free_bytes=free));break
 rd=OUT/f'round-{i+1:03d}';rd.mkdir();p=rd/'prep';p.mkdir();s=read(state);candidate=Path(s['candidate'])
 if read(s['point_summary'])['inside_nodes']==0 and s.get('joint_merit',{}).get('after_surface_pairs')==0:
  write(rd/'result.json',dict(status='ZERO_REST_GEOMETRY_WITNESSES_ONLY_NATIVE_NOT_ADMITTED'));break
 write(p/'input-state.json',s)
 if i==0 and args.first_target_dir is not None:
  import shutil
  tr=read(args.first_target_dir/'report.json')
  assert tr['inputs']['candidate_sha256']==sha(candidate) and tr['runner_sha256']==sha(args.face_builder) and tr['direction_mode']==args.face_direction
  shutil.copytree(args.first_target_dir,p/'targets')
  write(rd/'face-target-reuse.json',dict(source=str(args.first_target_dir),report_sha256=sha(args.first_target_dir/'report.json'),npz_sha256=sha(args.first_target_dir/'targets.npz')))
 else:
  run([PY,args.face_builder,'--candidate',candidate,'--surface-report',s['surface_report'],'--output-dir',p/'targets', '--direction-mode',args.face_direction],rd/'face-targets.log')
 point=Path(s['point_summary']);ip=read(point)
 query=point.parent/'query';target=p/'point-targets';target.mkdir()
 if ip['inside_nodes']>0:
  expected=dict(candidate=sha(candidate),**{'query_'+k:sha(query/(k+'.csv'))for k in('points','pairs','nodes')})
  targetpy=target/'build_targets.py'
  m.emit(m.TARGET,targetpy,dict(CAND=candidate,QUERY=query,OUT=target,RPT=target/'nearest-union-exits.json',NPZ=target/'nearest-union-exits.npz',EXPECTED_QUERY_HASHES=expected),'target')
  run([PY,targetpy],rd/'point-targets.log')
 else:
  run([PY,emitter,'--candidate',candidate,'--summary',point,'--query-dir',query,'--npz',target/'nearest-union-exits.npz','--report',target/'nearest-union-exits.json'],rd/'point-targets.log')
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
 extra+=['--slack-tolerance',str(args.slack_tolerance),'--time-limit-s',str(args.time_limit_s)]
 if args.max_edge_gradient>0:extra+=['--max-edge-gradient',str(args.max_edge_gradient)]
 run([PY,runner,solver,sha(solver),json.dumps(extra)],rd/'joint.log',False)
 selected=out/'selected-result.json'
 if not selected.exists():
  write(rd/'result.json',dict(status='NO_COMBINED_PROGRESS',native_admission=False,accepted_physical_steps=0));break
 state=selected;r=read(state);write(OUT/'selected-result.json',r)
 print(json.dumps(dict(round=i+1,joint_merit=r.get('joint_merit'),depth_merit=r.get('depth_merit'))),flush=True)
 for pp in sorted(out.rglob('surface-self-pairs.npy')):
  pairs.update(tuple(map(int,x))for x in np.load(pp))
write(OUT/'completed.json',dict(status='BOUNDED_CONTINUATION_FINISHED',final_state=str(state),native_admission=False,accepted_physical_steps=0))
