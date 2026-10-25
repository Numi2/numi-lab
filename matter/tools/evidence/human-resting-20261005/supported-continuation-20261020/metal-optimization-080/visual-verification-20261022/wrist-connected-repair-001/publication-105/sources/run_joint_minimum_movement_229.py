from pathlib import Path
import json,subprocess,hashlib,numpy as np,time,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
OUT=H/'joint-minimum-movement-229-run001';OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def run(args,log,slot=True):
 cmd=([PY,'/tmp/numi-human-geometry-slot.py']if slot else[])+list(map(str,args));st=time.monotonic()
 with Path(log).open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 write(str(log)+'.execution.json',dict(command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-st,log_sha256=sha(log)))
 assert r.returncode==0,(log,r.returncode)
prepdir=H/'whole-connected-prep-225-run001';prep=read(prepdir/'prepared.json');state=prep['state'];raw=Path(state['candidate']) # Controlled variant: stage2 minimizes total displacement, stage1 unchanged.
allpairs={tuple(map(int,r))for r in np.load(prep['pairs'])}
for pp in [H/'whole-witness-local-210-run001/all-learned-self-pairs.npy',*sorted((H/'whole-connected-halfstep-223-run001').glob('pass-*/audit-130/surface-self-pairs.npy'))]:
 allpairs.update(tuple(map(int,r))for r in np.load(pp))
solver=Path(sys.argv[1]);assert sha(solver)==sys.argv[2]
write(OUT/'input-pins.json',dict(prepared=str(prepdir/'prepared.json'),prepared_sha256=sha(prepdir/'prepared.json'),solver=str(solver),solver_sha256=sha(solver),state=state,native_admission=False,accepted_physical_steps=0))
for i in range(1):
 rd=OUT/f'joint-{i+1:03d}';rd.mkdir();ps=rd/'self-pairs.npy';np.save(ps,np.asarray(sorted(allpairs),np.int64).reshape(-1,2))
 # Caller supplies the exact target CLI names after builder self-tests freeze.
 extra=json.loads(sys.argv[3])+['--slack-tolerance','0.05']
 run([PY,solver,'--base-candidate',state['candidate'],'--raw-trial',raw,'--self-pairs',ps,'--source-base-npz',D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz','--area-reference-npz',D/'full-connected-clearance-150/run-037-local-refinement-006-prep/run-001/refined-candidate.npz','--output',rd/'proposal',*extra],rd/'fit.log')
 rawlp=rd/'proposal/raw-lp-candidate.npz'
 if not rawlp.exists():print('NO_RAW_LP_CANDIDATE',rd,flush=True);break
 z=np.load(state['candidate']);a=np.load(rawlp)
 keys=[k for k in a.files if k.endswith('_lp_delta_mm_f64')];assert len(keys)==1,keys
 prefix=keys[0][:-len('delta_mm_f64')];support=a[prefix+'support_rows'];delta=a[prefix+'delta_mm_f64']
 xp=z['points_world_m'].copy();xp[support]=((xp[support].astype(np.float64)*1000.+.5*delta)*.001).astype(np.float32)
 anchors=z['typed_coarse_anchor_output_rows_018'];xp[anchors]=z['points_world_m'][anchors]
 candidate=rd/'connected-halfstep.npz';np.savez_compressed(candidate,**{**{k:z[k]for k in z.files},'points_world_m':xp})
 write(rd/'halfstep-provenance.json',dict(base_sha256=sha(state['candidate']),raw_lp_sha256=sha(rawlp),alpha=.5,candidate_sha256=sha(candidate),native_admission=False,accepted_physical_steps=0))
 helper=D/'D/local-witness-backtracking-183/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py'
 cfg=dict(state=state,trial=str(candidate),out=str(rd/'local-backtrack'),parent_fit=prep['parent_fit'],helper=str(helper),helper_sha256=sha(helper),helper_entry='reduce_surface_witness_node_scales_183',initial_scale=1.,zero_below=1/16)
 write(rd/'local-config.json',cfg)
 run([PY,H/'run_local_witness_backtracking_211.py',rd/'local-config.json'],rd/'local.log',False)
 selected=rd/'local-backtrack/selected-result.json'
 if selected.exists():
  result=read(selected);write(OUT/'selected-result.json',result);print(json.dumps(dict(iteration=i+1,joint_merit=result['joint_merit'],depth_merit=result['depth_merit'])),flush=True);break
 old=len(allpairs)
 for pp in sorted((rd/'local-backtrack').glob('pass-*/audit-130/surface-self-pairs.npy')):
  allpairs.update(tuple(map(int,r))for r in np.load(pp))
 print(json.dumps(dict(iteration=i+1,learned=len(allpairs),new=len(allpairs)-old)),flush=True)
 if len(allpairs)==old:break
write(OUT/'completed.json',dict(status='BOUNDED_DIRECT_CLEARANCE_FINISHED',native_admission=False,accepted_physical_steps=0))
