from pathlib import Path
import json,subprocess,hashlib,numpy as np,time,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
OUT=Path(sys.argv[1]);PREP=Path(sys.argv[2]);solver=Path(sys.argv[3]);expected=sys.argv[4];OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
assert sha(solver)==expected
def run(args,log,slot=True):
 cmd=([PY,'/tmp/numi-human-geometry-slot.py']if slot else[])+list(map(str,args));st=time.monotonic()
 with Path(log).open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 write(str(log)+'.execution.json',dict(command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-st,log_sha256=sha(log)))
 assert r.returncode==0,(log,r.returncode)
prep=read(PREP);state=prep['state'];raw=Path(prep['raw']);assert sha(raw)==prep['raw_sha256']
allpairs={tuple(map(int,r))for r in np.load(prep['pairs'])}
# Recompute all pair constraints from the current base; these old witnesses add only construction guides.
for pp in [H/'whole-witness-local-210-run001/all-learned-self-pairs.npy',*sorted((H/'whole-witness-halfstep-215-run001').glob('pass-*/audit-130/surface-self-pairs.npy'))]:
 allpairs.update(tuple(map(int,r))for r in np.load(pp))
write(OUT/'input-pins.json',dict(prepared_path=str(PREP),prepared_sha256=sha(PREP),solver=str(solver),solver_sha256=sha(solver),state=state,native_admission=False))
for i in range(4):
 rd=OUT/f'joint-{i+1:03d}';rd.mkdir();ps=rd/'self-pairs.npy';np.save(ps,np.asarray(sorted(allpairs),np.int64).reshape(-1,2))
 run([PY,solver,'--base-candidate',state['candidate'],'--raw-trial',raw,'--self-pairs',ps,'--source-base-npz',D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz','--area-reference-npz',D/'full-connected-clearance-150/run-037-local-refinement-006-prep/run-001/refined-candidate.npz','--output',rd/'proposal'],rd/'fit.log')
 candidate=rd/'proposal/candidate.npz'
 if not candidate.exists():print('NO_CANDIDATE',rd,flush=True);break
 helper=D/'D/local-witness-backtracking-183/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py'
 cfg=dict(state=state,trial=str(candidate),out=str(rd/'local-backtrack'),parent_fit=prep['parent_fit'],helper=str(helper),helper_sha256=sha(helper),helper_entry='reduce_surface_witness_node_scales_183',initial_scale=1.,zero_below=1/16)
 write(rd/'local-config.json',cfg)
 run([PY,H/'run_local_witness_backtracking_211.py',rd/'local-config.json'],rd/'local.log',False)
 selected=rd/'local-backtrack/selected-result.json'
 if selected.exists():
  result=read(selected);write(OUT/'selected-result.json',result);print(json.dumps(dict(iteration=i+1,selected=result)),flush=True);break
 old=len(allpairs)
 for pp in sorted((rd/'local-backtrack').glob('pass-*/audit-130/surface-self-pairs.npy')):
  allpairs.update(tuple(map(int,r))for r in np.load(pp))
 print(json.dumps(dict(iteration=i+1,learned=len(allpairs),new=len(allpairs)-old)),flush=True)
 if len(allpairs)==old:break
write(OUT/'completed.json',dict(status='BOUNDED_CONNECTED_SOLVE_FINISHED',native_admission=False,accepted_physical_steps=0))
