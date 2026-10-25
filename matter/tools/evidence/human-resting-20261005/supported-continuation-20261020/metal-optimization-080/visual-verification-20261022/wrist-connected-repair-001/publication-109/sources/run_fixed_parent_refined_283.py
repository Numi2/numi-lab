from pathlib import Path
import json,subprocess,hashlib,numpy as np,time,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
ROOT=H/'connected-refined-278'
OUT=H/'fixed-parent-refined-283';OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def run(cmd,log,slot=True):
 cmd=([PY,'/tmp/numi-human-geometry-slot.py']if slot else[])+list(map(str,cmd));st=time.monotonic()
 with Path(log).open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 write(str(log)+'.execution.json',dict(command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-st,log_sha256=sha(log)))
 assert r.returncode==0,(log,r.returncode)
solver=Path(sys.argv[1]);assert sha(solver)==sys.argv[2]
state=read(ROOT/'baseline-state.json');area=ROOT/'candidate.npz'
assert sha(area)=='c2a076267c2a8261a5f1a51b742a0b4e44471d31b200f5b809b8c98d6205348f'
assert sha(H/'run_local_witness_backtracking_282.py')=='6a408325c0a65dafec3ab422bec0d08f02fbc5fed898adee2ee04ef3dac6884b'
write(OUT/'source-pins.json',dict(state=state,solver=str(solver),solver_sha256=sha(solver),matched_area_reference=str(area),area_sha256=sha(area),fixed_prefix_nodes=18430,native_admission=False,accepted_physical_steps=0))
pairs=set()
for i in range(2):
 rd=OUT/f'attempt-{i+1:03d}';rd.mkdir()
 ps=rd/'self-pairs.npy';np.save(ps,np.asarray(sorted(pairs),np.int64).reshape(-1,2))
 cmd=[PY,solver,'--base-candidate',state['candidate'],'--raw-trial',state['candidate'],'--self-pairs',ps,'--source-base-npz',D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz','--area-reference-npz',area,'--area-reference-points-key','area_reference_037_points_world_m_f64','--area-reference-faces-key','area_reference_037_source_O_faces','--fixed-prefix-nodes','18430','--point-target-npz',ROOT/'point-targets/nearest-union-exits.npz','--point-target-report',ROOT/'point-targets/nearest-union-exits.json','--face-target-npz',ROOT/'face-targets-nearest/targets.npz','--face-target-report',ROOT/'face-targets-nearest/report.json','--output',rd/'proposal','--max-edge-gradient','.2','--slack-tolerance','.05','--time-limit-s','60']
 run(cmd,rd/'fit.log')
 rawlp=rd/'proposal/raw-lp-candidate.npz'
 if not rawlp.exists():write(rd/'result.json',dict(status='NO_RAW_PROPOSAL'));break
 z=np.load(state['candidate']);a=np.load(rawlp);keys=[k for k in a.files if k.endswith('_lp_delta_mm_f64')];assert len(keys)==1
 prefix=keys[0][:-len('delta_mm_f64')];support=a[prefix+'support_rows'];delta=a[prefix+'delta_mm_f64']
 xp=z['points_world_m'].copy();xp[support]=((xp[support].astype(float)*1000.+delta)*.001).astype(np.float32)
 xp[z['typed_coarse_anchor_output_rows_018']]=z['points_world_m'][z['typed_coarse_anchor_output_rows_018']]
 assert np.array_equal(xp[:18430].view(np.uint32),z['points_world_m'][:18430].view(np.uint32))
 trial=rd/'trial.npz';np.savez_compressed(trial,**{**{k:z[k]for k in z.files},'points_world_m':xp})
 helper=D/'D/local-witness-backtracking-183/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py'
 cfg=dict(state=state,trial=str(trial),out=str(rd/'local-backtrack'),parent_fit=state['parent_fit'],helper=str(helper),helper_sha256=sha(helper),helper_entry='reduce_surface_witness_node_scales_183',initial_scale=1.,zero_below=1/32,maximum_exact_audits=8,maximum_construction_passes=24,area_reference_npz=str(area),area_reference_sha256=sha(area),area_reference_points_key='area_reference_037_points_world_m_f64',area_reference_faces_key='area_reference_037_source_O_faces',fixed_prefix_nodes=18430)
 write(rd/'config.json',cfg)
 run([PY,H/'run_local_witness_backtracking_282.py',rd/'config.json'],rd/'local.log',False)
 selected=rd/'local-backtrack/selected-result.json'
 if selected.exists():
  write(OUT/'selected-result.json',read(selected));break
 old=len(pairs)
 for f in (rd/'local-backtrack').rglob('surface-self-pairs.npy'):pairs.update(tuple(map(int,x))for x in np.load(f))
 if len(pairs)==old:break
write(OUT/'completed.json',dict(status='BOUNDED_FIXED_PARENT_REFINEMENT_CONSTRUCTION_FINISHED',native_admission=False,accepted_physical_steps=0))
