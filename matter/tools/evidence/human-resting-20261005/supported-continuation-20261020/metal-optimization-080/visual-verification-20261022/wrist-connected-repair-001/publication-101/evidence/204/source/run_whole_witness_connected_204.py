from pathlib import Path
import json,subprocess,hashlib,importlib.util,numpy as np,time,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
OUT=H/'whole-witness-connected-204-run001';OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
def run(args,log,slot=True):
 cmd=([PY,'/tmp/numi-human-geometry-slot.py']if slot else[])+list(map(str,args));st=time.monotonic()
 with Path(log).open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 write(str(log)+'.execution.json',dict(command=cmd,returncode=r.returncode,wall_seconds=time.monotonic()-st,log_sha256=sha(log)))
 assert r.returncode==0,(log,r.returncode)
state=read(H/'joint-clusters-199-run001/selected-result.json')
target=OUT/'targets'
run([PY,D/'H/nearest-face-union-190/build_nearest_face_union_targets_198.py','--candidate',state['candidate'],'--surface-report',state['surface_report'],'--output-dir',target],OUT/'targets.log')
src=H/'refined-point-face-193-template/point_driver.py';td=H/'refined-point-face-203-template';td.mkdir(exist_ok=False)
s=src.read_text().replace("emitted_sample=np.asarray(B@Xmm).astype(np.float32)/1000","emitted_sample=np.asarray(B@Xmm)/1000").replace("source_sample=np.asarray(bt['inside_sample_world_f32_m'],np.float32)","source_sample=np.asarray(bt['inside_sample_world_m'],np.float64)").replace("<3e-7, 'barycentric sample mismatch'","<1e-12, 'barycentric sample mismatch'")
(td/'point_driver.py').write_text(s)
for n in ['area_scp_support_023.py','self_separation_026.py']:shutil.copy2(src.with_name(n),td/n)
spec=importlib.util.spec_from_file_location('prepare_185',H/'prepare_refined_proposal_185.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.DRIVER=td/'point_driver.py';emit0=m.emit
bn=target/'targets.npz';bj=target/'report.json'
def emit(t,d,ov,k):
 if k=='driver':ov={**ov,'BARY_TARGET_NPZ':bn,'BARY_TARGET_JSON':bj,'PIN':{**ov['PIN'],'bary_npz':sha(bn),'bary_json':sha(bj)}}
 return emit0(t,d,ov,k)
m.emit=emit
m.prepare(Path(state['candidate']),Path(state['point_summary']),Path(state['surface_report']),Path(state['parent_fit']),OUT/'proposal',True)
raw=OUT/'proposal/fit/attempt-001/scp-01-alpha-1p0/candidate.npz';assert raw.exists()
run([PY,D/'audit_connected_outer_geometry_130.py',raw,OUT/'raw-audit130'],OUT/'raw-audit130.log')
pairs=np.load(OUT/'raw-audit130/surface-self-pairs.npy').astype(np.int64);allpairs={tuple(map(int,p))for p in pairs}
assert len(allpairs),'raw proposal has no threatened self pairs; explicit alternate admission required'
spec=importlib.util.spec_from_file_location('strict_owner',D/'publication-093/canonical/matter/tools/human_skin_fat_bone_audit.py');owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
for i in range(4):
 rd=OUT/f'joint-{i+1:03d}';rd.mkdir()
 ps=rd/'self-pairs.npy';np.save(ps,np.asarray(sorted(allpairs),np.int64).reshape(-1,2))
 run([PY,D/'D/connected-local-witness-proposal-199/propose_all_witness_clusters_199.py','--base-candidate',state['candidate'],'--raw-trial',raw,'--self-pairs',ps,'--source-base-npz',D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz','--output',rd/'proposal'],rd/'fit.log')
 candidate=rd/'proposal/candidate.npz'
 run([PY,D/'audit_connected_outer_geometry_130.py',candidate,rd/'audit130'],rd/'audit130.log')
 after=read(rd/'audit130/summary.json')
 if after['exact_PLC_surface_pairs']:
  pp=np.load(rd/'audit130/surface-self-pairs.npy');old=len(allpairs);allpairs.update(tuple(map(int,p))for p in pp)
  write(rd/'result.json',dict(status='SELF_WITNESSES_LEARNED',self_pairs=len(pp),learned_pair_count=len(allpairs),native_admission=False))
  print(json.dumps(dict(iteration=i+1,self_pairs=len(pp),all_learned_pairs=len(allpairs))),flush=True)
  if len(allpairs)==old:break
  continue
 run([PY,D/'audit_skin_bone_surfaces_159.py',candidate,rd/'audit159','--max-seconds','600'],rd/'audit159.log')
 merit=owner.assess_joint_clearance_construction_progress(read(state['point_summary']),after,read(state['surface_report']),read(rd/'audit159/report.json'))
 result=dict(status='CONSTRUCTION_PROGRESS_ONLY'if merit['construction_progress']else'NO_COMBINED_PROGRESS',candidate=str(candidate),candidate_sha256=sha(candidate),point_summary=str(rd/'audit130/summary.json'),surface_report=str(rd/'audit159/report.json'),parent_fit=str(OUT/'proposal/fit/attempt-001/fit-report.json'),joint_merit=merit,native_admission=False,accepted_physical_steps=0)
 write(rd/'result.json',result);print(json.dumps(merit),flush=True)
 if merit['construction_progress']:write(OUT/'selected-result.json',result)
 break
write(OUT/'completed.json',dict(status='BOUNDED_WHOLE_WITNESS_ROUND_FINISHED',native_admission=False,accepted_physical_steps=0))
