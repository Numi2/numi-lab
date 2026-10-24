#!/usr/bin/env python3
from pathlib import Path
import numpy as np,json,hashlib,ast,collections,time
from scipy.spatial import ConvexHull,QhullError,cKDTree
D=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse')
O=D/'local-quality-cavities-398';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((O/'result.json').read_text());assert r['geometry_stage_pass']
p395=D/'p1-interior-face-support-395/refined-background-checkpoint.npz'
assert sha(p395)==r['input_sha256']
p398=O/'candidate-geometry-only.npz';assert sha(p398)==r['candidate_sha256']
a=np.load(p395,allow_pickle=False);b=np.load(p398,allow_pickle=False)
X=a['points_world_m'].astype(np.float32);assert np.array_equal(X,b['points_world_m'])
T=a['tetrahedra'];rows=b['replaced_395_tet_rows'];oldT=T[rows];newT=b['new_local_tets']
owner=D/'build_014-replay.py';assert sha(owner)==r['owner014_sha256']
ns=dict(np=np,collections=collections,ConvexHull=ConvexHull,QhullError=QhullError)
tree=ast.parse(owner.read_text());names={'dets','volume','quality','tet_intersection_volume'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(owner),'exec'),ns)
globals().update({k:ns[k] for k in names})
ox=X[oldT].astype(float);nx=X[newT].astype(float)
oldvol=dets(X,oldT)/6;newvol=dets(X,newT)/6
ct=ox.mean(1);rad=np.linalg.norm(ox-ct[:,None,:],axis=2).max(1);tree=cKDTree(ct);maxrad=rad.max()
rr=[];cc=[];vv=[];t=time.monotonic();pairs=0
for i,A in enumerate(nx):
 ctr=A.mean(0);r0=np.linalg.norm(A-ctr,axis=1).max()
 for j in sorted(tree.query_ball_point(ctr,r0+maxrad+1e-12)):
  B=ox[j]
  if np.any(B.max(0)<A.min(0)-1e-12) or np.any(B.min(0)>A.max(0)+1e-12):continue
  pairs+=1;v=tet_intersection_volume(A,B)
  if v>max(1e-24,oldvol[j]*1e-14):rr.append(i);cc.append(j);vv.append(v)
 if i%50==0:print(json.dumps({'completed_new_tets':i,'all_new_tets':len(nx),'candidate_pairs':pairs,'nonzeros':len(vv),'elapsed_s':time.monotonic()-t}),flush=True)
rr=np.array(rr,np.int64);cc=np.array(cc,np.int64);vv=np.array(vv,np.float64)
rs=np.bincount(rr,weights=vv,minlength=len(newT));cs=np.bincount(cc,weights=vv,minlength=len(oldT))
rcov=rs/newvol;pcov=cs/oldvol
sort=np.lexsort((cc,rr));rr=rr[sort];cc=cc[sort];vv=vv[sort]
childfrac=vv/rs[rr];parentfrac=vv/cs[cc]
counts=np.bincount(rr,minlength=len(newT));offs=np.r_[0,np.cumsum(counts)]
rng=np.random.default_rng(398);oldmass=rng.uniform(.1,2,len(oldT))*oldvol
newmass=np.bincount(rr,weights=parentfrac*oldmass[cc],minlength=len(newT))
massres=abs(newmass.sum()-oldmass.sum())/oldmass.sum()
# An interface node keeps exactly its existing coordinate and source embedding.
# No new nodes were introduced; old coordinate rows removed from connectivity
# stay explicitly unreferenced until the consuming compiler compacts them.
active395=np.unique(T);active398=np.unique(b['tetrahedra'])
unused_new=np.setdiff1d(active395,active398)
report={'input395_sha256':sha(p395),'geometry398_sha256':sha(p398),'overlap_owner014_sha256':sha(owner),'algorithm':'retained014 clipped-polyhedron numerical geometric overlap; no exact-predicate distance claim','elapsed_s':time.monotonic()-t,'local_old_tets':len(oldT),'local_new_tets':len(newT),'candidate_pairs':pairs,'overlap_nonzeros':len(vv),'child_coverage_min_max':[float(rcov.min()),float(rcov.max())],'parent_coverage_min_max':[float(pcov.min()),float(pcov.max())],'parent_coverage_max_abs_error':float(np.max(np.abs(pcov-1))),'child_coverage_max_abs_error':float(np.max(np.abs(rcov-1))),'every_parent_contributes':bool(np.all(cs>0)),'every_child_covered':bool(np.all(rs>0)),'synthetic_mass_conservation_relative_error':float(massres),'synthetic_mass_note':'numerical transfer check using deterministic synthetic density; no absolute kg ledger exists in395 or014','newly_unreferenced_coordinate_ids':unused_new.tolist(),'coordinate_prefix_and_node_embeddings_unchanged':True,'field_values_transfer':'same XYZ row IDs exactly; tet tables and all constraints must be regenerated','native_admission':False}
report['geometric_transfer_gate_pass']=bool(np.all(rs>0) and np.all(cs>0) and np.max(abs(pcov-1))<2e-5 and np.max(abs(rcov-1))<2e-5 and massres<1e-12)
data={k:b[k] for k in b.files}
data.update(node_source_indptr=a['node_source_indptr'],node_source_indices=a['node_source_indices'],node_source_weights=a['node_source_weights'],physical_outer_node_ids=a['physical_outer_node_ids'],overlap_child_offsets=offs,overlap_parent_395_tet_row=rows[cc],overlap_intersection_volume_m3=vv,overlap_child_normalized_fraction=childfrac,overlap_parent_conservative_fraction=parentfrac,overlap_child_coverage_ratio=rcov,overlap_parent_coverage_ratio=pcov,overlap_parent_395_input_tet_rows=rows,overlap_parent_014_input_tet_row=a['child_source_tetrahedron_rows'][rows[cc]],newly_unreferenced_coordinate_ids=unused_new)
np.savez_compressed(O/'candidate-with-transfer.npz',**data)
report['output_sha256']=sha(O/'candidate-with-transfer.npz')
(O/'transfer-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if not report['geometric_transfer_gate_pass']:raise SystemExit(2)
