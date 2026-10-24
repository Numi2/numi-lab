#!/usr/bin/env python3
import ast, hashlib, json, math, time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, QhullError

BASE=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
P2=BASE/'global-inner-p1-edge-support-403/build-002/candidate.npz'
P3=BASE/'global-inner-p1-edge-support-403/quality-bistellar-003/candidate-geometry-only.npz'
P4=BASE/'global-inner-p1-edge-support-403/quality-bistellar-004/candidate-geometry-only.npz'
P5=BASE/'global-inner-p1-edge-support-403/quality-coordinate-005/candidate.npz'
P398MAP=BASE/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/material-parent-014-conservative-map.npz'
OVERLAP_SRC=BASE/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/build_014-replay.py'
OUT=BASE/'global-inner-p1-edge-support-403/quality-transfer-006'
started=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
EXPECTED={
 str(P2):'9c8c23c8229c46494d19f6fc12d6131667b901864dfab21cd1ef7de6a2fc6744',
 str(P3):'176900d57eedb40255298ad77860317ab97e31a6792bc4dc76c4d48ec7d060e8',
 str(P4):'ff882067af08d4d749d516fdf51fc8f81bb83f85e611f352e394d00c67c52932',
 str(P5):'5b54ffd34d74d717579fcb14677cbf2bfba01f4d4562652605b9207e65f8f698',
 str(P398MAP):'e4ac929347673e55082c8bf2b4fa239617c18ac297195c18fae60ce8602bf38d',
}
for p,h in EXPECTED.items():
 if sha(p)!=h: raise RuntimeError(f'input SHA mismatch {p}: {sha(p)} != {h}')
mod=ast.parse(OVERLAP_SRC.read_text())
func=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='tet_intersection_volume')
env={'np':np,'ConvexHull':ConvexHull,'QhullError':QhullError}
exec(compile(ast.Module(body=[func],type_ignores=[]),str(OVERLAP_SRC),'exec'),env)
tet_intersection_volume=env['tet_intersection_volume']
z2=np.load(P2,allow_pickle=False); z3=np.load(P3,allow_pickle=False); z4=np.load(P4,allow_pickle=False); z5=np.load(P5,allow_pickle=False)
X2=z2['points_world_m'].astype(np.float32,copy=False); T2=z2['tetrahedra'].astype(np.int64,copy=False); A2=z2['region_attributes'].astype(np.int64,copy=False)
X3=z3['points_world_m'].astype(np.float32,copy=False); T3=z3['tetrahedra'].astype(np.int64,copy=False)
X4=z4['points_world_m'].astype(np.float32,copy=False); T4=z4['tetrahedra'].astype(np.int64,copy=False)
X5=z5['points_world_m'].astype(np.float32,copy=False); T5=z5['tetrahedra'].astype(np.int64,copy=False); A5=z5['region_attributes'].astype(np.int64,copy=False)
if not (X2.shape==X3.shape==X4.shape==X5.shape and T4.shape==T5.shape and np.array_equal(T4,T5)):
 raise RuntimeError('expected shared point domain and 005/004 identical tetrahedron connectivity')
changed_nodes=np.flatnonzero(np.any(X2!=X5,axis=1)).astype(np.int64)
if changed_nodes.tolist()!=[55529,55530]: raise RuntimeError(f'unexpected changed point rows {changed_nodes.tolist()}')
# Canonical keys are built once per mesh.
def make_keys(T): return [tuple(sorted(map(int,t))) for t in T]
K2=make_keys(T2); K3=make_keys(T3); K4=make_keys(T4); K5=make_keys(T5)
if len(set(K2))!=len(K2) or len(set(K3))!=len(K3) or len(set(K4))!=len(K4) or len(set(K5))!=len(K5):
 raise RuntimeError('duplicate canonical tet keys in an input candidate')
M2={k:i for i,k in enumerate(K2)}; M3={k:i for i,k in enumerate(K3)}; M5={k:i for i,k in enumerate(K5)}
o4=z5['parent_input_candidate_offsets'].astype(np.int64); r4=z5['parent_input_candidate_rows'].astype(np.int64)
o3=z3['parent_403_candidate_offsets'].astype(np.int64); r3=z3['parent_403_candidate_rows'].astype(np.int64)
if len(o4)!=len(T5)+1 or len(o3)!=len(T3)+1 or o4[0]!=0 or o4[-1]!=len(r4) or o3[0]!=0 or o3[-1]!=len(r3):
 raise RuntimeError('malformed 005→003→002 lineage CSR')
only5=np.asarray([i for i,k in enumerate(K5) if k not in M2],np.int64)
only2=np.asarray([i for i,k in enumerate(K2) if k not in M5],np.int64)
changed_set=set(map(int,changed_nodes))
seed5=set(map(int,only5)); seed2=set(map(int,only2))
for i,t in enumerate(T5):
 if any(int(v) in changed_set for v in t): seed5.add(i)
for i,t in enumerate(T2):
 if any(int(v) in changed_set for v in t): seed2.add(i)
lineage_cache={}
def old_ancestors(i5):
 i5=int(i5)
 if i5 in lineage_cache:return lineage_cache[i5]
 olds=set()
 for i3 in r4[o4[i5]:o4[i5+1]]:
  i3=int(i3)
  if i3<0 or i3>=len(T3): raise RuntimeError('005→003 lineage index out of range')
  for i2 in r3[o3[i3]:o3[i3+1]]:
   i2=int(i2)
   if i2<0 or i2>=len(T2): raise RuntimeError('003→002 lineage index out of range')
   olds.add(i2)
 lineage_cache[i5]=tuple(sorted(olds)); return lineage_cache[i5]
# Build reverse lineage once in linear time.
old_to_new={}
for i5 in range(len(T5)):
 for i2 in old_ancestors(i5): old_to_new.setdefault(i2,[]).append(i5)
old_region=set(seed2)
for i5 in seed5: old_region.update(old_ancestors(i5))
new_region=set(seed5)
for _ in range(4):
 prior=(len(old_region),len(new_region))
 for i2 in tuple(old_region): new_region.update(old_to_new.get(i2,()))
 for i5 in tuple(new_region): old_region.update(old_ancestors(i5))
 if prior==(len(old_region),len(new_region)): break
else: raise RuntimeError('lineage region failed to stabilize')
new_region=np.asarray(sorted(new_region),np.int64); old_region=np.asarray(sorted(old_region),np.int64)
if len(new_region)>1000 or len(old_region)>1000: raise RuntimeError(f'local region exceeds 1000-tet bound: new={len(new_region)} old={len(old_region)}')
new_set=set(map(int,new_region)); old_set=set(map(int,old_region))
for i5,k in enumerate(K5):
 if i5 in new_set: continue
 i2=M2.get(k)
 if i2 is None or not np.array_equal(X5[T5[i5]],X2[T2[i2]]):
  raise RuntimeError(f'unexpected changed geometry/topology outside local region at new row {i5}')
def boundary_faces(T, rows):
 inc={}
 for ti in rows:
  tet=list(map(int,T[int(ti)]))
  for opp in range(4):
   f=tuple(tet[j] for j in range(4) if j!=opp); k=tuple(sorted(f)); inc[k]=inc.get(k,0)+1
 return {k for k,n in inc.items() if n==1}, {k:n for k,n in inc.items() if n>2}
b5,nonman5=boundary_faces(T5,new_region); b2,nonman2=boundary_faces(T2,old_region)
if nonman5 or nonman2: raise RuntimeError(f'nonmanifold cavity boundary new={len(nonman5)} old={len(nonman2)}')
boundary_delta={'new_only':len(b5-b2),'old_only':len(b2-b5),'new_examples':sorted(b5-b2)[:5],'old_examples':sorted(b2-b5)[:5]}
if b5!=b2: raise RuntimeError(f'local cavity boundary node facets differ: {boundary_delta}')
boundary_nodes=np.unique(np.asarray(list(b5),np.int64)) if b5 else np.empty(0,np.int64)
boundary_moved=boundary_nodes[np.any(X5[boundary_nodes]!=X2[boundary_nodes],axis=1)] if len(boundary_nodes) else np.empty(0,np.int64)
if len(boundary_moved):raise RuntimeError('local transfer cavity boundary coordinates moved')
def tet_volumes(X,T):
 p=X[T].astype(np.float64)
 d=np.linalg.det(np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]),axis=2))
 return np.abs(d)/6.0
V2=tet_volumes(X2,T2); V5=tet_volumes(X5,T5)
if np.any(V2<=0) or np.any(V5<=0): raise RuntimeError('nonpositive cell volume in transfer inputs')
oldX=X2[T2[old_region]].astype(np.float64); newX=X5[T5[new_region]].astype(np.float64)
oldlo=oldX.min(axis=1); oldhi=oldX.max(axis=1); newlo=newX.min(axis=1); newhi=newX.max(axis=1)
raw=[]; raw_dropped=0.0; pair_candidates=0; exact_identity_pairs=0
for j,ni in enumerate(new_region):
 mask=np.all(oldhi>=newlo[j]-1e-14,axis=1)&np.all(newhi[j]>=oldlo-1e-14,axis=1)
 js=np.flatnonzero(mask); pair_candidates+=len(js)
 for q in js:
  oi=int(old_region[q]); vv=0.0
  if K5[int(ni)]==K2[oi] and np.array_equal(X5[T5[int(ni)]],X2[T2[oi]]):
   vv=float(V5[int(ni)]); exact_identity_pairs+=1
  else: vv=tet_intersection_volume(newX[j],oldX[q])
  threshold=max(1e-30,min(float(V5[int(ni)]),float(V2[oi]))*1e-14)
  if vv>threshold:
   if A5[int(ni)]!=A2[oi]:raise RuntimeError('positive overlap crosses material identities')
   raw.append((int(ni),oi,float(vv)))
  elif vv>0: raw_dropped+=float(vv)
if not raw: raise RuntimeError('no positive local overlap volumes')
new_sum={int(i):0.0 for i in new_region}; old_sum={int(i):0.0 for i in old_region}
for ni,oi,v in raw: new_sum[ni]+=v; old_sum[oi]+=v
new_cov=np.asarray([new_sum[int(i)]/V5[int(i)] for i in new_region])
old_cov=np.asarray([old_sum[int(i)]/V2[int(i)] for i in old_region])
new_gap=np.asarray([new_sum[int(i)]-V5[int(i)] for i in new_region]); old_gap=np.asarray([old_sum[int(i)]-V2[int(i)] for i in old_region])
# Parent-normalized conservative fractions; preserve raw overlap and normalizers explicitly.
raw_by_old={int(i):[] for i in old_region}
for ni,oi,v in raw: raw_by_old[oi].append((ni,v))
B_rows={int(i):[] for i in new_region}; column_norm={}
for oi in old_region:
 oi=int(oi); cov=old_sum[oi]/V2[oi]
 if not cov>0: raise RuntimeError(f'old source cell {oi} has no positive overlap')
 column_norm[oi]=1.0/cov
 for ni,v in raw_by_old[oi]: B_rows[ni].append((oi,(v/V2[oi])*column_norm[oi]))
Boff=[0]; Bparent=[]; Bfrac=[]; changed_new_rows=[]
for i5,k in enumerate(K5):
 if i5 in new_set:
  entries=sorted(B_rows.get(i5,[]))
  if not entries: raise RuntimeError(f'new local row {i5} has no overlap parents')
  changed_new_rows.append(i5)
 else: entries=[(M2[k],1.0)]
 s=sum(v for _,v in entries)
 if not math.isfinite(s) or s<=0: raise RuntimeError(f'new→old transfer row {i5} has no finite positive attribution')
 Bparent.extend(i2 for i2,_ in entries); Bfrac.extend(v for _,v in entries); Boff.append(len(Bparent))
Boff=np.asarray(Boff,np.int64); Bparent=np.asarray(Bparent,np.int64); Bfrac=np.asarray(Bfrac,np.float64)
Aoff=z2['material_parent_child_offsets'].astype(np.int64); Aparent=z2['material_parent_014_tet_rows'].astype(np.int64); Afrac=z2['material_parent_014_conservative_fraction'].astype(np.float64)
if len(Aoff)!=len(T2)+1 or Aoff[0]!=0 or Aoff[-1]!=len(Aparent) or len(Aparent)!=len(Afrac): raise RuntimeError('002→014 CSR malformed')
Coff=[0]; Cparent=[]; Cfrac=[]
for i5 in range(len(T5)):
 accum={}
 for p2,bw in zip(Bparent[Boff[i5]:Boff[i5+1]],Bfrac[Boff[i5]:Boff[i5+1]]):
  p2=int(p2)
  for k in range(Aoff[p2],Aoff[p2+1]):
   p14=int(Aparent[k]); accum[p14]=accum.get(p14,0.0)+float(bw)*float(Afrac[k])
 for p14,v in sorted(accum.items()):
  if v>0: Cparent.append(p14); Cfrac.append(v)
 s=sum(accum.values())
 if not math.isfinite(s) or s<=0: raise RuntimeError(f'new→014 row {i5} has no finite positive attribution')
 Coff.append(len(Cparent))
Coff=np.asarray(Coff,np.int64); Cparent=np.asarray(Cparent,np.int64); Cfrac=np.asarray(Cfrac,np.float64)
old_parent_col=np.bincount(Aparent,weights=Afrac,minlength=int(Aparent.max())+1)
new_parent_col=np.bincount(Cparent,weights=Cfrac,minlength=len(old_parent_col))
if len(new_parent_col)<len(old_parent_col): new_parent_col=np.pad(new_parent_col,(0,len(old_parent_col)-len(new_parent_col)))
colerr=np.abs(new_parent_col-old_parent_col)
rng=np.random.default_rng(6006); parent_mass=rng.uniform(.2,2.0,size=len(old_parent_col))
def attributed_child_mass(offsets, parents, fractions, parent_mass):
 offsets=np.asarray(offsets);parents=np.asarray(parents);fractions=np.asarray(fractions);parent_mass=np.asarray(parent_mass)
 if offsets.ndim!=1 or offsets.size<1 or offsets[0]!=0 or offsets[-1]!=len(parents) or len(parents)!=len(fractions) or np.any(np.diff(offsets)<0):
  raise ValueError('malformed mass attribution CSR')
 if np.any(parents<0) or np.any(parents>=len(parent_mass)) or not np.isfinite(fractions).all() or np.any(fractions<=0):
  raise ValueError('invalid parent mass attribution')
 child_rows=np.repeat(np.arange(len(offsets)-1,dtype=np.int64),np.diff(offsets))
 return np.bincount(child_rows,weights=fractions*parent_mass[parents],minlength=len(offsets)-1)
old_mass=attributed_child_mass(Aoff,Aparent,Afrac,parent_mass)
new_mass=attributed_child_mass(Boff,Bparent,Bfrac,old_mass)
composed_mass=attributed_child_mass(Coff,Cparent,Cfrac,parent_mass)
Bcol=np.bincount(Bparent,weights=Bfrac,minlength=len(T2))
if np.max(np.abs(Bcol-1.0))>2e-12:raise RuntimeError('new-to-002 parent mass columns fail conservation')
if not np.allclose(new_mass,composed_mass,rtol=2e-12,atol=2e-12): raise RuntimeError('sequential/composed synthetic mass transfer mismatch')
region_summary={}
for code in sorted(set(map(int,A5))):
 s5=A5==code; s2=A2==code
 region_summary[str(code)]={'old_volume_m3':float(V2[s2].sum()),'new_volume_m3':float(V5[s5].sum()),'delta_volume_m3':float(V5[s5].sum()-V2[s2].sum()),'old_synthetic_mass':float(old_mass[s2].sum()),'new_synthetic_mass':float(new_mass[s5].sum()),'synthetic_mass_delta':float(new_mass[s5].sum()-old_mass[s2].sum())}
np.savez_compressed(OUT/'local-overlap-005-to-002.npz',new_region_rows=new_region,old_region_rows=old_region,raw_overlap_new_row=np.asarray([x[0] for x in raw],np.int64),raw_overlap_old_row=np.asarray([x[1] for x in raw],np.int64),raw_overlap_volume_m3=np.asarray([x[2] for x in raw],np.float64),new_region_raw_coverage=new_cov,old_region_raw_coverage=old_cov,new_region_raw_gap_m3=new_gap,old_region_raw_gap_m3=old_gap,old_parent_column_normalizer=np.asarray([column_norm[int(i)] for i in old_region],np.float64),transfer_new_to_002_offsets=Boff,transfer_new_to_002_parent_rows=Bparent,transfer_new_to_002_parent_fractions=Bfrac,transfer_new_to_014_offsets=Coff,transfer_new_to_014_parent_rows=Cparent,transfer_new_to_014_parent_fractions=Cfrac)
input_hashes={str(p):sha(p) for p in (P2,P3,P4,P5,P398MAP,OVERLAP_SRC)}
report={'schema':'numi.human.global-inner-p1-quality-transfer-006.v1','candidate_admission':False,'inputs':input_hashes,
 'rows':{'old_002_tets':len(T2),'new_005_tets':len(T5),'changed_node_rows':changed_nodes.tolist(),'changed_topology_new_rows':only5.tolist(),'changed_topology_old_rows':only2.tolist(),'local_new_rows':new_region.tolist(),'local_old_rows':old_region.tolist(),'local_new_count':len(new_region),'local_old_count':len(old_region),'local_boundary_face_count':len(b5),'local_boundary_topology_delta':boundary_delta,'moved_local_boundary_node_rows':boundary_moved.tolist()},
 'overlap':{'algorithm':'existing normalized-frame tetrahedron intersection routine from build_014-replay.py; numerical Float64/ConvexHull, not an exact predicate','candidate_aabb_pairs':int(pair_candidates),'positive_overlap_entries':len(raw),'exact_unchanged_pairs_shortcut':int(exact_identity_pairs),'discarded_below_relative_1e-14_overlap_volume_m3':float(raw_dropped),'raw_new_row_coverage_min':float(new_cov.min()),'raw_new_row_coverage_max':float(new_cov.max()),'raw_new_row_max_abs_gap_fraction':float(np.max(np.abs(new_cov-1.0))),'raw_old_column_coverage_min':float(old_cov.min()),'raw_old_column_coverage_max':float(old_cov.max()),'raw_old_column_max_abs_gap_fraction':float(np.max(np.abs(old_cov-1.0))),'raw_total_overlap_m3':float(sum(x[2] for x in raw)),'old_local_volume_m3':float(V2[old_region].sum()),'new_local_volume_m3':float(V5[new_region].sum()),'local_volume_delta_m3':float(V5[new_region].sum()-V2[old_region].sum()),'raw_swept_old_volume_m3':float(V2[old_region].sum()-sum(x[2] for x in raw)),'raw_uncovered_new_volume_m3':float(V5[new_region].sum()-sum(x[2] for x in raw)),'column_normalization':'each old-parent column divided by its measured raw overlap coverage; raw coverage and scale retained; no hidden geometric exactness claim','max_parent_column_sum_delta_vs_002_to_014':float(colerr.max()),'parent_column_sum_delta_rows_over_1e-10':int(np.count_nonzero(colerr>1e-10))},
 'transfer':{'new_to_002_offsets_key':'transfer_new_to_002_offsets','new_to_002_parent_rows_key':'transfer_new_to_002_parent_rows','new_to_002_fractions_key':'transfer_new_to_002_parent_fractions','composed_new_to_014_offsets_key':'transfer_new_to_014_offsets','composed_new_to_014_parent_rows_key':'transfer_new_to_014_parent_rows','composed_new_to_014_fractions_key':'transfer_new_to_014_parent_fractions','row_sum_max_abs_difference_from_one_new_to_002_diagnostic_only':float(max(abs(sum(Bfrac[Boff[i]:Boff[i+1]])-1) for i in range(len(T5)))),'row_sum_max_abs_difference_from_one_new_to_014_diagnostic_only':float(max(abs(sum(Cfrac[Coff[i]:Coff[i+1]])-1) for i in range(len(T5)))),'synthetic_parent_mass_seed':6006,'synthetic_total_mass_old':float(old_mass.sum()),'synthetic_total_mass_new':float(new_mass.sum()),'synthetic_total_relative_error':float(abs(new_mass.sum()-old_mass.sum())/old_mass.sum()),'synthetic_sequential_vs_composed_max_abs_error':float(np.max(np.abs(new_mass-composed_mass))),'by_material_code':region_summary},
 'gates':{'input_hashes_match':True,'005_tet_connectivity_matches_004':True,'changed_node_rows_expected':changed_nodes.tolist()==[55529,55530],'local_region_under_1000':bool(len(new_region)<=1000 and len(old_region)<=1000),'unchanged_rows_exact':True,'local_boundary_facets_match':b5==b2,'positive_old_new_cells':True,'raw_overlap_coverage_within_1e-6':bool(np.max(np.abs(new_cov-1))<=1e-6 and np.max(np.abs(old_cov-1))<=1e-6),'composed_parent_columns_match':bool(colerr.max()<=1e-10),'synthetic_mass_closes':bool(abs(new_mass.sum()-old_mass.sum())<=1e-10*old_mass.sum()),'overall_transfer_ready':bool(b5==b2 and np.max(np.abs(new_cov-1))<=1e-6 and np.max(np.abs(old_cov-1))<=1e-6 and colerr.max()<=1e-10)},
 'limitations':['conservative attribution only; geometric overlap values are numerical not exact-predicate certificates','parent-column normalization is reported explicitly and does not conceal raw volume gaps','no absolute kg ledger, physical-material admission, registered-source admission, field validity, or native run is claimed'],'elapsed_seconds':time.monotonic()-started}
(OUT/'transfer-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(OUT/'transfer-report.json'),'npz':str(OUT/'local-overlap-005-to-002.npz'),'new_region':len(new_region),'old_region':len(old_region),'overlap_entries':len(raw),'candidate_aabb_pairs':int(pair_candidates),'new_cov_range':[float(new_cov.min()),float(new_cov.max())],'old_cov_range':[float(old_cov.min()),float(old_cov.max())],'local_volumes_m3':[float(V2[old_region].sum()),float(V5[new_region].sum())],'parent_column_maxerr':float(colerr.max()),'elapsed_s':time.monotonic()-started},indent=2))
