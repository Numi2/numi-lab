#!/usr/bin/env python3
"""Checkpoint first 256 changed-cavity children against original 013 parents."""
import hashlib,json,pathlib,sys,time,os
import numpy as np
from scipy.spatial import cKDTree,ConvexHull,QhullError

E=pathlib.Path(sys.argv[1]);OUT=pathlib.Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
BASE=E/'E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz'
PROPOSAL=E/'connected-skin-safe-flow-017/run-002/connected-rest-proposal.npz'
SEL=E/'connected-skin-cavity-remesh-018/run-002-holefix-001/cavity-selection.npz'
NEW=E/'connected-skin-cavity-remesh-018/run-002-edge-splits-001/edge-split-geometry.npz'
OLD_OWNER=E/'ankle-parent-tet-refinement-026/generated-225-actual007-cut-stitch-domain-preflight/local-quality-star-remesh-232/overlap-234/local-tet-overlap-csr.npz'
OLD_OWNER_SCRIPT=pathlib.Path('/tmp/audit_overlap_234.py')
OVERLAP_OWNER_SHA='c75479baa803f2abdad530a731bd9c350f2debae29feba816c196ff8918dc310'
SEL_SHA='646ab7648cf2f8d20c19904190fab8f40328b58f7da77f855b5e102185e19727'
CAPACITY=256;CHECKPOINT_EVERY=32;started=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def signed_det(X,T):
 y=X[T];return np.einsum('ij,ij->i',y[:,1]-y[:,0],np.cross(y[:,2]-y[:,0],y[:,3]-y[:,0]))
EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)); FACETS=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
def bary(tet_xyz,p):
 A=np.stack((tet_xyz[0]-tet_xyz[3],tet_xyz[1]-tet_xyz[3],tet_xyz[2]-tet_xyz[3]),axis=1)
 u=np.linalg.solve(A,p-tet_xyz[3]);return np.array([u[0],u[1],u[2],1-u.sum()])
def intersection_volume(Axyz,Bxyz):
 # Same construction semantics as overlap-234: contained vertices + edge/face cuts + normalized ConvexHull.
 scale=max(float(np.linalg.norm(Axyz[i]-Axyz[j])) for i,j in EDGES)
 scale=max(scale,max(float(np.linalg.norm(Bxyz[i]-Bxyz[j])) for i,j in EDGES))
 if not scale>0:return 0.0
 origin=(Axyz.mean(axis=0)+Bxyz.mean(axis=0))*0.5;pts=[]
 for p in Axyz:
  if np.min(bary(Bxyz,p))>=-2e-10:pts.append((p-origin)/scale)
 for p in Bxyz:
  if np.min(bary(Axyz,p))>=-2e-10:pts.append((p-origin)/scale)
 def edge_plane(src,other):
  for i,j in EDGES:
   p0,p1=src[i],src[j]
   for face in FACETS:
    tri=other[list(face)];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);nl=np.linalg.norm(n)
    if nl<=1e-30:continue
    d0=float(n@(p0-tri[0]));d1=float(n@(p1-tri[0]))
    if d0*d1>=0:continue
    u=d0/(d0-d1)
    if u<=1e-12 or u>=1-1e-12:continue
    p=p0+u*(p1-p0)
    if np.min(bary(other,p))>=-2e-10:pts.append((p-origin)/scale)
 edge_plane(Axyz,Bxyz);edge_plane(Bxyz,Axyz)
 if len(pts)<4:return 0.0
 unique=[]
 for p in pts:
  if not any(float((p-q)@(p-q))<=1e-18 for q in unique):unique.append(p)
 if len(unique)<4:return 0.0
 try:return float(ConvexHull(np.asarray(unique),qhull_options='Qt Qx').volume*scale**3)
 except QhullError:return 0.0

# Pin current geometry and provenance before overlap work.
pins={BASE:'1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1',PROPOSAL:'64f40e915703e35950ba2d3ab95b01a5e75681616c316cace25518c3a0ccb054',NEW:'c3ca2e5383a6424ec3feeb83ccfde2d269950eced21cf4d9e578c4a0385ac417'}
for p,h in pins.items():
 if sha(p)!=h:raise RuntimeError(f'input hash mismatch {p}')
if sha(SEL)!=SEL_SHA:raise RuntimeError(f'selection hash mismatch {SEL}')
if sha(OLD_OWNER_SCRIPT)!=OVERLAP_OWNER_SHA:raise RuntimeError('reference overlap owner script drift')
if sha(OLD_OWNER)!='7f46e9bf9791884c19a8a2037a0c5ae637574af397eea908ce9516150d54e378':raise RuntimeError('reference overlap CSR drift')
# Tiny analytic regression before geometry arrays or spatial indexes are built.
toy=np.asarray([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float64)
if abs(intersection_volume(toy,toy)-1/6)>1e-11:raise RuntimeError('overlap owner self-volume toy failed')
inner=toy.copy();inner[1:]*=.5
if abs(intersection_volume(toy,inner)-1/48)>1e-11:raise RuntimeError('overlap owner contained-cell toy failed')
shifted=toy+np.asarray([2.,0,0])
if intersection_volume(toy,shifted)!=0.:raise RuntimeError('overlap owner disjoint-cell toy failed')
if os.environ.get('OVERLAP018_PREFLIGHT_ONLY')=='1':
 print(json.dumps({'stage':'preflight_pass','owner_sha256':OVERLAP_OWNER_SHA,'toy_self_volume':1/6,'toy_contained_volume':1/48,'toy_disjoint_volume':0}),flush=True);raise SystemExit(0)
with np.load(BASE,allow_pickle=False) as z:X0=z['points_world_m'].astype(np.float64);T0=z['tetrahedra'].astype(np.int64);A0=z['region_attributes'].astype(np.int64)
with np.load(SEL,allow_pickle=False) as z:old_rows=np.asarray(z['selected_input_tet_rows'],np.int64)
with np.load(PROPOSAL,allow_pickle=False) as z:changed=np.asarray(z['changed_node_ids'],np.int64)
with np.load(NEW,allow_pickle=False) as z:
 X1=z['points_world_m'].astype(np.float64);T1=np.asarray(z['tetrahedra'],np.int64);A1=np.asarray(z['region_attributes'],np.int64);par=np.asarray(z['parent_merged_global_tet_rows'],np.int64);split_edges=z['mechanical_midpoint_edge_endpoints'].astype(np.int64)
keep=np.ones(len(T0),bool);keep[old_rows]=False;retained_rows=np.flatnonzero(keep);nret=len(retained_rows)
new_rows=np.flatnonzero(par>=nret)
if len(new_rows)==0:raise RuntimeError('no TetGen cavity descendants identified')
if not np.array_equal(np.unique(A0[old_rows]),np.array([1])) or not np.array_equal(np.unique(A1[new_rows]),np.array([1])):raise RuntimeError('selected cavity is not a single skin material region')
# Prove the 017 moved-node star is wholly contained by the selected old-parent cavity.
old_cavity_nodes=np.unique(T0[old_rows]);missing_changed=np.setdiff1d(changed,old_cavity_nodes)
outside_star=np.flatnonzero(np.any(np.isin(T0,changed),axis=1)&keep)
if len(missing_changed) or len(outside_star):raise RuntimeError(f'changed domain escapes selected cavity: missing_nodes={len(missing_changed)} outside_tets={len(outside_star)}')
oldT=T0[old_rows];old_xyz=X0[oldT];newT=T1[new_rows];new_xyz=X1[newT]
old_vol=np.abs(signed_det(X0,T0[old_rows]))/6.;new_vol=np.abs(signed_det(X1,T1[new_rows]))/6.
if np.any(old_vol<=0) or np.any(new_vol<=0):raise RuntimeError('nonpositive old/new overlap cell')
# Radius-binned centroid index; each bin uses a conservative maximum old radius, then exact AABB pruning.
old_cent=old_xyz.mean(axis=1);old_rad=np.linalg.norm(old_xyz-old_cent[:,None,:],axis=2).max(axis=1)
old_lo=old_xyz.min(axis=1);old_hi=old_xyz.max(axis=1)
new_cent=new_xyz.mean(axis=1);new_rad=np.linalg.norm(new_xyz-new_cent[:,None,:],axis=2).max(axis=1)
new_lo=new_xyz.min(axis=1);new_hi=new_xyz.max(axis=1)
positive=old_rad[old_rad>0]
if len(positive)==0:raise RuntimeError('empty old parent radius index')
bins=[];edge=2.0**np.floor(np.log2(positive.min()));previous=0.0
while True:
 loc=np.flatnonzero((old_rad>previous)&(old_rad<=edge))
 if len(loc):bins.append((edge,loc,cKDTree(old_cent[loc])))
 if edge>=positive.max()*(1-1e-14):break
 previous=edge;edge*=2.
if not bins:bins=[(float(old_rad.max()),np.arange(len(old_rad)),cKDTree(old_cent))]
for ceiling,loc,tree in bins:
 if np.any(old_rad[loc]>ceiling*(1+1e-14)):raise AssertionError('radius bin is not conservative')
print(json.dumps({'stage':'index_ready','old_parent_tets':len(old_rows),'new_cavity_children_total':len(new_rows),'capacity_children':min(CAPACITY,len(new_rows)),'old_radius_bins':len(bins),'old_radius_max_m':float(old_rad.max()),'changed_nodes':len(changed),'all_changed_node_stars_inside_cavity':True,'split_edges':split_edges.tolist()},separators=(',',':')),flush=True)

csr_offsets=[0];csr_parent=[];csr_vol=[];csr_frac_new=[];csr_frac_old=[]
processed_child_rows=[];processed_vol=[];processed_coverage=[];candidate_counts=[];aabb_counts=[];positive_counts=[]
partial_parent_overlap=np.zeros(len(old_rows),np.float64)
limit=min(CAPACITY,len(new_rows))
for ni in range(limit):
 qxyz=new_xyz[ni];candidates=[]
 for ceiling,loc,tree in bins:
  cand=tree.query_ball_point(new_cent[ni],float(new_rad[ni]+ceiling))
  if len(cand):candidates.extend(loc[np.asarray(cand,np.int64)].tolist())
 if candidates:
  ci=np.asarray(candidates,np.int64)
  if len(ci)!=len(np.unique(ci)):raise AssertionError('spatial bins emitted duplicate old parents')
  candidate_counts.append(len(ci));overlap=~np.any((new_hi[ni]<old_lo[ci])|(old_hi[ci]<new_lo[ni]),axis=1);ci=ci[overlap]
 else:candidate_counts.append(0);ci=np.empty(0,np.int64)
 aabb_counts.append(len(ci));found=0
 for oi in ci:
  iv=intersection_volume(qxyz,old_xyz[oi])
  if iv>max(1e-26,min(old_vol[oi],new_vol[ni])*1e-14):
   csr_parent.append(int(old_rows[oi]));csr_vol.append(float(iv));csr_frac_new.append(float(iv/new_vol[ni]));csr_frac_old.append(float(iv/old_vol[oi]));partial_parent_overlap[oi]+=iv;found+=1
 csr_offsets.append(len(csr_parent));positive_counts.append(found)
 cv=float(np.sum(np.asarray(csr_vol[csr_offsets[-2]:csr_offsets[-1]],np.float64)))
 processed_coverage.append(cv/float(new_vol[ni]));processed_vol.append(float(new_vol[ni]));processed_child_rows.append(int(new_rows[ni]))
 if (ni+1)%CHECKPOINT_EVERY==0 or ni+1==limit:
  checkpoint={'schema':'numi.edge-split-cavity-overlap-prefix-018.v1','status':'partial_prefix_checkpoint','geometry_candidate_sha256':sha(NEW),'parent_domain_sha256':sha(BASE),'selection_sha256':sha(SEL),'processed_child_count':ni+1,'capacity_children':limit,'total_changed_cavity_children':len(new_rows),'complete_global_candidate_map':False,'complete_cavity_map':False}
  np.savez_compressed(OUT/'overlap-prefix-256.npz',new_global_tet_rows=np.asarray(processed_child_rows,np.int64),old_global_tet_rows=old_rows.copy(),child_offsets=np.asarray(csr_offsets,np.int64),old_parent_tet_rows=np.asarray(csr_parent,np.int64),intersection_volume_m3=np.asarray(csr_vol,np.float64),fraction_of_new_tet=np.asarray(csr_frac_new,np.float64),fraction_of_old_tet=np.asarray(csr_frac_old,np.float64),new_cell_volume_m3=np.asarray(processed_vol,np.float64),new_cell_covered_volume_m3=np.asarray([c*v for c,v in zip(processed_coverage,processed_vol)],np.float64),new_cell_coverage_ratio=np.asarray(processed_coverage,np.float64),partial_old_parent_overlap_volume_m3=partial_parent_overlap.copy(),old_parent_cell_volume_m3=old_vol.copy(),completion_json=np.asarray(json.dumps(checkpoint)))
  print(json.dumps({'stage':'checkpoint','processed':ni+1,'capacity':limit,'candidate_pairs_cumulative':int(sum(candidate_counts)),'aabb_pairs_cumulative':int(sum(aabb_counts)),'positive_overlaps_cumulative':len(csr_vol),'elapsed_s':time.monotonic()-started},separators=(',',':')),flush=True)

elapsed=time.monotonic()-started;processed=len(processed_child_rows);rate=processed/max(elapsed,1e-9);remaining=len(new_rows)-processed;estimate=remaining/max(rate,1e-12)
report={'schema':'numi.edge-split-cavity-overlap-prefix-018.v1','status':'partial_checkpoint_complete_bounded_probe','geometry_candidate_path':str(NEW),'geometry_candidate_sha256':sha(NEW),'original_parent_domain_path':str(BASE),'original_parent_domain_sha256':sha(BASE),'old_cavity_selection_path':str(SEL),'old_cavity_selection_sha256':sha(SEL),'overlap_owner_reference_path':str(OLD_OWNER_SCRIPT),'overlap_owner_reference_sha256':sha(OLD_OWNER_SCRIPT),'overlap_reference_checkpoint_path':str(OLD_OWNER),'overlap_reference_checkpoint_sha256':sha(OLD_OWNER),'method':'Overlap-234 contained-vertex + edge/face-intersection + normalized ConvexHull semantics, with radius-binned centroid cKDTree and AABB pruning against only the selected old cavity parents. F64 construction overlaps are not rational exact predicates. Fractions remain raw; no normalization.','domain':{'old_parent_global_tet_rows':old_rows.tolist(),'old_parent_count':len(old_rows),'old_parent_material_counts':{'1':int(len(old_rows))},'new_changed_cavity_global_tet_rows_count':len(new_rows),'processed_new_global_tet_rows':processed_child_rows,'processed_child_count':processed,'capacity_children':limit,'unprocessed_children':remaining,'all_017_changed_node_stars_in_selected_cavity':True,'other_world_cells_processed':False},'spatial_index':{'type':'cKDTree over old-parent tetrahedron centroids, grouped by conservative circumsphere-radius upper bounds, then AABB pruning','radius_bins':len(bins),'max_old_radius_m':float(old_rad.max()),'broadphase_pairs':int(sum(candidate_counts)),'aabb_candidate_pairs':int(sum(aabb_counts)),'positive_overlap_pairs':len(csr_vol),'broadphase_count_min_median_max':[int(min(candidate_counts,default=0)),float(np.median(candidate_counts)) if candidate_counts else 0,int(max(candidate_counts,default=0))],'aabb_count_min_median_max':[int(min(aabb_counts,default=0)),float(np.median(aabb_counts)) if aabb_counts else 0,int(max(aabb_counts,default=0))]},'coverage_prefix_only':{'child_coverage_ratio_min':float(min(processed_coverage)),'child_coverage_ratio_median':float(np.median(processed_coverage)),'child_coverage_ratio_max':float(max(processed_coverage)),'children_below_1_minus_1e-9':int(np.sum(np.asarray(processed_coverage)<1-1e-9)),'children_above_1_plus_1e-9':int(np.sum(np.asarray(processed_coverage)>1+1e-9)),'partial_old_parent_rows_touched':int(np.count_nonzero(partial_parent_overlap)),'partial_old_parent_overlap_volume_m3':float(partial_parent_overlap.sum()),'partial_old_parent_fraction_of_full_domain_max':float(np.max(partial_parent_overlap/old_vol))},'completion':{'current_batch_complete':True,'all_new_cavity_children_complete':False,'all_global_candidate_children_complete':False,'full_parent_CSR_available':False},'runtime':{'elapsed_s':elapsed,'processed_children_per_s':rate,'estimated_remaining_s_at_probe_rate':estimate,'estimated_full_cavity_s_at_probe_rate':len(new_rows)/max(rate,1e-12)},'outputs':{'partial_overlap_csr_path':str(OUT/'overlap-prefix-256.npz'),'partial_overlap_csr_sha256':sha(OUT/'overlap-prefix-256.npz')},'uncovered_old_new_volume':'Not inferred from a prefix. Unprocessed children and unvisited old-parent overlaps remain unknown; no conservation or mass closure is claimed.'}
(OUT/'overlap-prefix-256-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'stage':'complete','processed':processed,'capacity':limit,'broadphase_pairs':int(sum(candidate_counts)),'aabb_pairs':int(sum(aabb_counts)),'positive_overlaps':len(csr_vol),'coverage_min':report['coverage_prefix_only']['child_coverage_ratio_min'],'coverage_median':report['coverage_prefix_only']['child_coverage_ratio_median'],'coverage_max':report['coverage_prefix_only']['child_coverage_ratio_max'],'runtime_s':elapsed,'estimated_full_cavity_s':report['runtime']['estimated_full_cavity_s_at_probe_rate'],'csr_sha256':report['outputs']['partial_overlap_csr_sha256']},indent=2),flush=True)
