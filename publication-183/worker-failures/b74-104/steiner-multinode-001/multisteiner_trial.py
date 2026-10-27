import sys,json,hashlib,importlib.util
from pathlib import Path
from fractions import Fraction
from collections import Counter
import numpy as np
from scipy.spatial import Delaunay
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069');S=ROOT/'D/surface-clearance-546/component-sat-selector-638';E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
helper_path=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py');exact_path=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py';sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
ex=module('numilab_human.cardiac_cavity_intersections',exact_path);h=module('bankhelper',helper_path)
C=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz';basehash=hashlib.sha256(C.read_bytes()).hexdigest();Z=np.load(C,allow_pickle=False);X=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64)
W=json.load(open(S/'per-bank-workers-20261027/b74-104/baseline-witnesses.json'));boundary=json.load(open(S/'per-bank-workers-20261027/b74-104/steiner-multinode-001/boundary-feasibility.json'))
if not boundary['candidate_653_compatible_with_refined_baseline'] or not boundary['fixed_boundary_vertices_all_outside'] or not boundary['fixed_boundary_edges_all_disjoint_from_bone']:raise RuntimeError('fixed boundary feasibility gate failed; expand connected ring before remeshing')
old_rows=list(map(int,W['skin_child_faces']));parent_face=22189;fineface=10342;subface=10344;loop=[int(boundary['fixed_boundary_vertices'][i]['node']) for i in range(6)]
if loop!=[19094,21136,19092,20914,12694,20915]:raise RuntimeError('unexpected boundary loop')
# Recover barycentric coordinates for the six fixed boundary identities from the existing child faces.
node_bary={}
for fr in old_rows:
 for node,bary in zip(F[fr],Z['refinement_face_parent_barycentric'][fr]):node_bary[int(node)]=np.asarray(bary,float)
if set(node_bary)!=set(loop):raise RuntimeError('source face boundary identity mismatch')
corners=[next(n for n,b in node_bary.items() if np.array_equal(b,np.eye(3)[i])) for i in range(3)]
# Four interior source locations follow the three intersecting corner sectors and central residual locus.
steiner_bary=np.asarray([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3],[1/3,1/3,1/3]],np.float64)
node_ids=list(loop)+list(range(len(X),len(X)+len(steiner_bary)))
source_bary=np.vstack([np.asarray([node_bary[n] for n in loop]),steiner_bary])
coord2=source_bary[:,1:]
triang=Delaunay(coord2)
simplices=np.asarray(triang.simplices,np.int64)
if len(simplices)!=12 or sorted(set(simplices.ravel()))!=list(range(10)):raise RuntimeError(f'unexpected constrained triangulation: {simplices.tolist()}')
edge_set={tuple(sorted((int(a),int(b)))) for tri in simplices for a,b in zip(tri,np.roll(tri,-1))}
if any(tuple(sorted((i,(i+1)%6))) not in edge_set for i in range(6)):raise RuntimeError('Delaunay triangulation did not preserve a fixed boundary segment')
# Load pinned bone 104 geometry and prepare exact surface / point-location records.
EVID=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight';T,ids,gids,bodies,M,off,pos,keys,pins,_=h.read_bank_inputs(EVID,E)
mask=(ids[:,0]==74)&(ids[:,1]==104);B=T[mask];bid=ids[mask,3];R=M[74];O=off[74];P=pos[74]
base_local=(X.astype(np.float64)-P)@R.T-O

def closest(p,t):
 a,b,c=t;ab=b-a;ac=c-a;ap=p-a;d1=ab@ap;d2=ac@ap
 if d1<=0 and d2<=0:return a
 bp=p-b;d3=ab@bp;d4=ac@bp
 if d3>=0 and d4<=d3:return b
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:return a+(d1/(d1-d3))*ab
 cp=p-c;d5=ab@cp;d6=ac@cp
 if d6>=0 and d5<=d6:return c
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:return a+(d2/(d2-d6))*ac
 va=d3*d6-d5*d4
 if va<=0 and d4-d3>=0 and d5-d6>=0:return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
 den=1/(va+vb+vc);v=vb*den;w=vc*den;return a+ab*v+ac*w

def records(v,faces,face_ids=None):
 out=[]
 for i,face in enumerate(faces):
  tri=tuple(tuple(Fraction.from_float(float(x)) for x in v[int(k)]) for k in face)
  out.append((tri,tuple(min(q[a] for q in tri) for a in range(3)),tuple(max(q[a] for q in tri) for a in range(3)),int(i if face_ids is None else face_ids[i]),tuple(map(int,face))))
 return out
bankrec=records(B.reshape(-1,3),np.arange(3*len(B),dtype=np.int64).reshape(-1,3),bid)
locprep=ex.prepare_point_location(records(B.reshape(-1,3),np.arange(3*len(B),dtype=np.int64).reshape(-1,3)))
# Fit each interior node independently to the nearest exact bone boundary exit; keep outside source samples at their affine locations.
trial_world=X.copy();node_motion=[]
for i,bary in enumerate(steiner_bary):
 n=len(X)+i;affine=np.asarray(bary@X[np.asarray(corners)].astype(np.float64),np.float64);local=(affine-P)@R.T-O
 state=ex.point_location_prepared(tuple(Fraction.from_float(float(x)) for x in local),locprep)['location']
 nearest=(1e99,None,None)
 for j,t in enumerate(B):
  q=closest(local,t);d=float(np.linalg.norm(local-q))
  if d<nearest[0]:nearest=(d,j,q)
 d,j,q=nearest
 if d<=1e-12:raise RuntimeError(f'Steiner source point {i} lies on bone boundary')
 moved_local=local.copy()
 if state=='inside':moved_local=q+(q-local)/d*0.00020
 elif state=='outside':moved_local=local
 else:raise RuntimeError(f'Steiner source point {i} is classified {state}')
 world=(R.T@(moved_local+O)+P).astype(np.float32)
 displacement=float(np.linalg.norm(world.astype(np.float64)-affine))
 if displacement>0.004000001:raise RuntimeError(f'Steiner point {i} exceeded the 4 mm trust: {displacement}')
 trial_world=np.vstack([trial_world,world.reshape(1,3)])
 replay=(world.astype(np.float64)-P)@R.T-O
 post=ex.point_location_prepared(tuple(Fraction.from_float(float(x)) for x in replay),locprep)['location']
 node_motion.append({'node':n,'source_barycentric':bary.tolist(),'source_location':state,'nearest_bone_triangle':int(bid[j]),'nearest_boundary_distance_mm':d*1000,'post_location':post,'world_f32_m':world.tolist(),'affine_source_world_m':affine.tolist(),'delta_world_m':(world.astype(np.float64)-affine).tolist(),'displacement_mm':displacement*1000})
# Make conforming 3D patch triangles, retaining Delaunay topology but orienting in source-face winding.
fan=[];fan_bary=[];parent_normal=np.cross(base_local[corners[1]]-base_local[corners[0]],base_local[corners[2]]-base_local[corners[0]])
trial_local=(trial_world.astype(np.float64)-P)@R.T-O
for tri0 in simplices:
 tri=np.asarray(tri0,np.int64);xy=coord2[tri]
 if ((xy[1,0]-xy[0,0])*(xy[2,1]-xy[0,1])-(xy[1,1]-xy[0,1])*(xy[2,0]-xy[0,0]))<0:tri=tri[[0,2,1]]
 outnodes=[node_ids[int(i)] for i in tri]
 xyz=trial_local[np.asarray(outnodes)]
 if float(np.dot(np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]),parent_normal))<0:outnodes=[outnodes[0],outnodes[2],outnodes[1]]
 fan.append(outnodes);fan_bary.append([source_bary[int(i)] for i in tri])
fan=np.asarray(fan,np.int64);fan_bary=np.asarray(fan_bary,np.float64);Fcount=len(F);fan_rows=old_rows+list(range(Fcount,Fcount+len(fan)))
trial_faces=F.copy();trial_faces[old_rows]=fan[:4];trial_faces=np.vstack([trial_faces,fan[4:]])
# Exact boundary and edge-incidence checks after remesh.
edge_counts=Counter(tuple(sorted(map(int,e))) for face in fan for e in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])))
full_edge_counts=Counter(tuple(sorted(map(int,e))) for face in trial_faces for e in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])))
boundary_edge_counts={f'{loop[i]}-{loop[(i+1)%6]}':{'patch_incidence':edge_counts[tuple(sorted((loop[i],loop[(i+1)%6])))],'full_mesh_incidence':full_edge_counts[tuple(sorted((loop[i],loop[(i+1)%6])))]} for i in range(6)}
if any(v['patch_incidence']!=1 or v['full_mesh_incidence']!=2 for v in boundary_edge_counts.values()):raise RuntimeError(f'conforming fixed boundary failed: {boundary_edge_counts}')
# Exact bank-104 triangle intersection test on all remeshed faces, then inside/outside samples.
skinrec=records(trial_local,fan,fan_rows);bank_check=ex._audit_pair(skinrec,bankrec,same_surface=False)
face_centers=[trial_local[face].mean(0) for face in fan]
samples=[trial_local[n] for n in range(len(X),len(X)+len(steiner_bary))]+face_centers
sample_locs=[ex.point_location_prepared(tuple(Fraction.from_float(float(x)) for x in p),locprep)['location'] for p in samples]
residual=[]
for skinrow,boneid0 in bank_check['triangle_pairs']:
 fi=fan_rows.index(int(skinrow));s=trial_local[fan[fi]];idx=np.flatnonzero(bid==int(boneid0))
 if len(idx)!=1:continue
 b=B[int(idx[0])];points=ex.triangle_intersection_points(tuple(tuple(Fraction.from_float(float(v)) for v in q) for q in s),tuple(tuple(Fraction.from_float(float(v)) for v in q) for q in b));uniq=[]
 for p in points:
  if p not in uniq:uniq.append(p)
 residual.append({'skin_face_row':int(skinrow),'stable_bone_triangle_row':int(boneid0),'intersection_vertices_body74_local_m':[[float(v) for v in p] for p in uniq]})
# Preserve an exact local failure artifact and stop before self testing if the bank gate fails.
out=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b74-104/steiner-multinode-001')
base_report={'schema':'numi.human.per_bank_worker_multisteiner.v1','input_candidate_path':str(C),'input_candidate_sha256':basehash,'base_candidate_status':'COMBINED_CONSTRUCTION_ONLY_653','boundary_feasibility_path':str(out/'boundary-feasibility.json'),'boundary_nodes_fixed':loop,'boundary_edge_incidence':boundary_edge_counts,'source_fine_detail_face':fineface,'source_fine_subface_row':subface,'immediate_parent_face_row':parent_face,'steiner_node_motion':node_motion,'steiner_triangulation_2d_simplices':simplices.tolist(),'fan_face_rows':fan_rows,'fan_faces':fan.tolist(),'fan_face_source_barycentric_vertices':fan_bary.tolist(),'raw_bank104_child_pairs_after':int(bank_check['count']),'bank104_pair_rows_after':bank_check['triangle_pairs'],'bank104_residual_exact_witnesses':residual,'steiner_and_face_centroid_point_locations':sample_locs,'local_exact_bank104_pass':int(bank_check['count'])==0 and all(x=='outside' for x in sample_locs),'bank_input_pins':pins,'accepted_physical_steps':0,'geometry_admitted':False,'full_audits_run':False}
if not base_report['local_exact_bank104_pass']:
 patch=out/'multi-steiner-failed-patch.npz';np.savez_compressed(patch,points_world_m=trial_world,source_O_faces=trial_faces,trial_fan_faces=fan,trial_fan_output_rows=np.asarray(fan_rows,np.int64),trial_fan_face_source_barycentric_vertices=fan_bary,trial_boundary_nodes=np.asarray(loop,np.int64),trial_steiner_nodes=np.arange(len(X),len(X)+len(steiner_bary),dtype=np.int64),trial_steiner_source_face_barycentric=steiner_bary,trial_steiner_delta_world_m=np.asarray([np.asarray(x['delta_world_m']) for x in node_motion],np.float64))
 base_report.update({'status':'FAIL_EXACT_LOCAL_SKIN_BONE','failed_patch_path':str(patch),'failed_patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'local_self_test_run':False,'next_method':'Expand the connected patch to the next intact ring only if an exact boundary test shows the current boundary is infeasible; otherwise refine the internal triangulation/Steiner placement around the residual exact loci, retaining the same exact source ancestry and 4 mm trust.'})
 (out/'multi-steiner-report.json').write_text(json.dumps(base_report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n');print(json.dumps({k:v for k,v in base_report.items() if k not in ['bank_input_pins','bank104_residual_exact_witnesses','steiner_triangulation_2d_simplices','fan_face_source_barycentric_vertices']},indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)));raise SystemExit(0)
# Exact local self test on all output faces touching the new patch.
allrecords,_=ex.float32_surface_records(trial_world,trial_faces,source_seam_groups=())
self_check=ex._audit_self_pairs_touching_faces(allrecords,set(fan_rows))
base_report.update({'local_self_test_run':True,'local_self_pair_count':int(self_check['count']),'local_self_pair_rows':self_check['triangle_pairs']})
if int(self_check['count']):
 base_report.update({'status':'FAIL_EXACT_LOCAL_SELF','next_method':'The multi-Steiner fan clears bone 104 but has exact local self pairs; retain this failed geometry and use a different internal triangulation.'})
 (out/'multi-steiner-report.json').write_text(json.dumps(base_report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n');print(json.dumps(base_report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)));raise SystemExit(0)
# Build a fully typed candidate; all existing unrelated payload and historical evidence are preserved.
N=len(X);new_count=len(steiner_bary);payload={k:np.array(Z[k]) for k in Z.files};face_fields={'face_kind':2,'parent_coarse_face_row':-1,'parent_fine_detail_face_row':fineface,'parent_source_face_marker':fineface,'parent_fine_subface_row':subface,'refinement_face_parent_face_row':parent_face,'refinement_face_parent_ancestry_face_rows':parent_face}
for key,val in face_fields.items():
 a=payload[key].copy();a[old_rows]=val;payload[key]=np.concatenate([a,np.full((len(fan)-4,)+a.shape[1:],val,dtype=a.dtype)],axis=0)
for key in ['source_O_faces','area_reference_037_source_O_faces']:
 a=payload[key].copy();a[old_rows]=fan[:4];payload[key]=np.vstack([a,fan[4:]])
payload['refinement_face_parent_barycentric'][old_rows]=fan_bary[:4];payload['refinement_face_parent_barycentric']=np.concatenate([payload['refinement_face_parent_barycentric'],fan_bary[4:]],axis=0)
for ptrkey,rowkey,newseg in [('face_source_ancestry_indptr','face_source_ancestry_face_rows',[fineface]),('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows',[subface]),('face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows',[fineface]),('refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows',[parent_face])]:
 oldptr=np.asarray(Z[ptrkey],np.int64);oldvals=np.asarray(Z[rowkey]);segs=[]
 for faceid in range(Fcount):segs.append(np.asarray(newseg,dtype=oldvals.dtype) if faceid in old_rows else oldvals[oldptr[faceid]:oldptr[faceid+1]])
 segs.extend([np.asarray(newseg,dtype=oldvals.dtype) for _ in range(len(fan)-4)])
 ptr=np.zeros(Fcount+len(fan)+1,np.int64);flat=[]
 for i,seg in enumerate(segs):flat.extend(seg.tolist());ptr[i+1]=len(flat)
 payload[ptrkey]=ptr;payload[rowkey]=np.asarray(flat,dtype=oldvals.dtype)
# Append typed point-domain rows by barycentric interpolation; use actual Float32 geometry only for points_world_m.
node_keys=['points_world_m','node_kind','node_source_row','node_source_virtual_row','node_parent_coarse_face_row','node_parent_coarse_edge_a','node_parent_coarse_edge_b','node_parent_coarse_edge_parameter','node_source_displacement_m','points_before_connected_source_unfold_m','node_connected_source_unfold_displacement_m','points_before_joint_self_pair_repair_m','node_joint_self_pair_repair_displacement_m','node_box_constrained_source_fidelity_delta_m','node_source_fidelity_target_mask','all_inherited_exit_target_mask','refinement_node_kind','refinement_node_parent_edge_rows','refinement_node_parent_edge_weights','local_witness_cluster_199_delta_m','local_witness_cluster_199_source_reference_m','area_reference_037_points_world_m_f64','local_witness_cluster_222_delta_m','local_witness_cluster_222_source_reference_R0_m','local_witness_cluster_222_area_reference_037_m']
for key in node_keys:
 a=payload[key];new=[]
 for i,bary in enumerate(steiner_bary):
  if key=='points_world_m':v=trial_world[N+i]
  elif key=='node_kind':v=np.asarray(-1,dtype=a.dtype)
  elif key in ['node_source_row','node_source_virtual_row','node_parent_coarse_face_row','node_parent_coarse_edge_a','node_parent_coarse_edge_b','node_parent_coarse_edge_parameter']:v=np.asarray(-1,dtype=a.dtype)
  elif key=='node_source_displacement_m':v=np.asarray(node_motion[i]['displacement_mm']/1000.0,dtype=a.dtype)
  elif key in ['node_source_fidelity_target_mask','all_inherited_exit_target_mask']:v=np.asarray(False,dtype=a.dtype)
  elif key=='refinement_node_kind':v=np.asarray(2,dtype=a.dtype)
  elif key=='refinement_node_parent_edge_rows' or key=='refinement_node_parent_edge_weights':v=np.asarray([-1,-1],dtype=a.dtype)
  else:v=np.tensordot(bary,a[np.asarray(corners)],axes=(0,0)).astype(a.dtype)
  new.append(np.asarray(v,dtype=a.dtype).reshape((1,)+a.shape[1:]))
 payload[key]=np.concatenate([a]+new,axis=0)
# Four parent-face CSR records and immediate/prolongation rows.
ptr=np.asarray(Z['refinement_node_parent_face_indptr'],np.int64);payload['refinement_node_parent_face_indptr']=np.concatenate([ptr,ptr[-1]+np.arange(1,new_count+1,dtype=np.int64)])
payload['refinement_node_parent_face_rows']=np.concatenate([Z['refinement_node_parent_face_rows'],np.full(new_count,parent_face,dtype=Z['refinement_node_parent_face_rows'].dtype)])
payload['refinement_node_parent_face_barycentric']=np.concatenate([Z['refinement_node_parent_face_barycentric'],steiner_bary],axis=0)
imm_idx=[];imm_w=[];src_idx=[];src_w=[];src_ptr=np.asarray(Z['source_prolongation_indptr'],np.int64);imm_ptr=np.asarray(Z['immediate_refinement_prolongation_indptr'],np.int64)
for bary in steiner_bary:
 imm_idx.extend(corners);imm_w.extend(bary.tolist());terms={}
 for j,node in enumerate(corners):
  a,b=map(int,src_ptr[node:node+2])
  for k,w in zip(Z['source_prolongation_node_rows'][a:b],Z['source_prolongation_weights'][a:b]):terms[int(k)]=terms.get(int(k),0.0)+float(bary[j])*float(w)
 terms={k:v for k,v in sorted(terms.items()) if abs(v)>1e-15}
 if not np.isclose(sum(terms.values()),1.0,rtol=0,atol=1e-12):raise RuntimeError('composed source interpolation row does not sum to one')
 src_idx.extend(terms);src_w.extend(terms.values());src_ptr=np.append(src_ptr,int(src_ptr[-1])+len(terms))
 imm_ptr=np.append(imm_ptr,int(imm_ptr[-1])+3)
payload['source_prolongation_indptr']=src_ptr;payload['source_prolongation_node_rows']=np.concatenate([Z['source_prolongation_node_rows'],np.asarray(src_idx,dtype=Z['source_prolongation_node_rows'].dtype)])
payload['source_prolongation_weights']=np.concatenate([Z['source_prolongation_weights'],np.asarray(src_w,dtype=Z['source_prolongation_weights'].dtype)])
payload['immediate_refinement_prolongation_indptr']=imm_ptr;payload['immediate_refinement_prolongation_node_rows']=np.concatenate([Z['immediate_refinement_prolongation_node_rows'],np.asarray(imm_idx,dtype=Z['immediate_refinement_prolongation_node_rows'].dtype)])
payload['immediate_refinement_prolongation_weights']=np.concatenate([Z['immediate_refinement_prolongation_weights'],np.asarray(imm_w,dtype=Z['immediate_refinement_prolongation_weights'].dtype)])
# Explicit per-worker geometry delta and source lineage ledger.
delta=np.zeros((N+new_count,3),np.float32)
for i,bary in enumerate(steiner_bary):delta[N+i]=trial_world[N+i]-np.asarray(bary@X[np.asarray(corners)].astype(np.float64),np.float32)
payload['worker_b74_104_delta_world_m']=delta
payload['worker_b74_104_scope_nodes']=np.asarray(loop+list(range(N,N+new_count)),np.int32)
payload['worker_b74_104_scope_faces']=np.asarray(fan_rows,np.int32)
payload['worker_b74_104_source_candidate_sha256']=np.asarray([basehash])
payload['worker_b74_104_method']=np.asarray(['fixed six-node boundary; four local interior Steiner points fitted independently to bone-104 nearest exterior along barycentric residual loci; Delaunay conforming patch; source barycentric ancestry explicit; actual Float32 offsets in worker_b74_104_delta_world_m'])
payload['worker_b74_104_construction_trust_limit_m']=np.asarray([0.004],np.float64)
payload['worker_b74_104_new_node_parent_face_row']=np.full(new_count,parent_face,np.int64)
payload['worker_b74_104_new_node_parent_barycentric']=steiner_bary
payload['refinement_node_kind_semantics']=np.asarray(['0=preexisting; 1=selected shared-edge midpoint; 2=local interior Steiner node; non-affine coordinate offsets are separately recorded in worker_b74_104_delta_world_m'])
payload['refinement_node_parent_face_barycentric_semantics']=np.asarray(['local face-interior Steiner ancestry in the immediate parent input face; actual geometry offset is separately recorded in worker_b74_104_delta_world_m'])
payload['source_prolongation_semantics']=np.asarray(['rows map output nodes through immediate refinement to preserved source rows; each local Steiner row is barycentric interpolation and its construction offset is separately recorded'])
payload['immediate_refinement_prolongation_semantics']=np.asarray(['rows map output nodes to immediate parent by identity, edge midpoint, or face-interior barycentric interpolation; construction offset is separately recorded'])
payload['refinement_face_parent_barycentric_semantics']=np.asarray(['output child vertices retain source barycentric ancestry in pre-refinement parent face; non-affine Steiner position offsets are separately recorded'])
payload['refinement_boundary_semantics']=np.asarray(['source face boundary node coordinates and loop identities preserved; face interior retriangulated with four barycentrically anchored Steiner nodes'])
# Structural consistency checks over every row-aligned array.
for key in Z.files:
 a=Z[key]
 if a.ndim and a.shape[0]==N and payload[key].shape[0]!=N+new_count:raise RuntimeError(f'node aligned payload mismatch {key} {payload[key].shape}')
 if a.ndim and a.shape[0]==Fcount and payload[key].shape[0]!=Fcount+len(fan)-4:raise RuntimeError(f'face aligned payload mismatch {key} {payload[key].shape}')
for key in ['refinement_node_parent_face_indptr','source_prolongation_indptr','immediate_refinement_prolongation_indptr']:
 if payload[key].shape[0]!=N+new_count+1:raise RuntimeError(f'node CSR pointer mismatch {key}')
for key in ['face_source_ancestry_indptr','face_fine_subface_ancestry_indptr','face_fine_detail_ancestry_indptr','refinement_face_parent_ancestry_indptr']:
 if payload[key].shape[0]!=Fcount+len(fan)-4+1:raise RuntimeError(f'face CSR pointer mismatch {key}')
if not np.array_equal(payload['source_O_faces'],payload['area_reference_037_source_O_faces']):raise RuntimeError('area-reference source faces differ from candidate')
for fr in fan_rows:
 if int(payload['parent_fine_detail_face_row'][fr])!=fineface or int(payload['parent_source_face_marker'][fr])!=fineface or int(payload['refinement_face_parent_face_row'][fr])!=parent_face:raise RuntimeError('fan source lineage mismatch')
cand=out/'candidate-multisteiner-001.npz';np.savez_compressed(cand,**payload)
base_report.update({'status':'PASS_LOCAL_EXACT_BANK_AND_SELF','local_self_test_run':True,'local_self_pair_count':0,'candidate_path':str(cand),'candidate_sha256':hashlib.sha256(cand.read_bytes()).hexdigest(),'candidate_point_count':N+new_count,'candidate_face_count':Fcount+len(fan)-4,'payload_key_count':len(payload),'face_ancestry_update':'replace four patch faces with twelve; every new fan face maps to fine-detail ancestry 10342, fine-subface ancestry 10344, immediate input face row 22189, and per-vertex source barycentrics','node_ancestry_update':'append four refinement-node-kind=2 Steiner rows; each has refinement parent-face CSR row 22189 and source barycentric coordinates, immediate Q interpolation to the three source-face corner node rows, composed source P interpolation, and a recorded out-of-plane Float32 worker delta','candidate_self_screen_scope':'exact complete-surface records with all pairs touching every new patch face','full_audits_run':False,'max_steiner_displacement_mm':max(x['displacement_mm'] for x in node_motion),'accepted_physical_steps':0,'geometry_admitted':False})
(out/'multi-steiner-report.json').write_text(json.dumps(base_report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
print(json.dumps({k:v for k,v in base_report.items() if k not in ['bank_input_pins','bank104_residual_exact_witnesses','steiner_triangulation_2d_simplices','fan_face_source_barycentric_vertices']},indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
