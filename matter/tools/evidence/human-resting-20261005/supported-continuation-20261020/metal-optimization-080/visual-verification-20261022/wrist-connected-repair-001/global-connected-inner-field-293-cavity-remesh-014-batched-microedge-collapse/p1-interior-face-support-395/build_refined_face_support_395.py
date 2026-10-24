#!/usr/bin/env python3
"""One deterministic shared-support refinement of the 393 unsolved P1 mesh."""
import hashlib, importlib.util, json, time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=OUT.parents[1]
IN=BASE/'p1-distance-initializer-393/p1-distance-initializer-unsolved.npz'
CAND=BASE/'candidate.npz'
HELP=ROOT/'conforming-interior-support-owner-312/tools/human_skin_p1_material_volume.py'
GUARD=ROOT/'p1-material-field-attainability-owner-390/tools/human_skin_p1_material_volume.py'
HELP126=ROOT/'ankle-parent-tet-refinement-026/build_common_volume_126.py'
HELP224=ROOT/'ankle-parent-tet-refinement-026/generated-224-commonfield-tetgen-volume-actual007/build_commonfield_224.py'
BANK=ROOT/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz'
INV=ROOT/'registered-surrounding-tissue-native013-191/inventory.json'
POSE=ROOT/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
TARGETS={'skin':0.003452570642,'skin_plus_fat':0.015104913881}
EXPECTED={'candidate014':'e70b3e08aa385c8a321ebb0849257ec4afa04d13050e26fc28647d36146d26ba','initializer393':'887eb4fb47bf47144397ae41cf77394763a2d1e120a977ba2a8176e4d3ed9740','banks689':'39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792','inventory689':'b0823cf8f33df2586ba1cd45e5b260cd75d1e35b8d2682b67c15f64ea22bbcc7','pose157':'0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd'}
FLOC=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
ELOC=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_functions(path,names,extra):
 import ast
 tree=ast.parse(Path(path).read_text());fs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 if {n.name for n in fs}!=set(names):raise RuntimeError(f'missing helper functions {names} in {path}')
 ns=dict(extra);exec(compile(ast.Module(body=fs,type_ignores=[]),str(path),'exec'),ns)
 return {n:ns[n] for n in names}
def dets(x,t):
 a=np.asarray(x,dtype=np.float64)[t];e=a[:,1:]-a[:,:1]
 return np.einsum('ij,ij->i',e[:,0],np.cross(e[:,1],e[:,2]))
def closest(P,V,F,k,cb,chunk=192):
 centers=np.concatenate([V[F[i:i+100000]].astype(np.float64).mean(axis=1) for i in range(0,len(F),100000)])
 tree=cKDTree(centers);n=len(P);dist=np.empty(n);face=np.empty(n,np.int64);bary=np.empty((n,3));point=np.empty((n,3))
 for lo in range(0,n,chunk):
  pp=np.asarray(P[lo:lo+chunk],np.float64);_,ix=tree.query(pp,k=min(k,len(F)),workers=-1)
  if ix.ndim==1:ix=ix[:,None]
  tris=V[F[ix]].astype(np.float64);bb,ds=cb(pp,tris);kk=np.argmin(ds,axis=1);rr=np.arange(len(pp));ids=ix[rr,kk]
  bary[lo:lo+len(pp)]=bb[rr,kk];face[lo:lo+len(pp)]=ids;dist[lo:lo+len(pp)]=np.sqrt(ds[rr,kk]);point[lo:lo+len(pp)]=np.einsum('ni,nij->nj',bary[lo:lo+len(pp)],V[F[ids]].astype(np.float64))
 return dist,face,bary,point
def write_npz(path,**kw):
 tmp=Path(str(path)+'.tmp.npz');np.savez_compressed(tmp,**kw);tmp.replace(path)
def main():
 t0=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
 hashes={'candidate014':sha(CAND),'initializer393':sha(IN),'refinement_owner312':sha(HELP),'volume_guard390':sha(GUARD),'closest_helper126':sha(HELP126),'winding_helper224':sha(HELP224),'banks689':sha(BANK),'inventory689':sha(INV),'pose157':sha(POSE)}
 for k,v in EXPECTED.items():
  if hashes[k]!=v:raise RuntimeError(f'input pin mismatch {k}: {hashes[k]} != {v}')
 iz=np.load(IN,allow_pickle=False);c=np.load(CAND,allow_pickle=False)
 X=np.asarray(iz['points_world_m'],np.float32);T=np.asarray(iz['tetrahedra'],np.int64);phi=np.asarray(iz['phi_initializer_unsolved'],np.float64)
 if not(np.array_equal(X,c['points_world_m']) and np.array_equal(T,c['tetrahedra'])):raise RuntimeError('initializer/candidate geometry mismatch')
 flat=np.flatnonzero(np.all(phi[T]==0.,axis=1));N=len(X)
 if len(flat)!=6041:raise RuntimeError(f'flat zero cell count changed: {len(flat)}')
 shell=np.asarray(c['physical_shell_faces'],np.int64);shell_sorted=np.unique(np.sort(shell,axis=1),axis=0)
 FF=np.sort(T[:,np.asarray(FLOC)].reshape(-1,3),axis=1);UF,inv,cnt=np.unique(FF,axis=0,return_inverse=True,return_counts=True)
 if not np.array_equal(UF[cnt==1],shell_sorted):raise RuntimeError('pinned shell != tetrahedral boundary')
 face_ids=inv.reshape(len(T),4)
 shell_edges=np.sort(shell[:,np.asarray(((0,1),(0,2),(1,2)))].reshape(-1,2),axis=1)
 bedges=np.unique(shell_edges[:,0].astype(np.int64)*N+shell_edges[:,1])
 def is_bedge(a,b):
  if a>b:a,b=b,a
  key=a*N+b;j=int(np.searchsorted(bedges,key));return j<len(bedges) and int(bedges[j])==key
 faces=set();edge_lengths=[];face_areas=[]
 for row in flat:
  fcand=[]
  for fid in face_ids[row]:
   if cnt[fid]!=2: continue
   f=tuple(map(int,UF[fid]));a,b,d=X[list(f)].astype(np.float64);area=float(np.linalg.norm(np.cross(b-a,d-a)))
   fcand.append((area,-f[0],-f[1],-f[2],f))
  if not fcand: raise RuntimeError(f'flat source tet {row} has no incidence-two face')
  best=max(fcand);faces.add(best[4]);face_areas.append(float(best[0]))
 edge_array=np.empty((0,2),np.int64);face_array=np.asarray(sorted(faces),np.int64).reshape(-1,3)
 outer_nodes=np.unique(c['physical_outer_faces']);cap_only=np.setdiff1d(np.unique(c['construction_closure_faces']),outer_nodes);shell_nodes=np.unique(shell)
 flatfaces=face_ids[flat];fb=(cnt[flatfaces]==1).sum(axis=1)
 pairs=np.asarray(ELOC);fe=np.sort(T[flat][:,pairs].reshape(-1,2),axis=1);key=fe[:,0]*N+fe[:,1];pos=np.searchsorted(bedges,key);eb=(pos<len(bedges))&(bedges[np.minimum(pos,len(bedges)-1)]==key);eb=eb.reshape(-1,6).sum(axis=1)
 vols=np.abs(dets(X,T[flat]))/6
 report={'schema':'numi.human.p1-interior-face-support-refinement.395.v1','status':'classified_supports','native_admission':False,'input_sha256':hashes,
  'mesh':{'nodes':len(X),'tets':len(T),'physical_outer_faces':len(c['physical_outer_faces']),'construction_closure_faces':len(c['construction_closure_faces']),'full_shell_faces':len(shell),'shell_exactly_matches_tet_boundary':True},
  'flat_initializer_cells':{'count':len(flat),'all_four_vertices_physical_O':int(np.sum(np.all(np.isin(T[flat],outer_nodes),axis=1))),'all_four_vertices_shell':int(np.sum(np.all(np.isin(T[flat],shell_nodes),axis=1))),'any_construction_cap_only_vertex':int(np.sum(np.any(np.isin(T[flat],cap_only),axis=1))),'inside_physical_O_witnesses':int(np.sum(iz['inside_closed_union'][outer_nodes])),'volume_m3':float(vols.sum()),'boundary_face_count_histogram':{str(int(k)):int(v) for k,v in zip(*np.unique(fb,return_counts=True))},'boundary_edge_count_histogram':{str(int(k)):int(v) for k,v in zip(*np.unique(eb,return_counts=True))},'all_six_edges_boundary':int(np.sum(eb==6)),'no_interior_edges':int(np.sum(eb==0)),'region_attributes':{str(int(k)):int(v) for k,v in zip(*np.unique(c['region_attributes'][flat],return_counts=True))}},
  'support_selection':{'method':'each flat tet selects its largest-area incidence-two face; deterministic lexicographic tie break','unique_interior_edges':len(edge_array),'unique_interior_faces':len(face_array),'edge_length_quantiles_m':np.quantile(edge_lengths,[0,.5,.95,1]).tolist() if edge_lengths else [],'fallback_face_double_area_quantiles_m2':np.quantile(face_areas,[0,.5,1]).tolist() if face_areas else [],'physical_O_and_full_shell_boundary_supports_excluded':True}}
 (OUT/'classification-and-selection.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 write_npz(OUT/'selected-interior-supports.npz',flat_source_tet_rows=flat,selected_interior_edges=edge_array,selected_interior_faces=face_array)
 spec=importlib.util.spec_from_file_location('human_support312',HELP);hm=importlib.util.module_from_spec(spec);spec.loader.exec_module(hm)
 refined=hm.refine_interior_support(X,T,selected_edges=edge_array,selected_faces=face_array)
 Y=np.asarray(refined['points'],np.float32);U=np.asarray(refined['tetrahedra'],np.int64);parent=np.asarray(refined['source_tetrahedron_rows'],np.int64)
 if not np.array_equal(Y[:N],X):raise RuntimeError('original coordinate prefix changed')
 if not np.array_equal(refined['boundary_faces'],shell_sorted):raise RuntimeError('full shell boundary changed')
 ip=np.asarray(refined['node_source_indptr'],np.int64);ix=np.asarray(refined['node_source_indices'],np.int64);wt=np.asarray(refined['node_source_weights'],np.float64)
 phi_prolonged=np.add.reduceat(wt*phi[ix],ip[:-1]);phi_prolonged[:N]=phi
 if not np.array_equal(phi_prolonged[:N],phi):raise RuntimeError('original P1 initializer prefix changed')
 vnew=np.abs(dets(Y,U))/6.;vold=np.abs(dets(X,T))/6.;sumv=np.bincount(parent,weights=vnew,minlength=len(T));coverage=sumv/vold
 if np.any(coverage<=0) or np.max(np.abs(coverage-1))>2e-5:raise RuntimeError(f'parent-volume partition error {np.max(np.abs(coverage-1))}')
 attrs=np.asarray(c['region_attributes'],np.int64);child_attrs=attrs[parent];fractions=vnew/sumv[parent]
 r0={str(int(r)):float(vold[attrs==r].sum()) for r in np.unique(attrs)};r1={str(int(r)):float(vnew[child_attrs==r].sum()) for r in np.unique(child_attrs)}
 report.update({'status':'positive_geometry_refined_unsolved_P1','elapsed_s':time.monotonic()-t0,'refinement':refined['report'],'output_nodes':len(Y),'output_tetrahedra':len(U),'added_nodes':len(Y)-N,'added_tetrahedra':len(U)-len(T),'nonpositive_F32_cells':int(np.sum(dets(Y,U)<=0)),'minimum_F32_determinant_m3':float(dets(Y,U).min()),'parent_volume_partition':{'max_abs_relative_delta':float(np.max(np.abs(coverage-1))),'old_region_volumes_m3':r0,'new_region_volumes_m3':r1,'child_volume_fraction_of_parent_sums_one':True,'mass_note':'Every child retains exact source parent tet and region attribute. The per-child volume fraction can distribute any parent mass conservatively; candidate014 has no per-tet mass array, so no absolute mass ledger is asserted.'},'boundary_preservation':{'original_coordinate_prefix_bitwise_equal':True,'complete_shell_boundary_facets_unchanged':True,'physical_O_rows_and_faces_unchanged':True,'construction_closure_tagged_separately':True},'field_note':'The field is only P1 prolongation until new-row values are recomputed; no LP/field solve or material extraction.', 'native_admission':False})
 write_npz(OUT/'refined-background-checkpoint.npz',points_world_m=Y,tetrahedra=U,phi_p1_prolonged_unsolved=phi_prolonged,child_source_tetrahedron_rows=parent,child_region_attributes=child_attrs,child_volume_fraction_of_parent=fractions,node_source_indptr=ip,node_source_indices=ix,node_source_weights=wt,physical_shell_faces=shell,physical_shell_parent_input_face_row=c['physical_shell_parent_input_face_row'],physical_shell_face_is_physical_O=c['physical_shell_face_is_physical_O'],physical_outer_faces=c['physical_outer_faces'],physical_outer_parent_input_face_row=c['physical_outer_parent_input_face_row'],construction_closure_faces=c['construction_closure_faces'],construction_closure_parent_input_face_row=c['construction_closure_parent_input_face_row'],original_phi=phi,physical_outer_node_ids=outer_nodes,construction_cap_only_node_ids=cap_only)
 (OUT/'construction-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'stage':'positive_refinement_saved','nodes':len(Y),'tets':len(U),'added_nodes':len(Y)-N,'added_tets':len(U)-len(T),'min_det_m3':float(dets(Y,U).min()),'min_quality':refined['report']['minimum_float32_quality'],'below_1e-5':refined['report']['below_quality_floor_1e_5'],'max_parent_volume_relerr':float(np.max(np.abs(coverage-1))),'elapsed_s':time.monotonic()-t0},sort_keys=True),flush=True)
 # Re-evaluate initializer only on new rows with identical 393 helpers and pinned 689 banks.
 P=Y[N:].astype(np.float64);invj=json.loads(INV.read_text());banks=np.load(BANK,allow_pickle=False);vparts=[];fparts=[];fbank=[];groups=defaultdict(list);vb=0
 for bi,m in enumerate(invj['banks']):
  vv=np.asarray(banks[f'bank_{bi:04d}_vertices'],np.float32);ff=np.asarray(banks[f'bank_{bi:04d}_faces'],np.int64)
  vparts.append(vv);fparts.append(ff+vb);vb+=len(vv);fbank.append(np.full(len(ff),bi,np.int32));groups[tuple(sorted(map(int,m['body_indices'])))].append((vv,ff))
 BV=np.concatenate(vparts);BF=np.concatenate(fparts);FB=np.concatenate(fbank);kind=np.asarray([0 if m['kind']=='anatomy' else 1 for m in invj['banks']],np.uint8);FK=kind[FB]
 cb=load_functions(HELP126,{'closest_bary'},{'np':np})['closest_bary']
 dB,bfi,bb,bq=closest(P,BV,BF,64,cb);OF=np.asarray(c['physical_outer_faces'],np.int64);dO,ofi,ob,oq=closest(P,Y,OF,48,cb)
 near=dB<=np.where(FK[bfi]==0,.00025,.00010);inside=np.zeros(len(P),bool);closed_count=open_count=tested=0
 wind_api=load_functions(HELP224,{'consistent_closed','solid_angle_winding'},{'np':np,'Counter':Counter,'math':__import__('math')});CLOSE,WIND=wind_api['consistent_closed'],wind_api['solid_angle_winding']
 for body_ids,items in groups.items():
  vvparts=[];ffparts=[];off=0
  for vv,ff in items:vvparts.append(vv);ffparts.append(ff+off);off+=len(vv)
  Vcat=np.concatenate(vvparts);Fcat=np.concatenate(ffparts);Vg,iv=np.unique(Vcat,axis=0,return_inverse=True);Fg=iv[Fcat]
  closed,_=CLOSE(Fg)
  if not closed:open_count+=1;continue
  closed_count+=1;mn=Vg.min(0).astype(float)-1e-6;mx=Vg.max(0).astype(float)+1e-6;ids=np.flatnonzero(np.all((P>=mn)&(P<=mx),axis=1))
  if len(ids):
   w=WIND(P[ids],Vg,Fg);inside[ids[np.abs(w)>.5]]=True;tested+=len(ids)
 phi_new=np.clip(np.where(inside|near,1.,dO/np.maximum(dO+np.maximum(dB,1e-9),1e-9)),0.,1.);phi_all=phi_prolonged.copy();phi_all[N:]=phi_new
 oldinside=np.asarray(iz['inside_closed_union'],bool);oldnear=np.asarray(iz['near_bank_candidate'],bool);inside_all=np.concatenate([oldinside,inside]);near_all=np.concatenate([oldnear,near])
 guard_spec=importlib.util.spec_from_file_location('human_guard390',GUARD);gm=importlib.util.module_from_spec(guard_spec);guard_spec.loader.exec_module(gm);gate=gm.check_material_volume_targets(Y,U,phi_all,TARGETS)
 write_npz(OUT/'refined-background-unsolved-initializer.npz',points_world_m=Y,tetrahedra=U,phi_initializer_unsolved=phi_all,distance_outer_candidate_upper_m=np.r_[iz['distance_outer_candidate_upper_m'],dO],distance_bank_candidate_upper_m=np.r_[iz['distance_bank_candidate_upper_m'],dB],nearest_outer_face_row=np.r_[iz['nearest_outer_face_row'],ofi],nearest_outer_barycentric=np.vstack([iz['nearest_outer_barycentric'],ob]),nearest_bank_face_row=np.r_[iz['nearest_bank_face_row'],bfi],nearest_bank_barycentric=np.vstack([iz['nearest_bank_barycentric'],bb]),nearest_bank_kind_code=np.r_[iz['nearest_bank_kind_code'],FK[bfi]],inside_closed_union=inside_all,near_bank_candidate=near_all,physical_outer_node_ids=outer_nodes,construction_cap_only_node_ids=cap_only,child_source_tetrahedron_rows=parent,child_region_attributes=child_attrs)
 report.update({'status':'refined_background_with_unsolved_new_node_initializer','new_node_count':len(P),'new_node_inside_closed_union':int(inside.sum()),'new_node_near_bank_candidate':int(near.sum()),'closed_groups':closed_count,'open_groups_unsigned_distance_only':open_count,'winding_point_queries':tested,'new_node_distance_method':'same 393 k=48 physical-O and k=64 registered-bank centroid candidates followed by exact point-triangle projections on candidates; upper bounds only','field_definition':'unsolved 393 initializer semantics recomputed only for new rows; original values retained. Not a solved material field.','volume_target_necessary_gate_390':gate,'unchanged_original_physical_O_inside_witnesses':int(oldinside[outer_nodes].sum()),'limitations':['26 original physical-O nodes inside closed registered tissue-bank unions remain unchanged and block geometry admission.','No obstacle LP, field solve, level extraction, I/D surface, conforming material split, or native admission.','Candidate-centroid distances are upper bounds, not globally nearest certificates.','Registered tissue is static source-pose input, not an accepted deformed state.'],'elapsed_total_s':time.monotonic()-t0})
 (OUT/'run-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'stage':'new_node_initializer_complete','new_nodes':len(P),'inside_new':int(inside.sum()),'near_new':int(near.sum()),'closed_groups':closed_count,'open_groups':open_count,'flat_zero_tets_after':int(np.sum(np.all(phi_all[U]==0.,axis=1))),'necessary_volume_targets_pass':gate.get('necessary_conditions_pass'),'elapsed_s':time.monotonic()-t0},sort_keys=True),flush=True)
if __name__=='__main__':main()
