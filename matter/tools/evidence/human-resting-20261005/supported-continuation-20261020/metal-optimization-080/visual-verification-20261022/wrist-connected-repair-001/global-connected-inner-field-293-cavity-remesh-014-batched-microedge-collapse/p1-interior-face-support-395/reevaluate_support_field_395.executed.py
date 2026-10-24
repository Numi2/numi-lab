#!/usr/bin/env python3
import collections,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BASE=ROOT/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse'; OUT=BASE/'p1-interior-face-support-395'
CAND=BASE/'candidate.npz'; INIT=BASE/'p1-distance-initializer-393/p1-distance-initializer-unsolved.npz'; CK=OUT/'refined-background-checkpoint.npz'
BANK=ROOT/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz'; INV=ROOT/'registered-surrounding-tissue-native013-191/inventory.json'; HELPER=ROOT/'ankle-parent-tet-refinement-026/build_common_volume_126.py'; WINDHELP=ROOT/'ankle-parent-tet-refinement-026/generated-224-commonfield-tetgen-volume-actual007/build_commonfield_224.py'; GUARD=ROOT/'p1-material-field-attainability-owner-390/tools/human_skin_p1_material_volume.py'
POSE=ROOT/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
TARGETS={'skin':0.003452570642,'skin_plus_fat':0.015104913881}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_functions(path,names,extra):
 import ast
 tree=ast.parse(Path(path).read_text()); fs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 if {n.name for n in fs}!=set(names): raise RuntimeError(f'missing {names} in {path}')
 ns=dict(extra); exec(compile(ast.Module(body=fs,type_ignores=[]),str(path),'exec'),ns); return {n:ns[n] for n in names}
def closest(P,V,F,k,cb,chunk=192):
 centers=np.concatenate([V[F[i:i+100000]].astype(np.float64).mean(axis=1) for i in range(0,len(F),100000)])
 tree=cKDTree(centers); n=len(P); dist=np.empty(n); face=np.empty(n,np.int64); bary=np.empty((n,3)); point=np.empty((n,3))
 for lo in range(0,n,chunk):
  pp=np.asarray(P[lo:lo+chunk],np.float64); _,ix=tree.query(pp,k=min(k,len(F)),workers=-1)
  if ix.ndim==1: ix=ix[:,None]
  tris=V[F[ix]].astype(np.float64); bb,ds=cb(pp,tris); kk=np.argmin(ds,axis=1); rr=np.arange(len(pp)); ids=ix[rr,kk]
  bary[lo:lo+len(pp)]=bb[rr,kk];face[lo:lo+len(pp)]=ids;dist[lo:lo+len(pp)]=np.sqrt(ds[rr,kk]);point[lo:lo+len(pp)]=np.einsum('ni,nij->nj',bary[lo:lo+len(pp)],V[F[ids]].astype(np.float64))
 return dist,face,bary,point
def write_npz(path,**kw):
 tmp=Path(str(path)+'.tmp.npz');np.savez_compressed(tmp,**kw);tmp.replace(path)
def main():
 t0=time.monotonic(); hashes={'candidate014':sha(CAND),'initializer393':sha(INIT),'support_checkpoint':sha(CK),'banks689':sha(BANK),'inventory689':sha(INV),'closest_helper126':sha(HELPER),'winding_helper224':sha(WINDHELP),'volume_guard390':sha(GUARD),'pose157':sha(POSE)}
 exp={'candidate014':'e70b3e08aa385c8a321ebb0849257ec4afa04d13050e26fc28647d36146d26ba','initializer393':'887eb4fb47bf47144397ae41cf77394763a2d1e120a977ba2a8176e4d3ed9740','banks689':'39deb100a99601dc48550c3e5b260cd75d1e35b8d2682b67bfc?','inventory689':'b0823cf8f33df2586ba1cd45e5b260cd75d1e35b8d2682b67c15f64ea22bbcc7','pose157':'0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd'}
 # Hash spelling for bank cache is pinned in executed 395 builder; verify using its literal.
 builder=OUT/'build_refined_face_support_395.py'; text=builder.read_text(); import re
 m=re.search(r"'banks689':'([0-9a-f]{64})'",text)
 if not m: raise RuntimeError('cannot recover pinned banks689 digest')
 exp['banks689']=m.group(1)
 for k,v in exp.items():
  if hashes.get(k)!=v: raise RuntimeError(f'input pin mismatch {k}: {hashes.get(k)} != {v}')
 c=np.load(CAND,allow_pickle=False); iz=np.load(INIT,allow_pickle=False); q=np.load(CK,allow_pickle=False)
 X=np.asarray(c['points_world_m'],np.float32);T=np.asarray(c['tetrahedra'],np.int64);Y=np.asarray(q['points_world_m'],np.float32);U=np.asarray(q['tetrahedra'],np.int64);N=len(X)
 if not(np.array_equal(X,iz['points_world_m']) and np.array_equal(T,iz['tetrahedra']) and np.array_equal(Y[:N],X)): raise RuntimeError('old-prefix or initializer geometry mismatch')
 P=Y[N:].astype(np.float64);invj=json.loads(INV.read_text()); banks=np.load(BANK,allow_pickle=False); vparts=[];fparts=[];fbank=[];groups=collections.defaultdict(list);vb=0
 for bi,m in enumerate(invj['banks']):
  vv=np.asarray(banks[f'bank_{bi:04d}_vertices'],np.float32);ff=np.asarray(banks[f'bank_{bi:04d}_faces'],np.int64)
  vparts.append(vv);fparts.append(ff+vb);vb+=len(vv);fbank.append(np.full(len(ff),bi,np.int32));groups[tuple(sorted(map(int,m['body_indices'])))].append((vv,ff))
 BV=np.concatenate(vparts);BF=np.concatenate(fparts);FB=np.concatenate(fbank);kind=np.asarray([0 if m['kind']=='anatomy' else 1 for m in invj['banks']],np.uint8);FK=kind[FB]
 cb=load_functions(HELPER,{'closest_bary'},{'np':np})['closest_bary'];dB,bfi,bb,bq=closest(P,BV,BF,64,cb)
 OF=np.asarray(c['physical_outer_faces'],np.int64);dO,ofi,ob,oq=closest(P,Y,OF,48,cb)
 near=dB<=np.where(FK[bfi]==0,.00025,.00010);inside=np.zeros(len(P),bool);closed_count=open_count=tested=0
 wind_api=load_functions(WINDHELP,{'consistent_closed','solid_angle_winding'},{'np':np,'Counter':collections.Counter,'math':__import__('math')});CLOSE,WIND=wind_api['consistent_closed'],wind_api['solid_angle_winding']
 for body_ids,items in groups.items():
  vvparts=[];ffparts=[];off=0
  for vv,ff in items:vvparts.append(vv);ffparts.append(ff+off);off+=len(vv)
  Vcat=np.concatenate(vvparts);Fcat=np.concatenate(ffparts);Vg,iv=np.unique(Vcat,axis=0,return_inverse=True);Fg=iv[Fcat];closed,_=CLOSE(Fg)
  if not closed:open_count+=1;continue
  closed_count+=1;mn=Vg.min(0).astype(float)-1e-6;mx=Vg.max(0).astype(float)+1e-6;ids=np.flatnonzero(np.all((P>=mn)&(P<=mx),axis=1))
  if len(ids):
   w=WIND(P[ids],Vg,Fg);inside[ids[np.abs(w)>.5]]=True;tested+=len(ids)
 phi0=np.asarray(iz['phi_initializer_unsolved'],np.float64);phi_all=np.asarray(q['phi_p1_prolonged_unsolved'],np.float64).copy()
 if len(phi_all)!=len(Y):raise RuntimeError('checkpoint phi row count mismatch')
 phi_new=np.clip(np.where(inside|near,1.,dO/np.maximum(dO+np.maximum(dB,1e-9),1e-9)),0.,1.);phi_all[N:]=phi_new;phi_all[:N]=phi0
 old_inside=np.asarray(iz['inside_closed_union'],bool);old_near=np.asarray(iz['near_bank_candidate'],bool);inside_all=np.r_[old_inside,inside];near_all=np.r_[old_near,near]
 guard_spec=importlib.util.spec_from_file_location('guard390',GUARD);gm=importlib.util.module_from_spec(guard_spec);guard_spec.loader.exec_module(gm);gate=gm.check_material_volume_targets(Y,U,phi_all,TARGETS)
 F=np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),np.int64);nzero=int(np.sum(np.all(phi_all[U]==0.,axis=1)))
 oldO=np.asarray(q['physical_outer_node_ids'],np.int64);oldInsideO=int(np.sum(old_inside[oldO]));cap_only=np.setdiff1d(np.unique(c['construction_closure_faces']),oldO);new_cell_zero=int(nzero)
 out=OUT/'refined-background-unsolved-initializer.npz'
 write_npz(out,points_world_m=Y,tetrahedra=U,phi_initializer_unsolved=phi_all,distance_outer_candidate_upper_m=np.r_[iz['distance_outer_candidate_upper_m'],dO],distance_bank_candidate_upper_m=np.r_[iz['distance_bank_candidate_upper_m'],dB],nearest_outer_face_row=np.r_[iz['nearest_outer_face_row'],ofi],nearest_outer_barycentric=np.vstack((iz['nearest_outer_barycentric'],ob)),nearest_bank_face_row=np.r_[iz['nearest_bank_face_row'],bfi],nearest_bank_barycentric=np.vstack((iz['nearest_bank_barycentric'],bb)),nearest_bank_kind_code=np.r_[iz['nearest_bank_kind_code'],FK[bfi]],inside_closed_union=inside_all,near_bank_candidate=near_all,physical_outer_node_ids=q['physical_outer_node_ids'],construction_cap_only_node_ids=cap_only,child_source_tetrahedron_rows=q['child_source_tetrahedron_rows'],child_region_attributes=q['child_region_attributes'])
 report={'schema':'numi.human.p1-support-reevaluated-unsolved-initializer.395.v1','status':'unsolved_initializer_only','native_admission':False,'input_sha256':hashes,'output':str(out),'output_sha256':sha(out),'mesh':{'nodes':len(Y),'tets':len(U),'new_support_nodes':len(P),'original_nodes_bitwise_unchanged':True,'physical_O_faces_bitwise_unchanged':True},'new_support_initializer':{'inside_closed_union':int(inside.sum()),'near_bank_candidate':int(near.sum()),'phi_quantiles':np.quantile(phi_new,[0,.01,.1,.5,.9,.99,1]).tolist(),'phi_exact_zero':int(np.sum(phi_new==0)),'phi_positive':int(np.sum(phi_new>0)),'closed_body_groups':closed_count,'open_body_groups_distance_only':open_count,'winding_point_queries':tested,'distance_method':'same 393 k=48 physical-O and k=64 bank-centroid candidates followed by exact point-triangle projection among those candidates; distances are candidate upper bounds, not globally nearest certificates'},'plateau':{'allzero_P1_tets_after_new_support_values':new_cell_zero,'allzero_old_014_tets_before':6041,'physical_O_nodes_inside_closed_union_unchanged':oldInsideO,'all_old_O_rows_retained':True},'volume_target_necessary_guard390':gate,'limitations':['The original 26 physical-O inside-union witnesses are preserved; this does not clear the source boundary.','No sparse obstacle LP, material extraction, conservative mass transfer, or physical/native admission was run.','Initializer values are not a solved field.','Distance nearest-face IDs are approximate centroid-candidate upper bounds.','All 689 static source banks are considered from the pinned cache; source pose is not deformed-state anatomy.'],'elapsed_s':time.monotonic()-t0}
 (OUT/'initializer-reassessment-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'stage':'new_node_field_reinitialized','new_nodes':len(P),'inside_new':int(inside.sum()),'near_new':int(near.sum()),'zeros_new':int(np.sum(phi_new==0)),'allzero_tets_after':new_cell_zero,'original_zero_tets':6041,'unchanged_O_inside':oldInsideO,'guard_pass':gate.get('necessary_conditions_pass'),'guard':gate,'output_sha':report['output_sha256'],'elapsed_s':time.monotonic()-t0},sort_keys=True),flush=True)
if __name__=='__main__':main()
