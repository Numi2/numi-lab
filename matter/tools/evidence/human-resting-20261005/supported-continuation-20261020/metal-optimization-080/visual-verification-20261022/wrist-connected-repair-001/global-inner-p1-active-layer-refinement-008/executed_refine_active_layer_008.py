from pathlib import Path
import sys, importlib.util, json, hashlib, time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=E/'global-inner-p1-active-layer-refinement-008'
P=E/'global-inner-p1-edge-support-403/quality-coordinate-005/candidate.npz'
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
r=load(E/'global-inner-p1-edge-support-403/refine_internal_o_edges_403.py','refine403')
a=load(E/'global-inner-p1-field-owner-integration-007/field_owner_integration_007.py','owner007')
c=load(E/'robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py','clip302')
assert hashlib.sha256(P.read_bytes()).hexdigest()=='5b54ffd34d74d717579fcb14677cbf2bfba01f4d4562652605b9207e65f8f698'
if OUT.exists():raise RuntimeError('refuse overwrite')
OUT.mkdir()
z=np.load(P);base={k:z[k] for k in z.files}
X=base['points_world_m'];T=base['tetrahedra'].astype(np.int64);O=base['physical_outer_node_ids'];shell=base['physical_shell_faces']
X0=X.copy();T0=T.copy();D0,Q0=r.tetra_metrics(X,T);parent=np.arange(len(T))
ptr=base['node_source_indptr'];ids=base['node_source_indices'];weights=base['node_source_weights']
bank=np.load(E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz');tri=bank['bank_0004_vertices'][bank['bank_0004_faces'][2562]]
bedges={tuple(sorted((int(f[i]),int(f[j])))) for f in shell for i,j in ((0,1),(0,2),(1,2))}
records=[];allpairs=[];started=time.monotonic()
for level in range(5):
 cons,polys,meta=a.clip_typed_triangle_to_mesh(X,T,tri,c.triangle_tet_barycentric_polygon)
 caps=a.fixed_zero_capacity_report(T,cons,O,len(X))
 bad=[x for x in caps if x['max_free_capacity']<.341-1e-12]
 D,Q=r.tetra_metrics(X,T)
 boundary=r.compare_oriented_boundary(shell,r.oriented_boundary_faces(T))
 item={'round':level,'nodes':len(X),'tets':len(T),'constraints':len(caps),'bad_capacity':len(bad),'min_capacity':min(x['max_free_capacity'] for x in caps),'min_q':float(Q.min()),'bad_q':int(np.sum(Q<1e-5)),'nonpositive':int(np.sum(D<=0)),'boundary':boundary}
 print(json.dumps(item),flush=True);records.append(item)
 np.savez_compressed(OUT/('round-%03d.npz'%level),points_world_m=X,tetrahedra=T,parent_005_tet_rows=parent,node_source_indptr=ptr,node_source_indices=ids,node_source_weights=weights,physical_outer_node_ids=O,physical_shell_faces=shell,region_attributes=base['region_attributes'][parent])
 if not bad:break
 if level==4:break
 if min(x['max_free_capacity'] for x in bad)<=1e-10:
  item['stop_reason']='true_zero_fixed_support';break
 fixed=np.zeros(len(X),bool);fixed[O]=1
 selected=set()
 for ti in sorted(set(x['tet_row'] for x in bad)):
  for edge in r.tet_edges(T[ti]):
   if fixed[edge[0]] != fixed[edge[1]] and edge not in bedges:selected.add(edge)
 if not selected:item['stop_reason']='no_internal_O_free_edges';break
 item['selected_edges']=len(selected)
 oldN=len(X);X,T,p,ep=r.split_edges_globally(X,T,selected)
 ptr,ids,weights,anc=r.append_midpoint_source_csr(ptr,ids,weights,ep)
 parent=parent[p];allpairs.append(ep)
 if len(T)-len(T0)>2500:
  item['stop_reason']='2500_added_tet_budget';break
report={'input_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'records':records,'final_support_pass':not bad,'candidate_admission':False,'scope':'one typed face only; adaptive internal O/free midpoint subdivision; no outer-boundary movement or target relaxation','elapsed_s':time.monotonic()-started}
(OUT/'exploration-report.json').write_text(json.dumps(report,indent=2)+'\n')

