from pathlib import Path
import json,hashlib,csv,sys,time,collections
import numpy as np
from scipy import sparse
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-volume-continuation-040/iteration-005');M=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-volume-continuation-040/canonical/matter/tools')
sys.path.insert(0,str(M));import human_skin_fat_connected_repair as repair
sys.path.insert(0,str(E/'route-audit-counter-fix-030/canonical/matter/tools'));import human_skin_fat_bone_audit as audit
AP=E/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz';REF=E/'bone-clearance-fit-040-shared-columns/attempt-004/fit-input.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(AP)=='39d110a33b5c997575e75f2447ab14a2416a7d52d30bb369414b82d0416220cf'
assert sha(REF)=='b248bf0ee823ca1ab3b6d787fe3d427d8e080dcd528a57421235c7d6e1ed375e'
ap=np.load(AP);ref=np.load(REF);X=np.load('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-volume-continuation-040/iteration-004/candidate.npz')['points_world_f32'];T=ap['tetrahedra'];R=ref['reference_points_m'];mat=ap['material_ids']
assert np.array_equal(T,ref['tetrahedra'])and np.array_equal(mat,ref['material_ids'])
tri,ids,gids,bodies,rot,off,pos,keys,pins,exact=audit.read_bank_inputs(E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight',E/'route-audit-counter-fix-030/canonical/matter')
assert np.array_equal(gids,ap['global_nodes'])
query=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-volume-continuation-040/iteration-004/native-query/source-preflight')
qkeys,qmap,err,qpins=audit.load_matched_query(query,gids,X,bodies,rot,off,pos,keys[:,0])
native=list(csv.DictReader((query/'all-bank-containment.csv').open()))
report=json.loads((Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-volume-continuation-040/iteration-004/exact-audit/tet-edge-report.json')).read_text())
groups=collections.defaultdict(list)
for w in report['witnesses']:groups[(w['edge_index'],w['body'],w['bone_stable_id'])].append(w)
lookup={(int(r[0]),int(r[1]),int(r[3])):k for k,r in enumerate(ids)}
rows=[];targets=[];provenance=[]
def add(nodes,weights,n,target,pro):
 row=len(targets)
 for v,w in zip(nodes,weights):
  for k in range(3):rows.append((row,3*int(v)+k,float(w*n[k])))
 targets.append(target);provenance.append(pro)
for key,ws in sorted(groups.items()):
 ws.sort(key=lambda w:w['exact_result']['t']);a,b=ws[0]['endpoint_local_nodes'];body,bone=key[1:]
 ca=qmap.get((int(gids[a]),body,bone),'outside_AABB_proven');cb=qmap.get((int(gids[b]),body,bone),'outside_AABB_proven')
 outside=lambda c:c.startswith('outside')
 if outside(ca):
  spans=[(ws[j],ws[j+1]['exact_result']['t']if j+1<len(ws)else 1.)for j in range(0,len(ws),2)]
  toward=X[a].astype(float)
 elif outside(cb):
  spans=[(ws[-1],0.)];toward=X[b].astype(float)
 else:continue
 for w,tend in spans:
  v=tri[lookup[(body,bone,w['triangle_stable_id'])]]
  bary=np.asarray(w['exact_result']['barycentric']);hlocal=bary@v
  h=pos[body]+np.linalg.solve(rot[body],hlocal+off[body])
  nlocal=np.cross(v[1]-v[0],v[2]-v[0]);n=rot[body].T@nlocal;n/=np.linalg.norm(n)
  if n@(toward-h)<0:n=-n
  tm=.5*(w['exact_result']['t']+tend)
  add([a,b],[1-tm,tm],n,float(n@h+5e-5),{'kind':'exact_bone_intrusion_segment','edge':key[0],'body':body,'bone':bone,'triangle':w['triangle_stable_id'],'t':tm,'normal_side_proven_at_baseline':ca if outside(ca)else cb})
for r in native:
 if r['classification']!='inside':continue
 node=int(r['local_node']);body=int(r['bank_body']);bone=int(r['bank_stable_id'])
 p=np.array([float(r['query_local_'+c])for c in 'xyz']);hlocal=np.array([float(r['closest_'+c])for c in 'xyz'])
 n=rot[body].T@(hlocal-p);n/=np.linalg.norm(n)
 h=pos[body]+np.linalg.solve(rot[body],hlocal+off[body])
 add([node],[1],n,float(n@h+5e-5),{'kind':'native_inside_vertex_exit','node':node,'body':body,'bone':bone})
rr,cc,vv=zip(*rows);C=sparse.coo_matrix((vv,(rr,cc)),shape=(len(targets),3*len(X))).tocsr();target=np.asarray(targets)
seeds=np.unique(C.indices//3);patch=seeds.copy()
# One complete tetrahedral neighbour ring; exact untouched boundary remains anchored.
patch=np.unique(T[np.isin(T,patch).any(axis=1)])
L=sparse.load_npz(E/'connected-volume-repair-032/layer-rows.npz');lb=np.load(E/'connected-volume-repair-032/repair-input.npz')['layer_targets']
(D/'contact-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
sparse.save_npz(D/'contact-rows.npz',C);sparse.save_npz(D/'layer-rows.npz',L)
np.savez_compressed(D/'repair-input.npz',points_world_f32=X,reference_points_m=R,tetrahedra=T,material_ids=mat,free_nodes=patch,contact_targets=target,layer_targets=lb,global_nodes=gids)
start=time.monotonic();history=[];Y=X.copy()
for it in range(4):
 Y2,r=repair.coupled_clearance_step(Y,R,T,patch,C,target,step_limit_mm=2.,time_limit_s=90.,lower_linear_matrix=L,lower_linear_targets=lb)
 r['iteration']=it;history.append(r);print(json.dumps(r),flush=True)
 if not r.get('step_applied'):break
 Y=Y2
 # End the fixed-witness construction after one effective step. Refresh exact
 # witnesses before any further solve; never retain obsolete separator choices.
 break
np.savez_compressed(D/'candidate.npz',points_world_f32=Y,tetrahedra=T,material_ids=mat)
out={'status':'CONNECTED_STATIC_CONSTRUCTION_NOT_PHYSICAL_ADMISSION','source_sha256':sha(Path(__file__)),'solver_source_sha256':sha(M/'human_skin_fat_connected_repair.py'),'inputs':{str(AP):sha(AP),str(REF):sha(REF)},'history':history,'elapsed_s':time.monotonic()-start,'changed_nodes':int(np.any(Y.view(np.uint32)!=X.view(np.uint32),axis=1).sum()),'candidate_sha256':sha(D/'candidate.npz'),'retained_reference_sha256':hashlib.sha256(R.tobytes()).hexdigest(),'accepted_physical_steps':0,'physical_qualification':False,'clearance_qualified':False,'limitations':['Static asset construction; no simulation state advanced.','Actual exact-edge, containment, surface, mass and detailed-source gates remain mandatory.']}
(D/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

