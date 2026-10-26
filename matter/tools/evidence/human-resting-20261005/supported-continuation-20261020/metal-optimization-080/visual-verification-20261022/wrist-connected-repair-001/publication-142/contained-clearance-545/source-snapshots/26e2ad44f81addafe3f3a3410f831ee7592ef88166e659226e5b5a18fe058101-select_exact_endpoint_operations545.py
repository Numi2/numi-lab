"""Select certified endpoint operations with exact touching-face predicates.

Each trial updates the complete connected surface. Unchanged pairs inherit a
pinned complete self-free baseline; every pair touching a changed face is tested.
The final full arithmetic/self/185-bank audit remains mandatory.
"""
from pathlib import Path
import argparse,hashlib,json,importlib.util,sys,numpy as np,time
W=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--state',required=True,type=Path);ap.add_argument('--targets',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 st=json.loads(a.state.read_text());assert sha(st['candidate'])==st['candidate_sha256'];sr=json.loads(Path(st['point_summary']).read_text());assert sr['exact_PLC_surface_pairs']==0 and sr['complete_surface_audit']
 z=np.load(st['candidate'],allow_pickle=False);t=np.load(a.targets,allow_pickle=False);rows=t['query_rows'].astype(np.int64);T=t['target_world_f32_m'];assert t['pass_mask'].all()and t['target_exactly_outside_all185'].all()
 X=z['points_world_m'];F=z['source_O_faces'];anchors=z['typed_coarse_anchor_output_rows_018'];assert not np.isin(rows,anchors).any()
 sp=importlib.util.spec_from_file_location('checks545_select',W/'run_candidate_checks545.py');c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c);owner=c.mod(c.OWNER);R0,_=owner.compose_source_reference(z,c.R0)
 cb=np.cross(X[F[:,1]].astype(float)-X[F[:,0]],X[F[:,2]].astype(float)-X[F[:,0]]);cb2=np.einsum('ij,ij->i',cb,cb)
 ref=z['area_reference_037_points_world_m_f64'];cr=np.cross(ref[F[:,1]]-ref[F[:,0]],ref[F[:,2]]-ref[F[:,0]]);crmag=np.linalg.norm(cr,axis=1)
 E=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0);L=np.linalg.norm(X[E[:,0]].astype(float)-X[E[:,1]],axis=1)
 pred=c.ROOT/'../x';EVID=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
 pred=EVID/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py';assert sha(pred)=='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
 sys.path.insert(0,str(EVID/'incremental-exact-audit-owner-253/src'));sp=importlib.util.spec_from_file_location('numilab_human._select545',pred);p=importlib.util.module_from_spec(sp);sys.modules[sp.name]=p;sp.loader.exec_module(p)
 def arithmetic(Y):
  d=Y.astype(float)-X.astype(float);cc=np.cross(Y[F[:,1]].astype(float)-Y[F[:,0]],Y[F[:,2]].astype(float)-Y[F[:,0]])
  badf=np.flatnonzero((np.einsum('ij,ij->i',cc,cb)/cb2<.1)|(np.linalg.norm(cc,axis=1)/crmag<.1));bade=np.flatnonzero(np.max(np.abs(d[E[:,0]]-d[E[:,1]]),axis=1)>.5*L+1e-11)
  badn=np.flatnonzero((np.linalg.norm(d,axis=1)>.001000001)|(np.linalg.norm(Y.astype(float)-R0,axis=1)>.012000001))
  return dict(bad_area_faces=badf.tolist(),bad_edge_rows=bade.tolist(),bad_bound_nodes=badn.tolist()),not(len(badf)or len(bade)or len(badn))
 lengths=np.linalg.norm(T.astype(float)-X[rows].astype(float),axis=1);orders=[np.lexsort((rows,lengths)),np.lexsort((rows,-lengths)),np.argsort(rows)]
 best=None;trials=[];start=time.monotonic()
 for oi,order in enumerate(orders):
  Y=X.copy();kept=[]
  for i in order:
   n=int(rows[i]);Q=Y.copy();Q[n]=T[i];ar,ok=arithmetic(Q);rec=dict(order=oi,node=n,arithmetic=ar,accepted=False)
   if ok:
    touched=set(map(int,np.flatnonzero(np.any(F==n,axis=1))));records,_=p.float32_surface_records(Q,F,source_seam_groups=());r=p._audit_self_pairs_touching_faces(records,touched);rec['self_pairs']=r['triangle_pairs'];rec['complete_pair_scope']=dict(changed_faces=sorted(touched),complete_surface_faces=len(F))
    if not r['triangle_pairs']:Y=Q;kept.append(n);rec['accepted']=True
   trials.append(rec)
  if best is None or len(kept)>len(best[1]):best=(Y,kept,oi)
 a.out.mkdir(parents=True,exist_ok=False);Y,kept,order=best;np.savez_compressed(a.out/'candidate.npz',**{**{k:z[k]for k in z.files},'points_world_m':Y})
 r=dict(status='CONNECTED_ENDPOINT_SELECTION_PENDING_FULL_EXACT_CHECKS',candidate_sha256=sha(a.out/'candidate.npz'),base_sha256=st['candidate_sha256'],source_sha256=sha(__file__),target_sha256=sha(a.targets),retained_node_rows=kept,retained_operations=len(kept),selected_order=order,trials=trials,seconds=time.monotonic()-start,native_admission=False,accepted_physical_steps=0)
 (a.out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ['retained_operations','candidate_sha256','seconds']}))
if __name__=='__main__':main()
