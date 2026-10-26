"""Select certified endpoint operations with exact touching-face predicates.

Each trial updates the complete connected surface. Unchanged pairs inherit a
pinned complete self-free baseline; every pair touching a changed face is tested.
The final full arithmetic/self/185-bank audit remains mandatory.
"""
from pathlib import Path
import argparse,hashlib,json,importlib.util,sys,numpy as np,time,csv,subprocess,tempfile
W=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_moved_points(Y,changed,old_inside,cleared,bank,binary,root):
 root.mkdir(parents=True,exist_ok=True);points=root/'points.csv'
 with points.open('w')as out:
  writer=csv.writer(out,lineterminator="\n");writer.writerow(['query_row','x_m','y_m','z_m','x_bits_u32','y_bits_u32','z_bits_u32'])
  for i,node in enumerate(changed):writer.writerow([i,*[format(float(v),'.9g')for v in Y[node]],*[int(v)for v in Y.view(np.uint32)[node]]])
 args=[str(binary),str(bank/'exact-emitted.triangles.csv'),str(bank/'exact-emitted.bodies.csv'),str(points),str(root/'pairs.csv'),str(root/'nodes.csv'),'185']
 cp=subprocess.run(args,capture_output=True,text=True);(root/'query.log').write_text(cp.stdout+cp.stderr)
 if cp.returncode:raise RuntimeError('point query failed')
 ns=list(csv.DictReader((root/'nodes.csv').open()));assert len(ns)==len(changed)
 for i,r in enumerate(ns):
  assert int(r['query_row'])==i and int(r['total_bone_banks'])==185 and int(r['ambiguous_banks'])==0 and int(r['invalid_banks'])==0
  assert sum(int(r[k])for k in ['aabb_proven_outside_banks','parity_outside_banks','inside_banks','ambiguous_banks','invalid_banks'])==185
 pairs=list(csv.DictReader((root/'pairs.csv').open()));assert all(r['classification']in('inside','outside')and int(r['ambiguous'])==0 for r in pairs)
 inside={(int(changed[int(r['query_row'])]),int(r['bank_body']),int(r['bank_stable_id']))for r in pairs if r['classification']=='inside'}
 allowed={key for key in old_inside if key[0]not in cleared}
 return inside<=allowed,dict(new_or_reentered_pairs=sorted(inside-allowed),inside_moved_pairs=sorted(inside),moved_point_rows=changed.tolist(),all185_banks_accounted=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spatial',action='store_true');ap.add_argument('--protect-new-inside',action='store_true');ap.add_argument('--state',required=True,type=Path);ap.add_argument('--targets',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 st=json.loads(a.state.read_text());assert sha(st['candidate'])==st['candidate_sha256'];sr=json.loads(Path(st['point_summary']).read_text());assert sr['exact_PLC_surface_pairs']==0 and sr['complete_surface_audit']
 a.out.mkdir(parents=True,exist_ok=False)
 old_inside={(int(r['output_node']),int(r['bank_body']),int(r['bank_stable_id']))for r in sr['inside_witnesses']}
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
 sp=importlib.util.spec_from_file_location('harmonic545_select',W/'build_connected_endpoint_patch545.py');harm=importlib.util.module_from_spec(sp);sp.loader.exec_module(harm)
 bank=EVID/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight';binary=Path('/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query');assert sha(binary)=='5dc7b5ce2a96f58bd03ccc85e38645d9da2ec8be105a532c461a3e0cc6d25460'
 lengths=np.linalg.norm(T.astype(float)-X[rows].astype(float),axis=1);orders=[np.lexsort((rows,lengths)),np.lexsort((rows,-lengths)),np.argsort(rows)]
 best=None;trials=[];start=time.monotonic()
 for oi,order in enumerate(orders[:1]):
  Y=X.copy();kept=[]
  for i in order:
   n=int(rows[i]);found=False
   for rings in (1,2,3,4):
    if a.spatial:
     radius=(1,2,4,8)[rings-1]*.001;u=np.minimum(1.,np.linalg.norm(Y.astype(float)-Y[n].astype(float),axis=1)/radius);weight=(1-u)**4*(4*u+1)
     Q=(Y.astype(float)+weight[:,None]*(T[i].astype(float)-Y[n].astype(float))).astype(np.float32);Q[n]=T[i];Q[anchors]=Y[anchors];Q[np.asarray(kept,dtype=np.int64)]=Y[np.asarray(kept,dtype=np.int64)]
    else:Q,pr=harm.solve_patch(Y,F,np.asarray([n]),T[i:i+1],anchors,np.asarray(kept,dtype=np.int64),rings)
    changed=np.flatnonzero(np.any(Q!=Y,axis=1));ar,ok=arithmetic(Q);rec=dict(order=oi,node=n,rings=rings,spatial_radius_mm=((1,2,4,8)[rings-1]if a.spatial else None),arithmetic=ar,accepted=False,changed_nodes=changed.tolist())
    if ok:
     touched=set(map(int,np.flatnonzero(np.isin(F,changed).any(axis=1))));records,_=p.float32_surface_records(Q,F,source_seam_groups=());r=p._audit_self_pairs_touching_faces(records,touched);rec['self_pairs']=r['triangle_pairs'];rec['complete_pair_scope']=dict(changed_faces=sorted(touched),complete_surface_faces=len(F))
     if not r['triangle_pairs']:
      outside,qr=check_moved_points(Q,changed,old_inside,set(kept+[n]),bank,binary,a.out/f'query-{len(trials):03d}');rec['point_query']=qr
      rec['new_inside_protection_trials']=[]
      if not outside and a.protect_new_inside:
       for repair in range(4):
        bad=np.unique(np.asarray(qr['new_or_reentered_pairs'],dtype=np.int64).reshape(-1,3)[:,0])
        if n in bad:break
        Q[bad]=Y[bad];ar,okay=arithmetic(Q);rr=dict(fixed_outside_witness_rows=bad.tolist(),arithmetic=ar)
        if not okay:rec['new_inside_protection_trials'].append(rr);break
        ch=np.flatnonzero(np.any(Q!=Y,axis=1));faces=set(map(int,np.flatnonzero(np.isin(F,ch).any(axis=1))));records,_=p.float32_surface_records(Q,F,source_seam_groups=());res=p._audit_self_pairs_touching_faces(records,faces);rr['self_pairs']=res['triangle_pairs']
        if res['triangle_pairs']:rec['new_inside_protection_trials'].append(rr);break
        outside,qr=check_moved_points(Q,ch,old_inside,set(kept+[n]),bank,binary,a.out/f'query-{len(trials):03d}-protect-{repair}');rr['point_query']=qr;rec['new_inside_protection_trials'].append(rr)
        if outside:break
      if outside:Y=Q;kept.append(n);rec['accepted']=True;found=True
    trials.append(rec)
    if found:break
   (a.out/'progress.json').write_text(json.dumps(dict(tried_node=n,retained_operations=len(kept),trials=len(trials)))+'\n')
  if best is None or len(kept)>len(best[1]):best=(Y,kept,oi)
 Y,kept,order=best;np.savez_compressed(a.out/'candidate.npz',**{**{k:z[k]for k in z.files},'points_world_m':Y})
 r=dict(status='CONNECTED_HARMONIC_ENDPOINT_SELECTION_PENDING_FULL_EXACT_CHECKS',candidate_sha256=sha(a.out/'candidate.npz'),base_sha256=st['candidate_sha256'],source_sha256=sha(__file__),target_sha256=sha(a.targets),retained_node_rows=kept,retained_operations=len(kept),selected_order=order,trials=trials,seconds=time.monotonic()-start,native_admission=False,accepted_physical_steps=0)
 (a.out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ['retained_operations','candidate_sha256','seconds']}))
if __name__=='__main__':main()
