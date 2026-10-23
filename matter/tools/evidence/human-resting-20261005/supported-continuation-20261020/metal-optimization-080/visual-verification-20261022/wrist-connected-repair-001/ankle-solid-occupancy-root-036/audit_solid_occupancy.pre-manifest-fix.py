#!/usr/bin/env python3
"""Offline exact signed winding of candidate021 ankle vertices in complete closed source bones."""
from pathlib import Path
import importlib.util,sys,json,hashlib,time
import numpy as np
E=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearby',E/'ankle-coupled-geometry-weight-021/audit_021_nearby_all_bones.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from numilab_human import common_atlas_skin_clearance as cc,cardiac_cavity_intersections as ci
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
Z=np.load(m.NPZ);G=np.load(E/'finger-profile-069i-left-ring-web-retreat/replacement-geometry.npz');F=G['faces'].astype(np.int64);faces=Z['changed_source_faces'].astype(np.int64);ids=np.unique(F[faces]);MAN,raw,records,bv,bi,anchors=m.audit.load_bones()
cases={'reference':E/'native-registered-skin-pose-clean-014/registered-reference-elevation-010-014','dorsiflexion':E/'native-registered-joint-reference-pairs-016/ankle-dorsiflexion-020-ref010','plantarflexion':E/'native-registered-joint-reference-pairs-016/ankle-plantarflexion-020-ref010'}
report={'candidate_sha256':sha(m.NPZ),'bone_manifest_sha256':sha(m.MAN),'bone_payload_sha256':sha(raw),'method':'Existing exact Float32 lattice signed winding against complete source-bone triangles; coordinate-identical seams identified before closure/orientation validation. Queries are all vertices incident to changed skin faces whose coordinates lie inside each complete bone AABB. Outside-AABB points are rigorously outside. Triangle crossing checks remain a separate gate.','poses':{},'limits':['Three static captured poses only; no FEM motion or admission.','Signed winding requires closed oriented topology but this receipt does not prove each bone shell is self-intersection free.','Source069i-to294 exact mapped patch equivalence is a separate retained receipt.']}
started=time.time()
for label,folder in cases.items():
 pose=json.loads((folder/'registered-skin-poses.json').read_text());V=np.asarray(Z[label+'_positions_m'],np.float32);points=V[ids];body={int(x['body_index']):x for x in pose['bodies']};out=[]
 for rec,a in zip(records,anchors):
  bodyid,fv,nv,fi,ni,stable,tx,ty,tz,qx,qy,qz,qw,scale,_=rec;cur=body[int(bodyid)]['current'];Rb=m.audit.mat((qx,qy,qz,qw));Rp=m.audit.mat(cur['orientation_world_xyzw']);local=(bv[fv:fv+nv].astype(np.float64)*scale)@Rb.T+np.array((tx,ty,tz));world=(local@Rp.T+np.asarray(cur['position_world_m'])).astype(np.float32);ff=bi[fi//3:(fi+ni)//3].astype(np.int64)-int(fv)
  lower=world.min(0);upper=world.max(0);query=np.flatnonzero(np.all(points>=lower,axis=1)&np.all(points<=upper,axis=1))
  if not len(query):continue
  qv,inv=np.unique(world,axis=0,return_inverse=True);qf=inv[ff];row={'body':str(a['myosim_body']),'bone_stable_id':int(stable),'source_vertices':int(nv),'unique_exact_coordinates':len(qv),'triangles':len(qf),'query_count':len(query),'inside':[],'boundary':[],'indeterminate':[]}
  try:prepared=ci.prepare_signed_winding(cc._exact_surface_records(qv,qf));row['closed_oriented_topology']=True
  except Exception as e:row.update(closed_oriented_topology=False,error=str(e));out.append(row);continue
  winding_counts={}
  for qi in query:
   result=ci.signed_winding_number(cc.float32_point_lattice_key(points[qi]),prepared);status=result['status'];w=result['winding_number'];key=str(w) if status=='resolved' else status;winding_counts[key]=winding_counts.get(key,0)+1
   if status!='resolved' or w!=0:
    key='inside' if status=='resolved' else status
    row[key if key in row else 'indeterminate'].append({'source_vertex':int(ids[qi]),'position_m':points[qi].tolist(),'result':result})
  row['winding_counts']=winding_counts;out.append(row);print(json.dumps({'pose':label,'body':row['body'],'bone':int(stable),'queries':len(query),'windings':winding_counts}),flush=True)
 report['poses'][label]={'pose_sha256':sha(folder/'registered-skin-poses.json'),'patch_vertices':len(ids),'bones_with_AABB_candidates':out}
 report['elapsed_seconds']=time.time()-started
 (OUT/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
report['passed']=all(r.get('closed_oriented_topology') and not r['inside'] and not r['boundary'] and not r['indeterminate'] for p in report['poses'].values() for r in p['bones_with_AABB_candidates'])
p=OUT/'solid-occupancy.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'receipt':str(p),'sha256':sha(p),'elapsed_seconds':report['elapsed_seconds']}),flush=True)
