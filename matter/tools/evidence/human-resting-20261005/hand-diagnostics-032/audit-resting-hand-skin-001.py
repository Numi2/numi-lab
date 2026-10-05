from pathlib import Path
import sys,json,hashlib,time
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005')
sys.path.insert(0,'/Users/n/numi-human-resting-launcher-source-001/src')
sys.path.insert(0,'/Users/n/numi-human-resting-vascular-structure-source-013/matter/tools')
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.surface_topology_audit import exact_embedding
from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
import accepted_mrvpack_surface_audit as a
skinpath=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/skin-boundaries-001/bodyparts3d-myosim-skinned-shell.nhskin')
skin=decode_payload(skinpath.read_bytes());bodies=skin['bindings_u'][:,0][skin['full_weights'].argmax(axis=1)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[E/'dynamic-release-native-024/resting-human.mrvpack',E/'terminal-posture-inspection-024/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell.mrvpack',E/'terminal-posture-inspection-025/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell.mrvpack']
out=E/'resting-hand-skin-audit-001';out.mkdir(exist_ok=False)
vertices=[];faces=None
for p in paths:
 raw,st,off,s=a.read_pack(p);f=np.asarray(s[(51007,1)]['faces']);lo=f.min();hi=f.max();v=np.ndarray((hi-lo+1,20),dtype='<f4',buffer=raw,offset=off+int(lo)*80)[:,:3].copy();f=f-lo
 assert len(v)==skin['vertex_count'] and np.array_equal(f,skin['indices'].reshape(-1,3))
 if faces is not None:assert np.array_equal(f,faces)
 faces=f;vertices.append(v);raw.close();st.close()
result={'scope':'Complete exact self-pair audits within explicit source dominant-owner hand subsets, including shared-edge tests. Subsets may have a cut wrist boundary; not whole-skin or hand-vs-body clearance, not skin mechanics. Terminal packs are zero-step native Metal FK of exact accepted terminal q.','script_sha256':sha(__file__),'skin_sha256':sha(skinpath),'pack_sha256':{str(p):sha(p) for p in paths},'cases':[]}
for side,low,high in [('right',45,76),('left',95,126)]:
 chosen=np.flatnonzero(((bodies[faces]>=low)&(bodies[faces]<=high)).any(axis=1));f=faces[chosen];used=np.unique(f);remap=np.full(len(bodies),-1);remap[used]=np.arange(len(used));lf=remap[f]
 edges=np.unique(np.sort(np.concatenate((lf[:,[0,1]],lf[:,[1,2]],lf[:,[2,0]])),axis=1),axis=0)
 reference=vertices[0][used];base=np.linalg.norm(reference[edges[:,1]]-reference[edges[:,0]],axis=1);positive=base>1.e-9
 for name,v in zip(['initial','terminal60s_release','terminal6s_upper_passive'],vertices):
  start=time.monotonic();p=v[used]
  qv,qf,_=coordinate_quotient(p.tolist(),lf.astype(int).tolist())
  r=exact_embedding(qv,qf);ratio=np.linalg.norm(p[edges[:,1]]-p[edges[:,0]],axis=1)[positive]/base[positive]
  row={'side':side,'case':name,'dominant_owner_range':[low,high],'selected_source_face_count':len(chosen),'selected_source_vertex_count':len(used),'complete':True,'elapsed_s':time.monotonic()-start,'exact':r,'native_bounds_m':[p.min(axis=0).tolist(),p.max(axis=0).tolist()],'source_edge_stretch_quantiles':np.quantile(ratio,[0,.01,.5,.99,1]).tolist(),'first_witness_source_faces':[[int(chosen[i]) for i in pair] for pair in r['triangle_pairs'][:16]]}
  result['cases'].append(row)
  (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
  print(side,name,'pairs',r['count'],'stretch',row['source_edge_stretch_quantiles'],'seconds',row['elapsed_s'],flush=True)
