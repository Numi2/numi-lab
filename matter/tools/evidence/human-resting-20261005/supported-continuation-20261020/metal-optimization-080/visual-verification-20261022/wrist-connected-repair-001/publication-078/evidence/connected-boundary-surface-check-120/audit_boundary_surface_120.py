from pathlib import Path
import numpy as np,json,hashlib,sys,importlib.util,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
C=D/'connected-boundary-refinement-114/run-002-source-normal-all-146/surface-proposal.npz';P=D/'connected-boundary-surface-check-120/run-001';assert not P.exists();P.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(C)=='fdcb88d99dd488afb59b5f49ad3b6f5975f37fc46479f74dd8d31e24a0e5d5b8'
z=np.load(C);F=z['surface_faces_zero_based'];M=z['surface_face_markers'];sid=z['surface_face_surface_ids']
sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
pred=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
spec=importlib.util.spec_from_file_location('numilab_human._curve120',pred);s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
out={'candidate_sha256':sha(C),'frames':{},'geometry_admitted':False,'accepted_physical_steps':0}
for frame,key in [('reference','reference_points_m'),('current037','current037_points_m_f32')]:
 X=z[key].astype(np.float32);t=time.monotonic();rec,_=s.float32_surface_records(X,F,source_seam_groups=());a=s._audit_self_pairs_touching_faces(rec,set(range(len(F))));pairs=np.asarray(a['triangle_pairs'],np.int64).reshape(-1,2);np.save(P/(frame+'-surface-pairs.npy'),pairs)
 groups={}
 for i,j in pairs:
  k=','.join(sorted([str(int(sid[i])),str(int(sid[j]))]));groups[k]=groups.get(k,0)+1
 witnesses=[{'face_rows':[int(i),int(j)],'triangles':[F[i].tolist(),F[j].tolist()],'markers':[int(M[i]),int(M[j])],'layers':[int(sid[i]),int(sid[j])]}for i,j in pairs]
 out['frames'][frame]={'pairs':len(pairs),'layer_pair_counts':groups,'complete':a['complete_surface_audit'],'wall_seconds':time.monotonic()-t,'witnesses':witnesses}
(P/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({f:{k:v for k,v in r.items()if k!='witnesses'}for f,r in out['frames'].items()},indent=2))
