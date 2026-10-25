from pathlib import Path
import numpy as np,json,hashlib,sys,importlib.util,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
C=D/'connected-source-patch-cut-123/run-017-expanded-target-hybrid/hybrid-O-surface-128.npz';P=D/'connected-source-patch-cut-123/run-018-expanded-self';assert not P.exists();P.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(C)=='541eb6eabf442c74592853d40aecea1c9c4cb6aac20a31d4d3aa0017cd32afb7'
z=np.load(C);F=z['source_O_faces'];M=z['parent_source_face_marker'];sid=z['face_kind']
sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
pred=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
spec=importlib.util.spec_from_file_location('numilab_human._curve120',pred);s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
out={'candidate_sha256':sha(C),'frames':{},'geometry_admitted':False,'accepted_physical_steps':0}
for frame,key in [('reference_emitted_float32','points_world_m')]:
 X=z[key].astype(np.float32);t=time.monotonic();rec,_=s.float32_surface_records(X,F,source_seam_groups=());a=s._audit_self_pairs_touching_faces(rec,set(range(len(F))));pairs=np.asarray(a['triangle_pairs'],np.int64).reshape(-1,2);np.save(P/(frame+'-surface-pairs.npy'),pairs)
 groups={}
 for i,j in pairs:
  k=','.join(sorted([str(int(sid[i])),str(int(sid[j]))]));groups[k]=groups.get(k,0)+1
 witnesses=[{'face_rows':[int(i),int(j)],'triangles':[F[i].tolist(),F[j].tolist()],'markers':[int(M[i]),int(M[j])],'layers':[int(sid[i]),int(sid[j])]}for i,j in pairs]
 out['frames'][frame]={'pairs':len(pairs),'layer_pair_counts':groups,'complete':a['complete_surface_audit'],'wall_seconds':time.monotonic()-t,'witnesses':witnesses}
(P/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({f:{k:v for k,v in r.items()if k!='witnesses'}for f,r in out['frames'].items()},indent=2))
