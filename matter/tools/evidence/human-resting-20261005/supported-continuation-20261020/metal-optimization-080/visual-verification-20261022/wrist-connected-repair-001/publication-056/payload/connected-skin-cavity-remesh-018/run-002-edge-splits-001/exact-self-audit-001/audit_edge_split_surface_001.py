from __future__ import annotations
import hashlib, importlib, json, pathlib, sys, time, traceback
import numpy as np
E=pathlib.Path(sys.argv[1]); OUT=pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True,exist_ok=True)
candidate=pathlib.Path(sys.argv[3]); started=time.monotonic()
owner=E/'collision-free-seam-records-owner-315'; src=owner/'src'; compat=pathlib.Path('/Users/n/numi-human-anatomy-finish-1378/src/numilab_human')
import types
pkg=types.ModuleType('numilab_human');pkg.__path__=[str(src/'numilab_human'),str(compat)];sys.modules['numilab_human']=pkg
ci=importlib.import_module('numilab_human.cardiac_cavity_intersections')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
with np.load(candidate,allow_pickle=False) as z:
 X=z['points_world_m'].copy();F=z['physical_outer_faces'].copy()
tri=X[F].astype(np.float64);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);deg=np.flatnonzero(np.all(cross==0.,axis=1))
result={'schema':'connected-skin-edge-split-exact-self-audit-018.v1','candidate_path':str(candidate),'candidate_sha256':sha(candidate),'points_f32':bool(X.dtype==np.float32),'point_count':len(X),'face_count':len(F),'degenerate_face_count':int(len(deg)),'degenerate_face_rows':deg[:64].tolist(),'owner_source_path':str(owner/'src/numilab_human/cardiac_cavity_intersections.py'),'owner_source_sha256':sha(owner/'src/numilab_human/cardiac_cavity_intersections.py'),'exact_predicate_algorithm':ci.ALGORITHM,'status':'FAIL_DEGENERATE' if len(deg) else 'RUNNING'}
if len(deg)==0:
 records,effective=ci.float32_surface_records(X,F,source_seam_groups=())
 if tuple(effective)!=tuple(range(len(X))):raise RuntimeError('unexpected effective IDs without seam groups')
 cache,screen=ci._prepare_complete_self_pair_baseline(records);pairs=tuple(map(tuple,screen['triangle_pairs']))
 result.update({'status':'EXACT_COMPLETE_SURFACE_AUDIT_PASS' if not pairs else 'EXACT_SELF_CROSSINGS_FOUND','exact_self_pair_count':len(pairs),'exact_pair_set_sha256':hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest(),'aabb_candidate_pairs':int(screen.get('aabb_candidate_pairs',0)),'allowed_shared_vertex_or_edge_pairs':int(screen.get('allowed_shared_vertex_or_edge_pairs',0)),'full_exact_audit':bool(screen.get('complete_surface_audit')),'deterministic_first_64_pairs':[list(p) for p in pairs[:64]],'elapsed_s':time.monotonic()-started})
result['elapsed_s']=time.monotonic()-started
(OUT/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2),flush=True)
