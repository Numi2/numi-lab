from pathlib import Path
from fractions import Fraction
import hashlib, importlib.util, json, sys, time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'per-bank-workers-20261027/b115-118/attempt-004/run-001/candidate.npz'
B=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A130=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-130/summary.json'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
EXACT=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
import importlib
exact=importlib.import_module('numilab_human.cardiac_cavity_intersections')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
with np.load(C,allow_pickle=False) as z:
 P=np.asarray(z['points_world_m'],dtype=np.float32).astype(np.float64); F=np.asarray(z['source_O_faces'],dtype=np.int64)
with np.load(B,allow_pickle=False) as z:
 PB=np.asarray(z['points_world_m'],dtype=np.float32)
with np.load(C,allow_pickle=False) as z:
 PC=np.asarray(z['points_world_m'],dtype=np.float32)
changed_nodes=np.flatnonzero(np.any(PC.view(np.uint32)!=PB.view(np.uint32),axis=1))
changed_faces=np.flatnonzero(np.any(np.isin(F,changed_nodes),axis=1))
rest_rows=np.setdiff1d(np.arange(len(F),dtype=np.int64),changed_faces,assume_unique=True)
tri=P[F]; nt=tri[changed_faces]; rt=tri[rest_rows]
lo,hi=nt.min(1),nt.max(1); rlo,rhi=rt.min(1),rt.max(1)
active=np.zeros(len(rest_rows),dtype=bool)
for l,h in zip(lo,hi): active|=np.all(rhi>=l,axis=1)&np.all(rlo<=h,axis=1)
active_rows=rest_rows[active]
points=[tuple(Fraction.from_float(float(v)) for v in p) for p in P]
def records(rows):
 ff=F[rows]
 raw=exact._records(points,ff)
 return [(t,lo,hi,int(rows[i]),vids) for i,(t,lo,hi,_rid,vids) in enumerate(raw)]
newrec=records(changed_faces); restrec=records(active_rows)
vs=exact._audit_pair(newrec,restrec,same_surface=True)
mut=exact._audit_pair(newrec,newrec,same_surface=True)
pairs={tuple(sorted(map(int,p))) for p in vs['triangle_pairs']+mut['triangle_pairs']}
base=json.loads(A130.read_text())
out={'schema':'numi.human.bank115_118.corrected_incremental_exact_self.v1','candidate_path':str(C),'candidate_sha256':sha(C),'current_candidate_sha256':sha(B),'baseline_complete_self_audit_sha256':sha(A130),'baseline_exact_PLC_surface_pairs':int(base['exact_PLC_surface_pairs']),'changed_node_count':int(len(changed_nodes)),'changed_face_count':int(len(changed_faces)),'changed_vs_unchanged_aabb_candidates':int(len(active_rows)),'changed_vs_unchanged_count':int(vs['count']),'changed_vs_unchanged_allowed_shared':int(vs['allowed_shared_vertex_or_edge_pairs']),'changed_mutual_count':int(mut['count']),'changed_mutual_allowed_shared':int(mut['allowed_shared_vertex_or_edge_pairs']),'new_exact_self_pair_count':int(len(pairs)),'new_exact_self_pairs':[list(p) for p in sorted(pairs)],'no_new_exact_self_pairs':bool(not pairs),'exact_predicate_sha256':sha(EXACT),'coordinate_rule':'exact Float32 emitted world coordinates promoted to Float64; rational exact narrowphase with shared global mesh vertex IDs preserved'}
O=S/'per-bank-workers-20261027/b115-118/attempt-004/run-001/exact-self-corrected.json';O.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
