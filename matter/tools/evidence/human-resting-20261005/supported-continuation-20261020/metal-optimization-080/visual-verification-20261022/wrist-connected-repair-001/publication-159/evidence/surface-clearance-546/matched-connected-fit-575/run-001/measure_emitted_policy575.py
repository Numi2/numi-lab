"""Report fit-policy and admission-policy residuals after Float32 retention."""
import importlib.util,sys,json,hashlib
from pathlib import Path
import numpy as np
N=Path(__file__).resolve().parent.parent;D=N.parent
def load(p):
 s=importlib.util.spec_from_file_location("edgeowner575",p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ownerpath=D/"surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
owner=load(ownerpath)
if sha(ownerpath)!="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2":raise ValueError("edge owner pin mismatch")
S=N/"connected-star-573/run-001/final/state.json";s=json.loads(S.read_text())
with np.load(s["candidate"])as z:b=z["points_world_m"].astype(np.float64)*1000
rows=[]
for label,p in[("raw002_local",N/"matched-connected-fit-575/run-001/local-retention/candidate.npz"),("retained1913",N/"matched-connected-fit-575/run-001/retained-regions/candidate.npz"),("witness1896",N/"witness-restore-578/checked-001/candidate.npz")]:
 with np.load(p)as z:x=z["points_world_m"].astype(np.float64)*1000;f=z["source_O_faces"].copy()
 support=np.flatnonzero(np.any(x!=b,axis=1));rec={"label":label,"candidate":str(p),"candidate_sha256":sha(p),"maximum_euclidean_step_mm":float(np.linalg.norm(x-b,axis=1).max(initial=0)),"support_nodes":len(support)}
 for budget in(.1,.5):
  A,rhs,detail=owner.build_edge_gradient_constraints(b,f,support,budget)
  slack=np.asarray(A@(x-b)[support].ravel()).ravel()-rhs
  rec[str(budget)]={"minimum_edge_slack_mm":float(slack.min(initial=1e30)),"violated_rows_tolerance_1e7":int(np.count_nonzero(slack< -1e-7)),"row_count":len(slack)}
 rows.append(rec)
out=N/"matched-connected-fit-575/run-001/emitted-policy-measurement.json"
out.write_text(json.dumps({"status":"MEASURED_POST_RETENTION_POLICY_RESIDUALS","source_sha256":sha(__file__),"edge_owner_sha256":sha(ownerpath),"baseline_state_sha256":sha(S),"fits_use_1mm_trust_and_0p1_gradient":"linear construction policy, separately measured here after local retention","existing_admission_limits":"4mm trust / 0.5 edge-gradient; unchanged","measurements":rows,"native_admission":False,"accepted_physical_steps":0},indent=2,sort_keys=True)+"\n")
print(json.dumps(rows,sort_keys=True))

