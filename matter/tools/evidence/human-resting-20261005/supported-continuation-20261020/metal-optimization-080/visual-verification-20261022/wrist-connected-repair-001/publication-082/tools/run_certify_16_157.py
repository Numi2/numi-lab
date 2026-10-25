from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from scipy import sparse
from coordinate_volume_bound_157 import certify_parent_volume_partition

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
INPUT = BASE / "run-005"
WITNESS = INPUT / "volume-diagnostic-004" / "clip-failure-witness-subset.npz"
SOURCE = INPUT / "outer-positive-input-checkpoint.npz"
OUT = INPUT / "volume-diagnostic-006"
OUT.mkdir(parents=True, exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
z=np.load(WITNESS,allow_pickle=False); s=np.load(SOURCE,allow_pickle=False)
W=sparse.csr_matrix((z["W_data"],z["W_indices"],z["W_indptr"]),shape=tuple(z["W_shape"]))
r=certify_parent_volume_partition(s["points_world_m_f64"],z["source_tets"],z["points"],z["children"],z["parents"],W)
r.update({"source_witness_npz_sha256":sha(WITNESS),"source_checkpoint_npz_sha256":sha(SOURCE),"helper_sha256":sha(HERE/"coordinate_volume_bound_157.py"),"test_source_sha256":sha(HERE/"test_coordinate_volume_bound_157.py"),"test_scope":"16 frozen parent rows / 50 children; no full extraction rerun","coordinate_allowance_is_a_priori":True,"measured_residual_in_determinant_bound":False})
(OUT/"arithmetic-certificate.json").write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:r[k] for k in ("status","parent_count","child_count","max_measured_point_vs_W_source_residual_m","max_point_encoding_residual_allowance_m","max_a_priori_coordinate_error_bound_m","max_stored_coordinate_partition_residual_m3","max_derived_arithmetic_bound_m3","max_stored_coordinate_partition_relative_residual","max_derived_arithmetic_bound_relative_to_parent","max_actual_vs_reference_relative_residual","source_witness_npz_sha256","source_checkpoint_npz_sha256","helper_sha256","test_source_sha256")},sort_keys=True,indent=2))
