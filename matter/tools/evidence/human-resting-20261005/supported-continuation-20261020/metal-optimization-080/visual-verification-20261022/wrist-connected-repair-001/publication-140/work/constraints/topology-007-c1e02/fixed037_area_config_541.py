"""Build a 524 area-reference config fragment using the registered fixed-037 arrays."""
from pathlib import Path
import hashlib
import numpy as np

AREA_POINTS_KEY = "area_reference_037_points_world_m_f64"
AREA_FACES_KEY = "area_reference_037_source_O_faces"

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def fixed037_reference_fields(candidate_path):
    path=Path(candidate_path)
    with np.load(path,allow_pickle=False) as z:
        required={AREA_POINTS_KEY,AREA_FACES_KEY,"source_O_faces","points_world_m"}
        if not required.issubset(z.files):
            raise ValueError("candidate does not carry the registered fixed-037 area reference")
        faces=np.asarray(z["source_O_faces"],dtype=np.int64)
        ref_faces=np.asarray(z[AREA_FACES_KEY],dtype=np.int64)
        ref_points=np.asarray(z[AREA_POINTS_KEY],dtype=np.float64)
        points=np.asarray(z["points_world_m"],dtype=np.float32)
        if not np.array_equal(faces,ref_faces):
            raise ValueError("fixed-037 area reference topology differs from candidate")
        if ref_points.shape!=points.shape or not np.isfinite(ref_points).all():
            raise ValueError("fixed-037 reference point domain is invalid")
    return {
        "area_reference_npz":str(path),
        "area_reference_sha256":sha256(path),
        "area_reference_points_key":AREA_POINTS_KEY,
        "area_reference_faces_key":AREA_FACES_KEY,
    }

def apply_fixed037_reference(config,candidate_path):
    out=dict(config)
    out.update(fixed037_reference_fields(candidate_path))
    return out
