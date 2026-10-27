"""Shared quality eligibility policy for the isolated topology-025 experiment."""
from pathlib import Path
import numpy as np
QUALITY_KEYS=("normal_cos","area_ratio","min_angle_deg")
MAX_ANGLE_LOSS_DEG=5.0
ANGLE_LOSS_FRACTION=0.5
EPS_DEG=1e-10

def evaluate_quality_arrays(q,before,after):
    old={};new={}
    for key in QUALITY_KEYS:
        if key not in before or key not in after:return None,"invalid_quality_metrics"
        a=np.asarray(before[key],dtype=np.float64).reshape(-1)
        b=np.asarray(after[key],dtype=np.float64).reshape(-1)
        if a.size==0 or b.size==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
            return None,"nonfinite_quality_metrics"
        old[key]=a;new[key]=b
    old_ratio=float(old["area_ratio"].min());new_ratio=float(new["area_ratio"].min())
    old_cos=float(old["normal_cos"].min());new_cos=float(new["normal_cos"].min())
    old_ang=float(old["min_angle_deg"].min());new_ang=float(new["min_angle_deg"].min())
    if old_ang<0.0 or new_ang<0.0:return None,"invalid_quality_metrics"
    if np.any(new["area_ratio"]<q.MIN_AREA_RATIO):return None,"fixed037_area_floor"
    if np.any(new["normal_cos"]<=0.0):return None,"new_face_not_aligned_to_fixed037"
    if new_ang<q.MIN_PATCH_ANGLE_DEG:return None,"angle_loss_bound"
    loss_limit=min(MAX_ANGLE_LOSS_DEG,ANGLE_LOSS_FRACTION*old_ang)
    angle_loss=old_ang-new_ang
    if angle_loss>loss_limit+EPS_DEG:return None,"angle_loss_bound"
    delta=(new_ratio-old_ratio,new_cos-old_cos,new_ang-old_ang)
    return {"score":delta,"before":{"minimum_area_ratio":old_ratio,"minimum_normal_cosine":old_cos,"minimum_angle_deg":old_ang},
      "after":{"minimum_area_ratio":new_ratio,"minimum_normal_cosine":new_cos,"minimum_angle_deg":new_ang},
      "deltas":{"area_ratio":delta[0],"normal_cosine":delta[1],"min_angle_deg":delta[2]},
      "angle_loss_limit_deg":loss_limit,"angle_loss_deg":angle_loss,
      "strict_quality_gain_required_before_exact_pairs":False,
      "exact_185_strict_reduction_still_required_downstream":True},"eligible"

def evaluate_op_local(q,c,op,quality_before,points_mm,ref_mm):
    rows=list(map(int,op["face_rows"]));nf=np.asarray(op["new_faces"],dtype=np.int64)
    before={key:np.asarray(quality_before[key])[rows] for key in QUALITY_KEYS}
    after=q.face_quality(points_mm,ref_mm,nf,nf)
    return evaluate_quality_arrays(q,before,after)

def evaluate_op_parity(q,c,op,quality_before,points_mm,ref_mm):
    rows=list(map(int,op["face_rows"]));nf=np.asarray(op["new_faces"],dtype=np.int64)
    current=q.face_quality(points_mm,ref_mm,np.asarray(c["source_O_faces"]),np.asarray(c["area_reference_037_source_O_faces"]))
    before={key:np.asarray(current[key])[rows] for key in QUALITY_KEYS}
    after=q.face_quality(points_mm,ref_mm,nf,nf)
    return evaluate_quality_arrays(q,before,after)

def install_policy_hooks(topology_module,policy_module,owner541_path):
    topology_module.evaluate_op_local=policy_module.evaluate_op_local
    original_loader=topology_module.loadmod;owner_path=Path(owner541_path).resolve();installed=[]
    def loader(name,path):
        loaded=original_loader(name,path)
        if Path(path).resolve()==owner_path:
            def bound_evaluate_op(c,op,quality,points_mm,ref_mm,_owner=loaded):
                return policy_module.evaluate_op_parity(_owner,c,op,quality,points_mm,ref_mm)
            loaded.evaluate_op=bound_evaluate_op
            installed.append(loaded)
        return loaded
    topology_module.loadmod=loader
    return installed
