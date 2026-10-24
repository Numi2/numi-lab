"""Connected offline surface reconstruction, with bounded incremental quality checks.

This constructs reference geometry only. Tet validity and exact intersections are
reported separately and remain mandatory before physical or native admission.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('flow017',D/'safe_connected_fit_017.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def surface_step_quality(previous,proposal,faces,initial_twice_area):
    """Incremental turn/degen guard, not a global self-collision certificate."""
    n0=base.face_normals(previous,faces);n1=base.face_normals(proposal,faces)
    a0=np.linalg.norm(n0,axis=1);a1=np.linalg.norm(n1,axis=1)
    cosine=np.einsum('ij,ij->i',n0,n1)/np.maximum(a0*a1,1e-300)
    finite=bool(np.isfinite(proposal).all() and np.isfinite(cosine).all())
    ratio=a1/np.maximum(initial_twice_area,1e-300)
    return bool(finite and np.all(cosine>.1) and np.all(ratio>1e-4)),cosine,ratio
def build(mesh,targets,patch,out):
    begun=time.monotonic();out.mkdir(exist_ok=False)
    m=dict(np.load(mesh));r=dict(np.load(targets));p=dict(np.load(patch))
    old=m['points_world_m'].astype(float);x=old.copy()
    F=m['physical_outer_faces'];T=m['tetrahedra'];nref=base.face_normals(old,F);aref=np.linalg.norm(nref,axis=1)
    if not np.isfinite(old).all() or np.any(aref<=0):raise ValueError('invalid initial surface')
    knots=np.flatnonzero(r['ray_lower_bounds_m']>0);target=old[knots]+r['node_displacements_m'][knots]
    selected=set(map(int,p['patch_face_ids']))
    af=np.array([int(i) in selected for i in m['physical_O_parent_face_rows']])
    active_nodes=np.unique(F[af]);protected=np.setdiff1d(m['physical_outer_node_ids'],active_nodes)
    history=[];clock=0.;stopped=None
    for iteration in range(80):
        residual=target-x[knots];energy=float(np.sum(residual**2))
        v=base.smooth_velocity(x,x[knots],residual,.012);v[protected]=0
        dt=min(.1,2.-clock);accepted=False
        while dt>=1e-5:
            proposal=(x+dt*v).astype(np.float32).astype(float);proposal[protected]=old[protected]
            safe,cos,ratio=surface_step_quality(x,proposal,F,aref)
            new_energy=float(np.sum((target-proposal[knots])**2))
            if safe and new_energy<energy:
                accepted=True;break
            dt*=.5
        if not accepted:stopped='bounded incremental surface/target objective line search exhausted';break
        x=proposal;clock+=dt
        history.append(dict(iteration=iteration,dt=dt,construction_time=clock,target_sum_squared_residual_m2=new_energy,
            max_target_residual_mm=float(np.linalg.norm(target-x[knots],axis=1).max()*1000),
            minimum_incremental_normal_cosine=float(cos.min()),minimum_reference_area_ratio=float(ratio.min())))
        if clock>=2.-1e-12 or np.linalg.norm(target-x[knots],axis=1).max()<2e-5:break
    new=x.astype(np.float32);changed=np.flatnonzero(np.any(new!=m['points_world_m'],axis=1))
    det=base.signed_det(x,T);e=x[T][:,1:]-x[T][:,:1]
    q=det/np.prod(np.linalg.norm(e,axis=2),axis=1)
    nr=base.face_normals(x,F);a=np.linalg.norm(nr,axis=1)
    reference_cos=np.einsum('ij,ij->i',nr,nref)/(a*aref)
    dst=out/'connected-rest-proposal.npz'
    np.savez_compressed(dst,points_world_m=new,tetrahedra=T,physical_outer_faces=F,
        physical_O_parent_face_rows=m['physical_O_parent_face_rows'],node_displacements_m=x-old,
        changed_node_ids=changed,protected_node_ids=protected)
    report=dict(status='UNADMITTED_REFERENCE_SURFACE_RECONSTRUCTION',steps=len(history),stopped_reason=stopped,construction_time=clock,
        targets=len(knots),moved_nodes=len(changed),max_displacement_mm=float(np.linalg.norm(x-old,axis=1).max()*1000),
        initial_max_target_residual_mm=float(np.linalg.norm(target-old[knots],axis=1).max()*1000),
        final_max_target_residual_mm=float(np.linalg.norm(target-x[knots],axis=1).max()*1000),
        nonpositive_tets=int(sum(det<=0)),min_normalized_det=float(q.min()),
        reference_normal_deviations_over_90_degrees=int(sum(reference_cos<=0)),
        reference_normal_deviation_is_not_self_intersection_proof=True,
        area_ratio_quantiles=np.quantile(a/aref,[0,.01,.5,.99,1]).tolist(),
        protected_outer_nodes_byte_identical=bool(np.array_equal(new[protected],m['points_world_m'][protected])),
        reference_volume_relative_delta=float((det.sum()-base.signed_det(old,T).sum())/base.signed_det(old,T).sum()),
        history=history,input_sha256={str(p):sha(p) for p in (mesh,targets,patch,Path(base.__file__),Path(__file__))},
        output_sha256=sha(dst),physical_admission=False,native_admission=False,
        exact_self_and_all_tissue_crossing_gate='NOT_RUN',volume_remesh_required=bool(np.any(det<=0)),elapsed_s=time.monotonic()-begun)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mesh',type=Path,required=True);ap.add_argument('--targets',type=Path,required=True)
    ap.add_argument('--patch',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    build(a.mesh,a.targets,a.patch,a.out)
if __name__=='__main__':main()
