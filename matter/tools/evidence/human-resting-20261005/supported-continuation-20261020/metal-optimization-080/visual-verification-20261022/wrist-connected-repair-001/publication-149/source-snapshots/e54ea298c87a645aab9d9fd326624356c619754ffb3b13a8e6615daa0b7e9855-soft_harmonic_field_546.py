"""Soft connected harmonic coordinate proposal; clearance remains uncertified.

The inverse-square rest-edge Laplacian matches the pinned 545 patch owner.
Unlike that endpoint solver, target rows stay free and receive a soft quadratic
penalty; only typed anchors and the selected outer graph ring are fixed.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, sys, warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
SOLVER545=ROOT/"D/contained-clearance-545/build_connected_endpoint_patch545.py"
SOLVER545_SHA="5934629949f183c02c47afd0dc9cf612bff16bd70a77c6a7e1db1ff6f841e142"
BATCH546=ROOT/"D/surface-clearance-546/select_batch_endpoint_operations546.py"
BATCH546_SHA="9c3257524e95b3e00ee41c154caa90cfbc4fcd1031719d46a591338232e03f8b"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(str(path))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def inverse_square_laplacian(points,faces):
    x=np.asarray(points);F=np.asarray(faces,dtype=np.int64)
    if x.ndim!=2 or x.shape[1]!=3 or x.dtype!=np.float32 or not np.isfinite(x).all(): raise ValueError("points must be finite Float32 Nx3")
    if F.ndim!=2 or F.shape[1]!=3 or len(F)==0 or F.min()<0 or F.max()>=len(x): raise ValueError("invalid triangular faces")
    edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
    lengths=np.linalg.norm(x[edges[:,0]].astype(np.float64)-x[edges[:,1]].astype(np.float64),axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths<=0): raise ValueError("degenerate or nonfinite rest edge")
    weights=1.0/(lengths*lengths)
    n=len(x);i=np.r_[edges[:,0],edges[:,1],edges[:,0],edges[:,1]];j=np.r_[edges[:,0],edges[:,1],edges[:,1],edges[:,0]]
    v=np.r_[weights,weights,-weights,-weights]
    L=sparse.coo_matrix((v,(i,j)),shape=(n,n)).tocsr()
    adj=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    degree=np.asarray(L.diagonal(),dtype=np.float64)
    if not np.isfinite(degree).all() or np.any(degree<=0): raise ValueError("nonpositive weighted incident degree")
    return edges,weights,L,adj,degree

def graph_patch(adj,target_rows,rings):
    if type(rings) is not int or rings<1: raise ValueError("rings must be a positive integer")
    n=adj.shape[0];distance=np.full(n,-1,dtype=np.int32);front=np.unique(np.asarray(target_rows,dtype=np.int64))
    if front.ndim!=1 or len(front)==0 or front.min()<0 or front.max()>=n: raise ValueError("invalid target rows")
    distance[front]=0
    for depth in range(1,rings+1):
        if not len(front): break
        neighbors=np.unique(adj[front].indices)
        front=neighbors[distance[neighbors]<0]
        distance[front]=depth
    patch=distance>=0;boundary=distance==rings;interior=patch&~boundary
    return distance,patch,boundary,interior

def solve_soft_harmonic(points,faces,target_rows,target_points,anchors,rings,lam):
    x=np.asarray(points);rows=np.asarray(target_rows);T=np.asarray(target_points);anchors=np.asarray(anchors)
    if not math.isfinite(float(lam)) or float(lam)<=0: raise ValueError("lambda must be finite and positive")
    if rows.ndim!=1 or rows.dtype.kind not in "iu": raise ValueError("target rows must be an integer vector")
    rows=rows.astype(np.int64,copy=False)
    if len(np.unique(rows))!=len(rows): raise ValueError("duplicate or conflicting target rows")
    if T.shape!=(len(rows),3) or T.dtype!=np.float32 or not np.isfinite(T).all(): raise ValueError("invalid Float32 target endpoints")
    if anchors.ndim!=1 or anchors.dtype.kind not in "iu": raise ValueError("anchors must be integer rows")
    anchors=anchors.astype(np.int64,copy=False)
    if len(np.unique(anchors))!=len(anchors) or np.any(anchors<0) or np.any(anchors>=len(x)): raise ValueError("invalid typed anchors")
    if np.any(rows<0) or np.any(rows>=len(x)): raise ValueError("target row outside candidate")
    if np.intersect1d(rows,anchors).size: raise ValueError("target attempts to move a typed anchor")
    edges,edge_weights,L,adj,degree=inverse_square_laplacian(x,faces)
    distance,patch,boundary,interior=graph_patch(adj,rows,rings)
    fixed=np.zeros(len(x),dtype=bool);fixed[anchors]=True;fixed[boundary]=True
    free=np.flatnonzero(interior&~fixed)
    if not np.isin(rows,free).all(): raise ValueError("every target row must remain a free optimization variable")
    target_weight=np.zeros(len(x),dtype=np.float64);target_weight[rows]=degree[rows]
    d=np.zeros((len(x),3),dtype=np.float64);d[rows]=T.astype(np.float64)-x[rows].astype(np.float64)
    A=L[free][:,free]+sparse.diags(float(lam)*target_weight[free],format="csr")
    rhs=float(lam)*target_weight[free,None]*d[free]
    with warnings.catch_warnings():
        warnings.simplefilter("error",MatrixRankWarning)
        u_free=np.asarray(spsolve(A.tocsc(),rhs),dtype=np.float64)
    if u_free.shape!=(len(free),3) or not np.isfinite(u_free).all(): raise ValueError("soft harmonic solve failed")
    residual=np.asarray(A@u_free-rhs,dtype=np.float64)
    residual_inf=float(np.abs(residual).max(initial=0.0))
    denominator=max(float(np.abs(rhs).max(initial=0.0)),float(np.abs(A@u_free).max(initial=0.0)),1e-30)
    u=np.zeros_like(d);u[free]=u_free
    y=x.copy();y[free]=(x[free].astype(np.float64)+u_free).astype(np.float32)
    if not np.array_equal(y[anchors].view(np.uint32),x[anchors].view(np.uint32)): raise AssertionError("typed anchor changed")
    if not np.array_equal(y[boundary].view(np.uint32),x[boundary].view(np.uint32)): raise AssertionError("outer ring changed")
    if not np.array_equal(y[~patch].view(np.uint32),x[~patch].view(np.uint32)): raise AssertionError("outside-patch coordinates changed")
    if not np.isin(rows,free).all(): raise AssertionError("target row was locked")
    target_displacement=d[rows]
    fit_displacement=u[rows]
    return y,dict(edges=edges,edge_weights=edge_weights,laplacian=L,adjacency=adj,weighted_degree=degree,distance=distance,patch_mask=patch,boundary_mask=boundary,free_rows=free,target_displacement=target_displacement,fit_displacement=fit_displacement,linear_residual_inf=residual_inf,linear_residual_relative=residual_inf/denominator)

def main(argv=None):
    ap=argparse.ArgumentParser(description="Construct a soft connected harmonic field; requires fresh full gates before selection")
    ap.add_argument("--state",type=Path,required=True);ap.add_argument("--target-report",type=Path,required=True);ap.add_argument("--targets",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--rings",type=int,default=4);ap.add_argument("--lambda",dest="lam",type=float,required=True)
    a=ap.parse_args(argv)
    if not math.isfinite(a.lam) or a.lam<=0: raise ValueError("lambda must be finite and positive")
    if a.rings<1: raise ValueError("rings must be positive")
    for p,h in ((SOLVER545,SOLVER545_SHA),(BATCH546,BATCH546_SHA)):
        if sha(p)!=h: raise ValueError("pinned source changed: "+str(p))
    batch=load_module(BATCH546,"batch_validation_546_soft_harmonic")
    st,base130,base159=batch.validate_baseline(a.state)
    target_report,target_archive,rows,T=batch.validate_targets(a.target_report,a.targets,st["candidate_sha256"])
    script=Path(__file__);script_hash=sha(script)
    input_paths=[a.state,Path(st["candidate"]),Path(st["point_summary"]),Path(st["surface_report"]),a.target_report,a.targets,SOLVER545,BATCH546]
    input_hashes={str(p):sha(p) for p in input_paths}
    with np.load(st["candidate"],allow_pickle=False) as z:
        x=z["points_world_m"];F=z["source_O_faces"];anchors=z["typed_coarse_anchor_output_rows_018"]
        rows=batch.validate_rows(rows,T,len(x),anchors)
        edges,edge_weights,L,adj,degree=inverse_square_laplacian(x,F)
        distance,patch,boundary,interior=graph_patch(adj,rows,a.rings)
        free=np.flatnonzero(interior&~np.isin(np.arange(len(x)),anchors))
        if not np.isin(rows,free).all(): raise ValueError("every target row must remain free")
        y,sol=solve_soft_harmonic(x,F,rows,T,anchors,a.rings,a.lam)
        payload={k:z[k] for k in z.files}
    if sha(script)!=script_hash or any(sha(p)!=h for p,h in input_hashes.items()): raise RuntimeError("source/input changed during solve")
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    candidate=a.out/"candidate.npz"
    payload["points_world_m"]=y
    np.savez_compressed(candidate,**payload)
    # Verify every non-coordinate archive field remained byte-identical.
    with np.load(st["candidate"],allow_pickle=False) as before,np.load(candidate,allow_pickle=False) as after:
        assert set(before.files)==set(after.files)
        for key in before.files:
            if key=="points_world_m": continue
            if before[key].dtype!=after[key].dtype or before[key].shape!=after[key].shape or before[key].tobytes()!=after[key].tobytes(): raise AssertionError("non-coordinate array changed: "+key)
        if not np.array_equal(after["points_world_m"][anchors].view(np.uint32),before["points_world_m"][anchors].view(np.uint32)): raise AssertionError("typed anchor changed in saved archive")
        if not np.array_equal(after["points_world_m"][sol["boundary_mask"]].view(np.uint32),before["points_world_m"][sol["boundary_mask"]].view(np.uint32)): raise AssertionError("fixed ring changed in saved archive")
        if not np.array_equal(after["points_world_m"][~sol["patch_mask"]].view(np.uint32),before["points_world_m"][~sol["patch_mask"]].view(np.uint32)): raise AssertionError("outside patch changed in saved archive")
    if not np.isin(rows,sol["free_rows"]).all(): raise AssertionError("target row was not a free variable")
    report=dict(status="SOFT_HARMONIC_CONSTRUCTION_PROPOSAL_UNCERTIFIED",candidate=str(candidate),candidate_sha256=sha(candidate),baseline_state=str(a.state),baseline_state_sha256=sha(a.state),base_candidate=str(st["candidate"]),base_candidate_sha256=st["candidate_sha256"],target_report=str(a.target_report),target_report_sha256=sha(a.target_report),target_npz=str(a.targets),target_npz_sha256=sha(a.targets),target_report_row_alignment_validated_by=str(BATCH546),target_rows=rows.tolist(),target_row_count=int(len(rows)),target_rows_unique=True,target_rows_are_free=True,target_rows_soft=True,all185_endpoints_are_not_inherited=True,clearance_claim="none",exact_audit_status="not_run",rings=int(a.rings),outer_fixed_graph_distance=int(a.rings),lambda_dimensionless=float(a.lam),target_weight="weighted_degree_i=sum_j(1/rest_edge_length_ij^2); scales W to Laplacian units",laplacian="pinned 545 inverse-square rest-edge graph; weights=1/length^2",edge_count=int(len(sol["edges"])),target_weight_min=float(degree[rows].min()),target_weight_median=float(np.median(degree[rows])),target_weight_max=float(degree[rows].max()),patch_node_count=int(np.count_nonzero(sol["patch_mask"])),free_node_count=int(len(sol["free_rows"])),fixed_outer_ring_count=int(np.count_nonzero(sol["boundary_mask"])),typed_anchor_count=int(len(anchors)),target_rows_locked=False,typed_anchors_bitwise_unchanged=True,outer_ring_bitwise_unchanged=True,outside_patch_bitwise_unchanged=True,noncoordinate_source_provenance_arrays_bitwise_unchanged=True,linear_residual_inf=sol["linear_residual_inf"],linear_residual_relative=sol["linear_residual_relative"],source_545_sha256=SOLVER545_SHA,batch_validation_546_sha256=BATCH546_SHA,input_sha256=input_hashes,source_sha256=script_hash,native_admission=False,accepted_physical_steps=0,required_before_selection=["run 545 full checker with --trust-mm 4 (12 mm source cap retained)","fresh current-topology area >=0.1 and edge-gradient >=0.5 gates", "exact 315 self-intersection check", "exact 130 self and point check", "complete 159 all-185-bank audit" ])
    if sha(script)!=script_hash or any(sha(p)!=h for p,h in input_hashes.items()): raise RuntimeError("source/input changed during proposal generation")
    (a.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"candidate_sha256":report["candidate_sha256"],"target_rows":len(rows),"free_nodes":len(sol["free_rows"]),"rings":a.rings,"lambda":a.lam,"linear_residual_relative":sol["linear_residual_relative"]}))

if __name__=="__main__": main()
