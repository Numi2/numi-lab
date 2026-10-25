from pathlib import Path
import sys,json,hashlib,importlib.util,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
sys.path.insert(0,str(D/"publication-078/canonical/matter/tools"))
from human_skin_fat_surface_separation import build_pair_inequalities
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
C,AP,OUT=map(Path,sys.argv[1:4]);OUT.mkdir(parents=True,exist_ok=False)
z=np.load(C);data={k:z[k]for k in z.files};P=z["points_world_m"].astype(float);F=z["source_O_faces"];nk=z["node_kind"]
pairs=np.load(AP);vertices=np.unique(F[np.unique(pairs)])
if not np.all(nk[vertices]==1):raise ValueError("source SAT baseline undefined for non-interior witness")
detailpath=D/"publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz"
detail=np.load(detailpath)["detail_baseline_world_m"]
B=P.copy();ids=np.flatnonzero(nk==1);B[ids]=detail[z["node_source_row"][ids]]
rows=[];rhs=[];metas=[];W=sparse.eye(len(P),format="csr")
for a,b in pairs:
 ci,bi,meta=build_pair_inequalities(F[a],F[b],B,W,B,B,candidate_detail_positions=P,pair_id=[int(a),int(b)])
 rows.append(ci);rhs.append(bi-ci@P.ravel());metas.append(meta)
CC=sparse.vstack(rows,format="csr");target=np.concatenate(rhs)
edges=np.unique(np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
adj=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(P),len(P))).tocsr()
selected=set(map(int,vertices));front=selected.copy()
for _ in range(3):
 new=set()
 for v in front:new.update(map(int,adj.indices[adj.indptr[v]:adj.indptr[v+1]]))
 new={v for v in new if nk[v]==1};front=new-selected;selected.update(new)
free=np.array(sorted(selected));K=(sparse.diags(np.asarray(adj.sum(axis=1)).ravel()+.05)-adj)[free][:,free].tocsc();lu=splu(K)
Cs=[CC[:,3*free+k]for k in range(3)];responses=[lu.solve(c.T.toarray())for c in Cs]
S=sum(c@a for c,a in zip(Cs,responses));S=.5*(S+S.T)
target_scaled=1000.*target
def fun(x):return .5*x@S@x-target_scaled@x,S@x-target_scaled
opt=minimize(fun,np.zeros(len(target)),jac=True,bounds=[(0,None)]*len(target),method="L-BFGS-B",options=dict(maxiter=4000,ftol=1e-15,gtol=1e-12,maxls=50,maxcor=30))
U=np.zeros_like(P)
for k in range(3):U[free,k]=.001*(responses[k]@opt.x)
slack=CC@U.ravel()-target;maxd=np.linalg.norm(U,axis=1).max()
r=dict(status="STATIC_SOURCE_PATCH_UNFOLD_ONLY",input_sha256=sha(C),pairs_sha256=sha(AP),source_detail_sha256=sha(detailpath),source_script_sha256=sha(__file__),free_nodes=len(free),constraints=len(target),maximum_displacement_m=float(maxd),minimum_plane_slack_m=float(slack.min(initial=0)),dual_displacement_unit="millimetres",optimization_success=bool(opt.success),optimization_iterations=int(opt.nit),accepted_physical_steps=0,geometry_admitted=False,surface_separators=metas)
if slack.min(initial=0)<-1e-8 or maxd>.003:
 r["status"]="REJECTED_CONSTRUCTION";(OUT/"report.json").write_text(json.dumps(r,indent=2)+"\n");raise ValueError("unfolding feasibility/displacement guard failed")
Y=(P+U).astype(np.float32).astype(float);areas=.5*np.linalg.norm(np.cross(Y[F[:,1]]-Y[F[:,0]],Y[F[:,2]]-Y[F[:,0]]),axis=1)
r["minimum_emitted_face_area_m2"]=float(areas.min())
if np.any(areas<=0):raise ValueError("degenerate emitted face")
data["points_world_m"]=P+U;data["points_before_connected_source_unfold_m"]=P;data["node_connected_source_unfold_displacement_m"]=U
np.savez_compressed(OUT/"candidate.npz",**data);r["candidate_sha256"]=sha(OUT/"candidate.npz")
(OUT/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:v for k,v in r.items()if k!="surface_separators"}))
