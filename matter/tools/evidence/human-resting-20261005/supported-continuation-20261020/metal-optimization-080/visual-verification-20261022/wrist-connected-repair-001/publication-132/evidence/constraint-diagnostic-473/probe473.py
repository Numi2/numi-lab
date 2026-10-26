from pathlib import Path
import importlib.util,sys,json,hashlib,numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
w=D/"D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
spec=importlib.util.spec_from_file_location("probe458",w);m=importlib.util.module_from_spec(spec);sys.modules["probe458"]=m;spec.loader.exec_module(m)
original=m.install_objective
class ProbeDone(Exception):pass
def install(owner,helper,crossing):
 original(owner,helper,crossing)
 def probe(A,b,target,lo,hi,C,h,weights,M,*args,**kw):
  Am=(A@M).tocsr();Cm=(C@M).tocsr()
  def stats(T):
   norms=np.sqrt(np.asarray(T.multiply(T).sum(axis=1)).ravel());nz=norms[norms>0]
   return dict(shape=list(T.shape),nnz=int(T.nnz),zero_rows=int((norms==0).sum()),row_norm_quantiles=np.quantile(nz,[0,.001,.01,.5,.99,1]).tolist(),coefficient_abs_min=float(np.min(np.abs(T.data[T.data!=0]))),coefficient_abs_max=float(np.max(np.abs(T.data))))
  V=sparse.vstack((Am,Cm,M),format="csr");V.data[:]=1
  # Each row is a factor; bipartite connectivity avoids a dense normal matrix.
  Zr=sparse.csr_matrix((V.shape[0],V.shape[0]));Zc=sparse.csr_matrix((V.shape[1],V.shape[1]))
  graph=sparse.bmat([[Zr,V],[V.T,Zc]],format="csr")
  nc,labels=connected_components(graph,directed=False)
  sizes=np.bincount(labels[V.shape[0]:]);sizes=np.sort(sizes[sizes>0])[::-1]
  out=dict(A=stats(A),Am=stats(Am),Cm=stats(Cm),M=stats(M),control_components=sizes.tolist(),largest_component_fraction=float(sizes[0]/M.shape[1]),rhs_abs_quantiles=np.quantile(np.abs(b),[0,.01,.5,.99,1]).tolist(),native_admission=False,accepted_physical_steps=0,solver_invoked=False)
  path=D/"D/constraint-diagnostic-473/report.json";path.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out));raise ProbeDone()
 owner.solve_sparse_l1_clearance_in_basis=probe
m.install_objective=install
try:m.main()
except ProbeDone:pass
