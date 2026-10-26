"""Actual first-SCP assembly parity; stops before LP, emits no mesh."""
from pathlib import Path
import sys,json,hashlib,importlib.util,inspect
import numpy as np
from scipy import sparse
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
class StopCapture(Exception):pass
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def mh(m):
 m=sparse.csr_matrix(m)
 return dict(shape=list(m.shape),data=ah(m.data),indices=ah(m.indices),indptr=ah(m.indptr))
def capture(path,argv,label):
 sys.path.insert(0,str(path.parent))
 spec=importlib.util.spec_from_file_location("capture_"+label,path)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 state={}
 fn=mod.solve_sparse_l1_clearance_in_basis;sig=inspect.signature(fn)
 def hook(*args,**kw):
  v=sig.bind(*args,**kw).arguments
  for k in ("A","C","output_map"):
   state[k]=sparse.csr_matrix(v[k]).copy()
  for k in ("lower_rows","target_output","lower_output","upper_output","h","weights"):
   state[k]=np.asarray(v[k]).copy()
  raise StopCapture()
 mod.solve_sparse_l1_clearance_in_basis=hook
 old=sys.argv
 try:
  sys.argv=[str(path),*argv]
  try:mod.main()
  except StopCapture:pass
 finally:sys.argv=old
 if not state:raise RuntimeError("did not reach actual basis solver")
 return state
def main():
 argv_path=Path(sys.argv[1]);out=Path(sys.argv[2])
 if out.exists():raise FileExistsError(out)
 argv=json.loads(argv_path.read_text())[2:]
 oi=argv.index("--output")+1
 argv[oi]=str(out.parent/"owner-output-never-created-405")
 assert not Path(argv[oi]).exists()
 old=D/"retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"
 new=D/"fine-point-basis-enrichment-405/propose_joint_clearance_witnesses_405.py"
 a=capture(old,argv,"original401")
 b=capture(new,argv,"405_default")
 c=capture(new,[*argv,"--fine-point-controls"],"405_enriched")
 for key in a:
  if sparse.issparse(a[key]):
   assert mh(a[key])==mh(b[key]),key
  else:assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes(),key
 for key in a:
  if key=="output_map":continue
  if sparse.issparse(a[key]):assert mh(a[key])==mh(c[key]),key
  else:assert a[key].dtype==c[key].dtype and a[key].shape==c[key].shape and a[key].tobytes()==c[key].tobytes(),key
 M0=a["output_map"];M=c["output_map"]
 assert M.shape[0]==M0.shape[0] and M.shape[1]>M0.shape[1]
 # scipy slicing may select a narrower integer dtype, compare values exactly.
 assert (M[:,:M0.shape[1]]-M0).nnz==0
 E=M[:,M0.shape[1]:].tocsc()
 assert np.all(np.diff(E.indptr)==1) and np.all(E.data==1.0)
 anchors=np.flatnonzero((c["lower_output"]==0)&(c["upper_output"]==0))
 assert M[anchors].nnz==0
 r=dict(status="PASS_ACTUAL_FIRST_SCP_PARITY_AND_CONSTRAINED_ENRICHMENT",argv_path=str(argv_path),argv_sha256=hashlib.sha256(argv_path.read_bytes()).hexdigest(),
        default_405_matches_401=True,enriched_output_space_constraints_identical=True,parent_span_preserved=True,
        output_dimension=M.shape[0],parent_coordinate_count=M0.shape[1],enriched_coordinate_count=M.shape[1],added_fine_nodes=(M.shape[1]-M0.shape[1])//3,
        fixed_output_rows=len(anchors),solver_invoked=False,candidate_emitted=False,native_admission=False,
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old,new,D/"fine-point-basis-enrichment-405/fine_point_basis_405.py")},
        matrix_receipts={key:mh(c[key]) for key in ("A","C","output_map")})
 out.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r))
if __name__=="__main__":main()
