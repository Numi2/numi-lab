from pathlib import Path
p=Path("/Users/n/cardboard-implementation-20261014-host-macmini/analysis/native497-two-level-preconditioner-001/run_experiment.py")
s=p.read_text()
def rep(old,new):
 global s
 if old not in s: raise SystemExit("missing fragment "+repr(old[:100]))
 s=s.replace(old,new)
helpers="""def norm2(v):
 total=0.0
 for a in np.asarray(v).reshape(-1): total+=float(a)*float(a)
 return math.sqrt(total)
def dot2(a,b):
 total=0.0
 aa=np.asarray(a).reshape(-1); bb=np.asarray(b).reshape(-1)
 for i in range(aa.size): total+=float(aa[i])*float(bb[i])
 return total
def sparse_mv(A,v):
 v=np.asarray(v).reshape(-1); out=np.zeros(A.shape[0],dtype=np.float64)
 for i in range(A.shape[0]):
  total=0.0
  for k in range(A.indptr[i],A.indptr[i+1]): total+=float(A.data[k])*float(v[A.indices[k]])
  out[i]=total
 return out
def sparse_mm(A,V):
 V=np.asarray(V); out=np.zeros((A.shape[0],V.shape[1]),dtype=np.float64)
 for i in range(A.shape[0]):
  row=out[i]
  for k in range(A.indptr[i],A.indptr[i+1]): row+=float(A.data[k])*V[A.indices[k]]
 return out
def combine_columns(V,c):
 out=np.zeros(V.shape[0],dtype=np.float64)
 for j in range(V.shape[1]):
  if c[j]!=0.0: out+=float(c[j])*V[:,j]
 return out
"""
rep("def gmres_right(A,M,b,x0,restart=16,cycles=8):",helpers+"def gmres_right(A,M,b,x0,restart=16,cycles=8):")
start=s.index("def gmres_right("); end=s.index("def main():",start)
gmres="""def gmres_right(A,M,b,x0,restart=16,cycles=8):
 x=x0.copy(); r=b-sparse_mv(A,x); beta0=norm2(r)
 hist=[{"iteration":0,"residual_l2":beta0,"relative_to_initial":1.0}]
 n=b.size; anorm=norm2(A.data)
 for cyc in range(cycles):
  r=b-sparse_mv(A,x); beta=norm2(r)
  if beta==0.0: break
  V=np.zeros((n,restart+1)); Z=np.zeros((n,restart)); H=np.zeros((restart+1,restart))
  V[:,0]=r/beta; xbase=x.copy(); breakdown=False
  for j in range(restart):
   Z[:,j]=M(V[:,j]); w=sparse_mv(A,Z[:,j])
   for _ in range(2):
    for i in range(j+1):
     hij=dot2(V[:,i],w); H[i,j]+=hij; w-=hij*V[:,i]
   H[j+1,j]=norm2(w)
   if H[j+1,j]>np.finfo(float).eps*max(1.0,anorm): V[:,j+1]=w/H[j+1,j]
   rhs=np.zeros(j+2); rhs[0]=beta
   y=sparse_lstsq(H[:j+2,:j+1],rhs)
   x=xbase+combine_columns(Z[:,:j+1],y)
   rn=norm2(b-sparse_mv(A,x))
   hist.append({"iteration":cyc*restart+j+1,"residual_l2":rn,
                "relative_to_initial":rn/beta0 if beta0 else None,
                "cycle":cyc+1,"inner_iteration":j+1})
   if H[j+1,j]<=np.finfo(float).eps*max(1.0,anorm): breakdown=True; break
  if breakdown: break
 return x,hist
"""
s=s[:start]+gmres+s[end:]
rep("from scipy.linalg import solve as dense_solve","from scipy.linalg import solve as dense_solve, lstsq as scipy_lstsq")
rep("def sparse_mv(A,v):","def sparse_lstsq(A,b): return scipy_lstsq(A,b,lapack_driver='gelsd')[0]\ndef sparse_mv(A,v):")
rep("AP=np.asarray(A@P,dtype=np.float64);R=P.T@AP","""AP=sparse_mm(A,P);R=np.zeros((P.shape[1],P.shape[1]),dtype=np.float64)
 for i in range(P.shape[1]):
  for j in range(P.shape[1]): R[i,j]=dot2(P[:,i],AP[:,j])""")
rep('def coarse_coeff(v):return dense_solve(R,P.T@v,assume_a="gen",check_finite=True)','def coarse_coeff(v):return dense_solve(R,np.array([dot2(P[:,j],v) for j in range(P.shape[1])]),assume_a="gen",check_finite=True)')
rep("def M_regional(v):return P@coarse_coeff(v)","def M_regional(v):return combine_columns(P,coarse_coeff(v))")
rep("""  cc=coarse_coeff(v);pc=P@cc
  return pc+h_apply(v-AP@cc)""","""  cc=coarse_coeff(v);pc=combine_columns(P,cc);apc=combine_columns(AP,cc)
  return pc+h_apply(v-apc)""")
rep(""" coarse_test=P@np.linspace(-0.75,1.25,P.shape[1]);two_err=float(np.linalg.norm(M_two(A@coarse_test)-coarse_test)/max(np.linalg.norm(coarse_test),1e-300))
 block_err=float(np.linalg.norm(M_block(A@coarse_test)-coarse_test)/max(np.linalg.norm(coarse_test),1e-300))""",""" coarse_test=combine_columns(P,np.linspace(-0.75,1.25,P.shape[1]));two_err=norm2(M_two(sparse_mv(A,coarse_test))-coarse_test)/max(norm2(coarse_test),1e-300)
 block_err=norm2(M_block(sparse_mv(A,coarse_test))-coarse_test)/max(norm2(coarse_test),1e-300)""")
rep("xstar=lu.solve(b);r0=b-A@x0","xstar=lu.solve(b);r0=b-sparse_mv(A,x0)")
rep("exact_res=float(np.linalg.norm(b-A@xstar))","exact_res=norm2(b-sparse_mv(A,xstar))")
rep("float(np.linalg.norm(b-A@X))","norm2(b-sparse_mv(A,X))")
rep("float(np.linalg.norm(xstar))","norm2(xstar)")
rep("float(np.linalg.norm(X-xstar))","norm2(X-xstar)")
rep('float(np.linalg.norm(solutions["captured_block3"]))','norm2(solutions["captured_block3"])')
rep('float(np.linalg.norm(solutions["source_block_RAP_two_level"]))','norm2(solutions["source_block_RAP_two_level"])')
rep('float(np.linalg.norm(solutions["captured_block3"]-xstar))','norm2(solutions["captured_block3"]-xstar)')
rep('float(np.linalg.norm(solutions["source_block_RAP_two_level"]-xstar))','norm2(solutions["source_block_RAP_two_level"]-xstar)')
rep("float(sparse.linalg.norm(A))","norm2(A.data)")
rep(";print(\"P_SHAPE\",P.shape,\"FINITE\",np.isfinite(P).all(),\"MAX\",np.nanmax(np.abs(P)))","")
p.write_text(s)
