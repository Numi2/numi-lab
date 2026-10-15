from pathlib import Path
import hashlib,json,importlib.util,warnings
import numpy as np
from scipy import sparse
B=Path('/Users/home/cardboard-implementation-20261014');P=B/'analysis/source125-full-element-assembly-fp64-audit-150-001';O=B/'analysis/source125-full-element-assembly-fp64-verification-150-002';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
S=B/'snapshots/source-status5-vtccd-diagnostics-125/matter/tools/cardboard_cached_operator_audit.py';spec=importlib.util.spec_from_file_location('audit150v2',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
manifest=json.loads((P/'manifest.json').read_text());assert all(sha(P/p)==h for p,h in manifest['files'].items());assert all(sha(Path(p))==h for p,h in manifest['inputs'].items());cap=B/'solver/two-stage-first-failure-rich-fd-419/operator-capture';h,n,t,_=a.capture_records(cap);n=sorted([x for x in n if x['environment']==0],key=lambda x:x['node_id']);t=sorted([x for x in t if x['environment']==0],key=lambda x:x['tet_id']);D=np.load(B/'analysis/source125-full-tangent-fp64-audit-147-001/matrices.npz');C=D['native_C_Pa'];ref=D['independent_C_Pa'];assert all(np.isfinite(x).all() for x in [C,ref,D['reference_F']]);assert D['valid_projection'].all()
# Explicit contraction avoids the BLAS matmul path that raised warning flags
# in the first consumer. Floating-point exceptions become actual failures.
def mm(x,y):return np.einsum('ik,kj->ij',x,y,optimize=False)
def fn(x):return float(np.sqrt(np.sum(np.square(x))))
free=np.array([i for i,r in enumerate(n) if np.float32(r['center']['constraint_tag']) not in (1,3)],int);dofs=np.full((len(n),3),-1,int);dofs[free]=3*np.arange(len(free))[:,None]+np.arange(3);mass=sparse.diags(np.repeat([np.float32(n[i]['center']['mass_kg']) for i in free],3).astype(float));rows=[];cols=[];dn=[];dr=[];maxecol=0.;maxemat=0.;warn=[]
with warnings.catch_warnings(record=True) as w,np.errstate(all='raise'):
 warnings.simplefilter('error')
 for i,r in enumerate(t):
  inv=np.array(r['inverse_rest_rows_per_m'],np.float32).astype(float);v=float(np.float32(r['rest_volume_m3']));meta=np.array(r['center']['dt_seconds_bulk_modulus_kind_frame_active'],np.float32).astype(float);q=np.array(r['material_frame_quaternion_xyzw'],np.float32).astype(float)
  # Same source-bound quaternion convention, evaluated with scalar products.
  if meta[3]==1:
   q=q/np.sqrt(np.sum(q*q));x,y,z,wq=q;rot=np.array([[1-2*(y*y+z*z),2*(x*y-z*wq),2*(x*z+y*wq)],[2*(x*y+z*wq),1-2*(x*x+z*z),2*(y*z-x*wq)],[2*(x*z-y*wq),2*(y*z+x*wq),1-2*(x*x+y*y)]])
  else:rot=np.eye(3)
  grad=np.vstack((-np.sum(inv,axis=0),inv));bl=np.einsum('ac,ib->abic',np.eye(3),mm(grad,rot),optimize=False).reshape(9,12);kn=v*mm(mm(bl.T,C[i]),bl);kr=v*mm(mm(bl.T,ref[i]),bl);assert np.isfinite(kn).all() and np.isfinite(kr).all();err=np.sqrt(np.sum((kn-kr)**2,axis=0));norm=np.sqrt(np.sum(kr**2,axis=0));assert np.all(norm>0);maxecol=max(maxecol,float(np.max(err/norm)));maxemat=max(maxemat,fn(kn-kr)/fn(kr));indices=dofs[np.array(r['nodes'],int)].reshape(12)
  for rr in range(12):
   if indices[rr]<0:continue
   for cc in range(12):
    if indices[cc]<0:continue
    rows.append(int(indices[rr]));cols.append(int(indices[cc]));dn.append(meta[0]**2*kn[rr,cc]);dr.append(meta[0]**2*kr[rr,cc])
 warn=[str(x.message) for x in w]
An=mass+sparse.coo_matrix((dn,(rows,cols)),shape=(2538,2538)).tocsr();Ar=mass+sparse.coo_matrix((dr,(rows,cols)),shape=(2538,2538)).tocsr();Afirst=sparse.load_npz(P/'independent_A_free_csr.npz');Aold=sparse.load_npz(B/'analysis/captured-base-state-cauchy-419-001/A_free_csr.npz');de=An-Ar;cn=np.sqrt(np.asarray(Ar.power(2).sum(axis=0)).ravel());ce=np.sqrt(np.asarray(de.power(2).sum(axis=0)).ravel());assert np.all(cn>0) and np.isfinite(ce).all();rdiff=fn((Ar-Afirst).data)/fn(Afirst.data);ndiff=fn((An-Aold).data)/fn(Aold.data);assert rdiff<1e-14 and ndiff<1e-14
report={'schema':'numi.cardboard.full_assembly_explicit_contraction_verification.v1','first_manifest_sha256':sha(P/'manifest.json'),'finite_input_and_all_local_matrix_checks':True,'floating_point_exceptions':'all raise during explicit small tensor contractions','warnings':warn,'element_columns':23040,'max_element_column_relative_error':maxecol,'max_element_matrix_relative_error':maxemat,'assembled_columns':2538,'max_assembled_column_relative_error':float(np.max(ce/cn)),'assembled_matrix_relative_error':fn(de.data)/fn(Ar.data),'reference_recomputation_relative_difference':rdiff,'cached_recomputation_relative_difference':ndiff,'gates_pass':maxecol<=1e-3 and np.max(ce/cn)<=.01,'scope_correction_to150_first_report':'The fixture has no independent cohesive interfaces. It retains the source explicit finite elastic starch geometry and its original substrate connectivity. The word untied in the first report limitations should not be interpreted as an unbonded limiting-control fixture.','first_run_warning_record':{'source':'root tool-returned stderr; original numeric report and manifest preserved','warnings':['divide by zero encountered in matmul','overflow encountered in matmul','invalid value encountered in matmul'],'outcome':'First consumer completed with every local/assembled result finite. Explicit-contraction recomputation with floating-point exceptions enabled has no warnings and agrees within1e-14. No arithmetic-noise floor is inferred.'},'classification':'same captured-center analytic material/element/frame/assembly agreement only; no fresh native all-column action sweep, FD noise resolution or physical qualification'}
(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(O/'verify.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({'inputs':{str(P/'manifest.json'):sha(P/'manifest.json'),**{p:h for p,h in manifest['inputs'].items()}},'files':{str(p.relative_to(O)):sha(p) for p in sorted(O.iterdir()) if p.is_file()}},indent=2)+'\n');print(json.dumps(report,indent=2))
