#!/usr/bin/env python3
"""Bounded CPU-only Schwarz size sensitivity on the source-175 captured A,b.

This intentionally does not load or compare a native RAP. It is exploratory
conditioning evidence only, with a 120 s and 1 GiB cap.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, signal, time, warnings
from pathlib import Path
import numpy as np

BASE=Path('/Users/home/cardboard-implementation-20261014/analysis')
REPORT=BASE/'offline-operator-tool-development-058/run-175/operator-audit.json'
SOURCE=Path('/Users/home/cardboard-implementation-20261014/analysis/source175-cached-operator-audit-001/cpu-schwarz-64-128/study.py')
OUT=SOURCE.parent
MAX_SECONDS=120.0
MAX_HOST_BYTES=1<<30
spec=importlib.util.spec_from_file_location('cpu175',BASE/'source175-cached-operator-audit-001/cpu_exploratory_conditioning.py')
mod=importlib.util.module_from_spec(spec);assert spec and spec.loader;spec.loader.exec_module(mod)
STUDY=mod.STUDY

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def build_graph_patches_extended(topology, free_nodes, max_patch_nodes, anchor_stride):
 if max_patch_nodes<1 or max_patch_nodes>128:raise ValueError('experimental maximum patch nodes must be in [1,128]')
 if anchor_stride<1 or anchor_stride>max_patch_nodes:raise ValueError('anchor stride must be in [1,patch size]')
 conn=np.asarray(topology.get('nodes'),dtype=np.int64)
 if conn.ndim!=2 or conn.shape[1]!=4:raise ValueError('invalid captured tet connectivity')
 node_count=int(np.asarray(topology['node_count']).reshape(-1)[0])
 if np.any(conn<0) or np.any(conn>=node_count):raise ValueError('tet connectivity out of bounds')
 free=np.asarray(free_nodes,dtype=np.int64);free_set=set(free.tolist())
 adjacency=[set() for _ in range(node_count)]
 for tet in conn:
  ids=[int(x) for x in tet if int(x) in free_set]
  for n in ids:adjacency[n].update(x for x in ids if x!=n)
 anchors=free[::anchor_stride].tolist()
 if not anchors or anchors[-1]!=int(free[-1]):anchors.append(int(free[-1]))
 patches=[]
 for anchor in anchors:
  selected=[int(anchor)];visited={int(anchor)};frontier=[int(anchor)]
  while frontier and len(selected)<max_patch_nodes:
   current=frontier.pop(0)
   for neighbor in sorted(adjacency[current]):
    if neighbor in visited:continue
    visited.add(neighbor);selected.append(neighbor);frontier.append(neighbor)
    if len(selected)==max_patch_nodes:break
  patches.append(np.asarray(sorted(selected),dtype=np.int64))
 covered=set(int(n) for patch in patches for n in patch)
 for node in free:
  if int(node) not in covered:patches.append(np.asarray([int(node)],dtype=np.int64))
 membership=np.zeros(node_count,dtype=np.int32)
 for patch in patches:membership[patch]+=1
 if np.any(membership[free]==0) or np.any(membership[np.setdiff1d(np.arange(node_count),free)]!=0):raise ValueError('patch coverage invalid')
 return patches,membership

def run():
 start=time.monotonic();timer_state={"expired":False}
 def timeout(_signum,_frame):
  timer_state['expired']=True;raise TimeoutError('CPU exploratory study exceeded 120 s')
 old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
 try:
  report,A,vectors,topology,operator_hashes=STUDY.load_hashed_operator(REPORT)
  if report.get('schema')!='numi.cardboard.cached-operator-audit.v1':raise ValueError('operator report schema mismatch')
  b=np.asarray(vectors['native_rhs_free_impulse_Ns'],dtype=np.float64);free=np.asarray(vectors['free_node_ids'],dtype=np.int64)
  if b.shape!=(A.shape[0],) or not np.isfinite(b).all() or not np.linalg.norm(b):raise ValueError('captured RHS invalid')
  if report.get('binding',{}).get('capture_run_fingerprints_match') is not True:raise ValueError('native source binding not admitted')
  if report.get('capture_integrity',{}).get('production_inputs_bitwise_unchanged') is not True:raise ValueError('source capture integrity failed')
  topo={'nodes':np.asarray(topology['nodes']),'node_count':int(np.asarray(topology['node_count']).reshape(-1)[0])}
  # Confirm this local extension leaves the previously used 16/32 traversal exact.
  lowchecks=[]
  for n,s in ((16,8),(32,16)):
   a,am=STUDY.build_graph_patches(topo,free,n,s);z,zm=build_graph_patches_extended(topo,free,n,s)
   equal=len(a)==len(z) and all(np.array_equal(x,y) for x,y in zip(a,z)) and np.array_equal(am,zm)
   if not equal:raise ValueError('extended patch traversal changed a previously checked 16/32 membership')
   lowchecks.append({'patch_nodes':n,'anchor_stride':s,'identical_to_reviewed_builder':True,'patch_count':len(a)})
  cases=[];all_warning_rows=[]
  for patch_nodes,anchor_stride in ((64,32),(128,64)):
   if time.monotonic()-start>MAX_SECONDS:raise TimeoutError('time cap reached before next patch size')
   patches,membership=build_graph_patches_extended(topo,free,patch_nodes,anchor_stride)
   apply,details=STUDY.schwarz_preconditioner(A,free,patches,membership)
   # The shared Schwarz builder enforces the 1 GiB factor-work cap.
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',RuntimeWarning)
    result=STUDY.restarted_right_gmres(A,b,apply,restart=16,max_iterations=64,relative_tolerance=1e-8)
   warning_counts={}
   for w in caught:
    key=(w.category.__name__,str(w.message),w.filename,w.lineno)
    warning_counts[key]=warning_counts.get(key,0)+1
   warning_rows=[{'category':k[0],'message':k[1],'file':k[2],'line':k[3],'count':n} for k,n in warning_counts.items()]
   all_warning_rows.extend(warning_rows)
   membership_object={'max_patch_nodes':patch_nodes,'anchor_stride':anchor_stride,
      'patches':[p.astype(int).tolist() for p in patches],
      'per_node_patch_coverage':membership.astype(int).tolist()}
   membership_path=OUT/f'patch-membership-{patch_nodes}.json'
   membership_path.write_text(json.dumps(membership_object,separators=(',',':'),sort_keys=True)+'\n')
   cases.append({'max_patch_nodes':patch_nodes,'anchor_stride':anchor_stride,
      'patch_count':int(len(patches)),'membership_file':membership_path.name,
      'membership_sha256':sha(membership_path),
      'min_patch_nodes':int(min(map(len,patches))),'max_actual_patch_nodes':int(max(map(len,patches))),
      'estimated_dense_factor_bytes':details['dense_factor_bytes_estimate'],
      'estimated_factorization_flops':details['estimated_factorization_flops'],
      'gmres_restart':result['restart'],'gmres_iteration_cap':result['max_iterations'],
      'gmres_relative_tolerance':result['relative_tolerance'],'gmres_status':result['status'],
      'gmres_true_relative_residual':result['final_true_relative_residual'],
      'gmres_residual_history':result['true_residual_history'],
      'gmres_operator_matvecs':result['true_operator_matvec_count'],
      'gmres_preconditioner_applications':result['preconditioner_application_count'],
      'runtime_warnings':warning_rows,'elapsed_seconds':time.monotonic()-start})
   # Do not retain one factorization while building the next. The reported
   # host peak below is source arrays plus the largest sequential workspace.
   del apply, details, patches, membership
  if time.monotonic()-start>MAX_SECONDS:raise TimeoutError('CPU exploratory study exceeded 120 s cap')
  retained_bytes=int(A.data.nbytes+A.indices.nbytes+A.indptr.nbytes+sum(np.asarray(x).nbytes for x in vectors.values())+sum(np.asarray(x).nbytes for x in topology.values()))
  # Cases run sequentially. Conservatively include the largest one-case factor
  # workspace alongside the retained CSR operator and source sidecars.
  total_bytes=retained_bytes+max(int(c['estimated_dense_factor_bytes']) for c in cases)
  if total_bytes>MAX_HOST_BYTES:raise ValueError('source/operator plus factor workspace exceeds 1 GiB host cap')
  import contextlib,io
  buf=io.StringIO()
  with contextlib.redirect_stdout(buf):np.show_config()
  payload={'schema':'numi.cardboard.source175-cpu-schwarz-size-study.v1',
   'classification':'CPU-only exploratory Schwarz sensitivity; native RAP unadmitted; no Matter or physical qualification',
   'binding':{'operator_report':str(REPORT),'operator_report_sha256':sha(REPORT),
      'operator_npz_sha256':operator_hashes,'prediction_sha256':report['binding']['prediction_sha256'],
      'receipt_sha256':report['binding']['receipt_sha256'],'capture_sha256':report['binding']['capture_sha256'],
      'source_snapshot':report['binding']['source_snapshot'],'selector':report['capture_selection'],
      'analysis_script_sha256':sha(SOURCE)},
   'runtime':{'python':'3.10.12','numpy':np.__version__,'scipy':__import__('scipy').__version__,
      'blas_configuration':buf.getvalue()},
   'operator':{'shape':list(A.shape),'nnz':int(A.nnz),'symmetric_skew_relative':report['operator']['relative_skew_norm'],
      'native_action_parity_relative':report['operator']['action_parity']['relative_l2_error'],
      'rhs_l2_Ns':float(np.linalg.norm(b)),'matrix_hash':operator_hashes[REPORT.stem+'.operator.npz']},
   'design':{'rhs':'native captured center residual impulse','operator':'same reconstructed nonsymmetric captured A; no symmetrization',
      'GMRES':'independent CPU right-preconditioned restart16; max64; true b-Ax residual each iteration',
      'Schwarz':'additive graph patches with partition-of-unity node weights, same reviewed traversal extended only to 64/128 nodes',
      'native_rap':'not loaded or compared; retrospective 175 RAP failed A*P admission',
      'time_cap_seconds':MAX_SECONDS,'host_analysis_cap_bytes':MAX_HOST_BYTES},
   'patch_builder_compatibility':lowchecks,'cases':cases,
   'host_memory':{'retained_operator_and_sidecar_bytes_estimate':retained_bytes,
      'maximum_case_factor_estimated_bytes':max(c['estimated_dense_factor_bytes'] for c in cases),
      'sequential_peak_host_bytes_estimate':total_bytes,
      'process_peak_rss_bytes':int(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss),
      'process_peak_rss_units':'bytes on macOS',
      'cap_bytes':MAX_HOST_BYTES,'allocation_note':'The shared Schwarz factor builder also enforces its aggregate factor/workspace estimate before retaining factors.'},
   'warning_summary':all_warning_rows,
   'elapsed_seconds':time.monotonic()-start,
   'limits':['Matrix capture is rejected root 3/Newton 23, not an accepted physical state.',
      'RAP remains inadmissible; this compares only the same A and b under CPU direct/Schwarz experiments.',
      'The study is exploratory conditioning evidence, not production Matter FGMRES or a physical box result.',
      'Historical source175 16/32 runs used original matmul; the current source60 study uses rowwise scalar math.fsum only for Z@y.']}
  path=OUT/'report.json';path.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')
  return payload
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

if __name__=='__main__':
 try:
  print(json.dumps(run(),indent=2,sort_keys=True,allow_nan=False))
 except Exception as e:
  print(f'REJECTED: {type(e).__name__}: {e}')
  raise
