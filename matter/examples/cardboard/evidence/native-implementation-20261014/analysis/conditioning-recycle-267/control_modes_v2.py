#!/usr/bin/env python3
"""Oracle-direction ceilings for the captured N63 coarse-space diagnostic.

Neither extra vector is an operational prior Newton direction. One is the
captured symmetric-part soft mode; the other is the offline direct solution.
They test whether an additional mode can help in principle, not whether it can
be produced causally by the native solver.
"""
from __future__ import annotations
import importlib.util, json, math, os, signal, time, hashlib
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy import sparse
BASE=Path('/Users/home/cardboard-implementation-20261014')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'run-005'
STUDY_PATH=ROOT/'study.py'
spec=importlib.util.spec_from_file_location('recycle_study', STUDY_PATH)
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

def sha(path):
 h=hashlib.sha256();
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 t0=time.monotonic(); timed={'value':False}
 def alarm(*_): timed['value']=True; raise TimeoutError('120s cap')
 prev=signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,S.MAX_SECONDS)
 try:
  report,A,vectors,topo,pm,graph,state,free,b,native,Z,ortherr=S.load_inputs()
  lowdata=np.load(S.AUDIT/'symmetric-part-modes.npz')
  direct=np.load(S.AUDIT/'offline-direct-correction.npz')
  candidates={
   'plus_lowest_symmetric_part_mode_oracle':np.asarray(lowdata['eigenvectors'][:,0],dtype=np.float64),
   'plus_offline_direct_solution_oracle':np.asarray(direct['direct'],dtype=np.float64),
   'plus_captured_current_Newton_FGMRES_correction_future_mode':np.asarray(native,dtype=np.float64),
  }
  base={'schema':'numi.cardboard.conditioning-recycle-oracle-controls.v1',
   'classification':'CPU offline diagnostic only; both added vectors use information unavailable to a causal native preconditioner at the selected current solve; not previous-Newton recycle qualification',
   'input_binding':{'audit_report_sha256':sha(S.AUDIT/'report.json'),'operator_npz_sha256':sha(S.AUDIT/'report.operator.npz'),
    'vectors_npz_sha256':sha(S.AUDIT/'report.vectors.npz'),'topology_npz_sha256':sha(S.AUDIT/'report.topology.npz'),
    'P_sha256':sha(S.P_DIR/'P.bin'),'graph_json_sha256':sha(S.GRAPH_JSON),'study_script_sha256':sha(STUDY_PATH)},
   'source_selection':report.get('capture_selection'),'operator_unsymmetrized':True,
   'method':'Restarted right-FGMRES(16), CPU double precision, exact captured A and b; AP/RAP rebuilt per basis; source graph patch map and principal free-free blocks; all explicit dense products use math.fsum.',
   'limits':['The N63 current operator solve failed natively; these are offline preconditioner sensitivity results.','Captured prior Newton-62 line-search correction is absent.','Soft eigenvector and direct solution are diagnostic oracle directions, not implementable without new state capture or additional solve.','CPU double patch factors are not Metal FP32 parity.'],
   'runs':{}}
  for name,v in candidates.items():
   Zp,meta=S.candidate_basis(Z,v)
   M,pinfo=S.build_preconditioner(A,Zp,graph,free,state.shape[0],A.data.nbytes+A.indices.nbytes+A.indptr.nbytes)
   rec={'candidate_info':meta,'preconditioner':pinfo,'iterations':{}}
   for limit in (32,128,512):
    x,hist,summ=S.right_fgmres(A,b,M,S.RESTART,limit,S.TOL)
    rec['iterations'][str(limit)]={'summary':summ,'history':hist,
      'direct_solution_cosine':float(np.dot(x,direct['direct'])/(np.linalg.norm(x)*np.linalg.norm(direct['direct']))),
      'relative_residual_native_correction':float(np.linalg.norm(b-A@x)/np.linalg.norm(b))}
   base['runs'][name]=rec
   if time.monotonic()-t0>S.MAX_SECONDS: raise TimeoutError('120s total cap')
  base['elapsed_s']=time.monotonic()-t0
  (OUT/'report.json').write_text(json.dumps(base,sort_keys=True,indent=2,allow_nan=False)+'\n')
 except Exception as e:
  (OUT/'failure.json').write_text(json.dumps({'error':str(e),'timed_out':timed['value']},indent=2)+'\n');raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prev)
 print(OUT/'report.json')
if __name__=='__main__': raise SystemExit(main())
