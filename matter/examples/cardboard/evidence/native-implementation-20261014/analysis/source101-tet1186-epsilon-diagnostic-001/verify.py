#!/usr/bin/env python3
import hashlib,json,pathlib,math
HERE=pathlib.Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 m=json.loads((HERE/'manifest.json').read_text())
 for rel,h in m['files'].items():
  if sha(HERE/rel)!=h: raise SystemExit('file hash mismatch: '+rel)
 for name,x in m['inputs'].items():
  if sha(x['path'])!=x['sha256']: raise SystemExit('input hash mismatch: '+name)
 r=json.loads((HERE/'output-001/row1186.json').read_text())
 cases=r['row']['continuation_v3']['epsilon_cases']
 expected=[1e-6,3e-7,1e-7,3e-8,1e-8]
 if [x['target_deltaF_frobenius'] for x in cases]!=expected: raise SystemExit('epsilon set mismatch')
 for x in cases:
  if not x['center_and_both_endpoints_v3_converged'] or not x['branch_signature_stable']:
   raise SystemExit('projection or recorded branch gate failed')
  if x['central_fd_vs_fp64_tangent']['relative_l2'] is None or not math.isfinite(x['central_fd_vs_fp64_tangent']['relative_l2']):
   raise SystemExit('nonfinite FD result')
 print(json.dumps({'status':'verified_bounded_single_row_sweep','row':1186,'cases':len(cases),'central_fd_relative_l2':[x['central_fd_vs_fp64_tangent']['relative_l2'] for x in cases]},indent=2))
if __name__=='__main__': main()
