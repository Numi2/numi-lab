"""Resume an existing fit using a verified legacy-compatible construction seed receipt."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
N=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,sort_keys=True,allow_nan=False)+"\n")
def main():
 state,fit,context_path,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=False)
 context=read(context_path);report_path=Path(context['seed_report']);s=read(report_path)
 pairs=Path(context['seed_pairs_path']);rows=Path(s['output_rows_path'])
 assert s['candidate_sha256']==read(state)['candidate_sha256']==sha(read(state)['candidate'])
 assert sha(pairs)==s['output_pairs_sha256'] and str(pairs)==s['output_pairs_path']
 p=np.load(pairs,allow_pickle=False);z=np.load(rows,allow_pickle=False)
 assert p.dtype.kind in 'iu' and p.ndim==2 and p.shape[1]==2 and len(p)==s['merged_unique_pair_count']
 assert s['zero_step_feasible'] is True and sha(rows)==s['output_rows_sha256']
 assert s['pair_owner_sha256']==sha(s['pair_owner_path'])=='c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3'
 assert np.isfinite(z['lower']).all() and float(np.max(z['lower'],initial=-np.inf))<=2e-8
 assert len(z['lower'])==s['rebuilt_linear_row_count']
 assert s['status'] in ('PASS_PARTIAL_EXACT_FACE_REMAP_AND_ZERO_STEP_026_CERTIFICATE','PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY')
 compatible={**s,'status':'PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY','mapping_status':s['status'],'prior_receipt_path':str(report_path),'prior_receipt_sha256':sha(report_path),'compatibility_scope':'same rebuilt seed rows and zero-step certificate; legacy431 status label; original mapping/drop receipts retained'}
 cp=out/'seed-preflight-compatible.json';write(cp,compatible)
 context.update(seed_report=str(cp),seed_report_sha256=sha(cp))
 ctx=out/'constraint-context.json';write(ctx,context)
 spec=importlib.util.spec_from_file_location('round546',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.run_stage([m.PYTHON,m.BACKTRACKER,state,fit,ctx,out/'backtrack'],out/'backtrack-stage',slot=False,source_paths=(Path(__file__),m.BACKTRACKER,m.FINALIZER))
 selected=m.read_json(out/'backtrack/backtrack-373/selected-result.json')
 m.run_stage([m.PYTHON,m.FINALIZER,'--candidate',selected['candidate'],'--baseline-state',state,'--point-summary',selected['point_summary'],'--surface-report',selected['surface_report'],'--out',out/'final-check'],out/'final-check-stage',slot=False,source_paths=(Path(__file__),m.FINALIZER))
 write(out/'state.json',read(out/'final-check/state.json'))
 print(json.dumps({k:read(out/'final-check/report.json').get(k) for k in ('status','surface_pairs','contained_pairs','self_pairs','construction_progress')}))
if __name__=='__main__':main()
