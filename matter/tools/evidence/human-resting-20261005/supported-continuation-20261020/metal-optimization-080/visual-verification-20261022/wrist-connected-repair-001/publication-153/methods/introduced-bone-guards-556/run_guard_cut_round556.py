"""Replay a matched connected solve with newly certified fixed-bone guard witnesses."""
from pathlib import Path
import argparse,importlib.util,json,sys
N=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage556',N/'continue_surface_round546.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
 ap=argparse.ArgumentParser()
 for k in ['state','prior-context','fit-argv','guard-npz','guard-report','round-dir']:
  ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args()
 state=json.loads(a.state.read_text());ctx=json.loads(a.prior_context.read_text())
 for k in ['candidate','point_summary','surface_report']:
  if m.sha(state[k])!=state[k+'_sha256']:raise ValueError('state pin '+k)
  if state[k+'_sha256']!=ctx[k+'_sha256']:raise ValueError('context baseline '+k)
 argv=json.loads(a.fit_argv.read_text())
 if Path(argv[argv.index('--base-candidate')+1])!=Path(state['candidate']):raise ValueError('fit baseline')
 # The exact existing 401/313 validator is mandatory before fitting, in addition
 # to the merge receipt's source checks. The fitter runs it again for actual rows.
 guard=json.loads(a.guard_report.read_text())
 if guard['base_candidate_sha256']!=state['candidate_sha256']:raise ValueError('guard baseline')
 if guard['output_npz_sha256']!=m.sha(a.guard_npz):raise ValueError('guard output pin')
 oldguard=ctx['guard_npz']
 ctx.update(guard_npz=str(a.guard_npz),guard_report=str(a.guard_report),
            guard_npz_sha256=m.sha(a.guard_npz),guard_report_sha256=m.sha(a.guard_report),
            guard_extension_556={'prior_context':str(a.prior_context),'prior_context_sha256':m.sha(a.prior_context),
                'prior_guard_npz':oldguard,'merged_report':str(a.guard_report),
                'native_admission':False,'accepted_physical_steps':0})
 T=a.round_dir;T.mkdir(exist_ok=False)
 (T/'constraint-context.json').write_text(json.dumps(ctx,indent=2,sort_keys=True)+'\n')
 for f,v in [('--fixed-boundary-npz',a.guard_npz),('--fixed-boundary-report',a.guard_report),('--output',T/'fit')]:
  m.replace_cli_value(argv,f,v)
 (T/'fit-argv.json').write_text(json.dumps(argv,indent=2)+'\n')
 pins={str(p):m.sha(p) for p in [a.state,a.prior_context,a.fit_argv,a.guard_npz,a.guard_report]}
 (T/'reuse-inputs.json').write_text(json.dumps({'source_pins':pins,
   'scope':'same baseline/topology/targets/depth/self witnesses; only fixed-bone witness inventory extended',
   'native_admission':False,'accepted_physical_steps':0},indent=2)+'\n')
 launcher=Path(argv[1])
 m.run_stage(['/usr/bin/env','NUMI_RAW_LP_DIRECTION_FIT_MODE_546=full-plane',
   'NUMI_RAW_LP_DIRECTION_CAPTURE_DIR_546='+str(T/'raw-directions'),*argv],
   T/'fit-stage',slot=True,source_paths=(Path(__file__),launcher,launcher.parent/'pair_sat_routing_551.py'),timeout=600)
 h=N.parent/'component-area-backtracking-549/component_area_backtracking549.py'
 m.run_stage([sys.executable,str(h),'--raw-dir',str(T/'raw-directions/raw-direction-001'),
   '--context',str(T/'constraint-context.json'),'--output',str(T/'component-proposal'),'--trust-mm','4','--exact-self'],
   T/'component-stage',slot=True,source_paths=(h,),timeout=600)
 check=N.parent/'contained-clearance-545/run_candidate_checks545.py'
 m.run_stage([sys.executable,str(check),'--candidate',str(T/'component-proposal/candidate.npz'),
   '--baseline-state',str(a.state),'--trust-mm','4','--output-dir',str(T/'full-check')],
   T/'full-check-stage',slot=False,source_paths=(check,),timeout=600)
 r=json.loads((T/'full-check/report.json').read_text())
 print(json.dumps({k:r.get(k) for k in ['status','surface_pairs','self_pairs','contained_pairs','construction_progress']}),flush=True)
 if r.get('construction_progress'):
  h=N/'finalize_existing_checks546.py'
  m.run_stage([sys.executable,str(h),'--candidate',str(T/'component-proposal/candidate.npz'),
    '--baseline-state',str(a.state),'--point-summary',str(T/'full-check/audit-130/summary.json'),
    '--surface-report',str(T/'full-check/audit-159/report.json'),'--out',str(T/'final'),'--trust-mm','4'],
    T/'final-stage',slot=False,source_paths=(h,),timeout=120)
 if not all(m.sha(p)==v for p,v in pins.items()):raise ValueError('reused source input changed')
if __name__=='__main__':main()
