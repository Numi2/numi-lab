"""One matched full-plane solve, then exact-self-safe independent region line search."""
from pathlib import Path
import argparse,importlib.util,json,sys
N=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('stages_one_scp549',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
PY=sys.executable
def main():
 ap=argparse.ArgumentParser()
 for k in ['state','prior-context','failed-candidate','failed-audit130','round-dir']:ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();T=a.round_dir;T.mkdir(exist_ok=False);P=T/'prepare';state=json.loads(a.state.read_text());B=Path(state['candidate'])
 for k in ['candidate','point_summary','surface_report']:
  if m.sha(state[k])!=state[k+'_sha256']:raise ValueError('state pin '+k)
 prep=N/'prepare_surface546.py'
 m.run_stage([PY,str(prep),'--state',str(a.state),'--out',str(P),'--prior-context',str(a.prior_context),'--seed-source-candidate',str(a.failed_candidate),'--seed-source-audit130',str(a.failed_audit130),'--ancestry-parent-candidate',str(B)],T/'prepare-stage',slot=False,source_paths=(prep,),timeout=1800)
 m.validate_prepared_baseline(P,a.state);ctx=json.loads((P/'constraint-context.json').read_text())
 pn,pr=m.write_empty_point_targets(P/'point-targets',B,Path(state['point_summary']),P/'depth/365-compatible-report.json')
 matched=P/'matched-inputs.json'
 m.matched_inputs_from_template(m.MATCHED_TEMPLATE,matched,{'candidate':B,'audit159':Path(state['surface_report']),'face_target_npz':P/'surface-targets/targets.npz','point_target_npz':pn})
 argv=m.fit_argv_from_template(m.FIT_TEMPLATE,candidate=B,audit159=Path(state['surface_report']),context=ctx,prep=P,point_npz=pn,point_report=pr,matched=matched,output=T/'fit')
 capture=N/'raw-lp-direction-capture-546/capture_raw_lp_directions546.py'
 if m.sha(capture)!='850893a0cc3271fee42e8342e8d73c786cc396eb2f1df4d267c28556c2ace073':raise ValueError('capture changed')
 argv[1]=str(capture)
 for f,v in [('--area-scp-iterations','1'),('--trust-mm','4')]:argv[argv.index(f)+1]=v
 (T/'fit-argv.json').write_text(json.dumps(argv,indent=2)+'\n')
 m.run_stage(['/usr/bin/env','NUMI_RAW_LP_DIRECTION_FIT_MODE_546=full-plane','NUMI_RAW_LP_DIRECTION_CAPTURE_DIR_546='+str(T/'raw-directions'),*argv],T/'fit-stage',slot=True,source_paths=(capture,N/'run_surface_only_full_coordinate_fit546.py'),timeout=600)
 h=N.parent/'component-area-backtracking-549/component_area_backtracking549.py'
 m.run_stage([PY,str(h),'--raw-dir',str(T/'raw-directions/raw-direction-001'),'--context',str(P/'constraint-context.json'),'--output',str(T/'component-proposal'),'--trust-mm','4','--exact-self'],T/'component-stage',slot=True,source_paths=(h,),timeout=600)
 check=N.parent/'contained-clearance-545/run_candidate_checks545.py'
 m.run_stage([PY,str(check),'--candidate',str(T/'component-proposal/candidate.npz'),'--baseline-state',str(a.state),'--trust-mm','4','--output-dir',str(T/'full-check')],T/'full-check-stage',slot=False,source_paths=(check,),timeout=300)
 r=json.loads((T/'full-check/report.json').read_text())
 print(json.dumps({k:r.get(k) for k in ['status','surface_pairs','self_pairs','contained_pairs','construction_progress']}),flush=True)
 if r.get('construction_progress'):
  h=N/'finalize_existing_checks546.py'
  m.run_stage([PY,str(h),'--candidate',str(T/'component-proposal/candidate.npz'),'--baseline-state',str(a.state),'--point-summary',str(T/'full-check/audit-130/summary.json'),'--surface-report',str(T/'full-check/audit-159/report.json'),'--out',str(T/'final'),'--trust-mm','4'],T/'final-stage',slot=False,source_paths=(h,),timeout=120)
if __name__=='__main__':main()
