"""Run existing witness-local retention and complete checks on matched domains."""
from pathlib import Path
import argparse,importlib.util,json,sys
N=Path(__file__).resolve().parent.parent
def main():
 ap=argparse.ArgumentParser()
 for k in ['state','target','out']:ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--point-continuation',action='store_true');ap.add_argument('--max-edge-gradient',type=float,default=.5);ap.add_argument('--trust-mm',type=float,default=4.);a=ap.parse_args()
 if not __import__('math').isfinite(a.max_edge_gradient) or a.max_edge_gradient<=0:raise ValueError('max-edge-gradient must be finite and positive')
 if not __import__('math').isfinite(a.trust_mm) or a.trust_mm<=0:raise ValueError('trust-mm must be finite and positive')
 sp=importlib.util.spec_from_file_location('g567',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
 T=a.out;owner=Path(__file__).with_name('retain_local_witnesses582.py')
 m.run_stage([sys.executable,owner,'--state',a.state,'--target',a.target,'--out',T,'--max-edge-gradient',str(a.max_edge_gradient),'--trust-mm',str(a.trust_mm)],T.with_name(T.name+'-stage'),slot=True,source_paths=(Path(__file__),owner,a.state,a.target),timeout=600)
 checker=N.parent/'contained-clearance-545/run_candidate_checks545.py'
 m.run_stage([sys.executable,checker,'--candidate',T/'candidate.npz','--baseline-state',a.state,'--trust-mm','4','--output-dir',T/'full-check'],T/'full-check-stage',slot=False,source_paths=(checker,),timeout=600)
 r=json.loads((T/'full-check/report.json').read_text());print(json.dumps({'configured_construction_gates':{'max_edge_gradient':a.max_edge_gradient,'trust_mm':a.trust_mm},'immutable_545_acceptance_gates':{'trust_mm':4.0,'edge_gradient':0.5},'check':{k:r.get(k)for k in ['status','surface_pairs','contained_pairs','self_pairs','construction_progress']}},sort_keys=True),flush=True)
 if r.get('construction_progress') and r.get('self_pairs')==0 and (a.point_continuation or r.get('contained_pairs')==0):
  f=N/'finalize_existing_checks546.py'
  argv=[sys.executable,f,'--candidate',T/'candidate.npz','--baseline-state',a.state,'--point-summary',T/'full-check/audit-130/summary.json','--surface-report',T/'full-check/audit-159/report.json','--out',T/'final','--trust-mm','4']
  if a.point_continuation:argv.append('--point-continuation')
  m.run_stage(argv,T/'final-stage',slot=False,source_paths=(f,),timeout=60)
if __name__=='__main__':main()

