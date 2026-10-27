"""Run retained 582/545/P175 checks on the matched 607 fit, no native admission."""
from pathlib import Path
import importlib.util,json,sys
N=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;ROOT=N.parents[1]
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 m=load('stages_check607',N/'continue_surface_round546.py')
 h=load('target_check607',N/'matched-connected-fit-583/run_matched_fit583_v2.py')
 prep=m.read_json(D/'prepared.json');state=Path(prep['state']);st=m.read_json(state)
 ex=m.read_json(D/'fit-stage.execution.json')
 if ex['returncode'] or not ex['source_pins_unchanged']:raise ValueError('fit execution failed')
 sel=h.choose_target(Path(st['candidate']),D/'fit','emitted');m.write_json(D/'target-selection.json',sel)
 retain=N/'local-witness-retention-582/retain_local_witnesses582.py';out=D/'retention-582-001'
 py=Path('/Users/n/numi-human-prep-venv-20261005/bin/python')
 m.run_stage([py,retain,'--state',state,'--target',sel['selected_target'],'--out',out,'--trust-mm','4','--max-edge-gradient','.5'],D/'retention-stage',slot=True,source_paths=(Path(__file__),retain,state,Path(sel['selected_target']),D/'target-selection.json'),timeout=600)
 candidate=out/'candidate.npz';checker=ROOT/'D/contained-clearance-545/run_candidate_checks545.py';checks=out/'full-check-545'
 m.run_stage([py,checker,'--candidate',candidate,'--baseline-state',state,'--output-dir',checks,'--trust-mm','4'],D/'full-check-stage',slot=False,source_paths=(checker,state,candidate),timeout=600)
 report=m.read_json(checks/'report.json');print(json.dumps({'stage':'FULL_CHECK','report':str(checks/'report.json'),'counts':{k:report.get(k) for k in ['surface_pairs','contained_pairs','self_pairs','complete_banks']}}),flush=True)
 final=N/'finalize_existing_checks546.py'
 m.run_stage([py,final,'--candidate',candidate,'--baseline-state',state,'--point-summary',checks/'audit-130/summary.json','--surface-report',checks/'audit-159/report.json','--out',out/'point-continuation-finalize','--point-continuation','--trust-mm','4'],D/'finalize-stage',slot=False,source_paths=(final,state,candidate,checks/'audit-130/summary.json',checks/'audit-159/report.json'),timeout=600)
 print('FINALIZED',out/'point-continuation-finalize/state.json',flush=True)
if __name__=='__main__':main()
