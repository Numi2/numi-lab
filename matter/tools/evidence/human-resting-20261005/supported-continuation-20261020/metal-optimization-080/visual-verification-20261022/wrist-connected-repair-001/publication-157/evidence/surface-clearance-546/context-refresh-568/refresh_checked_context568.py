"""Refresh only existing constraints for a newly checked topology; reuse all owners."""
from pathlib import Path
import argparse,importlib.util,json,sys
from types import SimpleNamespace
N=Path(__file__).resolve().parent.parent
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path,required=True);ap.add_argument('--prior-context',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 sp=importlib.util.spec_from_file_location('prep568',N/'prepare_surface546.py');p=importlib.util.module_from_spec(sp);sys.modules[sp.name]=p;sp.loader.exec_module(p)
 a.out.mkdir(parents=True,exist_ok=False)
 st=p.m.read(a.state);prior=p.m.read(a.prior_context);base=Path(st['candidate'])
 for k in ['candidate','point_summary','surface_report']:
  if p.m.sha(st[k])!=st[k+'_sha256']:raise ValueError('state changed: '+k)
 trial=N/'round-026-guard-cuts556/raw-directions/raw-direction-001/candidate.npz'
 audit=N/'round-026-guard-cuts556/raw-self-audit130/summary.json'
 p.m.write(a.out/'input-pins.json',{'paths':{str(x):p.m.sha(x)for x in [Path(__file__),N/'prepare_surface546.py',a.state,a.prior_context,base,trial,audit]},'native_admission':False,'accepted_physical_steps':0})
 p.topology_aware_refresh(SimpleNamespace(ancestry_parent_candidate=base),a.out,st,base,prior,Path(st['point_summary']).parent/'query',trial,audit,Path(prior['surface_report']))
 out=a.out/'constraints/constraint-context.json'
 ctx=p.m.read(out)
 if ctx['candidate_sha256']!=st['candidate_sha256']:raise ValueError('context mismatch')
 print(json.dumps({'context':str(out),'sha256':p.m.sha(out),'candidate_sha256':ctx['candidate_sha256']}),flush=True)
if __name__=='__main__':main()

