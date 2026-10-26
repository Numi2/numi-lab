"""Apply an existing refined offline direction to exact parent material nodes."""
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
N=Path(__file__).resolve().parent.parent
def main():
 ap=argparse.ArgumentParser()
 for k in ['state','refined-base','refined-trial','raw-report','out']:ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 hp=Path(__file__).with_name('retain_parent_correction565.py')
 sp=importlib.util.spec_from_file_location('transfer565existing',hp);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
 s=json.loads(a.state.read_text());base=Path(s['candidate'])
 if m.sha(base)!=s['candidate_sha256']:raise ValueError('state changed')
 rr=json.loads(a.raw_report.read_text())
 if rr.get('candidate_sha256')!=m.sha(a.refined_trial):raise ValueError('raw capture does not bind candidate')
 if rr.get('base_sha256')!=m.sha(a.refined_base):raise ValueError('raw capture does not bind refined base')
 if rr.get('native_admission') is not False or rr.get('accepted_physical_steps')!=0:raise ValueError('unexpected physical admission')
 base_payload=m.load(base)
 result,proof=m.transfer(base_payload,m.load(a.refined_base),m.load(a.refined_trial))
 a.out.mkdir(parents=True);cp=a.out/'candidate.npz';np.savez_compressed(cp,**result)
 free=np.flatnonzero(np.any(result['points_world_m']!=base_payload['points_world_m'],axis=1))
 report={'status':'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED','base_candidate':str(base),'base_sha256':m.sha(base),
 'candidate_path':str(cp),'candidate_sha256':m.sha(cp),'free_nodes':free.tolist(),'proof':proof,
 'scope':'transferred refined-coordinate correction on exact parent material rows; original partial-LP failure retained at source; no LP feasibility claimed for transferred target',
 'source_pins':{str(p):m.sha(p)for p in [Path(__file__),hp,a.state,base,a.refined_base,a.refined_trial,a.raw_report]},
 'native_admission':False,'accepted_physical_steps':0,'full_545_required':True}
 (a.out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'candidate':str(cp),'sha256':m.sha(cp),'proof':proof}),flush=True)
if __name__=='__main__':main()

