"""Run existing saved-proposal backtracking with a bounded process group."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time
N=Path(__file__).resolve().parent.parent
T=N/'swept-backtrack-prep-558/backtrack-031-001'
T.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
h=N/'construction-trust-546/run_checked_batch_context546.py'
state=N/'connected-source-refinement-553/refine-frozen30886-002/state.json'
fit=N/'round-031-refined-connected557/fit'
context=N/'swept-backtrack-prep-558/prep-031-001/constraint-context.json'
argv=[PY,str(h),str(state),str(fit),str(context),str(T/'run'),'--construction-trust-mm','4']
pins={str(p):sha(p)for p in [Path(__file__),h,state,fit/'candidate.npz',fit/'report.json',context]}
(T/'root-launch.json').write_text(json.dumps({'argv':argv,'source_pins':pins,'timeout_seconds':600,
'construction_trust_mm':4,'trust_reason':'Matched to retained031fit and existing545 checker; this is an offline construction limit, not a physical step.',
'native_admission':False,'accepted_physical_steps':0},indent=2)+'\n')
start=time.monotonic()
with(T/'root-execution.log').open('x')as log:
 p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (T/'process.json').write_text(json.dumps({'pid':p.pid,'process_group':p.pid})+'\n')
 timed=False
 try:rc=p.wait(timeout=600)
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:rc=p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
result={'returncode':rc,'wall_seconds':time.monotonic()-start,'timed_out':timed,
'input_pins_unchanged':all(sha(p)==v for p,v in pins.items()),'native_admission':False,'accepted_physical_steps':0}
(T/'root-execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
raise SystemExit(rc if not timed else 124)
