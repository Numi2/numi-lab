from pathlib import Path
import json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-production-integration-20261109'
cmd=json.loads((R/'artifacts/crow-production-navigation-eval-20261109/training/results.json').read_text())['argv'].copy()
for k,v in [('--envs','1'),('--state-trace-environment','0'),('--steps','1'),('--policy-pack',str(R/'build/crow-gate-terminal-training-20261109/recovery-imitation.policypack'))]:cmd[cmd.index(k)+1]=v
env={**os.environ,'DYLD_LIBRARY_PATH':str(R/'build/lib')}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=60)
expected='hierarchical crow stage selection requires deterministic actor-only policies'
r={'returncode':p.returncode,'expected_message':expected,'rejected_with_expected_message':p.returncode!=0 and expected in p.stderr,'stderr':p.stderr,'argv':cmd}
(A/'policy-guard-results.json').write_text(json.dumps(r,indent=2)+'\n')
print('stochastic/critic policy rejected:',r['rejected_with_expected_message'],flush=True)
assert r['rejected_with_expected_message']
