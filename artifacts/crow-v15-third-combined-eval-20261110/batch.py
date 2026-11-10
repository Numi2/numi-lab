from pathlib import Path
import os,json,subprocess,sys
r=Path.cwd();a=r/'artifacts/crow-v15-third-combined-eval-20261110';seeds=json.loads((r/'artifacts/crow-v15-fresh-start-dev-20261110/seeds.json').read_text());actor=str(r/'build/crow-v15-third-clearance-learning-20261110/corrective-actor.policypack')
for name,side,seed in [('third-j-hard','j',seeds['development_j']),('third-i-hard','i',seeds['development_i']),('third-j-original','j',3193706597),('third-i-original','i',4038889268)]:
 env={**os.environ,'CROW_EVAL_NAME':name,'CROW_EVAL_COURSE':'reserved-final-'+side,'CROW_EVAL_SEED':str(seed),'CROW_EVAL_ACTOR':actor}
 with (a/(name+'-run.log')).open('w') as out:p=subprocess.run([sys.executable,str(a/'run.py')],env=env,stdout=out,stderr=subprocess.STDOUT)
 print(name,'exit',p.returncode,flush=True)
