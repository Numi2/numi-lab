from pathlib import Path
import os,subprocess,sys
r=Path.cwd();a=r/'artifacts/crow-v15-front-grip-floor-20261110';actor=str(r/'build/crow-v15-stage-aware-learning-20261110/corrective-actor.policypack')
for side,seed in [('j',3193706597),('i',4038889268)]:
 name=side+'-grip-original';env={**os.environ,'CROW_EVAL_NAME':name,'CROW_EVAL_COURSE':'reserved-final-'+side,'CROW_EVAL_SEED':str(seed),'CROW_EVAL_ACTOR':actor}
 with (a/(name+'-run.log')).open('w') as out:p=subprocess.run([sys.executable,str(a/'run.py')],env=env,stdout=out,stderr=subprocess.STDOUT)
 print(name,'exit',p.returncode,flush=True)
