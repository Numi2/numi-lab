from pathlib import Path
import os,json,subprocess,sys
r=Path.cwd();a=r/'artifacts/crow-v15-anchored-i-recoveries-20261110';seeds=json.loads((r/'artifacts/crow-v15-fresh-start-dev-20261110/seeds.json').read_text());actor=str(r/'build/crow-v15-anchored-paired-learning-20261110/corrective-actor.policypack')
for name,seed,takeover,shader in [('i-original-recovery',4038889268,1850,r/'build/crow-v15-delayed-i-recovery-20261110/teacher.metallib'),('i-hard-recovery',seeds['development_i'],2450,r/'build/crow-v15-anchored-i-recoveries-20261110/hard.metallib')]:
 env={**os.environ,'CROW_EVAL_NAME':name,'CROW_EVAL_COURSE':'reserved-final-i','CROW_EVAL_SEED':str(seed),'CROW_EVAL_ACTOR':actor,'CROW_RECOVERY_SHADER':str(shader),'CROW_TAKEOVER_STEP':str(takeover)}
 with (a/(name+'-run.log')).open('w') as out:p=subprocess.run([sys.executable,str(a/'run.py')],env=env,stdout=out,stderr=subprocess.STDOUT)
 print(name,'exit',p.returncode,flush=True)
