from pathlib import Path
import os,subprocess,sys
R=Path.cwd();A=R/'artifacts/crow-v15-full-substep-sensitivity-20261110';actor=str(R/'build/crow-v15-second-clearance-learning-20261110/corrective-actor.policypack')
for sub in (8,16):
 name='j-original-'+str(sub);env={**os.environ,'CROW_EVAL_NAME':name,'CROW_EVAL_COURSE':'reserved-final-j','CROW_EVAL_SEED':'3193706597','CROW_EVAL_ACTOR':actor,'CROW_PHYSICS_SUBSTEPS':str(sub)}
 with (A/(name+'-run.log')).open('w') as out:p=subprocess.run([sys.executable,str(A/'run.py')],env=env,stdout=out,stderr=subprocess.STDOUT)
 print(name,'exit',p.returncode,flush=True)
