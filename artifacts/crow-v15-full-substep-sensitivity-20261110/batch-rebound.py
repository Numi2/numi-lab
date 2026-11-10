from pathlib import Path
import os,subprocess,sys
R=Path.cwd();A=R/'artifacts/crow-v15-full-substep-sensitivity-20261110';P=R/'build/crow-v15-full-substep-sensitivity-20261110'
for sub in (8,16):
 name='j-original-'+str(sub)+'-rebound';actor=str(P/('route-'+str(sub)+'.policypack'));env={**os.environ,'CROW_EVAL_NAME':name,'CROW_EVAL_COURSE':'reserved-final-j','CROW_EVAL_SEED':'3193706597','CROW_EVAL_ACTOR':actor,'CROW_PHYSICS_SUBSTEPS':str(sub),'CROW_LANDING_POLICY':str(P/('landing-'+str(sub)+'.policypack'))}
 with (A/(name+'-run.log')).open('w') as out:p=subprocess.run([sys.executable,str(A/'run-rebound.py')],env=env,stdout=out,stderr=subprocess.STDOUT)
 print(name,'exit',p.returncode,flush=True)
