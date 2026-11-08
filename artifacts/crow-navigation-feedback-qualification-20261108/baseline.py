from pathlib import Path
import json,os,subprocess
R=Path.cwd();B=R/'build/crow-navigation-feedback-qualification-01'; A=R/'artifacts/crow-navigation-feedback-qualification-20261108'
p=R/'build/crow-active-navigation-transfer-01/trainable.policypack'
commands={}
for name,shader,teacher in [('old-autonomous',R/'build/crow-navigation-teacher-fix-01/baseline.metallib',False),('old-teacher-labels',R/'build/crow-navigation-teacher-fix-01/baseline.metallib',True),('fixed-teacher-labels',R/'build/crow-navigation-teacher-fix-01/corrected-feedback.metallib',True)]:
 cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
  '--birdflow-navigation-course','training','--envs','1','--steps','180','--repeats','1','--chunk','1','--physics-substeps','4',
  '--seed','20261111','--no-scheduled-resets','--minimum-difficulty-band','10','--maximum-difficulty-band','10',
  '--policy-pack',str(p),'--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json',
  '--metallib',str(shader),'--state-trace',str(B/(name+'.tsv'))]
 if teacher:cmd+=['--birdflow-journey-teacher','--birdflow-journey-student-authority','1','--rollout-pack',str(B/(name+'.rolloutpack'))]
 proc=subprocess.run(cmd,cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True,timeout=240)
 (B/(name+'.stdout')).write_text(proc.stdout);(B/(name+'.stderr')).write_text(proc.stderr)
 commands[name]={'argv':cmd,'returncode':proc.returncode,'stderr':proc.stderr[-1500:]}
 (A/'baseline-execution.json').write_text(json.dumps(commands,indent=2,sort_keys=True)+'\n')
 print(name,proc.returncode,flush=True)
