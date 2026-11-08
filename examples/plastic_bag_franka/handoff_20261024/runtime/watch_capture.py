from pathlib import Path
import subprocess,json,time,traceback
W=Path(__file__).resolve().parents[1];O=W.parents[1]/'outputs'/W.name;B=Path('/private/tmp/numi_bag_franka_handoff_20261024');REMOTE='/Users/n/numi-plastic-bag-franka-handoff-20261024'
def run(args,**kw):return subprocess.run(args,check=True,**kw)
def partial(p):
 s=p.read_text();d=json.JSONDecoder();i=s.index('[',s.index('"frames"'))+1;fs=[]
 while i<len(s):
  try:f,j=d.raw_decode(s,i)
  except json.JSONDecodeError:break
  fs.append(f);i=j+1
 return fs
last=0
while True:
 status=subprocess.run(['ssh','macmini',f'test -f {REMOTE}/capture/run_summary.json && echo COMPLETE; tail -1 {REMOTE}/capture/solver_accuracy.csv; pgrep -f "^./build/numi_classic_bag_handoff --empty-bag --second-gripper --output capture "'],capture_output=True,text=True).stdout
 print(status.strip().replace('\n',' | '),flush=True);complete='COMPLETE' in status
 run(['rsync','-az',f'macmini:{REMOTE}/capture/trajectory.json',str(B/'capture_raw.json')]);fs=partial(B/'capture_raw.json')
 if fs and (fs[-1]['step']-last>=48 or complete):
  p=B/'capture_progress.json';p.write_text(json.dumps({'frames':fs},separators=(',',':')))
  with (O/'capture_contact_watch.log').open('w') as log:run(['/Applications/Blender.app/Contents/MacOS/Blender','-b','--python-exit-code','1','--python',str(W/'runtime/robot_film_overlap.py'),'--',str(p)],stdout=log,stderr=subprocess.STDOUT)
  a=json.loads(next(l.split('ROBOT_FILM_AUDIT ',1)[1] for l in (O/'capture_contact_watch.log').read_text().splitlines() if l.startswith('ROBOT_FILM_AUDIT ')));(O/'capture_contact_watch.json').write_text(json.dumps(a,indent=2));print('AUDIT '+json.dumps(a),flush=True);last=fs[-1]['step']
  if a['intersection_state_count']:
   # Pause only this exact owned capture invocation, after verifying its PID.
   pidlines=subprocess.check_output(['ssh','macmini',f'pgrep -f "^./build/numi_classic_bag_handoff --empty-bag --second-gripper --output capture --resume checkpoint_1200_parent.bin --steps 1560 --checkpoint-every 120$"'],text=True).split()
   if len(pidlines)==1:run(['ssh','macmini','kill -STOP '+pidlines[0]])
   raise RuntimeError('Native capture housing crossing, paused owned native job')
 if complete:
  run(['rsync','-az',f'macmini:{REMOTE}/capture/',str(O/'capture')+'/']);run(['rsync','-az',f'macmini:{REMOTE}/capture.log',str(O/'capture.log')]);print('CAPTURE COMPLETE',flush=True);break
 if 'numi' not in status and not any(x.strip().isdigit() for x in status.splitlines()):raise RuntimeError('Native process stopped before completion: '+status)
 time.sleep(30)
