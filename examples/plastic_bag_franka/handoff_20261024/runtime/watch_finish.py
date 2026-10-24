"""Stream actual native states into audited, normal-speed two-angle presentation."""
from pathlib import Path
import json,time,subprocess,sys
W=Path(__file__).resolve().parents[1];O=W.parents[1]/'outputs'/W.name;B=Path('/private/tmp/numi_bag_franka_handoff_20261024');REMOTE='/Users/n/numi-plastic-bag-franka-handoff-20261024';BLENDER='/Applications/Blender.app/Contents/MacOS/Blender'
def call(a,**kw):return subprocess.run(a,check=True,**kw)
geometry=json.loads((W/'robot2.json').read_text())['frames']
old_finish=[f for f in json.loads((O/'finish_prefix/trajectory.json').read_text())['frames'] if f['step']<1680]
def partial(p):
 s=p.read_text();d=json.JSONDecoder();i=s.index('[',s.index('"frames"'))+1;fs=[]
 while i<len(s):
  try:f,j=d.raw_decode(s,i)
  except json.JSONDecodeError:break
  f['robot2']=geometry[f['step']];fs.append(f);i=j+1
 return fs
while True:
 if subprocess.run(['ssh','macmini',f'test -f {REMOTE}/finish/trajectory.json'],capture_output=True).returncode==0:break
 if (O/'capture_contact_watch.json').exists() and json.loads((O/'capture_contact_watch.json').read_text())['intersection_state_count']:raise RuntimeError('Capture needs correction')
 time.sleep(20)
first=json.loads((O/'prefix/trajectory.json').read_text())['frames'];capture=json.loads((O/'capture/trajectory.json').read_text())['frames']
assert first[-1]['front']==capture[0]['front'] and first[-1]['back']==capture[0]['back'];prefix=first[:-1]+capture;assert len(prefix)==1561
last=0
while True:
 status=subprocess.run(['ssh','macmini',f'test -f {REMOTE}/finish/run_summary.json && echo COMPLETE; tail -1 {REMOTE}/finish/solver_accuracy.csv; pgrep -f "^./build/numi_classic_bag_handoff --empty-bag --second-gripper --output finish "'],capture_output=True,text=True).stdout
 print(status.strip().replace('\n',' | '),flush=True);complete='COMPLETE' in status
 call(['rsync','-az',f'macmini:{REMOTE}/finish/trajectory.json',str(B/'finish_raw.json')]);seg=old_finish+partial(B/'finish_raw.json')
 if seg and (seg[-1]['step']-last>=48 or complete):
  assert seg[0]['step']==1560 and seg[0]['front']==prefix[-1]['front'] and seg[0]['back']==prefix[-1]['back'];fs=prefix[:-1]+seg;assert [f['step'] for f in fs]==list(range(len(fs)))
  p=B/f'accepted_{len(fs)}.json';p.write_text(json.dumps({'fps':240,'frames':fs},separators=(',',':')))
  audit_path=B/'finish_audit_segment.json';audit_path.write_text(json.dumps({'frames':seg},separators=(',',':')))
  with (O/'finish_contact_watch.log').open('w') as log:call([BLENDER,'-b','--python-exit-code','1','--python',str(W/'runtime/robot_film_overlap.py'),'--',str(audit_path)],stdout=log,stderr=subprocess.STDOUT)
  a=json.loads(next(l.split('ROBOT_FILM_AUDIT ',1)[1] for l in (O/'finish_contact_watch.log').read_text().splitlines() if l.startswith('ROBOT_FILM_AUDIT ')));(O/'finish_contact_watch.json').write_text(json.dumps(a,indent=2));print('AUDIT '+json.dumps(a),flush=True)
  if a['intersection_state_count']:
   pidlines=subprocess.check_output(['ssh','macmini',f'pgrep -f "^./build/numi_classic_bag_handoff --empty-bag --second-gripper --output finish --resume finish_prefix/checkpoint_1680.bin --steps 1920 --checkpoint-every 120$"'],text=True).split()
   if len(pidlines)==1:call(['ssh','macmini','kill -STOP '+pidlines[0]])
   raise RuntimeError('Native handoff housing crossing; paused only owned native process')
  # Concurrent views, independent of the native GPU on macmini.
  jobs=[]
  for mode in ['wide','close']:
   log=(B/f'final_render_{mode}.log').open('a');jobs.append((subprocess.Popen([BLENDER,'-b','--python-exit-code','1','--python',str(W/'runtime/render.py'),'--',str(p),str(B/f'final_frames_{mode}'),mode],stdout=log,stderr=subprocess.STDOUT),log))
  for job,log in jobs:
   code=job.wait();log.close();assert code==0
  last=seg[-1]['step'];(O/'STATUS.json').write_text(json.dumps({'status':'native_run_and_rendering','through_s':fs[-1]['time_s'],'target_s':8,'housing_crossings':0},indent=2))
  if complete:
   (O/'trajectory.json').write_bytes(p.read_bytes());assert len(fs)==1921 and fs[-1]['time_s']==8;break
 if 'numi' not in status and not any(x.strip().isdigit() for x in status.splitlines()) and not complete:raise RuntimeError('Native finish stopped before completion')
 time.sleep(20)
call(['rsync','-az',f'macmini:{REMOTE}/finish/',str(O/'finish')+'/']);call(['rsync','-az',f'macmini:{REMOTE}/finish.log',str(O/'finish.log')]);print('FULL NATIVE RUN AND RENDERS COMPLETE',flush=True)
