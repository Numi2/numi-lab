"""Independent incremental native mesh check, while video views render."""
from pathlib import Path
import json,subprocess,time
W=Path(__file__).resolve().parents[1];O=W.parents[1]/'outputs'/W.name;B=Path('/private/tmp/numi_bag_franka_handoff_20261024');REMOTE='/Users/n/numi-plastic-bag-franka-handoff-20261024';seen=1680;audited=0
(O/'native_clearance_segments').mkdir(exist_ok=True)
geometry=json.loads((W/'robot2.json').read_text())['frames']
def partial(p):
 s=p.read_text();d=json.JSONDecoder();i=s.index('[',s.index('"frames"'))+1;fs=[]
 while i<len(s):
  try:f,j=d.raw_decode(s,i)
  except json.JSONDecodeError:break
  f['robot2']=geometry[f['step']];fs.append(f);i=j+1
 return fs
while True:
 exists=subprocess.run(['ssh','macmini',f'test -f {REMOTE}/finish/trajectory.json'],capture_output=True).returncode==0
 if not exists:time.sleep(10);continue
 subprocess.run(['rsync','-az',f'macmini:{REMOTE}/finish/trajectory.json',str(B/'contact_only_raw.json')],check=True)
 fs=partial(B/'contact_only_raw.json');segment=[f for f in fs if f['step']>=seen];complete=subprocess.run(['ssh','macmini',f'test -f {REMOTE}/finish/run_summary.json'],capture_output=True).returncode==0
 if segment and (segment[-1]['step']>seen or complete):
  assert segment[0]['step']==seen and [f['step'] for f in segment]==list(range(seen,segment[-1]['step']+1));p=B/'contact_only_segment.json';p.write_text(json.dumps({'frames':segment},separators=(',',':')))
  log=O/'native_clearance_segments'/f'through_{segment[-1]["step"]}.log'
  with log.open('w') as out:subprocess.run(['/Applications/Blender.app/Contents/MacOS/Blender','-b','--python-exit-code','1','--python',str(W/'runtime/robot_film_overlap.py'),'--',str(p)],check=True,stdout=out,stderr=subprocess.STDOUT)
  a=json.loads(next(l.split('ROBOT_FILM_AUDIT ',1)[1] for l in log.read_text().splitlines() if l.startswith('ROBOT_FILM_AUDIT ')));audited+=len(segment)-1;a['unique_intervals_checked']=audited;(O/'safety_contact_latest.json').write_text(json.dumps(a,indent=2));print(json.dumps(a),flush=True)
  if a['intersection_state_count']:
   ids=subprocess.check_output(['ssh','macmini',f'pgrep -f "^./build/numi_classic_bag_handoff --empty-bag --second-gripper --output finish --resume finish_prefix/checkpoint_1680.bin --steps 1920 --checkpoint-every 120$"'],text=True).split()
   if len(ids)==1:subprocess.run(['ssh','macmini','kill -STOP '+ids[0]],check=True)
   raise RuntimeError('Housing crossing in native finish; owned process paused')
  seen=segment[-1]['step']
 if complete:break
 time.sleep(20)
