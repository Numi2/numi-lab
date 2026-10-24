"""Read-only saved-state housing audit independent of presentation rendering."""
from pathlib import Path
import subprocess,json,time
R=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');W=R/'work/plastic_bag_empty_franka_20261024';O=R/'outputs/plastic_bag_empty_franka_20261024';B=Path('/private/tmp/numi_empty_bag_contact_watch_20261024');B.mkdir(exist_ok=True);REMOTE='/Users/n/numi-plastic-bag-empty-franka-20261024'
def call(a,**kw):return subprocess.run(a,check=True,**kw)
last=720
while True:
 call(['rsync','-a',f'macmini:{REMOTE}/combined/trajectory.json',str(B/'raw.json')]);s=(B/'raw.json').read_text();i=s.index('[',s.index('"frames"'))+1;d=json.JSONDecoder();fs=[]
 while i<len(s):
  try:f,j=d.raw_decode(s,i)
  except json.JSONDecodeError:break
  if f['step']>=last:fs.append(f)
  i=j+1
 if fs:
  p=B/'audit.json';p.write_text(json.dumps({'frames':fs},separators=(',',':')))
  x=call(['/Applications/Blender.app/Contents/MacOS/Blender','-b','--python-exit-code','1','--python',str(W/'runtime/robot_film_overlap.py'),'--',str(p)],capture_output=True,text=True)
  audit=json.loads(next(l.split('ROBOT_FILM_AUDIT ',1)[1] for l in x.stdout.splitlines() if l.startswith('ROBOT_FILM_AUDIT ')));(O/'contact_watch_latest.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
  if audit['intersection_state_count']:
   pid=call(['ssh','macmini',f'cat {REMOTE}/native.pid'],capture_output=True,text=True).stdout.strip();assert pid.isdigit()
   command=call(['ssh','macmini',f'ps -p {pid} -o command='],capture_output=True,text=True).stdout.strip();assert command.startswith('./build/numi_classic_bag_drop --empty-bag --output combined --resume prefix/checkpoint_720.bin')
   call(['ssh','macmini',f'kill -STOP {pid}']);(O/'contact_watch_failure.json').write_text(json.dumps(audit,indent=2));raise RuntimeError('Owned native run paused for housing-film crossing')
  last=fs[-1]['step']
  if last==1920:break
 time.sleep(40)
