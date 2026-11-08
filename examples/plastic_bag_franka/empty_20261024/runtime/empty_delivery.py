"""Complete saved-state audit and presentation for the empty-bag native run."""
from pathlib import Path
import json,subprocess,time,hashlib,csv,sys,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');W=ROOT/'work/plastic_bag_empty_franka_20261024';R=W/'runtime';OUT=ROOT/'outputs/plastic_bag_empty_franka_20261024';B=Path('/private/tmp/numi_empty_bag_franka_20261024');B.mkdir(exist_ok=True)
REMOTE='/Users/n/numi-plastic-bag-empty-franka-20261024';BLENDER='/Applications/Blender.app/Contents/MacOS/Blender'
def call(a,**kw):return subprocess.run(a,check=True,**kw)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_frames(p):
 s=p.read_text();i=s.index('[',s.index('"frames"'))+1;fs=[];d=json.JSONDecoder()
 while i<len(s):
  try:f,j=d.raw_decode(s,i)
  except json.JSONDecodeError:break
  fs.append(f);i=j+1
 return fs

def render(p):
 for mode in ['wide','close']:
  with (B/f'render_{mode}.log').open('a') as log:call([BLENDER,'-b','--python-exit-code','1','--python',str(R/'render.py'),'--',str(p),str(B/f'frames_{mode}'),mode],stdout=log,stderr=subprocess.STDOUT)
def robot_audit(p,name):
 with (OUT/(name+'.log')).open('w') as log:call([BLENDER,'-b','--python-exit-code','1','--python',str(R/'robot_film_overlap.py'),'--',str(p)],stdout=log,stderr=subprocess.STDOUT)
 x=json.loads(next(l.split('ROBOT_FILM_AUDIT ',1)[1] for l in (OUT/(name+'.log')).read_text().splitlines() if l.startswith('ROBOT_FILM_AUDIT ')))
 (OUT/(name+'.json')).write_text(json.dumps(x,indent=2));return x

def main():
 first=json.loads((OUT/'prefix/trajectory.json').read_text())['frames'];assert len(first)==721
 rendered=0
 while True:
  state=subprocess.run(['ssh','macmini',f'test -f {REMOTE}/combined/run_summary.json && echo COMPLETE; pgrep -fl "numi_classic_bag_drop --empty-bag --output combined"; tail -1 {REMOTE}/combined/solver_accuracy.csv; tail -1 {REMOTE}/combined.log'],capture_output=True,text=True).stdout
  print(state.strip().replace('\n',' | '),flush=True);complete='COMPLETE' in state
  if not complete and 'numi_classic_bag_drop' not in state:raise RuntimeError('Native run stopped before full completion: '+state)
  call(['rsync','-a',f'macmini:{REMOTE}/combined/trajectory.json',str(B/'raw.json')])
  seg=read_frames(B/'raw.json')
  if not seg:time.sleep(15);continue
  assert first[-1]==seg[0], 'checkpoint state join is not exact'
  fs=first[:-1]+seg;assert [f['step'] for f in fs]==list(range(len(fs)))
  if complete or len(fs)-rendered>=120:
   p=B/f'states_{len(fs)}.json';p.write_text(json.dumps({'fps':240,'frames':fs},separators=(',',':')))
   audit=robot_audit(p,'robot_film_progress')
   if audit['intersection_state_count']:
    call(['ssh','macmini',f'kill -STOP $(cat {REMOTE}/native.pid)']);raise RuntimeError('Robot housing/film intersection; native run paused: '+json.dumps(audit))
   render(p);rendered=len(fs)
  if complete:break
  time.sleep(25)
 call(['rsync','-a',f'macmini:{REMOTE}/combined/',str(OUT/'combined')+'/'])
 call(['rsync','-a',f'macmini:{REMOTE}/combined.log',str(OUT/'combined.log')])
 seg=json.loads((OUT/'combined/trajectory.json').read_text())['frames'];assert seg[0]==first[-1];fs=first[:-1]+seg;assert len(fs)==1921 and fs[-1]['time_s']==8
 p=OUT/'trajectory.json';p.write_text(json.dumps({'fps':240,'frames':fs},separators=(',',':')))
 assert all(len(f['coins'])==0 and len(f['coin_velocities'])==0 for f in fs)
 with (OUT/'grip_alignment.json').open('w') as f:call([sys.executable,str(R/'grip_alignment_audit.py'),str(p)],stdout=f)
 grip=json.loads((OUT/'grip_alignment.json').read_text());assert grip['all_forces_at_visible_stock_jaws'] and grip['contact_point_samples']>0 and grip['no_grip_force_during_first_drop']
 mesh=robot_audit(p,'robot_film_full');assert mesh['intersection_state_count']==0
 balances=[]
 for part in ['prefix','combined']:balances+=list(csv.DictReader((OUT/part/'energy_balance.csv').open()))
 assert [int(x['step']) for x in balances]==list(range(1,1921))
 errors=np.array([float(x['frame_energy_balance_J']) for x in balances]);tol=np.array([float(x['local_tolerance_J']) for x in balances]);assert np.all(errors<=tol+1e-12)
 positions=np.array([f['front']+f['back'] for f in fs]);minfilm=float(positions[:,:,1].min()-.108-.0000375);assert minfilm>=-1e-7
 lift=float(fs[1488]['front'][-1][1]-fs[240]['front'][-1][1]);assert lift>=.45, ('insufficient actual corner lift',lift)
 assert fs[-1]['grip_contact_nodes']==0
 r={'full_native_run':True,'duration_s':8,'native_hz':240,'saved_states':1921,'coin_count':0,'requested_corner_lift_m':.5,'actual_corner_lift_m':lift,'initial_drop_clearance_m':.25,'minimum_saved_film_outer_table_gap_m':minfilm,'max_accepted_frame_energy_balance_J':float(errors.max()),'max_energy_balance_to_tolerance_ratio':float((errors/tol).max()),'final_grip_contact_nodes':fs[-1]['grip_contact_nodes'],'grip':grip,'robot_mesh':mesh,'physics_state_join_exact':True,'source':'same approved material and contact equations, active coin count zero; empty coin dispatch omission measured separately','limitations':['No aerodynamic drag, as in approved comparison','Uncalibrated LDPE and pad parameters','Prescribed robot joints; no motor or whole-robot collision dynamics','Robot clearance audited at saved states, not continuous whole-robot CCD']}
 (OUT/'assessment.json').write_text(json.dumps(r,indent=2));render(p)
 video=OUT/'empty_bag_air_drop_high_pickup_drop_8s_normal.mp4';proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1024','-r','30','-i','-','-an','-frames:v','241','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
 font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
 for i,index in enumerate(range(0,1921,8)):
  im=Image.new('RGB',(1920,1024),(24,36,44))
  for j,mode in enumerate(['wide','close']):
   with Image.open(B/f'frames_{mode}'/f'frame_{i:04d}.png') as f:im.paste(f.convert('RGB'),(960*j,64))
  d=ImageDraw.Draw(im);d.text((20,18),'NUMI LAB | EMPTY PLASTIC BAG | AIR DROP + HIGH PICKUP + DROP | 240 Hz | NORMAL SPEED',font=font,fill='white');d.text((1720,18),f't={fs[index]["time_s"]:.2f}s',font=font,fill='white');proc.stdin.write(im.tobytes())
 proc.stdin.close();assert proc.wait()==0
 call(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(video),'-f','null','-'])
 probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_entries','format=duration:stream=nb_frames,r_frame_rate','-of','json',str(video)],text=True));assert probe['streams'][0]['nb_frames']=='241'
 (OUT/'video_record.json').write_text(json.dumps({'normal_speed':True,'native_hz':240,'physics_duration_s':8,'native_states':1921,'end_state_included':True,'video_sha256':sha(video),'trajectory_sha256':sha(p),'decode_pass':True,'probe':probe},indent=2));(OUT/'STATUS.json').write_text(json.dumps({'status':'complete','video':str(video)},indent=2));print('VIDEO READY '+str(video),flush=True)
try:main()
except Exception:
 (OUT/'STATUS.json').write_text(json.dumps({'status':'needs_attention','error':traceback.format_exc()},indent=2));raise
