from pathlib import Path
import json, subprocess, time, sys, hashlib, csv, traceback, struct
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa')
W=ROOT/'work/plastic_bag_franka_contact_finish_20261024';R=W/'runtime';OUT=ROOT/'outputs/plastic_bag_franka_contact_finish_20261024';B=Path('/private/tmp/numi_bag_franka_contact_finish_20261024');B.mkdir(exist_ok=True)
REMOTE='/Users/n/numi-plastic-bag-franka-contact-finish-20261024'
BLENDER='/Applications/Blender.app/Contents/MacOS/Blender'
def call(cmd,**kw):return subprocess.run(cmd,check=True,**kw)
def remote(s):return call(['ssh','macmini',s],capture_output=True,text=True).stdout

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_frames(path):
 s=path.read_text();i=s.index('[',s.index('"frames"'))+1;fs=[];decoder=json.JSONDecoder()
 while i<len(s):
  try:f,j=decoder.raw_decode(s,i)
  except json.JSONDecodeError:break
  fs.append(f);i=j+1
 return fs

def sync_file(name,dst):call(['rsync','-a',f'macmini:{REMOTE}/{name}',str(dst)])
def sync_dir(name):
 d=OUT/name;d.mkdir(exist_ok=True);call(['rsync','-a',f'macmini:{REMOTE}/{name}/',str(d)+'/']);return d

def merge(parts):
 frames=[]
 for fs in parts:
  if not fs:continue
  start=fs[0]['step']
  if start:
   assert len(frames)>=start,('native gap',len(frames),start)
   if len(frames)>start:assert frames[start]==fs[0],('join differs',start)
   frames=frames[:start]
  frames+=fs
 assert [f['step'] for f in frames]==list(range(len(frames)))
 return frames

def snapshot(parts,name):
 fs=merge(parts);p=B/name;p.write_text(json.dumps({'fps':240,'frames':fs},separators=(',',':')));return fs,p

def render(path):
 for mode in ['wide','close']:
  with (B/f'render_{mode}.log').open('a') as log:call([BLENDER,'-b','--python',str(R/'render.py'),'--',str(path),str(B/f'frames_{mode}'),mode],stdout=log,stderr=subprocess.STDOUT)

def status(stage):
 x=remote(f'test -f {REMOTE}/{stage}/run_summary.json && echo COMPLETE; pgrep -fl "numi_classic_bag_drop --output {stage}"; tail -1 {REMOTE}/{stage}/energy_history.csv; tail -1 {REMOTE}/{stage}.log')
 print(stage+': '+x.strip().replace('\n',' | '),flush=True)
 if 'COMPLETE' not in x and 'numi_classic_bag_drop' not in x:raise RuntimeError('Native scene stopped before completion: '+x)
 return 'COMPLETE' in x

def write_json(p,a):p.write_text(json.dumps(a,indent=2))

def main():
 first=json.loads((OUT/'prefix/trajectory.json').read_text())['frames'];assert len(first)==1517 and first[-1]['step']==1516
 corner=json.loads((OUT/'grasp_target.json').read_text())['actual_position_m']
 last_rendered=len(first);render(OUT/'prefix/trajectory.json')
 while True:
  complete=status('combined');sync_file('combined/trajectory.json',B/'combined_raw.json')
  seg=read_frames(B/'combined_raw.json');assert seg[0]==first[-1], 'checkpoint saved-state join differs'
  fs=merge([first,seg])
  if complete or len(fs)-last_rendered>=64:
   fs,p=snapshot([first,seg],f'combined_{time.time_ns()}.json')
   logpath=OUT/'robot_film_progress.log'
   with logpath.open('w') as log:call([BLENDER,'-b','--python-exit-code','1','--python',str(R/'robot_film_overlap.py'),'--',str(p)],stdout=log,stderr=subprocess.STDOUT)
   mesh=json.loads(next(line.split('ROBOT_FILM_AUDIT ',1)[1] for line in logpath.read_text().splitlines() if line.startswith('ROBOT_FILM_AUDIT ')))
   write_json(OUT/'robot_film_progress.json',mesh)
   if mesh['intersection_state_count']:
    remote(f'kill -STOP $(cat {REMOTE}/native.pid)')
    raise RuntimeError('Actual native film crosses non-pad robot housing: '+json.dumps(mesh))
   render(p);last_rendered=len(fs)
  if complete:break
  time.sleep(25)
 sync_dir('combined');sync_file('combined.log',OUT/'combined.log');sync_file('combined_identity_sha256.txt',OUT/'combined_identity_sha256.txt')
 seg=json.loads((OUT/'combined/trajectory.json').read_text())['frames'];assert seg[0]==first[-1]
 fs=merge([first,seg]);assert len(fs)==1921 and fs[-1]['time_s']==8
 trajectory=OUT/'trajectory.json';trajectory.write_text(json.dumps({'fps':240,'frames':fs},separators=(',',':')))
 write_json(OUT/'lineage.json',{'continuous_native_checkpoint_run':True,'hz':240,'duration_s':8,'states':len(fs),'segments':[{'name':'prior first_landing','first_step':0,'last_step':240},{'name':'prior combined before closure','first_step':240,'last_step':600},{'name':'outside-corner lift','first_step':600,'last_step':1440},{'name':'release on lift completion','first_step':1440,'last_step':1516},{'name':'reciprocal contact continuation','first_step':1516,'last_step':1920}],'executed_gripper_prefix_exact':True,'physics_state_joins_exact':True,'no_interpolated_or_render_engine_motion':True,'initial_transform_m':[0,.25,0],'initial_coins_at_rest':True,'release_begin_s':6.2,'outside_corner_offset_m':.006,'contact_geometry':'Stock 17 mm rubber faces derived from native articulated finger bodies9/10; all material nodes dynamic.','numerical_budget':'Unchanged material/contact equations and tolerances; reciprocal body-order MPR/EPA/GJK retry on unresolved queries added after step1516. Bounded exact rollback.'})
 audits=[]
 for name,cmd in [('film_clearance.json',[str(R/'audit_clearance')]),('pair_support_bound.json',[sys.executable,str(R/'pair_support_audit.py')]),('pair_boundary.json',[sys.executable,str(R/'pair_audit.py')]),('grip_alignment.json',[sys.executable,str(R/'grip_alignment_audit.py')])]:
  f=(OUT/name).open('w');log=(OUT/(name+'.log')).open('w');audits.append((subprocess.Popen(cmd+[str(trajectory)],stdout=f,stderr=log),f,log))
 render(trajectory)
 for proc,f,log in audits:assert proc.wait()==0;f.close();log.close()
 rows=[]
 for n in ['prefix','combined']:rows+=list(csv.DictReader((OUT/n/'energy_balance.csv').open()))
 rows={int(r['step']):r for r in rows};assert sorted(rows)==list(range(1,1921))
 balances=np.array([float(r['frame_energy_balance_J']) for r in rows.values()]);tols=np.array([float(r['local_tolerance_J']) for r in rows.values()]);assert np.all(balances<=tols+1e-12)
 film=json.loads((OUT/'film_clearance.json').read_text());pair=json.loads((OUT/'pair_support_bound.json').read_text())
 grip=json.loads((OUT/'grip_alignment.json').read_text())
 assessment={'native_run_complete':True,'duration_s':8,'hz':240,'coins':10,'release_begin_s':6.2,'outside_corner_offset_m':.006,'first_drop_m':.25,'requested_lift_m':.32,'actual_opening_corner_lift_m':float(fs[1488]['front'][-1][1]-fs[240]['front'][-1][1]),'first_landing_corner_m':corner,'maximum_accepted_unaccounted_energy_increase_J':float(balances.max()),'max_balance_to_tolerance_ratio':float((balances/tols).max()),'final_inside_coins':fs[-1]['inside_coins'],'final_grip_contact_nodes':fs[-1]['grip_contact_nodes'],'final_max_coin_speed_m_s':float(np.linalg.norm(fs[-1]['coin_velocities'],axis=-1).max()),'film_clearance':film,'pair_support_bound':pair,'stock_grip_alignment':grip,'limits':['Prescribed Panda joint trajectory; motor dynamics are not simulated.','Film and pad properties/friction are assumed, not calibrated.','Finite compliant contact patches coincide with the authored rubber fingertip faces. Full robot/film mesh collisions and motor dynamics are not simulated.','Numerical geometry and energy checks are not physical validation.']}
 write_json(OUT/'assessment.json',assessment)
 print('ASSESSMENT '+json.dumps(assessment),flush=True)
 video=OUT/'air_drop_corner_grip_pickup_drop_8s_normal.mp4';indices=list(range(0,1921,8));assert len(indices)==241
 proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1024','-r','30','-i','-','-an','-frames:v','241','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
 font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
 for i,index in enumerate(indices):
  im=Image.new('RGB',(1920,1024),(24,36,44))
  for j,mode in enumerate(['wide','close']):
   with Image.open(B/f'frames_{mode}'/f'frame_{i:04d}.png') as f:im.paste(f.convert('RGB'),(960*j,64))
  d=ImageDraw.Draw(im);d.text((20,18),'NUMI LAB | AIR DROP + ROBOT PICKUP + DROP | 10 COINS | 240 Hz | NORMAL SPEED',font=font,fill='white');d.text((1720,18),f't={fs[index]["time_s"]:.2f}s',font=font,fill='white');proc.stdin.write(im.tobytes())
 proc.stdin.close();assert proc.wait()==0
 call(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(video),'-f','null','-'])
 probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_entries','format=duration:stream=nb_frames,r_frame_rate','-of','json',str(video)],text=True));assert probe['streams'][0]['nb_frames']=='241'
 write_json(OUT/'video_record.json',{'normal_speed':True,'native_hz':240,'duration_s':8,'video_fps':30,'end_native_state_included':True,'video_sha256':sha(video),'trajectory_sha256':sha(trajectory),'decode_pass':True,'probe':probe})
 write_json(OUT/'STATUS.json',{'status':'complete','video':str(video),'numerical_checks':{'film':film['minimum_thickness_clearance_lower_bound_um']>=-2.001,'pair':pair['within_tolerance'],'energy':bool(np.all(balances<=tols+1e-12))}})
 print('VIDEO READY '+str(video),flush=True)
try:main()
except Exception:
 write_json(OUT/'STATUS.json',{'status':'needs_attention','error':traceback.format_exc()});print(traceback.format_exc(),flush=True);raise
