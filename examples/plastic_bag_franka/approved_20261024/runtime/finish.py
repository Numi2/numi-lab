from pathlib import Path
import json,subprocess,time,sys,hashlib,traceback,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa')
OUT=ROOT/'outputs/plastic_bag_franka_release_20261024';B=Path('/private/tmp/numi_bag_franka_release_20261024');R=B/'runtime'
REMOTE='/Users/n/numi-plastic-bag-franka-release-adaptive-time-20261024'
PREFIX=OUT/'prefix'
PREFIX_OWNER='Approved six-second pickup and accepted drop through step 1855; see prefix/lineage.json'
def call(cmd,**kwargs):return subprocess.run(cmd,check=True,**kwargs)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def accepted_frames(path):
 s=path.read_text();i=s.index('[',s.index('"frames"'))+1;frames=[];decoder=json.JSONDecoder()
 while i<len(s):
  try:f,j=decoder.raw_decode(s,i)
  except json.JSONDecodeError:break
  frames.append(f);i=j+1
 return frames
def render(path):
 for mode in ['wide','close']:
  with (B/f'render_{mode}.log').open('a') as log:call(['/Applications/Blender.app/Contents/MacOS/Blender','-b','--python',str(R/'render.py'),'--',str(path),str(B/f'frames_{mode}'),mode],stdout=log,stderr=subprocess.STDOUT)
def main():
 prefix=accepted_frames(PREFIX/'trajectory.json');assert len(prefix)==1878 and prefix[-1]['step']==1877
 last_rendered=1878
 while True:
  x=call(['ssh','macmini',f'test -f {REMOTE}/continuation/run_summary.json && echo COMPLETE; pgrep -fl numi_classic_bag_drop; tail -1 {REMOTE}/continuation/energy_history.csv; tail -1 {REMOTE}/continuation.log'],capture_output=True,text=True).stdout
  print(x.strip(),flush=True)
  complete='COMPLETE' in x
  if not complete and ('scene failed:' in x or 'numi_classic_bag_drop' not in x):raise RuntimeError('Native drop extension stopped before eight seconds. Preserve states; do not render a completion claim.')
  call(['rsync','-a',f'macmini:{REMOTE}/continuation/trajectory.json',str(B/'trajectory_raw.json')])
  segment=accepted_frames(B/'trajectory_raw.json')
  if not segment:time.sleep(2);continue
  assert segment[0]==prefix[-1]
  frames=prefix[:-1]+segment
  if complete:break
  # Render saved native prefixes while the separate remote GPU continues.
  # Files are immutable per batch; existing frame images belong to this one run.
  if len(frames)-last_rendered>=60:
   path=B/f'prefix_{len(frames)}.json';path.write_text(json.dumps({'fps':240,'frames':frames},separators=(',',':')));render(path);last_rendered=len(frames)
  else:time.sleep(20)
 call(['ssh','macmini',f'cd {REMOTE} && shasum -a 256 source/* include/metalrobo/gpu_types.h build/numi_classic_bag_drop build/PlasticBag.metallib initial_state.json grip.csv robot.json continuation/trajectory.json > continuation/final_identity_sha256.txt'])
 call(['rsync','-a',f'macmini:{REMOTE}/continuation/',str(OUT/'run')+'/'])
 call(['rsync','-a',f'macmini:{REMOTE}/continuation.log',str(OUT/'run.log')])
 segment=json.loads((OUT/'run/trajectory.json').read_text())['frames'];assert segment[0]==prefix[-1]
 frames=prefix[:-1]+segment
 trajectory=OUT/'trajectory.json';trajectory.write_text(json.dumps({'fps':240,'frames':frames},separators=(',',':')))
 lineage={'native_hz':240,'physics_duration_s':8,'state_count':len(frames),'prefix':{'owner':PREFIX_OWNER,'steps':[0,1877],'raw_trajectory_sha256':sha(PREFIX/'trajectory.json'),'lineage_file':str(PREFIX/'lineage.json')},'continuation':{'owner':REMOTE,'initial_step':1877,'new_steps':[1878,1920],'trajectory_sha256':sha(OUT/'run/trajectory.json'),'identity_file':str(OUT/'run/final_identity_sha256.txt')},'join_state_exactly_equal':segment[0]==prefix[-1],'method':'Exact native checkpoint continuation. No interpolated or invented physical states. Physical parameters and contact equations unchanged. Resume step 1877 with a bounded joint internal-step/contact-iteration search and unchanged physical equations and acceptance gates; identical energy and geometry acceptance gates.'}
 (OUT/'lineage.json').write_text(json.dumps(lineage,indent=2))
 assert len(frames)==1921 and frames[0]['step']==0 and frames[-1]['step']==1920 and frames[-1]['time_s']==8
 audits=[]
 for name,cmd in [('film_clearance.json',[str(R/'audit_clearance')]),('pair_support_bound.json',[sys.executable,str(R/'pair_support_audit.py')]),('pair_boundary.json',[sys.executable,str(R/'pair_audit.py')])]:
  f=(OUT/name).open('w');log=(OUT/(name+'.log')).open('w')
  audits.append((subprocess.Popen(cmd+[str(trajectory)],stdout=f,stderr=log),f,log))
 render(trajectory)
 for proc,f,log in audits:
  assert proc.wait()==0;f.close();log.close()
 rows=list(csv.DictReader((PREFIX/'energy_balance.csv').open()))+list(csv.DictReader((OUT/'run/energy_balance.csv').open()));balance=np.array([float(r['frame_energy_balance_J']) for r in rows]);tol=np.array([float(r['local_tolerance_J']) for r in rows]);assert np.all(balance<=tol+1e-12)
 summary=json.loads((OUT/'run/run_summary.json').read_text());pair=json.loads((OUT/'pair_support_bound.json').read_text());assert pair['within_tolerance']
 film=json.loads((OUT/'film_clearance.json').read_text());assert film['max_full_triangle_mid_sheet_penetration_mm']<=1e-9 and film['minimum_thickness_clearance_lower_bound_um']>=-2.001
 positions=np.array([f['coins'] for f in frames]);quats=np.array([f['coin_orientations'] for f in frames]);quats/=np.linalg.norm(quats,axis=-1,keepdims=True)
 axis=np.stack([2*(quats[...,0]*quats[...,1]-quats[...,3]*quats[...,2]),1-2*(quats[...,0]**2+quats[...,2]**2),2*(quats[...,1]*quats[...,2]+quats[...,3]*quats[...,0])],axis=-1);axis/=np.linalg.norm(axis,axis=-1,keepdims=True)
 extent=.015305*np.linalg.norm(axis[...,[0,2]],axis=-1)+.001075*abs(axis[...,1]);clearance=positions[...,1]-extent-.108;ti,ci=np.unravel_index(clearance.argmin(),clearance.shape)
 table={'minimum_coin_table_clearance_m':float(clearance.min()),'worst_step':frames[ti]['step'],'worst_coin':int(ci),'within_float_tolerance':bool(clearance.min()>=-1e-7),'method':'FP64 analytic finite-cylinder vertical extent at every saved state; horizontal infinite plane at y=0.108 m.'};assert table['within_float_tolerance'];(OUT/'table_clearance.json').write_text(json.dumps(table,indent=2))
 assessment={'full_8_second_native_run':True,'native_hz':240,'accepted_steps':1920,'source_owner':REMOTE,'lineage':'lineage.json','continued_from_exact_checkpoint_step':1877,'max_unaccounted_accepted_frame_energy_increase_J':float(balance.max()),'max_positive_balance_to_tolerance_ratio':float((balance/tol).max()),'continuation_maximum_edge_stretch_ratio':summary['max_edge_stretch_ratio'],'all_ten_coins_inside_final':frames[-1]['inside_coins']==10,'last_grip_contact_nodes':frames[-1]['grip_contact_nodes'],'opening_corner_lift_m':float(frames[-1]['front'][-1][1]-frames[0]['front'][-1][1]),'minimum_coin_center_height_above_table_m':float(np.array(frames[-1]['coins'])[:,1].min()-.108),'max_coin_speed_last_half_second_m_s':float(np.linalg.norm(np.array([f['coin_velocities'] for f in frames if f['time_s']>=7.5]),axis=-1).max()),'pair_support_bound':pair,'film_clearance':film,'table_clearance':table,'limits':['Prescribed Panda FK trajectory; motor dynamics not simulated','Film and pad material/friction assumptions are not calibrated','Whole-triangle film/coin audit and whole-cylinder pair support bounds are numerical geometry evidence, not physical validation']}
 (OUT/'assessment.json').write_text(json.dumps(assessment,indent=2));print(json.dumps(assessment),flush=True)
 video=OUT/'franka_pickup_and_drop_8s_normal.mp4';indices=list(range(0,1921,8));assert len(indices)==241
 proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1024','-r','30','-i','-','-an','-frames:v','241','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
 font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
 for i,index in enumerate(indices):
  im=Image.new('RGB',(1920,1024),(24,36,44))
  for j,mode in enumerate(['wide','close']):
   with Image.open(B/f'frames_{mode}'/f'frame_{i:04d}.png') as f:im.paste(f.convert('RGB'),(960*j,64))
  d=ImageDraw.Draw(im);d.text((20,18),'NUMI LAB | FRANKA PICKUP + DROP | 10 COINS | 240 Hz | NORMAL SPEED',font=font,fill='white');d.text((1720,18),f't={frames[index]["time_s"]:.2f}s',font=font,fill='white');proc.stdin.write(im.tobytes())
 proc.stdin.close();assert proc.wait()==0
 call(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(video),'-f','null','-'])
 probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_entries','format=duration:stream=nb_frames,r_frame_rate','-of','json',str(video)],text=True));assert probe['streams'][0]['nb_frames']=='241'
 (OUT/'video_record.json').write_text(json.dumps({'physics_duration_s':8,'native_steps':1920,'native_hz':240,'video_fps':30,'normal_speed':True,'end_native_state_included':True,'probe':probe,'video_sha256':sha(video),'trajectory_sha256':sha(trajectory),'decode_pass':True,'renderer':'Native saved states only. No render-engine physics or invented motion.'},indent=2));print('VIDEO READY '+str(video),flush=True)
try:main()
except Exception:
 (OUT/'pipeline_failure.txt').write_text(traceback.format_exc());raise
