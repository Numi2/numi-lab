"""Verify and encode the completed native handoff, never a preflight clip."""
from pathlib import Path
import subprocess,json,time,csv,hashlib,sys,numpy as np
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).resolve().parents[1];O=W.parents[1]/'outputs'/W.name;B=Path('/private/tmp/numi_bag_franka_handoff_20261024');P=O.parent/'plastic_bag_empty_franka_20261024';BLENDER='/Applications/Blender.app/Contents/MacOS/Blender'
def call(a,**kw):return subprocess.run(a,check=True,**kw)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
while not ((O/'trajectory.json').exists() and (O/'finish/run_summary.json').exists() and (O/'watch_finish.log').exists() and 'FULL NATIVE RUN AND RENDERS COMPLETE' in (O/'watch_finish.log').read_text()):
 for n in ['capture_contact_watch.json','finish_contact_watch.json']:
  if (O/n).exists() and json.loads((O/n).read_text())['intersection_state_count']:raise RuntimeError('Native geometry needs correction')
 time.sleep(20)
p=O/'trajectory.json';fs=json.loads(p.read_text())['frames'];assert len(fs)==1921 and fs[-1]['step']==1920
with (O/'handoff_metrics.log').open('w') as log:call([sys.executable,str(W/'runtime/handoff_metrics.py'),str(O)],stdout=log,stderr=subprocess.STDOUT)
cache_path=O/'precomputed_audit_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {};assets=W.parent/'plastic_bag_franka/franka_assets'
cache_valid=(cache.get('trajectory_sha256')==sha(p) and cache.get('source_sha256')=={n:sha(W/'runtime'/n) for n in ['robot_film_overlap.py','interrobot_overlap.py','grip_alignment_audit.py','grip_alignment_single.py']} and cache.get('pad_input_sha256')=={n:sha(W/n) for n in ['grip.csv','grip2.csv']} and cache.get('asset_sha256')=={str(x.relative_to(assets)):sha(x) for x in sorted(assets.rglob('*')) if x.is_file()})
if cache_valid:
 g=cache['reports']['grip_alignment'];(O/'grip_alignment.json').write_text(json.dumps(g,indent=2))
else:
 with (O/'grip_alignment.json').open('w') as log:call([sys.executable,str(W/'runtime/grip_alignment_audit.py'),str(p)],stdout=log)
 g=json.loads((O/'grip_alignment.json').read_text())
assert g['all_forces_at_actual_pad_faces']
for script,tag,name in [('robot_film_overlap.py','ROBOT_FILM_AUDIT ','robot_film_full'),('interrobot_overlap.py','INTERROBOT_AUDIT ','interrobot_full')]:
 if cache_valid:a=cache['reports'][name]
 else:
  with (O/(name+'.log')).open('w') as log:call([BLENDER,'-b','--python-exit-code','1','--python',str(W/'runtime'/script),'--',str(p)],stdout=log,stderr=subprocess.STDOUT)
  a=json.loads(next(l.split(tag,1)[1] for l in (O/(name+'.log')).read_text().splitlines() if l.startswith(tag)))
 assert a['intersection_state_count']==0;(O/(name+'.json')).write_text(json.dumps(a,indent=2))
(O/'audit_reuse_receipt.json').write_text(json.dumps({'exact_input_and_source_and_all_asset_hashes_matched':cache_valid,'trajectory_sha256':sha(p),'origin':'Full-scene audits executed in this request before the video finished rendering.'},indent=2))
# Every accepted physical interval, with no use of rejected reference holds.
accuracy=[];balances=[];energy=[]
for path,low,high in [(P/'prefix',1,720),(P/'combined',721,1200),(O/'capture',1201,1560),(O/'finish_prefix',1561,1680),(O/'finish',1681,1920)]:
 for file,rows in [('solver_accuracy.csv',accuracy),('energy_balance.csv',balances),('energy_history.csv',energy)]:rows.extend(x for x in csv.DictReader((path/file).open()) if low<=int(x['step'])<=high)
trials=accuracy;accuracy=list({int(x['step']):x for x in trials}.values())
for rows in [accuracy,balances,energy]:assert [int(x['step']) for x in rows]==list(range(1,1921))
assert all(int(x['energy_bad'])==int(x['film_bad'])==0 for x in accuracy)
errors=np.array([float(x['frame_energy_balance_J']) for x in balances]);tolerance=np.array([float(x['local_tolerance_J']) for x in balances]);assert np.all(errors<=tolerance+1e-12)
m=json.loads((O/'handoff_metrics.json').read_text());m.update(handling_plan=json.loads((O/'handling_plan.json').read_text()),kernel_identity=json.loads((O/'film_kernel_identity.json').read_text()),runtime_identity=json.loads((O/'direct_runtime_binding.json').read_text()),total_rejected_trials=sum(int(x['energy_bad'])!=0 or int(x['film_bad'])!=0 for x in trials),recorded_trial_rows=len(trials),accepted_substeps=sorted(set(int(x['substeps']) for x in accuracy)),accepted_refinement=sorted(set(int(x['material_refinement']) for x in accuracy)),max_frame_energy_balance_J=float(errors.max()),max_balance_to_tolerance=float((errors/tolerance).max()),final_film_kinetic_J=float(energy[-1]['kinetic_J']),final_mass_weighted_rms_speed_m_s=float(np.sqrt(2*float(energy[-1]['kinetic_J'])/m['film_mass_kg'])),prefix_source_sha256=sha(P/'trajectory.json'),trajectory_sha256=sha(p),film_solver_preservation=json.loads((O/'solver_preservation.json').read_text()),actual_support_before_release=json.loads((O/'two_hand_support_before_release.json').read_text()),scene_geometry_binding=json.loads((O/'scene_geometry_binding.json').read_text()))
(O/'assessment.json').write_text(json.dumps(m,indent=2))
video=O/'empty_bag_two_franka_handoff_8s_normal.mp4';proc=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1040','-r','30','-i','-','-an','-frames:v','241','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',25);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',22)
for ordinal,index in enumerate(range(0,1921,8)):
 im=Image.new('RGB',(1920,1040),(24,36,44))
 for j,mode in enumerate(['wide','close']):
  with Image.open(B/f'final_frames_{mode}'/f'frame_{ordinal:04d}.png') as image:im.paste(image.convert('RGB'),(960*j,80))
 d=ImageDraw.Draw(im);d.text((18,10),'NUMI LAB | EMPTY BAG | TWO FRANKA HANDOFF | 240 Hz | NORMAL SPEED',font=font,fill='white');d.text((1670,10),f'{index/240:.2f} / 8.00 s',font=font,fill='white')
 active=fs[index]['grip_contact_nodes_by_robot'];label='both hands' if all(active) else 'first hand (right)' if active[0] else 'second hand (left)' if active[1] else 'none';d.text((18,46),'Gripper contact: '+label,font=small,fill=(190,226,230));d.text((980,46),'Contact view',font=small,fill=(190,226,230));proc.stdin.write(im.tobytes())
proc.stdin.close();assert proc.wait()==0
call(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(video),'-f','null','-'])
probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_entries','format=duration:stream=nb_frames,r_frame_rate,width,height','-of','json',str(video)],text=True));assert probe['streams'][0]['nb_frames']=='241'
record={'video':str(video),'native_hz':240,'native_states':1921,'physics_duration_s':8,'normal_motion':True,'endpoint_included':True,'decoded_without_errors':True,'video_sha256':sha(video),'trajectory_sha256':sha(p),'probe':probe};(O/'video_record.json').write_text(json.dumps(record,indent=2));(O/'STATUS.json').write_text(json.dumps({'status':'complete','video':str(video)},indent=2));print('VIDEO READY '+str(video),flush=True)
