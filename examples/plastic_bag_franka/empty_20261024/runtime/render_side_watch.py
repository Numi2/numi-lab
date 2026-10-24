"""Render a second inspection angle of the same immutable native prefixes."""
from pathlib import Path
import json,subprocess,time,hashlib
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');W=ROOT/'work/plastic_bag_franka_contact_finish_20261024';OUT=ROOT/'outputs/plastic_bag_franka_contact_finish_20261024';B=Path('/private/tmp/numi_bag_franka_contact_finish_20261024');last=0
while True:
 status=json.loads((OUT/'STATUS.json').read_text());final=OUT/'trajectory.json'
 if status['status']=='needs_attention':raise RuntimeError(status['error'])
 if final.exists():p=final;done=True
 else:
  paths=list(B.glob('combined_[0-9]*.json'))
  if not paths:time.sleep(25);continue
  p=max(paths,key=lambda x:x.name);done=False
 fs=json.loads(p.read_text())['frames']
 if done or len(fs)-last>=64:
  with (B/'render_side.log').open('a') as log:subprocess.run(['/Applications/Blender.app/Contents/MacOS/Blender','-b','--python-exit-code','1','--python',str(W/'runtime/render.py'),'--',str(p),str(B/'frames_side'),'side'],check=True,stdout=log,stderr=subprocess.STDOUT)
  last=len(fs);print('Rendered side through native state',last-1,flush=True)
 if done:break
 time.sleep(25)
assert len(fs)==1921 and fs[-1]['time_s']==8
video=OUT/'stock_grip_side_8s_normal.mp4'
subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-v','error','-framerate','30','-i',str(B/'frames_side/frame_%04d.png'),'-frames:v','241','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_entries','format=duration:stream=nb_frames,r_frame_rate','-of','json',str(video)],text=True));assert probe['streams'][0]['nb_frames']=='241'
(OUT/'side_video_record.json').write_text(json.dumps({'same_native_trajectory':str(final),'normal_speed':True,'native_hz':240,'video_fps':30,'end_native_state_included':True,'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'trajectory_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'decode_pass':True,'probe':probe},indent=2))
print('SIDE VIDEO READY',video,flush=True)
