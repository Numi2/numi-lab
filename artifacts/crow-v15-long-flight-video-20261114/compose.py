from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import hashlib,subprocess
S=Path('/Users/home/Documents/Codex/2026-11-07/c/crow-research-staging/artifacts/crow-v15-long-flight-video-20261114'); P=Path('artifacts/crow-v15-handoff-failure-video-20261114/handoff-failure.mp4')
assert hashlib.sha256(P.read_bytes()).hexdigest()=='2f8ffdbe45b000611145e16a782887c96d1dd2383403e2503eab4a4311872a5f'
assert hashlib.sha256((S/'side.mp4').read_bytes()).hexdigest()=='2851064b0ad98b51a3c5b0f036bfd9cc364831e75f6e3f2a48d4ab15933ced27'
im=Image.new('RGBA',(1280,766),(0,0,0,0));d=ImageDraw.Draw(im);font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18);d.rectangle((0,720,1279,765),fill=(12,15,19,255));d.text((16,724),'DEVELOPMENT FLIGHT: USED SEED AND TRAINING SCENE; NO COMPLETED MISSION OR LANDING',font=font,fill='white');d.text((16,744),'Body-relative replay: root movement removed; scenery and waypoint display are illustrative',font=font,fill=(255,204,98,255));im.save(S/'long-label.png')
cmd=['ffmpeg','-v','error','-y','-i',str(P),'-i',str(S/'side.mp4'),'-framerate','50','-loop','1','-i',str(S/'long-label.png'),'-filter_complex','[0:v]setsar=1,format=yuv420p,setpts=PTS-STARTPTS[a];[1:v]pad=1280:766:0:0:color=black[b];[b][2:v]overlay=shortest=1,trim=end_frame=1800,setsar=1,format=yuv420p,setpts=PTS-STARTPTS[c];[a][c]concat=n=2:v=1:a=0[out]','-map','[out]','-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(S/'crow-development-progress.mp4')];subprocess.run(cmd,check=True)
