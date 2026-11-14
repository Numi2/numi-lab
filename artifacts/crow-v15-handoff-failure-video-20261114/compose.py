from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import hashlib,subprocess
A=Path('/Users/home/Documents/Codex/2026-11-07/c/crow-research-staging/artifacts/crow-v15-handoff-failure-video-corrected-20261114');walk=Path('/Users/home/Documents/Codex/2026-11-07/c/crow-walking-diagnostic-20261113.mp4')
assert hashlib.sha256(walk.read_bytes()).hexdigest()=='109c518f8a3db464af3b5bc6a7a0cf4bbb80bd7c9c7b4184e985fe17a4dcf0ff'
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
for view in ('side','front'):
 im=Image.new('RGBA',(1280,766),(0,0,0,0));d=ImageDraw.Draw(im);d.rectangle((0,720,1279,765),fill=(12,15,19,255));d.text((16,724),f'FAILED DEVELOPMENT RUN: {view.upper()} VIEW OF THE SAME REPLAY, USED SEED',font=font,fill='white');d.text((16,744),'Body-relative presentation; root motion removed; scenery and HUD illustrative; no landing',font=font,fill=(255,204,98,255));im.save(A/f'{view}-label.png')
cmd=['ffmpeg','-v','error','-y','-i',str(walk),'-i',str(A/'side.mp4'),'-i',str(A/'front.mp4'),'-framerate','50','-loop','1','-i',str(A/'side-label.png'),'-framerate','50','-loop','1','-i',str(A/'front-label.png'),'-filter_complex','[0:v]setsar=1,format=yuv420p,setpts=PTS-STARTPTS[w];[1:v]pad=1280:766:0:0:color=black[sa];[sa][3:v]overlay=shortest=1,trim=end_frame=668,setsar=1,format=yuv420p,setpts=PTS-STARTPTS[s];[2:v]pad=1280:766:0:0:color=black[fa];[fa][4:v]overlay=shortest=1,trim=end_frame=668,setsar=1,format=yuv420p,setpts=PTS-STARTPTS[f];[w][s][f]concat=n=3:v=1:a=0[out]','-map','[out]','-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(A/'handoff-failure.mp4')]
subprocess.run(cmd,check=True)
