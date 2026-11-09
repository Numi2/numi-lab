from pathlib import Path
import json,subprocess
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).parent;S=P/'video-full-sections';out=P/'crow-rigid-foot-route-both-layouts.mp4';font='/System/Library/Fonts/Supplemental/Arial.ttf';bold='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
header=Image.new('RGB',(1920,66),(24,32,35));d=ImageDraw.Draw(header);d.text((24,8),'RECORDED SIMULATION  •  DEVELOPMENT LAYOUTS',font=ImageFont.truetype(bold,25),fill=(240,244,242));d.text((24,39),'Rigid feet in these runs; articulated toe grip is not shown  |  All accepted states at 50 fps; landing pose uses estimated presentation cues',font=ImageFont.truetype(font,19),fill=(205,215,210));header.save(P/'header.png')
for layout in 'ij':
 files=sorted(S.glob(layout+'-[0-9][0-9][0-9][0-9].mp4'));assert len(files)==5,files
 (P/(layout+'-concat.txt')).write_text(''.join("file '"+str(f)+"'\n" for f in files))
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(P/(layout+'-concat.txt')),'-c','copy',str(P/(layout+'-joined.mp4'))],check=True)
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(P/'i-joined.mp4'),'-i',str(P/'j-joined.mp4'),'-loop','1','-framerate','50','-i',str(P/'header.png'),'-filter_complex','[0:v][1:v]hstack=inputs=2,pad=1920:606:0:66:color=0x182023[base];[base][2:v]overlay=0:0:shortest=1[out]','-map','[out]','-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
p=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(out)],capture_output=True,text=True,check=True);j=json.loads(p.stdout);print(json.dumps(j,indent=2));assert int(j['streams'][0]['nb_read_frames'])==3000
subprocess.run(['ffmpeg','-v','error','-i',str(out),'-f','null','-'],check=True)
