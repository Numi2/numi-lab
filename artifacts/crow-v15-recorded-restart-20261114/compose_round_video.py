from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import subprocess
root=Path(__file__).resolve().parent;p=root/'round-video'
im=Image.new('RGBA',(1920,960),(0,0,0,0));d=ImageDraw.Draw(im)
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',23);small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',19)
for label,x,y in [('FRONT',0,0),('BACK',640,0),('ABOVE',0,480),('BELOW',640,480),('HIP / FEET CLOSE-UP',1280,0)]:
 d.rectangle((x,y,x+639,y+60),fill=(10,15,21,255));d.text((x+12,y+4),label,font=font,fill='white')
d.rectangle((1280,480,1919,959),fill=(16,22,29,255))
lines=['ROUND PERCH: FAILED DEVELOPMENT TRIAL','Five views of the same recorded physics states','Reconstructed 48-second checkpoint; first episode only','','Static round collider was moved near the bird.','Landing mode was forced; this is not autonomous','navigation to the original perch.','','Both feet met grip and balance gates for 1.84 s.','Body tipped; first termination at 4.18 s.','No supported landing or sustained hold demonstrated.','','Original flight had course collisions disabled.','Bird-relative cameras: ground motion is centered.']
for i,line in enumerate(lines):d.text((1300,501+i*31),line,font=small,fill=(255,190,110) if i in (0,4,5,6,8,9,10,12) else (235,240,245))
labels=p/'round-labels.png';im.save(labels)
original=root.parent/'crow-latest-multiview-20261114/crow-latest-48s-multiview.mp4'
cmd=['ffmpeg','-v','error','-y','-i',str(original)]
for name in ('front','back','above','below','feet'):cmd+=['-i',str(p/(name+'.mp4'))]
cmd+=['-loop','1','-framerate','50','-i',str(labels)]
graph=('[1:v][2:v][3:v][4:v][5:v]xstack=inputs=5:layout=0_0|640_0|0_480|640_480|1280_0:fill=0x10161d:shortest=1[grid];'
       '[grid][6:v]overlay=shortest=1[labeled];[0:v]setpts=PTS-STARTPTS[original];'
       '[labeled]setpts=PTS-STARTPTS[resumed];[original][resumed]concat=n=2:v=1:a=0,format=yuv420p[v]')
cmd+=['-filter_complex',graph,'-map','[v]','-an','-c:v','libx264','-preset','veryfast','-crf','21','-movflags','+faststart',str(p/'crow-48s-plus-round-perch-failure.mp4')]
subprocess.run(cmd,check=True)
