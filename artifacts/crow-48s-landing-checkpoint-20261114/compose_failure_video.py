"""Present the original 48s and an honestly labeled failed continuation."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess

root=Path(__file__).resolve().parent
p=root/'failure-video'
im=Image.new('RGBA',(1920,960),(0,0,0,0));d=ImageDraw.Draw(im)
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',23)
small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',19)
for label,x,y in [('FRONT',0,0),('BACK',640,0),('ABOVE',0,480),('BELOW',640,480),('HIP / FEET CLOSE-UP',1280,0)]:
    # Suppress the renderer's misleading 5/5 waypoint HUD: landing phase was forced.
    d.rectangle((x,y,x+639,y+60),fill=(10,15,21,255));d.text((x+12,y+4),label,font=font,fill='white')
d.rectangle((1280,480,1919,959),fill=(16,22,29,255))
lines=['FAILED LANDING DEVELOPMENT TRIAL',
       'Recorded physics replay, five synchronized views',
       'Starts from the reconstructed 48-second state',
       '',
       'Perch moved near bird; landing mode forced.',
       'The fixture is a stationary box, not a natural branch.',
       '',
       'TILT FAILURES AND AUTOMATIC RESTARTS:',
       'At 3.14, 8.92 and 14.70 s of this segment.',
       'These are separate attempts, not one continuous hold.',
       '',
       'Stable autonomous landing is NOT demonstrated.',
       'Earlier flight had course collisions disabled.',
       'Forced phase does not mean waypoints were reached.']
for i,line in enumerate(lines):
    d.text((1300,501+i*31),line,font=small,fill=(255,190,110) if i in (0,4,5,7,8,9,11,12) else (235,240,245))
labels=p/'failure-labels.png';im.save(labels)
warning=Image.new('RGB',(640,50),(190,25,25));wd=ImageDraw.Draw(warning)
wd.text((30,9),'AUTO RESET AFTER TILT FAILURE',font=font,fill='white')
warning_path=p/'reset-warning.png';warning.save(warning_path)
original=root.parent/'crow-latest-multiview-20261114/crow-latest-48s-multiview.mp4'
cmd=['ffmpeg','-v','error','-y','-i',str(original)]
for name in ('front','back','above','below','feet'):cmd+=['-i',str(p/(name+'.mp4'))]
cmd+=['-loop','1','-framerate','50','-i',str(labels),'-loop','1','-framerate','50','-i',str(warning_path)]
alert="between(t,3.14,3.94)+between(t,8.92,9.72)+between(t,14.70,15.50)"
graph=('[1:v][2:v][3:v][4:v][5:v]xstack=inputs=5:layout=0_0|640_0|0_480|640_480|1280_0:fill=0x10161d:shortest=1[grid];'
       '[grid][6:v]overlay=shortest=1[labeled];'
       "[labeled][7:v]overlay=x=640:y=70:shortest=1:enable='"+alert+"'[segment];"
       '[0:v]setpts=PTS-STARTPTS[original];[segment]setpts=PTS-STARTPTS[resumed];'
       '[original][resumed]concat=n=2:v=1:a=0,format=yuv420p[v]')
cmd+=['-filter_complex',graph,'-map','[v]','-an','-c:v','libx264','-preset','veryfast','-crf','21','-movflags','+faststart',str(p/'crow-48s-plus-failed-landing-multiview.mp4')]
subprocess.run(cmd,check=True)
