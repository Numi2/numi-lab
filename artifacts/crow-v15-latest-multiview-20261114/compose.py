"""Combine five synchronized views of the same 2,400 accepted states."""
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

p = Path(__file__).resolve().parent
im = Image.new("RGBA", (1920, 960), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 23)
small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
for label, x, y in [('FRONT',0,0), ('BACK',640,0), ('ABOVE',0,480), ('BELOW',640,480), ('HIP / FEET CLOSE-UP',1280,0)]:
    d.rectangle((x,y,x+639,y+34), fill=(10,15,21,218))
    d.text((x+12,y+4), label, font=font, fill='white')
d.rectangle((1280,480,1919,959), fill=(16,22,29,255))
lines = [
    'LATEST 48-SECOND DEVELOPMENT REPLAY',
    'One recorded flight, five synchronized camera views',
    'Seed 20261127, lane 5, training layout',
    'Recorded at 50 physics control steps per second',
    '',
    'COURSE COLLISIONS DISABLED IN THIS TEST.',
    'Body-relative inspection; course geometry hidden.',
    'Angles show the same run, not five trials.',
    'Feet are shown in flight; grip is a separate test.',
    '',
    'One waypoint recorded; no completed mission or landing.',
    'Current flight controller is not yet qualified.',
]
for i, line in enumerate(lines):
    d.text((1301, 504+i*33), line, font=small, fill=(235,240,245) if i < 5 else (255,205,127))
im.save(p/'labels.png')
views = ('front-close','rear-close','top-close','below-close','hipfeet-close')
cmd = ['ffmpeg','-v','error','-y']
for view in views:
    cmd += ['-i',str(p/(view+'.mp4'))]
cmd += ['-loop','1','-framerate','50','-i',str(p/'labels.png'),'-filter_complex',
        '[0:v][1:v][2:v][3:v][4:v]xstack=inputs=5:layout=0_0|640_0|0_480|640_480|1280_0:fill=0x10161d:shortest=1[grid];[grid][5:v]overlay=shortest=1,format=yuv420p[v]',
        '-map','[v]','-an','-c:v','libx264','-preset','medium','-crf','20','-movflags','+faststart',str(p/'crow-latest-48s-multiview.mp4')]
subprocess.run(cmd,check=True)
