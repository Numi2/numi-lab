import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

base = os.path.dirname(os.path.abspath(__file__))
font = '/System/Library/Fonts/Supplemental/Arial.ttf'
bold = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
f_small = ImageFont.truetype(font, 17)
f_title = ImageFont.truetype(bold, 23)
f_status = ImageFont.truetype(bold, 20)
fg = (235, 241, 237, 255)
muted = (175, 190, 184, 255)
green = (119, 213, 169, 255)
bg = (25, 34, 35, 255)

head = Image.new('RGBA', (1920, 65), bg)
d = ImageDraw.Draw(head)
d.text((28, 10), 'UNUSED LAYOUT I', font=f_title, fill=fg)
d.text((28, 39), 'Seed 4038889268', font=f_small, fill=muted)
d.text((988, 10), 'UNUSED LAYOUT J', font=f_title, fill=fg)
d.text((988, 39), 'Seed 3193706597', font=f_small, fill=muted)
d.text((500, 14), 'Recorded simulation replay  ·  2× speed', font=f_small, fill=muted)
d.text((1460, 14), 'Teacher off  ·  no intermediate resets', font=f_small, fill=muted)
header = os.path.join(base, 'header.png')
head.save(header)

data = [np.load(os.path.join(base, 'inputs', f'reserved-final-{l}-visual.npz')) for l in 'ij']
cmd = ['/opt/homebrew/bin/ffmpeg', '-y', '-loglevel', 'error',
       '-i', os.path.join(base, 'layout-i.mp4'), '-i', os.path.join(base, 'layout-j.mp4'),
       '-loop', '1', '-framerate', '25', '-i', header,
       '-f', 'rawvideo', '-pixel_format', 'rgba', '-video_size', '1920x65', '-framerate', '25', '-i', '-',
       '-filter_complex', '[0:v][1:v]hstack=inputs=2,pad=1920:670:0:65:color=0x192223[base];'
                          '[base][2:v]overlay=0:0:shortest=1[head];'
                          '[head][3:v]overlay=0:605:shortest=1[out]',
       '-map', '[out]', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19',
       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', os.path.join(base, 'crow-navigation-base.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame in range(750):
    footer = Image.new('RGBA', (1920, 65), bg)
    draw = ImageDraw.Draw(footer)
    step = min(2999, round(frame * 2999 / 749))
    for panel, values in enumerate(data):
        x = panel * 960 + 28
        n = int(values['waypoints'][step])
        draw.text((x, 9), f'WAYPOINTS  {n} / 5', font=f_status, fill=fg)
        if frame == 749:
            draw.text((x + 230, 10), 'SUPPORTED LANDING', font=f_status, fill=green)
        else:
            draw.text((x + 230, 11), 'CONTINUOUS AUTONOMOUS RUN', font=f_small, fill=muted)
        draw.text((x, 39), f'Simulation time {step * .02:04.1f} / 60.0 s', font=f_small, fill=muted)
    proc.stdin.write(footer.tobytes())
proc.stdin.close()
if proc.wait():
    raise SystemExit('video composition failed')
subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-loglevel', 'error',
                '-i', os.path.join(base, 'crow-navigation-base.mp4'),
                '-vf', 'tpad=stop_mode=clone:stop_duration=2', '-an', '-c:v', 'libx264',
                '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                os.path.join(base, 'crow-navigation.mp4')], check=True)
os.remove(os.path.join(base, 'crow-navigation-base.mp4'))
