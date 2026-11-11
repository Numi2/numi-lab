from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
p = Path('/tmp/crow-video-compose-20261111')
labels = [
 ('ORIGINAL WALK', 'Development seed 20261127 | No unsupported substeps in this segment'),
 ('REDUCED-STRIDE WALK - UNSTABLE', 'Same start | 497 unsupported substeps, moved backward'),
 ('WING-TAPER LANDING - FALL', 'Development seed 20261126 | Tips backward after foot contact'),
 ('TIP RECOVERY - COLLISION', 'Same seed | Trace reports tail hitting the left gate'),
]
title = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 31)
sub = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 22)
for i, (a,b) in enumerate(labels):
 im = Image.new('RGBA', (1280, 112), (0,0,0,195))
 d = ImageDraw.Draw(im)
 d.text((32,14), a, font=title, fill='white')
 d.text((32,62), b, font=sub, fill='white')
 im.save(p / f'label-{i}.png')
