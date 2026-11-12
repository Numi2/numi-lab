from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
import sys
p=Path(sys.argv[1]);font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',25);small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',20)
for seed in ('20261126','20261127'):
 im=Image.new('RGBA',(1280,84),(0,0,0,170));d=ImageDraw.Draw(im);d.text((26,9),f'DEVELOPMENT WALK - SEED {seed}',font=font,fill='white');d.text((26,46),'TRAINING RUN - NOT A COMPLETE FLIGHT OR LANDING',font=small,fill='white');im.save(p/f'label-{seed}.png')
