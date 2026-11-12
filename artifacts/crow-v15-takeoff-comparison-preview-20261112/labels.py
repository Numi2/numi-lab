from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
p=Path(__file__).parent
im=Image.new('RGBA',(1280,360),(0,0,0,0));d=ImageDraw.Draw(im)
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',23)
small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',17)
d.rectangle((0,0,1279,48),fill=(0,0,0,195));d.rectangle((0,326,1279,359),fill=(0,0,0,195))
d.text((20,12),'BEFORE - FALLS BACK',font=font,fill='white');d.text((660,12),'AFTER - AIRBORNE AT 8 S',font=font,fill='white')
d.text((18,334),'DEVELOPMENT REPLAY - ASSISTED FLIGHT - NOT A COMPLETE ROUTE OR LANDING',font=small,fill='white')
im.save(p/'labels.png')
