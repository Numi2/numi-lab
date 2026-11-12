from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
p=Path(__file__).parent
im=Image.new('RGBA',(1280,360),(0,0,0,0));d=ImageDraw.Draw(im)
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',21)
small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',16)
d.rectangle((0,0,1279,45),fill=(0,0,0,195));d.rectangle((0,326,1279,359),fill=(0,0,0,195))
d.text((16,12),'NO EARLY WINGS: GROUNDED AT 10 S',font=font,fill='white')
d.text((655,12),'EARLY WINGS: AIRBORNE AT 10 S',font=font,fill='white')
d.text((15,334),'DEVELOPMENT REPLAY | BODY MOMENTS OFF FROM 6.3 S | NOT A COMPLETE ROUTE OR LANDING',font=small,fill='white')
im.save(p/'labels.png')
