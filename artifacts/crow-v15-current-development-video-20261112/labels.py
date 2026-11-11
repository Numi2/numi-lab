from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
p=Path("/tmp/crow-current-development-video-20261112");p.mkdir(exist_ok=True)
labels=[("LEG AND WING TAKEOFF - DEVELOPMENT","7.00 s wing gate | used seed and layout | half speed"),("EARLY-WING LANDING - FAILURE","Earlier 6.90 s gate | lane 5 grips, then tips | used layout")]
title=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",31);sub=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",22)
for i,(a,b) in enumerate(labels):
 im=Image.new("RGBA",(1280,112),(0,0,0,195));d=ImageDraw.Draw(im);d.text((32,14),a,font=title,fill="white");d.text((32,62),b,font=sub,fill="white");im.save(p/f"label-{i}.png")
