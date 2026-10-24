from pathlib import Path
import sys,json
from PIL import Image,ImageDraw,ImageFont
out=Path(sys.argv[1]);data=json.loads((out/'construction.json').read_text())
canvas=Image.new('RGB',(1800,940),(23,34,43));draw=ImageDraw.Draw(canvas)
font='/System/Library/Fonts/Supplemental/Arial.ttf'
title=ImageFont.truetype(font,31);label=ImageFont.truetype(font,23);small=ImageFont.truetype(font,20)
draw.text((24,17),'NUMI LAB  |  12 × 16 INCH BOTTOM-SEAL BAG',font=title,fill='white')
draw.text((24,57),'304.8 × 406.4 mm  ·  75 µm clear LDPE  ·  empty  ·  construction preview',font=label,fill=(191,220,226))
for i,(name,caption) in enumerate([('front','Front · open top / bottom heat seal'),('three_quarter','Angled view · no side welds'),('mouth','Mouth detail · continuous side fold')]):
    with Image.open(out/'views'/(name+'.png')) as im:canvas.paste(im.convert('RGB').resize((600,791),Image.Resampling.LANCZOS),(i*600,96))
    draw.text((i*600+18,900),caption,font=small,fill='white')
canvas.save(out/'construction_preview.png')
