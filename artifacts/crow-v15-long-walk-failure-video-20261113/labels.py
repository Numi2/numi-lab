from PIL import Image, ImageDraw, ImageFont
p='labels.png'
im=Image.new('RGBA',(1280,766),(0,0,0,0));d=ImageDraw.Draw(im)
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',19); small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
for x,y,t,color in [(0,0,'SUPPORT FAILURE  ·  SIDE VIEW',(244,195,175,255)),(640,0,'SUPPORT FAILURE  ·  FRONT VIEW',(244,195,175,255)),(0,360,'COMPARISON LANE  ·  SIDE VIEW',(224,235,235,255)),(640,360,'COMPARISON LANE  ·  FRONT VIEW',(224,235,235,255))]:
 d.rectangle((x,y,x+639,y+31),fill=(14,22,25,225));d.text((x+15,y+6),t,font=font,fill=color)
d.rectangle((0,720,1280,766),fill=(14,22,25,255));d.text((15,732),'SIMULATED REPLAY  ·  8-second development walk  ·  same used seed  ·  no mission or landing shown',font=small,fill=(224,235,235,255))
im.save(p)
