"""Create a compact visual contact sheet from the final saved-state presentation."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');O=ROOT/'outputs/plastic_bag_empty_franka_20261024';B=Path('/private/tmp/numi_empty_bag_franka_20261024')
assert json.loads((O/'STATUS.json').read_text())['status']=='complete'
frames=[0,30,90,141,180,186,195,201,225,240];im=Image.new('RGB',(5*384,2*414),(24,36,44));d=ImageDraw.Draw(im);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
for j,k in enumerate(frames):
 with Image.open(B/'frames_close'/f'frame_{k:04d}.png') as f:im.paste(f.resize((384,384)),((j%5)*384,(j//5)*414+30))
 d.text(((j%5)*384+10,(j//5)*414+5),f't={k/30:.2f}s',fill='white',font=font)
im.save(O/'inspection_contact_sheet.jpg',quality=92)
print(O/'inspection_contact_sheet.jpg')
