from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).parent
image = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
for y in (0, 360):
    draw.rectangle((0, y, 1279, y + 38), fill=(0, 0, 0, 205))
draw.text((14, 9), "BASELINE | SIDE", font=font, fill="white")
draw.text((654, 9), "CORVID-INFORMED TEST | SIDE", font=font, fill="white")
draw.text((14, 369), "BASELINE | FRONT", font=font, fill="white")
draw.text((654, 369), "CORVID-INFORMED TEST | FRONT", font=font, fill="white")
draw.rectangle((0, 683, 1279, 719), fill=(0, 0, 0, 205))
draw.text((14, 692), "DEVELOPMENT WALK REPLAY | CANDIDATE FAILED CONTACT-MOTION TEST | NOT A COMPLETE MISSION", font=small, fill="white")
image.save(root / "labels.png")
