"""Draw recorded toe-body centres from the archived stationary grip test.

This is a schematic of measured body poses, not a surface render or contact map.
"""
import gzip
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
frames = {i: {"poses": {}} for i in range(450)}
with gzip.open(HERE / "grip-trace.log.gz", "rt") as f:
    for line in f:
        p = line.split()
        if len(p) < 4 or p[1] != "0":
            continue
        t = int(p[2])
        if p[0] == "pose":
            frames[t]["poses"][p[3]] = tuple(map(float, p[4:7]))
        elif p[0] == "step":
            qi, vi = p.index("q"), p.index("v")
            frames[t]["q"] = list(map(float, p[qi + 1:vi]))
assert all(len(x["poses"]) == 18 and len(x["q"]) == 41 for x in frames.values())

font = "/System/Library/Fonts/Helvetica.ttc"
large = ImageFont.truetype(font, 23)
normal = ImageFont.truetype(font, 17)
small = ImageFont.truetype(font, 14)
proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "960x540", "-framerate", "50", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(HERE / "latest-grip-trace.mp4")], stdin=subprocess.PIPE)
for t, data in frames.items():
    im = Image.new("RGB", (960, 540), (17, 21, 27))
    d = ImageDraw.Draw(im)
    d.text((20, 12), "LATEST STATIONARY GRIP TEST — lane 0 / control", font=large, fill="white")
    d.text((20, 43), "Recorded body centres, side projection; connecting lines are schematic", font=normal, fill=(192, 205, 218))
    for side_i, side in enumerate(("left", "right")):
        x0 = 24 + side_i * 466
        d.rectangle((x0, 82, x0 + 442, 412), fill=(26, 33, 42), outline=(84, 99, 115))
        d.text((x0 + 12, 91), side.upper() + " FOOT", font=normal, fill="white")
        # Fixed world coordinates in metres, with 0.05 m perch width and top z=0.
        def xy(p):
            return (x0 + 221 + (p[0] - .025) * 2500, 322 - p[2] * 2500)
        d.rectangle((*xy((-.025, 0, 0)), *xy((.025, 0, -.03))), fill=(82, 76, 61), outline=(160, 150, 123))
        foot = data["poses"][f"crow_{side}_foot"]
        colors = [(255, 200, 101), (255, 154, 90), (253, 124, 120), (114, 197, 245)]
        for idx, digit in enumerate(("front_toe_1", "front_toe_2", "front_toe_3", "rear_toe")):
            a = data["poses"][f"crow_{side}_{digit}_proximal"]
            b = data["poses"][f"crow_{side}_{digit}_distal"]
            points = [xy(foot), xy(a), xy(b)]
            d.line(points, fill=colors[idx], width=3)
            for x, y in points[1:]:
                d.ellipse((x-4, y-4, x+4, y+4), fill=colors[idx])
        x, y = xy(foot)
        d.ellipse((x-5, y-5, x+5, y+5), fill="white")
        d.text((x0+12, 390), "amber: front toes   blue: rear toe   white: foot", font=small, fill=(190, 202, 214))
    state = "release / return" if t < 297 else ("brief opposed support: 11 intervals" if t <= 307 else ("grip not sustained" if t < 370 else "fall detected; grip test failed"))
    d.text((24, 430), f"Step {t:03d}/449   Time {t * .02:4.2f} s   {state}", font=normal, fill=(255, 207, 125))
    q = data["q"]
    d.text((24, 458), f"Recorded hip joint values (rad): L pitch {q[14]:+.2f}, R pitch {q[17]:+.2f}, L roll {q[39]:+.2f}, R roll {q[40]:+.2f}", font=normal, fill=(208, 219, 229))
    d.text((24, 487), "The 100-interval terminal grip requirement was not met. Contact and pose epochs are not independently verified.", font=small, fill=(195, 204, 215))
    proc.stdin.write(im.tobytes())
proc.stdin.close()
assert proc.wait() == 0
