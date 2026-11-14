from pathlib import Path
import hashlib, json, subprocess

r = Path('/Users/n/Developer/numi-crow-rl-20261107')
a = r / 'artifacts/crow-v15-recorded-restart-20261114'
b = r / 'build/crow-v15-recorded-restart-20261114'
old = (a / 'restart-near-perch.metal').read_text()
needle = 'state.navigation.z=1.0f;'
assert old.count(needle) == 1
source = a / 'restart-near-perch-landingmode.metal'
source.write_text(old.replace(needle, 'state.navigation.z=5.0f;'))
build = json.loads((a / 'shader-build.json').read_text())
cmd = build['compile']
cmd[cmd.index('-c') + 1] = str(source)
cmd[cmd.index('-o') + 1] = str(b / 'restart-near-perch-landingmode.air')
subprocess.run(cmd, check=True)
link = build['link']
link[link.index(str(b / 'restart.air'))] = str(b / 'restart-near-perch-landingmode.air')
link[link.index('-o') + 1] = str(b / 'restart-near-perch-landingmode.metallib')
subprocess.run(link, check=True)
(a / 'landingmode-shader-build.json').write_text(json.dumps({
    'restored_navigation_phase': 5, 'compile': cmd, 'link': link,
    'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (source, b / 'restart-near-perch-landingmode.air', b / 'restart-near-perch-landingmode.metallib')}
}, indent=2) + '\n')
print('built landing-phase diagnostic shader')
