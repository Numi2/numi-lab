from pathlib import Path
import hashlib, json, subprocess, sys

sign = sys.argv[1]
assert sign in ('negative', 'positive')
r = Path('/Users/n/Developer/numi-crow-rl-20261107')
a = r / 'artifacts/crow-v15-recorded-restart-20261114'
b = r / 'build/crow-v15-recorded-restart-20261114'
old = (a / 'restart-near-perch-landingmode.metal').read_text()
needle = '''        const float target = clamp(
            targetCandidate,
            binding.parameters.y,'''
assert old.count(needle) == 1
coefficient = '-1.0f' if sign == 'negative' else '1.0f'
insert = '''        // Exploratory balance is gated by actual bilateral toe contact ramps.
        // It changes only bounded physical hip position targets; the solver
        // and contact impulses remain authoritative.
        if (state.navigation.z>=5.0f && (action==7u || action==10u) &&
            (state.status.w & 63u)>=12u && ((state.status.w>>6u)&63u)>=12u) {
            const float correction=clamp(SIGN*(0.40f*(state.threatGeometry.z-0.10f)+
                                               0.08f*state.threatGeometry.w),-0.12f,0.12f);
            targetCandidate+=correction;
        }
'''.replace('SIGN', coefficient)
source = a / ('restart-pitch-balance-' + sign + '.metal')
source.write_text(old.replace(needle, insert + needle))
build = json.loads((a / 'shader-build.json').read_text())
cmd = build['compile']; cmd[cmd.index('-c')+1] = str(source)
air = b / ('restart-pitch-balance-' + sign + '.air')
lib = b / ('restart-pitch-balance-' + sign + '.metallib')
cmd[cmd.index('-o')+1] = str(air); subprocess.run(cmd, check=True)
link = build['link']; link[link.index(str(b/'restart.air'))] = str(air)
link[link.index('-o')+1] = str(lib); subprocess.run(link, check=True)
(a/('pitch-balance-'+sign+'-build.json')).write_text(json.dumps({'sign':sign,'compile':cmd,'link':link,
    'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,air,lib)}},indent=2)+'\n')
print('built', sign)
