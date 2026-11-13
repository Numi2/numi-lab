from pathlib import Path
import difflib,hashlib,json,shutil,subprocess
R=Path.cwd();new='crow-v15-consistent-wingheld-leg-control-20261113';previous='crow-v15-consistent-wing-drive-20261113';amp='crow-v15-passive-aero-flap-amplitude-20261113';A=R/'artifacts'/new;B=R/'build'/new;O=R/'artifacts'/amp
for name in ('push-wing','hold-wing'):
 old=(O/(name+'.metal')).read_text();needle='''        // Exploratory, bounded knee target feedback through the existing position drive.''';assert old.count(needle)==1
 insert='''        // Research-only wing-held control: actual forces and motion remain observed.
        if (fixedFlightController && state.episode.z==10u && action<=5u && float(state.episode.x)*dispatch.timing.x>=6.30f) {
            targetCandidate=defaultQ[binding.indices.z];
        }
''';s=old.replace(needle,insert+needle);(A/(name+'.metal')).write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/wing-on.metal',tofile='b/wing-held.metal')))
shutil.copyfile(R/'build'/amp/'World.air',B/'World.air');s=(O/'build.py').read_text().replace(amp,new).replace("R/'build/crow-v15-consistent-wingheld-leg-control-20261113/libmetalrobo.dylib'", "R/'build/crow-v15-consistent-wing-drive-20261113/libmetalrobo.dylib'");(A/'build.py').write_text(s)
for name in ('push-wing','hold-wing'):subprocess.run(['python3',str(A/'build.py'),name],check=True)
s=(R/'artifacts'/previous/'run.py').read_text().replace(previous,new).replace("M=R/'build/crow-v15-consistent-wingheld-leg-control-20261113'", "M=R/'build/crow-v15-consistent-wing-drive-20261113'").replace("L=R/'build/crow-v15-consistent-wingheld-leg-control-20261113'", "L=R/'build/crow-v15-consistent-wing-drive-20261113'");(A/'run.py').write_text(s)
for f in ('analyze.py','com-analyze.py','target-pair-check.py','drive-validation.py','power.py'):
 s=(R/'artifacts'/previous/f).read_text().replace(previous,new).replace('rebound-consistent-rebound','rebound-wingheld');(A/f).write_text(s)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'build-hashes.json').write_text(json.dumps({str(p.relative_to(R)):h(p) for p in (B/'push-wing.metallib',B/'hold-wing.metallib',B/'World.air')},indent=2)+'\n');print((A/'build-hashes.json').read_text())
