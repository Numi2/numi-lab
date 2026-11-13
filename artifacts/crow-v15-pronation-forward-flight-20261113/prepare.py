from pathlib import Path
import difflib,hashlib,json,shutil,subprocess
R=Path.cwd();new='crow-v15-pronation-forward-flight-20261113';old='crow-v15-consistent-long-flight-20261113';A=R/'artifacts'/new;B=R/'build'/new;O=R/'artifacts'/old
for name in ('push-wing','hold-wing'):
 base=(O/(name+'.metal')).read_text();needle='const float amplitude=environment%8u<4u ? 0.25f : environment%8u<6u ? 0.45f : 0.65f;';assert base.count(needle)==1;s=base.replace(needle,'const float amplitude=0.65f;');needle='''        // Exploratory, bounded knee target feedback through the existing position drive.''';assert s.count(needle)==1
 add='''        // Research-only bounded wing pronation sensitivity, no direct body wrench.
        if (fixedFlightController && state.episode.z==10u && (action==4u || action==5u) && environment%8u>=2u) {
            const float seconds=float(state.episode.x)*dispatch.timing.x;
            if (seconds>=7.04f && seconds<10.02f) {
                const float offset=environment%8u<4u ? 0.0f : environment%8u<6u ? 1.57079632679f : -1.57079632679f;
                const float phase=kTwoPi*6.4f*(seconds-7.04f)+offset;
                const float coordinated=defaultQ[binding.indices.z]+0.18f*sin(phase);
                targetCandidate=mix(targetCandidate,coordinated,smoothstep(7.04f,7.12f,seconds));
            }
        }
''';s=s.replace(needle,add+needle);(A/(name+'.metal')).write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/long.metal',tofile='b/pronation.metal')))
shutil.copyfile(R/'build'/old/'World.air',B/'World.air');s=(O/'build.py').read_text().replace(old,new);(A/'build.py').write_text(s)
for name in ('push-wing','hold-wing'):subprocess.run(['python3',str(A/'build.py'),name],check=True)
s=(O/'run.py').read_text().replace(old,new).replace("L=R/'build/crow-v15-pronation-forward-flight-20261113'", "L=R/'build/crow-v15-consistent-long-flight-20261113'");(A/'run.py').write_text(s)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'build-hashes.json').write_text(json.dumps({str(p.relative_to(R)):h(p) for p in (B/'push-wing.metallib',B/'hold-wing.metallib',B/'World.air')},indent=2)+'\n');print((A/'build-hashes.json').read_text())
