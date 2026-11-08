from pathlib import Path
import json,shlex,subprocess,hashlib,re
p=Path('/Users/numi/ccd-frictionless-oblique-20261108'); root=Path('/Users/numi/numi-lab-reconciliation-20261108')
s=(root/'src/metal/MetalWorldContact.metal').read_text()
helper='''inline void frictionlessNormalBlock(
    thread float effective[3][3],
    device const MREvaluatedConstraintIRConeGPU& cone
) {
    if (cone.effectiveFrictionU == 0.0f && cone.effectiveFrictionV == 0.0f) {
        effective[0][1] = effective[0][2] = 0.0f;
        effective[1][0] = effective[2][0] = 0.0f;
    }
}

'''
needle='inline bool invert3x3(\n'
assert s.count(needle)==1;s=s.replace(needle,helper+needle)
# Only the two Wave32 block constructions and the rigid ordered fallback.
pattern=r"(?m)^( +)float inverse\[3\]\[3\];\n\1if \(!invert3x3\(effective, inverse\)\) \{"
matches=list(re.finditer(pattern,s));assert len(matches)==4,len(matches)
for match in reversed(matches[:3]):
 indent=match.group(1)
 s=s[:match.start()]+indent+'frictionlessNormalBlock(effective, evaluatedCones[constraintBase + localConstraint]);\n'+s[match.start():]
source=p/'candidate.metal';source.write_text(s)
cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(root/'include'),'-c',str(source),'-o',str(p/'candidate.air')]
logs=[]
def run(cmd):
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);logs.append({'command':cmd,'exit':r.returncode,'output':r.stdout});(p/'build.json').write_text(json.dumps(logs,indent=2));print('exit',r.returncode,flush=True);print(r.stdout,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
run(cmd)
commands=(p/'build-commands.txt').read_text()
segments=commands.replace('\n',' && ').split(' && ')
links=[shlex.split(x) for x in segments if 'metallib ' in x and 'MetalRobo.metallib' in x]
assert len(links)==1,len(links)
cmd=links[0];cmd=[str(p/'candidate.air') if x.endswith('/MetalRoboWorldContact.air') else str(p/'candidate.metallib') if x.endswith('/MetalRobo.metallib') else x for x in cmd]
assert str(p/'candidate.air') in cmd and str(p/'candidate.metallib') in cmd
run(cmd)
