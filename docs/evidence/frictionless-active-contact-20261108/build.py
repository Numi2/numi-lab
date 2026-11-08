from pathlib import Path
import json,re,shlex,subprocess
root=Path('/Users/numi/numi-lab-reconciliation-20261108')
evidence=root/'docs/evidence/frictionless-active-contact-20261108'
out=Path('/Users/numi/ccd-active-contact-20261108')
out.mkdir(exist_ok=True)
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
assert s.count(needle)==1
s=s.replace(needle,helper+needle)
pattern=r'(?m)^( +)float inverse\[3\]\[3\];\n\1if \(!invert3x3\(effective, inverse\)\) \{'
matches=list(re.finditer(pattern,s)); assert len(matches)==4,len(matches)
for m in reversed(matches[:3]):
 s=s[:m.start()]+m.group(1)+'frictionlessNormalBlock(effective, evaluatedCones[constraintBase + localConstraint]);\n'+s[m.start():]
(out/'candidate.metal').write_text(s)
commands=(root/'docs/evidence/frictionless-oblique-contact-20261108/build-commands.txt').read_text()
links=[shlex.split(x) for x in commands.replace('\n',' && ').split(' && ') if 'metallib ' in x and 'MetalRobo.metallib' in x]
assert len(links)==1
link=[str(out/'candidate.air') if x.endswith('/MetalRoboWorldContact.air') else str(out/'candidate.metallib') if x.endswith('/MetalRobo.metallib') else x for x in links[0]]
assert str(out/'candidate.air') in link and str(out/'candidate.metallib') in link
cmds=[['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(root/'include'),'-c',str(out/'candidate.metal'),'-o',str(out/'candidate.air')],link]
logs=[]
for cmd in cmds:
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 logs.append({'command':cmd,'exit':p.returncode,'output':p.stdout})
 (evidence/'build.json').write_text(json.dumps(logs,indent=2)+'\n')
 if p.returncode: raise SystemExit(p.returncode)
