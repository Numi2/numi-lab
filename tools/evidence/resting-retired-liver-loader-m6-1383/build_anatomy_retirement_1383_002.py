#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,os,subprocess,time
S=Path('/Users/n/numi-lab-anatomy-retirement-1383')
B=Path('/Users/n/numi-lab-anatomy-retirement-build-1383-002')
OLD=Path('/Users/n/numi-lab-parent-bindings-build-1353-003')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not B.exists()
B.mkdir();(B/'bin').mkdir()
for folder in ('lib','shaders','matter/shaders'):
    target=B/folder;target.mkdir(parents=True,exist_ok=True)
    for p in (OLD/folder).iterdir():
        if p.is_file():os.link(p,target/p.name)
tracked=subprocess.check_output(['git','-C',str(S),'ls-files'],text=True).splitlines()
before={str(S/p):sha(S/p) for p in tracked if (S/p).is_file()}
before[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
old=json.loads((OLD/'build-report.json').read_text())
commands=[]
for rec in old['commands']:
    argv=[x.replace('/Users/n/numi-lab-parent-bindings-1353',str(S)).replace(str(OLD),str(B)) for x in rec['argv']]
    if rec['phase']=='compile':argv+=['-MMD','-MF',str(B/'native-dependencies.d')]
    start=time.monotonic()
    with (B/(rec['phase']+'.log')).open('x') as log:
        result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
    commands.append(dict(phase=rec['phase'],argv=argv,returncode=result.returncode,wall_seconds=time.monotonic()-start))
    if result.returncode:break
after={p:sha(Path(p)) for p in before}
changed={p:dict(before=h,after=after[p]) for p,h in before.items() if after[p]!=h}
report=dict(source=str(S),source_head=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip(),
    source_patch=subprocess.check_output(['git','-C',str(S),'diff'],text=True),
    commands=commands,input_sha256=before,changed_inputs=changed,
    reused_native_runtime_sha256={str(p):sha(p) for p in B.rglob('*') if p.is_file() and p.suffix in ('.dylib','.metallib')},
    native_execution=False)
if all(r['returncode']==0 for r in commands):report['binary_sha256']=sha(B/'bin/numi-human-native')
(B/'build-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'build':str(B),'commands':commands,'changed_inputs':changed,'binary_sha256':report.get('binary_sha256')}))
assert len(commands)==2 and all(r['returncode']==0 for r in commands) and not changed
