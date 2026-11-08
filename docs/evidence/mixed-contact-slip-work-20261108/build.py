from pathlib import Path
import hashlib,json,subprocess
p=Path('/Users/numi/frictionless-mixed-rod-20261108');dest=Path.cwd()
old=json.loads((p/'split-build.json').read_text());commands=[]
for item in old:
 args=[x.replace(str(p/'split.metal'),str(dest/'slip-trace.metal')).replace(str(p/'split.air'),str(dest/'slip-trace.air')).replace(str(p/'split.metallib'),str(dest/'slip-trace.metallib')) for x in item['command']]
 commands.append(args)
commands.append(['xcrun','clang++','-std=c++20','-O2','-I','/Users/numi/numi-lab-reconciliation-20261108/include',str(dest/'probe.cpp'),'-L','/Users/numi/numi-reconciliation-build-20261108/lib','-lmetalrobo','-Wl,-rpath,/Users/numi/numi-reconciliation-build-20261108/lib','-o',str(dest/'probe')])
records=[]
for args in commands:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 records.append({'command':args,'exit':r.returncode,'output':r.stdout});(dest/'build.json').write_text(json.dumps(records,indent=2)+'\n')
 print('exit',r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
files=[Path(x) for x in commands[1] if x.endswith('.air')]+[Path('/Users/numi/frictionless-mixed-rod-20261108/split.metallib'),Path('/Users/numi/numi-reconciliation-build-20261108/lib/libmetalrobo.dylib'),dest/'slip-trace.metallib',dest/'probe']
(dest/'binaries.sha256').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+str(x)+'\n' for x in files))
