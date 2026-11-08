import json,subprocess
from pathlib import Path
root=Path.cwd(); build=Path('/Users/numi/numi-reconciliation-build-20261108'); checkout=Path('/Users/numi/numi-lab-reconciliation-20261108')
link=json.loads(Path('../mixed-sustained-contact-20261108/rowtrace-build.json').read_text())[1]['command']
records=[]
def run(cmd):
 p=subprocess.run([str(c) for c in cmd],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 records.append({'command':[str(c) for c in cmd],'returncode':p.returncode,'output':p.stdout});Path('build.json').write_text(json.dumps(records,indent=2)+'\n')
 if p.returncode: raise SystemExit(p.returncode)
run(['xcrun','clang++','-std=c++20','-O2','-fobjc-arc','-I',checkout/'include','probe.mm','-L',build/'lib','-lmetalrobo','-framework','Foundation','-framework','Metal','-o','probe'])
for arm in ('baseline','witness','velocity'):
 run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',checkout/'include','-c',root/(arm+'.metal'),'-o',root/(arm+'.air')])
 cmd=[(str(root/(arm+'.air')) if x.endswith('/candidate-2e-5-rowtrace.air') else str(root/(arm+'.metallib')) if x.endswith('/candidate-2e-5-rowtrace.metallib') else x) for x in link]
 run(cmd)
