from pathlib import Path
import json,subprocess,hashlib
E=Path('/Users/n/numi-human-resting-evidence-20261005');old='/Users/n/numi-human-resting-equality-cooperative-';new='/Users/n/numi-human-resting-equality-finish-'
report=json.loads((E/'equality-cooperative-build-019-invocation.json').read_text())
commands=[[x.replace(old+'source-019',new+'source-020').replace(old+'build-019',new+'build-020') for x in row] for row in report['commands'][:2]]
commands += [['install_name_tool','-rpath',old+'build-019/lib',new+'build-020/lib',new+'build-020/bin/numi-human-native'],['codesign','--force','--sign','-',new+'build-020/bin/numi-human-native']]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Source019 bundle cloned; changed GPU triangular solve compiled and linked using existing CMake arguments. CPU machine code unchanged; executable rpath retargeted to cloned library.','commands':commands,'source_sha256':{f:sha(Path(new+'source-020')/f) for f in ['src/metal/NumiHumanStand.metal','src/metal/NumiHumanStandSolve.metalinc']}}
(E/'equality-finish-build-020-invocation.json').write_text(json.dumps(report,indent=2)+'\n')
with (E/'equality-finish-build-020.log').open('wb') as log:
 for argv in commands:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
report['binary_sha256']=sha(new+'build-020/bin/numi-human-native');report['metallib_sha256']=sha(new+'build-020/shaders/MetalRobo.metallib')
(E/'equality-finish-build-020-result.json').write_text(json.dumps(report,indent=2)+'\n')
for inp,out in [('run-resting-equality-profile-034.py','run-resting-equality-finish-profile-036.py'),('run-resting-equality-035.py','run-resting-equality-finish-037.py')]:
 s=(E/inp).read_text().replace(old+'source-019',new+'source-020').replace(old+'build-019',new+'build-020')
 p=E/out;assert not p.exists();p.write_text(s)
print('build succeeded')

