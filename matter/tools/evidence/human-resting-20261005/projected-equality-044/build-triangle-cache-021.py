from pathlib import Path
import json, subprocess, hashlib
E=Path('/Users/n/numi-human-resting-evidence-20261005')
old='/Users/n/numi-human-resting-equality-finish-';new='/Users/n/numi-human-resting-triangle-cache-'
prior=json.loads((E/'equality-finish-build-020-invocation.json').read_text())
commands=[[x.replace(old+'source-020',new+'source-021').replace(old+'build-020',new+'build-021') for x in row] for row in prior['commands'][:2]]
commands += [['install_name_tool','-rpath',old+'build-020/lib',new+'build-021/lib',new+'build-021/bin/numi-human-native'],['codesign','--force','--sign','-',new+'build-021/bin/numi-human-native']]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Source020 CPU bundle cloned; existing equality triangle cache GPU optimization compiled and linked with original CMake commands. CPU code unchanged; executable rpath retargeted.','commands':commands,'source_sha256':{f:sha(Path(new+'source-021')/f) for f in ['src/metal/NumiHumanStand.metal','src/metal/NumiHumanStandSolve.metalinc']}}
(E/'triangle-cache-build-021-invocation.json').write_text(json.dumps(report,indent=2)+'\n')
with (E/'triangle-cache-build-021.log').open('wb') as log:
 for argv in commands:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
report['binary_sha256']=sha(new+'build-021/bin/numi-human-native');report['metallib_sha256']=sha(new+'build-021/shaders/MetalRobo.metallib')
(E/'triangle-cache-build-021-result.json').write_text(json.dumps(report,indent=2)+'\n')
for inp,out in [('run-resting-equality-finish-profile-036.py','run-resting-triangle-cache-profile-041.py'),('run-resting-equality-finish-037.py','run-resting-triangle-cache-042.py')]:
 s=(E/inp).read_text().replace(old+'source-020',new+'source-021').replace(old+'build-020',new+'build-021')
 p=E/out;assert not p.exists();p.write_text(s)
print('build succeeded')

