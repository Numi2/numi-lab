from pathlib import Path
import json, subprocess, hashlib
E=Path('/Users/n/numi-human-resting-evidence-20261005')
old='/Users/n/numi-human-resting-equality-finish-';new='/Users/n/numi-human-resting-projected-equality-'
prior=json.loads((E/'equality-finish-build-020-invocation.json').read_text())
commands=[[x.replace(old+'source-020',new+'source-022').replace(old+'build-020',new+'build-022') for x in row] for row in prior['commands'][:2]]
commands += [['install_name_tool','-rpath',old+'build-020/lib',new+'build-022/lib',new+'build-022/bin/numi-human-native'],['codesign','--force','--sign','-',new+'build-022/bin/numi-human-native']]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Source020 CPU bundle cloned; existing projected-contact GPU solve extended beyond its obsolete factor-cache size gate compiled and linked with original CMake commands. CPU code unchanged; executable rpath retargeted.','commands':commands,'source_sha256':{f:sha(Path(new+'source-022')/f) for f in ['src/metal/NumiHumanStand.metal','src/metal/NumiHumanStandSolve.metalinc']}}
(E/'projected-equality-build-022-invocation.json').write_text(json.dumps(report,indent=2)+'\n')
with (E/'projected-equality-build-022.log').open('wb') as log:
 for argv in commands:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
report['binary_sha256']=sha(new+'build-022/bin/numi-human-native');report['metallib_sha256']=sha(new+'build-022/shaders/MetalRobo.metallib')
(E/'projected-equality-build-022-result.json').write_text(json.dumps(report,indent=2)+'\n')
for inp,out in [('run-resting-equality-finish-profile-036.py','run-resting-projected-equality-profile-042.py'),('run-resting-equality-finish-037.py','run-resting-projected-equality-043.py')]:
 s=(E/inp).read_text().replace(old+'source-020',new+'source-022').replace(old+'build-020',new+'build-022')
 p=E/out;assert not p.exists();p.write_text(s)
print('build succeeded')

