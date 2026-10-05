from pathlib import Path
import json,subprocess,hashlib
E=Path('/Users/n/numi-human-resting-evidence-20261005')
old='/Users/n/numi-human-resting-projected-equality-';new='/Users/n/numi-human-resting-limit-lanes-'
prior=json.loads((E/'projected-equality-build-022-invocation.json').read_text())
commands=[[x.replace(old+'source-022',new+'source-023').replace(old+'build-022',new+'build-023') for x in row] for row in prior['commands'][:2]]
commands += [['install_name_tool','-rpath',old+'build-022/lib',new+'build-023/lib',new+'build-023/bin/numi-human-native'],['codesign','--force','--sign','-',new+'build-023/bin/numi-human-native']]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Source022 CPU bundle cloned; existing SIMD source-limit solver extended from two to three equality rows per lane. GPU only, same ordered impulse application, no new buffers. CPU code unchanged; executable rpath retargeted.','commands':commands,'source_sha256':{f:sha(Path(new+'source-023')/f) for f in ['src/metal/NumiHumanStand.metal','src/metal/NumiHumanStandSolve.metalinc']}}
(E/'limit-lanes-build-023-invocation.json').write_text(json.dumps(report,indent=2)+'\n')
with (E/'limit-lanes-build-023.log').open('wb') as log:
 for argv in commands:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
report['binary_sha256']=sha(new+'build-023/bin/numi-human-native');report['metallib_sha256']=sha(new+'build-023/shaders/MetalRobo.metallib')
(E/'limit-lanes-build-023-result.json').write_text(json.dumps(report,indent=2)+'\n')
for inp,out in [('run-resting-projected-equality-profile-042.py','run-resting-limit-lanes-profile-045.py'),('run-resting-projected-equality-043.py','run-resting-limit-lanes-046.py')]:
 s=(E/inp).read_text().replace(old+'source-022',new+'source-023').replace(old+'build-022',new+'build-023')
 p=E/out;assert not p.exists();p.write_text(s)
print('build succeeded')

