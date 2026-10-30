from pathlib import Path
import subprocess,json,time,hashlib,os
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
W=Path('/Users/numi/Work/skin-fat-native-integration-209'); D=P/'canonical-attachment-native-integration-688'
td=D/'tests'; td.mkdir(exist_ok=True)
src=D/'publication-source/apps/NumiHumanCanonicalDeepAttachmentRemapCheck.cpp'
results=[]
for name,extra in [('assert',[]),('asan-ubsan',['-fsanitize=address,undefined','-fno-omit-frame-pointer'])]:
 binary=td/('canonical-remap-'+name)
 cmd=['/usr/bin/clang++','-std=c++20','-O2','-Wall','-Wextra','-Wpedantic','-Werror','-UNDEBUG','-I'+str(src.parent),str(src),'-o',str(binary)]+extra
 start=time.monotonic(); c=subprocess.run(cmd,capture_output=True,text=True)
 (td/(name+'-compile.log')).write_text(c.stdout+c.stderr)
 row={'name':name,'compiler_argv':cmd,'compile_exit':c.returncode,'compile_seconds':time.monotonic()-start}
 if c.returncode==0:
  env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=0'
  start=time.monotonic();r=subprocess.run([str(binary)],capture_output=True,text=True,env=env)
  (td/(name+'-run.log')).write_text(r.stdout+r.stderr)
  row.update(run_exit=r.returncode,run_seconds=time.monotonic()-start,stdout=r.stdout,stderr=r.stderr,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
 results.append(row)
(td/'test-results.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(r['compile_exit']==0 and r.get('run_exit')==0 for r in results)
print(json.dumps(results,indent=2))
b=D/'native-build-001';b.mkdir(exist_ok=True)
old=P/'operational-spatial-reference-685/publication-native-build-003'
cmd=json.loads((old/'compile-result.json').read_text())['compiler_argv']
cmd=[arg.replace(str(P/'operational-spatial-reference-685/publication-source/apps'),str(D/'publication-source/apps')).replace(str(P/'operational-spatial-reference-685/publication-overlay.json'),str(D/'publication-overlay.json')).replace(str(old/'probe.o'),str(b/'probe.o')) for arg in cmd]
start=time.monotonic()
with (b/'compile.stdout').open('w') as so,(b/'compile.stderr').open('w') as se:c=subprocess.run(cmd,cwd=W/'build-native',stdout=so,stderr=se)
receipt={'compiler_argv':cmd,'exit_code':c.returncode,'seconds':time.monotonic()-start}
if c.returncode==0:receipt['object_sha256']=hashlib.sha256((b/'probe.o').read_bytes()).hexdigest()
(b/'compile-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('NATIVE_COMPILE',json.dumps(receipt))
assert c.returncode==0,(b/'compile.stderr').read_text()[-12000:]
cmd=json.loads((old/'link-result.json').read_text())['argv']
cmd=[arg.replace(str(old),str(b)) for arg in cmd]
start=time.monotonic();r=subprocess.run(cmd,cwd=W/'build-native',capture_output=True,text=True)
(b/'link.log').write_text(r.stdout+r.stderr)
receipt={'argv':cmd,'exit_code':r.returncode,'seconds':time.monotonic()-start,'run_executed':False}
if r.returncode==0:receipt['binary_sha256']=hashlib.sha256((b/'numi-human-native').read_bytes()).hexdigest()
(b/'link-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('NATIVE_LINK',json.dumps(receipt))
assert r.returncode==0
