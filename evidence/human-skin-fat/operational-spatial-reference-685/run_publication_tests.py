from pathlib import Path
import subprocess,time,json,hashlib,os
d=Path(__file__).resolve().parent
w=Path('/Users/numi/Work/skin-fat-native-integration-209')
out=d/'publication-tests';out.mkdir(exist_ok=True)
results=[]
for mode in ['assert','sanitize']:
 binary=out/('spatial-integration-'+mode)
 cmd=['/usr/bin/clang++','-std=c++23','-O1','-g0','-UNDEBUG','-Wno-deprecated-declarations',
 '-I'+str(w/'apps'),'-I'+str(w/'include'),'-I'+str(w/'matter/include'),'-ivfsoverlay',str(d/'publication-overlay.json'),
 str(d/'publication-source/apps/NumiHumanPhysicalSkinChildRegisteredVolumeIntegrationCheck.cpp'),'-o',str(binary)]
 if mode=='sanitize':cmd[1:1]=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
 t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True)
 (out/(mode+'-compile.stdout')).write_text(r.stdout);(out/(mode+'-compile.stderr')).write_text(r.stderr)
 result={'mode':mode,'compile_argv':cmd,'compile_exit':r.returncode,'compile_seconds':time.monotonic()-t}
 if r.returncode==0:
  env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=0'
  t=time.monotonic();r=subprocess.run([str(binary)],capture_output=True,text=True,env=env)
  (out/(mode+'-run.stdout')).write_text(r.stdout);(out/(mode+'-run.stderr')).write_text(r.stderr)
  result.update(run_exit=r.returncode,run_seconds=time.monotonic()-t,stdout=r.stdout,stderr=r.stderr,
   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
 else:result['stderr']=r.stderr
 results.append(result)
 (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(result),flush=True)
 if r.returncode:break
