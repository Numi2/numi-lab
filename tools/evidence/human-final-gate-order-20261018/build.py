from pathlib import Path
import json,subprocess,hashlib,shutil
c=Path('/Users/n/numi-human-separation-20261018/C');p=c/'current-operator-integration-207';d=c/'body-final-gate-order-215';b=d/'build';b.mkdir(exist_ok=True)
inc=d/'include/metalrobo';inc.mkdir(parents=True,exist_ok=True);shutil.copyfile(d/'source/MetalArticulatedOperator.hpp',inc/'MetalArticulatedOperator.hpp')
def run(args,name):
 with (b/(name+'.log')).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name+' failed see '+str(b/(name+'.log')))
commands=json.load(open(p/'full-build/compile-receipt.json'))['commands'];a=next(x['argv'] for x in commands if x['source']=='src/metal/MetalArticulatedOperator.mm').copy();a.insert(1,'-I'+str(d/'include'));a[a.index('-MF')+1]=str(b/'operator.o.d');a[a.index('-o')+1]=str(b/'operator.o');a[a.index('-c')+1]=str(d/'source/MetalArticulatedOperator.mm');run(a,'operator-compile')
link=json.load(open(c/'blood-transfer-208/build/build.json'))['link'].copy();old=str(p/'full-build/CMakeFiles/metalrobo.dir/src/metal/MetalArticulatedOperator.mm.o');assert link.count(old)==1;link[link.index(old)]=str(b/'operator.o');link[link.index('-o')+1]=str(b/'libmetalrobo.dylib');run(link,'library-link')
a2=['/usr/bin/clang++','-std=c++23','-O3','-arch','arm64','-fobjc-arc','-x','objective-c++','-I'+str(d/'include'),'-I'+str(p/'full-source/include'),'-I'+str(c/'blood-transfer-208/matter/include'),'-I'+str(p/'full-source/matter/include'),'-DMETALROBO_DEFAULT_METALLIB="/Users/n/numi-lab-complete-body-capture-build-1276-001/shaders/MetalRobo.metallib"','-c',str(d/'source/final_gate_probe.mm'),'-o',str(b/'probe.o')];run(a2,'probe-compile')
a3=['/usr/bin/clang++','-arch','arm64',str(b/'probe.o'),'-L'+str(b),'-lmetalrobo','-framework','Foundation','-framework','Metal','-Wl,-rpath,'+str(b),'-o',str(b/'final-gate-probe')];run(a3,'probe-link')
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r={'compile':a,'link':link,'probe_compile':a2,'probe_link':a3,'source_sha256':h(d/'source/MetalArticulatedOperator.mm'),'header_sha256':h(d/'source/MetalArticulatedOperator.hpp'),'fixture_sha256':h(d/'source/final_gate_probe.mm'),'library_sha256':h(b/'libmetalrobo.dylib'),'binary_sha256':h(b/'final-gate-probe')};(b/'build.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k.endswith('sha256')}))
