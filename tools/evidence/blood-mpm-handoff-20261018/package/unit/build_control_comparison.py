#!/usr/bin/env python3
import hashlib, json, subprocess
from pathlib import Path
d=Path('/Users/n/numi-human-separation-20261018/C/blood-transfer-208/transaction-fixture')
src=d/'blood_mpm_handoff_transaction.mm'
obj=d/'blood_mpm_handoff_transaction.o'
exe=d/'blood_mpm_handoff_transaction'
matter='/Users/n/numi-human-separation-20261018/C/blood-transfer-208'
full='/Users/n/numi-human-separation-20261018/C/current-operator-integration-207/full-source'
lib=Path(matter+'/build/libmetalrobo.dylib')
msl=Path(matter+'/shader-preflight/HumanRespiration208.metal')
material='/Users/n/numi-human-separation-20261018/C/bleeding-liquid-205/main-owner-206/matter/materials/blood_high_shear_newtonian_proxy.nmatter'
matter_source='/Users/n/numi-human-separation-20261018/C/aortic-vascular-control-020/source/MatterSource.metal'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
expected_lib='bd1d6d273c06a5d4e00729d0a436ee341827ee67def28a0107fef6a5713f7673'
if sha(lib)!=expected_lib: raise SystemExit('runtime library SHA changed; pin and review before build')
cmd=['/usr/bin/clang++','-x','objective-c++','-std=c++23','-O3','-DNDEBUG','-arch','arm64','-fobjc-arc','-Wall','-Wextra','-Wpedantic','-I'+matter+'/matter/include','-I'+matter+'/source/matter/include','-I'+full+'/matter/include','-I'+full+'/include','-I'+full+'/src','-DNUMI_MATTER_MATERIAL="'+material+'"','-DNUMI_MATTER_METALLIB="'+matter_source+'"','-DNUMI_BLOOD_MPM_HANDOFF_METAL="'+str(msl)+'"',str(src),'-c','-o',str(obj)]
cp=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(d/'compile-control-comparison-002.log').write_text(cp.stdout+cp.stderr)
if cp.returncode: raise SystemExit(cp.returncode)
link=['/usr/bin/clang++','-arch','arm64',str(obj),'-L'+str(lib.parent),'-lmetalrobo','-Wl,-rpath,'+str(lib.parent),'-framework','Foundation','-framework','Metal','-framework','QuartzCore','-framework','Accelerate','-framework','CoreGraphics','-framework','CoreImage','-framework','ImageIO','-framework','MetalKit','-framework','ModelIO','-lsqlite3','-lxml2','-o',str(exe)]
lp=subprocess.run(link,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(d/'link-control-comparison-002.log').write_text(lp.stdout+lp.stderr)
if lp.returncode: raise SystemExit(lp.returncode)
r={'schema':'numi.blood_mpm_transaction_fixture_build.v1','source':str(src),'source_sha256':sha(src),'source_handoff_msl':matter+'/source/matter/src/human_respiration.metal','source_handoff_msl_sha256':sha(matter+'/source/matter/src/human_respiration.metal'),'flattened_handoff_msl':str(msl),'flattened_handoff_msl_sha256':sha(msl),'matter_source':matter_source,'matter_source_sha256':sha(matter_source),'runtime_library':str(lib),'runtime_library_sha256':sha(lib),'compile_argv':cmd,'link_argv':link,'binary':str(exe),'binary_sha256':sha(exe),'compile_exit':cp.returncode,'link_exit':lp.returncode}
(d/'build-control-comparison-005.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ('compile_exit','link_exit','source_sha256','source_handoff_msl_sha256','flattened_handoff_msl_sha256','matter_source_sha256','runtime_library_sha256','binary_sha256')},indent=2))
