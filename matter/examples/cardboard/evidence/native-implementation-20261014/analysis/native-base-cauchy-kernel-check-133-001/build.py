from pathlib import Path
import json,hashlib,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-base-state-cauchy-armijo-133';D=B/'analysis/native-base-cauchy-kernel-check-133-001';build=R/'build-cardboard-provenance-133';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
j=json.loads((build/'source-build.json').read_text());assert j['success']
for p,h in j['artifacts_sha256'].items():assert sha(Path(p))==h,p
source=D/'native_cauchy_kernel_check.mm';output=D/'native_cauchy_kernel_check';cmd=['xcrun','clang++','-std=c++2b','-arch','arm64','-fobjc-arc','-Wall','-Wextra','-Wpedantic','-Werror','-I'+str(S/'include'),'-I'+str(S/'matter/include'),'-DNUMI_MATTER_METALLIB="'+str(build/'matter/shaders/NumiMatter.metallib')+'"',str(source),str(build/'matter/libnumi_matter_runtime.a'),str(build/'matter/libnumi_matter_compiler.a'),'-framework','Foundation','-framework','Metal','-o',str(output)]
with (D/'compile.stdout.log').open('x') as out,(D/'compile.stderr.log').open('x') as err:q=subprocess.run(cmd,stdout=out,stderr=err)
report=dict(schema='numi.cardboard.cauchy_native_checker_build.v1',command=cmd,exit_code=q.returncode,source133_build_sha256=sha(build/'source-build.json'),source_sha256=sha(source),transitive_helper_source_sha256=sha(B/'snapshots/source-boundary-reset-order-101/matter/tools/cardboard_free_box_angular_native_check.mm'),source133_artifacts=j['artifacts_sha256'],physical_transactions=0)
if q.returncode==0:report['binary_sha256']=sha(output)
(D/'compiled-program.json').write_text(json.dumps(report,indent=2)+'\n');(D/'build.py').write_bytes(Path(__file__).read_bytes());print('checker133build',q.returncode);raise SystemExit(q.returncode)
