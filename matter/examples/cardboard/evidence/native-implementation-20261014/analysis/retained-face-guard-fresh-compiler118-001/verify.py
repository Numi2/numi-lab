from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');P=B/'analysis/shipping-damage-face-ownership-memory-audit-115-002';S=B/'snapshots/source-scaled-material-merit-118';D=R/'build-cardboard-provenance-118-full';O=B/'analysis/retained-face-guard-fresh-compiler118-001';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=json.loads((P/'manifest.json').read_text());assert sha(P/'manifest.json')=='483d94eedc5e36ffa8fffc31548e8a64062de8aa64ee774dff35bf92369c807e'
for row in manifest['artifacts']:assert sha(P/row['path'])==row['sha256'],row['path']
build=json.loads((D/'source-build.json').read_text());assert build['success'] and len(build['targets'])==69
for p,h in build['artifacts_sha256'].items():assert sha(Path(p))==h,p
library=D/'matter/libnumi_matter_compiler.a';h=sha(library)
compile=['/usr/bin/c++','-O3','-DNDEBUG','-std=c++2b','-arch','arm64','-Wall','-Wextra','-Wpedantic','-Werror','-I'+str(P/'src/matter/tools'),'-I'+str(S/'matter/tools'),'-I'+str(S/'matter/include'),'-I'+str(S/'include'),'-c',str(P/'src/matter/tools/cardboard_retained_matrix_damage_check.cpp'),'-o',str(O/'check.o')]
link=['/usr/bin/c++','-O3','-DNDEBUG','-arch','arm64','-Wl,-search_paths_first','-o',str(O/'check'),str(O/'check.o'),str(library)]
run=[str(O/'check'),str(S/'matter/materials')]
for i,cmd in enumerate([compile,link,run]):
 (O/f'command{i}.json').write_text(json.dumps(cmd,indent=2)+'\n')
 with (O/f'stdout{i}.log').open('x') as out,(O/f'stderr{i}.log').open('x') as err:result=subprocess.run(cmd,stdout=out,stderr=err)
 assert result.returncode==0,(i,result.returncode)
assert sha(library)==h
(O/'report.json').write_text(json.dumps(dict(status='pass',source_build_sha256=sha(D/'source-build.json'),source_snapshot=str(S),compiler_archive=str(library),compiler_archive_sha256=h,overlay_manifest_sha256=sha(P/'manifest.json'),checker_sha256=sha(O/'check'),gpu_used=False,full_shipping_mesh_traversal=False,physical_validation=False),indent=2)+'\n');shutil.copy2(__file__,O/'verify.py');print('PASS complete retained CPU guard checker against fresh118compiler')
