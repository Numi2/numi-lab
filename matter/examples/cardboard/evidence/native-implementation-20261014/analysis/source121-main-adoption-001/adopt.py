from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');S=B/'snapshots/source-valid-summary-tet858-121';O=B/'analysis/source121-main-adoption-001';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='7ebc64a666c4bd5e1ff61b3684655979ab80dc60'
paths=['matter/CMakeLists.txt','matter/src/metal/common.metalinc','matter/src/runtime.mm','matter/tools/cardboard_asymmetric_c1_paper_gpu_check.mm','matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm','matter/tools/cardboard_fracture_regularization.hpp','matter/tools/cardboard_paper_check.metal','matter/tools/cardboard_retained_matrix_damage.hpp','matter/tools/cardboard_retained_matrix_damage_check.cpp','matter/tools/cardboard_source370_material_point_check.mm','matter/tools/cardboard_source392_tet_point_check.mm']
frozen=json.loads((S/'frozen-source.json').read_text())['source_sha256'];rows=[]
for p in paths:
 assert sha(S/p)==frozen[p]
 q=subprocess.run(['git','diff','--exit-code','--',p],cwd=R,capture_output=True);assert q.returncode==0,(p,q.stdout)
 rows.append(dict(path=p,prior_sha256=sha(R/p) if (R/p).is_file() else None,source121_sha256=frozen[p]))
(O/'preflight.json').write_text(json.dumps(dict(schema='numi.cardboard.reviewed_main_adoption.v1',git_base='7ebc64a666c4bd5e1ff61b3684655979ab80dc60',source121_build_sha256=sha(R/'build-cardboard-provenance-121/source-build.json'),paths=rows,source119_full_build_and_74_cpu_checks=True,source121_native_summary396_strict_payload_and_production_parity=True,physical_coupon392_failed=True,all_physical_thresholds_unchanged=True),indent=2)+'\n')
for row in rows:shutil.copy2(S/row['path'],R/row['path']);assert sha(R/row['path'])==row['source121_sha256']
(O/'adopt.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2)+'\n');print('Adopted11source121files; notcommitted; nativeC1trace safety/replay and publicationverification pending')
