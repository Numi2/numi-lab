from pathlib import Path
import hashlib,json
pkg=Path('/Users/home/cardboard-implementation-20261014/analysis/source106-ccd33-direct-regression-overlay-001')
base=Path('/Users/home/cardboard-implementation-20261014/snapshots/source-capture-callback-lifetime-106')
s103=Path('/Users/home/cardboard-implementation-20261014/snapshots/source-whole-mortar-proof-103')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files=[
 'README.md','source106-ccd33.patch',
 'overlay/matter/src/metal/contact.metalinc',
 'overlay/matter/tools/cardboard_contact_ccd_native_check.mm',
 'evidence/captured-ee-ray.json','tests/check_contract.py',
 'logs/cpu-contract-compile-source.json','logs/patch-dry-run.log','logs/compile-only-build.log',
 'build-matter/shaders/NumiMatter.metallib',
 'build-matter/numi-matter-cardboard-contact-ccd-native-check',
]
manifest={
 'schema':'numi.cardboard.source106_ccd33_direct_regression_overlay.v1',
 'status':'sealed_compile_only_overlay',
 'base_snapshot':str(base),
 'base_frozen_source_json_sha256':sha(base/'frozen-source.json'),
 'base_changed_file_bindings':{
   'matter/src/metal/contact.metalinc':sha(base/'matter/src/metal/contact.metalinc'),
   'matter/tools/cardboard_contact_ccd_native_check.mm':sha(base/'matter/tools/cardboard_contact_ccd_native_check.mm')},
 'source103_function_comparison':{
   'snapshot':str(s103),
   'contact_sha256':sha(s103/'matter/src/metal/contact.metalinc'),
   'production_edge_edge_function_matches_source106_after_bound_normalization':True,
   'baseline_native_result_sha256':json.loads((pkg/'evidence/captured-ee-ray.json').read_text())['native_source103_comparison']['baseline']['result_sha256'],
   'continuation_native_result_sha256':json.loads((pkg/'evidence/captured-ee-ray.json').read_text())['native_source103_comparison']['continuation']['result_sha256']},
 'scope':{
   'production_change':'Only the production edgeEdgeCCD iteration cap changes 17 to 33.',
   'vertex_triangle_iteration_cap_unchanged':16,
   'other_production_physics_changed':False,
   'native_checker_addition':'Exact captured near-band EE case in direct CCD and Newton helper paths.',
   'nominal_crossing_claim':False,
   'full_contact_same_build_run_present':False},
 'validation':{
   'cpu_contract_test':'pass',
   'patch_dry_run':'pass',
   'standalone_matter_compile':'pass',
   'build_target':'numi-matter-cardboard-contact-ccd-native-check',
   'build_exit_code':0,
   'native_executable_invoked':False,
   'gpu_dispatch_performed':False,
   'warnings':['Two pre-existing Metal warnings in fgmres.metalinc about sharedValid and refinementValid threadgroup initialization; no changed file is implicated.'],
   'cmake_configure':'Standalone matter/CMakeLists.txt used; the frozen top-level snapshot omits unrelated Swift binding source needed by root configure.'},
 'commands':{
   'cpu_contract':'python3 tests/check_contract.py --base <source106> --overlay <package>/compile-source --evidence <package>/evidence/captured-ee-ray.json --source103 <source103>/matter/src/metal/contact.metalinc',
   'compile_configure':'cmake -S <package>/compile-source/matter -B <package>/build-matter -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF',
   'compile_build':'cmake --build <package>/build-matter --target numi-matter-cardboard-contact-ccd-native-check --clean-first -j2',
   'native_run':'intentionally not run in this package'},
 'files':{rel:{'sha256':sha(pkg/rel),'bytes':(pkg/rel).stat().st_size} for rel in files},
 'manifest_generator_sha256':sha(pkg/'write_manifest.py'),
 'artifacts':{
   'metallib_sha256':sha(pkg/'build-matter/shaders/NumiMatter.metallib'),
   'checker_executable_sha256':sha(pkg/'build-matter/numi-matter-cardboard-contact-ccd-native-check')},
}
(pkg/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'manifest':str(pkg/'manifest.json'),'manifest_sha256':sha(pkg/'manifest.json'),'patch_sha256':sha(pkg/'source106-ccd33.patch'),'overlay_contact_sha256':sha(pkg/'overlay/matter/src/metal/contact.metalinc'),'overlay_checker_sha256':sha(pkg/'overlay/matter/tools/cardboard_contact_ccd_native_check.mm'),'metallib_sha256':manifest['artifacts']['metallib_sha256'],'checker_sha256':manifest['artifacts']['checker_executable_sha256']},indent=2))
