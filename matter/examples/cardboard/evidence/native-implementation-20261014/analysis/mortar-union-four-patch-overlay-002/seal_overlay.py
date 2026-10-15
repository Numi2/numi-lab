from pathlib import Path
import difflib, hashlib, json, subprocess
root=Path('/Users/home/cardboard-implementation-20261014')
pkg=root/'analysis/mortar-union-four-patch-overlay-002'
base=root/'snapshots/source-rationalized-paper-and-native-checkers-091'
work=pkg/'build-source'
rels=['matter/src/metal/contact.metalinc','matter/src/compiler_impl.cpp','matter/tools/cardboard_full_contact_armijo_check.mm']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
patch=[]
for rel in rels:
    a=(base/rel).read_text().splitlines(True)
    b=(work/rel).read_text().splitlines(True)
    patch.extend(difflib.unified_diff(a,b,fromfile='a/'+rel,tofile='b/'+rel,n=3))
patch_path=pkg/'mortar-union-four-patch-092.patch'
patch_path.write_text(''.join(patch))
# Verify the exact patch applies cleanly to the frozen source without writing it.
check=subprocess.run(['patch','--dry-run','-p1','-d',str(base),'-i',str(patch_path)],text=True,capture_output=True)
if check.returncode:
    raise SystemExit('patch dry-run failed:\n'+check.stdout+check.stderr)
source_manifest=base/'frozen-source.json'
manifest={
  'schema':'numi.cardboard.mortar-union-overlay.v1',
  'evidence_class':'external source overlay; compilation and CPU admission only; native GPU tests not run',
  'base_snapshot':str(base),
  'base_snapshot_manifest_sha256':sha(source_manifest),
  'base_parent_revision':json.loads(source_manifest.read_text())['parent_revision'],
  'base_file_sha256':{rel:sha(base/rel) for rel in rels},
  'overlay_file_sha256':{rel:sha(work/rel) for rel in rels},
  'patch_file':patch_path.name,
  'patch_sha256':sha(patch_path),
  'patch_apply_dry_run':{'status':'PASS','command':'patch --dry-run -p1 -d <frozen-source-091> -i mortar-union-four-patch-092.patch','output':check.stdout.strip()},
  'build':{
    'status':'PASS',
    'command':'cmake --build build-overlay --target numi-matter-cardboard-full-contact-armijo-check -j2',
    'workdir':str(work/'build-overlay'),
    'executable':str(work/'build-overlay/matter/numi-matter-cardboard-full-contact-armijo-check'),
    'executable_sha256':sha(work/'build-overlay/matter/numi-matter-cardboard-full-contact-armijo-check'),
    'metallib':str(work/'build-overlay/matter/shaders/NumiMatter.metallib'),
    'metallib_sha256':sha(work/'build-overlay/matter/shaders/NumiMatter.metallib'),
    'gpu_executed':False,
  },
  'cpu_tests':{
    'source_admission':{'status':'PASS','command':'build-overlay/matter/numi-matter-cardboard-full-contact-armijo-check matter/materials/hajali2009_liner_hill_asymmetric_c1_transfer.nmatter --cpu-only','native_fem_transactions':0,'observed_output':(pkg/'cpu-only-output.txt').read_text().strip()},
    'paired_boundary_predicate':{'status':'PASS','command':'/Users/home/.pyenv/versions/3.10.12/bin/python3 cpu_boundary_contract_check.py','script_sha256':sha(pkg/'cpu_boundary_contract_check.py'),'result_sha256':sha(pkg/'cpu_boundary_contract_check.json')},
    'shipping_hma_parent_pair_multiplicity':{'status':'PASS','command':'xcrun --sdk macosx clang++ -std=c++20 -O2 -I <source091>/matter/tools -I <source091>/matter/include -I <source091>/include hma_parent_pair_multiplicity.cpp -o hma_parent_pair_multiplicity && ./hma_parent_pair_multiplicity','source_sha256':sha(pkg/'hma_parent_pair_multiplicity.cpp'),'binary_sha256':sha(pkg/'hma_parent_pair_multiplicity'),'stdout_sha256':sha(pkg/'hma_parent_pair_multiplicity.stdout.txt'),'report_sha256':sha(pkg/'shipping_hma_parent_pair_multiplicity.json'),'stdout':(pkg/'hma_parent_pair_multiplicity.stdout.txt').read_text().strip()},
  },
  'limitations':['Native private-sweep, four-patch CCD, shared-edge sliding, full shipping execution and GPU behavior remain unrun by this worker.','Native source run 301 reported an aggregate failure and did not capture the first pair/witness; its exact triggering pair remains a CPU reconstruction.','The shipping multiplicity count is for the exact default 300x200x200 mm design, literature2009 geometry, nxPerPitch=8, thicknessSlices=1, 16 subdivisions per band, zero engineering allowance, and current source091 builder. It is not a proof for other mesh resolutions, score coupons, or altered HMA footprints.','Four-patch admission is a bounded tested ceiling; capacity greater than four remains explicitly unsupported.']
}
(pkg/'cpu-only-output.txt').write_text(manifest['cpu_tests']['source_admission']['observed_output']+'\n')
# Hash after creating the transcript; ensure manifest records it below.
manifest['cpu_tests']['source_admission']['stdout_sha256']=sha(pkg/'cpu-only-output.txt')
manifest_path=pkg/'manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'patch':str(patch_path),'patch_sha256':sha(patch_path),'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'dry_run':check.stdout.strip()},indent=2))
