from pathlib import Path
import hashlib,json,sys
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
base=Q/'snapshots/source-strict-contact-sample-pages-247-001'
snap=Q/'snapshots/source-contact-compaction-fixture-249-001'
pkg=Q/'analysis/source249-contact-compaction-fixture-001'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def obj(p): return json.loads(Path(p).read_text())
basef=obj(base/'frozen-source.json')
frozen=obj(snap/'frozen-source.json')
changed=[]
for rel,h in basef['source_sha256'].items():
    actual=sha(snap/rel)
    if actual!=h: changed.append((rel,h,actual))
assert len(basef['source_sha256'])==842
assert len(frozen['source_sha256'])==842
assert changed==[('matter/tools/cardboard_contact_sample_compaction_check.mm',
                  basef['source_sha256']['matter/tools/cardboard_contact_sample_compaction_check.mm'],
                  frozen['source_sha256']['matter/tools/cardboard_contact_sample_compaction_check.mm'])]
for rel,h in frozen['source_sha256'].items(): assert sha(snap/rel)==h,rel
build=pkg/'build-001'
br=obj(build/'source-build.json')
assert br['success'] is True and br['changed_sources']==[]
assert br['source_snapshot']==str(snap)
assert len(br['source_sha256'])==837
for rel,h in br['source_sha256'].items(): assert frozen['source_sha256'][rel]==h,rel
artifacts={}
for path,h in br['artifacts_sha256'].items():
    assert sha(path)==h,path
    artifacts[path]=h
cpu=obj(pkg/'cpu-validation.json')
assert cpu['all_exit_zero'] is True and cpu['native_gpu_execution'] is False
assert len(cpu['cases'])==5 and all(c['exit_code']==0 for c in cpu['cases'])
# Bind and preserve the exact prior native failure and its inputs.
case=Q/'native/production-contact-compactor-592'
receipt=obj(case/'receipt.json')
assert receipt['execution_started'] is True and receipt['exit_code']==1 and receipt['timed_out'] is False
err=(case/'stderr.log').read_text()
assert err=='contact sample compaction checker failed: compact FEM force gather changed canonical accumulation\n'
pred=obj(case/'prediction.json')
argv=pred['argv']
oldchecker=Path(argv[0]); oldlib=Path(argv[1])
oldbuild=obj(pred['build_provenance'])
source235=Q/'snapshots/source-shipping-contact-sample-pages-235-007'
source235f=obj(source235/'frozen-source.json')
source235build=Q/'analysis/source235-sample-compactor-007/build/source-build.json'
assert source235f['source_sha256']['matter/tools/cardboard_contact_sample_compaction_check.mm']==sha(source235/'matter/tools/cardboard_contact_sample_compaction_check.mm')
assert sha(source235build)==sha(pred['build_provenance'])
old_build_record=obj(source235build)
assert old_build_record['source_sha256']['matter/tools/cardboard_contact_sample_compaction_check.mm']==sha(source235/'matter/tools/cardboard_contact_sample_compaction_check.mm')
assert old_build_record['artifacts_sha256'][str(oldchecker)]==sha(oldchecker)
assert old_build_record['artifacts_sha256'][str(oldlib)]==sha(oldlib)
assert sha(oldchecker)=='ccc34fc4bccce5e71157f845558718fbf799cbeb89da5f835d68e7a04373f53a'
assert sha(oldlib)=='6ea7f46b4054ecb6497aeff1b9f3e9f4ed80c31331361bad0111362600ea5c28'
report={
 'schema':'numi.cardboard.source249.independent-audit.v1',
 'source249_frozen_sha256':sha(snap/'frozen-source.json'),
 'source249_selected_source_count':len(frozen['source_sha256']),
 'only_source_delta_from_247':{'path':changed[0][0],'parent_sha256':changed[0][1],'current_sha256':changed[0][2]},
 'fresh_build_record_sha256':sha(build/'source-build.json'),
 'fresh_build_success':br['success'],
 'fresh_build_source_count':len(br['source_sha256']),
 'fresh_build_artifacts':artifacts,
 'cpu_validation_sha256':sha(pkg/'cpu-validation.json'),
 'cpu_tests_passed':len(cpu['cases']),
 'native592':{
   'receipt_sha256':sha(case/'receipt.json'),
   'stderr_sha256':sha(case/'stderr.log'),
   'source_build_sha256':sha(pred['build_provenance']),
   'checker_sha256':sha(oldchecker),
   'metallib_sha256':sha(oldlib),
   'exit_code':receipt['exit_code'],
   'error':err.strip()
 },
 'classification':'The failure is explained by a fixture mismatch: before the kernel comparison, the ordinary compact FEM residual starts as (0,1,2,0) per node while the dense residual starts at (0,0,0,0). Source249 initializes the normal compact residual to zero and leaves the overflow sentinel (17,18,19,0) unchanged.',
 'native_gpu_execution':False,
 'full_shipping_qualification':False
}
print(json.dumps(report,indent=2,sort_keys=True))
