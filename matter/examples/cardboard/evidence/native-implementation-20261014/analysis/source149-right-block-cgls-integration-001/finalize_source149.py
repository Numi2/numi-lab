#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os
from pathlib import Path

B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
BASE=B/'snapshots/source-cgls-post-release-relaxation-145'
SRC=B/'snapshots/source-cgls-block-right-preconditioned-149'
OUT=B/'analysis/source149-right-block-cgls-integration-001'
OVER=B/'analysis/native-cgls-block-right-preconditioning-overlay-149-002'
BUILD=R/'build-cardboard-provenance-149'
CHANGED=['matter/src/runtime.mm','matter/src/metal/fgmres.metalinc','matter/src/metal/base_cgls.metalinc']
CRITICAL_UNCHANGED=['matter/src/metal/fem.metalinc','matter/src/metal/base_cgls_runtime.metalinc','matter/src/metal/common.metalinc']
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def inv(root:Path):
 out={}
 for cur,dirs,files in os.walk(root,followlinks=False):
  dirs.sort(); files.sort(); cp=Path(cur)
  for name in files:
   p=cp/name; rel=str(p.relative_to(root))
   if p.is_symlink(): out[rel]={'kind':'symlink','target':os.readlink(p)}
   elif p.is_file(): out[rel]={'kind':'file','bytes':p.stat().st_size,'sha256':sha(p)}
 return out

def main():
 frozen=json.loads((SRC/'frozen-source.json').read_text())
 support=json.loads((SRC/'configure-support-source.json').read_text())
 base_frozen=json.loads((BASE/'frozen-source.json').read_text())
 base_support=json.loads((BASE/'configure-support-source.json').read_text())
 build=json.loads((BUILD/'source-build.json').read_text())
 assert build['success'] is True
 assert build['source_snapshot']==str(SRC)
 assert build['targets']==['numi-matter-cardboard-probe']
 assert build['changed_sources']==[]
 assert build['source_sha256']==frozen['source_sha256']
 assert len(frozen['source_sha256'])==787
 assert support==base_support and len(support['files'])==1034
 for rel,d in frozen['source_sha256'].items(): assert sha(SRC/rel)==d,rel
 for rel,d in support['files'].items(): assert sha(SRC/rel)==d,rel
 baseinv=inv(BASE); srcinv=inv(SRC)
 assert len(baseinv)==1859 and len(srcinv)==1859 and baseinv.keys()==srcinv.keys()
 allowed=set(CHANGED)|{'frozen-source.json'}
 delta={p for p in baseinv if baseinv[p]!=srcinv[p]}
 assert delta==allowed, sorted(delta^allowed)
 critical={rel:{'source145_sha256':sha(BASE/rel),'source149_sha256':sha(SRC/rel),'byte_identical':(BASE/rel).read_bytes()==(SRC/rel).read_bytes()} for rel in CRITICAL_UNCHANGED}
 assert all(v['byte_identical'] for v in critical.values())
 overlay=json.loads((OVER/'source-overlay.json').read_text())
 assert overlay['overlay_id']=='native-cgls-block-right-preconditioning-149-002'
 assert overlay['base_snapshot']==str(BASE)
 assert overlay['overlay_source_hashes']=={rel:sha(SRC/rel) for rel in CHANGED+['matter/src/metal/base_cgls_runtime.metalinc']}
 assert sha(OVER/'source-overlay.patch')==overlay['patch_sha256']
 # Hash only compiled outputs and the exact logs; no executable is invoked here.
 artifacts={}
 for path,digest in build['artifacts_sha256'].items():
  p=Path(path); assert p.is_file() and sha(p)==digest
  artifacts[str(p)]={'bytes':p.stat().st_size,'sha256':digest}
 logs=[]
 for p in sorted(BUILD.glob('provenance-command-*.log')):
  logs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 buildpath=BUILD/'source-build.json'
 post={
  'schema':'numi.cardboard.source149_right_block_cgls_build_evidence.v1',
  'source_snapshot':str(SRC),'source_snapshot_frozen_sha256':sha(SRC/'frozen-source.json'),
  'source_hash_entries_verified':len(frozen['source_sha256']),'configure_support_entries_verified':len(support['files']),
  'configure_support_manifest_sha256':sha(SRC/'configure-support-source.json'),
  'source145_inventory_entries':len(baseinv),'source149_inventory_entries':len(srcinv),
  'expected_delta_paths':sorted(delta),'unchanged_file_count':len(srcinv)-len(delta),
  'source_build_receipt':{'path':str(buildpath),'bytes':buildpath.stat().st_size,'sha256':sha(buildpath),'success':True,'targets':build['targets'],'changed_sources':[]},
  'build_command_logs':logs,'build_outputs':artifacts,
  'build_command':json.loads((OUT/'build-command.json').read_text()),
  'disk_available_bytes_after_build':os.statvfs(B).f_bavail*os.statvfs(B).f_frsize,
  'native_probe_or_gpu_simulation':'not launched'
 }
 (OUT/'post-build.json').write_text(json.dumps(post,indent=2,sort_keys=True)+'\n')
 scope={
  'schema':'numi.cardboard.source149_scope_review.v1',
  'base':'source145 post-release snapshot','result':'source149 block-right-CGLS overlay',
  'changed_source_paths':CHANGED,
  'diff_review':{
   'runtime.mm':'Adds default-off selector/scratch/capture metadata and dispatches the existing production tangent block builder into private femBaseCGLSRightScalePhysical scratch. The CGLS sequence applies S before the existing A action, S^T after A^T, and S before the final existing A*x/install. It does not edit Newton acceptance thresholds or Armijo evaluation kernels.',
   'fgmres.metalinc':'Changes only the existing nodal tangent block preconditioner builder ABI and selected-environment initialization/skip behavior. No element operator kernel is modified.',
   'base_cgls.metalinc':'Adds right-scale validation and vector transform kernels and the initial S^T(A^T b) transform. Existing CGLS recurrence and install kernel are unchanged.'
  },
  'critical_unchanged_files':critical,
  'operator_and_acceptance_review':{
   'physical_matrix_free_operator_file':'matter/src/metal/fem.metalinc',
   'physical_operator_file_unchanged':critical['matter/src/metal/fem.metalinc']['byte_identical'],
   'final_direction_installer_file':'matter/src/metal/base_cgls_runtime.metalinc',
   'installer_and_certificate_file_unchanged':critical['matter/src/metal/base_cgls_runtime.metalinc']['byte_identical'],
   'remaining_acceptance_common_file_unchanged':critical['matter/src/metal/common.metalinc']['byte_identical'],
   'whole_delta':'Only the three listed source files plus frozen-source.json differ from source145; no changes were made to FEM material law, residual, Armijo trial, acceptance gate, or final certificate implementation.',
   'scratch_erratum':'149-001 wrote inverse diagonal builder output to femPreconditioned; 149-002 binds private femBaseCGLSRightScalePhysical instead. The private U-vector is completely overwritten by S*p before next consumption.'
  },
  'validation_limits':['CPU reference/static tests pass 14/14.','Source-bound probe target compiled successfully with four jobs; no probe executable was run.','No native GPU parity, full history parity, or full coupon result is included. Root-reported prior alpha=1 evidence was mixed; no performance/convergence benefit is claimed.']
 }
 (OUT/'scope-review.json').write_text(json.dumps(scope,indent=2,sort_keys=True)+'\n')
 integration=json.loads((OUT/'integration.json').read_text())
 integration.update({'build_status':'passed','source_build_receipt_path':str(buildpath),'source_build_receipt_sha256':sha(buildpath),'build_outputs':artifacts,'build_command_logs':logs,'changed_sources':[],'post_build_evidence_path':str(OUT/'post-build.json'),'post_build_evidence_sha256':sha(OUT/'post-build.json'),'scope_review_path':str(OUT/'scope-review.json'),'scope_review_sha256':sha(OUT/'scope-review.json'),'disk_available_bytes_after_build':post['disk_available_bytes_after_build'],'native_probe_or_gpu_launch':'not launched'})
 (OUT/'integration.json').write_text(json.dumps(integration,indent=2,sort_keys=True)+'\n')
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='source149-integration-manifest.json')
 manifest={'schema':'numi.cardboard.source149_integration_manifest.v1','files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}}
 (OUT/'source149-integration-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'frozen_source_sha256':sha(SRC/'frozen-source.json'),'build_receipt_sha256':sha(buildpath),'post_build_sha256':sha(OUT/'post-build.json'),'scope_review_sha256':sha(OUT/'scope-review.json'),'integration_sha256':sha(OUT/'integration.json'),'manifest_sha256':sha(OUT/'source149-integration-manifest.json'),'source_entries':len(frozen['source_sha256']),'configure_entries':len(support['files']),'inventory':len(srcinv),'unchanged':len(srcinv)-len(delta),'build_success':build['success'],'target':build['targets'][0],'artifact_count':len(artifacts),'disk_bytes_after_build':post['disk_available_bytes_after_build']},indent=2))
if __name__=='__main__': main()
