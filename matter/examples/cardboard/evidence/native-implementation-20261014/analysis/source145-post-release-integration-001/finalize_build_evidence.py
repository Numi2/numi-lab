#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

B=Path('/Users/home/cardboard-implementation-20261014')
BASE=B/'snapshots/source-cgls-state-sidecar-144'
SRC=B/'snapshots/source-cgls-post-release-relaxation-145'
OUT=B/'analysis/source145-post-release-integration-001'
BUILD=Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-145')
OVERLAY=B/'analysis/cardboard-convergence-tool-overlay-145-001'
CHANGED={
 'matter/tools/cardboard_probe.mm',
 'matter/tools/cardboard_convergence.py',
 'matter/tools/cardboard_resolution_study.py',
 'matter/tools/test_cardboard_convergence.py',
 'matter/tools/test_cardboard_resolution_study.py',
 'frozen-source.json',
}
def sha(p: Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def inv(root:Path):
 out={}
 for cur,dirs,files in os.walk(root,followlinks=False):
  dirs.sort(); files.sort(); cp=Path(cur)
  for n in files:
   p=cp/n; rel=str(p.relative_to(root))
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
 assert support==base_support
 assert len(support['files'])==1034
 for rel,d in frozen['source_sha256'].items():
  assert sha(SRC/rel)==d, rel
 for rel,d in support['files'].items():
  assert sha(SRC/rel)==d, rel
 before=inv(BASE); after=inv(SRC)
 assert before.keys()==after.keys()
 delta={p for p in before if before[p]!=after[p]}
 assert delta==CHANGED, sorted(delta ^ CHANGED)
 assert len(after)==1859
 assert len(after)-len(delta)==1853
 # The five intended edits were based byte-for-byte on the same frozen source in 143 and 144.
 base143=B/'snapshots/source-cgls-primary-policy-143'
 f143=json.loads((base143/'frozen-source.json').read_text())
 for rel in CHANGED-{'frozen-source.json'}:
  assert sha(BASE/rel)==sha(base143/rel)==f143['source_sha256'][rel]
 # Compiler outputs are only hashed, never executed.
 artifacts={}
 for p,d in build['artifacts_sha256'].items():
  q=Path(p); assert q.is_file() and sha(q)==d
  artifacts[str(q)]={'bytes':q.stat().st_size,'sha256':d}
 logs=[]
 for p in sorted(BUILD.glob('provenance-command-*.log')):
  logs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 build_path=BUILD/'source-build.json'
 runner=OUT/'build-runner.log'
 post={
  'schema':'numi.cardboard.source145_post_release_build_evidence.v1',
  'source_snapshot':str(SRC),
  'source_snapshot_frozen_sha256':sha(SRC/'frozen-source.json'),
  'source_map_entries_verified':len(frozen['source_sha256']),
  'configure_support_source_sha256':sha(SRC/'configure-support-source.json'),
  'configure_support_entries_verified':len(support['files']),
  'configure_support_file_byte_identical_to_source144':True,
  'source144_file_inventory_entries':len(before),
  'source145_file_inventory_entries':len(after),
  'only_expected_differences_from_source144':sorted(delta),
  'unchanged_inventory_entries':len(after)-len(delta),
  'overlay_manifest_sha256':sha(OVERLAY/'manifest.json'),
  'overlay_patch_sha256':sha(OVERLAY/'post-release-relaxation.patch'),
  'source_build_receipt':{'path':str(build_path),'bytes':build_path.stat().st_size,'sha256':sha(build_path),'success':build['success'],'targets':build['targets'],'changed_sources':build['changed_sources']},
  'build_runner_log':{'path':str(runner),'bytes':runner.stat().st_size,'sha256':sha(runner),'content':runner.read_text().strip()},
  'build_command_logs':logs,
  'build_outputs':artifacts,
  'native_probe_or_gpu_simulation_launched':False,
  'disk_available_bytes_after_build':os.statvfs(B).f_bavail*os.statvfs(B).f_frsize,
 }
 (OUT/'post-build.json').write_text(json.dumps(post,indent=2,sort_keys=True)+'\n')
 integration=json.loads((OUT/'integration.json').read_text())
 integration.update({
  'build_status':'passed',
  'source_build_receipt_sha256':sha(build_path),
  'source_build_receipt_path':str(build_path),
  'build_runner_log_sha256':sha(runner),
  'build_command_logs':logs,
  'build_outputs':artifacts,
  'changed_sources':build['changed_sources'],
  'native_probe_or_gpu_simulation':'not launched',
  'post_build_evidence_path':str(OUT/'post-build.json'),
  'post_build_evidence_sha256':sha(OUT/'post-build.json'),
  'disk_available_bytes_after_build':post['disk_available_bytes_after_build'],
 })
 (OUT/'integration.json').write_text(json.dumps(integration,indent=2,sort_keys=True)+'\n')
 # Seal a hash list for the integration record files, excluding the hash list itself.
 files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='integration-manifest.json')
 manifest={'schema':'numi.cardboard.source145_integration_manifest.v1','files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}}
 (OUT/'integration-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'source145_frozen_sha256':sha(SRC/'frozen-source.json'),'source_build_receipt_sha256':sha(build_path),'post_build_sha256':sha(OUT/'post-build.json'),'integration_json_sha256':sha(OUT/'integration.json'),'integration_manifest_sha256':sha(OUT/'integration-manifest.json'),'inventory_entries':len(after),'unchanged':len(after)-len(delta),'expected_deltas':sorted(delta),'build_success':build['success'],'target':build['targets'][0],'artifact_count':len(artifacts),'disk_available_bytes':post['disk_available_bytes_after_build']},indent=2))
if __name__=='__main__': main()
