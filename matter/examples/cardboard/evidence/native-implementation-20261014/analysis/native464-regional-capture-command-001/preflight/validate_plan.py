#!/usr/bin/env python3
"""Read-only admission of the prepared 464 command; never launches it."""
from __future__ import annotations
import hashlib,json,os,py_compile
from pathlib import Path
B=Path('/Users/home/cardboard-implementation-20261014')
R=B/'analysis/native464-regional-capture-command-001'
base=json.loads((B/'native-462-command.json').read_text())
cmd=json.loads((R/'native-464-command.json').read_text())
bind=json.loads((R/'source151-regional-capture-bindings.json').read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pairopt(a,opt): return a[a.index(opt)+1]
for key,val in (('--source-snapshot','/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-regional-state-capture-151'),('--runtime-dir','/Users/home/numi-cardboard-20261006/build-cardboard-provenance-151'),('--build-provenance','/Users/home/numi-cardboard-20261006/build-cardboard-provenance-151/source-build.json'),('--case','/Users/home/cardboard-implementation-20261014/solver/cgls-regional-state-capture-464')):
 assert pairopt(cmd,key)==val,(key,pairopt(cmd,key))
assert '--' in cmd and '--' in base
# Every post-separator run argument is byte-for-byte the source462 invocation
# after only source/build/case path substitutions.
def normalize_run(a):
 return [x.replace('source-cgls-block-right-preconditioned-149','source-cgls-regional-state-capture-151')
          .replace('build-cardboard-provenance-149','build-cardboard-provenance-151')
          .replace('cgls-block-right-disabled-parity-462','cgls-regional-state-capture-464')
         for x in a[a.index('--')+1:]]
assert normalize_run(cmd)==normalize_run(base),'native physics/solver argv diverges beyond named source/case paths'
# Environment delta must be solely diagnostic and explicit.
def envs(a):
 return dict(x.split('=',1) for i,x in enumerate(a) if i and a[i-1]=='--env')
e0=envs(base); e1=envs(cmd)
removed={k for k in e0 if k not in e1}; added={k for k in e1 if k not in e0}
assert removed=={'NM_FEM_COUPLED_CAPTURE_SELECTOR','NM_FEM_COUPLED_CAPTURE_PATH'},removed
assert added=={'NM_FEM_BASE_CGLS_STATE_CAPTURE_REGIONAL'},added
allowed_value_path_changes={'NM_FEM_STATUS5_CAPTURE_PATH','NM_FEM_BASE_CGLS_CAPTURE_PATH'}
for k in set(e0)&set(e1):
 left=e0[k].replace('cgls-block-right-disabled-parity-462','cgls-regional-state-capture-464')
 right=e1[k]
 assert left==right,(k,left,right)
assert e1['NM_FEM_BASE_CGLS_STATE_CAPTURE']=='1'
assert e1['NM_FEM_BASE_CGLS_STATE_CAPTURE_REGIONAL']=='1'
assert not any(k in e1 for k in ('NM_FEM_OPERATOR_FD_SELECTOR','NM_FEM_COUPLED_TRIAL_CAPTURE_SELECTOR','NM_FEM_COUPLED_TRIAL_SUMMARY_SELECTOR','NM_FEM_REGIONAL_RAP_CAPTURE_ONLY'))
# Source/build integrity and no changed source inputs.
source=Path(pairopt(cmd,'--source-snapshot')); build_path=Path(pairopt(cmd,'--build-provenance'))
build=json.loads(build_path.read_text())
assert build['success'] and build['changed_sources']==[]
assert Path(build['source_snapshot']).resolve()==source.resolve()
assert Path(build['build_dir']).resolve()==Path(pairopt(cmd,'--runtime-dir')).resolve()
source_checks=build['source_sha256']
for rel,expected in source_checks.items():
 p=source/rel
 assert p.is_file() and sha(p)==expected, f'source drift: {rel}'
for name,expected in build['artifacts_sha256'].items():
 p=Path(name)
 assert p.is_file() and sha(p)==expected, f'artifact drift: {name}'
assert sha(source/'frozen-source.json')==bind['source151_frozen_manifest_sha256']
# Exact material bytes equal retained source149 run.
for filename,expected in [('hajali2009_liner_hill_bilinear_fit.nmatter','a2720b90f2004f2eccf180063e7ecc4e0acdda93f700c4fd357914fc24d2255a'),('hajali2009_medium_hill_bilinear_fit.nmatter','76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9')]:
 p=source/'matter/materials'/filename
 assert sha(p)==expected
# All inputs copied by wrapper must be existing regular files with unique basenames.
inputs=[cmd[i+1] for i,x in enumerate(cmd) if x=='--input-file']
assert len(inputs)==len(set(Path(p).name for p in inputs))
assert all(Path(p).is_file() and not Path(p).is_symlink() for p in inputs)
case=Path(pairopt(cmd,'--case'))
assert not case.exists(),'planned case is already occupied'
# Checked conservative readback budget from exact graph layout and source151 append shapes.
capacity=json.loads((R/'preflight/capture-capacity-preflight.json').read_text())
assert capacity['aggregate_fits_96MiB'] and capacity['aggregate_readback_bytes']<96*1024*1024
assert capacity['arrays']['state_count']==24 and capacity['arrays']['regional_max_count']==30
# Capture collision guard in source151: legacy coupled observer is absent.
assert 'NM_FEM_COUPLED_CAPTURE_SELECTOR' not in e1 and 'NM_FEM_COUPLED_CAPTURE_PATH' not in e1
for f in (R/'run_native464.py', R/'preflight/build_graph_input.py', R/'preflight/preflight_capacity.py', R/'validator/read_regional_capture_v2.py'):
 py_compile.compile(str(f),doraise=True)
report={
 'schema':'numi.cardboard.native464.command-preflight.v1','native_executed':False,
 'source_build_success':build['success'],'source_sha256_entries_checked':len(source_checks),
 'build_artifacts_checked':len(build['artifacts_sha256']),
 'run_argv_physics_solver_match_source462_after_path_normalization':True,
 'removed_only_incompatible_observer_envs':sorted(removed),
 'added_only_regional_observer_env':sorted(added),
 'same_liner_material_bytes':True,'same_medium_material_bytes':True,
 'input_files_unique_regular':len(inputs),'source151_regional_state_capture_requested':True,
 'selector5_graph_preflight':json.loads((R/'preflight/graph-layout.json').read_text()),
 'aggregate_readback_upper_bound_bytes':capacity['aggregate_readback_bytes'],
 'aggregate_cap_bytes':96*1024*1024,
 'case_path_unoccupied':True,
 'verdict':'READY_FOR_PARENT_TO_LAUNCH; no native run or qualification claimed',
}
(R/'preflight/command-validation.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
