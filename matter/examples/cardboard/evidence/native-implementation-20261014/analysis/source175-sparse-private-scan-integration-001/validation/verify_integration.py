#!/usr/bin/env python3
"""Verify frozen parent bindings, 169/170/173 lineage merge, and patch reconstruction."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile

B=Path('/Users/home/cardboard-implementation-20261014')
OUT=B/'analysis/source175-sparse-private-scan-integration-001'
P169=B/'snapshots/source-warm-cgls-refinement-169'
P173=B/'snapshots/source-checkpoint-warm-breakdown-173'
O170=B/'analysis/hierarchical-sparse-private-contact-scan-overlay-170-001'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def verify_snapshot(root):
    frozen=json.loads((root/'frozen-source.json').read_text())
    bad=[]
    for rel,digest in frozen['source_sha256'].items():
        p=root/rel
        if p.is_symlink() or not p.is_file() or sha(p)!=digest: bad.append(rel)
    if bad: raise RuntimeError(f'{root.name}: bad frozen source files {bad[:10]}')
    return sha(root/'frozen-source.json'),len(frozen['source_sha256'])

sha169,n169=verify_snapshot(P169)
sha173,n173=verify_snapshot(P173)
if sha169!='f5054091c27a7e1add41630aae1b45ee6ecc3e6d4112ee85c931cc4c70f44ea4': raise RuntimeError('source169 manifest drift')
if sha173!='6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3': raise RuntimeError('source173 manifest drift')
overlay=json.loads((O170/'source-overlay.json').read_text())
if sha(O170/'source-overlay.json')!='55c7aa0d6b35c2bf4912b572e569899ee31c5ddceb56b48bf276118ffbc62e66': raise RuntimeError('source170 manifest drift')
if sha(O170/'source-overlay.patch')!='be6644bf143a32ed9cfcc6d776c455345f7d8631d96dbdd7735df6613a7a2e0e': raise RuntimeError('source170 patch drift')
if overlay['base_frozen_source_json_sha256']!=sha169: raise RuntimeError('source170 does not bind source169')
rows=[]
for entry in overlay['changed_files']:
    rel=entry['path']; candidate=OUT/'candidate'/rel
    src170=O170/'candidate'/rel; parent173=P173/rel; base169=P169/rel
    if sha(candidate)!=entry['candidate_sha256'] and rel!='matter/src/runtime.mm':
        raise RuntimeError(f'source170 candidate changed: {rel}')
    if rel!='matter/src/runtime.mm':
        if parent173.exists() != base169.exists(): raise RuntimeError(f'parent/base presence differs: {rel}')
        if parent173.exists() and sha(parent173)!=sha(base169): raise RuntimeError(f'unexpected source173 overlap: {rel}')
        if sha(candidate)!=sha(src170): raise RuntimeError(f'nonoverlap source170 path not copied exactly: {rel}')
    rows.append({'path':rel,'parent173_sha256':sha(parent173) if parent173.exists() else None,
                 'source170_candidate_sha256':sha(src170),'integrated_candidate_sha256':sha(candidate),
                 'runtime_three_way':rel=='matter/src/runtime.mm'})
rel='matter/src/runtime.mm'
merged=subprocess.run(['git','merge-file','--stdout','-L','source173','-L','source169','-L','source170',str(P173/rel),str(P169/rel),str(O170/'candidate'/rel)],capture_output=True)
if merged.returncode!=0 or merged.stdout!=(OUT/'candidate'/rel).read_bytes() or b'<<<<<<<' in merged.stdout:
    raise RuntimeError('runtime three-way merge does not exactly recreate the integrated runtime')
patch=OUT/'source175-integration.patch'
with tempfile.TemporaryDirectory(prefix='source175-patch-check-') as td:
    root=Path(td)
    for entry in overlay['changed_files']:
        rel=entry['path']; source=P173/rel
        if source.exists():
            dst=root/rel; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(source.read_bytes())
    apply=subprocess.run(['patch','--batch','-p1','-d',td,'-i',str(patch)],text=True,capture_output=True)
    if apply.returncode!=0: raise RuntimeError('patch failed on source173 subset: '+apply.stdout+apply.stderr)
    for entry in overlay['changed_files']:
        rel=entry['path']
        if sha(root/rel)!=sha(OUT/'candidate'/rel): raise RuntimeError('patch result hash mismatch: '+rel)
proof=json.loads((OUT/'three-way-proof.json').read_text())
if proof['runtime_three_way_merge']['runtime_candidate_sha256']!=sha(OUT/'candidate/matter/src/runtime.mm'):
    raise RuntimeError('three-way proof runtime hash mismatch')
report={'schema':'numi.cardboard.source175-integration-verification.v1','source169_files_verified':n169,'source173_files_verified':n173,'source169_frozen_sha256':sha169,'source173_frozen_sha256':sha173,'source170_manifest_sha256':sha(O170/'source-overlay.json'),'source170_patch_sha256':sha(O170/'source-overlay.patch'),'runtime_merge_exit_code':merged.returncode,'runtime_conflict_markers':merged.stdout.count(b'<<<<<<<'),'patch_apply_exit_code':apply.returncode,'changed_files':rows,'gpu_dispatch':False}
(OUT/'validation/integration-verification.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'verified169':n169,'verified173':n173,'changed':len(rows),'patch_apply':'PASS','gpu_dispatch':False},sort_keys=True))
