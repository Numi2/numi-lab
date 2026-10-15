#!/usr/bin/env python3
"""Build and seal the source-bound native VTCCD budget replay package."""
from __future__ import annotations
import hashlib
import json
import pathlib

PKG=pathlib.Path(__file__).parent
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014/snapshots/source-native-tet858-vtccd-traces-122')
NATIVE=pathlib.Path('/Users/home/cardboard-implementation-20261014/native')
ADJ=pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/trial21-22-exact-vtccd-adjudication-001')

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def add_path(table: dict[str,dict], path: pathlib.Path, label: str) -> None:
    path=path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data=path.read_bytes()
    key=str(path)
    if key in table:
        assert table[key]['bytes']==len(data) and table[key]['sha256']==sha(data)
        return
    table[key]={'path':key,'label':label,'bytes':len(data),'sha256':sha(data)}

def dependencies(dep_path: pathlib.Path) -> list[pathlib.Path]:
    text=dep_path.read_text().replace('\\\n',' ')
    return [pathlib.Path(p) for p in sorted(set(text.split(':',1)[1].split()))]

def main() -> None:
    evidence:dict[str,dict]={}
    source_files=[
        ('matter/src/metal/contact.metalinc','source122/contact.metalinc'),
        ('matter/src/matter_kernels.metal','source122/matter_kernels.metal'),
        ('matter/CMakeLists.txt','source122/matter/CMakeLists.txt'),
        ('matter/include/numi/matter/shared.h','source122/shared.h'),
        ('matter/src/metal/common.metalinc','source122/common.metalinc'),
    ]
    for rel,label in source_files: add_path(evidence,BASE/rel,label)
    assert sha((BASE/'matter/CMakeLists.txt').read_bytes())=='d64689e33063913c13493d6e621767124519c29432b45ee236deab01ca1c7448'
    for dep_path in (PKG/'validation/matter-kernel-dependencies.d',PKG/'validation/host-dependencies.d'):
        for dep in dependencies(dep_path):
            add_path(evidence,dep,'compiler dependency')
    for trial,observed,baseline in [
        (21,'full-contact-trial21-vtccd-call-405','full-contact-trial21-capture-398'),
        (22,'full-contact-trial22-vtccd-call-406','full-contact-trial22-capture-399')]:
        for dirname in (observed,baseline):
            run=NATIVE/dirname
            for name in ('prediction.json','receipt.json','stdout.log','stderr.log',
                         'trial-capture.coupled.json','trial-capture.coupled.bin'):
                add_path(evidence,run/name,f'trial{trial}/{dirname}/{name}')
    for name in ('README.md','audit.py','report.json','manifest.json','seal.json','validation.log'):
        add_path(evidence,ADJ/name,f'previous sealed adjudication/{name}')

    all_files=[]
    for path in sorted(PKG.rglob('*')):
        if path.is_dir(): continue
        if path.name in {'manifest.json','seal.json'}: continue
        if '__pycache__' in path.parts: raise RuntimeError(f'bytecode cache in package: {path}')
        if path.is_symlink(): raise RuntimeError(f'symlink not allowed in sealed package: {path}')
        if not path.is_file(): raise RuntimeError(f'non-regular package entry: {path}')
        data=path.read_bytes()
        all_files.append({'path':str(path.relative_to(PKG)),'bytes':len(data),'sha256':sha(data)})
    candidate=PKG/'overlay/matter/src/metal/contact.metalinc'
    assert sha(candidate.read_bytes())=='cd5aaee17244b52e0b6eebb81cf2fcaff4fbbfb47fe43726fc5bbacec2e9527a'
    patch=PKG/'overlay.patch'
    manifest={
      'schema':'numi.cardboard.vtccd_budget_replay_overlay_manifest.v1',
      'package':PKG.name,
      'status':'PASS_CPU_SOURCE_CAPTURE_METAL_AND_OBJC_SYNTAX; NATIVE_REPLAY_NOT_RUN',
      'base_snapshot':str(BASE),
      'base_contact_sha256':'ea5a9caf6f47588258aeeb5e8be57168826acfb94a566ecbfea7e647ad50fb5c',
      'base_cmake_sha256':'d64689e33063913c13493d6e621767124519c29432b45ee236deab01ca1c7448',
      'candidate_contact_sha256':sha(candidate.read_bytes()),
      'overlay_patch_sha256':sha(patch.read_bytes()),
      'default_production_iteration_budget':16,
      'diagnostic_budgets':[16,32,64,128],
      'physical_tolerances_or_roundoff_changed':False,
      'native_gpu_replay_performed':False,
      'outputs':all_files,
      'input_evidence':list(evidence.values()),
    }
    manifest_path=PKG/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    manifest_sha=sha(manifest_path.read_bytes())
    seal={'schema':'numi.cardboard.analysis-package-seal.v1','package':PKG.name,
          'manifest_sha256':manifest_sha,
          'files':{row['path']:row['sha256'] for row in all_files}}
    seal['files']['manifest.json']=manifest_sha
    seal_path=PKG/'seal.json'
    seal_path.write_text(json.dumps(seal,indent=2)+'\n')
    for row in all_files:
        assert sha((PKG/row['path']).read_bytes())==row['sha256']
    assert sha(manifest_path.read_bytes())==manifest_sha
    print(json.dumps({'manifest_sha256':manifest_sha,'seal_sha256':sha(seal_path.read_bytes()),
      'overlay_patch_sha256':sha(patch.read_bytes()),'output_count':len(all_files),
      'input_evidence_count':len(evidence),'candidate_contact_sha256':sha(candidate.read_bytes())},sort_keys=True))

if __name__=='__main__': main()
