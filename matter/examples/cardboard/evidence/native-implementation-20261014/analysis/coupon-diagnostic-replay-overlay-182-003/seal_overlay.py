#!/usr/bin/env python3
"""Verify and seal immutable coupon diagnostic replay overlay 182-003."""
from __future__ import annotations
import hashlib, json, pathlib

PACKAGE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path('/Users/home/cardboard-implementation-20261014')
BASE = ROOT / 'snapshots/source-admission-encoder-181'
PARENT = ROOT / 'analysis/coupon-diagnostic-replay-overlay-182-002'
MANIFEST = PACKAGE / 'overlay-manifest.json'
SEAL = PACKAGE / 'validation/seal.txt'
EXCLUDED = {'overlay-manifest.json', 'validation/seal.txt'}

def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    if not json.loads((PACKAGE / 'validation/validation-result.json').read_text()).get('patch_application') == 'PASS':
        raise SystemExit('validation result is not passing')
    result = json.loads((PACKAGE / 'validation/validation-result.json').read_text())
    if result.get('contract_test', {}).get('run_exit') != 0 or result.get('objcxx_syntax_targets') != ['default', 'box-native']:
        raise SystemExit('CPU contract or syntax validation is incomplete')
    parent_manifest = PARENT / 'overlay-manifest.json'
    parent_seal = PARENT / 'validation/seal.txt'
    parent_manifest_sha = sha(parent_manifest)
    parent_seal_text = parent_seal.read_text().strip().split()[0]
    if parent_seal_text != parent_manifest_sha:
        raise SystemExit('182-002 manifest does not match its preserved seal')
    frozen = json.loads((BASE / 'frozen-source.json').read_text())
    files = []
    for path in sorted(PACKAGE.rglob('*')):
        if path.is_dir():
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        if rel in EXCLUDED:
            continue
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f'non-regular overlay inventory entry: {rel}')
        files.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    manifest = {
        'schema': 'numi.cardboard.source-overlay.v1',
        'overlay_id': 'coupon-diagnostic-replay-overlay-182-003',
        'status': 'SEALED_ISOLATED_OVERLAY',
        'base_snapshot': str(BASE),
        'base_revision': 'a5ccffa52ffe368b49c877094a5081b7ee246ebf',
        'base_frozen_source_sha256': sha(BASE / 'frozen-source.json'),
        'base_source_file_sha256': {
            name: frozen['source_sha256'][f'matter/tools/{name}']
            for name in ('cardboard_coupon_resume.hpp', 'cardboard_probe.mm',
                         'cardboard_coupon_resume_check.cpp')},
        'validation_host': 'macmini',
        'base_source_build_json_sha256': result['base_source_build_json_sha256'],
        'base_compile_commands_sha256': result['base_compile_commands_sha256'],
        'changed_source_files': [
            'matter/tools/cardboard_coupon_resume.hpp',
            'matter/tools/cardboard_probe.mm',
            'matter/tools/cardboard_coupon_resume_check.cpp'],
        'supersedes_overlay': 'coupon-diagnostic-replay-overlay-182-002',
        'supersedes_overlay_manifest_sha256': parent_manifest_sha,
        'supersedes_overlay_seal_sha256': parent_seal_text,
        'correction': 'Publish validated current runner proof digests even when output paths are absent; report explicit source/current proof modes; keep older aggregate initializers warning-clean and exercise replay admission in the existing CTest.',
        'native_gpu_run': 'NOT_RUN',
        'physical_or_full_trajectory_qualification': 'NOT_CLAIMED',
        'inventory': files,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    manifest_sha = sha(MANIFEST)
    SEAL.write_text(f'{manifest_sha}  overlay-manifest.json\n')
    print(json.dumps({'manifest_sha256': manifest_sha, 'inventory_files': len(files),
                      'parent_manifest_sha256': parent_manifest_sha}, indent=2))

if __name__ == '__main__':
    main()
