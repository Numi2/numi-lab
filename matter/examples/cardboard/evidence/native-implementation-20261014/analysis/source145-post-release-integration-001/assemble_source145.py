#!/usr/bin/env python3
"""Clone source144 and integrate the sealed post-release overlay for source145."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
BASE = B / 'snapshots/source-cgls-state-sidecar-144'
BASE143 = B / 'snapshots/source-cgls-primary-policy-143'
TARGET = B / 'snapshots/source-cgls-post-release-relaxation-145'
OVERLAY = B / 'analysis/cardboard-convergence-tool-overlay-145-001'
OUT = B / 'analysis/source145-post-release-integration-001'
BUILD = R / 'build-cardboard-provenance-145'
PATCH = OVERLAY / 'post-release-relaxation.patch'
OVERLAY_MANIFEST = OVERLAY / 'manifest.json'
MIN_FREE = 2 * 1024**3
CHANGED = [
    'matter/tools/cardboard_probe.mm',
    'matter/tools/cardboard_convergence.py',
    'matter/tools/cardboard_resolution_study.py',
    'matter/tools/test_cardboard_convergence.py',
    'matter/tools/test_cardboard_resolution_study.py',
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_bytes(path: Path = B) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def inventory(root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        current_path = Path(current)
        for name in files:
            path = current_path / name
            rel = str(path.relative_to(root))
            if path.is_symlink():
                result[rel] = {'kind': 'symlink', 'target': os.readlink(path)}
            elif path.is_file():
                result[rel] = {'kind': 'file', 'sha256': sha(path), 'bytes': path.stat().st_size}
    return result


def main() -> None:
    existing_outputs = (set(OUT.iterdir()) - {OUT / 'assemble_source145.py'}) if OUT.exists() else set()
    if TARGET.exists() or BUILD.exists() or existing_outputs:
        raise SystemExit('source145, integration record, or build directory already exists; refusing to overwrite')
    if free_bytes() < MIN_FREE:
        raise SystemExit(f'free disk is below 2 GiB: {free_bytes()} bytes')

    base_frozen = json.loads((BASE / 'frozen-source.json').read_text())
    base_support = json.loads((BASE / 'configure-support-source.json').read_text())
    base143_frozen = json.loads((BASE143 / 'frozen-source.json').read_text())
    overlay_manifest = json.loads(OVERLAY_MANIFEST.read_text())
    if not PATCH.is_file():
        raise SystemExit(f'missing overlay patch: {PATCH}')

    for rel, digest in base_frozen['source_sha256'].items():
        path = BASE / rel
        if not path.is_file() or sha(path) != digest:
            raise SystemExit(f'source144 frozen hash mismatch: {rel}')
    for rel, digest in base_support['files'].items():
        path = BASE / rel
        if not path.is_file() or sha(path) != digest:
            raise SystemExit(f'source144 configure-support hash mismatch: {rel}')
    for rel in CHANGED:
        if sha(BASE143 / rel) != sha(BASE / rel):
            raise SystemExit(f'overlay patch base differs from source144: {rel}')
        expected = base143_frozen['source_sha256'].get(rel)
        if expected != sha(BASE / rel):
            raise SystemExit(f'source143 frozen hash does not match source144 for {rel}')
    if overlay_manifest.get('report_sha256') != sha(OVERLAY / 'report.json'):
        raise SystemExit('overlay report hash does not match its manifest')

    before = free_bytes()
    OUT.mkdir(exist_ok=True)
    (OUT / 'preflight.json').write_text(json.dumps({
        'base_snapshot': str(BASE),
        'base_frozen_source_sha256': sha(BASE / 'frozen-source.json'),
        'base_configure_support_sha256': sha(BASE / 'configure-support-source.json'),
        'base_inventory_file_count': len(inventory(BASE)),
        'base_frozen_source_entries': len(base_frozen['source_sha256']),
        'base_configure_support_entries': len(base_support['files']),
        'overlay_manifest_sha256': sha(OVERLAY_MANIFEST),
        'overlay_patch_sha256': sha(PATCH),
        'changed_paths': CHANGED,
        'base_hashes_match_143_for_all_changed_paths': True,
        'available_bytes_before_clone': before,
        'available_gib_before_clone': before / 1024**3,
        'minimum_required_bytes': MIN_FREE,
        'native_gpu_launch': False,
    }, indent=2, sort_keys=True) + '\n')

    cp = subprocess.run(['/bin/cp', '-cR', str(BASE), str(TARGET)],
                        capture_output=True, text=True, check=False)
    (OUT / 'clone.stdout.log').write_text(cp.stdout)
    (OUT / 'clone.stderr.log').write_text(cp.stderr)
    if cp.returncode != 0:
        raise SystemExit(f'APFS clone failed ({cp.returncode}): {cp.stderr}')

    patch = subprocess.run(['patch', '--batch', '--forward', '-p1', '-d', str(TARGET),
                            '-i', str(PATCH)], capture_output=True, text=True,
                           check=False)
    (OUT / 'patch.stdout.log').write_text(patch.stdout)
    (OUT / 'patch.stderr.log').write_text(patch.stderr)
    if patch.returncode != 0:
        raise SystemExit(f'overlay application failed ({patch.returncode}): {patch.stderr}')

    frozen = json.loads((TARGET / 'frozen-source.json').read_text())
    if frozen['source_sha256'] != base_frozen['source_sha256']:
        raise SystemExit('unexpected source map mutation during clone or patch')
    for rel in CHANGED:
        frozen['source_sha256'][rel] = sha(TARGET / rel)
    frozen['base_snapshot'] = str(BASE)
    frozen['scope'] = (
        'Source144 plus a default-zero explicit post-release relaxation phase for the end-grip coupon. '
        'A positive post-release segment follows the existing accepted free-release segment, with released '
        'right-grip nodes and accepted material history carried forward; complete resolution studies require '
        'positive counts for both phases. Source144 solver policy and shipping released_assembly_settle are '
        'preserved. This change adds a prescribed schedule/analysis phase only; no physical qualification is implied.'
    )
    (TARGET / 'frozen-source.json').write_text(json.dumps(frozen, indent=2) + '\n')

    if (TARGET / 'configure-support-source.json').read_bytes() != (BASE / 'configure-support-source.json').read_bytes():
        raise SystemExit('configure-support-source.json changed')
    for rel, digest in frozen['source_sha256'].items():
        if sha(TARGET / rel) != digest:
            raise SystemExit(f'source145 frozen hash mismatch: {rel}')
    for rel, digest in base_support['files'].items():
        if sha(TARGET / rel) != digest:
            raise SystemExit(f'source145 configure-support hash mismatch: {rel}')

    before_inventory = inventory(BASE)
    after_inventory = inventory(TARGET)
    if before_inventory.keys() != after_inventory.keys():
        missing = sorted(before_inventory.keys() - after_inventory.keys())
        extra = sorted(after_inventory.keys() - before_inventory.keys())
        raise SystemExit(f'clone inventory changed; missing={missing[:8]} extra={extra[:8]}')
    allowed = set(CHANGED) | {'frozen-source.json'}
    unexpected = [rel for rel in before_inventory
                  if rel not in allowed and before_inventory[rel] != after_inventory[rel]]
    if unexpected:
        raise SystemExit(f'unexpected changed paths outside overlay: {unexpected[:16]}')
    changed = {rel: after_inventory[rel] for rel in sorted(allowed)}

    command = [
        '/Users/home/.pyenv/versions/3.10.12/bin/python3',
        str(TARGET / 'matter/tools/cardboard_source_bound_build.py'),
        '--source-snapshot', str(TARGET),
        '--build-dir', str(BUILD),
        '--cmake-source-relative', '.',
        '--jobs', '4',
        '--target', 'numi-matter-cardboard-probe',
    ]
    (OUT / 'build-command.json').write_text(json.dumps(command, indent=2) + '\n')
    after_clone = free_bytes()
    integration = {
        'schema': 'numi.cardboard.source145_post_release_integration.v1',
        'snapshot': str(TARGET),
        'base_snapshot': str(BASE),
        'base_frozen_source_sha256': sha(BASE / 'frozen-source.json'),
        'frozen_source_sha256': sha(TARGET / 'frozen-source.json'),
        'configure_support_source_sha256': sha(TARGET / 'configure-support-source.json'),
        'overlay_manifest_sha256': sha(OVERLAY_MANIFEST),
        'overlay_patch_sha256': sha(PATCH),
        'changed_files': changed,
        'unchanged_file_count': len(before_inventory) - len(allowed),
        'full_file_inventory_count': len(after_inventory),
        'configure_support_file_count': len(base_support['files']),
        'source_hash_entry_count': len(frozen['source_sha256']),
        'configure_support_hashes_preserved': True,
        'all_non_overlay_files_byte_identical_to_source144': True,
        'disk_available_bytes_before_clone': before,
        'disk_available_bytes_after_clone': after_clone,
        'build_command_path': str(OUT / 'build-command.json'),
        'build_directory': str(BUILD),
        'build_status': 'prepared_not_yet_run',
        'native_probe_or_gpu_simulation': 'not launched',
    }
    (OUT / 'integration.json').write_text(json.dumps(integration, indent=2, sort_keys=True) + '\n')
    (OUT / 'base-inventory.json').write_text(json.dumps(before_inventory, indent=2, sort_keys=True) + '\n')
    (OUT / 'source145-inventory.json').write_text(json.dumps(after_inventory, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'snapshot': str(TARGET), 'integration': str(OUT / 'integration.json'),
                      'build_command': command, 'available_bytes_before': before,
                      'available_bytes_after_clone': after_clone,
                      'full_file_inventory_count': len(after_inventory),
                      'unchanged_file_count': len(before_inventory) - len(allowed)}, indent=2))


if __name__ == '__main__':
    main()
