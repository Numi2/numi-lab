#!/usr/bin/env python3
"""Run source-181-bound ObjC++ syntax checks for overlay 182-003."""
from __future__ import annotations
import hashlib, json, pathlib, shlex, subprocess, sys

ROOT = pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
BASE = ROOT / 'snapshots/source-admission-encoder-181'
BUILD = ROOT / 'build-source181-001'
PACKAGE = ROOT / 'analysis/coupon-diagnostic-replay-overlay-182-003'
OUT = PACKAGE / 'validation/objcxx-syntax.json'

def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def argv_of(entry: dict) -> list[str]:
    return list(entry['arguments']) if 'arguments' in entry else shlex.split(entry['command'])

def main() -> int:
    frozen = json.loads((BASE / 'frozen-source.json').read_text())
    source_build = BUILD / 'source-build.json'
    commands_path = BUILD / 'compile_commands.json'
    if sha(BASE / 'frozen-source.json') != 'f3ac253b7924dc4cb6ba1d5d9580ceea42150b5e49c6eee569e36b0e204fef7a':
        raise SystemExit('source181 frozen manifest hash mismatch')
    if sha(source_build) != 'd5ed1e058065435d7faeb48897681306907e61a7b9f765f3c131c38261e1c5be':
        raise SystemExit('source181 source-build hash mismatch')
    if sha(commands_path) != '050008734269639ebd1b27163f4b4d5e7e099d56de42c73ad89abb25163a34af':
        raise SystemExit('source181 compile_commands hash mismatch')
    source_names = [('cardboard_coupon_resume.hpp', 'matter/tools/cardboard_coupon_resume.hpp'),
                    ('cardboard_probe.mm', 'matter/tools/cardboard_probe.mm'),
                    ('cardboard_coupon_resume_check.cpp', 'matter/tools/cardboard_coupon_resume_check.cpp')]
    for name, rel in source_names:
        candidate = PACKAGE / 'candidate' / rel
        if not candidate.is_file() or candidate.is_symlink():
            raise SystemExit(f'missing or aliased candidate {rel}')
        original = BASE / rel
        if frozen['source_sha256'].get(rel) != sha(original):
            raise SystemExit(f'source181 file does not match frozen manifest: {rel}')
    entries = json.loads(commands_path.read_text())
    expected = str(BASE / 'matter/tools/cardboard_probe.mm')
    matches = [e for e in entries if e.get('file') == expected]
    if len(matches) != 2:
        raise SystemExit(f'expected exactly two probe compile entries, got {len(matches)}')
    results = []
    for entry in matches:
        args = argv_of(entry)
        args = [str(PACKAGE / 'candidate/matter/tools/cardboard_probe.mm') if x == expected else x for x in args]
        cleaned = []
        skip = False
        for arg in args:
            if skip:
                skip = False
                continue
            if arg == '-o':
                skip = True
                continue
            if arg == '-c':
                continue
            cleaned.append(arg)
        if skip:
            raise SystemExit('compile command ended after -o')
        cleaned.extend(['-fsyntax-only', '-I', str(PACKAGE / 'candidate/matter/tools'), '-I', str(BASE / 'matter/tools')])
        target = 'box-native' if '-DNUMI_CARDBOARD_BOX_NATIVE_DRIVER=1' in cleaned else 'default'
        run = subprocess.run(cleaned, cwd=entry.get('directory', str(ROOT)), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        results.append({'target': target, 'argv': cleaned, 'exit': run.returncode, 'output': run.stdout})
    if {r['target'] for r in results} != {'default', 'box-native'} or any(r['exit'] for r in results):
        OUT.write_text(json.dumps(results, indent=2) + '\n')
        return 1
    OUT.write_text(json.dumps(results, indent=2) + '\n')
    bindings = {
        'schema': 'numi.cardboard.coupon-diagnostic-replay-host-bindings.v2',
        'host': 'macmini',
        'source_build_json_sha256': sha(source_build),
        'compile_commands_sha256': sha(commands_path),
        'frozen_source_sha256': sha(BASE / 'frozen-source.json'),
        'source181_probe_sha256': sha(BASE / 'matter/tools/cardboard_probe.mm'),
        'candidate_sha256': {
            'cardboard_coupon_resume.hpp': sha(PACKAGE / 'candidate/matter/tools/cardboard_coupon_resume.hpp'),
            'cardboard_probe.mm': sha(PACKAGE / 'candidate/matter/tools/cardboard_probe.mm'),
            'cardboard_coupon_resume_check.cpp': sha(PACKAGE / 'candidate/matter/tools/cardboard_coupon_resume_check.cpp')},
        'syntax_log_sha256': sha(OUT),
        'syntax_targets': [r['target'] for r in results],
        'syntax_exit_codes': [r['exit'] for r in results],
    }
    (PACKAGE / 'validation/host-bindings.json').write_text(json.dumps(bindings, indent=2) + '\n')
    print(json.dumps({'targets': bindings['syntax_targets'], 'exit_codes': bindings['syntax_exit_codes'],
                      'source_build_sha256': bindings['source_build_json_sha256'],
                      'syntax_log_sha256': bindings['syntax_log_sha256']}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
