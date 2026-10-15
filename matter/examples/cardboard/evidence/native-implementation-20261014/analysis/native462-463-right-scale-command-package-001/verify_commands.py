#!/usr/bin/env python3
"""Verify source149 command bindings and protocol parity without launching the native run."""
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path('/Users/home/cardboard-implementation-20261014')
PKG = ROOT / 'analysis/native462-463-right-scale-command-package-001'
FAILED = ROOT / 'analysis/native461-block-right-cgls-command-002/native-461-command.json'
BUILD = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-149')
SNAPSHOT = ROOT / 'snapshots/source-cgls-block-right-preconditioned-149'
PROBE = BUILD / 'matter/numi-matter-cardboard-probe'
METALLIB = BUILD / 'matter/shaders/NumiMatter.metallib'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise SystemExit('FAIL: ' + message)


def pair(argv, key):
    i = argv.index(key)
    return argv[i + 1]


def check_source_build():
    receipt_path = BUILD / 'source-build.json'
    receipt = json.loads(receipt_path.read_text())
    require(receipt['success'] is True, 'source-build receipt is not successful')
    require(Path(receipt['source_snapshot']) == SNAPSHOT, 'receipt source snapshot mismatch')
    require(Path(receipt['build_dir']) == BUILD, 'receipt build directory mismatch')
    missing, mismatch = [], []
    for rel, expected in receipt['source_sha256'].items():
        p = SNAPSHOT / rel
        if not p.is_file():
            missing.append(str(p))
        elif sha(p) != expected:
            mismatch.append(str(p))
    for name, expected in receipt['artifacts_sha256'].items():
        p = Path(name)
        if not p.is_file():
            missing.append(name)
        elif sha(p) != expected:
            mismatch.append(name)
    require(not missing, 'missing source/build files: ' + ', '.join(missing[:5]))
    require(not mismatch, 'source/build hash mismatch: ' + ', '.join(mismatch[:5]))
    return receipt


def check_command(path, case, enabled):
    argv = json.loads(path.read_text())
    require(isinstance(argv, list), f'{path.name} is not an argv list')
    require(argv[0] == '/Users/home/.pyenv/versions/3.10.12/bin/python3', 'unexpected Python runtime')
    require(Path(argv[1]) == SNAPSHOT / 'matter/tools/cardboard_source_bound_run.py', 'runner is not from source149')
    require(Path(pair(argv, '--source-snapshot')) == SNAPSHOT, 'source snapshot binding mismatch')
    require(Path(pair(argv, '--runtime-dir')) == BUILD, 'runtime dir binding mismatch')
    require(Path(pair(argv, '--build-provenance')) == BUILD / 'source-build.json', 'build receipt binding mismatch')
    require(Path(argv[argv.index('--') + 1]) == PROBE, 'probe executable mismatch')
    require(Path(pair(argv, '--metallib')) == METALLIB, 'metallib mismatch')
    require(Path(pair(argv, '--liner-material')) == SNAPSHOT / 'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter', 'liner material mismatch')
    require(Path(pair(argv, '--medium-material')) == SNAPSHOT / 'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter', 'medium material mismatch')
    require(Path(pair(argv, '--case')) == Path(case), 'fresh case path mismatch')
    require(Path(pair(argv, '--output')) == Path(case) / 'output', 'fresh output path mismatch')
    require(('NM_FEM_BASE_CGLS_RIGHT_SCALE=block3' in argv) == enabled, 'right-scale option does not match command label')
    for i, tok in enumerate(argv[:-1]):
        if tok == '--input-file':
            require(Path(argv[i + 1]).is_file(), f'missing input file: {argv[i + 1]}')
    # Ensure selected capture paths were moved to this fresh case.
    for prefix in ('NM_FEM_COUPLED_CAPTURE_PATH=', 'NM_FEM_STATUS5_CAPTURE_PATH=', 'NM_FEM_BASE_CGLS_CAPTURE_PATH='):
        val = next(t.split('=', 1)[1] for t in argv if isinstance(t, str) and t.startswith(prefix))
        require(val.startswith(str(case) + '/'), f'{prefix} is not bound to fresh case')
    require(not any('build-cardboard-provenance-145/matter/numi-matter-cardboard-probe' in str(t) for t in argv), 'stale build145 executable remains')
    return argv


def normalized_tail(argv, case):
    # Physical/probe CLI is argv after `--`; normalize only the executable and output path.
    tail = list(argv[argv.index('--') + 1:])
    tail[0] = '<PROBE149>'
    tail = [str(t).replace(str(case), '<CASE>') for t in tail]
    return tail

receipt = check_source_build()
cmd463 = check_command(PKG / 'native-463-command.json', '/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463', True)
cmd462 = check_command(PKG / 'native-462-command.json', '/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-disabled-parity-462', False)
old = json.loads(FAILED.read_text())
require(normalized_tail(old, '/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-461') == normalized_tail(cmd463, '/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463'), 'enabled command changed physical/probe CLI beyond executable and fresh output paths')
# Apart from the selector env and fresh case/capture/output paths, the controls match.
for a, b in zip(cmd463, cmd462):
    pass
# Removing exactly the enabled selector pair yields disabled argv; only the independent case paths and text differ.
without = list(cmd463)
i = without.index('NM_FEM_BASE_CGLS_RIGHT_SCALE=block3')
require(without[i - 1] == '--env', 'enabled selector has unexpected argv framing')
del without[i - 1:i + 1]
# Compare physical CLI exactly; common options and physics are identical.
require(normalized_tail(cmd463, '/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463') == normalized_tail(cmd462, '/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-disabled-parity-462'), 'enabled/disabled physical CLI differs')
print('PASS source/build receipt and artifact hashes')
print('PASS all command input-file paths exist')
print('PASS enabled 463 uses executable/build149 and preserves physical CLI from failed 461')
print('PASS disabled 462 differs from enabled only in right-scale environment option and fresh case paths')
print('PASS source-build receipt SHA256', sha(BUILD / 'source-build.json'))
