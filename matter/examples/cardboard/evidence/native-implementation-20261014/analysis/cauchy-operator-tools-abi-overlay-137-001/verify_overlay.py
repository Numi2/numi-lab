#!/usr/bin/env python3
"""Source-bound static/patch checks for the direct FEM operator ABI shim."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import sys

PKG = Path(__file__).resolve().parent
ROOT = Path('/Users/home/cardboard-implementation-20261014')
BASE = ROOT / 'snapshots/source-cauchy-positive-mass-admission-135'
RETAINED = ROOT / 'analysis/retained-source-snapshot135-001'
EXPECTED = {
    'matter/tools/cardboard_coupled_free_box_operator_check.mm': 'f0106a2a005c04bae44376c73394f75e39ede61876aa3010bbd819b5c53499b3',
    'matter/tools/fem_regional_material_check.mm': '8e70f606481f249eda8d4fe59ea206890969a213f44a6ee35255d570f6986bf8',
    'matter/tools/fem_material_frame_check.mm': 'eb036270e9afd06ab5dafa0a8f4b2cb00823fa5eb9ee8b09f4c9512093ac3101',
    'matter/tools/cardboard_gpu_check.mm': '423a03bddd9ab8321188c25fbd5aa73c6d28fb5ac538885829e65fda00d592dc',
    'matter/tools/fem_reference_check.mm': '9aabd75024c23824c19fb9fc71a0b2d888baeb1570295addefa45c324e805c13',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def main() -> int:
    snapshot = json.loads((RETAINED / 'manifest.json').read_text())['snapshots'][0]
    bundle = RETAINED / snapshot['manifest']
    compressed = bundle.read_bytes()
    raw = gzip.decompress(compressed)
    doc = json.loads(raw)
    need(sha(bundle) == snapshot['compressed_sha256'], 'retained source135 compressed hash mismatch')
    need(hashlib.sha256(raw).hexdigest() == snapshot['uncompressed_sha256'], 'retained source135 bundle hash mismatch')
    need(doc['number'] == 135 and Path(doc['source_snapshot']) == BASE, 'retained bundle is not source135')
    by_path = {entry['path']: entry for entry in doc['files']}

    hashes = {'base': {}, 'candidate': {}}
    checks = {}
    for rel, expected in EXPECTED.items():
        base = PKG / 'base' / rel
        candidate = PKG / 'candidate' / rel
        base_hash = sha(base)
        candidate_hash = sha(candidate)
        need(base_hash == expected, f'overlay base mismatch: {rel}')
        need(sha(BASE / rel) == expected, f'frozen source135 mismatch: {rel}')
        need(by_path.get(rel, {}).get('sha256') == expected, f'source135 manifest mismatch: {rel}')
        hashes['base'][rel] = base_hash
        hashes['candidate'][rel] = candidate_hash
        text = candidate.read_text()
        need('nm_fem_apply_operator_elements' in text, f'operator call disappeared: {rel}')
        need('std::vector<nm_float4>' in text and 'w.dispatch.environmentCount' in text or
             'std::vector<nm_float4>' in text and 'world.dispatch.environmentCount' in text,
             f'environment-sized metric vector absent: {rel}')
        if rel.endswith('cardboard_coupled_free_box_operator_check.mm'):
            need('id<MTLBuffer> femArgs[27] = {};' in text, 'coupled argument array was not extended')
            need('femArgs[26] = fallbackMetricsB;' in text, 'coupled argument 26 is not assigned')
            need('for (NSUInteger i = 2u; i < 27u; ++i)' in text, 'coupled binding loop does not include index 26')
            checks[rel] = {'direct_operator_buffer26': True, 'array_and_loop_extended': True}
        else:
            need('[encoder setBuffer:fallbackMetrics offset:0 atIndex:26];' in text,
                 f'direct operator buffer 26 missing: {rel}')
            checks[rel] = {'direct_operator_buffer26': True}
        import re
        need(re.search(r'std::vector<nm_float4>\s*\(\s*(?:w|world)\.dispatch\.environmentCount\s*,\s*nm_float4\{\}\s*\)', text) is not None,
             f'buffer 26 is not environment-count-sized and zero-initialized: {rel}')
    symlinks = [str(p.relative_to(PKG)) for p in PKG.rglob('*') if p.is_symlink()]
    need(not symlinks, f'symlinks are not allowed: {symlinks}')
    patch = PKG / 'overlay.patch'
    result = subprocess.run(['patch','--dry-run','-p1','-d',str(PKG/'base')],
                            input=patch.read_bytes(),stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    need(result.returncode == 0, 'overlay.patch does not dry-run against source135 base')
    report = {
        'schema': 'numi.cardboard.operator_tools_buffer26_abi_review.v1',
        'base_snapshot': str(BASE),
        'base_file_sha256': hashes['base'],
        'candidate_file_sha256': hashes['candidate'],
        'direct_operator_checks': checks,
        'all_buffer26_values_zero_initialized': True,
        'patch_sha256': sha(patch),
        'patch_dry_run': 'PASS',
        'symlinks': symlinks,
        'gpu_run': 'NOT_RUN',
        'physical_qualification': False,
    }
    (PKG/'logs/static-review-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'verification failed: {e}',file=sys.stderr)
        raise SystemExit(1)
