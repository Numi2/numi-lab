#!/usr/bin/env python3
"""Recheck every C++ FixtureBinding digest against its immutable run input."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/Users/home/cardboard-implementation-20261014')
CHECKER = Path(__file__).resolve().parents[1] / 'prototype/cardboard_tangent_cache_projection_reuse_check.mm'
SPECS = {
    '360': ('reset-augmentation-first-failure-capture-360', 3, [
        'capture', 'package', 'candidate', 'accepted', 'run_manifest',
        'condition_manifest', 'source_build', 'receipt', 'result']),
    '419': ('two-stage-first-failure-rich-fd-419', 2, [
        'capture', 'package', 'candidate', 'accepted', 'run_manifest',
        'condition_manifest', 'source_build', 'receipt', 'result']),
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def paths(run_name: str, step: int) -> dict[str, Path]:
    run = ROOT / 'solver' / run_name
    output = run / 'output'
    stem = f'{step:06d}'
    condition = output / f'rejected-conditioning-{stem}'
    return {
        'capture': run / 'operator-capture',
        'package': output / 'compiled.nmatterpack',
        'candidate': condition / 'candidate_nodes.bin',
        'accepted': output / f'material_state_{stem}_bent.json',
        'run_manifest': output / 'manifest.json',
        'condition_manifest': condition / 'manifest.json',
        'source_build': run / 'inputs/external/source-build.json',
        'receipt': run / 'receipt.json',
        'result': output / 'result.json',
    }

source = CHECKER.read_text()
rows = []
for label, (run_name, step, field_order) in SPECS.items():
    match = re.search(rf'constexpr FixtureBinding kFixture{label}\{{(.*?)\n\}};', source, re.S)
    if not match:
        raise SystemExit(f'missing kFixture{label} initializer')
    literals = re.findall(r'"([^"]*)"', match.group(1))
    hashes = literals[1:1 + len(field_order)]
    if len(hashes) != len(field_order):
        raise SystemExit(f'{label}: expected 9 digest fields in C++ initializer, found {len(hashes)}')
    actual_paths = paths(run_name, step)
    for name, literal in zip(field_order, hashes):
        path = actual_paths[name]
        actual = sha(path)
        ok = bool(re.fullmatch(r'[0-9a-fA-F]{64}', literal)) and actual == literal
        rows.append({'fixture': label, 'field': name, 'path': str(path),
                     'cpp_sha256': literal, 'actual_sha256': actual, 'pass': ok})
        if not ok:
            raise SystemExit(f'{label}.{name}: C++ binding malformed or mismatched: {literal!r} != {actual}')

print(json.dumps({'schema':'numi.cardboard.fixture-binding-audit.v1',
                  'status':'pass','checked_digest_count':len(rows),'bindings':rows},
                 indent=2, sort_keys=True))
