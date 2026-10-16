#!/usr/bin/env python3
"""Verify source157 frozen/build/compile-only bindings without running simulation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

BASE = Path('/Users/home/cardboard-implementation-20261014')
SNAP = BASE / 'snapshots/source-right-grip-telemetry-157'
PARENT = BASE / 'snapshots/source-cgls-all-controls-newton128-154'
BUILD = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-157')
BUILD154 = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-154')
OUT = BASE / 'analysis/source157-telemetry-native-build-001'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    frozen = json.loads((SNAP / 'frozen-source.json').read_text())
    parent_frozen = json.loads((PARENT / 'frozen-source.json').read_text())
    source_build = json.loads((BUILD / 'source-build.json').read_text())
    assert frozen['base_snapshot'] == str(PARENT)
    assert len(frozen['source_sha256']) == 787
    assert source_build['success'] is True
    assert source_build['source_snapshot'] == str(SNAP)
    assert source_build['cmake_source'] == str(SNAP / 'matter')
    assert source_build['targets'] == ['numi-matter-cardboard-probe']
    assert source_build['changed_sources'] == []
    assert source_build['source_sha256'] == frozen['source_sha256']
    assert frozen['source_sha256']['matter/tools/cardboard_probe.mm'] != parent_frozen['source_sha256']['matter/tools/cardboard_probe.mm']
    assert all(frozen['source_sha256'][k] == v for k,v in parent_frozen['source_sha256'].items()
               if k != 'matter/tools/cardboard_probe.mm')
    support = json.loads((SNAP / 'configure-support-source.json').read_text())
    parent_support = json.loads((PARENT / 'configure-support-source.json').read_text())
    assert support == parent_support and len(support['files']) == 1034

    build_command = source_build['commands'][1]
    assert build_command[:4] == ['cmake', '--build', str(BUILD), '--target']
    assert build_command[4:] == ['numi-matter-cardboard-probe', '-j', '2']
    assert source_build['cmake_source'] == str(SNAP / 'matter')
    artifacts = source_build['artifacts_sha256']
    assert len(artifacts) == 3, len(artifacts)
    expected_names = {'numi-matter-cardboard-probe', 'NumiMatter.metallib',
                      'NumiMatterPhysicalStateDigest.metallib'}
    assert {Path(k).name for k in artifacts} == expected_names
    for raw, expected in artifacts.items():
        assert sha(Path(raw)) == expected, raw
    metallib_checks = {}
    for name in ('NumiMatter.metallib', 'NumiMatterPhysicalStateDigest.metallib'):
        p157 = next(Path(k) for k in artifacts if Path(k).name == name)
        p154 = BUILD154 / 'shaders' / name
        metallib_checks[name] = {
            'source157_sha256': sha(p157), 'source154_sha256': sha(p154),
            'byte_identical_to_source154': p157.read_bytes() == p154.read_bytes(),
            'bytes': p157.stat().st_size,
        }
        assert metallib_checks[name]['byte_identical_to_source154']

    protocols = {}
    for budget in (64,128):
        d=OUT/f'compile-only-{budget}'
        result=json.loads((d/'result.json').read_text())
        manifest=json.loads((d/'manifest.json').read_text())
        assert result['status']=='compile_only' and result['accepted_steps']==0
        assert result['physical_validation'] is False
        assert manifest['solver']['newton_iteration_budget']==budget
        assert manifest['solver']['runtime_execution'] is False
        assert manifest['solver']['steps']==12
        assert manifest['solver']['release_steps']==3
        assert manifest['solver']['post_release_relaxation_steps']==0
        assert (d/'compiled.nmatterpack').is_file()
        assert not (d/'observations.csv').exists()
        command=json.loads((OUT/f'compile-only-{budget}-command.json').read_text())
        assert command['native_or_gpu_execution'] is False
        assert command['physical_steps_expected']==0
        protocols[str(budget)]={
            'status':result['status'], 'accepted_steps':result['accepted_steps'],
            'runtime_execution':manifest['solver']['runtime_execution'],
            'newton_iteration_budget':manifest['solver']['newton_iteration_budget'],
            'nodes':result['nodes'], 'tetrahedra':result['tetrahedra'],
            'compiled_world_fingerprint':result['compiled_world_fingerprint'],
            'compiled_pack_sha256':sha(d/'compiled.nmatterpack'),
            'manifest_sha256':sha(d/'manifest.json'),
            'result_sha256':sha(d/'result.json'),
        }

    report={
        'schema':'numi.cardboard.source157.native-build-verification.v1',
        'snapshot':str(SNAP), 'parent_snapshot':str(PARENT),
        'source157_frozen_sha256':sha(SNAP/'frozen-source.json'),
        'source157_overlay_sha256':sha(SNAP/'source-overlay.json'),
        'recovery_metadata_sha256':sha(SNAP/'source157-recovery.json'),
        'source_bound_build_record_sha256':sha(BUILD/'source-build.json'),
        'source_bound_source_count':len(source_build['source_sha256']),
        'configure_support_count':len(support['files']),
        'source_diff_vs_parent':['matter/tools/cardboard_probe.mm'],
        'cmake_source_relative_to_root':'matter',
        'target':'numi-matter-cardboard-probe', 'jobs':2,
        'artifacts_sha256':{Path(k).name:v for k,v in artifacts.items()},
        'metallib_source154_comparison':metallib_checks,
        'compile_only_protocols':protocols,
        'native_or_gpu_execution':'not performed',
        'physical_steps':0,
        'qualification_boundary':'compile-only protocol admission is not a physical/native trajectory validation',
    }
    (OUT/'verification-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        raise
