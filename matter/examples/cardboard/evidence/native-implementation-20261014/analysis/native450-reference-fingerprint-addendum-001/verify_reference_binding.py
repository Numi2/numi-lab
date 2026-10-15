#!/usr/bin/env python3
"""Verify the exact source-bound 419 capture/stderr binding and its relation to run450."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

BASE = Path('/Users/home/cardboard-implementation-20261014')
RUN = BASE / 'solver/bounded-cgls64-exact-state-450'
REF = BASE / 'solver/two-stage-first-failure-rich-fd-419'
SNAPSHOT = BASE / 'snapshots/source-status5-vtccd-diagnostics-125'
AUDIT = BASE / 'analysis/native450-cgls-candidate-audit-001/report.json'
MODULE_PATH = SNAPSHOT / 'matter/tools/cardboard_cached_operator_audit.py'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('bound_operator_audit', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load source-bound operator audit module')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module(MODULE_PATH)
    capture = REF / 'operator-capture'
    header, nodes, tets, integrity = module.capture_records(capture)
    binding = module.bind_inputs(capture, REF / 'prediction.json', REF / 'receipt.json',
                                 REF / 'stderr.log', 0, source_snapshot_override=SNAPSHOT)
    diagnostic = binding['operator_fd_iteration']
    selector_match = binding['prediction_selector'] == f"{header['root']}:{header['newton_iteration']}"
    world_match = int(header['world_fingerprint_hex'], 16) == int(diagnostic['world_fingerprint'])
    execution_match = int(header['execution_fingerprint_hex'], 16) == int(diagnostic['execution_fingerprint'])
    if not (selector_match and world_match and execution_match):
        raise RuntimeError('419 capture/header/stderr/prediction binding failed')

    run_prediction = json.loads((RUN / 'prediction.json').read_text())
    run_receipt = json.loads((RUN / 'receipt.json').read_text())
    run_manifest = json.loads((RUN / 'output/manifest.json').read_text())
    ref_manifest = json.loads((REF / 'output/manifest.json').read_text())
    audit = json.loads(AUDIT.read_text())
    run_pack = RUN / 'output/compiled.nmatterpack'
    ref_pack = REF / 'output/compiled.nmatterpack'
    binary = RUN / 'base-cgls.bin'
    magic, version, envs, nodes_per_env, unknown, cap, segments, reserved = struct.unpack_from('<8I', binary.read_bytes())
    run_world = f"{int(run_manifest['compiled_world_fingerprint']):016x}"
    ref_world = f"{int(ref_manifest['compiled_world_fingerprint']):016x}"
    same_pack = sha(run_pack) == sha(ref_pack)
    same_world = run_world == ref_world == header['world_fingerprint_hex']
    same_dims = (envs == int(header['environment_count']) and
                 nodes_per_env == int(header['node_count_per_environment']) and
                 int(header['tetrahedron_count_per_environment']) == int(ref_manifest['mesh']['tetrahedra']))
    same_selector = (int(header['root']) == int(audit['run_binding']['CGLS_root']) and
                     int(header['newton_iteration']) == int(audit['run_binding']['CGLS_newton_iteration']))
    if not (same_pack and same_world and same_dims and same_selector):
        raise RuntimeError('419 capture is not compatible with run450 world/pack/dimensions/selector')
    out = {
        'schema': 'numi.cardboard.native450_reference_capture_fingerprint_addendum.v1',
        'classification': 'exact source-bound 419 capture-to-stderr fingerprint verification plus run450 world/compiled-pack/dimension/selector correspondence; directional action reference only',
        'reference_419': {
            'capture_path': str(capture), 'capture_sha256': sha(capture),
            'prediction_sha256': sha(REF / 'prediction.json'), 'receipt_sha256': sha(REF / 'receipt.json'),
            'stderr_sha256': sha(REF / 'stderr.log'), 'source_snapshot': str(SNAPSHOT),
            'source_module_path': str(MODULE_PATH), 'source_module_sha256': sha(MODULE_PATH),
            'bound_input_count': len(binding['source_hashes_verified_against_prediction']) + len(binding['runtime_and_external_input_hashes_verified_against_prediction']),
            'capture_header_root_iteration': [int(header['root']), int(header['newton_iteration'])],
            'prediction_selector': binding['prediction_selector'],
            'world_fingerprint_header_hex': header['world_fingerprint_hex'],
            'world_fingerprint_stderr_hex': f"{int(diagnostic['world_fingerprint']):016x}",
            'execution_fingerprint_header_hex': header['execution_fingerprint_hex'],
            'execution_fingerprint_stderr_hex': f"{int(diagnostic['execution_fingerprint']):016x}",
            'selector_match': selector_match, 'world_fingerprint_match': world_match,
            'execution_fingerprint_match': execution_match,
            'capture_integrity': integrity,
        },
        'run_450': {
            'prediction_sha256': sha(RUN / 'prediction.json'), 'receipt_sha256': sha(RUN / 'receipt.json'),
            'stderr_sha256': sha(RUN / 'stderr.log'), 'compiled_pack_sha256': sha(run_pack),
            'compiled_world_fingerprint_hex': run_world,
            'compiled_pack_sha256_419': sha(ref_pack), 'compiled_world_fingerprint_hex_419': ref_world,
            'same_compiled_pack': same_pack, 'same_compiled_world': same_world,
            'same_environment_node_tet_dimensions': same_dims, 'same_root_newton_selector': same_selector,
            'source_inputs_unchanged': run_receipt.get('source_inputs_unchanged'),
            'exit_code': run_receipt.get('exit_code'), 'timed_out': run_receipt.get('timed_out'),
            'cgls_binary_sha256': sha(binary), 'snapshot_header': {
                'magic': f'0x{magic:08x}', 'version': version, 'environments': envs,
                'nodes_per_environment': nodes_per_env, 'unknowns': unknown,
                'iteration_cap': cap, 'segments': segments, 'reserved': reserved,
            },
        },
        'evidence_report_sha256': sha(AUDIT),
        'interpretation': [
            'The 419 capture is bound to its own source, prediction, receipt, stderr, root/Newton selector, world fingerprint and execution fingerprint.',
            'Run450 and the 419 capture share the compiled package hash and world fingerprint, and have matching dimensions and selected root/Newton.',
            'Run450 itself emitted a rich-capture skip for environment_count_out_of_scope; this addendum does not claim a full run450 operator capture or exact internal state-history identity.',
            'The 419 operator capture supports selected-vector directional A*x and A-transpose*r comparisons only, not whole-matrix identity or physical qualification.'
        ],
    }
    out_path = Path(__file__).with_name('addendum.json')
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'result':'VERIFIED','output':str(out_path),'world_match':same_world,
                      'execution_fingerprint_match':execution_match,'same_pack':same_pack,
                      'same_dimensions':same_dims,'selector_match':same_selector},indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
