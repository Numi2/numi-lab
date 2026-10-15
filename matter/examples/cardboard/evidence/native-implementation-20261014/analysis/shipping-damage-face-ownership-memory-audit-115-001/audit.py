#!/usr/bin/env python3
"""Read-only source audit of the crack-band face-ownership workspace."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, re, subprocess
from pathlib import Path

DEFAULT_ROOT = Path('/Users/home/numi-cardboard-20261006')
FILES = [
    'matter/tools/cardboard_damage_builder.hpp',
    'matter/tools/cardboard_retained_matrix_damage.hpp',
    'matter/tools/cardboard_probe.mm',
    'matter/tools/cardboard_fracture_regularization.hpp',
    'matter/tools/cardboard_shipping_memory.hpp',
    'matter/tools/cardboard_shipping_allocation_preflight.cpp',
    'matter/tools/cardboard_shipping_allocation_preflight.py',
]
class FaceRecord(ctypes.Structure):
    _fields_ = [('key', ctypes.c_uint64), ('requestIndex', ctypes.c_size_t),
                ('faceIndex', ctypes.c_size_t), ('selectedModeMask', ctypes.c_uint8)]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def line_of(text: str, token: str) -> int:
    return text[:text.index(token)].count('\n') + 1

def audit(root: Path, requests: int) -> dict:
    src = {name: (root / name).read_text(encoding='utf-8') for name in FILES}
    damage = src[FILES[0]]
    retained = src[FILES[1]]
    probe = src[FILES[2]]
    fracture = src[FILES[3]]
    memory = src[FILES[4]]
    guard_calls = [m.start() for m in re.finditer(r'validateNoDuplicateSharedFaceCrackBandArea\s*\(', damage)]
    shipping_call = 'buildMeshProgramPool(uniqueRequests)' in probe
    retained_pool_start = retained.index('inline MeshProgramPool buildMeshProgramPool(')
    retained_pool_body = retained[retained_pool_start: retained.index('\n}\n', retained_pool_start) + 3]
    probe_builder_calls = [name for name in ('buildMeshDamageProgramPool(', 'buildMeshDamagePrograms(') if name in probe]
    required_fragments = [
        ('probe uses retained pool', shipping_call),
        ('probe does not call generic per-element crack-band pool', not probe_builder_calls),
        ('retained pool does not call generic ownership guard', 'validateNoDuplicateSharedFaceCrackBandArea(' not in retained),
        ('retained builder uses no generic damage pool', 'buildMeshDamageProgramPool(' not in retained and 'buildMeshDamagePrograms(' not in retained),
        ('generic owner guard reserves four face records per request', 'records.reserve(requests.size() * 4u)' in damage),
        ('retained damage plane offset is exactly zero', '0.0, materialToReference, 1.0e-12' in probe),
        ('geometry rejects degenerate tetrahedra', 'volume > degeneracyFloor' in fracture),
        ('generic owner check used only by generic pool/program builders', len(guard_calls) == 3),
        ('shipping builder input is uniqueRequests', shipping_call),
    ]
    missing = [name for name, passed in required_fragments if not passed]
    if missing:
        raise RuntimeError('source contract changed: ' + ', '.join(missing))
    record_bytes = ctypes.sizeof(FaceRecord)
    count = requests * 4
    bytes_ = count * record_bytes
    # The generic validator has one O(1)-per-request local geometry array, not
    # a retained second mesh-wide geometry array. Four vertices maximum per cut.
    return {
        'schema': 'numi.cardboard.damage_face_ownership_memory_audit.v1',
        'root': str(root),
        'git_head': subprocess.run(['git','rev-parse','HEAD'], cwd=root, check=True, capture_output=True, text=True).stdout.strip(),
        'generic_guard': {
            'source': FILES[0],
            'definition_line': line_of(damage, 'inline void validateNoDuplicateSharedFaceCrackBandArea'),
            'pool_call_lines': [damage[:i].count('\n') + 1 for i in guard_calls[1:]],
            'record_fields': ['uint64_t key','size_t requestIndex','size_t faceIndex','uint8_t selectedModeMask'],
            'record_size_bytes_current_host_abi': record_bytes,
            'record_size_basis': 'ctypes layout of the exact source field order/types on this 64-bit host; standard layout rounds 25 field bytes to 32-byte alignment',
            'records_per_request': 4,
            'estimated_workspace_formula': '4 * request_count * sizeof(FaceRecord)',
            'hypothetical_request_count': requests,
            'hypothetical_record_count': count,
            'hypothetical_record_vector_bytes': bytes_,
            'hypothetical_record_vector_decimal_MB': bytes_ / 1e6,
            'hypothetical_record_vector_MiB': bytes_ / (1024**2),
            'workspace_lifetime': 'host std::vector local to validateNoDuplicateSharedFaceCrackBandArea; reserved once, filled, sorted in place, destroyed when validation returns; no second mesh-wide record array',
            'other_local_workspace': 'four TetCrackPlaneGeometry values and their cut-polygon vectors for the currently visited request only; O(1) request-local workspace plus allocator overhead',
            'memory_domain': 'CPU host heap, not a Metal buffer or runtime resident arena',
        },
        'shipping_retained_damage_path': {
            'source': FILES[2],
            'apply_overlay_line': line_of(probe, 'void applyRetainedMatrixDamageOverlay('),
            'pool_call_line': line_of(probe, 'buildMeshProgramPool(uniqueRequests)'),
            'pool_source': FILES[1],
            'uses_generic_face_guard': False,
            'request_dedup_key': 'sourceMaterialIndex plus four double modeLengths',
            'face_guard_temporary_allocated': False,
            'selected_planes': 'four material-frame normals; each offsetFromCentroidM is 0.0',
            'degeneracy': 'computeTetCrackPlaneGeometry rejects zero/degenerate reference tetrahedra',
            'geometric_reason': 'for a nondegenerate tetrahedron, the centroid is strictly interior. A plane through that centroid cannot be the supporting plane of a boundary face; distinct tet interiors share at most a lower-dimensional boundary. Thus the generic guard’s exact full-face duplicate case is excluded by this authored zero-offset construction, subject to the implementation tolerance and the normal mesh-quality checks.',
            'retained_builder_other_host_work': 'uniqueRequests, requestByTetrahedron, modeLengthsByTetrahedron, geometryToRequest and pool mapping/program vectors are host-side builder temporaries/persistent output; they are separate from FaceRecord and only coarsely represented by the prebuild host reserve.',
        },
        'prebuild_memory_model': {
            'source': FILES[4],
            'prebuild_source_tet_formula': '512 * tetrahedra',
            'prebuild_compiled_tet_formula': '256 * tetrahedra',
            'ownership_workspace_in_current_shipping_overlay': 0,
            'reason': 'current shipping path does not call the generic builder/guard',
            'not_a_measured_peak': True,
            'future_generic_builder_requirement': 'if shipping is changed to invoke the generic builder with R requests, add 4*R*sizeof(FaceRecord) to the host pre-mesh transient estimate, with checked uint64 arithmetic; do not add it to GPU resident bytes. The allocation occurs before compileWorld and Runtime::initialize.',
        },
        'contract_checks': [{'check': name, 'passed': passed} for name, passed in required_fragments],
        'source_sha256': {name: sha(root / name) for name in FILES},
        'conclusion': 'No shipping-prebuild correction is warranted for the new generic FaceRecord vector: it is not allocated by the live retained-matrix shipping damage builder. Its hypothetical generic-builder workspace is host-only and should be a separate transient input if that call path is ever adopted.',
    }

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--requests', type=int, default=1477728, help='hypothetical request count; defaults to the reference shipping tetrahedron count, not a measured retained-builder request count')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.requests < 0:
        raise SystemExit('requests must be nonnegative')
    result = audit(a.root.resolve(), a.requests)
    data = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if a.output:
        a.output.write_text(data, encoding='utf-8')
    else:
        print(data, end='')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
