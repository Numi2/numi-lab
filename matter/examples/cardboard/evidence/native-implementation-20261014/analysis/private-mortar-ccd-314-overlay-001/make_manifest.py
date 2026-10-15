from __future__ import annotations
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
base = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-contact-capture-and-resolved-material-sweep-095')
prior = Path('/Users/home/cardboard-implementation-20261014/analysis/mortar-union-four-patch-overlay-005')
run = Path('/Users/home/cardboard-implementation-20261014/native/exhaustive-private-mortar-ccd-314')
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()
def record(path: Path, rel: str | None = None) -> dict:
    return {'path': rel if rel is not None else str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
files = [
    'private-mortar-ccd-314.patch', 'README.md',
    'build-source/matter/CMakeLists.txt',
    'build-source/matter/src/metal/contact.metalinc',
    'build-source/matter/tools/cardboard_full_contact_armijo_check.mm',
    'build-source/matter/include/numi/matter/mortar_shared_edge_guard.h',
    'mortar_shared_edge_guard_check.cpp', 'mortar_shared_edge_guard_check',
    'cpu-check-build-output.txt', 'cpu-check-output.txt',
    'fixture_geometry_check.py', 'fixture-geometry-check.json',
    'build-host-output.txt', 'build-host-clean-output.txt',
    'build-host/matter/shaders/NumiMatter.metallib',
    'build-host/matter/numi-matter-cardboard-full-contact-armijo-check',
]
manifest = {
    'schema': 'numi.cardboard.private_mortar_ccd_overlay.v1',
    'overlay': 'private-mortar-ccd-314-overlay-001',
    'classification': 'external compile-and-CPU-check overlay; native GPU execution not run',
    'base': {
        'source_snapshot': str(base),
        'frozen_source_manifest': record(base/'frozen-source.json'),
        'parent_revision': json.loads((base/'frozen-source.json').read_text()).get('parent_revision'),
        'exact_shared_edge_overlay_005_manifest': record(prior/'manifest.json'),
        'exact_shared_edge_overlay_005_patch': record(prior/'exact-shared-edge-guard-005.patch'),
    },
    'original_native_314': {
        'directory': str(run),
        'source_audit_manifest': record(Path('/Users/home/cardboard-implementation-20261014/analysis/private-mortar-all-cases-314-001/manifest.json')),
        'original_report': record(Path('/Users/home/cardboard-implementation-20261014/analysis/private-mortar-all-cases-314-001/report.json')),
        'stdout': record(run/'stdout.log'),
        'stderr': record(run/'stderr.log'),
        'receipt': record(run/'receipt.json'),
        'verdict': '9 of 14 expected gates passed; five expected-accept cases rejected. This overlay does not overwrite or reinterpret the run.',
    },
    'changes': {
        'paired_map_identity': 'Require exact FP32 xyz equality of both left and right reference-map endpoints; do not accept 8e-6 near matches as seam identity.',
        'witness_location': 'Existing 8e-6 barycentric feature-location tolerance is unchanged and explicitly targeted by a zero-motion off-edge initial-overlap native negative control; that control has not been run.',
        'overlap_threshold': 1.0e-10,
        'clip_tolerance': 1.0e-7,
        'partition_diagnostic_codes': {'left': '1..13', 'right': '101..113', 'sweep': 200},
        'fixture_additions': ['positive-gap partial uncovered crossing with active-patch clearance above contact thickness', 'zero-motion 4 nm off-edge initial cross-patch overlap'],
    },
    'checks': {
        'host_shader_and_fixture_compile': 'PASS; clean CMake target build regenerated NumiMatter.metallib and linked numi-matter-cardboard-full-contact-armijo-check; existing unrelated Metal warnings remain; no GPU dispatch',
        'shared_edge_cpu_checker': 'PASS; 14 printed checks',
        'fixture_geometry_arithmetic': 'PASS; positive-gap, covered-complement clearance and off-edge tolerance relations checked',
        'patch_dry_run_against_base': 'PASS with patch -p1',
        'native_gpu_execution': 'NOT RUN',
        'physical_qualification': False,
    },
    'commands': {
        'cmake_build': 'cmake --build build-host --target numi-matter-cardboard-full-contact-armijo-check -j 4',
        'cpu_compile': 'clang++ -std=c++20 -O2 -ffp-contract=fast -Wall -Wextra -Wpedantic -Werror -I build-source/matter/include mortar_shared_edge_guard_check.cpp -o mortar_shared_edge_guard_check',
        'cpu_run': './mortar_shared_edge_guard_check',
        'geometry_run': '/Users/home/.pyenv/versions/3.10.12/bin/python3.10 fixture_geometry_check.py',
    },
    'files': {f: record(root/f, f) for f in files},
    'limitations': [
        'No native result exists for the new zero-translation off-edge initial-overlap case or positive-gap complement crossing.',
        'The 8e-6 witness-location tolerance remains; whether its numerical allowance can hide an initial off-edge collision is unresolved pending the native negative control.',
        'The four-patch rejection in run 314 remains unexplained; stage codes only improve the next diagnostic.',
        'The original 314 five expected-accept rejections remain preserved and unqualified.',
        'This is a synthetic private-kernel fixture and not a shipping-board or physical validation result.',
    ],
}
(root/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
