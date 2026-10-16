#!/usr/bin/env python3
"""Compile and run the shared.h sizeof probe without dispatching Metal."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
BASE = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-checkpoint-warm-breakdown-173')
SOURCE = HERE / 'shipping_arena_size_probe.cpp'
OUTPUT = HERE / 'shipping-arena-size-probe'
INCLUDE = BASE / 'matter/include'
ARGV = [
    '/usr/bin/clang++', '-std=c++20', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
    '-I', str(INCLUDE), str(SOURCE), '-o', str(OUTPUT),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


compile_result = subprocess.run(ARGV, capture_output=True, text=True)
(HERE / 'shipping_arena_size_probe_compile.stdout').write_text(compile_result.stdout)
(HERE / 'shipping_arena_size_probe_compile.stderr').write_text(compile_result.stderr)
if compile_result.returncode:
    raise SystemExit(compile_result.returncode)
run_result = subprocess.run([str(OUTPUT)], capture_output=True, text=True)
(HERE / 'shipping_arena_size_probe_run.stdout').write_text(run_result.stdout)
(HERE / 'shipping_arena_size_probe_run.stderr').write_text(run_result.stderr)
if run_result.returncode:
    raise SystemExit(run_result.returncode)
sizes = {}
for line in run_result.stdout.splitlines():
    name, value = line.split('=', 1)
    sizes[name] = int(value)
expected = {
    'NMFEMNodeStateGPU': 80,
    'NMTetrahedronGPU': 96,
    'NMCohesiveFaceGPU': 384,
    'NMCohesiveFaceVectorGPU': 96,
    'NMContactPairGPU': 16,
    'NMContactSampleGPU': 160,
    'NMContinuumSurfacePrimitiveGPU': 64,
    'NMDeformableContactCandidateGPU': 16,
    'NMDeformableContactGPU': 432,
    'NMIncidenceRangeGPU': 16,
    'NMIndirectDispatchGPU': 16,
}
if sizes != expected:
    raise SystemExit(f'size probe mismatch: {sizes!r}')
report = {
    'schema': 'numi.cardboard.source175-size-probe-validation.v1',
    'gpu_dispatch': False,
    'compile_argv': ARGV,
    'compile_exit_code': compile_result.returncode,
    'compile_stderr_sha256': sha(HERE / 'shipping_arena_size_probe_compile.stderr'),
    'probe_source_sha256': sha(SOURCE),
    'probe_binary_sha256': sha(OUTPUT),
    'run_exit_code': run_result.returncode,
    'run_stdout_sha256': sha(HERE / 'shipping_arena_size_probe_run.stdout'),
    'sizeof_bytes': sizes,
}
(HERE / 'shipping_arena_size_probe_validation.json').write_text(
    json.dumps(report, sort_keys=True, indent=2) + '\n'
)
print(json.dumps({'compile': 'PASS', 'run': 'PASS', 'gpu_dispatch': False}, sort_keys=True))
