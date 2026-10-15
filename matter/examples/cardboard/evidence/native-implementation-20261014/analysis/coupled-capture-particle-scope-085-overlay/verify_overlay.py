#!/usr/bin/env python3
"""Verify the isolated coupled-capture scope patch against frozen source 085."""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path('/Users/home/cardboard-implementation-20261014')
BASE = ROOT / 'snapshots/source-accepted-correction-regional-recycle-085/matter/src/runtime.mm'
PATCHED = Path(__file__).with_name('runtime.mm')
EXPECTED_BASE = 'd482a43db6af754b9578f78ac7a814267b104c33ff2750bfe14101b9ec301919'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

base = BASE.read_text()
source = PATCHED.read_text()
require(sha(BASE) == EXPECTED_BASE, 'frozen source-085 runtime hash changed')
require(source != base, 'overlay has no changes')
require(source.count('femCoupledCaptureScopeFailure() == nullptr') == 1,
        'Newton RHS-preservation selection does not share the scoped predicate')
require(source.count('if (const char* scopeFailure = femCoupledCaptureScopeFailure())') == 1,
        'readback callback does not use the same fail-closed scope predicate')
require('state.dispatch.particleCount != 0u &&' not in source,
        'an unconditional nonzero-particle rejection remains')
require('state.dispatch.particleCount == 0u &&' not in source,
        'the Newton selection still rejects all particle records')
for token in (
    'state.dispatch.gridNodeCount != 0u',
    'state.dispatch.mpmGridCount != 0u',
    'state.dispatch.mpmBlockCount != 0u',
    'state.dispatch.mpmBlockLookupCount != 0u',
):
    require(token in source, f'missing fail-closed MPM guard: {token}')
require('state.dispatch.materialStateStride > NM_MAX_MATERIAL_STATE' in source,
        'material history stride is not bounded by the validated material-state limit')
require('state.dispatch.materialStateStride > 64u' not in source,
        'obsolete arbitrary stride cap remains')
require('envTets * state.dispatch.materialStateStride * sizeof(float)' in source,
        'FEM history extent is not based on tetrahedra')
for name in (
    'particle_defaults', 'particles_accepted', 'particles_candidate',
    'particle_checkpoint', 'particle_material_state_defaults',
    'particle_material_state_accepted', 'particle_material_state_candidate',
    'particle_material_state_checkpoint', 'mpm_active_node_counts',
    'mpm_active_dispatch', 'mpm_particle_keys',
    'mpm_sorted_particle_indices', 'mpm_environment_active_counts',
):
    require(source.count(f'appendBuffer("{name}"') == 1,
            f'capture does not append exactly one {name} buffer')
require('captureArrayCap = 192u' in source, 'array descriptor ceiling changed')
require('32u * 1024u * 1024u' in source, 'aggregate payload cap changed')
require('mechanics_scope' in source and 'no_grid_or_block_topology' in source,
        'manifest does not state the restricted MPM scope')
require('material_history_exceeds_capture_byte_bound' in source,
        'history-byte bound has no specific skip reason')

# The source-084 receipt establishes this narrow fixture shape; it is not a
# candidate-state capture and is used only to test the scope arithmetic.
def scope(env: int, nodes: int, tets: int, particles: int,
          grid_nodes: int, grids: int, blocks: int, lookup: int,
          stride: int) -> tuple[bool, int]:
    if not (env == 1 and nodes <= 2048 and tets <= 8192 and particles <= 64):
        return False, 0
    if grid_nodes or grids or blocks or lookup or stride > 32:
        return False, 0
    # Two FEM history arenas and four particle history arenas (default,
    # accepted, candidate, checkpoint), checked against the 32 MiB cap.
    elements = (2 * env * tets + 4 * env * particles) * stride
    return elements * 4 <= 32 * 1024 * 1024, elements * 4

ok, history_bytes = scope(1, 36, 48, 1, 0, 0, 0, 0, 0)
require(ok and history_bytes == 0,
        'one inert particle record with no MPM grid should enter bounded capture scope')
for bad in (
    (1, 36, 48, 1, 1, 0, 0, 0, 0),
    (1, 36, 48, 1, 0, 1, 0, 0, 0),
    (1, 36, 48, 1, 0, 0, 1, 0, 0),
    (1, 36, 48, 1, 0, 0, 0, 1, 0),
    (1, 36, 48, 65, 0, 0, 0, 0, 0),
    (1, 36, 48, 1, 0, 0, 0, 0, 33),
):
    require(not scope(*bad)[0], f'out-of-scope fixture entered capture: {bad}')
print('coupled particle capture overlay checks passed')
print('base_sha256=' + EXPECTED_BASE)
print('patched_sha256=' + sha(PATCHED))
