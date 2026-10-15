#!/usr/bin/env python3
"""Independent descriptor-ledger contract for summary overlay 002."""
from __future__ import annotations
import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
BASE1 = Path('/Users/home/cardboard-implementation-20261014/analysis/armijo-all-trial-summary-overlay-001/overlay/matter/src/runtime.mm')
CANDIDATE = PACKAGE / 'overlay/matter/src/runtime.mm'
SHA_BASE1 = 'e671e3e7d903cb7ddf01b53d80d03a91579857c2757e8d8586144e06cb62f8c3'
SHA_CANDIDATE = 'feaf5a062138e8dc1a84994090fde27a59a2605e863bb0d5c8ed1280e07b7fc4'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(test: bool, message: str) -> None:
    if not test:
        raise AssertionError(message)


def checked_scratch_bytes(logical_bytes: int, element_bytes: int) -> int:
    # runtime.mm checkedBytes() contract: retain at least one complete typed
    # element and a 16-byte sentinel for a short logical range.
    return max(logical_bytes, element_bytes, 16)


def descriptor_ledger(envs: int, objects: int, budget: int) -> tuple[int, int]:
    """Sum emitted descriptor extents independently, by capture phase."""
    status_record = 8 * 4 + 4 * 4
    float4 = 4 * 4
    scalar = 4
    stages_per_trial = 8
    alpha_extent = checked_scratch_bytes(envs * scalar, scalar)
    fem_line_search_extent = checked_scratch_bytes(envs * objects * float4, float4)

    # Fixed descriptors: entry status; final global/FEM statuses; final
    # metrics; final FEM line-search array; final environment-alpha buffer.
    fixed = (
        envs * status_record +
        2 * envs * status_record +
        envs * float4 +
        fem_line_search_extent +
        alpha_extent
    )
    # Packed status arrays contain exactly eight stage records and one global
    # postdecision record per trial/environment.
    packed_status = budget * envs * (
        stages_per_trial * status_record + status_record
    )
    per_trial_descriptors = [
        ('metrics_loop_entry', envs * float4),
        ('metrics_predecision', envs * float4),
        ('metrics_postdecision', envs * float4),
        ('fem_line_search_postdecision', fem_line_search_extent),
        ('environment_line_search_postdecision', alpha_extent),
    ]
    per_trial_data = sum(extent for _, extent in per_trial_descriptors)
    descriptor_count = 8 + len(per_trial_descriptors) * budget
    total = fixed + packed_status + budget * per_trial_data
    return total, descriptor_count


def main() -> None:
    require(digest(BASE1) == SHA_BASE1, 'overlay-001 base runtime hash changed')
    require(digest(CANDIDATE) == SHA_CANDIDATE, 'overlay-002 candidate runtime hash changed')
    old = BASE1.read_text()
    new = CANDIDATE.read_text()
    require('!checkedMultiply(expectedMetricBytes, 3u,' in old,
            'expected original undercount was not present in overlay 001')
    require('!checkedMultiply(expectedMetricBytes, 3u,' in new,
            'three per-object metric descriptors should remain separately counted')
    require('objectLineSearchBytes' in new and
            'objectLineSearchLogicalBytes' in new,
            'object-sized line-search descriptor extent is not accounted')
    require('expectedEnvironmentAlphaLogicalBytes' in new and
            'expectedEnvironmentAlphaBytes = std::max<std::uint64_t>' in new,
            'typed sentinel tail of environment alpha buffer is not accounted')
    require('summary_source_extent_does_not_match_checked_scratch_layout' in new,
            'source extents are not checked against the real scratch allocation')

    # Bind the independent ledger to actual emitted descriptor names and ABI.
    for token in (
        '"armijo_trial_status_checkpoints"',
        '"armijo_trial_global_status_after_decision"',
        '"armijo_entry_global_status"',
        '"armijo_final_global_status"',
        '"armijo_final_fem_status"',
        '"armijo_final_metrics"',
        '"armijo_final_fem_line_search"',
        '"armijo_final_environment_line_search"',
        '"_metrics_loop_entry"',
        '"_metrics_predecision"',
        '"_metrics_postdecision"',
        '"_fem_line_search_postdecision"',
        '"_environment_line_search_postdecision"',
        'candidate->femLineSearch = privateScratch<nm_float4>',
        'multiplied(world.dispatch.objectCount)',
        'candidate->environmentLineSearch = privateScratch<float>',
        'return static_cast<NSUInteger>(std::max({',
        'std::size_t{16u}',
    ):
        require(token in new, f'missing descriptor/allocation contract {token}')

    for envs in (1, 2, 4):
        for objects in (1, 2, 4):
            for budget in (1, 16, 24, 32):
                total, arrays = descriptor_ledger(envs, objects, budget)
                alpha = max(envs * 4, 16)
                line = max(envs * objects * 16, 16)
                exact_fixed = 176 * envs + 16 * envs * (objects - 1) + alpha
                exact_per_trial = 496 * envs + 16 * envs * (objects - 1) + alpha
                require(total == exact_fixed + budget * exact_per_trial,
                        f'extent does not equal independently derived form E={envs}, O={objects}, B={budget}')
                require(arrays == 8 + 5 * budget,
                        f'descriptor count mismatch E={envs}, O={objects}, B={budget}')
                require(total <= 32 * 1024 * 1024, '32 MiB cap exceeded')
                require(arrays <= 192, 'descriptor cap exceeded')
    require(descriptor_ledger(1, 1, 24)[0] == 12480,
            'E=1/O=1/B=24 capture extent changed')
    require(descriptor_ledger(4, 1, 32) == (64720, 168),
            'E=4/O=1/B=32 layout changed')
    print('PASS independent descriptor ledger: exact buffer extents, object-sized FEM line-search, checkedBytes sentinel tails; environments 1/2/4, objects 1/2/4, budgets 1/16/24/32; 32 MiB/192-array caps')


if __name__ == '__main__':
    main()
