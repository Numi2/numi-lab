#!/usr/bin/env python3
"""CPU parity check for flattened indirect candidate mapping, env counts 1..4."""
from itertools import product


def scan_counts(counts, capacity):
    offsets = []
    total = 0
    for count in counts:
        count = min(count, capacity)
        offsets.append(total)
        total += count
    return offsets, total


def compact_domain(counts, capacity):
    offsets, total = scan_counts(counts, capacity)
    result = []
    for q in range(total):
        match = None
        for env, count in enumerate(counts):
            count = min(count, capacity)
            if offsets[env] <= q < offsets[env] + count:
                slot = q - offsets[env]
                match = (env, slot, env * capacity + slot)
                break
        assert match is not None
        result.append(match)
    return offsets, total, result


def fixed_capacity_domain(counts, capacity):
    return [
        (env, slot, env * capacity + slot)
        for env, count in enumerate(counts)
        for slot in range(min(count, capacity))
    ]


def check_dispatch(counts, capacity):
    offsets, total, compact = compact_domain(counts, capacity)
    fixed = fixed_capacity_domain(counts, capacity)
    assert compact == fixed, (counts, capacity, compact, fixed)
    groups = (total + 255) // 256
    launched = groups * 256
    mapped = []
    for q in range(launched):
        found = None
        for env, count in enumerate(counts):
            count = min(count, capacity)
            if offsets[env] <= q < offsets[env] + count:
                slot = q - offsets[env]
                found = (env, slot, env * capacity + slot)
                break
        if found is not None:
            mapped.append(found)
    assert mapped == fixed, (counts, capacity, mapped, fixed)
    assert len({row[2] for row in mapped}) == total
    if all(min(count, capacity) == capacity for count in counts):
        # The old capacity-sized dispatch and compact dispatch visit the same
        # padded candidate records in the same flattened order at full counts.
        assert [row[2] for row in mapped] == list(range(len(counts) * capacity))
    return offsets, total, groups


def main():
    capacity = 7
    cases = 0
    for env_count in range(1, 5):
        for counts in product(range(capacity + 1), repeat=env_count):
            check_dispatch(counts, capacity)
            cases += 1
        check_dispatch((capacity + 3,) * env_count, capacity)  # overflow is clamped
        check_dispatch((0,) * env_count, capacity)
        cases += 2
    assert check_dispatch((7, 0, 2, 5), capacity)[:2] == ([0, 7, 7, 9], 14)
    assert check_dispatch((7, 7, 7, 7), capacity)[1] == 28
    assert check_dispatch((0, 0, 0, 0), capacity)[2] == 0
    print(f"PASS environments=1..4 exhaustive_count_vectors={cases} capacity={capacity}")


if __name__ == "__main__":
    main()
