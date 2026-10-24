"""Persistent resolved-contact guards for iterative geometry construction.

A guard is retained even if its contact recurs. These sets are construction
history, not an assertion that a candidate passed an anatomical collision gate.
"""
def accumulate_resolved_contacts(resolved, previous_pairs, current_pairs):
    """Return independent sets; never forget an earlier resolved witness."""
    if set(resolved) != set(previous_pairs) or set(resolved) != set(current_pairs):
        raise ValueError("pose domains differ")
    result = {}
    for pose in resolved:
        result[pose] = set(resolved[pose]) | (set(previous_pairs[pose]) - set(current_pairs[pose]))
    return result
