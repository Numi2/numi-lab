#!/usr/bin/env python3
import copy
from strict_native_trace_audit import check_rows, finite_tree

row = {"step": 0, "accepted": True, "status": 0, "outer_status": 0,
       "certificate_accepted": 1, "gap": .0002, "mass": .0088,
       "force": .1, "accepted_normal_impulse": .00001, "normal_impulse": .00001}
check_rows([row], .0001)
for field, value in (("force", float("nan")), ("gap", float("inf")),
                     ("accepted_normal_impulse", .1), ("outer_status", 10)):
    damaged = copy.deepcopy(row)
    damaged[field] = value
    try:
        check_rows([damaged], .0001)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted corrupted " + field)
assert not finite_tree({"diagnostic": [0., float("nan")]})
print("strict native trace corruption checks passed")
