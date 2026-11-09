"""Summarize three paired, alternating, complete physical cost rollouts."""
import math
import statistics


def summarize(records):
    expected = [("ipc_area", 1), ("ipc_area_midpoint", 1),
                ("ipc_area_midpoint", 2), ("ipc_area", 2),
                ("ipc_area", 3), ("ipc_area_midpoint", 3)]
    actual = [(record["arm"], record["repetition"]) for record in records]
    if actual != expected[:len(actual)]:
        raise ValueError("cost order differs from the declared alternating schedule")
    pairs = {}
    for record in records:
        arm, repetition = record["arm"], record["repetition"]
        if arm not in ("ipc_area", "ipc_area_midpoint") or repetition not in (1, 2, 3):
            raise ValueError("unexpected cost arm or repetition")
        pair = pairs.setdefault(repetition, {})
        if arm in pair:
            raise ValueError("duplicate cost run")
        rate = record.get("accepted_seconds_per_wall_second")
        if not record.get("accepted_and_matches_accuracy_state") or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            return {"complete": False, "practical_benefit": False,
                    "reason": "failed or invalid cost rollout"}
        pair[arm] = rate
    if len(records) != 6 or set(pairs) != {1, 2, 3} or any(len(pairs[k]) != 2 for k in pairs):
        return {"complete": False, "practical_benefit": False,
                "reason": "three complete pairs required"}
    be = [pairs[i]["ipc_area"] for i in (1, 2, 3)]
    mid = [pairs[i]["ipc_area_midpoint"] for i in (1, 2, 3)]
    ratio = statistics.median(mid) / statistics.median(be)
    all_pairs = all(m > c for m, c in zip(mid, be))
    return {"complete": True, "backward_euler_rates": be, "midpoint_rates": mid,
            "median_backward_euler_rate": statistics.median(be),
            "median_midpoint_rate": statistics.median(mid), "median_rate_ratio": ratio,
            "every_paired_midpoint_faster": all_pairs,
            "practical_benefit": ratio >= 1.1 and all_pairs,
            "rate_units": "accepted simulated seconds per wall second"}
