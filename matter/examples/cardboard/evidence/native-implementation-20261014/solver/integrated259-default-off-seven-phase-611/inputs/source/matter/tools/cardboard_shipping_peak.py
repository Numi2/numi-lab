from __future__ import annotations
import math
DEFAULT_REQUIRED_DROP_FRACTION = 0.01
DEFAULT_REQUIRED_CONSECUTIVE_DECLINES = 2

def first_local_peak_index(strokes_m, forces_n, contact_samples, *,
                           required_drop_fraction=DEFAULT_REQUIRED_DROP_FRACTION,
                           required_consecutive_declines=DEFAULT_REQUIRED_CONSECUTIVE_DECLINES):
    strokes, forces, contacts = list(strokes_m), list(forces_n), list(contact_samples)
    if not (len(strokes) == len(forces) == len(contacts)):
        raise ValueError("compression peak inputs must have equal lengths")
    if not (math.isfinite(required_drop_fraction) and 0.0 < required_drop_fraction < 1.0):
        raise ValueError("required peak drop must be finite and between zero and one")
    if isinstance(required_consecutive_declines, bool) or required_consecutive_declines < 2:
        raise ValueError("peak detector requires at least two consecutive declines")
    previous = previous_previous = candidate = None
    have_previous = have_previous_previous = False
    declines = 0
    for index, (stroke, force, contact) in enumerate(zip(strokes, forces, contacts)):
        try:
            stroke, force, contact = float(stroke), float(force), float(contact)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"compression peak row {index} is nonnumeric") from exc
        if not all(map(math.isfinite, (stroke, force, contact))):
            raise ValueError(f"compression peak row {index} is nonfinite")
        if min(stroke, force, contact) < 0.0:
            raise ValueError(f"compression peak row {index} is negative")
        if contact <= 0.0 or force <= 0.0:
            previous = previous_previous = candidate = None
            have_previous = have_previous_previous = False
            declines = 0
            continue
        current = (index, stroke, force, contact)
        if candidate is not None:
            if force > candidate[2]:
                candidate = current
                declines = 0
            elif previous is not None and force < previous[2]:
                declines += 1
            else:
                declines = 0
            if declines >= required_consecutive_declines and 1.0 - force / candidate[2] >= required_drop_fraction:
                return candidate[0]
        elif have_previous_previous and have_previous and previous[2] > previous_previous[2] and force < previous[2]:
            candidate = previous
            declines = 1
        previous_previous = previous
        have_previous_previous = have_previous
        previous = current
        have_previous = True
    return None
