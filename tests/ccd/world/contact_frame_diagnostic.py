#!/usr/bin/env python3
"""Independent rigid contact frame checks for NUMI_CCD_EVENT_TRACE output.

The trace adapter below is intentionally restricted to the single-free-body,
static-plane fixture. The functions accept arbitrary moving rigid endpoints.
An evaluated velocity residual is a diagnostic, not an acceptance rule.
"""
import argparse
import json
import math


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def rotate(q, v):
    x, y, z, w = q
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid orientation")
    u = (x/norm, y/norm, z/norm)
    t = tuple(2*x for x in cross(u, v))
    return add(v, add(tuple((w/norm)*x for x in t), cross(u, t)))


def contact_frame(position_a, quaternion_a, linear_a, angular_a, anchor_a,
                  position_b, quaternion_b, linear_b, angular_b, anchor_b, normal):
    ra, rb = rotate(quaternion_a, anchor_a), rotate(quaternion_b, anchor_b)
    point_a, point_b = add(position_a, ra), add(position_b, rb)
    velocity_a = add(linear_a, cross(angular_a, ra))
    velocity_b = add(linear_b, cross(angular_b, rb))
    return {
        "point_a": point_a,
        "point_b": point_b,
        "midpoint": tuple((a+b)/2 for a, b in zip(point_a, point_b)),
        "geometric_gap": dot(sub(point_b, point_a), normal),
        "relative_vn": dot(sub(velocity_b, velocity_a), normal),
    }


def fixture_result(event, contact):
    # Adapter is intentionally limited to the probe's body 0 static ground,
    # body 1 free box. Never infer this mapping for another scene.
    if (contact["body_a"], contact["body_b"]) != (0, 1):
        raise ValueError("unexpected body order: fixture adapter is not general")
    q, v = event["q"], event["v"]
    result = contact_frame(
        (0, 0, 0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0), contact["anchor_a"],
        q[:3], q[3:7], v[:3], v[3:6], contact["anchor_b"], contact["normal"])
    result["midpoint_error"] = math.sqrt(sum(x*x for x in sub(result["midpoint"], contact["point"])))
    result["gap_difference"] = result["geometric_gap"] - contact["gap"]
    result["target_minus_outgoing_vn"] = contact["target_vn"] - result["relative_vn"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--fixture", type=int)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.)
    args = parser.parse_args()
    case = None
    with open(args.trace, encoding="utf-8") as trace:
        for line in trace:
            if line.startswith("ccd_case="):
                case = json.loads(line.partition("=")[2])
            elif line.startswith("ccd_event_trace=") and case:
                if (case["limit"] != args.limit or case["mode"] != args.mode
                    or case["scale"] != args.scale
                    or (args.fixture is not None and case["fixture"] != args.fixture)):
                    continue
                event = json.loads(line.partition("=")[2])
                # On rejection contact scratch may contain the previous event.
                if event["status"] != 0:
                    continue
                for i, contact in enumerate(event["contacts"]):
                    if "target_vn" not in contact:
                        raise ValueError("trace lacks evaluated row; rebuild instrument")
                    result = fixture_result(event, contact)
                    print(json.dumps({"case": case, "pass": event["pass"],
                                      "contact": i, "result": result}, sort_keys=True))

if __name__ == "__main__":
    main()
