#!/usr/bin/env python3
"""Attribute a bounded native Human capture; never replay or advance physics."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct

def analyze(path):
    manifest = json.loads(path.read_text())
    if manifest.get("capture_profile") != "rhs_attribution_subset_v1":
        raise ValueError("requires the explicitly non-replayable RHS subset")
    payload = manifest["payload"]
    data = Path(payload["file"]).read_bytes()
    if len(data) != payload["byte_count"] or hashlib.sha256(data).hexdigest() != payload["sha256"]:
        raise ValueError("payload identity mismatch")
    arrays = {}
    intervals = []
    for a in manifest["arrays"]:
        name = a["name"]
        if name in arrays:
            raise ValueError("duplicate array: " + name)
        start, size = a["offset"], a["byte_count"]
        if min(start, size) < 0 or start + size > len(data):
            raise ValueError("array bounds: " + name)
        if size != a["count"] * a["element_bytes"]:
            raise ValueError("array extent: " + name)
        raw = memoryview(data)[start:start + size]
        if hashlib.sha256(raw).hexdigest() != a["sha256"]:
            raise ValueError("array hash: " + name)
        for lo, hi in intervals:
            if size and start < hi and start + size > lo:
                raise ValueError("overlapping array: " + name)
        if size:
            intervals.append((start, start + size))
        arrays[name] = (a, raw)
    def rows(name, fmt):
        a, raw = arrays[name]
        if struct.calcsize(fmt) != a["element_bytes"]:
            raise ValueError("record stride mismatch: " + name)
        return list(struct.iter_unpack(fmt, raw))
    b = rows("full_rhs_before_fgmres", "<4f")
    action = rows("center_production_jdelta_action", "<4f")
    row_map = rows("fgmres_vector_row_map", "<4I")
    if len(b) != len(action) or len(b) != len(row_map):
        raise ValueError("inconsistent vector/map counts")
    node_desc, node_raw = arrays["fem_candidate_nodes"]
    if node_desc["element_bytes"] != 96:
        raise ValueError("unrecognized NMFEMNodeStateGPU layout")
    object_desc, object_raw = arrays["objects"]
    objects = []
    for i in range(object_desc["count"]):
        offset = i * object_desc["element_bytes"]
        words = struct.unpack_from("<16I", object_raw, offset)
        objects.append(dict(index=i, representation=words[0], material=words[1],
                            flags=words[2], node_offset=words[4], nodes=words[5],
                            element_offset=words[6], elements=words[7],
                            maximum_exponent=words[13]))
    micro_desc, _ = arrays["microstep_record"]
    micro_format = {48: "<8I4f", 64: "<8I8f"}.get(
        micro_desc["element_bytes"])
    if micro_format is None:
        raise ValueError("unrecognized NMMicrostepGPU record size")
    micro = rows("microstep_record", micro_format)[0]
    sched_desc, sched_raw = arrays["scheduler_state"]
    adapt_desc, adapt_raw = arrays["adaptive_state"]
    due = {}
    for obj in objects:
        i = obj["index"]
        offset = i * sched_desc["element_bytes"]
        active_exp = struct.unpack_from("<I", sched_raw, offset + 4)[0]
        numerical_enabled = struct.unpack_from("<f", sched_raw, offset + 44)[0]
        adaptive_rep = struct.unpack_from("<I", adapt_raw, i * adapt_desc["element_bytes"])[0]
        maximum = min(obj["maximum_exponent"], 8)
        period = 1 << (maximum - min(active_exp, maximum))
        due[i] = bool(obj["flags"] & 1 and adaptive_rep == obj["representation"] and
                      numerical_enabled > 0 and micro[1] % period == 0)
        obj.update(active_exponent=active_exp, due=due[i])
    totals = defaultdict(lambda: dict(rows=0, rhs_squared=0., action_squared=0.,
                                     residual_squared=0., nonfinite=0, peaks=[]))
    seen = set()
    for slot, (rhs, ax, mapping) in enumerate(zip(b, action, row_map)):
        env, block, local, source = mapping
        if env != 0 or block not in (1, 2, 3, 4, 5) or mapping in seen:
            raise ValueError("invalid or duplicate typed row mapping")
        seen.add(mapping)
        if block in (1, 2):
            if source >= len(objects) or local >= node_desc["count"]:
                raise ValueError("invalid FEM row identity")
            obj = objects[source]
            if not obj["node_offset"] <= local < obj["node_offset"] + obj["nodes"]:
                raise ValueError("FEM row owner mismatch")
        residual = tuple(x-y for x,y in zip(rhs, ax))
        labels = ["block_" + str(block)]
        if block == 1:
            labels += ["object_" + str(source), "due_mechanical" if due[source] else "non_due_mechanical"]
        finite = all(math.isfinite(x) for x in rhs + ax + residual)
        for label in labels:
            t = totals[label]; t["rows"] += 1
            if not finite:
                t["nonfinite"] += 1
                continue
            t["rhs_squared"] += math.fsum(x*x for x in rhs)
            t["action_squared"] += math.fsum(x*x for x in ax)
            norm2 = math.fsum(x*x for x in residual)
            t["residual_squared"] += norm2
            peak = dict(slot=slot, block=block, local=local, source=source,
                        residual_norm=math.sqrt(norm2), rhs=list(rhs), action=list(ax),
                        residual=list(residual))
            if block == 1:
                peak["position_m"] = list(struct.unpack_from("<3f", node_raw, local*96))
            t["peaks"].append(peak)
            if len(t["peaks"]) > 32:
                t["peaks"] = sorted(t["peaks"], key=lambda x: (-x["residual_norm"], x["slot"]))[:16]
    result = {}
    for label, t in totals.items():
        for name in ("rhs", "action", "residual"):
            t[name + "_l2"] = math.sqrt(t.pop(name + "_squared"))
        t["peaks"] = sorted(t["peaks"], key=lambda x: (-x["residual_norm"], x["slot"]))[:16]
        result[label] = t
    statuses = {}
    for name in ("solver_statuses", "center_operator_diagnostic_status"):
        decoded = rows(name, "<8I4f")
        statuses[name] = [
            dict(code=r[0], environment=r[1], object=r[2], failing_index=r[3],
                 completed_microsteps=r[4], maximum_linear_iterations=r[5],
                 diagnostics=[v if math.isfinite(v) else str(v) for v in r[8:]])
            for r in decoded]
    return dict(statuses=statuses,
                operator_evaluated_on_healthy_state=all(r["code"] == 0 for r in statuses["solver_statuses"]),
                manifest=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                payload_sha256=payload["sha256"], microstep=list(micro), objects=objects,
                due_nodes=sum(x["nodes"] for x in objects if x["due"]),
                due_tetrahedra=sum(x["elements"] for x in objects if x["due"]),
                totals=result,
                interpretation="Raw b-Ax vector diagnostics by typed block; heterogeneous block units are not summed or substituted for native acceptance norms. Contact arenas omitted; no replay or physical state update.")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = analyze(args.manifest)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("preserve previous analysis: " + str(args.output))
        args.output.write_text(text)
    else:
        print(text, end="")
if __name__ == "__main__":
    main()
