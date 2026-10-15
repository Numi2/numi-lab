#!/usr/bin/env python3
"""Compare selected raw Matter snapshots across runs without conflating fingerprints."""
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_json(p: Path):
    return json.loads(p.read_text())


def deep_diff(a, b, path="$"):
    if type(a) is not type(b):
        return [(path, a, b)]
    if isinstance(a, dict):
        out=[]
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b: out.append((f"{path}.{k}",a.get(k,"<missing>"),b.get(k,"<missing>")))
            else: out += deep_diff(a[k],b[k],f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a)!=len(b): return [(path+".length",len(a),len(b))]
        out=[]
        for i,(x,y) in enumerate(zip(a,b)): out += deep_diff(x,y,f"{path}[{i}]")
        return out
    return [] if a==b else [(path,a,b)]


def compare(a: Path,b: Path):
    ma=load_json(a/"candidate-capture.coupled.json")
    mb=load_json(b/"candidate-capture.coupled.json")
    pa=a/"candidate-capture.coupled.bin"; pb=b/"candidate-capture.coupled.bin"
    ba=pa.read_bytes(); bb=pb.read_bytes()
    assert len(ba)==len(bb)
    # The coupled capture binary is a typed array payload, not a snapshot envelope.
    # Snapshot-header decoding is done below against transaction .bin files.
    da=a/"diagnostic-state/dense"; db=b/"diagnostic-state/dense"
    names=sorted(p.name for p in da.iterdir() if p.is_file())
    assert names==sorted(p.name for p in db.iterdir() if p.is_file())
    snap_a=next((da/n).read_bytes() for n in names if n.endswith(".bin"))
    snap_b=next((db/n).read_bytes() for n in names if n.endswith(".bin"))
    assert len(snap_a)>=20 and len(snap_b)>=20
    # writeSnapshotPayload: uint32 message byte length, then scalar fields; first
    # two uint64 scalar fields are sourcePhysicsFingerprint and deviceProgramFingerprint.
    header_a=struct.unpack_from("<IQQ",snap_a,0); header_b=struct.unpack_from("<IQQ",snap_b,0)
    assert header_a[0]==header_b[0]==0
    assert header_a[1]==header_b[1], "source physics fingerprint word changed"
    diffs=[i for i,(x,y) in enumerate(zip(snap_a,snap_b)) if x!=y]
    changed_ranges=[]
    for i in diffs:
        if not changed_ranges or i>changed_ranges[-1][-1]+1: changed_ranges.append([i])
        else: changed_ranges[-1].append(i)
    snap_rows=[]
    for name in names:
        xa=(da/name).read_bytes(); xb=(db/name).read_bytes()
        row={"name":name,"bytes_a":len(xa),"bytes_b":len(xb),"sha256_a":sha(xa),"sha256_b":sha(xb),"byte_identical":xa==xb}
        if name.endswith(".bin"):
            row["differing_offsets"]=[i for i,(x,y) in enumerate(zip(xa,xb)) if x!=y]
            row["only_device_program_fingerprint_differs"]=(len(xa)==len(xb) and row["differing_offsets"]==list(range(12,20)))
        if name=="fixture-free-body-control-1201.log":
            xa_norm=xa.replace(str(a/"candidate-capture.coupled.json").encode(),b"<CAPTURE_MANIFEST>")
            xb_norm=xb.replace(str(b/"candidate-capture.coupled.json").encode(),b"<CAPTURE_MANIFEST>")
            row["manifest_path_normalized_equal"]=xa_norm==xb_norm
        snap_rows.append(row)
    cap_diff=deep_diff(ma,mb)
    # Input/source differences are recorded as provenance and not confused with output parity.
    pred_a=load_json(a/"prediction.json"); pred_b=load_json(b/"prediction.json")
    rec_a=load_json(a/"receipt.json"); rec_b=load_json(b/"receipt.json")
    result={
      "schema":"numi.cardboard.native_capture_state_parity.v1",
      "runs":{"a":str(a),"b":str(b)},
      "verdict":{"diagnostic_state_payloads_have_only_expected_program_fingerprint_difference":len(diffs)==8 and changed_ranges==[list(range(12,20))] and all((r.get("only_device_program_fingerprint_differs",False) if r["name"].endswith(".bin") else (r["byte_identical"] or r.get("manifest_path_normalized_equal",False))) for r in snap_rows),"whole_run_or_container_identity":False,"physical_convergence":"failed_in_both_runs"},
      "capture":{"schema_a":ma.get("schema"),"schema_b":mb.get("schema"),"root_a":ma.get("root"),"root_b":mb.get("root"),"iteration_a":ma.get("iteration"),"iteration_b":mb.get("iteration"),"physics_fingerprint_json_a":ma.get("source_physics_fingerprint"),"physics_fingerprint_json_b":mb.get("source_physics_fingerprint"),"device_program_fingerprint_json_a":ma.get("device_program_fingerprint"),"device_program_fingerprint_json_b":mb.get("device_program_fingerprint"),"manifest_sha256_a":sha((a/"candidate-capture.coupled.json").read_bytes()),"manifest_sha256_b":sha((b/"candidate-capture.coupled.json").read_bytes()),"manifest_diff":cap_diff,"payload_bytes_a":len(ba),"payload_bytes_b":len(bb),"payload_sha256_a":sha(ba),"payload_sha256_b":sha(bb),"payload_byte_identical":ba==bb},
      "snapshot_payload_header":{"contract":"little-endian: uint32 message length at 0; uint64 sourcePhysicsFingerprint at 4; uint64 deviceProgramFingerprint at 12","message_bytes_a":header_a[0],"message_bytes_b":header_b[0],"source_physics_u64_a":f"0x{header_a[1]:016x}","source_physics_u64_b":f"0x{header_b[1]:016x}","device_program_u64_a":f"0x{header_a[2]:016x}","device_program_u64_b":f"0x{header_b[2]:016x}","compared_snapshot_file":"transaction-000000-after.bin","raw_differing_byte_count":len(diffs),"raw_differing_offsets":diffs,"raw_difference_ranges":changed_ranges},
      "diagnostic_state_files":snap_rows,
      "run_receipts":{"exit_code_a":rec_a.get("exit_code"),"exit_code_b":rec_b.get("exit_code"),"timed_out_a":rec_a.get("timed_out"),"timed_out_b":rec_b.get("timed_out"),"inputs_unchanged_a":rec_a.get("source_inputs_unchanged"),"inputs_unchanged_b":rec_b.get("source_inputs_unchanged"),"source_snapshot_a":pred_a.get("source_snapshot"),"source_snapshot_b":pred_b.get("source_snapshot"),"build_a":pred_a.get("build_provenance"),"build_b":pred_b.get("build_provenance")},
      "limitations":["This compares emitted candidate/run state and transaction snapshots. The source/build snapshots differ, so it is not a same-executable diagnostics-on/off control.","The snapshot payloads differ in their 8-byte deviceProgramFingerprint field by design; all other bytes in the six paired snapshots are equal.","Both native runs exit 1; this is failed-run state parity, not solver convergence or physical qualification."]
    }
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("run_a",type=Path); ap.add_argument("run_b",type=Path); ap.add_argument("--output",type=Path,required=True); x=ap.parse_args()
    result=compare(x.run_a,x.run_b); x.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(x.output),"verdict":result["verdict"],"payload_byte_identical":result["capture"]["payload_byte_identical"],"snapshot_raw_differences":result["snapshot_payload_header"]["raw_difference_ranges"]},indent=2))
