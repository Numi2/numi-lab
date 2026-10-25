#!/usr/bin/env python3
import hashlib,json,subprocess,sys,time
from pathlib import Path
D=Path("/Users/n/Work/numi-human-connected-surface-069")
R=D/"full-connected-clearance-150"
RUN=R/"run-029-affine-crossed-face-refinement"
C=RUN/"refined-candidate.npz"
AUD130=D/"audit_connected_outer_geometry_130.py"
AUD159=D/"audit_skin_bone_surfaces_159.py"
PY="/Users/n/numi-human-prep-venv-20261005/bin/python"
OUT=RUN/"post-audits-002"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir()
    start=time.monotonic(); manifest={"candidate_sha256":sha(C),"audit130_sha256":sha(AUD130),"audit159_sha256":sha(AUD159),"commands":{},"status":"RUNNING","physical_steps":0,"geometry_admitted":False}
    p130=OUT/"exact-point-self-audit-130"
    cmd130=[PY,str(AUD130),str(C),str(p130)]
    r=subprocess.run(cmd130,capture_output=True,text=True)
    (OUT/"audit-130.log").write_text(r.stdout+r.stderr)
    manifest["commands"]["audit130"]=cmd130
    if r.returncode or not (p130/"summary.json").is_file():
        manifest.update(status="AUDIT_130_FAILED_TO_COMPLETE",returncode130=r.returncode)
        (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        raise RuntimeError("130 audit failed; see log")
    q=json.loads((p130/"summary.json").read_text())
    manifest["audit130"]={"summary_sha256":sha(p130/"summary.json"),"complete_surface_audit":q.get("complete_surface_audit"),"exact_PLC_surface_pairs":q.get("exact_PLC_surface_pairs"),"inside_node_bank_pairs":q.get("inside_node_bank_pairs"),"inside_nodes":q.get("inside_nodes"),"query_pair_count":q.get("query_pair_count"),"status":q.get("status")}
    if q.get("complete_surface_audit") is not True or int(q.get("exact_PLC_surface_pairs",-1))!=0:
        manifest.update(status="STOPPED_AFTER_130_SELF_GATE",elapsed_seconds=time.monotonic()-start)
        (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps(manifest,indent=2));return
    p159=OUT/"all-bank-face-audit-159"
    cmd159=[PY,str(AUD159),str(C),str(p159)]
    r=subprocess.run(cmd159,capture_output=True,text=True)
    (OUT/"audit-159.log").write_text(r.stdout+r.stderr)
    manifest["commands"]["audit159"]=cmd159
    if r.returncode or not (p159/"report.json").is_file():
        manifest.update(status="AUDIT_159_FAILED_TO_COMPLETE",returncode159=r.returncode,elapsed_seconds=time.monotonic()-start)
        (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        raise RuntimeError("159 audit failed; see log")
    a=json.loads((p159/"report.json").read_text())
    if a.get("complete") is not True or len(a.get("banks",[])) != 185 or int(a.get("bank_count",-1)) != 185 or int(a.get("expected_bank_count",-1)) != 185:
        raise RuntimeError("159 report is not a complete 185-bank audit")
    per_bank=[]
    for bank in a["banks"]:
        if type(bank.get("body")) is not int or type(bank.get("stable_bone_id")) is not int:
            raise RuntimeError("159 bank key is untyped")
        result=bank.get("result")
        if result is None:
            if bank.get("skin_aabb_active") != 0 or bank.get("bank_aabb_active") != 0:
                raise RuntimeError("missing narrowphase result without explicit AABB exclusion")
            count=0
        else:
            count=result.get("count")
            if type(count) is not int or count < 0 or not isinstance(result.get("triangle_pairs"),list) or len(result["triangle_pairs"]) != count:
                raise RuntimeError("explicit exact count does not match triangle pair rows")
        per_bank.append({"body":bank["body"],"stable_bone_id":bank["stable_bone_id"],"count":count})
    keys={(row["body"],row["stable_bone_id"]) for row in per_bank}
    if len(keys) != 185: raise RuntimeError("duplicate typed bank identity")
    total=sum(row["count"] for row in per_bank)
    (p159/"canonical-count-receipt.json").write_text(json.dumps({"status":"COMPLETE_EXPLICIT_PER_BANK_COUNT_SUM","bank_count":185,"expected_bank_count":185,"total_crossing_pairs":total,"nonzero_bank_count":sum(row["count"]>0 for row in per_bank),"per_bank":per_bank},indent=2)+"\\n")
    manifest["audit159"]={"report_sha256":sha(p159/"report.json"),"status":a.get("status"),"complete":a.get("complete"),"bank_count":a.get("bank_count"),"expected_bank_count":a.get("expected_bank_count"),"crossing_pair_count":total,"nonzero_bank_count":sum(row["count"]>0 for row in per_bank),"wall_seconds":a.get("elapsed_seconds"),"canonical_count_receipt_sha256":sha(p159/"canonical-count-receipt.json")}
    manifest.update(status="COMPLETE_DIAGNOSTICS_UNADMITTED",elapsed_seconds=time.monotonic()-start)
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
