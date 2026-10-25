#!/usr/bin/env python3
import hashlib,json,subprocess,sys,time
from pathlib import Path
D=Path("/Users/n/Work/numi-human-connected-surface-069")
R=D/"full-connected-clearance-150"
RUN=R/"run-027-self-separation-connected-fit-001"
C=RUN/"candidate.npz"
AUD130=D/"audit_connected_outer_geometry_130.py"
AUD159=D/"audit_skin_bone_surfaces_159.py"
PY="/Users/n/numi-human-prep-venv-20261005/bin/python"
OUT=RUN/"post-audits-001"
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
    manifest["audit159"]={"report_sha256":sha(p159/"report.json"),"status":a.get("status"),"complete":a.get("complete"),"bank_count":a.get("bank_count"),"expected_bank_count":a.get("expected_bank_count"),"crossing_pair_count":sum(int(x.get("result",{}).get("pair_count",0)) for x in a.get("banks",[]) if isinstance(x.get("result"),dict)),"wall_seconds":a.get("elapsed_seconds")}
    manifest.update(status="COMPLETE_DIAGNOSTICS_UNADMITTED",elapsed_seconds=time.monotonic()-start)
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
