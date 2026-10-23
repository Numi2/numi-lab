#!/usr/bin/env python3
"""Compare hash-bound native Human diagnostics; this does not qualify anatomy or ROM."""
import argparse,csv,hashlib,json,re
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def kv(line):return {m[1]:m[2].rstrip(",") for m in re.finditer(r"([A-Za-z0-9_]+)=([^ ]+)",line)}
def read_run(d):
    m=json.loads((d/"launch-manifest.snapshot.json").read_text())
    r=json.loads((d/"execution-receipt.raw.json").read_text())
    assert r["returncode"]==0 and not r.get("launch_error"),r
    assert r["launch_manifest_sha256"]==sha(d/"launch-manifest.snapshot.json")
    assert all(sha(p)==h for p,h in m["input_hashes"].items()),"input changed"
    lines=(d/"stdout.log").read_text(errors="replace").splitlines()
    def get(prefix):
        matches=[s for s in lines if s.startswith(prefix)]
        assert matches,prefix
        return kv(matches[-1])
    a={"fem":get("resting_FEM_terminal_audit="),
       "fascia":get("resting_fascia_fem_terminal_audit="),
       "mass":get("whole_body_tissue_mass_reference="),
       "body":get("resting_integrated_body="),
       "source":get("resting_skin_fat_whole_body_preparation=")}
    trace=rows(d/"output/resting-coupled.csv")
    assert len(trace)==m["step_count"]==int(a["fem"]["accepted_steps"])
    assert float(a["fem"]["min_J"])>0
    assert a["fascia"]["passive_ligament_latest_transaction_accepted"]=="1"
    mass={}
    for line in (d/"tissue-mass-reference.txt").read_text().splitlines():
        p=line.split()
        if len(p)>=4 and p[0]=="body":mass[int(p[1])]=float(p[3])
    assert mass
    return m,r,a,trace,mass
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline",type=Path);p.add_argument("candidate",type=Path);p.add_argument("--output",type=Path,required=True)
    ns=p.parse_args();b=read_run(ns.baseline);c=read_run(ns.candidate)
    assert (b[0]["step_seconds"],b[0]["step_count"])==(c[0]["step_seconds"],c[0]["step_count"])
    massdiff=max(abs(c[4].get(k,0)-b[4].get(k,0)) for k in set(b[4])|set(c[4]))
    assert massdiff<1e-12,("per-body mass changed",massdiff)
    fields=("min_contact_gap_m","peak_penetration_m","root_assistance_n","root_assistance_nm",
        "body_mass_residual_kg","max_abs_body_mass_residual_kg","algebraic_transfer_momentum_residual_kg_m_s",
        "algebraic_transfer_angular_residual_kg_m2_s")
    comparison={}
    for f in fields:
        x=[float(r[f]) for r in b[3]];y=[float(r[f]) for r in c[3]]
        comparison[f]={"baseline_min":min(x),"baseline_max":max(x),"candidate_min":min(y),"candidate_max":max(y),
                       "max_abs_difference":max(abs(u-v) for u,v in zip(x,y))}
        if f!="min_contact_gap_m":assert max(abs(v) for v in y)==0,(f,y)
    out={"scope":"Matched native mechanics diagnostic only; not anatomy, joint-ROM or coupled-physiology qualification.",
         "physical_qualification":False,"baseline":str(ns.baseline),"candidate":str(ns.candidate),
         "baseline_manifest_sha256":sha(ns.baseline/"launch-manifest.snapshot.json"),
         "candidate_manifest_sha256":sha(ns.candidate/"launch-manifest.snapshot.json"),
         "candidate_receipt_sha256":sha(ns.candidate/"execution-receipt.raw.json"),
         "accepted_steps":len(c[3]),"simulated_seconds":c[0]["step_count"]*c[0]["step_seconds"],
         "wall_seconds":c[1]["elapsed_seconds"],"per_body_mass_max_abs_difference_kg":massdiff,
         "candidate_native_metrics":c[2],"trace_comparison":comparison,
         "limits":["Only the requested short diagnostic interval is tested.","A header-only surface audit is not evidence of collision-free anatomy."]}
    with ns.output.open("x") as f:json.dump(out,f,indent=2);f.write("\n")
    print(json.dumps({k:out[k] for k in ["accepted_steps","simulated_seconds","wall_seconds","per_body_mass_max_abs_difference_kg","physical_qualification"]}))
if __name__=="__main__":main()
