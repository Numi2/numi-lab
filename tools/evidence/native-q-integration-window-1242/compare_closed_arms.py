"""Compare closed native observation files by streaming all rows without changing state."""
import argparse, csv, hashlib, itertools, json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def pin(p,pins):
    p=Path(p); pins[str(p)]=sha(p); return p

def compare_csv(a,b,allowed,first,last,pins):
    pin(a,pins);pin(b,pins)
    diffs={};counts=[0,0]; outside=0; fieldnames=None
    with a.open(newline="") as fa,b.open(newline="") as fb:
        ra,rb=csv.DictReader(fa),csv.DictReader(fb)
        headers_equal=ra.fieldnames==rb.fieldnames;fieldnames=ra.fieldnames
        for index,(x,y) in enumerate(itertools.zip_longest(ra,rb)):
            counts[0]+=x is not None;counts[1]+=y is not None
            if not headers_equal or x is None or y is None:continue
            step=int(x.get("accepted_step",x.get("step","-1")))
            for k in fieldnames:
                if x[k]!=y[k]:
                    d=diffs.setdefault(k,{"different_rows":0,"first_examples":[]})
                    d["different_rows"]+=1
                    if len(d["first_examples"])<3:
                        d["first_examples"].append({"row_index":index,"step":step,"control":x[k],"window":y[k]})
                    outside+= not(first<=step<=((last+7)//8)*8)
    return {"headers_equal":headers_equal,"control_rows":counts[0],"window_rows":counts[1],
            "all_cells_equal":headers_equal and counts[0]==counts[1] and not diffs,
            "differing_columns":diffs,"unexpected_differing_columns":sorted(set(diffs)-set(allowed)),
            "different_cells_outside_sampling_intervals_overlapping_window":outside,
            "all_non_submission_diagnostic_cells_equal":headers_equal and counts[0]==counts[1] and not(set(diffs)-set(allowed)),
            "fingerprint_columns_present":all(k in (fieldnames or []) for k in
                ["stand_q_fingerprint_fnv64","stand_v_fingerprint_fnv64","root_translation_fingerprint_fnv64"])}

def grid(p,step_key,expected,pins,contacts=False):
    pin(p,pins);count=0;bad=[]
    wanted=iter(expected);expected_step=next(wanted,None);seen_indices=set();last=None
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            s=int(row[step_key]);count+=1
            if contacts:
                ci=int(row["contact_index"])
                if s!=last:
                    if last is not None:
                        if seen_indices!=set(range(32)):bad.append([last,"contact set incomplete"])
                    if s!=expected_step:bad.append([s,"unexpected step",expected_step])
                    expected_step=next(wanted,None);seen_indices=set();last=s
                if ci in seen_indices:bad.append([s,"duplicate contact",ci])
                seen_indices.add(ci)
            else:
                if s!=expected_step:bad.append([s,"unexpected step",expected_step])
                expected_step=next(wanted,None)
    if contacts and seen_indices!=set(range(32)):bad.append([last,"terminal contact set incomplete"])
    if expected_step is not None:bad.append(["missing step",expected_step])
    return {"rows":count,"complete":not bad,"errors":bad[:20]}

def main():
    ap=argparse.ArgumentParser()
    for x in ["control","window","output"]:ap.add_argument(x,type=Path)
    ap.add_argument("--steps",type=int,required=True)
    ap.add_argument("--first",type=int,required=True)
    ap.add_argument("--last",type=int,required=True)
    ap.add_argument("--captures",required=True)
    a=ap.parse_args()
    if a.output.exists():raise ValueError("output exists")
    if not 1<=a.first<=a.last<=a.steps:raise ValueError("invalid bounds")
    pins={}; receipts={}
    for arm in ["control","window"]:
        p=getattr(a,arm).parent/"execution.json";pin(p,pins)
        d=json.loads(p.read_text())
        if d["returncode"]!=0 or d["changed_inputs"]:raise ValueError("run is not closed with unchanged inputs")
        receipts[arm]=d
    reference=Path("/Users/n/numi-human-retained-delivery-20261009/q-integration-audit-window-1240/root-comparison-001/control-vs-window.json")
    expected="eac675c8c7fae9d5d342f1b7443b8bf3835784c0086d6ac86b1a2ce8129ae54a"
    if sha(pin(reference,pins))!=expected:raise ValueError("short reference changed")
    ref=json.loads(reference.read_text())
    allowed={k:sorted(v["differing_columns"]) for k,v in ref["csv_comparisons"].items()}
    reports={n:compare_csv(a.control/n,a.window/n,cols,a.first,a.last,pins) for n,cols in allowed.items()}
    grids={}
    for arm in ["control","window"]:
        for name,key in [("resting-coupled.csv","step"),("resting-com-momentum-diagnostic.csv","accepted_step")]:
            grids[arm+"/"+name]=grid(getattr(a,arm)/name,key,range(8,a.steps+1,8),pins)
        grids[arm+"/support"]=grid(getattr(a,arm)/"resting-com-support-impulses.csv","accepted_step",range(8,a.steps+1,8),pins,True)
    grids["q"]=grid(a.window/"resting-com-q-integration.csv","accepted_step",range(a.first,a.last+1),pins)
    grids["q_slip"]=grid(a.window/"resting-com-q-support-slip.csv","accepted_step",range(a.first,a.last+1),pins,True)
    captures={}
    for s in map(int,a.captures.split(",")):
        name="accepted-geometry/step-%d.mrvpack"%s
        x,y=pin(a.control/name,pins),pin(a.window/name,pins)
        captures[str(s)]={"control_sha256":pins[str(x)],"window_sha256":pins[str(y)],"identical":pins[str(x)]==pins[str(y)]}
    fp=["stand_q_fingerprint_fnv64","stand_v_fingerprint_fnv64","root_translation_fingerprint_fnv64"]
    com=reports["resting-com-momentum-diagnostic.csv"]
    fp_equal=com["headers_equal"] and com["control_rows"]==com["window_rows"] and com["fingerprint_columns_present"] and not(set(fp)&set(com["differing_columns"]))
    unchanged=all(sha(Path(p))==s for p,s in pins.items())
    report={"scope":"Closed native observer equivalence, not anatomy, momentum closure, or physiological validation.",
        "script_sha256":sha(Path(__file__)),"steps":a.steps,"q_first":a.first,"q_last":a.last,
        "submission_diagnostic_columns":allowed,"csv_comparisons":reports,"grids":grids,"captures":captures,
        "q_v_compensated_root_fingerprints_equal_at_all_normal_rows":fp_equal,
        "all_non_submission_diagnostic_cells_equal":all(v["all_non_submission_diagnostic_cells_equal"] for v in reports.values()),
        "different_cells_outside_sampling_intervals_overlapping_window":sum(v["different_cells_outside_sampling_intervals_overlapping_window"] for v in reports.values()),
        "input_sha256":pins,"input_pins_unchanged":unchanged,
        "limitations":["Fingerprints are not retained raw-array equality.",
                       "The normal COM sample interval is eight steps; status maxima and work sums describe the latest GPU submission.",
                       "All differing columns and values are reported; no numeric tolerance hides mismatches.",
                       "This comparison does not admit the existing skin crossings or explain late motion."]}
    report["observer_equivalence_checks_passed"]=(unchanged and fp_equal and
        report["all_non_submission_diagnostic_cells_equal"] and not report["different_cells_outside_sampling_intervals_overlapping_window"] and
        all(v["complete"] for v in grids.values()) and all(v["identical"] for v in captures.values()))
    with a.output.open("x") as f:json.dump(report,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"output":str(a.output),"sha256":sha(a.output),"passed":report["observer_equivalence_checks_passed"],
        "grids":{k:v["complete"] for k,v in grids.items()},"different_columns":{k:list(v["differing_columns"]) for k,v in reports.items()}}))
if __name__=="__main__": main()
