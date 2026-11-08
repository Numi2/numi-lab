#!/usr/bin/env python3
"""Compare retained native axial reactions with the pinned physical specimens."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from screen_ophthalmic_strand import parse, NAMES, EXPECTED_SHA

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("workbook",type=Path)
    ap.add_argument("native_stdout",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    series=parse(args.workbook)
    samples={0:[],1:[]}
    lines=args.native_stdout.read_text().splitlines()
    for line in lines:
        if not line.startswith("sample,"):
            continue
        _,env,step,x,f=line.split(",")
        env,step=int(env),int(step)
        x,f=float(x),float(f)
        if env not in samples or not math.isfinite(x) or not math.isfinite(f):
            raise ValueError("invalid native sample")
        if step != len(samples[env]) or abs(x-(2+step*.05))>1e-12:
            raise ValueError("missing or misordered native sample")
        samples[env].append((x,f))
    if any(len(rows)!=41 for rows in samples.values()):
        raise ValueError("incomplete native samples")
    if len([line for line in lines if "accepted=41" in line and "passed=1" in line])!=2:
        raise ValueError("native physical gate did not pass")
    result=[]
    for env,rows in samples.items():
        held=[]
        for name,pairs in zip(NAMES[1:],series[1:]):
            numerator=denominator=0.0
            for x,observed in pairs:
                if not 2<=x<=4:
                    raise ValueError("extension outside native domain")
                i=min(39,int((x-2)/.05))
                lo,hi=rows[i],rows[i+1]
                pred=lo[1]+(hi[1]-lo[1])*(x-lo[0])/(hi[0]-lo[0])
                numerator+=abs(pred-observed)
                denominator+=abs(observed)
            if not pairs or denominator<=0:
                raise ValueError("empty specimen")
            error=numerator/denominator
            held.append(dict(specimen=name,count=len(pairs),relative_absolute_error=error,passed=error<=.10))
        result.append(dict(environment=env,offset_m=(0 if env==0 else 16),heldout=held,passed=all(r["passed"] for r in held)))
    data=dict(workbook_sha256=EXPECTED_SHA,native_stdout_sha256=hashlib.sha256(args.native_stdout.read_bytes()).hexdigest(),
              interval_mm=[2,4],threshold=.10,environments=result,passed=all(r["passed"] for r in result))
    with args.output.open("x") as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps(data,indent=2))
    if not data["passed"]:
        raise SystemExit(1)

if __name__=="__main__":
    main()
