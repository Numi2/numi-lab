#!/usr/bin/env python3
"""Sample accepted fixed-plane box segments; evidence, not a certificate."""
import argparse
import json
import math
from pathlib import Path


def slerp(q0, q1, t):
    n0=math.sqrt(sum(x*x for x in q0))
    n1=math.sqrt(sum(x*x for x in q1))
    q0=[x/n0 for x in q0];q1=[x/n1 for x in q1]
    dot=sum(a*b for a,b in zip(q0,q1))
    if dot<0:
        q1=[-x for x in q1];dot=-dot
    theta=math.acos(max(-1,min(1,dot)))
    if theta<1e-8:
        return q0
    denominator=math.sin(theta)
    return [(math.sin((1-t)*theta)*a+math.sin(t*theta)*b)/denominator
            for a,b in zip(q0,q1)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("trace",type=Path)
    parser.add_argument("--samples",type=int,default=10001)
    args=parser.parse_args()
    if args.samples<2:
        parser.error("at least two samples required")
    run=json.loads(args.run.read_text())["runs"][0]
    if not run["accepted"] or run["fixture"] not in (12,14):
        raise ValueError("adapter expects an accepted thin-impact fixture")
    events=[json.loads(line.partition("=")[2])
            for line in args.trace.read_text().splitlines()
            if line.startswith("ccd_event_trace=")]
    previous=[0,run["initial_q_y"],0,0,0,0,1]
    minimum=float("inf")
    segments=0
    hx,hy,hz=run["half_extents"]
    for event in events:
        if event["status"]!=0 or event["time"][3]<=0:
            continue
        end=event["q"]
        for j in range(args.samples):
            t=j/(args.samples-1)
            x,y,z,w=slerp(previous[3:7],end[3:7],t)
            nx=2*(x*y+z*w);ny=1-2*(x*x+z*z);nz=2*(y*z-x*w)
            center=previous[1]+t*(end[1]-previous[1])
            minimum=min(minimum,center-hx*abs(nx)-hy*abs(ny)-hz*abs(nz))
        segments+=1
        previous=end
        if event["time"][1]<=1e-9:
            break  # Do not include replay's duplicate events.
    if segments==0:
        raise ValueError("no complete accepted segment observed")
    print(json.dumps({"fixture":run["fixture"],"scale":run["scale"],
                      "segments":segments,"samples_per_segment":args.samples,
                      "sampled_minimum_gap_m":minimum,
                      "limitation":"samples do not certify an entire interval"}))


if __name__=="__main__":
    main()
