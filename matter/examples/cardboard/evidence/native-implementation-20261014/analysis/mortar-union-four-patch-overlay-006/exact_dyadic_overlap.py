#!/usr/bin/env python3
"""Exact rational clipping for binary32 barycentric triangle vertices."""
import csv, fractions, json, math, pathlib, struct, sys
from fractions import Fraction as F
base=pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/mortar-union-four-patch-overlay-006')
src=base/'candidates.log'
out=base/'exact-dyadic-results.json'
def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def exact32(s): return F.from_float(f32(float.fromhex(s)))
def sub(a,b): return (a[0]-b[0],a[1]-b[1])
def cross(a,b): return a[0]*b[1]-a[1]*b[0]
def orient(a,b,c): return cross(sub(b,a),sub(c,a))
def intersect_area(a,b):
    o=orient(b[0],b[1],b[2])
    if o==0: return None
    sign=1 if o>0 else -1
    poly=list(a)
    for i in range(3):
        e0,e1=b[i],b[(i+1)%3]
        def side(p): return sign*cross(sub(e1,e0),sub(p,e0))
        nxt=[]
        for j,p in enumerate(poly):
            q=poly[(j+1)%len(poly)]
            sp,sq=side(p),side(q)
            pin,qin=sp>=0,sq>=0
            if pin: nxt.append(p)
            if pin != qin:
                den=sp-sq
                if den==0: return None
                t=sp/den
                nxt.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
        poly=nxt
        if not poly: return F(0)
    if len(poly)<3: return F(0)
    twice=sum((cross(poly[i],poly[(i+1)%len(poly)]) for i in range(len(poly))),F(0))
    return abs(twice)/2
rows=[]
for line in src.read_text().splitlines():
    if not line.startswith('UNPROVEN,'): continue
    fields=line.split(',')
    side,a_id,b_id,src_area=fields[1:5]
    vals=[exact32(x) for x in fields[5:]]
    assert len(vals)==18
    tri1=[(vals[3*i],vals[3*i+1]) for i in range(3)]
    tri2=[(vals[9+3*i],vals[10+3*i]) for i in range(3)]
    area=intersect_area(tri1,tri2)
    if area is None: state='degenerate_or_undefined'
    elif area==0: state='exact_zero'
    else: state='positive'
    threshold=F.from_float(f32(1e-10))
    rows.append({'side':side,'stable_ids':[int(a_id),int(b_id)],'source_float_area':float.fromhex(src_area),'exact_area_numerator':str(area.numerator) if area is not None else None,'exact_area_denominator':str(area.denominator) if area is not None else None,'exact_area_float':float(area) if area is not None else None,'exact_area_state':state,'above_exact_fp32_threshold':bool(area is not None and area>threshold)})
from collections import Counter
counts=Counter(r['exact_area_state'] for r in rows)
above=sum(r['above_exact_fp32_threshold'] for r in rows)
by_pair={}
for r in rows: by_pair.setdefault(tuple(r['stable_ids']),[]).append(r)
summary={'schema':'numi.cardboard.exact_dyadic_overlap_audit.v1','input_path':str(src),'candidate_side_records':len(rows),'unique_stable_id_pairs':len(by_pair),'state_counts':dict(counts),'side_records_above_fp32_overlap_threshold':above,'unique_pairs_with_any_exact_area_above_threshold':sum(any(x['above_exact_fp32_threshold'] for x in v) for v in by_pair.values()),'unique_pairs_all_areas_exact_zero':sum(all(x['exact_area_state']=='exact_zero' for x in v) for v in by_pair.values()),'threshold_exact_value':str(F.from_float(f32(1e-10))),'method':'Sutherland-Hodgman convex polygon intersection with exact Fraction arithmetic; each input coordinate is first rounded/reconstructed as exact IEEE-754 binary32 rational. No epsilon, no long-double assumption.' ,'rows':rows}
out.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
