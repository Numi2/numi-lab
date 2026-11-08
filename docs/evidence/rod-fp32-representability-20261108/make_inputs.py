#!/usr/bin/env python3
"""Construct declared scalar snapshots; no simulation steps are executed."""
import hashlib
import json
import math
from pathlib import Path
import struct

here = Path(__file__).resolve().parent
prior = json.loads((here.parent/"rod-local-momentum-20261108/analysis-v3.json").read_text())
selected = [x for x in prior["cases"] if x["arm"] == "guarded" and x["n"] == 8 and
            x["axis"] == "x" and x["offset"] == .25 and x["kind"] == "fixed"]
assert len(selected) == 1
d = selected[0]
previous = json.loads((here/"result.json").read_text())
rows=[]
def add_rows(label, x, v0, vnew, lengths, stiffness, masses, h):
    for i in range(1,len(x)-1):
        items=[x[i-1],x[i],x[i+1],lengths[i-1],lengths[i],
               stiffness[i-1],stiffness[i],v0[i],vnew[i],h/masses[i]]
        assert all(math.isfinite(a) for a in items)
        rows.append({"label":label,"node":i,"values":items})
add_rows("retained-offset-0.25", previous["reference_positions"],d["v0"],
         previous["reference_velocities"],d["lengths"],
         [a/l for a,l in zip(d["ea"],d["lengths"])],d["mass"],d["h"])

def reference(n, offset):
    h=.001
    rest=.11758755785232791/(n-1)
    ea=2.3818718058345718
    stiffness=ea/rest
    mass=1100*math.pi*.000020**2*rest
    old=[.122*i/(n-1)+(.00017*math.sin(1.37*i) if 0<i<n-1 else 0) for i in range(n)]
    oldv=[(.012*math.cos(.71*i) if 0<i<n-1 else 0) for i in range(n)]
    A=[]; B=[]
    for i in range(1,n-1):
        row=[0.0]*(n-2);row[i-1]=mass/h**2+2*stiffness
        b=mass/h**2*(old[i]+h*oldv[i])
        if i>1: row[i-2]=-stiffness
        else: b += stiffness*old[0]
        if i<n-2: row[i]=-stiffness
        else: b += stiffness*old[-1]
        A.append(row);B.append(b)
    for j in range(n-2):
        p=max(range(j,n-2),key=lambda z:abs(A[z][j]))
        A[j],A[p]=A[p],A[j];B[j],B[p]=B[p],B[j]
        for i in range(j+1,n-2):
            a=A[i][j]/A[j][j]
            for q in range(j,n-2):A[i][q]-=a*A[j][q]
            B[i]-=a*B[j]
    sol=[0.0]*(n-2)
    for i in range(n-3,-1,-1):sol[i]=(B[i]-sum(A[i][j]*sol[j] for j in range(i+1,n-2)))/A[i][i]
    x=[old[0],*sol,old[-1]]
    vel=[(a-b)/h for a,b in zip(x,old)]
    # Residual validation in local FP64 coordinates.
    residual=max(abs(vel[i]-oldv[i]+h/mass*stiffness*(2*x[i]-x[i-1]-x[i+1])) for i in range(1,n-1))
    assert residual < 1e-8, residual
    add_rows(f"n{n}-offset-{offset}",[offset+a for a in x],oldv,vel,
             [rest]*(n-1),[stiffness]*(n-1),[mass]*n,h)
for offset in (.4,1.25,100.0):
    for n in (3,6,9):
        reference(n,offset)
assert len(rows)==42

def f32(x):
    return struct.unpack("<f",struct.pack("<f",x))[0]
def pair(x):
    high=f32(x)
    return high,f32(x-high)
blob=bytearray()
for row in rows:
    pairs=[pair(x) for x in row["values"]]
    row["pairs"]=[list(p) for p in pairs]
    blob.extend(struct.pack("<20f",*(v for p in pairs for v in p)))
(here/"inputs.bin").write_bytes(blob)
(here/"inputs.json").write_text(json.dumps(rows,indent=2)+"\n")
print("rows",len(rows),"bytes",len(blob),"sha256",hashlib.sha256(blob).hexdigest())
