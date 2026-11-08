#!/usr/bin/env python3
"""Compare a straight-rest local bend law against the native DER energy."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

p = Path(__file__).resolve().parent
exe = Path(sys.argv[1])
L = 0.11758755785232791
cases = []
for n in (3, 5, 9):
    for k in range(8):
        rng = np.random.default_rng(809 + len(cases))
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        q = [np.full(3, 0.5 if k % 2 else 0.0)]
        for _ in range(n - 1):
            axis = axis + rng.normal(size=3) * 0.22
            axis /= np.linalg.norm(axis)
            q.append(q[-1] + axis * (L / (n-1)) * rng.uniform(0.97, 1.03))
        twist = rng.uniform(-1.4, 1.4, size=n-1)
        cases.append((np.asarray(q), twist, "seeded"))
null = np.stack((np.linspace(0, L, 5), np.zeros(5), np.zeros(5)), axis=1)
cases.append((null, np.zeros(4), "null"))
rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
cases.append((cases[0][0] @ rotation.T + [0.3, -0.2, 0.1], cases[0][1].copy(), "transform"))

def native_input(case):
    x, t, _ = case
    return " ".join(map(lambda y: format(float(y), ".17g"), [len(x), *x.ravel(), *t]))
text = "\n".join(native_input(c) for c in cases) + "\n"
(p / "inputs.txt").write_text(text)
run = subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=False)
(p / "native.stdout").write_text(run.stdout); (p / "native.stderr").write_text(run.stderr)
if run.returncode != 0: raise RuntimeError(f"native exit {run.returncode}")
records = [json.loads(line) for line in run.stdout.splitlines()]
assert len(records) == len(cases)

def analytic(x,t,ei,gj,l0):
    n = len(x); g = np.zeros_like(x); gt = np.zeros(n-1); tt = np.zeros(n-1)
    bend = 0.; twist = 0.; bad = 0.; minc = 1.
    for i in range(n-2):
        e0=x[i+1]-x[i]; e1=x[i+2]-x[i+1]
        a=np.linalg.norm(e0); b=np.linalg.norm(e1); u=e0/a; v=e1/b
        c=float(u@v); minc=min(minc,c)
        d=t[i+1]-t[i]; r=(1-c)/(1+c); k=ei/l0
        eb=k*r*(1+np.cos(d)); bend+=eb; bad+=2*k*r
        deriv=-2*k*(1+np.cos(d))/(1+c)**2
        gu=deriv*(v-c*u)/a; gv=deriv*(u-c*v)/b
        g[i]-=gu; g[i+1]+=gu-gv; g[i+2]+=gv
        gd=-k*r*np.sin(d); gt[i]-=gd; gt[i+1]+=gd
        twist+=0.5*gj/l0*d*d; gtd=gj/l0*d; tt[i]-=gtd; tt[i+1]+=gtd
    return bend,twist,np.r_[g.ravel(),gt],tt,bad,minc

def diff(a,b,atol,rtol=3e-4):
    a=np.asarray(a); b=np.asarray(b); delta=np.abs(a-b)
    gate=np.maximum(atol,rtol*np.maximum(np.abs(a),np.abs(b)))
    return bool(np.all(delta<=gate)), float(np.max(delta)), float(np.max(delta/gate))

rows=[]; bad_detected=0
for index, ((x,t,kind), ref) in enumerate(zip(cases,records)):
    bend,twist,g,gt,bad,minc=analytic(x,t,ref["EI"],ref["GJ"],ref["restLength"])
    assert minc >= -0.5
    ben_ok=abs(bend-ref["bend"])<=max(1e-18,2e-10*abs(ref["bend"]))
    tw_ok=abs(twist-ref["twist"])<=max(1e-18,2e-10*abs(ref["twist"]))
    spatial=3*len(x)
    tests=[diff(ref["bendGradient0"][:spatial], ref["bendGradient1"][:spatial],5e-12),
           diff(g[:spatial], ref["bendGradient1"][:spatial],5e-12),
           diff(ref["bendGradient0"][spatial:],ref["bendGradient1"][spatial:],1e-15),
           diff(g[spatial:],ref["bendGradient1"][spatial:],1e-15),
           diff(ref["twistGradient0"],ref["twistGradient1"],1e-15),
           diff(gt,ref["twistGradient1"],1e-15)]
    if kind=="seeded" and abs(bad-ref["bend"])>max(1e-18,2e-10*abs(ref["bend"])): bad_detected+=1
    rows.append(dict(index=index,kind=kind,n=len(x),minimum_cosine=minc,bend_native_J=ref["bend"],bend_local_J=bend,
        twist_native_J=ref["twist"],twist_local_J=twist,bend_energy_pass=bool(ben_ok),twist_energy_pass=bool(tw_ok),
        gradient_checks=[dict(passed=a,max_error=b,max_gate_ratio=c) for a,b,c in tests],passed=bool(ben_ok and tw_ok and all(t[0] for t in tests))))
null_ok=abs(records[-2]["bend"])<=1e-18 and abs(records[-2]["twist"])<=1e-18
invariance=all(abs(records[0][key]-records[-1][key])<=max(1e-18,2e-10*abs(records[0][key])) for key in ("bend","twist","stretch"))
result=dict(seed="809..832",cases=rows,null_pass=bool(null_ok),rigid_transform_pass=bool(invariance),
            negative_control_detected=bad_detected,negative_control_expected=24,
            all_pass=bool(all(x["passed"] for x in rows) and null_ok and invariance and bad_detected==24))
(p/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))
print("failed_indices",[r["index"] for r in rows if not r["passed"]])
sys.exit(0 if result["all_pass"] else 1)
