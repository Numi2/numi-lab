#!/usr/bin/env python3
"""Replay accepted STATIC SOURCE solution; no optimizer or physics emulation."""
from pathlib import Path
import argparse,hashlib,json,sys,numpy as np
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/"tools/rebind_human_skin_reference.py").is_file())
sys.path.insert(0,str(ROOT/"tools"));import rebind_human_skin_reference as rb
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path);args=ap.parse_args()
base=HERE.parent/"bilateral-knee-connected-weight-repair-100";source=base/"candidate.nhskin";pose=base/"neutral.json";saved=HERE/"solution.npz"
assert sha(source)=="27368e4bb9a0f59f41e67241a61b17de515a6ffb651b36dce6d47a81902869f3"
assert sha(saved)=="fd38e2816dffc6c73db460cd19d4590af756a82982dc4b8216b95d6982f8489d"
assert sha(pose)=="0abeef83c47ffcf304ef62be9321ed6d0169d942f3c20c05dd94e4df59c15db8"
p=rb.decode(source.read_bytes());z=np.load(saved,allow_pickle=False);sel=z["selected"];fraction=z["delta"]
assert sel.ndim==1 and len(np.unique(sel))==len(sel) and fraction.shape==sel.shape
assert np.all(np.isfinite(fraction)) and np.all((fraction>=0)&(fraction<=1))
lanes={int(b["body"]):i for i,b in enumerate(p["binds"])};ti,ta,ca=lanes[150],lanes[151],lanes[152]
W=p["weights"].astype(float);w=W.copy();w[sel,ti]+=fraction*(W[sel,ca]+W[sel,ta]);w[sel,ca]-=fraction*W[sel,ca];w[sel,ta]-=fraction*W[sel,ta]
a,b=rb.reference_transforms(p,json.loads(pose.read_text()),[0,0,.35]);blob,report=rb.rebind(p,w,a,b)
digest=hashlib.sha256(blob).hexdigest();assert digest=="4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5",digest
if args.output:
 with args.output.open("xb") as f:f.write(blob)
print(json.dumps({"sha256":digest,"bytes":len(blob),"changed_rows":len(report["changed_rows"]),"reference_error_max_m":report["reference_error_max_m"],"physical_admission":False}))
