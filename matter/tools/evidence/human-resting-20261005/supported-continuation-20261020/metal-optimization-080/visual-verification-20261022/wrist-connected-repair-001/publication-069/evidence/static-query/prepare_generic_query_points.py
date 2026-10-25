#!/usr/bin/env python3
import argparse,csv,hashlib,json,struct
from pathlib import Path
import numpy as np
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32bits(v): return struct.unpack("<I",struct.pack("<f",float(v)))[0]
def main():
 ap=argparse.ArgumentParser()
 g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--baseline-nodes",type=Path);g.add_argument("--candidate-npz",type=Path)
 ap.add_argument("--input-sha256",required=True);ap.add_argument("--field",default="current_points_f32");ap.add_argument("--output",required=True,type=Path)
 a=ap.parse_args()
 if a.output.exists() or a.output.is_symlink(): raise ValueError("output must be new")
 inp=a.baseline_nodes or a.candidate_npz; actual=sha(inp)
 if actual!=a.input_sha256: raise ValueError(f"input SHA mismatch: {actual}")
 lines=["query_row,x_m,y_m,z_m,x_bits_u32,y_bits_u32,z_bits_u32"]
 if a.baseline_nodes:
  with inp.open(newline="") as f:
   rd=csv.DictReader(f)
   if rd.fieldnames!=["global_node","layer","row","x_m","y_m","z_m","mass_kg"]: raise ValueError("baseline node schema mismatch")
   seen=set()
   for i,r in enumerate(rd):
    gid=int(r["global_node"])
    if gid in seen: raise ValueError("duplicate baseline global id")
    seen.add(gid);vals=[float(r[k]) for k in ("x_m","y_m","z_m")];bb=[f32bits(x) for x in vals]
    lines.append(",".join([str(i),*(format(struct.unpack("<f",struct.pack("<f",x))[0],".9g") for x in vals),*(str(x) for x in bb)]))
 else:
  with np.load(inp,allow_pickle=False) as z:
   if a.field not in z.files: raise ValueError("candidate field absent")
   points=np.asarray(z[a.field])
  if points.dtype!=np.float32 or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all(): raise ValueError("candidate field must be finite Nx3 Float32")
  bb=points.view(np.uint32)
  for i in range(len(points)): lines.append(",".join([str(i),*(format(float(x),".9g") for x in points[i]),*(str(int(x)) for x in bb[i])]))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("x",newline="") as f:f.write("\n".join(lines)+"\n")
 print(json.dumps({"input":str(inp),"input_sha256":actual,"field":a.field if a.candidate_npz else "baseline_registered_node_positions","query_count":len(lines)-1,"output":str(a.output),"output_sha256":sha(a.output),"semantics":"query row only; source global IDs are validated for uniqueness but intentionally not exported"},indent=2))
if __name__=="__main__":main()

