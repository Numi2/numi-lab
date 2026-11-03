"""Replay the witnessed cavity selection; native import remains authoritative."""
from pathlib import Path
import argparse,hashlib,json,numpy as np
def pin(p):
 p=Path(p)
 return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--proposal",type=Path,required=True)
 ap.add_argument("--geometry-screen",type=Path,required=True)
 ap.add_argument("--native-transport-log",type=Path,required=True)
 ap.add_argument("--source-cell-count",type=int,required=True)
 ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args();a.output.mkdir(exist_ok=False)
 g=json.loads(a.geometry_screen.read_text())
 eligible=np.asarray(g["eligible_indices"],dtype=np.int64)
 records=[json.loads(l.split("=",1)[1]) for l in a.native_transport_log.read_text().splitlines()
          if l.startswith("skin_fat_cavity_transport_failures=")]
 if len(records)!=1 or not records[0].get("complete"):raise ValueError("complete native transport witness required")
 with np.load(a.proposal) as z:
  offsets=z["replacement_tet_offsets"];n=len(offsets)-1
  if not np.array_equal(offsets,np.arange(n+1)*2):raise ValueError("paired replacement offsets")
  r=z["replacement_tetrahedra"].reshape(n,2,4)
  p=z["replacement_parent_cell_rows"].reshape(n,2,3)
  if not np.array_equal(p[:,0],p[:,1]):raise ValueError("complete common parent ancestry")
  if not np.array_equal(z["replacement_parent_offsets"],np.arange(2*n+1)*3):
   raise ValueError("parent CSR")
  source=z["source_volume_sha256"].copy()
  if len(eligible)==0 or eligible.min()<0 or eligible.max()>=n or not np.all(np.diff(eligible)>0):
   raise ValueError("geometry selection domain or ordering")
  rejected=np.asarray(records[0]["cavity_indices"],np.int64)
  if rejected.size and (rejected.min()<0 or rejected.max()>=len(eligible) or len(np.unique(rejected))!=len(rejected)):
   raise ValueError("native transport witness domain or duplicates")
  chosen=eligible[~np.isin(np.arange(len(eligible)),rejected)]
  r=r[chosen].reshape(-1,4);p=p[chosen].reshape(-1,3)
 retired=np.unique(p.ravel())
 if len(retired)!=3*len(chosen) or retired.min()<0 or retired.max()>=a.source_cell_count:
  raise ValueError("disjoint complete retired-cell ownership")
 retained=np.flatnonzero(~np.isin(np.arange(a.source_cell_count),retired))
 out=a.output/"fixed-boundary-volume-proposal.npz"
 np.savez_compressed(out,retained_input_cell_rows=retained,replacement_tetrahedra=r,
  replacement_tet_offsets=np.arange(len(chosen)+1,dtype=np.int64)*2,
  replacement_parent_offsets=np.arange(len(r)+1,dtype=np.int64)*3,
  replacement_parent_cell_rows=p.ravel(),source_volume_sha256=source)
 np.save(a.output/"input-patch-indices.npy",chosen)
 report={"status":"CONSTRUCTION_ONLY_NATIVE_COMPOSITION_AND_MOTION_REQUIRED",
  "inputs":{k:pin(v) for k,v in {"proposal":a.proposal,"geometry_screen":a.geometry_screen,
      "native_transport_log":a.native_transport_log}.items()},
  "selected_cavities":len(chosen),"rejected_geometry":n-len(eligible),
  "rejected_transport":len(rejected),"output":pin(out),"coordinates_changed":False,
  "geometry_or_transport_gates_changed":False,"source_cell_count":a.source_cell_count}
 (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k!="inputs"},indent=2))
if __name__=="__main__":main()
