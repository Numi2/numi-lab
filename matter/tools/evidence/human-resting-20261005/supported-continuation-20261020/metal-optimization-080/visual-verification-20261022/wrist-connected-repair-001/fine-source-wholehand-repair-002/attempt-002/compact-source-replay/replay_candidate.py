#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib
import numpy as np
p=argparse.ArgumentParser(description="Restore the retained failed002 source exactly; this is not an accepted anatomy asset")
p.add_argument("baseline",type=Path);p.add_argument("--output",type=Path)
a=p.parse_args();z=np.load(Path(__file__).with_name("candidate-vertex-records.npz"),allow_pickle=False)
b=a.baseline.read_bytes();assert hashlib.sha256(b).hexdigest()==str(z["baseline_payload_sha256"])
rows=z["changed_source_vertex_ids"];records=z["changed_vertex_record_bytes"];off=int(z["vertex_record_byte_offset"]);nv=int(z["payload_vertex_count"])
assert rows.ndim==1 and rows.dtype.kind=="u" and records.shape==(len(rows),56) and records.dtype==np.uint8
assert np.all(rows[1:]>rows[:-1]) and (not len(rows) or int(rows[-1])<nv)
assert off>=60 and off+56*nv<=len(b)
out=bytearray(b)
for i,row in zip(rows,records):
 s=off+56*int(i);out[s:s+56]=row.tobytes()
assert hashlib.sha256(out).hexdigest()==str(z["candidate_payload_sha256"])
if a.output:
 with a.output.open("xb") as f:f.write(out)
print("exact candidate SHA256",hashlib.sha256(out).hexdigest(),"geometry remains unqualified")
