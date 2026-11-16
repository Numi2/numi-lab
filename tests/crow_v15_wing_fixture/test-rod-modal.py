#!/usr/bin/env python3
"""Narrow linearized free-rod modal GPU reference; not a full feather solver."""
import hashlib,json,math,struct,subprocess,sys,tempfile
from pathlib import Path
here=Path(__file__).resolve().parent;input_path=here/'rod-modal-input.bin';oracle=json.loads((here/'rod-modal-oracle.json').read_text());assert hashlib.sha256(input_path.read_bytes()).hexdigest()==oracle['input_sha256']
with tempfile.TemporaryDirectory(prefix='crow-rod-modal-') as tmp:
 output=Path(tmp)/'output.bin';subprocess.run([sys.argv[1],str(here/'rod-modal.metal'),str(input_path),str(output)],check=True,timeout=60);data=output.read_bytes()
assert len(data)==80*18*4;values=struct.unpack('<1440f',data);assert all(math.isfinite(x) for x in values)
k=oracle['K'];m=oracle['mass'];q0=oracle['q0'];v0=oracle['v0'];initial=.5*sum(q0[i]*sum(k[i][j]*q0[j] for j in range(9)) for i in range(9))+.5*sum(m[i]*v0[i]**2 for i in range(9));maxpos=maxenergy=maxmomentum=0.
for step in range(80):
 q=values[step*18:step*18+9];v=values[step*18+9:step*18+18];ref=oracle['expected_positions'][step];energy=.5*sum(q[i]*sum(k[i][j]*q[j] for j in range(9)) for i in range(9))+.5*sum(m[i]*v[i]**2 for i in range(9));maxpos=max(maxpos,max(abs(q[i]-ref[i]) for i in range(9)));maxenergy=max(maxenergy,abs(energy-initial)/initial);maxmomentum=max(maxmomentum,abs(sum(m[i]*(v[i]-v0[i]) for i in range(9))))
print('max_position_error_m',maxpos,'max_relative_energy_error',maxenergy,'max_momentum_error',maxmomentum)
assert maxpos<1e-6 and maxenergy<.01 and maxmomentum<1e-8
