#!/usr/bin/env python3
"""Small 3D linearized rod plus point masses; not a MetalWorld or wing solver."""
import hashlib,json,math,struct,subprocess,sys,tempfile
from pathlib import Path
here=Path(__file__).resolve().parent;input_path=here/'rod-3d-input.bin';o=json.loads((here/'rod-3d-oracle.json').read_text());assert hashlib.sha256(input_path.read_bytes()).hexdigest()==o['input_sha256']
with tempfile.TemporaryDirectory(prefix='crow-rod-3d-') as tmp:
 output=Path(tmp)/'out.bin';subprocess.run([sys.argv[1],str(here/'rod-3d.metal'),str(input_path),str(output)],check=True,timeout=60);data=output.read_bytes()
assert len(data)==80*81*4;values=struct.unpack('<6480f',data);assert all(math.isfinite(x) for x in values)
k=o['K'];mass=o['effective_mass'];rod=o['rod_mass'];q0=o['q0'];v0=o['v0'];initial=.5*sum(q0[i]*sum(k[i][j]*q0[j] for j in range(27)) for i in range(27))+.5*sum(mass[i]*v0[i]**2 for i in range(27));maxpos=maxenergy=maxmomentum=maxforce=0.;effective_node=[rod[i]+(o['support'][o['indices'].index(i)] if i in o['indices'] else 0.) for i in range(9)]
for step in range(80):
 q=values[step*81:step*81+27];v=values[step*81+27:step*81+54];a=values[step*81+54:step*81+81];ref=o['expected_positions'][step];kq=[sum(k[i][j]*q[j] for j in range(27)) for i in range(27)];energy=.5*sum(q[i]*kq[i] for i in range(27))+.5*sum(mass[i]*v[i]**2 for i in range(27));momentum=math.sqrt(sum(sum(mass[3*i+axis]*(v[3*i+axis]-v0[3*i+axis]) for i in range(9))**2 for axis in range(3)));force=max(abs(effective_node[i//3]*a[i]+kq[i]) for i in range(27));maxpos=max(maxpos,max(abs(q[i]-ref[i]) for i in range(27)));maxenergy=max(maxenergy,abs(energy-initial)/initial);maxmomentum=max(maxmomentum,momentum);maxforce=max(maxforce,force)
print('max_position_error_m',maxpos,'max_relative_energy_error',maxenergy,'max_momentum_error',maxmomentum,'max_force_residual_N',maxforce)
assert maxpos<1e-6 and maxenergy<.01 and maxmomentum<1e-8 and maxforce<.01
