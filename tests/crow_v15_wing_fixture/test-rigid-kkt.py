#!/usr/bin/env python3
"""Research-only linearized rod and free v15 hand; not articulated MetalWorld."""
import hashlib,json,math,struct,subprocess,sys,tempfile
from pathlib import Path
here=Path(__file__).resolve().parent;ip=here/'rigid-kkt-input.bin';o=json.loads((here/'rigid-kkt-oracle.json').read_text());assert hashlib.sha256(ip.read_bytes()).hexdigest()==o['input_sha256']
with tempfile.TemporaryDirectory(prefix='crow-rigid-kkt-') as td:
 out=Path(td)/'out.bin';subprocess.run([sys.argv[1],str(here/'rigid-kkt.metal'),str(ip),str(out)],check=True,timeout=60);data=out.read_bytes()
assert len(data)==4*148;h=o['dt_s'];r=o['anchor'];mapped={}
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def norm(a):return math.sqrt(sum(v*v for v in a))
for i,c in enumerate(o['cases']):
 values=struct.unpack_from('<36fI',data,i*148);x=values[:33];lam=values[33:36];assert values[36]==0 and all(math.isfinite(v) for v in values[:36]),c['label'];A,b,J,K,m,x0,v0=c['A'],c['b'],c['J'],c['K'],c['mass'],c['x0'],c['v0'];reaction=[math.fsum(J[row][n]*lam[row] for row in range(3)) for n in range(33)];kq=[math.fsum(K[n][j]*x[j] for j in range(33)) for n in range(33)];delta=[m[n]*((x[n]-x0[n])/h-v0[n])+h*kq[n] for n in range(33)];force=max(abs(math.fsum(A[n][j]*x[j] for j in range(33))-b[n]-reaction[n]) for n in range(33));equality=max(abs(math.fsum(J[row][n]*x[n] for n in range(33))-c['target'][row]) for row in range(3));error=max(abs(x[n]-c['reference'][n]) for n in range(33));linear=[sum(delta[3*n+axis] for n in range(9))+delta[27+axis] for axis in range(3)];angular=list(delta[30:33])
 for n,node in enumerate(c['nodes']):
  lever=[r[0]+o['length_m']*node/8,r[1],r[2]];torque=cross(lever,delta[3*n:3*n+3]);angular=[angular[j]+torque[j] for j in range(3)]
 rod_force=[sum(reaction[3*n+axis] for n in range(9)) for axis in range(3)];body_force=reaction[27:30];body_torque=reaction[30:33];rt=cross(r,rod_force);pair_force=norm([rod_force[j]+body_force[j] for j in range(3)]);pair_torque=norm([rt[j]+body_torque[j] for j in range(3)])
 assert force<.01 and equality<1e-9 and error<1e-6 and norm(linear)<1e-8 and norm(angular)<1e-9 and pair_force<1e-10 and pair_torque<1e-10,(c['label'],force,equality,error,norm(linear),norm(angular),pair_force,pair_torque)
 mapped[c['label']]=[x[n] for n in c['inverse']]+list(x[27:]);print(c['label'],'force_residual',force,'equality',equality,'reference_error',error,'linear_momentum',norm(linear),'angular_momentum',norm(angular))
for name in ('rest','incoming'):
 gap=max(abs(a-b) for a,b in zip(mapped[name+'_forward'],mapped[name+'_reverse']));assert gap<1e-6;print(name,'node_order_gap',gap)
