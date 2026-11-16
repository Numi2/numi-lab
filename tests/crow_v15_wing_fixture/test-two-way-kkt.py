#!/usr/bin/env python3
"""Research-only linearized two-way GPU rod regression; not a MetalWorld qualification."""
import hashlib,json,math,struct,subprocess,sys,tempfile
from pathlib import Path
binary=Path(sys.argv[1]);here=Path(__file__).resolve().parent
source=here/'two-way-kkt.metal';input_path=here/'two-way-kkt-input.bin';oracle=json.loads((here/'two-way-kkt-oracle.json').read_text())
assert hashlib.sha256(input_path.read_bytes()).hexdigest()==oracle['input_sha256'],'fixture input changed'
with tempfile.TemporaryDirectory(prefix='crow-two-way-kkt-') as tmp:
 output=Path(tmp)/'output.bin';subprocess.run([str(binary),str(source),str(input_path),str(output)],check=True,timeout=60);data=output.read_bytes()
assert len(data)==4*64
mapped={}
for i,c in enumerate(oracle['cases']):
 vals=struct.unpack_from('<15fI',data,i*64);x=vals[:12];lam=vals[12:15];status=vals[15];assert status==0 and all(math.isfinite(v) for v in vals[:15]),c['label']
 a,b,j,k=c['A'],c['b'],c['J'],c['K'];target=c['target'];m,x0,v0=c['mass'],c['x0'],c['v0'];h=oracle['dt_s']
 reaction=[sum(j[r][n]*lam[r] for r in range(3)) for n in range(12)]
 force=max(abs(sum(a[n][q]*x[q] for q in range(12))-b[n]-reaction[n]) for n in range(12));equality=max(abs(sum(j[r][n]*x[n] for n in range(12))-target[r]) for r in range(3));error=max(abs(x[n]-c['reference'][n]) for n in range(12));momentum=abs(sum(m[n]*((x[n]-x0[n])/h-v0[n]) for n in range(12))+h*sum(sum(k[n][q]*x[q] for q in range(12)) for n in range(12)))
 assert force<.01 and equality<1e-9 and error<.05e-6 and momentum<1e-8,(c['label'],force,equality,error,momentum)
 mapped[c['label']]=[x[n] for n in c['inverse']]+list(x[9:])
 print(c['label'],'force_residual',force,'equality_error',equality,'reference_error',error,'momentum_error',momentum)
for name in ('rest','incoming'):
 gap=max(abs(a-b) for a,b in zip(mapped[name+'_forward'],mapped[name+'_reverse']));assert gap<.05e-6,(name,gap);print(name,'order_gap',gap)
