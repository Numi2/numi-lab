#!/usr/bin/env python3
"""Independent FP64 references for bounded mixed scalar systems."""
import hashlib,json,math,struct,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent;refs=json.loads((p/'random-reference.json').read_text());count=0
for n,h in ((2,1),(4,2),(8,2)):
 f=p/f'random-n{n}.bin';blob=f.read_bytes();assert hashlib.sha256(blob).hexdigest()==refs[str(n)]['sha256'];N,nn,hh=struct.unpack_from('<III',blob);assert (N,nn,hh)==(64,n,h);m=h+1;z=struct.unpack_from('<%df'%((len(blob)-12)//4),blob,12);k=0;A=z[k:k+N*n*n];k+=N*n*n;J=z[k:k+N*m*n];k+=N*m*n;b=z[k:k+N*m];k+=N*m;v=z[k:k+N*n];out=json.loads(subprocess.run([sys.argv[1],str(f)],capture_output=True,text=True,check=True).stdout);assert out['success'] and out['published'] and len(out['paths'])==N
 for i in range(N):
  x=out['velocity'][i*n:(i+1)*n];lam=out['impulse'][i*m:(i+1)*m];ref=refs[str(n)]['cases'][i];aa=A[i*n*n:(i+1)*n*n];jj=J[i*m*n:(i+1)*m*n];bb=b[i*m:(i+1)*m];vv=v[i*n:(i+1)*n]
  av=[math.fsum(aa[r*n+c]*(x[c]-vv[c]) for c in range(n)) for r in range(n)];jt=[math.fsum(jj[r*n+c]*lam[r] for r in range(m)) for c in range(n)];mom=max(abs(a-t) for a,t in zip(av,jt))/(1+max(map(abs,av))+max(map(abs,jt)));hard=max(abs(math.fsum(jj[r*n+c]*x[c] for c in range(n))+bb[r]) for r in range(h));slack=math.fsum(jj[h*n+c]*x[c] for c in range(n))+bb[h]+struct.unpack('<f',struct.pack('<f',.02))[0]*lam[h];comp=abs(lam[h]*slack);dx=max(abs(a-b) for a,b in zip(x,ref['velocity']));dl=max(abs(a-b) for a,b in zip(lam,ref['impulse']))
  assert out['paths'][i]==4 and all(map(math.isfinite,x+lam)) and dx<2e-5 and dl<2e-5 and mom<=1e-5 and hard<=2e-5 and lam[h]>=-2e-5 and slack>=-2e-5 and comp<=2e-5,(n,i,dx,dl,mom,hard,slack,comp)
  count+=1
print('passed',count,'bounded mixed scalar FP64 controls; not mixed cone or world qualification')
