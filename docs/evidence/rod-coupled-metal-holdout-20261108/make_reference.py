#!/usr/bin/env python3
import json, subprocess, sys
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
base=D.parent/'rod-coupled-bend-20261108'
source=(base/'solve.py').read_text()
marker='\nrecords=[solve(c) for c in cases]'
assert source.count(marker)==1
ns={'__file__':str(base/'solve.py'),'__name__':'rod_reference'}
exec(compile(source.split(marker)[0],str(base/'solve.py'),'exec'),ns)
cases=[ns['make_case']((3,5,8,9)[i%4],2101+i) for i in range(12)]
serial=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in cases]
(D/'cases.json').write_text(json.dumps(serial,indent=2,allow_nan=False)+'\n')
records=[ns['solve'](c) for c in cases]
valid=[i for i,r in enumerate(records) if r['status']=='converged']
inputs='\n'.join(' '.join(format(float(x),'.17g') for x in [cases[i]['n'],*np.asarray(records[i]['q']).ravel(),*records[i]['theta']]) for i in valid)+'\n'
(D/'reference-native-inputs.txt').write_text(inputs)
p=subprocess.run([sys.argv[1]],input=inputs,capture_output=True,text=True,timeout=60)
(D/'reference-native.stdout').write_text(p.stdout);(D/'reference-native.stderr').write_text(p.stderr)
if p.returncode or p.stderr:raise RuntimeError(f'native exit={p.returncode} stderr={p.stderr}')
values=[json.loads(x) for x in p.stdout.splitlines()]
assert len(values)==len(valid)
for i,ev in zip(valid,values):
 c=cases[i];r=records[i];n=c['n'];h=.001;ell,mass,inert=ns['setup'](c);q=np.asarray(r['q']);reaction=np.asarray(r['reaction']); rv=[];rw=[];ar=[]
 for k in (0,1):
  b=np.asarray(ev[f'bendGradient{k}']);f=(np.asarray(ev[f'stretchGradient{k}'])+b[:3*n]).reshape(n,3);tor=b[3*n:]+np.asarray(ev[f'twistGradient{k}'])
  rv.append(np.asarray(r['v'])-c['v']+h*f/mass[:,None]);rw.append(np.asarray(r['omega'])-c['omega']+h*tor/inert)
  ar.append(-(mass[:,None]/h**2*((q-c['q'])-h*c['v'])+f)[[0,-1]])
 lm=float(np.max(np.abs(rv[1][1:-1])));am=float(np.max(np.abs(rw[1])))
 la=float(np.max(np.abs((rv[1]-rv[0])[1:-1])));aa=float(np.max(np.abs(rw[1]-rw[0])))
 ferr=np.linalg.norm(reaction-ar[1],axis=1);lim=np.maximum(1e-6,.01*np.linalg.norm(ar[1],axis=1))
 r.update(reference_momentum=dict(linear=lm,angular=am,linear_agreement=la,angular_agreement=aa,force_errors=ferr.tolist(),force_limits=lim.tolist()))
 r['passed']=bool(lm<=.0001 and am<=.0001 and la<=.0001 and aa<=.0001 and np.all(ferr<=lim))
out={'cases':records,'all_pass':bool(len(valid)==12 and all(r.get('passed',False) for r in records))}
(D/'reference.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'all_pass':out['all_pass'],'statuses':[r['status'] for r in records],'failed':[i for i,r in enumerate(records) if not r.get('passed',False)]},sort_keys=True))
sys.exit(0 if out['all_pass'] else 1)
