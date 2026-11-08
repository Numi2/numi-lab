#!/usr/bin/env python3
import json,subprocess,sys
import numpy as np
probe,metal,native=sys.argv[1:]
n=3; L=.11758755785232791; ell=L/2; h=.001;rad=2e-5;rho=1100
q=[[0.,0.,0.],[ell,0.,0.],[L,0.,0.]];z=np.zeros((3,3));tw=np.zeros(2)
def flat(x):return np.asarray(x).reshape(-1).tolist()
def run(args,vals):
 p=subprocess.run(args,input=' '.join(format(float(x),'.17g') for x in vals)+'\n',text=True,capture_output=True,timeout=30)
 if p.returncode or p.stderr:raise RuntimeError((p.returncode,p.stdout,p.stderr))
 return json.loads(p.stdout)
a=run([probe,metal],[n,24]+flat(q)+flat(z)+flat(tw)+flat(tw)+flat([q[0],q[-1]]))
assert a['status']==0 and a['gpu']==0 and a['published'] and a['states']==1 and a['reactions']==2
e=run([native],[n]+flat(a['q'])+flat(a['theta']))
forces=(np.asarray(e['stretchGradient1'])+np.asarray(e['bendGradient1'])[:9]).reshape(n,3)
mass=rho*np.pi*rad**2*ell
rv=np.array(a['v'])[1]+h*forces[1]/mass
perturbed=rv.copy();perturbed[0]+=.002 if rv[0]>=0 else -.002
out={'actual':a,'native_force_N':forces.tolist(),'residual_m_s':rv.tolist(),'perturbed_residual_m_s':perturbed.tolist(),'baseline_pass':bool(np.max(np.abs(rv))<=.001),'negative_detected':bool(np.max(np.abs(perturbed))>.001)}
print(json.dumps(out,sort_keys=True,allow_nan=False))
sys.exit(0 if out['baseline_pass'] and out['negative_detected'] else 1)
