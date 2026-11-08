#!/usr/bin/env python3
import json, subprocess, sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
rho=1100.; young=5e8; ell=.06; g=-9.81
initial=np.array([-.00004,.06013,.12002]); velocity=np.array([-.014,.007,.021])
units=[]; queries=[]
for r in (1e-5,2e-5,5e-5):
 for hh in (.001,.002):
  h=float(np.float32(hh)); area=np.pi*r*r; k=young*area/ell; b=young*np.pi*r**4/(4*ell**3)
  m=np.array([.5,1.,.5])*rho*area*ell; M=np.diag(m/(h*h)); xp=initial+h*velocity+h*h*g
  K=k*np.array([[1.,-1,0],[-1,2,-1],[0,-1,1]])
  xa=np.linalg.solve(M+K,M@xp+np.array([-k*ell,0.,k*ell]))
  for stage,x in (('predictor',xp),('axial_solution',xa)):
   lengths=np.diff(x); H=M.copy()
   for i in range(2):
    v=np.zeros(3); v[i]=-1; v[i+1]=1; H+=k*(lengths[i]-ell)/lengths[i]*np.outer(v,v)
   v=np.array([1.,-2.,1.]); H+=b*np.outer(v,v)
   unit={'radius':r,'young':young,'h':hh,'stage':stage,'position':x.tolist(),'matrix':H.tolist(),'eigenvalues':np.linalg.eigvalsh(H).tolist(),'analytic_diagonal':float(H[2,2]),'inertia':float(M[2,2])}
   units.append(unit)
   for eps in (0.,1e-6,-1e-6,5e-7,-5e-7): queries.append(' '.join(format(z,'.17g') for z in [r,young,*x,eps]))
p=subprocess.run([sys.argv[1]],input='\n'.join(queries)+'\n',text=True,capture_output=True,timeout=30)
if p.returncode: raise RuntimeError(p.stderr)
energies=[float(x) for x in p.stdout.splitlines()]
if len(energies)!=len(queries): raise ValueError('native energy query count')
valid=True
for i,u in enumerate(units):
 e=energies[i*5:i*5+5]; ref=u['analytic_diagonal']; est=[(e[1]+e[2]-2*e[0])/1e-12+u['inertia'],(e[3]+e[4]-2*e[0])/2.5e-13+u['inertia']]
 tol=max(1e-7,.002*abs(ref)); passed=all(np.isfinite(z) and abs(z-ref)<=tol for z in est)
 u.update(native_energy=e,native_diagonal=est,tolerance=tol,passed=bool(passed)); valid=valid and passed
out={'units':units,'native_queries':queries,'all_native_checks':bool(valid),'predictor_indefinite_all_long_h':all(u['eigenvalues'][0]<0 for u in units if u['h']==.002 and u['stage']=='predictor'),'axial_spd_all_long_h':all(u['eigenvalues'][0]>0 for u in units if u['h']==.002 and u['stage']=='axial_solution')}
(root/'curvature-analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'all_native_checks':out['all_native_checks'],'predictor_indefinite_all_long_h':out['predictor_indefinite_all_long_h'],'axial_spd_all_long_h':out['axial_spd_all_long_h']},indent=2))
sys.exit(0 if valid else 1)
