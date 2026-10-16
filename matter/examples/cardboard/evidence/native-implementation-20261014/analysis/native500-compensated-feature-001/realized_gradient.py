import runpy, contextlib, io, json
from decimal import localcontext
from pathlib import Path
base=Path('/Users/n/cardboard-implementation-20261014-host-macmini/analysis/native500-compensated-feature-001')
with contextlib.redirect_stdout(io.StringIO()):
 n=runpy.run_path(str(base/'energy_armijo_analysis.py'))
out=[]
with localcontext() as ctx:
 ctx.prec=90
 for i in range(8):
  cp,ctri,_,_,sep,bary,branch=n['endpoint'](i,'current')
  q,bary,branch=n['closest'](cp[0],*ctri)
  r=n['sub'](cp[0],q)
  dEdd=n['barrier_derivative'](sep,n['thick'],n['stiff'])
  gp=n['scale'](dEdd/sep,r)
  row={'row':i,'alpha':str(n['df'](n['flt'](n['alphas'],i*4))),'branch':branch,'gradient_point_J_per_m':list(map(str,gp)),'endpoints':{}}
  for label in ('contact_limiter','fem_applied'):
   tp,ttri,_,_,tsep,tbary,tbranch=n['endpoint'](i,label)
   dx=[n['sub'](tp[j],cp[j]) for j in range(4)]
   # Under a valid edge_bc projection, the envelope derivative is gp dot dp
   # minus barycentric weighted triangle endpoint displacements.
   rel=n['sub'](n['sub'](dx[0],dx[2]),n['scale'](bary[2],n['sub'](dx[3],dx[2])))
   pred=n['dot'](gp,rel)
   ec=n['phi_of'](ctri)[0]*n['barrier_energy'](sep,n['thick'],n['stiff'])
   et=n['phi_of'](ttri)[0]*n['barrier_energy'](tsep,n['thick'],n['stiff'])
   row['endpoints'][label]={'trial_branch':tbranch,'trial_sep_m':str(tsep),'actual_secant_deltaE_J':str(et-ec),'center_gradient_dot_actual_rounded_node_jump_J':str(pred),'gradient_dot_minus_actual_deltaE_J':str(pred-(et-ec)),'realized_relative_displacement_m':list(map(str,rel))}
  out.append(row)
(base/'realized-gradient.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps([{'row':x['row'],'alpha':x['alpha'],'contact':x['endpoints']['contact_limiter']} for x in out],indent=2))
