#!/usr/bin/env python3
"""Independent final-state Coulomb check for the centered plane fixtures.

Sliding is > 1e-5 m/s. At each sliding contact, require impulse-vector
maximum-dissipation error <= 1e-5 N s + 0.1% of the friction limit. For
near-sticking rows require cone admissibility. These diagnostics use the
pre-step witness and the accepted final spatial velocity, not solver status.
"""
import json,math,sys
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def norm(a):return math.sqrt(dot(a,a))
a=json.load(open(sys.argv[1]))['cases'];out=[]
for c in a:
 if not c['accepted']:
  out.append({'geometry':c['geometry'],'initial_vn':c['initial_vn'],'initial_vt':c['initial_vt'],'far':c['far'],'status':c['status'],'accepted':False,'passed':False});continue
 v=c['final_v']; mu=c['friction'];rout=[]
 for r in c['rows']:
  assert r['body_a']==0 and r['body_b']==1
  n=r['normal'];t=r['tangent'];s=cross(n,t);anchor=r['anchor_b'];omega=v[3:6]
  rel=[v[i]+cross(omega,anchor)[i] for i in range(3)]
  slip=[dot(rel,t),dot(rel,s)];speed=norm(slip);J=r['normal_impulse'];jt=r['tangent_impulses'];limit=mu*J
  cone=max(norm(jt)-limit,0.);tol=1e-5+.001*limit
  err=norm([jt[i]+limit*slip[i]/speed for i in range(2)]) if speed>1e-5 else cone
  rout.append({'speed':speed,'normal_impulse_ns':J,'friction_impulse_ns':jt,'maximum_dissipation_error_ns':err,'limit_ns':tol,'pass':err<=tol})
 out.append({'geometry':c['geometry'],'initial_vn':c['initial_vn'],'initial_vt':c['initial_vt'],'far':c['far'],'status':c['status'],'accepted':True,'passed':all(r['pass'] for r in rout),'rows':rout})
print(json.dumps({'slide_speed_threshold_mps':1e-5,'absolute_impulse_tolerance_ns':1e-5,'relative_friction_limit_tolerance':.001,'cases':out},indent=2))
