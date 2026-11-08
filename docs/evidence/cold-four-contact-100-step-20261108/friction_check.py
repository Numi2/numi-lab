#!/usr/bin/env python3
import json,math,sys
d=json.load(open(sys.argv[1]));out=[]
for cidx,c in enumerate(d['cases']):
 for si,s in enumerate(c['steps']):
  if not s['accepted']: continue
  q=c['steps'][si-1]['q'] if si else [0,.01+c['initial_gap'],0];v=s['v']
  for ri,row in enumerate(s['rows']):
   p=[row['point'][i]-q[i] for i in range(3)];w=v[3:];vel=[v[0]+w[1]*p[2]-w[2]*p[1],v[1]+w[2]*p[0]-w[0]*p[2],v[2]+w[0]*p[1]-w[1]*p[0]];dirs=row['directions'];vt=[sum(a*b for a,b in zip(dirs[k],vel))-row['tangent_targets'][k-1] for k in (1,2)];pn=row['impulse'];pt=row['tangent_impulses'];speed=math.hypot(*vt);limit=.5*pn;tol=1e-5+.001*limit;err=math.hypot(pt[0]+limit*vt[0]/speed,pt[1]+limit*vt[1]/speed) if speed>1e-5 else max(math.hypot(*pt)-limit,0.0)
   if err>tol: out.append({'case':cidx,'step':si+1,'row':ri,'speed':speed,'normal_impulse':pn,'tangent_impulse':pt,'error':err,'tolerance':tol})
print(json.dumps({'failures':out,'failure_count':len(out)},indent=2))
