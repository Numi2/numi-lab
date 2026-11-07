#!/usr/bin/env python3
"""Compare captured GPU nodal history against independently integrated FP64 BE."""
import json, math, sys
native=json.load(open(sys.argv[1]));cpu=json.load(open(sys.argv[2]))
assert native['arm']=='ipc_area' and native['coordinate_mode']=='reference_displacement'
assert native['h']==cpu['h'] and native['accepted_steps']==len(cpu['traces'])
nodes=native['compiled_nodes'];tets=native['compiled_tetrahedra']
assert len(nodes)==27 and len(tets)==48
mu,lam=2000.,5000.;width=.0001;k=native['area_ipc_k_N_per_m5']
# Reconstruct source-order mesh, reference weights and lumped masses without
# consuming the native compiled arrays. Compare each node and connectivity.
n=2;idx=lambda x,y,z:(z*(n+1)+y)*(n+1)+x
reference=[[.02*x/n,.02*y/n,.0002+.02*z/n,0.,0.]
           for z in range(n+1) for y in range(n+1) for x in range(n+1)]
shape=((0,1,3,7),(0,3,2,7),(0,2,6,7),(0,6,4,7),(0,4,5,7),(0,5,1,7))
connectivity=[]
for z in range(n):
 for y in range(n):
  for x in range(n):
   v=(idx(x,y,z),idx(x+1,y,z),idx(x,y+1,z),idx(x+1,y+1,z),
      idx(x,y,z+1),idx(x+1,y,z+1),idx(x,y+1,z+1),idx(x+1,y+1,z+1))
   for t in shape:
    ids=tuple(v[i] for i in t);connectivity.append(ids)
    for i in ids:reference[i][3]+=1100*(.02/n)**3/24
   if z==0:
    for j in (0,1,3,0,3,2):reference[v[j]][4]+=(.02/n)**2/6
assert [tuple(map(int,t[:4])) for t in tets]==connectivity
geometry_delta=max(abs(a-b) for x,y in zip(nodes,reference) for a,b in zip(x[:3],y[:3]))
mass_delta=max(abs(x[3]-y[3]) for x,y in zip(nodes,reference))
area_delta=max(abs(x[4]-y[4]) for x,y in zip(nodes,reference))
assert geometry_delta<1e-8 and mass_delta<1e-9 and area_delta<1e-9
def det(a):
 return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
   -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
   +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
def energies(state):
 potential=kinetic=0.
 for i,node in enumerate(nodes):
  z=state[i][2];mass,area=node[3:]
  potential+=mass*9.81*z
  kinetic+=mass*sum(v*v for v in state[i][3:])/2
  if area>0 and z<width:
   if z<=0:raise ValueError('nonpositive contact gap')
   q=z*z-width*width;potential-=area*k*q*q*math.log(z*z/width**2)
 for tet in tets:
  ids=list(map(int,tet[:4]));vol=tet[4];b=[tet[5:8],tet[8:11],tet[11:14]]
  d=[[state[ids[j+1]][i]-state[ids[0]][i] for j in range(3)] for i in range(3)]
  f=[[sum(d[i][q]*b[q][j] for q in range(3)) for j in range(3)] for i in range(3)]
  j=det(f)
  if j<=0:raise ValueError('inverted tetrahedron')
  log=math.log(j);sq=sum(v*v for row in f for v in row)
  potential+=vol*(mu*(sq-3)/2-mu*log+lam*log*log/2)
 return potential,kinetic
maxp=maxv=maxforce=maxpot=maxkin=0.;at={}
for nt,ct in zip(native['traces'],cpu['traces']):
 assert nt['accepted'] and nt['step']==ct['step'] and len(nt['node_state'])==len(ct['node_state'])==27
 p=max(abs(a-b) for na,nb in zip(nt['node_state'],ct['node_state']) for a,b in zip(na[:3],nb[:3]))
 v=max(abs(a-b) for na,nb in zip(nt['node_state'],ct['node_state']) for a,b in zip(na[3:],nb[3:]))
 pot,kin=energies(nt['node_state'])
 maxp=max(maxp,p);maxv=max(maxv,v);maxforce=max(maxforce,abs(nt['force']-ct['force']))
 maxpot=max(maxpot,abs(pot-ct['potential_J']));maxkin=max(maxkin,abs(kin-ct['kinetic_J']))
print(json.dumps({'schema':'numi.contact.nodal-parity.v1','h':native['h'],
 'steps':len(cpu['traces']),'total_compiled_mass_kg':sum(n[3] for n in nodes),
 'total_compiled_reference_area_m2':sum(n[4] for n in nodes),
 'max_compiled_rest_coordinate_difference_m':geometry_delta,
 'max_compiled_nodal_mass_difference_kg':mass_delta,'max_compiled_nodal_area_difference_m2':area_delta,
 'compiled_connectivity_matches':True,
 'max_node_coordinate_difference_m':maxp,'max_node_velocity_component_difference_mps':maxv,
 'max_normal_force_difference_N':maxforce,'max_potential_difference_J':maxpot,
 'max_kinetic_difference_J':maxkin,'compiled_nodes':len(nodes),'compiled_tetrahedra':len(tets),
 'limits':['energy from captured nodal coordinates independently re-evaluated using compiled inverse rest rows',
 'potential includes elasticity, gravity and the fixed contact potential',
 'independent CPU constitutive parameters are checked against declared fixture values, not read from GPU material program',
 'matching BE trajectories does not certify force accuracy against continuous dynamics']},indent=2))
