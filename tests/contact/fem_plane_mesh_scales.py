#!/usr/bin/env python3
"""Linear small-strain wave and lumped contact-mass scale estimates."""
import json,math
rho,mu,lam=1100.,2000.,5000.
cp=math.sqrt((lam+2*mu)/rho);cs=math.sqrt(mu/rho)
tets=((0,1,3,7),(0,3,2,7),(0,2,6,7),(0,6,4,7),(0,4,5,7),(0,5,1,7))
out=[]
for n in (2,4,8,16):
 idx=lambda x,y,z:(z*(n+1)+y)*(n+1)+x
 mass=[0.] *((n+1)**3)
 for z in range(n):
  for y in range(n):
   for x in range(n):
    v=[idx(x,y,z),idx(x+1,y,z),idx(x,y+1,z),idx(x+1,y+1,z),
       idx(x,y,z+1),idx(x+1,y,z+1),idx(x,y+1,z+1),idx(x+1,y+1,z+1)]
    for tet in tets:
     for j in tet:mass[v[j]]+=rho*(.02/n)**3/24
 bottom=sum(mass[:(n+1)**2]);total=sum(mass);dx=.02/n
 out.append(dict(divisions=n,nodes=len(mass),bottom_mass_kg=bottom,
  bottom_mass_fraction=bottom/total,cell_m=dx,
  longitudinal_cell_crossing_s=dx/cp,longitudinal_wave_cfl_at_1us=cp*1e-6/dx))
print(json.dumps({'schema':'numi.contact.fem-wave-scales.v1',
 'longitudinal_speed_mps':cp,'shear_speed_mps':cs,
 'longitudinal_block_crossing_s':.02/cp,'meshes':out,
 'limits':['small-strain continuum estimates, not the discrete FEM eigenfrequencies',
 'contact stiffness produces additional modes; a wave-speed CFL alone does not prove stability',
 'bottom nodal masses are spatial lumping, not a separately simulated layer']},indent=2))
