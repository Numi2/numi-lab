#!/usr/bin/env python3
"""Independent small-strain 1D half-space contact-onset diagnostic.

Linear acoustic radiation impedance; not the finite 3D FEM fixture. Initial
free fall is integrated analytically and the boundary pressure/gap ODE is
then integrated with FP64 RK4 without Numi code or solver.
"""
import json,math,sys
h=float(sys.argv[1]);rho,mu,lam=1100.,2000.,5000.
g0,d,k,A,gravity=.0002,.0001,1e16,.0004,9.81
c=math.sqrt((lam+2*mu)/rho);Z=rho*c;t0=math.sqrt(2*(g0-d)/gravity)
def pressure(g):
 if g>=d:return 0.
 if g<=0:raise RuntimeError('nonpositive gap')
 q=g*g-d*d
 return 2*g*k*(2*q*math.log(g*g/(d*d))+q*q/(g*g))
def rhs(t,g):
 p=pressure(g)
 return -gravity*t+p/Z,A*p
t,g,I=t0,d,0.;bins=[0.]*10
while t<.01-1e-15:
 bin_index=min(9,int(t/.001))
 end=min(.01,(bin_index+1)*.001,t+h);dt=end-t
 a=rhs(t,g);b=rhs(t+dt/2,g+dt*a[0]/2)
 c_=rhs(t+dt/2,g+dt*b[0]/2);d_=rhs(t+dt,g+dt*c_[0])
 g+=dt*(a[0]+2*b[0]+2*c_[0]+d_[0])/6
 di=dt*(a[1]+2*b[1]+2*c_[1]+d_[1])/6
 I+=di;bins[bin_index]+=di/.001;t=end
print(json.dumps({'schema':'numi.contact.acoustic-halfspace.v1','h':h,'onset_s':t0,
 'longitudinal_speed_mps':c,'impedance_Pa_s_per_m':Z,'force_bins_N':bins,'impulse_Ns':I,
 'normal_arrival_from_top_s':t0+2*.02/c,
 'limits':['linear semi-infinite one-dimensional stress wave model',
 'omits finite-width free edges and all transverse deformation',
 'later bins are not a qualified 3D finite-body reference']},indent=2))
