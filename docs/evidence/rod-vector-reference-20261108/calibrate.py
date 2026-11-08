import numpy as np
# Independent closed-form one-free-node implicit chain against dense assembly.
L0=.058793778926163955;L1=L0;k0=2.3818718058345718/L0;k1=k0
m=7.0e-6;h=.001;left=0.;right=.122;x0=.060;v0=.01
A=np.diag([m/h**2]*3);rhs=np.array([m/h**2*left,m/h**2*(x0+h*v0),m/h**2*right])
for i,(k,L) in enumerate(((k0,L0),(k1,L1))):
 A[i,i]+=k;A[i+1,i+1]+=k;A[i,i+1]-=k;A[i+1,i]-=k;rhs[i]-=k*L;rhs[i+1]+=k*L
sol=np.linalg.solve(A[1:2,1:2],rhs[1:2]-A[1:2,[0,2]]@np.array([left,right]))[0]
closed=(m*(x0+h*v0)/h**2+k0*(left+L0)+k1*(right-L1))/(m/h**2+k0+k1)
assert abs(sol-closed)<1e-15
print("three-node dense reference agrees with closed form",abs(sol-closed))
