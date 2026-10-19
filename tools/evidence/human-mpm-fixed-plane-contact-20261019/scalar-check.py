import math, random, json
random.seed(137)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def project(r,v,m,n):
    rn=dot(r,n); vn=dot(v,n)
    return [r[i]+n[i]*(min(m*vn,rn)-rn) for i in range(3)]
checks=0; max_error=0.
for k in range(2000):
    a=[random.uniform(-1,1) for _ in range(3)]
    z=math.sqrt(dot(a,a)); n=[x/z for x in a]
    m=10**random.uniform(-9,1)
    v=[random.uniform(-2,2) for _ in range(3)]
    r=[m*random.uniform(-3,3) for _ in range(3)]
    d=[random.uniform(-1,1) for _ in range(3)]
    hd=[m*random.uniform(-2,2) for _ in range(3)]
    vn=m*dot(v,n); rn=dot(r,n)
    if abs(vn-rn)<1e-4*m: continue
    active=vn<rn
    j=[hd[i]+n[i]*((m*dot(d,n) if active else dot(hd,n))-dot(hd,n)) for i in range(3)]
    e=1e-6
    plus=project([r[i]+e*hd[i] for i in range(3)],[v[i]+e*d[i] for i in range(3)],m,n)
    minus=project([r[i]-e*hd[i] for i in range(3)],[v[i]-e*d[i] for i in range(3)],m,n)
    err=max(abs((plus[i]-minus[i])/(2*e)-j[i])/m for i in range(3))
    assert err<2e-8, (k,err)
    max_error=max(max_error,err); checks+=1
n=[0,0,1]; m=2.
assert project([0,0,0],[0,0,0],m,n)==[0,0,0]
# Downward external load yields positive unconstrained momentum residual at zero normal velocity.
assert project([0,0,3],[0,0,0],m,n)==[0,0,0]
# Separating motion with zero residual stays unconstrained.
assert project([0,0,0],[0,0,1],m,n)==[0,0,0]
# Inward motion cannot solve the constrained row.
assert project([0,0,0],[0,0,-1],m,n)==[0,0,-2]
print(json.dumps({"checks":checks,"max_mass_normalized_jvp_error":max_error,"rest_load_separation_inward_checks":"pass","scope":"algebra only; not native GPU validation"}))

# Runtime stores b=-r. The projected RHS is -project(-b,v,m,n).
for _ in range(1000):
    a=[random.uniform(-1,1) for i in range(3)]; z=math.sqrt(dot(a,a)); n=[x/z for x in a]
    m=10**random.uniform(-9,1); v=[random.uniform(-2,2) for i in range(3)]; b=[m*random.uniform(-3,3) for i in range(3)]
    bn=dot(b,n); vn=dot(v,n)
    actual=[b[i]+n[i]*(max(-m*vn,bn)-bn) for i in range(3)]
    expected=[-x for x in project([-x for x in b],v,m,n)]
    assert max(abs(x-y) for x,y in zip(actual,expected))<1e-12*max(m,1)
print('runtime_negative_residual_sign_checks=1000_pass')
