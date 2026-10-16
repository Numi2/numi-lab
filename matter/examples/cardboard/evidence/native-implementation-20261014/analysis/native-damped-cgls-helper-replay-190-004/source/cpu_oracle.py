import json, math

A0 = [[2.0e-3, -0.75e-3, 0.0], [0.5e-3, 3.0e-3, 0.0], [0.0, 0.0, 1.25e-3]]
b = [1.25e-4, -2.0e-4, 0.75e-4]

def solve(M, rhs):
    n = len(rhs)
    a = [list(map(float, M[i])) + [float(rhs[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c,n), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-30: raise AssertionError("singular oracle system")
        a[c],a[p]=a[p],a[c]
        d=a[c][c]
        a[c]=[v/d for v in a[c]]
        for r in range(n):
            if r==c: continue
            f=a[r][c]
            a[r]=[a[r][k]-f*a[c][k] for k in range(n+1)]
    return [a[i][n] for i in range(n)]

def transpose(A): return [list(v) for v in zip(*A)]
def matvec(A,x): return [sum(a*xj for a,xj in zip(row,x)) for row in A]
def dot(a,b): return math.fsum(x*y for x,y in zip(a,b))

rows=[]
for scale in (1.0,2.0):
    A=[[scale*v for v in row] for row in A0]
    AT=transpose(A)
    AtA=[[dot(AT[i], [A[k][j] for k in range(3)]) for j in range(3)] for i in range(3)]
    Atb=matvec(AT,b)
    pair=[]
    for mu in (0.0,1.0e-3):
        M=[row[:] for row in AtA]
        for i in range(3): M[i][i]+=mu*mu
        x=solve(M,Atb)
        residual=[b[i]-matvec(A,x)[i] for i in range(3)]
        gradient=[matvec(AT,residual)[i]-mu*mu*x[i] for i in range(3)]
        tol=1e-11*max(1.0, math.sqrt(dot(Atb,Atb)))
        assert all(math.isfinite(v) for v in x+residual+gradient)
        assert math.sqrt(dot(gradient,gradient)) <= tol, (scale,mu,gradient,tol)
        pair.append({"scale":scale,"mu_kg":mu,"x":x,"residual_norm":math.sqrt(dot(residual,residual)),"regularized_objective":dot(residual,residual)+mu*mu*dot(x,x),"normal_gradient_norm":math.sqrt(dot(gradient,gradient))})
    delta=math.sqrt(dot([pair[0]["x"][i]-pair[1]["x"][i] for i in range(3)],[pair[0]["x"][i]-pair[1]["x"][i] for i in range(3)]))
    assert delta>1e-3,(scale,delta)
    rows.extend(pair)
print(json.dumps({"schema":"cpu-damped-cgls-reference-check.v1","tests":4,"damping_direction_separation_l2":[math.dist(rows[0]["x"],rows[1]["x"]),math.dist(rows[2]["x"],rows[3]["x"])],"cases":rows,"result":"PASS"},sort_keys=True,allow_nan=False))
