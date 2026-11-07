import sys,json,math
sys.path.insert(0,'tests/ccd/world')
from contact_frame_diagnostic import rotate,cross,dot,add

e=next(json.loads(l.split('=',1)[1]) for l in open(sys.argv[1]) if l.startswith('ccd_event_trace='))
q=e['q'][3:7];conj=[-q[0],-q[1],-q[2],q[3]]
# Free-body fixture has unit mass and box inertia (kg m2).
hx,hy,hz=.2,.0002,.03
idiag=[3/(hy*hy+hz*hz),3/(hx*hx+hz*hz),3/(hx*hx+hy*hy)]
def inertia(v):
    local=rotate(conj,v)
    return rotate(q,[a*b for a,b in zip(local,idiag)])
def response(r_target,r_source,d):
    return add(d,cross(inertia(cross(r_source,d)),r_target))
dirs=[(0,1,0),(1,0,0),(0,0,1)]
r=[rotate(q,c['anchor_b']) for c in e['contacts']]
W=[[dot(dirs[i%3],response(r[i//3],r[j//3],dirs[j%3])) for j in range(6)] for i in range(6)]
before=e['source_v']
u=[]
for point in r:
    vel=add(before[:3],cross(before[3:],point))
    u.extend(dot(d,vel) for d in dirs)
target=[e['contacts'][i//3]['target_vn'] if i%3==0 else 0 for i in range(6)]
def inverse3(a):
    m=[[a[i][j] for j in range(3)]+[float(i==j) for j in range(3)] for i in range(3)]
    for k in range(3):
        pivot=max(range(k,3),key=lambda j:abs(m[j][k]));m[k],m[pivot]=m[pivot],m[k]
        d=m[k][k];m[k]=[v/d for v in m[k]]
        for i in range(3):
            if i!=k:
                d=m[i][k];m[i]=[a-d*b for a,b in zip(m[i],m[k])]
    return [row[3:] for row in m]
inv=[inverse3([row[3*c:3*c+3] for row in W[3*c:3*c+3]]) for c in range(2)]
alpha=[abs(W[3*c][3*c])/sum(abs(W[3*c][3*j]) for j in range(2)) for c in range(2)]
lam=[0.]*6
corrected="--corrected" in sys.argv
for k in range(1,257):
    rhs=[target[i]-u[i]-sum(W[i][j]*lam[j] for j in range(6)) for i in range(6)]
    new=[]
    for c in range(2):
        if corrected:
            dn=max(lam[3*c]+rhs[3*c]/W[3*c][3*c],0)-lam[3*c]
            b=inv[c]; t=rhs[3*c+1:3*c+3]
            p=[lam[3*c]+dn]
            for i in (1,2):
                p.append(lam[3*c+i]+sum((b[i][j]-b[i][0]*b[0][j]/b[0][0])*t[j-1] for j in (1,2)) + b[i][0]/b[0][0]*dn)
        else:
            p=[lam[3*c+i]+sum(inv[c][i][j]*rhs[3*c+j] for j in range(3)) for i in range(3)]
        p[0]=max(p[0],0); tang=math.hypot(p[1],p[2]); limit=.6*p[0]
        if tang>limit and tang>0:p[1]*=limit/tang;p[2]*=limit/tang
        new.extend(lam[3*c+i]+alpha[c]*(p[i]-lam[3*c+i]) for i in range(3))
    lam=new
    if k in (3,16,128,256):
        vel=[u[i]+sum(W[i][j]*lam[j] for j in range(6)) for i in range(6)]
        print(json.dumps({'iteration':k,'normal_impulse':lam[0],'tangent_impulse':lam[1:3],
          'normal_velocity':vel[0],'normal_residual':target[0]-vel[0],
          'normal_relaxation':alpha[0]}))
print('gpu_residual',e['solver_residuals'][1], 'gpu_J', e['contacts'][0]['J'])
