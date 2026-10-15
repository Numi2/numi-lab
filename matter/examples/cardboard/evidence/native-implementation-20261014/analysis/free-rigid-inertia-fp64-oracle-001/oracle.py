#!/usr/bin/env python3
"""Independent FP64 check of the free-BOX angular inertia Jacobian chain.

This is a CPU mathematics check only. It does not call Metal, the production
C++ checker, or solver code. Inputs match the frozen checker fixture after
canonicalizing authored GPU fields to FP32 and promoting those exact values to
FP64 for the independent smooth map.
"""
from __future__ import annotations
import hashlib, json, math, pathlib, struct

BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
CHECKER=BASE/'analysis/free-box-angular-residual-sign-070-overlay/cardboard_free_box_angular_native_check.mm'
SHADER=BASE/'snapshots/source-graph-tail-guard-angular-instrument-069/matter/src/metal/fgmres.metalinc'
NATIVE219=BASE/'native/free-box-angular-corrected-fixture-kernels-219'
NATIVE219_CHECKER=BASE/'analysis/free-box-angular-sample-approach-fixture-069-overlay/cardboard_free_box_angular_native_check.mm'
OUT=pathlib.Path(__file__).resolve().parent

def f32(x): return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def muls(s,a): return tuple(s*x for x in a)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def norm(a): return math.sqrt(dot(a,a))
def mm(a,b): return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))
def mt(a): return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))
def mv(a,x): return tuple(sum(a[i][j]*x[j] for j in range(3)) for i in range(3))
def ident(): return ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
def qnorm(q):
    n=math.sqrt(sum(x*x for x in q)); return tuple(x/n for x in q)
def qmul(a,b):
    x,y,z,w=a; X,Y,Z,W=b
    return (w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z)
def qconj(q): return (-q[0],-q[1],-q[2],q[3])
def qexp(phi):
    a=norm(phi); h=.5*a
    s=(math.sin(h)/a) if a>1e-14 else (.5-a*a/48.)
    return (phi[0]*s,phi[1]*s,phi[2]*s,math.cos(h))
def qrot(q):
    x,y,z,w=qnorm(q)
    return ((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
            (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
            (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
def inv3(a):
    a00,a01,a02=a[0];a10,a11,a12=a[1];a20,a21,a22=a[2]
    c=((a11*a22-a12*a21,a02*a21-a01*a22,a01*a12-a02*a11),
       (a12*a20-a10*a22,a00*a22-a02*a20,a02*a10-a00*a12),
       (a10*a21-a11*a20,a01*a20-a00*a21,a00*a11-a01*a10))
    det=a00*c[0][0]+a01*c[1][0]+a02*c[2][0]
    return tuple(tuple(v/det for v in row) for row in c)
def left_jac(phi,v):
    t2=dot(phi,phi)
    if t2<1e-12:
        a=.5-t2/24+t2*t2/720
        b=1/6-t2/120+t2*t2/5040
    else:
        t=math.sqrt(t2); a=(1-math.cos(t))/t2; b=(t-math.sin(t))/(t2*t)
    pv=cross(phi,v)
    return add(add(v,muls(a,pv)),muls(b,cross(phi,pv)))

# Match the exact scalar inputs and arithmetic boundary documented by checker.
raw_q=(.03,-.04,.02,.9985)
q_source=qnorm(raw_q)
q_source=tuple(f32(v) for v in q_source)  # NMRigidStateGPU storage
q_source=qnorm(q_source)
omega0=tuple(f32(v) for v in (.7,-.4,.5))
dt=f32(.01)
local_inv=((2.,0.,0.),(0.,1.3,0.),(0.,0.,.7))
R0=qrot(q_source)
# Baseline code computes this in host double then stores 9 FP32 body fields.
invI0=mm(mm(R0,local_inv),mt(R0))
invI0=tuple(tuple(f32(x) for x in row) for row in invI0)
candidate=tuple(f32(x) for x in (.13,-.09,.17))

def candidate_map(delta):
    total=add(omega0,delta)
    q_delta=qexp(muls(dt,total))
    Rdelta=qrot(q_delta)
    invIc=mm(mm(Rdelta,invI0),mt(Rdelta))
    Ic=inv3(invIc)
    # Free rigid residual in the angular rows is R=-I_world(q)*DeltaOmega.
    return Ic, muls(-1.,mv(Ic,delta))

def analytic_action(direction):
    I,R=candidate_map(candidate)
    total=add(omega0,candidate)
    phi=muls(dt,total)
    dtheta=muls(dt,left_jac(phi,direction))
    # A=-dR = I*d(DeltaOmega) + dI*DeltaOmega, with
    # dI=[dtheta]x I - I [dtheta]x for a spatial/world rotation.
    return add(mv(I,direction),sub(cross(dtheta,mv(I,candidate)),mv(I,cross(dtheta,candidate))))
def fd_action(direction,eps):
    plus=candidate_map(add(candidate,muls(eps,direction)))[1]
    minus=candidate_map(sub(candidate,muls(eps,direction)))[1]
    return muls(-1/(2*eps),sub(plus,minus))
def relerr(a,b): return norm(sub(a,b))/max(norm(b),1e-300)

dirs=[(1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(1/math.sqrt(3),)*3,
      (0.3,-0.4,math.sqrt(.75)),(-.7,.2,math.sqrt(.47))]
epses=(1e-4,1e-5,1e-6,1e-7)
rows=[]
for i,d in enumerate(dirs):
    aa=analytic_action(d)
    for eps in epses:
        ff=fd_action(d,eps)
        rows.append({'direction':i,'direction_xyz':d,'epsilon':eps,'analytic_A':aa,'fp64_fd_A':ff,'relative_error':relerr(aa,ff),'max_abs_error':max(abs(x-y) for x,y in zip(aa,ff))})

checker_hash=hashlib.sha256(CHECKER.read_bytes()).hexdigest()
shader_hash=hashlib.sha256(SHADER.read_bytes()).hexdigest()
native219_checker_hash=hashlib.sha256(NATIVE219_CHECKER.read_bytes()).hexdigest()
receipt=json.loads((NATIVE219/'receipt.json').read_text())
prediction=json.loads((NATIVE219/'prediction.json').read_text())
report={
 'schema':'numi.cardboard.free_rigid_inertia_fp64_oracle.v1',
 'classification':'independent_cpu_math_only_not_native_qualification',
 'inputs':{'checker_path':str(CHECKER),'checker_sha256':checker_hash,'production_shader_path':str(SHADER),'production_shader_sha256':shader_hash,
   'native_case_219':{'path':str(NATIVE219),'checker_bound_sha256':native219_checker_hash,'corrected_checker_sha256':checker_hash,'receipt_exit_code':receipt.get('exit_code'),'receipt_timed_out':receipt.get('timed_out'),'receipt_source_inputs_unchanged':receipt.get('source_inputs_unchanged'),'prediction_sha256':hashlib.sha256((NATIVE219/'prediction.json').read_bytes()).hexdigest(),'classification':'prior run used base checker, not corrected sign/branch overlay'},
   'q_source_authored':[.03,-.04,.02,.9985],'q_source_fp32_promoted':q_source,'omega0_fp32_promoted':omega0,
   'candidate_delta_omega_fp32_promoted':candidate,'dt_fp32_promoted':dt,
   'source_local_inverse_inertia':local_inv,'source_world_inverse_inertia_fp32_promoted':invI0,
   'equations':'q=Exp(dt*(omega0+DeltaOmega))*q0; Iinv(q)=R(Exp(phi))*Iinv0_world*R^T; R=-I(q)*DeltaOmega; A=-dR; dtheta=dt*J_left(phi)*dDeltaOmega; dI=[dtheta]x I-I[dtheta]x'},
 'max_relative_error_by_epsilon':{str(e):max(r['relative_error'] for r in rows if r['epsilon']==e) for e in epses},
 'max_abs_error_by_epsilon':{str(e):max(r['max_abs_error'] for r in rows if r['epsilon']==e) for e in epses},
 'records':rows,
 'interpretation':'The analytic SO(3)/world-inertia derivative agrees with independent FP64 central differences at epsilon 1e-6. This validates the mathematical chain at the authored source state only; the 219 native run was bound to the pre-correction checker, so it does not qualify the corrected checker or production action.'
}
assert report['max_relative_error_by_epsilon']['1e-06'] < 1e-8
assert native219_checker_hash != checker_hash
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'checker_sha256':checker_hash,'shader_sha256':shader_hash,'max_relative_error_by_epsilon':report['max_relative_error_by_epsilon'],'max_abs_error_by_epsilon':report['max_abs_error_by_epsilon']},indent=2))
