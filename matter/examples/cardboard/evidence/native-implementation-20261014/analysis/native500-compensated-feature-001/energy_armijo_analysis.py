import json, math, struct, hashlib
from decimal import Decimal as D, localcontext
from pathlib import Path
BASE=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PKG=BASE/'analysis/native500-compensated-feature-001'
CASE=BASE/'analysis/contact-armijo-energy-replay-499-001/case-496-inputs-v3'
OUT=BASE/'native/contact-barrier-energy-replay-500/output.bin'
EXPECTED='cb1f014b70d5bd21e08f385e4fbea2c6d857328b2fd4a9c0506d64cb18ff0237'
assert hashlib.sha256(OUT.read_bytes()).hexdigest()==EXPECTED
def f32(x): return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def flt(buf,off): return struct.unpack_from('<f',buf,off)[0]
def u32(buf,off): return struct.unpack_from('<I',buf,off)[0]
def df(x): return D.from_float(float(x))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def scale(s,a): return tuple(s*x for x in a)
def dot(a,b): return sum((x*y for x,y in zip(a,b)),D(0))
def closest(p,a,b,c):
 ab,ac,ap=sub(b,a),sub(c,a),sub(p,a)
 d1,d2=dot(ab,ap),dot(ac,ap)
 if d1<=0 and d2<=0:return a,(D(1),D(0),D(0)),'vertex_a'
 bp=sub(p,b);d3,d4=dot(ab,bp),dot(ac,bp)
 if d3>=0 and d4<=d3:return b,(D(0),D(1),D(0)),'vertex_b'
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:
  v=d1/max(d1-d3,df(f32(1e-12)));return add(a,scale(v,ab)),(1-v,v,D(0)),'edge_ab'
 cp=sub(p,c);d5,d6=dot(ab,cp),dot(ac,cp)
 if d6>=0 and d5<=d6:return c,(D(0),D(0),D(1)),'vertex_c'
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:
  w=d2/max(d2-d6,df(f32(1e-12)));return add(a,scale(w,ac)),(1-w,D(0),w),'edge_ac'
 va=d3*d6-d5*d4
 d4d3,d5d6=d4-d3,d5-d6
 if va<=0 and d4d3>=0 and d5d6>=0:
  den=max(d4d3+d5d6,df(f32(1e-12)))
  w=d4d3/den
  return add(b,scale(w,sub(c,b))),(D(0),1-w,w),'edge_bc'
 den=max(va+vb+vc,df(f32(1e-12)))
 v,w=vb/den,vc/den
 return add(add(a,scale(v,ab)),scale(w,ac)),(1-v-w,v,w),'face'
def vec(buf,off): return tuple(df(flt(buf,off+4*i)) for i in range(3))
def barrier_energy(sep,t,k):
 if t<=0 or k<=0 or sep>=t:return D(0)
 floor=df(f32(f32(1e-4)*f32(float(t))))
 d=max(sep,floor); s=d*d; tt=t*t; off=s-tt
 return max(-k*off*off*(s/tt).ln(),D(0))
def barrier_derivative(sep,t,k):
 if sep<=0 or sep>=t:return D(0)
 s=sep*sep;tt=t*t;off=s-tt
 ds=-k*(D(2)*off*(s/tt).ln()+off*off/s)
 return D(2)*sep*ds
def fenergy(sep,t,k):
 # Single-precision source-order transcription of ipcBarrierEnergy, normal branch.
 if not (t>0 and k>0) or sep>=t:return f32(0)
 floor=f32(f32(1e-4)*t)
 d=max(sep,floor)
 s=f32(d*d);tt=f32(t*t);off=f32(s-tt)
 ratio=f32(s/tt);lg=f32(math.log(ratio))
 e=f32(f32(f32(-k)*f32(off*off))*lg)
 return max(e,0.0)
case_contact=(CASE/'contacts.bin').read_bytes()
cor=(CASE/'correction.bin').read_bytes()
timestep=flt((CASE/'timestep.bin').read_bytes(),0)
out=OUT.read_bytes()
alphas=(CASE/'alphas.bin').read_bytes()
mjson=json.loads((CASE/'manifest.json').read_text())
# Use contact record 0 = reported slot7.
cb=case_contact[:432]
nodes=[u32(cb,4*i) for i in range(4)]
normal=tuple(df(flt(cb,64+4*i)) for i in range(3))
weights=[df(flt(cb,32+4*i)) for i in range(4)]
barrierx=df(flt(cb,144))
mollifierenergy=df(flt(cb,17*16+4))
thick=df(flt(cb,9*16+8))
stiff=df(flt(cb,17*16+12))
dt=df(timestep)
def endpoint(record,label):
 if label=='current': po,to,no=0,16,196
 elif label=='contact_limiter': po,to,no=64,80,204
 elif label=='fem_applied': po,to,no=128,144,212
 pts=[vec(out,record*224+po+16*i) for i in range(4)]
 tri=[vec(out,record*224+to+16*i) for i in range(3)]
 native_sep=df(flt(out,record*224+no))
 native_energy=df(flt(out,record*224+(192 if label=='current' else (200 if label=='contact_limiter' else 208))))
 q,bary,branch=closest(*pts[:1],*tri) # p; a,b,c
 delta=sub(pts[0],q)
 sep=dot(delta,delta).sqrt()
 return pts,tri,native_sep,native_energy,sep,bary,branch
def phi_of(tri):
 a,b,c=tri; x,y=sub(b,a),sub(c,a)
 scalev=dot(x,x)*dot(y,y); inner=dot(x,y)
 threshold=max(df(f32(f32(1e-3)*f32(float(scalev)))),df(f32(1e-12)))
 ratio=min(max((scalev-inner*inner)/threshold,D(0)),D(1))
 return ratio*(D(2)-ratio),scalev,ratio
# Source-expression float derivative from stored contact fields/correction arrays.
deriv_records=[]
for local in range(4):
 node=nodes[local]
 grad=tuple(flt(cb,208+16*local+4*i) for i in range(3))
 direction=tuple(f32(timestep*flt(cor,node*16+4*i)) for i in range(3))
 w=float(weights[local])
 impulse=tuple(f32(f32(f32(w*flt(cb,144))*flt(cb,64+4*i))-f32(flt(cb,276)*grad[i])) for i in range(3))
 dp=f32(f32(f32(impulse[0]*direction[0])+f32(impulse[1]*direction[1]))+f32(impulse[2]*direction[2]))
 deriv_records.append(dp)
impdot=f32(sum(deriv_records))
source_dd=f32(-impdot/timestep)
with localcontext() as ctx:
 ctx.prec=90
 rows=[]
 for i in range(8):
  alpha=df(flt(alphas,i*4))
  entries={}
  for label in ('current','contact_limiter','fem_applied'):
   pts,tri,native_sep,native_energy,sep,bary,branch=endpoint(i,label)
   phi,scalev,ratio=phi_of(tri)
   exactE=phi*barrier_energy(sep,thick,stiff)
   exactDerivative=phi*barrier_derivative(endpoint(i,'current')[4],thick,stiff)
   rounded_sep=f32(float(sep))
   roundedE=fenergy(rounded_sep,float(thick),float(stiff))
   entries[label]={
    'branch':branch,'phi':str(phi),'mollifier_scale':str(scalev),'mollifier_ratio':str(ratio),
    'native_sep':str(native_sep),'exact_sep':str(sep),'native_energy':str(native_energy),
    'corrected_decimal_energy':str(exactE),'source_energy_from_rounded_corrected_sep_f32':repr(roundedE),
    'native_vs_corrected_energy':str(native_energy-exactE),
   }
  cur=entries['current']['corrected_decimal_energy']
  for label in ('contact_limiter','fem_applied'):
   trial=D(entries[label]['corrected_decimal_energy'])
   # Exact source Armijo right side, using manifest's captured float derivative for independent binding.
   sourceDer=df(mjson['records'][0]['native_directional_derivative_j_per_alpha'])
   rhs=D(cur)+df(f32(f32(1e-4)*f32(float(alpha*sourceDer))))
   entries[label]['armijo_rhs_using_captured_native_derivative']=str(rhs)
   entries[label]['armijo_delta']=str(trial-D(cur))
   entries[label]['armijo_margin_trial_minus_rhs']=str(trial-rhs)
   entries[label]['armijo_accept']=bool(trial<=rhs)
  rows.append({'row':i,'alpha':str(alpha),'native_directional_derivative_manifest':str(df(mjson['records'][0]['native_directional_derivative_j_per_alpha'])),'source_float_directional_derivative_transcription':repr(source_dd),'current':entries['current'],'contact_limiter':entries['contact_limiter'],'fem_applied':entries['fem_applied']})
print(json.dumps({'schema':'native500.compensated_vt_energy_armijo.v1','scope':'CPU-only Decimal energy re-evaluation of exact FP32 native500 endpoint words using two-float closest distances; source directional derivative held at captured diagnostic value, no production change.','native500_output_sha256':EXPECTED,'rows':rows},indent=2,allow_nan=False))
