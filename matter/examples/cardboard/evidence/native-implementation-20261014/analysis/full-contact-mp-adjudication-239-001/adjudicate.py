from pathlib import Path
import hashlib,json,re,struct
import mpmath as mp
ROOT=Path('/Users/home/cardboard-implementation-20261014')
SOURCE=ROOT/'native/full-fp64-signed-box-contact-239/stdout.log'
OUTPUT=Path(__file__).resolve().parent

def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))
def norm(v):return mp.sqrt(sum(x*x for x in v))
def add(a,b):return [x+y for x,y in zip(a,b)]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def scale(a,s):return [x*s for x in a]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def normalize(q):return scale(q,1/norm(q))
def qmul(a,b):
 x,y,z,w=a;u,v,k,s=b
 return [w*u+x*s+y*k-z*v,w*v-x*k+y*s+z*u,w*k+x*v-y*u+z*s,w*s-x*u-y*v-z*k]
def rotation(q):
 x,y,z,w=normalize(q)
 return mp.matrix([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def apply(R,v):return list(R*mp.matrix(v))
def integrate(q,w,dt):
 phi=scale(w,dt);angle=norm(phi);half=angle/2
 factor=mp.sin(half)/angle if angle else mp.mpf('0.5')
 return normalize(qmul(scale(phi,factor)+[mp.cos(half)],q))
def decode(hexwords):
 # struct widens each finite binary32 word exactly to binary64 before mpf.
 return [mp.mpf(struct.unpack('<f',struct.pack('<I',int(w,16)))[0]) for w in hexwords.split(',')]

def residual(x,c):
 pos=x[0:3];q=normalize(x[4:8]);v=add(x[8:11],c[:3]);w=add(x[12:15],c[3:]);dt=x[48];th=x[49]
 bp=add(pos,scale(v,dt));bq=integrate(q,w,dt);delta=normalize(qmul(bq,[-q[0],-q[1],-q[2],q[3]]));D=rotation(delta)
 invI=mp.matrix([x[16:19],x[20:23],x[24:27]]);invWorld=D*invI*D.T
 proxyQ=qmul(bq,normalize(x[36:40]));R=rotation(proxyQ);center=add(bp,apply(rotation(bq),x[28:31]));half=x[32:35];point=x[40:43]
 local=apply(R.T,sub(point,center));closest=[max(-h,min(y,h)) for y,h in zip(local,half)];offset=sub(local,closest);distance=norm(offset);normal=apply(R,scale(offset,1/distance));witness=add(center,apply(R,closest));lever=sub(witness,bp)
 pointVelocity=add(v,cross(w,sub(witness,bp)));normalVelocity=dot(sub(x[44:47],pointVelocity),normal);arm=cross(lever,normal);response=x[47]+x[11]+dot(arm,apply(invWorld,arm))
 gap=max(distance,mp.mpf('1e-4')*th);remaining=max(th-gap,mp.mpf('1e-3')*th);t2=th*th;s2=gap*gap;offset2=s2-t2
 inertial=10/max(response*dt*dt*t2,mp.mpf('1e-20'));impact=max(-normalVelocity,0)/max(response*dt*remaining*t2,mp.mpf('1e-20'));stiffness=max(inertial,impact)
 first=-stiffness*(2*offset2*mp.log(s2/t2)+offset2*offset2/s2);impulse=scale(normal,-max(-2*gap*first,0)*dt);torque=cross(lever,impulse);length=norm(x[28:31])+norm(x[32:35]);result=impulse+scale(torque,1/length)
 feature=sum((1<<i) for i in range(3) if abs(local[i])>half[i]);branch=0 if inertial>impact else 1
 return result,feature,branch

lines=SOURCE.read_text().splitlines();inputs={};summaries={};centers={}
for line in lines:
 if line.startswith('contact_fp64_continuous_input'):
  row=fields(line);inputs[(row['case'],int(row['direction']))]=row
 if line.startswith('contact_fp64_continuous_summary'):
  row=fields(line);summaries[(row['case'],int(row['direction']))]=row
 if line.startswith('contact_fp64_continuous_center'):
  row=fields(line);centers[row['case']]=row
assert len(inputs)==len(summaries)==84
results=[]
for key,record in inputs.items():
 refs={}
 for precision in [50,80]:
  with mp.workdps(precision):
   x=decode(record['exact_f32_inputs_hex']);assert len(x)==62
   c=x[50:56];direction=x[56:62]
   f=lambda t:residual(x,add(c,scale(direction,t)))[0]
   # Independent high precision derivative of the complete scalar residual
   # components; no production tangent or copied analytic derivative is used.
   derivative=[-mp.diff(lambda t,i=i:f(t)[i],mp.mpf(0)) for i in range(6)]
   center,feature,branch=residual(x,c)
   stencils=[]
   for epsilon in [mp.mpf('1e-3'),mp.mpf('5e-4'),mp.mpf('2.5e-4'),mp.mpf('1.25e-4'),mp.mpf('6.25e-5')]:
    p,pfeature,pbranch=residual(x,add(c,scale(direction,epsilon)));m,mfeature,mbranch=residual(x,add(c,scale(direction,-epsilon)))
    fd=scale(sub(m,p),1/(2*epsilon));stencils.append(dict(epsilon=str(epsilon),relative_error=str(norm(sub(fd,derivative))/norm(derivative)),same_branch=pfeature==mfeature==feature and pbranch==mbranch==branch))
   refs[precision]=dict(action=[str(v) for v in derivative],center=[str(v) for v in center],stencils=stencils,feature=feature,branch=branch)
 with mp.workdps(80):
  a=[mp.mpf(v) for v in refs[80]['action']];b=[mp.mpf(v) for v in refs[50]['action']];s=summaries[key];native=[mp.mpf(v) for v in s['native_action_energy_scaled'].split(',')];fp64=[mp.mpf(v) for v in s['fp64_A_minus_dR_energy_scaled'].split(',')];an=norm(a)
  nativeRel=norm(sub(native,a))/an;fp64Rel=norm(sub(fp64,a))/an;precisionRel=norm(sub(a,b))/an
  passed=precisionRel<mp.mpf('1e-35') and nativeRel<=mp.mpf('1e-3') and fp64Rel<=mp.mpf('1e-6') and all(z['same_branch'] for z in refs[80]['stencils'])
  results.append(dict(case=key[0],direction=key[1],native_outcome=s['outcome'],high_precision=refs,precision_50_vs_80_relative=str(precisionRel),native_action_relative=str(nativeRel),fp64_final_fd_relative=str(fp64Rel),pass_=bool(passed)))
report=dict(schema='numi.cardboard.high-precision-contact-adjudication.v1',source_stdout=str(SOURCE),source_stdout_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_representation='Exact binary32 input words widened exactly; continuous independent high precision full contact residual. A=-dR.',precision_digits=[50,80],directions=len(results),all_pass=all(z['pass_'] for z in results),worst_native_action_relative=max(float(z['native_action_relative']) for z in results),worst_fp64_final_fd_relative=max(float(z['fp64_final_fd_relative']) for z in results),worst_50_vs_80_relative=max(float(z['precision_50_vs_80_relative']) for z in results),native_outcome_retained='Native239remainsINCONCLUSIVE due4nondecreasingFP64stencils and separateFP32projectedcandidateFD. This independent high-precision audit adjudicates same-input smooth contact derivative accuracy only.',scope='Six rigid generalized contact-only rows in six smooth frictionless BOX configurations; not assembled FEM/self-contact/cohesive/fullbox qualification.',rows=results)
(OUTPUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2));raise SystemExit(0 if report['all_pass'] else 2)
