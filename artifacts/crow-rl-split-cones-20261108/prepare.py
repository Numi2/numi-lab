from pathlib import Path
import hashlib,json,os,struct,subprocess,sys
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-split-cones-20261108';B=R/'build/crow-split-cones-build-01'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
sys.path.insert(0,str(A));import independent as ref
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
B.mkdir();(B/'inputs').mkdir();cases=[]
def emit(name,nv,blocks,M,J,bias,free,start,group,rounding=0,**extra):
 rows=len(bias);raw=struct.pack('<8I',0x53504331,1,nv,rows,len(blocks),20,rounding,0)+b''.join(blocks)
 raw+=b''.join(np.asarray(x,dtype='<f4').tobytes() for x in [M,J,bias,free,start])
 p=B/'inputs'/(name+'.bin');p.write_bytes(raw);ref.parse(raw)
 cases.append(dict(name=name,path=str(p.relative_to(R)),sha256=sha(p),group=group,rounding=rounding,**extra))
def block(at,n,kind=1,r=None,s=None,adh=0.,cap=0.,bounds=None):
 r=np.ones(n) if r is None else r;s=np.ones(n) if s is None else s
 f=list(s)+[1.]*(8-n)+list(r)+[1.]*(8-n)+([adh,cap,0.,0.] if bounds is None else list(bounds)+[0.,0.])
 return struct.pack('<8I20f4I',at,n,kind,0,at+1,0,0,0,*f,0,0,0,0)
entries=archive.tree()
with archive.Objects() as objects:
 manifest=json.loads(objects.read(entries['artifacts/crow-rl-contact-precision-20261108/payload/manifest.json'][1]))
 for suffix,name in [('/substep7-capture.bin','substep7'),('/control-substep2.bin','substep2')]:
  matches=[f for f in manifest['files'] if f['path'].endswith(suffix)];assert len(matches)==1
  raw=b''.join(archive.content(matches[0],entries,objects));assert hashlib.sha256(raw).hexdigest()==matches[0]['sha256']
  (B/'inputs'/(name+'-capture.bin')).write_bytes(raw)
  dispatch=struct.unpack_from('<16I8f',raw,32);nv,rows,nb=dispatch[2:5];offset=256+128*nb
  blocks=[raw[256+i*128:256+(i+1)*128] for i in range(nb)]
  def take(n):
   global offset
   x=np.frombuffer(raw,dtype='<f4',count=n,offset=offset).copy();offset+=4*n;return x
  M=take(nv*nv).reshape(nv,nv);J=take(rows*nv).reshape(rows,nv);bias=take(rows);free=take(nv);warm=take(nv);wp=take(rows);failed=take(nv)
  emit(name,nv,blocks,M,J,bias,free,failed,'retained',capture_sha256=matches[0]['sha256'])
  if name=='substep7':emit(name+'-rounding',nv,blocks,M,J,bias,free,failed,'rounding',rounding=1)
# Constructed stationary controls; only FP32 A/J/bias/free/start are native inputs.
# The target is used only to choose free velocity, then independently solved after FP32 packing.
rng=np.random.default_rng(831007)
for n in [3,4,6]:
 for kind in ['boundary','adhesion','cap-boundary','cap-interior']:
  nv=n+1;s=np.r_[1.,np.linspace(.4,1.2,n-1)].astype('f4');r=np.linspace(.65,1.7,n).astype('f4')
  adh=.25 if kind=='adhesion' else .125 if kind=='cap-boundary' else 0.
  cap={3:.25,4:1.,6:4.}[n] if kind=='cap-boundary' else .5 if kind=='cap-interior' else 0.
  w=np.r_[.35,np.linspace(-2.,1.8,n-1)].astype('f4')
  if kind=='cap-boundary':w[0]=-12.
  if kind=='cap-interior':w=np.r_[-8.,np.linspace(.01,.025,n-1)].astype('f4')
  X=rng.normal(0,.08,(nv,nv));M=(np.diag(np.linspace(.7,1.9,nv))+X.T@X).astype('f4');M=(M+M.T)*np.float32(.5)
  J=rng.normal(0,.17,(n,nv)).astype('f4');v=rng.normal(0,.2,nv);bias=(w-J@v).astype('f4')
  impulse=ref.oracle(J.astype(float)@v+bias,r,s,adh,cap)[0]
  free=(v-np.linalg.solve(M.astype(float),J.astype(float).T@impulse)).astype('f4')
  emit(f'n{n}-{kind}',nv,[block(0,n,r=r,s=s,adh=adh,cap=cap)],M,J,bias,free,np.zeros(nv),'constructed')
# One fixed product exercises all seven regions and exact lower/upper endpoints.
specs=[(3,1,[-3.,.2,-.1],0.,0.,None),(3,1,[3.,.2,-.1],0.,0.,None),
 (4,1,[.3,-2.,.4,.8],.125,0.,None),(3,1,[-8.,.02,-.03],0.,.25,None),
 (6,1,[-12.,-3.,2.,-1.,.5,1.],.125,.5,None),
 (1,0,[-.2],0.,0.,[-.5,.5]),(1,0,[-3.],0.,0.,[-.5,.5]),
 (1,0,[.5],0.,0.,[-.5,.5]),(1,0,[-.5],0.,0.,[-.5,.5])]
nv=8;blocks=[];bias=[];at=0
for n,k,w,adh,cap,bounds in specs:
 blocks.append(block(at,n,k,adh=adh,cap=cap,bounds=bounds));bias.extend(w);at+=n
J=np.zeros((at,nv),dtype='f4');M=np.diag(np.linspace(.8,1.6,nv)).astype('f4')
emit('seven-regions',nv,blocks,M,J,bias,np.linspace(-.2,.3,nv),np.ones(nv),'constructed')
emit('null',3,[],np.diag([1.,2.,3.]),np.empty((0,3)),[],[.2,-.3,.5],[0.,0.,0.],'null')
emit('invalid-r',3,[block(0,1,0,r=[0.],bounds=[-.5,.5])],np.eye(3),np.zeros((1,3)),[1.],[.1,.2,.3],[.4,.5,.6],'invalid')
assert len(cases)==18
write(A/'cases.json',cases)
calibration=ref.calibrate();write(A/'reference-calibration.json',calibration)
sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(A.iterdir()) if p.is_file()}
plan=dict(schema='crow-split-cones-plan-v1',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 sources=sources,inputs={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted((B/'inputs').iterdir())},
 order=[c['name'] for c in cases],repeats=2,
 predictions=dict(valid_split_native_and_independent_residual_max=1e-10,valid_split_feasibility_max=1e-10,
 reference_velocity_abs_error_max=2e-9,impulse_scaled_error_max=2e-10,derivative_scaled_error_max=2e-10,
 gradient_scaled_error_max=2e-11,all_seven_regions=True,repeated_outputs_exact=True,
 rounding_ablation_rejected=True,invalid_r_typed_and_start_unchanged=True),
 limits=dict(native_commands=36,per_command_seconds=30,total_seconds=240,raw_bytes=10000000),
 scope='Research native two-FP32 cone, cap, adhesion and scalar solver. No world integration, FP32 output certificate, production/policy promotion or physical ecology qualification.')
write(A/'plan.json',plan);print(json.dumps(dict(cases=len(cases),inputs=len(plan['inputs']),calibration=calibration)))
