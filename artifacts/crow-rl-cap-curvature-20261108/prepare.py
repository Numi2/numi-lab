from pathlib import Path
import hashlib,json,shutil,struct,subprocess
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-cap-curvature-20261108';B=R/'build/crow-cap-curvature-build-01';B.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
owner=R/'src/metal/MetalUnifiedQualitySolver.metal';source=owner.read_text()
needle='                    capAlpha / radius *';assert source.count(needle)==1
candidate=source.replace(needle,'                    capAlpha *')
for name,text in [('old',source),('fixed',candidate)]:
 p=B/name;p.mkdir();(p/'MetalUnifiedQualitySolver.metal').write_text(text)
(B/'runtime').mkdir();shutil.copy2(R/'build/lib/libmetalrobo.dylib',B/'runtime/libmetalrobo.dylib')
rows=[]
def save(name,w,r,s,adh,cap,group):
 n=len(w);assert 3<=n<=6
 values=[*np.pad(np.array(s,dtype='<f4'),(0,8-n),constant_values=1),*np.pad(np.array(r,dtype='<f4'),(0,8-n),constant_values=1),adh,cap,0,0]
 block=struct.pack('<8I20f',0,n,1,0,0,0,0,0,*values)+bytes(16)
 v=np.pad(np.array(w,dtype='<f4'),(0,6-n))
 rows.append(dict(name=name,n=n,w=v[:n].tolist(),r=np.array(r,dtype='<f4').tolist(),s=np.array(s,dtype='<f4').tolist(),adh=float(np.float32(adh)),cap=float(np.float32(cap)),group=group,raw=block+v.tobytes()))
for c in [.25,1,2,4]:save('isotropic-'+str(c),[-10,3,4],[1]*3,[1]*3,0,c,'closed')
save('adhesive-0.25',[-10,3,4],[1]*3,[1]*3,.125,.125,'closed')
rng=np.random.default_rng(525208)
for n in [3,4,6]:
 for i in range(8):
  s=np.r_[1.,rng.uniform(.5,1.3,n-1)];r=rng.uniform(.6,1.8,n);C=float(rng.choice([.25,.5,1.,2.,4.]));adh=C*.125 if i%2 else 0.;cap=C-adh
  unit=rng.normal(size=n-1);unit/=np.linalg.norm(unit);target=unit*C*rng.uniform(2.,4.)
  w=np.r_[-r[0]*(np.linalg.norm(target)*1.5+adh),-r[1:]*s[1:]*target]
  save(f'anisotropic-{n}-{i}',w,r,s,adh,cap,'anisotropic')
for name,w,cap in [('cap-interior',[-10,.01,-.02],1),('interior',[-2,.2,.1],0),('vertex',[3,.2,-.1],0),('boundary',[0,2,1],0)]:
 save(name,w,[1]*3,[1]*3,0,cap,'control')
assert len(rows)==33
(B/'cases.bin').write_bytes(struct.pack('<4I',0x43415031,1,len(rows),0)+b''.join(x.pop('raw') for x in rows))
(A/'cases.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
# Retain the previously calibrated independent uncapped FP64 projection.
reference=subprocess.check_output(['git','show','HEAD:artifacts/crow-rl-contact-precision-20261108/reference-source.py'])
(A/'independent-reference.py').write_bytes(reference)
plan=dict(schema='crow-cap-derivative-study-v1',question='Does capped-cone ball KKT use curvature alpha rather than alpha/radius, and does correcting it restore one-step Newton closure?',
 mechanism='Ball root returns y_t=-c_t/(d_t+alpha). The norm-constraint multiplier is alpha*radius; its tangential curvature is alpha*(I-u*uT). Dividing alpha by radius again is inconsistent.',
 baseline_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 owner=str(owner.relative_to(R)),owner_sha256=sha(owner),candidate_sha256=sha(B/'fixed/MetalUnifiedQualitySolver.metal'),candidate_patch=dict(old=needle,new='                    capAlpha *'),
 sources={n:dict(bytes=(A/n).stat().st_size,sha256=sha(A/n)) for n in ['prepare.py','run.py','project.metal','project.mm','closure.cpp','independent-reference.py','cases.json']},
 inputs={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [B/'cases.bin',B/'old/MetalUnifiedQualitySolver.metal',B/'fixed/MetalUnifiedQualitySolver.metal',B/'runtime/libmetalrobo.dylib']},
 controls=['isotropic closed form at radii .25,1,2,4','adhesive radius .25','unit-radius control','cap-interior and uncapped interior/vertex/boundary controls','independent finite-difference calibration of weighted capped-ball derivative','both direct and PCG full-solver Newton closure','exact repeated submissions'],
 predictions=dict(candidate_derivative_abs_error_max=2e-5,baseline_nonunit_closed_derivative_abs_error_min=.005,unit_control_baseline_abs_error_max=2e-5,projection_impulses_and_merits_bit_exact=True,unaffected_control_derivatives_bit_exact=True,candidate_one_newton_closure_passes=10,baseline_unit_one_newton_closure_passes=2,baseline_nonunit_one_newton_rejections=8,repeated_binary_and_typed_outputs_exact=True),
 experimental_units='33 fixed constitutive inputs and ten deterministic solver probes; repeats are determinism controls, not ecological or independent training replicates.',
 order=['old-0','fixed-0','fixed-1','old-1'],
 budget=dict(native_commands=8,projection_cases_per_command=33,closure_cases_per_command=10,timeout_seconds_each=60,raw_bytes_max=15000000),
 stopping='Stop unexpected process, malformed or nonfinite output. Preserve scientific prediction failures, no new candidates or budget extension.',
 limits='Corrects a capped projection derivative only; current crow captured problems are uncapped. Does not solve their precision rejection, qualify physical feeding or authorize tolerance changes.',
 production_changes_before_qualification=False)
(A/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(cases=len(rows),candidate_sha256=plan['candidate_sha256'],raw_bytes=sum(p.stat().st_size for p in B.rglob('*') if p.is_file()))))
