from pathlib import Path
import gzip,hashlib,json,math,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-consistent-wing-drive-20261113';import sys
arm=sys.argv[1]; assert arm in ('push','hold')
p=A/(arm+'-wing-20261126-rebound-consistent-rebound');d={'aero':{},'wing':{},'kin':{},'input':{},'wrench':{},'errors':[]}
with gzip.open(p/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  typ=z[0]
  if typ in ('CROW_AERO_FORCE','CROW_WING_EFFORT','CROW_KIN_STATE','CROW_AERO_INPUT','CROW_AERO_WRENCH'):
   t,k,e=map(int,z[1:4]);key=e,t,k
   if typ=='CROW_AERO_FORCE':target=d['aero'];values=list(map(float,z[5:]));assert len(values)==12
   elif typ=='CROW_WING_EFFORT':target=d['wing'];values=list(map(float,z[4:]));assert len(values)==14
   elif typ=='CROW_KIN_STATE':target=d['kin'];assert z[4:6]==['41','40'];values=list(map(float,z[6:]));assert len(values)==81
   elif typ=='CROW_AERO_INPUT':target=d['input'];values=list(map(float,z[4:]));assert len(values)==81
   else:target=d['wrench'];key+=(int(z[4]),);values=list(map(float,z[5:]));assert len(values)==6
   assert key not in target and all(map(math.isfinite,values)),(typ,key);target[key]=values
  elif typ in ('CROW_SUBSTEP_ERROR','CROW_AERO_OBSERVER_ERROR'):d['errors'].append(line.strip())
for e in range(8):
 for typ in ('aero','wing','kin','input'):
  assert sum((e,t,k) in d[typ] for t in range(348,401) for k in range(16))==53*16,(typ,e)
 assert sum((e,t,k,b) in d['wrench'] for t in range(348,401) for k in range(16) for b in (0,5,6,7))==53*16*4
assert not d['errors']
timing=0.;otherforce=0.
for key,inp in d['input'].items():
 timing=max(timing,max(abs(a-b) for a,b in zip(inp,d['kin'][key])))
 for j in range(3):otherforce=max(otherforce,abs(sum(d['wrench'][key+(b,)][j] for b in (0,5,6,7))-d['aero'][key][j]))
if timing or otherforce>=1e-7:
 (A/'integrity-failed.json').write_text(json.dumps({'kernel_input_max_diff':timing,'unaccounted_force_max':otherforce},indent=2)+'\n');raise SystemExit('observer timing or force coverage failed')
inputs=[];keys=[]
for e in range(8):
 for t in range(376,401):
  for k in range(16):
   keys.append((e,t,k));inputs.append(str(len(keys)-1)+' '+' '.join(map(str,d['input'][e,t,k])))
q=d['input'][5,376,0][:41];z=[0.]*40;v=z.copy();v[:3]=[.3,-.2,.1];cal=len(keys)
for idx,velocity in ((cal,z),(cal+1,v)):inputs.append(str(idx)+' '+' '.join(map(str,q+velocity)))
exe=R/'build/crow-v15-aero-power-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib';r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});out={}
for l in r.stdout.splitlines():
 x=l.split();i=int(x[0]);values=list(map(float,x[1:]));assert len(values)==24 and all(map(math.isfinite,values));out[i]=values
assert len(out)==len(inputs)
err=max(abs((out[cal+1][6*b+j]-out[cal][6*b+j])-v[j]) for b in range(4) for j in range(3));assert err<1e-8
rows=[]
for e in range(8):
 energies={b:0. for b in (0,5,6,7)};act=0.; powers=[]
 for i,key in enumerate(keys):
  if key[0]!=e:continue
  instant=0.
  for n,b in enumerate((0,5,6,7)):
   wrench=d['wrench'][key+(b,)];twist=out[i][6*n:6*n+6];power=sum(x*y for x,y in zip(wrench,twist));energies[b]+=power*.00125;instant+=power
  powers.append(instant)
  w=d['wing'][key];act+=sum(w[j]*(w[j+3]-w[j+2]) for j in (0,7))
 rows.append({'lane':e,'aero_signed_work_j_by_body':energies,'aero_signed_work_total_j':sum(energies.values()),'max_instantaneous_aero_power_w':max(powers),'positive_substeps':sum(x>1e-6 for x in powers),'wing_effort_displacement_proxy_j':act})
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();result={'valid':True,'kernel_input_vs_kin_max_abs':timing,'omitted_body_force_max_abs_n':otherforce,'uniform_translation_error_m_s':err,'substep_s':.00125,'window_ticks':[376,400],'rows':rows,'hashes':{'new_log':h(p/'run.stderr.gz'),'fk_binary':h(exe),'fk_library':h(lib)},'scope':'Signed work of modeled wrenches with substep-start COM twists in training seed, not complete energy balance or biological measurement. Wing effort times observed displacement is a discrete proxy, not independently measured muscle work. Correlated lanes. Wind configuration not assumed.'};(A/(arm+'-power-analysis.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
