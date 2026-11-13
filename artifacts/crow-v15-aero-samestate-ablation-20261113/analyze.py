from pathlib import Path
import gzip,hashlib,json,math,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-aero-samestate-ablation-20261113';p=A/'push-wing-20261126-rebound-samestate';d={'aero':{},'wing':{},'kin':{},'input':{},'wrench':{},'ablate':{},'config':{},'errors':[]}
with gzip.open(p/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  typ=z[0]
  if typ in ('CROW_AERO_FORCE','CROW_WING_EFFORT','CROW_KIN_STATE','CROW_AERO_INPUT','CROW_AERO_WRENCH','CROW_AERO_ABLATE'):
   t,k,e=map(int,z[1:4]);key=e,t,k
   if typ=='CROW_AERO_FORCE':target=d['aero'];values=list(map(float,z[5:]));assert len(values)==12
   elif typ=='CROW_WING_EFFORT':target=d['wing'];values=list(map(float,z[4:]));assert len(values)==14
   elif typ=='CROW_KIN_STATE':target=d['kin'];assert z[4:6]==['41','40'];values=list(map(float,z[6:]));assert len(values)==81
   elif typ=='CROW_AERO_INPUT':target=d['input'];values=list(map(float,z[4:]));assert len(values)==81
   else:target=d['ablate'] if typ=='CROW_AERO_ABLATE' else d['wrench'];key+=(int(z[4]),);values=list(map(float,z[5:]));assert len(values)==6
   assert key not in target and all(map(math.isfinite,values)),(typ,key);target[key]=values
  elif typ=='CROW_AERO_ABLATE_CONFIG':
   key=tuple(map(int,z[1:3]));assert key not in d['config'];d['config'][key]=list(map(float,z[3:]));assert len(d['config'][key])==7
  elif typ in ('CROW_SUBSTEP_ERROR','CROW_AERO_OBSERVER_ERROR'):d['errors'].append(line.strip())
for e in range(8):
 for typ in ('aero','wing','kin','input'):
  assert sum((e,t,k) in d[typ] for t in range(348,401) for k in range(16))==53*16,(typ,e)
 assert sum((e,t,k,b) in d['wrench'] for t in range(348,401) for k in range(16) for b in (0,5,6,7))==53*16*4
assert not d['errors']
assert len(d['config'])==53*16
assert all(x[0]==1 and x[1]!=0 and x[2]!=0 for x in d['config'].values())
assert len({tuple(x) for x in d['config'].values()})==1
assert len(d['ablate'])==8*53*16*4
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
 energies={b:0. for b in (0,5,6,7)};ablated={b:0. for b in (0,5,6,7)};act=0.
 for i,key in enumerate(keys):
  if key[0]!=e:continue
  for n,b in enumerate((0,5,6,7)):
   wrench=d['wrench'][key+(b,)];twist=out[i][6*n:6*n+6];energies[b]+=sum(x*y for x,y in zip(wrench,twist))*.00125;ablated[b]+=sum(x*y for x,y in zip(d['ablate'][key+(b,)],twist))*.00125
  w=d['wing'][key];act+=sum(w[j]*(w[j+3]-w[j+2]) for j in (0,7))
 rows.append({'lane':e,'aero_signed_work_j_by_body':energies,'aero_signed_work_total_j':sum(energies.values()),'ablation_work_total_j':sum(ablated.values()),'unsteady_delta_work_j':sum(energies.values())-sum(ablated.values()),'ablation_work_j_by_body':ablated,'wing_effort_displacement_proxy_j':act})
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();result={'valid':True,'kernel_input_vs_kin_max_abs':timing,'omitted_body_force_max_abs_n':otherforce,'uniform_translation_error_m_s':err,'actual_dispatch_config':next(iter(d['config'].values())), 'substep_s':.00125,'window_ticks':[376,400],'rows':rows,'hashes':{'new_log':h(p/'run.stderr.gz'),'fk_binary':h(exe),'fk_library':h(lib)},'scope':'Signed work of modeled wrenches with substep-start COM twists in training seed, ablation is same-state diagnostic, not complete energy balance or biological measurement. Wing effort times observed displacement is a discrete proxy, not independently measured muscle work. Correlated lanes. Wind configuration not assumed.'};(A/'ablation-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
