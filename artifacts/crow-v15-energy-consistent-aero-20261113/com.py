from pathlib import Path
import gzip,json,subprocess,os,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-energy-consistent-aero-20261113';data={};inputs=[];keys=[]
for arm in ('push','hold'):
 p=A/(arm+'-wing-20261126-rebound-passive')/'run.stderr.gz';state={}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
    if 350<=t<=400:
     assert (e,t) not in state;state[e,t]=x
 assert len(state)==8*51
 for e in range(8):
  for t in range(350,401):
   x=state[e,t];keys.append((arm,e,t));inputs.append(str(len(keys)-1)+' 0 '+' '.join(map(str,x['q']+x['v'])))
exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib';r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)})
for l in r.stdout.splitlines():
 z=l.split();assert z[0]=='C';i=int(z[1]);v=list(map(float,z[3:]));assert len(v)==7;data[keys[i]]=v
assert len(data)==len(keys)
rows=[]
for e in range(8):
 row={'lane':e}
 for arm in ('push','hold'):
  base=data[arm,e,350];peak=max(data[arm,e,t][3]-base[3] for t in range(351,401));row[arm]={'com_peak_rise_m':peak,'com_z_350_m':base[3],'com_z_400_m':data[arm,e,400][3],'com_vz_400_m_s':data[arm,e,400][6]}
 rows.append(row)
(A/'com-analysis.json').write_text(json.dumps({'rows':rows,'kinematics_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'scope':'Training seed; COM from modeled body FP64 kinematics, no qualification.'},indent=2)+'\n');print(json.dumps(rows,indent=2))
