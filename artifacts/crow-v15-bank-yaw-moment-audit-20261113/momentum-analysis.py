from pathlib import Path
import os,subprocess,json,runpy,statistics,hashlib
D=Path("artifacts/crow-v15-bank-yaw-moment-audit-20261113");g=runpy.run_path(str(D/"analyze.py"));keys=g["keys"];kin=g["kin"];wrench=g["wrench"];base=g["inputs"];exe=Path("build/crow-v15-bank-yaw-moment-audit-20261113/momentum");fk=Path("build/crow-v15-bank-yaw-moment-audit-20261113/kinematics");lib=Path("build/crow-v15-mirrored-leg-isolation-20261113");qv=kin[keys[0]];shift=qv[:];shift[41]+=.3;shift[42]-=.2;shift[43]+=.1;zero=qv[:41]+[0.]*40;extras="\n"+str(len(keys))+" 0 "+" ".join(map(str,shift))+"\n"+str(len(keys)+1)+" 0 "+" ".join(map(str,zero))+"\n";env={**os.environ,"DYLD_LIBRARY_PATH":str(lib.resolve())};hres=subprocess.run([str(exe)],input=base+extras,text=True,capture_output=True,check=True,env=env);fres=subprocess.run([str(fk)],input=base+"\n",text=True,capture_output=True,check=True,env=env);hs={}
for l in hres.stdout.splitlines():
 z=l.split();assert z[0]=="H";hs[int(z[1])]=list(map(float,z[3:]))
assert len(hs)==len(keys)+2;trans=max(abs(a-b) for a,b in zip(hs[0],hs[len(keys)]));zeroerr=max(map(abs,hs[len(keys)+1]));assert trans<1e-8 and zeroerr<1e-12,(trans,zeroerr)
data={6:[],7:[]}
for l in fres.stdout.splitlines():
 z=l.split();i=int(z[1]);v=list(map(float,z[3:]));key=keys[i];com=v[1:4];positions={body:v[7+3*j:10+3*j] for j,body in enumerate((0,5,6,7))};tau=0.
 for body in (0,5,6,7):
  w=wrench[key+(body,)];p=positions[body];tau+=w[5]+(p[0]-com[0])*w[1]-(p[1]-com[1])*w[0]
 data[key[0]].append((key,hs[i][2],tau))
rows=[]
for e,a in data.items():
 assert len(a)==8016;measured=a[-1][1]-a[0][1];aero=sum(x[2] for x in a[:-1])*.00125;rows.append({"lane":e,"sample_start":a[0][0][1:],"sample_end":a[-1][0][1:],"initial_world_z_angular_momentum":a[0][1],"final_world_z_angular_momentum":a[-1][1],"delta_world_z_angular_momentum":measured,"integrated_observed_aero_yaw_impulse":aero,"difference_delta_momentum_minus_observed_aero_impulse":measured-aero,"scope":"Difference can include other external wrenches, contact/constraint effects, numerical integration and observer convention; not proof of solver error."})
out={"rows":rows,"velocity_translation_invariance_max_error":trans,"zero_velocity_momentum_max_error":zeroerr,"momentum_binary_sha256":hashlib.sha256(exe.read_bytes()).hexdigest(),"scope":"Posthoc modeled angular momentum about whole-body COM and observed aerodynamic moment only; used seed, not a full external torque audit or biological qualification."};(D/"momentum-analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
