from pathlib import Path
import subprocess,json
R=Path.cwd();A=R/'artifacts/crow-v15-aba-existing-model-controls-20261111';B=R/'build/crow-v15-aba-existing-model-controls-20261111';P=R/'build/crow-v15-aba-40-body-isolation-20261111';ml=R/'build/crow-v15-aba-floating-gravity-screen-20261111/aba.metallib';res=[]
for name,app in [('g1','articulated_aba_host_probe.cpp'),('franka','franka_engine_model_probe.cpp'),('psm','surgical_psm_probe.cpp')]:
 s=(R/'apps'/app).read_text();count=0
 for var in ('context','transient','metalContext'):
  old='metalrobo::MetalArticulatedABAContext '+var+';';new='metalrobo::MetalArticulatedABAContext '+var+'(metalrobo::MetalArticulatedABAConfig{.metallibPath="'+str(ml)+'"});';count+=s.count(old);s=s.replace(old,new)
 assert count==(2 if name=='g1' else 1),(name,count);src=A/(name+'.cpp');src.write_text(s);exe=B/name;compileResult=subprocess.run(['xcrun','clang++','-std=c++20','-O2','-I'+str(P/'include'),'-Iinclude',str(src),'-L'+str(P),'-lmetalrobo','-Wl,-rpath,'+str(P),'-o',str(exe)],capture_output=True,text=True);(A/(name+'-build.log')).write_text(compileResult.stdout+compileResult.stderr)
 if compileResult.returncode:res.append({'case':name,'build_exit':compileResult.returncode});continue
 with (A/(name+'.stdout')).open('wb') as out,(A/(name+'.stderr')).open('wb') as err:r=subprocess.run([str(exe)],stdout=out,stderr=err)
 res.append({'case':name,'build_exit':0,'exit':r.returncode});print(name,'exit',r.returncode,flush=True)
(A/'summary.json').write_text(json.dumps(res,indent=2)+'\n')
