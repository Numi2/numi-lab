from pathlib import Path
import subprocess,json
R=Path.cwd();A=R/'artifacts/crow-v15-aba-gravity-wrench-screen-20261111';B=R/'build/crow-v15-aba-gravity-wrench-screen-20261111';P=R/'build/crow-v15-aba-40-body-isolation-20261111';runs=[]
for n in ('diagnostic','control'):
 base=(R/('artifacts/crow-v15-aba-floating-gravity-screen-20261111/'+n+'.cpp')).read_text()
 for gi,g in enumerate([(0.,0.,-9.81),(0.,0.,0.),(2.75,-1.25,-7.25)]):
  for withWrench in (False,True):
   s=base;old='const unsigned n=96,nq=';new='b.world.gravityAndTimestep.x='+str(g[0])+'f;b.world.gravityAndTimestep.y='+str(g[1])+'f;b.world.gravityAndTimestep.z='+str(g[2])+'f; const unsigned n=96,nq=';assert s.count(old)==1;s=s.replace(old,new)
   old='MetalArticulatedABAInput in{.articulationIndex=0u,.environmentCount=n,.q=q,.v=v,.effort=e,.applyBodyDamping=false};'
   if withWrench:
    new='''std::vector<MRABABodyWrenchGPU> wrenches(n*b.bodies.size());for(unsigned k=0;k<n;k++)for(unsigned bi=0;bi<b.bodies.size();bi++)if(bi%7u==0u){auto& w=wrenches[k*b.bodies.size()+bi];w.force={.005f*std::sin(.3f*k+.2f*bi),.004f*std::cos(.2f*k+.1f*bi),.003f*std::sin(.1f*k+.5f*bi),0};w.torque={.0001f*std::cos(.4f*k),.00008f*std::sin(.3f*k),.00005f*std::cos(.2f*bi),0};} MetalArticulatedABAInput in{.articulationIndex=0u,.environmentCount=n,.q=q,.v=v,.effort=e,.bodyWrenches=wrenches,.applyBodyDamping=false};'''
    assert s.count(old)==1;s=s.replace(old,new)
    old='auto s=computeArticulatedForwardDynamics(b,0,cq,cv,ce,{},acc,config);';new='''std::vector<ArticulatedBodyWrench> cpuWrenches(b.bodies.size());for(unsigned bi=0;bi<b.bodies.size();bi++){const auto& w=wrenches[k*b.bodies.size()+bi];cpuWrenches[bi].force={double(w.force.x),double(w.force.y),double(w.force.z)};cpuWrenches[bi].torque={double(w.torque.x),double(w.torque.y),double(w.torque.z)};}auto s=computeArticulatedForwardDynamics(b,0,cq,cv,ce,cpuWrenches,acc,config);''';assert s.count(old)==1;s=s.replace(old,new)
   old='config.applyBodyDamping=false;';new='config.applyBodyDamping=false;config.gravity={double(b.world.gravityAndTimestep.x),double(b.world.gravityAndTimestep.y),double(b.world.gravityAndTimestep.z)};';assert s.count(old)==1;s=s.replace(old,new)
   tag=n+'-g'+str(gi)+'-w'+str(int(withWrench));src=B/(tag+'.cpp');src.write_text(s);exe=B/tag;subprocess.run(['xcrun','clang++','-std=c++20','-O2','-I'+str(P/'include'),'-Iinclude',str(src),'-L'+str(P),'-lmetalrobo','-Wl,-rpath,'+str(P),'-o',str(exe)],check=True)
   with (A/(tag+'.json')).open('wb') as out,(A/(tag+'.stderr')).open('wb') as err:r=subprocess.run([str(exe)],stdout=out,stderr=err)
   runs.append({'case':tag,'exit':r.returncode,'gravity':g,'wrenches':withWrench,'stderr':(A/(tag+'.stderr')).read_text().strip()});print(tag,'exit',r.returncode,flush=True)
(A/'summary.json').write_text(json.dumps(runs,indent=2)+'\n');print('cases',len(runs),'failed',sum(x['exit']!=0 for x in runs),flush=True)
