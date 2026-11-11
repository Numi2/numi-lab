from pathlib import Path
import subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-aba-varied-state-screen-20261111';B=R/'build/crow-v15-aba-varied-state-screen-20261111';P=R/'build/crow-v15-aba-40-body-isolation-20261111'
for name,extra in [('diagnostic',16),('control',8)]:
 s=(R/('artifacts/crow-v15-aba-math-mode-comparison-20261111/'+name+'.cpp')).read_text();s=s.replace('const unsigned n=12,nq=', 'const unsigned n=96,nq=')
 old='''for(unsigned k=0;k<n;k++) { q.insert(q.end(),b.defaultQ.begin(),b.defaultQ.end()); v.insert(v.end(),nv,0); e.insert(e.end(),nv,0);'''
 new='''for(unsigned k=0;k<n;k++) { q.insert(q.end(),b.defaultQ.begin(),b.defaultQ.end()); v.insert(v.end(),nv,0); e.insert(e.end(),nv,0);
    const float yaw=.013f*float(int(k%13)-6); q[k*nq+5]=std::sin(yaw*.5f);q[k*nq+6]=std::cos(yaw*.5f);
    for(unsigned j=0;j<nv;j++){v[k*nv+j]=.05f*std::sin(.31f*float(k)+.37f*float(j));if(j>=6u)e[k*nv+j]=.00001f*std::cos(.21f*float(k)+.19f*float(j));}'''
 assert s.count(old)==1;s=s.replace(old,new);s=s.replace('"gpu_lanes\\":12','"gpu_lanes\\":96');src=A/(name+'.cpp');src.write_text(s);exe=B/name;subprocess.run(['xcrun','clang++','-std=c++20','-O2','-I'+str(P/'include'),'-Iinclude',str(src),'-L'+str(P),'-lmetalrobo','-Wl,-rpath,'+str(P),'-o',str(exe)],check=True)
 with (A/(name+'.json')).open('wb') as out,(A/(name+'.stderr')).open('wb') as err:r=subprocess.run([str(exe)],stdout=out,stderr=err)
 print(name,'exit',r.returncode,flush=True)
