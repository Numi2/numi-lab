from pathlib import Path
import subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-balance-response-20261110';B=R/'build/crow-v15-balance-response-20261110';s=(R/'artifacts/crow-v15-sat-hybrid-catch-20261110/probe.cpp').read_text()
pairs=[('for(unsigned cs=0;cs<4;cs++) for(unsigned arm=0;arm<3;arm++) try {','for(unsigned arm=0;arm<5;arm++) try { unsigned cs=0;'),('m.profile.controlSteps=160;','m.profile.controlSteps=80;'),('artifacts/crow-v15-aligned-catch-probe-20261110/reached-input.txt','artifacts/crow-v15-balance-response-20261110/reached-input.txt'),('q[0]+=(cs&1)? .003f:-.003f;v[2]*=(cs&2)?1.25f:.75f;',''),('for(unsigned t=0;t<160;t++)','for(unsigned t=0;t<80;t++)')]
for a,b in pairs:assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
start=s.index('if(d.vIndex>=nv-8){');end=s.index('commands[d.vIndex]=target;',start)
s=s[:start]+'''if(d.qIndex==14u || d.qIndex==17u) { const float delta[5]={-.06f,-.03f,0.f,.03f,.06f};target+=delta[arm];}'''+s[end:]
start=s.index(' std::cout<<"trace "');end=s.index('\n }std::cout<<"case_result "',start)
s=s[:start]+''' std::cout<<"trace "<<arm<<" "<<t;for(unsigned k=0;k<7;k++)std::cout<<" "<<out.finalQ[k];for(unsigned k=0;k<6;k++)std::cout<<" "<<out.finalV[k];std::cout<<" "<<out.finalQ[14]<<" "<<out.finalQ[17]<<std::endl;'''+s[end:]
s=s.replace('<<cs<<','<<0<<')
(A/'probe.cpp').write_text(s)
subprocess.run(['xcrun','clang++','-std=c++20','-O2','-Iinclude','-Iartifacts/crow-v15-wing-inertia-prototype-20261110',str(A/'probe.cpp'),'-Lbuild/crow-v15-ccd-budget-20261110','-lmetalrobo','-Wl,-rpath,@loader_path/../crow-v15-ccd-budget-20261110','-o',str(B/'probe')],check=True)
