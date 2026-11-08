from pathlib import Path
root=Path('/Users/numi/numi-lab-reconciliation-20261108')
out=root/'docs/evidence/frictionless-active-contact-20261108'
s=(root/'docs/evidence/frictionless-oblique-contact-20261108/probe.cpp').read_text()
def rep(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
rep('for(int testcase=0;testcase<4;testcase++) {','for(float angle : {-0.37f,0.29f}) for(float speed : {-0.4f,-1.0f}) for(int iterations : {2,32}) for(int testcase=0;testcase<3;testcase++) {')
rep('constexpr float hx=.2f,hy=.02f,hz=.03f, angle=.3f, gap=.0001f;','constexpr float hx=.2f,hy=.02f,hz=.03f, gap=.0001f;')
rep('m.defaultQ[1]=hx*std::sin(angle)+hy*std::cos(angle)+gap;','m.defaultQ[1]=hx*std::abs(std::sin(angle))+hy*std::abs(std::cos(angle))+gap;')
rep('''  if(testcase==1)m.defaultV[0]=1.f;
  if(testcase==2)m.defaultV[0]=-1.f;
  if(testcase==3)m.defaultV[1]=-1.f;''','''  m.defaultV[0]=float(testcase-1); m.defaultV[1]=speed;''')
rep('cfg.velocityIterations=2;','cfg.velocityIterations=iterations;')
rep('''std::cout<<"{\\"case\\":"<<testcase<<",\\"success\\":"''','''std::cout<<"{\\"angle\\":"<<angle<<",\\"speed\\":"<<speed<<",\\"iterations\\":"<<iterations<<",\\"case\\":"<<testcase<<",\\"success\\":"''')
(out/'probe.cpp').write_text(s)
