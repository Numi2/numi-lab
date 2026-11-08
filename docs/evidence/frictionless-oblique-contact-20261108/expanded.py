from pathlib import Path
p=Path('/Users/numi/ccd-frictionless-oblique-20261108');s=(p/'probe.cpp').read_text()
def rep(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
rep('for(int testcase=0;testcase<4;testcase++) {','for(float angle : {-0.55f,-0.16f,0.13f,0.49f}) for(float mu : {0.0f,0.6f}) for(int testcase=0;testcase<5;testcase++) {')
rep('for(auto &mat:m.materials) {mat.friction={};mat.response={};}','for(auto &mat:m.materials) {mat.friction={mu,mu,0,0};mat.response={};}')
rep('constexpr float hx=.2f,hy=.02f,hz=.03f, angle=.3f, gap=.0001f;','constexpr float hx=.2f,hy=.02f,hz=.03f; const float gap=testcase==4 ? .01f : .0001f;')
rep('m.defaultQ[1]=hx*std::sin(angle)+hy*std::cos(angle)+gap;','m.defaultQ[1]=hx*std::abs(std::sin(angle))+hy*std::abs(std::cos(angle))+gap;')
rep('''  if(testcase==1)m.defaultV[0]=1.f;
  if(testcase==2)m.defaultV[0]=-1.f;
  if(testcase==3)m.defaultV[1]=-1.f;''','''  if(testcase==0 || testcase==4)m.defaultV[0]=1.f;
  if(testcase==1)m.defaultV[0]=-1.f;
  if(testcase==2)m.defaultV[2]=1.f;
  if(testcase==3)m.defaultV[2]=-1.f;''')
rep('''std::cout<<"{\\"case\\":"<<testcase<<",\\"success\\":"''','''std::cout<<"{\\"angle\\":"<<angle<<",\\"mu\\":"<<mu<<",\\"case\\":"<<testcase<<",\\"success\\":"''')
(p/'expanded.cpp').write_text(s)
