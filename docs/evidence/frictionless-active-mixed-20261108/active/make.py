from pathlib import Path
s=Path('/Users/numi/ccd-frictionless-oblique-20261108/probe.cpp').read_text()
def rep(a,b):
 global s
 assert s.count(a)==1,(a[:40],s.count(a));s=s.replace(a,b)
rep('for(int testcase=0;testcase<4;testcase++) {','for(float angle : {0.3f,-0.41f}) for(int iterations : {2,8,32}) for(int testcase=0;testcase<3;testcase++) {')
rep('constexpr float hx=.2f,hy=.02f,hz=.03f, angle=.3f, gap=.0001f;','constexpr float hx=.2f,hy=.02f,hz=.03f, gap=.0001f;')
rep('m.defaultQ[1]=hx*std::sin(angle)+hy*std::cos(angle)+gap;','m.defaultQ[1]=hx*std::abs(std::sin(angle))+hy*std::abs(std::cos(angle))+gap;')
rep('''  if(testcase==1)m.defaultV[0]=1.f;
  if(testcase==2)m.defaultV[0]=-1.f;
  if(testcase==3)m.defaultV[1]=-1.f;''','''  m.defaultV[0]=float(testcase-1);
  m.defaultV[1]=-1.f;''')
rep('cfg.velocityIterations=2;cfg.finalVelocityIterations=1;','cfg.velocityIterations=iterations;cfg.finalVelocityIterations=0;')
rep('''std::cout<<"{\\"case\\":"<<testcase<<",\\"success\\":"''','''std::cout<<"{\\"angle\\":"<<angle<<",\\"iterations\\":"<<iterations<<",\\"case\\":"<<testcase<<",\\"initial_y\\":"<<m.defaultQ[1]<<",\\"success\\":"''')
rep('''  std::cout<<"]}";''','''  std::cout<<"],\\"solver_residual\\":"<<(r.contactStatuses.empty() ? -1 : r.contactStatuses[0].residuals.y)<<",\\"rows\\":[";
  unsigned count=r.contactStatuses.empty()?0:r.contactStatuses[0].activeContacts;
  if (count>r.contactEvidence.contacts.size() || 3*count>r.contactEvidence.evaluatedRows.size()) return 4;
  for(unsigned i=0;i<count;++i) {
   if(i)std::cout<<",";const auto &c=r.contactEvidence.contacts[i];const auto &e=r.contactEvidence.evaluatedRows[3*i];
   std::cout<<"{\\"bodyA\\":"<<c.bodyA<<",\\"bodyB\\":"<<c.bodyB<<",\\"point\\":["<<c.pointAndSeparation.x<<","<<c.pointAndSeparation.y<<","<<c.pointAndSeparation.z<<"],\\"direction\\":["<<e.direction.x<<","<<e.direction.y<<","<<e.direction.z<<"],\\"lambda\\":"<<c.impulses.x<<",\\"tangent\\":["<<c.impulses.y<<","<<c.impulses.z<<"],\\"target\\":"<<e.targetVelocity<<",\\"reg\\":"<<e.regularization<<"}";
  }
  std::cout<<"]}";''')
Path('/Users/numi/frictionless-active-normal-20261108/probe.cpp').write_text(s)
