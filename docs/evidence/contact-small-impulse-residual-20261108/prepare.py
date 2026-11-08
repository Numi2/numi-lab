from pathlib import Path
root=Path('/Users/numi/numi-lab-reconciliation-20261108')
p=root/'docs/evidence/contact-small-impulse-residual-20261108'
s=(root/'docs/evidence/frictionless-active-mixed-20261108/active/probe.cpp').read_text()
def rep(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
rep('for(float angle : {0.3f,-0.41f}) for(int iterations : {2,8,32}) for(int testcase=0;testcase<3;testcase++) {','for(float mass : {1.e-7f,1.e-6f,1.f}) for(float angle : {-0.37f,0.29f}) for(int iterations : {2,32}) for(int testcase=0;testcase<3;testcase++) {')
rep('  auto &b=m.bodies[1];const float ix=(hy*hy+hz*hz)/3.f,iy=(hx*hx+hz*hz)/3.f,iz=(hx*hx+hy*hy)/3.f;','  auto &b=m.bodies[1];b.massAndInverseMass={mass,1.f/mass,0,0};const float ix=mass*(hy*hy+hz*hz)/3.f,iy=mass*(hx*hx+hz*hz)/3.f,iz=mass*(hx*hx+hy*hy)/3.f;')
rep('<<"{\\"angle\\":"<<angle','<<"{\\"mass\\":"<<mass<<",\\"angle\\":"<<angle')
(p/'probe.cpp').write_text(s)
