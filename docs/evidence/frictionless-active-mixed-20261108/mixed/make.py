from pathlib import Path
s=Path('/Users/numi/numi-lab-ccd-thin-evidence-20261107/cold-four-sliding-broad-v1/mixed_rod.cpp').read_text()
def rep(a,b):
 global s
 assert s.count(a)==1,(a[:70],s.count(a));s=s.replace(a,b)
rep('for(float vn:{-1.f,0.f,1.f}){','for(float mu:{.5f,0.f}) for(float vn:{-1.f,0.f,1.f}){')
rep('x.friction={.5f,.5f,0,0};','x.friction={mu,mu,0,0};')
rep('''std::cout<<"{\\"initial_vn\\":"<<vn''','''std::cout<<"{\\"mu\\":"<<mu<<",\\"initial_vn\\":"<<vn''')
rep('''auto& md=a.contactEvidence.contactMetadata.at(i);''','''auto& md=a.contactEvidence.contactMetadata.at(i);auto& cone=a.contactEvidence.evaluatedCones.at(i);''')
rep('''<<"],\\"target\\":"<<r.targetVelocity''','''<<"],\\"cone\\":["<<cone.effectiveFrictionU<<","<<cone.effectiveFrictionV<<"],\\"target\\":"<<r.targetVelocity''')
rep('''  std::cout<<"]}";''','''  std::cout<<"],\\"final_v\\":[";
  for(size_t i=0;i<a.finalV.size();i++){if(i)std::cout<<",";std::cout<<a.finalV[i];}
  std::cout<<"],\\"rod_v\\":[";
  for(size_t i=0;i<a.finalRodNodes.size();i++){if(i)std::cout<<",";const auto &v=a.finalRodNodes[i].velocity;std::cout<<"["<<v.x<<","<<v.y<<","<<v.z<<"]";}
  std::cout<<"]}";''')
Path('/Users/numi/frictionless-mixed-rod-20261108/probe.cpp').write_text(s)
