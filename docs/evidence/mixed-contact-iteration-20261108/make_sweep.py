from pathlib import Path
p=Path('/Users/numi/frictionless-mixed-rod-20261108');s=(p/'probe.cpp').read_text()
def rep(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
rep('for(float mu:{.5f,0.f}) for(float vn:{-1.f,0.f,1.f}){','for(int iterations:{1,2,4,8,16,32,64}) for(float mu:{.5f,0.f}) for(float vn:{-1.f,0.f,1.f}){')
rep('cfg.velocityIterations=2;','cfg.velocityIterations=iterations;')
rep('''std::cout<<"{\\"mu\\":"<<mu''','''std::cout<<"{\\"iterations\\":"<<iterations<<",\\"mu\\":"<<mu''')
(p/'sweep_probe.cpp').write_text(s)
