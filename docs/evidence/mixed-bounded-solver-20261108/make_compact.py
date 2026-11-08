from pathlib import Path
s=Path('isolation.cpp').read_text()
a='template<class T>std::vector<T> twice(std::span<const T> s){std::vector<T>v(s.begin(),s.end());v.insert(v.end(),s.begin(),s.end());return v;}'
b='template<class T>std::vector<T> twice(std::span<const T> s){std::vector<T>v;for(int i=0;i<4;++i)v.insert(v.end(),s.begin(),s.end());return v;}'
assert s.count(a)==1;s=s.replace(a,b)
a='for(size_t i=nc;i<nodes.size();++i)nodes[i].position.y+=.5f;std::vector<float>effort(2*world.nv(),0);'
b='for(size_t i=0;i<nodes.size();++i)nodes[i].position.y+=.5f;std::vector<float>effort(4*world.nv(),0);'
assert s.count(a)==1;s=s.replace(a,b)
a='.environmentCount=2,.controlStepCount=1';b='.environmentCount=4,.controlStepCount=1';assert s.count(a)==1;s=s.replace(a,b)
a='out[i].statuses.size()<2||out[i].contactStatuses.size()<2';b='out[i].statuses.size()<4||out[i].contactStatuses.size()<4';assert s.count(a)==1;s=s.replace(a,b)
a='eqenv(out[0],out[i],1)?';b='(eqenv(out[0],out[i],0)&&eqenv(out[0],out[i],1)&&eqenv(out[0],out[i],2)&&eqenv(out[0],out[i],3))?';assert s.count(a)==1;s=s.replace(a,b)
a='for(int e=0;e<2;e++)';b='for(int e=0;e<4;e++)';assert s.count(a)==1;s=s.replace(a,b)
Path('compact.cpp').write_text(s)
