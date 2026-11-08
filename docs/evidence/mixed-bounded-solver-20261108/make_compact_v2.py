from pathlib import Path
s=Path('compact.cpp').read_text()
a='<<",\\\"env\\\":[";for(int e=0;e<4;e++)'
b='<<",\\\"repeated_env2\\\":"<<(eqenv(out[i],repeat[i],2)?"true":"false")<<",\\\"repeated_env3\\\":"<<(eqenv(out[i],repeat[i],3)?"true":"false")<<",\\\"env\\\":[";for(int e=0;e<4;e++)'
assert s.count(a)==1,s.count(a);s=s.replace(a,b);Path('compact-v2.cpp').write_text(s)
