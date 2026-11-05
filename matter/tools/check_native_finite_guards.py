#!/usr/bin/env python3
"""Offline guard-order oracle; never linked into the simulation update path.

Lift the same scalar Metal arithmetic into two C++ namespaces and compare old
and new lowering, including invalid-input flags and partial expression-cache
side effects. Native Metal complete-step parity remains a separate requirement.
"""
from pathlib import Path
import argparse, re, subprocess, json, hashlib
ap=argparse.ArgumentParser()
ap.add_argument("before",type=Path)
ap.add_argument("after",type=Path)
ap.add_argument("--include",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
def programs(path):
    text=path.read_text()
    found={}
    for match in re.finditer(r"template <typename CachePointer>\ninline float (nmNative(?:Direct)?P\d+)\(",text):
        first=text.index("{",match.end());depth=1;end=first+1
        while depth:
            depth += (text[end]=="{")-(text[end]=="}")
            end+=1
        found[match.group(1)]=text[match.start():end]
    return found
old,new=programs(a.before),programs(a.after)
assert old.keys()==new.keys() and old
pmax=max(int(x) for src in old.values() for x in re.findall(r"p\[(\d+)\]",src))+1
cmax=max(int(x) for src in old.values() for x in re.findall(r"c\[(\d+)\]",src))+1
pmax=max(16,pmax);cmax=max(16,cmax)
head=r"""
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include "numi/matter/fiber_exp_linear.h"
#define device
#define thread
using uint=unsigned;
using std::isfinite;using std::abs;using std::min;using std::max;
using std::clamp;using std::exp;using std::log;using std::sqrt;
template<typename T> T as_type(std::uint32_t x){return std::bit_cast<T>(x);}
constexpr float kGeometryFloor=1e-12f;
"""
body=head
for namespace,src in [("before",old),("after",new)]:
    body+="\nnamespace "+namespace+" {\n"+"\n".join(src.values())+"\n}\n"
body+="""
using Fn=float(*)(const float*,const float*,const float*,const float*,float,float,float*,bool&,bool&);
struct Entry{const char* name;Fn old;Fn now;};
Entry functions[]={
"""
for name in old:
    body+='{"'+name+'",&before::'+name+'<float*>,&after::'+name+'<float*>},\n'
body+="};\n"
body+=r"""
int main(){
constexpr unsigned pc=PARAM_COUNT,cc=CACHE_COUNT;
unsigned checks=0,accepted=0,rejected=0,nonfinite=0;
const float bad[]={std::numeric_limits<float>::quiet_NaN(),
    INFINITY,-INFINITY,std::numeric_limits<float>::max(),
    -std::numeric_limits<float>::max(),1e-35f,-1e-35f,0.0f};
for(const Entry& fn:functions)for(unsigned trial=0;trial<128;++trial){
    std::array<float,pc> p;std::array<float,cc> ca,cb;
    std::array<float,9> f{},d{},r{};
    for(unsigned i=0;i<pc;++i)p[i]=1.01f+float((i+trial)%7)*.013f;
    for(unsigned i=0;i<cc;++i)ca[i]=.01f+float((i+trial)%11)*.001f;
    for(unsigned i=0;i<9;++i){
        f[i]=(i%4==0?1.0f:0.0f)+float(int((i+trial)%7)-3)*.001f;
        d[i]=float(int((i+trial)%5)-2)*.01f;
        r[i]=float(int((i+trial)%3)-1)*.001f;
    }
    float dt=1e-4f,temp=310.0f;
    if(trial>=16){
        const float poison=bad[(trial/7)%8];
        switch(trial%7){
        case 0:f[(trial/7)%9]=poison;break;
        case 1:d[(trial/7)%9]=poison;break;
        case 2:r[(trial/7)%9]=poison;break;
        case 3:p[(trial/7)%pc]=poison;break;
        case 4:ca[(trial/7)%cc]=poison;break;
        case 5:dt=poison;break;
        case 6:temp=poison;break;
        }
    }
    cb=ca;bool va=true,vb=true,na=false,nb=false;
    const float x=fn.old(p.data(),f.data(),d.data(),r.data(),dt,temp,ca.data(),va,na);
    const float y=fn.now(p.data(),f.data(),d.data(),r.data(),dt,temp,cb.data(),vb,nb);
    if(va!=vb||na!=nb||std::bit_cast<uint>(x)!=std::bit_cast<uint>(y)||
       std::memcmp(ca.data(),cb.data(),sizeof(ca))!=0){
        std::cerr<<"FAIL native guard oracle program="<<fn.name<<" trial="<<trial
            <<" valid="<<va<<","<<vb<<" nonfinite="<<na<<","<<nb<<"\n";return 1;
    }
    ++checks;accepted+=va;rejected+=!va;nonfinite+=na;
}
std::cout<<"PASS native scalar finite-guard oracle programs="
    <<std::size(functions)<<" cases="<<checks<<" valid="<<accepted
    <<" rejected="<<rejected<<" nonfinite="<<nonfinite
    <<" exact_return_flags_and_cache=true host_reference_only=true\n";
}
""".replace("PARAM_COUNT",str(pmax)).replace("CACHE_COUNT",str(cmax))
cpp=a.output/"finite-guard-oracle.cpp";cpp.write_text(body)
exe=a.output/"finite-guard-oracle"
subprocess.run(["/usr/bin/clang++","-O0","-std=c++20","-ffp-contract=off",
                "-I"+str(a.include),str(cpp),"-o",str(exe)],check=True)
subprocess.run([str(exe)],check=True)
print(json.dumps({"before_sha256":hashlib.sha256(a.before.read_bytes()).hexdigest(),
 "after_sha256":hashlib.sha256(a.after.read_bytes()).hexdigest(),
 "programs":len(old),"parameter_slots":pmax,"cache_slots":cmax,
 "scope":"offline guard semantics; native GPU parity required separately"}))
