#include "metalrobo/HumanBehaviorProgram.hpp"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
using namespace metalrobo;
int main(){try{
    HumanBehaviorProgramSource s;s.bodyCount=3;s.nq=9;s.nv=8;s.task=0;s.timestepNanoseconds=25000;
    for(auto& h:s.sourceSHA256)h.fill(1);
    for(unsigned i=0;i<3;++i){s.bodies[i].sourceBodyRecordIndex=i;s.bodies[i].coreBodyIndex=i;}
    s.bodies[0].sourceOriginInOriginalCOMFrame={.1,.2,-.3};
    s.numerical={1000000,0,0,0,0,1,1,0,0,0,0,1,.7,.3,.1,0};s.forbiddenBodies={{{0,0}}};
    HumanBehaviorCompileBinding b;b.bodyCount=3;b.nq=9;b.nv=8;b.timestepNanoseconds=25000;
    b.sourceArchiveSHA256=s.sourceSHA256[0];b.rigidSHA256=s.sourceSHA256[1];b.sourceToCore={{{0,0}},{{1,1}},{{2,2}}};
    b.cookedCOMOffset={{{.01,0,0}},{{0,0,0}},{{0,0,0}}};
    std::string error;CompiledHumanBehaviorProgram cooked;
    auto require=[](bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);};
    require(compileHumanBehaviorProgram(s,b,cooked,error),error.c_str());
    const auto fingerprint=cooked.gpu.fingerprint;
    require(std::abs(double(cooked.gpu.rootOriginHigh.x)+cooked.gpu.rootOriginLow.x-.09)<1e-15,"actual cooked source-origin rebase lost");
    require(!cooked.fullBehaviorEvidenceSupported,"missing audit coverage promoted");
    unsigned negative=0;
    auto deny=[&](auto source,auto binding){auto before=cooked.gpu.fingerprint;require(!compileHumanBehaviorProgram(source,binding,cooked,error),"invalid program admitted");require(cooked.gpu.fingerprint==before,"failed compile mutated output");++negative;};
    {auto x=s;x.sourceSHA256[0][0]=2;deny(x,b);}{auto x=b;x.rigidSHA256[0]=2;deny(s,x);}
    {auto x=s;x.nv++;deny(x,b);}{auto x=s;x.timestepNanoseconds=12500;deny(x,b);}
    {auto x=s;x.bodies[0].sourceBodyRecordIndex=999;deny(x,b);}{auto x=s;x.bodies[0].coreBodyIndex=1;deny(x,b);}
    {auto x=b;x.sourceToCore.push_back(x.sourceToCore[0]);deny(s,x);}{auto x=b;x.cookedCOMOffset.pop_back();deny(s,x);}
    {auto x=b;x.cookedCOMOffset[0][0]=NAN;deny(s,x);}{auto x=s;x.numerical[0]=INFINITY;deny(x,b);}
    {auto x=s;x.numerical[0]=std::numeric_limits<double>::max();deny(x,b);}
    {auto x=s;x.numerical[12]=1e-300;deny(x,b);}{auto x=s;x.numerical[12]=-.1;deny(x,b);}
    {auto x=s;x.numerical[13]=4;deny(x,b);}{auto x=s;x.numerical[3]=1;deny(x,b);}
    {auto x=s;x.numerical[6]=0;x.numerical[8]=1;deny(x,b);}
    {auto x=s;x.bodies[1].sourceInertialQuaternionBody[3]=0;deny(x,b);}
    {auto x=s;x.forbiddenBodies.push_back(x.forbiddenBodies[0]);deny(x,b);}
    {auto x=s;x.forbiddenBodies.clear();deny(x,b);}{auto x=s;x.sourceSHA256[5].fill(0);deny(x,b);}
    {auto x=s;x.task=3;deny(x,b);}{auto x=s;x.numerical[15]=1;deny(x,b);}
    {auto x=s;x.task=2;x.numerical[15]=1;require(compileHumanBehaviorProgram(x,b,cooked,error),error.c_str());require(cooked.gpu.fingerprint!=fingerprint,"task change not bound");}
    {auto x=b;x.cookedCOMOffset[0][0]=.02;require(compileHumanBehaviorProgram(s,x,cooked,error),error.c_str());require(cooked.gpu.fingerprint!=fingerprint,"actual COM rebase not bound");}
    {auto x=s;auto y=b;x.timestepNanoseconds=y.timestepNanoseconds=12500;require(compileHumanBehaviorProgram(x,y,cooked,error),error.c_str());}
    HumanBehaviorProgramSource decoded;decoded.bodyCount=99;std::vector<std::byte> bad(600);
    require(!decodeHumanBehaviorProgram(bad,decoded,error)&&decoded.bodyCount==99,"bad source decode mutated output");
    std::printf("human_behavior_program=pass positive=4 negative=%u source_origin_rebase=pass partial_coverage=explicit exact_ns=pass physical_steps=0\n",negative+1);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"human_behavior_program=fail %s\n",e.what());return 1;}}
