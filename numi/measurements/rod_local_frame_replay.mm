#include "rod_local_frame.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numbers>
#include <vector>
using namespace metalrobo;

template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
bool sameState(const DiscreteElasticRodState& a,const DiscreteElasticRodState& b) {
    return same(a.positions,b.positions)&&same(a.velocities,b.velocities)&&same(a.twists,b.twists)&&same(a.twistRates,b.twistRates);
}
bool sameReaction(const DiscreteRodAttachmentReaction& a,const DiscreteRodAttachmentReaction& b) {
    return a.nodeIndex==b.nodeIndex && a.bodyIndex==b.bodyIndex &&
        std::memcmp(a.impulseOnTarget.data(),b.impulseOnTarget.data(),sizeof(a.impulseOnTarget))==0 &&
        std::memcmp(a.averageForceOnTarget.data(),b.averageForceOnTarget.data(),sizeof(a.averageForceOnTarget))==0 &&
        std::memcmp(&a.finalPositionError,&b.finalPositionError,sizeof(double))==0;
}
template<class T> void writeVector(std::ostream& out,const std::vector<T>& a) {
    const auto n=std::uint64_t(a.size());out.write(reinterpret_cast<const char*>(&n),sizeof(n));
    out.write(reinterpret_cast<const char*>(a.data()),std::streamsize(a.size()*sizeof(T)));
}
template<class T> bool readVector(std::istream& in,std::vector<T>& a,std::uint64_t expected) {
    std::uint64_t n=0;in.read(reinterpret_cast<char*>(&n),sizeof(n));
    if (!in || n!=expected) return false;
    a.resize(std::size_t(n));
    in.read(reinterpret_cast<char*>(a.data()),std::streamsize(a.size()*sizeof(T)));
    return bool(in);
}
bool save(const char* path,const std::vector<DiscreteElasticRodState>& states) {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    const std::array<char,8> magic={'R','O','D','T','E','S','T','1'};
    out.write(magic.data(),magic.size());std::uint64_t count=states.size();
    out.write(reinterpret_cast<const char*>(&count),sizeof(count));
    for (const auto& s:states) {writeVector(out,s.positions);writeVector(out,s.velocities);writeVector(out,s.twists);writeVector(out,s.twistRates);}
    out.close();return bool(out);
}
bool load(const char* path,std::vector<DiscreteElasticRodState>& dest) {
    std::ifstream in(path,std::ios::binary);std::array<char,8> magic{};std::uint64_t count=0;
    in.read(magic.data(),magic.size());in.read(reinterpret_cast<char*>(&count),sizeof(count));
    if(!in || magic!=std::array<char,8>{'R','O','D','T','E','S','T','1'} || count!=2) return false;
    std::vector<DiscreteElasticRodState> next(count);
    for (auto& s:next) {
        if(!readVector(in,s.positions,128)||!readVector(in,s.velocities,128)||
           !readVector(in,s.twists,127)||!readVector(in,s.twistRates,127)) return false;
        for(const auto& p:s.positions) for(double v:p) if(!std::isfinite(v)) return false;
        for(const auto& p:s.velocities) for(double v:p) if(!std::isfinite(v)) return false;
        for(double v:s.twists) if(!std::isfinite(v)) return false;
        for(double v:s.twistRates) if(!std::isfinite(v)) return false;
    }
    if(in.peek()!=std::char_traits<char>::eof()) return false;
    dest=std::move(next);return true;
}
int main(int argc,char** argv) {
    if(argc!=3) return 2;
    constexpr unsigned n=128; constexpr double rest=.11758755785232791,ea=2.3818718058345718;
    constexpr double radius=.000020,h=.1,speed=.0005;
    DiscreteRodMaterial mat;mat.radius=radius;mat.density=1100;mat.youngModulus=ea/(std::numbers::pi*radius*radius);
    const auto model=makeStraightSutureRod(n,rest,mat);const std::array<double,2> origin{0,16};
    std::vector<DiscreteElasticRodState> states;
    std::vector<DiscreteRodAttachment> at(4);
    for(unsigned e=0;e<2;++e) {
        auto x=makeDiscreteElasticRodDefaultState(model);
        for(unsigned i=0;i<n;++i) { x.positions[i][0]=origin[e]+.122*double(i)/(n-1);x.velocities[i][0]=speed*double(i)/(n-1); }
        states.push_back(x);at[e*2].nodeIndex=0;at[e*2+1].nodeIndex=n-1;at[e*2+1].targetVelocity={speed,0,0};
    }
    MetalDiscreteElasticRodConfig cfg;cfg.metallibPath=argv[1];cfg.step.timestep=h;cfg.step.gravity={0,0,0};
    cfg.step.linearDamping=0;cfg.step.twistDamping=0;cfg.step.solverIterations=128;cfg.step.constraintTolerance=1e-7;
    struct Observation {std::vector<DiscreteElasticRodState> states;std::vector<DiscreteRodAttachmentReaction> reactions;};
    std::vector<Observation> expected;std::vector<DiscreteElasticRodState> split;
    auto advance=[&](unsigned step,MetalDiscreteElasticRodResult& out) {
        for(unsigned e=0;e<2;++e) {at[e*2].targetPosition={origin[e],0,0};at[e*2+1].targetPosition={origin[e]+.122+step*h*speed,0,0};}
        MetalDiscreteElasticRodInput in;in.states=states;in.attachmentCount=2;in.attachments=at;
        auto d=rod_local_research::run(model,in,out,cfg);
        if(!d.succeeded()||!d.published||out.states.size()!=2||out.reactions.size()!=4) {
            std::cerr<<"step "<<step<<" failed: "<<d.message<<"\n";return false;
        }
        states=out.states;return true;
    };
    for(unsigned step=0;step<=40;++step) {
        MetalDiscreteElasticRodResult out;if(!advance(step,out)) return 1;
        if(step==20) split=states;
        if(step>20) expected.push_back({out.states,out.reactions});
    }
    if(!save(argv[2],split)) return 2;
    std::vector<DiscreteElasticRodState> restored;
    if(!load(argv[2],restored)) return 2;
    for(unsigned e=0;e<2;++e) if(!sameState(restored[e],split[e])) return 1;
    states=restored;
    for(unsigned step=21;step<=40;++step) {
        MetalDiscreteElasticRodResult out;if(!advance(step,out)) return 1;
        auto& ref=expected[step-21];
        for(unsigned e=0;e<2;++e) if(!sameState(out.states[e],ref.states[e])) {std::cerr<<"state mismatch "<<step<<"\n";return 1;}
        for(unsigned i=0;i<4;++i) if(!sameReaction(out.reactions[i],ref.reactions[i])) {std::cerr<<"reaction mismatch "<<step<<"\n";return 1;}
    }
    std::string truncated=std::string(argv[2])+".truncated";
    {std::ofstream f(truncated,std::ios::binary|std::ios::trunc);f<<"ROD";}
    auto before=restored;
    if(load(truncated.c_str(),restored)) return 1;
    for(unsigned e=0;e<2;++e) if(!sameState(restored[e],before[e])) return 1;
    std::cout<<"continuous_steps=41 replay_steps=20 environments=2 bitwise_states_and_reactions=1 truncated_rejected_unchanged=1\n";
    return 0;
}
