#include "metalrobo/MujocoMuscleReference.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace metalrobo;
std::size_t checks = 0;
std::uint64_t fingerprint = 14695981039346656037ull;
void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void record(double value) {
    const auto bits=std::bit_cast<std::uint64_t>(value);
    for(unsigned i=0;i<8;++i) { fingerprint^=(bits>>(8*i))&255u; fingerprint*=1099511628211ull; }
}
void equal(double a,double b) {
    require(std::bit_cast<std::uint64_t>(a)==std::bit_cast<std::uint64_t>(b),"batch changed source-order FP64 path arithmetic");
    record(a);
}
void equal(const MujocoMusclePathResult& a,const MujocoMusclePathResult& b) {
    equal(a.length,b.length); equal(a.velocity,b.velocity);
    require(a.appliedWrapCount==b.appliedWrapCount,"batch changed wrap selection");
    require(a.lengthJacobian.size()==b.lengthJacobian.size(),"batch changed DoF coverage");
    require(a.centreline.size()==b.centreline.size(),"batch changed route centreline");
    for(std::size_t i=0;i<a.lengthJacobian.size();++i) equal(a.lengthJacobian[i],b.lengthJacobian[i]);
    for(std::size_t i=0;i<a.centreline.size();++i) {
        require(a.centreline[i].attachmentBodyIndex==b.centreline[i].attachmentBodyIndex,"batch changed attachment ownership");
        for(unsigned j=0;j<3;++j) equal(a.centreline[i].world[j],b.centreline[i].world[j]);
    }
}
EngineModel model() {
    EngineModel m;
    MRArticulationGPU art{};
    art.rootBody=0;art.rootType=MR_ROOT_FIXED;art.bodyCount=3;art.jointCount=2;art.nq=2;art.nv=2;
    m.articulations.push_back(art);
    for(unsigned i=0;i<3;++i) {
        MRBodyPropertiesGPU b{};
        b.parentBody=i?i-1:MR_INVALID_INDEX; b.inboundJoint=i?i-1:MR_INVALID_INDEX;
        b.motionType=MR_MOTION_DYNAMIC; b.massAndInverseMass={1,1,0,0};
        b.inertiaRow0=b.inverseInertiaRow0={1,0,0,0};
        b.inertiaRow1=b.inverseInertiaRow1={0,1,0,0};
        b.inertiaRow2=b.inverseInertiaRow2={0,0,1,0};
        b.dampingAndSpeedLimits={0,0,1e6f,1e6f};m.bodies.push_back(b);
    }
    for(unsigned i=0;i<2;++i) {
        MRJointDescriptorGPU j{};
        j.parentBody=i;j.childBody=i+1;j.jointType=MR_JOINT_PRISMATIC;
        j.qOffset=i;j.vOffset=i;j.nq=1;j.nv=1;j.axis0={0,1,0,0};
        j.parentRotation=j.childRotation={0,0,0,1};m.joints.push_back(j);
        MRDofPropertiesGPU d{};d.jointIndex=i;d.qIndex=i;d.vIndex=i;m.dofs.push_back(d);
    }
    m.defaultQ={0,0};m.defaultV={0,0};return m;
}
}
int main() {
    try {
        using namespace metalrobo;
        const auto m=model();
        std::vector<MujocoMuscleSite> sites{{0,{-1,0.1,0.1}},{2,{1,0.1,0.2}},
                                           {1,{0,1,0.1}},{0,{-1,0.1,0.1}}};
        const std::array<double,9> identity{1,0,0,0,1,0,0,0,1};
        const std::vector<MujocoWrapGeometry> wraps{
            {1,MujocoRouteNodeType::sphere,{0,0,0.1},identity,0.3},
            {1,MujocoRouteNodeType::cylinder,{0,0,0.1},identity,0.3}};
        MujocoMuscleDefinition base;
        base.lengthRange={0.8,3};base.accelerationScale=1;base.controlRange={0,1};
        base.gainParameters={0.75,1.05,1,200,0.5,1.6,1.5,1.3,1.2,0};
        base.biasParameters=base.gainParameters;base.dynamicParameters={0.01,0.04,0};
        std::vector<MujocoMuscleDefinition> definitions(97,base);
        for(unsigned i=0;i<definitions.size();++i) {
            auto& route=definitions[i].route;
            route.push_back({MujocoRouteNodeType::site,0});
            if(i%4==1) route.push_back({MujocoRouteNodeType::sphere,0,2});
            if(i%4==2) route.push_back({MujocoRouteNodeType::cylinder,1,2});
            if(i%4==3) route.push_back({MujocoRouteNodeType::site,3});
            route.push_back({MujocoRouteNodeType::site,1});
        }
        std::vector<MujocoMusclePathResult> paths;
        std::size_t wrapped=0;
        double scalarMs=0,batchMs=0;
        for(unsigned pose=0;pose<8;++pose) {
            const std::vector<double> q{-0.03+0.01*pose,0.02-0.008*pose};
            const std::vector<double> v{0.1-0.02*pose,-0.07+0.03*pose};
            std::vector<MujocoMusclePathResult> scalar;
            auto start=std::chrono::steady_clock::now();
            for(const auto& definition:definitions) {
                MujocoMuscleResult result;
                require(evaluateMujocoMuscle(m,0,q,v,sites,wraps,definition,{},result).succeeded(),"scalar route fixture failed");
                scalar.push_back(std::move(result.path));
            }
            scalarMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            start=std::chrono::steady_clock::now();
#ifdef HUMAN_SCALAR_ORACLE_ONLY
            paths=scalar;
#else
            require(evaluateMujocoMusclePaths(m,0,q,v,sites,wraps,definitions,paths).succeeded(),"batched route fixture failed");
#endif
            batchMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            require(paths.size()==definitions.size(),"batch lost source muscles");
            for(std::size_t i=0;i<paths.size();++i) { equal(paths[i],scalar[i]); wrapped+=paths[i].appliedWrapCount; }
            const double expectedLength=std::sqrt(4+std::pow(q[0]+q[1],2)+0.01);
            require(std::abs(paths[0].length-expectedLength)<1e-14,"straight route differs from analytic geometry");
            require(std::abs(paths[0].lengthJacobian[0]-(q[0]+q[1])/expectedLength)<1e-14,"straight route differs from analytic derivative");
        }
        require(wrapped>0,"wrapping was not exercised");
        const auto validFingerprint=fingerprint;
#ifndef HUMAN_SCALAR_ORACLE_ONLY
        const auto accepted=paths;
        const std::vector<double> q{0,0},v{0,0};
        auto invalid=definitions;invalid[52].route[0].targetIndex=999;
        auto status=evaluateMujocoMusclePaths(m,0,q,v,sites,wraps,invalid,paths);
        require(!status.succeeded()&&status.failingIndex==52,"batch lost invalid muscle identity");
        require(paths.size()==accepted.size(),"failed batch resized accepted output");
        for(std::size_t i=0;i<paths.size();++i) equal(paths[i],accepted[i]);
        auto badQ=q;badQ[0]=std::numeric_limits<double>::quiet_NaN();
        require(!evaluateMujocoMusclePaths(m,0,badQ,v,sites,wraps,definitions,paths).succeeded(),"batch admitted nonfinite state");
        require(!evaluateMujocoMusclePaths(m,3,q,v,sites,wraps,definitions,paths).succeeded(),"batch admitted unknown articulation");
        require(!evaluateMujocoMusclePaths(m,0,std::span<const double>{},v,sites,wraps,definitions,paths).succeeded(),"batch admitted wrong state dimensions");
        for(std::size_t i=0;i<paths.size();++i) equal(paths[i],accepted[i]);
        require(evaluateMujocoMusclePaths(m,0,q,v,sites,wraps,{},paths).succeeded()&&paths.empty(),"empty muscle batch changed semantics");
#endif
        std::cout<<"muscle_path_batch=passed checks="<<checks<<" wrapped_routes="<<wrapped
                 <<" valid_fingerprint="<<validFingerprint<<" scalar_ms="<<scalarMs<<" batch_ms="<<batchMs<<'\n';
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
