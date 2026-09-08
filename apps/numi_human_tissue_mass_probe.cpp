#include "metalrobo/NumiHumanTissueMass.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
MRBodyPropertiesGPU donor() {
    MRBodyPropertiesGPU b{};
    b.motionType = MR_MOTION_DYNAMIC;
    b.massAndInverseMass = {10.0f, 0.1f, 0.0f, 0.0f};
    b.inertiaRow0 = {2.0f, 0.0f, 0.0f, 0.0f};
    b.inertiaRow1 = {0.0f, 3.0f, 0.0f, 0.0f};
    b.inertiaRow2 = {0.0f, 0.0f, 4.0f, 0.0f};
    return b;
}
double dot(const std::array<double,3>& a, const std::array<double,3>& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
std::array<double,3> cross(const std::array<double,3>& a, const std::array<double,3>& b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
}
int main() {
    try {
        const std::vector<MRBodyPropertiesGPU> bodies{donor()};
        const std::vector<metalrobo::NumiHumanTissueMassNode> nodes{
            {3u,0u,0.4,{0.2,-0.1,0.3}}, {7u,0u,0.2,{-0.1,0.2,0.1}}};
        const auto result = metalrobo::compileNumiHumanTissueMassPartition(bodies,nodes);
        require(result.succeeded() && result.partitions.size()==1, "valid partition rejected");
        const auto& p=result.partitions[0];
        require(std::abs(p.tissueMassKg-0.6)<1e-14, "mass not retained");
        require(std::abs(p.remainingCOMOffsetM[0]+0.06/9.4)<1e-14, "first moment not retained");
        require(std::abs(p.remainingBody.inertiaRow0.y)>1e-4, "off-diagonal inertia discarded");
        // Independent rigid-motion recombination: momentum, angular momentum,
        // kinetic energy and gravitational first moment across 64 twists.
        double maxError=0.0;
        for (int k=1;k<=64;++k) {
            const std::array<double,3> v{0.013*k,-0.031*k,0.017*k};
            const std::array<double,3> w{0.021*k,0.011*k,-0.009*k};
            const auto c=p.remainingCOMOffsetM;
            const auto wx=cross(w,c);
            const std::array<double,3> vr{v[0]+wx[0],v[1]+wx[1],v[2]+wx[2]};
            const auto& b=p.remainingBody;
            const std::array<double,3> iw{
                b.inertiaRow0.x*w[0]+b.inertiaRow0.y*w[1]+b.inertiaRow0.z*w[2],
                b.inertiaRow1.x*w[0]+b.inertiaRow1.y*w[1]+b.inertiaRow1.z*w[2],
                b.inertiaRow2.x*w[0]+b.inertiaRow2.y*w[1]+b.inertiaRow2.z*w[2]};
            const double m=b.massAndInverseMass.x;
            std::array<double,3> momentum{m*vr[0],m*vr[1],m*vr[2]};
            auto angular=cross(c,momentum);
            for (int a=0;a<3;++a) angular[a]+=iw[a];
            double energy=0.5*m*dot(vr,vr)+0.5*dot(w,iw);
            for(const auto& n:nodes) {
                const auto wn=cross(w,n.localPosition);
                const std::array<double,3> vn{v[0]+wn[0],v[1]+wn[1],v[2]+wn[2]};
                const std::array<double,3> pn{n.massKg*vn[0],n.massKg*vn[1],n.massKg*vn[2]};
                const auto ln=cross(n.localPosition,pn);
                for(int a=0;a<3;++a) {momentum[a]+=pn[a];angular[a]+=ln[a];}
                energy+=0.5*n.massKg*dot(vn,vn);
            }
            const double expected=5.0*dot(v,v)+0.5*(2*w[0]*w[0]+3*w[1]*w[1]+4*w[2]*w[2]);
            maxError=std::max(maxError,std::abs(energy-expected)/std::max(1.0,expected));
            for(int a=0;a<3;++a) {
                maxError=std::max(maxError,std::abs(momentum[a]-10*v[a])/std::max(1.0,std::abs(10*v[a])));
                maxError=std::max(maxError,std::abs(angular[a]-(2+a)*w[a])/std::max(1.0,std::abs((2+a)*w[a])));
            }
        }
        require(maxError<2e-6,"rigid-motion invariants changed after tissue partition");
        auto reject=[&](auto b, auto n) {
            auto r=metalrobo::compileNumiHumanTissueMassPartition(b,n);
            require(!r.succeeded()&&r.partitions.empty(),"invalid partition published partial output");
        };
        auto bad=nodes; bad.push_back(nodes[0]); reject(bodies,bad);
        bad=nodes; bad[0].donorBody=1u; reject(bodies,bad);
        bad=nodes; bad[0].massKg=0.0; reject(bodies,bad);
        bad=nodes; bad[0].massKg=11.0; reject(bodies,bad);
        bad=nodes; bad[0].localPosition[0]=20.0; reject(bodies,bad);
        bad=nodes; bad[0].massKg=std::numeric_limits<double>::quiet_NaN(); reject(bodies,bad);
        bad=nodes; bad[0].localPosition[0]=std::numeric_limits<double>::infinity(); reject(bodies,bad);
        auto b=bodies; b[0].inertiaRow2.z=6.0f; reject(b,nodes); // positive I, impossible second moment
        b=bodies; b[0].inertiaRow0.y=0.01f; reject(b,nodes);
        b=bodies; b[0].motionType=MR_MOTION_STATIC; reject(b,nodes);
        reject(bodies,std::vector<metalrobo::NumiHumanTissueMassNode>{});
        // A rotating floating donor needs BOTH root COM position and root
        // linear velocity rebased. Its physical collider centre stays put.
        auto model=metalrobo::makeFreeSphereEngineModel();
        const auto root=model.articulations[0].rootBody;
        model.defaultQ[5]=std::sqrt(0.5f);model.defaultQ[6]=std::sqrt(0.5f);
        model.defaultV={.2f,-.3f,.4f,.5f,-.6f,.7f};
        const std::vector<metalrobo::NumiHumanTissueMassNode> rootNodes{{9u,root,.01,{.1,.02,.03}}};
        const auto rootMass=metalrobo::compileNumiHumanTissueMassPartition(model.bodies,rootNodes);
        require(rootMass.succeeded(),"floating-root mass partition failed");
        metalrobo::EngineModel rebased;std::string error;
        require(metalrobo::rebaseNumiHumanTissueMassPartition(model,rootMass.partitions,rebased,error),error.c_str());
        const auto& offset=rootMass.partitions[0].remainingCOMOffsetM;
        const std::vector<metalrobo::ArticulatedPointQuery> originalPoint{{root,{0,0,0}}},
            newPoint{{root,{-offset[0],-offset[1],-offset[2]}}};
        const std::vector<double> oq(model.defaultQ.begin(),model.defaultQ.end()),
            ov(model.defaultV.begin(),model.defaultV.end()),
            nq(rebased.defaultQ.begin(),rebased.defaultQ.end()),
            nv(rebased.defaultV.begin(),rebased.defaultV.end());
        std::vector<metalrobo::ArticulatedPointKinematics> op(1),np(1);
        std::vector<double> oj(18),nj(18);
        require(metalrobo::computeArticulatedPointJacobians(model,0,oq,ov,originalPoint,op,oj).succeeded(),"floating source point failed");
        require(metalrobo::computeArticulatedPointJacobians(rebased,0,nq,nv,newPoint,np,nj).succeeded(),"floating rebased point failed");
        for(unsigned a=0;a<3;++a) {
            require(std::abs(op[0].position[a]-np[0].position[a])<2e-7,"floating root position changed");
            require(std::abs(op[0].linearVelocity[a]-np[0].linearVelocity[a])<2e-7,"floating root velocity changed");
        }
        require(std::abs(rebased.shapes[1].localPosition.x+offset[0])<1e-9,"collider was not rebased");
        auto unchanged=rebased;
        require(!metalrobo::rebaseNumiHumanTissueMassPartition(rebased,rootMass.partitions,unchanged,error),"double mass rebase admitted");
        require(unchanged.defaultQ==rebased.defaultQ,"rejected rebase mutated output");
        std::cout<<"{\"status\":\"passed\",\"rigid_motion_cases\":64,\"negative_cases\":11,"
                 <<"\"rotating_floating_root_rebase\":true,"
                 <<"\"maximum_relative_invariant_error\":"<<maxError<<"}\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
