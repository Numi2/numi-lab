#include "NumiHumanInferredACL.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using metalrobo::inferred_acl::V3;
namespace d=metalrobo::inferred_acl::detail;

static void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
static bool near(V3 a,V3 b,double tol=1.0e-12) {
    return d::norm(d::sub(a,b))<=tol;
}
static std::array<double,3> independentBarycentric(V3 q,V3 a,V3 b,V3 c) {
    const V3 ab=d::sub(b,a),ac=d::sub(c,a),aq=d::sub(q,a);
    const double d00=d::dot(ab,ab),d01=d::dot(ab,ac),d11=d::dot(ac,ac);
    const double d20=d::dot(aq,ab),d21=d::dot(aq,ac);
    const double den=d00*d11-d01*d01;
    require(std::isfinite(den)&&den>1.0e-24,"test triangle degenerate");
    const double v=(d11*d20-d01*d21)/den;
    const double w=(d00*d21-d01*d20)/den;
    return {1.0-v-w,v,w};
}
static void verifyInside(V3 q,V3 a,V3 b,V3 c) {
    const auto w=independentBarycentric(q,a,b,c);
    double sum=0.0;
    for(double x:w) {
        require(std::isfinite(x)&&x>=-1.0e-10&&x<=1.0+1.0e-10,
                "closest result escaped triangle barycentric domain");
        sum+=x;
    }
    const V3 rebuilt=d::add(a,d::add(d::mul(d::sub(b,a),w[1]),
                                     d::mul(d::sub(c,a),w[2])));
    require(std::abs(sum-1.0)<=1.0e-12&&near(rebuilt,q,1.0e-10),
            "closest result failed barycentric reconstruction");
}
static V3 segmentOracle(V3 p,V3 a,V3 b) {
    const V3 ab=d::sub(b,a);
    const double den=d::dot(ab,ab);
    const double t=std::clamp(d::dot(d::sub(p,a),ab)/den,0.0,1.0);
    return d::add(a,d::mul(ab,t));
}
static V3 independentOracle(V3 p,V3 a,V3 b,V3 c) {
    const V3 n=d::unit(d::cross(d::sub(b,a),d::sub(c,a)),"test triangle degenerate");
    const V3 plane=d::sub(p,d::mul(n,d::dot(d::sub(p,a),n)));
    const auto w=independentBarycentric(plane,a,b,c);
    if(std::all_of(w.begin(),w.end(),[](double x){return x>=0.0;}))return plane;
    const std::array<V3,3> edges{
        segmentOracle(p,a,b),segmentOracle(p,b,c),segmentOracle(p,c,a)};
    return *std::min_element(edges.begin(),edges.end(),[&](V3 x,V3 y){
        return d::dot(d::sub(p,x),d::sub(p,x))<d::dot(d::sub(p,y),d::sub(p,y));
    });
}
static void verify(V3 p,V3 a,V3 b,V3 c,V3 expected,const char* label) {
    const V3 actual=d::closest(p,a,b,c);
    verifyInside(actual,a,b,c);
    require(near(actual,expected),label);
    require(near(actual,independentOracle(p,a,b,c)),
            "closest result disagrees with independent region oracle");
}
static void verifySlidingKneeFrame() {
    metalrobo::EngineModel model;
    model.bodies.resize(5);
    model.defaultQ={0.0f};
    MRJointDescriptorGPU sliding{};
    sliding.parentBody=0; sliding.childBody=1;
    sliding.jointType=MR_JOINT_PRISMATIC;
    sliding.nq=sliding.nv=1; sliding.qOffset=0;
    sliding.axis0={2,0,0,0}; // FK normalizes source axes.
    sliding.parentAnchor={0,0,-1,0};
    const float h=std::sqrt(.5f);
    sliding.parentRotation={0,0,h,h};
    MRJointDescriptorGPU hinge{};
    hinge.parentBody=1;hinge.childBody=2;
    hinge.jointType=MR_JOINT_REVOLUTE;
    hinge.childAnchor={0,0,.5f,0};
    MRJointDescriptorGPU hip{};
    hip.parentBody=3;hip.childBody=0;hip.jointType=MR_JOINT_FIXED;
    hip.childAnchor={0,0,1,0};
    model.joints={sliding,hinge,hip};
    std::vector<MRBodyStateGPU> poses(5);
    for(auto& pose:poses) pose.orientation={0,0,0,1};
    poses[0].position={0,0,1,1};
    poses[1].position={0,.01f,0,1};
    poses[2].position={0,.01f,-.5f,1};
    poses[3].position={0,0,2,1};
    poses[4].position={.2f,.01f,0,1};
    const std::array<float,1> q{.01f};
    const auto frame=metalrobo::inferred_acl::deriveRegisteredKneeFrame(
        model,poses,0,0,2,4,24,q);
    require(near(frame.jointCenterWorld,{0,.01,0},1e-8),
            "sliding knee origin must include rotated prismatic displacement");
    const auto rejects=[&](const std::vector<MRBodyStateGPU>& states,
                           std::span<const float> coordinates) {
        try { (void)metalrobo::inferred_acl::deriveRegisteredKneeFrame(
            model,states,0,0,2,4,24,coordinates); }
        catch(const std::runtime_error&) { return true; }
        return false;
    };
    require(rejects(poses,std::array<float,1>{.02f}),
            "wrong sliding coordinate must retain 0.2mm rejection");
    auto transverse=poses;transverse[1].position.x=.0003f;
    require(rejects(transverse,q),"transverse anchor mismatch must reject");
    auto invalid=std::array<float,1>{std::numeric_limits<float>::quiet_NaN()};
    require(rejects(poses,invalid),"nonfinite sliding coordinate must reject");
    require(rejects(poses,{}),"default reference q must not stand in for a posed q");
    model.defaultQ={.01f};
    const auto defaultFrame=metalrobo::inferred_acl::deriveRegisteredKneeFrame(
        model,poses,0,0,2,4,24);
    require(near(defaultFrame.jointCenterWorld,frame.jointCenterWorld),
            "explicit default coordinates must match the same supplied poses");
    model.joints[0].axis0={0,0,0,0};
    require(rejects(poses,q),"invalid source axis must reject");
}
int main() {
    verifySlidingKneeFrame();
    const V3 a{0,0,0},b{1,0,0},c{0,1,0};
    verify({-1,-1,.2},a,b,c,a,"vertex A region mismatch");
    verify({2,-1,.2},a,b,c,b,"vertex B region mismatch");
    verify({-1,2,.2},a,b,c,c,"vertex C region mismatch");
    verify({.4,-.2,.3},a,b,c,{.4,0,0},"edge AB region mismatch");
    verify({-.2,.4,.3},a,b,c,{0,.4,0},"edge AC region mismatch");
    verify({1,1,.3},a,b,c,{.5,.5,0},"edge BC region mismatch");
    verify({.2,.3,2},a,b,c,{.2,.3,0},"face region mismatch");

    metalrobo::inferred_acl::Frame frame{};
    frame.jointCenterWorld={.1,.2,.3};
    frame.anteriorWorld={0,1,0}; frame.superiorWorld={0,0,1};
    const auto left=metalrobo::inferred_acl::deriveRegisteredACLSeedSelection(frame);
    require(near(left.femurSeedWorld,{.104,.183,.301}),"left seed must compose base plus adjustment");
    require(near(left.tibiaSeedWorld,{.095,.207,.293}),"left tibia seed must compose base plus adjustment");
    frame.side=1;
    const auto right=metalrobo::inferred_acl::deriveRegisteredACLSeedSelection(frame);
    require(near(right.femurSeedWorld,{.096,.183,.301}),"right femur lateral offset must mirror");
    require(near(right.tibiaSeedWorld,{.105,.207,.293}),"right tibia lateral offset must mirror");
    frame.side=2; bool rejected=false;
    try { (void)metalrobo::inferred_acl::deriveRegisteredACLSeedSelection(frame); }
    catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"invalid laterality must reject");
    std::cout<<"acl_geometry_regression=PASS voronoi_regions=7 seed_composition=checked laterality=checked prismatic_anchor=checked\n";
}
