#include "NumiHumanRegisteredSkinCandidateCSV.hpp"
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
int main() {
    RegisteredSurface s;
    s.worldPoints={V3{0,0,0},V3{.1,.2,.3},V3{-.2,.05,.9}};
    s.donorLocalPoints=s.worldPoints;s.donorBodyIndices={1,2,3};
    s.worldToDonorLocalLinear.assign(3,{1,0,0,0,1,0,0,0,1});
    RegisteredSkinRigidPlacement truth;truth.translation={0,0,-5.662331395348288e-6};
    const double c=std::cos(.37),t=std::sin(.37);truth.rotation={c,-t,0,t,c,0,0,0,1};
    std::vector<V3> points;for(const auto& x:s.worldPoints)points.push_back(truth.point(x));
    auto locals=s.donorLocalPoints;
    auto frames=s.worldToDonorLocalLinear;for(auto& m:frames)m=truth.donorFrame(m);
    const auto p=[&](std::size_t i){return points[i];};
    const auto l=[&](std::size_t i,const V3&){return locals[i];};
    const auto f=[&](std::size_t i){return frames[i];};
    RegisteredSkinCandidatePlacementReport report;std::string error;
    if(!validateRegisteredSkinCandidateNativePlacement(s,p,l,f,report,error)||
       report.placement.validatedVertices!=3||report.maximumSkinningResidualM>1e-14) return 1;
    const auto old=report;
    points[2][1]+=.003;
    if(validateRegisteredSkinCandidateNativePlacement(s,p,l,f,report,error)||
       report.placement.translation!=old.placement.translation) return 2;
    points[2]=truth.point(s.worldPoints[2]);locals[1][2]+=.0001;
    if(validateRegisteredSkinCandidateNativePlacement(s,p,l,f,report,error))return 3;
    locals=s.donorLocalPoints;frames[2][3]+=.001;
    if(validateRegisteredSkinCandidateNativePlacement(s,p,l,f,report,error))return 4;
    frames[2]=truth.donorFrame(s.worldToDonorLocalLinear[2]);
    points[2][1]=std::numeric_limits<double>::quiet_NaN();
    if(validateRegisteredSkinCandidateNativePlacement(s,p,l,f,report,error))return 5;
    RegisteredSurface empty;
    if(validateRegisteredSkinCandidateNativePlacement(empty,p,l,f,report,error))return 6;
    std::cout<<"registered_skin_native_placement=PASS common_SE3=true local_joint_change=reject "
        "local_frame_change=reject donor_rotation_change=reject nonfinite=reject empty=reject "
        "output_atomic=true physical_steps=0\n";
}
