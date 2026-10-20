#include "NumiHumanRegisteredLigaments.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace metalrobo::registered_ligaments;
using metalrobo::EngineModel;
namespace inferred_acl = metalrobo::inferred_acl;
using metalrobo::inferred_acl::BoneSurface;
using metalrobo::inferred_acl::V3;

static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static bool near(double a,double b,double tol){return std::abs(a-b)<=tol;}
static V3 worldPoint(const MRBodyStateGPU& b,nm_float4 p){
 return inferred_acl::detail::add(
     inferred_acl::detail::xyz(b.position),
     inferred_acl::detail::rotate(b.orientation,{p.x,p.y,p.z}));
}
static mr_float4 qmul(mr_float4 a,mr_float4 b){
 return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
         a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
         a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
         a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
}

int main(){
 const auto catalog=referenceCatalog();
 check(catalog.size()==26u,"catalog row count");
 std::array<unsigned,3> counts{};
 for(const auto& s:catalog){
  check(s.side<2u,"side range");
  check(s.stiffnessNPerM>0&&std::isfinite(s.stiffnessNPerM),"finite positive literature stiffness");
  check(s.areaM2>0&&std::isfinite(s.areaM2),"finite positive normalization area");
  check(!s.stiffnessSource.empty()&&!s.geometrySource.empty()&&!s.limitations.empty(),
        "source and limitation provenance");
  check(s.group!=Group::knee,"knee owner excluded");
  counts[static_cast<unsigned>(s.group)]++;
 }
 check(counts[0]==8u&&counts[1]==4u&&counts[2]==14u,"bilateral regional coverage");
 for(const auto name:{"iliofemoral_lateral","iliofemoral_medial","pubofemoral",
                     "ischiofemoral","superior_glenohumeral_capsuloligament","coracohumeral_capsuloligament",
                     "anterior_talofibular_superior","anterior_talofibular_inferior","calcaneofibular",
                     "posterior_talofibular","anterior_tibiotalar",
                     "tibiocalcaneal","posterior_tibiotalar"}){
  const auto left=std::find_if(catalog.begin(),catalog.end(),[&](const auto& s){return s.side==0u&&s.name==name;});
  const auto right=std::find_if(catalog.begin(),catalog.end(),[&](const auto& s){return s.side==1u&&s.name==name;});
  check(left!=catalog.end()&&right!=catalog.end(),"bilateral named row");
  check(!left->seedLateralityMirrored&&!right->seedLateralityMirrored,
        "catalog carries side-specific surface witnesses");
  check(inferred_acl::detail::norm(inferred_acl::detail::sub(left->seed0,right->seed0))>1e-6,
        "source witnesses are independently stored by side");
  if(name==std::string_view("iliofemoral_lateral")||
     name==std::string_view("iliofemoral_medial"))
   check(left->areaIsAssumedSplit&&right->areaIsAssumedSplit&&
         !left->areaMeasured&&!right->areaMeasured,
         "split ILF normalization is not reported as a measured full-band area");
  if(name==std::string_view("coracohumeral_capsuloligament"))
   check(near(left->stiffnessNPerM,36700.0,1e-9)&&
         near(right->stiffnessNPerM,36700.0,1e-9)&&
         left->areaMeasured&&right->areaMeasured&&
         left->limitations.find("route-first sites 672/1093")!=std::string_view::npos,
         "CHL retains its cited cohort prior and source-site limitation");
 }
 Frame f;f.q={0,0,0,1};
 const auto mirroredLeft=detail::seed(f,{.01,.02,.03},0u,true);
 const auto unmirroredRight=detail::seed(f,{.01,.02,.03},1u,true);
 check(near(mirroredLeft[0],unmirroredRight[0],1e-12)&&
       near(mirroredLeft[1],unmirroredRight[1],1e-12)&&
       near(mirroredLeft[2],-unmirroredRight[2],1e-12),
       "generic fallback mirrors the verified local-Z laterality axis");

 Spec hip=catalog.front();
 EngineModel emptyModel;
 std::array<MRBodyStateGPU,0> noPoses{};
 std::array<BoneSurface,0> noSurfaces{};
 bool aggregateRejected=false;
 try{(void)cookReference(hip,emptyModel,noPoses,noSurfaces,{},OwnershipPolicy{false,false,true,false});}
 catch(const std::runtime_error&){aggregateRejected=true;}
 check(aggregateRejected,"hip aggregate owner exclusion");

 const std::array<V3,3> midVerts{{{.05,-.02,-.02},{.05,.02,-.02},{.05,0,.02}}};
 const std::array<V3,3> midNormals{{{1,0,0},{1,0,0},{1,0,0}}};
 const std::array<std::uint32_t,3> midTri{{0,1,2}};
 BoneSurface crossing{0u,12u,9u,{},midVerts,midNormals,midTri};
 check(detail::segmentHitsSurface({0,0,0},{.1,0,0},crossing),
       "endpoint-surface segment crossing is rejected");

 Spec synthetic;
 synthetic.name="synthetic_reference_fiber";
 synthetic.group=Group::shoulderCapsuloligament;
 synthetic.side=1u;
 synthetic.body0=0u;synthetic.stable0=10u;
 synthetic.body1=1u;synthetic.stable1=11u;
 synthetic.frameParent=0u;synthetic.frameChild=1u;
 synthetic.seed0={0,0,0};synthetic.seed1={.1,0,0};
 synthetic.stiffnessNPerM=42000;
 synthetic.areaM2=20e-6;
 synthetic.stiffnessSource="synthetic test input";
 synthetic.geometrySource="synthetic registered surface fixture";
 synthetic.limitations="not anatomical data";

 EngineModel model;
 MRJointDescriptorGPU joint{};
 joint.parentBody=0u;joint.childBody=1u;
 joint.parentAnchor={0,0,0,0};joint.childAnchor={-.1f,0,0,0};
 joint.parentRotation={0,0,0,1};joint.childRotation={0,0,0,1};
 model.joints.push_back(joint);
 std::array<MRBodyStateGPU,2> poses{};
 poses[0].position={0,0,0,0};poses[0].orientation={0,0,0,1};
 poses[1].position={.1f,0,0,0};poses[1].orientation={0,0,0,1};
 std::array<V3,3> va{{{0,-.02,-.02},{0,.02,-.02},{0,0,.02}}};
 std::array<V3,3> vb{{{.1,-.02,-.02},{.1,.02,-.02},{.1,0,.02}}};
 std::array<V3,3> na{{{1,0,0},{1,0,0},{1,0,0}}};
 std::array<V3,3> nb{{{-1,0,0},{-1,0,0},{-1,0,0}}};
 std::array<std::uint32_t,3> tri{{0,1,2}};
 std::array<std::uint8_t,32> hash0{},hash1{};hash0[0]=1;hash1[0]=2;
 std::array<BoneSurface,2> surfaces{{
  {0u,10u,7u,hash0,va,na,tri},
  {1u,11u,8u,hash1,vb,nb,tri}}};
 check(!detail::segmentHitsSurface({0,0,0},{.1,0,0},surfaces[0])&&
       !detail::segmentHitsSurface({0,0,0},{.1,0,0},surfaces[1]),
       "endpoint-only intersections are excluded");
 auto built=cookReference(synthetic,model,poses,surfaces);
 BoneSurface thirdBodyCrossing{2u,12u,9u,hash0,midVerts,midNormals,midTri};
 std::array<BoneSurface,3> allRegisteredSurfaces{{
  surfaces[0],surfaces[1],thirdBodyCrossing}};
 bool thirdBodyRejected=false;
 try{(void)cookReference(synthetic,model,poses,allRegisteredSurfaces);}
 catch(const std::runtime_error& error){
  thirdBodyRejected=std::string(error.what()).find("intersected_stable_id=12")!=
      std::string::npos;
 }
 check(thirdBodyRejected,
       "registered fiber path crossing a third source bone is rejected");
 check(built.provenance.inferred&&!built.provenance.subjectCalibrated&&
       !built.provenance.sourceEnthesesMeasured,"inferred provenance flags");
 check(built.provenance.stable0==10u&&built.provenance.stable1==11u&&
       built.provenance.record0==7u&&built.provenance.record1==8u,
       "source surface identity");
 check(near(built.row.reference.x,.1,1e-6),"reference attachment length");
 check(near(double(built.row.reference.y)*double(built.row.material.x)*
            double(built.row.material.y)/double(built.row.reference.x),
            synthetic.stiffnessNPerM,1e-2),"SI force-length tangent matches source N/m");
 check(built.row.reference.z==1.0f&&built.provenance.inSituStretch==1.0,
       "zero-prestrain reference construction");
 check(built.provenance.triangle0==0u&&built.provenance.triangle1==0u&&
       built.provenance.residual0M<1e-9&&built.provenance.residual1M<1e-8,
       "bounded projection and selected source triangles");

 const auto p0=worldPoint(poses[0],built.row.firstLocalPoint);
 const auto p1=worldPoint(poses[1],built.row.secondLocalPoint);
 const mr_float4 commonQ{0,0,0.3826834324f,0.9238795325f};
 const V3 translation{.23,-.11,.07};
 for(auto& pose:poses){
  const V3 old=inferred_acl::detail::xyz(pose.position);
  const V3 moved=inferred_acl::detail::add(
      inferred_acl::detail::rotate(commonQ,old),translation);
  pose.position={float(moved[0]),float(moved[1]),float(moved[2]),0};
  pose.orientation=qmul(commonQ,pose.orientation);
 }
 const auto moved0=worldPoint(poses[0],built.row.firstLocalPoint);
 const auto moved1=worldPoint(poses[1],built.row.secondLocalPoint);
 const auto expected0=inferred_acl::detail::add(
     inferred_acl::detail::rotate(commonQ,p0),translation);
 const auto expected1=inferred_acl::detail::add(
     inferred_acl::detail::rotate(commonQ,p1),translation);
 check(inferred_acl::detail::norm(inferred_acl::detail::sub(moved0,expected0))<2e-7&&
       inferred_acl::detail::norm(inferred_acl::detail::sub(moved1,expected1))<2e-7,
       "common rigid placement covariance");

 const auto before0=worldPoint(poses[0],built.row.firstLocalPoint);
 const auto before1=worldPoint(poses[1],built.row.secondLocalPoint);
 const V3 com0{.012,-.006,.004},com1{-.009,.003,.007};
 // COM-origin rebasing shifts the body origin and local attachment oppositely.
 const V3 worldCom0=inferred_acl::detail::rotate(poses[0].orientation,com0);
 const V3 worldCom1=inferred_acl::detail::rotate(poses[1].orientation,com1);
 poses[0].position.x+=float(worldCom0[0]);poses[0].position.y+=float(worldCom0[1]);poses[0].position.z+=float(worldCom0[2]);
 poses[1].position.x+=float(worldCom1[0]);poses[1].position.y+=float(worldCom1[1]);poses[1].position.z+=float(worldCom1[2]);
 built.row.firstLocalPoint.x-=float(com0[0]);built.row.firstLocalPoint.y-=float(com0[1]);built.row.firstLocalPoint.z-=float(com0[2]);
 built.row.secondLocalPoint.x-=float(com1[0]);built.row.secondLocalPoint.y-=float(com1[1]);built.row.secondLocalPoint.z-=float(com1[2]);
 check(inferred_acl::detail::norm(inferred_acl::detail::sub(
       worldPoint(poses[0],built.row.firstLocalPoint),before0))<2e-7&&
       inferred_acl::detail::norm(inferred_acl::detail::sub(
       worldPoint(poses[1],built.row.secondLocalPoint),before1))<2e-7,
       "COM rebase preserves attachment world points");
 std::cout<<"registered_ligament_cpu_fixture=passed catalog_rows="<<catalog.size()
          <<" synthetic_stiffness_N_per_m="<<synthetic.stiffnessNPerM<<"\n";
}
