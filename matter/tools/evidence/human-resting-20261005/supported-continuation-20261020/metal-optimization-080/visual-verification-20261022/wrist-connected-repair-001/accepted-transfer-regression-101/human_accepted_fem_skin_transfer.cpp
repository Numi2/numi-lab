#include "NumiHumanAcceptedFEMSkinTransfer.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using T=numi::human::AcceptedFEMSkinTransfer;
int main(){
  std::vector<T::Point> sourceRest{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}};
  std::vector<T::Point> coarse{{.02,0,0},{1.02,0,0},{.02,1,0},{.02,0,1},{1.02,0,1},{.02,1,1}};
  std::vector<unsigned> cs{0,1,2,3,4,5}, cd{1,1,1,2,2,2};
  std::vector<T::Face> f{{0,1,2},{3,4,5}};
  std::vector<T::Point> detail{{.02,0,0},{.27,.25,.00001},{.02,0,1},{99,99,99}};
  std::vector<unsigned> ds{0,9,3,99}, dd{1,1,2,77};
  std::vector<std::uint8_t> used{1,1,1,0};
  T x; T::BuildReport report; std::string err;
  assert(x.build(detail,ds,dd,used,coarse,cs,cd,f,&report,&err));
  assert(report.exactAncestryRows==2 && report.projectedRows==1 && report.inactiveRows==1);
  const auto& row=x.rows()[1];
  assert(row.donorBodyIndex==1);
  assert(std::abs(row.weight[0]+row.weight[1]+row.weight[2]-1.f)<1e-6f);
  assert(row.coarseVertex[0]<3 && row.coarseVertex[1]<3 && row.coarseVertex[2]<3);
  std::vector<T::Pose> restPose(3), currentPose(3);
  for(auto& p:restPose)p.rotation={0,0,0,1};
  for(auto& p:currentPose)p.rotation={0,0,0,1};
  std::vector<T::Point> currentSource=sourceRest, baseline;
  assert(x.makeCurrentCoarseBaseline(sourceRest,currentSource,restPose,currentPose,baseline,&err));
  // Zero FEM displacement reproduces the rigidly carried remesh offset exactly.
  std::vector<std::array<float,4>> delta;
  assert(x.makeCoarseDisplacements(baseline,baseline,delta,&err));
  std::vector<T::Point> out;
  assert(x.applyReference(detail,delta,out,&err));
  assert(out==detail);
  // Accepted FEM displacement is added on top of, not instead of, that baseline.
  auto accepted=baseline;
  for(auto& p:accepted)p[2]+=.1;
  assert(x.makeCoarseDisplacements(accepted,baseline,delta,&err));
  assert(x.applyReference(detail,delta,out,&err));
  assert(std::abs(out[0][2]-.1)<1e-6);
  assert(std::abs(out[1][2]-.10001)<1e-6);
  assert(std::abs(out[2][2]-1.1)<1e-6);
  assert(out[3]==detail[3]);
  // A donor rotation carries the mechanical rest offset with its source.
  currentPose[1].rotation={0,0,std::sin(M_PI/4),std::cos(M_PI/4)};
  assert(x.makeCurrentCoarseBaseline(sourceRest,currentSource,restPose,currentPose,baseline,&err));
  assert(std::abs(baseline[0][0])<1e-9);
  assert(std::abs(baseline[0][1]-.02)<1e-9);
  std::cout<<"PASS donor ancestry/BVH, inactive orphans, zero-FEM remesh baseline, donor pose, displacement transfer\n";
}
