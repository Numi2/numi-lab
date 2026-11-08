#include "metalrobo/VisualPresentation.hpp"
#include "metalrobo/RunProgram.hpp"
#include <iostream>
int main(int argc,char** argv) {
 for(int i=1;i<argc;++i) {
  metalrobo::VisualAssetPackV2 p;std::string why;
  if(!metalrobo::readVisualAssetPackIndex(argv[i],p,&why)) { std::cerr<<why<<"\n";return 1; }
  for(const auto& v:p.instances) std::cout<<argv[i]<<" bind_asset="<<v.binding.x
   <<" body="<<v.binding.y<<" kind="<<v.binding.z<<" local=("
   <<v.translationAndScale.x<<","<<v.translationAndScale.y<<","<<v.translationAndScale.z<<")\n";
 }
 auto r=metalrobo::makeAmericanCrowActiveHeadRobotPack();
 for(size_t i=0;i<r.mechanics.bodyNames.size();++i)
  std::cout<<"robot body "<<i<<" "<<r.mechanics.bodyNames[i]<<"\n";
}
