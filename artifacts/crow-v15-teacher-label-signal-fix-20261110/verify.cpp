#include "metalrobo/LearningPacks.hpp"
#include <cmath>
#include <iostream>
using namespace metalrobo;
int main(int argc,char**argv){
 if(argc!=3)return 2;
 PolicyRolloutPack a,b;auto x=readPolicyRolloutPack(argv[1],a),y=readPolicyRolloutPack(argv[2],b);
 if(!x.succeeded()||!y.succeeded()||a.environmentCount!=1||b.environmentCount!=1||a.actionCount!=b.actionCount||a.teacherActions.empty()||b.teacherActions.empty())return 3;
 for(int t:{1000,1200,1400,1500}) {
  auto i=static_cast<size_t>(t)*a.actionCount;
  std::cout<<"step "<<t<<" old_x "<<a.teacherActions[i]<<" fixed_x "<<b.teacherActions[i]<<" fixed_y "<<b.teacherActions[i+1]<<"\n";
  if(std::abs(b.teacherActions[i]-.5f)>1e-5f || std::abs(a.teacherActions[i]+1.f/3.f)>1e-4f)return 4;
 }
}
