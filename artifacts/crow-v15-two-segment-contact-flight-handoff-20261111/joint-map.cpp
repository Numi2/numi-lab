#define main toe_prototype_main
#include "../crow-v15-two-segment-toe-prototype-20261111/probe.cpp"
#undef main
int main(){auto p=toeModel();auto& m=p.mechanics;for(unsigned i=0;i<m.joints.size();i++)if(i<7)std::cout<<i<<" "<<m.jointNames[i]<<" q="<<m.joints[i].qOffset<<" default="<<m.defaultQ[m.joints[i].qOffset]<<"\n";}
