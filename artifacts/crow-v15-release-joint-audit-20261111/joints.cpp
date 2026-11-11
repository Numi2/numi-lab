#define main toe_probe_main
#include "../crow-v15-two-segment-toe-prototype-20261111/probe.cpp"
#undef main
int main(){auto p=toeModel();auto &m=p.mechanics;for(size_t i=0;i<m.joints.size();++i){const auto& j=m.joints[i];std::cout<<m.jointNames[i]<<" q "<<j.qOffset<<" v "<<j.vOffset<<" axis "<<j.axis0.x<<" "<<j.axis0.y<<" "<<j.axis0.z<<" parent "<<m.bodyNames[j.parentBody]<<" child "<<m.bodyNames[j.childBody]<<"\n";}}
