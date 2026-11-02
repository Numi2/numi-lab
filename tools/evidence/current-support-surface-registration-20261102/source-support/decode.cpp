#include "metalrobo/NumiHumanSupport.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
int main(int argc,char**argv){if(argc!=2)return 2;std::ifstream f(argv[1],std::ios::binary);std::vector<char> raw((std::istreambuf_iterator<char>(f)),{});if(raw.size()<sizeof(metalrobo::NumiHumanSupportHeader))return 3;metalrobo::NumiHumanSupportHeader h;std::memcpy(&h,raw.data(),sizeof(h));metalrobo::NumiHumanSupportPayload p;std::string error;if(!metalrobo::decodeNumiHumanSupportPayload(std::as_bytes(std::span(raw)),h.engineBodyCount,h.sourceSha256,p,error)){std::cerr<<error;return 4;}std::cout<<std::setprecision(17)<<"source_row,donor,lx,ly,lz,wx,wy,wz\n";for(auto r:p.contacts)std::cout<<r.sourceGeometryIndex-1u<<","<<r.bodyIndex<<","<<r.localPointX<<","<<r.localPointY<<","<<r.localPointZ<<","<<r.worldWitnessX<<","<<r.worldWitnessY<<","<<r.worldWitnessZ<<"\n";}
