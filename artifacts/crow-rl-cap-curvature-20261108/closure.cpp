#include "metalrobo/MetalUnifiedQualitySolver.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <stdexcept>
using namespace metalrobo;
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("METALLIB OUTPUT");
 if(std::filesystem::exists(argv[2]))throw std::runtime_error("output exists");
 std::ofstream out(argv[2]);out<<std::setprecision(17);
 for(auto spec:std::array<std::array<float,2>,5>{{{.25f,0},{1,0},{2,0},{4,0},{.25f,.125f}}})for(unsigned path=0;path<2;++path){
  float radius=spec[0],adh=spec[1],normal=radius-adh;
  MRUnifiedQualityBlockGPU b{};b.layout={0,3,MR_UNIFIED_QUALITY_ELLIPTIC_CONE,0};b.scale0=b.scale1={1,1,1,1};b.regularization0=b.regularization1={1,1,1,1};b.boundsAndShift={adh,normal,0,0};
  std::array<float,9>A{1,0,0,0,1,0,0,0,1},J=A;
  std::array<float,3>bias{-10,3,4},free{-normal,.6f*radius,.8f*radius},warm{0,.0008f,-.0006f},wp{};
  MetalUnifiedQualityProblem problem;problem.problemCount=1;problem.generalizedVelocityCount=3;problem.rowCount=3;problem.blocks=std::span(&b,1);problem.dynamics=A;problem.jacobian=J;problem.bias=bias;problem.freeVelocity=free;problem.warmVelocity=warm;problem.warmImpulses=wp;
  MetalUnifiedQualityConfig c;c.metallibPath=argv[1];c.maximumNewtonIterations=1;c.maximumPCGIterations=128;c.optimalityTolerance=1e-6f;c.feasibilityTolerance=2e-5f;c.directMaximumGeneralizedVelocities=path?1:64;
  MetalUnifiedQualityResult r;auto d=solveMetalUnifiedQuality(problem,r,c);
  MRUnifiedQualityStatusGPU s{};s.code=d.firstGPUStatusCode;
  const bool published=r.statuses.size()==1;if(published)s=r.statuses[0];
  out<<"{\"radius\":"<<radius<<",\"adhesion\":"<<adh<<",\"path\":"<<path<<",\"host_status\":"<<unsigned(d.status)<<",\"code\":"<<s.code<<",\"newton\":"<<(published?std::to_string(s.newtonIterations):"null")<<",\"pcg\":"<<(published?std::to_string(s.pcgIterations):"null")<<",\"residual\":"<<(published?([&]{std::ostringstream z;z<<std::setprecision(17)<<s.certificates0.x;return z.str();})():"null")<<",\"feasibility\":"<<(published?([&]{std::ostringstream z;z<<std::setprecision(17)<<s.certificates0.y;return z.str();})():"null")<<",\"velocity\":[";
  for(unsigned i=0;i<r.velocity.size();++i)out<<(i?",":"")<<r.velocity[i];out<<"],\"impulses\":[";
  for(unsigned i=0;i<r.impulses.size();++i)out<<(i?",":"")<<r.impulses[i];out<<"]}\n";out.flush();
 }
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
