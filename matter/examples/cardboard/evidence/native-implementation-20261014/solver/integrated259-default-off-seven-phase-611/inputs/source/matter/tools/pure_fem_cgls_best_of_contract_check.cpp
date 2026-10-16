#include "numi/matter/detail/pure_fem_armijo_cgls_arena_plan.hpp"
#include "numi/matter/detail/pure_fem_cgls_best_of_policy.hpp"
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace numi::matter::detail;
static void require(bool v){if(!v)throw std::runtime_error("best-of CGLS contract failed");}
int main(){
 FEMPureArmijoArenaShape s{};s.environmentCount=2;s.femNodeCount=8192;s.tetrahedronCount=32768;
 s.unknownCount=4u*8192u;s.maxBufferLength=512ull*1024ull*1024ull;
 s.accountedResidentBytes=6ull*1024ull*1024ull*1024ull;s.oneStaticExplicitFEM=true;
 s.includeCauchy=true;s.includeCGLS=true;s.includeRightScale=true;s.includeBestOf=true;
 auto p=planFEMPureArmijoCGLSArena(s);require(p.admitted()&&p.includeBestOf&&p.bufferCount==34u);
 require(std::strcmp(p.buffers[p.bufferCount-9u].name,"cgls_best_direction")==0);
 require(std::strcmp(p.buffers[p.bufferCount-1u].name,"cgls_best_committed")==0);
 auto bad=s;bad.includeRightScale=false;bad.includeCGLS=false;
 require(planFEMPureArmijoCGLSArena(bad).failure==FEMPureArmijoArenaFailure::best_of_layout);
 bad=s;bad.includeBestOf=false;bad.includeRightScale=false;bad.includeCauchy=false;
 require(planFEMPureArmijoCGLSArena(bad).failure==FEMPureArmijoArenaFailure::cgls_layout);
 bad=s;bad.environmentCount=3;bad.unknownCount=6u*8192u;
 require(planFEMPureArmijoCGLSArena(bad).failure==FEMPureArmijoArenaFailure::cgls_layout);
 FEMPureCGLSBestOfEligibility e{true,true,true,true,1.e-3f,1.e-7f,9.e-4f,.25f};
 require(femPureCGLSBestOfEligible(e));e.originalCertificate=false;require(!femPureCGLSBestOfEligible(e));
 e.originalCertificate=true;e.acceptedResidual=1.e-3f;require(!femPureCGLSBestOfEligible(e));
 e.acceptedResidual=9.e-4f;e.baseResidual=1.e-7f;require(!femPureCGLSBestOfEligible(e));
 const unsigned char n[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
 const unsigned char cgls[16]={15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};unsigned char out[16]{};
 FEMPureCGLSBestOfChoice c{true,true,true,true,.25f,.20f};
 require(femPureCGLSBestOfResolveBytes(c,n,cgls,out)&&std::memcmp(out,cgls,16)==0);
 auto reject=[&](FEMPureCGLSBestOfChoice x){std::memset(out,0xaa,16);
   require(!femPureCGLSBestOfResolveBytes(x,n,cgls,out)&&std::memcmp(out,n,16)==0);};
 c.cglsResidual=.25f;reject(c);c.cglsResidual=.3f;reject(c);
 c.cglsResidual=.2f;c.cglsAccepted=false;reject(c);c.cglsAccepted=true;c.cglsHealthy=false;reject(c);
 c.cglsHealthy=true;c.cglsFinite=false;reject(c);c.cglsFinite=true;c.savedNewtonCandidate=false;reject(c);
 c.savedNewtonCandidate=true;c.newtonResidual=std::numeric_limits<float>::quiet_NaN();reject(c);
 std::cout<<"pure_fem_cgls_best_of_contract=PASS scratch="<<p.scratchBytes<<" buffers="<<p.bufferCount<<"\n";
}
