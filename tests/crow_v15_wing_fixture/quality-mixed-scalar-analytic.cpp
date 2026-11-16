#include "metalrobo/MetalUnifiedQualitySolver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace metalrobo;
static void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static MRUnifiedQualityBlockGPU block(unsigned row,bool hard){MRUnifiedQualityBlockGPU b{};b.layout={row,1,MR_UNIFIED_QUALITY_SCALAR_INTERVAL,hard?MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY:0u};b.stableKey={500+row,0,0,0};b.scale0=b.scale1={1,1,1,1};b.regularization0=b.regularization1={.02f,1,1,1};b.boundsAndShift={hard?-std::numeric_limits<float>::max():0.f,std::numeric_limits<float>::max(),0,0};return b;}
static void run(int mode,bool reverse){const bool invalid=mode==3;const unsigned n=invalid?3:2;std::vector<MRUnifiedQualityBlockGPU> blocks{block(0,true),block(1,invalid)};if(invalid)blocks.push_back(block(2,false));if(reverse)std::reverse(blocks.begin(),blocks.end());std::vector<float>A=invalid?std::vector<float>{1,0,0,0,1,0,0,0,1}:mode==2?std::vector<float>{1,.5f,.5f,1}:std::vector<float>{1,0,0,1};std::vector<float>J=invalid?std::vector<float>{1,0,0,1,0,0,0,1,0}:std::vector<float>{1,0,0,1};std::vector<float>bias=invalid?std::vector<float>{-.1f,-.2f,.2f}:std::vector<float>{-.1f,mode==1?.2f:-.2f};std::vector<float>zero(n,0),imp(blocks.size(),0);MetalUnifiedQualityProblem p;p.problemCount=1;p.generalizedVelocityCount=n;p.rowCount=blocks.size();p.blocks=blocks;p.dynamics=A;p.jacobian=J;p.bias=bias;p.freeVelocity=zero;p.warmVelocity=zero;p.warmImpulses=imp;MetalUnifiedQualityConfig c;c.maximumNewtonIterations=40;c.optimalityTolerance=1e-5f;c.feasibilityTolerance=2e-5f;MetalUnifiedQualityResult r;r.velocity={123};r.impulses={456};auto d=solveMetalUnifiedQuality(p,r,c);
 if(invalid){check(!d.succeeded()&&d.dispatched&&!d.published&&r.velocity==std::vector<float>{123}&&r.impulses==std::vector<float>{456},"dependent hard rows must reject transactionally");return;}
 check(d.succeeded()&&d.published&&r.statuses.size()==1&&r.statuses[0].solvePath==MR_UNIFIED_QUALITY_PATH_EXACT_MIXED_SCALAR,"mixed solve not published on expected path");const double y=mode==1?0:mode==2?.199/1.02:.2/1.02;const double normal=mode==1?0:y+(mode==2?.05:0);check(std::abs(r.velocity[0]-.1)<2e-5&&std::abs(r.velocity[1]-y)<2e-5&&std::abs(r.impulses[1]-normal)<2e-5,"analytic mixed solution mismatch");const double hard=r.velocity[0]-.1,slack=r.velocity[1]+bias[1]+.02*r.impulses[1],normalImpulse=r.impulses[1];check(std::abs(hard)<2e-5&&slack>=-2e-5&&normalImpulse>=0&&std::abs(slack*normalImpulse)<2e-5,"mixed KKT audit failed");const double ax=r.velocity[0]+(mode==2?.5*r.velocity[1]:0),ay=r.velocity[1]+(mode==2?.5*r.velocity[0]:0);const double objective=.5*(r.velocity[0]*ax+r.velocity[1]*ay)+.01*normalImpulse*normalImpulse;check(std::abs(r.statuses[0].certificates1.w-objective)<2e-5,"mixed objective certificate mismatch");}
static void checkUnsupportedTwoContacts(){
 std::vector<MRUnifiedQualityBlockGPU> blocks{block(0,true),block(1,false),block(2,false)};
 MetalUnifiedQualityProblem p;p.problemCount=1;p.generalizedVelocityCount=3;p.rowCount=3;p.blocks=blocks;
 std::vector<float> identity{1,0,0,0,1,0,0,0,1},bias{-.1f,-.2f,-.3f},zero{0,0,0};
 p.dynamics=identity;p.jacobian=identity;p.bias=bias;p.freeVelocity=zero;p.warmVelocity=zero;p.warmImpulses=zero;
 MetalUnifiedQualityResult r;r.velocity={123};r.impulses={456};auto d=solveMetalUnifiedQuality(p,r,{});
 if(d.succeeded())check(r.statuses.size()==1&&r.statuses[0].solvePath!=MR_UNIFIED_QUALITY_PATH_EXACT_MIXED_SCALAR,"two contacts must not use single-contact path");
 else check(!d.published&&r.velocity==std::vector<float>{123}&&r.impulses==std::vector<float>{456},"unsupported case must reject transactionally");
}
int main(){try{checkUnsupportedTwoContacts();for(int i=0;i<4;++i)for(bool reverse:{false,true})run(i,reverse);std::cout<<"mixed scalar analytical and transactional checks passed\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
