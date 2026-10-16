#include "numi/matter/detail/pure_fem_armijo_cgls_arena_plan.hpp"
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <limits>
using namespace numi::matter::detail;
static void require(bool value) { if(!value) throw std::runtime_error("arena contract failed"); }
static FEMPureArmijoArenaShape shape(){
 FEMPureArmijoArenaShape s{};s.environmentCount=2;s.femNodeCount=8192;s.tetrahedronCount=32768;
 s.unknownCount=2*s.environmentCount*s.femNodeCount;s.maxBufferLength=512ull*1024ull*1024ull;
 s.accountedResidentBytes=6ull*1024ull*1024ull*1024ull;s.oneStaticExplicitFEM=true;return s;
}
int main(){
 auto s=shape();auto a=planFEMPureArmijoCGLSArena(s);
 require(a.admitted()&&a.bufferCount==9&&a.scratchBytes<kFEMPureArmijoMaximumScratchBytes);
 auto e4=shape();e4.environmentCount=4;e4.unknownCount=65536u;
 require(planFEMPureArmijoCGLSArena(e4).admitted());
 auto oneByteShort=shape();oneByteShort.maxBufferLength=a.buffers[0].allocatedBytes-1u;
 require(planFEMPureArmijoCGLSArena(oneByteShort).failure==
         FEMPureArmijoArenaFailure::per_buffer_limit);
 s.includeCauchy=true;s.includeCGLS=true;s.includeRightScale=true;
 auto b=planFEMPureArmijoCGLSArena(s);require(b.admitted()&&b.bufferCount==25);
 require(b.accountedResidentPlusScratchBytes==s.accountedResidentBytes+b.scratchBytes);
 require(b.buffers[b.bufferCount-2].logicalCount==49152u);
 s=shape();s.includeCauchy=true;s.includeCGLS=true;s.environmentCount=3;s.unknownCount=49152;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::cgls_layout);
 s=shape();s.includeRightScale=true;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::right_scale_layout);
 s=shape();s.femNodeCount=8193;s.unknownCount=32772;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::fem_node_count);
 s=shape();s.tetrahedronCount=32769;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::tetrahedron_count);
 s=shape();s.environmentCount=5;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::environment_count);
 s=shape();s.unknownCount++;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::unknown_count);
 s=shape();s.contactPairCount=1;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::unsupported_world);
 s=shape();s.oneStaticExplicitFEM=false;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::unsupported_world);
 s=shape();s.maxBufferLength=16;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::per_buffer_limit);
 s=shape();s.accountedResidentBytes=UINT64_MAX;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::arithmetic_overflow);
 s=shape();s.accountedResidentBytes=16ull*1024ull*1024ull*1024ull;
 require(planFEMPureArmijoCGLSArena(s).failure==FEMPureArmijoArenaFailure::accounted_resident_limit);
 s=shape();s.maxBufferLength=512ull*1024ull*1024ull;s.environmentCount=2;s.femNodeCount=8192;
 std::uint64_t out=0;require(!femPureArmijoCheckedMultiply(UINT64_MAX,2,out));
 require(!femPureArmijoCheckedAdd(UINT64_MAX,1,out));
 std::cout<<"pure_fem_armijo_cgls_arena_plan=PASS scratch="<<b.scratchBytes<<" buffers="<<b.bufferCount<<'\n';
}
