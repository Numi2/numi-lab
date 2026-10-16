#include "numi/matter/detail/true_residual_arena_plan.hpp"
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace {unsigned checks=0;void req(bool v,const char*m){++checks;if(!v)throw std::runtime_error(m);}using namespace numi::matter::detail;FEMTrueResidualArenaShape s(std::uint64_t e,std::uint64_t n,std::uint64_t t,std::uint64_t c,std::uint64_t r=0u,std::uint64_t m=(1ull<<40)){return {e,n,t,c,r,m};}}
int main(){try{
 auto n4=planFEMTrueResidualArena(s(1,4096,32768,4096));req(n4.admitted()&&n4.unknownCount==8192u,"E1/N4096");req(n4.rightHandSide.allocatedBytes==8192u*sizeof(nm_float4),"RHS bytes");
 auto n8=planFEMTrueResidualArena(s(1,8192,32768,4096));req(n8.admitted()&&n8.unknownCount==16384u,"E1/N8192");
 auto box=planFEMTrueResidualArena(s(1,8192,32768,4096,6));req(box.admitted()&&box.unknownCount==16390u,"six-tail N8192");
 auto e4=planFEMTrueResidualArena(s(4,8192,32768,4096));req(e4.admitted()&&e4.aggregateBytes<=e4.maximumAggregateBytes,"maximum aggregate derived from supported envelope");
 auto small=planFEMTrueResidualArena(s(1,8192,32768,4096,0,n8.elementOperator.allocatedBytes-1u));req(!small.admitted()&&small.failure==FEMTrueResidualArenaFailure::per_buffer_limit,"one-byte-short buffer limit");
 req(!planFEMTrueResidualArena(s(5,1,1,0)).admitted(),"E5");req(!planFEMTrueResidualArena(s(1,8193,1,0)).admitted(),"N8193");req(!planFEMTrueResidualArena(s(1,1,32769,0)).admitted(),"T32769");req(!planFEMTrueResidualArena(s(1,1,1,4097)).admitted(),"C4097");req(!planFEMTrueResidualArena(s(2,1,1,0,6)).admitted(),"six-tail E2");req(!planFEMTrueResidualArena(s(1,1,1,0,5)).admitted(),"non-six tail");
 std::uint64_t out=0u;req(!femTrueResidualCheckedMultiply(std::numeric_limits<std::uint64_t>::max(),2u,out),"multiply overflow");req(!femTrueResidualCheckedAdd(std::numeric_limits<std::uint64_t>::max(),1u,out),"add overflow");
 auto zero=planFEMTrueResidualArena(s(1,1,1,0));req(zero.admitted()&&zero.cohesiveOperator.allocatedBytes==sizeof(NMCohesiveFaceVectorGPU),"zero cohesive typed sentinel");
 auto refinedShape=s(2,20682,89088,0);refinedShape.extendedPureFEM=true;auto refined=planFEMTrueResidualArena(refinedShape);req(refined.admitted(),"actual finest coupon");
 auto maxShape=s(4,32768,131072,0);maxShape.extendedPureFEM=true;auto maxPure=planFEMTrueResidualArena(maxShape);req(maxPure.admitted()&&maxPure.aggregateBytes==maxPure.maximumAggregateBytes,"extended maximum aggregate");
 maxShape.cohesiveFaceCount=1;req(planFEMTrueResidualArena(maxShape).failure==FEMTrueResidualArenaFailure::extended_pure_layout,"extended cohesive rejected");
 maxShape.cohesiveFaceCount=0;maxShape.rigidGeneralizedCapacity=6;req(planFEMTrueResidualArena(maxShape).failure==FEMTrueResidualArenaFailure::extended_pure_layout,"extended rigid rejected");
 maxShape.rigidGeneralizedCapacity=0;maxShape.femNodeCount=32769;req(!planFEMTrueResidualArena(maxShape).admitted(),"extended node cap");
 maxShape.femNodeCount=32768;maxShape.tetrahedronCount=131073;req(!planFEMTrueResidualArena(maxShape).admitted(),"extended tet cap");
 std::cout<<"true_residual_arena_plan_checks="<<checks<<" max_E4_N8192_T32768_C4096_bytes="<<e4.aggregateBytes<<" transactions=0\\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}}
