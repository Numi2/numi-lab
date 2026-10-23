#define main originalReplayMain
#include "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/cardboard_vtccd_iteration_budget_check.mm"
#undef main
int main(){@autoreleasepool{
 id<MTLDevice> d=MTLCreateSystemDefaultDevice(); NSError* e=nil;
 id<MTLLibrary> l=[d newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&e];
 auto pipe=makeHumanEdgeFailurePipeline(d,l); auto q=[d newCommandQueue];
 const float h=1e-5f, ro=8.f*1.1920928955078125e-7f*1e-3f;
 const auto rec=makeEEIntervalRecord(h, EEPoint{-1e-3f,0,0}, EEPoint{1e-3f,0,0},EEPoint{-1e-3f,h+1.5f*ro,0}, EEPoint{1e-3f,h+1.5f*ro,0});
 const auto rows=runEEIntervalCase(d,q,pipe,rec);
 for(unsigned stage: {0u,6u}) {std::cout<<stage; for(float v:rows[2*stage])std::cout<<","<<std::setprecision(17)<<v; std::cout<<"\n";}
 return 0;}}
