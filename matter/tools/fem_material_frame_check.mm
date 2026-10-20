#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include "cardiac_material_reference.hpp"
#include "../src/native_tangent_source.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>

// Synthetic conforming two-tet constitutive and transaction fixture. The
// Guccione law is sourced; this geometry, density, loading and other path
// controls are not a calibrated heart or an anatomical-wall admission.
namespace {
using namespace numi::matter;
namespace ref=numi_cardiac_reference;
using ref::Matrix;
using Vec=std::array<double,3>;
void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::string number(double x){std::ostringstream s;s<<std::setprecision(17)<<x;return s.str();}
std::string diagnostic(const std::vector<Diagnostic>& values){std::string out;for(const auto& v:values)out+=v.message+"; ";return out;}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
template<class T> std::vector<T> repeat(const std::vector<T>& a,unsigned count=2){std::vector<T> out;for(unsigned i=0;i<count;++i)out.insert(out.end(),a.begin(),a.end());return out;}
template<class T> void paired(const std::vector<T>& a,const char* label){need(a.size()%2==0,std::string(label)+" arity");const auto n=a.size()/2;need(!n||std::memcmp(a.data(),a.data()+n,n*sizeof(T))==0,std::string(label)+" paired environments differ");}
template<class T> void envSame(const std::vector<T>& a,const std::vector<T>& b,unsigned e,const char* label){need(a.size()==b.size()&&a.size()%2==0,std::string(label)+" arity");const auto n=a.size()/2;need(!n||std::memcmp(a.data()+e*n,b.data()+e*n,n*sizeof(T))==0,std::string(label)+" isolated transaction changed state");}
template<class T> id<MTLBuffer> buffer(id<MTLDevice> d,const std::vector<T>& input){const T zero{};return [d newBufferWithBytes:(input.empty()?&zero:input.data()) length:std::max(std::size_t(1),input.size())*sizeof(T) options:MTLResourceStorageModeShared];}
Vec xyz(const nm_float4& a){return {a.x,a.y,a.z};}
Vec sub(Vec a,const Vec& b){for(unsigned i=0;i<3;++i)a[i]-=b[i];return a;}
Vec transform(const Matrix& a,const Vec& b){Vec c{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)c[i]+=a[3*i+j]*b[j];return c;}
double dot(const Vec& a,const Vec& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
Matrix outer(const Vec& a,const Vec& b){Matrix c{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)c[3*i+j]=a[i]*b[j];return c;}
Matrix columns(const Vec& a,const Vec& b,const Vec& c){return {a[0],b[0],c[0],a[1],b[1],c[1],a[2],b[2],c[2]};}
Matrix inverseRest(const NMTetrahedronGPU& t){return {t.inverseRestRow0.x,t.inverseRestRow0.y,t.inverseRestRow0.z,t.inverseRestRow1.x,t.inverseRestRow1.y,t.inverseRestRow1.z,t.inverseRestRow2.x,t.inverseRestRow2.y,t.inverseRestRow2.z};}
Matrix frame(const nm_float4& raw){
    const double inv=1/std::sqrt(double(raw.x)*raw.x+double(raw.y)*raw.y+double(raw.z)*raw.z+double(raw.w)*raw.w);
    const double x=raw.x*inv,y=raw.y*inv,z=raw.z*inv,w=raw.w*inv;
    return {1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
}
void set(MaterialProgram& m,const std::string& name,double value){for(auto& p:m.parameters)if(p.name==name){p.defaultValue=value;return;}throw std::runtime_error("missing "+name);}
WorldSource source(const MaterialProgram& material,bool framed=true,bool mixed=false){
    WorldSource s;s.environmentCount=2;s.gravity={0,0,0};s.frameTimestep=1e-4;s.materials={material};
    s.mixedSolver.relativeResidual=1e-7;s.mixedSolver.newtonIterations=12;s.mixedSolver.fgmresIterations=64;
    ObjectSource o;o.name="synthetic_shared_node_material_frames";o.representation=Representation::fem;o.mixedFEM=mixed;
    o.deformableContact=false;o.deformableSelfContact=false;o.characteristicLength=.01;
    o.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}},{{0,0,-.01}}};
    o.tetrahedra={{{0,1,2,3}},{{0,2,1,4}}};o.femFixedNodes={0,1,2};o.femInitialVelocity={.025,.017,.013};
    if(framed){o.femMaterialFrameRotations={{{0,0,std::sin(.305),std::cos(.305)}},{{0,std::sin(-.245),0,std::cos(-.245)}}};
        // Opaque nonzero fixture identity. It represents synthetic frame
        // authoring, not a source anatomical mesh or measured microstructure.
        o.femMaterialFrameSourceIdentity={0xf731c96c934e625eull,0x836db2ec25021611ull,0x71a72c65dc7b2f5aull,0x36ab5d1b57e2bb96ull};}
    s.objects={o};return s;
}
CompiledWorld cook(const WorldSource& s){auto c=compileWorld(s,{.maximumRateExponent=0,.emitSpecializedMetal=false});need(c.succeeded(),"frame fixture compile: "+diagnostic(c.diagnostics));return std::move(c.world);}

struct Device {
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLLibrary> library;
    id<MTLComputePipelineState> forcePipeline,operatorPipeline,tangentCachePipeline,cacheProjectionReusePipeline,cachePreparePipeline,cacheColumnsPipeline,cacheBundleCooperativePipeline,cacheFinalizePipeline,cacheFinalizeDiagnosticPipeline,factorTestPipeline,newtonTailDispatchPipeline;
    std::uint32_t functionConstant30PipelineCount=0u;
    std::uint32_t functionConstant32PipelineCount=0u;
    Device(const bool specializeStatelessMaterials=false,
           const bool nativeDiagnosticCheckOnly=false){device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal unavailable");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound&&[[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple Metal required");
        queue=[device newCommandQueue];NSError* error=nil;
        const char* sourcePath=std::getenv("NUMI_MATTER_TEST_SOURCE");
        if(sourcePath!=nullptr&&sourcePath[0]!=0){
            NSString* path=[NSString stringWithUTF8String:sourcePath];
            NSString* source=[NSString stringWithContentsOfFile:path
                encoding:NSUTF8StringEncoding error:&error];
            need(source!=nil,"native Matter test source unavailable");
            MTLCompileOptions* options=[MTLCompileOptions new];
            options.languageVersion=MTLLanguageVersion4_0;
            options.mathMode=MTLMathModeSafe;
            options.mathFloatingPointFunctions=
                MTLMathFloatingPointFunctionsPrecise;
            library=[device newLibraryWithSource:source options:options
                error:&error];
        }else{
            library=[device newLibraryWithURL:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:NUMI_MATTER_METALLIB]]
                error:&error];
        }
        need(library!=nil,"production Matter Metal program unavailable");
        if(nativeDiagnosticCheckOnly){
            cacheFinalizeDiagnosticPipeline=make(
                @"numi_matter_metal::nm_fem_finalize_algorithmic_tangent_columns",
                false,false,true);
            need(functionConstant32PipelineCount==1u,
                "focused native diagnostic did not specialize only finalizer constant 32");
            return;
        }
        // Internal forces evaluate stress/projection only and have no
        // algorithmic-tangent function constant. Keep that independent path.
        forcePipeline=make(@"numi_matter_metal::nm_fem_internal_forces");
        newtonTailDispatchPipeline=make(
            @"numi_matter_metal::nm_fgmres_prepare_newton_tail_dispatch");
        operatorPipeline=make(@"numi_matter_metal::nm_fem_apply_operator_elements",
            specializeStatelessMaterials,specializeStatelessMaterials);
        tangentCachePipeline=make(@"numi_matter_metal::nm_fem_build_algorithmic_tangent_cache",
            specializeStatelessMaterials,specializeStatelessMaterials);
        cacheProjectionReusePipeline=make(@"numi_matter_metal::nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic",
            specializeStatelessMaterials,specializeStatelessMaterials);
        cachePreparePipeline=make(@"numi_matter_metal::nm_fem_prepare_algorithmic_tangent_columns");
        cacheColumnsPipeline=make(@"numi_matter_metal::nm_fem_build_algorithmic_tangent_cache_columns",
            specializeStatelessMaterials,specializeStatelessMaterials);
        cacheBundleCooperativePipeline=make(@"numi_matter_metal::nm_fem_build_algorithmic_tangent_cache_bundle_cooperative",
            specializeStatelessMaterials,specializeStatelessMaterials);
        cacheFinalizePipeline=make(@"numi_matter_metal::nm_fem_finalize_algorithmic_tangent_columns");
        factorTestPipeline=make(@"numi_matter_metal::nm_fem_test_compensated_local_factor");
        if(specializeStatelessMaterials)
            need(functionConstant30PipelineCount>=5u,
                "focused specialization did not specialize all required stateless FEM PSOs");}
    id<MTLComputePipelineState> make(NSString* name,
                                     const bool specializeStatelessMaterials=false,
                                     const bool requireFunctionConstant30=false,
                                     const bool enableNativeDiagnostic32=false){
        NSError* error=nil;
        auto f=[library newFunctionWithName:name];
        need(f!=nil,"production FEM function missing");
        bool hasFunctionConstant30=false;
        bool hasFunctionConstant32=false;
        for(MTLFunctionConstant* constant in
                f.functionConstantsDictionary.allValues){
            if(constant.index==30u)hasFunctionConstant30=true;
            if(constant.index==32u)hasFunctionConstant32=true;
        }
        if(requireFunctionConstant30)
            need(hasFunctionConstant30,
                std::string("required function constant 30 missing from ")+
                name.UTF8String);
        if(enableNativeDiagnostic32)
            need(hasFunctionConstant32,
                std::string("required function constant 32 missing from ")+
                name.UTF8String);
        MTLFunctionConstantValues* values=[MTLFunctionConstantValues new];
        bool hasConstants=false;
        for(MTLFunctionConstant* constant in f.functionConstantsDictionary.allValues){
            if(constant.index!=30u && constant.index!=31u && constant.index!=32u)continue;
            const bool enabled=constant.index==30u ? specializeStatelessMaterials :
                (constant.index==32u ? enableNativeDiagnostic32 : false);
            [values setConstantValue:&enabled type:MTLDataTypeBool atIndex:constant.index];
            hasConstants=true;
        }
        if(hasConstants){
            if(specializeStatelessMaterials&&hasFunctionConstant30)
                ++functionConstant30PipelineCount;
            if(enableNativeDiagnostic32&&hasFunctionConstant32)
                ++functionConstant32PipelineCount;
            f=[library newFunctionWithName:name constantValues:values error:&error];
            need(f!=nil,"specialized FEM function failed: "+
                std::string(name.UTF8String));
        }
        // Constant 31 is explicitly false; this instrument compares bytecode.
        auto pipeline=[device newComputePipelineStateWithFunction:f error:&error];
        need(pipeline!=nil,"production FEM pipeline failed: "+
            std::string(name.UTF8String));
        return pipeline;
    }
};
// Incomplete/failed columns must never publish a valid cache, and a completed
// Newton tail must preserve its scratch rather than resetting it.
void checkColumnValidityPublication(Device& d) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=2;dispatch.tetrahedronCount=3;
    NMMicrostepGPU micro{};micro.solverIteration=1;
    std::uint32_t force=0;
    std::vector<NMFGMRESStateGPU> states(2);states[1].nonlinear.w=1;
    const auto state=buffer(d.device,states);
    const auto statuses=buffer(d.device,std::vector<NMMatterStatusGPU>(2));
    const auto run=[&](id<MTLComputePipelineState> pipeline,std::vector<std::uint32_t> values) {
        auto valid=buffer(d.device,values);auto command=[d.queue commandBuffer];
        auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:1];
        [encoder setBuffer:valid offset:0 atIndex:13];
        [encoder setBuffer:statuses offset:0 atIndex:14];
        [encoder setBuffer:state offset:0 atIndex:15];
        [encoder setBytes:&force length:sizeof(force) atIndex:16];
        [encoder dispatchThreads:MTLSizeMake(values.size(),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,"column validity command failed");
        const auto* result=static_cast<const std::uint32_t*>(valid.contents);
        return std::vector<std::uint32_t>(result,result+values.size());
    };
    need(run(d.cacheFinalizePipeline,{0x1ff,1,0xff,1,0,1})==
             std::vector<std::uint32_t>({1,0,0,1,0,1}),"partial columns published or completed tail changed");
    need(run(d.cachePreparePipeline,{1,1,1,1,0,1})==
             std::vector<std::uint32_t>({0,0,0,1,0,1}),"cache reset changed completed tail");
    micro.solverIteration=0;
    need(run(d.cachePreparePipeline,{1,1,1,1,0,1})==std::vector<std::uint32_t>(6,0),
         "new transaction retained prior convergence latch cache");
    micro.solverIteration=1;force=1;
    need(run(d.cachePreparePipeline,{1,1,1,1,0,1})==std::vector<std::uint32_t>(6,0),
         "forced completed cache rebuild did not reset all entries");
    std::cout<<"parallel_tangent_validity=passed partial_columns=rejected completed_tail=preserved new_transaction=reset\n";
}
void checkNewtonTailDispatch(Device& d) {
    need(d.newtonTailDispatchPipeline!=nil,
         "completed-Newton tail range pipeline is unavailable");
    const auto runCase=[&](const std::uint32_t environments,
                           const std::uint32_t iteration,
                           const bool completedLatch,
                           const bool healthy) {
        NMMatterDispatchGPU dispatch{};
        dispatch.environmentCount=environments;
        dispatch.femNodeCount=257u;
        dispatch.tetrahedronCount=513u;
        NMMicrostepGPU micro{};
        micro.solverIteration=iteration;
        std::vector<NMFGMRESStateGPU> states(environments);
        if(completedLatch)
            for(auto& state:states) state.nonlinear.w=1.0f;
        std::vector<NMMatterStatusGPU> statuses(environments);
        for(std::uint32_t e=0u;e<environments;++e) {
            statuses[e].code=NM_STATUS_SUCCESS;
            statuses[e].environment=e;
        }
        if(!healthy) statuses[0].code=NM_STATUS_INVALID_DEFORMATION;
        const auto stateBuffer=buffer(d.device,states);
        const auto statusBuffer=buffer(d.device,statuses);
        const NMIndirectDispatchGPU sentinel{77u,78u,79u,80u};
        auto nodeRange=buffer(d.device,std::vector<NMIndirectDispatchGPU>{sentinel});
        auto tetRange=buffer(d.device,std::vector<NMIndirectDispatchGPU>{sentinel});
        auto command=[d.queue commandBuffer];
        auto encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:d.newtonTailDispatchPipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
        [encoder setBuffer:stateBuffer offset:0u atIndex:2u];
        [encoder setBuffer:statusBuffer offset:0u atIndex:3u];
        [encoder setBuffer:nodeRange offset:0u atIndex:4u];
        [encoder setBuffer:tetRange offset:0u atIndex:5u];
        [encoder dispatchThreads:MTLSizeMake(1u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,
             "completed-Newton tail range fixture command failed");
        const bool skipped=environments==1u && iteration>0u &&
            completedLatch && healthy;
        const auto expected=[&](const std::uint32_t extent) {
            return skipped?0u:extent/256u+(extent%256u!=0u?1u:0u);
        };
        const auto* nodes=static_cast<const NMIndirectDispatchGPU*>(
            nodeRange.contents);
        const auto* tets=static_cast<const NMIndirectDispatchGPU*>(
            tetRange.contents);
        need(nodes[0].threadgroupsX==expected(environments*257u) &&
             nodes[0].threadgroupsY==1u && nodes[0].threadgroupsZ==1u &&
             tets[0].threadgroupsX==expected(environments*513u) &&
             tets[0].threadgroupsY==1u && tets[0].threadgroupsZ==1u,
             "completed-Newton tail helper emitted an incorrect full/zero range");
        return skipped;
    };
    need(runCase(1u,1u,true,true),
         "healthy completed latch did not produce zero assembly ranges");
    need(!runCase(1u,0u,true,true),
         "iteration zero incorrectly reused the prior transaction latch");
    need(!runCase(1u,1u,false,true),
         "unlatched Newton state incorrectly suppressed assembly");
    need(!runCase(1u,1u,true,false),
         "unhealthy status incorrectly suppressed assembly");
    need(!runCase(2u,1u,true,true),
         "multi-environment dispatch incorrectly used the single-environment range");
    std::cout<<"newton_tail_ranges=passed zero=completed_latch full=iteration0_unlatched_unhealthy_multienv\n";
}
void checkNativeDiagnosticFinalizer(Device& d) {
    need(d.cacheFinalizeDiagnosticPipeline!=nil &&
         d.functionConstant32PipelineCount==1u,
         "diagnostic-only finalizer pipeline was not specialized at constant 32");
    NMMatterDispatchGPU dispatch{};
    dispatch.environmentCount=1u;
    dispatch.tetrahedronCount=4u;
    NMMicrostepGPU micro{};
    micro.solverIteration=3u;
    const std::uint32_t forceCompletedCacheRebuild=1u;
    std::vector<NMFGMRESStateGPU> solver(1u);
    solver[0].nonlinear.w=1.0f;
    std::vector<NMMatterStatusGPU> statuses(1u);
    constexpr std::uint32_t verificationMarker=0x80000000u;
    std::vector<std::uint32_t> entries{0x1ffu,0u,
        verificationMarker,verificationMarker};
    const auto solverBuffer=buffer(d.device,solver);
    const auto statusBuffer=buffer(d.device,statuses);
    const auto entriesBuffer=buffer(d.device,entries);
    auto command=[d.queue commandBuffer];
    auto encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:d.cacheFinalizeDiagnosticPipeline];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
    [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
    [encoder setBuffer:entriesBuffer offset:0u atIndex:13u];
    [encoder setBuffer:statusBuffer offset:0u atIndex:14u];
    [encoder setBuffer:solverBuffer offset:0u atIndex:15u];
    [encoder setBytes:&forceCompletedCacheRebuild
              length:sizeof(forceCompletedCacheRebuild) atIndex:16u];
    [encoder dispatchThreads:MTLSizeMake(entries.size(),1u,1u)
        threadsPerThreadgroup:MTLSizeMake(32u,1u,1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    need(command.status==MTLCommandBufferStatusCompleted,
         "native diagnostic finalizer fixture command failed");
    const auto* finalized=static_cast<const std::uint32_t*>(
        entriesBuffer.contents);
    need(finalized[0]==1u&&finalized[1]==0u&&finalized[2]==0u&&
         finalized[3]==0u,
         "diagnostic finalizer did not normalize valid and marked cache words");
    const auto* result=static_cast<const NMMatterStatusGPU*>(
        statusBuffer.contents);
    need(result[0].code==NM_STATUS_NONLINEAR_SOLVER_FAILURE &&
         result[0].objectIndex==NM_INVALID_INDEX &&
         result[0].failingIndex==2u &&
         result[0].diagnostics.x==32.0f &&
         result[0].diagnostics.y==2.0f,
         "native verification marker did not force a deterministic solver failure");
    std::cout<<"native_verification_failure=passed code="
        <<result[0].code<<" first_marker_tet="<<result[0].failingIndex
        <<" marker_cleared=true cache_normalized=true"<<std::endl;
}
void checkCompensatedLocalFactor(Device& d) {
    constexpr unsigned n=15u;
    constexpr unsigned rhsCount=9u;
    constexpr unsigned caseCount=3u;
    using WordPair=std::array<float,2>;
    std::vector<WordPair> matrices(caseCount*n*n);
    std::vector<WordPair> rhs(caseCount*rhsCount*n);
    // A nonsymmetric, well-conditioned matrix with a forced first-column row
    // pivot and low-word perturbations. Each RHS mixes cancellation scales.
    for(unsigned row=0;row<n;++row)for(unsigned column=0;column<n;++column) {
        const int code=int((row*37u+column*19u+7u)%17u)-8;
        const float value=row==column?4.0f+0.125f*row:0.017f*code;
        matrices[row*n+column]={value,std::ldexp(
            ((row+column)&1u)?1.0f:-1.0f,-25)};
    }
    for(unsigned column=0;column<n;++column)
        std::swap(matrices[column],matrices[(n-1u)*n+column]);
    // A second nonsymmetric system starts with rows [2,0,1,...]. The first
    // pivot swaps at column zero and the second at column one, after nonzero
    // L coefficients exist, exercising full-row swaps in the stored factors.
    const std::vector<WordPair> pivotSource(
        matrices.begin(), matrices.begin() + n*n);
    const unsigned permutedCase=2u*n*n;
    for(unsigned row=0;row<n;++row) {
        const unsigned sourceRow=row<3u?(row+1u)%3u:row;
        std::copy_n(pivotSource.begin()+sourceRow*n,n,
                    matrices.begin()+permutedCase+row*n);
    }
    for(unsigned testCase=0;testCase<caseCount;++testCase)
        for(unsigned r=0;r<rhsCount;++r)for(unsigned row=0;row<n;++row) {
            const int code=int((r*23u+row*11u+testCase*7u+3u)%19u)-9;
            const float high=(row%5u==0u?8192.0f:0.125f)*float(code);
            const float low=std::ldexp(
                ((r+row+testCase)&1u)?1.0f:-1.0f,-17);
            rhs[testCase*rhsCount*n+r*n+row]={high,low};
        }
    // The second matrix lies below the production pivot floor, so both
    // one-RHS and factor-once solves must reject it for all nine RHS.
    const unsigned second= n*n;
    for(unsigned i=0;i<n;++i)matrices[second+i*n+i]={i+1u==n?1.0e-9f:1.0f,0.0f};
    const auto matrixBuffer=buffer(d.device,matrices);
    const auto rhsBuffer=buffer(d.device,rhs);
    const auto factoredBuffer=buffer(d.device,
        std::vector<WordPair>(caseCount*rhsCount*n));
    const auto legacyBuffer=buffer(d.device,
        std::vector<WordPair>(caseCount*rhsCount*n));
    const auto residualBuffer=buffer(d.device,
        std::vector<float>(caseCount*rhsCount,-1.0f));
    const auto maskBuffer=buffer(d.device,std::vector<std::uint32_t>(caseCount*3u));
    const auto pivotBuffer=buffer(d.device,
        std::vector<std::uint32_t>(caseCount*n,0xffffffffu));
    const std::uint32_t count=n;
    auto command=[d.queue commandBuffer];auto encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:d.factorTestPipeline];
    [encoder setBuffer:matrixBuffer offset:0 atIndex:0];
    [encoder setBuffer:rhsBuffer offset:0 atIndex:1];
    [encoder setBuffer:factoredBuffer offset:0 atIndex:2];
    [encoder setBuffer:legacyBuffer offset:0 atIndex:3];
    [encoder setBuffer:residualBuffer offset:0 atIndex:4];
    [encoder setBuffer:maskBuffer offset:0 atIndex:5];
    [encoder setBytes:&count length:sizeof(count) atIndex:6];
    [encoder setBuffer:pivotBuffer offset:0 atIndex:7];
    [encoder dispatchThreads:MTLSizeMake(caseCount,1,1)
        threadsPerThreadgroup:MTLSizeMake(2,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    need(command.status==MTLCommandBufferStatusCompleted,
         "compensated factor test command failed");
    const auto* masks=static_cast<const std::uint32_t*>(maskBuffer.contents);
    const auto* pivots=static_cast<const std::uint32_t*>(pivotBuffer.contents);
    need(pivots[2u*n]==n-1u&&pivots[2u*n+1u]==n-1u,
         "later-pivot fixture did not swap rows after nonzero L was stored");
    need(masks[0]==1u&&masks[1]==0x1ffu&&masks[2]==0x1ffu,
         "row-pivot compensated factor did not solve all nine RHS");
    need(masks[3]==0u&&masks[4]==0u&&masks[5]==0u,
         "near-singular compensated factor did not preserve pivot-floor rejection");
    need(masks[6]==1u&&masks[7]==0x1ffu&&masks[8]==0x1ffu,
         "later-pivot compensated factor did not solve all nine RHS");
    const auto* factored=static_cast<const WordPair*>(factoredBuffer.contents);
    const auto* legacy=static_cast<const WordPair*>(legacyBuffer.contents);
    const auto* residual=static_cast<const float*>(residualBuffer.contents);
    double maximumParity=0.0,maximumScale=0.0,maximumResidual=0.0;
    const unsigned parityCases[2]={0u,2u};
    for(const unsigned testCase:parityCases)for(unsigned r=0;r<rhsCount;++r) {
        const unsigned base=testCase*rhsCount*n+r*n;
        maximumResidual=std::max(maximumResidual,
            double(residual[testCase*rhsCount+r]));
        for(unsigned row=0;row<n;++row) {
            const unsigned i=base+row;
            const double x=double(factored[i][0])+factored[i][1];
            const double y=double(legacy[i][0])+legacy[i][1];
            maximumParity=std::max(maximumParity,std::abs(x-y));
            maximumScale=std::max(maximumScale,std::abs(y));
        }
    }
    need(maximumParity<=2.0e-6*std::max(1.0,maximumScale),
         "factor-once RHS solutions differ from legacy compensated elimination");
    need(maximumResidual<=2.0e-3,
         "factor-once solution has an excessive compensated local residual");
    std::cout<<std::setprecision(12)
        <<"compensated_factor_rhs=9 pivoted_matrix=passed"
        <<" near_singular_rejected=passed"
        <<" maximum_solution_parity="<<maximumParity
        <<" maximum_local_residual="<<maximumResidual<<"\n";
}
void checkSharedFactorCompilerProof(MaterialProgram paperMaterial) {
    constexpr std::uint32_t tag =
        NM_MATERIAL_SHARED_15_STATE_TANGENT_FACTORIZATION;
    paperMaterial.genericCompensatedImplicitProjection = true;
    paperMaterial.genericBilinear15AccurateProjection = true;
    const CompiledWorld tagged = cook(source(paperMaterial, true, false));
    need(tagged.materials.size() == 1u &&
         (tagged.materials[0].flags & tag) != 0u,
         "eligible 15-state material did not receive the shared-factor proof tag");
    need(tagged.materials[0].validityProgram != NM_INVALID_INDEX,
         "shared-factor native fixture does not exercise its validity program");
    std::string error;
    need(validateCompiledWorldLayout(tagged, &error),
         "compiler-tagged shared-factor world failed layout proof: " + error);

    const auto replaceRoleLeafOpcode = [](CompiledWorld& world,
                                          const std::uint32_t first,
                                          const std::uint32_t count,
                                          const std::uint32_t replacement) {
        if (first == NM_INVALID_INDEX ||
            first > world.scalarPrograms.size() ||
            count > world.scalarPrograms.size() - first) return false;
        const auto replaceLeaf = [&](const bool constantsOnly) {
            for (std::uint32_t local = 0u; local < count; ++local) {
                const auto& scalar = world.scalarPrograms[first + local];
                if (scalar.firstInstruction > world.instructions.size() ||
                    scalar.instructionCount > world.instructions.size() -
                        scalar.firstInstruction) return false;
                for (std::uint32_t item = 0u;
                     item < scalar.instructionCount; ++item) {
                    auto& instruction = world.instructions[
                        scalar.firstInstruction + item];
                    const bool leaf = instruction.opcode == NM_EXPR_CONSTANT ||
                        instruction.opcode == NM_EXPR_PARAMETER ||
                        instruction.opcode == NM_EXPR_F ||
                        instruction.opcode == NM_EXPR_DF ||
                        instruction.opcode == NM_EXPR_STATE ||
                        instruction.opcode == NM_EXPR_RATE ||
                        instruction.opcode == NM_EXPR_NEXT_STATE;
                    if (!leaf || (constantsOnly &&
                                  instruction.opcode != NM_EXPR_CONSTANT))
                        continue;
                    // These opcodes each push one scalar. Use a valid tensor
                    // component while leaving the stack contract unchanged.
                    instruction.opcode = replacement;
                    instruction.index = 0u;
                    return true;
                }
            }
            return false;
        };
        return replaceLeaf(true) || replaceLeaf(false);
    };
    const auto requireDirectionRejected = [&](
            const std::uint32_t first, const std::uint32_t count,
            const char* role) {
        CompiledWorld damaged = tagged;
        need(replaceRoleLeafOpcode(damaged, first, count, NM_EXPR_DF),
             std::string("shared-factor proof fixture has no input leaf in ") + role);
        damaged.fingerprint = compiledWorldFingerprint(damaged);
        std::string diagnosticText;
        need(!validateCompiledWorldLayout(damaged, &diagnosticText) &&
             diagnosticText.find(
                 "shared 15-state tangent factorization tag lacks its emitted no-direction proof") !=
                 std::string::npos,
             std::string("direction opcode in ") + role +
                 " was not rejected by the shared-factor proof: " + diagnosticText);
    };
    requireDirectionRejected(
        tagged.materials[0].implicitJacobianProgramOffset,
        tagged.materials[0].stateCount * tagged.materials[0].stateCount,
        "A=R_z");
    requireDirectionRejected(
        tagged.materials[0].stressStateDerivativeProgramOffset,
        9u * tagged.materials[0].stateCount,
        "B=P_z");

    // RATE is fixed across all nine Newton-tangent columns and must remain
    // legal in A/B; replacing an input leaf preserves its push/stack width.
    CompiledWorld rateInput = tagged;
    need(replaceRoleLeafOpcode(
             rateInput, rateInput.materials[0].implicitJacobianProgramOffset,
             rateInput.materials[0].stateCount * rateInput.materials[0].stateCount,
             NM_EXPR_RATE),
         "shared-factor rate proof fixture has no input leaf in A=R_z");
    rateInput.fingerprint = compiledWorldFingerprint(rateInput);
    error.clear();
    need(validateCompiledWorldLayout(rateInput, &error),
         "fixed RATE input incorrectly invalidated shared-factor proof: " + error);

    paperMaterial.genericCompensatedImplicitProjection = false;
    paperMaterial.genericBilinear15AccurateProjection = false;
    const CompiledWorld fallback = cook(source(paperMaterial, true, false));
    need(fallback.materials.size() == 1u &&
         (fallback.materials[0].flags & tag) == 0u,
         "ineligible legacy 15-state material did not retain the old cache path");
    error.clear();
    need(validateCompiledWorldLayout(fallback, &error),
         "ineligible legacy 15-state world failed validation: " + error);
    std::cout << "shared_15_state_factor_proof=passed A_DF=rejected B_DF=rejected RATE=allowed ineligible=legacy\n";
}

struct Inputs {
    std::vector<NMFEMNodeStateGPU> nodes;
    std::vector<NMFEMFieldStateGPU> fields;
    std::vector<nm_float4> direction;
    std::vector<float> state;
};
Inputs manufactured(const CompiledWorld& w,
                    const Matrix& f=Matrix{1.13,.12,.04,-.025,.94,.06,.015,-.035,1.025},
                    const Matrix& h=Matrix{.19,-.13,.07,.11,-.17,.04,-.05,.09,.14},
                    const Matrix& rate=Matrix{.3,-.2,.1,.12,.22,-.18,-.1,.08,.25}){
    Inputs v;v.nodes=repeat(w.fem.nodes);v.fields=repeat(w.fem.fields);v.direction.resize(4*w.dispatch.femNodeCount);
    for(unsigned env=0;env<2;++env)for(unsigned n=0;n<w.dispatch.femNodeCount;++n){const unsigned i=env*w.dispatch.femNodeCount+n;
        const Vec x=xyz(w.fem.nodes[n].positionAndMass),p=transform(f,x),r=transform(rate,x),d=transform(h,x);
        v.nodes[i].positionAndMass.x=p[0];v.nodes[i].positionAndMass.y=p[1];v.nodes[i].positionAndMass.z=p[2];
        v.nodes[i].velocityAndInverseMass.x=r[0];v.nodes[i].velocityAndInverseMass.y=r[1];v.nodes[i].velocityAndInverseMass.z=r[2];
        v.direction[i]={float(d[0]),float(d[1]),float(d[2]),0};}
    v.state.resize(2*w.dispatch.tetrahedronCount*w.dispatch.materialStateStride);
    for(unsigned e=0;e<2*w.dispatch.tetrahedronCount;++e)for(unsigned i=0;i<w.stateInitials.size();++i)v.state[e*w.dispatch.materialStateStride+i]=w.stateInitials[i];
    return v;
}
std::vector<NMFEMElementVectorGPU> dispatch(Device& d,const CompiledWorld& w,const Inputs& input,bool tangent,bool cached=false,bool projectionReuseDiagnostic=false,bool cooperativeBundle=false){
    need(!cached||tangent,"cached tangent action requires the FEM operator");
    NMMicrostepGPU micro{};micro.flags=NM_MICROSTEP_FGMRES_OPERATOR;micro.time={w.dispatch.gravityAndTimestep.w,1/w.dispatch.gravityAndTimestep.w,0,0};
    const std::uint32_t forceCompletedCacheRebuild=0u;
    std::vector<float> parameters;for(unsigned e=0;e<2;++e)for(const auto& p:w.parameters)parameters.push_back(p.valueAndBounds.x);
    const auto objects=buffer(d.device,w.objects),materials=buffer(d.device,w.materials),programs=buffer(d.device,w.scalarPrograms),instructions=buffer(d.device,w.instructions);
    const auto params=buffer(d.device,parameters),nodes=buffer(d.device,input.nodes),tets=buffer(d.device,repeat(w.fem.tetrahedra)),state=buffer(d.device,input.state);
    const auto schedulers=buffer(d.device,repeat(w.schedulers)),adaptive=buffer(d.device,repeat(w.adaptive)),mixed=buffer(d.device,w.mixedMaterials),fields=buffer(d.device,input.fields);
    const auto learned=buffer(d.device,w.learnedMaterials),layers=buffer(d.device,w.learnedLayers),weights=buffer(d.device,w.learnedWeights),direction=buffer(d.device,input.direction);
    const auto solver=buffer(d.device,std::vector<NMFGMRESStateGPU>(2)),status=buffer(d.device,std::vector<NMMatterStatusGPU>(2));
    const unsigned count=2*w.dispatch.tetrahedronCount;const auto output=buffer(d.device,std::vector<NMFEMElementVectorGPU>(count));
    const auto tangentCache=buffer(d.device,std::vector<float>(std::size_t(count)*81u));
    const auto tangentCacheValid=buffer(d.device,std::vector<std::uint32_t>(count));
    // Full-domain reference dispatches use a zero selected count; this valid
    // one-word allocation keeps the optional worklist ABI bound without being
    // read by either builder.
    const auto emptyTetrahedronWorklist=buffer(d.device,
        std::vector<std::uint32_t>{});
    const std::uint32_t fullSourceTetrahedronSelection=0u;
    const auto serialCache=buffer(d.device,std::vector<float>(std::size_t(count)*81u));
    const auto serialValid=buffer(d.device,std::vector<std::uint32_t>(count));
    const auto fallbackMetrics=buffer(d.device,
        std::vector<nm_float4>(w.dispatch.environmentCount,nm_float4{}));
    auto command=[d.queue commandBuffer];auto encoder=[command computeCommandEncoder];
    if(cached){
        [encoder setComputePipelineState:projectionReuseDiagnostic
            ? d.cacheProjectionReusePipeline : d.tangentCachePipeline];
        [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];
        [encoder setBytes:&micro length:sizeof(micro) atIndex:1];
        [encoder setBuffer:objects offset:0 atIndex:2];
        [encoder setBuffer:materials offset:0 atIndex:3];
        [encoder setBuffer:programs offset:0 atIndex:4];
        [encoder setBuffer:instructions offset:0 atIndex:5];
        [encoder setBuffer:params offset:0 atIndex:6];
        [encoder setBuffer:nodes offset:0 atIndex:7];
        [encoder setBuffer:tets offset:0 atIndex:8];
        [encoder setBuffer:state offset:0 atIndex:9];
        [encoder setBuffer:schedulers offset:0 atIndex:10];
        [encoder setBuffer:adaptive offset:0 atIndex:11];
        [encoder setBuffer:serialCache offset:0 atIndex:12];
        [encoder setBuffer:serialValid offset:0 atIndex:13];
        [encoder setBuffer:status offset:0 atIndex:14];
        [encoder setBuffer:solver offset:0 atIndex:15];
        [encoder setBytes:&forceCompletedCacheRebuild
            length:sizeof(forceCompletedCacheRebuild) atIndex:16];
        [encoder dispatchThreads:MTLSizeMake(count,1,1)
            threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setBuffer:tangentCache offset:0 atIndex:12];
        [encoder setBuffer:tangentCacheValid offset:0 atIndex:13];
        [encoder setComputePipelineState:d.cachePreparePipeline];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:cooperativeBundle
            ? d.cacheBundleCooperativePipeline : d.cacheColumnsPipeline];
        [encoder setBytes:&fullSourceTetrahedronSelection
                   length:sizeof(fullSourceTetrahedronSelection) atIndex:17u];
        [encoder setBuffer:emptyTetrahedronWorklist offset:0u atIndex:18u];
        [encoder setBytes:&fullSourceTetrahedronSelection
                   length:sizeof(fullSourceTetrahedronSelection) atIndex:19u];
        if(cooperativeBundle){
            const NSUInteger groups=(count+7u)/8u;
            [encoder dispatchThreadgroups:MTLSizeMake(groups,1,1)
                threadsPerThreadgroup:MTLSizeMake(96,1,1)];
        }else{
            [encoder dispatchThreads:MTLSizeMake(count*9u,1,1)
                threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        }
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:d.cacheFinalizePipeline];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [encoder setComputePipelineState:tangent?d.operatorPipeline:d.forcePipeline];
    [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&micro length:sizeof(micro) atIndex:1];
    [encoder setBuffer:objects offset:0 atIndex:2];[encoder setBuffer:materials offset:0 atIndex:3];[encoder setBuffer:programs offset:0 atIndex:4];[encoder setBuffer:instructions offset:0 atIndex:5];
    [encoder setBuffer:params offset:0 atIndex:6];[encoder setBuffer:nodes offset:0 atIndex:7];[encoder setBuffer:tets offset:0 atIndex:8];[encoder setBuffer:state offset:0 atIndex:9];
    if(tangent){[encoder setBuffer:direction offset:0 atIndex:10];[encoder setBuffer:schedulers offset:0 atIndex:11];[encoder setBuffer:adaptive offset:0 atIndex:12];[encoder setBuffer:output offset:0 atIndex:13];[encoder setBuffer:status offset:0 atIndex:14];
        [encoder setBuffer:mixed offset:0 atIndex:16];[encoder setBuffer:fields offset:0 atIndex:17];[encoder setBuffer:learned offset:0 atIndex:18];[encoder setBuffer:layers offset:0 atIndex:19];[encoder setBuffer:weights offset:0 atIndex:20];
        [encoder setBytes:&w.mixedSolver length:sizeof(w.mixedSolver) atIndex:21];
        [encoder setBuffer:solver offset:0 atIndex:22];
        [encoder setBuffer:tangentCache offset:0 atIndex:23];
        [encoder setBuffer:tangentCacheValid offset:0 atIndex:24];
        const std::uint32_t useTangentCache=cached?1u:0u;
        [encoder setBytes:&useTangentCache length:sizeof(useTangentCache) atIndex:25];
        [encoder setBuffer:fallbackMetrics offset:0 atIndex:26];}
    else{[encoder setBuffer:schedulers offset:0 atIndex:10];[encoder setBuffer:adaptive offset:0 atIndex:11];[encoder setBuffer:output offset:0 atIndex:12];[encoder setBuffer:status offset:0 atIndex:13];
        [encoder setBuffer:mixed offset:0 atIndex:14];[encoder setBuffer:fields offset:0 atIndex:15];[encoder setBuffer:learned offset:0 atIndex:16];[encoder setBuffer:layers offset:0 atIndex:17];[encoder setBuffer:weights offset:0 atIndex:18];}
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
    need(command.status==MTLCommandBufferStatusCompleted,"direct production FEM command failed");
    const auto* statuses=static_cast<const NMMatterStatusGPU*>(status.contents);for(unsigned e=0;e<2;++e)need(statuses[e].code==NM_STATUS_SUCCESS,"direct production FEM rejected state code="+std::to_string(statuses[e].code));
    if(cached){
        const auto* valid=static_cast<const std::uint32_t*>(tangentCacheValid.contents);
        for(unsigned i=0;i<count;++i)need(valid[i]==1u,
            "production tangent cache did not build an active tetrahedron entry index="+std::to_string(i));
        need(std::memcmp(serialValid.contents,tangentCacheValid.contents,count*sizeof(std::uint32_t))==0,
             "parallel tangent cache validity differs from serial reference");
        need(std::memcmp(serialCache.contents,tangentCache.contents,std::size_t(count)*81u*sizeof(float))==0,
             "parallel tangent columns differ from serial reference");
        std::cout<<"parallel_tangent_columns=bitwise_equal reference_entries="<<count
                 <<" scalar_entries="<<count*81u<<" environments=2\n";
    }
    const auto* values=static_cast<const NMFEMElementVectorGPU*>(output.contents);return {values,values+count};
}
Matrix gradient(const CompiledWorld& w,const Inputs& v,unsigned e,unsigned ti,bool rate=false,bool direction=false){
    const auto& t=w.fem.tetrahedra[ti];const unsigned base=e*w.dispatch.femNodeCount;
    const auto at=[&](unsigned n){return direction?xyz(v.direction[base+n]):rate?xyz(v.nodes[base+n].velocityAndInverseMass):xyz(v.nodes[base+n].positionAndMass);};
    const Vec x=at(t.nodes.x);return ref::multiply(columns(sub(at(t.nodes.y),x),sub(at(t.nodes.z),x),sub(at(t.nodes.w),x)),inverseRest(t));
}
std::array<Vec,4> forces(const NMTetrahedronGPU& t,const Matrix& p){
    const Matrix a=ref::scale(ref::multiply(p,ref::transpose(inverseRest(t))),-double(t.inverseRestRow0.w));
    std::array<Vec,4> out{};for(unsigned n=1;n<4;++n)for(unsigned i=0;i<3;++i){out[n][i]=a[3*i+n-1];out[0][i]-=out[n][i];}return out;
}
std::array<Vec,4> forces(const NMFEMElementVectorGPU& v){return {xyz(v.node0),xyz(v.node1),xyz(v.node2),xyz(v.node3)};}
double difference(const std::vector<NMFEMElementVectorGPU>& a,const std::vector<NMFEMElementVectorGPU>& b){double error=0,scale=0;need(a.size()==b.size(),"element output arity");for(unsigned e=0;e<a.size();++e){const auto x=forces(a[e]),y=forces(b[e]);for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i){error=std::max(error,std::abs(x[n][i]-y[n][i]));scale=std::max(scale,std::abs(y[n][i]));}}return error/std::max(1e-10,scale);}
struct ActionDifference { double maximumAbsolute=0.0; double maximumDirect=0.0; };
ActionDifference actionDifference(const std::vector<NMFEMElementVectorGPU>& direct,
                                  const std::vector<NMFEMElementVectorGPU>& cached){
    need(direct.size()==cached.size(),"direct/cache element output arity");
    ActionDifference result;
    for(std::size_t element=0;element<direct.size();++element){
        const auto a=forces(direct[element]),b=forces(cached[element]);
        for(unsigned node=0;node<4;++node)for(unsigned axis=0;axis<3;++axis){
            need(std::isfinite(a[node][axis])&&std::isfinite(b[node][axis]),
                 "nonfinite direct/cache element action");
            result.maximumAbsolute=std::max(result.maximumAbsolute,
                std::abs(a[node][axis]-b[node][axis]));
            result.maximumDirect=std::max(result.maximumDirect,
                std::abs(a[node][axis]));
        }
    }
    return result;
}
void randomizeWorldDirection(Inputs& input,const unsigned seed){
    std::uint32_t state=seed==0u?0x6d2b79f5u:seed;
    const auto sample=[&](){
        state^=state<<13u;state^=state>>17u;state^=state<<5u;
        const double unit=double(state>>8u)/16777216.0;
        return static_cast<float>((2.0*unit-1.0)*0.002);
    };
    for(auto& value:input.direction)value={sample(),sample(),sample(),0.0f};
}
void affineBasisWorldDirection(Inputs& input,const CompiledWorld& world,
                               const unsigned row,const unsigned column){
    need(row<3u&&column<3u,"affine material tangent basis index");
    std::fill(input.direction.begin(),input.direction.end(),nm_float4{});
    const unsigned nodesPerEnvironment=world.dispatch.femNodeCount;
    for(unsigned environment=0;environment<world.dispatch.environmentCount;
        ++environment){
        const unsigned base=environment*nodesPerEnvironment;
        for(unsigned node=0;node<nodesPerEnvironment;++node){
            const auto& reference=world.fem.nodes[node].positionAndMass;
            const float component=column==0u?reference.x:
                column==1u?reference.y:reference.z;
            auto& direction=input.direction[base+node];
            if(row==0u)direction.x=component;
            else if(row==1u)direction.y=component;
            else direction.z=component;
        }
    }
}
void checkCachedOperatorActions(Device& device,const CompiledWorld& world,
                                const Inputs& base,const char* stateName,
                                const bool cooperativeBundle=false){
    double maximumRelative=0.0,maximumAbsolute=0.0,maximumDirect=0.0;
    constexpr unsigned basisSamples=9u;
    constexpr unsigned randomSamples=6u;
    for(unsigned sample=0;sample<basisSamples+randomSamples;++sample){
        Inputs input=base;
        if(sample<basisSamples){
            affineBasisWorldDirection(input,world,sample/3u,sample%3u);
        }else{
            randomizeWorldDirection(input,
                0x9e3779b9u+0x85ebca6bu*(sample-basisSamples));
        }
        const auto direct=dispatch(device,world,input,true,false);
        const auto cached=dispatch(device,world,input,true,true,false,
                                   cooperativeBundle);
        const ActionDifference error=actionDifference(direct,cached);
        maximumAbsolute=std::max(maximumAbsolute,error.maximumAbsolute);
        maximumDirect=std::max(maximumDirect,error.maximumDirect);
        maximumRelative=std::max(maximumRelative,difference(direct,cached));
    }
    const double absoluteBound=1.0e-6+1.0e-3*maximumDirect;
    std::cout<<std::setprecision(12)
        <<"{\"schema\":1,\"record\":\"native_algorithmic_tangent_cache_action_parity\""
        <<",\"physical_validation\":false,\"material_state\":\""<<stateName<<"\""
        <<",\"material_frames\":\"rotated per tetrahedron\""
        <<",\"affine_deformation_basis_actions\":"<<basisSamples
        <<",\"fixed_seed_random_world_direction_samples\":"<<randomSamples
        <<",\"maximum_relative_element_force_error\":"<<maximumRelative
        <<",\"maximum_absolute_element_force_error_N\":"<<maximumAbsolute
        <<",\"maximum_direct_element_force_N\":"<<maximumDirect
        <<",\"absolute_bound_N\":"<<absoluteBound
        <<",\"pass\":"<<(maximumAbsolute<=absoluteBound?"true":"false")<<"}\n";
    need(maximumAbsolute<=absoluteBound,
         std::string("native cached tangent action differs from direct production action for ")+stateName);
}
Inputs acceptedInputs(const RuntimeStateSnapshot& snapshot){
    Inputs input;
    input.nodes=snapshot.femNodes;
    input.fields=snapshot.femFields;
    input.state=snapshot.femMaterialState;
    const std::size_t environments=snapshot.statuses.size();
    need(environments==2u&&input.nodes.size()%environments==0u,
         "accepted snapshot environment/node count is inconsistent");
    input.direction.resize(input.nodes.size()*2u);
    need(input.nodes.size()>0u,
         "accepted snapshot FEM node count is inconsistent");
    need(input.fields.size()==input.nodes.size(),
         "accepted snapshot FEM field count is inconsistent");
    return input;
}
std::uint32_t materialStateIndex(const MaterialProgram& material,
                                const char* name){
    for(std::uint32_t index=0;index<material.internalState.size();++index)
        if(material.internalState[index].name==name)return index;
    throw std::runtime_error(std::string("missing source state ")+name);
}
double maximumMaterialHistory(const RuntimeStateSnapshot& snapshot,
                              const std::uint32_t stateIndex){
    need(snapshot.materialStateStride>stateIndex,
         "accepted material snapshot lacks requested history");
    const std::size_t environments=snapshot.statuses.size();
    need(environments==2u&&snapshot.femTopologyTetrahedra.size()%environments==0u,
         "accepted snapshot environment/tetrahedron count is inconsistent");
    const std::size_t expected=snapshot.femTopologyTetrahedra.size()*
        snapshot.materialStateStride;
    need(snapshot.femMaterialState.size()==expected,
         "accepted material snapshot state arity is inconsistent");
    double maximum=-std::numeric_limits<double>::infinity();
    for(std::size_t cell=0;cell<snapshot.femTopologyTetrahedra.size();++cell){
        const float value=snapshot.femMaterialState[
            cell*snapshot.materialStateStride+stateIndex];
        need(std::isfinite(value),"accepted plastic history is nonfinite");
        maximum=std::max(maximum,double(value));
    }
    return maximum;
}
std::vector<nm_float4> paperShearTargets(
    const RuntimeStateSnapshot& reference,const float displacement){
    std::vector<nm_float4> targets(reference.femNodes.size());
    const std::size_t environments=reference.statuses.size();
    need(environments==2u&&reference.femNodes.size()%environments==0u,
         "bilinear-paper fixture environment count changed");
    const std::size_t nodesPerEnvironment=reference.femNodes.size()/environments;
    need(nodesPerEnvironment>=5u,
         "bilinear-paper fixture topology changed");
    for(std::uint32_t environment=0;environment<environments;
        ++environment){
        const std::size_t base=std::size_t(environment)*nodesPerEnvironment;
        for(const auto& [node,sign]:
                {std::pair<std::uint32_t,float>{3u,1.0f},
                 std::pair<std::uint32_t,float>{4u,-1.0f}}){
            const auto& position=reference.femNodes[base+node].positionAndMass;
            targets[base+node]={position.x+sign*displacement,position.y,
                                position.z,3.0f};
        }
    }
    return targets;
}
using Oracle=std::function<ref::Result(const Matrix&,const Matrix&,const Matrix&,const Matrix&)>;
double compare(const CompiledWorld& w,const Inputs& v,const std::vector<NMFEMElementVectorGPU>& actual,bool tangent,const Oracle& oracle){
    double error=0,scale=0;
    for(unsigned e=0;e<2;++e)for(unsigned ti=0;ti<w.dispatch.tetrahedronCount;++ti){const auto& t=w.fem.tetrahedra[ti];
        const Matrix q=(w.objects[t.identity.y].flags&NM_OBJECT_FEM_MATERIAL_FRAME)?frame(t.materialFrameRotation):ref::identity;
        const auto result=oracle(gradient(w,v,e,ti),gradient(w,v,e,ti,false,true),gradient(w,v,e,ti,true),q);
        const auto expected=forces(t,tangent?result.tangent:result.stress),observed=forces(actual[e*w.dispatch.tetrahedronCount+ti]);
        for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i){need(std::isfinite(observed[n][i]),"nonfinite production force/tangent");error=std::max(error,std::abs(observed[n][i]-expected[n][i]));scale=std::max(scale,std::abs(expected[n][i]));}}
    return error/std::max(1e-10,scale);
}
double finiteDifference(Device& d,const CompiledWorld& w,const Inputs& v,const std::vector<NMFEMElementVectorGPU>& tangent){
    constexpr double epsilon=1./2048;Inputs plus=v,minus=v;const double dt=w.dispatch.gravityAndTimestep.w;const unsigned count=2*w.dispatch.femNodeCount;
    for(unsigned i=0;i<count;++i){const auto p=v.direction[i];
        plus.nodes[i].positionAndMass.x+=epsilon*p.x;minus.nodes[i].positionAndMass.x-=epsilon*p.x;
        plus.nodes[i].positionAndMass.y+=epsilon*p.y;minus.nodes[i].positionAndMass.y-=epsilon*p.y;
        plus.nodes[i].positionAndMass.z+=epsilon*p.z;minus.nodes[i].positionAndMass.z-=epsilon*p.z;
        plus.nodes[i].velocityAndInverseMass.x+=epsilon*p.x/dt;minus.nodes[i].velocityAndInverseMass.x-=epsilon*p.x/dt;
        plus.nodes[i].velocityAndInverseMass.y+=epsilon*p.y/dt;minus.nodes[i].velocityAndInverseMass.y-=epsilon*p.y/dt;
        plus.nodes[i].velocityAndInverseMass.z+=epsilon*p.z/dt;minus.nodes[i].velocityAndInverseMass.z-=epsilon*p.z/dt;
        plus.fields[i].secondary.x+=epsilon*v.direction[count+i].w/dt;minus.fields[i].secondary.x-=epsilon*v.direction[count+i].w/dt;}
    const auto fp=dispatch(d,w,plus,false),fm=dispatch(d,w,minus,false);double error=0,scale=0;
    for(unsigned e=0;e<tangent.size();++e){const auto a=forces(fp[e]),b=forces(fm[e]),t=forces(tangent[e]);for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i){const double fd=(a[n][i]-b[n][i])/(2*epsilon);error=std::max(error,std::abs(fd-t[n][i]));scale=std::max(scale,std::abs(t[n][i]));}}
    return error/std::max(1e-10,scale);
}
void learnedFrameChecks(Device& d){
    // Synthetic single-neuron ICNN: W=softplus_beta(w*(I4-1)+b)
    // +g*(J+1/J-2). This closed form permits an independent tensor
    // derivative, rather than comparing the production network to itself.
    auto parsed=parseMatter(R"(material synthetic_framed_icnn_control {
        parameter density : kg/m^3 = 1000; parameter scale : Pa = 1;
        model generic; energy = 0*scale; valid = J(); supports fem;
    })");need(parsed.succeeded(),diagnostic(parsed.diagnostics));
    auto material=parsed.material;material.hint=ConstitutiveHint::polyconvexICNN;
    material.mixed.fibreDirection={.6,.8,0};
    LearnedMaterialSource network;network.invariantCount=5;network.softplusBeta=2.5f;
    network.determinantFloor=.05f;network.growthCoefficient=.03125f;
    LearnedLayerSource layer;layer.inputWidth=5;layer.outputWidth=1;
    layer.inputWeights={0,0,0,0,.75f};layer.biases={.125f};network.layers={layer};material.learned=network;
    const auto authored=source(material);const auto world=cook(authored);const auto input=manufactured(world);
    const auto rawFibre=world.mixedMaterials[0].fibre;
    Vec a=xyz(rawFibre);const double length=std::sqrt(dot(a,a));for(double& value:a)value/=length;
    const Oracle oracle=[a](const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){
        const Vec fibre=transform(q,a),u=transform(f,fibre),du=transform(h,fibre);
        const double j=ref::determinant(f),ell=ref::contract(ref::inverseTranspose(f),h),dj=j*ell;
        const double z=.75*(dot(u,u)-1)+.125,slope=1/(1+std::exp(-2.5*z));
        const double dz=1.5*dot(u,du),dslope=2.5*slope*(1-slope)*dz;
        const Matrix it=ref::inverseTranspose(f),cofactor=ref::scale(it,j);
        const Matrix dit=ref::scale(ref::multiply(ref::multiply(it,ref::transpose(h)),it),-1);
        const Matrix dcofactor=ref::add(ref::scale(it,dj),ref::scale(dit,j));
        ref::Result result;
        result.stress=ref::add(ref::scale(outer(u,fibre),1.5*slope),ref::scale(cofactor,.03125*(1-1/(j*j))));
        result.tangent=ref::add(ref::scale(ref::add(ref::scale(outer(u,fibre),dslope),ref::scale(outer(du,fibre),slope)),1.5),
            ref::scale(ref::add(ref::scale(cofactor,2*dj/(j*j*j)),ref::scale(dcofactor,1-1/(j*j))),.03125));return result;
    };
    const auto force=dispatch(d,world,input,false),tangent=dispatch(d,world,input,true);
    const double fe=compare(world,input,force,false,oracle),te=compare(world,input,tangent,true,oracle),fd=finiteDifference(d,world,input,tangent);
    need(fe<3e-4&&te<3e-4&&fd<3e-3,"framed ICNN independent oracle mismatch force="+number(fe)+" tangent="+number(te)+" fd="+number(fd));
    // Independent frame equivalence: the same physical fibre is authored
    // directly in world-reference coordinates, with all frame logic disabled.
    // Each element has its own frame, so make one unframed control per frame.
    auto equivalentForce=force,equivalentTangent=tangent;
    for(unsigned ti=0;ti<world.dispatch.tetrahedronCount;++ti){
        auto unframed=source(material,false);unframed.materials[0].mixed.fibreDirection=transform(frame(world.fem.tetrahedra[ti].materialFrameRotation),a);
        const auto control=cook(unframed);const auto cf=dispatch(d,control,input,false),ct=dispatch(d,control,input,true);
        for(unsigned env=0;env<2;++env){const auto index=env*world.dispatch.tetrahedronCount+ti;equivalentForce[index]=cf[index];equivalentTangent[index]=ct[index];}}
    const double frameForce=difference(force,equivalentForce),frameTangent=difference(tangent,equivalentTangent);
    need(frameForce<3e-4&&frameTangent<3e-4,"framed ICNN differs from independently authored physical fibre");
    const auto unrotated=cook(source(material,false));const double angular=difference(force,dispatch(d,unrotated,input,false));
    need(angular>1e-3,"ICNN fixture did not resolve material-frame angular response");
    std::cout<<"framed_icnn_force_error="<<fe<<" framed_icnn_tangent_error="<<te<<" framed_icnn_fd_error="<<fd
        <<" framed_icnn_world_fibre_force_error="<<frameForce<<" framed_icnn_world_fibre_tangent_error="<<frameTangent
        <<" framed_icnn_angular_consequence="<<angular<<" trained_material_qualified=false\n";
}
double materialParameter(const MaterialProgram& material,const std::string& name){
    for(const auto& parameter:material.parameters)if(parameter.name==name)return parameter.defaultValue;
    throw std::runtime_error("missing canonical parameter "+name);
}
void canonicalTangentChecks(Device& d,const std::filesystem::path& materialsDirectory){
    constexpr std::uint32_t tag=NM_MATERIAL_CANONICAL_COMPRESSIBLE_NEOHOOKEAN_TANGENT;
    const std::array<const char*,3> names{
        "human_skin_dermis_frequency_specific_effective.nmatter",
        "human_subcutaneous_adipose_low_strain_effective.nmatter",
        "human_optic_nerve_longitudinal_low_strain_isotropic_effective.nmatter"};
    std::uint32_t checked=0u;
    for(const char* name:names){
        const auto parsed=parseMatterFile((materialsDirectory/name).string());
        need(parsed.succeeded(),"canonical tangent material parse failed: "+std::string(name)+" "+diagnostic(parsed.diagnostics));
        const auto world=cook(source(parsed.material,true));
        need(world.materials.size()==1u&&(world.materials[0].flags&tag)!=0u,
             "exact canonical material was not classified: "+std::string(name));
        auto fallback=world;
        fallback.materials[0].flags&=~tag;
        const double shear=materialParameter(parsed.material,"shear");
        const double bulk=materialParameter(parsed.material,"bulk");
        const Oracle oracle=[shear,bulk](const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){
            const Matrix localF=ref::multiply(f,q),localH=ref::multiply(h,q);
            const double j=ref::determinant(localF);
            need(j>0.0&&std::isfinite(j),"canonical tangent fixture has invalid J");
            const double logJ=std::log(j);
            const Matrix inverseT=ref::inverseTranspose(localF);
            const Matrix dInverseT=ref::scale(ref::multiply(ref::multiply(inverseT,ref::transpose(localH)),inverseT),-1.0);
            const double volumeDirection=ref::contract(inverseT,localH);
            ref::Result result;
            result.stress=ref::multiply(ref::add(ref::scale(ref::add(localF,ref::scale(inverseT,-1.0)),shear),
                ref::scale(inverseT,bulk*logJ)),ref::transpose(q));
            const Matrix localTangent=ref::add(ref::add(ref::scale(localH,shear),
                ref::scale(inverseT,bulk*volumeDirection)),
                ref::scale(dInverseT,bulk*logJ-shear));
            result.tangent=ref::multiply(localTangent,ref::transpose(q));
            return result;
        };
        const std::array<Matrix,3> deformations{
            Matrix{1.13,.12,.04,-.025,.94,.06,.015,-.035,1.025},
            Matrix{.79,.015,0,.035,.91,.01,0,.02,.84},
            Matrix{1.0,.37,.04,-.025,1.0,.11,.015,-.035,1.0}};
        const std::array<Matrix,3> directions{
            Matrix{.19,-.13,.07,.11,-.17,.04,-.05,.09,.14},
            Matrix{-.11,.08,.03,.12,.07,-.06,.04,-.09,.16},
            Matrix{.03,.21,-.04,-.12,.06,.11,.08,-.05,-.02}};
        double maximumOracle=0.0,maximumFallback=0.0,maximumCachedOracle=0.0,maximumFD=0.0;
        for(std::size_t sample=0;sample<deformations.size();++sample){
            const Inputs input=manufactured(world,deformations[sample],directions[sample]);
            const auto force=dispatch(d,world,input,false);
            const auto fallbackForce=dispatch(d,fallback,input,false);
            need(same(force,fallbackForce),"canonical tangent flag changed force output");
            const auto tangent=dispatch(d,world,input,true);
            const auto fallbackTangent=dispatch(d,fallback,input,true);
            const auto cached=dispatch(d,world,input,true,true);
            const auto fallbackCached=dispatch(d,fallback,input,true,true);
            const double oracleError=compare(world,input,tangent,true,oracle);
            const double cachedOracleError=compare(world,input,cached,true,oracle);
            maximumOracle=std::max(maximumOracle,oracleError);
            maximumCachedOracle=std::max(maximumCachedOracle,cachedOracleError);
            maximumFallback=std::max({maximumFallback,difference(tangent,fallbackTangent),difference(cached,fallbackCached)});
            maximumFD=std::max({maximumFD,finiteDifference(d,world,input,tangent),finiteDifference(d,world,input,cached)});
        }
        // All nine cache columns must use the same tangent-selection policy
        // as an arbitrary direct action, including when the canonical tag is
        // removed. Random nodal directions exercise products beyond the
        // coordinate columns and both rotated tetrahedral material frames.
        const Inputs parityInput=manufactured(world,deformations[0],directions[0]);
        checkCachedOperatorActions(d,world,parityInput,
            (std::string(name)+":canonical-selected").c_str());
        checkCachedOperatorActions(d,fallback,parityInput,
            (std::string(name)+":bytecode-selected").c_str());
        need(maximumOracle<5e-4&&maximumCachedOracle<5e-4,
             "canonical tangent closed form disagrees with independent oracle material="+std::string(name)+" direct="+number(maximumOracle)+" cached="+number(maximumCachedOracle));
        need(maximumFallback<8e-4,
             "canonical tangent differs from bytecode fallback material="+std::string(name)+" error="+number(maximumFallback));
        need(maximumFD<4e-3,
             "canonical tangent differs from force finite difference material="+std::string(name)+" error="+number(maximumFD));
        std::cout<<"canonical_tangent_material="<<name<<" direct_oracle="<<maximumOracle
            <<" cached_oracle="<<maximumCachedOracle<<" fallback="<<maximumFallback<<" fd="<<maximumFD<<"\n";
        ++checked;
    }
    need(checked==names.size(),"canonical tangent target count changed");

    const auto exact=parseMatter(R"(material synthetic_canonical_neohookean {
        parameter density : kg/m^3 = 1000; parameter shear : Pa = 18000; parameter bulk : Pa = 700000;
        model generic; energy = 0.5*shear*(I1()-3-2*log(J()))+0.5*bulk*pow(log(J()),2);
        valid = J(); supports fem;
    })");
    need(exact.succeeded(),"synthetic canonical source parse: "+diagnostic(exact.diagnostics));
    const auto requireUntagged=[&](MaterialProgram material,const char* label){
        const auto candidate=cook(source(material,false));
        need((candidate.materials[0].flags&tag)==0u,std::string("false-positive canonical tangent tag: ")+label);
    };
    auto stateMaterial=exact.material;InternalState state;state.name="history";state.initialValue=.1;stateMaterial.internalState.push_back(state);
    requireUntagged(stateMaterial,"stateful material");
    auto fibreMaterial=exact.material;fibreMaterial.mixed.fibreDirection={0.0,1.0,0.0};
    requireUntagged(fibreMaterial,"fibre-coupled material");
    auto unrelated=parseMatter(R"(material synthetic_noncanonical_neohookean {
        parameter density : kg/m^3 = 1000; parameter shear : Pa = 18000; parameter bulk : Pa = 700000;
        model generic; energy = 0.5*shear*(I1()-3)+0.5*bulk*pow(log(J()),2);
        valid = J(); supports fem;
    })");
    need(unrelated.succeeded(),"synthetic noncanonical source parse: "+diagnostic(unrelated.diagnostics));
    requireUntagged(unrelated.material,"unrelated energy");
    auto dissipative=parseMatter(R"(material synthetic_dissipative_neohookean {
        parameter density : kg/m^3 = 1000; parameter shear : Pa = 18000; parameter bulk : Pa = 700000; parameter eta : Pa*s = 1;
        model generic; energy = 0.5*shear*(I1()-3-2*log(J()))+0.5*bulk*pow(log(J()),2);
        dissipation = 0.5*eta*pow(Fdot(0,0),2); valid = J(); supports fem;
    })");
    need(dissipative.succeeded(),"synthetic dissipative source parse: "+diagnostic(dissipative.diagnostics));
    requireUntagged(dissipative.material,"dissipative material");
    std::cout<<"canonical_tangent_false_positives=none state=fallback fibre=fallback unrelated_energy=fallback dissipation=fallback\n";
}

void requireStatelessGenericZeroState(const CompiledWorld& world,
                                     const char* fixtureName){
    need(!world.materials.empty(),
         std::string(fixtureName)+": fixture has no materials");
    for(std::size_t index=0;index<world.materials.size();++index){
        const auto& material=world.materials[index];
        need(material.stateCount==0u &&
             material.projectionKind==NM_MATERIAL_PROJECTION_GENERIC &&
             (material.flags&(NM_MATERIAL_HAS_STATE|
                              NM_MATERIAL_HAS_IMPLICIT_STATE))==0u,
             std::string(fixtureName)+": material "+
                 std::to_string(index)+
                 " is not generic and stateless before GPU dispatch");
    }
}
void nativeViscousCompilerProofChecks(const CompiledWorld& world) {
    need(world.materials.size() == 1u &&
         nmNativeStatelessViscousMaterialEligible(
             world, world.materials.front()),
         "compiled stateless viscous material failed exact native float-program proof");

    CompiledWorld compensated = world;
    compensated.materials[0].flags |= NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC;
    need(!nmNativeStatelessViscousMaterialEligible(
             compensated, compensated.materials[0]),
         "native viscous proof admitted a compensated material");

    CompiledWorld directionInStress = world;
    const auto& stressMaterial = directionInStress.materials[0];
    bool changed = false;
    for (std::uint32_t component = 0u; component < 9u && !changed; ++component) {
        const auto& program = directionInStress.scalarPrograms[
            stressMaterial.stressProgramOffset + component];
        for (std::uint32_t local = 0u; local < program.instructionCount; ++local) {
            auto& instruction = directionInStress.instructions[
                program.firstInstruction + local];
            if (instruction.opcode == NM_EXPR_F) {
                instruction.opcode = NM_EXPR_DF;
                changed = true;
                break;
            }
        }
    }
    need(changed &&
         !nmNativeStatelessViscousMaterialEligible(
             directionInStress, directionInStress.materials[0]),
         "native viscous proof admitted a direction-dependent stress program");

    CompiledWorld unsupportedTangent = world;
    const auto& tangentMaterial = unsupportedTangent.materials[0];
    const auto& program = unsupportedTangent.scalarPrograms[
        tangentMaterial.viscousTangentProgramOffset];
    unsupportedTangent.instructions[program.firstInstruction].opcode =
        NM_EXPR_STATE;
    need(!nmNativeStatelessViscousMaterialEligible(
             unsupportedTangent, unsupportedTangent.materials[0]),
         "native viscous proof admitted stateful or unsupported bytecode");
    std::cout << "stateless_viscous_native_program_proof=passed "
        "base_stress=9 base_tangent=9 viscous_stress=9 viscous_tangent=9 "
        "compensated=excluded direction_dependent_stress=excluded "
        "state_opcode=excluded\n";
}

void statelessViscousSpecializationChecks(Device& specialized,
                                           Device& legacy){
    const auto parsed=parseMatter(R"(material synthetic_stateless_viscous_frame {
        parameter density : kg/m^3 = 1000;
        parameter stiffness : Pa = 1200;
        parameter viscosity : Pa*s = 20;
        model generic;
        energy = 0.5*stiffness*pow(F(0,0)-1,2);
        dissipation = 0.5*viscosity*pow(Fdot(0,2),2);
        valid = J(); supports fem;
    })");
    need(parsed.succeeded(),"stateless viscous source parse: "+
         diagnostic(parsed.diagnostics));
    const CompiledWorld world=cook(source(parsed.material,true,false));
    need(world.materials.size()==1u&&world.materials[0].stateCount==0u&&
         world.materials[0].projectionKind==NM_MATERIAL_PROJECTION_GENERIC&&
         (world.materials[0].flags&NM_MATERIAL_HAS_DISSIPATION)!=0u&&
         world.materials[0].viscousStressProgramOffset!=NM_INVALID_INDEX&&
         world.materials[0].viscousTangentProgramOffset!=NM_INVALID_INDEX,
         "stateless viscous fixture did not compile as generic zero-state dissipation");
    nativeViscousCompilerProofChecks(world);
    const Inputs input=manufactured(world);
    const auto specializedForce=dispatch(specialized,world,input,false);
    const auto legacyForce=dispatch(legacy,world,input,false);
    const auto specializedTangent=dispatch(specialized,world,input,true);
    const auto legacyTangent=dispatch(legacy,world,input,true);
    need(same(specializedForce,legacyForce)&&
         same(specializedTangent,legacyTangent),
         "stateless specialization changed viscous force or tangent words");
    const double dt=world.dispatch.gravityAndTimestep.w;
    const Oracle oracle=[dt](const Matrix& f,const Matrix& h,
                            const Matrix& rate,const Matrix& q){
        const Matrix localF=ref::multiply(f,q);
        const Matrix localH=ref::multiply(h,q);
        const Matrix localRate=ref::multiply(rate,q);
        Matrix stress{},tangent{};
        stress[0]=1200.0*(localF[0]-1.0);
        stress[2]=20.0*localRate[2];
        tangent[0]=1200.0*localH[0];
        tangent[2]=20.0*localH[2]/dt;
        ref::Result result;
        result.stress=ref::multiply(stress,ref::transpose(q));
        result.tangent=ref::multiply(tangent,ref::transpose(q));
        return result;
    };
    const double forceError=compare(world,input,specializedForce,false,oracle);
    const double tangentError=compare(world,input,specializedTangent,true,oracle);
    const double fdError=finiteDifference(specialized,world,input,
                                          specializedTangent);
    need(forceError<3e-4&&tangentError<3e-3&&fdError<4e-3,
         "stateless viscous specialized oracle/FD mismatch force="+
         number(forceError)+" tangent="+number(tangentError)+
         " fd="+number(fdError));
    std::cout<<"stateless_viscous_specialization=passed bitwise_legacy=yes "
        <<"force_error="<<forceError<<" tangent_error="<<tangentError
        <<" fd_error="<<fdError<<"\n";
}
void statelessBundleChecks(Device& d,const MaterialProgram& guccione){
    // Exercise the compiler-emitted stateless tangent bundle against the
    // unchanged nine legacy tangent programs. The serial-column and 8-tet /
    // 96-thread cooperative paths share the same transaction fixture,
    // rotated material frames, nine affine basis actions, and fixed-seed
    // random directions.
    auto bundleMaterial=guccione;
    bundleMaterial.cacheExpressions=true;
    const CompiledWorld bundleWorld=cook(source(bundleMaterial,true,false));
    requireStatelessGenericZeroState(bundleWorld,"base stateless bundle");
    need(bundleWorld.materials.size()==1u &&
         (bundleWorld.materials[0].flags & NM_MATERIAL_HAS_TANGENT_BUNDLE)!=0u,
         "generic stateless fixture did not emit an eligible tangent bundle");
    const Inputs bundleInputs=manufactured(bundleWorld);
    checkCachedOperatorActions(d,bundleWorld,bundleInputs,
        "rotated-anisotropic-stateless-bundle",false);
    checkCachedOperatorActions(d,bundleWorld,bundleInputs,
        "rotated-anisotropic-stateless-bundle",true);

    // Keep a preceding material's parameter block in front of the bundle
    // target. This exercises the production stress hook's parameter-offset
    // contract instead of the degenerate offset-zero fixture above.
    auto offsetBundleSource=source(bundleMaterial,true,false);
    auto prefixMaterial=guccione;
    prefixMaterial.name="synthetic_tangent_bundle_parameter_prefix";
    auto offsetTarget=bundleMaterial;
    offsetTarget.name="synthetic_tangent_bundle_parameter_target";
    offsetBundleSource.materials={prefixMaterial,offsetTarget};
    offsetBundleSource.objects[0].materialIndex=1u;
    const CompiledWorld offsetBundleWorld=cook(offsetBundleSource);
    requireStatelessGenericZeroState(offsetBundleWorld,
        "nonzero parameter offset stateless bundle");
    need(offsetBundleWorld.materials.size()==2u &&
         offsetBundleWorld.materials[1].parameterOffset>0u &&
         (offsetBundleWorld.materials[1].flags &
             NM_MATERIAL_HAS_TANGENT_BUNDLE)!=0u,
         "nonzero-offset target did not retain its eligible tangent bundle");
    const Inputs offsetBundleInputs=manufactured(offsetBundleWorld);
    checkCachedOperatorActions(d,offsetBundleWorld,offsetBundleInputs,
        "rotated-anisotropic-stateless-bundle-nonzero-parameter-offset",false);
    checkCachedOperatorActions(d,offsetBundleWorld,offsetBundleInputs,
        "rotated-anisotropic-stateless-bundle-nonzero-parameter-offset",true);
}
void directChecks(Device& d,const MaterialProgram& guccione){
    const auto s=source(guccione);const auto w=cook(s);const auto v=manufactured(w);
    const auto force=dispatch(d,w,v,false),tangent=dispatch(d,w,v,true);
    const Oracle passive=[](const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){return ref::framedGuccione(f,h,q);};
    const double forceError=compare(w,v,force,false,passive),tangentError=compare(w,v,tangent,true,passive),fd=finiteDifference(d,w,v,tangent);
    need(forceError<3e-4&&tangentError<3e-4,"source Guccione production frame oracle mismatch force="+number(forceError)+" tangent="+number(tangentError));
    need(fd<3e-3,"production frame tangent differs from force FD: "+number(fd));
    checkCachedOperatorActions(d,w,v,"rotated-anisotropic-elastic");

    checkNewtonTailDispatch(d);
    statelessBundleChecks(d,guccione);
    auto identitySource=s;for(auto& q:identitySource.objects[0].femMaterialFrameRotations)q={0,0,0,1};
    const auto wi=cook(identitySource),wl=cook(source(guccione,false));
    const auto identityForce=dispatch(d,wi,v,false),legacyForce=dispatch(d,wl,v,false);
    const double identityError=difference(identityForce,legacyForce),angular=difference(force,identityForce);
    need(identityError==0,"identity frame changed legacy force arithmetic");need(angular>1e-3,"anisotropic frame rotation had no resolvable force consequence");
    need(difference(dispatch(d,wi,v,true),dispatch(d,wl,v,true))==0,"identity frame changed legacy tangent arithmetic");
    paired(force,"source element forces");paired(tangent,"source element tangent");

    const auto parsed=parseMatter(R"(material synthetic_frame_state_rate_path {
        parameter density : kg/m^3 = 1000; parameter k : Pa = 1000; parameter eta : Pa*s = 20;
        state history : one = 0.23 transfer average;
        model generic; energy = 0.5*k*pow(F(0,0)-next(history),2);
        dissipation = 0.5*eta*pow(Fdot(0,2),2);
        update history = history + dt()*Fdot(0,1); valid = J(); supports fem;
    })");need(parsed.succeeded(),diagnostic(parsed.diagnostics));
    const auto wr=cook(source(parsed.material));const auto vr=manufactured(wr);const double dt=wr.dispatch.gravityAndTimestep.w;
    const Oracle rate=[dt](const Matrix& f,const Matrix& h,const Matrix& r,const Matrix& q){
        const Matrix fl=ref::multiply(f,q),hl=ref::multiply(h,q),rl=ref::multiply(r,q);ref::Result out;
        const double old=double(float(.23)),next=old+dt*rl[1];Matrix p{},dp{};p[0]=1000*(fl[0]-next);p[2]=20*rl[2];
        dp[0]=1000*(hl[0]-hl[1]);dp[2]=20*hl[2]/dt;out.stress=ref::multiply(p,ref::transpose(q));out.tangent=ref::multiply(dp,ref::transpose(q));return out;};
    const auto rf=dispatch(d,wr,vr,false),rt=dispatch(d,wr,vr,true);
    const double rfe=compare(wr,vr,rf,false,rate),rte=compare(wr,vr,rt,true,rate),rfd=finiteDifference(d,wr,vr,rt);
    need(rfe<3e-4&&rte<3e-3&&rfd<4e-3,"synthetic state/rate frame path mismatch force="+number(rfe)+" tangent="+number(rte)+" fd="+number(rfd));
    checkCachedOperatorActions(d,wr,vr,"rotated-state-rate");

    auto activeMaterial=guccione;activeMaterial.mixed.maximumActiveTension=12000;
    const auto wa=cook(source(activeMaterial,true,true));auto va=manufactured(wa);const unsigned count=2*wa.dispatch.femNodeCount;
    for(unsigned i=0;i<count;++i){va.fields[i].secondary.x=.3f;va.direction[count+i].w=.001f;}
    auto off=va;for(auto& f:off.fields)f.secondary.x=0;for(unsigned i=0;i<count;++i)off.direction[count+i].w=0;
    const auto af=dispatch(d,wa,va,false),at=dispatch(d,wa,va,true),of=dispatch(d,wa,off,false),ot=dispatch(d,wa,off,true);
    auto activeOnly=af,activeTangent=at;
    for(unsigned i=0;i<af.size();++i){auto* a=reinterpret_cast<float*>(&activeOnly[i]);auto* t=reinterpret_cast<float*>(&activeTangent[i]);const auto* b=reinterpret_cast<const float*>(&of[i]);const auto* u=reinterpret_cast<const float*>(&ot[i]);for(unsigned j=0;j<16;++j){a[j]-=b[j];t[j]-=u[j];}}
    const double activeDt=wa.dispatch.gravityAndTimestep.w;
    const Oracle active=[activeDt](const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){
        const Vec fibre{q[0],q[3],q[6]},u=transform(f,fibre),du=transform(h,fibre);const double l2=dot(u,u),j=ref::determinant(f),ell=ref::contract(ref::inverseTranspose(f),h);
        const double a=double(float(.3)),da=double(float(.001))/activeDt,prefactor=12000*j/l2;ref::Result out;
        out.stress=ref::scale(outer(u,fibre),prefactor*a);
        out.tangent=ref::scale(ref::add(ref::scale(outer(u,fibre),a*(ell-2*dot(u,du)/l2)+da),ref::scale(outer(du,fibre),a)),prefactor);return out;};
    const double afe=compare(wa,va,activeOnly,false,active),ate=compare(wa,va,activeTangent,true,active),afd=finiteDifference(d,wa,va,at);
    need(afe<3e-4&&ate<3e-4&&afd<4e-3,"mixed active frame control mismatch force="+number(afe)+" tangent="+number(ate)+" fd="+number(afd));
    std::cout<<"source_frame_force_error="<<forceError<<" source_frame_tangent_error="<<tangentError<<" source_frame_fd_error="<<fd<<" identity_error="<<identityError<<" anisotropic_angular_consequence="<<angular
        <<" synthetic_state_rate_force_error="<<rfe<<" synthetic_state_rate_tangent_error="<<rte<<" synthetic_state_rate_fd_error="<<rfd
        <<" mixed_active_force_error="<<afe<<" mixed_active_tangent_error="<<ate<<" mixed_active_fd_error="<<afd<<" mixed_source_equivalent=false\n";
}

struct ScopedSharedFactorEnvironment {
    bool hadOriginal = false;
    std::string original;
    ScopedSharedFactorEnvironment(const bool enabled) {
        if (const char* value = std::getenv(
                "NM_FEM_SHARED_LOCAL_FACTORIZATION")) {
            hadOriginal = true;
            original = value;
        }
        need(setenv("NM_FEM_SHARED_LOCAL_FACTORIZATION",
                    enabled ? "1" : "0", 1) == 0,
             "cannot set shared-factor test policy");
    }
    ~ScopedSharedFactorEnvironment() {
        if (hadOriginal)
            setenv("NM_FEM_SHARED_LOCAL_FACTORIZATION", original.c_str(), 1);
        else
            unsetenv("NM_FEM_SHARED_LOCAL_FACTORIZATION");
    }
};
struct Run {
    Device& device;CompiledWorld world;Runtime runtime;id<MTLBuffer> statuses;
    Run(Device& d,const WorldSource& s,bool sharedFactorEnabled=false):device(d),world(cook(s)){
        const auto path=std::filesystem::temp_directory_path()/(std::string("numi-frame-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
        CompileResult package;package.world=world;std::string error;need(writePackage(package,path,&error),error);CompiledWorld loaded;
        const bool ok=readPackage(path,loaded,nullptr,&error);std::error_code remove;std::filesystem::remove(path,remove);need(ok&&!remove&&loaded.fingerprint==world.fingerprint,"frame package roundtrip: "+error);world=std::move(loaded);
        RuntimeConfiguration c;c.metallib=NUMI_MATTER_METALLIB;c.environmentCount=2;c.adaptiveTransfer=false;c.captureEvents=false;c.captureDiagnostics=true;
        const auto initialized=[&](){ScopedSharedFactorEnvironment policy(sharedFactorEnabled);return runtime.initialize(world,c);}();need(initialized.encoded,initialized.message);statuses=buffer(device.device,std::vector<MRMetalWorldStatusGPU>(2));}
    RuntimeStateSnapshot state(){auto s=runtime.snapshot();need(s.available,s.message);return s;}
    void restore(const RuntimeStateSnapshot& s){const auto r=runtime.restore(s);need(r.encoded,r.message);}
    void step(unsigned index,int reject=-1,int reset=-1,
              const std::vector<nm_float4>& femTargets={}){
        auto* incoming=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);for(unsigned e=0;e<2;++e){incoming[e]={};incoming[e].environment=e;}
        auto command=[device.queue commandBuffer];EncodeRequest request;request.commandBuffer=(__bridge void*)command;request.environmentStatuses=(__bridge void*)statuses;
        request.controlStep=index;request.physicsSubsteps=1;request.timestepSeconds=runtime.timestepSeconds();request.runAdaptiveTransfer=false;
        id<MTLBuffer> resets=nil,failure=nil,targets=nil;
        if(!femTargets.empty()){
            const std::size_t expected=std::size_t(world.dispatch.environmentCount)*
                world.dispatch.femNodeCount;
            need(femTargets.size()==expected,"frame test FEM target arity");
            targets=buffer(device.device,femTargets);
            request.femKinematicTargets=(__bridge void*)targets;
            request.femKinematicTargetCount=static_cast<std::uint32_t>(femTargets.size());
        }
        if(reset>=0){std::vector<unsigned> mask((index+1)*2);mask[index*2+unsigned(reset)]=1;resets=buffer(device.device,mask);request.resetMasks=(__bridge void*)resets;request.resetMaskStepStride=2;}
        request.phase=EncodePhase::preDynamics;auto result=runtime.encode(request);need(result.encoded,result.message);
        if(reject>=0){MRMetalWorldStatusGPU bad{};bad.environment=unsigned(reject);bad.code=MR_STEP_DID_NOT_CONVERGE;failure=buffer(device.device,std::vector<MRMetalWorldStatusGPU>{bad});auto blit=[command blitCommandEncoder];[blit copyFromBuffer:failure sourceOffset:0 toBuffer:statuses destinationOffset:unsigned(reject)*sizeof(bad) size:sizeof(bad)];[blit endEncoding];}
        request.phase=EncodePhase::postCommit;result=runtime.encode(request);need(result.encoded,result.message);[command commit];[command waitUntilCompleted];need(command.status==MTLCommandBufferStatusCompleted,"integrated frame command failed");
        const auto after=state();for(unsigned e=0;e<2;++e)if(int(e)!=reject)need(after.statuses[e].code==NM_STATUS_SUCCESS,"integrated frame step="+std::to_string(index)+" env="+std::to_string(e)+" status="+std::to_string(after.statuses[e].code)+" residual="+number(after.statuses[e].diagnostics.z));
        if(reject>=0)need(after.statuses[unsigned(reject)].code!=NM_STATUS_SUCCESS,"injected frame rejection lost");(void)resets;(void)failure;(void)targets;
    }
};
void checkAcceptedPlasticCacheActions(Device& device,
                                     MaterialProgram paperMaterial,
                                     const bool accurateMode){
    need(std::getenv("NM_FEM_FORCE_COMPLETED_TANGENT_CACHE_REBUILD") == nullptr,
         "cache tail parity fixture owns the completed-cache rebuild policy");
    // This source-derived 15-state bilinear-tension-fit program is used here
    // as a native evaluator fixture. Geometry, imposed displacement, and
    // accepted path remain synthetic and do not calibrate a corrugated board.
    paperMaterial.genericCompensatedImplicitProjection=true;
    paperMaterial.genericBilinear15AccurateProjection=accurateMode;
    WorldSource authored=source(paperMaterial,true,false);
    authored.objects.front().femInitialVelocity={0.0,0.0,0.0};
    authored.mixedSolver.femPreconditioner =
        FEMPreconditionerMode::regionalTangentDiagonal;
    authored.mixedSolver.cacheFEMAlgorithmicTangent=true;
    WorldSource directAuthored = authored;
    directAuthored.mixedSolver.cacheFEMAlgorithmicTangent = false;
    const CompiledWorld world=cook(authored);
    need(world.dispatch.environmentCount==2u&&
         world.dispatch.tetrahedronCount==2u&&
         world.dispatch.materialStateStride>=paperMaterial.internalState.size(),
         "source-derived bilinear paper fixture compiled with an unexpected layout");
    const bool hasSharedFactorization =
        (world.materials[0].flags &
         NM_MATERIAL_SHARED_15_STATE_TANGENT_FACTORIZATION) != 0u;
    need(hasSharedFactorization == accurateMode &&
         world.materials[0].validityProgram != NM_INVALID_INDEX,
         "native tangent-cache fixture disagrees with its accurate shared-factor mode");

    Run run(device,authored,accurateMode);
    Run legacyCacheRun(device,authored,false);
    Run directRun(device,directAuthored,false);
    // The override is captured at initialization. Both cache runs then use
    // identical authored worlds and full accepted loading/unloading paths.
    // Restore any caller policy even if runtime initialization throws.
    struct ForcedTailRebuild {
        bool hadOriginal = false;
        std::string original;
        ForcedTailRebuild() {
            if (const char* value = std::getenv(
                    "NM_FEM_FORCE_COMPLETED_TANGENT_CACHE_REBUILD")) {
                hadOriginal = true;
                original = value;
            }
            need(setenv("NM_FEM_FORCE_COMPLETED_TANGENT_CACHE_REBUILD", "1", 1) == 0,
                 "cannot author tangent-cache rebuild control");
        }
        ~ForcedTailRebuild() {
            if (hadOriginal)
                setenv("NM_FEM_FORCE_COMPLETED_TANGENT_CACHE_REBUILD",
                       original.c_str(), 1);
            else unsetenv("NM_FEM_FORCE_COMPLETED_TANGENT_CACHE_REBUILD");
        }
    };
    std::unique_ptr<Run> forcedRebuildRun;
    {
        ForcedTailRebuild force;
        forcedRebuildRun = std::make_unique<Run>(
            device, authored, accurateMode);
    }
    const auto compareTailRebuild = [&](const RuntimeStateSnapshot& guarded,
                                       const RuntimeStateSnapshot& forced,
                                       const char* phase,
                                       const bool rejected = false) {
        need(same(guarded.femNodes, forced.femNodes) &&
             same(guarded.femFields, forced.femFields) &&
             same(guarded.femMaterialState, forced.femMaterialState) &&
             same(guarded.femTopologyTetrahedra, forced.femTopologyTetrahedra) &&
             same(guarded.solverCertificates, forced.solverCertificates),
             std::string("completed-cache rebuild changed physical history/certificate at ") + phase);
        need(guarded.statuses.size() == forced.statuses.size(),
             "cache tail comparison status arity");
        for (std::size_t environment = 0u; environment < guarded.statuses.size();
             ++environment) {
            need(guarded.statuses[environment].code == forced.statuses[environment].code,
                 "cache tail changed terminal status");
            if (rejected && environment == 0u)
                need(guarded.solverCertificates[environment].validity.w == 0.0f,
                     "cache tail late rejection retained an accepted certificate");
        }
        std::cout << "{\"schema\":1,\"record\":\"completed_tangent_cache_rebuild_parity\","
                     "\"phase\":\"" << phase
                  << "\",\"physical_history_bitwise_equal\":true,"
                     "\"certificates_bitwise_equal\":true,"
                     "\"late_rejection\":" << (rejected ? "true" : "false")
                  << ",\"physical_validation\":false}\n";
    };
    const auto compareAcceptedRuns = [&](const RuntimeStateSnapshot& direct,
                                         const RuntimeStateSnapshot& cached,
                                         const char* phase) {
        need(direct.femNodes.size() == cached.femNodes.size() &&
             direct.femMaterialState.size() == cached.femMaterialState.size() &&
             direct.solverCertificates.size() == cached.solverCertificates.size(),
             std::string("cached preconditioner changed readback arity at ") + phase);
        double maximumPositionDifference = 0.0;
        double displacementScale = 0.0;
        double maximumDisplacementDifference = 0.0;
        double maximumStateDifference = 0.0;
        double stateScale = 0.0;
        double maximumCertificateDifference = 0.0;
        double certificateScale = 1.0;
        for (std::size_t node = 0; node < direct.femNodes.size(); ++node) {
            const auto& a = direct.femNodes[node];
            const auto& b = cached.femNodes[node];
            const auto& reference = world.fem.nodes[node];
            for (unsigned axis = 0; axis < 3; ++axis) {
                const double positionA = reinterpret_cast<const float*>(
                    &a.positionAndMass)[axis];
                const double positionB = reinterpret_cast<const float*>(
                    &b.positionAndMass)[axis];
                const double initial = reinterpret_cast<const float*>(
                    &reference.positionAndMass)[axis];
                need(std::isfinite(positionA) && std::isfinite(positionB) &&
                     std::isfinite(initial),
                     std::string("nonfinite accepted position at ") + phase);
                maximumPositionDifference = std::max(
                    maximumPositionDifference, std::abs(positionA-positionB));
                maximumDisplacementDifference = std::max(
                    maximumDisplacementDifference,
                    std::abs((positionA-initial)-(positionB-initial)));
                displacementScale = std::max({displacementScale,
                    std::abs(positionA-initial), std::abs(positionB-initial)});
            }
            need(a.positionAndMass.w == b.positionAndMass.w &&
                 a.velocityAndInverseMass.w == b.velocityAndInverseMass.w &&
                 a.restAndFixed.w == b.restAndFixed.w,
                 std::string("cache changed node mass or fixed marker at ") + phase);
        }
        for (std::size_t i = 0; i < direct.femMaterialState.size(); ++i) {
            const double a = direct.femMaterialState[i];
            const double b = cached.femMaterialState[i];
            need(std::isfinite(a) && std::isfinite(b),
                 std::string("nonfinite accepted material state at ") + phase);
            maximumStateDifference = std::max(maximumStateDifference,
                std::abs(a-b));
            stateScale = std::max({stateScale, std::abs(a), std::abs(b)});
        }
        for (std::size_t e = 0; e < direct.solverCertificates.size(); ++e) {
            const auto& a = direct.solverCertificates[e];
            const auto& b = cached.solverCertificates[e];
            need(a.validity.w > 0.5f && b.validity.w > 0.5f,
                 std::string("direct/cache comparison lacks accepted certificate at ") + phase);
            const float* fieldsA = reinterpret_cast<const float*>(&a);
            const float* fieldsB = reinterpret_cast<const float*>(&b);
            for (std::size_t i = 0; i < sizeof(a)/sizeof(float); ++i) {
                need(std::isfinite(fieldsA[i]) && std::isfinite(fieldsB[i]),
                     std::string("nonfinite certificate at ") + phase);
                maximumCertificateDifference = std::max(
                    maximumCertificateDifference,
                    std::abs(double(fieldsA[i])-fieldsB[i]));
                certificateScale = std::max({certificateScale,
                    std::abs(double(fieldsA[i])), std::abs(double(fieldsB[i]))});
            }
        }
        const double displacementRelative = maximumDisplacementDifference /
            std::max(displacementScale, 1.0e-12);
        const double stateRelative = maximumStateDifference /
            std::max(stateScale, 1.0e-12);
        std::cout << std::setprecision(12)
            << "{\"schema\":1,\"record\":\"paper_preconditioner_cache_parity\""
            << ",\"phase\":\"" << phase << "\""
            << ",\"state_count\":" << paperMaterial.internalState.size()
            << ",\"selector\":1"
            << ",\"position_max_m\":" << maximumPositionDifference
            << ",\"displacement_relative\":" << displacementRelative
            << ",\"state_max_abs\":" << maximumStateDifference
            << ",\"state_relative\":" << stateRelative
            << ",\"certificate_max_abs\":" << maximumCertificateDifference
            << ",\"certificate_scale\":" << certificateScale
            << ",\"physical_validation\":false}\n";
        need(maximumPositionDifference < 1.0e-7 &&
             displacementRelative < 3.0e-3 && stateRelative < 3.0e-3 &&
             maximumCertificateDifference <= 2.0e-6*certificateScale,
             std::string("cached preconditioner changed accepted solve beyond parity bounds at ") + phase);
    };
    const auto plasticIndex=materialStateIndex(paperMaterial,
                                               "accumulated_multiplier");
    run.step(0u);
    legacyCacheRun.step(0u);
    directRun.step(0u);
    forcedRebuildRun->step(0u);
    const RuntimeStateSnapshot elastic=run.state();
    const RuntimeStateSnapshot legacyElastic=legacyCacheRun.state();
    const RuntimeStateSnapshot directElastic=directRun.state();
    compareAcceptedRuns(directElastic,elastic,"elastic-initial");
    compareAcceptedRuns(legacyElastic,elastic,accurateMode
        ? "shared-factor-vs-legacy-cache-elastic"
        : "nonaccurate-vs-legacy-cache-elastic");
    const RuntimeStateSnapshot forcedElastic = forcedRebuildRun->state();
    compareTailRebuild(elastic, forcedElastic, "elastic-initial");
    need(elastic.solverCertificates.size()==2u,
         "elastic bilinear paper snapshot lacks certificates");
    for(const auto& cert:elastic.solverCertificates)
        need(cert.validity.w>0.5f,
             "elastic bilinear paper state is not solver-certified");
    const auto loadTargets=paperShearTargets(elastic,5.0e-6f);
    const auto directLoadTargets=paperShearTargets(directElastic,5.0e-6f);
    const auto legacyLoadTargets=paperShearTargets(legacyElastic,5.0e-6f);
    run.step(1u,-1,-1,loadTargets);
    legacyCacheRun.step(1u,-1,-1,legacyLoadTargets);
    directRun.step(1u,-1,-1,directLoadTargets);
    forcedRebuildRun->step(1u,-1,-1,paperShearTargets(forcedElastic,5.0e-6f));
    const RuntimeStateSnapshot loaded=run.state();
    const RuntimeStateSnapshot legacyLoaded=legacyCacheRun.state();
    const RuntimeStateSnapshot directLoaded=directRun.state();
    compareAcceptedRuns(directLoaded,loaded,"accepted-plastic-load");
    compareAcceptedRuns(legacyLoaded,loaded,accurateMode
        ? "shared-factor-vs-legacy-cache-load"
        : "nonaccurate-vs-legacy-cache-load");
    compareTailRebuild(loaded,forcedRebuildRun->state(),"accepted-plastic-load");
    need(loaded.solverCertificates.size()==2u,
         "loaded bilinear paper snapshot lacks certificates");
    for(const auto& cert:loaded.solverCertificates)
        need(cert.validity.w>0.5f,
             "loaded bilinear paper state is not solver-certified");
    const double loadedPlastic=maximumMaterialHistory(loaded,plasticIndex);
    need(loadedPlastic>1.0e-9,
         "accepted bilinear-paper load did not activate plastic history");
    const Inputs elasticInputs=acceptedInputs(elastic);
    checkCachedOperatorActions(device,world,elasticInputs,
                               "accepted-elastic-initial");
    (void)dispatch(device,world,elasticInputs,true,true,true);
    const Inputs loadedInputs=acceptedInputs(loaded);
    checkCachedOperatorActions(device,world,loadedInputs,
                               "accepted-plastic-after-load");

    // Return both temporarily gripped nodes to their original x positions.
    // Stress-like states unload, while the accepted multiplier remains
    // irreversible; this is a distinct accepted history, not a trial state.
    const auto unloadTargets=paperShearTargets(elastic,0.0f);
    run.step(2u,-1,-1,unloadTargets);
    legacyCacheRun.step(2u,-1,-1,
                       paperShearTargets(legacyElastic,0.0f));
    directRun.step(2u,-1,-1,
                   paperShearTargets(directElastic,0.0f));
    forcedRebuildRun->step(2u,-1,-1,paperShearTargets(forcedElastic,0.0f));
    const RuntimeStateSnapshot unloaded=run.state();
    const RuntimeStateSnapshot legacyUnloaded=legacyCacheRun.state();
    const RuntimeStateSnapshot directUnloaded=directRun.state();
    compareAcceptedRuns(directUnloaded,unloaded,"accepted-unload");
    compareAcceptedRuns(legacyUnloaded,unloaded,accurateMode
        ? "shared-factor-vs-legacy-cache-unload"
        : "nonaccurate-vs-legacy-cache-unload");
    compareTailRebuild(unloaded,forcedRebuildRun->state(),"accepted-unload");
    need(unloaded.solverCertificates.size()==2u,
         "unloaded bilinear paper snapshot lacks certificates");
    for(const auto& cert:unloaded.solverCertificates)
        need(cert.validity.w>0.5f,
             "unloaded bilinear paper state is not solver-certified");
    const double unloadedPlastic=maximumMaterialHistory(unloaded,plasticIndex);
    need(unloadedPlastic+2.0e-7>=loadedPlastic,
         "accepted plastic multiplier decreased during unloading");
    const Inputs unloadedInputs=acceptedInputs(unloaded);
    checkCachedOperatorActions(device,world,unloadedInputs,
                               "accepted-plastic-after-unload");
    double maximumHistoryChange=0.0;
    need(loaded.femMaterialState.size()==unloaded.femMaterialState.size(),
         "load/unload material history arity changed");
    for(std::size_t index=0;index<loaded.femMaterialState.size();++index)
        maximumHistoryChange=std::max(maximumHistoryChange,
            std::abs(double(unloaded.femMaterialState[index])-
                     double(loaded.femMaterialState[index])));
    need(maximumHistoryChange>1.0e-7,
         "accepted unload did not produce a distinct constitutive state");
    std::cout<<std::setprecision(12)
        <<"{\"schema\":1,\"record\":\"accepted_bilinear_paper_plastic_history\""
        <<",\"source_material\":\""<<paperMaterial.name<<"\""
        <<",\"state_count\":"<<paperMaterial.internalState.size()
        <<",\"material_frames\":\"rotated per tetrahedron\""
        <<",\"loading_accepted\":true,\"unloading_accepted\":true"
        <<",\"maximum_loaded_plastic_multiplier\":"<<loadedPlastic
        <<",\"maximum_unloaded_plastic_multiplier\":"<<unloadedPlastic
        <<",\"maximum_load_to_unload_state_difference\":"<<maximumHistoryChange
        <<",\"tangent_preconditioner_direct_cache_history_parity\":true"
        <<",\"shared_factor_enabled\":"
        <<(accurateMode?"true":"false")
        <<",\"physical_validation\":false}\n";
    // Deliberately reject postCommit for environment zero after the complete
    // accepted load/unload fixture, then compare restored history and cleared
    // publication certificates for both rebuilding policies.
    run.step(3u,0);
    legacyCacheRun.step(3u,0);
    directRun.step(3u,0);
    forcedRebuildRun->step(3u,0);
    const auto rejected = run.state();
    const auto legacyRejected = legacyCacheRun.state();
    const auto directRejected = directRun.state();
    compareTailRebuild(rejected,forcedRebuildRun->state(),
                       "late-postcommit-rejection",true);
    need(rejected.statuses.size()==legacyRejected.statuses.size()&&
         rejected.statuses.size()==directRejected.statuses.size(),
         "shared-factor failure status comparison arity");
    for(std::size_t environment=0;environment<rejected.statuses.size();++environment)
        need(rejected.statuses[environment].code==
                 legacyRejected.statuses[environment].code&&
             rejected.statuses[environment].code==
                 directRejected.statuses[environment].code,
             "shared-factor opt-in changed final status on rejection");
    envSame(unloaded.femNodes,rejected.femNodes,0u,"guarded rejected nodes");
    envSame(unloaded.femMaterialState,rejected.femMaterialState,0u,
            "guarded rejected material history");
    envSame(legacyUnloaded.femNodes,legacyRejected.femNodes,0u,
            "legacy-cache rejected nodes");
    envSame(legacyUnloaded.femMaterialState,legacyRejected.femMaterialState,0u,
            "legacy-cache rejected material history");
    envSame(directUnloaded.femNodes,directRejected.femNodes,0u,
            "direct rejected nodes");
    envSame(directUnloaded.femMaterialState,directRejected.femMaterialState,0u,
            "direct rejected material history");
}
bool physicalSame(const RuntimeStateSnapshot& a,const RuntimeStateSnapshot& b){return same(a.femNodes,b.femNodes)&&same(a.femFields,b.femFields)&&same(a.femMaterialState,b.femMaterialState)&&same(a.femTopologyTetrahedra,b.femTopologyTetrahedra);}
void integratedChecks(Device& d,const MaterialProgram& material){
    Run run(d,source(material));const auto initial=run.state();RuntimeStateSnapshot one;constexpr unsigned steps=16;
    double motion=0;for(unsigned i=0;i<steps;++i){run.step(i);const auto s=run.state();if(i==0)one=s;paired(s.femNodes,"accepted nodes");paired(s.femMaterialState,"accepted material state");
        for(unsigned n=0;n<s.femNodes.size();++n){need(std::bit_cast<unsigned>(s.femNodes[n].positionAndMass.w)==std::bit_cast<unsigned>(initial.femNodes[n].positionAndMass.w),"material frame changed mechanical nodal mass");
            const auto delta=sub(xyz(s.femNodes[n].positionAndMass),xyz(initial.femNodes[n].positionAndMass));motion=std::max(motion,std::sqrt(dot(delta,delta)));}}
    const auto final=run.state();need(motion>1e-7,"integrated frame trajectory did not move");run.restore(initial);for(unsigned i=0;i<steps;++i)run.step(i);need(physicalSame(final,run.state()),"material frame replay not bitwise");
    const auto before=run.state();run.step(steps,1);const auto rejected=run.state();envSame(before.femNodes,rejected.femNodes,1,"rejected nodes");envSame(before.femMaterialState,rejected.femMaterialState,1,"rejected material state");envSame(before.femTopologyTetrahedra,rejected.femTopologyTetrahedra,1,"rejected frames");
    auto corrupt=rejected;need(!corrupt.femTopologyTetrahedra.empty(),"missing frame snapshot topology");corrupt.femTopologyTetrahedra[0].materialFrameRotation={0,0,0,1};
    auto denied=run.runtime.restore(corrupt);need(!denied.encoded&&physicalSame(rejected,run.state()),"valid alternate quaternion bypassed snapshot frame identity");
    corrupt=rejected;corrupt.femTopologyTetrahedra[0].materialFrameRotation.x=std::numeric_limits<float>::quiet_NaN();denied=run.runtime.restore(corrupt);
    need(!denied.encoded&&physicalSame(rejected,run.state()),"nonfinite snapshot frame accepted or mutated state");
    // A frame belongs to its exact immutable cell. Checking only q would
    // allow the same quaternion to be rebound through nodes/rest/material.
    const std::vector<std::pair<std::string,std::function<void(NMTetrahedronGPU&)>>> rebindings{
        {"nodes",[](auto& t){std::swap(t.nodes.x,t.nodes.y);}},
        {"rest operator",[](auto& t){t.inverseRestRow0.x=std::nextafter(t.inverseRestRow0.x,std::numeric_limits<float>::infinity());}},
        {"material identity",[](auto& t){++t.identity.x;}},
        {"object identity",[](auto& t){++t.identity.y;}},
        {"active flag",[](auto& t){t.identity.w^=NM_OBJECT_ACTIVE;}}};
    for(const auto& [name,change]:rebindings){corrupt=rejected;change(corrupt.femTopologyTetrahedra[0]);denied=run.runtime.restore(corrupt);
        need(!denied.encoded&&physicalSame(rejected,run.state()),"snapshot frame "+name+" rebinding was accepted or mutated state");}
    corrupt=rejected;corrupt.adaptive[0].activeRepresentation=NM_REPRESENTATION_RIGID;denied=run.runtime.restore(corrupt);
    need(!denied.encoded&&physicalSame(rejected,run.state()),"snapshot changed framed FEM into rigid representation");
    corrupt=rejected;++corrupt.allocationGeneration;denied=run.runtime.restore(corrupt);
    need(!denied.encoded&&physicalSame(rejected,run.state()),"snapshot changed immutable framed allocation generation");
    run.step(steps+1,-1,1);const auto reset=run.state();envSame(one.femNodes,reset.femNodes,1,"reset nodes");envSame(one.femTopologyTetrahedra,reset.femTopologyTetrahedra,1,"reset frames");
    std::cout<<"accepted_steps="<<steps<<" environments=2 motion_m="<<motion<<" replay=bitwise rollback=isolated reset=bitwise frame_snapshot_tamper=denied immutable_cell_rebinding=denied mass=unchanged\n";
}
} // namespace

int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==2,"usage: fem-material-frame-check GUCCIONE.nmatter");
    const char* bundleCheckOnly=std::getenv("NM_FEM_MATERIAL_BUNDLE_CHECK_ONLY");
    need(bundleCheckOnly==nullptr || std::strcmp(bundleCheckOnly,"0")==0 ||
         std::strcmp(bundleCheckOnly,"1")==0,
         "NM_FEM_MATERIAL_BUNDLE_CHECK_ONLY must be 0 or 1");
    const char* statelessSpecialization=
        std::getenv("NM_FEM_STATELESS_MATERIAL_SPECIALIZATION");
    need(statelessSpecialization==nullptr ||
         std::strcmp(statelessSpecialization,"0")==0 ||
         std::strcmp(statelessSpecialization,"1")==0,
         "NM_FEM_STATELESS_MATERIAL_SPECIALIZATION must be 0 or 1");
    const bool specializeStatelessMaterials=statelessSpecialization!=nullptr &&
        std::strcmp(statelessSpecialization,"1")==0;
    const char* nativeDiagnosticCheck=
        std::getenv("NM_FEM_MATERIAL_NATIVE_DIAGNOSTIC_CHECK_ONLY");
    need(nativeDiagnosticCheck==nullptr ||
         std::strcmp(nativeDiagnosticCheck,"0")==0 ||
         std::strcmp(nativeDiagnosticCheck,"1")==0,
         "NM_FEM_MATERIAL_NATIVE_DIAGNOSTIC_CHECK_ONLY must be 0 or 1");
    const bool nativeDiagnosticCheckOnly=nativeDiagnosticCheck!=nullptr &&
        std::strcmp(nativeDiagnosticCheck,"1")==0;
    if(specializeStatelessMaterials)
        need(bundleCheckOnly!=nullptr&&std::strcmp(bundleCheckOnly,"1")==0,
            "stateless material specialization is supported here only with NM_FEM_MATERIAL_BUNDLE_CHECK_ONLY=1");
    if(nativeDiagnosticCheckOnly){
        need(!specializeStatelessMaterials &&
             (bundleCheckOnly==nullptr||std::strcmp(bundleCheckOnly,"0")==0),
             "native diagnostic finalizer check is mutually exclusive with function-constant-30 bundle specialization");
        Device diagnosticDevice(false,true);
        checkNativeDiagnosticFinalizer(diagnosticDevice);
        std::cout<<"fem_material_native_diagnostic_check=pass "
            "scope=finalizer_marker_only function_constant30=undefined "
            "function_constant31=undefined"<<std::endl;
        return 0;
    }
    auto parsed=parseMatterFile(argv[1]);
    need(parsed.succeeded(),diagnostic(parsed.diagnostics));
    set(parsed.material,"density",1000); // synthetic only
    if(bundleCheckOnly!=nullptr && std::strcmp(bundleCheckOnly,"1")==0){
        Device bundleDevice(specializeStatelessMaterials);
        std::cout<<std::setprecision(17)<<"frame_device="
            <<[bundleDevice.device name].UTF8String
            <<" abi="<<NM_MATTER_ABI_VERSION<<'\n';
        std::cout<<"stateless_material_specialization="
            <<(specializeStatelessMaterials?"on":"off")
            <<" function_constant30_pipelines="
            <<bundleDevice.functionConstant30PipelineCount
            <<" function_constant31=undefined"<<std::endl;
        checkNewtonTailDispatch(bundleDevice);
        statelessBundleChecks(bundleDevice,parsed.material);
        if(specializeStatelessMaterials){
            Device legacyDevice(false);
            statelessViscousSpecializationChecks(bundleDevice,legacyDevice);
        }
        std::cout<<"fem_material_bundle_check=pass scope=serial_cooperative_nonzero_parameter_offset full_suite=skipped\n";
        return 0;
    }
    const auto paperPath=std::filesystem::path(argv[1]).parent_path()/
        "hajali2009_liner_hill_bilinear_fit.nmatter";
    auto paper=parseMatterFile(paperPath.string());
    need(paper.succeeded(),"source-derived bilinear paper material parse failed: "+
         diagnostic(paper.diagnostics));
    checkSharedFactorCompilerProof(paper.material);
    Device device;std::cout<<std::setprecision(17)<<"frame_device="<<[device.device name].UTF8String<<" abi="<<NM_MATTER_ABI_VERSION<<'\n';
    checkColumnValidityPublication(device);
    checkCompensatedLocalFactor(device);
    canonicalTangentChecks(device,std::filesystem::path(argv[1]).parent_path());
    directChecks(device,parsed.material);learnedFrameChecks(device);integratedChecks(device,parsed.material);
    checkAcceptedPlasticCacheActions(device,paper.material,false);
    checkAcceptedPlasticCacheActions(device,paper.material,true);
    std::cout<<"fem_material_frame_check=pass anatomical_wall_qualified=false source_density_supplied=false active_cardiac_source_reproduced=false\n";return 0;
}catch(const std::exception& error){std::cerr<<"fem material frame check failed: "<<error.what()<<'\n';return 1;}}}
