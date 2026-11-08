#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalCoupledRod.hpp"
#include "metalrobo/coupled_rod_gpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace metalrobo {
namespace {
using V = std::array<double, 3>;
bool finite3(const V& x) {
    return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
}
bool asFloat(double x, float& out, bool positive = false) {
    if (!std::isfinite(x)) return false;
    out = static_cast<float>(x);
    return std::isfinite(out) && (!positive || out >= std::numeric_limits<float>::min());
}
bool split(double x, float& high, float& low) {
    return asFloat(x, high) && asFloat(x - static_cast<double>(high), low);
}
V subtract(const V& a, const V& b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
double norm(const V& x) { return std::hypot(x[0],x[1],x[2]); }
V cross(const V& a, const V& b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
double dot(const V& a, const V& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
bool uniform(const std::vector<double>& a) {
    return !a.empty() && std::all_of(a.begin(), a.end(), [&](double x) { return x == a.front(); });
}
MetalDiscreteElasticRodDiagnostics fail(MetalDiscreteElasticRodHostStatus status, const char* message) {
    MetalDiscreteElasticRodDiagnostics d; d.status=status; d.message=message; return d;
}
}

MetalDiscreteElasticRodDiagnostics runResearchMetalCoupledRod(
    const DiscreteElasticRodModel& model,
    const DiscreteElasticRodState& initial,
    std::span<const DiscreteRodAttachment> attachments,
    MetalCoupledRodResult& output,
    const MetalCoupledRodConfig& config
) {
    @autoreleasepool {
        std::string reason;
        if (!model.valid(&reason)) return fail(MetalDiscreteElasticRodHostStatus::invalidModel, "invalid rod model");
        const auto n=model.restPositions.size();
        if (n<3 || n>ROD_N || !uniform(model.restLengths) || !uniform(model.stretchStiffness) ||
            !uniform(model.bendStiffness) || !uniform(model.twistStiffness) ||
            !std::all_of(model.restTwists.begin(),model.restTwists.end(),[](double t){return t==0.;}))
            return fail(MetalDiscreteElasticRodHostStatus::invalidConfiguration, "unsupported rod topology or material");
        const V axis=subtract(model.restPositions[1],model.restPositions[0]);
        const double ell=model.restLengths[0];
        for (std::size_t i=1; i<n-1; ++i) {
            const V e=subtract(model.restPositions[i+1],model.restPositions[i]);
            if (!(dot(axis,e)>0.) || norm(cross(axis,e))>1e-11*ell*ell)
                return fail(MetalDiscreteElasticRodHostStatus::invalidConfiguration, "rest geometry is not straight");
        }
        if (initial.positions.size()!=n || initial.velocities.size()!=n ||
            initial.twists.size()!=n-1 || initial.twistRates.size()!=n-1)
            return fail(MetalDiscreteElasticRodHostStatus::invalidState, "invalid state dimensions");
        if (attachments.size()!=2 || attachments[0].nodeIndex!=0 || attachments[1].nodeIndex!=n-1 ||
            attachments[0].compliance!=0. || attachments[1].compliance!=0.)
            return fail(MetalDiscreteElasticRodHostStatus::invalidAttachment, "two ordered hard endpoint supports required");
        if (!config.maximumIterations || config.metallibPath.empty())
            return fail(MetalDiscreteElasticRodHostStatus::invalidConfiguration, "invalid solver configuration");
        RodInput in{}; in.n=static_cast<unsigned>(n); in.maxIterations=config.maximumIterations;
        if (!asFloat(config.timestep,in.h,true) || !asFloat(config.linearResidualTolerance,in.linearTolerance,true) ||
            !asFloat(config.angularResidualTolerance,in.angularTolerance,true) ||
            !split(ell,in.ellHigh,in.ellLow) || !(in.ellHigh>0) ||
            !asFloat(model.stretchStiffness[0],in.ea,true) || !asFloat(model.bendStiffness[0],in.ei,true) ||
            !asFloat(model.twistStiffness[0],in.gj,true))
            return fail(MetalDiscreteElasticRodHostStatus::invalidConfiguration, "configuration is not representable in Metal");
        const double h=in.h;
        for (std::size_t i=0;i<n;i++) {
            if (!finite3(initial.positions[i]) || !finite3(initial.velocities[i]) ||
                !asFloat(model.nodeMasses[i],in.mass[i],true))
                return fail(MetalDiscreteElasticRodHostStatus::invalidState, "invalid node or mass");
            for (unsigned j=0;j<3;j++)
                if (!asFloat(initial.velocities[i][j],in.vel[3*i+j]))
                    return fail(MetalDiscreteElasticRodHostStatus::invalidState, "velocity is not representable");
        }
        double center=0.;
        for (double x:initial.twists) { if (!std::isfinite(x)) return fail(MetalDiscreteElasticRodHostStatus::invalidState, "nonfinite twist"); center+=x/static_cast<double>(n-1); }
        if (!std::isfinite(center)) return fail(MetalDiscreteElasticRodHostStatus::invalidState, "invalid twist center");
        for (std::size_t i=0;i<n-1;i++) {
            for (unsigned j=0;j<3;j++)
                if (!split(initial.positions[i+1][j]-initial.positions[i][j],in.edgeHigh[3*i+j],in.edgeLow[3*i+j]))
                    return fail(MetalDiscreteElasticRodHostStatus::invalidState, "edge is not representable");
            if (!asFloat(initial.twists[i]-center,in.twist[i]) || !asFloat(initial.twistRates[i],in.omega[i]) ||
                !asFloat(model.edgeRotationalInertias[i],in.inertia[i],true))
                return fail(MetalDiscreteElasticRodHostStatus::invalidState, "twist or inertia is not representable");
        }
        for (unsigned k=0;k<2;k++) {
            const auto& a=attachments[k]; const std::size_t node=k ? n-1 : 0;
            if (!finite3(a.targetPosition) || !finite3(a.targetVelocity))
                return fail(MetalDiscreteElasticRodHostStatus::invalidAttachment, "nonfinite support");
            for (unsigned j=0;j<3;j++) {
                const double delta=a.targetPosition[j]-initial.positions[node][j];
                if (!split(delta,in.targetDelta[3*k+j],in.targetDeltaLow[3*k+j]) ||
                    std::abs(a.targetVelocity[j]-delta/h)>1e-9*std::max(1.,std::abs(delta/h)))
                    return fail(MetalDiscreteElasticRodHostStatus::invalidAttachment, "support velocity or displacement is inconsistent");
            }
        }
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if (!device) return fail(MetalDiscreteElasticRodHostStatus::metalDeviceUnavailable, "Metal device unavailable");
        NSError* error=nil;
        NSURL* url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:config.metallibPath.c_str()]];
        id<MTLLibrary> library=[device newLibraryWithURL:url error:&error];
        if (!library) return fail(MetalDiscreteElasticRodHostStatus::metalLibraryFailure, "Metal library unavailable");
        id<MTLFunction> function=[library newFunctionWithName:@"mr_research_coupled_rod_step"];
        if (!function) return fail(MetalDiscreteElasticRodHostStatus::metalPipelineFailure, "coupled rod kernel unavailable");
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) return fail(MetalDiscreteElasticRodHostStatus::metalPipelineFailure, "coupled rod pipeline failed");
        RodOutput zero{};
        id<MTLBuffer> ib=[device newBufferWithBytes:&in length:sizeof(in) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob=[device newBufferWithBytes:&zero length:sizeof(zero) options:MTLResourceStorageModeShared];
        if (!ib || !ob) return fail(MetalDiscreteElasticRodHostStatus::metalBufferFailure, "coupled rod allocation failed");
        id<MTLCommandQueue> queue=[device newCommandQueue];
        id<MTLCommandBuffer> cmd=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[cmd computeCommandEncoder];
        if (!queue || !cmd || !encoder) return fail(MetalDiscreteElasticRodHostStatus::metalCommandFailure, "could not encode coupled rod");
        [encoder setComputePipelineState:pipeline]; [encoder setBuffer:ib offset:0 atIndex:0]; [encoder setBuffer:ob offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [encoder endEncoding];
        [cmd commit]; [cmd waitUntilCompleted];
        if (cmd.status!=MTLCommandBufferStatusCompleted) return fail(MetalDiscreteElasticRodHostStatus::metalCommandFailure, "coupled rod command failed");
        const RodOutput& result=*static_cast<const RodOutput*>(ob.contents);
        if (result.status!=0 || !std::isfinite(result.maxLinear) || !std::isfinite(result.maxAngular) ||
            result.maxLinear>in.linearTolerance || result.maxAngular>in.angularTolerance) {
            auto d=fail(MetalDiscreteElasticRodHostStatus::gpuEnvironmentFailure, "coupled rod did not converge");
            d.dispatched=true; d.firstFailingEnvironment=0;
            d.firstGPUStatusCode=result.status==2 ? MR_ROD_GPU_DEGENERATE_GEOMETRY : MR_ROD_GPU_DID_NOT_CONVERGE;
            d.message += " (research kernel status " + std::to_string(result.status) + ")"; return d;
        }
        auto publicationFailure=[&]() {
            auto d=fail(MetalDiscreteElasticRodHostStatus::gpuEnvironmentFailure, "public state cannot represent the accepted coupled rod step");
            d.dispatched=true; d.firstFailingEnvironment=0; d.firstGPUStatusCode=MR_ROD_GPU_SUCCESS; return d;
        };
        MetalCoupledRodResult staged; staged.state=initial;
        double linearConversion=0.0, angularConversion=0.0;
        for (std::size_t i=0;i<n;i++) for (unsigned j=0;j<3;j++) {
            const double delta=double(result.displacement[3*i+j])+double(result.displacementLow[3*i+j]);
            staged.state.positions[i][j]+=delta; staged.state.velocities[i][j]=delta/h;
            const double represented=staged.state.positions[i][j]-initial.positions[i][j];
            linearConversion=std::max(linearConversion, std::abs(represented-delta)/h + std::abs(double(in.vel[3*i+j])-initial.velocities[i][j]));
            if (!std::isfinite(staged.state.positions[i][j]) || !std::isfinite(staged.state.velocities[i][j]))
                return publicationFailure();
        }
        for (std::size_t i=0;i<n-1;i++) {
            staged.state.twists[i]=center+double(result.twistDelta[i]);
            staged.state.twistRates[i]=(staged.state.twists[i]-initial.twists[i])/h;
            const double represented=staged.state.twists[i]-initial.twists[i];
            const double gpuDelta=double(result.twistDelta[i])-double(in.twist[i]);
            angularConversion=std::max(angularConversion, std::abs(represented-gpuDelta)/h + std::abs(double(in.omega[i])-initial.twistRates[i]));
            if (!std::isfinite(staged.state.twists[i]) || !std::isfinite(staged.state.twistRates[i]))
                return publicationFailure();
        }
        if (!std::isfinite(linearConversion) || !std::isfinite(angularConversion) ||
            double(result.maxLinear)+linearConversion>in.linearTolerance ||
            double(result.maxAngular)+angularConversion>in.angularTolerance) return publicationFailure();
        for (unsigned k=0;k<2;k++) {
            auto& reaction=staged.reactions[k]; reaction.nodeIndex=attachments[k].nodeIndex;
            for (unsigned j=0;j<3;j++) {
                reaction.averageForceOnTarget[j]=result.reaction[3*k+j];
                reaction.impulseOnTarget[j]=reaction.averageForceOnTarget[j]*h;
                if (!std::isfinite(reaction.averageForceOnTarget[j]))
                    return publicationFailure();
            }
            reaction.finalPositionError=norm(subtract(staged.state.positions[reaction.nodeIndex],attachments[k].targetPosition));
        }
        staged.iterations=result.iterations; staged.backtracks=result.backtracks;
        staged.maximumLinearResidual=double(result.maxLinear)+linearConversion;
        staged.maximumAngularResidual=double(result.maxAngular)+angularConversion;
        output=std::move(staged);
        MetalDiscreteElasticRodDiagnostics d; d.dispatched=true; d.published=true; d.firstGPUStatusCode=MR_ROD_GPU_SUCCESS; d.deviceName=device.name.UTF8String; return d;
    }
}
} // namespace metalrobo
