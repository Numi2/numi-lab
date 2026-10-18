
#pragma once
#import <Metal/Metal.h>
#include "metalrobo/MetalArticulatedOperator.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>
// Unlike ground contact, these six rows are INTERNAL interface reactions.
// Their query Jacobians are relative (J_B-J_A). Audit them separately from
// external support impulses. No force or pose is scripted at release.
class NumiHumanSeparationSupport final {
public:
    struct alignas(16) GPU { mr_uint4 bodies,strides; mr_float4 parentAnchor; mr_uint4 event,rows[6]; };
private:
    metalrobo::MetalNumiHumanSupportGeometryProgram base_;
    GPU cfg_{};
    __strong id<MTLDevice> device_;
    __strong id<MTLComputePipelineState> pipeline_;
    std::uint64_t fingerprint_;
    static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
    static std::uint64_t mix(std::uint64_t h,std::uint64_t x){return (h^x)*1099511628211ull;}
    static bool callback(void* c,const metalrobo::MetalNumiHumanSupportGeometryPass& p) noexcept {
        try{return c&&static_cast<NumiHumanSeparationSupport*>(c)->encode(p);}catch(...){return false;}
    }
    static void abort(void* c,void* cb) noexcept {
        auto* s=static_cast<NumiHumanSeparationSupport*>(c);if(s&&s->base_.abort)s->base_.abort(s->base_.context,cb);
    }
    bool encode(const metalrobo::MetalNumiHumanSupportGeometryPass& p) {
        if((p.phase!=metalrobo::MetalNumiHumanSupportGeometryPhase::currentPose&&p.phase!=metalrobo::MetalNumiHumanSupportGeometryPhase::predictedFreeEndpoint)||p.abiVersion!=3||p.structSize!=sizeof(p)||p.environmentCount!=1||
           p.articulationFirstBody!=cfg_.bodies.z||p.dofCount!=cfg_.strides.w||
           p.bodyCount==0||p.bodyCount>192||p.bodyPoseStride<p.bodyCount||
           p.bodyPoseElementCount<p.bodyPoseStride||
           p.pointWorldStride<p.pointCount||p.pointWorldElementCount<p.pointWorldStride||
           p.pointJacobianStride<p.pointWorldStride*3*p.dofCount||
           p.pointJacobianElementCount<p.pointJacobianStride||
           p.bodyJacobianPointOffset==MR_INVALID_INDEX||
           p.bodyJacobianPointOffset+4*p.bodyCount>p.pointCount||
           cfg_.bodies.x<p.articulationFirstBody||cfg_.bodies.y<p.articulationFirstBody||
           cfg_.bodies.x-p.articulationFirstBody>=p.bodyCount||
           cfg_.bodies.y-p.articulationFirstBody>=p.bodyCount)return false;
        GPU d=cfg_;d.bodies.w=p.bodyJacobianPointOffset;
        d.strides.x=std::uint32_t(p.bodyPoseStride);d.strides.y=std::uint32_t(p.pointWorldStride);
        d.strides.z=std::uint32_t(p.pointJacobianStride);d.event.y=p.stepIndex;
        for(const auto& row:d.rows)
            if(row.x>=p.standContactCount||row.y>=p.pointCount||
               (row.y>=p.bodyJacobianPointOffset&&row.y<p.bodyJacobianPointOffset+4*p.bodyCount))return false;
        id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)p.commandBuffer;
        id<MTLBuffer> buffers[]={
            (__bridge id<MTLBuffer>)p.bodyPoses,(__bridge id<MTLBuffer>)p.bodyPositionLow,
            (__bridge id<MTLBuffer>)p.pointWorld,(__bridge id<MTLBuffer>)p.pointPositionLow,
            (__bridge id<MTLBuffer>)p.pointJacobians,(__bridge id<MTLBuffer>)p.standContacts};
        const std::uint64_t sizes[]={p.bodyPoseElementCount*sizeof(MRArticulatedBodyPoseGPU),
            p.bodyPoseElementCount*sizeof(mr_float4),p.pointWorldElementCount*sizeof(MRArticulatedPointWorldGPU),
            p.pointWorldElementCount*sizeof(mr_float4),p.pointJacobianElementCount*sizeof(float),
            p.standContactCount*sizeof(MRNumiHumanStandContactGPU)};
        if(!cb||cb.device.registryID!=device_.registryID)return false;
        for(unsigned i=0;i<6;++i)if(!buffers[i]||buffers[i].device.registryID!=device_.registryID||buffers[i].length<sizes[i])return false;
        if(!base_.encodePreDynamics(base_.context,p))return false;
        auto e=[cb computeCommandEncoder];if(!e)return false;
        e.label=@"Human internal separation interface";[e setComputePipelineState:pipeline_];
        [e setBytes:&d length:sizeof(d) atIndex:0];
        for(unsigned i=0;i<6;++i)[e setBuffer:buffers[i] offset:0 atIndex:i+1];
        [e dispatchThreads:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(6,1,1)];
        [e endEncoding];return true;
    }
public:
    template<class Contact>
    static std::array<std::uint32_t,6> selectIndices(
        std::span<const Contact> rows,std::uint32_t protectedRow,
        std::uint32_t upperBodyBegin=20u,std::uint32_t upperBodyEnd=130u) {
        std::array<std::uint32_t,6> result{};unsigned n=0;
        need(upperBodyBegin<upperBodyEnd,"invalid upper-body contact ownership range");
        for(std::uint32_t i=0;i<rows.size()&&n<6;++i)
            if(i!=protectedRow&&rows[i].bodyIndex>=upperBodyBegin&&rows[i].bodyIndex<upperBodyEnd){
                bool duplicate=false;
                for(unsigned j=0;j<n;++j) {
                    if constexpr(requires(const Contact& contact) { contact.pointQueryIndex; })
                        duplicate|=rows[result[j]].pointQueryIndex==rows[i].pointQueryIndex;
                    else
                        duplicate|=result[j]==i; // NHCNT source rows compile to distinct sequential queries.
                }
                if(duplicate)continue;
                result[n++]=i;
            }
        need(n==6,"separation needs six dedicated existing interface rows");return result;
    }
    static std::array<std::uint32_t,6> selectRows(
        std::span<MRNumiHumanStandContactGPU> rows,std::uint32_t protectedRow,
        std::uint32_t upperBodyBegin=20u,std::uint32_t upperBodyEnd=130u) {
        const auto result=selectIndices(
            std::span<const MRNumiHumanStandContactGPU>(rows.data(),rows.size()),
            protectedRow,upperBodyBegin,upperBodyEnd);
        for(const auto row:result)rows[row].frictionSlopAndStabilization={0,0,.25f,0};
        return result;
    }
    NumiHumanSeparationSupport(id<MTLDevice> device,id<MTLLibrary> library,
        metalrobo::MetalNumiHumanSupportGeometryProgram base,
        std::uint32_t a,std::uint32_t b,std::uint32_t firstBody,std::uint32_t dofs,
        std::uint32_t releaseStep,std::span<const MRNumiHumanStandContactGPU> contacts,
        const std::array<std::uint32_t,6>& selected, mr_float4 parentAnchor={})
        :base_(base),device_(device),fingerprint_(1469598103934665603ull){
        need(base.valid()&&device&&library&&a!=b&&dofs>0&&releaseStep>0,"invalid separation configuration");
        cfg_.parentAnchor=parentAnchor;cfg_.bodies={a,b,firstBody,0};cfg_.strides={0,0,0,dofs};cfg_.event={releaseStep,0,1,0};
        fingerprint_=mix(fingerprint_,base.fingerprint);
        for(unsigned i=0;i<6;++i){
            need(selected[i]<contacts.size(),"interface contact outside source rows");
            const auto& c=contacts[selected[i]];
            need(c.frictionSlopAndStabilization.x==0,"interface must be frictionless");
            for(unsigned j=0;j<i;++j)need(selected[i]!=selected[j]&&c.pointQueryIndex!=cfg_.rows[j].y,"interface rows alias");
            cfg_.rows[i]={selected[i],c.pointQueryIndex,i/2,i%2};
        }
        const auto bytes=std::as_bytes(std::span(&cfg_,1));
        for(auto byte:bytes)fingerprint_=mix(fingerprint_,std::to_integer<unsigned char>(byte));
        NSError* error=nil;
        auto f=[library newFunctionWithName:@"nm_human_separation_constraints"];
        need(f!=nil,"separation shader absent");pipeline_=[device newComputePipelineStateWithFunction:f error:&error];
        need(pipeline_!=nil,"separation pipeline unavailable");
    }
    metalrobo::MetalNumiHumanSupportGeometryProgram program(){
        return {.context=this,.encodePreDynamics=&callback,.abort=&abort,
            .fingerprint=fingerprint_,.usePerContactSupportPlanes=true};
    }
};
static_assert(sizeof(NumiHumanSeparationSupport::GPU)==160);
