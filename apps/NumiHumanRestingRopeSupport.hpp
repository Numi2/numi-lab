#pragma once
#include <Metal/Metal.h>
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_resting_rope_gpu.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

class NumiHumanRestingRopeSupport final {
public:
    struct Config {
        std::array<float,4u> anchorAndLength{};
        std::array<float,3u> bodyLocalSite{};
        std::uint32_t globalBodyIndex=MR_INVALID_INDEX;
    };
private:
    __strong id<MTLDevice> device_=nil;
    __strong id<MTLComputePipelineState> pipeline_=nil;
    metalrobo::MetalNumiHumanSupportGeometryProgram base_{};
    MRHumanRestingRopeGPU parameters_{};
    mr_float4 swing_{};
    std::uint64_t fingerprint_=0u;
    static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
    static std::uint64_t mix(std::uint64_t h,std::uint64_t v) noexcept {
        h^=v+0x9e3779b97f4a7c15ULL+(h<<6u)+(h>>2u);return h;
    }
    template<std::size_t N>
    static std::array<float,N> parse(const char* text,const char* error) {
        require(text&&*text,error);std::array<float,N> out{};const char* p=text;
        for(std::size_t i=0;i<N;++i){char* end=nullptr;const float x=std::strtof(p,&end);
            require(end!=p&&std::isfinite(x),error);out[i]=x;
            if(i+1<N){require(*end==',',error);p=end+1;}else require(*end=='\0',error);}
        return out;
    }
    static bool encodeCallback(void* opaque,const metalrobo::MetalNumiHumanSupportGeometryPass& pass) noexcept {
        auto* self=static_cast<NumiHumanRestingRopeSupport*>(opaque);if(!self)return false;
        try{return self->encode(pass);}catch(...){return false;}
    }
    static void abortCallback(void* opaque,void* commandBuffer) noexcept {
        auto* self=static_cast<NumiHumanRestingRopeSupport*>(opaque);
        if(self&&self->base_.abort)self->base_.abort(self->base_.context,commandBuffer);
    }
    static bool sameDevice(id<MTLDevice> d,id<MTLBuffer> b) {
        return d&&b&&b.device&&d.registryID==b.device.registryID;
    }
    bool encode(const metalrobo::MetalNumiHumanSupportGeometryPass& pass) {
        constexpr std::uint64_t maxU32=std::numeric_limits<std::uint32_t>::max();
        const auto fits=[](std::uint64_t a,std::uint64_t b){return b==0||a<=std::numeric_limits<std::uint64_t>::max()/b;};
        if(!base_.valid()||pass.abiVersion!=2u||pass.structSize!=sizeof(pass)||!pass.commandBuffer||
           !pass.bodyPoses||!pass.bodyPositionLow||!pass.pointWorld||!pass.pointPositionLow||
           !pass.pointJacobians||!pass.standContacts||pass.environmentCount!=1u||
           pass.standContactCount!=MR_NUMI_HUMAN_STAND_MAX_CONTACTS||pass.bodyCount==0||pass.dofCount==0||
           pass.bodyPoseStride<pass.bodyCount||pass.pointWorldStride<pass.pointCount||
           !fits(pass.pointWorldStride,3ull*pass.dofCount)||
           pass.pointJacobianStride<pass.pointWorldStride*3ull*pass.dofCount||
           pass.articulationFirstBody!=parameters_.indices.w||pass.bodyCount!=parameters_.counts.y||
           pass.dofCount!=parameters_.counts.z||pass.pointCount>maxU32||
           pass.bodyCount>maxU32||pass.dofCount>maxU32||
           parameters_.indices.y>=pass.pointCount||parameters_.indices.z<pass.articulationFirstBody||
           parameters_.indices.z-pass.articulationFirstBody>=pass.bodyCount||
           pass.bodyJacobianPointOffset==MR_INVALID_INDEX||
           static_cast<std::uint64_t>(pass.bodyJacobianPointOffset)+4ull*pass.bodyCount>pass.pointWorldStride||
           (parameters_.indices.y>=pass.bodyJacobianPointOffset&&
            parameters_.indices.y<static_cast<std::uint64_t>(pass.bodyJacobianPointOffset)+4ull*pass.bodyCount)||
           !fits(pass.environmentCount,pass.bodyPoseStride)||!fits(pass.environmentCount,pass.pointWorldStride)||
           !fits(pass.environmentCount,pass.pointJacobianStride)||
           pass.environmentCount*pass.bodyPoseStride>pass.bodyPoseElementCount||
           pass.environmentCount*pass.pointWorldStride>pass.pointWorldElementCount||
           pass.environmentCount*pass.pointJacobianStride>pass.pointJacobianElementCount) return false;
        if(pass.bodyPoseStride>maxU32||pass.bodyJacobianPointOffset>maxU32||
           pass.pointWorldStride>maxU32||pass.pointJacobianStride>maxU32)return false;
        MRHumanRestingRopeGPU dispatchParameters=parameters_;
        dispatchParameters.strides={std::uint32_t(pass.bodyPoseStride),
            std::uint32_t(pass.bodyJacobianPointOffset),
            std::uint32_t(pass.pointWorldStride),
            std::uint32_t(pass.pointJacobianStride)};
        dispatchParameters.counts={1u,std::uint32_t(pass.bodyCount),
            pass.dofCount,1u};
        id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> bodies=(__bridge id<MTLBuffer>)pass.bodyPoses;
        id<MTLBuffer> bodyLow=(__bridge id<MTLBuffer>)pass.bodyPositionLow;
        id<MTLBuffer> points=(__bridge id<MTLBuffer>)pass.pointWorld;
        id<MTLBuffer> pointLow=(__bridge id<MTLBuffer>)pass.pointPositionLow;
        id<MTLBuffer> jacobians=(__bridge id<MTLBuffer>)pass.pointJacobians;
        id<MTLBuffer> contacts=(__bridge id<MTLBuffer>)pass.standContacts;
        constexpr auto maxLength=std::numeric_limits<NSUInteger>::max();
        if(pass.bodyPoseElementCount>maxLength/sizeof(MRArticulatedBodyPoseGPU)||
           pass.bodyPoseElementCount>maxLength/sizeof(mr_float4)||
           pass.pointWorldElementCount>maxLength/sizeof(MRArticulatedPointWorldGPU)||
           pass.pointWorldElementCount>maxLength/sizeof(mr_float4)||
           pass.pointJacobianElementCount>maxLength/sizeof(float))return false;
        if(!command||!sameDevice(device_,bodies)||!sameDevice(device_,bodyLow)||!sameDevice(device_,points)||
           !sameDevice(device_,pointLow)||!sameDevice(device_,jacobians)||!sameDevice(device_,contacts)||
           bodies.length<pass.bodyPoseElementCount*sizeof(MRArticulatedBodyPoseGPU)||
           bodyLow.length<pass.bodyPoseElementCount*sizeof(mr_float4)||
           points.length<pass.pointWorldElementCount*sizeof(MRArticulatedPointWorldGPU)||
           pointLow.length<pass.pointWorldElementCount*sizeof(mr_float4)||
           jacobians.length<pass.pointJacobianElementCount*sizeof(float)||
           contacts.length<pass.standContactCount*sizeof(MRNumiHumanStandContactGPU)) return false;
        if(!base_.encodePreDynamics(base_.context,pass))return false;
        id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];if(!enc)return false;
        enc.label=@"Numi Human unilateral rope support geometry";[enc setComputePipelineState:pipeline_];
        [enc setBytes:&dispatchParameters length:sizeof(dispatchParameters) atIndex:0u];
        [enc setBuffer:bodies offset:0u atIndex:1u];[enc setBuffer:bodyLow offset:0u atIndex:2u];
        [enc setBuffer:points offset:0u atIndex:3u];[enc setBuffer:pointLow offset:0u atIndex:4u];
        [enc setBuffer:jacobians offset:0u atIndex:5u];[enc setBuffer:contacts offset:0u atIndex:6u];
        auto motion=swing_;motion.x=float(pass.stepIndex)*.002f;
        [enc setBytes:&motion length:sizeof(motion) atIndex:7u];
        [enc dispatchThreadgroups:MTLSizeMake(1u,1u,1u) threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
        [enc endEncoding];return true;
    }
public:
    static mr_float4 swingMotion(float seconds) {
        const char* text=std::getenv("NUMI_HUMAN_ROPE_SWING");
        if(!text)return {seconds,0,1,1};
        const auto p=parse<3u>(text,"NUMI_HUMAN_ROPE_SWING must be amplitude_m,period_s,ramp_s");
        require(p[0]>=0&&p[0]<=.75f&&p[1]>=2&&p[2]>=1,"rope swing bounds invalid");
        return {seconds,p[0],p[1],p[2]};
    }
    static bool configuredFromEnvironment() {
        return std::getenv("NUMI_HUMAN_ROPE_ANCHOR")||std::getenv("NUMI_HUMAN_ROPE_SITE");
    }
    static Config readConfig() {
        Config c;c.anchorAndLength=parse<4u>(std::getenv("NUMI_HUMAN_ROPE_ANCHOR"),
            "NUMI_HUMAN_ROPE_ANCHOR must be x,y,z,length");
        c.bodyLocalSite=parse<3u>(std::getenv("NUMI_HUMAN_ROPE_SITE"),
            "NUMI_HUMAN_ROPE_SITE must be body-local x,y,z");
        require(c.anchorAndLength[3]>0,"rope length must be finite and positive");return c;
    }
    static void prepareContacts(std::vector<MRNumiHumanStandContactGPU>& contacts,
                                std::uint32_t& selectedIndex,
                                std::uint32_t bodyIndex=MR_INVALID_INDEX) {
        require(bodyIndex!=MR_INVALID_INDEX,
            "rope contact selection requires an explicitly mapped runtime body index");
        require(contacts.size()==MR_NUMI_HUMAN_STAND_MAX_CONTACTS,
            "rope mode requires the existing 32 support contacts");
        selectedIndex=MR_INVALID_INDEX;unsigned matches=0;
        for(std::uint32_t i=0;i<contacts.size();++i)if(contacts[i].bodyIndex==bodyIndex){
            if(selectedIndex==MR_INVALID_INDEX)selectedIndex=i;
            ++matches;
        }
        require(matches>=1u,"rope mode requires a support row on the selected torso body");
        for(auto& contact:contacts)contact.frictionSlopAndStabilization.w=0.0f;
        contacts[selectedIndex].frictionSlopAndStabilization.x=0.0f;
    }
    NumiHumanRestingRopeSupport(id<MTLDevice> device,id<MTLLibrary> library,
        const metalrobo::MetalNumiHumanSupportGeometryProgram& base,const Config& c,
        std::uint32_t contactIndex,std::uint32_t queryIndex,std::uint32_t firstBody,
        std::uint32_t bodyCount,std::uint32_t dofCount,const char* sourceSHA256)
        :device_(device),base_(base) {
        require(device_&&library&&base_.valid()&&contactIndex<MR_NUMI_HUMAN_STAND_MAX_CONTACTS&&
            firstBody!=MR_INVALID_INDEX&&bodyCount>0&&dofCount>0&&c.globalBodyIndex>=firstBody&&
            c.globalBodyIndex-firstBody<bodyCount,"rope support owner configuration is invalid");
        swing_=swingMotion(0);
        parameters_.indices={contactIndex,queryIndex,c.globalBodyIndex,firstBody};
        parameters_.counts={1u,bodyCount,dofCount,1u};
        parameters_.anchorAndLength={c.anchorAndLength[0],c.anchorAndLength[1],
            c.anchorAndLength[2],c.anchorAndLength[3]};
        parameters_.bodyLocalSite={c.bodyLocalSite[0],c.bodyLocalSite[1],c.bodyLocalSite[2],0};
        require(sourceSHA256&&std::strlen(sourceSHA256)==64u,
            "rope source SHA-256 pin is absent");
        fingerprint_=mix(0x6e756d692d726f70ULL,base_.fingerprint);
        for(const auto* c=sourceSHA256;*c;++c)fingerprint_=mix(fingerprint_,static_cast<unsigned char>(*c));
        fingerprint_=mix(fingerprint_,contactIndex);fingerprint_=mix(fingerprint_,queryIndex);
        fingerprint_=mix(fingerprint_,c.globalBodyIndex);fingerprint_=mix(fingerprint_,firstBody);
        for(float x:c.anchorAndLength)fingerprint_=mix(fingerprint_,std::bit_cast<std::uint32_t>(x));
        for(float x:c.bodyLocalSite)fingerprint_=mix(fingerprint_,std::bit_cast<std::uint32_t>(x));
        for(float x:{swing_.y,swing_.z,swing_.w})fingerprint_=mix(fingerprint_,std::bit_cast<std::uint32_t>(x));
        if(fingerprint_==0)fingerprint_=1;
        NSError* error=nil;id<MTLFunction> function=[library newFunctionWithName:@"nm_human_resting_rope_support"];
        require(function!=nil,"rope mode requires the pinned source-built Metal library");
        pipeline_=[device_ newComputePipelineStateWithFunction:function error:&error];
        require(pipeline_!=nil,error.localizedDescription.UTF8String?
            error.localizedDescription.UTF8String:"rope support pipeline creation failed");
    }
    metalrobo::MetalNumiHumanSupportGeometryProgram program() {
        return {.context=this,.encodePreDynamics=&encodeCallback,.abort=&abortCallback,
            .fingerprint=fingerprint_,.usePerContactSupportPlanes=true};
    }
};
