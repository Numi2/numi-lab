#pragma once
#include <Metal/Metal.h>
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/VisualPresentation.hpp"
#include "metalrobo/engine_types.h"
#include "metalrobo/numi_human_resting_rope_visual_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

class NumiHumanRestingRopeVisual final {
    struct Vec3 { float x,y,z; };
public:
    struct Layout {
        MRHumanRestingRopeVisualGPU gpu{};
        std::uint32_t primitiveIndex=MR_INVALID_INDEX;
        std::uint32_t instanceIndex=MR_INVALID_INDEX;
        std::uint32_t materialIndex=MR_INVALID_INDEX;
        std::uint32_t dynamicVertexCount() const noexcept {
            return gpu.counts.x + gpu.counts.w;
        }
    };

    static Layout appendGeometry(
        metalrobo::VisualAssetPackV2& pack,
        const MRHumanRestingRopeGPU& rope,
        std::uint32_t semantic=52001u,
        std::uint32_t stableId=52001u,
        float ropeRadius=0.0035f,
        float harnessRadius=0.018f,
        float harnessTubeRadius=0.003f,
        float visualSlackSagLimit=0.12f,
        std::uint32_t ropeRings=32u,
        std::uint32_t ropeSides=8u,
        std::uint32_t harnessMajorSegments=20u,
        std::uint32_t harnessMinorSegments=8u) {
        require(allFinite(rope.anchorAndLength) && allFinite(rope.bodyLocalSite) &&
                rope.anchorAndLength.w > 0.0f && rope.bodyLocalSite.w == 0.0f,
                "rope visual requires finite source-bound rope parameters");
        require(ropeRings >= 2u && ropeSides >= 3u &&
                harnessMajorSegments >= 3u && harnessMinorSegments >= 3u &&
                finitePositive(ropeRadius) && finitePositive(harnessRadius) &&
                finitePositive(harnessTubeRadius) &&
                finitePositive(visualSlackSagLimit),
                "rope visual tessellation or dimensions are invalid");
        const std::uint64_t ropeCount64=std::uint64_t(ropeRings)*ropeSides;
        const std::uint64_t harnessCount64=std::uint64_t(harnessMajorSegments)*harnessMinorSegments;
        const std::uint64_t fixedEyeCount=20u*8u;
        const std::uint64_t additions=ropeCount64+harnessCount64+fixedEyeCount+4u;
        require(std::uint64_t(pack.vertices.size())+additions<=std::numeric_limits<std::uint32_t>::max(),
                "rope visual vertex indices exceed uint32");
        require(pack.primitives.size()<std::numeric_limits<std::uint32_t>::max() &&
                pack.instances.size()<std::numeric_limits<std::uint32_t>::max() &&
                pack.materials.size()<std::numeric_limits<std::uint32_t>::max(),
                "rope visual table indices exceed uint32");

        const auto firstIndex=std::uint32_t(pack.indices.size());
        const auto firstPrimitive=std::uint32_t(pack.primitives.size());
        const auto firstInstance=std::uint32_t(pack.instances.size());
        const auto materialIndex=std::uint32_t(pack.materials.size());
        const Vec3 anchor{rope.anchorAndLength.x,rope.anchorAndLength.y,rope.anchorAndLength.z};
        const float maximumLength=rope.anchorAndLength.w;

        MRVisualMaterialGPUV2 material{};
        material.baseColorAndOpacity={0.50f,0.34f,0.18f,1.0f};
        material.emissionAndStrength={0.0f,0.0f,0.0f,0.0f};
        material.surface={0.80f,0.0f,1.0f,1.0f};
        material.coatingAndAlphaCutoff={0.0f,0.25f,0.35f,0.5f};
        material.textureIndices0={MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX};
        material.textureIndices1={MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX};
        material.flags={MR_VISUAL_ALPHA_OPAQUE,MR_VISUAL_MATERIAL_DOUBLE_SIDED,0u,stableId};
        material.reserved={MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX};
        pack.materials.push_back(material);

        const auto dynamicRopeFirst=std::uint32_t(pack.vertices.size());
        for(std::uint32_t ring=0;ring<ropeRings;++ring) {
            const float t=float(ring)/float(ropeRings-1u);
            const Vec3 center{anchor.x,anchor.y,anchor.z-maximumLength*t};
            for(std::uint32_t side=0;side<ropeSides;++side) {
                const float theta=6.283185307179586f*float(side)/float(ropeSides);
                const Vec3 radial{std::sin(theta),std::cos(theta),0.0f};
                appendVertex(pack,add(center,mul(radial,ropeRadius)),radial,{0.0f,0.0f,-1.0f},
                             0.0f,float(side)/float(ropeSides));
            }
        }
        for(std::uint32_t ring=0;ring+1u<ropeRings;++ring) {
            for(std::uint32_t side=0;side<ropeSides;++side) {
                const auto a=dynamicRopeFirst+ring*ropeSides+side;
                const auto b=dynamicRopeFirst+ring*ropeSides+(side+1u)%ropeSides;
                const auto c=dynamicRopeFirst+(ring+1u)*ropeSides+side;
                const auto d=dynamicRopeFirst+(ring+1u)*ropeSides+(side+1u)%ropeSides;
                pack.indices.insert(pack.indices.end(),{a,b,c,b,d,c});
            }
        }

        const auto dynamicHarnessFirst=std::uint32_t(pack.vertices.size());
        const Vec3 placeholderEnd{anchor.x,anchor.y,anchor.z-maximumLength};
        appendTorusVertices(pack,placeholderEnd,{0.0f,0.0f,1.0f},
                            harnessRadius,harnessTubeRadius,
                            harnessMajorSegments,harnessMinorSegments);
        appendTorusIndices(pack,dynamicHarnessFirst,harnessMajorSegments,harnessMinorSegments);

        const auto staticEyeFirst=std::uint32_t(pack.vertices.size());
        constexpr float anchorEyeRadius=0.018f;
        constexpr float anchorEyeTube=0.003f;
        appendTorusVertices(pack,anchor,{0.0f,0.0f,1.0f},
                            anchorEyeRadius,anchorEyeTube,20u,8u);
        appendTorusIndices(pack,staticEyeFirst,20u,8u);

        const auto plateFirst=std::uint32_t(pack.vertices.size());
        constexpr float halfPlate=0.07f;
        const float plateZ=anchor.z+0.012f;
        appendVertex(pack,{anchor.x-halfPlate,anchor.y-halfPlate,plateZ},
                     {0,0,1},{1,0,0},0,1);
        appendVertex(pack,{anchor.x+halfPlate,anchor.y-halfPlate,plateZ},
                     {0,0,1},{1,0,0},1,1);
        appendVertex(pack,{anchor.x+halfPlate,anchor.y+halfPlate,plateZ},
                     {0,0,1},{1,0,0},1,0);
        appendVertex(pack,{anchor.x-halfPlate,anchor.y+halfPlate,plateZ},
                     {0,0,1},{1,0,0},0,0);
        pack.indices.insert(pack.indices.end(),{plateFirst,plateFirst+1u,plateFirst+2u,
                                               plateFirst,plateFirst+2u,plateFirst+3u});

        const auto indexCount=std::uint32_t(pack.indices.size())-firstIndex;
        MRVisualPrimitiveGPUV2 primitive{};
        primitive.geometry={firstIndex,indexCount,materialIndex,firstInstance};
        primitive.identity={semantic,firstInstance,MR_INVALID_INDEX,stableId};
        const float extent=maximumLength+visualSlackSagLimit+
            std::max({ropeRadius,harnessRadius+harnessTubeRadius,
                      anchorEyeRadius+anchorEyeTube,halfPlate})+0.02f;
        primitive.boundsMinimum={anchor.x-extent,anchor.y-extent,anchor.z-extent,1.0f};
        primitive.boundsMaximum={anchor.x+extent,anchor.y+extent,anchor.z+extent,1.0f};
        MRVisualInstanceGPUV2 instance{};
        instance.translationAndScale={0,0,0,1};
        instance.orientation={0,0,0,1};
        instance.binding={0,MR_INVALID_INDEX,MR_VISUAL_BINDING_WORLD,
            MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR|MR_VISUAL_INSTANCE_CASTS_SHADOW|
            MR_VISUAL_INSTANCE_RECEIVES_SHADOW};
        instance.identity=primitive.identity;
        instance.geometry={firstPrimitive,1u,0u,0u};
        pack.primitives.push_back(primitive);
        pack.instances.push_back(instance);

        Layout layout;
        layout.primitiveIndex=firstPrimitive;
        layout.instanceIndex=firstInstance;
        layout.materialIndex=materialIndex;
        layout.gpu.ranges={dynamicRopeFirst,ropeRings,ropeSides,dynamicHarnessFirst};
        layout.gpu.counts={std::uint32_t(ropeCount64),harnessMajorSegments,
                           harnessMinorSegments,std::uint32_t(harnessCount64)};
        layout.gpu.radiiAndSag={ropeRadius,harnessRadius,harnessTubeRadius,visualSlackSagLimit};
        return layout;
    }

    NumiHumanRestingRopeVisual(
        id<MTLDevice> device,
        id<MTLLibrary> library,
        MRHumanRestingRopeGPU rope,
        Layout layout)
        : device_(device), rope_(rope), layout_(layout) {
        require(device_ && library, "rope visual requires a Metal device and source library");
        id<MTLFunction> function=[library newFunctionWithName:@"nm_human_resting_rope_visual"];
        NSError* error=nil;
        pipeline_=[device_ newComputePipelineStateWithFunction:function error:&error];
        if(!function || !pipeline_) {
            const char* detail=error.localizedDescription.UTF8String;
            throw std::runtime_error(detail?detail:"rope visual Metal pipeline could not be created");
        }
        require(rope_.indices.z>=rope_.indices.w &&
                rope_.indices.z-rope_.indices.w<rope_.counts.y &&
                rope_.bodyLocalSite.w==0.0f &&
                layout_.gpu.counts.x==layout_.gpu.ranges.y*layout_.gpu.ranges.z &&
                layout_.gpu.counts.w==layout_.gpu.counts.y*layout_.gpu.counts.z &&
                layout_.gpu.ranges.w==layout_.gpu.ranges.x+layout_.gpu.counts.x,
                "rope visual owner identity or mesh layout is invalid");
    }

    bool encode(
        const metalrobo::MetalHybridMeshDeformationLease& lease,
        id<MTLBuffer> acceptedBodyStates,id<MTLBuffer> acceptedRespiration,
        std::size_t statusOffset,const mr_float4& motionParameters) const noexcept {
        try {
            if(!pipeline_||!device_||!acceptedBodyStates||!acceptedRespiration||
               acceptedRespiration.device.registryID!=device_.registryID||
               statusOffset+sizeof(mr_uint4)>acceptedRespiration.length||
               !lease.encoder||
               !lease.encoder->valid()||!lease.meshVertices)return false;
            if(acceptedBodyStates.device.registryID!=device_.registryID)return false;
            const auto bodyBytes=(std::uint64_t(rope_.indices.z)+1ull)*sizeof(MRBodyStateGPU);
            if(acceptedBodyStates.length<bodyBytes||
               std::uint64_t(layout_.gpu.ranges.x)+layout_.dynamicVertexCount()>lease.meshVertexCount||
               std::uint64_t(layout_.gpu.ranges.w)+layout_.gpu.counts.w>lease.meshVertexCount)return false;
            id<MTLBuffer> vertices=(__bridge id<MTLBuffer>)lease.meshVertices;
            if(!vertices||vertices.device.registryID!=device_.registryID||
               vertices.length<std::uint64_t(lease.meshVertexCount)*sizeof(MRVisualVertexGPUV2))return false;
            const auto& e=*lease.encoder;
            e.setPipeline(e.context,(__bridge void*)pipeline_);
            e.setBytes(e.context,&rope_,sizeof(rope_),0u);
            e.setBytes(e.context,&layout_.gpu,sizeof(layout_.gpu),1u);
            e.setBuffer(e.context,(__bridge void*)acceptedBodyStates,0u,2u);
            e.setBuffer(e.context,lease.meshVertices,0u,3u);
            e.setBuffer(e.context,(__bridge void*)acceptedRespiration,statusOffset,4u);
            e.setBytes(e.context,&motionParameters,sizeof(motionParameters),5u);
            e.dispatchThreads(e.context,layout_.dynamicVertexCount(),64u);
            return true;
        } catch(...) { return false; }
    }

    const Layout& layout() const noexcept { return layout_; }

private:
    __strong id<MTLDevice> device_=nil;
    __strong id<MTLComputePipelineState> pipeline_=nil;
    MRHumanRestingRopeGPU rope_{};
    Layout layout_{};

    static void require(bool condition,const char* message) {
        if(!condition)throw std::runtime_error(message);
    }
    static bool finitePositive(float x) noexcept {
        return std::isfinite(x)&&x>0.0f;
    }
    static bool allFinite(mr_float4 x) noexcept {
        return std::isfinite(x.x)&&std::isfinite(x.y)&&std::isfinite(x.z)&&std::isfinite(x.w);
    }
    static Vec3 add(Vec3 a,Vec3 b) noexcept { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
    static Vec3 mul(Vec3 a,float s) noexcept { return {a.x*s,a.y*s,a.z*s}; }

    static void appendVertex(
        metalrobo::VisualAssetPackV2& pack,Vec3 p,Vec3 n,Vec3 tangent,float u,float v) {
        MRVisualVertexGPUV2 vertex{};
        vertex.position={p.x,p.y,p.z,1.0f};
        vertex.normalAndTangentSign={n.x,n.y,n.z,1.0f};
        vertex.tangent={tangent.x,tangent.y,tangent.z,0.0f};
        vertex.texcoord01={u,v,0.0f,0.0f};
        vertex.color={1.0f,1.0f,1.0f,1.0f};
        pack.vertices.push_back(vertex);
    }

    static void appendTorusVertices(
        metalrobo::VisualAssetPackV2& pack,Vec3 center,Vec3 axis,
        float majorRadius,float tubeRadius,std::uint32_t majorSegments,std::uint32_t minorSegments) {
        const float axisNorm=std::sqrt(axis.x*axis.x+axis.y*axis.y+axis.z*axis.z);
        require(axisNorm>0.0f,"rope attachment ring axis is degenerate");
        axis={axis.x/axisNorm,axis.y/axisNorm,axis.z/axisNorm};
        Vec3 u{1,0,0};
        const float projection=u.x*axis.x+u.y*axis.y+u.z*axis.z;
        u={u.x-projection*axis.x,u.y-projection*axis.y,u.z-projection*axis.z};
        float uNorm=std::sqrt(u.x*u.x+u.y*u.y+u.z*u.z);
        if(uNorm<1.0e-5f) {
            u={0,1,0};
            const float p=u.x*axis.x+u.y*axis.y+u.z*axis.z;
            u={u.x-p*axis.x,u.y-p*axis.y,u.z-p*axis.z};
            uNorm=std::sqrt(u.x*u.x+u.y*u.y+u.z*u.z);
        }
        require(uNorm>0.0f,"rope attachment ring frame is degenerate");
        u={u.x/uNorm,u.y/uNorm,u.z/uNorm};
        const Vec3 v{axis.y*u.z-axis.z*u.y,
                     axis.z*u.x-axis.x*u.z,
                     axis.x*u.y-axis.y*u.x};
        for(std::uint32_t i=0;i<majorSegments;++i) {
            const float theta=6.283185307179586f*float(i)/float(majorSegments);
            const Vec3 radial{std::cos(theta)*u.x+std::sin(theta)*v.x,
                              std::cos(theta)*u.y+std::sin(theta)*v.y,
                              std::cos(theta)*u.z+std::sin(theta)*v.z};
            const Vec3 tangent{-std::sin(theta)*u.x+std::cos(theta)*v.x,
                               -std::sin(theta)*u.y+std::cos(theta)*v.y,
                               -std::sin(theta)*u.z+std::cos(theta)*v.z};
            for(std::uint32_t j=0;j<minorSegments;++j) {
                const float phi=6.283185307179586f*float(j)/float(minorSegments);
                const Vec3 normal{std::cos(phi)*radial.x+std::sin(phi)*axis.x,
                                  std::cos(phi)*radial.y+std::sin(phi)*axis.y,
                                  std::cos(phi)*radial.z+std::sin(phi)*axis.z};
                const Vec3 p=add(center,add(mul(radial,majorRadius+tubeRadius*std::cos(phi)),
                                            mul(axis,tubeRadius*std::sin(phi))));
                appendVertex(pack,p,normal,tangent,float(i)/float(majorSegments),
                             float(j)/float(minorSegments));
            }
        }
    }

    static void appendTorusIndices(
        metalrobo::VisualAssetPackV2& pack,std::uint32_t first,
        std::uint32_t majorSegments,std::uint32_t minorSegments) {
        for(std::uint32_t i=0;i<majorSegments;++i)for(std::uint32_t j=0;j<minorSegments;++j) {
            const auto a=first+i*minorSegments+j;
            const auto b=first+((i+1u)%majorSegments)*minorSegments+j;
            const auto c=first+i*minorSegments+(j+1u)%minorSegments;
            const auto d=first+((i+1u)%majorSegments)*minorSegments+(j+1u)%minorSegments;
            pack.indices.insert(pack.indices.end(),{a,b,c,b,d,c});
        }
    }
};
