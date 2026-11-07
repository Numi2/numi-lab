#include "metalrobo/CollisionVisualProxy.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace metalrobo {
namespace {
mr_float4 scaled(mr_float4 a,float scale) {return {a.x*scale,a.y*scale,a.z*scale,0};}
mr_float4 sum(mr_float4 a,mr_float4 b) {return {a.x+b.x,a.y+b.y,a.z+b.z,1};}
void face(VisualAssetPackV2& pack,mr_float4 center,mr_float4 u,mr_float4 v,mr_float4 normal) {
    const auto base=static_cast<std::uint32_t>(pack.vertices.size());
    const float length=std::sqrt(u.x*u.x+u.y*u.y+u.z*u.z);
    const mr_float4 tangent=scaled(u,1.0f/length);
    normal.w=1.0f; // Right-handed tangent frame, required by VisualAssetPackV2.
    for (const auto corner : std::array<std::array<float,2>,4>{{{-1,-1},{1,-1},{1,1},{-1,1}}}) {
        pack.vertices.push_back({sum(center,sum(scaled(u,corner[0]),scaled(v,corner[1]))),
            normal,tangent,{0,0,0,0},{1,1,1,1}});
    }
    for (const unsigned index : {0u,1u,2u,0u,2u,3u}) pack.indices.push_back(base+index);
}
}
VisualAssetPackV2 makeCollisionProxyVisualAssetPack(
    const EngineModel& model,std::span<const std::uint32_t> bodyIndices,
    std::uint64_t sourceWorldFingerprint,const std::string& id,
    mr_float4 linearColor,float planeHalfExtentMeters) {
    if (id.empty() || sourceWorldFingerprint==0u || bodyIndices.empty() ||
        !std::isfinite(planeHalfExtentMeters) || !(planeHalfExtentMeters>0) ||
        !std::isfinite(linearColor.x) || !std::isfinite(linearColor.y) || !std::isfinite(linearColor.z) ||
        linearColor.x<0 || linearColor.y<0 || linearColor.z<0 || linearColor.w!=1)
        throw std::invalid_argument("Invalid collision-proxy visual authoring inputs");
    std::unordered_set<std::uint32_t> bodies;
    for (const auto body : bodyIndices) if (body>=model.bodies.size() || !bodies.insert(body).second)
        throw std::invalid_argument("Collision-proxy body ownership is invalid or duplicated");
    VisualAssetPackV2 pack;
    pack.id=id;pack.sourceUri="native-collision://"+model.name;
    pack.sourceContentHash="world:"+std::to_string(sourceWorldFingerprint);
    pack.license="generated-from-source-mechanics";
    pack.preprocessingProvenance="native-collision-proxy/v1; exact boxes; +Y planes clipped to half extent "
        +std::to_string(planeHalfExtentMeters)+" m; no physics authority";
    MRVisualMaterialGPUV2 material{};
    material.baseColorAndOpacity=linearColor;material.surface={0.8f,0,1,1};
    material.coatingAndAlphaCutoff={0,0,1,0.5f};
    material.textureIndices0={MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX,MR_INVALID_INDEX};
    material.textureIndices1=material.textureIndices0;material.reserved=material.textureIndices0;
    material.flags={MR_VISUAL_ALPHA_OPAQUE,MR_VISUAL_MATERIAL_DOUBLE_SIDED,0,1};
    pack.materials.push_back(material);
    for (std::size_t shapeIndex=0;shapeIndex<model.shapes.size();++shapeIndex) {
        const auto& shape=model.shapes[shapeIndex];
        if (!bodies.contains(shape.bodyIndex)) continue;
        if (shape.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED)
            throw std::invalid_argument("Collision proxy refuses simulation-disabled geometry");
        if (shape.shapeType!=MR_SHAPE_BOX && shape.shapeType!=MR_SHAPE_PLANE)
            throw std::invalid_argument("Collision proxy currently admits exact boxes and bounded +Y planes only");
        const auto firstIndex=static_cast<std::uint32_t>(pack.indices.size());
        mr_float4 lower{},upper{};
        if (shape.shapeType==MR_SHAPE_BOX) {
            const auto h=shape.dimensions;
            if (!std::isfinite(h.x) || !std::isfinite(h.y) || !std::isfinite(h.z) ||
                !(h.x>0 && h.y>0 && h.z>0)) throw std::invalid_argument("Invalid proxy box extents");
            face(pack,{h.x,0,0,0},{0,h.y,0,0},{0,0,h.z,0},{1,0,0,0});
            face(pack,{-h.x,0,0,0},{0,-h.y,0,0},{0,0,h.z,0},{-1,0,0,0});
            face(pack,{0,h.y,0,0},{-h.x,0,0,0},{0,0,h.z,0},{0,1,0,0});
            face(pack,{0,-h.y,0,0},{h.x,0,0,0},{0,0,h.z,0},{0,-1,0,0});
            face(pack,{0,0,h.z,0},{h.x,0,0,0},{0,h.y,0,0},{0,0,1,0});
            face(pack,{0,0,-h.z,0},{h.x,0,0,0},{0,-h.y,0,0},{0,0,-1,0});
            lower={-h.x,-h.y,-h.z,0};upper={h.x,h.y,h.z,0};
        } else {
            const float h=planeHalfExtentMeters;
            face(pack,{0,0,0,0},{0,0,h,0},{h,0,0,0},{0,1,0,0});
            lower={-h,0,-h,0};upper={h,0,h,0};
        }
        const auto primitiveIndex=static_cast<std::uint32_t>(pack.primitives.size());
        MRVisualPrimitiveGPUV2 primitive{};
        primitive.geometry={firstIndex,static_cast<std::uint32_t>(pack.indices.size())-firstIndex,0,
                            static_cast<std::uint32_t>(pack.instances.size())};
        primitive.identity={1,1,shape.bodyIndex,static_cast<std::uint32_t>(shapeIndex)};
        primitive.boundsMinimum=lower;primitive.boundsMaximum=upper;pack.primitives.push_back(primitive);
        const bool articulated=model.bodies[shape.bodyIndex].articulationIndex!=MR_INVALID_INDEX;
        const auto binding=articulated ? MR_VISUAL_BINDING_ARTICULATED_LINK : MR_VISUAL_BINDING_RIGID_BODY;
        MRVisualInstanceGPUV2 instance{};
        instance.translationAndScale=shape.localPosition;instance.translationAndScale.w=1;
        instance.orientation=shape.localRotation;
        instance.binding={0,shape.bodyIndex,binding,MR_VISUAL_INSTANCE_CASTS_SHADOW |
            MR_VISUAL_INSTANCE_RECEIVES_SHADOW | MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR};
        instance.identity=primitive.identity;instance.geometry={primitiveIndex,1,0,0};
        const auto instanceIndex=static_cast<std::uint32_t>(pack.instances.size());
        pack.instances.push_back(instance);
        pack.symbolicBindings.push_back({model.shapeNames.at(shapeIndex),model.bodyNames.at(shape.bodyIndex),
                                        instanceIndex,shape.bodyIndex,binding});
    }
    if (pack.primitives.empty()) throw std::invalid_argument("Collision-proxy body set owns no supported shapes");
    pack.contentHash=computeVisualAssetPackContentHash(pack);
    std::string reason;
    if (!pack.valid(&reason)) throw std::invalid_argument("Collision proxy failed asset validation: "+reason);
    return pack;
}
} // namespace metalrobo
