#pragma once
// App-facing adapter for the existing BodyParts3D .nhskin payload. The caller
// supplies the exact renderer skinning functions and donor-local frame maps;
// the original LoadedSkin stays the authority for all binding weights.
#include "NumiHumanCoarseTissuePreparation.hpp"
#include "numi/matter/numi_human_shared.h"
#include <string_view>

namespace metalrobo::human::coarse_tissue {
template<class LoadedSkinT,class PositionFn,class NormalFn,class DonorLocalFn,class OwnerFn,class WorldToDonorFn>
bool makeRegisteredSkinMechanicalRegion(const LoadedSkinT& skin,
        std::string_view provider,std::string_view sourceMember,std::string_view sourcePayloadSHA256,
        PositionFn&& positionAt,NormalFn&& normalAt,DonorLocalFn&& donorLocalAt,
        OwnerFn&& bodyOwnerAt,WorldToDonorFn&& worldToDonorAt,
        std::uint32_t seedFace,std::uint32_t maxFaces,RegisteredSurface& region,std::string& error){
    auto fail=[&](const char* message){error=message;return false;};
    if(provider.empty()||sourceMember.empty()||skin.vertices.empty()||
       skin.indices.empty()||skin.indices.size()%3!=0||skin.bindings.empty()||sourcePayloadSHA256.size()!=64)
        return fail("registered skin adapter identity or payload arrays are incomplete");
    RegisteredSurface source;source.name="BodyParts3D registered skinned shell";source.provider=provider;
    source.sourceMember=sourceMember;source.sourcePayloadSHA256=sourcePayloadSHA256;
    // The rendered shell aggregates many components; registrationFingerprint below is payload provenance,
    // not an anatomical stable ID.
    source.stableIdentifier=0u;source.aggregateIdentity=true;source.sourceLayer=NM_INVALID_INDEX;source.sourceBodyIndex=NM_INVALID_INDEX;
    source.sourceMeshSHA256.reserve(64);static constexpr char hex[]="0123456789abcdef";
    for(const auto byte:skin.header.sourceSha256){source.sourceMeshSHA256.push_back(hex[(byte>>4)&15]);source.sourceMeshSHA256.push_back(hex[byte&15]);}
    source.worldPoints.reserve(skin.vertices.size());source.donorLocalPoints.reserve(skin.vertices.size());
    source.outwardWorldNormals.reserve(skin.vertices.size());source.donorBodyIndices.reserve(skin.vertices.size());
    source.worldToDonorLocalLinear.reserve(skin.vertices.size());
    for(std::uint32_t vertex=0;vertex<skin.vertices.size();vertex++){
        const auto world=positionAt(vertex,skin.vertices[vertex]);
        const auto normal=normalAt(vertex,skin.vertices[vertex]);
        const auto owner=bodyOwnerAt(vertex,skin.vertices[vertex]);
        if(owner==NM_INVALID_INDEX)return fail("skinned source vertex has no registered Human donor owner");
        const auto local=donorLocalAt(vertex,owner,world);
        const auto frame=worldToDonorAt(vertex,owner);
        for(double x:world)if(!std::isfinite(x))return fail("renderer-registered skin position is nonfinite");
        for(double x:normal)if(!std::isfinite(x))return fail("renderer-registered skin normal is nonfinite");
        for(double x:local)if(!std::isfinite(x))return fail("donor-local skin point is nonfinite");
        for(double x:frame)if(!std::isfinite(x))return fail("world-to-donor normal map is nonfinite");
        source.worldPoints.push_back(world);source.donorLocalPoints.push_back(local);
        source.outwardWorldNormals.push_back(normal);source.donorBodyIndices.push_back(owner);
        source.worldToDonorLocalLinear.push_back(frame);
    }
    source.faces.reserve(skin.indices.size()/3);
    for(std::uint32_t index=0;index<skin.indices.size();index+=3){
        const auto a=skin.indices[index],b=skin.indices[index+1],c=skin.indices[index+2];
        if(a>=skin.vertices.size()||b>=skin.vertices.size()||c>=skin.vertices.size())return fail("registered skin source index is out of range");
        source.faces.push_back({a,b,c});
    }
    // Parent maps let the existing adapter consult the original full-weight
    // 86-body skin payload for precisely the retained coarse mechanical nodes.
    return makeConnectedSurfacePatch(source,seedFace,maxFaces,region,error);
}

// Deterministic source-owner choice for the existing mass partition only.
// This does not generate an attachment or kinematic anchor; full source
// weights remain available through region.sourceVertexIndices.
template<class LoadedSkinT>
std::uint32_t dominantSkinDonorBody(const LoadedSkinT& skin,std::uint32_t vertex){
    if(vertex>=skin.vertices.size()||skin.bindings.empty())return NM_INVALID_INDEX;
    const auto& v=skin.vertices[vertex];std::size_t best=0;float bestWeight=-1.0f;
    if(!skin.fullWeights.empty()){
        const std::size_t count=skin.bindings.size();if(skin.fullWeights.size()!=skin.vertices.size()*count)return NM_INVALID_INDEX;
        for(std::size_t b=0;b<count;b++){const float weight=skin.fullWeights[std::size_t(vertex)*count+b];if(!std::isfinite(weight)||weight<0)return NM_INVALID_INDEX;if(weight>bestWeight){bestWeight=weight;best=b;}}
    }else{
        for(std::size_t influence=0;influence<4;influence++){const float weight=v.weight[influence];if(!std::isfinite(weight)||weight<0)return NM_INVALID_INDEX;if(weight>bestWeight){bestWeight=weight;best=v.bindingIndex[influence];}}
    }
    return best<skin.bindings.size()?skin.bindings[best].bodyIndex:NM_INVALID_INDEX;
}
} // namespace metalrobo::human::coarse_tissue
