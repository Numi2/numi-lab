#pragma once
// App-facing adapter for the existing BodyParts3D .nhskin payload. The caller
// supplies the exact renderer skinning functions and donor-local frame maps;
// the original LoadedSkin stays the authority for all binding weights.
#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanWholeBodySkinReduction.hpp"
#include "NumiHumanSkinRigidPlacement.hpp"
#include "numi/matter/numi_human_shared.h"
#include <string_view>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

namespace metalrobo::human::coarse_tissue {
struct RegisteredSkinMechanicalDeltaAudit {
    std::size_t vertexCount=0u;
    std::size_t movedVertexCount=0u;
    double maximumWorldDeltaM=0.0;
    double maximumHeadHandDeltaM=0.0;
    double maximumOtherBodyDeltaM=0.0;
    double maximumWorldToDonorResidualM=0.0;
};

// Preparation-time evaluation of one coarse node's existing detailed-surface
// ancestry. This is registration math only: it neither updates the simulator
// nor transfers contact forces. The output donor is the largest of the exact
// barycentrically interpolated 86-body weights, with its confidence reported.
template<class LoadedSkinT,class PositionFn>
inline bool evaluateRegisteredSkinSourceEmbedding(const LoadedSkinT& skin,
        std::uint32_t sourceFaceIndex,const std::array<double,3>& beta,
        PositionFn&& positionAt,V3& sourceWorldPoint,
        std::span<double> fullBodyWeights,std::uint32_t& dominantBodyIndex,
        double& dominantBindingWeight,std::string& error,
        double weightNormalizationTolerance=1.0e-4) {
    const auto fail=[&](const char* message){error=message;return false;};
    constexpr std::size_t kRequiredFullBodyBindings=86u;
    if(skin.vertices.empty()||skin.indices.empty()||skin.indices.size()%3u!=0u||
       skin.bindings.size()!=kRequiredFullBodyBindings||
       skin.fullWeights.size()!=skin.vertices.size()*kRequiredFullBodyBindings)
        return fail("source embedding requires indexed geometry and the exact full 86-body weight table");
    if(fullBodyWeights.size()!=kRequiredFullBodyBindings)
        return fail("source embedding output does not have capacity for all 86 body weights");
    if(!std::isfinite(weightNormalizationTolerance)||weightNormalizationTolerance<0.0)
        return fail("source embedding weight tolerance is invalid");
    if(!validateBarycentricCoordinates(beta,error))return false;
    const std::size_t faceOffset=std::size_t(sourceFaceIndex)*3u;
    if(faceOffset+2u>=skin.indices.size())
        return fail("source embedding face index is outside the detailed source topology");
    std::array<std::uint32_t,3> vertices{
        skin.indices[faceOffset],skin.indices[faceOffset+1u],skin.indices[faceOffset+2u]};
    if(vertices[0]>=skin.vertices.size()||vertices[1]>=skin.vertices.size()||vertices[2]>=skin.vertices.size()||
       vertices[0]==vertices[1]||vertices[1]==vertices[2]||vertices[2]==vertices[0])
        return fail("source embedding triangle has invalid or repeated source vertex indices");
    std::array<V3,3> points{};std::array<double,kRequiredFullBodyBindings> interpolatedWeights{};
    for(unsigned corner=0;corner<3u;++corner){
        points[corner]=positionAt(vertices[corner],skin.vertices[vertices[corner]]);
        for(double value:points[corner])if(!std::isfinite(value))
            return fail("source embedding triangle contains a nonfinite registered point");
        double rowSum=0.0;
        const auto rowOffset=std::size_t(vertices[corner])*kRequiredFullBodyBindings;
        for(std::size_t binding=0;binding<kRequiredFullBodyBindings;++binding){
            const float weight=skin.fullWeights[rowOffset+binding];
            if(!std::isfinite(weight)||weight<0.0f)
                return fail("source embedding triangle has an invalid full-body skin weight");
            rowSum+=double(weight);interpolatedWeights[binding]+=beta[corner]*double(weight);
        }
        if(!std::isfinite(rowSum)||std::abs(rowSum-1.0)>weightNormalizationTolerance)
            return fail("source embedding triangle full-body weights do not normalize");
    }
    V3 point{};for(unsigned corner=0;corner<3u;++corner)
        point=add(point,mul(points[corner],beta[corner]));
    double interpolatedSum=0.0;std::size_t dominantBinding=0u;
    double negativeBarycentricBound=0.0,positiveBarycentricBound=0.0;
    for(double coefficient:beta){
        if(coefficient<0.0)negativeBarycentricBound+=coefficient;
        else positiveBarycentricBound+=coefficient;
    }
    const double weightRoundoff=32.0*std::numeric_limits<double>::epsilon();
    for(std::size_t binding=0;binding<kRequiredFullBodyBindings;++binding){
        const double weight=interpolatedWeights[binding];
        // The registered barycentric input admits tiny negative roundoff
        // (e.g. -1e-13); retain the exact affine result without clamping it.
        // Bound any such output by the negative/positive coefficient sums.
        const double lowerBound=negativeBarycentricBound-weightRoundoff;
        const double upperBound=positiveBarycentricBound*(1.0+weightNormalizationTolerance)+weightRoundoff;
        if(!std::isfinite(weight)||weight<lowerBound||weight>upperBound)
            return fail("interpolated source skin weight is outside its barycentric input bound");
        interpolatedSum+=weight;
        if(weight>interpolatedWeights[dominantBinding])dominantBinding=binding;
    }
    if(!std::isfinite(interpolatedSum)||std::abs(interpolatedSum-1.0)>weightNormalizationTolerance)
        return fail("interpolated source skin weights do not normalize");
    const auto owner=skin.bindings[dominantBinding].bodyIndex;
    if(owner==NM_INVALID_INDEX)return fail("dominant interpolated skin binding has no Human body owner");
    for(std::size_t binding=0;binding<kRequiredFullBodyBindings;++binding)
        fullBodyWeights[binding]=interpolatedWeights[binding];
    sourceWorldPoint=point;dominantBodyIndex=owner;
    dominantBindingWeight=interpolatedWeights[dominantBinding];
    return true;
}

// Applies the cumulative source-referenced repair only after the original
// registered donor frames have been validated by makeRegisteredSkinMechanicalRegion.
// Positions, normals, donor identities, faces, and ancestry are otherwise owned
// by the original registered payload and reduced topology header.
inline bool applyRegisteredSkinMechanicalDeltas(
        RegisteredSurface& region,
        std::span<const std::uint32_t> expectedSourceVertices,
        std::span<const std::array<double,3>> worldDeltas,
        std::span<const std::array<double,3>> donorLocalDeltas,
        std::span<const std::uint8_t> capClasses,
        double headHandCapM,double otherBodyCapM,
        RegisteredSkinMechanicalDeltaAudit& audit,std::string& error,
        double transformToleranceM=2.0e-6) {
    const auto fail=[&](const char* message){error=message;return false;};
    const std::size_t n=region.worldPoints.size();
    if(n==0u||expectedSourceVertices.size()!=n||worldDeltas.size()!=n||
       donorLocalDeltas.size()!=n||capClasses.size()!=n||
       region.sourceVertexIndices.size()!=n||region.donorLocalPoints.size()!=n||
       region.donorBodyIndices.size()!=n||region.worldToDonorLocalLinear.size()!=n)
        return fail("registered mechanical repair arrays do not exactly cover the reduced source surface");
    if(!std::isfinite(headHandCapM)||headHandCapM<=0.0||
       !std::isfinite(otherBodyCapM)||otherBodyCapM<=0.0||
       !std::isfinite(transformToleranceM)||transformToleranceM<=0.0)
        return fail("registered mechanical repair caps or transform tolerance are invalid");
    RegisteredSkinMechanicalDeltaAudit next{};
    next.vertexCount=n;
    for(std::size_t i=0;i<n;++i) {
        if(region.sourceVertexIndices[i]!=expectedSourceVertices[i]||
           (capClasses[i]!=0u&&capClasses[i]!=1u))
            return fail("registered mechanical repair ancestry or cap class differs from reduced source rows");
        const auto& dw=worldDeltas[i];
        const auto& dl=donorLocalDeltas[i];
        for(unsigned c=0;c<3u;++c)
            if(!std::isfinite(dw[c])||!std::isfinite(dl[c])||
               !std::isfinite(region.worldPoints[i][c])||
               !std::isfinite(region.donorLocalPoints[i][c]))
                return fail("registered mechanical repair contains a nonfinite source or delta coordinate");
        const double worldNorm=std::sqrt(double(dw[0])*dw[0]+double(dw[1])*dw[1]+double(dw[2])*dw[2]);
        const double cap=capClasses[i]==1u?headHandCapM:otherBodyCapM;
        if(!std::isfinite(worldNorm)||worldNorm>cap+1.0e-7)
            return fail("registered mechanical repair exceeds its declared cumulative regional displacement cap");
        const auto& m=region.worldToDonorLocalLinear[i];
        for(double x:m)if(!std::isfinite(x))
            return fail("registered mechanical repair donor transform is nonfinite");
        double residualSquared=0.0;
        for(unsigned row=0;row<3u;++row) {
            const double mapped=double(m[3u*row])*dw[0]+
                double(m[3u*row+1u])*dw[1]+double(m[3u*row+2u])*dw[2];
            const double difference=mapped-double(dl[row]);
            residualSquared+=difference*difference;
        }
        const double residual=std::sqrt(residualSquared);
        if(!std::isfinite(residual)||residual>transformToleranceM)
            return fail("registered mechanical repair donor-local delta disagrees with its pinned world-to-donor map");
        next.maximumWorldDeltaM=std::max(next.maximumWorldDeltaM,worldNorm);
        if(capClasses[i]==1u)next.maximumHeadHandDeltaM=std::max(next.maximumHeadHandDeltaM,worldNorm);
        else next.maximumOtherBodyDeltaM=std::max(next.maximumOtherBodyDeltaM,worldNorm);
        next.maximumWorldToDonorResidualM=std::max(next.maximumWorldToDonorResidualM,residual);
        if(worldNorm>0.0)++next.movedVertexCount;
        for(unsigned c=0;c<3u;++c)
            if(!std::isfinite(region.worldPoints[i][c]+double(dw[c]))||
               !std::isfinite(region.donorLocalPoints[i][c]+double(dl[c])))
                return fail("registered mechanical repair would produce a nonfinite adjusted source point");
    }
    // All checks complete before mutation so a rejected candidate leaves the
    // registered source surface unchanged.
    for(std::size_t i=0;i<n;++i)
        for(unsigned c=0;c<3u;++c) {
            region.worldPoints[i][c]+=double(worldDeltas[i][c]);
            region.donorLocalPoints[i][c]+=double(donorLocalDeltas[i][c]);
        }
    audit=next;
    return true;
}

template<class LoadedSkinT,class PositionFn,class NormalFn,class DonorLocalFn,class OwnerFn,class WorldToDonorFn>
bool makeRegisteredSkinMechanicalRegion(const LoadedSkinT& skin,
        std::string_view provider,std::string_view sourceMember,std::string_view sourcePayloadSHA256,
        PositionFn&& positionAt,NormalFn&& normalAt,DonorLocalFn&& donorLocalAt,
        OwnerFn&& bodyOwnerAt,WorldToDonorFn&& worldToDonorAt,
        std::uint32_t seedFace,std::uint32_t maxFaces,RegisteredSurface& region,std::string& error,
        RegisteredSkinRigidPlacement* placementOut=nullptr){
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
    // Initialization-only source-coordinate observation; no simulation update.
    if(const char* path=std::getenv("NUMI_HUMAN_SKIN_REFERENCE_DUMP")){
        std::ofstream dump(path);if(!dump){error="cannot open skin reference observation";return false;}
        dump<<std::setprecision(17)<<"vertex,donor,wx,wy,wz,nx,ny,nz,lx,ly,lz,m00,m01,m02,m10,m11,m12,m20,m21,m22\n";
        for(std::size_t i=0;i<source.worldPoints.size();i++){
            dump<<i<<','<<source.donorBodyIndices[i];
            for(const auto& p:{source.worldPoints[i],source.outwardWorldNormals[i],source.donorLocalPoints[i]})for(double x:p)dump<<','<<x;
            for(double x:source.worldToDonorLocalLinear[i])dump<<','<<x;dump<<'\n';
        }
        if(!dump){error="skin reference observation write failed";return false;}
    }
    // Parent maps let the existing adapter consult the original full-weight
    // 86-body skin payload for precisely the retained coarse mechanical nodes.
    if(maxFaces!=0u)return makeConnectedSurfacePatch(source,seedFace,maxFaces,region,error);
    // Zero requests the separately audited whole-body reduction encoded by the pinned arrays.
    // All coordinates, donor frames, and full source weights remain owned by
    // the pinned NHSKIN payload; only its reduced connectivity is admitted.
    if(sourcePayloadSHA256!=kWholeBodySkinSourcePayloadSHA256||
       skin.vertices.size()!=61950u||skin.indices.size()/3u!=108987u||
       source.worldPoints.size()!=61950u||source.faces.size()!=108987u||
       kWholeBodySkinSourceVertices.size()!=kWholeBodySkinSourceVertexCount||
       kWholeBodySkinFaces.size()!=kWholeBodySkinFaceCount||
       kWholeBodySkinSourceFaces.size()!=kWholeBodySkinFaceCount||
       kWholeBodySkinDonorFrames.size()!=kWholeBodySkinSourceVertexCount)
        return fail("whole-body mechanical reduction requires its exact registered source and ancestry arrays");
    if(skin.bindings.size()!=86u||skin.fullWeights.size()!=skin.vertices.size()*skin.bindings.size())
        return fail("whole-body skin requires the retained full 86-body source weights and binding table");
    auto close=[](double a,double b,double tolerance){return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance;};
    region=source;region.name=source.name+" whole-body reduced skin-fat mechanical shell";
    region.worldPoints.clear();region.donorLocalPoints.clear();region.outwardWorldNormals.clear();
    region.donorBodyIndices.clear();region.worldToDonorLocalLinear.clear();
    std::vector<std::uint32_t> sortedSourceVertices(kWholeBodySkinSourceVertices.begin(),kWholeBodySkinSourceVertices.end());
    std::sort(sortedSourceVertices.begin(),sortedSourceVertices.end());
    if(std::adjacent_find(sortedSourceVertices.begin(),sortedSourceVertices.end())!=sortedSourceVertices.end())
        return fail("whole-body reduced source vertex ancestry contains duplicates");
    RegisteredSkinRigidPlacement placement;
    for(std::size_t i=0;i<kWholeBodySkinSourceVertices.size();++i){
        const auto v=kWholeBodySkinSourceVertices[i];const auto& expected=kWholeBodySkinDonorFrames[i];
        if(v>=source.worldPoints.size()||expected.donorBodyIndex!=source.donorBodyIndices[v])
            return fail("whole-body reduced vertex has invalid or mismatched donor ancestry");
        double weightSum=0.0;
        for(std::size_t b=0;b<skin.bindings.size();b++){
            const float weight=skin.fullWeights[std::size_t(v)*skin.bindings.size()+b];
            if(!std::isfinite(weight)||weight<0.0f)return fail("whole-body reduced vertex has an invalid full-body skin weight");
            weightSum+=weight;
        }
        if(!std::isfinite(weightSum)||std::abs(weightSum-1.0)>1.0e-4)
            return fail("whole-body reduced vertex full-body skin weights do not normalize");
        const auto& world=source.worldPoints[v];const auto& local=source.donorLocalPoints[v];
        const auto& normal=source.outwardWorldNormals[v];const auto& frame=source.worldToDonorLocalLinear[v];
        if(placementOut) {
            const auto asDouble=[]<class T,std::size_t N>(const std::array<T,N>& a){
                std::array<double,N> out{};std::copy(a.begin(),a.end(),out.begin());return out;
            };
            const auto referencePoint=asDouble(expected.registeredWorldPoint);
            const auto referenceLocal=asDouble(expected.donorLocalPoint);
            const auto referenceNormal=asDouble(expected.registeredWorldNormal);
            const auto referenceFrame=asDouble(expected.worldToDonorLocalLinear);
            if(i==0u && !deriveRegisteredSkinRigidPlacement(
                referencePoint,referenceFrame,
                world,frame,placement,error)) return false;
            if(!validateRegisteredSkinPlacementVertex(
                referencePoint,referenceLocal,
                referenceNormal,referenceFrame,
                world,local,normal,frame,placement,error)) return false;
            // Keep exact admitted reference coordinates while applying frozen repairs
            // and constructing layers. The caller places all layers together afterward.
            region.worldPoints.push_back(referencePoint);
            region.donorLocalPoints.push_back(referenceLocal);
            region.outwardWorldNormals.push_back(referenceNormal);
            region.worldToDonorLocalLinear.push_back(referenceFrame);
        } else {
            for(unsigned c=0;c<3;c++){
                if(!close(world[c],expected.registeredWorldPoint[c],2.0e-5)||
                   !close(local[c],expected.donorLocalPoint[c],2.0e-5)||
                   !close(normal[c],expected.registeredWorldNormal[c],2.0e-4))
                    return fail("whole-body reduced renderer point differs from its pinned registered donor frame");
            }
            for(unsigned c=0;c<9;c++)if(!close(frame[c],expected.worldToDonorLocalLinear[c],2.0e-5))
                return fail("whole-body reduced normal frame differs from its pinned registered donor frame");
            region.worldPoints.push_back(world);region.donorLocalPoints.push_back(local);
            region.outwardWorldNormals.push_back(normal);
            region.worldToDonorLocalLinear.push_back(frame);
        }
        region.donorBodyIndices.push_back(source.donorBodyIndices[v]);
    }
    region.faces.assign(kWholeBodySkinFaces.begin(),kWholeBodySkinFaces.end());
    region.sourceVertexIndices.assign(kWholeBodySkinSourceVertices.begin(),kWholeBodySkinSourceVertices.end());
    region.sourceFaceIndices.assign(kWholeBodySkinSourceFaces.begin(),kWholeBodySkinSourceFaces.end());
    for(std::size_t i=0;i<region.faces.size();++i){
        const auto f=region.faces[i];for(auto v:f)if(v>=region.worldPoints.size())return fail("whole-body reduced face index is out of range");
        if(region.sourceFaceIndices[i]>=source.faces.size())return fail("whole-body reduced face parent is outside the source payload");
    }
    if(placementOut) *placementOut=placement;
    // The caller validates the exact original frames, applies the pinned repair,
    // then audits the repaired surface before any volume or runtime assembly.
    return true;
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
