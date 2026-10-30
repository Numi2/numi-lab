#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo::human::canonical_deep_attachment_remap {

struct SourceLineage {
    std::uint64_t stableAttachmentID=0u;
    std::uint32_t donorBody=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t derivedSourceVertex=std::numeric_limits<std::uint32_t>::max();
    std::string sourceDomain;
    std::uint32_t surfaceVertexRow=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t volumeNodeRow=std::numeric_limits<std::uint32_t>::max();
    std::int64_t surfaceSourceVertex=-1,surfaceSourceFace=-1;
    std::string surfaceSourceDomain;
    std::int64_t volumeSourceVertex=-1,volumeSourceFace=-1;
    std::string volumeSourceDomain;
    std::uint32_t surfaceParentTet=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t volumeParentTet=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t originalParentVolumeNode=std::numeric_limits<std::uint32_t>::max();
    std::array<std::uint32_t,4> sourceParentTetNodes{};
    std::array<std::uint32_t,4> volumeParentTetNodes{};
    std::uint32_t surfaceDonor=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t frameOwnerDonor=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t massOwnerDonor=std::numeric_limits<std::uint32_t>::max();
    // Kept distinct from frameOwnerDonor: it is the typed source geometry label.
    std::uint32_t sourceParentGeometricDonor=std::numeric_limits<std::uint32_t>::max();
    std::uint32_t sourceParentRegion=std::numeric_limits<std::uint32_t>::max();
    double donorFull86Weight=0.0;
    double incidentAltitudeM=0.0;
};

template<class SiteRange,class RequestRange>
bool authenticateCanonicalRequests(
        const SiteRange& canonicalSites,
        const RequestRange& requests,
        const std::string_view expectedCurrentDomain,
        std::vector<const typename SiteRange::value_type*>& siteByRequest,
        std::string& error) {
    using Site=typename SiteRange::value_type;
    siteByRequest.clear();
    if(canonicalSites.size()!=14u||requests.size()!=14u||expectedCurrentDomain.empty()) {
        error="canonical attachment/request inventory must contain exactly 14 rows and a named current domain";
        return false;
    }
    std::vector<const Site*> canonicalByID;
    canonicalByID.reserve(canonicalSites.size());
    for(const auto& site:canonicalSites) {
        if(site.sourceVertex==0u||site.donorBody==std::numeric_limits<std::uint32_t>::max()||
           site.reducedVertex==std::numeric_limits<std::uint32_t>::max()) {
            error="immutable canonical attachment table contains an invalid identity";
            return false;
        }
        if(std::any_of(canonicalByID.begin(),canonicalByID.end(),[&](const Site* prior){
               return prior->sourceVertex==site.sourceVertex;
           })) {
            error="immutable canonical attachment table repeats a stable ID";
            return false;
        }
        canonicalByID.push_back(&site);
    }
    std::set<std::uint64_t> seenIDs;
    std::set<std::pair<std::uint32_t,std::uint32_t>> seenCurrentSourceRows;
    siteByRequest.reserve(requests.size());
    for(const auto& request:requests) {
        if(request.stableAttachmentID==0u||
           request.donorBody==std::numeric_limits<std::uint32_t>::max()||
           request.stableSourceVertex==std::numeric_limits<std::uint32_t>::max()||
           request.sourceDomain!=expectedCurrentDomain||
           !seenIDs.insert(request.stableAttachmentID).second||
           !seenCurrentSourceRows.emplace(request.donorBody,request.stableSourceVertex).second) {
            error="current attachment request has a duplicate, invalid, or unpinned source identity";
            return false;
        }
        const auto site=std::find_if(canonicalByID.begin(),canonicalByID.end(),[&](const Site* value){
            return value->sourceVertex==request.stableAttachmentID;
        });
        if(site==canonicalByID.end()||(*site)->donorBody!=request.donorBody) {
            error="current attachment request does not match an immutable canonical ID/body pair";
            return false;
        }
        siteByRequest.push_back(*site);
    }
    if(seenIDs.size()!=canonicalByID.size()||
       std::any_of(canonicalByID.begin(),canonicalByID.end(),[&](const Site* site){
           return !seenIDs.contains(site->sourceVertex);
       })) {
        error="current attachment requests do not cover the immutable canonical ID set exactly";
        return false;
    }
    error.clear();
    return true;
}

inline bool validateSourceLineage(
        const auto& site,
        const auto& request,
        const SourceLineage& row,
        const double minimumDonorWeight,
        const double minimumIncidentAltitudeM,
        std::string& error) {
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    if(row.stableAttachmentID!=site.sourceVertex||
       row.stableAttachmentID!=request.stableAttachmentID||
       row.donorBody!=site.donorBody||row.donorBody!=request.donorBody||
       row.derivedSourceVertex!=request.stableSourceVertex||
       row.sourceDomain!=request.sourceDomain||row.sourceDomain.empty()||
       row.surfaceSourceVertex<0||
       static_cast<std::uint64_t>(row.surfaceSourceVertex)!=row.derivedSourceVertex||
       row.volumeSourceVertex<0||
       static_cast<std::uint64_t>(row.volumeSourceVertex)!=row.derivedSourceVertex||
       row.surfaceSourceFace>=0||row.volumeSourceFace>=0||
       row.surfaceSourceDomain!=row.sourceDomain||row.volumeSourceDomain!=row.sourceDomain||
       row.surfaceVertexRow==invalid||row.volumeNodeRow==invalid||
       row.surfaceParentTet==invalid||row.volumeParentTet==invalid||
       row.originalParentVolumeNode==invalid||
       std::find(row.sourceParentTetNodes.begin(),row.sourceParentTetNodes.end(),
                 row.originalParentVolumeNode)==row.sourceParentTetNodes.end()||
       std::find(row.volumeParentTetNodes.begin(),row.volumeParentTetNodes.end(),
                 row.originalParentVolumeNode)==row.volumeParentTetNodes.end()||
       row.surfaceDonor!=site.donorBody||row.frameOwnerDonor!=site.donorBody||
       row.massOwnerDonor!=site.donorBody||
       row.sourceParentGeometricDonor==invalid||row.sourceParentRegion==invalid||
       !std::isfinite(row.donorFull86Weight)||row.donorFull86Weight<minimumDonorWeight||
       !std::isfinite(row.incidentAltitudeM)||row.incidentAltitudeM<minimumIncidentAltitudeM) {
        error="canonical attachment lacks exact source-row, parent-star, donor, or local-quality proof";
        return false;
    }
    error.clear();
    return true;
}

template<class Anchor>
bool rebindHardAnchorToCurrentRest(
        const Anchor& canonicalSource,
        const std::uint32_t expectedBody,
        const std::uint32_t expectedFlags,
        const std::array<float,3>& currentDonorLocalPoint,
        Anchor& output) {
    if(canonicalSource.bodyIndex!=expectedBody||canonicalSource.flags!=expectedFlags||
       !std::all_of(currentDonorLocalPoint.begin(),currentDonorLocalPoint.end(),
                    [](float v){return std::isfinite(v);})) return false;
    output=canonicalSource;
    output.localPoint={currentDonorLocalPoint[0],currentDonorLocalPoint[1],
                       currentDonorLocalPoint[2],0.0f};
    return output.bodyIndex==canonicalSource.bodyIndex&&
           output.flags==canonicalSource.flags&&output.localPoint.w==0.0f;
}

template<class Sample>
bool rebindElasticSampleToCurrentRest(
        const Sample& canonicalSource,
        const std::uint32_t newSlaveNode,
        const std::uint32_t expectedBody,
        const std::uint32_t expectedFlags,
        const std::array<float,3>& currentDonorLocalPoint,
        const std::array<float,3>& currentDonorLocalUnitNormal,
        Sample& output) {
    const auto finite3=[](const auto& v) {
        return std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);});
    };
    if(canonicalSource.bodyIndex!=expectedBody||canonicalSource.flags!=expectedFlags||
       canonicalSource.reserved0!=0u||canonicalSource.bodyLocalPointAndArea.w<=0.0f||
       !std::isfinite(canonicalSource.bodyLocalPointAndArea.w)||
       canonicalSource.bodyLocalNormalAndReferenceSeparation.w!=0.0f||
       !std::isfinite(canonicalSource.stiffnessAndNormalStrainPerPressure.x)||
       !(canonicalSource.stiffnessAndNormalStrainPerPressure.x>0.0f)||
       !std::isfinite(canonicalSource.stiffnessAndNormalStrainPerPressure.y)||
       !(canonicalSource.stiffnessAndNormalStrainPerPressure.y>0.0f)||
       canonicalSource.stiffnessAndNormalStrainPerPressure.z!=0.0f||
       canonicalSource.stiffnessAndNormalStrainPerPressure.w!=0.0f||
       !finite3(currentDonorLocalPoint)||!finite3(currentDonorLocalUnitNormal)) return false;
    double normal2=0.0;
    for(const float value:currentDonorLocalUnitNormal)normal2+=double(value)*value;
    if(!std::isfinite(normal2)||std::abs(normal2-1.0)>1.0e-5)return false;
    output=canonicalSource;
    output.slaveNode=newSlaveNode;
    output.bodyIndex=expectedBody;
    output.bodyLocalPointAndArea.x=currentDonorLocalPoint[0];
    output.bodyLocalPointAndArea.y=currentDonorLocalPoint[1];
    output.bodyLocalPointAndArea.z=currentDonorLocalPoint[2];
    output.bodyLocalNormalAndReferenceSeparation.x=currentDonorLocalUnitNormal[0];
    output.bodyLocalNormalAndReferenceSeparation.y=currentDonorLocalUnitNormal[1];
    output.bodyLocalNormalAndReferenceSeparation.z=currentDonorLocalUnitNormal[2];
    output.bodyLocalNormalAndReferenceSeparation.w=0.0f;
    return output.bodyLocalPointAndArea.w==canonicalSource.bodyLocalPointAndArea.w&&
           output.stiffnessAndNormalStrainPerPressure.x==
               canonicalSource.stiffnessAndNormalStrainPerPressure.x&&
           output.stiffnessAndNormalStrainPerPressure.y==
               canonicalSource.stiffnessAndNormalStrainPerPressure.y&&
           output.stiffnessAndNormalStrainPerPressure.z==
               canonicalSource.stiffnessAndNormalStrainPerPressure.z&&
           output.stiffnessAndNormalStrainPerPressure.w==
               canonicalSource.stiffnessAndNormalStrainPerPressure.w;
}

template<class Sample>
std::pair<double,std::array<double,3>> elasticResponseForDisplacement(
        const Sample& sample,const std::array<double,3>& displacement) {
    const double stiffness=double(sample.stiffnessAndNormalStrainPerPressure.x)*
                           double(sample.bodyLocalPointAndArea.w);
    const double norm2=displacement[0]*displacement[0]+
                       displacement[1]*displacement[1]+
                       displacement[2]*displacement[2];
    return {0.5*stiffness*norm2,{-stiffness*displacement[0],
                                  -stiffness*displacement[1],
                                  -stiffness*displacement[2]}};
}

} // namespace metalrobo::human::canonical_deep_attachment_remap
