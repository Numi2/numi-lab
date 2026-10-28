#pragma once

// Narrow bridge from existing combined TetGen node rows and registered O/I/D
// sidecars into the child mass-mapping bundle. Source ancestry remains separate
// from current-parent registration.
#include "NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

enum class PhysicalSkinRegisteredVolumeNodeRole : std::uint8_t {
    interior=0, outerO=1, innerI=2, deepD=3
};

struct PhysicalSkinRegisteredVolumeNodeRow {
    std::uint32_t volumeNodeRow=NM_INVALID_INDEX;
    PhysicalSkinRegisteredVolumeNodeRole role=PhysicalSkinRegisteredVolumeNodeRole::interior;
    std::uint32_t surfaceVertexRow=NM_INVALID_INDEX;
    V3 currentWorldPoint{};
    std::uint32_t donorBody=NM_INVALID_INDEX;
    std::uint32_t donorBinding=NM_INVALID_INDEX;
    V3 donorLocalPoint{};
    std::uint32_t incidentMaterialMask=0u;
    std::uint32_t geometricDonorBody=NM_INVALID_INDEX;
    std::uint32_t anatomicalRegion=NM_INVALID_INDEX;
    std::string sourceMappingKind;

    // Source ancestry is typed by domain and is not a parent-tet ID or mass donor.
    std::string sourceDomain,frameMethod;
    std::uint32_t sourceVertex=NM_INVALID_INDEX,sourceFace=NM_INVALID_INDEX;
    std::array<double,3> sourceBarycentric{};
    V3 sourceResidualWorldM{};

    // Current-frame parent registration. The reference residual is computed
    // at the same world target from this exact tet and barycentric tuple.
    std::string parentRegistrationKind;
    std::uint32_t parentTet=NM_INVALID_INDEX,parentTieCount=0;
    std::array<double,4> parentBarycentric{};
    V3 parentCurrentResidualWorldM{};
    double parentRegistrationDistanceM=0.0;
};

struct PhysicalSkinStableDeepAttachmentRequest {
    std::uint64_t stableAttachmentID=0;
    std::uint32_t donorBody=NM_INVALID_INDEX;
    std::uint32_t stableSourceVertex=NM_INVALID_INDEX;
    std::string sourceDomain;
};
struct PhysicalSkinStableDeepAttachmentMapRow {
    std::uint64_t stableAttachmentID=0;
    std::uint32_t donorBody=NM_INVALID_INDEX;
    std::uint32_t stableSourceVertex=NM_INVALID_INDEX;
    std::uint32_t deepSurfaceVertexRow=NM_INVALID_INDEX;
    std::uint32_t volumeNodeRow=NM_INVALID_INDEX;
};
struct PhysicalSkinRegisteredBodyPose {
    std::uint32_t bodyIndex=NM_INVALID_INDEX;
    std::uint32_t binding=NM_INVALID_INDEX;
    bool available=false;
    V3 currentOrigin{},restOrigin{};
    std::array<double,9> currentLocalToWorld{},restLocalToWorld{};
};
struct PhysicalSkinRegisteredVolumeImportReport {
    std::size_t volumeNodeCount=0;
    std::array<std::size_t,3> mappedBoundaryNodes{};
    std::size_t registeredParentRows=0,inferredCurrentParentRows=0,sourceIdentityRows=0;
    bool oneSourceDomain=false,canonicalDetailedSourceMapPopulated=false;
    double maximumCurrentParentReconstructionErrorM=0.0;
    double maximumReferenceParentReconstructionErrorM=0.0;
    double maximumReferenceResidualDifferenceM=0.0;
};
struct PhysicalSkinRegisteredVolumeImportSidecars {
    // Retain exact source semantics even when rows span multiple source domains.
    std::vector<PhysicalSkinNodeSourceRow> sourceAncestryRows;
    std::vector<PhysicalSkinParentRegistrationRow> currentParentRows;
    std::vector<std::string> sourceDomains;
    std::vector<PhysicalSkinStableDeepAttachmentMapRow> deepAttachments;
};

namespace physical_skin_registered_volume_import_detail {

inline bool finitePoint(const V3& p) {
    return std::all_of(p.begin(),p.end(),[](double x){return std::isfinite(x);});
}
inline bool finiteBary(const std::array<double,4>& p) {
    return std::all_of(p.begin(),p.end(),[](double x){return std::isfinite(x);});
}
inline bool closePoint(const V3& a,const V3& b,double tolerance) {
    return norm(sub(a,b))<=tolerance;
}
inline V3 baryPoint(const std::vector<V3>& points,const std::array<std::uint32_t,4>& tet,
                    const std::array<double,4>& bary) {
    V3 out{};
    for(unsigned c=0;c<4;++c)for(unsigned a=0;a<3;++a)out[a]+=bary[c]*points[tet[c]][a];
    return out;
}
inline double signedSixVolume(const std::vector<V3>& points,const std::array<std::uint32_t,4>& tet) {
    return dot(sub(points[tet[1]],points[tet[0]]),
               cross(sub(points[tet[2]],points[tet[0]]),sub(points[tet[3]],points[tet[0]])));
}
inline unsigned roleIndex(PhysicalSkinRegisteredVolumeNodeRole role) {
    switch(role){
        case PhysicalSkinRegisteredVolumeNodeRole::outerO:return 0u;
        case PhysicalSkinRegisteredVolumeNodeRole::innerI:return 1u;
        case PhysicalSkinRegisteredVolumeNodeRole::deepD:return 2u;
        case PhysicalSkinRegisteredVolumeNodeRole::interior:return 3u;
    }
    return 4u;
}
inline bool validSourceIdentity(const PhysicalSkinRegisteredVolumeNodeRow& row,std::string& error) {
    if(!finitePoint(row.sourceResidualWorldM)||
       !std::all_of(row.sourceBarycentric.begin(),row.sourceBarycentric.end(),
                    [](double value){return std::isfinite(value);})){
        error="volume node source barycentric or residual is nonfinite";return false;
    }
    const bool direct=row.sourceVertex!=NM_INVALID_INDEX;
    const bool embedded=row.sourceFace!=NM_INVALID_INDEX;
    if(row.sourceDomain=="derived_volume_interior_no_surface_ancestry"){
        if(row.role!=PhysicalSkinRegisteredVolumeNodeRole::interior||direct||embedded||
           row.sourceMappingKind!="none"||row.frameMethod!="derived_full_all_bodies_geometric_donor_pose"||
           row.sourceBarycentric!=std::array<double,3>{}||row.sourceResidualWorldM!=V3{}){
            error="derived interior source absence has an invalid role, identity, or explicit method";return false;
        }
        return true;
    }
    if(row.sourceDomain.empty()){
        error="volume node lacks a typed source domain or explicit no-surface-ancestry state";return false;
    }
    if(direct==embedded){error="volume node needs exactly one direct source vertex or embedded source face";return false;}
    if((direct&&row.sourceMappingKind!="vertex")||(embedded&&row.sourceMappingKind!="face")){
        error="volume node source mapping kind differs from its explicit vertex/face identity";return false;
    }
    if(embedded){
        double sum=0.0;
        for(double x:row.sourceBarycentric){
            if(!std::isfinite(x)||x< -1.0e-8||x>1.0+1.0e-8){
                error="volume node source-face barycentric is outside its declared source face";return false;
            }
            sum+=x;
        }
        if(std::abs(sum-1.0)>1.0e-8){error="volume node source-face barycentric does not sum to one";return false;}
    }
    if(row.frameMethod.empty()){error="volume node source/frame method is missing";return false;}
    return true;
}
inline bool validParentRegistration(const PhysicalSkinRegisteredVolumeNodeRow& row,
        const PhysicalSkinParentMassPacket& parent,const double maximumDistance,
        const V3& referenceTarget,V3& referenceResidual,double& currentError,
        double& referenceError,std::string& error) {
    if(row.parentRegistrationKind!="source_parent_tet_reference_registration"&&
       row.parentRegistrationKind!="inferred_current_parent_tet_same_material_donor_region"){
        error="volume node has an absent or unsupported current-parent registration method";return false;
    }
    if(row.parentTet>=parent.tetrahedra.size()||
       parent.materialIndices.size()!=parent.tetrahedra.size()||
       parent.donorBodyIndices.size()!=parent.tetrahedra.size()||
       parent.anatomicalRegionIndices.size()!=parent.tetrahedra.size()){
        error="volume node current-parent registration has an invalid parent label domain";return false;
    }
    if(parent.donorBodyIndices[row.parentTet]!=row.geometricDonorBody||
       parent.anatomicalRegionIndices[row.parentTet]!=row.anatomicalRegion){
        error="volume node current-parent cell donor/region differs from its declared semantic tuple";return false;
    }
    if(!finiteBary(row.parentBarycentric)||!finitePoint(row.parentCurrentResidualWorldM)||
       !std::isfinite(row.parentRegistrationDistanceM)||row.parentRegistrationDistanceM<0.0||
       row.parentRegistrationDistanceM>maximumDistance){
        error="volume node current-parent registration has an invalid tet, barycentric, residual, or bound";return false;
    }
    double sum=0.0;
    for(double b:row.parentBarycentric){
        if(b< -1.0e-8||b>1.0+1.0e-8){error="volume node current-parent barycentric is outside its tet";return false;}
        sum+=b;
    }
    if(std::abs(sum-1.0)>1.0e-8||
       std::abs(norm(row.parentCurrentResidualWorldM)-row.parentRegistrationDistanceM)>2.0e-10){
        error="volume node current-parent residual does not match its recorded distance or barycentric sum";return false;
    }
    const auto tet=parent.tetrahedra[row.parentTet];
    for(const auto node:tet)if(node>=parent.currentWorldPoints.size()||
                                  node>=parent.referenceWorldPoints.size()){
        error="volume node current-parent tet references a missing parent point";return false;
    }
    const double currentSixVolume=signedSixVolume(parent.currentWorldPoints,tet);
    const double referenceSixVolume=signedSixVolume(parent.referenceWorldPoints,tet);
    if(!std::isfinite(currentSixVolume)||!std::isfinite(referenceSixVolume)||
       currentSixVolume<=1.0e-24||referenceSixVolume<=1.0e-24){
        error="volume node parent current/reference tet is nonpositive, singular, or inverted";return false;
    }
    const auto currentBase=baryPoint(parent.currentWorldPoints,tet,row.parentBarycentric);
    currentError=norm(sub(add(currentBase,row.parentCurrentResidualWorldM),row.currentWorldPoint));
    if(!std::isfinite(currentError)||currentError>2.0e-8){
        error="volume node current-parent registration does not reconstruct its current world point";return false;
    }

    // Current and child TetGen reference rows use the same initialization-world
    // target. Preserve beta, then move the anchor from parent-current to
    // parent-reference: r_ref = r_cur + sum(beta * (parent_current-parent_ref)).
    referenceResidual=row.parentCurrentResidualWorldM;
    for(unsigned c=0;c<4;++c)for(unsigned a=0;a<3;++a)
        referenceResidual[a]+=row.parentBarycentric[c]*
            (parent.currentWorldPoints[tet[c]][a]-parent.referenceWorldPoints[tet[c]][a]);
    if(!finitePoint(referenceResidual)){
        error="same-target reference parent residual is nonfinite";return false;
    }
    const auto referenceBase=baryPoint(parent.referenceWorldPoints,tet,row.parentBarycentric);
    referenceError=norm(sub(add(referenceBase,referenceResidual),referenceTarget));
    if(!std::isfinite(referenceError)||referenceError>1.0e-8){
        error="same-target converted parent residual does not reconstruct the TetGen reference point";return false;
    }
    return true;
}
inline bool compareSource(const PhysicalSkinRegisteredVolumeNodeRow& row,
                          const PhysicalSkinNodeSourceRow& source,std::string& error) {
    const std::string expectedMapping=source.vertex>=0?"vertex":(source.face>=0?"face":"none");
    if(row.sourceMappingKind!=expectedMapping||row.sourceDomain!=source.domain||row.frameMethod!=source.frameMethod||
       row.sourceResidualWorldM!=source.residual||
       (source.vertex<0?row.sourceVertex!=NM_INVALID_INDEX:
          row.sourceVertex!=static_cast<std::uint32_t>(source.vertex))||
       (source.face<0?row.sourceFace!=NM_INVALID_INDEX:
          row.sourceFace!=static_cast<std::uint32_t>(source.face))||
       row.sourceBarycentric!=source.bary){
        error="volume-node source ancestry differs from its exact registered O/I/D sidecar row";return false;
    }
    return true;
}
} // namespace physical_skin_registered_volume_import_detail

[[nodiscard]] inline bool mapPhysicalSkinStableDeepAttachments(
        const std::span<const PhysicalSkinStableDeepAttachmentRequest> requests,
        const RegisteredSurface& deepSurface,
        const std::vector<PhysicalSkinNodeSourceRow>& deepSourceRows,
        const numi::human::SkinBoundaryNodeMap& boundaryMap,
        std::vector<PhysicalSkinStableDeepAttachmentMapRow>& output,
        std::string& error) {
    using namespace physical_skin_registered_volume_import_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(deepSurface.worldPoints.empty()||deepSurface.donorBodyIndices.size()!=deepSurface.worldPoints.size()||
       deepSurface.sourceVertexIndices.size()!=deepSurface.worldPoints.size()||
       deepSourceRows.size()!=deepSurface.worldPoints.size()||
       boundaryMap.fatDeep.size()!=deepSurface.worldPoints.size())
        return fail("deep attachment remap lacks a complete exact D source and boundary table");
    std::string mapError;
    if(!numi::human::validateSkinBoundaryNodeMap(boundaryMap,
            {boundaryMap.outer.size(),boundaryMap.skinInner.size(),boundaryMap.fatDeep.size()},mapError))
        return fail("deep attachment boundary map is invalid: "+mapError);
    std::set<std::uint64_t> stableIDs;
    std::set<std::pair<std::uint32_t,std::uint32_t>> stableSourceKeys;
    std::set<std::uint32_t> usedVolumeNodes;
    std::vector<PhysicalSkinStableDeepAttachmentMapRow> next;
    next.reserve(requests.size());
    for(const auto& request:requests){
        if(request.stableAttachmentID==0u||request.donorBody==NM_INVALID_INDEX||
           request.stableSourceVertex==NM_INVALID_INDEX||request.sourceDomain.empty()||
           !stableIDs.insert(request.stableAttachmentID).second||
           !stableSourceKeys.emplace(request.donorBody,request.stableSourceVertex).second)
            return fail("deep attachment request has a missing or duplicate stable identity");
        std::uint32_t match=NM_INVALID_INDEX;std::size_t matches=0u;
        for(std::size_t row=0;row<deepSurface.worldPoints.size();++row)
            if(deepSurface.donorBodyIndices[row]==request.donorBody&&
               deepSurface.sourceVertexIndices[row]==request.stableSourceVertex&&
               deepSourceRows[row].vertex>=0&&
               static_cast<std::uint32_t>(deepSourceRows[row].vertex)==request.stableSourceVertex&&
               deepSourceRows[row].face<0&&deepSourceRows[row].domain==request.sourceDomain){
                match=static_cast<std::uint32_t>(row);++matches;
            }
        if(matches!=1u)
            return fail("deep attachment stable source identity maps to "+std::to_string(matches)+
                        " exact D rows; exactly one is required");
        const auto volume=boundaryMap.fatDeep[match];
        if(volume>=boundaryMap.volumeNodeCount||!usedVolumeNodes.insert(volume).second)
            return fail("deep attachment rows collapse to a duplicate or invalid child volume node");
        next.push_back({request.stableAttachmentID,request.donorBody,request.stableSourceVertex,
                        match,volume});
    }
    output=std::move(next);error.clear();return true;
}

[[nodiscard]] inline bool compilePhysicalSkinRegisteredVolumeImport(
        PhysicalSkinChildMassMappingBundle& bundle,
        const PhysicalSkinRegisteredSurfaceCSVSet& registered,
        const std::span<const PhysicalSkinRegisteredVolumeNodeRow> nodeRows,
        const std::span<const PhysicalSkinStableDeepAttachmentRequest> deepAttachments,
        const std::span<const PhysicalSkinRegisteredBodyPose> allBodyCurrentPoses,
        const std::size_t sourceBodyCount,const double maximumParentRegistrationDistanceM,
        const std::string_view canonicalDetailedSourceDomain,
        const std::size_t detailedSourceVertexCount,const std::size_t detailedSourceFaceCount,
        PhysicalSkinRegisteredVolumeImportSidecars& sidecars,
        PhysicalSkinRegisteredVolumeImportReport& report,std::string& error) {
    using namespace physical_skin_registered_volume_import_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t nodeCount=bundle.referenceWorldPoints.size();
    if(nodeCount<4u||nodeRows.size()!=nodeCount||nodeCount>std::numeric_limits<std::uint32_t>::max()||
       bundle.childTetrahedra.empty()||bundle.childMaterialIndices.size()!=bundle.childTetrahedra.size()||
       bundle.childDonorBodyIndices.size()!=bundle.childTetrahedra.size()||
       bundle.childAnatomicalRegionIndices.size()!=bundle.childTetrahedra.size()||
       bundle.childStableSourceFaceIdentities.size()!=bundle.childTetrahedra.size()||
       bundle.parent.currentWorldPoints.size()!=bundle.parent.referenceWorldPoints.size()||
       bundle.parent.tetrahedra.empty()||sourceBodyCount==0u||
       registered.framesByDonor.size()!=sourceBodyCount||allBodyCurrentPoses.size()!=sourceBodyCount||
       (!bundle.currentWorldPoints.empty()&&bundle.currentWorldPoints!=bundle.referenceWorldPoints)||
       !std::isfinite(maximumParentRegistrationDistanceM)||maximumParentRegistrationDistanceM<=0.0)
        return fail("registered volume import requires complete TetGen cells/labels, same-target current/reference nodes, parent geometry, donor frames, and node rows");
    std::set<std::uint32_t> registeredBindings;
    for(std::size_t body=0;body<sourceBodyCount;++body){
        const auto& pose=allBodyCurrentPoses[body];
        if(!pose.available||pose.bodyIndex!=body||!finitePoint(pose.currentOrigin)||
           !finitePoint(pose.restOrigin)||
           !physical_skin_registered_surface_csv_detail::rotation(pose.currentLocalToWorld)||
           !physical_skin_registered_surface_csv_detail::rotation(pose.restLocalToWorld))
            return fail("full native all-bodies current/rest pose table is incomplete or invalid");
        const auto& registeredFrame=registered.framesByDonor[body];
        if(registeredFrame.available){
            if(pose.binding==NM_INVALID_INDEX||!registeredBindings.insert(pose.binding).second||
               pose.binding!=registeredFrame.binding||
               registeredFrame.currentOrigin!=pose.currentOrigin||
               registeredFrame.currentR!=pose.currentLocalToWorld||
               registeredFrame.restOrigin!=pose.restOrigin||
               registeredFrame.restR!=pose.restLocalToWorld)
                return fail("registered donor binding current/rest frame differs from the pinned full native body pose");
        }else if(pose.binding!=NM_INVALID_INDEX){
            return fail("unregistered all-bodies pose carries a fabricated registered skin binding");
        }
    }
    for(unsigned layer=0;layer<3u;++layer){
        const auto& surface=registered.surfaces[layer];
        if(surface.worldPoints.empty()||surface.donorLocalPoints.size()!=surface.worldPoints.size()||
           surface.donorBodyIndices.size()!=surface.worldPoints.size()||
           surface.worldToDonorLocalLinear.size()!=surface.worldPoints.size()||
           registered.nodeSource[layer].size()!=surface.worldPoints.size()||
           registered.nodeParent[layer].size()!=surface.worldPoints.size()||surface.faces.empty())
            return fail("registered O/I/D surface or its exact node sidecars are incomplete");
    }
    PhysicalSkinChildMassMappingBundle nextBundle=bundle;
    PhysicalSkinRegisteredVolumeImportSidecars nextSidecars;
    PhysicalSkinRegisteredVolumeImportReport measured;
    nextBundle.outerSkin=registered.surfaces[0];
    nextBundle.independentSkinInner=registered.surfaces[1];
    nextBundle.independentFatDeep=registered.surfaces[2];
    nextBundle.boundaryMap.volumeNodeCount=static_cast<std::uint32_t>(nodeCount);
    nextBundle.boundaryMap.outer.assign(registered.surfaces[0].worldPoints.size(),NM_INVALID_INDEX);
    nextBundle.boundaryMap.skinInner.assign(registered.surfaces[1].worldPoints.size(),NM_INVALID_INDEX);
    nextBundle.boundaryMap.fatDeep.assign(registered.surfaces[2].worldPoints.size(),NM_INVALID_INDEX);
    nextBundle.volumeNodeSourceVertexIndices.clear();
    nextBundle.volumeNodeSourceFaceIndices.clear();
    nextBundle.volumeNodeSourceBarycentric.clear();
    nextBundle.detailedSourceVertexCount=0u;
    nextBundle.detailedSourceFaceCount=0u;
    nextBundle.currentWorldPoints.resize(nodeCount);
    nextBundle.volumeNodeDonorBodyIndices.resize(nodeCount);
    nextBundle.volumeNodeDonorLocalPoints.resize(nodeCount);
    nextBundle.volumeNodeDonorOriginWorld.resize(nodeCount);
    nextBundle.volumeNodeDonorLocalToWorldLinear.resize(nodeCount);
    nextBundle.volumeNodeParentTetrahedra.resize(nodeCount);
    nextBundle.volumeNodeParentBarycentric.resize(nodeCount);
    nextBundle.volumeNodeParentResidualWorldM.resize(nodeCount);
    nextSidecars.sourceAncestryRows.resize(nodeCount);
    nextSidecars.currentParentRows.resize(nodeCount);
    nextSidecars.sourceDomains.resize(nodeCount);
    std::vector<std::uint8_t> seen(nodeCount,0u);
    std::vector<std::vector<std::uint32_t>> incidentCells(nodeCount);
    for(std::size_t cell=0;cell<bundle.childTetrahedra.size();++cell){
        const auto tet=bundle.childTetrahedra[cell];
        if(bundle.childMaterialIndices[cell]!=15u&&bundle.childMaterialIndices[cell]!=16u)
            return fail("child incident material is not skin 15 or fat 16");
        const auto donor=bundle.childDonorBodyIndices[cell];
        const auto region=bundle.childAnatomicalRegionIndices[cell];
        if(donor>=sourceBodyCount||region==NM_INVALID_INDEX||
           bundle.childStableSourceFaceIdentities[cell]==NM_INVALID_INDEX)
            return fail("child incident semantic tuple has an invalid donor, region, or stable source face");
        std::set<std::uint32_t> cellNodes;
        for(const auto node:tet){
            if(node>=nodeCount||!cellNodes.insert(node).second)
                return fail("child tetrahedron has an invalid or repeated joint node row");
            incidentCells[node].push_back(static_cast<std::uint32_t>(cell));
        }
    }
    std::vector<std::uint32_t> directSource(nodeCount,NM_INVALID_INDEX),faceSource(nodeCount,NM_INVALID_INDEX);
    std::vector<std::array<double,3>> sourceBary(nodeCount);
    std::set<std::string> nonemptySourceDomains;
    std::size_t sourceIdentityRows=0u,inferredRows=0u;
    double maxCurrentError=0.0,maxReferenceError=0.0,maxResidualDifference=0.0;
    for(std::size_t inputRow=0;inputRow<nodeRows.size();++inputRow){
        const auto& row=nodeRows[inputRow];
        const std::size_t node=row.volumeNodeRow;
        if(node>=nodeCount||seen[node]||node!=inputRow)
            return fail("combined TetGen node-map has a duplicate, missing, or out-of-order joint node row");
        seen[node]=1u;
        if(!finitePoint(row.currentWorldPoint)||!finitePoint(row.donorLocalPoint)||
           row.donorBody>=sourceBodyCount||!allBodyCurrentPoses[row.donorBody].available||
           row.donorBinding!=allBodyCurrentPoses[row.donorBody].binding||
           row.geometricDonorBody>=sourceBodyCount||row.anatomicalRegion==NM_INVALID_INDEX)
            return fail("combined TetGen node row has invalid frame donor/binding, semantic donor, region, or all-bodies pose");
        if(!closePoint(row.currentWorldPoint,bundle.referenceWorldPoints[node],2.0e-8))
            return fail("current registration and TetGen reference node are not the same declared initialization-world target");
        std::string sourceError;
        if(!validSourceIdentity(row,sourceError))return fail(sourceError);
        if(row.sourceMappingKind=="vertex"||row.sourceMappingKind=="face"){
            nonemptySourceDomains.insert(row.sourceDomain);++sourceIdentityRows;
        }
        const auto& frame=allBodyCurrentPoses[row.donorBody];
        V3 fromFrame=frame.currentOrigin;
        for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)
            fromFrame[a]+=frame.currentLocalToWorld[3*a+b]*row.donorLocalPoint[b];
        if(!closePoint(fromFrame,row.currentWorldPoint,1.0e-7))
            return fail("combined TetGen node donor-local point does not reconstruct its current-world position");
        const auto layer=roleIndex(row.role);
        if(layer<3u){
            const auto& surface=registered.surfaces[layer];
            if(row.surfaceVertexRow>=surface.worldPoints.size())
                return fail("registered O/I/D TetGen row references an invalid surface vertex row");
            auto& mapped=layer==0u?nextBundle.boundaryMap.outer:
                         layer==1u?nextBundle.boundaryMap.skinInner:nextBundle.boundaryMap.fatDeep;
            if(mapped[row.surfaceVertexRow]!=NM_INVALID_INDEX)
                return fail("registered O/I/D source row is mapped to multiple TetGen volume nodes");
            const auto& sidecarFrame=registered.framesByDonor[row.donorBody];
            const auto& worldToLocal=surface.worldToDonorLocalLinear[row.surfaceVertexRow];
            std::array<double,9> expectedWorldToLocal{
                frame.currentLocalToWorld[0],frame.currentLocalToWorld[3],frame.currentLocalToWorld[6],
                frame.currentLocalToWorld[1],frame.currentLocalToWorld[4],frame.currentLocalToWorld[7],
                frame.currentLocalToWorld[2],frame.currentLocalToWorld[5],frame.currentLocalToWorld[8]};
            if(!sidecarFrame.available||
               !closePoint(row.currentWorldPoint,surface.worldPoints[row.surfaceVertexRow],2.0e-8)||
               row.donorBody!=surface.donorBodyIndices[row.surfaceVertexRow]||
               !closePoint(row.donorLocalPoint,surface.donorLocalPoints[row.surfaceVertexRow],2.0e-8)||
               worldToLocal!=expectedWorldToLocal||
               !compareSource(row,registered.nodeSource[layer][row.surfaceVertexRow],sourceError))
                return fail(sourceError.empty()?"registered O/I/D TetGen node differs from its exact surface row":sourceError);
            const auto& parentRow=registered.nodeParent[layer][row.surfaceVertexRow];
            const bool inferred=row.parentRegistrationKind==
                "inferred_current_parent_tet_same_material_donor_region";
            if(!parentRow.available||parentRow.tet!=row.parentTet||parentRow.tieCount!=row.parentTieCount||
               parentRow.bary!=row.parentBarycentric||parentRow.residual!=row.parentCurrentResidualWorldM||
               parentRow.distance!=row.parentRegistrationDistanceM||
               parentRow.inferredCurrentParent!=inferred)
                return fail("registered O/I/D parent-current row differs from the exact sidecar registration");
            mapped[row.surfaceVertexRow]=static_cast<std::uint32_t>(node);
            ++measured.mappedBoundaryNodes[layer];
        }else if(layer!=3u||row.surfaceVertexRow!=NM_INVALID_INDEX){
            return fail("interior TetGen node unexpectedly claims a registered surface row");
        }
        if(row.parentTet>=bundle.parent.tetrahedra.size()||
           row.parentTet>=bundle.parent.materialIndices.size())
            return fail("node parent registration references a missing parent tet or label row");
        if(incidentCells[node].empty())
            return fail("combined TetGen node has no incident child cell for its declared semantics");
        std::uint32_t actualMaterialMask=0u;
        bool selectedSemanticIncident=false;
        for(const auto cell:incidentCells[node]){
            const auto material=bundle.childMaterialIndices[cell];
            actualMaterialMask|=(1u<<material);
            if(material==bundle.parent.materialIndices[row.parentTet]&&
               bundle.childDonorBodyIndices[cell]==row.geometricDonorBody&&
               bundle.childAnatomicalRegionIndices[cell]==row.anatomicalRegion)
                selectedSemanticIncident=true;
        }
        if(actualMaterialMask!=row.incidentMaterialMask)
            return fail("node incident material mask differs from the actual child cell labels");
        if(!selectedSemanticIncident)
            return fail("node parent material/donor/region tuple is not present on an incident child cell");
        V3 referenceResidual{};double currentError=0.0,referenceError=0.0;
        if(!validParentRegistration(row,bundle.parent,maximumParentRegistrationDistanceM,
                bundle.referenceWorldPoints[node],referenceResidual,currentError,referenceError,sourceError))
            return fail(sourceError);
        maxCurrentError=std::max(maxCurrentError,currentError);
        maxReferenceError=std::max(maxReferenceError,referenceError);
        maxResidualDifference=std::max(maxResidualDifference,
            norm(sub(referenceResidual,row.parentCurrentResidualWorldM)));
        nextBundle.currentWorldPoints[node]=row.currentWorldPoint;
        nextBundle.volumeNodeDonorBodyIndices[node]=row.donorBody;
        nextBundle.volumeNodeDonorLocalPoints[node]=row.donorLocalPoint;
        nextBundle.volumeNodeDonorOriginWorld[node]=frame.currentOrigin;
        nextBundle.volumeNodeDonorLocalToWorldLinear[node]=frame.currentLocalToWorld;
        nextBundle.volumeNodeParentTetrahedra[node]=row.parentTet;
        nextBundle.volumeNodeParentBarycentric[node]=row.parentBarycentric;
        nextBundle.volumeNodeParentResidualWorldM[node]=referenceResidual;
        PhysicalSkinNodeSourceRow source;
        source.vertex=row.sourceVertex==NM_INVALID_INDEX?-1:row.sourceVertex;
        source.face=row.sourceFace==NM_INVALID_INDEX?-1:row.sourceFace;
        source.bary=row.sourceBarycentric;source.residual=row.sourceResidualWorldM;
        source.domain=row.sourceDomain;source.frameMethod=row.frameMethod;
        nextSidecars.sourceAncestryRows[node]=std::move(source);
        PhysicalSkinParentRegistrationRow parent;
        parent.available=true;parent.tet=row.parentTet;parent.tieCount=row.parentTieCount;
        parent.bary=row.parentBarycentric;parent.residual=row.parentCurrentResidualWorldM;
        parent.distance=row.parentRegistrationDistanceM;
        parent.inferredCurrentParent=
            row.parentRegistrationKind=="inferred_current_parent_tet_same_material_donor_region";
        nextSidecars.currentParentRows[node]=parent;
        nextSidecars.sourceDomains[node]=row.sourceDomain;
        ++measured.registeredParentRows;
        if(parent.inferredCurrentParent)++inferredRows;
        directSource[node]=row.sourceVertex;faceSource[node]=row.sourceFace;
        sourceBary[node]=row.sourceBarycentric;
    }
    if(std::find(seen.begin(),seen.end(),0u)!=seen.end())
        return fail("combined TetGen node-map omits one or more joint volume node rows");
    if(std::any_of(nextBundle.boundaryMap.outer.begin(),nextBundle.boundaryMap.outer.end(),
                   [](auto n){return n==NM_INVALID_INDEX;})||
       std::any_of(nextBundle.boundaryMap.skinInner.begin(),nextBundle.boundaryMap.skinInner.end(),
                   [](auto n){return n==NM_INVALID_INDEX;})||
       std::any_of(nextBundle.boundaryMap.fatDeep.begin(),nextBundle.boundaryMap.fatDeep.end(),
                   [](auto n){return n==NM_INVALID_INDEX;}))
        return fail("combined TetGen node-map does not bijectively cover all O/I/D surface vertices");
    std::string mapError;
    if(!numi::human::validateSkinBoundaryNodeMap(nextBundle.boundaryMap,
            {registered.surfaces[0].worldPoints.size(),registered.surfaces[1].worldPoints.size(),
             registered.surfaces[2].worldPoints.size()},mapError))
        return fail("combined TetGen O/I/D boundary maps are invalid: "+mapError);
    if(nonemptySourceDomains.size()==1u)measured.oneSourceDomain=true;
    bool canonicalSourceMap=!canonicalDetailedSourceDomain.empty()&&measured.oneSourceDomain;
    for(const auto& domain:nonemptySourceDomains)
        if(domain!=canonicalDetailedSourceDomain)canonicalSourceMap=false;
    if(canonicalSourceMap){
        std::vector<std::uint32_t> vertexRows(nodeCount,NM_INVALID_INDEX),faceRows(nodeCount,NM_INVALID_INDEX);
        for(std::size_t node=0;node<nodeCount;++node){
            if(directSource[node]!=NM_INVALID_INDEX){
                if(directSource[node]>=detailedSourceVertexCount)
                    return fail("registered volume direct source identity exceeds the declared canonical detail domain");
                vertexRows[node]=directSource[node];
            }
            if(faceSource[node]!=NM_INVALID_INDEX){
                if(faceSource[node]>=detailedSourceFaceCount)
                    return fail("registered volume source face exceeds the declared canonical detail domain");
                faceRows[node]=faceSource[node];
            }
        }
        nextBundle.detailedSourceVertexCount=detailedSourceVertexCount;
        nextBundle.detailedSourceFaceCount=detailedSourceFaceCount;
        nextBundle.volumeNodeSourceVertexIndices=std::move(vertexRows);
        nextBundle.volumeNodeSourceFaceIndices=std::move(faceRows);
        nextBundle.volumeNodeSourceBarycentric=std::move(sourceBary);
        measured.canonicalDetailedSourceMapPopulated=true;
    }
    nextBundle.retainedOriginalNodeCount=0u;
    nextBundle.registeredBodyLocalToWorldLinear.resize(sourceBodyCount);
    nextBundle.registeredBodyOriginWorld.resize(sourceBodyCount);
    for(std::size_t donor=0;donor<sourceBodyCount;++donor){
        const auto& frame=allBodyCurrentPoses[donor];
        nextBundle.registeredBodyLocalToWorldLinear[donor]=frame.currentLocalToWorld;
        nextBundle.registeredBodyOriginWorld[donor]=frame.currentOrigin;
    }
    if(!mapPhysicalSkinStableDeepAttachments(deepAttachments,nextBundle.independentFatDeep,
            registered.nodeSource[2],nextBundle.boundaryMap,nextSidecars.deepAttachments,error))
        return false;
    measured.volumeNodeCount=nodeCount;measured.registeredParentRows=nodeCount;
    measured.inferredCurrentParentRows=inferredRows;measured.sourceIdentityRows=sourceIdentityRows;
    measured.maximumCurrentParentReconstructionErrorM=maxCurrentError;
    measured.maximumReferenceParentReconstructionErrorM=maxReferenceError;
    measured.maximumReferenceResidualDifferenceM=maxResidualDifference;
    bundle=std::move(nextBundle);sidecars=std::move(nextSidecars);report=measured;
    error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue
