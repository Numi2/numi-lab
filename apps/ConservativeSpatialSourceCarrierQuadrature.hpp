#pragma once

#include "ConservativeSpatialFractionalSemanticSupport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

// Optional 090 source-tetra carrier quadrature. The legacy entry point remains
// unchanged when this API is not selected. A nonempty row is an explicitly
// witnessed source-cell quadrature: every carrier has a pinned admissible child
// and remains under the original transport radius, while positive weights
// preserve the source cell's first moment, mass, measure, and donor split.
// Barycentric coordinates are in the ordered source parent tet node domain;
// positive weights partition, rather than replace, the original cell mass,
// source measure, and source-donor contribution rows.
struct SourceCellCarrierQuadratureInput {
    std::span<const std::uint32_t> parentOffsets; // nParent + 1; empty rows use equal offsets
    std::span<const std::uint32_t> globalSourceTetRows; // one original source330 tet row per parent input row
    std::span<const std::array<std::uint32_t,4>> sourceTetNodeRows; // parent input row -> ordered source330 node rows
    std::span<const std::array<std::array<double,3>,4>> parentReferenceTetWorldVertices;
    std::span<const double> carrierWeights;
    std::span<const std::array<double,4>> carrierBarycentric;
    std::span<const std::uint32_t> witnessChildRows;
    std::span<const double> witnessChildCentroidDistancesM;
    std::string_view sourceCarrierPacketSHA256;
    std::string_view sourceTetTableSHA256;
    std::string_view sourceNodeTableSHA256;
};

// Exact carrier-level flows remain available beside the existing aggregate
// parent/child CSR. This preserves the original source parent IDs at the public
// boundary while making every virtual carrier allocation auditable.
struct SourceCellCarrierTransportTrace {
    std::string sourceCarrierPacketSHA256;
    std::string sourceCarrierArraySHA256;
    std::string expandedSolverReceiptSHA256;
    std::vector<std::uint32_t> expandedParentToSourceParentRows;
    std::vector<std::uint32_t> expandedParentToCarrierRows; // kInvalid means unchanged centroid route
    std::vector<std::uint32_t> carrierSourceParentRows;
    std::vector<double> originalCentroidNearestAdmissibleDistanceM;
    // CSR aligned with the aggregate result's childParentRows edges.
    std::vector<std::uint32_t> childParentCarrierOffsets;
    std::vector<std::uint32_t> childParentCarrierRows; // expanded solver parent rows
    std::vector<double> childParentCarrierMassKg;
    std::vector<double> childParentCarrierMeasureM3;
    std::vector<double> childParentCarrierDistanceM;
    std::string receiptSHA256;
};

namespace source_carrier_transport_detail {

using Point = SpatialMassTransportPoint;
using TetVertices = std::array<Point,4>;
using Bary = std::array<double,4>;
constexpr std::uint32_t kInvalidCarrierRow = std::numeric_limits<std::uint32_t>::max();

inline Point barycentricPoint(const TetVertices& tet, const Bary& bary) {
    Point point{};
    for (std::size_t node=0; node<4u; ++node)
        for (std::size_t axis=0; axis<3u; ++axis)
            point[axis] += bary[node]*tet[node][axis];
    return point;
}

inline double pointDistance(const Point& a, const Point& b) {
    const double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
    return std::sqrt(dx*dx+dy*dy+dz*dz);
}

inline std::string carrierArrayDigest(const ConservativeSpatialMassTransportInput& in,
                                      const SourceCellCarrierQuadratureInput& carrier) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("numi.human.source-tetra-centroid-carrier-quadrature.arrays.v1");
    hash.string(carrier.sourceCarrierPacketSHA256);
    hash.string(carrier.sourceTetTableSHA256);
    hash.string(carrier.sourceNodeTableSHA256);
    hash.u64(carrier.parentOffsets.size());
    for (const auto value : carrier.parentOffsets) hash.u32(value);
    hash.u64(carrier.globalSourceTetRows.size());
    for (const auto value : carrier.globalSourceTetRows) hash.u32(value);
    for (const auto& tet : carrier.sourceTetNodeRows)
        for (const auto value : tet) hash.u32(value);
    for (const auto& tet : carrier.parentReferenceTetWorldVertices)
        for (const auto& point : tet)
            for (const auto value : point) hash.real(value);
    hash.u64(carrier.carrierWeights.size());
    for (const auto value : carrier.carrierWeights) hash.real(value);
    for (const auto& bary : carrier.carrierBarycentric)
        for (const auto value : bary) hash.real(value);
    hash.u64(carrier.witnessChildRows.size());
    for (const auto value : carrier.witnessChildRows) hash.u32(value);
    for (const auto value : carrier.witnessChildCentroidDistancesM) hash.real(value);
    hash.u64(in.parentMassKg.size());
    hash.string(in.parentMassPacketSHA256);
    hash.string(in.parentSourceContributionSHA256);
    hash.string(in.childMeshSHA256);
    hash.string(in.regionLabelSHA256);
    hash.string(in.coordinateFrameSHA256);
    return hash.finish();
}

inline std::string combinedSupportDigest(const ConservativeSpatialChildSemanticSupport& support,
                                         const std::string_view carrierDigest,
                                         const std::string_view expandedSupportDigest) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("conservativeSpatialMassTransport.sourceCarrierQuadratureSupport.v1");
    hash.string(fractional_semantic_transport_detail::supportDigest(support));
    hash.string(carrierDigest);
    hash.string(expandedSupportDigest);
    return hash.finish();
}

inline bool validateCarrierInput(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        std::vector<std::uint32_t>& parentGroup,
        std::vector<std::map<std::uint32_t,double>>& regionPreference,
        std::vector<std::vector<std::uint32_t>>& admissibleRegions,
        std::vector<Point>& carrierWorldPoints,
        std::vector<std::uint32_t>& carrierRows,
        std::vector<double>& strictCentroidNearest,
        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t np=in.parentMassKg.size(), nc=in.childReferenceVolumeM3.size();
    if (!fractional_semantic_transport_detail::validateSharedCapacitySupport(
            in,support,parentGroup,regionPreference,admissibleRegions,error)) return false;
    if (carrier.parentOffsets.size()!=np+1u || carrier.parentOffsets.empty() ||
        carrier.parentOffsets.front()!=0u ||
        carrier.parentOffsets.back()!=carrier.carrierWeights.size() ||
        carrier.carrierBarycentric.size()!=carrier.carrierWeights.size() ||
        carrier.witnessChildRows.size()!=carrier.carrierWeights.size() ||
        carrier.witnessChildCentroidDistancesM.size()!=carrier.carrierWeights.size() ||
        carrier.globalSourceTetRows.size()!=np || carrier.sourceTetNodeRows.size()!=np ||
        carrier.parentReferenceTetWorldVertices.size()!=np ||
        !validSHA256(carrier.sourceCarrierPacketSHA256) ||
        !validSHA256(carrier.sourceTetTableSHA256) || !validSHA256(carrier.sourceNodeTableSHA256))
        return fail("source carrier packet arrays, source geometry maps or SHA-256 pins are incomplete");
    for (std::size_t row=1u; row<carrier.parentOffsets.size(); ++row)
        if (carrier.parentOffsets[row]<carrier.parentOffsets[row-1u] ||
            carrier.parentOffsets[row]>carrier.carrierWeights.size())
            return fail("source carrier parent CSR offsets are reversed or out of range");
    std::set<std::uint32_t> uniqueTetRows;
    for (std::size_t parent=0; parent<np; ++parent) {
        if (!uniqueTetRows.insert(carrier.globalSourceTetRows[parent]).second)
            return fail("source carrier source330 tet-row map is not one-to-one");
        std::set<std::uint32_t> nodes;
        Point centroid{};
        for (std::size_t node=0; node<4u; ++node) {
            if (!nodes.insert(carrier.sourceTetNodeRows[parent][node]).second)
                return fail("source carrier source tetrahedron repeats a source node row");
            for (std::size_t axis=0; axis<3u; ++axis) {
                const double x=carrier.parentReferenceTetWorldVertices[parent][node][axis];
                if (!std::isfinite(x)) return fail("source carrier source tetrahedron has a nonfinite reference coordinate");
                centroid[axis]+=0.25*x;
            }
        }
        if (pointDistance(centroid,in.parentCentroidsWorld[parent])>1.0e-11)
            return fail("source carrier source-tet centroid does not equal the authenticated parent reference centroid");
        const auto begin=carrier.parentOffsets[parent], end=carrier.parentOffsets[parent+1u];
        if (begin==end) continue;
        if (in.parentMaterialIndices[parent]==kInvalidCarrierRow ||
            in.parentDonorBodyIndices[parent]==kInvalidCarrierRow ||
            in.parentAnatomicalRegionIndices[parent]==kInvalidCarrierRow)
            return fail("source carrier parent has invalid material/donor/region labels");
        carrierRows.push_back(static_cast<std::uint32_t>(parent));
        double weightSum=0.0;
        Bary baryMoment{};
        Point worldMoment{};
        double strictNearest=std::numeric_limits<double>::infinity();
        const auto material=in.parentMaterialIndices[parent];
        const auto region=in.parentAnatomicalRegionIndices[parent];
        for (std::size_t child=0; child<nc; ++child) {
            if (in.childMaterialIndices[child]!=material ||
                !std::binary_search(admissibleRegions[child].begin(),admissibleRegions[child].end(),region))
                continue;
            strictNearest=std::min(strictNearest,pointDistance(in.parentCentroidsWorld[parent],
                                                               in.childCentroidsWorld[child]));
        }
        strictCentroidNearest.push_back(strictNearest);
        for (std::uint32_t row=begin; row<end; ++row) {
            const double weight=carrier.carrierWeights[row];
            const auto& bary=carrier.carrierBarycentric[row];
            if (!std::isfinite(weight) || weight<=0.0 ||
                carrier.witnessChildRows[row]>=nc ||
                !std::isfinite(carrier.witnessChildCentroidDistancesM[row]))
                return fail("source carrier weight or child witness row is invalid");
            double barySum=0.0;
            for (std::size_t node=0; node<4u; ++node) {
                if (!std::isfinite(bary[node]) || bary[node]<-1.0e-13 || bary[node]>1.0+1.0e-13)
                    return fail("source carrier barycentric coordinate lies outside its source tetrahedron");
                barySum+=bary[node];
                baryMoment[node]+=weight*bary[node];
            }
            if (std::abs(barySum-1.0)>2.0e-12)
                return fail("source carrier barycentric coordinates do not sum to one");
            const auto point=barycentricPoint(carrier.parentReferenceTetWorldVertices[parent],bary);
            const auto child=carrier.witnessChildRows[row];
            if (in.childMaterialIndices[child]!=material ||
                !std::binary_search(admissibleRegions[child].begin(),admissibleRegions[child].end(),region))
                return fail("source carrier witness child has the wrong material or is not semantically admissible for the source region");
            const double distance=pointDistance(point,in.childCentroidsWorld[child]);
            const double tolerance=std::max(1.0e-12,64.0*std::numeric_limits<double>::epsilon()*
                std::max({1.0,distance,carrier.witnessChildCentroidDistancesM[row]}));
            if (std::abs(distance-carrier.witnessChildCentroidDistancesM[row])>tolerance ||
                distance>in.maximumTransportDistanceM+tolerance)
                return fail("source carrier witness distance does not recompute within the unchanged centroid bound");
            weightSum+=weight;
            for (std::size_t axis=0; axis<3u; ++axis) worldMoment[axis]+=weight*point[axis];
            carrierWorldPoints.push_back(point);
        }
        if (std::abs(weightSum-1.0)>2.0e-12)
            return fail("source carrier weights do not conserve the original parent mass and reference measure");
        for (std::size_t node=0; node<4u; ++node)
            if (std::abs(baryMoment[node]-0.25)>2.0e-10)
                return fail("source carrier weighted barycentric moment does not preserve the parent centroid");
        if (pointDistance(worldMoment,in.parentCentroidsWorld[parent])>1.0e-11)
            return fail("source carrier weighted world moment does not preserve the parent centroid");
    }
    if (carrierRows.empty() || carrier.carrierWeights.empty())
        return fail("source carrier packet is present but contains no explicitly witnessed quadrature source rows");
    error.clear(); return true;
}

inline bool expandParentRows(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        std::vector<std::uint32_t>& virtualToParent,
        std::vector<std::uint32_t>& virtualToCarrier,
        std::vector<std::vector<std::uint32_t>>& parentToVirtual,
        std::vector<Point>& centroids,
        std::vector<double>& volumes,
        std::vector<double>& masses,
        std::vector<std::uint32_t>& materials,
        std::vector<std::uint32_t>& donors,
        std::vector<std::uint32_t>& regions,
        std::vector<std::uint32_t>& donorOffsets,
        std::vector<std::uint32_t>& donorIDs,
        std::vector<double>& donorMass,
        ConservativeSpatialMassTransportInput& expandedInput,
        ConservativeSpatialChildSemanticSupportData& expandedSupport,
        const std::string_view carrierArraySHA,
        std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t np=in.parentMassKg.size();
    constexpr std::uint32_t invalid=std::numeric_limits<std::uint32_t>::max();
    for (std::size_t parent=0; parent<np; ++parent) {
        const auto begin=carrier.parentOffsets[parent],end=carrier.parentOffsets[parent+1u];
        if (begin==end) {
            virtualToParent.push_back(static_cast<std::uint32_t>(parent));
            virtualToCarrier.push_back(invalid);
            parentToVirtual[parent].push_back(static_cast<std::uint32_t>(virtualToParent.size()-1u));
            centroids.push_back(in.parentCentroidsWorld[parent]);
            volumes.push_back(in.parentReferenceVolumeM3[parent]); masses.push_back(in.parentMassKg[parent]);
            materials.push_back(in.parentMaterialIndices[parent]); donors.push_back(in.parentDonorBodyIndices[parent]);
            regions.push_back(in.parentAnatomicalRegionIndices[parent]);
            const auto sb=in.parentSourceDonorOffsets[parent],se=in.parentSourceDonorOffsets[parent+1u];
            for (std::uint32_t r=sb;r<se;++r) { donorIDs.push_back(in.parentSourceDonorBodyIndices[r]); donorMass.push_back(in.parentSourceDonorMassKg[r]); }
            donorOffsets.push_back(static_cast<std::uint32_t>(donorIDs.size()));
            continue;
        }
        for (std::uint32_t carrierRow=begin; carrierRow<end; ++carrierRow) {
            const double weight=carrier.carrierWeights[carrierRow];
            const Bary& bary=carrier.carrierBarycentric[carrierRow];
            virtualToParent.push_back(static_cast<std::uint32_t>(parent));
            virtualToCarrier.push_back(carrierRow);
            parentToVirtual[parent].push_back(static_cast<std::uint32_t>(virtualToParent.size()-1u));
            centroids.push_back(barycentricPoint(carrier.parentReferenceTetWorldVertices[parent],bary));
            volumes.push_back(in.parentReferenceVolumeM3[parent]*weight);
            masses.push_back(in.parentMassKg[parent]*weight);
            materials.push_back(in.parentMaterialIndices[parent]); donors.push_back(in.parentDonorBodyIndices[parent]);
            regions.push_back(in.parentAnatomicalRegionIndices[parent]);
            const auto sb=in.parentSourceDonorOffsets[parent],se=in.parentSourceDonorOffsets[parent+1u];
            double priorMass=0.0;
            for (std::uint32_t r=sb;r<se;++r) {
                const double scaled=(r+1u==se)?(masses.back()-priorMass):in.parentSourceDonorMassKg[r]*weight;
                if (!std::isfinite(scaled) || scaled<=0.0)
                    return fail("source carrier scaling produced a nonpositive donor contribution");
                donorIDs.push_back(in.parentSourceDonorBodyIndices[r]); donorMass.push_back(scaled); priorMass+=scaled;
            }
            donorOffsets.push_back(static_cast<std::uint32_t>(donorIDs.size()));
        }
    }
    if (virtualToParent.empty() || virtualToParent.size()>invalid)
        return fail("expanded source carrier row count exceeds the bounded index range");
    expandedInput=in;
    expandedInput.parentCentroidsWorld=centroids;
    expandedInput.parentReferenceVolumeM3=volumes;
    expandedInput.parentMassKg=masses;
    expandedInput.parentMaterialIndices=materials;
    expandedInput.parentDonorBodyIndices=donors;
    expandedInput.parentAnatomicalRegionIndices=regions;
    expandedInput.parentSourceDonorOffsets=donorOffsets;
    expandedInput.parentSourceDonorBodyIndices=donorIDs;
    expandedInput.parentSourceDonorMassKg=donorMass;

    expandedSupport.childOffsets.assign(support.childOffsets.begin(),support.childOffsets.end());
    expandedSupport.slotGroupIndices.assign(support.slotGroupIndices.begin(),support.slotGroupIndices.end());
    expandedSupport.volumeFractions.assign(support.volumeFractions.begin(),support.volumeFractions.end());
    expandedSupport.groupMaterialIndices.assign(support.groupMaterialIndices.begin(),support.groupMaterialIndices.end());
    expandedSupport.donorBodyIndices.assign(support.donorBodyIndices.begin(),support.donorBodyIndices.end());
    expandedSupport.anatomicalRegionIndices.assign(support.anatomicalRegionIndices.begin(),support.anatomicalRegionIndices.end());
    expandedSupport.derivedGroupIdentityIDs.assign(support.derivedGroupIdentityIDs.begin(),support.derivedGroupIdentityIDs.end());
    expandedSupport.sourceStableFaceOffsets.assign(support.sourceStableFaceOffsets.begin(),support.sourceStableFaceOffsets.end());
    expandedSupport.sourceStableFaceRows.assign(support.sourceStableFaceRows.begin(),support.sourceStableFaceRows.end());
    expandedSupport.typedSupportSHA256.assign(support.typedSupportSHA256);
    expandedSupport.cellAdmissibleRegionOffsets.assign(support.cellAdmissibleRegionOffsets.begin(),support.cellAdmissibleRegionOffsets.end());
    expandedSupport.cellAdmissibleRegionIds.assign(support.cellAdmissibleRegionIds.begin(),support.cellAdmissibleRegionIds.end());
    expandedSupport.admissibilitySourceSupportSHA256.assign(support.admissibilitySourceSupportSHA256);
    expandedSupport.admissibilityMaximumSourceRegionHops=support.admissibilityMaximumSourceRegionHops;
    expandedSupport.parentStableSourceFaceIdentities.reserve(virtualToParent.size());
    for (const auto parent : virtualToParent)
        expandedSupport.parentStableSourceFaceIdentities.push_back(support.parentStableSourceFaceIdentities[parent]);
    expandedSupport.sourceParentTetOffsets.assign(support.groupMaterialIndices.size()+1u,0u);
    for (std::size_t group=0;group<support.groupMaterialIndices.size();++group) {
        const auto begin=support.sourceParentTetOffsets[group],end=support.sourceParentTetOffsets[group+1u];
        for (std::uint32_t r=begin;r<end;++r) {
            const auto parent=support.sourceParentTetRows[r];
            if (parent>=parentToVirtual.size() || parentToVirtual[parent].empty())
                return fail("source carrier expansion lost a source parent in semantic group ancestry");
            for (const auto virtualRow:parentToVirtual[parent])
                expandedSupport.sourceParentTetRows.push_back(virtualRow);
        }
        expandedSupport.sourceParentTetOffsets[group+1u]=
            static_cast<std::uint32_t>(expandedSupport.sourceParentTetRows.size());
    }
    spatial_mass_transport_detail::SHA256Builder bound;
    bound.string("numi.human.source-tetra-centroid-carrier-expanded-typed-support.v1");
    bound.string(fractional_semantic_transport_detail::supportDigest(support));
    bound.string(carrierArraySHA);
    bound.string(carrier.sourceCarrierPacketSHA256);
    expandedSupport.typedSupportSHA256=bound.finish();
    return true;
}

inline bool aggregateCarrierResult(
        const ConservativeSpatialMassTransportInput& originalInput,
        const ConservativeSpatialChildSemanticSupport& originalSupport,
        const SourceCellCarrierQuadratureInput& carrier,
        const std::vector<std::uint32_t>& virtualToParent,
        const std::vector<std::uint32_t>& virtualToCarrier,
        const ConservativeSpatialMassTransportResult& expandedResult,
        ConservativeSpatialMassTransportResult& out,
        SourceCellCarrierTransportTrace& trace,
        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t np=originalInput.parentMassKg.size(),nc=originalInput.childReferenceVolumeM3.size();
    if (expandedResult.childParentOffsets.size()!=nc+1u ||
        expandedResult.childParentOffsets.empty() ||
        expandedResult.childParentOffsets.back()!=expandedResult.childParentRows.size() ||
        expandedResult.childParentTransportMassKg.size()!=expandedResult.childParentRows.size() ||
        expandedResult.childParentTransportMeasureM3.size()!=expandedResult.childParentRows.size() ||
        expandedResult.childParentTransportDistanceM.size()!=expandedResult.childParentRows.size() ||
        expandedResult.parentMassTransferredKg.size()!=virtualToParent.size() ||
        expandedResult.parentMeasureTransferredM3.size()!=virtualToParent.size())
        return fail("expanded source carrier solver result has incomplete flow arrays");

    struct Piece { std::uint32_t virtualParent; double mass,measure,distance; };
    out=expandedResult;
    out.parentMassTransferredKg.assign(np,0.0);
    out.parentMeasureTransferredM3.assign(np,0.0);
    std::vector<ExactDoubleSum> parentMassSum(np),parentMeasureSum(np);
    for (std::size_t v=0;v<virtualToParent.size();++v) {
        const auto p=virtualToParent[v];
        if (p>=np) return fail("expanded source carrier map references an invalid source parent row");
        parentMassSum[p].add(expandedResult.parentMassTransferredKg[v]);
        parentMeasureSum[p].add(expandedResult.parentMeasureTransferredM3[v]);
    }
    for (std::size_t p=0;p<np;++p) {
        out.parentMassTransferredKg[p]=parentMassSum[p].rounded();
        out.parentMeasureTransferredM3[p]=parentMeasureSum[p].rounded();
        const double massError=std::abs(out.parentMassTransferredKg[p]-originalInput.parentMassKg[p]);
        const double measureError=std::abs(out.parentMeasureTransferredM3[p]-originalInput.parentReferenceVolumeM3[p]);
        out.maximumParentMassClosureErrorKg=std::max(out.maximumParentMassClosureErrorKg,massError);
        out.maximumParentMeasureClosureErrorM3=std::max(out.maximumParentMeasureClosureErrorM3,measureError);
        if (massError>std::max(originalInput.absoluteMassToleranceKg,
                originalInput.relativeClosureTolerance*originalInput.parentMassKg[p]) ||
            measureError>std::max(originalInput.absoluteMeasureToleranceM3,
                originalInput.relativeClosureTolerance*originalInput.parentReferenceVolumeM3[p]))
            return fail("collapsed source carrier flows do not close original source parent mass and measure");
    }

    out.childParentOffsets.assign(nc+1u,0u);
    out.childParentRows.clear(); out.childParentTransportMassKg.clear();
    out.childParentTransportMeasureM3.clear(); out.childParentTransportDistanceM.clear();
    out.childParentVolumeFractions.clear();
    out.childDominantParentRows.assign(nc,kInvalid);
    trace.childParentCarrierOffsets.assign(1u,0u);
    trace.childParentCarrierRows.clear(); trace.childParentCarrierMassKg.clear();
    trace.childParentCarrierMeasureM3.clear(); trace.childParentCarrierDistanceM.clear();
    trace.expandedParentToSourceParentRows=virtualToParent;
    trace.expandedParentToCarrierRows=virtualToCarrier;
    trace.sourceCarrierPacketSHA256=std::string(carrier.sourceCarrierPacketSHA256);
    trace.sourceCarrierArraySHA256=carrierArrayDigest(originalInput,carrier);
    long double weightedDistance=0.0L,totalMass=0.0L;
    for(std::size_t child=0;child<nc;++child){
        out.childParentOffsets[child]=static_cast<std::uint32_t>(out.childParentRows.size());
        std::map<std::uint32_t,std::vector<Piece>> grouped;
        const auto begin=expandedResult.childParentOffsets[child],end=expandedResult.childParentOffsets[child+1u];
        if(begin>=end)return fail("source carrier transport left a real child without a parent flow");
        for(std::uint32_t edge=begin;edge<end;++edge){
            const auto virtualParent=expandedResult.childParentRows[edge];
            if(virtualParent>=virtualToParent.size())
                return fail("expanded source carrier flow references an invalid virtual parent row");
            grouped[virtualToParent[virtualParent]].push_back({virtualParent,
                expandedResult.childParentTransportMassKg[edge],
                expandedResult.childParentTransportMeasureM3[edge],
                expandedResult.childParentTransportDistanceM[edge]});
        }
        const std::size_t firstOutputEdge=out.childParentRows.size();
        ExactDoubleSum childMeasureSum;
        double bestMeasure=-1.0;std::uint32_t dominant=kInvalid;
        for(auto& [parent,pieces]:grouped){
            std::sort(pieces.begin(),pieces.end(),[](const Piece& a,const Piece& b){return a.virtualParent<b.virtualParent;});
            ExactDoubleSum massSum,measureSum;
            long double distanceMass=0.0L;
            for(const auto& piece:pieces){
                if(!std::isfinite(piece.mass)||piece.mass<=0.0||
                   !std::isfinite(piece.measure)||piece.measure<=0.0||
                   !std::isfinite(piece.distance)||piece.distance>originalInput.maximumTransportDistanceM)
                    return fail("expanded source carrier flow contains invalid mass, measure or centroid distance");
                massSum.add(piece.mass);measureSum.add(piece.measure);
                distanceMass+=static_cast<long double>(piece.mass)*piece.distance;
                trace.childParentCarrierRows.push_back(piece.virtualParent);
                trace.childParentCarrierMassKg.push_back(piece.mass);
                trace.childParentCarrierMeasureM3.push_back(piece.measure);
                trace.childParentCarrierDistanceM.push_back(piece.distance);
            }
            const double mass=massSum.rounded(),measure=measureSum.rounded();
            if(!std::isfinite(mass)||mass<=0.0||!std::isfinite(measure)||measure<=0.0)
                return fail("collapsed source carrier parent-child flow is invalid");
            out.childParentRows.push_back(parent);
            out.childParentTransportMassKg.push_back(mass);
            out.childParentTransportMeasureM3.push_back(measure);
            out.childParentTransportDistanceM.push_back(static_cast<double>(distanceMass/mass));
            childMeasureSum.add(measure);
            if(measure>bestMeasure||(measure==bestMeasure&&parent<dominant)){bestMeasure=measure;dominant=parent;}
            for(const auto& piece:pieces){weightedDistance+=static_cast<long double>(piece.mass)*piece.distance;totalMass+=piece.mass;}
            trace.childParentCarrierOffsets.push_back(static_cast<std::uint32_t>(trace.childParentCarrierRows.size()));
        }
        const double childMeasure=childMeasureSum.rounded();
        if(childMeasure<=0.0||!std::isfinite(childMeasure))
            return fail("collapsed source carrier child measure is nonpositive");
        ExactDoubleSum fractionSum;
        for(std::size_t edge=firstOutputEdge;edge<out.childParentRows.size();++edge){
            const double fraction=out.childParentTransportMeasureM3[edge]/childMeasure;
            if(!std::isfinite(fraction)||fraction<=0.0)
                return fail("collapsed source carrier child measure fraction is invalid");
            out.childParentVolumeFractions.push_back(fraction);
            fractionSum.add(fraction);
        }
        out.maximumChildFractionSumError=std::max(out.maximumChildFractionSumError,
            std::abs(fractionSum.rounded()-1.0));
        if(out.maximumChildFractionSumError>2.0e-12)
            return fail("collapsed source carrier child fractions do not sum to one");
        out.childDominantParentRows[child]=dominant;
        out.childParentOffsets[child+1u]=static_cast<std::uint32_t>(out.childParentRows.size());
    }
    out.transportEdgeCount=out.childParentRows.size();
    out.massWeightedMeanTransportDistanceM=totalMass>0.0L?
        static_cast<double>(weightedDistance/totalMass):0.0;
    out.maximumChildFractionSumError=0.0;
    out.semanticSupportSHA256=combinedSupportDigest(originalSupport,
        carrierArrayDigest(originalInput,carrier),expandedResult.semanticSupportSHA256);
    std::string parentHash,childHash,boundsHash;
    hashInputArrays(originalInput,parentHash,childHash,boundsHash);
    out.parentDataSHA256=std::move(parentHash);
    out.childDataSHA256=fractional_semantic_transport_detail::bindSupportToChildHash(childHash,out.semanticSupportSHA256);
    out.materialBoundsSHA256=std::move(boundsHash);
    out.receiptSHA256.clear();
    trace.expandedSolverReceiptSHA256=expandedResult.receiptSHA256;
    spatial_mass_transport_detail::SHA256Builder traceHash;
    traceHash.string("numi.human.source-tetra-centroid-carrier-flow-trace.v1");
    traceHash.string(carrierArrayDigest(originalInput,carrier));
    traceHash.string(expandedResult.receiptSHA256);
    for (const auto value:trace.expandedParentToSourceParentRows) traceHash.u32(value);
    for (const auto value:trace.expandedParentToCarrierRows) traceHash.u32(value);
    for (const auto value:trace.carrierSourceParentRows) traceHash.u32(value);
    for (const auto value:trace.originalCentroidNearestAdmissibleDistanceM) traceHash.real(value);
    for (const auto value:trace.childParentCarrierOffsets) traceHash.u32(value);
    for (const auto value:trace.childParentCarrierRows) traceHash.u32(value);
    for (const auto value:trace.childParentCarrierMassKg) traceHash.real(value);
    for (const auto value:trace.childParentCarrierMeasureM3) traceHash.real(value);
    for (const auto value:trace.childParentCarrierDistanceM) traceHash.real(value);
    trace.receiptSHA256=traceHash.finish();
    const std::string base= fractional_semantic_transport_detail::sharedCapacityReceipt(
        originalInput,out,out.semanticSupportSHA256);
    SHA256Builder receipt;
    receipt.string("numi.human.conservative-spatial-mass-transport.source-carrier-result.v1");
    receipt.string(base);receipt.string(trace.receiptSHA256);receipt.string(expandedResult.receiptSHA256);
    out.receiptSHA256=receipt.finish();
    return true;
}

} // namespace source_carrier_transport_detail

[[nodiscard]] inline bool assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        const ConservativeSpatialMassTransportInput& input,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        ConservativeSpatialMassTransportResult& out,
        SourceCellCarrierTransportTrace& trace,
        std::string& error) {
    using namespace source_carrier_transport_detail;
    std::vector<std::uint32_t> parentGroup;
    std::vector<std::map<std::uint32_t,double>> preferences;
    std::vector<std::vector<std::uint32_t>> admissibleRegions;
    std::vector<Point> carrierPoints;
    std::vector<std::uint32_t> carrierRows;
    std::vector<double> strictNearest;
    if (!validateCarrierInput(input,support,carrier,parentGroup,preferences,admissibleRegions,
                              carrierPoints,carrierRows,strictNearest,error)) return false;
    const std::string carrierSHA=carrierArrayDigest(input,carrier);
    std::vector<std::uint32_t> virtualToParent,virtualToCarrier;
    std::vector<std::vector<std::uint32_t>> parentToVirtual(input.parentMassKg.size());
    std::vector<Point> parentCentroids;
    std::vector<double> parentVolumes,parentMasses,donorMasses;
    std::vector<std::uint32_t> parentMaterials,parentDonors,parentRegions,donorOffsets{0u},donorIDs;
    ConservativeSpatialMassTransportInput expandedInput;
    ConservativeSpatialChildSemanticSupportData expandedSupport;
    if (!expandParentRows(input,support,carrier,virtualToParent,virtualToCarrier,parentToVirtual,
                          parentCentroids,parentVolumes,parentMasses,parentMaterials,parentDonors,
                          parentRegions,donorOffsets,donorIDs,donorMasses,expandedInput,
                          expandedSupport,carrierSHA,error)) return false;
    ConservativeSpatialMassTransportResult expandedResult;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupport(
            expandedInput,expandedSupport.view(),expandedResult,error)) return false;
    SourceCellCarrierTransportTrace nextTrace;
    nextTrace.carrierSourceParentRows=std::move(carrierRows);
    nextTrace.originalCentroidNearestAdmissibleDistanceM=std::move(strictNearest);
    if (!aggregateCarrierResult(input,support,carrier,virtualToParent,virtualToCarrier,
                                expandedResult,out,nextTrace,error)) return false;
    trace=std::move(nextTrace);
    error.clear();return true;
}

[[nodiscard]] inline bool verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        const ConservativeSpatialMassTransportInput& input,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        const ConservativeSpatialMassTransportResult& supplied,
        const SourceCellCarrierTransportTrace& suppliedTrace,
        std::string& error) {
    ConservativeSpatialMassTransportResult expected;
    SourceCellCarrierTransportTrace expectedTrace;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
            input,support,carrier,expected,expectedTrace,error)) return false;
    if (!fractional_semantic_transport_detail::sameResult(expected,supplied))
        return (error="source carrier aggregate mass-transport result differs from deterministic replay",false);
    if (expectedTrace.sourceCarrierPacketSHA256!=suppliedTrace.sourceCarrierPacketSHA256 ||
        expectedTrace.sourceCarrierArraySHA256!=suppliedTrace.sourceCarrierArraySHA256 ||
        expectedTrace.expandedSolverReceiptSHA256!=suppliedTrace.expandedSolverReceiptSHA256 ||
        expectedTrace.expandedParentToSourceParentRows!=suppliedTrace.expandedParentToSourceParentRows ||
        expectedTrace.expandedParentToCarrierRows!=suppliedTrace.expandedParentToCarrierRows ||
        expectedTrace.carrierSourceParentRows!=suppliedTrace.carrierSourceParentRows ||
        expectedTrace.originalCentroidNearestAdmissibleDistanceM!=suppliedTrace.originalCentroidNearestAdmissibleDistanceM ||
        expectedTrace.childParentCarrierOffsets!=suppliedTrace.childParentCarrierOffsets ||
        expectedTrace.childParentCarrierRows!=suppliedTrace.childParentCarrierRows ||
        expectedTrace.childParentCarrierMassKg!=suppliedTrace.childParentCarrierMassKg ||
        expectedTrace.childParentCarrierMeasureM3!=suppliedTrace.childParentCarrierMeasureM3 ||
        expectedTrace.childParentCarrierDistanceM!=suppliedTrace.childParentCarrierDistanceM ||
        expectedTrace.receiptSHA256!=suppliedTrace.receiptSHA256)
        return (error="source carrier flow trace differs from deterministic replay",false);
    error.clear();return true;
}


// Solver-free verification for a recorded source-carrier allocation. It checks
// the imported flow as a feasible coupling and reaggregates the carrier trace;
// it deliberately does not invoke the allocation optimizer.
[[nodiscard]] inline bool finalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
        const ConservativeSpatialMassTransportInput& input,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        ConservativeSpatialMassTransportResult& supplied,
        SourceCellCarrierTransportTrace& suppliedTrace,
        const bool canonicalizeImportedMetadata,
        std::string& error) {
    using namespace source_carrier_transport_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    std::vector<std::uint32_t> parentGroup,carrierParents;
    std::vector<std::map<std::uint32_t,double>> preferences;
    std::vector<std::vector<std::uint32_t>> admissibleRegions;
    std::vector<Point> carrierPoints;
    std::vector<double> strictNearest;
    if(!validateCarrierInput(input,support,carrier,parentGroup,preferences,admissibleRegions,
                             carrierPoints,carrierParents,strictNearest,error))return false;

    const auto np=input.parentMassKg.size(),nc=input.childReferenceVolumeM3.size();
    if(supplied.method!=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport||
       supplied.parentMassPacketSHA256!=input.parentMassPacketSHA256||
       supplied.parentSourceContributionSHA256!=input.parentSourceContributionSHA256||
       supplied.childMeshSHA256!=input.childMeshSHA256||
       supplied.regionLabelSHA256!=input.regionLabelSHA256||
       supplied.coordinateFrameSHA256!=input.coordinateFrameSHA256||
       supplied.maximumTransportDistanceM!=input.maximumTransportDistanceM||
       supplied.relativeClosureTolerance!=input.relativeClosureTolerance||
       supplied.absoluteMassToleranceKg!=input.absoluteMassToleranceKg||
       supplied.absoluteMeasureToleranceM3!=input.absoluteMeasureToleranceM3||
       supplied.maximumAllowedEdges!=input.maximumAllowedEdges||
       supplied.maximumAugmentations!=input.maximumAugmentations||
       supplied.transportEdgeCount==0u||supplied.transportEdgeCount>input.maximumAllowedEdges||
       supplied.augmentationCount==0u||supplied.augmentationCount>input.maximumAugmentations)
        return fail("recorded source-carrier result policy, pins, or bounded flow counts differ from current inputs");
    if(supplied.childParentOffsets.size()!=nc+1u||supplied.childParentOffsets.empty()||
       supplied.childParentOffsets.front()!=0u||
       supplied.childParentOffsets.back()!=supplied.childParentRows.size()||
       supplied.childParentRows.empty()||
       supplied.childParentTransportMassKg.size()!=supplied.childParentRows.size()||
       supplied.childParentTransportMeasureM3.size()!=supplied.childParentRows.size()||
       supplied.childParentTransportDistanceM.size()!=supplied.childParentRows.size()||
       supplied.childParentVolumeFractions.size()!=supplied.childParentRows.size()||
       supplied.childMassKg.size()!=nc||supplied.childAncestryMeasureVolumeM3.size()!=nc||
       supplied.childImpliedDensityKgM3.size()!=nc||supplied.childDominantParentRows.size()!=nc||
       supplied.parentMassTransferredKg.size()!=np||supplied.parentMeasureTransferredM3.size()!=np||
       supplied.childSourceDonorOffsets.size()!=nc+1u||supplied.childSourceDonorOffsets.empty()||
       supplied.childSourceDonorOffsets.front()!=0u||
       supplied.childSourceDonorOffsets.back()!=supplied.childSourceDonorBodyIndices.size()||
       supplied.childSourceDonorBodyIndices.size()!=supplied.childSourceDonorMassKg.size()||
       (!canonicalizeImportedMetadata&&supplied.groupClosure.size()!=support.groupMaterialIndices.size())||
       !supplied.usesFractionalSemanticSupport||
       (!canonicalizeImportedMetadata&&(supplied.childSemanticGroupOffsets.size()!=nc+1u||
       supplied.childSemanticGroupOffsets.empty()||
       supplied.childSemanticGroupOffsets.front()!=0u||
       supplied.childSemanticGroupOffsets.back()!=supplied.childSemanticGroupIndices.size()||
       supplied.childSemanticGroupIndices.size()!=supplied.childSemanticVolumeFractions.size())))
        return fail("recorded source-carrier aggregate arrays or semantic CSR are incomplete");

    std::vector<std::uint32_t> virtualToParent,virtualToCarrier;
    std::vector<std::vector<std::uint32_t>> parentToVirtual(np);
    std::vector<Point> virtualCentroids;
    std::vector<double> virtualVolumes,virtualMasses,virtualDonorMass;
    std::vector<std::uint32_t> virtualMaterials,virtualDonors,virtualRegions,virtualDonorOffsets{0u},virtualDonorIDs;
    ConservativeSpatialMassTransportInput expandedInput;
    ConservativeSpatialChildSemanticSupportData expandedSupport;
    const auto arraySHA=carrierArrayDigest(input,carrier);
    if(!expandParentRows(input,support,carrier,virtualToParent,virtualToCarrier,parentToVirtual,
            virtualCentroids,virtualVolumes,virtualMasses,virtualMaterials,virtualDonors,
            virtualRegions,virtualDonorOffsets,virtualDonorIDs,virtualDonorMass,
            expandedInput,expandedSupport,arraySHA,error))return false;
    if(suppliedTrace.expandedParentToSourceParentRows!=virtualToParent||
       suppliedTrace.expandedParentToCarrierRows!=virtualToCarrier||
       suppliedTrace.carrierSourceParentRows!=carrierParents||
       suppliedTrace.originalCentroidNearestAdmissibleDistanceM!=strictNearest||
       suppliedTrace.childParentCarrierOffsets.size()!=supplied.childParentRows.size()+1u||
       suppliedTrace.childParentCarrierOffsets.empty()||
       suppliedTrace.childParentCarrierOffsets.front()!=0u||
       suppliedTrace.childParentCarrierOffsets.back()!=suppliedTrace.childParentCarrierRows.size()||
       suppliedTrace.childParentCarrierRows.size()!=suppliedTrace.childParentCarrierMassKg.size()||
       suppliedTrace.childParentCarrierRows.size()!=suppliedTrace.childParentCarrierMeasureM3.size()||
       suppliedTrace.childParentCarrierRows.size()!=suppliedTrace.childParentCarrierDistanceM.size()||
       suppliedTrace.sourceCarrierPacketSHA256!=carrier.sourceCarrierPacketSHA256||
       suppliedTrace.sourceCarrierArraySHA256!=arraySHA||
       !spatial_mass_transport_detail::validSHA256(suppliedTrace.expandedSolverReceiptSHA256))
        return fail("recorded source-carrier trace maps or input pins differ from current carrier arrays");

    ConservativeSpatialMassTransportResult expanded;
    expanded.method=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    expanded.childParentOffsets.assign(nc+1u,0u);
    expanded.parentMassTransferredKg.assign(virtualToParent.size(),0.0);
    expanded.parentMeasureTransferredM3.assign(virtualToParent.size(),0.0);
    expanded.childMassKg.assign(nc,0.0);
    expanded.childAncestryMeasureVolumeM3.assign(nc,0.0);
    std::vector<std::map<std::uint32_t,double>> donorMassByChild(nc);
    std::vector<std::map<std::uint32_t,double>> semanticMassByChild(nc);
    std::vector<double> actualParentMass(virtualToParent.size(),0.0);
    std::vector<double> actualParentMeasure(virtualToParent.size(),0.0);
    std::vector<double> actualGroupParentMass(support.groupMaterialIndices.size(),0.0);
    std::vector<double> actualGroupChildMass(support.groupMaterialIndices.size(),0.0);
    std::map<std::pair<std::uint32_t,std::uint32_t>,double> parentSourceByDonor,childSourceByDonor;
    for(std::size_t v=0;v<virtualToParent.size();++v){
        const auto p=virtualToParent[v],g=parentGroup[p];
        actualGroupParentMass[g]+=virtualMasses[v];
        for(std::uint32_t r=virtualDonorOffsets[v];r<virtualDonorOffsets[v+1u];++r)
            parentSourceByDonor[{virtualMaterials[v],virtualDonorIDs[r]}]+=virtualDonorMass[r];
    }

    long double weightedDistance=0.0L,totalMass=0.0L;
    double maximumDistance=0.0,maximumParentMassError=0.0,maximumParentMeasureError=0.0;
    double maximumSourceError=0.0,minimumDensity=std::numeric_limits<double>::infinity();
    double maximumDensity=0.0,maximumFractionError=0.0;
    expanded.childParentRows.reserve(suppliedTrace.childParentCarrierRows.size());
    expanded.childParentTransportMassKg.reserve(suppliedTrace.childParentCarrierRows.size());
    expanded.childParentTransportMeasureM3.reserve(suppliedTrace.childParentCarrierRows.size());
    expanded.childParentTransportDistanceM.reserve(suppliedTrace.childParentCarrierRows.size());
    expanded.childParentVolumeFractions.reserve(suppliedTrace.childParentCarrierRows.size());
    for(std::size_t c=0;c<nc;++c){
        const auto eb=supplied.childParentOffsets[c],ee=supplied.childParentOffsets[c+1u];
        if(eb>=ee||ee>supplied.childParentRows.size())
            return fail("recorded source-carrier child row is empty or out of range");
        expanded.childParentOffsets[c]=static_cast<std::uint32_t>(expanded.childParentRows.size());
        std::uint32_t priorParent=0u;
        for(std::uint32_t e=eb;e<ee;++e){
            const auto p=supplied.childParentRows[e];
            if(p>=np||(e>eb&&p<=priorParent))
                return fail("recorded source-carrier aggregate parent rows are not sorted and unique");
            priorParent=p;
            const auto kb=suppliedTrace.childParentCarrierOffsets[e];
            const auto ke=suppliedTrace.childParentCarrierOffsets[e+1u];
            if(kb>=ke||ke>suppliedTrace.childParentCarrierRows.size())
                return fail("recorded source-carrier aggregate edge has no expanded carrier flow");
            double massSum=0.0,measureSum=0.0;
            long double weightedEdgeDistance=0.0L;
            std::uint32_t previousVirtual=0u;
            for(std::uint32_t k=kb;k<ke;++k){
                const auto v=suppliedTrace.childParentCarrierRows[k];
                const double mass=suppliedTrace.childParentCarrierMassKg[k];
                const double measure=suppliedTrace.childParentCarrierMeasureM3[k];
                const double distance=suppliedTrace.childParentCarrierDistanceM[k];
                if(v>=virtualToParent.size()||virtualToParent[v]!=p||
                   (k>kb&&v<=previousVirtual)||!std::isfinite(mass)||mass<=0.0||
                   !std::isfinite(measure)||measure<=0.0||
                   !std::isfinite(distance)||distance<0.0)
                    return fail("recorded source-carrier expanded flow map or value is invalid");
                previousVirtual=v;
                const auto group=parentGroup[p];
                if(input.parentMaterialIndices[p]!=input.childMaterialIndices[c]||
                   support.groupMaterialIndices[group]!=input.parentMaterialIndices[p])
                    return fail("recorded source-carrier edge crosses its material or source group");
                const auto parentRegion=input.parentAnatomicalRegionIndices[p];
                if(!std::binary_search(admissibleRegions[c].begin(),admissibleRegions[c].end(),parentRegion))
                    return fail("recorded source-carrier edge source region is outside the child's one-hop registered support");
                const double actualDistance=pointDistance(expandedInput.parentCentroidsWorld[v],
                                                           input.childCentroidsWorld[c]);
                const double density=virtualMasses[v]/virtualVolumes[v];
                const double expectedMeasure=mass/density;
                if(actualDistance!=distance||actualDistance>input.maximumTransportDistanceM||
                   measure!=expectedMeasure)
                    return fail("recorded source-carrier edge distance or source measure differs from current inputs");
                massSum+=mass;measureSum+=measure;
                weightedEdgeDistance+=static_cast<long double>(mass)*distance;
                actualParentMass[v]+=mass;actualParentMeasure[v]+=measure;
                expanded.childParentRows.push_back(v);
                expanded.childParentTransportMassKg.push_back(mass);
                expanded.childParentTransportMeasureM3.push_back(measure);
                expanded.childParentTransportDistanceM.push_back(distance);
                const double parentFraction=mass/virtualMasses[v];
                for(std::uint32_t sr=virtualDonorOffsets[v];sr<virtualDonorOffsets[v+1u];++sr){
                    const auto donor=virtualDonorIDs[sr];
                    const double sourceMass=virtualDonorMass[sr]*parentFraction;
                    donorMassByChild[c][donor]+=sourceMass;
                    childSourceByDonor[{virtualMaterials[v],donor}]+=sourceMass;
                }
                semanticMassByChild[c][group]+=mass;
                actualGroupChildMass[group]+=mass;
                maximumDistance=std::max(maximumDistance,distance);
                weightedDistance+=static_cast<long double>(mass)*distance;totalMass+=mass;
            }
            if(massSum!=supplied.childParentTransportMassKg[e]||
               measureSum!=supplied.childParentTransportMeasureM3[e]||
               static_cast<double>(weightedEdgeDistance/massSum)!=supplied.childParentTransportDistanceM[e])
                return fail("recorded source-carrier aggregate edge differs from its carrier rows");
            const double fraction=measureSum/supplied.childAncestryMeasureVolumeM3[c];
            if(supplied.childParentVolumeFractions[e]!=fraction)
                return fail("recorded source-carrier aggregate measure fraction differs from its flow rows");
        }
        expanded.childParentOffsets[c+1u]=static_cast<std::uint32_t>(expanded.childParentRows.size());
    }
    expanded.childParentOffsets[nc]=static_cast<std::uint32_t>(expanded.childParentRows.size());
    expanded.transportEdgeCount=expanded.childParentRows.size();
    expanded.augmentationCount=supplied.augmentationCount;
    expanded.semanticCandidateEdgeCount=supplied.semanticCandidateEdgeCount;
    expanded.usesFractionalSemanticSupport=true;
    expanded.semanticSupportSHA256=fractional_semantic_transport_detail::supportDigest(expandedSupport.view());
    if(!canonicalizeImportedMetadata){
        expanded.childSemanticGroupOffsets=supplied.childSemanticGroupOffsets;
        expanded.childSemanticGroupIndices=supplied.childSemanticGroupIndices;
        expanded.childSemanticVolumeFractions=supplied.childSemanticVolumeFractions;
    }
    expanded.groupClosure.clear();
    expanded.sourceDonorClosure.clear();
    expanded.parentMassPacketSHA256=std::string(expandedInput.parentMassPacketSHA256);
    expanded.parentSourceContributionSHA256=std::string(expandedInput.parentSourceContributionSHA256);
    expanded.childMeshSHA256=std::string(expandedInput.childMeshSHA256);
    expanded.regionLabelSHA256=std::string(expandedInput.regionLabelSHA256);
    expanded.coordinateFrameSHA256=std::string(expandedInput.coordinateFrameSHA256);
    expanded.maximumTransportDistanceM=expandedInput.maximumTransportDistanceM;
    expanded.relativeClosureTolerance=expandedInput.relativeClosureTolerance;
    expanded.absoluteMassToleranceKg=expandedInput.absoluteMassToleranceKg;
    expanded.absoluteMeasureToleranceM3=expandedInput.absoluteMeasureToleranceM3;
    expanded.maximumAllowedEdges=expandedInput.maximumAllowedEdges;
    expanded.maximumAugmentations=expandedInput.maximumAugmentations;

    std::vector<spatial_mass_transport_detail::ExactDoubleSum> childMassSums(nc),childMeasureSums(nc);
    for(std::size_t e=0;e<expanded.childParentRows.size();++e){
        const auto c=static_cast<std::size_t>(std::upper_bound(expanded.childParentOffsets.begin(),
            expanded.childParentOffsets.end(),static_cast<std::uint32_t>(e))-expanded.childParentOffsets.begin()-1u);
        childMassSums[c].add(expanded.childParentTransportMassKg[e]);
        childMeasureSums[c].add(expanded.childParentTransportMeasureM3[e]);
    }
    std::vector<double> computedChildMass(nc,0.0),computedChildMeasure(nc,0.0);
    for(std::size_t c=0;c<nc;++c){
        computedChildMass[c]=childMassSums[c].rounded();
        computedChildMeasure[c]=childMeasureSums[c].rounded();
    }
    expanded.childMassKg=computedChildMass;
    expanded.childAncestryMeasureVolumeM3=computedChildMeasure;
    expanded.childImpliedDensityKgM3.resize(nc);
    expanded.childDominantParentRows.assign(nc,spatial_mass_transport_detail::kInvalid);
    expanded.childSourceDonorOffsets.assign(nc+1u,0u);
    expanded.childParentVolumeFractions.clear();
    for(std::size_t c=0;c<nc;++c){
        if(!(computedChildMass[c]>0.0&&computedChildMeasure[c]>0.0))
            return fail("recorded source-carrier allocation left a child without positive mass or measure");
        double bestMeasure=-1.0;
        for(std::uint32_t e=expanded.childParentOffsets[c];e<expanded.childParentOffsets[c+1u];++e){
            const auto measure=expanded.childParentTransportMeasureM3[e];
            const auto parent=expanded.childParentRows[e];
            const double fraction=measure/computedChildMeasure[c];
            expanded.childParentVolumeFractions.push_back(fraction);
            if(measure>bestMeasure||(measure==bestMeasure&&parent<expanded.childDominantParentRows[c])){
                bestMeasure=measure;expanded.childDominantParentRows[c]=parent;
            }
        }
        const auto material=input.childMaterialIndices[c];
        const double density=computedChildMass[c]/input.childReferenceVolumeM3[c];
        if(!std::isfinite(density)||density<input.materialDensityLowerKgM3[material]-1.0e-7||
           density>input.materialDensityUpperKgM3[material]+1.0e-7)
            return fail("recorded source-carrier child density is outside actual material bounds");
        expanded.childImpliedDensityKgM3[c]=density;
        minimumDensity=std::min(minimumDensity,density);maximumDensity=std::max(maximumDensity,density);
        expanded.childSourceDonorOffsets[c]=static_cast<std::uint32_t>(expanded.childSourceDonorBodyIndices.size());
        for(const auto& [donor,mass]:donorMassByChild[c]){
            expanded.childSourceDonorBodyIndices.push_back(donor);
            expanded.childSourceDonorMassKg.push_back(mass);
        }
        expanded.childSourceDonorOffsets[c+1u]=static_cast<std::uint32_t>(expanded.childSourceDonorBodyIndices.size());
        double sum=0.0;for(const auto& [donor,mass]:donorMassByChild[c]){(void)donor;sum+=mass;}
        if(std::abs(sum-computedChildMass[c])>std::max(input.absoluteMassToleranceKg,
            input.relativeClosureTolerance*computedChildMass[c]))
            return fail("recorded source-carrier child source donors do not close child mass");
    }
    expanded.parentMassTransferredKg=actualParentMass;
    expanded.parentMeasureTransferredM3=actualParentMeasure;
    for(std::size_t v=0;v<virtualToParent.size();++v){
        const double massError=std::abs(actualParentMass[v]-virtualMasses[v]);
        const double measureError=std::abs(actualParentMeasure[v]-virtualVolumes[v]);
        maximumParentMassError=std::max(maximumParentMassError,massError);
        maximumParentMeasureError=std::max(maximumParentMeasureError,measureError);
        if(massError>std::max(input.absoluteMassToleranceKg,input.relativeClosureTolerance*virtualMasses[v])||
           measureError>std::max(input.absoluteMeasureToleranceM3,input.relativeClosureTolerance*virtualVolumes[v]))
            return fail("recorded source-carrier expanded parent mass or measure supply does not close");
    }
    expanded.maximumParentMassClosureErrorKg=maximumParentMassError;
    expanded.maximumParentMeasureClosureErrorM3=maximumParentMeasureError;
    if(!canonicalizeImportedMetadata&&supplied.sourceDonorClosure.size()!=parentSourceByDonor.size())
        return fail("recorded source-carrier source-donor closure rows do not cover the current parent ledger");
    std::size_t sourceClosureRow=0u;
    for(const auto& [key,parentMass]:parentSourceByDonor){
        const auto found=childSourceByDonor.find(key);
        if(found==childSourceByDonor.end())
            return fail("recorded source-carrier source-donor ledger has no matching child mass");
        const double childMass=found->second;
        const double closureError=std::abs(parentMass-childMass);
        maximumSourceError=std::max(maximumSourceError,closureError);
        const double tolerance=std::max(input.absoluteMassToleranceKg,
            input.relativeClosureTolerance*std::max(std::abs(parentMass),std::abs(childMass)));
        if(closureError>tolerance)
            return fail("recorded source-carrier per-source-donor mass closure differs from current flows");
        if(canonicalizeImportedMetadata){
            expanded.sourceDonorClosure.push_back({.materialIndex=key.first,
                .sourceDonorBodyIndex=key.second,.parentMassKg=parentMass,.childMassKg=childMass});
        }else{
            const auto& recorded=supplied.sourceDonorClosure[sourceClosureRow++];
            if(recorded.materialIndex!=key.first||recorded.sourceDonorBodyIndex!=key.second||
               std::abs(recorded.parentMassKg-parentMass)>tolerance||
               std::abs(recorded.childMassKg-childMass)>tolerance)
                return fail("recorded source-carrier per-source-donor mass closure differs from current flows");
            expanded.sourceDonorClosure.push_back(recorded);
        }
    }
    if(!canonicalizeImportedMetadata){
        const double sourceErrorTolerance=std::max(input.absoluteMassToleranceKg,
            input.relativeClosureTolerance*std::max(maximumSourceError,
                supplied.maximumSourceDonorMassClosureErrorKg));
        if(std::abs(supplied.maximumSourceDonorMassClosureErrorKg-maximumSourceError)>sourceErrorTolerance)
            return fail("recorded source-carrier maximum source-donor closure metric differs from current flows");
        expanded.maximumSourceDonorMassClosureErrorKg=supplied.maximumSourceDonorMassClosureErrorKg;
    }else{
        expanded.maximumSourceDonorMassClosureErrorKg=maximumSourceError;
    }
    expanded.maximumTransportDistanceUsedM=maximumDistance;
    expanded.massWeightedMeanTransportDistanceM=totalMass>0.0L?static_cast<double>(weightedDistance/totalMass):0.0;
    expanded.minimumChildImpliedDensityKgM3=minimumDensity;
    expanded.maximumChildImpliedDensityKgM3=maximumDensity;
    expanded.maximumChildFractionSumError=0.0;
    std::string parentHash,childHash,boundsHash;
    spatial_mass_transport_detail::hashInputArrays(expandedInput,parentHash,childHash,boundsHash);
    expanded.parentDataSHA256=parentHash;
    expanded.childDataSHA256=fractional_semantic_transport_detail::bindSupportToChildHash(
        childHash,expanded.semanticSupportSHA256);
    expanded.materialBoundsSHA256=boundsHash;
    // The receipt string is evidence identity from the already pinned recorded
    // allocation. Feasibility is independently established below from every flow.
    expanded.receiptSHA256=suppliedTrace.expandedSolverReceiptSHA256;

    ConservativeSpatialMassTransportResult normalized=expanded;
    std::vector<std::uint32_t> zerosP(virtualToParent.size(),0u),zerosC(nc,0u);
    ConservativeSpatialMassTransportInput normalizedInput=expandedInput;
    normalizedInput.parentDonorBodyIndices=zerosP;
    normalizedInput.parentAnatomicalRegionIndices=zerosP;
    normalizedInput.childDonorBodyIndices=zerosC;
    normalizedInput.childAnatomicalRegionIndices=zerosC;
    normalized.usesFractionalSemanticSupport=false;
    normalized.semanticSupportSHA256.clear();
    normalized.childSemanticGroupOffsets.clear();normalized.childSemanticGroupIndices.clear();
    normalized.childSemanticVolumeFractions.clear();
    normalized.groupClosure.clear();
    std::map<std::uint32_t,double> normalizedParentGroups,normalizedChildGroups,
        normalizedMinimum,normalizedMaximum;
    for(std::size_t v=0;v<virtualToParent.size();++v)normalizedParentGroups[virtualMaterials[v]]+=virtualMasses[v];
    for(std::size_t c=0;c<nc;++c){
        const auto material=input.childMaterialIndices[c];
        normalizedChildGroups[material]+=computedChildMass[c];
        normalizedMinimum[material]+=input.childReferenceVolumeM3[c]*input.materialDensityLowerKgM3[material];
        normalizedMaximum[material]+=input.childReferenceVolumeM3[c]*input.materialDensityUpperKgM3[material];
    }
    for(const auto& [material,parentMass]:normalizedParentGroups)
        normalized.groupClosure.push_back({.materialIndex=material,.donorBodyIndex=0u,
            .anatomicalRegionIndex=0u,.parentMassKg=parentMass,.childMassKg=normalizedChildGroups[material],
            .childMinimumCapacityKg=normalizedMinimum[material],.childMaximumCapacityKg=normalizedMaximum[material]});
    normalized.sourceDonorClosure.clear();
    for(const auto& [key,parentMass]:parentSourceByDonor)
        normalized.sourceDonorClosure.push_back({.materialIndex=key.first,.sourceDonorBodyIndex=key.second,
            .parentMassKg=parentMass,.childMassKg=childSourceByDonor[key]});
    normalized.maximumSourceDonorMassClosureErrorKg=maximumSourceError;
    spatial_mass_transport_detail::hashInputArrays(normalizedInput,normalized.parentDataSHA256,
        normalized.childDataSHA256,normalized.materialBoundsSHA256);
    normalized.receiptSHA256=spatial_mass_transport_detail::receiptSHA256(normalizedInput,normalized);
    if(!verifyRecordedConservativeSpatialMassTransport(normalizedInput,normalized,error))
        return fail("recorded source-carrier mass/measure/density/donor feasibility failed: "+error);

    for(std::size_t group=0;group<support.groupMaterialIndices.size();++group){
        const double parentMass=actualGroupParentMass[group],childMass=actualGroupChildMass[group];
        const double massTolerance=std::max(input.absoluteMassToleranceKg,
            input.relativeClosureTolerance*std::max(std::abs(parentMass),std::abs(childMass)));
        if(std::abs(parentMass-childMass)>massTolerance)
            return fail("recorded source-carrier semantic donor/region group mass does not close");
        if(canonicalizeImportedMetadata){
            expanded.groupClosure.push_back({.materialIndex=support.groupMaterialIndices[group],
                .donorBodyIndex=support.donorBodyIndices[group],
                .anatomicalRegionIndex=support.anatomicalRegionIndices[group],
                .parentMassKg=parentMass,.childMassKg=childMass,
                .childMinimumCapacityKg=0.0,.childMaximumCapacityKg=childMass});
        }else{
            const auto& recorded=supplied.groupClosure[group];
            if(!std::isfinite(recorded.childMinimumCapacityKg)||recorded.childMinimumCapacityKg!=0.0||
               !std::isfinite(recorded.childMaximumCapacityKg)||recorded.childMaximumCapacityKg<0.0||
               recorded.childMaximumCapacityKg+massTolerance<childMass||
               recorded.materialIndex!=support.groupMaterialIndices[group]||
               recorded.donorBodyIndex!=support.donorBodyIndices[group]||
               recorded.anatomicalRegionIndex!=support.anatomicalRegionIndices[group]||
               std::abs(recorded.parentMassKg-parentMass)>massTolerance||
               std::abs(recorded.childMassKg-childMass)>massTolerance)
                return fail("recorded source-carrier semantic group closure or reachable capacity differs from current flows");
            expanded.groupClosure.push_back(recorded);
        }
    }
    if(canonicalizeImportedMetadata){
        expanded.childSemanticGroupOffsets.assign(nc+1u,0u);
        expanded.childSemanticGroupIndices.clear();expanded.childSemanticVolumeFractions.clear();
    }
    for(std::size_t c=0;c<nc;++c){
        if(canonicalizeImportedMetadata)
            expanded.childSemanticGroupOffsets[c]=static_cast<std::uint32_t>(expanded.childSemanticGroupIndices.size());
        const auto begin=canonicalizeImportedMetadata?0u:supplied.childSemanticGroupOffsets[c];
        const auto end=canonicalizeImportedMetadata?0u:supplied.childSemanticGroupOffsets[c+1u];
        if(!canonicalizeImportedMetadata&&(begin>=end||end>supplied.childSemanticGroupIndices.size()))
            return fail("recorded source-carrier semantic child row is empty or out of range");
        if(!canonicalizeImportedMetadata&&semanticMassByChild[c].size()!=static_cast<std::size_t>(end-begin))
            return fail("recorded source-carrier semantic group count differs from positive child flows");
        std::size_t at=begin;double fractionSum=0.0;
        for(const auto& [group,mass]:semanticMassByChild[c]){
            const double fraction=mass/computedChildMass[c];
            if(!canonicalizeImportedMetadata&&
               (supplied.childSemanticGroupIndices[at]!=group||
                supplied.childSemanticVolumeFractions[at]!=fraction))
                return fail("recorded source-carrier semantic fraction differs from its expanded flows");
            if(canonicalizeImportedMetadata){
                expanded.childSemanticGroupIndices.push_back(group);
                expanded.childSemanticVolumeFractions.push_back(fraction);
            }
            fractionSum+=fraction;++at;
        }
        if(canonicalizeImportedMetadata)
            expanded.childSemanticGroupOffsets[c+1u]=static_cast<std::uint32_t>(expanded.childSemanticGroupIndices.size());
        maximumFractionError=std::max(maximumFractionError,std::abs(fractionSum-1.0));
        if(std::abs(fractionSum-1.0)>1.0e-12)
            return fail("recorded source-carrier semantic child fractions do not sum to one");
    }

    ConservativeSpatialMassTransportResult expected;
    SourceCellCarrierTransportTrace expectedTrace;
    expectedTrace.carrierSourceParentRows=carrierParents;
    expectedTrace.originalCentroidNearestAdmissibleDistanceM=strictNearest;
    expectedTrace.expandedSolverReceiptSHA256=suppliedTrace.expandedSolverReceiptSHA256;
    if(!aggregateCarrierResult(input,support,carrier,virtualToParent,virtualToCarrier,
            expanded,expected,expectedTrace,error))return false;
    if(canonicalizeImportedMetadata){
        supplied=std::move(expected);suppliedTrace=std::move(expectedTrace);
        error.clear();return true;
    }
    if(!fractional_semantic_transport_detail::sameResult(expected,supplied)){
#define SOURCE_CARRIER_CHECK_FIELD(field) if(expected.field!=supplied.field) return fail("recorded source-carrier aggregate field differs from expanded feasible flow: " #field)
        SOURCE_CARRIER_CHECK_FIELD(method);
        SOURCE_CARRIER_CHECK_FIELD(childParentOffsets);
        SOURCE_CARRIER_CHECK_FIELD(childParentRows);
        SOURCE_CARRIER_CHECK_FIELD(childParentTransportMassKg);
        SOURCE_CARRIER_CHECK_FIELD(childParentTransportMeasureM3);
        SOURCE_CARRIER_CHECK_FIELD(childParentTransportDistanceM);
        SOURCE_CARRIER_CHECK_FIELD(childParentVolumeFractions);
        SOURCE_CARRIER_CHECK_FIELD(childAncestryMeasureVolumeM3);
        SOURCE_CARRIER_CHECK_FIELD(childMassKg);
        SOURCE_CARRIER_CHECK_FIELD(childImpliedDensityKgM3);
        SOURCE_CARRIER_CHECK_FIELD(childDominantParentRows);
        SOURCE_CARRIER_CHECK_FIELD(parentMassTransferredKg);
        SOURCE_CARRIER_CHECK_FIELD(parentMeasureTransferredM3);
        SOURCE_CARRIER_CHECK_FIELD(childSourceDonorOffsets);
        SOURCE_CARRIER_CHECK_FIELD(childSourceDonorBodyIndices);
        SOURCE_CARRIER_CHECK_FIELD(childSourceDonorMassKg);
        SOURCE_CARRIER_CHECK_FIELD(transportEdgeCount);
        SOURCE_CARRIER_CHECK_FIELD(augmentationCount);
        SOURCE_CARRIER_CHECK_FIELD(maximumParentMassClosureErrorKg);
        SOURCE_CARRIER_CHECK_FIELD(maximumParentMeasureClosureErrorM3);
        SOURCE_CARRIER_CHECK_FIELD(maximumSourceDonorMassClosureErrorKg);
        SOURCE_CARRIER_CHECK_FIELD(maximumChildFractionSumError);
        SOURCE_CARRIER_CHECK_FIELD(maximumTransportDistanceUsedM);
        SOURCE_CARRIER_CHECK_FIELD(massWeightedMeanTransportDistanceM);
        SOURCE_CARRIER_CHECK_FIELD(minimumChildImpliedDensityKgM3);
        SOURCE_CARRIER_CHECK_FIELD(maximumChildImpliedDensityKgM3);
        SOURCE_CARRIER_CHECK_FIELD(parentDataSHA256);
        SOURCE_CARRIER_CHECK_FIELD(childDataSHA256);
        SOURCE_CARRIER_CHECK_FIELD(materialBoundsSHA256);
        SOURCE_CARRIER_CHECK_FIELD(parentMassPacketSHA256);
        SOURCE_CARRIER_CHECK_FIELD(parentSourceContributionSHA256);
        SOURCE_CARRIER_CHECK_FIELD(childMeshSHA256);
        SOURCE_CARRIER_CHECK_FIELD(regionLabelSHA256);
        SOURCE_CARRIER_CHECK_FIELD(coordinateFrameSHA256);
        SOURCE_CARRIER_CHECK_FIELD(maximumTransportDistanceM);
        SOURCE_CARRIER_CHECK_FIELD(relativeClosureTolerance);
        SOURCE_CARRIER_CHECK_FIELD(absoluteMassToleranceKg);
        SOURCE_CARRIER_CHECK_FIELD(absoluteMeasureToleranceM3);
        SOURCE_CARRIER_CHECK_FIELD(maximumAllowedEdges);
        SOURCE_CARRIER_CHECK_FIELD(maximumAugmentations);
        SOURCE_CARRIER_CHECK_FIELD(receiptSHA256);
        SOURCE_CARRIER_CHECK_FIELD(semanticCandidateEdgeCount);
        SOURCE_CARRIER_CHECK_FIELD(usesFractionalSemanticSupport);
        SOURCE_CARRIER_CHECK_FIELD(semanticSupportSHA256);
        SOURCE_CARRIER_CHECK_FIELD(childSemanticGroupOffsets);
        SOURCE_CARRIER_CHECK_FIELD(childSemanticGroupIndices);
        SOURCE_CARRIER_CHECK_FIELD(childSemanticVolumeFractions);
#undef SOURCE_CARRIER_CHECK_FIELD
        if(expected.groupClosure.size()!=supplied.groupClosure.size())
            return fail("recorded source-carrier aggregate group-closure row count differs from expanded flow");
        for(std::size_t i=0;i<expected.groupClosure.size();++i){
            const auto& a=expected.groupClosure[i];const auto& b=supplied.groupClosure[i];
            if(a.materialIndex!=b.materialIndex||a.donorBodyIndex!=b.donorBodyIndex||
               a.anatomicalRegionIndex!=b.anatomicalRegionIndex||a.parentMassKg!=b.parentMassKg||
               a.childMassKg!=b.childMassKg||a.childMinimumCapacityKg!=b.childMinimumCapacityKg||
               a.childMaximumCapacityKg!=b.childMaximumCapacityKg)
                return fail("recorded source-carrier aggregate semantic group closure differs from expanded flow");
        }
        return fail("recorded source-carrier aggregate source-donor closure differs from expanded flow");
    }
    if(expectedTrace.sourceCarrierPacketSHA256!=suppliedTrace.sourceCarrierPacketSHA256||
       expectedTrace.sourceCarrierArraySHA256!=suppliedTrace.sourceCarrierArraySHA256||
       expectedTrace.expandedParentToSourceParentRows!=suppliedTrace.expandedParentToSourceParentRows||
       expectedTrace.expandedParentToCarrierRows!=suppliedTrace.expandedParentToCarrierRows||
       expectedTrace.carrierSourceParentRows!=suppliedTrace.carrierSourceParentRows||
       expectedTrace.originalCentroidNearestAdmissibleDistanceM!=suppliedTrace.originalCentroidNearestAdmissibleDistanceM||
       expectedTrace.childParentCarrierOffsets!=suppliedTrace.childParentCarrierOffsets||
       expectedTrace.childParentCarrierRows!=suppliedTrace.childParentCarrierRows||
       expectedTrace.childParentCarrierMassKg!=suppliedTrace.childParentCarrierMassKg||
       expectedTrace.childParentCarrierMeasureM3!=suppliedTrace.childParentCarrierMeasureM3||
       expectedTrace.childParentCarrierDistanceM!=suppliedTrace.childParentCarrierDistanceM||
       expectedTrace.receiptSHA256!=suppliedTrace.receiptSHA256)
        return fail("recorded source-carrier trace digest or arrays differ from the current feasible flow");
    error.clear();return true;
}

[[nodiscard]] inline bool verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
        const ConservativeSpatialMassTransportInput& input,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        const ConservativeSpatialMassTransportResult& supplied,
        const SourceCellCarrierTransportTrace& suppliedTrace,
        std::string& error) {
    auto result=supplied;auto trace=suppliedTrace;
    return finalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
        input,support,carrier,result,trace,false,error);
}

[[nodiscard]] inline bool canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
        const ConservativeSpatialMassTransportInput& input,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        ConservativeSpatialMassTransportResult& supplied,
        SourceCellCarrierTransportTrace& suppliedTrace,
        std::string& error) {
    return finalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
        input,support,carrier,supplied,suppliedTrace,true,error);
}

} // namespace metalrobo::human::coarse_tissue
