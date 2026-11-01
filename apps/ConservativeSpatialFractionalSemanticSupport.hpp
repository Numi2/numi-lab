#pragma once

// Optional semantic receiver partition for changed-topology child cells.
// The legacy conservative spatial transport remains the underlying solver.
// A supported real child is split into colocated virtual mass receivers whose
// reference volumes sum to the real cell volume; returned flows are aggregated
// back to the real child row before leaving this API.
#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include <iostream>

namespace metalrobo::human::coarse_tissue {

struct ConservativeSpatialChildSemanticSupport {
    // Child CSR addresses support slots; each slot references one global group.
    std::span<const std::uint32_t> childOffsets;
    std::span<const std::uint32_t> slotGroupIndices;
    std::span<const double> volumeFractions;

    // Optional exact per-child semantic-group allowlist. P1 slots above remain
    // the preference table; additional groups require an independently pinned
    // source-projection evidence hash and do not split or multiply child mass.
    std::span<const std::uint32_t> childAdmissibleGroupOffsets;
    std::span<const std::uint32_t> childAdmissibleGroupIndices;
    std::string_view admissibleGroupEvidenceSHA256;

    // Global semantic-group registry, keyed by (material, donor, region).
    std::span<const std::uint32_t> groupMaterialIndices;
    std::span<const std::uint32_t> donorBodyIndices;
    std::span<const std::uint32_t> anatomicalRegionIndices;
    std::span<const std::uint64_t> derivedGroupIdentityIDs;

    // Complete source-330 parent-tet and stable-face unions, once per group.
    std::span<const std::uint32_t> parentStableSourceFaceIdentities;
    std::span<const std::uint32_t> sourceParentTetOffsets;
    std::span<const std::uint32_t> sourceParentTetRows;
    std::span<const std::uint32_t> sourceStableFaceOffsets;
    std::span<const std::uint32_t> sourceStableFaceRows;
    std::string_view typedSupportSHA256;

    // Optional one-hop, same-material source-region adjacency prepared from
    // authenticated shared source-tet faces. P1 support remains unchanged and
    // supplies a soft preference; these rows extend admissible mass ancestry.
    std::span<const std::uint32_t> cellAdmissibleRegionOffsets;
    std::span<const std::uint32_t> cellAdmissibleRegionIds;
    std::string_view admissibilitySourceSupportSHA256;
    std::uint32_t admissibilityMaximumSourceRegionHops = 0u;
};

// Owning counterpart for imported/compiled bundles. Keep the large source-parent
// unions global; a per-child duplicate would multiply 330 rows by every child.
struct ConservativeSpatialChildSemanticSupportData {
    std::vector<std::uint32_t> childOffsets, slotGroupIndices;
    std::vector<double> volumeFractions;
    std::vector<std::uint32_t> childAdmissibleGroupOffsets, childAdmissibleGroupIndices;
    std::string admissibleGroupEvidenceSHA256;
    std::vector<std::uint32_t> groupMaterialIndices, donorBodyIndices, anatomicalRegionIndices;
    std::vector<std::uint64_t> derivedGroupIdentityIDs;
    std::vector<std::uint32_t> parentStableSourceFaceIdentities;
    std::vector<std::uint32_t> sourceParentTetOffsets, sourceParentTetRows;
    std::vector<std::uint32_t> sourceStableFaceOffsets, sourceStableFaceRows;
    std::string typedSupportSHA256;
    std::vector<std::uint32_t> cellAdmissibleRegionOffsets, cellAdmissibleRegionIds;
    std::string admissibilitySourceSupportSHA256;
    std::uint32_t admissibilityMaximumSourceRegionHops = 0u;

    [[nodiscard]] ConservativeSpatialChildSemanticSupport view() const {
        return {childOffsets, slotGroupIndices, volumeFractions,
            childAdmissibleGroupOffsets, childAdmissibleGroupIndices,
            admissibleGroupEvidenceSHA256, groupMaterialIndices,
            donorBodyIndices, anatomicalRegionIndices, derivedGroupIdentityIDs,
            parentStableSourceFaceIdentities, sourceParentTetOffsets, sourceParentTetRows,
            sourceStableFaceOffsets, sourceStableFaceRows, typedSupportSHA256,
            cellAdmissibleRegionOffsets, cellAdmissibleRegionIds,
            admissibilitySourceSupportSHA256, admissibilityMaximumSourceRegionHops};
    }
    [[nodiscard]] bool empty() const { return childOffsets.empty() && slotGroupIndices.empty(); }
};

namespace fractional_semantic_transport_detail {

using Group = std::pair<std::uint32_t, std::uint32_t>;

inline bool buildSourceParentGroupMap(
        const ConservativeSpatialChildSemanticSupport& support,
        const std::size_t parentCount,
        std::vector<std::uint32_t>& groupForParent) {
    constexpr auto invalid=std::numeric_limits<std::uint32_t>::max();
    if(support.sourceParentTetOffsets.size()!=support.groupMaterialIndices.size()+1u ||
       support.sourceParentTetOffsets.empty() || support.sourceParentTetOffsets.front()!=0u ||
       support.sourceParentTetOffsets.back()!=support.sourceParentTetRows.size()) return false;
    groupForParent.assign(parentCount,invalid);
    for(std::size_t group=0;group<support.groupMaterialIndices.size();++group){
        const auto begin=support.sourceParentTetOffsets[group];
        const auto end=support.sourceParentTetOffsets[group+1u];
        if(begin>end||end>support.sourceParentTetRows.size()) return false;
        for(std::uint32_t row=begin;row<end;++row){
            const auto parent=support.sourceParentTetRows[row];
            if(parent>=parentCount||groupForParent[parent]!=invalid) return false;
            groupForParent[parent]=static_cast<std::uint32_t>(group);
        }
    }
    return std::all_of(groupForParent.begin(),groupForParent.end(),
        [&](const auto group){return group!=invalid;});
}

inline bool childPrefersSemanticGroup(
        const ConservativeSpatialChildSemanticSupport& support,
        const std::uint32_t child,const std::uint32_t group) {
    if(child+1u>=support.childOffsets.size()) return false;
    const auto begin=support.childOffsets[child],end=support.childOffsets[child+1u];
    if(begin>end||end>support.slotGroupIndices.size()) return false;
    return std::find(support.slotGroupIndices.begin()+begin,
                     support.slotGroupIndices.begin()+end,group)!=
           support.slotGroupIndices.begin()+end;
}

inline bool childAllowsSourceRegion(
        const ConservativeSpatialChildSemanticSupport& support,
        const std::uint32_t child,const std::uint32_t region) {
    if(!support.cellAdmissibleRegionOffsets.empty()){
        if(child+1u>=support.cellAdmissibleRegionOffsets.size()) return false;
        const auto begin=support.cellAdmissibleRegionOffsets[child];
        const auto end=support.cellAdmissibleRegionOffsets[child+1u];
        if(begin>end||end>support.cellAdmissibleRegionIds.size()) return false;
        return std::binary_search(support.cellAdmissibleRegionIds.begin()+begin,
                                  support.cellAdmissibleRegionIds.begin()+end,region);
    }
    if(child+1u>=support.childOffsets.size()) return false;
    const auto begin=support.childOffsets[child],end=support.childOffsets[child+1u];
    if(begin>end||end>support.slotGroupIndices.size()) return false;
    for(auto row=begin;row<end;++row){
        const auto group=support.slotGroupIndices[row];
        if(group<support.anatomicalRegionIndices.size()&&
           support.anatomicalRegionIndices[group]==region) return true;
    }
    return false;
}

inline bool childAllowsSourceGroup(
        const ConservativeSpatialChildSemanticSupport& support,
        const std::uint32_t child,const std::uint32_t group) {
    if(group>=support.groupMaterialIndices.size()||
       group>=support.anatomicalRegionIndices.size()) return false;
    if(!support.childAdmissibleGroupOffsets.empty()||
       !support.childAdmissibleGroupIndices.empty()) {
        if(child+1u>=support.childAdmissibleGroupOffsets.size()) return false;
        const auto begin=support.childAdmissibleGroupOffsets[child];
        const auto end=support.childAdmissibleGroupOffsets[child+1u];
        if(begin>end||end>support.childAdmissibleGroupIndices.size()) return false;
        return std::binary_search(support.childAdmissibleGroupIndices.begin()+begin,
                                  support.childAdmissibleGroupIndices.begin()+end,group);
    }
    if(!support.cellAdmissibleRegionOffsets.empty())
        return childAllowsSourceRegion(support,child,support.anatomicalRegionIndices[group]);
    return childPrefersSemanticGroup(support,child,group);
}

struct ExpandedInput {
    std::vector<SpatialMassTransportPoint> centroids;
    std::vector<double> volumes;
    std::vector<std::uint32_t> materials, donors, regions;
    ConservativeSpatialMassTransportInput input;
    std::string supportSHA256;
};

inline std::string supportDigest(const ConservativeSpatialChildSemanticSupport& support) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("conservativeSpatialMassTransport.sharedCapacitySemanticSupport.v4");
    hash.string(support.typedSupportSHA256);
    hash.u64(support.childOffsets.size());
    for (auto value : support.childOffsets) hash.u32(value);
    hash.u64(support.slotGroupIndices.size());
    for (auto value : support.slotGroupIndices) hash.u32(value);
    for (auto value : support.volumeFractions) hash.real(value);
    hash.u64(support.childAdmissibleGroupOffsets.size());
    for (auto value : support.childAdmissibleGroupOffsets) hash.u32(value);
    hash.u64(support.childAdmissibleGroupIndices.size());
    for (auto value : support.childAdmissibleGroupIndices) hash.u32(value);
    hash.string(support.admissibleGroupEvidenceSHA256);
    hash.u64(support.groupMaterialIndices.size());
    for (auto value : support.groupMaterialIndices) hash.u32(value);
    for (auto value : support.donorBodyIndices) hash.u32(value);
    for (auto value : support.anatomicalRegionIndices) hash.u32(value);
    for (auto value : support.derivedGroupIdentityIDs) hash.u64(value);
    hash.u64(support.parentStableSourceFaceIdentities.size());
    for (auto value : support.parentStableSourceFaceIdentities) hash.u32(value);
    hash.u64(support.sourceParentTetOffsets.size());
    for (auto value : support.sourceParentTetOffsets) hash.u32(value);
    for (auto value : support.sourceParentTetRows) hash.u32(value);
    hash.u64(support.sourceStableFaceOffsets.size());
    for (auto value : support.sourceStableFaceOffsets) hash.u32(value);
    for (auto value : support.sourceStableFaceRows) hash.u32(value);
    hash.u64(support.cellAdmissibleRegionOffsets.size());
    for (auto value : support.cellAdmissibleRegionOffsets) hash.u32(value);
    hash.u64(support.cellAdmissibleRegionIds.size());
    for (auto value : support.cellAdmissibleRegionIds) hash.u32(value);
    hash.string(support.admissibilitySourceSupportSHA256);
    hash.u32(support.admissibilityMaximumSourceRegionHops);
    return hash.finish();
}

inline std::string bindSupportToChildHash(std::string_view childHash,
                                          std::string_view supportHash) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("conservativeSpatialMassTransport.fractional-child-data.v1");
    hash.string(childHash);
    hash.string(supportHash);
    return hash.finish();
}

inline std::string semanticReceipt(std::string_view baseReceipt,
                                   std::string_view supportHash) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("conservativeSpatialMassTransport.fractional-receipt.v1");
    hash.string(baseReceipt);
    hash.string(supportHash);
    return hash.finish();
}

inline bool expandInput(const ConservativeSpatialMassTransportInput& in,
                        const ConservativeSpatialChildSemanticSupport& support,
                        ExpandedInput& expanded,
                        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail = [&](std::string message) {
        error = std::move(message);
        return false;
    };
    const std::size_t nc = in.childReferenceVolumeM3.size();
    const std::size_t np = in.parentMassKg.size();
    const std::size_t ng = support.groupMaterialIndices.size();
    constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
    if (nc == 0u || np == 0u || ng == 0u || nc > invalid || np > invalid || ng > invalid ||
        in.childCentroidsWorld.size() != nc || in.childMaterialIndices.size() != nc ||
        in.childDonorBodyIndices.size() != nc || in.childAnatomicalRegionIndices.size() != nc ||
        in.parentMaterialIndices.size() != np || in.parentDonorBodyIndices.size() != np ||
        in.parentAnatomicalRegionIndices.size() != np ||
        support.childOffsets.size() != nc + 1u || support.childOffsets.empty() ||
        support.childOffsets.front() != 0u ||
        support.childOffsets.back() != support.slotGroupIndices.size() ||
        support.slotGroupIndices.size() != support.volumeFractions.size() ||
        support.slotGroupIndices.empty() || support.groupMaterialIndices.size() != ng ||
        support.donorBodyIndices.size() != ng || support.anatomicalRegionIndices.size() != ng ||
        support.derivedGroupIdentityIDs.size() != ng ||
        support.parentStableSourceFaceIdentities.size() != np ||
        support.sourceParentTetOffsets.size() != ng + 1u ||
        support.sourceParentTetOffsets.empty() || support.sourceParentTetOffsets.front() != 0u ||
        support.sourceParentTetOffsets.back() != support.sourceParentTetRows.size() ||
        support.sourceStableFaceOffsets.size() != ng + 1u ||
        support.sourceStableFaceOffsets.empty() || support.sourceStableFaceOffsets.front() != 0u ||
        support.sourceStableFaceOffsets.back() != support.sourceStableFaceRows.size() ||
        !validSHA256(support.typedSupportSHA256))
        return fail("fractional semantic support arrays, provenance CSR or proof digest are missing or malformed");
    if (support.childOffsets.back() > invalid ||
        support.sourceParentTetRows.empty() || support.sourceStableFaceRows.empty())
        return fail("fractional semantic support exceeds bounded index range or lacks source ancestry");
    for (std::size_t i = 1u; i < support.childOffsets.size(); ++i)
        if (support.childOffsets[i] < support.childOffsets[i - 1u] ||
            support.childOffsets[i] == support.childOffsets[i - 1u])
            return fail("fractional semantic child support CSR contains an empty or reversed child row");
    for (std::size_t i = 1u; i < support.sourceParentTetOffsets.size(); ++i)
        if (support.sourceParentTetOffsets[i] < support.sourceParentTetOffsets[i - 1u])
            return fail("fractional semantic group parent-tet CSR offsets are reversed");
    for (std::size_t i = 1u; i < support.sourceStableFaceOffsets.size(); ++i)
        if (support.sourceStableFaceOffsets[i] < support.sourceStableFaceOffsets[i - 1u])
            return fail("fractional semantic group stable-face CSR offsets are reversed");

    using SemanticTuple = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    std::set<SemanticTuple> uniqueTuples;
    std::set<std::uint64_t> uniqueIdentities;
    std::unordered_set<std::uint64_t> registeredMaterialRegions;
    registeredMaterialRegions.reserve(ng*2u);
    for (std::size_t group = 0u; group < ng; ++group) {
        const auto material = support.groupMaterialIndices[group];
        const auto donor = support.donorBodyIndices[group];
        const auto region = support.anatomicalRegionIndices[group];
        const auto identity = support.derivedGroupIdentityIDs[group];
        const auto pb = support.sourceParentTetOffsets[group];
        const auto pe = support.sourceParentTetOffsets[group + 1u];
        const auto fb = support.sourceStableFaceOffsets[group];
        const auto fe = support.sourceStableFaceOffsets[group + 1u];
        if (material == invalid || donor == invalid || region == invalid || identity == 0u ||
            material >= in.materialDensityLowerKgM3.size() || pb >= pe ||
            pe > support.sourceParentTetRows.size() || fb >= fe ||
            fe > support.sourceStableFaceRows.size() ||
            !uniqueTuples.emplace(material, donor, region).second ||
            !uniqueIdentities.insert(identity).second)
            return fail("fractional semantic group identity or complete source-tet/face CSR is invalid");
        std::set<std::uint32_t> expectedFaces;
        std::uint32_t priorParent = 0u;
        for (std::uint32_t row = pb; row < pe; ++row) {
            const auto parent = support.sourceParentTetRows[row];
            // Check membership before indexing any source-parent array.
            if (parent >= np || (row > pb && parent <= priorParent))
                return fail("fractional semantic source-tet CSR is out of range, repeated, or noncanonical");
            if (in.parentMaterialIndices[parent] != material ||
                in.parentDonorBodyIndices[parent] != donor ||
                in.parentAnatomicalRegionIndices[parent] != region ||
                support.parentStableSourceFaceIdentities[parent] == invalid)
                return fail("fractional semantic source-tet CSR crosses its registered material/donor/region group");
            priorParent = parent;
            expectedFaces.insert(support.parentStableSourceFaceIdentities[parent]);
        }
        if (expectedFaces.size() != static_cast<std::size_t>(fe - fb))
            return fail("fractional semantic stable-face CSR is not the complete source-tet face union");
        std::uint32_t priorFace = 0u;
        auto expected = expectedFaces.begin();
        for (std::uint32_t row = fb; row < fe; ++row) {
            const auto face = support.sourceStableFaceRows[row];
            if (face == invalid || (row > fb && face <= priorFace) ||
                expected == expectedFaces.end() || face != *expected)
                return fail("fractional semantic stable-face CSR differs from the complete source-tet face union");
            priorFace = face;
            ++expected;
        }
    }

    std::size_t slotCount = 0u;
    for (std::size_t child = 0u; child < nc; ++child) {
        const auto begin = support.childOffsets[child];
        const auto end = support.childOffsets[child + 1u];
        if (begin >= end || end > support.slotGroupIndices.size() ||
            in.childMaterialIndices[child] == invalid ||
            in.childDonorBodyIndices[child] == invalid ||
            in.childAnatomicalRegionIndices[child] == invalid ||
            !std::isfinite(in.childReferenceVolumeM3[child]) ||
            in.childReferenceVolumeM3[child] <= 0.0)
            return fail("fractional semantic child row has invalid labels, volume or support range");
        double fractionSum = 0.0;
        double maximumFraction = -1.0;
        bool foundPrimary = false;
        Group previous{};
        bool havePrevious = false;
        for (std::uint32_t slot = begin; slot < end; ++slot) {
            const auto group = support.slotGroupIndices[slot];
            if (group >= ng || support.groupMaterialIndices[group] != in.childMaterialIndices[child])
                return fail("fractional semantic child support references a missing or cross-material group");
            const Group key{support.donorBodyIndices[group], support.anatomicalRegionIndices[group]};
            if (havePrevious && !(previous < key))
                return fail("fractional semantic child support repeats or reorders a donor/region group");
            previous = key;
            havePrevious = true;
            const double fraction = support.volumeFractions[slot];
            if (!std::isfinite(fraction) || fraction <= 0.0 || fraction > 1.0)
                return fail("fractional semantic receiver volume share is not finite and strictly positive");
            fractionSum += fraction;
            maximumFraction = std::max(maximumFraction, fraction);
            if (key == Group{in.childDonorBodyIndices[child],
                             in.childAnatomicalRegionIndices[child]})
                foundPrimary = true;
            ++slotCount;
        }
        if (!foundPrimary)
            return fail("fractional semantic primary child label is absent from its registered support groups");
        if (fractionSum < 1.0 - 1.0e-12 || fractionSum > 1.0 + 1.0e-12)
            return fail("fractional semantic child volume shares do not sum to one");
        for (std::uint32_t slot = begin; slot < end; ++slot) {
            const auto group = support.slotGroupIndices[slot];
            if (support.donorBodyIndices[group] == in.childDonorBodyIndices[child] &&
                support.anatomicalRegionIndices[group] == in.childAnatomicalRegionIndices[child] &&
                support.volumeFractions[slot] != maximumFraction)
                return fail("fractional semantic primary child label is not a maximum-support group");
        }
    }
    if (slotCount != support.slotGroupIndices.size())
        return fail("fractional semantic support contains an unowned virtual receiver slot");

    expanded.centroids.reserve(slotCount);
    expanded.volumes.reserve(slotCount);
    expanded.materials.reserve(slotCount);
    expanded.donors.reserve(slotCount);
    expanded.regions.reserve(slotCount);
    for (std::size_t child = 0u; child < nc; ++child) {
        for (std::uint32_t slot = support.childOffsets[child];
             slot < support.childOffsets[child + 1u]; ++slot) {
            const auto group = support.slotGroupIndices[slot];
            expanded.centroids.push_back(in.childCentroidsWorld[child]);
            expanded.volumes.push_back(in.childReferenceVolumeM3[child] * support.volumeFractions[slot]);
            expanded.materials.push_back(support.groupMaterialIndices[group]);
            expanded.donors.push_back(support.donorBodyIndices[group]);
            expanded.regions.push_back(support.anatomicalRegionIndices[group]);
        }
    }
    expanded.input = in;
    expanded.input.childCentroidsWorld = expanded.centroids;
    expanded.input.childReferenceVolumeM3 = expanded.volumes;
    expanded.input.childMaterialIndices = expanded.materials;
    expanded.input.childDonorBodyIndices = expanded.donors;
    expanded.input.childAnatomicalRegionIndices = expanded.regions;
    expanded.supportSHA256 = supportDigest(support);
    error.clear();
    return true;
}

struct FlowRow {
    std::uint32_t parent = 0u;
    double mass = 0.0;
    double measure = 0.0;
    double distance = 0.0;
};

inline bool aggregateVirtualResult(const ConservativeSpatialMassTransportInput& realInput,
                                   const ConservativeSpatialChildSemanticSupport& support,
                                   const ExpandedInput& expanded,
                                   const ConservativeSpatialMassTransportResult& virtualResult,
                                   ConservativeSpatialMassTransportResult& out,
                                   std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail = [&](std::string message) { error = std::move(message); return false; };
    const std::size_t nc = realInput.childReferenceVolumeM3.size();
    out = virtualResult;
    out.childParentOffsets.assign(nc + 1u, 0u);
    out.childParentRows.clear();
    out.childParentTransportMassKg.clear();
    out.childParentTransportMeasureM3.clear();
    out.childParentTransportDistanceM.clear();
    out.childParentVolumeFractions.clear();
    out.childAncestryMeasureVolumeM3.assign(nc, 0.0);
    out.childMassKg.assign(nc, 0.0);
    out.childImpliedDensityKgM3.assign(nc, 0.0);
    out.childDominantParentRows.assign(nc, kInvalid);
    out.childSourceDonorOffsets.assign(nc + 1u, 0u);
    out.childSourceDonorBodyIndices.clear();
    out.childSourceDonorMassKg.clear();
    out.maximumChildFractionSumError = 0.0;
    out.minimumChildImpliedDensityKgM3 = std::numeric_limits<double>::infinity();
    out.maximumChildImpliedDensityKgM3 = 0.0;
    std::size_t slot = 0u;
    for (std::size_t child = 0; child < nc; ++child) {
        std::vector<FlowRow> rows;
        for (std::uint32_t supportSlot = support.childOffsets[child];
             supportSlot < support.childOffsets[child + 1u]; ++supportSlot, ++slot) {
            const auto begin = virtualResult.childParentOffsets[slot];
            const auto end = virtualResult.childParentOffsets[slot + 1u];
            for (std::uint32_t edge = begin; edge < end; ++edge)
                rows.push_back({virtualResult.childParentRows[edge],
                    virtualResult.childParentTransportMassKg[edge],
                    virtualResult.childParentTransportMeasureM3[edge],
                    virtualResult.childParentTransportDistanceM[edge]});
        }
        std::sort(rows.begin(), rows.end(), [](const FlowRow& a, const FlowRow& b) {
            return a.parent < b.parent;
        });
        if (rows.empty())
            return fail("fractional semantic aggregation produced a child without transported mass");
        for (std::size_t i = 1u; i < rows.size(); ++i)
            if (rows[i - 1u].parent == rows[i].parent)
                return fail("one source parent row was transported through multiple semantic slots of one child");
        out.childParentOffsets[child] = static_cast<std::uint32_t>(out.childParentRows.size());
        double mass = 0.0, measure = 0.0, fractionSum = 0.0, bestMeasure = -1.0;
        std::uint32_t dominant = kInvalid;
        std::map<std::uint32_t, double> sourceByDonor;
        for (const auto& row : rows) {
            if (row.parent >= realInput.parentMassKg.size())
                return fail("fractional semantic flow references an invalid parent row");
            mass += row.mass;
            measure += row.measure;
            out.childParentRows.push_back(row.parent);
            out.childParentTransportMassKg.push_back(row.mass);
            out.childParentTransportMeasureM3.push_back(row.measure);
            out.childParentTransportDistanceM.push_back(row.distance);
            if (row.measure > bestMeasure ||
                (row.measure == bestMeasure && row.parent < dominant)) {
                bestMeasure = row.measure;
                dominant = row.parent;
            }
            for (std::uint32_t source = realInput.parentSourceDonorOffsets[row.parent];
                 source < realInput.parentSourceDonorOffsets[row.parent + 1u]; ++source) {
                sourceByDonor[realInput.parentSourceDonorBodyIndices[source]] +=
                    row.mass * (realInput.parentSourceDonorMassKg[source] /
                                realInput.parentMassKg[row.parent]);
            }
        }
        for (std::size_t i = out.childParentRows.size() - rows.size();
             i < out.childParentRows.size(); ++i) {
            const double fraction = out.childParentTransportMeasureM3[i] / measure;
            if (!std::isfinite(fraction) || fraction <= 0.0)
                return fail("fractional semantic aggregate flow measure fraction is invalid");
            out.childParentVolumeFractions.push_back(fraction);
            fractionSum += fraction;
        }
        if (!std::isfinite(mass) || mass <= 0.0 || !std::isfinite(measure) || measure <= 0.0 ||
            std::abs(fractionSum - 1.0) > 5.0e-13)
            return fail("fractional semantic aggregate child flow fails mass/measure closure");
        out.childMassKg[child] = mass;
        out.childAncestryMeasureVolumeM3[child] = measure;
        out.childDominantParentRows[child] = dominant;
        out.maximumChildFractionSumError = std::max(out.maximumChildFractionSumError,
                                                     std::abs(fractionSum - 1.0));
        const auto material = realInput.childMaterialIndices[child];
        if (material >= realInput.materialDensityLowerKgM3.size())
            return fail("fractional semantic aggregate child material is outside the density domain");
        const double density = mass / realInput.childReferenceVolumeM3[child];
        const double lower = realInput.materialDensityLowerKgM3[material];
        const double upper = realInput.materialDensityUpperKgM3[material];
        const double tolerance = densityBoundRoundoff(lower, upper, density);
        if (!std::isfinite(density) || density < lower - tolerance || density > upper + tolerance)
            return fail("fractional semantic aggregate cell density is outside its actual material bounds");
        out.childImpliedDensityKgM3[child] = density;
        out.minimumChildImpliedDensityKgM3 =
            std::min(out.minimumChildImpliedDensityKgM3, density);
        out.maximumChildImpliedDensityKgM3 =
            std::max(out.maximumChildImpliedDensityKgM3, density);
        out.childSourceDonorOffsets[child] =
            static_cast<std::uint32_t>(out.childSourceDonorBodyIndices.size());
        double donorTotal = 0.0;
        for (const auto& [donor, donorMass] : sourceByDonor) {
            if (donor == kInvalid || !std::isfinite(donorMass) || donorMass <= 0.0)
                return fail("fractional semantic aggregate source-donor row is invalid");
            out.childSourceDonorBodyIndices.push_back(donor);
            out.childSourceDonorMassKg.push_back(donorMass);
            donorTotal += donorMass;
        }
        out.childSourceDonorOffsets[child + 1u] =
            static_cast<std::uint32_t>(out.childSourceDonorBodyIndices.size());
        if (std::abs(donorTotal - mass) >
            std::max(realInput.absoluteMassToleranceKg,
                     realInput.relativeClosureTolerance * mass))
            return fail("fractional semantic aggregate source-donor mass does not close cell mass");
    }
    out.childParentOffsets[nc] = static_cast<std::uint32_t>(out.childParentRows.size());
    out.childSourceDonorOffsets[nc] =
        static_cast<std::uint32_t>(out.childSourceDonorBodyIndices.size());
    out.transportEdgeCount = out.childParentRows.size();

    std::string parentHash, realChildHash, boundsHash;
    hashInputArrays(realInput, parentHash, realChildHash, boundsHash);
    out.parentDataSHA256 = std::move(parentHash);
    out.childDataSHA256 = bindSupportToChildHash(realChildHash, expanded.supportSHA256);
    out.materialBoundsSHA256 = std::move(boundsHash);
    const auto baseReceipt = receiptSHA256(realInput, out);
    out.receiptSHA256 = semanticReceipt(baseReceipt, expanded.supportSHA256);
    error.clear();
    return true;
}

inline bool inflateAggregatedResult(const ConservativeSpatialMassTransportInput& realInput,
                                    const ConservativeSpatialChildSemanticSupport& support,
                                    const ExpandedInput& expanded,
                                    const ConservativeSpatialMassTransportResult& supplied,
                                    ConservativeSpatialMassTransportResult& virtualResult,
                                    std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail = [&](std::string message) { error = std::move(message); return false; };
    const std::size_t nc = realInput.childReferenceVolumeM3.size();
    const std::size_t slotCount = expanded.volumes.size();
    if (supplied.childParentOffsets.size() != nc + 1u ||
        supplied.childParentOffsets.empty() || supplied.childParentOffsets.front() != 0u ||
        supplied.childParentOffsets.back() != supplied.childParentRows.size() ||
        supplied.childParentRows.empty() ||
        supplied.childParentTransportMassKg.size() != supplied.childParentRows.size() ||
        supplied.childParentTransportMeasureM3.size() != supplied.childParentRows.size() ||
        supplied.childParentTransportDistanceM.size() != supplied.childParentRows.size() ||
        supplied.childParentVolumeFractions.size() != supplied.childParentRows.size() ||
        supplied.childMassKg.size() != nc ||
        supplied.childAncestryMeasureVolumeM3.size() != nc ||
        supplied.childDominantParentRows.size() != nc)
        return fail("fractional semantic recorded aggregate flow CSR is incomplete or malformed");

    std::vector<std::vector<FlowRow>> rowsBySlot(slotCount);
    std::size_t slot = 0u;
    std::vector<std::map<Group, std::uint32_t>> groupToSlot(nc);
    for (std::size_t child = 0; child < nc; ++child) {
        for (std::uint32_t supportSlot = support.childOffsets[child];
             supportSlot < support.childOffsets[child + 1u]; ++supportSlot, ++slot) {
            const auto group = support.slotGroupIndices[supportSlot];
            groupToSlot[child].emplace(Group{support.donorBodyIndices[group],
                                             support.anatomicalRegionIndices[group]},
                                      static_cast<std::uint32_t>(slot));
        }
    }
    for (std::size_t child = 0; child < nc; ++child) {
        const auto begin = supplied.childParentOffsets[child];
        const auto end = supplied.childParentOffsets[child + 1u];
        if (begin >= end || end > supplied.childParentRows.size())
            return fail("fractional semantic real-child flow row is empty or out of range");
        std::uint32_t prior = 0u;
        double aggregateMass = 0.0, aggregateMeasure = 0.0, fractionSum = 0.0;
        std::uint32_t dominant = kInvalid;
        double bestMeasure = -1.0;
        for (std::uint32_t edge = begin; edge < end; ++edge) {
            const auto parent = supplied.childParentRows[edge];
            const double mass = supplied.childParentTransportMassKg[edge];
            const double measure = supplied.childParentTransportMeasureM3[edge];
            const double distanceValue = supplied.childParentTransportDistanceM[edge];
            if (parent >= realInput.parentMassKg.size() ||
                (edge > begin && parent <= prior) ||
                realInput.parentMaterialIndices[parent] != realInput.childMaterialIndices[child] ||
                !std::isfinite(mass) || mass <= 0.0 ||
                !std::isfinite(measure) || measure <= 0.0 ||
                !std::isfinite(distanceValue))
                return fail("fractional semantic recorded flow contains invalid ancestry or values");
            prior = parent;
            const Group group{realInput.parentDonorBodyIndices[parent],
                              realInput.parentAnatomicalRegionIndices[parent]};
            const auto found = groupToSlot[child].find(group);
            if (found == groupToSlot[child].end())
                return fail("fractional semantic recorded parent group is absent from the child support CSR");
            rowsBySlot[found->second].push_back({parent, mass, measure, distanceValue});
            aggregateMass += mass;
            aggregateMeasure += measure;
            fractionSum += supplied.childParentVolumeFractions[edge];
            if (measure > bestMeasure || (measure == bestMeasure && parent < dominant)) {
                bestMeasure = measure;
                dominant = parent;
            }
        }
        if (std::abs(aggregateMass - supplied.childMassKg[child]) >
                std::max(realInput.absoluteMassToleranceKg,
                         realInput.relativeClosureTolerance * aggregateMass) ||
            std::abs(aggregateMeasure - supplied.childAncestryMeasureVolumeM3[child]) >
                std::max(realInput.absoluteMeasureToleranceM3,
                         realInput.relativeClosureTolerance * aggregateMeasure) ||
            std::abs(fractionSum - 1.0) > 5.0e-13 ||
            supplied.childDominantParentRows[child] != dominant)
            return fail("fractional semantic aggregate mass, measure, fraction or dominant-parent summary differs from its flows");
    }

    virtualResult = supplied;
    virtualResult.childParentOffsets.assign(slotCount + 1u, 0u);
    virtualResult.childParentRows.clear();
    virtualResult.childParentTransportMassKg.clear();
    virtualResult.childParentTransportMeasureM3.clear();
    virtualResult.childParentTransportDistanceM.clear();
    virtualResult.childParentVolumeFractions.clear();
    virtualResult.childAncestryMeasureVolumeM3.assign(slotCount, 0.0);
    virtualResult.childMassKg.assign(slotCount, 0.0);
    virtualResult.childImpliedDensityKgM3.assign(slotCount, 0.0);
    virtualResult.childDominantParentRows.assign(slotCount, kInvalid);
    virtualResult.childSourceDonorOffsets.assign(slotCount + 1u, 0u);
    virtualResult.childSourceDonorBodyIndices.clear();
    virtualResult.childSourceDonorMassKg.clear();
    virtualResult.maximumChildFractionSumError = 0.0;
    virtualResult.minimumChildImpliedDensityKgM3 = std::numeric_limits<double>::infinity();
    virtualResult.maximumChildImpliedDensityKgM3 = 0.0;

    double maximumDistance = 0.0;
    long double weightedDistance = 0.0L, totalMass = 0.0L;
    std::size_t edgeCount = 0u;
    for (std::size_t s = 0; s < slotCount; ++s) {
        auto& rows = rowsBySlot[s];
        std::sort(rows.begin(), rows.end(), [](const FlowRow& a, const FlowRow& b) {
            return a.parent < b.parent;
        });
        if (rows.empty())
            return fail("fractional semantic recorded support slot has no source mass flow");
        for (std::size_t i = 1u; i < rows.size(); ++i)
            if (rows[i - 1u].parent == rows[i].parent)
                return fail("fractional semantic recorded support slot repeats a parent flow");
        virtualResult.childParentOffsets[s] =
            static_cast<std::uint32_t>(virtualResult.childParentRows.size());
        double mass = 0.0, measure = 0.0, fractionSum = 0.0, bestMeasure = -1.0;
        std::uint32_t dominant = kInvalid;
        std::map<std::uint32_t, double> sourceByDonor;
        for (const auto& row : rows) {
            mass += row.mass;
            measure += row.measure;
            virtualResult.childParentRows.push_back(row.parent);
            virtualResult.childParentTransportMassKg.push_back(row.mass);
            virtualResult.childParentTransportMeasureM3.push_back(row.measure);
            virtualResult.childParentTransportDistanceM.push_back(row.distance);
            if (row.measure > bestMeasure ||
                (row.measure == bestMeasure && row.parent < dominant)) {
                bestMeasure = row.measure;
                dominant = row.parent;
            }
            for (std::uint32_t source = realInput.parentSourceDonorOffsets[row.parent];
                 source < realInput.parentSourceDonorOffsets[row.parent + 1u]; ++source)
                sourceByDonor[realInput.parentSourceDonorBodyIndices[source]] +=
                    row.mass * (realInput.parentSourceDonorMassKg[source] /
                                realInput.parentMassKg[row.parent]);
            maximumDistance = std::max(maximumDistance, row.distance);
            weightedDistance += static_cast<long double>(row.mass) * row.distance;
            totalMass += row.mass;
            ++edgeCount;
        }
        for (const auto& row : rows) {
            const double fraction = row.measure / measure;
            virtualResult.childParentVolumeFractions.push_back(fraction);
            fractionSum += fraction;
        }
        virtualResult.childParentOffsets[s + 1u] =
            static_cast<std::uint32_t>(virtualResult.childParentRows.size());
        virtualResult.childAncestryMeasureVolumeM3[s] = measure;
        virtualResult.childMassKg[s] = mass;
        virtualResult.childDominantParentRows[s] = dominant;
        virtualResult.maximumChildFractionSumError =
            std::max(virtualResult.maximumChildFractionSumError, std::abs(fractionSum - 1.0));
        const double density = mass / expanded.volumes[s];
        virtualResult.childImpliedDensityKgM3[s] = density;
        virtualResult.minimumChildImpliedDensityKgM3 =
            std::min(virtualResult.minimumChildImpliedDensityKgM3, density);
        virtualResult.maximumChildImpliedDensityKgM3 =
            std::max(virtualResult.maximumChildImpliedDensityKgM3, density);
        virtualResult.childSourceDonorOffsets[s] =
            static_cast<std::uint32_t>(virtualResult.childSourceDonorBodyIndices.size());
        double donorTotal = 0.0;
        for (const auto& [donor, donorMass] : sourceByDonor) {
            virtualResult.childSourceDonorBodyIndices.push_back(donor);
            virtualResult.childSourceDonorMassKg.push_back(donorMass);
            donorTotal += donorMass;
        }
        if (std::abs(donorTotal - mass) >
            std::max(realInput.absoluteMassToleranceKg,
                     realInput.relativeClosureTolerance * mass))
            return fail("fractional semantic virtual source-donor mass does not close");
        virtualResult.childSourceDonorOffsets[s + 1u] =
            static_cast<std::uint32_t>(virtualResult.childSourceDonorBodyIndices.size());
    }
    virtualResult.childParentOffsets[slotCount] =
        static_cast<std::uint32_t>(virtualResult.childParentRows.size());
    virtualResult.childSourceDonorOffsets[slotCount] =
        static_cast<std::uint32_t>(virtualResult.childSourceDonorBodyIndices.size());
    virtualResult.transportEdgeCount = edgeCount;
    virtualResult.maximumTransportDistanceUsedM = maximumDistance;
    virtualResult.massWeightedMeanTransportDistanceM = totalMass > 0.0L
        ? static_cast<double>(weightedDistance / totalMass) : 0.0;
    std::string parentHash, childHash, boundsHash;
    hashInputArrays(expanded.input, parentHash, childHash, boundsHash);
    virtualResult.parentDataSHA256 = std::move(parentHash);
    virtualResult.childDataSHA256 = std::move(childHash);
    virtualResult.materialBoundsSHA256 = std::move(boundsHash);
    virtualResult.receiptSHA256 = receiptSHA256(expanded.input, virtualResult);
    error.clear();
    return true;
}

inline bool sameResult(const ConservativeSpatialMassTransportResult& a,
                       const ConservativeSpatialMassTransportResult& b) {
    if (a.method != b.method ||
        a.childParentOffsets != b.childParentOffsets ||
        a.childParentRows != b.childParentRows ||
        a.childParentTransportMassKg != b.childParentTransportMassKg ||
        a.childParentTransportMeasureM3 != b.childParentTransportMeasureM3 ||
        a.childParentTransportDistanceM != b.childParentTransportDistanceM ||
        a.childParentVolumeFractions != b.childParentVolumeFractions ||
        a.childAncestryMeasureVolumeM3 != b.childAncestryMeasureVolumeM3 ||
        a.childMassKg != b.childMassKg ||
        a.childImpliedDensityKgM3 != b.childImpliedDensityKgM3 ||
        a.childDominantParentRows != b.childDominantParentRows ||
        a.parentMassTransferredKg != b.parentMassTransferredKg ||
        a.parentMeasureTransferredM3 != b.parentMeasureTransferredM3 ||
        a.childSourceDonorOffsets != b.childSourceDonorOffsets ||
        a.childSourceDonorBodyIndices != b.childSourceDonorBodyIndices ||
        a.childSourceDonorMassKg != b.childSourceDonorMassKg ||
        a.transportEdgeCount != b.transportEdgeCount ||
        a.augmentationCount != b.augmentationCount ||
        a.maximumParentMassClosureErrorKg != b.maximumParentMassClosureErrorKg ||
        a.maximumParentMeasureClosureErrorM3 != b.maximumParentMeasureClosureErrorM3 ||
        a.maximumSourceDonorMassClosureErrorKg != b.maximumSourceDonorMassClosureErrorKg ||
        a.maximumChildFractionSumError != b.maximumChildFractionSumError ||
        a.maximumTransportDistanceUsedM != b.maximumTransportDistanceUsedM ||
        a.massWeightedMeanTransportDistanceM != b.massWeightedMeanTransportDistanceM ||
        a.minimumChildImpliedDensityKgM3 != b.minimumChildImpliedDensityKgM3 ||
        a.maximumChildImpliedDensityKgM3 != b.maximumChildImpliedDensityKgM3 ||
        a.parentDataSHA256 != b.parentDataSHA256 ||
        a.childDataSHA256 != b.childDataSHA256 ||
        a.materialBoundsSHA256 != b.materialBoundsSHA256 ||
        a.parentMassPacketSHA256 != b.parentMassPacketSHA256 ||
        a.parentSourceContributionSHA256 != b.parentSourceContributionSHA256 ||
        a.childMeshSHA256 != b.childMeshSHA256 ||
        a.regionLabelSHA256 != b.regionLabelSHA256 ||
        a.coordinateFrameSHA256 != b.coordinateFrameSHA256 ||
        a.maximumTransportDistanceM != b.maximumTransportDistanceM ||
        a.relativeClosureTolerance != b.relativeClosureTolerance ||
        a.absoluteMassToleranceKg != b.absoluteMassToleranceKg ||
        a.absoluteMeasureToleranceM3 != b.absoluteMeasureToleranceM3 ||
        a.maximumAllowedEdges != b.maximumAllowedEdges ||
        a.maximumAugmentations != b.maximumAugmentations ||
        a.receiptSHA256 != b.receiptSHA256 ||
        a.semanticCandidateEdgeCount != b.semanticCandidateEdgeCount ||
        a.usesFractionalSemanticSupport != b.usesFractionalSemanticSupport ||
        a.semanticSupportSHA256 != b.semanticSupportSHA256 ||
        a.childSemanticGroupOffsets != b.childSemanticGroupOffsets ||
        a.childSemanticGroupIndices != b.childSemanticGroupIndices ||
        a.childSemanticVolumeFractions != b.childSemanticVolumeFractions ||
        a.groupClosure.size() != b.groupClosure.size() ||
        a.sourceDonorClosure.size() != b.sourceDonorClosure.size())
        return false;
    for (std::size_t i = 0; i < a.groupClosure.size(); ++i) {
        const auto& x = a.groupClosure[i]; const auto& y = b.groupClosure[i];
        if (x.materialIndex != y.materialIndex || x.donorBodyIndex != y.donorBodyIndex ||
            x.anatomicalRegionIndex != y.anatomicalRegionIndex ||
            x.parentMassKg != y.parentMassKg || x.childMassKg != y.childMassKg ||
            x.childMinimumCapacityKg != y.childMinimumCapacityKg ||
            x.childMaximumCapacityKg != y.childMaximumCapacityKg) return false;
    }
    for (std::size_t i = 0; i < a.sourceDonorClosure.size(); ++i) {
        const auto& x = a.sourceDonorClosure[i]; const auto& y = b.sourceDonorClosure[i];
        if (x.materialIndex != y.materialIndex ||
            x.sourceDonorBodyIndex != y.sourceDonorBodyIndex ||
            x.parentMassKg != y.parentMassKg || x.childMassKg != y.childMassKg) return false;
    }
    return true;
}
inline bool validateSharedCapacitySupport(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialChildSemanticSupport& support,
        std::vector<std::uint32_t>& parentGroup,
        std::vector<std::map<std::uint32_t,double>>& childGroupPreference,
        std::vector<std::vector<std::uint32_t>>& childAdmissibleGroups,
        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    constexpr std::uint32_t invalid=std::numeric_limits<std::uint32_t>::max();
    const std::size_t np=in.parentMassKg.size(), nc=in.childReferenceVolumeM3.size();
    const std::size_t ng=support.groupMaterialIndices.size();
    if(np==0u||nc==0u||ng==0u||np>invalid||nc>invalid||ng>invalid||
       in.parentCentroidsWorld.size()!=np||in.parentReferenceVolumeM3.size()!=np||
       in.parentMaterialIndices.size()!=np||in.parentDonorBodyIndices.size()!=np||
       in.parentAnatomicalRegionIndices.size()!=np||
       in.parentSourceDonorOffsets.size()!=np+1u||in.parentSourceDonorOffsets.empty()||
       in.parentSourceDonorOffsets.front()!=0u||
       in.parentSourceDonorOffsets.back()!=in.parentSourceDonorBodyIndices.size()||
       in.parentSourceDonorBodyIndices.size()!=in.parentSourceDonorMassKg.size()||
       in.childCentroidsWorld.size()!=nc||in.childMaterialIndices.size()!=nc||
       in.childDonorBodyIndices.size()!=nc||in.childAnatomicalRegionIndices.size()!=nc||
       in.materialDensityLowerKgM3.empty()||
       in.materialDensityUpperKgM3.size()!=in.materialDensityLowerKgM3.size()||
       support.childOffsets.size()!=nc+1u||support.childOffsets.empty()||
       support.childOffsets.front()!=0u||
       support.childOffsets.back()!=support.slotGroupIndices.size()||
       support.slotGroupIndices.empty()||
       support.slotGroupIndices.size()!=support.volumeFractions.size()||
       support.groupMaterialIndices.size()!=ng||support.donorBodyIndices.size()!=ng||
       support.anatomicalRegionIndices.size()!=ng||support.derivedGroupIdentityIDs.size()!=ng||
       support.parentStableSourceFaceIdentities.size()!=np||
       support.sourceParentTetOffsets.size()!=ng+1u||support.sourceParentTetOffsets.empty()||
       support.sourceParentTetOffsets.front()!=0u||
       support.sourceParentTetOffsets.back()!=support.sourceParentTetRows.size()||
       support.sourceStableFaceOffsets.size()!=ng+1u||support.sourceStableFaceOffsets.empty()||
       support.sourceStableFaceOffsets.front()!=0u||
       support.sourceStableFaceOffsets.back()!=support.sourceStableFaceRows.size()||
       !validSHA256(support.typedSupportSHA256))
        return fail("shared-capacity semantic input arrays, source CSR or typed proof digest are malformed");
    if(!validSHA256(in.parentMassPacketSHA256)||
       !validSHA256(in.parentSourceContributionSHA256)||
       !validSHA256(in.childMeshSHA256)||!validSHA256(in.regionLabelSHA256)||
       !validSHA256(in.coordinateFrameSHA256)||
       !std::isfinite(in.maximumTransportDistanceM)||in.maximumTransportDistanceM<=0.0||
       !std::isfinite(in.relativeClosureTolerance)||in.relativeClosureTolerance<0.0||
       !std::isfinite(in.absoluteMassToleranceKg)||in.absoluteMassToleranceKg<0.0||
       !std::isfinite(in.absoluteMeasureToleranceM3)||in.absoluteMeasureToleranceM3<0.0||
       in.maximumAllowedEdges==0u||in.maximumAugmentations==0u)
        return fail("shared-capacity semantic input hashes, distance, tolerances or budgets are invalid");
    if(support.childOffsets.back()>invalid||
       support.sourceParentTetRows.empty()||support.sourceStableFaceRows.empty())
        return fail("shared-capacity semantic CSR exceeds the bounded index range or lacks parent ancestry");
    for(std::size_t i=1;i<support.childOffsets.size();++i)
        if(support.childOffsets[i]<=support.childOffsets[i-1u])
            return fail("shared-capacity child preference CSR has an empty, repeated or reversed row");
    for(std::size_t i=1;i<support.sourceParentTetOffsets.size();++i)
        if(support.sourceParentTetOffsets[i]<support.sourceParentTetOffsets[i-1u])
            return fail("shared-capacity source-parent offsets are reversed");
    for(std::size_t i=1;i<support.sourceStableFaceOffsets.size();++i)
        if(support.sourceStableFaceOffsets[i]<support.sourceStableFaceOffsets[i-1u])
            return fail("shared-capacity stable-face offsets are reversed");

    using SemanticTuple=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;
    std::set<SemanticTuple> uniqueTuples;
    std::set<std::uint64_t> uniqueIdentities;
    std::unordered_set<std::uint64_t> registeredMaterialRegions;
    registeredMaterialRegions.reserve(ng);
    parentGroup.assign(np,invalid);
    for(std::size_t g=0;g<ng;++g){
        const auto material=support.groupMaterialIndices[g];
        const auto donor=support.donorBodyIndices[g];
        const auto region=support.anatomicalRegionIndices[g];
        const auto identity=support.derivedGroupIdentityIDs[g];
        const auto pb=support.sourceParentTetOffsets[g],pe=support.sourceParentTetOffsets[g+1u];
        const auto fb=support.sourceStableFaceOffsets[g],fe=support.sourceStableFaceOffsets[g+1u];
        if(material==invalid||donor==invalid||region==invalid||identity==0u||
           material>=in.materialDensityLowerKgM3.size()||pb>=pe||pe>support.sourceParentTetRows.size()||
           fb>=fe||fe>support.sourceStableFaceRows.size()||
           !uniqueTuples.emplace(material,donor,region).second||
           !uniqueIdentities.insert(identity).second)
            return fail("shared-capacity semantic group identity or complete ancestry row is invalid");
        registeredMaterialRegions.insert((static_cast<std::uint64_t>(material)<<32u)|region);
        std::set<std::uint32_t> expectedFaces;
        std::uint32_t previousParent=0u;
        for(std::uint32_t row=pb;row<pe;++row){
            const auto p=support.sourceParentTetRows[row];
            if(p>=np||(row>pb&&p<=previousParent)||parentGroup[p]!=invalid)
                return fail("shared-capacity source-parent CSR is out of range, repeated or noncanonical");
            if(in.parentMaterialIndices[p]!=material||in.parentDonorBodyIndices[p]!=donor||
               in.parentAnatomicalRegionIndices[p]!=region||
               support.parentStableSourceFaceIdentities[p]==invalid)
                return fail("shared-capacity source-parent row crosses its typed material/donor/region group");
            parentGroup[p]=static_cast<std::uint32_t>(g);
            previousParent=p;
            expectedFaces.insert(support.parentStableSourceFaceIdentities[p]);
        }
        if(expectedFaces.size()!=static_cast<std::size_t>(fe-fb))
            return fail("shared-capacity stable-face CSR is not the complete parent-face union");
        auto face=expectedFaces.begin();
        std::uint32_t previousFace=0u;
        for(std::uint32_t row=fb;row<fe;++row){
            const auto value=support.sourceStableFaceRows[row];
            if(value==invalid||(row>fb&&value<=previousFace)||
               face==expectedFaces.end()||value!=*face)
                return fail("shared-capacity stable-face CSR differs from the complete parent-face union");
            previousFace=value;++face;
        }
    }
    for(std::size_t p=0;p<np;++p){
        const auto material=in.parentMaterialIndices[p];
        if(parentGroup[p]==invalid||material>=in.materialDensityLowerKgM3.size()||
           in.parentDonorBodyIndices[p]==invalid||in.parentAnatomicalRegionIndices[p]==invalid||
           !std::isfinite(in.parentReferenceVolumeM3[p])||in.parentReferenceVolumeM3[p]<=0.0||
           !std::isfinite(in.parentMassKg[p])||in.parentMassKg[p]<=0.0||
           !std::all_of(in.parentCentroidsWorld[p].begin(),in.parentCentroidsWorld[p].end(),
                        [](double x){return std::isfinite(x);}))
            return fail("shared-capacity source parent has invalid mass, volume, label, centroid or group");
        const double lower=in.materialDensityLowerKgM3[material];
        const double upper=in.materialDensityUpperKgM3[material];
        const double density=in.parentMassKg[p]/in.parentReferenceVolumeM3[p];
        const double tol=densityBoundRoundoff(lower,upper,density);
        if(!std::isfinite(lower)||!std::isfinite(upper)||lower<=0.0||lower>upper||
           !std::isfinite(density)||density<lower-tol||density>upper+tol)
            return fail("shared-capacity authoritative source-parent density is outside material bounds");
        const auto begin=in.parentSourceDonorOffsets[p],end=in.parentSourceDonorOffsets[p+1u];
        if(begin>=end||end>in.parentSourceDonorBodyIndices.size())
            return fail("shared-capacity parent source-donor CSR row is empty or out of range");
        std::uint32_t previousDonor=0u;double split=0.0;
        for(std::uint32_t row=begin;row<end;++row){
            const auto donor=in.parentSourceDonorBodyIndices[row];
            const double mass=in.parentSourceDonorMassKg[row];
            if(donor==invalid||(row>begin&&donor<=previousDonor)||
               !std::isfinite(mass)||mass<=0.0)
                return fail("shared-capacity source-donor contribution row is invalid or noncanonical");
            previousDonor=donor;split+=mass;
        }
        if(std::abs(split-in.parentMassKg[p])>
           std::max(in.absoluteMassToleranceKg,in.relativeClosureTolerance*in.parentMassKg[p]))
            return fail("shared-capacity source-donor contributions do not close the source-parent mass");
    }

    childGroupPreference.assign(nc,{});
    for(std::size_t c=0;c<nc;++c){
        const auto material=in.childMaterialIndices[c];
        const auto begin=support.childOffsets[c],end=support.childOffsets[c+1u];
        if(begin>=end||end>support.slotGroupIndices.size()||
           material>=in.materialDensityLowerKgM3.size()||
           in.childDonorBodyIndices[c]==invalid||in.childAnatomicalRegionIndices[c]==invalid||
           !std::isfinite(in.childReferenceVolumeM3[c])||in.childReferenceVolumeM3[c]<=0.0||
           !std::all_of(in.childCentroidsWorld[c].begin(),in.childCentroidsWorld[c].end(),
                        [](double x){return std::isfinite(x);}))
            return fail("shared-capacity child has invalid material, region, volume or centroid");
        const double lower=in.materialDensityLowerKgM3[material];
        const double upper=in.materialDensityUpperKgM3[material];
        if(!std::isfinite(lower)||!std::isfinite(upper)||lower<=0.0||lower>upper)
            return fail("shared-capacity child material density bounds are invalid");
        std::uint32_t previousGroup=0u;bool havePrevious=false;
        double scoreSum=0.0,maximumScore=-1.0,primaryScore=-1.0;
        bool foundPrimaryGroup=false;
        for(std::uint32_t row=begin;row<end;++row){
            const auto g=support.slotGroupIndices[row];
            const double score=support.volumeFractions[row];
            if(g>=ng||support.groupMaterialIndices[g]!=material||
               (havePrevious&&g<=previousGroup)||!std::isfinite(score)||score<=0.0||score>1.0)
                return fail("shared-capacity child P1 region-support score/group row is invalid");
            previousGroup=g;havePrevious=true;
            childGroupPreference[c][g]+=score;
            maximumScore=std::max(maximumScore,score);
            if(support.donorBodyIndices[g]==in.childDonorBodyIndices[c] &&
               support.anatomicalRegionIndices[g]==in.childAnatomicalRegionIndices[c]){
                foundPrimaryGroup=true;
                primaryScore=score;
            }
            scoreSum+=score;
        }
        if(std::abs(scoreSum-1.0)>1.0e-12||!foundPrimaryGroup||primaryScore!=maximumScore)
            return fail("shared-capacity child exact P1 tuple support does not sum to one or its primary donor/region tuple is not a maximum-share group");
    }

    const bool hasRegionAdmissibility = !support.cellAdmissibleRegionOffsets.empty() ||
        !support.cellAdmissibleRegionIds.empty() ||
        !support.admissibilitySourceSupportSHA256.empty() ||
        support.admissibilityMaximumSourceRegionHops != 0u;
    const bool hasExplicitGroupAllowlist = !support.childAdmissibleGroupOffsets.empty() ||
        !support.childAdmissibleGroupIndices.empty();
    if (!hasExplicitGroupAllowlist && !support.admissibleGroupEvidenceSHA256.empty())
        return fail("shared-capacity group evidence hash is present without its allowlist CSR");
    if (hasExplicitGroupAllowlist &&
        (support.childAdmissibleGroupOffsets.size()!=nc+1u ||
         support.childAdmissibleGroupOffsets.empty() ||
         support.childAdmissibleGroupOffsets.front()!=0u ||
         support.childAdmissibleGroupOffsets.back()!=support.childAdmissibleGroupIndices.size()))
        return fail("shared-capacity explicit exact-group allowlist CSR is malformed");
    if (hasExplicitGroupAllowlist)
        for (std::size_t c=1u;c<support.childAdmissibleGroupOffsets.size();++c)
            if (support.childAdmissibleGroupOffsets[c]<=support.childAdmissibleGroupOffsets[c-1u])
                return fail("shared-capacity explicit exact-group allowlist has an empty or reversed row");

    std::vector<std::vector<std::uint32_t>> allowedRegionsByChild(nc);
    if(hasRegionAdmissibility){
        if(support.cellAdmissibleRegionOffsets.size()!=nc+1u ||
           support.cellAdmissibleRegionOffsets.empty() ||
           support.cellAdmissibleRegionOffsets.front()!=0u ||
           support.cellAdmissibleRegionOffsets.back()!=support.cellAdmissibleRegionIds.size() ||
           support.cellAdmissibleRegionIds.empty() ||
           !validSHA256(support.admissibilitySourceSupportSHA256) ||
           support.admissibilityMaximumSourceRegionHops!=1u)
            return fail("shared-capacity one-hop admissibility CSR or source proof is malformed");
        for(std::size_t c=0;c<nc;++c){
            const auto begin=support.cellAdmissibleRegionOffsets[c];
            const auto end=support.cellAdmissibleRegionOffsets[c+1u];
            if(begin>=end||end>support.cellAdmissibleRegionIds.size())
                return fail("shared-capacity admissible-region row is empty or out of range");
            auto& allowedRegions=allowedRegionsByChild[c];
            allowedRegions.reserve(end-begin);
            std::uint32_t previous=0u;
            for(std::uint32_t row=begin;row<end;++row){
                const auto region=support.cellAdmissibleRegionIds[row];
                if(region==invalid||(row>begin&&region<=previous))
                    return fail("shared-capacity admissible-region CSR is not sorted unique");
                const auto material=in.childMaterialIndices[c];
                const auto key=(static_cast<std::uint64_t>(material)<<32u)|region;
                if(!registeredMaterialRegions.contains(key))
                    return fail("shared-capacity admissible region has no same-material source semantic group");
                allowedRegions.push_back(region);previous=region;
            }
        }
    }else if(!support.admissibilitySourceSupportSHA256.empty()||
             support.admissibilityMaximumSourceRegionHops!=0u){
        return fail("shared-capacity legacy P1 support has stray adjacency proof metadata");
    }

    childAdmissibleGroups.assign(nc,{});
    bool hasAdditionalExactGroup=false;
    for(std::size_t c=0;c<nc;++c){
        auto& allowed=childAdmissibleGroups[c];
        if(hasExplicitGroupAllowlist){
            const auto begin=support.childAdmissibleGroupOffsets[c];
            const auto end=support.childAdmissibleGroupOffsets[c+1u];
            if(begin>=end||end>support.childAdmissibleGroupIndices.size())
                return fail("shared-capacity explicit exact-group allowlist row is empty or out of range");
            std::uint32_t previous=0u;
            for(std::uint32_t row=begin;row<end;++row){
                const auto group=support.childAdmissibleGroupIndices[row];
                if(group>=ng||(row>begin&&group<=previous)||
                   support.groupMaterialIndices[group]!=in.childMaterialIndices[c])
                    return fail("shared-capacity explicit exact-group allowlist has an invalid, repeated, unordered or cross-material tuple");
                if(hasRegionAdmissibility&&!std::binary_search(allowedRegionsByChild[c].begin(),
                    allowedRegionsByChild[c].end(),support.anatomicalRegionIndices[group]))
                    return fail("shared-capacity exact group is outside the authenticated one-hop region support");
                hasAdditionalExactGroup|=!childGroupPreference[c].contains(group);
                allowed.push_back(group);previous=group;
            }
            for(const auto& [group,score]:childGroupPreference[c]){
                (void)score;
                if(!std::binary_search(allowed.begin(),allowed.end(),group))
                    return fail("shared-capacity explicit exact-group allowlist omits an original P1 support tuple");
            }
        }else{
            for(const auto& [group,score]:childGroupPreference[c]){
                (void)score;
                if(hasRegionAdmissibility&&!std::binary_search(allowedRegionsByChild[c].begin(),
                    allowedRegionsByChild[c].end(),support.anatomicalRegionIndices[group]))
                    return fail("shared-capacity admissible-region CSR omits an original exact P1 tuple region");
                allowed.push_back(group);
            }
        }
    }
    if(hasAdditionalExactGroup&&!validSHA256(support.admissibleGroupEvidenceSHA256))
        return fail("shared-capacity additional exact-group slots lack a pinned source-projection evidence hash");
    error.clear();return true;
}

struct SharedCapacityFlow {
    std::uint32_t parent=0u,child=0u;
    double mass=0.0,measure=0.0,distance=0.0;
};

inline std::string sharedCapacityReceipt(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialMassTransportResult& result,
        const std::string_view supportHash) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("conservativeSpatialMassTransport.sharedRealChildCapacity.sparseDinicFeasibility.v2");
    hash.string(spatial_mass_transport_detail::receiptSHA256(in,result));
    hash.string(supportHash);
    hash.string("allocation_solver=deterministic_dinic_feasibility");
    hash.string("cost_optimality_certified=false");
    hash.u64(result.semanticCandidateEdgeCount);
    hash.u64(result.childSemanticGroupIndices.size());
    for(std::size_t c=0;c+1u<result.childSemanticGroupOffsets.size();++c){
        hash.u32(result.childSemanticGroupOffsets[c]);
        hash.u32(result.childSemanticGroupOffsets[c+1u]);
        for(std::uint32_t row=result.childSemanticGroupOffsets[c];
            row<result.childSemanticGroupOffsets[c+1u];++row){
            hash.u32(result.childSemanticGroupIndices[row]);
            hash.real(result.childSemanticVolumeFractions[row]);
        }
    }
    return hash.finish();
}

inline bool assembleSharedRealChildCapacity(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialChildSemanticSupport& support,
        ConservativeSpatialMassTransportResult& out,
        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    struct SharedCapacityPerfCounters {
        const char* stage="support_validation";
        std::size_t seedEdges=0u,cutRounds=0u,cutEdgesAdded=0u;
        std::size_t augmentCalls=0u,flowAugmentations=0u;
        ~SharedCapacityPerfCounters(){
            std::cerr<<"shared_capacity_perf stage="<<stage
                <<" seed_edges="<<seedEdges
                <<" cut_rounds="<<cutRounds
                <<" cut_edges_added="<<cutEdgesAdded
                <<" augment_calls="<<augmentCalls
                <<" flow_augmentations="<<flowAugmentations<<"\n";
        }
    } perf;
    constexpr std::uint32_t invalid=std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> parentGroup;
    std::vector<std::map<std::uint32_t,double>> groupPreference;
    std::vector<std::vector<std::uint32_t>> childAdmissibleGroups;
    if(!validateSharedCapacitySupport(in,support,parentGroup,groupPreference,
                                      childAdmissibleGroups,error))return false;
    perf.stage="bucket_index";
    const std::size_t np=in.parentMassKg.size(),nc=in.childReferenceVolumeM3.size();
    if(np+nc+3u>std::numeric_limits<std::uint32_t>::max())
        return fail("shared-capacity flow node count exceeds its bounded index range");
    const std::uint32_t parentBase=0u;
    const std::uint32_t childBase=static_cast<std::uint32_t>(np);
    const std::uint32_t sink=static_cast<std::uint32_t>(np+nc);
    const std::uint32_t superSource=sink+1u,superSink=sink+2u;
    const double radiusSquared=in.maximumTransportDistanceM*in.maximumTransportDistanceM;
    if(!std::isfinite(radiusSquared)||radiusSquared<=0.0)
        return fail("shared-capacity transport radius squared is invalid");

    using Bucket=std::tuple<std::uint32_t,GridKey>;
    std::map<Bucket,std::vector<std::uint32_t>> childrenByGroupBucket;
    std::map<Bucket,std::vector<std::uint32_t>> parentsByGroupBucket;
    std::vector<GridKey> parentGrid(np),childGrid(nc);
    for(std::size_t p=0;p<np;++p){
        if(!gridKey(in.parentCentroidsWorld[p],in.maximumTransportDistanceM,parentGrid[p]))
            return fail("shared-capacity parent centroid cannot be indexed by the bounded search grid");
        parentsByGroupBucket[{parentGroup[p],parentGrid[p]}]
            .push_back(static_cast<std::uint32_t>(p));
    }
    for(std::size_t c=0;c<nc;++c){
        if(!gridKey(in.childCentroidsWorld[c],in.maximumTransportDistanceM,childGrid[c]))
            return fail("shared-capacity child centroid cannot be indexed by the bounded search grid");
        for(const auto group:childAdmissibleGroups[c])
            childrenByGroupBucket[{group,childGrid[c]}]
                .push_back(static_cast<std::uint32_t>(c));
    }

    ExactDoubleSum totalSupplyExact,totalMinimumExact,totalMaximumExact;
    long double supplyLong=0.0L,minimumLong=0.0L,maximumLong=0.0L;
    std::vector<double> childMinimum(nc),childMaximum(nc);
    for(double mass:in.parentMassKg){supplyLong+=mass;totalSupplyExact.add(mass);}
    for(std::size_t c=0;c<nc;++c){
        const auto material=in.childMaterialIndices[c];
        childMinimum[c]=in.childReferenceVolumeM3[c]*in.materialDensityLowerKgM3[material];
        childMaximum[c]=in.childReferenceVolumeM3[c]*in.materialDensityUpperKgM3[material];
        if(!std::isfinite(childMinimum[c])||!std::isfinite(childMaximum[c])||
           childMinimum[c]<=0.0||childMaximum[c]<childMinimum[c])
            return fail("shared-capacity real-child density capacity is invalid");
        minimumLong+=childMinimum[c];maximumLong+=childMaximum[c];
        totalMinimumExact.add(childMinimum[c]);totalMaximumExact.add(childMaximum[c]);
    }
    const double supply=static_cast<double>(supplyLong);
    const double minimum=static_cast<double>(minimumLong);
    const double maximum=static_cast<double>(maximumLong);
    auto supplyMinusMinimum=totalSupplyExact;
    auto supplyMinusMaximum=totalSupplyExact;
    for(std::size_t c=0;c<nc;++c){
        supplyMinusMinimum.add(-childMinimum[c]);
        supplyMinusMaximum.add(-childMaximum[c]);
    }
    if(!totalSupplyExact.valid()||!totalMinimumExact.valid()||!totalMaximumExact.valid()||
       !supplyMinusMinimum.valid()||!supplyMinusMaximum.valid()||
       supplyMinusMinimum.sign()<0||supplyMinusMaximum.sign()>0)
        return fail("shared-capacity total source mass is outside the real-child density capacity bounds");
    const double residualCapacity=supplyMinusMinimum.roundedDownPositive();

    MinCostFlow network(static_cast<std::size_t>(superSink)+1u);
    std::vector<std::uint32_t> parentSourceArcs(np);
    for(std::size_t p=0;p<np;++p)
        parentSourceArcs[p]=network.addEdge(superSource,
            parentBase+static_cast<std::uint32_t>(p),in.parentMassKg[p],0.0);
    std::vector<std::uint32_t> minimumArcs(nc);
    for(std::size_t c=0;c<nc;++c)
        minimumArcs[c]=network.addEdge(childBase+static_cast<std::uint32_t>(c),superSink,
                                      childMinimum[c],0.0);

    struct PendingEdge {std::uint32_t parent,child;double distance,cost;};
    std::vector<PendingEdge> pending;
    pending.reserve(std::min(in.maximumAllowedEdges,nc+np*2u));
    std::vector<std::vector<std::uint32_t>> activeChildrenByParent(np);
    std::vector<std::vector<std::uint32_t>> reachableChildren(support.groupMaterialIndices.size());
    const double semanticPreferenceWeight=1.0e-5;
    constexpr std::size_t parentSeedNearestK=4u;
    constexpr std::size_t childSeedNearestK=1u;
    constexpr std::size_t cutExpansionBatch=65536u;
    const auto candidateLess=[](const Candidate& a,const Candidate& b){
        return a.distanceM!=b.distanceM?a.distanceM<b.distanceM:a.child<b.child;
    };
    const auto keepNearest=[&](std::vector<Candidate>& row,const Candidate& value,
                               const std::size_t limit){
        const auto duplicate=std::find_if(row.begin(),row.end(),[&](const Candidate& x){
            return x.child==value.child;
        });
        if(duplicate!=row.end()){
            if(candidateLess(value,*duplicate))*duplicate=value;
            std::sort(row.begin(),row.end(),candidateLess);
            return;
        }
        const auto position=std::lower_bound(row.begin(),row.end(),value,candidateLess);
        row.insert(position,value);
        if(row.size()>limit)row.pop_back();
    };
    const auto addSeed=[&](const std::uint32_t p,const Candidate& candidate){
        const auto c=candidate.child;
        const auto group=parentGroup[p];
        const auto found=groupPreference[c].find(group);
        const double penalty=found==groupPreference[c].end()?0.0:
            semanticPreferenceWeight*(-std::log(found->second));
        const double cost=(candidate.distanceM*candidate.distanceM)/radiusSquared+penalty;
        if(!std::isfinite(cost)||cost<0.0)return false;
        pending.push_back({p,c,candidate.distanceM,cost});
        return true;
    };
    const auto scanNearestChildForParent=[&](const std::uint32_t p,const std::size_t k){
        std::vector<Candidate> nearest;
        const auto group=parentGroup[p];
        const auto base=parentGrid[p];
        for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)for(int dz=-1;dz<=1;++dz){
            const GridKey key{base.x+dx,base.y+dy,base.z+dz};
            const auto found=childrenByGroupBucket.find({group,key});
            if(found==childrenByGroupBucket.end())continue;
            for(const auto c:found->second){
                const double d=distance(in.parentCentroidsWorld[p],in.childCentroidsWorld[c]);
                if(d<=in.maximumTransportDistanceM&&d*d<=radiusSquared*(1.0+1.0e-14))
                    keepNearest(nearest,Candidate{c,d},k);
            }
        }
        return nearest;
    };
    const auto scanNearestParentForChildGroup=[&](const std::uint32_t c,
                                                    const std::uint32_t group,
                                                    const std::size_t k){
        std::vector<Candidate> nearest;
        const auto base=childGrid[c];
        for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)for(int dz=-1;dz<=1;++dz){
            const GridKey key{base.x+dx,base.y+dy,base.z+dz};
            const auto found=parentsByGroupBucket.find({group,key});
            if(found==parentsByGroupBucket.end())continue;
            for(const auto p:found->second){
                const double d=distance(in.parentCentroidsWorld[p],in.childCentroidsWorld[c]);
                if(d<=in.maximumTransportDistanceM&&d*d<=radiusSquared*(1.0+1.0e-14))
                    keepNearest(nearest,Candidate{p,d},k);
            }
        }
        return nearest;
    };

    // Sparse seed: each source parent receives its nearest admissible child,
    // and each real child receives its nearest admissible parent in every
    // authenticated exact material/donor/region support row. All child minimum-density
    // demand is therefore represented before any flow attempt.
    for(std::size_t p=0;p<np;++p){
        const auto nearest=scanNearestChildForParent(static_cast<std::uint32_t>(p),parentSeedNearestK);
        if(nearest.empty()){
            std::ostringstream witness;
            witness<<"INFEASIBLE shared-capacity parent has no same-material/donor/region exact-group-supported child within distance bound"
                <<" parent="<<p<<" material="<<in.parentMaterialIndices[p]
                <<" region="<<in.parentAnatomicalRegionIndices[p]
                <<" radius_m="<<in.maximumTransportDistanceM;
            return fail(witness.str());
        }
        for(const auto& candidate:nearest)
            if(!addSeed(static_cast<std::uint32_t>(p),candidate))
                return fail("shared-capacity sparse seed cost is invalid");
    }
    std::vector<bool> childHasSeed(nc,false);
    for(std::size_t c=0;c<nc;++c){
        for(const auto group:childAdmissibleGroups[c]){
            const auto nearest=scanNearestParentForChildGroup(static_cast<std::uint32_t>(c),group,
                                                               childSeedNearestK);
            for(const auto& candidate:nearest){
                childHasSeed[c]=true;
                if(!addSeed(candidate.child,Candidate{static_cast<std::uint32_t>(c),candidate.distanceM}))
                    return fail("shared-capacity sparse reverse seed cost is invalid");
            }
        }
        if(!childHasSeed[c]){
            std::ostringstream witness;
            witness<<"INFEASIBLE shared-capacity real child has no same-material/donor/region exact-group-supported parent within distance bound"
                <<" child="<<c<<" material="<<in.childMaterialIndices[c]
                <<" allowed_exact_tuples="<<childAdmissibleGroups[c].size()
                <<" radius_m="<<in.maximumTransportDistanceM;
            return fail(witness.str());
        }
    }
    std::sort(pending.begin(),pending.end(),[](const auto& a,const auto& b){
        return a.parent!=b.parent?a.parent<b.parent:a.child<b.child;
    });
    pending.erase(std::unique(pending.begin(),pending.end(),[](const auto& a,const auto& b){
        return a.parent==b.parent&&a.child==b.child;
    }),pending.end());
    // Dinic is feasibility-only; deterministic edge insertion order keeps the
    // distance/P1 penalty as a preference without claiming a global optimum.
    std::sort(pending.begin(),pending.end(),[](const auto& a,const auto& b){
        if(a.cost!=b.cost)return a.cost<b.cost;
        if(a.parent!=b.parent)return a.parent<b.parent;
        return a.child<b.child;
    });
    // Feasibility Dinic is deterministic but not cost-optimal. Insert seeded
    // candidate edges in distance/P1 preference order, then parent/child order.
    std::sort(pending.begin(),pending.end(),[](const auto& a,const auto& b){
        if(a.cost!=b.cost)return a.cost<b.cost;
        if(a.parent!=b.parent)return a.parent<b.parent;
        return a.child<b.child;
    });
    if(pending.size()>in.maximumAllowedEdges)
        return fail("INCONCLUSIVE shared-capacity sparse seed graph exceeds pinned edge budget");
    perf.seedEdges=pending.size();
    perf.stage="initial_flow";

    std::vector<EdgeRef> references;
    references.reserve(std::min(in.maximumAllowedEdges,nc+np*2u));
    for(const auto& edge:pending){
        const auto from=parentBase+edge.parent;
        const auto arc=network.addEdge(from,childBase+edge.child,in.parentMassKg[edge.parent],edge.cost);
        references.push_back({edge.parent,edge.child,from,arc,edge.distance});
        activeChildrenByParent[edge.parent].push_back(edge.child);
        reachableChildren[parentGroup[edge.parent]].push_back(edge.child);
    }
    for(auto& row:activeChildrenByParent){
        std::sort(row.begin(),row.end());
        row.erase(std::unique(row.begin(),row.end()),row.end());
    }

    std::size_t augmentationCount=0u;
    std::string flowError;
    const auto augmentToMaximum=[&](){
        std::size_t roundAugmentations=0u;double sent=0.0;
        const std::size_t nextCall=perf.augmentCalls+1u;
        if(nextCall<=16u || nextCall%16u==0u){
            std::cerr<<"shared_capacity_progress phase=before_run_feasible"
                <<" stage="<<perf.stage
                <<" seed_edges="<<perf.seedEdges
                <<" cut_rounds="<<perf.cutRounds
                <<" cut_edges_added="<<perf.cutEdgesAdded
                <<" augment_calls="<<nextCall
                <<" prior_flow_augmentations="<<perf.flowAugmentations
                <<std::endl;
        }
        ++perf.augmentCalls;
        if(!network.runFeasible(superSource,superSink,
                in.maximumAugmentations-augmentationCount,roundAugmentations,sent,flowError)){
            perf.flowAugmentations+=roundAugmentations;
            return fail("shared-capacity sparse feasible-flow failed: "+flowError);
        }
        augmentationCount+=roundAugmentations;
        perf.flowAugmentations=augmentationCount;
        return true;
    };
    const auto allMinimumsSatisfied=[&](){
        for(std::size_t c=0;c<nc;++c)
            if(network.hasResidualCapacity(childBase+static_cast<std::uint32_t>(c),minimumArcs[c]))
                return false;
        return true;
    };
    // Retain child-extra arcs so maximum-flow cut expansion can favor children
    // with remaining density capacity as well as minimum-phase lower deficits.
    std::vector<std::uint32_t> maximumArcs(nc,invalid);
    const auto minimumResidualSummary=[&](){
        std::size_t unsatisfied=0u;
        long double remaining=0.0L;
        for(std::size_t c=0;c<nc;++c){
            const auto node=childBase+static_cast<std::uint32_t>(c);
            if(!network.hasResidualCapacity(node,minimumArcs[c]))continue;
            ++unsatisfied;
            remaining+=static_cast<long double>(network.remainingCapacity(node,minimumArcs[c]));
        }
        return std::pair<std::size_t,long double>{unsatisfied,remaining};
    };
    const auto residualSourceMassKg=[&](){
        return std::max(0.0,supply-network.sentFromSource(superSource));
    };
    const auto residualChildWitnesses=[&](const bool minimumPhase,
                                           const std::vector<bool>& reached){
        struct Item {double residual;std::uint32_t child;};
        std::vector<Item> top;
        constexpr std::size_t limit=8u;
        const auto minHeapOrder=[](const Item& a,const Item& b){
            return a.residual!=b.residual?a.residual>b.residual:a.child<b.child;
        };
        for(std::size_t c=0;c<nc;++c){
            const auto arc=minimumPhase?minimumArcs[c]:maximumArcs[c];
            if(arc==invalid)continue;
            const double residual=network.remainingCapacity(
                childBase+static_cast<std::uint32_t>(c),arc);
            if(!std::isfinite(residual)||residual<=0.0)continue;
            const Item item{residual,static_cast<std::uint32_t>(c)};
            if(top.size()<limit){top.push_back(item);std::push_heap(top.begin(),top.end(),minHeapOrder);}
            else if(item.residual>top.front().residual||
                    (item.residual==top.front().residual&&item.child<top.front().child)){
                std::pop_heap(top.begin(),top.end(),minHeapOrder);top.back()=item;
                std::push_heap(top.begin(),top.end(),minHeapOrder);
            }
        }
        std::sort(top.begin(),top.end(),[](const Item& a,const Item& b){
            return a.residual!=b.residual?a.residual>b.residual:a.child<b.child;
        });
        std::ostringstream out;
        out<<"witness_count="<<top.size();
        for(const auto& item:top){
            const auto c=item.child;
            std::size_t exactParents=0u,reachedParents=0u;
            double nearestDistance=std::numeric_limits<double>::infinity();
            std::uint32_t nearestParent=invalid;
            const auto base=childGrid[c];
            for(const auto group:childAdmissibleGroups[c]){
                for(int dx=-2;dx<=2;++dx)for(int dy=-2;dy<=2;++dy)for(int dz=-2;dz<=2;++dz){
                    const GridKey key{base.x+dx,base.y+dy,base.z+dz};
                    const auto found=parentsByGroupBucket.find({group,key});
                    if(found==parentsByGroupBucket.end())continue;
                    for(const auto p:found->second){
                        const double d=distance(in.parentCentroidsWorld[p],in.childCentroidsWorld[c]);
                        if(d>in.maximumTransportDistanceM||d*d>radiusSquared*(1.0+1.0e-14))continue;
                        ++exactParents;
                        if(reached[parentBase+p])++reachedParents;
                        if(d<nearestDistance||(d==nearestDistance&&p<nearestParent)){
                            nearestDistance=d;nearestParent=p;
                        }
                    }
                }
            }
            out<<" child="<<c
                <<",material="<<in.childMaterialIndices[c]
                <<",donor="<<in.childDonorBodyIndices[c]
                <<",region="<<in.childAnatomicalRegionIndices[c]
                <<",xyz="<<std::setprecision(17)<<in.childCentroidsWorld[c][0]<<":"
                <<in.childCentroidsWorld[c][1]<<":"<<in.childCentroidsWorld[c][2]
                <<",residual_kg="<<item.residual
                <<",groups="<<childAdmissibleGroups[c].size()
                <<",exact_parents="<<exactParents
                <<",reached_exact_parents="<<reachedParents
                <<",reached_child="<<(reached[childBase+c]?1:0);
            if(nearestParent!=invalid)
                out<<",nearest_parent="<<nearestParent
                   <<",nearest_distance_m="<<nearestDistance
                   <<",nearest_parent_initial_kg="<<in.parentMassKg[nearestParent];
            out<<";";
        }
        return out.str();
    };
    const auto cutMaskSummary=[&](const std::vector<bool>& reached){
        spatial_mass_transport_detail::SHA256Builder hash;
        hash.string("numi.h706.full-implicit-residual-cut-mask.v1");
        hash.string(in.parentMassPacketSHA256);hash.string(in.parentSourceContributionSHA256);
        hash.string(in.childMeshSHA256);hash.string(in.regionLabelSHA256);
        hash.string(support.typedSupportSHA256);
        hash.u64(np);hash.u64(nc);
        std::size_t parentRuns=0u,childRuns=0u;
        bool previous=false;
        for(std::size_t p=0;p<np;++p){
            const bool value=reached[parentBase+p];hash.u32(value?1u:0u);
            if(value&&!previous)++parentRuns;
            previous=value;
        }
        previous=false;
        for(std::size_t c=0;c<nc;++c){
            const bool value=reached[childBase+c];hash.u32(value?1u:0u);
            if(value&&!previous)++childRuns;
            previous=value;
        }
        std::ostringstream out;
        out<<"sha256="<<hash.finish()<<",parent_runs="<<parentRuns<<",child_runs="<<childRuns;
        return out.str();
    };
    const auto emitFullCutWitness=[&](const bool minimumPhase,
                                      const std::vector<bool>& reached,
                                      const std::size_t reachedParents,
                                      const std::size_t reachedChildren,
                                      const std::uint64_t implicitPairs,
                                      const std::uint64_t exactOmittedPairs){
        std::ostringstream out;
        std::string parentArrayHash,childArrayHash,boundsHash;
        hashInputArrays(in,parentArrayHash,childArrayHash,boundsHash);
        const auto maskHex=[&](const std::size_t start,const std::size_t count){
            constexpr char hex[]="0123456789abcdef";
            std::string value;value.reserve(((count+7u)/8u)*2u);
            for(std::size_t first=0;first<count;first+=8u){
                std::uint8_t bits=0u;
                for(std::size_t bit=0;bit<8u&&first+bit<count;++bit)
                    if(reached[start+first+bit])bits=static_cast<std::uint8_t>(bits|(1u<<bit));
                value.push_back(hex[bits>>4u]);value.push_back(hex[bits&0x0fu]);
            }
            return value;
        };
        struct GroupSummary {std::size_t parents=0u,reached=0u;long double remaining=0.0L;};
        std::vector<GroupSummary> groupSummary(support.groupMaterialIndices.size());
        for(std::size_t p=0;p<np;++p){
            auto& summary=groupSummary[parentGroup[p]];
            ++summary.parents;
            if(reached[parentBase+p])++summary.reached;
            summary.remaining+=std::max(0.0,network.remainingCapacity(
                superSource,parentSourceArcs[p]));
        }
        long double cutSourceCapacity=0.0L,cutParentChildCapacity=0.0L;
        long double cutMinimumCapacity=0.0L,cutMaximumCapacity=0.0L,cutFinalCapacity=0.0L;
        std::size_t cutSourceEdges=0u,cutParentChildEdges=0u;
        std::size_t cutMinimumEdges=0u,cutMaximumEdges=0u,cutFinalEdges=0u;
        for(std::size_t p=0;p<np;++p)
            if(!reached[parentBase+p]){cutSourceCapacity+=in.parentMassKg[p];++cutSourceEdges;}
        for(const auto& ref:references)
            if(reached[parentBase+ref.parent]&&!reached[childBase+ref.child]){
                cutParentChildCapacity+=in.parentMassKg[ref.parent];++cutParentChildEdges;
            }
        for(std::size_t c=0;c<nc;++c){
            if(!reached[childBase+c])continue;
            if(!reached[superSink]){cutMinimumCapacity+=childMinimum[c];++cutMinimumEdges;}
            if(!minimumPhase&&!reached[sink]){
                cutMaximumCapacity+=childMaximum[c]-childMinimum[c];++cutMaximumEdges;
            }
        }
        if(!minimumPhase&&reached[sink]&&!reached[superSink]){
            cutFinalCapacity+=residualCapacity;++cutFinalEdges;
        }
        const long double totalCutCapacity=cutSourceCapacity+cutParentChildCapacity+
            cutMinimumCapacity+cutMaximumCapacity+cutFinalCapacity;
        const double sent=network.sentFromSource(superSource);
        std::size_t residualChildRows=0u,unreachableResidualChildRows=0u;
        long double residualChildKg=0.0L,unreachableResidualChildKg=0.0L;
        for(std::size_t c=0;c<nc;++c){
            const auto arc=minimumPhase?minimumArcs[c]:maximumArcs[c];
            if(arc==invalid)continue;
            const double residual=network.remainingCapacity(childBase+static_cast<std::uint32_t>(c),arc);
            if(!std::isfinite(residual)||residual<=0.0)continue;
            ++residualChildRows;residualChildKg+=residual;
            if(!reached[childBase+c]){
                ++unreachableResidualChildRows;unreachableResidualChildKg+=residual;
            }
        }
        out<<"shared_capacity_full_cut_begin"
            <<" schema=numi.h706.full-implicit-residual-cut.v1"
            <<" phase="<<(minimumPhase?"minimum_flow":"maximum_flow")
            <<" input_parent_mass_sha256="<<in.parentMassPacketSHA256
            <<" input_parent_source_sha256="<<in.parentSourceContributionSHA256
            <<" input_child_mesh_sha256="<<in.childMeshSHA256
            <<" input_region_sha256="<<in.regionLabelSHA256
            <<" input_coordinate_frame_sha256="<<in.coordinateFrameSHA256
            <<" typed_support_sha256="<<support.typedSupportSHA256
            <<" semantic_support_sha256="<<supportDigest(support)
            <<" parent_numeric_sha256="<<parentArrayHash
            <<" child_numeric_sha256="<<childArrayHash
            <<" bounds_numeric_sha256="<<boundsHash
            <<" parent_count="<<np<<" child_count="<<nc
            <<" exact_group_count="<<support.groupMaterialIndices.size()
            <<" reached_parent_count="<<reachedParents
            <<" reached_child_count="<<reachedChildren
            <<" implicit_pairs_examined="<<implicitPairs
            <<" exact_eligible_omitted_pairs="<<exactOmittedPairs
            <<" residual_child_rows="<<residualChildRows
            <<" residual_child_kg="<<std::setprecision(17)<<static_cast<double>(residualChildKg)
            <<" unreachable_residual_child_rows="<<unreachableResidualChildRows
            <<" unreachable_residual_child_kg="<<static_cast<double>(unreachableResidualChildKg)
            <<" sent_kg="<<std::setprecision(17)<<sent
            <<" residual_source_kg="<<residualSourceMassKg()
            <<" supply_kg="<<supply<<" lower_kg="<<minimum<<" upper_kg="<<maximum
            <<" cut_capacity_kg="<<static_cast<double>(totalCutCapacity)
            <<" cut_capacity_minus_flow_kg="<<static_cast<double>(totalCutCapacity-static_cast<long double>(sent))
            <<" cut_source_edges="<<cutSourceEdges
            <<" cut_source_capacity_kg="<<static_cast<double>(cutSourceCapacity)
            <<" cut_parent_child_edges="<<cutParentChildEdges
            <<" cut_parent_child_capacity_kg="<<static_cast<double>(cutParentChildCapacity)
            <<" cut_minimum_edges="<<cutMinimumEdges
            <<" cut_minimum_capacity_kg="<<static_cast<double>(cutMinimumCapacity)
            <<" cut_maximum_edges="<<cutMaximumEdges
            <<" cut_maximum_capacity_kg="<<static_cast<double>(cutMaximumCapacity)
            <<" cut_final_edges="<<cutFinalEdges
            <<" cut_final_capacity_kg="<<static_cast<double>(cutFinalCapacity)
            <<" mask_encoding=hex_lsb_first_index_order_parent_then_child"
            <<" parent_mask_hex="<<maskHex(parentBase,np)
            <<" child_mask_hex="<<maskHex(childBase,nc)<<"\n";
        out<<"group_summary_csv_begin\ngroup,material,donor,region,identity,parent_rows,reached_parent_rows,residual_source_kg\n";
        for(std::size_t g=0;g<groupSummary.size();++g){
            const auto& q=groupSummary[g];
            out<<g<<","<<support.groupMaterialIndices[g]<<","<<support.donorBodyIndices[g]
                <<","<<support.anatomicalRegionIndices[g]<<","<<support.derivedGroupIdentityIDs[g]
                <<","<<q.parents<<","<<q.reached<<","<<std::setprecision(17)
                <<static_cast<double>(q.remaining)<<"\n";
        }
        out<<"group_summary_csv_end\n";
        out<<"parent_group_rows_csv_begin\nparent,group,material,donor,region,x,y,z,reached,initial_mass_kg,residual_source_kg,source_parent_tet_rows,source_stable_face_rows\n";
        for(std::size_t p=0;p<np;++p){
            const auto group=static_cast<std::size_t>(parentGroup[p]);
            out<<p<<","<<group<<","<<in.parentMaterialIndices[p]
                <<","<<in.parentDonorBodyIndices[p]<<","<<in.parentAnatomicalRegionIndices[p]
                <<","<<std::setprecision(17)<<in.parentCentroidsWorld[p][0]
                <<","<<in.parentCentroidsWorld[p][1]<<","<<in.parentCentroidsWorld[p][2]
                <<","<<(reached[parentBase+p]?1:0)<<","<<in.parentMassKg[p]
                <<","<<std::max(0.0,network.remainingCapacity(
                    superSource,parentSourceArcs[p]))<<",\"";
            for(std::size_t i=support.sourceParentTetOffsets[group];
                i<support.sourceParentTetOffsets[group+1u];++i){
                if(i!=support.sourceParentTetOffsets[group])out<<";";
                out<<support.sourceParentTetRows[i];
            }
            out<<"\",\"";
            for(std::size_t i=support.sourceStableFaceOffsets[group];
                i<support.sourceStableFaceOffsets[group+1u];++i){
                if(i!=support.sourceStableFaceOffsets[group])out<<";";
                out<<support.sourceStableFaceRows[i];
            }
            out<<"\"\n";
        }
        out<<"parent_group_rows_csv_end\n";
        out<<"residual_child_rows_csv_begin\nchild,material,donor,region,x,y,z,residual_kg,reachable,admissible_groups\n";
        for(std::size_t c=0;c<nc;++c){
            const auto arc=minimumPhase?minimumArcs[c]:maximumArcs[c];
            if(arc==invalid)continue;
            const double residual=network.remainingCapacity(childBase+static_cast<std::uint32_t>(c),arc);
            if(!std::isfinite(residual)||residual<=0.0)continue;
            out<<c<<","<<in.childMaterialIndices[c]<<","<<in.childDonorBodyIndices[c]
                <<","<<in.childAnatomicalRegionIndices[c]<<","<<std::setprecision(17)
                <<in.childCentroidsWorld[c][0]<<","<<in.childCentroidsWorld[c][1]<<","<<in.childCentroidsWorld[c][2]
                <<","<<residual<<","<<(reached[childBase+c]?1:0)<<",\"";
            bool first=true;
            for(const auto g:childAdmissibleGroups[c]){
                if(!first)out<<";";
                first=false;
                const auto pref=groupPreference[c].find(g);
                const double score=pref==groupPreference[c].end()?0.0:pref->second;
                out<<g<<":"<<support.groupMaterialIndices[g]<<":"<<support.donorBodyIndices[g]
                    <<":"<<support.anatomicalRegionIndices[g]<<":"<<std::setprecision(17)<<score;
            }
            out<<"\"\n";
        }
        out<<"residual_child_rows_csv_end\nshared_capacity_full_cut_end\n";
        return out.str();
    };
    const auto expandAcrossResidualCut=[&](const bool minimumPhase){
        perf.stage=minimumPhase?"minimum_flow":"maximum_flow";
        ++perf.cutRounds;
        std::vector<bool> reached(static_cast<std::size_t>(superSink)+1u,false);
        std::vector<std::uint32_t> queue;
        queue.reserve(np+nc);
        queue.push_back(superSource);reached[superSource]=true;
        std::vector<PendingEdge> additions;
        additions.reserve(std::min(nc,cutExpansionBatch));
        std::uint64_t implicitPairsExamined=0u,exactEligibleOmittedPairs=0u;
        bool finalSinkReached=false;
        constexpr std::int64_t implicitBucketRadius=2;
        for(std::size_t cursor=0;cursor<queue.size();++cursor){
            const auto from=queue[cursor];
            // superSink is the final terminal. Never follow its reverse arcs,
            // which would falsely make already-satisfied children look newly reachable.
            if(from==superSink)continue;
            network.forEachPositiveResidualEdge(from,[&](const std::uint32_t to){
                if(to>=reached.size()||reached[to])return;
                reached[to]=true;
                if(to==superSink){finalSinkReached=true;return;}
                queue.push_back(to);
            });
            if(from<parentBase||from>=childBase)continue;
            const auto p=from-parentBase;
            const auto group=parentGroup[p];
            const auto base=parentGrid[p];
            for(std::int64_t dx=-implicitBucketRadius;dx<=implicitBucketRadius;++dx)
                for(std::int64_t dy=-implicitBucketRadius;dy<=implicitBucketRadius;++dy)
                    for(std::int64_t dz=-implicitBucketRadius;dz<=implicitBucketRadius;++dz){
                        const GridKey key{base.x+dx,base.y+dy,base.z+dz};
                        const auto found=childrenByGroupBucket.find({group,key});
                        if(found==childrenByGroupBucket.end())continue;
                        for(const auto c:found->second){
                            ++implicitPairsExamined;
                            if(std::binary_search(activeChildrenByParent[p].begin(),
                                                  activeChildrenByParent[p].end(),c))continue;
                            const double d=distance(in.parentCentroidsWorld[p],in.childCentroidsWorld[c]);
                            if(d>in.maximumTransportDistanceM||
                               d*d>radiusSquared*(1.0+1.0e-14))continue;
                            ++exactEligibleOmittedPairs;
                            const auto childNode=childBase+c;
                            if(reached[childNode])continue;
                            const auto pref=groupPreference[c].find(group);
                            const double penalty=pref==groupPreference[c].end()?0.0:
                                semanticPreferenceWeight*(-std::log(pref->second));
                            const double cost=(d*d)/radiusSquared+penalty;
                            if(!std::isfinite(cost)||cost<0.0)
                                return fail("shared-capacity implicit residual-forest edge cost is invalid");
                            additions.push_back({static_cast<std::uint32_t>(p),c,d,cost});
                            reached[childNode]=true;
                            queue.push_back(childNode);
                        }
                    }
        }
        const auto lowerBefore=minimumResidualSummary();
        const double sourceBefore=residualSourceMassKg();
        std::size_t reachedParentCount=0u,reachedChildCount=0u;
        for(std::size_t p=0;p<np;++p)if(reached[parentBase+p])++reachedParentCount;
        for(std::size_t c=0;c<nc;++c)if(reached[childBase+c])++reachedChildCount;
        if(additions.empty()){
            if(finalSinkReached)
                return fail("INTERNAL shared-capacity residual forest found final sink without a new admissible edge");
            std::ostringstream witness;
            witness<<(minimumPhase?
                "INFEASIBLE full admissible shared-capacity graph cannot satisfy every child minimum density":
                "INFEASIBLE full admissible shared-capacity graph cannot transfer all source mass within child maximum densities")
                <<" full_implicit_residual_closure=exhausted no_augmenting_path=1"
                <<" stage="<<(minimumPhase?"minimum_flow":"maximum_flow")
                <<" sent_kg="<<std::setprecision(17)<<network.sentFromSource(superSource)
                <<" residual_source_kg="<<sourceBefore
                <<" supply_kg="<<supply<<" density_min_kg="<<minimum<<" density_max_kg="<<maximum
                <<" cut_reachable_parent_rows="<<reachedParentCount
                <<" cut_reachable_child_rows="<<reachedChildCount
                <<" implicit_parent_child_pairs_examined="<<implicitPairsExamined
                <<" exact_eligible_omitted_pairs="<<exactEligibleOmittedPairs
                <<" cut_mask="<<cutMaskSummary(reached)
                <<" residual_child_witnesses={"<<residualChildWitnesses(minimumPhase,reached)<<"}";
            witness<<"\n"<<emitFullCutWitness(minimumPhase,reached,reachedParentCount,reachedChildCount,
                               implicitPairsExamined,exactEligibleOmittedPairs);
            return fail(witness.str());
        }
        const std::size_t remainingEdges=in.maximumAllowedEdges-
            std::min(in.maximumAllowedEdges,pending.size());
        if(remainingEdges==0u){
            std::ostringstream witness;
            witness<<"INCONCLUSIVE shared-capacity edge budget prevents implicit residual-forest insertion"
                <<" full_implicit_residual_closure=found_frontier"
                <<" stage="<<(minimumPhase?"minimum_flow":"maximum_flow")
                <<" sent_kg="<<std::setprecision(17)<<network.sentFromSource(superSource)
                <<" residual_source_kg="<<sourceBefore
                <<" candidate_edges="<<pending.size()
                <<" forest_edges_found="<<additions.size()
                <<" cut_reachable_parent_rows="<<reachedParentCount
                <<" cut_reachable_child_rows="<<reachedChildCount
                <<" cut_mask="<<cutMaskSummary(reached)
                <<" residual_child_witnesses={"<<residualChildWitnesses(minimumPhase,reached)<<"}";
            return fail(witness.str());
        }
        const std::size_t addCount=std::min(additions.size(),remainingEdges);
        perf.cutEdgesAdded+=addCount;
        std::vector<std::uint32_t> modifiedParents;
        modifiedParents.reserve(addCount);
        for(std::size_t i=0;i<addCount;++i){
            const auto& edge=additions[i];
            const auto from=parentBase+edge.parent;
            const auto arc=network.addEdge(from,childBase+edge.child,
                                           in.parentMassKg[edge.parent],edge.cost);
            references.push_back({edge.parent,edge.child,from,arc,edge.distance});
            pending.push_back(edge);
            activeChildrenByParent[edge.parent].push_back(edge.child);
            modifiedParents.push_back(edge.parent);
            reachableChildren[parentGroup[edge.parent]].push_back(edge.child);
        }
        std::sort(modifiedParents.begin(),modifiedParents.end());
        modifiedParents.erase(std::unique(modifiedParents.begin(),modifiedParents.end()),
                              modifiedParents.end());
        for(const auto p:modifiedParents){
            auto& active=activeChildrenByParent[p];
            std::sort(active.begin(),active.end());
            active.erase(std::unique(active.begin(),active.end()),active.end());
        }
        const bool flowOK=augmentToMaximum();
        const auto lowerAfter=minimumResidualSummary();
        const double sourceAfter=residualSourceMassKg();
        std::cerr<<"shared_capacity_cut_diagnostics"
            <<" stage="<<(minimumPhase?"minimum_flow":"maximum_flow")
            <<" cut_round="<<perf.cutRounds
            <<" reached_parents="<<reachedParentCount
            <<" reached_children="<<reachedChildCount
            <<" implicit_pairs_examined="<<implicitPairsExamined
            <<" exact_eligible_omitted_pairs="<<exactEligibleOmittedPairs
            <<" forest_edges_found="<<additions.size()
            <<" final_super_sink_reachable="<<(finalSinkReached?1:0)
            <<" selected_edges="<<addCount
            <<" min_unsatisfied_before="<<lowerBefore.first
            <<" remaining_lower_kg_before="<<std::setprecision(17)<<static_cast<double>(lowerBefore.second)
            <<" residual_source_kg_before="<<sourceBefore
            <<" min_unsatisfied_after="<<lowerAfter.first
            <<" remaining_lower_kg_after="<<static_cast<double>(lowerAfter.second)
            <<" residual_source_kg_after="<<sourceAfter
            <<" flow_ok="<<(flowOK?1:0)<<std::endl;
        return flowOK;
    };
    if(!augmentToMaximum())return false;
    while(!allMinimumsSatisfied())
        if(!expandAcrossResidualCut(true))return false;
    perf.stage="minimum_satisfied";

    // Minimum density is pinned in every real child before extra capacity is
    // exposed. This prevents source mass from bypassing tiny child floors.
    for(std::size_t c=0;c<nc;++c)
        maximumArcs[c]=network.addEdge(childBase+static_cast<std::uint32_t>(c),sink,
                        childMaximum[c]-childMinimum[c],0.0);
    network.addEdge(sink,superSink,residualCapacity,0.0);
    perf.stage="maximum_flow";
    if(!augmentToMaximum())return false;
    while(network.hasResidualSourceCapacity(superSource))
        if(!expandAcrossResidualCut(false))return false;
    if(augmentationCount==0u||augmentationCount>in.maximumAugmentations)
        return fail("shared-capacity sparse feasible-flow augmentation count is outside its pinned budget");

    if(augmentationCount==0u||augmentationCount>in.maximumAugmentations)
        return fail("shared-capacity sparse flow augmentation count is outside its pinned budget");

    std::vector<SharedCapacityFlow> flows;
    flows.reserve(references.size());
    for(const auto& ref:references){
        const double mass=network.reverseFlow(ref);
        if(mass<=0.0)continue;
        const auto p=ref.parent,c=ref.child;
        const double parentDensity=in.parentMassKg[p]/in.parentReferenceVolumeM3[p];
        const double measure=mass/parentDensity;
        if(!std::isfinite(mass)||mass<=0.0||!std::isfinite(measure)||measure<=0.0||
           ref.distanceM>in.maximumTransportDistanceM)
            return fail("shared-capacity min-cost flow produced invalid mass, measure or transport distance");
        flows.push_back({p,c,mass,measure,ref.distanceM});
    }
    std::sort(flows.begin(),flows.end(),[](const auto& a,const auto& b){
        return a.child!=b.child?a.child<b.child:a.parent<b.parent;
    });
    for(std::size_t i=1;i<flows.size();++i)
        if(flows[i-1u].child==flows[i].child&&flows[i-1u].parent==flows[i].parent)
            return fail("shared-capacity graph produced a duplicate parent/child edge");

    ConservativeSpatialMassTransportResult next;
    next.method=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    next.parentMassPacketSHA256=std::string(in.parentMassPacketSHA256);
    next.parentSourceContributionSHA256=std::string(in.parentSourceContributionSHA256);
    next.childMeshSHA256=std::string(in.childMeshSHA256);
    next.regionLabelSHA256=std::string(in.regionLabelSHA256);
    next.coordinateFrameSHA256=std::string(in.coordinateFrameSHA256);
    next.maximumTransportDistanceM=in.maximumTransportDistanceM;
    next.relativeClosureTolerance=in.relativeClosureTolerance;
    next.absoluteMassToleranceKg=in.absoluteMassToleranceKg;
    next.absoluteMeasureToleranceM3=in.absoluteMeasureToleranceM3;
    next.maximumAllowedEdges=in.maximumAllowedEdges;
    next.maximumAugmentations=in.maximumAugmentations;
    next.augmentationCount=augmentationCount;
    next.semanticCandidateEdgeCount=pending.size();
    next.usesFractionalSemanticSupport=true;
    next.semanticSupportSHA256=supportDigest(support);
    hashInputArrays(in,next.parentDataSHA256,next.childDataSHA256,next.materialBoundsSHA256);
    next.childDataSHA256=bindSupportToChildHash(next.childDataSHA256,next.semanticSupportSHA256);
    next.parentMassTransferredKg.assign(np,0.0);
    next.parentMeasureTransferredM3.assign(np,0.0);
    next.childMassKg.assign(nc,0.0);
    next.childAncestryMeasureVolumeM3.assign(nc,0.0);
    next.childImpliedDensityKgM3.assign(nc,0.0);
    next.childDominantParentRows.assign(nc,invalid);
    next.childSourceDonorOffsets.assign(nc+1u,0u);
    next.childSemanticGroupOffsets.assign(nc+1u,0u);
    std::vector<std::map<std::uint32_t,double>> sourceMassByChild(nc);
    std::vector<std::map<std::uint32_t,double>> semanticMassByChild(nc);
    std::vector<double> groupParentMass(support.groupMaterialIndices.size(),0.0);
    std::vector<double> groupChildMass(support.groupMaterialIndices.size(),0.0);
    std::vector<double> groupReachableCapacity(support.groupMaterialIndices.size(),0.0);
    std::map<std::pair<std::uint32_t,std::uint32_t>,double> parentSourceByDonor,childSourceByDonor;
    for(std::size_t p=0;p<np;++p){
        const auto g=parentGroup[p];
        groupParentMass[g]+=in.parentMassKg[p];
        const auto material=in.parentMaterialIndices[p];
        for(std::uint32_t r=in.parentSourceDonorOffsets[p];r<in.parentSourceDonorOffsets[p+1u];++r)
            parentSourceByDonor[{material,in.parentSourceDonorBodyIndices[r]}]+=
                in.parentSourceDonorMassKg[r];
    }
    for(std::size_t g=0;g<reachableChildren.size();++g){
        auto& children=reachableChildren[g];
        std::sort(children.begin(),children.end());
        children.erase(std::unique(children.begin(),children.end()),children.end());
        for(const auto c:children)groupReachableCapacity[g]+=childMaximum[c];
    }
    for(std::size_t p=0;p<np;++p){
        // groupChildMass is populated below from the canonical flow order.
        (void)p;
    }
    for(const auto& flow:flows){
        const auto p=flow.parent,c=flow.child,g=parentGroup[p];
        const double parentFraction=flow.mass/in.parentMassKg[p];
        next.parentMassTransferredKg[p]+=flow.mass;
        next.parentMeasureTransferredM3[p]+=flow.measure;
        next.childMassKg[c]+=flow.mass;
        next.childAncestryMeasureVolumeM3[c]+=flow.measure;
        groupChildMass[g]+=flow.mass;
        semanticMassByChild[c][g]+=flow.mass;
        for(std::uint32_t r=in.parentSourceDonorOffsets[p];r<in.parentSourceDonorOffsets[p+1u];++r){
            const auto donor=in.parentSourceDonorBodyIndices[r];
            const double sourceMass=in.parentSourceDonorMassKg[r]*parentFraction;
            sourceMassByChild[c][donor]+=sourceMass;
            childSourceByDonor[{in.parentMaterialIndices[p],donor}]+=sourceMass;
        }
        next.maximumTransportDistanceUsedM=std::max(next.maximumTransportDistanceUsedM,flow.distance);
    }
    next.childParentOffsets.assign(nc+1u,0u);
    next.childParentRows.reserve(flows.size());
    next.childParentTransportMassKg.reserve(flows.size());
    next.childParentTransportMeasureM3.reserve(flows.size());
    next.childParentTransportDistanceM.reserve(flows.size());
    next.childParentVolumeFractions.reserve(flows.size());
    std::vector<std::uint32_t> flowCountByChild(nc,0u);
    for(const auto& flow:flows)++flowCountByChild[flow.child];
    long double weightedDistance=0.0L,totalMass=0.0L;
    std::size_t at=0u;
    for(std::size_t c=0;c<nc;++c){
        if(flowCountByChild[c]==0u||next.childMassKg[c]<=0.0||
           next.childAncestryMeasureVolumeM3[c]<=0.0)
            return fail("shared-capacity flow left a real child without positive mass");
        next.childParentOffsets[c]=static_cast<std::uint32_t>(next.childParentRows.size());
        double measureSum=0.0,bestMeasure=-1.0;std::uint32_t dominant=invalid;
        while(at<flows.size()&&flows[at].child==c){
            const auto& flow=flows[at++];
            next.childParentRows.push_back(flow.parent);
            next.childParentTransportMassKg.push_back(flow.mass);
            next.childParentTransportMeasureM3.push_back(flow.measure);
            next.childParentTransportDistanceM.push_back(flow.distance);
            const double fraction=flow.measure/next.childAncestryMeasureVolumeM3[c];
            if(!std::isfinite(fraction)||fraction<=0.0)
                return fail("shared-capacity child parent-measure fraction is invalid");
            next.childParentVolumeFractions.push_back(fraction);
            measureSum+=flow.measure;
            if(flow.measure>bestMeasure||(flow.measure==bestMeasure&&flow.parent<dominant)){
                bestMeasure=flow.measure;dominant=flow.parent;
            }
            weightedDistance+=static_cast<long double>(flow.mass)*flow.distance;
            totalMass+=flow.mass;
        }
        next.childParentOffsets[c+1u]=static_cast<std::uint32_t>(next.childParentRows.size());
        next.childDominantParentRows[c]=dominant;
        if(std::abs(measureSum/next.childAncestryMeasureVolumeM3[c]-1.0)>5.0e-13)
            return fail("shared-capacity child parent measures do not close");
        const auto material=in.childMaterialIndices[c];
        const double density=next.childMassKg[c]/in.childReferenceVolumeM3[c];
        const double tol=densityBoundRoundoff(in.materialDensityLowerKgM3[material],
                                               in.materialDensityUpperKgM3[material],density);
        if(!std::isfinite(density)||density<in.materialDensityLowerKgM3[material]-tol||
           density>in.materialDensityUpperKgM3[material]+tol){
            std::ostringstream witness;
            witness<<"shared-capacity real child density is outside its material bounds"
                <<" child_row="<<c<<" material="<<material
                <<" mass_kg="<<std::setprecision(17)<<next.childMassKg[c]
                <<" volume_m3="<<in.childReferenceVolumeM3[c]
                <<" density_kg_m3="<<density
                <<" lower="<<in.materialDensityLowerKgM3[material]
                <<" upper="<<in.materialDensityUpperKgM3[material];
            return fail(witness.str());
        }
        next.childImpliedDensityKgM3[c]=density;
        next.minimumChildImpliedDensityKgM3=
            c==0u?density:std::min(next.minimumChildImpliedDensityKgM3,density);
        next.maximumChildImpliedDensityKgM3=std::max(next.maximumChildImpliedDensityKgM3,density);
        next.childSourceDonorOffsets[c]=static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
        double sourceTotal=0.0;
        for(const auto& [donor,mass]:sourceMassByChild[c]){
            if(donor==invalid||!std::isfinite(mass)||mass<=0.0)
                return fail("shared-capacity child source-donor contribution is invalid");
            next.childSourceDonorBodyIndices.push_back(donor);
            next.childSourceDonorMassKg.push_back(mass);
            sourceTotal+=mass;
        }
        next.childSourceDonorOffsets[c+1u]=
            static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
        if(std::abs(sourceTotal-next.childMassKg[c])>
           std::max(in.absoluteMassToleranceKg,in.relativeClosureTolerance*next.childMassKg[c]))
            return fail("shared-capacity child source-donor mass does not close");

        next.childSemanticGroupOffsets[c]=
            static_cast<std::uint32_t>(next.childSemanticGroupIndices.size());
        double semanticSum=0.0;
        for(const auto& [g,mass]:semanticMassByChild[c]){
            const double fraction=mass/next.childMassKg[c];
            if(g>=support.groupMaterialIndices.size()||!std::isfinite(fraction)||fraction<=0.0)
                return fail("shared-capacity realized semantic group fraction is invalid");
            next.childSemanticGroupIndices.push_back(g);
            next.childSemanticVolumeFractions.push_back(fraction);
            semanticSum+=fraction;
        }
        next.childSemanticGroupOffsets[c+1u]=
            static_cast<std::uint32_t>(next.childSemanticGroupIndices.size());
        if(std::abs(semanticSum-1.0)>1.0e-12)
            return fail("shared-capacity realized semantic fractions do not sum to one");
    }
    next.childParentOffsets[nc]=static_cast<std::uint32_t>(next.childParentRows.size());
    next.childSourceDonorOffsets[nc]=
        static_cast<std::uint32_t>(next.childSourceDonorBodyIndices.size());
    next.transportEdgeCount=flows.size();
    next.massWeightedMeanTransportDistanceM=totalMass>0.0L?
        static_cast<double>(weightedDistance/totalMass):0.0;
    for(std::size_t p=0;p<np;++p){
        const double massError=std::abs(next.parentMassTransferredKg[p]-in.parentMassKg[p]);
        const double measureError=std::abs(next.parentMeasureTransferredM3[p]-in.parentReferenceVolumeM3[p]);
        next.maximumParentMassClosureErrorKg=std::max(next.maximumParentMassClosureErrorKg,massError);
        next.maximumParentMeasureClosureErrorM3=std::max(next.maximumParentMeasureClosureErrorM3,measureError);
        if(massError>std::max(in.absoluteMassToleranceKg,in.relativeClosureTolerance*in.parentMassKg[p])||
           measureError>std::max(in.absoluteMeasureToleranceM3,
                                 in.relativeClosureTolerance*in.parentReferenceVolumeM3[p]))
            return fail("shared-capacity parent mass or source measure does not close");
    }
    for(std::size_t g=0;g<groupParentMass.size();++g){
        if(std::abs(groupParentMass[g]-groupChildMass[g])>
           std::max(in.absoluteMassToleranceKg,in.relativeClosureTolerance*groupParentMass[g]))
            return fail("shared-capacity semantic source group mass does not close");
        next.groupClosure.push_back({
            .materialIndex=support.groupMaterialIndices[g],
            .donorBodyIndex=support.donorBodyIndices[g],
            .anatomicalRegionIndex=support.anatomicalRegionIndices[g],
            .parentMassKg=groupParentMass[g],
            .childMassKg=groupChildMass[g],
            .childMinimumCapacityKg=0.0,
            .childMaximumCapacityKg=groupReachableCapacity[g]});
    }
    std::set<std::pair<std::uint32_t,std::uint32_t>> sourceKeys;
    for(const auto& [key,mass]:parentSourceByDonor){(void)mass;sourceKeys.insert(key);}
    for(const auto& [key,mass]:childSourceByDonor){(void)mass;sourceKeys.insert(key);}
    for(const auto& key:sourceKeys){
        const double parentMass=parentSourceByDonor[key],childMass=childSourceByDonor[key];
        const double closure=std::abs(parentMass-childMass);
        next.maximumSourceDonorMassClosureErrorKg=
            std::max(next.maximumSourceDonorMassClosureErrorKg,closure);
        if(closure>std::max(in.absoluteMassToleranceKg,in.relativeClosureTolerance*parentMass))
            return fail("shared-capacity source donor contribution mass does not close");
        next.sourceDonorClosure.push_back({.materialIndex=key.first,
            .sourceDonorBodyIndex=key.second,.parentMassKg=parentMass,.childMassKg=childMass});
    }
    next.maximumChildFractionSumError=0.0;
    next.receiptSHA256=sharedCapacityReceipt(in,next,next.semanticSupportSHA256);
    out=std::move(next);
    perf.stage="complete";
    error.clear();return true;
}


} // namespace fractional_semantic_transport_detail

[[nodiscard]] inline bool assembleConservativeSpatialMassTransportWithSemanticSupport(
        const ConservativeSpatialMassTransportInput& realInput,
        const ConservativeSpatialChildSemanticSupport& support,
        ConservativeSpatialMassTransportResult& out,
        std::string& error) {
    return fractional_semantic_transport_detail::assembleSharedRealChildCapacity(
        realInput,support,out,error);
}

[[nodiscard]] inline bool verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
        const ConservativeSpatialMassTransportInput& realInput,
        const ConservativeSpatialChildSemanticSupport& support,
        const ConservativeSpatialMassTransportResult& supplied,
        std::string& error) {
    ConservativeSpatialMassTransportResult expected;
    if(!fractional_semantic_transport_detail::assembleSharedRealChildCapacity(
            realInput,support,expected,error))return false;
    if(!fractional_semantic_transport_detail::sameResult(expected,supplied))
        return (error="shared-capacity semantic receipt differs from deterministic current-input replay",false);
    error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
