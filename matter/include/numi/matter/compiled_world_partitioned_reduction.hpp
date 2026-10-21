#pragma once

// CPU-side extraction of an anatomically partitioned FEM basis from the
// compiler's immutable CompiledWorld. This preserves the compiled source mesh
// and provenance. It does not enable runtime coordinates x = Bq.

#include "numi/matter/matter.hpp"
#include "numi/matter/partitioned_reduction_basis.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace numi::matter {

enum CompiledReductionAnchorReason : std::uint32_t {
    compiledReductionAnchorAttachment = 1u << 0u,
    compiledReductionAnchorContact = 1u << 1u,
    compiledReductionAnchorThorax = 1u << 2u,
    compiledReductionAnchorVascularBinding = 1u << 3u,
    compiledReductionAnchorVascularCavity = 1u << 4u,
};

struct CompiledWorldPartitionedReductionOptions {
    std::uint32_t minimumCoordinateReductionFactor = 4u;
    // Output is a prerequisite/mapping artifact, not the runtime's existing
    // 32-mode preconditioner ABI. Callers can retain larger candidate bases,
    // while production must validate its consumer's mode limit independently.
    std::uint32_t maximumModeCount = 128u;
    // Only residual-refinement candidates may opt into the conservative
    // fallback that moves a rank-deficient source region to the exact fine
    // complement and removes that region's affine block.
    bool promoteRankDeficientRegionsToFine = false;
    // Contact pairs are eligibility candidates, not current active contacts.
    // The runtime contact owner supplies only presently active global FEM nodes.
    std::span<const std::uint32_t> activeContactAnchorGlobalFEMNodes;
    // Rows whose xyz coordinates stay on the exact fine complement path.
    // They are zeroed in B, but are not promoted to one-hot anchor modes.
    std::span<const std::uint32_t> fineCoordinateOnlyGlobalFEMNodes;
    // CompiledWorld has no semantic thorax selector. These must be explicit
    // global FEM node IDs selected by an anatomy-aware source owner.
    std::span<const std::uint32_t> thoraxAnchorGlobalFEMNodes;
    // Selectors 0 and 6 do not retain source region IDs in CompiledWorld.
    // The source owner may pass the exact authored label sequence, in source
    // tetrahedron order, bound to this object and this physical world.
    std::span<const std::uint32_t> explicitAuthoredTetrahedronRegionLabels;
    std::uint32_t explicitRegionLabelObjectIndex = NM_INVALID_INDEX;
    std::uint64_t explicitRegionLabelWorldPhysicsFingerprint = 0u;
};

enum class CompiledReductionRegionLabelSource : std::uint8_t {
    compiledTetrahedronFlags,
    callerBoundSourcePayload,
};

struct CompiledWorldPartitionedReductionBasis {
    PartitionedReductionBasis basis;
    std::uint32_t objectIndex = NM_INVALID_INDEX;
    CompiledReductionRegionLabelSource regionLabelSource =
        CompiledReductionRegionLabelSource::compiledTetrahedronFlags;
    std::uint64_t sourceWorldPhysicsFingerprint = 0u;
    // One entry per basis source row, retaining compiled and authored-source
    // identities for a future runtime prolongation map.
    std::vector<std::uint32_t> globalFEMNodeIndices;
    std::vector<std::uint32_t> sourceNodeIndices;
    // Aligned with the active input tetrahedra in the exact source order.
    std::vector<std::uint32_t> globalFEMTetrahedronIndices;
    std::vector<std::uint32_t> sourceTetrahedronIndices;
    std::vector<std::uint32_t> tetrahedronRegionLabels;
    // Aligned with basis.anchorNodes, which are object-local row indices.
    std::vector<std::uint32_t> anchorReasonMasks;
    // Regions whose affine blocks were rank deficient after fine-row
    // selection and were therefore promoted wholesale to the fine path.
    std::vector<std::uint32_t> rankPromotedFineRegionLabels;
    std::vector<std::uint32_t> rankPromotedFineGlobalFEMNodeIndices;
    std::string error;

    [[nodiscard]] bool valid() const noexcept {
        const std::size_t tetrahedronCount = tetrahedronRegionLabels.size();
        return error.empty() && basis.valid() &&
            globalFEMNodeIndices.size() == basis.nodeCount &&
            sourceNodeIndices.size() == basis.nodeCount &&
            globalFEMTetrahedronIndices.size() == tetrahedronCount &&
            sourceTetrahedronIndices.size() == tetrahedronCount &&
            anchorReasonMasks.size() == basis.anchorNodes.size();
    }
};

// Exact source-layout token for binding one completion-only witness to one basis.
[[nodiscard]] inline std::uint64_t partitionedReductionSourceLayoutFingerprint(
    const detail::HumanPartitionedPreconditionerSource& source
) noexcept {
    std::uint64_t h=1469598103934665603ull;
    const auto mix=[&h](std::uint32_t x){for(unsigned b=0;b<4;++b){h^=(x>>(8*b))&255u;h*=1099511628211ull;}};
    mix(source.objectIndex); mix(source.basisNodeCount); mix(source.regionCount);
    mix(source.blockModeCount); mix(source.modeCount); mix(source.freeCoordinateCount);
    mix(source.minimumCoordinateReductionFactor);
    for(auto x:source.regionLabels) mix(x);
    for(auto x:source.globalFEMNodeIndices) mix(x);
    for(auto x:source.blockModeIndices) mix(x);
    for(const auto& v:source.basisValues){mix(std::bit_cast<std::uint32_t>(v.x));mix(std::bit_cast<std::uint32_t>(v.y));mix(std::bit_cast<std::uint32_t>(v.z));mix(std::bit_cast<std::uint32_t>(v.w));}
    return h==0u?1u:h;
}
struct PartitionedReductionResidualWitnessNode {
    std::uint32_t globalFEMNodeIndex=NM_INVALID_INDEX, regionLabel=NM_INVALID_INDEX;
    float residualL2=0.0f, residualMaxAbs=0.0f;
};
struct PartitionedReductionResidualWitness {
    std::uint64_t sourceWorldPhysicsFingerprint=0u, sourceLayoutFingerprint=0u;
    std::uint32_t objectIndex=NM_INVALID_INDEX, basisNodeCount=0u, regionCount=0u;
    std::uint32_t blockModeCount=0u, modeCount=0u;
    std::vector<PartitionedReductionResidualWitnessNode> candidates;
};

struct PartitionedReductionResidualEnrichmentSelection {
    std::size_t candidateCount = 0u;
    std::size_t eligibleCandidateCount = 0u;
    std::size_t perRegionCandidateCount = 0u;
    std::size_t selectedCandidateCount = 0u;
    std::size_t droppedByGlobalCap = 0u;
    std::uint32_t perRegionMaximum = 0u;
    std::uint32_t globalMaximum = 0u;
    std::uint64_t availableFineRowBudget = 0u;
    bool globalCapApplied = false;
};
namespace residual_witness_detail {
[[nodiscard]] inline bool property(std::string_view object, std::string_view key,
                                   std::string_view& value) {
    std::string token; token.reserve(key.size()+2u); token.push_back('"');
    token.append(key); token.push_back('"');
    std::size_t at=object.find(token);
    if(at==std::string_view::npos)return false;
    at+=token.size();
    while(at<object.size()&&(object[at]==' '||object[at]=='\n'||object[at]=='\r'||object[at]=='\t'))++at;
    if(at>=object.size()||object[at++]!=':')return false;
    while(at<object.size()&&(object[at]==' '||object[at]=='\n'||object[at]=='\r'||object[at]=='\t'))++at;
    const std::size_t begin=at;
    if(at>=object.size())return false;
    if(object[at]=='"'){
        ++at; bool escaped=false;
        for(;at<object.size();++at){const char c=object[at];if(escaped){escaped=false;continue;}if(c=='\\'){escaped=true;continue;}if(c=='"'){++at;value=object.substr(begin,at-begin);return true;}}
        return false;
    }
    if(object[at]=='{'||object[at]=='['){
        const char open=object[at],close=open=='{'?'}':']'; int depth=0; bool inString=false,escaped=false;
        for(;at<object.size();++at){const char c=object[at];if(inString){if(escaped)escaped=false;else if(c=='\\')escaped=true;else if(c=='"')inString=false;continue;}if(c=='"'){inString=true;continue;}if(c==open)++depth;else if(c==close&&--depth==0){++at;value=object.substr(begin,at-begin);return true;}}
        return false;
    }
    while(at<object.size()&&object[at]!=','&&object[at]!=']'&&object[at]!='}'&&object[at]!=' '&&object[at]!='\n'&&object[at]!='\r'&&object[at]!='\t')++at;
    if(at==begin)return false; value=object.substr(begin,at-begin); return true;
}
[[nodiscard]] inline bool arrayObjects(std::string_view array,
                                       std::vector<std::string_view>& objects) {
    if(array.size()<2u||array.front()!='['||array.back()!=']')return false;
    std::size_t i=1u; const std::size_t end=array.size()-1u;
    while(i<end){while(i<end&&(array[i]==' '||array[i]=='\n'||array[i]=='\r'||array[i]=='\t'||array[i]==','))++i;if(i==end)break;if(array[i]!='{')return false;const auto begin=i;int depth=0;bool inString=false,escaped=false;for(;i<end;++i){const char c=array[i];if(inString){if(escaped)escaped=false;else if(c=='\\')escaped=true;else if(c=='"')inString=false;continue;}if(c=='"'){inString=true;continue;}if(c=='{')++depth;else if(c=='}'&&--depth==0){++i;objects.push_back(array.substr(begin,i-begin));break;}}if(depth!=0)return false;}
    return true;
}
template<class T> [[nodiscard]] inline bool unsignedNumber(std::string_view value,T& output) noexcept {
    if(value.empty()||value.front()=='-')return false; T parsed{};const auto result=std::from_chars(value.data(),value.data()+value.size(),parsed);if(result.ec!=std::errc{}||result.ptr!=value.data()+value.size())return false;output=parsed;return true;
}
[[nodiscard]] inline bool floatNumber(std::string_view value,float& output) noexcept {
    float parsed=0.0f;const auto result=std::from_chars(value.data(),value.data()+value.size(),parsed,std::chars_format::general);if(result.ec!=std::errc{}||result.ptr!=value.data()+value.size()||!std::isfinite(parsed))return false;output=parsed;return true;
}
[[nodiscard]] inline bool hexNumber(std::string_view value,std::uint64_t& output) noexcept {
    if(value.size()<3u||value.front()!='"'||value.back()!='"')return false;
    value.remove_prefix(1u);value.remove_suffix(1u);if(value.starts_with("0x")||value.starts_with("0X"))value.remove_prefix(2u);
    std::uint64_t parsed=0u;const auto result=std::from_chars(value.data(),value.data()+value.size(),parsed,16);if(result.ec!=std::errc{}||result.ptr!=value.data()+value.size())return false;output=parsed;return true;
}
}

// Parse the existing final fem_true_residual_cycle JSON record. The caller
// supplies one line; stale or ambiguous witnesses are rejected by fingerprints.
[[nodiscard]] inline bool parsePartitionedReductionResidualWitnessTrace(
    std::string_view line, PartitionedReductionResidualWitness& output,
    std::string& error) {
    using namespace residual_witness_detail;
    const auto fail=[&](const char* message){error=message;return false;};
    std::string_view schema,finalCycle,witnessValue;
    if(!property(line,"schema",schema)||schema!="2"||
       !property(line,"final_cycle",finalCycle)||finalCycle!="true"||
       !property(line,"hybrid_physical_residual_witness",witnessValue))
        return fail("trace line is not a final hybrid physical residual witness");
    PartitionedReductionResidualWitness candidate;
    std::string_view value;
    if(!property(witnessValue,"source_physics_fingerprint",value)||!hexNumber(value,candidate.sourceWorldPhysicsFingerprint)||
       !property(witnessValue,"basis_layout_fingerprint_fnv1a64",value)||!hexNumber(value,candidate.sourceLayoutFingerprint)||
       !property(witnessValue,"object_index",value)||!unsignedNumber(value,candidate.objectIndex)||
       !property(witnessValue,"basis_node_count",value)||!unsignedNumber(value,candidate.basisNodeCount)||
       !property(witnessValue,"region_count",value)||!unsignedNumber(value,candidate.regionCount)||
       !property(witnessValue,"block_mode_count",value)||!unsignedNumber(value,candidate.blockModeCount)||
       !property(witnessValue,"mode_count",value)||!unsignedNumber(value,candidate.modeCount))
        return fail("trace witness metadata is malformed");
    std::string_view regionsValue;
    if(!property(witnessValue,"regions",regionsValue))return fail("trace witness regions are absent");
    std::vector<std::string_view> regions;
    if(!arrayObjects(regionsValue,regions)||regions.size()!=candidate.regionCount)
        return fail("trace witness region array does not match its declared count");
    std::set<std::uint32_t> regionLabels;
    for(const auto regionObject:regions){
        std::uint32_t label=0u; std::string_view nodesValue;
        if(!property(regionObject,"region_label",value)||!unsignedNumber(value,label)||
           !regionLabels.insert(label).second||!property(regionObject,"nodes",nodesValue))
            return fail("trace witness region record is malformed");
        std::vector<std::string_view> nodes;
        if(!arrayObjects(nodesValue,nodes)||nodes.size()>4u)
            return fail("trace witness node list exceeds its bounded regional sample");
        for(const auto nodeObject:nodes){
            PartitionedReductionResidualWitnessNode node;node.regionLabel=label;
            if(!property(nodeObject,"global_node",value)||!unsignedNumber(value,node.globalFEMNodeIndex)||
               !property(nodeObject,"residual_l2",value)||!floatNumber(value,node.residualL2)||
               !property(nodeObject,"residual_max_abs",value)||!floatNumber(value,node.residualMaxAbs))
                return fail("trace witness node record is malformed");
            candidate.candidates.push_back(node);
        }
    }
    if(candidate.sourceWorldPhysicsFingerprint==0u||candidate.sourceLayoutFingerprint==0u||
       candidate.objectIndex==NM_INVALID_INDEX||candidate.basisNodeCount==0u||
       candidate.regionCount==0u||candidate.modeCount==0u||candidate.candidates.empty())
        return fail("trace witness metadata or candidate set is empty");
    output=std::move(candidate);error.clear();return true;
}

// Select only exact mapped, region-owned interior rows; keep existing fine rows.
[[nodiscard]] inline bool selectPartitionedResidualEnrichmentFineNodes(
    const detail::HumanPartitionedPreconditionerSource& source,
    const PartitionedReductionResidualWitness& witness,
    std::span<const std::uint32_t> existingFineRows,
    std::uint32_t requestedMinimumCoordinateReductionFactor,
    std::uint32_t maximumPerRegion, std::uint32_t maximumTotal,
    std::vector<std::uint32_t>& outputFineRows, std::string& error,
    PartitionedReductionResidualEnrichmentSelection* selection = nullptr) {
    const auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    outputFineRows.clear();
    if (selection != nullptr) {
        *selection = {};
        selection->perRegionMaximum = maximumPerRegion;
        selection->globalMaximum = maximumTotal;
    }
    if (!source.enabled || source.sourceWorldPhysicsFingerprint == 0u ||
        witness.sourceWorldPhysicsFingerprint != source.sourceWorldPhysicsFingerprint ||
        witness.sourceLayoutFingerprint != partitionedReductionSourceLayoutFingerprint(source) ||
        witness.objectIndex != source.objectIndex ||
        witness.basisNodeCount != source.basisNodeCount ||
        witness.regionCount != source.regionCount ||
        witness.blockModeCount != source.blockModeCount ||
        witness.modeCount != source.modeCount)
        return fail("residual witness does not match the current Human source layout");
    if (source.regionCount == 0u || source.blockModeCount == 0u ||
        std::uint64_t(source.regionCount) * source.blockModeCount != source.modeCount ||
        requestedMinimumCoordinateReductionFactor < 2u ||
        source.regionLabels.size() != source.regionCount ||
        source.blockModeIndices.size() != source.modeCount ||
        source.globalFEMNodeIndices.size() != source.basisNodeCount ||
        !std::is_sorted(source.globalFEMNodeIndices.begin(),
                        source.globalFEMNodeIndices.end()) ||
        std::adjacent_find(source.globalFEMNodeIndices.begin(),
                           source.globalFEMNodeIndices.end()) !=
            source.globalFEMNodeIndices.end() ||
        source.basisValues.size() !=
            std::size_t(source.modeCount) * source.basisNodeCount ||
        maximumPerRegion == 0u || maximumPerRegion > 8u ||
        maximumTotal == 0u ||
        maximumTotal > kHumanPartitionedBasisMaximumRegionCount * 4u)
        return fail("residual enrichment source or bounded selection dimensions are invalid");

    const std::set<std::uint32_t> uniqueLabels(
        source.regionLabels.begin(), source.regionLabels.end());
    if (uniqueLabels.size() != source.regionCount)
        return fail("residual enrichment source region labels are not unique");
    std::vector<std::uint8_t> modeSeen(source.modeCount, 0u);
    for (const auto mode : source.blockModeIndices) {
        if (mode >= source.modeCount || modeSeen[mode] != 0u)
            return fail("residual enrichment source mode map is not a permutation");
        modeSeen[mode] = 1u;
    }

    const std::uint64_t required =
        std::uint64_t(source.modeCount) * requestedMinimumCoordinateReductionFactor;
    if (source.freeCoordinateCount < required)
        return fail("requested residual-enrichment reduction target is infeasible");
    const std::uint64_t availableRows =
        (source.freeCoordinateCount - required) / 3u;
    if (selection != nullptr)
        selection->availableFineRowBudget = availableRows;

    const std::set<std::uint32_t> existing(
        existingFineRows.begin(), existingFineRows.end());
    std::set<std::uint32_t> seen;
    std::vector<std::vector<PartitionedReductionResidualWitnessNode>> ranked(
        source.regionCount);
    if (selection != nullptr)
        selection->candidateCount = witness.candidates.size();
    for (const auto& candidate : witness.candidates) {
        if (candidate.globalFEMNodeIndex == NM_INVALID_INDEX ||
            candidate.regionLabel == NM_INVALID_INDEX ||
            !std::isfinite(candidate.residualL2) || candidate.residualL2 < 0.0f ||
            !std::isfinite(candidate.residualMaxAbs) || candidate.residualMaxAbs < 0.0f ||
            candidate.residualMaxAbs > candidate.residualL2 * (1.0f + 1.0e-5f) + 1.0e-12f ||
            !seen.insert(candidate.globalFEMNodeIndex).second)
            return fail("residual witness contains an invalid or duplicate node record");
        const auto it = std::lower_bound(source.globalFEMNodeIndices.begin(),
                                         source.globalFEMNodeIndices.end(),
                                         candidate.globalFEMNodeIndex);
        if (it == source.globalFEMNodeIndices.end() ||
            *it != candidate.globalFEMNodeIndex)
            return fail("residual witness node is outside the exact source map");
        const auto row = static_cast<std::uint32_t>(
            std::distance(source.globalFEMNodeIndices.begin(), it));
        std::uint32_t owner = NM_INVALID_INDEX;
        double strongest = 0.0;
        for (std::uint32_t region = 0u; region < source.regionCount; ++region) {
            double energy = 0.0;
            for (std::uint32_t local = 0u; local < source.blockModeCount; ++local) {
                const auto mode = source.blockModeIndices[
                    std::size_t(region) * source.blockModeCount + local];
                if (mode >= source.modeCount)
                    return fail("residual witness mode map escapes basis");
                const auto value = source.basisValues[
                    std::size_t(mode) * source.basisNodeCount + row];
                if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                    !std::isfinite(value.z) || !std::isfinite(value.w))
                    return fail("residual witness source basis is non-finite");
                energy += double(value.x) * value.x +
                    double(value.y) * value.y + double(value.z) * value.z;
            }
            if (energy > strongest) {
                strongest = energy;
                owner = region;
            }
        }
        if (owner == NM_INVALID_INDEX || strongest == 0.0 ||
            source.regionLabels[owner] != candidate.regionLabel)
            return fail("residual witness region does not own a reducible row");
        if (!existing.contains(candidate.globalFEMNodeIndex) &&
            candidate.residualL2 > 0.0f) {
            ranked[owner].push_back(candidate);
            if (selection != nullptr) ++selection->eligibleCandidateCount;
        }
    }

    const auto descendingResidualThenNode = [](const auto& a, const auto& b) {
        return a.residualL2 != b.residualL2
            ? a.residualL2 > b.residualL2
            : a.globalFEMNodeIndex < b.globalFEMNodeIndex;
    };
    std::vector<PartitionedReductionResidualWitnessNode> selected;
    for (auto& region : ranked) {
        std::sort(region.begin(), region.end(), descendingResidualThenNode);
        const std::size_t count = std::min<std::size_t>(maximumPerRegion, region.size());
        selected.insert(selected.end(), region.begin(), region.begin() + count);
    }
    if (selection != nullptr)
        selection->perRegionCandidateCount = selected.size();
    std::sort(selected.begin(), selected.end(), descendingResidualThenNode);
    const std::size_t beforeGlobalCap = selected.size();
    if (selected.size() > maximumTotal) {
        selected.resize(maximumTotal);
        if (selection != nullptr) {
            selection->globalCapApplied = true;
            selection->droppedByGlobalCap = beforeGlobalCap - selected.size();
        }
    }
    if (selection != nullptr)
        selection->selectedCandidateCount = selected.size();
    if (selected.size() > availableRows)
        return fail("selected residual rows exceed the available fine-row budget for the requested reduction target");

    std::vector<std::uint32_t> merged(existing.begin(), existing.end());
    merged.reserve(merged.size() + selected.size());
    for (const auto& candidate : selected)
        merged.push_back(candidate.globalFEMNodeIndex);
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    outputFineRows = std::move(merged);
    error.clear();
    return true;
}

namespace compiled_reduction_detail {

[[nodiscard]] inline std::size_t firstFailedCholeskyPivot(
    const std::span<const double> matrix, const std::size_t dimension) {
    if (dimension == 0u || matrix.size() != dimension * dimension) return 0u;
    std::vector<double> lower(dimension * dimension, 0.0);
    double scale = 0.0;
    for (std::size_t i = 0u; i < dimension; ++i)
        scale = std::max(scale, std::abs(matrix[i * dimension + i]));
    if (!(scale > 0.0) || !std::isfinite(scale)) return 0u;
    const double threshold = scale * 1.0e-12;
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrix[row * dimension + column];
            for (std::size_t k = 0u; k < column; ++k)
                value -= lower[row * dimension + k] *
                    lower[column * dimension + k];
            if (row == column) {
                if (!(value > threshold) || !std::isfinite(value)) return row;
                lower[row * dimension + column] = std::sqrt(value);
            } else {
                lower[row * dimension + column] = value /
                    lower[column * dimension + column];
            }
        }
    }
    return dimension;
}

[[nodiscard]] inline std::array<double, 3> affineValue(
    const std::array<double, 3>& translation,
    const std::array<double, 9>& gradient,
    const std::array<double, 3>& position
) noexcept {
    return {
        translation[0] + gradient[0] * position[0] +
            gradient[1] * position[1] + gradient[2] * position[2],
        translation[1] + gradient[3] * position[0] +
            gradient[4] * position[1] + gradient[5] * position[2],
        translation[2] + gradient[6] * position[0] +
            gradient[7] * position[1] + gradient[8] * position[2],
    };
}

[[nodiscard]] inline bool finalizeAffineModes(
    PartitionedReductionBasis& basis,
    const std::uint32_t minimumCoordinateReductionFactor,
    const std::uint32_t maximumModeCount,
    const std::span<const std::array<double, 3>> positions,
    const std::span<const double> masses,
    const std::span<const std::uint8_t> fixedMask,
    std::uint32_t* rankDeficientRegionLabel = nullptr
) {
    if (rankDeficientRegionLabel != nullptr)
        *rankDeficientRegionLabel = NM_INVALID_INDEX;
    if (positions.size() != basis.nodeCount || masses.size() != basis.nodeCount ||
        fixedMask.size() != basis.nodeCount || basis.regions.empty()) {
        basis.error = "compiled affine basis source arrays do not match the extracted mesh";
        return false;
    }

    basis.modes.clear();
    basis.values.clear();
    basis.projectedMass.clear();
    const std::size_t nodeCount = basis.nodeCount;
    const std::size_t regionCount = basis.regions.size();
    std::size_t freeAnchorCount = 0u;
    for (const std::uint32_t node : basis.anchorNodes) {
        if (node < nodeCount && fixedMask[node] == 0u) ++freeAnchorCount;
    }
    const std::uint64_t requestedModes =
        static_cast<std::uint64_t>(regionCount) * 12u +
        static_cast<std::uint64_t>(freeAnchorCount) * 3u;
    if (requestedModes == 0u || requestedModes > maximumModeCount ||
        requestedModes > std::numeric_limits<std::uint32_t>::max()) {
        basis.error = "compiled affine basis exceeds its configured mode limit after sensitive-anchor preservation";
        return false;
    }
    if (requestedModes * minimumCoordinateReductionFactor >
        basis.freeCoordinateCount) {
        basis.error = "compiled affine basis misses the requested minimum coordinate reduction after sensitive-anchor preservation";
        return false;
    }

    const auto append = [&](const PartitionedReductionMode mode,
                            const std::vector<std::array<double, 3>>& field) {
        basis.modes.push_back(mode);
        for (const auto& value : field) {
            basis.values.push_back({static_cast<float>(value[0]),
                                    static_cast<float>(value[1]),
                                    static_cast<float>(value[2]), 0.0f});
        }
    };
    for (std::size_t region = 0u; region < regionCount; ++region) {
        const PartitionedReductionRegion& metadata = basis.regions[region];
        for (std::uint32_t outputAxis = 0u; outputAxis < 3u; ++outputAxis) {
            std::vector<std::array<double, 3>> field(nodeCount);
            for (std::size_t node = 0u; node < nodeCount; ++node) {
                if (fixedMask[node] != 0u) continue;
                const double weight = basis.partitionWeights[
                    node * basis.partitionCount + region];
                field[node][outputAxis] = weight;
            }
            append({PartitionedReductionModeKind::regionalTranslation,
                    metadata.sourceLabel, outputAxis, NM_INVALID_INDEX, 1.0},
                   field);
        }
        for (std::uint32_t outputAxis = 0u; outputAxis < 3u; ++outputAxis) {
            for (std::uint32_t inputAxis = 0u; inputAxis < 3u; ++inputAxis) {
                std::vector<std::array<double, 3>> field(nodeCount);
                for (std::size_t node = 0u; node < nodeCount; ++node) {
                    if (fixedMask[node] != 0u) continue;
                    const double weight = basis.partitionWeights[
                        node * basis.partitionCount + region];
                    field[node][outputAxis] = weight *
                        (positions[node][inputAxis] - metadata.center[inputAxis]) /
                        metadata.rotationRadius;
                }
                append({PartitionedReductionModeKind::regionalAffineGradient,
                        metadata.sourceLabel, outputAxis * 3u + inputAxis,
                        NM_INVALID_INDEX, metadata.rotationRadius}, field);
            }
        }
    }
    for (const std::uint32_t anchorNode : basis.anchorNodes) {
        if (anchorNode >= nodeCount || fixedMask[anchorNode] != 0u) continue;
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            std::vector<std::array<double, 3>> field(nodeCount);
            field[anchorNode][component] = 1.0;
            append({PartitionedReductionModeKind::anchorTranslation,
                    NM_INVALID_INDEX, component, anchorNode, 1.0}, field);
        }
    }
    basis.modeCount = static_cast<std::uint32_t>(basis.modes.size());
    if (basis.modeCount != requestedModes) {
        basis.error = "compiled affine basis mode accounting is inconsistent";
        return false;
    }

    basis.projectedMass.assign(
        static_cast<std::size_t>(basis.modeCount) * basis.modeCount, 0.0);
    std::vector<std::uint32_t> activeModes;
    for (std::size_t node = 0u; node < nodeCount; ++node) {
        activeModes.clear();
        for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
            const auto& value = basis.values[
                static_cast<std::size_t>(mode) * nodeCount + node];
            if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f)
                activeModes.push_back(mode);
        }
        for (const std::uint32_t row : activeModes) {
            const auto& a = basis.values[
                static_cast<std::size_t>(row) * nodeCount + node];
            for (const std::uint32_t column : activeModes) {
                const auto& b = basis.values[
                    static_cast<std::size_t>(column) * nodeCount + node];
                basis.projectedMass[static_cast<std::size_t>(row) *
                                    basis.modeCount + column] +=
                    masses[node] *
                    (static_cast<double>(a.x) * b.x +
                     static_cast<double>(a.y) * b.y +
                     static_cast<double>(a.z) * b.z);
            }
        }
    }
    if (!partitioned_reduction_detail::choleskyPositiveDefinite(
            basis.projectedMass, basis.modeCount)) {
        // Identify the first anatomical block that makes the deterministic
        // regional prefix lose rank.  The following fallback can then retire
        // that entire block rather than dropping arbitrary modes or rows.
        if (rankDeficientRegionLabel != nullptr) {
            constexpr std::size_t modesPerRegion = 12u;
            const std::size_t pivot = firstFailedCholeskyPivot(
                basis.projectedMass, basis.modeCount);
            if (pivot < regionCount * modesPerRegion) {
                *rankDeficientRegionLabel = basis.regions[
                    pivot / modesPerRegion].sourceLabel;
            }
        }
        basis.error = "compiled affine basis is rank deficient under the exact cooked nodal masses";
        return false;
    }
    return true;
}

} // namespace compiled_reduction_detail

// Extracts one immutable, non-mutable FEM object's exact compiled source rows.
// Region labels come from compiler-retained flags or a caller-bound source
// payload; this function never infers anatomy from pooled materials.
// Attachments, caller-selected currently active contact nodes, every vascular
// tissue/cavity node on the object, and caller-selected thorax nodes are held
// as one-hot fine anchor partitions.
[[nodiscard]] inline CompiledWorldPartitionedReductionBasis
buildCompiledWorldPartitionedReductionBasis(
    const CompiledWorld& world,
    const std::uint32_t objectIndex,
    const CompiledWorldPartitionedReductionOptions& options = {}
) {
    using namespace compiled_reduction_detail;
    CompiledWorldPartitionedReductionBasis output;
    output.objectIndex = objectIndex;
    const auto fail = [&](const char* message) {
        output.error = message;
        output.basis.error = message;
        output.basis.values.clear();
        output.basis.modes.clear();
        output.basis.projectedMass.clear();
        return output;
    };
    if (objectIndex >= world.objects.size())
        return fail("compiled reduction object index is out of range");
    if (options.minimumCoordinateReductionFactor < 2u ||
        options.maximumModeCount == 0u || options.maximumModeCount > kPartitionedReductionMaximumConstructionModeCount)
        return fail("compiled reduction factor or mode limit is invalid");
    const NMContinuumObjectGPU& object = world.objects[objectIndex];
    if (object.representation != NM_REPRESENTATION_FEM ||
        object.stateCount == 0u || object.elementCount == 0u)
        return fail("compiled reduction requires a nonempty FEM object");
    if ((object.flags & NM_OBJECT_MUTABLE_TOPOLOGY) != 0u)
        return fail("compiled reduction refuses mutable FEM topology because its source-node map is not immutable");
    const std::uint32_t regionalSelector =
        world.mixedSolver.executionBudgets.z;
    const bool compiledLabelsAvailable =
        regionalSelector >= 2u && regionalSelector <= 5u;
    output.sourceWorldPhysicsFingerprint = world.physicsFingerprint;

    const std::uint64_t stateEnd =
        static_cast<std::uint64_t>(object.stateOffset) + object.stateCount;
    const std::uint64_t elementEnd =
        static_cast<std::uint64_t>(object.elementOffset) + object.elementCount;
    if (stateEnd > world.fem.nodes.size() ||
        stateEnd > world.fem.topologyNodes.size() ||
        elementEnd > world.fem.tetrahedra.size() ||
        stateEnd > std::numeric_limits<std::uint32_t>::max() ||
        elementEnd > std::numeric_limits<std::uint32_t>::max())
        return fail("compiled FEM object node or tetrahedron range exceeds its cooked arena");

    std::vector<std::array<double, 3>> positions;
    std::vector<double> masses;
    std::vector<std::uint8_t> fixedMask;
    std::vector<std::uint32_t> globalNodeIndices;
    std::vector<std::uint32_t> sourceNodeIndices;
    std::vector<std::int64_t> localIndex(world.fem.nodes.size(), -1);
    std::set<std::uint32_t> sourceNodeSet;
    for (std::uint32_t global = object.stateOffset; global < stateEnd; ++global) {
        const NMFEMTopologyNodeGPU& topology = world.fem.topologyNodes[global];
        if (topology.identity.y != objectIndex ||
            (topology.identity.w & NM_TOPOLOGY_ACTIVE) == 0u) continue;
        if (topology.identity.z != object.topologyGeneration ||
            !sourceNodeSet.insert(topology.identity.x).second)
            return fail("compiled FEM source ancestry is duplicated or has a stale topology generation");
        const NMFEMNodeStateGPU& node = world.fem.nodes[global];
        const nm_float4 position = node.restAndFixed;
        const nm_float4 mass = node.positionAndMass;
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(position.w) ||
            !std::isfinite(mass.w) || !(mass.w > 0.0f))
            return fail("compiled FEM source rest position or cooked nodal mass is non-finite or non-positive");
        std::uint8_t fixed = 0u;
        if (position.w == 1.0f || position.w == 3.0f) fixed = 1u;
        else if (position.w != 0.0f && position.w != 2.0f)
            return fail("compiled FEM source contains an unknown constraint tag");
        localIndex[global] = static_cast<std::int64_t>(positions.size());
        positions.push_back({position.x, position.y, position.z});
        masses.push_back(mass.w);
        fixedMask.push_back(fixed);
        globalNodeIndices.push_back(global);
        sourceNodeIndices.push_back(topology.identity.x);
    }
    if (positions.empty() ||
        positions.size() > std::numeric_limits<std::uint32_t>::max())
        return fail("compiled FEM object has no active source nodes or exceeds the index range");
    const std::vector<std::uint8_t> physicallyFixedMask = fixedMask;
    std::set<std::uint32_t> fineCoordinateOnly;
    for (const std::uint32_t global : options.fineCoordinateOnlyGlobalFEMNodes) {
        if (global >= localIndex.size() || localIndex[global] < 0 ||
            !fineCoordinateOnly.insert(global).second)
            return fail("fine-coordinate-only source row is inactive, foreign, or duplicated");
        fixedMask[static_cast<std::size_t>(localIndex[global])] = 1u;
    }

    std::vector<std::array<std::uint32_t, 4>> tetrahedra;
    std::vector<std::uint32_t> compiledRegionLabels;
    std::vector<std::uint32_t> globalTetrahedronIndices;
    std::vector<std::uint32_t> sourceTetrahedronIndices;
    for (std::uint32_t index = object.elementOffset; index < elementEnd; ++index) {
        const NMTetrahedronGPU& tet = world.fem.tetrahedra[index];
        if ((tet.identity.w & NM_OBJECT_ACTIVE) == 0u) continue;
        if (tet.identity.y != objectIndex)
            return fail("active compiled FEM tetrahedron range contains a foreign object");
        const std::array<std::uint32_t, 4> globalNodes{
            tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w};
        std::array<std::uint32_t, 4> localNodes{};
        for (std::size_t corner = 0u; corner < 4u; ++corner) {
            const std::uint32_t global = globalNodes[corner];
            if (global >= localIndex.size() || localIndex[global] < 0)
                return fail("active compiled FEM tetrahedron references an omitted or foreign source node");
            localNodes[corner] =
                static_cast<std::uint32_t>(localIndex[global]);
        }
        tetrahedra.push_back(localNodes);
        compiledRegionLabels.push_back((tet.identity.w &
            NM_TETRAHEDRON_PRECONDITIONER_REGION_MASK) >>
            NM_TETRAHEDRON_PRECONDITIONER_REGION_SHIFT);
        globalTetrahedronIndices.push_back(index);
        sourceTetrahedronIndices.push_back(index - object.elementOffset);
    }
    if (tetrahedra.empty())
        return fail("compiled FEM object has no active source tetrahedra");

    std::vector<std::uint32_t> regionLabels;
    if (compiledLabelsAvailable) {
        output.regionLabelSource =
            CompiledReductionRegionLabelSource::compiledTetrahedronFlags;
        regionLabels = compiledRegionLabels;
        if (!options.explicitAuthoredTetrahedronRegionLabels.empty() &&
            !std::equal(regionLabels.begin(), regionLabels.end(),
                        options.explicitAuthoredTetrahedronRegionLabels.begin(),
                        options.explicitAuthoredTetrahedronRegionLabels.end()))
            return fail("explicit source region labels disagree with the compiled source-bound labels");
    } else {
        if (options.explicitRegionLabelObjectIndex != objectIndex ||
            options.explicitRegionLabelWorldPhysicsFingerprint == 0u ||
            options.explicitRegionLabelWorldPhysicsFingerprint !=
                world.physicsFingerprint ||
            options.explicitAuthoredTetrahedronRegionLabels.size() !=
                tetrahedra.size())
            return fail("selector 0/6 requires exact source labels bound to the selected FEM object and compiled-world physics fingerprint");
        output.regionLabelSource =
            CompiledReductionRegionLabelSource::callerBoundSourcePayload;
        regionLabels.assign(
            options.explicitAuthoredTetrahedronRegionLabels.begin(),
            options.explicitAuthoredTetrahedronRegionLabels.end());
        if (std::any_of(regionLabels.begin(), regionLabels.end(),
                        [](const std::uint32_t label) { return label == NM_INVALID_INDEX; }))
            return fail("explicit source region label contains the invalid sentinel");
    }

    std::map<std::uint32_t, std::uint32_t> anchorReasons;
    const auto addAnchor = [&](const std::uint32_t global,
                               const std::uint32_t reason) -> bool {
        if (global >= localIndex.size() || localIndex[global] < 0) return false;
        anchorReasons[global] |= reason;
        return true;
    };
    for (const NMFEMHumanAttachmentGPU& attachment : world.fem.humanAttachments) {
        if (attachment.identity.z != objectIndex) continue;
        if (!addAnchor(attachment.identity.x,
                       compiledReductionAnchorAttachment))
            return fail("compiled Human attachment references an inactive or foreign FEM source node");
    }
    std::set<std::uint32_t> compiledContactCandidates;
    for (const NMContactPairGPU& pair : world.contact.pairs) {
        if (pair.objectIndex != objectIndex) continue;
        if (pair.continuumNode < world.dispatch.gridNodeCount)
            return fail("compiled FEM contact candidate uses an MPM rather than FEM unified-node index");
        const std::uint64_t global =
            static_cast<std::uint64_t>(pair.continuumNode) -
            world.dispatch.gridNodeCount;
        if (global >= world.fem.nodes.size() || global > UINT32_MAX)
            return fail("compiled FEM contact candidate is outside the FEM node arena");
        compiledContactCandidates.insert(static_cast<std::uint32_t>(global));
    }
    for (const std::uint32_t global :
         options.activeContactAnchorGlobalFEMNodes) {
        if (!compiledContactCandidates.contains(global) ||
            !addAnchor(global, compiledReductionAnchorContact))
            return fail("active-contact anchor is not an eligible compiled contact node of the selected FEM object");
    }
    for (const NMVascularTissueBindingGPU& binding :
         world.vascular.tissueBindings) {
        const std::uint32_t global = binding.identity.y;
        if (global < localIndex.size() && localIndex[global] >= 0)
            anchorReasons[global] |= compiledReductionAnchorVascularBinding;
    }
    for (const NMVascularCavityFaceGPU& face : world.vascular.cavityFaces) {
        if (face.identity.x >= world.vascular.cavities.size())
            return fail("compiled vascular cavity face references a missing cavity");
        if (world.vascular.cavities[face.identity.x].identity.z != objectIndex)
            continue;
        for (const std::uint32_t global :
             {face.nodes.x, face.nodes.y, face.nodes.z}) {
            if (!addAnchor(global, compiledReductionAnchorVascularCavity))
                return fail("compiled vascular cavity face references an inactive or foreign FEM source node");
        }
    }
    for (const std::uint32_t global : options.thoraxAnchorGlobalFEMNodes) {
        if (!addAnchor(global, compiledReductionAnchorThorax))
            return fail("caller-selected thorax anchor is not an active node of the selected compiled FEM object");
    }

    std::vector<std::uint32_t> localAnchors;
    std::vector<std::uint32_t> anchorReasonMasks;
    localAnchors.reserve(anchorReasons.size());
    anchorReasonMasks.reserve(anchorReasons.size());
    for (const auto& [global, reason] : anchorReasons) {
        localAnchors.push_back(static_cast<std::uint32_t>(localIndex[global]));
        anchorReasonMasks.push_back(reason);
    }

    PartitionedReductionBasisOptions baseOptions;
    baseOptions.minimumCoordinateReductionFactor =
        options.minimumCoordinateReductionFactor;
    // Reuse the existing helper's validated source partition construction.
    // The future runtime preconditioner still has its own smaller ABI limit.
    baseOptions.maximumModeCount = kPartitionedReductionMaximumConstructionModeCount;
    std::vector<std::uint8_t> basisFixedMask = fixedMask;
    std::set<std::uint32_t> anchorSet(localAnchors.begin(), localAnchors.end());
    std::map<std::uint32_t, std::set<std::uint32_t>> regionNodes;
    for (std::size_t tet = 0u; tet < tetrahedra.size(); ++tet)
        for (const std::uint32_t node : tetrahedra[tet])
            regionNodes[regionLabels[tet]].insert(node);
    std::set<std::uint32_t> promotedRegions;
    std::set<std::uint32_t> promotedFineNodes;
    for (;;) {
        std::vector<std::array<std::uint32_t, 4>> activeTetrahedra;
        std::vector<std::uint32_t> activeRegionLabels;
        activeTetrahedra.reserve(tetrahedra.size());
        activeRegionLabels.reserve(regionLabels.size());
        for (std::size_t tet = 0u; tet < tetrahedra.size(); ++tet) {
            if (promotedRegions.contains(regionLabels[tet])) continue;
            activeTetrahedra.push_back(tetrahedra[tet]);
            activeRegionLabels.push_back(regionLabels[tet]);
        }
        if (activeTetrahedra.empty())
            return fail("rank-deficient region promotion would remove every affine region");
        const PartitionedReductionBasisInput input{
            positions, masses, basisFixedMask, activeTetrahedra,
            activeRegionLabels, localAnchors};
        output.basis = buildPartitionedReductionBasis(input, baseOptions);
        if (!output.basis.valid()) {
            output.error = output.basis.error;
            return output;
        }
        std::uint32_t deficientRegion = NM_INVALID_INDEX;
        if (finalizeAffineModes(output.basis,
                                options.minimumCoordinateReductionFactor,
                                options.maximumModeCount, positions, masses,
                                basisFixedMask, &deficientRegion)) {
            output.error.clear();
            break;
        }
        output.error = output.basis.error;
        if (deficientRegion != NM_INVALID_INDEX)
            output.error += " source_region=" + std::to_string(deficientRegion);
        if (!options.promoteRankDeficientRegionsToFine ||
            deficientRegion == NM_INVALID_INDEX ||
            !promotedRegions.insert(deficientRegion).second)
            return output;
        const auto foundNodes = regionNodes.find(deficientRegion);
        if (foundNodes == regionNodes.end())
            return fail("rank-deficient affine block has no exact source-region node map");
        for (const std::uint32_t node : foundNodes->second) {
            if (anchorSet.contains(node)) continue;
            basisFixedMask[node] = 1u;
            if (physicallyFixedMask[node] == 0u)
                promotedFineNodes.insert(node);
        }
        // The normal builder checks the requested reduction budget again on
        // the next pass, after both the affine block and all region nodes have
        // moved out of the reduced space.
    }
    output.rankPromotedFineRegionLabels.assign(
        promotedRegions.begin(), promotedRegions.end());
    for (const std::uint32_t local : promotedFineNodes)
        output.rankPromotedFineGlobalFEMNodeIndices.push_back(
            globalNodeIndices[local]);
    output.globalFEMNodeIndices = std::move(globalNodeIndices);
    output.sourceNodeIndices = std::move(sourceNodeIndices);
    output.globalFEMTetrahedronIndices = std::move(globalTetrahedronIndices);
    output.sourceTetrahedronIndices = std::move(sourceTetrahedronIndices);
    output.tetrahedronRegionLabels = std::move(regionLabels);
    output.anchorReasonMasks = std::move(anchorReasonMasks);
    return output;
}

// Rebuild an exact compiled candidate basis with residual-selected rows moved
// to its fine complement. The caller must still pack/install and qualify this
// candidate; no runtime state or acceptance threshold is altered here.
[[nodiscard]] inline CompiledWorldPartitionedReductionBasis
buildResidualEnrichedPartitionedReductionBasis(
    const CompiledWorld& world,
    const std::uint32_t objectIndex,
    CompiledWorldPartitionedReductionOptions options,
    const detail::HumanPartitionedPreconditionerSource& source,
    const PartitionedReductionResidualWitness& witness,
    const std::span<const std::uint32_t> existingFineRows,
    const std::uint32_t requestedMinimumCoordinateReductionFactor,
    const std::uint32_t maximumPerRegion,
    const std::uint32_t maximumTotal,
    std::string& error
) {
    CompiledWorldPartitionedReductionBasis failed;
    failed.objectIndex = objectIndex;
    const auto fail = [&](const char* message) {
        error = message;
        failed.error = message;
        failed.basis.error = message;
        return failed;
    };
    if (objectIndex != source.objectIndex ||
        source.sourceWorldPhysicsFingerprint != world.physicsFingerprint)
        return fail("residual refinement source does not belong to this compiled world object");
    std::vector<std::uint32_t> fineRows;
    if (!selectPartitionedResidualEnrichmentFineNodes(
            source, witness, existingFineRows,
            requestedMinimumCoordinateReductionFactor,
            maximumPerRegion, maximumTotal, fineRows, error)) {
        failed.error = error;
        failed.basis.error = error;
        return failed;
    }
    options.minimumCoordinateReductionFactor =
        requestedMinimumCoordinateReductionFactor;
    options.promoteRankDeficientRegionsToFine = true;
    options.fineCoordinateOnlyGlobalFEMNodes =
        std::span<const std::uint32_t>(fineRows);
    auto result = buildCompiledWorldPartitionedReductionBasis(
        world, objectIndex, options);
    if (!result.valid()) {
        error = result.error;
        return result;
    }
    error.clear();
    return result;
}

} // namespace numi::matter
