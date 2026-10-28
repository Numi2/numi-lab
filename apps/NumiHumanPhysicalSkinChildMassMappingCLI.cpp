#include "NumiHumanPhysicalSkinChildTetGen.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace metalrobo::human::coarse_tissue;

std::string identityChildMeshSHA256(const PhysicalSkinChildMassMappingBundle& bundle) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("identity-transfer-control.child-mesh.v1");
    hash.string(bundle.parent.parentMassPacketSHA256);
    hash.string(bundle.parent.parentSourceContributionSHA256);
    hash.string(bundle.parent.parentTopologySHA256);
    hash.string(bundle.parent.parentLabelSHA256);
    hash.u64(bundle.currentWorldPoints.size());
    for (const auto& point : bundle.currentWorldPoints)
        for (const auto value : point) hash.real(value);
    hash.u64(bundle.referenceWorldPoints.size());
    for (const auto& point : bundle.referenceWorldPoints)
        for (const auto value : point) hash.real(value);
    hash.u64(bundle.childTetrahedra.size());
    for (std::size_t cell = 0; cell < bundle.childTetrahedra.size(); ++cell) {
        for (const auto node : bundle.childTetrahedra[cell]) hash.u32(node);
        hash.u32(bundle.childMaterialIndices[cell]);
        hash.u32(bundle.childDonorBodyIndices[cell]);
        hash.u32(bundle.childAnatomicalRegionIndices[cell]);
        hash.u32(bundle.childStableSourceFaceIdentities[cell]);
    }
    return hash.finish();
}

std::string identityCoordinateFrameSHA256(const PhysicalSkinChildMassMappingBundle& bundle) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("identity-transfer-control.coordinate-frame.v1");
    hash.string("child world/reference nodes are IEEE-F32 casts of native parent currentWorldPoints; no frame transform");
    hash.string(bundle.parent.parentMassPacketSHA256);
    hash.u64(bundle.parent.currentWorldPoints.size());
    for (std::size_t node = 0; node < bundle.parent.currentWorldPoints.size(); ++node) {
        hash.u32(bundle.parent.compiledGlobalNodeIndices[node]);
        hash.u32(bundle.parent.nodeFrameOwnerBodyIndices[node]);
        for (const auto value : bundle.parent.currentWorldPoints[node]) hash.real(value);
        for (const auto value : bundle.parent.referenceWorldPoints[node]) hash.real(value);
        for (const auto value : bundle.parent.nodeFrameOwnerLocalPoints[node]) hash.real(value);
        for (const auto value : bundle.currentWorldPoints[node]) hash.real(value);
    }
    return hash.finish();
}

bool prepareIdentityTransfer(PhysicalSkinParentMassPacket&& parent,
                             PhysicalSkinChildMassMappingBundle& bundle,
                             std::size_t& sourceBodyCount,
                             std::size_t& donorRegionMismatchCount,
                             std::string& error) {
    if (parent.referenceWorldPoints.empty() ||
        parent.currentWorldPoints.size() != parent.referenceWorldPoints.size() ||
        parent.tetrahedra.empty() ||
        parent.materialIndices.size() != parent.tetrahedra.size() ||
        parent.donorBodyIndices.size() != parent.tetrahedra.size() ||
        parent.anatomicalRegionIndices.size() != parent.tetrahedra.size() ||
        parent.stableSourceFaceIdentities.size() != parent.tetrahedra.size() ||
        parent.nodeFrameOwnerBodyIndices.size() != parent.referenceWorldPoints.size() ||
        parent.nodeFrameOwnerLocalPoints.size() != parent.referenceWorldPoints.size() ||
        parent.compiledGlobalNodeIndices.size() != parent.referenceWorldPoints.size()) {
        error = "identity control requires complete packet, labels, node frame owners, and stable source faces";
        return false;
    }
    if (parent.anatomicalRegionIndices.empty() ||
        parent.stableSourceFaceIdentities.empty() ||
        parent.parentLabelSHA256.size() != 64u) {
        error = "identity control requires the separately captured native region/source-face label CSV";
        return false;
    }

    const auto maximumOwner = *std::max_element(parent.nodeFrameOwnerBodyIndices.begin(),
                                                parent.nodeFrameOwnerBodyIndices.end());
    if (maximumOwner == std::numeric_limits<std::uint32_t>::max()) {
        error = "identity control cannot derive a bounded source body count from native frame owners";
        return false;
    }
    sourceBodyCount = static_cast<std::size_t>(maximumOwner) + 1u;
    if (sourceBodyCount == 0u) {
        error = "identity control derived an empty source body domain";
        return false;
    }

    bundle.parent = std::move(parent);
    const auto& native = bundle.parent;
    bundle.currentWorldPoints.reserve(native.currentWorldPoints.size());
    for (const auto& point : native.currentWorldPoints) {
        V3 f32Point{};
        for (unsigned axis = 0; axis < 3u; ++axis) {
            const auto rounded = static_cast<float>(point[axis]);
            if (!std::isfinite(rounded)) {
                error = "native parent current point cannot be represented as finite Float32";
                return false;
            }
            f32Point[axis] = static_cast<double>(rounded);
        }
        bundle.currentWorldPoints.push_back(f32Point);
    }
    // The identity control uses the actual F32 current surface as both child
    // reference and child current geometry, with no invented deformation.
    bundle.referenceWorldPoints = bundle.currentWorldPoints;
    bundle.childTetrahedra = native.tetrahedra;
    bundle.childMaterialIndices = native.materialIndices;
    bundle.childDonorBodyIndices = native.donorBodyIndices;
    bundle.childAnatomicalRegionIndices = native.anatomicalRegionIndices;
    bundle.childStableSourceFaceIdentities = native.stableSourceFaceIdentities;
    bundle.regionLabelSHA256 = native.parentLabelSHA256;
    bundle.maximumSpatialMassTransportDistanceM = 1.0e-6;

    std::set<std::uint32_t> materials, regions, stableFaces;
    donorRegionMismatchCount = 0u;
    for (std::size_t cell = 0; cell < native.tetrahedra.size(); ++cell) {
        const auto material = native.materialIndices[cell];
        if (material != 15u && material != 16u) {
            error = "identity control only has authoritative skin material 15 and fat material 16 bounds";
            return false;
        }
        materials.insert(material);
        regions.insert(native.anatomicalRegionIndices[cell]);
        stableFaces.insert(native.stableSourceFaceIdentities[cell]);
        donorRegionMismatchCount += native.donorBodyIndices[cell] !=
                                    native.anatomicalRegionIndices[cell];
        if (native.donorBodyIndices[cell] >= sourceBodyCount) {
            error = "geometric donor index exceeds the native frame-owner-derived body domain";
            return false;
        }
    }
    for (const auto& row : native.sourceMassContributions) {
        if (row.donorBodyIndex >= sourceBodyCount) {
            error = "source contribution donor index exceeds the native frame-owner-derived body domain";
            return false;
        }
    }
    if (materials != std::set<std::uint32_t>{15u, 16u}) {
        error = "identity control requires both native skin material 15 and fat material 16";
        return false;
    }

    bundle.childMeshSHA256 = identityChildMeshSHA256(bundle);
    bundle.coordinateFrameSHA256 = identityCoordinateFrameSHA256(bundle);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    using namespace metalrobo::human::coarse_tissue;
    std::string massCSV, labelsCSV, childNodes, childElements, childLabels, coordinateFrame;
    std::size_t explicitBodyCount=0u;
    double transportDistance=0.0;
    bool identityTransfer = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--parent-mass-csv" && i + 1 < argc) massCSV = argv[++i];
        else if (argument == "--labels-csv" && i + 1 < argc) labelsCSV = argv[++i];
        else if (argument == "--identity-transfer-control") identityTransfer = true;
        else if (argument == "--child-node-file" && i + 1 < argc) childNodes = argv[++i];
        else if (argument == "--child-ele-file" && i + 1 < argc) childElements = argv[++i];
        else if (argument == "--child-labels-csv" && i + 1 < argc) childLabels = argv[++i];
        else if (argument == "--coordinate-frame-sha256" && i + 1 < argc) coordinateFrame = argv[++i];
        else if (argument == "--source-body-count" && i + 1 < argc) {
            if(!physical_skin_child_mass_detail::parseNumber(std::string(argv[++i]),explicitBodyCount))return 2;
        } else if (argument == "--maximum-transport-distance-m" && i + 1 < argc) {
            if(!physical_skin_child_mass_detail::parseNumber(std::string(argv[++i]),transportDistance))return 2;
        }
        else if (argument == "--help") {
            std::cout << "usage: numi-human-physical-skin-child-mass-mapping-cli "
                         "--parent-mass-csv <compiled-parent-cell-mass.csv> "
                         "--labels-csv <compiled-parent-cell-mass.csv.labels.csv> "
                         "[--identity-transfer-control | --child-node-file <mesh.node> "
                         "--child-ele-file <mesh.ele> --child-labels-csv <labels.csv> "
                         "--coordinate-frame-sha256 <sha256> --source-body-count <count> "
                         "--maximum-transport-distance-m <metres>]\n"
                         "The identity control exercises conservative spatial mass "
                         "transport on exact native parent tetrahedra with child world/"
                         "reference nodes formed from parent current positions rounded to "
                         "Float32. It makes no geometric-overlap or physical-admission claim.\n";
            return 0;
        } else {
            std::cerr << "invalid argument: " << argument << "; run --help\n";
            return 2;
        }
    }
    const bool hasChildArguments=!childNodes.empty()||!childElements.empty()||!childLabels.empty()||
        !coordinateFrame.empty()||explicitBodyCount!=0u||transportDistance!=0.0;
    if (massCSV.empty() || ((identityTransfer||hasChildArguments) && labelsCSV.empty()) ||
        (identityTransfer&&hasChildArguments) ||
        (hasChildArguments&&(childNodes.empty()||childElements.empty()||childLabels.empty()||
            coordinateFrame.empty()||explicitBodyCount==0u||transportDistance<=0.0))) {
        std::cerr << "invalid arguments; run --help\n";
        return 2;
    }

    PhysicalSkinParentMassPacket parent;
    std::string error;
    if (!loadCompiledSkinFatParentMassPacketCSV(massCSV, parent, error)) {
        std::cerr << "parent_packet=REJECTED reason=" << error << '\n';
        return 3;
    }
    if (!labelsCSV.empty() &&
        !loadCompiledSkinFatParentLabelsCSV(labelsCSV, parent, error)) {
        std::cerr << "parent_labels=REJECTED reason=" << error << '\n';
        return 4;
    }
    if (!labelsCSV.empty()) {
        std::cout << "parent_labels=PASS rows=" << parent.anatomicalRegionIndices.size()
                  << " source=exact_native_registered_skin_fat_metadata sha256="
                  << parent.parentLabelSHA256 << '\n';
    }
    std::cout << std::setprecision(17)
              << "parent_packet=PASS nodes=" << parent.referenceWorldPoints.size()
              << " tetrahedra=" << parent.tetrahedra.size()
              << " source_rows=" << parent.sourceMassContributions.size()
              << " mass_kg=" << parent.totalMassKg
              << " nodal_closure_abs_kg=" << parent.maximumNodalMassClosureErrorKg
              << " nodal_closure_rel=" << parent.maximumNodalMassClosureRelativeError
              << " mass_csv_sha256=" << parent.parentMassPacketSHA256
              << " source_csv_sha256=" << parent.parentSourceContributionSHA256
              << " packet_topology_sha256=" << parent.parentTopologySHA256 << '\n';

    if (!identityTransfer&&!hasChildArguments) {
        std::cout << "child_transfer=NOT_RUN reason=identity-transfer-control-not-requested\n";
        return 0;
    }

    PhysicalSkinChildMassMappingBundle bundle;
    std::size_t sourceBodyCount = 0u, donorRegionMismatchCount = 0u;
    const std::string transferLabel=identityTransfer?"identity_transfer":"child_reference_transfer";
    if(identityTransfer) {
        if (!prepareIdentityTransfer(std::move(parent), bundle, sourceBodyCount,
                                     donorRegionMismatchCount, error)) {
            std::cerr << transferLabel << "=REJECTED reason=" << error << '\n';return 5;
        }
        transportDistance=1.0e-6;
    } else {
        sourceBodyCount=explicitBodyCount;bundle.parent=std::move(parent);
        PhysicalSkinChildTetGenImportReport importReport;
        if(!loadPhysicalSkinChildTetGenReference(childNodes,childElements,childLabels,
                coordinateFrame,sourceBodyCount,transportDistance,bundle,importReport,error)) {
            std::cerr<<transferLabel<<"=REJECTED reason="<<error<<'\n';return 5;
        }
        for(std::size_t cell=0u;cell<bundle.childTetrahedra.size();++cell)
            donorRegionMismatchCount+=bundle.childDonorBodyIndices[cell]!=bundle.childAnatomicalRegionIndices[cell];
        std::cout<<std::setprecision(17)<<"child_tetgen_reference_import=PASS nodes="<<importReport.nodeCount
            <<" cells="<<importReport.cellCount<<" minimum_f32_volume_m3="<<importReport.minimumFloat32ReferenceVolumeM3
            <<" maximum_f32_rounding_m="<<importReport.maximumFloat32RoundingDistanceM
            <<" node_sha256="<<importReport.nodeFileSHA256<<" element_sha256="<<importReport.elementFileSHA256
            <<" labels_sha256="<<importReport.labelFileSHA256
            <<" current_equals_reference=1 physical_admission=NOT_CLAIMED\n";
    }

    std::vector<double> densityLower(17u, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> densityUpper(17u, std::numeric_limits<double>::quiet_NaN());
    densityLower[15] = 900.0;
    densityUpper[15] = 1100.0;
    densityLower[16] = 850.0;
    densityUpper[16] = 950.0;
    const auto mappingStart = std::chrono::steady_clock::now();
    if (!preparePhysicalSkinChildMassMapping(bundle, densityLower, densityUpper,
            transportDistance, sourceBodyCount, error)) {
        std::cerr << transferLabel << "=REJECTED reason=" << error << '\n';
        return 6;
    }

    const auto mappingElapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - mappingStart).count();
    const auto& result = bundle.spatialMassTransportResult;
    const double childMass = std::accumulate(result.childMassKg.begin(),
                                             result.childMassKg.end(), 0.0);
    std::map<std::uint32_t, std::pair<double, double>> densityRange;
    for (std::size_t cell = 0; cell < result.childImpliedDensityKgM3.size(); ++cell) {
        const auto material = bundle.childMaterialIndices[cell];
        auto [it, inserted] = densityRange.emplace(material,
            std::pair{result.childImpliedDensityKgM3[cell],
                      result.childImpliedDensityKgM3[cell]});
        if (!inserted) {
            it->second.first = std::min(it->second.first,
                                        result.childImpliedDensityKgM3[cell]);
            it->second.second = std::max(it->second.second,
                                         result.childImpliedDensityKgM3[cell]);
        }
    }
    std::size_t sameIndexedParent = 0u;
    for (std::size_t child = 0; child < result.childDominantParentRows.size(); ++child)
        sameIndexedParent += result.childDominantParentRows[child] == child;
    std::size_t crossStableSourceFaceEdges = 0u;
    std::size_t crossMaterialEdges = 0u, crossGeometricDonorEdges = 0u;
    std::size_t crossAnatomicalRegionEdges = 0u;
    if (result.childParentOffsets.size() != bundle.childTetrahedra.size() + 1u ||
        result.childParentRows.size() != result.childParentTransportMassKg.size()) {
        std::cerr << transferLabel << "=REJECTED reason=recorded_flow_CSR_is_incomplete\n";
        return 7;
    }
    for (std::size_t child = 0; child < bundle.childTetrahedra.size(); ++child) {
        for (std::size_t edge = result.childParentOffsets[child];
             edge < result.childParentOffsets[child + 1u]; ++edge) {
            const auto parentRow = result.childParentRows[edge];
            if (parentRow >= bundle.parent.tetrahedra.size()) {
                std::cerr << transferLabel << "=REJECTED reason=recorded_flow_parent_row_is_out_of_range\n";
                return 7;
            }
            crossMaterialEdges += bundle.parent.materialIndices[parentRow] !=
                                  bundle.childMaterialIndices[child];
            crossGeometricDonorEdges += bundle.parent.donorBodyIndices[parentRow] !=
                                        bundle.childDonorBodyIndices[child];
            crossAnatomicalRegionEdges += bundle.parent.anatomicalRegionIndices[parentRow] !=
                                          bundle.childAnatomicalRegionIndices[child];
            crossStableSourceFaceEdges += bundle.parent.stableSourceFaceIdentities[parentRow] !=
                                          bundle.childStableSourceFaceIdentities[child];
        }
    }
    if (crossMaterialEdges != 0u || crossGeometricDonorEdges != 0u ||
        crossAnatomicalRegionEdges != 0u) {
        std::cerr << std::setprecision(17)
                  << transferLabel << "=REJECTED reason=recorded_flow_crosses_material_donor_or_region_labels"
                  << " cross_material_edges=" << crossMaterialEdges
                  << " cross_geometric_donor_edges=" << crossGeometricDonorEdges
                  << " cross_anatomical_region_edges=" << crossAnatomicalRegionEdges
                  << " transport_edges=" << result.transportEdgeCount
                  << " mass_kg_parent=" << bundle.parent.totalMassKg
                  << " mass_kg_child=" << childMass
                  << " max_source_donor_closure_abs_kg="
                  << result.maximumSourceDonorMassClosureErrorKg
                  << " max_transport_distance_used_m=" << result.maximumTransportDistanceUsedM
                  << " transport_receipt_sha256=" << result.receiptSHA256 << '\n';
        return 7;
    }

    std::cout << std::setprecision(17)
              << transferLabel << "=PASS method=conservativeSpatialMassTransport"
              << " scope=" << (identityTransfer?"parent_geometry_identity_control":"explicit_child_reference_preparation")
              << " physical_admission=NOT_CLAIMED"
              << " source_body_count=" << sourceBodyCount
              << " child_cells_mapped=" << result.childDominantParentRows.size()
              << " child_tets=" << bundle.childTetrahedra.size()
              << " donor_region_mismatch_cells=" << donorRegionMismatchCount
              << " unique_regions=" << std::set<std::uint32_t>(
                     bundle.childAnatomicalRegionIndices.begin(),
                     bundle.childAnatomicalRegionIndices.end()).size()
              << " unique_stable_source_faces=" << std::set<std::uint32_t>(
                     bundle.childStableSourceFaceIdentities.begin(),
                     bundle.childStableSourceFaceIdentities.end()).size()
              << " exact_index_dominant_parent_cells=" << sameIndexedParent
              << " flow_edges_crossing_stable_source_face_identity_diagnostic="
              << crossStableSourceFaceEdges
              << " transport_edges=" << result.transportEdgeCount
              << " augmentations=" << result.augmentationCount
              << " source_donor_closure_groups=" << result.sourceDonorClosure.size()
              << " mapping_elapsed_s=" << mappingElapsedSeconds
              << " mass_kg_parent=" << bundle.parent.totalMassKg
              << " mass_kg_child=" << childMass
              << " total_mass_closure_abs_kg=" << std::abs(childMass-bundle.parent.totalMassKg)
              << " max_parent_mass_closure_abs_kg=" << result.maximumParentMassClosureErrorKg
              << " max_source_donor_closure_abs_kg=" << result.maximumSourceDonorMassClosureErrorKg
              << " max_child_fraction_sum_error=" << result.maximumChildFractionSumError
              << " max_transport_distance_used_m=" << result.maximumTransportDistanceUsedM
              << " max_transport_distance_allowed_m=" << result.maximumTransportDistanceM
              << " min_child_density_kgm3=" << result.minimumChildImpliedDensityKgM3
              << " max_child_density_kgm3=" << result.maximumChildImpliedDensityKgM3
              << " density_skin_15_bounds_kgm3=900:1100"
              << " density_fat_16_bounds_kgm3=850:950"
              << " child_mesh_sha256=" << bundle.childMeshSHA256
              << " region_label_sha256=" << bundle.regionLabelSHA256
              << " coordinate_frame_sha256=" << bundle.coordinateFrameSHA256
              << " transport_receipt_sha256=" << result.receiptSHA256
              << '\n';
    for (const auto& [material, range] : densityRange) {
        std::cout << "transfer_density_range material=" << material
                  << " min_kgm3=" << range.first
                  << " max_kgm3=" << range.second << '\n';
    }
    return 0;
}

