#include "NumiHumanPhysicalSkinChildTetGen.hpp"
#include "NumiHumanCurrentFractionalSupportNPZ.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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


struct FractionalOutputPins {
    std::filesystem::path flow,semanticGroups,sourceDonors,carrierTrace;
    std::array<std::string,4> sha256;
};

bool writeCurrentFractionalArtifacts(
        const std::filesystem::path& directory,
        const PhysicalSkinChildMassMappingBundle& bundle,
        const ConservativeSpatialMassTransportResult& result,
        const SourceCellCarrierTransportTrace& trace,
        const ConservativeSpatialChildSemanticSupportData& support,
        FractionalOutputPins& pins,
        std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(directory.empty())return fail("fractional artifact output directory is required");
    std::error_code ec;
    if(std::filesystem::exists(directory,ec)){
        if(ec||!std::filesystem::is_directory(directory,ec)||ec)
            return fail("fractional artifact output path exists and is not a directory");
        if(std::filesystem::directory_iterator(directory,ec)!=std::filesystem::directory_iterator{}||ec)
            return fail("fractional artifact output directory must be new or empty");
    } else if(!std::filesystem::create_directories(directory,ec)||ec)
        return fail("cannot create fractional artifact output directory");
    const auto cleanTemporary=[&](){
        std::error_code ignored;
        for(const auto* name:{"flow.csv.tmp","semantic-groups.csv.tmp","source-donors.csv.tmp","carrier-trace.csv.tmp"})
            std::filesystem::remove(directory/name,ignored);
    };
    const std::array<std::string,4> names{"flow.csv","semantic-groups.csv","source-donors.csv","carrier-trace.csv"};
    const std::array<std::filesystem::path,4> temps{
        directory/"flow.csv.tmp",directory/"semantic-groups.csv.tmp",
        directory/"source-donors.csv.tmp",directory/"carrier-trace.csv.tmp"};
    const std::array<std::filesystem::path,4> finals{
        directory/names[0],directory/names[1],directory/names[2],directory/names[3]};
    if(result.childParentOffsets.size()!=bundle.childTetrahedra.size()+1u||
       result.childParentRows.size()!=result.childParentTransportMassKg.size()||
       result.childParentRows.size()!=result.childParentTransportMeasureM3.size()||
       result.childParentRows.size()!=result.childParentTransportDistanceM.size()||
       result.childParentRows.size()!=result.childParentVolumeFractions.size()||
       result.childSemanticGroupOffsets.size()!=bundle.childTetrahedra.size()+1u||
       result.childSemanticGroupOffsets.empty()||
       result.childSemanticGroupOffsets.back()!=result.childSemanticGroupIndices.size()||
       result.childSemanticGroupIndices.size()!=result.childSemanticVolumeFractions.size()||
       support.groupMaterialIndices.size()!=support.donorBodyIndices.size()||
       support.groupMaterialIndices.size()!=support.anatomicalRegionIndices.size()||
       trace.childParentCarrierOffsets.size()!=result.childParentRows.size()+1u||
       trace.childParentCarrierRows.size()!=trace.childParentCarrierMassKg.size()||
       trace.childParentCarrierRows.size()!=trace.childParentCarrierMeasureM3.size()||
       trace.childParentCarrierRows.size()!=trace.childParentCarrierDistanceM.size()||
       trace.childParentCarrierOffsets.empty()||trace.childParentCarrierOffsets.front()!=0u||
       trace.childParentCarrierOffsets.back()!=trace.childParentCarrierRows.size()||
       trace.expandedParentToSourceParentRows.size()!=trace.expandedParentToCarrierRows.size())
        return fail("fractional result/trace arrays are incomplete for artifact serialization");
    try {
        {
            std::ofstream f(temps[0],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot open fractional flow artifact");
            f<<"child_row,source_parent_row,mass_kg,reference_measure_m3,child_parent_measure_fraction,distance_m\n";
            f<<std::setprecision(17);
            for(std::size_t c=0;c<bundle.childTetrahedra.size();++c){
                for(std::uint32_t e=result.childParentOffsets[c];e<result.childParentOffsets[c+1u];++e){
                    if(e>=result.childParentRows.size()||result.childParentRows[e]>=bundle.parent.tetrahedra.size())
                        return fail("fractional flow artifact contains an out-of-range source parent row");
                    f<<c<<','<<result.childParentRows[e]<<','<<result.childParentTransportMassKg[e]<<','
                     <<result.childParentTransportMeasureM3[e]<<','<<result.childParentVolumeFractions[e]<<','
                     <<result.childParentTransportDistanceM[e]<<'\n';
                }
            }
            f.flush();if(!f)return fail("failed writing fractional flow artifact");
        }
        {
            std::ofstream f(temps[1],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot open fractional semantic-group artifact");
            f<<"child_row,group_index,material_index,geometric_donor_body,anatomical_region_id,child_mass_fraction,mass_kg\n";
            f<<std::setprecision(17);
            for(std::size_t c=0;c<bundle.childTetrahedra.size();++c){
                for(std::uint32_t r=result.childSemanticGroupOffsets[c];r<result.childSemanticGroupOffsets[c+1u];++r){
                    const auto g=result.childSemanticGroupIndices[r];
                    const double fraction=result.childSemanticVolumeFractions[r];
                    if(g>=support.groupMaterialIndices.size()||!std::isfinite(fraction)||fraction<=0.0)
                        return fail("fractional semantic-group artifact contains an invalid group/fraction");
                    f<<c<<','<<g<<','<<support.groupMaterialIndices[g]<<','<<support.donorBodyIndices[g]<<','
                     <<support.anatomicalRegionIndices[g]<<','<<fraction<<','<<result.childMassKg[c]*fraction<<'\n';
                }
            }
            f.flush();if(!f)return fail("failed writing fractional semantic-group artifact");
        }
        {
            std::ofstream f(temps[2],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot open fractional source-donor artifact");
            f<<"material_index,source_donor_body_index,parent_mass_kg,child_mass_kg,closure_abs_kg\n";
            f<<std::setprecision(17);
            for(const auto& row:result.sourceDonorClosure)
                f<<row.materialIndex<<','<<row.sourceDonorBodyIndex<<','<<row.parentMassKg<<','
                 <<row.childMassKg<<','<<std::abs(row.parentMassKg-row.childMassKg)<<'\n';
            f.flush();if(!f)return fail("failed writing fractional source-donor artifact");
        }
        {
            std::ofstream f(temps[3],std::ios::binary|std::ios::trunc);
            if(!f)return fail("cannot open fractional carrier-trace artifact");
            f<<"flow_edge_row,child_row,expanded_parent_row,source_parent_row,carrier_row,mass_kg,reference_measure_m3,distance_m\n";
            f<<std::setprecision(17);
            constexpr auto invalid=std::numeric_limits<std::uint32_t>::max();
            for(std::size_t c=0;c<bundle.childTetrahedra.size();++c){
                for(std::uint32_t e=result.childParentOffsets[c];e<result.childParentOffsets[c+1u];++e){
                    if(e+1u>=trace.childParentCarrierOffsets.size())
                        return fail("fractional carrier trace does not cover every output flow edge");
                    const auto begin=trace.childParentCarrierOffsets[e],end=trace.childParentCarrierOffsets[e+1u];
                    if(begin>=end||end>trace.childParentCarrierRows.size())
                        return fail("fractional carrier trace contains an empty or malformed flow-edge row");
                    for(std::uint32_t r=begin;r<end;++r){
                        const auto v=trace.childParentCarrierRows[r];
                        if(v>=trace.expandedParentToSourceParentRows.size())
                            return fail("fractional carrier trace has an out-of-range expanded parent row");
                        const auto source=trace.expandedParentToSourceParentRows[v];
                        const auto carrier=trace.expandedParentToCarrierRows[v];
                        f<<e<<','<<c<<','<<v<<','<<source<<',';
                        if(carrier==invalid)f<<-1;else f<<carrier;
                        f<<','<<trace.childParentCarrierMassKg[r]<<','
                         <<trace.childParentCarrierMeasureM3[r]<<','
                         <<trace.childParentCarrierDistanceM[r]<<'\n';
                    }
                }
            }
            f.flush();if(!f)return fail("failed writing fractional carrier-trace artifact");
        }
        for(std::size_t i=0;i<temps.size();++i){
            std::filesystem::rename(temps[i],finals[i],ec);
            if(ec){cleanTemporary();return fail("cannot atomically publish fractional result CSV: "+ec.message());}
        }
    } catch(const std::exception& e) {
        cleanTemporary();return fail(std::string("fractional artifact serialization exception: ")+e.what());
    }
    std::array<std::string,4> hashes{};
    for(std::size_t i=0;i<finals.size();++i){
        if(!physical_skin_child_mass_detail::sha256File(finals[i],hashes[i],error))
            return fail("cannot hash serialized fractional result CSV: "+error);
    }
    pins={finals[0],finals[1],finals[2],finals[3],hashes};
    error.clear();return true;
}

} // namespace

int main(int argc, char** argv) {
    using namespace metalrobo::human::coarse_tissue;
    std::string massCSV, labelsCSV, childNodes, childElements, childLabels, coordinateFrame;
    std::size_t explicitBodyCount=0u;
    double transportDistance=0.0;
    bool identityTransfer = false;
    bool currentFractionalOptIn=false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--parent-mass-csv" && i + 1 < argc) massCSV = argv[++i];
        else if (argument == "--labels-csv" && i + 1 < argc) labelsCSV = argv[++i];
        else if (argument == "--identity-transfer-control") identityTransfer = true;
        else if (argument == "--current-fractional-semantic") currentFractionalOptIn=true;
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

    const auto envValue=[](const char* name)->std::string {
        const char* value=std::getenv(name);return value?std::string(value):std::string{};
    };
    std::map<std::string,std::string> fractionalEnv;
    const std::array<const char*,24> fractionalEnvNames{{
        "NUMI_H706_FRACTIONAL_ENABLE","NUMI_H706_FRACTIONAL_STATE_SHA256",
        "NUMI_H706_FRACTIONAL_GEOMETRY_SKIN_SHA256","NUMI_H706_FRACTIONAL_VOLUME_SKIN_SHA256",
        "NUMI_H706_FRACTIONAL_GEOMETRY_FAT_SHA256","NUMI_H706_FRACTIONAL_VOLUME_FAT_SHA256",
        "NUMI_H706_FRACTIONAL_SOURCE330_MASS_SHA256","NUMI_H706_FRACTIONAL_SOURCE330_NODES",
        "NUMI_H706_FRACTIONAL_SOURCE330_NODES_SHA256","NUMI_H706_FRACTIONAL_SOURCE330_TETS",
        "NUMI_H706_FRACTIONAL_SOURCE330_TETS_SHA256","NUMI_H706_FRACTIONAL_P1_SKIN_NPZ",
        "NUMI_H706_FRACTIONAL_P1_SKIN_SHA256","NUMI_H706_FRACTIONAL_P1_FAT_NPZ",
        "NUMI_H706_FRACTIONAL_P1_FAT_SHA256","NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ",
        "NUMI_H706_FRACTIONAL_CARRIERS_SKIN_SHA256","NUMI_H706_FRACTIONAL_CARRIERS_FAT_NPZ",
        "NUMI_H706_FRACTIONAL_CARRIERS_FAT_SHA256","NUMI_H706_FRACTIONAL_SOURCE_CENTROID_CSV",
        "NUMI_H706_FRACTIONAL_SOURCE_CENTROID_SHA256","NUMI_H706_FRACTIONAL_EXACT_PROJECTION_CSV",
        "NUMI_H706_FRACTIONAL_EXACT_PROJECTION_SHA256","NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV"}};
    for(const auto* name:fractionalEnvNames)fractionalEnv.emplace(name,envValue(name));
    const std::array<const char*,3> finalEnvNames{{
        "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256",
        "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV",
        "NUMI_H706_FRACTIONAL_OUTPUT_DIR"}};
    for(const auto* name:finalEnvNames)fractionalEnv.emplace(name,envValue(name));
    fractionalEnv.emplace("NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256",
                          envValue("NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256"));
    if(currentFractionalOptIn){
        if(envValue("NUMI_H706_FRACTIONAL_ENABLE")!="1"){
            std::cerr<<"fractional_opt_in=REJECTED reason=explicit_enable_environment_must_equal_1\n";return 2;
        }
        for(const auto& [name,value]:fractionalEnv)
            if(name!="NUMI_H706_FRACTIONAL_ENABLE"&&value.empty()){
                std::cerr<<"fractional_opt_in=REJECTED reason=missing_required_environment_"<<name<<"\n";return 2;
            }
    }else if(envValue("NUMI_H706_FRACTIONAL_ENABLE")=="1"){
        std::cerr<<"fractional_opt_in=REJECTED reason=environment_enable_requires_explicit_cli_flag\n";return 2;
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
    ConservativeSpatialChildSemanticSupportData fractionalSupport;
    CurrentSourceCarrierQuadratureData carrierData;
    SourceCellCarrierTransportTrace carrierTrace;
    CurrentFractionalSemanticProofFiles proofFiles;
    SourceCellCarrierQuadratureInput carrierView;
    bool fractionalPrepared=false;
    if(currentFractionalOptIn){
        const auto value=[&](const char* name)->const std::string&{return fractionalEnv.at(name);};
        const auto asPath=[&](const char* name){return std::filesystem::path(value(name));};
        if(value("NUMI_H706_FRACTIONAL_SOURCE330_MASS_SHA256")!=bundle.parent.parentMassPacketSHA256){
            std::cerr<<"fractional_opt_in=REJECTED reason=source330_mass_input_does_not_match_loaded_parent_packet\n";return 5;
        }
        std::vector<std::uint32_t> skinChildRows,fatChildRows;
        skinChildRows.reserve(bundle.childTetrahedra.size());fatChildRows.reserve(bundle.childTetrahedra.size());
        for(std::size_t child=0;child<bundle.childMaterialIndices.size();++child){
            if(bundle.childMaterialIndices[child]==15u)skinChildRows.push_back(static_cast<std::uint32_t>(child));
            else if(bundle.childMaterialIndices[child]==16u)fatChildRows.push_back(static_cast<std::uint32_t>(child));
            else {std::cerr<<"fractional_opt_in=REJECTED reason=current_child_material_outside_skin_fat\n";return 5;}
        }
        if(skinChildRows.empty()||fatChildRows.empty()){
            std::cerr<<"fractional_opt_in=REJECTED reason=both_current_material_domains_are_required\n";return 5;
        }
        CurrentFractionalSupportPins skinPins{
            value("NUMI_H706_FRACTIONAL_P1_SKIN_SHA256"),
            value("NUMI_H706_FRACTIONAL_STATE_SHA256"),
            value("NUMI_H706_FRACTIONAL_GEOMETRY_SKIN_SHA256"),
            value("NUMI_H706_FRACTIONAL_VOLUME_SKIN_SHA256"),
            bundle.parent.parentMassPacketSHA256};
        CurrentFractionalSupportPins fatPins{
            value("NUMI_H706_FRACTIONAL_P1_FAT_SHA256"),
            value("NUMI_H706_FRACTIONAL_STATE_SHA256"),
            value("NUMI_H706_FRACTIONAL_GEOMETRY_FAT_SHA256"),
            value("NUMI_H706_FRACTIONAL_VOLUME_FAT_SHA256"),
            bundle.parent.parentMassPacketSHA256};
        ConservativeSpatialChildSemanticSupportData skinSupport,fatSupport;
        if(!loadCurrentP1SemanticSupportNPZ(asPath("NUMI_H706_FRACTIONAL_P1_SKIN_NPZ"),
                skinPins,bundle.parent,skinChildRows.size(),skinSupport,error)||
           !loadCurrentP1SemanticSupportNPZ(asPath("NUMI_H706_FRACTIONAL_P1_FAT_NPZ"),
                fatPins,bundle.parent,fatChildRows.size(),fatSupport,error)||
           !mergeCurrentPairedP1SemanticSupport(skinSupport,fatSupport,skinChildRows,fatChildRows,
                bundle.childMaterialIndices,bundle.parent,fractionalSupport,error)){
            std::cerr<<"fractional_p1_support=REJECTED reason="<<error<<"\n";return 5;
        }
        CurrentSourceCarrierPins carrierPins;
        carrierPins.skinPacketSHA256=value("NUMI_H706_FRACTIONAL_CARRIERS_SKIN_SHA256");
        carrierPins.fatPacketSHA256=value("NUMI_H706_FRACTIONAL_CARRIERS_FAT_SHA256");
        carrierPins.sourceNodeTableSHA256=value("NUMI_H706_FRACTIONAL_SOURCE330_NODES_SHA256");
        carrierPins.sourceTetTableSHA256=value("NUMI_H706_FRACTIONAL_SOURCE330_TETS_SHA256");
        if(!loadCurrentPairedSourceCarrierQuadratureNPZ(
                asPath("NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ"),
                asPath("NUMI_H706_FRACTIONAL_CARRIERS_FAT_NPZ"),
                asPath("NUMI_H706_FRACTIONAL_SOURCE330_TETS"),
                asPath("NUMI_H706_FRACTIONAL_SOURCE330_NODES"),
                carrierPins,bundle.parent,skinChildRows,fatChildRows,carrierData,error)){
            std::cerr<<"fractional_source_carriers=REJECTED reason="<<error<<"\n";return 5;
        }
        carrierView=carrierData.view();
        proofFiles.exactProjectionCSVPath=asPath("NUMI_H706_FRACTIONAL_EXACT_PROJECTION_CSV");
        proofFiles.exactProjectionCSV_SHA256=value("NUMI_H706_FRACTIONAL_EXACT_PROJECTION_SHA256");
        proofFiles.sourceCentroidSlotsCSVPath=asPath("NUMI_H706_FRACTIONAL_SOURCE_CENTROID_CSV");
        proofFiles.sourceCentroidSlotsCSV_SHA256=value("NUMI_H706_FRACTIONAL_SOURCE_CENTROID_SHA256");
        proofFiles.skinCarrierWitnessCSVPath=asPath("NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV");
        proofFiles.skinCarrierWitnessCSV_SHA256=value("NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256");
        proofFiles.fatCarrierWitnessCSVPath=asPath("NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV");
        proofFiles.fatCarrierWitnessCSV_SHA256=value("NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256");
        proofFiles.carriers=&carrierData;
        proofFiles.skinChildGlobalRows=skinChildRows;
        proofFiles.fatChildGlobalRows=fatChildRows;
        fractionalPrepared=true;
        std::cout<<"fractional_p1_support=PASS skin_children="<<skinChildRows.size()
                 <<" fat_children="<<fatChildRows.size()
                 <<" support_sha256="<<fractionalSupport.typedSupportSHA256<<"\n";
        std::cout<<"fractional_source_carriers=PASS carrier_rows="<<carrierData.carrierWeights.size()
                 <<" packet_sha256="<<carrierData.sourceCarrierPacketSHA256
                 <<" max_distance_m="<<transportDistance<<"\n";
    }

    const auto mappingStart = std::chrono::steady_clock::now();
    const bool mappingPrepared=fractionalPrepared?
        preparePhysicalSkinChildMassMapping(bundle,densityLower,densityUpper,transportDistance,
            sourceBodyCount,error,nullptr,&carrierView,&carrierTrace,&fractionalSupport,
            &bindCurrentFractionalSemanticSupportProofs,&proofFiles):
        preparePhysicalSkinChildMassMapping(bundle,densityLower,densityUpper,transportDistance,
            sourceBodyCount,error);
    if (!mappingPrepared) {
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
    if (crossMaterialEdges != 0u ||
        (!fractionalPrepared&&(crossGeometricDonorEdges != 0u||crossAnatomicalRegionEdges != 0u))) {
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

    FractionalOutputPins artifactPins;
    if(fractionalPrepared){
        const auto outputDirectory=std::filesystem::path(envValue("NUMI_H706_FRACTIONAL_OUTPUT_DIR"));
        if(!writeCurrentFractionalArtifacts(outputDirectory,bundle,result,carrierTrace,fractionalSupport,artifactPins,error)){
            std::cerr<<transferLabel<<"=REJECTED reason=verified_flow_artifact_serialization_failed:"<<error<<'\n';return 8;
        }
    }
    std::cout << std::setprecision(17)
              << transferLabel << "=PASS method=conservativeSpatialMassTransport"
              << " mapping_path=" << (bundle.exactTopologyIdentityTransportUsed?
                    "exact_topology_identity":"bounded_conservative_spatial")
              << " mapping_path_sha256=" << bundle.massMappingPathSHA256
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
              << " fractional_semantic_opt_in=" << (fractionalPrepared?1:0)
              << " primary_donor_label_crossings_diagnostic=" << crossGeometricDonorEdges
              << " primary_region_label_crossings_diagnostic=" << crossAnatomicalRegionEdges
              << " fractional_semantic_group_allocations="
              << bundle.spatialMassTransportResult.childSemanticGroupIndices.size()
              << " fractional_admissible_evidence_sha256="
              << (fractionalPrepared?fractionalSupport.admissibleGroupEvidenceSHA256:std::string{})
              << " fractional_carrier_packet_sha256="
              << (fractionalPrepared?carrierData.sourceCarrierPacketSHA256:std::string{})
              << " exact_projection_rows="
              << (fractionalPrepared?proofFiles.exactProjectionRows.size():0u)
              << " rest_basis_projection_rejects="
              << (fractionalPrepared?proofFiles.referenceBasisRejects.size():0u)
              << " source_centroid_slot_rows="
              << (fractionalPrepared?proofFiles.sourceCentroidRows.size():0u)
              << " carrier_sidecar_rows="
              << (fractionalPrepared?proofFiles.carrierWitnessRows.size():0u)
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
    if(fractionalPrepared){
        std::cout<<"fractional_outputs=PASS"
            <<" flow_path="<<artifactPins.flow.string()<<" flow_sha256="<<artifactPins.sha256[0]
            <<" semantic_groups_path="<<artifactPins.semanticGroups.string()<<" semantic_groups_sha256="<<artifactPins.sha256[1]
            <<" source_donors_path="<<artifactPins.sourceDonors.string()<<" source_donors_sha256="<<artifactPins.sha256[2]
            <<" carrier_trace_path="<<artifactPins.carrierTrace.string()<<" carrier_trace_sha256="<<artifactPins.sha256[3]<<'\n';
    }
    for (const auto& [material, range] : densityRange) {
        std::cout << "transfer_density_range material=" << material
                  << " min_kgm3=" << range.first
                  << " max_kgm3=" << range.second << '\n';
    }
    return 0;
}
