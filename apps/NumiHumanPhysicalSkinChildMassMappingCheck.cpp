#include "NumiHumanPhysicalSkinChildMassMapping.hpp"

#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

namespace {

std::filesystem::path makePacket(const std::filesystem::path& directory) {
    const auto mass=directory/"compiled-parent-cell-mass.csv";
    const auto nodes=std::filesystem::path(mass.string()+".nodes.csv");
    const auto sources=std::filesystem::path(mass.string()+".source.csv");
    const auto tets=std::filesystem::path(mass.string()+".tets.csv");
    const std::uint32_t bits=std::bit_cast<std::uint32_t>(25.0f);
    {
        std::ofstream out(mass);
        out<<"source_tet_row,material_index,mass_kg\n0,0,100\n";
    }
    {
        std::ofstream out(nodes);
        out<<"parent_node_index,compiled_global_node,current_x_m,current_y_m,current_z_m,"
              "rest_x_m,rest_y_m,rest_z_m,compiled_lumped_mass_kg,node_frame_owner_body,"
              "donor_local_x_m,donor_local_y_m,donor_local_z_m,mass_bits\n";
        const std::array<V3,4> p{{{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}}};
        for(std::uint32_t node=0;node<4u;++node){
            const auto& x=p[node];
            out<<node<<','<<(10u+node)<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','
               <<x[0]<<','<<x[1]<<','<<x[2]<<",25,7,"
               <<x[0]<<','<<x[1]<<','<<x[2]<<','<<bits<<'\n';
        }
    }
    {
        std::ofstream out(sources);
        out<<"object_local_node,material_index,donor_body,mass_kg,local_x_m,local_y_m,local_z_m\n";
        const std::array<V3,4> p{{{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}}};
        for(std::uint32_t node=0;node<4u;++node)
            out<<node<<",0,7,25,"<<p[node][0]<<','<<p[node][1]<<','<<p[node][2]<<'\n';
    }
    {
        std::ofstream out(tets);
        out<<"parent_tet_row,material_index,donor_body,parent_node0,parent_node1,"
              "parent_node2,parent_node3,parent_reference_volume_m3,parent_mass_kg\n"
           <<"0,0,7,0,1,2,3,0.16666666666666666,100\n";
    }
    return mass;
}

void fillOneChild(PhysicalSkinChildMassMappingBundle& bundle) {
    const std::array<V3,4> p{{{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}}};
    bundle.currentWorldPoints.assign(p.begin(),p.end());
    bundle.referenceWorldPoints.assign(p.begin(),p.end());
    bundle.childTetrahedra.push_back({0u,1u,2u,3u});
    bundle.childMaterialIndices.push_back(0u);
    bundle.childDonorBodyIndices.push_back(7u);
    bundle.childAnatomicalRegionIndices.push_back(3u);
    bundle.childStableSourceFaceIdentities.push_back(901u);
    bundle.childMeshSHA256=std::string(64u,'a');
    bundle.regionLabelSHA256=std::string(64u,'b');
    bundle.coordinateFrameSHA256=std::string(64u,'c');
    bundle.maximumSpatialMassTransportDistanceM=1.0;
}

} // namespace

int main() {
    const auto directory=std::filesystem::temp_directory_path()/
        ("numi-physical-child-mass-check-"+
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    const auto massCSV=makePacket(directory);

    PhysicalSkinParentMassPacket packet;
    std::string error;
    if(!loadCompiledSkinFatParentMassPacketCSV(massCSV,packet,error)){
        std::cerr<<"parent CSV fixture rejected: "<<error<<'\n';return 2;
    }
    if(packet.tetrahedra.size()!=1u||packet.referenceWorldPoints.size()!=4u||
       packet.sourceMassContributions.size()!=4u||packet.totalMassKg!=100.0||
       packet.maximumNodalMassClosureErrorKg!=0.0||
       packet.parentMassPacketSHA256.size()!=64u||
       packet.parentSourceContributionSHA256.size()!=64u||
       packet.parentTopologySHA256.size()!=64u){
        std::cerr<<"parent CSV fixture did not preserve the complete native packet\n";return 3;
    }
    const std::array<std::uint32_t,1> region{3u},stableFace{901u};
    if(!attachExplicitParentMassMappingLabels(packet,region,stableFace,error)){
        std::cerr<<"explicit parent labels rejected: "<<error<<'\n';return 4;
    }

    const auto labels=std::filesystem::path(massCSV.string()+".labels.csv");
    const std::string labelHeader="parent_tet_row,material_index,geometric_donor_body,"
        "anatomical_region_id,stable_source_face_id\n";
    std::ofstream(labels)<<labelHeader<<"0,0,7,3,901\n";
    if(!loadCompiledSkinFatParentLabelsCSV(labels,packet,error)||
       packet.anatomicalRegionIndices!=std::vector<std::uint32_t>{3u}||
       packet.stableSourceFaceIdentities!=std::vector<std::uint32_t>{901u}||
       packet.parentLabelSHA256.size()!=64u) {
        std::cerr<<"native parent label roundtrip rejected: "<<error<<'\n';return 10;
    }
    const auto validLabelHash=packet.parentLabelSHA256;
    std::ofstream(labels)<<labelHeader<<"0,0,8,4,902\n";
    if(loadCompiledSkinFatParentLabelsCSV(labels,packet,error)||
       packet.parentLabelSHA256!=validLabelHash||
       packet.anatomicalRegionIndices!=std::vector<std::uint32_t>{3u}||
       packet.stableSourceFaceIdentities!=std::vector<std::uint32_t>{901u}) {
        std::cerr<<"foreign donor labels accepted or mutated source identity\n";return 11;
    }
    std::ofstream(labels)<<labelHeader;
    if(loadCompiledSkinFatParentLabelsCSV(labels,packet,error)||
       packet.parentLabelSHA256!=validLabelHash) {
        std::cerr<<"truncated labels accepted or mutated parent packet\n";return 12;
    }

    PhysicalSkinChildMassMappingBundle bundle;
    bundle.parent=std::move(packet);
    fillOneChild(bundle);
    const std::array<double,1> densityLower{599.0},densityUpper{601.0};
    if(!preparePhysicalSkinChildMassMapping(bundle,densityLower,densityUpper,1.0,8u,error)){
        std::cerr<<"positive mass mapping fixture rejected: "<<error<<'\n';return 5;
    }
    if(bundle.spatialMassTransportResult.childMassKg.size()!=1u||
       std::abs(bundle.spatialMassTransportResult.childMassKg[0]-100.0)>1.0e-10||
       bundle.spatialMassTransportResult.childDominantParentRows!=std::vector<std::uint32_t>{0u}||
       bundle.spatialMassTransportResult.maximumTransportDistanceUsedM!=0.0||
       bundle.childDominantParentTetrahedra!=std::vector<std::uint32_t>{0u}){
        std::cerr<<"positive mapping fixture did not close the single eligible child\n";return 6;
    }

    auto wrongRegion=bundle;
    wrongRegion.childAnatomicalRegionIndices[0]=4u;
    const auto priorReceipt=bundle.spatialMassTransportResult.receiptSHA256;
    if(preparePhysicalSkinChildMassMapping(wrongRegion,densityLower,densityUpper,1.0,8u,error)||
       error.find("not feasible")==std::string::npos||
       bundle.spatialMassTransportResult.receiptSHA256!=priorReceipt){
        std::cerr<<"region barrier negative case was not rejected transactionally: "<<error<<'\n';return 7;
    }

    auto negativeTet=bundle;
    negativeTet.childTetrahedra[0]={0u,2u,1u,3u};
    if(preparePhysicalSkinChildMassMapping(negativeTet,densityLower,densityUpper,1.0,8u,error)||
       error.find("nonpositive")==std::string::npos){
        std::cerr<<"negative child orientation was not rejected: "<<error<<'\n';return 8;
    }

    PhysicalSkinParentMassPacket unchanged=bundle.parent;
    std::ofstream(massCSV,std::ios::trunc)
        <<"source_tet_row,material_index,mass_kg\n0,0,-1\n";
    if(loadCompiledSkinFatParentMassPacketCSV(massCSV,unchanged,error)||
       unchanged.totalMassKg!=bundle.parent.totalMassKg){
        std::cerr<<"invalid parent packet was accepted or mutated output\n";return 9;
    }

    std::filesystem::remove_all(directory);
    std::cout<<"physical_child_mass_mapping_fixture=PASS "
             <<"parent_csv_roundtrip=PASS strict_region_barrier=PASS "
             <<"positive_tet_gate=PASS transfer=fixture_only\n";
    return 0;
}

