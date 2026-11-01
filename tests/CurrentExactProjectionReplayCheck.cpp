#include "NumiHumanCurrentFractionalSupportNPZ.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
namespace fs=std::filesystem;

static std::vector<std::string> header(){
    return {"source_material","source_donor","source_region","source_projection_parent_tet_row",
        "source_projection_bary0","source_projection_bary1","source_projection_bary2","source_projection_bary3",
        "projected_source_x_m","projected_source_y_m","projected_source_z_m","exact_projection_distance_m",
        "receiver_child_tet_row","receiver_current_material","receiver_current_donor","receiver_current_region",
        "receiver_centroid_x_m","receiver_centroid_y_m","receiver_centroid_z_m","937_tuple_material",
        "937_tuple_donor","937_tuple_region","937_selected_parent_tet_row","937_carrier_x_m","937_carrier_y_m",
        "937_carrier_z_m","937_residual_x_m","937_residual_y_m","937_residual_z_m","937_bary0","937_bary1",
        "937_bary2","937_bary3","937_registration_distance_m","candidate_P1_node_anchor_x_m",
        "candidate_P1_node_anchor_y_m","candidate_P1_node_anchor_z_m","P1_anchor_offset_from_child_centroid_m",
        "child_volume_m3","projection_status"};
}
static std::string csvRow(double projX=0.25,double currentDistance=0.0,std::uint32_t sourceParent=1u,
                          std::uint32_t sourceDonor=146u,double bary0=0.25){
    std::vector<std::string> f(40,"0");
    f[0]="15";f[1]=std::to_string(sourceDonor);f[2]="10";f[3]=std::to_string(sourceParent);
    f[4]=std::to_string(bary0);f[5]="0.25";f[6]="0.25";f[7]=std::to_string(0.5-bary0);
    f[8]=std::to_string(projX);f[9]="0.25";f[10]="0.25";f[11]=std::to_string(currentDistance);
    f[12]="0";f[13]="15";f[14]="145";f[15]="10";
    f[16]="0.25";f[17]="0.25";f[18]="0.25";
    f[39]="PASS_EXACT_CURRENT_CENTROID_SOURCE_TUPLE_PROJECTION";
    std::string out;
    for(std::size_t i=0;i<f.size();++i){if(i)out+=',';out+=f[i];}
    return out;
}
static fs::path writeCSV(std::string row){
    const auto path=fs::temp_directory_path()/"h706-current-exact-projection-fixture.csv";
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    const auto h=header();
    for(std::size_t i=0;i<h.size();++i){if(i)out<<',';out<<h[i];}
    out<<'\n'<<row<<'\n';out.close();
    return path;
}
struct Fixture{
    std::array<SpatialMassTransportPoint,2> parentCentroids{{{0.25,0.25,0.25},{0.25,0.25,0.25}}};
    std::array<double,2> parentVolumes{1.0/6.0,1.0/6.0},parentMass{0.5,0.5};
    std::array<std::uint32_t,2> parentMaterial{15,15},parentDonor{145,146},parentRegion{10,10};
    std::array<SpatialMassTransportPoint,1> childCentroids{{{0.25,0.25,0.25}}};
    std::array<double,1> childVolumes{1.0e-3};
    std::array<std::uint32_t,1> childMaterial{15},childDonor{145},childRegion{10};
    ConservativeSpatialMassTransportInput input{};
    PhysicalSkinParentMassPacket parent{};
    ConservativeSpatialChildSemanticSupportData support{};
    Fixture(){
        input.parentCentroidsWorld=parentCentroids;input.parentReferenceVolumeM3=parentVolumes;input.parentMassKg=parentMass;
        input.parentMaterialIndices=parentMaterial;input.parentDonorBodyIndices=parentDonor;input.parentAnatomicalRegionIndices=parentRegion;
        input.childCentroidsWorld=childCentroids;input.childReferenceVolumeM3=childVolumes;
        input.childMaterialIndices=childMaterial;input.childDonorBodyIndices=childDonor;input.childAnatomicalRegionIndices=childRegion;
        input.maximumTransportDistanceM=0.019353;
        parent.currentWorldPoints={{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}};
        parent.referenceWorldPoints=parent.currentWorldPoints;parent.tetrahedra={{{0,1,2,3}},{{0,1,2,3}}};
        parent.referenceTetVolumeM3.assign(parentVolumes.begin(),parentVolumes.end());parent.tetrahedronMassKg.assign(parentMass.begin(),parentMass.end());
        parent.materialIndices={15,15};parent.donorBodyIndices={145,146};parent.anatomicalRegionIndices={10,10};
        parent.parentMassPacketSHA256=std::string(64,'f');
        support.childOffsets={0,1};support.slotGroupIndices={0};support.volumeFractions={1.0};
        support.groupMaterialIndices={15,15};support.donorBodyIndices={145,146};support.anatomicalRegionIndices={10,10};
        support.derivedGroupIdentityIDs={1,2};support.sourceParentTetOffsets={0,1,2};support.sourceParentTetRows={0,1};
        support.sourceStableFaceOffsets={0,1,2};support.sourceStableFaceRows={100,101};
        support.parentStableSourceFaceIdentities={100,101};
        support.cellAdmissibleRegionOffsets={0,1};support.cellAdmissibleRegionIds={10};
        support.admissibilitySourceSupportSHA256=std::string(64,'b');
        support.admissibilityMaximumSourceRegionHops=1;support.typedSupportSHA256=std::string(64,'c');
    }
};

int main(){
    std::string error,sha;
    std::vector<CurrentExactTupleProjectionRow> rows;
    std::vector<CurrentExactTupleProjectionReject> rejects;
    auto path=writeCSV(csvRow());
    if(!physical_skin_child_mass_detail::sha256File(path,sha,error)){std::cerr<<error<<"\n";return 1;}
    Fixture valid;
    if(!loadCurrentExactTupleProjectionCSV(path,sha,valid.input,valid.parent,0.019353,rows,error,&rejects)||
       rows.size()!=1||!rejects.empty()||!rows[0].referenceBasisAllowed||rows[0].sourceParentRow!=1||
       std::abs(rows[0].referenceTransportDistanceM)>1e-12){
        std::cerr<<"valid current/rest exact projection did not replay: "<<error<<"\n";return 2;
    }
    if(!bindCurrentExactTupleProjectionRows(valid.support,valid.input,valid.parent,rows,sha,error)){
        std::cerr<<"valid exact tuple projection did not bind: "<<error<<"\n";return 3;
    }
    if(valid.support.childAdmissibleGroupOffsets!=std::vector<std::uint32_t>{0,2}||
       valid.support.childAdmissibleGroupIndices!=std::vector<std::uint32_t>{0,1}||
       valid.support.slotGroupIndices!=std::vector<std::uint32_t>{0}||
       valid.support.volumeFractions!=std::vector<double>{1.0}){
        std::cerr<<"projection slot changed primary P1 preference data\n";return 4;
    }

    Fixture movedReference;
    for(auto& point:movedReference.parent.referenceWorldPoints)point[0]+=0.1;
    movedReference.input.parentCentroidsWorld=movedReference.parentCentroids;
    path=writeCSV(csvRow());
    if(!physical_skin_child_mass_detail::sha256File(path,sha,error)||
       !loadCurrentExactTupleProjectionCSV(path,sha,movedReference.input,movedReference.parent,
                                           0.019353,rows,error,&rejects)||
       rows.size()!=1||rows[0].referenceBasisAllowed||rejects.size()!=1||
       rejects[0].reason!="reference_rest_source_point_exceeds_transport_radius"){
        std::cerr<<"current-coordinate PASS was not rejected/recorded when rest-coordinate transport exceeds radius: "<<error<<"\n";return 5;
    }
    if(!bindCurrentExactTupleProjectionRows(movedReference.support,movedReference.input,movedReference.parent,
                                            rows,sha,error)||
       movedReference.support.childAdmissibleGroupIndices!=std::vector<std::uint32_t>{0}){
        std::cerr<<"reference-coordinate failed witness added a mass support slot: "<<error<<"\n";return 6;
    }

    Fixture badPoint;
    path=writeCSV(csvRow(0.5));
    if(!physical_skin_child_mass_detail::sha256File(path,sha,error)||
       loadCurrentExactTupleProjectionCSV(path,sha,badPoint.input,badPoint.parent,0.019353,rows,error)){
        std::cerr<<"projection row with wrong source point coordinates was accepted\n";return 7;
    }
    Fixture badTuple;
    path=writeCSV(csvRow(0.25,0.0,1u,999u));
    if(!physical_skin_child_mass_detail::sha256File(path,sha,error)||
       loadCurrentExactTupleProjectionCSV(path,sha,badTuple.input,badTuple.parent,0.019353,rows,error)){
        std::cerr<<"projection row with wrong source donor identity was accepted\n";return 8;
    }
    Fixture badBary;
    path=writeCSV(csvRow(0.25,0.0,1u,146u,-0.1));
    if(!physical_skin_child_mass_detail::sha256File(path,sha,error)||
       loadCurrentExactTupleProjectionCSV(path,sha,badBary.input,badBary.parent,0.019353,rows,error)){
        std::cerr<<"projection row with a negative barycentric coordinate was accepted\n";return 9;
    }
    Fixture wrongHop;
    path=writeCSV(csvRow());
    if(!physical_skin_child_mass_detail::sha256File(path,sha,error)||
       !loadCurrentExactTupleProjectionCSV(path,sha,wrongHop.input,wrongHop.parent,0.019353,rows,error)){
        std::cerr<<"valid CSV failed before the one-hop negative control: "<<error<<"\n";return 10;
    }
    wrongHop.support.cellAdmissibleRegionIds={11};
    if(bindCurrentExactTupleProjectionRows(wrongHop.support,wrongHop.input,wrongHop.parent,rows,sha,error)){
        std::cerr<<"projection row outside the authenticated one-hop region was admitted\n";return 11;
    }
    fs::remove(path);
    std::cout<<"PASS current exact projection replay: current/rest geometry, barycentric, tuple, material, one-hop and unchanged-radius gates\n";
    return 0;
}
