#include "NumiHumanPhysicalSkinChildRegisteredVolumeCSV.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

using namespace metalrobo::human::coarse_tissue;
using Fields=std::vector<std::string>;

namespace {
Fields row(std::string id,std::string partition,std::string sourceID,std::string role,
           std::string surfaceRow) {
    Fields fields(42u,"0");
    fields[0]=id;fields[1]=partition;fields[2]=sourceID;
    fields[3]="4294967295";fields[4]="";fields[5]="";
    fields[6]=role=="interior"?"":role;fields[7]=surfaceRow;fields[8]=role;
    fields[9]="0.25";fields[10]="0.125";fields[11]="0.0625";
    fields[12]="0";fields[13]="0";
    fields[14]=fields[9];fields[15]=fields[10];fields[16]=fields[11];
    fields[17]="fixture-source";fields[18]="vertex";fields[19]="12";fields[20]="4294967295";
    fields[27]="registered_source";fields[28]="source_parent_tet_reference_registration";
    fields[29]="0";fields[30]="0.25";fields[31]="0.25";fields[32]="0.25";fields[33]="0.25";
    fields[38]="0";fields[39]=role=="I"?"98304":(partition=="skin"?"32768":"65536");
    fields[40]="0";fields[41]="42";
    if(role=="I") {
        fields[3]=surfaceRow;fields[4]=partition=="skin"?"fat":"skin";
        fields[5]=partition=="skin"?"222":"111";
    }
    if(role=="interior") {
        fields[13]="4294967295";fields[17]="derived_volume_interior_no_surface_ancestry";
        fields[18]="none";fields[19]="4294967295";fields[20]="4294967295";
        fields[27]="derived_full_all_bodies_geometric_donor_pose";
    }
    return fields;
}
std::vector<Fields> fixture() {
    // Nontrivial joint-node ordering and reciprocal source IDs.
    return {row("3","fat","333","D","0"),row("2","skin","111","I","10"),
            row("1","skin","999","O","0"),row("4","fat","444","interior","4294967295"),
            row("2","fat","222","I","10")};
}
std::string serialize(const std::vector<Fields>& rows,bool validHeader=true) {
    std::ostringstream output;
    auto header=physicalSkinCombinedNodeMapCSVHeader();
    if(!validHeader)header[0]="untyped_node";
    const auto line=[&](const Fields& fields) {
        for(std::size_t col=0;col<fields.size();++col)output<<(col?",":"")<<fields[col];
        output<<"\r\n";
    };
    line(header);for(const auto& fields:rows)line(fields);return output.str();
}
void reject(const std::vector<Fields>& rows,std::string_view expected,bool validHeader=true) {
    std::istringstream input(serialize(rows,validHeader));
    std::vector<PhysicalSkinRegisteredVolumeNodeRow> output(1);
    output[0].volumeNodeRow=98765u;
    PhysicalSkinCombinedNodeMapCSVReport report;report.volumeNodes=76543u;
    std::string error;
    assert(!readPhysicalSkinCombinedNodeMapCSV(input,4u,157u,output,report,error));
    assert(output.size()==1u&&output[0].volumeNodeRow==98765u&&report.volumeNodes==76543u);
    if(error.find(expected)==std::string::npos) {
        std::cerr<<"expected rejection "<<expected<<", got "<<error<<'\n';std::abort();
    }
}
}

int main(int argc,char** argv) {
    if(argc==4) {
        std::uint32_t nodes=0,bodies=0;
        if(!physical_skin_child_mass_detail::parseNumber(std::string(argv[2]),nodes)||
           !physical_skin_child_mass_detail::parseNumber(std::string(argv[3]),bodies))return 2;
        std::ifstream input(argv[1]);if(!input)return 2;
        std::vector<PhysicalSkinRegisteredVolumeNodeRow> parsed;
        PhysicalSkinCombinedNodeMapCSVReport report;std::string error;
        if(!readPhysicalSkinCombinedNodeMapCSV(input,nodes,bodies,parsed,report,error)) {
            std::cerr<<error<<'\n';return 1;
        }
        std::cout<<"producer_node_map_cpp_reader=PASS nodes="<<report.volumeNodes
            <<" source_rows="<<report.fileRows<<" shared_I_pairs="<<report.sharedInterfacePairs
            <<" physical_volume_admission=NOT_CLAIMED\n";return 0;
    }
    if(argc!=1)return 2;
    auto rows=fixture();std::istringstream input(serialize(rows));
    std::vector<PhysicalSkinRegisteredVolumeNodeRow> parsed;
    PhysicalSkinCombinedNodeMapCSVReport report;std::string error;
    assert(readPhysicalSkinCombinedNodeMapCSV(input,4u,157u,parsed,report,error));
    assert(parsed.size()==4u&&report.fileRows==5u&&report.sharedInterfacePairs==1u);
    for(std::size_t node=0;node<parsed.size();++node)assert(parsed[node].volumeNodeRow==node);
    assert(parsed[0].role==PhysicalSkinRegisteredVolumeNodeRole::outerO);
    assert(parsed[1].role==PhysicalSkinRegisteredVolumeNodeRole::innerI&&parsed[1].surfaceVertexRow==10u);
    assert(parsed[2].role==PhysicalSkinRegisteredVolumeNodeRole::deepD);
    assert(parsed[3].role==PhysicalSkinRegisteredVolumeNodeRole::interior);
    assert(parsed[3].sourceVertex==NM_INVALID_INDEX&&parsed[3].donorBinding==NM_INVALID_INDEX);
    assert(parsed[0].parentTieCount==0u);
    auto signedAbsent=rows;
    for(auto& fields:signedAbsent)for(unsigned col:{3u,7u,13u,19u,20u})
        if(fields[col]=="4294967295")fields[col]="-1";
    std::istringstream signedInput(serialize(signedAbsent));
    std::vector<PhysicalSkinRegisteredVolumeNodeRow> signedParsed;
    assert(readPhysicalSkinCombinedNodeMapCSV(signedInput,4u,157u,signedParsed,report,error));
    assert(signedParsed[3].sourceVertex==NM_INVALID_INDEX);
    auto bad=rows;bad.pop_back();reject(bad,"reciprocal");
    bad=rows;bad[4][9]="0.5";reject(bad,"duplicate joint");
    bad=rows;bad.erase(bad.begin()+3);reject(bad,"omits");
    bad=rows;bad[2][8]="D";reject(bad,"surface role");
    bad=rows;bad[2][9]="nan";reject(bad,"invalid finite");
    bad=rows;bad[2].pop_back();reject(bad,"invalid width");
    reject(rows,"header",false);
    bad=rows;bad[2][18]="untyped";reject(bad,"mapping kind");
    bad=rows;bad[4][5]="112";reject(bad,"duplicate joint");
    bad=rows;bad[2][2]="111";reject(bad,"source TetGen");
    bad=rows;bad[2][9]="0.1";reject(bad,"exact Float32");
    bad=rows;bad[3][17]="";reject(bad,"typed source domain");
    bad=rows;bad[2][0]="0";reject(bad,"joint/source identity");
    bad=rows;bad[2][13]="-2";reject(bad,"invalid finite");
    bad=rows;bad[4][3]="11";reject(bad,"exact reciprocal");
    bad=rows;bad.push_back(rows[4]);reject(bad,"source TetGen");
    std::cout<<"physical_skin_registered_volume_csv=PASS shuffled_joint_rows=1 reciprocal_I_merge=1"
        <<" corruptions_rejected=16 failure_transaction_preserved=1 physical_volume_admission=NOT_CLAIMED\n";
}
