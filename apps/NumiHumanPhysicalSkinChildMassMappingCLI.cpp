#include "NumiHumanPhysicalSkinChildMassMapping.hpp"

#include <iomanip>
#include <iostream>
#include <string>

int main(int argc,char** argv) {
    using namespace metalrobo::human::coarse_tissue;
    if(argc==2&&std::string_view(argv[1])=="--help"){
        std::cout<<"usage: numi-human-physical-skin-child-mass-mapping-cli "
                    "--parent-mass-csv <compiled-parent-cell-mass.csv>\n"
                    "Parses the exact native four-file parent mass packet only. "
                    "Child transfer is assembled by preparePhysicalSkinChildMassMapping "
                    "after the owning app supplies real child mesh, O/I/D, region, donor, "
                    "material, ancestry, and registered-frame inputs.\n";
        return 0;
    }
    if(argc!=3||std::string_view(argv[1])!="--parent-mass-csv"){
        std::cerr<<"invalid arguments; run --help\n";return 2;
    }
    PhysicalSkinParentMassPacket packet;
    std::string error;
    if(!loadCompiledSkinFatParentMassPacketCSV(argv[2],packet,error)){
        std::cerr<<"parent_packet=REJECTED reason="<<error<<'\n';return 3;
    }
    std::cout<<std::setprecision(17)<<"parent_packet=PASS nodes="<<packet.referenceWorldPoints.size()
             <<" tetrahedra="<<packet.tetrahedra.size()
             <<" source_rows="<<packet.sourceMassContributions.size()
             <<" mass_kg="<<packet.totalMassKg
             <<" nodal_closure_abs_kg="<<packet.maximumNodalMassClosureErrorKg
             <<" nodal_closure_rel="<<packet.maximumNodalMassClosureRelativeError
             <<" mass_csv_sha256="<<packet.parentMassPacketSHA256
             <<" source_csv_sha256="<<packet.parentSourceContributionSHA256
             <<" packet_topology_sha256="<<packet.parentTopologySHA256<<'\n';
    std::cout<<"child_transfer=NOT_RUN reason=real-child-mesh-and-explicit-region/donor/frame-ancestry-inputs-not-provided\n";
    return 0;
}

