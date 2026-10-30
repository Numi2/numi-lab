#include "NumiHumanPhysicalSkinChildTetGen.hpp"
#include "NumiHumanPhysicalSkinChildRegisteredFacetMaps.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
namespace cap=physical_skin_child_capacity_detail;
static_assert(cap::maxCandidateEdges==(cap::fullReference?12000000u:3000000u));
static_assert(cap::maxAugmentations==(cap::fullReference?4000000u:300000u));
int main(){
 auto dir=std::filesystem::temp_directory_path()/("numi-child-cap-"+std::to_string(cap::maxCells));std::filesystem::create_directory(dir);
 auto n=dir/"mesh.node",e=dir/"mesh.ele",l=dir/"labels.csv";
 auto write=[](auto p,auto s){std::ofstream f(p);f<<s;};
 const std::string nodes="4 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n";
 const std::string labels="child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n0,15,0,1,1\n";
 auto attempt=[&](std::string& error){PhysicalSkinChildMassMappingBundle out;PhysicalSkinChildTetGenImportReport report;return loadPhysicalSkinChildTetGenReference(n,e,l,std::string(64,'a'),1,.019353,out,report,error);};
 write(n,nodes);write(e,"1 4 1\n1 1 2 3 4 15\n");write(l,labels);std::string error;
 assert(attempt(error));assert(error.empty());
 write(n,std::to_string(cap::maxNodes+1)+" 3 0 0\n");
 assert(!attempt(error));assert(error.find("node header")!=std::string::npos);
 write(n,nodes);write(e,std::to_string(cap::maxCells+1)+" 4 1\n");
 assert(!attempt(error));assert(error.find("element header")!=std::string::npos);
 write(n,"300001 3 0 0\n");write(e,"1 4 1\n");
 assert(!attempt(error));assert(error.find(cap::fullReference?"node row":"node header")!=std::string::npos);
 write(n,nodes);write(e,"200001 4 1\n");
 assert(!attempt(error));assert(error.find(cap::fullReference?"element row":"element header")!=std::string::npos);
 // Larger capacity never waives physical or identity checks.
 write(e,"1 4 1\n1 1 3 2 4 15\n");
 assert(!attempt(error));assert(error.find("nonpositive")!=std::string::npos);
 write(e,"1 4 1\n1 1 2 3 99 15\n");
 assert(!attempt(error));assert(error.find("missing node")!=std::string::npos);
 write(e,"1 4 1\n1 1 2 3 4 15\n");write(l,"child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n0,16,0,1,1\n");
 assert(!attempt(error));assert(error.find("violate")!=std::string::npos);
 std::filesystem::remove_all(dir);
 std::cout<<"capacity_profile="<<(cap::fullReference?"full-reference-v1":"legacy")<<" max_nodes="<<cap::maxNodes<<" max_cells="<<cap::maxCells<<" max_candidate_edges="<<cap::maxCandidateEdges<<" max_augmentations="<<cap::maxAugmentations<<" checks=10 passed\n";
}