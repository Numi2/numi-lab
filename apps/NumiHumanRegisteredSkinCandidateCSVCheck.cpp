#include "NumiHumanRegisteredSkinCandidateCSV.hpp"
#include <iostream>
#include <chrono>
using namespace metalrobo::human::coarse_tissue;
int main(){
 const auto root=std::filesystem::temp_directory_path()/
    ("numi-registered-csv-check-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 std::filesystem::create_directory(root);const auto nodes=root/"nodes.csv",faces=root/"faces.csv";
 const std::string head="vertex,donor,wx,wy,wz,nx,ny,nz,lx,ly,lz,m00,m01,m02,m10,m11,m12,m20,m21,m22,d37_source_vertex_row,d37_source_face_row,beta0,beta1,beta2,projection_distance_m\n";
 const std::string rows="0,7,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,-1,1,0,0,0\n"
 "1,7,1,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,1,1,-1,1,0,0,0\n"
 "2,7,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,2,-1,1,0,0,0\n"
 "3,7,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,-1,0,0,0,1,0\n";
 const std::string fs="face,v0,v1,v2,baseline806_source_face_row\n0,0,1,2,0\n1,0,2,3,-1\n";
 std::ofstream(nodes)<<head<<rows;std::ofstream(faces)<<fs;
 RegisteredSurface id;id.name="numerical fixture";id.provider="fixture";id.sourceMember="fixture.nhskin";
 id.sourcePayloadSHA256=std::string(64,'a');id.sourceMeshSHA256=std::string(64,'b');id.aggregateIdentity=true;
 const auto fakeHasher=[&](const std::filesystem::path& p){return std::string(64,p==nodes?'b':'c');};
 RegisteredSurface out;RegisteredSkinCandidateCSVReport report;std::string error;
 const auto read=[&]{return loadRegisteredSkinCandidateCSV(nodes,faces,8,id,std::string(64,'c'),fakeHasher,out,report,error);};
 if(!read()||report.nodeRows!=4||report.faceRows!=2||report.directPredecessorRows!=3||report.derivedPredecessorRows!=1){std::cerr<<error;return 2;}
 const auto prior=out.worldPoints;unsigned rejected=0;
 auto bad=[&](const std::string& n,const std::string& f){
    std::ofstream(nodes)<<n;std::ofstream(faces)<<f;
    if(read()||out.worldPoints!=prior){std::cerr<<"invalid CSV accepted or output mutated: "<<error;return false;}
    ++rejected;return true;};
 std::string changed=rows;changed.replace(changed.find("0,7,"),4,"0,8,");if(!bad(head+changed,fs))return 3;
 if(!bad(head+rows,"face,v0,v1,v2,baseline806_source_face_row\n0,0,1,9,0\n"))return 4;
 if(!bad(head+rows,"face,v0,v1,v2,baseline806_source_face_row\n0,0,0,1,0\n"))return 5;
 if(!bad(head+rows+rows,fs))return 6;
 if(!bad(head+rows,"face,v0,v1,v2,baseline806_source_face_row\n"))return 7;
 if(!bad(head+"0,7,NaN\n",fs))return 8;
 std::ofstream(nodes)<<head<<rows;std::ofstream(faces)<<fs;
 id.sourceMeshSHA256=std::string(64,'d');if(read()||out.worldPoints!=prior)return 9;++rejected;
 std::filesystem::remove_all(root);
 std::cout<<"registered_skin_candidate_csv=PASS nodes="<<report.nodeRows<<" faces="<<report.faceRows
          <<" negative_cases="<<rejected<<" physical_qualification=not_claimed\n";return 0;
}
