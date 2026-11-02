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
 // Explicit current row domains retain their true source IDs separately from
 // the 433 compatibility slots. The historical D37 reader stays unchanged.
 id.sourceMeshSHA256=std::string(64,'b');
 RegisteredSkinCurrentSourceDomain current{"current-typed-fixture",std::string(64,'e'),80000,120000};
 const auto extendedHead=head.substr(0,head.size()-1)+",source_vertex_row,source_face_row,source_bary0,source_bary1,source_bary2,source_domain,source_residual_x_m,source_residual_y_m,source_residual_z_m\n";
 const std::string currentRows=
 "0,7,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,-1,1,0,0,0,70000,-1,0,0,0,current-typed-fixture,0,0,0\n"
 "1,7,1,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,1,1,-1,1,0,0,0,70001,-1,0,0,0,current-typed-fixture,0,0,0\n"
 "2,7,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,2,-1,1,0,0,0,70002,-1,0,0,0,current-typed-fixture,0,0,0\n"
 "3,7,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,3,-1,1,0,0,0,-1,110000,0,0,1,current-typed-fixture,0,0,0\n";
 const auto typedRead=[&]{return loadRegisteredSkinCandidateCSV(nodes,faces,8,id,std::string(64,'c'),fakeHasher,out,report,error,&current);};
 std::ofstream(nodes)<<extendedHead<<currentRows;std::ofstream(faces)<<fs;
 if(!typedRead()||report.currentRegisteredRows!=4||report.directPredecessorRows||report.derivedPredecessorRows||
 out.sourceVertexIndices!=std::vector<std::uint32_t>{70000,70001,70002,NM_INVALID_INDEX}||
 out.sourceEmbeddingFaceIndices.back()!=110000){std::cerr<<error;return 10;}
 const auto typedBad=[&](std::string data){std::ofstream(nodes)<<extendedHead<<data;
  if(typedRead()||out.worldPoints!=prior){std::cerr<<"invalid typed source admitted: "<<error;return false;}++rejected;return true;};
 auto replaceOnce=[](std::string data,std::string_view a,std::string_view b){data.replace(data.find(a),a.size(),b);return data;};
 if(!typedBad(replaceOnce(currentRows,"70000,-1","80000,-1")))return 11;
 if(!typedBad(replaceOnce(currentRows,"-1,110000","-1,120000")))return 12;
 if(!typedBad(replaceOnce(currentRows,"current-typed-fixture","wrong-domain")))return 13;
 if(!typedBad(replaceOnce(currentRows,"0,-1,1,0,0,0,70000","5,-1,1,0,0,0,70000")))return 14;
 if(!typedBad(replaceOnce(currentRows,"-1,110000,0,0,1","-1,110000,0,0,2")))return 15;
 std::ofstream(nodes)<<head<<rows;if(typedRead())return 16;++rejected;
 current.vertexCount=0;std::ofstream(nodes)<<extendedHead<<currentRows;if(typedRead())return 17;++rejected;
 std::filesystem::remove_all(root);
 std::cout<<"registered_skin_candidate_csv=PASS nodes="<<report.nodeRows<<" faces="<<report.faceRows
          <<" negative_cases="<<rejected<<" physical_qualification=not_claimed\n";return 0;
}
