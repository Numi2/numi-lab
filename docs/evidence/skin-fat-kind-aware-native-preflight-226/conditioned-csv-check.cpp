
#include "NumiHumanRegisteredSkinCandidateCSV.hpp"
#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include <iostream>
using namespace metalrobo::human::coarse_tissue;
int main(int argc,char**argv){
 if(argc!=3)return 2;
 const std::filesystem::path d(argv[1]),out(argv[2]);std::string error;
 auto sha=[&](const std::filesystem::path& p){std::string h;if(!physical_skin_child_mass_detail::sha256File(p,h,error))return std::string{};return h;};
 RegisteredSurface id;id.provider="native source preparation test";id.sourceMember=d.string();id.aggregateIdentity=true;
 id.sourcePayloadSHA256="7a6c44b641b286497fbef73157e2945e2b7c4d5d2c423eff3b26590aa496abf4";
 id.sourceMeshSHA256="9e1fcdcb2d8812718a2cabd37908b4064553d89f8e3bab54a151afb75045b3cd";
 RegisteredSurface loaded;RegisteredSkinCandidateCSVReport r;
 if(!loadRegisteredSkinCandidateCSV(d/"registered-O.csv",d/"registered-O-faces.csv",157,id,
    "77138f1ebf882611a57955f310cb974141334818dda3c059656ef003d25a8ad8",sha,loaded,r,error)||r.nodeRows!=32002||r.faceRows!=63959){std::cerr<<error<<"\n";return 1;}
 std::ifstream f(d/"registered-O-faces.csv");std::string head,line,rest;std::getline(f,head);std::getline(f,line);rest.assign(std::istreambuf_iterator<char>(f),{});
 auto fields=registered_skin_candidate_csv_detail::split(line);fields[4]="-2";std::string changed;
 for(const auto& v:fields){if(!changed.empty())changed+=",";changed+=v;}
 auto bad=out/"invalid-predecessor.csv";{std::ofstream b(bad);b<<head<<"\n"<<changed<<"\n"<<rest;}
 RegisteredSurface kept=loaded;RegisteredSkinCandidateCSVReport keptReport;keptReport.nodeRows=123;
 if(loadRegisteredSkinCandidateCSV(d/"registered-O.csv",bad,157,id,sha(bad),sha,kept,keptReport,error)||
    error.find("invalid predecessor identity")==std::string::npos||kept.faces!=loaded.faces||kept.worldPoints!=loaded.worldPoints||keptReport.nodeRows!=123){std::cerr<<"conditioned predecessor rejection not atomic: "<<error<<"\n";return 1;}
 auto badHeader=out/"invalid-header.csv";{std::ofstream b(badHeader);b<<head<<",ignored_extra\n"<<line<<",0\n";}
 if(loadRegisteredSkinCandidateCSV(d/"registered-O.csv",badHeader,157,id,sha(badHeader),sha,kept,keptReport,error)||error.find("wrong face schema")==std::string::npos){std::cerr<<"unknown face contract accepted: "<<error<<"\n";return 1;}
 std::cout<<"conditioned_registered_skin_csv=PASS nodes="<<r.nodeRows<<" faces="<<r.faceRows<<" bad_predecessor=reject_atomic unknown_schema=reject physical_admission=false\n";
}
