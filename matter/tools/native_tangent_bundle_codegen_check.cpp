#include "cardboard_expression_cache_fixture.hpp"
#include "../src/native_tangent_source.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(int argc,char** argv){
 try{
  need(argc==2,"output required");std::filesystem::create_directories(argv[1]);
  const auto input=numi_cardboard_expression_cache::makeWorld(
   numi_cardboard_expression_cache::makeMaterial(true,false));
  for(unsigned variant=0;variant<3;++variant){
   auto compiled=numi::matter::compileWorld(input);need(compiled.succeeded(),"compile fixture");
   auto& w=compiled.world;auto& m=w.materials.at(0);
   std::uint32_t pi=m.tangentProgramOffset+9;
   while((w.scalarPrograms.at(pi).flags&NM_SCALAR_PROGRAM_FLAG_TANGENT_BUNDLE_PRELUDE)!=0u)++pi;
   if(variant==1){
    auto& program=w.scalarPrograms.at(pi);
    std::vector<NMExpressionInstructionGPU> copy(w.instructions.begin()+program.firstInstruction,
     w.instructions.begin()+program.firstInstruction+program.instructionCount);
    auto store=copy.at(0);store.opcode=NM_EXPR_CACHE_STORE;store.index=0;
    copy.insert(copy.begin()+1,store);
    program.firstInstruction=w.instructions.size();program.instructionCount=copy.size();
    w.instructions.insert(w.instructions.end(),copy.begin(),copy.end());
   }
   if(variant==2){m.flags|=NM_MATERIAL_HAS_STATE;m.stateCount=1;}
   std::string error;const auto path=std::filesystem::path(argv[1])/("admission-"+std::to_string(variant)+".metalinc");
   need(nmWriteNativeTangentSource(w,path.c_str(),error),error.c_str());
   std::ifstream f(path);std::string text{std::istreambuf_iterator<char>(f),{}};
   need((text.find("// Exact read-only output bundle:")!=std::string::npos)==(variant==0),
    "read-only/state eligibility mismatch");
  }
  std::cout<<"PASS shared-bundle admission: eligible pure, cache-write fallback, stateful fallback\n";
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
