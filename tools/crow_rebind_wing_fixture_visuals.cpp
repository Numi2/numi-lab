#include "metalrobo/VisualPresentation.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <string>

// Exact v15 visual binding adapter for the isolated six-link wing fixture.
// The inputs remain untouched and all original bindings must be present.
int main(int argc, char** argv) {
    if(argc!=3) return 2;
    const std::array<const char*,5> names{"gate-left","gate-right","slalom-a","slalom-b","perch"};
    for(std::uint32_t i=0;i<names.size();++i) {
        const std::string input=std::string(argv[1])+"/"+names[i]+".hip-roll.mrvpack";
        const std::string output=std::string(argv[2])+"/"+names[i]+".wing-fixture.mrvpack";
        metalrobo::VisualAssetPackV2 p;std::string why;
        if(!metalrobo::readVisualAssetPack(input,p,&why)){std::cerr<<why<<"\n";return 1;}
        if(p.instances.empty()||p.symbolicBindings.empty()){std::cerr<<"missing bindings\n";return 1;}
        for(auto& x:p.instances){if(x.binding.y!=36u+i)return 1;x.binding.y=42u+i;}
        for(auto& x:p.symbolicBindings){if(x.bodyIndex!=36u+i)return 1;x.bodyIndex=42u+i;}
        p.id+=".wing-mechanics-fixture";
        p.preprocessingProvenance+="; wing fixture: explicit six-body scene index rebind";
        p.contentHash=metalrobo::computeVisualAssetPackContentHash(p);
        if(!metalrobo::writeVisualAssetPack(p,output,&why)){std::cerr<<why<<"\n";return 1;}
        metalrobo::VisualAssetPackV2 check;
        if(!metalrobo::readVisualAssetPack(output,check,&why)){std::cerr<<why<<"\n";return 1;}
        for(const auto& x:check.instances)if(x.binding.y!=42u+i)return 1;
        for(const auto& x:check.symbolicBindings)if(x.bodyIndex!=42u+i)return 1;
        std::cout<<names[i]<<" "<<36u+i<<" -> "<<42u+i<<" "<<check.contentHash<<"\n";
    }
}
