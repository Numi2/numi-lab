#include "metalrobo/VisualPresentation.hpp"
#include <array>
#include <iostream>
#include <string>
int main() {
    const std::array<const char*, 5> names{"gate-left","gate-right","slalom-a","slalom-b","perch"};
    for (std::uint32_t i=0; i<names.size(); ++i) {
        const std::string base="assets/crow_navigation_course/";
        const std::string input=base+names[i]+".mrvpack";
        const std::string output=base+names[i]+".active-head.mrvpack";
        metalrobo::VisualAssetPackV2 p; std::string why;
        if (!metalrobo::readVisualAssetPack(input,p,&why)) { std::cerr<<input<<": "<<why<<"\n"; return 1; }
        if (p.instances.empty()) { std::cerr<<"empty instances\n"; return 1; }
        for (auto& instance:p.instances) {
            if (instance.binding.y != 15u+i) { std::cerr<<"unexpected legacy instance binding\n"; return 1; }
            instance.binding.y=18u+i;
        }
        for (auto& binding:p.symbolicBindings) {
            if (binding.bodyIndex != 15u+i) { std::cerr<<"unexpected legacy symbolic binding\n"; return 1; }
            binding.bodyIndex=18u+i;
        }
        p.id += ".active-head";
        p.preprocessingProvenance += "; crow_rebind_navigation_visuals";
        p.contentHash=metalrobo::computeVisualAssetPackContentHash(p);
        if (!metalrobo::writeVisualAssetPack(p,output,&why)) { std::cerr<<output<<": "<<why<<"\n"; return 1; }
        std::cout<<input<<" -> "<<output<<" body "<<15u+i<<" -> "<<18u+i<<" hash "<<p.contentHash<<"\n";
    }
}
