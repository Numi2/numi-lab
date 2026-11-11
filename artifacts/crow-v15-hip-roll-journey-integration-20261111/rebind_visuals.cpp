#include "metalrobo/VisualPresentation.hpp"
#include <array>
#include <iostream>
#include <string>
int main() {
    const std::array<const char*, 5> names{"gate-left","gate-right","slalom-a","slalom-b","perch"};
    for (std::uint32_t i=0; i<names.size(); ++i) {
        const std::string input=std::string("artifacts/crow-v15-two-segment-frozen-flight-visual-bindings-20261111/")+names[i]+".two-segment.mrvpack";
        const std::string output=std::string("artifacts/crow-v15-hip-roll-journey-integration-20261111/")+names[i]+".hip-roll.mrvpack";
        metalrobo::VisualAssetPackV2 p; std::string why;
        if (!metalrobo::readVisualAssetPack(input,p,&why)) { std::cerr<<input<<": "<<why<<"\n"; return 1; }
        if (p.instances.empty() || p.symbolicBindings.empty()) { std::cerr<<"missing required bindings\n"; return 1; }
        for (auto& v:p.instances) { if(v.binding.y!=34u+i) {std::cerr<<"unexpected input instance\n";return 1;} v.binding.y=36u+i; }
        for (auto& v:p.symbolicBindings) { if(v.bodyIndex!=34u+i) {std::cerr<<"unexpected input symbolic binding\n";return 1;} v.bodyIndex=36u+i; }
        p.id += ".hip-roll"; p.preprocessingProvenance += "; crow experimental hip roll explicit visual rebind";
        p.contentHash=metalrobo::computeVisualAssetPackContentHash(p);
        if(!metalrobo::writeVisualAssetPack(p,output,&why)) {std::cerr<<output<<": "<<why<<"\n";return 1;}
        metalrobo::VisualAssetPackV2 check;
        if(!metalrobo::readVisualAssetPack(output,check,&why)) {std::cerr<<why<<"\n";return 1;}
        for(const auto& v:check.instances) if(v.binding.y!=36u+i) return 1;
        for(const auto& v:check.symbolicBindings) if(v.bodyIndex!=36u+i) return 1;
        std::cout<<names[i]<<" old="<<34u+i<<" new="<<36u+i<<" hash="<<p.contentHash<<"\n";
    }
}
