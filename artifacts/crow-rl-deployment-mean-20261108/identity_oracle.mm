#import <Foundation/Foundation.h>
#include "CrowCameraFixture.hpp"
#include "metalrobo/VisualPresentation.hpp"
#include <iostream>
#include <charconv>
using namespace crowCameraFixture;
int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc==5,"Usage: identity_oracle ORIGINAL_ROOT RESTORED_ROOT NEW_ASSETS SEED_BASE");
    std::uint64_t seedBase=0;const std::string raw=argv[4];const auto parsed=std::from_chars(raw.data(),raw.data()+raw.size(),seedBase);
    require(parsed.ec==std::errc{} && parsed.ptr==raw.data()+raw.size() && seedBase<=std::numeric_limits<std::uint64_t>::max()-2,"Invalid seed base");
    const std::filesystem::path original=argv[1],restored=argv[2],assets=argv[3];
    require(std::filesystem::create_directory(assets),"Oracle assets exist");
    auto m=makeManifest(false);applyAmericanCrowForagingFixtureLimits(m);
    m.profile.environmentCount=64;m.profile.controlSteps=320;m.reality.reset={};
    CompiledRun base;check(compileRun(m,base),"Compile physical fixture");
    const auto visual=makeVisual(base,assets,true);
    std::cout<<"{";bool first=true;
    for(unsigned offset=0;offset<3;++offset)for(unsigned arm=0;arm<4;++arm) {
        const auto seed=seedBase+offset;
        const auto name=std::to_string(seed)+"-arm"+std::to_string(arm);
        if(!first)std::cout<<',';first=false;std::cout<<'"'<<name<<"\":{";
        for(unsigned location=0;location<2;++location) {
            auto v=visual.manifest;
            for(auto& pack:v.renderScene.visualPacks)pack.packPath=(location ? restored : original)/name/pack.packPath.filename();
            v.renderScene.fingerprint=computeVisualRenderSceneV3Fingerprint(v.renderScene);
            v.fingerprint=computeVisualSceneManifestV3Fingerprint(v);
            if(location)std::cout<<',';
            std::cout<<'"'<<(location ? "restored" : "original")<<"\":{\"manifest\":"<<v.fingerprint<<",\"render_scene\":"<<v.renderScene.fingerprint<<'}';
        }
        std::cout<<'}';
    }
    std::cout<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
