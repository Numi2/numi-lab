#import <Foundation/Foundation.h>
#include "metalrobo/CrowManipulation.hpp"
#include "metalrobo/VisualPresentation.hpp"
#include <iostream>
#include <stdexcept>
using namespace metalrobo;
int main(int argc,char**argv){@autoreleasepool{try{
    if(argc!=4)throw std::runtime_error("ORIGINAL_ROOT RESTORED_ROOT NEW_ASSETS");
    const std::filesystem::path original=argv[1],restored=argv[2],assets=argv[3];
    if(!std::filesystem::create_directory(assets))throw std::runtime_error("Assets exist");
    std::cout<<'{';
    for(unsigned disabled=0;disabled<2;++disabled){
        const auto name=disabled?"no-crow-contact-fixture":"normal-fixture";
        const auto dir=assets/name;std::filesystem::create_directory(dir);
        auto manifest=makeAmericanCrowObjectPushFixtureManifest(!disabled);
        CompiledRun run;const auto result=compileRun(manifest,run);
        if(!result.succeeded())throw std::runtime_error(result.message);
        const auto visual=authorAmericanCrowForagingVisual(run,dir,false,crowObjectPushCameraOrientation);
        if(disabled)std::cout<<',';std::cout<<'"'<<name<<"\":{";
        for(unsigned location=0;location<2;++location){
            auto value=visual.manifest;
            for(auto& pack:value.renderScene.visualPacks)
                pack.packPath=(location?restored:original)/name/pack.packPath.filename();
            value.renderScene.fingerprint=computeVisualRenderSceneV3Fingerprint(value.renderScene);
            value.fingerprint=computeVisualSceneManifestV3Fingerprint(value);
            if(location)std::cout<<',';
            std::cout<<'"'<<(location?"restored":"original")<<"\":{\"manifest\":"<<value.fingerprint<<",\"render_scene\":"<<value.renderScene.fingerprint<<'}';
        }
        std::cout<<'}';
    }
    std::cout<<"}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}}
