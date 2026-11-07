// Authoring-only reconstruction: no physics, actions, policy, or resource stepping.
#define main retained_cue_instrument_main
#include "crow_cue_foraging_check.mm"
#undef main
int main(int argc,char** argv) {@autoreleasepool {try {
    require(argc==4,"Usage: scene_relocation_check NEW_OUTPUT ORIGINAL_ROOT REPLAY_ROOT");
    const std::filesystem::path out=argv[1];
    require(std::filesystem::create_directory(out),"Reconstruction output exists");
    const std::array<std::string,8> arms{"visible-chunk1","visible-chunk16","paired-seed","occluded","post-cue-reset","left-port","right-port","left-partition"};
    for(bool blocked:{false,true}) {
        auto authored=out/(blocked ? "blocked-packs" : "clear-packs");
        std::filesystem::create_directory(authored);
        auto m=manifest(blocked);CompiledRun base;check(compileRun(m,base),"compile unchanged authoring fixture");
        auto visual=makeVisual(base,authored,[](const WorldAsset& asset,VisualAssetPackV2& pack){
            if(asset.id=="camera_target")for(unsigned side=0;side<2;++side)beacon(pack,asset.bodyIndices.front(),side);
        });
        for(const auto& arm:arms)if((arm=="occluded")==blocked)for(unsigned variant=0;variant<2;++variant) {
            auto reconstructed=visual.manifest;
            for(auto& ref:reconstructed.renderScene.visualPacks)
                ref.packPath=std::filesystem::path(argv[variant+2])/arm/ref.packPath.filename();
            reconstructed.renderScene.fingerprint=computeVisualRenderSceneV3Fingerprint(reconstructed.renderScene);
            reconstructed.fingerprint=computeVisualSceneManifestV3Fingerprint(reconstructed);
            std::string reason;
            require(writeVisualSceneManifestV3(reconstructed,out/(variant ? "replay" : "original")/arm/"visual-scene.json",&reason),reason);
        }
    }
    std::cout<<"{\"status\":\"passed\",\"reconstructed_manifests\":16,\"native_simulation_steps\":0}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
