#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/VisualPlatform.hpp"

#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require(
    const metalrobo::WorldCompileResult& result,
    const char* stage
) {
    if (!result.succeeded()) {
        throw std::runtime_error(
            std::string{stage} + ": " + result.message
        );
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr
                << "usage: metalrobo_room_scan_scene PACK.mrvpack "
                   "OUTPUT.json\n";
            return 2;
        }

        const std::filesystem::path packPath{argv[1]};
        const std::filesystem::path manifestPath{argv[2]};

        metalrobo::VisualAssetPackV2 pack;
        std::string reason;
        require(
            metalrobo::readVisualAssetPack(
                packPath,
                pack,
                &reason
            ),
            "read visual pack: " + reason
        );

        metalrobo::WorldTemplate world;
        require(
            metalrobo::compileEpisodeTwin(
                metalrobo::makeFrankaPickPlaceEpisodeTwin(),
                metalrobo::makeFrankaPickPlaceEngineModel(),
                world
            ),
            "compile reference world"
        );

        const std::uint32_t workspaceAsset =
            world.assetIndex("workspace");
        require(
            workspaceAsset != MR_INVALID_INDEX,
            "reference world has no workspace asset"
        );

        const metalrobo::VisualAssetReferenceV3 reference{
            packPath,
            pack.contentHash,
            workspaceAsset,
            1u,
            1u,
        };
        metalrobo::VisualSceneManifestV3 manifest;
        require(
            metalrobo::compileVisualSceneManifestV3(
                world,
                std::array{reference},
                metalrobo::makeNeutralStudioEnvironmentV2(),
                metalrobo::makeIndoorAreaLightRigV1(),
                manifest,
                &reason
            ),
            "compile visual scene: " + reason
        );

        manifest.id = "numi_property_roomplan_scan.visual.v3";
        manifest.renderScene.id =
            "numi_property_roomplan_scan.visual.v3.runtime";
        manifest.preprocessingProvenance =
            "numi_roomplan_import/v1/native_visual_scene_compile";
        manifest.renderScene.fingerprint =
            metalrobo::computeVisualRenderSceneV3Fingerprint(
                manifest.renderScene
            );
        manifest.fingerprint =
            metalrobo::computeVisualSceneManifestV3Fingerprint(
                manifest
            );
        require(
            metalrobo::writeVisualSceneManifestV3(
                manifest,
                manifestPath,
                &reason
            ),
            "write visual scene: " + reason
        );

        std::cout
            << "scene_id=" << manifest.id << '\n'
            << "manifest=" << manifestPath << '\n'
            << "manifest_fingerprint=" << manifest.fingerprint << '\n'
            << "world_fingerprint=" << manifest.worldFingerprint << '\n'
            << "workspace_asset_index=" << workspaceAsset << '\n'
            << "source_hash=" << pack.sourceContentHash << '\n'
            << "visual_pack_hash=" << pack.contentHash << '\n'
            << "vertices=" << pack.vertices.size() << '\n'
            << "indices=" << pack.indices.size() << '\n'
            << "materials=" << pack.materials.size() << '\n'
            << "textures=" << pack.textures.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_room_scan_scene: "
                  << error.what() << '\n';
        return 1;
    }
}
